"""Noise-aware local-incumbent gate for SOL58 measurements."""

from __future__ import annotations

import math
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from orchestrator.sol58_pes.evaluator_state import (
    atomic_write_json,
    file_lock,
    read_json,
)


@dataclass(frozen=True)
class LocalGatePolicy:
    enabled: bool
    sigma_multiplier: float
    relative_noise_floor: float
    recheck_pairs: int
    uncertain_relative_tolerance: float
    allow_uncertain_official: bool
    challenger_cooldown_seconds: float


def uncertainty_band(
    candidate_latencies_ms: list[float],
    incumbent_latencies_ms: list[float],
    *,
    policy: LocalGatePolicy,
) -> float:
    candidate_logs = [math.log(value) for value in candidate_latencies_ms if value > 0]
    incumbent_logs = [math.log(value) for value in incumbent_latencies_ms if value > 0]
    standard_error = 0.0
    if len(candidate_logs) >= 2:
        standard_error += statistics.variance(candidate_logs) / len(candidate_logs)
    if len(incumbent_logs) >= 2:
        standard_error += statistics.variance(incumbent_logs) / len(incumbent_logs)
    return max(
        policy.relative_noise_floor,
        policy.sigma_multiplier * math.sqrt(max(0.0, standard_error)),
    )


def classify_delta(relative_delta: float, uncertainty: float) -> str:
    if relative_delta + uncertainty < 0:
        return "confirmed_faster"
    if relative_delta - uncertainty > 0:
        return "confirmed_slower"
    return "uncertain"


def claim_uncertain_challenger(
    registry_path: Path,
    source_sha256: str,
    *,
    policy: LocalGatePolicy,
    format_timestamp: Callable[[float], str],
) -> tuple[bool, str]:
    if not policy.allow_uncertain_official:
        return False, "uncertain official submissions disabled"
    now = time.time()
    with file_lock(registry_path):
        registry = read_json(registry_path, {})
        if not isinstance(registry, dict):
            registry = {}
        sources = registry.get("sources")
        if not isinstance(sources, dict):
            sources = {}
        if source_sha256 in sources:
            return False, "source already claimed as an uncertain challenger"
        last_claimed_at = float(registry.get("last_claimed_at") or 0.0)
        remaining = policy.challenger_cooldown_seconds - (now - last_claimed_at)
        if remaining > 0:
            return (
                False,
                f"uncertain challenger cooldown has {remaining:.1f}s remaining",
            )
        sources[source_sha256] = {
            "claimed_at": now,
            "claimed_at_iso": format_timestamp(now),
        }
        registry.update(
            {
                "schema_version": 1,
                "last_claimed_at": now,
                "sources": sources,
            }
        )
        atomic_write_json(registry_path, registry)
    return True, "uncertain challenger slot claimed"


def evaluate_local_best_gate(
    *,
    workspace: Path,
    kernel_source: str,
    source_language: str,
    source_sha256: str,
    candidate_latencies_ms: list[float],
    local_best: dict[str, Any] | None,
    measurement_profile: dict[str, Any],
    program_path: str,
    start: float,
    policy: LocalGatePolicy,
    read_local_best_source: Callable[[dict[str, Any]], tuple[str, str] | None],
    kernel_source_hash: Callable[[str], str],
    collect_local_evaluation: Callable[..., dict[str, Any]],
    claim_challenger: Callable[[str], tuple[bool, str]],
) -> dict[str, Any]:
    """Use paired remeasurement to separate a real improvement from clock drift."""
    candidate_median = float(statistics.median(candidate_latencies_ms))
    if not policy.enabled:
        return {
            "status": "disabled",
            "update_local_best": True,
            "submit_official": True,
            "candidate_latency_ms": candidate_median,
            "candidate_repeats_ms": candidate_latencies_ms,
        }
    if not local_best:
        return {
            "status": "no_incumbent",
            "update_local_best": True,
            "submit_official": True,
            "candidate_latency_ms": candidate_median,
            "candidate_repeats_ms": candidate_latencies_ms,
        }

    incumbent_latency = float(local_best.get("latency_ms_median") or 0.0)
    if incumbent_latency <= 0:
        return {
            "status": "invalid_incumbent",
            "update_local_best": True,
            "submit_official": True,
            "candidate_latency_ms": candidate_median,
            "candidate_repeats_ms": candidate_latencies_ms,
        }
    if str(local_best.get("kernel_sha256") or "") == source_sha256:
        return {
            "status": "same_source",
            "update_local_best": False,
            "submit_official": False,
            "candidate_latency_ms": candidate_median,
            "candidate_repeats_ms": candidate_latencies_ms,
        }

    incumbent_repeats = [
        float(value)
        for value in (local_best.get("repeat_latencies_ms") or [incumbent_latency])
        if isinstance(value, (int, float)) and float(value) > 0
    ]
    initial_delta = candidate_median / incumbent_latency - 1.0
    initial_uncertainty = uncertainty_band(
        candidate_latencies_ms,
        incumbent_repeats,
        policy=policy,
    )
    initial_classification = classify_delta(initial_delta, initial_uncertainty)
    result: dict[str, Any] = {
        "status": initial_classification,
        "initial_relative_delta": initial_delta,
        "initial_uncertainty": initial_uncertainty,
        "rechecked": False,
        "update_local_best": False,
        "submit_official": False,
        "candidate_latency_ms": candidate_median,
        "candidate_repeats_ms": candidate_latencies_ms,
    }
    if initial_classification == "confirmed_slower":
        return result

    incumbent_source = read_local_best_source(local_best)
    if policy.recheck_pairs <= 0 or incumbent_source is None:
        result["status"] = f"{initial_classification}_without_paired_recheck"
        result["update_local_best"] = initial_classification == "confirmed_faster"
        if result["update_local_best"]:
            result["submit_official"] = True
        elif initial_delta <= policy.uncertain_relative_tolerance:
            claimed, reason = claim_challenger(source_sha256)
            result.update(
                {
                    "submit_official": claimed,
                    "challenger_claimed": claimed,
                    "challenger_reason": reason,
                }
            )
        if incumbent_source is None:
            result["recheck_error"] = "incumbent source unavailable"
        return result

    incumbent_kernel, incumbent_language = incumbent_source
    candidate_pairs: list[float] = []
    incumbent_pairs: list[float] = []
    pair_records: list[dict[str, Any]] = []
    recheck_root = workspace / "local_gate_recheck"
    for pair_index in range(policy.recheck_pairs):
        order = (
            ("candidate", "incumbent")
            if pair_index % 2 == 0
            else ("incumbent", "candidate")
        )
        measured: dict[str, float] = {}
        for label in order:
            source = kernel_source if label == "candidate" else incumbent_kernel
            language = source_language if label == "candidate" else incumbent_language
            source_hash = (
                source_sha256
                if label == "candidate"
                else kernel_source_hash(incumbent_kernel)
            )
            pair_workspace = recheck_root / f"pair_{pair_index + 1}_{label}"
            pair_workspace.mkdir(parents=True, exist_ok=True)
            measurement = collect_local_evaluation(
                workspace=pair_workspace,
                kernel_source=source,
                source_language=language,
                source_sha256=source_hash,
                measurement_profile=measurement_profile,
                program_path=program_path,
                start=start,
                repeat_count=1,
                cache_enabled=False,
            )
            if measurement.get("error_result"):
                error_result = measurement["error_result"]
                result.update(
                    {
                        "status": "recheck_failed",
                        "rechecked": True,
                        "recheck_error": str(error_result.get("summary") or "")[:500],
                        "pair_records": pair_records,
                    }
                )
                return result
            measured[label] = float(measurement["local_latencies_ms"][0])
        candidate_pairs.append(measured["candidate"])
        incumbent_pairs.append(measured["incumbent"])
        pair_records.append(
            {
                "pair": pair_index + 1,
                "order": list(order),
                "candidate_latency_ms": measured["candidate"],
                "incumbent_latency_ms": measured["incumbent"],
                "ratio": measured["candidate"] / measured["incumbent"],
            }
        )

    log_ratios = [
        math.log(candidate / incumbent)
        for candidate, incumbent in zip(candidate_pairs, incumbent_pairs, strict=True)
    ]
    paired_log_delta = float(statistics.median(log_ratios))
    paired_delta = math.exp(paired_log_delta) - 1.0
    paired_standard_error = (
        statistics.stdev(log_ratios) / math.sqrt(len(log_ratios))
        if len(log_ratios) >= 2
        else 0.0
    )
    paired_uncertainty = max(
        policy.relative_noise_floor,
        policy.sigma_multiplier * paired_standard_error,
    )
    classification = classify_delta(paired_delta, paired_uncertainty)
    normalized_repeats = [
        incumbent_latency * candidate / incumbent
        for candidate, incumbent in zip(candidate_pairs, incumbent_pairs, strict=True)
    ]
    normalized_latency = float(statistics.median(normalized_repeats))
    update_local_best = classification == "confirmed_faster"
    submit_official = update_local_best
    challenger_claimed = False
    challenger_reason = ""
    if (
        classification == "uncertain"
        and paired_delta <= policy.uncertain_relative_tolerance
    ):
        challenger_claimed, challenger_reason = claim_challenger(source_sha256)
        submit_official = challenger_claimed

    result.update(
        {
            "status": classification,
            "rechecked": True,
            "relative_delta": paired_delta,
            "uncertainty": paired_uncertainty,
            "paired_standard_error": paired_standard_error,
            "pair_records": pair_records,
            "candidate_pair_latencies_ms": candidate_pairs,
            "incumbent_pair_latencies_ms": incumbent_pairs,
            "normalized_candidate_repeats_ms": normalized_repeats,
            "candidate_latency_ms": normalized_latency,
            "candidate_repeats_ms": normalized_repeats,
            "update_local_best": update_local_best,
            "submit_official": submit_official,
            "challenger_claimed": challenger_claimed,
            "challenger_reason": challenger_reason,
        }
    )
    return result
