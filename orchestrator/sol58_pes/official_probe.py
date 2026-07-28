"""Budgeted official probes for architecture-sensitive local regressions."""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from orchestrator.loongflow_compat.architecture_islands import (
    architecture_label,
    architecture_tags,
    extract_architecture_features,
)
from orchestrator.sol58_pes.evaluator_state import (
    atomic_write_json,
    file_lock,
    read_json,
)


_PROXY_SENSITIVE_FEATURES = (
    "warp_specialization",
    "cluster_launch",
    "tma",
    "cluster_tma_broadcast",
    "wgmma",
    "cp_async",
    "barrier_pipeline",
    "cooperative_grid_sync",
    "persistent_kernel",
    "radix_sort",
    "cub_radix_sort",
    "block_radix_sort",
    "bitwise_radix_sort",
    "expert_parallel_scan",
)


@dataclass(frozen=True)
class OfficialProbePolicy:
    enabled: bool
    interval: int
    cooldown_seconds: float
    max_per_day: int
    max_relative_regression: float
    architecture_only: bool


def architecture_probe_evidence(
    candidate_source: str,
    incumbent_source: str | None,
) -> dict[str, Any]:
    candidate = extract_architecture_features(candidate_source)
    incumbent = (
        extract_architecture_features(incumbent_source) if incumbent_source else {}
    )
    active = [
        name for name in _PROXY_SENSITIVE_FEATURES if int(candidate.get(name, 0)) > 0
    ]
    changed = [
        name
        for name in _PROXY_SENSITIVE_FEATURES
        if int(candidate.get(name, 0)) != int(incumbent.get(name, 0))
    ]
    return {
        "candidate_label": architecture_label(candidate),
        "candidate_tags": architecture_tags(candidate),
        "incumbent_label": architecture_label(incumbent) if incumbent else None,
        "active_proxy_sensitive_features": active,
        "changed_proxy_sensitive_features": changed,
        "architecture_sensitive": bool(set(active) & set(changed)),
    }


def _iso(timestamp: float) -> str:
    return datetime.fromtimestamp(timestamp, tz=timezone.utc).isoformat()


def claim_official_probe(
    registry_path: Path,
    *,
    source_sha256: str,
    relative_regression: float,
    evidence: dict[str, Any],
    policy: OfficialProbePolicy,
    now: float | None = None,
) -> dict[str, Any]:
    timestamp = time.time() if now is None else float(now)
    result: dict[str, Any] = {
        "claimed": False,
        "relative_regression": relative_regression,
        "architecture": evidence,
        "policy": asdict(policy),
    }
    if not policy.enabled:
        return {**result, "reason": "official probes disabled"}
    if relative_regression < 0 or relative_regression > policy.max_relative_regression:
        return {**result, "reason": "local regression exceeds probe ceiling"}
    if policy.architecture_only and not evidence.get("architecture_sensitive"):
        return {**result, "reason": "candidate has no changed proxy-sensitive feature"}
    if policy.max_per_day <= 0:
        return {**result, "reason": "daily official probe budget is zero"}

    with file_lock(registry_path):
        registry = read_json(registry_path, {})
        if not isinstance(registry, dict):
            registry = {}
        sources = registry.get("sources")
        if not isinstance(sources, dict):
            sources = {}
        if source_sha256 in sources:
            return {
                **result,
                "reason": "source already considered for an official probe",
            }

        eligible_count = int(registry.get("eligible_count") or 0) + 1
        last_claim_count = int(registry.get("last_claim_eligible_count") or 0)
        due = eligible_count - last_claim_count >= max(1, policy.interval)
        source_record: dict[str, Any] = {
            "eligible_at": timestamp,
            "eligible_at_iso": _iso(timestamp),
            "eligible_count": eligible_count,
            "relative_regression": relative_regression,
            "architecture": evidence,
            "claimed": False,
        }
        sources[source_sha256] = source_record

        claims = [
            claim
            for claim in registry.get("claims", [])
            if isinstance(claim, dict)
            and float(claim.get("claimed_at") or 0.0) >= timestamp - 86400
        ]
        last_claimed_at = float(registry.get("last_claimed_at") or 0.0)
        if not due:
            reason = (
                f"probe interval not reached ({eligible_count - last_claim_count}/"
                f"{max(1, policy.interval)})"
            )
        elif len(claims) >= policy.max_per_day:
            reason = "daily official probe budget exhausted"
        elif timestamp - last_claimed_at < policy.cooldown_seconds:
            remaining = policy.cooldown_seconds - (timestamp - last_claimed_at)
            reason = f"official probe cooldown has {remaining:.1f}s remaining"
        else:
            reason = "periodic architecture-sensitive official probe"
            claim = {
                "source_sha256": source_sha256,
                "claimed_at": timestamp,
                "claimed_at_iso": _iso(timestamp),
                "eligible_count": eligible_count,
                "relative_regression": relative_regression,
                "architecture": evidence,
            }
            claims.append(claim)
            source_record["claimed"] = True
            source_record["reason"] = reason
            registry["last_claimed_at"] = timestamp
            registry["last_claim_eligible_count"] = eligible_count
            result["claimed"] = True

        if len(sources) > 5000:
            sources = dict(list(sources.items())[-5000:])
        registry.update(
            {
                "schema_version": 1,
                "eligible_count": eligible_count,
                "sources": sources,
                "claims": claims,
                "updated_at": timestamp,
                "updated_at_iso": _iso(timestamp),
            }
        )
        atomic_write_json(registry_path, registry)

    return {
        **result,
        "reason": reason,
        "eligible_count": eligible_count,
        "probe_due": due,
        "daily_claims": len(claims),
    }
