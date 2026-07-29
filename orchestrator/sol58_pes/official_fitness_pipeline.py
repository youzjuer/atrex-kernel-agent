"""Task-independent orchestration for official fitness submission and scoring."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable


Result = dict[str, Any]


@dataclass(frozen=True)
class OfficialFitnessConfig:
    target_latency_ms: float
    minimum_local_score: float
    local_best_gate: bool
    pending_score_policy: str
    provisional_projection_floor: float
    evaluation_stack_version: str
    gpu_type: str
    submission_mode: str
    refreshable_statuses: frozenset[str]


@dataclass
class OfficialEvaluationState:
    workspace: Path
    kernel_source: str
    source_language: str
    local_score: float
    gate_latency_ms: float
    best_latency_before: float
    local_best_before: dict[str, Any] | None
    same_as_local_best: bool
    passes_official_gate: bool
    reuse_cached_official: bool
    gate_result: dict[str, Any]
    official_probe: dict[str, Any]
    new_local_best_record: dict[str, Any] | None
    measurement_profile: dict[str, Any]
    metrics: dict[str, Any]
    artifacts: dict[str, Any]
    local_summary: str
    started_at: float


@dataclass(frozen=True)
class OfficialFitnessHooks:
    result: Callable[..., Result]
    fitness_anchor: Callable[[dict[str, Any] | None], dict[str, Any] | None]
    record_authoritative_source: Callable[..., None]
    anchored_provisional_score: Callable[..., tuple[float, dict[str, Any]]]
    submit_official: Callable[..., dict[str, Any]]
    official_status: Callable[[dict[str, Any]], str]
    save_local_best: Callable[[dict[str, Any], str], bool]
    record_calibration: Callable[..., None]
    local_provisional_score: Callable[[float], float]
    provisional_official_score: Callable[[float], tuple[float, float]]


def _reuse_completed_local_best(
    state: OfficialEvaluationState,
    hooks: OfficialFitnessHooks,
) -> Result | None:
    anchor = hooks.fitness_anchor(state.local_best_before)
    if not (
        state.same_as_local_best
        and anchor is not None
        and anchor["source"] == "completed_local_best"
    ):
        return None

    score = float(anchor["score"])
    local_best = state.local_best_before or {}
    hooks.record_authoritative_source(
        state.kernel_source,
        source_language=state.source_language,
        official_score=score,
        official_latency_ms=float(local_best.get("official_latency_ms") or 0.0),
        local_latency_ms=state.gate_latency_ms,
        submission_id=anchor.get("submission_id"),
    )
    state.metrics["official"] = {
        "enabled": True,
        "submitted": False,
        "status": "REUSED_LOCAL_BEST_OFFICIAL",
        "authoritative": True,
        "fitness_source": "official_local_best_record",
        "submission_id": anchor.get("submission_id"),
        "sol_score": score,
        "record_hit": True,
        "cache_hit": False,
    }
    summary = (
        f"{state.local_summary} Reused completed official fitness from the persisted "
        f"local-best record: sol_score={score:.6f}, "
        f"submission_id={anchor.get('submission_id')}."
    )
    return hooks.result(
        "success",
        summary,
        score,
        metrics=state.metrics,
        artifacts=state.artifacts,
    )


def _skip_below_local_prefilter(
    state: OfficialEvaluationState,
    config: OfficialFitnessConfig,
    hooks: OfficialFitnessHooks,
) -> Result | None:
    if state.local_score >= config.minimum_local_score:
        return None
    state.metrics["official"] = {
        "enabled": True,
        "status": "SKIPPED_LOCAL_PREFILTER",
        "authoritative": False,
        "fitness_source": "none",
    }
    summary = (
        f"{state.local_summary} Official fitness skipped because local_score "
        f"{state.local_score:.6f} < minimum_local_score "
        f"{config.minimum_local_score:.6f}; PES score forced to 0."
    )
    return hooks.result(
        "success",
        summary,
        0.0,
        metrics=state.metrics,
        artifacts=state.artifacts,
    )


def _skip_outside_official_gate(
    state: OfficialEvaluationState,
    config: OfficialFitnessConfig,
    hooks: OfficialFitnessHooks,
) -> Result | None:
    if not (
        config.local_best_gate
        and not state.passes_official_gate
        and not state.reuse_cached_official
    ):
        return None

    scoring_latency_ms = (
        state.best_latency_before
        if state.same_as_local_best and state.best_latency_before > 0
        else state.gate_latency_ms
    )
    scoring_local_score = config.target_latency_ms / scoring_latency_ms
    provisional_score, details = hooks.anchored_provisional_score(
        scoring_local_score,
        scoring_latency_ms,
        state.local_best_before,
    )
    skip_status = (
        "SKIPPED_SAME_AS_LOCAL_BEST"
        if state.same_as_local_best
        else "SKIPPED_NOT_LOCAL_BEST"
    )
    anchored = details["source"] == "incumbent_official_anchor"
    state.metrics["official"] = {
        "enabled": True,
        "submitted": False,
        "status": skip_status,
        "authoritative": False,
        "fitness_source": (
            "provisional_official_anchor" if anchored else "provisional_local_proxy"
        ),
        "provisional": True,
        "provisional_score": provisional_score,
        "search_score": details["search_score"],
        "provisional_local_score": scoring_local_score,
        "provisional_projection_floor": config.provisional_projection_floor,
        "provisional_calibration": details,
    }
    if state.same_as_local_best:
        gate_detail = "candidate is the persisted local-best kernel"
    else:
        gate_status = str(state.gate_result.get("status") or "rejected")
        gate_detail = (
            f"uncertainty gate classified the candidate as {gate_status}; gate latency "
            f"{state.gate_latency_ms:.6f} ms vs local best "
            f"{state.best_latency_before:.6f} ms"
        )
    label = "incumbent-anchored" if anchored else "calibrated local"
    summary = (
        f"{state.local_summary} Official submission skipped because {gate_detail}. "
        f"Using {label} provisional score {provisional_score:.6f}; no remote slot "
        "was consumed."
    )
    return hooks.result(
        "success",
        summary,
        provisional_score,
        metrics=state.metrics,
        artifacts=state.artifacts,
    )


def _submission_reason(state: OfficialEvaluationState) -> str:
    if state.official_probe.get("claimed"):
        return "architecture_probe"
    if state.gate_result.get("challenger_claimed"):
        return "uncertain_challenger"
    return "local_best_improvement"


def _official_metrics(
    state: OfficialEvaluationState,
    config: OfficialFitnessConfig,
    official: dict[str, Any],
    *,
    request_time_s: float,
    submission_reason: str,
) -> dict[str, Any]:
    status = str(official.get("status") or "UNKNOWN").upper()
    metrics = {
        "enabled": True,
        "submitted": True,
        "submission_id": official.get("id"),
        "status": status,
        "is_correct": bool(official.get("is_correct")),
        "sol_score": float(official.get("sol_score") or 0.0),
        "latency_ms": float(official.get("latency_ms") or 0.0),
        "fast_1_count": official.get("fast_1_count"),
        "fast_1_total": official.get("fast_1_total"),
        "avg_speedup": official.get("avg_speedup"),
        "gpu_type": official.get("gpu_type") or config.gpu_type,
        "evaluation_stack_version": (
            official.get("evaluation_stack_version") or config.evaluation_stack_version
        ),
        "submission_mode": config.submission_mode,
        "cache_hit": bool(official.get("cache_hit")),
        "upstream_status": official.get("upstream_status"),
        "next_refresh_at": official.get("next_refresh_at"),
        "request_time_s": request_time_s,
        "submission_reason": submission_reason,
        "official_probe": state.official_probe,
    }
    latency_ms = float(metrics["latency_ms"])
    if latency_ms > 0:
        metrics["local_to_official_latency_ratio"] = state.gate_latency_ms / latency_ms
    return metrics


def _persist_authoritative_result(
    state: OfficialEvaluationState,
    hooks: OfficialFitnessHooks,
    official: dict[str, Any],
    metrics: dict[str, Any],
    *,
    submission_reason: str,
) -> None:
    score = float(metrics["sol_score"])
    latency_ms = float(metrics["latency_ms"])
    hooks.record_authoritative_source(
        state.kernel_source,
        source_language=state.source_language,
        official_score=score,
        official_latency_ms=latency_ms,
        local_latency_ms=state.gate_latency_ms,
        submission_id=official.get("id"),
    )
    hooks.record_calibration(
        local_score=state.local_score,
        official_score=score,
        local_latency_ms=state.gate_latency_ms,
        official_latency_ms=latency_ms,
        submission_id=official.get("id"),
        submission_reason=submission_reason,
    )
    persisted_best = state.new_local_best_record
    if persisted_best is None and state.same_as_local_best and state.local_best_before:
        persisted_best = dict(state.local_best_before)
    if persisted_best is None:
        return
    persisted_best.update(
        {
            "remote_submitted": True,
            "official_submission_id": official.get("id"),
            "official_status": str(metrics["status"]),
            "official_score": score,
            "official_latency_ms": latency_ms,
            "official_anchor_score": score,
            "official_anchor_latency_ms": state.gate_latency_ms,
            "official_anchor_submission_id": official.get("id"),
        }
    )
    hooks.save_local_best(persisted_best, state.kernel_source)


def _provisional_pending_result(
    state: OfficialEvaluationState,
    config: OfficialFitnessConfig,
    hooks: OfficialFitnessHooks,
    official: dict[str, Any],
    official_metrics: dict[str, Any],
) -> Result | None:
    status = str(official_metrics["status"])
    pending_like = status in config.refreshable_statuses
    if not pending_like or config.pending_score_policy not in {
        "provisional",
        "calibrated",
        "local",
        "local_proxy",
    }:
        return None

    scoring_latency_ms = (
        state.best_latency_before
        if state.same_as_local_best and state.best_latency_before > 0
        else state.gate_latency_ms
    )
    scoring_local_score = config.target_latency_ms / scoring_latency_ms
    provisional_score, details = hooks.anchored_provisional_score(
        scoring_local_score,
        scoring_latency_ms,
        state.local_best_before,
    )
    if details["source"] == "incumbent_official_anchor":
        ratio = float(details["candidate_to_anchor_latency_ratio"])
        search_score = float(details["search_score"])
        label = "incumbent-anchored"
    elif config.pending_score_policy in {"local", "local_proxy"}:
        provisional_score = hooks.local_provisional_score(scoring_local_score)
        ratio = 1.0
        search_score = scoring_local_score
        label = "local-proxy fallback"
    else:
        provisional_score, ratio = hooks.provisional_official_score(scoring_local_score)
        search_score = scoring_local_score * ratio
        label = "calibrated"
    official_metrics.update(
        {
            "authoritative": False,
            "fitness_source": "provisional",
            "provisional": True,
            "provisional_score": provisional_score,
            "search_score": search_score,
            "provisional_ratio": ratio,
            "provisional_projection_floor": config.provisional_projection_floor,
            "pending_policy": config.pending_score_policy,
            "provisional_calibration": details,
        }
    )
    pending_detail = (
        "was accepted and queued asynchronously"
        if official.get("upstream_status") == "QUEUED"
        else f"is {status} after bounded polling/cache refresh"
    )
    stack = official_metrics["evaluation_stack_version"]
    summary = (
        f"{state.local_summary} Official {stack} {config.gpu_type} submission "
        f"{official.get('id')} {pending_detail}; using provisional {label} "
        f"search_score {search_score:.6f}, selection_score={provisional_score:.6f} "
        f"(ratio={ratio:.4f}, projection_floor="
        f"{config.provisional_projection_floor:.6f}) so PES can continue. The "
        "projected selection score remains below target and does not certify "
        "leaderboard rank."
    )
    return hooks.result(
        "success",
        summary,
        provisional_score,
        metrics=state.metrics,
        artifacts=state.artifacts,
    )


def evaluate_official_fitness(
    state: OfficialEvaluationState,
    config: OfficialFitnessConfig,
    hooks: OfficialFitnessHooks,
) -> Result:
    """Resolve cache/gate/submission states into the score returned to PES."""
    for resolver in (
        lambda: _reuse_completed_local_best(state, hooks),
        lambda: _skip_below_local_prefilter(state, config, hooks),
        lambda: _skip_outside_official_gate(state, config, hooks),
    ):
        resolved = resolver()
        if resolved is not None:
            return resolved

    reason = _submission_reason(state)
    request_started = time.time()
    official = hooks.submit_official(
        state.workspace,
        state.kernel_source,
        source_language=state.source_language,
        local_score=state.local_score,
        local_latency_ms=state.gate_latency_ms,
        measurement_profile=state.measurement_profile,
        allow_upload=not state.same_as_local_best,
        submission_reason=reason,
    )
    request_time_s = time.time() - request_started
    cache_metadata = official.get("_atrex")
    actual_reason = (
        str(cache_metadata.get("submission_reason"))
        if isinstance(cache_metadata, dict) and cache_metadata.get("submission_reason")
        else reason
    )
    if state.new_local_best_record is not None:
        state.new_local_best_record.update(
            {
                "remote_submitted": True,
                "official_submission_id": official.get("id"),
                "official_status": hooks.official_status(official),
                "official_submission_reason": actual_reason,
            }
        )
        hooks.save_local_best(state.new_local_best_record, state.kernel_source)

    state.metrics["eval_time_s"] = time.time() - state.started_at
    official["status"] = hooks.official_status(official)
    official_metrics = _official_metrics(
        state,
        config,
        official,
        request_time_s=request_time_s,
        submission_reason=actual_reason,
    )
    state.metrics["official"] = official_metrics
    state.artifacts.update(
        {
            "official_submission_id": official.get("id"),
            "official_status": official_metrics["status"],
            "official_result_path": str(
                state.workspace / "official_submission_result.json"
            ),
        }
    )
    submission_path = state.workspace / "official_submission.json"
    if submission_path.exists():
        state.artifacts["official_submission_path"] = str(submission_path)

    authoritative = (
        official_metrics["status"] == "COMPLETED"
        and bool(official_metrics["is_correct"])
        and float(official_metrics["sol_score"]) > 0
    )
    if authoritative:
        official_metrics.update({"authoritative": True, "fitness_source": "official"})
        _persist_authoritative_result(
            state,
            hooks,
            official,
            official_metrics,
            submission_reason=actual_reason,
        )
        stack = official_metrics["evaluation_stack_version"]
        summary = (
            f"Official {stack} {config.gpu_type} fitness: "
            f"submission_id={official.get('id')}, status={official_metrics['status']}, "
            f"is_correct={official_metrics['is_correct']}, "
            f"sol_score={float(official_metrics['sol_score']):.6f}, "
            f"latency={float(official_metrics['latency_ms']):.6f} ms. "
            f"{state.local_summary}"
        )
        return hooks.result(
            "success",
            summary,
            float(official_metrics["sol_score"]),
            metrics=state.metrics,
            artifacts=state.artifacts,
        )

    pending = _provisional_pending_result(
        state,
        config,
        hooks,
        official,
        official_metrics,
    )
    if pending is not None:
        return pending

    summary = (
        f"{state.local_summary} Official "
        f"{official_metrics['evaluation_stack_version']} fitness failed: "
        f"status={official_metrics['status']}, "
        f"is_correct={official_metrics['is_correct']}, "
        f"sol_score={float(official_metrics['sol_score']):.6f}; PES score forced to 0."
    )
    return hooks.result(
        "validation_failed",
        summary,
        0.0,
        metrics=state.metrics,
        artifacts=state.artifacts,
    )
