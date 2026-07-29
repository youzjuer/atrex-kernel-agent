"""SOL-ExecBench kernel 58 policy hooks used by the LoongFlow bridge.

Keep task-specific prompts, architecture escape policy, and seed validation out of
the process-wide compatibility adapter.  The adapter imports these hooks and owns
only the upstream integration points.
"""

from __future__ import annotations

import json
import logging
import os

from orchestrator.loongflow_compat.architecture_islands import (
    architecture_island_id,
    architecture_label,
    architecture_tags,
    extract_architecture_features,
    island_profile,
)
from orchestrator.loongflow_compat.env_flags import enabled as _enabled
from orchestrator.loongflow_compat.protocols import EvolutionMemory, EvolutionSolution
from orchestrator.loongflow_compat.sol58_seed_bank import (
    load_seed_bank,
    seed_bank_fingerprint,
)


logger = logging.getLogger("atrex.pes_compat.sol58")

_NCU_SUMMARY_MARKER = "Atrex NCU evidence protocol"
_NCU_SUMMARY_INSTRUCTIONS = f"""

### {_NCU_SUMMARY_MARKER}
The parent and child evaluation JSON may contain `metrics.ncu_analysis`. Interpret it in the
reflection when present:
1. Trust counters only when `status` is `completed`; `failed`, `timeout`, `unavailable`, and
   `skipped` are profiler availability states and must never change the fitness assessment.
2. NCU covers the named kernel and one representative workload, not end-to-end latency. State
   that scope and do not generalize one launch to every workload or pipeline stage.
3. Compare parent and child counters only when workload UUID/axes and profiled kernel role are
   comparable. Connect code changes -> counter evidence -> measured latency, and identify conflicts
   between counters and timing instead of forcing a causal story.
4. Use `findings`, their confidence/evidence, and `optimization_implications` to propose at most
   three concrete next experiments. Do not invent unavailable metrics or treat heuristic findings
   as proof.
5. Include a concise `NCU interpretation` section in the reflection. The evaluator score and
   correctness result remain the sole promotion criteria.
"""


def _positive_int_env(name: str, default: int) -> int:
    try:
        value = int(os.environ.get(name, str(default)))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a positive integer") from exc
    if value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _append_ncu_summary_instructions(prompt: str) -> str:
    if _NCU_SUMMARY_MARKER in prompt:
        return prompt
    return prompt.rstrip() + _NCU_SUMMARY_INSTRUCTIONS


def _patch_summary_ncu_interpretation() -> None:
    if not _enabled("SOL58_NCU_SUMMARY", "0"):
        return
    try:
        from agents.math_agent.summary import summary_agent
    except ImportError as exc:
        raise RuntimeError("cannot import the LoongFlow Summary prompt target") from exc

    prompt = getattr(summary_agent, "EVOLVE_SUMMARY_USER_PROMPT", "")
    if not isinstance(prompt, str):
        return
    summary_agent.EVOLVE_SUMMARY_USER_PROMPT = _append_ncu_summary_instructions(prompt)


def _stagnation_state(memory: EvolutionMemory) -> dict[str, float | int | str]:
    """Measure plateau age from the first iteration that reached the best score."""
    populations = getattr(memory, "populations", {})
    candidates = list(populations.values()) if isinstance(populations, dict) else []
    current_iteration = max(0, int(getattr(memory, "last_iteration", 0) or 0))
    if not candidates:
        return {
            "current_iteration": current_iteration,
            "best_iteration": current_iteration,
            "plateau_rounds": 0,
            "best_score": 0.0,
            "incumbent_family": "baseline",
        }

    def score(item: object) -> float:
        try:
            return float(getattr(item, "score", 0.0) or 0.0)
        except (TypeError, ValueError):
            return 0.0

    best_score = max(score(candidate) for candidate in candidates)
    best_iterations = [
        max(0, int(getattr(candidate, "iteration", 0) or 0))
        for candidate in candidates
        if abs(score(candidate) - best_score) <= 1e-12
    ]
    incumbent = min(
        (
            candidate
            for candidate in candidates
            if abs(score(candidate) - best_score) <= 1e-12
        ),
        key=lambda candidate: int(getattr(candidate, "iteration", 0) or 0),
    )
    incumbent_family = architecture_label(
        extract_architecture_features(getattr(incumbent, "solution", ""))
    )
    best_iteration = min(best_iterations, default=current_iteration)
    current_iteration = max(
        current_iteration,
        max(
            (int(getattr(candidate, "iteration", 0) or 0) for candidate in candidates),
            default=0,
        ),
    )
    return {
        "current_iteration": current_iteration,
        "best_iteration": best_iteration,
        "plateau_rounds": max(0, current_iteration - best_iteration),
        "best_score": best_score,
        "incumbent_family": incumbent_family,
    }


def _stagnation_seed_parent(
    memory: EvolutionMemory,
    *,
    requested_island: int | None,
    num_islands: int,
) -> dict[str, object] | None:
    """Return an unscored architecture seed at most once per plateau bucket."""
    if not _enabled("ATREX_PES_STAGNATION_SEEDS", "0"):
        return None
    if os.environ.get("SOL58_CODE_LANGUAGE", "cuda_cpp").strip().lower() not in {
        "cuda",
        "cuda_cpp",
        "auto",
    }:
        return None

    threshold = _positive_int_env("ATREX_PES_STAGNATION_ARCHITECTURE_ROUNDS", 12)
    interval = _positive_int_env("ATREX_PES_STAGNATION_SEED_INTERVAL", 20)
    state = _stagnation_state(memory)
    best_marker = (
        int(state["best_iteration"]),
        round(float(state["best_score"]), 12),
    )
    if getattr(memory, "_atrex_stagnation_best_marker", None) != best_marker:
        memory._atrex_stagnation_best_marker = best_marker
        memory._atrex_stagnation_seed_bucket = None
        memory._atrex_stagnation_retry_bucket = None
        memory._atrex_stagnation_seed_retries = 0
    plateau_rounds = int(state["plateau_rounds"])
    if plateau_rounds < threshold:
        return None
    bucket = (plateau_rounds - threshold) // interval
    if getattr(memory, "_atrex_stagnation_retry_bucket", None) != bucket:
        memory._atrex_stagnation_retry_bucket = bucket
        memory._atrex_stagnation_seed_retries = 0
    if getattr(memory, "_atrex_stagnation_seed_bucket", None) == bucket:
        return None

    seeds = load_seed_bank(language="cuda_cpp")
    retries = max(0, int(getattr(memory, "_atrex_stagnation_seed_retries", 0) or 0))
    seed = seeds[(bucket + retries) % len(seeds)]
    features = extract_architecture_features(seed.source)
    detected_family = architecture_label(features)
    if detected_family != seed.family:
        raise RuntimeError(
            f"SOL58 seed {seed.seed_id!r} declares {seed.family!r} but feature "
            f"analysis classified it as {detected_family!r}"
        )
    if num_islands >= 8:
        seed_island = architecture_island_id(detected_family, 0, num_islands)
    else:
        seed_island = int(requested_island or 0) % max(1, int(num_islands))

    directive = (
        "MANDATORY STAGNATION ESCAPE: treat this unscored seed as a different "
        f"algorithm-family starting point ({seed.family}). Produce a complete, "
        "self-contained CUDA architecture experiment. The child must stay in the "
        f"seed family or another family different from the incumbent family "
        f"({state['incumbent_family']}). Reconstructing the incumbent family does "
        "not satisfy this escape unless the evaluator proves a new local best. "
        "Do not fall back to a constant-only or launch-bound-only edit. Preserve "
        "stable ordering and all 16-workload correctness."
    )
    memory._atrex_stagnation_seed_bucket = bucket
    memory._atrex_last_stagnation_seed_id = seed.seed_id
    memory._atrex_pending_stagnation_escape = {
        "expected_iteration": int(state["current_iteration"]) + 1,
        "bucket": bucket,
        "seed_id": seed.seed_id,
        "seed_family": seed.family,
        "incumbent_family": str(state["incumbent_family"]),
    }
    logger.warning(
        "Forced stagnation escape at iteration %d (best iteration %d, plateau %d): "
        "seed=%s family=%s island=%d bank=%s",
        int(state["current_iteration"]),
        int(state["best_iteration"]),
        plateau_rounds,
        seed.seed_id,
        seed.family,
        seed_island,
        seed_bank_fingerprint(seeds)[:12],
    )
    return {
        "solution": seed.source,
        "solution_id": "",
        "generate_plan": directive,
        "parent_id": "",
        "island_id": seed_island,
        "iteration": int(state["current_iteration"]),
        "generation": 0,
        "sample_cnt": 0,
        "sample_weight": 0.05,
        "score": 0.0,
        "evaluation": "Unscored architecture seed; evaluator evidence is required.",
        "summary": f"{directive} Seed purpose: {seed.purpose}",
        "metadata": {
            "trace": [],
            "stagnation_escape": {
                "required": True,
                "seed_id": seed.seed_id,
                "seed_family": seed.family,
                "current_iteration": int(state["current_iteration"]),
                "best_iteration": int(state["best_iteration"]),
                "plateau_rounds": plateau_rounds,
                "incumbent_score": float(state["best_score"]),
                "incumbent_family": str(state["incumbent_family"]),
                "directive": directive,
            },
            "source_sha256": seed.source_sha256,
            "architecture_features": features,
            "architecture_tags": architecture_tags(features),
            "architecture_label": detected_family,
            "architecture_island_id": seed_island,
            "architecture_home_island_id": seed_island,
            "architecture_island_profile": island_profile(seed_island),
        },
    }


def _local_best_improved(solution: EvolutionSolution) -> bool:
    evaluation = getattr(solution, "evaluation", "")
    if isinstance(evaluation, str):
        try:
            evaluation = json.loads(evaluation)
        except (TypeError, ValueError):
            return False
    if not isinstance(evaluation, dict):
        return False
    metrics = evaluation.get("metrics")
    local_best = metrics.get("local_best") if isinstance(metrics, dict) else None
    return bool(
        isinstance(local_best, dict) and local_best.get("strictly_improved", False)
    )


def _apply_stagnation_architecture_gate(
    memory: EvolutionMemory,
    solution: EvolutionSolution,
    child_family: str,
) -> bool:
    """Reject a forced escape that silently reconstructs the stagnant family."""
    pending = getattr(memory, "_atrex_pending_stagnation_escape", None)
    if not isinstance(pending, dict):
        return True
    expected_iteration = int(pending.get("expected_iteration", -1))
    child_iteration = int(getattr(solution, "iteration", -2) or -2)
    if child_iteration != expected_iteration:
        if child_iteration > expected_iteration:
            logger.warning(
                "Discarding stale stagnation escape gate for iteration %d while "
                "adding iteration %d",
                expected_iteration,
                child_iteration,
            )
            memory._atrex_pending_stagnation_escape = None
        return True

    memory._atrex_pending_stagnation_escape = None
    incumbent_family = str(pending.get("incumbent_family") or "")
    metadata = getattr(solution, "metadata", None)
    if not isinstance(metadata, dict):
        metadata = {}
        solution.metadata = metadata
    local_best_improved = _local_best_improved(solution)
    if child_family != incumbent_family or local_best_improved:
        metadata["stagnation_escape_satisfied"] = {
            **pending,
            "child_family": child_family,
            "local_best_improved": local_best_improved,
        }
        memory._atrex_stagnation_seed_retries = 0
        return True

    original_score = float(getattr(solution, "score", 0.0) or 0.0)
    max_attempts = _positive_int_env("ATREX_PES_STAGNATION_MAX_ATTEMPTS", 2)
    retry_bucket = int(pending.get("bucket", -1))
    if getattr(memory, "_atrex_stagnation_retry_bucket", None) != retry_bucket:
        memory._atrex_stagnation_retry_bucket = retry_bucket
        memory._atrex_stagnation_seed_retries = 0
    retries = max(0, int(getattr(memory, "_atrex_stagnation_seed_retries", 0) or 0)) + 1
    memory._atrex_stagnation_seed_retries = retries
    if retries < max_attempts:
        memory._atrex_stagnation_seed_bucket = None

    violation = {
        **pending,
        "child_family": child_family,
        "reason": "child reconstructed the stagnant incumbent family without a new local best",
        "score_before_gate": original_score,
        "retry": retries,
        "max_attempts": max_attempts,
    }
    metadata["stagnation_escape_violation"] = violation
    solution.score = 0.0
    evaluation = getattr(solution, "evaluation", "")
    payload = evaluation if isinstance(evaluation, dict) else None
    if isinstance(evaluation, str):
        try:
            payload = json.loads(evaluation)
        except (TypeError, ValueError):
            payload = None
    if isinstance(payload, dict):
        payload["score"] = 0.0
        metrics = payload.setdefault("metrics", {})
        if isinstance(metrics, dict):
            metrics["architecture_escape_gate"] = violation
        if isinstance(evaluation, str):
            solution.evaluation = json.dumps(payload, ensure_ascii=False, indent=2)
    logger.error(
        "Rejected stagnation escape child at iteration %d: seed=%s incumbent=%s "
        "child=%s score=%.6f retry=%d/%d",
        child_iteration,
        pending.get("seed_id"),
        incumbent_family,
        child_family,
        original_score,
        retries,
        max_attempts,
    )
    return False


def _finalize_rejected_stagnation_child(
    memory: EvolutionMemory, solution: EvolutionSolution
) -> str:
    """Advance iteration bookkeeping without admitting the rejected child anywhere."""
    lock = getattr(memory, "_lock", None)
    if lock is None:
        raise RuntimeError("stagnation rejection requires an evolution-memory lock")
    with lock:
        prepare = getattr(memory, "_prepare_solution", None)
        if callable(prepare):
            prepare(solution)
        else:
            child_iteration = int(getattr(solution, "iteration", 0) or 0)
            memory.last_iteration = max(
                int(getattr(memory, "last_iteration", 0) or 0), child_iteration
            )
        solution_id = str(getattr(solution, "solution_id", "") or "")
        if not solution_id:
            raise RuntimeError("rejected stagnation child has no solution id")
        metadata = getattr(solution, "metadata", None)
        if not isinstance(metadata, dict):
            metadata = {}
            solution.metadata = metadata
        metadata["database_admission"] = "rejected_stagnation_escape"
    logger.warning(
        "Hard-rejected stagnation escape child %s at iteration %d; no history or "
        "population record was created",
        solution_id,
        int(getattr(solution, "iteration", 0) or 0),
    )
    return solution_id


def _verify_stagnation_seed_bank() -> tuple[bool, str]:
    if not _enabled("ATREX_PES_STAGNATION_SEEDS", "0"):
        return True, "disabled"
    seeds = load_seed_bank(language="cuda_cpp")
    families = {seed.family for seed in seeds}
    required = {"cub_radix_sort", "expert_parallel_scan"}
    missing = sorted(required - families)
    if missing:
        return False, f"missing architecture seed families {missing}"
    return True, f"{len(seeds)} CUDA seeds sha256={seed_bank_fingerprint(seeds)[:12]}"


def _verify_ncu_patch() -> tuple[bool, str]:
    if not _enabled("SOL58_NCU_SUMMARY", "0"):
        return True, "disabled"
    from agents.math_agent.summary import summary_agent

    prompt = getattr(summary_agent, "EVOLVE_SUMMARY_USER_PROMPT", "")
    applied = isinstance(prompt, str) and _NCU_SUMMARY_MARKER in prompt
    return applied, "summary NCU prompt" if applied else "summary NCU marker missing"
