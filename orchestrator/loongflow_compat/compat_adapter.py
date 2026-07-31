"""Explicit compatibility adapter for supported LoongFlow entrypoints."""

from __future__ import annotations

import hashlib
import inspect
import json
import logging
import os
from pathlib import Path
from collections.abc import Callable, Sequence
from typing import Any

from orchestrator.loongflow_compat.architecture_islands import (
    architecture_home_occupancy,
    architecture_label,
    classify_and_route_solution,
    enforce_architecture_population_limit,
    extract_architecture_features,
    initialize_architecture_memory,
    maybe_exchange_islands,
    restore_architecture_checkpoint,
    restore_stagnation_checkpoint,
    validate_architecture_island_count,
    write_architecture_checkpoint,
)
from orchestrator.loongflow_compat.checkpoint_compat import (
    restore_checkpoint_population_indexes as _restore_checkpoint_population_indexes_impl,
)
from orchestrator.loongflow_compat.env_flags import enabled as _enabled
from orchestrator.loongflow_compat.protocols import EvolutionMemory, EvolutionSolution
from orchestrator.loongflow_compat.sol58_task_hooks import (
    _NCU_SUMMARY_MARKER,
    _architecture_bootstrap_seed_parent,
    _append_ncu_summary_instructions,
    _apply_stagnation_architecture_gate,
    _finalize_rejected_stagnation_child,
    _local_best_improved,
    _patch_summary_ncu_interpretation,
    _positive_int_env,
    _stagnation_seed_parent,
    _stagnation_state,
    _record_architecture_bootstrap_outcome,
    _verify_ncu_patch,
    _verify_stagnation_seed_bank,
)
from orchestrator.sol58_pes.fitness_calibration import project_provisional_score

__all__ = [
    "_NCU_SUMMARY_MARKER",
    "_append_ncu_summary_instructions",
    "_local_best_improved",
    "_positive_int_env",
    "_stagnation_state",
]

logger = logging.getLogger("atrex.pes_compat")
PATCH_MANIFEST: dict[str, dict[str, object]] = {}


def _patch_litellm_drop_params() -> None:
    if not _enabled("ATREX_LITELLM_DROP_PARAMS", "0"):
        return
    try:
        import litellm

        litellm.drop_params = True
    except Exception as exc:
        raise RuntimeError("cannot enable LiteLLM drop_params compatibility") from exc


def _verify_litellm_drop_params() -> tuple[bool, str]:
    if not _enabled("ATREX_LITELLM_DROP_PARAMS", "0"):
        return True, "disabled"
    import litellm

    applied = bool(getattr(litellm, "drop_params", False))
    return applied, "litellm.drop_params" if applied else "drop_params remains false"


def _truncate_text(value: object, limit: int) -> object:
    if not isinstance(value, str) or limit <= 0 or len(value) <= limit:
        return value
    head = max(0, limit * 2 // 3)
    tail = max(0, limit - head)
    return (
        value[:head]
        + f"\n\n...[atrex compacted {len(value) - limit} chars]...\n\n"
        + value[-tail:]
    )


def _compact_solution_record(record: object) -> object:
    if not isinstance(record, dict):
        return record

    compact = dict(record)
    solution = compact.get("solution")
    if isinstance(solution, str):
        compact["solution_sha1"] = hashlib.sha1(solution.encode("utf-8")).hexdigest()
        compact["solution_chars"] = len(solution)
        compact["solution"] = _truncate_text(
            solution, int(os.environ.get("ATREX_PES_DB_SOLUTION_CHARS", "65536"))
        )

    summary = compact.get("summary")
    if isinstance(summary, str):
        compact["summary"] = _truncate_text(
            summary, int(os.environ.get("ATREX_PES_DB_SUMMARY_CHARS", "16384"))
        )

    evaluation = compact.get("evaluation")
    if isinstance(evaluation, str):
        try:
            parsed = json.loads(evaluation)
        except (TypeError, ValueError):
            compact["evaluation"] = _truncate_text(
                evaluation,
                int(os.environ.get("ATREX_PES_DB_EVALUATION_CHARS", "16384")),
            )
        else:
            metrics = parsed.get("metrics") if isinstance(parsed, dict) else None
            per_workload = (
                parsed.get("per_workload") if isinstance(parsed, dict) else None
            )
            compact_eval = {
                "status": parsed.get("status") if isinstance(parsed, dict) else None,
                "summary": parsed.get("summary") if isinstance(parsed, dict) else None,
                "score": parsed.get("score") if isinstance(parsed, dict) else None,
                "metrics": metrics,
            }
            if isinstance(per_workload, list):
                compact_eval["per_workload_latency_ms"] = [
                    {
                        "index": item.get("index"),
                        "axes": item.get("axes"),
                        "status": item.get("status"),
                        "latency_ms": item.get("latency_ms"),
                    }
                    for item in per_workload[:32]
                    if isinstance(item, dict)
                ]
            compact["evaluation"] = compact_eval
    return compact


def _compact_result(value: object) -> object:
    if isinstance(value, list):
        return [_compact_solution_record(item) for item in value]
    if isinstance(value, dict):
        return _compact_solution_record(value)
    return value


def _solution_source_hash(solution: EvolutionSolution | str) -> str:
    source = getattr(solution, "solution", solution)
    text = source if isinstance(source, str) else str(source or "")
    return hashlib.sha256(text.strip().encode("utf-8")).hexdigest()


def _adaptive_exploration_rate(
    base_rate: float,
    recent_scores: list[float],
    *,
    plateau_rounds: int = 0,
    stagnation_rounds: int = 12,
    empty_home_islands: int = 0,
    num_islands: int = 1,
) -> float:
    """Raise exploration from either local score similarity or global stagnation."""
    rate = max(0.0, float(base_rate))
    multiplier = 1
    if len(recent_scores) >= 5:
        scores = recent_scores[:5]
        deltas = [abs(scores[i] - scores[i + 1]) for i in range(4)]
        if all(delta < 0.001 for delta in deltas):
            multiplier = max(multiplier, 4)
        elif all(delta < 0.01 for delta in deltas):
            multiplier = max(multiplier, 2)

    stagnation_rounds = max(1, int(stagnation_rounds))
    plateau_rounds = max(0, int(plateau_rounds))
    if plateau_rounds >= 2 * stagnation_rounds:
        multiplier = max(multiplier, 4)
    elif plateau_rounds >= stagnation_rounds:
        multiplier = max(multiplier, 2)

    empty_home_islands = max(0, int(empty_home_islands))
    num_islands = max(1, int(num_islands))
    if empty_home_islands >= max(1, num_islands // 2):
        multiplier = max(multiplier, 4)
    elif empty_home_islands:
        multiplier = max(multiplier, 2)
    return min(rate * multiplier, 0.9)


def _canonical_solution(
    solutions: Sequence[EvolutionSolution],
) -> EvolutionSolution:
    return min(
        solutions,
        key=lambda solution: (
            -float(getattr(solution, "score", 0.0) or 0.0),
            int(getattr(solution, "iteration", 0) or 0),
            float(getattr(solution, "timestamp", 0.0) or 0.0),
        ),
    )


def _bounded_sample_weight(score: float, weights: list[float]) -> float:
    """Retain lineage preference without allowing equal-source snowball growth."""
    positive = [float(weight) for weight in weights if weight and float(weight) > 0]
    observed = max(positive, default=1.0)
    quality_cap = max(1.0, 1.0 + 3.0 * max(0.0, float(score or 0.0)))
    return max(0.05, min(observed, quality_cap))


def _load_authoritative_fitness_registry() -> dict[str, dict[str, object]]:
    eval_root = Path(os.environ.get("SOL58_EVAL_ROOT", "/tmp/sol58_pes_eval"))
    path = eval_root / "official_cache" / "authoritative_fitness.json"
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, TypeError, ValueError) as exc:
        logger.warning(
            "Ignoring unreadable authoritative fitness registry %s: %s: %s",
            path,
            type(exc).__name__,
            exc,
        )
        return {}
    sources = payload.get("sources") if isinstance(payload, dict) else None
    return sources if isinstance(sources, dict) else {}


def _reconcile_authoritative_scores(memory: EvolutionMemory) -> int:
    """Replace stale provisional scores with terminal official source fitness."""
    registry = _load_authoritative_fitness_registry()
    solutions = getattr(memory, "solutions", None)
    populations = getattr(memory, "populations", None)
    lock = getattr(memory, "_lock", None)
    if not registry:
        return 0
    if (
        not isinstance(solutions, dict)
        or not isinstance(populations, dict)
        or lock is None
    ):
        raise RuntimeError(
            "authoritative fitness reconciliation requires LoongFlow "
            "solutions/populations dictionaries and _lock"
        )

    changed_ids: set[str] = set()
    with lock:
        records: dict[str, EvolutionSolution] = dict(solutions)
        if isinstance(populations, dict):
            records.update(populations)

        for solution_id, solution in records.items():
            source_hash = _solution_source_hash(solution)
            authoritative = registry.get(source_hash)
            if not isinstance(authoritative, dict):
                continue
            raw_official_score = authoritative.get("official_score")
            if raw_official_score is not None and not isinstance(
                raw_official_score, (int, float, str)
            ):
                continue
            try:
                official_score = float(raw_official_score or 0.0)
            except (TypeError, ValueError):
                continue
            official_status = str(authoritative.get("status") or "").strip().upper()
            if not official_status and official_score > 0:
                # Version-1 registries stored successful rows without a status.
                official_status = "COMPLETED"
            if official_status not in {"COMPLETED", "FAILED", "ERROR", "CANCELLED"}:
                continue
            raw_is_correct = authoritative.get("is_correct")
            is_correct = (
                raw_is_correct
                if isinstance(raw_is_correct, bool)
                else official_score > 0
            )
            if official_status != "COMPLETED" or not is_correct or official_score <= 0:
                official_score = 0.0

            previous_score = float(getattr(solution, "score", 0.0) or 0.0)
            if previous_score != official_score:
                solution.score = official_score
                changed_ids.add(str(solution_id))
            metadata = getattr(solution, "metadata", None)
            if not isinstance(metadata, dict):
                metadata = {}
                solution.metadata = metadata
            metadata.update(
                {
                    "source_sha256": source_hash,
                    "fitness_source": "official",
                    "official_submission_id": authoritative.get("submission_id"),
                    "official_score": official_score,
                    "official_status": official_status,
                    "official_is_correct": bool(is_correct),
                }
            )
            solution.sample_weight = _bounded_sample_weight(
                official_score,
                [getattr(solution, "sample_weight", 0.0)],
            )

        if isinstance(populations, dict) and populations:
            best = max(
                populations.values(),
                key=lambda solution: (
                    float(getattr(solution, "score", 0.0) or 0.0),
                    -int(getattr(solution, "iteration", 0) or 0),
                ),
            )
            memory.best_solution_id = best.solution_id

            islands = getattr(memory, "islands", None)
            island_bests = getattr(memory, "island_best_solution", None)
            if isinstance(islands, list) and isinstance(island_bests, list):
                for island_index, island in enumerate(islands):
                    candidates = [
                        populations[solution_id]
                        for solution_id in island
                        if solution_id in populations
                    ]
                    if not candidates or island_index >= len(island_bests):
                        continue
                    island_best = max(
                        candidates,
                        key=lambda solution: (
                            float(getattr(solution, "score", 0.0) or 0.0),
                            -int(getattr(solution, "iteration", 0) or 0),
                        ),
                    )
                    island_bests[island_index] = island_best.solution_id

    return len(changed_ids)


def _evaluation_payload(
    solution: EvolutionSolution,
) -> tuple[dict[str, Any] | None, bool]:
    evaluation = getattr(solution, "evaluation", None)
    if isinstance(evaluation, dict):
        return evaluation, False
    if not isinstance(evaluation, str):
        return None, False
    try:
        payload = json.loads(evaluation)
    except (TypeError, ValueError):
        return None, True
    return (payload if isinstance(payload, dict) else None), True


def _is_authoritative_evaluation(
    solution: EvolutionSolution, payload: dict[str, Any], metrics: dict[str, Any]
) -> bool:
    official = metrics.get("official")
    official = official if isinstance(official, dict) else {}
    certified = metrics.get("certified_score")
    metadata = getattr(solution, "metadata", None)
    return bool(
        official.get("authoritative", False)
        or certified is not None
        or (isinstance(metadata, dict) and metadata.get("fitness_source") == "official")
        or str(metrics.get("fitness_source") or "").lower() == "official"
        or str(payload.get("fitness_source") or "").lower() == "official"
    )


def _migrate_checkpoint_selection_scores(memory: EvolutionMemory) -> int:
    """Recover strict provisional ordering from legacy flat-cap checkpoints."""
    solutions = getattr(memory, "solutions", None)
    populations = getattr(memory, "populations", None)
    lock = getattr(memory, "_lock", None)
    if (
        not isinstance(solutions, dict)
        or not isinstance(populations, dict)
        or lock is None
    ):
        raise RuntimeError(
            "checkpoint score migration requires solutions/populations dictionaries and _lock"
        )
    try:
        target = float(os.environ.get("SOL58_TARGET_SCORE", "0.904135"))
        floor = float(
            os.environ.get("SOL58_OFFICIAL_PROVISIONAL_SCORE_CAP", "0.899135")
        )
    except ValueError as exc:
        raise RuntimeError("invalid provisional score migration environment") from exc

    changed = 0
    with lock:
        records: dict[str, EvolutionSolution] = dict(solutions)
        records.update(populations)
        for solution in records.values():
            payload, serialized = _evaluation_payload(solution)
            if payload is None:
                continue
            metrics = payload.get("metrics")
            if not isinstance(metrics, dict) or _is_authoritative_evaluation(
                solution, payload, metrics
            ):
                continue

            official = metrics.get("official")
            official = official if isinstance(official, dict) else {}
            calibration = official.get("provisional_calibration")
            calibration = calibration if isinstance(calibration, dict) else {}
            selection = metrics.get("selection_score")
            search = metrics.get("search_score")
            source = "metrics.selection_score"
            if selection is not None:
                try:
                    selection_score = max(0.0, float(selection))
                    search_score = max(
                        0.0,
                        float(search if search is not None else selection_score),
                    )
                except (TypeError, ValueError):
                    continue
            else:
                source = "metrics.search_score"
                if search is None:
                    search = calibration.get("search_score")
                    source = "provisional_calibration.search_score"
                if search is None:
                    anchor = calibration.get("anchor_score")
                    ratio = calibration.get("candidate_to_anchor_latency_ratio")
                    if anchor is None or ratio is None:
                        continue
                    try:
                        search = float(anchor) * float(ratio)
                    except (TypeError, ValueError):
                        continue
                    source = "legacy_anchor_latency_ratio"
                try:
                    search_score = max(0.0, float(search))
                except (TypeError, ValueError):
                    continue
                selection_score = project_provisional_score(
                    search_score,
                    target=target,
                    floor=floor,
                )

            previous = float(getattr(solution, "score", 0.0) or 0.0)
            metrics["search_score"] = search_score
            metrics["selection_score"] = selection_score
            if isinstance(calibration, dict):
                calibration["search_score"] = search_score
                calibration["selection_score"] = selection_score
            official["search_score"] = search_score
            official["provisional_score"] = selection_score
            payload["score"] = selection_score
            solution.score = selection_score
            solution.sample_weight = _bounded_sample_weight(
                selection_score,
                [getattr(solution, "sample_weight", 0.0)],
            )
            metadata = getattr(solution, "metadata", None)
            if not isinstance(metadata, dict):
                metadata = {}
                solution.metadata = metadata
            metadata["checkpoint_score_migration"] = {
                "version": 1,
                "source": source,
                "previous_selection_score": previous,
                "search_score": search_score,
                "selection_score": selection_score,
                "target": target,
                "floor": floor,
            }
            if serialized:
                solution.evaluation = json.dumps(payload, ensure_ascii=False, indent=2)
            if abs(previous - selection_score) > 1e-15:
                changed += 1

        if populations:
            best = max(populations.values(), key=lambda item: float(item.score or 0.0))
            memory.best_solution_id = best.solution_id
    return changed


def _deduplicate_memory_indexes(memory: EvolutionMemory) -> int:
    """Keep one selectable representative per source and island; retain lineage records."""
    populations = getattr(memory, "populations", None)
    islands = getattr(memory, "islands", None)
    if not isinstance(populations, dict) or not isinstance(islands, list):
        raise RuntimeError(
            "source deduplication requires LoongFlow populations:dict and islands:list"
        )

    duplicate_to_canonical: dict[str, str] = {}
    lock = getattr(memory, "_lock", None)
    if lock is None:
        raise RuntimeError("source deduplication requires LoongFlow memory._lock")

    with lock:
        assigned_ids = set().union(*islands) if islands else set()
        for solution_id in list(populations):
            if solution_id not in assigned_ids:
                populations.pop(solution_id, None)

        for island in islands:
            groups: dict[str, list[EvolutionSolution]] = {}
            for solution_id in list(island):
                solution = populations.get(solution_id)
                if solution is None:
                    island.discard(solution_id)
                    continue
                groups.setdefault(_solution_source_hash(solution), []).append(solution)

            for source_hash, group in groups.items():
                canonical = _canonical_solution(group)
                metadata = getattr(canonical, "metadata", None)
                if isinstance(metadata, dict):
                    metadata["source_sha256"] = source_hash
                    metadata["source_variant_count"] = max(
                        int(metadata.get("source_variant_count", 0)),
                        len(group),
                    )
                canonical.sample_weight = _bounded_sample_weight(
                    getattr(canonical, "score", 0.0),
                    [getattr(solution, "sample_weight", 0.0) for solution in group],
                )

                for duplicate in group:
                    if duplicate.solution_id == canonical.solution_id:
                        continue
                    duplicate_to_canonical[duplicate.solution_id] = (
                        canonical.solution_id
                    )
                    duplicate_metadata = getattr(duplicate, "metadata", None)
                    if isinstance(duplicate_metadata, dict):
                        duplicate_metadata["duplicate_of"] = canonical.solution_id
                        duplicate_metadata["source_sha256"] = source_hash
                    island.discard(duplicate.solution_id)
                    populations.pop(duplicate.solution_id, None)

        elites = getattr(memory, "elites", None)
        if isinstance(elites, set):
            elite_groups: dict[str, list[EvolutionSolution]] = {}
            for solution_id in list(elites):
                solution_id = duplicate_to_canonical.get(solution_id, solution_id)
                solution = populations.get(solution_id)
                if solution is not None:
                    elite_groups.setdefault(_solution_source_hash(solution), []).append(
                        solution
                    )
            elites.clear()
            for group in elite_groups.values():
                elites.add(_canonical_solution(group).solution_id)

        feature_maps = getattr(memory, "island_feature_maps", None)
        if isinstance(feature_maps, list):
            for feature_map in feature_maps:
                seen_sources: set[str] = set()
                for key, solution_id in list(feature_map.items()):
                    solution_id = duplicate_to_canonical.get(solution_id, solution_id)
                    solution = populations.get(solution_id)
                    if solution is None:
                        feature_map.pop(key, None)
                        continue
                    source_hash = _solution_source_hash(solution)
                    if source_hash in seen_sources:
                        feature_map.pop(key, None)
                        continue
                    seen_sources.add(source_hash)
                    feature_map[key] = solution_id

        best_solution_id = getattr(memory, "best_solution_id", None)
        if best_solution_id in duplicate_to_canonical:
            memory.best_solution_id = duplicate_to_canonical[best_solution_id]

        island_bests = getattr(memory, "island_best_solution", None)
        if isinstance(island_bests, list):
            for index, solution_id in enumerate(island_bests):
                island_bests[index] = duplicate_to_canonical.get(
                    solution_id, solution_id
                )

        if hasattr(memory, "island_capacity"):
            memory.island_capacity = [len(island) for island in islands]

    return len(duplicate_to_canonical)


def _restore_checkpoint_population_indexes(
    memory: EvolutionMemory, checkpoint_path: str
):
    """Validate and repair selectable checkpoint indexes, or fail with context."""
    return _restore_checkpoint_population_indexes_impl(
        memory, checkpoint_path, _canonical_solution
    )


def _require_signature(
    callable_obj: Callable[..., object], expected: tuple[str, ...]
) -> None:
    actual = tuple(inspect.signature(callable_obj).parameters)
    if actual != expected:
        name = getattr(callable_obj, "__qualname__", repr(callable_obj))
        raise RuntimeError(
            f"LoongFlow compatibility signature mismatch for {name}: "
            f"expected {expected}, got {actual}"
        )


def _architecture_num_islands(config: object) -> int:
    return validate_architecture_island_count(
        int(getattr(config, "num_islands", 8) or 8)
    )


def _architecture_retention_settings() -> tuple[int, float]:
    try:
        minimum_home = int(os.environ.get("ATREX_PES_MIN_HOME_PER_ISLAND", "6"))
        maximum_migrants = float(
            os.environ.get("ATREX_PES_MAX_MIGRANT_FRACTION", "0.2")
        )
    except ValueError as exc:
        raise ValueError("invalid architecture retention environment") from exc
    if minimum_home < 0:
        raise ValueError("ATREX_PES_MIN_HOME_PER_ISLAND must be >= 0")
    if not 0.0 <= maximum_migrants <= 1.0:
        raise ValueError("ATREX_PES_MAX_MIGRANT_FRACTION must be in [0, 1]")
    return minimum_home, maximum_migrants


def _patch_evolution_database_selection() -> None:
    source_dedup_enabled = _enabled("ATREX_PES_SOURCE_DEDUP")
    architecture_enabled = _enabled("ATREX_PES_ARCHITECTURE_ISLANDS")
    stagnation_seeds_enabled = _enabled("ATREX_PES_STAGNATION_SEEDS", "0")
    if (
        not source_dedup_enabled
        and not architecture_enabled
        and not stagnation_seeds_enabled
    ):
        return
    try:
        from loongflow.agentsdk.memory.evolution.in_memory import InMemory
        from loongflow.agentsdk.memory.evolution.boltzmann import (
            select_parents_with_dynamic_temperature,
        )
        from loongflow.framework.pes.database.database import EvolveDatabase
    except Exception as exc:
        raise RuntimeError(
            "cannot import LoongFlow evolution database targets"
        ) from exc

    if getattr(EvolveDatabase, "_atrex_evolution_patched", False):
        return

    _require_signature(EvolveDatabase.__init__, ("self", "config"))
    _require_signature(EvolveDatabase.sample_solution, ("self", "island_id"))
    _require_signature(EvolveDatabase.add_solution, ("self", "solution"))
    _require_signature(EvolveDatabase.load_checkpoint, ("self", "checkpoint_path"))
    _require_signature(
        EvolveDatabase.save_checkpoint, ("self", "checkpoint_path", "tag")
    )
    if architecture_enabled:
        _require_signature(InMemory._prepare_solution, ("self", "solution"))
        _require_signature(InMemory._check_migration, ("self",))
        _require_signature(
            InMemory._enforce_population_limit,
            ("self", "exclude_solution_id"),
        )

    original_init = EvolveDatabase.__init__
    original_add_solution = EvolveDatabase.add_solution
    original_load_checkpoint = EvolveDatabase.load_checkpoint
    original_save_checkpoint = EvolveDatabase.save_checkpoint

    def architecture_settings(database):
        num_islands = _architecture_num_islands(database.config)
        migration_interval = max(
            1, int(getattr(database.config, "migration_interval", 20) or 20)
        )
        return num_islands, migration_interval

    def patched_init(self, config):
        if architecture_enabled:
            _architecture_num_islands(config)
        original_init(self, config)

    patched_init._atrex_island_config_validated = True

    if architecture_enabled and not getattr(
        InMemory, "_atrex_architecture_patched", False
    ):
        original_prepare_solution = InMemory._prepare_solution

        def patched_prepare_solution(self, solution):
            original_prepare_solution(self, solution)
            metadata = getattr(solution, "metadata", None)
            if not isinstance(metadata, dict):
                return
            forced_island = metadata.get("architecture_island_id")
            if forced_island is None:
                return
            solution.island_id = int(forced_island) % max(1, int(self.num_islands))

        async def patched_check_migration(self):
            maybe_exchange_islands(
                self,
                validate_architecture_island_count(self.num_islands),
                max(1, int(self.migration_interval)),
            )
            minimum_home, maximum_migrants = _architecture_retention_settings()
            enforce_architecture_population_limit(
                self,
                minimum_home_per_island=minimum_home,
                maximum_migrant_fraction=maximum_migrants,
            )

        def patched_enforce_population_limit(self, exclude_solution_id=None):
            minimum_home, maximum_migrants = _architecture_retention_settings()
            enforce_architecture_population_limit(
                self,
                exclude_solution_id=exclude_solution_id,
                minimum_home_per_island=minimum_home,
                maximum_migrant_fraction=maximum_migrants,
            )

        patched_check_migration._atrex_fixed_interval_exchange = True
        patched_enforce_population_limit._atrex_architecture_retention = True
        InMemory._prepare_solution = patched_prepare_solution
        InMemory._check_migration = patched_check_migration
        InMemory._enforce_population_limit = patched_enforce_population_limit
        InMemory._atrex_architecture_patched = True

    def sample_architecture_parent(memory, island_id, exploration_rate):
        with memory._lock:
            populations = memory.populations
            if island_id is None:
                allowed_ids = set(populations)
            elif 0 <= int(island_id) < len(memory.islands):
                allowed_ids = set(memory.islands[int(island_id)])
            else:
                return None
            candidates = [
                populations[solution_id]
                for solution_id in allowed_ids
                if solution_id in populations
            ]
            elites = [
                populations[solution_id]
                for solution_id in memory.elites
                if solution_id in allowed_ids and solution_id in populations
            ]
            if not candidates and island_id is not None:
                candidates = list(populations.values())
                elites = [
                    populations[solution_id]
                    for solution_id in memory.elites
                    if solution_id in populations
                ]
                logger.warning(
                    "Island %s has no selectable members; using global parent fallback",
                    island_id,
                )
        if not candidates:
            return None
        return select_parents_with_dynamic_temperature(
            solutions=candidates,
            elites=elites,
            initial_temp=memory.boltzmann_temperature,
            use_sampling_weight=memory.use_sampling_weight,
            sampling_weight_power=memory.sampling_weight_power,
            exploration_rate=exploration_rate,
        )

    def patched_sample_solution(self, island_id=None):
        memory = getattr(self._evolution_memory, "_memory", None)
        if memory is None:
            raise RuntimeError(
                "LoongFlow compatibility contract mismatch: "
                "EvolveDatabase._evolution_memory._memory is unavailable"
            )

        if architecture_enabled:
            num_islands, migration_interval = architecture_settings(self)
            initialize_architecture_memory(memory, num_islands, migration_interval)
        reconciled = _reconcile_authoritative_scores(memory)
        removed = _deduplicate_memory_indexes(memory) if source_dedup_enabled else 0
        configured_islands = (
            architecture_settings(self)[0]
            if architecture_enabled
            else max(1, int(getattr(self.config, "num_islands", 1) or 1))
        )
        forced_seed = None
        if architecture_enabled:
            forced_seed = _architecture_bootstrap_seed_parent(
                memory,
                requested_island=island_id,
                num_islands=configured_islands,
            )
        if forced_seed is None:
            forced_seed = _stagnation_seed_parent(
                memory, requested_island=island_id, num_islands=configured_islands
            )
        if forced_seed is not None:
            return forced_seed
        recent = memory.list_solutions(filter_type="desc", limit=5)
        recent_scores = [
            float(solution.score)
            for solution in recent
            if getattr(solution, "score", None) is not None
        ]
        stagnation = _stagnation_state(memory)
        home_occupancy = (
            architecture_home_occupancy(memory) if architecture_enabled else []
        )
        exploration_rate = _adaptive_exploration_rate(
            self.config.exploration_rate,
            recent_scores,
            plateau_rounds=int(stagnation["plateau_rounds"]),
            stagnation_rounds=_positive_int_env(
                "ATREX_PES_STAGNATION_ARCHITECTURE_ROUNDS", 12
            ),
            empty_home_islands=sum(count == 0 for count in home_occupancy),
            num_islands=len(home_occupancy) or configured_islands,
        )
        if removed:
            logger.info(
                "Collapsed %d duplicate source variants before parent sampling",
                removed,
            )
        if reconciled:
            logger.info(
                "Reconciled %d solution scores from completed official fitness",
                reconciled,
            )
        if exploration_rate != self.config.exploration_rate:
            logger.info(
                "Adaptive exploration rate %.3f -> %.3f (plateau=%d empty_home=%d)",
                self.config.exploration_rate,
                exploration_rate,
                int(stagnation["plateau_rounds"]),
                sum(count == 0 for count in home_occupancy),
            )
        solution = (
            sample_architecture_parent(memory, island_id, exploration_rate)
            if architecture_enabled
            else self._evolution_memory.sample(island_id, exploration_rate)
        )
        return solution.to_dict() if solution is not None else {}

    async def patched_add_solution(self, solution):
        memory = getattr(self._evolution_memory, "_memory", None)
        source = getattr(solution, "solution", "")
        if memory is None:
            raise RuntimeError(
                "LoongFlow compatibility contract mismatch while adding a solution: "
                "EvolveDatabase._evolution_memory._memory is unavailable"
            )

        child_family = "baseline"
        if architecture_enabled:
            num_islands, migration_interval = architecture_settings(self)
            initialize_architecture_memory(memory, num_islands, migration_interval)
            if isinstance(source, str) and source.strip():
                analysis = classify_and_route_solution(memory, solution, num_islands)
                child_family = str(analysis["label"])
                logger.info(
                    "Summary architecture route: label=%s pca_cluster=%d island=%d",
                    analysis["label"],
                    analysis["cluster"],
                    analysis["island_id"],
                )
        elif isinstance(source, str) and source.strip():
            child_family = architecture_label(extract_architecture_features(source))

        if isinstance(source, str) and source.strip():
            _record_architecture_bootstrap_outcome(memory, solution, child_family)
            admitted = _apply_stagnation_architecture_gate(
                memory, solution, child_family
            )
            if not admitted:
                solution_id = _finalize_rejected_stagnation_child(memory, solution)
                if architecture_enabled:
                    await memory._check_migration()
                return solution_id

        if not isinstance(source, str) or not source.strip():
            solution_id = await original_add_solution(self, solution)
            if architecture_enabled:
                await memory._check_migration()
            return solution_id

        _reconcile_authoritative_scores(memory)
        if source_dedup_enabled:
            _deduplicate_memory_indexes(memory)
        source_hash = _solution_source_hash(solution)
        island_id = int(getattr(solution, "island_id", 0) or 0)
        island_ids = (
            memory.islands[island_id] if 0 <= island_id < len(memory.islands) else set()
        )
        matching = [
            memory.populations[solution_id]
            for solution_id in island_ids
            if solution_id in memory.populations
            and _solution_source_hash(memory.populations[solution_id]) == source_hash
        ]

        if source_dedup_enabled and matching:
            canonical = _canonical_solution(matching)
            with memory._lock:
                memory._prepare_solution(solution)
                if not isinstance(getattr(solution, "metadata", None), dict):
                    solution.metadata = {}
                solution.metadata["duplicate_of"] = canonical.solution_id
                solution.metadata["source_sha256"] = source_hash
                solution.metadata["MAP_Elite_feature"] = canonical.metadata.get(
                    "MAP_Elite_feature", ""
                )
                solution.sample_weight = 0.05
                memory.solutions[solution.solution_id] = solution
                if not isinstance(getattr(canonical, "metadata", None), dict):
                    canonical.metadata = {}
                canonical.metadata["duplicate_attempts"] = (
                    int(canonical.metadata.get("duplicate_attempts", 0)) + 1
                )
                logger.info(
                    "Recorded duplicate source %s as lineage-only solution %s (canonical %s)",
                    source_hash[:12],
                    solution.solution_id,
                    canonical.solution_id,
                )
            if architecture_enabled:
                await memory._check_migration()
            return solution.solution_id

        solution_id = await original_add_solution(self, solution)
        if source_dedup_enabled:
            _deduplicate_memory_indexes(memory)
        if architecture_enabled:
            # The upstream positive-score path already called this hook. The
            # second call is a no-op, but covers zero-score and lineage-only rounds.
            await memory._check_migration()
        return solution_id

    def patched_load_checkpoint(self, checkpoint_path):
        result = original_load_checkpoint(self, checkpoint_path)
        memory = getattr(self._evolution_memory, "_memory", None)
        if memory is None:
            raise RuntimeError(
                "LoongFlow compatibility contract mismatch after checkpoint load: "
                "EvolveDatabase._evolution_memory._memory is unavailable"
            )
        population_restore = _restore_checkpoint_population_indexes(
            memory, checkpoint_path
        )
        logger.info(
            "Restored checkpoint population indexes: selectable=%d loaded=%d "
            "lineage_excluded=%d metadata=%s",
            population_restore.selectable_population_count,
            population_restore.loaded_population_count,
            population_restore.removed_lineage_count,
            population_restore.metadata_path,
        )
        migrated_scores = _migrate_checkpoint_selection_scores(memory)
        if migrated_scores:
            logger.info(
                "Migrated %d legacy checkpoint scores away from flat provisional caps",
                migrated_scores,
            )
        _reconcile_authoritative_scores(memory)
        if source_dedup_enabled:
            removed = _deduplicate_memory_indexes(memory)
            if removed:
                logger.info(
                    "Collapsed %d duplicate source variants before architecture restore",
                    removed,
                )
        if architecture_enabled:
            num_islands, migration_interval = architecture_settings(self)
            restored = restore_architecture_checkpoint(
                memory,
                checkpoint_path,
                num_islands,
                migration_interval,
            )
            logger.info(
                "Restored architecture checkpoint into %d islands at migration iteration %d",
                restored["islands"],
                restored["last_migration_iteration"],
            )
        elif stagnation_seeds_enabled:
            restore_stagnation_checkpoint(memory, checkpoint_path)
        if architecture_enabled or stagnation_seeds_enabled:
            stagnation = getattr(memory, "_atrex_stagnation_checkpoint_status", {})
            if isinstance(stagnation, dict):
                logger.info(
                    "Restored stagnation checkpoint: persisted=%s legacy=%s "
                    "bucket=%s retries=%d pending=%s",
                    stagnation.get("restored", False),
                    stagnation.get("legacy_reconstructed", False),
                    stagnation.get("seed_bucket"),
                    int(stagnation.get("seed_retries", 0) or 0),
                    stagnation.get("pending_escape", False),
                )
        if source_dedup_enabled:
            _deduplicate_memory_indexes(memory)
        return result

    async def patched_save_checkpoint(self, checkpoint_path, tag):
        result = await original_save_checkpoint(self, checkpoint_path, tag)
        memory = getattr(self._evolution_memory, "_memory", None)
        if (architecture_enabled or stagnation_seeds_enabled) and memory is None:
            raise RuntimeError(
                "LoongFlow compatibility contract mismatch after checkpoint save: "
                "EvolveDatabase._evolution_memory._memory is unavailable"
            )
        if (architecture_enabled or stagnation_seeds_enabled) and memory is not None:
            if not write_architecture_checkpoint(memory, checkpoint_path, tag):
                logger.warning(
                    "Atrex checkpoint metadata was not written for tag %s",
                    tag,
                )
        return result

    EvolveDatabase.__init__ = patched_init
    EvolveDatabase.sample_solution = patched_sample_solution
    EvolveDatabase.add_solution = patched_add_solution
    EvolveDatabase.load_checkpoint = patched_load_checkpoint
    EvolveDatabase.save_checkpoint = patched_save_checkpoint
    patched_sample_solution._atrex_stagnation_seed_patched = True
    EvolveDatabase._atrex_evolution_patched = True


def _wrap_database_func(func):
    if func is None or getattr(func, "_atrex_compacted", False):
        return func

    def compacted_func(*args, **kwargs):
        return _compact_result(func(*args, **kwargs))

    compacted_func._atrex_compacted = True
    return compacted_func


def _patch_database_tools() -> None:
    if not _enabled("ATREX_PES_COMPACT_DB_TOOLS"):
        return
    try:
        from loongflow.framework.pes.database import database_tool
    except Exception as exc:
        raise RuntimeError("cannot import LoongFlow database tool targets") from exc

    for class_name in (
        "GetSolutionsTool",
        "GetBestSolutionsTool",
        "GetParentsByChildIdTool",
        "GetChildsByParentTool",
    ):
        tool_cls = getattr(database_tool, class_name, None)
        if tool_cls is None or getattr(tool_cls, "_atrex_patched", False):
            continue
        original_init = tool_cls.__init__
        _require_signature(original_init, ("self", "func"))

        def patched_init(self, func=None, _original_init=original_init):
            _original_init(self, _wrap_database_func(func))
            self.description += (
                " Atrex compatibility: solution, evaluation, and summary fields "
                "are compacted to keep PES ReAct memory below the context limit."
            )

        tool_cls.__init__ = patched_init
        tool_cls._atrex_patched = True


def _patch_planner_write_tool() -> None:
    try:
        from agents.math_agent.planner import build_tool
        from loongflow.agentsdk.tools import FunctionTool
        from loongflow.framework.pes.context import Workspace
    except Exception as exc:
        raise RuntimeError("cannot import LoongFlow planner Write targets") from exc

    if getattr(build_tool, "_atrex_write_patched", False):
        return

    _require_signature(build_tool.build_planner_write_tool, ("context",))

    def build_planner_write_tool(context):
        async def write_func(file_path: str, content: str):
            planner_base = Path(Workspace.get_planner_path(context))
            planner_base.mkdir(parents=True, exist_ok=True)

            requested = Path(file_path)
            target = (
                requested if requested.is_absolute() else planner_base / requested.name
            )
            if not str(target).startswith(str(planner_base)):
                target = planner_base / target.name

            aliases = {
                "plan1.txt": ("plan1.txt", "plan_1.txt"),
                "plan_1.txt": ("plan1.txt", "plan_1.txt"),
                "plan2.txt": ("plan2.txt", "plan_2.txt"),
                "plan_2.txt": ("plan2.txt", "plan_2.txt"),
                "plan3.txt": ("plan3.txt", "plan_3.txt"),
                "plan_3.txt": ("plan3.txt", "plan_3.txt"),
            }
            names = aliases.get(target.name, (target.name,))
            for name in names:
                out = planner_base / name
                out.parent.mkdir(parents=True, exist_ok=True)
                out.write_text(content, encoding="utf-8")
            return "File written successfully"

        return FunctionTool(
            func=write_func,
            args_schema=build_tool.WriteToolArgs,
            name="Write",
            description=(
                "Writes a file to the planner workspace. Supports plan1.txt and "
                "plan_1.txt naming variants."
            ),
        )

    build_tool.build_planner_write_tool = build_planner_write_tool
    build_tool._atrex_write_patched = True


def _patch_planner_empty_plan_fallback() -> None:
    try:
        from agents.math_agent.planner.plan_agent import EvolvePlanAgent
        from loongflow.agentsdk.message import ContentElement
    except Exception as exc:
        raise RuntimeError("cannot import LoongFlow planner fallback targets") from exc

    if getattr(EvolvePlanAgent, "_atrex_empty_plan_patched", False):
        return

    original_run = EvolvePlanAgent.run
    _require_signature(original_run, ("self", "context", "message"))

    async def patched_run(self, context, message):
        result = await original_run(self, context, message)
        try:
            elements = result.get_elements(ContentElement)
            data = elements[0].data if elements else {}
            best_plan_path = Path(data.get("best_plan_file_path", ""))
            if (
                best_plan_path.exists()
                and best_plan_path.read_text(encoding="utf-8").strip()
            ):
                return result

            planner_dir = best_plan_path.parent
            fallback = ""
            for name in (
                "plan_3.txt",
                "plan3.txt",
                "plan_2.txt",
                "plan2.txt",
                "plan_1.txt",
                "plan1.txt",
            ):
                candidate = planner_dir / name
                if candidate.exists():
                    text = candidate.read_text(encoding="utf-8").strip()
                    if text:
                        fallback = text
                        break
            if not fallback:
                fallback = (
                    "### Final Child Solution Generation Plan\n\n"
                    "**Objective:** Improve the sampled parent while preserving correctness.\n\n"
                    "**Best Plan:**\n"
                    "1. Keep the parent solution as the implementation substrate; do not rewrite unrelated kernels.\n"
                    "2. Use the parent summary and evaluation metrics to identify the slowest workload groups.\n"
                    "3. Make one localized CUDA dispatch or kernel change that targets those workloads only.\n"
                    "4. Preserve known fast paths and graph-cache behavior unless the evaluation proves a change is faster.\n"
                    "5. Return a complete candidate source and rely on the evaluator as the only promotion gate.\n"
                )
            best_plan_path.write_text(fallback, encoding="utf-8")
        except Exception as exc:
            logger.warning("Could not materialize planner fallback plan: %s", exc)
        return result

    EvolvePlanAgent.run = patched_run
    EvolvePlanAgent._atrex_empty_plan_patched = True


def _patch_fuse_executor_parallel_cap() -> None:
    cap_raw = os.environ.get("ATREX_PES_MAX_PARALLEL_CANDIDATES", "")
    if not cap_raw:
        return
    try:
        cap = max(1, int(cap_raw))
    except ValueError as exc:
        raise ValueError(
            "ATREX_PES_MAX_PARALLEL_CANDIDATES must be an integer"
        ) from exc

    try:
        from agents.math_agent.executor.execute_fuse.execute_agent_fuse import (
            EvolveExecuteAgentFuse,
        )
    except Exception as exc:
        raise RuntimeError("cannot import LoongFlow fuse executor target") from exc

    if getattr(EvolveExecuteAgentFuse, "_atrex_parallel_cap_patched", False):
        return

    original_gen_multi_candidate = EvolveExecuteAgentFuse.gen_multi_candidate
    _require_signature(
        original_gen_multi_candidate,
        (
            "self",
            "context",
            "parent_ctx",
            "round_idx",
            "parallel_candidates",
            "previous_attempts",
        ),
    )

    async def patched_gen_multi_candidate(
        self,
        context,
        parent_ctx,
        round_idx,
        parallel_candidates,
        previous_attempts,
    ):
        return await original_gen_multi_candidate(
            self,
            context,
            parent_ctx,
            round_idx,
            min(parallel_candidates, cap),
            previous_attempts,
        )

    EvolveExecuteAgentFuse.gen_multi_candidate = patched_gen_multi_candidate
    EvolveExecuteAgentFuse._atrex_parallel_cap_patched = True


def _verify_evolution_patch() -> tuple[bool, str]:
    if not (
        _enabled("ATREX_PES_SOURCE_DEDUP")
        or _enabled("ATREX_PES_ARCHITECTURE_ISLANDS")
        or _enabled("ATREX_PES_STAGNATION_SEEDS", "0")
    ):
        return True, "disabled"
    from loongflow.framework.pes.database.database import EvolveDatabase

    if not getattr(EvolveDatabase, "_atrex_evolution_patched", False):
        return False, "EvolveDatabase sentinel missing"
    if _enabled("ATREX_PES_ARCHITECTURE_ISLANDS"):
        from loongflow.agentsdk.memory.evolution.in_memory import InMemory

        if not getattr(
            EvolveDatabase.__init__, "_atrex_island_config_validated", False
        ):
            return False, "EvolveDatabase config validation sentinel missing"
        if not getattr(InMemory, "_atrex_architecture_patched", False):
            return False, "InMemory architecture sentinel missing"
        if not getattr(
            InMemory._enforce_population_limit,
            "_atrex_architecture_retention",
            False,
        ):
            return False, "InMemory architecture retention sentinel missing"
    if _enabled("ATREX_PES_STAGNATION_SEEDS", "0") and not getattr(
        EvolveDatabase.sample_solution, "_atrex_stagnation_seed_patched", False
    ):
        return False, "stagnation seed sampler sentinel missing"
    return True, "EvolveDatabase/InMemory"


def _verify_database_tools_patch() -> tuple[bool, str]:
    if not _enabled("ATREX_PES_COMPACT_DB_TOOLS"):
        return True, "disabled"
    from loongflow.framework.pes.database import database_tool

    names = (
        "GetSolutionsTool",
        "GetBestSolutionsTool",
        "GetParentsByChildIdTool",
        "GetChildsByParentTool",
    )
    missing = [
        name
        for name in names
        if not getattr(getattr(database_tool, name, object), "_atrex_patched", False)
    ]
    return (not missing, "database tools" if not missing else f"missing {missing}")


def _verify_planner_write_patch() -> tuple[bool, str]:
    from agents.math_agent.planner import build_tool

    applied = bool(getattr(build_tool, "_atrex_write_patched", False))
    return applied, (
        "planner Write tool" if applied else "planner Write sentinel missing"
    )


def _verify_empty_plan_patch() -> tuple[bool, str]:
    from agents.math_agent.planner.plan_agent import EvolvePlanAgent

    applied = bool(getattr(EvolvePlanAgent, "_atrex_empty_plan_patched", False))
    return applied, (
        "planner fallback" if applied else "planner fallback sentinel missing"
    )


def _verify_fuse_patch() -> tuple[bool, str]:
    if not os.environ.get("ATREX_PES_MAX_PARALLEL_CANDIDATES", ""):
        return True, "disabled"
    from agents.math_agent.executor.execute_fuse.execute_agent_fuse import (
        EvolveExecuteAgentFuse,
    )

    applied = bool(
        getattr(EvolveExecuteAgentFuse, "_atrex_parallel_cap_patched", False)
    )
    return applied, "fuse parallel cap" if applied else "fuse cap sentinel missing"


def _apply_manifest_patch(
    name: str,
    patcher: Callable[[], Any],
    verifier: Callable[[], tuple[bool, str]],
    *,
    required: bool,
) -> None:
    error = ""
    try:
        patcher()
        applied, target = verifier()
    except Exception as exc:
        applied, target = False, ""
        error = f"{type(exc).__name__}: {exc}"
    PATCH_MANIFEST[name] = {
        "required": bool(required),
        "applied": bool(applied),
        "target": target,
        "error": error,
    }


def apply_compat_patches() -> None:
    PATCH_MANIFEST.clear()
    _apply_manifest_patch(
        "litellm_drop_params",
        _patch_litellm_drop_params,
        _verify_litellm_drop_params,
        required=_enabled("ATREX_LITELLM_DROP_PARAMS", "0"),
    )
    _apply_manifest_patch(
        "evolution_database",
        _patch_evolution_database_selection,
        _verify_evolution_patch,
        required=(
            _enabled("ATREX_PES_SOURCE_DEDUP")
            or _enabled("ATREX_PES_ARCHITECTURE_ISLANDS")
            or _enabled("ATREX_PES_STAGNATION_SEEDS", "0")
        ),
    )
    _apply_manifest_patch(
        "stagnation_seed_bank",
        lambda: None,
        _verify_stagnation_seed_bank,
        required=_enabled("ATREX_PES_STAGNATION_SEEDS", "0"),
    )
    _apply_manifest_patch(
        "database_tools",
        _patch_database_tools,
        _verify_database_tools_patch,
        required=_enabled("ATREX_PES_COMPACT_DB_TOOLS"),
    )
    _apply_manifest_patch(
        "planner_write",
        _patch_planner_write_tool,
        _verify_planner_write_patch,
        required=True,
    )
    _apply_manifest_patch(
        "planner_empty_plan",
        _patch_planner_empty_plan_fallback,
        _verify_empty_plan_patch,
        required=True,
    )
    _apply_manifest_patch(
        "executor_parallel_cap",
        _patch_fuse_executor_parallel_cap,
        _verify_fuse_patch,
        required=bool(os.environ.get("ATREX_PES_MAX_PARALLEL_CANDIDATES", "")),
    )
    _apply_manifest_patch(
        "summary_ncu",
        _patch_summary_ncu_interpretation,
        _verify_ncu_patch,
        required=_enabled("SOL58_NCU_SUMMARY", "0"),
    )


def validate_patch_manifest() -> dict[str, dict[str, object]]:
    failures = {
        name: entry
        for name, entry in PATCH_MANIFEST.items()
        if entry.get("required") and not entry.get("applied")
    }
    if failures:
        details = "; ".join(
            f"{name}: {entry.get('error') or entry.get('target')}"
            for name, entry in failures.items()
        )
        raise RuntimeError(f"required Atrex LoongFlow patches are inactive: {details}")
    return json.loads(json.dumps(PATCH_MANIFEST))
