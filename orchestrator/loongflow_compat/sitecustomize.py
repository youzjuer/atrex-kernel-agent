"""Process-wide compatibility hooks for local LoongFlow runners."""

from __future__ import annotations

import hashlib
import inspect
import json
import logging
import os
from pathlib import Path

from orchestrator.loongflow_compat.architecture_islands import (
    architecture_island_id,
    architecture_label,
    architecture_tags,
    classify_and_route_solution,
    extract_architecture_features,
    initialize_architecture_memory,
    island_profile,
    maybe_exchange_islands,
    restore_architecture_checkpoint,
    validate_architecture_island_count,
    write_architecture_checkpoint,
)
from orchestrator.loongflow_compat.sol58_seed_bank import (
    load_seed_bank,
    seed_bank_fingerprint,
)

logger = logging.getLogger("atrex.pes_compat")
PATCH_MANIFEST: dict[str, dict[str, object]] = {}

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


if os.environ.get("ATREX_LITELLM_DROP_PARAMS", "0") == "1":
    try:
        import litellm

        litellm.drop_params = True
    except Exception:
        pass


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
        except Exception:
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


def _append_ncu_summary_instructions(prompt: str) -> str:
    if _NCU_SUMMARY_MARKER in prompt:
        return prompt
    return prompt.rstrip() + _NCU_SUMMARY_INSTRUCTIONS


def _patch_summary_ncu_interpretation() -> None:
    if os.environ.get("SOL58_NCU_SUMMARY", "0").strip().lower() not in {
        "1",
        "true",
        "yes",
        "on",
    }:
        return
    try:
        from agents.math_agent.summary import summary_agent
    except Exception:
        return

    prompt = getattr(summary_agent, "EVOLVE_SUMMARY_USER_PROMPT", "")
    if not isinstance(prompt, str):
        return
    summary_agent.EVOLVE_SUMMARY_USER_PROMPT = _append_ncu_summary_instructions(prompt)


def _solution_source_hash(solution: object) -> str:
    source = getattr(solution, "solution", solution)
    text = source if isinstance(source, str) else str(source or "")
    return hashlib.sha256(text.strip().encode("utf-8")).hexdigest()


def _adaptive_exploration_rate(base_rate: float, recent_scores: list[float]) -> float:
    """Raise exploration only after five real attempts, checking hard stagnation first."""
    rate = max(0.0, float(base_rate))
    if len(recent_scores) >= 5:
        scores = recent_scores[:5]
        deltas = [abs(scores[i] - scores[i + 1]) for i in range(4)]
        if all(delta < 0.001 for delta in deltas):
            rate *= 4
        elif all(delta < 0.01 for delta in deltas):
            rate *= 2
    return min(rate, 0.9)


def _positive_int_env(name: str, default: int) -> int:
    try:
        value = int(os.environ.get(name, str(default)))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a positive integer") from exc
    if value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _stagnation_state(memory: object) -> dict[str, float | int]:
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
    }


def _stagnation_seed_parent(
    memory: object,
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
    plateau_rounds = int(state["plateau_rounds"])
    if plateau_rounds < threshold:
        return None
    bucket = (plateau_rounds - threshold) // interval
    if getattr(memory, "_atrex_stagnation_seed_bucket", None) == bucket:
        return None

    seeds = load_seed_bank(language="cuda_cpp")
    seed = seeds[bucket % len(seeds)]
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
        "self-contained CUDA architecture experiment. Do not fall back to a "
        "constant-only or launch-bound-only edit of the incumbent. Preserve "
        "stable ordering and all 16-workload correctness."
    )
    memory._atrex_stagnation_seed_bucket = bucket
    memory._atrex_last_stagnation_seed_id = seed.seed_id
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


def _canonical_solution(solutions: list[object]) -> object:
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
    except Exception:
        return {}
    sources = payload.get("sources") if isinstance(payload, dict) else None
    return sources if isinstance(sources, dict) else {}


def _reconcile_authoritative_scores(memory: object) -> int:
    """Replace stale provisional scores with completed official source fitness."""
    registry = _load_authoritative_fitness_registry()
    solutions = getattr(memory, "solutions", None)
    populations = getattr(memory, "populations", None)
    lock = getattr(memory, "_lock", None)
    if not registry or not isinstance(solutions, dict) or lock is None:
        return 0

    changed_ids: set[str] = set()
    with lock:
        records: dict[str, object] = dict(solutions)
        if isinstance(populations, dict):
            records.update(populations)

        for solution_id, solution in records.items():
            source_hash = _solution_source_hash(solution)
            authoritative = registry.get(source_hash)
            if not isinstance(authoritative, dict):
                continue
            try:
                official_score = float(authoritative.get("official_score") or 0.0)
            except (TypeError, ValueError):
                continue
            if official_score <= 0:
                continue

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


def _deduplicate_memory_indexes(memory: object) -> int:
    """Keep one selectable representative per source and island; retain lineage records."""
    populations = getattr(memory, "populations", None)
    islands = getattr(memory, "islands", None)
    if not isinstance(populations, dict) or not isinstance(islands, list):
        return 0

    duplicate_to_canonical: dict[str, str] = {}
    lock = getattr(memory, "_lock", None)
    if lock is None:
        return 0

    with lock:
        assigned_ids = set().union(*islands) if islands else set()
        for solution_id in list(populations):
            if solution_id not in assigned_ids:
                populations.pop(solution_id, None)

        for island in islands:
            groups: dict[str, list[object]] = {}
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
            elite_groups: dict[str, list[object]] = {}
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


def _require_signature(callable_obj: object, expected: tuple[str, ...]) -> None:
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
    except Exception:
        return

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

        InMemory._prepare_solution = patched_prepare_solution
        InMemory._check_migration = patched_check_migration
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
            return {}

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
        forced_seed = _stagnation_seed_parent(
            memory,
            requested_island=island_id,
            num_islands=configured_islands,
        )
        if forced_seed is not None:
            return forced_seed
        recent = memory.list_solutions(filter_type="desc", limit=5)
        recent_scores = [
            float(solution.score)
            for solution in recent
            if getattr(solution, "score", None) is not None
        ]
        exploration_rate = _adaptive_exploration_rate(
            self.config.exploration_rate,
            recent_scores,
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
                "Adaptive exploration rate %.3f -> %.3f from recent scores",
                self.config.exploration_rate,
                exploration_rate,
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
            return await original_add_solution(self, solution)

        if architecture_enabled:
            num_islands, migration_interval = architecture_settings(self)
            initialize_architecture_memory(memory, num_islands, migration_interval)
            if isinstance(source, str) and source.strip():
                analysis = classify_and_route_solution(memory, solution, num_islands)
                logger.info(
                    "Summary architecture route: label=%s pca_cluster=%d island=%d",
                    analysis["label"],
                    analysis["cluster"],
                    analysis["island_id"],
                )

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
            return result
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
        if source_dedup_enabled:
            _deduplicate_memory_indexes(memory)
        return result

    async def patched_save_checkpoint(self, checkpoint_path, tag):
        result = await original_save_checkpoint(self, checkpoint_path, tag)
        memory = getattr(self._evolution_memory, "_memory", None)
        if architecture_enabled and memory is not None:
            if not write_architecture_checkpoint(memory, checkpoint_path, tag):
                logger.warning(
                    "Architecture checkpoint metadata was not written for tag %s",
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
    except Exception:
        return

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
    except Exception:
        return

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
    except Exception:
        return

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
        except Exception:
            pass
        return result

    EvolvePlanAgent.run = patched_run
    EvolvePlanAgent._atrex_empty_plan_patched = True


def _patch_fuse_executor_parallel_cap() -> None:
    cap_raw = os.environ.get("ATREX_PES_MAX_PARALLEL_CANDIDATES", "")
    if not cap_raw:
        return
    try:
        cap = max(1, int(cap_raw))
    except ValueError:
        return

    try:
        from agents.math_agent.executor.execute_fuse.execute_agent_fuse import (
            EvolveExecuteAgentFuse,
        )
    except Exception:
        return

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


def _enabled(name: str, default: str = "1") -> bool:
    return os.environ.get(name, default).strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


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
    if _enabled("ATREX_PES_STAGNATION_SEEDS", "0") and not getattr(
        EvolveDatabase.sample_solution, "_atrex_stagnation_seed_patched", False
    ):
        return False, "stagnation seed sampler sentinel missing"
    return True, "EvolveDatabase/InMemory"


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


def _verify_ncu_patch() -> tuple[bool, str]:
    if not _enabled("SOL58_NCU_SUMMARY", "0"):
        return True, "disabled"
    from agents.math_agent.summary import summary_agent

    prompt = getattr(summary_agent, "EVOLVE_SUMMARY_USER_PROMPT", "")
    applied = isinstance(prompt, str) and _NCU_SUMMARY_MARKER in prompt
    return applied, "summary NCU prompt" if applied else "summary NCU marker missing"


def _apply_manifest_patch(
    name: str,
    patcher: object,
    verifier: object,
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


apply_compat_patches()
