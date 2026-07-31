from __future__ import annotations

import json
import math
import os
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from orchestrator.loongflow_compat import architecture_islands
from orchestrator.loongflow_compat import compat_adapter
from orchestrator.loongflow_compat.checkpoint_compat import (
    CheckpointCompatibilityError,
)
from orchestrator.loongflow_compat.sol58_seed_bank import (
    load_seed_bank,
    seed_bank_fingerprint,
)


def _solution(
    solution_id: str,
    source: str,
    *,
    iteration: int,
    score: float,
    weight: float,
):
    return SimpleNamespace(
        solution_id=solution_id,
        solution=source,
        parent_id="",
        island_id=0,
        generation=0,
        iteration=iteration,
        timestamp=float(iteration),
        score=score,
        sample_weight=weight,
        metadata={},
    )


def _memory(*solutions, last_iteration: int = 0):
    populations = {solution.solution_id: solution for solution in solutions}
    return SimpleNamespace(
        _lock=threading.RLock(),
        _island_locks={0: threading.RLock()},
        populations=dict(populations),
        solutions=dict(populations),
        islands=[set(populations)],
        island_feature_maps=[{}],
        island_best_solution=[next(iter(populations), None)],
        island_capacity=[len(populations)],
        elites=set(populations),
        best_solution_id=next(iter(populations), ""),
        population_size=100,
        migration_rate=0.2,
        migration_interval=20,
        num_islands=1,
        current_island=0,
        solutions_per_island=100,
        last_iteration=last_iteration,
        last_migration_generation=0,
    )


class TestAdaptiveExploration(unittest.TestCase):
    def test_requires_five_attempts(self) -> None:
        self.assertEqual(
            compat_adapter._adaptive_exploration_rate(0.1, [1.0, 1.0, 1.0, 1.0]),
            0.1,
        )

    def test_hard_stagnation_is_checked_before_medium(self) -> None:
        self.assertEqual(
            compat_adapter._adaptive_exploration_rate(0.1, [0.8] * 5),
            0.4,
        )
        self.assertEqual(
            compat_adapter._adaptive_exploration_rate(
                0.1,
                [0.80, 0.805, 0.81, 0.815, 0.82],
            ),
            0.2,
        )

    def test_exploration_is_capped(self) -> None:
        self.assertEqual(
            compat_adapter._adaptive_exploration_rate(0.3, [0.8] * 5),
            0.9,
        )

    def test_global_plateau_and_empty_islands_do_not_require_five_scores(self) -> None:
        self.assertEqual(
            compat_adapter._adaptive_exploration_rate(
                0.2,
                [0.8, 0.7],
                plateau_rounds=24,
                stagnation_rounds=12,
            ),
            0.8,
        )
        self.assertEqual(
            compat_adapter._adaptive_exploration_rate(
                0.2,
                [],
                empty_home_islands=4,
                num_islands=8,
            ),
            0.8,
        )


class TestStagnationArchitectureEscape(unittest.TestCase):
    @staticmethod
    def _set_pending_escape(
        memory,
        *,
        incumbent_family: str = "hierarchical_histogram",
        expected_iteration: int = 12,
        bucket: int = 0,
    ) -> None:
        memory._atrex_stagnation_seed_bucket = bucket
        memory._atrex_stagnation_retry_bucket = bucket
        memory._atrex_stagnation_seed_retries = 0
        memory._atrex_pending_stagnation_escape = {
            "expected_iteration": expected_iteration,
            "bucket": bucket,
            "seed_id": "cub_device_radix_sort",
            "seed_family": "cub_radix_sort",
            "incumbent_family": incumbent_family,
        }

    def test_plateau_age_uses_first_best_iteration(self) -> None:
        first_best = _solution("best", "a", iteration=4, score=0.9, weight=1)
        equal_later = _solution("tie", "b", iteration=11, score=0.9, weight=1)
        weaker = _solution("weak", "c", iteration=18, score=0.8, weight=1)
        state = compat_adapter._stagnation_state(
            _memory(first_best, equal_later, weaker, last_iteration=20)
        )

        self.assertEqual(state["best_iteration"], 4)
        self.assertEqual(state["plateau_rounds"], 16)
        self.assertEqual(state["best_score"], 0.9)

    def test_stagnation_seeds_are_deterministic_and_once_per_bucket(self) -> None:
        best = _solution("best", "kernel", iteration=1, score=0.9, weight=1)
        memory = _memory(best, last_iteration=17)
        environment = {
            "ATREX_PES_STAGNATION_SEEDS": "1",
            "ATREX_PES_STAGNATION_ARCHITECTURE_ROUNDS": "12",
            "ATREX_PES_STAGNATION_SEED_INTERVAL": "4",
            "SOL58_CODE_LANGUAGE": "cuda_cpp",
        }
        with mock.patch.dict(os.environ, environment):
            first = compat_adapter._stagnation_seed_parent(
                memory, requested_island=0, num_islands=8
            )
            repeated = compat_adapter._stagnation_seed_parent(
                memory, requested_island=0, num_islands=8
            )
            memory.last_iteration = 21
            second = compat_adapter._stagnation_seed_parent(
                memory, requested_island=0, num_islands=8
            )

        self.assertEqual(first["solution_id"], "")
        self.assertEqual(
            first["metadata"]["architecture_label"], "expert_parallel_scan"
        )
        self.assertEqual(first["island_id"], 7)
        self.assertTrue(first["metadata"]["stagnation_escape"]["required"])
        self.assertEqual(
            first["metadata"]["stagnation_escape"]["incumbent_family"],
            "baseline",
        )
        self.assertIn("incumbent family (baseline)", first["generate_plan"])
        self.assertIsNone(repeated)
        self.assertEqual(
            second["metadata"]["architecture_label"], "warp_specialization"
        )
        self.assertEqual(second["island_id"], 0)

    def test_unimproved_incumbent_child_is_zeroed_and_retried(self) -> None:
        memory = _memory(last_iteration=11)
        self._set_pending_escape(memory)
        memory._prepare_solution = lambda solution: setattr(
            memory,
            "last_iteration",
            max(memory.last_iteration, int(solution.iteration)),
        )
        child = _solution(
            "child", "hierarchical histogram", iteration=12, score=0.88, weight=1
        )
        child.evaluation = json.dumps(
            {
                "score": 0.88,
                "metrics": {"local_best": {"strictly_improved": False}},
            }
        )

        with mock.patch.dict(os.environ, {"ATREX_PES_STAGNATION_MAX_ATTEMPTS": "2"}):
            accepted = compat_adapter._apply_stagnation_architecture_gate(
                memory, child, "hierarchical_histogram"
            )

        evaluation = json.loads(child.evaluation)
        self.assertFalse(accepted)
        self.assertEqual(child.score, 0.0)
        self.assertEqual(evaluation["score"], 0.0)
        self.assertIn("architecture_escape_gate", evaluation["metrics"])
        self.assertIsNone(memory._atrex_stagnation_seed_bucket)
        self.assertEqual(memory._atrex_stagnation_seed_retries, 1)
        self.assertEqual(
            child.metadata["stagnation_escape_violation"]["seed_family"],
            "cub_radix_sort",
        )
        solution_id = compat_adapter._finalize_rejected_stagnation_child(memory, child)
        self.assertEqual(solution_id, "child")
        self.assertEqual(memory.last_iteration, 12)
        self.assertNotIn("child", memory.solutions)
        self.assertNotIn("child", memory.populations)
        self.assertEqual(
            child.metadata["database_admission"], "rejected_stagnation_escape"
        )

    def test_different_family_child_satisfies_escape(self) -> None:
        memory = _memory(last_iteration=11)
        self._set_pending_escape(memory)
        child = _solution("child", "radix", iteration=12, score=0.7, weight=1)
        child.evaluation = {
            "status": "success",
            "metrics": {
                "passed": 16,
                "total": 16,
                "expected_total": 16,
                "local_best": {"strictly_improved": False},
            },
        }

        accepted = compat_adapter._apply_stagnation_architecture_gate(
            memory, child, "cub_radix_sort"
        )

        self.assertTrue(accepted)
        self.assertEqual(child.score, 0.7)
        self.assertEqual(memory._atrex_stagnation_seed_retries, 0)
        self.assertEqual(
            child.metadata["stagnation_escape_satisfied"]["child_family"],
            "cub_radix_sort",
        )

    def test_improved_incumbent_child_satisfies_escape(self) -> None:
        memory = _memory(last_iteration=11)
        self._set_pending_escape(memory)
        child = _solution(
            "child", "hierarchical histogram", iteration=12, score=0.91, weight=1
        )
        child.evaluation = json.dumps(
            {
                "status": "success",
                "score": 0.91,
                "metrics": {
                    "passed": 16,
                    "total": 16,
                    "expected_total": 16,
                    "local_best": {"strictly_improved": True},
                },
            }
        )

        accepted = compat_adapter._apply_stagnation_architecture_gate(
            memory, child, "hierarchical_histogram"
        )

        self.assertTrue(accepted)
        self.assertEqual(child.score, 0.91)
        self.assertTrue(
            child.metadata["stagnation_escape_satisfied"]["local_best_improved"]
        )

    def test_failed_alternate_family_is_rejected_and_retried(self) -> None:
        memory = _memory(last_iteration=11)
        self._set_pending_escape(memory)
        child = _solution("child", "radix", iteration=12, score=0.7, weight=1)
        child.evaluation = {
            "status": "error",
            "metrics": {"passed": 0, "total": 16, "expected_total": 16},
        }

        with mock.patch.dict(os.environ, {"ATREX_PES_STAGNATION_MAX_ATTEMPTS": "5"}):
            accepted = compat_adapter._apply_stagnation_architecture_gate(
                memory, child, "cub_radix_sort"
            )

        self.assertFalse(accepted)
        self.assertEqual(child.score, 0.0)
        violation = child.metadata["stagnation_escape_violation"]
        self.assertFalse(violation["all_workloads_correct"])
        self.assertIn("all workloads correct", violation["reason"])
        self.assertIsNone(memory._atrex_stagnation_seed_bucket)

    def test_retry_budget_resets_for_each_stagnation_bucket(self) -> None:
        best = _solution("best", "kernel", iteration=1, score=0.9, weight=1)
        memory = _memory(best, last_iteration=13)
        memory._atrex_stagnation_best_marker = (1, 0.9)
        memory._atrex_stagnation_retry_bucket = 0
        memory._atrex_stagnation_seed_bucket = 0
        memory._atrex_stagnation_seed_retries = 2
        environment = {
            "ATREX_PES_STAGNATION_SEEDS": "1",
            "ATREX_PES_STAGNATION_ARCHITECTURE_ROUNDS": "12",
            "ATREX_PES_STAGNATION_SEED_INTERVAL": "4",
            "SOL58_CODE_LANGUAGE": "cuda_cpp",
        }

        memory.last_iteration = 17
        with mock.patch.dict(os.environ, environment):
            seed = compat_adapter._stagnation_seed_parent(
                memory, requested_island=0, num_islands=8
            )

        self.assertIsNotNone(seed)
        self.assertEqual(memory._atrex_stagnation_retry_bucket, 1)
        self.assertEqual(memory._atrex_stagnation_seed_retries, 0)

    def test_seed_bank_is_complete_and_fingerprinted(self) -> None:
        seeds = load_seed_bank()
        self.assertEqual(
            {seed.family for seed in seeds},
            {
                "cluster_dsmem",
                "cub_radix_sort",
                "expert_parallel_scan",
                "persistent_cooperative",
                "warp_specialization",
            },
        )
        self.assertEqual(len(seed_bank_fingerprint(seeds)), 64)

    def test_new_best_resets_forced_seed_buckets(self) -> None:
        old_best = _solution("old", "kernel-a", iteration=1, score=0.9, weight=1)
        memory = _memory(old_best, last_iteration=13)
        environment = {
            "ATREX_PES_STAGNATION_SEEDS": "1",
            "ATREX_PES_STAGNATION_ARCHITECTURE_ROUNDS": "12",
            "ATREX_PES_STAGNATION_SEED_INTERVAL": "4",
            "SOL58_CODE_LANGUAGE": "cuda_cpp",
        }
        with mock.patch.dict(os.environ, environment):
            self.assertIsNotNone(
                compat_adapter._stagnation_seed_parent(
                    memory, requested_island=0, num_islands=8
                )
            )
            new_best = _solution("new", "kernel-b", iteration=14, score=0.91, weight=1)
            memory.populations[new_best.solution_id] = new_best
            memory.last_iteration = 26
            reset = compat_adapter._stagnation_seed_parent(
                memory, requested_island=0, num_islands=8
            )

        self.assertIsNotNone(reset)
        self.assertEqual(reset["metadata"]["stagnation_escape"]["best_iteration"], 14)


class TestNcuSummaryPrompt(unittest.TestCase):
    def test_ncu_interpretation_contract_is_added_once(self) -> None:
        prompt = compat_adapter._append_ncu_summary_instructions("Base summary prompt")
        repeated = compat_adapter._append_ncu_summary_instructions(prompt)

        self.assertEqual(prompt, repeated)
        self.assertEqual(prompt.count(compat_adapter._NCU_SUMMARY_MARKER), 1)
        self.assertIn("metrics.ncu_analysis", prompt)
        self.assertIn("must never change the fitness assessment", prompt)


class TestPatchManifest(unittest.TestCase):
    def test_truthy_patch_flags_use_one_parser(self) -> None:
        with mock.patch.dict(
            os.environ,
            {
                "ATREX_PES_SOURCE_DEDUP": "true",
                "ATREX_PES_ARCHITECTURE_ISLANDS": "YES",
                "ATREX_PES_COMPACT_DB_TOOLS": "On",
            },
        ):
            self.assertTrue(compat_adapter._enabled("ATREX_PES_SOURCE_DEDUP"))
            self.assertTrue(compat_adapter._enabled("ATREX_PES_ARCHITECTURE_ISLANDS"))
            self.assertTrue(compat_adapter._enabled("ATREX_PES_COMPACT_DB_TOOLS"))

    def test_database_constructor_rejects_too_few_architecture_islands(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least 8 islands"):
            compat_adapter._architecture_num_islands(SimpleNamespace(num_islands=4))
        self.assertEqual(
            compat_adapter._architecture_num_islands(SimpleNamespace(num_islands=8)),
            8,
        )

    def test_required_patch_failure_is_not_silent(self) -> None:
        broken = {
            "evolution_database": {
                "required": True,
                "applied": False,
                "target": "sentinel missing",
                "error": "",
            }
        }
        with mock.patch.object(compat_adapter, "PATCH_MANIFEST", broken):
            with self.assertRaisesRegex(RuntimeError, "evolution_database"):
                compat_adapter.validate_patch_manifest()

    def test_compaction_keeps_normal_kernel_source_complete(self) -> None:
        source = "x" * 12000
        with mock.patch.dict(os.environ, {"ATREX_PES_DB_SOLUTION_CHARS": "65536"}):
            compact = compat_adapter._compact_solution_record({"solution": source})

        self.assertEqual(compact["solution"], source)
        self.assertEqual(compact["solution_chars"], len(source))


class TestSourceDeduplication(unittest.TestCase):
    def test_hash_ignores_outer_whitespace(self) -> None:
        self.assertEqual(
            compat_adapter._solution_source_hash(" kernel\n"),
            compat_adapter._solution_source_hash("kernel"),
        )

    def test_population_indexes_keep_one_source_representative(self) -> None:
        canonical = _solution("a", "kernel", iteration=1, score=0.85, weight=0.05)
        duplicate = _solution("b", "kernel\n", iteration=2, score=0.85, weight=15.0)
        distinct = _solution("c", "other", iteration=3, score=0.70, weight=8.0)
        unassigned = _solution("d", "history", iteration=4, score=0.60, weight=2.0)
        memory = SimpleNamespace(
            _lock=threading.RLock(),
            populations={
                item.solution_id: item
                for item in (canonical, duplicate, distinct, unassigned)
            },
            solutions={
                item.solution_id: item
                for item in (canonical, duplicate, distinct, unassigned)
            },
            islands=[{"a", "b", "c"}],
            elites={"a", "b", "c"},
            island_feature_maps=[{"0-0-0": "a", "1-1-1": "b", "2-2-2": "c"}],
            best_solution_id="b",
            island_best_solution=["b"],
            island_capacity=[3],
        )

        removed = compat_adapter._deduplicate_memory_indexes(memory)

        self.assertEqual(removed, 1)
        self.assertEqual(set(memory.populations), {"a", "c"})
        self.assertEqual(memory.islands, [{"a", "c"}])
        self.assertEqual(memory.elites, {"a", "c"})
        self.assertEqual(memory.best_solution_id, "a")
        self.assertEqual(memory.island_best_solution, ["a"])
        self.assertEqual(memory.island_capacity, [2])
        self.assertIn("b", memory.solutions)
        self.assertEqual(duplicate.metadata["duplicate_of"], "a")
        self.assertAlmostEqual(canonical.sample_weight, 1.0 + 3.0 * 0.85)
        self.assertAlmostEqual(distinct.sample_weight, 1.0 + 3.0 * 0.70)
        mapped_ids = list(memory.island_feature_maps[0].values())
        self.assertEqual(mapped_ids.count("a"), 1)

    def test_checkpoint_restore_excludes_lineage_only_population_records(self) -> None:
        valid = _solution("valid", "kernel", iteration=1, score=0.8, weight=1)
        rejected = _solution("rejected", "bad", iteration=2, score=0.0, weight=1)
        duplicate = _solution("duplicate", "kernel", iteration=3, score=0.8, weight=1)
        memory = SimpleNamespace(
            _lock=threading.RLock(),
            populations={
                item.solution_id: item for item in (valid, rejected, duplicate)
            },
            solutions={item.solution_id: item for item in (valid, rejected, duplicate)},
            islands=[{"valid"}],
            elites={"valid", "rejected", "duplicate"},
            island_feature_maps=[
                {"0-0-0": "valid", "1-1-1": "rejected", "2-2-2": "duplicate"}
            ],
            best_solution_id="duplicate",
            island_best_solution=["duplicate"],
            island_capacity=[3],
        )

        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = Path(tmp)
            (checkpoint / "metadata.json").write_text(
                json.dumps({"islands": [["valid"]]}), encoding="utf-8"
            )
            report = compat_adapter._restore_checkpoint_population_indexes(
                memory, str(checkpoint)
            )

        self.assertEqual(report.removed_lineage_count, 2)
        self.assertEqual(report.selectable_population_count, 1)
        self.assertEqual(report.loaded_population_count, 3)
        self.assertEqual(set(memory.populations), {"valid"})
        self.assertEqual(set(memory.solutions), {"valid", "rejected", "duplicate"})
        self.assertEqual(memory.islands, [{"valid"}])
        self.assertEqual(memory.elites, {"valid"})
        self.assertEqual(memory.best_solution_id, "valid")
        self.assertEqual(memory.island_best_solution, ["valid"])
        self.assertEqual(memory.island_capacity, [1])
        self.assertEqual(memory.island_feature_maps, [{"0-0-0": "valid"}])

    def test_checkpoint_restore_rejects_malformed_metadata(self) -> None:
        memory = _memory(_solution("valid", "kernel", iteration=1, score=0.8, weight=1))
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = Path(tmp)
            (checkpoint / "metadata.json").write_text("{not-json", encoding="utf-8")
            with self.assertRaisesRegex(CheckpointCompatibilityError, "not valid JSON"):
                compat_adapter._restore_checkpoint_population_indexes(
                    memory, str(checkpoint)
                )

    def test_checkpoint_restore_rejects_memory_contract_drift(self) -> None:
        valid = _solution("valid", "kernel", iteration=1, score=0.8, weight=1)
        memory = _memory(valid)
        memory._lock = None
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = Path(tmp)
            (checkpoint / "metadata.json").write_text(
                json.dumps({"islands": [["valid"]]}), encoding="utf-8"
            )
            with self.assertRaisesRegex(
                CheckpointCompatibilityError, "_lock:context-manager"
            ):
                compat_adapter._restore_checkpoint_population_indexes(
                    memory, str(checkpoint)
                )


class TestAuthoritativeFitnessReconciliation(unittest.TestCase):
    def test_completed_official_scores_replace_provisional_scores_and_best(
        self,
    ) -> None:
        stale = _solution("stale", "kernel-a", iteration=1, score=0.899135, weight=10.0)
        official_best = _solution(
            "official-best", "kernel-b", iteration=2, score=0.873, weight=9.0
        )
        local_only = _solution("local", "kernel-c", iteration=3, score=0.86, weight=8.0)
        memory = SimpleNamespace(
            _lock=threading.RLock(),
            populations={
                item.solution_id: item for item in (stale, official_best, local_only)
            },
            solutions={
                item.solution_id: item for item in (stale, official_best, local_only)
            },
            islands=[{"stale", "official-best", "local"}],
            best_solution_id="stale",
            island_best_solution=["stale"],
        )

        registry = {
            "version": 1,
            "sources": {
                compat_adapter._solution_source_hash(stale): {
                    "official_score": 0.852202,
                    "submission_id": 25255,
                },
                compat_adapter._solution_source_hash(official_best): {
                    "official_score": 0.856272,
                    "submission_id": 25293,
                },
            },
        }
        with tempfile.TemporaryDirectory() as tmp:
            registry_path = Path(tmp) / "official_cache" / "authoritative_fitness.json"
            registry_path.parent.mkdir()
            registry_path.write_text(json.dumps(registry))
            with mock.patch.dict(os.environ, {"SOL58_EVAL_ROOT": tmp}):
                changed = compat_adapter._reconcile_authoritative_scores(memory)

        self.assertEqual(changed, 2)
        self.assertAlmostEqual(stale.score, 0.852202)
        self.assertAlmostEqual(official_best.score, 0.856272)
        self.assertEqual(stale.metadata["fitness_source"], "official")
        self.assertEqual(memory.best_solution_id, "local")
        self.assertEqual(memory.island_best_solution, ["local"])

    def test_terminal_official_failure_replaces_provisional_score_with_zero(
        self,
    ) -> None:
        failed = _solution(
            "failed", "kernel-a", iteration=1, score=0.899135, weight=10.0
        )
        local = _solution("local", "kernel-b", iteration=2, score=0.8, weight=2.0)
        memory = _memory(failed, local)
        registry = {
            "version": 1,
            "sources": {
                compat_adapter._solution_source_hash(failed): {
                    "official_score": 0.0,
                    "status": "COMPLETED",
                    "is_correct": False,
                    "submission_id": 25296,
                }
            },
        }
        with tempfile.TemporaryDirectory() as tmp:
            registry_path = Path(tmp) / "official_cache" / "authoritative_fitness.json"
            registry_path.parent.mkdir()
            registry_path.write_text(json.dumps(registry))
            with mock.patch.dict(os.environ, {"SOL58_EVAL_ROOT": tmp}):
                changed = compat_adapter._reconcile_authoritative_scores(memory)

        self.assertEqual(changed, 1)
        self.assertEqual(failed.score, 0.0)
        self.assertEqual(failed.metadata["official_status"], "COMPLETED")
        self.assertFalse(failed.metadata["official_is_correct"])
        self.assertEqual(memory.best_solution_id, "local")

    def test_legacy_flat_cap_is_reconstructed_from_anchor_latency_ratio(self) -> None:
        legacy = _solution(
            "legacy", "kernel-a", iteration=1, score=0.899135, weight=99.0
        )
        legacy.evaluation = json.dumps(
            {
                "status": "success",
                "score": 0.899135,
                "metrics": {
                    "passed": 16,
                    "total": 16,
                    "expected_total": 16,
                    "official": {
                        "authoritative": False,
                        "provisional_calibration": {
                            "anchor_score": 0.856301,
                            "candidate_to_anchor_latency_ratio": 1.0610521819558296,
                        },
                    },
                },
            }
        )
        memory = _memory(legacy)

        with mock.patch.dict(
            os.environ,
            {
                "SOL58_TARGET_SCORE": "0.904135",
                "SOL58_OFFICIAL_PROVISIONAL_SCORE_CAP": "0.899135",
            },
        ):
            changed = compat_adapter._migrate_checkpoint_selection_scores(memory)

        payload = json.loads(legacy.evaluation)
        metrics = payload["metrics"]
        expected_search = 0.856301 * 1.0610521819558296
        expected_selection = compat_adapter.project_provisional_score(
            expected_search,
            target=0.904135,
            floor=0.899135,
        )
        self.assertEqual(changed, 1)
        self.assertAlmostEqual(metrics["search_score"], expected_search)
        self.assertAlmostEqual(legacy.score, expected_selection)
        self.assertAlmostEqual(metrics["selection_score"], expected_selection)
        self.assertLessEqual(legacy.sample_weight, 1.0 + 3.0 * expected_selection)
        self.assertEqual(
            legacy.metadata["checkpoint_score_migration"]["source"],
            "legacy_anchor_latency_ratio",
        )

    def test_authoritative_checkpoint_score_is_not_migrated(self) -> None:
        official = _solution("official", "kernel", iteration=1, score=0.856, weight=2)
        official.evaluation = {
            "status": "success",
            "score": 0.856,
            "metrics": {
                "certified_score": 0.856,
                "selection_score": 0.856,
                "search_score": 0.856,
                "official": {"authoritative": True},
            },
        }
        memory = _memory(official)

        changed = compat_adapter._migrate_checkpoint_selection_scores(memory)

        self.assertEqual(changed, 0)
        self.assertEqual(official.score, 0.856)
        self.assertNotIn("checkpoint_score_migration", official.metadata)


class TestArchitectureFeatureAnalysis(unittest.TestCase):
    def test_detects_named_cuda_and_cute_routes(self) -> None:
        warp_source = r"""
        __global__ void staged() {
          __shared__ int tile[256];
          int warp_id = threadIdx.x / 32;
          bool producer_warp = warp_id == 0;
          bool consumer_warp = warp_id > 0;
          if (producer_warp) { asm("cp.async.ca.shared.global"); }
          if (consumer_warp) { tile[threadIdx.x] += 1; }
        }
        """
        cluster_source = r"""
        __cluster_dims__(2, 1, 1) __global__ void clustered(CUtensorMap map) {
          asm("cp.async.bulk.tensor.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster");
        }
        """
        cute_source = (
            "from cutlass import cute\n@cute.kernel\ndef kernel():\n    pass\n"
        )

        warp = architecture_islands.extract_architecture_features(warp_source)
        cluster = architecture_islands.extract_architecture_features(cluster_source)
        cute = architecture_islands.extract_architecture_features(cute_source)

        self.assertEqual(warp["warp_specialization"], 1)
        self.assertEqual(
            architecture_islands.architecture_label(warp), "warp_specialization"
        )
        self.assertEqual(cluster["cluster_tma_broadcast"], 1)
        self.assertEqual(
            architecture_islands.architecture_label(cluster),
            "cluster_tma_broadcast",
        )
        self.assertEqual(cute["cute_dsl"], 1)
        self.assertEqual(architecture_islands.architecture_label(cute), "cute_dsl")

        dsmem = architecture_islands.extract_architecture_features(
            "cg::cluster_group cluster = cg::this_cluster(); "
            "cluster.map_shared_rank(cluster_shared + threadIdx.x, 1);"
        )
        self.assertEqual(dsmem["cluster_dsmem"], 1)
        self.assertEqual(
            architecture_islands.architecture_label(dsmem), "cluster_dsmem"
        )

    def test_detects_sort_families_and_hierarchical_histograms(self) -> None:
        cub = architecture_islands.extract_architecture_features(
            "cub::DeviceRadixSort::SortPairs(nullptr, bytes, a, b, c, d, n, 0, 8);"
        )
        expert = architecture_islands.extract_architecture_features(
            "__global__ void expert_parallel_stable_scan() { "
            "const int expert = blockIdx.x; }"
        )
        hierarchical = architecture_islands.extract_architecture_features(
            "__shared__ int histogram_banks[4][256]; int block_offsets[256];"
        )

        self.assertEqual(cub["cub_radix_sort"], 1)
        self.assertEqual(cub["radix_bits"], 8)
        self.assertEqual(architecture_islands.architecture_label(cub), "cub_radix_sort")
        self.assertEqual(
            architecture_islands.architecture_label(expert), "expert_parallel_scan"
        )
        self.assertEqual(hierarchical["bank_replicated_histogram"], 1)
        self.assertEqual(hierarchical["hierarchical_histogram"], 1)
        self.assertEqual(
            architecture_islands.architecture_label(hierarchical),
            "hierarchical_histogram",
        )

    def test_expert_indexed_prefix_is_not_misclassified_as_expert_scan(self) -> None:
        prefix = architecture_islands.extract_architecture_features(
            "__global__ void prefix_counts(const int* counts, int* block_offsets) {"
            " const int expert = blockIdx.x;"
            " block_offsets[threadIdx.x * 256 + expert] = counts[expert];"
            " }"
        )

        self.assertEqual(prefix["expert_parallel_scan"], 0)
        self.assertNotEqual(
            architecture_islands.architecture_label(prefix),
            "expert_parallel_scan",
        )

    def test_pca_and_cluster_output_is_finite_and_deterministic(self) -> None:
        sources = [
            "__global__ void plain() {}",
            "__global__ void atomic() { atomicAdd((int*)0, 1); }",
            "__cluster_dims__(2,1,1) __global__ void tma(CUtensorMap x) { /* multicast */ }",
            "from cutlass import cute\n@cute.kernel\ndef k(): pass",
        ]
        rows = [
            architecture_islands.extract_architecture_features(source)
            for source in sources
        ]
        first = architecture_islands.fit_architecture_pca(rows)
        second = architecture_islands.fit_architecture_pca(rows)
        self.assertEqual(first["components"], second["components"])
        self.assertEqual(first["coordinates"], second["coordinates"])
        self.assertEqual(len(first["coordinates"]), len(rows))
        self.assertTrue(
            all(
                math.isfinite(value)
                for coordinate in first["coordinates"]
                for value in coordinate
            )
        )
        labels_a, centroids_a = architecture_islands.cluster_pca_coordinates(
            first["coordinates"], 4
        )
        labels_b, centroids_b = architecture_islands.cluster_pca_coordinates(
            second["coordinates"], 4
        )
        self.assertEqual(labels_a, labels_b)
        self.assertEqual(centroids_a, centroids_b)

    def test_semantic_anchors_keep_key_routes_on_distinct_islands(self) -> None:
        self.assertEqual(
            architecture_islands.architecture_island_id("warp_specialization", 7, 8),
            0,
        )
        self.assertEqual(
            architecture_islands.architecture_island_id("cluster_tma_broadcast", 0, 8),
            1,
        )
        self.assertEqual(
            architecture_islands.architecture_island_id("cute_dsl", 0, 8), 4
        )
        self.assertEqual(
            architecture_islands.architecture_island_id("cub_radix_sort", 0, 8),
            6,
        )
        self.assertEqual(
            architecture_islands.architecture_island_id("expert_parallel_scan", 0, 8),
            7,
        )

    def test_too_few_islands_is_rejected_instead_of_colliding(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least 8 islands"):
            architecture_islands.architecture_island_id("warp_specialization", 0, 4)


class TestArchitecturePopulationRetention(unittest.TestCase):
    @staticmethod
    def _routed_solution(
        solution_id: str,
        island: int,
        score: float,
        *,
        migrated: bool = False,
    ):
        solution = _solution(
            solution_id,
            f"kernel-{solution_id}",
            iteration=int(solution_id.rsplit("-", 1)[-1]),
            score=score,
            weight=1,
        )
        solution.island_id = island
        solution.metadata = {
            "architecture_home_island_id": (island - 1) % 8 if migrated else island,
            "architecture_island_id": island,
            "migrated": migrated,
        }
        return solution

    @staticmethod
    def _retention_memory(solutions):
        populations = {solution.solution_id: solution for solution in solutions}
        islands = [set() for _ in range(8)]
        for solution in solutions:
            islands[solution.island_id].add(solution.solution_id)
        return SimpleNamespace(
            _lock=threading.RLock(),
            populations=populations,
            solutions=dict(populations),
            islands=islands,
            island_feature_maps=[{} for _ in range(8)],
            island_best_solution=[None] * 8,
            island_capacity=[len(island) for island in islands],
            elites=set(populations),
            best_solution_id=max(solutions, key=lambda item: item.score).solution_id,
            population_size=100,
            num_islands=8,
            last_iteration=200,
        )

    def test_weak_home_quota_survives_global_score_eviction(self) -> None:
        solutions = []
        for island in range(8):
            solutions.extend(
                self._routed_solution(
                    f"home{island}-{index}", island, 0.1 + island * 0.01
                )
                for index in range(6)
            )
        solutions.extend(
            self._routed_solution(f"dominant-{index}", 7, 0.8 + index * 0.001)
            for index in range(53)
        )
        memory = self._retention_memory(solutions)

        result = architecture_islands.enforce_architecture_population_limit(
            memory,
            minimum_home_per_island=6,
            maximum_migrant_fraction=0.2,
        )

        self.assertEqual(result["removed"], 1)
        self.assertEqual(result["home_occupancy"][:7], [6] * 7)
        self.assertGreaterEqual(result["home_occupancy"][7], 6)
        self.assertEqual(len(memory.populations), 100)

    def test_migrants_are_bounded_even_when_population_drops_below_capacity(
        self,
    ) -> None:
        homes = [
            self._routed_solution(f"home-{index}", index % 8, 0.2 + index * 0.001)
            for index in range(78)
        ]
        migrants = [
            self._routed_solution(
                f"migrant-{index}", index % 8, 0.95 + index * 0.001, migrated=True
            )
            for index in range(25)
        ]
        memory = self._retention_memory(homes + migrants)

        result = architecture_islands.enforce_architecture_population_limit(
            memory,
            minimum_home_per_island=6,
            maximum_migrant_fraction=0.2,
        )

        remaining_migrants = sum(
            architecture_islands.is_migration_copy(solution)
            for solution in memory.populations.values()
        )
        self.assertEqual(result["migrants_removed"], 6)
        self.assertEqual(remaining_migrants, 19)
        self.assertEqual(len(memory.populations), 97)
        self.assertLessEqual(remaining_migrants / len(memory.populations), 0.2)

    def test_migrants_are_bounded_when_capacity_eviction_dominates(self) -> None:
        homes = [
            self._routed_solution(f"home-{index}", index % 8, 0.2 + index * 0.001)
            for index in range(95)
        ]
        migrants = [
            self._routed_solution(
                f"migrant-{index}",
                index % 8,
                0.95 + index * 0.001,
                migrated=True,
            )
            for index in range(25)
        ]
        memory = self._retention_memory(homes + migrants)

        result = architecture_islands.enforce_architecture_population_limit(
            memory,
            minimum_home_per_island=6,
            maximum_migrant_fraction=0.2,
        )

        remaining_migrants = sum(
            architecture_islands.is_migration_copy(solution)
            for solution in memory.populations.values()
        )
        self.assertEqual(result["removed"], 20)
        self.assertEqual(result["migrants_removed"], 5)
        self.assertEqual(len(memory.populations), 100)
        self.assertLessEqual(remaining_migrants / len(memory.populations), 0.2)


class TestArchitectureIslandRouting(unittest.TestCase):
    WARP_SOURCE = r"""
    __global__ void warp_specialized() {
      __shared__ int tile[256];
      int warp_id = threadIdx.x / 32;
      bool producer_warp = warp_id == 0;
      bool consumer_warp = warp_id > 0;
      if (producer_warp) asm("cp.async.ca.shared.global");
      if (consumer_warp) tile[threadIdx.x] += 1;
    }
    """
    CLUSTER_SOURCE = r"""
    __cluster_dims__(2,1,1) __global__ void cluster_tma(CUtensorMap map) {
      asm("cp.async.bulk.tensor.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster");
    }
    """

    def test_summary_child_gets_explainable_metadata(self) -> None:
        memory = _memory()
        child = _solution("warp", self.WARP_SOURCE, iteration=1, score=0.8, weight=1.0)

        result = architecture_islands.classify_and_route_solution(memory, child, 8)

        self.assertEqual(result["label"], "warp_specialization")
        self.assertEqual(child.island_id, 0)
        self.assertEqual(child.metadata["architecture_island_id"], 0)
        self.assertEqual(
            child.metadata["architecture_island_profile"], "warp_specialization"
        )
        self.assertEqual(len(child.metadata["pca_coordinates"]), 3)
        self.assertIn("warp_specialization", child.metadata["architecture_tags"])

    def test_empty_semantic_island_bootstrap_attempt_is_checkpointed(self) -> None:
        memory = _memory(last_iteration=10)
        architecture_islands.ensure_architecture_islands(memory, 8)
        environment = {
            "ATREX_PES_STAGNATION_SEEDS": "1",
            "ATREX_PES_MIN_HOME_PER_ISLAND": "6",
            "ATREX_PES_BOOTSTRAP_MAX_ATTEMPTS": "8",
            "SOL58_CODE_LANGUAGE": "cuda_cpp",
        }

        with mock.patch.dict(os.environ, environment):
            parent = compat_adapter._architecture_bootstrap_seed_parent(
                memory, requested_island=1, num_islands=8
            )

        self.assertEqual(parent["metadata"]["architecture_label"], "cluster_dsmem")
        self.assertEqual(parent["island_id"], 1)
        self.assertEqual(memory._atrex_bootstrap_attempts["1:cluster_dsmem"], 1)
        state = architecture_islands.architecture_checkpoint_payload(memory)[
            "stagnation_escape"
        ]
        restored = _memory(last_iteration=10)
        architecture_islands.ensure_architecture_islands(restored, 8)
        architecture_islands.restore_stagnation_checkpoint_state(restored, state)
        self.assertEqual(restored._atrex_bootstrap_attempts, {"1:cluster_dsmem": 1})
        self.assertEqual(
            restored._atrex_pending_architecture_bootstrap["expected_iteration"], 11
        )

        child = _solution(
            "cluster-child", self.CLUSTER_SOURCE, iteration=11, score=0.5, weight=1
        )
        child.island_id = 1
        child.evaluation = {
            "status": "success",
            "metrics": {"passed": 16, "total": 16, "expected_total": 16},
        }
        satisfied = compat_adapter._record_architecture_bootstrap_outcome(
            restored, child, "cluster_dsmem"
        )
        self.assertTrue(satisfied)
        self.assertTrue(child.metadata["architecture_bootstrap_satisfied"]["satisfied"])

    def test_single_island_checkpoint_population_is_reclassified(self) -> None:
        warp = _solution("warp", self.WARP_SOURCE, iteration=1, score=0.8, weight=1)
        cluster = _solution(
            "cluster", self.CLUSTER_SOURCE, iteration=2, score=0.7, weight=1
        )
        cute = _solution(
            "cute",
            "from cutlass import cute\n@cute.kernel\ndef k(): pass",
            iteration=3,
            score=0.6,
            weight=1,
        )
        memory = _memory(warp, cluster, cute, last_iteration=85)

        result = architecture_islands.rebuild_architecture_islands(memory, 8)

        self.assertEqual(result, {"solutions": 3, "islands": 8})
        self.assertEqual(len(memory.islands), 8)
        self.assertIn("warp", memory.islands[0])
        self.assertIn("cluster", memory.islands[1])
        self.assertIn("cute", memory.islands[4])
        self.assertEqual(sum(len(island) for island in memory.islands), 3)
        self.assertEqual(memory.island_capacity, [len(i) for i in memory.islands])

    def test_exchange_occurs_only_when_crossing_20_round_boundaries(self) -> None:
        warp_a = _solution("wa", self.WARP_SOURCE, iteration=1, score=0.8, weight=1)
        warp_b = _solution(
            "wb", self.WARP_SOURCE + "\n// variant", iteration=2, score=0.7, weight=1
        )
        cluster_a = _solution(
            "ca", self.CLUSTER_SOURCE, iteration=3, score=0.9, weight=1
        )
        cluster_b = _solution(
            "cb",
            self.CLUSTER_SOURCE + "\n// variant",
            iteration=4,
            score=0.6,
            weight=1,
        )
        memory = _memory(warp_a, warp_b, cluster_a, cluster_b, last_iteration=19)

        before = architecture_islands.maybe_exchange_islands(memory, 8, 20)
        self.assertEqual(before["migrated"], 0)
        memory.last_iteration = 20
        first = architecture_islands.maybe_exchange_islands(memory, 8, 20)
        self.assertGreater(first["migrated"], 0)
        self.assertEqual(memory._atrex_last_migration_iteration, 20)
        first_migrants = {
            solution_id
            for solution_id, solution in memory.populations.items()
            if solution.metadata.get("migrated")
        }

        memory.last_iteration = 39
        between = architecture_islands.maybe_exchange_islands(memory, 8, 20)
        self.assertEqual(between["migrated"], 0)
        self.assertEqual(
            {
                solution_id
                for solution_id, solution in memory.populations.items()
                if solution.metadata.get("migrated")
            },
            first_migrants,
        )

        memory.last_iteration = 40
        second = architecture_islands.maybe_exchange_islands(memory, 8, 20)
        self.assertEqual(second["expired"], len(first_migrants))
        self.assertGreater(second["migrated"], 0)
        self.assertEqual(memory._atrex_last_migration_iteration, 40)

    def test_zero_migration_rate_disables_exchange(self) -> None:
        warp = _solution("warp", self.WARP_SOURCE, iteration=1, score=0.8, weight=1)
        memory = _memory(warp, last_iteration=20)
        architecture_islands.rebuild_architecture_islands(memory, 8, 20)
        memory.migration_rate = 0

        migrated = architecture_islands.perform_island_exchange(memory, 20)

        self.assertEqual(migrated, 0)
        self.assertFalse(
            any(
                solution.metadata.get("migrated")
                for solution in memory.populations.values()
            )
        )

    def test_architecture_state_is_persisted_and_restored(self) -> None:
        warp = _solution("warp", self.WARP_SOURCE, iteration=20, score=0.8, weight=1)
        memory = _memory(warp, last_iteration=20)
        architecture_islands.rebuild_architecture_islands(memory, 8, 20)
        memory._atrex_last_migration_iteration = 20
        memory.migration_interval = 20
        memory._atrex_stagnation_best_marker = (20, 0.8)
        memory._atrex_stagnation_seed_bucket = None
        memory._atrex_stagnation_retry_bucket = 4
        memory._atrex_stagnation_seed_retries = 1
        memory._atrex_last_stagnation_seed_id = "cub_device_radix"
        memory._atrex_pending_stagnation_escape = {
            "expected_iteration": 21,
            "bucket": 4,
            "seed_id": "cub_device_radix",
            "seed_family": "cub_radix_sort",
            "incumbent_family": "hierarchical_histogram",
        }

        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = Path(tmp) / "checkpoints" / "checkpoint-test"
            checkpoint.mkdir(parents=True)
            (checkpoint / "metadata.json").write_text("{}", encoding="utf-8")
            self.assertTrue(
                architecture_islands.write_architecture_checkpoint(memory, tmp, "test")
            )
            payload = json.loads((checkpoint / "metadata.json").read_text())
            self.assertEqual(
                payload["atrex_architecture"]["last_migration_iteration"], 20
            )
            self.assertNotIn("coordinates", payload["atrex_architecture"]["pca_model"])
            stagnation = payload["atrex_architecture"]["stagnation_escape"]
            self.assertEqual(stagnation["best_marker"], [20, 0.8])
            self.assertIsNone(stagnation["seed_bucket"])
            self.assertEqual(stagnation["retry_bucket"], 4)
            self.assertEqual(stagnation["seed_retries"], 1)
            self.assertEqual(stagnation["pending_escape"]["expected_iteration"], 21)

            restored = _memory(warp, last_iteration=20)
            with mock.patch.object(
                architecture_islands,
                "rebuild_architecture_islands",
                wraps=architecture_islands.rebuild_architecture_islands,
            ) as rebuild:
                result = architecture_islands.restore_architecture_checkpoint(
                    restored, str(checkpoint), 8, 20
                )
            rebuild.assert_not_called()
            self.assertEqual(result["islands"], 8)
            self.assertEqual(result["last_migration_iteration"], 20)
            self.assertEqual(restored._atrex_last_migration_iteration, 20)
            self.assertEqual(restored._atrex_stagnation_best_marker, (20, 0.8))
            self.assertIsNone(restored._atrex_stagnation_seed_bucket)
            self.assertEqual(restored._atrex_stagnation_retry_bucket, 4)
            self.assertEqual(restored._atrex_stagnation_seed_retries, 1)
            self.assertEqual(
                restored._atrex_pending_stagnation_escape["expected_iteration"], 21
            )

    def test_legacy_checkpoint_recovers_latest_escape_bucket(self) -> None:
        older = _solution("older", self.WARP_SOURCE, iteration=20, score=0.8, weight=1)
        newer = _solution(
            "newer", self.CLUSTER_SOURCE, iteration=40, score=0.7, weight=1
        )
        older.metadata["stagnation_escape_satisfied"] = {
            "expected_iteration": 20,
            "bucket": 2,
            "seed_id": "cub_device_radix",
        }
        newer.metadata["stagnation_escape_satisfied"] = {
            "expected_iteration": 40,
            "bucket": 3,
            "seed_id": "expert_parallel_scan",
        }
        memory = _memory(older, newer, last_iteration=40)

        status = architecture_islands.restore_stagnation_checkpoint_state(memory, {})

        self.assertTrue(status["legacy_reconstructed"])
        self.assertEqual(memory._atrex_stagnation_best_marker, (20, 0.8))
        self.assertEqual(memory._atrex_stagnation_seed_bucket, 3)
        self.assertEqual(memory._atrex_stagnation_retry_bucket, 3)
        self.assertEqual(memory._atrex_stagnation_seed_retries, 0)
        self.assertEqual(memory._atrex_last_stagnation_seed_id, "expert_parallel_scan")
        self.assertIsNone(memory._atrex_pending_stagnation_escape)

    def test_legacy_checkpoint_preserves_remaining_violation_retry(self) -> None:
        incumbent = _solution(
            "incumbent", self.CLUSTER_SOURCE, iteration=4, score=0.9, weight=1
        )
        violation = _solution(
            "violation", self.WARP_SOURCE, iteration=40, score=0.0, weight=1
        )
        violation.metadata["stagnation_escape_violation"] = {
            "expected_iteration": 40,
            "bucket": 3,
            "seed_id": "cub_device_radix",
            "retry": 1,
            "max_attempts": 2,
        }
        memory = _memory(incumbent, last_iteration=40)
        memory.solutions[violation.solution_id] = violation

        status = architecture_islands.restore_stagnation_checkpoint_state(memory, {})

        self.assertTrue(status["legacy_reconstructed"])
        self.assertIsNone(memory._atrex_stagnation_seed_bucket)
        self.assertEqual(memory._atrex_stagnation_retry_bucket, 3)
        self.assertEqual(memory._atrex_stagnation_seed_retries, 1)

    def test_stagnation_state_restores_without_architecture_routing(self) -> None:
        memory = _memory(last_iteration=20)
        state = {
            "version": architecture_islands.STAGNATION_CHECKPOINT_VERSION,
            "best_marker": [4, 0.9],
            "seed_bucket": None,
            "retry_bucket": 2,
            "seed_retries": 1,
            "last_seed_id": "cub_device_radix",
            "pending_escape": {"expected_iteration": 21, "bucket": 2},
        }
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = Path(tmp)
            (checkpoint / "metadata.json").write_text(
                json.dumps({"atrex_architecture": {"stagnation_escape": state}}),
                encoding="utf-8",
            )
            status = architecture_islands.restore_stagnation_checkpoint(
                memory, str(checkpoint)
            )

        self.assertTrue(status["restored"])
        self.assertEqual(memory._atrex_stagnation_best_marker, (4, 0.9))
        self.assertEqual(memory._atrex_stagnation_retry_bucket, 2)
        self.assertEqual(memory._atrex_stagnation_seed_retries, 1)
        self.assertEqual(
            memory._atrex_pending_stagnation_escape["expected_iteration"], 21
        )


if __name__ == "__main__":
    unittest.main()
