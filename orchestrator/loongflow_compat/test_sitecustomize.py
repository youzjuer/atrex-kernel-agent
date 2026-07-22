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
from orchestrator.loongflow_compat import sitecustomize


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
            sitecustomize._adaptive_exploration_rate(0.1, [1.0, 1.0, 1.0, 1.0]),
            0.1,
        )

    def test_hard_stagnation_is_checked_before_medium(self) -> None:
        self.assertEqual(
            sitecustomize._adaptive_exploration_rate(0.1, [0.8] * 5),
            0.4,
        )
        self.assertEqual(
            sitecustomize._adaptive_exploration_rate(
                0.1,
                [0.80, 0.805, 0.81, 0.815, 0.82],
            ),
            0.2,
        )

    def test_exploration_is_capped(self) -> None:
        self.assertEqual(
            sitecustomize._adaptive_exploration_rate(0.3, [0.8] * 5),
            0.9,
        )


class TestNcuSummaryPrompt(unittest.TestCase):
    def test_ncu_interpretation_contract_is_added_once(self) -> None:
        prompt = sitecustomize._append_ncu_summary_instructions("Base summary prompt")
        repeated = sitecustomize._append_ncu_summary_instructions(prompt)

        self.assertEqual(prompt, repeated)
        self.assertEqual(prompt.count(sitecustomize._NCU_SUMMARY_MARKER), 1)
        self.assertIn("metrics.ncu_analysis", prompt)
        self.assertIn("must never change the fitness assessment", prompt)


class TestSourceDeduplication(unittest.TestCase):
    def test_hash_ignores_outer_whitespace(self) -> None:
        self.assertEqual(
            sitecustomize._solution_source_hash(" kernel\n"),
            sitecustomize._solution_source_hash("kernel"),
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

        removed = sitecustomize._deduplicate_memory_indexes(memory)

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
                sitecustomize._solution_source_hash(stale): {
                    "official_score": 0.852202,
                    "submission_id": 25255,
                },
                sitecustomize._solution_source_hash(official_best): {
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
                changed = sitecustomize._reconcile_authoritative_scores(memory)

        self.assertEqual(changed, 2)
        self.assertAlmostEqual(stale.score, 0.852202)
        self.assertAlmostEqual(official_best.score, 0.856272)
        self.assertEqual(stale.metadata["fitness_source"], "official")
        self.assertEqual(memory.best_solution_id, "local")
        self.assertEqual(memory.island_best_solution, ["local"])


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

    def test_architecture_state_is_persisted_and_restored(self) -> None:
        warp = _solution("warp", self.WARP_SOURCE, iteration=20, score=0.8, weight=1)
        memory = _memory(warp, last_iteration=20)
        architecture_islands.rebuild_architecture_islands(memory, 8, 20)
        memory._atrex_last_migration_iteration = 20
        memory.migration_interval = 20

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


if __name__ == "__main__":
    unittest.main()
