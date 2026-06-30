#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Unit tests for tools/evolution_db.py (stdlib unittest; no GPU, no deps).

Run: python3 -m unittest tools.test_evolution_db   (from repo root)
  or python3 tools/test_evolution_db.py
"""

import argparse
import json
import random
import tempfile
import unittest
from pathlib import Path

import evolution_db as edb
from evolution_db import (
    EvolutionDB,
    NotFound,
    diverse_reference_set,
    fast_code_distance,
    population_diversity,
)


def make_add_ns(**over) -> argparse.Namespace:
    base = dict(
        workspace="",
        json=False,
        generation=1,
        parent=None,
        code="",
        lang="triton",
        score=None,
        correctness="PASS",
        rel_err=None,
        latency_us=None,
        tflops=None,
        bandwidth_gbps=None,
        island=None,
        generate_plan=None,
        action_category=None,
        evaluation=None,
        evidence_file=None,
        iteration_ref=None,
    )
    base.update(over)
    return argparse.Namespace(**base)


class TestDiversityFns(unittest.TestCase):
    def test_identical_distance_zero(self):
        self.assertEqual(fast_code_distance("abc\nxyz", "abc\nxyz"), 0.0)

    def test_different_distance_positive(self):
        self.assertGreater(fast_code_distance("a", "b\nc\nd\ne"), 0.0)

    def test_population_diversity_range(self):
        d = population_diversity(["aaa", "bbbb\ncc", "x"])
        self.assertGreaterEqual(d, 0.0)
        self.assertLessEqual(d, 1.0)

    def test_population_diversity_singleton(self):
        self.assertEqual(population_diversity(["only"]), 0.0)

    def test_reference_set_capped(self):
        codes = [f"code-{i}\n" * i for i in range(1, 30)]
        ref = diverse_reference_set(codes, 5)
        self.assertEqual(len(ref), 5)


class DBTestBase(unittest.TestCase):
    def setUp(self):
        random.seed(0)
        self.tmp = tempfile.TemporaryDirectory()
        self.ws = Path(self.tmp.name)
        (self.ws / "kernel.py").write_text("def run():\n    return 0\n")
        (self.ws / "memory").mkdir()
        (self.ws / "memory" / "v0.json").write_text(
            json.dumps({"performance": {"latency_us": 1000.0}})
        )

    def tearDown(self):
        self.tmp.cleanup()

    def _init(self, **cfg_over):
        cfg_path = self.ws / "cfg.json"
        cfg_path.write_text(json.dumps(cfg_over))
        db = EvolutionDB(str(self.ws))
        db.init(str(cfg_path) if cfg_over else None)
        db.load()
        return db

    def _write_kernel(self, name, text):
        p = self.ws / name
        p.write_text(text)
        return name


class TestInitAndSeed(DBTestBase):
    def test_init_structure(self):
        db = self._init()
        self.assertTrue((self.ws / "database" / "config.json").exists())
        self.assertTrue((self.ws / "database" / "solutions").is_dir())
        self.assertTrue((self.ws / "database" / "checkpoints").is_dir())
        self.assertEqual(db.config["num_islands"], 3)

    def test_import_seed(self):
        db = self._init()
        res = db.import_seed("memory/v0.json", "kernel.py")
        self.assertEqual(res["baseline_latency_us"], 1000.0)
        db.load()
        self.assertEqual(db.state["baseline"]["latency_us"], 1000.0)
        self.assertEqual(len(db.state["solutions"]), 1)
        self.assertEqual(db.best(None)["score"], 1.0)
        self.assertTrue((self.ws / "database" / "checkpoints" / "iter-0").is_dir())


class TestAdd(DBTestBase):
    def test_score_from_latency(self):
        db = self._init()
        db.import_seed("memory/v0.json", "kernel.py")
        db.load()
        self._write_kernel("c1.py", "def run():\n    return 1  # faster\n")
        res = db.add(make_add_ns(code="c1.py", generation=1, latency_us=500.0))
        self.assertAlmostEqual(res["score"], 2.0)  # 1000 / 500

    def test_failed_candidate_scores_zero(self):
        db = self._init()
        db.import_seed("memory/v0.json", "kernel.py")
        db.load()
        self._write_kernel("c2.py", "def run():\n    raise RuntimeError\n")
        res = db.add(
            make_add_ns(code="c2.py", generation=1, latency_us=100.0, correctness="FAIL")
        )
        self.assertEqual(res["score"], 0.0)

    def test_explicit_score_overrides(self):
        db = self._init()
        db.import_seed("memory/v0.json", "kernel.py")
        db.load()
        self._write_kernel("c3.py", "def run():\n    return 3\n")
        res = db.add(make_add_ns(code="c3.py", generation=1, score=5.5))
        self.assertEqual(res["score"], 5.5)


class TestMapElitesAndElites(DBTestBase):
    def _seed_sol(self, db, sid, score, key, island=0):
        sol = {
            "solution_id": sid,
            "parent_id": None,
            "score": score,
            "island_id": island,
            "generation": 1,
            "metadata": {"MAP_Elite_feature": key, "optimization": {}},
            "summary": "",
            "generate_plan": "",
            "sample_cnt": 0,
            "sample_weight": 0.0,
            "solution": "",
        }
        db.state["solutions"][sid] = sol
        return sol

    def test_cell_replacement_keeps_higher_score(self):
        db = self._init(num_islands=1)
        low = self._seed_sol(db, "low00000", 1.0, "1-1-1")
        high = self._seed_sol(db, "high0000", 2.0, "1-1-1")
        db._update_island(low)
        db._update_island(high)
        fmap = db.state["island_feature_maps"][0]
        self.assertEqual(fmap["1-1-1"], "high0000")
        self.assertIn("high0000", db.state["islands"][0])
        self.assertNotIn("low00000", db.state["islands"][0])

    def test_worse_does_not_evict(self):
        db = self._init(num_islands=1)
        high = self._seed_sol(db, "high0000", 2.0, "2-2-2")
        low = self._seed_sol(db, "low00000", 1.0, "2-2-2")
        db._update_island(high)
        db._update_island(low)
        self.assertEqual(db.state["island_feature_maps"][0]["2-2-2"], "high0000")

    def test_elite_archive_cap(self):
        db = self._init(elite_archive_size=3)
        for i in range(6):
            sol = self._seed_sol(db, f"sol{i:05d}", float(i), f"{i}-0-0")
            db._update_elites(sol)
        elites = db.state["elites"]
        self.assertEqual(len(elites), 3)
        # only the top-3 scores survive (5,4,3)
        self.assertEqual(set(elites), {"sol00005", "sol00004", "sol00003"})


class TestMigrationAndPrune(DBTestBase):
    def test_migration_copies_to_next_island(self):
        db = self._init(num_islands=2, migration_interval=1, migration_rate=0.5)
        db.import_seed("memory/v0.json", "kernel.py")  # gen0, island0
        db.load()
        self._write_kernel("m1.py", "def run():\n    return 11\n")
        db.add(make_add_ns(code="m1.py", generation=1, island=0, latency_us=500.0))
        self.assertEqual(db.state["last_migration_generation"], 1)
        self.assertGreaterEqual(len(db.state["islands"][1]), 1)

    def test_prune_caps_population(self):
        db = self._init(population_size=3, num_islands=1)
        db.import_seed("memory/v0.json", "kernel.py")
        db.load()
        for i in range(5):
            self._write_kernel(f"p{i}.py", f"def run():\n    return {i}\n" + "x" * i)
            db.add(make_add_ns(code=f"p{i}.py", generation=1, score=float(i + 1)))
        self.assertLessEqual(len(db.state["solutions"]), 3)


class TestSelectAndLineage(DBTestBase):
    def test_select_parents_returns_n_and_counts(self):
        db = self._init()
        db.import_seed("memory/v0.json", "kernel.py")
        db.load()
        self._write_kernel("s1.py", "def run():\n    return 7\n" + "y" * 50)
        db.add(make_add_ns(code="s1.py", generation=1, latency_us=250.0))
        parents = db.select_parents(3)
        self.assertEqual(len(parents), 3)
        db.load()
        total = sum(s["sample_cnt"] for s in db.state["solutions"].values())
        self.assertEqual(total, 3)

    def test_lineage_chain(self):
        db = self._init()
        seed = db.import_seed("memory/v0.json", "kernel.py")["solution_id"]
        db.load()
        self._write_kernel("l1.py", "def run():\n    return 9\n")
        child = db.add(
            make_add_ns(code="l1.py", generation=1, parent=seed, latency_us=400.0)
        )["solution_id"]
        chain = db.lineage(child)
        self.assertEqual([c["solution_id"] for c in chain], [child, seed])

    def test_lineage_missing_raises(self):
        db = self._init()
        db.import_seed("memory/v0.json", "kernel.py")
        db.load()
        with self.assertRaises(NotFound):
            db.lineage("deadbeef")


class TestConvergence(DBTestBase):
    def test_no_improve_triggers_stop(self):
        db = self._init(num_islands=1)
        db.config["convergence"] = {"no_improve_patience": 2, "min_rel_improve": 0.01}
        db.config["budget"] = {"max_generations": 100}
        db._write_config()
        db.import_seed("memory/v0.json", "kernel.py")  # checkpoint(0)
        db.load()
        r1 = db.checkpoint(1)
        self.assertFalse(r1["stopped"])
        r2 = db.checkpoint(2)
        self.assertTrue(r2["stopped"])
        self.assertEqual(r2["stop_reason"], "no_improve")

    def test_budget_exhausted(self):
        db = self._init(num_islands=1)
        db.config["convergence"] = {"no_improve_patience": 100, "min_rel_improve": 0.01}
        db.config["budget"] = {"max_generations": 1}
        db._write_config()
        db.import_seed("memory/v0.json", "kernel.py")
        db.load()
        r = db.checkpoint(1)
        self.assertTrue(r["stopped"])
        self.assertEqual(r["stop_reason"], "budget_exhausted")


class TestGenerationFanout(DBTestBase):
    """M3: a full N-candidate generation (select -> add x N -> checkpoint)."""

    def _add_child(self, db, name, body, gen, parent, **over):
        self._write_kernel(name, body)
        return db.add(make_add_ns(code=name, generation=gen, parent=parent, **over))

    def test_n3_generation_admits_all_and_branches_lineage(self):
        db = self._init()
        seed = db.import_seed("memory/v0.json", "kernel.py")["solution_id"]
        db.load()

        # one generation, three concurrent candidates of the same parent (seed)
        a = self._add_child(db, "a.py", "def run():\n    return 1\n" + "a" * 10,
                            1, seed, latency_us=500.0, action_category="vectorized_load")
        b = self._add_child(db, "b.py", "def run():\n    return 2\n" + "b" * 40,
                            1, seed, latency_us=250.0, action_category="k_split")
        c = self._add_child(db, "c.py", "def run():\n    return 3\n" + "c" * 70,
                            1, seed, correctness="FAIL", latency_us=100.0,
                            action_category="swizzle")

        # all three admitted to the population alongside the seed
        self.assertEqual(len(db.state["solutions"]), 4)
        gens = sorted(db.state["solutions"][s]["generation"] for s in (a["solution_id"], b["solution_id"], c["solution_id"]))
        self.assertEqual(gens, [1, 1, 1])

        # scores: speedup for PASS, 0 for the failed candidate
        self.assertAlmostEqual(a["score"], 2.0)
        self.assertAlmostEqual(b["score"], 4.0)
        self.assertEqual(c["score"], 0.0)

        # best of the generation is the fastest PASS candidate
        ck = db.checkpoint(1)
        self.assertEqual(ck["best_solution_id"], b["solution_id"])
        self.assertAlmostEqual(ck["best_score"], 4.0)

        # siblings branch from the same parent in the lineage DAG
        for child in (a, b, c):
            chain = db.lineage(child["solution_id"])
            self.assertEqual([x["solution_id"] for x in chain], [child["solution_id"], seed])

        # first-generation checkpoint artifacts exist and list the candidates
        cdir = self.ws / "database" / "checkpoints" / "iter-1"
        self.assertTrue((cdir / "best_solution.json").exists())
        self.assertTrue((cdir / "metadata.json").exists())
        self.assertEqual(len(list((cdir / "solutions").glob("*.json"))), 4)

    def test_two_generations_best_is_monotonic(self):
        db = self._init()
        seed = db.import_seed("memory/v0.json", "kernel.py")["solution_id"]
        db.load()
        # gen1: best speedup 2x
        self._add_child(db, "g1a.py", "def run():\n    return 1\n" + "x" * 15,
                        1, seed, latency_us=500.0)
        self._add_child(db, "g1b.py", "def run():\n    return 1\n" + "y" * 35,
                        1, seed, latency_us=800.0)
        ck1 = db.checkpoint(1)
        # gen2: a better child (4x) from the gen1 best
        parent2 = ck1["best_solution_id"]
        self._add_child(db, "g2a.py", "def run():\n    return 2\n" + "z" * 55,
                        2, parent2, latency_us=250.0)
        ck2 = db.checkpoint(2)
        self.assertGreaterEqual(ck2["best_score"], ck1["best_score"])
        self.assertAlmostEqual(ck2["best_score"], 4.0)
        # history records seed(0), gen1, gen2
        self.assertEqual([h["generation"] for h in db.state["history"]], [0, 1, 2])

    def test_select_after_fanout_counts_samples(self):
        db = self._init()
        seed = db.import_seed("memory/v0.json", "kernel.py")["solution_id"]
        db.load()
        self._add_child(db, "f1.py", "def run():\n    return 1\n" + "q" * 20,
                        1, seed, latency_us=500.0)
        parents = db.select_parents(3)
        self.assertEqual(len(parents), 3)
        db.load()
        self.assertEqual(sum(s["sample_cnt"] for s in db.state["solutions"].values()), 3)


class TestCLISmoke(DBTestBase):
    def test_cli_end_to_end(self):
        ws = str(self.ws)
        self.assertEqual(edb.main(["init", "--workspace", ws]), 0)
        self.assertEqual(
            edb.main(["--seed", "1", "import-seed", "--workspace", ws]), 0
        )
        self._write_kernel("k1.py", "def run():\n    return 1\n" + "z" * 20)
        rc = edb.main(
            [
                "--seed", "1", "add", "--workspace", ws, "--generation", "1",
                "--code", "k1.py", "--correctness", "PASS", "--latency-us", "500",
            ]
        )
        self.assertEqual(rc, 0)
        self.assertEqual(edb.main(["checkpoint", "--workspace", ws, "--generation", "1"]), 0)
        self.assertEqual(edb.main(["summary", "--workspace", ws]), 0)
        self.assertEqual(edb.main(["best", "--workspace", ws]), 0)

    def test_cli_uninitialized_returns_3(self):
        rc = edb.main(["summary", "--workspace", str(self.ws / "nope")])
        self.assertEqual(rc, 3)


if __name__ == "__main__":
    unittest.main(verbosity=2)
