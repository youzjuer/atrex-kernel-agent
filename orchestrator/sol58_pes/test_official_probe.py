from __future__ import annotations

import json
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from orchestrator.sol58_pes.official_probe import (
    OfficialProbePolicy,
    architecture_probe_evidence,
    claim_official_probe,
)


def _evidence() -> dict:
    return {
        "candidate_label": "cluster_tma_broadcast",
        "candidate_tags": ["cluster_tma_broadcast"],
        "incumbent_label": "graph_histogram",
        "active_proxy_sensitive_features": ["tma", "cluster_tma_broadcast"],
        "changed_proxy_sensitive_features": ["tma", "cluster_tma_broadcast"],
        "architecture_sensitive": True,
    }


class TestOfficialProbe(unittest.TestCase):
    def _policy(self, **overrides) -> OfficialProbePolicy:
        values = {
            "enabled": True,
            "interval": 2,
            "cooldown_seconds": 0.0,
            "max_per_day": 6,
            "max_relative_regression": 0.1,
            "architecture_only": True,
        }
        values.update(overrides)
        return OfficialProbePolicy(**values)

    def test_detects_changed_blackwell_architecture_features(self) -> None:
        incumbent = "__global__ void histogram_kernel() { atomicAdd((int*)0, 1); }"
        candidate = """
        // cluster TMA broadcast with specialized producer and consumer warps
        __global__ void radix_pass() {
          int warp_id = threadIdx.x / 32;
          if (warp_id == 0) { asm volatile("cp.async.bulk.tensor"); }
          if (warp_id == 1) { cluster.sync(); }
        }
        """

        evidence = architecture_probe_evidence(candidate, incumbent)

        self.assertTrue(evidence["architecture_sensitive"])
        self.assertIn("tma", evidence["active_proxy_sensitive_features"])
        self.assertIn("tma", evidence["changed_proxy_sensitive_features"])

    def test_claims_every_nth_unique_eligible_candidate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "probes.json"
            first = claim_official_probe(
                path,
                source_sha256="a",
                relative_regression=0.02,
                evidence=_evidence(),
                policy=self._policy(),
                now=1000,
            )
            second = claim_official_probe(
                path,
                source_sha256="b",
                relative_regression=0.03,
                evidence=_evidence(),
                policy=self._policy(),
                now=1001,
            )
            duplicate = claim_official_probe(
                path,
                source_sha256="b",
                relative_regression=0.03,
                evidence=_evidence(),
                policy=self._policy(),
                now=1002,
            )

            registry = json.loads(path.read_text(encoding="utf-8"))

        self.assertFalse(first["claimed"])
        self.assertTrue(second["claimed"])
        self.assertFalse(duplicate["claimed"])
        self.assertEqual(registry["eligible_count"], 2)
        self.assertEqual(len(registry["claims"]), 1)

    def test_applies_architecture_regression_cooldown_and_daily_limits(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            no_architecture = claim_official_probe(
                root / "no_arch.json",
                source_sha256="a",
                relative_regression=0.01,
                evidence={**_evidence(), "architecture_sensitive": False},
                policy=self._policy(interval=1),
                now=1000,
            )
            too_slow = claim_official_probe(
                root / "slow.json",
                source_sha256="b",
                relative_regression=0.11,
                evidence=_evidence(),
                policy=self._policy(interval=1),
                now=1000,
            )

            cooldown_path = root / "cooldown.json"
            claimed = claim_official_probe(
                cooldown_path,
                source_sha256="c",
                relative_regression=0.01,
                evidence=_evidence(),
                policy=self._policy(interval=1, cooldown_seconds=60),
                now=1000,
            )
            cooled_down = claim_official_probe(
                cooldown_path,
                source_sha256="d",
                relative_regression=0.01,
                evidence=_evidence(),
                policy=self._policy(interval=1, cooldown_seconds=60),
                now=1010,
            )

            daily_path = root / "daily.json"
            daily_first = claim_official_probe(
                daily_path,
                source_sha256="e",
                relative_regression=0.01,
                evidence=_evidence(),
                policy=self._policy(interval=1, max_per_day=1),
                now=1000,
            )
            daily_second = claim_official_probe(
                daily_path,
                source_sha256="f",
                relative_regression=0.01,
                evidence=_evidence(),
                policy=self._policy(interval=1, max_per_day=1),
                now=1001,
            )

        self.assertIn("no changed", no_architecture["reason"])
        self.assertIn("ceiling", too_slow["reason"])
        self.assertTrue(claimed["claimed"])
        self.assertIn("cooldown", cooled_down["reason"])
        self.assertTrue(daily_first["claimed"])
        self.assertIn("budget exhausted", daily_second["reason"])

    def test_registry_lock_enforces_daily_budget_across_workers(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "concurrent.json"
            policy = self._policy(interval=1, max_per_day=1)

            def claim(index: int) -> dict:
                return claim_official_probe(
                    path,
                    source_sha256=f"source-{index}",
                    relative_regression=0.01,
                    evidence=_evidence(),
                    policy=policy,
                    now=1000,
                )

            with ThreadPoolExecutor(max_workers=8) as executor:
                results = list(executor.map(claim, range(8)))

            registry = json.loads(path.read_text(encoding="utf-8"))

        self.assertEqual(sum(bool(result["claimed"]) for result in results), 1)
        self.assertEqual(len(registry["claims"]), 1)


if __name__ == "__main__":
    unittest.main()
