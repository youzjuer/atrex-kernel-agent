from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from orchestrator.sol58_pes.local_gate import (
    LocalGatePolicy,
    claim_uncertain_challenger,
    classify_delta,
    evaluate_local_best_gate,
    uncertainty_band,
)


def _policy(**overrides: object) -> LocalGatePolicy:
    values: dict[str, object] = {
        "enabled": True,
        "sigma_multiplier": 2.0,
        "relative_noise_floor": 0.001,
        "recheck_pairs": 2,
        "uncertain_relative_tolerance": 0.005,
        "allow_uncertain_official": True,
        "challenger_cooldown_seconds": 60.0,
    }
    values.update(overrides)
    return LocalGatePolicy(**values)  # type: ignore[arg-type]


class TestLocalGate(unittest.TestCase):
    def test_classification_uses_noise_floor(self) -> None:
        band = uncertainty_band(
            [1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0],
            policy=_policy(relative_noise_floor=0.01),
        )

        self.assertEqual(band, 0.01)
        self.assertEqual(classify_delta(-0.02, band), "confirmed_faster")
        self.assertEqual(classify_delta(0.02, band), "confirmed_slower")
        self.assertEqual(classify_delta(0.005, band), "uncertain")

    def test_paired_recheck_confirms_improvement_and_alternates_order(self) -> None:
        measurements = iter([0.89, 1.0, 1.0, 0.90])
        calls: list[str] = []

        def collect(**kwargs: object) -> dict[str, object]:
            workspace = Path(str(kwargs["workspace"]))
            calls.append(workspace.name)
            return {"local_latencies_ms": [next(measurements)]}

        result = evaluate_local_best_gate(
            workspace=Path("/tmp/gate"),
            kernel_source="candidate",
            source_language="cuda_cpp",
            source_sha256="candidate-hash",
            candidate_latencies_ms=[0.9, 0.9, 0.9],
            local_best={
                "kernel_sha256": "incumbent-hash",
                "latency_ms_median": 1.0,
                "repeat_latencies_ms": [1.0, 1.0, 1.0],
            },
            measurement_profile={"id": "profile"},
            program_path="candidate.cu",
            start=0.0,
            policy=_policy(),
            read_local_best_source=lambda _: ("incumbent", "cuda_cpp"),
            kernel_source_hash=lambda source: f"hash-{source}",
            collect_local_evaluation=collect,
            claim_challenger=lambda _: (False, "not needed"),
        )

        self.assertEqual(result["status"], "confirmed_faster")
        self.assertTrue(result["rechecked"])
        self.assertTrue(result["update_local_best"])
        self.assertTrue(result["submit_official"])
        self.assertEqual(
            calls,
            [
                "pair_1_candidate",
                "pair_1_incumbent",
                "pair_2_incumbent",
                "pair_2_candidate",
            ],
        )

    def test_recheck_failure_never_promotes_candidate(self) -> None:
        result = evaluate_local_best_gate(
            workspace=Path("/tmp/gate"),
            kernel_source="candidate",
            source_language="cuda_cpp",
            source_sha256="candidate-hash",
            candidate_latencies_ms=[0.99, 0.99, 0.99],
            local_best={
                "kernel_sha256": "incumbent-hash",
                "latency_ms_median": 1.0,
                "repeat_latencies_ms": [1.0, 1.0, 1.0],
            },
            measurement_profile={"id": "profile"},
            program_path="candidate.cu",
            start=0.0,
            policy=_policy(),
            read_local_best_source=lambda _: ("incumbent", "cuda_cpp"),
            kernel_source_hash=lambda source: source,
            collect_local_evaluation=lambda **_: {
                "error_result": {"summary": "measurement failed"}
            },
            claim_challenger=lambda _: (True, "must not be called"),
        )

        self.assertEqual(result["status"], "recheck_failed")
        self.assertFalse(result["update_local_best"])
        self.assertFalse(result["submit_official"])

    def test_challenger_registry_is_deduplicated_and_cooled_down(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "challengers.json"
            with mock.patch(
                "orchestrator.sol58_pes.local_gate.time.time",
                side_effect=[1000.0, 1001.0, 1061.0],
            ):
                first = claim_uncertain_challenger(
                    path,
                    "a",
                    policy=_policy(),
                    format_timestamp=str,
                )
                duplicate = claim_uncertain_challenger(
                    path,
                    "a",
                    policy=_policy(),
                    format_timestamp=str,
                )
                second = claim_uncertain_challenger(
                    path,
                    "b",
                    policy=_policy(),
                    format_timestamp=str,
                )
            registry = json.loads(path.read_text(encoding="utf-8"))

        self.assertTrue(first[0])
        self.assertIn("already claimed", duplicate[1])
        self.assertTrue(second[0])
        self.assertEqual(set(registry["sources"]), {"a", "b"})


if __name__ == "__main__":
    unittest.main()
