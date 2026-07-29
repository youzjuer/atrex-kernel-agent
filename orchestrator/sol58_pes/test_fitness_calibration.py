from __future__ import annotations

import json
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest import mock

from orchestrator.sol58_pes.fitness_calibration import (
    CalibrationContext,
    FitnessCalibration,
    official_fitness_anchor,
    project_provisional_score,
)


def _calibration(root: Path, profile_id: str = "profile-a") -> FitnessCalibration:
    context = CalibrationContext(
        root=root,
        target_latency_ms=0.01,
        evaluation_stack_version="v1.1",
        gpu_type="B200",
        half_life_hours=24.0,
        measurement_profile=lambda: {"id": profile_id, "name": "official"},
        parse_traces=lambda *args, **kwargs: {},
        official_status=lambda row: str(row.get("status") or "").upper(),
        parse_timestamp=lambda value: float(value) if value is not None else None,
    )
    return FitnessCalibration(context)


class TestFitnessCalibration(unittest.TestCase):
    def test_ratio_loading_is_profile_scoped_and_recency_weighted(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            calibration = _calibration(Path(tmp))
            calibration.path.parent.mkdir(parents=True)
            rows = [
                {
                    "created_at": 1000.0,
                    "local_score": 1.0,
                    "official_score": 0.5,
                    "measurement_profile_id": "old-profile",
                    "eval_stack": "v1.1",
                    "gpu_type": "B200",
                },
                {
                    "created_at": 1900.0,
                    "local_score": 1.0,
                    "official_score": 0.8,
                    "measurement_profile_id": "profile-a",
                    "eval_stack": "v1.1",
                    "gpu_type": "B200",
                },
            ]
            calibration.path.write_text(
                "\n".join(json.dumps(row) for row in rows) + "\n",
                encoding="utf-8",
            )
            with mock.patch(
                "orchestrator.sol58_pes.fitness_calibration.time.time",
                return_value=2000.0,
            ):
                ratio = calibration.load_ratio(environ={})

        self.assertEqual(ratio, 0.8)

    def test_concurrent_records_remain_valid_json_lines(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            calibration = _calibration(Path(tmp))

            def record(index: int) -> None:
                calibration.record(
                    local_score=1.0,
                    official_score=0.8 + index / 100.0,
                    local_latency_ms=0.007,
                    official_latency_ms=0.006,
                    submission_id=index,
                    measurement_profile_id=None,
                    submission_reason="test",
                )

            with ThreadPoolExecutor(max_workers=8) as executor:
                list(executor.map(record, range(8)))
            rows = [
                json.loads(line)
                for line in calibration.path.read_text(encoding="utf-8").splitlines()
            ]

        self.assertEqual(len(rows), 8)
        self.assertEqual({row["submission_id"] for row in rows}, set(range(8)))
        self.assertTrue(
            all(row["measurement_profile_id"] == "profile-a" for row in rows)
        )

    def test_projection_is_monotonic_but_never_certifies_target(self) -> None:
        low = project_provisional_score(0.91, target=0.95, floor=0.90)
        high = project_provisional_score(1.1, target=0.95, floor=0.90)

        self.assertLess(low, high)
        self.assertLess(high, 0.95)

    def test_anchor_prefers_completed_official_result(self) -> None:
        anchor = official_fitness_anchor(
            {
                "official_status": "COMPLETED",
                "official_score": 0.91,
                "latency_ms_median": 0.007,
                "official_submission_id": 42,
                "official_anchor_score": 0.8,
                "official_anchor_latency_ms": 0.008,
            }
        )

        self.assertEqual(anchor["source"], "completed_local_best")
        self.assertEqual(anchor["submission_id"], 42)


if __name__ == "__main__":
    unittest.main()
