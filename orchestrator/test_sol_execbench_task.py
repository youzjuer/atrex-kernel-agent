from __future__ import annotations

import json
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from orchestrator.sol_execbench_task import (
    TaskSpecError,
    load_task_spec,
    reject_task_identity_overrides,
)


class TestSolExecBenchTaskSpec(unittest.TestCase):
    def setUp(self) -> None:
        self.path = Path(__file__).parent / "sol58_pes" / "task_spec.json"

    def test_loads_identity_and_environment(self) -> None:
        spec = load_task_spec(self.path)

        self.assertEqual(spec.kernel_id, 58)
        self.assertEqual(spec.reference.name, "Recursive")
        self.assertEqual(spec.environment()["SOL58_OFFICIAL_GPU_TYPE"], "B200")
        self.assertIsNone(
            spec.staleness_warning(datetime(2026, 8, 1, tzinfo=timezone.utc))
        )

    def test_warns_when_leaderboard_snapshot_is_stale(self) -> None:
        spec = load_task_spec(self.path)

        warning = spec.staleness_warning(datetime(2026, 8, 20, tzinfo=timezone.utc))

        self.assertIsNotNone(warning)
        self.assertIn("22.0 days old", warning or "")

    def test_rejects_conflicting_identity_override(self) -> None:
        spec = load_task_spec(self.path)

        with self.assertRaisesRegex(TaskSpecError, "conflicts with task spec"):
            reject_task_identity_overrides(spec, {"SOL58_OFFICIAL_KERNEL_ID": "59"})

    def test_rejects_mismatched_leaderboard_url(self) -> None:
        payload = json.loads(self.path.read_text(encoding="utf-8"))
        payload["official"]["leaderboard_url"] = (
            "https://research.nvidia.com/benchmarks/sol-execbench/leaderboard/kernel/59/B200"
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "task.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(TaskSpecError, "must end with"):
                load_task_spec(path)


if __name__ == "__main__":
    unittest.main()
