from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from orchestrator.sol58_pes.runtime_config import (
    RuntimeConfigError,
    load_runtime_config,
    write_resolved_config,
)


class TestRuntimeConfig(unittest.TestCase):
    def setUp(self) -> None:
        self.template = json.loads(
            (Path(__file__).with_name("runtime_config.json")).read_text(
                encoding="utf-8"
            )
        )

    def _load(
        self,
        root: Path,
        *,
        config: dict | None = None,
        environ: dict[str, str] | None = None,
    ) -> dict:
        repo = root / "atrex-kernel-agent"
        task = repo / "orchestrator" / "sol58_pes"
        task.mkdir(parents=True)
        config_path = task / "runtime_config.json"
        (task / "task_spec.json").write_text(
            (Path(__file__).with_name("task_spec.json")).read_text(encoding="utf-8"),
            encoding="utf-8",
        )
        config_path.write_text(
            json.dumps(config if config is not None else self.template),
            encoding="utf-8",
        )
        return load_runtime_config(
            config_path,
            repo_root=repo,
            task_dir=task,
            environ={} if environ is None else environ,
        )

    def test_rejects_unknown_and_missing_keys(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            unknown = json.loads(json.dumps(self.template))
            unknown["search"]["unknown"] = 1
            with self.assertRaisesRegex(RuntimeConfigError, "unknown"):
                self._load(root / "unknown", config=unknown)

            missing = json.loads(json.dumps(self.template))
            del missing["search"]["concurrency"]
            with self.assertRaisesRegex(RuntimeConfigError, "missing"):
                self._load(root / "missing", config=missing)

    def test_validates_boolean_number_and_range_values(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            invalid_bool = json.loads(json.dumps(self.template))
            invalid_bool["official"]["enabled"] = "yes"
            with self.assertRaisesRegex(RuntimeConfigError, "expected boolean"):
                self._load(root / "boolean", config=invalid_bool)

            invalid_rate = json.loads(json.dumps(self.template))
            invalid_rate["search"]["cutedsl_generation_rate"] = 1.1
            with self.assertRaisesRegex(RuntimeConfigError, "must be <= 1"):
                self._load(root / "rate", config=invalid_rate)

            invalid_delta = json.loads(json.dumps(self.template))
            invalid_delta["official"]["provisional_floor_delta"] = -0.1
            with self.assertRaisesRegex(RuntimeConfigError, "must be >= 0"):
                self._load(root / "delta", config=invalid_delta)

    def test_environment_overrides_are_typed_and_audited(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            result = self._load(
                Path(tmp),
                environ={
                    "SOL58_LOCAL_BEST_GATE": "off",
                    "SOL58_LOCAL_REPEAT_COUNT": "5",
                    "SOL58_MEASUREMENT_PROFILE": "official",
                    "SOL58_OFFICIAL_MAX_LOCAL_LATENCY_MS": "0.0064",
                    "SOL58_EXPLORATION_RATE": "0.3",
                    "SOL58_SAMPLING_WEIGHT_POWER": "1.5",
                    "SOL58_PES_STOP_SCORE": "9999",
                    "ATREX_PES_MIN_HOME_PER_ISLAND": "7",
                    "ATREX_PES_MAX_MIGRANT_FRACTION": "0.15",
                    "ATREX_LITELLM_BUFFERED_STREAM": "off",
                },
            )

        self.assertEqual(result["values"]["SOL58_LOCAL_BEST_GATE"], "0")
        self.assertEqual(result["values"]["SOL58_LOCAL_REPEAT_COUNT"], "5")
        self.assertEqual(
            result["values"]["SOL58_MEASUREMENT_PROFILE"],
            "official_v1_1_b200",
        )
        self.assertEqual(
            result["values"]["SOL58_OFFICIAL_MAX_LOCAL_LATENCY_MS"],
            "0.0064",
        )
        self.assertEqual(result["values"]["SOL58_EXPLORATION_RATE"], "0.3")
        self.assertEqual(result["values"]["SOL58_SAMPLING_WEIGHT_POWER"], "1.5")
        self.assertEqual(result["values"]["SOL58_PES_STOP_SCORE"], "9999.0")
        self.assertEqual(result["values"]["ATREX_PES_MIN_HOME_PER_ISLAND"], "7")
        self.assertEqual(result["values"]["ATREX_PES_MAX_MIGRANT_FRACTION"], "0.15")
        self.assertEqual(result["values"]["ATREX_LITELLM_BUFFERED_STREAM"], "0")
        self.assertEqual(
            result["records"]["SOL58_LOCAL_REPEAT_COUNT"]["source"],
            "environment",
        )

    def test_task_identity_comes_from_task_spec(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            result = self._load(Path(tmp))

        self.assertEqual(result["values"]["SOL58_OFFICIAL_KERNEL_ID"], "58")
        self.assertEqual(result["records"]["SOL58_TARGET_SCORE"]["source"], "task_spec")
        self.assertEqual(
            result["task_spec"]["leaderboard_url"],
            "https://research.nvidia.com/benchmarks/sol-execbench/leaderboard/kernel/58/B200",
        )

    def test_rejects_environment_task_identity_drift(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(RuntimeConfigError, "conflicts with task spec"):
                self._load(Path(tmp), environ={"SOL58_OFFICIAL_KERNEL_ID": "59"})

    def test_auto_paths_use_repository_siblings(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            executable = root / "sol-execbench" / ".venv" / "bin" / "sol-execbench"
            executable.parent.mkdir(parents=True)
            executable.write_text("#!/bin/sh\n", encoding="utf-8")
            result = self._load(root)

        values = result["values"]
        self.assertEqual(
            values["MLSYS26_FLASHINFER_CONTEST_ROOT"],
            str(root / "mlsys26-flashinfer-contest"),
        )
        self.assertEqual(values["SOL_EXECBENCH"], str(executable))
        self.assertEqual(
            values["SOL58_PROBLEM_DIR"],
            str(
                root
                / "sol-problems"
                / "058_moe_expert_token_radix_sort_with_prefix_sum"
            ),
        )

    def test_architecture_islands_require_eight_islands(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(RuntimeConfigError, "at least 8 islands"):
                self._load(
                    Path(tmp),
                    environ={
                        "ATREX_PES_ARCHITECTURE_ISLANDS": "true",
                        "SOL58_NUM_ISLANDS": "4",
                    },
                )

    def test_writes_shell_exports_and_resolved_audit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            result = self._load(root / "input")
            env_path = root / "runtime.env"
            audit_path = root / "runtime.json"

            write_resolved_config(result, env_path, audit_path)

            self.assertIn("export SOL58_NUM_ISLANDS=8", env_path.read_text())
            self.assertEqual(
                json.loads(audit_path.read_text())["records"]["SOL58_NUM_ISLANDS"][
                    "source"
                ],
                "config",
            )


if __name__ == "__main__":
    unittest.main()
