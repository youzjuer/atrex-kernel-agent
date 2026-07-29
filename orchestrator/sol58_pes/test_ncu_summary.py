from __future__ import annotations

import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from orchestrator.sol58_pes import ncu_summary

CUDA_SOURCE = """
#include <torch/extension.h>
__global__ void histogram_kernel(const int *input) {}
void run(torch::Tensor a, torch::Tensor b, torch::Tensor c) {}
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {}
"""


class TestNcuSummary(unittest.TestCase):
    def _workspace(self, root: Path) -> Path:
        workspace = root / "eval"
        workspace.mkdir()
        workloads = [
            {"uuid": "small", "axes": {"batch_size": 1, "seq_len": 128}},
            {"uuid": "slow", "axes": {"batch_size": 64, "seq_len": 128}},
        ]
        (workspace / "workload.jsonl").write_text(
            "\n".join(json.dumps(item) for item in workloads) + "\n"
        )
        (workspace / "definition.json").write_text("{}\n")
        (workspace / "reference.py").write_text("def run(x): return x\n")
        (workspace / "solution.json").write_text("{}\n")
        (workspace / "kernel.cu").write_text(CUDA_SOURCE)
        return workspace

    def test_extracts_cuda_and_cute_kernel_names(self) -> None:
        cuda = """
        template<int N> __global__ __launch_bounds__(256) void tiled(int *x) {}
        __global__ void scatter() {}
        """
        cute = "@cute.kernel\ndef cute_sort(x):\n    pass\n"
        self.assertEqual(
            ncu_summary.extract_kernel_names(cuda, "cuda_cpp"), ["tiled", "scatter"]
        )
        self.assertEqual(
            ncu_summary.extract_kernel_names(cute, "cute_dsl"), ["cute_sort"]
        )

    def test_slowest_workload_is_selected(self) -> None:
        lines = [
            json.dumps({"uuid": "a", "axes": {"n": 1}}),
            json.dumps({"uuid": "b", "axes": {"n": 2}}),
        ]
        index, descriptor, _ = ncu_summary.select_workload(
            lines,
            [
                {"index": 0, "latency_ms": 0.01},
                {"index": 1, "latency_ms": 0.03},
            ],
            "slowest",
        )
        self.assertEqual(index, 1)
        self.assertEqual(descriptor["uuid"], "b")
        self.assertEqual(descriptor["selection"], "slowest_measured")

    def test_profile_is_structured_and_cached(self) -> None:
        raw_metrics = {
            "__kernel_name__": "histogram_kernel(int const*)",
            "launch__grid_size": 64,
            "launch__block_size": 256,
            "launch__registers_per_thread": 48,
            "sm__throughput.avg.pct_of_peak_sustained_elapsed": 22.5,
        }
        classification = {
            "findings": [
                {
                    "pattern": "A",
                    "label": "Small grid / SM idle",
                    "confidence": "high",
                    "evidence": "waves_per_multiprocessor=0.20 < 0.5",
                }
            ],
            "symptoms": ["low-sm-utilization"],
        }

        def fake_profile(command, **kwargs):
            self.assertEqual(kwargs["env"]["PATH"].split(":", 1)[0], "/usr/local/bin")
            self.assertNotIn("PYTHONPATH", kwargs["env"])
            if command[0].endswith("sol-execbench"):
                staging = Path(kwargs["cwd"]).parent / "generated_staging"
                staging.mkdir()
                (staging / "eval_driver.py").write_text("pass\n")
                (staging / "benchmark_kernel.so").write_bytes(b"extension")
                return subprocess.CompletedProcess(
                    command, 0, "", f"Staging dir: {staging}\n"
                )
            self.assertIn("application-only", command)
            self.assertIn("--clock-control", command)
            report_base = Path(command[command.index("-o") + 1])
            report_base.with_suffix(".ncu-rep").write_bytes(b"report")
            return subprocess.CompletedProcess(command, 0, "profiled", "")

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = self._workspace(root)
            arguments = {
                "workspace": workspace,
                "kernel_source": CUDA_SOURCE,
                "source_language": "cuda_cpp",
                "source_sha256": "abc123",
                "measurement_profile": {"id": "profile-a", "seed": 200},
                "per_workload": [
                    {"index": 0, "latency_ms": 0.01},
                    {"index": 1, "latency_ms": 0.03},
                ],
                "cache_root": root / "cache",
                "sol_execbench": "/usr/local/bin/sol-execbench",
                "compile_timeout": 10,
                "run_timeout": 10,
            }
            with (
                mock.patch.object(
                    ncu_summary.shutil, "which", return_value="/usr/local/bin/ncu"
                ),
                mock.patch.object(
                    ncu_summary, "_run_profile_command", side_effect=fake_profile
                ) as run_profile,
                mock.patch.object(
                    ncu_summary,
                    "_parse_report",
                    return_value=(raw_metrics, classification),
                ),
            ):
                first = ncu_summary.collect_ncu_analysis(**arguments)
                second = ncu_summary.collect_ncu_analysis(**arguments)

        self.assertEqual(first["status"], "completed")
        self.assertFalse(first["cache_hit"])
        self.assertEqual(first["workload"]["uuid"], "slow")
        self.assertEqual(first["kernel_name"], "histogram_kernel(int const*)")
        self.assertEqual(first["key_metrics"]["registers_per_thread"], 48)
        self.assertEqual(first["findings"][0]["pattern"], "A")
        self.assertIn("parallel work", first["optimization_implications"][0])
        self.assertTrue(second["cache_hit"])
        self.assertEqual(run_profile.call_count, 2)

    def test_report_parser_retries_transient_failure(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            profile_dir = Path(tmp)
            report_path = profile_dir / "profile.ncu-rep"
            report_path.write_bytes(b"report")
            parser_calls = 0

            def fake_run(command, **kwargs):
                nonlocal parser_calls
                if str(ncu_summary.ANALYZE_REPORTS) in command:
                    parser_calls += 1
                    if parser_calls == 1:
                        return subprocess.CompletedProcess(command, 11, "busy", "")
                    analysis = profile_dir / "analysis"
                    analysis.mkdir(exist_ok=True)
                    (analysis / "metrics_key_summary.json").write_text(
                        json.dumps({"launch__grid_size": 64})
                    )
                    return subprocess.CompletedProcess(command, 0, "parsed", "")
                return subprocess.CompletedProcess(
                    command,
                    0,
                    json.dumps({"findings": [], "symptoms": []}),
                    "",
                )

            with (
                mock.patch.object(ncu_summary.subprocess, "run", side_effect=fake_run),
                mock.patch.object(ncu_summary.time, "sleep") as sleep,
            ):
                metrics, classification = ncu_summary._parse_report(
                    report_path, profile_dir, 10
                )

        self.assertEqual(parser_calls, 2)
        self.assertEqual(metrics["launch__grid_size"], 64)
        self.assertEqual(classification["findings"], [])
        sleep.assert_called_once_with(1)

    def test_report_parser_uses_isolated_csv_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            profile_dir = Path(tmp)
            report_path = profile_dir / "profile.ncu-rep"
            report_path.write_bytes(b"report")
            parser_environments = []

            def fake_run(command, **kwargs):
                parser_environments.append(kwargs["env"])
                if str(ncu_summary.ANALYZE_REPORTS) in command:
                    return subprocess.CompletedProcess(command, -11, "", "")
                if "--import" in command:
                    csv_output = (
                        '"ID","Kernel Name","launch__grid_size",'
                        '"sm__throughput.avg.pct_of_peak_sustained_elapsed"\n'
                        '"","","block","%"\n'
                        '"0","histogram_kernel","64","22.5"\n'
                    )
                    return subprocess.CompletedProcess(command, 0, csv_output, "")
                metrics = json.loads(
                    (profile_dir / "analysis" / "metrics_key_summary.json").read_text()
                )
                self.assertEqual(metrics["launch__grid_size"], 64)
                return subprocess.CompletedProcess(
                    command,
                    0,
                    json.dumps({"findings": [], "symptoms": []}),
                    "",
                )

            with (
                mock.patch.dict(
                    ncu_summary.os.environ,
                    {
                        "PYTHONPATH": "/tmp/loongflow-sitecustomize",
                        "SOL58_NCU_PARSE_RETRIES": "1",
                    },
                ),
                mock.patch.object(
                    ncu_summary.shutil, "which", return_value="/usr/local/bin/ncu"
                ),
                mock.patch.object(ncu_summary.subprocess, "run", side_effect=fake_run),
            ):
                metrics, classification = ncu_summary._parse_report(
                    report_path, profile_dir, 10
                )

        self.assertEqual(metrics["__kernel_name__"], "histogram_kernel")
        self.assertEqual(metrics["launch__grid_size"], 64)
        self.assertEqual(classification["findings"], [])
        self.assertTrue(parser_environments)
        self.assertTrue(
            all("PYTHONPATH" not in environment for environment in parser_environments)
        )
        self.assertTrue(
            all(
                environment.get("PYTHONNOUSERSITE") == "1"
                for environment in parser_environments
            )
        )

    def test_timeout_is_diagnostic_not_an_exception(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = self._workspace(root)
            with (
                mock.patch.object(
                    ncu_summary.shutil, "which", return_value="/usr/local/bin/ncu"
                ),
                mock.patch.object(
                    ncu_summary,
                    "_run_profile_command",
                    side_effect=subprocess.TimeoutExpired(["ncu"], 3),
                ),
            ):
                result = ncu_summary.collect_ncu_analysis(
                    workspace=workspace,
                    kernel_source=CUDA_SOURCE,
                    source_language="cuda_cpp",
                    source_sha256="abc123",
                    measurement_profile={"id": "profile-a"},
                    per_workload=[],
                    cache_root=root / "cache",
                    sol_execbench="sol-execbench",
                    compile_timeout=10,
                    run_timeout=10,
                )

            failure_path = Path(result["artifacts"]["failure"])
            self.assertTrue(failure_path.is_file())
            preserved = json.loads(failure_path.read_text())

        self.assertEqual(result["status"], "timeout")
        self.assertEqual(result["timeout_s"], 3)
        self.assertEqual(preserved["status"], "timeout")

    def test_policy_can_limit_expensive_profiles(self) -> None:
        self.assertEqual(
            ncu_summary.should_profile(
                enabled=True, policy="local_best", improved=False, iteration=3
            ),
            (False, "not_local_best"),
        )
        with mock.patch.dict(
            ncu_summary.os.environ, {"SOL58_NCU_PROFILE_INTERVAL": "5"}
        ):
            self.assertEqual(
                ncu_summary.should_profile(
                    enabled=True, policy="periodic", improved=False, iteration=10
                ),
                (True, "local_best_or_periodic"),
            )


if __name__ == "__main__":
    unittest.main()
