#!/usr/bin/env python3
"""Focused tests for deferred official-v1.1 fitness handling."""

from __future__ import annotations

import json
import tempfile
import threading
import time
import unittest
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from orchestrator.sol58_pes import eval_program_sol58 as evaluator


def _parsed_result(latency_ms: float) -> dict:
    return {
        "total": 16,
        "passed": 16,
        "failures": [],
        "latency_ms_geomean": latency_ms,
        "latency_ms_arith_mean": latency_ms,
        "max_abs_err": 0.0,
        "max_rel_err": 0.0,
        "per_workload": [],
    }


_CUTE_SOURCE = """\
import torch
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack

_compiled = None

@cute.kernel
def _sort_kernel(g_in, g_sorted, g_offsets):
    pass

@cute.jit
def _launch(m_in, m_sorted, m_offsets):
    _sort_kernel(m_in, m_sorted, m_offsets).launch(grid=[1, 1, 1], block=[256, 1, 1])

def run(topk_idx, sorted_token_indices, expert_offsets):
    global _compiled
    args = tuple(from_dlpack(t).mark_layout_dynamic() for t in (
        topk_idx, sorted_token_indices, expert_offsets
    ))
    if _compiled is None:
        _compiled = cute.compile(_launch, *args)
    _compiled(*args)
"""


class TestCalibration(unittest.TestCase):
    def test_legacy_completed_workspace_bootstraps_ratio(self) -> None:
        local_latency_ms = 0.01
        official_score = 0.8
        expected = official_score / (evaluator.TARGET_LATENCY_MS / local_latency_ms)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "eval_legacy"
            workspace.mkdir()
            (workspace / "official_submission_result.json").write_text(
                json.dumps(
                    {
                        "id": 123,
                        "status": "COMPLETED",
                        "is_correct": True,
                        "sol_score": official_score,
                    }
                )
            )
            (workspace / "traces.jsonl").write_text(
                json.dumps(
                    {
                        "workload": {"uuid": "test", "axes": {}},
                        "evaluation": {
                            "status": "PASSED",
                            "correctness": {},
                            "performance": {"latency_ms": local_latency_ms},
                        },
                    }
                )
                + "\n"
            )

            with mock.patch.object(evaluator, "EVAL_ROOT", root), mock.patch.dict(
                evaluator.os.environ,
                {"SOL58_OFFICIAL_PROVISIONAL_RATIO": ""},
            ):
                ratio = evaluator._load_official_calibration_ratio()

        self.assertAlmostEqual(ratio, expected)

    def test_calibration_is_scoped_to_measurement_profile(self) -> None:
        now = datetime.now(timezone.utc).isoformat()
        rows = [
            {
                "created_at": now,
                "local_score": 1.0,
                "official_score": 0.8,
                "measurement_profile_id": "profile-a",
                "eval_stack": "v1.1",
                "gpu_type": "B200",
            },
            {
                "created_at": now,
                "local_score": 1.0,
                "official_score": 0.2,
                "measurement_profile_id": "profile-b",
                "eval_stack": "v1.1",
                "gpu_type": "B200",
            },
        ]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "official_cache" / "calibration.jsonl"
            path.parent.mkdir()
            path.write_text("".join(json.dumps(row) + "\n" for row in rows))
            with mock.patch.object(evaluator, "EVAL_ROOT", root), mock.patch.object(
                evaluator,
                "_measurement_profile",
                return_value={"id": "profile-a", "name": "official_v1_1_b200"},
            ), mock.patch.dict(
                evaluator.os.environ, {"SOL58_OFFICIAL_PROVISIONAL_RATIO": ""}
            ):
                ratio = evaluator._load_official_calibration_ratio()

        self.assertEqual(ratio, 0.8)


class TestMeasurementProfile(unittest.TestCase):
    def test_profile_identity_changes_with_codegen_and_clocks(self) -> None:
        common = {
            "SOL58_LOCK_CLOCKS": "1",
            "SOL_EXECBENCH_GPU_CLK_MHZ": "1500",
            "SOL_EXECBENCH_DRAM_CLK_MHZ": "3996",
            "SOL58_CUDA_GENCODE": "-gencode=arch=compute_100,code=sm_100",
            "SOL58_LOCAL_EVAL_STACK_ID": "v1.1",
        }
        with mock.patch.object(
            evaluator, "MEASUREMENT_PROFILE_NAME", "official_v1_1_b200"
        ), mock.patch.dict(evaluator.os.environ, common):
            official = evaluator._measurement_profile()
        with mock.patch.object(
            evaluator, "MEASUREMENT_PROFILE_NAME", "official_v1_1_b200"
        ), mock.patch.dict(
            evaluator.os.environ,
            {**common, "SOL58_CUDA_GENCODE": "-gencode=arch=compute_103,code=sm_103"},
        ):
            native_codegen = evaluator._measurement_profile()
        with mock.patch.object(
            evaluator, "MEASUREMENT_PROFILE_NAME", "official_v1_1_b200"
        ), mock.patch.dict(
            evaluator.os.environ,
            {**common, "SOL58_MEASUREMENT_DEVICE_ID": "GPU-a"},
        ):
            device_a = evaluator._measurement_profile()
        with mock.patch.object(
            evaluator, "MEASUREMENT_PROFILE_NAME", "official_v1_1_b200"
        ), mock.patch.dict(
            evaluator.os.environ,
            {**common, "SOL58_MEASUREMENT_DEVICE_ID": "GPU-b"},
        ):
            device_b = evaluator._measurement_profile()

        self.assertNotEqual(official["id"], native_codegen["id"])
        self.assertNotEqual(device_a["id"], device_b["id"])
        self.assertEqual(official["gpu_clock_mhz"], 1500)

    def test_legacy_best_only_matches_native_profile(self) -> None:
        legacy = {"latency_ms_median": 0.01}

        self.assertTrue(
            evaluator._local_best_profile_matches(
                legacy, {"name": "native", "id": "native-id"}
            )
        )
        self.assertFalse(
            evaluator._local_best_profile_matches(
                legacy, {"name": "official_v1_1_b200", "id": "official-id"}
            )
        )

    def test_clock_drift_is_relocked_and_verified(self) -> None:
        profile = {
            "name": "official_v1_1_b200",
            "id": "profile",
            "lock_clocks": True,
            "gpu_clock_mhz": 1500,
            "dram_clock_mhz": 3996,
        }
        stale = {"gpu_index": "0", "sm_clock_mhz": 2032, "dram_clock_mhz": 3996}
        restored = {"gpu_index": "0", "sm_clock_mhz": 1500, "dram_clock_mhz": 3996}
        proc = SimpleNamespace(returncode=0, stdout="", stderr="")

        with mock.patch.object(
            evaluator, "_measurement_profile", return_value=profile
        ), mock.patch.object(
            evaluator, "_query_gpu_clocks", side_effect=[stale, restored]
        ), mock.patch.object(
            evaluator.subprocess, "run", return_value=proc
        ) as run, mock.patch.object(
            evaluator.time, "sleep"
        ), mock.patch.dict(
            evaluator.os.environ, {"SOL58_AUTO_RELOCK_CLOCKS": "1"}
        ):
            state = evaluator._ensure_measurement_clocks()

        self.assertTrue(state["relocked"])
        self.assertEqual(run.call_count, 2)


class TestStandaloneSource(unittest.TestCase):
    def test_system_includes_are_allowed(self) -> None:
        source = (
            "#include <torch/extension.h>\n"
            "#include <ATen/cuda/CUDAContext.h>\n"
            "#include <cuda_runtime.h>\n"
        )
        self.assertIsNone(evaluator._source_dependency_violation(source))

    def test_external_source_includes_are_rejected(self) -> None:
        cases = (
            '#include "/tmp/executor/kernel.cu"\n',
            '#include "local_header.h"\n',
            "#include <../parent.cuh>\n",
            "#include <parent.cpp>\n",
            "#include PARENT_KERNEL\n",
        )
        for source in cases:
            with self.subTest(source=source):
                self.assertIsNotNone(evaluator._source_dependency_violation(source))

    def test_evaluate_rejects_external_source_before_compilation(self) -> None:
        source = (
            "#include <torch/extension.h>\n"
            '#include "/tmp/executor/kernel.cu"\n'
            "PYBIND11_MODULE(x, m) {}\n"
        )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            program = root / "candidate.cu"
            program.write_text(source)
            with mock.patch.object(
                evaluator, "EVAL_ROOT", root / "eval"
            ), mock.patch.object(evaluator, "_copy_problem_files") as copy_problem:
                result = evaluator.evaluate(str(program))

        self.assertEqual(result["status"], "validation_failed")
        self.assertIn("quoted include", result["summary"])
        copy_problem.assert_not_called()


class TestCuTeDslSource(unittest.TestCase):
    def test_auto_schedule_enforces_requested_cute_ratio(self) -> None:
        for iteration in range(51, 58):
            required = evaluator._required_source_language(
                f"/run/{iteration}/executor/0_0/solution.py",
                "auto",
                cutedsl_rate=0.7,
                schedule_period=10,
            )
            self.assertEqual(required, "cute_dsl")
        for iteration in range(58, 61):
            required = evaluator._required_source_language(
                f"/run/{iteration}/executor/0_0/solution.py",
                "auto",
                cutedsl_rate=0.7,
                schedule_period=10,
            )
            self.assertIsNone(required)

    def test_cute_source_is_detected_and_valid_in_auto_mode(self) -> None:
        language, error = evaluator._candidate_validation_error(_CUTE_SOURCE, "auto")

        self.assertEqual(language, evaluator.SOURCE_LANGUAGE_CUTE)
        self.assertIsNone(error)

    def test_cute_mode_extracts_python_source_from_solution_json(self) -> None:
        raw = json.dumps(
            {
                "sources": [
                    {
                        "path": "kernel.cu",
                        "content": "#include <torch/extension.h>\nPYBIND11_MODULE(x, m) {}\n",
                    },
                    {"path": "kernel.py", "content": _CUTE_SOURCE},
                ]
            }
        )

        source = evaluator._extract_kernel_source(raw, "cute_dsl")

        self.assertEqual(source, _CUTE_SOURCE.strip())

    def test_cute_source_is_rejected_in_cuda_only_mode(self) -> None:
        language, error = evaluator._candidate_validation_error(
            _CUTE_SOURCE, "cuda_cpp"
        )

        self.assertEqual(language, evaluator.SOURCE_LANGUAGE_CUTE)
        self.assertIn("not allowed", error)

    def test_cute_source_rejects_local_python_import(self) -> None:
        source = _CUTE_SOURCE.replace(
            "import torch\n", "import torch\nfrom local_helper import sort\n"
        )

        language, error = evaluator._candidate_validation_error(source, "cute_dsl")

        self.assertEqual(language, evaluator.SOURCE_LANGUAGE_CUTE)
        self.assertIn("not an allowed standalone", error)

    def test_write_solution_uses_cute_dsl_contract(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            evaluator._write_solution(workspace, _CUTE_SOURCE, "cute_dsl")
            solution = json.loads((workspace / "solution.json").read_text())

        self.assertEqual(solution["spec"]["languages"], ["cute_dsl"])
        self.assertEqual(solution["spec"]["entry_point"], "kernel.py::run")
        self.assertEqual(solution["spec"]["dependencies"], ["torch", "cutlass"])
        self.assertTrue(solution["spec"]["destination_passing_style"])
        self.assertNotIn("compile_options", solution["spec"])
        self.assertNotIn("binding", solution["spec"])

    def test_official_cute_submission_embeds_python_source(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            evaluator._write_solution(workspace, _CUTE_SOURCE, "cute_dsl")
            submission = evaluator._build_official_submission(workspace)

        self.assertEqual(submission["spec"]["languages"], ["cute_dsl"])
        self.assertEqual(submission["spec"]["target_hardware"], ["B200"])
        self.assertNotIn("compile_options", submission["spec"])
        self.assertEqual(submission["sources"][0]["path"], "kernel.py")
        self.assertEqual(submission["sources"][0]["content"], _CUTE_SOURCE)

    def test_cute_local_best_is_persisted_as_python(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(
            evaluator, "EVAL_ROOT", Path(tmp)
        ):
            evaluator._save_local_best(
                {
                    "kernel_sha256": evaluator._kernel_source_hash(_CUTE_SOURCE),
                    "source_language": "cute_dsl",
                    "latency_ms_median": 0.01,
                },
                _CUTE_SOURCE,
            )
            record = json.loads(evaluator._local_best_path().read_text())
            kernel_path = Path(record["kernel_path"])
            persisted_source = kernel_path.read_text()
            loaded = evaluator._load_local_best()

        self.assertEqual(record["source_language"], "cute_dsl")
        self.assertEqual(kernel_path.name, "local_best_kernel.py")
        self.assertEqual(persisted_source, _CUTE_SOURCE)
        self.assertEqual(loaded["source_language"], "cute_dsl")

    def test_evaluate_routes_cute_candidate_through_python_solution(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            program = root / "candidate.py"
            program.write_text(_CUTE_SOURCE)
            with mock.patch.object(
                evaluator, "EVAL_ROOT", root / "eval"
            ), mock.patch.object(evaluator, "CODE_LANGUAGE", "auto"), mock.patch.object(
                evaluator, "LOCAL_REPEAT_COUNT", 1
            ), mock.patch.object(
                evaluator, "LOCAL_BEST_GATE", False
            ), mock.patch.object(
                evaluator, "OFFICIAL_FITNESS", False
            ), mock.patch.object(
                evaluator, "_copy_problem_files"
            ), mock.patch.object(
                evaluator,
                "_run_sol_execbench",
                return_value=SimpleNamespace(returncode=0, stdout="", stderr=""),
            ), mock.patch.object(
                evaluator, "_parse_traces", return_value=_parsed_result(0.01)
            ), mock.patch.object(
                evaluator, "_load_workload_count", return_value=16
            ):
                result = evaluator.evaluate(str(program))
                solution = json.loads(
                    (
                        Path(result["artifacts"]["workspace"]) / "solution.json"
                    ).read_text()
                )

        self.assertEqual(result["status"], "success")
        self.assertEqual(result["metrics"]["source_language"], "cute_dsl")
        self.assertEqual(solution["spec"]["languages"], ["cute_dsl"])

    def test_evaluate_rejects_cuda_in_mandatory_cute_slot_before_compilation(
        self,
    ) -> None:
        cuda_source = "#include <torch/extension.h>\nPYBIND11_MODULE(x, m) {}\n"
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            program = root / "51" / "executor" / "0_0" / "candidate.cu"
            program.parent.mkdir(parents=True)
            program.write_text(cuda_source)
            with mock.patch.object(
                evaluator, "EVAL_ROOT", root / "eval"
            ), mock.patch.object(evaluator, "CODE_LANGUAGE", "auto"), mock.patch.object(
                evaluator, "CUTEDSL_GENERATION_RATE", 0.7
            ), mock.patch.object(
                evaluator, "CUTEDSL_SCHEDULE_PERIOD", 10
            ), mock.patch.object(
                evaluator, "_copy_problem_files"
            ) as copy_problem:
                result = evaluator.evaluate(str(program))

        self.assertEqual(result["status"], "validation_failed")
        self.assertIn("mandatory cute_dsl slot", result["summary"])
        copy_problem.assert_not_called()

    def test_authoritative_fitness_registry_is_keyed_by_source_hash(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(
            evaluator, "EVAL_ROOT", Path(tmp)
        ):
            evaluator._record_authoritative_source_fitness(
                _CUTE_SOURCE,
                source_language="cute_dsl",
                official_score=0.9,
                official_latency_ms=0.006,
                local_latency_ms=0.0059,
                submission_id=123,
            )
            registry = json.loads(
                evaluator._authoritative_fitness_registry_path().read_text()
            )

        source_hash = evaluator._kernel_source_hash(_CUTE_SOURCE)
        self.assertEqual(registry["sources"][source_hash]["official_score"], 0.9)
        self.assertEqual(registry["sources"][source_hash]["submission_id"], 123)


class TestLocalBestGate(unittest.TestCase):
    def test_ncu_timeout_is_attached_without_changing_fitness(self) -> None:
        proc = SimpleNamespace(returncode=0, stdout="", stderr="")
        source = "#include <torch/extension.h>\nPYBIND11_MODULE(x, m) {}\n"

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            program = root / "candidate.cu"
            program.write_text(source)
            with mock.patch.object(
                evaluator, "EVAL_ROOT", root / "eval"
            ), mock.patch.object(evaluator, "LOCAL_REPEAT_COUNT", 1), mock.patch.object(
                evaluator, "LOCAL_BEST_GATE", False
            ), mock.patch.object(
                evaluator, "OFFICIAL_FITNESS", False
            ), mock.patch.object(
                evaluator, "_copy_problem_files"
            ), mock.patch.object(
                evaluator, "_write_solution"
            ), mock.patch.object(
                evaluator, "_run_sol_execbench_monitored", return_value=(proc, [])
            ), mock.patch.object(
                evaluator, "_parse_traces", return_value=_parsed_result(0.008)
            ), mock.patch.object(
                evaluator, "_load_workload_count", return_value=16
            ), mock.patch.object(
                evaluator,
                "_ncu_summary_evidence",
                return_value={"enabled": True, "status": "timeout", "timeout_s": 3},
            ):
                result = evaluator.evaluate(str(program))

        self.assertEqual(result["status"], "success")
        self.assertAlmostEqual(result["score"], evaluator.TARGET_LATENCY_MS / 0.008)
        self.assertEqual(result["metrics"]["ncu_analysis"]["status"], "timeout")

    def test_clock_drift_attempt_is_not_used_in_local_median(self) -> None:
        proc = SimpleNamespace(returncode=0, stdout="", stderr="")
        source = "#include <torch/extension.h>\nPYBIND11_MODULE(x, m) {}\n"

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            program = root / "candidate.cu"
            program.write_text(source)
            with mock.patch.object(
                evaluator, "EVAL_ROOT", root / "eval"
            ), mock.patch.object(evaluator, "LOCAL_REPEAT_COUNT", 1), mock.patch.object(
                evaluator, "LOCAL_BEST_GATE", False
            ), mock.patch.object(
                evaluator, "OFFICIAL_FITNESS", False
            ), mock.patch.object(
                evaluator, "_copy_problem_files"
            ), mock.patch.object(
                evaluator, "_write_solution"
            ), mock.patch.object(
                evaluator,
                "_run_sol_execbench_monitored",
                side_effect=[
                    (proc, [{"sm_clock_mhz": 2032, "dram_clock_mhz": 3996}]),
                    (proc, []),
                ],
            ), mock.patch.object(
                evaluator,
                "_parse_traces",
                side_effect=[_parsed_result(0.001), _parsed_result(0.008)],
            ), mock.patch.object(
                evaluator, "_load_workload_count", return_value=16
            ):
                result = evaluator.evaluate(str(program))

        self.assertEqual(result["status"], "success")
        self.assertEqual(result["metrics"]["latency_ms_geomean"], 0.008)
        self.assertEqual(result["metrics"]["local_attempt_count"], 2)
        self.assertEqual(result["metrics"]["clock_rejected_attempt_count"], 1)

    def test_discovery_requires_repeat_count_and_uses_median(self) -> None:
        def write_trace(path: Path, latency_ms: float) -> None:
            path.write_text(
                json.dumps(
                    {
                        "workload": {"uuid": "test", "axes": {}},
                        "evaluation": {
                            "status": "PASSED",
                            "correctness": {},
                            "performance": {"latency_ms": latency_ms},
                        },
                    }
                )
                + "\n"
            )

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            repeated = root / "eval_repeated"
            repeated.mkdir()
            (repeated / "kernel.cu").write_text("repeated kernel\n")
            write_trace(repeated / "traces.jsonl", 0.01)
            write_trace(repeated / "traces_repeat_2.jsonl", 0.03)
            write_trace(repeated / "traces_repeat_3.jsonl", 0.02)

            insufficient = root / "eval_insufficient"
            insufficient.mkdir()
            (insufficient / "kernel.cu").write_text("faster but insufficient\n")
            write_trace(insufficient / "traces.jsonl", 0.001)
            write_trace(insufficient / "traces_repeat_2.jsonl", 0.001)

            external = root / "eval_external_dependency"
            external.mkdir()
            (external / "kernel.cu").write_text(
                '#include "/tmp/executor/kernel.cu"\nPYBIND11_MODULE(x, m) {}\n'
            )
            write_trace(external / "traces.jsonl", 0.0005)
            write_trace(external / "traces_repeat_2.jsonl", 0.0005)
            write_trace(external / "traces_repeat_3.jsonl", 0.0005)

            with mock.patch.object(evaluator, "EVAL_ROOT", root), mock.patch.object(
                evaluator, "LOCAL_REPEAT_COUNT", 3
            ):
                best = evaluator._discover_local_best()

        self.assertIsNotNone(best)
        self.assertAlmostEqual(best["latency_ms_median"], 0.02)
        self.assertEqual(best["sample_count"], 3)

    def test_non_improving_median_does_not_submit(self) -> None:
        parsed_runs = [_parsed_result(value) for value in (0.009, 0.007, 0.008)]
        proc = SimpleNamespace(returncode=0, stdout="", stderr="")
        incumbent_latency = 0.0075
        incumbent_official_score = 0.85

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            program = root / "candidate.cu"
            program.write_text(
                "#include <torch/extension.h>\nPYBIND11_MODULE(x, m) {}\n"
            )

            with mock.patch.object(
                evaluator, "EVAL_ROOT", root / "eval"
            ), mock.patch.object(evaluator, "LOCAL_REPEAT_COUNT", 3), mock.patch.object(
                evaluator, "LOCAL_BEST_GATE", True
            ), mock.patch.object(
                evaluator, "OFFICIAL_FITNESS", True
            ), mock.patch.object(
                evaluator,
                "_load_local_best",
                return_value={
                    "latency_ms_median": incumbent_latency,
                    "kernel_sha256": "best",
                    "official_status": "COMPLETED",
                    "official_score": incumbent_official_score,
                    "official_submission_id": 123,
                },
            ), mock.patch.object(
                evaluator, "_save_local_best"
            ) as save_best, mock.patch.object(
                evaluator, "_copy_problem_files"
            ), mock.patch.object(
                evaluator, "_write_solution"
            ), mock.patch.object(
                evaluator, "_run_sol_execbench", return_value=proc
            ) as local_run, mock.patch.object(
                evaluator, "_parse_traces", side_effect=parsed_runs
            ), mock.patch.object(
                evaluator, "_load_workload_count", return_value=16
            ), mock.patch.object(
                evaluator, "_load_official_calibration_ratio", return_value=1.0
            ), mock.patch.object(
                evaluator, "_submit_official"
            ) as submit:
                result = evaluator.evaluate(str(program))

        self.assertEqual(result["status"], "success")
        self.assertEqual(
            result["metrics"]["official"]["status"], "SKIPPED_NOT_LOCAL_BEST"
        )
        self.assertAlmostEqual(result["metrics"]["latency_ms_geomean"], 0.008)
        self.assertAlmostEqual(
            result["score"],
            incumbent_official_score * incumbent_latency / 0.008,
        )
        self.assertLess(result["score"], incumbent_official_score)
        self.assertEqual(
            result["metrics"]["official"]["fitness_source"],
            "provisional_official_anchor",
        )
        self.assertEqual(local_run.call_count, 3)
        save_best.assert_not_called()
        submit.assert_not_called()

    def test_same_kernel_reuses_record_without_local_rerun(self) -> None:
        parsed_runs = [_parsed_result(value) for value in (0.0073, 0.0074, 0.0075)]
        proc = SimpleNamespace(returncode=0, stdout="", stderr="")

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            program = root / "candidate.cu"
            source = "#include <torch/extension.h>\nPYBIND11_MODULE(x, m) {}\n"
            program.write_text(source)
            incumbent_latency = 0.0076
            incumbent_official_score = 0.85

            with mock.patch.object(
                evaluator, "EVAL_ROOT", root / "eval"
            ), mock.patch.object(evaluator, "LOCAL_REPEAT_COUNT", 3), mock.patch.object(
                evaluator, "LOCAL_BEST_GATE", True
            ), mock.patch.object(
                evaluator, "OFFICIAL_FITNESS", True
            ), mock.patch.object(
                evaluator,
                "_load_local_best",
                return_value={
                    "latency_ms_median": incumbent_latency,
                    "local_score": evaluator.TARGET_LATENCY_MS / incumbent_latency,
                    "kernel_sha256": evaluator._kernel_source_hash(source),
                    "official_status": "COMPLETED",
                    "official_score": incumbent_official_score,
                    "official_submission_id": 123,
                },
            ), mock.patch.object(
                evaluator, "_save_local_best"
            ) as save_best, mock.patch.object(
                evaluator, "_copy_problem_files"
            ) as copy_problem, mock.patch.object(
                evaluator, "_write_solution"
            ), mock.patch.object(
                evaluator, "_run_sol_execbench", return_value=proc
            ) as local_run, mock.patch.object(
                evaluator, "_parse_traces", side_effect=parsed_runs
            ), mock.patch.object(
                evaluator, "_load_workload_count", return_value=16
            ), mock.patch.object(
                evaluator, "_submit_official"
            ) as submit:
                result = evaluator.evaluate(str(program))

        self.assertEqual(result["status"], "success")
        self.assertEqual(
            result["metrics"]["official"]["status"],
            "REUSED_LOCAL_BEST_OFFICIAL",
        )
        self.assertTrue(result["metrics"]["local_best"]["same_kernel"])
        self.assertFalse(result["metrics"]["local_best"]["strictly_improved"])
        self.assertTrue(result["metrics"]["official"]["authoritative"])
        self.assertAlmostEqual(result["score"], incumbent_official_score)
        copy_problem.assert_not_called()
        local_run.assert_not_called()
        save_best.assert_not_called()
        submit.assert_not_called()

    def test_same_kernel_is_remeasured_when_profile_changes(self) -> None:
        source = "#include <torch/extension.h>\nPYBIND11_MODULE(x, m) {}\n"
        profile = {
            "name": "official_v1_1_b200",
            "id": "new-profile",
            "lock_clocks": False,
        }
        incumbent = {
            "latency_ms_median": 0.0066,
            "kernel_sha256": evaluator._kernel_source_hash(source),
            "measurement_profile_id": "old-profile",
            "official_status": "COMPLETED",
            "official_score": 0.856272,
            "official_latency_ms": 0.009261,
            "official_submission_id": 25293,
            "remote_submitted": True,
        }
        proc = SimpleNamespace(returncode=0, stdout="", stderr="")

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            program = root / "candidate.cu"
            program.write_text(source)
            with mock.patch.object(
                evaluator, "EVAL_ROOT", root / "eval"
            ), mock.patch.object(evaluator, "LOCAL_REPEAT_COUNT", 1), mock.patch.object(
                evaluator, "LOCAL_BEST_GATE", True
            ), mock.patch.object(
                evaluator, "OFFICIAL_FITNESS", True
            ), mock.patch.object(
                evaluator, "_measurement_profile", return_value=profile
            ), mock.patch.object(
                evaluator, "_load_local_best", return_value=incumbent
            ), mock.patch.object(
                evaluator, "_save_local_best"
            ) as save_best, mock.patch.object(
                evaluator, "_record_authoritative_source_fitness"
            ), mock.patch.object(
                evaluator, "_copy_problem_files"
            ), mock.patch.object(
                evaluator, "_write_solution"
            ), mock.patch.object(
                evaluator, "_run_sol_execbench", return_value=proc
            ) as local_run, mock.patch.object(
                evaluator, "_parse_traces", return_value=_parsed_result(0.0081)
            ), mock.patch.object(
                evaluator, "_load_workload_count", return_value=16
            ):
                result = evaluator.evaluate(str(program))

        self.assertEqual(result["status"], "success")
        self.assertEqual(result["score"], incumbent["official_score"])
        self.assertTrue(result["metrics"]["profile_recalibration"])
        local_run.assert_called_once()
        saved = save_best.call_args.args[0]
        self.assertEqual(saved["measurement_profile_id"], "new-profile")
        self.assertEqual(saved["official_submission_id"], 25293)
        self.assertEqual(saved["official_score"], 0.856272)
        self.assertEqual(saved["official_anchor_latency_ms"], 0.0081)

    def test_different_kernel_cannot_use_incompatible_best_profile(self) -> None:
        incumbent_source = "#include <torch/extension.h>\nPYBIND11_MODULE(x, m) {}\n"
        candidate_source = incumbent_source + "// distinct\n"
        profile = {
            "name": "official_v1_1_b200",
            "id": "new-profile",
            "lock_clocks": False,
        }
        incumbent = {
            "latency_ms_median": 0.0066,
            "kernel_sha256": evaluator._kernel_source_hash(incumbent_source),
            "measurement_profile_id": "old-profile",
        }

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            program = root / "candidate.cu"
            program.write_text(candidate_source)
            with mock.patch.object(
                evaluator, "EVAL_ROOT", root / "eval"
            ), mock.patch.object(evaluator, "LOCAL_BEST_GATE", True), mock.patch.object(
                evaluator, "_measurement_profile", return_value=profile
            ), mock.patch.object(
                evaluator, "_load_local_best", return_value=incumbent
            ), mock.patch.object(
                evaluator, "_copy_problem_files"
            ) as copy_problem:
                result = evaluator.evaluate(str(program))

        self.assertEqual(result["status"], "framework_error")
        self.assertTrue(result["metrics"]["profile_recalibration_required"])
        copy_problem.assert_not_called()


class TestLocalEvaluationCache(unittest.TestCase):
    def test_non_best_duplicate_reuses_complete_local_result(self) -> None:
        source = "#include <torch/extension.h>\nPYBIND11_MODULE(x, m) {}\n"
        proc = SimpleNamespace(returncode=0, stdout="", stderr="")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            program = root / "candidate.cu"
            program.write_text(source)
            with mock.patch.object(
                evaluator, "EVAL_ROOT", root / "eval"
            ), mock.patch.object(evaluator, "LOCAL_REPEAT_COUNT", 1), mock.patch.object(
                evaluator, "LOCAL_BEST_GATE", False
            ), mock.patch.object(
                evaluator, "OFFICIAL_FITNESS", False
            ), mock.patch.object(
                evaluator, "LOCAL_EVAL_CACHE", True
            ), mock.patch.object(
                evaluator, "_copy_problem_files"
            ), mock.patch.object(
                evaluator, "_write_solution"
            ), mock.patch.object(
                evaluator, "_detect_cuda_gencode_flags", return_value=[]
            ), mock.patch.object(
                evaluator,
                "_run_sol_execbench_monitored",
                return_value=(proc, []),
            ) as local_run, mock.patch.object(
                evaluator, "_parse_traces", return_value=_parsed_result(0.008)
            ) as parse, mock.patch.object(
                evaluator, "_load_workload_count", return_value=16
            ):
                first = evaluator.evaluate(str(program))
                second = evaluator.evaluate(str(program))

        self.assertFalse(first["metrics"]["local_eval_cache"]["hit"])
        self.assertTrue(second["metrics"]["local_eval_cache"]["hit"])
        self.assertEqual(local_run.call_count, 1)
        self.assertEqual(parse.call_count, 1)
        self.assertEqual(first["score"], second["score"])

    def test_cache_key_changes_with_profile_and_contract(self) -> None:
        source = "kernel"
        profile_a = {"id": "a", "name": "native"}
        profile_b = {"id": "b", "name": "native"}
        with mock.patch.object(
            evaluator, "_detect_cuda_gencode_flags", return_value=[]
        ), mock.patch.object(evaluator, "LOCAL_REPEAT_COUNT", 3):
            path_a, _, _ = evaluator._local_evaluation_cache_path(
                source, evaluator.SOURCE_LANGUAGE_CUDA, profile_a
            )
            path_b, _, _ = evaluator._local_evaluation_cache_path(
                source, evaluator.SOURCE_LANGUAGE_CUDA, profile_b
            )
        with mock.patch.object(
            evaluator, "_detect_cuda_gencode_flags", return_value=[]
        ), mock.patch.object(evaluator, "LOCAL_REPEAT_COUNT", 5):
            path_repeat, _, _ = evaluator._local_evaluation_cache_path(
                source, evaluator.SOURCE_LANGUAGE_CUDA, profile_a
            )

        self.assertNotEqual(path_a, path_b)
        self.assertNotEqual(path_a, path_repeat)


class TestConcurrentState(unittest.TestCase):
    def test_slower_writer_cannot_replace_faster_local_best(self) -> None:
        profile = evaluator._measurement_profile()
        barrier = threading.Barrier(2)

        def write(latency: float, source: str) -> None:
            barrier.wait()
            evaluator._save_local_best(
                {
                    "kernel_sha256": evaluator._kernel_source_hash(source),
                    "source_language": "cuda_cpp",
                    "latency_ms_median": latency,
                    "local_score": evaluator.TARGET_LATENCY_MS / latency,
                    "measurement_profile_id": profile["id"],
                    "measurement_profile": profile,
                },
                source,
            )

        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(
            evaluator, "EVAL_ROOT", Path(tmp)
        ):
            threads = [
                threading.Thread(target=write, args=(0.009, "slow")),
                threading.Thread(target=write, args=(0.007, "fast")),
            ]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()
            best = json.loads(evaluator._local_best_path().read_text())

        self.assertEqual(best["latency_ms_median"], 0.007)
        self.assertEqual(best["kernel_sha256"], evaluator._kernel_source_hash("fast"))


class TestOfficialRefresh(unittest.TestCase):
    def test_due_pending_result_populates_authoritative_registry(self) -> None:
        source_hash = evaluator._kernel_source_hash("kernel")
        metadata = {
            "source_sha256": source_hash,
            "source_language": "cuda_cpp",
            "local_score": 0.9,
            "local_latency_ms": 0.007,
            "measurement_profile_id": "profile-a",
        }
        pending = {
            "id": 123,
            "status": "DEFERRED_RESULT",
            "next_refresh_at": datetime.fromtimestamp(
                time.time() - 10, timezone.utc
            ).isoformat(),
            "_atrex": metadata,
        }
        completed = {
            "id": 123,
            "status": "COMPLETED",
            "is_correct": True,
            "sol_score": 0.91,
            "latency_ms": 0.006,
        }
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cache_path = root / "official_cache" / ("a" * 64 + ".json")
            cache_path.parent.mkdir()
            cache_path.write_text(json.dumps(pending))
            with mock.patch.object(evaluator, "EVAL_ROOT", root), mock.patch.object(
                evaluator, "OFFICIAL_FITNESS", True
            ), mock.patch.object(evaluator, "OFFICIAL_CACHE", True), mock.patch.object(
                evaluator, "OFFICIAL_REFRESH_BATCH_SIZE", 1
            ), mock.patch.object(
                evaluator, "_official_token", return_value="token"
            ), mock.patch.object(
                evaluator, "_get_official_submission", return_value=completed
            ):
                report = evaluator._refresh_due_official_results()
                registry = json.loads(
                    evaluator._authoritative_fitness_registry_path().read_text()
                )

        self.assertEqual(report["completed"], 1)
        self.assertEqual(registry["sources"][source_hash]["official_score"], 0.91)


class TestOfficialPolling(unittest.TestCase):
    def test_async_submit_returns_after_upload(self) -> None:
        upload = {
            "data": {
                "submission_id": 123,
                "message": "Submission received and queued for evaluation",
            }
        }

        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(
            evaluator, "OFFICIAL_CACHE", False
        ), mock.patch.object(
            evaluator, "OFFICIAL_ASYNC_SUBMIT", True
        ), mock.patch.object(
            evaluator, "_official_token", return_value="test-token"
        ), mock.patch.object(
            evaluator, "_build_official_submission", return_value={"name": "test"}
        ), mock.patch.object(
            evaluator, "_http_json", return_value=upload
        ) as request, mock.patch.object(
            evaluator, "_poll_official_submission"
        ) as poll:
            result = evaluator._submit_official(Path(tmp), "kernel source")

        self.assertEqual(result["status"], "DEFERRED_RESULT")
        self.assertEqual(result["upstream_status"], "QUEUED")
        self.assertIsNotNone(evaluator._parse_timestamp(result["next_refresh_at"]))
        self.assertEqual(request.call_count, 1)
        self.assertEqual(
            request.call_args.args[:3],
            ("POST", "/api/submissions/upload", "test-token"),
        )
        poll.assert_not_called()

    def test_future_result_is_deferred_without_sleeping(self) -> None:
        available_at = datetime.fromtimestamp(
            time.time() + 3600, timezone.utc
        ).isoformat()
        payload = {
            "id": 123,
            "status": "PENDING_RESULT",
            "result_available_at": available_at,
        }

        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(
            evaluator, "_get_official_submission", return_value=payload
        ), mock.patch.object(evaluator.time, "sleep") as sleep:
            result = evaluator._poll_official_submission(
                123,
                "test-token",
                Path(tmp),
                poll_timeout=30,
            )

        self.assertEqual(result["status"], "DEFERRED_RESULT")
        sleep.assert_not_called()

    def test_future_result_skips_redundant_list_request(self) -> None:
        available_at = datetime.fromtimestamp(
            time.time() + 3600, timezone.utc
        ).isoformat()
        payload = {
            "id": 123,
            "status": "PENDING_RESULT",
            "result_available_at": available_at,
        }

        with mock.patch.object(
            evaluator, "_http_json", return_value=payload
        ) as request:
            result = evaluator._get_official_submission(123, "test-token")

        self.assertEqual(result["status"], "PENDING_RESULT")
        request.assert_called_once_with("GET", "/api/submissions/123", "test-token")


class TestProvisionalFitness(unittest.TestCase):
    def test_above_floor_candidates_remain_ordered_without_certifying_target(
        self,
    ) -> None:
        with mock.patch.object(
            evaluator, "OFFICIAL_TARGET_SCORE", 0.904135
        ), mock.patch.object(
            evaluator, "OFFICIAL_PROVISIONAL_SCORE_CAP", 0.899135
        ), mock.patch.object(
            evaluator, "OFFICIAL_FITNESS", True
        ):
            lower = evaluator._project_provisional_search_score(0.91)
            higher = evaluator._project_provisional_search_score(1.05)
            result = evaluator._result(
                "success",
                "pending",
                higher,
                metrics={
                    "official": {
                        "authoritative": False,
                        "fitness_source": "provisional",
                        "search_score": 1.05,
                    }
                },
            )

        self.assertGreater(higher, lower)
        self.assertLess(higher, 0.904135)
        self.assertEqual(result["metrics"]["search_score"], 1.05)
        self.assertIsNone(result["metrics"]["certified_score"])
        self.assertFalse(result["metrics"]["target_certified"])

    def test_official_anchor_keeps_slower_candidate_below_incumbent(self) -> None:
        local_best = {
            "latency_ms_median": 0.0075,
            "official_status": "COMPLETED",
            "official_score": 0.85,
            "official_submission_id": 123,
        }

        with mock.patch.object(evaluator, "OFFICIAL_PROVISIONAL_SCORE_CAP", 0.899135):
            score, details = evaluator._anchored_provisional_score(
                evaluator.TARGET_LATENCY_MS / 0.008,
                0.008,
                local_best,
            )

        self.assertAlmostEqual(score, 0.85 * 0.0075 / 0.008)
        self.assertLess(score, 0.85)
        self.assertEqual(details["source"], "incumbent_official_anchor")

    def test_local_proxy_keeps_distinct_candidates_ordered(self) -> None:
        with mock.patch.object(evaluator, "OFFICIAL_PROVISIONAL_SCORE_CAP", 0.899135):
            faster = evaluator._local_provisional_score(0.8825)
            slower = evaluator._local_provisional_score(0.8359)

        self.assertGreater(faster, slower)
        self.assertEqual(faster, 0.8825)
        self.assertEqual(slower, 0.8359)

    def test_pending_official_result_is_loongflow_success(self) -> None:
        parsed = {
            "total": 16,
            "passed": 16,
            "failures": [],
            "latency_ms_geomean": 0.008,
            "latency_ms_arith_mean": 0.008,
            "max_abs_err": 0.0,
            "max_rel_err": 0.0,
            "per_workload": [],
        }
        official = {
            "id": 123,
            "status": "DEFERRED_RESULT",
            "evaluation_stack_version": "v1.1",
            "result_available_at": datetime.fromtimestamp(
                time.time() + 3600, timezone.utc
            ).isoformat(),
        }

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            program = root / "candidate.cu"
            program.write_text(
                "#include <torch/extension.h>\nPYBIND11_MODULE(x, m) {}\n"
            )

            patches = (
                mock.patch.object(evaluator, "EVAL_ROOT", root / "eval"),
                mock.patch.object(evaluator, "OFFICIAL_FITNESS", True),
                mock.patch.object(evaluator, "OFFICIAL_MIN_LOCAL_SCORE", 0.0),
                mock.patch.object(
                    evaluator, "OFFICIAL_PENDING_SCORE_POLICY", "provisional"
                ),
                mock.patch.object(
                    evaluator, "OFFICIAL_PROVISIONAL_SCORE_CAP", 0.899135
                ),
                mock.patch.object(evaluator, "_copy_problem_files"),
                mock.patch.object(evaluator, "_write_solution"),
                mock.patch.object(
                    evaluator,
                    "_run_sol_execbench",
                    return_value=SimpleNamespace(returncode=0, stdout="", stderr=""),
                ),
                mock.patch.object(evaluator, "_parse_traces", return_value=parsed),
                mock.patch.object(evaluator, "_load_workload_count", return_value=16),
                mock.patch.object(evaluator, "_submit_official", return_value=official),
                mock.patch.object(
                    evaluator, "_load_official_calibration_ratio", return_value=1.0
                ),
            )
            for patcher in patches:
                patcher.start()
                self.addCleanup(patcher.stop)

            result = evaluator.evaluate(str(program))

        self.assertEqual(result["status"], "success")
        self.assertGreater(result["score"], 0.0)
        self.assertLess(result["score"], evaluator.OFFICIAL_TARGET_SCORE)
        self.assertTrue(result["metrics"]["official"]["provisional"])
        self.assertFalse(result["metrics"]["official"]["authoritative"])
        self.assertEqual(result["metrics"]["official"]["fitness_source"], "provisional")


if __name__ == "__main__":
    unittest.main()
