#!/usr/bin/env python3
"""Focused tests for deferred official-v1.1 fitness handling."""

from __future__ import annotations

import json
import tempfile
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


class TestLocalBestGate(unittest.TestCase):
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

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            program = root / "candidate.cu"
            program.write_text("#include <torch/extension.h>\nPYBIND11_MODULE(x, m) {}\n")

            with mock.patch.object(evaluator, "EVAL_ROOT", root / "eval"), mock.patch.object(
                evaluator, "LOCAL_REPEAT_COUNT", 3
            ), mock.patch.object(evaluator, "LOCAL_BEST_GATE", True), mock.patch.object(
                evaluator, "OFFICIAL_FITNESS", True
            ), mock.patch.object(
                evaluator,
                "_load_local_best",
                return_value={"latency_ms_median": 0.0075, "kernel_sha256": "best"},
            ), mock.patch.object(evaluator, "_save_local_best") as save_best, mock.patch.object(
                evaluator, "_copy_problem_files"
            ), mock.patch.object(evaluator, "_write_solution"), mock.patch.object(
                evaluator, "_run_sol_execbench", return_value=proc
            ) as local_run, mock.patch.object(
                evaluator, "_parse_traces", side_effect=parsed_runs
            ), mock.patch.object(
                evaluator, "_load_workload_count", return_value=16
            ), mock.patch.object(
                evaluator, "_load_official_calibration_ratio", return_value=1.0
            ), mock.patch.object(evaluator, "_submit_official") as submit:
                result = evaluator.evaluate(str(program))

        self.assertEqual(result["status"], "success")
        self.assertEqual(result["metrics"]["official"]["status"], "SKIPPED_NOT_LOCAL_BEST")
        self.assertAlmostEqual(result["metrics"]["latency_ms_geomean"], 0.008)
        self.assertAlmostEqual(result["score"], evaluator.TARGET_LATENCY_MS / 0.008)
        self.assertEqual(
            result["metrics"]["official"]["fitness_source"],
            "provisional_local_proxy",
        )
        self.assertEqual(local_run.call_count, 3)
        save_best.assert_not_called()
        submit.assert_not_called()

    def test_same_kernel_is_not_resubmitted_after_a_faster_remeasurement(self) -> None:
        parsed_runs = [_parsed_result(value) for value in (0.0073, 0.0074, 0.0075)]
        proc = SimpleNamespace(returncode=0, stdout="", stderr="")

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            program = root / "candidate.cu"
            source = "#include <torch/extension.h>\nPYBIND11_MODULE(x, m) {}\n"
            program.write_text(source)
            incumbent_latency = 0.0076

            with mock.patch.object(evaluator, "EVAL_ROOT", root / "eval"), mock.patch.object(
                evaluator, "LOCAL_REPEAT_COUNT", 3
            ), mock.patch.object(evaluator, "LOCAL_BEST_GATE", True), mock.patch.object(
                evaluator, "OFFICIAL_FITNESS", True
            ), mock.patch.object(
                evaluator,
                "_load_local_best",
                return_value={
                    "latency_ms_median": incumbent_latency,
                    "local_score": evaluator.TARGET_LATENCY_MS / incumbent_latency,
                    "kernel_sha256": evaluator._kernel_source_hash(source),
                },
            ), mock.patch.object(evaluator, "_save_local_best") as save_best, mock.patch.object(
                evaluator, "_copy_problem_files"
            ), mock.patch.object(evaluator, "_write_solution"), mock.patch.object(
                evaluator, "_run_sol_execbench", return_value=proc
            ), mock.patch.object(
                evaluator, "_parse_traces", side_effect=parsed_runs
            ), mock.patch.object(
                evaluator, "_load_workload_count", return_value=16
            ), mock.patch.object(evaluator, "_submit_official") as submit:
                result = evaluator.evaluate(str(program))

        self.assertEqual(result["status"], "success")
        self.assertEqual(result["metrics"]["official"]["status"], "SKIPPED_SAME_AS_LOCAL_BEST")
        self.assertTrue(result["metrics"]["local_best"]["same_kernel"])
        self.assertFalse(result["metrics"]["local_best"]["strictly_improved"])
        self.assertAlmostEqual(result["score"], evaluator.TARGET_LATENCY_MS / incumbent_latency)
        save_best.assert_not_called()
        submit.assert_not_called()


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
        ), mock.patch.object(evaluator, "OFFICIAL_ASYNC_SUBMIT", True), mock.patch.object(
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
        self.assertEqual(request.call_args.args[:3], ("POST", "/api/submissions/upload", "test-token"))
        poll.assert_not_called()

    def test_future_result_is_deferred_without_sleeping(self) -> None:
        available_at = datetime.fromtimestamp(time.time() + 3600, timezone.utc).isoformat()
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
        available_at = datetime.fromtimestamp(time.time() + 3600, timezone.utc).isoformat()
        payload = {
            "id": 123,
            "status": "PENDING_RESULT",
            "result_available_at": available_at,
        }

        with mock.patch.object(evaluator, "_http_json", return_value=payload) as request:
            result = evaluator._get_official_submission(123, "test-token")

        self.assertEqual(result["status"], "PENDING_RESULT")
        request.assert_called_once_with("GET", "/api/submissions/123", "test-token")


class TestProvisionalFitness(unittest.TestCase):
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
            program.write_text("#include <torch/extension.h>\nPYBIND11_MODULE(x, m) {}\n")

            patches = (
                mock.patch.object(evaluator, "EVAL_ROOT", root / "eval"),
                mock.patch.object(evaluator, "OFFICIAL_FITNESS", True),
                mock.patch.object(evaluator, "OFFICIAL_MIN_LOCAL_SCORE", 0.0),
                mock.patch.object(evaluator, "OFFICIAL_PENDING_SCORE_POLICY", "provisional"),
                mock.patch.object(evaluator, "OFFICIAL_PROVISIONAL_SCORE_CAP", 0.899135),
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
                mock.patch.object(evaluator, "_load_official_calibration_ratio", return_value=1.0),
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
