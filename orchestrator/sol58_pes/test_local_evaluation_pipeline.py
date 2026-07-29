from __future__ import annotations

import contextlib
import tempfile
import unittest
from pathlib import Path
from typing import Any

from orchestrator.sol58_pes.local_evaluation_pipeline import (
    LocalEvaluationConfig,
    LocalEvaluationHooks,
    LocalEvaluationRequest,
    collect_local_evaluation,
)


class TestLocalEvaluationPipeline(unittest.TestCase):
    def test_valid_cache_hit_skips_gpu_execution(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            cache_path = workspace / "cache.json"
            calls: list[str] = []
            cached = {
                "complete": True,
                "processes": [
                    {"returncode": 0, "stdout_tail": "cached", "stderr_tail": ""}
                ],
                "local_latencies_ms": [0.006],
            }

            def unexpected(*args: Any, **kwargs: Any) -> Any:
                raise AssertionError("GPU execution must not run on a cache hit")

            hooks = LocalEvaluationHooks(
                copy_problem_files=lambda path: calls.append("copy"),
                write_solution=lambda path, source, language: calls.append("write"),
                load_workload_count=lambda path: 16,
                cache_path=lambda *args, **kwargs: (
                    cache_path,
                    {"max_local_attempts": 3},
                    "contract",
                ),
                file_lock=lambda path: contextlib.nullcontext(),
                read_json=lambda path: cached,
                cache_is_valid=lambda *args, **kwargs: True,
                ensure_clocks=unexpected,
                run_monitored=unexpected,
                parse_traces=unexpected,
                set_clocks=unexpected,
                result=unexpected,
                tail=lambda value, limit=4000: str(value)[-limit:],
                atomic_write_json=unexpected,
            )
            result = collect_local_evaluation(
                LocalEvaluationRequest(
                    workspace=workspace,
                    kernel_source="// source",
                    source_language="cuda_cpp",
                    source_sha256="source",
                    measurement_profile={"id": "profile"},
                    program_path="candidate.txt",
                    started_at=0.0,
                ),
                LocalEvaluationConfig(
                    repeat_count=1,
                    cache_enabled=True,
                    sol_execbench="sol-execbench",
                    target_latency_ms=0.0068,
                    cache_schema_version=3,
                ),
                hooks,
            )

        self.assertTrue(result["cache_hit"])
        self.assertEqual(result["processes"][0].stdout, "cached")
        self.assertEqual(calls, ["copy", "write"])


if __name__ == "__main__":
    unittest.main()
