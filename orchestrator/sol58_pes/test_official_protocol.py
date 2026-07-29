from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from orchestrator.sol58_pes.official_protocol import (
    OfficialApiConfig,
    build_official_submission,
    get_official_submission,
    official_compile_options,
    poll_official_submission,
)


def _config(**overrides: object) -> OfficialApiConfig:
    values: dict[str, object] = {
        "base_url": "https://example.invalid",
        "request_timeout": 10.0,
        "poll_interval": 2.0,
        "pending_result_grace": 5.0,
        "async_refresh_delay": 60.0,
        "poll_timeout": 10.0,
        "kernel_id": 58,
        "gpu_type": "B200",
        "evaluation_stack_version": "v1.1",
        "terminal_statuses": frozenset({"COMPLETED", "FAILED"}),
    }
    values.update(overrides)
    return OfficialApiConfig(**values)  # type: ignore[arg-type]


class TestOfficialProtocol(unittest.TestCase):
    def test_compile_options_strip_local_architecture_flags(self) -> None:
        options = official_compile_options(
            {
                "spec": {
                    "compile_options": {
                        "cuda_cflags": ["-O3", "-arch=sm_103", "-gencode=x"]
                    }
                }
            }
        )

        self.assertEqual(options["cuda_cflags"], ["-O3"])
        self.assertEqual(options["ld_flags"], ["-lcuda"])

    def test_build_submission_embeds_declared_sources(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            (workspace / "kernel.cu").write_text("// kernel", encoding="utf-8")
            (workspace / "solution.json").write_text(
                json.dumps(
                    {
                        "spec": {
                            "languages": ["cuda_cpp"],
                            "compile_options": {"cuda_cflags": ["-arch=sm_103"]},
                        },
                        "sources": [{"path": "kernel.cu"}],
                    }
                ),
                encoding="utf-8",
            )
            submission = build_official_submission(
                workspace,
                gpu_type="B200",
                cute_language="cute_dsl",
                author="tester",
            )

        self.assertEqual(submission["author"], "tester")
        self.assertEqual(submission["spec"]["target_hardware"], ["B200"])
        self.assertEqual(submission["sources"][0]["content"], "// kernel")

    def test_submission_lookup_falls_back_to_filtered_list(self) -> None:
        paths: list[str] = []

        def request(method: str, path: str, token: str, **kwargs: object) -> dict:
            paths.append(path)
            if path.startswith("/api/submissions/42"):
                return {"data": {"id": 42, "status": "PENDING_RESULT"}}
            return {
                "data": {
                    "submissions": [{"id": 42, "status": "COMPLETED", "sol_score": 0.9}]
                }
            }

        result = get_official_submission(
            42,
            "token",
            config=_config(),
            request=request,
        )

        self.assertEqual(result["status"], "COMPLETED")
        self.assertEqual(result["sol_score"], 0.9)
        self.assertEqual(len(paths), 2)

    def test_poll_defers_result_beyond_deadline_without_sleeping(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            sleeps: list[float] = []
            result = poll_official_submission(
                42,
                "token",
                Path(tmp),
                config=_config(
                    poll_timeout=1.0,
                    terminal_statuses=frozenset(
                        {"COMPLETED", "FAILED", "DEFERRED_RESULT"}
                    ),
                ),
                get_submission=lambda *args: {
                    "id": 42,
                    "status": "PENDING_RESULT",
                    "result_available_at": "1970-01-01T00:03:20+00:00",
                },
                upload=None,
                poll_timeout=None,
                sleep=sleeps.append,
                wall_time=lambda: 100.0,
            )

        self.assertEqual(result["status"], "DEFERRED_RESULT")
        self.assertEqual(sleeps, [])


if __name__ == "__main__":
    unittest.main()
