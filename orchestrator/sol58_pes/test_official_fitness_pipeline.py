from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from typing import Any

from orchestrator.sol58_pes.official_fitness_pipeline import (
    OfficialEvaluationState,
    OfficialFitnessConfig,
    OfficialFitnessHooks,
    evaluate_official_fitness,
)


def _config(**overrides: Any) -> OfficialFitnessConfig:
    values = {
        "target_latency_ms": 0.0068,
        "minimum_local_score": 0.0,
        "maximum_local_latency_ms": 0.0,
        "local_best_gate": True,
        "pending_score_policy": "local_proxy",
        "provisional_projection_floor": 0.899,
        "evaluation_stack_version": "v1.1",
        "gpu_type": "B200",
        "submission_mode": "private",
        "refreshable_statuses": frozenset({"PENDING", "DEFERRED_RESULT"}),
    }
    values.update(overrides)
    return OfficialFitnessConfig(**values)


def _state(workspace: Path, **overrides: Any) -> OfficialEvaluationState:
    values = {
        "workspace": workspace,
        "kernel_source": "// candidate",
        "source_language": "cuda_cpp",
        "local_score": 1.1,
        "local_latency_ms": 0.006,
        "gate_latency_ms": 0.006,
        "best_latency_before": 0.0065,
        "local_best_before": None,
        "same_as_local_best": False,
        "passes_official_gate": True,
        "reuse_cached_official": False,
        "gate_result": {},
        "official_probe": {"claimed": False},
        "new_local_best_record": None,
        "measurement_profile": {"id": "profile"},
        "metrics": {},
        "artifacts": {},
        "local_summary": "Local measurement passed.",
        "started_at": 0.0,
    }
    values.update(overrides)
    return OfficialEvaluationState(**values)


def _hooks(
    *,
    anchor: dict[str, Any] | None = None,
    submission: dict[str, Any] | None = None,
    submissions: list[dict[str, Any]] | None = None,
) -> OfficialFitnessHooks:
    submitted = submissions if submissions is not None else []

    def result(status: str, summary: str, score: float, **kwargs: Any) -> dict:
        return {"status": status, "summary": summary, "score": score, **kwargs}

    def submit(*args: Any, **kwargs: Any) -> dict[str, Any]:
        submitted.append({"args": args, "kwargs": kwargs})
        return dict(submission or {})

    return OfficialFitnessHooks(
        result=result,
        fitness_anchor=lambda local_best: anchor,
        record_authoritative_source=lambda *args, **kwargs: None,
        anchored_provisional_score=lambda *args: (
            0.88,
            {"source": "calibration", "search_score": 0.91},
        ),
        submit_official=submit,
        official_status=lambda data: str(data.get("status") or "UNKNOWN").upper(),
        save_local_best=lambda record, source: True,
        record_calibration=lambda **kwargs: None,
        local_provisional_score=lambda score: score,
        provisional_official_score=lambda score: (score * 0.8, 0.8),
    )


class TestOfficialFitnessPipeline(unittest.TestCase):
    def test_reuses_completed_local_best_without_submission(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            submissions: list[dict[str, Any]] = []
            state = _state(
                Path(tmp),
                local_best_before={"official_latency_ms": 0.0061},
                same_as_local_best=True,
            )
            result = evaluate_official_fitness(
                state,
                _config(),
                _hooks(
                    anchor={
                        "source": "completed_local_best",
                        "score": 0.91,
                        "submission_id": 42,
                    },
                    submissions=submissions,
                ),
            )

        self.assertEqual(result["score"], 0.91)
        self.assertEqual(
            result["metrics"]["official"]["status"], "REUSED_LOCAL_BEST_OFFICIAL"
        )
        self.assertEqual(submissions, [])

    def test_gate_rejection_returns_provisional_without_submission(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            submissions: list[dict[str, Any]] = []
            state = _state(
                Path(tmp),
                passes_official_gate=False,
                gate_result={"status": "confirmed_slower"},
            )
            result = evaluate_official_fitness(
                state,
                _config(),
                _hooks(submissions=submissions),
            )

        self.assertEqual(result["score"], 0.88)
        self.assertEqual(
            result["metrics"]["official"]["status"], "SKIPPED_NOT_LOCAL_BEST"
        )
        self.assertEqual(submissions, [])

    def test_pending_submission_returns_searchable_provisional_score(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            submissions: list[dict[str, Any]] = []
            state = _state(Path(tmp))
            result = evaluate_official_fitness(
                state,
                _config(),
                _hooks(
                    submission={
                        "id": 77,
                        "status": "PENDING",
                        "upstream_status": "QUEUED",
                    },
                    submissions=submissions,
                ),
            )

        self.assertEqual(result["status"], "success")
        self.assertAlmostEqual(
            result["score"], _config().target_latency_ms / state.gate_latency_ms
        )
        self.assertEqual(result["metrics"]["official"]["fitness_source"], "provisional")
        self.assertEqual(len(submissions), 1)

    def test_strict_local_latency_threshold_blocks_upload_at_boundary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            submissions: list[dict[str, Any]] = []
            state = _state(Path(tmp), local_latency_ms=0.0065)
            result = evaluate_official_fitness(
                state,
                _config(maximum_local_latency_ms=0.0065),
                _hooks(submissions=submissions),
            )

        self.assertEqual(result["score"], 0.88)
        self.assertEqual(
            result["metrics"]["official"]["status"],
            "SKIPPED_LOCAL_LATENCY_THRESHOLD",
        )
        self.assertEqual(submissions, [])
        self.assertIn("not below the strict upload threshold", result["summary"])

    def test_strict_local_latency_threshold_allows_faster_candidate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            submissions: list[dict[str, Any]] = []
            state = _state(Path(tmp), local_latency_ms=0.006499)
            evaluate_official_fitness(
                state,
                _config(maximum_local_latency_ms=0.0065),
                _hooks(
                    submission={"id": 78, "status": "PENDING"},
                    submissions=submissions,
                ),
            )

        self.assertEqual(len(submissions), 1)
        self.assertTrue(submissions[0]["kwargs"]["allow_upload"])

    def test_threshold_cache_lookup_cannot_fall_through_to_upload(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            submissions: list[dict[str, Any]] = []
            state = _state(
                Path(tmp),
                local_latency_ms=0.007,
                reuse_cached_official=True,
            )
            evaluate_official_fitness(
                state,
                _config(maximum_local_latency_ms=0.0065),
                _hooks(
                    submission={"id": 79, "status": "PENDING"},
                    submissions=submissions,
                ),
            )

        self.assertEqual(len(submissions), 1)
        self.assertFalse(submissions[0]["kwargs"]["allow_upload"])


if __name__ == "__main__":
    unittest.main()
