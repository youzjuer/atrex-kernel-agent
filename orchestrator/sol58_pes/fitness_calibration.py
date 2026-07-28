"""Profile-scoped official/local calibration and provisional PES scoring."""

from __future__ import annotations

import json
import math
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping

from orchestrator.sol58_pes.evaluator_state import file_lock, read_json


@dataclass(frozen=True)
class CalibrationContext:
    root: Path
    target_latency_ms: float
    evaluation_stack_version: str
    gpu_type: str
    half_life_hours: float
    measurement_profile: Callable[[], dict[str, Any]]
    parse_traces: Callable[..., dict[str, Any]]
    official_status: Callable[[dict[str, Any]], str]
    parse_timestamp: Callable[[Any], float | None]


class FitnessCalibration:
    def __init__(self, context: CalibrationContext) -> None:
        self.context = context

    @property
    def path(self) -> Path:
        return self.context.root / "official_cache" / "calibration.jsonl"

    def scope(self) -> dict[str, str]:
        return {
            "measurement_profile_id": str(self.context.measurement_profile()["id"]),
            "eval_stack": self.context.evaluation_stack_version,
            "gpu_type": self.context.gpu_type,
        }

    def discover_ratios(self) -> list[tuple[float, float]]:
        scope = self.scope()
        by_submission: dict[str, tuple[float, float]] = {}
        result_paths = sorted(
            self.context.root.glob("eval_*/official_submission_result.json"),
            key=lambda path: path.stat().st_mtime,
        )
        for result_path in result_paths:
            try:
                official = read_json(result_path, {})
                if (
                    self.context.official_status(official) != "COMPLETED"
                    or not official.get("is_correct")
                    or float(official.get("sol_score") or 0.0) <= 0
                ):
                    continue
                if (
                    str(official.get("evaluation_stack_version") or scope["eval_stack"])
                    != scope["eval_stack"]
                ):
                    continue
                if (
                    str(official.get("gpu_type") or scope["gpu_type"])
                    != scope["gpu_type"]
                ):
                    continue

                profile_path = result_path.parent / "measurement_profile.json"
                workspace_profile = (
                    read_json(profile_path, {}) if profile_path.is_file() else {}
                )
                workspace_profile_id = str(workspace_profile.get("id") or "")
                if workspace_profile_id:
                    if workspace_profile_id != scope["measurement_profile_id"]:
                        continue
                elif self.context.measurement_profile()["name"] != "native":
                    continue

                parsed = self.context.parse_traces(result_path.parent)
                total = int(parsed.get("total") or 0)
                passed = int(parsed.get("passed") or 0)
                local_latency_ms = float(parsed.get("latency_ms_geomean") or 0.0)
                if total <= 0 or passed != total or local_latency_ms <= 0:
                    continue
                local_score = self.context.target_latency_ms / local_latency_ms
                submission_id = str(official.get("id") or result_path.parent.name)
                by_submission[submission_id] = (
                    float(official["sol_score"]) / local_score,
                    result_path.stat().st_mtime,
                )
            except (OSError, TypeError, ValueError):
                continue
        return list(by_submission.values())

    def recency_weighted_median(self, samples: list[tuple[float, float]]) -> float:
        if not samples:
            return 1.0
        now = time.time()
        weighted = []
        for ratio, timestamp in samples:
            age_hours = max(0.0, (now - timestamp) / 3600.0)
            weight = 0.5 ** (age_hours / max(1.0, self.context.half_life_hours))
            weighted.append((float(ratio), max(weight, 1e-6)))
        weighted.sort(key=lambda item: item[0])
        threshold = sum(weight for _, weight in weighted) / 2.0
        cumulative = 0.0
        for ratio, weight in weighted:
            cumulative += weight
            if cumulative >= threshold:
                return ratio
        return weighted[-1][0]

    def load_ratio(self, environ: Mapping[str, str] | None = None) -> float:
        environment = os.environ if environ is None else environ
        override = environment.get("SOL58_OFFICIAL_PROVISIONAL_RATIO", "").strip()
        if override:
            try:
                return max(0.0, float(override))
            except ValueError:
                pass

        scope = self.scope()
        samples: list[tuple[float, float]] = []
        if self.path.exists():
            with file_lock(self.path):
                lines = self.path.read_text(encoding="utf-8").splitlines()[-200:]
            for line in lines:
                try:
                    row = json.loads(line)
                except (TypeError, ValueError):
                    continue
                if any(
                    str(row.get(key) or "") != value for key, value in scope.items()
                ):
                    continue
                local_score = float(row.get("local_score") or 0.0)
                official_score = float(row.get("official_score") or 0.0)
                if local_score > 0 and official_score > 0:
                    created_at = self.context.parse_timestamp(row.get("created_at"))
                    samples.append(
                        (
                            official_score / local_score,
                            created_at or self.path.stat().st_mtime,
                        )
                    )
        if not samples:
            samples = self.discover_ratios()
        return self.recency_weighted_median(samples) if samples else 1.0

    def record(
        self,
        *,
        local_score: float,
        official_score: float,
        local_latency_ms: float,
        official_latency_ms: float,
        submission_id: Any,
        measurement_profile_id: str | None,
        submission_reason: str,
    ) -> None:
        if local_score <= 0 or official_score <= 0:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        row = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "submission_id": submission_id,
            "local_score": local_score,
            "official_score": official_score,
            "ratio": official_score / local_score,
            "local_latency_ms": local_latency_ms,
            "official_latency_ms": official_latency_ms,
            "eval_stack": self.context.evaluation_stack_version,
            "gpu_type": self.context.gpu_type,
            "measurement_profile_id": (
                measurement_profile_id or self.context.measurement_profile()["id"]
            ),
            "submission_reason": submission_reason,
        }
        with file_lock(self.path):
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                handle.flush()
                os.fsync(handle.fileno())


def project_provisional_score(
    search_score: float, *, target: float, floor: float
) -> float:
    search_score = max(0.0, float(search_score))
    target = max(1e-12, float(target))
    floor = min(float(floor), math.nextafter(target, 0.0))
    if floor <= 0 or floor >= target:
        floor = max(0.0, target - max(1e-6, target * 1e-6))
    if search_score <= floor:
        return search_score
    span = target - floor
    projected = target - span / (1.0 + (search_score - floor) / span)
    return min(math.nextafter(target, 0.0), max(floor, projected))


def official_fitness_anchor(local_best: dict[str, Any] | None) -> dict[str, Any] | None:
    if not local_best:
        return None
    official_score = float(local_best.get("official_score") or 0.0)
    local_latency_ms = float(local_best.get("latency_ms_median") or 0.0)
    official_status = str(local_best.get("official_status") or "").upper()
    if official_status == "COMPLETED" and official_score > 0 and local_latency_ms > 0:
        return {
            "score": official_score,
            "local_latency_ms": local_latency_ms,
            "submission_id": local_best.get("official_submission_id"),
            "source": "completed_local_best",
        }
    anchor_score = float(local_best.get("official_anchor_score") or 0.0)
    anchor_latency_ms = float(local_best.get("official_anchor_latency_ms") or 0.0)
    if anchor_score > 0 and anchor_latency_ms > 0:
        return {
            "score": anchor_score,
            "local_latency_ms": anchor_latency_ms,
            "submission_id": local_best.get("official_anchor_submission_id"),
            "source": "inherited_completed_official",
        }
    return None
