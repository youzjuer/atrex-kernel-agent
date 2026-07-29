"""Validated task identity and leaderboard snapshots for SOL-ExecBench PES runs."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import urlparse


class TaskSpecError(ValueError):
    """Raised when a SOL-ExecBench task specification is incomplete or invalid."""


def _object(
    value: Any,
    name: str,
    keys: set[str],
) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TaskSpecError(f"{name} must be an object")
    actual = set(value)
    if actual != keys:
        raise TaskSpecError(
            f"{name} schema mismatch; missing={sorted(keys - actual)}, "
            f"unknown={sorted(actual - keys)}"
        )
    return value


def _positive_number(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise TaskSpecError(f"{name} must be a positive number")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise TaskSpecError(f"{name} must be a positive number") from exc
    if result <= 0:
        raise TaskSpecError(f"{name} must be a positive number")
    return result


def _positive_int(value: Any, name: str) -> int:
    result = int(_positive_number(value, name))
    if result != float(value):
        raise TaskSpecError(f"{name} must be a positive integer")
    return result


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise TaskSpecError(f"{name} must be a non-empty string")
    return value.strip()


def _timestamp(value: Any, name: str) -> datetime:
    text = _text(value, name)
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as exc:
        raise TaskSpecError(f"{name} must be an ISO-8601 timestamp") from exc
    if parsed.tzinfo is None:
        raise TaskSpecError(f"{name} must include a timezone")
    return parsed.astimezone(timezone.utc)


def _url(value: Any, name: str) -> str:
    text = _text(value, name).rstrip("/")
    parsed = urlparse(text)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise TaskSpecError(f"{name} must be an absolute HTTP(S) URL")
    return text


@dataclass(frozen=True)
class LeaderboardReference:
    name: str
    rank: int
    score: float
    latency_ms: float


@dataclass(frozen=True)
class SolExecBenchTaskSpec:
    path: Path
    task_name: str
    kernel_id: int
    problem_slug: str
    gpu_type: str
    evaluation_stack_version: str
    official_base_url: str
    leaderboard_url: str
    target_score: float
    local_target_latency_ms: float
    snapshot_at: datetime
    snapshot_max_age_days: int
    reference: LeaderboardReference

    def age_days(self, now: datetime | None = None) -> float:
        current = now or datetime.now(timezone.utc)
        if current.tzinfo is None:
            raise TaskSpecError("current time must include a timezone")
        return max(
            0.0,
            (current.astimezone(timezone.utc) - self.snapshot_at).total_seconds()
            / 86400.0,
        )

    def staleness_warning(self, now: datetime | None = None) -> str | None:
        age = self.age_days(now)
        if age <= self.snapshot_max_age_days:
            return None
        return (
            f"leaderboard snapshot for {self.task_name} is {age:.1f} days old "
            f"(limit {self.snapshot_max_age_days}); refresh {self.path} before "
            "using its rank target"
        )

    def environment(self) -> dict[str, str]:
        return {
            "ATREX_TASK_SPEC_PATH": str(self.path),
            "ATREX_TASK_NAME": self.task_name,
            "SOL58_OFFICIAL_KERNEL_ID": str(self.kernel_id),
            "SOL58_PROBLEM_SLUG": self.problem_slug,
            "SOL58_OFFICIAL_GPU_TYPE": self.gpu_type,
            "SOL58_OFFICIAL_EVAL_STACK_VERSION": self.evaluation_stack_version,
            "SOL58_OFFICIAL_BASE_URL": self.official_base_url,
            "SOL58_LEADERBOARD_URL": self.leaderboard_url,
            "SOL58_TARGET_SCORE": str(self.target_score),
            "SOL58_TARGET_LATENCY_MS": str(self.local_target_latency_ms),
            "SOL58_TARGET_REFERENCE_NAME": self.reference.name,
            "SOL58_TARGET_REFERENCE_SCORE": str(self.reference.score),
            "SOL58_TARGET_REFERENCE_LATENCY_MS": str(self.reference.latency_ms),
            "SOL58_LEADERBOARD_SNAPSHOT_AT": self.snapshot_at.isoformat(),
        }


def load_task_spec(path: str | Path) -> SolExecBenchTaskSpec:
    spec_path = Path(path).resolve()
    try:
        payload = json.loads(spec_path.read_text(encoding="utf-8"))
    except (OSError, TypeError, ValueError) as exc:
        raise TaskSpecError(f"cannot read task spec {spec_path}: {exc}") from exc
    root = _object(
        payload,
        "task spec",
        {
            "schema_version",
            "task_name",
            "kernel",
            "hardware",
            "official",
            "leaderboard_snapshot",
        },
    )
    if root["schema_version"] != 1:
        raise TaskSpecError("task spec must use schema_version 1")
    kernel = _object(root["kernel"], "kernel", {"id", "problem_slug"})
    hardware = _object(root["hardware"], "hardware", {"gpu_type"})
    official = _object(
        root["official"],
        "official",
        {"base_url", "leaderboard_url", "evaluation_stack_version"},
    )
    snapshot = _object(
        root["leaderboard_snapshot"],
        "leaderboard_snapshot",
        {
            "captured_at",
            "max_age_days",
            "target_score",
            "local_target_latency_ms",
            "reference",
        },
    )
    reference = _object(
        snapshot["reference"],
        "leaderboard_snapshot.reference",
        {"name", "rank", "score", "latency_ms"},
    )
    kernel_id = _positive_int(kernel["id"], "kernel.id")
    gpu_type = _text(hardware["gpu_type"], "hardware.gpu_type")
    leaderboard_url = _url(official["leaderboard_url"], "official.leaderboard_url")
    expected_suffix = f"/leaderboard/kernel/{kernel_id}/{gpu_type}"
    if not leaderboard_url.endswith(expected_suffix):
        raise TaskSpecError(
            f"official.leaderboard_url must end with {expected_suffix!r}"
        )
    return SolExecBenchTaskSpec(
        path=spec_path,
        task_name=_text(root["task_name"], "task_name"),
        kernel_id=kernel_id,
        problem_slug=_text(kernel["problem_slug"], "kernel.problem_slug"),
        gpu_type=gpu_type,
        evaluation_stack_version=_text(
            official["evaluation_stack_version"],
            "official.evaluation_stack_version",
        ),
        official_base_url=_url(official["base_url"], "official.base_url"),
        leaderboard_url=leaderboard_url,
        target_score=_positive_number(
            snapshot["target_score"], "leaderboard_snapshot.target_score"
        ),
        local_target_latency_ms=_positive_number(
            snapshot["local_target_latency_ms"],
            "leaderboard_snapshot.local_target_latency_ms",
        ),
        snapshot_at=_timestamp(
            snapshot["captured_at"], "leaderboard_snapshot.captured_at"
        ),
        snapshot_max_age_days=_positive_int(
            snapshot["max_age_days"], "leaderboard_snapshot.max_age_days"
        ),
        reference=LeaderboardReference(
            name=_text(reference["name"], "leaderboard_snapshot.reference.name"),
            rank=_positive_int(
                reference["rank"], "leaderboard_snapshot.reference.rank"
            ),
            score=_positive_number(
                reference["score"], "leaderboard_snapshot.reference.score"
            ),
            latency_ms=_positive_number(
                reference["latency_ms"],
                "leaderboard_snapshot.reference.latency_ms",
            ),
        ),
    )


def reject_task_identity_overrides(
    spec: SolExecBenchTaskSpec,
    environ: Mapping[str, str],
) -> None:
    """Reject conflicting legacy env values instead of silently changing task facts."""
    for name, expected in spec.environment().items():
        actual = environ.get(name)
        if actual is not None and actual != expected:
            raise TaskSpecError(
                f"{name}={actual!r} conflicts with task spec value {expected!r}; "
                f"edit {spec.path} or select another task spec"
            )
