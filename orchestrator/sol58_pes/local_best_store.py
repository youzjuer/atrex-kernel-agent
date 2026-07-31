"""Durable local-incumbent and authoritative-fitness storage for SOL58."""

from __future__ import annotations

import hashlib
import json
import statistics
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from orchestrator.sol58_pes.evaluator_state import (
    atomic_write_json,
    atomic_write_text,
    file_lock,
    read_json,
)


@dataclass(frozen=True)
class LocalBestContext:
    root: Path
    target_latency_ms: float
    repeat_count: int
    cache_schema_version: int
    cuda_language: str
    cute_language: str
    evaluation_stack_version: str
    gpu_type: str
    measurement_profile: Callable[[], dict[str, Any]]
    local_evaluation_contract: Callable[[str, dict[str, Any]], dict[str, Any]]
    parse_traces: Callable[..., dict[str, Any]]
    language_from_source_path: Callable[[str], str | None]
    detect_source_language: Callable[[str], str | None]
    source_dependency_violation: Callable[[str], str | None]
    python_dependency_violation: Callable[[str], str | None]


class LocalBestStore:
    def __init__(self, context: LocalBestContext) -> None:
        self.context = context

    @property
    def path(self) -> Path:
        return self.context.root / "official_cache" / "local_best.json"

    @property
    def recovery_path(self) -> Path:
        return self.context.root / "official_cache" / "local_best_recovery.json"

    @property
    def authoritative_fitness_path(self) -> Path:
        return self.context.root / "official_cache" / "authoritative_fitness.json"

    def kernel_path(self, source_language: str) -> Path:
        suffix = ".py" if source_language == self.context.cute_language else ".cu"
        return self.context.root / "official_cache" / f"local_best_kernel{suffix}"

    @staticmethod
    def source_hash(kernel_source: str) -> str:
        return hashlib.sha256(kernel_source.strip().encode("utf-8")).hexdigest()

    def record_authoritative_fitness(
        self,
        source_hash: str,
        *,
        source_language: str,
        official_score: float,
        official_latency_ms: float,
        local_latency_ms: float,
        submission_id: Any,
        measurement_profile_id: str | None = None,
        official_status: str = "COMPLETED",
        is_correct: bool = True,
    ) -> None:
        path = self.authoritative_fitness_path
        path.parent.mkdir(parents=True, exist_ok=True)
        with file_lock(path):
            registry = read_json(path, {})
            if not isinstance(registry, dict):
                registry = {}
            sources = registry.setdefault("sources", {})
            sources[source_hash] = {
                "source_sha256": source_hash,
                "source_language": source_language,
                "official_score": float(official_score),
                "official_latency_ms": float(official_latency_ms),
                "local_latency_ms": float(local_latency_ms),
                "measurement_profile_id": (
                    measurement_profile_id or self.context.measurement_profile()["id"]
                ),
                "submission_id": submission_id,
                "evaluation_stack_version": self.context.evaluation_stack_version,
                "gpu_type": self.context.gpu_type,
                "status": str(official_status or "COMPLETED").strip().upper(),
                "is_correct": bool(is_correct),
                "updated_at": datetime.now(timezone.utc).isoformat(),
            }
            registry["version"] = 1
            atomic_write_json(path, registry)

    def record_recovery(self, best: dict[str, Any]) -> None:
        profile_id = str(best.get("measurement_profile_id") or "")
        if not profile_id:
            return
        with file_lock(self.recovery_path):
            registry = read_json(self.recovery_path, {})
            if not isinstance(registry, dict):
                registry = {}
            profiles = registry.get("profiles")
            if not isinstance(profiles, dict):
                profiles = {}
            current = profiles.get(profile_id)
            if (
                isinstance(current, dict)
                and current.get("kernel_sha256") == best.get("kernel_sha256")
                and current.get("updated_at") == best.get("updated_at")
            ):
                return
            if isinstance(current, dict):
                current_hash = str(current.get("kernel_sha256") or "")
                candidate_hash = str(best.get("kernel_sha256") or "")
                current_latency = float(current.get("latency_ms_median") or 0.0)
                candidate_latency = float(best.get("latency_ms_median") or 0.0)
                if (
                    current_hash
                    and candidate_hash
                    and current_hash != candidate_hash
                    and current_latency > 0
                    and candidate_latency >= current_latency
                ):
                    return
                if current_hash == candidate_hash and str(
                    current.get("updated_at") or ""
                ) > str(best.get("updated_at") or ""):
                    return
            profiles[profile_id] = dict(best)
            registry.update({"schema_version": 1, "profiles": profiles})
            atomic_write_json(self.recovery_path, registry)

    def save(
        self,
        best: dict[str, Any],
        kernel_source: str | None = None,
        *,
        force: bool = False,
    ) -> bool:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = dict(best)
        profile = self.context.measurement_profile()
        payload.setdefault("measurement_profile_id", profile["id"])
        payload.setdefault("measurement_profile", profile)

        if kernel_source is None:
            workspace = Path(str(payload.get("workspace") or ""))
            workspace_kernel = next(
                (
                    path
                    for path in (workspace / "kernel.cu", workspace / "kernel.py")
                    if path.is_file()
                ),
                None,
            )
            if workspace_kernel is not None:
                try:
                    kernel_source = workspace_kernel.read_text(encoding="utf-8")
                except OSError:
                    kernel_source = None

        source_language = str(payload.get("source_language") or "").strip().lower()
        recorded_kernel_path = Path(str(payload.get("kernel_path") or ""))
        supported = {self.context.cuda_language, self.context.cute_language}
        if source_language not in supported:
            source_language = (
                self.context.language_from_source_path(recorded_kernel_path.name) or ""
            )
        if source_language not in supported and kernel_source:
            source_language = (
                self.context.detect_source_language(kernel_source)
                or self.context.cuda_language
            )
        if source_language not in supported:
            source_language = self.context.cuda_language
        payload["source_language"] = source_language

        expected_hash = str(payload.get("kernel_sha256") or "")
        with file_lock(self.path):
            current = read_json(self.path, {})
            if not force and isinstance(current, dict) and current:
                current_profile = str(current.get("measurement_profile_id") or "")
                candidate_profile = str(payload.get("measurement_profile_id") or "")
                current_hash = str(current.get("kernel_sha256") or "")
                current_latency = float(current.get("latency_ms_median") or 0.0)
                candidate_latency = float(payload.get("latency_ms_median") or 0.0)
                if (
                    current_profile
                    and current_profile == candidate_profile
                    and current_hash
                    and current_hash != expected_hash
                    and current_latency > 0
                    and candidate_latency > 0
                    and candidate_latency >= current_latency
                ):
                    return False

            if kernel_source and (
                not expected_hash or self.source_hash(kernel_source) == expected_hash
            ):
                kernel_path = self.kernel_path(source_language)
                atomic_write_text(kernel_path, kernel_source)
                payload["kernel_path"] = str(kernel_path)
            payload["updated_at"] = datetime.now(timezone.utc).isoformat()
            atomic_write_json(self.path, payload)
        self.record_recovery(payload)
        return True

    def discover_from_cache(self) -> dict[str, Any] | None:
        profile = self.context.measurement_profile()
        candidates: list[dict[str, Any]] = []
        for cache_path in (self.context.root / "local_cache").glob("*.json"):
            cached = read_json(cache_path, {})
            if (
                not isinstance(cached, dict)
                or not cached.get("complete")
                or cached.get("schema_version") != self.context.cache_schema_version
                or str(cached.get("measurement_profile_id") or "") != profile["id"]
            ):
                continue
            latencies = [
                float(value)
                for value in (cached.get("local_latencies_ms") or [])
                if isinstance(value, (int, float)) and float(value) > 0
            ]
            if len(latencies) < self.context.repeat_count:
                continue
            workspace = Path(str(cached.get("source_workspace") or ""))
            source_language = str(
                cached.get("source_language") or self.context.cuda_language
            )
            expected_contract = self.context.local_evaluation_contract(
                source_language, profile
            )
            contract_hash = hashlib.sha256(
                json.dumps(
                    expected_contract, sort_keys=True, separators=(",", ":")
                ).encode("utf-8")
            ).hexdigest()
            if cached.get("contract_sha256") != contract_hash:
                continue
            source_path = workspace / (
                "kernel.py"
                if source_language == self.context.cute_language
                else "kernel.cu"
            )
            if not source_path.is_file():
                continue
            source = source_path.read_text(encoding="utf-8")
            source_hash = str(cached.get("source_sha256") or "")
            if not source_hash or self.source_hash(source) != source_hash:
                continue
            median_latency = float(statistics.median(latencies))
            candidates.append(
                {
                    "kernel_sha256": source_hash,
                    "source_language": source_language,
                    "latency_ms_median": median_latency,
                    "local_score": self.context.target_latency_ms / median_latency,
                    "repeat_count": len(latencies),
                    "repeat_latencies_ms": latencies,
                    "workspace": str(workspace),
                    "source": "local_cache_recovery",
                    "measurement_profile_id": profile["id"],
                    "measurement_profile": profile,
                }
            )
        if not candidates:
            return None
        return min(candidates, key=lambda row: float(row["latency_ms_median"]))

    def discover(self) -> dict[str, Any] | None:
        cached_best = self.discover_from_cache()
        if cached_best is not None:
            return cached_best
        samples: dict[str, dict[str, Any]] = {}
        profile = self.context.measurement_profile()
        for workspace in self.context.root.glob("eval_*"):
            profile_path = workspace / "measurement_profile.json"
            if profile_path.is_file():
                workspace_profile = read_json(profile_path, {})
                if str(workspace_profile.get("id") or "") != profile["id"]:
                    continue
            elif profile["name"] != "native":
                continue

            kernel_path = next(
                (
                    path
                    for path in (workspace / "kernel.cu", workspace / "kernel.py")
                    if path.is_file()
                ),
                None,
            )
            if kernel_path is None:
                continue
            try:
                kernel_source = kernel_path.read_text(encoding="utf-8")
                source_language = (
                    self.context.language_from_source_path(kernel_path.name)
                    or self.context.detect_source_language(kernel_source)
                    or self.context.cuda_language
                )
                dependency_violation = (
                    self.context.source_dependency_violation(kernel_source)
                    if source_language == self.context.cuda_language
                    else self.context.python_dependency_violation(kernel_source)
                )
                if dependency_violation:
                    continue
                kernel_hash = self.source_hash(kernel_source)
            except (OSError, TypeError, ValueError):
                continue

            entry = samples.setdefault(
                kernel_hash,
                {
                    "kernel_sha256": kernel_hash,
                    "source_language": source_language,
                    "latencies_ms": [],
                    "workspaces": [],
                },
            )
            for traces_path in sorted(workspace.glob("traces*.jsonl")):
                try:
                    parsed = self.context.parse_traces(workspace, traces_path.name)
                    total = int(parsed.get("total") or 0)
                    passed = int(parsed.get("passed") or 0)
                    latency_ms = float(parsed.get("latency_ms_geomean") or 0.0)
                    if total <= 0 or passed != total or latency_ms <= 0:
                        continue
                    entry["latencies_ms"].append(latency_ms)
                    entry["workspaces"].append(str(workspace))
                except (OSError, TypeError, ValueError):
                    continue

        candidates: list[dict[str, Any]] = []
        for entry in samples.values():
            latencies = entry["latencies_ms"]
            if len(latencies) < self.context.repeat_count:
                continue
            median_latency = float(statistics.median(latencies))
            candidates.append(
                {
                    "kernel_sha256": entry["kernel_sha256"],
                    "source_language": entry["source_language"],
                    "latency_ms_median": median_latency,
                    "local_score": self.context.target_latency_ms / median_latency,
                    "sample_count": len(latencies),
                    "repeat_count_required": self.context.repeat_count,
                    "workspace": entry["workspaces"][-1],
                    "source": "historical_bootstrap",
                    "measurement_profile_id": profile["id"],
                    "measurement_profile": profile,
                }
            )
        if not candidates:
            return None
        return min(candidates, key=lambda row: float(row["latency_ms_median"]))
