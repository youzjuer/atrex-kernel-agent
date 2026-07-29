"""Reusable local measurement loop with cache, clock, and correctness gates."""

from __future__ import annotations

import contextlib
import subprocess
import time
from contextlib import AbstractContextManager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable


@dataclass(frozen=True)
class LocalEvaluationConfig:
    repeat_count: int
    cache_enabled: bool
    sol_execbench: str
    target_latency_ms: float
    cache_schema_version: int


@dataclass(frozen=True)
class LocalEvaluationRequest:
    workspace: Path
    kernel_source: str
    source_language: str
    source_sha256: str
    measurement_profile: dict[str, Any]
    program_path: str
    started_at: float


@dataclass(frozen=True)
class LocalEvaluationHooks:
    copy_problem_files: Callable[[Path], None]
    write_solution: Callable[[Path, str, str], None]
    load_workload_count: Callable[[Path], int]
    cache_path: Callable[..., tuple[Path, dict[str, Any], str]]
    file_lock: Callable[[Path], AbstractContextManager[Any]]
    read_json: Callable[[Path], dict[str, Any] | None]
    cache_is_valid: Callable[..., bool]
    ensure_clocks: Callable[..., dict[str, Any]]
    run_monitored: Callable[
        [Path, str],
        tuple[subprocess.CompletedProcess[str], list[dict[str, Any]]],
    ]
    parse_traces: Callable[[Path, str], dict[str, Any]]
    set_clocks: Callable[..., None]
    result: Callable[..., dict[str, Any]]
    tail: Callable[..., str]
    atomic_write_json: Callable[[Path, dict[str, Any]], None]


def _error_result(
    hooks: LocalEvaluationHooks,
    status: str,
    summary: str,
    *,
    metrics: dict[str, Any],
    artifacts: dict[str, Any],
) -> dict[str, Any]:
    return {
        "error_result": hooks.result(
            status,
            summary,
            0.0,
            metrics=metrics,
            artifacts=artifacts,
        )
    }


def _restore_cached_processes(
    cached: dict[str, Any],
    config: LocalEvaluationConfig,
    cache_path: Path,
) -> dict[str, Any]:
    processes = [
        subprocess.CompletedProcess(
            args=[config.sol_execbench],
            returncode=int(item.get("returncode") or 0),
            stdout=str(item.get("stdout_tail") or ""),
            stderr=str(item.get("stderr_tail") or ""),
        )
        for item in cached.get("processes", [])
        if isinstance(item, dict)
    ]
    while len(processes) < config.repeat_count:
        processes.append(
            subprocess.CompletedProcess(
                args=[config.sol_execbench], returncode=0, stdout="", stderr=""
            )
        )
    return {
        **cached,
        "processes": processes,
        "cache_hit": True,
        "cache_path": str(cache_path),
    }


def _clock_failure(
    request: LocalEvaluationRequest,
    hooks: LocalEvaluationHooks,
    summary: str,
    *,
    completed: int,
    local_runs: list[dict[str, Any]],
    rejected: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    artifacts: dict[str, Any] = {
        "workspace": str(request.workspace),
        "program_path": request.program_path,
        "local_repeats": local_runs,
    }
    metrics = {
        "eval_time_s": time.time() - request.started_at,
        "measurement_profile": request.measurement_profile,
        "local_repeat_count_completed": completed,
    }
    if rejected is not None:
        artifacts["rejected_clock_attempts"] = rejected
        metrics["clock_rejected_attempt_count"] = len(rejected)
    return _error_result(
        hooks,
        "execution_failed",
        summary,
        metrics=metrics,
        artifacts=artifacts,
    )


def _measurement_failure(
    request: LocalEvaluationRequest,
    config: LocalEvaluationConfig,
    hooks: LocalEvaluationHooks,
    *,
    status: str,
    summary: str,
    completed: int,
    proc: subprocess.CompletedProcess[str],
    parsed: dict[str, Any],
    local_runs: list[dict[str, Any]],
    rejected: list[dict[str, Any]],
) -> dict[str, Any]:
    return _error_result(
        hooks,
        status,
        summary,
        metrics={
            "eval_time_s": time.time() - request.started_at,
            "target_latency_ms": config.target_latency_ms,
            "local_repeat_count_completed": completed,
        },
        artifacts={
            "workspace": str(request.workspace),
            "source_language": request.source_language,
            "measurement_profile": request.measurement_profile,
            "returncode": proc.returncode,
            "stdout_tail": hooks.tail(proc.stdout),
            "stderr_tail": hooks.tail(proc.stderr),
            "per_workload": parsed.get("per_workload", []),
            "local_repeats": local_runs,
            "rejected_clock_attempts": rejected,
        },
    )


def collect_local_evaluation(
    request: LocalEvaluationRequest,
    config: LocalEvaluationConfig,
    hooks: LocalEvaluationHooks,
) -> dict[str, Any]:
    """Collect clean repeats or return one structured evaluator error."""
    if config.repeat_count < 1:
        raise ValueError("repeat_count must be positive")
    hooks.copy_problem_files(request.workspace)
    hooks.write_solution(
        request.workspace, request.kernel_source, request.source_language
    )
    expected = hooks.load_workload_count(request.workspace)
    cache_path, contract, contract_sha256 = hooks.cache_path(
        request.kernel_source,
        request.source_language,
        request.measurement_profile,
        repeat_count=config.repeat_count,
    )
    lock = (
        hooks.file_lock(cache_path)
        if config.cache_enabled
        else contextlib.nullcontext()
    )
    with lock:
        cached = hooks.read_json(cache_path) if config.cache_enabled else None
        if isinstance(cached, dict) and hooks.cache_is_valid(
            cached,
            source_sha256=request.source_sha256,
            contract_sha256=contract_sha256,
            repeat_count=config.repeat_count,
        ):
            return _restore_cached_processes(cached, config, cache_path)

        parsed_runs: list[dict[str, Any]] = []
        processes: list[subprocess.CompletedProcess[str]] = []
        local_runs: list[dict[str, Any]] = []
        latencies_ms: list[float] = []
        rejected: list[dict[str, Any]] = []
        max_attempts = int(contract["max_local_attempts"])
        repeat_index = 0
        attempt_index = 0

        while repeat_index < config.repeat_count:
            attempt_index += 1
            traces_name = (
                "traces.jsonl"
                if repeat_index == 0
                else f"traces_repeat_{repeat_index + 1}.jsonl"
            )
            try:
                clocks_before = hooks.ensure_clocks()
            except Exception as exc:
                return _clock_failure(
                    request,
                    hooks,
                    f"Measurement environment rejected before local repeat "
                    f"{repeat_index + 1}/{config.repeat_count}: {exc}",
                    completed=repeat_index,
                    local_runs=local_runs,
                )

            proc, drift_events = hooks.run_monitored(request.workspace, traces_name)
            parsed = hooks.parse_traces(request.workspace, traces_name)
            try:
                clocks_after = hooks.ensure_clocks(allow_relock=False)
            except Exception as exc:
                clocks_after = {"validation_error": str(exc)}
                drift_events.append(
                    {
                        "detected_at": time.time(),
                        "post_repeat_validation_error": str(exc),
                    }
                )

            if drift_events:
                rejected_path = request.workspace / (
                    f"traces_clock_rejected_attempt_{attempt_index}.jsonl.rejected"
                )
                traces_path = request.workspace / traces_name
                if traces_path.is_file():
                    traces_path.replace(rejected_path)
                rejected.append(
                    {
                        "attempt": attempt_index,
                        "intended_repeat": repeat_index + 1,
                        "rejected_traces_path": str(rejected_path),
                        "clock_drift_events": drift_events,
                        "clocks_before": clocks_before,
                        "clocks_after": clocks_after,
                    }
                )
                if attempt_index >= max_attempts:
                    return _clock_failure(
                        request,
                        hooks,
                        "Official-like local measurement could not collect "
                        f"{config.repeat_count} clean repeats in {max_attempts} attempts "
                        "because GPU clocks changed during evaluation.",
                        completed=repeat_index,
                        local_runs=local_runs,
                        rejected=rejected,
                    )
                try:
                    hooks.set_clocks(request.measurement_profile, stabilize=True)
                except Exception as exc:
                    return _clock_failure(
                        request,
                        hooks,
                        f"Failed to restore clocks after rejected attempt: {exc}",
                        completed=repeat_index,
                        local_runs=local_runs,
                        rejected=rejected,
                    )
                continue

            processes.append(proc)
            parsed_runs.append(parsed)
            total = int(parsed.get("total", 0))
            passed = int(parsed.get("passed", 0))
            failures = parsed.get("failures", [])
            latency_ms = float(parsed.get("latency_ms_geomean", 0.0) or 0.0)
            local_runs.append(
                {
                    "repeat": repeat_index + 1,
                    "attempt": attempt_index,
                    "traces_path": str(request.workspace / traces_name),
                    "returncode": proc.returncode,
                    "passed": passed,
                    "total": total,
                    "latency_ms_geomean": latency_ms,
                    "latency_ms_arith_mean": parsed.get("latency_ms_arith_mean", 0.0),
                    "failures": failures[:6],
                    "clocks_before": clocks_before,
                    "clocks_after": clocks_after,
                }
            )
            if parsed.get("error"):
                return _measurement_failure(
                    request,
                    config,
                    hooks,
                    status="execution_failed",
                    summary=(
                        f"SOL-ExecBench repeat {repeat_index + 1}/{config.repeat_count} "
                        f"failed before producing traces: {parsed['error']}. "
                        f"stderr_tail={hooks.tail(proc.stderr, 1200)}"
                    ),
                    completed=repeat_index,
                    proc=proc,
                    parsed=parsed,
                    local_runs=local_runs,
                    rejected=rejected,
                )
            if (
                proc.returncode != 0
                or total != expected
                or passed != expected
                or failures
            ):
                return _measurement_failure(
                    request,
                    config,
                    hooks,
                    status="validation_failed",
                    summary=(
                        f"Correctness/coverage gate failed on local repeat "
                        f"{repeat_index + 1}/{config.repeat_count}: passed "
                        f"{passed}/{expected}, returncode={proc.returncode}. "
                        f"failures={failures[:6]}"
                    ),
                    completed=repeat_index + 1,
                    proc=proc,
                    parsed=parsed,
                    local_runs=local_runs,
                    rejected=rejected,
                )
            if latency_ms <= 0:
                return _measurement_failure(
                    request,
                    config,
                    hooks,
                    status="execution_failed",
                    summary=(
                        f"Local repeat {repeat_index + 1}/{config.repeat_count} passed "
                        "all workloads but produced no positive latency."
                    ),
                    completed=repeat_index,
                    proc=proc,
                    parsed=parsed,
                    local_runs=local_runs,
                    rejected=rejected,
                )
            latencies_ms.append(latency_ms)
            repeat_index += 1

        payload = {
            "schema_version": config.cache_schema_version,
            "complete": True,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "source_sha256": request.source_sha256,
            "source_language": request.source_language,
            "contract": contract,
            "contract_sha256": contract_sha256,
            "measurement_profile_id": request.measurement_profile["id"],
            "source_workspace": str(request.workspace),
            "expected": expected,
            "parsed_runs": parsed_runs,
            "processes": [
                {
                    "returncode": proc.returncode,
                    "stdout_tail": hooks.tail(proc.stdout),
                    "stderr_tail": hooks.tail(proc.stderr),
                }
                for proc in processes
            ],
            "local_run_records": local_runs,
            "local_latencies_ms": latencies_ms,
            "rejected_clock_attempts": rejected,
            "attempt_count": attempt_index,
        }
        if config.cache_enabled:
            hooks.atomic_write_json(cache_path, payload)
        return {
            **payload,
            "processes": processes,
            "cache_hit": False,
            "cache_path": str(cache_path),
        }
