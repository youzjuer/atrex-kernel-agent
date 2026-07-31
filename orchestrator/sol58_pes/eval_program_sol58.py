#!/usr/bin/env python3
"""LoongFlow evaluator for SOL-ExecBench kernel 58.

LoongFlow passes a file containing the generated solution. This evaluator
materializes that content as kernel.cu in a temporary SOL-ExecBench workspace,
runs the official local sol-execbench CLI over all workloads, and returns the
standard LoongFlow {status, summary, score, metrics, artifacts} dictionary.
"""

from __future__ import annotations

import ast
import hashlib
import json
import math
import os
import re
import shlex
import shutil
import statistics
import subprocess
import sys
import threading
import time
import traceback
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from orchestrator.sol58_pes import source_contract
from orchestrator.sol58_pes.evaluator_state import (
    atomic_write_json as _atomic_write_json,
    file_lock as _file_lock,
    geomean as _geomean,
    read_json as _read_json,
    tail as _tail,
)
from orchestrator.sol58_pes.ncu_summary import collect_ncu_analysis, should_profile
from orchestrator.sol58_pes import official_protocol
from orchestrator.sol58_pes import local_gate
from orchestrator.sol58_pes.local_best_store import LocalBestContext, LocalBestStore
from orchestrator.sol58_pes.local_evaluation_pipeline import (
    LocalEvaluationConfig,
    LocalEvaluationHooks,
    LocalEvaluationRequest,
    collect_local_evaluation,
)
from orchestrator.sol58_pes.fitness_calibration import (
    CalibrationContext,
    FitnessCalibration,
    official_fitness_anchor,
    project_provisional_score,
)
from orchestrator.sol58_pes.official_probe import (
    OfficialProbePolicy,
    architecture_probe_evidence,
    claim_official_probe,
)
from orchestrator.sol58_pes.official_fitness_pipeline import (
    OfficialEvaluationState,
    OfficialFitnessConfig,
    OfficialFitnessHooks,
    evaluate_official_fitness,
)


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


TARGET_LATENCY_MS = float(os.environ.get("SOL58_TARGET_LATENCY_MS", "0.006797"))
PROBLEM_DIR = Path(
    os.environ.get(
        "SOL58_PROBLEM_DIR",
        str(
            Path(__file__).resolve().parents[3]
            / "sol-problems"
            / "058_moe_expert_token_radix_sort_with_prefix_sum"
        ),
    )
)
EVAL_ROOT = Path(os.environ.get("SOL58_EVAL_ROOT", "/tmp/sol58_pes_eval"))
SOL_EXECBENCH = os.environ.get("SOL_EXECBENCH", "sol-execbench")
COMPILE_TIMEOUT = int(os.environ.get("SOL58_COMPILE_TIMEOUT", "180"))
RUN_TIMEOUT = int(os.environ.get("SOL58_SOL_TIMEOUT", "120"))
LOCAL_REPEAT_COUNT = max(1, int(os.environ.get("SOL58_LOCAL_REPEAT_COUNT", "3")))
LOCAL_BEST_GATE = _env_bool("SOL58_LOCAL_BEST_GATE", True)
LOCAL_GATE_SIGMA_MULTIPLIER = max(
    0.0, float(os.environ.get("SOL58_LOCAL_GATE_SIGMA_MULTIPLIER", "2.0"))
)
LOCAL_GATE_RELATIVE_NOISE_FLOOR = max(
    0.0, float(os.environ.get("SOL58_LOCAL_GATE_RELATIVE_NOISE_FLOOR", "0.01"))
)
LOCAL_GATE_RECHECK_PAIRS = max(
    0, int(os.environ.get("SOL58_LOCAL_GATE_RECHECK_PAIRS", "2"))
)
LOCAL_GATE_UNCERTAIN_RELATIVE_TOLERANCE = max(
    0.0,
    float(os.environ.get("SOL58_LOCAL_GATE_UNCERTAIN_RELATIVE_TOLERANCE", "0.005")),
)
LOCAL_GATE_ALLOW_UNCERTAIN_OFFICIAL = _env_bool(
    "SOL58_LOCAL_GATE_ALLOW_UNCERTAIN_OFFICIAL", True
)
LOCAL_GATE_CHALLENGER_COOLDOWN_S = max(
    0.0, float(os.environ.get("SOL58_LOCAL_GATE_CHALLENGER_COOLDOWN_S", "900"))
)
MEASUREMENT_PROFILE_NAME = (
    os.environ.get("SOL58_MEASUREMENT_PROFILE", "native").strip().lower()
)
CODE_LANGUAGE = os.environ.get("SOL58_CODE_LANGUAGE", "cuda_cpp").strip().lower()
CUTEDSL_GENERATION_RATE = float(os.environ.get("SOL58_CUTEDSL_GENERATION_RATE", "0.5"))
CUTEDSL_SCHEDULE_PERIOD = max(
    1,
    int(os.environ.get("SOL58_CUTEDSL_SCHEDULE_PERIOD", "10")),
)
NCU_SUMMARY_ENABLED = _env_bool("SOL58_NCU_SUMMARY", False)
NCU_PROFILE_POLICY = (
    os.environ.get("SOL58_NCU_PROFILE_POLICY", "all_correct").strip().lower()
)
OFFICIAL_FITNESS = _env_bool("SOL58_OFFICIAL_FITNESS", False)
OFFICIAL_BASE_URL = os.environ.get(
    "SOL58_OFFICIAL_BASE_URL",
    "https://research.nvidia.com/benchmarks/sol-execbench",
).rstrip("/")
OFFICIAL_KERNEL_ID = int(os.environ.get("SOL58_OFFICIAL_KERNEL_ID", "58"))
OFFICIAL_GPU_TYPE = os.environ.get("SOL58_OFFICIAL_GPU_TYPE", "B200")
OFFICIAL_EVAL_STACK_VERSION = os.environ.get(
    "SOL58_OFFICIAL_EVAL_STACK_VERSION", "v1.1"
)
OFFICIAL_SUBMISSION_MODE = os.environ.get("SOL58_OFFICIAL_SUBMISSION_MODE", "private")
OFFICIAL_POLL_INTERVAL = float(os.environ.get("SOL58_OFFICIAL_POLL_INTERVAL", "10"))
OFFICIAL_POLL_TIMEOUT = float(os.environ.get("SOL58_OFFICIAL_POLL_TIMEOUT", "180"))
OFFICIAL_REQUEST_TIMEOUT = float(os.environ.get("SOL58_OFFICIAL_REQUEST_TIMEOUT", "10"))
OFFICIAL_CACHE = _env_bool("SOL58_OFFICIAL_CACHE", True)
LOCAL_EVAL_CACHE = _env_bool("SOL58_LOCAL_EVAL_CACHE", True)
OFFICIAL_ASYNC_SUBMIT = _env_bool("SOL58_OFFICIAL_ASYNC_SUBMIT", True)
OFFICIAL_ASYNC_REFRESH_DELAY = float(
    os.environ.get("SOL58_OFFICIAL_ASYNC_REFRESH_DELAY", "60")
)
OFFICIAL_MIN_LOCAL_SCORE = float(os.environ.get("SOL58_OFFICIAL_MIN_LOCAL_SCORE", "0"))
OFFICIAL_MAX_LOCAL_LATENCY_MS = max(
    0.0,
    float(os.environ.get("SOL58_OFFICIAL_MAX_LOCAL_LATENCY_MS", "0")),
)
OFFICIAL_PENDING_RESULT_GRACE = float(
    os.environ.get("SOL58_OFFICIAL_PENDING_RESULT_GRACE", "60")
)
OFFICIAL_CACHE_REFRESH_TIMEOUT = float(
    os.environ.get("SOL58_OFFICIAL_CACHE_REFRESH_TIMEOUT", "0")
)
OFFICIAL_REFRESH_BATCH_SIZE = max(
    1, int(os.environ.get("SOL58_OFFICIAL_REFRESH_BATCH_SIZE", "4"))
)
OFFICIAL_REFRESH_TIME_BUDGET = max(
    0.0,
    float(
        os.environ.get(
            "SOL58_OFFICIAL_REFRESH_TIME_BUDGET",
            str(OFFICIAL_REQUEST_TIMEOUT),
        )
    ),
)
OFFICIAL_PENDING_SCORE_POLICY = (
    os.environ.get(
        "SOL58_OFFICIAL_PENDING_SCORE_POLICY",
        "local_proxy",
    )
    .strip()
    .lower()
)
OFFICIAL_PROBE_POLICY = OfficialProbePolicy(
    enabled=_env_bool("SOL58_OFFICIAL_PROBE_ENABLED", True),
    interval=max(1, int(os.environ.get("SOL58_OFFICIAL_PROBE_INTERVAL", "20"))),
    cooldown_seconds=max(
        0.0, float(os.environ.get("SOL58_OFFICIAL_PROBE_COOLDOWN_S", "3600"))
    ),
    max_per_day=max(0, int(os.environ.get("SOL58_OFFICIAL_PROBE_MAX_PER_DAY", "6"))),
    max_relative_regression=max(
        0.0,
        float(os.environ.get("SOL58_OFFICIAL_PROBE_MAX_RELATIVE_REGRESSION", "0.10")),
    ),
    architecture_only=_env_bool("SOL58_OFFICIAL_PROBE_ARCHITECTURE_ONLY", True),
)
OFFICIAL_TARGET_SCORE = float(os.environ.get("SOL58_TARGET_SCORE", "0.904135"))
OFFICIAL_PROVISIONAL_SCORE_CAP = float(
    os.environ.get(
        "SOL58_OFFICIAL_PROVISIONAL_SCORE_CAP",
        f"{max(0.0, OFFICIAL_TARGET_SCORE - 0.005):.6f}",
    )
)
OFFICIAL_TERMINAL_STATUSES = {
    "COMPLETED",
    "FAILED",
    "ERROR",
    "CANCELLED",
    "TIMEOUT",
    "STALE_PENDING_RESULT",
    "DEFERRED_RESULT",
}
OFFICIAL_REFRESHABLE_STATUSES = {
    "TIMEOUT",
    "STALE_PENDING_RESULT",
    "DEFERRED_RESULT",
    "PENDING_RESULT",
    "PENDING",
    "RUNNING",
    "EVALUATING",
}

SOURCE_LANGUAGE_CUDA = source_contract.SOURCE_LANGUAGE_CUDA
SOURCE_LANGUAGE_CUTE = source_contract.SOURCE_LANGUAGE_CUTE
SOURCE_LANGUAGE_AUTO = source_contract.SOURCE_LANGUAGE_AUTO
SUPPORTED_CODE_LANGUAGES = source_contract.SUPPORTED_CODE_LANGUAGES

LOCAL_EVAL_CACHE_SCHEMA_VERSION = 3
LOCAL_EVAL_MEASUREMENT_CONTRACT_VERSION = os.environ.get(
    "SOL58_LOCAL_EVAL_CONTRACT_VERSION", "sol58-v3"
)


def _measurement_profile() -> dict[str, Any]:
    """Return the configured local timing contract and a stable identity for it."""
    profile = {
        "schema_version": 1,
        "name": MEASUREMENT_PROFILE_NAME or "native",
        "measurement_device": os.environ.get(
            "SOL58_MEASUREMENT_DEVICE_ID", "unspecified"
        ).strip(),
        "warmup_runs": int(os.environ.get("SOL58_WARMUP_RUNS", "10")),
        "iterations": int(os.environ.get("SOL58_ITERATIONS", "50")),
        "seed": int(os.environ.get("SOL58_SEED", "200")),
        "benchmark_reference": _env_bool("SOL58_BENCHMARK_REFERENCE", False),
        "lock_clocks": _env_bool("SOL58_LOCK_CLOCKS", False),
        "gpu_clock_mhz": int(os.environ.get("SOL_EXECBENCH_GPU_CLK_MHZ", "0") or 0),
        "dram_clock_mhz": int(os.environ.get("SOL_EXECBENCH_DRAM_CLK_MHZ", "0") or 0),
        "clock_tolerance_mhz": int(os.environ.get("SOL58_CLOCK_TOLERANCE_MHZ", "10")),
        "clock_monitor_interval_seconds": float(
            os.environ.get("SOL58_CLOCK_MONITOR_INTERVAL_SECONDS", "0.1")
        ),
        "cuda_gencode": os.environ.get("SOL58_CUDA_GENCODE", "runtime-native").strip(),
        "local_eval_stack": os.environ.get(
            "SOL58_LOCAL_EVAL_STACK_ID", Path(SOL_EXECBENCH).name
        ).strip(),
    }
    canonical = json.dumps(profile, sort_keys=True, separators=(",", ":"))
    profile["id"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]
    return profile


def _local_best_profile_matches(
    best: dict[str, Any] | None,
    profile: dict[str, Any] | None = None,
) -> bool:
    if not best:
        return False
    current = profile or _measurement_profile()
    recorded_id = str(best.get("measurement_profile_id") or "")
    if recorded_id:
        return recorded_id == current["id"]

    # Historical records predate profile identities. They are only comparable
    # with the legacy native evaluator, never with an official-like profile.
    return current["name"] == "native"


def _clock_gpu_index() -> str:
    explicit = os.environ.get("SOL58_CLOCK_GPU_INDEX", "").strip()
    if explicit:
        return explicit
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",", 1)[0].strip()
    return visible if visible.isdigit() else "0"


def _query_gpu_clocks() -> dict[str, Any]:
    index = _clock_gpu_index()
    proc = subprocess.run(
        [
            "nvidia-smi",
            "-i",
            index,
            "--query-gpu=uuid,name,clocks.current.sm,clocks.current.memory",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        timeout=10,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"nvidia-smi clock query failed: {_tail(proc.stderr, 500)}")
    parts = [part.strip() for part in proc.stdout.strip().split(",")]
    if len(parts) != 4:
        raise RuntimeError(f"unexpected nvidia-smi clock output: {proc.stdout!r}")
    return {
        "gpu_index": index,
        "gpu_uuid": parts[0],
        "gpu_name": parts[1],
        "sm_clock_mhz": int(parts[2]),
        "dram_clock_mhz": int(parts[3]),
    }


def _clock_state_matches(state: dict[str, Any], profile: dict[str, Any]) -> bool:
    tolerance = int(os.environ.get("SOL58_CLOCK_TOLERANCE_MHZ", "10"))
    return (
        abs(int(state["sm_clock_mhz"]) - int(profile["gpu_clock_mhz"])) <= tolerance
        and abs(int(state["dram_clock_mhz"]) - int(profile["dram_clock_mhz"]))
        <= tolerance
    )


def _set_measurement_clocks(
    profile: dict[str, Any],
    *,
    stabilize: bool,
) -> None:
    index = _clock_gpu_index()
    commands = (
        [
            "sudo",
            "-n",
            "nvidia-smi",
            "-i",
            index,
            "-lgc",
            str(profile["gpu_clock_mhz"]),
        ],
        [
            "sudo",
            "-n",
            "nvidia-smi",
            "-i",
            index,
            "-lmc",
            str(profile["dram_clock_mhz"]),
        ],
    )
    for command in commands:
        proc = subprocess.run(command, capture_output=True, text=True, timeout=15)
        if proc.returncode != 0:
            raise RuntimeError(
                f"failed to restore official-like clocks with {shlex.join(command)}: "
                f"{_tail(proc.stderr, 500)}"
            )
    if stabilize:
        time.sleep(float(os.environ.get("SOL58_CLOCK_STABILIZE_SECONDS", "2")))


def _ensure_measurement_clocks(*, allow_relock: bool = True) -> dict[str, Any]:
    profile = _measurement_profile()
    if not profile["lock_clocks"]:
        return {"required": False}
    if profile["gpu_clock_mhz"] <= 0 or profile["dram_clock_mhz"] <= 0:
        raise RuntimeError(
            "locked measurement profile requires SOL_EXECBENCH_GPU_CLK_MHZ and "
            "SOL_EXECBENCH_DRAM_CLK_MHZ"
        )

    state = _query_gpu_clocks()
    if _clock_state_matches(state, profile):
        return {**state, "required": True, "relocked": False}

    if allow_relock and _env_bool("SOL58_AUTO_RELOCK_CLOCKS", False):
        _set_measurement_clocks(profile, stabilize=True)
        state = _query_gpu_clocks()
        if _clock_state_matches(state, profile):
            return {**state, "required": True, "relocked": True}

    raise RuntimeError(
        "measurement clock mismatch on GPU "
        f"{state.get('gpu_index')}: observed {state.get('sm_clock_mhz')}/"
        f"{state.get('dram_clock_mhz')} MHz, expected {profile['gpu_clock_mhz']}/"
        f"{profile['dram_clock_mhz']} MHz"
    )


def _monitor_measurement_clocks(
    stop: threading.Event,
    drift_events: list[dict[str, Any]],
) -> None:
    profile = _measurement_profile()
    interval = max(
        0.05,
        float(os.environ.get("SOL58_CLOCK_MONITOR_INTERVAL_SECONDS", "0.1")),
    )
    while not stop.is_set():
        try:
            state = _query_gpu_clocks()
            if not _clock_state_matches(state, profile):
                event = {**state, "detected_at": time.time()}
                drift_events.append(event)
                if _env_bool("SOL58_AUTO_RELOCK_CLOCKS", False):
                    _set_measurement_clocks(profile, stabilize=False)
        except Exception as exc:
            drift_events.append({"detected_at": time.time(), "monitor_error": str(exc)})
        stop.wait(interval)


def _run_sol_execbench_monitored(
    workspace: Path,
    traces_filename: str,
) -> tuple[subprocess.CompletedProcess[str], list[dict[str, Any]]]:
    profile = _measurement_profile()
    if not profile["lock_clocks"]:
        return _run_sol_execbench(workspace, traces_filename), []

    stop = threading.Event()
    drift_events: list[dict[str, Any]] = []
    monitor = threading.Thread(
        target=_monitor_measurement_clocks,
        args=(stop, drift_events),
        name="sol58-clock-monitor",
        daemon=True,
    )
    monitor.start()
    try:
        proc = _run_sol_execbench(workspace, traces_filename)
    finally:
        stop.set()
        monitor.join(timeout=5)
    return proc, drift_events


def _normalize_code_language(value: str | None = None) -> str:
    return source_contract.normalize_code_language(value, default=CODE_LANGUAGE)


def _program_iteration(program_path: str) -> int | None:
    return source_contract.program_iteration(program_path)


def _required_source_language(
    program_path: str,
    code_language: str | None = None,
    cutedsl_rate: float | None = None,
    schedule_period: int | None = None,
) -> str | None:
    return source_contract.required_source_language(
        program_path,
        code_language,
        cutedsl_rate,
        schedule_period,
        default_code_language=CODE_LANGUAGE,
        default_cutedsl_rate=CUTEDSL_GENERATION_RATE,
        default_schedule_period=CUTEDSL_SCHEDULE_PERIOD,
    )


def _language_from_source_path(path: str) -> str | None:
    return source_contract.language_from_source_path(path)


def _detect_source_language(kernel_source: str) -> str | None:
    return source_contract.detect_source_language(kernel_source)


def _extract_kernel_source(raw: str, code_language: str | None = None) -> str:
    return source_contract.extract_kernel_source(
        raw,
        code_language,
        default_code_language=CODE_LANGUAGE,
    )


def _source_dependency_violation(kernel_source: str) -> str | None:
    return source_contract.source_dependency_violation(kernel_source)


def _python_dependency_violation(kernel_source: str) -> str | None:
    return source_contract.python_dependency_violation(kernel_source)


def _cute_dsl_validation_error(kernel_source: str) -> str | None:
    return source_contract.cute_dsl_validation_error(kernel_source)


def _candidate_validation_error(
    kernel_source: str,
    code_language: str | None = None,
) -> tuple[str | None, str | None]:
    return source_contract.candidate_validation_error(
        kernel_source,
        code_language,
        default_code_language=CODE_LANGUAGE,
    )


def _copy_problem_files(workspace: Path) -> None:
    if not PROBLEM_DIR.exists():
        raise FileNotFoundError(f"SOL58_PROBLEM_DIR not found: {PROBLEM_DIR}")

    for name in ("definition.json", "workload.jsonl", "reference.py"):
        src = PROBLEM_DIR / name
        if src.exists():
            shutil.copy2(src, workspace / name)

    # Keep evaluator cost close to leaderboard defaults but avoid benchmarking the
    # Python reference on every candidate.
    config = {
        "warmup_runs": int(os.environ.get("SOL58_WARMUP_RUNS", "10")),
        "iterations": int(os.environ.get("SOL58_ITERATIONS", "50")),
        "lock_clocks": os.environ.get("SOL58_LOCK_CLOCKS", "0") == "1",
        "benchmark_reference": os.environ.get("SOL58_BENCHMARK_REFERENCE", "0") == "1",
        "seed": int(os.environ.get("SOL58_SEED", "200")),
    }
    (workspace / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    (workspace / "measurement_profile.json").write_text(
        json.dumps(_measurement_profile(), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _detect_cuda_gencode_flags() -> list[str]:
    override = os.environ.get("SOL58_CUDA_GENCODE", "").strip()
    if override:
        return shlex.split(override)

    if os.environ.get("SOL58_AUTO_GENCODE", "0") == "1":
        return []

    try:
        import torch

        if torch.cuda.is_available():
            major, minor = torch.cuda.get_device_capability()
            arch = f"{major}{minor}"
            return [f"-gencode=arch=compute_{arch},code=sm_{arch}"]
    except Exception:
        pass

    try:
        out = (
            subprocess.check_output(
                ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"],
                text=True,
                stderr=subprocess.DEVNULL,
            )
            .strip()
            .splitlines()[0]
            .strip()
        )
        arch = out.replace(".", "")
        return [f"-gencode=arch=compute_{arch},code=sm_{arch}"]
    except Exception:
        return []


def _write_solution(workspace: Path, kernel_source: str, source_language: str) -> None:
    source_language = _normalize_code_language(source_language)
    if source_language == SOURCE_LANGUAGE_AUTO:
        raise ValueError("auto is a selection mode, not a concrete solution language")

    if source_language == SOURCE_LANGUAGE_CUDA:
        source_path = "kernel.cu"
        description = "LoongFlow PES-generated CUDA C++ candidate for SOL kernel 58."
        cuda_cflags = [
            "-O3",
            "--use_fast_math",
            "-std=c++17",
        ] + _detect_cuda_gencode_flags()
        spec = {
            "languages": [SOURCE_LANGUAGE_CUDA],
            "target_hardware": ["B200", "LOCAL"],
            "entry_point": f"{source_path}::run",
            "dependencies": [],
            "compile_options": {
                "cuda_cflags": cuda_cflags,
                "ld_flags": ["-lcuda"],
            },
            "destination_passing_style": True,
            "binding": "torch",
        }
    else:
        source_path = "kernel.py"
        description = "LoongFlow PES-generated CuTe DSL candidate for SOL kernel 58."
        spec = {
            "languages": [SOURCE_LANGUAGE_CUTE],
            "target_hardware": ["B200", "LOCAL"],
            "entry_point": f"{source_path}::run",
            "dependencies": ["torch", "cutlass"],
            "destination_passing_style": True,
        }

    (workspace / source_path).write_text(kernel_source, encoding="utf-8")
    solution = {
        "name": f"sol58_loongflow_candidate_{uuid.uuid4().hex[:8]}",
        "definition": "058_moe_expert_token_radix_sort_with_prefix_sum",
        "author": "atrex-loongflow-pes",
        "description": description,
        "spec": spec,
        "sources": [{"path": source_path}],
    }
    (workspace / "solution.json").write_text(
        json.dumps(solution, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _run_sol_execbench(
    workspace: Path,
    traces_filename: str = "traces.jsonl",
) -> subprocess.CompletedProcess[str]:
    traces = workspace / traces_filename
    cmd = [
        SOL_EXECBENCH,
        ".",
        "--solution",
        "solution.json",
        "--config",
        "config.json",
        "--compile-timeout",
        str(COMPILE_TIMEOUT),
        "--timeout",
        str(RUN_TIMEOUT),
        "-o",
        str(traces),
    ]
    if os.environ.get("SOL58_VERBOSE_EVAL") == "1":
        cmd.append("-v")
    child_env = {**os.environ, "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}
    sol_execbench_path = Path(SOL_EXECBENCH)
    if sol_execbench_path.parent != Path(".") and sol_execbench_path.is_file():
        child_env["PATH"] = (
            f"{sol_execbench_path.resolve().parent}:{child_env.get('PATH', '')}"
        )
    return subprocess.run(
        cmd,
        cwd=str(workspace),
        capture_output=True,
        text=True,
        timeout=COMPILE_TIMEOUT + RUN_TIMEOUT + 60,
        env=child_env,
    )


def _ncu_summary_evidence(
    *,
    workspace: Path,
    kernel_source: str,
    source_language: str,
    source_sha256: str,
    measurement_profile: dict[str, Any],
    per_workload: list[dict[str, Any]],
    improved: bool,
    program_path: str,
) -> dict[str, Any]:
    """Collect optional profiler evidence without changing evaluator fitness."""
    iteration = _program_iteration(program_path)
    selected, reason = should_profile(
        enabled=NCU_SUMMARY_ENABLED,
        policy=NCU_PROFILE_POLICY,
        improved=improved,
        iteration=iteration,
    )
    if not selected:
        return {
            "enabled": NCU_SUMMARY_ENABLED,
            "status": "disabled" if not NCU_SUMMARY_ENABLED else "skipped",
            "reason": reason,
            "policy": NCU_PROFILE_POLICY,
        }

    try:
        if not (workspace / "solution.json").is_file():
            _copy_problem_files(workspace)
            _write_solution(workspace, kernel_source, source_language)
        evidence = collect_ncu_analysis(
            workspace=workspace,
            kernel_source=kernel_source,
            source_language=source_language,
            source_sha256=source_sha256,
            measurement_profile=measurement_profile,
            per_workload=per_workload,
            cache_root=Path(
                os.environ.get("SOL58_NCU_CACHE_DIR", str(EVAL_ROOT / "ncu_cache"))
            ),
            sol_execbench=SOL_EXECBENCH,
            compile_timeout=COMPILE_TIMEOUT,
            run_timeout=RUN_TIMEOUT,
        )
        evidence["policy"] = NCU_PROFILE_POLICY
        return evidence
    except Exception as exc:
        return {
            "enabled": True,
            "status": "failed",
            "policy": NCU_PROFILE_POLICY,
            "error": f"NCU summary integration failed: {exc}"[-1200:],
        }


def _load_workload_count(workspace: Path) -> int:
    workload_path = workspace / "workload.jsonl"
    return sum(1 for line in workload_path.read_text().splitlines() if line.strip())


def _parse_traces(
    workspace: Path,
    traces_filename: str = "traces.jsonl",
) -> dict[str, Any]:
    traces_path = workspace / traces_filename
    if not traces_path.exists():
        return {"traces": [], "error": f"no traces produced at {traces_path}"}

    traces = [
        json.loads(line)
        for line in traces_path.read_text().splitlines()
        if line.strip()
    ]
    latencies_ms: list[float] = []
    per_workload: list[dict[str, Any]] = []
    failures: list[str] = []
    max_abs = 0.0
    max_rel = 0.0

    for idx, trace in enumerate(traces):
        workload = trace.get("workload") or {}
        axes = workload.get("axes") or {}
        ev = trace.get("evaluation") or {}
        status = ev.get("status") or "NO_EVAL"
        corr = ev.get("correctness") or {}
        perf = ev.get("performance") or {}
        latency_ms = perf.get("latency_ms")

        if isinstance(corr.get("max_absolute_error"), (int, float)):
            max_abs = max(max_abs, float(corr["max_absolute_error"]))
        if isinstance(corr.get("max_relative_error"), (int, float)):
            max_rel = max(max_rel, float(corr["max_relative_error"]))

        entry = {
            "index": idx,
            "uuid": workload.get("uuid"),
            "axes": axes,
            "status": status,
            "latency_ms": latency_ms,
            "reference_latency_ms": perf.get("reference_latency_ms"),
            "speedup_factor": perf.get("speedup_factor"),
            "max_abs_err": corr.get("max_absolute_error"),
            "max_rel_err": corr.get("max_relative_error"),
        }
        per_workload.append(entry)

        if (
            status == "PASSED"
            and isinstance(latency_ms, (int, float))
            and latency_ms > 0
        ):
            latencies_ms.append(float(latency_ms))
        else:
            failures.append(f"{idx}:{status}:{axes}")

    return {
        "traces": traces,
        "total": len(traces),
        "passed": len(traces) - len(failures),
        "failures": failures,
        "per_workload": per_workload,
        "latency_ms_geomean": _geomean(latencies_ms),
        "latency_ms_arith_mean": (
            (sum(latencies_ms) / len(latencies_ms)) if latencies_ms else 0.0
        ),
        "max_abs_err": max_abs,
        "max_rel_err": max_rel,
    }


def _local_best_store() -> LocalBestStore:
    return LocalBestStore(
        LocalBestContext(
            root=EVAL_ROOT,
            target_latency_ms=TARGET_LATENCY_MS,
            repeat_count=LOCAL_REPEAT_COUNT,
            cache_schema_version=LOCAL_EVAL_CACHE_SCHEMA_VERSION,
            cuda_language=SOURCE_LANGUAGE_CUDA,
            cute_language=SOURCE_LANGUAGE_CUTE,
            evaluation_stack_version=OFFICIAL_EVAL_STACK_VERSION,
            gpu_type=OFFICIAL_GPU_TYPE,
            measurement_profile=_measurement_profile,
            local_evaluation_contract=_local_evaluation_contract,
            parse_traces=_parse_traces,
            language_from_source_path=_language_from_source_path,
            detect_source_language=_detect_source_language,
            source_dependency_violation=_source_dependency_violation,
            python_dependency_violation=_python_dependency_violation,
        )
    )


def _local_best_path() -> Path:
    return _local_best_store().path


def _local_best_recovery_path() -> Path:
    return _local_best_store().recovery_path


def _local_best_kernel_path(source_language: str = SOURCE_LANGUAGE_CUDA) -> Path:
    return _local_best_store().kernel_path(source_language)


def _kernel_source_hash(kernel_source: str) -> str:
    return LocalBestStore.source_hash(kernel_source)


def _authoritative_fitness_registry_path() -> Path:
    return _local_best_store().authoritative_fitness_path


def _record_authoritative_fitness_by_hash(
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
    _local_best_store().record_authoritative_fitness(
        source_hash,
        source_language=source_language,
        official_score=official_score,
        official_latency_ms=official_latency_ms,
        local_latency_ms=local_latency_ms,
        submission_id=submission_id,
        measurement_profile_id=measurement_profile_id,
        official_status=official_status,
        is_correct=is_correct,
    )


def _record_authoritative_source_fitness(
    kernel_source: str,
    *,
    source_language: str,
    official_score: float,
    official_latency_ms: float,
    local_latency_ms: float,
    submission_id: Any,
) -> None:
    source_hash = _kernel_source_hash(kernel_source)
    _record_authoritative_fitness_by_hash(
        source_hash,
        source_language=source_language,
        official_score=official_score,
        official_latency_ms=official_latency_ms,
        local_latency_ms=local_latency_ms,
        submission_id=submission_id,
    )


def _record_local_best_recovery(best: dict[str, Any]) -> None:
    _local_best_store().record_recovery(best)


def _save_local_best(
    best: dict[str, Any],
    kernel_source: str | None = None,
    *,
    force: bool = False,
) -> bool:
    return _local_best_store().save(best, kernel_source, force=force)


def _discover_local_best_from_cache() -> dict[str, Any] | None:
    return _local_best_store().discover_from_cache()


def _discover_local_best() -> dict[str, Any] | None:
    return _local_best_store().discover()


def _validated_local_best(
    best: Any,
) -> tuple[dict[str, Any], str] | None:
    if not isinstance(best, dict) or float(best.get("latency_ms_median") or 0.0) <= 0:
        return None
    source_data = _read_local_best_source(best)
    if source_data is None:
        return None
    kernel_source, source_language = source_data
    dependency_violation = (
        _source_dependency_violation(kernel_source)
        if source_language == SOURCE_LANGUAGE_CUDA
        else _python_dependency_violation(kernel_source)
    )
    if dependency_violation:
        return None
    validated = dict(best)
    validated["source_language"] = source_language
    return validated, kernel_source


def _load_local_best() -> dict[str, Any] | None:
    path = _local_best_path()
    if path.exists():
        with _file_lock(path):
            validated = _validated_local_best(_read_json(path, {}))
        if validated is not None:
            best, _ = validated
            _record_local_best_recovery(best)
            return best

    recovery_path = _local_best_recovery_path()
    with _file_lock(recovery_path):
        recovery = _read_json(recovery_path, {})
    profiles = recovery.get("profiles") if isinstance(recovery, dict) else None
    if isinstance(profiles, dict):
        recovered = _validated_local_best(profiles.get(_measurement_profile()["id"]))
        if recovered is not None:
            best, kernel_source = recovered
            _save_local_best(best, kernel_source, force=True)
            return _read_json(path, best)

    best = _discover_local_best()
    if best is not None:
        _save_local_best(best)
    return best


def _result(
    status: str,
    summary: str,
    score: float,
    metrics: dict[str, Any] | None = None,
    artifacts: dict[str, Any] | None = None,
) -> dict[str, Any]:
    metrics = dict(metrics or {})
    official = metrics.get("official")
    official = official if isinstance(official, dict) else {}
    provisional_details = official.get("provisional_calibration")
    provisional_details = (
        provisional_details if isinstance(provisional_details, dict) else {}
    )
    search_score = float(
        metrics.get(
            "search_score",
            official.get(
                "search_score",
                provisional_details.get("search_score", score),
            ),
        )
        or 0.0
    )
    authoritative = bool(official.get("authoritative"))
    if OFFICIAL_FITNESS:
        certified_score: float | None = float(score) if authoritative else None
        fitness_source = str(official.get("fitness_source") or "uncertified")
    else:
        certified_score = float(score) if status == "success" else None
        fitness_source = "local" if status == "success" else "none"
    metrics.update(
        {
            "search_score": search_score,
            "selection_score": float(score),
            "certified_score": certified_score,
            "fitness_source": fitness_source,
            "target_certified": bool(
                certified_score is not None and certified_score >= OFFICIAL_TARGET_SCORE
            ),
        }
    )
    return {
        "status": status,
        "summary": summary,
        "score": float(score),
        "metrics": metrics,
        "artifacts": artifacts or {},
    }


def _problem_contract_hash() -> str:
    digest = hashlib.sha256()
    for name in ("definition.json", "workload.jsonl", "reference.py"):
        path = PROBLEM_DIR / name
        digest.update(name.encode("utf-8"))
        try:
            digest.update(path.read_bytes())
        except OSError:
            digest.update(b"<missing>")
    return digest.hexdigest()


def _tool_path_identity(command: str) -> dict[str, Any]:
    resolved = shutil.which(command) or command
    path = Path(resolved).expanduser()
    try:
        stat = path.resolve().stat()
        return {
            "path": str(path.resolve()),
            "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
        }
    except OSError:
        return {"path": str(path)}


_MEASUREMENT_IMPLEMENTATION_SHA256: str | None = None
_MEASUREMENT_FUNCTIONS = {
    "_measurement_profile",
    "_clock_state_matches",
    "_ensure_measurement_clocks",
    "_run_sol_execbench_monitored",
    "_copy_problem_files",
    "_write_solution",
    "_run_sol_execbench",
    "_parse_traces",
    "_collect_local_evaluation",
}


def _measurement_implementation_hash() -> str:
    """Hash measurement semantics while ignoring comments and formatting."""
    global _MEASUREMENT_IMPLEMENTATION_SHA256
    if _MEASUREMENT_IMPLEMENTATION_SHA256 is not None:
        return _MEASUREMENT_IMPLEMENTATION_SHA256

    source_tree = ast.parse(Path(__file__).read_text(encoding="utf-8"))
    selected_functions = [
        node
        for node in source_tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name in _MEASUREMENT_FUNCTIONS
    ]
    found = {node.name for node in selected_functions}
    missing = sorted(_MEASUREMENT_FUNCTIONS - found)
    if missing:
        raise RuntimeError(
            "local evaluation contract is missing measurement functions: "
            + ", ".join(missing)
        )
    selected: list[ast.stmt] = list(selected_functions)
    canonical = ast.dump(
        ast.Module(body=selected, type_ignores=[]),
        annotate_fields=True,
        include_attributes=False,
    )
    _MEASUREMENT_IMPLEMENTATION_SHA256 = hashlib.sha256(
        canonical.encode("utf-8")
    ).hexdigest()
    return _MEASUREMENT_IMPLEMENTATION_SHA256


def _local_evaluation_contract(
    source_language: str,
    measurement_profile: dict[str, Any],
    *,
    repeat_count: int | None = None,
) -> dict[str, Any]:
    """Describe every input that can change a reusable local measurement."""
    required_repeats = LOCAL_REPEAT_COUNT if repeat_count is None else repeat_count
    return {
        "schema_version": LOCAL_EVAL_CACHE_SCHEMA_VERSION,
        "measurement_contract_version": LOCAL_EVAL_MEASUREMENT_CONTRACT_VERSION,
        "measurement_implementation_sha256": _measurement_implementation_hash(),
        "kernel_id": OFFICIAL_KERNEL_ID,
        "source_language": source_language,
        "measurement_profile": measurement_profile,
        "repeat_count": required_repeats,
        "max_local_attempts": max(
            required_repeats,
            int(os.environ.get("SOL58_MAX_LOCAL_ATTEMPTS", str(required_repeats + 3))),
        ),
        "compile_timeout_s": COMPILE_TIMEOUT,
        "run_timeout_s": RUN_TIMEOUT,
        "problem_contract_sha256": _problem_contract_hash(),
        "sol_execbench": _tool_path_identity(SOL_EXECBENCH),
        "toolchain_id": os.environ.get(
            "SOL58_LOCAL_EVAL_STACK_ID", measurement_profile.get("local_eval_stack", "")
        ),
        "cuda_home": os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH") or "",
        "torch_cuda_arch_list": os.environ.get("TORCH_CUDA_ARCH_LIST", ""),
        "cuda_gencode_flags": (
            _detect_cuda_gencode_flags()
            if source_language == SOURCE_LANGUAGE_CUDA
            else []
        ),
        "python": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
    }


def _local_evaluation_cache_path(
    kernel_source: str,
    source_language: str,
    measurement_profile: dict[str, Any],
    *,
    repeat_count: int | None = None,
) -> tuple[Path, dict[str, Any], str]:
    contract = _local_evaluation_contract(
        source_language,
        measurement_profile,
        repeat_count=repeat_count,
    )
    canonical = json.dumps(contract, sort_keys=True, separators=(",", ":"))
    contract_hash = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    key = hashlib.sha256(
        json.dumps(
            {
                "source_sha256": _kernel_source_hash(kernel_source),
                "contract_sha256": contract_hash,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    return EVAL_ROOT / "local_cache" / f"{key}.json", contract, contract_hash


def _cached_local_evaluation_is_valid(
    payload: Any,
    *,
    source_sha256: str,
    contract_sha256: str,
    repeat_count: int | None = None,
) -> bool:
    required_repeats = LOCAL_REPEAT_COUNT if repeat_count is None else repeat_count
    if not isinstance(payload, dict):
        return False
    latencies = payload.get("local_latencies_ms")
    parsed_runs = payload.get("parsed_runs")
    return bool(
        payload.get("complete")
        and payload.get("schema_version") == LOCAL_EVAL_CACHE_SCHEMA_VERSION
        and payload.get("source_sha256") == source_sha256
        and payload.get("contract_sha256") == contract_sha256
        and isinstance(latencies, list)
        and len(latencies) == required_repeats
        and all(isinstance(value, (int, float)) and value > 0 for value in latencies)
        and isinstance(parsed_runs, list)
        and len(parsed_runs) == required_repeats
    )


def _collect_local_evaluation(
    *,
    workspace: Path,
    kernel_source: str,
    source_language: str,
    source_sha256: str,
    measurement_profile: dict[str, Any],
    program_path: str,
    start: float,
    repeat_count: int | None = None,
    cache_enabled: bool | None = None,
) -> dict[str, Any]:
    """Run the local pipeline with facade hooks kept patchable by evaluator tests."""
    config = LocalEvaluationConfig(
        repeat_count=LOCAL_REPEAT_COUNT if repeat_count is None else repeat_count,
        cache_enabled=LOCAL_EVAL_CACHE if cache_enabled is None else cache_enabled,
        sol_execbench=SOL_EXECBENCH,
        target_latency_ms=TARGET_LATENCY_MS,
        cache_schema_version=LOCAL_EVAL_CACHE_SCHEMA_VERSION,
    )
    request = LocalEvaluationRequest(
        workspace=workspace,
        kernel_source=kernel_source,
        source_language=source_language,
        source_sha256=source_sha256,
        measurement_profile=measurement_profile,
        program_path=program_path,
        started_at=start,
    )
    hooks = LocalEvaluationHooks(
        copy_problem_files=_copy_problem_files,
        write_solution=_write_solution,
        load_workload_count=_load_workload_count,
        cache_path=_local_evaluation_cache_path,
        file_lock=_file_lock,
        read_json=_read_json,
        cache_is_valid=_cached_local_evaluation_is_valid,
        ensure_clocks=_ensure_measurement_clocks,
        run_monitored=_run_sol_execbench_monitored,
        parse_traces=_parse_traces,
        set_clocks=_set_measurement_clocks,
        result=_result,
        tail=_tail,
        atomic_write_json=_atomic_write_json,
    )
    return collect_local_evaluation(request, config, hooks)


def _gate_uncertainty_band(
    candidate_latencies_ms: list[float],
    incumbent_latencies_ms: list[float],
) -> float:
    return local_gate.uncertainty_band(
        candidate_latencies_ms,
        incumbent_latencies_ms,
        policy=_local_gate_policy(),
    )


def _classify_gate_delta(relative_delta: float, uncertainty: float) -> str:
    return local_gate.classify_delta(relative_delta, uncertainty)


def _local_gate_policy() -> local_gate.LocalGatePolicy:
    return local_gate.LocalGatePolicy(
        enabled=LOCAL_BEST_GATE,
        sigma_multiplier=LOCAL_GATE_SIGMA_MULTIPLIER,
        relative_noise_floor=LOCAL_GATE_RELATIVE_NOISE_FLOOR,
        recheck_pairs=LOCAL_GATE_RECHECK_PAIRS,
        uncertain_relative_tolerance=LOCAL_GATE_UNCERTAIN_RELATIVE_TOLERANCE,
        allow_uncertain_official=LOCAL_GATE_ALLOW_UNCERTAIN_OFFICIAL,
        challenger_cooldown_seconds=LOCAL_GATE_CHALLENGER_COOLDOWN_S,
    )


def _read_local_best_source(best: dict[str, Any]) -> tuple[str, str] | None:
    paths: list[Path] = []
    if best.get("kernel_path"):
        paths.append(Path(str(best["kernel_path"])))
    if best.get("workspace"):
        workspace = Path(str(best["workspace"]))
        paths.extend((workspace / "kernel.cu", workspace / "kernel.py"))
    expected_hash = str(best.get("kernel_sha256") or "")
    for path in paths:
        if not path.is_file():
            continue
        try:
            source = path.read_text(encoding="utf-8")
        except OSError:
            continue
        if expected_hash and _kernel_source_hash(source) != expected_hash:
            continue
        language = (
            str(best.get("source_language") or "").strip().lower()
            or _language_from_source_path(path.name)
            or _detect_source_language(source)
            or SOURCE_LANGUAGE_CUDA
        )
        return source, language
    return None


def _claim_uncertain_challenger(source_sha256: str) -> tuple[bool, str]:
    return local_gate.claim_uncertain_challenger(
        EVAL_ROOT / "official_cache" / "local_gate_challengers.json",
        source_sha256,
        policy=_local_gate_policy(),
        format_timestamp=_format_timestamp,
    )


def _claim_architecture_official_probe(
    *,
    kernel_source: str,
    source_sha256: str,
    local_best: dict[str, Any] | None,
    candidate_latency_ms: float,
) -> dict[str, Any]:
    incumbent_latency_ms = float((local_best or {}).get("latency_ms_median") or 0.0)
    if incumbent_latency_ms <= 0:
        return {"claimed": False, "reason": "no valid local incumbent"}
    incumbent = _read_local_best_source(local_best or {})
    incumbent_source = incumbent[0] if incumbent else None
    evidence = architecture_probe_evidence(kernel_source, incumbent_source)
    return claim_official_probe(
        EVAL_ROOT / "official_cache" / "official_probes.json",
        source_sha256=source_sha256,
        relative_regression=candidate_latency_ms / incumbent_latency_ms - 1.0,
        evidence=evidence,
        policy=OFFICIAL_PROBE_POLICY,
    )


def _evaluate_local_best_gate(
    *,
    workspace: Path,
    kernel_source: str,
    source_language: str,
    source_sha256: str,
    candidate_latencies_ms: list[float],
    local_best: dict[str, Any] | None,
    measurement_profile: dict[str, Any],
    program_path: str,
    start: float,
) -> dict[str, Any]:
    return local_gate.evaluate_local_best_gate(
        workspace=workspace,
        kernel_source=kernel_source,
        source_language=source_language,
        source_sha256=source_sha256,
        candidate_latencies_ms=candidate_latencies_ms,
        local_best=local_best,
        measurement_profile=measurement_profile,
        program_path=program_path,
        start=start,
        policy=_local_gate_policy(),
        read_local_best_source=_read_local_best_source,
        kernel_source_hash=_kernel_source_hash,
        collect_local_evaluation=_collect_local_evaluation,
        claim_challenger=_claim_uncertain_challenger,
    )


def _official_token() -> str:
    return (
        os.environ.get("SOL58_SOLBENCH_TOKEN") or os.environ.get("SOLBENCH_TOKEN") or ""
    )


def _official_api_config() -> official_protocol.OfficialApiConfig:
    return official_protocol.OfficialApiConfig(
        base_url=OFFICIAL_BASE_URL,
        request_timeout=OFFICIAL_REQUEST_TIMEOUT,
        poll_interval=OFFICIAL_POLL_INTERVAL,
        pending_result_grace=OFFICIAL_PENDING_RESULT_GRACE,
        async_refresh_delay=OFFICIAL_ASYNC_REFRESH_DELAY,
        poll_timeout=OFFICIAL_POLL_TIMEOUT,
        kernel_id=OFFICIAL_KERNEL_ID,
        gpu_type=OFFICIAL_GPU_TYPE,
        evaluation_stack_version=OFFICIAL_EVAL_STACK_VERSION,
        terminal_statuses=frozenset(OFFICIAL_TERMINAL_STATUSES),
    )


def _official_compile_options(solution: dict[str, Any]) -> dict[str, Any]:
    return official_protocol.official_compile_options(solution)


def _build_official_submission(workspace: Path) -> dict[str, Any]:
    """Build a self-contained submission.json for the official B200 evaluator."""
    return official_protocol.build_official_submission(
        workspace,
        gpu_type=OFFICIAL_GPU_TYPE,
        cute_language=SOURCE_LANGUAGE_CUTE,
    )


def _multipart_form(
    fields: dict[str, str],
    file_field: str,
    filename: str,
    content: bytes,
    content_type: str,
) -> tuple[bytes, str]:
    return official_protocol.multipart_form(
        fields,
        file_field,
        filename,
        content,
        content_type,
    )


def _http_json(
    method: str,
    path: str,
    token: str,
    body: bytes | None = None,
    headers: dict[str, str] | None = None,
    request_timeout: float | None = None,
) -> dict[str, Any]:
    return official_protocol.http_json(
        method,
        path,
        token,
        base_url=OFFICIAL_BASE_URL,
        maximum_timeout=OFFICIAL_REQUEST_TIMEOUT,
        body=body,
        headers=headers,
        request_timeout=request_timeout,
    )


def _parse_timestamp(value: Any) -> float | None:
    return official_protocol.parse_timestamp(value)


def _format_timestamp(value: float) -> str:
    return official_protocol.format_timestamp(value)


def _has_official_score(data: dict[str, Any]) -> bool:
    return official_protocol.has_official_score(data)


def _official_status(data: dict[str, Any]) -> str:
    return official_protocol.official_status(data)


def _is_stale_pending_result(data: dict[str, Any], now: float | None = None) -> bool:
    return official_protocol.is_stale_pending_result(
        data,
        pending_result_grace=OFFICIAL_PENDING_RESULT_GRACE,
        now=time.time() if now is None else now,
    )


def _is_deferred_pending_result(
    data: dict[str, Any], deadline: float, now: float | None = None
) -> bool:
    return official_protocol.is_deferred_pending_result(
        data,
        deadline=deadline,
        poll_interval=OFFICIAL_POLL_INTERVAL,
        now=time.time() if now is None else now,
    )


def _normalize_submission_data(payload: dict[str, Any]) -> dict[str, Any]:
    return official_protocol.normalize_submission_data(payload)


def _get_official_submission(
    submission_id: Any,
    token: str,
    *,
    request_timeout: float | None = None,
) -> dict[str, Any]:
    return official_protocol.get_official_submission(
        submission_id,
        token,
        config=_official_api_config(),
        request=_http_json,
        request_timeout=request_timeout,
    )


def _write_official_result(workspace: Path, result: dict[str, Any]) -> None:
    official_protocol.write_official_result(workspace, result)


def _poll_official_submission(
    submission_id: Any,
    token: str,
    workspace: Path,
    upload: dict[str, Any] | None = None,
    poll_timeout: float | None = None,
) -> dict[str, Any]:
    return official_protocol.poll_official_submission(
        submission_id,
        token,
        workspace,
        config=_official_api_config(),
        get_submission=_get_official_submission,
        upload=upload,
        poll_timeout=poll_timeout,
        sleep=time.sleep,
        wall_time=time.time,
    )


def _calibration_path() -> Path:
    return _fitness_calibration().path


def _fitness_calibration() -> FitnessCalibration:
    return FitnessCalibration(
        CalibrationContext(
            root=EVAL_ROOT,
            target_latency_ms=TARGET_LATENCY_MS,
            evaluation_stack_version=OFFICIAL_EVAL_STACK_VERSION,
            gpu_type=OFFICIAL_GPU_TYPE,
            half_life_hours=max(
                1.0,
                float(os.environ.get("SOL58_CALIBRATION_HALF_LIFE_HOURS", "336")),
            ),
            measurement_profile=_measurement_profile,
            parse_traces=_parse_traces,
            official_status=_official_status,
            parse_timestamp=_parse_timestamp,
        )
    )


def _calibration_scope() -> dict[str, str]:
    return _fitness_calibration().scope()


def _discover_official_calibration_ratios() -> list[tuple[float, float]]:
    """Recover only same-profile calibration from completed legacy workspaces."""
    return _fitness_calibration().discover_ratios()


def _recency_weighted_median(samples: list[tuple[float, float]]) -> float:
    return _fitness_calibration().recency_weighted_median(samples)


def _load_official_calibration_ratio() -> float:
    return _fitness_calibration().load_ratio(os.environ)


def _record_official_calibration(
    local_score: float,
    official_score: float,
    local_latency_ms: float,
    official_latency_ms: float,
    submission_id: Any,
    measurement_profile_id: str | None = None,
    submission_reason: str = "local_best_gate",
) -> None:
    _fitness_calibration().record(
        local_score=local_score,
        official_score=official_score,
        local_latency_ms=local_latency_ms,
        official_latency_ms=official_latency_ms,
        submission_id=submission_id,
        measurement_profile_id=measurement_profile_id,
        submission_reason=submission_reason,
    )


def _project_provisional_search_score(search_score: float) -> float:
    """Keep provisional ranking strict while reserving target crossing for official fitness."""
    return project_provisional_score(
        search_score,
        target=OFFICIAL_TARGET_SCORE,
        floor=OFFICIAL_PROVISIONAL_SCORE_CAP,
    )


def _provisional_official_score(local_score: float) -> tuple[float, float]:
    ratio = _load_official_calibration_ratio()
    search_score = max(0.0, local_score * ratio)
    return _project_provisional_search_score(search_score), ratio


def _local_provisional_score(local_score: float) -> float:
    """Keep local candidates ordered without allowing them to certify the target."""
    return _project_provisional_search_score(local_score)


def _official_fitness_anchor(
    local_best: dict[str, Any] | None,
) -> dict[str, Any] | None:
    """Return a completed official score and its corresponding local-latency anchor."""
    return official_fitness_anchor(local_best)


def _anchored_provisional_score(
    local_score: float,
    candidate_latency_ms: float,
    local_best: dict[str, Any] | None,
) -> tuple[float, dict[str, Any]]:
    """Map local latency onto the incumbent's official scale for PES comparisons."""
    anchor = _official_fitness_anchor(local_best)
    if anchor is None or candidate_latency_ms <= 0:
        score, calibration_ratio = _provisional_official_score(local_score)
        return score, {
            "source": "historical_calibration",
            "calibration_ratio": calibration_ratio,
            "search_score": max(0.0, local_score * calibration_ratio),
            "selection_score": score,
        }

    latency_ratio = float(anchor["local_latency_ms"]) / candidate_latency_ms
    search_score = max(0.0, float(anchor["score"]) * latency_ratio)
    selection_score = _project_provisional_search_score(search_score)
    return selection_score, {
        "source": "incumbent_official_anchor",
        "anchor_score": float(anchor["score"]),
        "anchor_local_latency_ms": float(anchor["local_latency_ms"]),
        "anchor_submission_id": anchor.get("submission_id"),
        "anchor_record_source": anchor["source"],
        "candidate_to_anchor_latency_ratio": latency_ratio,
        "search_score": search_score,
        "selection_score": selection_score,
    }


def _cache_path(kernel_source: str) -> Path:
    cache_key = hashlib.sha256(
        json.dumps(
            {
                "kernel_sha256": hashlib.sha256(
                    kernel_source.encode("utf-8")
                ).hexdigest(),
                "kernel_id": OFFICIAL_KERNEL_ID,
                "gpu_type": OFFICIAL_GPU_TYPE,
                "eval_stack": OFFICIAL_EVAL_STACK_VERSION,
                "mode": OFFICIAL_SUBMISSION_MODE,
            },
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()
    return EVAL_ROOT / "official_cache" / f"{cache_key}.json"


def _official_cache_metadata(
    kernel_source: str,
    *,
    source_language: str,
    local_score: float,
    local_latency_ms: float,
    measurement_profile: dict[str, Any],
    submission_reason: str = "local_best_gate",
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "source_sha256": _kernel_source_hash(kernel_source),
        "source_language": source_language,
        "local_score": float(local_score),
        "local_latency_ms": float(local_latency_ms),
        "measurement_profile_id": str(measurement_profile["id"]),
        "kernel_id": OFFICIAL_KERNEL_ID,
        "gpu_type": OFFICIAL_GPU_TYPE,
        "evaluation_stack_version": OFFICIAL_EVAL_STACK_VERSION,
        "submission_reason": submission_reason,
    }


def _with_official_cache_metadata(
    result: dict[str, Any], metadata: dict[str, Any]
) -> dict[str, Any]:
    merged = dict(result)
    merged["_atrex"] = dict(metadata)
    return merged


def _submit_official_unlocked(
    workspace: Path,
    kernel_source: str,
    *,
    source_language: str = SOURCE_LANGUAGE_CUDA,
    local_score: float = 0.0,
    local_latency_ms: float = 0.0,
    measurement_profile: dict[str, Any] | None = None,
    allow_upload: bool = True,
    submission_reason: str = "local_best_gate",
) -> dict[str, Any]:
    token = _official_token()
    if not token:
        raise RuntimeError(
            "SOL58_OFFICIAL_FITNESS=1 requires SOLBENCH_TOKEN or SOL58_SOLBENCH_TOKEN"
        )

    cache_path = _cache_path(kernel_source)
    cache_metadata = _official_cache_metadata(
        kernel_source,
        source_language=source_language,
        local_score=local_score,
        local_latency_ms=local_latency_ms,
        measurement_profile=measurement_profile or _measurement_profile(),
        submission_reason=submission_reason,
    )
    if OFFICIAL_CACHE and cache_path.exists():
        cached = _read_json(cache_path, {})
        existing_metadata = cached.get("_atrex") if isinstance(cached, dict) else None
        if isinstance(existing_metadata, dict) and existing_metadata.get(
            "submission_reason"
        ):
            cache_metadata["submission_reason"] = existing_metadata["submission_reason"]
        cached = _with_official_cache_metadata(cached, cache_metadata)
        cached_status = _official_status(cached)
        next_refresh_at = _parse_timestamp(cached.get("next_refresh_at"))
        refresh_due = next_refresh_at is None or next_refresh_at <= time.time()
        if (
            cached_status in OFFICIAL_REFRESHABLE_STATUSES
            and cached.get("id")
            and OFFICIAL_CACHE_REFRESH_TIMEOUT >= 0
            and refresh_due
        ):
            refreshed = _poll_official_submission(
                cached["id"],
                token,
                workspace,
                upload=cached.get("upload") or {},
                poll_timeout=OFFICIAL_CACHE_REFRESH_TIMEOUT,
            )
            cached = _with_official_cache_metadata(refreshed, cache_metadata)
            _atomic_write_json(cache_path, cached)
            cached_status = _official_status(cached)

        if cached_status in OFFICIAL_TERMINAL_STATUSES or _has_official_score(cached):
            cached["cache_hit"] = True
            _atomic_write_json(cache_path, cached)
            _write_official_result(workspace, cached)
            return cached

        if not allow_upload:
            cached["cache_hit"] = True
            _atomic_write_json(cache_path, cached)
            _write_official_result(workspace, cached)
            return cached

    if not allow_upload:
        raise RuntimeError(
            "No cached official submission exists for the incumbent kernel"
        )

    submission = _build_official_submission(workspace)
    submission_path = workspace / "official_submission.json"
    submission_path.write_text(
        json.dumps(submission, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    body, content_type = _multipart_form(
        {
            "kernel_id": str(OFFICIAL_KERNEL_ID),
            "gpu_type": OFFICIAL_GPU_TYPE,
            "submission_mode": OFFICIAL_SUBMISSION_MODE,
        },
        "file",
        "submission.json",
        submission_path.read_bytes(),
        "application/json",
    )
    upload = _http_json(
        "POST",
        "/api/submissions/upload",
        token,
        body=body,
        headers={"Content-Type": content_type, "Content-Length": str(len(body))},
    )
    (workspace / "official_upload_response.json").write_text(
        json.dumps(upload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    submission_id = (upload.get("data") or {}).get("submission_id")
    if not submission_id:
        raise RuntimeError(f"official upload response missing submission_id: {upload}")

    if OFFICIAL_ASYNC_SUBMIT:
        now = time.time()
        last = {
            "id": submission_id,
            "status": "DEFERRED_RESULT",
            "upstream_status": "QUEUED",
            "submitted_at": _format_timestamp(now),
            "next_refresh_at": _format_timestamp(now + OFFICIAL_ASYNC_REFRESH_DELAY),
            "upload": upload,
            "cache_hit": False,
            "evaluation_stack_version": OFFICIAL_EVAL_STACK_VERSION,
            "error_log": (
                "official submission accepted; result collection deferred so PES does not "
                "wait for the v1.1 worker"
            ),
        }
        _write_official_result(workspace, last)
    else:
        last = _poll_official_submission(submission_id, token, workspace, upload=upload)

    last = _with_official_cache_metadata(last, cache_metadata)
    if OFFICIAL_CACHE:
        _atomic_write_json(cache_path, last)
    return last


def _submit_official(
    workspace: Path,
    kernel_source: str,
    *,
    source_language: str = SOURCE_LANGUAGE_CUDA,
    local_score: float = 0.0,
    local_latency_ms: float = 0.0,
    measurement_profile: dict[str, Any] | None = None,
    allow_upload: bool = True,
    submission_reason: str = "local_best_gate",
) -> dict[str, Any]:
    cache_path = _cache_path(kernel_source)
    with _file_lock(cache_path):
        return _submit_official_unlocked(
            workspace,
            kernel_source,
            source_language=source_language,
            local_score=local_score,
            local_latency_ms=local_latency_ms,
            measurement_profile=measurement_profile,
            allow_upload=allow_upload,
            submission_reason=submission_reason,
        )


def _official_metadata_from_local_best(result: dict[str, Any]) -> dict[str, Any]:
    path = _local_best_path()
    if not path.is_file():
        return {}
    with _file_lock(path):
        best = _read_json(path, {})
    if not isinstance(best, dict) or str(best.get("official_submission_id")) != str(
        result.get("id")
    ):
        return {}
    source_hash = str(best.get("kernel_sha256") or "")
    if not source_hash:
        return {}
    return {
        "schema_version": 1,
        "source_sha256": source_hash,
        "source_language": str(best.get("source_language") or SOURCE_LANGUAGE_CUDA),
        "local_score": float(best.get("local_score") or 0.0),
        "local_latency_ms": float(best.get("latency_ms_median") or 0.0),
        "measurement_profile_id": str(best.get("measurement_profile_id") or ""),
        "kernel_id": OFFICIAL_KERNEL_ID,
        "gpu_type": OFFICIAL_GPU_TYPE,
        "evaluation_stack_version": OFFICIAL_EVAL_STACK_VERSION,
        "submission_reason": str(
            best.get("official_submission_reason") or "local_best_gate"
        ),
    }


def _update_local_best_from_official(
    metadata: dict[str, Any], result: dict[str, Any]
) -> bool:
    path = _local_best_path()
    if not path.is_file():
        return False
    source_hash = str(metadata.get("source_sha256") or "")
    with _file_lock(path):
        best = _read_json(path, {})
        if (
            not isinstance(best, dict)
            or str(best.get("kernel_sha256") or "") != source_hash
        ):
            return False
        official_score = float(result.get("sol_score") or 0.0)
        official_latency_ms = float(result.get("latency_ms") or 0.0)
        official_status = _official_status(result)
        is_correct = bool(result.get("is_correct"))
        best.update(
            {
                "remote_submitted": True,
                "official_submission_id": result.get("id"),
                "official_status": official_status,
                "official_is_correct": is_correct,
                "official_score": official_score,
                "official_latency_ms": official_latency_ms,
                "updated_at": datetime.now(timezone.utc).isoformat(),
            }
        )
        if official_status == "COMPLETED" and is_correct and official_score > 0:
            best.update(
                {
                    "official_anchor_score": official_score,
                    "official_anchor_latency_ms": float(
                        metadata.get("local_latency_ms") or 0.0
                    ),
                    "official_anchor_submission_id": result.get("id"),
                }
            )
        _atomic_write_json(path, best)
    return True


def _refresh_due_official_results() -> dict[str, Any]:
    """Refresh a bounded set of pending submissions and publish completed fitness."""
    report: dict[str, Any] = {
        "enabled": bool(OFFICIAL_FITNESS and OFFICIAL_CACHE),
        "checked": 0,
        "completed": 0,
        "failed": 0,
        "pending": 0,
        "time_budget_s": OFFICIAL_REFRESH_TIME_BUDGET,
        "budget_exhausted": False,
        "errors": [],
    }
    token = _official_token()
    if not report["enabled"] or not token:
        return report

    cache_root = EVAL_ROOT / "official_cache"
    paths = sorted(
        (
            path
            for path in cache_root.glob("*.json")
            if re.fullmatch(r"[0-9a-f]{64}\.json", path.name)
        ),
        key=lambda path: path.stat().st_mtime,
    )
    now = time.time()
    refresh_started = time.monotonic()
    for cache_path in paths:
        if report["checked"] >= OFFICIAL_REFRESH_BATCH_SIZE:
            break
        with _file_lock(cache_path):
            cached = _read_json(cache_path, {})
            if not isinstance(cached, dict):
                continue
            status = _official_status(cached)
            due_at = _parse_timestamp(cached.get("next_refresh_at"))
            if (
                status not in OFFICIAL_REFRESHABLE_STATUSES
                or not cached.get("id")
                or (due_at is not None and due_at > now)
            ):
                continue
            metadata = cached.get("_atrex")
            if not isinstance(metadata, dict) or not metadata.get("source_sha256"):
                metadata = _official_metadata_from_local_best(cached)
            if not metadata:
                continue

            elapsed = time.monotonic() - refresh_started
            remaining_budget = OFFICIAL_REFRESH_TIME_BUDGET - elapsed
            if remaining_budget <= 0:
                report["budget_exhausted"] = True
                break
            report["checked"] += 1
            try:
                remote = _get_official_submission(
                    cached["id"],
                    token,
                    request_timeout=remaining_budget,
                )
                refreshed = dict(cached)
                refreshed.update(
                    {key: value for key, value in remote.items() if value is not None}
                )
                refreshed["_atrex"] = metadata
                refreshed_status = _official_status(refreshed)
                is_authoritative = bool(
                    refreshed_status == "COMPLETED"
                    and refreshed.get("is_correct")
                    and float(refreshed.get("sol_score") or 0.0) > 0
                )
                is_final_failure = bool(
                    refreshed_status in OFFICIAL_TERMINAL_STATUSES
                    and refreshed_status not in OFFICIAL_REFRESHABLE_STATUSES
                    and not is_authoritative
                )
                if is_authoritative:
                    refreshed.pop("next_refresh_at", None)
                    _record_authoritative_fitness_by_hash(
                        str(metadata["source_sha256"]),
                        source_language=str(
                            metadata.get("source_language") or SOURCE_LANGUAGE_CUDA
                        ),
                        official_score=float(refreshed["sol_score"]),
                        official_latency_ms=float(refreshed.get("latency_ms") or 0.0),
                        local_latency_ms=float(metadata.get("local_latency_ms") or 0.0),
                        submission_id=refreshed.get("id"),
                        measurement_profile_id=str(
                            metadata.get("measurement_profile_id") or ""
                        ),
                    )
                    _record_official_calibration(
                        local_score=float(metadata.get("local_score") or 0.0),
                        official_score=float(refreshed["sol_score"]),
                        local_latency_ms=float(metadata.get("local_latency_ms") or 0.0),
                        official_latency_ms=float(refreshed.get("latency_ms") or 0.0),
                        submission_id=refreshed.get("id"),
                        measurement_profile_id=str(
                            metadata.get("measurement_profile_id") or ""
                        ),
                        submission_reason=str(
                            metadata.get("submission_reason") or "local_best_gate"
                        ),
                    )
                    _update_local_best_from_official(metadata, refreshed)
                    report["completed"] += 1
                elif is_final_failure:
                    refreshed.pop("next_refresh_at", None)
                    _record_authoritative_fitness_by_hash(
                        str(metadata["source_sha256"]),
                        source_language=str(
                            metadata.get("source_language") or SOURCE_LANGUAGE_CUDA
                        ),
                        official_score=0.0,
                        official_latency_ms=float(refreshed.get("latency_ms") or 0.0),
                        local_latency_ms=float(metadata.get("local_latency_ms") or 0.0),
                        submission_id=refreshed.get("id"),
                        measurement_profile_id=str(
                            metadata.get("measurement_profile_id") or ""
                        ),
                        official_status=refreshed_status,
                        is_correct=bool(refreshed.get("is_correct")),
                    )
                    _update_local_best_from_official(metadata, refreshed)
                    report["failed"] += 1
                else:
                    available_at = _parse_timestamp(
                        refreshed.get("result_available_at")
                    )
                    next_refresh = max(
                        time.time() + OFFICIAL_ASYNC_REFRESH_DELAY,
                        available_at or 0.0,
                    )
                    refreshed["next_refresh_at"] = _format_timestamp(next_refresh)
                    report["pending"] += 1
                _atomic_write_json(cache_path, refreshed)
            except Exception as exc:
                cached["next_refresh_at"] = _format_timestamp(
                    time.time() + OFFICIAL_ASYNC_REFRESH_DELAY
                )
                cached["refresh_error"] = _tail(str(exc), 500)
                _atomic_write_json(cache_path, cached)
                report["errors"].append(_tail(str(exc), 300))
    report["elapsed_s"] = time.monotonic() - refresh_started
    return report


def _official_fitness_pipeline_config() -> OfficialFitnessConfig:
    return OfficialFitnessConfig(
        target_latency_ms=TARGET_LATENCY_MS,
        minimum_local_score=OFFICIAL_MIN_LOCAL_SCORE,
        maximum_local_latency_ms=OFFICIAL_MAX_LOCAL_LATENCY_MS,
        local_best_gate=LOCAL_BEST_GATE,
        pending_score_policy=OFFICIAL_PENDING_SCORE_POLICY,
        provisional_projection_floor=OFFICIAL_PROVISIONAL_SCORE_CAP,
        evaluation_stack_version=OFFICIAL_EVAL_STACK_VERSION,
        gpu_type=OFFICIAL_GPU_TYPE,
        submission_mode=OFFICIAL_SUBMISSION_MODE,
        refreshable_statuses=frozenset(OFFICIAL_REFRESHABLE_STATUSES),
    )


def _official_fitness_pipeline_hooks() -> OfficialFitnessHooks:
    return OfficialFitnessHooks(
        result=_result,
        fitness_anchor=_official_fitness_anchor,
        record_authoritative_source=_record_authoritative_source_fitness,
        anchored_provisional_score=_anchored_provisional_score,
        submit_official=_submit_official,
        official_status=_official_status,
        save_local_best=_save_local_best,
        record_calibration=_record_official_calibration,
        local_provisional_score=_local_provisional_score,
        provisional_official_score=_provisional_official_score,
    )


def evaluate(program_path: str) -> dict[str, Any]:
    start = time.time()
    eval_id = uuid.uuid4().hex[:12]
    workspace = EVAL_ROOT / f"eval_{eval_id}"
    workspace.mkdir(parents=True, exist_ok=True)
    official_refresh = _refresh_due_official_results()

    try:
        raw = Path(program_path).read_text(encoding="utf-8")
        code_language = _normalize_code_language()
        required_source_language = _required_source_language(
            program_path,
            code_language,
        )
        kernel_source = _extract_kernel_source(raw, code_language)
        source_language, validation_error = _candidate_validation_error(
            kernel_source,
            code_language,
        )
        if (
            validation_error is None
            and required_source_language is not None
            and source_language != required_source_language
        ):
            iteration = _program_iteration(program_path)
            validation_error = (
                f"iteration {iteration} is a mandatory {required_source_language} slot under "
                f"SOL58_CUTEDSL_GENERATION_RATE={CUTEDSL_GENERATION_RATE}; received "
                f"{source_language}"
            )

        if not kernel_source or validation_error or source_language is None:
            return _result(
                "validation_failed",
                f"Generated candidate violates the {code_language} source contract: "
                f"{validation_error or 'empty source'}.",
                0.0,
                metrics={
                    "eval_time_s": time.time() - start,
                    "configured_code_language": code_language,
                    "detected_source_language": source_language,
                    "required_source_language": required_source_language,
                    "measurement_profile": _measurement_profile(),
                },
                artifacts={"workspace": str(workspace), "program_path": program_path},
            )

        kernel_sha256 = _kernel_source_hash(kernel_source)
        measurement_profile = _measurement_profile()
        local_best_before = _load_local_best() if LOCAL_BEST_GATE else None
        local_best_hash = (
            str(local_best_before.get("kernel_sha256") or "")
            if local_best_before
            else ""
        )
        same_as_local_best = bool(local_best_hash and kernel_sha256 == local_best_hash)
        local_best_profile_matches = _local_best_profile_matches(
            local_best_before, measurement_profile
        )
        profile_recalibration = bool(
            local_best_before and same_as_local_best and not local_best_profile_matches
        )
        if (
            local_best_before
            and not local_best_profile_matches
            and not same_as_local_best
        ):
            return _result(
                "framework_error",
                "Persisted local best belongs to a different measurement profile. "
                "Re-evaluate its exact kernel before comparing or submitting a new candidate.",
                0.0,
                metrics={
                    "eval_time_s": time.time() - start,
                    "measurement_profile": measurement_profile,
                    "local_best_measurement_profile_id": local_best_before.get(
                        "measurement_profile_id"
                    ),
                    "profile_recalibration_required": True,
                },
                artifacts={
                    "workspace": str(workspace),
                    "program_path": program_path,
                    "local_best_kernel_path": local_best_before.get("kernel_path"),
                },
            )
        recorded_anchor = _official_fitness_anchor(local_best_before)
        if (
            OFFICIAL_FITNESS
            and same_as_local_best
            and local_best_profile_matches
            and local_best_before is not None
            and recorded_anchor is not None
            and recorded_anchor["source"] == "completed_local_best"
        ):
            recorded_score = float(recorded_anchor["score"])
            recorded_latency_ms = float(
                local_best_before.get("latency_ms_median") or 0.0
            )
            _record_authoritative_source_fitness(
                kernel_source,
                source_language=source_language,
                official_score=recorded_score,
                official_latency_ms=float(
                    local_best_before.get("official_latency_ms") or 0.0
                ),
                local_latency_ms=recorded_latency_ms,
                submission_id=recorded_anchor.get("submission_id"),
            )
            summary = (
                "Candidate exactly matches the persisted standalone local-best kernel; "
                f"reused its completed {local_best_before.get('repeat_count') or LOCAL_REPEAT_COUNT}-repeat "
                f"local median {recorded_latency_ms:.6f} ms and official v1.1 "
                f"sol_score={recorded_score:.6f}, "
                f"submission_id={recorded_anchor.get('submission_id')}."
            )
            reused_metrics = {
                "eval_time_s": time.time() - start,
                "source_language": source_language,
                "required_source_language": required_source_language,
                "measurement_profile": measurement_profile,
                "official_refresh": official_refresh,
                "local_eval_reused": True,
                "latency_ms_geomean": recorded_latency_ms,
                "local_repeat_count": int(
                    local_best_before.get("repeat_count") or LOCAL_REPEAT_COUNT
                ),
                "local_best": {
                    "gate_enabled": True,
                    "candidate_kernel_sha256": kernel_sha256,
                    "previous_latency_ms_median": recorded_latency_ms,
                    "same_kernel": True,
                    "strictly_improved": False,
                    "measurement_profile_matches": True,
                },
                "official": {
                    "enabled": True,
                    "submitted": False,
                    "status": "REUSED_LOCAL_BEST_OFFICIAL",
                    "authoritative": True,
                    "fitness_source": "official_local_best_record",
                    "submission_id": recorded_anchor.get("submission_id"),
                    "sol_score": recorded_score,
                    "record_hit": True,
                    "cache_hit": False,
                },
            }
            reused_artifacts = {
                "workspace": str(workspace),
                "program_path": program_path,
                "source_language": source_language,
                "local_best_kernel_path": str(
                    local_best_before.get("kernel_path")
                    or _local_best_kernel_path(source_language)
                ),
            }
            ncu_analysis = _ncu_summary_evidence(
                workspace=workspace,
                kernel_source=kernel_source,
                source_language=source_language,
                source_sha256=kernel_sha256,
                measurement_profile=measurement_profile,
                per_workload=list(local_best_before.get("per_workload") or []),
                improved=True,
                program_path=program_path,
            )
            reused_metrics["ncu_analysis"] = ncu_analysis
            if isinstance(ncu_analysis.get("artifacts"), dict):
                reused_artifacts["ncu_profile"] = ncu_analysis["artifacts"]
            return _result(
                "success",
                summary,
                recorded_score,
                metrics=reused_metrics,
                artifacts=reused_artifacts,
            )

        local_evaluation = _collect_local_evaluation(
            workspace=workspace,
            kernel_source=kernel_source,
            source_language=source_language,
            source_sha256=kernel_sha256,
            measurement_profile=measurement_profile,
            program_path=program_path,
            start=start,
        )
        if local_evaluation.get("error_result") is not None:
            return local_evaluation["error_result"]

        expected = int(local_evaluation["expected"])
        parsed_runs = list(local_evaluation["parsed_runs"])
        processes = list(local_evaluation["processes"])
        local_run_records = list(local_evaluation["local_run_records"])
        local_latencies_ms = [
            float(value) for value in local_evaluation["local_latencies_ms"]
        ]
        rejected_clock_attempts = list(
            local_evaluation.get("rejected_clock_attempts") or []
        )
        attempt_index = int(local_evaluation.get("attempt_count") or LOCAL_REPEAT_COUNT)

        latency_ms = float(statistics.median(local_latencies_ms))
        representative_index = min(
            range(len(local_latencies_ms)),
            key=lambda index: abs(local_latencies_ms[index] - latency_ms),
        )
        parsed = parsed_runs[representative_index]
        proc = processes[representative_index]
        total = int(parsed.get("total", 0))
        passed = int(parsed.get("passed", 0))
        elapsed = time.time() - start

        common_artifacts = {
            "workspace": str(workspace),
            "source_language": source_language,
            "measurement_profile_path": str(workspace / "measurement_profile.json"),
            "source_path": str(
                workspace
                / (
                    "kernel.py"
                    if source_language == SOURCE_LANGUAGE_CUTE
                    else "kernel.cu"
                )
            ),
            "returncode": proc.returncode,
            "stdout_tail": _tail(proc.stdout),
            "stderr_tail": _tail(proc.stderr),
            "per_workload": parsed.get("per_workload", []),
            "local_repeats": local_run_records,
            "rejected_clock_attempts": rejected_clock_attempts,
            "local_evaluation_cache_path": local_evaluation["cache_path"],
        }

        metrics = {
            "eval_time_s": elapsed,
            "local_eval_time_s": elapsed,
            "source_language": source_language,
            "configured_code_language": code_language,
            "required_source_language": required_source_language,
            "measurement_profile": measurement_profile,
            "official_refresh": official_refresh,
            "profile_recalibration": profile_recalibration,
            "target_latency_ms": TARGET_LATENCY_MS,
            "latency_ms_geomean": latency_ms,
            "latency_ms_geomean_repeats": local_latencies_ms,
            "latency_ms_arith_mean": float(
                statistics.median(
                    float(run.get("latency_ms_arith_mean") or 0.0)
                    for run in parsed_runs
                )
            ),
            "local_repeat_count": LOCAL_REPEAT_COUNT,
            "local_attempt_count": attempt_index,
            "clock_rejected_attempt_count": len(rejected_clock_attempts),
            "local_eval_cache": {
                "enabled": LOCAL_EVAL_CACHE,
                "hit": bool(local_evaluation.get("cache_hit")),
                "contract_sha256": local_evaluation.get("contract_sha256"),
                "source_workspace": local_evaluation.get("source_workspace"),
            },
            "passed": passed,
            "total": total,
            "expected_total": expected,
            "max_abs_err": max(
                float(run.get("max_abs_err") or 0.0) for run in parsed_runs
            ),
            "max_rel_err": max(
                float(run.get("max_rel_err") or 0.0) for run in parsed_runs
            ),
        }

        best_latency_before = (
            float(local_best_before.get("latency_ms_median") or 0.0)
            if local_best_before and local_best_profile_matches
            else 0.0
        )
        gate_result = _evaluate_local_best_gate(
            workspace=workspace,
            kernel_source=kernel_source,
            source_language=source_language,
            source_sha256=kernel_sha256,
            candidate_latencies_ms=local_latencies_ms,
            local_best=(local_best_before if local_best_profile_matches else None),
            measurement_profile=measurement_profile,
            program_path=program_path,
            start=start,
        )
        gate_latency_ms = float(gate_result.get("candidate_latency_ms") or latency_ms)
        gate_repeats_ms = [
            float(value)
            for value in (gate_result.get("candidate_repeats_ms") or local_latencies_ms)
        ]
        beats_local_best = bool(gate_result.get("update_local_best"))
        passes_official_gate = bool(gate_result.get("submit_official"))
        raw_local_score = TARGET_LATENCY_MS / latency_ms
        local_score = TARGET_LATENCY_MS / gate_latency_ms
        below_official_latency_threshold = (
            OFFICIAL_MAX_LOCAL_LATENCY_MS <= 0
            or latency_ms < OFFICIAL_MAX_LOCAL_LATENCY_MS
        )
        metrics["raw_local_score"] = raw_local_score
        metrics["local_score"] = local_score
        metrics["local_best"] = {
            "gate_enabled": LOCAL_BEST_GATE,
            "candidate_kernel_sha256": kernel_sha256,
            "candidate_latency_ms_median": latency_ms,
            "gate_latency_ms_median": gate_latency_ms,
            "previous_latency_ms_median": best_latency_before or None,
            "same_kernel": same_as_local_best,
            "strictly_improved": beats_local_best,
            "official_gate_passed": passes_official_gate,
            "below_official_latency_threshold": below_official_latency_threshold,
            "official_max_local_latency_ms": (OFFICIAL_MAX_LOCAL_LATENCY_MS or None),
            "measurement_profile_matches": local_best_profile_matches,
            "profile_recalibration": profile_recalibration,
            "improvement_ms": (
                best_latency_before - gate_latency_ms
                if best_latency_before > 0
                else None
            ),
            "uncertainty_gate": gate_result,
        }
        reuse_cached_official = bool(
            OFFICIAL_CACHE and _cache_path(kernel_source).is_file()
        )
        metrics["local_best"]["reuse_cached_official"] = reuse_cached_official
        official_probe: dict[str, Any] = {
            "claimed": False,
            "reason": "candidate did not enter the confirmed-slower probe path",
        }
        if (
            OFFICIAL_FITNESS
            and below_official_latency_threshold
            and LOCAL_BEST_GATE
            and str(gate_result.get("status") or "") == "confirmed_slower"
            and not same_as_local_best
            and not reuse_cached_official
        ):
            official_probe = _claim_architecture_official_probe(
                kernel_source=kernel_source,
                source_sha256=kernel_sha256,
                local_best=local_best_before,
                candidate_latency_ms=gate_latency_ms,
            )
            if official_probe.get("claimed"):
                passes_official_gate = True
        elif OFFICIAL_FITNESS and not below_official_latency_threshold:
            official_probe["reason"] = (
                f"local median latency {latency_ms:.6f} ms is not below the strict "
                f"upload threshold {OFFICIAL_MAX_LOCAL_LATENCY_MS:.6f} ms"
            )
        if not below_official_latency_threshold:
            passes_official_gate = False
        gate_result["official_probe"] = official_probe
        metrics["local_best"].update(
            {
                "official_gate_passed": passes_official_gate,
                "official_probe": official_probe,
            }
        )
        new_local_best_record: dict[str, Any] | None = None
        if LOCAL_BEST_GATE and beats_local_best:
            new_local_best_record = {
                "kernel_sha256": kernel_sha256,
                "source_language": source_language,
                "latency_ms_median": gate_latency_ms,
                "local_score": local_score,
                "repeat_count": len(gate_repeats_ms),
                "repeat_latencies_ms": gate_repeats_ms,
                "raw_latency_ms_median": latency_ms,
                "raw_repeat_latencies_ms": local_latencies_ms,
                "uncertainty_gate": gate_result,
                "per_workload": parsed.get("per_workload", []),
                "workspace": str(workspace),
                "source": "pes_evaluation",
                "remote_submitted": False,
                "measurement_profile_id": measurement_profile["id"],
                "measurement_profile": measurement_profile,
            }
            fitness_anchor = _official_fitness_anchor(local_best_before)
            if (
                same_as_local_best
                and local_best_before
                and str(local_best_before.get("official_status") or "").upper()
                == "COMPLETED"
                and float(local_best_before.get("official_score") or 0.0) > 0
            ):
                new_local_best_record.update(
                    {
                        "remote_submitted": bool(
                            local_best_before.get("remote_submitted", True)
                        ),
                        "official_submission_id": local_best_before.get(
                            "official_submission_id"
                        ),
                        "official_status": "COMPLETED",
                        "official_score": float(local_best_before["official_score"]),
                        "official_latency_ms": float(
                            local_best_before.get("official_latency_ms") or 0.0
                        ),
                        "official_anchor_score": float(
                            local_best_before["official_score"]
                        ),
                        "official_anchor_latency_ms": gate_latency_ms,
                        "official_anchor_submission_id": local_best_before.get(
                            "official_submission_id"
                        ),
                    }
                )
            elif fitness_anchor is not None:
                new_local_best_record.update(
                    {
                        "official_anchor_score": fitness_anchor["score"],
                        "official_anchor_latency_ms": fitness_anchor[
                            "local_latency_ms"
                        ],
                        "official_anchor_submission_id": fitness_anchor.get(
                            "submission_id"
                        ),
                    }
                )
            if not _save_local_best(new_local_best_record, kernel_source):
                concurrent_best = _load_local_best()
                concurrent_latency = float(
                    (concurrent_best or {}).get("latency_ms_median") or 0.0
                )
                metrics["local_best"].update(
                    {
                        "strictly_improved": False,
                        "lost_concurrent_update": True,
                        "previous_latency_ms_median": concurrent_latency or None,
                    }
                )
                best_latency_before = concurrent_latency
                local_best_before = concurrent_best
                beats_local_best = False
                passes_official_gate = False
                new_local_best_record = None
        ncu_analysis = _ncu_summary_evidence(
            workspace=workspace,
            kernel_source=kernel_source,
            source_language=source_language,
            source_sha256=kernel_sha256,
            measurement_profile=measurement_profile,
            per_workload=list(parsed.get("per_workload") or []),
            improved=beats_local_best,
            program_path=program_path,
        )
        metrics["ncu_analysis"] = ncu_analysis
        if isinstance(ncu_analysis.get("artifacts"), dict):
            common_artifacts["ncu_profile"] = ncu_analysis["artifacts"]
        status_line = "target met" if local_score >= 1.0 else "target not met"
        gate_note = ""
        if not math.isclose(gate_latency_ms, latency_ms, rel_tol=0.0, abs_tol=1e-15):
            gate_note = f" drift-normalized gate latency={gate_latency_ms:.6f} ms;"
        local_summary = (
            f"Local {source_language} prefilter passed all {passed}/{expected} workloads in "
            f"{LOCAL_REPEAT_COUNT}/{LOCAL_REPEAT_COUNT} repeats; median geomean latency "
            f"{latency_ms:.6f} ms from {[round(value, 6) for value in local_latencies_ms]} "
            f"vs target {TARGET_LATENCY_MS:.6f} ms;{gate_note} "
            f"local_score={local_score:.6f} ({status_line}); measurement_profile="
            f"{measurement_profile['name']}:{measurement_profile['id']}."
        )

        if OFFICIAL_FITNESS:
            return evaluate_official_fitness(
                OfficialEvaluationState(
                    workspace=workspace,
                    kernel_source=kernel_source,
                    source_language=source_language,
                    local_score=local_score,
                    local_latency_ms=latency_ms,
                    gate_latency_ms=gate_latency_ms,
                    best_latency_before=best_latency_before,
                    local_best_before=local_best_before,
                    same_as_local_best=same_as_local_best,
                    passes_official_gate=passes_official_gate,
                    reuse_cached_official=reuse_cached_official,
                    gate_result=gate_result,
                    official_probe=official_probe,
                    new_local_best_record=new_local_best_record,
                    measurement_profile=measurement_profile,
                    metrics=metrics,
                    artifacts=common_artifacts,
                    local_summary=local_summary,
                    started_at=start,
                ),
                _official_fitness_pipeline_config(),
                _official_fitness_pipeline_hooks(),
            )
        summary = local_summary.replace("Local prefilter ", "")
        return _result(
            "success",
            summary,
            local_score,
            metrics=metrics,
            artifacts=common_artifacts,
        )

    except subprocess.TimeoutExpired as exc:
        return _result(
            "execution_failed",
            f"SOL-ExecBench timed out: {exc}",
            0.0,
            metrics={
                "eval_time_s": time.time() - start,
                "target_latency_ms": TARGET_LATENCY_MS,
            },
            artifacts={"workspace": str(workspace), "program_path": program_path},
        )
    except Exception as exc:
        return _result(
            "framework_error",
            f"Evaluation failed: {exc}",
            0.0,
            metrics={
                "eval_time_s": time.time() - start,
                "target_latency_ms": TARGET_LATENCY_MS,
            },
            artifacts={
                "workspace": str(workspace),
                "program_path": program_path,
                "traceback": traceback.format_exc(),
            },
        )
