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
import urllib.error
import urllib.request
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from orchestrator.sol58_pes.ncu_summary import collect_ncu_analysis, should_profile


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


TARGET_LATENCY_MS = float(os.environ.get("SOL58_TARGET_LATENCY_MS", "0.006797"))
PROBLEM_DIR = Path(
    os.environ.get(
        "SOL58_PROBLEM_DIR",
        "/home/youchunbo/code/sol-problems/058_moe_expert_token_radix_sort_with_prefix_sum",
    )
)
EVAL_ROOT = Path(os.environ.get("SOL58_EVAL_ROOT", "/tmp/sol58_pes_eval"))
SOL_EXECBENCH = os.environ.get("SOL_EXECBENCH", "sol-execbench")
COMPILE_TIMEOUT = int(os.environ.get("SOL58_COMPILE_TIMEOUT", "180"))
RUN_TIMEOUT = int(os.environ.get("SOL58_SOL_TIMEOUT", "120"))
LOCAL_REPEAT_COUNT = max(1, int(os.environ.get("SOL58_LOCAL_REPEAT_COUNT", "3")))
LOCAL_BEST_GATE = _env_bool("SOL58_LOCAL_BEST_GATE", True)
MEASUREMENT_PROFILE_NAME = os.environ.get(
    "SOL58_MEASUREMENT_PROFILE", "native"
).strip().lower()
CODE_LANGUAGE = os.environ.get("SOL58_CODE_LANGUAGE", "cuda_cpp").strip().lower()
CUTEDSL_GENERATION_RATE = float(
    os.environ.get("SOL58_CUTEDSL_GENERATION_RATE", "0.5")
)
CUTEDSL_SCHEDULE_PERIOD = max(
    1,
    int(os.environ.get("SOL58_CUTEDSL_SCHEDULE_PERIOD", "10")),
)
NCU_SUMMARY_ENABLED = _env_bool("SOL58_NCU_SUMMARY", False)
NCU_PROFILE_POLICY = os.environ.get(
    "SOL58_NCU_PROFILE_POLICY", "all_correct"
).strip().lower()
OFFICIAL_FITNESS = _env_bool("SOL58_OFFICIAL_FITNESS", False)
OFFICIAL_BASE_URL = os.environ.get(
    "SOL58_OFFICIAL_BASE_URL",
    "https://research.nvidia.com/benchmarks/sol-execbench",
).rstrip("/")
OFFICIAL_KERNEL_ID = int(os.environ.get("SOL58_OFFICIAL_KERNEL_ID", "58"))
OFFICIAL_GPU_TYPE = os.environ.get("SOL58_OFFICIAL_GPU_TYPE", "B200")
OFFICIAL_EVAL_STACK_VERSION = os.environ.get("SOL58_OFFICIAL_EVAL_STACK_VERSION", "v1.1")
OFFICIAL_SUBMISSION_MODE = os.environ.get("SOL58_OFFICIAL_SUBMISSION_MODE", "private")
OFFICIAL_POLL_INTERVAL = float(os.environ.get("SOL58_OFFICIAL_POLL_INTERVAL", "10"))
OFFICIAL_POLL_TIMEOUT = float(os.environ.get("SOL58_OFFICIAL_POLL_TIMEOUT", "180"))
OFFICIAL_REQUEST_TIMEOUT = float(os.environ.get("SOL58_OFFICIAL_REQUEST_TIMEOUT", "10"))
OFFICIAL_CACHE = _env_bool("SOL58_OFFICIAL_CACHE", True)
OFFICIAL_ASYNC_SUBMIT = _env_bool("SOL58_OFFICIAL_ASYNC_SUBMIT", True)
OFFICIAL_ASYNC_REFRESH_DELAY = float(
    os.environ.get("SOL58_OFFICIAL_ASYNC_REFRESH_DELAY", "60")
)
OFFICIAL_MIN_LOCAL_SCORE = float(os.environ.get("SOL58_OFFICIAL_MIN_LOCAL_SCORE", "0"))
OFFICIAL_PENDING_RESULT_GRACE = float(os.environ.get("SOL58_OFFICIAL_PENDING_RESULT_GRACE", "60"))
OFFICIAL_CACHE_REFRESH_TIMEOUT = float(os.environ.get("SOL58_OFFICIAL_CACHE_REFRESH_TIMEOUT", "0"))
OFFICIAL_PENDING_SCORE_POLICY = os.environ.get(
    "SOL58_OFFICIAL_PENDING_SCORE_POLICY",
    "local_proxy",
).strip().lower()
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

SOURCE_LANGUAGE_CUDA = "cuda_cpp"
SOURCE_LANGUAGE_CUTE = "cute_dsl"
SOURCE_LANGUAGE_AUTO = "auto"
SUPPORTED_CODE_LANGUAGES = {
    SOURCE_LANGUAGE_CUDA,
    SOURCE_LANGUAGE_CUTE,
    SOURCE_LANGUAGE_AUTO,
}


def _tail(text: str, limit: int = 4000) -> str:
    if not text:
        return ""
    return text[-limit:]


def _geomean(xs: list[float]) -> float:
    xs = [x for x in xs if x and x > 0]
    if not xs:
        return 0.0
    return math.exp(sum(math.log(x) for x in xs) / len(xs))


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
        "clock_tolerance_mhz": int(
            os.environ.get("SOL58_CLOCK_TOLERANCE_MHZ", "10")
        ),
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
            drift_events.append(
                {"detected_at": time.time(), "monitor_error": str(exc)}
            )
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
    language = (value or CODE_LANGUAGE or SOURCE_LANGUAGE_CUDA).strip().lower()
    aliases = {
        "cuda": SOURCE_LANGUAGE_CUDA,
        "cu": SOURCE_LANGUAGE_CUDA,
        "cutedsl": SOURCE_LANGUAGE_CUTE,
        "cute": SOURCE_LANGUAGE_CUTE,
    }
    language = aliases.get(language, language)
    if language not in SUPPORTED_CODE_LANGUAGES:
        raise ValueError(
            f"unsupported SOL58 code language {language!r}; expected cuda_cpp, cute_dsl, or auto"
        )
    return language


def _program_iteration(program_path: str) -> int | None:
    normalized = str(program_path).replace("\\", "/")
    matches = re.findall(r"(?:^|/)(\d+)/executor(?:/|$)", normalized)
    return int(matches[-1]) if matches else None


def _required_source_language(
    program_path: str,
    code_language: str | None = None,
    cutedsl_rate: float | None = None,
    schedule_period: int | None = None,
) -> str | None:
    mode = _normalize_code_language(code_language)
    if mode != SOURCE_LANGUAGE_AUTO:
        return mode

    rate = CUTEDSL_GENERATION_RATE if cutedsl_rate is None else float(cutedsl_rate)
    period = CUTEDSL_SCHEDULE_PERIOD if schedule_period is None else int(schedule_period)
    if not 0.0 <= rate <= 1.0:
        raise ValueError("SOL58_CUTEDSL_GENERATION_RATE must be between 0 and 1")
    if period <= 0:
        raise ValueError("SOL58_CUTEDSL_SCHEDULE_PERIOD must be positive")

    iteration = _program_iteration(program_path)
    if iteration is None or rate <= 0.0:
        return None
    required_slots = min(period, math.ceil(rate * period))
    slot = (iteration - 1) % period
    return SOURCE_LANGUAGE_CUTE if slot < required_slots else None


def _language_from_source_path(path: str) -> str | None:
    suffix = Path(path).suffix.lower()
    if suffix in {".cu", ".cpp", ".cc", ".cxx"}:
        return SOURCE_LANGUAGE_CUDA
    if suffix == ".py":
        return SOURCE_LANGUAGE_CUTE
    return None


def _detect_source_language(kernel_source: str) -> str | None:
    if "PYBIND11_MODULE" in kernel_source and "#include" in kernel_source:
        return SOURCE_LANGUAGE_CUDA
    cute_import = re.search(
        r"(?m)^\s*(?:import\s+cutlass\.cute(?:\s+as\s+\w+)?|"
        r"from\s+cutlass(?:\.cute)?\s+import\s+)",
        kernel_source,
    )
    if cute_import and re.search(r"(?m)^\s*(?:async\s+)?def\s+run\s*\(", kernel_source):
        return SOURCE_LANGUAGE_CUTE
    return None


def _extract_kernel_source(raw: str, code_language: str | None = None) -> str:
    text = raw.strip()
    mode = _normalize_code_language(code_language)

    if text.startswith("{"):
        try:
            data = json.loads(text)
            candidates: list[tuple[str, str, str | None]] = []
            for src in data.get("sources", []):
                path = str(src.get("path", ""))
                content = src.get("content")
                if content:
                    candidates.append(
                        (path, str(content).strip(), _language_from_source_path(path))
                    )
            if mode != SOURCE_LANGUAGE_AUTO:
                for _, content, path_language in candidates:
                    if path_language == mode or _detect_source_language(content) == mode:
                        return content
            else:
                for _, content, _ in candidates:
                    if _detect_source_language(content) is not None:
                        return content
                for _, content, path_language in candidates:
                    if path_language is not None:
                        return content
        except Exception:
            pass

    if text.startswith("```") and text.endswith("```"):
        lines = text.splitlines()
        if len(lines) >= 3:
            text = "\n".join(lines[1:-1]).strip()

    lines = text.splitlines()
    if lines and lines[0].strip().lower() in {
        "cuda",
        "cu",
        "cpp",
        "c++",
        "cuda_cpp",
        "kernel.cu",
        "python",
        "py",
        "cute",
        "cutedsl",
        "cute_dsl",
        "kernel.py",
    }:
        text = "\n".join(lines[1:]).strip()

    return text


_INCLUDE_DIRECTIVE_RE = re.compile(r"^\s*#\s*include\b(.*)$")
_SOURCE_INCLUDE_SUFFIXES = {".c", ".cc", ".cpp", ".cxx", ".cu"}


def _source_dependency_violation(kernel_source: str) -> str | None:
    """Return why a purported single-file kernel depends on another source file."""
    for line_number, line in enumerate(kernel_source.splitlines(), start=1):
        match = _INCLUDE_DIRECTIVE_RE.match(line)
        if not match:
            continue

        operand = match.group(1).strip()
        if operand.startswith('"'):
            return (
                f"line {line_number}: quoted include {operand!r} is not allowed; "
                "the submitted kernel must be self-contained"
            )
        if not operand.startswith("<") or ">" not in operand:
            return f"line {line_number}: dynamic include {operand!r} is not allowed"

        include_path = operand[1 : operand.index(">")].strip()
        normalized = include_path.replace("\\", "/")
        path_parts = normalized.split("/")
        suffix = Path(normalized).suffix.lower()
        if (
            not normalized
            or normalized.startswith("/")
            or ".." in path_parts
            or re.match(r"^[A-Za-z]:/", normalized)
        ):
            return f"line {line_number}: non-portable include path {include_path!r}"
        if suffix in _SOURCE_INCLUDE_SUFFIXES:
            return (
                f"line {line_number}: including source file {include_path!r} is not allowed; "
                "emit one complete kernel.cu"
            )

    return None


_ALLOWED_CUTE_IMPORT_ROOTS = {"torch", "cutlass", "cuda"}
_STDLIB_IMPORT_ROOTS = set(getattr(sys, "stdlib_module_names", ())) | {"__future__"}


def _python_dependency_violation(kernel_source: str) -> str | None:
    """Reject syntax errors, relative imports, and unavailable local Python modules."""
    try:
        tree = ast.parse(kernel_source)
    except SyntaxError as exc:
        return f"Python syntax error at line {exc.lineno}: {exc.msg}"

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                return f"line {node.lineno}: relative imports are not allowed"
            modules = [node.module or ""]
        else:
            continue

        for module in modules:
            root = module.split(".", 1)[0]
            if root not in _ALLOWED_CUTE_IMPORT_ROOTS and root not in _STDLIB_IMPORT_ROOTS:
                return (
                    f"line {node.lineno}: import {module!r} is not an allowed standalone "
                    "CuTe DSL runtime dependency"
                )
    return None


def _cute_dsl_validation_error(kernel_source: str) -> str | None:
    dependency_violation = _python_dependency_violation(kernel_source)
    if dependency_violation:
        return dependency_violation

    if not re.search(
        r"(?m)^\s*(?:import\s+cutlass\.cute\s+as\s+cute|"
        r"from\s+cutlass\s+import\s+cute)",
        kernel_source,
    ):
        return "missing `import cutlass.cute as cute` (or equivalent `from cutlass import cute`)"
    if not re.search(r"@cute\.(?:kernel|jit)\b", kernel_source):
        return "missing a CuTe DSL `@cute.kernel` or `@cute.jit` definition"
    if "cute.compile" not in kernel_source:
        return "missing `cute.compile`"

    tree = ast.parse(kernel_source)
    run_function = next(
        (
            node
            for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "run"
        ),
        None,
    )
    if run_function is None:
        return "missing destination-passing `run` function"
    positional = [*run_function.args.posonlyargs, *run_function.args.args]
    argument_names = [argument.arg for argument in positional]
    required = ["topk_idx", "sorted_token_indices", "expert_offsets"]
    if (
        argument_names != required
        or run_function.args.vararg is not None
        or run_function.args.kwonlyargs
        or run_function.args.defaults
    ):
        return (
            "run signature must be exactly `run(topk_idx, sorted_token_indices, "
            "expert_offsets)` with only optional **kwargs beyond those arguments"
        )
    return None


def _candidate_validation_error(
    kernel_source: str,
    code_language: str | None = None,
) -> tuple[str | None, str | None]:
    mode = _normalize_code_language(code_language)
    detected = _detect_source_language(kernel_source)
    if detected is None:
        return (
            None,
            "candidate is neither a complete CUDA C++ source with PYBIND11_MODULE nor a "
            "complete CuTe DSL Python source with run()",
        )
    if mode != SOURCE_LANGUAGE_AUTO and detected != mode:
        return detected, f"candidate language {detected!r} is not allowed in {mode!r} mode"

    if detected == SOURCE_LANGUAGE_CUDA:
        if "#include" not in kernel_source or "PYBIND11_MODULE" not in kernel_source:
            return detected, "CUDA C++ source must contain includes and PYBIND11_MODULE"
        dependency_violation = _source_dependency_violation(kernel_source)
        if dependency_violation:
            return detected, dependency_violation
        return detected, None

    cute_error = _cute_dsl_validation_error(kernel_source)
    return detected, cute_error


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
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip().splitlines()[0].strip()
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
        cuda_cflags = ["-O3", "--use_fast_math", "-std=c++17"] + _detect_cuda_gencode_flags()
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
            f"{sol_execbench_path.resolve().parent}:"
            f"{child_env.get('PATH', '')}"
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

    traces = [json.loads(line) for line in traces_path.read_text().splitlines() if line.strip()]
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

        if status == "PASSED" and isinstance(latency_ms, (int, float)) and latency_ms > 0:
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
        "latency_ms_arith_mean": (sum(latencies_ms) / len(latencies_ms)) if latencies_ms else 0.0,
        "max_abs_err": max_abs,
        "max_rel_err": max_rel,
    }


def _local_best_path() -> Path:
    return EVAL_ROOT / "official_cache" / "local_best.json"


def _local_best_kernel_path(source_language: str = SOURCE_LANGUAGE_CUDA) -> Path:
    suffix = ".py" if source_language == SOURCE_LANGUAGE_CUTE else ".cu"
    return EVAL_ROOT / "official_cache" / f"local_best_kernel{suffix}"


def _kernel_source_hash(kernel_source: str) -> str:
    return hashlib.sha256(kernel_source.strip().encode("utf-8")).hexdigest()


def _authoritative_fitness_registry_path() -> Path:
    return EVAL_ROOT / "official_cache" / "authoritative_fitness.json"


def _record_authoritative_source_fitness(
    kernel_source: str,
    *,
    source_language: str,
    official_score: float,
    official_latency_ms: float,
    local_latency_ms: float,
    submission_id: Any,
) -> None:
    path = _authoritative_fitness_registry_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        registry = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    except Exception:
        registry = {}
    sources = registry.setdefault("sources", {})
    source_hash = _kernel_source_hash(kernel_source)
    sources[source_hash] = {
        "source_sha256": source_hash,
        "source_language": source_language,
        "official_score": float(official_score),
        "official_latency_ms": float(official_latency_ms),
        "local_latency_ms": float(local_latency_ms),
        "measurement_profile_id": _measurement_profile()["id"],
        "submission_id": submission_id,
        "evaluation_stack_version": OFFICIAL_EVAL_STACK_VERSION,
        "gpu_type": OFFICIAL_GPU_TYPE,
        "status": "COMPLETED",
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    registry["version"] = 1
    temp_path = path.with_suffix(".tmp")
    temp_path.write_text(
        json.dumps(registry, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temp_path.replace(path)


def _save_local_best(best: dict[str, Any], kernel_source: str | None = None) -> None:
    path = _local_best_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(best)
    profile = _measurement_profile()
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
    if source_language not in {SOURCE_LANGUAGE_CUDA, SOURCE_LANGUAGE_CUTE}:
        source_language = _language_from_source_path(recorded_kernel_path.name) or ""
    if source_language not in {SOURCE_LANGUAGE_CUDA, SOURCE_LANGUAGE_CUTE} and kernel_source:
        source_language = _detect_source_language(kernel_source) or SOURCE_LANGUAGE_CUDA
    if source_language not in {SOURCE_LANGUAGE_CUDA, SOURCE_LANGUAGE_CUTE}:
        source_language = SOURCE_LANGUAGE_CUDA
    payload["source_language"] = source_language

    expected_hash = str(payload.get("kernel_sha256") or "")
    if kernel_source and (not expected_hash or _kernel_source_hash(kernel_source) == expected_hash):
        kernel_path = _local_best_kernel_path(source_language)
        kernel_temp_path = kernel_path.with_suffix(".tmp")
        kernel_temp_path.write_text(kernel_source, encoding="utf-8")
        kernel_temp_path.replace(kernel_path)
        payload["kernel_path"] = str(kernel_path)

    payload["updated_at"] = datetime.now(timezone.utc).isoformat()
    temp_path = path.with_suffix(".tmp")
    temp_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temp_path.replace(path)


def _discover_local_best() -> dict[str, Any] | None:
    samples: dict[str, dict[str, Any]] = {}
    current_profile = _measurement_profile()
    for workspace in EVAL_ROOT.glob("eval_*"):
        profile_path = workspace / "measurement_profile.json"
        if profile_path.is_file():
            try:
                workspace_profile = json.loads(profile_path.read_text(encoding="utf-8"))
            except Exception:
                continue
            if str(workspace_profile.get("id") or "") != current_profile["id"]:
                continue
        elif current_profile["name"] != "native":
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
                _language_from_source_path(kernel_path.name)
                or _detect_source_language(kernel_source)
                or SOURCE_LANGUAGE_CUDA
            )
            dependency_violation = (
                _source_dependency_violation(kernel_source)
                if source_language == SOURCE_LANGUAGE_CUDA
                else _python_dependency_violation(kernel_source)
            )
            if dependency_violation:
                continue
            kernel_hash = _kernel_source_hash(kernel_source)
        except Exception:
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
                parsed = _parse_traces(workspace, traces_path.name)
                total = int(parsed.get("total") or 0)
                passed = int(parsed.get("passed") or 0)
                latency_ms = float(parsed.get("latency_ms_geomean") or 0.0)
                if total <= 0 or passed != total or latency_ms <= 0:
                    continue
                entry["latencies_ms"].append(latency_ms)
                entry["workspaces"].append(str(workspace))
            except Exception:
                continue

    candidates: list[dict[str, Any]] = []
    for entry in samples.values():
        latencies = entry["latencies_ms"]
        if len(latencies) < LOCAL_REPEAT_COUNT:
            continue
        median_latency = float(statistics.median(latencies))
        candidates.append(
            {
                "kernel_sha256": entry["kernel_sha256"],
                "source_language": entry["source_language"],
                "latency_ms_median": median_latency,
                "local_score": TARGET_LATENCY_MS / median_latency,
                "sample_count": len(latencies),
                "repeat_count_required": LOCAL_REPEAT_COUNT,
                "workspace": entry["workspaces"][-1],
                "source": "historical_bootstrap",
                "measurement_profile_id": current_profile["id"],
                "measurement_profile": current_profile,
            }
        )
    if not candidates:
        return None
    return min(candidates, key=lambda row: float(row["latency_ms_median"]))


def _load_local_best() -> dict[str, Any] | None:
    path = _local_best_path()
    if path.exists():
        try:
            best = json.loads(path.read_text(encoding="utf-8"))
            source_paths: list[Path] = []
            if best.get("kernel_path"):
                source_paths.append(Path(str(best["kernel_path"])))
            if best.get("workspace"):
                workspace = Path(str(best["workspace"]))
                source_paths.extend((workspace / "kernel.cu", workspace / "kernel.py"))
            for source_path in source_paths:
                if not source_path.is_file():
                    continue
                kernel_source = source_path.read_text(encoding="utf-8")
                source_language = (
                    str(best.get("source_language") or "").strip().lower()
                    or _language_from_source_path(source_path.name)
                    or _detect_source_language(kernel_source)
                    or SOURCE_LANGUAGE_CUDA
                )
                dependency_violation = (
                    _source_dependency_violation(kernel_source)
                    if source_language == SOURCE_LANGUAGE_CUDA
                    else _python_dependency_violation(kernel_source)
                )
                if (
                    float(best.get("latency_ms_median") or 0.0) > 0
                    and not dependency_violation
                    and (
                        not best.get("kernel_sha256")
                        or _kernel_source_hash(kernel_source) == best["kernel_sha256"]
                    )
                ):
                    best["source_language"] = source_language
                    best.setdefault("kernel_path", str(source_path))
                    return best
        except Exception:
            pass

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
    return {
        "status": status,
        "summary": summary,
        "score": float(score),
        "metrics": metrics or {},
        "artifacts": artifacts or {},
    }


def _official_token() -> str:
    return os.environ.get("SOL58_SOLBENCH_TOKEN") or os.environ.get("SOLBENCH_TOKEN") or ""


def _official_compile_options(solution: dict[str, Any]) -> dict[str, Any]:
    spec = dict(solution.get("spec") or {})
    compile_options = dict(spec.get("compile_options") or {})
    cuda_cflags = list(compile_options.get("cuda_cflags") or [])
    compile_options["cuda_cflags"] = [
        flag
        for flag in cuda_cflags
        if "-gencode" not in str(flag) and not str(flag).startswith("-arch")
    ]
    if "ld_flags" not in compile_options:
        compile_options["ld_flags"] = ["-lcuda"]
    return compile_options


def _build_official_submission(workspace: Path) -> dict[str, Any]:
    """Build a self-contained submission.json for the official B200 evaluator."""
    solution = json.loads((workspace / "solution.json").read_text(encoding="utf-8"))
    solution["name"] = f"sol58_pes_official_{uuid.uuid4().hex[:8]}"
    solution["author"] = os.environ.get("SOL58_OFFICIAL_AUTHOR", "atrex-loongflow-pes")
    languages = set((solution.get("spec") or {}).get("languages") or [])
    language_label = "CuTe DSL" if SOURCE_LANGUAGE_CUTE in languages else "CUDA C++"
    solution["description"] = (
        f"LoongFlow PES-generated {language_label} candidate for SOL kernel 58; "
        "official v1.1 fitness calibration."
    )

    spec = dict(solution.get("spec") or {})
    spec["target_hardware"] = [OFFICIAL_GPU_TYPE]
    if SOURCE_LANGUAGE_CUTE in languages:
        spec.pop("compile_options", None)
        spec.pop("binding", None)
    else:
        spec["compile_options"] = _official_compile_options(solution)
    solution["spec"] = spec

    for src in solution.get("sources", []):
        path = workspace / str(src.get("path", ""))
        if not path.is_file():
            raise FileNotFoundError(f"official submission source not found: {path}")
        src["content"] = path.read_text(encoding="utf-8")
    if not solution.get("sources"):
        raise ValueError("official submission has no sources")
    return solution


def _multipart_form(
    fields: dict[str, str],
    file_field: str,
    filename: str,
    content: bytes,
    content_type: str,
) -> tuple[bytes, str]:
    boundary = f"----atrex-sol58-{uuid.uuid4().hex}"
    chunks: list[bytes] = []
    for name, value in fields.items():
        chunks.extend(
            [
                f"--{boundary}\r\n".encode(),
                f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode(),
                str(value).encode(),
                b"\r\n",
            ]
        )
    chunks.extend(
        [
            f"--{boundary}\r\n".encode(),
            (
                f'Content-Disposition: form-data; name="{file_field}"; '
                f'filename="{filename}"\r\n'
            ).encode(),
            f"Content-Type: {content_type}\r\n\r\n".encode(),
            content,
            b"\r\n",
            f"--{boundary}--\r\n".encode(),
        ]
    )
    return b"".join(chunks), f"multipart/form-data; boundary={boundary}"


def _http_json(
    method: str,
    path: str,
    token: str,
    body: bytes | None = None,
    headers: dict[str, str] | None = None,
) -> dict[str, Any]:
    url = f"{OFFICIAL_BASE_URL}{path}"
    req = urllib.request.Request(url, data=body, method=method)
    req.add_header("Accept", "application/json")
    req.add_header("Authorization", f"Bearer {token}")
    for key, value in (headers or {}).items():
        req.add_header(key, value)
    try:
        with urllib.request.urlopen(req, timeout=OFFICIAL_REQUEST_TIMEOUT) as resp:
            text = resp.read().decode("utf-8", "replace")
            return json.loads(text) if text.strip() else {}
    except urllib.error.HTTPError as exc:
        text = exc.read().decode("utf-8", "replace")
        try:
            payload = json.loads(text)
        except Exception:
            payload = {"detail": _tail(text, 1000)}
        detail = payload.get("detail") if isinstance(payload, dict) else payload
        raise RuntimeError(f"official API {method} {path} failed HTTP {exc.code}: {detail}") from exc


def _parse_timestamp(value: Any) -> float | None:
    if not value:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        parsed = datetime.fromisoformat(text)
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.timestamp()
    except Exception:
        return None


def _format_timestamp(value: float) -> str:
    return datetime.fromtimestamp(value, timezone.utc).isoformat()


def _has_official_score(data: dict[str, Any]) -> bool:
    return data.get("sol_score") is not None and data.get("latency_ms") is not None


def _official_status(data: dict[str, Any]) -> str:
    return str(data.get("status") or "").upper()


def _is_stale_pending_result(data: dict[str, Any], now: float | None = None) -> bool:
    if _official_status(data) != "PENDING_RESULT" or _has_official_score(data):
        return False
    ts = _parse_timestamp(data.get("result_available_at"))
    if ts is None:
        ts = _parse_timestamp(data.get("finished_at"))
    if ts is None:
        return False
    return (now or time.time()) - ts >= OFFICIAL_PENDING_RESULT_GRACE


def _is_deferred_pending_result(data: dict[str, Any], deadline: float, now: float | None = None) -> bool:
    if _official_status(data) != "PENDING_RESULT" or _has_official_score(data):
        return False
    ts = _parse_timestamp(data.get("result_available_at"))
    if ts is None:
        return False
    current = now or time.time()
    return ts > deadline and ts - current > OFFICIAL_POLL_INTERVAL


def _normalize_submission_data(payload: dict[str, Any]) -> dict[str, Any]:
    data = payload.get("data") if isinstance(payload.get("data"), dict) else payload
    return dict(data or {})


def _get_official_submission(submission_id: Any, token: str) -> dict[str, Any]:
    primary = _normalize_submission_data(_http_json("GET", f"/api/submissions/{submission_id}", token))

    # A future availability timestamp is an explicit server-side delay. The list
    # endpoint cannot make that result available sooner, so avoid another slow
    # request on the common v1.1 deferred-result path.
    available_at = _parse_timestamp(primary.get("result_available_at"))
    primary_status = _official_status(primary)
    if _has_official_score(primary) or primary_status in {
        "PENDING",
        "QUEUED",
        "RUNNING",
        "EVALUATING",
    } or (
        primary_status == "PENDING_RESULT"
        and available_at is not None
        and available_at - time.time() > OFFICIAL_POLL_INTERVAL
    ):
        return primary

    # The web app also polls the authenticated submissions list. In practice this
    # can contain fresher score fields than the detail endpoint while the detail
    # endpoint is still stuck in PENDING_RESULT.
    try:
        params = (
            f"kernel_id={OFFICIAL_KERNEL_ID}&gpu_type={OFFICIAL_GPU_TYPE}"
            f"&offset=0&limit=50"
        )
        listed = _http_json("GET", f"/api/submissions?{params}", token)
        rows = (listed.get("data") or {}).get("submissions") or []
        for row in rows:
            if str(row.get("id")) == str(submission_id):
                merged = dict(primary)
                merged.update({k: v for k, v in row.items() if v is not None})
                return merged
    except Exception as exc:
        primary["list_refresh_error"] = str(exc)

    return primary


def _write_official_result(workspace: Path, result: dict[str, Any]) -> None:
    (workspace / "official_submission_result.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _poll_official_submission(
    submission_id: Any,
    token: str,
    workspace: Path,
    upload: dict[str, Any] | None = None,
    poll_timeout: float | None = None,
) -> dict[str, Any]:
    timeout = OFFICIAL_POLL_TIMEOUT if poll_timeout is None else poll_timeout
    deadline = time.time() + max(0.0, timeout)
    last: dict[str, Any] = {
        "id": submission_id,
        "status": "PENDING",
        "upload": upload or {},
        "cache_hit": False,
        "evaluation_stack_version": OFFICIAL_EVAL_STACK_VERSION,
    }

    while True:
        now = time.time()
        try:
            data = _get_official_submission(submission_id, token)
            last = dict(data)
            last.setdefault("id", submission_id)
            last["upload"] = upload or last.get("upload") or {}
            last["cache_hit"] = False
            last["evaluation_stack_version"] = data.get(
                "evaluation_stack_version",
                OFFICIAL_EVAL_STACK_VERSION,
            )
        except Exception as exc:
            last["poll_error"] = str(exc)

        if _is_deferred_pending_result(last, deadline=deadline, now=now):
            last["upstream_status"] = _official_status(last)
            last["status"] = "DEFERRED_RESULT"
            last["next_refresh_at"] = last.get("result_available_at")
            last["error_log"] = (
                "official result is not available until "
                f"{last.get('result_available_at')}; deferring final score refresh"
            )
        elif _is_stale_pending_result(last, now=now):
            last["upstream_status"] = _official_status(last)
            last["status"] = "STALE_PENDING_RESULT"
            last["next_refresh_at"] = _format_timestamp(
                now + OFFICIAL_ASYNC_REFRESH_DELAY
            )
            last["error_log"] = (
                "official result stayed PENDING_RESULT after "
                f"result availability for >{OFFICIAL_PENDING_RESULT_GRACE:.0f}s"
            )

        _write_official_result(workspace, last)

        status = _official_status(last)
        if status in OFFICIAL_TERMINAL_STATUSES or _has_official_score(last):
            break

        remaining = deadline - time.time()
        if remaining <= 0:
            last["upstream_status"] = _official_status(last)
            last["status"] = "TIMEOUT"
            last["next_refresh_at"] = _format_timestamp(
                time.time() + OFFICIAL_ASYNC_REFRESH_DELAY
            )
            last["error_log"] = f"official polling timed out after {timeout:.0f}s"
            _write_official_result(workspace, last)
            break

        time.sleep(min(OFFICIAL_POLL_INTERVAL, remaining))

    return last


def _calibration_path() -> Path:
    return EVAL_ROOT / "official_cache" / "calibration.jsonl"


def _discover_official_calibration_ratios() -> list[float]:
    """Recover calibration from completed legacy eval workspaces."""
    by_submission: dict[str, float] = {}
    result_paths = sorted(
        EVAL_ROOT.glob("eval_*/official_submission_result.json"),
        key=lambda path: path.stat().st_mtime,
    )
    for result_path in result_paths:
        try:
            official = json.loads(result_path.read_text(encoding="utf-8"))
            if (
                _official_status(official) != "COMPLETED"
                or not official.get("is_correct")
                or float(official.get("sol_score") or 0.0) <= 0
            ):
                continue

            parsed = _parse_traces(result_path.parent)
            total = int(parsed.get("total") or 0)
            passed = int(parsed.get("passed") or 0)
            local_latency_ms = float(parsed.get("latency_ms_geomean") or 0.0)
            if total <= 0 or passed != total or local_latency_ms <= 0:
                continue

            local_score = TARGET_LATENCY_MS / local_latency_ms
            submission_id = str(official.get("id") or result_path.parent.name)
            by_submission[submission_id] = float(official["sol_score"]) / local_score
        except Exception:
            continue
    return list(by_submission.values())


def _load_official_calibration_ratio() -> float:
    override = os.environ.get("SOL58_OFFICIAL_PROVISIONAL_RATIO", "").strip()
    if override:
        try:
            return max(0.0, float(override))
        except ValueError:
            pass

    path = _calibration_path()
    ratios: list[float] = []
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines()[-100:]:
            try:
                row = json.loads(line)
            except Exception:
                continue
            local_score = float(row.get("local_score") or 0.0)
            official_score = float(row.get("official_score") or 0.0)
            if local_score > 0 and official_score > 0:
                ratios.append(official_score / local_score)
    if not ratios:
        ratios = _discover_official_calibration_ratios()
    if not ratios:
        return 1.0

    ratios.sort()
    mid = len(ratios) // 2
    if len(ratios) % 2:
        return ratios[mid]
    return (ratios[mid - 1] + ratios[mid]) / 2.0


def _record_official_calibration(
    local_score: float,
    official_score: float,
    local_latency_ms: float,
    official_latency_ms: float,
    submission_id: Any,
) -> None:
    if local_score <= 0 or official_score <= 0:
        return
    path = _calibration_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "submission_id": submission_id,
        "local_score": local_score,
        "official_score": official_score,
        "ratio": official_score / local_score,
        "local_latency_ms": local_latency_ms,
        "official_latency_ms": official_latency_ms,
        "eval_stack": OFFICIAL_EVAL_STACK_VERSION,
        "gpu_type": OFFICIAL_GPU_TYPE,
    }
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _provisional_official_score(local_score: float) -> tuple[float, float]:
    ratio = _load_official_calibration_ratio()
    score = max(0.0, local_score * ratio)
    return min(score, OFFICIAL_PROVISIONAL_SCORE_CAP), ratio


def _local_provisional_score(local_score: float) -> float:
    """Keep local candidates ordered without allowing them to certify the target."""
    return min(max(0.0, local_score), OFFICIAL_PROVISIONAL_SCORE_CAP)


def _official_fitness_anchor(local_best: dict[str, Any] | None) -> dict[str, Any] | None:
    """Return a completed official score and its corresponding local-latency anchor."""
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
        }

    latency_ratio = float(anchor["local_latency_ms"]) / candidate_latency_ms
    score = float(anchor["score"]) * latency_ratio
    score = min(max(0.0, score), OFFICIAL_PROVISIONAL_SCORE_CAP)
    return score, {
        "source": "incumbent_official_anchor",
        "anchor_score": float(anchor["score"]),
        "anchor_local_latency_ms": float(anchor["local_latency_ms"]),
        "anchor_submission_id": anchor.get("submission_id"),
        "anchor_record_source": anchor["source"],
        "candidate_to_anchor_latency_ratio": latency_ratio,
    }


def _cache_path(kernel_source: str) -> Path:
    cache_key = hashlib.sha256(
        json.dumps(
            {
                "kernel_sha256": hashlib.sha256(kernel_source.encode("utf-8")).hexdigest(),
                "kernel_id": OFFICIAL_KERNEL_ID,
                "gpu_type": OFFICIAL_GPU_TYPE,
                "eval_stack": OFFICIAL_EVAL_STACK_VERSION,
                "mode": OFFICIAL_SUBMISSION_MODE,
            },
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()
    return EVAL_ROOT / "official_cache" / f"{cache_key}.json"


def _submit_official(
    workspace: Path,
    kernel_source: str,
    *,
    allow_upload: bool = True,
) -> dict[str, Any]:
    token = _official_token()
    if not token:
        raise RuntimeError("SOL58_OFFICIAL_FITNESS=1 requires SOLBENCH_TOKEN or SOL58_SOLBENCH_TOKEN")

    cache_path = _cache_path(kernel_source)
    if OFFICIAL_CACHE and cache_path.exists():
        cached = json.loads(cache_path.read_text(encoding="utf-8"))
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
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(
                json.dumps(refreshed, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            cached = refreshed
            cached_status = _official_status(cached)

        if cached_status in OFFICIAL_TERMINAL_STATUSES or _has_official_score(cached):
            cached["cache_hit"] = True
            _write_official_result(workspace, cached)
            return cached

        if not allow_upload:
            cached["cache_hit"] = True
            _write_official_result(workspace, cached)
            return cached

    if not allow_upload:
        raise RuntimeError("No cached official submission exists for the incumbent kernel")

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

    submission_id = ((upload.get("data") or {}).get("submission_id"))
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

    if OFFICIAL_CACHE:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(last, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return last


def evaluate(program_path: str) -> dict[str, Any]:
    start = time.time()
    eval_id = uuid.uuid4().hex[:12]
    workspace = EVAL_ROOT / f"eval_{eval_id}"
    workspace.mkdir(parents=True, exist_ok=True)

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
        if local_best_before and not local_best_profile_matches and not same_as_local_best:
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

        _copy_problem_files(workspace)
        _write_solution(workspace, kernel_source, source_language)
        expected = _load_workload_count(workspace)
        parsed_runs: list[dict[str, Any]] = []
        processes: list[subprocess.CompletedProcess[str]] = []
        local_run_records: list[dict[str, Any]] = []
        local_latencies_ms: list[float] = []
        rejected_clock_attempts: list[dict[str, Any]] = []
        max_local_attempts = max(
            LOCAL_REPEAT_COUNT,
            int(
                os.environ.get(
                    "SOL58_MAX_LOCAL_ATTEMPTS", str(LOCAL_REPEAT_COUNT + 3)
                )
            ),
        )
        repeat_index = 0
        attempt_index = 0

        while repeat_index < LOCAL_REPEAT_COUNT:
            attempt_index += 1
            traces_filename = (
                "traces.jsonl"
                if repeat_index == 0
                else f"traces_repeat_{repeat_index + 1}.jsonl"
            )
            try:
                clocks_before = _ensure_measurement_clocks()
            except Exception as exc:
                return _result(
                    "execution_failed",
                    f"Measurement environment rejected before local repeat "
                    f"{repeat_index + 1}/{LOCAL_REPEAT_COUNT}: {exc}",
                    0.0,
                    metrics={
                        "eval_time_s": time.time() - start,
                        "measurement_profile": measurement_profile,
                        "local_repeat_count_completed": repeat_index,
                    },
                    artifacts={
                        "workspace": str(workspace),
                        "program_path": program_path,
                        "local_repeats": local_run_records,
                    },
                )
            proc, clock_drift_events = _run_sol_execbench_monitored(
                workspace, traces_filename
            )
            parsed = _parse_traces(workspace, traces_filename)
            try:
                clocks_after = _ensure_measurement_clocks(allow_relock=False)
            except Exception as exc:
                clocks_after = {"validation_error": str(exc)}
                clock_drift_events.append(
                    {
                        "detected_at": time.time(),
                        "post_repeat_validation_error": str(exc),
                    }
                )

            if clock_drift_events:
                rejected_path = workspace / (
                    f"traces_clock_rejected_attempt_{attempt_index}.jsonl.rejected"
                )
                traces_path = workspace / traces_filename
                if traces_path.is_file():
                    traces_path.replace(rejected_path)
                rejected_clock_attempts.append(
                    {
                        "attempt": attempt_index,
                        "intended_repeat": repeat_index + 1,
                        "rejected_traces_path": str(rejected_path),
                        "clock_drift_events": clock_drift_events,
                        "clocks_before": clocks_before,
                        "clocks_after": clocks_after,
                    }
                )
                if attempt_index >= max_local_attempts:
                    return _result(
                        "execution_failed",
                        "Official-like local measurement could not collect "
                        f"{LOCAL_REPEAT_COUNT} clean repeats in {max_local_attempts} attempts "
                        "because GPU clocks changed during evaluation.",
                        0.0,
                        metrics={
                            "eval_time_s": time.time() - start,
                            "measurement_profile": measurement_profile,
                            "local_repeat_count_completed": repeat_index,
                            "clock_rejected_attempt_count": len(
                                rejected_clock_attempts
                            ),
                        },
                        artifacts={
                            "workspace": str(workspace),
                            "program_path": program_path,
                            "local_repeats": local_run_records,
                            "rejected_clock_attempts": rejected_clock_attempts,
                        },
                    )
                try:
                    _set_measurement_clocks(measurement_profile, stabilize=True)
                except Exception as exc:
                    return _result(
                        "execution_failed",
                        f"Failed to restore clocks after rejected attempt: {exc}",
                        0.0,
                        metrics={
                            "eval_time_s": time.time() - start,
                            "measurement_profile": measurement_profile,
                        },
                        artifacts={
                            "workspace": str(workspace),
                            "rejected_clock_attempts": rejected_clock_attempts,
                        },
                    )
                continue

            processes.append(proc)
            parsed_runs.append(parsed)

            total = int(parsed.get("total", 0))
            passed = int(parsed.get("passed", 0))
            failures = parsed.get("failures", [])
            latency_ms = float(parsed.get("latency_ms_geomean", 0.0) or 0.0)
            local_run_records.append(
                {
                    "repeat": repeat_index + 1,
                    "attempt": attempt_index,
                    "traces_path": str(workspace / traces_filename),
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
            failure_artifacts = {
                "workspace": str(workspace),
                "source_language": source_language,
                "measurement_profile": measurement_profile,
                "returncode": proc.returncode,
                "stdout_tail": _tail(proc.stdout),
                "stderr_tail": _tail(proc.stderr),
                "per_workload": parsed.get("per_workload", []),
                "local_repeats": local_run_records,
                "rejected_clock_attempts": rejected_clock_attempts,
            }
            elapsed = time.time() - start

            if parsed.get("error"):
                return _result(
                    "execution_failed",
                    f"SOL-ExecBench repeat {repeat_index + 1}/{LOCAL_REPEAT_COUNT} failed "
                    f"before producing traces: {parsed['error']}. "
                    f"stderr_tail={_tail(proc.stderr, 1200)}",
                    0.0,
                    metrics={"eval_time_s": elapsed, "target_latency_ms": TARGET_LATENCY_MS},
                    artifacts=failure_artifacts,
                )

            if proc.returncode != 0 or total != expected or passed != expected or failures:
                summary = (
                    f"Correctness/coverage gate failed on local repeat "
                    f"{repeat_index + 1}/{LOCAL_REPEAT_COUNT}: passed {passed}/{expected}, "
                    f"returncode={proc.returncode}. failures={failures[:6]}"
                )
                return _result(
                    "validation_failed",
                    summary,
                    0.0,
                    metrics={
                        "eval_time_s": elapsed,
                        "target_latency_ms": TARGET_LATENCY_MS,
                        "local_repeat_count_completed": repeat_index + 1,
                    },
                    artifacts=failure_artifacts,
                )

            if latency_ms <= 0:
                return _result(
                    "execution_failed",
                    f"Local repeat {repeat_index + 1}/{LOCAL_REPEAT_COUNT} passed all "
                    "workloads but produced no positive latency.",
                    0.0,
                    metrics={"eval_time_s": elapsed, "target_latency_ms": TARGET_LATENCY_MS},
                    artifacts=failure_artifacts,
                )
            local_latencies_ms.append(latency_ms)
            repeat_index += 1

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
                workspace / ("kernel.py" if source_language == SOURCE_LANGUAGE_CUTE else "kernel.cu")
            ),
            "returncode": proc.returncode,
            "stdout_tail": _tail(proc.stdout),
            "stderr_tail": _tail(proc.stderr),
            "per_workload": parsed.get("per_workload", []),
            "local_repeats": local_run_records,
            "rejected_clock_attempts": rejected_clock_attempts,
        }

        metrics = {
            "eval_time_s": elapsed,
            "local_eval_time_s": elapsed,
            "source_language": source_language,
            "configured_code_language": code_language,
            "required_source_language": required_source_language,
            "measurement_profile": measurement_profile,
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
            "passed": passed,
            "total": total,
            "expected_total": expected,
            "max_abs_err": max(float(run.get("max_abs_err") or 0.0) for run in parsed_runs),
            "max_rel_err": max(float(run.get("max_rel_err") or 0.0) for run in parsed_runs),
        }

        local_score = TARGET_LATENCY_MS / latency_ms
        metrics["local_score"] = local_score
        best_latency_before = (
            float(local_best_before.get("latency_ms_median") or 0.0)
            if local_best_before and local_best_profile_matches
            else 0.0
        )
        beats_local_best = (
            not LOCAL_BEST_GATE
            or best_latency_before <= 0
            or (not same_as_local_best and latency_ms < best_latency_before)
        )
        metrics["local_best"] = {
            "gate_enabled": LOCAL_BEST_GATE,
            "candidate_kernel_sha256": kernel_sha256,
            "candidate_latency_ms_median": latency_ms,
            "previous_latency_ms_median": best_latency_before or None,
            "same_kernel": same_as_local_best,
            "strictly_improved": beats_local_best,
            "measurement_profile_matches": local_best_profile_matches,
            "profile_recalibration": profile_recalibration,
            "improvement_ms": (
                best_latency_before - latency_ms if best_latency_before > 0 else None
            ),
        }
        reuse_cached_official = bool(
            same_as_local_best
            and OFFICIAL_CACHE
            and _cache_path(kernel_source).is_file()
        )
        metrics["local_best"]["reuse_cached_official"] = reuse_cached_official
        new_local_best_record: dict[str, Any] | None = None
        if LOCAL_BEST_GATE and beats_local_best:
            new_local_best_record = {
                "kernel_sha256": kernel_sha256,
                "source_language": source_language,
                "latency_ms_median": latency_ms,
                "local_score": local_score,
                "repeat_count": LOCAL_REPEAT_COUNT,
                "repeat_latencies_ms": local_latencies_ms,
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
                        "official_anchor_latency_ms": latency_ms,
                        "official_anchor_submission_id": local_best_before.get(
                            "official_submission_id"
                        ),
                    }
                )
            elif fitness_anchor is not None:
                new_local_best_record.update(
                    {
                        "official_anchor_score": fitness_anchor["score"],
                        "official_anchor_latency_ms": fitness_anchor["local_latency_ms"],
                        "official_anchor_submission_id": fitness_anchor.get("submission_id"),
                    }
                )
            _save_local_best(new_local_best_record, kernel_source)
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
        local_summary = (
            f"Local {source_language} prefilter passed all {passed}/{expected} workloads in "
            f"{LOCAL_REPEAT_COUNT}/{LOCAL_REPEAT_COUNT} repeats; median geomean latency "
            f"{latency_ms:.6f} ms from {[round(value, 6) for value in local_latencies_ms]} "
            f"vs target {TARGET_LATENCY_MS:.6f} ms; "
            f"local_score={local_score:.6f} ({status_line}); measurement_profile="
            f"{measurement_profile['name']}:{measurement_profile['id']}."
        )

        if OFFICIAL_FITNESS:
            recorded_anchor = _official_fitness_anchor(local_best_before)
            if (
                same_as_local_best
                and recorded_anchor is not None
                and recorded_anchor["source"] == "completed_local_best"
            ):
                recorded_score = float(recorded_anchor["score"])
                _record_authoritative_source_fitness(
                    kernel_source,
                    source_language=source_language,
                    official_score=recorded_score,
                    official_latency_ms=float(
                        local_best_before.get("official_latency_ms") or 0.0
                    ),
                    local_latency_ms=latency_ms,
                    submission_id=recorded_anchor.get("submission_id"),
                )
                metrics["official"] = {
                    "enabled": True,
                    "submitted": False,
                    "status": "REUSED_LOCAL_BEST_OFFICIAL",
                    "authoritative": True,
                    "fitness_source": "official_local_best_record",
                    "submission_id": recorded_anchor.get("submission_id"),
                    "sol_score": recorded_score,
                    "record_hit": True,
                    "cache_hit": False,
                }
                summary = (
                    f"{local_summary} Reused completed official v1.1 fitness from the "
                    f"persisted local-best record: sol_score={recorded_score:.6f}, "
                    f"submission_id={recorded_anchor.get('submission_id')}."
                )
                return _result(
                    "success",
                    summary,
                    recorded_score,
                    metrics=metrics,
                    artifacts=common_artifacts,
                )

            if local_score < OFFICIAL_MIN_LOCAL_SCORE:
                metrics["official"] = {
                    "enabled": True,
                    "status": "SKIPPED_LOCAL_PREFILTER",
                    "authoritative": False,
                    "fitness_source": "none",
                }
                summary = (
                    f"{local_summary} Official v1.1 fitness skipped because local_score "
                    f"{local_score:.6f} < SOL58_OFFICIAL_MIN_LOCAL_SCORE "
                    f"{OFFICIAL_MIN_LOCAL_SCORE:.6f}; PES score forced to 0."
                )
                return _result(
                    "success",
                    summary,
                    0.0,
                    metrics=metrics,
                    artifacts=common_artifacts,
                )

            if LOCAL_BEST_GATE and not beats_local_best and not reuse_cached_official:
                scoring_latency_ms = (
                    best_latency_before
                    if same_as_local_best and best_latency_before > 0
                    else latency_ms
                )
                scoring_local_score = TARGET_LATENCY_MS / scoring_latency_ms
                provisional_score, provisional_details = _anchored_provisional_score(
                    scoring_local_score,
                    scoring_latency_ms,
                    local_best_before,
                )
                skip_status = (
                    "SKIPPED_SAME_AS_LOCAL_BEST"
                    if same_as_local_best
                    else "SKIPPED_NOT_LOCAL_BEST"
                )
                anchored = provisional_details["source"] == "incumbent_official_anchor"
                provisional_label = (
                    "incumbent-anchored" if anchored else "calibrated local"
                )
                metrics["official"] = {
                    "enabled": True,
                    "submitted": False,
                    "status": skip_status,
                    "authoritative": False,
                    "fitness_source": (
                        "provisional_official_anchor"
                        if anchored
                        else "provisional_local_proxy"
                    ),
                    "provisional": True,
                    "provisional_score": provisional_score,
                    "provisional_local_score": scoring_local_score,
                    "provisional_score_cap": OFFICIAL_PROVISIONAL_SCORE_CAP,
                    "provisional_calibration": provisional_details,
                }
                if same_as_local_best:
                    gate_detail = "candidate is the persisted local-best kernel"
                else:
                    gate_detail = (
                        f"median latency {latency_ms:.6f} ms did not strictly beat "
                        f"local best {best_latency_before:.6f} ms"
                    )
                summary = (
                    f"{local_summary} Official v1.1 submission skipped because {gate_detail}. "
                    f"Using {provisional_label} provisional score {provisional_score:.6f}; "
                    "no remote slot was consumed."
                )
                return _result(
                    "success",
                    summary,
                    provisional_score,
                    metrics=metrics,
                    artifacts=common_artifacts,
                )

            official_started = time.time()
            official = _submit_official(
                workspace,
                kernel_source,
                allow_upload=not same_as_local_best,
            )
            official_request_time = time.time() - official_started
            if new_local_best_record is not None:
                new_local_best_record.update(
                    {
                        "remote_submitted": True,
                        "official_submission_id": official.get("id"),
                        "official_status": _official_status(official),
                    }
                )
                _save_local_best(new_local_best_record, kernel_source)
            metrics["eval_time_s"] = time.time() - start
            official_status = _official_status(official)
            official_correct = bool(official.get("is_correct"))
            official_score = float(official.get("sol_score") or 0.0)
            official_latency_ms = float(official.get("latency_ms") or 0.0)
            official_stack = official.get("evaluation_stack_version") or OFFICIAL_EVAL_STACK_VERSION
            official_metrics = {
                "enabled": True,
                "submitted": True,
                "submission_id": official.get("id"),
                "status": official_status,
                "is_correct": official_correct,
                "sol_score": official_score,
                "latency_ms": official_latency_ms,
                "fast_1_count": official.get("fast_1_count"),
                "fast_1_total": official.get("fast_1_total"),
                "avg_speedup": official.get("avg_speedup"),
                "gpu_type": official.get("gpu_type") or OFFICIAL_GPU_TYPE,
                "evaluation_stack_version": official_stack,
                "submission_mode": OFFICIAL_SUBMISSION_MODE,
                "cache_hit": bool(official.get("cache_hit")),
                "upstream_status": official.get("upstream_status"),
                "next_refresh_at": official.get("next_refresh_at"),
                "request_time_s": official_request_time,
            }
            if official_latency_ms > 0:
                official_metrics["local_to_official_latency_ratio"] = latency_ms / official_latency_ms
            metrics["official"] = official_metrics
            common_artifacts["official_submission_id"] = official.get("id")
            common_artifacts["official_status"] = official_status
            common_artifacts["official_result_path"] = str(workspace / "official_submission_result.json")
            if (workspace / "official_submission.json").exists():
                common_artifacts["official_submission_path"] = str(workspace / "official_submission.json")

            authoritative_score = (
                official_status == "COMPLETED"
                and official_correct
                and official_score > 0
            )
            if authoritative_score:
                official_metrics.update(
                    {
                        "authoritative": True,
                        "fitness_source": "official",
                    }
                )
                _record_authoritative_source_fitness(
                    kernel_source,
                    source_language=source_language,
                    official_score=official_score,
                    official_latency_ms=official_latency_ms,
                    local_latency_ms=latency_ms,
                    submission_id=official.get("id"),
                )
                _record_official_calibration(
                    local_score=local_score,
                    official_score=official_score,
                    local_latency_ms=latency_ms,
                    official_latency_ms=official_latency_ms,
                    submission_id=official.get("id"),
                )
                persisted_best = new_local_best_record
                if persisted_best is None and same_as_local_best and local_best_before:
                    persisted_best = dict(local_best_before)
                if persisted_best is not None:
                    persisted_best.update(
                        {
                            "remote_submitted": True,
                            "official_submission_id": official.get("id"),
                            "official_status": official_status,
                            "official_score": official_score,
                            "official_latency_ms": official_latency_ms,
                            "official_anchor_score": official_score,
                            "official_anchor_latency_ms": latency_ms,
                            "official_anchor_submission_id": official.get("id"),
                        }
                    )
                    _save_local_best(persisted_best, kernel_source)
                summary = (
                    f"Official {official_stack} B200 fitness: submission_id={official.get('id')}, "
                    f"status={official_status}, is_correct={official_correct}, "
                    f"sol_score={official_score:.6f}, latency={official_latency_ms:.6f} ms. "
                    f"{local_summary}"
                )
                return _result(
                    "success",
                    summary,
                    official_score,
                    metrics=metrics,
                    artifacts=common_artifacts,
                )

            pending_like = official_status in OFFICIAL_REFRESHABLE_STATUSES or official_status in {
                "TIMEOUT",
                "STALE_PENDING_RESULT",
            }
            if pending_like and OFFICIAL_PENDING_SCORE_POLICY in {
                "provisional",
                "calibrated",
                "local",
                "local_proxy",
            }:
                scoring_latency_ms = (
                    best_latency_before
                    if same_as_local_best and best_latency_before > 0
                    else latency_ms
                )
                scoring_local_score = TARGET_LATENCY_MS / scoring_latency_ms
                provisional_score, provisional_details = _anchored_provisional_score(
                    scoring_local_score,
                    scoring_latency_ms,
                    local_best_before,
                )
                if provisional_details["source"] == "incumbent_official_anchor":
                    provisional_ratio = provisional_details[
                        "candidate_to_anchor_latency_ratio"
                    ]
                    provisional_label = "incumbent-anchored"
                elif OFFICIAL_PENDING_SCORE_POLICY in {"local", "local_proxy"}:
                    provisional_score = _local_provisional_score(scoring_local_score)
                    provisional_ratio = 1.0
                    provisional_label = "local-proxy fallback"
                else:
                    provisional_score, provisional_ratio = _provisional_official_score(
                        scoring_local_score
                    )
                    provisional_label = "calibrated"
                official_metrics.update(
                    {
                        "authoritative": False,
                        "fitness_source": "provisional",
                        "provisional": True,
                        "provisional_score": provisional_score,
                        "provisional_ratio": provisional_ratio,
                        "provisional_score_cap": OFFICIAL_PROVISIONAL_SCORE_CAP,
                        "pending_policy": OFFICIAL_PENDING_SCORE_POLICY,
                        "provisional_calibration": provisional_details,
                    }
                )
                if official.get("upstream_status") == "QUEUED":
                    pending_detail = "was accepted and queued asynchronously"
                else:
                    pending_detail = f"is {official_status} after bounded polling/cache refresh"
                summary = (
                    f"{local_summary} Official {official_stack} B200 submission "
                    f"{official.get('id')} {pending_detail}; "
                    f"using provisional {provisional_label} score {provisional_score:.6f} "
                    f"(ratio={provisional_ratio:.4f}, cap={OFFICIAL_PROVISIONAL_SCORE_CAP:.6f}) "
                    "so PES can continue. This provisional score is capped below target and "
                    "does not certify leaderboard rank."
                )
                return _result(
                    "success",
                    summary,
                    provisional_score,
                    metrics=metrics,
                    artifacts=common_artifacts,
                )

            if official_status != "COMPLETED" or not official_correct or official_score <= 0:
                summary = (
                    f"{local_summary} Official {official_stack} fitness failed: status={official_status}, "
                    f"is_correct={official_correct}, sol_score={official_score:.6f}; "
                    "PES score forced to 0."
                )
                return _result(
                    "validation_failed",
                    summary,
                    0.0,
                    metrics=metrics,
                    artifacts=common_artifacts,
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
            metrics={"eval_time_s": time.time() - start, "target_latency_ms": TARGET_LATENCY_MS},
            artifacts={"workspace": str(workspace), "program_path": program_path},
        )
    except Exception as exc:
        return _result(
            "framework_error",
            f"Evaluation failed: {exc}",
            0.0,
            metrics={"eval_time_s": time.time() - start, "target_latency_ms": TARGET_LATENCY_MS},
            artifacts={
                "workspace": str(workspace),
                "program_path": program_path,
                "traceback": traceback.format_exc(),
            },
        )
