#!/usr/bin/env python3
"""LoongFlow evaluator for SOL-ExecBench kernel 58.

LoongFlow passes a file containing the generated solution. This evaluator
materializes that content as kernel.cu in a temporary SOL-ExecBench workspace,
runs the official local sol-execbench CLI over all workloads, and returns the
standard LoongFlow {status, summary, score, metrics, artifacts} dictionary.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import shlex
import shutil
import statistics
import subprocess
import time
import traceback
import urllib.error
import urllib.request
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


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


def _tail(text: str, limit: int = 4000) -> str:
    if not text:
        return ""
    return text[-limit:]


def _geomean(xs: list[float]) -> float:
    xs = [x for x in xs if x and x > 0]
    if not xs:
        return 0.0
    return math.exp(sum(math.log(x) for x in xs) / len(xs))


def _extract_kernel_source(raw: str) -> str:
    text = raw.strip()

    if text.startswith("{"):
        try:
            data = json.loads(text)
            for src in data.get("sources", []):
                path = str(src.get("path", ""))
                content = src.get("content")
                if path.endswith(".cu") and content:
                    return str(content).strip()
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
    }:
        text = "\n".join(lines[1:]).strip()

    return text


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


def _write_solution(workspace: Path, kernel_source: str) -> None:
    (workspace / "kernel.cu").write_text(kernel_source, encoding="utf-8")
    cuda_cflags = ["-O3", "--use_fast_math", "-std=c++17"] + _detect_cuda_gencode_flags()
    solution = {
        "name": f"sol58_loongflow_candidate_{uuid.uuid4().hex[:8]}",
        "definition": "058_moe_expert_token_radix_sort_with_prefix_sum",
        "author": "atrex-loongflow-pes",
        "description": "LoongFlow PES-generated CUDA C++ candidate for SOL kernel 58.",
        "spec": {
            "languages": ["cuda_cpp"],
            "target_hardware": ["B200", "LOCAL"],
            "entry_point": "kernel.cu::run",
            "dependencies": [],
            "compile_options": {
                "cuda_cflags": cuda_cflags,
                "ld_flags": ["-lcuda"],
            },
            "destination_passing_style": True,
            "binding": "torch",
        },
        "sources": [{"path": "kernel.cu"}],
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
    return subprocess.run(
        cmd,
        cwd=str(workspace),
        capture_output=True,
        text=True,
        timeout=COMPILE_TIMEOUT + RUN_TIMEOUT + 60,
        env={**os.environ, "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"},
    )


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


def _local_best_kernel_path() -> Path:
    return EVAL_ROOT / "official_cache" / "local_best_kernel.cu"


def _kernel_source_hash(kernel_source: str) -> str:
    return hashlib.sha256(kernel_source.strip().encode("utf-8")).hexdigest()


def _save_local_best(best: dict[str, Any], kernel_source: str | None = None) -> None:
    path = _local_best_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(best)

    if kernel_source is None:
        workspace = Path(str(payload.get("workspace") or ""))
        workspace_kernel = workspace / "kernel.cu"
        if workspace_kernel.is_file():
            try:
                kernel_source = workspace_kernel.read_text(encoding="utf-8")
            except OSError:
                kernel_source = None

    expected_hash = str(payload.get("kernel_sha256") or "")
    if kernel_source and (not expected_hash or _kernel_source_hash(kernel_source) == expected_hash):
        kernel_path = _local_best_kernel_path()
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
    for workspace in EVAL_ROOT.glob("eval_*"):
        kernel_path = workspace / "kernel.cu"
        if not kernel_path.is_file():
            continue
        try:
            kernel_source = kernel_path.read_text(encoding="utf-8")
            kernel_hash = _kernel_source_hash(kernel_source)
        except Exception:
            continue

        entry = samples.setdefault(
            kernel_hash,
            {
                "kernel_sha256": kernel_hash,
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
                "latency_ms_median": median_latency,
                "local_score": TARGET_LATENCY_MS / median_latency,
                "sample_count": len(latencies),
                "repeat_count_required": LOCAL_REPEAT_COUNT,
                "workspace": entry["workspaces"][-1],
                "source": "historical_bootstrap",
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
            if float(best.get("latency_ms_median") or 0.0) > 0:
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
    solution["description"] = (
        "LoongFlow PES-generated CUDA C++ candidate for SOL kernel 58; "
        "official v1.1 fitness calibration."
    )

    spec = dict(solution.get("spec") or {})
    spec["target_hardware"] = [OFFICIAL_GPU_TYPE]
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
        kernel_source = _extract_kernel_source(raw)

        if not kernel_source or "#include" not in kernel_source or "PYBIND11_MODULE" not in kernel_source:
            return _result(
                "validation_failed",
                "Generated candidate is not a complete CUDA C++ kernel.cu source with PYBIND11_MODULE.",
                0.0,
                metrics={"eval_time_s": time.time() - start},
                artifacts={"workspace": str(workspace), "program_path": program_path},
            )

        _copy_problem_files(workspace)
        _write_solution(workspace, kernel_source)
        local_best_before = _load_local_best() if LOCAL_BEST_GATE else None
        expected = _load_workload_count(workspace)
        parsed_runs: list[dict[str, Any]] = []
        processes: list[subprocess.CompletedProcess[str]] = []
        local_run_records: list[dict[str, Any]] = []
        local_latencies_ms: list[float] = []

        for repeat_index in range(LOCAL_REPEAT_COUNT):
            traces_filename = (
                "traces.jsonl"
                if repeat_index == 0
                else f"traces_repeat_{repeat_index + 1}.jsonl"
            )
            proc = _run_sol_execbench(workspace, traces_filename)
            parsed = _parse_traces(workspace, traces_filename)
            processes.append(proc)
            parsed_runs.append(parsed)

            total = int(parsed.get("total", 0))
            passed = int(parsed.get("passed", 0))
            failures = parsed.get("failures", [])
            latency_ms = float(parsed.get("latency_ms_geomean", 0.0) or 0.0)
            local_run_records.append(
                {
                    "repeat": repeat_index + 1,
                    "traces_path": str(workspace / traces_filename),
                    "returncode": proc.returncode,
                    "passed": passed,
                    "total": total,
                    "latency_ms_geomean": latency_ms,
                    "latency_ms_arith_mean": parsed.get("latency_ms_arith_mean", 0.0),
                    "failures": failures[:6],
                }
            )
            failure_artifacts = {
                "workspace": str(workspace),
                "returncode": proc.returncode,
                "stdout_tail": _tail(proc.stdout),
                "stderr_tail": _tail(proc.stderr),
                "per_workload": parsed.get("per_workload", []),
                "local_repeats": local_run_records,
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
            "returncode": proc.returncode,
            "stdout_tail": _tail(proc.stdout),
            "stderr_tail": _tail(proc.stderr),
            "per_workload": parsed.get("per_workload", []),
            "local_repeats": local_run_records,
        }

        metrics = {
            "eval_time_s": elapsed,
            "local_eval_time_s": elapsed,
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
            "passed": passed,
            "total": total,
            "expected_total": expected,
            "max_abs_err": max(float(run.get("max_abs_err") or 0.0) for run in parsed_runs),
            "max_rel_err": max(float(run.get("max_rel_err") or 0.0) for run in parsed_runs),
        }

        local_score = TARGET_LATENCY_MS / latency_ms
        metrics["local_score"] = local_score
        kernel_sha256 = _kernel_source_hash(kernel_source)
        local_best_hash = (
            str(local_best_before.get("kernel_sha256") or "")
            if local_best_before
            else ""
        )
        same_as_local_best = bool(local_best_hash and kernel_sha256 == local_best_hash)
        best_latency_before = (
            float(local_best_before.get("latency_ms_median") or 0.0)
            if local_best_before
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
                "latency_ms_median": latency_ms,
                "local_score": local_score,
                "repeat_count": LOCAL_REPEAT_COUNT,
                "repeat_latencies_ms": local_latencies_ms,
                "workspace": str(workspace),
                "source": "pes_evaluation",
                "remote_submitted": False,
            }
            _save_local_best(new_local_best_record, kernel_source)
        status_line = "target met" if local_score >= 1.0 else "target not met"
        local_summary = (
            f"Local prefilter passed all {passed}/{expected} workloads in "
            f"{LOCAL_REPEAT_COUNT}/{LOCAL_REPEAT_COUNT} repeats; median geomean latency "
            f"{latency_ms:.6f} ms from {[round(value, 6) for value in local_latencies_ms]} "
            f"vs target {TARGET_LATENCY_MS:.6f} ms; "
            f"local_score={local_score:.6f} ({status_line})."
        )

        if OFFICIAL_FITNESS:
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
                scoring_local_score = local_score
                if same_as_local_best and best_latency_before > 0:
                    scoring_local_score = TARGET_LATENCY_MS / best_latency_before
                provisional_score = _local_provisional_score(scoring_local_score)
                skip_status = (
                    "SKIPPED_SAME_AS_LOCAL_BEST"
                    if same_as_local_best
                    else "SKIPPED_NOT_LOCAL_BEST"
                )
                metrics["official"] = {
                    "enabled": True,
                    "submitted": False,
                    "status": skip_status,
                    "authoritative": False,
                    "fitness_source": "provisional_local_proxy",
                    "provisional": True,
                    "provisional_score": provisional_score,
                    "provisional_local_score": scoring_local_score,
                    "provisional_score_cap": OFFICIAL_PROVISIONAL_SCORE_CAP,
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
                    f"Using monotonic local-proxy score {provisional_score:.6f}; "
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
                _record_official_calibration(
                    local_score=local_score,
                    official_score=official_score,
                    local_latency_ms=latency_ms,
                    official_latency_ms=official_latency_ms,
                    submission_id=official.get("id"),
                )
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
                if OFFICIAL_PENDING_SCORE_POLICY in {"local", "local_proxy"}:
                    provisional_score = _local_provisional_score(local_score)
                    provisional_ratio = 1.0
                    provisional_label = "local-proxy"
                else:
                    provisional_score, provisional_ratio = _provisional_official_score(local_score)
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
