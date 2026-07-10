#!/usr/bin/env python3
"""LoongFlow evaluator for SOL-ExecBench kernel 58.

LoongFlow passes a file containing the generated solution. This evaluator
materializes that content as kernel.cu in a temporary SOL-ExecBench workspace,
runs the official local sol-execbench CLI over all workloads, and returns the
standard LoongFlow {status, summary, score, metrics, artifacts} dictionary.
"""

from __future__ import annotations

import json
import math
import os
import shlex
import shutil
import subprocess
import time
import traceback
import uuid
from pathlib import Path
from typing import Any


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
RUN_TIMEOUT = int(os.environ.get("SOL58_SOL_TIMEOUT", "900"))


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


def _run_sol_execbench(workspace: Path) -> subprocess.CompletedProcess[str]:
    traces = workspace / "traces.jsonl"
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


def _parse_traces(workspace: Path) -> dict[str, Any]:
    traces_path = workspace / "traces.jsonl"
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

        proc = _run_sol_execbench(workspace)
        parsed = _parse_traces(workspace)
        expected = _load_workload_count(workspace)
        elapsed = time.time() - start

        common_artifacts = {
            "workspace": str(workspace),
            "returncode": proc.returncode,
            "stdout_tail": _tail(proc.stdout),
            "stderr_tail": _tail(proc.stderr),
            "per_workload": parsed.get("per_workload", []),
        }

        if parsed.get("error"):
            return _result(
                "execution_failed",
                f"SOL-ExecBench failed before producing traces: {parsed['error']}. stderr_tail={_tail(proc.stderr, 1200)}",
                0.0,
                metrics={"eval_time_s": elapsed, "target_latency_ms": TARGET_LATENCY_MS},
                artifacts=common_artifacts,
            )

        total = int(parsed.get("total", 0))
        passed = int(parsed.get("passed", 0))
        failures = parsed.get("failures", [])
        latency_ms = float(parsed.get("latency_ms_geomean", 0.0) or 0.0)

        metrics = {
            "eval_time_s": elapsed,
            "target_latency_ms": TARGET_LATENCY_MS,
            "latency_ms_geomean": latency_ms,
            "latency_ms_arith_mean": parsed.get("latency_ms_arith_mean", 0.0),
            "passed": passed,
            "total": total,
            "expected_total": expected,
            "max_abs_err": parsed.get("max_abs_err", 0.0),
            "max_rel_err": parsed.get("max_rel_err", 0.0),
        }

        if proc.returncode != 0 or total != expected or passed != expected or failures:
            summary = (
                f"Correctness/coverage gate failed: passed {passed}/{expected}, "
                f"returncode={proc.returncode}. failures={failures[:6]}"
            )
            return _result(
                "validation_failed",
                summary,
                0.0,
                metrics=metrics,
                artifacts=common_artifacts,
            )

        if latency_ms <= 0:
            return _result(
                "execution_failed",
                "All workloads passed but no positive latency was parsed.",
                0.0,
                metrics=metrics,
                artifacts=common_artifacts,
            )

        score = TARGET_LATENCY_MS / latency_ms
        status_line = "target met" if score >= 1.0 else "target not met"
        summary = (
            f"All {passed}/{expected} workloads passed; geomean latency "
            f"{latency_ms:.6f} ms vs target {TARGET_LATENCY_MS:.6f} ms; "
            f"score={score:.6f} ({status_line})."
        )
        return _result(
            "success",
            summary,
            score,
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
