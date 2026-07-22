#!/usr/bin/env python3
"""Best-effort Nsight Compute evidence for the SOL58 PES Summary stage.

Profiling is deliberately separate from the evaluator's correctness and timing
runs. A profiling failure is diagnostic data, never a fitness failure.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
ANALYZE_REPORTS = REPO_ROOT / "tools" / "ncu_helpers" / "analyze_reports.py"
CLASSIFY_NCU = REPO_ROOT / "tools" / "classify_ncu.py"

_CUDA_KERNEL_RE = re.compile(
    r"\b__global__\s+"
    r"(?:__launch_bounds__\s*\([^)]*\)\s*)?"
    r"(?:[A-Za-z_]\w*(?:::\w+)*(?:\s*<[^;{}()]*>)?[\s*&]+)+"
    r"(?P<name>[A-Za-z_]\w*)\s*\(",
    re.MULTILINE,
)
_CUTE_KERNEL_RE = re.compile(
    r"@(?:cute\.)?kernel\b[\s\S]{0,300}?\bdef\s+(?P<name>[A-Za-z_]\w*)\s*\(",
    re.MULTILINE,
)
_STAGING_DIR_RE = re.compile(r"Staging dir:\s*(?P<path>[^\r\n\x1b]+)")

_KEY_METRIC_ALIASES = {
    "gpu__time_duration.sum": "duration_ns",
    "launch__grid_size": "grid_size",
    "launch__block_size": "block_size",
    "launch__waves_per_multiprocessor": "waves_per_sm",
    "launch__registers_per_thread": "registers_per_thread",
    "sm__throughput.avg.pct_of_peak_sustained_elapsed": "sm_throughput_pct",
    "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed": "memory_throughput_pct",
    "dram__bytes_read.sum.pct_of_peak_sustained_elapsed": "dram_read_pct",
    "dram__bytes_write.sum.pct_of_peak_sustained_elapsed": "dram_write_pct",
    "lts__t_sector_hit_rate.pct": "l2_hit_rate_pct",
    "sm__warps_active.avg.pct_of_peak_sustained_active": "achieved_occupancy_pct",
    "sm__maximum_warps_per_active_cycle_pct": "theoretical_occupancy_pct",
    "smsp__sass_inst_executed_op_local_ld.sum": "local_load_instructions",
    "smsp__sass_inst_executed_op_local_st.sum": "local_store_instructions",
    "smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio": "stall_long_scoreboard",
    "smsp__average_warps_issue_stalled_short_scoreboard_per_issue_active.ratio": "stall_short_scoreboard",
    "smsp__average_warps_issue_stalled_wait_per_issue_active.ratio": "stall_wait",
    "smsp__average_warps_issue_stalled_barrier_per_issue_active.ratio": "stall_barrier",
    "smsp__average_warps_issue_stalled_mio_throttle_per_issue_active.ratio": "stall_mio_throttle",
    "smsp__average_warps_issue_stalled_lg_throttle_per_issue_active.ratio": "stall_lg_throttle",
}

_IMPLICATIONS = {
    "A": "Increase useful parallel work per launch or fuse/persist small-grid stages to reduce idle SMs.",
    "B": "Retile the grid or use persistent work distribution to reduce the partially occupied final wave.",
    "C": "Make global loads contiguous and vectorized; verify lane-to-address mapping before adding staging.",
    "D": "Aggregate or vectorize sparse stores so each memory sector carries more useful output bytes.",
    "E": "Expose independent memory operations with more warps, prefetching, or a staged async/TMA pipeline.",
    "F": "Reduce scalar instruction work on the hot path; tensor cores are usually not useful for this integer sort.",
    "G": "Privatize histogram updates and merge hierarchically to reduce global atomic serialization.",
    "H": "Pad or permute shared-memory layouts and inspect bank mapping for the reported access width.",
    "I": "Remove redundant block barriers or replace block-wide synchronization with warp-local primitives.",
    "J": "Reduce register/shared-memory residency limits or increase independent blocks without adding tail waste.",
    "K": "Reduce live ranges and thread-local arrays; confirm spills disappear before accepting the rewrite.",
    "L": "Remove accidental FP64 expressions or constants from the device hot path.",
    "M": "Overlap independent load, transform, and store stages with double buffering or producer/consumer warps.",
    "N": "Reduce lane-divergent branches with predication, warp-uniform dispatch, or shape-specialized kernels.",
    "_mem": "Reduce transferred bytes and increase on-chip reuse before spending complexity on more arithmetic throughput.",
}


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def extract_kernel_names(source: str, source_language: str) -> list[str]:
    """Return stable user-kernel names suitable for an NCU regex filter."""
    pattern = _CUTE_KERNEL_RE if source_language == "cute_dsl" else _CUDA_KERNEL_RE
    names: list[str] = []
    for match in pattern.finditer(source):
        name = match.group("name")
        if name not in names:
            names.append(name)
    return names[:32]


def should_profile(
    *, enabled: bool, policy: str, improved: bool, iteration: int | None
) -> tuple[bool, str]:
    if not enabled:
        return False, "disabled"
    policy = (policy or "all_correct").strip().lower()
    if policy == "all_correct":
        return True, "all_correct"
    if policy in {"local_best", "improving"}:
        return improved, "local_best" if improved else "not_local_best"
    if policy == "periodic":
        try:
            interval = max(1, int(os.environ.get("SOL58_NCU_PROFILE_INTERVAL", "5")))
        except (TypeError, ValueError):
            return False, "invalid_profile_interval"
        selected = improved or (iteration is not None and iteration % interval == 0)
        return selected, (
            "local_best_or_periodic" if selected else "not_periodic_iteration"
        )
    return False, f"invalid_policy:{policy}"


def select_workload(
    workload_lines: list[str],
    per_workload: list[dict[str, Any]],
    selector: str,
) -> tuple[int, dict[str, Any], str]:
    """Choose one workload, defaulting to the slowest measured representative."""
    if not workload_lines:
        raise ValueError("workload.jsonl is empty")
    records = [json.loads(line) for line in workload_lines]
    selector = (selector or "slowest").strip()

    if selector == "slowest":
        measured = [
            item
            for item in per_workload
            if isinstance(item, dict)
            and isinstance(item.get("index"), int)
            and isinstance(item.get("latency_ms"), (int, float))
            and 0 <= item["index"] < len(records)
        ]
        index = (
            max(measured, key=lambda item: float(item["latency_ms"]))["index"]
            if measured
            else 0
        )
        reason = "slowest_measured" if measured else "fallback_first"
    elif selector.isdigit():
        index = int(selector)
        if not 0 <= index < len(records):
            raise ValueError(
                f"NCU workload index {index} is outside 0..{len(records) - 1}"
            )
        reason = "configured_index"
    else:
        matches = [
            i for i, record in enumerate(records) if str(record.get("uuid")) == selector
        ]
        if not matches:
            raise ValueError(
                f"NCU workload selector {selector!r} is neither an index nor a known UUID"
            )
        index = matches[0]
        reason = "configured_uuid"

    measured_entry = next(
        (
            item
            for item in per_workload
            if isinstance(item, dict) and item.get("index") == index
        ),
        {},
    )
    record = records[index]
    descriptor = {
        "index": index,
        "uuid": record.get("uuid"),
        "axes": record.get("axes") or measured_entry.get("axes") or {},
        "measured_latency_ms": measured_entry.get("latency_ms"),
        "selection": reason,
    }
    return index, descriptor, json.dumps(record, sort_keys=True, separators=(",", ":"))


def _run_profile_command(
    command: list[str],
    *,
    cwd: Path,
    env: dict[str, str],
    timeout: float,
) -> subprocess.CompletedProcess[str]:
    process = subprocess.Popen(
        command,
        cwd=str(cwd),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    try:
        stdout, stderr = process.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            stdout, stderr = process.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            stdout, stderr = process.communicate()
        raise subprocess.TimeoutExpired(command, timeout, output=stdout, stderr=stderr)
    return subprocess.CompletedProcess(command, process.returncode, stdout, stderr)


def _parse_report(
    report_path: Path, profile_dir: Path, timeout: float
) -> tuple[dict[str, Any], dict[str, Any]]:
    parse_command = [
        sys.executable,
        str(ANALYZE_REPORTS),
        "--run-dir",
        str(profile_dir),
        "--report",
        str(report_path),
        "--tag",
        "summary",
    ]
    try:
        parse_retries = max(1, int(os.environ.get("SOL58_NCU_PARSE_RETRIES", "3")))
    except ValueError:
        parse_retries = 3
    parse_proc: subprocess.CompletedProcess[str] | None = None
    for attempt in range(parse_retries):
        parse_proc = subprocess.run(
            parse_command,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if parse_proc.returncode == 0:
            break
        if attempt + 1 < parse_retries:
            time.sleep(min(4, 2**attempt))
    assert parse_proc is not None
    if parse_proc.returncode != 0:
        detail = (parse_proc.stderr or parse_proc.stdout or "no parser output")[-1200:]
        raise RuntimeError(
            f"NCU report parser failed after {parse_retries} attempts "
            f"(returncode={parse_proc.returncode}): {detail}"
        )

    metrics_path = profile_dir / "analysis" / "metrics_key_summary.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    classify_proc = subprocess.run(
        [sys.executable, str(CLASSIFY_NCU), "--metrics", str(metrics_path), "--json"],
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    if classify_proc.returncode != 0:
        detail = (classify_proc.stderr or classify_proc.stdout or "no classifier output")[
            -1200:
        ]
        raise RuntimeError(
            f"NCU classifier failed (returncode={classify_proc.returncode}): {detail}"
        )
    classification = json.loads(classify_proc.stdout)
    return metrics, classification


def _compact_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    compact: dict[str, Any] = {}
    for raw_name, alias in _KEY_METRIC_ALIASES.items():
        value = metrics.get(raw_name)
        if isinstance(value, (int, float)):
            compact[alias] = value
    return compact


def _implications(findings: list[dict[str, Any]]) -> list[str]:
    actions: list[str] = []
    for finding in findings:
        action = _IMPLICATIONS.get(str(finding.get("pattern")))
        if action and action not in actions:
            actions.append(action)
    return actions


def _write_log(
    path: Path, content: str | bytes | None, limit: int = 100_000
) -> None:
    text = content or ""
    if isinstance(text, bytes):
        text = text.decode("utf-8", errors="replace")
    if len(text) > limit:
        text = text[: limit // 2] + "\n...[truncated]...\n" + text[-limit // 2 :]
    path.write_text(text, encoding="utf-8")


def _resolve_executable(program: str) -> Path | None:
    executable = Path(program)
    if executable.is_file():
        return executable.absolute()
    resolved = shutil.which(program)
    return Path(resolved).absolute() if resolved else None


def _profile_environment(sol_execbench: str) -> dict[str, str]:
    """Use the evaluator CLI's virtualenv for its generated build subprocesses."""
    env = {
        **os.environ,
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    }
    # The staged evaluator is self-contained. Inheriting LoongFlow's
    # sitecustomize path can load ABI-incompatible packages under NCU injection.
    env.pop("PYTHONPATH", None)
    executable = _resolve_executable(sol_execbench)
    if executable is not None:
        executable_dir = str(executable.parent)
        path_entries = env.get("PATH", "").split(os.pathsep)
        env["PATH"] = os.pathsep.join(
            [executable_dir, *(item for item in path_entries if item != executable_dir)]
        )
    return env


def _sol_execbench_python(sol_execbench: str) -> str:
    executable = _resolve_executable(sol_execbench)
    if executable is not None:
        for name in ("python", "python3"):
            candidate = executable.parent / name
            if candidate.is_file():
                return str(candidate)
    return sys.executable


def _extract_staging_dir(stdout: str | None, stderr: str | None) -> Path:
    output = f"{stdout or ''}\n{stderr or ''}"
    match = _STAGING_DIR_RE.search(output)
    if not match:
        raise RuntimeError("sol-execbench preflight did not report its staging directory")
    staging_dir = Path(match.group("path").strip())
    if not staging_dir.is_dir():
        raise RuntimeError(
            f"sol-execbench preflight staging directory is missing: {staging_dir}"
        )
    return staging_dir


def _preserve_failed_profile(
    temp_dir: Path,
    cache_root: Path,
    cache_key: str,
    result: dict[str, Any],
) -> dict[str, Any]:
    """Keep the latest failed capture per cache key for reproducible diagnosis."""
    failure_dir = cache_root / "failed" / cache_key
    artifacts = {
        "workspace": str(failure_dir / "workspace"),
        "staging": str(failure_dir / "staging"),
        "preflight_stdout_log": str(failure_dir / "preflight_stdout.log"),
        "preflight_stderr_log": str(failure_dir / "preflight_stderr.log"),
        "stdout_log": str(failure_dir / "ncu_stdout.log"),
        "stderr_log": str(failure_dir / "ncu_stderr.log"),
        "failure": str(failure_dir / "ncu_analysis.json"),
    }
    result["artifacts"] = artifacts
    try:
        (temp_dir / "ncu_analysis.json").write_text(
            json.dumps(result, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        failure_dir.parent.mkdir(parents=True, exist_ok=True)
        if failure_dir.exists():
            shutil.rmtree(failure_dir)
        temp_dir.replace(failure_dir)
    except Exception as exc:
        result.pop("artifacts", None)
        result["artifact_error"] = str(exc)[-500:]
        shutil.rmtree(temp_dir, ignore_errors=True)
    return result


def collect_ncu_analysis(
    *,
    workspace: Path,
    kernel_source: str,
    source_language: str,
    source_sha256: str,
    measurement_profile: dict[str, Any],
    per_workload: list[dict[str, Any]],
    cache_root: Path,
    sol_execbench: str,
    compile_timeout: int,
    run_timeout: int,
) -> dict[str, Any]:
    """Collect and classify one representative launch, with source-level caching."""
    started = time.time()
    ncu_binary = shutil.which(os.environ.get("SOL58_NCU_BINARY", "ncu"))
    if not ncu_binary:
        return {
            "enabled": True,
            "status": "unavailable",
            "error": "ncu executable not found",
        }
    if not ANALYZE_REPORTS.is_file() or not CLASSIFY_NCU.is_file():
        return {
            "enabled": True,
            "status": "unavailable",
            "error": "bundled NCU analysis tools not found",
        }

    kernel_names = extract_kernel_names(kernel_source, source_language)
    if not kernel_names:
        return {
            "enabled": True,
            "status": "skipped",
            "reason": "no_user_kernel_name_detected",
            "profile_time_s": time.time() - started,
        }

    workload_path = workspace / "workload.jsonl"
    try:
        workload_lines = [
            line
            for line in workload_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        _, workload, canonical_workload = select_workload(
            workload_lines,
            per_workload,
            os.environ.get("SOL58_NCU_WORKLOAD", "slowest"),
        )
    except Exception as exc:
        return {
            "enabled": True,
            "status": "skipped",
            "reason": f"workload_selection_failed:{exc}",
            "profile_time_s": time.time() - started,
        }

    ncu_set = os.environ.get("SOL58_NCU_SET", "full").strip() or "full"
    launch_count = max(1, int(os.environ.get("SOL58_NCU_LAUNCH_COUNT", "1")))
    identity = {
        "schema_version": 2,
        "profile_strategy": "precompile_eval_driver_v1",
        "source_sha256": source_sha256,
        "source_language": source_language,
        "measurement_profile_id": measurement_profile.get("id"),
        "workload": canonical_workload,
        "kernel_names": kernel_names,
        "ncu_set": ncu_set,
        "launch_count": launch_count,
        "sol_execbench": (
            str(Path(sol_execbench).resolve())
            if Path(sol_execbench).exists()
            else sol_execbench
        ),
    }
    cache_key = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    cache_root.mkdir(parents=True, exist_ok=True)
    final_dir = cache_root / cache_key
    result_path = final_dir / "ncu_analysis.json"
    lock_path = cache_root / f"{cache_key}.lock"

    with lock_path.open("a+") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        if result_path.is_file():
            try:
                cached = json.loads(result_path.read_text(encoding="utf-8"))
                if cached.get("status") == "completed":
                    cached["cache_hit"] = True
                    return cached
            except Exception:
                pass

        temp_dir = cache_root / f".{cache_key}.{os.getpid()}.{uuid.uuid4().hex[:8]}"
        profile_workspace = temp_dir / "workspace"
        profile_workspace.mkdir(parents=True)
        try:
            for name in ("definition.json", "reference.py", "solution.json"):
                source_path = workspace / name
                if source_path.is_file():
                    shutil.copy2(source_path, profile_workspace / name)
            source_name = "kernel.py" if source_language == "cute_dsl" else "kernel.cu"
            shutil.copy2(workspace / source_name, profile_workspace / source_name)
            (profile_workspace / "workload.jsonl").write_text(
                canonical_workload + "\n", encoding="utf-8"
            )
            profile_config = {
                "warmup_runs": 0,
                "iterations": 1,
                # The outer evaluator owns clock locking. Avoid profiler injection
                # into nvidia-smi and preserve the already selected frequencies.
                "lock_clocks": False,
                "benchmark_reference": False,
                "seed": int(measurement_profile.get("seed", 200)),
            }
            (profile_workspace / "config.json").write_text(
                json.dumps(profile_config, indent=2) + "\n", encoding="utf-8"
            )

            report_base = temp_dir / "ncu_profile"
            report_path = report_base.with_suffix(".ncu-rep")
            escaped_names = "|".join(re.escape(name) for name in kernel_names)
            kernel_filter = f"regex:.*(?:{escaped_names}).*"
            profile_env = _profile_environment(sol_execbench)
            preflight_command = [
                sol_execbench,
                ".",
                "--solution",
                "solution.json",
                "--config",
                "config.json",
                "--compile-timeout",
                str(compile_timeout),
                "--timeout",
                str(run_timeout),
                "--keep-staging",
                "--verbose",
                "-o",
                "ncu_preflight_traces.jsonl",
            ]
            preflight = _run_profile_command(
                preflight_command,
                cwd=profile_workspace,
                env=profile_env,
                timeout=max(1.0, compile_timeout + run_timeout + 60),
            )
            _write_log(temp_dir / "preflight_stdout.log", preflight.stdout)
            _write_log(temp_dir / "preflight_stderr.log", preflight.stderr)
            generated_staging = _extract_staging_dir(
                preflight.stdout, preflight.stderr
            )
            profile_staging = temp_dir / "staging"
            shutil.move(str(generated_staging), profile_staging)
            if preflight.returncode != 0:
                raise RuntimeError(
                    "sol-execbench NCU preflight failed "
                    f"(returncode={preflight.returncode}): "
                    f"{(preflight.stderr or preflight.stdout or '')[-1200:]}"
                )
            if not (profile_staging / "eval_driver.py").is_file():
                raise RuntimeError("sol-execbench preflight produced no eval_driver.py")
            if source_language == "cuda_cpp" and not (
                profile_staging / "benchmark_kernel.so"
            ).is_file():
                raise RuntimeError("sol-execbench preflight produced no CUDA artifact")

            command = [
                ncu_binary,
                "--target-processes",
                "application-only",
                "--set",
                ncu_set,
                "--replay-mode",
                "kernel",
                "--clock-control",
                "none",
                "--kernel-name-base",
                "demangled",
                "--kernel-name",
                kernel_filter,
                "--launch-count",
                str(launch_count),
                "--kill",
                "yes",
                "--check-exit-code",
                "0",
                "--force-overwrite",
                "-o",
                str(report_base),
                _sol_execbench_python(sol_execbench),
                "eval_driver.py",
            ]
            timeout = max(1.0, float(os.environ.get("SOL58_NCU_TIMEOUT", "180")))
            proc = _run_profile_command(
                command,
                cwd=profile_staging,
                env=profile_env,
                timeout=timeout,
            )
            _write_log(temp_dir / "ncu_stdout.log", proc.stdout)
            _write_log(temp_dir / "ncu_stderr.log", proc.stderr)
            if not report_path.is_file():
                raise RuntimeError(
                    f"ncu produced no report (returncode={proc.returncode}): {(proc.stderr or '')[-1000:]}"
                )

            raw_metrics, classification = _parse_report(
                report_path,
                temp_dir,
                max(1.0, float(os.environ.get("SOL58_NCU_PARSE_TIMEOUT", "60"))),
            )
            max_findings = max(1, int(os.environ.get("SOL58_NCU_MAX_FINDINGS", "8")))
            findings = list(classification.get("findings") or [])[:max_findings]
            final_artifacts = {
                "report": str(final_dir / "ncu_profile.ncu-rep"),
                "key_metrics": str(final_dir / "analysis" / "metrics_key_summary.json"),
                "all_metrics": str(final_dir / "analysis" / "metrics_all_summary.json"),
                "preflight_stdout_log": str(final_dir / "preflight_stdout.log"),
                "preflight_stderr_log": str(final_dir / "preflight_stderr.log"),
                "stdout_log": str(final_dir / "ncu_stdout.log"),
                "stderr_log": str(final_dir / "ncu_stderr.log"),
            }
            result = {
                "enabled": True,
                "status": "completed",
                "cache_hit": False,
                "scope": "first matching user-kernel launch on one representative workload",
                "workload": workload,
                "kernel_name": raw_metrics.get("__kernel_name__"),
                "kernel_filter_names": kernel_names,
                "key_metrics": _compact_metrics(raw_metrics),
                "findings": findings,
                "symptoms": list(classification.get("symptoms") or []),
                "optimization_implications": _implications(findings),
                "profile_time_s": time.time() - started,
                "artifacts": final_artifacts,
            }
            (temp_dir / "ncu_analysis.json").write_text(
                json.dumps(result, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            shutil.rmtree(profile_staging, ignore_errors=True)
            if final_dir.exists():
                shutil.rmtree(final_dir)
            temp_dir.replace(final_dir)
            return result
        except subprocess.TimeoutExpired as exc:
            _write_log(temp_dir / "ncu_stdout.log", exc.output)
            _write_log(temp_dir / "ncu_stderr.log", exc.stderr)
            result = {
                "enabled": True,
                "status": "timeout",
                "workload": workload,
                "kernel_filter_names": kernel_names,
                "timeout_s": exc.timeout,
                "profile_time_s": time.time() - started,
            }
            return _preserve_failed_profile(temp_dir, cache_root, cache_key, result)
        except Exception as exc:
            result = {
                "enabled": True,
                "status": "failed",
                "workload": workload,
                "kernel_filter_names": kernel_names,
                "error": str(exc)[-1200:],
                "profile_time_s": time.time() - started,
            }
            return _preserve_failed_profile(temp_dir, cache_root, cache_key, result)
