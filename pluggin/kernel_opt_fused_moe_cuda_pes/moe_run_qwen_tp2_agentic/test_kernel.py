#!/usr/bin/env python3
"""Correctness and latency evaluator for FlashInfer-aligned FP4 MoE candidates."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import torch

from reference import ARG_NAMES, make_inputs, run_reference


def _load_candidate(path: Path):
    workspace = Path(__file__).resolve().parent
    candidate_root = path.parent if (path.parent / "src").exists() else workspace
    os.environ["FUSED_MOE_CUDA_ROOT"] = str(candidate_root)
    if str(workspace) not in sys.path:
        sys.path.insert(0, str(workspace))
    if str(path.parent) not in sys.path:
        sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location("candidate_kernel", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import candidate from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["candidate_kernel"] = module
    spec.loader.exec_module(module)
    return module


def _bench(fn, warmup: int, rep: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    times = []
    for _ in range(rep):
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end) * 1000.0)
    times.sort()
    return times[len(times) // 2]


def _first(result):
    if isinstance(result, (list, tuple)):
        return result[0]
    return result


def _case_kwargs(args, tokens: int):
    return {
        "preset": args.preset,
        "tokens": tokens,
        "hidden_size": args.hidden_size,
        "intermediate_size": args.intermediate_size,
        "local_num_experts": args.local_num_experts,
        "local_expert_offset": args.local_expert_offset,
        "seed": args.seed,
    }


def _shape_from_inputs(inputs) -> dict[str, int]:
    return {
        "hidden_size": inputs[2].shape[1],
        "intermediate_size": inputs[20],
        "num_experts": inputs[16],
        "top_k": inputs[17],
        "local_expert_offset": inputs[21],
        "local_num_experts": inputs[22],
        "routing_method_type": inputs[24],
    }


def _error_text(exc: BaseException) -> str:
    text = f"{type(exc).__name__}: {exc}"
    return text if len(text) <= 2000 else text[:2000] + "...<truncated>"


def _disable_core_dumps() -> None:
    try:
        import resource

        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    except Exception:
        pass


def _bench_flashinfer_in_worker(token_values, warmup: int, rep: int, args) -> dict:
    try:
        import flashinfer
    except Exception as exc:  # pragma: no cover - depends on host install
        return {"status": "UNAVAILABLE", "error": _error_text(exc), "rows": []}

    fn = getattr(flashinfer, "trtllm_fp4_block_scale_moe", None)
    if fn is None:
        return {
            "status": "UNAVAILABLE",
            "error": "flashinfer.trtllm_fp4_block_scale_moe is not available",
            "rows": [],
        }

    rows = []
    max_abs = 0.0
    max_rel = 0.0
    latencies = []
    for tokens in token_values:
        inputs = make_inputs(**_case_kwargs(args, tokens))
        row = {
            "tokens": tokens,
            "preset": args.preset,
            "shape": _shape_from_inputs(inputs),
        }
        try:
            ref = _first(run_reference(*inputs))
            out = _first(fn(*inputs))
            torch.cuda.synchronize()
            diff = (out.float() - ref.float()).abs()
            abs_err = float(diff.max().item())
            rel_err = float((diff / ref.float().abs().clamp_min(1e-3)).max().item())
            max_abs = max(max_abs, abs_err)
            max_rel = max(max_rel, rel_err)
            row["max_abs"] = abs_err
            row["max_rel"] = rel_err
            if abs_err > args.atol and rel_err > args.rtol:
                row["status"] = "FAIL"
            else:
                row["status"] = "PASS"
                if args.mode == "profile":
                    row["flashinfer_us"] = _bench(lambda: fn(*inputs), warmup, rep)
                    latencies.append(row["flashinfer_us"])
        except Exception as exc:
            row["status"] = "ERROR"
            row["error"] = _error_text(exc)
            rows.append(row)
            break
        rows.append(row)

    status = "PASS" if rows and all(row["status"] == "PASS" for row in rows) else "ERROR"
    if any(row.get("status") == "FAIL" for row in rows):
        status = "FAIL"
    return {
        "status": status,
        "version": getattr(flashinfer, "__version__", "unknown"),
        "operator": "flashinfer.trtllm_fp4_block_scale_moe",
        "max_abs": max_abs,
        "max_rel": max_rel,
        "mean_latency_us": sum(latencies) / len(latencies) if latencies else None,
        "rows": rows,
    }


def _run_flashinfer_worker(token_values, warmup: int, rep: int, args) -> dict:
    with tempfile.NamedTemporaryFile(prefix="flashinfer_moe_", suffix=".json", delete=False) as tmp:
        tmp_path = Path(tmp.name)
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--flashinfer-worker",
        "--mode",
        args.mode,
        "--preset",
        args.preset,
        "--tokens",
        ",".join(str(t) for t in token_values),
        "--seed",
        str(args.seed),
        "--warmup",
        str(warmup),
        "--rep",
        str(rep),
        "--atol",
        str(args.atol),
        "--rtol",
        str(args.rtol),
        "--local-expert-offset",
        str(args.local_expert_offset),
        "--json-out",
        str(tmp_path),
    ]
    if args.hidden_size is not None:
        cmd += ["--hidden-size", str(args.hidden_size)]
    if args.intermediate_size is not None:
        cmd += ["--intermediate-size", str(args.intermediate_size)]
    if args.local_num_experts is not None:
        cmd += ["--local-num-experts", str(args.local_num_experts)]

    try:
        proc = subprocess.run(
            cmd,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=args.flashinfer_timeout_s,
            check=False,
            preexec_fn=_disable_core_dumps if os.name == "posix" else None,
        )
        if tmp_path.exists() and tmp_path.stat().st_size > 0:
            return json.loads(tmp_path.read_text())
        return {
            "status": "ERROR",
            "error": (
                f"worker exited with code {proc.returncode}; "
                f"stdout={proc.stdout[-1000:]!r}; stderr={proc.stderr[-1000:]!r}"
            ),
            "rows": [],
        }
    except subprocess.TimeoutExpired as exc:
        return {"status": "ERROR", "error": f"worker timeout after {exc.timeout}s", "rows": []}
    finally:
        try:
            tmp_path.unlink()
        except OSError:
            pass


def _attach_flashinfer_gate(result: dict, flashinfer_result: dict, target_speedup: float) -> None:
    result["flashinfer"] = flashinfer_result
    flash_status = flashinfer_result.get("status")
    if flash_status != "PASS":
        result["target_status"] = "BLOCKED"
        result["target_reason"] = flashinfer_result.get("error") or f"flashinfer_status={flash_status}"
        return

    by_tokens = {row.get("tokens"): row for row in flashinfer_result.get("rows") or []}
    speedups = []
    for row in result["rows"]:
        flash_row = by_tokens.get(row.get("tokens"))
        flash_us = (flash_row or {}).get("flashinfer_us")
        if flash_us is None:
            continue
        row["flashinfer_us"] = flash_us
        candidate_us = row.get("candidate_us")
        if candidate_us:
            row["speedup_vs_flashinfer"] = flash_us / candidate_us
            row["beats_flashinfer"] = row["speedup_vs_flashinfer"] > target_speedup
            speedups.append(row["speedup_vs_flashinfer"])

    if not speedups:
        result["target_status"] = "BLOCKED"
        result["target_reason"] = "FlashInfer did not produce comparable latency rows"
        return
    result["target_speedup_vs_flashinfer"] = sum(speedups) / len(speedups)
    result["target_status"] = (
        "MET"
        if result["status"] == "PASS" and result["target_speedup_vs_flashinfer"] > target_speedup
        else "MISS"
    )


def evaluate(candidate_path: Path, token_values, warmup: int, rep: int, args):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    cand = _load_candidate(candidate_path)

    rows = []
    max_abs = 0.0
    max_rel = 0.0
    for tokens in token_values:
        inputs = make_inputs(**_case_kwargs(args, tokens))
        ref = _first(run_reference(*inputs))
        out = _first(cand.run(*inputs))
        torch.cuda.synchronize()
        diff = (out.float() - ref.float()).abs()
        abs_err = float(diff.max().item())
        rel_err = float((diff / ref.float().abs().clamp_min(1e-3)).max().item())
        max_abs = max(max_abs, abs_err)
        max_rel = max(max_rel, rel_err)
        row = {
            "tokens": tokens,
            "preset": args.preset,
            "shape": _shape_from_inputs(inputs),
            "max_abs": abs_err,
            "max_rel": rel_err,
        }
        if abs_err > args.atol and rel_err > args.rtol:
            row["status"] = "FAIL"
            rows.append(row)
            continue

        row["status"] = "PASS"
        if args.mode == "profile":
            row["candidate_us"] = _bench(lambda: cand.run(*inputs), warmup, rep)
            row["reference_us"] = _bench(
                lambda: run_reference(*inputs), max(1, warmup // 2), max(3, rep // 3)
            )
            row["speedup_vs_reference"] = (
                row["reference_us"] / row["candidate_us"] if row["candidate_us"] > 0 else 0.0
            )
        rows.append(row)

    status = "PASS" if all(row["status"] == "PASS" for row in rows) else "FAIL"
    latency_rows = [row["candidate_us"] for row in rows if row.get("candidate_us") is not None]
    result = {
        "status": status,
        "operator": "flashinfer.trtllm_fp4_block_scale_moe",
        "arg_names": ARG_NAMES,
        "max_abs": max_abs,
        "max_rel": max_rel,
        "mean_latency_us": sum(latency_rows) / len(latency_rows) if latency_rows else None,
        "rows": rows,
    }
    if args.mode == "profile" and args.compare_flashinfer:
        flashinfer_result = _run_flashinfer_worker(token_values, warmup, rep, args)
        _attach_flashinfer_gate(result, flashinfer_result, args.flashinfer_target_speedup)
        if args.require_flashinfer and result.get("target_status") == "BLOCKED":
            result["status"] = "BLOCKED"
    return result


def _parse_tokens(raw: str):
    return [int(item) for item in raw.split(",") if item.strip()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kernel", default="kernel.py")
    parser.add_argument("--mode", choices=("correctness", "profile"), default="correctness")
    parser.add_argument("--preset", choices=("smoke", "qwen_micro", "qwen_tp2"), default="smoke")
    parser.add_argument("--tokens", default="2,4")
    parser.add_argument("--hidden-size", type=int)
    parser.add_argument("--intermediate-size", type=int)
    parser.add_argument("--local-num-experts", type=int)
    parser.add_argument("--local-expert-offset", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--rep", type=int, default=9)
    parser.add_argument("--atol", type=float, default=8e-2)
    parser.add_argument("--rtol", type=float, default=8e-2)
    parser.add_argument("--compare-flashinfer", action="store_true")
    parser.add_argument("--require-flashinfer", action="store_true")
    parser.add_argument("--flashinfer-worker", action="store_true")
    parser.add_argument("--flashinfer-target-speedup", type=float, default=1.0)
    parser.add_argument("--flashinfer-timeout-s", type=float, default=120.0)
    parser.add_argument("--json-out")
    args = parser.parse_args()

    token_values = _parse_tokens(args.tokens)
    if args.flashinfer_worker:
        result = _bench_flashinfer_in_worker(token_values, args.warmup, args.rep, args)
    else:
        result = evaluate(Path(args.kernel).resolve(), token_values, args.warmup, args.rep, args)
    text = json.dumps(result, indent=2)
    print(text)
    if args.json_out:
        Path(args.json_out).write_text(text + "\n")
    if result["status"] == "PASS":
        return 0
    if result["status"] == "BLOCKED":
        return 2
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
