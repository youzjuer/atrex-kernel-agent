#!/usr/bin/env python3
"""Correctness and latency evaluator for FlashInfer-aligned FP4 MoE candidates."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
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
            "shape": {
                "hidden_size": inputs[2].shape[1],
                "intermediate_size": inputs[20],
                "num_experts": inputs[16],
                "top_k": inputs[17],
                "local_expert_offset": inputs[21],
                "local_num_experts": inputs[22],
                "routing_method_type": inputs[24],
            },
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
    return {
        "status": status,
        "operator": "flashinfer.trtllm_fp4_block_scale_moe",
        "arg_names": ARG_NAMES,
        "max_abs": max_abs,
        "max_rel": max_rel,
        "mean_latency_us": sum(latency_rows) / len(latency_rows) if latency_rows else None,
        "rows": rows,
    }


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
    parser.add_argument("--json-out")
    args = parser.parse_args()

    result = evaluate(
        Path(args.kernel).resolve(), _parse_tokens(args.tokens), args.warmup, args.rep, args
    )
    text = json.dumps(result, indent=2)
    print(text)
    if args.json_out:
        Path(args.json_out).write_text(text + "\n")
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
