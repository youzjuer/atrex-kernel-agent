#!/usr/bin/env python3
"""Correctness and latency evaluator for CUDA fused-MoE PES candidates."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path

import torch

from reference import make_inputs, run_reference


M_VALUES = [8, 16, 32]


def _load_candidate(path: Path):
    workspace = Path(__file__).resolve().parent
    candidate_root = path.parent if (path.parent / "src").exists() else workspace
    os.environ["FUSED_MOE_CUDA_ROOT"] = str(candidate_root)
    if str(path.parent) not in sys.path:
        sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location("candidate_kernel", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import candidate from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["candidate_kernel"] = module
    spec.loader.exec_module(module)
    return module


def _parse_m_values(raw: str):
    if raw == "all":
        return M_VALUES
    return [int(item) for item in raw.split(",") if item.strip()]


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


def evaluate(candidate_path: Path, m_values, warmup: int, rep: int):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    cand = _load_candidate(candidate_path)

    rows = []
    max_abs = 0.0
    max_rel = 0.0
    for M in m_values:
        inputs = make_inputs(M=M)
        ref = run_reference(*inputs)
        out = cand.run(*inputs)
        torch.cuda.synchronize()
        diff = (out - ref).abs()
        abs_err = float(diff.max().item())
        rel_err = float((diff / ref.abs().clamp_min(1e-6)).max().item())
        max_abs = max(max_abs, abs_err)
        max_rel = max(max_rel, rel_err)
        if abs_err > 1e-3 or rel_err > 1e-3:
            rows.append({"M": M, "status": "FAIL", "max_abs": abs_err, "max_rel": rel_err})
            continue

        cand_us = _bench(lambda: cand.run(*inputs), warmup, rep)
        ref_us = _bench(lambda: run_reference(*inputs), max(1, warmup // 2), max(3, rep // 3))
        rows.append(
            {
                "M": M,
                "status": "PASS",
                "max_abs": abs_err,
                "max_rel": rel_err,
                "candidate_us": cand_us,
                "reference_us": ref_us,
                "speedup_vs_reference": ref_us / cand_us if cand_us > 0 else 0.0,
            }
        )
    status = "PASS" if all(row["status"] == "PASS" for row in rows) else "FAIL"
    latency_rows = [row["candidate_us"] for row in rows if row["status"] == "PASS"]
    mean_latency = sum(latency_rows) / len(latency_rows) if latency_rows else None
    result = {
        "status": status,
        "max_abs": max_abs,
        "max_rel": max_rel,
        "mean_latency_us": mean_latency,
        "rows": rows,
    }
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kernel", default="kernel.py")
    parser.add_argument("--mode", choices=("correctness", "profile"), default="correctness")
    parser.add_argument("--m-values", default="all")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--rep", type=int, default=15)
    parser.add_argument("--json-out")
    args = parser.parse_args()

    result = evaluate(Path(args.kernel).resolve(), _parse_m_values(args.m_values), args.warmup, args.rep)
    text = json.dumps(result, indent=2)
    print(text)
    if args.json_out:
        Path(args.json_out).write_text(text + "\n")
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
