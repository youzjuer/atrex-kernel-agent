#!/usr/bin/env python3
"""Localize GEMM2 epilogue corruption against the stable postsync path."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

import torch

import kernel
from probe_task08_end_to_end import _build_routing_and_hidden_pack, _make_inputs
from probe_task08_gemm2_runtime import (
    ATREX_GEMM2_TILEGRID_SPLITCOL_POSTSYNC_SYMBOL,
    ATREX_GEMM2_TILEGRID_SPLITCOL_WARPSHUFFLE_POSTLDSYNC_SYMBOL,
    DIAG_GEMM2_SYMBOLS,
    _bench,
    _compile_gemm2_extension,
    _prepare_gemm1_tree,
    _prepare_gemm2_tree,
)
from probe_task08_mpad_compile import DEFAULT_TASK08_ROOT, V310_SYMBOL, _compile_extension
from workload_shapes import get_shape


def _trace(args: argparse.Namespace, message: str) -> None:
    if args.trace:
        print(f"[trace] {message}", flush=True)


def _new_state(thresholds: list[float], topk: int, bad_threshold: float) -> dict[str, Any]:
    return {
        "numel": 0,
        "rows": 0,
        "sum_abs": 0.0,
        "max_abs": 0.0,
        "threshold_counts": {f"{threshold:g}": 0 for threshold in thresholds},
        "top": [],
        "topk": int(topk),
        "bad_threshold": float(bad_threshold),
        "first_bad": None,
    }


def _sample_record(
    *,
    row_flat: int,
    col: int,
    diff: float,
    reference: float,
    candidate: float,
    padded_rows: int,
    expert_counts_cpu: list[int],
) -> dict[str, int | float | bool]:
    local_expert = row_flat // padded_rows
    row_in_expert = row_flat - local_expert * padded_rows
    expert_count = int(expert_counts_cpu[local_expert])
    return {
        "row_flat": int(row_flat),
        "local_expert": int(local_expert),
        "row_in_expert": int(row_in_expert),
        "expert_count": expert_count,
        "valid_row": bool(row_in_expert < expert_count),
        "col": int(col),
        "col_mod_16": int(col % 16),
        "col_mod_32": int(col % 32),
        "col_mod_64": int(col % 64),
        "col_mod_128": int(col % 128),
        "col_group_16": int(col // 16),
        "col_group_32": int(col // 32),
        "col_group_64": int(col // 64),
        "col_group_128": int(col // 128),
        "abs_diff": float(diff),
        "reference": float(reference),
        "candidate": float(candidate),
    }


def _update_state(
    state: dict[str, Any],
    *,
    diff: torch.Tensor,
    reference: torch.Tensor,
    candidate: torch.Tensor,
    rows: torch.Tensor,
    thresholds: list[float],
    padded_rows: int,
    expert_counts_cpu: list[int],
) -> None:
    if diff.numel() == 0:
        return
    cols = int(diff.shape[1])
    flat = diff.reshape(-1)
    state["numel"] += int(flat.numel())
    state["rows"] += int(diff.shape[0])
    state["sum_abs"] += float(flat.sum().item())
    state["max_abs"] = max(float(state["max_abs"]), float(flat.max().item()))
    for threshold in thresholds:
        state["threshold_counts"][f"{threshold:g}"] += int((flat > threshold).sum().item())

    bad_threshold = float(state["bad_threshold"])
    if state["first_bad"] is None:
        bad_mask = flat > bad_threshold
        if bool(bad_mask.any().item()):
            bad_idx = int(torch.nonzero(bad_mask, as_tuple=False)[0].item())
            row_pos = bad_idx // cols
            col = bad_idx - row_pos * cols
            row_flat = int(rows[row_pos].item())
            state["first_bad"] = _sample_record(
                row_flat=row_flat,
                col=col,
                diff=float(flat[bad_idx].item()),
                reference=float(reference.reshape(-1)[bad_idx].float().item()),
                candidate=float(candidate.reshape(-1)[bad_idx].float().item()),
                padded_rows=padded_rows,
                expert_counts_cpu=expert_counts_cpu,
            )

    topk = int(state["topk"])
    if topk <= 0:
        return
    k = min(topk, int(flat.numel()))
    values, indices = torch.topk(flat, k)
    ref_flat = reference.reshape(-1)
    cand_flat = candidate.reshape(-1)
    additions = []
    for value, index in zip(values.detach().cpu().tolist(), indices.detach().cpu().tolist()):
        if value <= 0.0:
            continue
        row_pos = int(index) // cols
        col = int(index) - row_pos * cols
        row_flat = int(rows[row_pos].item())
        additions.append(
            _sample_record(
                row_flat=row_flat,
                col=col,
                diff=float(value),
                reference=float(ref_flat[int(index)].float().item()),
                candidate=float(cand_flat[int(index)].float().item()),
                padded_rows=padded_rows,
                expert_counts_cpu=expert_counts_cpu,
            )
        )
    if additions:
        merged = list(state["top"]) + additions
        merged.sort(key=lambda item: float(item["abs_diff"]), reverse=True)
        state["top"] = merged[:topk]


def _finalize_state(state: dict[str, Any]) -> dict[str, Any]:
    numel = int(state["numel"])
    return {
        "rows": int(state["rows"]),
        "numel": numel,
        "max_abs": float(state["max_abs"]) if numel else 0.0,
        "mean_abs": float(state["sum_abs"] / numel) if numel else 0.0,
        "threshold_counts": dict(state["threshold_counts"]),
        "bad_threshold": float(state["bad_threshold"]),
        "first_bad": state["first_bad"],
        "top": list(state["top"]),
    }


@torch.no_grad()
def _summarize_diff(
    *,
    reference: torch.Tensor,
    candidate: torch.Tensor,
    expert_counts: torch.Tensor,
    padded_rows: int,
    chunk_rows: int,
    thresholds: list[float],
    topk: int,
    bad_threshold: float,
    row_filter: str,
) -> dict[str, Any]:
    if reference.shape != candidate.shape:
        raise ValueError(f"shape mismatch: reference={tuple(reference.shape)} candidate={tuple(candidate.shape)}")
    if reference.dtype != torch.bfloat16 or candidate.dtype != torch.bfloat16:
        raise TypeError("GEMM2 diff expects bf16 tensors")
    rows_total = int(reference.shape[0])
    cols = int(reference.shape[1])
    expert_counts_device = expert_counts.to(device=reference.device)
    expert_counts_cpu = [int(v) for v in expert_counts.detach().cpu().tolist()]
    state = _new_state(thresholds, topk, bad_threshold)
    for start in range(0, rows_total, chunk_rows):
        end = min(rows_total, start + chunk_rows)
        row_ids = torch.arange(start, end, device=reference.device, dtype=torch.long)
        local_experts = torch.div(row_ids, int(padded_rows), rounding_mode="floor")
        row_in_expert = row_ids - local_experts * int(padded_rows)
        valid_rows = row_in_expert < expert_counts_device.index_select(0, local_experts)
        if row_filter == "valid":
            selected = valid_rows
        elif row_filter == "padding":
            selected = ~valid_rows
        elif row_filter == "all":
            selected = torch.ones_like(valid_rows, dtype=torch.bool)
        else:
            raise ValueError(f"invalid row_filter={row_filter}")
        if not bool(selected.any().item()):
            continue
        ref_chunk = reference[start:end].index_select(0, torch.nonzero(selected, as_tuple=False).flatten())
        cand_chunk = candidate[start:end].index_select(0, torch.nonzero(selected, as_tuple=False).flatten())
        selected_rows = row_ids.index_select(0, torch.nonzero(selected, as_tuple=False).flatten())
        diff = (cand_chunk.float() - ref_chunk.float()).abs()
        if diff.shape[1] != cols:
            raise RuntimeError("internal diff column mismatch")
        _update_state(
            state,
            diff=diff,
            reference=ref_chunk,
            candidate=cand_chunk,
            rows=selected_rows,
            thresholds=thresholds,
            padded_rows=padded_rows,
            expert_counts_cpu=expert_counts_cpu,
        )
    return _finalize_state(state)


def _run_gemm2(gemm2_fn, case: dict, mid_q_flat: torch.Tensor, mid_scale_swizzled: torch.Tensor):
    return gemm2_fn(
        mid_q_flat.contiguous(),
        mid_scale_swizzled.view(torch.uint8).reshape(int(case["local_num_experts"]), -1),
        case["w2_fp4"],
        case["w2_scale_swizzled_u8"],
        case["affine_expert_offsets"],
    )


def _run_swiglu(case: dict, gemm1_out: torch.Tensor, args: argparse.Namespace):
    if args.legacy_swiglu_requant:
        return kernel.swiglu_requant_from_bmm(
            gemm1_out,
            case["expert_counts"],
            padded_rows=int(args.mpad),
            intermediate_size=int(case["intermediate_size"]),
        )
    return kernel.swiglu_requant_from_bmm_metadata(
        gemm1_out,
        case["topk_packed"],
        case["expanded"],
        case["expert_padded_offsets"],
        local_expert_offset=int(case["local_expert_offset"]),
        padded_rows=int(args.mpad),
        intermediate_size=int(case["intermediate_size"]),
    )


def run_probe(args: argparse.Namespace) -> dict:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA device is required")
    started = time.perf_counter()
    shape = get_shape(args.preset)
    if shape["dtype"] != "nvfp4":
        raise NotImplementedError("task08 GEMM2 diff targets packed NVFP4")
    tokens = int(args.tokens if args.tokens is not None else shape["tokens"][0])
    thresholds = sorted(set(float(v) for v in args.threshold))
    if args.max_abs not in thresholds:
        thresholds.append(float(args.max_abs))
        thresholds.sort()

    task08_root = Path(args.task08_root).resolve()
    _trace(args, "compile GEMM1 extension")
    gemm1_root, gemm1_patch_counts = _prepare_gemm1_tree(task08_root, args.mpad)
    gemm1_module = _compile_extension(gemm1_root, args.mpad, args.verbose_build)
    gemm1_fn = getattr(gemm1_module, V310_SYMBOL)

    reference_is_diag = args.reference_symbol in DIAG_GEMM2_SYMBOLS
    candidate_is_diag = args.candidate_symbol in DIAG_GEMM2_SYMBOLS
    _trace(args, "compile reference GEMM2 extension")
    reference_gemm2_root, reference_gemm2_patch_counts = _prepare_gemm2_tree(
        task08_root,
        args.mpad,
        include_diag_symbols=reference_is_diag,
    )
    reference_gemm2_module = _compile_gemm2_extension(
        reference_gemm2_root,
        args.mpad,
        args.verbose_build,
        extension_tag="diag" if reference_is_diag else "",
    )
    if reference_is_diag == candidate_is_diag:
        candidate_gemm2_root = reference_gemm2_root
        candidate_gemm2_patch_counts = reference_gemm2_patch_counts
        candidate_gemm2_module = reference_gemm2_module
    else:
        _trace(args, "compile candidate GEMM2 extension")
        candidate_gemm2_root, candidate_gemm2_patch_counts = _prepare_gemm2_tree(
            task08_root,
            args.mpad,
            include_diag_symbols=candidate_is_diag,
        )
        candidate_gemm2_module = _compile_gemm2_extension(
            candidate_gemm2_root,
            args.mpad,
            args.verbose_build,
            extension_tag="diag" if candidate_is_diag else "",
        )
    if not hasattr(reference_gemm2_module, args.reference_symbol):
        raise RuntimeError(f"compiled reference GEMM2 module missing symbol {args.reference_symbol}")
    if not hasattr(candidate_gemm2_module, args.candidate_symbol):
        raise RuntimeError(f"compiled candidate GEMM2 module missing symbol {args.candidate_symbol}")
    reference_fn = getattr(reference_gemm2_module, args.reference_symbol)
    candidate_fn = getattr(candidate_gemm2_module, args.candidate_symbol)

    _trace(args, "build real G11 case")
    inputs = _make_inputs(args, tokens)
    case = _build_routing_and_hidden_pack(args, inputs)
    _trace(args, "run GEMM1 and SwiGLU requant")
    gemm1_out = gemm1_fn(
        case["a_fp4"],
        case["a_scale_swizzled_u8"],
        case["w13_fp4"],
        case["w13_scale_swizzled_u8"],
        case["affine_expert_offsets"],
    )
    torch.cuda.synchronize()
    mid_q, mid_scale, mid_scale_swizzled = _run_swiglu(case, gemm1_out, args)
    torch.cuda.synchronize()
    mid_q_flat = mid_q.reshape(int(case["local_num_experts"]) * int(args.mpad), -1)

    if args.candidate_first:
        _trace(args, f"run candidate GEMM2 first: {args.candidate_symbol}")
        candidate_first = _run_gemm2(candidate_fn, case, mid_q_flat, mid_scale_swizzled)
        torch.cuda.synchronize()
        _trace(args, f"run reference GEMM2 second: {args.reference_symbol}")
        reference_out = _run_gemm2(reference_fn, case, mid_q_flat, mid_scale_swizzled)
        torch.cuda.synchronize()
    else:
        _trace(args, f"run reference GEMM2 first: {args.reference_symbol}")
        reference_out = _run_gemm2(reference_fn, case, mid_q_flat, mid_scale_swizzled)
        torch.cuda.synchronize()
        _trace(args, f"run candidate GEMM2 second: {args.candidate_symbol}")
        candidate_first = _run_gemm2(candidate_fn, case, mid_q_flat, mid_scale_swizzled)
        torch.cuda.synchronize()

    _trace(args, "summarize candidate vs reference diff")
    valid_diff = _summarize_diff(
        reference=reference_out,
        candidate=candidate_first,
        expert_counts=case["expert_counts"],
        padded_rows=int(args.mpad),
        chunk_rows=int(args.chunk_rows),
        thresholds=thresholds,
        topk=int(args.topk),
        bad_threshold=float(args.bad_threshold),
        row_filter="valid",
    )
    padding_diff = _summarize_diff(
        reference=reference_out,
        candidate=candidate_first,
        expert_counts=case["expert_counts"],
        padded_rows=int(args.mpad),
        chunk_rows=int(args.chunk_rows),
        thresholds=thresholds,
        topk=min(int(args.topk), 8),
        bad_threshold=float(args.bad_threshold),
        row_filter="padding",
    )

    stability = []
    for run_idx in range(2, int(args.candidate_runs) + 1):
        _trace(args, f"run candidate GEMM2 repeat {run_idx}")
        candidate_next = _run_gemm2(candidate_fn, case, mid_q_flat, mid_scale_swizzled)
        torch.cuda.synchronize()
        stability.append(
            {
                "run": run_idx,
                "valid_diff_vs_candidate_run1": _summarize_diff(
                    reference=candidate_first,
                    candidate=candidate_next,
                    expert_counts=case["expert_counts"],
                    padded_rows=int(args.mpad),
                    chunk_rows=int(args.chunk_rows),
                    thresholds=thresholds,
                    topk=min(int(args.topk), 8),
                    bad_threshold=float(args.bad_threshold),
                    row_filter="valid",
                ),
            }
        )
        del candidate_next

    latency_us = {}
    if not args.skip_bench:
        _trace(args, "benchmark GEMM2 reference and candidate")
        latency_us = {
            "reference": _bench(
                lambda: _run_gemm2(reference_fn, case, mid_q_flat, mid_scale_swizzled),
                args.warmup,
                args.rep,
            ),
            "candidate": _bench(
                lambda: _run_gemm2(candidate_fn, case, mid_q_flat, mid_scale_swizzled),
                args.warmup,
                args.rep,
            ),
        }
        latency_us["candidate_vs_reference_speedup"] = (
            latency_us["reference"] / latency_us["candidate"]
            if latency_us["candidate"] > 0
            else None
        )

    candidate_matches_reference = bool(valid_diff["max_abs"] <= args.max_abs)
    candidate_deterministic = all(
        item["valid_diff_vs_candidate_run1"]["max_abs"] <= args.max_abs for item in stability
    )
    status = "PASS" if candidate_matches_reference and candidate_deterministic else "FAIL"
    return {
        "status": status,
        "operator": "task08 GEMM2 chunked diff diagnostic",
        "preset": args.preset,
        "tokens": tokens,
        "task08_root": str(task08_root),
        "work_roots": {
            "gemm1": str(gemm1_root),
            "reference_gemm2": str(reference_gemm2_root),
            "candidate_gemm2": str(candidate_gemm2_root),
        },
        "compiled_symbol": {
            "gemm1": V310_SYMBOL,
            "reference_gemm2": args.reference_symbol,
            "candidate_gemm2": args.candidate_symbol,
        },
        "patch_counts": {
            "gemm1": gemm1_patch_counts,
            "reference_gemm2": reference_gemm2_patch_counts,
            "candidate_gemm2": candidate_gemm2_patch_counts,
        },
        "shape": {
            "hidden_size": int(case["hidden_size"]),
            "intermediate_size": int(case["intermediate_size"]),
            "local_num_experts": int(case["local_num_experts"]),
            "top_k": int(case["top_k"]),
            "padded_rows": int(args.mpad),
            "mid_q_flat": tuple(mid_q_flat.shape),
            "mid_scale": tuple(mid_scale.shape),
            "mid_scale_swizzled": tuple(mid_scale_swizzled.shape),
            "gemm2_out": tuple(reference_out.shape),
        },
        "expert_counts": {
            "min": int(case["expert_counts"].min().item()),
            "max": int(case["expert_counts"].max().item()),
            "mean": float(case["expert_counts"].to(torch.float32).mean().item()),
            "total": int(case["expert_counts"].sum().item()),
        },
        "swiglu_requant_path": "legacy_count_dense" if args.legacy_swiglu_requant else "metadata_valid_slots",
        "run_order": "candidate_first" if args.candidate_first else "reference_first",
        "checks": {
            "candidate_matches_reference": candidate_matches_reference,
            "candidate_deterministic": candidate_deterministic,
            "max_abs_threshold": float(args.max_abs),
            "thresholds": thresholds,
            "valid_rows_diff": valid_diff,
            "padding_rows_diff": padding_diff,
            "candidate_repeat_stability": stability,
        },
        "latency_us": latency_us,
        "elapsed_s": time.perf_counter() - started,
        "next_step": (
            "Promote candidate only after repeated end-to-end FlashInfer PASS."
            if status == "PASS"
            else "Use top bad row/column coordinates to inspect the GEMM2 warp-shuffle epilogue."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task08-root", default=str(DEFAULT_TASK08_ROOT))
    parser.add_argument("--preset", choices=("g11", "g8"), default="g11")
    parser.add_argument("--tokens", type=int)
    parser.add_argument("--mpad", type=int, default=238)
    parser.add_argument("--tile-tokens-dim", type=int)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--reference-symbol", default=ATREX_GEMM2_TILEGRID_SPLITCOL_POSTSYNC_SYMBOL)
    parser.add_argument(
        "--candidate-symbol",
        default=ATREX_GEMM2_TILEGRID_SPLITCOL_WARPSHUFFLE_POSTLDSYNC_SYMBOL,
    )
    parser.add_argument("--candidate-runs", type=int, default=2)
    parser.add_argument("--candidate-first", action="store_true")
    parser.add_argument("--chunk-rows", type=int, default=256)
    parser.add_argument("--topk", type=int, default=32)
    parser.add_argument("--threshold", type=float, action="append", default=[0.0, 0.75, 8.0, 64.0])
    parser.add_argument("--bad-threshold", type=float, default=0.75)
    parser.add_argument("--max-abs", type=float, default=0.0)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--rep", type=int, default=3)
    parser.add_argument("--skip-bench", action="store_true")
    parser.add_argument("--legacy-swiglu-requant", action="store_true")
    parser.add_argument("--trace", action="store_true")
    parser.add_argument("--verbose-build", action="store_true")
    parser.add_argument("--json-out")
    args = parser.parse_args()

    try:
        result = run_probe(args)
    except Exception as exc:  # noqa: BLE001
        result = {
            "status": "FAIL",
            "operator": "task08 GEMM2 chunked diff diagnostic",
            "preset": args.preset,
            "mpad": int(args.mpad),
            "reference_symbol": args.reference_symbol,
            "candidate_symbol": args.candidate_symbol,
            "error": f"{type(exc).__name__}: {exc}",
        }
    text = json.dumps(result, indent=2)
    print(text)
    if args.json_out:
        Path(args.json_out).write_text(text + "\n")
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
