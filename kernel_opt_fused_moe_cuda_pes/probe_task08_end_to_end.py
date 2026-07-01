#!/usr/bin/env python3
"""Run the task08 packed-NVFP4 MoE pipeline end-to-end against FlashInfer."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from pathlib import Path
from types import SimpleNamespace

import torch

import kernel
from probe_task08_gemm2_runtime import (
    ATREX_GEMM2_TILEGRID_SPLITCOL_POSTSYNC_SYMBOL,
    _compile_gemm2_extension,
    _prepare_gemm1_tree,
    _prepare_gemm2_tree,
)
from probe_task08_mpad_compile import DEFAULT_TASK08_ROOT, V310_SYMBOL, _compile_extension
from test_kernel import _call_flashinfer, _make_real_nvfp4_flashinfer_inputs
from workload_shapes import get_shape


def _bench(fn, warmup: int, rep: int) -> float:
    if rep <= 0:
        return float("nan")
    holder = []
    for _ in range(warmup):
        holder[:] = [fn()]
    torch.cuda.synchronize()
    times = []
    for _ in range(rep):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        holder[:] = [fn()]
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end) * 1000.0)
    return float(statistics.median(times))


def _trace(args: argparse.Namespace, message: str) -> None:
    if args.trace:
        print(f"[trace] {message}", flush=True)


def _first(value):
    return value[0] if isinstance(value, (list, tuple)) else value


def _as_u8(tensor: torch.Tensor) -> torch.Tensor:
    return tensor if tensor.dtype == torch.uint8 else tensor.view(torch.uint8)


def _select_tile_tokens_dim(tokens: int, top_k: int, local_num_experts: int) -> int:
    supported = (8, 16, 32, 64, 128, 256)
    avg_ceil = max(1, math.ceil((tokens * top_k) / local_num_experts))
    tile = 1 << (avg_ceil - 1).bit_length()
    for candidate in supported:
        if tile <= candidate:
            return candidate
    return supported[-1]


def _count_summary(expert_counts: torch.Tensor) -> dict[str, int | float]:
    counts_f = expert_counts.to(torch.float32)
    return {
        "min": int(expert_counts.min().item()),
        "max": int(expert_counts.max().item()),
        "mean": float(counts_f.mean().item()),
        "nonzero": int((expert_counts > 0).sum().item()),
        "total": int(expert_counts.sum().item()),
    }


def _errors(actual: torch.Tensor, expected: torch.Tensor) -> dict[str, float | bool]:
    actual_f = actual.float()
    expected_f = expected.float()
    diff = (actual_f - expected_f).abs()
    rel = diff / expected_f.abs().clamp_min(1e-3)
    return {
        "finite": bool(torch.isfinite(actual_f).all() and torch.isfinite(expected_f).all()),
        "max_abs": float(diff.max().item()),
        "mean_abs": float(diff.mean().item()),
        "max_rel": float(rel.max().item()),
        "mean_rel": float(rel.mean().item()),
    }


def _abs_stats(tensor: torch.Tensor) -> dict[str, float | bool]:
    values = tensor.float().abs()
    return {
        "finite": bool(torch.isfinite(values).all()),
        "max": float(values.max().item()),
        "mean": float(values.mean().item()),
    }


def _gemm2_row_group_abs_stats(
    tensor: torch.Tensor,
    expert_counts: torch.Tensor,
    padded_rows: int,
) -> dict[str, dict[str, float | int | bool]]:
    values = tensor.float().abs()
    rows = values.shape[0]
    hidden = values.shape[1]
    local_num_experts = int(expert_counts.numel())
    if rows != local_num_experts * int(padded_rows):
        return {
            "valid_rows": {"finite": False, "rows": 0, "max": float("nan"), "mean": float("nan")},
            "padding_rows": {"finite": False, "rows": 0, "max": float("nan"), "mean": float("nan")},
        }

    row_max = values.amax(dim=1)
    row_sum = values.sum(dim=1)
    row_idx = torch.arange(int(padded_rows), device=tensor.device, dtype=expert_counts.dtype)
    valid_mask = (row_idx.unsqueeze(0) < expert_counts.unsqueeze(1)).reshape(-1)

    def summarize(mask: torch.Tensor) -> dict[str, float | int | bool]:
        row_count = int(mask.sum().item())
        if row_count == 0:
            return {"finite": True, "rows": 0, "max": 0.0, "mean": 0.0}
        selected_max = row_max[mask]
        selected_sum = row_sum[mask]
        return {
            "finite": bool(torch.isfinite(selected_max).all() and torch.isfinite(selected_sum).all()),
            "rows": row_count,
            "max": float(selected_max.max().item()),
            "mean": float((selected_sum.sum() / (row_count * hidden)).item()),
        }

    return {
        "valid_rows": summarize(valid_mask),
        "padding_rows": summarize(~valid_mask),
    }


def _make_inputs(args: argparse.Namespace, tokens: int) -> tuple:
    ns = SimpleNamespace(
        preset=args.preset,
        hidden_size=None,
        intermediate_size=None,
        local_num_experts=None,
        local_expert_offset=0,
        seed=args.seed,
    )
    return _make_real_nvfp4_flashinfer_inputs(ns, tokens)


def _build_routing_and_hidden_pack(args: argparse.Namespace, inputs: tuple) -> dict:
    routing_logits = inputs[0]
    hidden_states = inputs[2]
    hidden_states_scale = inputs[3]
    gemm1_weights = inputs[4]
    gemm1_weights_scale = inputs[5]
    gemm2_weights = inputs[10]
    gemm2_weights_scale = inputs[11]
    num_experts = int(inputs[16])
    top_k = int(inputs[17])
    intermediate_size = int(inputs[20])
    local_expert_offset = int(inputs[21])
    local_num_experts = int(inputs[22])
    tokens = int(routing_logits.size(0))
    hidden_size = int(hidden_states.size(1) * 2)
    tile_tokens_dim = int(
        args.tile_tokens_dim
        if args.tile_tokens_dim is not None
        else _select_tile_tokens_dim(tokens, top_k, local_num_experts)
    )

    ext = kernel._load_ext()
    topk_packed = ext.routing_pack_type1(routing_logits.contiguous(), top_k, 1.0)
    (
        expanded,
        _permuted_to_token,
        _cta_batch,
        _cta_mn_limit,
        _num_ctas,
        _total_padded,
        expert_counts,
        expert_padded_offsets,
    ) = kernel.routing_metadata_from_packed(
        topk_packed,
        num_experts=num_experts,
        local_expert_offset=local_expert_offset,
        local_num_experts=local_num_experts,
        tile_tokens_dim=tile_tokens_dim,
    )
    torch.cuda.synchronize()
    max_count = int(expert_counts.max().item())
    if max_count > args.mpad:
        raise ValueError(f"max expert count {max_count} exceeds requested Mpad={args.mpad}")

    hidden_bmm, hidden_scale_swizzled = kernel.pack_hidden_bmm_swizzled_from_metadata(
        topk_packed,
        expanded,
        expert_padded_offsets,
        hidden_states,
        hidden_states_scale,
        num_experts=num_experts,
        local_expert_offset=local_expert_offset,
        local_num_experts=local_num_experts,
        padded_rows=args.mpad,
    )
    affine_expert_offsets = (
        torch.arange(local_num_experts + 1, device=hidden_states.device, dtype=torch.int32)
        * int(args.mpad)
    ).contiguous()
    return {
        "tokens": tokens,
        "hidden_size": hidden_size,
        "intermediate_size": intermediate_size,
        "num_experts": num_experts,
        "top_k": top_k,
        "local_expert_offset": local_expert_offset,
        "local_num_experts": local_num_experts,
        "tile_tokens_dim": tile_tokens_dim,
        "topk_packed": topk_packed,
        "expanded": expanded,
        "expert_counts": expert_counts,
        "expert_padded_offsets": expert_padded_offsets,
        "hidden_bmm": hidden_bmm,
        "hidden_scale_swizzled": hidden_scale_swizzled,
        "a_fp4": hidden_bmm.reshape(local_num_experts * int(args.mpad), -1).contiguous(),
        "a_scale_swizzled_u8": _as_u8(hidden_scale_swizzled)
        .reshape(local_num_experts, -1)
        .contiguous(),
        "w13_fp4": gemm1_weights.contiguous(),
        "w13_scale_swizzled_u8": _as_u8(gemm1_weights_scale)
        .reshape(local_num_experts, -1)
        .contiguous(),
        "w2_fp4": gemm2_weights.contiguous(),
        "w2_scale_swizzled_u8": _as_u8(gemm2_weights_scale)
        .reshape(local_num_experts, -1)
        .contiguous(),
        "affine_expert_offsets": affine_expert_offsets,
    }


def _run_task08_pipeline(case: dict, gemm1_fn, gemm2_fn, args: argparse.Namespace) -> dict:
    gemm1_out = gemm1_fn(
        case["a_fp4"],
        case["a_scale_swizzled_u8"],
        case["w13_fp4"],
        case["w13_scale_swizzled_u8"],
        case["affine_expert_offsets"],
    )
    mid_q, mid_scale, mid_scale_swizzled = kernel.swiglu_requant_from_bmm(
        gemm1_out,
        case["expert_counts"],
        padded_rows=int(args.mpad),
        intermediate_size=int(case["intermediate_size"]),
    )
    mid_q_flat = mid_q.reshape(int(case["local_num_experts"]) * int(args.mpad), -1)
    gemm2_out = gemm2_fn(
        mid_q_flat.contiguous(),
        mid_scale_swizzled.view(torch.uint8).reshape(int(case["local_num_experts"]), -1),
        case["w2_fp4"],
        case["w2_scale_swizzled_u8"],
        case["affine_expert_offsets"],
    )
    out = kernel.final_scatter_from_bmm(
        gemm2_out,
        case["topk_packed"],
        case["expanded"],
        case["expert_padded_offsets"],
        local_expert_offset=int(case["local_expert_offset"]),
        padded_rows=int(args.mpad),
        use_prepared_output_layout=not args.no_prepared_output_layout,
    )
    return {
        "out": out,
        "gemm1_out": gemm1_out,
        "mid_q": mid_q,
        "mid_scale": mid_scale,
        "mid_scale_swizzled": mid_scale_swizzled,
        "gemm2_out": gemm2_out,
    }


def _compile_task08(args: argparse.Namespace):
    task08_root = Path(args.task08_root).resolve()
    gemm1_root, gemm1_patch_counts = _prepare_gemm1_tree(task08_root, args.mpad)
    gemm1_module = _compile_extension(gemm1_root, args.mpad, args.verbose_build)
    gemm2_root, gemm2_patch_counts = _prepare_gemm2_tree(task08_root, args.mpad)
    gemm2_module = _compile_gemm2_extension(gemm2_root, args.mpad, args.verbose_build)
    if not hasattr(gemm1_module, V310_SYMBOL):
        raise RuntimeError(f"compiled GEMM1 module missing symbol {V310_SYMBOL}")
    if not hasattr(gemm2_module, args.gemm2_symbol):
        raise RuntimeError(f"compiled GEMM2 module missing symbol {args.gemm2_symbol}")
    return (
        getattr(gemm1_module, V310_SYMBOL),
        getattr(gemm2_module, args.gemm2_symbol),
        {
            "task08_root": str(task08_root),
            "work_roots": {"gemm1": str(gemm1_root), "gemm2": str(gemm2_root)},
            "compiled_symbol": {"gemm1": V310_SYMBOL, "gemm2": args.gemm2_symbol},
            "patch_counts": {"gemm1": gemm1_patch_counts, "gemm2": gemm2_patch_counts},
        },
    )


def run_probe(args: argparse.Namespace) -> dict:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA device is required")
    started = time.perf_counter()
    shape = get_shape(args.preset)
    if shape["dtype"] != "nvfp4":
        raise NotImplementedError("task08 end-to-end probe targets packed NVFP4")
    tokens = int(args.tokens if args.tokens is not None else shape["tokens"][0])

    _trace(args, "compile task08 GEMM1/GEMM2 extensions")
    gemm1_fn, gemm2_fn, compile_info = _compile_task08(args)

    _trace(args, "build real FlashInfer packed-NVFP4 inputs")
    inputs = _make_inputs(args, tokens)
    _trace(args, "build routing metadata and hidden BMM pack")
    case = _build_routing_and_hidden_pack(args, inputs)

    _trace(args, "run candidate pipeline")
    candidate = _run_task08_pipeline(case, gemm1_fn, gemm2_fn, args)
    torch.cuda.synchronize()
    gemm2_abs = _abs_stats(candidate["gemm2_out"])
    gemm2_row_group_abs = _gemm2_row_group_abs_stats(
        candidate["gemm2_out"], case["expert_counts"], int(args.mpad)
    )

    _trace(args, "run FlashInfer reference")
    import flashinfer

    flash_fn = getattr(flashinfer, "trtllm_fp4_block_scale_moe", None)
    if flash_fn is None:
        raise RuntimeError("flashinfer.trtllm_fp4_block_scale_moe is not available")
    flash_out = _first(_call_flashinfer(flash_fn, inputs))
    torch.cuda.synchronize()

    errors = _errors(candidate["out"], flash_out)
    correctness_ok = bool(
        errors["finite"]
        and (errors["max_abs"] <= args.max_abs or errors["max_rel"] <= args.max_rel)
    )
    direct_layout_errors = None
    if args.compare_direct_layout:
        direct_out = kernel.final_scatter_from_bmm(
            candidate["gemm2_out"],
            case["topk_packed"],
            case["expanded"],
            case["expert_padded_offsets"],
            local_expert_offset=int(case["local_expert_offset"]),
            padded_rows=int(args.mpad),
            use_prepared_output_layout=False,
        )
        torch.cuda.synchronize()
        direct_layout_errors = _errors(direct_out, flash_out)

    stage_us = {}
    candidate_total_us = float("nan")
    flashinfer_us = float("nan")
    speedup = None
    target_status = "SKIPPED"
    if not args.skip_bench:
        _trace(args, "benchmark candidate total pipeline")

        def candidate_total():
            local_case = _build_routing_and_hidden_pack(args, inputs)
            return _run_task08_pipeline(local_case, gemm1_fn, gemm2_fn, args)["out"]

        candidate_total_us = _bench(candidate_total, args.warmup, args.rep)

        _trace(args, "benchmark stage latencies")
        gemm1_out = candidate["gemm1_out"]
        mid_q = candidate["mid_q"]
        mid_scale_swizzled = candidate["mid_scale_swizzled"]
        gemm2_out = candidate["gemm2_out"]
        mid_q_flat = mid_q.reshape(int(case["local_num_experts"]) * int(args.mpad), -1)
        stage_us = {
            "gemm1_bmm": _bench(
                lambda: gemm1_fn(
                    case["a_fp4"],
                    case["a_scale_swizzled_u8"],
                    case["w13_fp4"],
                    case["w13_scale_swizzled_u8"],
                    case["affine_expert_offsets"],
                ),
                args.warmup,
                args.rep,
            ),
            "swiglu_requant": _bench(
                lambda: kernel.swiglu_requant_from_bmm(
                    gemm1_out,
                    case["expert_counts"],
                    padded_rows=int(args.mpad),
                    intermediate_size=int(case["intermediate_size"]),
                ),
                args.warmup,
                args.rep,
            ),
            "gemm2_bmm": _bench(
                lambda: gemm2_fn(
                    mid_q_flat,
                    mid_scale_swizzled.view(torch.uint8).reshape(
                        int(case["local_num_experts"]), -1
                    ),
                    case["w2_fp4"],
                    case["w2_scale_swizzled_u8"],
                    case["affine_expert_offsets"],
                ),
                args.warmup,
                args.rep,
            ),
            "final_scatter": _bench(
                lambda: kernel.final_scatter_from_bmm(
                    gemm2_out,
                    case["topk_packed"],
                    case["expanded"],
                    case["expert_padded_offsets"],
                    local_expert_offset=int(case["local_expert_offset"]),
                    padded_rows=int(args.mpad),
                    use_prepared_output_layout=not args.no_prepared_output_layout,
                ),
                args.warmup,
                args.rep,
            ),
        }
        if not args.skip_flashinfer_bench:
            _trace(args, "benchmark FlashInfer baseline")
            flashinfer_us = _bench(lambda: _call_flashinfer(flash_fn, inputs), args.warmup, args.rep)
            speedup = flashinfer_us / candidate_total_us if candidate_total_us > 0 else None
            target_status = (
                "MET"
                if correctness_ok and speedup is not None and speedup > args.target_speedup
                else "MISS"
            )
        else:
            target_status = "NO_BASELINE"

    status = "PASS" if correctness_ok else "FAIL"
    if args.require_speedup and target_status == "MISS":
        status = "FAIL"
    if not correctness_ok:
        next_step = "Fix GEMM2 BMM correctness/stability before performance tuning."
    elif target_status == "MISS":
        next_step = "Optimize slowest candidate stage until speedup_vs_flashinfer exceeds target."
    else:
        next_step = "Integrate the verified pipeline into the main candidate entrypoint."
    return {
        "status": status,
        "target_status": target_status,
        "operator": "task08 packed-NVFP4 MoE end-to-end candidate",
        "correctness_oracle": "flashinfer.trtllm_fp4_block_scale_moe",
        "performance_baseline": "flashinfer.trtllm_fp4_block_scale_moe",
        "preset": args.preset,
        "tokens": tokens,
        **compile_info,
        "shape": {
            "hidden_size": int(case["hidden_size"]),
            "intermediate_size": int(case["intermediate_size"]),
            "num_experts": int(case["num_experts"]),
            "local_num_experts": int(case["local_num_experts"]),
            "top_k": int(case["top_k"]),
            "tile_tokens_dim": int(case["tile_tokens_dim"]),
            "padded_rows": int(args.mpad),
            "hidden_states": tuple(inputs[2].shape),
            "hidden_states_scale": tuple(inputs[3].shape),
            "a_fp4": tuple(case["a_fp4"].shape),
            "a_scale_swizzled_u8": tuple(case["a_scale_swizzled_u8"].shape),
            "gemm1_out": tuple(candidate["gemm1_out"].shape),
            "mid_q": tuple(candidate["mid_q"].shape),
            "mid_scale": tuple(candidate["mid_scale"].shape),
            "mid_scale_swizzled": tuple(candidate["mid_scale_swizzled"].shape),
            "gemm2_out": tuple(candidate["gemm2_out"].shape),
            "out": tuple(candidate["out"].shape),
        },
        "expert_counts": _count_summary(case["expert_counts"]),
        "checks": {
            "correctness": "PASS" if correctness_ok else "FAIL",
            "prepared_output_layout": not args.no_prepared_output_layout,
            "errors_vs_flashinfer": errors,
            "direct_layout_errors_vs_flashinfer": direct_layout_errors,
            "gemm2_out_abs": gemm2_abs,
            "gemm2_row_group_abs": gemm2_row_group_abs,
            "thresholds": {"max_abs": args.max_abs, "max_rel": args.max_rel},
        },
        "latency_us": {
            "candidate_total": candidate_total_us,
            "flashinfer_trtllm_fp4_block_scale_moe": flashinfer_us,
            "speedup_vs_flashinfer": speedup,
            "target_speedup": args.target_speedup,
            "stages": stage_us,
        },
        "elapsed_s": time.perf_counter() - started,
        "next_step": next_step,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preset", choices=("g11", "g8"), default="g11")
    parser.add_argument("--tokens", type=int)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--task08-root", default=DEFAULT_TASK08_ROOT)
    parser.add_argument("--mpad", type=int, default=238)
    parser.add_argument("--tile-tokens-dim", type=int)
    parser.add_argument("--gemm2-symbol", default=ATREX_GEMM2_TILEGRID_SPLITCOL_POSTSYNC_SYMBOL)
    parser.add_argument("--max-abs", type=float, default=0.75)
    parser.add_argument("--max-rel", type=float, default=8.0)
    parser.add_argument("--target-speedup", type=float, default=1.0)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--rep", type=int, default=9)
    parser.add_argument("--skip-bench", action="store_true")
    parser.add_argument("--skip-flashinfer-bench", action="store_true")
    parser.add_argument("--require-speedup", action="store_true")
    parser.add_argument("--compare-direct-layout", action="store_true")
    parser.add_argument("--no-prepared-output-layout", action="store_true")
    parser.add_argument("--verbose-build", action="store_true")
    parser.add_argument("--trace", action="store_true")
    parser.add_argument("--json-out")
    args = parser.parse_args()
    result = run_probe(args)
    text = json.dumps(result, indent=2)
    print(text)
    if args.json_out:
        Path(args.json_out).write_text(text + "\n")
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
