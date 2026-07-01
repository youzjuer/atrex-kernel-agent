#!/usr/bin/env python3
"""Run the generated task08 Mpad BMM against true G11 packed-NVFP4 inputs."""

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
from probe_task08_mpad_compile import (
    DEFAULT_TASK08_ROOT,
    V310_SYMBOL,
    _compile_extension,
    _prepare_tree,
)
from test_kernel import _make_real_nvfp4_flashinfer_inputs
from workload_shapes import get_shape


def _bench(fn, warmup: int, rep: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(rep):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end) * 1000.0)
    return float(statistics.median(times))


def _select_tile_tokens_dim(tokens: int, top_k: int, local_num_experts: int) -> int:
    supported = (8, 16, 32, 64, 128, 256)
    avg_ceil = max(1, math.ceil((tokens * top_k) / local_num_experts))
    tile = 1 << (avg_ceil - 1).bit_length()
    for candidate in supported:
        if tile <= candidate:
            return candidate
    return supported[-1]


def _as_u8(tensor: torch.Tensor) -> torch.Tensor:
    return tensor if tensor.dtype == torch.uint8 else tensor.view(torch.uint8)


def _count_summary(expert_counts: torch.Tensor) -> dict[str, int | float]:
    counts = expert_counts.to(torch.float32)
    return {
        "min": int(expert_counts.min().item()),
        "max": int(expert_counts.max().item()),
        "mean": float(counts.mean().item()),
        "nonzero": int((expert_counts > 0).sum().item()),
        "total": int(expert_counts.sum().item()),
    }


def _valid_row_sample(
    expert_counts: torch.Tensor,
    *,
    padded_rows: int,
    limit: int,
    device: torch.device,
) -> torch.Tensor:
    rows: list[int] = []
    counts = expert_counts.detach().cpu().tolist()
    for expert, count in enumerate(counts):
        for row in range(min(int(count), padded_rows)):
            rows.append(expert * padded_rows + row)
            if len(rows) >= limit:
                return torch.tensor(rows, device=device, dtype=torch.long)
    return torch.tensor(rows, device=device, dtype=torch.long)


def _make_case(args: argparse.Namespace) -> dict:
    shape = get_shape(args.preset)
    if shape["dtype"] != "nvfp4":
        raise NotImplementedError("task08 runtime smoke targets packed NVFP4")
    tokens = int(args.tokens if args.tokens is not None else shape["tokens"][0])
    ns = SimpleNamespace(
        preset=args.preset,
        hidden_size=None,
        intermediate_size=None,
        local_num_experts=None,
        local_expert_offset=0,
        seed=args.seed,
    )
    inputs = _make_real_nvfp4_flashinfer_inputs(ns, tokens)
    routing_logits = inputs[0]
    hidden_states = inputs[2]
    hidden_states_scale = inputs[3]
    gemm1_weights = inputs[4]
    gemm1_weights_scale = inputs[5]
    gemm2_weights = inputs[10]
    gemm2_weights_scale = inputs[11]
    num_experts = int(inputs[16])
    top_k = int(inputs[17])
    local_expert_offset = int(inputs[21])
    local_num_experts = int(inputs[22])
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
        expert_offsets,
    ) = kernel.routing_metadata_from_packed(
        topk_packed,
        num_experts=num_experts,
        local_expert_offset=local_expert_offset,
        local_num_experts=local_num_experts,
        tile_tokens_dim=tile_tokens_dim,
    )
    torch.cuda.synchronize()
    padded_rows = int(args.mpad)
    max_count = int(expert_counts.max().item())
    if max_count > padded_rows:
        raise ValueError(f"max expert count {max_count} exceeds requested Mpad={padded_rows}")

    affine_offsets = (
        torch.arange(local_num_experts + 1, device=hidden_states.device, dtype=torch.int32)
        * padded_rows
    ).contiguous()
    hidden_bmm, hidden_scale_swizzled = kernel.pack_hidden_bmm_swizzled_from_metadata(
        topk_packed,
        expanded,
        expert_offsets,
        hidden_states,
        hidden_states_scale,
        num_experts=num_experts,
        local_expert_offset=local_expert_offset,
        local_num_experts=local_num_experts,
        padded_rows=padded_rows,
    )
    torch.cuda.synchronize()

    a_fp4 = hidden_bmm.reshape(local_num_experts * padded_rows, hidden_bmm.shape[-1])
    a_scale_swizzled_u8 = _as_u8(hidden_scale_swizzled).reshape(local_num_experts, -1)
    w13_scale_swizzled_u8 = _as_u8(gemm1_weights_scale).reshape(local_num_experts, -1)

    return {
        "tokens": tokens,
        "shape": shape,
        "topk_packed": topk_packed,
        "a_fp4": a_fp4.contiguous(),
        "a_scale_swizzled_u8": a_scale_swizzled_u8.contiguous(),
        "w13_fp4": gemm1_weights.contiguous(),
        "w13_scale_swizzled_u8": w13_scale_swizzled_u8.contiguous(),
        "w2_fp4": gemm2_weights.contiguous(),
        "w2_scale_swizzled_u8": _as_u8(gemm2_weights_scale).reshape(
            local_num_experts, -1
        ).contiguous(),
        "expert_offsets": affine_offsets,
        "num_experts": num_experts,
        "local_num_experts": local_num_experts,
        "top_k": top_k,
        "tile_tokens_dim": tile_tokens_dim,
        "padded_rows": padded_rows,
        "expert_counts": expert_counts,
    }


def run_probe(args: argparse.Namespace) -> dict:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA device is required")
    started = time.perf_counter()
    task08_root = Path(args.task08_root).resolve()
    work_root, patch_counts = _prepare_tree(task08_root, args.mpad)
    module = _compile_extension(work_root, args.mpad, args.verbose_build)
    if not hasattr(module, V310_SYMBOL):
        raise RuntimeError(f"compiled module missing symbol {V310_SYMBOL}")
    fn = getattr(module, V310_SYMBOL)

    case = _make_case(args)
    out = fn(
        case["a_fp4"],
        case["a_scale_swizzled_u8"],
        case["w13_fp4"],
        case["w13_scale_swizzled_u8"],
        case["expert_offsets"],
    )
    torch.cuda.synchronize()
    expected_out_shape = (
        int(case["local_num_experts"]) * int(case["padded_rows"]),
        int(case["w13_fp4"].shape[1]),
    )
    shape_ok = tuple(out.shape) == expected_out_shape and out.dtype == torch.bfloat16
    valid_rows = _valid_row_sample(
        case["expert_counts"],
        padded_rows=int(case["padded_rows"]),
        limit=args.check_valid_rows,
        device=out.device,
    )
    sample = out.index_select(0, valid_rows)[:, : min(64, out.shape[1])].float()
    finite_sample = int(torch.isfinite(sample).sum().item())
    sample_numel = int(sample.numel())

    task08_us = _bench(
        lambda: fn(
            case["a_fp4"],
            case["a_scale_swizzled_u8"],
            case["w13_fp4"],
            case["w13_scale_swizzled_u8"],
            case["expert_offsets"],
        ),
        args.warmup,
        args.rep,
    )
    elapsed = time.perf_counter() - started
    status = "PASS" if shape_ok and finite_sample == sample_numel else "FAIL"
    return {
        "status": status,
        "operator": "generated task08 SM103 FP4 GEMM1 BMM runtime smoke",
        "preset": args.preset,
        "tokens": int(case["tokens"]),
        "task08_root": str(task08_root),
        "work_root": str(work_root),
        "compiled_symbol": V310_SYMBOL,
        "patch_counts": patch_counts,
        "shape": {
            "hidden_size": int(case["shape"]["hidden_size"]),
            "intermediate_size": int(case["shape"]["intermediate_size"]),
            "num_experts": int(case["num_experts"]),
            "local_num_experts": int(case["local_num_experts"]),
            "top_k": int(case["top_k"]),
            "tile_tokens_dim": int(case["tile_tokens_dim"]),
            "padded_rows": int(case["padded_rows"]),
            "a_fp4": tuple(case["a_fp4"].shape),
            "a_scale_swizzled_u8": tuple(case["a_scale_swizzled_u8"].shape),
            "w13_fp4": tuple(case["w13_fp4"].shape),
            "w13_scale_swizzled_u8": tuple(case["w13_scale_swizzled_u8"].shape),
            "expert_offsets": tuple(case["expert_offsets"].shape),
            "out": tuple(out.shape),
            "out_dtype": str(out.dtype),
        },
        "expert_counts": _count_summary(case["expert_counts"]),
        "checks": {
            "out_shape_dtype": "PASS" if shape_ok else "FAIL",
            "finite_valid_rows_sample": "PASS" if finite_sample == sample_numel else "FAIL",
            "checked_valid_rows": int(valid_rows.numel()),
            "finite_sample_count": finite_sample,
            "finite_sample_numel": sample_numel,
        },
        "latency_us": {
            "task08_v310_gemm1_bmm_only": task08_us,
        },
        "elapsed_s": elapsed,
        "next_step": (
            "Add prepared-row-aware SwiGLU/requant epilogue and wire GEMM2/final scatter before "
            "using this as a full flashinfer.trtllm_fp4_block_scale_moe replacement."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task08-root", default=str(DEFAULT_TASK08_ROOT))
    parser.add_argument("--preset", default="g11")
    parser.add_argument("--tokens", type=int)
    parser.add_argument("--mpad", type=int, default=238)
    parser.add_argument("--tile-tokens-dim", type=int)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--rep", type=int, default=3)
    parser.add_argument("--check-valid-rows", type=int, default=4096)
    parser.add_argument("--verbose-build", action="store_true")
    parser.add_argument("--json-out")
    args = parser.parse_args()

    try:
        result = run_probe(args)
    except Exception as exc:  # noqa: BLE001
        result = {
            "status": "FAIL",
            "operator": "generated task08 SM103 FP4 GEMM1 BMM runtime smoke",
            "preset": args.preset,
            "mpad": int(args.mpad),
            "compiled_symbol": V310_SYMBOL,
            "error": f"{type(exc).__name__}: {exc}",
        }
    text = json.dumps(result, indent=2)
    print(text)
    if args.json_out:
        Path(args.json_out).write_text(text + "\n")
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
