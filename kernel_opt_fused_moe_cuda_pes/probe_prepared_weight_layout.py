#!/usr/bin/env python3
"""Validate the FlashInfer/TRT-LLM prepared FP4 MoE weight layout locally."""

from __future__ import annotations

import argparse
import gc
import json
import statistics
from pathlib import Path

import torch

import kernel
from workload_shapes import get_shape


SRC_TO_DST_BLK16_ROW_MAP = (
    0,
    8,
    1,
    9,
    2,
    10,
    3,
    11,
    4,
    12,
    5,
    13,
    6,
    14,
    7,
    15,
)

SRC_TO_DST_BLK32_ROW_MAP = (
    0,
    8,
    16,
    24,
    1,
    9,
    17,
    25,
    2,
    10,
    18,
    26,
    3,
    11,
    19,
    27,
    4,
    12,
    20,
    28,
    5,
    13,
    21,
    29,
    6,
    14,
    22,
    30,
    7,
    15,
    23,
    31,
)


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


def _shuffle_row_indices(rows: int, epilogue_tile_m: int, device: torch.device) -> torch.Tensor:
    block = 32 if epilogue_tile_m % 128 == 0 else 16
    if rows % block != 0:
        raise ValueError(f"rows must be divisible by {block}")
    row_map = SRC_TO_DST_BLK32_ROW_MAP if block == 32 else SRC_TO_DST_BLK16_ROW_MAP
    row_map_tensor = torch.tensor(row_map, device=device, dtype=torch.long)
    old_rows = torch.arange(rows, device=device, dtype=torch.long)
    mapped_rows = row_map_tensor[old_rows % block]
    new_rows = (old_rows // block) * block + mapped_rows
    row_indices = torch.empty((rows,), device=device, dtype=torch.long)
    row_indices[new_rows] = old_rows
    return row_indices


def _gated_act_row_indices(rows: int, device: torch.device) -> torch.Tensor:
    if rows % 2 != 0:
        raise ValueError("gated activation GEMM rows must be even")
    row_indices = torch.empty((rows,), device=device, dtype=torch.long)
    half = rows // 2
    row_indices[0::2] = torch.arange(half, device=device, dtype=torch.long)
    row_indices[1::2] = torch.arange(half, rows, device=device, dtype=torch.long)
    return row_indices


def _w3_w1_indices(rows: int, epilogue_tile_m: int, device: torch.device) -> torch.Tensor:
    return _gated_act_row_indices(rows, device)[
        _shuffle_row_indices(rows, epilogue_tile_m, device)
    ]


def _make_uint8(shape: tuple[int, ...], seed: int, device: torch.device) -> torch.Tensor:
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    return torch.randint(0, 256, shape, device=device, dtype=torch.uint8, generator=gen)


def _prepare_stage_local(
    weights: torch.Tensor,
    scales: torch.Tensor,
    weight_row_indices: torch.Tensor,
    scale_row_indices: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    prepared_weights = weights.index_select(1, weight_row_indices).contiguous()
    prepared_scales_linear = scales.index_select(1, scale_row_indices).contiguous()
    prepared_scales = (
        kernel.nvfp4_block_scale_interleave(prepared_scales_linear)
        .view(scales.dtype)
        .reshape_as(scales)
    )
    return prepared_weights, prepared_scales


def _prepare_stage_reference(
    weights: torch.Tensor,
    scales: torch.Tensor,
    weight_row_indices: torch.Tensor,
    scale_row_indices: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    from flashinfer import nvfp4_block_scale_interleave

    prepared_weights = weights.index_select(1, weight_row_indices).contiguous()
    prepared_scales_linear = scales.index_select(1, scale_row_indices).contiguous()
    prepared_scales = (
        nvfp4_block_scale_interleave(prepared_scales_linear.view(torch.uint8))
        .view(scales.dtype)
        .reshape_as(scales)
    )
    return prepared_weights, prepared_scales


def _reference_row_indices(
    stage: str,
    rows: int,
    packed_cols: int,
    scale_cols: int,
    epilogue_tile_m: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    from flashinfer.fused_moe.core import (
        _maybe_get_cached_w3_w1_permute_indices,
        get_w2_permute_indices_with_cache,
    )

    cache: dict[tuple[str, torch.Size], torch.Tensor] = {}
    weight_dummy = torch.empty((rows, packed_cols), device=device, dtype=torch.uint8)
    scale_dummy = torch.empty((rows, scale_cols), device=device, dtype=torch.uint8)
    if stage == "gemm1":
        weight_indices = _maybe_get_cached_w3_w1_permute_indices(
            cache, weight_dummy, epilogue_tile_m
        )
        scale_indices = _maybe_get_cached_w3_w1_permute_indices(
            cache, scale_dummy, epilogue_tile_m, num_elts_per_sf=16
        )
    elif stage == "gemm2":
        weight_indices = get_w2_permute_indices_with_cache(
            cache, weight_dummy, epilogue_tile_m
        )
        scale_indices = get_w2_permute_indices_with_cache(
            cache, scale_dummy, epilogue_tile_m, num_elts_per_sf=16
        )
    else:
        raise ValueError(f"unknown stage {stage!r}")
    return weight_indices.to(device), scale_indices.to(device)


def _compare_indices(actual: torch.Tensor, expected: torch.Tensor) -> dict[str, int | str]:
    mismatches = int((actual != expected).sum().item())
    return {
        "status": "PASS" if mismatches == 0 else "FAIL",
        "mismatches": mismatches,
        "numel": int(actual.numel()),
    }


def _compare_tensors(actual: torch.Tensor, expected: torch.Tensor) -> dict[str, int | str]:
    actual_u8 = actual.view(torch.uint8).reshape(-1)
    expected_u8 = expected.view(torch.uint8).reshape(-1)
    mismatches = int((actual_u8 != expected_u8).sum().item())
    return {
        "status": "PASS" if mismatches == 0 else "FAIL",
        "mismatches": mismatches,
        "numel": int(actual_u8.numel()),
        "shape": str(tuple(actual.shape)),
        "dtype": str(actual.dtype),
    }


def _run_stage(
    *,
    stage: str,
    shape: dict,
    epilogue_tile_m: int,
    seed: int,
    warmup: int,
    rep: int,
    device: torch.device,
) -> dict:
    hidden_size = int(shape["hidden_size"])
    intermediate_size = int(shape["intermediate_size"])
    local_num_experts = int(shape["local_num_experts"])
    if stage == "gemm1":
        rows = 2 * intermediate_size
        packed_cols = hidden_size // 2
        scale_cols = hidden_size // 16
        weight_row_indices = _w3_w1_indices(rows, epilogue_tile_m, device)
        scale_row_indices = weight_row_indices
    elif stage == "gemm2":
        rows = hidden_size
        packed_cols = intermediate_size // 2
        scale_cols = intermediate_size // 16
        weight_row_indices = _shuffle_row_indices(rows, epilogue_tile_m, device)
        scale_row_indices = weight_row_indices
    else:
        raise ValueError(f"unknown stage {stage!r}")
    ref_weight_row_indices, ref_scale_row_indices = _reference_row_indices(
        stage, rows, packed_cols, scale_cols, epilogue_tile_m, device
    )
    weight_indices_cmp = _compare_indices(weight_row_indices, ref_weight_row_indices)
    scale_indices_cmp = _compare_indices(scale_row_indices, ref_scale_row_indices)

    weights = _make_uint8((local_num_experts, rows, packed_cols), seed, device)
    scales = _make_uint8((local_num_experts, rows, scale_cols), seed + 1, device).view(
        torch.float8_e4m3fn
    )
    local_weights, local_scales = _prepare_stage_local(
        weights, scales, weight_row_indices, scale_row_indices
    )
    ref_weights, ref_scales = _prepare_stage_reference(
        weights, scales, ref_weight_row_indices, ref_scale_row_indices
    )
    torch.cuda.synchronize()
    weights_cmp = _compare_tensors(local_weights, ref_weights)
    scales_cmp = _compare_tensors(local_scales, ref_scales)

    # Keep these microbenchmarks separate: static weights are prepared offline, but the
    # numbers are useful for tracking the cost of the local layout transform itself.
    prepare_scale_us = _bench(
        lambda: kernel.nvfp4_block_scale_interleave(
            scales.index_select(1, scale_row_indices)
        ),
        warmup,
        rep,
    )

    del weights, scales, local_weights, local_scales, ref_weights, ref_scales
    gc.collect()
    torch.cuda.empty_cache()
    return {
        "stage": stage,
        "rows": rows,
        "packed_cols": packed_cols,
        "scale_cols": scale_cols,
        "row_indices_head": weight_row_indices[: min(32, rows)].detach().cpu().tolist(),
        "weight_row_indices_compare": weight_indices_cmp,
        "scale_row_indices_compare": scale_indices_cmp,
        "weights_compare": weights_cmp,
        "scales_compare": scales_cmp,
        "latency_us": {
            "local_scale_swizzle_after_row_permute": prepare_scale_us,
        },
    }


def run_probe(args: argparse.Namespace) -> dict:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA device is required")
    device = torch.device("cuda")
    shape = get_shape(args.preset)
    if shape["dtype"] != "nvfp4":
        raise NotImplementedError("prepared weight probe currently targets NVFP4")
    if int(shape["local_num_experts"]) != int(shape["num_experts"]):
        raise NotImplementedError("prepared weight probe currently targets TP=1 local experts")
    stages = ("gemm1", "gemm2") if args.stage == "all" else (args.stage,)
    rows = [
        _run_stage(
            stage=stage,
            shape=shape,
            epilogue_tile_m=args.epilogue_tile_m,
            seed=args.seed + idx * 1009,
            warmup=args.warmup,
            rep=args.rep,
            device=device,
        )
        for idx, stage in enumerate(stages)
    ]
    status = (
        "PASS"
        if rows
        and all(row["weight_row_indices_compare"]["status"] == "PASS" for row in rows)
        and all(row["scale_row_indices_compare"]["status"] == "PASS" for row in rows)
        and all(row["weights_compare"]["status"] == "PASS" for row in rows)
        and all(row["scales_compare"]["status"] == "PASS" for row in rows)
        else "FAIL"
    )
    return {
        "status": status,
        "operator": "FlashInfer/TRT-LLM prepared FP4 MoE weight layout",
        "preset": args.preset,
        "epilogue_tile_m": args.epilogue_tile_m,
        "shape": {
            "hidden_size": int(shape["hidden_size"]),
            "intermediate_size": int(shape["intermediate_size"]),
            "num_experts": int(shape["num_experts"]),
            "local_num_experts": int(shape["local_num_experts"]),
            "top_k": int(shape["top_k"]),
            "dtype": str(shape["dtype"]),
        },
        "layout": {
            "gemm1": "row permutation = gated w3/w1 row interleave followed by block32 shuffle; scales then use NVFP4 128x4 swizzle",
            "gemm2": "row permutation = block32 shuffle; scales then use NVFP4 128x4 swizzle",
        },
        "rows": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preset", default="g11")
    parser.add_argument("--stage", choices=("all", "gemm1", "gemm2"), default="all")
    parser.add_argument("--epilogue-tile-m", type=int, default=128)
    parser.add_argument("--seed", type=int, default=20260702)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--rep", type=int, default=3)
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
