#!/usr/bin/env python3
"""Check whether the task08 self-written FP4 BMM contract fits current G11 MoE.

Task08 contains a hand-written SM103 FP4 BMM kernel, but it was developed as an
independent GEMM1 BMM workload.  This probe compares that fixed contract against
the current TP=1 G11 routed/packed inputs so the integration work is explicit.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from types import SimpleNamespace

import torch

import kernel
from test_kernel import _make_real_nvfp4_flashinfer_inputs
from workload_shapes import get_shape


TASK08_EXPERTS = 512
TASK08_ROWS_PER_EXPERT = 235
TASK08_K_BYTES = 2048
TASK08_K_SCALE = 256
TASK08_N = 2048


def _select_tile_tokens_dim(tokens: int, top_k: int, local_num_experts: int) -> int:
    supported = (8, 16, 32, 64, 128, 256)
    avg_ceil = max(1, math.ceil((tokens * top_k) / local_num_experts))
    tile = 1 << (avg_ceil - 1).bit_length()
    for candidate in supported:
        if tile <= candidate:
            return candidate
    return supported[-1]


def _swizzled_scale_bytes(rows: int, scale_cols: int) -> int:
    return ((rows + 127) // 128) * ((scale_cols + 3) // 4) * 512


def _bool_row(name: str, ok: bool, actual, expected, note: str = "") -> dict:
    return {
        "name": name,
        "status": "PASS" if ok else "FAIL",
        "actual": actual,
        "expected": expected,
        "note": note,
    }


def _count_summary(expert_counts: torch.Tensor) -> dict[str, int | float]:
    counts = expert_counts.to(torch.float32)
    return {
        "min": int(expert_counts.min().item()),
        "max": int(expert_counts.max().item()),
        "mean": float(counts.mean().item()),
        "nonzero": int((expert_counts > 0).sum().item()),
        "total": int(expert_counts.sum().item()),
    }


def run_probe(args: argparse.Namespace) -> dict:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA device is required")

    shape = get_shape(args.preset)
    if shape["dtype"] != "nvfp4":
        raise NotImplementedError("task08 bridge probe targets packed NVFP4 shapes")

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
    num_experts = int(inputs[16])
    top_k = int(inputs[17])
    local_expert_offset = int(inputs[21])
    local_num_experts = int(inputs[22])
    hidden_size = int(shape["hidden_size"])
    intermediate_size = int(shape["intermediate_size"])
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
    padded_rows = int(args.padded_rows or expert_counts.max().item())
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

    actual_a_shape = tuple(hidden_bmm.shape)
    actual_a_scale_shape = tuple(hidden_scale_swizzled.shape)
    actual_b_shape = tuple(gemm1_weights.shape)
    actual_b_scale_shape = tuple(gemm1_weights_scale.shape)
    task08_a_scale_bytes = _swizzled_scale_bytes(
        TASK08_ROWS_PER_EXPERT, TASK08_K_SCALE
    )
    actual_a_scale_bytes = _swizzled_scale_bytes(padded_rows, hidden_states_scale.shape[1])
    task08_b_scale_bytes = _swizzled_scale_bytes(TASK08_N, TASK08_K_SCALE)
    actual_b_scale_bytes = _swizzled_scale_bytes(
        gemm1_weights.shape[1], gemm1_weights_scale.shape[2]
    )

    checks = [
        _bool_row(
            "experts",
            local_num_experts == TASK08_EXPERTS and num_experts == TASK08_EXPERTS,
            {"num_experts": num_experts, "local_num_experts": local_num_experts},
            {"num_experts": TASK08_EXPERTS, "local_num_experts": TASK08_EXPERTS},
        ),
        _bool_row(
            "activation_packed_bytes",
            hidden_bmm.shape[2] == TASK08_K_BYTES,
            int(hidden_bmm.shape[2]),
            TASK08_K_BYTES,
        ),
        _bool_row(
            "activation_scale_cols",
            hidden_states_scale.shape[1] == TASK08_K_SCALE,
            int(hidden_states_scale.shape[1]),
            TASK08_K_SCALE,
        ),
        _bool_row(
            "rows_per_expert",
            padded_rows == TASK08_ROWS_PER_EXPERT,
            padded_rows,
            TASK08_ROWS_PER_EXPERT,
            "task08 v83/v310 lineage is compiled around fixed Mpad=235",
        ),
        _bool_row(
            "activation_swizzled_scale_bytes",
            actual_a_scale_bytes == task08_a_scale_bytes,
            actual_a_scale_bytes,
            task08_a_scale_bytes,
            "same bytes when both row counts stay in the 129..256 swizzle block",
        ),
        _bool_row(
            "gemm1_weight_shape",
            gemm1_weights.shape
            == (TASK08_EXPERTS, TASK08_N, TASK08_K_BYTES),
            actual_b_shape,
            (TASK08_EXPERTS, TASK08_N, TASK08_K_BYTES),
        ),
        _bool_row(
            "gemm1_scale_shape",
            gemm1_weights_scale.shape
            == (TASK08_EXPERTS, TASK08_N, TASK08_K_SCALE),
            actual_b_scale_shape,
            (TASK08_EXPERTS, TASK08_N, TASK08_K_SCALE),
        ),
        _bool_row(
            "gemm1_swizzled_scale_bytes",
            actual_b_scale_bytes == task08_b_scale_bytes,
            actual_b_scale_bytes,
            task08_b_scale_bytes,
        ),
    ]
    fixed_mpad_ok = padded_rows <= TASK08_ROWS_PER_EXPERT
    all_binary_checks = all(row["status"] == "PASS" for row in checks)
    bridge_status = "PASS" if all_binary_checks and fixed_mpad_ok else "FAIL"

    return {
        "status": bridge_status,
        "operator": "task08 hand-written SM103 FP4 BMM bridge contract",
        "preset": args.preset,
        "tokens": tokens,
        "shape": {
            "hidden_size": hidden_size,
            "intermediate_size": intermediate_size,
            "num_experts": num_experts,
            "local_num_experts": local_num_experts,
            "top_k": top_k,
            "tile_tokens_dim": tile_tokens_dim,
            "padded_rows": padded_rows,
        },
        "task08_contract": {
            "a_fp4": (TASK08_EXPERTS, TASK08_ROWS_PER_EXPERT, TASK08_K_BYTES),
            "a_scale_swizzled": (TASK08_EXPERTS, task08_a_scale_bytes),
            "b_fp4": (TASK08_EXPERTS, TASK08_N, TASK08_K_BYTES),
            "b_scale_swizzled": (TASK08_EXPERTS, task08_b_scale_bytes),
            "out": (TASK08_EXPERTS, TASK08_ROWS_PER_EXPERT, TASK08_N),
        },
        "current_contract": {
            "hidden_bmm": actual_a_shape,
            "hidden_scale_swizzled": actual_a_scale_shape,
            "gemm1_weights_prepared": actual_b_shape,
            "gemm1_weights_scale_prepared": actual_b_scale_shape,
            "gemm1_scale_swizzled_bytes_per_expert": actual_b_scale_bytes,
        },
        "expert_counts": _count_summary(expert_counts),
        "checks": checks,
        "integration_requirements": [
            "Make the task08 BMM M dimension runtime-driven or recompile for the routed padded_rows; current full G11 seed needs more rows than fixed Mpad=235 when status is FAIL on rows_per_expert.",
            "Keep A as expert-major packed NVFP4 [E, padded_rows, H/2] with FlashInfer/SM100 swizzled scales; the existing pack_hidden_bmm_swizzled_from_metadata output already matches that memory family.",
            "Consume FlashInfer prepared GEMM1 weights/scales as B, but add an epilogue or output reorder that restores logical GEMM1 row semantics before SwiGLU; prepared storage order is not the activation order.",
            "Fuse or account for routing metadata, hidden pack, GEMM1 SwiGLU/FP4 requant, GEMM2, and final top-k scatter before comparing against flashinfer.trtllm_fp4_block_scale_moe.",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preset", default="g11")
    parser.add_argument("--tokens", type=int)
    parser.add_argument("--tile-tokens-dim", type=int)
    parser.add_argument("--padded-rows", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1234)
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
