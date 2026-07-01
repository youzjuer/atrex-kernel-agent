#!/usr/bin/env python3
"""Probe GEMM1 row-order semantics for FlashInfer prepared FP4 MoE weights.

The prepared weight layout is byte-validated separately.  This probe answers the
next integration question: after consuming the prepared GEMM1 rows, which
logical halves feed the SwiGLU epilogue that FlashInfer implements?

The oracle here is the real ``flashinfer.trtllm_fp4_block_scale_moe`` operator,
not the local PyTorch reference.
"""

from __future__ import annotations

import argparse
import gc
import json
import statistics
from pathlib import Path
from types import SimpleNamespace

import torch

from probe_prepared_weight_layout import _shuffle_row_indices, _w3_w1_indices
from reference import decode_block_scales, dequantize_fp4_tensor, unpack_fp4_e2m1
from test_kernel import _call_flashinfer, _make_real_nvfp4_flashinfer_inputs
from workload_shapes import get_shape


ARG_TOP_K = 17
ARG_ROUTING_METHOD_TYPE = 24
ARG_TUNE_MAX_NUM_TOKENS = 29


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


def _inverse_indices(new_to_old: torch.Tensor) -> torch.Tensor:
    old_to_new = torch.empty_like(new_to_old)
    old_to_new[new_to_old] = torch.arange(
        new_to_old.numel(), device=new_to_old.device, dtype=new_to_old.dtype
    )
    return old_to_new


def _scale_offsets_128x4(rows: torch.Tensor, cols: int) -> torch.Tensor:
    scale_cols = torch.arange(cols, device=rows.device, dtype=torch.long)
    row = rows.to(torch.long).reshape(-1, 1)
    col = scale_cols.reshape(1, -1)
    padded_cols = (cols + 3) & ~3
    return (
        (row >> 7) * 128 * padded_cols
        + (col >> 2) * 512
        + (row & 31) * 16
        + ((row & 127) >> 5) * 4
        + (col & 3)
    )


@torch.no_grad()
def _decode_prepared_rows(
    prepared_weight: torch.Tensor,
    prepared_scale: torch.Tensor,
    physical_rows: torch.Tensor,
) -> torch.Tensor:
    packed = prepared_weight.index_select(0, physical_rows.to(torch.long))
    unpacked = unpack_fp4_e2m1(packed)
    rows = prepared_weight.shape[0]
    scale_cols = prepared_scale.shape[1]
    if rows != prepared_scale.shape[0]:
        raise ValueError("prepared weight/scale row mismatch")
    scale_offsets = _scale_offsets_128x4(physical_rows, scale_cols)
    flat_scales = decode_block_scales(prepared_scale.reshape(-1))
    scales = flat_scales.index_select(0, scale_offsets.reshape(-1)).reshape(
        physical_rows.numel(), scale_cols
    )
    return unpacked * scales.repeat_interleave(16, dim=1)


@torch.no_grad()
def _local_output_variant(
    *,
    hidden_states: torch.Tensor,
    hidden_states_scale: torch.Tensor,
    gemm1_weight: torch.Tensor,
    gemm1_scale: torch.Tensor,
    gemm2_weight: torch.Tensor,
    gemm2_scale: torch.Tensor,
    intermediate_size: int,
    row_mode: str,
    activation_mode: str,
) -> torch.Tensor:
    device = hidden_states.device
    hidden = dequantize_fp4_tensor(hidden_states, hidden_states_scale)
    hidden_size = hidden.shape[1]
    gemm1_rows = 2 * intermediate_size

    if row_mode == "logical":
        gemm1_new_to_old = _w3_w1_indices(gemm1_rows, 128, device)
        gemm1_physical = _inverse_indices(gemm1_new_to_old)
    elif row_mode == "stored":
        gemm1_physical = torch.arange(gemm1_rows, device=device, dtype=torch.long)
    else:
        raise ValueError(f"unknown row_mode {row_mode!r}")

    w1 = _decode_prepared_rows(gemm1_weight, gemm1_scale, gemm1_physical)
    gate_up = hidden.matmul(w1.t())
    first = gate_up[:, :intermediate_size]
    second = gate_up[:, intermediate_size:]
    if activation_mode == "silu_second_times_first":
        activated = torch.nn.functional.silu(second) * first
    elif activation_mode == "silu_first_times_second":
        activated = torch.nn.functional.silu(first) * second
    else:
        raise ValueError(f"unknown activation_mode {activation_mode!r}")

    gemm2_new_to_old = _shuffle_row_indices(hidden_size, 128, device)
    gemm2_physical = _inverse_indices(gemm2_new_to_old)
    w2 = _decode_prepared_rows(gemm2_weight, gemm2_scale, gemm2_physical)
    return activated.matmul(w2.t()).to(torch.bfloat16)


def _first(value):
    return value[0] if isinstance(value, (list, tuple)) else value


def _make_top1_inputs(args: argparse.Namespace) -> tuple:
    ns = SimpleNamespace(
        preset=args.preset,
        hidden_size=None,
        intermediate_size=None,
        local_num_experts=None,
        local_expert_offset=0,
        seed=args.seed,
    )
    inputs = list(_make_real_nvfp4_flashinfer_inputs(ns, args.tokens))
    routing_logits = torch.full_like(inputs[0], -80.0)
    expert = min(max(args.expert, 0), routing_logits.shape[1] - 1)
    routing_logits[:, expert] = 80.0
    inputs[0] = routing_logits
    inputs[ARG_TOP_K] = 1
    inputs[ARG_ROUTING_METHOD_TYPE] = 1
    inputs[ARG_TUNE_MAX_NUM_TOKENS] = max(args.tokens, 8192)
    return tuple(inputs)


def _errors(actual: torch.Tensor, expected: torch.Tensor) -> dict[str, float | bool | None]:
    actual_f = actual.to(torch.float32)
    expected_f = expected.to(torch.float32)
    diff = (actual_f - expected_f).abs()
    rel = diff / expected_f.abs().clamp_min(1e-3)
    finite = bool(torch.isfinite(actual_f).all() and torch.isfinite(expected_f).all())
    return {
        "finite": finite,
        "max_abs": float(diff.max().item()) if diff.numel() else None,
        "mean_abs": float(diff.mean().item()) if diff.numel() else None,
        "max_rel": float(rel.max().item()) if rel.numel() else None,
        "mean_rel": float(rel.mean().item()) if rel.numel() else None,
    }


def run_probe(args: argparse.Namespace) -> dict:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA device is required")
    import flashinfer

    shape = get_shape(args.preset)
    if shape["dtype"] != "nvfp4":
        raise NotImplementedError("this probe targets packed NVFP4 shapes")
    if int(shape["local_num_experts"]) != int(shape["num_experts"]):
        raise NotImplementedError("this probe targets TP=1 all-local experts")

    inputs = _make_top1_inputs(args)
    hidden_states = inputs[2]
    hidden_states_scale = inputs[3]
    gemm1_weights = inputs[4]
    gemm1_weights_scale = inputs[5]
    gemm2_weights = inputs[10]
    gemm2_weights_scale = inputs[11]
    intermediate_size = int(inputs[20])

    expert = min(max(args.expert, 0), gemm1_weights.shape[0] - 1)
    flash_fn = flashinfer.trtllm_fp4_block_scale_moe
    flash_out = _first(_call_flashinfer(flash_fn, inputs))
    torch.cuda.synchronize()

    variants: list[dict] = []
    variant_specs = (
        (
            "logical_silu_second_times_first",
            "logical",
            "silu_second_times_first",
            "Recover logical GEMM1 rows from prepared layout, then use silu(second_half) * first_half.",
        ),
        (
            "logical_silu_first_times_second",
            "logical",
            "silu_first_times_second",
            "Recover logical GEMM1 rows from prepared layout, then use silu(first_half) * second_half.",
        ),
        (
            "stored_silu_second_times_first",
            "stored",
            "silu_second_times_first",
            "Treat prepared storage rows as the direct GEMM1 output order and split the stored tensor halves.",
        ),
        (
            "stored_silu_first_times_second",
            "stored",
            "silu_first_times_second",
            "Treat prepared storage rows as the direct GEMM1 output order and swap the SwiGLU halves.",
        ),
    )
    for name, row_mode, activation_mode, note in variant_specs:
        out = _local_output_variant(
            hidden_states=hidden_states,
            hidden_states_scale=hidden_states_scale,
            gemm1_weight=gemm1_weights[expert],
            gemm1_scale=gemm1_weights_scale[expert],
            gemm2_weight=gemm2_weights[expert],
            gemm2_scale=gemm2_weights_scale[expert],
            intermediate_size=intermediate_size,
            row_mode=row_mode,
            activation_mode=activation_mode,
        )
        torch.cuda.synchronize()
        item = {
            "name": name,
            "row_mode": row_mode,
            "activation_mode": activation_mode,
            "note": note,
            "errors_vs_flashinfer": _errors(out, flash_out),
        }
        variants.append(item)
        del out
        gc.collect()
        torch.cuda.empty_cache()

    variants_sorted = sorted(
        variants,
        key=lambda row: (
            float("inf")
            if row["errors_vs_flashinfer"]["mean_abs"] is None
            else float(row["errors_vs_flashinfer"]["mean_abs"])
        ),
    )
    winner = variants_sorted[0]["name"] if variants_sorted else None
    expected = "logical_silu_second_times_first"
    result = {
        "status": "PASS" if winner == expected else "FAIL",
        "operator": "flashinfer.trtllm_fp4_block_scale_moe",
        "oracle": "flashinfer.trtllm_fp4_block_scale_moe real packed NVFP4 output",
        "preset": args.preset,
        "tokens": args.tokens,
        "expert": expert,
        "forced_top_k": 1,
        "shape": {
            "hidden_size": int(shape["hidden_size"]),
            "intermediate_size": int(shape["intermediate_size"]),
            "num_experts": int(shape["num_experts"]),
            "local_num_experts": int(shape["local_num_experts"]),
            "catalog_top_k": int(shape["top_k"]),
            "dtype": str(shape["dtype"]),
        },
        "expected_winner": expected,
        "winner": winner,
        "variants": variants_sorted,
        "conclusion": (
            "Prepared GEMM1 rows must be consumed through the inverse prepared row map; "
            "the logical first half is multiplied by silu(logical second half)."
            if winner == expected
            else "The measured winner does not match the current CUDA GEMM1 row/activation assumption."
        ),
    }
    if args.bench_flashinfer:
        result["latency_us"] = {
            "flashinfer_trtllm_fp4_block_scale_moe_top1": _bench(
                lambda: _call_flashinfer(flash_fn, inputs), args.warmup, args.rep
            )
        }
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preset", default="g11")
    parser.add_argument("--tokens", type=int, default=2)
    parser.add_argument("--expert", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260702)
    parser.add_argument("--bench-flashinfer", action="store_true")
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
