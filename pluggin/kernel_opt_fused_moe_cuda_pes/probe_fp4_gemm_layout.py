"""Probe SM103 NVFP4 GEMM building blocks for the G11 MoE target.

This script is evidence collection only.  It does not provide the candidate
kernel and it must not be used as the performance baseline.  The default
candidate remains ``kernel.py``.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch


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


def _max_errors(actual: torch.Tensor, expected: torch.Tensor) -> dict[str, float]:
    actual_f = actual.to(torch.float32)
    expected_f = expected.to(torch.float32)
    diff = (actual_f - expected_f).abs()
    denom = expected_f.abs().clamp_min(1e-6)
    finite = bool(torch.isfinite(actual_f).all() and torch.isfinite(expected_f).all())
    max_abs = float(diff.max().item())
    max_rel = float((diff / denom).max().item())
    return {
        "finite": finite,
        "max_abs": _finite_or_none(max_abs),
        "max_rel": _finite_or_none(max_rel),
    }


def _finite_or_none(value: float) -> float | None:
    return value if math.isfinite(value) else None


def _shuffle32_src_to_dst_row(row: int) -> int:
    in_block = row & 31
    return (row & ~31) + ((in_block & 3) << 3) + (in_block >> 2)


def _prepared_gemm1_row(logical_row: int, intermediate_size: int) -> int:
    if logical_row < intermediate_size:
        gated_row = logical_row << 1
    else:
        gated_row = ((logical_row - intermediate_size) << 1) + 1
    return _shuffle32_src_to_dst_row(gated_row)


def _scale_offset_128x4(row: int, col: int, cols: int) -> int:
    padded_cols = (cols + 3) & ~3
    column_idx_in_group = col & 3
    column_group_idx = col >> 2
    row_idx_in_group0 = row & 31
    row_idx_in_group1 = (row & 127) >> 5
    row_group_idx = row >> 7
    return (
        row_group_idx * 128 * padded_cols
        + column_group_idx * 512
        + row_idx_in_group0 * 16
        + row_idx_in_group1 * 4
        + column_idx_in_group
    )


def _prepared_stage1_reference(
    hidden_states: torch.Tensor,
    hidden_states_scale: torch.Tensor,
    gemm1_weight: torch.Tensor,
    gemm1_scale: torch.Tensor,
    intermediate_size: int,
) -> torch.Tensor:
    from reference import decode_block_scales, dequantize_fp4_tensor, unpack_fp4_e2m1

    hidden = dequantize_fp4_tensor(hidden_states, hidden_states_scale)
    hidden_size = hidden.shape[1]
    rows = torch.tensor(
        [
            _prepared_gemm1_row(row, intermediate_size)
            for row in range(2 * intermediate_size)
        ],
        device=hidden_states.device,
        dtype=torch.long,
    )
    packed = gemm1_weight.index_select(0, rows)
    unpacked = unpack_fp4_e2m1(packed)
    scale_cols = hidden_size // 16
    scale_offsets = torch.empty(
        (2 * intermediate_size, scale_cols),
        device=hidden_states.device,
        dtype=torch.long,
    )
    for row in range(2 * intermediate_size):
        prepared_row = int(rows[row].item())
        for col in range(scale_cols):
            scale_offsets[row, col] = _scale_offset_128x4(
                prepared_row, col, scale_cols
            )
    flat_scales = decode_block_scales(gemm1_scale).reshape(-1)
    scales = flat_scales.index_select(0, scale_offsets.reshape(-1)).reshape(
        2 * intermediate_size, scale_cols
    )
    weights = unpacked * scales.repeat_interleave(16, dim=1)
    return hidden.matmul(weights.T)


def _dense_cutlass_sanity(args: argparse.Namespace) -> dict[str, Any]:
    from flashinfer import SfLayout, mm_fp4, nvfp4_quantize

    torch.manual_seed(args.seed)
    device = torch.device("cuda")
    alpha = torch.tensor(1.0, device=device, dtype=torch.float32)
    a = torch.randn(args.tokens, args.hidden_size, device=device, dtype=torch.bfloat16) / 10
    b = torch.randn(args.out_features, args.hidden_size, device=device, dtype=torch.bfloat16) / 10
    a_fp4, a_sf = nvfp4_quantize(
        a,
        alpha,
        sfLayout=SfLayout.layout_128x4,
        do_shuffle=False,
        sf_vec_size=16,
    )
    b_fp4, b_sf = nvfp4_quantize(
        b,
        alpha,
        sfLayout=SfLayout.layout_128x4,
        do_shuffle=False,
        sf_vec_size=16,
    )
    out = torch.empty((args.tokens, args.out_features), device=device, dtype=torch.bfloat16)

    def run_once() -> torch.Tensor:
        return mm_fp4(
            a_fp4,
            b_fp4.T,
            a_sf,
            b_sf.T,
            alpha,
            torch.bfloat16,
            out,
            block_size=16,
            backend="cutlass",
            use_nvfp4=True,
        )

    run_once()
    latency_us = _bench(run_once, args.warmup, args.rep)
    reference = a.to(torch.float32).matmul(b.to(torch.float32).T)
    errors = _max_errors(out, reference)
    return {
        "name": "dense_cutlass_128x4_sanity",
        "backend": "flashinfer.mm_fp4(cutlass)",
        "shape": {
            "m": args.tokens,
            "n": args.out_features,
            "k": args.hidden_size,
        },
        "latency_us": latency_us,
        "output_sample": _finite_or_none(float(out.flatten()[0].item())),
        "reference": "bf16 matmul against the original unquantized tensors",
        "errors": errors,
        "notes": [
            "Uses nvfp4_quantize with 128x4 scale layout and do_shuffle=False, the layout documented for cutlass mm_fp4.",
            "Quantization error is expected; this check is mainly an interface and finite-output sanity test.",
        ],
    }


def _prepared_weight_probe(args: argparse.Namespace) -> dict[str, Any]:
    from flashinfer import mm_fp4

    from test_kernel import _make_real_nvfp4_flashinfer_inputs

    ns = SimpleNamespace(
        preset="g11",
        hidden_size=None,
        intermediate_size=None,
        local_num_experts=None,
        local_expert_offset=0,
        seed=args.seed,
    )
    inputs = _make_real_nvfp4_flashinfer_inputs(ns, args.tokens)
    hidden_states = inputs[2]
    hidden_states_scale = inputs[3]
    gemm1_weights = inputs[4]
    gemm1_weights_scale = inputs[5]
    gemm2_weights = inputs[10]
    gemm2_weights_scale = inputs[11]
    intermediate_size = inputs[20]
    device = hidden_states.device
    alpha = torch.tensor(1.0, device=device, dtype=torch.float32)

    expert = min(max(args.expert, 0), gemm1_weights.shape[0] - 1)
    stage1_out = torch.empty(
        (args.tokens, 2 * int(intermediate_size)), device=device, dtype=torch.bfloat16
    )

    def run_stage1() -> torch.Tensor:
        return mm_fp4(
            hidden_states,
            gemm1_weights[expert].T,
            hidden_states_scale,
            gemm1_weights_scale[expert].T,
            alpha,
            torch.bfloat16,
            stage1_out,
            block_size=16,
            backend="cutlass",
            use_nvfp4=True,
        )

    trtllm_stage1: dict[str, Any] | None = None

    def run_trtllm_stage1() -> torch.Tensor:
        return mm_fp4(
            hidden_states,
            gemm1_weights[expert].T,
            hidden_states_scale,
            gemm1_weights_scale[expert].T,
            alpha,
            torch.bfloat16,
            trtllm_out,
            block_size=16,
            backend="trtllm",
            use_nvfp4=True,
        )

    stage2_in = torch.randn(
        args.tokens, int(intermediate_size), device=device, dtype=torch.bfloat16
    ) / 10
    from flashinfer import fp4_quantize

    stage2_fp4, stage2_sf = fp4_quantize(
        stage2_in,
        alpha,
        sf_vec_size=16,
        sf_use_ue8m0=False,
        is_sf_swizzled_layout=False,
    )
    stage2_out = torch.empty(
        (args.tokens, gemm2_weights.shape[1]), device=device, dtype=torch.bfloat16
    )

    def run_stage2() -> torch.Tensor:
        return mm_fp4(
            stage2_fp4,
            gemm2_weights[expert].T,
            stage2_sf.view(torch.float8_e4m3fn),
            gemm2_weights_scale[expert].T,
            alpha,
            torch.bfloat16,
            stage2_out,
            block_size=16,
            backend="cutlass",
            use_nvfp4=True,
        )

    stage1_error = None
    stage2_error = None
    try:
        run_stage1()
        stage1_us = _bench(run_stage1, args.warmup, args.rep)
        raw_stage1_sample = float(stage1_out.flatten()[0].item())
        stage1_sample = _finite_or_none(raw_stage1_sample)
        if stage1_sample is None:
            stage1_error = f"non-finite sample {raw_stage1_sample}"
    except Exception as exc:  # pragma: no cover - diagnostic path
        stage1_us = None
        stage1_sample = None
        stage1_error = f"{type(exc).__name__}: {exc}"

    try:
        run_stage2()
        stage2_us = _bench(run_stage2, args.warmup, args.rep)
        raw_stage2_sample = float(stage2_out.flatten()[0].item())
        stage2_sample = _finite_or_none(raw_stage2_sample)
        if stage2_sample is None:
            stage2_error = f"non-finite sample {raw_stage2_sample}"
    except Exception as exc:  # pragma: no cover - diagnostic path
        stage2_us = None
        stage2_sample = None
        stage2_error = f"{type(exc).__name__}: {exc}"

    if args.include_trtllm:
        trtllm_out = torch.empty_like(stage1_out)
        try:
            run_trtllm_stage1()
            trtllm_us = _bench(run_trtllm_stage1, args.warmup, args.rep)
            raw_sample = float(trtllm_out.flatten()[0].item())
            trtllm_stage1 = {
                "latency_us": trtllm_us,
                "sample": _finite_or_none(raw_sample),
                "error": None,
            }
            if args.check_prepared_reference:
                expected = _prepared_stage1_reference(
                    hidden_states,
                    hidden_states_scale,
                    gemm1_weights[expert],
                    gemm1_weights_scale[expert],
                    int(intermediate_size),
                )
                trtllm_stage1["prepared_reference_errors"] = _max_errors(
                    trtllm_out, expected
                )
                trtllm_stage1["prepared_reference_sample"] = _finite_or_none(
                    float(expected.flatten()[0].item())
                )
        except Exception as exc:  # pragma: no cover - diagnostic path
            trtllm_stage1 = {
                "latency_us": None,
                "sample": None,
                "error": f"{type(exc).__name__}: {exc}",
            }

    return {
        "name": "g11_prepared_weight_cutlass_probe",
        "backend": "flashinfer.mm_fp4(cutlass)",
        "tokens": args.tokens,
        "expert": expert,
        "stage1": {
            "m": args.tokens,
            "n": 2 * int(intermediate_size),
            "k": hidden_states.shape[1] * 2,
            "latency_us": stage1_us,
            "sample": stage1_sample,
            "error": stage1_error,
        },
        "stage2": {
            "m": args.tokens,
            "n": gemm2_weights.shape[1],
            "k": int(intermediate_size),
            "latency_us": stage2_us,
            "sample": stage2_sample,
            "error": stage2_error,
        },
        "trtllm_stage1": trtllm_stage1,
        "notes": [
            "The real G11 tensors use FlashInfer/TRT-LLM prepared weights from prepare_static_weights_for_trtllm_fp4_moe.",
            "This probe intentionally tests whether the prepared layout can be consumed by cutlass mm_fp4 directly.",
            "The optional trtllm_stage1 probe tests the lower-level standalone trtllm GEMM API, not the fused MoE operator.",
            "A finite latency is not a correctness proof; the final custom kernel still needs a prepared-layout grouped GEMM/finalize implementation.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["dense", "prepared", "both"], default="both")
    parser.add_argument("--tokens", type=int, default=8)
    parser.add_argument("--hidden-size", type=int, default=4096)
    parser.add_argument("--out-features", type=int, default=2048)
    parser.add_argument("--expert", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--rep", type=int, default=5)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--include-trtllm", action="store_true")
    parser.add_argument("--check-prepared-reference", action="store_true")
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    result: dict[str, Any] = {
        "script": "probe_fp4_gemm_layout.py",
        "device": torch.cuda.get_device_name(),
        "capability": torch.cuda.get_device_capability(),
        "results": [],
    }
    if args.mode in {"dense", "both"}:
        result["results"].append(_dense_cutlass_sanity(args))
    if args.mode in {"prepared", "both"}:
        result["results"].append(_prepared_weight_probe(args))

    text = json.dumps(result, indent=2, sort_keys=True, allow_nan=False)
    print(text)
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(text + "\n")


if __name__ == "__main__":
    main()
