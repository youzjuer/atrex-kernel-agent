#!/usr/bin/env python3
"""Validate the CUDA SwiGLU/requant epilogue after task08 GEMM1 BMM."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import torch

import kernel
from probe_task08_mpad_compile import DEFAULT_TASK08_ROOT, V310_SYMBOL, _compile_extension, _prepare_tree
from probe_task08_runtime_smoke import _make_case, _valid_row_sample
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


def _shuffle32_src_to_dst(row: torch.Tensor) -> torch.Tensor:
    in_block = row & 31
    return (row & ~31) + ((in_block & 3) << 3) + (in_block >> 2)


def _prepared_gemm1_row(logical_row: torch.Tensor, intermediate_size: int) -> torch.Tensor:
    gated = torch.where(
        logical_row < intermediate_size,
        logical_row << 1,
        ((logical_row - intermediate_size) << 1) + 1,
    )
    return _shuffle32_src_to_dst(gated)


def _encode_e2m1(values: torch.Tensor) -> torch.Tensor:
    neg = values < 0
    ax = values.abs()
    mag = torch.empty_like(ax, dtype=torch.uint8)
    mag[ax < 0.25] = 0
    mag[(ax >= 0.25) & (ax < 0.75)] = 1
    mag[(ax >= 0.75) & (ax < 1.25)] = 2
    mag[(ax >= 1.25) & (ax < 1.75)] = 3
    mag[(ax >= 1.75) & (ax < 2.5)] = 4
    mag[(ax >= 2.5) & (ax < 3.5)] = 5
    mag[(ax >= 3.5) & (ax < 5.0)] = 6
    mag[ax >= 5.0] = 7
    mag = mag | (neg.to(torch.uint8) << 3)
    return torch.where(torch.isfinite(values), mag, torch.zeros_like(mag))


def _scale_offsets(ranks: torch.Tensor, scale_cols: int, groups_k: int) -> torch.Tensor:
    cols = torch.arange(scale_cols, device=ranks.device, dtype=torch.long)
    rows = ranks[:, None].to(torch.long)
    return (
        (rows // 128) * groups_k * 512
        + (cols[None, :] // 4) * 512
        + (rows % 32) * 16
        + ((rows % 128) // 32) * 4
        + (cols[None, :] % 4)
    )


@torch.no_grad()
def _reference_sample(
    gemm1_out: torch.Tensor,
    rows: torch.Tensor,
    *,
    intermediate_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    cols = torch.arange(intermediate_size, device=gemm1_out.device, dtype=torch.long)
    phys1 = _prepared_gemm1_row(cols, intermediate_size)
    phys2 = _prepared_gemm1_row(cols + intermediate_size, intermediate_size)
    selected = gemm1_out.index_select(0, rows)
    x1 = selected.index_select(1, phys1).float()
    x2 = selected.index_select(1, phys2).float()
    activated = torch.nn.functional.silu(x2) * x1
    blocks = activated.reshape(activated.shape[0], intermediate_size // 16, 16)
    finite_blocks = torch.where(torch.isfinite(blocks), blocks, torch.zeros_like(blocks))
    amax = finite_blocks.abs().amax(dim=2)
    scale_fp8 = (amax * (1.0 / 6.0)).to(torch.float8_e4m3fn)
    scale = scale_fp8.float()
    scaled = torch.where(
        scale[:, :, None] == 0,
        torch.zeros_like(blocks),
        blocks / scale[:, :, None],
    )
    codes = _encode_e2m1(scaled)
    packed = codes[:, :, 0::2] | (codes[:, :, 1::2] << 4)
    return packed.reshape(rows.numel(), intermediate_size // 2), scale_fp8


def _invalid_rows(expert_counts: torch.Tensor, *, padded_rows: int, limit: int) -> torch.Tensor:
    rows: list[int] = []
    for expert, count in enumerate(expert_counts.detach().cpu().tolist()):
        for rank in range(int(count), padded_rows):
            rows.append(expert * padded_rows + rank)
            if len(rows) >= limit:
                return torch.tensor(rows, device=expert_counts.device, dtype=torch.long)
    return torch.tensor(rows, device=expert_counts.device, dtype=torch.long)


def run_probe(args: argparse.Namespace) -> dict:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA device is required")
    shape = get_shape(args.preset)
    intermediate_size = int(shape["intermediate_size"])
    task08_root = Path(args.task08_root).resolve()
    work_root, patch_counts = _prepare_tree(task08_root, args.mpad)
    module = _compile_extension(work_root, args.mpad, args.verbose_build)
    fn = getattr(module, V310_SYMBOL)

    case = _make_case(args)
    gemm1_out = fn(
        case["a_fp4"],
        case["a_scale_swizzled_u8"],
        case["w13_fp4"],
        case["w13_scale_swizzled_u8"],
        case["expert_offsets"],
    )
    torch.cuda.synchronize()
    mid_q, mid_scale, mid_scale_swizzled = kernel.swiglu_requant_from_bmm(
        gemm1_out,
        case["expert_counts"],
        padded_rows=int(case["padded_rows"]),
        intermediate_size=intermediate_size,
    )
    torch.cuda.synchronize()

    valid_rows = _valid_row_sample(
        case["expert_counts"],
        padded_rows=int(case["padded_rows"]),
        limit=args.check_valid_rows,
        device=gemm1_out.device,
    )
    ref_q, ref_scale = _reference_sample(
        gemm1_out,
        valid_rows,
        intermediate_size=intermediate_size,
    )
    actual_q = mid_q.reshape(-1, intermediate_size // 2).index_select(0, valid_rows)
    actual_scale = mid_scale.view(torch.uint8).reshape(-1, intermediate_size // 16).index_select(
        0, valid_rows
    )
    q_mismatches = int((actual_q != ref_q).sum().item())
    scale_mismatches = int((actual_scale != ref_scale.view(torch.uint8)).sum().item())

    ranks = valid_rows % int(case["padded_rows"])
    experts = valid_rows // int(case["padded_rows"])
    scale_cols = intermediate_size // 16
    groups_k = (scale_cols + 3) // 4
    swizzled_offsets = _scale_offsets(ranks, scale_cols, groups_k)
    swizzled_bytes = int(mid_scale_swizzled.shape[1])
    flat_offsets = experts[:, None] * swizzled_bytes + swizzled_offsets
    actual_swizzled = mid_scale_swizzled.view(torch.uint8).reshape(-1).index_select(
        0, flat_offsets.reshape(-1)
    ).reshape(valid_rows.numel(), scale_cols)
    swizzled_mismatches = int((actual_swizzled != ref_scale.view(torch.uint8)).sum().item())

    invalid_rows = _invalid_rows(
        case["expert_counts"],
        padded_rows=int(case["padded_rows"]),
        limit=args.check_invalid_rows,
    )
    invalid_q_nonzero = 0
    invalid_scale_nonzero = 0
    if invalid_rows.numel() > 0:
        invalid_q = mid_q.reshape(-1, intermediate_size // 2).index_select(0, invalid_rows)
        invalid_scale = mid_scale.view(torch.uint8).reshape(-1, scale_cols).index_select(
            0, invalid_rows
        )
        invalid_q_nonzero = int((invalid_q != 0).sum().item())
        invalid_scale_nonzero = int((invalid_scale != 0).sum().item())

    epilogue_us = _bench(
        lambda: kernel.swiglu_requant_from_bmm(
            gemm1_out,
            case["expert_counts"],
            padded_rows=int(case["padded_rows"]),
            intermediate_size=intermediate_size,
        ),
        args.warmup,
        args.rep,
    )
    status = (
        "PASS"
        if q_mismatches == 0
        and scale_mismatches == 0
        and swizzled_mismatches == 0
        and invalid_q_nonzero == 0
        and invalid_scale_nonzero == 0
        else "FAIL"
    )
    return {
        "status": status,
        "operator": "task08 GEMM1 BMM + prepared-row-aware SwiGLU NVFP4 requant",
        "preset": args.preset,
        "tokens": int(case["tokens"]),
        "task08_root": str(task08_root),
        "work_root": str(work_root),
        "compiled_symbol": V310_SYMBOL,
        "patch_counts": patch_counts,
        "shape": {
            "intermediate_size": intermediate_size,
            "padded_rows": int(case["padded_rows"]),
            "gemm1_out": tuple(gemm1_out.shape),
            "mid_q": tuple(mid_q.shape),
            "mid_scale": tuple(mid_scale.shape),
            "mid_scale_swizzled": tuple(mid_scale_swizzled.shape),
        },
        "checks": {
            "valid_rows_checked": int(valid_rows.numel()),
            "invalid_rows_checked": int(invalid_rows.numel()),
            "packed_mismatches": q_mismatches,
            "scale_mismatches": scale_mismatches,
            "swizzled_scale_mismatches": swizzled_mismatches,
            "invalid_q_nonzero": invalid_q_nonzero,
            "invalid_scale_nonzero": invalid_scale_nonzero,
        },
        "latency_us": {
            "swiglu_requant_from_bmm": epilogue_us,
        },
        "next_step": "Feed mid_q and mid_scale_swizzled into a GEMM2 BMM/final scatter path.",
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
    parser.add_argument("--check-valid-rows", type=int, default=1024)
    parser.add_argument("--check-invalid-rows", type=int, default=1024)
    parser.add_argument("--verbose-build", action="store_true")
    parser.add_argument("--json-out")
    args = parser.parse_args()

    try:
        result = run_probe(args)
    except Exception as exc:  # noqa: BLE001
        result = {
            "status": "FAIL",
            "operator": "task08 GEMM1 BMM + prepared-row-aware SwiGLU NVFP4 requant",
            "preset": args.preset,
            "mpad": int(args.mpad),
            "error": f"{type(exc).__name__}: {exc}",
        }
    text = json.dumps(result, indent=2)
    print(text)
    if args.json_out:
        Path(args.json_out).write_text(text + "\n")
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
