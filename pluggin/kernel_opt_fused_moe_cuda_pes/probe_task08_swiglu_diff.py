#!/usr/bin/env python3
"""Compare legacy and metadata SwiGLU requant outputs for task08 G11."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import torch

import kernel
from probe_task08_end_to_end import _build_routing_and_hidden_pack, _make_inputs
from probe_task08_gemm2_runtime import _prepare_gemm1_tree
from probe_task08_mpad_compile import DEFAULT_TASK08_ROOT, V310_SYMBOL, _compile_extension
from workload_shapes import get_shape


def _trace(args: argparse.Namespace, message: str) -> None:
    if args.trace:
        print(f"[trace] {message}", flush=True)


def _byte_tensor(tensor: torch.Tensor) -> torch.Tensor:
    return tensor if tensor.dtype == torch.uint8 else tensor.view(torch.uint8)


def _flat_row_bytes(tensor: torch.Tensor) -> torch.Tensor:
    return _byte_tensor(tensor).reshape(tensor.shape[0] * tensor.shape[1], -1).contiguous()


def _record_row_byte(
    *,
    row_flat: int,
    col: int,
    diff: int,
    reference: int,
    candidate: int,
    padded_rows: int,
    expert_counts_cpu: list[int],
) -> dict[str, int | bool]:
    local_expert = row_flat // padded_rows
    row_in_expert = row_flat - local_expert * padded_rows
    expert_count = int(expert_counts_cpu[local_expert])
    return {
        "row_flat": int(row_flat),
        "local_expert": int(local_expert),
        "row_in_expert": int(row_in_expert),
        "expert_count": expert_count,
        "valid_row": bool(row_in_expert < expert_count),
        "byte_col": int(col),
        "byte_col_mod_8": int(col % 8),
        "byte_col_mod_16": int(col % 16),
        "byte_col_mod_64": int(col % 64),
        "abs_diff": int(diff),
        "reference": int(reference),
        "candidate": int(candidate),
    }


@torch.no_grad()
def _summarize_row_bytes(
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
    rows_total = int(reference.shape[0])
    cols = int(reference.shape[1])
    expert_counts_device = expert_counts.to(device=reference.device)
    expert_counts_cpu = [int(v) for v in expert_counts.detach().cpu().tolist()]
    total_numel = 0
    total_rows = 0
    sum_abs = 0.0
    max_abs = 0
    threshold_counts = {f"{threshold:g}": 0 for threshold in thresholds}
    first_bad = None
    top: list[dict[str, int | bool]] = []

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
        idx = torch.nonzero(selected, as_tuple=False).flatten()
        selected_rows = row_ids.index_select(0, idx)
        ref_chunk = reference[start:end].index_select(0, idx)
        cand_chunk = candidate[start:end].index_select(0, idx)
        diff = (cand_chunk.to(torch.int16) - ref_chunk.to(torch.int16)).abs()
        flat = diff.reshape(-1)
        total_numel += int(flat.numel())
        total_rows += int(diff.shape[0])
        sum_abs += float(flat.to(torch.float32).sum().item())
        max_abs = max(max_abs, int(flat.max().item()))
        for threshold in thresholds:
            threshold_counts[f"{threshold:g}"] += int((flat > threshold).sum().item())
        if first_bad is None:
            bad = flat > bad_threshold
            if bool(bad.any().item()):
                bad_idx = int(torch.nonzero(bad, as_tuple=False)[0].item())
                row_pos = bad_idx // cols
                col = bad_idx - row_pos * cols
                first_bad = _record_row_byte(
                    row_flat=int(selected_rows[row_pos].item()),
                    col=col,
                    diff=int(flat[bad_idx].item()),
                    reference=int(ref_chunk.reshape(-1)[bad_idx].item()),
                    candidate=int(cand_chunk.reshape(-1)[bad_idx].item()),
                    padded_rows=padded_rows,
                    expert_counts_cpu=expert_counts_cpu,
                )
        k = min(int(topk), int(flat.numel()))
        if k > 0:
            values, indices = torch.topk(flat, k)
            additions = []
            ref_flat = ref_chunk.reshape(-1)
            cand_flat = cand_chunk.reshape(-1)
            for value, index in zip(values.detach().cpu().tolist(), indices.detach().cpu().tolist()):
                if int(value) <= 0:
                    continue
                row_pos = int(index) // cols
                col = int(index) - row_pos * cols
                additions.append(
                    _record_row_byte(
                        row_flat=int(selected_rows[row_pos].item()),
                        col=col,
                        diff=int(value),
                        reference=int(ref_flat[int(index)].item()),
                        candidate=int(cand_flat[int(index)].item()),
                        padded_rows=padded_rows,
                        expert_counts_cpu=expert_counts_cpu,
                    )
                )
            if additions:
                top = (top + additions)
                top.sort(key=lambda item: int(item["abs_diff"]), reverse=True)
                top = top[: int(topk)]

    return {
        "rows": total_rows,
        "numel": total_numel,
        "max_abs": max_abs,
        "mean_abs": float(sum_abs / total_numel) if total_numel else 0.0,
        "threshold_counts": threshold_counts,
        "bad_threshold": float(bad_threshold),
        "first_bad": first_bad,
        "top": top,
    }


def _summarize_all_bytes(
    reference: torch.Tensor,
    candidate: torch.Tensor,
    *,
    thresholds: list[float],
    topk: int,
    bad_threshold: float,
) -> dict[str, Any]:
    if reference.shape != candidate.shape:
        raise ValueError(f"shape mismatch: reference={tuple(reference.shape)} candidate={tuple(candidate.shape)}")
    ref = _byte_tensor(reference).reshape(-1)
    cand = _byte_tensor(candidate).reshape(-1)
    diff = (cand.to(torch.int16) - ref.to(torch.int16)).abs()
    numel = int(diff.numel())
    result: dict[str, Any] = {
        "numel": numel,
        "max_abs": int(diff.max().item()) if numel else 0,
        "mean_abs": float(diff.to(torch.float32).mean().item()) if numel else 0.0,
        "threshold_counts": {f"{threshold:g}": int((diff > threshold).sum().item()) for threshold in thresholds},
        "bad_threshold": float(bad_threshold),
        "first_bad": None,
        "top": [],
    }
    bad = diff > bad_threshold
    if bool(bad.any().item()):
        first = int(torch.nonzero(bad, as_tuple=False)[0].item())
        result["first_bad"] = {
            "byte_offset": first,
            "abs_diff": int(diff[first].item()),
            "reference": int(ref[first].item()),
            "candidate": int(cand[first].item()),
        }
    k = min(int(topk), numel)
    if k > 0:
        values, indices = torch.topk(diff, k)
        top = []
        for value, index in zip(values.detach().cpu().tolist(), indices.detach().cpu().tolist()):
            if int(value) <= 0:
                continue
            idx = int(index)
            top.append(
                {
                    "byte_offset": idx,
                    "abs_diff": int(value),
                    "reference": int(ref[idx].item()),
                    "candidate": int(cand[idx].item()),
                }
            )
        result["top"] = top
    return result


def _run_legacy(case: dict, gemm1_out: torch.Tensor, args: argparse.Namespace):
    return kernel.swiglu_requant_from_bmm(
        gemm1_out,
        case["expert_counts"],
        padded_rows=int(args.mpad),
        intermediate_size=int(case["intermediate_size"]),
    )


def _run_metadata(case: dict, gemm1_out: torch.Tensor, args: argparse.Namespace):
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
        raise NotImplementedError("task08 SwiGLU diff targets packed NVFP4")
    tokens = int(args.tokens if args.tokens is not None else shape["tokens"][0])
    thresholds = sorted(set(float(v) for v in args.threshold))

    task08_root = Path(args.task08_root).resolve()
    _trace(args, "compile GEMM1 extension")
    gemm1_root, gemm1_patch_counts = _prepare_gemm1_tree(task08_root, args.mpad)
    gemm1_module = _compile_extension(gemm1_root, args.mpad, args.verbose_build)
    gemm1_fn = getattr(gemm1_module, V310_SYMBOL)

    _trace(args, "build real G11 case")
    inputs = _make_inputs(args, tokens)
    case = _build_routing_and_hidden_pack(args, inputs)
    _trace(args, "run GEMM1")
    gemm1_out = gemm1_fn(
        case["a_fp4"],
        case["a_scale_swizzled_u8"],
        case["w13_fp4"],
        case["w13_scale_swizzled_u8"],
        case["affine_expert_offsets"],
    )
    torch.cuda.synchronize()

    _trace(args, "run legacy SwiGLU")
    legacy_mid_q, legacy_mid_scale, legacy_mid_scale_swizzled = _run_legacy(case, gemm1_out, args)
    torch.cuda.synchronize()
    _trace(args, "run metadata SwiGLU")
    metadata_mid_q, metadata_mid_scale, metadata_mid_scale_swizzled = _run_metadata(case, gemm1_out, args)
    torch.cuda.synchronize()

    padded_rows = int(args.mpad)
    mid_q_valid = _summarize_row_bytes(
        reference=_flat_row_bytes(legacy_mid_q),
        candidate=_flat_row_bytes(metadata_mid_q),
        expert_counts=case["expert_counts"],
        padded_rows=padded_rows,
        chunk_rows=int(args.chunk_rows),
        thresholds=thresholds,
        topk=int(args.topk),
        bad_threshold=float(args.bad_threshold),
        row_filter="valid",
    )
    mid_q_padding = _summarize_row_bytes(
        reference=_flat_row_bytes(legacy_mid_q),
        candidate=_flat_row_bytes(metadata_mid_q),
        expert_counts=case["expert_counts"],
        padded_rows=padded_rows,
        chunk_rows=int(args.chunk_rows),
        thresholds=thresholds,
        topk=min(int(args.topk), 8),
        bad_threshold=float(args.bad_threshold),
        row_filter="padding",
    )
    scale_valid = _summarize_row_bytes(
        reference=_flat_row_bytes(legacy_mid_scale),
        candidate=_flat_row_bytes(metadata_mid_scale),
        expert_counts=case["expert_counts"],
        padded_rows=padded_rows,
        chunk_rows=int(args.chunk_rows),
        thresholds=thresholds,
        topk=int(args.topk),
        bad_threshold=float(args.bad_threshold),
        row_filter="valid",
    )
    scale_padding = _summarize_row_bytes(
        reference=_flat_row_bytes(legacy_mid_scale),
        candidate=_flat_row_bytes(metadata_mid_scale),
        expert_counts=case["expert_counts"],
        padded_rows=padded_rows,
        chunk_rows=int(args.chunk_rows),
        thresholds=thresholds,
        topk=min(int(args.topk), 8),
        bad_threshold=float(args.bad_threshold),
        row_filter="padding",
    )
    swizzled = _summarize_all_bytes(
        legacy_mid_scale_swizzled,
        metadata_mid_scale_swizzled,
        thresholds=thresholds,
        topk=int(args.topk),
        bad_threshold=float(args.bad_threshold),
    )
    status = (
        "PASS"
        if mid_q_valid["max_abs"] == 0
        and scale_valid["max_abs"] == 0
        and swizzled["max_abs"] == 0
        else "FAIL"
    )
    return {
        "status": status,
        "operator": "task08 SwiGLU legacy-vs-metadata diff",
        "preset": args.preset,
        "tokens": tokens,
        "task08_root": str(task08_root),
        "work_roots": {"gemm1": str(gemm1_root)},
        "compiled_symbol": {"gemm1": V310_SYMBOL},
        "patch_counts": {"gemm1": gemm1_patch_counts},
        "shape": {
            "padded_rows": padded_rows,
            "intermediate_size": int(case["intermediate_size"]),
            "local_num_experts": int(case["local_num_experts"]),
            "mid_q": tuple(legacy_mid_q.shape),
            "mid_scale": tuple(legacy_mid_scale.shape),
            "mid_scale_swizzled": tuple(legacy_mid_scale_swizzled.shape),
        },
        "expert_counts": {
            "min": int(case["expert_counts"].min().item()),
            "max": int(case["expert_counts"].max().item()),
            "mean": float(case["expert_counts"].to(torch.float32).mean().item()),
            "total": int(case["expert_counts"].sum().item()),
        },
        "checks": {
            "thresholds": thresholds,
            "mid_q_valid": mid_q_valid,
            "mid_q_padding": mid_q_padding,
            "mid_scale_valid": scale_valid,
            "mid_scale_padding": scale_padding,
            "mid_scale_swizzled_all": swizzled,
        },
        "elapsed_s": time.perf_counter() - started,
        "next_step": (
            "Metadata SwiGLU matches legacy on valid rows."
            if status == "PASS"
            else "Use first/top diff locations to fix metadata SwiGLU row or scale layout."
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
    parser.add_argument("--chunk-rows", type=int, default=256)
    parser.add_argument("--topk", type=int, default=32)
    parser.add_argument("--threshold", type=float, action="append", default=[0.0])
    parser.add_argument("--bad-threshold", type=float, default=0.0)
    parser.add_argument("--trace", action="store_true")
    parser.add_argument("--verbose-build", action="store_true")
    parser.add_argument("--json-out")
    args = parser.parse_args()

    try:
        result = run_probe(args)
    except Exception as exc:  # noqa: BLE001
        result = {
            "status": "FAIL",
            "operator": "task08 SwiGLU legacy-vs-metadata diff",
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
