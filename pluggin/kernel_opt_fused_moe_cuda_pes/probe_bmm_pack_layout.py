#!/usr/bin/env python3
"""Validate and time packing hidden FP4 activations into expert-major BMM layout."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from types import SimpleNamespace

import torch

import kernel
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


def _sample_slots(total_slots: int, limit: int, device: torch.device) -> torch.Tensor:
    if limit <= 0 or limit >= total_slots:
        return torch.arange(total_slots, device=device, dtype=torch.long)
    if limit == 1:
        return torch.zeros((1,), device=device, dtype=torch.long)
    return torch.linspace(0, total_slots - 1, steps=limit, device=device).to(torch.long)


def _verify_pack(
    *,
    topk_packed: torch.Tensor,
    expanded: torch.Tensor,
    expert_offsets: torch.Tensor,
    hidden_states: torch.Tensor,
    hidden_states_scale: torch.Tensor,
    hidden_bmm: torch.Tensor,
    hidden_scale_bmm: torch.Tensor,
    padded_rows: int,
    local_expert_offset: int,
    check_slots: int,
) -> dict[str, int | bool]:
    tokens, top_k = map(int, topk_packed.shape)
    total_slots = tokens * top_k
    slots = _sample_slots(total_slots, check_slots, topk_packed.device)
    expanded_flat = expanded.reshape(-1).to(torch.long)
    packed_flat = topk_packed.reshape(-1).to(torch.long)

    experts = ((packed_flat.index_select(0, slots) >> 16) & 0xFFFF).to(torch.long)
    local_experts = experts - int(local_expert_offset)
    offsets = expert_offsets.to(torch.long).index_select(0, local_experts)
    ranks = expanded_flat.index_select(0, slots) - offsets
    tokens_idx = slots // top_k
    dst_rows = local_experts * int(padded_rows) + ranks

    hidden_flat = hidden_bmm.reshape(-1, hidden_bmm.shape[-1])
    actual_hidden = hidden_flat.index_select(0, dst_rows)
    expected_hidden = hidden_states.index_select(0, tokens_idx)
    hidden_mismatches = int((actual_hidden != expected_hidden).sum().item())

    scale_flat = hidden_scale_bmm.reshape(-1, hidden_scale_bmm.shape[-1])
    actual_scale = scale_flat.index_select(0, dst_rows)
    expected_scale = hidden_states_scale.index_select(0, tokens_idx)
    scale_mismatches = int((actual_scale != expected_scale).sum().item())

    return {
        "checked_slots": int(slots.numel()),
        "hidden_mismatches": hidden_mismatches,
        "scale_mismatches": scale_mismatches,
        "rank_min": int(ranks.min().item()),
        "rank_max": int(ranks.max().item()),
        "finite_scale": bool(torch.isfinite(actual_scale.to(torch.float32)).all().item()),
    }


def run_probe(args: argparse.Namespace) -> dict:
    shape = get_shape(args.preset)
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

    padded_rows = int(args.padded_rows or expert_counts.max().item())
    hidden_bmm, hidden_scale_bmm = kernel.pack_hidden_bmm_from_metadata(
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
    verification = _verify_pack(
        topk_packed=topk_packed,
        expanded=expanded,
        expert_offsets=expert_offsets,
        hidden_states=hidden_states,
        hidden_states_scale=hidden_states_scale,
        hidden_bmm=hidden_bmm,
        hidden_scale_bmm=hidden_scale_bmm,
        padded_rows=padded_rows,
        local_expert_offset=local_expert_offset,
        check_slots=args.check_slots,
    )
    status = (
        "PASS"
        if verification["hidden_mismatches"] == 0
        and verification["scale_mismatches"] == 0
        and verification["rank_min"] >= 0
        and verification["rank_max"] < padded_rows
        else "FAIL"
    )

    pack_us = _bench(
        lambda: kernel.pack_hidden_bmm_from_metadata(
            topk_packed,
            expanded,
            expert_offsets,
            hidden_states,
            hidden_states_scale,
            num_experts=num_experts,
            local_expert_offset=local_expert_offset,
            local_num_experts=local_num_experts,
            padded_rows=padded_rows,
        ),
        args.warmup,
        args.rep,
    )

    def metadata_plus_pack():
        meta = kernel.routing_metadata_from_packed(
            topk_packed,
            num_experts=num_experts,
            local_expert_offset=local_expert_offset,
            local_num_experts=local_num_experts,
            tile_tokens_dim=tile_tokens_dim,
        )
        kernel.pack_hidden_bmm_from_metadata(
            topk_packed,
            meta[0],
            meta[7],
            hidden_states,
            hidden_states_scale,
            num_experts=num_experts,
            local_expert_offset=local_expert_offset,
            local_num_experts=local_num_experts,
            padded_rows=padded_rows,
        )

    metadata_plus_pack_us = _bench(metadata_plus_pack, args.warmup, args.rep)

    return {
        "status": status,
        "operator": "G11 hidden FP4 expert-major BMM pack",
        "preset": args.preset,
        "tokens": tokens,
        "shape": {
            "hidden_states": tuple(hidden_states.shape),
            "hidden_states_dtype": str(hidden_states.dtype),
            "hidden_states_scale": tuple(hidden_states_scale.shape),
            "hidden_states_scale_dtype": str(hidden_states_scale.dtype),
            "hidden_bmm": tuple(hidden_bmm.shape),
            "hidden_scale_bmm": tuple(hidden_scale_bmm.shape),
            "hidden_size": int(shape["hidden_size"]),
            "num_experts": num_experts,
            "local_num_experts": local_num_experts,
            "top_k": top_k,
            "tile_tokens_dim": tile_tokens_dim,
            "padded_rows": padded_rows,
            "max_expert_count": int(expert_counts.max().item()),
        },
        "verification": verification,
        "latency_us": {
            "pack_hidden_bmm_from_metadata": pack_us,
            "metadata_plus_pack": metadata_plus_pack_us,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preset", default="g11")
    parser.add_argument("--tokens", type=int)
    parser.add_argument("--tile-tokens-dim", type=int)
    parser.add_argument("--padded-rows", type=int, default=0)
    parser.add_argument("--check-slots", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--rep", type=int, default=9)
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
