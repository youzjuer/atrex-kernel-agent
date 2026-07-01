#!/usr/bin/env python3
"""Validate and time TRT-LLM MoE routing metadata built from packed top-k."""

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


def _max_num_ctas(num_tokens: int, top_k: int, num_experts: int, tile: int) -> int:
    remaining = num_tokens * top_k
    filled = min(num_experts, remaining)
    ctas = filled
    remaining -= filled
    if remaining > 0:
        ctas += remaining // tile
    return ctas


def _unpack_experts(topk_packed: torch.Tensor) -> list[int]:
    values = topk_packed.detach().to(torch.int64).cpu().reshape(-1).tolist()
    experts = []
    for value in values:
        idx = (int(value) >> 16) & 0xFFFF
        if idx >= 0x8000:
            idx -= 0x10000
        experts.append(idx)
    return experts


def _reference_metadata(
    topk_packed: torch.Tensor,
    *,
    num_experts: int,
    local_expert_offset: int,
    local_num_experts: int,
    tile_tokens_dim: int,
) -> dict[str, torch.Tensor | int]:
    tokens, top_k = map(int, topk_packed.shape)
    total_slots = tokens * top_k
    max_ctas = _max_num_ctas(tokens, top_k, num_experts, tile_tokens_dim)
    max_padded = max_ctas * tile_tokens_dim

    expanded = torch.full((total_slots,), -1, dtype=torch.int32)
    permuted_to_token = torch.full((max_padded,), -1, dtype=torch.int32)
    cta_batch = torch.full((max_ctas,), -1, dtype=torch.int32)
    cta_mn_limit = torch.full((max_ctas,), -1, dtype=torch.int32)
    counts = [0 for _ in range(local_num_experts)]
    ranks = [-1 for _ in range(total_slots)]
    local_experts = [-1 for _ in range(total_slots)]

    for slot, expert in enumerate(_unpack_experts(topk_packed)):
        local_expert = expert - local_expert_offset
        if expert < 0 or expert >= num_experts:
            continue
        if local_expert < 0 or local_expert >= local_num_experts:
            continue
        ranks[slot] = counts[local_expert]
        local_experts[slot] = local_expert
        counts[local_expert] += 1

    offsets = [0 for _ in range(local_num_experts + 1)]
    padded_running = 0
    cta_running = 0
    for local_expert, count in enumerate(counts):
        ctas = (count + tile_tokens_dim - 1) // tile_tokens_dim
        offsets[local_expert] = padded_running
        for cta in range(ctas):
            cta_idx = cta_running + cta
            cta_batch[cta_idx] = local_expert
            cta_mn_limit[cta_idx] = min(
                (cta_idx + 1) * tile_tokens_dim,
                padded_running + count,
            )
        padded_running += ctas * tile_tokens_dim
        cta_running += ctas
    offsets[local_num_experts] = padded_running

    for slot, rank in enumerate(ranks):
        if rank < 0:
            continue
        permuted = offsets[local_experts[slot]] + rank
        expanded[slot] = permuted
        permuted_to_token[permuted] = slot // top_k

    return {
        "expanded": expanded.reshape(tokens, top_k),
        "permuted_to_token": permuted_to_token,
        "cta_batch": cta_batch,
        "cta_mn_limit": cta_mn_limit,
        "num_ctas": cta_running,
        "total_padded": padded_running,
        "expert_counts": torch.tensor(counts, dtype=torch.int32),
        "expert_offsets": torch.tensor(offsets, dtype=torch.int32),
    }


def _mismatch_count(actual: torch.Tensor, expected: torch.Tensor) -> int:
    return int((actual.detach().cpu() != expected.cpu()).sum().item())


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
    num_experts = int(inputs[16])
    top_k = int(inputs[17])
    intermediate_size = int(inputs[20])
    local_expert_offset = int(inputs[21])
    local_num_experts = int(inputs[22])
    hidden_states = inputs[2]
    hidden_states_scale = inputs[3]

    tile_tokens_dim = int(
        args.tile_tokens_dim
        if args.tile_tokens_dim is not None
        else _select_tile_tokens_dim(tokens, top_k, local_num_experts)
    )
    ext = kernel._load_ext()
    topk_packed = ext.routing_pack_type1(routing_logits.contiguous(), top_k, 1.0)
    (
        expanded,
        permuted_to_token,
        cta_batch,
        cta_mn_limit,
        num_ctas,
        total_padded,
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

    ref = _reference_metadata(
        topk_packed,
        num_experts=num_experts,
        local_expert_offset=local_expert_offset,
        local_num_experts=local_num_experts,
        tile_tokens_dim=tile_tokens_dim,
    )
    actual_num_ctas = int(num_ctas.cpu().item())
    actual_total_padded = int(total_padded.cpu().item())
    expected_num_ctas = int(ref["num_ctas"])
    expected_total_padded = int(ref["total_padded"])

    mismatches = {
        "expanded": _mismatch_count(expanded, ref["expanded"]),
        "permuted_to_token": _mismatch_count(permuted_to_token, ref["permuted_to_token"]),
        "cta_batch_active": _mismatch_count(
            cta_batch[:expected_num_ctas],
            ref["cta_batch"][:expected_num_ctas],
        ),
        "cta_mn_limit_active": _mismatch_count(
            cta_mn_limit[:expected_num_ctas],
            ref["cta_mn_limit"][:expected_num_ctas],
        ),
        "expert_counts": _mismatch_count(expert_counts, ref["expert_counts"]),
        "expert_offsets": _mismatch_count(expert_offsets, ref["expert_offsets"]),
        "num_ctas": int(actual_num_ctas != expected_num_ctas),
        "total_padded": int(actual_total_padded != expected_total_padded),
    }
    status = "PASS" if sum(mismatches.values()) == 0 else "FAIL"

    routing_pack_us = _bench(
        lambda: ext.routing_pack_type1(routing_logits.contiguous(), top_k, 1.0),
        args.warmup,
        args.rep,
    )
    metadata_us = _bench(
        lambda: kernel.routing_metadata_from_packed(
            topk_packed,
            num_experts=num_experts,
            local_expert_offset=local_expert_offset,
            local_num_experts=local_num_experts,
            tile_tokens_dim=tile_tokens_dim,
        ),
        args.warmup,
        args.rep,
    )

    def pack_plus_metadata():
        packed = ext.routing_pack_type1(routing_logits.contiguous(), top_k, 1.0)
        kernel.routing_metadata_from_packed(
            packed,
            num_experts=num_experts,
            local_expert_offset=local_expert_offset,
            local_num_experts=local_num_experts,
            tile_tokens_dim=tile_tokens_dim,
        )

    pack_plus_metadata_us = _bench(pack_plus_metadata, args.warmup, args.rep)

    return {
        "status": status,
        "operator": "flashinfer.trtllm_fp4_block_scale_moe routing metadata contract",
        "preset": args.preset,
        "tokens": tokens,
        "shape": {
            "hidden_states": tuple(hidden_states.shape),
            "hidden_states_dtype": str(hidden_states.dtype),
            "hidden_states_scale": tuple(hidden_states_scale.shape),
            "hidden_states_scale_dtype": str(hidden_states_scale.dtype),
            "hidden_size": int(shape["hidden_size"]),
            "intermediate_size": intermediate_size,
            "num_experts": num_experts,
            "local_num_experts": local_num_experts,
            "top_k": top_k,
            "tile_tokens_dim": tile_tokens_dim,
        },
        "metadata": {
            "num_non_exiting_ctas": actual_num_ctas,
            "total_num_padded_tokens": actual_total_padded,
            "allocated_ctas": int(cta_batch.numel()),
            "allocated_padded_tokens": int(permuted_to_token.numel()),
            "max_expert_count": int(expert_counts.max().item()),
            "nonzero_experts": int((expert_counts > 0).sum().item()),
        },
        "mismatches": mismatches,
        "latency_us": {
            "routing_pack_type1": routing_pack_us,
            "routing_metadata_from_packed": metadata_us,
            "routing_pack_plus_metadata": pack_plus_metadata_us,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preset", default="g11")
    parser.add_argument("--tokens", type=int)
    parser.add_argument("--tile-tokens-dim", type=int)
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
