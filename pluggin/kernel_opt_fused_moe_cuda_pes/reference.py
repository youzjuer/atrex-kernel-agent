"""FlashInfer-aligned reference for TRT-LLM FP4 block-scale MoE.

The operator contract mirrors ``flashinfer.trtllm_fp4_block_scale_moe``.  The
default case models the Qwen3.5-397B/Plus MoE prefill TP2 metadata:

* global experts: 512
* top_k: 10
* hidden_size: 4096
* intermediate_size: 1024
* local experts under TP2/EP-style sharding: 256 per rank

The smoke tests intentionally use smaller hidden/intermediate dimensions while
keeping the same routing/expert contract, because the checked-in CUDA baseline
is a correctness-first scalar implementation.

Performance presets ``g11`` / ``g8`` are the tp=1 production shapes sourced from
the proj_019 workload catalog (see ``workload_shapes.py``); tp=1 keeps the full
intermediate_size=1024 and all 512 experts local (unlike the TP2 default, which
shards intermediate/experts per rank).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch


ARG_NAMES = (
    "routing_logits",
    "routing_bias",
    "hidden_states",
    "hidden_states_scale",
    "gemm1_weights",
    "gemm1_weights_scale",
    "gemm1_bias",
    "gemm1_alpha",
    "gemm1_beta",
    "gemm1_clamp_limit",
    "gemm2_weights",
    "gemm2_weights_scale",
    "gemm2_bias",
    "output1_scale_scalar",
    "output1_scale_gate_scalar",
    "output2_scale_scalar",
    "num_experts",
    "top_k",
    "n_group",
    "topk_group",
    "intermediate_size",
    "local_expert_offset",
    "local_num_experts",
    "routed_scaling_factor",
    "routing_method_type",
    "do_finalize",
    "enable_pdl",
    "activation_type",
    "output",
    "tune_max_num_tokens",
    "norm_topk_prob",
    "routing_replay_out",
)


@dataclass(frozen=True)
class MoeShape:
    name: str
    tokens: int
    hidden_size: int
    intermediate_size: int
    num_experts: int = 512
    top_k: int = 10
    local_expert_offset: int = 0
    local_num_experts: int = 16
    routed_scaling_factor: float = 1.0
    routing_method_type: int = 0
    n_group: Optional[int] = None
    topk_group: Optional[int] = None


QWEN3_5_PLUS_PREFILL_TP2 = MoeShape(
    name="Qwen3_5-Plus_prefill_TP2",
    tokens=4096,
    hidden_size=4096,
    intermediate_size=1024,
    num_experts=512,
    top_k=10,
    local_expert_offset=0,
    local_num_experts=256,
    routing_method_type=0,
)


SMOKE_SHAPES = {
    "smoke": MoeShape(
        name="Qwen3_5-Plus_prefill_TP2_smoke",
        tokens=2,
        hidden_size=128,
        intermediate_size=64,
    ),
    "qwen_micro": MoeShape(
        name="Qwen3_5-Plus_prefill_TP2_micro",
        tokens=4,
        hidden_size=256,
        intermediate_size=128,
    ),
}


_E2M1_LUT_VALUES = [
    0.0,
    0.5,
    1.0,
    1.5,
    2.0,
    3.0,
    4.0,
    6.0,
    -0.0,
    -0.5,
    -1.0,
    -1.5,
    -2.0,
    -3.0,
    -4.0,
    -6.0,
]


def _as_scale(routed_scaling_factor: Optional[float]) -> float:
    return 1.0 if routed_scaling_factor is None else float(routed_scaling_factor)


@torch.no_grad()
def unpack_fp4_e2m1(packed: torch.Tensor) -> torch.Tensor:
    """Unpack packed e2m1 FP4 values. Low nibble is the first element."""
    lut = torch.tensor(_E2M1_LUT_VALUES, dtype=torch.float32, device=packed.device)
    p = packed.view(torch.uint8).to(torch.int64)
    lo = lut[p & 0x0F]
    hi = lut[(p >> 4) & 0x0F]
    return torch.stack([lo, hi], dim=-1).reshape(*packed.shape[:-1], packed.shape[-1] * 2)


@torch.no_grad()
def _ue8m0_to_float32(scales: torch.Tensor) -> torch.Tensor:
    e = scales.view(torch.uint8).to(torch.int64)
    return torch.pow(torch.tensor(2.0, device=scales.device), (e - 127).float())


@torch.no_grad()
def decode_block_scales(scales: torch.Tensor) -> torch.Tensor:
    """Decode FP4 block scales to float32.

    FlashInfer's public wrapper currently requires fp8_e4m3fn scale tensors for
    this op, but UE8M0 uint8 scales are kept here for reference compatibility.
    """
    if scales.dtype == torch.uint8:
        return _ue8m0_to_float32(scales)
    return scales.to(torch.float32)


@torch.no_grad()
def dequantize_fp4_tensor(packed: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    unpacked = unpack_fp4_e2m1(packed)
    block_size = unpacked.shape[-1] // scales.shape[-1]
    expanded_scales = decode_block_scales(scales).repeat_interleave(block_size, dim=-1)
    return unpacked * expanded_scales


@torch.no_grad()
def dequantize_hidden_states(
    hidden_states: torch.Tensor,
    hidden_states_scale: Optional[torch.Tensor],
) -> torch.Tensor:
    if hidden_states.dtype == torch.bfloat16:
        return hidden_states.to(torch.float32)
    if hidden_states.dtype == torch.float8_e4m3fn:
        if hidden_states_scale is None:
            return hidden_states.to(torch.float32)
        scales = decode_block_scales(hidden_states_scale)
        block_size = hidden_states.shape[-1] // hidden_states_scale.shape[-1]
        return hidden_states.to(torch.float32) * scales.repeat_interleave(block_size, dim=-1)
    if hidden_states.dtype == torch.uint8:
        if hidden_states_scale is None:
            raise ValueError("hidden_states_scale is required for packed FP4 activations")
        return dequantize_fp4_tensor(hidden_states, hidden_states_scale)
    raise TypeError(f"unsupported hidden_states dtype: {hidden_states.dtype}")


@torch.no_grad()
def compute_routing(
    routing_logits: torch.Tensor,
    routing_bias: Optional[torch.Tensor],
    *,
    top_k: int,
    n_group: Optional[int],
    topk_group: Optional[int],
    routed_scaling_factor: Optional[float],
    routing_method_type: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(topk_idx, topk_weights)`` matching FlashInfer routing modes."""
    logits = routing_logits.to(torch.float32)
    scale = _as_scale(routed_scaling_factor)

    if routing_method_type == 0:
        if routing_bias is not None:
            logits = logits + routing_bias.to(torch.float32).reshape(1, -1)
        scores = torch.softmax(logits, dim=-1)
        _, topk_idx = torch.topk(scores, k=top_k, dim=1, largest=True, sorted=False)
        weights = scores.gather(1, topk_idx) * scale
        return topk_idx.to(torch.int64), weights.to(torch.float32)

    if routing_method_type == 1:
        if routing_bias is not None:
            logits = logits + routing_bias.to(torch.float32).reshape(1, -1)
        _, topk_idx = torch.topk(logits, k=top_k, dim=1, largest=True, sorted=False)
        weights = torch.softmax(logits.gather(1, topk_idx), dim=-1) * scale
        return topk_idx.to(torch.int64), weights.to(torch.float32)

    if routing_method_type == 2:
        if routing_bias is None:
            raise ValueError("routing_bias is required for DeepSeekV3-style routing")
        if n_group is None or topk_group is None:
            raise ValueError("n_group and topk_group are required for grouped routing")
        T, E_global = logits.shape
        sigmoid_scores = torch.sigmoid(logits)
        rank_scores = sigmoid_scores + routing_bias.to(torch.float32).reshape(1, -1)
        group_size = E_global // int(n_group)
        grouped = rank_scores.view(T, int(n_group), group_size)
        top2_vals, _ = torch.topk(grouped, k=2, dim=2, largest=True, sorted=False)
        group_scores = top2_vals.sum(dim=2)
        _, group_idx = torch.topk(
            group_scores, k=int(topk_group), dim=1, largest=True, sorted=False
        )
        group_mask = torch.zeros_like(group_scores)
        group_mask.scatter_(1, group_idx, 1.0)
        score_mask = (
            group_mask.unsqueeze(2).expand(T, int(n_group), group_size).reshape(T, E_global)
        )
        pruned = rank_scores.masked_fill(score_mask == 0, torch.finfo(torch.float32).min)
        _, topk_idx = torch.topk(pruned, k=top_k, dim=1, largest=True, sorted=False)

        selected = torch.zeros_like(sigmoid_scores)
        selected.scatter_(1, topk_idx, 1.0)
        raw_weights = sigmoid_scores * selected
        full_weights = raw_weights / (raw_weights.sum(dim=1, keepdim=True) + 1e-20)
        weights = full_weights.gather(1, topk_idx) * scale
        return topk_idx.to(torch.int64), weights.to(torch.float32)

    if routing_method_type == 4:
        if routing_bias is not None:
            logits = logits + routing_bias.to(torch.float32).reshape(1, -1)
        scores = torch.softmax(logits, dim=-1)
        _, topk_idx = torch.topk(scores, k=top_k, dim=1, largest=True, sorted=False)
        gathered = scores.gather(1, topk_idx)
        weights = gathered / (gathered.sum(dim=1, keepdim=True) + 1e-20)
        return topk_idx.to(torch.int64), (weights * scale).to(torch.float32)

    raise NotImplementedError(f"routing_method_type={routing_method_type} is not implemented")


@torch.no_grad()
def _run_experts(
    hidden_states: torch.Tensor,
    hidden_states_scale: Optional[torch.Tensor],
    gemm1_weights: torch.Tensor,
    gemm1_weights_scale: torch.Tensor,
    gemm1_bias: Optional[torch.Tensor],
    gemm2_weights: torch.Tensor,
    gemm2_weights_scale: torch.Tensor,
    gemm2_bias: Optional[torch.Tensor],
    topk_idx: torch.Tensor,
    topk_weights: torch.Tensor,
    local_expert_offset: int,
    num_experts: int,
) -> torch.Tensor:
    A = dequantize_hidden_states(hidden_states, hidden_states_scale)
    W1 = dequantize_fp4_tensor(gemm1_weights, gemm1_weights_scale)
    W2 = dequantize_fp4_tensor(gemm2_weights, gemm2_weights_scale)

    E_local, two_intermediate, hidden_size = W1.shape
    intermediate_size = two_intermediate // 2
    output = torch.zeros((A.shape[0], hidden_size), dtype=torch.float32, device=A.device)
    local_start = int(local_expert_offset)

    for le in range(E_local):
        ge = local_start + le
        if ge < 0 or ge >= int(num_experts):
            continue
        selected = topk_idx == ge
        if not selected.any():
            continue
        token_idx = torch.nonzero(selected.any(dim=1), as_tuple=False).squeeze(1)
        gate_up = A.index_select(0, token_idx).matmul(W1[le].t())
        if gemm1_bias is not None:
            gate_up = gate_up + gemm1_bias[le].to(torch.float32)

        # TRT-LLM convention: first half X1, second half X2, activation = silu(X2) * X1.
        x1 = gate_up[:, :intermediate_size]
        x2 = gate_up[:, intermediate_size:]
        activated = torch.nn.functional.silu(x2) * x1
        expert_out = activated.matmul(W2[le].t())
        if gemm2_bias is not None:
            expert_out = expert_out + gemm2_bias[le].to(torch.float32)

        selected_rows = selected.index_select(0, token_idx).to(torch.float32)
        weights = (topk_weights.index_select(0, token_idx) * selected_rows).sum(dim=1)
        output.index_add_(0, token_idx, expert_out * weights.unsqueeze(1))

    return output.to(torch.bfloat16)


@torch.no_grad()
def run_reference(
    routing_logits: torch.Tensor,
    routing_bias: Optional[torch.Tensor],
    hidden_states: torch.Tensor,
    hidden_states_scale: Optional[torch.Tensor],
    gemm1_weights: torch.Tensor,
    gemm1_weights_scale: torch.Tensor,
    gemm1_bias: Optional[torch.Tensor],
    gemm1_alpha: Optional[torch.Tensor],
    gemm1_beta: Optional[torch.Tensor],
    gemm1_clamp_limit: Optional[torch.Tensor],
    gemm2_weights: torch.Tensor,
    gemm2_weights_scale: torch.Tensor,
    gemm2_bias: Optional[torch.Tensor],
    output1_scale_scalar: Optional[torch.Tensor],
    output1_scale_gate_scalar: Optional[torch.Tensor],
    output2_scale_scalar: Optional[torch.Tensor],
    num_experts: int,
    top_k: int,
    n_group: Optional[int],
    topk_group: Optional[int],
    intermediate_size: int,
    local_expert_offset: int,
    local_num_experts: int,
    routed_scaling_factor: Optional[float],
    routing_method_type: int = 0,
    do_finalize: bool = True,
    enable_pdl: Optional[bool] = None,
    activation_type: int = 3,
    output: Optional[torch.Tensor] = None,
    tune_max_num_tokens: int = 8192,
    norm_topk_prob: bool = True,
    routing_replay_out: Optional[torch.Tensor] = None,
) -> list[torch.Tensor]:
    del (
        gemm1_alpha,
        gemm1_beta,
        gemm1_clamp_limit,
        output1_scale_scalar,
        output1_scale_gate_scalar,
        output2_scale_scalar,
        local_num_experts,
        enable_pdl,
        tune_max_num_tokens,
        norm_topk_prob,
        routing_replay_out,
    )
    if activation_type != 3:
        raise NotImplementedError("only activation_type=3 (SwiGLU) is implemented")
    if not do_finalize:
        raise NotImplementedError("this reference currently returns finalized output only")
    if intermediate_size != gemm2_weights.shape[2] * 2:
        raise ValueError("intermediate_size must match packed gemm2 weight shape")

    topk_idx, weights = compute_routing(
        routing_logits,
        routing_bias,
        top_k=top_k,
        n_group=n_group,
        topk_group=topk_group,
        routed_scaling_factor=routed_scaling_factor,
        routing_method_type=routing_method_type,
    )
    out = _run_experts(
        hidden_states,
        hidden_states_scale,
        gemm1_weights,
        gemm1_weights_scale,
        gemm1_bias,
        gemm2_weights,
        gemm2_weights_scale,
        gemm2_bias,
        topk_idx,
        weights,
        local_expert_offset,
        num_experts,
    )
    if output is not None:
        output.copy_(out)
        out = output
    return [out]


def _scale_tensor(shape: tuple[int, ...], *, device: str, value: float = 0.03125) -> torch.Tensor:
    return torch.full(shape, value, device=device, dtype=torch.float32).to(torch.float8_e4m3fn)


def _resolve_base(preset: str) -> MoeShape:
    """Resolve a preset name to a base ``MoeShape``.

    ``smoke`` / ``qwen_micro`` / ``qwen_tp2`` are local; tp=1 performance presets
    (``g11``, ``g8``, ...) are sourced from the proj_019 workload catalog via
    ``workload_shapes`` (never hardcoded here).
    """
    if preset == "qwen_tp2":
        return QWEN3_5_PLUS_PREFILL_TP2
    if preset in SMOKE_SHAPES:
        return SMOKE_SHAPES[preset]
    from workload_shapes import get_shape

    s = get_shape(preset)
    return MoeShape(
        name=s["name"],
        tokens=s["tokens"][0],
        hidden_size=s["hidden_size"],
        intermediate_size=s["intermediate_size"],
        num_experts=s["num_experts"],
        top_k=s["top_k"],
        local_expert_offset=0,
        local_num_experts=s["local_num_experts"],
    )


def make_inputs(
    *,
    preset: str = "smoke",
    tokens: Optional[int] = None,
    hidden_size: Optional[int] = None,
    intermediate_size: Optional[int] = None,
    local_num_experts: Optional[int] = None,
    local_expert_offset: int = 0,
    device: str = "cuda",
    seed: int = 1234,
) -> tuple:
    """Create deterministic FlashInfer-compatible FP4 MoE inputs."""
    base = _resolve_base(preset)
    shape = MoeShape(
        name=base.name,
        tokens=tokens if tokens is not None else base.tokens,
        hidden_size=hidden_size if hidden_size is not None else base.hidden_size,
        intermediate_size=intermediate_size if intermediate_size is not None else base.intermediate_size,
        num_experts=base.num_experts,
        top_k=base.top_k,
        local_expert_offset=local_expert_offset,
        local_num_experts=local_num_experts if local_num_experts is not None else base.local_num_experts,
        routed_scaling_factor=base.routed_scaling_factor,
        routing_method_type=base.routing_method_type,
        n_group=base.n_group,
        topk_group=base.topk_group,
    )
    if shape.hidden_size % 32 != 0 or shape.intermediate_size % 32 != 0:
        raise ValueError("hidden_size and intermediate_size must be divisible by 32")
    if shape.local_num_experts < shape.top_k:
        raise ValueError("local_num_experts must be >= top_k for the deterministic smoke input")

    g = torch.Generator(device=device)
    g.manual_seed(seed + shape.tokens + shape.hidden_size + shape.intermediate_size)
    routing_logits = torch.randn(
        (shape.tokens, shape.num_experts), device=device, dtype=torch.float32, generator=g
    ) * 0.01
    local_end = shape.local_expert_offset + shape.local_num_experts
    routing_logits[:, shape.local_expert_offset:local_end] += 4.0
    routing_logits = routing_logits.to(torch.bfloat16)

    routing_bias = None
    hidden_states = (
        torch.randn((shape.tokens, shape.hidden_size), device=device, dtype=torch.float32, generator=g)
        * 0.05
    ).to(torch.bfloat16)
    hidden_states_scale = _scale_tensor(
        (shape.tokens, shape.hidden_size // 32), device=device, value=1.0
    )

    gemm1_weights = torch.randint(
        0,
        256,
        (shape.local_num_experts, 2 * shape.intermediate_size, shape.hidden_size // 2),
        device=device,
        dtype=torch.uint8,
        generator=g,
    )
    gemm2_weights = torch.randint(
        0,
        256,
        (shape.local_num_experts, shape.hidden_size, shape.intermediate_size // 2),
        device=device,
        dtype=torch.uint8,
        generator=g,
    )
    gemm1_weights_scale = _scale_tensor(
        (
            shape.local_num_experts,
            2 * shape.intermediate_size,
            shape.hidden_size // 32,
        ),
        device=device,
    )
    gemm2_weights_scale = _scale_tensor(
        (
            shape.local_num_experts,
            shape.hidden_size,
            shape.intermediate_size // 32,
        ),
        device=device,
    )

    none = None
    return (
        routing_logits,
        routing_bias,
        hidden_states,
        hidden_states_scale,
        gemm1_weights,
        gemm1_weights_scale,
        none,
        none,
        none,
        none,
        gemm2_weights,
        gemm2_weights_scale,
        none,
        none,
        none,
        none,
        shape.num_experts,
        shape.top_k,
        shape.n_group,
        shape.topk_group,
        shape.intermediate_size,
        shape.local_expert_offset,
        shape.local_num_experts,
        shape.routed_scaling_factor,
        shape.routing_method_type,
        True,
        None,
        3,
        None,
        8192,
        True,
        None,
    )
