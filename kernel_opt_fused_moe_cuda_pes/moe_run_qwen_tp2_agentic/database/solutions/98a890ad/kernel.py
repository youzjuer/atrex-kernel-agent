"""CUDA candidate for FlashInfer-aligned TRT-LLM FP4 block-scale MoE."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional

import torch
from torch.utils.cpp_extension import load


_EXT = None


def _workspace_root() -> Path:
    return Path(os.environ.get("FUSED_MOE_CUDA_ROOT", Path(__file__).resolve().parent)).resolve()


def _load_ext():
    global _EXT
    if _EXT is not None:
        return _EXT

    root = _workspace_root()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    if "FUSED_MOE_CUDA_ARCH_LIST" in os.environ:
        arch_list = os.environ["FUSED_MOE_CUDA_ARCH_LIST"]
    elif torch.cuda.is_available():
        major, minor = torch.cuda.get_device_capability()
        arch_list = f"{major}.{minor}"
    else:
        arch_list = "8.9"
    os.environ["TORCH_CUDA_ARCH_LIST"] = arch_list
    ext_suffix = arch_list.replace(".", "").replace("+", "p").replace(";", "_").replace(" ", "")
    ext_name = f"flashinfer_fp4_moe_cuda_ext_sm{ext_suffix}"
    build_dir = root / ".torch_extensions" / ext_name
    build_dir.mkdir(parents=True, exist_ok=True)
    _EXT = load(
        name=ext_name,
        sources=[
            str(root / "src" / "fused_moe_kernel.cpp"),
            str(root / "src" / "fused_moe_kernel.cu"),
        ],
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        build_directory=str(build_dir),
        verbose=os.environ.get("FUSED_MOE_CUDA_VERBOSE", "0") == "1",
    )
    return _EXT


def _routing():
    root = _workspace_root()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from reference import compute_routing

    return compute_routing


def _empty_optional_like(tensor: torch.Tensor, *, dtype: torch.dtype) -> torch.Tensor:
    return torch.empty((0,), device=tensor.device, dtype=dtype)


@torch.no_grad()
def run(
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
        hidden_states_scale,
        gemm1_alpha,
        gemm1_beta,
        gemm1_clamp_limit,
        output1_scale_scalar,
        output1_scale_gate_scalar,
        output2_scale_scalar,
        enable_pdl,
        tune_max_num_tokens,
        norm_topk_prob,
        routing_replay_out,
    )
    if not do_finalize:
        raise NotImplementedError("candidate currently returns finalized output only")
    if activation_type != 3:
        raise NotImplementedError("candidate currently supports activation_type=3 (SwiGLU) only")
    if hidden_states.dtype != torch.bfloat16:
        raise NotImplementedError("candidate currently supports bf16 hidden_states only")
    if local_num_experts != gemm1_weights.shape[0]:
        raise ValueError("local_num_experts must match gemm1_weights.shape[0]")

    compute_routing = _routing()
    topk_idx, topk_weights = compute_routing(
        routing_logits,
        routing_bias,
        top_k=top_k,
        n_group=n_group,
        topk_group=topk_group,
        routed_scaling_factor=routed_scaling_factor,
        routing_method_type=routing_method_type,
    )

    ext = _load_ext()
    empty_bias = _empty_optional_like(hidden_states, dtype=torch.float32)
    gemm1_bias_arg = empty_bias if gemm1_bias is None else gemm1_bias.to(torch.float32).contiguous()
    gemm2_bias_arg = empty_bias if gemm2_bias is None else gemm2_bias.to(torch.float32).contiguous()
    out = ext.forward(
        hidden_states.contiguous(),
        gemm1_weights.contiguous(),
        gemm1_weights_scale.to(torch.float32).contiguous(),
        gemm1_bias_arg,
        gemm2_weights.contiguous(),
        gemm2_weights_scale.to(torch.float32).contiguous(),
        gemm2_bias_arg,
        topk_idx.to(torch.int64).contiguous(),
        topk_weights.to(torch.float32).contiguous(),
        int(num_experts),
        int(local_expert_offset),
        int(intermediate_size),
    )
    if output is not None:
        output.copy_(out)
        out = output
    return [out]
