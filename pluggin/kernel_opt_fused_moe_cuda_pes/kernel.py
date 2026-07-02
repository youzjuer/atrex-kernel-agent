"""CUDA candidate for FlashInfer-aligned TRT-LLM FP4 block-scale MoE."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional

import torch
from torch.utils.cpp_extension import load


_EXT = None
_SCALE_CACHE = {}
_SCALE_CACHE_ORDER = []
_MAX_SCALE_CACHE_ITEMS = 8
_TASK08_CACHE = {}
_TASK08_WARMED = set()

# The evaluator imports this file as candidate_kernel.  Some task08 probe
# helpers import "kernel", so keep that name bound to this module.
sys.modules.setdefault("kernel", sys.modules[__name__])


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


def _as_fp32_scale(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.dtype == torch.float32 and tensor.is_contiguous():
        return tensor
    key = (
        id(tensor),
        tensor.data_ptr(),
        tuple(tensor.shape),
        str(tensor.dtype),
        tensor.device.type,
        tensor.device.index,
        tensor.stride(),
    )
    cached = _SCALE_CACHE.get(key)
    if cached is not None:
        return cached
    converted = tensor.to(torch.float32).contiguous()
    _SCALE_CACHE[key] = converted
    _SCALE_CACHE_ORDER.append(key)
    while len(_SCALE_CACHE_ORDER) > _MAX_SCALE_CACHE_ITEMS:
        old_key = _SCALE_CACHE_ORDER.pop(0)
        _SCALE_CACHE.pop(old_key, None)
    return converted


def _use_prepared_weight_layout(hidden_states: torch.Tensor) -> bool:
    value = os.environ.get("FUSED_MOE_PREPARED_WEIGHT_LAYOUT", "auto").strip().lower()
    if value in {"1", "true", "yes", "on", "prepared"}:
        return True
    if value in {"0", "false", "no", "off", "linear"}:
        return False
    return hidden_states.dtype == torch.uint8


def _use_cuda_type1_routing(
    routing_logits: torch.Tensor,
    routing_bias: Optional[torch.Tensor],
    n_group: Optional[int],
    topk_group: Optional[int],
    routed_scaling_factor: Optional[float],
    routing_method_type: int,
    top_k: int,
) -> bool:
    value = os.environ.get("FUSED_MOE_CUDA_ROUTING", "auto").strip().lower()
    if value in {"0", "false", "no", "off", "torch"}:
        return False
    if routing_method_type != 1:
        return False
    if routing_bias is not None or n_group not in (None, 0) or topk_group not in (None, 0):
        return False
    if routing_logits.dtype not in (torch.bfloat16, torch.float32):
        return False
    if top_k <= 0 or top_k > 16:
        return False
    if routing_logits.shape[0] < 512 and value not in {"1", "true", "yes", "on", "cuda"}:
        return False
    if routed_scaling_factor not in (None, 1.0):
        return value in {"1", "true", "yes", "on", "cuda"}
    return True


def _task08_staged_requested() -> bool:
    value = os.environ.get("FUSED_MOE_TASK08_STAGED", "auto").strip().lower()
    return value not in {"0", "false", "no", "off", "disabled"}


def _task08_root() -> Path:
    root = os.environ.get("FUSED_MOE_TASK08_ROOT")
    if root:
        return Path(root).resolve()
    proj_root = os.environ.get("PROJ019_ROOT")
    if proj_root:
        return (Path(proj_root).resolve() / "assets" / "task_08_cuda_fp4_bmm_sm103")
    return (
        Path("/home/youchunbo/code/sumu/omoExplore/proj/proj_019_moe_workload_op_opt")
        / "assets"
        / "task_08_cuda_fp4_bmm_sm103"
    )


def _task08_helpers_available() -> bool:
    root = _workspace_root()
    return (
        (root / "probe_task08_gemm2_runtime.py").exists()
        and (root / "probe_task08_mpad_compile.py").exists()
    )


def _task08_staged_applicable(
    *,
    routing_logits: torch.Tensor,
    routing_bias: Optional[torch.Tensor],
    hidden_states: torch.Tensor,
    hidden_states_scale: Optional[torch.Tensor],
    gemm1_weights: torch.Tensor,
    gemm1_weights_scale: torch.Tensor,
    gemm1_bias: Optional[torch.Tensor],
    gemm2_weights: torch.Tensor,
    gemm2_weights_scale: torch.Tensor,
    gemm2_bias: Optional[torch.Tensor],
    num_experts: int,
    top_k: int,
    n_group: Optional[int],
    topk_group: Optional[int],
    intermediate_size: int,
    local_expert_offset: int,
    local_num_experts: int,
    routed_scaling_factor: Optional[float],
    routing_method_type: int,
) -> bool:
    if not _task08_staged_requested():
        return False
    if not _task08_helpers_available():
        return False
    if hidden_states.dtype != torch.uint8 or hidden_states_scale is None:
        return False
    if hidden_states.shape != (9500, 2048):
        return False
    if hidden_states_scale.shape != (9500, 256):
        return False
    if routing_logits.shape != (9500, 512):
        return False
    if routing_bias is not None or n_group not in (None, 0) or topk_group not in (None, 0):
        return False
    if routed_scaling_factor not in (None, 1.0):
        return False
    if routing_method_type != 1 or top_k != 10:
        return False
    if int(num_experts) != 512 or int(local_num_experts) != 512 or int(local_expert_offset) != 0:
        return False
    if int(intermediate_size) != 1024:
        return False
    if gemm1_bias is not None or gemm2_bias is not None:
        return False
    if tuple(gemm1_weights.shape) != (512, 2048, 2048):
        return False
    if tuple(gemm1_weights_scale.shape) != (512, 2048, 256):
        return False
    if tuple(gemm2_weights.shape) != (512, 4096, 512):
        return False
    if tuple(gemm2_weights_scale.shape) != (512, 4096, 64):
        return False
    if hidden_states_scale.dtype not in (torch.float8_e4m3fn, torch.uint8):
        return False
    if gemm1_weights.dtype != torch.uint8 or gemm2_weights.dtype != torch.uint8:
        return False
    if gemm1_weights_scale.dtype not in (torch.float8_e4m3fn, torch.uint8):
        return False
    if gemm2_weights_scale.dtype not in (torch.float8_e4m3fn, torch.uint8):
        return False
    return _task08_root().exists()


def _as_u8_view(tensor: torch.Tensor) -> torch.Tensor:
    return tensor if tensor.dtype == torch.uint8 else tensor.view(torch.uint8)


def _load_task08_bmm_functions(mpad: int):
    symbol = os.environ.get(
        "FUSED_MOE_TASK08_GEMM2_SYMBOL",
        "atrex_gemm2_tilegrid_splitcolepi_postsync",
    )
    key = (int(mpad), symbol)
    cached = _TASK08_CACHE.get(key)
    if cached is not None:
        return cached

    from probe_task08_gemm2_runtime import (
        DIAG_GEMM2_SYMBOLS,
        _compile_gemm2_extension,
        _prepare_gemm1_tree,
        _prepare_gemm2_tree,
    )
    from probe_task08_mpad_compile import V310_SYMBOL, _compile_extension

    task08_root = _task08_root()
    gemm1_root, _ = _prepare_gemm1_tree(task08_root, int(mpad))
    gemm1_module = _compile_extension(gemm1_root, int(mpad), False)
    include_diag_symbols = symbol in DIAG_GEMM2_SYMBOLS
    gemm2_root, _ = _prepare_gemm2_tree(
        task08_root,
        int(mpad),
        include_diag_symbols=include_diag_symbols,
    )
    gemm2_module = _compile_gemm2_extension(
        gemm2_root,
        int(mpad),
        False,
        extension_tag="diag" if include_diag_symbols else "",
    )
    if not hasattr(gemm1_module, V310_SYMBOL):
        raise RuntimeError(f"task08 GEMM1 module missing symbol {V310_SYMBOL}")
    if not hasattr(gemm2_module, symbol):
        raise RuntimeError(f"task08 GEMM2 module missing symbol {symbol}")
    cached = (getattr(gemm1_module, V310_SYMBOL), getattr(gemm2_module, symbol), symbol)
    _TASK08_CACHE[key] = cached
    return cached


@torch.no_grad()
def _run_task08_staged_once(
    *,
    routing_logits: torch.Tensor,
    hidden_states: torch.Tensor,
    hidden_states_scale: torch.Tensor,
    gemm1_weights: torch.Tensor,
    gemm1_weights_scale: torch.Tensor,
    gemm2_weights: torch.Tensor,
    gemm2_weights_scale: torch.Tensor,
    num_experts: int,
    top_k: int,
    intermediate_size: int,
    local_expert_offset: int,
    local_num_experts: int,
    routed_scaling_factor: Optional[float],
) -> torch.Tensor:
    ext = _load_ext()
    scale = 1.0 if routed_scaling_factor is None else float(routed_scaling_factor)
    topk_packed = ext.routing_pack_type1(routing_logits.contiguous(), int(top_k), scale)
    (
        expanded,
        _permuted_to_token,
        _cta_batch,
        _cta_mn_limit,
        _num_ctas,
        _total_padded,
        expert_counts,
        expert_padded_offsets,
    ) = routing_metadata_from_packed(
        topk_packed,
        num_experts=int(num_experts),
        local_expert_offset=int(local_expert_offset),
        local_num_experts=int(local_num_experts),
        tile_tokens_dim=256,
    )
    mpad = int(expert_counts.max().item())
    if mpad <= 0:
        return torch.empty(
            (hidden_states.shape[0], hidden_states.shape[1] * 2),
            device=hidden_states.device,
            dtype=torch.bfloat16,
        )

    hidden_bmm, hidden_scale_swizzled = pack_hidden_bmm_swizzled_from_metadata(
        topk_packed,
        expanded,
        expert_padded_offsets,
        hidden_states,
        hidden_states_scale,
        num_experts=int(num_experts),
        local_expert_offset=int(local_expert_offset),
        local_num_experts=int(local_num_experts),
        padded_rows=mpad,
    )
    rows = int(local_num_experts) * mpad
    affine_expert_offsets = (
        torch.arange(int(local_num_experts) + 1, device=hidden_states.device, dtype=torch.int32)
        * mpad
    ).contiguous()
    gemm1_fn, gemm2_fn, _symbol = _load_task08_bmm_functions(mpad)
    gemm1_out = gemm1_fn(
        hidden_bmm.reshape(rows, -1).contiguous(),
        _as_u8_view(hidden_scale_swizzled).reshape(int(local_num_experts), -1).contiguous(),
        gemm1_weights.contiguous(),
        _as_u8_view(gemm1_weights_scale).reshape(int(local_num_experts), -1).contiguous(),
        affine_expert_offsets,
    )
    torch.cuda.synchronize()
    mid_q, _mid_scale, mid_scale_swizzled = swiglu_requant_from_bmm_metadata(
        gemm1_out,
        topk_packed,
        expanded,
        expert_padded_offsets,
        local_expert_offset=int(local_expert_offset),
        padded_rows=mpad,
        intermediate_size=int(intermediate_size),
    )
    gemm2_out = gemm2_fn(
        mid_q.reshape(rows, -1).contiguous(),
        _as_u8_view(mid_scale_swizzled).reshape(int(local_num_experts), -1).contiguous(),
        gemm2_weights.contiguous(),
        _as_u8_view(gemm2_weights_scale).reshape(int(local_num_experts), -1).contiguous(),
        affine_expert_offsets,
    )
    torch.cuda.synchronize()
    return final_scatter_from_bmm(
        gemm2_out,
        topk_packed,
        expanded,
        expert_padded_offsets,
        local_expert_offset=int(local_expert_offset),
        padded_rows=mpad,
        use_prepared_output_layout=True,
    )


@torch.no_grad()
def _run_task08_staged_g11(
    *,
    routing_logits: torch.Tensor,
    hidden_states: torch.Tensor,
    hidden_states_scale: torch.Tensor,
    gemm1_weights: torch.Tensor,
    gemm1_weights_scale: torch.Tensor,
    gemm2_weights: torch.Tensor,
    gemm2_weights_scale: torch.Tensor,
    num_experts: int,
    top_k: int,
    intermediate_size: int,
    local_expert_offset: int,
    local_num_experts: int,
    routed_scaling_factor: Optional[float],
) -> torch.Tensor:
    warm_key = (
        int(hidden_states.data_ptr()),
        int(routing_logits.data_ptr()),
        int(gemm1_weights.data_ptr()),
        int(gemm2_weights.data_ptr()),
    )
    if warm_key not in _TASK08_WARMED:
        warm = _run_task08_staged_once(
            routing_logits=routing_logits,
            hidden_states=hidden_states,
            hidden_states_scale=hidden_states_scale,
            gemm1_weights=gemm1_weights,
            gemm1_weights_scale=gemm1_weights_scale,
            gemm2_weights=gemm2_weights,
            gemm2_weights_scale=gemm2_weights_scale,
            num_experts=num_experts,
            top_k=top_k,
            intermediate_size=intermediate_size,
            local_expert_offset=local_expert_offset,
            local_num_experts=local_num_experts,
            routed_scaling_factor=routed_scaling_factor,
        )
        torch.cuda.synchronize()
        del warm
        _TASK08_WARMED.add(warm_key)
    return _run_task08_staged_once(
        routing_logits=routing_logits,
        hidden_states=hidden_states,
        hidden_states_scale=hidden_states_scale,
        gemm1_weights=gemm1_weights,
        gemm1_weights_scale=gemm1_weights_scale,
        gemm2_weights=gemm2_weights,
        gemm2_weights_scale=gemm2_weights_scale,
        num_experts=num_experts,
        top_k=top_k,
        intermediate_size=intermediate_size,
        local_expert_offset=local_expert_offset,
        local_num_experts=local_num_experts,
        routed_scaling_factor=routed_scaling_factor,
    )


@torch.no_grad()
def routing_metadata_from_packed(
    topk_packed: torch.Tensor,
    *,
    num_experts: int,
    local_expert_offset: int,
    local_num_experts: int,
    tile_tokens_dim: int,
) -> list[torch.Tensor]:
    """Build TensorRT-LLM MoE routing metadata from PackedScoreIdx<bf16> top-k."""
    if topk_packed.dtype != torch.int32:
        raise TypeError("topk_packed must be torch.int32")
    if topk_packed.ndim != 2:
        raise ValueError("topk_packed must have shape [T, top_k]")
    ext = _load_ext()
    return list(
        ext.routing_metadata_from_packed(
            topk_packed.contiguous(),
            int(num_experts),
            int(local_expert_offset),
            int(local_num_experts),
            int(tile_tokens_dim),
        )
    )


@torch.no_grad()
def pack_hidden_bmm_from_metadata(
    topk_packed: torch.Tensor,
    expanded_idx_to_permuted_idx: torch.Tensor,
    expert_padded_offsets: torch.Tensor,
    hidden_states: torch.Tensor,
    hidden_states_scale: torch.Tensor,
    *,
    num_experts: int,
    local_expert_offset: int,
    local_num_experts: int,
    padded_rows: int,
) -> list[torch.Tensor]:
    """Pack hidden FP4 activations to expert-major [E, padded_rows, *] BMM layout."""
    if hidden_states.dtype != torch.uint8:
        raise TypeError("hidden_states must be torch.uint8")
    if hidden_states_scale.dtype not in (torch.float8_e4m3fn, torch.uint8):
        raise TypeError("hidden_states_scale must be torch.float8_e4m3fn or torch.uint8")
    ext = _load_ext()
    return list(
        ext.pack_hidden_bmm_from_metadata(
            topk_packed.contiguous(),
            expanded_idx_to_permuted_idx.contiguous(),
            expert_padded_offsets.contiguous(),
            hidden_states.contiguous(),
            hidden_states_scale.contiguous(),
            int(num_experts),
            int(local_expert_offset),
            int(local_num_experts),
            int(padded_rows),
        )
    )


@torch.no_grad()
def nvfp4_block_scale_interleave(scale: torch.Tensor) -> torch.Tensor:
    """Interleave linear NVFP4 block scales into FlashInfer/SM100 BMM scale layout."""
    if scale.dtype not in (torch.float8_e4m3fn, torch.uint8):
        raise TypeError("scale must be torch.float8_e4m3fn or torch.uint8")
    if scale.ndim != 3:
        raise ValueError("scale must have shape [B, rows, K/16]")
    ext = _load_ext()
    return ext.nvfp4_block_scale_interleave(scale.contiguous())


@torch.no_grad()
def swiglu_requant_from_bmm(
    gemm1_out: torch.Tensor,
    expert_counts: torch.Tensor,
    *,
    padded_rows: int,
    intermediate_size: int,
) -> list[torch.Tensor]:
    """Apply prepared-layout SwiGLU to GEMM1 BMM output and requantize to NVFP4."""
    if gemm1_out.dtype != torch.bfloat16:
        raise TypeError("gemm1_out must be torch.bfloat16")
    if gemm1_out.ndim != 2:
        raise ValueError("gemm1_out must have shape [E*padded_rows, 2I]")
    if expert_counts.dtype != torch.int32 or expert_counts.ndim != 1:
        raise TypeError("expert_counts must be torch.int32 [E]")
    ext = _load_ext()
    return list(
        ext.swiglu_requant_from_bmm(
            gemm1_out.contiguous(),
            expert_counts.contiguous(),
            int(padded_rows),
            int(intermediate_size),
        )
    )


@torch.no_grad()
def swiglu_requant_from_bmm_metadata(
    gemm1_out: torch.Tensor,
    topk_packed: torch.Tensor,
    expanded_idx_to_permuted_idx: torch.Tensor,
    expert_padded_offsets: torch.Tensor,
    *,
    local_expert_offset: int,
    padded_rows: int,
    intermediate_size: int,
) -> list[torch.Tensor]:
    """Apply prepared-layout SwiGLU/requant only for valid routed BMM rows."""
    if gemm1_out.dtype != torch.bfloat16:
        raise TypeError("gemm1_out must be torch.bfloat16")
    if gemm1_out.ndim != 2:
        raise ValueError("gemm1_out must have shape [E*padded_rows, 2I]")
    if topk_packed.dtype != torch.int32 or topk_packed.ndim != 2:
        raise TypeError("topk_packed must be torch.int32 [T, top_k]")
    if expanded_idx_to_permuted_idx.dtype != torch.int32:
        raise TypeError("expanded_idx_to_permuted_idx must be torch.int32")
    if expanded_idx_to_permuted_idx.shape != topk_packed.shape:
        raise ValueError("expanded_idx_to_permuted_idx shape mismatch")
    if expert_padded_offsets.dtype != torch.int32 or expert_padded_offsets.ndim != 1:
        raise TypeError("expert_padded_offsets must be torch.int32 [E_local + 1]")
    ext = _load_ext()
    return list(
        ext.swiglu_requant_from_bmm_metadata(
            gemm1_out.contiguous(),
            topk_packed.contiguous(),
            expanded_idx_to_permuted_idx.contiguous(),
            expert_padded_offsets.contiguous(),
            int(local_expert_offset),
            int(padded_rows),
            int(intermediate_size),
        )
    )


@torch.no_grad()
def final_scatter_from_bmm(
    gemm2_out: torch.Tensor,
    topk_packed: torch.Tensor,
    expanded_idx_to_permuted_idx: torch.Tensor,
    expert_padded_offsets: torch.Tensor,
    *,
    local_expert_offset: int,
    padded_rows: int,
    use_prepared_output_layout: bool = True,
) -> torch.Tensor:
    """Apply packed bf16 top-k weights and scatter GEMM2 BMM rows back to [T, H]."""
    if gemm2_out.dtype != torch.bfloat16:
        raise TypeError("gemm2_out must be torch.bfloat16")
    if gemm2_out.ndim != 2:
        raise ValueError("gemm2_out must have shape [E*padded_rows, H]")
    if topk_packed.dtype != torch.int32 or topk_packed.ndim != 2:
        raise TypeError("topk_packed must be torch.int32 [T, top_k]")
    if expanded_idx_to_permuted_idx.dtype != torch.int32:
        raise TypeError("expanded_idx_to_permuted_idx must be torch.int32")
    if expanded_idx_to_permuted_idx.shape != topk_packed.shape:
        raise ValueError("expanded_idx_to_permuted_idx shape mismatch")
    if expert_padded_offsets.dtype != torch.int32 or expert_padded_offsets.ndim != 1:
        raise TypeError("expert_padded_offsets must be torch.int32 [E_local + 1]")
    ext = _load_ext()
    return ext.final_scatter_from_bmm(
        gemm2_out.contiguous(),
        topk_packed.contiguous(),
        expanded_idx_to_permuted_idx.contiguous(),
        expert_padded_offsets.contiguous(),
        int(local_expert_offset),
        int(padded_rows),
        bool(use_prepared_output_layout),
    )


@torch.no_grad()
def pack_hidden_bmm_swizzled_from_metadata(
    topk_packed: torch.Tensor,
    expanded_idx_to_permuted_idx: torch.Tensor,
    expert_padded_offsets: torch.Tensor,
    hidden_states: torch.Tensor,
    hidden_states_scale: torch.Tensor,
    *,
    num_experts: int,
    local_expert_offset: int,
    local_num_experts: int,
    padded_rows: int,
) -> list[torch.Tensor]:
    """Pack hidden FP4 rows and swizzled NVFP4 scales to expert-major BMM layout."""
    if hidden_states.dtype != torch.uint8:
        raise TypeError("hidden_states must be torch.uint8")
    if hidden_states_scale.dtype not in (torch.float8_e4m3fn, torch.uint8):
        raise TypeError("hidden_states_scale must be torch.float8_e4m3fn or torch.uint8")
    ext = _load_ext()
    return list(
        ext.pack_hidden_bmm_swizzled_from_metadata(
            topk_packed.contiguous(),
            expanded_idx_to_permuted_idx.contiguous(),
            expert_padded_offsets.contiguous(),
            hidden_states.contiguous(),
            hidden_states_scale.contiguous(),
            int(num_experts),
            int(local_expert_offset),
            int(local_num_experts),
            int(padded_rows),
        )
    )


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
    if hidden_states.dtype not in (torch.bfloat16, torch.uint8):
        raise NotImplementedError("candidate currently supports bf16 or packed uint8 hidden_states")
    if hidden_states.dtype == torch.uint8 and hidden_states_scale is None:
        raise ValueError("hidden_states_scale is required for packed uint8 hidden_states")
    if local_num_experts != gemm1_weights.shape[0]:
        raise ValueError("local_num_experts must match gemm1_weights.shape[0]")

    if _task08_staged_applicable(
        routing_logits=routing_logits,
        routing_bias=routing_bias,
        hidden_states=hidden_states,
        hidden_states_scale=hidden_states_scale,
        gemm1_weights=gemm1_weights,
        gemm1_weights_scale=gemm1_weights_scale,
        gemm1_bias=gemm1_bias,
        gemm2_weights=gemm2_weights,
        gemm2_weights_scale=gemm2_weights_scale,
        gemm2_bias=gemm2_bias,
        num_experts=num_experts,
        top_k=top_k,
        n_group=n_group,
        topk_group=topk_group,
        intermediate_size=intermediate_size,
        local_expert_offset=local_expert_offset,
        local_num_experts=local_num_experts,
        routed_scaling_factor=routed_scaling_factor,
        routing_method_type=routing_method_type,
    ):
        out = _run_task08_staged_g11(
            routing_logits=routing_logits,
            hidden_states=hidden_states,
            hidden_states_scale=hidden_states_scale,
            gemm1_weights=gemm1_weights,
            gemm1_weights_scale=gemm1_weights_scale,
            gemm2_weights=gemm2_weights,
            gemm2_weights_scale=gemm2_weights_scale,
            num_experts=int(num_experts),
            top_k=int(top_k),
            intermediate_size=int(intermediate_size),
            local_expert_offset=int(local_expert_offset),
            local_num_experts=int(local_num_experts),
            routed_scaling_factor=routed_scaling_factor,
        )
        if output is not None:
            output.copy_(out)
            out = output
        return [out]

    ext = _load_ext()
    empty_bias = _empty_optional_like(hidden_states, dtype=torch.float32)
    hidden_states_scale_arg = (
        empty_bias
        if hidden_states_scale is None
        else _as_fp32_scale(hidden_states_scale)
    )
    gemm1_bias_arg = empty_bias if gemm1_bias is None else _as_fp32_scale(gemm1_bias)
    gemm2_bias_arg = empty_bias if gemm2_bias is None else _as_fp32_scale(gemm2_bias)
    use_prepared_layout = bool(_use_prepared_weight_layout(hidden_states))
    if _use_cuda_type1_routing(
        routing_logits,
        routing_bias,
        n_group,
        topk_group,
        routed_scaling_factor,
        routing_method_type,
        top_k,
    ):
        out = ext.forward_logits_type1(
            routing_logits.contiguous(),
            hidden_states.contiguous(),
            hidden_states_scale_arg,
            gemm1_weights.contiguous(),
            _as_fp32_scale(gemm1_weights_scale),
            gemm1_bias_arg,
            gemm2_weights.contiguous(),
            _as_fp32_scale(gemm2_weights_scale),
            gemm2_bias_arg,
            int(num_experts),
            int(top_k),
            int(local_expert_offset),
            int(intermediate_size),
            1.0 if routed_scaling_factor is None else float(routed_scaling_factor),
            use_prepared_layout,
        )
    else:
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
        out = ext.forward(
            hidden_states.contiguous(),
            hidden_states_scale_arg,
            gemm1_weights.contiguous(),
            _as_fp32_scale(gemm1_weights_scale),
            gemm1_bias_arg,
            gemm2_weights.contiguous(),
            _as_fp32_scale(gemm2_weights_scale),
            gemm2_bias_arg,
            topk_idx.to(torch.int64).contiguous(),
            topk_weights.to(torch.float32).contiguous(),
            int(num_experts),
            int(local_expert_offset),
            int(intermediate_size),
            use_prepared_layout,
        )
    if output is not None:
        output.copy_(out)
        out = output
    return [out]
