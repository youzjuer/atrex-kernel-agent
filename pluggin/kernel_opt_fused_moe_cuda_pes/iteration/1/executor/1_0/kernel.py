"""CUDA fused-MoE candidate entrypoint for the local Full-Agent PES workspace."""

from __future__ import annotations

import os
from pathlib import Path

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
    if "FUSED_MOE_CUDA_ARCH_LIST" in os.environ:
        arch_list = os.environ["FUSED_MOE_CUDA_ARCH_LIST"]
    elif torch.cuda.is_available():
        major, minor = torch.cuda.get_device_capability()
        arch_list = f"{major}.{minor}"
    else:
        arch_list = "8.9"
    os.environ["TORCH_CUDA_ARCH_LIST"] = arch_list
    ext_suffix = arch_list.replace(".", "").replace("+", "p").replace(";", "_").replace(" ", "")
    ext_name = f"fused_moe_cuda_ext_sm{ext_suffix}"
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


@torch.no_grad()
def run(
    x: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
) -> torch.Tensor:
    """Return fused MoE output with shape `[M, H]`."""
    ext = _load_ext()
    return ext.forward(
        x.contiguous(),
        w1.contiguous(),
        w2.contiguous(),
        topk_ids.to(torch.int64).contiguous(),
        topk_weights.contiguous(),
    )
