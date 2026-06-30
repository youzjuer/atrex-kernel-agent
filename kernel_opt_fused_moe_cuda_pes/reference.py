"""PyTorch reference for the local CUDA fused-MoE task."""

from __future__ import annotations

import torch


@torch.no_grad()
def run_reference(
    x: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
) -> torch.Tensor:
    M, H = x.shape
    I = w2.shape[2]
    out = torch.zeros((M, H), device=x.device, dtype=torch.float32)
    for m in range(M):
        xm = x[m].float()
        for k in range(topk_ids.shape[1]):
            e = int(topk_ids[m, k].item())
            gate_up = torch.matmul(w1[e].float(), xm)
            gate = gate_up[:I]
            up = gate_up[I:]
            act = torch.nn.functional.silu(gate) * up
            out[m] += topk_weights[m, k].float() * torch.matmul(w2[e].float(), act)
    return out


def make_inputs(
    *,
    M: int,
    E: int = 8,
    TOPK: int = 2,
    H: int = 64,
    I: int = 32,
    device: str = "cuda",
    seed: int = 1234,
):
    g = torch.Generator(device=device)
    g.manual_seed(seed + M)
    x = torch.randn((M, H), device=device, dtype=torch.float32, generator=g) * 0.1
    w1 = torch.randn((E, 2 * I, H), device=device, dtype=torch.float32, generator=g) * 0.1
    w2 = torch.randn((E, H, I), device=device, dtype=torch.float32, generator=g) * 0.1
    topk_ids = torch.randint(0, E, (M, TOPK), device=device, dtype=torch.int64, generator=g)
    weights = torch.rand((M, TOPK), device=device, dtype=torch.float32, generator=g)
    topk_weights = weights / weights.sum(dim=1, keepdim=True)
    return x, w1, w2, topk_ids, topk_weights
