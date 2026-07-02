# Iteration 2 Plan

## Input Evidence

- Selected parents: `7ea9f8ad` baseline and `3c90a4b6` grouped GEMM child.
- Current profile: `profiles/auto_baseline.json` reports `qwen_tp2`, `T=1`, `H=4096`, `I=1024`, `E_local=256`, `top_k=10`, baseline latency `1438.912 us`, max_rel `1.192e-4`; FlashInfer comparison is still blocked by a worker segfault.
- Generation 1 results: `1_2` grouped GEMM passes at `1504.352 us`, `1_1` tiled FP4 dequant passes at `1897.984 us`, and `1_0` grouped expert scheduling passes at `3136.896 us`. None beat the baseline.
- New parent-code evidence: `3c90a4b6` removed the standalone finalize launch, but its grouped GEMM2 stage launches over every local expert. For the current shape, `ceil(H/32) * E_local = 128 * 256 = 32768` stage2 blocks, while `T * top_k = 10` routed slots allow at most `1280` active expert/hidden-column blocks.
- Hardware note: local `nvidia-smi` reports L20D compute capability `8.9`; gpu-wiki's compute-capability table lists BF16/FP8 support for CC 8.9 but not FP4 tensor-core support. Do not assume native FP4 MMA for this run.

## Already Tried

| Child | Category | Outcome | Novelty constraint for iteration 2 |
|---|---|---:|---|
| `1_0` | `grouped_expert_scheduling` | `3136.896 us` | Do not repeat full E_local metadata generation with compute fallback unchanged. |
| `1_1` | `tiled_fp4_dequantization` | `1897.984 us` | Do not repeat GEMM1-only shared hidden tile plus 128-bit load experiment. |
| `1_2` | `grouped_gemm` | `1504.352 us` | Do not repeat E_local-wide expert-major GEMM2/finalize with completion counters. |

## Search Log

| Source | Layer | New? | Finding |
|---|---:|---|---|
| `database/solutions/3c90a4b6/src/fused_moe_kernel.cu` | workspace | Yes | Grouped parent schedules GEMM2 over all 256 local experts even when only top-k slots are active. |
| `iteration/1/executor/1_2/result.json` | workspace | Yes | Grouped finalize parent passes but regresses to `1504.352 us`. |
| `gpu-wiki/docs/ref-docs/nvidia/cutedsl/sm120/sm120-moe-data-prep-optimization.md` | L1 | Yes | Small-route data prep should be single-CTA; V7 per-CTA offsets remove hot global scatter atomics; sort/warp-specialized fixes regressed. |
| `gpu-wiki/reference-kernels/nvidia/blackwell-geforce/cutedsl/moe_data_prep/fused_moe_data_v9.py` | L1 | Yes | Concrete metadata layout for counts, offsets, `a_map`, `c_map`, and problem sizes without CUB sort. |
| `gpu-wiki/docs/kernel-opt/nvidia/common/blackwell/techniques/cache-policy.md` | L1 | Yes | Use cache policy separation for streamed vs reused FP4 data. |
| `gpu-wiki/docs/ref-docs/nvidia/cuda/sm120/sm120-nvfp4-split-k-gemv-bf16-optimization.md` | L1 | Yes | Split-K should first use FP32 partial workspace plus deterministic reduce, gated by shape/alignment. |
| `gpu-wiki/docs/ref-docs/nvidia/cuda/sm120/sm120-nvfp4-decode-gemm-production-lessons.md` | L1 | Yes | Split-K is a parallelism fix, not a universal route; workspace and graph stability matter. |
| `gpu-wiki/docs/kernel-opt/nvidia/common/blackwell/kernels/grouped-gemm.md` | L1 | Yes | MoE grouped GEMM is variable-M small-GEMM batching; static schedules work when group sizes are known. |
| `reference-projects/flashinfer`, `DeepGEMM`, `aiter` | L2 | Checked | Submodule directories are present but no relevant checked-out files are available; use gpu-wiki extracted references. |

## Strategy 2_0

- Parent: `3c90a4b6`
- Action category: `grouped_expert_scheduling`
- Action: Replace the parent grouped-down scheduler's E_local-wide launch with active-expert scheduling. Build compact `active_experts`, active offsets/counts, `grouped_slots`, and `token_counts` from the local top-k slots in one small CUDA metadata kernel, then launch grouped GEMM2/finalize over `active_expert_count` instead of `E_local=256`.
- Evidence chain: `3c90a4b6` is slower than baseline and launches up to `32768` stage2 blocks despite at most `10` routed slots. The SM120 MoE data-prep recipe supports single-CTA small-route scheduling and warns against heavy sort/CUB paths.
- Expected impact: Prunes zero-expert work while preserving the parent arithmetic and fused finalize boundary. The stage2 expert-grid reduction is up to `25.6x` before metadata overhead.
- Risks: Metadata overhead can dominate at `topk_length=10`. Keep the active list fixed-size/single-CTA, avoid sort, and validate duplicate/non-local expert semantics.

## Strategy 2_1

- Parent: `3c90a4b6`
- Action category: `tiled_fp4_dequantization`
- Action: Retile only GEMM2 FP4 dequant in the grouped parent. Replace `accumulate_fp4_scaled_tile` for GEMM2 with an I-tile microkernel using aligned 128/256-bit packed FP4 loads, vectorized scale loads, PTX byte extraction or branchless nibble decode, and cache-policy separation for streamed weights versus reused `mid`.
- Evidence chain: Generation 1's dequant child optimized GEMM1 only and regressed; GEMM2 in the selected grouped parent remains scalar. gpu-wiki cache-policy/vectorized-load guidance and FlashInfer FP4 helpers point to wide loads and careful scale handling as the viable CC 8.9 software-dequant route.
- Expected impact: Reduces GEMM2 load/decode instruction overhead without changing scheduling behavior.
- Risks: Alignment, register pressure, and PTX qualifier support are fragile. Keep scalar fallback for unaligned tails and do not assume native FP4 conversion on CC 8.9.

## Strategy 2_2

- Parent: `7ea9f8ad`
- Action category: `grouped_gemm`
- Action: Rewrite baseline GEMM2 as a topk-slot grouped split-K GEMM with deterministic reduction. Phase 1 writes FP32 partials for each routed slot, hidden tile, and K split into workspace. Phase 2 reduces splits and top-k contributions per token/hidden column, applies router weights and bias/final cast, and removes the `out_accum` memset plus standalone finalize pass.
- Evidence chain: Baseline is still fastest but uses split-count atomics into `out_accum` followed by a finalize kernel. Generation 1's E_local expert-major grouped GEMM regressed. The split-K GEMV reference recommends shape-gated FP32 partial workspace plus reduce before atomics.
- Expected impact: Keeps useful K parallelism while removing atomics and finalize/memset overhead, using a grouped GEMM boundary distinct from the prior expert-major rewrite.
- Risks: Workspace traffic can outweigh atomic removal for `T=1`. Gate split count by alignment/work count and retain the baseline atomic stage2 as rollback.

## Exhaustion

`false`. New actionable evidence was found in the generation-1 grouped parent code/profile and in local gpu-wiki split-K/cache-policy/data-prep references.
