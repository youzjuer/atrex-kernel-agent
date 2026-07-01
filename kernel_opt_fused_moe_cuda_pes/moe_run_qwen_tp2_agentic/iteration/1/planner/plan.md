# Iteration 1 Plan

## Input Evidence

- Parent lineage: only `7ea9f8ad`, action category `baseline`; no prior non-baseline optimization has been tried.
- Profile: `profiles/auto_baseline.json` reports PASS for `qwen_tp2`, `T=1`, `H=4096`, `I=1024`, `E_local=256`, `top_k=10`, latency `1438.912 us`, max_abs `1.19e-7`.
- Baseline CUDA structure: `stage1_activation_kernel` computes scalar gate/up dots for each `(topk slot, I)`, materializes `mid`; `stage2_grouped_down_kernel` computes scalar down-projection dots for each `(topk slot, H, split)` and uses `atomicAdd`; `finalize_output_kernel` casts `out_accum` to bf16.
- Bottleneck inference: no expert-major schedule, repeated hidden/mid rereads, repeated FP4 nibble decode/scale loads, scalar GEMV-style loops, split atomics, and a separate finalize launch dominate the current staged implementation.
- Hardware note: the gpu-wiki compute-capability table covers CC 8.9 as BF16/FP8 capable but not native FP4 tensor-core capable. The local wiki does not include a specific L20D peak table, so this plan makes no peak-utilization claim.

## Search Log

| Source | Layer | Query | Finding | New? | Actionability |
|---|---:|---|---|---|---|
| workspace `README.md` | workspace | operator contract | FlashInfer FP4 MoE, Qwen TP2 shape, staged CUDA baseline | Yes | Constrains all strategies |
| `profiles/auto_baseline.json` | workspace | baseline profile | PASS, `1438.912 us`, FlashInfer target blocked by segfault | Yes | Current latency/correctness target |
| `evolution_db.py lineage` | workspace | already tried | only `baseline` | Yes | Novelty check |
| parent `src/fused_moe_kernel.cu` | workspace | code bottleneck | scalar stage1, scalar stage2, split atomics, separate finalize | Yes | Direct optimization targets |
| `gpu-wiki/docs/kernel-opt/nvidia/common/nvidia-compute-capabilities.md` | L1 | CC 8.9 support | BF16/FP8 supported, FP4 tensor core only listed for CC 10.x | Yes | Avoid assuming native FP4 tensor cores |
| `gpu-wiki/docs/kernel-opt/nvidia/cutedsl/sm120/sm120-moe-data-prep.md` | L1 | MoE data prep | single-CTA path, expert offsets, a_map/c_map, avoid CUB sort | Yes | Strategy 1 |
| `gpu-wiki/reference-kernels/nvidia/blackwell-geforce/cutedsl/moe_data_prep/README.md` | L1 | scheduling implementation | V9 outputs expert metadata with contention-free scatter | Yes | Strategy 1 schema |
| `gpu-wiki/docs/kernel-opt/nvidia/common/blackwell/techniques/vectorized-loads.md` | L1 | FP4 vectorized loads | sub-byte FP4 benefits from wide vector loads/cache policy | Yes | Strategy 2 |
| `gpu-wiki/reference-kernels/nvidia/blackwell/cutedsl/flashinfer/fp4_common.py` | L1 | FlashInfer FP4 utilities | `SF_VEC_SIZE=16`, `COPY_BITS=128`, `ld_global_v4_u32` | Yes | Strategy 2 |
| `gpu-wiki/reference-kernels/nvidia/blackwell/cutedsl/flashinfer/blockscaled_contiguous_gather_grouped_gemm_swiglu_fusion.py` | L1 | grouped GEMM SwiGLU | token gather, valid-M grouping, tile-to-expert mapping | Yes | Grouped GEMM layout evidence |
| `gpu-wiki/reference-kernels/nvidia/blackwell/cutedsl/flashinfer/blockscaled_contiguous_grouped_gemm_finalize_fusion.py` | L1 | grouped GEMM finalize | router-scale and finalize/scatter fused into GEMM epilogue | Yes | Strategy 3 |
| `gpu-wiki/docs/kernel-opt/generic/hands-on/grouped-gemm-deepgemm.md` | L1 | grouped GEMM pattern | MoE is a variable small-GEMM batching use case | Yes | Strategy 3 |
| `reference-projects/README.md` | L2 | upstream refs | flashinfer submodule has no files locally; gpu-wiki extracted refs are usable | Yes | L2 checked |

## Strategy 1: child `1_0`

- Parent: `7ea9f8ad`
- Action category: `grouped_expert_scheduling`
- Action: Add a CUDA scheduling/data-prep path that converts top-k slots into expert-major metadata: int32 topk slots, per-local-expert counts/prefix offsets, `a_map`, `c_map`, `tile_idx_to_expert_idx`, and grouped problem sizes. For this generation, keep compute unchanged or fallback-compatible and use a single-CTA small-token path.
- Evidence chain: baseline repeatedly checks expert ids inside scalar compute kernels; FlashInfer-style references build expert offsets and maps once; SM120 data-prep evidence says single-CTA wins for small `topk_length` and CUB sort regresses.
- Expected impact: Creates the required structure for grouped GEMM and removes repeated routing decode from future kernels. Current `T=1` speedup may be small, but it is the cleanest prerequisite.
- Risks and rollback: Extra scheduling kernels can dominate at `topk_length=10`; use a single-CTA path and keep parent staged compute fallback if metadata validation fails.

## Strategy 2: child `1_1`

- Parent: `7ea9f8ad`
- Action category: `tiled_fp4_dequantization`
- Action: Replace the scalar FP4 accumulation helpers with a tiled/vectorized dequant path for one projection first, preferably GEMM1. Reuse a hidden K tile across several gate/up output columns, load packed FP4 and scales in vector chunks, and accumulate a small output tile while preserving the staged algorithm.
- Evidence chain: baseline decodes nibbles via LUT and rereads hidden/mid for every output row; FlashInfer FP4 utilities use 128-bit vector loads and `SF_VEC_SIZE=16`; vectorized-load knowledge says FP4 sub-byte kernels need wide loads and correct cache policy.
- Expected impact: Reduces load/decode instruction count and redundant A-tile traffic without requiring a full grouped GEMM rewrite.
- Risks and rollback: Alignment and scale indexing are fragile. Keep the scalar helper for unaligned/odd cases and validate against existing max_abs/max_rel thresholds.

## Strategy 3: child `1_2`

- Parent: `7ea9f8ad`
- Action category: `grouped_gemm`
- Action: Rewrite GEMM2/down-projection as an expert-major grouped GEMM/finalize kernel. Each tile computes a block of hidden columns for one expert group from `mid`, applies router weight in the epilogue, and accumulates/scatters to final output. Target GEMM2 first because the parent currently uses split atomics plus a separate finalize launch.
- Evidence chain: baseline `stage2_grouped_down_kernel` uses scalar per-hidden-column dots, `split_count`, `atomicAdd`, and then a finalize kernel; FlashInfer finalize-fusion reference applies router scale and scatter in the grouped GEMM epilogue; grouped-GEMM references identify MoE expert batches as the target use case.
- Expected impact: Can remove finalize launch, reduce split atomic traffic, and improve weight/scale locality for down projection while leaving GEMM1 unchanged for attribution.
- Risks and rollback: Native FP4 tensor cores are not available on CC 8.9 per local wiki, so the first grouped GEMM may still need software dequant. Tiny per-expert M at `T=1` can limit occupancy; preserve weighted scatter-add semantics for duplicate token routes.

## Exhaustion

`false`. The lineage contains only the baseline, and each strategy is backed by new local evidence.
