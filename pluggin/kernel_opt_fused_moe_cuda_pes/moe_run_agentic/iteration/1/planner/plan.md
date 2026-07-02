# Generation 1 Planner Plan

## Context

- Workspace: `/home/youchunbo/code/atrex-kernel-agent/kernel_opt_fused_moe_cuda_pes/moe_run_agentic`
- Operator surface: `flashinfer.trtllm_fp4_block_scale_moe`
- Application profile: `Qwen3_5-Plus_prefill_TP2`
- Parent: `dec7821d`
- Current profile: PASS, `527.008 us` on smoke `T=2,H=128,I=64,top_k=10`
- Lineage tried set: `baseline` only
- Exhaustion: `false`

## Evidence Summary

The baseline is correct but still scalar. `stage1_activation_kernel` maps one thread to one `(route, intermediate)` output and loops across all `H`; `stage2_output_kernel` maps one thread to one `(token, hidden)` output and loops across `TOPK` and all `I`. Both paths call scalar FP4 unpack/dequant inside the innermost dot-product loop. For the smoke shape, the staged arithmetic count is about `988,160` FLOPs, so `527.008 us` is roughly `0.0019 TFLOPS`, indicating instruction overhead, serial dot loops, and poor data reuse rather than math throughput.

The L20D target is treated as an Ada/CC 8.9-style CUDA target in this workspace. gpu-wiki's compute capability table shows CC 8.9 supports BF16/FP8 tensor cores but not native FP4 tensor cores, so the next steps should focus on software FP4 dequantization, CUDA-core tiling, and grouped MoE scheduling rather than Blackwell-only FP4 tensor core instructions.

## Strategy 1_0

- child: `1_0`
- parent_id: `dec7821d`
- action_category: `grouped_expert_scheduling`
- action_description: Add a lightweight CUDA data-prep/scheduling path that converts local top-k routes into expert-contiguous metadata: `expert_counts`/`expert_offsets`, `grouped_route_ids` holding `t * TOPK + k` in expert order, and an inverse grouped-to-route map. For the smoke path, start with a single-CTA shared-memory histogram and scatter over `T * TOPK` route slots; keep invalid/non-local experts mapped to empty groups. Feed stage1/stage2 from `grouped_route_ids` so consecutive CTAs work on the same expert before any GEMM rewrite.
- evidence_chain: The baseline profile is correct but slow for the smoke shape. The baseline kernels process raw top-k order and branch per route in both stage1 and stage2, so expert weights are accessed in router order rather than expert-contiguous order. FlashInfer/CUTLASS reference MoE data prep builds `expert_offsets`, problem sizes, `a_map`/`c_map` from `topk_ids`, and the FlashInfer grouped kernels consume `token_id_mapping` plus `tile_idx_to_expert_idx` for expert-contiguous work. New finding: the local reference data-prep V9 uses shared-memory histogram/scatter with per-CTA base offsets and avoids global atomics in the scatter hot path.
- expected_impact: Medium near-term impact on smoke because `T * TOPK` is tiny and the scheduling launch adds overhead, but high strategic value because it creates the metadata needed for expert-local weight reuse and later grouped GEMM.
- risks: Extra temporary tensors and an added launch can regress the T=2 smoke benchmark. Route ordering changes require careful inverse mapping so final accumulation still matches FlashInfer top_k semantics. Roll back by restoring raw topk traversal if scheduling overhead dominates.

## Strategy 1_1

- child: `1_1`
- parent_id: `dec7821d`
- action_category: `tiled_fp4_dequantization`
- action_description: Replace scalar per-element FP4 dequantization in the dot-product inner loops with tile-level decode. Use a 16-entry e2m1 lookup table or branchless nibble decode, vectorized `uint32`/`uint4` packed loads where alignment allows, and decode one K tile at a time while reusing each loaded block scale for all FP4 values covered by `scale_vec_size`. Keep the existing two-stage algorithm and output order unchanged; only change how FP4 weights and scales are loaded/dequantized inside each dot.
- evidence_chain: The baseline helper `load_fp4_scaled` loads one byte, extracts one nibble, calls a switch-based e2m1 decoder, loads a scale, and multiplies for every scalar weight. Stage1 calls it twice per `H` element for every `(token, topk, I)`, and stage2 calls it once per `I` element for every `(token, topk, H)`. gpu-wiki's CC table rules out native FP4 tensor cores for CC 8.9, and the NVFP4 reference recommends wide vectorized loads because packed FP4 has only 0.5 byte per element.
- expected_impact: Medium to high for both smoke and larger shapes by removing switch-heavy scalar decode, reducing scale reloads, and improving memory instruction efficiency without changing the algorithmic structure.
- risks: Vectorized loads require alignment and bounds handling for small H/I. Larger decode tiles can increase register pressure or shared-memory use. `scale_vec_size` must be derived from tensor shape, not hard-coded.

## Strategy 1_2

- child: `1_2`
- parent_id: `dec7821d`
- action_category: `grouped_gemm_tiling`
- action_description: Replace the one-output-per-thread serial dot loops with CTA-level grouped GEMM tiles. Treat the routed rows as `M = T * TOPK` grouped by local expert, compute gate/up as grouped GEMM `[M_e, H] x [2I, H]^T` with a `BM x BN` tile such as `8x16` or `16x16` for the smoke path, and compute down projection as `[M_e, I] x [H, I]^T`. Use cooperative K-split within a CTA and warp/block reductions for each output tile; keep FP32 accumulation and the same SwiGLU/final weighted sum semantics.
- evidence_chain: Baseline stage1 assigns one thread to one `(route, I)` output and loops over all `H` serially; stage2 assigns one thread to one `(token, H)` output and loops over `TOPK` and `I` serially. This is the naive GEMM pattern identified in gpu-wiki: each thread independently reads whole rows/columns, has low compute-to-memory ratio, and repeats global reads. The GEMM guide shows shared-memory tiling improves reuse, and the grouped GEMM wiki/reference material identifies MoE as the direct use case where multiple expert GEMMs share N/K and differ in M.
- expected_impact: Highest upside on `qwen_micro` and TP2-like dimensions because it changes the dominant serial dot loops into cooperative tiled GEMM work with data reuse. On smoke, it may still help if launch count stays at two and tile sizes are tuned for `T * TOPK <= 40`.
- risks: Largest implementation risk: reductions and scatter-add semantics can introduce numerical-order differences, and small `M` per expert can underutilize CTAs. It may depend on grouped route metadata; if implemented standalone, it should include a minimal local grouping path or fall back to raw route rows.

## Sources Searched

- `agents/gpu-kernel-planner.md` - planner contract and novelty rules.
- `moe_run_agentic/README.md` - FlashInfer API surface, Qwen3.5 Plus TP2 contract, and current staged CUDA baseline.
- `moe_run_agentic/profiles/auto_baseline.json` - baseline smoke profile.
- `database/solutions/dec7821d/src/fused_moe_kernel.cu` - scalar staged CUDA implementation.
- `tools/evolution_db.py lineage --workspace moe_run_agentic --solution dec7821d --json` - lineage contains only baseline.
- `gpu-wiki/docs/kernel-opt/nvidia/common/nvidia-compute-capabilities.md` - CC 8.9 tensor core support matrix.
- `gpu-wiki/docs/ref-docs/generic/gemm-optimization-guide.md` - naive GEMM, shared-memory tiling, register tiling evidence.
- `gpu-wiki/docs/kernel-opt/nvidia/common/blackwell/kernels/grouped-gemm.md` - grouped GEMM scheduling concepts and MoE caveats.
- `gpu-wiki/docs/kernel-opt/nvidia/common/blackwell/techniques/vectorized-loads.md` - packed FP4 wide-load guidance.
- `gpu-wiki/reference-kernels/nvidia/blackwell-geforce/cutedsl/moe_data_prep/` - MoE histogram/prefix/scatter metadata reference.
- `gpu-wiki/reference-kernels/nvidia/blackwell/cutedsl/flashinfer/blockscaled_contiguous_gather_grouped_gemm_swiglu_fusion.py` - FlashInfer grouped SwiGLU metadata and epilogue structure.
- `gpu-wiki/reference-kernels/nvidia/blackwell/cutedsl/flashinfer/blockscaled_contiguous_grouped_gemm_finalize_fusion.py` - FlashInfer finalize/scatter metadata structure.
- `reference-projects/{flashinfer,cutlass,DeepGEMM}` - searched; submodule directories were empty in this checkout.
