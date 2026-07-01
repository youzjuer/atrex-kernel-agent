# Planner Plan - Generation 4

- operator: `flashinfer.trtllm_fp4_block_scale_moe`
- profile: `Qwen3_5-Plus_prefill_TP2`
- focus: grouped expert scheduling, tiled FP4 dequantization, grouped GEMM

## Evidence Summary

The current evidence is CUDA-event evaluator data, not an NCU summary. The stable current source baseline passes qwen_tp2 T=1 at 1055.200 us with max_rel 1.192e-4. Generation 3's best retest was 1114.016 us and was not promoted. The selected parents are slower but contain useful structures: `98a890ad` has GEMM2 dequant/grouped-down work, `0aa690ad` has a grouped GEMM1 producer, and `b9545b81` has scheduling metadata only.

Approximate useful GEMM work is 251.66 MFLOP, so the stable source path reaches only about 0.238 TFLOPS before software FP4 decode, atomics, launch overhead, and memory traffic. The next plans avoid restating prior metadata creation, active-expert pruning, GEMM2 vectorization, GEMM2 H-tiling, split-K partial reduction, expert-major GEMM2 finalize, or grouped GEMM1.

## Strategies

### `4_0` - `grouped_expert_scheduling`

- parent: `0aa690ad`
- action: Extend the single-CTA grouped schedule to emit `row_to_expert[grouped_idx]` and a grouped-row count. Replace `stage1_grouped_activation_kernel`'s scan over `tile_idx_to_expert_idx` and `problem_sizes` with a direct `row_to_expert` load. Keep GEMM1 arithmetic, GEMM2, routing, `topk_slots`, `a_map`, and `c_map` semantics unchanged.
- evidence: `0aa690ad` rebuilds expert identity inside every grouped stage1 tile even though the scheduler already knows it. FlashInfer grouped kernels carry direct tile-to-expert mappings and use predicates for invalid tiles.
- expected impact: Removes an O(E_local) metadata-consumer lookup from the grouped stage1 hot path without adding host synchronization or sorting.
- risks: Duplicate routes and non-local filtering must remain exact; launch with the host-known `total_routes` upper bound if grouped row count stays device-side.

### `4_1` - `tiled_fp4_dequantization`

- parent: `98a890ad`
- action: Add a warp-sliced K-parallel dequant path for GEMM1/gate-up only. For qwen_tp2 `H=4096` and `scale_vec=32`, assign a warp or half-warp to a gate/up output pair, have lanes process disjoint packed FP4 K chunks with 128-bit loads, decode/apply scales, accumulate `x1/x2`, and warp-reduce to one `mid[tk,i]` store. Leave the parent grouped GEMM2/finalize path unchanged.
- evidence: `98a890ad` improved GEMM2 dequant but GEMM1 remains one thread doing a full H dot per output pair. Local NVFP4 GEMV guidance uses many lanes per row/dot, wide loads, byte decomposition, and warp reductions for low-batch software FP4 work.
- expected impact: Increases parallelism in the dominant GEMM1 work and attacks scalar decode instruction chains without changing scheduling or finalization.
- risks: Shuffle/reduction overhead and register pressure can erase gains. Specialize the fast path to qwen-aligned shapes and keep the scalar stage1 fallback.

### `4_2` - `grouped_gemm`

- parent: `0aa690ad`
- action: Preserve the grouped intermediate layout across both GEMMs. Write GEMM1 SwiGLU output as `mid_grouped[grouped_idx, i]` in expert-major order, then make GEMM2 consume `grouped_idx` and `expert_offsets` directly. Use `c_map` or a `permuted_idx_to_expanded_idx`-style mapping only in the final router-weighted scatter.
- evidence: `0aa690ad` creates grouped rows for GEMM1, then scatters `mid` back to original top-k slot order and loses that locality before GEMM2. FlashInfer grouped FC1/finalize references keep valid M rows grouped and postpone row mapping to the epilogue/final scatter.
- expected impact: Removes an intermediate layout round trip and lets GEMM2 reuse the expert-major grouping created by GEMM1. This is distinct from prior GEMM2-only finalize and GEMM1-only producer plans.
- risks: Row mapping mistakes can silently swap top-k rows; final accumulation must still handle multiple routes per token correctly.

## Sources

- workspace `README.md`: FlashInfer-compatible Qwen TP2 shape and constraints.
- `profiles/stable_baseline_result.json`: stable source baseline at 1055.200 us.
- `iteration/3/summarizer/summary.md`: generation 3 retest did not promote 3_1.
- `tools/evolution_db.py lineage`: selected parent lineages and already-tried categories.
- `database/solutions/0aa690ad/src/fused_moe_kernel.cu`: grouped stage1 scan and original-slot `mid` scatter.
- `database/solutions/98a890ad/src/fused_moe_kernel.cu`: GEMM2 dequant changes with scalar GEMM1 still present.
- `gpu-wiki/docs/hardware-specs/hardware_specs_hopper.md`: local Hopper/H20 hardware guidance; no fabricated L20D FP4 tensor-core claim.
- `gpu-wiki/docs/kernel-opt/nvidia/common/blackwell/kernels/grouped-gemm.md`: variable-M MoE grouped GEMM and tile-to-expert mapping.
- `gpu-wiki/docs/kernel-opt/nvidia/common/blackwell/kernels/fused-moe.md`: grouped gate-up, down, and combine dataflow.
- `gpu-wiki/docs/kernel-opt/nvidia/common/blackwell/kernels/nvfp4-gemv.md`: many-lane FP4 GEMV work partitioning and reductions.
- `gpu-wiki/reference-kernels/nvidia/blackwell/cutedsl/flashinfer/*grouped*_fusion.py`: FlashInfer grouped row mappings and final epilogue mapping.
- `reference-projects/README.md`: submodules checked; local useful code came from gpu-wiki extracted references.

Exhaustion: false.
