# Iteration 2 Planner Plan

Workspace: `/home/youchunbo/code/atrex-kernel-agent/kernel_opt_fused_moe_cuda_pes/moe_run_agentic`

Operator: `flashinfer.trtllm_fp4_block_scale_moe`

Current parents:

| Parent | Gen | Category | Latency |
|---|---:|---|---:|
| `d660c1aa` | 1 | `tiled_fp4_dequantization` | 223.744 us |
| `f1aa1c2d` | 1 | `grouped_gemm_tiling` | 247.424 us |

Already tried in lineage: `baseline`, `grouped_expert_scheduling`, `tiled_fp4_dequantization`, `grouped_gemm_tiling`. The strategies below keep the same optimization direction but avoid repeating the exact generation-1 plans.

## Bottleneck Evidence

No `ncu` `summary.txt` or source evidence bundle exists under this run's `profiles/`; available evidence is the evaluator/database CUDA-event profile. For smoke `T=2,H=128,I=64,top_k=10`, the staged MoE work is about 983,040 FLOPs. The baseline was 530.848 us, while the best parent is 223.744 us, still only about 0.00439 TFLOPS. This is far below any GPU compute roofline and points to scalar instruction overhead, launch structure, and under-parallel serial loops rather than peak math throughput.

Important local findings:

- CC 8.9 supports BF16/FP8 tensor cores but not FP4 tensor cores, so this target needs software FP4 dequantization rather than native FP4 MMA.
- `d660c1aa` removed the largest scalar decode cost with scale-tiled packed loads, but its stage2 still serially loops over `TOPK * I` per `(token, hidden)` output.
- `f1aa1c2d` added CTA-level grouped GEMM tiling, but its grouped kernels still use scalar switch-based FP4 decode.
- The generation-1 scheduling-only child regressed to 330.816 us, so global route metadata and extra launches must be avoided unless they feed enough computation.
- FlashInfer reference kernels use `token_id_mapping` and `tile_idx_to_expert_idx` and separate gate/up SwiGLU fusion from down/finalize fusion, which supports more focused children.

## Strategy 2_0: Inline Grouped Expert Scheduling

Parent: `d660c1aa`

Category: `grouped_expert_scheduling_inline`

Keep the current best scale-tiled FP4 decode path, but test grouped expert scheduling without a standalone route-compaction launch. Re-map stage1 to expert-major, on-the-fly route traversal: each CTA owns a local expert and an `I` tile, scans the small `M = T * TOPK` route list in registers/shared memory to find routes for that expert, and writes `mid` by original route index. Stage2 output order stays unchanged.

Expected impact: low to medium on smoke, but it isolates whether expert-contiguous weight access helps once the failed global metadata launch is removed.

Risk: repeated scans and CTAs for empty experts can regress small shapes. Roll back to raw topk traversal.

## Strategy 2_1: Tiled FP4 Decode Inside Grouped GEMM

Parent: `f1aa1c2d`

Category: `tiled_fp4_dequantization_grouped_gemm`

Keep the grouped-GEMM parent, but change only FP4 load/dequant inside `stage1_grouped_gemm_kernel` and `stage2_grouped_gemm_kernel`. Replace scalar `load_fp4_scaled` switch decoding with d660-style scale-tiled helpers: one scale load per scale vector, constant/LUT or branchless decode, aligned packed loads, and K-split-safe handling at scale boundaries. Preserve route grouping, 8x16 tiles, 4-way K split, atomics, and output semantics.

Expected impact: medium to high because it combines the best two generation-1 ideas while keeping the effect attributable to FP4 dequantization.

Risk: wrong scale handling across K-split lanes or increased register pressure. Roll back to f1's scalar helper.

## Strategy 2_2: Stage2-Only Grouped Down Projection

Parent: `d660c1aa`

Category: `grouped_gemm_down_projection`

Keep d660 stage1 unchanged, and replace only `stage2_output_kernel` with route-parallel grouped down projection. Treat valid local routes as the M dimension, compute route-tile by hidden-tile CTAs with cooperative split over `I`, reuse d660 scale-tiled FP4 decode for `gemm2_weights`, multiply by `topk_weights`, atomically accumulate into FP32 `out_accum`, then finalize bf16.

Expected impact: medium. This attacks the best parent's remaining serial `TOPK * I` stage2 loop without also changing the gate/up stage.

Risk: `out_accum` memset, atomics, and finalize launch can dominate at `T=2`. Roll back to d660 stage2.

## Search Sources

- Workspace `README.md`
- `profiles/auto_baseline.json`
- `iteration/1/summarizer/summary.md`
- `iteration/1/executor/*/result.json`
- `database/state.json` and `evolution_db.py lineage`
- Parent CUDA sources under `database/solutions/d660c1aa/`, `f1aa1c2d/`, and `333db3e8/`
- `gpu-wiki/docs/kernel-opt/nvidia/common/nvidia-compute-capabilities.md`
- `gpu-wiki/docs/kernel-opt/nvidia/common/blackwell/kernels/grouped-gemm.md`
- `gpu-wiki/docs/kernel-opt/nvidia/common/blackwell/patterns/moe-load-imbalance.md`
- `gpu-wiki/reference-kernels/nvidia/blackwell/cutedsl/flashinfer/`
- `gpu-wiki/docs/kernel-opt/nvidia/cutedsl/sm120/sm120-moe-data-prep.md`
- `gpu-wiki/docs/ref-docs/nvidia/cutedsl/sm120/sm120-moe-data-prep-optimization.md`
- `gpu-wiki/docs/kernel-opt/nvidia/common/blackwell/techniques/vectorized-loads.md`
- `gpu-wiki/reference-kernels/nvidia/hopper/cutedsl/tilelang/quantize.py`
- `reference-projects/flashinfer` was searched but is empty in this checkout.
