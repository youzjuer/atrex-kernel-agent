# Executor 2_2 History

## Parent

- Parent id: `d660c1aa`
- Candidate: `iteration/2/executor/2_2/kernel.py`

## Strategy Applied

- Category: `grouped_gemm_down_projection`
- Applied only the down-projection/output-stage strategy: kept the parent stage1 gate/up tiled FP4 decode and activation path unchanged, and replaced the serial top-k loop in stage2 with a route-parallel grouped down-projection path.

## Edit Summary

- Replaced `stage2_output_kernel` in `src/fused_moe_kernel.cu` with `stage2_grouped_down_kernel`.
- The new stage2 maps CTAs across hidden-column tiles, local route rows (`T * TOPK`), and a cooperative split over gemm2 scale/I tiles.
- Each valid local route computes a partial down-projection using the existing FP4 decode helper, applies the route `topk_weight`, and `atomicAdd`s into an FP32 `out_accum` buffer.
- Added `finalize_output_kernel` to convert the FP32 accumulation buffer to bf16 output.
- Updated the CUDA launcher to allocate and zero `out_accum`, launch the grouped down-projection kernel, then launch finalize.

## Files Touched

- `src/fused_moe_kernel.cu`
- `history.md`
- `executor_result.json`

## Self-Check

- `timeout 60 python3 -c "... spec.loader.exec_module(m)"`: ok
- `timeout 60 python3 -c "... spec.loader.exec_module(m); m._load_ext()"`: ok
- No evaluator or benchmark commands were run.
