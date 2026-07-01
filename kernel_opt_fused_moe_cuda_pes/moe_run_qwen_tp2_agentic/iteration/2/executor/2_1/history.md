# Executor 2_1 History

Parent solution: `3c90a4b6`

Strategy applied: `tiled_fp4_dequantization`

Applied exactly the assigned GEMM2/down-projection FP4 dequantization retile inside the grouped parent. The expert grouping, metadata kernels, stage1/GEMM1 path, and finalize/atomic output logic were left unchanged.

Concrete edits:
- Added GEMM2-only 128-bit PTX load helpers for streamed packed FP4 weights with `L1::no_allocate`.
- Added vectorized 128-bit scale loads for aligned GEMM2 scale rows.
- Added branchless E2M1 FP4 nibble decode for the GEMM2 microkernel.
- Added vectorized `mid` tile loads with `L1::evict_last` so the reused activation tile is biased to stay hot while weights stream.
- Replaced only the GEMM2 `accumulate_fp4_scaled_tile` scale-block loop with `accumulate_fp4_scaled_i_tiles_gemm2`, keeping a scalar fallback for non-32-element scale vectors or unaligned rows.

Files touched:
- `src/fused_moe_kernel.cu`
- `history.md`
- `executor_result.json`

Self-check:
- `timeout 60 python -c "... spec.loader.exec_module(m)"` from the child workspace completed successfully.
- No benchmark or evaluator command was run.
