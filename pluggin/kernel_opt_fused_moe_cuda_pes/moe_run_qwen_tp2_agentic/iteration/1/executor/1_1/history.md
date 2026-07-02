# Executor History

- Child: `1_1`
- Parent: `7ea9f8ad`
- Strategy: `tiled_fp4_dequantization`

## Edit Summary

- Updated `src/fused_moe_kernel.cu`.
- Added a GEMM1 FP4 dequant path that stages each hidden K scale block into CTA shared memory for the four `tk` rows handled by the block.
- Added an aligned 128-bit packed FP4 load path for staged GEMM1 dequant, with 32-bit and scalar fallbacks for unaligned or tail chunks.
- Kept the existing scalar GEMM1 helper as the fallback for scale-vector sizes larger than the staged tile limit.
- Left GEMM2 and the staged kernel topology unchanged.

## Self-Check

- Python import smoke check: passed.
- CUDA extension compile-load smoke check: passed (`m._load_ext()`).
