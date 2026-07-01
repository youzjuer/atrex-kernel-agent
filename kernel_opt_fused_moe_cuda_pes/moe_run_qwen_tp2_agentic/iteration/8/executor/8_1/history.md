# Executor 8_1 History

Parent id: `a09b7b69`

Strategy applied: `tiled_fp4_dequantization_byte_pair_lut`

Concrete edit summary:
- Added a 256-entry constant-memory E2M1 byte-pair decode LUT where each packed FP4 byte returns the low/high decoded float values as `float2`.
- Replaced the repeated per-nibble branchless decode calls inside `accumulate_fp4_word8_gemm2` with four byte-pair decode helper calls.
- Replaced the repeated per-nibble branchless decode calls inside `accumulate_fp4_pair_word8_gemm1` with four byte-pair helpers that process two bf16 hidden values and two packed FP4 bytes at a time.
- Left launch structure, routing, partial/finalize logic, vectorized 128-bit loads, and scalar fallback tile paths unchanged.

Files touched:
- `src/fused_moe_kernel.cu`
- `history.md`
- `executor_result.json`

Self-check:
- `ok`: `timeout 60 python -c "... spec.loader.exec_module(m) ..."` completed with `import ok`.
