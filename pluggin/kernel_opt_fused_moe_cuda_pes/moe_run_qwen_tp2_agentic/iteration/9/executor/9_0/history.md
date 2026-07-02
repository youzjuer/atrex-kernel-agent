# Executor 9_0 History

Parent id: `a09b7b69`

Strategy applied: `single_token_stage2_bias_hoist_finalize_slim`

Concrete edit summary:
- Moved `gemm2_bias` application from the single-token finalize kernel into `stage2_single_token_slots_down_partials_kernel`.
- Added `finalize_single_token_weighted_slots_kernel`, a slimmer finalize pass that only reads compact `local_slots`, FP32 slot partials, and `topk_weights`.
- Kept GEMM1, local-slot compaction, FP4 dequant helpers, `[TOPK,H]` partial workspace shape, and multi-token fallback unchanged.

Files touched:
- `src/fused_moe_kernel.cu`
- `history.md`
- `executor_result.json`

Self-check:
- `ok`: module import smoke completed with `import ok`.
- Evaluator PASS: `502.8480291366577 us`, `max_rel=0.00023841856454964727`.
