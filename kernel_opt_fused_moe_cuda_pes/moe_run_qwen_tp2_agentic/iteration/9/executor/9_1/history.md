# Executor 9_1 History

Parent id: `84634c40`

Strategy applied: `single_token_stage2_weighted_partials`

Concrete edit summary:
- Built on `9_0` by also moving `topk_weights[tk]` multiplication into `stage2_single_token_slots_down_partials_kernel`.
- Replaced the finalize kernel with `finalize_single_token_sum_slots_kernel`, which only sums already-weighted slot partials and writes bf16 output.
- Left GEMM1, local-slot compaction, FP4 dequant helpers, workspace shape, and multi-token fallback unchanged.

Files touched:
- `src/fused_moe_kernel.cu`
- `history.md`
- `executor_result.json`

Self-check:
- `ok`: module import smoke completed with `import ok`.
- Evaluator PASS: `514.3679976463318 us`, `max_rel=0.00023841856454964727`.
