# Executor 9_2 History

Parent id: `84634c40`

Strategy applied: `single_token_finalize_block128`

Concrete edit summary:
- Built on `9_0` and changed only the slim finalize kernel launch block size from 256 to 128 threads.
- Left GEMM1, local-slot compaction, GEMM2 partials with hoisted bias, FP4 dequant helpers, workspace shape, and multi-token fallback unchanged.

Files touched:
- `src/fused_moe_kernel.cu`
- `history.md`
- `executor_result.json`

Self-check:
- `ok`: evaluator compiled and ran.
- Evaluator PASS: `556.3200116157532 us`, `max_rel=0.00023841856454964727`.
