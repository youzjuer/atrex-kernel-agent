# Executor 10_0 History

Parent id: `84634c40`

Strategy applied: `single_token_finalize_block512`

Concrete edit summary:
- Built on `9_0` and changed only the slim finalize kernel launch block size from 256 to 512 threads.
- Left GEMM1, local-slot compaction, GEMM2 partials with hoisted bias, FP4 dequant helpers, workspace shape, and multi-token fallback unchanged.

Files touched:
- `src/fused_moe_kernel.cu`
- `history.md`
- `executor_result.json`

Self-check:
- `ok`: evaluator compiled and ran.
- Evaluator PASS: `513.4080052375793 us`, `max_rel=0.00023841856454964727`.
