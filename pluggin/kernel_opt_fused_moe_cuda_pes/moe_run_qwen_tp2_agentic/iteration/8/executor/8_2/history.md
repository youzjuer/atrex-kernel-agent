# Executor 8_2 History

Parent id: `a09b7b69`

Strategy applied: `grouped_gemm_cta_local_finalize`

Concrete edit summary:
- Added `stage2_single_token_cta_finalize_kernel` for the `T == 1 && TOPK <= 10` path.
- The fused kernel keeps per-slot GEMM2 work parallel with `threadIdx.y` over local compact slots and `threadIdx.x` over an H tile.
- Each slot thread computes the existing serial GEMM2 dot with the unchanged FP4 dequant helper, adds `gemm2_bias`, applies `topk_weights`, stores the weighted FP32 partial in shared memory, and `threadIdx.y == 0` reduces the slot dimension to write the final bf16 output.
- Replaced the previous single-token stage2 `[TOPK,H]` global partial workspace plus separate finalize launch with one CTA-local fused launch.
- Left GEMM1, routing/compaction metadata, multi-token grouped fallback, and FP4 dequant helpers unchanged.

Files touched:
- `src/fused_moe_kernel.cu`
- `history.md`
- `executor_result.json`

Self-check:
- `ok`: module import smoke completed with `import ok`.
- Evaluator later reported PASS at `515.8720016479492 us`, `max_rel=0.00023841856454964727`.
