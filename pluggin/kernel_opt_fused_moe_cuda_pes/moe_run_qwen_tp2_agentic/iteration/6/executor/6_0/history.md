# Executor History

Parent id: 06419529

Child: 6_0

Action category: grouped_gemm_atomic_free_reduce

Strategy applied:
Replace only the GEMM2/finalize accumulation for the T=1 Qwen TP2 path with an atomic-free two-phase reducer. GEMM1 and the existing grouped expert metadata schedule are unchanged.

Concrete edits:
- Renamed the original GEMM2 grouped finalize kernel to `stage2_grouped_down_finalize_atomic_kernel` and kept it as the non-T=1 fallback.
- Added `stage2_grouped_down_partial_kernel`, which keeps the grouped expert traversal and FP4 GEMM2 inner math but writes one FP32 matmul partial per compact active local slot and hidden column.
- Added `finalize_t1_slot_partials_kernel`, which reduces compact local slots for token 0 using `topk_weights`, applies `gemm2_bias` once per local slot contribution, casts to bf16, and writes zero when there are no local slots.
- Routed only `T == 1` through the new partial workspace and finalize kernel. The old atomic accumulation path remains for other token counts.

Files touched:
- `src/fused_moe_kernel.cu`
- `history.md`
- `executor_result.json`

Self-check:
- Python import smoke check passed.
- Extension load/compile smoke check passed.
- No benchmark or evaluator command was run.
