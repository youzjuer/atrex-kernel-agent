# Executor 7_2 History

Parent id: 6ebb0ad6

Strategy applied: grouped_gemm_compact_atomic_free_finalize

Assigned action: keep the parent GEMM1 and compact single-token local_slots/local_experts path, but replace the T=1 GEMM2 atomic finalize with a compact FP32 [TOPK, H] slot-partial workspace and a deterministic H-wise finalize kernel.

Concrete edit summary:

- Replaced the T=1 `stage2_single_token_slots_down_finalize_kernel` with `stage2_single_token_slots_down_partials_kernel`, which computes GEMM2 dot products for compact local slots and writes FP32 partials by local slot index into a fixed `[TOPK, H]` workspace.
- Added `finalize_single_token_slots_kernel`, launched over H, which loops only `local_slot_count[0]`, adds `gemm2_bias` per local route, applies `topk_weights`, casts directly to bf16, and writes zero when there are no local slots.
- Removed the T=1 path's `out_accum` and `completion_counts` allocations, per-output atomics, threadfence synchronization, and large output/accumulator/count memsets. The inherited grouped expert-count path remains only as the non-T=1 fallback.

Files touched:

- `src/fused_moe_kernel.cu`
- `history.md`
- `executor_result.json`

Self-check:

- Passed: `timeout 60 python3 -c "import importlib.util, sys; spec=importlib.util.spec_from_file_location('cand','kernel.py'); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)"`
- No benchmark or evaluator command was run.
