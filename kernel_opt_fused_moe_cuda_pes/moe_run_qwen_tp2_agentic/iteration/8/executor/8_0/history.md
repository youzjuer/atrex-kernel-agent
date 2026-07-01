# Executor 8_0 History

Parent id: `a09b7b69`

Strategy applied: `grouped_expert_scheduling_fixed_topk_metadata`

Concrete edit summary:
- Added `prepare_single_token_topk_local_experts_kernel` for the `T == 1 && TOPK <= 10` path. It fills `slot_to_local_expert[TOPK]` with the local expert id or `-1`, and records `local_slot_count[0]`.
- Added fixed-slot GEMM1 kernels for both the warp-per-output path and the fallback stage1 path. These consume `slot_to_local_expert[tk]` by original top-k slot instead of repeatedly loading and classifying `topk_ids` per GEMM1 output.
- Reworked the single-token GEMM2 partial and finalize kernels to keep `mid[tk, i]`, `slot_partials[tk, h]`, and `topk_weights[tk]` indexed by the original top-k slot. The compact/remapped `local_slots` and `local_experts` metadata are no longer used in this path.
- Left the general multi-token grouped scheduling path unchanged.

Files touched:
- `src/fused_moe_kernel.cu`

Self-check:
- `ok`: `timeout 60 python -c "import importlib.util; ... spec.loader.exec_module(m)"` completed successfully.
- No benchmark or evaluator command was run.
