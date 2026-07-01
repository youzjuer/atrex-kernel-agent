# Executor 2_0 History

- Parent id: `3c90a4b6`
- Action category: `grouped_expert_scheduling`
- Strategy: replace the parent grouped-down scheduler's `E_local`-wide stage-2 launch with a compact active-expert schedule for the small Qwen TP2 route set.

## Edit Summary

- Added `build_active_expert_schedule_kernel` in `src/fused_moe_kernel.cu`.
- The new normal-path metadata kernel builds `active_experts`, compact `active_offsets`/`active_counts`, `grouped_slots`, `token_counts`, and a device `active_expert_count` from valid local `T * top_k` slots.
- Updated `stage2_grouped_down_finalize_kernel` so `blockIdx.y` indexes the compact active-expert list, while preserving the existing GEMM2/dequant/finalize arithmetic.
- Changed the stage-2 launch grid from `(ceil(H / 32), E_local)` to `(ceil(H / 32), active_expert_count)`.
- Kept the old full-`E_local` metadata path as a validation fallback if the copied-back active count is outside `[0, min(E_local, T * top_k)]`.

## Files Touched

- `src/fused_moe_kernel.cu`
- `history.md`
- `executor_result.json`

## Self-Check

- Ran the executor import smoke check:
  `timeout 60 python -c "import importlib.util; spec=importlib.util.spec_from_file_location('cand','.../kernel.py'); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)"`
- Result: `ok`
- No benchmark or evaluator command was run.
