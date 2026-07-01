# Executor 3_0 History

- Parent id: `3c90a4b6`
- Action category: `grouped_expert_scheduling`
- Strategy: replace the grouped-down stage's `E_local`-wide launch with a device-only compact active-expert schedule and avoid copying `active_expert_count` back to the host.

## Edit Summary

- Replaced the parent three-kernel full-local-expert metadata path with `build_active_expert_schedule_kernel`.
- The metadata kernel builds `active_experts`, `active_offsets`, `active_counts`, `grouped_slots`, `token_counts`, and device `active_expert_count` from valid local `T * top_k` slots.
- Updated `stage2_grouped_down_finalize_kernel` so `blockIdx.y` indexes the compact active list and returns before reading `active_experts` when `blockIdx.y >= active_expert_count[0]`.
- Changed the stage-2 launch grid from `(ceil(H / 32), E_local)` to `(ceil(H / 32), min(E_local, T * top_k))`.
- Removed the host `active_expert_count` copy/synchronization path; GEMM1 and GEMM2 arithmetic were left unchanged.

## Files Touched

- `src/fused_moe_kernel.cu`
- `history.md`
- `executor_result.json`

## Self-Check

- Ran the executor import smoke check:
  `timeout 60 python -c "import importlib.util, sys; spec=importlib.util.spec_from_file_location('cand','.../kernel.py'); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)"`
- Result: `ok`
- No benchmark or evaluator command was run.
