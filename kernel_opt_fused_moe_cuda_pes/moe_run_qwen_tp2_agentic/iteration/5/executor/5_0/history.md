# Executor 5_0 History

- Parent id: `06419529`
- Action category: `grouped_expert_scheduling`
- Strategy applied: Port the active-expert schedule to the current best parent without host synchronization. Replace the `count_local_slots_kernel` + `build_expert_offsets_kernel` + `fill_grouped_slots_kernel` metadata sequence with one device-side active schedule, launch GEMM2/finalize over `min(E_local, T * TOPK)`, and guard with `active_expert_count[0]`.

## Edit Summary

- Replaced the three stage-2 metadata kernels in `src/fused_moe_kernel.cu` with `build_active_expert_schedule_kernel`.
- The new metadata kernel emits `active_experts`, `active_offsets`, compact `active_counts`, `grouped_slots`, `token_counts`, and device-resident `active_expert_count`.
- Updated `stage2_grouped_down_finalize_kernel` to use compact active expert indices and return before loading `active_experts` when `blockIdx.y >= active_expert_count[0]`.
- Updated the host launcher to allocate active schedule buffers, launch the single metadata kernel, and launch stage2 over `max_active_experts = min(E_local, T * TOPK)` without copying the active count to host.
- Left the warp-sliced GEMM1 path and vectorized GEMM2 accumulation math unchanged.

## Files Touched

- `src/fused_moe_kernel.cu`
- `history.md`
- `executor_result.json`

## Self-Check

- `timeout 60 python -c "import importlib.util; ... spec.loader.exec_module(m)"`: ok
- Evaluator/benchmark commands were not run.
