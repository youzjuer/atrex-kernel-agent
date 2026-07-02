# Executor 7_0 History

- Parent: `6ebb0ad6`
- Strategy: `grouped_expert_scheduling`
- Applied change: moved the `T == 1 && TOPK <= 10` local-slot compaction ahead of GEMM1, then reused `local_slots`, `local_experts`, `token_counts`, and `local_slot_count` for both GEMM stages.
- CUDA edit summary:
  - Added compact single-token GEMM1 kernels that launch over the host-known `TOPK * I` upper bound and return when `slot_idx >= local_slot_count[0]`.
  - Wrote GEMM1 activations into compact rows `mid[slot_idx, i]` rather than full top-k rows.
  - Updated the single-token GEMM2 path to read compact intermediate rows while preserving `topk_weights[tk]` indexing by the original top-k slot.
  - Left the multi-token/grouped fallback path and FP4 arithmetic helpers unchanged.
- Files touched:
  - `src/fused_moe_kernel.cu`
  - `history.md`
  - `executor_result.json`
- Self-check: `ok`
  - Ran: `timeout 60 python -c "import importlib.util; p='/home/youchunbo/code/atrex-kernel-agent/kernel_opt_fused_moe_cuda_pes/moe_run_qwen_tp2_agentic/iteration/7/executor/7_0/kernel.py'; spec=importlib.util.spec_from_file_location('cand', p); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)"`
  - Ran compile-only extension load with `m._load_ext()` under `timeout 180`.
  - Evaluator/benchmark commands were not run.
