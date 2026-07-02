# Executor History: 3_2

Parent id: b9545b81

Strategy applied: grouped_gemm

Applied change:
- Replaced the active stage1 producer path with a grouped metadata-driven gate/up stage for the scheduled route case.
- Added `stage1_grouped_activation_kernel`, which consumes `topk_slots`, `a_map`, `expert_offsets`, `tile_idx_to_expert_idx`, and `problem_sizes_mnkl`.
- The grouped kernel maps each grouped row back to its source token, selects the local expert from the existing expert-major schedule, computes paired gate/up FP4 block-scaled accumulators over `K=H`, applies the existing SwiGLU convention, and scatters the activation to `mid[original_topk_slot, i]`.
- Kept `stage2_grouped_down_kernel` and finalize logic unchanged. The original scalar `stage1_activation_kernel` remains only as the fallback when the parent schedule is not built.

Files touched:
- `src/fused_moe_kernel.cu`

Self-check:
- `timeout 60 python -c "import importlib.util, sys; spec=importlib.util.spec_from_file_location('cand','/home/youchunbo/code/atrex-kernel-agent/kernel_opt_fused_moe_cuda_pes/moe_run_qwen_tp2_agentic/iteration/3/executor/3_2/kernel.py'); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)"` passed.
- Additional non-evaluator extension load/compile check passed.

Notes:
- No evaluator or benchmark command was run.
- No files outside the child workspace were edited.
