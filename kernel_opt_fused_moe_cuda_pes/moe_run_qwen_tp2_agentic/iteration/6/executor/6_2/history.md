Parent id: 06419529

Strategy applied: tiled_fp4_dequantization

Assigned description:
Retile only the GEMM2 FP4 dequant inner loop so each stage2 thread computes a small adjacent hidden-column microtile, preferably two H columns, for the same active expert slot. Load each mid activation block once, load the two packed FP4 weight rows and scale rows with the existing cache-policy helpers, maintain two FP32 accumulators, and write/reduce the two outputs independently through the parent finalize path. Keep scheduling, GEMM1, and T/top-k semantics unchanged.

Concrete edit summary:
- Added GEMM2-only paired FP4 accumulation helpers that load one mid activation vector and apply it to two adjacent GEMM2 weight rows/scales with separate FP32 accumulators.
- Updated `stage2_grouped_down_finalize_kernel` so each thread owns columns `h` and `h + 1` for the same local expert/grouped slot, computes both partials from the same `mid_row`, and finalizes each output independently.
- Kept GEMM1 kernels, grouped slot construction, routing/top-k handling, and Python run surface unchanged.

Files touched:
- `src/fused_moe_kernel.cu`
- `history.md`
- `executor_result.json`

Self-check:
- `timeout 60 python -c "import importlib.util; spec=importlib.util.spec_from_file_location('cand','/home/youchunbo/code/atrex-kernel-agent/kernel_opt_fused_moe_cuda_pes/moe_run_qwen_tp2_agentic/iteration/6/executor/6_2/kernel.py'); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)"` passed.
- No benchmark or evaluator command was run.
