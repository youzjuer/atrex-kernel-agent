# Executor 2_2 History

- Parent id: `7ea9f8ad`
- Action category: `grouped_gemm`
- Strategy applied: rewrote GEMM2/down-projection from atomic split-K accumulation into a two-phase topk-slot grouped split-K path. GEMM1/stage1 was left unchanged.

## Edit Summary

- Replaced `stage2_grouped_down_kernel` with `stage2_split_partial_kernel`, which computes unweighted FP32 GEMM2 partials for each valid routed top-k slot, output hidden column, and K split into a `split_partials` tensor workspace.
- Replaced the standalone `finalize_output_kernel` with `stage2_reduce_finalize_kernel`, which reduces split partials in deterministic `top_k` then split order, applies `topk_weights`, adds GEMM2 bias during reduction, and writes the final bf16 output.
- Removed the `out_accum` fp32 tensor and `cudaMemsetAsync` accumulation initialization from the forward path.

## Files Touched

- `src/fused_moe_kernel.cu`
- `history.md`
- `executor_result.json`

## Self-Check

- `timeout 60 python -c "import importlib.util; spec=importlib.util.spec_from_file_location('cand','/home/youchunbo/code/atrex-kernel-agent/kernel_opt_fused_moe_cuda_pes/moe_run_qwen_tp2_agentic/iteration/2/executor/2_2/kernel.py'); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)"` passed.
- Evaluator and benchmark commands were not run.
