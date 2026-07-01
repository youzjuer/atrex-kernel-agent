# Executor History: 1_2

Parent: `7ea9f8ad`

Strategy applied: `grouped_gemm`

## Edit Summary

- Replaced the GEMM2 split down-projection kernel with an expert-major grouped down/finalize kernel.
- Added local routing metadata construction for GEMM2: per-expert counts, offsets, cursors, grouped top-k slot list, and per-token local contribution counts.
- Removed the separate finalize kernel launch from the GEMM2 path. The grouped down kernel applies `topk_weights` in its epilogue, accumulates local expert contributions into fp32 scratch, and writes bf16 output when each output element has received its expected local contributions.
- Kept GEMM1/stage1 activation unchanged for attribution.

## Files Touched

- `src/fused_moe_kernel.cu`
- `history.md`
- `executor_result.json`

## Self-Check

Status: `ok`

Commands run:

```bash
timeout 60 python -c "import importlib.util; p='/home/youchunbo/code/atrex-kernel-agent/kernel_opt_fused_moe_cuda_pes/moe_run_qwen_tp2_agentic/iteration/1/executor/1_2/kernel.py'; spec=importlib.util.spec_from_file_location('cand', p); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)"
timeout 180 python - <<'PY'
import importlib.util
p = '/home/youchunbo/code/atrex-kernel-agent/kernel_opt_fused_moe_cuda_pes/moe_run_qwen_tp2_agentic/iteration/1/executor/1_2/kernel.py'
spec = importlib.util.spec_from_file_location('cand', p)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
m._load_ext()
PY
```

No benchmark or evaluator command was run.
