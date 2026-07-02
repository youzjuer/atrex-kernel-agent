# Executor 4_2 History

Parent: `0aa690ad`

Strategy: `grouped_gemm`

Applied change:
- Preserved the grouped intermediate dataflow boundary by changing grouped stage1 to write SwiGLU activations to `mid[grouped_idx, i]` instead of scattering back to `mid[original_topk_slot, i]`.
- Extended the grouped schedule metadata with `grouped_experts[grouped_idx]` so GEMM2 can consume grouped rows directly without rereading `topk_ids`.
- Added a grouped GEMM2 down-projection path that reads `mid[grouped_idx, :]`, uses the grouped schedule expert id for weights/scales, and uses `topk_slots[grouped_idx]` only at the final router-weighted `out_accum[token, h]` scatter.
- Kept the existing top-k order GEMM2 path as the fallback when the single-CTA grouped schedule is not built.

Files touched:
- `src/fused_moe_kernel.cu`
- `history.md`
- `executor_result.json`

Self-check:
- `timeout 60 python3 -c "import importlib.util; spec=importlib.util.spec_from_file_location('cand','kernel.py'); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)"` passed.
- `python3 -m py_compile kernel.py` passed.

Evaluator/benchmark:
- Not run, per executor constraints.
