# Executor 4_1 History

Parent id: `98a890ad`

Strategy applied: `tiled_fp4_dequantization`

Implemented the assigned GEMM1/gate-up warp-sliced FP4 dequantization path only. The candidate adds a `stage1_activation_warp_k_kernel` that maps one warp to one `(token, top-k slot, intermediate)` output pair for `scale_vec == 32` shapes. Each lane walks disjoint 32-element K scale blocks, uses 128-bit loads for packed FP4 gate/up rows and bf16 hidden chunks, accumulates partial `x1` and `x2`, then warp-reduces before the single `mid[tk, i]` store.

Concrete edits:
- Added GEMM1-only block32/word8 FP4 decode helpers in `src/fused_moe_kernel.cu`.
- Added the warp-parallel stage1 kernel with scalar lane-0 fallback for unexpected alignment inside the specialized launch path.
- Updated the stage1 launch selector to use the warp kernel only when `H / gemm1_scale_cols == 32` and `H` is a multiple of 32; otherwise it launches the parent scalar stage1 kernel.
- Left `kernel.py`, `src/fused_moe_kernel.cpp`, and the grouped GEMM2/finalize CUDA path unchanged.

Files touched:
- `src/fused_moe_kernel.cu`
- `history.md`
- `executor_result.json`

Self-check:
- `timeout 60 python -c "import importlib.util, sys; spec=importlib.util.spec_from_file_location('cand','.../kernel.py'); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)"` passed.
- Evaluator and benchmark commands were not run.
