# Executor 5_1 History

Parent: `06419529`

Strategy: `tiled_fp4_dequantization`

Applied change:
- Added a GEMM1-only tiled FP4 dequant fast path, `stage1_activation_warp_i_tile_kernel`.
- The new fast path maps one CTA to one routed slot `tk` and an 8-column contiguous `i` tile.
- Within each CTA, the hidden row is staged once per 32-scale-column group into shared memory as packed bf16 words, then the eight warps reuse that staged hidden data while keeping independent x1/x2 accumulators and packed FP4 weight loads.
- Kept the prior warp-per-output GEMM1 kernel as fallback when the tiled fast-path shape assumptions are not met.
- Left routing, GEMM2, finalization, Python `run(...)`, and fallback scalar stage1 behavior unchanged.

Files touched:
- `src/fused_moe_kernel.cu`
- `history.md`
- `executor_result.json`

Self-check:
- `ok`: `timeout 60 python -c "import importlib.util; ... spec.loader.exec_module(m)"` completed successfully.
- Evaluator and benchmark commands were not run.
