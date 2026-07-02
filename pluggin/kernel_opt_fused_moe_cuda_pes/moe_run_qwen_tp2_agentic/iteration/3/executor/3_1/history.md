# Executor 3_1 History

Parent solution: `7ea9f8ad`

Strategy applied: `tiled_fp4_dequantization`

Applied exactly the assigned GEMM2/down-projection FP4 dequantization retile. Routing, stage1
activation, split_count policy, atomic accumulation, and finalize behavior were left unchanged.

Concrete edits:
- Added `stage2_grouped_down_h_tile_kernel`, a one-warp GEMM2 microkernel where each CTA handles one
  routed top-k slot, one split, and an 8-column hidden tile.
- For 32-element GEMM2 scale blocks, the warp loads each 32-float `mid` block once into lane
  registers and reuses it across the H tile.
- The warp cooperatively loads the packed FP4 words for the H tile, broadcasts those words and the
  per-column scales with warp shuffles, decodes e2m1 nibbles, reduces across K lanes, and writes via
  the existing weighted `atomicAdd` path.
- Kept the original scalar `stage2_grouped_down_kernel` as the fallback when
  `I / gemm2_scale_cols != 32`.

Files touched:
- `src/fused_moe_kernel.cu`
- `history.md`
- `executor_result.json`

Self-check:
- `timeout 60 python -c "... spec.loader.exec_module(m)"` completed successfully.
- `python -m py_compile kernel.py` completed successfully.
- `timeout 90 python -c "... spec.loader.exec_module(m); m._load_ext()"` completed successfully.
- No benchmark or evaluator command was run.
