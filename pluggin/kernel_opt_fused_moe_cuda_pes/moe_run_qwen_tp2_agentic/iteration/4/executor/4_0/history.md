# Executor History

- Child: `4_0`
- Parent: `0aa690ad`
- Action category: `grouped_expert_scheduling`

## Strategy Applied

Kept the parent grouped GEMM1 arithmetic and top-k slot/token mapping unchanged, while extending the single-CTA grouped scheduler to emit direct grouped-row expert metadata. The grouped stage1 consumer now loads `row_to_expert[grouped_idx]` and uses the device `grouped_rows_count` predicate instead of scanning `tile_idx_to_expert_idx` and `problem_sizes_mnkl` for each grouped row.

## Concrete Edits

- Added `row_to_expert` and `grouped_rows_count` tensors in `src/fused_moe_kernel.cu`.
- Initialized `row_to_expert` to `-1`, wrote `grouped_rows_count[0]` after expert offsets/counts are computed, and stored `row_to_expert[grouped_idx] = le` when each valid local route is grouped.
- Replaced the grouped stage1 per-row expert recovery scan with one direct `row_to_expert` load.
- Preserved `topk_slots`, `a_map`, and `c_map` behavior, and left GEMM1 math, GEMM2, routing, and final reduction unchanged.

## Files Touched

- `src/fused_moe_kernel.cu`

## Self-Check

- `timeout 60 python -c "import importlib.util; ... spec.loader.exec_module(m)"`: ok
- `python -m py_compile kernel.py`: ok
- Evaluator/benchmark commands were not run.
