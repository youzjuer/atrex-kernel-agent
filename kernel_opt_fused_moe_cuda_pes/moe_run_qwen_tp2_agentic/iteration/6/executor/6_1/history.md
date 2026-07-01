# Executor History

- Child: `6_1`
- Parent id: `06419529`
- Action category: `single_token_slot_scheduling`
- Strategy: Specialize only the `T=1` / `TOPK<=10` scheduling path by replacing the general local expert count/offset/fill metadata sequence with a TOPK-only single-token local-slot compaction path, while keeping the parent GEMM2 arithmetic and completion-count finalize behavior.

## Edits

- Updated `src/fused_moe_kernel.cu`.
- Added `compact_single_token_local_slots_kernel`, which scans the single token's TOPK entries and emits `local_slots`, `local_experts`, `token_counts[0]`, and `local_slot_count[0]`.
- Added `stage2_single_token_slots_down_finalize_kernel`, which launches over `(ceil(H/32), TOPK)` and uses device guards on `local_slot_count`, slot validity, and local expert validity.
- Added a `T == 1 && TOPK <= 10` branch after stage1 to use the compacted slot metadata and slot-indexed GEMM2 finalize launch.
- Left the parent `count_local_slots_kernel`, `build_expert_offsets_kernel`, `fill_grouped_slots_kernel`, and `stage2_grouped_down_finalize_kernel` fallback unchanged for all other shapes.

## Self-Check

- `timeout 60 python -c "import importlib.util; ... spec.loader.exec_module(m)"`: ok.
- `timeout 60 python - <<'PY' ... m._load_ext()`: ok.
