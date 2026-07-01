# Executor History: 1_0

Parent solution: `7ea9f8ad`

Strategy applied: `grouped_expert_scheduling`

Implemented the assigned standalone grouped-expert scheduling/data-prep path in the CUDA extension. A new single-CTA CUDA kernel builds expert-major metadata from the flattened `topk_ids` routes for the current small-route path (`T * top_k <= 32768`):

- `topk_slots`: int32 flattened top-k route ids in expert-major order.
- `expert_counts` and `expert_offsets`: per-local-expert counts and prefix offsets.
- `a_map`: grouped row to source token id.
- `c_map`: original flattened route id to grouped row inverse map, with invalid/non-local routes left as `-1`.
- `tile_idx_to_expert_idx`: local expert id per non-empty expert slot, `-1` for empty slots.
- `problem_sizes_mnkl`: per-local-expert gate/up grouped-GEMM problem sizes `[count, 2 * intermediate_size, hidden_size, 1]`.

The parent staged compute path remains the active fallback consumer: `stage1_activation_kernel`, `stage2_grouped_down_kernel`, and `finalize_output_kernel` still consume the original route tensors and preserve the baseline output path. No CUB/radix sort, FlyDSL, evaluator, or benchmark command was used.

Files touched:

- `src/fused_moe_kernel.cu`
- `history.md`
- `executor_result.json`

Self-check:

- `timeout 60 python -c ...` import smoke: ok.
- `timeout 120 python -c ...; m._load_ext()` compile-only extension load: ok.
