# Executor History: 1_0

Parent solution: `dec7821d`

Strategy applied: `grouped_expert_scheduling`

Implemented the assigned grouped expert scheduling path inside the CUDA extension. A new single-CTA scheduling kernel builds local expert metadata from `topk_ids`: `expert_counts`, `expert_offsets`, `grouped_route_ids` in expert-contiguous order, and `route_to_grouped` as the inverse route map. Invalid and non-local routes are left unmapped with `-1`, so empty expert groups remain empty.

Stage1 now consumes `grouped_route_ids` and writes intermediate activations by grouped route slot. Stage2 now also runs over grouped route slots, checks the inverse map, computes the expert output for each grouped route, atomically accumulates FP32 contributions into a token output buffer, and finalizes to bf16.

Files touched:
- `src/fused_moe_kernel.cu`
- `history.md`
- `executor_result.json`

Self-check:
- `timeout 60 python -c ...` import smoke: ok
- Compile-only extension load via `mod._load_ext()`: ok
- No benchmark or evaluator command was run.
