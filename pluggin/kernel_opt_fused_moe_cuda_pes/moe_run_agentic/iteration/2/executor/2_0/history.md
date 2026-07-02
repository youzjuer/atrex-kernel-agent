# Executor 2_0 History

Parent id: d660c1aa

Strategy applied: grouped_expert_scheduling_inline

Action summary:
- Kept the d660c1aa staged FP4 decode, activation, and stage2 accumulation path.
- Remapped stage1 scheduling from route-major CTAs to expert-major CTAs.
- Each stage1 CTA now owns one local expert and one I tile, scans the T * TOPK route list on the fly, and computes only routes whose expert matches that CTA.
- Stage1 still writes mid by the original route index, preserving the existing stage2 output order.
- No global expert_counts, expert_offsets, grouped_route_ids, compaction launch, atomics, or finalize path were added.

Files touched:
- src/fused_moe_kernel.cu
- history.md
- executor_result.json

Self-check:
- ok: imported kernel.py and loaded the CUDA extension with timeout 60.

Notes:
- No evaluator or benchmark commands were run.
