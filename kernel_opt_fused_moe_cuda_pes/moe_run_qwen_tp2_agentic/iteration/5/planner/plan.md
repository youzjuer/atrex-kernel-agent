# Planner Plan - Generation 5

- operator: `flashinfer.trtllm_fp4_block_scale_moe`
- profile: `Qwen3_5-Plus_prefill_TP2`
- current best parent: `06419529`
- promoted source profile: PASS, `424.831986 us`, `max_rel=2.384e-4`
- target status: `BLOCKED`, because the real FlashInfer worker exits with code `-11`

## Evidence Summary

Generation 4 found the first large stable improvement: `4_1` added a GEMM1 warp-sliced K-parallel FP4 dequant path and measured `544.480 us`, then retested at `550.720 us`. After promotion, the source profile measured `512.800 us`, and the generation 5 auto baseline measured `424.832 us` in the same qwen_tp2 T=1 surface.

The current CUDA source still has several visible costs after the successful GEMM1 change: it builds grouped metadata through separate count/offset/fill kernels, launches GEMM2 over all `E_local=256` experts even though `T * TOPK <= 10`, reloads the same hidden row across adjacent GEMM1 output columns, and uses `out_accum` plus `completion_counts` bookkeeping for one output token.

## Strategies

### `5_0` - `grouped_expert_scheduling`

- parent: `06419529`
- action: Replace the three current metadata kernels with one small device-side active schedule that emits `active_experts`, `active_offsets`, `active_counts`, `grouped_slots`, `token_counts`, and `active_expert_count`. Launch GEMM2/finalize over `min(E_local, T * TOPK)` and guard with `active_expert_count[0]`; do not copy the count to host.
- expected impact: Fewer metadata launches and far fewer empty expert blocks while keeping warp-sliced GEMM1 and GEMM2 math unchanged.
- risk: Must preserve duplicate top-k slots and local expert filtering exactly.

### `5_1` - `tiled_fp4_dequantization`

- parent: `06419529`
- action: Retile only the GEMM1 warp-sliced fast path so one CTA owns several adjacent intermediate columns for the same routed slot. Reuse each 32-element bf16 hidden K block across those warps, while each warp keeps its own packed FP4 rows and `x1/x2` accumulators.
- expected impact: Reduces repeated hidden-row loads in the dominant GEMM1 path.
- risk: Shared-memory staging or register pressure can erase the gain; keep the current warp-per-output path as fallback.

### `5_2` - `grouped_gemm`

- parent: `06419529`
- action: Add a qwen_tp2-gated topk-slot grouped GEMM2/finalize reducer for `T==1` and `TOPK<=10`. Use the existing vectorized block32 GEMM2 dequant helper, apply router weights, reduce local top-k contributions deterministically per hidden column, and write bf16 output directly. Keep the current expert-major grouped finalize as fallback.
- expected impact: Removes empty expert-grid work, atomics, threadfence/completion bookkeeping, and extra output workspace traffic for the single-token target.
- risk: Too much per-hidden-column serialization can reduce parallelism; row mapping mistakes can silently swap top-k slots.

Exhaustion: false.
