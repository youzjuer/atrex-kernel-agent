# Planner Plan - Generation 6

- operator: `flashinfer.trtllm_fp4_block_scale_moe`
- profile: `Qwen3_5-Plus_prefill_TP2`
- parent focus: current fastest source-equivalent solution `06419529`
- baseline evidence: qwen_tp2 tokens=1 passes with about 520 us; real FlashInfer comparison remains blocked by the local FlashInfer worker.

## Strategies

0. `6_0` `grouped_gemm_atomic_free_reduce`

Replace only GEMM2/finalize accumulation with a compact atomic-free partial workspace plus finalize reducer for the T=1 path. Keep GEMM1, scheduling, and vectorized FP4 GEMM2 math unchanged.

1. `6_1` `single_token_slot_scheduling`

Specialize only scheduling for T=1/TOPK<=10 by compacting TOPK local slots directly and launching GEMM2 over the TOPK upper bound. Keep parent GEMM2 and completion finalize behavior unchanged.

2. `6_2` `tiled_fp4_dequantization`

Retile GEMM2 dequant so each stage2 thread computes a two-column H microtile for one active slot, reusing mid activations across adjacent output columns while preserving the parent finalize path.
