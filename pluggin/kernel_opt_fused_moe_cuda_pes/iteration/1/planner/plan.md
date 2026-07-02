# Generation 1 Planner Plan

## Parent
- parent_id: `0e31b197`
- parent code: `database/solutions/0e31b197/kernel.py`
- baseline latency: `154.15466328461966 us`

## Strategy 1_0
- action_category: `stage1_intermediate_reuse`
- action_description: Split the fused baseline into two CUDA kernels. Stage1 computes
  `silu(gate) * up` once per `(token, topk, intermediate)` into a temporary tensor; stage2 reuses
  that intermediate for all output hidden columns.
- evidence_chain: v0 maps one thread to `(token, output_hidden)` and recomputes the gate/up dot for
  every output hidden column -> redundant work is `O(M * TOPK * H_out * I * H_in)` -> computing
  stage1 once changes the dot-product work to `O(M * TOPK * I * H_in)`.
- expected_impact: Large latency reduction for the local fp32 shape because redundant gate/up work is
  removed.
- risks: Extra temporary allocation and second launch can hurt tiny shapes; correctness must remain
  bit-close to PyTorch reference.

## Strategy 1_1
- action_category: `block_parallel_reduction`
- action_description: Keep single fused output but parallelize dot products across threads in a block.
- status: planned only; not executed in this first CUDA smoke generation.

## Strategy 1_2
- action_category: `expert_grouping`
- action_description: Group token/topk entries by expert before stage kernels to improve W1/W2 locality.
- status: planned only; not executed in this first CUDA smoke generation.
