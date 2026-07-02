# Executor History: 1_0

- parent_id: `0e31b197`
- action_category: `stage1_intermediate_reuse`
- candidate_code: `iteration/1/executor/1_0/kernel.py`
- files touched:
  - `iteration/1/executor/1_0/src/fused_moe_kernel.cu`

## Change
Replaced the single CUDA kernel with two kernels:

1. `stage1_intermediate_kernel` computes `silu(gate) * up` once per `(token, topk, I)`.
2. `stage2_output_kernel` reuses the intermediate to compute output `[M, H]`.

The Python entrypoint and C++ binding contract remain unchanged.
