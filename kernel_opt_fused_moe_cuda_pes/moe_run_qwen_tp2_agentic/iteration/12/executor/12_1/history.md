# Executor 12_1 History

Parent id: `99555166`

Strategy applied: `single_token_cuda_routing_topk`

Concrete edit summary:
- Added a single-token CUDA routing kernel for `routing_method_type == 0`, no `routing_bias`, `top_k <= 16`.
- The fast routing kernel computes bf16 softmax denominator, top-k ids, and top-k weights in one extension kernel.
- `run(...)` uses this fast path for Qwen TP2 and keeps the existing Python `compute_routing` fallback for other routing surfaces.
- Preserved direct fp8 scale decode, packed FP4 weights, and the existing staged MoE kernels from `99555166`.

Evaluation:
- Correctness: `PASS`, `max_rel=0.00023841856454964727`.
- Profile: `latency_us=149.21599626541138`.
- FlashInfer comparison remained blocked by worker `-11`.
