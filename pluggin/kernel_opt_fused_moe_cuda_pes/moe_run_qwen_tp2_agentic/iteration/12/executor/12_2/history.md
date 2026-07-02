# Executor 12_2 History

Parent id: `64441d10`

Strategy applied: `parallel_single_token_cuda_routing_topk`

Concrete edit summary:
- Replaced the single-thread fast routing kernel from `12_1` with a 512-thread block implementation.
- Routing now loads logits into shared memory, performs parallel max and softmax denominator reductions, and lets thread 0 select the top-10 ids from shared logits.
- Kept the same fast-path guard and Python routing fallback from `12_1`.

Evaluation:
- Correctness: `PASS`, `max_rel=0.00023841856454964727`.
- Profile: `latency_us=139.71200585365295`.
- FlashInfer comparison remained blocked by worker `-11`.
