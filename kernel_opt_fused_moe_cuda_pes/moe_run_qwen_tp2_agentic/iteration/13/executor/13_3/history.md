# Executor 13_3 History

Parent id: `7b67f03d`

Strategy applied: `routing_128t_four_experts_per_thread`

Concrete edit summary:
- Retuned the fast routing kernel to 128 threads with four experts handled by each thread.
- Kept max-subtracted softmax and the existing staged MoE path from `7b67f03d`.

Evaluation:
- Correctness: `PASS`, `max_rel=0.00023841856454964727`.
- Profile: `latency_us=141.4719969034195`.
- FlashInfer comparison remained blocked by worker `-11`.
