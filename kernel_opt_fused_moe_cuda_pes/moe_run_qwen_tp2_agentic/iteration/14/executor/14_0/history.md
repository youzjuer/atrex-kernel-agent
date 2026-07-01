# Executor 14_0 History

Parent id: `7b67f03d`

Strategy applied: `routing_warp_shuffle_reduction`

Concrete edit summary:
- Replaced shared-memory tree reductions in the fast routing kernel with warp-shuffle reductions plus one warp-level cross-warp reduction.
- Kept the 256-thread, two-experts-per-thread routing shape.

Evaluation:
- Correctness: `PASS`, `max_rel=0.00023841856454964727`.
- Profile: `latency_us=139.00800049304962`.
- FlashInfer comparison remained blocked by worker timeout.
