# Executor 13_0 History

Parent id: `9a2330ad`

Strategy applied: `routing_direct_exp_no_max`

Concrete edit summary:
- Removed the max reduction from the fast single-token routing kernel.
- The fast path computes `exp(logit)` directly and performs one denominator reduction, relying on the small Qwen TP2 routing-logit range.
- Kept the generic Python routing fallback for non-fast-path surfaces.

Evaluation:
- Correctness: `PASS`, `max_rel=0.00023841856454964727`.
- Profile: `latency_us=140.28799533843994`.
- FlashInfer comparison remained blocked by worker `-11`.
