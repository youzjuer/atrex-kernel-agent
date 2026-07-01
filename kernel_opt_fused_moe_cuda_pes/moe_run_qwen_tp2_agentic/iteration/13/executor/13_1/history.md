# Executor 13_1 History

Parent id: `9a2330ad`

Strategy applied: `routing_256t_two_experts_per_thread`

Concrete edit summary:
- Kept the stable max-subtracted fast routing math from `9a2330ad`.
- Changed the routing kernel launch from 512 threads to 256 threads, with each thread processing two experts for the max and denominator reductions.
- Preserved direct fp8 scale decode and the existing staged MoE path.

Evaluation:
- Correctness: `PASS`, `max_rel=0.00023841856454964727`.
- Profile: `latency_us=133.12000036239624`.
- FlashInfer comparison remained blocked by worker `-11`.
