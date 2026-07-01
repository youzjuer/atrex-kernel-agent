# Executor 11_1 History

Parent id: `84634c40`

Strategy applied: `single_token_stage2_block128`

Concrete edit summary:
- Changed only the single-token `stage2_single_token_slots_down_partials_kernel` launch block from 32 to 128 threads.
- Preserved the FlashInfer-compatible operator surface, slot compaction, bias-hoisted stage2 partial, and slim finalize path from `84634c40`.

Evaluation:
- `PASS`, `max_rel=0.00023841856454964727`, `latency_us=504.5120120048523`.
- FlashInfer comparison remained blocked by worker `-11`.
