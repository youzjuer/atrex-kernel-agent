# Executor 14_1 History

Parent id: `7b67f03d`

Strategy applied: `stage1_warp_block128`

Concrete edit summary:
- Changed `stage1_activation_warp_k_kernel` launch from 256 to 128 threads per block.
- Left routing, fp8 scale decode, stage2, compact metadata, and finalize unchanged.

Evaluation:
- Correctness: `PASS`, `max_rel=0.00023841856454964727`.
- Profile without FlashInfer compare: `latency_us=138.047993183136`.
