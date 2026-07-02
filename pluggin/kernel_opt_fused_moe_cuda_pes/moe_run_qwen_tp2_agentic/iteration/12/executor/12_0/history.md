# Executor 12_0 History

Parent id: `84634c40`

Strategy applied: `direct_fp8_scale_decode`

Concrete edit summary:
- Removed the Python-side `to(torch.float32)` conversions for `gemm1_weights_scale` and `gemm2_weights_scale`.
- Updated the extension contract to accept FlashInfer-style `float8_e4m3fn` scale tensors directly.
- Added device-side E4M3FN scale decode and changed GEMM1/GEMM2 scale reads from fp32 pointers to raw fp8 bytes.
- Preserved the FlashInfer-compatible `run(...)` surface, packed FP4 weights, routing semantics, and single-token grouped scheduling path.

Evaluation:
- Correctness: `PASS`, `max_rel=0.00023841856454964727`.
- Profile: `latency_us=244.9280023574829`.
- FlashInfer comparison remained blocked by worker `-11`.
