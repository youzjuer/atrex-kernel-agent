# Auto PES Summary - Generation 3

- `3_1` `tiled_fp4_dequantization`: PASS, latency_us=1333.631992340088, max_rel=0.00011920928227482364, flashinfer_us=None, target_status=BLOCKED, speedup_vs_flashinfer=None
- `3_0` `grouped_expert_scheduling`: PASS, latency_us=1432.8320026397705, max_rel=0.00011920928227482364, flashinfer_us=None, target_status=BLOCKED, speedup_vs_flashinfer=None
- `3_2` `grouped_gemm`: PASS, latency_us=1723.199963569641, max_rel=0.00011920928227482364, flashinfer_us=None, target_status=BLOCKED, speedup_vs_flashinfer=None

Best candidate: `3_1` with strategy `tiled_fp4_dequantization`.
Real FlashInfer target not met; continue grouped scheduling, tiled FP4 dequant, and grouped GEMM.

## Stable Retest

- `3_1` with `warmup=2, rep=5`: PASS, latency_us=1114.016056060791, max_rel=0.00011920928227482364, target_status=BLOCKED.
- Current source baseline with `warmup=2, rep=5`: PASS, latency_us=1055.1999807357788, max_rel=0.00011920928227482364, target_status=BLOCKED.
- Conclusion: do not promote `3_1`; its `rep=1` DB score was not stable against the current source baseline.
