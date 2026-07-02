# Auto PES Summary - Generation 5

- `5_0` `grouped_expert_scheduling`: PASS, latency_us=543.8399910926819, max_rel=0.00023841856454964727, flashinfer_us=None, target_status=BLOCKED, speedup_vs_flashinfer=None
- `5_1` `tiled_fp4_dequantization`: PASS, latency_us=548.5439896583557, max_rel=0.00023841856454964727, flashinfer_us=None, target_status=BLOCKED, speedup_vs_flashinfer=None
- `5_2` `grouped_gemm`: PASS, latency_us=803.551971912384, max_rel=0.00023841856454964727, flashinfer_us=None, target_status=BLOCKED, speedup_vs_flashinfer=None

Best candidate: `5_0` with strategy `grouped_expert_scheduling`.
Real FlashInfer target not met; continue grouped scheduling, tiled FP4 dequant, and grouped GEMM.
