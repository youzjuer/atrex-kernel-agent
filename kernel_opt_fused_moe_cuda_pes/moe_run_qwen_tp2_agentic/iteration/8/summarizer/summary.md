# Auto PES Summary - Generation 8

- `8_0` `grouped_expert_scheduling_fixed_topk_metadata`: PASS, latency_us=511.4240050315857, max_rel=0.00023841856454964727, flashinfer_us=None, target_status=BLOCKED, speedup_vs_flashinfer=None
- `8_2` `grouped_gemm_cta_local_finalize`: PASS, latency_us=515.8720016479492, max_rel=0.00023841856454964727, flashinfer_us=None, target_status=BLOCKED, speedup_vs_flashinfer=None
- `8_1` `tiled_fp4_dequantization_byte_pair_lut`: PASS, latency_us=867.1039938926697, max_rel=0.00023841856454964727, flashinfer_us=None, target_status=BLOCKED, speedup_vs_flashinfer=None

Best candidate: `8_0` with strategy `grouped_expert_scheduling_fixed_topk_metadata`.
Real FlashInfer target not met; continue grouped scheduling, tiled FP4 dequant, and grouped GEMM.
