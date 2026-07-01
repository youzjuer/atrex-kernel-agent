# Auto PES Summary - Generation 6

- `6_1` `single_token_slot_scheduling`: PASS, latency_us=533.8559746742249, max_rel=0.00023841856454964727, flashinfer_us=None, target_status=BLOCKED, speedup_vs_flashinfer=None
- `6_0` `grouped_gemm_atomic_free_reduce`: PASS, latency_us=554.1120171546936, max_rel=0.00023841856454964727, flashinfer_us=None, target_status=BLOCKED, speedup_vs_flashinfer=None
- `6_2` `tiled_fp4_dequantization`: PASS, latency_us=566.3040280342102, max_rel=0.00023841856454964727, flashinfer_us=None, target_status=BLOCKED, speedup_vs_flashinfer=None

Best candidate: `6_1` with strategy `single_token_slot_scheduling`.
Real FlashInfer target not met; continue grouped scheduling, tiled FP4 dequant, and grouped GEMM.
