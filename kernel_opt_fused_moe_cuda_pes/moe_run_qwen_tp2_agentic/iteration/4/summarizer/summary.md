# Auto PES Summary - Generation 4

- `4_1` `tiled_fp4_dequantization`: PASS, latency_us=544.48002576828, max_rel=0.00023841856454964727, flashinfer_us=None, target_status=BLOCKED, speedup_vs_flashinfer=None
- `4_0` `grouped_expert_scheduling`: PASS, latency_us=1298.975944519043, max_rel=0.00011920928227482364, flashinfer_us=None, target_status=BLOCKED, speedup_vs_flashinfer=None
- `4_2` `grouped_gemm`: PASS, latency_us=1381.9199800491333, max_rel=0.00011920928227482364, flashinfer_us=None, target_status=BLOCKED, speedup_vs_flashinfer=None

Best candidate: `4_1` with strategy `tiled_fp4_dequantization`.
Real FlashInfer target not met; continue grouped scheduling, tiled FP4 dequant, and grouped GEMM.
