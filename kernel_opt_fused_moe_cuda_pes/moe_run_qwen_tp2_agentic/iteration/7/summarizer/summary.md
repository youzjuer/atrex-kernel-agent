# Auto PES Summary - Generation 7

- `7_2` `grouped_gemm_compact_atomic_free_finalize`: PASS, latency_us=509.5679759979248, max_rel=0.00023841856454964727, flashinfer_us=None, target_status=BLOCKED, speedup_vs_flashinfer=None
- `7_0` `grouped_expert_scheduling`: PASS, latency_us=518.8480019569397, max_rel=0.00023841856454964727, flashinfer_us=None, target_status=BLOCKED, speedup_vs_flashinfer=None
- `7_1` `tiled_fp4_dequantization`: PASS, latency_us=570.14399766922, max_rel=0.00011920928227482364, flashinfer_us=None, target_status=BLOCKED, speedup_vs_flashinfer=None

Best candidate: `7_2` with strategy `grouped_gemm_compact_atomic_free_finalize`.
Real FlashInfer target not met; continue grouped scheduling, tiled FP4 dequant, and grouped GEMM.
