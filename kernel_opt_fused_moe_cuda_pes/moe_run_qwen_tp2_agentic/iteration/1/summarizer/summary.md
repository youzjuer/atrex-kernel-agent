# Auto PES Summary - Generation 1

- `1_2` `grouped_gemm`: PASS, latency_us=1504.3519735336304, max_rel=0.00011920928227482364, flashinfer_us=None, target_status=BLOCKED, speedup_vs_flashinfer=None
- `1_1` `tiled_fp4_dequantization`: PASS, latency_us=1897.9840278625488, max_rel=0.00011920928227482364, flashinfer_us=None, target_status=BLOCKED, speedup_vs_flashinfer=None
- `1_0` `grouped_expert_scheduling`: PASS, latency_us=3136.8958950042725, max_rel=0.00011920928227482364, flashinfer_us=None, target_status=BLOCKED, speedup_vs_flashinfer=None

Best candidate: `1_2` with strategy `grouped_gemm`.
Real FlashInfer target not met; continue grouped scheduling, tiled FP4 dequant, and grouped GEMM.
