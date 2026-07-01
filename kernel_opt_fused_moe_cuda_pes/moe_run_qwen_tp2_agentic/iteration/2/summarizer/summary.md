# Auto PES Summary - Generation 2

- `2_2` `grouped_gemm`: PASS, latency_us=1392.2879695892334, max_rel=0.00011920928227482364, flashinfer_us=None, target_status=BLOCKED, speedup_vs_flashinfer=None
- `2_0` `grouped_expert_scheduling`: PASS, latency_us=1399.5519876480103, max_rel=0.00011920928227482364, flashinfer_us=None, target_status=BLOCKED, speedup_vs_flashinfer=None
- `2_1` `tiled_fp4_dequantization`: PASS, latency_us=1701.3440132141113, max_rel=0.00011920928227482364, flashinfer_us=None, target_status=BLOCKED, speedup_vs_flashinfer=None

Best candidate: `2_2` with strategy `grouped_gemm`.
Real FlashInfer target not met; continue grouped scheduling, tiled FP4 dequant, and grouped GEMM.
