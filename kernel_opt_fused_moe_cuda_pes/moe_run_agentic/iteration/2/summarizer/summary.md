# Auto PES Summary - Generation 2

- `2_2` `grouped_gemm_down_projection`: PASS, latency_us=191.52000546455383, max_rel=0.0, flashinfer_us=None, target_status=BLOCKED, speedup_vs_flashinfer=None
- `2_1` `tiled_fp4_dequantization_grouped_gemm`: PASS, latency_us=209.9200040102005, max_rel=0.0, flashinfer_us=None, target_status=BLOCKED, speedup_vs_flashinfer=None
- `2_0` `grouped_expert_scheduling_inline`: PASS, latency_us=250.5280077457428, max_rel=0.0, flashinfer_us=None, target_status=BLOCKED, speedup_vs_flashinfer=None

Best candidate: `2_2` with strategy `grouped_gemm_down_projection`.
Real FlashInfer target not met; continue grouped scheduling, tiled FP4 dequant, and grouped GEMM.
