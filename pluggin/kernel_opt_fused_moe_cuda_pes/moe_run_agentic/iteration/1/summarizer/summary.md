# Auto PES Summary - Generation 1

- `1_1` `tiled_fp4_dequantization`: PASS, latency_us=223.7440049648285, max_rel=0.0
- `1_2` `grouped_gemm_tiling`: PASS, latency_us=247.42400646209717, max_rel=0.0
- `1_0` `grouped_expert_scheduling`: PASS, latency_us=330.81600069999695, max_rel=0.0

Best candidate: `1_1` with strategy `tiled_fp4_dequantization`.
Next direction: keep planner/executor focused on grouped scheduling, tiled FP4 dequant, and grouped GEMM.
