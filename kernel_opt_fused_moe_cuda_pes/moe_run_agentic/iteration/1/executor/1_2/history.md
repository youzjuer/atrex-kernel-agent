# Executor History: 1_2

Parent solution: `dec7821d`

Strategy applied: `grouped_gemm_tiling`

Implemented the assigned grouped GEMM tiling category in the CUDA extension while preserving the
FlashInfer-compatible Python `run(...)` surface. The staged scalar dot loops were replaced with:

- A local-route grouping path that counts valid local expert routes, computes expert offsets, and
  scatters route rows into an expert-grouped compact route list.
- A CTA-level tiled stage-1 kernel over route tiles and intermediate columns using an 8x16 output
  tile and 4-way cooperative K split for the gate/up FP4 dot products, followed by the existing
  SwiGLU semantics in FP32.
- A CTA-level tiled stage-2 kernel over grouped route tiles and hidden columns using an 8x16 output
  tile and 4-way cooperative K split for the down projection, accumulating weighted route
  contributions into a FP32 output buffer before bf16 finalization.

Files touched:

- `src/fused_moe_kernel.cu`
- `history.md`
- `executor_result.json`

Self-check:

- Python import smoke check: ok
- CUDA extension loader/compile smoke check: ok
