# Kernel Opt Session Configuration

## Target
- platform: MI308X
- arch: CDNA3 / gfx942
- framework: FlyDSL
- dtype: FP8 GEMM inputs, BF16 stage/output activations, F32 scales, I32 metadata
- kernel type: FP8 PTPC fused_moe v2 full pipeline

## Execution
- execution_mode: local
- intended_gpu: AMD Instinct MI308X / CDNA3 / gfx942
- current_host_gpu: NVIDIA L20D (environment mismatch; real FlyDSL/AITER/rocprof gate is blocked here)

## Inputs
- kernel_demo: `/home/youchunbo/code/atrex-kernel-agent/gpu-wiki/reference-kernels/amd/cdna3/flydsl/FlyDSL/moe_fp8_ptpc_mi308x_atrex_v2/fused_moe_flydsl_fp8.py`
- shapes: E=512, topk=10, model_dim=4096, inter_dim=256
- tokens: 1/16/32/64/128/256/512
- reference: AITER `torch_moe_stage1` + `torch_moe_stage2` reference in `test_kernel.py`
- correctness threshold: `checkAllclose(..., rtol=1e-2, atol=1e-2)` error ratio <= 0.22; no NaN

## Hard Constraints
- profiling: AMD evidence must come from `tools/profile_kernel.sh` / `rocprofv3` on MI308X.
- timing: evaluator may use `torch.cuda.Event` / `do_bench` for latency, but not as bottleneck evidence.
- workspace: `kernel_opt_fused_moe_pes/`; commit every accepted generation in this workspace Git repo.
- knowledge base: `/home/youchunbo/code/atrex-kernel-agent/gpu-wiki/`.
- reference_project: `/home/youchunbo/code/atrex-kernel-agent/reference-projects/`.

## Hardware Spec
- MI308X GPU: AMD Instinct MI308X <- `<gpu-wiki>/docs/ref-docs/amd/flydsl/gfx942/cdna3-fused-moe-bf16-optimization.md:L13`
- MI308X architecture: CDNA3 (gfx942) <- `<gpu-wiki>/docs/ref-docs/amd/flydsl/gfx942/cdna3-fused-moe-bf16-optimization.md:L14`
- MI308X CU count: 80 <- `<gpu-wiki>/docs/ref-docs/amd/flydsl/gfx942/cdna3-fused-moe-bf16-optimization.md:L15`
- MI308X BF16 peak: 206 TFLOPS <- `<gpu-wiki>/docs/ref-docs/amd/flydsl/gfx942/cdna3-fused-moe-bf16-optimization.md:L16`
- MI308X FP8 peak: ~466 TFLOPS <- `<gpu-wiki>/docs/ref-docs/amd/flydsl/gfx942/cdna3-sage-attention-flydsl-optimization.md:L17`
- MI308X HBM bandwidth: 5.3 TB/s <- `<gpu-wiki>/docs/ref-docs/amd/flydsl/gfx942/cdna3-fused-moe-bf16-optimization.md:L17`
- MI308X LDS capacity/CU: 64 KB <- `<gpu-wiki>/docs/ref-docs/amd/flydsl/gfx942/cdna3-fused-moe-bf16-optimization.md:L18`

## Step 0 Roofline Analysis
- Stage1 FLOPs per token: `2 * topk * model_dim * (2 * inter_dim) = 2 * 10 * 4096 * 512 = 41,943,040`.
- Stage2 FLOPs per token: `2 * topk * inter_dim * model_dim = 2 * 10 * 256 * 4096 = 20,971,520`.
- Total GEMM FLOPs per token: `62,914,560`; total for M=512: `32.212 GFLOPs`.
- Arithmetic intensity is workload-dependent because routed experts reuse weights by token/expert distribution. Small M rows are launch/routing/weight-traffic dominated; larger M rows are compute-heavy but still far below theoretical FP8 peak due sparse MoE routing, per-token quantization, and two-stage pipeline overhead.
- Bound classification for this PES run: mixed; optimize stage1/stage2 MFMA throughput and reduce routing/quant/overhead buckets, using profile evidence before each accepted code change.

## Stop Conditions
Hardware-derived targets:
- FP8 compute-bound target: `466 * 90% = 419.4 TFLOPS`.
- BF16 compute-bound target: `206 * 90% = 185.4 TFLOPS`.
- memory-bound target: `5.3 * 90% = 4.77 TB/s`.

Fused-MoE task gate:
- preserve correctness on all formal token rows M=1/16/32/64/128/256/512.
- no profiler-visible `Memcpy DtoD`, `Memcpy DtoH`, or `Memcpy HtoD` in the v2 FlyDSL path.
- `other == 0.0` for all formal token rows.
- optimize against the archived v2 baseline metric, using mean `e2e_avg_us = 695.1714` across formal token rows as the seed scalar score denominator.

## Baseline From gpu-wiki Archive
Source: `<gpu-wiki>/reference-kernels/amd/cdna3/flydsl/FlyDSL/moe_fp8_ptpc_mi308x_atrex_v2/README.md:L81-L120`.

| M | routing | quant | stage1 | stage2 | overhead | other | kernel sum | e2e avg | e2e min |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 17.7 | 5.1 | 11.6 | 7.9 | 1.5 | 0.0 | 43.9 | 544.9 | 528.9 |
| 16 | 12.2 | 8.3 | 91.4 | 54.6 | 2.4 | 0.0 | 169.0 | 601.0 | 575.2 |
| 32 | 12.0 | 8.7 | 120.9 | 85.3 | 4.3 | 0.0 | 231.2 | 587.6 | 563.8 |
| 64 | 13.1 | 8.4 | 180.5 | 128.6 | 4.6 | 0.0 | 335.2 | 663.4 | 620.2 |
| 128 | 13.9 | 9.3 | 244.4 | 170.6 | 5.2 | 0.0 | 443.4 | 735.1 | 714.8 |
| 256 | 15.8 | 14.4 | 275.1 | 185.5 | 5.7 | 0.0 | 496.4 | 796.2 | 782.6 |
| 512 | 20.1 | 23.5 | 334.2 | 200.0 | 5.9 | 0.0 | 583.8 | 938.0 | 925.4 |

## Evolve Config
- mode: Full-Agent PES
- n_candidates: 3
- num_islands: 3
- max_generations: 30
- no_improve_patience: 5
- score_metric: speedup_vs_baseline_mean_e2e_avg_us
- baseline_latency_us: 695.1714

## Environment Status
- Blocked on this host for real evaluator execution:
  - current GPU is NVIDIA L20D, not AMD MI308X.
  - `AITER_BASE` is unset and `aiter` is not importable.
  - `flydsl` is not importable.
  - `rocprofv3` is not available.

## Task Context
- platform: MI308X
- arch: CDNA3 / gfx942
- framework: FlyDSL
- dtype: FP8 GEMM inputs + BF16 activations/output
- shapes: task16, E=512, topk=10, model_dim=4096, inter_dim=256, M in [1,16,32,64,128,256,512]
- correctness_threshold: error ratio <= 0.22 under `checkAllclose(..., rtol=1e-2, atol=1e-2)`
- stop_condition: hardware targets plus fused-MoE task gate above
- reference_project: `/home/youchunbo/code/atrex-kernel-agent/reference-projects/`

## ISA Optimization Targets

### AMD
- Global memory: increase the share of vectorized buffer loads where profile evidence localizes scalar load pressure.
- LDS memory: preserve aligned vector LDS operations; avoid known FP8 LDS stride/bank-conflict regressions.
- Registers: keep `vgpr_spill_count == 0`; keep scratch load/store counts at 0.
- Routing/quant overhead: reduce routing, quant, and overhead buckets without introducing profiler-visible memcpy.
- Compute utilization: improve stage1 and stage2 MFMA utilization for M >= 64.
- Pipeline: reduce `s_waitcnt`/memory dependency stalls only when rocprof evidence localizes them.
