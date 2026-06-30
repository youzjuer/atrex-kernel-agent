# CUDA Fused MoE PES Workspace

## Target
- platform: NVIDIA L20D
- arch: CUDA / sm_89-compatible local GPU
- framework: CUDA extension loaded through PyTorch
- kernel type: fused MoE forward
- dtype: fp32 baseline for local correctness and PES plumbing

## Execution
- execution_mode: local
- local_gpu: NVIDIA L20D
- CUDA_HOME: `/usr/local/cuda`
- evaluator: `test_kernel.py`

## Inputs
- shapes: `M in [8, 16, 32]`, `E=8`, `TOPK=2`, `H=64`, `I=32`
- input tensors:
  - `x`: `[M, H]` fp32
  - `w1`: `[E, 2 * I, H]` fp32
  - `w2`: `[E, H, I]` fp32
  - `topk_ids`: `[M, TOPK]` int64
  - `topk_weights`: `[M, TOPK]` fp32, normalized
- reference: PyTorch implementation in `reference.py`
- correctness threshold: max abs error <= `1e-3`, max rel error <= `1e-3`

## Full-Agent PES Reference
This workspace follows the LoongFlow-style PES layout from the cloned reference repository:

- planner/executor/summarizer traces under `iteration/<K>/`
- population/checkpoint state under `database/`
- evaluator evidence under `profiles/<K>/<child>/`

The reference repo's MoE trace is used for workflow structure and strategy vocabulary, but this
workspace intentionally emits CUDA code for the current local environment rather than the archived
Triton or FlyDSL implementations.

## Baseline
- `kernel.py`: Python wrapper and PyTorch extension loader.
- `src/fused_moe_kernel.cpp`: C++ binding and validation.
- `src/fused_moe_kernel.cu`: one-kernel fused baseline.

The baseline launches one CUDA thread per `(token, hidden_out)` element and computes both MoE GEMMs
inside the thread. This is deliberately simple and correct; PES candidates should improve data reuse,
parallelism, and decomposition.

## Stop Conditions
- Correctness PASS on all local M values.
- Performance score >= `2.0x` speedup over v0 baseline, or budget exhausted.
- Candidate must be CUDA source, not FlyDSL or Triton.

## Evolve Config
- mode: Full-Agent PES
- n_candidates: 3
- num_islands: 3
- max_generations: 10
- no_improve_patience: 5
- score_metric: speedup_vs_baseline_latency
