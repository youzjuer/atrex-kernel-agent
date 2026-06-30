# Baseline Report

## Environment
- GPU: NVIDIA L20D
- CUDA: `/usr/local/cuda`
- `nvcc`: 13.2.51
- PyTorch CUDA: available

## Baseline
- Candidate: `kernel.py`
- CUDA sources: `src/fused_moe_kernel.cpp`, `src/fused_moe_kernel.cu`
- Reference: `reference.py`
- Evaluator: `test_kernel.py`

## Validation
Command:

```bash
python3 test_kernel.py --kernel kernel.py --mode profile --m-values all --warmup 3 --rep 9 --json-out profiles/baseline_v0.json
```

Result:

```text
status: PASS
mean_latency_us: 154.15466328461966
max_abs: 2.270098775625229e-09
max_rel: 0.0002325464301975444
```

## Next PES Directions
- Reduce redundant gate/up dot recomputation across output columns.
- Split computation into stage1 intermediate + stage2 GEMM-like CUDA kernels.
- Add token/expert grouping to improve locality.
