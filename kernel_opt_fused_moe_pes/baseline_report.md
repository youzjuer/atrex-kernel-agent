# Baseline Report

## Baseline
- Seed implementation: archived atrex-open FlyDSL v2 FP8 PTPC fused_moe full pipeline.
- Kernel: `kernel.py`
- Support files: `moe_kernels.py`, `kernels/`
- Evaluator entry: `test_kernel.py`

## Correctness
- Archived gpu-wiki result: `7 passed in 8.08s`.
- Local run status: blocked on environment mismatch. This host has NVIDIA L20D and lacks AITER/FlyDSL/rocprofv3.

## Performance
- Archived scalar baseline for PES scoring: mean `e2e_avg_us = 695.1714` across M=1/16/32/64/128/256/512.
- Full baseline table is recorded in `README.md`.

## Next Step
Run generation 1 on an MI308X environment with:

```bash
export AITER_BASE=/path/to/aiter
cd /home/youchunbo/code/atrex-kernel-agent/kernel_opt_fused_moe_pes
timeout 60 python test_kernel.py --kernel kernel.py --mode correctness
python test_kernel.py --kernel kernel.py --mode profile
```

## Local Verification
- `python3 -m py_compile kernel.py moe_kernels.py kernels/*.py test_kernel.py`: PASS.
- `timeout 60 python3 test_kernel.py --kernel kernel.py --mode correctness --m-values 1`: blocked before candidate import because `AITER_BASE` is unset.
