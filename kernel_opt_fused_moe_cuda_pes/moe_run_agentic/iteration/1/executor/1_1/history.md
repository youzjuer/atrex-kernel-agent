# Executor History

- Child: 1_1
- Parent id: dec7821d
- Action category: tiled_fp4_dequantization
- Strategy: replace scalar per-element FP4 dequantization in the existing two-stage CUDA MoE kernels with tile-level decode that reuses each block scale across its scale vector and uses packed 32-bit FP4 loads when aligned.

## Edits

- Updated `src/fused_moe_kernel.cu`.
- Replaced the switch-based scalar FP4 decoder and `load_fp4_scaled` helper with a device 16-entry E2M1 lookup table.
- Added scale-tile accumulation helpers for GEMM1 and GEMM2 that:
  - load each FP4 scale once per scale tile derived from tensor shape,
  - decode packed nibbles in increasing K order,
  - use aligned `uint32_t` loads for eight FP4 values at a time,
  - fall back to byte/nibble handling for odd boundaries and tails.
- Changed the stage1 and stage2 dot-product loops to iterate by scale columns while preserving the existing staged algorithm, routing behavior, activation, bias handling, and output order.

## Files Touched

- `src/fused_moe_kernel.cu`
- `history.md`
- `executor_result.json`

## Self Check

- `timeout 60 python -c "import importlib.util; spec=importlib.util.spec_from_file_location('cand','kernel.py'); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)"`: ok
- `timeout 120 python -c "import importlib.util; spec=importlib.util.spec_from_file_location('cand','kernel.py'); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m); m._load_ext()"`: ok

No benchmark or evaluator command was run.
