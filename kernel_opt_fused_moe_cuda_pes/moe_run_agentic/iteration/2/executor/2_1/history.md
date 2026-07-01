# Executor History: 2_1

- Parent id: f1aa1c2d
- Action category: tiled_fp4_dequantization_grouped_gemm
- Strategy: preserve the grouped GEMM parent structure while replacing only the FP4 load/dequant path inside `stage1_grouped_gemm_kernel` and `stage2_grouped_gemm_kernel` with scale-tiled packed decode helpers.

## Edit Summary

- Replaced the switch-based scalar E2M1 decoder and per-element `load_fp4_scaled` helper with a device constant LUT decoder.
- Added K-split-aware scale-tiled FP4 accumulation helpers for the grouped GEMM inner loops.
- The helpers load one block scale per scale vector, use aligned `uint32_t` packed FP4 loads when the full packed word stays inside the scale tile, and fall back to byte loads for unaligned/tail cases.
- Updated stage1 and stage2 grouped GEMM inner loops to iterate by scale tile and call the new helpers while preserving route grouping, 8x16 tiles, 4-way K split, atomics, and output semantics.

## Files Touched

- `src/fused_moe_kernel.cu`
- `history.md`
- `executor_result.json`

## Self Check

- `timeout 60 python -c "import importlib.util; spec=importlib.util.spec_from_file_location('cand','/home/youchunbo/code/atrex-kernel-agent/kernel_opt_fused_moe_cuda_pes/moe_run_agentic/iteration/2/executor/2_1/kernel.py'); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m); print('self-check import ok')"`
- Result: ok
