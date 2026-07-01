# Planner Plan - Generation 7

- operator: `flashinfer.trtllm_fp4_block_scale_moe`
- profile: `Qwen3_5-Plus_prefill_TP2`
- eval shape: `T=1`, `H=4096`, `I=1024`, `TOPK=10`, `E_local=256`, `E_global=512`
- selected parents: `6ebb0ad6`, `83d98be7`, `b1433778`

## Input Evidence

The current profile evidence is CUDA event timing, not NCU. `profiles/auto_baseline.json`
passes qwen_tp2 T=1 at `527.168 us`; selected parents pass at `533.856 us`
(`6ebb0ad6`), `566.304 us` (`83d98be7`), and `1381.920 us` (`b1433778`).
The real FlashInfer comparison is still blocked by a local worker crash/error.

For `T=1,H=4096,I=1024,TOPK=10`, useful GEMM work is about `251.66 MFLOP`, so the
best current paths remain below about `0.5 TFLOP/s` of useful software-FP4 MoE
work. Code inspection localizes the remaining obvious costs:

- `6ebb0ad6` launches warp-sliced GEMM1 over all TOPK slots before compacting
  local slots, so non-local routes still consume stage1 warp scheduling and checks.
- `83d98be7` reuses mid activations across two adjacent H columns in GEMM2, but
  each participating thread still walks the full I scale-loop serially.
- `6ebb0ad6` uses compact TOPK scheduling for GEMM2, but the finalize path still
  uses `atomicAdd`, `__threadfence`, `completion_counts`, `out_accum`, and output
  zeroing for the multi-local-slot case.

## Search Log

| Source | Layer | Query | Finding | New? | Actionability |
|---|---|---|---|---|---|
| `workspace README.md` | input | operator contract and qwen_tp2 shape | FlashInfer-compatible signature; T=1 eval uses H=4096, I=1024, E_global=512, TOPK=10, E_local=256. | No | Defines predicates and semantics. |
| `profiles/auto_baseline.json`, iteration 6 evidence | input | current latency and target status | Current profile passes at 527.168 us; generation 6 children pass at 533.856, 554.112, and 566.304 us; FlashInfer remains blocked. | Yes | Optimizes local CUDA path rather than comparing to FlashInfer. |
| `database/state.json`, `evolution_db.py lineage` | input | tried plans | Lineages and siblings already cover active scheduling, GEMM1 warp-slice, GEMM2 H-tiling, topk direct reducer, and atomic-free reduce. | No | Novelty guard. |
| `database/solutions/6ebb0ad6/src/fused_moe_kernel.cu` | input | fastest parent | GEMM1 runs before compact local-slot scheduling; compact GEMM2 still uses atomics and completion counts. | Yes | Supports `7_0` and `7_2`. |
| `database/solutions/83d98be7/src/fused_moe_kernel.cu` | input | GEMM2 dequant | Two-column GEMM2 still leaves a serial I-loop per thread. | Yes | Supports `7_1`. |
| `gpu-wiki/docs/kernel-opt/nvidia/cutedsl/sm120/sm120-moe-data-prep.md` | L1 | small topk data prep | Single-CTA small-topk scheduling is preferred; compact maps are built once for downstream grouped GEMMs. | Yes | Supports front-loaded scheduling. |
| `gpu-wiki/docs/ref-docs/nvidia/cutedsl/sm120/sm120-moe-data-prep-optimization.md` | L1 | retained/regressed data-prep patterns | V7/V9 retained simple contention controls; warp aggregation, CUB sort, and warp-specialized histogram regressed. | Yes | Avoids sort-heavy scheduling changes. |
| `gpu-wiki/docs/ref-docs/nvidia/cuda/sm120/sm120-nvfp4-split-k-gemv-bf16-optimization.md` | L1 | NVFP4 small-M long-K decomposition | K-splitting and two-phase FP32 partial/reduce help when small-M long-K work under-fills or serializes too much; gate by alignment. | Yes | Supports GEMM2 K-parallel dequant and compact partial/finalize. |
| `gpu-wiki/docs/kernel-opt/nvidia/common/blackwell/techniques/vectorized-loads.md` | L1 | FP4 dequant loads | 128/256-bit FP4 loads, cache-policy separation, and many lanes per row/dot are useful for software FP4 GEMV-like work. | No | Supports `7_1`. |
| `gpu-wiki/docs/kernel-opt/nvidia/common/blackwell/kernels/grouped-gemm.md`, `fused-moe.md` | L1 | grouped MoE dataflow | Grouped MoE keeps compact rows and applies router-weighted combine in the epilogue/finalize boundary. | No | Supports `7_2`. |
| `reference-projects/README.md`, `reference-projects/{flashinfer,DeepGEMM,cutlass,aiter}` | L2 | upstream code | Submodule directories exist but are empty in this checkout. | Yes | No direct code source available; use extracted gpu-wiki references. |
| public web | L3 | not used | Local evidence was sufficient. | No | Keeps plan reproducible. |

## Strategies

### `7_0` - `grouped_expert_scheduling`

- parent: `6ebb0ad6`
- action: Front-load the `T=1/TOPK<=10` local-slot compaction before GEMM1. Build
  `local_slots`, `local_experts`, `token_counts`, and `local_slot_count` first,
  then launch GEMM1 over the TOPK upper bound with device guards and write
  `mid_compact[slot_idx, i]`. Reuse the same compact rows for GEMM2.
- expected impact: Skips stage1 work for non-local TOPK routes and removes repeated
  local-expert checks from GEMM1's hot path.
- risks: If all routed slots are local, speedup may be small. The original TOPK
  slot index must still be preserved for `topk_weights`.

### `7_1` - `tiled_fp4_dequantization`

- parent: `83d98be7`
- action: Replace GEMM2's serial per-thread I-loop with a warp-sliced K/I dot for
  the qwen path. One warp owns one `(local_slot, h)` output; lanes process disjoint
  32-element scale blocks, vector-load mid and packed FP4 weights, then reduce
  lane partials through warp shuffles into the parent finalize path.
- expected impact: Shortens GEMM2's serial software-FP4 dequant chain and applies
  the successful GEMM1 warp-slice pattern to the down projection.
- risks: More warps and shuffle reductions may cost more than the serial loop.
  Gate to `I_scale_vec==32` and keep the parent two-column path as fallback.

### `7_2` - `grouped_gemm_compact_atomic_free_finalize`

- parent: `6ebb0ad6`
- action: Keep the fastest compact single-token schedule and GEMM1 path, but make
  GEMM2 write FP32 partials into a fixed `[TOPK, H]` workspace. A finalize kernel
  loops only `local_slot_count` slots, applies `topk_weights` and `gemm2_bias`,
  casts to bf16, and writes zero when no local slots exist.
- expected impact: Removes per-output atomics, threadfence, completion-count
  synchronization, and large zero-initialization traffic from the current compact
  TOPK GEMM2 path.
- risks: Adds one launch and a compact FP32 workspace. If GEMM1 dominates or atomics
  are hidden, this can be neutral or slower.

Exhaustion: false.
