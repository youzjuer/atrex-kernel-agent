# Planner Plan - Generation 8

- operator: `flashinfer.trtllm_fp4_block_scale_moe`
- profile: `Qwen3_5-Plus_prefill_TP2`
- eval shape: `T=1`, `H=4096`, `I=1024`, `TOPK=10`, `E_local=256`, `E_global=512`
- selected parents: `a09b7b69`, `859dab14`, `66e4700d`

## Input Evidence

The current profile evidence is CUDA event timing, not NCU. Generation 7 selected
`a09b7b69` as best at `509.568 us`; `66e4700d` passed at `518.848 us`; and
`859dab14` remains a correctness-clean older grouped reducer parent. The local
FlashInfer comparison is still blocked by worker crash or GEMM runner errors, so
the next generation should optimize the local CUDA path directly.

Code inspection of `a09b7b69` localizes three remaining surfaces:

- GEMM1 still launches over original TOPK rows before reusable local-expert metadata
  exists, so every GEMM1 warp reloads `topk_ids` and recomputes local expert checks.
- GEMM1/GEMM2 already use vectorized block32 loads, but each packed byte is still
  decoded through repeated per-nibble software E2M1 logic.
- GEMM2 removed atomics, but still writes `[TOPK,H]` FP32 slot partials and reads
  them back in a separate finalize launch.

The lineages already include grouped GEMM, two tiled FP4 dequant attempts, compact
single-token scheduling, compact-first GEMM1 scheduling, a serial direct topk reducer,
and atomic-free partial/finalize. The strategies below avoid repeating those exact
plans by changing metadata shape, decode primitive, and epilogue reduction boundary.

## Search Log

| Source | Layer | Query | Finding | New? | Actionability |
|---|---|---|---|---|---|
| `README.md` | input | operator contract and qwen shape | FlashInfer-compatible signature; qwen_tp2 uses `T=1,H=4096,I=1024,TOPK=10,E_local=256`. | No | Defines predicates and semantics. |
| `profiles/*.json`, iteration 7 results | input | current latency | `a09b7b69` is best at `509.568 us`; `7_1` GEMM2 warp-sliced dequant regressed to `570.144 us`; FlashInfer is blocked. | Yes | Optimizes local CUDA path and avoids repeating regressed work partitioning. |
| `evolution_db.py lineage`, `database/state.json` | input | tried plans | Active lineages and siblings already cover compact scheduling, serial direct reducer, K/H dequant repartitioning, and atomic-free finalize. | No | Novelty guard. |
| `database/solutions/a09b7b69/src/fused_moe_kernel.cu` | input | best parent implementation | GEMM1 repeats topk/local checks; block32 helpers still decode per nibble; GEMM2 uses global slot partials plus finalize. | Yes | Direct support for all three strategies. |
| `database/solutions/66e4700d/src/fused_moe_kernel.cu` | input | compact-first sibling | Compact-first scheduling passed but used compact row remapping and remained slower than `a09b7b69`. | Yes | Supports a different fixed-slot scheduling mutation. |
| `database/solutions/859dab14/src/fused_moe_kernel.cu` | input | direct reducer sibling | Direct reducer fuses GEMM2/finalize but serializes local slots per H output. | Yes | Motivates slot-parallel CTA-local finalization. |
| `gpu-wiki/docs/kernel-opt/nvidia/cutedsl/sm120/sm120-moe-data-prep.md` | L1 | small topk data prep | Simple single-CTA metadata built once is preferred; sort-heavy and warp-aggregated alternatives regressed. | Yes | Supports fixed TOPK local-expert metadata. |
| `gpu-wiki/docs/kernel-opt/nvidia/common/blackwell/kernels/nvfp4-gemv.md` | L1 | FP4 unpack/decode | Wide FP4 loads must be paired with reduced byte/nibble unpack overhead. | Yes | Supports byte-pair dequant retile. |
| `gpu-wiki/docs/kernel-opt/nvidia/common/blackwell/techniques/vectorized-loads.md` | L1 | vectorized FP4 loads | 128/256-bit loads help only if unpack does not become the bottleneck. | Yes | Supports optimizing the decode primitive rather than changing launch shape. |
| `gpu-wiki/docs/kernel-opt/nvidia/common/blackwell/techniques/cache-policy.md` | L1 | cache policy | Current parent already follows streaming/reused cache-policy separation. | No | Deprioritizes another cache qualifier change. |
| `gpu-wiki/reference-kernels/nvidia/blackwell/cutedsl/flashinfer/blockscaled_contiguous_grouped_gemm_finalize_fusion.py` | L1 | grouped GEMM finalize | FlashInfer-style grouped finalize applies router scaling at the epilogue/finalize boundary. | Yes | Supports CTA-local GEMM2 finalize fusion. |
| `gpu-wiki/reference-kernels/nvidia/blackwell/cutedsl/flashinfer/blockscaled_contiguous_gather_grouped_gemm_swiglu_fusion.py` | L1 | grouped GEMM gather | FC1 reference uses token/expert metadata and processes only valid tiles. | Yes | Supports reusable scheduling metadata for GEMM1. |
| `reference-projects/README.md` | L2 | upstream projects | Only the submodule manifest is present in this checkout. | Yes | Use extracted gpu-wiki references instead. |
| public web | L3 | not used | Local evidence was sufficient. | No | Keeps the PES plan reproducible. |

## Strategies

### `8_0` - `grouped_expert_scheduling_fixed_topk_metadata`

- parent: `a09b7b69`
- action: Build a fixed `slot_to_local_expert[TOPK]` table before GEMM1, with
  each original top-k slot mapped to local expert id or `-1`. Make GEMM1, GEMM2
  partials, and finalize consume this table by original slot. Keep `mid[topk_slot,i]`
  layout and the parent atomic-free partial/finalize path; do not compact rows.
- expected impact: Removes repeated `topk_ids` loads and local range checks from
  GEMM1 while avoiding the compact-row remap already tried in `66e4700d`.
- risks: Savings may be small versus FP4 dot cost, and non-local slots still launch
  and return early. Preserve original top-k slot indexing for `topk_weights`.

### `8_1` - `tiled_fp4_dequantization_byte_pair_lut`

- parent: `a09b7b69`
- action: Retile only the software E2M1 decode primitive in the block32 fast paths.
  Keep current vectorized FP4 and scale loads, but decode packed bytes through a
  byte-pair lookup or PTX byte-unpack helper so one byte yields two FP4 float values
  with fewer integer dependency chains. Apply to GEMM1 and GEMM2 helpers; keep scalar
  branchless fallback.
- expected impact: Reduces decode instruction count in both projections without
  changing scheduling, finalization, or memory layout.
- risks: Constant-memory lookup divergence or extra registers can offset the decode
  savings. Keep the current branchless path as a fallback.

### `8_2` - `grouped_gemm_cta_local_finalize`

- parent: `a09b7b69`
- action: Replace the two-kernel GEMM2 partial/finalize boundary with one CTA-local
  grouped reducer. A CTA covers an H tile and TOPK slot dimension; slot threads compute
  existing GEMM2 dots in parallel, store weighted partials in shared memory, reduce the
  slot dimension inside the CTA, and write bf16 output.
- expected impact: Removes the `[TOPK,H]` global partial write/read and one finalize
  launch while preserving slot-level parallelism that the older direct reducer lost.
- risks: Fewer CTAs and shared-memory reduction may reduce occupancy. Keep the parent
  two-kernel path as fallback and preserve bias/router-weight semantics exactly.

Exhaustion: false.
