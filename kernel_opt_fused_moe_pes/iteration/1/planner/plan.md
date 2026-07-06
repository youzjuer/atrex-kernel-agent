# Iteration 1 Planner Plan

## Execution Status

Execution and evaluation are blocked on this host pending an AMD MI308X / CDNA3 / gfx942 environment with AITER, FlyDSL, and rocprofv3 available. The current host is NVIDIA L20D, `AITER_BASE` is unset, `aiter` is not importable, `flydsl` is not importable, and `rocprofv3` is not available. This plan is research only and does not invent any new performance numbers.

## Input Evidence

- Workspace target: MI308X, CDNA3/gfx942, FlyDSL, FP8 PTPC fused_moe v2 full pipeline; formal rows M=1/16/32/64/128/256/512.
- Parent selection: all three generation-1 children start from `parent_id=89fc7be1`, `score=1.0`, `action_category=baseline`, `generate_plan="baseline seed"`.
- Lineage already tried: only `89fc7be1` baseline; no prior optimization category beyond `baseline`.
- Seed evidence: archived v2 profile parity baseline, mixed bottleneck, mean `e2e_avg_us=695.1714`, mean `kernel_sum_us=328.9857`; local gate status blocked on this host.
- Dominant archived buckets: stage1 and stage2 dominate kernel sum at larger M, quant grows with M, and routing/overhead must preserve the v2 trace guard (`other == 0.0`, no profiler-visible memcpy).

## Search Log

| Source | Layer | Query | Finding | New? | Actionability |
|---|---|---|---|---|---|
| `kernel_opt_fused_moe_pes/README.md:3-104` and `profiles/0/seed/evidence.json:1-22` | L1 | Workspace target, seed evidence, and local gate status | Mixed bottleneck; stage1/stage2 MFMA throughput plus routing/quant/overhead are the relevant buckets; profiling must be on MI308X via `tools/profile_kernel.sh` / `rocprofv3`; local execution is blocked. | Yes | Constrains all children to planning only and one localized action per candidate. |
| `gpu-wiki/docs/ref-docs/amd/flydsl/gfx942/cdna3-fused-moe-fp8-ptpc-atrex-v2.md:118-229` | L1 | v2 full-pipeline baseline and parity requirements | Baseline rows expose routing, quant, stage1, stage2, overhead, `other`; parity requires no memcpy, `other == 0.0`, AITER routing/quant trace compatibility, M=1 stream boundary preservation. | Yes | Provides guardrails for every strategy and prevents isolated-stage-only changes. |
| `gpu-wiki/docs/ref-docs/amd/flydsl/gfx942/cdna3-fused-moe-fp8-ptpc-pause-checkpoint.md:153-226` | L1 | Remaining FP8 PTPC continuation map | Stage2 BF16 atomic/output cost is the primary unresolved bottleneck in the stage checkpoint; stage1 near-target rows should be prioritized separately; stage2 work must preserve BF16 output and avoid the full intermediate reduce path. | Yes | Supports one stage2 atomic/output strategy and one separate stage1 config strategy. |
| `gpu-wiki/docs/pitfalls/amd/flydsl/fused-moe-fp8-ptpc-pitfalls.md:49-142` | L1 | FP8 PTPC pitfalls | Do not count skip-atomic/no-output as success, do not use `block_m=8`, do not reintroduce hot-path host `valid_blocks`, do not re-sort stage2, and preserve M=1 stream marker. | Yes | Defines rollback and risk gates for all candidates. |
| `kernel_opt_fused_moe_pes/kernel.py:388-518`, `kernel.py:916-1229`, `moe_kernels.py:280-317`, `moe_kernels.py:391-430`, `kernels/moe_gemm_2stage.py:2953-3193`, `kernels/moe_gemm_2stage.py:4700-5045` | L1 | Parent implementation affordances | Parent v2 uses generic stage1 config for M=16..256, atomic stage2 for all rows, separate inter-stage quant, AITER opus sort reuse, and existing compile/config hooks for stage1/stage2 FP8 PTPC tile choices and stage2 CShuffle/global BF16 atomic epilogue. | Yes | Supplies concrete executor edit points without changing ownership here. |
| `reference-projects/README.md:1-24` and `rg` scan of `reference-projects/` | L2 | Local reference-project availability | Directory is only a submodule index in this checkout; no cloned local FlyDSL/AITER source is available beyond gpu-wiki and workspace code. | No | No actionable L2 implementation source found in this workspace. |

## Strategies

### Child 1_0

- parent_id: `89fc7be1`
- action_category: `stage2_atomic_epilogue`
- action_description: Probe a stage2 BF16 output/atomic epilogue change for M>=64 that reduces direct global BF16 atomic RMW pressure while preserving the required BF16 output. The executor should keep the integrated v2 routing topology and reuse stage1 sort metadata; do not use skip-atomic, no-output, racy non-atomic, `block_m=8`, or full `[tokens, topk, model_dim]` reduce as acceptance paths. If using the existing final-init scaffolding, it must be made profiler-safe or proven warm-cache safe because the current helper builds groups through CPU tensors.
- evidence_chain: archived v2 rows show stage2 is a large bucket for M>=64 -> pause checkpoint names stage2 BF16 atomic/output cost as the primary unresolved bottleneck -> gfx942 path uses global `llvm.AtomicRMWOp` for BF16 output -> change only the output/atomic epilogue category and gate on correctness plus trace (`other == 0.0`, no memcpy).
- expected_impact: Reduce stage2 output serialization and atomic overhead on larger token rows without changing FP8 GEMM input, BF16 output, routing topology, or stage1 behavior.
- risks: Extra grouping or launches can erase any atomic savings; CPU-built grouping can introduce profiler-visible DtoH/HtoD artifacts; direct-store/add ordering can break topk accumulation correctness; small rows can regress from launch overhead.
- rollback note: Restore v2 `_stage2_config(..., version="v2")` to `mode="atomic"` with the original per-token env map, and remove any added final-init or epilogue branch from the executor candidate.
- sources: `gpu-wiki/docs/ref-docs/amd/flydsl/gfx942/cdna3-fused-moe-fp8-ptpc-pause-checkpoint.md:204-226`; `gpu-wiki/docs/pitfalls/amd/flydsl/fused-moe-fp8-ptpc-pitfalls.md:49-77,109-117,136-142`; `kernel_opt_fused_moe_pes/kernel.py:478-518,765-913,1059-1227`; `kernel_opt_fused_moe_pes/kernels/moe_gemm_2stage.py:2982-2986,4985-5011`.

### Child 1_1

- parent_id: `89fc7be1`
- action_category: `stage1_tile_config`
- action_description: Probe token-specific stage1 FP8 PTPC tile/config selection for near-target rows M=16/64/128/256 using existing compile parameters only (`block_m`, `tile_n`, `tile_k`, `b_nt`, `waves_per_eu`, `xcd_swizzle` where already registered). Keep kernel builders unchanged and avoid `block_m=8`. Start from the parent v2 generic config as the control and compare one coherent token map against it on MI308X.
- evidence_chain: parent v2 uses one generic stage1 config for M=16..256 -> stage1 is a major archived bucket and the pause checkpoint says future stage1 work should prioritize M=16, M=64, M=128, then M=256 -> the local registry already supports FP8 PTPC stage1 tile choices across tile_m/tile_n/tile_k/b_nt/xcd -> a config-only child isolates tile-shape impact from epilogue or routing changes.
- expected_impact: Improve stage1 MFMA/data-movement balance on rows that are close enough to be plausible wins, without changing correctness semantics or adding new kernels beyond existing FlyDSL compile variants.
- risks: A token map can overfit one M and hurt mean e2e; larger tile_k/tile_n can increase register or LDS pressure; xcd swizzle can perturb occupancy or cache behavior; compile-cache naming must include tiling to avoid stale binaries.
- rollback note: Revert `_stage1_config(..., version="v2")` to the parent mapping: M=1 special case, M=512 hybrid16sort, and generic `block_m=16`, `tile_n=64`, `tile_k=128`, `b_nt=0 if token==16 else 2` for M=16..256.
- sources: `kernel_opt_fused_moe_pes/README.md:60-68,96-104`; `gpu-wiki/docs/ref-docs/amd/flydsl/gfx942/cdna3-fused-moe-fp8-ptpc-pause-checkpoint.md:153-170,220-226`; `gpu-wiki/docs/pitfalls/amd/flydsl/fused-moe-fp8-ptpc-pitfalls.md:59-67`; `kernel_opt_fused_moe_pes/kernel.py:388-430,962-1049`; `kernel_opt_fused_moe_pes/moe_kernels.py:280-317`.

### Child 1_2

- parent_id: `89fc7be1`
- action_category: `interstage_quant_fusion`
- action_description: Probe fusing the stage1 output activation/quantization path into the FlyDSL stage1 output for the inter-stage tensor, replacing the separate `aiter.dynamic_per_token_scaled_quant(a2_qt, gemm1_out, a2_scale)` launch only if the resulting FP8 tensor and scale layout can be consumed by stage2 without changing the full-pipeline trace contract. This child must not alter routing, stage2 atomic semantics, or stage1 tile config at the same time.
- evidence_chain: archived v2 rows show quant as a distinct bucket that grows at larger M -> parent wrapper materializes BF16 `gemm1_out` and then launches separate AITER dynamic per-token quant for `a2_qt/a2_scale` -> local `flydsl_moe_stage1` already has FP8 output/fused-quant code paths and returns an output scale buffer when `out_dtype="fp8"` -> isolate quant fusion as its own category.
- expected_impact: Remove or shrink the inter-stage quant bucket and reduce intermediate BF16 write/read traffic between stage1 and stage2, while keeping FP8 stage2 inputs and BF16 final output.
- risks: The fused scale buffer is sorted/tiled and may not match stage2's current `a2_scale` layout; numerical differences can fail the FP8 PTPC tolerance; changing quant placement can disturb AITER trace parity; a fallback copy or host conversion would violate the no-memcpy guard.
- rollback note: Restore the parent flow that allocates BF16 `gemm1_out`, calls `aiter.dynamic_per_token_scaled_quant` for `a2_qt/a2_scale`, and feeds those tensors to stage2 unchanged.
- sources: `gpu-wiki/docs/ref-docs/amd/flydsl/gfx942/cdna3-fused-moe-fp8-ptpc-atrex-v2.md:118-142,191-203`; `kernel_opt_fused_moe_pes/kernel.py:978-1057`; `kernel_opt_fused_moe_pes/moe_kernels.py:986-1004,1048-1079,1129-1157,1277-1283`; `kernel_opt_fused_moe_pes/README.md:51-55,96-104`.

## Evidence Summary

The available evidence is archived, not newly measured locally. It supports a mixed bottleneck model: stage1 and stage2 dominate kernel time for larger M, quant is a separate and growing full-pipeline bucket, and routing/overhead must preserve the atrex-open v2 trace guard. The most constrained bottleneck is stage2 BF16 output/atomic behavior; stage1 config tuning and inter-stage quant fusion are separate, attributable alternatives for the other major buckets.

## Exhaustion

`exhaustion = false`. Three new, diverse, single-category strategies were found from L1 gpu-wiki plus parent-code evidence. L2 did not add actionable local source in this checkout.
