# SOL58 PES Knowledge Pack

This evidence is specific to `058_moe_expert_token_radix_sort_with_prefix_sum`.
Use it in Plan, Execute, and Summary. It is search guidance, not a substitute for
the 16-workload correctness gate or measured local/official fitness.

## Evidence boundaries

- The public leaderboard exposes participant names and scores, not source code.
  In particular, `Recursive` is a participant name. It is **not evidence** that
  the implementation uses recursion, recursive radix sort, or any other named
  algorithm.
- The benchmark title contains `radix_sort`, but the observable contract is a
  stable grouping of 8-bit expert ids plus 257 cumulative offsets. A 256-bin
  counting/histogram + prefix + stable scatter is also a one-digit radix-256
  implementation. Do not call it the wrong algorithm merely because it does not
  invoke a radix-sort library.
- Results cited below came from related Blackwell MoE data-preparation work on
  sm_120, not this exact B200 sm_100a benchmark. Treat them as priors that must be
  remeasured here.

## Algorithm families that must remain searchable

1. **Hierarchical stable bucket / one-pass radix-256**
   - Per-CTA histograms, a global or cooperative prefix, precomputed per-CTA
     expert bases, and stable local rank/scatter.
   - Variants include 256/512-token tiles, one or several CTAs per expert, CUDA
     Graph launch amortization, and cooperative-grid fusion.
2. **Bank-replicated hierarchical histogram**
   - Replicate 256-bin shared histograms by warp group, merge replicas, then use
     contention-free per-CTA expert offsets for scatter.
   - This attacks inter-warp shared-memory contention without paying for
     `match_any` on low-duplicate warps.
3. **Bitwise or small-digit stable radix**
   - One, two, four, or eight stable partition passes using 8-, 4-, 2-, or 1-bit
     digits. Count every extra full-array read/write, scan, launch, and temporary
     buffer before selecting this family.
4. **CUB DeviceRadixSort / BlockRadixSort**
   - A useful correctness-ready architecture probe and possible large-shape
     specialization. It is not presumed faster; library setup and pass overhead
     can dominate a sub-10-microsecond target.
5. **Expert-parallel stable scan**
   - CTAs own experts or expert groups and scan source positions in ascending
     order. It has simple stability but can reread the input excessively. Use it
     as an architectural seed, then reduce expert-by-token traffic through
     hierarchy, tiling, or shared bitmasks.

When the incumbent has not improved for the configured stagnation window,
Planner must choose a different family or a holistic rewrite. A plan that only
changes launch bounds, unrolling, comments, dead code, or one constant does not
satisfy a forced stagnation escape.

## Blackwell evidence worth transferring carefully

Local references:

- `gpu-wiki/docs/kernel-opt/nvidia/cutedsl/sm120/sm120-moe-data-prep.md`
- `gpu-wiki/docs/pitfalls/nvidia/cutedsl/sm120-moe-data-prep-pitfalls.md`

Observed in that related workload:

- Precomputed per-CTA per-expert base offsets removed globally contended scatter
  atomics and improved the measured path by 14.2%.
- Four-way bank-replicated shared histograms improved eligible warps/scheduler by
  52% and reduced warp cycles/instruction by 10.6%, although that isolated change
  was wall-time neutral because barriers remained.
- `match_any_sync` warp aggregation regressed 24.5% at 256 experts: expected
  same-expert duplicates inside one warp were too sparse to repay MIO and branch
  costs. Check the actual duplicate distribution before reusing it.
- A CUB radix experiment over 6144 keys regressed 41.5% and increased issued
  instructions by 5.8x. This does not ban CUB for SOL58, but requires shape-
  specialized evidence and explicit accounting for passes, temporary storage,
  and launch cost.
- Letting most warps exit after a histogram phase reduced barrier instructions
  but collapsed achieved occupancy from about 64% to 21% and regressed 25.1%.
  Warp specialization must preserve enough active warps to hide memory latency.
- When several NCU stall categories are simultaneously large, fixing one counter
  can expose another. Summary must reconcile counter movement with end-to-end
  latency rather than treating NCU estimated speedup as additive.

## SOL58-specific decision rules

- Stability is by original flattened assignment index inside each expert. Any
  parallel scatter must prove that rank order, not only histogram equality.
- The workload range is roughly 8K-65K assignments. Specialize by assignment
  count only when all 16 shapes retain a correct fallback.
- This is integer data movement and scan, not matrix multiplication. TMA,
  cluster multicast, WGMMA, and TMEM are useful only when a concrete transfer or
  synchronization cost they remove exceeds setup and launch overhead.
- A new family should first target correctness and structural viability. Once it
  passes, use local paired measurements and official v1.1 fitness to decide
  whether it deserves refinement.
- Negative results are family- and shape-specific. Record the exact source
  structure, workload, instruction/traffic cost, and measured regression so a
  later plan does not repeat the same experiment under a new name.
