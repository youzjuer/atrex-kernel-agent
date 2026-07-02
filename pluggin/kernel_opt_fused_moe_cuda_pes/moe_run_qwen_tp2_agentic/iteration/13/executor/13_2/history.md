# Executor 13_2 History

Parent id: `7b67f03d`

Strategy applied: `single_token_fixed_topk_no_compact`

Concrete edit summary:
- Removed the single-token compact metadata kernel from the fast path.
- Stage2 now indexes the original top-k slots directly and writes zero partials for non-local experts.
- Finalize scans fixed `TOPK` slot partials instead of compacted local slot metadata.

Evaluation:
- Correctness: `PASS`, `max_rel=0.00023841856454964727`.
- Profile: `latency_us=140.6400054693222`.
- FlashInfer comparison remained blocked; this run reported a package version mismatch.
