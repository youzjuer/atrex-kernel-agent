# Generation 1 Summary

## Worked
- `1_0 stage1_intermediate_reuse`: PASS.
- Latency improved from `154.1547us` to `17.6533us`.
- Score: `8.7323x` vs v0 baseline.
- Root cause: v0 recomputed gate/up dot products for every output hidden column; `1_0` computes the intermediate once and reuses it.

## Failed
- No failed candidate admitted in this smoke generation.

## Next Directions
- Parallelize stage1 dot products across threads instead of one thread per intermediate element.
- Fuse temporary allocation or reuse a preallocated workspace.
- Add expert/token grouping once larger shapes are introduced.

## Stop
- `target_score=2.0` met by `f4392cab`; checkpoint stopped with `target_met`.
