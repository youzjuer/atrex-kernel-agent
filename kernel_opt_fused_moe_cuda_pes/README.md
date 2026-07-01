# FlashInfer-Aligned FP4 MoE CUDA PES Workspace

## Target

- API contract: `flashinfer.trtllm_fp4_block_scale_moe`
- performance baseline: the real `flashinfer.trtllm_fp4_block_scale_moe` operator. The local
  `reference.py` implementation is only a PyTorch correctness oracle and must not be used as the
  performance baseline.
- **performance target profile: tp=1** — `G11` (active P0) and `G8`, sourced from the proj_019
  workload catalog (see *Performance Shapes* below). The legacy `qwen_tp2` preset is kept only as a
  reference/interface profile, not the perf target.
- platform used here: NVIDIA L20D, CUDA, PyTorch extension
- candidate code: CUDA C++ extension, not FlyDSL

## Qwen3.5 Plus TP2 Contract

The checked-in metadata follows the public Qwen3.5-397B/Plus MoE configuration used for the
prefill TP2 target:

- `num_experts = 512`
- `top_k = 10`
- `hidden_size = 4096`
- `intermediate_size = 1024`
- TP2 local expert shard default: `local_num_experts = 256`
- default routing mode: `routing_method_type = 0` (`Softmax -> TopK`)

The default smoke tests keep the same expert/routing contract but reduce hidden/intermediate sizes
to make the scalar CUDA baseline practical:

- `smoke`: `T in [2, 4]`, `H=128`, `I=64`, `E_global=512`, `top_k=10`, `E_local=16`
- `qwen_micro`: `T in [2, 4]`, `H=256`, `I=128`, `E_global=512`, `top_k=10`, `E_local=16`
- `qwen_tp2`: full metadata (`H=4096`, `I=1024`, `E_local=256`) unless overridden, but the current
  scalar baseline is intended only for very small token counts.

## Performance Shapes (tp=1, from proj_019 catalog)

The performance test must use **tp=1** shapes, read from the single source of truth
`proj_019_moe_workload_op_opt/assets/workload_shape_catalog.py` via `workload_shapes.py` (never
hardcoded — task03 acceptance). tp=1 keeps the **full** `intermediate_size = 1024` and all **512
experts local** (`E_local = 512`), unlike the TP2 preset which shards per rank.

| preset | group | tokens | H | I | E_global | E_local | top_k | dtype | note |
|---|---|---|---|---|---|---|---|---|---|
| `g11` | G11 | 9500 | 4096 | 1024 | 512 | 512 | 10 | nvfp4 | active P0 |
| `g8` | G8 | 7680,8064,8320,8576 | 4096 | 1024 | 512 | 512 | 10 | mxfp4 | P2 backlog |

Point the loader at proj_019 (auto-discovered if it sits near this repo, else set the env var):

```bash
export PROJ019_ROOT=/path/to/proj_019_moe_workload_op_opt
python workload_shapes.py   # prints the resolved g11/g8 shapes
```

## Inputs

The candidate `kernel.py::run` matches the FlashInfer signature:

- `routing_logits`: `[T, num_experts]`, bf16/fp32
- `routing_bias`: optional `[num_experts]`
- true G11 FlashInfer target: `hidden_states` is packed NVFP4 `[T, H // 2]` uint8 and
  `hidden_states_scale` is `[T, H // 16]` fp8; the evaluator constructs this path with
  FlashInfer `fp4_quantize`
- local smoke oracle path: `hidden_states` is `[T, H]` bf16 in the current CUDA baseline
- `gemm1_weights`: `[E_local, 2 * I, H // 2]`, packed e2m1 FP4 in uint8
- true G11 FlashInfer target: weights/scales are passed through
  `prepare_static_weights_for_trtllm_fp4_moe` before the op call
- local smoke oracle path: `gemm1_weights_scale`: `[E_local, 2 * I, H // 32]`, fp8 block scale
- `gemm2_weights`: `[E_local, H, I // 2]`, packed e2m1 FP4 in uint8
- `gemm2_weights_scale`: `[E_local, H, I // 32]`, fp8 block scale
- `local_expert_offset`, `local_num_experts`, `routed_scaling_factor`, `routing_method_type`

The local PyTorch oracle in `reference.py` implements FlashInfer's documented semantics: unpack
e2m1 FP4, apply block scales, route from logits, use TRT-LLM SwiGLU convention `silu(X2) * X1`,
and scatter-add local expert outputs into `[T, H]` bf16. It exists for correctness checks only.

## Commands

```bash
cd kernel_opt_fused_moe_cuda_pes
python test_kernel.py --mode correctness --preset smoke --tokens 2,4
python test_kernel.py --mode profile --preset smoke --tokens 2
python test_kernel.py --mode correctness --preset qwen_micro --tokens 1
```

Performance test on the tp=1 target shapes (tokens default to the catalog buckets; omit `--tokens`).
The `--compare-flashinfer` flag benchmarks the real FlashInfer operator and reports
`speedup_vs_flashinfer`; `--require-flashinfer` fails the run instead of accepting a blocked target:

```bash
# G11 (active P0): tokens=9500, H=4096, I=1024, E=512, top_k=10, tp=1
python test_kernel.py --mode profile --preset g11 --compare-flashinfer --require-flashinfer
# G8 is mxfp4 and needs a separate real-input builder before it should be used as a target gate.
```

For the G11 catalog preset, correctness is checked against the real FlashInfer output under the
packed NVFP4 contract. The seed `kernel.py` consumes `hidden_states` `[T, H // 2]` plus
`hidden_states_scale` `[T, H // 16]`, reads FlashInfer prepared weights/scales, and mirrors the
FP4 intermediate quantize/dequantize step. The implementation is still scalar/two-stage and is not
expected to be competitive with FlashInfer until the GEMM stages are parallelized.

For a full metadata allocation smoke, override token count and local experts carefully:

```bash
python test_kernel.py --mode correctness --preset qwen_tp2 --tokens 1 --local-num-experts 16
```

Using the full TP2 local expert shard (`--local-num-experts 256`) allocates production-sized FP4
weights and is not appropriate for the current scalar baseline except as an allocation/interface
probe.

## Automatic PES Run

The runnable entry point mirrors the structure of
`syhya/mlsys26-flashinfer-contest/full-agent/moe/run_moe.sh`, but uses the local Atrex PES database
and evaluator instead of the external LoongFlow runtime:

```bash
cd kernel_opt_fused_moe_cuda_pes
./run_moe.sh --fresh --generations 1 --n-candidates 3 --preset smoke --tokens 2
```

This creates `moe_run/` and automatically:

- copies the FlashInfer-aligned task surface into the run directory
- initializes `database/` with `tools/evolution_db.py`
- profiles the seed kernel
- asks the planner backend to write `iteration/<K>/planner/plan.json` and `plan.md`
- asks the executor backend to materialize `n_candidates` CUDA workspaces
- evaluates each candidate with `test_kernel.py`
- records `evidence.json`, `history.md`, `summarizer/summary.md`, and database checkpoints

By default the runner uses `codex exec` as the planner and executor backend when the Codex CLI is on
`PATH`. This makes candidate generation agentic rather than template-based. For an external
full-agent/runtime, provide command hooks that receive the JSON context on stdin:

```bash
./run_moe.sh --planner-backend command --planner-cmd /path/to/planner \
  --executor-backend command --executor-cmd /path/to/executor \
  --generations 1 --n-candidates 3 --preset smoke --tokens 2
```

The planner must write or print JSON with a `strategies` list matching the agent contract:
`child`, `parent_id`, `action_category`, `action_description`, `evidence_chain`,
`expected_impact`, and `risks`. The executor receives one strategy plus a pre-populated candidate
directory and must edit only that directory. `--planner-backend local --executor-backend local` is
available only as a no-op orchestration smoke fallback.

## Current Baseline

- `kernel.py`: FlashInfer-compatible Python entrypoint and routing glue.
- `src/fused_moe_kernel.cpp`: C++ validation and PyTorch binding.
- `src/fused_moe_kernel.cu`: staged CUDA baseline for packed FP4 block-scale MoE:
  - `stage1_activation_kernel`: computes `silu(X2) * X1` once for each `(token, topk, I)`.
  - `stage2_output_kernel`: reuses the intermediate activation for all output columns.
- `reference.py`: FlashInfer-aligned PyTorch oracle and deterministic input generator.
- `test_kernel.py`: correctness/profile evaluator. Catalog packed-NVFP4 presets use real
  FlashInfer as the correctness oracle; performance target gating is checked only against real
  FlashInfer latency.

This seed implementation is still not a production grouped-GEMM implementation, but it removes the
largest scalar redundancy from the initial correctness seed and gives PES a better starting point
for expert grouping and tiled FP4 dequantization. It is not the performance baseline.
