---
name: gpu-kernel-evolve
description: |
  Full-agent PES optimization bridge. The supported path delegates to the local
  `mlsys26-flashinfer-contest/full-agent/*/run_*.sh` LoongFlow runners instead of
  reimplementing the full-agent loop inside atrex. Use this when evolutionary search is selected,
  and always use it when the user includes the keyword "pes".
---

# GPU Kernel Evolve (LoongFlow Bridge)

This skill owns Stage 2 when the user selects full-agent evolutionary search. The keyword `pes`
is a hard trigger for this skill. When the user says `pes`, `PES`, `full-agent PES`,
`Plan-Execute-Summary`, `pes优化`, or `完整pes流程`, the main agent must enter this workflow and
must not replace it with a hand-written single-trajectory loop.

The implementation style is the same as:

```text
$MLSYS26_FLASHINFER_CONTEST_ROOT/full-agent/moe/run_moe.sh
```

That script does four things:

1. Requires `LLM_API_KEY`.
2. Sets `PROJECT_ROOT` to the bundled LoongFlow checkout.
3. Renders `task_config.yaml` with `envsubst`.
4. Runs `agents/math_agent/math_evolve_agent.py` with:

```bash
--config <rendered task_config.yaml>
--task-file <task_prompt.txt>
--initial-file <task_definition.json>
--eval-file <eval_program_modal.py>
--log-level INFO
```

The LoongFlow runner then creates `PESAgent`, registers planner / executor / summary workers, and
owns the full loop: Plan -> Execute -> Summary, evaluator calls, evolution memory, checkpoints, and
target-score termination.

## Recommended Entry Point

For the MoE full-agent contest task, run:

```bash
export LLM_API_KEY=sk-...
bash orchestrator/pes.sh moe
```

For SOL-ExecBench kernel 58, run:

```bash
export LLM_API_KEY=sk-...
bash orchestrator/pes.sh sol58
```

Optional override when the contest checkout is not at the default path:

```bash
export MLSYS26_FLASHINFER_CONTEST_ROOT=/path/to/mlsys26-flashinfer-contest
bash orchestrator/pes.sh moe
```

The MoE bridge intentionally calls the local contest runner directly, so the authoritative config,
prompts, evaluator, and LoongFlow implementation stay under:

```text
$MLSYS26_FLASHINFER_CONTEST_ROOT/full-agent/moe/
```

The SOL58 bridge keeps its task files under `orchestrator/sol58_pes/` and runs the local LoongFlow
`agents/math_agent/math_evolve_agent.py` with `task_config.yaml`, `task_prompt.txt`,
`initial_kernel.cu`, and `eval_program_sol58.py`. The SOL evaluator writes each generated candidate
as `kernel.cu`, runs the official `sol-execbench` CLI over all 16 workloads, and scores all-pass
candidates as `0.006797 / geomean_latency_ms`.

## LoongFlow Configuration Shape

The MoE runner uses:

- `task_config.yaml`: LLM config, worker names, executor settings, `max_iterations`, `target_score`,
  `concurrency`, evaluator timeout, database config.
- `task_prompt.txt`: the MoE FP8 block-scale task prompt and output contract.
- `moe_fp8_block_scale_ds_routing_topk8_ng8_kg4_e32_h7168_i2048.json`: initial task definition.
- `eval_program_modal.py`: evaluator program used by the LoongFlow evaluator.

The key MoE defaults from the local runner are:

```yaml
evolve:
  planner_name: evolve_planner
  executor_name: evolve_executor_fuse
  summary_name: evolve_summary
  max_iterations: 40
  target_score: 100.0
  concurrency: 1
  initial_score: 0.0
  database:
    storage_type: in_memory
    num_islands: 1
    population_size: 100
    checkpoint_interval: 1
    sampling_weight_power: 2
```

The SOL58 runner uses the same shape with task-specific defaults:

```yaml
evolve:
  planner_name: evolve_planner
  executor_name: evolve_executor_fuse
  summary_name: evolve_summary
  max_iterations: 40
  target_score: 1.0
  concurrency: 1
  database:
    storage_type: in_memory
    num_islands: 1
    checkpoint_interval: 1
    sampling_weight_power: 2
```

## Relationship to Atrex

The linear atrex optimizer still uses `orchestrator/optimize.py` and
`gpu-kernel-profile-optimizer`. Full-agent mode does not use the atrex JSON-hook runner as its main
path. It delegates to LoongFlow so behavior matches the MLSys26 FlashInfer full-agent traces.

`orchestrator/evolve.py` and `tools/evolution_db.py` remain available as experimental / compatibility
tooling, but they are not the recommended full-agent MoE path.

## PES Execution Contract

A real PES run must satisfy all of the following:

1. Invoke a LoongFlow full-agent runner through `orchestrator/pes.sh` or a task-specific bridge
   called by that script.
2. Start `agents/math_agent/math_evolve_agent.py` with rendered YAML config, task prompt, initial
   task definition, and evaluator file.
3. Let LoongFlow create and run `PESAgent`.
4. Register and use planner, executor, evaluator, and summary workers.
5. Preserve population state, evolution memory, lineage, reflections, checkpoints, and target-score
   termination in the LoongFlow run directory.
6. Treat LoongFlow traces/checkpoints as the source of truth for accepted children and failed
   negative examples.

The following are not valid responses to a `pes` request:

- Direct manual edits to one candidate kernel without LoongFlow.
- Running `orchestrator/optimize.py` linear profile iterations.
- Running `orchestrator/evolve.py` unless the user explicitly asks for the experimental JSON-hook
  runner.
- Describing a PES plan without starting or preparing the actual full-agent runner.

If no supported bridge exists for the requested task, report that explicitly and stop instead of
silently downgrading.

## Constraints

- Do not modify the contest evaluator, task definition, or prompt unless the user explicitly asks.
- Keep `LLM_API_KEY` and optional base-url values in environment variables; do not commit secrets.
- If the local contest checkout path differs, set `MLSYS26_FLASHINFER_CONTEST_ROOT`.
- For SOL58, set `SOL58_CUDA_GENCODE` only when the target GPU architecture needs an explicit
  override; otherwise the evaluator derives it from PyTorch's current CUDA device.
- Treat LoongFlow checkpoints and traces as the source of truth for full-agent runs.
