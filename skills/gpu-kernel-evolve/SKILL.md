---
name: gpu-kernel-evolve
description: |
  Evolutionary (Plan-Execute-Summary) GPU kernel optimization skill. Replaces the single-trajectory
  Stage 2 loop with a population-based search: each generation selects parents from an island-model
  MAP-Elites database, fans out N candidate kernels, evaluates them on the local GPU as the sole
  promotion gate, admits them by score/diversity, distills a summary, checkpoints, and repeats until
  Stop Conditions or the iteration budget are met. At N=1 it degenerates to the classic
  single-trajectory profile-driven loop. State lives in the evolution database, not linear memory.
---

# GPU Kernel Evolve (PES Loop)

This skill owns Stage 2 of the optimizer when the evolutionary loop is enabled. It is the orchestrator
only: it prepares inputs, launches the four sub-agents (planner / executor / evaluator / summarizer),
and drives the evolution database (`tools/evolution_db.py`). It does not search knowledge, implement
kernels, run benchmarks, or judge correctness directly — those belong to the sub-agents.

All global constraints from the top-level `SKILL.md` still apply unchanged: hardware specs come from
`<gpu-wiki>/`, optimization decisions require official profiler evidence (`ncu` /
`profile_kernel.sh`), correctness must pass before any performance conclusion, and every accepted
generation is committed with git.

## When to Use

- Stage 2 of `gpu-kernel-optimizer` when evolutionary search is selected.
- A baseline (`memory/v0.json` + `kernel.py`) already exists and you want population-based search
  instead of the linear `gpu-kernel-profile-optimizer` loop.

## Relationship to the Linear Optimizer

`gpu-kernel-profile-optimizer` (single trajectory) and this skill solve the same problem with a
different search shape. This skill is a strict generalization:

| | Linear optimizer | This skill (evolve) |
|---|---|---|
| Candidates / generation | 1 | N (`config.n_candidates`, default 3) |
| State | `memory/v<N>.json` linear | `database/` (islands + MAP-Elites + elites + lineage) |
| Selection | git commit / revert | Boltzmann parent selection from the population |
| **N = 1 behavior** | — | **degenerates to the linear loop (no regression)** |

**M2 acceptance:** run with `n_candidates = 1` and confirm one generation = one
profile→plan→implement→evaluate→commit step, equivalent to the linear optimizer. N > 1 (true fan-out)
is exercised in M3.

## Building Blocks

- **Database / state hub:** `tools/evolution_db.py` (see `docs/evolution-db-design.md`).
- **Sub-agents:**
  - Planner — [gpu-kernel-planner](../../agents/gpu-kernel-planner.md)
  - Executor — [gpu-kernel-executor](../../agents/gpu-kernel-executor.md)
  - Evaluator — [gpu-kernel-evaluator](../../agents/gpu-kernel-evaluator.md)
  - Summarizer — [gpu-kernel-summarizer](../../agents/gpu-kernel-summarizer.md)
- **Helper skill:** [gpu-kernel-bottleneck-analysis](../gpu-kernel-bottleneck-analysis/SKILL.md) for
  Roofline / evidence extraction.
- **Knowledge sources:** `<gpu-wiki>/`, `reference-projects/`, public web (planner only, by priority).

## Skill Entry Points

The skill-owned MoE launcher is:

```bash
./pluggin/kernel_opt_fused_moe_cuda_pes/run_moe.sh
```

This is a convenience entry point for the current FlashInfer-aligned MoE PES workspace. The run
parameters live in the script's top configuration block, including GPU id, `PROJ019_ROOT`, preset,
token count, warmup/rep, candidate count, backend selection, and FlashInfer baseline gates. It
resolves the repository root from the plugin workspace path and runs the PES orchestrator directly.

## Workspace Layout

```text
pluggin/kernel_opt_<name>/
  kernel.py                      # current best (synced from the DB best after each checkpoint)
  test_kernel.py
  README.md                      # static config, specs, Roofline, Stop Conditions, Evolve Config
  memory/v0.json                 # baseline (imported as the evolution seed)
  database/                      # evolution DB — single source of truth (evolution_db.py)
    config.json  state.json  solutions/<id>/  checkpoints/iter-<K>/
  iteration/<K>/
    planner/                     # plan prompt + response per generation
    executor/<child>/            # each child's candidate kernel + history
    summarizer/                  # distilled feedback for the next generation
  profiles/<K>/<child>/          # per-candidate profiler artifacts
```

## Stage 0: One-Time Initialization

Run once, after the baseline (`memory/v0.json` + `kernel.py`) exists and before generation 1.

1. Initialize the database (writes `database/config.json` with upstream defaults):

   ```bash
   python tools/evolution_db.py init --workspace kernel_opt_<name>
   ```

2. (Optional) Override evolve parameters recorded under `README.md` → `Evolve Config`:

   ```bash
   python tools/evolution_db.py config --workspace kernel_opt_<name> --set n_candidates=1
   ```

3. Import the baseline as the seed (generation 0, `parent_id=null`, score 1.0; records baseline
   latency for speedup normalization):

   ```bash
   python tools/evolution_db.py import-seed --workspace kernel_opt_<name> \
     --from memory/v0.json --code kernel.py
   ```

The database now holds one seed solution and an `iter-0` checkpoint. `README.md`, the seed, and the
DB are the source of truth from here on.

## Stage Loop: One Generation (K = 1, 2, …)

Repeat the following per generation until Stop Conditions or the budget are met. Let
`N = config.n_candidates`.

### Step 1 — Select Parents

```bash
python tools/evolution_db.py select-parents --workspace kernel_opt_<name> --n <N> --json
```

This returns `N` parent records (id, score, code path, generate_plan, action_category, summary) via
diversity-adaptive Boltzmann selection. At `N=1` this is the current best/seed.

### Step 2 — Plan (subagent)

Launch the **planner** subagent once, passing the parents from Step 1, the latest summary
(`iteration/<K-1>/summarizer/`), current profile evidence, and knowledge-source paths. The planner
returns **N strategies** (one per child), each a concrete single-category optimization plan derived
from new, evidence-backed knowledge. The main agent must not search or plan directly.

### Step 3 — Execute (fan-out, concurrent)

Spawn one **executor** subagent per strategy. Name children `<K>_0 … <K>_{N-1}` →
`iteration/<K>/executor/<K>_<i>/`. Each child reads exactly one strategy + its parent's `kernel.py`,
implements one candidate at `iteration/<K>/executor/<K>_<i>/kernel.py`, and records its own history.

- **True fan-out:** launch all N executors **in a single message** so they run concurrently. Do not
  serialize them; do not let one child read another's files (isolation preserves diversity).
- At `N=1` there is one child (degenerate = linear loop).
- A child that errors out and produces **no** candidate file is logged and dropped from this
  generation (no DB entry). A child that produces a candidate always proceeds to evaluation — even if
  its self-check failed — so the evaluator can record it as negative evidence.

### Step 4 — Evaluate (batch, concurrent) — the only promotion gate

Spawn one **evaluator** subagent per produced candidate, **launched together** so the batch runs
concurrently. Each evaluator runs `test_kernel.py` correctness (timeout guard), measures latency via
`do_bench`, and returns `(correctness, score, latency_us, evidence)`;
`score = baseline_latency / candidate_latency` when PASS, else `score = 0`.

Two-tier profiling to bound cost under fan-out:

- **Screen (all N):** correctness + `do_bench` latency for every candidate — cheap, decides the gate.
- **Deep (survivors only):** full `ncu` / `profile_kernel.sh` evidence only for PASS candidates that
  beat or approach the current best; these feed the next planner. Do not run full `ncu` on every
  child of every generation.

Collect all N results before admitting (a batch barrier) so the generation is admitted atomically.

### Step 5 — Admit the Batch to the Database

Loop `add` over **every evaluated candidate this generation**, PASS and FAIL alike (failures carry
`score = 0` as negative evidence — never silently drop them):

```bash
for child in <K>_0 <K>_1 … <K>_{N-1}; do
  python tools/evolution_db.py add --workspace kernel_opt_<name> --generation <K> \
    --parent <parent_id_of_child> --code iteration/<K>/executor/<child>/kernel.py --lang <framework> \
    --correctness <PASS|FAIL|TIMEOUT_FAIL> --latency-us <us> \
    --action-category <category> --generate-plan "<strategy>" \
    --evidence-file profiles/<K>/<child>/evidence.json \
    --iteration-ref iteration/<K>/executor/<child>/ --json
done
```

Each `add` computes the candidate's MAP-Elites cell and applies island placement / cell replacement /
elite archive / migration / pruning, and records the `parent_id` lineage (siblings of one generation
branch from their respective parents). Omit `--score` to let the tool compute speedup from
`--latency-us`, or pass `--score` to use the evaluator's value directly.

### Generation Failure Handling

- **Partial failure** (some children fail): expected and fine — failures are admitted as negative
  evidence, the surviving candidates compete normally.
- **Total failure** (all N fail correctness this generation): still summarize and checkpoint so the
  negative evidence is preserved; the best does not change. If total failure (or no score
  improvement) persists for `convergence.no_improve_patience` generations, the `checkpoint` gate
  reports `stopped = true, stop_reason = no_improve`. Before giving up, the orchestrator MAY escalate
  to [gpu-kernel-partial-restart](../../agents/gpu-kernel-partial-restart.md) to reset a stalled
  island / mask stale memory and reseed from the current best.
- **Planner exhaustion** (planner returns `exhaustion = true`): no new actionable knowledge — escalate
  to partial-restart rather than emitting a speculative plan.

### Step 6 — Summarize (subagent) — closes the feedback loop

Launch the **summarizer** subagent over this generation's candidates (what worked, what failed, why)
and write the distilled feedback to `iteration/<K>/summarizer/summary.md`. This is the PES feedback
loop: in Step 2 of generation `K+1` the planner reads `iteration/<K>/summarizer/` as `prev_summary`,
so each generation's lessons (failed categories not to repeat, promising directions, crowded vs empty
MAP-Elites regions) directly shape the next plan. The loop is `plan → execute → evaluate → summarize
→ plan`.

### Step 7 — Checkpoint, Sync, Commit

```bash
# pass --target-met when README Stop Conditions are reached this generation
# (or set config target_score once, and checkpoint detects it automatically)
python tools/evolution_db.py checkpoint --workspace kernel_opt_<name> --generation <K> [--target-met] --json
```

This writes `checkpoints/iter-<K>/{best_solution.json, metadata.json, solutions/}` and reports
`best`, `stopped`, and `stop_reason`. Then:

1. Sync the global best into the workspace so packaging/inspection always see the latest:

   ```bash
   BEST=$(python tools/evolution_db.py best --workspace kernel_opt_<name> --json)
   # copy the best solution's code over kernel.py (path is in the best record)
   ```

2. Commit the generation:

   ```bash
   git add kernel.py database/ iteration/<K>/ profiles/<K>/
   git commit -m "gen <K>: best <solution_id> score <x> (<n> candidates, <admitted> admitted)"
   ```

### Step 8 — Stop or Continue

Read the `checkpoint` result:

- `stopped = true` → stop. `stop_reason` is one of `target_met` (Stop Conditions reached),
  `no_improve` (no best improvement for `convergence.no_improve_patience` generations), or
  `budget_exhausted` (`budget.max_generations` reached).
- otherwise → next generation (K+1), back to Step 1.

## Finalization (on stop)

When the loop stops, produce the deliverables:

1. Print the final summary and the winning lineage (audit trail back to the seed):

   ```bash
   python tools/evolution_db.py summary --workspace kernel_opt_<name>
   BEST=$(python tools/evolution_db.py best --workspace kernel_opt_<name> --json)
   python tools/evolution_db.py lineage --workspace kernel_opt_<name> --solution <best_id>
   ```

2. Confirm the workspace `kernel.py` is the global best (synced in Step 7).

3. If a hidden evaluator needs a clean candidate, hand the winning `kernel.py` to
   [gpu-kernel-output-contract](../gpu-kernel-output-contract/SKILL.md) to package
   `generated_kernel.py` (valid runtime code only — no tests, benchmarks, or debug output).

4. Report the result: best `solution_id`, score (speedup vs baseline), `stop_reason`, generations
   run, and the winning lineage. If `stop_reason` is `no_improve` and Stop Conditions were not met,
   note that the run converged below target (consider partial-restart or a larger budget / N).

## Islands, Migration & Lineage

The database keeps an **island-model population** so the search explores several lineages in parallel
instead of collapsing onto one. All of this is handled inside `tools/evolution_db.py` (defaults in
`database/config.json`); the orchestrator only needs to drive it and inspect it.

- **Islands** (`num_islands`, default 3): new candidates are spread across islands round-robin
  (`population_size // num_islands` per island before advancing), unless `add --island <id>` pins one.
  Each island runs its own MAP-Elites grid over `[complexity, diversity, score]`.
- **Migration** (`migration_interval` 10 generations, `migration_rate` 0.2): periodically the top
  fraction of each island is copied to the next island (ring), so good ideas cross-pollinate without
  erasing local diversity. Triggered automatically inside `add` on the generation boundary.
- **Selection** (`select-parents`): diversity-adaptive-temperature Boltzmann. Temperature rises when
  the population is diverse (explore) and falls when it converges (exploit), bounded
  `[min_temperature, max_temperature]`; an `exploration_rate` reserves some purely random picks. Higher
  `score` ⇒ higher selection probability.
- **Lineage**: every candidate stores `parent_id`; siblings of one generation branch from their
  parents, and any winner is fully traceable back to the seed.

Inspect during or after a run:

```bash
python tools/evolution_db.py best    --workspace kernel_opt_<name> --island <id>   # per-island best
python tools/evolution_db.py list    --workspace kernel_opt_<name> --island <id>   # island members
python tools/evolution_db.py lineage --workspace kernel_opt_<name> --solution <id> # parent chain to seed
python tools/evolution_db.py summary --workspace kernel_opt_<name>                 # islands, elites, history
```

## Constraints

- The main agent orchestrates only: it MUST NOT search knowledge, implement kernels, run benchmarks,
  or judge correctness directly — those are the planner / executor / evaluator jobs.
- Every code change MUST trace to profiler evidence; every hardware spec MUST come from `<gpu-wiki>/`.
- The evaluator is the ONLY promotion gate. Do not admit a candidate without its evaluator result.
- Failed candidates MUST still be admitted (score 0) as negative evidence — do not silently drop them.
- The database (`database/`) is the single source of truth. `memory/v<N>.json` is a compatibility
  mirror only; do not treat it as authoritative.
- Do not exit the loop while `stopped = false` unless the user explicitly stops the workflow.
- At `N=1`, behavior MUST match the linear optimizer — this is the M2 no-regression check.
