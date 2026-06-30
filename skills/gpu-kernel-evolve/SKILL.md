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

## Workspace Layout

```text
kernel_opt_<name>/
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

### Step 3 — Execute (subagents, fan-out)

For each of the N strategies, launch an **executor** subagent. Each child:

- reads exactly one strategy + its parent's `kernel.py`,
- implements one candidate kernel at `iteration/<K>/executor/<child>/kernel.py`,
- records its own history under the same directory.

Children are independent and must not see each other. At `N=1` there is a single child.
**Launch the N executors in a single message so they run concurrently (M3 exercises true fan-out).**

### Step 4 — Evaluate (subagents) — the only promotion gate

For each child, launch an **evaluator** subagent. Each evaluator compiles the candidate, runs
`test_kernel.py` correctness (with timeout guard), measures latency via `do_bench`, optionally
profiles with `ncu` / `profile_kernel.sh`, and returns `(correctness, score, latency_us, evidence)`.
`score = baseline_latency / candidate_latency` when correctness is PASS, else `score = 0`. Failed
candidates are not discarded — they become negative evidence.

### Step 5 — Admit to the Database

Register every evaluated child (including failures, score 0):

```bash
python tools/evolution_db.py add --workspace kernel_opt_<name> --generation <K> \
  --parent <parent_id> --code iteration/<K>/executor/<child>/kernel.py --lang <framework> \
  --correctness <PASS|FAIL|TIMEOUT_FAIL> --latency-us <us> \
  --action-category <category> --generate-plan "<strategy>" \
  --evidence-file profiles/<K>/<child>/evidence.json \
  --iteration-ref iteration/<K>/executor/<child>/ --json
```

`add` computes the MAP-Elites cell, applies island/elite/migration/prune, and writes the
`parent_id` lineage. Omit `--score` to let the tool compute speedup from `--latency-us`, or pass
`--score` to use the evaluator's value directly.

### Step 6 — Summarize (subagent)

Launch the **summarizer** subagent over this generation's candidates (what worked, what failed, why)
and write the distilled feedback to `iteration/<K>/summarizer/`. It is the planner's context next
generation.

### Step 7 — Checkpoint, Sync, Commit

```bash
python tools/evolution_db.py checkpoint --workspace kernel_opt_<name> --generation <K> --json
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

On stop, print the final summary and the winning lineage:

```bash
python tools/evolution_db.py summary --workspace kernel_opt_<name>
python tools/evolution_db.py lineage --workspace kernel_opt_<name> --solution <best_id>
```

Then hand the winning `kernel.py` to `gpu-kernel-output-contract` if a clean evaluator candidate is
required.

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
