# Atrex Kernel Agent

AKA is an end-to-end Agent project for GPU kernel implementation, analysis, profiling, and iterative optimization. It helps an Agent turn PyTorch logic or an existing kernel into a high-performance GPU kernel through a structured, profile-driven workflow.

![Atrex architecture](atrex-architecture.png)

![Atrex optimization loop](atrex-optimization-loop.png)

## What It Does

- Creates an isolated optimization workspace under `kernel_opt_<name>/`.
- Looks up target hardware specs from the local `gpu-wiki` knowledge base.
- Runs Roofline analysis and sets auditable performance targets.
- Implements a correct baseline kernel before entering optimization.
- Runs the profile-driven optimization loop: profile with `ncu` or `rocprofv3`, extract bottleneck evidence, query `gpu-wiki` / reference projects / web sources for relevant optimization knowledge, write an evidence-based plan, apply one optimization category, validate correctness and performance, record memory, commit, then repeat until Stop Conditions are met.
- Optionally delegates full-agent **Plan-Execute-Summary** search to the local MLSys26 FlashInfer LoongFlow runner, matching `mlsys26-flashinfer-contest/full-agent/moe/run_moe.sh`. See [`skills/gpu-kernel-evolve/SKILL.md`](skills/gpu-kernel-evolve/SKILL.md).
- Records plans, profile artifacts, structured memory, reports, and Git commits for every accepted iteration.

For the full architecture and workflow design, see [`docs/design.md`](docs/design.md).

## Requirements

Installation requires:

- `bash`
- `git`
- `jq`
- A compatible coding runtime installed

Running optimization tasks also requires platform-specific profiling tools:

- NVIDIA: `ncu`, wrapped by `tools/profile_nvidia.sh`
- AMD: `rocprofv3`, wrapped by `tools/profile_kernel.sh`

## Installation

### 1. Internal Development Environment Setup (internal users only — optional)

Internal users should configure git `insteadOf` URL redirect rules so that submodules and
dependencies resolve against the internal network before running `git submodule update` below.
**External users can skip this step entirely.**

### 2. Pull reference-projects Submodule

```bash
git submodule update --init
```

Downloads all reference projects managed under `reference-projects/`.

### 3. Run the Installer

```bash
bash install.sh --prefix [install-path]
```

The install path is optional; defaults to `~/aka_kernel_opt`.

Common options:

```bash
bash install.sh ~/my_path          # Install to a custom directory
bash install.sh --hooks-only        # Install or update hooks only
bash install.sh --max-iterations N  # Configure hook stop behavior after memory/vN.json exceeds N
bash install.sh --uninstall         # Remove hooks installed by this script
```

The installer detects supported runtime home directories and prepares local hooks when available.

After installation, restart the coding runtime or open a new session so the hooks are loaded.

## Quick Start

Ask the Agent to optimize a kernel with at least:

- `platform`: target hardware platform, such as `H20` or `MI308X`.
- `framework`: target implementation framework, such as `CuteDSL` or `FlyDSL`.
- `kernel_demo`: path to the initial PyTorch logic or kernel implementation file.

Example:

```text
/gpu-kernel-optimizer Optimize /path/to/kernel_demo.py on MI308X with FlyDSL, dtype bf16, rel_err < 0.01.
```

The Agent will initialize a workspace, source hardware specs from `gpu-wiki`, write the workspace configuration, build a baseline, profile the kernel, and iterate until the configured Stop Conditions are met.

### Full-Agent PES Quick Start

When a request includes the keyword `pes`, Atrex treats it as a request for the real
Plan-Execute-Summary full-agent flow, not the linear optimizer and not a manual single-candidate
loop. Supported real PES tasks include MLSys26 FlashInfer MoE and SOL-ExecBench kernel 58:

```bash
export LLM_API_KEY=sk-...
export SOLBENCH_TOKEN=...  # required for sol58 official-v1.1 fitness
bash orchestrator/pes.sh moe
bash orchestrator/pes.sh sol58
bash orchestrator/pes.sh sol58 -- --code-language auto
```

`orchestrator/pes.sh` delegates to task-specific LoongFlow bridges. `moe` runs the local
`mlsys26-flashinfer-contest/full-agent/moe/run_moe.sh` runner. `sol58` runs
`orchestrator/run_sol58_pes.sh`, using the same LoongFlow `math_evolve_agent.py` with a
SOL-ExecBench evaluator. LoongFlow owns planner, executor, evaluator, summary, population memory,
lineage, reflections, checkpoints, and target-score termination.

SOL58 source generation supports three modes through `--code-language` or
`SOL58_CODE_LANGUAGE`:

- `cuda_cpp` (default): accept only a standalone CUDA C++ `kernel.cu`.
- `cute_dsl`: accept only a standalone CuTe DSL `kernel.py`.
- `auto`: let each child choose CUDA C++ or CuTe DSL while keeping one language per candidate.

For example, resume a CUDA checkpoint while allowing architectural migration to CuTe DSL with
`bash orchestrator/run_sol58_pes.sh --code-language auto --checkpoint-path <checkpoint>`. The
evaluator detects the emitted source, creates the matching SOL solution specification, and compares
both languages under the same three-repeat local gate and official v1.1 fitness.
Use `--cutedsl-rate 0.7` to reserve 70% of each 10-iteration schedule for mandatory CuTeDSL
generation (`--cutedsl-period` changes the period). CUDA output in a mandatory slot is rejected
before compilation, so the ratio is enforced instead of being prompt-only guidance.

SOL58 defaults to the `official_v1_1_b200` local measurement profile. It reserves physical GPU 0
(override with `--clock-gpu-index`), locks SM/DRAM clocks to the leaderboard preset
`1500/3996 MHz`, uses the official 10-warmup/50-iteration/seed-200 timing contract, and compiles
CUDA C++ to an `sm_100` compatibility cubin. The compatibility target is executable on this
machine's `sm_103` GPU and tracks the official `sm_100a` ordering more closely than native
`sm_103` code generation. Each repeat verifies actual clocks and re-locks after external drift.
Set `--measurement-profile native` for an unlocked host-speed dry run.

Local-best records include a measurement-profile fingerprint. Results from different GPU UUIDs,
clocks, code-generation targets, or evaluator stacks are never compared. On a profile change, the runner
remeasures the exact incumbent three times and carries its completed official score/submission into
the new record before PES resumes. A sibling `sol-execbench/.venv` checkout is discovered
automatically when installed. CuTe DSL still
JIT-compiles for the local `sm_103` device, so remote v1.1 remains the authoritative fitness for
architecture-sensitive changes.

For `sol58`, `SOL58_OFFICIAL_FITNESS=1` is the default. The evaluator first runs the local
SOL-ExecBench correctness and statistical local-best gates. A confirmed local improvement is sent
as a private official B200 v1.1 submission; a candidate inside the measurement uncertainty band may
also use one rate-limited challenger slot without replacing the persisted local best. Completed
official submissions return the authoritative `sol_score`. By default,
`SOL58_OFFICIAL_ASYNC_SUBMIT=1` returns
as soon as the upload is accepted, schedules a one-shot status refresh, and then waits until the
server's `result_available_at` before refreshing again on a later cache hit. If the official service remains pending,
the evaluator returns an incumbent-anchored provisional selection score and exposes the uncapped
`search_score` separately so PES preserves candidate ordering. Only a completed official result sets
`certified_score` and `target_certified`; provisional scores are never accepted as a leaderboard/rank
result. Set
`SOL58_OFFICIAL_FITNESS=0` only for local dry-runs where official fitness is not needed.

Remote submissions are also protected by a local-best gate. Every locally correct candidate is
measured three complete times (`SOL58_LOCAL_REPEAT_COUNT=3`), and the median geomean latency is
compared with the persisted local best. Non-slower candidates are remeasured in alternating
candidate/incumbent pairs and classified as confirmed faster, confirmed slower, or uncertain from a
configurable noise band. Confirmed faster candidates replace the local best and are uploaded;
uncertain candidates within the tolerance may be uploaded at most once per cooldown. Confirmed
slower candidates normally remain local, but changed architecture-sensitive implementations can use
a persisted periodic probe slot (default every 20 eligible unique sources, at most 6/day, 1-hour
cooldown, and no more than 10% local regression). This samples local/official ordering drift for TMA,
cluster, warp-specialized, persistent, and radix families without opening an unbounded upload path.
Exact source-hash duplicates reuse cached local and official results instead of rerunning CUDA.

The SOL58 runner enables Summary-stage Nsight Compute evidence by default. Profiling starts only after
the normal correctness and timing passes, targets the slowest measured workload and first matching
user-kernel launch, and is cached by source/profile/workload identity. Summary receives compact key
metrics, confidence-labelled bottleneck findings, and optimization implications in
`metrics.ncu_analysis`; `.ncu-rep` files and raw logs remain artifacts. NCU timeout, permission, parser,
or capture failures never alter fitness. Use `--no-ncu-summary` to disable it, `--ncu-policy local_best`
to profile only incumbent-quality candidates, `--ncu-policy periodic` with
`SOL58_NCU_PROFILE_INTERVAL`, or `--ncu-workload <index|uuid|slowest>` to override selection.

Generated CUDA and CuTe DSL candidates must be standalone single-file sources. The local evaluator
rejects quoted, absolute, dynamic, and source-file CUDA includes, plus relative or non-runtime Python
imports, before compilation/import so a candidate cannot pass locally by referring to files absent
from the official sandbox.

Atrex also deduplicates selectable SOL58 PES members by stripped source SHA-256 within each island.
Repeated source attempts remain in lineage/checkpoint history for diagnosis, but do not occupy
population, elite, or MAP-Elites slots and cannot compound sampling weight. Parent sampling repairs
older checkpoints before selection. Adaptive exploration requires five attempts and checks hard
stagnation (`<0.001`, 4x) before medium stagnation (`<0.01`, 2x), capped at `0.9`.

SOL58 PES defaults to eight architecture islands. At the Summary-to-database boundary, Atrex extracts
named CUDA/CuTe features (including warp specialization, cluster TMA broadcast, WGMMA, cp.async,
cooperative/persistent kernels, CUDA Graphs, histogram/atomic structure, and CuTe DSL), standardizes
the vectors, and applies NumPy SVD PCA plus deterministic clustering. Named routes have stable island
anchors while PCA separates general variants. Every 20 completed iterations, the PCA model is refit,
home members are reclassified, and each island exchanges its top fraction with adjacent islands.
`SOL58_NUM_ISLANDS` and `SOL58_ARCHITECTURE_MIGRATION_INTERVAL` override these defaults. Architecture
metadata and migration state are persisted in every checkpoint; legacy one-island checkpoints are
expanded and reindexed on load.

`orchestrator/sol58_pes/runtime_config.json` is the checked-in, schema-validated default for search,
paths, local measurement, official submission/probes, NCU, LLM, and compatibility settings. Existing
environment variables remain compatibility overrides and CLI flags have highest precedence. Use
`--runtime-config <file.json>` for a reviewed alternate configuration; unknown, missing, mistyped, or
out-of-range fields fail before LoongFlow starts.

Every SOL58 launch writes a redacted run record under
`<run-dir>/run_manifests/<UTC-time>-<pid>/` and updates `<run-dir>/run_manifest.json`. The record
contains the resolved `SOL58_*`, `ATREX_*`, and `LLM_*` settings with their
source precedence, CLI arguments, `resolved_runtime_config.json`, copied resolved YAML/task prompt,
source hashes, repository commits, and a search-configuration fingerprint. API keys, bearer tokens,
passwords, and credentials are redacted before anything is persisted.

The process-wide LoongFlow adapter is guarded by a pinned upstream contract in
`orchestrator/loongflow_compat/loongflow_contract.json`. The runner fails before GPU or LLM work when
the checkout commit, target-file SHA-256, patched callable signatures, or required runtime sentinels
do not match. `ATREX_LOONGFLOW_EXPECTED_COMMIT` deliberately selects another reviewed commit;
`ATREX_LOONGFLOW_ALLOW_UNPINNED=1` and `ATREX_LOONGFLOW_ALLOW_DIRTY=1` require a non-empty
`ATREX_LOONGFLOW_OVERRIDE_REASON`, are recorded in the run manifest, and are forbidden in CI.
Generic upstream integration remains in the explicit `compat_adapter.py`; kernel-58 NCU and stagnation policy
lives in `sol58_task_hooks.py`.

`tools/evolution_db.py` supports the standalone linear optimizer and its workspace memory. It is not
the authoritative database used by full-agent PES; LoongFlow owns that population, lineage, island,
and checkpoint state. The SOL58 evaluator delegates source contracts, durable state, local-best
storage, statistical gates, official protocol, calibration, and probes to focused modules under
`orchestrator/sol58_pes/`. CI runs unit tests, Ruff, scoped Pyright checks over explicit LoongFlow
memory/solution protocols, and ShellCheck without requiring a GPU.

## Main Files

```text
.
├── SKILL.md                         # Top-level gpu-kernel-optimizer router manifest
├── install.sh                       # Installer / uninstaller
├── docs/                            # Detailed project design docs
├── orchestrator/                    # sol-execbench orchestration and LoongFlow full-agent bridge
├── reference/                       # Workspace, plan, memory, and profiling templates
├── skills/                          # Baseline, optimizer, restart, and output-contract modules
├── tools/                           # Profiling, utilization, memory, and measurement tools
└── gpu-wiki/                        # Local GPU knowledge base
```

## License

Licensed under the [Apache License 2.0](LICENSE).
