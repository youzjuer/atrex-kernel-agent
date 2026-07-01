#!/usr/bin/env python3
"""Automatic PES runner for the FlashInfer-aligned FP4 MoE CUDA task.

The runner owns orchestration only: it selects parents from the local evolution
database, asks a planner backend for structured strategies, asks an executor
backend to materialize one child workspace per strategy, evaluates those
candidates with ``test_kernel.py``, and records the full
planner/executor/evaluator/summarizer trace.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class PlanStrategy:
    child: str
    parent_id: str | None
    action_category: str
    action_description: str
    evidence_chain: str
    expected_impact: str
    risks: str
    raw: dict[str, Any]


def run_cmd(cmd: list[str], *, cwd: Path, env: dict[str, str] | None = None) -> subprocess.CompletedProcess:
    print("+", " ".join(cmd), flush=True)
    return subprocess.run(cmd, cwd=cwd, env=env, text=True, check=True)


def run_capture(cmd: list[str], *, cwd: Path, env: dict[str, str] | None = None) -> str:
    print("+", " ".join(cmd), flush=True)
    proc = subprocess.run(cmd, cwd=cwd, env=env, text=True, check=True, stdout=subprocess.PIPE)
    if proc.stdout:
        print(proc.stdout, end="")
    return proc.stdout


def run_shell(command: str, payload: dict[str, Any], *, cwd: Path) -> str:
    print("+", command, flush=True)
    proc = subprocess.run(
        command,
        cwd=cwd,
        input=json.dumps(payload, indent=2),
        text=True,
        shell=True,
        check=True,
        stdout=subprocess.PIPE,
        stderr=None,
    )
    if proc.stdout:
        print(proc.stdout, end="")
    return proc.stdout


def copy_task(source: Path, run_dir: Path, *, fresh: bool) -> None:
    if fresh and run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    for rel in ("kernel.py", "reference.py", "test_kernel.py", "README.md", ".gitignore"):
        shutil.copy2(source / rel, run_dir / rel)
    src_dst = run_dir / "src"
    if src_dst.exists():
        shutil.rmtree(src_dst)
    shutil.copytree(source / "src", src_dst)
    (run_dir / "profiles").mkdir(exist_ok=True)
    (run_dir / "memory").mkdir(exist_ok=True)


def parse_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n")


def init_db(repo_root: Path, source: Path, run_dir: Path) -> None:
    db = run_dir / "database"
    if db.exists():
        return
    run_cmd(
        [
            sys.executable,
            str(repo_root / "tools" / "evolution_db.py"),
            "init",
            "--workspace",
            str(run_dir),
            "--config",
            str(source / "database" / "config.json"),
        ],
        cwd=repo_root,
    )


def configure_real_target_gate(repo_root: Path, run_dir: Path) -> None:
    """Prevent local-baseline speedup from satisfying the real FlashInfer target."""
    run_cmd(
        [
            sys.executable,
            str(repo_root / "tools" / "evolution_db.py"),
            "config",
            "--workspace",
            str(run_dir),
            "--set",
            "target_score=null",
            "--set",
            'score_metric="local_baseline_speedup_with_flashinfer_gate"',
        ],
        cwd=repo_root,
    )


def clear_stale_target_stop(run_dir: Path) -> None:
    state_path = run_dir / "database" / "state.json"
    if not state_path.exists():
        return
    state = parse_json(state_path)
    convergence = state.get("convergence") or {}
    if convergence.get("stop_reason") != "target_met":
        return
    convergence["stopped"] = False
    convergence["stop_reason"] = None
    state["convergence"] = convergence
    write_json(state_path, state)


def profile_kernel(run_dir: Path, kernel: Path, out_json: Path, args: argparse.Namespace) -> dict[str, Any]:
    cmd = [
        sys.executable,
        str(run_dir / "test_kernel.py"),
        "--kernel",
        str(kernel),
        "--mode",
        "profile",
        "--preset",
        args.preset,
        "--tokens",
        args.tokens,
        "--warmup",
        str(args.warmup),
        "--rep",
        str(args.rep),
        "--flashinfer-target-speedup",
        str(args.flashinfer_target_speedup),
        "--flashinfer-timeout-s",
        str(args.flashinfer_timeout_s),
        "--json-out",
        str(out_json),
    ]
    if args.compare_flashinfer:
        cmd += ["--compare-flashinfer"]
    if args.require_flashinfer:
        cmd += ["--require-flashinfer"]
    if args.hidden_size is not None:
        cmd += ["--hidden-size", str(args.hidden_size)]
    if args.intermediate_size is not None:
        cmd += ["--intermediate-size", str(args.intermediate_size)]
    if args.local_num_experts is not None:
        cmd += ["--local-num-experts", str(args.local_num_experts)]
    cmd += ["--local-expert-offset", str(args.local_expert_offset)]
    run_cmd(cmd, cwd=run_dir)
    return parse_json(out_json)


def first_latency(result: dict[str, Any]) -> float | None:
    lat = result.get("mean_latency_us")
    if lat is not None:
        return float(lat)
    rows = result.get("rows") or []
    vals = [float(row["candidate_us"]) for row in rows if row.get("candidate_us") is not None]
    return sum(vals) / len(vals) if vals else None


def max_rel(result: dict[str, Any]) -> float | None:
    val = result.get("max_rel")
    return None if val is None else float(val)


def flashinfer_latency(result: dict[str, Any]) -> float | None:
    flash = result.get("flashinfer") or {}
    lat = flash.get("mean_latency_us")
    return None if lat is None else float(lat)


def target_speedup(result: dict[str, Any]) -> float | None:
    val = result.get("target_speedup_vs_flashinfer")
    return None if val is None else float(val)


def ensure_seed(repo_root: Path, run_dir: Path, baseline_result: dict[str, Any]) -> str:
    state_path = run_dir / "database" / "state.json"
    state = parse_json(state_path)
    if state.get("baseline", {}).get("solution_id"):
        seed_id = state["baseline"]["solution_id"]
        snapshot_solution_src(run_dir, seed_id, run_dir / "src")
        return seed_id

    latency = first_latency(baseline_result)
    mem = {
        "version": "auto_seed",
        "performance": {"latency_us": latency},
        "correctness": {
            "status": baseline_result.get("status"),
            "max_rel": baseline_result.get("max_rel"),
        },
    }
    write_json(run_dir / "memory" / "auto_seed.json", mem)
    out = run_capture(
        [
            sys.executable,
            str(repo_root / "tools" / "evolution_db.py"),
            "import-seed",
            "--workspace",
            str(run_dir),
            "--from",
            "memory/auto_seed.json",
            "--code",
            "kernel.py",
            "--json",
        ],
        cwd=repo_root,
    )
    seed_id = json.loads(out)["solution_id"]
    snapshot_solution_src(run_dir, seed_id, run_dir / "src")
    return seed_id


def snapshot_solution_src(run_dir: Path, solution_id: str, src_dir: Path) -> None:
    """Keep CUDA companion sources with the DB kernel snapshot.

    ``evolution_db.py`` snapshots a single code file.  This task is a Python
    extension candidate whose implementation also lives under ``src/``, so the
    orchestrator mirrors that tree beside the DB snapshot for future parent
    materialization.
    """
    db_solution_dir = run_dir / "database" / "solutions" / solution_id
    if not db_solution_dir.exists() or not src_dir.exists():
        return
    dst = db_solution_dir / "src"
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src_dir, dst)


def next_generation(run_dir: Path) -> int:
    state = parse_json(run_dir / "database" / "state.json")
    history = state.get("history") or []
    if not history:
        return 1
    return max(int(row["generation"]) for row in history) + 1


def parent_record_from_solution(sol: dict[str, Any]) -> dict[str, Any]:
    metadata = sol.get("metadata") or {}
    optimization = metadata.get("optimization") or {}
    return {
        "solution_id": sol.get("solution_id"),
        "score": sol.get("score"),
        "island_id": sol.get("island_id"),
        "generation": sol.get("generation"),
        "code": sol.get("solution"),
        "generate_plan": sol.get("generate_plan"),
        "action_category": optimization.get("action_category", ""),
        "summary": sol.get("summary", ""),
    }


def selected_parents(repo_root: Path, run_dir: Path, n: int) -> list[dict[str, Any]]:
    out = run_capture(
        [
            sys.executable,
            str(repo_root / "tools" / "evolution_db.py"),
            "select-parents",
            "--workspace",
            str(run_dir),
            "--n",
            str(n),
            "--json",
        ],
        cwd=repo_root,
    )
    sampled = json.loads(out)
    state_path = run_dir / "database" / "state.json"
    if not state_path.exists():
        return sampled

    state = parse_json(state_path)
    solutions = state.get("solutions") or {}
    best_id = state.get("best_solution_id")
    selected: list[dict[str, Any]] = []
    seen: set[str] = set()

    if best_id and best_id in solutions:
        selected.append(parent_record_from_solution(solutions[best_id]))
        seen.add(best_id)
        sampled_ids = [p.get("solution_id") for p in sampled]
        if not sampled_ids or sampled_ids[0] != best_id:
            print(f"[Atrex] parent set anchored at best solution: {best_id}", flush=True)

    for parent in sampled:
        sid = parent.get("solution_id")
        if not sid or sid in seen:
            continue
        selected.append(parent)
        seen.add(sid)

    if len(selected) < n:
        ranked_ids = [sid for sid in (state.get("elites") or []) if sid in solutions]
        ranked_ids.extend(
            sid
            for sid in sorted(
                solutions,
                key=lambda x: float((solutions[x] or {}).get("score") or 0.0),
                reverse=True,
            )
            if sid not in ranked_ids
        )
        for sid in ranked_ids:
            if sid in seen:
                continue
            selected.append(parent_record_from_solution(solutions[sid]))
            seen.add(sid)
            if len(selected) >= n:
                break

    return selected[:n] if selected else sampled


def solution_record(run_dir: Path, solution_id: str | None) -> dict[str, Any] | None:
    if solution_id is None:
        return None
    state_path = run_dir / "database" / "state.json"
    if not state_path.exists():
        return None
    return (parse_json(state_path).get("solutions") or {}).get(solution_id)


def solution_for_iteration_ref(run_dir: Path, iteration_ref: str) -> str | None:
    state_path = run_dir / "database" / "state.json"
    if not state_path.exists():
        return None
    for sid, sol in (parse_json(state_path).get("solutions") or {}).items():
        if ((sol.get("metadata") or {}).get("iteration_ref") or "") == iteration_ref:
            return sid
    return None


def parent_workspace(run_dir: Path, parent_id: str | None) -> Path:
    if parent_id is None:
        return run_dir
    db_solution_dir = run_dir / "database" / "solutions" / parent_id
    if (db_solution_dir / "kernel.py").exists() and (db_solution_dir / "src").exists():
        return db_solution_dir

    sol = solution_record(run_dir, parent_id)
    iter_ref = ((sol or {}).get("metadata") or {}).get("iteration_ref")
    if iter_ref:
        candidate_dir = run_dir / iter_ref
        if (candidate_dir / "kernel.py").exists() and (candidate_dir / "src").exists():
            return candidate_dir
    return run_dir


def materialize_candidate_base(run_dir: Path, child_dir: Path, parent_id: str | None) -> Path:
    parent_dir = parent_workspace(run_dir, parent_id)
    child_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(parent_dir / "kernel.py", child_dir / "kernel.py")
    if (child_dir / "src").exists():
        shutil.rmtree(child_dir / "src")
    shutil.copytree(parent_dir / "src", child_dir / "src")
    return parent_dir


def normalize_strategy(
    raw: dict[str, Any],
    *,
    generation: int,
    idx: int,
    parent_id: str | None,
) -> PlanStrategy:
    child = f"{generation}_{idx}"
    action_category = str(raw.get("action_category") or "agent_generated")
    action_description = str(raw.get("action_description") or raw.get("description") or "")
    raw_parent = raw.get("parent_id")
    if raw_parent in (None, "", "null", "None", "..."):
        raw_parent = parent_id
    return PlanStrategy(
        child=child,
        parent_id=raw_parent,
        action_category=action_category,
        action_description=action_description,
        evidence_chain=str(raw.get("evidence_chain") or ""),
        expected_impact=str(raw.get("expected_impact") or ""),
        risks=str(raw.get("risks") or ""),
        raw=raw,
    )


def strategy_to_json(strategy: PlanStrategy) -> dict[str, Any]:
    data = dict(strategy.raw)
    data.update(
        {
            "child": strategy.child,
            "parent_id": strategy.parent_id,
            "action_category": strategy.action_category,
            "action_description": strategy.action_description,
            "evidence_chain": strategy.evidence_chain,
            "expected_impact": strategy.expected_impact,
            "risks": strategy.risks,
        }
    )
    return data


def write_plan(run_dir: Path, generation: int, plan: dict[str, Any]) -> None:
    plan_dir = run_dir / "iteration" / str(generation) / "planner"
    plan_dir.mkdir(parents=True, exist_ok=True)
    write_json(plan_dir / "plan.json", plan)
    if (plan_dir / "plan.md").exists():
        return
    lines = [
        f"# Planner Plan - Generation {generation}",
        "",
        "- operator: `flashinfer.trtllm_fp4_block_scale_moe`",
        "- profile: `Qwen3_5-Plus_prefill_TP2`",
        "",
        "## Strategies",
    ]
    for idx, strategy in enumerate(plan.get("strategies") or []):
        lines.append(
            f"{idx}. `{strategy.get('child')}` `{strategy.get('action_category')}` - "
            f"{strategy.get('action_description')}"
        )
    (plan_dir / "plan.md").write_text("\n".join(lines) + "\n")


def backend_choice(kind: str, requested: str, command: str | None) -> str:
    if requested != "auto":
        if requested == "command" and not command:
            raise RuntimeError(f"--{kind}-backend command requires --{kind}-cmd")
        if requested == "codex" and shutil.which("codex") is None:
            raise RuntimeError("codex backend requested but `codex` is not on PATH")
        return requested
    if command:
        return "command"
    if shutil.which("codex") is not None:
        return "codex"
    print(f"[Atrex] warning: no {kind} backend found; using local no-op backend", flush=True)
    return "local"


def codex_exec(repo_root: Path, prompt: str, args: argparse.Namespace) -> None:
    cmd = [
        "codex",
        "exec",
        "--cd",
        str(repo_root),
        "--dangerously-bypass-approvals-and-sandbox",
    ]
    if args.codex_model:
        cmd += ["--model", args.codex_model]
    cmd.append(prompt)
    run_cmd(cmd, cwd=repo_root)


def previous_summary_path(run_dir: Path, generation: int) -> str | None:
    if generation <= 1:
        return None
    path = run_dir / "iteration" / str(generation - 1) / "summarizer" / "summary.md"
    return str(path) if path.exists() else None


def planner_context(
    repo_root: Path,
    run_dir: Path,
    generation: int,
    n_candidates: int,
    parents: list[dict[str, Any]],
    baseline_json: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    enriched_parents: list[dict[str, Any]] = []
    for parent in parents:
        parent_id = parent.get("solution_id")
        sol = solution_record(run_dir, parent_id)
        workspace = parent_workspace(run_dir, parent_id)
        enriched = dict(parent)
        enriched["workspace"] = str(workspace)
        enriched["kernel_path"] = str(workspace / "kernel.py")
        enriched["src_path"] = str(workspace / "src")
        enriched["iteration_ref"] = ((sol or {}).get("metadata") or {}).get("iteration_ref")
        enriched["optimization"] = (((sol or {}).get("metadata") or {}).get("optimization") or {})
        enriched_parents.append(enriched)

    return {
        "workspace_path": str(run_dir),
        "repo_root": str(repo_root),
        "generation": generation,
        "n_candidates": n_candidates,
        "parents": enriched_parents,
        "previous_summary": previous_summary_path(run_dir, generation),
        "profiles": {
            "baseline": str(baseline_json),
            "directory": str(run_dir / "profiles"),
        },
        "operator_surface": "flashinfer.trtllm_fp4_block_scale_moe",
        "application_profile": "Qwen3_5-Plus_prefill_TP2",
        "optimization_focus": [
            "grouped expert scheduling",
            "tiled FP4 dequantization",
            "grouped GEMM",
        ],
        "framework": "CUDA C++ PyTorch extension",
        "constraints": [
            "Do not generate FlyDSL.",
            "Preserve the FlashInfer-compatible run(...) signature.",
            "Preserve Qwen3.5 Plus TP2 routing/top_k/local expert semantics.",
            "Each strategy must target exactly one optimization category.",
        ],
        "eval_args": {
            "preset": args.preset,
            "tokens": args.tokens,
            "hidden_size": args.hidden_size,
            "intermediate_size": args.intermediate_size,
            "local_num_experts": args.local_num_experts,
            "local_expert_offset": args.local_expert_offset,
            "warmup": args.warmup,
            "rep": args.rep,
        },
    }


def local_plan(generation: int, n_candidates: int, parents: list[dict[str, Any]]) -> dict[str, Any]:
    strategies: list[dict[str, Any]] = []
    for idx in range(n_candidates):
        parent = parents[idx % len(parents)] if parents else {}
        strategies.append(
            {
                "child": f"{generation}_{idx}",
                "parent_id": parent.get("solution_id"),
                "action_category": "control_parent_clone",
                "action_description": "Local fallback only: clone the selected parent without optimization.",
                "evidence_chain": "No planner backend was available.",
                "expected_impact": "Correctness/control signal only.",
                "risks": "No performance improvement expected.",
            }
        )
    return {
        "plan_path": f"iteration/{generation}/planner/plan.md",
        "strategies": strategies,
        "evidence_summary": "local fallback; no agent planning",
        "search_sources": [],
        "exhaustion": False,
    }


def read_plan(plan_path: Path) -> dict[str, Any]:
    if not plan_path.exists():
        raise RuntimeError(f"planner did not produce {plan_path}")
    plan = parse_json(plan_path)
    if not isinstance(plan.get("strategies"), list) or not plan["strategies"]:
        raise RuntimeError(f"planner output missing non-empty strategies: {plan_path}")
    return plan


def run_planner(
    repo_root: Path,
    run_dir: Path,
    generation: int,
    parents: list[dict[str, Any]],
    baseline_json: Path,
    args: argparse.Namespace,
) -> tuple[dict[str, Any], list[PlanStrategy]]:
    plan_dir = run_dir / "iteration" / str(generation) / "planner"
    plan_dir.mkdir(parents=True, exist_ok=True)
    plan_json = plan_dir / "plan.json"
    plan_md = plan_dir / "plan.md"
    if args.resume_existing and plan_json.exists():
        print(f"[Atrex] resume: using existing planner output {plan_json}", flush=True)
        plan = read_plan(plan_json)
        default_parent = parents[0]["solution_id"] if parents else None
        strategies = [
            normalize_strategy(raw, generation=generation, idx=idx, parent_id=default_parent)
            for idx, raw in enumerate(plan["strategies"][: args.n_candidates])
        ]
        if len(strategies) < args.n_candidates:
            raise RuntimeError(
                f"planner produced {len(strategies)} strategies, expected {args.n_candidates}"
            )
        plan["strategies"] = [strategy_to_json(strategy) for strategy in strategies]
        write_plan(run_dir, generation, plan)
        return plan, strategies
    if plan_json.exists():
        plan_json.unlink()
    if plan_md.exists():
        plan_md.unlink()
    context = planner_context(
        repo_root, run_dir, generation, args.n_candidates, parents, baseline_json, args
    )
    context["output_plan_json"] = str(plan_json)
    context["output_plan_md"] = str(plan_md)
    context_path = plan_dir / "planner_context.json"
    write_json(context_path, context)

    backend = backend_choice("planner", args.planner_backend, args.planner_cmd)
    print(f"[Atrex] planner backend: {backend}", flush=True)
    if backend == "command":
        stdout = run_shell(args.planner_cmd or "", context, cwd=repo_root)
        if not plan_json.exists() and stdout.strip():
            write_json(plan_json, json.loads(stdout))
    elif backend == "codex":
        prompt = f"""You are the gpu-kernel-planner agent for an automatic PES run.

Read and follow `{repo_root / "agents" / "gpu-kernel-planner.md"}`.
Use the JSON context at `{context_path}`. Inspect the referenced CUDA task files,
database lineage, profiles, local gpu-wiki/reference-projects as needed.

Produce exactly {args.n_candidates} strategies for the FlashInfer-aligned CUDA
FP4 MoE operator. The optimization direction is grouped expert scheduling,
tiled FP4 dequantization, and grouped GEMM. Do not edit implementation files.

Write machine-readable output to `{plan_json}` with this shape:
{{"plan_path": "...", "strategies": [{{"child": "{generation}_0", "parent_id": "...",
"action_category": "...", "action_description": "...", "evidence_chain": "...",
"expected_impact": "...", "risks": "..."}}], "evidence_summary": "...",
"search_sources": [...], "exhaustion": false}}

Also write a human-readable plan to `{plan_md}`. Return only after both files
exist."""
        codex_exec(repo_root, prompt, args)
    else:
        write_json(plan_json, local_plan(generation, args.n_candidates, parents))

    plan = read_plan(plan_json)
    if plan.get("exhaustion"):
        raise RuntimeError("planner reported exhaustion; no speculative candidates generated")

    default_parent = parents[0]["solution_id"] if parents else None
    strategies = [
        normalize_strategy(raw, generation=generation, idx=idx, parent_id=default_parent)
        for idx, raw in enumerate(plan["strategies"][: args.n_candidates])
    ]
    if len(strategies) < args.n_candidates:
        raise RuntimeError(
            f"planner produced {len(strategies)} strategies, expected {args.n_candidates}"
        )
    plan["strategies"] = [strategy_to_json(strategy) for strategy in strategies]
    write_plan(run_dir, generation, plan)
    return plan, strategies


def local_execute(child_dir: Path, strategy: PlanStrategy, parent_id: str | None) -> dict[str, Any]:
    history = child_dir / "history.md"
    history.write_text(
        f"# {strategy.child}\n\n"
        f"- parent: `{parent_id}`\n"
        f"- action_category: `{strategy.action_category}`\n"
        f"- action_description: {strategy.action_description}\n"
        "- backend: local fallback clone\n"
    )
    return {
        "child": strategy.child,
        "parent_id": parent_id,
        "candidate_code": str(child_dir / "kernel.py"),
        "action_category": strategy.action_category,
        "action_description": strategy.action_description,
        "self_check": "skipped",
        "history_path": str(history),
    }


def run_executor(
    repo_root: Path,
    run_dir: Path,
    generation: int,
    strategy: PlanStrategy,
    args: argparse.Namespace,
) -> dict[str, Any]:
    child_dir = run_dir / "iteration" / str(generation) / "executor" / strategy.child
    parent_id = strategy.parent_id
    result_path = child_dir / "executor_result.json"
    if (
        args.resume_existing
        and result_path.exists()
        and (child_dir / "kernel.py").exists()
        and (child_dir / "src").exists()
    ):
        print(f"[Atrex] resume: using existing executor output {result_path}", flush=True)
        return parse_json(result_path)

    parent_dir = materialize_candidate_base(run_dir, child_dir, parent_id)
    if result_path.exists():
        result_path.unlink()
    history_path = child_dir / "history.md"
    if history_path.exists():
        history_path.unlink()
    context = {
        "workspace_path": str(run_dir),
        "repo_root": str(repo_root),
        "generation": generation,
        "child": strategy.child,
        "strategy": strategy_to_json(strategy),
        "parent_id": parent_id,
        "parent_workspace": str(parent_dir),
        "candidate_dir": str(child_dir),
        "candidate_code": str(child_dir / "kernel.py"),
        "candidate_src": str(child_dir / "src"),
        "output_result_json": str(result_path),
        "operator_surface": "flashinfer.trtllm_fp4_block_scale_moe",
        "application_profile": "Qwen3_5-Plus_prefill_TP2",
        "constraints": [
            "Edit only files under candidate_dir.",
            "Do not generate FlyDSL.",
            "Preserve the FlashInfer-compatible run(...) signature.",
            "Apply exactly one optimization category.",
            "Do not run the evaluator; this runner evaluates after executor returns.",
        ],
    }
    context_path = child_dir / "executor_context.json"
    write_json(context_path, context)

    backend = backend_choice("executor", args.executor_backend, args.executor_cmd)
    print(f"[Atrex] executor backend for {strategy.child}: {backend}", flush=True)
    if backend == "command":
        stdout = run_shell(args.executor_cmd or "", context, cwd=repo_root)
        if not result_path.exists() and stdout.strip():
            write_json(result_path, json.loads(stdout))
    elif backend == "codex":
        prompt = f"""You are one gpu-kernel-executor child in an automatic PES run.

Read and follow `{repo_root / "agents" / "gpu-kernel-executor.md"}`.
Use the JSON context at `{context_path}`. The child workspace is already
materialized from its parent at `{child_dir}`.

Implement exactly the assigned strategy and edit only files under `{child_dir}`.
The target is CUDA C++/PyTorch extension code for
`flashinfer.trtllm_fp4_block_scale_moe`, used by Qwen3_5-Plus_prefill_TP2.
Do not generate FlyDSL and do not run benchmark/evaluator commands.

Write `{child_dir / "history.md"}` and machine-readable `{result_path}` with
fields: child, parent_id, candidate_code, action_category, action_description,
self_check, history_path. Return only after the candidate files exist."""
        codex_exec(repo_root, prompt, args)
    else:
        write_json(result_path, local_execute(child_dir, strategy, parent_id))

    if not (child_dir / "kernel.py").exists() or not (child_dir / "src").exists():
        raise RuntimeError(f"executor did not produce a complete candidate: {child_dir}")
    if not result_path.exists():
        write_json(result_path, local_execute(child_dir, strategy, parent_id))
    return parse_json(result_path)


def add_candidate(
    repo_root: Path,
    run_dir: Path,
    generation: int,
    child_name: str,
    parent_id: str | None,
    strategy: PlanStrategy,
    result: dict[str, Any],
    evidence_path: Path,
) -> str:
    latency = first_latency(result)
    correctness = "PASS" if result.get("status") == "PASS" else "FAIL"
    rel = max_rel(result)
    evaluation = f"{correctness}; latency_us={latency}; strategy={strategy.action_category}"
    cmd = [
        sys.executable,
        str(repo_root / "tools" / "evolution_db.py"),
        "add",
        "--workspace",
        str(run_dir),
        "--generation",
        str(generation),
        "--code",
        f"iteration/{generation}/executor/{child_name}/kernel.py",
        "--lang",
        "cuda",
        "--correctness",
        correctness,
        "--action-category",
        strategy.action_category,
        "--generate-plan",
        strategy.action_description,
        "--evaluation",
        evaluation,
        "--evidence-file",
        str(evidence_path),
        "--iteration-ref",
        f"iteration/{generation}/executor/{child_name}",
        "--json",
    ]
    if parent_id is not None:
        cmd += ["--parent", parent_id]
    if latency is not None:
        cmd += ["--latency-us", str(latency)]
    if rel is not None:
        cmd += ["--rel-err", str(rel)]
    out = run_capture(cmd, cwd=repo_root)
    solution_id = json.loads(out)["solution_id"]
    snapshot_solution_src(
        run_dir,
        solution_id,
        run_dir / "iteration" / str(generation) / "executor" / child_name / "src",
    )
    return solution_id


def write_summary(run_dir: Path, generation: int, rows: list[dict[str, Any]]) -> None:
    summary_dir = run_dir / "iteration" / str(generation) / "summarizer"
    summary_dir.mkdir(parents=True, exist_ok=True)
    ranked = sorted(rows, key=lambda r: (r["status"] != "PASS", r.get("latency_us") or float("inf")))
    lines = [f"# Auto PES Summary - Generation {generation}", ""]
    for row in ranked:
        lines.append(
            f"- `{row['child']}` `{row['strategy']}`: {row['status']}, "
            f"latency_us={row.get('latency_us')}, max_rel={row.get('max_rel')}, "
            f"flashinfer_us={row.get('flashinfer_us')}, target_status={row.get('target_status')}, "
            f"speedup_vs_flashinfer={row.get('target_speedup_vs_flashinfer')}"
        )
    if ranked:
        best = ranked[0]
        lines += [
            "",
            f"Best candidate: `{best['child']}` with strategy `{best['strategy']}`.",
        ]
        met = [row for row in ranked if row.get("target_status") == "MET"]
        if met:
            target_best = max(met, key=lambda r: r.get("target_speedup_vs_flashinfer") or 0.0)
            lines.append(
                f"Real FlashInfer target met by `{target_best['child']}` "
                f"at speedup {target_best.get('target_speedup_vs_flashinfer')}.")
        else:
            lines.append(
                "Real FlashInfer target not met; continue grouped scheduling, tiled FP4 dequant, and grouped GEMM."
            )
    (summary_dir / "summary.md").write_text("\n".join(lines) + "\n")


def checkpoint(repo_root: Path, run_dir: Path, generation: int, *, target_met: bool) -> None:
    cmd = [
        sys.executable,
        str(repo_root / "tools" / "evolution_db.py"),
        "checkpoint",
        "--workspace",
        str(run_dir),
        "--generation",
        str(generation),
    ]
    if target_met:
        cmd.append("--target-met")
    run_cmd(cmd, cwd=repo_root)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--fresh", action="store_true", help="Delete and recreate run dir first.")
    parser.add_argument("--generations", type=int, default=1)
    parser.add_argument("--n-candidates", type=int, default=3)
    parser.add_argument("--preset", choices=("smoke", "qwen_micro", "qwen_tp2"), default="smoke")
    parser.add_argument("--tokens", default="2")
    parser.add_argument("--hidden-size", type=int)
    parser.add_argument("--intermediate-size", type=int)
    parser.add_argument("--local-num-experts", type=int)
    parser.add_argument("--local-expert-offset", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--rep", type=int, default=5)
    parser.add_argument(
        "--compare-flashinfer",
        action=argparse.BooleanOptionalAction,
        default=os.environ.get("ATREX_COMPARE_FLASHINFER", "1") != "0",
        help="Benchmark real flashinfer.trtllm_fp4_block_scale_moe in an isolated worker.",
    )
    parser.add_argument(
        "--require-flashinfer",
        action="store_true",
        help="Fail the evaluator if the real FlashInfer benchmark is unavailable.",
    )
    parser.add_argument(
        "--flashinfer-target-speedup",
        type=float,
        default=float(os.environ.get("ATREX_FLASHINFER_TARGET_SPEEDUP", "1.0")),
        help="Real target is met only when candidate_us is faster than FlashInfer by this factor.",
    )
    parser.add_argument(
        "--flashinfer-timeout-s",
        type=float,
        default=float(os.environ.get("ATREX_FLASHINFER_TIMEOUT_S", "120")),
        help="Timeout for the isolated real FlashInfer benchmark worker.",
    )
    parser.add_argument(
        "--planner-backend",
        choices=("auto", "codex", "command", "local"),
        default=os.environ.get("ATREX_PLANNER_BACKEND", "auto"),
        help="Strategy planner backend. auto uses --planner-cmd, then codex, then local fallback.",
    )
    parser.add_argument(
        "--executor-backend",
        choices=("auto", "codex", "command", "local"),
        default=os.environ.get("ATREX_EXECUTOR_BACKEND", "auto"),
        help="Candidate executor backend. auto uses --executor-cmd, then codex, then local fallback.",
    )
    parser.add_argument(
        "--planner-cmd",
        default=os.environ.get("ATREX_PLANNER_CMD"),
        help="Shell command for planner backend. Receives JSON context on stdin.",
    )
    parser.add_argument(
        "--executor-cmd",
        default=os.environ.get("ATREX_EXECUTOR_CMD"),
        help="Shell command for executor backend. Receives JSON context on stdin.",
    )
    parser.add_argument(
        "--codex-model",
        default=os.environ.get("ATREX_CODEX_MODEL"),
        help="Optional model override for codex exec planner/executor backends.",
    )
    parser.add_argument(
        "--resume-existing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Reuse existing incomplete-generation planner/executor/evaluator artifacts.",
    )
    args = parser.parse_args()

    source = args.source.resolve()
    repo_root = args.repo_root.resolve()
    run_dir = args.run_dir.resolve()
    copy_task(source, run_dir, fresh=args.fresh)
    init_db(repo_root, source, run_dir)
    configure_real_target_gate(repo_root, run_dir)
    clear_stale_target_stop(run_dir)

    baseline_json = run_dir / "profiles" / "auto_baseline.json"
    baseline = profile_kernel(run_dir, run_dir / "kernel.py", baseline_json, args)
    seed_id = ensure_seed(repo_root, run_dir, baseline)
    print(f"[Atrex] seed solution: {seed_id}")

    for _ in range(args.generations):
        generation = next_generation(run_dir)
        parents = selected_parents(repo_root, run_dir, max(1, args.n_candidates))
        if not parents:
            parents = [{"solution_id": seed_id, "code": "kernel.py", "score": 1.0}]
        _, strategies = run_planner(repo_root, run_dir, generation, parents, baseline_json, args)
        rows: list[dict[str, Any]] = []
        for strategy in strategies:
            child = strategy.child
            child_dir = run_dir / "iteration" / str(generation) / "executor" / child
            executor_result = run_executor(repo_root, run_dir, generation, strategy, args)
            parent_id = strategy.parent_id
            result_path = child_dir / "result.json"
            if args.resume_existing and result_path.exists():
                print(f"[Atrex] resume: using existing evaluator output {result_path}", flush=True)
                result = parse_json(result_path)
            else:
                result = profile_kernel(run_dir, child_dir / "kernel.py", result_path, args)
            evidence = {
                "tool_used": "torch.cuda.Event",
                "strategy": strategy_to_json(strategy),
                "executor_result": executor_result,
                "status": result.get("status"),
                "latency_us": first_latency(result),
                "max_rel": result.get("max_rel"),
                "flashinfer_us": flashinfer_latency(result),
                "target_status": result.get("target_status"),
                "target_speedup_vs_flashinfer": target_speedup(result),
                "flashinfer": result.get("flashinfer"),
                "result_file": str(result_path.relative_to(run_dir)),
            }
            evidence_path = child_dir / "evidence.json"
            write_json(evidence_path, evidence)
            iteration_ref = f"iteration/{generation}/executor/{child}"
            if args.resume_existing and solution_for_iteration_ref(run_dir, iteration_ref):
                print(f"[Atrex] resume: DB already has {iteration_ref}", flush=True)
            else:
                add_candidate(
                    repo_root,
                    run_dir,
                    generation,
                    child,
                    parent_id,
                    strategy,
                    result,
                    evidence_path,
                )
            rows.append(
                {
                    "child": child,
                    "strategy": strategy.action_category,
                    "status": result.get("status"),
                    "latency_us": first_latency(result),
                    "max_rel": result.get("max_rel"),
                    "flashinfer_us": flashinfer_latency(result),
                    "target_status": result.get("target_status"),
                    "target_speedup_vs_flashinfer": target_speedup(result),
                }
            )
        write_summary(run_dir, generation, rows)
        checkpoint(
            repo_root,
            run_dir,
            generation,
            target_met=any(row.get("target_status") == "MET" for row in rows),
        )

    run_cmd(
        [
            sys.executable,
            str(repo_root / "tools" / "evolution_db.py"),
            "summary",
            "--workspace",
            str(run_dir),
        ],
        cwd=repo_root,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
