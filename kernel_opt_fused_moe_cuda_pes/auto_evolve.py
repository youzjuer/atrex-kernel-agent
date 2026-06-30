#!/usr/bin/env python3
"""Local automatic PES runner for the FlashInfer-aligned FP4 MoE task.

This is the runnable counterpart to the LoongFlow-style ``run_moe.sh`` entry
point.  It does not call an LLM by itself; instead it drives the local Atrex
evolution database, creates candidate workspaces from deterministic strategy
templates, evaluates them with ``test_kernel.py``, and records the full
planner/executor/evaluator/summarizer trace.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Strategy:
    name: str
    description: str
    replacements: tuple[tuple[str, str], ...]


STRATEGIES = (
    Strategy(
        name="keep_stage1_reuse",
        description="Control candidate: keep the staged activation reuse kernel unchanged.",
        replacements=(),
    ),
    Strategy(
        name="wide_stage1_tiles",
        description="Increase stage1 x tile width to expose more intermediate columns per CTA.",
        replacements=(("constexpr dim3 block_stage1(16, 4);", "constexpr dim3 block_stage1(32, 2);"),),
    ),
    Strategy(
        name="wide_stage2_tiles",
        description="Increase stage2 hidden-column tile width to improve output write coalescing.",
        replacements=(("constexpr dim3 block_stage2(16, 8);", "constexpr dim3 block_stage2(32, 4);"),),
    ),
    Strategy(
        name="tall_stage1_tiles",
        description="Increase stage1 token/top-k tile height to improve occupancy on small I.",
        replacements=(("constexpr dim3 block_stage1(16, 4);", "constexpr dim3 block_stage1(16, 8);"),),
    ),
    Strategy(
        name="narrow_stage2_tiles",
        description="Use narrower stage2 hidden tiles with more token rows per CTA.",
        replacements=(("constexpr dim3 block_stage2(16, 8);", "constexpr dim3 block_stage2(8, 16);"),),
    ),
)


def run_cmd(cmd: list[str], *, cwd: Path, env: dict[str, str] | None = None) -> subprocess.CompletedProcess:
    print("+", " ".join(cmd), flush=True)
    return subprocess.run(cmd, cwd=cwd, env=env, text=True, check=True)


def run_capture(cmd: list[str], *, cwd: Path, env: dict[str, str] | None = None) -> str:
    print("+", " ".join(cmd), flush=True)
    proc = subprocess.run(cmd, cwd=cwd, env=env, text=True, check=True, stdout=subprocess.PIPE)
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


def write_json(path: Path, data: dict[str, Any]) -> None:
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
        "--json-out",
        str(out_json),
    ]
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


def ensure_seed(repo_root: Path, run_dir: Path, baseline_result: dict[str, Any]) -> str:
    state_path = run_dir / "database" / "state.json"
    state = parse_json(state_path)
    if state.get("baseline", {}).get("solution_id"):
        return state["baseline"]["solution_id"]

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
    return json.loads(out)["solution_id"]


def next_generation(run_dir: Path) -> int:
    state = parse_json(run_dir / "database" / "state.json")
    history = state.get("history") or []
    if not history:
        return 1
    return max(int(row["generation"]) for row in history) + 1


def selected_parent(repo_root: Path, run_dir: Path) -> str | None:
    out = run_capture(
        [
            sys.executable,
            str(repo_root / "tools" / "evolution_db.py"),
            "select-parents",
            "--workspace",
            str(run_dir),
            "--n",
            "1",
            "--json",
        ],
        cwd=repo_root,
    )
    parents = json.loads(out)
    if not parents:
        return None
    return parents[0]["solution_id"]


def mutate_candidate(run_dir: Path, child_dir: Path, strategy: Strategy) -> None:
    child_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(run_dir / "kernel.py", child_dir / "kernel.py")
    if (child_dir / "src").exists():
        shutil.rmtree(child_dir / "src")
    shutil.copytree(run_dir / "src", child_dir / "src")
    cu_path = child_dir / "src" / "fused_moe_kernel.cu"
    text = cu_path.read_text()
    for old, new in strategy.replacements:
        if old not in text:
            raise RuntimeError(f"strategy {strategy.name} pattern not found: {old}")
        text = text.replace(old, new)
    cu_path.write_text(text)


def write_plan(run_dir: Path, generation: int, strategies: list[Strategy], parent_id: str | None) -> None:
    plan_dir = run_dir / "iteration" / str(generation) / "planner"
    plan_dir.mkdir(parents=True, exist_ok=True)
    lines = [
        f"# Auto PES Plan - Generation {generation}",
        "",
        f"- parent: `{parent_id}`",
        "- operator: `flashinfer.trtllm_fp4_block_scale_moe`",
        "- profile: `Qwen3_5-Plus_prefill_TP2`",
        "",
        "## Strategies",
    ]
    for idx, strategy in enumerate(strategies):
        lines.append(f"{idx}. `{strategy.name}` - {strategy.description}")
    (plan_dir / "plan.md").write_text("\n".join(lines) + "\n")


def add_candidate(
    repo_root: Path,
    run_dir: Path,
    generation: int,
    child_name: str,
    parent_id: str | None,
    strategy: Strategy,
    result: dict[str, Any],
    evidence_path: Path,
) -> None:
    latency = first_latency(result)
    correctness = "PASS" if result.get("status") == "PASS" else "FAIL"
    rel = max_rel(result)
    evaluation = f"{correctness}; latency_us={latency}; strategy={strategy.name}"
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
        strategy.name,
        "--generate-plan",
        strategy.description,
        "--evaluation",
        evaluation,
        "--evidence-file",
        str(evidence_path),
        "--iteration-ref",
        f"iteration/{generation}/executor/{child_name}",
    ]
    if parent_id is not None:
        cmd += ["--parent", parent_id]
    if latency is not None:
        cmd += ["--latency-us", str(latency)]
    if rel is not None:
        cmd += ["--rel-err", str(rel)]
    run_cmd(cmd, cwd=repo_root)


def write_summary(run_dir: Path, generation: int, rows: list[dict[str, Any]]) -> None:
    summary_dir = run_dir / "iteration" / str(generation) / "summarizer"
    summary_dir.mkdir(parents=True, exist_ok=True)
    ranked = sorted(rows, key=lambda r: (r["status"] != "PASS", r.get("latency_us") or float("inf")))
    lines = [f"# Auto PES Summary - Generation {generation}", ""]
    for row in ranked:
        lines.append(
            f"- `{row['child']}` `{row['strategy']}`: {row['status']}, "
            f"latency_us={row.get('latency_us')}, max_rel={row.get('max_rel')}"
        )
    if ranked:
        best = ranked[0]
        lines += [
            "",
            f"Best candidate: `{best['child']}` with strategy `{best['strategy']}`.",
            "Next direction: replace deterministic tile mutations with expert compaction and tiled FP4 dequant candidates.",
        ]
    (summary_dir / "summary.md").write_text("\n".join(lines) + "\n")


def checkpoint(repo_root: Path, run_dir: Path, generation: int) -> None:
    run_cmd(
        [
            sys.executable,
            str(repo_root / "tools" / "evolution_db.py"),
            "checkpoint",
            "--workspace",
            str(run_dir),
            "--generation",
            str(generation),
        ],
        cwd=repo_root,
    )


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
    args = parser.parse_args()

    source = args.source.resolve()
    repo_root = args.repo_root.resolve()
    run_dir = args.run_dir.resolve()
    copy_task(source, run_dir, fresh=args.fresh)
    init_db(repo_root, source, run_dir)

    baseline_json = run_dir / "profiles" / "auto_baseline.json"
    baseline = profile_kernel(run_dir, run_dir / "kernel.py", baseline_json, args)
    seed_id = ensure_seed(repo_root, run_dir, baseline)
    print(f"[Atrex] seed solution: {seed_id}")

    strategies = list(STRATEGIES[: max(1, args.n_candidates)])
    for _ in range(args.generations):
        generation = next_generation(run_dir)
        parent_id = selected_parent(repo_root, run_dir) or seed_id
        write_plan(run_dir, generation, strategies, parent_id)
        rows: list[dict[str, Any]] = []
        for idx, strategy in enumerate(strategies):
            child = f"{generation}_{idx}"
            child_dir = run_dir / "iteration" / str(generation) / "executor" / child
            mutate_candidate(run_dir, child_dir, strategy)
            (child_dir / "history.md").write_text(
                f"# {child}\n\n- parent: `{parent_id}`\n- strategy: `{strategy.name}`\n"
                f"- description: {strategy.description}\n"
            )
            result_path = child_dir / "result.json"
            result = profile_kernel(run_dir, child_dir / "kernel.py", result_path, args)
            evidence = {
                "tool_used": "torch.cuda.Event",
                "strategy": strategy.name,
                "status": result.get("status"),
                "latency_us": first_latency(result),
                "max_rel": result.get("max_rel"),
                "result_file": str(result_path.relative_to(run_dir)),
            }
            evidence_path = child_dir / "evidence.json"
            write_json(evidence_path, evidence)
            add_candidate(repo_root, run_dir, generation, child, parent_id, strategy, result, evidence_path)
            rows.append(
                {
                    "child": child,
                    "strategy": strategy.name,
                    "status": result.get("status"),
                    "latency_us": first_latency(result),
                    "max_rel": result.get("max_rel"),
                }
            )
        write_summary(run_dir, generation, rows)
        checkpoint(repo_root, run_dir, generation)

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
