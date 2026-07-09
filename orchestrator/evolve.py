#!/usr/bin/env python3
"""Generic PES/full-agent generation runner.

This is the task-agnostic slice of the full-agent runner: it owns the mechanics
around tools/evolution_db.py and delegates planning, execution, evaluation, and
summarization to command hooks. Hooks receive JSON on stdin and must print JSON
on stdout. The runner intentionally contains no FlashInfer, MoE, or benchmark
specific logic.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
EVOLUTION_DB = REPO_ROOT / "tools" / "evolution_db.py"


def run_json(cmd: str, payload: dict[str, Any], *, cwd: Path) -> dict[str, Any]:
    proc = subprocess.run(
        cmd,
        cwd=str(cwd),
        input=json.dumps(payload, indent=2),
        text=True,
        shell=True,
        check=True,
        stdout=subprocess.PIPE,
    )
    text = proc.stdout.strip()
    if not text:
        return {}
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"command did not return JSON: {cmd}\n{text[-2000:]}") from exc


def db(args: list[str], *, cwd: Path, as_json: bool = True) -> Any:
    cmd = [sys.executable, str(EVOLUTION_DB), *args]
    if as_json and "--json" not in cmd:
        cmd.append("--json")
    proc = subprocess.run(cmd, cwd=str(cwd), text=True, check=True, stdout=subprocess.PIPE)
    out = proc.stdout.strip()
    if as_json:
        return json.loads(out) if out else None
    return out


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")


def next_generation(workspace: Path) -> int:
    ckpt = workspace / "database" / "checkpoints"
    if not ckpt.exists():
        return 1
    seen: list[int] = []
    for path in ckpt.glob("iter-*"):
        try:
            seen.append(int(path.name.split("-", 1)[1]))
        except (IndexError, ValueError):
            continue
    return (max(seen) + 1) if seen else 1


def strategy_child(strategy: dict[str, Any], generation: int, index: int) -> str:
    child = strategy.get("child")
    return str(child) if child else f"{generation}_{index}"


def main() -> int:
    parser = argparse.ArgumentParser(description="Run one generic PES/full-agent generation.")
    parser.add_argument("--workspace", required=True, help="Kernel optimization workspace.")
    parser.add_argument("--generation", type=int, help="Generation number. Defaults to next checkpoint.")
    parser.add_argument("--n-candidates", type=int, default=3)
    parser.add_argument("--init-db", action="store_true")
    parser.add_argument("--import-seed", action="store_true")
    parser.add_argument("--seed-memory", default="memory/v0.json")
    parser.add_argument("--seed-code", default="kernel.py")
    parser.add_argument("--planner-cmd", required=True, help="Command hook; returns {'strategies': [...]}.")
    parser.add_argument("--executor-cmd", required=True, help="Command hook; returns candidate metadata.")
    parser.add_argument("--evaluator-cmd", required=True, help="Command hook; returns correctness/score/latency.")
    parser.add_argument("--summarizer-cmd", help="Optional command hook; returns summary metadata.")
    parser.add_argument("--lang", default="triton")
    parser.add_argument("--target-met", action="store_true")
    args = parser.parse_args()

    workspace = Path(args.workspace).resolve()
    generation = args.generation or next_generation(workspace)

    if args.init_db:
        db(["init", "--workspace", str(workspace)], cwd=REPO_ROOT, as_json=False)
    if args.import_seed:
        db(
            [
                "import-seed",
                "--workspace",
                str(workspace),
                "--from",
                args.seed_memory,
                "--code",
                args.seed_code,
            ],
            cwd=REPO_ROOT,
            as_json=True,
        )

    parents = db(
        ["select-parents", "--workspace", str(workspace), "--n", str(args.n_candidates)],
        cwd=REPO_ROOT,
        as_json=True,
    )
    evolve_config = db(["config", "--workspace", str(workspace)], cwd=REPO_ROOT, as_json=True)
    executor_config = evolve_config.get("executor") or {}
    react_threshold = float(executor_config.get("react_score_threshold", 2.0))
    parent_by_id = {p.get("solution_id"): p for p in parents if isinstance(p, dict)}

    planner_dir = workspace / "iteration" / str(generation) / "planner"
    planner_payload = {
        "workspace_path": str(workspace),
        "repo_root": str(REPO_ROOT),
        "generation": generation,
        "n_candidates": args.n_candidates,
        "parents": parents,
        "prev_summary": str(workspace / "iteration" / str(generation - 1) / "summarizer"),
        "database": str(workspace / "database"),
    }
    write_json(planner_dir / "planner_context.json", planner_payload)
    plan = run_json(args.planner_cmd, planner_payload, cwd=workspace)
    write_json(planner_dir / "plan.json", plan)
    strategies = plan.get("strategies")
    if not isinstance(strategies, list) or not strategies:
        raise RuntimeError("planner output must contain a non-empty strategies list")

    children: list[dict[str, Any]] = []
    for index, strategy in enumerate(strategies):
        child = strategy_child(strategy, generation, index)
        child_dir = workspace / "iteration" / str(generation) / "executor" / child
        child_dir.mkdir(parents=True, exist_ok=True)
        parent_id = strategy.get("parent_id")
        parent_record = parent_by_id.get(parent_id)
        if parent_record is None and index < len(parents):
            parent_record = parents[index]
            parent_id = parent_id or parent_record.get("solution_id")
        parent_score = float((parent_record or {}).get("score") or 0.0)
        execution_mode = "react" if parent_score >= react_threshold else "chat"
        parent_code = (parent_record or {}).get("code")
        parent_code_path = str((workspace / parent_code).resolve()) if parent_code else ""

        executor_payload = {
            "workspace_path": str(workspace),
            "repo_root": str(REPO_ROOT),
            "generation": generation,
            "child": child,
            "strategy": strategy,
            "parent_id": parent_id,
            "parent_score": parent_score,
            "parent_code": parent_code_path,
            "execution_mode": execution_mode,
            "executor_config": executor_config,
            "candidate_dir": str(child_dir),
            "candidate_code": str(child_dir / "kernel.py"),
            "candidate_solution": str(child_dir / "solution.json"),
        }
        write_json(child_dir / "executor_context.json", executor_payload)
        children.append(
            {
                "index": index,
                "child": child,
                "child_dir": child_dir,
                "strategy": strategy,
                "parent_id": parent_id,
                "executor_payload": executor_payload,
            }
        )

    def run_executor(item: dict[str, Any]) -> dict[str, Any]:
        result = run_json(args.executor_cmd, item["executor_payload"], cwd=workspace)
        write_json(item["child_dir"] / "executor_result.json", result)
        return {**item, "executor_result": result}

    def run_evaluator(item: dict[str, Any]) -> dict[str, Any]:
        child = item["child"]
        child_dir = item["child_dir"]
        executor_result = item["executor_result"]
        candidate_code = executor_result.get("candidate_code") or str(child_dir / "kernel.py")
        evaluator_payload = {
            "workspace_path": str(workspace),
            "repo_root": str(REPO_ROOT),
            "generation": generation,
            "child": child,
            "strategy": item["strategy"],
            "executor_result": executor_result,
            "candidate_code": candidate_code,
            "profiles_dir": str(workspace / "profiles" / str(generation) / child),
        }
        write_json(child_dir / "evaluator_context.json", evaluator_payload)
        evaluator_result = run_json(args.evaluator_cmd, evaluator_payload, cwd=workspace)
        write_json(child_dir / "evaluator_result.json", evaluator_result)
        return {
            **item,
            "candidate_code": candidate_code,
            "evaluator_result": evaluator_result,
        }

    max_workers = max(1, len(children))
    executor_results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = [pool.submit(run_executor, item) for item in children]
        for future in as_completed(futures):
            executor_results.append(future.result())
    executor_results.sort(key=lambda item: item["index"])

    evaluated_results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = [pool.submit(run_evaluator, item) for item in executor_results]
        for future in as_completed(futures):
            evaluated_results.append(future.result())
    evaluated_results.sort(key=lambda item: item["index"])

    results: list[dict[str, Any]] = []
    for item in evaluated_results:
        child = item["child"]
        strategy = item["strategy"]
        executor_result = item["executor_result"]
        evaluator_result = item["evaluator_result"]
        candidate_code = item["candidate_code"]
        correctness = evaluator_result.get("correctness") or evaluator_result.get("status") or "FAIL"
        latency = evaluator_result.get("latency_us")
        score = evaluator_result.get("score")
        evidence_file = evaluator_result.get("evidence_file")
        add_args = [
            "add",
            "--workspace",
            str(workspace),
            "--generation",
            str(generation),
            "--parent",
            str(item.get("parent_id") or "null"),
            "--code",
            str(Path(candidate_code).resolve()),
            "--lang",
            str(executor_result.get("lang") or args.lang),
            "--correctness",
            str(correctness),
            "--action-category",
            str(strategy.get("action_category") or ""),
            "--generate-plan",
            str(strategy.get("action_description") or strategy.get("generate_plan") or ""),
            "--iteration-ref",
            f"iteration/{generation}/executor/{child}",
        ]
        if score is not None:
            add_args += ["--score", str(score)]
        if latency is not None:
            add_args += ["--latency-us", str(latency)]
        if evidence_file:
            add_args += ["--evidence-file", str(evidence_file)]
        admitted = db(add_args, cwd=REPO_ROOT, as_json=True)
        results.append(
            {
                "child": child,
                "strategy": strategy,
                "executor_result": executor_result,
                "evaluator_result": evaluator_result,
                "admitted": admitted,
            }
        )

    summarizer_dir = workspace / "iteration" / str(generation) / "summarizer"
    summary_payload = {
        "workspace_path": str(workspace),
        "repo_root": str(REPO_ROOT),
        "generation": generation,
        "results": results,
    }
    write_json(summarizer_dir / "summarizer_context.json", summary_payload)
    if args.summarizer_cmd:
        summary = run_json(args.summarizer_cmd, summary_payload, cwd=workspace)
        summary_file = summarizer_dir / "summary.json"
        write_json(summary_file, summary)
        db(
            [
                "annotate-generation",
                "--workspace",
                str(workspace),
                "--generation",
                str(generation),
                "--summary-file",
                str(summary_file),
            ],
            cwd=REPO_ROOT,
            as_json=True,
        )

    checkpoint_args = ["checkpoint", "--workspace", str(workspace), "--generation", str(generation)]
    if args.target_met:
        checkpoint_args.append("--target-met")
    checkpoint = db(checkpoint_args, cwd=REPO_ROOT, as_json=True)
    print(json.dumps({"generation": generation, "results": results, "checkpoint": checkpoint}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
