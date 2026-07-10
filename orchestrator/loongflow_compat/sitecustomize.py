"""Process-wide compatibility hooks for local LoongFlow runners."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path


if os.environ.get("ATREX_LITELLM_DROP_PARAMS", "0") == "1":
    try:
        import litellm

        litellm.drop_params = True
    except Exception:
        pass


def _truncate_text(value: object, limit: int) -> object:
    if not isinstance(value, str) or len(value) <= limit:
        return value
    head = max(0, limit * 2 // 3)
    tail = max(0, limit - head)
    return (
        value[:head]
        + f"\n\n...[atrex compacted {len(value) - limit} chars]...\n\n"
        + value[-tail:]
    )


def _compact_solution_record(record: object) -> object:
    if not isinstance(record, dict):
        return record

    compact = dict(record)
    solution = compact.get("solution")
    if isinstance(solution, str):
        compact["solution_sha1"] = hashlib.sha1(solution.encode("utf-8")).hexdigest()
        compact["solution_chars"] = len(solution)
        compact["solution"] = _truncate_text(solution, 2400)

    summary = compact.get("summary")
    if isinstance(summary, str):
        compact["summary"] = _truncate_text(summary, 3500)

    evaluation = compact.get("evaluation")
    if isinstance(evaluation, str):
        try:
            parsed = json.loads(evaluation)
        except Exception:
            compact["evaluation"] = _truncate_text(evaluation, 3000)
        else:
            metrics = parsed.get("metrics") if isinstance(parsed, dict) else None
            per_workload = parsed.get("per_workload") if isinstance(parsed, dict) else None
            compact_eval = {
                "status": parsed.get("status") if isinstance(parsed, dict) else None,
                "summary": parsed.get("summary") if isinstance(parsed, dict) else None,
                "score": parsed.get("score") if isinstance(parsed, dict) else None,
                "metrics": metrics,
            }
            if isinstance(per_workload, list):
                compact_eval["per_workload_latency_ms"] = [
                    {
                        "index": item.get("index"),
                        "axes": item.get("axes"),
                        "status": item.get("status"),
                        "latency_ms": item.get("latency_ms"),
                    }
                    for item in per_workload[:32]
                    if isinstance(item, dict)
                ]
            compact["evaluation"] = compact_eval
    return compact


def _compact_result(value: object) -> object:
    if isinstance(value, list):
        return [_compact_solution_record(item) for item in value]
    if isinstance(value, dict):
        return _compact_solution_record(value)
    return value


def _wrap_database_func(func):
    if func is None or getattr(func, "_atrex_compacted", False):
        return func

    def compacted_func(*args, **kwargs):
        return _compact_result(func(*args, **kwargs))

    compacted_func._atrex_compacted = True
    return compacted_func


def _patch_database_tools() -> None:
    if os.environ.get("ATREX_PES_COMPACT_DB_TOOLS", "1") != "1":
        return
    try:
        from loongflow.framework.pes.database import database_tool
    except Exception:
        return

    for class_name in (
        "GetSolutionsTool",
        "GetBestSolutionsTool",
        "GetParentsByChildIdTool",
        "GetChildsByParentTool",
    ):
        tool_cls = getattr(database_tool, class_name, None)
        if tool_cls is None or getattr(tool_cls, "_atrex_patched", False):
            continue
        original_init = tool_cls.__init__

        def patched_init(self, func=None, _original_init=original_init):
            _original_init(self, _wrap_database_func(func))
            self.description += (
                " Atrex compatibility: solution, evaluation, and summary fields "
                "are compacted to keep PES ReAct memory below the context limit."
            )

        tool_cls.__init__ = patched_init
        tool_cls._atrex_patched = True


def _patch_planner_write_tool() -> None:
    try:
        from agents.math_agent.planner import build_tool
        from loongflow.agentsdk.tools import FunctionTool
        from loongflow.framework.pes.context import Workspace
    except Exception:
        return

    if getattr(build_tool, "_atrex_write_patched", False):
        return

    def build_planner_write_tool(context):
        async def write_func(file_path: str, content: str):
            planner_base = Path(Workspace.get_planner_path(context))
            planner_base.mkdir(parents=True, exist_ok=True)

            requested = Path(file_path)
            target = requested if requested.is_absolute() else planner_base / requested.name
            if not str(target).startswith(str(planner_base)):
                target = planner_base / target.name

            aliases = {
                "plan1.txt": ("plan1.txt", "plan_1.txt"),
                "plan_1.txt": ("plan1.txt", "plan_1.txt"),
                "plan2.txt": ("plan2.txt", "plan_2.txt"),
                "plan_2.txt": ("plan2.txt", "plan_2.txt"),
                "plan3.txt": ("plan3.txt", "plan_3.txt"),
                "plan_3.txt": ("plan3.txt", "plan_3.txt"),
            }
            names = aliases.get(target.name, (target.name,))
            for name in names:
                out = planner_base / name
                out.parent.mkdir(parents=True, exist_ok=True)
                out.write_text(content, encoding="utf-8")
            return "File written successfully"

        return FunctionTool(
            func=write_func,
            args_schema=build_tool.WriteToolArgs,
            name="Write",
            description=(
                "Writes a file to the planner workspace. Supports plan1.txt and "
                "plan_1.txt naming variants."
            ),
        )

    build_tool.build_planner_write_tool = build_planner_write_tool
    build_tool._atrex_write_patched = True


def _patch_planner_empty_plan_fallback() -> None:
    try:
        from agents.math_agent.planner.plan_agent import EvolvePlanAgent
        from loongflow.agentsdk.message import ContentElement
    except Exception:
        return

    if getattr(EvolvePlanAgent, "_atrex_empty_plan_patched", False):
        return

    original_run = EvolvePlanAgent.run

    async def patched_run(self, context, message):
        result = await original_run(self, context, message)
        try:
            elements = result.get_elements(ContentElement)
            data = elements[0].data if elements else {}
            best_plan_path = Path(data.get("best_plan_file_path", ""))
            if best_plan_path.exists() and best_plan_path.read_text(encoding="utf-8").strip():
                return result

            planner_dir = best_plan_path.parent
            fallback = ""
            for name in ("plan_3.txt", "plan3.txt", "plan_2.txt", "plan2.txt", "plan_1.txt", "plan1.txt"):
                candidate = planner_dir / name
                if candidate.exists():
                    text = candidate.read_text(encoding="utf-8").strip()
                    if text:
                        fallback = text
                        break
            if not fallback:
                fallback = (
                    "### Final Child Solution Generation Plan\n\n"
                    "**Objective:** Improve the sampled parent while preserving correctness.\n\n"
                    "**Best Plan:**\n"
                    "1. Keep the parent solution as the implementation substrate; do not rewrite unrelated kernels.\n"
                    "2. Use the parent summary and evaluation metrics to identify the slowest workload groups.\n"
                    "3. Make one localized CUDA dispatch or kernel change that targets those workloads only.\n"
                    "4. Preserve known fast paths and graph-cache behavior unless the evaluation proves a change is faster.\n"
                    "5. Return a complete candidate source and rely on the evaluator as the only promotion gate.\n"
                )
            best_plan_path.write_text(fallback, encoding="utf-8")
        except Exception:
            pass
        return result

    EvolvePlanAgent.run = patched_run
    EvolvePlanAgent._atrex_empty_plan_patched = True


def _patch_fuse_executor_parallel_cap() -> None:
    cap_raw = os.environ.get("ATREX_PES_MAX_PARALLEL_CANDIDATES", "")
    if not cap_raw:
        return
    try:
        cap = max(1, int(cap_raw))
    except ValueError:
        return

    try:
        from agents.math_agent.executor.execute_fuse.execute_agent_fuse import (
            EvolveExecuteAgentFuse,
        )
    except Exception:
        return

    if getattr(EvolveExecuteAgentFuse, "_atrex_parallel_cap_patched", False):
        return

    original_gen_multi_candidate = EvolveExecuteAgentFuse.gen_multi_candidate

    async def patched_gen_multi_candidate(
        self,
        context,
        parent_ctx,
        round_idx,
        parallel_candidates,
        previous_attempts,
    ):
        return await original_gen_multi_candidate(
            self,
            context,
            parent_ctx,
            round_idx,
            min(parallel_candidates, cap),
            previous_attempts,
        )

    EvolveExecuteAgentFuse.gen_multi_candidate = patched_gen_multi_candidate
    EvolveExecuteAgentFuse._atrex_parallel_cap_patched = True


_patch_database_tools()
_patch_planner_write_tool()
_patch_planner_empty_plan_fallback()
_patch_fuse_executor_parallel_cap()
