#!/usr/bin/env python3
"""Validate the external LoongFlow checkout against a pinned compatibility contract."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any, Sequence


class UpstreamContractError(RuntimeError):
    """Raised when the external LoongFlow checkout is incompatible or unpinned."""


def _truthy(name: str) -> bool:
    return os.environ.get(name, "0").strip().lower() in {"1", "true", "yes", "on"}


def _override_state() -> dict[str, Any]:
    allow_unpinned = _truthy("ATREX_LOONGFLOW_ALLOW_UNPINNED")
    allow_dirty = _truthy("ATREX_LOONGFLOW_ALLOW_DIRTY")
    requested = allow_unpinned or allow_dirty
    reason = os.environ.get("ATREX_LOONGFLOW_OVERRIDE_REASON", "").strip()
    if requested and (_truthy("CI") or _truthy("GITHUB_ACTIONS")):
        raise UpstreamContractError(
            "LoongFlow compatibility overrides are forbidden in CI"
        )
    if requested and not reason:
        raise UpstreamContractError(
            "LoongFlow compatibility overrides require ATREX_LOONGFLOW_OVERRIDE_REASON"
        )
    return {
        "allow_unpinned": allow_unpinned,
        "allow_dirty": allow_dirty,
        "reason": reason,
    }


def _sha256(path: Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as exc:
        raise UpstreamContractError(
            f"cannot hash LoongFlow target {path}: {exc}"
        ) from exc


def _callable_parameters(path: Path, qualified_name: str) -> list[str]:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except (OSError, SyntaxError) as exc:
        raise UpstreamContractError(
            f"cannot parse LoongFlow target {path}: {exc}"
        ) from exc
    parts = qualified_name.split(".")
    body: list[ast.stmt] = tree.body
    node: ast.AST | None = None
    for index, part in enumerate(parts):
        node = next(
            (
                item
                for item in body
                if isinstance(
                    item,
                    (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef),
                )
                and item.name == part
            ),
            None,
        )
        if node is None:
            raise UpstreamContractError(
                f"LoongFlow callable {qualified_name!r} not found in {path}"
            )
        if index < len(parts) - 1:
            if not isinstance(node, ast.ClassDef):
                raise UpstreamContractError(
                    f"LoongFlow callable path {qualified_name!r} crosses non-class {part!r}"
                )
            body = node.body
    if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        raise UpstreamContractError(
            f"LoongFlow target {qualified_name!r} in {path} is not callable"
        )
    arguments = node.args
    names = [argument.arg for argument in arguments.posonlyargs]
    names.extend(argument.arg for argument in arguments.args)
    if arguments.vararg is not None:
        names.append(arguments.vararg.arg)
    names.extend(argument.arg for argument in arguments.kwonlyargs)
    if arguments.kwarg is not None:
        names.append(arguments.kwarg.arg)
    return names


def _git(root: Path, *arguments: str) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(root), *arguments],
            text=True,
            stderr=subprocess.STDOUT,
            timeout=10,
        ).strip()
    except (OSError, subprocess.SubprocessError) as exc:
        raise UpstreamContractError(
            f"cannot inspect LoongFlow git checkout at {root}: {exc}"
        ) from exc


def validate_upstream_checkout(
    project_root: str | Path, contract_path: str | Path
) -> dict[str, Any]:
    project = Path(project_root).resolve()
    contract_file = Path(contract_path).resolve()
    try:
        contract = json.loads(contract_file.read_text(encoding="utf-8"))
    except (OSError, TypeError, ValueError) as exc:
        raise UpstreamContractError(
            f"cannot read LoongFlow compatibility contract {contract_file}: {exc}"
        ) from exc
    if not isinstance(contract, dict):
        raise UpstreamContractError(
            "LoongFlow compatibility contract must be an object"
        )
    if contract.get("schema_version") != 2:
        raise UpstreamContractError(
            "LoongFlow compatibility contract must use schema_version 2"
        )
    targets = contract.get("targets")
    if not isinstance(targets, dict) or not targets:
        raise UpstreamContractError("LoongFlow compatibility contract has no targets")
    overrides = _override_state()

    repository_root = Path(_git(project, "rev-parse", "--show-toplevel"))
    commit = _git(repository_root, "rev-parse", "HEAD")
    expected = str(
        os.environ.get("ATREX_LOONGFLOW_EXPECTED_COMMIT")
        or contract.get("expected_commit")
        or ""
    ).strip()
    if not expected:
        raise UpstreamContractError(
            "LoongFlow compatibility contract has no expected_commit"
        )
    if commit != expected and not overrides["allow_unpinned"]:
        raise UpstreamContractError(
            "LoongFlow commit mismatch: "
            f"expected {expected}, got {commit}. Revalidate patches and update "
            f"{contract_file}, or set ATREX_LOONGFLOW_ALLOW_UNPINNED=1 with an explicit "
            "ATREX_LOONGFLOW_OVERRIDE_REASON for a deliberate local dry run."
        )

    tracked_status = _git(
        repository_root, "status", "--porcelain=v1", "--untracked-files=no"
    )
    if tracked_status and not overrides["allow_dirty"]:
        raise UpstreamContractError(
            "LoongFlow checkout has tracked modifications; set "
            "ATREX_LOONGFLOW_ALLOW_DIRTY=1 with ATREX_LOONGFLOW_OVERRIDE_REASON "
            "only after reviewing the compatibility diff"
        )

    missing = [
        relative for relative in targets if not (project / str(relative)).is_file()
    ]
    if missing:
        raise UpstreamContractError(
            "LoongFlow checkout is missing required compatibility targets: "
            + ", ".join(str(path) for path in missing)
        )

    target_results: dict[str, Any] = {}
    content_drift_allowed = bool(
        overrides["allow_unpinned"] or overrides["allow_dirty"]
    )
    for relative, target_contract in targets.items():
        if not isinstance(target_contract, dict):
            raise UpstreamContractError(f"invalid target contract for {relative}")
        expected_hash = str(target_contract.get("sha256") or "")
        if len(expected_hash) != 64:
            raise UpstreamContractError(f"target {relative} has no valid sha256")
        target_path = project / relative
        actual_hash = _sha256(target_path)
        hash_matches = actual_hash == expected_hash
        if not hash_matches and not content_drift_allowed:
            raise UpstreamContractError(
                f"LoongFlow target hash mismatch for {relative}: "
                f"expected {expected_hash}, got {actual_hash}"
            )
        callable_results: dict[str, Any] = {}
        callables = target_contract.get("callables", {})
        if not isinstance(callables, dict):
            raise UpstreamContractError(
                f"target {relative} callables must be an object"
            )
        for qualified_name, expected_parameters in callables.items():
            if not isinstance(expected_parameters, list) or not all(
                isinstance(parameter, str) for parameter in expected_parameters
            ):
                raise UpstreamContractError(
                    f"target {relative} callable {qualified_name} has invalid parameters"
                )
            actual_parameters = _callable_parameters(target_path, qualified_name)
            if actual_parameters != expected_parameters:
                raise UpstreamContractError(
                    f"LoongFlow signature mismatch for {relative}:{qualified_name}: "
                    f"expected {expected_parameters}, got {actual_parameters}"
                )
            callable_results[qualified_name] = actual_parameters
        target_results[relative] = {
            "expected_sha256": expected_hash,
            "actual_sha256": actual_hash,
            "hash_matches": hash_matches,
            "callables": callable_results,
        }

    return {
        "contract_schema_version": 2,
        "contract_path": str(contract_file),
        "project_root": str(project),
        "repository_root": str(repository_root),
        "expected_commit": expected,
        "actual_commit": commit,
        "commit_matches": commit == expected,
        "tracked_dirty": bool(tracked_status),
        "targets": target_results,
        "overrides": {
            "active": bool(overrides["allow_unpinned"] or overrides["allow_dirty"]),
            **overrides,
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--contract", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = validate_upstream_checkout(args.project_root, args.contract)
    print("[Atrex] LoongFlow contract: " + json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
