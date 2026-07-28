#!/usr/bin/env python3
"""Validate the external LoongFlow checkout against a pinned compatibility contract."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path
from typing import Any, Sequence


class UpstreamContractError(RuntimeError):
    """Raised when the external LoongFlow checkout is incompatible or unpinned."""


def _truthy(name: str) -> bool:
    return os.environ.get(name, "0").strip().lower() in {"1", "true", "yes", "on"}


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
        raise UpstreamContractError("LoongFlow compatibility contract must be an object")

    repository_root = Path(_git(project, "rev-parse", "--show-toplevel"))
    commit = _git(repository_root, "rev-parse", "HEAD")
    expected = str(
        os.environ.get("ATREX_LOONGFLOW_EXPECTED_COMMIT")
        or contract.get("expected_commit")
        or ""
    ).strip()
    if not expected:
        raise UpstreamContractError("LoongFlow compatibility contract has no expected_commit")
    if commit != expected and not _truthy("ATREX_LOONGFLOW_ALLOW_UNPINNED"):
        raise UpstreamContractError(
            "LoongFlow commit mismatch: "
            f"expected {expected}, got {commit}. Revalidate patches and update "
            f"{contract_file}, or set ATREX_LOONGFLOW_ALLOW_UNPINNED=1 for a deliberate dry run."
        )

    tracked_status = _git(
        repository_root, "status", "--porcelain=v1", "--untracked-files=no"
    )
    if tracked_status and not _truthy("ATREX_LOONGFLOW_ALLOW_DIRTY"):
        raise UpstreamContractError(
            "LoongFlow checkout has tracked modifications; set "
            "ATREX_LOONGFLOW_ALLOW_DIRTY=1 only after reviewing the compatibility diff"
        )

    missing = [
        relative
        for relative in contract.get("required_paths", [])
        if not (project / str(relative)).is_file()
    ]
    if missing:
        raise UpstreamContractError(
            "LoongFlow checkout is missing required compatibility targets: "
            + ", ".join(str(path) for path in missing)
        )

    return {
        "contract_schema_version": int(contract.get("schema_version", 1)),
        "contract_path": str(contract_file),
        "project_root": str(project),
        "repository_root": str(repository_root),
        "expected_commit": expected,
        "actual_commit": commit,
        "commit_matches": commit == expected,
        "tracked_dirty": bool(tracked_status),
        "required_paths": list(contract.get("required_paths", [])),
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
