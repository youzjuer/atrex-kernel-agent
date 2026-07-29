"""Install validated Atrex adapters, then run a LoongFlow Python entrypoint."""

from __future__ import annotations

import argparse
import json
import runpy
import sys
from pathlib import Path
from typing import Sequence

from orchestrator.loongflow_compat import compat_adapter


def install_and_validate() -> dict[str, dict[str, object]]:
    """Apply compatibility adapters explicitly and fail on any required patch."""
    compat_adapter.apply_compat_patches()
    return compat_adapter.validate_patch_manifest()


def run_entrypoint(entrypoint: str | Path, arguments: Sequence[str]) -> None:
    path = Path(entrypoint).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"LoongFlow entrypoint does not exist: {path}")
    sys.argv = [str(path), *arguments]
    runpy.run_path(str(path), run_name="__main__")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="apply and validate adapters without running an entrypoint",
    )
    parser.add_argument("entrypoint", nargs="?")
    parser.add_argument("arguments", nargs=argparse.REMAINDER)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    manifest = install_and_validate()
    print("[Atrex] Patch manifest: " + json.dumps(manifest, sort_keys=True))
    if args.validate_only:
        if args.entrypoint is not None:
            raise SystemExit("--validate-only does not accept an entrypoint")
        return 0
    if args.entrypoint is None:
        raise SystemExit("a LoongFlow Python entrypoint is required")
    run_entrypoint(args.entrypoint, args.arguments)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
