#!/usr/bin/env python3
"""Resolve and validate the single SOL58 runtime configuration."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

from orchestrator.sol_execbench_task import (
    TaskSpecError,
    load_task_spec,
    reject_task_identity_overrides,
)
from orchestrator.sol58_pes.runtime_schema import (
    PATH_ENV,
    PROFILE_SETTINGS,
    SETTINGS,
    Setting,
    _setting,
)


class RuntimeConfigError(ValueError):
    """Raised when the checked-in config or an environment override is invalid."""


_PROFILE_ALIASES = {
    "official": "official_v1_1_b200",
    "official_v1_1": "official_v1_1_b200",
    "official_v1_1_b200": "official_v1_1_b200",
    "native": "native",
}
_TRUE = {"1", "true", "yes", "on"}
_FALSE = {"0", "false", "no", "off"}


def _flatten(value: Any, prefix: str = "") -> dict[str, Any]:
    if not isinstance(value, dict):
        return {prefix: value}
    flattened: dict[str, Any] = {}
    for key, child in value.items():
        path = f"{prefix}.{key}" if prefix else str(key)
        flattened.update(_flatten(child, path))
    return flattened


def _convert(value: Any, setting: Setting, *, source: str) -> Any:
    try:
        if setting.kind == "bool":
            if isinstance(value, bool):
                converted = value
            elif (
                source == "environment" and str(value).strip().lower() in _TRUE | _FALSE
            ):
                converted = str(value).strip().lower() in _TRUE
            else:
                raise ValueError("expected boolean")
        elif setting.kind == "int":
            if isinstance(value, bool):
                raise ValueError("expected integer")
            converted = int(value)
        elif setting.kind == "float":
            if isinstance(value, bool):
                raise ValueError("expected number")
            converted = float(value)
        elif setting.kind == "str":
            converted = str(value).strip()
            if not converted:
                raise ValueError("expected non-empty string")
        else:
            raise ValueError(f"unsupported schema kind {setting.kind!r}")
    except (TypeError, ValueError) as exc:
        raise RuntimeConfigError(
            f"invalid {source} value for {setting.path} ({setting.env}): "
            f"{value!r} ({exc})"
        ) from exc
    if setting.choices and converted not in setting.choices:
        raise RuntimeConfigError(
            f"{setting.path} must be one of {setting.choices}, got {converted!r}"
        )
    if (
        setting.minimum is not None
        and setting.kind in {"int", "float"}
        and float(converted) < setting.minimum
    ):
        raise RuntimeConfigError(
            f"{setting.path} must be >= {setting.minimum}, got {converted}"
        )
    if (
        setting.maximum is not None
        and setting.kind in {"int", "float"}
        and float(converted) > setting.maximum
    ):
        raise RuntimeConfigError(
            f"{setting.path} must be <= {setting.maximum}, got {converted}"
        )
    return converted


def _shell_value(value: Any) -> str:
    if isinstance(value, bool):
        return "1" if value else "0"
    return str(value)


def _resolve_relative(path: str, repo_root: Path) -> Path:
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = repo_root / candidate
    return candidate.resolve(strict=False)


def _auto_path(
    name: str,
    repo_root: Path,
    task_dir: Path,
    resolved: Mapping[str, str],
    *,
    problem_slug: str,
) -> str:
    sibling_root = repo_root.parent
    if name == "contest_root":
        return str((sibling_root / "mlsys26-flashinfer-contest").resolve(strict=False))
    if name == "problem_dir":
        return str((sibling_root / "sol-problems" / problem_slug).resolve(strict=False))
    if name == "sol_execbench":
        sibling_binary = (
            sibling_root / "sol-execbench" / ".venv" / "bin" / "sol-execbench"
        )
        if sibling_binary.is_file():
            return str(sibling_binary.resolve())
        return shutil.which("sol-execbench") or "sol-execbench"
    if name == "workspace":
        return str((Path(resolved["run_dir"]) / "output").resolve(strict=False))
    if name == "knowledge_pack":
        return str((task_dir / "knowledge_pack.md").resolve())
    if name == "seed_manifest":
        return str((task_dir / "seed_bank.json").resolve())
    if name == "ncu_cache_dir":
        return str((Path(resolved["eval_root"]) / "ncu_cache").resolve(strict=False))
    raise RuntimeConfigError(f"path {name!r} does not support auto discovery")


def _resolve_paths(
    configured: Mapping[str, Any],
    environ: Mapping[str, str],
    repo_root: Path,
    task_dir: Path,
    *,
    problem_slug: str,
) -> tuple[dict[str, str], dict[str, str]]:
    values: dict[str, str] = {}
    sources: dict[str, str] = {}
    for name, env_name in PATH_ENV.items():
        raw = environ.get(env_name)
        source = "environment" if raw else "config"
        if not raw and name == "sol_execbench":
            raw = environ.get("SOL58_OFFICIAL_LOCAL_SOL_EXECBENCH")
            source = "legacy_environment" if raw else source
        if not raw:
            raw = configured.get(name)
        if not isinstance(raw, str) or not raw.strip():
            raise RuntimeConfigError(f"paths.{name} must be a non-empty string")
        raw = raw.strip()
        if raw == "auto":
            values[name] = _auto_path(
                name,
                repo_root,
                task_dir,
                values,
                problem_slug=problem_slug,
            )
            sources[name] = "auto_discovery"
        elif name == "sol_execbench" and "/" not in raw:
            values[name] = raw
            sources[name] = source
        else:
            values[name] = str(_resolve_relative(raw, repo_root))
            sources[name] = source
    return values, sources


def load_runtime_config(
    config_path: str | Path,
    *,
    repo_root: str | Path,
    task_dir: str | Path,
    environ: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    config_file = Path(config_path).resolve()
    try:
        config = json.loads(config_file.read_text(encoding="utf-8"))
    except (OSError, TypeError, ValueError) as exc:
        raise RuntimeConfigError(
            f"cannot read runtime config {config_file}: {exc}"
        ) from exc
    if not isinstance(config, dict) or config.get("schema_version") != 1:
        raise RuntimeConfigError("runtime config must use schema_version 1")

    flattened = _flatten(config)
    expected = {
        "schema_version",
        "task_spec",
        "official.provisional_floor_delta",
    }
    expected.update(setting.path for setting in SETTINGS)
    expected.update(f"paths.{name}" for name in PATH_ENV)
    profiles = config.get("measurement", {}).get("profiles", {})
    if not isinstance(profiles, dict) or set(profiles) != {
        "official_v1_1_b200",
        "native",
    }:
        raise RuntimeConfigError(
            "measurement.profiles must define official_v1_1_b200 and native"
        )
    for profile_name in profiles:
        expected.update(
            f"measurement.profiles.{profile_name}.{setting.path}"
            for setting in PROFILE_SETTINGS
        )
    actual = set(flattened)
    if actual != expected:
        missing = sorted(expected - actual)
        unknown = sorted(actual - expected)
        raise RuntimeConfigError(
            f"runtime config schema mismatch; missing={missing}, unknown={unknown}"
        )
    provisional_delta = _convert(
        flattened["official.provisional_floor_delta"],
        _setting(
            "official.provisional_floor_delta",
            "SOL58_OFFICIAL_PROVISIONAL_SCORE_CAP",
            "float",
            minimum=0,
        ),
        source="config",
    )

    source_env = os.environ if environ is None else environ
    task_spec_value = config.get("task_spec")
    if not isinstance(task_spec_value, str) or not task_spec_value.strip():
        raise RuntimeConfigError("task_spec must be a non-empty path")
    task_spec_path = Path(task_spec_value).expanduser()
    if not task_spec_path.is_absolute():
        task_spec_path = Path(task_dir).resolve() / task_spec_path
    try:
        task_spec = load_task_spec(task_spec_path)
        reject_task_identity_overrides(task_spec, source_env)
    except TaskSpecError as exc:
        raise RuntimeConfigError(str(exc)) from exc

    values: dict[str, str] = {}
    records: dict[str, dict[str, Any]] = {}
    for setting in SETTINGS:
        raw = source_env.get(setting.env)
        source = "environment" if raw is not None else "config"
        if raw is None:
            raw = flattened[setting.path]
        if setting.env == "SOL58_MEASUREMENT_PROFILE":
            raw = _PROFILE_ALIASES.get(str(raw).strip().lower(), raw)
        converted = _convert(raw, setting, source=source)
        values[setting.env] = _shell_value(converted)
        records[setting.env] = {
            "value": converted,
            "source": source,
            "path": setting.path,
        }

    profile_name = values["SOL58_MEASUREMENT_PROFILE"]
    for setting in PROFILE_SETTINGS:
        path = f"measurement.profiles.{profile_name}.{setting.path}"
        raw = source_env.get(setting.env)
        source = "environment" if raw is not None else f"profile:{profile_name}"
        if raw is None:
            raw = flattened[path]
        converted = _convert(raw, setting, source=source)
        values[setting.env] = _shell_value(converted)
        records[setting.env] = {"value": converted, "source": source, "path": path}

    paths, path_sources = _resolve_paths(
        config["paths"],
        source_env,
        Path(repo_root).resolve(),
        Path(task_dir).resolve(),
        problem_slug=task_spec.problem_slug,
    )
    for name, env_name in PATH_ENV.items():
        values[env_name] = paths[name]
        records[env_name] = {
            "value": paths[name],
            "source": path_sources[name],
            "path": f"paths.{name}",
        }

    for name, value in task_spec.environment().items():
        values[name] = value
        records[name] = {
            "value": value,
            "source": "task_spec",
            "path": str(task_spec.path),
        }

    visible = source_env.get("CUDA_VISIBLE_DEVICES")
    if visible:
        values["CUDA_VISIBLE_DEVICES"] = visible
        visible_source = "environment"
    else:
        values["CUDA_VISIBLE_DEVICES"] = values["SOL58_CLOCK_GPU_INDEX"]
        visible_source = "derived:measurement.clock_gpu_index"
    records["CUDA_VISIBLE_DEVICES"] = {
        "value": values["CUDA_VISIBLE_DEVICES"],
        "source": visible_source,
        "path": "measurement.clock_gpu_index",
    }

    cap_raw = source_env.get("SOL58_OFFICIAL_PROVISIONAL_SCORE_CAP")
    if cap_raw is None:
        delta = float(provisional_delta)
        cap = float(values["SOL58_TARGET_SCORE"]) - delta
        cap_source = "derived:official.provisional_floor_delta"
    else:
        cap = float(cap_raw)
        cap_source = "environment"
    values["SOL58_OFFICIAL_PROVISIONAL_SCORE_CAP"] = str(cap)
    records["SOL58_OFFICIAL_PROVISIONAL_SCORE_CAP"] = {
        "value": cap,
        "source": cap_source,
        "path": "official.provisional_floor_delta",
    }

    if (
        values["ATREX_PES_ARCHITECTURE_ISLANDS"] == "1"
        and int(values["SOL58_NUM_ISLANDS"]) < 8
    ):
        raise RuntimeConfigError("architecture-aware PES requires at least 8 islands")

    warning = task_spec.staleness_warning()
    return {
        "schema_version": 1,
        "config_path": str(config_file),
        "task_spec": {
            "path": str(task_spec.path),
            "task_name": task_spec.task_name,
            "kernel_id": task_spec.kernel_id,
            "leaderboard_url": task_spec.leaderboard_url,
            "snapshot_at": task_spec.snapshot_at.isoformat(),
            "snapshot_max_age_days": task_spec.snapshot_max_age_days,
        },
        "profile": profile_name,
        "warnings": [warning] if warning else [],
        "values": dict(sorted(values.items())),
        "records": dict(sorted(records.items())),
    }


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def write_resolved_config(
    result: Mapping[str, Any], env_output: str | Path, json_output: str | Path
) -> None:
    shell_lines = [
        f"export {name}={shlex.quote(str(value))}"
        for name, value in result["values"].items()
    ]
    _atomic_write(Path(env_output), "\n".join(shell_lines) + "\n")
    _atomic_write(
        Path(json_output), json.dumps(result, indent=2, sort_keys=True) + "\n"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--task-dir", required=True)
    parser.add_argument("--env-output", required=True)
    parser.add_argument("--json-output", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = load_runtime_config(
        args.config, repo_root=args.repo_root, task_dir=args.task_dir
    )
    for warning in result["warnings"]:
        print(f"[Atrex] warning: {warning}", file=sys.stderr)
    write_resolved_config(result, args.env_output, args.json_output)
    print(
        "[Atrex] Resolved SOL58 config: "
        f"profile={result['profile']} settings={len(result['values'])}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
