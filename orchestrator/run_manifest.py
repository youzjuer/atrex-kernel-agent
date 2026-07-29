#!/usr/bin/env python3
"""Create redacted, reproducible manifests for long-running optimizer jobs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence


_ENV_PREFIXES = (
    "SOL58_",
    "ATREX_",
    "LLM_",
    "SOL_EXECBENCH_",
    "MLSYS26_",
)
_ENV_NAMES = {
    "CUDA_VISIBLE_DEVICES",
    "MLSYS26_FLASHINFER_CONTEST_ROOT",
    "PYTHONPATH",
    "SOLBENCH_TOKEN",
    "SOL_EXECBENCH",
}
_SENSITIVE_KEY = re.compile(
    r"(?:^|_)(?:API_?KEY|TOKEN|SECRET|PASSWORD|CREDENTIAL|AUTH)(?:$|_)",
    re.IGNORECASE,
)
_SENSITIVE_ASSIGNMENT = re.compile(
    r"^(?P<prefix>\s*['\"]?[A-Za-z0-9_.-]*"
    r"(?:api[_-]?key|token|secret|password|credential|auth)"
    r"[A-Za-z0-9_.-]*['\"]?\s*(?::|=)\s*).*$",
    re.IGNORECASE,
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _is_sensitive_key(name: str) -> bool:
    return bool(_SENSITIVE_KEY.search(name))


def _redact_value(name: str, value: str) -> str:
    return "<redacted>" if _is_sensitive_key(name) and value else value


def selected_environment(environ: Mapping[str, str] | None = None) -> dict[str, str]:
    source = os.environ if environ is None else environ
    selected = {
        name: _redact_value(name, str(value))
        for name, value in source.items()
        if name in _ENV_NAMES or name.startswith(_ENV_PREFIXES)
    }
    return dict(sorted(selected.items()))


def redact_text(text: str, environ: Mapping[str, str] | None = None) -> str:
    """Redact structured secret assignments and any exact in-memory secret values."""
    source = os.environ if environ is None else environ
    secret_values = sorted(
        {
            str(value)
            for name, value in source.items()
            if _is_sensitive_key(name) and value
        },
        key=len,
        reverse=True,
    )
    lines = []
    for line in text.splitlines(keepends=True):
        newline = "\n" if line.endswith("\n") else ""
        content = line[:-1] if newline else line
        match = _SENSITIVE_ASSIGNMENT.match(content)
        if match:
            content = match.group("prefix") + '"<redacted>"'
        else:
            for value in secret_values:
                content = content.replace(value, "<redacted>")
        lines.append(content + newline)
    return "".join(lines)


def redact_arguments(arguments: Sequence[str]) -> list[str]:
    redacted: list[str] = []
    redact_next = False
    for argument in arguments:
        if redact_next:
            redacted.append("<redacted>")
            redact_next = False
            continue
        if "=" in argument:
            name, value = argument.split("=", 1)
            normalized_name = name.lstrip("-").replace("-", "_")
            value = (
                "<redacted>" if _is_sensitive_key(normalized_name) and value else value
            )
            redacted.append(f"{name}={value}")
            continue
        redacted.append(argument)
        redact_next = _is_sensitive_key(argument.lstrip("-").replace("-", "_"))
    return redacted


def _atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def _sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def file_identity(path: str | Path) -> dict[str, Any]:
    candidate = Path(path).expanduser()
    identity: dict[str, Any] = {"path": str(candidate)}
    try:
        resolved = candidate.resolve(strict=True)
        content = resolved.read_bytes()
    except OSError as exc:
        identity.update({"exists": False, "error": f"{type(exc).__name__}: {exc}"})
        return identity
    identity.update(
        {
            "path": str(resolved),
            "exists": True,
            "size": len(content),
            "sha256": _sha256_bytes(content),
        }
    )
    return identity


def _git(path: Path, *arguments: str) -> str:
    base = path.parent if path.is_file() else path
    return subprocess.check_output(
        ["git", "-C", str(base), *arguments],
        text=True,
        stderr=subprocess.STDOUT,
        timeout=10,
    ).strip()


def repository_identity(path: str | Path) -> dict[str, Any]:
    candidate = Path(path).expanduser()
    result: dict[str, Any] = {"path": str(candidate)}
    try:
        root = Path(_git(candidate, "rev-parse", "--show-toplevel"))
        status = _git(root, "status", "--porcelain=v1", "--untracked-files=normal")
        tracked_status = _git(root, "status", "--porcelain=v1", "--untracked-files=no")
        result.update(
            {
                "root": str(root),
                "commit": _git(root, "rev-parse", "HEAD"),
                "branch": _git(root, "rev-parse", "--abbrev-ref", "HEAD"),
                "describe": _git(root, "describe", "--always", "--dirty", "--tags"),
                "dirty": bool(status),
                "tracked_dirty": bool(tracked_status),
                "untracked": any(line.startswith("??") for line in status.splitlines()),
            }
        )
    except (OSError, subprocess.SubprocessError) as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    return result


def _parse_named_paths(values: Sequence[str]) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"expected NAME=PATH, got {value!r}")
        name, path = value.split("=", 1)
        if not name or not path:
            raise ValueError(f"expected NAME=PATH, got {value!r}")
        parsed[name] = path
    return parsed


def capture_environment(path: str | Path) -> dict[str, Any]:
    payload = {"captured_at": _utc_now(), "environment": selected_environment()}
    _atomic_write(Path(path), json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return payload


def _load_initial_environment(path: str | Path | None) -> dict[str, str]:
    if not path:
        return {}
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    environment = payload.get("environment") if isinstance(payload, dict) else None
    if not isinstance(environment, dict):
        raise ValueError("initial environment snapshot has no environment object")
    return {str(key): str(value) for key, value in environment.items()}


def build_manifest(
    *,
    task: str,
    run_dir: str,
    workspace: str,
    arguments: Sequence[str],
    cli_keys: Sequence[str],
    initial_environment: Mapping[str, str],
    repositories: Mapping[str, str],
    source_files: Mapping[str, str],
    rendered_config: str,
    rendered_task: str,
    config_copy: str,
    task_copy: str,
) -> dict[str, Any]:
    resolved_environment = selected_environment()
    cli_key_set = set(cli_keys)
    environment_records = {
        name: {
            "value": value,
            "source": (
                "cli"
                if name in cli_key_set
                else "environment"
                if name in initial_environment
                else "runner_default_or_derived"
            ),
        }
        for name, value in resolved_environment.items()
    }

    config_text = redact_text(Path(rendered_config).read_text(encoding="utf-8"))
    task_text = redact_text(Path(rendered_task).read_text(encoding="utf-8"))
    _atomic_write(Path(config_copy), config_text)
    _atomic_write(Path(task_copy), task_text)

    fingerprint_payload = {
        "arguments": redact_arguments(arguments),
        "environment": resolved_environment,
        "resolved_config_sha256": _sha256_bytes(config_text.encode("utf-8")),
        "resolved_task_sha256": _sha256_bytes(task_text.encode("utf-8")),
    }
    fingerprint = _sha256_bytes(
        json.dumps(fingerprint_payload, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
    )
    return {
        "schema_version": 1,
        "created_at": _utc_now(),
        "task": task,
        "run_dir": str(Path(run_dir).resolve()),
        "workspace": str(Path(workspace).resolve()),
        "command": {
            "program": "orchestrator/run_sol58_pes.sh",
            "arguments": redact_arguments(arguments),
        },
        "configuration": {
            "fingerprint_sha256": fingerprint,
            "cli_override_keys": sorted(cli_key_set),
            "initial_environment": dict(sorted(initial_environment.items())),
            "resolved_environment": environment_records,
            "resolved_config": file_identity(config_copy),
            "resolved_task": file_identity(task_copy),
        },
        "repositories": {
            name: repository_identity(path)
            for name, path in sorted(repositories.items())
        },
        "sources": {
            name: file_identity(path) for name, path in sorted(source_files.items())
        },
    }


def write_manifest(args: argparse.Namespace) -> dict[str, Any]:
    manifest = build_manifest(
        task=args.task,
        run_dir=args.run_dir,
        workspace=args.workspace,
        arguments=args.runner_arg,
        cli_keys=args.cli_key,
        initial_environment=_load_initial_environment(args.initial_environment),
        repositories=_parse_named_paths(args.repository),
        source_files=_parse_named_paths(args.source),
        rendered_config=args.rendered_config,
        rendered_task=args.rendered_task,
        config_copy=args.config_copy,
        task_copy=args.task_copy,
    )
    content = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    _atomic_write(Path(args.output), content)
    if args.latest:
        _atomic_write(Path(args.latest), content)
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    capture = subparsers.add_parser("capture-env")
    capture.add_argument("--output", required=True)

    write = subparsers.add_parser("write")
    write.add_argument("--output", required=True)
    write.add_argument("--latest")
    write.add_argument("--task", required=True)
    write.add_argument("--run-dir", required=True)
    write.add_argument("--workspace", required=True)
    write.add_argument("--rendered-config", required=True)
    write.add_argument("--rendered-task", required=True)
    write.add_argument("--config-copy", required=True)
    write.add_argument("--task-copy", required=True)
    write.add_argument("--initial-environment")
    write.add_argument("--repository", action="append", default=[])
    write.add_argument("--source", action="append", default=[])
    write.add_argument("--cli-key", action="append", default=[])
    write.add_argument("--runner-arg", action="append", default=[])
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "capture-env":
        capture_environment(args.output)
    else:
        manifest = write_manifest(args)
        print(
            "[Atrex] Run manifest: "
            f"{args.output} config_sha256="
            f"{manifest['configuration']['fingerprint_sha256'][:12]}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
