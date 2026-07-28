#!/usr/bin/env python3
"""Resolve and validate the single SOL58 runtime configuration."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


class RuntimeConfigError(ValueError):
    """Raised when the checked-in config or an environment override is invalid."""


@dataclass(frozen=True)
class Setting:
    path: str
    env: str
    kind: str
    minimum: float | None = None
    maximum: float | None = None
    choices: tuple[str, ...] = ()


def _setting(
    path: str,
    env: str,
    kind: str,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
    choices: tuple[str, ...] = (),
) -> Setting:
    return Setting(path, env, kind, minimum, maximum, choices)


SETTINGS = (
    _setting(
        "search.code_language",
        "SOL58_CODE_LANGUAGE",
        "str",
        choices=("cuda_cpp", "cute_dsl", "auto"),
    ),
    _setting(
        "search.cutedsl_generation_rate",
        "SOL58_CUTEDSL_GENERATION_RATE",
        "float",
        minimum=0,
        maximum=1,
    ),
    _setting(
        "search.cutedsl_schedule_period",
        "SOL58_CUTEDSL_SCHEDULE_PERIOD",
        "int",
        minimum=1,
    ),
    _setting("search.max_iterations", "SOL58_MAX_ITERATIONS", "int", minimum=1),
    _setting("search.concurrency", "SOL58_PES_CONCURRENCY", "int", minimum=1),
    _setting("search.num_islands", "SOL58_NUM_ISLANDS", "int", minimum=1),
    _setting(
        "search.migration_interval",
        "SOL58_ARCHITECTURE_MIGRATION_INTERVAL",
        "int",
        minimum=1,
    ),
    _setting("search.knowledge_grounding", "SOL58_PES_KNOWLEDGE_GROUNDING", "bool"),
    _setting(
        "search.react_score_threshold",
        "SOL58_REACT_SCORE_THRESHOLD",
        "float",
        minimum=0,
    ),
    _setting("search.seed_local_best", "SOL58_SEED_LOCAL_BEST", "bool"),
    _setting(
        "local_evaluation.target_latency_ms",
        "SOL58_TARGET_LATENCY_MS",
        "float",
        minimum=0,
    ),
    _setting(
        "local_evaluation.repeat_count", "SOL58_LOCAL_REPEAT_COUNT", "int", minimum=1
    ),
    _setting("local_evaluation.best_gate", "SOL58_LOCAL_BEST_GATE", "bool"),
    _setting("local_evaluation.cache", "SOL58_LOCAL_EVAL_CACHE", "bool"),
    _setting(
        "local_evaluation.contract_version", "SOL58_LOCAL_EVAL_CONTRACT_VERSION", "str"
    ),
    _setting(
        "local_evaluation.compile_timeout_seconds",
        "SOL58_COMPILE_TIMEOUT",
        "int",
        minimum=1,
    ),
    _setting(
        "local_evaluation.run_timeout_seconds",
        "SOL58_SOL_TIMEOUT",
        "int",
        minimum=1,
    ),
    _setting(
        "local_evaluation.evaluator_timeout_seconds",
        "SOL58_EVAL_TIMEOUT",
        "int",
        minimum=1,
    ),
    _setting(
        "local_evaluation.gate.sigma_multiplier",
        "SOL58_LOCAL_GATE_SIGMA_MULTIPLIER",
        "float",
        minimum=0,
    ),
    _setting(
        "local_evaluation.gate.relative_noise_floor",
        "SOL58_LOCAL_GATE_RELATIVE_NOISE_FLOOR",
        "float",
        minimum=0,
    ),
    _setting(
        "local_evaluation.gate.recheck_pairs",
        "SOL58_LOCAL_GATE_RECHECK_PAIRS",
        "int",
        minimum=0,
    ),
    _setting(
        "local_evaluation.gate.uncertain_relative_tolerance",
        "SOL58_LOCAL_GATE_UNCERTAIN_RELATIVE_TOLERANCE",
        "float",
        minimum=0,
    ),
    _setting(
        "local_evaluation.gate.allow_uncertain_official",
        "SOL58_LOCAL_GATE_ALLOW_UNCERTAIN_OFFICIAL",
        "bool",
    ),
    _setting(
        "local_evaluation.gate.challenger_cooldown_seconds",
        "SOL58_LOCAL_GATE_CHALLENGER_COOLDOWN_S",
        "float",
        minimum=0,
    ),
    _setting("official.enabled", "SOL58_OFFICIAL_FITNESS", "bool"),
    _setting("official.base_url", "SOL58_OFFICIAL_BASE_URL", "str"),
    _setting("official.kernel_id", "SOL58_OFFICIAL_KERNEL_ID", "int", minimum=1),
    _setting("official.gpu_type", "SOL58_OFFICIAL_GPU_TYPE", "str"),
    _setting("official.target_score", "SOL58_TARGET_SCORE", "float", minimum=0),
    _setting(
        "official.evaluation_stack_version",
        "SOL58_OFFICIAL_EVAL_STACK_VERSION",
        "str",
    ),
    _setting(
        "official.submission_mode",
        "SOL58_OFFICIAL_SUBMISSION_MODE",
        "str",
        choices=("private", "public"),
    ),
    _setting("official.cache", "SOL58_OFFICIAL_CACHE", "bool"),
    _setting(
        "official.minimum_local_score",
        "SOL58_OFFICIAL_MIN_LOCAL_SCORE",
        "float",
        minimum=0,
    ),
    _setting(
        "official.poll_timeout_seconds",
        "SOL58_OFFICIAL_POLL_TIMEOUT",
        "float",
        minimum=0,
    ),
    _setting(
        "official.poll_interval_seconds",
        "SOL58_OFFICIAL_POLL_INTERVAL",
        "float",
        minimum=0.01,
    ),
    _setting(
        "official.request_timeout_seconds",
        "SOL58_OFFICIAL_REQUEST_TIMEOUT",
        "float",
        minimum=0.01,
    ),
    _setting("official.async_submit", "SOL58_OFFICIAL_ASYNC_SUBMIT", "bool"),
    _setting(
        "official.async_refresh_delay_seconds",
        "SOL58_OFFICIAL_ASYNC_REFRESH_DELAY",
        "float",
        minimum=0,
    ),
    _setting(
        "official.pending_result_grace_seconds",
        "SOL58_OFFICIAL_PENDING_RESULT_GRACE",
        "float",
        minimum=0,
    ),
    _setting(
        "official.cache_refresh_timeout_seconds",
        "SOL58_OFFICIAL_CACHE_REFRESH_TIMEOUT",
        "float",
        minimum=0,
    ),
    _setting(
        "official.refresh_batch_size",
        "SOL58_OFFICIAL_REFRESH_BATCH_SIZE",
        "int",
        minimum=1,
    ),
    _setting(
        "official.refresh_time_budget_seconds",
        "SOL58_OFFICIAL_REFRESH_TIME_BUDGET",
        "float",
        minimum=0,
    ),
    _setting(
        "official.calibration_half_life_hours",
        "SOL58_CALIBRATION_HALF_LIFE_HOURS",
        "float",
        minimum=0.01,
    ),
    _setting(
        "official.pending_score_policy",
        "SOL58_OFFICIAL_PENDING_SCORE_POLICY",
        "str",
        choices=("local_proxy", "provisional"),
    ),
    _setting("official.probe.enabled", "SOL58_OFFICIAL_PROBE_ENABLED", "bool"),
    _setting(
        "official.probe.interval", "SOL58_OFFICIAL_PROBE_INTERVAL", "int", minimum=1
    ),
    _setting(
        "official.probe.cooldown_seconds",
        "SOL58_OFFICIAL_PROBE_COOLDOWN_S",
        "float",
        minimum=0,
    ),
    _setting(
        "official.probe.max_per_day",
        "SOL58_OFFICIAL_PROBE_MAX_PER_DAY",
        "int",
        minimum=0,
    ),
    _setting(
        "official.probe.max_relative_regression",
        "SOL58_OFFICIAL_PROBE_MAX_RELATIVE_REGRESSION",
        "float",
        minimum=0,
    ),
    _setting(
        "official.probe.architecture_only",
        "SOL58_OFFICIAL_PROBE_ARCHITECTURE_ONLY",
        "bool",
    ),
    _setting(
        "measurement.profile",
        "SOL58_MEASUREMENT_PROFILE",
        "str",
        choices=("official_v1_1_b200", "native"),
    ),
    _setting("measurement.clock_gpu_index", "SOL58_CLOCK_GPU_INDEX", "int", minimum=0),
    _setting(
        "measurement.clock_tolerance_mhz", "SOL58_CLOCK_TOLERANCE_MHZ", "int", minimum=0
    ),
    _setting(
        "measurement.clock_monitor_interval_seconds",
        "SOL58_CLOCK_MONITOR_INTERVAL_SECONDS",
        "float",
        minimum=0.01,
    ),
    _setting(
        "measurement.clock_stabilize_seconds",
        "SOL58_CLOCK_STABILIZE_SECONDS",
        "float",
        minimum=0,
    ),
    _setting("ncu.summary", "SOL58_NCU_SUMMARY", "bool"),
    _setting(
        "ncu.profile_policy",
        "SOL58_NCU_PROFILE_POLICY",
        "str",
        choices=("all_correct", "local_best", "improving", "periodic"),
    ),
    _setting("ncu.profile_interval", "SOL58_NCU_PROFILE_INTERVAL", "int", minimum=1),
    _setting("ncu.timeout_seconds", "SOL58_NCU_TIMEOUT", "float", minimum=0.01),
    _setting(
        "ncu.parse_timeout_seconds", "SOL58_NCU_PARSE_TIMEOUT", "float", minimum=0.01
    ),
    _setting("ncu.workload", "SOL58_NCU_WORKLOAD", "str"),
    _setting("ncu.set", "SOL58_NCU_SET", "str"),
    _setting("ncu.launch_count", "SOL58_NCU_LAUNCH_COUNT", "int", minimum=1),
    _setting("llm.base_url", "LLM_BASE_URL", "str"),
    _setting("llm.model", "LLM_MODEL", "str"),
    _setting("llm.provider", "LLM_PROVIDER", "str"),
    _setting("llm.temperature", "LLM_TEMPERATURE", "float", minimum=0),
    _setting("llm.context_length", "LLM_CONTEXT_LENGTH", "int", minimum=1),
    _setting("llm.max_tokens", "LLM_MAX_TOKENS", "int", minimum=1),
    _setting("llm.timeout_seconds", "LLM_TIMEOUT", "int", minimum=1),
    _setting("llm.drop_unsupported_params", "ATREX_LITELLM_DROP_PARAMS", "bool"),
    _setting(
        "compatibility.max_parallel_candidates",
        "ATREX_PES_MAX_PARALLEL_CANDIDATES",
        "int",
        minimum=1,
    ),
    _setting("compatibility.source_dedup", "ATREX_PES_SOURCE_DEDUP", "bool"),
    _setting(
        "compatibility.architecture_islands", "ATREX_PES_ARCHITECTURE_ISLANDS", "bool"
    ),
    _setting("compatibility.stagnation_seeds", "ATREX_PES_STAGNATION_SEEDS", "bool"),
    _setting(
        "compatibility.stagnation_rounds",
        "ATREX_PES_STAGNATION_ARCHITECTURE_ROUNDS",
        "int",
        minimum=1,
    ),
    _setting(
        "compatibility.stagnation_seed_interval",
        "ATREX_PES_STAGNATION_SEED_INTERVAL",
        "int",
        minimum=1,
    ),
    _setting(
        "compatibility.stagnation_max_attempts",
        "ATREX_PES_STAGNATION_MAX_ATTEMPTS",
        "int",
        minimum=1,
    ),
    _setting(
        "compatibility.compact_database_tools", "ATREX_PES_COMPACT_DB_TOOLS", "bool"
    ),
    _setting(
        "compatibility.database_solution_chars",
        "ATREX_PES_DB_SOLUTION_CHARS",
        "int",
        minimum=1,
    ),
    _setting(
        "compatibility.database_summary_chars",
        "ATREX_PES_DB_SUMMARY_CHARS",
        "int",
        minimum=1,
    ),
    _setting(
        "compatibility.database_evaluation_chars",
        "ATREX_PES_DB_EVALUATION_CHARS",
        "int",
        minimum=1,
    ),
)

PROFILE_SETTINGS = (
    _setting("lock_clocks", "SOL58_LOCK_CLOCKS", "bool"),
    _setting("auto_relock_clocks", "SOL58_AUTO_RELOCK_CLOCKS", "bool"),
    _setting("require_exclusive_gpu", "SOL58_REQUIRE_EXCLUSIVE_GPU", "bool"),
    _setting("unlock_clocks_on_exit", "SOL58_UNLOCK_CLOCKS_ON_EXIT", "bool"),
    _setting("gpu_clock_mhz", "SOL_EXECBENCH_GPU_CLK_MHZ", "int", minimum=0),
    _setting("dram_clock_mhz", "SOL_EXECBENCH_DRAM_CLK_MHZ", "int", minimum=0),
    _setting("cuda_gencode", "SOL58_CUDA_GENCODE", "str"),
    _setting("warmup_runs", "SOL58_WARMUP_RUNS", "int", minimum=0),
    _setting("iterations", "SOL58_ITERATIONS", "int", minimum=1),
    _setting("seed", "SOL58_SEED", "int", minimum=0),
    _setting("local_eval_stack_id", "SOL58_LOCAL_EVAL_STACK_ID", "str"),
    _setting("recalibrate_local_best", "SOL58_RECALIBRATE_LOCAL_BEST", "bool"),
)

PATH_ENV = {
    "contest_root": "MLSYS26_FLASHINFER_CONTEST_ROOT",
    "problem_dir": "SOL58_PROBLEM_DIR",
    "sol_execbench": "SOL_EXECBENCH",
    "run_dir": "SOL58_PES_RUN_DIR",
    "workspace": "SOL58_PES_WORKSPACE",
    "eval_root": "SOL58_EVAL_ROOT",
    "knowledge_pack": "SOL58_PES_KNOWLEDGE_PACK",
    "seed_manifest": "ATREX_PES_SEED_MANIFEST",
    "ncu_cache_dir": "SOL58_NCU_CACHE_DIR",
}

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
    name: str, repo_root: Path, task_dir: Path, resolved: Mapping[str, str]
) -> str:
    sibling_root = repo_root.parent
    if name == "contest_root":
        return str((sibling_root / "mlsys26-flashinfer-contest").resolve(strict=False))
    if name == "problem_dir":
        return str(
            (
                sibling_root
                / "sol-problems"
                / "058_moe_expert_token_radix_sort_with_prefix_sum"
            ).resolve(strict=False)
        )
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
            values[name] = _auto_path(name, repo_root, task_dir, values)
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
    expected = {"schema_version", "official.provisional_floor_delta"}
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
        config["paths"], source_env, Path(repo_root).resolve(), Path(task_dir).resolve()
    )
    for name, env_name in PATH_ENV.items():
        values[env_name] = paths[name]
        records[env_name] = {
            "value": paths[name],
            "source": path_sources[name],
            "path": f"paths.{name}",
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

    return {
        "schema_version": 1,
        "config_path": str(config_file),
        "profile": profile_name,
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
    write_resolved_config(result, args.env_output, args.json_output)
    print(
        "[Atrex] Resolved SOL58 config: "
        f"profile={result['profile']} settings={len(result['values'])}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
