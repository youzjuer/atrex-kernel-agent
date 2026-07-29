"""Declarative schema for the SOL-ExecBench PES runtime configuration."""

from __future__ import annotations

from dataclasses import dataclass


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
