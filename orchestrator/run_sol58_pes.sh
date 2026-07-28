#!/usr/bin/env bash
set -euo pipefail

: "${LLM_API_KEY:?LLM_API_KEY not set. export LLM_API_KEY before running PES.}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
TASK_DIR="${REPO_ROOT}/orchestrator/sol58_pes"
ORIGINAL_ARGS=("$@")
CLI_OVERRIDE_KEYS=()
INPUT_ENV_SNAPSHOT="$(mktemp --suffix=.json -t sol58_initial_env.XXXXXX)"
CLOCKS_LOCKED_BY_RUNNER=0
RENDERED_CONFIG=""
RENDERED_TASK=""

is_truthy() {
  case "${1,,}" in
    1|true|yes|on) return 0 ;;
    *) return 1 ;;
  esac
}

cleanup() {
  rm -f "${INPUT_ENV_SNAPSHOT:-}" "${RENDERED_CONFIG:-}" "${RENDERED_TASK:-}"
  if ((CLOCKS_LOCKED_BY_RUNNER)) && is_truthy "${SOL58_UNLOCK_CLOCKS_ON_EXIT:-1}"; then
    sudo -n nvidia-smi -i "${SOL58_CLOCK_GPU_INDEX}" -rgc >/dev/null 2>&1 || true
    sudo -n nvidia-smi -i "${SOL58_CLOCK_GPU_INDEX}" -rmc >/dev/null 2>&1 || true
  fi
}

trap cleanup EXIT
trap 'exit 130' INT TERM

python "${REPO_ROOT}/orchestrator/run_manifest.py" capture-env \
  --output "${INPUT_ENV_SNAPSHOT}"

RUNNER_ARGS=()
while (($#)); do
  case "$1" in
    --code-language)
      if (($# < 2)); then
        echo "error: --code-language requires cuda_cpp, cute_dsl, or auto" >&2
        exit 2
      fi
      SOL58_CODE_LANGUAGE="$2"
      CLI_OVERRIDE_KEYS+=("SOL58_CODE_LANGUAGE")
      shift 2
      ;;
    --code-language=*)
      SOL58_CODE_LANGUAGE="${1#*=}"
      CLI_OVERRIDE_KEYS+=("SOL58_CODE_LANGUAGE")
      shift
      ;;
    --cutedsl-rate|--cute-dsl-rate)
      if (($# < 2)); then
        echo "error: --cutedsl-rate requires a value from 0 through 1" >&2
        exit 2
      fi
      SOL58_CUTEDSL_GENERATION_RATE="$2"
      CLI_OVERRIDE_KEYS+=("SOL58_CUTEDSL_GENERATION_RATE")
      shift 2
      ;;
    --cutedsl-rate=*|--cute-dsl-rate=*)
      SOL58_CUTEDSL_GENERATION_RATE="${1#*=}"
      CLI_OVERRIDE_KEYS+=("SOL58_CUTEDSL_GENERATION_RATE")
      shift
      ;;
    --cutedsl-period|--cute-dsl-period)
      if (($# < 2)); then
        echo "error: --cutedsl-period requires a positive integer" >&2
        exit 2
      fi
      SOL58_CUTEDSL_SCHEDULE_PERIOD="$2"
      CLI_OVERRIDE_KEYS+=("SOL58_CUTEDSL_SCHEDULE_PERIOD")
      shift 2
      ;;
    --cutedsl-period=*|--cute-dsl-period=*)
      SOL58_CUTEDSL_SCHEDULE_PERIOD="${1#*=}"
      CLI_OVERRIDE_KEYS+=("SOL58_CUTEDSL_SCHEDULE_PERIOD")
      shift
      ;;
    --measurement-profile)
      if (($# < 2)); then
        echo "error: --measurement-profile requires official_v1_1_b200 or native" >&2
        exit 2
      fi
      SOL58_MEASUREMENT_PROFILE="$2"
      CLI_OVERRIDE_KEYS+=("SOL58_MEASUREMENT_PROFILE")
      shift 2
      ;;
    --measurement-profile=*)
      SOL58_MEASUREMENT_PROFILE="${1#*=}"
      CLI_OVERRIDE_KEYS+=("SOL58_MEASUREMENT_PROFILE")
      shift
      ;;
    --clock-gpu-index)
      if (($# < 2)); then
        echo "error: --clock-gpu-index requires a physical nvidia-smi GPU index" >&2
        exit 2
      fi
      SOL58_CLOCK_GPU_INDEX="$2"
      CLI_OVERRIDE_KEYS+=("SOL58_CLOCK_GPU_INDEX")
      shift 2
      ;;
    --clock-gpu-index=*)
      SOL58_CLOCK_GPU_INDEX="${1#*=}"
      CLI_OVERRIDE_KEYS+=("SOL58_CLOCK_GPU_INDEX")
      shift
      ;;
    --ncu-summary)
      SOL58_NCU_SUMMARY=1
      CLI_OVERRIDE_KEYS+=("SOL58_NCU_SUMMARY")
      shift
      ;;
    --no-ncu-summary)
      SOL58_NCU_SUMMARY=0
      CLI_OVERRIDE_KEYS+=("SOL58_NCU_SUMMARY")
      shift
      ;;
    --ncu-policy)
      if (($# < 2)); then
        echo "error: --ncu-policy requires all_correct, local_best, or periodic" >&2
        exit 2
      fi
      SOL58_NCU_PROFILE_POLICY="$2"
      CLI_OVERRIDE_KEYS+=("SOL58_NCU_PROFILE_POLICY")
      shift 2
      ;;
    --ncu-policy=*)
      SOL58_NCU_PROFILE_POLICY="${1#*=}"
      CLI_OVERRIDE_KEYS+=("SOL58_NCU_PROFILE_POLICY")
      shift
      ;;
    --ncu-timeout)
      if (($# < 2)); then
        echo "error: --ncu-timeout requires seconds" >&2
        exit 2
      fi
      SOL58_NCU_TIMEOUT="$2"
      CLI_OVERRIDE_KEYS+=("SOL58_NCU_TIMEOUT")
      shift 2
      ;;
    --ncu-timeout=*)
      SOL58_NCU_TIMEOUT="${1#*=}"
      CLI_OVERRIDE_KEYS+=("SOL58_NCU_TIMEOUT")
      shift
      ;;
    --ncu-workload)
      if (($# < 2)); then
        echo "error: --ncu-workload requires slowest, an index, or a workload UUID" >&2
        exit 2
      fi
      SOL58_NCU_WORKLOAD="$2"
      CLI_OVERRIDE_KEYS+=("SOL58_NCU_WORKLOAD")
      shift 2
      ;;
    --ncu-workload=*)
      SOL58_NCU_WORKLOAD="${1#*=}"
      CLI_OVERRIDE_KEYS+=("SOL58_NCU_WORKLOAD")
      shift
      ;;
    *)
      RUNNER_ARGS+=("$1")
      shift
      ;;
  esac
done

case "${SOL58_CODE_LANGUAGE:-cuda_cpp}" in
  cuda|cuda_cpp)
    export SOL58_CODE_LANGUAGE="cuda_cpp"
    ;;
  cute|cutedsl|cute_dsl)
    export SOL58_CODE_LANGUAGE="cute_dsl"
    ;;
  auto)
    export SOL58_CODE_LANGUAGE="auto"
    ;;
  *)
    echo "error: unsupported SOL58 code language '${SOL58_CODE_LANGUAGE}'" >&2
    echo "       expected cuda_cpp, cute_dsl, or auto" >&2
    exit 2
    ;;
esac

export SOL58_CUTEDSL_GENERATION_RATE="${SOL58_CUTEDSL_GENERATION_RATE:-0.5}"
export SOL58_CUTEDSL_SCHEDULE_PERIOD="${SOL58_CUTEDSL_SCHEDULE_PERIOD:-10}"
python - "${SOL58_CUTEDSL_GENERATION_RATE}" "${SOL58_CUTEDSL_SCHEDULE_PERIOD}" <<'PY'
import sys

try:
    rate = float(sys.argv[1])
    period = int(sys.argv[2])
except ValueError as exc:
    raise SystemExit(f"error: invalid CuTeDSL schedule: {exc}")
if not 0.0 <= rate <= 1.0:
    raise SystemExit("error: --cutedsl-rate must be between 0 and 1")
if period <= 0:
    raise SystemExit("error: --cutedsl-period must be a positive integer")
PY

CONTEST_ROOT="${MLSYS26_FLASHINFER_CONTEST_ROOT:-/home/youchunbo/code/mlsys26-flashinfer-contest}"
PROJECT_ROOT="${CONTEST_ROOT}/full-agent/moe/agent/loongflow"
RUN_DIR="$(realpath -m "${SOL58_PES_RUN_DIR:-/tmp/sol58_pes_run}")"

export PYTHONPATH="${REPO_ROOT}:${SCRIPT_DIR}/loongflow_compat:${PROJECT_ROOT}:${PROJECT_ROOT}/src:${PYTHONPATH:-}"
export ATREX_LITELLM_DROP_PARAMS="${ATREX_LITELLM_DROP_PARAMS:-1}"
export SOL58_PROBLEM_DIR="${SOL58_PROBLEM_DIR:-/home/youchunbo/code/sol-problems/058_moe_expert_token_radix_sort_with_prefix_sum}"
export SOL58_EVAL_ROOT="${SOL58_EVAL_ROOT:-/tmp/sol58_pes_eval}"
export SOL58_TARGET_LATENCY_MS="${SOL58_TARGET_LATENCY_MS:-0.006797}"
export SOL58_TARGET_SCORE="${SOL58_TARGET_SCORE:-0.904135}"
export SOL58_LOCAL_REPEAT_COUNT="${SOL58_LOCAL_REPEAT_COUNT:-3}"
export SOL58_LOCAL_BEST_GATE="${SOL58_LOCAL_BEST_GATE:-1}"
export SOL58_LOCAL_EVAL_CACHE="${SOL58_LOCAL_EVAL_CACHE:-1}"
export SOL58_LOCAL_EVAL_CONTRACT_VERSION="${SOL58_LOCAL_EVAL_CONTRACT_VERSION:-sol58-v3}"
export SOL58_LOCAL_GATE_SIGMA_MULTIPLIER="${SOL58_LOCAL_GATE_SIGMA_MULTIPLIER:-2.0}"
export SOL58_LOCAL_GATE_RELATIVE_NOISE_FLOOR="${SOL58_LOCAL_GATE_RELATIVE_NOISE_FLOOR:-0.01}"
export SOL58_LOCAL_GATE_RECHECK_PAIRS="${SOL58_LOCAL_GATE_RECHECK_PAIRS:-2}"
export SOL58_LOCAL_GATE_UNCERTAIN_RELATIVE_TOLERANCE="${SOL58_LOCAL_GATE_UNCERTAIN_RELATIVE_TOLERANCE:-0.005}"
export SOL58_LOCAL_GATE_ALLOW_UNCERTAIN_OFFICIAL="${SOL58_LOCAL_GATE_ALLOW_UNCERTAIN_OFFICIAL:-1}"
export SOL58_LOCAL_GATE_CHALLENGER_COOLDOWN_S="${SOL58_LOCAL_GATE_CHALLENGER_COOLDOWN_S:-900}"
export SOL58_OFFICIAL_FITNESS="${SOL58_OFFICIAL_FITNESS:-1}"
export SOL58_OFFICIAL_EVAL_STACK_VERSION="${SOL58_OFFICIAL_EVAL_STACK_VERSION:-v1.1}"
export SOL58_OFFICIAL_SUBMISSION_MODE="${SOL58_OFFICIAL_SUBMISSION_MODE:-private}"
export SOL58_OFFICIAL_POLL_TIMEOUT="${SOL58_OFFICIAL_POLL_TIMEOUT:-180}"
export SOL58_OFFICIAL_POLL_INTERVAL="${SOL58_OFFICIAL_POLL_INTERVAL:-10}"
export SOL58_OFFICIAL_REQUEST_TIMEOUT="${SOL58_OFFICIAL_REQUEST_TIMEOUT:-10}"
export SOL58_OFFICIAL_ASYNC_SUBMIT="${SOL58_OFFICIAL_ASYNC_SUBMIT:-1}"
export SOL58_OFFICIAL_ASYNC_REFRESH_DELAY="${SOL58_OFFICIAL_ASYNC_REFRESH_DELAY:-60}"
export SOL58_OFFICIAL_PENDING_RESULT_GRACE="${SOL58_OFFICIAL_PENDING_RESULT_GRACE:-60}"
export SOL58_OFFICIAL_CACHE_REFRESH_TIMEOUT="${SOL58_OFFICIAL_CACHE_REFRESH_TIMEOUT:-0}"
export SOL58_OFFICIAL_REFRESH_BATCH_SIZE="${SOL58_OFFICIAL_REFRESH_BATCH_SIZE:-2}"
export SOL58_OFFICIAL_REFRESH_TIME_BUDGET="${SOL58_OFFICIAL_REFRESH_TIME_BUDGET:-10}"
export SOL58_CALIBRATION_HALF_LIFE_HOURS="${SOL58_CALIBRATION_HALF_LIFE_HOURS:-336}"
export SOL58_OFFICIAL_PENDING_SCORE_POLICY="${SOL58_OFFICIAL_PENDING_SCORE_POLICY:-local_proxy}"
DEFAULT_PROVISIONAL_SCORE_CAP="$(awk -v target="${SOL58_TARGET_SCORE}" 'BEGIN { printf "%.6f", target - 0.005 }')"
export SOL58_OFFICIAL_PROVISIONAL_SCORE_CAP="${SOL58_OFFICIAL_PROVISIONAL_SCORE_CAP:-${DEFAULT_PROVISIONAL_SCORE_CAP}}"

export SOL58_PES_WORKSPACE="$(realpath -m "${SOL58_PES_WORKSPACE:-${RUN_DIR}/output}")"
export SOL58_MAX_ITERATIONS="${SOL58_MAX_ITERATIONS:-40}"
export SOL58_PES_CONCURRENCY="${SOL58_PES_CONCURRENCY:-1}"
export SOL58_NUM_ISLANDS="${SOL58_NUM_ISLANDS:-8}"
export SOL58_ARCHITECTURE_MIGRATION_INTERVAL="${SOL58_ARCHITECTURE_MIGRATION_INTERVAL:-20}"
export SOL58_PES_KNOWLEDGE_GROUNDING="${SOL58_PES_KNOWLEDGE_GROUNDING:-1}"
export SOL58_PES_KNOWLEDGE_PACK="${SOL58_PES_KNOWLEDGE_PACK:-${TASK_DIR}/knowledge_pack.md}"
export SOL58_REACT_SCORE_THRESHOLD="${SOL58_REACT_SCORE_THRESHOLD:-0.84}"
export SOL58_SEED_LOCAL_BEST="${SOL58_SEED_LOCAL_BEST:-1}"
export SOL58_EVAL_TIMEOUT="${SOL58_EVAL_TIMEOUT:-1200}"
export SOL58_COMPILE_TIMEOUT="${SOL58_COMPILE_TIMEOUT:-180}"
export SOL58_SOL_TIMEOUT="${SOL58_SOL_TIMEOUT:-120}"
export SOL58_NCU_SUMMARY="${SOL58_NCU_SUMMARY:-1}"
export SOL58_NCU_PROFILE_POLICY="${SOL58_NCU_PROFILE_POLICY:-all_correct}"
export SOL58_NCU_PROFILE_INTERVAL="${SOL58_NCU_PROFILE_INTERVAL:-5}"
export SOL58_NCU_TIMEOUT="${SOL58_NCU_TIMEOUT:-180}"
export SOL58_NCU_PARSE_TIMEOUT="${SOL58_NCU_PARSE_TIMEOUT:-60}"
export SOL58_NCU_WORKLOAD="${SOL58_NCU_WORKLOAD:-slowest}"
export SOL58_NCU_SET="${SOL58_NCU_SET:-full}"
export SOL58_NCU_LAUNCH_COUNT="${SOL58_NCU_LAUNCH_COUNT:-1}"
export SOL58_NCU_CACHE_DIR="${SOL58_NCU_CACHE_DIR:-${SOL58_EVAL_ROOT}/ncu_cache}"

case "${SOL58_NCU_PROFILE_POLICY}" in
  all_correct|local_best|improving|periodic) ;;
  *)
    echo "error: SOL58_NCU_PROFILE_POLICY must be all_correct, local_best, or periodic" >&2
    exit 2
    ;;
esac
if [[ ! "${SOL58_NCU_PROFILE_INTERVAL}" =~ ^[1-9][0-9]*$ ]]; then
  echo "error: SOL58_NCU_PROFILE_INTERVAL must be a positive integer" >&2
  exit 2
fi
if ! awk -v value="${SOL58_NCU_TIMEOUT}" 'BEGIN { exit !(value ~ /^[0-9]+([.][0-9]+)?$/ && value + 0 > 0) }'; then
  echo "error: SOL58_NCU_TIMEOUT must be a positive number" >&2
  exit 2
fi

export SOL58_MEASUREMENT_PROFILE="${SOL58_MEASUREMENT_PROFILE:-official_v1_1_b200}"
case "${SOL58_MEASUREMENT_PROFILE}" in
  official|official_v1_1|official_v1_1_b200)
    export SOL58_MEASUREMENT_PROFILE="official_v1_1_b200"
    export SOL58_CLOCK_GPU_INDEX="${SOL58_CLOCK_GPU_INDEX:-0}"
    export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-${SOL58_CLOCK_GPU_INDEX}}"
    export SOL58_LOCK_CLOCKS="${SOL58_LOCK_CLOCKS:-1}"
    export SOL58_AUTO_RELOCK_CLOCKS="${SOL58_AUTO_RELOCK_CLOCKS:-1}"
    export SOL58_REQUIRE_EXCLUSIVE_GPU="${SOL58_REQUIRE_EXCLUSIVE_GPU:-1}"
    export SOL58_UNLOCK_CLOCKS_ON_EXIT="${SOL58_UNLOCK_CLOCKS_ON_EXIT:-1}"
    export SOL_EXECBENCH_GPU_CLK_MHZ="${SOL_EXECBENCH_GPU_CLK_MHZ:-1500}"
    export SOL_EXECBENCH_DRAM_CLK_MHZ="${SOL_EXECBENCH_DRAM_CLK_MHZ:-3996}"
    export SOL58_CUDA_GENCODE="${SOL58_CUDA_GENCODE:--gencode=arch=compute_100,code=sm_100}"
    export SOL58_WARMUP_RUNS="${SOL58_WARMUP_RUNS:-10}"
    export SOL58_ITERATIONS="${SOL58_ITERATIONS:-50}"
    export SOL58_SEED="${SOL58_SEED:-200}"
    export SOL58_LOCAL_EVAL_STACK_ID="${SOL58_LOCAL_EVAL_STACK_ID:-sol-execbench-v1.1-sm100-proxy}"
    export SOL58_RECALIBRATE_LOCAL_BEST="${SOL58_RECALIBRATE_LOCAL_BEST:-1}"
    ;;
  native)
    export SOL58_LOCK_CLOCKS="${SOL58_LOCK_CLOCKS:-0}"
    export SOL58_AUTO_RELOCK_CLOCKS="${SOL58_AUTO_RELOCK_CLOCKS:-0}"
    export SOL58_LOCAL_EVAL_STACK_ID="${SOL58_LOCAL_EVAL_STACK_ID:-native}"
    export SOL58_RECALIBRATE_LOCAL_BEST="${SOL58_RECALIBRATE_LOCAL_BEST:-0}"
    ;;
  *)
    echo "error: unsupported measurement profile '${SOL58_MEASUREMENT_PROFILE}'" >&2
    echo "       expected official_v1_1_b200 or native" >&2
    exit 2
    ;;
esac

OFFICIAL_LOCAL_SOL_EXECBENCH="${SOL58_OFFICIAL_LOCAL_SOL_EXECBENCH:-/home/youchunbo/code/sol-execbench/.venv/bin/sol-execbench}"
if [[ -z "${SOL_EXECBENCH:-}" && -x "${OFFICIAL_LOCAL_SOL_EXECBENCH}" ]]; then
  export SOL_EXECBENCH="${OFFICIAL_LOCAL_SOL_EXECBENCH}"
fi
export LLM_BASE_URL="${LLM_BASE_URL:-https://api.chatanywhere.tech/v1}"
export LLM_MODEL="${LLM_MODEL:-openai/claude-opus-4-7}"
export LLM_PROVIDER="${LLM_PROVIDER:-openai}"
export LLM_TEMPERATURE="${LLM_TEMPERATURE:-0.9}"
export LLM_CONTEXT_LENGTH="${LLM_CONTEXT_LENGTH:-1000000}"
export LLM_MAX_TOKENS="${LLM_MAX_TOKENS:-32768}"
export LLM_TIMEOUT="${LLM_TIMEOUT:-240}"
export ATREX_PES_MAX_PARALLEL_CANDIDATES="${ATREX_PES_MAX_PARALLEL_CANDIDATES:-1}"
export ATREX_PES_SOURCE_DEDUP="${ATREX_PES_SOURCE_DEDUP:-1}"
export ATREX_PES_ARCHITECTURE_ISLANDS="${ATREX_PES_ARCHITECTURE_ISLANDS:-1}"
export ATREX_PES_STAGNATION_SEEDS="${ATREX_PES_STAGNATION_SEEDS:-1}"
export ATREX_PES_STAGNATION_ARCHITECTURE_ROUNDS="${ATREX_PES_STAGNATION_ARCHITECTURE_ROUNDS:-12}"
export ATREX_PES_STAGNATION_SEED_INTERVAL="${ATREX_PES_STAGNATION_SEED_INTERVAL:-20}"
export ATREX_PES_STAGNATION_MAX_ATTEMPTS="${ATREX_PES_STAGNATION_MAX_ATTEMPTS:-2}"
export ATREX_PES_SEED_MANIFEST="${ATREX_PES_SEED_MANIFEST:-${TASK_DIR}/seed_bank.json}"
export ATREX_PES_COMPACT_DB_TOOLS="${ATREX_PES_COMPACT_DB_TOOLS:-1}"
export ATREX_PES_DB_SOLUTION_CHARS="${ATREX_PES_DB_SOLUTION_CHARS:-65536}"
export ATREX_PES_DB_SUMMARY_CHARS="${ATREX_PES_DB_SUMMARY_CHARS:-16384}"
export ATREX_PES_DB_EVALUATION_CHARS="${ATREX_PES_DB_EVALUATION_CHARS:-16384}"

if [[ ! "${SOL58_NUM_ISLANDS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "error: SOL58_NUM_ISLANDS must be a positive integer" >&2
  exit 2
fi
if [[ ! "${SOL58_ARCHITECTURE_MIGRATION_INTERVAL}" =~ ^[1-9][0-9]*$ ]]; then
  echo "error: SOL58_ARCHITECTURE_MIGRATION_INTERVAL must be a positive integer" >&2
  exit 2
fi
if is_truthy "${ATREX_PES_ARCHITECTURE_ISLANDS}" && ((SOL58_NUM_ISLANDS < 8)); then
  echo "error: architecture-aware PES requires at least 8 islands; got ${SOL58_NUM_ISLANDS}" >&2
  exit 2
fi
for integer_setting in ATREX_PES_STAGNATION_ARCHITECTURE_ROUNDS ATREX_PES_STAGNATION_SEED_INTERVAL ATREX_PES_STAGNATION_MAX_ATTEMPTS; do
  if [[ ! "${!integer_setting}" =~ ^[1-9][0-9]*$ ]]; then
    echo "error: ${integer_setting} must be a positive integer" >&2
    exit 2
  fi
done
if is_truthy "${SOL58_PES_KNOWLEDGE_GROUNDING}" && [[ ! -f "${SOL58_PES_KNOWLEDGE_PACK}" ]]; then
  echo "error: SOL58 knowledge pack not found: ${SOL58_PES_KNOWLEDGE_PACK}" >&2
  exit 1
fi

prepare_measurement_environment() {
  if ! is_truthy "${SOL58_LOCK_CLOCKS}"; then
    export SOL_EXECBENCH_CLOCKS_LOCKED=0
    return
  fi

  if [[ ! "${SOL58_CLOCK_GPU_INDEX}" =~ ^[0-9]+$ ]]; then
    echo "error: SOL58_CLOCK_GPU_INDEX must be a physical numeric nvidia-smi index" >&2
    return 1
  fi
  if ! sudo -n true >/dev/null 2>&1; then
    echo "error: official-like measurement requires passwordless sudo for nvidia-smi" >&2
    return 1
  fi

  local gpu_uuid busy
  gpu_uuid="$(nvidia-smi -i "${SOL58_CLOCK_GPU_INDEX}" --query-gpu=uuid --format=csv,noheader,nounits | xargs)"
  export SOL58_MEASUREMENT_DEVICE_ID="${gpu_uuid}"
  if is_truthy "${SOL58_REQUIRE_EXCLUSIVE_GPU:-0}"; then
    busy="$(nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name --format=csv,noheader,nounits 2>/dev/null \
      | awk -F, -v uuid="${gpu_uuid}" '$1 == uuid {print}' || true)"
    if [[ -n "${busy}" ]]; then
      echo "error: GPU ${SOL58_CLOCK_GPU_INDEX} (${gpu_uuid}) already has a compute process:" >&2
      echo "${busy}" >&2
      return 1
    fi
  fi

  sudo -n nvidia-smi -i "${SOL58_CLOCK_GPU_INDEX}" -lgc "${SOL_EXECBENCH_GPU_CLK_MHZ}" >/dev/null
  sudo -n nvidia-smi -i "${SOL58_CLOCK_GPU_INDEX}" -lmc "${SOL_EXECBENCH_DRAM_CLK_MHZ}" >/dev/null
  sleep "${SOL58_CLOCK_STABILIZE_SECONDS:-2}"

  local observed
  observed="$(nvidia-smi -i "${SOL58_CLOCK_GPU_INDEX}" \
    --query-gpu=clocks.current.sm,clocks.current.memory --format=csv,noheader,nounits \
    | tr -d ' ')"
  if [[ "${observed}" != "${SOL_EXECBENCH_GPU_CLK_MHZ},${SOL_EXECBENCH_DRAM_CLK_MHZ}" ]]; then
    echo "error: clock verification failed on GPU ${SOL58_CLOCK_GPU_INDEX}: ${observed}" >&2
    return 1
  fi
  export SOL_EXECBENCH_CLOCKS_LOCKED=1
  CLOCKS_LOCKED_BY_RUNNER=1
}

INITIAL_FILE="${SOL58_INITIAL_FILE:-}"
LOCAL_BEST_RECORD="${SOL58_EVAL_ROOT}/official_cache/local_best.json"
LOCAL_BEST_KERNEL=""
if [[ -f "${LOCAL_BEST_RECORD}" ]]; then
  LOCAL_BEST_KERNEL="$(python - "${LOCAL_BEST_RECORD}" "${SOL58_CODE_LANGUAGE}" <<'PY'
import json
import sys
from pathlib import Path

record_path = Path(sys.argv[1])
mode = sys.argv[2]
try:
    record = json.loads(record_path.read_text(encoding="utf-8"))
except Exception:
    raise SystemExit(0)

kernel_path = Path(str(record.get("kernel_path") or ""))
language = str(record.get("source_language") or "").strip().lower()
if not language:
    language = "cute_dsl" if kernel_path.suffix == ".py" else "cuda_cpp"
if kernel_path.is_file() and (mode == "auto" or mode == language):
    print(kernel_path)
PY
)"
fi
if [[ -z "${INITIAL_FILE}" ]] && is_truthy "${SOL58_SEED_LOCAL_BEST}" && [[ -f "${LOCAL_BEST_KERNEL}" ]]; then
  INITIAL_FILE="${LOCAL_BEST_KERNEL}"
fi
INITIAL_FILE="${INITIAL_FILE:-${TASK_DIR}/initial_kernel.cu}"

if [[ ! -f "${INITIAL_FILE}" ]]; then
  echo "error: SOL58 initial kernel not found: ${INITIAL_FILE}" >&2
  exit 1
fi

if is_truthy "${SOL58_OFFICIAL_FITNESS}"; then
  if [[ -z "${SOLBENCH_TOKEN:-}" && -z "${SOL58_SOLBENCH_TOKEN:-}" ]]; then
    echo "error: SOL58_OFFICIAL_FITNESS=1 requires SOLBENCH_TOKEN or SOL58_SOLBENCH_TOKEN." >&2
    echo "       Get it from the SOL-ExecBench browser localStorage key 'solbench_token'." >&2
    exit 1
  fi
fi

if [[ ! -f "${PROJECT_ROOT}/agents/math_agent/math_evolve_agent.py" ]]; then
  echo "error: LoongFlow math_evolve_agent.py not found under ${PROJECT_ROOT}" >&2
  echo "set MLSYS26_FLASHINFER_CONTEST_ROOT to your mlsys26-flashinfer-contest checkout" >&2
  exit 1
fi

bash "${SCRIPT_DIR}/ensure_loongflow_compat.sh" "${PROJECT_ROOT}"
python "${SCRIPT_DIR}/loongflow_compat/upstream_contract.py" \
  --project-root "${PROJECT_ROOT}" \
  --contract "${SCRIPT_DIR}/loongflow_compat/loongflow_contract.json"

for required in task_config.yaml task_prompt.txt initial_kernel.cu eval_program_sol58.py seed_bank.json; do
  if [[ ! -f "${TASK_DIR}/${required}" ]]; then
    echo "error: missing SOL58 PES task file: ${TASK_DIR}/${required}" >&2
    exit 1
  fi
done

echo "[Atrex] Validating required LoongFlow compatibility patches..."
python - <<'PY'
import json
import sitecustomize

manifest = sitecustomize.validate_patch_manifest()
print("[Atrex] Patch manifest: " + json.dumps(manifest, sort_keys=True))
PY

mkdir -p "${RUN_DIR}" "${SOL58_PES_WORKSPACE}" "${SOL58_EVAL_ROOT}"
cd "${RUN_DIR}" || exit 1

if [[ "${SOL58_SKIP_LLM_PREFLIGHT:-0}" != "1" ]]; then
  echo "[Atrex] Checking LLM endpoint before SOL evaluator..."
  python - <<'PY'
import os
import re
import sys

from litellm import completion

try:
    completion(
        model=os.environ["LLM_MODEL"],
        custom_llm_provider=os.environ.get("LLM_PROVIDER", "openai"),
        api_key=os.environ["LLM_API_KEY"],
        api_base=os.environ["LLM_BASE_URL"],
        messages=[{"role": "user", "content": "Reply OK."}],
        max_tokens=4,
        temperature=float(os.environ.get("LLM_TEMPERATURE", "1")),
        timeout=min(60, int(os.environ.get("LLM_TIMEOUT", "60"))),
    )
except Exception as exc:
    api_key = os.environ.get("LLM_API_KEY", "")
    detail = str(exc)
    if api_key:
        detail = detail.replace(api_key, "<redacted>")
    detail = re.sub(r"([A-Za-z0-9]{4,})\*+([A-Za-z0-9]{4,})", r"\1****\2", detail)
    print(
        "error: LLM preflight failed. Check LLM_API_KEY, LLM_BASE_URL, and LLM_MODEL before running SOL58 PES.",
        file=sys.stderr,
    )
    print(detail[:1200], file=sys.stderr)
    raise SystemExit(1)
PY
fi

echo "[Atrex] Preparing SOL58 measurement profile ${SOL58_MEASUREMENT_PROFILE}..."
prepare_measurement_environment

if is_truthy "${SOL58_RECALIBRATE_LOCAL_BEST}" && [[ -f "${LOCAL_BEST_KERNEL}" ]]; then
  echo "[Atrex] Verifying local-best baseline under ${SOL58_MEASUREMENT_PROFILE}..."
  SOL58_CODE_LANGUAGE=auto python - "${LOCAL_BEST_KERNEL}" <<'PY'
import sys

from orchestrator.sol58_pes.eval_program_sol58 import evaluate

result = evaluate(sys.argv[1])
print(f"[Atrex] Local-best calibration: {result['summary']}")
if result["status"] != "success":
    raise SystemExit(
        f"local-best calibration failed with status={result['status']}"
    )
PY
fi

RENDERED_CONFIG="$(mktemp --suffix=.yaml -t sol58_task_config.XXXXXX)"
RENDERED_TASK="$(mktemp --suffix=.txt -t sol58_task_prompt.XXXXXX)"
envsubst < "${TASK_DIR}/task_config.yaml" > "${RENDERED_CONFIG}"
envsubst '${SOL58_CODE_LANGUAGE} ${SOL58_CUTEDSL_GENERATION_RATE} ${SOL58_CUTEDSL_SCHEDULE_PERIOD} ${SOL58_NUM_ISLANDS} ${SOL58_ARCHITECTURE_MIGRATION_INTERVAL}' \
  < "${TASK_DIR}/task_prompt.txt" > "${RENDERED_TASK}"
KNOWLEDGE_FINGERPRINT="disabled"
if is_truthy "${SOL58_PES_KNOWLEDGE_GROUNDING}"; then
  python - "${RENDERED_TASK}" "${SOL58_PES_KNOWLEDGE_PACK}" <<'PY'
import sys
from pathlib import Path

task_path = Path(sys.argv[1])
knowledge_path = Path(sys.argv[2])
task = task_path.read_text(encoding="utf-8").rstrip()
knowledge = knowledge_path.read_text(encoding="utf-8").strip()
task_path.write_text(
    task + "\n\n--- BEGIN INJECTED SOL58 KNOWLEDGE PACK ---\n\n"
    + knowledge
    + "\n\n--- END INJECTED SOL58 KNOWLEDGE PACK ---\n",
    encoding="utf-8",
)
PY
  KNOWLEDGE_FINGERPRINT="$(sha256sum "${SOL58_PES_KNOWLEDGE_PACK}" | awk '{print substr($1, 1, 12)}')"
fi

RUN_INSTANCE_ID="$(date -u +%Y%m%dT%H%M%SZ)-$$"
RUN_RECORD_DIR="${RUN_DIR}/run_manifests/${RUN_INSTANCE_ID}"
RUN_MANIFEST_ARGS=(
  write
  --output "${RUN_RECORD_DIR}/manifest.json"
  --latest "${RUN_DIR}/run_manifest.json"
  --task sol58
  --run-dir "${RUN_DIR}"
  --workspace "${SOL58_PES_WORKSPACE}"
  --rendered-config "${RENDERED_CONFIG}"
  --rendered-task "${RENDERED_TASK}"
  --config-copy "${RUN_RECORD_DIR}/resolved_task_config.yaml"
  --task-copy "${RUN_RECORD_DIR}/resolved_task_prompt.txt"
  --initial-environment "${INPUT_ENV_SNAPSHOT}"
  --repository "atrex=${REPO_ROOT}"
  --repository "loongflow=${PROJECT_ROOT}"
  --repository "sol_execbench=${OFFICIAL_LOCAL_SOL_EXECBENCH}"
  --repository "problem=${SOL58_PROBLEM_DIR}"
  --source "runner=${SCRIPT_DIR}/run_sol58_pes.sh"
  --source "task_config=${TASK_DIR}/task_config.yaml"
  --source "task_prompt=${TASK_DIR}/task_prompt.txt"
  --source "knowledge_pack=${SOL58_PES_KNOWLEDGE_PACK}"
  --source "seed_manifest=${ATREX_PES_SEED_MANIFEST}"
  --source "initial_file=${INITIAL_FILE}"
  --source "evaluator=${TASK_DIR}/eval_program_sol58.py"
  --source "ncu_summary=${TASK_DIR}/ncu_summary.py"
  --source "run_manifest=${REPO_ROOT}/orchestrator/run_manifest.py"
  --source "sitecustomize=${SCRIPT_DIR}/loongflow_compat/sitecustomize.py"
  --source "sol58_task_hooks=${SCRIPT_DIR}/loongflow_compat/sol58_task_hooks.py"
  --source "architecture_islands=${SCRIPT_DIR}/loongflow_compat/architecture_islands.py"
  --source "checkpoint_compat=${SCRIPT_DIR}/loongflow_compat/checkpoint_compat.py"
  --source "env_flags=${SCRIPT_DIR}/loongflow_compat/env_flags.py"
  --source "upstream_contract=${SCRIPT_DIR}/loongflow_compat/upstream_contract.py"
  --source "loongflow_contract=${SCRIPT_DIR}/loongflow_compat/loongflow_contract.json"
)
for key in "${CLI_OVERRIDE_KEYS[@]}"; do
  RUN_MANIFEST_ARGS+=(--cli-key "${key}")
done
for argument in "${ORIGINAL_ARGS[@]}"; do
  RUN_MANIFEST_ARGS+=(--runner-arg="${argument}")
done
python "${REPO_ROOT}/orchestrator/run_manifest.py" "${RUN_MANIFEST_ARGS[@]}"

echo "[Atrex] Starting real LoongFlow PES for SOL-ExecBench kernel 58"
echo "        target_latency_ms=${SOL58_TARGET_LATENCY_MS}"
echo "        target_score=${SOL58_TARGET_SCORE}"
echo "        code_language=${SOL58_CODE_LANGUAGE}"
echo "        cutedsl_rate=${SOL58_CUTEDSL_GENERATION_RATE} period=${SOL58_CUTEDSL_SCHEDULE_PERIOD}"
echo "        architecture_islands=${SOL58_NUM_ISLANDS} exchange_interval=${SOL58_ARCHITECTURE_MIGRATION_INTERVAL}"
echo "        knowledge_grounding=${SOL58_PES_KNOWLEDGE_GROUNDING} knowledge_sha256=${KNOWLEDGE_FINGERPRINT}"
echo "        stagnation_seeds=${ATREX_PES_STAGNATION_SEEDS} threshold=${ATREX_PES_STAGNATION_ARCHITECTURE_ROUNDS} interval=${ATREX_PES_STAGNATION_SEED_INTERVAL} max_attempts=${ATREX_PES_STAGNATION_MAX_ATTEMPTS}"
echo "        seed_manifest=${ATREX_PES_SEED_MANIFEST}"
echo "        local_repeat_count=${SOL58_LOCAL_REPEAT_COUNT} local_best_gate=${SOL58_LOCAL_BEST_GATE} local_cache=${SOL58_LOCAL_EVAL_CACHE}"
echo "        local_gate=sigma:${SOL58_LOCAL_GATE_SIGMA_MULTIPLIER} noise_floor:${SOL58_LOCAL_GATE_RELATIVE_NOISE_FLOOR} recheck_pairs:${SOL58_LOCAL_GATE_RECHECK_PAIRS} uncertain_tolerance:${SOL58_LOCAL_GATE_UNCERTAIN_RELATIVE_TOLERANCE} cooldown:${SOL58_LOCAL_GATE_CHALLENGER_COOLDOWN_S}s"
echo "        measurement_profile=${SOL58_MEASUREMENT_PROFILE} gpu=${CUDA_VISIBLE_DEVICES:-auto} physical_clock_gpu=${SOL58_CLOCK_GPU_INDEX:-n/a}"
echo "        clocks=${SOL_EXECBENCH_GPU_CLK_MHZ:-auto}/${SOL_EXECBENCH_DRAM_CLK_MHZ:-auto}MHz lock=${SOL58_LOCK_CLOCKS} cuda_gencode=${SOL58_CUDA_GENCODE:-runtime-native}"
echo "        local_eval_stack=${SOL58_LOCAL_EVAL_STACK_ID} sol_execbench=${SOL_EXECBENCH:-sol-execbench}"
echo "        ncu_summary=${SOL58_NCU_SUMMARY} policy=${SOL58_NCU_PROFILE_POLICY} workload=${SOL58_NCU_WORKLOAD} timeout=${SOL58_NCU_TIMEOUT}s set=${SOL58_NCU_SET}"
echo "        initial_file=${INITIAL_FILE} react_score_threshold=${SOL58_REACT_SCORE_THRESHOLD}"
echo "        official_fitness=${SOL58_OFFICIAL_FITNESS} stack=${SOL58_OFFICIAL_EVAL_STACK_VERSION} mode=${SOL58_OFFICIAL_SUBMISSION_MODE}"
echo "        official_async_submit=${SOL58_OFFICIAL_ASYNC_SUBMIT} refresh_delay=${SOL58_OFFICIAL_ASYNC_REFRESH_DELAY}s request_timeout=${SOL58_OFFICIAL_REQUEST_TIMEOUT}s"
echo "        official_refresh_batch=${SOL58_OFFICIAL_REFRESH_BATCH_SIZE} refresh_budget=${SOL58_OFFICIAL_REFRESH_TIME_BUDGET}s calibration_half_life=${SOL58_CALIBRATION_HALF_LIFE_HOURS}h"
echo "        official_poll_timeout=${SOL58_OFFICIAL_POLL_TIMEOUT}s pending_policy=${SOL58_OFFICIAL_PENDING_SCORE_POLICY} provisional_cap=${SOL58_OFFICIAL_PROVISIONAL_SCORE_CAP}"
echo "        run_dir=${RUN_DIR}"
echo "        workspace=${SOL58_PES_WORKSPACE}"
echo "        eval_root=${SOL58_EVAL_ROOT}"
echo "        problem_dir=${SOL58_PROBLEM_DIR}"
echo "        compile_timeout=${SOL58_COMPILE_TIMEOUT}s run_timeout=${SOL58_SOL_TIMEOUT}s evaluator_timeout=${SOL58_EVAL_TIMEOUT}s"

python "${PROJECT_ROOT}/agents/math_agent/math_evolve_agent.py" \
  --config "${RENDERED_CONFIG}" \
  --task-file "${RENDERED_TASK}" \
  --initial-file "${INITIAL_FILE}" \
  --eval-file "${TASK_DIR}/eval_program_sol58.py" \
  --log-level INFO \
  "${RUNNER_ARGS[@]}"
