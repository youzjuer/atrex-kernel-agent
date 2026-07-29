#!/usr/bin/env bash
set -euo pipefail

: "${LLM_API_KEY:?LLM_API_KEY not set. export LLM_API_KEY before running PES.}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
TASK_DIR="${REPO_ROOT}/orchestrator/sol58_pes"
ORIGINAL_ARGS=("$@")
INPUT_ENV_SNAPSHOT="$(mktemp --suffix=.json -t sol58_initial_env.XXXXXX)"
CLOCKS_LOCKED_BY_RUNNER=0
RENDERED_CONFIG=""
RENDERED_TASK=""
RUNTIME_ENV=""
RESOLVED_RUNTIME_CONFIG=""

is_truthy() {
  case "${1,,}" in
    1|true|yes|on) return 0 ;;
    *) return 1 ;;
  esac
}

cleanup() {
  rm -f "${INPUT_ENV_SNAPSHOT:-}" "${RENDERED_CONFIG:-}" "${RENDERED_TASK:-}" \
    "${RUNTIME_ENV:-}" "${RESOLVED_RUNTIME_CONFIG:-}"
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
    --runtime-config)
      if (($# < 2)); then
        echo "error: --runtime-config requires a JSON file" >&2
        exit 2
      fi
      SOL58_RUNTIME_CONFIG="$2"
      shift 2
      ;;
    --runtime-config=*)
      SOL58_RUNTIME_CONFIG="${1#*=}"
      shift
      ;;
    *)
      RUNNER_ARGS+=("$1")
      shift
      ;;
  esac
done

case "${SOL58_CODE_LANGUAGE:-}" in
  "") ;;
  cuda|cuda_cpp)
    SOL58_CODE_LANGUAGE="cuda_cpp"
    ;;
  cute|cutedsl|cute_dsl)
    SOL58_CODE_LANGUAGE="cute_dsl"
    ;;
  auto)
    SOL58_CODE_LANGUAGE="auto"
    ;;
  *)
    echo "error: unsupported SOL58 code language '${SOL58_CODE_LANGUAGE}'" >&2
    echo "       expected cuda_cpp, cute_dsl, or auto" >&2
    exit 2
    ;;
esac

export SOL58_RUNTIME_CONFIG="${SOL58_RUNTIME_CONFIG:-${TASK_DIR}/runtime_config.json}"
RUNTIME_ENV="$(mktemp --suffix=.env -t sol58_runtime.XXXXXX)"
RESOLVED_RUNTIME_CONFIG="$(mktemp --suffix=.json -t sol58_runtime_resolved.XXXXXX)"
python "${TASK_DIR}/runtime_config.py" \
  --config "${SOL58_RUNTIME_CONFIG}" \
  --repo-root "${REPO_ROOT}" \
  --task-dir "${TASK_DIR}" \
  --env-output "${RUNTIME_ENV}" \
  --json-output "${RESOLVED_RUNTIME_CONFIG}"
# The generated file contains only shell-quoted exports from the validated schema.
# shellcheck disable=SC1090
source "${RUNTIME_ENV}"

CONTEST_ROOT="${MLSYS26_FLASHINFER_CONTEST_ROOT}"
PROJECT_ROOT="${CONTEST_ROOT}/full-agent/moe/agent/loongflow"
RUN_DIR="${SOL58_PES_RUN_DIR}"
OFFICIAL_LOCAL_SOL_EXECBENCH="${SOL_EXECBENCH}"
export PYTHONPATH="${REPO_ROOT}:${PROJECT_ROOT}:${PROJECT_ROOT}/src:${PYTHONPATH:-}"

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

for required in task_spec.json task_config.yaml task_prompt.txt initial_kernel.cu eval_program_sol58.py seed_bank.json; do
  if [[ ! -f "${TASK_DIR}/${required}" ]]; then
    echo "error: missing SOL58 PES task file: ${TASK_DIR}/${required}" >&2
    exit 1
  fi
done

echo "[Atrex] Validating required LoongFlow compatibility patches..."
python -m orchestrator.loongflow_compat.bootstrap --validate-only

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
mkdir -p "${RUN_RECORD_DIR}"
cp "${RESOLVED_RUNTIME_CONFIG}" "${RUN_RECORD_DIR}/resolved_runtime_config.json"
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
  --source "runtime_config=${SOL58_RUNTIME_CONFIG}"
  --source "task_spec=${ATREX_TASK_SPEC_PATH}"
  --source "runtime_config_loader=${TASK_DIR}/runtime_config.py"
  --source "resolved_runtime_config=${RUN_RECORD_DIR}/resolved_runtime_config.json"
  --source "task_config=${TASK_DIR}/task_config.yaml"
  --source "task_prompt=${TASK_DIR}/task_prompt.txt"
  --source "knowledge_pack=${SOL58_PES_KNOWLEDGE_PACK}"
  --source "seed_manifest=${ATREX_PES_SEED_MANIFEST}"
  --source "initial_file=${INITIAL_FILE}"
  --source "evaluator=${TASK_DIR}/eval_program_sol58.py"
  --source "ncu_summary=${TASK_DIR}/ncu_summary.py"
  --source "run_manifest=${REPO_ROOT}/orchestrator/run_manifest.py"
  --source "compat_adapter=${SCRIPT_DIR}/loongflow_compat/compat_adapter.py"
  --source "compat_bootstrap=${SCRIPT_DIR}/loongflow_compat/bootstrap.py"
  --source "sol58_task_hooks=${SCRIPT_DIR}/loongflow_compat/sol58_task_hooks.py"
  --source "architecture_islands=${SCRIPT_DIR}/loongflow_compat/architecture_islands.py"
  --source "checkpoint_compat=${SCRIPT_DIR}/loongflow_compat/checkpoint_compat.py"
  --source "env_flags=${SCRIPT_DIR}/loongflow_compat/env_flags.py"
  --source "upstream_contract=${SCRIPT_DIR}/loongflow_compat/upstream_contract.py"
  --source "loongflow_contract=${SCRIPT_DIR}/loongflow_compat/loongflow_contract.json"
)
for argument in "${ORIGINAL_ARGS[@]}"; do
  RUN_MANIFEST_ARGS+=(--runner-arg="${argument}")
done
python "${REPO_ROOT}/orchestrator/run_manifest.py" "${RUN_MANIFEST_ARGS[@]}"

echo "[Atrex] Starting real LoongFlow PES for ${ATREX_TASK_NAME}"
echo "        task_spec=${ATREX_TASK_SPEC_PATH} leaderboard=${SOL58_LEADERBOARD_URL}"
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

python -m orchestrator.loongflow_compat.bootstrap \
  "${PROJECT_ROOT}/agents/math_agent/math_evolve_agent.py" \
  --config "${RENDERED_CONFIG}" \
  --task-file "${RENDERED_TASK}" \
  --initial-file "${INITIAL_FILE}" \
  --eval-file "${TASK_DIR}/eval_program_sol58.py" \
  --log-level INFO \
  "${RUNNER_ARGS[@]}"
