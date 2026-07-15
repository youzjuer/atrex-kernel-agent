#!/usr/bin/env bash
set -euo pipefail

: "${LLM_API_KEY:?LLM_API_KEY not set. export LLM_API_KEY before running PES.}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
TASK_DIR="${REPO_ROOT}/orchestrator/sol58_pes"

CONTEST_ROOT="${MLSYS26_FLASHINFER_CONTEST_ROOT:-/home/youchunbo/code/mlsys26-flashinfer-contest}"
PROJECT_ROOT="${CONTEST_ROOT}/full-agent/moe/agent/loongflow"
RUN_DIR="${SOL58_PES_RUN_DIR:-/tmp/sol58_pes_run}"

export PYTHONPATH="${SCRIPT_DIR}/loongflow_compat:${PROJECT_ROOT}:${PROJECT_ROOT}/src:${PYTHONPATH:-}"
export ATREX_LITELLM_DROP_PARAMS="${ATREX_LITELLM_DROP_PARAMS:-1}"
export SOL58_PROBLEM_DIR="${SOL58_PROBLEM_DIR:-/home/youchunbo/code/sol-problems/058_moe_expert_token_radix_sort_with_prefix_sum}"
export SOL58_EVAL_ROOT="${SOL58_EVAL_ROOT:-/tmp/sol58_pes_eval}"
export SOL58_TARGET_LATENCY_MS="${SOL58_TARGET_LATENCY_MS:-0.006797}"
export SOL58_TARGET_SCORE="${SOL58_TARGET_SCORE:-0.904135}"
export SOL58_LOCAL_REPEAT_COUNT="${SOL58_LOCAL_REPEAT_COUNT:-3}"
export SOL58_LOCAL_BEST_GATE="${SOL58_LOCAL_BEST_GATE:-1}"
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
export SOL58_OFFICIAL_PENDING_SCORE_POLICY="${SOL58_OFFICIAL_PENDING_SCORE_POLICY:-local_proxy}"
export SOL58_OFFICIAL_PROVISIONAL_SCORE_CAP="${SOL58_OFFICIAL_PROVISIONAL_SCORE_CAP:-0.899135}"

export SOL58_PES_WORKSPACE="${SOL58_PES_WORKSPACE:-${RUN_DIR}/output}"
export SOL58_MAX_ITERATIONS="${SOL58_MAX_ITERATIONS:-40}"
export SOL58_PES_CONCURRENCY="${SOL58_PES_CONCURRENCY:-1}"
export SOL58_NUM_ISLANDS="${SOL58_NUM_ISLANDS:-1}"
export SOL58_REACT_SCORE_THRESHOLD="${SOL58_REACT_SCORE_THRESHOLD:-0.84}"
export SOL58_SEED_LOCAL_BEST="${SOL58_SEED_LOCAL_BEST:-1}"
export SOL58_EVAL_TIMEOUT="${SOL58_EVAL_TIMEOUT:-600}"
export SOL58_COMPILE_TIMEOUT="${SOL58_COMPILE_TIMEOUT:-180}"
export SOL58_SOL_TIMEOUT="${SOL58_SOL_TIMEOUT:-120}"

export LLM_BASE_URL="${LLM_BASE_URL:-https://api.chatanywhere.tech/v1}"
export LLM_MODEL="${LLM_MODEL:-openai/claude-opus-4-7}"
export LLM_PROVIDER="${LLM_PROVIDER:-openai}"
export LLM_TEMPERATURE="${LLM_TEMPERATURE:-0.9}"
export LLM_CONTEXT_LENGTH="${LLM_CONTEXT_LENGTH:-128000}"
export LLM_MAX_TOKENS="${LLM_MAX_TOKENS:-32768}"
export LLM_TIMEOUT="${LLM_TIMEOUT:-240}"
export ATREX_PES_MAX_PARALLEL_CANDIDATES="${ATREX_PES_MAX_PARALLEL_CANDIDATES:-1}"

is_truthy() {
  case "${1}" in
    1|true|TRUE|yes|YES|on|ON) return 0 ;;
    *) return 1 ;;
  esac
}

INITIAL_FILE="${SOL58_INITIAL_FILE:-}"
LOCAL_BEST_KERNEL="${SOL58_EVAL_ROOT}/official_cache/local_best_kernel.cu"
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

for required in task_config.yaml task_prompt.txt initial_kernel.cu eval_program_sol58.py; do
  if [[ ! -f "${TASK_DIR}/${required}" ]]; then
    echo "error: missing SOL58 PES task file: ${TASK_DIR}/${required}" >&2
    exit 1
  fi
done

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

RENDERED_CONFIG="$(mktemp -t sol58_task_config.XXXXXX).yaml"
trap 'rm -f "${RENDERED_CONFIG}"' EXIT
envsubst < "${TASK_DIR}/task_config.yaml" > "${RENDERED_CONFIG}"

echo "[Atrex] Starting real LoongFlow PES for SOL-ExecBench kernel 58"
echo "        target_latency_ms=${SOL58_TARGET_LATENCY_MS}"
echo "        target_score=${SOL58_TARGET_SCORE}"
echo "        local_repeat_count=${SOL58_LOCAL_REPEAT_COUNT} local_best_gate=${SOL58_LOCAL_BEST_GATE}"
echo "        initial_file=${INITIAL_FILE} react_score_threshold=${SOL58_REACT_SCORE_THRESHOLD}"
echo "        official_fitness=${SOL58_OFFICIAL_FITNESS} stack=${SOL58_OFFICIAL_EVAL_STACK_VERSION} mode=${SOL58_OFFICIAL_SUBMISSION_MODE}"
echo "        official_async_submit=${SOL58_OFFICIAL_ASYNC_SUBMIT} refresh_delay=${SOL58_OFFICIAL_ASYNC_REFRESH_DELAY}s request_timeout=${SOL58_OFFICIAL_REQUEST_TIMEOUT}s"
echo "        official_poll_timeout=${SOL58_OFFICIAL_POLL_TIMEOUT}s pending_policy=${SOL58_OFFICIAL_PENDING_SCORE_POLICY} provisional_cap=${SOL58_OFFICIAL_PROVISIONAL_SCORE_CAP}"
echo "        run_dir=${RUN_DIR}"
echo "        workspace=${SOL58_PES_WORKSPACE}"
echo "        eval_root=${SOL58_EVAL_ROOT}"
echo "        problem_dir=${SOL58_PROBLEM_DIR}"
echo "        compile_timeout=${SOL58_COMPILE_TIMEOUT}s run_timeout=${SOL58_SOL_TIMEOUT}s evaluator_timeout=${SOL58_EVAL_TIMEOUT}s"

python "${PROJECT_ROOT}/agents/math_agent/math_evolve_agent.py" \
  --config "${RENDERED_CONFIG}" \
  --task-file "${TASK_DIR}/task_prompt.txt" \
  --initial-file "${INITIAL_FILE}" \
  --eval-file "${TASK_DIR}/eval_program_sol58.py" \
  --log-level INFO \
  "$@"
