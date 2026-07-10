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

export SOL58_PES_WORKSPACE="${SOL58_PES_WORKSPACE:-${RUN_DIR}/output}"
export SOL58_MAX_ITERATIONS="${SOL58_MAX_ITERATIONS:-40}"
export SOL58_PES_CONCURRENCY="${SOL58_PES_CONCURRENCY:-1}"
export SOL58_NUM_ISLANDS="${SOL58_NUM_ISLANDS:-1}"
export SOL58_EVAL_TIMEOUT="${SOL58_EVAL_TIMEOUT:-1200}"
export SOL58_COMPILE_TIMEOUT="${SOL58_COMPILE_TIMEOUT:-240}"
export SOL58_SOL_TIMEOUT="${SOL58_SOL_TIMEOUT:-900}"

export LLM_BASE_URL="${LLM_BASE_URL:-https://api.chatanywhere.tech/v1}"
export LLM_MODEL="${LLM_MODEL:-openai/claude-opus-4-7}"
export LLM_TEMPERATURE="${LLM_TEMPERATURE:-0.9}"
export LLM_CONTEXT_LENGTH="${LLM_CONTEXT_LENGTH:-128000}"
export LLM_MAX_TOKENS="${LLM_MAX_TOKENS:-32768}"
export LLM_TIMEOUT="${LLM_TIMEOUT:-1200}"
export ATREX_PES_MAX_PARALLEL_CANDIDATES="${ATREX_PES_MAX_PARALLEL_CANDIDATES:-1}"

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
echo "        run_dir=${RUN_DIR}"
echo "        workspace=${SOL58_PES_WORKSPACE}"
echo "        eval_root=${SOL58_EVAL_ROOT}"
echo "        problem_dir=${SOL58_PROBLEM_DIR}"

python "${PROJECT_ROOT}/agents/math_agent/math_evolve_agent.py" \
  --config "${RENDERED_CONFIG}" \
  --task-file "${TASK_DIR}/task_prompt.txt" \
  --initial-file "${TASK_DIR}/initial_kernel.cu" \
  --eval-file "${TASK_DIR}/eval_program_sol58.py" \
  --log-level INFO \
  "$@"
