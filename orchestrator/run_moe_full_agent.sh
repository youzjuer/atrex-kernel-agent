#!/usr/bin/env bash
set -euo pipefail

: "${LLM_API_KEY:?LLM_API_KEY not set. export LLM_API_KEY=sk-... before running.}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONTEST_ROOT="${MLSYS26_FLASHINFER_CONTEST_ROOT:-/home/youchunbo/code/mlsys26-flashinfer-contest}"
FULL_AGENT_DIR="${CONTEST_ROOT}/full-agent/moe"
RUNNER="${FULL_AGENT_DIR}/run_moe.sh"
PROJECT_ROOT="${FULL_AGENT_DIR}/agent/loongflow"

if [[ ! -f "$RUNNER" ]]; then
  echo "error: LoongFlow MoE runner not found: $RUNNER" >&2
  echo "set MLSYS26_FLASHINFER_CONTEST_ROOT to your mlsys26-flashinfer-contest checkout" >&2
  exit 1
fi

bash "${SCRIPT_DIR}/ensure_loongflow_compat.sh" "${PROJECT_ROOT}"

echo "[Atrex] Delegating full-agent MoE optimization to LoongFlow runner:"
echo "        $RUNNER"

exec bash "$RUNNER" "$@"
