#!/usr/bin/env bash
set -euo pipefail

: "${LLM_API_KEY:?LLM_API_KEY not set. export LLM_API_KEY=sk-... before running.}"

CONTEST_ROOT="${MLSYS26_FLASHINFER_CONTEST_ROOT:-/home/youchunbo/code/mlsys26-flashinfer-contest}"
FULL_AGENT_DIR="${CONTEST_ROOT}/full-agent/moe"
RUNNER="${FULL_AGENT_DIR}/run_moe.sh"

if [[ ! -f "$RUNNER" ]]; then
  echo "error: LoongFlow MoE runner not found: $RUNNER" >&2
  echo "set MLSYS26_FLASHINFER_CONTEST_ROOT to your mlsys26-flashinfer-contest checkout" >&2
  exit 1
fi

echo "[Atrex] Delegating full-agent MoE optimization to LoongFlow runner:"
echo "        $RUNNER"

exec bash "$RUNNER" "$@"
