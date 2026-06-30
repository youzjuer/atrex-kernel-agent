#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
RUN_DIR="${RUN_DIR:-$SCRIPT_DIR/moe_run}"

export PYTHONPATH="$SCRIPT_DIR:$REPO_ROOT:${PYTHONPATH:-}"

echo "[Atrex] Starting FlashInfer FP4 MoE PES task (CUDA)"
echo "[Atrex] Source: $SCRIPT_DIR"
echo "[Atrex] Run dir: $RUN_DIR"

python "$SCRIPT_DIR/auto_evolve.py" \
  --source "$SCRIPT_DIR" \
  --repo-root "$REPO_ROOT" \
  --run-dir "$RUN_DIR" \
  "$@"
