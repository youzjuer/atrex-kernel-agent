#!/usr/bin/env bash
set -euo pipefail

# Edit this block to change the default MoE optimization run.
CUDA_VISIBLE_DEVICES_VALUE="${CUDA_VISIBLE_DEVICES_VALUE:-${CUDA_VISIBLE_DEVICES:-6}}"
PROJ019_ROOT_VALUE="${PROJ019_ROOT_VALUE:-${PROJ019_ROOT:-/home/youchunbo/code/sumu/omoExplore/proj/proj_019_moe_workload_op_opt}}"

PRESET="${PRESET:-g11}"
TOKENS="${TOKENS:-9500}"
WARMUP="${WARMUP:-1}"
REP="${REP:-3}"

GENERATIONS="${GENERATIONS:-1}"
N_CANDIDATES="${N_CANDIDATES:-1}"
FRESH_RUN="${FRESH_RUN:-0}"
RESUME_EXISTING="${RESUME_EXISTING:-1}"

COMPARE_FLASHINFER="${COMPARE_FLASHINFER:-1}"
REQUIRE_FLASHINFER="${REQUIRE_FLASHINFER:-1}"
FLASHINFER_TARGET_SPEEDUP="${FLASHINFER_TARGET_SPEEDUP:-1.0}"
FLASHINFER_TIMEOUT_S="${FLASHINFER_TIMEOUT_S:-120}"

PLANNER_BACKEND="${PLANNER_BACKEND:-auto}"
EXECUTOR_BACKEND="${EXECUTOR_BACKEND:-auto}"
PLANNER_CMD="${PLANNER_CMD:-}"
EXECUTOR_CMD="${EXECUTOR_CMD:-}"
CODEX_MODEL="${CODEX_MODEL:-}"
CODEX_TIMEOUT_S="${CODEX_TIMEOUT_S:-180}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
RUN_DIR="${RUN_DIR:-$SCRIPT_DIR/moe_run}"

export CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES_VALUE"
export PROJ019_ROOT="$PROJ019_ROOT_VALUE"
export PYTHONPATH="$SCRIPT_DIR:$REPO_ROOT:${PYTHONPATH:-}"

args=(
  --generations "$GENERATIONS"
  --n-candidates "$N_CANDIDATES"
  --preset "$PRESET"
  --tokens "$TOKENS"
  --warmup "$WARMUP"
  --rep "$REP"
  --flashinfer-target-speedup "$FLASHINFER_TARGET_SPEEDUP"
  --flashinfer-timeout-s "$FLASHINFER_TIMEOUT_S"
  --planner-backend "$PLANNER_BACKEND"
  --executor-backend "$EXECUTOR_BACKEND"
  --codex-timeout-s "$CODEX_TIMEOUT_S"
)

if [[ "$FRESH_RUN" == "1" ]]; then
  args+=(--fresh)
fi

if [[ "$RESUME_EXISTING" == "1" ]]; then
  args+=(--resume-existing)
else
  args+=(--no-resume-existing)
fi

if [[ "$COMPARE_FLASHINFER" == "1" ]]; then
  args+=(--compare-flashinfer)
else
  args+=(--no-compare-flashinfer)
fi

if [[ "$REQUIRE_FLASHINFER" == "1" ]]; then
  args+=(--require-flashinfer)
fi

if [[ -n "$PLANNER_CMD" ]]; then
  args+=(--planner-cmd "$PLANNER_CMD")
fi

if [[ -n "$EXECUTOR_CMD" ]]; then
  args+=(--executor-cmd "$EXECUTOR_CMD")
fi

if [[ -n "$CODEX_MODEL" ]]; then
  args+=(--codex-model "$CODEX_MODEL")
fi

echo "[Atrex] FlashInfer FP4 MoE PES plugin entry"
echo "[Atrex] Source:   $SCRIPT_DIR"
echo "[Atrex] Repo:     $REPO_ROOT"
echo "[Atrex] Run dir:  $RUN_DIR"
echo "[Atrex] Shape:    preset=$PRESET tokens=$TOKENS"
echo "[Atrex] Baseline: flashinfer.trtllm_fp4_block_scale_moe"
echo "[Atrex] GPU:      CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

python "$SCRIPT_DIR/auto_evolve.py" \
  --source "$SCRIPT_DIR" \
  --repo-root "$REPO_ROOT" \
  --run-dir "$RUN_DIR" \
  "${args[@]}" \
  "$@"
