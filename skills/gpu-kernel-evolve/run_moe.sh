#!/usr/bin/env bash
set -euo pipefail

# Skill-level MoE PES entrypoint. Edit this block for the default run.
CUDA_VISIBLE_DEVICES_VALUE="${CUDA_VISIBLE_DEVICES_VALUE:-6}"
PROJ019_ROOT_VALUE="${PROJ019_ROOT_VALUE:-/home/youchunbo/code/sumu/omoExplore/proj/proj_019_moe_workload_op_opt}"

WORKSPACE_REL="${WORKSPACE_REL:-kernel_opt_fused_moe_cuda_pes}"
RUN_DIR_REL="${RUN_DIR_REL:-moe_run}"

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

SKILL_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SKILL_DIR/../.." && pwd)"
WORKSPACE_DIR="$REPO_ROOT/$WORKSPACE_REL"
RUN_DIR="${RUN_DIR:-$WORKSPACE_DIR/$RUN_DIR_REL}"

export CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES_VALUE"
export PROJ019_ROOT="$PROJ019_ROOT_VALUE"

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

echo "[Atrex] GPU Kernel Evolve skill MoE entry"
echo "[Atrex] Skill:     $SKILL_DIR"
echo "[Atrex] Workspace: $WORKSPACE_DIR"
echo "[Atrex] Run dir:   $RUN_DIR"
echo "[Atrex] Shape:     preset=$PRESET tokens=$TOKENS"
echo "[Atrex] Baseline:  flashinfer.trtllm_fp4_block_scale_moe"
echo "[Atrex] GPU:       CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

RUN_DIR="$RUN_DIR" "$WORKSPACE_DIR/run_moe.sh" "${args[@]}" "$@"
