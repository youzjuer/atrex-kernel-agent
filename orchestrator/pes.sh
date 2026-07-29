#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat >&2 <<'USAGE'
Usage:
  bash orchestrator/pes.sh [moe|sol58] [--dry-run] [-- <runner args>]
  PES_TASK=sol58 bash orchestrator/pes.sh

Purpose:
  Atrex PES keyword entry point. This intentionally delegates to a real
  full-agent PES runner instead of running the experimental JSON-hook runner or
  a hand-written single-trajectory optimization loop.

Supported tasks:
  moe      MLSys26 FlashInfer MoE LoongFlow runner
  sol58    SOL-ExecBench kernel 58 LoongFlow runner

Required for real runs:
  LLM_API_KEY

Optional:
  MLSYS26_FLASHINFER_CONTEST_ROOT=/path/to/mlsys26-flashinfer-contest
  SOL58_PROBLEM_DIR=/path/to/058_moe_expert_token_radix_sort_with_prefix_sum
  SOL58_CODE_LANGUAGE=cuda_cpp|cute_dsl|auto
  SOL58_NCU_SUMMARY=1
  SOL58_CODE_LANGUAGE=auto bash orchestrator/pes.sh sol58
  SOL58_NCU_PROFILE_POLICY=local_best bash orchestrator/pes.sh sol58
USAGE
}

task="${PES_TASK:-moe}"
dry_run=0
runner_args=()

while (($#)); do
  case "$1" in
    -h|--help)
      usage
      exit 0
      ;;
    --dry-run)
      dry_run=1
      shift
      ;;
    --task)
      if (($# < 2)); then
        echo "error: --task requires a value" >&2
        exit 2
      fi
      task="$2"
      shift 2
      ;;
    --)
      shift
      runner_args+=("$@")
      break
      ;;
    moe|flashinfer-moe|mlsys26-moe)
      task="moe"
      shift
      ;;
    sol58|kernel58|sol-execbench-58|sol_execbench_58)
      task="sol58"
      shift
      ;;
    *)
      runner_args+=("$1")
      shift
      ;;
  esac
done

case "$task" in
  moe)
    runner="orchestrator/run_moe_full_agent.sh"
    ;;
  sol58)
    runner="orchestrator/run_sol58_pes.sh"
    ;;
  *)
    echo "error: unsupported PES task '$task'" >&2
    echo "Atrex will not fall back to the linear optimizer or experimental JSON-hook runner for a 'pes' request." >&2
    usage
    exit 2
    ;;
esac

if [[ ! -f "$runner" ]]; then
  echo "error: PES runner not found: $runner" >&2
  exit 1
fi

if ((dry_run)); then
  echo "[Atrex PES] task=$task"
  echo "[Atrex PES] runner=$runner"
  echo "[Atrex PES] runner_args=${runner_args[*]-}"
  exit 0
fi

: "${LLM_API_KEY:?LLM_API_KEY not set. export LLM_API_KEY=sk-... before running PES.}"

exec bash "$runner" "${runner_args[@]}"
