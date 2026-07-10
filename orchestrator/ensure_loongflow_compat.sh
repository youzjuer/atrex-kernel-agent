#!/usr/bin/env bash
set -euo pipefail

project_root="${1:?usage: ensure_loongflow_compat.sh /path/to/loongflow}"
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

source_file="${script_dir}/loongflow_compat/llm_switch.py"
target_file="${project_root}/agents/math_agent/llm_switch.py"

if [[ ! -f "${source_file}" ]]; then
  echo "error: missing compat file: ${source_file}" >&2
  exit 1
fi

if [[ ! -f "${target_file}" ]]; then
  mkdir -p "$(dirname "${target_file}")"
  install -m 0644 "${source_file}" "${target_file}"
  echo "[Atrex] Installed LoongFlow compatibility module: ${target_file}"
fi
