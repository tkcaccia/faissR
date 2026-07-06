#!/usr/bin/env bash
set -euo pipefail

SCRIPT_PATH="${BASH_SOURCE[0]:-$0}"
if command -v readlink >/dev/null 2>&1; then
  SCRIPT_PATH="$(readlink -f "${SCRIPT_PATH}" 2>/dev/null || printf '%s\n' "${SCRIPT_PATH}")"
fi
SCRIPT_DIR="$(cd "$(dirname "${SCRIPT_PATH}")" && pwd)"
REPO_DIR="${REPO_DIR:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
DATA_ROOT="${DATA_ROOT:-${REPO_DIR}/Data}"
STAMP="$(date +%Y%m%d_%H%M%S)"
OUT_ROOT="${OUT_ROOT:-${REPO_DIR}}"
OUT_DIR="${OUT_DIR:-${OUT_ROOT}/faissR_BENCHMARK1_CUDA_EUCLIDEAN_${STAMP}}"

THREADS="${THREADS:-2}"
K_VALUES="${K_VALUES:-15,30,50,100}"
TIMEOUT="${TIMEOUT:-2000}"
QUALITY_N="${QUALITY_N:-512}"
QUALITY_MAX_OPS="${QUALITY_MAX_OPS:-5e9}"

if [[ -n "${ENV_DIR:-}" && -z "${FAISSR_ENV_DIR:-}" ]]; then
  export FAISSR_ENV_DIR="${ENV_DIR}"
fi
if [[ -n "${CUDA_HOME:-}" && -z "${FAISSR_CUDA_LIB_DIR:-}" ]]; then
  export FAISSR_CUDA_LIB_DIR="${CUDA_HOME}"
fi

mkdir -p "${OUT_DIR}"
cd "${REPO_DIR}"

Rscript "${SCRIPT_DIR}/benchmark1_nn_speed.R" \
  --data_root="${DATA_ROOT}" \
  --out_dir="${OUT_DIR}" \
  --method_group=cuda \
  --include_faissr=TRUE \
  --include_external=TRUE \
  --include_non_knn=FALSE \
  --metrics=euclidean \
  --k_values="${K_VALUES}" \
  --threads="${THREADS}" \
  --timeout="${TIMEOUT}" \
  --quality_n="${QUALITY_N}" \
  --quality_max_ops="${QUALITY_MAX_OPS}" \
  "$@" 2>&1 | tee "${OUT_DIR}/benchmark1_cuda_euclidean.log"
