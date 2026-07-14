#!/usr/bin/env bash

#SBATCH --account=l40sfree
#SBATCH --partition=l40s
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --gres=gpu:l40s:1
#SBATCH --time=48:00:00
#SBATCH --job-name="faissR_NND_CUDA_IP"
#SBATCH --chdir=/scratch/firenze/NN
#SBATCH --output=/scratch/firenze/NN/benchmark_logs/faissR_nndescent_cuda_inner_product_%j.out
#SBATCH --error=/scratch/firenze/NN/benchmark_logs/faissR_nndescent_cuda_inner_product_%j.err

set -euo pipefail

export METRICS="inner_product"
export FAISSR_SINGLE_METRIC="inner_product"
export BASE_DIR="${BASE_DIR:-/scratch/firenze/NN}"
export OUTPUT_VALUES="${OUTPUT_VALUES:-double}"
export GRID_LEVEL="${GRID_LEVEL:-wide}"
export SKIP_PREVIOUS_TIMEOUTS="${SKIP_PREVIOUS_TIMEOUTS:-FALSE}"
if [[ -z "${OUT_DIR:-}" ]]; then
  export OUT_DIR="${BASE_DIR}/faissR_NNDESCENT_TUNING_CUDA_inner_product_$(date +%Y%m%d_%H%M%S)"
fi

WRAPPER_SCRIPT="${BASH_SOURCE[0]:-$0}"
if command -v readlink >/dev/null 2>&1; then
  WRAPPER_SCRIPT="$(readlink -f "${WRAPPER_SCRIPT}" 2>/dev/null || printf '%s\n' "${WRAPPER_SCRIPT}")"
fi
WRAPPER_SCRIPT_DIR="$(cd "$(dirname "${WRAPPER_SCRIPT}")" && pwd)"
export SCRIPT_DIR="${SCRIPT_DIR:-${WRAPPER_SCRIPT_DIR}}"
exec bash "${SCRIPT_DIR}/run_hpc_nndescent_tuning_cuda.sh"
