#!/usr/bin/env bash

#SBATCH --account=immunology
#SBATCH --partition=ada
#SBATCH --nodes=1
#SBATCH --ntasks=12
#SBATCH --time=48:00:00
#SBATCH --job-name="faissR_HNSW_CPU12_IP"
#SBATCH --chdir=/scratch/firenze/NN
#SBATCH --output=/scratch/firenze/NN/benchmark_logs/faissR_hnsw_cpu12_inner_product_%j.out
#SBATCH --error=/scratch/firenze/NN/benchmark_logs/faissR_hnsw_cpu12_inner_product_%j.err

set -euo pipefail

# Generated metric-specific wrapper for run_hpc_hnsw_tuning_cpu12.sh.
# Submit this file directly with sbatch to run exactly one metric.
export METRICS="inner_product"
export FAISSR_SINGLE_METRIC="inner_product"
export BASE_DIR="${BASE_DIR:-/scratch/firenze/NN}"
if [[ -z "${OUT_DIR:-}" ]]; then
  export OUT_DIR="${BASE_DIR}/faissR_HNSW_TUNING_CPU12_inner_product_$(date +%Y%m%d_%H%M%S)"
fi

WRAPPER_SCRIPT="${BASH_SOURCE[0]:-$0}"
if command -v readlink >/dev/null 2>&1; then
  WRAPPER_SCRIPT="$(readlink -f "${WRAPPER_SCRIPT}" 2>/dev/null || printf '%s\n' "${WRAPPER_SCRIPT}")"
fi
WRAPPER_SCRIPT_DIR="$(cd "$(dirname "${WRAPPER_SCRIPT}")" && pwd)"
if [[ -z "${SCRIPT_DIR:-}" ]]; then
  if [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "${SLURM_SUBMIT_DIR}/benchmark_scripts/run_hpc_hnsw_tuning_cpu12.sh" ]]; then
    export SCRIPT_DIR="${SLURM_SUBMIT_DIR}/benchmark_scripts"
  elif [[ -f "${BASE_DIR}/benchmark_scripts/run_hpc_hnsw_tuning_cpu12.sh" ]]; then
    export SCRIPT_DIR="${BASE_DIR}/benchmark_scripts"
  elif [[ -f "${WRAPPER_SCRIPT_DIR}/run_hpc_hnsw_tuning_cpu12.sh" ]]; then
    export SCRIPT_DIR="${WRAPPER_SCRIPT_DIR}"
  else
    echo "Cannot locate base launcher run_hpc_hnsw_tuning_cpu12.sh. Set SCRIPT_DIR to the faissR benchmark_scripts folder." >&2
    exit 1
  fi
fi
exec bash "${SCRIPT_DIR}/run_hpc_hnsw_tuning_cpu12.sh"
