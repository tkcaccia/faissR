#!/usr/bin/env bash

#SBATCH --account=l40sfree
#SBATCH --partition=l40s
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --gres=gpu:l40s:1
#SBATCH --time=48:00:00
#SBATCH --job-name="faissR_B1_CUDA_EUCL"
#SBATCH --chdir=/scratch/firenze/NN
#SBATCH --output=/scratch/firenze/NN/benchmark_logs/faissR_benchmark1_compare_cuda_euclidean_%j.out
#SBATCH --error=/scratch/firenze/NN/benchmark_logs/faissR_benchmark1_compare_cuda_euclidean_%j.err

set -euo pipefail

# Benchmark #1 CUDA-only Euclidean comparison.
#
# This job compares faissR CUDA/FAISS-GPU/cuVS methods with CUDA-capable
# external R packages, such as cuda.ml when installed. CPU-only external
# packages are excluded by method_group=cuda.

export BASE_DIR="${BASE_DIR:-/scratch/firenze/NN}"
export DATA_ROOT="${DATA_ROOT:-${BASE_DIR}/Data}"
export LOG_DIR="${LOG_DIR:-${BASE_DIR}/benchmark_logs}"
export THREADS="${THREADS:-2}"
export K_VALUES="${K_VALUES:-15,30,50,100}"
export TIMEOUT="${TIMEOUT:-2000}"
export QUALITY_N="${QUALITY_N:-512}"
export QUALITY_MAX_OPS="${QUALITY_MAX_OPS:-5e9}"
export OUT_DIR="${OUT_DIR:-${BASE_DIR}/faissR_BENCHMARK1_COMPARE_CUDA_EUCLIDEAN_$(date +%Y%m%d_%H%M%S)}"
export SINGULARITY_IMAGE="${SINGULARITY_IMAGE:-${BASE_DIR}/singularity/fastembedr_cuda.sif}"
export SINGULARITY_GPU_FLAG="${SINGULARITY_GPU_FLAG:---nv}"
export R_BIN="${R_BIN:-Rscript}"

export OMP_NUM_THREADS="${THREADS}"
export OPENBLAS_NUM_THREADS="${THREADS}"
export MKL_NUM_THREADS="${THREADS}"
export VECLIB_MAXIMUM_THREADS="${THREADS}"
export RCPP_PARALLEL_NUM_THREADS="${THREADS}"

SUBMIT_SCRIPT="${BASH_SOURCE[0]:-$0}"
if command -v readlink >/dev/null 2>&1; then
  SUBMIT_SCRIPT="$(readlink -f "${SUBMIT_SCRIPT}" 2>/dev/null || printf '%s\n' "${SUBMIT_SCRIPT}")"
fi
SUBMIT_SCRIPT_DIR="$(cd "$(dirname "${SUBMIT_SCRIPT}")" && pwd)"
export SCRIPT_DIR="${SCRIPT_DIR:-${SUBMIT_SCRIPT_DIR}}"

mkdir -p "${OUT_DIR}" "${LOG_DIR}"
cd "${BASE_DIR}"

resolve_script() {
  local name="$1"
  if [[ -f "${SUBMIT_SCRIPT_DIR}/${name}" ]]; then
    printf '%s\n' "${SUBMIT_SCRIPT_DIR}/${name}"
  elif [[ -f "${SCRIPT_DIR}/${name}" ]]; then
    printf '%s\n' "${SCRIPT_DIR}/${name}"
  elif [[ -f "${SCRIPT_DIR}/benchmark_scripts/${name}" ]]; then
    printf '%s\n' "${SCRIPT_DIR}/benchmark_scripts/${name}"
  elif [[ -f "${PWD}/${name}" ]]; then
    printf '%s\n' "${PWD}/${name}"
  elif [[ -f "${PWD}/benchmark_scripts/${name}" ]]; then
    printf '%s\n' "${PWD}/benchmark_scripts/${name}"
  elif [[ -f "${BASE_DIR}/benchmark_scripts/${name}" ]]; then
    printf '%s\n' "${BASE_DIR}/benchmark_scripts/${name}"
  else
    echo "Cannot find ${name}." >&2
    echo "Set SCRIPT_DIR to the faissR benchmark_scripts folder if needed." >&2
    exit 1
  fi
}

BENCH_SCRIPT="$(resolve_script benchmark1_nn_speed.R)"

RUNNER=()
if [[ -n "${SINGULARITY_IMAGE}" && -f "${SINGULARITY_IMAGE}" ]]; then
  RUNNER=(singularity exec ${SINGULARITY_GPU_FLAG} --bind "${BASE_DIR}:${BASE_DIR}" "${SINGULARITY_IMAGE}")
fi

{
  echo "SCRIPT_DIR=${SCRIPT_DIR}"
  echo "BENCH_SCRIPT=${BENCH_SCRIPT}"
  echo "DATA_ROOT=${DATA_ROOT}"
  echo "OUT_DIR=${OUT_DIR}"
  echo "THREADS=${THREADS}"
  echo "K_VALUES=${K_VALUES}"
  echo "TIMEOUT=${TIMEOUT}"
  echo "[$(date --iso-8601=seconds)] running Benchmark #1 CUDA Euclidean comparison"
  "${RUNNER[@]}" "${R_BIN}" "${BENCH_SCRIPT}" \
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
    --quality_max_ops="${QUALITY_MAX_OPS}"
  echo "DONE: ${OUT_DIR}"
} 2>&1 | tee -a "${OUT_DIR}/benchmark1_compare_cuda_euclidean.log"
