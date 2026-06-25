#!/usr/bin/env bash

#SBATCH --account=immunology
#SBATCH --partition=ada
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --time=48:00:00
#SBATCH --job-name="faissR_HNSW_CPU12"
#SBATCH --chdir=/scratch/firenze/NN
#SBATCH --output=/scratch/firenze/NN/benchmark_logs/faissR_hnsw_cpu12_%j.out
#SBATCH --error=/scratch/firenze/NN/benchmark_logs/faissR_hnsw_cpu12_%j.err

set -euo pipefail

# CPU-only faissR HNSW tuning-grid benchmark.
#
# Submit on the HPC with:
#   sbatch /scratch/firenze/NN/benchmark_scripts/run_hpc_hnsw_tuning_cpu12.sh
#
# The job scans float32 .RData files, benchmarks explicit backend="cpu",
# method="hnsw" settings for k=10,15,50,100, and writes recommendation tables
# for target recall 0.90, 0.95, and 0.99.

export BASE_DIR="${BASE_DIR:-/scratch/firenze/NN}"
export DATA_ROOT="${DATA_ROOT:-${BASE_DIR}/Data}"
SUBMIT_SCRIPT="${BASH_SOURCE[0]:-$0}"
if command -v readlink >/dev/null 2>&1; then
  SUBMIT_SCRIPT="$(readlink -f "${SUBMIT_SCRIPT}" 2>/dev/null || printf '%s\n' "${SUBMIT_SCRIPT}")"
fi
SUBMIT_SCRIPT_DIR="$(cd "$(dirname "${SUBMIT_SCRIPT}")" && pwd)"
export SCRIPT_DIR="${SCRIPT_DIR:-${SUBMIT_SCRIPT_DIR}}"
export THREADS_CPU="${THREADS_CPU:-12}"
export TIMEOUT="${TIMEOUT:-600}"
export QUALITY_N="${QUALITY_N:-256}"
export GRID_LEVEL="${GRID_LEVEL:-standard}"
export OUT_DIR="${OUT_DIR:-${BASE_DIR}/faissR_HNSW_TUNING_CPU12_$(date +%Y%m%d_%H%M%S)}"
export LOG_DIR="${LOG_DIR:-${BASE_DIR}/benchmark_logs}"
export SINGULARITY_IMAGE="${SINGULARITY_IMAGE:-${BASE_DIR}/singularity/fastembedr_cuda.sif}"
export SINGULARITY_GPU_FLAG="${SINGULARITY_GPU_FLAG:-}"
export R_BIN="${R_BIN:-Rscript}"

export DATASETS="${DATASETS:-COIL20,USPS,FashionMNIST,FlowRepository_FR-FCM-ZYRM_files,flow18,MNIST,imagenet,MetRef,mass41,TabulaMuris}"
export K_VALUES="${K_VALUES:-10,15,50,100}"
export TARGET_RECALLS="${TARGET_RECALLS:-0.9,0.95,0.99}"
export OUTPUT="${OUTPUT:-float}"

export OMP_NUM_THREADS="${THREADS_CPU}"
export OPENBLAS_NUM_THREADS="${THREADS_CPU}"
export MKL_NUM_THREADS="${THREADS_CPU}"
export VECLIB_MAXIMUM_THREADS="${THREADS_CPU}"
export RCPP_PARALLEL_NUM_THREADS="${THREADS_CPU}"

mkdir -p "${OUT_DIR}" "${LOG_DIR}"
cd "${BASE_DIR}"

if [[ -z "${FAISSR_SOURCE_DIR:-}" && -f "${SCRIPT_DIR}/../DESCRIPTION" ]]; then
  export FAISSR_SOURCE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
fi

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
    echo "Searched: ${SUBMIT_SCRIPT_DIR}, ${SCRIPT_DIR}, ${SCRIPT_DIR}/benchmark_scripts, ${PWD}, ${PWD}/benchmark_scripts, ${BASE_DIR}/benchmark_scripts" >&2
    echo "Set SCRIPT_DIR to the faissR benchmark_scripts folder if needed." >&2
    exit 1
  fi
}

MANIFEST_SCRIPT="$(resolve_script make_hpc_float32_manifest.R)"
BENCH_SCRIPT="$(resolve_script benchmark_hnsw_tuning_grid.R)"
MANIFEST="${MANIFEST:-${OUT_DIR}/float32_dataset_manifest.csv}"

RUNNER=()
if [[ -n "${SINGULARITY_IMAGE}" && -f "${SINGULARITY_IMAGE}" ]]; then
  RUNNER=(singularity exec ${SINGULARITY_GPU_FLAG} --bind "${BASE_DIR}:${BASE_DIR}" "${SINGULARITY_IMAGE}")
fi

{
  echo "SCRIPT_DIR=${SCRIPT_DIR}"
  echo "SUBMIT_SCRIPT_DIR=${SUBMIT_SCRIPT_DIR}"
  echo "MANIFEST_SCRIPT=${MANIFEST_SCRIPT}"
  echo "BENCH_SCRIPT=${BENCH_SCRIPT}"
  echo "[$(date --iso-8601=seconds)] building float32 manifest"
  "${RUNNER[@]}" "${R_BIN}" "${MANIFEST_SCRIPT}" \
    --data_root="${DATA_ROOT}" \
    --out="${MANIFEST}" \
    --datasets="${DATASETS}"

  echo "[$(date --iso-8601=seconds)] running CPU HNSW tuning grid"
  "${RUNNER[@]}" "${R_BIN}" "${BENCH_SCRIPT}" \
    --manifest="${MANIFEST}" \
    --out_dir="${OUT_DIR}" \
    --datasets="${DATASETS}" \
    --backend=cpu \
    --k_values="${K_VALUES}" \
    --target_recalls="${TARGET_RECALLS}" \
    --threads="${THREADS_CPU}" \
    --timeout="${TIMEOUT}" \
    --quality_n="${QUALITY_N}" \
    --output="${OUTPUT}" \
    --grid_level="${GRID_LEVEL}" \
    --reference_backend=cpu \
    --resume=TRUE

  echo "DONE: ${OUT_DIR}"
} 2>&1 | tee -a "${OUT_DIR}/hnsw_tuning_cpu12.log"
