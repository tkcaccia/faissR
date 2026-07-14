#!/usr/bin/env bash

#SBATCH --account=immunology
#SBATCH --partition=ada
#SBATCH --nodes=1
#SBATCH --ntasks=12
#SBATCH --time=48:00:00
#SBATCH --job-name="faissR_JMLR_CPU"
#SBATCH --chdir=/scratch/firenze/NN
#SBATCH --output=/scratch/firenze/NN/benchmark_logs/faissR_jmlr_tuned_cpu12_%j.out
#SBATCH --error=/scratch/firenze/NN/benchmark_logs/faissR_jmlr_tuned_cpu12_%j.err

set -euo pipefail

export BASE_DIR="${BASE_DIR:-/scratch/firenze/NN}"
export DATA_ROOT="${DATA_ROOT:-${BASE_DIR}/Data}"
export LOG_DIR="${LOG_DIR:-${BASE_DIR}/benchmark_logs}"
export THREADS="${THREADS:-12}"
export K_VALUES="${K_VALUES:-15,30,50,100}"
export METRICS="${METRICS:-euclidean,cosine,correlation,inner_product}"
export TARGET_RECALLS="${TARGET_RECALLS:-0.9,0.95,0.99}"
export TIMEOUT="${TIMEOUT:-2000}"
export QUALITY_N="${QUALITY_N:-1024}"
export QUALITY_MAX_OPS="${QUALITY_MAX_OPS:-5e9}"
export VALIDATION_SEEDS="${VALIDATION_SEEDS:-20260706,20260807}"
export REPEATS="${REPEATS:-3}"
export OUTPUT="${OUTPUT:-double}"
export INCLUDE_EXTERNAL="${INCLUDE_EXTERNAL:-TRUE}"
export BUILD_MANIFEST="${BUILD_MANIFEST:-TRUE}"
export OUT_DIR="${OUT_DIR:-${BASE_DIR}/faissR_JMLR_TUNED_CPU12_$(date +%Y%m%d_%H%M%S)}"
export SINGULARITY_IMAGE="${SINGULARITY_IMAGE:-${BASE_DIR}/singularity/fastembedr_cuda.sif}"
export SINGULARITY_GPU_FLAG="${SINGULARITY_GPU_FLAG:-}"
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
  elif [[ -f "${BASE_DIR}/benchmark_scripts/${name}" ]]; then
    printf '%s\n' "${BASE_DIR}/benchmark_scripts/${name}"
  else
    echo "Cannot find ${name}. Set SCRIPT_DIR to the faissR benchmark_scripts folder." >&2
    exit 1
  fi
}

MANIFEST_SCRIPT="$(resolve_script make_hpc_float32_manifest.R)"
BENCH_SCRIPT="$(resolve_script benchmark_jmlr_tuned_methods.R)"
REF_SCRIPT="$(resolve_script benchmark_precompute_exact_references.R)"
MANIFEST="${MANIFEST:-${OUT_DIR}/float32_dataset_manifest.csv}"

RUNNER=()
if [[ -n "${SINGULARITY_IMAGE}" && -f "${SINGULARITY_IMAGE}" ]]; then
  RUNNER=(singularity exec ${SINGULARITY_GPU_FLAG} --bind "${BASE_DIR}:${BASE_DIR}" "${SINGULARITY_IMAGE}")
fi

{
  echo "SCRIPT_DIR=${SCRIPT_DIR}"
  echo "MANIFEST_SCRIPT=${MANIFEST_SCRIPT}"
  echo "BENCH_SCRIPT=${BENCH_SCRIPT}"
  echo "DATA_ROOT=${DATA_ROOT}"
  echo "OUT_DIR=${OUT_DIR}"
  echo "SINGULARITY_IMAGE=${SINGULARITY_IMAGE}"
  echo "THREADS=${THREADS}"
  echo "K_VALUES=${K_VALUES}"
  echo "METRICS=${METRICS}"
  echo "TARGET_RECALLS=${TARGET_RECALLS}"
  echo "TIMEOUT=${TIMEOUT}"
  echo "VALIDATION_SEEDS=${VALIDATION_SEEDS}"
  echo "REPEATS=${REPEATS}"
  if [[ "${BUILD_MANIFEST}" == "TRUE" || "${BUILD_MANIFEST}" == "true" || "${BUILD_MANIFEST}" == "1" ]]; then
    echo "[$(date --iso-8601=seconds)] building float32 manifest"
    "${RUNNER[@]}" "${R_BIN}" "${MANIFEST_SCRIPT}" \
      --data_root="${DATA_ROOT}" \
      --out="${MANIFEST}" \
      ${DATASETS:+--datasets="${DATASETS}"}
  else
    echo "[$(date --iso-8601=seconds)] using existing manifest: ${MANIFEST}"
  fi

  IFS=',' read -r -a REF_SEEDS <<< "${VALIDATION_SEEDS}"
  for REF_SEED in "${REF_SEEDS[@]}"; do
    echo "[$(date --iso-8601=seconds)] precomputing exact references for validation seed ${REF_SEED}"
    "${RUNNER[@]}" "${R_BIN}" "${REF_SCRIPT}" \
      --manifest="${MANIFEST}" \
      --out_dir="${OUT_DIR}/references_seed${REF_SEED}" \
      --datasets="${DATASETS:-}" \
      --k_values="${K_VALUES}" \
      --metrics="${METRICS}" \
      --threads="${THREADS}" \
      --timeout="${TIMEOUT}" \
      --quality_n="${QUALITY_N}" \
      --seed="${REF_SEED}" \
      --resume=TRUE
  done

  echo "[$(date --iso-8601=seconds)] running JMLR tuned CPU benchmark"
  "${RUNNER[@]}" "${R_BIN}" "${BENCH_SCRIPT}" \
    --manifest="${MANIFEST}" \
    --out_dir="${OUT_DIR}" \
    --backend=cpu \
    --threads="${THREADS}" \
    --k_values="${K_VALUES}" \
    --metrics="${METRICS}" \
    --target_recalls="${TARGET_RECALLS}" \
    --timeout="${TIMEOUT}" \
    --quality_n="${QUALITY_N}" \
    --quality_max_ops="${QUALITY_MAX_OPS}" \
    --validation_seeds="${VALIDATION_SEEDS}" \
    --repeats="${REPEATS}" \
    --output="${OUTPUT}" \
    --include_external="${INCLUDE_EXTERNAL}" \
    ${DATASETS:+--datasets="${DATASETS}"} \
    ${METHODS:+--methods="${METHODS}"}
  echo "DONE: ${OUT_DIR}"
} 2>&1 | tee -a "${OUT_DIR}/jmlr_tuned_cpu12.log"
