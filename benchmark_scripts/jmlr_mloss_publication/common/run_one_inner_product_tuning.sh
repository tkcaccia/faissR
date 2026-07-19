#!/usr/bin/env bash

set -euo pipefail

: "${METHOD:?METHOD must identify exactly one faissR method}"
: "${METHOD_LABEL:?METHOD_LABEL must provide a filesystem-safe label}"
: "${BACKEND:?BACKEND must be cpu or cuda}"

BASE_DIR="${BASE_DIR:-/scratch/firenze/NN}"
SUITE_ROOT="${SUITE_ROOT:-/scratch/firenze/NN/benchmark_scripts/jmlr_mloss_publication}"
COMMON_DIR="${SUITE_ROOT}/common"
DATA_ROOT="${DATA_ROOT:-${BASE_DIR}/Data}"
THREADS="${THREADS:-$(if [[ "${BACKEND}" == "cpu" ]]; then echo 12; else echo 2; fi)}"
THREAD_VALUES="${THREAD_VALUES:-${THREADS}}"
K_VALUES="${K_VALUES:-15,30,50,100}"
TARGET_RECALLS="${TARGET_RECALLS:-0.9,0.95,0.99}"
METRICS="${METRICS:-inner_product}"
TIMEOUT="${TIMEOUT:-2000}"
QUALITY_N="${QUALITY_N:-1024}"
CALIBRATION_SEED="${CALIBRATION_SEED:-4}"
GRID_LEVEL="${GRID_LEVEL:-wide}"
OUTPUT_VALUES="${OUTPUT_VALUES:-double}"
REAL_MANIFEST="${REAL_MANIFEST:-${DATA_ROOT}/float32_dataset_manifest_jmlr.csv}"
SINGULARITY_IMAGE="${SINGULARITY_IMAGE:-${BASE_DIR}/singularity/fastembedr_cuda.sif}"
SINGULARITY_GPU_FLAG="${SINGULARITY_GPU_FLAG:-$(if [[ "${BACKEND}" == "cuda" ]]; then echo --nv; fi)}"
R_BIN="${R_BIN:-Rscript}"
STAMP="${SLURM_JOB_ID:-manual}_$(date +%Y%m%d_%H%M%S)"
OUT_DIR="${OUT_DIR:-${BASE_DIR}/faissR_JMLR_MLOSS/calibration/${BACKEND}/${METHOD_LABEL}_${STAMP}}"

export OMP_NUM_THREADS="${THREADS}"
export OPENBLAS_NUM_THREADS="${THREADS}"
export MKL_NUM_THREADS="${THREADS}"
export VECLIB_MAXIMUM_THREADS="${THREADS}"
export RCPP_PARALLEL_NUM_THREADS="${THREADS}"

mkdir -p "${OUT_DIR}" "${BASE_DIR}/benchmark_logs"
cd "${BASE_DIR}"

if [[ -n "${SINGULARITY_IMAGE}" && -f "${SINGULARITY_IMAGE}" ]]; then
  if [[ -n "${SINGULARITY_GPU_FLAG}" ]]; then
    run_r() {
      singularity exec "${SINGULARITY_GPU_FLAG}" --bind "${BASE_DIR}:${BASE_DIR}" \
        "${SINGULARITY_IMAGE}" "${R_BIN}" "$@"
    }
  else
    run_r() {
      singularity exec --bind "${BASE_DIR}:${BASE_DIR}" \
        "${SINGULARITY_IMAGE}" "${R_BIN}" "$@"
    }
  fi
else
  run_r() { "${R_BIN}" "$@"; }
fi

if [[ ! -f "${REAL_MANIFEST}" ]]; then
  run_r "${COMMON_DIR}/make_hpc_float32_manifest.R" \
    --data_root="${DATA_ROOT}" --out="${REAL_MANIFEST}"
fi

TUNING_ARGS=(
  "${COMMON_DIR}/benchmark_method_tuning_from_reference.R"
  --manifest="${REAL_MANIFEST}"
  --out_dir="${OUT_DIR}"
  --method="${METHOD}"
  --backend="${BACKEND}"
  --metrics="${METRICS}"
  --k_values="${K_VALUES}"
  --reference_k=100
  --target_recalls="${TARGET_RECALLS}"
  --threads="${THREADS}"
  --thread_values="${THREAD_VALUES}"
  --output_values="${OUTPUT_VALUES}"
  --timeout="${TIMEOUT}"
  --quality_n="${QUALITY_N}"
  --seed="${CALIBRATION_SEED}"
  --grid_level="${GRID_LEVEL}"
  --resume=TRUE
  --skip_previous_timeouts=FALSE
)
if [[ -n "${DATASETS:-}" ]]; then TUNING_ARGS+=(--datasets="${DATASETS}"); fi
run_r "${TUNING_ARGS[@]}" \
  2>&1 | tee -a "${OUT_DIR}/${METHOD_LABEL}_${BACKEND}_inner_product_tuning.log"
