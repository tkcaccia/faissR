#!/usr/bin/env bash

set -euo pipefail

: "${METHOD_ID:?METHOD_ID must identify exactly one benchmark method}"
: "${METHOD_LABEL:?METHOD_LABEL must provide a filesystem-safe method label}"
: "${BACKEND:?BACKEND must be cpu or cuda}"

BASE_DIR="${BASE_DIR:-/scratch/firenze/NN}"
SUITE_ROOT="${SUITE_ROOT:-/scratch/firenze/NN/benchmark_scripts/jmlr_mloss_publication}"
COMMON_DIR="${SUITE_ROOT}/common"
DATA_ROOT="${DATA_ROOT:-${BASE_DIR}/Data}"
LOG_DIR="${LOG_DIR:-${BASE_DIR}/benchmark_logs}"
THREADS="${THREADS:-$(if [[ "${BACKEND}" == "cpu" ]]; then echo 12; else echo 2; fi)}"
K_VALUES="${K_VALUES:-15,30,50,100}"
TARGET_RECALLS="${TARGET_RECALLS:-0.9,0.95,0.99}"
METHOD_METRICS="${METHOD_METRICS:-euclidean,cosine,correlation,inner_product}"
TIMEOUT="${TIMEOUT:-2000}"
QUALITY_N="${QUALITY_N:-1024}"
QUALITY_MAX_OPS="${QUALITY_MAX_OPS:-5e9}"
VALIDATION_SEEDS="${VALIDATION_SEEDS:-20260706,20260807}"
REPEATS="${REPEATS:-3}"
OUTPUT="${OUTPUT:-double}"
INCLUDE_EXTERNAL="${INCLUDE_EXTERNAL:-FALSE}"
INCLUDE_GPU_RESIDENT="${INCLUDE_GPU_RESIDENT:-TRUE}"
RUN_REAL="${RUN_REAL:-TRUE}"
RUN_MIPS="${RUN_MIPS:-FALSE}"
RUN_SPATIAL="${RUN_SPATIAL:-FALSE}"
R_BIN="${R_BIN:-Rscript}"
SINGULARITY_IMAGE="${SINGULARITY_IMAGE:-${BASE_DIR}/singularity/fastembedr_cuda.sif}"
SINGULARITY_GPU_FLAG="${SINGULARITY_GPU_FLAG:-$(if [[ "${BACKEND}" == "cuda" ]]; then echo --nv; fi)}"
STAMP="${SLURM_JOB_ID:-manual}_$(date +%Y%m%d_%H%M%S)"
OUT_DIR="${OUT_DIR:-${BASE_DIR}/faissR_JMLR_MLOSS/${BACKEND}/${METHOD_LABEL}_${STAMP}}"
REAL_MANIFEST="${REAL_MANIFEST:-${DATA_ROOT}/float32_dataset_manifest_jmlr.csv}"
SYNTH_DIR="${SYNTH_DIR:-${DATA_ROOT}/JMLR_synthetic_MIPS}"
SYNTH_MANIFEST="${SYNTH_MANIFEST:-${SYNTH_DIR}/jmlr_synthetic_mips_manifest.csv}"

export OMP_NUM_THREADS="${THREADS}"
export OPENBLAS_NUM_THREADS="${THREADS}"
export MKL_NUM_THREADS="${THREADS}"
export VECLIB_MAXIMUM_THREADS="${THREADS}"
export RCPP_PARALLEL_NUM_THREADS="${THREADS}"

mkdir -p "${OUT_DIR}" "${LOG_DIR}"
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

run_r -e 'library(faissR); stopifnot(faissR::faiss_available()); cat("faissR benchmark preflight OK\n")'
if [[ "${BACKEND}" == "cuda" ]]; then
  run_r -e 'library(faissR); stopifnot(faissR::cuda_available()); cat("faissR CUDA preflight OK\n")'
fi

if [[ ! -f "${REAL_MANIFEST}" && "${RUN_REAL}" == "TRUE" ]]; then
  run_r "${COMMON_DIR}/make_hpc_float32_manifest.R" \
    --data_root="${DATA_ROOT}" --out="${REAL_MANIFEST}"
fi

if [[ ( "${RUN_MIPS}" == "TRUE" || "${RUN_SPATIAL}" == "TRUE" ) && ! -f "${SYNTH_MANIFEST}" ]]; then
  run_r "${COMMON_DIR}/make_jmlr_synthetic_mips_manifest.R" \
    --out_dir="${SYNTH_DIR}" --manifest="${SYNTH_MANIFEST}"
fi

run_suite() {
  local suite="$1"
  local manifest="$2"
  local metrics="$3"
  local datasets="$4"
  local suite_out="${OUT_DIR}/${suite}"
  local bench_args=(
    "${COMMON_DIR}/benchmark_jmlr_tuned_methods.R"
    --manifest="${manifest}"
    --out_dir="${suite_out}"
    --backend="${BACKEND}"
    --methods="${METHOD_ID}"
    --threads="${THREADS}"
    --k_values="${K_VALUES}"
    --metrics="${metrics}"
    --target_recalls="${TARGET_RECALLS}"
    --timeout="${TIMEOUT}"
    --quality_n="${QUALITY_N}"
    --quality_max_ops="${QUALITY_MAX_OPS}"
    --validation_seeds="${VALIDATION_SEEDS}"
    --repeats="${REPEATS}"
    --output="${OUTPUT}"
    --include_external="${INCLUDE_EXTERNAL}"
    --include_gpu_resident="${INCLUDE_GPU_RESIDENT}"
  )
  if [[ -n "${datasets}" ]]; then bench_args+=(--datasets="${datasets}"); fi
  run_r "${bench_args[@]}"
}

{
  echo "METHOD_ID=${METHOD_ID}"
  echo "BACKEND=${BACKEND}"
  echo "THREADS=${THREADS}"
  echo "K_VALUES=${K_VALUES}"
  echo "TARGET_RECALLS=${TARGET_RECALLS}"
  echo "VALIDATION_SEEDS=${VALIDATION_SEEDS}"
  echo "REPEATS=${REPEATS}"
  echo "SINGULARITY_IMAGE=${SINGULARITY_IMAGE}"
  echo "OUT_DIR=${OUT_DIR}"

  if [[ "${RUN_REAL}" == "TRUE" ]]; then
    run_suite real "${REAL_MANIFEST}" "${METHOD_METRICS}" "${DATASETS:-}"
  fi

  if [[ "${RUN_MIPS}" == "TRUE" ]]; then
    MIPS_DATASETS="${MIPS_DATASETS:-synthetic_mips_n20000_p32_unit,synthetic_mips_n20000_p32_lognormal,synthetic_mips_n20000_p32_pareto,synthetic_mips_n70000_p128_unit,synthetic_mips_n70000_p128_lognormal,synthetic_mips_n70000_p128_pareto,synthetic_mips_n70000_p512_unit,synthetic_mips_n70000_p512_lognormal,synthetic_mips_n70000_p512_pareto,synthetic_mips_n200000_p64_unit,synthetic_mips_n200000_p64_lognormal,synthetic_mips_n200000_p64_pareto}"
    run_suite mips "${SYNTH_MANIFEST}" inner_product "${MIPS_DATASETS}"
  fi

  if [[ "${RUN_SPATIAL}" == "TRUE" ]]; then
    SPATIAL_DATASETS="${SPATIAL_DATASETS:-synthetic_spatial_n10000_p2_unit,synthetic_spatial_n10000_p3_unit}"
    run_suite spatial "${SYNTH_MANIFEST}" euclidean "${SPATIAL_DATASETS}"
  fi

  echo "DONE: ${OUT_DIR}"
} 2>&1 | tee -a "${OUT_DIR}/${METHOD_LABEL}_${BACKEND}.log"
