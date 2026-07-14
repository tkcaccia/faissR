#!/usr/bin/env bash

#SBATCH --account=immunology
#SBATCH --partition=ada
#SBATCH --nodes=1
#SBATCH --ntasks=12
#SBATCH --time=48:00:00
#SBATCH --job-name="faissR_JMLR_REF"
#SBATCH --chdir=/scratch/firenze/NN
#SBATCH --output=/scratch/firenze/NN/benchmark_logs/faissR_jmlr_reference_cpu12_%j.out
#SBATCH --error=/scratch/firenze/NN/benchmark_logs/faissR_jmlr_reference_cpu12_%j.err

set -euo pipefail

BASE_DIR="${BASE_DIR:-/scratch/firenze/NN}"
SUITE_ROOT="${SUITE_ROOT:-/scratch/firenze/NN/benchmark_scripts/jmlr_mloss_publication}"
COMMON_DIR="${SUITE_ROOT}/common"
DATA_ROOT="${DATA_ROOT:-${BASE_DIR}/Data}"
LOG_DIR="${LOG_DIR:-${BASE_DIR}/benchmark_logs}"
THREADS="${THREADS:-12}"
K_VALUES="${K_VALUES:-15,30,50,100}"
METRICS="${METRICS:-euclidean,cosine,correlation,inner_product}"
QUALITY_N="${QUALITY_N:-1024}"
VALIDATION_SEEDS="${VALIDATION_SEEDS:-20260706,20260807}"
CALIBRATION_SEED="${CALIBRATION_SEED:-4}"
TIMEOUT="${TIMEOUT:-2000}"
REAL_MANIFEST="${REAL_MANIFEST:-${DATA_ROOT}/float32_dataset_manifest_jmlr.csv}"
SYNTH_DIR="${SYNTH_DIR:-${DATA_ROOT}/JMLR_synthetic_MIPS}"
SYNTH_MANIFEST="${SYNTH_MANIFEST:-${SYNTH_DIR}/jmlr_synthetic_mips_manifest.csv}"
OUT_DIR="${OUT_DIR:-${BASE_DIR}/faissR_JMLR_MLOSS/references_${SLURM_JOB_ID:-manual}_$(date +%Y%m%d_%H%M%S)}"
SINGULARITY_IMAGE="${SINGULARITY_IMAGE:-${BASE_DIR}/singularity/fastembedr_cuda.sif}"
R_BIN="${R_BIN:-Rscript}"

export OMP_NUM_THREADS="${THREADS}"
export OPENBLAS_NUM_THREADS="${THREADS}"
export MKL_NUM_THREADS="${THREADS}"
export VECLIB_MAXIMUM_THREADS="${THREADS}"
export RCPP_PARALLEL_NUM_THREADS="${THREADS}"

mkdir -p "${OUT_DIR}" "${LOG_DIR}" "${SYNTH_DIR}"
cd "${BASE_DIR}"

if [[ -n "${SINGULARITY_IMAGE}" && -f "${SINGULARITY_IMAGE}" ]]; then
  run_r() {
    singularity exec --bind "${BASE_DIR}:${BASE_DIR}" \
      "${SINGULARITY_IMAGE}" "${R_BIN}" "$@"
  }
else
  run_r() { "${R_BIN}" "$@"; }
fi

{
  if [[ ! -f "${REAL_MANIFEST}" ]]; then
    run_r "${COMMON_DIR}/make_hpc_float32_manifest.R" \
      --data_root="${DATA_ROOT}" --out="${REAL_MANIFEST}"
  fi
  if [[ ! -f "${SYNTH_MANIFEST}" ]]; then
    run_r "${COMMON_DIR}/make_jmlr_synthetic_mips_manifest.R" \
      --out_dir="${SYNTH_DIR}" --manifest="${SYNTH_MANIFEST}"
  fi

  ALL_REFERENCE_SEEDS="${CALIBRATION_SEED},${VALIDATION_SEEDS}"
  IFS=',' read -r -a SEEDS <<< "${ALL_REFERENCE_SEEDS}"
  for seed in "${SEEDS[@]}"; do
    run_r "${COMMON_DIR}/benchmark_precompute_exact_references.R" \
      --manifest="${REAL_MANIFEST}" \
      --out_dir="${OUT_DIR}/real_seed${seed}" \
      --k_values="${K_VALUES}" \
      --reference_k=100 \
      --metrics="${METRICS}" \
      --threads="${THREADS}" \
      --timeout="${TIMEOUT}" \
      --quality_n="${QUALITY_N}" \
      --seed="${seed}" \
      --reference_methods=flat,exact \
      --resume=TRUE

    run_r "${COMMON_DIR}/benchmark_precompute_exact_references.R" \
      --manifest="${SYNTH_MANIFEST}" \
      --out_dir="${OUT_DIR}/synthetic_seed${seed}" \
      --k_values="${K_VALUES}" \
      --reference_k=100 \
      --metrics=euclidean,inner_product \
      --threads="${THREADS}" \
      --timeout="${TIMEOUT}" \
      --quality_n="${QUALITY_N}" \
      --seed="${seed}" \
      --reference_methods=flat,exact \
      --resume=TRUE
  done
  echo "DONE: ${OUT_DIR}"
} 2>&1 | tee -a "${OUT_DIR}/exact_references_cpu12.log"
