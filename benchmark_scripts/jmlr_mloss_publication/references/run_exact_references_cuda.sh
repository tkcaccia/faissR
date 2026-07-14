#!/usr/bin/env bash

#SBATCH --account=l40sfree
#SBATCH --partition=l40s
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --gres=gpu:l40s:1
#SBATCH --time=48:00:00
#SBATCH --job-name="faissR_JMLR_REF_GPU"
#SBATCH --chdir=/scratch/firenze/NN
#SBATCH --output=/scratch/firenze/NN/benchmark_logs/faissR_jmlr_reference_cuda_%j.out
#SBATCH --error=/scratch/firenze/NN/benchmark_logs/faissR_jmlr_reference_cuda_%j.err

set -euo pipefail

BASE_DIR="${BASE_DIR:-/scratch/firenze/NN}"
SUITE_ROOT="${SUITE_ROOT:-/scratch/firenze/NN/benchmark_scripts/jmlr_mloss_publication}"
COMMON_DIR="${SUITE_ROOT}/common"
DATA_ROOT="${DATA_ROOT:-${BASE_DIR}/Data}"
THREADS="${THREADS:-2}"
K_VALUES="${K_VALUES:-15,30,50,100}"
REFERENCE_K="${REFERENCE_K:-100}"
QUALITY_N="${QUALITY_N:-1024}"
AUDIT_N="${AUDIT_N:-64}"
AUDIT_MAX_OPS="${AUDIT_MAX_OPS:-5e9}"
AUDIT_ATOL="${AUDIT_ATOL:-1e-5}"
AUDIT_RTOL="${AUDIT_RTOL:-1e-4}"
SEEDS="${SEEDS:-4,20260706,20260807}"
TIMEOUT="${TIMEOUT:-2000}"
REAL_MANIFEST="${REAL_MANIFEST:-${DATA_ROOT}/float32_dataset_manifest_jmlr.csv}"
SYNTH_DIR="${SYNTH_DIR:-${DATA_ROOT}/JMLR_synthetic_MIPS}"
SYNTH_MANIFEST="${SYNTH_MANIFEST:-${SYNTH_DIR}/jmlr_synthetic_mips_manifest.csv}"
OUT_DIR="${OUT_DIR:-${BASE_DIR}/faissR_JMLR_MLOSS/cuda_references_${SLURM_JOB_ID:-manual}_$(date +%Y%m%d_%H%M%S)}"
SINGULARITY_IMAGE="${SINGULARITY_IMAGE:-${BASE_DIR}/singularity/fastembedr_cuda.sif}"
R_BIN="${R_BIN:-Rscript}"

export OMP_NUM_THREADS="${THREADS}"
export OPENBLAS_NUM_THREADS="${THREADS}"
export MKL_NUM_THREADS="${THREADS}"
export VECLIB_MAXIMUM_THREADS="${THREADS}"
export RCPP_PARALLEL_NUM_THREADS="${THREADS}"
export R_BIN

mkdir -p "${OUT_DIR}" "${BASE_DIR}/benchmark_logs" "${SYNTH_DIR}"
cd "${BASE_DIR}"

run_r() {
  singularity exec --nv --bind "${BASE_DIR}:${BASE_DIR}" \
    "${SINGULARITY_IMAGE}" "${R_BIN}" "$@"
}

{
  if [[ ! -f "${REAL_MANIFEST}" ]]; then
    run_r "${COMMON_DIR}/make_hpc_float32_manifest.R" \
      --data_root="${DATA_ROOT}" --out="${REAL_MANIFEST}"
  fi
  if [[ ! -f "${SYNTH_MANIFEST}" ]]; then
    run_r "${COMMON_DIR}/make_jmlr_synthetic_mips_manifest.R" \
      --out_dir="${SYNTH_DIR}" --manifest="${SYNTH_MANIFEST}"
  fi

  run_r "${COMMON_DIR}/benchmark_precompute_exact_references_cuda.R" \
    --manifest="${REAL_MANIFEST}" \
    --out_dir="${OUT_DIR}/real" \
    --metrics=euclidean,cosine,correlation,inner_product \
    --seeds="${SEEDS}" \
    --reference_k="${REFERENCE_K}" \
    --quality_n="${QUALITY_N}" \
    --audit_n="${AUDIT_N}" \
    --audit_max_ops="${AUDIT_MAX_OPS}" \
    --audit_atol="${AUDIT_ATOL}" \
    --audit_rtol="${AUDIT_RTOL}" \
    --threads="${THREADS}" \
    --timeout="${TIMEOUT}" \
    --resume=TRUE

  run_r "${COMMON_DIR}/benchmark_precompute_exact_references_cuda.R" \
    --manifest="${SYNTH_MANIFEST}" \
    --out_dir="${OUT_DIR}/synthetic" \
    --metrics=euclidean,inner_product \
    --seeds="${SEEDS}" \
    --reference_k="${REFERENCE_K}" \
    --quality_n="${QUALITY_N}" \
    --audit_n="${AUDIT_N}" \
    --audit_max_ops="${AUDIT_MAX_OPS}" \
    --audit_atol="${AUDIT_ATOL}" \
    --audit_rtol="${AUDIT_RTOL}" \
    --threads="${THREADS}" \
    --timeout="${TIMEOUT}" \
    --resume=TRUE

  echo "DONE: ${OUT_DIR}"
} 2>&1 | tee -a "${OUT_DIR}/exact_references_cuda.log"
