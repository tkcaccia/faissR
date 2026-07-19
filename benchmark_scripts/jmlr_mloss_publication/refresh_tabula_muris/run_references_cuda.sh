#!/usr/bin/env bash

#SBATCH --account=l40sfree
#SBATCH --partition=l40s
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --gres=gpu:l40s:1
#SBATCH --time=48:00:00
#SBATCH --job-name="frJ_ref_tabula"
#SBATCH --chdir=/scratch/firenze/NN
#SBATCH --output=/scratch/firenze/NN/benchmark_logs/frJ_ref_tabula_cuda_%j.out
#SBATCH --error=/scratch/firenze/NN/benchmark_logs/frJ_ref_tabula_cuda_%j.err

set -euo pipefail

BASE_DIR="${BASE_DIR:-/scratch/firenze/NN}"
SUITE_ROOT="${SUITE_ROOT:-${BASE_DIR}/benchmark_scripts/jmlr_mloss_publication}"
COMMON_DIR="${SUITE_ROOT}/common"
DATA_ROOT="${DATA_ROOT:-${BASE_DIR}/Data}"
SINGULARITY_IMAGE="${SINGULARITY_IMAGE:-${BASE_DIR}/singularity/fastembedr_cuda.sif}"
MANIFEST="${MANIFEST:-${DATA_ROOT}/float32_dataset_manifest_tabula_muris_refresh.csv}"
OUT_DIR="${OUT_DIR:-${BASE_DIR}/faissR_JMLR_MLOSS/tabula_muris_refresh/references_cuda_${SLURM_JOB_ID:-manual}_$(date +%Y%m%d_%H%M%S)}"

mkdir -p "${OUT_DIR}" "${BASE_DIR}/benchmark_logs"

run_r() {
  singularity exec --nv --bind "${BASE_DIR}:${BASE_DIR}" \
    "${SINGULARITY_IMAGE}" Rscript "$@"
}

run_r -e 'library(faissR); stopifnot(faissR::cuda_available(), faissR::faiss_gpu_available()); cat("TabulaMuris CUDA reference preflight OK\n")'
run_r "${COMMON_DIR}/make_hpc_float32_manifest.R" \
  --data_root="${DATA_ROOT}" --datasets=TabulaMuris --out="${MANIFEST}"

run_r "${COMMON_DIR}/benchmark_precompute_exact_references_cuda.R" \
  --manifest="${MANIFEST}" \
  --datasets=TabulaMuris \
  --out_dir="${OUT_DIR}" \
  --metrics=euclidean,cosine,correlation,inner_product \
  --seeds=4,20260706,20260807 \
  --reference_k=100 \
  --quality_n=1024 \
  --audit_n=64 \
  --audit_max_ops=5e9 \
  --audit_atol=1e-5 \
  --audit_rtol=1e-4 \
  --threads=2 \
  --timeout=2000 \
  --resume=FALSE

echo "DONE: ${OUT_DIR}"
