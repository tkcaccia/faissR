#!/usr/bin/env bash

#SBATCH --account=l40sfree
#SBATCH --partition=l40s
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --gres=gpu:l40s:1
#SBATCH --time=48:00:00
#SBATCH --job-name="frJ_aggregate_cuda"
#SBATCH --chdir=/scratch/firenze/NN
#SBATCH --output=/scratch/firenze/NN/benchmark_logs/frJ_aggregate_cuda_%j.out
#SBATCH --error=/scratch/firenze/NN/benchmark_logs/frJ_aggregate_cuda_%j.err

set -euo pipefail

BASE_DIR="${BASE_DIR:-/scratch/firenze/NN}"
SUITE_ROOT="${SUITE_ROOT:-${BASE_DIR}/benchmark_scripts/jmlr_mloss_publication}"
RESULTS_ROOT="${RESULTS_ROOT:-${BASE_DIR}/faissR_JMLR_MLOSS/cuda}"
SINGULARITY_IMAGE="${SINGULARITY_IMAGE:-${BASE_DIR}/singularity/fastembedr_cuda.sif}"
STAMP="${SLURM_JOB_ID:-manual}_$(date +%Y%m%d_%H%M%S)"
OUT_DIR="${OUT_DIR:-${BASE_DIR}/faissR_JMLR_MLOSS/analysis/cuda_${STAMP}}"

mkdir -p "${OUT_DIR}" "${BASE_DIR}/benchmark_logs"
singularity exec --nv --bind "${BASE_DIR}:${BASE_DIR}" "${SINGULARITY_IMAGE}" \
  Rscript "${SUITE_ROOT}/analysis/aggregate_publication_results.R" \
  --results_root="${RESULTS_ROOT}" \
  --out_dir="${OUT_DIR}" \
  --backend=cuda \
  --target_recalls=0.9,0.95,0.99 \
  --expected_seeds=2 \
  --expected_repeats=3
