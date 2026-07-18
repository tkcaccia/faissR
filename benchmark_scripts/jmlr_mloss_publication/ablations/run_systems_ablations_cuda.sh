#!/usr/bin/env bash

#SBATCH --account=l40sfree
#SBATCH --partition=l40s
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --gres=gpu:l40s:1
#SBATCH --time=48:00:00
#SBATCH --job-name="frJ_ablation_cuda"
#SBATCH --chdir=/scratch/firenze/NN
#SBATCH --output=/scratch/firenze/NN/benchmark_logs/frJ_ablation_cuda_%j.out
#SBATCH --error=/scratch/firenze/NN/benchmark_logs/frJ_ablation_cuda_%j.err

set -euo pipefail

BASE_DIR="${BASE_DIR:-/scratch/firenze/NN}"
SUITE_ROOT="${SUITE_ROOT:-${BASE_DIR}/benchmark_scripts/jmlr_mloss_publication}"
SINGULARITY_IMAGE="${SINGULARITY_IMAGE:-${BASE_DIR}/singularity/fastembedr_cuda.sif}"
MANIFEST="${MANIFEST:-${BASE_DIR}/Data/float32_dataset_manifest_jmlr.csv}"
STAMP="${SLURM_JOB_ID:-manual}_$(date +%Y%m%d_%H%M%S)"
OUT_DIR="${OUT_DIR:-${BASE_DIR}/faissR_JMLR_MLOSS/ablations/cuda_${STAMP}}"

mkdir -p "${OUT_DIR}" "${BASE_DIR}/benchmark_logs"
singularity exec --nv --bind "${BASE_DIR}:${BASE_DIR}" "${SINGULARITY_IMAGE}" \
  Rscript "${SUITE_ROOT}/common/benchmark_jss_systems_ablations.R" \
  --manifest="${MANIFEST}" \
  --out_dir="${OUT_DIR}" \
  --backend=cuda \
  --datasets=COIL20,MNIST,TabulaMuris \
  --methods=flat,cagra,hnsw,ivf \
  --input_types=float32,double \
  --metric=euclidean \
  --k=30 \
  --threads=2 \
  --repeats=3 \
  --timeout=2000
