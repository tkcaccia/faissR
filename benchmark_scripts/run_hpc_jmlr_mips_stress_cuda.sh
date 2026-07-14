#!/usr/bin/env bash

#SBATCH --account=l40sfree
#SBATCH --partition=l40s
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --gres=gpu:l40s:1
#SBATCH --time=48:00:00
#SBATCH --job-name="faissR_MIPS_CUDA"
#SBATCH --chdir=/scratch/firenze/NN
#SBATCH --output=/scratch/firenze/NN/benchmark_logs/faissR_jmlr_mips_cuda_%j.out
#SBATCH --error=/scratch/firenze/NN/benchmark_logs/faissR_jmlr_mips_cuda_%j.err

set -euo pipefail

BASE_DIR="${BASE_DIR:-/scratch/firenze/NN}"
SCRIPT_DIR="${SCRIPT_DIR:-${BASE_DIR}/benchmark_scripts}"
SYNTH_DIR="${SYNTH_DIR:-${BASE_DIR}/Data/JMLR_synthetic_MIPS}"
MANIFEST="${MANIFEST:-${SYNTH_DIR}/jmlr_synthetic_mips_manifest.csv}"
IMAGE="${SINGULARITY_IMAGE:-${BASE_DIR}/singularity/fastembedr_cuda.sif}"

if [[ ! -f "${MANIFEST}" ]]; then
  singularity exec --nv --bind "${BASE_DIR}:${BASE_DIR}" "${IMAGE}" Rscript \
    "${SCRIPT_DIR}/make_jmlr_synthetic_mips_manifest.R" \
    --out_dir="${SYNTH_DIR}" --manifest="${MANIFEST}"
fi

export BUILD_MANIFEST=FALSE MANIFEST METRICS=inner_product INCLUDE_EXTERNAL=FALSE
export METHODS="${METHODS:-exact,flat,bruteforce,hnsw,ivf,ivfpq,ivfpq_fastscan,nndescent,nsg,vamana,cagra}"
export OUT_DIR="${OUT_DIR:-${BASE_DIR}/faissR_JMLR_MIPS_CUDA_$(date +%Y%m%d_%H%M%S)}"
exec bash "${SCRIPT_DIR}/run_hpc_jmlr_tuned_benchmark_cuda.sh"
