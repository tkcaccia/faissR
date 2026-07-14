#!/usr/bin/env bash

#SBATCH --account=l40sfree
#SBATCH --partition=l40s
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --gres=gpu:l40s:1
#SBATCH --time=48:00:00
#SBATCH --job-name="frJ_faissr_flat"
#SBATCH --chdir=/scratch/firenze/NN
#SBATCH --output=/scratch/firenze/NN/benchmark_logs/frJ_faissr_flat_cuda_%j.out
#SBATCH --error=/scratch/firenze/NN/benchmark_logs/frJ_faissr_flat_cuda_%j.err

set -euo pipefail

SUITE_ROOT="${SUITE_ROOT:-/scratch/firenze/NN/benchmark_scripts/jmlr_mloss_publication}"
export SUITE_ROOT
export METHOD_ID="faissR_cuda_flat"
export METHOD_LABEL="faissR_flat"
export BACKEND="cuda"
export THREADS="${THREADS:-2}"
export METHOD_METRICS="${METHOD_METRICS:-euclidean,cosine,correlation,inner_product}"
export INCLUDE_EXTERNAL="FALSE"
export INCLUDE_GPU_RESIDENT="TRUE"
export RUN_REAL="TRUE"
export RUN_MIPS="TRUE"
export RUN_SPATIAL="FALSE"
export SINGULARITY_GPU_FLAG="--nv"
exec bash "${SUITE_ROOT}/common/run_one_method.sh"
