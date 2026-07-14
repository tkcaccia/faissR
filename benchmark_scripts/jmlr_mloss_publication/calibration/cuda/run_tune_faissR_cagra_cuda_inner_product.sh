#!/usr/bin/env bash

#SBATCH --account=l40sfree
#SBATCH --partition=l40s
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --gres=gpu:l40s:1
#SBATCH --time=48:00:00
#SBATCH --job-name="frT_cagra_cuda"
#SBATCH --chdir=/scratch/firenze/NN
#SBATCH --output=/scratch/firenze/NN/benchmark_logs/frT_cagra_cuda_ip_%j.out
#SBATCH --error=/scratch/firenze/NN/benchmark_logs/frT_cagra_cuda_ip_%j.err

set -euo pipefail

SUITE_ROOT="${SUITE_ROOT:-/scratch/firenze/NN/benchmark_scripts/jmlr_mloss_publication}"
export SUITE_ROOT
export METHOD="cagra"
export METHOD_LABEL="faissR_cagra"
export BACKEND="cuda"
export THREADS="${THREADS:-2}"
export THREAD_VALUES="${THREAD_VALUES:-${THREADS}}"
export SINGULARITY_GPU_FLAG="--nv"
exec bash "${SUITE_ROOT}/common/run_one_inner_product_tuning.sh"
