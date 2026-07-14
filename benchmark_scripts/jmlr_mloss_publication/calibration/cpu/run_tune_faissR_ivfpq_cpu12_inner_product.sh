#!/usr/bin/env bash

#SBATCH --account=immunology
#SBATCH --partition=ada
#SBATCH --nodes=1
#SBATCH --ntasks=12
#SBATCH --time=48:00:00
#SBATCH --job-name="frT_ivfpq_cpu12"
#SBATCH --chdir=/scratch/firenze/NN
#SBATCH --output=/scratch/firenze/NN/benchmark_logs/frT_ivfpq_cpu12_ip_%j.out
#SBATCH --error=/scratch/firenze/NN/benchmark_logs/frT_ivfpq_cpu12_ip_%j.err

set -euo pipefail

SUITE_ROOT="${SUITE_ROOT:-/scratch/firenze/NN/benchmark_scripts/jmlr_mloss_publication}"
export SUITE_ROOT
export METHOD="ivfpq"
export METHOD_LABEL="faissR_ivfpq"
export BACKEND="cpu"
export THREADS="${THREADS:-12}"
export THREAD_VALUES="${THREAD_VALUES:-${THREADS}}"
export SINGULARITY_GPU_FLAG=""
exec bash "${SUITE_ROOT}/common/run_one_inner_product_tuning.sh"
