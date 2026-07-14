#!/usr/bin/env bash

set -euo pipefail

# Run this script on the HPC login node from /scratch/firenze/NN.
# It submits one exact-reference job, then all method-specific CPU and CUDA
# inner-product tuning jobs with an afterok dependency on that reference.

BASE_DIR="${BASE_DIR:-/scratch/firenze/NN}"
SCRIPT_DIR="${SCRIPT_DIR:-${BASE_DIR}/benchmark_scripts}"
QUALITY_N="${QUALITY_N:-1024}"
SEED="${SEED:-4}"
K_VALUES="${K_VALUES:-15,30,50,100}"
TARGET_RECALLS="${TARGET_RECALLS:-0.9,0.95,0.99}"
TIMEOUT="${TIMEOUT:-2000}"
DATASETS="${DATASETS:-COIL20,USPS,FashionMNIST,FlowRepository_FR-FCM-ZYRM_files,flow18,MNIST,imagenet,MetRef,mass41,TabulaMuris}"

common_export="ALL,BASE_DIR=${BASE_DIR},SCRIPT_DIR=${SCRIPT_DIR},METRICS=inner_product,QUALITY_N=${QUALITY_N},SEED=${SEED},K_VALUES=${K_VALUES},TARGET_RECALLS=${TARGET_RECALLS},TIMEOUT=${TIMEOUT},GRID_LEVEL=wide,OUTPUT_VALUES=double,SKIP_PREVIOUS_TIMEOUTS=FALSE,DATASETS=${DATASETS}"

reference_job=$(sbatch --parsable \
  --export="${common_export}" \
  "${SCRIPT_DIR}/run_hpc_precompute_exact_references_cpu12.sh")
reference_job="${reference_job%%;*}"
echo "Exact-reference job: ${reference_job}"

cpu_methods=(exact flat bruteforce hnsw ivf ivfpq ivfpq_fastscan nndescent nsg vamana)
cuda_methods=(exact flat bruteforce hnsw ivf ivfpq ivfpq_fastscan nndescent nsg vamana cagra)

for method in "${cpu_methods[@]}"; do
  script="${SCRIPT_DIR}/run_hpc_${method}_tuning_cpu12_inner_product.sh"
  job=$(sbatch --parsable --dependency="afterok:${reference_job}" \
    --export="${common_export}" "${script}")
  echo "CPU ${method}: ${job%%;*}"
done

for method in "${cuda_methods[@]}"; do
  script="${SCRIPT_DIR}/run_hpc_${method}_tuning_cuda_inner_product.sh"
  job=$(sbatch --parsable --dependency="afterok:${reference_job}" \
    --export="${common_export}" "${script}")
  echo "CUDA ${method}: ${job%%;*}"
done
