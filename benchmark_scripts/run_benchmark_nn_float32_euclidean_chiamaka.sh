#!/usr/bin/env bash
set -euo pipefail

ENV_DIR="${ENV_DIR:-/home/chiamaka/.fastEmbedR/micromamba/envs/fastembedr-faissgpu-cuvs}"
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-13.0}"
REPO_DIR="${REPO_DIR:-/mnt/sata_ssd/faissR_work/faissR}"
MANIFEST="${MANIFEST:-/mnt/sata_ssd/fastEmbedR_Data/float32_dataset_manifest.csv}"
STAMP="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="${OUT_DIR:-/mnt/sata_ssd/faissR_FLOAT32_NN_EUCLIDEAN_K50_${STAMP}}"

export CONDA_PREFIX="$ENV_DIR"
export FAISS_HOME="${FAISS_HOME:-$ENV_DIR}"
export CUVS_HOME="${CUVS_HOME:-$ENV_DIR}"
export CUGRAPH_HOME="${CUGRAPH_HOME:-$ENV_DIR}"
export CUDA_HOME
export NVCC="${NVCC:-$CUDA_HOME/bin/nvcc}"
export FAISSR_USE_CUDA="${FAISSR_USE_CUDA:-1}"
export FAISSR_USE_CUVS="${FAISSR_USE_CUVS:-1}"
export LD_LIBRARY_PATH="$ENV_DIR/lib:$ENV_DIR/targets/x86_64-linux/lib:$CUDA_HOME/targets/x86_64-linux/lib:${LD_LIBRARY_PATH:-}"
export LD_PRELOAD="$ENV_DIR/lib/libstdc++.so.6${LD_PRELOAD:+:$LD_PRELOAD}"

mkdir -p "$OUT_DIR"
cd "$REPO_DIR"

Rscript benchmark_scripts/benchmark_nn_float32_euclidean.R \
  --manifest="$MANIFEST" \
  --out_dir="$OUT_DIR" \
  --backends=cpu,cuda \
  --methods=exact,flat,bruteforce,grid,hnsw,ivf,ivfpq,vamana,nsg,nndescent,usearch,cagra \
  --k=50 \
  --threads=4 \
  --timeout=600 \
  --quality_n=128 \
  --output=float \
  "$@" 2>&1 | tee "$OUT_DIR/float32_nn_benchmark.log"
