#!/usr/bin/env bash
set -euo pipefail

ENV_DIR="${ENV_DIR:-/home/chiamaka/.fastEmbedR/micromamba/envs/fastembedr-faissgpu-cuvs}"
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-13.0}"
REPO_DIR="${REPO_DIR:-/mnt/sata_ssd/faissR_work/faissR}"
MANIFEST="${MANIFEST:-/mnt/sata_ssd/fastEmbedR_Data/float32_dataset_manifest.csv}"
STAMP="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="${OUT_DIR:-/mnt/sata_ssd/faissR_HNSW_TARGET_RECALL_FLOAT32_${STAMP}}"

DATASETS="${DATASETS:-MNIST,FashionMNIST,USPS,MetRef,TabulaMuris,flow18,mass41,imagenet,FlowRepository_FR-FCM-ZYRM_files,COIL20}"
BACKENDS="${BACKENDS:-cpu,cuda}"
K_VALUES="${K_VALUES:-10,15,50,100}"
TARGET_RECALLS="${TARGET_RECALLS:-0.9,0.95,0.99}"
THREADS="${THREADS:-4}"
TIMEOUT="${TIMEOUT:-600}"
QUALITY_N="${QUALITY_N:-128}"
OUTPUT="${OUTPUT:-float}"

export CONDA_PREFIX="$ENV_DIR"
export FAISS_HOME="${FAISS_HOME:-$ENV_DIR}"
export CUVS_HOME="${CUVS_HOME:-$ENV_DIR}"
export CUGRAPH_HOME="${CUGRAPH_HOME:-$ENV_DIR}"
export CUDA_HOME
export NVCC="${NVCC:-$CUDA_HOME/bin/nvcc}"
export FAISSR_USE_CUDA="${FAISSR_USE_CUDA:-1}"
export FAISSR_USE_CUVS="${FAISSR_USE_CUVS:-1}"
export FAISSR_ENV_DIR="$ENV_DIR"
export LD_LIBRARY_PATH="$ENV_DIR/lib:$ENV_DIR/targets/x86_64-linux/lib:$CUDA_HOME/targets/x86_64-linux/lib:${LD_LIBRARY_PATH:-}"
export LD_PRELOAD="$ENV_DIR/lib/libstdc++.so.6${LD_PRELOAD:+:$LD_PRELOAD}"

mkdir -p "$OUT_DIR"
cd "$REPO_DIR"

csv_quote() {
  local value="${1//\"/\"\"}"
  printf '"%s"' "$value"
}

write_config_row() {
  printf '%s,' "$1"
  csv_quote "$2"
  printf '\n'
}

{
  printf 'key,value\n'
  write_config_row manifest "$MANIFEST"
  write_config_row out_dir "$OUT_DIR"
  write_config_row datasets "$DATASETS"
  write_config_row backends "$BACKENDS"
  write_config_row method hnsw
  write_config_row metric euclidean
  write_config_row k_values "$K_VALUES"
  write_config_row target_recalls "$TARGET_RECALLS"
  write_config_row threads "$THREADS"
  write_config_row timeout "$TIMEOUT"
  write_config_row quality_n "$QUALITY_N"
  write_config_row output "$OUTPUT"
} > "$OUT_DIR/hnsw_target_recall_config.csv"

IFS=',' read -r -a k_values <<< "$K_VALUES"
IFS=',' read -r -a target_recalls <<< "$TARGET_RECALLS"

for k in "${k_values[@]}"; do
  for target in "${target_recalls[@]}"; do
    printf '[%s] k=%s target_recall=%s\n' "$(date --iso-8601=seconds)" "$k" "$target" | tee -a "$OUT_DIR/hnsw_target_recall.log"
    Rscript benchmark_scripts/benchmark_nn_float32_euclidean.R \
      --manifest="$MANIFEST" \
      --out_dir="$OUT_DIR" \
      --datasets="$DATASETS" \
      --backends="$BACKENDS" \
      --methods=hnsw \
      --k="$k" \
      --target_recall="$target" \
      --threads="$THREADS" \
      --timeout="$TIMEOUT" \
      --quality_n="$QUALITY_N" \
      --output="$OUTPUT" \
      "$@" 2>&1 | tee -a "$OUT_DIR/hnsw_target_recall.log"
  done
done

Rscript benchmark_scripts/summarize_hnsw_target_recall.R \
  --out_dir="$OUT_DIR" \
  --require_complete=TRUE 2>&1 | tee -a "$OUT_DIR/hnsw_target_recall.log"

printf 'DONE: %s\n' "$OUT_DIR" | tee -a "$OUT_DIR/hnsw_target_recall.log"
