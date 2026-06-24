#!/usr/bin/env bash
set -euo pipefail

ENV_DIR="${ENV_DIR:-/home/chiamaka/.fastEmbedR/micromamba/envs/fastembedr-faissgpu-cuvs}"
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-13.0}"
REPO_DIR="${REPO_DIR:-/mnt/sata_ssd/faissR_work/faissR}"
MANIFEST="${MANIFEST:-/mnt/sata_ssd/fastEmbedR_Data/float32_dataset_manifest.csv}"
OUT_DIR="${OUT_DIR:?Set OUT_DIR to the existing HNSW target-recall benchmark directory.}"

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

cd "$REPO_DIR"

Rscript benchmark_scripts/summarize_hnsw_target_recall.R \
  --out_dir="$OUT_DIR" 2>&1 | tee -a "$OUT_DIR/hnsw_target_recall_resume.log"

MISSING_TSV="$(mktemp)"
export OUT_DIR MISSING_TSV
Rscript - <<'RS'
path <- file.path(Sys.getenv("OUT_DIR"), "hnsw_target_recall_missing_rows.csv")
if (!file.exists(path)) stop("Missing-row file was not created: ", path, call. = FALSE)
missing <- read.csv(path, stringsAsFactors = FALSE)
if (nrow(missing)) {
  write.table(
    missing[, c("dataset", "backend", "k", "target_recall_requested"), drop = FALSE],
    Sys.getenv("MISSING_TSV"),
    sep = "\t",
    row.names = FALSE,
    col.names = FALSE,
    quote = FALSE
  )
} else {
  file.create(Sys.getenv("MISSING_TSV"))
}
RS

if [[ ! -s "$MISSING_TSV" ]]; then
  printf '[%s] no missing HNSW target-recall rows in %s\n' \
    "$(date --iso-8601=seconds)" "$OUT_DIR" | tee -a "$OUT_DIR/hnsw_target_recall_resume.log"
  rm -f "$MISSING_TSV"
  Rscript benchmark_scripts/summarize_hnsw_target_recall.R \
    --out_dir="$OUT_DIR" \
    --require_complete=TRUE 2>&1 | tee -a "$OUT_DIR/hnsw_target_recall_resume.log"
  exit 0
fi

while IFS=$'\t' read -r dataset backend k target; do
  printf '[%s] resuming dataset=%s backend=%s k=%s target_recall=%s\n' \
    "$(date --iso-8601=seconds)" "$dataset" "$backend" "$k" "$target" |
    tee -a "$OUT_DIR/hnsw_target_recall_resume.log"
  Rscript benchmark_scripts/benchmark_nn_float32_euclidean.R \
    --manifest="$MANIFEST" \
    --out_dir="$OUT_DIR" \
    --datasets="$dataset" \
    --backends="$backend" \
    --methods=hnsw \
    --k="$k" \
    --target_recall="$target" \
    --threads="$THREADS" \
    --timeout="$TIMEOUT" \
    --quality_n="$QUALITY_N" \
    --output="$OUTPUT" \
    "$@" 2>&1 | tee -a "$OUT_DIR/hnsw_target_recall_resume.log"
done < "$MISSING_TSV"

rm -f "$MISSING_TSV"

Rscript benchmark_scripts/summarize_hnsw_target_recall.R \
  --out_dir="$OUT_DIR" \
  --require_complete=TRUE 2>&1 | tee -a "$OUT_DIR/hnsw_target_recall_resume.log"

printf 'RESUME DONE: %s\n' "$OUT_DIR" | tee -a "$OUT_DIR/hnsw_target_recall_resume.log"
