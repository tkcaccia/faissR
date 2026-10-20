#!/usr/bin/env bash
set -euo pipefail
ARCHIVE=${1:?Usage: macos.sh ARCHIVE OUTPUT}
OUT=${2:?Unique output directory required}
HERE=$(cd "$(dirname "$0")" && pwd)
export LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8
export FAISSR_REQUIRE_FAISS=1 FAISSR_USE_CUDA=0 FAISSR_USE_CUVS=0
export FAISSR_REQUIRE_CUDA=0 FAISSR_REQUIRE_CUVS=0
export MAKEFLAGS=-j2
mkdir -p "$OUT"
shasum -a 256 "$ARCHIVE" > "$OUT/inputs.sha256"
Rscript "$HERE/run.R" "$ARCHIVE" "$OUT" functional "$HERE/faissR-smoke.R"
