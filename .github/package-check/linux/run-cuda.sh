#!/usr/bin/env bash
set -euo pipefail

IMAGE=$(realpath "${1:?Usage: run-cuda.sh IMAGE ARCHIVE OUTPUT cuda-native|cuda-cuvs}")
ARCHIVE=$(realpath "${2:?Source archive required}")
CONTAINER_ARCHIVE="/$(basename "$ARCHIVE")"
OUT=${3:?Unique output directory required}
PROFILE=${4:-cuda-native}
case "$PROFILE" in
    cuda-native|cuda-cuvs) ;;
    *) echo "Profile must be cuda-native or cuda-cuvs." >&2; exit 2 ;;
esac

HERE=$(cd "$(dirname "$0")/.." && pwd)
mkdir -p "$OUT"
OUT=$(realpath "$OUT")
test ! -e "$OUT/status.csv" || {
    echo "Output already contains a run: $OUT" >&2
    exit 1
}
mkdir -p "$OUT/tmp"
sha256sum "$IMAGE" "$ARCHIVE" > "$OUT/inputs.sha256"

ENGINE=${CONTAINER_ENGINE:-singularity}
CUDA_ROOT=${CONTAINER_CUDA_HOME:-/usr/local/cuda}
CUVS_ROOT=${CONTAINER_CUVS_HOME:-/opt/cuvs}
FAISS_ROOT=${CONTAINER_FAISS_HOME:-/usr/local}
R_BIN=${CONTAINER_R_BIN:-/opt/R/bin}
CHECK_TOOLS_ROOT=${CONTAINER_CHECK_TOOLS_ROOT:-}
CHECK_TOOL_BIN=""
CHECK_TOOL_LIB=""
CHECK_TOOL_BIND=()
if [ -n "$CHECK_TOOLS_ROOT" ]; then
    CHECK_TOOLS_ROOT=$(realpath "$CHECK_TOOLS_ROOT")
    test -x "$CHECK_TOOLS_ROOT/usr/bin/checkbashisms"
    test -x "$CHECK_TOOLS_ROOT/usr/bin/nm"
    CHECK_TOOL_BIND=(--bind "$CHECK_TOOLS_ROOT:/check-tools:ro")
    CHECK_TOOL_BIN="/check-tools/usr/bin:"
    CHECK_TOOL_LIB="/check-tools/usr/lib/x86_64-linux-gnu:"
fi
REQUIRE_CUVS=0
USE_CUVS=0
if [ "$PROFILE" = "cuda-cuvs" ]; then
    REQUIRE_CUVS=1
    USE_CUVS=1
fi

"$ENGINE" exec --nv --cleanenv --containall --no-home \
    --bind "$HERE:/harness:ro" \
    --bind "$ARCHIVE:$CONTAINER_ARCHIVE:ro" \
    --bind "$OUT:/results" \
    --bind "$OUT/tmp:/tmp" \
    "${CHECK_TOOL_BIND[@]}" \
    --env "PACKAGE_TEST_COMMIT=${PACKAGE_TEST_COMMIT:-UNRECORDED}" \
    --env "PACKAGE_TEST_IMAGE=$(basename "$IMAGE")" \
    --env "PACKAGE_TEST_BOOTSTRAP_DEPENDENCIES=true" \
    --env "CUDA_HOME=$CUDA_ROOT" \
    --env "PATH=${CHECK_TOOL_BIN}$R_BIN:/usr/local/bin:/usr/bin:/bin:$CUDA_ROOT/bin" \
    --env "LD_LIBRARY_PATH=${CHECK_TOOL_LIB}$CUDA_ROOT/lib:$CUDA_ROOT/lib64:$FAISS_ROOT/lib:$FAISS_ROOT/lib64:$CUVS_ROOT/lib:$CUVS_ROOT/lib64" \
    --env "FAISS_HOME=$FAISS_ROOT" \
    --env "CUVS_HOME=$CUVS_ROOT" \
    --env "FAISSR_REQUIRE_FAISS=1" \
    --env "FAISSR_REQUIRE_CUDA=1" \
    --env "FAISSR_USE_CUVS=$USE_CUVS" \
    --env "FAISSR_REQUIRE_CUVS=$REQUIRE_CUVS" \
    --env "FAISSR_CUDA_ARCH=${FAISSR_CUDA_ARCH:-}" \
    --env "FAISSR_CUDA_PTX_ARCH=${FAISSR_CUDA_PTX_ARCH:-}" \
    "$IMAGE" "$R_BIN/Rscript" /harness/run.R \
        "$CONTAINER_ARCHIVE" /results "$PROFILE" /harness/faissR-smoke.R
