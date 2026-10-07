#!/usr/bin/env bash
set -euo pipefail
IMAGE=$(realpath "${1:?Usage: run.sh IMAGE ARCHIVE OUTPUT [functional|diagnostic]}")
ARCHIVE=$(realpath "${2:?Source archive required}")
CONTAINER_ARCHIVE="/$(basename "$ARCHIVE")"
OUT=${3:?Unique output directory required}
PROFILE=${4:-functional}
HERE=$(cd "$(dirname "$0")/.." && pwd)
mkdir -p "$OUT"
OUT=$(realpath "$OUT")
mkdir -p "$OUT/tmp"
sha256sum "$IMAGE" "$ARCHIVE" > "$OUT/inputs.sha256"
ENGINE=${CONTAINER_ENGINE:-singularity}
"$ENGINE" exec --cleanenv --containall --no-home \
    --bind "$HERE:/harness:ro" --bind "$ARCHIVE:$CONTAINER_ARCHIVE:ro" \
    --bind "$OUT:/results" \
    --bind "$OUT/tmp:/tmp" \
    --env "PACKAGE_TEST_COMMIT=${PACKAGE_TEST_COMMIT:-UNRECORDED}" \
    --env "PACKAGE_TEST_IMAGE=$(basename "$IMAGE")" \
    --env "PACKAGE_TEST_BOOTSTRAP_DEPENDENCIES=${PACKAGE_TEST_BOOTSTRAP_DEPENDENCIES:-false}" \
    --env "PACKAGE_TEST_FORCE_SUGGESTS=${PACKAGE_TEST_FORCE_SUGGESTS:-false}" \
    "$IMAGE" Rscript /harness/run.R "$CONTAINER_ARCHIVE" /results "$PROFILE" /harness/faissR-smoke.R
