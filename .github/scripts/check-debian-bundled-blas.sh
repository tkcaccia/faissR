#!/bin/sh
set -eu

# Run inside Dockerfile.debian-bundled-blas, with an absolute source tarball.
archive=$1
work=$(mktemp -d)
trap 'rm -rf "$work"' 0
log_dir=${FAISSR_TEST_LOG_DIR:-$(pwd)}
R_HOME=$(R RHOME)
export R_HOME
mkdir -p "$log_dir" "$work/library"
R CMD config BLAS_LIBS | grep -- '-lRblas'
R CMD config LAPACK_LIBS | grep -- '-lRlapack'
R --vanilla --slave -e 'sessionInfo()' > "$log_dir/debian-session.txt"

if [ -n "${FAISSR_CONTROL_ARCHIVE:-}" ]; then
  mkdir "$work/control-library"
  if R CMD INSTALL --preclean --library="$work/control-library" \
      "$FAISSR_CONTROL_ARCHIVE" > "$log_dir/control-install.log" 2>&1; then
    echo 'The pre-fix control unexpectedly installed successfully.' >&2
    exit 1
  fi
  grep 'undefined symbol: ssyrk_' "$log_dir/control-install.log"
fi

tar -xzf "$archive" -C "$work"
cd "$work/faissR"
R CMD INSTALL --preclean --install-tests --library="$work/library" . \
  > "$log_dir/fixed-install.log" 2>&1
cp config.log "$log_dir/automatic-config.log"
cp src/Makevars "$log_dir/automatic-Makevars"
# Prove that bundled R alone was insufficient and automatic discovery fixed it.
grep 'undefined symbol:' "$log_dir/automatic-config.log"
grep 'FAISS compile/load check passed' "$log_dir/fixed-install.log"

R_LIBS_USER="$work/library" R --vanilla --slave -e '
library(faissR)
stopifnot(faiss_available())
set.seed(42)
x <- matrix(rnorm(2048 * 16), 2048, 16)
for (metric in c("euclidean", "cosine", "correlation")) {
    z <- nn(x, k = 5, method = "flat", backend = "cpu", metric = metric,
            exclude_self = TRUE)
    stopifnot(identical(dim(z$indices), c(2048L, 5L)),
              all(is.finite(z$distances)))
    reference <- if (metric == "correlation") x - rowMeans(x) else x
    if (metric != "euclidean") {
        reference <- reference / sqrt(rowSums(reference^2))
    }
    for (i in seq_len(5L)) {
        distances <- rowSums(sweep(reference, 2L, reference[i, ], "-")^2)
        distances[i] <- Inf
        expected <- head(order(distances), 5L)
        stopifnot(identical(as.integer(z$indices[i, ]), expected))
    }
}
cat("Debian bundled-BLAS functional smoke passed\n")
' > "$log_dir/smoke.log" 2>&1

if FAISSR_NUMERICAL_LIBS=-lfaissr_missing_blas sh configure \
    > "$log_dir/invalid-override.log" 2>&1; then
  echo 'An invalid explicit numerical provider was accepted.' >&2
  exit 1
fi
grep 'numerical-library compile/load check failed' "$log_dir/invalid-override.log"
FAISSR_NUMERICAL_LIBS='-llapack -lblas' sh configure \
  > "$log_dir/valid-override.log" 2>&1
grep -- '-llapack -lblas' src/Makevars

# An R configuration that already has complete external libraries needs no extra.
printf 'LAPACK_LIBS = -llapack\nBLAS_LIBS = -lblas\n' > "$work/external-Makevars"
R_MAKEVARS_USER="$work/external-Makevars" sh configure \
  > "$log_dir/external-config.log" 2>&1
grep 'extra numerical libraries: (none)' "$log_dir/external-config.log"
echo 'DEBIAN BUNDLED-BLAS REGRESSION PASSED'
