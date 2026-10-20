# Shared by configure and configure.win. Only package-local files are written.
faissr_source_mode() {
  faiss_source_mode=${FAISSR_FAISS_SOURCE:-auto}
  for faiss_arg in "$@"; do
    case "$faiss_arg" in
      --with-faiss=*) faiss_source_mode=${faiss_arg#*=} ;;
    esac
  done
  case "$faiss_source_mode" in
    auto|system|bundled|none) ;;
    *)
      echo "--with-faiss must be auto, system, bundled, or none." >&2
      exit 1
      ;;
  esac
  if [ "$faiss_source_mode" = none ] && \
     [ "${FAISSR_REQUIRE_FAISS:-0}" = 1 ]; then
    echo "--with-faiss=none conflicts with FAISSR_REQUIRE_FAISS=1." >&2
    exit 1
  fi
}

faissr_configure_openmp() {
  faiss_omp_dir=$(mktemp -d "${TMPDIR:-/tmp}/faissR-openmp-XXXXXX")
  faiss_omp_log="$(pwd)/config.log"
  faiss_omp_makevars=Makevars
  faiss_omp_ext=so
  if [ "$faiss_windows" = 1 ]; then
    faiss_omp_makevars=Makevars.win
    faiss_omp_ext=dll
  fi
  cat > "$faiss_omp_dir/conftest.cpp" <<'EOF'
#include <omp.h>
#ifndef _OPENMP
#error OpenMP compilation is required for FAISS
#endif
extern "C" void faissr_openmp_probe(int* threads) {
    *threads = omp_get_max_threads();
}
EOF
  faiss_openmp_flags=""
  faiss_omp_ok=0
  for faiss_omp_kind in r compiler; do
    if [ "$faiss_omp_kind" = compiler ]; then
      faiss_openmp_flags=-fopenmp
    fi
    cat > "$faiss_omp_dir/$faiss_omp_makevars" <<EOF
CXX_STD = CXX20
PKG_CPPFLAGS = ${libomp_cppflags:-}
PKG_CXXFLAGS = \$(SHLIB_OPENMP_CXXFLAGS) ${extra_cxxflags:-} $faiss_openmp_flags
PKG_LIBS = \$(SHLIB_OPENMP_CXXFLAGS) ${libomp_libs:-} $faiss_openmp_flags
EOF
    rm -f "$faiss_omp_dir/conftest.o" \
      "$faiss_omp_dir/conftest.$faiss_omp_ext"
    printf '\nOpenMP compile/load probe: %s\n' "$faiss_omp_kind" >> "$faiss_omp_log"
    if (cd "$faiss_omp_dir" && \
        "$faiss_r_bin" CMD SHLIB conftest.cpp && \
        "$faiss_r_bin" --vanilla --slave -e \
        'dyn.load(commandArgs(TRUE)[1L], local=TRUE, now=TRUE); stopifnot(.C("faissr_openmp_probe", threads=integer(1L))$threads > 0L)' \
        --args "conftest.$faiss_omp_ext") >> "$faiss_omp_log" 2>&1; then
      faiss_omp_ok=1
      break
    fi
  done
  rm -rf "$faiss_omp_dir"
  if [ "$faiss_omp_ok" != 1 ]; then
    echo "OpenMP compilation/linkage failed with the active R toolchain." >&2
    tail -n 80 "$faiss_omp_log" >&2
    exit 1
  fi
  extra_cxxflags="${extra_cxxflags:-} $faiss_openmp_flags"
  echo "OpenMP compile/load check passed; extra flags: ${faiss_openmp_flags:-(R flags)}"
}

faissr_probe_library() {
  faiss_probe_dir=$(mktemp -d "${TMPDIR:-/tmp}/faissR-link-XXXXXX")
  faiss_probe_log="$(pwd)/config.log"
  cp src/faissr_console.cpp src/faissr_console.hpp "$faiss_probe_dir/"
  cat > "$faiss_probe_dir/conftest.cpp" <<'EOF'
#include <faiss/IndexFlat.h>
#ifdef FAISSR_FAISS_CPU_ONLY
#include <faiss/IndexAdditiveQuantizer.h>
#include <faiss/impl/io.h>
#include <faiss/impl/io_macros.h>
#endif
#if FAISS_VERSION_MAJOR < 1 || \
    (FAISS_VERSION_MAJOR == 1 && FAISS_VERSION_MINOR < 11)
#error faissR requires FAISS 1.11.0 or newer
#endif
extern "C" {
void sgemm_(); void ssyrk_(); void sgeqrf_(); void sorgqr_();
void sgesvd_(); void sgelsd_(); void sgetrf_(); void sgetri_();
void dgemm_(); void dsyev_(); void dgesvd_(); void dgetrf_(); void dgetri_();
void (*volatile faissr_numerical_symbols[])() = {
    sgemm_, ssyrk_, sgeqrf_, sorgqr_, sgesvd_, sgelsd_, sgetrf_, sgetri_,
    dgemm_, dsyev_, dgesvd_, dgetrf_, dgetri_
};
void *faissr_link_probe() { return new faiss::IndexFlatL2(2); }
#ifdef FAISSR_FAISS_CPU_ONLY
void faissr_io_header_probe(faiss::IOWriter* f) {
    int value = 0;
    WRITE1(value);
}
void faissr_quantizer_probe(int* ok) {
    faiss::IndexResidualQuantizer rq(4, 2, 4);
    faiss::IndexLocalSearchQuantizer lsq(4, 2, 4);
    faiss::IndexProductResidualQuantizer prq(4, 2, 1, 4);
    faiss::IndexProductLocalSearchQuantizer plsq(4, 2, 1, 4);
    *ok = rq.aq && rq.code_size == rq.aq->code_size &&
        lsq.aq && lsq.code_size == lsq.aq->code_size &&
        prq.aq && prq.code_size == prq.aq->code_size &&
        plsq.aq && plsq.code_size == plsq.aq->code_size;
}
#endif
}
EOF
  faiss_probe_ext=so
  faiss_probe_makevars=Makevars
  if [ "$faiss_windows" = 1 ]; then
    faiss_probe_ext=dll
    faiss_probe_makevars=Makevars.win
  fi
  faiss_numerical_libs=${FAISSR_NUMERICAL_LIBS:-}
  faiss_numerical_ok=0
  # This is an ABI/load test, not merely a successful unresolved shared link.
  for faiss_numeric_kind in explicit r lapack openblas accelerate; do
    case "$faiss_numeric_kind" in
      explicit)
        [ -n "$faiss_numerical_libs" ] || continue
        faiss_candidate=$faiss_numerical_libs ;;
      r) faiss_candidate="" ;;
      lapack) faiss_candidate="-llapack -lblas" ;;
      openblas) faiss_candidate="-lopenblas" ;;
      accelerate)
        [ "$(uname -s 2>/dev/null || echo unknown)" = Darwin ] || continue
        faiss_candidate="-framework Accelerate" ;;
    esac
    cat > "$faiss_probe_dir/$faiss_probe_makevars" <<EOF
CXX_STD = CXX20
PKG_CPPFLAGS = $faiss_cppflags ${libomp_cppflags:-}
PKG_CXXFLAGS = \$(SHLIB_OPENMP_CXXFLAGS) ${extra_cxxflags:-}
PKG_LIBS = \$(SHLIB_OPENMP_CXXFLAGS) ${faiss_openmp_flags:-} $faiss_libs ${libomp_libs:-} $faiss_candidate \$(LAPACK_LIBS) \$(BLAS_LIBS) \$(FLIBS)
EOF
    rm -f "$faiss_probe_dir/conftest.o" \
      "$faiss_probe_dir/conftest.$faiss_probe_ext"
    printf '\nFAISS compile/load probe; extra numerical flags: %s\n' \
      "${faiss_candidate:-(R libraries)}" >> "$faiss_probe_log"
    if (cd "$faiss_probe_dir" && \
        "$faiss_r_bin" CMD SHLIB conftest.cpp faissr_console.cpp && \
        LD_BIND_NOW=1 "$faiss_r_bin" --vanilla --slave -e \
        'p <- commandArgs(TRUE); dyn.load(p[1L], local=TRUE, now=TRUE); if(p[2L] == "1") stopifnot(.C("faissr_quantizer_probe", ok=integer(1L))$ok == 1L)' \
        --args "conftest.$faiss_probe_ext" "$faiss_bundled") >> "$faiss_probe_log" 2>&1; then
      faiss_numerical_ok=1
      faiss_numerical_libs=$faiss_candidate
      break
    fi
    # An explicit numerical-library override is never silently ignored.
    [ "$faiss_numeric_kind" != explicit ] || break
  done
  rm -rf "$faiss_probe_dir"
  if [ "$faiss_numerical_ok" = 1 ]; then
    echo "FAISS compile/load check passed; extra numerical libraries: ${faiss_numerical_libs:-(R libraries)}"
    return 0
  fi
  echo "FAISS version, ABI or numerical-library probe failed; see config.log." >&2
  tail -n 100 "$faiss_probe_log" >&2
  return 1
}

faissr_build_bundled() {
  faiss_vendor_dir="$(pwd)/src/vendor"
  faiss_build_dir="$faiss_vendor_dir/build"
  faiss_source_dir="$faiss_build_dir/faiss-1.11.0"
  faiss_archive="$faiss_vendor_dir/faiss-1.11.0.tar.gz"
  echo "Building bundled CPU-only FAISS 1.11.0 (static, generic CPU target)."
  if [ ! -r "$faiss_archive" ]; then
    echo "Bundled FAISS source archive is missing: $faiss_archive" >&2
    exit 1
  fi
  mkdir -p "$faiss_build_dir"
  # Start afresh on every configure; never reuse objects from another R/ABI.
  rm -rf "$faiss_source_dir"
  if ! "$faiss_r_bin" --vanilla --slave -e \
      'a <- commandArgs(TRUE); utils::untar(a[1L], exdir = a[2L])' \
      --args "$faiss_archive" "$faiss_build_dir"; then
    echo "Could not extract the bundled FAISS source archive." >&2
    exit 1
  fi
  if ! patch -s -p1 -d "$faiss_source_dir" \
      < "$faiss_vendor_dir/r-compat.patch"; then
    echo "Could not apply the documented FAISS R compatibility patch." >&2
    exit 1
  fi
  faiss_makeconf="$R_HOME/etc${R_ARCH:-}/Makeconf"
  if [ ! -f "$faiss_makeconf" ]; then
    faiss_makeconf="$R_HOME/etc/Makeconf"
  fi
  if [ ! -f "$faiss_makeconf" ]; then
    echo "Cannot find the active R installation's Makeconf." >&2
    exit 1
  fi
  {
    printf 'R_HOME = %s\n' "$R_HOME"
    printf 'R_MAKECONF = %s\n' "$faiss_makeconf"
    printf 'FAISS_CPPFLAGS = %s\n' "${libomp_cppflags:-}"
    printf 'FAISS_CXXFLAGS = %s\n' "${extra_cxxflags:-}"
    printf 'FAISS_WINDOWS = %s\n' "${faiss_windows:-0}"
  } > "$faiss_source_dir/faissR-config.mk"
  {
    for faiss_config_var in CXX20 CXX20STD CXX20FLAGS CXX20PICFLAGS \
        CPPFLAGS AR RANLIB; do
      faiss_config_value=$("$faiss_r_bin" CMD config "$faiss_config_var")
      printf '%s = %s\n' "$faiss_config_var" "$faiss_config_value"
    done
  } > "$faiss_source_dir/faissR-toolchain.mk"
  # Inherit the operator's MAKE/MAKEFLAGS; do not start an unbounded build.
  if ! ${MAKE:-make} -C "$faiss_source_dir" \
      -f "$faiss_vendor_dir/Makefile.static" libfaiss.a; then
    echo "Bundled FAISS compilation failed with the active R toolchain." >&2
    exit 1
  fi
  faiss_cppflags="-I\"$faiss_source_dir\" -DFAISS_STATIC -DFAISSR_R_SAFE_ASSERTS -DFAISSR_FAISS_CPU_ONLY"
  faiss_libdir="$faiss_source_dir"
  faiss_libs="\"$faiss_source_dir/libfaiss.a\""
  faiss_bundled=1
  echo "Bundled CPU FAISS static library built successfully."
}
