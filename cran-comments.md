## Initial submission, corrected faissR 0.1

The initial-submission version remains 0.1.

### Installation corrections

The Debian incoming log compiled the bundled FAISS source using Clang without
OpenMP flags. Configure now compiles and loads an OpenMP probe and reuses the
successful flags throughout FAISS and package compilation/linkage. Failed final
probes print their diagnostics into 00install.out.

The included FAISS 1.11.0 source fallback is CPU-only and uses the active R
toolchain and GNU make. It requires no CMake, precompiled binary or native-library
download during installation. A suitable installed FAISS library is preferred.
Complete LP64 BLAS/LAPACK and OpenMP remain system requirements; R's BLAS and
the host NVIDIA driver are never replaced. CUDA/cuVS are optional.

The documented compatibility patch also fixes explicit standard-header
requirements under libc++, header-only helpers, MinGW alignment and GCC
diagnostics concerning initialization and a file-handle deleter type.

### Incoming NOTE

The previous Windows incoming check completed without errors or warnings.
The title now quotes FAISS and spells out prediction/clustering rather than
using kNN. The expected new-submission NOTE remains.

### Validation

Source archive: 1,579,976 bytes. SHA256:

    77fd0e3dc7cd1d5c0bbd892466f5c37c547a38b4b3fcfee8d0ac719553d41842

Seven functional profiles completed R CMD check --as-cran with zero errors
and zero warnings: R-devel Debian Clang/Flang 23 with libc++, Ubuntu GCC,
Debian GCC with both installed/bundled FAISS, macOS arm64, and Windows x64
with both installed/bundled FAISS. Every profile rebuilt both vignettes and
ran examples, unit tests and independent-reference search smoke tests.

The R-devel profile also completed network incoming/URL checks and PDF/HTML
manual generation. Its two NOTEs are the initial submission and the test
toolchain's inherited -Wp,-D_FORTIFY_SOURCE=3 flag. Other functional profiles
reported no R check NOTEs. Upstream FAISS signedness compiler diagnostics
remain visible and documented separately from R check WARNINGs; warning flags
were not disabled. No fresh CUDA, Windows ARM64 or macOS Intel claim is made
for this archive.

The explicit Windows diagnostic-only profile also completed with zero errors,
warnings and NOTEs. FAISS is disabled there; it is not counted as functional
FAISS support.

The checker now validates dependency versions, preserves configure diagnostics,
audits compiler logs and each NOTE, and verifies completed profiles and identical
source checksums before producing its summary.
