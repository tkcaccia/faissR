# CRAN incoming-check corrections

Date: 20 October 2026. Package version remains **0.1**. This report documents
maintainer validation of the corrections; it is not a CRAN acceptance result.

## Source identity

The corrected source archive is 1,579,976 bytes, with SHA-256:

`77fd0e3dc7cd1d5c0bbd892466f5c37c547a38b4b3fcfee8d0ac719553d41842`

It contains pristine FAISS 1.11.0 CPU source and a documented compatibility
patch, not a precompiled library. Generated vignette outputs were retained
from a successful build; the final checks regenerate both vignettes from source.

## Corrections

The reported Windows incoming check completed with no errors or warnings; its
NOTE identified a new submission and two technical terms in the title. The
title now quotes FAISS and spells out the prediction/clustering scope.

The Debian incoming build used Clang/Flang and compiled bundled FAISS without
OpenMP flags. Configuration now compiles and loads an explicit OpenMP probe.
When R supplies no working flag, configure tests `-fopenmp` and reuses the
successful flags for FAISS, package compilation and linkage. Final failed
probes print diagnostics into the installation log, not only a temporary file.

Additional Clang/libc++ testing found missing explicit errno headers in FAISS
I/O implementation files and macros. The compatibility patch fixes those
includes, header-only SIMD helpers, MinGW alignment, an exhaustive NSG-reader
branch, and the mapped-file deleter type that GCC diagnosed. Search algorithms
and public tuning rules are unchanged.

## Verification

The completed profile outcomes are below. Installation,
independent-reference search smoke tests, unit tests, examples and both rebuilt
vignettes are required for every functional profile. The R-devel profile also
enables network incoming checks and PDF/HTML reference-manual checks.

| Profile | FAISS configuration | R check errors | Warnings | NOTEs |
| --- | --- | ---: | ---: | ---: |
| R-devel, Debian, Clang 23 / Flang 23, libc++ | Bundled CPU | 0 | 0 | 2 |
| Ubuntu 24.04, GCC / GFortran | Bundled CPU | 0 | 0 | 0 |
| Debian 13, GCC / GFortran | Installed CPU | 0 | 0 | 0 |
| Debian 13, GCC / GFortran | Bundled CPU | 0 | 0 | 0 |
| macOS arm64 | Bundled CPU | 0 | 0 | 0 |
| Windows x64, Rtools GCC | Default bundled fallback | 0 | 0 | 0 |
| Windows x64, Rtools GCC | Installed CPU | 0 | 0 | 0 |

These seven functional profiles all tested the identical archive above.
A separate Windows diagnostic-only profile also completed with zero errors,
warnings or NOTEs. It explicitly disables FAISS and is not counted as functional
FAISS support. All eight final profile results passed the completion/source
identity collector; the retained manifest also hashes each installed native
library.

Raw compiler diagnostics are audited separately from R CMD check WARNINGs.
Upstream FAISS signed/unsigned comparisons remain visible and classified;
they are not suppressed or represented as a compiler-warning-free build.
macOS can also report duplicate Fortran runtime libraries inherited from R's
link command. Unknown warnings, initialization diagnostics and ABI errors fail
the checker. The new-submission NOTE is expected; the R-hub profile's inherited
`-Wp,-D_FORTIFY_SOURCE=3` flag is recorded separately as a toolchain NOTE.

## Improved test infrastructure

The harness retains failed source/configure trees, snapshots the source and
runner, verifies checksums, records R configuration and checks every declared
dependency version before compilation. An outdated dependency in an inherited
library is no longer accepted just because its namespace exists. Bootstrap can
build a required source version when a platform binary is older.

Every run writes `dependencies.csv`, `compiler-diagnostics.csv` and
`check-notes.csv`; unreviewed items fail its gate. Regression tests exercise the
checker itself, including unexpected diagnostics, dependency constraints and
paths containing spaces. R-devel Clang/Flang is a separate profile from Linux
GCC/GFortran. The added CI workflow builds one archive and passes that exact
artifact into its Clang profile.

The full-check image provisions the R curl checker dependency and a
checksum-verified Inconsolata font for the indexed manual. The float dependency
uses the base image's GCC toolchain because its Fortran flags are not accepted
by Flang; faissR itself uses Clang/Flang in that profile.

No host R, BLAS, NVIDIA driver, SSH authorization or unrelated project was
modified. No HPC jobs or manuscript benchmarks were run.
This iteration targets CPU incoming checks; fresh CUDA, Windows GPU,
Windows ARM64 and macOS Intel results are not claimed for this archive.

See [the test procedure](package-testing.md) for reproduction and interpretation.
