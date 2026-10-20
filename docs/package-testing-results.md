# Earlier faissR 0.1 portability validation

This page records an earlier archive. The current CRAN resubmission corrections
and final source identity are described in
[the incoming-check report](cran-incoming-20261020.md). Do not transfer results
between archives solely because their package version is the same.

Test date: 20 October 2026. These are local validation results, not CRAN
acceptance or a published release. No commit or push was made for this run.

The identical source archive was used on every platform:

`d55e95106c8b2bff180c4a74d49f2bbc16a4c760a47ed624727dd82536ec051b`

The archive contains 1,577,144 bytes (about 1.5 MiB). It includes the pristine
FAISS 1.11.0 source archive (about 1.1 MiB), its licenses and a documented
compatibility patch. No precompiled native library is shipped. Only CPU FAISS
is built from that source; NVIDIA libraries remain optional external providers.

## Completed functional checks

Each row passed installation, an independent-reference search smoke test and
`R CMD check --as-cran --no-manual`, with every suggested R package available
and both vignettes rebuilt. Remote CRAN incoming checks were disabled; this
matrix does not replace CRAN's incoming service.

| Platform | R | FAISS/provider configuration | Errors | Warnings | Notes |
| --- | --- | --- | ---: | ---: | ---: |
| macOS 14.5 arm64 | 4.6.0 | Bundled CPU FAISS | 0 | 0 | 0 |
| Ubuntu 24.04 x86_64 | 4.6.0 | Bundled CPU FAISS, OpenBLAS | 0 | 0 | 0 |
| Debian 13 x86_64 | 4.6.0 | Installed FAISS, bundled R BLAS | 0 | 0 | 0 |
| Debian 13 x86_64 | 4.6.0 | Bundled CPU FAISS, bundled R BLAS | 0 | 0 | 0 |
| Fedora 44 x86_64 | 4.6.1 | Default fallback, no installed FAISS | 0 | 0 | 1 |
| Windows x64 | 4.6.1 | Default fallback, no FAISS prefix | 0 | 0 | 1 |
| Windows x64 | 4.6.1 | Installed Rtools-compatible FAISS | 0 | 0 | 0 |
| Debian 13, NVIDIA RTX 5060 Ti | 4.6.1 | Bundled CPU FAISS + native CUDA 13.2 | 0 | 0 | 0 |
| Debian 13, NVIDIA RTX 5060 Ti | 4.6.1 | Bundled CPU FAISS + direct cuVS/CUDA 13.2 | 0 | 0 | 0 |

The Fedora and Windows default rows used `FAISSR_FAISS_SOURCE=auto`; their
installation logs confirm that the included source was compiled. The other
bundled rows explicitly selected `bundled`, including hosts with installed
FAISS. The Debian and Windows installed-library rows selected `system`.
Linux distribution tests use isolated user-space images on one host kernel.

CPU rows passed 2,905-2,908 assertions each, with 130-131 skips. Native CUDA
passed 2,942 assertions with 126 skips; direct cuVS passed 2,961 assertions with
124 skips. Every row had zero failed assertions and zero test warnings. Skips
cover unavailable providers and repository-only checks. CUDA smoke tests assert
the resolved GPU provider and compare identifiers with an independent R
reference; a CPU fallback cannot pass them.

## Notes and supplementary tests

A separate Windows `--with-faiss=none` diagnostic profile also completed with
zero errors, warnings or notes: 1,805 assertions passed and 159 tests skipped.
Its FAISS capability is false; this is not a functional FAISS installation and
is not counted among the functional rows above.

Fedora's NOTE concerns flags supplied by its distribution R build, including
`-march=x86-64`, `-mtls-dialect=gnu2` and hardening flags. The package preserves
R's compiler configuration; no warning or NOTE check was disabled to conceal it.

Windows's default-fallback NOTE reports a 5.56-second `backend_info()` example
(0.33 seconds of R user/system CPU time). The installed-library check was
clean. This is an example-timing NOTE, not an installation or search failure.

Separate repository-only configure tests passed 63 assertions. Negative Windows
tests confirmed that invalid explicit numerical flags and missing FAISS under
`--with-faiss=system` are rejected. An additional Windows runtime smoke passed
with Rtools removed from `PATH`. macOS linkage inspection confirmed no runtime
dependency on a separate FAISS dynamic library.

The PDF reference manual compiled successfully (27 pages). Installed package
sizes were about 7.5 MiB on macOS, 7.7 MiB on Windows and 56 MiB on Ubuntu;
the Linux installation retains the active R toolchain's debug information.
These installed sizes are not source-download sizes. Source compilation adds
build time, but requires neither CMake nor a native-library download.

## Changes and scope

The installation first probes suitable installed FAISS, including its version,
ABI and complete LP64 single/double-precision numerical dependencies. Otherwise
it builds the included generic CPU static library with R's compiler and GNU
make. Compilation failure is fatal, not an automatic diagnostic fallback.

The documented upstream patch fixes static Windows declarations, error-message
formats, quantizer initialization, ARM compiler detection and deprecated OpenMP
controls. FAISS diagnostics are buffered from worker threads and emitted through
R's console on the calling thread. Numerical-library settings are tested by
loading a compiled probe, addressing missing symbols such as `ssyrk_` without
replacing R's BLAS. The GPU vignette uses a supported external-query workflow
when only native CUDA is available.

Complete BLAS/LAPACK, OpenMP and the appropriate R development toolchain remain
external requirements. CUDA/cuVS results apply to the prepared provider stack
tested here, not every toolkit, GPU or operating system. No NVIDIA driver,
system R, system BLAS or SSH authorization was changed. No HPC job or manuscript
benchmark was run. Windows ARM64, macOS Intel, Windows NVIDIA GPU, sanitizers and
performance comparisons were not validated in this run.

Reproduction instructions and artifact interpretation are in
[the test-lab guide](package-testing.md). Private host configuration, raw logs,
source snapshots and image checksums remain in the operator's local test lab;
they are not published automatically.
