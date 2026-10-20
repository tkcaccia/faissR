## Resubmission of faissR 0.1

The initial-submission version remains 0.1.

### CPU-time corrections

Small examples and vignette calculations now explicitly request one search
worker. Tests use the same budget, including internal helper calls and child R
processes; explicit two-thread correctness tests remain enabled. The test entry
point restores prior BLAS, OpenMP and environment settings, including on failure.
RhpcBLASctl is suggested for controlling the independent numerical-library pool
in tests and vignette rendering. Normal search-thread defaults are unchanged.

Native float32 validation and matrix-layout conversion now have explicit thread
budgets; small buffers are processed serially. Recall diagnostics also request
one worker. Numerical and metadata regression tests exercise these changes.

### Validation

The source archive is 1,583,854 bytes. SHA256:

    14a1fdc5df188c9082297e4e82fadc91380acb46e975394607625bd92508edec

Four functional profiles completed R CMD check --as-cran with zero errors,
zero warnings and no CPU-timing NOTEs:

- macOS arm64: tests used 16 CPU seconds over 20 elapsed seconds.
- Ubuntu 24.04, GCC/GFortran: 23 CPU seconds over 22 elapsed seconds.
- Debian R-devel, Clang/Flang 23 and libc++: 21 CPU seconds over 20 elapsed.
- Windows x64, Rtools: tests completed in 97 elapsed seconds.

Every profile ran examples, unit tests, independent-reference search smoke
tests, both evaluated vignettes, incoming checks, and PDF/HTML manual checks.
All suggested dependencies were present. The example, test and vignette
CPU-to-elapsed thresholds were explicitly enabled at 2.5. The final check
environment left OpenMP unrestricted and selected one BLAS worker before
starting R. A retained fully uncapped control showed that even a `1 + 1`
vignette could trigger excess CPU use during BLAS startup on the high-core
Linux image. Timing NOTEs were not exempted from the checker.

The new-submission NOTE is expected. The R-hub Clang image additionally reports
its inherited -Wp,-D_FORTIFY_SOURCE=3 flag; this is recorded as a toolchain NOTE.
Raw upstream FAISS signedness diagnostics and macOS duplicate Fortran-runtime
link diagnostics remain visible and are audited separately from R check
WARNINGs. Warning flags were not disabled.

### Native installation

The previously corrected configure logic continues to compile/load an OpenMP
probe and reuse the successful flags with the active R toolchain. A compatible
installed FAISS is preferred; the included FAISS 1.11.0 CPU source fallback
requires no native-library download or precompiled binary during installation.
Complete LP64 BLAS/LAPACK and OpenMP remain system requirements. CUDA/cuVS are
optional; system R, BLAS and NVIDIA drivers are never replaced.

No fresh CUDA, Windows ARM64, macOS Intel or diagnostic-only result is claimed
for this archive. Source identities, native-library hashes and final check
completion were verified before producing the report.
