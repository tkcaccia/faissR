# CRAN CPU-Time Corrections

Date: 20 October 2026. Package version remains **0.1**. These are maintainer
checks, not a CRAN acceptance result.

## Source Identity

Source archive: 1,583,854 bytes. SHA-256:

`14a1fdc5df188c9082297e4e82fadc91380acb46e975394607625bd92508edec`

All four final functional profiles tested this identical archive. Earlier
attempts are retained separately and are not pooled with its results.

## Corrections

- Small examples, tests and vignette calculations explicitly request one search
  worker. Existing explicit two-worker correctness tests remain.
- Float32 validation and layout conversion obey an explicit native thread
  budget. Small buffers do not create a parallel team.
- Internal recall diagnostics request one worker instead of detecting all cores.
- Tests control independent BLAS/OpenMP defaults and pass a one-worker budget to
  child R processes. The test entry point restores previous state on exit,
  including failure. Normal user search defaults are unchanged.
- Vignette sessions use one BLAS worker through calculation and formatting.
  Re-enabling a large worker pool mid-render caused additional CPU activity.
- The suggested RhpcBLASctl package supports this check/example setup; it is not
  a new mandatory runtime dependency.

## Final Checks

| Functional profile | Errors | Warnings | NOTEs | Unit-test CPU / elapsed |
| --- | ---: | ---: | ---: | --- |
| macOS arm64 | 0 | 0 | 1 | 16 s / 20 s |
| Ubuntu 24.04, GCC/GFortran | 0 | 0 | 1 | 23 s / 22 s |
| Debian R-devel, Clang/Flang 23, libc++ | 0 | 0 | 2 | 21 s / 20 s |
| Windows x64, Rtools, bundled FAISS | 0 | 0 | 1 | 97 s elapsed |

Every profile ran examples, unit tests, independent-reference search smoke
tests, both rebuilt vignettes, incoming checks and PDF/HTML manual checks with
all suggested dependencies. None reported a CPU-timing NOTE. GPU and
source-only tests can skip in installed CPU-only checks; their reasons remain
in the test logs. A separate source-level run passed 19 thread-budget assertions
and the source-width assertion.

One NOTE in each profile is **New submission**. The Clang image additionally
reports its inherited `-Wp,-D_FORTIFY_SOURCE=3` flag, not a flag added by faissR.
Upstream FAISS signedness and macOS duplicate Fortran-runtime link diagnostics
remain visible in compiler logs and are audited separately. The builds are not
described as compiler-warning-free.

## Environment And Checker

All three R CPU-to-elapsed thresholds were explicitly set to 2.5. The default
test profile leaves OpenMP unrestricted and sets BLAS to one worker before
starting child R processes. A fully uncapped diagnostic on the Linux image
also produced excess CPU use for a vignette containing only `1 + 1`, confirming
that BLAS startup could contribute before any package code ran. That evidence
is retained; timing NOTEs were not whitelisted or their thresholds relaxed.

The checker now records thread settings and rejects all unreviewed timing
NOTEs. Regression fixtures include the three reported resource NOTEs. A parser
checks explicit budgets in small test calls; native OpenMP directives are
required to declare their own worker budget. CI enables the same timing checks.
The final collector checks completion, every required stage, native-library
hashes, exact profile coverage and identical source checksums.

Missing HTML Tidy and Windows TeX tools were provisioned privately. No host
R/BLAS installation, NVIDIA driver, SSH authorization, HPC job or manuscript
benchmark was changed. No fresh CUDA, Windows ARM64, macOS Intel or
diagnostic-only claim is made for this archive.

See [the test procedure](package-testing.md) for commands and interpretation.
