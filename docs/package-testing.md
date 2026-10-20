# Cross-platform source-package test lab

## Scope and entry points

`.github/package-check/` contains the reusable runner, Linux image definitions,
native Windows/macOS launchers, and an SSH matrix controller. `run.R` also works
for another R source package: omit its optional fourth (package-specific smoke)
argument and provide that package's dependencies in a separate library. The
matrix wrappers and `dependencies.R` currently select the faissR profile.

The native targets are macOS arm64 and Windows x64. Linux distributions run as
read-only Singularity images on the same Linux kernel. This tests distribution
libraries and compilers, not different kernels. Windows ARM64 and CUDA are not
covered by a CPU pass. Never silently substitute a diagnostic build for FAISS.

## First-time setup

Prerequisites: SSH/SCP and Python 3.11+ on the controller; R and its development
toolchain on native hosts; Singularity with working fakeroot on Linux. Existing
SSH authorization is used without changing server settings. Keep a local copy
of `hosts.example.json` outside the repository with actual host paths.

On macOS, use a compatible R, compiler, Fortran and OpenMP installation
as described in the installation documentation. Install missing R dependencies
without changing the default library:

```sh
Rscript .github/package-check/dependencies.R /absolute/lab/dependencies
export R_LIBS_USER=/absolute/lab/dependencies
```

For CRAN-style macOS dependency testing, use the candidate recipe
in `.github/package-check/macos-recipes/` with the public R-macos recipes build
system. Test both arm64 and x86_64 in its **Cook from Recipes** workflow before
proposing the recipe upstream. The resulting prefix is `/opt/R/<architecture>`;
faissR detects it without `FAISS_HOME`. Keep local package-manager testing as a
separate compatibility row, not as the installation path recommended to users.

On Windows, transfer the harness to `r-package-test-lab/harness` in the user's
home and run `windows/prepare.ps1` in PowerShell. Normal installation compiles
the included CPU FAISS source if no compatible installed static library is
found. `windows/build-faiss.ps1` is needed only for a separate installed-library
test row, not for using the package's bundled fallback.
For the default lab location:

```powershell
cd "$HOME/r-package-test-lab/harness"
powershell -NoProfile -ExecutionPolicy Bypass -File windows/prepare.ps1
# Optional: prepare the independent installed-library test row.
powershell -NoProfile -ExecutionPolicy Bypass -File windows/build-faiss.ps1
```

The optional latter command builds CPU FAISS 1.11.0 with Rtools45 MinGW,
not MSVC. It changes PATH only for its process and installs under the lab.
FAISS 1.11.0 requires the included `faiss-1.11.0-mingw.patch`: two Windows-only
plain error messages use `FAISS_THROW_MSG` instead of the variadic formatting
macro. This is a dependency compatibility patch, not a search-algorithm change;
report the patched FAISS build rather than calling it an unmodified binary.
`prepare.ps1` also copies R into a private path without spaces (avoiding Make
failures when Windows lacks short names for Program Files) and installs a
private portable Pandoc 3.6.4 for vignette rendering. It does not replace system R.
Adjust paths in the scripts/config for another installed R/Rtools version.
Do not install or replace R, Rtools, or machine-wide numerical libraries blindly.

On Linux, put images, temporary files, caches and results on a disk with at least
20 GB free. Definitions use mutable upstream package repositories; the resulting
SIF checksum, not merely a definition filename, identifies a frozen environment.
Preserve SIF files with their checksums and build logs for exact reruns.

```sh
export PACKAGE_TEST_ROOT=/large-disk/r-package-test-lab
bash harness/linux/build.sh harness/linux/ubuntu24.def ubuntu24
bash harness/linux/build.sh harness/linux/debian13-bundled.def debian13-r46-bundled
```

The Debian bundled-BLAS definition reuses the image built from
`.github/scripts/Dockerfile.debian-bundled-blas`. If absent, build that Dockerfile
where Docker has sufficient storage, then set `DEBIAN_BASE_IMAGE` to its tag.
`build.sh` exports the image to the lab before entering fakeroot, avoiding Docker
socket group-access problems. It rebuilds compiled CRAN dependencies for custom
R 4.6 because Debian's prebuilt R binaries may have an incompatible R ABI.
Ubuntu uses distribution R and a source build of FAISS 1.11.0 with OpenBLAS.
The minimum-Rcpp profile adds `ubuntu24-rcpp11.def` with Rcpp exactly 1.1.0:

```sh
export UBUNTU_BASE_IMAGE="$PACKAGE_TEST_ROOT/images/ubuntu24.sif"
bash harness/linux/build.sh harness/linux/ubuntu24-rcpp11.def ubuntu24-rcpp110
```

Point the Ubuntu matrix entry at this image. The distribution's older Rcpp is
intentionally retained in the base image for negative dependency testing;
faissR requires Rcpp >= 1.1.0 in both Imports and LinkingTo. The latter is
essential to reject old headers before compilation. The package currently
declares R >= 4.6.0 as its tested submission baseline; this is not an
algorithmic requirement.
The recipes install `checkbashisms` (devscripts) and qpdf for complete checks.
`linux/fedora44-bundled.def` supplies a separate Fedora environment without
installed FAISS. It installs `devscripts-checkbashisms`, the Fedora package
name for that check tool. `linux/fedora-check-tools.def` adds the checker to an
older Fedora image without replacing it. Keep distribution-supplied R compiler
flags intact; record any compiler-flag NOTE separately from package warnings.
`check-tools.def` adds these to an older image without replacing the original.
The Debian custom R must have Cairo support for rendered vignette figures. The
current Docker base recipe includes the development libraries; for an existing
minimal Debian SIF use the following overlay (the variable name is shared with
the Ubuntu local-image builder):

```sh
export UBUNTU_BASE_IMAGE="$PACKAGE_TEST_ROOT/images/debian13-r46-bundled.sif"
bash harness/linux/build.sh harness/linux/debian13-cairo.def debian13-r46-cairo
```

Use the completed Cairo image in the matrix. Do not skip vignette rebuilding
to conceal missing R graphics capabilities.

## Run the complete matrix

The controller requires at least 4 GiB free before a new matrix starts.
`PACKAGE_TEST_MIN_FREE_GIB` sets the operator's budget. Full native artifacts
can be large; this guard never deletes earlier evidence automatically.

After changing an `Rcpp::export` declaration, regenerate and format the native
wrappers before building the source archive:

```sh
Rscript -e 'Rcpp::compileAttributes(".")'
Rscript .github/scripts/format-rcpp-exports.R
```

The second command changes layout only. It keeps generated wrapper signatures
within the package's 80-column source convention without changing their parsed
R expressions.

Build the package using `R CMD build` first. Run from the controller:

```sh
python3 .github/package-check/matrix.py /absolute/lab/hosts.json \
  /absolute/faissR_VERSION.tar.gz /absolute/lab/runs/UNIQUE_RUN \
  --commit SOURCE_COMMIT
```

Targets run sequentially and failures do not suppress later targets. The
controller first snapshots the archive and harness into the local run folder,
so later edits cannot change the inputs of an in-progress matrix. The
controller copies the harness and identical source archive, verifies remote
source checksums, runs checks, and retrieves artifacts. A new output directory
is mandatory. A failed SSH connection is BLOCKED, not a package test failure.
Supply a dirty-tree identifier and retain the patch if testing uncommitted fixes.
No code is pushed and no remote results are deleted automatically.
Each remote run gets a private harness copy, avoiding Windows file locks and
changes to scripts belonging to an active test. Windows `scp_root` is the
SFTP-visible path corresponding to `root` (usually relative to the SSH home).
Use `--only target1,target2` to rerun selected targets in a new output directory.
Each target can specify `"faiss_source": "bundled"` or `"system"` to test
both paths even on a host with installed FAISS. The default is `auto`.
Use an image without FAISS for at least one Linux row. The Windows diagnostic
profile explicitly selects `none`; it cannot silently replace a failed
functional installation.
SSH keepalives detect an unresponsive connection after three missed 30-second
probes. A transport failure is not evidence of a package failure: inspect the
remote `status.csv` and `00check.log` before rerunning or stopping anything.
Linux binds a disk-backed private `/tmp`; the small default Singularity session
filesystem is insufficient for compiling the C++ sources with debug information.

Individual calls are also available:

```sh
bash .github/package-check/macos.sh /absolute/source.tar.gz /absolute/new-output
bash harness/linux/run.sh /absolute/image.sif /absolute/source.tar.gz /absolute/new-output
```

CUDA is a separate strict matrix because CPU Singularity tests do not expose a
GPU. Its toolkit-only images preserve the host driver and test CUDA 12.4, 12.8,
and 13.2 across Ubuntu 22.04, Ubuntu 24.04, and Debian 13, plus the direct cuVS
stack. See `.github/package-check/CUDA.md`:

```sh
export PACKAGE_TEST_ROOT=/large-disk/r-package-test-lab
bash .github/package-check/linux/build-cuda-matrix.sh
bash .github/package-check/linux/run-cuda-matrix.sh \
  /absolute/source.tar.gz "$PACKAGE_TEST_ROOT/runs/cuda-UNIQUE-ID"
```

Every GPU row uses `singularity exec --nv`, requires CUDA at configuration and
runtime, and rejects CPU fallback. Never install or replace the host NVIDIA
driver as part of a package check.

For Windows use `windows/run.ps1 -Archive ... -Output ... -FaissHome ...
-DependencyLibrary ... -FaissSource bundled`. The `diagnostic` profile explicitly
disables FAISS; the default `functional` profile requires installed or bundled
FAISS. A Windows diagnostic check can
validate portability/error reporting but must not be advertised as useful FAISS.

## Evidence and interpretation

### CRAN Compiler And Diagnostic Gates

Include the R-devel Clang/Flang profile in
`linux/debian-clang-devel.def`, not only Debian GCC/GFortran. It intentionally
clears R's OpenMP C++ flag to exercise compiler-based detection. The `float`
dependency is built using the base R GCC toolchain because its own Fortran
flags are not accepted by Flang; faissR itself is compiled with Clang/Flang.
Record both toolchains. The mutable R-hub base must be identified by the
completed SIF checksum and exact R/compiler versions.

`run.R` now retains `install-source/`, including `config.log`, on installation
failure, and records `R-config.txt` and an explicit user Makevars file. It writes
`dependencies.csv` with all declared version constraints before compilation.
The dependency bootstrap checks versions, not only namespace availability,
and retries a required source version if the platform binary is too old.
It writes
`compiler-diagnostics.csv` from raw installation logs and `check-notes.csv`
from every NOTE block. Unreviewed diagnostics and NOTEs fail the harness even
when `R CMD check` returns zero. No warning flag is disabled. Two narrow
compiler classifications are retained visibly: upstream FAISS signedness
comparisons, and macOS's duplicate R Fortran-runtime library list. Neither
initialization nor ABI diagnostics are exempt. These reviewed diagnostics must
still appear in the final report; do not call such a run compiler-warning-free.

Set `PACKAGE_TEST_MANUAL=true` on a TeX-enabled profile for PDF/HTML manual
checks, and `PACKAGE_TEST_INCOMING_REMOTE=true` for network-enabled incoming
checks. Run both, with every Suggests package, before resubmission. They are
off in the default portability mode. Incoming checks also require the R package
`curl` in the test library; it is a checker dependency, not a faissR runtime
dependency. Bootstrap installs it when incoming checks are enabled. These modes are
separate, so a portability PASS is not a full
submission PASS. The initial-submission NOTE is expected; spelling/URL NOTEs
are not exempt. Test these gates with:

```sh
Rscript .github/package-check/test-audit.R
```

The source-only OpenMP configure regression test covers usable R flags,
successful compiler-based flags, and a failed probe with visible diagnostics.

Each run installs into its own library, checks that the smoke test loaded that
installation, records session/capability information, and executes
`R CMD check --as-cran --no-manual`. The default exploratory portability mode
sets `_R_CHECK_FORCE_SUGGESTS_=false`, because suggested packages are not
necessarily available on every architecture. Any resulting NOTE is still
reported and fails the unreviewed-NOTE gate; it is not a submission PASS.
Examples and unit tests remain enabled; vignette source
and installed outputs are inspected, but vignette code is not executed in this
mode. Set
`PACKAGE_TEST_BOOTSTRAP_DEPENDENCIES=true` and
`PACKAGE_TEST_FORCE_SUGGESTS=true` for a networked full-dependency run. A full
submission check with every `Suggests` package remains mandatory on at least one
properly provisioned environment. PDF reference-manual generation needs a
TeX-enabled run with `PACKAGE_TEST_MANUAL=true`. Use
`linux/debian-clang-check-tools.def` on the Clang base image to add the checked
Inconsolata font archive and the R `curl` checker dependency. A missing font
must be provisioned, not worked around with `--no-manual` in the full run.
Network-dependent incoming checks are off unless
`PACKAGE_TEST_INCOMING_REMOTE=true`; local checks do not replace CRAN's service.

Read `status.csv`, `check-summary.txt`, `00check.log`, install logs, smoke logs,
and failing test output. The smoke profile asserts functional capability and
checks Euclidean, cosine and correlation neighbor identifiers against an
independent R reference. GPU tests may legitimately skip on CPU-only profiles.
Report those skips and NOTEs separately from errors/warnings. Keep original
failure evidence, the patch, and a new successful run after fixing a bug.

`matrix.json` reports command-level PASS/FAIL/BLOCKED. Always read the profile:
PASS with `diagnostic` means diagnostic-only success, never functional success.
Record package version, source SHA256, Git state, SIF SHA256, FAISS/R versions,
compiler, BLAS/LAPACK, OS/architecture and date in the final report.
Before reporting a complete matrix, verify the final logs, stages, expected
profile set and shared archive identity with:

```sh
python3 .github/package-check/summarize.py /lab/run-a /lab/run-b \
  --expect target-a,target-b
```

This fails on pending/missing results, unreviewed diagnostics, failed stages,
mixed archives or an omitted target. Keep diagnostic profiles labelled as such.

## Negative regression tests

`linux/test-rcpp-minimum.sh BASE_IMAGE SOURCE_ARCHIVE NEW_OUTPUT` checks that an
image with old Rcpp rejects the declared dependency before native compilation.
The baseline Ubuntu image supplies that intentionally old dependency.
`windows/test-linkage.ps1 -Archive SOURCE_ARCHIVE -Output NEW_OUTPUT
-FaissHome FAISS_PREFIX -DependencyLibrary DEPENDENCY_LIBRARY` verifies
that invalid explicit numerical-library flags and a missing required FAISS
installation fail with the intended messages. These are expected-failure tests;
their success must not be confused with a functional package installation.
