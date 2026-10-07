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

On macOS, use a compatible R, compiler, Fortran, FAISS and OpenMP installation
as described in the installation documentation. Install missing R dependencies
without changing the default library:

```sh
Rscript .github/package-check/dependencies.R /absolute/lab/dependencies
export R_LIBS_USER=/absolute/lab/dependencies
```

For Bioconductor/CRAN-style macOS dependency testing, use the candidate recipe
in `.github/package-check/macos-recipes/` with the public R-macos recipes build
system. Test both arm64 and x86_64 in its **Cook from Recipes** workflow before
proposing the recipe upstream. The resulting prefix is `/opt/R/<architecture>`;
faissR detects it without `FAISS_HOME`. Keep local package-manager testing as a
separate compatibility row, not as the installation path recommended to users.

On Windows, transfer the harness to `r-package-test-lab/harness` in the user's
home and run `windows/prepare.ps1`, then `windows/build-faiss.ps1` in PowerShell.
For the default lab location:

```powershell
cd "$HOME/r-package-test-lab/harness"
powershell -NoProfile -ExecutionPolicy Bypass -File windows/prepare.ps1
powershell -NoProfile -ExecutionPolicy Bypass -File windows/build-faiss.ps1
```

The latter builds CPU FAISS 1.11.0 with the existing Rtools45 MinGW compiler,
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
declares R >= 4.6.0 because it targets the Bioconductor 3.24 development
branch; this is a submission baseline rather than an algorithmic requirement.
The recipes install `checkbashisms` (devscripts) and qpdf for complete checks.
`check-tools.def` adds these to an older image without replacing the original.
The Debian custom R must have Cairo support for BiocStyle's SVG device. The
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

After changing an `Rcpp::export` declaration, regenerate and format the native
wrappers before building the source archive:

```sh
Rscript -e 'Rcpp::compileAttributes(".")'
Rscript .github/scripts/format-rcpp-exports.R
```

The second command changes layout only. It keeps generated wrapper signatures
within Bioconductor's 80-column recommendation without changing their parsed R
expressions.

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
-DependencyLibrary ...`. The `diagnostic` profile expects FAISS to be absent;
the default `functional` profile requires it. A Windows diagnostic check can
validate portability/error reporting but must not be advertised as useful FAISS.

## Evidence and interpretation

Each run installs into its own library, checks that the smoke test loaded that
installation, records session/capability information, and executes
`R CMD check --as-cran --no-manual`. The default portability matrix permits an
unavailable `Suggests` package and records the resulting NOTE, because optional
Bioconductor data and analysis stacks are not necessarily available on every
native architecture. Examples and unit tests remain enabled; vignette source
and installed outputs are inspected, but vignette code is not executed in this
mode. Set
`PACKAGE_TEST_BOOTSTRAP_DEPENDENCIES=true` and
`PACKAGE_TEST_FORCE_SUGGESTS=true` for a networked full-dependency run. A full
submission check with every `Suggests` package remains mandatory on at least one
properly provisioned environment and in Bioconductor infrastructure. PDF
reference-manual generation is not part of this portable matrix and needs a
separate TeX-enabled `R CMD check --as-cran` run. Network-dependent CRAN incoming
checks are disabled; this matrix does not replace submission checks or
`BiocCheck`.

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

## Negative regression tests

`linux/test-rcpp-minimum.sh BASE_IMAGE SOURCE_ARCHIVE NEW_OUTPUT` checks that an
image with old Rcpp rejects the declared dependency before native compilation.
The baseline Ubuntu image supplies that intentionally old dependency.
`windows/test-linkage.ps1 -Archive SOURCE_ARCHIVE -Output NEW_OUTPUT
-FaissHome FAISS_PREFIX -DependencyLibrary DEPENDENCY_LIBRARY` verifies
that invalid explicit numerical-library flags and a missing required FAISS
installation fail with the intended messages. These are expected-failure tests;
their success must not be confused with a functional package installation.
