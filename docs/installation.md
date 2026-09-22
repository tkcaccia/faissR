# Installation

[Home](../README.md) |
**Installation** |
[Implementation](implementation.md) |
[Examples](examples.md) |
[Benchmarks](benchmarks.md) |
[API](usage-api.md) |
[NN Methods](nn-methods.md) |
[Backends](backend-capabilities.md) |
[References](references.md)

`faissR` is an R source package that links to external system libraries. FAISS
is mandatory. CUDA, FAISS GPU/cuVS integration, and RAPIDS cuVS are optional
for CPU-only builds and are compiled only when the matching headers and
libraries are available [1-3,13-16,30-33]. For a NVIDIA GPU build, request the
GPU features explicitly; then missing CUDA/cuVS
libraries are fatal rather than silently producing a CPU-only installation.

The package code does not depend on Python or conda. Conda/micromamba can still
be a convenient way to install compatible C/C++ libraries for development or
benchmarking, especially on Linux GPU systems. For CRAN-style source builds,
the important point is that the compiler and dynamic linker can find FAISS and,
optionally, CUDA/RAPIDS libraries.

## What Gets Installed

| Build type | Required external libraries | faissR result |
|---|---|---|
| CPU/FAISS | FAISS C++ library | FAISS CPU Flat, IVF, IVFPQ, HNSW, NSG/NNDescent where supported, native CPU routes, kNN models, k-means |
| CUDA with FAISS GPU | FAISS built with GPU support, CUDA toolkit | FAISS GPU Flat, IVF, IVFPQ, CAGRA where the linked FAISS build exposes them |
| CUDA with direct cuVS | CUDA toolkit plus RAPIDS cuVS C/C++ library | Direct cuVS brute force, IVF, IVFPQ, CAGRA, HNSW, NN-descent, cuVS k-means. cuVS HNSW builds a CAGRA seed graph and converts it with `cuvsHnswFromCagraWithDataset` using the host dataset and cuVS CPU hierarchy; result metadata marks this wrapper design. |

GPU requests are explicit. If a GPU backend was not compiled or is unavailable
at runtime, faissR errors instead of silently falling back to CPU.

For submission/build systems such as Bioconductor, the intended CPU build
requires FAISS but not NVIDIA libraries. For NVIDIA GPU users, the intended
strict build uses `FAISSR_REQUIRE_CUDA=1` and, where relevant,
`FAISSR_REQUIRE_CUVS=1`.

On Debian/Ubuntu builders, the mandatory CPU dependency is the FAISS
development package, typically `libfaiss-dev`, and complete LP64 BLAS/LAPACK
development libraries (`libblas-dev` and `liblapack-dev`). Automated systems such as
r-universe resolve this from the package `SystemRequirements` field through
their system-requirements database. If that database or base image does not yet
provide FAISS, the package will fail early at `configure` with a clear
diagnostic rather than building a non-FAISS stub. NVIDIA CUDA/RAPIDS libraries
are intentionally not listed as mandatory CPU-builder requirements; they should
be supplied only by GPU-capable builders or users who explicitly request a GPU
build.

The configure diagnostics use the conventional system-package names
`libfaiss-dev` for Debian-family systems and `faiss-devel` for RPM-family
systems where that package is available. Not every Fedora/RHEL or Alpine
repository currently ships FAISS. On a distribution without a native
development package, build FAISS separately and set `FAISS_HOME`, or pass
`INCLUDE_DIR` and `LIB_DIR`; faissR does not download a substitute library
during installation.

## Known cuVS NN-Descent Issue

Direct RAPIDS cuVS NN-descent can fail on high-dimensional FP32 Euclidean/L2
inputs in affected cuVS builds with `cudaErrorInvalidValue` from
`cuvsNNDescentBuild`. The confirmed cause is a cuVS kernel launch that requires
more than CUDA's default dynamic shared memory per block but does not opt in to
the larger device-supported dynamic shared-memory limit. faissR reports this
case with a specific diagnostic and does not vendor a patched cuVS library.
Users who need direct cuVS NN-descent on such data should update to a patched
cuVS release or rebuild cuVS with the upstream-style fix described in
[the cuVS issue report](cuvs-nndescent-shared-memory-issue.md).

## R Package Install

After the system libraries are installed:

```r
install.packages("remotes")
remotes::install_github("tkcaccia/faissR")
```

For local source checkout:

```sh
R CMD INSTALL .
```

## Required Build Tools

Functional Unix builds need:

- R and R development headers;
- `Rcpp >= 1.1.0`;
- a C++20 compiler;
- a Fortran compiler;
- FAISS headers and library.

The mandatory R dependencies are `Rcpp` and the Bioconductor package
`Biobase`; `methods` ships with R. Package installers normally resolve them
automatically. Building the vignettes also requires `BiocStyle`, `knitr`, and
`rmarkdown`. Native Windows builds use the package's C++ exact fallback and do
not compile the Fortran source.

`configure` searches common compiler/linker paths, `pkg-config`, and
environment variables. It does not invoke `apt`, `dnf`, `apk`, Homebrew, or
another package manager. The most portable explicit installs are:

```sh
FAISS_HOME=/path/to/faiss R CMD INSTALL .
R CMD INSTALL --configure-vars='INCLUDE_DIR=/path/include LIB_DIR=/path/lib' .
```

`FAISS_HOME` should be a prefix containing files such as:

```text
/path/to/faiss/include/faiss/IndexFlat.h
/path/to/faiss/lib/libfaiss.so      # Linux
/path/to/faiss/lib/libfaiss.dylib   # macOS
/path/to/faiss/lib/libfaiss.a       # static Unix or Rtools build
```

## Debian With R's Bundled Numerical Libraries

R's bundled `libRblas` and `libRlapack` do not include all the single-precision
routines needed by FAISS. In particular, merely adding R's `BLAS_LIBS` to the
link command does not resolve `ssyrk_`. Static FAISS archives require these
dependencies to be linked by the package using them.

```sh
sudo apt-get install libfaiss-dev libblas-dev liblapack-dev
R CMD INSTALL faissR_0.99.48.tar.gz
```

On Linux, configure compiles a small FAISS client and loads it in a fresh R
process with immediate symbol resolution. If R's configured libraries are
insufficient, it tests complete external BLAS/LAPACK providers and adds the
working libraries after FAISS, retaining R's own link flags. Failures stop
installation early and are recorded in `config.log`. This checks symbol
availability, not every possible provider ABI or numerical operation.

An administrator can explicitly select an ABI-compatible LP64 provider:

```sh
FAISSR_NUMERICAL_LIBS="-llapack -lblas" R CMD INSTALL faissR_0.99.48.tar.gz
```

For nonstandard prefixes, include `-L` and runtime-search-path flags in that
variable. A supplied value must pass the check; it is not silently replaced.
Do not replace R's `libRblas` or `libRlapack`, and do not use ILP64 libraries
with an LP64 FAISS build. CUDA is not required for this CPU installation.

## Optional Linux Performance Configuration

For Linux CPU performance, an ABI-compatible LP64 OpenBLAS build for FAISS is
optional. It can affect dense linear algebra, but does not guarantee faster
graph search. Preserve R's configured BLAS/LAPACK/Fortran link flags; do not
replace `libRblas` by hand. Benchmark method, batch size, and thread settings
on the intended workload. Start with one BLAS thread when using FAISS OpenMP
parallelism to avoid oversubscription (`OPENBLAS_NUM_THREADS=1` for pthread
OpenBLAS; OpenMP builds use `OMP_NUM_THREADS`). See the
[installation vignette](../vignettes/installation.Rmd#optional-linux-performance-configuration)
for the optional performance configuration and the
[OpenBLAS runtime documentation](https://www.openmathlib.org/OpenBLAS/docs/runtime_variables/)
for provider-specific thread controls.

## macOS CPU Installation

macOS supports CPU/FAISS builds. NVIDIA CUDA is not supported on modern Apple
Silicon/macOS systems, so CUDA/cuVS backends are expected to be unavailable.

Bioconductor and CRAN macOS binary builders use the
[R-macos recipes](https://github.com/R-macos/recipes) system for static system
dependencies. The faissR repository contains a candidate FAISS recipe under
`.github/package-check/macos-recipes/`; it must pass the upstream arm64 and
x86_64 recipe builds before it is proposed for the shared builder. `configure`
detects recipe prefixes under `/opt/R/arm64` or `/opt/R/x86_64` automatically.

A source build outside that environment must provide compatible FAISS and
OpenMP installations without asking `configure` to modify the system. Set
`FAISS_HOME` and `LIBOMP_HOME`, or use `INCLUDE_DIR` and `LIB_DIR` for FAISS:

```sh
FAISS_HOME=/path/to/faiss \
LIBOMP_HOME=/path/to/openmp \
FAISSR_REQUIRE_FAISS=1 \
R CMD INSTALL .
```

If FAISS is visible through `pkg-config` and OpenMP is in the R toolchain
prefix, explicit variables may not be needed. Validate with:

```r
library(faissR)
faiss_available()
backend_info()
```

Expected macOS result: FAISS CPU should be available; CUDA and cuVS
should report unavailable.

A pre-existing conda or mamba environment can also provide a local CPU FAISS
prefix for source installation:

```sh
conda install -c conda-forge faiss-cpu libomp
export FAISS_HOME="$CONDA_PREFIX"
export LIBOMP_HOME="$CONDA_PREFIX"
R CMD INSTALL .
```

The configure script also detects `CONDA_PREFIX` directly when FAISS and libomp
are installed in the active environment. It never creates or modifies that
environment.

## Linux CPU/FAISS

Install R development tools, a C++20 compiler, Fortran, and FAISS. The exact
package manager command depends on your distribution. A source-build pattern is:

```sh
git clone https://github.com/facebookresearch/faiss.git
cd faiss
cmake -B build \
  -DFAISS_ENABLE_GPU=OFF \
  -DFAISS_ENABLE_PYTHON=OFF \
  -DBUILD_TESTING=OFF \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build -j"$(nproc)"
cmake --install build --prefix "$HOME/.local/faiss-cpu"
```

Then install faissR:

```sh
FAISS_HOME="$HOME/.local/faiss-cpu" R CMD INSTALL .
```

If FAISS is installed in a non-standard prefix, runtime loading may also need:

```sh
export LD_LIBRARY_PATH="$HOME/.local/faiss-cpu/lib:${LD_LIBRARY_PATH:-}"
```

## Linux CUDA, FAISS GPU, And cuVS

CUDA builds require:

- a compatible NVIDIA driver;
- the NVIDIA CUDA toolkit, including `nvcc`;
- FAISS built with GPU support if you want FAISS GPU indexes;
- RAPIDS cuVS headers/library if you want direct cuVS routes.

`faissR` never installs or changes an NVIDIA driver. On Debian or Ubuntu, do
not add a distribution CUDA metapackage to a server whose working driver is
managed separately. With NVIDIA's package repository, use a versioned
`cuda-toolkit-X-Y` package for development tools without a driver. The `cuda`
and `cuda-runtime-X-Y` metapackages can include driver packages. Always inspect
a package-manager simulation before changing a shared GPU host.

Multiple toolkit versions can coexist in separate prefixes. Select one with
`CUDA_HOME`, and use FAISS GPU and cuVS libraries built for a compatible CUDA
stack. Configuration compiles and links a small CUDA probe before building the
package. Toolkit headers and libraries stored below
`CUDA_HOME/targets/<platform>/` are detected in addition to the conventional
`CUDA_HOME/include`, `CUDA_HOME/lib`, and `CUDA_HOME/lib64` directories. This
also supports the layout used by conda and micromamba CUDA toolkits.

The source-build install command is:

```sh
CUDA_HOME=/usr/local/cuda \
FAISS_HOME=/path/to/faiss-gpu \
CUVS_HOME=/path/to/rapids \
FAISSR_REQUIRE_CUDA=1 \
FAISSR_REQUIRE_CUDA_RUNTIME=1 \
FAISSR_REQUIRE_CUVS=1 \
R CMD INSTALL .
```

`FAISSR_REQUIRE_CUDA_RUNTIME=1` additionally requires a usable GPU during
configuration. Omit it for a build container with no attached device. The
installed package reports compiled-toolkit, runtime, driver API, and compute
capability information through `backend_info()`.

Set only the features you actually have. For example, FAISS GPU without direct
cuVS:

```sh
CUDA_HOME=/usr/local/cuda \
FAISS_HOME=/path/to/faiss-gpu \
FAISSR_REQUIRE_CUDA=1 \
FAISSR_USE_CUVS=0 \
R CMD INSTALL .
```

Direct cuVS:

```sh
CUDA_HOME=/usr/local/cuda \
FAISS_HOME=/path/to/faiss \
CUVS_HOME=/path/to/rapids \
FAISSR_REQUIRE_CUDA=1 \
FAISSR_REQUIRE_CUVS=1 \
R CMD INSTALL .
```

If libraries are in non-standard prefixes, set the runtime path before loading
R:

```sh
export LD_LIBRARY_PATH="/path/to/faiss/lib:/path/to/rapids/lib:/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}"
```

When using a conda/micromamba environment only as a library provider, the
important prefix is the environment itself:

```sh
ENV_DIR="$HOME/.local/share/mamba/envs/faissr-gpu"

FAISS_HOME="$ENV_DIR" \
CUVS_HOME="$ENV_DIR" \
CUDA_HOME=/usr/local/cuda \
FAISSR_REQUIRE_CUDA=1 \
FAISSR_REQUIRE_CUVS=1 \
R CMD INSTALL .

export LD_LIBRARY_PATH="$ENV_DIR/lib:/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}"
```

This uses the conda environment as a normal C/C++ library prefix. faissR does
not import Python.

## Windows

`faissR` supports Windows package installation and native CPU FAISS execution.
Automated Windows builders may not provide a compatible FAISS development
library. On those builders, the package compiles diagnostic stubs and reports
FAISS as unavailable instead of failing with invalid `/include` and `/lib`
paths. A functional Windows FAISS route requires an Rtools-compatible library;
set `FAISSR_REQUIRE_FAISS=1` to make its absence a configuration error.

Windows users can use a native Rtools-compatible CPU build, or WSL2 for the
Linux-style CPU and CUDA installation paths.

### Windows CPU/FAISS

Use R for Windows plus Rtools with a C++20-capable compiler. FAISS is not
vendored, so you must provide a FAISS build that matches your R/Rtools
toolchain.

Recommended practical options:

1. Build FAISS from source with CMake using the same compiler family used by
   Rtools, then set `FAISS_HOME` to the install prefix.
2. Use Windows Subsystem for Linux (WSL2) and follow the Linux instructions.
   This is often simpler for FAISS and is the recommended path if you also want
   GPU/cuVS.

Native Windows CPU install shape:

```bat
set FAISS_HOME=C:\path\to\faiss
set FAISSR_REQUIRE_FAISS=1
R CMD INSTALL .
```

The FAISS prefix must contain `include/faiss/IndexFlat.h` (or
`Library/include/faiss/IndexFlat.h`) and a MinGW/Rtools-compatible
`libfaiss.a` or `libfaiss.dll.a`. A Microsoft Visual C++ `.lib` file is not
assumed to be ABI-compatible with Rtools. If a FAISS DLL is not on the runtime
search path, add its directory to `PATH` before loading R:

```bat
set PATH=C:\path\to\faiss\bin;%PATH%
```

Windows configure compiles and loads a DLL that references the required
single- and double-precision BLAS/LAPACK routines. R's bundled libraries may
not export the single-precision symbols. When necessary, configure adds
complete Rtools `-llapack -lblas` libraries while preserving R's own numerical
and Fortran link flags. Use `FAISSR_NUMERICAL_LIBS` for an explicit compatible
LP64 library selection; failed probes are recorded in `config.log` and abort
the functional build. The [cross-platform test lab](package-testing.md) includes
a tested Rtools FAISS recipe and isolated Windows checks.

### Windows CUDA/cuVS

For CUDA/cuVS, use WSL2 Linux unless you are maintaining your own native
Windows builds of FAISS GPU and RAPIDS libraries. RAPIDS cuVS prebuilt C/C++
packages are Linux-oriented, and WSL2 is the practical Windows route for GPU
benchmarks [3,31-33].

Inside WSL2:

1. Install the NVIDIA Windows driver with WSL support.
2. Install a Linux distribution through WSL2.
3. Install the CUDA toolkit and external FAISS/RAPIDS libraries inside the WSL2
   Linux environment.
4. Follow the Linux CUDA instructions above.

Validate inside WSL2:

```sh
nvidia-smi
Rscript -e 'library(faissR); print(backend_info())'
```

## WebAssembly / webR

`faissR` is native nearest-neighbour infrastructure around FAISS, CUDA, and
RAPIDS libraries. These libraries are not available as normal webR/WebAssembly
system libraries. When r-universe attempts a WebAssembly binary build,
`configure` detects the `wasm32-unknown-emscripten` target and builds diagnostic
stubs instead of using host `/usr/include` FAISS headers, which would corrupt
the Emscripten sysroot. The resulting WASM artifact can report backend
availability, but FAISS/CUDA/cuVS methods are unavailable there. Supported
Linux and macOS source builds still require real FAISS.

## Environment Variables

| Variable | Purpose |
|---|---|
| `FAISS_HOME` | Prefix containing FAISS headers and libraries. Mandatory when FAISS is not visible through compiler defaults or `pkg-config`. |
| `INCLUDE_DIR`, `LIB_DIR` | Conventional configure overrides for separate FAISS include and library directories. Both must be set together. |
| `FAISSR_REQUIRE_FAISS` | Set to `1` in production or CI to reject diagnostic-only builds when a functional FAISS library is required. |
| `FAISSR_NUMERICAL_LIBS` | Explicit link flags for complete, ABI-compatible LP64 BLAS/LAPACK dependencies of FAISS. On Linux and Windows the flags must pass a compile/load check. R's own numerical-library flags are retained. |
| `FAISSR_RUNIVERSE_MACOS_STUBS` | r-universe/BiocStaging macOS-only diagnostic switch. Defaults to `1`, allowing diagnostic stubs only on those macOS binary workers when FAISS is absent. Set to `0` to make that worker fail instead. User macOS installs are unaffected and still require FAISS. |
| `LIBOMP_HOME` or `FAISSR_LIBOMP_HOME` | macOS OpenMP prefix containing `include/omp.h` and `lib/libomp.*`. Recipe and R toolchain prefixes are detected automatically. |
| `CONDA_PREFIX` | Active conda/mamba prefix. Used only as a passive fallback when `faiss-cpu` and `libomp` are already installed there. |
| `FAISSR_USE_CUDA` | Set to `1` to request CUDA native/FAISS GPU build paths; set to `0` to force CPU-only stubs. |
| `FAISSR_USE_CUVS` | Set to `1` to request direct RAPIDS cuVS build paths; set to `0` to force cuVS stubs. |
| `FAISSR_REQUIRE_CUDA` | Strict alias for a NVIDIA GPU build. Set to `1` to make missing CUDA toolkit/`nvcc` fatal at configure time. |
| `FAISSR_REQUIRE_CUDA_RUNTIME` | Set to `1` to require a visible, usable CUDA device during configuration as well as a successful compiler/linker probe. Leave unset on build-only hosts. |
| `FAISSR_REQUIRE_CUVS` | Strict direct cuVS request. Set to `1` to make missing RAPIDS cuVS fatal at configure time. |
| `CUDA_HOME` | CUDA toolkit prefix, for example `/usr/local/cuda`. |
| `CUVS_HOME` | RAPIDS cuVS prefix containing headers and `libcuvs`. |
| `NVCC` | Optional explicit CUDA compiler path. |
| `FAISSR_CUDA_ARCH` | Optional space-separated numeric CUDA architectures passed to `nvcc`, for example `80 89 120`. |
| `FAISSR_CUDA_PTX_ARCH` | Optional PTX target. By default, the highest value in `FAISSR_CUDA_ARCH` is also retained as forward-compatible PTX. Use `none` only when PTX must be disabled deliberately. |
| `FAISSR_CUDA_FLAGS` | Optional extra flags appended to CUDA compilation. |
| `PKG_CONFIG_PATH` | Helps locate FAISS/cuVS `.pc` files. |
| `LD_LIBRARY_PATH` | Linux runtime library search path. |
| `DYLD_LIBRARY_PATH` | macOS runtime library search path when needed. |
| `PATH` | Windows runtime DLL search path. |
| `LD_PRELOAD` | Optional Linux preload for cases where the wrong `libstdc++.so.6` is loaded before FAISS/RAPIDS libraries. |
| `FAISSR_LD_PRELOAD` | Benchmark-launcher convenience variable forwarded to worker R processes as `LD_PRELOAD`. |

## CPU-Only Build On A GPU Machine

To guarantee a CPU-only build even when CUDA libraries are installed:

```sh
FAISS_HOME=/path/to/faiss \
FAISSR_USE_CUDA=0 \
FAISSR_USE_CUVS=0 \
R CMD INSTALL .
```

This build links FAISS and compiles CUDA/cuVS stubs. Explicit GPU
requests fail clearly.

## Functional Versus Diagnostic Builds

A functional build links a compatible FAISS library. It must satisfy:

```r
library(faissR)
stopifnot(faiss_available())
x <- matrix(seq_len(240), ncol = 6)
z <- nn(x, k = 3, method = "exact", backend = "cpu")
stopifnot(identical(dim(z$indices), c(nrow(x), 3L)))
```

A diagnostic-only build is emitted only on explicitly recognized automated
targets without a usable native FAISS library. It can load and report
capabilities, but `faiss_available()` is false and search calls fail with a
provider diagnostic. A diagnostic artifact is not evidence of functional
platform support. Use `FAISSR_REQUIRE_FAISS=1` in CI and production builds to
prevent accidental diagnostic-only installation.

## Tested Configurations

A functional source build was installed and smoke-tested on macOS arm64 with R 4.6.0,
FAISS 1.14.3, clang 22.1.1, GNU Fortran 12.2.0, and libomp 22.1.x. The
publication CUDA environment used Debian 13, R 4.5.3, FAISS
1.14.3, cuVS 26.06, CUDA 13.2, and an NVIDIA L40S with driver 595.58.03. The
exact compiler executable used to construct that frozen CUDA image was not
retained, so this runtime combination is reported without inventing compiler
provenance. Repository CI definitions build FAISS 1.14.3 from source on Linux;
the macOS dependency path is being migrated to the R-macos recipe candidate.

## Validation

After installation:

```r
library(faissR)

backend_info()
faiss_available()
faiss_gpu_available()
cuda_available()
cuvs_available()
```

Minimum expected CPU build:

```r
stopifnot(faiss_available())
```

Expected GPU build checks:

```r
stopifnot(faiss_available())
backend_info()
```

`backend_info()` should show which optional CUDA routes were compiled and
available at runtime.

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `FAISS library not found` during install | FAISS headers/library are not in compiler paths | Set `FAISS_HOME` or `PKG_CONFIG_PATH`; verify `include/faiss/IndexFlat.h` and `lib/libfaiss.*` exist. |
| Package installs but cannot load `libfaiss` | Runtime linker cannot find FAISS | Set `LD_LIBRARY_PATH`, `DYLD_LIBRARY_PATH`, or Windows `PATH`. |
| Package loading reports an undefined numerical symbol such as `ssyrk_` | R's bundled numerical libraries can lack single-precision FAISS dependencies | Install complete LP64 BLAS/LAPACK development libraries and use the current Linux or Windows compile/load check; inspect `config.log` or set `FAISSR_NUMERICAL_LIBS` for a custom provider. R's link flags alone are not sufficient on every installation. |
| `GLIBCXX_* not found` on Linux | R loaded an older system `libstdc++` before FAISS/RAPIDS libraries | Use a consistent compiler/runtime stack; set `LD_LIBRARY_PATH` and, if necessary for benchmarks, `LD_PRELOAD` to the intended `libstdc++.so.6`. |
| CUDA build cannot find `nvcc` | CUDA toolkit is missing or not on path | Set `CUDA_HOME` and/or `NVCC`; check `nvcc --version`. |
| CUDA compiler/linker probe fails | Compiler, headers, runtime libraries, or architecture flags come from incompatible toolkit installations | Select one `CUDA_HOME`, inspect `config.log`, and do not replace the host driver as a package-install workaround. |
| CUDA runtime probe reports no device | The build host has no passed-through GPU, or the driver cannot use the selected runtime | Use `nvidia-smi`; compare toolkit/runtime/driver metadata in `backend_info()`. Omit `FAISSR_REQUIRE_CUDA_RUNTIME=1` only for an intentional build-only host. |
| cuVS routes unavailable | cuVS headers/library were not found at build time | Set `CUVS_HOME`, `FAISSR_USE_CUVS=1`, and runtime `LD_LIBRARY_PATH`. |
| Windows GPU build is difficult | Native RAPIDS/cuVS C++ libraries are Linux-oriented | Use WSL2 and follow the Linux CUDA instructions. |

## Bioconductor-Style Check

For Bioconductor-style checks, build from a source tarball and use a valid
UTF-8 locale. Some R installations emit startup locale warnings under `LC_ALL=C`;
those warnings can be counted during metadata checks even when `DESCRIPTION`
itself is valid.

```sh
R CMD build .
LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8 \
R CMD check --as-cran faissR_0.99.48.tar.gz
```

Bioconductor submission checks are run in addition to `R CMD check`:

```r
BiocCheck::BiocCheckGitClone(".")
BiocCheck::BiocCheck("faissR_0.99.48.tar.gz", `new-package` = TRUE)
```

A CPU-only check should still finish with `Status: OK` once FAISS is installed;
CUDA/cuVS tests are skipped unless those optional backends were
compiled and are available at runtime. New Bioconductor submissions also
require the maintainer to be registered on the Bioconductor Support Site and
subscribed to the bioc-devel mailing list. FAISS is a mandatory external system
dependency, so the submitted package and review notes should make the FAISS
installation path clear for the Bioconductor build system. NVIDIA libraries
should not be required on CPU-only Bioconductor builders, but GPU builders
should set the strict `FAISSR_REQUIRE_*` variables to avoid accidental CPU-only
builds.

For r-universe/BiocStaging logs, a failure that installs `nvidia-cuda-dev` but
not `libfaiss-dev` indicates a system-requirements resolver issue rather than a
package compile error: FAISS is mandatory for functional builds, whereas
CUDA/RAPIDS is optional unless a GPU build is requested.

Until the upstream r-universe system-requirements database includes a FAISS
rule, the repository includes a top-level `.prepare` hook for r-universe source
builds. The hook installs `libfaiss-dev`, `libblas-dev`, and `liblapack-dev`
on Debian/Ubuntu before `R CMD build`.
It is excluded from the package tarball with `.Rbuildignore`; regular package
installation still relies on normal system-library discovery through
`configure`.

Bioconductor GPU builders are requested through repository metadata, not by
making CUDA mandatory in `DESCRIPTION`. faissR therefore uses:

```text
biocViews: ..., GPU, ...
```

and a top-level `.BBSoptions` file:

```text
GPU_reliance: optional
```

This follows the GPU-optional package pattern: regular Bioconductor checks can
build the CPU/FAISS package, while GPU build machines can exercise CUDA/cuVS
tests when the NVIDIA stack is present.
