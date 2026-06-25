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
is mandatory. CUDA, FAISS GPU/cuVS integration, RAPIDS cuVS, and RAPIDS
libcugraph are optional and are compiled only when the matching headers and
libraries are available [1-3,12-16,30-33].

The package code does not depend on Python or conda. Conda/micromamba can still
be a convenient way to install compatible C/C++ libraries for development or
benchmarking, especially on Linux GPU systems. For CRAN-style source builds,
the important point is that the compiler and dynamic linker can find FAISS and,
optionally, CUDA/RAPIDS libraries.

## What Gets Installed

| Build type | Required external libraries | faissR result |
|---|---|---|
| CPU/FAISS | FAISS C++ library | FAISS CPU Flat, IVF, IVFPQ, HNSW, NSG/NNDescent where supported, native CPU routes, graph clustering, kNN models, k-means |
| CUDA with FAISS GPU | FAISS built with GPU support, CUDA toolkit | FAISS GPU Flat, IVF, IVFPQ, CAGRA where the linked FAISS build exposes them |
| CUDA with direct cuVS | CUDA toolkit plus RAPIDS cuVS C/C++ library | Direct cuVS brute force, IVF, IVFPQ, CAGRA, HNSW, NN-descent, cuVS k-means. cuVS HNSW builds a CAGRA seed graph and converts it with `cuvsHnswFromCagraWithDataset` using the host dataset and cuVS CPU hierarchy; result metadata marks this wrapper design. |
| CUDA graph clustering | CUDA toolkit plus RAPIDS libcugraph | CUDA Louvain/Leiden in `graph_cluster()` |

GPU requests are explicit. If a GPU backend was not compiled or is unavailable
at runtime, faissR errors instead of silently falling back to CPU.

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

All platforms need:

- R and R development headers;
- `Rcpp`;
- a C++20 compiler;
- a Fortran compiler;
- FAISS headers and library.

`configure` searches common compiler/linker paths, `pkg-config`, and
environment variables. The most portable explicit install is:

```sh
FAISS_HOME=/path/to/faiss R CMD INSTALL .
```

`FAISS_HOME` should be a prefix containing files such as:

```text
/path/to/faiss/include/faiss/Index.h
/path/to/faiss/lib/libfaiss.so      # Linux
/path/to/faiss/lib/libfaiss.dylib   # macOS
/path/to/faiss/lib/faiss.lib        # Windows-style toolchains
```

## macOS

macOS is recommended for CPU/FAISS builds. NVIDIA CUDA is not supported on
modern Apple Silicon/macOS systems, so CUDA/cuVS backends are expected to be
unavailable.

Install R, Xcode command line tools, GNU Fortran if needed by your R setup, and
FAISS:

```sh
xcode-select --install
brew install faiss
```

Then install faissR:

```sh
FAISS_HOME="$(brew --prefix faiss)" R CMD INSTALL .
```

If Homebrew FAISS is already visible to `pkg-config` and the dynamic linker,
`FAISS_HOME` may not be needed. Validate with:

```r
library(faissR)
faiss_available()
backend_info()
```

Expected macOS result: FAISS CPU should be available; CUDA, cuVS, and cuGraph
should report unavailable.

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
- RAPIDS cuVS headers/library if you want direct cuVS routes;
- RAPIDS libcugraph headers/library if you want CUDA Louvain/Leiden.

The source-build install command is:

```sh
CUDA_HOME=/usr/local/cuda \
FAISS_HOME=/path/to/faiss-gpu \
CUVS_HOME=/path/to/rapids \
CUGRAPH_HOME=/path/to/rapids \
FAISSR_USE_CUDA=1 \
FAISSR_USE_CUVS=1 \
FAISSR_USE_CUGRAPH=1 \
R CMD INSTALL .
```

Set only the features you actually have. For example, FAISS GPU without direct
cuVS:

```sh
CUDA_HOME=/usr/local/cuda \
FAISS_HOME=/path/to/faiss-gpu \
FAISSR_USE_CUDA=1 \
FAISSR_USE_CUVS=0 \
FAISSR_USE_CUGRAPH=0 \
R CMD INSTALL .
```

Direct cuVS without cuGraph:

```sh
CUDA_HOME=/usr/local/cuda \
FAISS_HOME=/path/to/faiss \
CUVS_HOME=/path/to/rapids \
FAISSR_USE_CUDA=1 \
FAISSR_USE_CUVS=1 \
FAISSR_USE_CUGRAPH=0 \
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
CUGRAPH_HOME="$ENV_DIR" \
CUDA_HOME=/usr/local/cuda \
FAISSR_USE_CUDA=1 \
FAISSR_USE_CUVS=1 \
FAISSR_USE_CUGRAPH=1 \
R CMD INSTALL .

export LD_LIBRARY_PATH="$ENV_DIR/lib:/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}"
```

This uses the conda environment as a normal C/C++ library prefix. faissR does
not import Python.

## Windows

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
R CMD INSTALL .
```

The FAISS prefix must contain compatible headers and libraries. If the DLL is
not on the runtime search path, add its directory to `PATH` before loading R:

```bat
set PATH=C:\path\to\faiss\bin;%PATH%
```

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

## Environment Variables

| Variable | Purpose |
|---|---|
| `FAISS_HOME` | Prefix containing FAISS headers and libraries. Mandatory when FAISS is not visible through compiler defaults or `pkg-config`. |
| `FAISSR_USE_CUDA` | Set to `1` to request CUDA native/FAISS GPU build paths; set to `0` to force CPU-only stubs. |
| `FAISSR_USE_CUVS` | Set to `1` to request direct RAPIDS cuVS build paths; set to `0` to force cuVS stubs. |
| `FAISSR_USE_CUGRAPH` | Set to `1` to request RAPIDS libcugraph graph clustering; set to `0` to force cuGraph stubs. |
| `CUDA_HOME` | CUDA toolkit prefix, for example `/usr/local/cuda`. |
| `CUVS_HOME` | RAPIDS cuVS prefix containing headers and `libcuvs`. |
| `CUGRAPH_HOME` | RAPIDS libcugraph prefix containing headers and `libcugraph`. |
| `NVCC` | Optional explicit CUDA compiler path. |
| `FAISSR_CUDA_ARCH` | Optional CUDA architectures passed to `nvcc`, for example `80 89`. |
| `FAISSR_CUDA_FLAGS` | Optional extra flags appended to CUDA compilation. |
| `PKG_CONFIG_PATH` | Helps locate FAISS/cuVS/cuGraph `.pc` files. |
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
FAISSR_USE_CUGRAPH=0 \
R CMD INSTALL .
```

This build links FAISS and compiles CUDA/cuVS/cuGraph stubs. Explicit GPU
requests fail clearly.

## Validation

After installation:

```r
library(faissR)

backend_info()
faiss_available()
faiss_gpu_available()
cuda_available()
cuvs_available()
cugraph_available()
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
| `FAISS library not found` during install | FAISS headers/library are not in compiler paths | Set `FAISS_HOME` or `PKG_CONFIG_PATH`; verify `include/faiss/Index.h` and `lib/libfaiss.*` exist. |
| Package installs but cannot load `libfaiss` | Runtime linker cannot find FAISS | Set `LD_LIBRARY_PATH`, `DYLD_LIBRARY_PATH`, or Windows `PATH`. |
| `GLIBCXX_* not found` on Linux | R loaded an older system `libstdc++` before FAISS/RAPIDS libraries | Use a consistent compiler/runtime stack; set `LD_LIBRARY_PATH` and, if necessary for benchmarks, `LD_PRELOAD` to the intended `libstdc++.so.6`. |
| CUDA build cannot find `nvcc` | CUDA toolkit is missing or not on path | Set `CUDA_HOME` and/or `NVCC`; check `nvcc --version`. |
| cuVS routes unavailable | cuVS headers/library were not found at build time | Set `CUVS_HOME`, `FAISSR_USE_CUVS=1`, and runtime `LD_LIBRARY_PATH`. |
| cuGraph routes unavailable | libcugraph headers/library were not found at build time | Set `CUGRAPH_HOME`, `FAISSR_USE_CUGRAPH=1`, and runtime `LD_LIBRARY_PATH`. |
| Windows GPU build is difficult | Native RAPIDS/cuVS C++ libraries are Linux-oriented | Use WSL2 and follow the Linux CUDA instructions. |

## CRAN-Style Check

For CRAN-style checks, build from a source tarball and use a valid UTF-8 locale.
Some R installations emit startup locale warnings under `LC_ALL=C`; those
warnings can be counted during metadata checks even when `DESCRIPTION` itself is
valid.

```sh
R CMD build .
LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8 \
R CMD check --no-manual --no-build-vignettes faissR_0.1.0.tar.gz
```

A CPU-only check should still finish with `Status: OK`; CUDA/cuVS/cuGraph tests
are skipped unless those optional backends were compiled and are available at
runtime.
