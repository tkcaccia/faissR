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

`faissR` is designed for source installation with a mandatory FAISS C++
library and optional CUDA/RAPIDS libraries [1-3,12-16]. The package code does
not require Python or conda.

## R Package

```r
install.packages("remotes")
remotes::install_github("tkcaccia/faissR")
```

## Required System Dependencies

`faissR` requires:

- R;
- a C++20 compiler;
- a Fortran compiler;
- `Rcpp`;
- a FAISS C++ library installation.

FAISS is mandatory [1-2,16]. CPU-only machines can install and use `faissR` without
CUDA, cuVS, or cuGraph. Optional GPU backends compile only when the matching
headers and libraries are available.

## CRAN And CPU-Only Source Builds

For CRAN-style source builds, provide FAISS through the system compiler/linker
paths, `pkg-config`, or `FAISS_HOME`:

```sh
FAISS_HOME=/path/to/faiss R CMD INSTALL .
```

To force a CPU-only build on a machine where CUDA libraries may also be present:

```sh
FAISS_HOME=/path/to/faiss \
FAISSR_USE_CUDA=0 \
FAISSR_USE_CUVS=0 \
FAISSR_USE_CUGRAPH=0 \
R CMD INSTALL .
```

This build links FAISS and compiles CUDA/cuVS/cuGraph stubs, so explicit GPU
requests fail clearly rather than falling back to CPU.

## macOS

The simplest local installation is usually Homebrew FAISS:

```sh
brew install faiss
FAISS_HOME="$(brew --prefix faiss)" R CMD INSTALL .
```

If FAISS is visible through `pkg-config` and the dynamic linker, `FAISS_HOME`
may not be needed.

## Linux

For a custom FAISS installation:

```sh
FAISS_HOME=/path/to/faiss R CMD INSTALL .
```

Conda or micromamba can provide compatible development libraries, but this is
not a package dependency and is not required by faissR code.

## Optional CUDA, RAPIDS cuVS, And RAPIDS libcugraph

CUDA nearest-neighbour and k-means backends require compatible NVIDIA drivers,
the CUDA toolkit, and either a FAISS GPU build or RAPIDS cuVS [1-3,13-15].

```sh
CUDA_HOME=/usr/local/cuda \
CUVS_HOME=/path/to/cuvs \
FAISS_HOME=/path/to/faiss \
FAISSR_USE_CUDA=1 \
FAISSR_USE_CUVS=1 \
R CMD INSTALL .
```

CUDA graph clustering uses native RAPIDS libcugraph for Louvain and Leiden
[9,11-12]:

```sh
CUDA_HOME=/usr/local/cuda \
CUGRAPH_HOME=/path/to/cugraph \
FAISS_HOME=/path/to/faiss \
FAISSR_USE_CUDA=1 \
FAISSR_USE_CUGRAPH=1 \
R CMD INSTALL .
```

If CUDA/cuVS/cuGraph is not available, install the CPU package with FAISS only.

## Environment Variables

| Variable | Purpose |
| --- | --- |
| `FAISS_HOME` | Prefix containing FAISS headers and libraries. |
| `FAISSR_USE_CUDA` | Enable CUDA/FAISS GPU build paths where available. |
| `FAISSR_USE_CUVS` | Enable RAPIDS cuVS build paths where available. |
| `FAISSR_USE_CUGRAPH` | Enable RAPIDS libcugraph graph-clustering paths where available. |
| `CUDA_HOME` | CUDA toolkit prefix. |
| `CUVS_HOME` | RAPIDS cuVS installation prefix. |
| `CUGRAPH_HOME` | RAPIDS libcugraph installation prefix. |
| `NVCC` | Optional explicit CUDA compiler path. |
| `PKG_CONFIG_PATH` | Helps locate FAISS/cuVS/cuGraph `.pc` files. |
| `LD_LIBRARY_PATH` / `DYLD_LIBRARY_PATH` | Runtime library search path on Linux/macOS. |

The older `FASTEMBEDR_USE_CUDA`, `FASTEMBEDR_USE_CUVS`, and
`FASTEMBEDR_USE_CUGRAPH` variables are accepted by `configure` for compatibility
with existing benchmark scripts. New scripts should use the `FAISSR_*` names.

## Validation

```r
library(faissR)

backend_info()
faiss_available()
faiss_gpu_available()
cuda_available()
cuvs_available()
cugraph_available()
```

For CRAN-style checks, run from a source tarball and use a valid UTF-8 locale.
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
