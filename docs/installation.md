# Installation

[Home](../README.md) |
**Installation** |
[Implementation](implementation.md) |
[Examples](examples.md) |
[Benchmarks](benchmarks.md) |
[API](usage-api.md) |
[Backends](backend-capabilities.md) |
[References](references.md)

`faissR` owns the FAISS/cuVS installation requirements for the
`fastEmbedR` ecosystem.

## R Package

```r
install.packages("remotes")
remotes::install_github("tkcaccia/faissR")
```

## Required System Dependencies

`faissR` requires:

- R;
- a C++20 compiler;
- `Rcpp`;
- a FAISS C++ library installation.

FAISS is mandatory. CPU-only machines can install and use `faissR` without
NVIDIA libraries.

## macOS

The simplest local installation is usually Homebrew FAISS:

```sh
brew install faiss
FAISS_HOME="$(brew --prefix faiss)" R CMD INSTALL .
```

If FAISS is visible through `pkg-config` and the dynamic linker, `FAISS_HOME`
may not be needed.

## Linux CPU

Conda-forge is a reproducible route for CPU FAISS:

```sh
conda create -n faissr -c conda-forge r-base r-rcpp faiss-cpu
conda activate faissr

FAISS_HOME="$CONDA_PREFIX" \
FAISSR_USE_FAISS=1 \
R CMD INSTALL .
```

For a custom FAISS installation:

```sh
FAISS_HOME=/path/to/faiss \
FAISSR_USE_FAISS=1 \
R CMD INSTALL .
```

## Optional CUDA And RAPIDS cuVS

CUDA backends require compatible NVIDIA drivers, the CUDA toolkit, FAISS GPU
when FAISS GPU indexes are used, and RAPIDS cuVS when cuVS indexes are used.

Example build:

```sh
CUDA_HOME=/usr/local/cuda \
CUVS_HOME=/path/to/cuvs \
FAISS_HOME=/path/to/faiss \
FAISSR_USE_FAISS=1 \
FAISSR_USE_CUDA=1 \
FAISSR_USE_CUVS=1 \
R CMD INSTALL .
```

If CUDA/cuVS is not available, install the CPU package with FAISS only. Explicit
CUDA/cuVS backend requests then fail clearly instead of falling back to CPU.

## Environment Variables

| Variable | Purpose |
| --- | --- |
| `FAISS_HOME` | Prefix containing FAISS headers and libraries. |
| `FAISSR_USE_FAISS` | Force FAISS build detection on. FAISS is required. |
| `FAISSR_USE_CUDA` | Enable CUDA/FAISS GPU build paths where available. |
| `FAISSR_USE_CUVS` | Enable RAPIDS cuVS build paths where available. |
| `CUDA_HOME` | CUDA toolkit prefix. |
| `CUVS_HOME` | RAPIDS cuVS installation prefix. |
| `NVCC` | Optional explicit CUDA compiler path. |
| `PKG_CONFIG_PATH` | Helps locate FAISS/cuVS `.pc` files. |
| `LD_LIBRARY_PATH` / `DYLD_LIBRARY_PATH` | Runtime library search path on Linux/macOS. |

The older `FASTEMBEDR_USE_CUDA` and `FASTEMBEDR_USE_CUVS` variables may still
be accepted by configure for compatibility with existing benchmark scripts, but
new scripts should use the `FAISSR_*` names.

## Validation

```r
library(faissR)

backend_info()
faiss_available()
cuda_available()
cuvs_available()
```

For a small runtime test:

```r
x <- scale(as.matrix(iris[, 1:4]))
nn(x, k = 15, backend = "auto", n_threads = 4)
```
