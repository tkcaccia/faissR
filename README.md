# faissR

`faissR` is an R package for fast nearest-neighbour search, graph construction,
kNN prediction, and k-means. It is designed as the nearest-neighbour companion
package for `fastEmbedR`, but the package can also be used directly whenever an
R workflow needs explicit CPU, FAISS, CUDA, or cuVS-backed KNN methods.

The package builds around [FAISS](https://faiss.ai/index.html), Meta FAIR's
C++ library for efficient similarity search and clustering of dense vectors.
FAISS provides exact and approximate indexes, GPU implementations, batched
search, Euclidean/L2 search, and maximum inner-product search. `faissR` exposes
those capabilities through R-friendly return objects while keeping explicit
backend requests honest: if a GPU backend is requested and unavailable, the call
fails instead of silently falling back to CPU.

It contains the KNN/search side of the workflow:

- `nn()` for exact CPU, FAISS, RcppHNSW, optional cuVS/CUDA, and small
  native spatial search paths;
- `candidate_knn()` for exact top-k ranking inside supplied candidate rows;
- `knn_graph()` for original-space or embedding-space `igraph` graph creation;
- `fast_kmeans()` for CPU/FAISS/cuVS k-means;
- `knn_fit()`, `faiss.fit()`, `cuvs.fit()`, `predict()`, and
  `predict_proba()` for kNN classification/regression.

`fastEmbedR` now depends on `faissR` for neighbour search and keeps UMAP and
openTSNE embedding optimizers.

## Installation

```r
install.packages("remotes")
remotes::install_github("tkcaccia/faissR")
```

FAISS is a required system dependency and is not vendored. `faissR` compiles
with C++20 because recent FAISS headers use C++20 syntax. RAPIDS cuVS/CUDA is
optional, so machines without an NVIDIA GPU can still compile and use the CPU
FAISS backends.

On macOS, the simplest route is usually:

```sh
brew install faiss
R CMD INSTALL .
```

On Linux or custom installations, set `FAISS_HOME` if FAISS is not visible via
`pkg-config` or standard library paths:

```sh
FAISS_HOME=/path/to/faiss R CMD INSTALL .
```

Optional CUDA/cuVS builds are enabled only when requested or auto-detected:

```sh
CUDA_HOME=/path/to/cuda CUVS_HOME=/path/to/cuvs \
FAISSR_USE_CUDA=1 FAISSR_USE_CUVS=1 R CMD INSTALL .
```

The old `FASTEMBEDR_USE_CUDA` and `FASTEMBEDR_USE_CUVS` environment variables
are still accepted by `configure` for compatibility with existing benchmark
scripts. FAISS is always required.

## FAISS GPU with cuVS

`faissR` distinguishes two GPU/cuVS routes. This distinction matters for
benchmarking and methods sections, because FAISS can call cuVS internally while
the package can also call RAPIDS cuVS directly.

- FAISS GPU indexes with NVIDIA cuVS integration, exposed through FAISS-backed
  backends such as `faiss_gpu_ivf_flat`, `faiss_gpu_ivfpq`, and
  `faiss_gpu_cagra`. When the linked FAISS library was built with cuVS support,
  these paths report backend labels such as `GpuIndexIVFFlat_cuVS`,
  `GpuIndexIVFPQ_cuVS`, and `GpuIndexCagra_cuVS`.
- Direct RAPIDS cuVS calls, exposed through explicit backends such as
  `cuda_cuvs_cagra`, `cuda_cuvs_nndescent`, `cuda_cuvs_bruteforce`,
  `cuda_cuvs_ivf_flat`, and `cuda_cuvs_ivfpq`.

Use `backend_info()` and the `backend` attribute returned by `nn()` to confirm
which route a result used.

The FAISS/cuVS route follows the direction described by Meta and NVIDIA for
FAISS 1.10 and later: IVF-Flat, IVF-PQ, and CAGRA are exposed as FAISS GPU
indexes while cuVS provides accelerated GPU implementations underneath. The
direct cuVS route is useful for testing the RAPIDS C API directly, but paper
benchmarks should label it separately from FAISS-integrated cuVS results.

Common explicit backends:

| Backend | Route | Notes |
| --- | --- | --- |
| `faiss_flat_l2` | FAISS CPU | Exact L2 search. |
| `faiss_ivf` | FAISS CPU | Inverted-file approximate L2 search. |
| `faiss_ivfpq` | FAISS CPU | IVF with product quantization. |
| `faiss_hnsw`, `faiss_nsg`, `faiss_nndescent` | FAISS CPU | Graph/ANN indexes when supported by the linked FAISS build. |
| `faiss_gpu_flat_l2` | FAISS GPU | Exact GPU L2 search. |
| `faiss_gpu_ivf_flat` | FAISS GPU + cuVS when available | FAISS-owned IVF-Flat GPU index. |
| `faiss_gpu_ivfpq` | FAISS GPU + cuVS when available | FAISS-owned IVF-PQ GPU index. |
| `faiss_gpu_cagra` | FAISS GPU + cuVS when available | FAISS-owned CAGRA graph index. |
| `cuda_cuvs_bruteforce` | Direct RAPIDS cuVS | Exact cuVS L2 search. |
| `cuda_cuvs_ivf_flat`, `cuda_cuvs_ivfpq`, `cuda_cuvs_nndescent` | Direct RAPIDS cuVS | Explicit direct-cuVS comparison paths. |

For no-inner-product benchmarks, use L2/cosine-capable methods only. FAISS
inner-product indexes solve a different objective, maximum dot-product search,
and should be benchmarked separately.

## Example

```r
library(faissR)

x <- scale(as.matrix(iris[, 1:4]))
knn <- nn(x, k = 15, backend = "auto", n_threads = 4)
knn

if (requireNamespace("igraph", quietly = TRUE)) {
  g <- knn_graph(knn, k = 15, weight = "snn")
  cl <- igraph::cluster_louvain(g, weights = igraph::E(g)$weight)
  table(igraph::membership(cl))
}
```

## With fastEmbedR

```r
library(faissR)
library(fastEmbedR)

x <- scale(as.matrix(iris[, 1:4]))
knn <- faissR::nn(x, k = 15, backend = "auto")

layout_umap <- fastEmbedR::umap_knn(knn)
layout_tsne <- fastEmbedR::opentsne_knn(knn, init_data = x)
```

Use `backend_info()` to inspect available NN backends. `faissR` never silently
runs CPU code when an explicit GPU backend is requested.

## Acknowledgements

`faissR` is an R interface layer and benchmarking companion around major open
source systems:

- [FAISS](https://faiss.ai/index.html), developed primarily by Meta FAIR, is the
  required C++ similarity-search and clustering library used by the FAISS
  backends.
- The FAISS documentation describes FAISS as supporting efficient search over
  dense vectors, GPU implementations, batched search, approximate speed/accuracy
  tradeoffs, and maximum inner-product search.
- [RAPIDS cuVS](https://github.com/rapidsai/cuvs) and NVIDIA CUDA provide the
  optional direct GPU vector-search backends.
- Meta and NVIDIA's FAISS/cuVS integration work explains why IVF-Flat, IVF-PQ,
  and CAGRA should be treated as important GPU methods in FAISS-backed
  benchmarks: [Accelerating GPU indexes in Faiss with NVIDIA cuVS](https://engineering.fb.com/2025/05/08/data-infrastructure/accelerating-gpu-indexes-in-faiss-with-nvidia-cuvs/).

Please cite and acknowledge FAISS, RAPIDS cuVS/NVIDIA CUDA, and the relevant
nearest-neighbour methods when publishing results obtained with `faissR`.
