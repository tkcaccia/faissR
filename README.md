# faissR

`faissR` is the nearest-neighbour companion package for `fastEmbedR`.
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

`faissR` distinguishes two GPU/cuVS routes:

- FAISS GPU indexes with NVIDIA cuVS integration, exposed through FAISS-backed
  backends such as `faiss_gpu_ivf_flat` and `faiss_gpu_ivfpq`. When the linked
  FAISS library was built with cuVS support, these paths report backend labels
  such as `GpuIndexIVFFlat_cuVS` and `GpuIndexIVFPQ_cuVS`.
- Direct RAPIDS cuVS calls, exposed through explicit backends such as
  `cuda_cuvs_cagra`, `cuda_cuvs_nndescent`, `cuda_cuvs_bruteforce`,
  `cuda_cuvs_ivf_flat`, and `cuda_cuvs_ivfpq`.

Use `backend_info()` and the `backend` attribute returned by `nn()` to confirm
which route a result used.

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
