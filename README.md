# faissR

`faissR` provides native nearest-neighbour search, graph construction, graph clustering, kNN models, and k-means for R workflows that need mandatory FAISS support and optional NVIDIA CUDA/cuVS acceleration.

It contains the KNN/search side of the workflow:

- `nn()` for exact native CPU references, FAISS CPU/GPU indexes, RcppHNSW fallback, optional RAPIDS cuVS/CUDA indexes, and small native spatial search paths;
- `candidate_knn()` for exact top-k ranking inside supplied candidate rows;
- `knn_graph()` for native original-space or embedding-space weighted graph creation;
- `graph_cluster()` for native C++/OpenMP random-walk, Louvain, and Leiden-style clustering on KNN graphs;
- `fast_kmeans()` for CPU, FAISS CPU/GPU, and cuVS k-means;
- `knn_fit()`, `faiss.fit()`, `cuvs.fit()`, and `predict()` for kNN classification/regression, including class probabilities with `predict(type = "prob")`.

`fastEmbedR` now depends on `faissR` for neighbour search and keeps UMAP and
openTSNE embedding optimizers.

## Installation

```r
install.packages("remotes")
remotes::install_github("tkcaccia/faissR")
```

FAISS is a required system dependency and is not vendored. `faissR` compiles
with C++20 because recent FAISS headers use C++20 syntax. RAPIDS cuVS/CUDA and RAPIDS libcugraph are optional, so machines without an
NVIDIA GPU can still compile and use the CPU FAISS backends.

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

Optional CUDA graph clustering uses native RAPIDS libcugraph when available:

```sh
CUDA_HOME=/path/to/cuda CUGRAPH_HOME=/path/to/cugraph \
FAISSR_USE_CUDA=1 FAISSR_USE_CUGRAPH=1 R CMD INSTALL .
```

The old `FASTEMBEDR_USE_CUDA`, `FASTEMBEDR_USE_CUVS`, and
`FASTEMBEDR_USE_CUGRAPH` environment variables are still accepted by
`configure` for compatibility with existing benchmark scripts. FAISS is always
required.

## FAISS GPU with cuVS

`faissR` distinguishes two GPU/cuVS routes:

- FAISS GPU indexes with NVIDIA cuVS integration, exposed through FAISS-backed
  backends such as `faiss_gpu_ivf_flat` and `faiss_gpu_ivfpq`. When the linked
  FAISS library was built with cuVS support, these paths report backend labels
  such as `GpuIndexIVFFlat_cuVS` and `GpuIndexIVFPQ_cuVS`.
- Direct RAPIDS cuVS calls, exposed through explicit backends such as
  `cuda_cuvs_cagra`, `cuda_cuvs_nndescent`, `cuda_cuvs_bruteforce`,
  `cuda_cuvs_ivf_flat`, and `cuda_cuvs_ivfpq`.

Use `backend_info()` and the `backend`, `resolved_backend`, `faiss`, `cuvs`, and `approximation` attributes returned by `nn()` to confirm which route and parameters a result used.

## kNN Models

```r
fit <- knn_fit(x, iris$Species, backend = "faiss", k = 5)
pred <- predict(fit, x)
prob <- predict(fit, x, type = "prob")
```


## Example

```r
library(faissR)

x <- scale(as.matrix(iris[, 1:4]))
knn <- nn(x, k = 15, backend = "auto", n_threads = 4)
knn

cl <- graph_cluster(knn, method = "louvain", backend = "cpu", n_runs = 2, n_threads = 2)
table(cl$membership)
```

## Graph Clustering and Acknowledgements

`graph_cluster()` runs CPU community detection with native faissR C++/OpenMP
code, without depending on `igraph`. It supports `method = "random_walking"`
through a random-walk label-propagation pass inspired by walktrap/random-walk
clustering, `method = "louvain"` through modularity local moving, and
`method = "leiden"` through local moving plus connected-community refinement.
CPU multicore execution is controlled with `n_threads`; repeated runs use
`n_runs`, and the best run by modularity is returned.

The CUDA graph-clustering backend is intentionally explicit. `backend = "cuda"`
uses native RAPIDS libcugraph for Louvain and Leiden when faissR is built with
libcugraph. `method = "random_walking"` is currently CPU-only until a dedicated
cuGraph random-walk clustering adapter is added. faissR does not use a
Python/cuGraph bridge. If libcugraph is not linked, CUDA graph clustering reports
unavailable rather than silently using CPU code.

Algorithmic and implementation acknowledgements: FAISS for nearest-neighbour
indexes; NVIDIA RAPIDS cuVS for optional CUDA nearest-neighbour/k-means
backends and FAISS GPU/cuVS integration; native faissR C++/OpenMP code for CPU
community detection; RAPIDS cuGraph/libcugraph for CUDA Louvain and Leiden graph
clustering; Blondel et al. (2008) for Louvain; Pons and Latapy (2006)
for walktrap/random-walk clustering; Traag et al. (2019) for Leiden; Sahu's
GVE-Leiden/OpenMP and dynamic Leiden work (arXiv:2312.13936 and
arXiv:2410.15451) as multicore and dynamic Leiden implementation
inspiration, including the referenced C++ repositories
`https://github.com/puzzlef/leiden-communities-openmp` and
`https://github.com/puzzlef/leiden-communities-openmp-heuristic-dynamic`; and Kapralov,
Lattanzi, Nouri, and Tardos (arXiv:2112.00655) for local parallel random-walk
motivation.

## With fastEmbedR

```r
library(faissR)
library(fastEmbedR)

x <- scale(as.matrix(iris[, 1:4]))
knn <- faissR::nn(x, k = 15, backend = "auto")

layout_umap <- fastEmbedR::umap_knn(knn)
layout_tsne <- fastEmbedR::opentsne_knn(knn, init_data = x)
```

Use `backend_info()` to inspect available NN backends. `faissR` never silently runs CPU code when an explicit GPU backend is requested. Public API helpers are focused on search, graph construction, kNN prediction, and k-means; benchmark-only quality helpers live in benchmark scripts rather than the package namespace.
