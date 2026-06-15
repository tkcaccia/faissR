# faissR

`faissR` is the nearest-neighbour companion package for `fastEmbedR`.
It contains the KNN/search side of the workflow:

- `nn()` for exact CPU, RcppHNSW, optional FAISS, optional cuVS/CUDA, and small
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

Optional FAISS and RAPIDS cuVS libraries are not vendored. Explicit optional
backend builds fail clearly if headers or libraries are unavailable:

```sh
FAISS_HOME=/path/to/faiss FAISSR_USE_FAISS=1 R CMD INSTALL .

CUDA_HOME=/path/to/cuda CUVS_HOME=/path/to/cuvs \
FAISSR_USE_CUDA=1 FAISSR_USE_CUVS=1 R CMD INSTALL .
```

The old `FASTEMBEDR_USE_*` environment variables are still accepted by
`configure` for compatibility with existing benchmark scripts.

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
