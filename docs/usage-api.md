# API

[Home](../README.md) |
[Installation](installation.md) |
[Implementation](implementation.md) |
[Examples](examples.md) |
[Benchmarks](benchmarks.md) |
**API** |
[Backends](backend-capabilities.md) |
[References](references.md)

## Main Functions

| Function | Purpose |
| --- | --- |
| `nn()` | Nearest-neighbour search for data/query matrices. |
| `nn_without_self()` | Convenience helper for self-search where the first self-neighbour is removed. |
| `candidate_knn()` | Top-k ranking inside supplied candidate rows. |
| `knn_graph()` | Build a native weighted graph from KNN output, data, or an embedding. |
| `fast_kmeans()` | FAISS/cuVS-backed k-means where available. |
| `knn_fit()` | Fit a reusable kNN classifier/regressor. |
| `faiss.fit()` | Alias for FAISS-oriented kNN model fitting. |
| `cuvs.fit()` | Alias for cuVS-oriented kNN model fitting. |
| `predict()` | kNN class or numeric prediction. |
| `backend_info()` | Report available FAISS, CUDA, cuVS, cuGraph, and optional backend capabilities. |

## Typical KNN Workflow

```r
library(faissR)

x <- scale(as.matrix(iris[, 1:4]))
knn <- nn(x, k = 50, backend = "auto", metric = "euclidean", n_threads = 4)

saveRDS(knn, "iris_knn_k50.rds")
```

## Backend Argument

`backend = "auto"` selects the strongest available backend for the requested
data and metric. Explicit backends are useful for benchmarks. If an explicit
GPU backend is unavailable, the call fails clearly.

The exact set of backend names depends on the compiled libraries. Use:

```r
backend_info()
```

## Metrics

Use:

- `metric = "euclidean"` for L2 distance;
- `metric = "cosine"` for row-normalized inner-product search.

The package intentionally keeps the public metric list short because these are
the two most common large-scale embedding and single-cell use cases.
