# API

[Home](../README.md) |
[Installation](installation.md) |
[Implementation](implementation.md) |
[Examples](examples.md) |
[Benchmarks](benchmarks.md) |
**API** |
[Backends](backend-capabilities.md) |
[References](references.md)

This page summarizes the public faissR functions and the arguments users are
expected to set. For the full R help page after installation, use
`?faissR::function_name`.

## Main Functions

| Function | Purpose |
| --- | --- |
| `nn()` | Low-level nearest-neighbour search for reference/query matrices. |
| `nn_without_self()` | Self-neighbour search that returns non-self neighbours only. |
| `candidate_knn()` | Exact top-k ranking inside a supplied candidate-neighbour matrix. |
| `knn_graph()` | Build a native weighted graph from data, an embedding, or KNN output. |
| `graph_cluster()` | Run native random-walking, Louvain, or Leiden graph clustering. |
| `fast_kmeans()` | CPU/FAISS/CUDA/cuVS k-means where available. |
| `knn()` | Fit a reusable kNN classifier/regressor or fit and predict immediately. |
| `predict()` | Predict labels, numeric responses, or class probabilities from `knn()`. |
| `backend_info()` | Report available CPU, FAISS, CUDA, cuVS, and cuGraph capabilities. |

## `nn()`

```r
nn(data, points = data, k = NULL, backend = "auto",
   metric = "euclidean", n_threads = NULL)
```

| Argument | Description |
| --- | --- |
| `data` | Numeric matrix, data frame, or sparse `Matrix` object with reference observations in rows and features in columns. |
| `points` | Optional query matrix/data frame/sparse matrix with the same number of columns as `data`. Defaults to `data` for self-search. |
| `k` | Number of neighbours to return. If `NULL`, faissR chooses an automatic neighbourhood size. |
| `backend` | Search backend. Use `"auto"` for the general selector, `"cpu_auto"` for shape-aware CPU-only selection, `"cuda_auto"`/`"gpu_auto"` for shape-aware CUDA-only selection, `"cpu"` for exact native CPU, `"faiss"`/`"faiss_flat_l2"` for FAISS CPU Flat, `"faiss_hnsw"`/`"faiss_ivf"` for FAISS approximate CPU indexes, `"faiss_gpu_flat_l2"` for exact FAISS GPU Flat, `"faiss_gpu_cagra"` for FAISS GPU CAGRA with cuVS integration, and `"cuda_cuvs_*"` for direct cuVS backends. Explicit GPU backends fail clearly if unavailable. |
| `metric` | Distance metric: `"euclidean"`, `"cosine"`, or `"correlation"`. Euclidean/L2 is the validated high-performance route for FAISS/CUDA/cuVS. Non-Euclidean metrics use supported CPU paths. |
| `n_threads` | Number of CPU worker threads for CPU/FAISS CPU backends. GPU backends ignore this argument. |

Returns a `faissR_nn` list with `indices` and `distances` matrices. Indices are
1-based R row numbers. The resolved backend is stored in
`attr(result, "backend")`.

## `nn_without_self()`

```r
nn_without_self(data, k, backend = "auto",
                metric = "euclidean", n_threads = NULL)
```

| Argument | Description |
| --- | --- |
| `data` | Numeric matrix, data frame, or sparse `Matrix` object with observations in rows. |
| `k` | Number of non-self neighbours to return per row. |
| `backend` | Same backend choices as `nn()`. This wrapper always performs self-search and removes the diagonal self match. |
| `metric` | `"euclidean"`, `"cosine"`, or `"correlation"`. |
| `n_threads` | CPU worker threads for CPU/FAISS CPU backends. |

Use this for graph construction and embedding workflows where each row should
not list itself as its nearest neighbour.

## `candidate_knn()`

```r
candidate_knn(data, candidates, points = data, k,
              backend = "auto", metric = "euclidean",
              n_threads = NULL, exclude_self = FALSE)
```

| Argument | Description |
| --- | --- |
| `data` | Numeric reference matrix with observations in rows. |
| `candidates` | Integer matrix of 1-based candidate reference row indices. It must have one row per query. Invalid, missing, zero, or out-of-range entries are ignored. |
| `points` | Optional query matrix. Defaults to `data` for self-query candidate scoring. |
| `k` | Number of best neighbours to keep from each candidate row. Must be no larger than `ncol(candidates)`. |
| `backend` | `"auto"`/`"cpu"` for exact CPU scoring inside candidates, or `"cuda"` for the native CUDA row-candidate kernel. |
| `metric` | `"euclidean"`, `"cosine"`, or `"correlation"` for CPU. CUDA candidate scoring currently supports Euclidean only. |
| `n_threads` | CPU worker threads. |
| `exclude_self` | If `TRUE`, remove each row from its own candidate list. This requires `points = data`. |

This function does not generate candidates; it only reranks candidates supplied
by another method.

## `knn_graph()`

```r
knn_graph(data, knn = NULL, k = 50L, backend = "auto",
          weight = "auto", mutual = FALSE, prune = 0,
          n_threads = NULL)
```

| Argument | Description |
| --- | --- |
| `data` | Numeric matrix/data frame, a `faissR_nn` object from `nn()`, or an embedding object with a matrix `layout`. |
| `knn` | Optional precomputed KNN object. If supplied, faissR reuses it instead of recomputing neighbours from `data`. |
| `k` | Number of neighbours used in the graph. If `knn` has fewer columns, faissR uses the available columns. |
| `backend` | Backend passed to `nn_without_self()` when neighbours must be computed from `data`. |
| `weight` | Edge weighting: `"auto"`, `"snn"`, `"adaptive"`, `"distance"`, or `"binary"`. `"auto"` uses shared-nearest-neighbour weights for input space and distance weights for embedding space. |
| `mutual` | If `TRUE`, keep only reciprocal nearest-neighbour edges. |
| `prune` | Drop edges with weight less than or equal to this non-negative threshold. |
| `n_threads` | CPU worker threads for neighbour search when KNN is computed inside the function. |

Returns a native `faissR_graph` edge-list object. No `igraph` dependency is
required.

## `graph_cluster()`

```r
graph_cluster(graph, method = "leiden", backend = "cpu",
              k = 50L, graph_backend = "auto", weight = "auto",
              mutual = FALSE, prune = 0, n_threads = NULL,
              n_runs = 1L, resolution = 1,
              objective_function = "modularity",
              n_iterations = 10L, steps = 4L, seed = NULL, ...)
```

| Argument | Description |
| --- | --- |
| `graph` | A `faissR_graph`, a KNN object returned by `nn()`, a numeric matrix/data frame, or an embedding object with `layout`. |
| `method` | Clustering algorithm: `"random_walking"`, `"louvain"`, or `"leiden"`. |
| `backend` | `"cpu"` for native C++/OpenMP clustering, or `"cuda"` for RAPIDS libcugraph Louvain/Leiden when compiled and available. CUDA random-walking is not enabled yet. |
| `k` | Number of neighbours when `graph` is raw data or an embedding rather than a graph/KNN object. |
| `graph_backend` | Backend passed to `nn_without_self()` when faissR needs to build the KNN graph internally. |
| `weight` | Graph edge weighting passed to `knn_graph()`: `"auto"`, `"snn"`, `"adaptive"`, `"distance"`, or `"binary"`. |
| `mutual` | If `TRUE`, build a mutual-nearest-neighbour graph. |
| `prune` | Non-negative edge pruning threshold. |
| `n_threads` | CPU threads for KNN construction and native CPU clustering. |
| `n_runs` | Number of independent clustering runs. faissR keeps the best modularity run. |
| `resolution` | Positive resolution parameter for Louvain/Leiden-style modularity scoring. Larger values tend to produce more communities. |
| `objective_function` | Reserved Leiden-compatible option. Currently accepts `"modularity"` or `"CPM"`. |
| `n_iterations` | Maximum native clustering iterations. |
| `steps` | Random-walk propagation depth for `method = "random_walking"`. |
| `seed` | Optional seed for reproducible repeated runs. |
| `...` | Reserved for future backend-specific options. |

Returns a `faissR_graph_cluster` object with `membership`, `modularity`,
parameters, backend metadata, and source acknowledgements.

## `fast_kmeans()`

```r
fast_kmeans(data, centers, backend = "auto",
            max_iter = 100L, n_init = 1L, tol = 1e-4,
            seed = 1L, n_threads = NULL,
            streaming_batch_size = 0L, init = "kmeans++")
```

| Argument | Description |
| --- | --- |
| `data` | Numeric matrix with observations in rows. |
| `centers` | Number of clusters. Must be between 1 and `nrow(data)`. |
| `backend` | `"auto"`, `"cpu"`, `"faiss"`, `"cuda"`, `"cuda_faiss"`, `"faiss_gpu"`, `"cuda_cuvs"`, or `"cuvs"`. `"auto"` prefers CUDA/cuVS when available, then FAISS, then base CPU. |
| `max_iter` | Maximum number of Lloyd iterations. |
| `n_init` | Number of random restarts where the selected backend supports it. |
| `tol` | Non-negative convergence tolerance where supported. |
| `seed` | Random seed for CPU/statistics and FAISS paths. |
| `n_threads` | CPU worker threads for FAISS/statistics paths. |
| `streaming_batch_size` | cuVS host-data streaming batch size. Use `0` to let cuVS choose its default. |
| `init` | Initialization method: `"kmeans++"` or `"random"` where supported. |

Returns cluster labels, centers, within-cluster sums of squares, cluster sizes,
iteration count, backend, and parameters.

## `knn()`

```r
model <- knn(Xtrain, Ytrain, backend = "auto", k = 15L)
pred  <- knn(Xtrain, Ytrain, Xtest, type = "response")
prob  <- knn(Xtrain, Ytrain, Xtest, type = "prob")
```

| Argument | Description |
| --- | --- |
| `Xtrain` | Numeric training matrix with observations in rows. |
| `Ytrain` | Training labels for classification or numeric response for regression. Must have one value per row of `Xtrain`. |
| `Xtest` | Optional query matrix. If supplied, `knn()` fits and predicts immediately; otherwise it returns a reusable model. |
| `backend` | Neighbour-search backend passed to `nn()`. |
| `metric` | Distance metric passed to `nn()`: `"euclidean"`, `"cosine"`, or `"correlation"`. |
| `task` | `"auto"`, `"classification"`, or `"regression"`. `"auto"` treats numeric `Ytrain` as regression and non-numeric `Ytrain` as classification. |
| `k` | Default number of neighbours used for prediction. |
| `n_threads` | CPU worker threads passed to `nn()`. |
| `vote` | `"majority"` for unweighted voting/means or `"weighted"` for inverse-distance weighted voting/means. Used for immediate prediction. |
| `type` | `"response"` for class labels or regression values; `"prob"` for classification probability matrices. |
| `...` | Reserved for future prediction options. |

When `Xtest` is omitted, the return value is a `faissR_knn_model`.

## `predict()`

```r
predict(object, newdata, k = NULL,
        vote = "majority", type = "response", ...)
```

| Argument | Description |
| --- | --- |
| `object` | A fitted model returned by `knn(Xtrain, Ytrain, ...)`. |
| `newdata` | Numeric query matrix with the same number of columns as the training matrix. |
| `k` | Number of neighbours for this prediction call. If `NULL`, uses the model default. |
| `vote` | `"majority"` for unweighted classification votes or regression means; `"weighted"` for inverse-distance weighting. |
| `type` | `"response"` for predicted labels/values or `"prob"` for classification probabilities. |
| `...` | Reserved for future options. |

For classification, `predict(type = "prob")` replaces a separate
`predict_proba()` function.

## Availability Helpers

```r
backend_info()
faiss_available()
cuda_available()
cuvs_available()
cugraph_available()
```

| Function | Arguments | Description |
| --- | --- | --- |
| `backend_info()` | None. | Returns a data frame with backend availability, explicit backend labels, device/runtime hints, and notes. |
| `faiss_available()` | None. | Returns `TRUE` when faissR was compiled and linked against FAISS. |
| `cuda_available()` | None. | Returns `TRUE` when native CUDA support was compiled and a CUDA device/runtime is available. |
| `cuvs_available()` | None. | Returns `TRUE` when direct RAPIDS cuVS backends were compiled and can be loaded. |
| `cugraph_available()` | None. | Returns `TRUE` when RAPIDS libcugraph graph clustering support was compiled and can be loaded. |

## Typical Workflow

```r
library(faissR)

x <- scale(as.matrix(iris[, 1:4]))

nn_res <- nn_without_self(x, k = 15, backend = "cpu_auto", n_threads = 4)
graph <- knn_graph(nn_res, k = 15, weight = "snn")
leiden <- graph_cluster(graph, method = "leiden", backend = "cpu", n_threads = 4)

table(leiden$membership)
```

For CUDA benchmarking on a GPU build:

```r
nn_gpu <- nn_without_self(x, k = 15, backend = "cuda_auto")
```
