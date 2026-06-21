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
| `nn()` | Low-level nearest-neighbour search for reference/query matrices [1-6,13-16]. |
| `nn_without_self()` | Self-neighbour search that returns non-self neighbours only. |
| `candidate_knn()` | Exact top-k ranking inside a supplied candidate-neighbour matrix. |
| `knn_graph()` | Build a native weighted graph from data, an embedding, or KNN output. |
| `graph_cluster()` | Run native random-walking, Louvain, or Leiden graph clustering [9-12]. |
| `fast_kmeans()` | CPU/FAISS/CUDA/cuVS k-means where available [7-8]. |
| `knn()` | Fit a reusable kNN classifier/regressor or fit and predict immediately. |
| `predict()` | Predict labels, numeric responses, or class probabilities from `knn()`. |
| `backend_info()` | Report available CPU, FAISS, CUDA, cuVS, and cuGraph capabilities. |

## `nn()`

```r
nn(data, points = data, k = NULL, backend = "auto",
   method = "auto", metric = "euclidean", tuning = "auto",
   n_threads = NULL)
```

| Argument | Description |
| --- | --- |
| `data` | Numeric matrix, data frame, or sparse `Matrix` object with reference observations in rows and features in columns. |
| `points` | Optional query matrix/data frame/sparse matrix with the same number of columns as `data`. Defaults to `data` for self-search. |
| `k` | Number of neighbours to return. If `NULL`, faissR chooses an automatic neighbourhood size. |
| `backend` | Device backend: `"auto"`, `"cpu"`, or `"cuda"`. `"auto"` uses CUDA when CUDA/cuVS is available and CPU otherwise. Explicit `"cuda"` fails clearly when CUDA support is unavailable. |
| `method` | Algorithm selector: `"auto"`, `"exact"`, `"flat"`, `"bruteforce"`, `"grid"`, `"vptree"`, `"sparse"`, `"HNSW"`, `"IVF"`, `"IVFPQ"`, `"NSG"`, `"NNDescent"`, or `"CAGRA"` [1-6,13-16]. For example, `method = "grid", backend = "cpu"` maps to the CPU grid implementation, while `method = "grid", backend = "cuda"` maps to the CUDA grid implementation. Distance choices belong in `metric`, not `method`. Invalid backend/method combinations, such as `method = "CAGRA", backend = "cpu"`, stop with a clear error. |
| `metric` | Distance metric: `"euclidean"`, `"cosine"`, or `"correlation"`. Euclidean/L2 is the validated high-performance route for FAISS/CUDA/cuVS. Non-Euclidean metrics use supported CPU paths. |
| `tuning` | Tuning policy for approximate GPU methods: `"auto"`, `"cache"`, `"pilot"`, `"fixed"`, `"off"`, or `"none"`. `"auto"` uses the appropriate tuned default for the resolved method. |
| `n_threads` | Number of CPU worker threads for CPU/FAISS CPU backends. GPU backends ignore this argument. |

Returns a `faissR_nn` list with `indices` and `distances` matrices. Indices are
1-based R row numbers. The resolved backend is stored in
`attr(result, "backend")`.

## `nn_without_self()`

```r
nn_without_self(data, k, backend = "auto",
                method = "auto", metric = "euclidean",
                tuning = "auto", n_threads = NULL)
```

| Argument | Description |
| --- | --- |
| `data` | Numeric matrix, data frame, or sparse `Matrix` object with observations in rows. |
| `k` | Number of non-self neighbours to return per row. |
| `backend` | Device backend: `"auto"`, `"cpu"`, or `"cuda"`. This wrapper always performs self-search and removes the diagonal self match. |
| `method` | Same algorithm selector as `nn()`. |
| `metric` | `"euclidean"`, `"cosine"`, or `"correlation"`. |
| `tuning` | Same tuning policy as `nn()`. |
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
| `backend` | Device backend passed to `nn_without_self()` when neighbours must be computed from `data`: `"auto"`, `"cpu"`, or `"cuda"`. |
| `weight` | Edge weighting: `"auto"`, `"snn"`, `"adaptive"`, `"distance"`, or `"binary"`. `"auto"` uses shared-nearest-neighbour weights for input space and distance weights for embedding space. |
| `mutual` | If `TRUE`, keep only reciprocal nearest-neighbour edges. |
| `prune` | Drop edges with weight less than or equal to this non-negative threshold. |
| `n_threads` | CPU worker threads for neighbour search when KNN is computed inside the function. |

Returns a native `faissR_graph` edge-list object. No `igraph` dependency is
required.

## `graph_cluster()`

```r
graph_cluster(graph, method = "leiden", backend = "auto",
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
| `backend` | `"auto"`, `"cpu"`, or `"cuda"`. `"auto"` uses CUDA when libcugraph is available and CPU otherwise. `"cuda"` uses RAPIDS libcugraph Louvain/Leiden when compiled and available [9-12]. CUDA random-walking is not enabled yet. |
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
| `backend` | `"auto"`, `"cpu"`, or `"cuda"`. `"auto"` uses CUDA/cuVS k-means when available and CPU otherwise [7-8]. |
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
model <- knn(Xtrain, Ytrain, backend = "auto", method = "auto",
             tuning = "auto", k = 15L)
pred  <- knn(Xtrain, Ytrain, Xtest, type = "response")
prob  <- knn(Xtrain, Ytrain, Xtest, type = "prob")
```

| Argument | Description |
| --- | --- |
| `Xtrain` | Numeric training matrix with observations in rows. |
| `Ytrain` | Training labels for classification or numeric response for regression. Must have one value per row of `Xtrain`. |
| `Xtest` | Optional query matrix. If supplied, `knn()` fits and predicts immediately; otherwise it returns a reusable model. |
| `backend` | Device backend passed to `nn()`: `"auto"`, `"cpu"`, or `"cuda"`. |
| `method` | Nearest-neighbour algorithm selector passed to `nn()`. `"auto"` chooses the most appropriate method for the selected backend. |
| `metric` | Distance metric passed to `nn()`: `"euclidean"`, `"cosine"`, or `"correlation"`. |
| `tuning` | Tuning policy passed to `nn()`. `"auto"` uses the tuned default for the resolved method. |
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
        backend = "auto", tuning = "auto",
        vote = "majority", type = "response", ...)
```

| Argument | Description |
| --- | --- |
| `object` | A fitted model returned by `knn(Xtrain, Ytrain, ...)`. |
| `newdata` | Numeric query matrix with the same number of columns as the training matrix. |
| `k` | Number of neighbours for this prediction call. If `NULL`, uses the model default. |
| `backend` | Device backend for the prediction-time neighbour search: `"auto"`, `"cpu"`, or `"cuda"`. |
| `tuning` | Prediction-time tuning policy. `"auto"` uses the tuned default for the resolved method. |
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

nn_res <- nn_without_self(x, k = 15, backend = "cpu", method = "auto", n_threads = 4)
graph <- knn_graph(nn_res, k = 15, weight = "snn")
leiden <- graph_cluster(graph, method = "leiden", backend = "cpu", n_threads = 4)

table(leiden$membership)
```

For CUDA benchmarking on a GPU build:

```r
nn_gpu <- nn_without_self(x, k = 15, backend = "cuda", method = "auto")
```
