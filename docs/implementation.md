# Implementation

[Home](../README.md) |
[Installation](installation.md) |
**Implementation** |
[Examples](examples.md) |
[Benchmarks](benchmarks.md) |
[API](usage-api.md) |
[Backends](backend-capabilities.md) |
[References](references.md)

`faissR` is designed as a standalone nearest-neighbour, graph, k-means, and
kNN-model layer. FAISS is the required vector-search dependency. CUDA, RAPIDS
cuVS, and RAPIDS libcugraph are optional compiled backends. The package code
does not call Python and does not require conda.

## Design Rules

1. FAISS is mandatory for package compilation.
2. CUDA, cuVS, and cuGraph are optional. CPU-only source installation must work
   when FAISS is available.
3. Explicit GPU requests never silently fall back to CPU.
4. Low-level KNN output is an R list with `indices` and `distances` matrices.
5. Graph and model functions reuse KNN output where possible so expensive
   neighbour search can be computed once.
6. Benchmark-only quality helpers are kept out of the public package API.

## `nn()`

`nn()` is the low-level nearest-neighbour search function. It accepts a
reference matrix and, optionally, a query matrix. If no query matrix is supplied,
it performs self-search. The result is a `faissR_nn` object containing 1-based
integer neighbour indices and numeric distances, with backend metadata stored as
attributes.

Implemented routes include:

- native exact CPU search;
- FAISS CPU Flat, IVF-Flat, IVF-PQ, HNSW, NSG, and NN-descent when supported by
  the linked FAISS build;
- FAISS GPU Flat, IVF-Flat, IVF-PQ, and CAGRA when FAISS GPU support is built;
- direct RAPIDS cuVS brute force, CAGRA, IVF-Flat, IVF-PQ, and NN-descent when
  cuVS is available;
- RcppHNSW fallback when the suggested package is installed;
- sparse/candidate and low-dimensional grid paths for specialized cases.

The public metrics are deliberately limited to `"euclidean"`, `"cosine"`, and
`"correlation"`. Euclidean/L2 is the validated high-performance path for FAISS,
CUDA, and cuVS. Cosine/correlation are routed to supported CPU paths unless a
backend explicitly supports the required semantics.

## `nn_without_self()`

`nn_without_self()` wraps `nn()` for self-search and removes the self-neighbour.
It is used by graph construction and clustering paths that need exactly `k`
non-self neighbours per row. Internally it requests enough neighbours to remove
the self column safely and returns the same KNN object shape as `nn()`.

## `candidate_knn()`

`candidate_knn()` ranks only a supplied candidate set for each query row. This
avoids constructing a full all-pairs distance matrix when another algorithm has
already generated likely neighbours. The CPU implementation computes exact
candidate distances and keeps the top `k`; CUDA candidate-ranking code is used
when compiled and explicitly requested. The output matches the `nn()` result
shape so it can feed downstream graph, prediction, or benchmark code.

## `knn_graph()`

`knn_graph()` builds a native `faissR_graph` edge-list object from:

- a numeric matrix or data frame;
- an existing `nn()` result;
- an embedding-like object with a matrix layout.

When KNN is not supplied, `knn_graph()` calls `nn_without_self()` using the
requested neighbour backend. It then builds weighted edges in C++ with one of
these weight modes:

- `"snn"`: shared-nearest-neighbour/Jaccard weights;
- `"adaptive"`: adaptive exponential distance weights;
- `"distance"`: `1 / (1 + distance)` weights;
- `"binary"`: unweighted edges;
- `"auto"`: SNN for input-space graphs and distance weights for embedding-space
  graphs.

The graph object stores metadata such as `k`, weighting mode, pruning, mutual
edge filtering, and the KNN backend used. It does not require `igraph`.

## `graph_cluster()`

`graph_cluster()` is the public community-detection API. It accepts either a
precomputed `faissR_graph`, a KNN object, a matrix/data frame, or an embedding
object. If necessary, it first builds a KNN graph through `knn_graph()` and then
runs clustering on the native edge list.

Supported methods:

- `method = "random_walking"`: CPU native random-walk label propagation inspired
  by walktrap/random-walk clustering. It uses the `steps` argument as the walk
  depth/control parameter and currently runs on CPU only.
- `method = "louvain"`: native CPU modularity local-moving implementation. With
  `backend = "cuda"` and a libcugraph-enabled build, it calls RAPIDS libcugraph
  Louvain instead.
- `method = "leiden"`: native CPU Louvain-style local moving plus a refinement
  pass that splits disconnected communities. With `backend = "cuda"` and a
  libcugraph-enabled build, it calls RAPIDS libcugraph Leiden instead.

Important backend behavior:

- `backend = "cpu"` uses faissR C++/OpenMP code and honors `n_threads` where
  OpenMP is available.
- `backend = "cuda"` requires RAPIDS libcugraph at build time for Louvain and
  Leiden. It never calls Python and never silently falls back to CPU.
- CUDA random-walking is intentionally unavailable until a dedicated cuGraph
  random-walk clustering adapter is implemented.

The returned `faissR_graph_cluster` object contains membership, modularity,
number of communities, method/backend labels, the graph used, run parameters,
and source acknowledgements.

## `fast_kmeans()`

`fast_kmeans()` runs k-means through the fastest compiled backend that matches
the request. CPU builds use FAISS/statistics paths. CUDA builds first try FAISS
GPU k-means when available, then direct RAPIDS cuVS k-means where supported.
Explicit GPU requests fail clearly when CUDA/cuVS is unavailable. The return
object includes cluster assignments, centers, within-cluster sums of squares,
cluster sizes, iteration count, selected backend, and parameters.

## `knn()`

`knn()` is the high-level supervised kNN model API. It replaces the older public
`knn_fit()`, `faiss.fit()`, and `cuvs.fit()` API.

Two call forms are supported:

```r
model <- knn(Xtrain, Ytrain)
pred <- predict(model, Xtest)
```

or immediate prediction:

```r
pred <- knn(Xtrain, Ytrain, Xtest)
prob <- knn(Xtrain, Ytrain, Xtest, type = "prob")
```

Internally, `knn()` stores the training matrix, response vector, task type,
backend, metric, `k`, and CPU-thread settings in a `faissR_knn_model` object.
It does not build a permanent FAISS/cuVS index object yet; prediction calls the
selected `nn()` backend using the stored training data.

## `predict()`

`predict.faissR_knn_model()` applies a model returned by `knn()`. For
classification it computes neighbour votes and returns a factor by default. With
`type = "prob"`, it returns a class-probability matrix. For regression it
returns a numeric vector of neighbour averages. `vote = "weighted"` uses
inverse-distance weights and handles exact zero-distance matches explicitly.

Saved model compatibility is maintained for objects created with older internal
field names, but the public API is now `knn()` plus `predict()`.

## `backend_info()`

`backend_info()` reports compiled/runtime backend availability in a data frame.
It checks native CPU, FAISS, FAISS GPU/cuVS-integrated routes, direct cuVS,
native CUDA, and cuGraph graph clustering. It is informational only; explicit
backend calls still validate availability at execution time.

## Availability Helpers

- `faiss_available()` returns whether faissR was compiled and linked against
  FAISS.
- `cuda_available()` returns whether the package was built with CUDA support and
  a CUDA device is visible.
- `cuvs_available()` returns whether direct RAPIDS cuVS backends are compiled
  and usable.
- `cugraph_available()` returns whether RAPIDS libcugraph support was compiled
  for CUDA graph clustering.

These helpers are useful in examples, benchmark scripts, and optional test
blocks, but production code should still handle explicit backend errors.

## Print Methods

`print.faissR_nn()` summarizes KNN result dimensions, backend metadata, metric,
and self-query status. `print.faissR_graph_cluster()` summarizes clustering
method, backend, number of communities, modularity, thread count, and the
selected run when repeated runs were used. These methods are S3 methods and are
called through base `print()`.

## Relationship To fastEmbedR

`fastEmbedR` imports `faissR` and calls `faissR::nn()` for matrix-input
`opentsne()` and `umap()`. The embedding package should not duplicate FAISS,
CUDA, or cuVS logic. Users who want explicit control over vector search should
load `faissR` directly, compute KNN once, and pass the result to downstream
embedding or clustering workflows.

## Licensing And Acknowledgement

FAISS, RAPIDS cuVS, RAPIDS cuGraph, HNSW, NN-descent, IVF, product
quantization, graph clustering, and k-means literature informed the
implementation choices [1-12]. External libraries are linked as system
dependencies and are not vendored into the package. faissR is released under the
MIT license.
