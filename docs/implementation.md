# Implementation

[Home](../README.md) |
[Installation](installation.md) |
**Implementation** |
[Examples](examples.md) |
[Benchmarks](benchmarks.md) |
[Autotuning](autotuning.md) |
[API](usage-api.md) |
[Backends](backend-capabilities.md) |
[References](references.md)

`faissR` is a standalone R package for nearest-neighbour search, graph
construction, graph clustering, kNN prediction, and k-means. FAISS is the
required compiled vector-search dependency. CUDA, RAPIDS cuVS, and RAPIDS
libcugraph are optional compiled backends; the package must still install from
source on CPU-only systems when FAISS is available. The code does not call
Python and does not require conda.

## Design Rules

1. FAISS CPU support is mandatory for the package build.
2. CUDA, FAISS GPU, cuVS, and libcugraph support are optional at compile time.
3. Explicit GPU backends fail clearly when GPU support is unavailable; they do
   not silently fall back to CPU.
4. KNN outputs use simple 1-based `indices` and numeric `distances` matrices.
5. Expensive KNN work should be reusable by graph, clustering, embedding, and
   supervised prediction functions.
6. Benchmark-only quality helpers are kept out of the public API.
7. External libraries are linked as system dependencies and are not vendored
   into the package.

## `nn()`

`nn()` is the low-level nearest-neighbour function. It accepts a reference
matrix/data frame and, optionally, a query matrix/data frame. If `points` is not
supplied, it performs self-search. Results are `faissR_nn` objects containing
neighbour indices, distances, the resolved backend, the metric, and exact versus
approximate metadata.

Supported CPU routes include:

- native exact dense CPU search;
- native exact sparse `dgCMatrix` search;
- FAISS CPU Flat, IVF-Flat, IVF-PQ, HNSW, NSG, and NN-Descent when present in
  the linked FAISS build;
- RcppHNSW as an optional CRAN-friendly fallback;
- exact 2D/3D grid and VP-tree routes for low-dimensional Euclidean self-KNN.

Supported CUDA routes include:

- FAISS GPU Flat, IVF-Flat, IVF-PQ, and CAGRA when FAISS GPU support is built;
- direct RAPIDS cuVS brute force, CAGRA, IVF-Flat, IVF-PQ, and NN-Descent when
  cuVS is available;
- native CUDA 2D/3D grid search for low-dimensional Euclidean self-KNN.

The validated high-performance metric is Euclidean/L2. Cosine and correlation
are exposed for exact CPU and RcppHNSW-compatible paths; accelerator backends
reject unsupported metric/backend combinations instead of returning Euclidean
neighbours under a different label.

### Automatic Backend Policy

`backend = "cpu_approx"` now prefers FAISS HNSW when FAISS is available,
then RcppHNSW, then exact CPU. This follows the chiamaka autotuning run where
FAISS HNSW gave the best CPU speed/recall balance on image and flow datasets.

`backend = "cuda_cuvs"`, `"cuda"`, and explicit CUDA aliases use CUDA-only
routes. Exact CUDA references are `faiss_gpu_flat_l2` and
`cuda_cuvs_bruteforce`. FAISS GPU CAGRA uses the FAISS/cuVS integration path.
Direct cuVS CAGRA is guarded by pilot recall tuning; if the pilot does not meet
the configured recall target, the function stops and recommends
`faiss_gpu_cagra` or `cuda_cuvs_bruteforce` rather than silently returning a
poor graph-search result.

IVF probe defaults are conservative enough to avoid misleading speed-only
results. IVFPQ is treated as an explicit memory-pressure backend because product
quantization can reduce recall substantially.

## `nn_without_self()`

`nn_without_self()` wraps self-KNN and returns exactly `k` non-self neighbours.
It is used internally by graph construction, clustering, and benchmarks. The
function requests enough neighbours to remove the self-match safely and keeps the
same `faissR_nn` result shape as `nn()`.

## `candidate_knn()`

`candidate_knn()` ranks a supplied candidate set for each query row. It avoids a
full all-pairs search when another algorithm has already proposed likely
neighbours. CPU candidate ranking is exact; CUDA candidate ranking is used only
when compiled and explicitly requested. The output can feed graph construction,
prediction, or diagnostic code.

## `knn_graph()`

`knn_graph()` converts a matrix, an existing KNN object, or an embedding-like
object into a native `faissR_graph` edge list. When KNN is not supplied, it calls
`nn_without_self()` with the requested backend. Supported edge weights include:

- `"distance"`: distance-derived edge strengths;
- `"snn"`: shared-nearest-neighbour/Jaccard weights;
- `"adaptive"`: local-scale exponential distance weights;
- `"binary"`: unweighted graph edges;
- `"auto"`: SNN for input-space graphs and distance-style weights for
  embedding-space graphs.

The graph object stores `k`, weighting, pruning, mutual-edge filtering, and KNN
backend metadata. It does not require `igraph`.

## `graph_cluster()`

`graph_cluster()` performs community detection on a precomputed `faissR_graph`,
a KNN object, or a matrix-like input. If a graph is not supplied, it builds one
with `knn_graph()` first.

Implemented methods are:

- `method = "random_walking"`: native CPU random-walk label propagation inspired
  by walktrap/random-walk clustering;
- `method = "louvain"`: native CPU modularity local-moving, with optional CUDA
  libcugraph Louvain when built and requested;
- `method = "leiden"`: native CPU local moving plus refinement to split
  disconnected communities, with optional CUDA libcugraph Leiden when built and
  requested.

CPU graph clustering uses faissR C++/OpenMP code and honors `n_threads` where
OpenMP is available. CUDA Louvain and Leiden require RAPIDS libcugraph and never
fall back to CPU for an explicit CUDA request. CUDA random-walking remains
unavailable until a dedicated CUDA implementation is added.

## `fast_kmeans()`

`fast_kmeans()` provides CPU and CUDA k-means routes. CPU builds use compiled
FAISS/native numeric paths. CUDA builds try FAISS GPU k-means and direct cuVS
k-means when those libraries are available. Results include cluster assignments,
centres, within-cluster sums of squares, cluster sizes, iteration count, selected
backend, and run parameters.

## `knn()` And `predict()`

`knn()` is the high-level supervised kNN API. It replaces the older public split
between `knn_fit()`, `faiss.fit()`, and `cuvs.fit()`.

Two forms are supported:

```r
model <- knn(Xtrain, Ytrain)
pred <- predict(model, Xtest)
```

and immediate prediction:

```r
pred <- knn(Xtrain, Ytrain, Xtest)
prob <- knn(Xtrain, Ytrain, Xtest, type = "prob")
```

The fitted model stores the training matrix, response, task type, backend,
metric, `k`, and thread settings. `predict()` performs classification or
regression from neighbour votes. `predict(type = "prob")` returns class
probabilities for classification, so a separate `predict_proba()` API is not
needed.

## Availability Helpers

- `faiss_available()` reports whether FAISS support was compiled and linked.
- `cuda_available()` reports CUDA build/runtime availability.
- `cuvs_available()` reports direct RAPIDS cuVS availability.
- `cugraph_available()` reports RAPIDS libcugraph graph-clustering support.
- `backend_info()` returns a data frame summarizing compiled/runtime backends,
  explicit backend labels, devices, and notes.

These helpers are informational; explicit backend calls still validate
availability at execution time.

## Large Data And ImageNet

The ImageNet feature file tested on chiamaka contains 1,281,167 rows by 1,024
features. It is stored as a double `data.table`, about 10 GB in R. Before FAISS
or cuVS can index it, R must create a contiguous numeric matrix and the backend
then creates float/index buffers. On the 31 GB RAM chiamaka host, full-reference
1.28M-row query tests were killed by the OS for FAISS HNSW, FAISS IVF, FAISS GPU
CAGRA, and direct cuVS IVF-Flat. This is a data representation and host-memory
limit, not evidence that those algorithms cannot handle ImageNet on a larger or
more memory-efficient setup.

Bounded ImageNet samples did work. On a 50,000-row sample with `k = 50`,
`faiss_gpu_flat_l2` completed in 1.208 seconds with exact recall, direct
`cuda_cuvs_bruteforce` completed in 1.747 seconds at 0.999999 recall, and
`faiss_hnsw` completed in 31.692 seconds at 0.999524 recall. The saved probe
files are listed in [Autotuning](autotuning.md).

For full ImageNet-scale runs on small-memory hosts, the preferred next
implementation step is to avoid loading the dataset as a double data frame. A
matrix or float32/on-disk representation would remove one large conversion copy
and make FAISS/cuVS index construction more practical.

## Licensing And Acknowledgement

faissR is released under the MIT license. The implementation is inspired by and
links against external work including FAISS, FAISS GPU/cuVS integration, RAPIDS
cuVS, RAPIDS libcugraph, HNSW, NN-Descent, IVF, product quantization, k-means,
Louvain, Leiden, and random-walk clustering. See [References](references.md) for
papers, software projects, and acknowledgements. External libraries remain
separate system dependencies with their own licenses.
