# Implementation

[Home](../README.md) |
[Installation](installation.md) |
**Implementation** |
[Examples](examples.md) |
[Benchmarks](benchmarks.md) |
[Autotuning](autotuning.md) |
[API](usage-api.md) |
[NN Methods](nn-methods.md) |
[Backends](backend-capabilities.md) |
[References](references.md)

`faissR` is a standalone R package for nearest-neighbour search, graph
construction, graph clustering, kNN prediction, and k-means. FAISS is the
required compiled vector-search dependency. CUDA, RAPIDS cuVS, and RAPIDS
libcugraph are optional compiled backends; the package must still install from
source on CPU-only systems when FAISS is available [1-3,12-16]. The code does
not call Python and does not require conda.

The implementation is organized around one public idea: users choose a device
family with `backend` and an algorithm family with `method`. The package then
resolves that request to a concrete compiled route, records the resolved backend
in the result attributes, and fails early for unsupported combinations. This is
important for reproducible benchmarking: `backend = "cuda"` must mean a CUDA
route was attempted, not that the package silently ran a CPU fallback.

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
8. Distance semantics are explicit: algorithm choice belongs in `method`, while
   Euclidean/cosine/correlation/inner-product choices belong in `metric`.
9. Approximate routes should expose enough metadata to make speed/quality
   trade-offs auditable.

## Build And Runtime Model

`configure` detects FAISS first. A build without FAISS is considered
misconfigured because FAISS is the core compiled dependency of faissR. On
machines without CUDA, the package can still build with FAISS CPU and native CPU
code. CUDA/cuVS/cuGraph support is enabled only when the required headers and
libraries are detected.

At runtime, availability helpers (`faiss_available()`, `cuda_available()`,
`cuvs_available()`, and `cugraph_available()`) report compiled/runtime support.
They are diagnostic helpers, not a substitute for execution-time validation.
Every explicit backend request is checked again when the method runs. For
example, `backend = "cuda", method = "cagra"` errors if neither FAISS GPU CAGRA
nor direct cuVS CAGRA is available.

Compiled backends are reached through Rcpp/C++ bridge files:

- FAISS CPU/GPU routines are isolated behind the FAISS bridge and use FAISS
  indexes such as Flat, IVF, IVFPQ, HNSW, NSG, NNDescent, and GPU CAGRA where
  the linked FAISS build exposes them [1-2,5-6,13-16].
- Direct RAPIDS cuVS routines are isolated behind the cuVS bridge and are
  optional at build time [3].
- CUDA-native helper kernels, such as the low-dimensional grid and candidate
  kernels, are isolated behind the CUDA bridge.
- RAPIDS libcugraph is used only for optional CUDA Louvain/Leiden graph
  clustering [12].

The package does not vendor FAISS, cuVS, or cuGraph. Those projects remain
system dependencies with their own release cadence and licenses.

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
  the linked FAISS build [1-6,16];
- RcppHNSW as an optional CRAN-friendly fallback;
- exact 2D/3D grid and VP-tree routes for low-dimensional Euclidean self-KNN;
  VP-tree also supports cosine/correlation through normalized Euclidean search
  when rows are nonzero/nonconstant.

Supported CUDA routes include:

- FAISS GPU Flat, IVF-Flat, IVF-PQ, and CAGRA when FAISS GPU support is built
  [1-2,13-16];
- direct RAPIDS cuVS brute force, CAGRA, IVF-Flat, IVF-PQ, and NN-Descent when
  cuVS is available [3,13-15];
- native CUDA 2D/3D grid search for low-dimensional Euclidean self-KNN.

The validated high-performance metric is Euclidean/L2. Cosine and correlation
are exposed for exact CPU, FAISS CPU/GPU Flat, FAISS HNSW, and
RcppHNSW-compatible paths. FAISS Flat and FAISS HNSW implement cosine by row L2
normalizing the inputs before inner-product search, and correlation by row
centering plus L2 normalization before inner-product search; both routes return
`1 - similarity` distances. Inner-product search is exposed for exact native CPU
scoring, FAISS Flat IP routes, FAISS HNSW IP, and the RcppHNSW/hnswlib IP
fallback. Approximate accelerator backends reject unsupported metric/backend
combinations instead of returning neighbours computed under a different metric
label.

### Result Metadata

All KNN routes return a `faissR_nn` object with:

- `indices`: an integer matrix of 1-based R row indices;
- `distances`: a numeric matrix aligned with `indices`;
- `attr(result, "backend")`: the public/resolved backend label returned to the
  user;
- `attr(result, "resolved_backend")`: when relevant, the concrete backend chosen
  behind an alias such as `backend = "cuda"`;
- `attr(result, "metric")`: the metric used;
- `attr(result, "exact")`: whether the route is exact by construction;
- `attr(result, "approximation")`: method-specific parameters for approximate
  routes, such as `nlist`, `nprobe`, HNSW `M`, CAGRA graph degree, or tuning
  metadata.

This metadata is intentionally simple because the same result object feeds graph
construction, clustering, benchmarking, and supervised prediction.

### Backend And Method Policy

The public KNN API separates device choice from algorithm choice:

- `backend = "auto"` uses a validated CUDA route only when the requested
  method/metric combination is supported, and otherwise resolves to CPU;
- `backend = "cpu"` forces CPU execution;
- `backend = "cuda"` forces CUDA execution and fails clearly if unavailable.

The public `method` argument selects the algorithm. `method = "auto"` is the
shape-aware selector for the chosen device. On CPU, it uses exact CPU for small
work, exact grid search for large 2D/3D Euclidean self-KNN, FAISS IVF for
million-row self-KNN where HNSW graph construction is too memory-heavy, FAISS
HNSW for large high-dimensional self-KNN across all supported CPU metrics when
FAISS is available, and FAISS Flat exact search for larger
cosine/correlation/inner-product query or exact workloads before falling back to
RcppHNSW/hnswlib for large non-Euclidean self-search when FAISS is unavailable
[1-2,5]. On CUDA, it uses CUDA grid search for large
2D/3D Euclidean self-KNN, exact FAISS GPU Flat or cuVS brute force for small and
medium searches, and FAISS GPU CAGRA for very large self-KNN when FAISS GPU/cuVS
integration is available [13-15].
The public `tuning` argument controls method-specific pilot tuning. The default
`tuning = "auto"` uses the recommended tuning policy for the resolved method;
`"cache"`, `"pilot"`, and `"fixed"` can be selected explicitly, and
`"off"`/`"none"` disables tuning.

Explicit methods map to the selected backend. For example,
`method = "grid", backend = "cpu"` resolves to the CPU grid implementation,
whereas `method = "grid", backend = "cuda"` resolves to the CUDA grid
implementation. Invalid combinations fail before computation; for example,
`method = "cagra", backend = "cpu"` errors because CAGRA is CUDA-only.

Direct cuVS CAGRA is guarded by pilot recall tuning; if the pilot does not meet
the configured recall target, the function stops and recommends FAISS GPU CAGRA
or cuVS brute force rather than silently returning a poor graph-search result.

IVF probe defaults are conservative enough to avoid misleading speed-only
results. IVFPQ is treated as an explicit memory-pressure backend because product
quantization can reduce recall substantially [6].

### Method Families

The public `method` names are stable user-facing labels. Each algorithm family
has one public method name; implementation labels such as `faiss_hnsw`,
`faiss_ivf`, `faiss_gpu_ivf_flat`, or `cuda_cuvs_cagra` are reserved for
resolved backend metadata and legacy explicit `backend` calls. Internally, the
public method names map to different concrete functions depending on `backend`.

| Method | CPU behavior | CUDA behavior | Notes |
| --- | --- | --- | --- |
| `auto` | Shape-aware exact/grid/FAISS IVF/FAISS HNSW selector. | Shape-aware CUDA grid, FAISS GPU Flat/cuVS brute force, or FAISS GPU CAGRA selector. | Default for general use. |
| `exact` | Native exact CPU route. | FAISS GPU Flat when available, otherwise cuVS brute force. | Accuracy-first baseline. |
| `flat` | FAISS Flat L2/IP index; cosine and correlation use normalized Flat IP. | FAISS GPU Flat L2/IP; cosine and correlation use normalized Flat IP. | Exact FAISS route [1-2,16]. |
| `bruteforce` | Native exact CPU route. | Direct cuVS brute force when available. | Useful for comparing direct cuVS against FAISS GPU Flat [3]. |
| `grid` | Native 2D/3D exact spatial grid. | CUDA 2D/3D grid. | Errors outside two or three columns. |
| `vptree` | Native exact CPU VP-tree for Euclidean, cosine, and correlation; zero-normalized non-Euclidean rows use exact CPU fallback. | Unsupported. | Low-dimensional CPU helper. |
| `sparse` | Native exact sparse `dgCMatrix` route. | Unsupported. | Avoids densifying sparse matrices. |
| `HNSW` | FAISS CPU HNSW. | Unsupported. | High-recall CPU graph-search route [5,16]. |
| `IVF` | FAISS CPU IVF-Flat. | FAISS GPU IVF-Flat. | Coarse-list approximate route [1-2,16]. |
| `IVFPQ` | FAISS CPU IVF-PQ. | FAISS GPU IVF-PQ. | Compressed approximate route [6,16]. |
| `NSG` | FAISS CPU NSG if available. | Unsupported. | Optional FAISS graph-search baseline [16]. |
| `NNDescent` | FAISS CPU NNDescent if available. | Direct cuVS NN-descent. | Approximate KNN graph construction [3-4,16]. |
| `CAGRA` | Unsupported. | FAISS GPU CAGRA preferred, direct cuVS CAGRA fallback. | CUDA-only graph-search method [3,13-16]. |

Unsupported method/backend pairs stop before computation. This makes benchmark
failures interpretable: a row marked unavailable or unsupported means the
requested algorithm/device combination does not exist in the package, not that a
different algorithm was substituted.

### Automatic Tuning

The public `tuning` argument controls method-specific tuning for approximate GPU
routes. Its default, `tuning = "auto"`, means “use the appropriate default
policy for the resolved method.” Current policies are intentionally conservative:

- `cache`: run a pilot tuning step when needed and reuse/store the selected
  parameters;
- `pilot`: run the pilot for this call without persisting the result;
- `fixed`: use fixed defaults but still record tuning metadata;
- `off`/`none`: disable tuning.

FAISS GPU IVF tuning tests candidate `nlist`/`nprobe` settings on a sample and
selects the fastest candidate that meets the recall target when possible. cuVS
CAGRA tuning tests graph/search parameters and rejects the route if the pilot
cannot meet the configured minimum recall. This behavior was added because some
direct cuVS CAGRA runs were fast but produced untrustworthy recall on raw
high-dimensional data.

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
`nn_without_self()` with the requested backend, NN method, metric, and tuning
policy. Supported edge weights include:

- `"distance"`: distance-derived edge strengths;
- `"snn"`: shared-nearest-neighbour/Jaccard weights;
- `"adaptive"`: local-scale exponential distance weights;
- `"binary"`: unweighted graph edges;
- `"auto"`: SNN for input-space graphs and distance-style weights for
  embedding-space graphs.

The graph object stores `k`, weighting, pruning, mutual-edge filtering, optional
target community count, and KNN backend/method/metric/tuning metadata. It does
not require `igraph`.

The graph construction layer deliberately accepts precomputed KNN output. This
allows expensive FAISS/cuVS searches to be reused across clustering, embedding,
or downstream analyses. It also makes benchmarking cleaner because nearest
neighbour speed can be measured independently from graph weighting and
community detection.

## `graph_cluster()`

`graph_cluster()` performs community detection on a precomputed `faissR_graph`,
a KNN object, or a matrix-like input. If a graph or KNN object is not supplied,
it builds the KNN graph internally with the requested `graph_backend`,
`graph_method`, `metric`, and `tuning` settings.

Implemented methods are:

- `method = "random_walking"`: native CPU random-walk label propagation inspired
  by walktrap/random-walk clustering and local parallel random-walk work
  [10,19];
- `method = "louvain"`: native CPU modularity local-moving, with optional CUDA
  libcugraph Louvain when built and requested [9,12];
- `method = "leiden"`: native CPU local moving plus refinement to split
  disconnected communities, with optional CUDA libcugraph Leiden when built and
  requested [11-12,17-18].

CPU graph clustering uses faissR C++/OpenMP code and honors `n_threads` where
OpenMP is available. CUDA Louvain and Leiden require RAPIDS libcugraph and never
fall back to CPU for an explicit CUDA request. CUDA random-walking remains
unavailable until a dedicated CUDA implementation is added; with
`backend = "auto"`, random-walking stays on the CPU even when libcugraph is
available.

`graph_cluster(n_clusters = m)` provides a target-community-count convenience
for Louvain and Leiden. The same target can also be stored on the graph with
`knn_graph(n_clusters = m)` and will be used by `graph_cluster()` unless the
caller supplies a different target. The graph is built once, then faissR
evaluates a small deterministic grid of resolution values around the supplied
`resolution` and keeps the result whose number of communities is closest to
`m`, breaking ties by modularity. The selected resolution and search table are
returned in the result metadata.

Native graph clustering does not depend on `igraph`. This keeps the graph API
under faissR's control and avoids a large mandatory graph dependency. The
returned object includes membership, modularity, backend, parameters, graph edge
list, and source acknowledgements. As with the nearest-neighbour and k-means
APIs, `backend` records the implementation that actually ran, while
`parameters$requested_backend` and `parameters$resolved_backend` preserve the
public backend request and the resolved device policy for benchmark auditing.

The clustering implementation benefits from fast KNN indirectly: FAISS/cuVS can
build the KNN graph faster, and `graph_cluster()` can then cluster that graph.
The community-detection step itself is a graph algorithm; it does not call FAISS
once the graph edges have been built.

## `fast_kmeans()`

`fast_kmeans()` provides CPU and CUDA k-means routes. CPU builds use compiled
FAISS/native numeric paths. CUDA builds try FAISS GPU k-means and direct cuVS
k-means when those libraries are available [7-8]. Results include cluster
assignments, centres, within-cluster sums of squares, cluster sizes, iteration
count, selected backend, and run parameters. The top-level `backend` field is
the implementation that actually ran, while `parameters$requested_backend` and
`parameters$resolved_backend` preserve the public backend request and resolved
device policy.

With `tuning = "auto"`, omitted `max_iter`, `n_init`, and `tol` values are
chosen by deterministic shape rules based on `nrow(data)`, `ncol(data)`, and
the requested number of centres. The function does not run pilot k-means jobs or
benchmark candidate parameter sets inside a user call. This keeps runtime
predictable while allowing small problems to use more restarts and large
high-dimensional problems to use cheaper convergence settings. The selected
policy is recorded in `result$parameters$tuning`; its shape metadata includes
estimated work, `n_per_center`, and flags such as `many_centers` and
`small_many_centers`. Small many-cluster problems can keep 100 Lloyd iterations
and use three restarts when `n / centers` remains large enough, while genuinely
large or high-dimensional problems keep cheaper settings. The `resolved_from`
field records whether `max_iter`, `n_init`, and `tol` were selected by
auto/default rules or supplied explicitly.

The public backend policy is the same as for KNN: `backend = "auto"` uses
CUDA only when CUDA plus FAISS GPU k-means or direct cuVS k-means is compiled
and available, and otherwise resolves to CPU;
`backend = "cpu"` forces the CPU route; `backend = "cuda"` requires an
accelerated route and errors if unavailable. This makes k-means behavior
consistent with `nn()`, `knn_graph()`, and `graph_cluster()`.

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
method, metric, tuning policy, `k`, and thread settings. `predict()` performs
classification or regression from neighbour votes. `predict(type = "prob")`
returns class probabilities for classification, so a separate `predict_proba()`
API is not needed.

The supervised API intentionally reuses `nn()` rather than creating independent
FAISS/cuVS model classes. That keeps method selection, backend validation,
tuning, and result semantics identical between unsupervised KNN and supervised
kNN prediction.

## Availability Helpers

- `faiss_available()` reports whether FAISS support was compiled and linked.
- `cuda_available()` reports CUDA build/runtime availability.
- `cuvs_available()` reports direct RAPIDS cuVS availability.
- `cugraph_available()` reports RAPIDS libcugraph graph-clustering support.
- `backend_info()` returns a data frame summarizing compiled/runtime backends,
  public call hints, resolved/internal route labels, devices, and notes.

These helpers are informational; explicit backend calls still validate
availability at execution time.

## Memory And Data Representation

R stores ordinary numeric matrices as double precision. FAISS and cuVS commonly
operate on float data internally. Most accelerated routes therefore need at
least one conversion/copy from R's column-major double matrix to backend-friendly
buffers. For moderate datasets this overhead is small relative to KNN search;
for very large datasets it can dominate memory use.

The implementation tries to keep public inputs simple (`matrix`, `data.frame`,
and sparse `Matrix` objects) while recording memory-sensitive behavior in
benchmark notes. Future improvements can reduce copies by supporting explicit
float32/on-disk data representations, but the current CRAN-oriented interface
does not require nonstandard R vector types.

Sparse input is handled separately. If the input is a sparse `Matrix`, the
native sparse CPU route can avoid densification. GPU and FAISS routes that do
not support sparse input fail or densify only when explicitly requested, rather
than pretending to preserve sparse semantics.

## Large Data And ImageNet

The ImageNet feature file used in package benchmarks contains 1,281,167 rows by
1,024 features. In the tested representation it was stored as a double
`data.table`, about 10 GB in R. Before FAISS or cuVS can index it, R must create
a contiguous numeric matrix and the backend then creates float/index buffers. On
a memory-limited workstation, full-reference 1.28M-row query tests were killed
by the OS for FAISS HNSW, FAISS IVF, FAISS GPU CAGRA, and direct cuVS IVF-Flat.
This is a data representation and host-memory limit, not evidence that those
algorithms cannot handle ImageNet on a larger or more memory-efficient setup.

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
Louvain, Leiden, and random-walk clustering [1-19]. See
[References](references.md) for papers, software projects, and acknowledgements.
External libraries remain separate system dependencies with their own licenses.
