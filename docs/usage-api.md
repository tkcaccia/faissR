# API

[Home](../README.md) |
[Installation](installation.md) |
[Implementation](implementation.md) |
[Examples](examples.md) |
[Benchmarks](benchmarks.md) |
**API** |
[NN Methods](nn-methods.md) |
[Backends](backend-capabilities.md) |
[References](references.md)

This page summarizes the public faissR functions and the arguments users are
expected to set. For the full R help page after installation, use
`?faissR::function_name`.

## Main Functions

| Function | Purpose |
| --- | --- |
| `nn()` | Low-level nearest-neighbour search for reference/query matrices [1-6,13-16,22-23]. |
| `nn_without_self()` | Self-neighbour search that returns non-self neighbours only. |
| `candidate_knn()` | Exact top-k ranking inside a supplied candidate-neighbour matrix. |
| `knn_graph()` | Build a native weighted graph from data, an embedding, or KNN output. |
| `graph_cluster()` | Run native random-walking, Louvain, or Leiden graph clustering [9-12]. |
| `fast_kmeans()` | CPU/FAISS/CUDA/cuVS k-means where available [7-8]. |
| `knn()` | Fit a reusable kNN classifier/regressor or fit and predict immediately. |
| `predict()` | Predict labels, numeric responses, or class probabilities from `knn()`. |
| `backend_info()` | Report available CPU, FAISS, CUDA, cuVS, and cuGraph capabilities. |
| `nn_capabilities()` | Report supported nearest-neighbour method/backend/metric combinations for preflight checks; `runtime = TRUE` adds current-build availability columns. |

## `nn()`

```r
nn(data, points = data, k = NULL, backend = "auto",
   method = "auto", metric = "euclidean", tuning = "auto",
   cagra_implementation = NULL, cagra_build_algo = NULL,
   output = "double", distances = NULL, n_threads = NULL)
```

| Argument | Description |
| --- | --- |
| `data` | Numeric matrix, data frame, or optional `float::fl()`/`float32` matrix with reference observations in rows and features in columns. Sparse Matrix input is not supported by the current public NN API. The first float32 input route supports CPU FAISS Flat for Euclidean, cosine, correlation, and inner-product searches without converting the float32 source object to an R double matrix; ordinary R double query inputs can be paired with float32 reference data. |
| `points` | Optional query matrix/data frame/float32 matrix with the same number of columns as `data`. Defaults to `data` for self-search. Float32 reference/query inputs can be mixed with ordinary R double matrices; the double side is converted once to float32 inside the FAISS adapter. |
| `k` | Number of neighbours to return. If `NULL`, faissR chooses an automatic neighbourhood size. |
| `backend` | Device backend: `"auto"`, `"cpu"`, or `"cuda"`. `"auto"` uses a validated CUDA route only when the requested method/metric combination is supported and CUDA/cuVS runtime support is available, and otherwise resolves to CPU. Explicit `"cuda"` fails clearly when CUDA support or the selected CUDA combination is unavailable. |
| `method` | Algorithm selector: `"auto"`, `"exact"`, `"flat"`, `"bruteforce"`, `"grid"`, `"hnsw"`, `"ivf"`, `"ivfpq"`, `"vamana"`, `"nsg"`, `"nndescent"`, or `"cagra"` [1-6,13-16,22-24]. These are canonical lowercase public labels; resolved implementation labels such as `faiss_hnsw`, `faiss_gpu_ivf_flat`, or `cuda_cuvs_cagra` are recorded in result metadata but are not public `method` values. For example, `method = "grid", backend = "cpu"` maps to the CPU grid implementation, while `method = "grid", backend = "cuda"` maps to the CUDA grid implementation. Distance choices belong in `metric`, not `method`. Invalid backend/method combinations, such as `method = "cagra", backend = "cpu"`, stop with a clear error. |
| `metric` | Distance metric: `"euclidean"`, `"cosine"`, `"correlation"`, or `"inner_product"`. Aliases such as `"l2"`, `"cor"`/`"pearson"`, and `"ip"` are accepted and stored as canonical metric labels. Inner product is the raw dot product; cosine is the dot product after row L2 normalization; correlation is centered cosine similarity after subtracting each row mean and L2-normalizing each row. For `metric = "inner_product"`, neighbours are ranked by larger raw dot product, but returned `distances` keep faissR's smaller-is-better convention: within each query row the best returned dot product has distance `0`, and lower dot products have larger shifted distances. Euclidean/L2 is the validated high-performance route for approximate FAISS/CUDA/cuVS. Cosine and correlation use validated exact paths, FAISS CPU/GPU Flat, IVF-Flat, IVFPQ, CPU HNSW, native CPU NSG, and CUDA NSG through normalized search. All-zero cosine rows and constant correlation rows are zero-normalized edge cases: faissR treats zero-vs-zero distance as `0` and zero-vs-nonzero distance as `1`; CPU FAISS Flat uses the exact CPU scorer for those rows to preserve deterministic small-`k` tie handling, while explicit CUDA routes remain on CUDA. Inner product is supported by native exact CPU scoring, FAISS Flat IP, FAISS IVF-Flat/IVFPQ IP, FAISS HNSW IP, native CPU NN-descent, native CPU NSG, RcppHNSW/hnswlib fallback paths when FAISS is unavailable, CUDA NSG self-KNN, CUDA NN-descent through faissR's native CUDA candidate-refinement route, and CUDA CAGRA/HNSW through a maximum-inner-product-to-L2 transform. Direct cuVS NN-descent still does not expose raw inner-product search. |
| `tuning` | Tuning policy for approximate methods: `"auto"`, `"cache"`, `"pilot"`, `"fixed"`, `"off"`, or `"none"`. `"auto"` uses deterministic no-pilot defaults for the resolved method; `"cache"` and `"pilot"` opt into pilot tuning. |
| `cagra_implementation` | CUDA CAGRA provider for this call. `NULL` uses `options(faissR.cagra_implementation = ...)`; `"auto"` prefers FAISS GPU CAGRA and falls back to direct RAPIDS cuVS CAGRA; `"faiss_gpu"` or `"cuvs"` force one provider for benchmark rows. This affects `backend = "cuda", method = "cagra"` and CUDA-auto routes that select CAGRA. |
| `cagra_build_algo` | Direct RAPIDS cuVS CAGRA graph-build algorithm for this call. `NULL` uses `options(faissR.cuvs_cagra_build_algo = "auto")`; `"auto"` keeps the cuVS default, `"ivf_pq"` requests the IVF-PQ graph builder, `"nn_descent"` requests cuVS NN-descent graph construction, and `"iterative_cagra_search"` requests cuVS iterative CAGRA graph building. This is a CAGRA construction parameter, not a fallback to a different public method, and successful results record it in `route_parameters`. |
| `output` | Distance storage type: `"double"` returns the default R numeric matrix; `"float"` returns `distances` as a `float::fl()`/`float32` matrix and records `distance_type = "float32"` plus `attr(result, "distance_type") = "float32"`. Float32 inputs and ordinary R double inputs on CPU FAISS Flat-style float-output requests use the float-pointer FAISS route, which constructs float distance payloads directly instead of first materializing R double distances, except for zero-row cosine/correlation correction. Float32-route results also expose `input_layout` and `input_owns_data`, so downstream packages can tell whether FAISS read a direct row-compatible float32 payload or used a one-time row-major adapter buffer. The `float` package is optional and used only when this output is requested or a float32 input object is supplied. |
| `distances` | Optional alias for `output`; use `distances = "float"` when downstream code wants the returned distance matrix to remain float32. |
| `n_threads` | Number of CPU worker threads for CPU/FAISS CPU backends. GPU backends ignore this argument. |

Advanced tuning and cache knobs use `options(faissR.<name> = ...)`.

Returns a `faissR_nn` list with `indices` and `distances` plus stable metadata
fields: `index_base`, `distance_type`, `metric`, and `backend_used`. Float32
routes also expose `input_layout` and `input_owns_data` to document the adapter
path used before FAISS consumed the `float*` data. Indices are 1-based R row
numbers. Normalized Euclidean graph routes used for cosine/correlation also
record `metric_transform` and `attr(result, "distance_transform")`, which makes
the distance conversion explicit in benchmark tables. The public request is
stored in
`attr(result, "requested_backend")`, `attr(result, "requested_method")`, and
`attr(result, "tuning")`; the implementation-facing route is stored in
`attr(result, "backend")` and, when it differs from the public label,
`attr(result, "resolved_backend")`.

### Nearest-Neighbour Methods

| `method` | Description |
| --- | --- |
| `"auto"` | Shape-aware selector for the chosen backend. CPU auto uses exact search for small work, grid search for large 2D/3D Euclidean/cosine/correlation self-search, FAISS IVF for million-row self-search where HNSW graph construction is too memory-heavy, FAISS HNSW for large high-dimensional CPU searches across all supported metrics when FAISS is available, native CPU NSG-style refinement for larger non-Euclidean self-KNN when FAISS/RcppHNSW are unavailable, and native CPU NN-descent for other large self-KNN fallback cases [1-2,5,21]. CUDA auto uses CUDA grid for large 2D/3D Euclidean/cosine/correlation self-search, exact FAISS GPU Flat or cuVS brute force for small/medium Euclidean searches, cuVS HNSW or FAISS GPU CAGRA for larger Euclidean/cosine/correlation searches when available, and FAISS GPU Flat IP routes for exact inner-product searches when FAISS GPU Flat is available [1-3,13-15,22]. |
| `"exact"` | Exact nearest-neighbour search. On CPU this uses faissR's native exact route; on CUDA it uses FAISS GPU Flat when FAISS reports GPU support. Euclidean CUDA exact search can otherwise use direct cuVS brute force when available; cosine, correlation, and inner product require the FAISS GPU Flat metric-aware routes [1-3,16]. |
| `"flat"` | FAISS Flat index route for exhaustive L2/IP search. Cosine/correlation use normalized Flat IP; CPU degenerate zero-normalized rows use exact CPU scoring to match deterministic exact tie semantics [1-2,16]. |
| `"bruteforce"` | Brute-force exhaustive search. On CPU it maps to the exact CPU route. On CUDA, Euclidean prefers RAPIDS cuVS brute force; cosine, correlation, and inner product use FAISS GPU Flat when available because direct cuVS brute force is Euclidean/L2-only in faissR [1-3,16]. |
| `"grid"` | Native spatial grid search for 2D/3D Euclidean, cosine, and correlation self-KNN. Cosine/correlation use normalized Euclidean grid search. It is intended for low-dimensional spatial or simulated data and errors clearly outside supported dimensions. |
| `"hnsw"` | HNSW graph-search index. CPU uses FAISS HNSW when available, with RcppHNSW/hnswlib fallback. CUDA uses RAPIDS cuVS HNSW converted from a CUDA CAGRA index for Euclidean plus normalized cosine/correlation; raw inner product is handled by a maximum-inner-product-to-L2 extra-dimension transform and returned as shifted inner-product distances [3,5,16,22-23]. |
| `"ivf"` | FAISS IVF-Flat inverted-file index. IVF partitions vectors into coarse lists and probes selected lists; it supports L2/IP plus normalized-IP cosine/correlation, trades exactness for speed/memory, and uses deterministic shape/k/metric-aware probe defaults for very large CPU/GPU searches [1-2,16]. |
| `"ivfpq"` | FAISS IVF with product quantization. IVFPQ supports L2/IP plus normalized-IP cosine/correlation, reuses the metric-aware IVF probing defaults, compresses vectors, and is best treated as a memory-pressure method rather than an accuracy-first default. CPU IVFPQ requires at least 624 training rows, the minimum needed for FAISS' smallest supported 4-bit product quantizer without undertrained-codebook warnings; between 624 and 9,983 rows, auto tuning uses 4-bit PQ rather than the 8-bit default [1-2,6,16]. |
| `"vamana"` | DiskANN/Vamana-style robust-pruned candidate graph implemented in faissR. CPU refines candidate rows with native CPU scoring; CUDA refines them with the native CUDA row-candidate kernel. cuVS Vamana is acknowledged for GPU build/serialization, but current cuVS does not expose KNN search [3,24]. |
| `"nsg"` | Navigating Spreading-out Graph style approximate search. CPU uses faissR's native NSG-style self-KNN route for all public metrics so public calls avoid unsafe linked-FAISS graph construction. CUDA uses faissR's native NSG-style self-KNN route for all public metrics, with normalized cosine/correlation and shifted dot-product inner-product distances. Native NSG `tuning = "auto"` uses backend-specific shape/k/metric defaults and reads `faissR.cpu_nsg_*` or `faissR.cuda_nsg_*` options [16,21,29]. |
| `"nndescent"` | NN-descent style approximate graph construction. CPU uses faissR's native NNDescent route by default. CUDA maps to direct cuVS NN-descent for Euclidean/L2 plus normalized cosine/correlation, and to faissR's native CUDA candidate-refinement route for raw inner-product self-KNN. FAISS NNDescent is disabled by default because linked FAISS builds can abort during graph construction [3-4,16]. |
| `"cagra"` | CUDA-only graph-search method. faissR prefers FAISS GPU CAGRA when FAISS is built with NVIDIA cuVS integration and otherwise uses direct cuVS CAGRA when available. Use `cagra_implementation = "faiss_gpu"` or `"cuvs"` to force the provider for a call, or `options(faissR.cagra_implementation = ...)` to set a session default; `"auto"` keeps the default FAISS-then-cuVS rule. Direct cuVS CAGRA also exposes `cagra_build_algo` (`"auto"`, `"ivf_pq"`, `"nn_descent"`, or `"iterative_cagra_search"`) as an explicit graph-construction parameter. It supports Euclidean/L2, cosine/correlation through normalized Euclidean graph search, and raw inner product through a MIPS-to-L2 extra-dimension transform [3,13-16]. |

## `nn_without_self()`

```r
nn_without_self(data, k, backend = "auto",
                method = "auto", metric = "euclidean",
                tuning = "auto", cagra_implementation = NULL,
                cagra_build_algo = NULL, output = "double",
                distances = NULL, n_threads = NULL)
```

| Argument | Description |
| --- | --- |
| `data` | Numeric matrix, data frame, or optional `float::fl()`/`float32` matrix with observations in rows. |
| `k` | Number of non-self neighbours to return per row. |
| `backend` | Device backend: `"auto"`, `"cpu"`, or `"cuda"`. This wrapper uses the same backend/method/metric resolver as `nn()`, always performs self-search, and removes the diagonal self match. |
| `method` | Same algorithm selector as `nn()`. |
| `metric` | `"euclidean"`, `"cosine"`, `"correlation"`, or `"inner_product"`; aliases such as `"l2"`, `"cor"`/`"pearson"`, and `"ip"` are accepted. Correlation is centered cosine similarity, not raw inner product. |
| `tuning` | Same tuning policy as `nn()`. |
| `cagra_implementation` | CUDA CAGRA provider for this call. See `nn()`. |
| `cagra_build_algo` | Direct RAPIDS cuVS CAGRA graph-build algorithm for this call. See `nn()`. |
| `output` | Distance storage type: `"double"` for an R numeric distance matrix or `"float"` for a `float::fl()`/`float32` distance matrix when the optional `float` package is installed. |
| `distances` | Optional alias for `output`. |
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
| `metric` | `"euclidean"`, `"cosine"`, `"correlation"`, or `"inner_product"`. Aliases such as `"l2"`, `"cor"`/`"pearson"`, and `"ip"` are accepted. Correlation is centered cosine similarity, not raw inner product. Inner-product candidate scoring ranks by larger raw dot product, while returned `distances` are shifted within each query so the best returned dot product has distance `0`. CUDA candidate scoring supports Euclidean directly, cosine/correlation through normalized Euclidean scoring, and raw inner-product scoring through a dedicated CUDA kernel mode with the same shifted-distance convention. |
| `n_threads` | CPU worker threads. |
| `exclude_self` | If `TRUE`, remove each row from its own candidate list. This requires `points = data`. |

This function does not generate candidates; it only reranks candidates supplied
by another method.

## `knn_graph()`

```r
knn_graph(data, knn = NULL, k = 50L, backend = "auto",
          method = NULL, nn_method = "auto",
          metric = "euclidean", tuning = "auto",
          cagra_implementation = NULL, weight = "auto",
          mutual = FALSE, prune = 0, n_threads = NULL)
```

| Argument | Description |
| --- | --- |
| `data` | Numeric matrix/data frame, a `faissR_nn` object from `nn()`, or an embedding object with a matrix `layout`. |
| `knn` | Optional precomputed KNN object. If supplied, faissR reuses it instead of recomputing neighbours from `data`. |
| `k` | Number of neighbours used in the graph. If `knn` has fewer columns, faissR uses the available columns. |
| `backend` | Device backend passed to `nn_without_self()` when neighbours must be computed from `data`: `"auto"`, `"cpu"`, or `"cuda"`. |
| `method` | Alias for `nn_method`, matching the public method argument used by `nn()` and `knn()`. If both are supplied they must agree after alias normalization. |
| `nn_method` | Nearest-neighbour method passed to `nn_without_self()` when neighbours must be computed from `data`; kept for existing graph-specific code. |
| `metric` | Distance metric passed to `nn_without_self()` when neighbours must be computed from `data`; aliases such as `"l2"`, `"cor"`/`"pearson"`, and `"ip"` are accepted. Correlation is centered cosine similarity, not raw inner product. Inner-product graph construction ranks by larger raw dot product and reuses faissR's shifted smaller-is-better distance convention from `nn()`. |
| `tuning` | Tuning policy passed to `nn_without_self()` when neighbours must be computed from `data`. |
| `cagra_implementation` | CUDA CAGRA provider passed to `nn_without_self()` when neighbours must be computed from `data`. |
| `weight` | Edge weighting: `"auto"`, `"snn"`, `"adaptive"`, `"distance"`, or `"binary"`. `"auto"` uses shared-nearest-neighbour weights for input space and distance weights for embedding space. |
| `mutual` | If `TRUE`, keep only reciprocal nearest-neighbour edges. |
| `prune` | Drop edges with weight less than or equal to this non-negative threshold. |
| `n_threads` | CPU worker threads for neighbour search when KNN is computed inside the function. |

Returns a native `faissR_graph` edge-list object. No `igraph` dependency is
required. When faissR builds neighbours internally, the `faissR_graph` metadata
includes graph size, weighting, nearest-neighbour method, metric, tuning
policy, and the requested/resolved public KNN backends. It also preserves compact-relevant KNN route metadata such as
approximation parameters, auto-selection metadata, normalized metric transform
metadata, and FAISS/cuVS/grid attributes for benchmark auditing.

## `graph_cluster()`

```r
graph_cluster(graph, method = "random_walking", backend = "auto",
              k = 50L, graph_backend = "auto", graph_method = "auto",
              metric = "euclidean", tuning = "auto",
              cagra_implementation = NULL, weight = "auto",
              mutual = FALSE, prune = 0, n_threads = NULL,
              n_runs = 1L, resolution = 1, n_clusters = NULL,
              objective_function = "modularity",
              n_iterations = 10L, steps = 4L, seed = NULL, ...)
```

| Argument | Description |
| --- | --- |
| `graph` | A `faissR_graph`, a KNN object returned by `nn()`, a numeric matrix/data frame, or an embedding object with `layout`. |
| `method` | Clustering algorithm: `"random_walking"`, `"louvain"`, or `"leiden"`. |
| `backend` | `"auto"`, `"cpu"`, or `"cuda"`. `"auto"` uses CUDA when libcugraph is available for Louvain/Leiden and CPU otherwise; auto keeps `"random_walking"` on CPU. `"cuda"` uses RAPIDS libcugraph Louvain/Leiden when compiled and available [9-12]. CUDA random-walking is not enabled yet. |
| `k` | Number of neighbours when `graph` is raw data or an embedding rather than a graph/KNN object. |
| `graph_backend` | Backend passed to `nn_without_self()` when faissR needs to build the KNN graph internally. |
| `graph_method` | Nearest-neighbour method passed to `nn_without_self()` when faissR needs to build the KNN graph internally. |
| `metric` | Distance metric passed to `nn_without_self()` when faissR needs to build the KNN graph internally; aliases such as `"l2"`, `"cor"`/`"pearson"`, and `"ip"` are accepted. Correlation is centered cosine similarity, not raw inner product. Inner-product graph construction ranks by larger raw dot product and reuses faissR's shifted smaller-is-better distance convention from `nn()`. |
| `tuning` | Tuning policy passed to `nn_without_self()` when faissR needs to build the KNN graph internally. |
| `cagra_implementation` | CUDA CAGRA provider passed to `nn_without_self()` when faissR builds the KNN graph internally. |
| `weight` | Graph edge weighting passed to `knn_graph()`: `"auto"`, `"snn"`, `"adaptive"`, `"distance"`, or `"binary"`. |
| `mutual` | If `TRUE`, build a mutual-nearest-neighbour graph. |
| `prune` | Non-negative edge pruning threshold. |
| `n_threads` | CPU threads for KNN construction and native CPU clustering. |
| `n_runs` | Number of independent clustering runs. faissR keeps the best modularity run. |
| `resolution` | Positive resolution parameter for Louvain/Leiden-style modularity scoring. Larger values tend to produce more communities. When `n_clusters` is supplied, omitted `resolution` or `resolution = NULL` asks faissR to seed the bounded target-count resolution grid from the graph-shape heuristic instead of the numeric default. |
| `n_clusters` | Optional target number of communities for Louvain/Leiden. If supplied, faissR builds the KNN graph once, evaluates a bounded deterministic resolution grid, and keeps the result closest to the requested count. Explicit numeric `resolution` values center the grid near that value and graph shape; omitted `resolution` or `resolution = NULL` uses the no-pilot target-count graph-shape seed directly. The grid is shape-aware: large graphs use fewer deterministic candidates than small graphs to reduce repeated clustering passes. This is a convenience target, not a hard guarantee. If `n_clusters` is supplied and `method` is omitted, faissR uses `"louvain"` as the target-count clustering method. The target must be a positive integer and cannot exceed the graph vertex count. |
| `objective_function` | Community objective. Only `"modularity"` is currently implemented by the native CPU and CUDA clustering paths; CPM is not accepted until faissR can optimize and report it honestly. |
| `n_iterations` | Maximum native clustering iterations. |
| `steps` | Random-walk propagation depth for `method = "random_walking"`. |
| `seed` | Optional seed for reproducible repeated runs. |
| `...` | Reserved for future backend-specific options. |

Returns a `faissR_graph_cluster` object with `membership`, `modularity`,
parameters, backend metadata, and source acknowledgements. `backend` records
the clustering implementation that actually ran, while
`parameters$requested_backend` and `parameters$resolved_backend` record the
public backend request and the device policy after resolving `"auto"`.
When `graph_cluster()` builds the graph internally or receives a `faissR_graph`,
`parameters$graph_backend`, `parameters$graph_requested_backend`, and
`parameters$graph_resolved_backend` separate the concrete KNN implementation
from the public graph backend request and resolved KNN route.
`parameters$n_vertices` and `parameters$n_edges` record the clustered graph
size for benchmark summaries. `parameters$nn_metric_transform` and
`parameters$nn_distance_transform` preserve normalized cosine/correlation graph
construction metadata from the KNN route. Direct graph builds also preserve
compact KNN route metadata in `parameters$nn_approximation`,
`parameters$nn_faiss`, `parameters$nn_cuvs`, `parameters$nn_spatial_index`,
and `parameters$nn_auto_selection` when those fields are
attached to the internal `nn()` result.
When a target community count is used,
`target_n_clusters`, `selected_resolution`, `target_gap`,
`resolution_selection`, and `resolution_search` record the requested target,
selected resolution, final community-count gap, deterministic selection rule,
candidate center, and full resolution search table. `parameters$resolution` is
`NA` for target-auto searches where `n_clusters` is supplied and `resolution`
is omitted or `NULL`; the actual automatic center is stored in
`resolution_selection$candidate_center`.
`parameters$resolution_source` is `"default"`, `"user"`, or `"target_auto"`
depending on whether the grid seed came from the default, an explicit numeric
`resolution`, or omitted/`NULL` `resolution` with `n_clusters`.

### Graph Clustering Methods

| `method` | Description |
| --- | --- |
| `"random_walking"` | Native CPU random-walk label-propagation/community method inspired by walktrap-style random-walk clustering and local parallel random-walk literature. CUDA random-walking is not enabled yet [10,19]. |
| `"louvain"` | Native CPU Louvain modularity local-moving implementation, with optional RAPIDS libcugraph CUDA execution when faissR is built with cuGraph [9,12]. |
| `"leiden"` | Native CPU Leiden-style local moving plus refinement to split disconnected communities, with optional RAPIDS libcugraph CUDA execution when available. The CPU implementation acknowledges Leiden and shared-memory/dynamic Leiden work [11-12,17-18]. |

## `fast_kmeans()`

```r
fast_kmeans(data, centers, backend = "auto",
            max_iter = "auto", n_init = "auto", tol = "auto",
            seed = 1L, n_threads = NULL,
            streaming_batch_size = 0L, init = "kmeans++",
            tuning = "auto")
```

| Argument | Description |
| --- | --- |
| `data` | Numeric matrix with observations in rows. |
| `centers` | Number of clusters. Must be between 1 and `nrow(data)`. |
| `backend` | `"auto"`, `"cpu"`, or `"cuda"`. `"auto"` uses CUDA only when CUDA plus FAISS GPU k-means or direct cuVS k-means is compiled and available and the shape rule estimates enough work to offset GPU launch and copy overhead; otherwise it resolves to CPU [7-8]. |
| `max_iter` | Maximum number of Lloyd iterations, or `"auto"` for a deterministic shape-aware default. |
| `n_init` | Number of random restarts where the selected backend supports it, or `"auto"` for a deterministic shape-aware default. |
| `tol` | Non-negative convergence tolerance where supported, or `"auto"` for a deterministic shape-aware default. |
| `seed` | Random seed for CPU/statistics and FAISS paths. The direct cuVS C API path currently does not expose an explicit seed in the stable params structure, so repeated cuVS runs should be interpreted as backend-controlled initialization. |
| `n_threads` | CPU worker threads for FAISS/statistics paths. |
| `streaming_batch_size` | cuVS host-data streaming batch size. Use `0` to let cuVS choose its default. |
| `init` | Initialization method: `"kmeans++"` or `"random"` where supported. |
| `tuning` | `"auto"` uses deterministic rules based on `nrow(data)`, `ncol(data)`, `centers`, and `n / centers`; small many-cluster jobs can use extra restarts without pilot runs, while large/high-dimensional jobs use cheaper defaults. `centers = 1` uses the exact column-mean solution for every backend request and records `single_cluster_exact_mean`; `centers = nrow(data)` uses the exact singleton assignment for every backend request and records `singleton_exact_identity`. Explicit CUDA requests for these exact cases return the trivial result and do not launch GPU work. `"fixed"`, `"off"`, and `"none"` keep the historical defaults unless explicit parameter values are supplied. |

Returns cluster labels, centers, within-cluster sums of squares, cluster sizes,
iteration count, `converged`, `hit_max_iter`, backend, and parameters,
including the k-means tuning rule used plus shape metadata, and whether
`max_iter`, `n_init`, and `tol` were
auto-selected or supplied explicitly. `parameters$tuning$rule` is a stable
grouping label such as `small_low_work_multistart`,
`medium_single_start`, or `large_fast_convergence`; `rule_detail` preserves
the exact shape/work values used for that decision.
`parameters$tuning$effective` records the
final values used after explicit overrides and `"auto"` defaults have been
resolved; `parameters$tuning$effective_max_iter`,
`parameters$tuning$effective_n_init`, and `parameters$tuning$effective_tol`
expose the same values as flat fields for benchmark summaries.
`parameters$tuning$backend_policy` records the deterministic `backend = "auto"`
shape decision, including `prefer_cuda`, `reason`, estimated work, ordinary R
input bytes, float32 GPU transfer bytes, and `n_per_center`. The CUDA auto gate
uses `gpu_transfer_nbytes` for the size threshold because FAISS/cuVS consume
float32 data; `nbytes` remains the R double input footprint for compatibility
and auditing. The CUDA auto gate can be adjusted for a benchmarked machine
with `options(faissR.kmeans_cuda_work_threshold = ...)`,
`options(faissR.kmeans_cuda_nbytes_threshold = ...)`,
`options(faissR.kmeans_cuda_large_n_threshold = ...)`, and
`options(faissR.kmeans_cuda_large_p_threshold = ...)`, and
`options(faissR.kmeans_cuda_min_n_per_center = ...)`; these options only
change the static threshold rule and do not run pilot tuning.
`parameters$tuning$selection` records the compact no-pilot device decision;
`selection$explicit_backend` and `selection$backend_decision` distinguish
explicit `"cpu"`/`"cuda"` calls from automatic shape-policy choices.
`hit_max_iter` records whether the run reached the effective iteration cap; this
helps benchmark cycles identify fast settings that may be under-iterating.
`parameters$requested_backend` records the public backend argument,
`parameters$resolved_backend` records the public device policy after resolving
`"auto"`, and `backend` records the implementation
that actually ran, such as `"faiss"`, `"cpu"`, `"cuda_faiss"`, or `"cuda_cuvs"`.

## `knn()`

```r
model <- knn(Xtrain, Ytrain, backend = "auto", method = "auto",
             tuning = "auto", cagra_implementation = NULL, k = 15L)
pred  <- knn(Xtrain, Ytrain, Xtest, type = "response")
prob  <- knn(Xtrain, Ytrain, Xtest, type = "prob")
```

| Argument | Description |
| --- | --- |
| `Xtrain` | Numeric training matrix with observations in rows. |
| `Ytrain` | Training labels for classification or numeric response for regression. Must have one value per row of `Xtrain`. |
| `Xtest` | Optional query matrix. If supplied, `knn()` fits and predicts immediately; otherwise it returns a reusable model. |
| `backend` | Device backend passed to `nn()`: `"auto"`, `"cpu"`, or `"cuda"`. `"auto"` follows `nn()` backend/method/metric resolution, using CUDA only for validated CUDA combinations when CUDA/cuVS runtime support is available, and CPU otherwise. |
| `method` | Nearest-neighbour algorithm selector passed to `nn()`. `"auto"` chooses the most appropriate method for the selected backend. |
| `metric` | Distance metric passed to `nn()`: `"euclidean"`, `"cosine"`, `"correlation"`, or `"inner_product"`. Aliases such as `"l2"`, `"cor"`/`"pearson"`, and `"ip"` are accepted and stored as canonical metric labels. Correlation is centered cosine similarity, not raw inner product. |
| `tuning` | Tuning policy passed to `nn()`. `"auto"` uses the deterministic default for the resolved method; `"cache"` and `"pilot"` opt into pilot tuning where implemented. FAISS GPU IVF pilot/cache tuning is Euclidean-only; non-Euclidean IVF routes use deterministic metric-aware defaults. |
| `cagra_implementation` | CUDA CAGRA provider passed to `nn()` for fitting/immediate prediction. |
| `task` | `"auto"`, `"classification"`, or `"regression"`. `"auto"` treats numeric `Ytrain` as regression and non-numeric `Ytrain` as classification. |
| `k` | Default number of neighbours used for prediction. |
| `n_threads` | CPU worker threads passed to `nn()`. |
| `vote` | `"majority"` for unweighted voting/means or `"weighted"` for inverse-distance weighted voting/means. Used for immediate prediction. |
| `type` | `"response"` for class labels or regression values; `"prob"` for classification probability matrices. |
| `...` | Reserved for future prediction options. |

When `Xtest` is omitted, the return value is a `faissR_knn_model`. Immediate
prediction outputs carry `attr(result, "faissR_nn")` metadata from the
underlying `nn()` route, including requested backend/method/tuning, resolved
backend, metric, `k`, whether the route was exact, approximation parameters,
FAISS/cuVS/native route metadata, normalized metric transforms, and
auto-selection metadata when present.

## `predict()`

```r
predict(object, newdata, k = NULL,
        backend = "auto", tuning = "auto", cagra_implementation = NULL,
        vote = "majority", type = "response", ...)
```

| Argument | Description |
| --- | --- |
| `object` | A fitted model returned by `knn(Xtrain, Ytrain, ...)`. |
| `newdata` | Numeric query matrix with the same number of columns as the training matrix. |
| `k` | Number of neighbours for this prediction call. If `NULL`, uses the model default. |
| `backend` | Device backend for the prediction-time neighbour search: `"auto"`, `"cpu"`, or `"cuda"`. The fitted model's method and metric are reused. |
| `tuning` | Prediction-time tuning policy. `"auto"` uses the deterministic default for the resolved method; `"cache"` and `"pilot"` opt into pilot tuning. |
| `cagra_implementation` | CUDA CAGRA provider for this prediction call. `NULL` reuses the fitted model setting, then the global option. |
| `vote` | `"majority"` for unweighted classification votes or regression means; `"weighted"` for inverse-distance weighting. |
| `type` | `"response"` for predicted labels/values or `"prob"` for classification probabilities. |
| `...` | Reserved for future options. |

For classification, use `predict(type = "prob")` to return class
probabilities. Prediction outputs carry the same `attr(result, "faissR_nn")`
route metadata, approximation parameters, and auto-selection metadata as
immediate `knn(..., Xtest)` predictions.

## Availability Helpers

```r
backend_info()
faiss_available()
faiss_gpu_available()
cuda_available()
cuvs_available()
cugraph_available()
```

| Function | Arguments | Description |
| --- | --- | --- |
| `backend_info()` | None. | Returns a data frame with backend availability, public call hints, public backend names, compact method/metric summaries, non-public implementation route labels, device/runtime hints, and notes. |
| `faiss_available()` | None. | Returns `TRUE` when faissR was compiled and linked against FAISS. |
| `faiss_gpu_available()` | None. | Returns `TRUE` when the linked FAISS build reports GPU support. |
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
