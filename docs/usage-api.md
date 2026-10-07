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
| `nn()` | Low-level nearest-neighbour search for reference/query matrices, including self-excluding search with `exclude_self = TRUE` [1-6,13-16,22-23]. |
| `nn_gpu()` | CUDA exact KNN with GPU-resident output buffers for downstream C/C++ packages. |
| `gpu_knn_to_host()` | Explicitly copy a `faissR_gpu_knn` result back to R matrices for inspection. |
| `candidate_knn()` | Exact top-k ranking inside a supplied candidate-neighbour matrix. |
| `fast_kmeans()` | CPU/FAISS/CUDA/cuVS k-means where available [7-8]. |
| `knn()` | Fit a reusable kNN classifier/regressor or fit and predict immediately. |
| `predict()` | S3 method for `faissR_knn_model`; predicts labels, numeric responses, or class probabilities from `knn()`. |
| `backend_info()` | Report available CPU, FAISS, CUDA, and cuVS capabilities. |
| `nn_capabilities()` | Report supported nearest-neighbour method/backend/metric combinations for preflight checks; `runtime = TRUE` adds current-build availability columns. |
| `nn_metric_preflight()` | Report non-finite rows and metric-degenerate zero/constant rows, together with the requested backend's action, without running a search. |
| `faiss_available()` | Logical check for compiled/linked FAISS CPU support. |
| `faiss_gpu_available()` | Logical check for FAISS GPU support in the linked FAISS build. |
| `cuda_available()` | Logical check for native CUDA support and an available CUDA runtime/device. |
| `cuvs_available()` | Logical check for direct RAPIDS cuVS support. |

## C/C++ Callable API

faissR registers a small stable ABI for downstream R packages that need to call
nearest-neighbour code from C/C++ without going through the R wrapper layer.
Downstream packages should list `faissR` in `LinkingTo`, include
`<faissR_api_v1.h>`, and use its typed getter functions. The compatibility
header `<faissR_api.h>` includes the current versioned header.

| Name | Signature | Description |
| --- | --- | --- |
| `faissR_c_api_version` | `int(void)` | Returns `1` for the installed callable ABI. Call the header helper before using the remaining entry points. |
| `faissR_nn_float32_call` | `(SEXP x, SEXP k, SEXP backend, SEXP metric, SEXP include_self, SEXP n_threads)` | CPU FAISS Flat float32 KNN. Accepts ordinary R double matrices or optional `float::fl()`/float32 matrices. Returns the stable host KNN list with double distances. |
| `faissR_nn_float32_call_output` | `(SEXP x, SEXP k, SEXP backend, SEXP metric, SEXP include_self, SEXP n_threads, SEXP distances)` | Same CPU FAISS Flat float32 route, with `distances = "double"` or `"float"` for the returned host distance matrix. |
| `faissR_nn_cuda_tuned_gpu_call` | `(SEXP x, SEXP k, SEXP method, SEXP metric, SEXP include_self, SEXP target_recall)` | CUDA self-KNN route for `method = "auto"`, `"exact"`, `"flat"`, or `"bruteforce"`. Returns a `faissR_gpu_knn` object with CUDA-device `indices_ptr` and `distances_ptr`, `result_residency = "cuda"`, and `device_to_host_result_copies = 0`. |
| `faissR_hnsw_search_v1` | `(SEXP data, SEXP query, SEXP n, SEXP p, SEXP k, SEXP target_recall, SEXP n_threads, SEXP distance_storage)` | One-shot CPU FAISS HNSW search with shape-aware parameter selection. |
| `faissR_hnsw_index_build_v1` | `(SEXP data, SEXP M, SEXP efConstruction, SEXP n_threads)` | Build and return an owning external pointer to a CPU FAISS HNSW index from float32 reference data. |
| `faissR_hnsw_index_search_v1` | `(SEXP index, SEXP query, SEXP k, SEXP efSearch, SEXP n_threads)` | Search an owned HNSW index repeatedly without rebuilding it. Returns one-based identifiers, Euclidean distances, and the requested and effective settings. |

The HNSW index pointer owns its native allocation through an R finalizer. Keep
the pointer reachable for the complete query workload. It is process-local,
not serializable, and not safe for concurrent calls. Each search reports a
cumulative `query_call_count`, making index reuse explicit.

The GPU-resident ABI is intentionally exact-family only at present. Approximate
CUDA methods such as IVF, CAGRA, HNSW, NN-descent, NSG, Vamana, and IVFPQ
FastScan are available through `nn()` where supported, but their provider result
buffers are not yet exposed through persistent GPU-resident ownership.
The `handle` component of a `faissR_gpu_knn` object owns its CUDA allocations
and must remain protected while either device pointer is in use.
Both host and GPU C-callable objects expose the same distance-contract fields
query rows; `similarity_materialized = FALSE` confirms that no raw-score host
matrix was created.

## `nn()`

```r
nn(data, points = data, k = NULL, exclude_self = FALSE, backend = NULL,
   method = "auto", metric = "euclidean", tuning = "auto",
   target_recall = 0.99,
   cagra_implementation = NULL, cagra_build_algo = NULL,
   output = "double", distances = NULL,
   n_threads = NULL)
```

| Argument | Description |
| --- | --- |
| `data` | Numeric matrix, data frame, or optional `float::fl()`/`float32` matrix with reference observations in rows and features in columns. FAISS CPU/GPU and RAPIDS cuVS nearest-neighbour routes use direct float-pointer input adapters without converting the float32 source object to an R double matrix. Resolved native routes without a direct float32 adapter fail clearly instead of silently converting benchmark input back to double. |
| `points` | Optional query matrix/data frame/float32 matrix with the same number of columns as `data`. Defaults to `data` for self-search. Float32 reference/query inputs can be mixed with ordinary R double matrices; direct FAISS/cuVS adapters convert only the double side once to row-major float32. |
| `k` | Number of neighbours to return. If `NULL`, faissR chooses an automatic neighbourhood size. |
| `exclude_self` | Logical; if `TRUE`, remove each query row from its own neighbour list. This is valid only for self-query calls where `points` is omitted or identical to `data`. The flag is passed into the compiled backend path, so self-neighbour removal is handled in C++/CUDA rather than by R-side row filtering. |
| `backend` | Device backend: `"cpu"` or `"cuda"`. `NULL` follows `options(faissR.backend)`, then `FAISSR_BACKEND`, and finally uses CPU. CUDA requests fail when CUDA support is unavailable. |
| `method` | Algorithm selector: `"auto"`, `"exact"`, `"flat"`, `"bruteforce"`, `"grid"`, `"hnsw"`, `"ivf"`, `"ivfpq"`, `"vamana_style"`, `"nsg_style"`, `"nndescent_style"`, `"ivfpq_fastscan"`, or `"cagra"` [1-6,13-16,22-24,34]. The shorter graph-family names remain compatibility aliases. Resolved implementation labels are metadata, not public values. `method = "grid"` maps to native CPU or CUDA code according to `backend`; unsupported combinations stop clearly. |
| `metric` | Canonical distance metric: `"euclidean"`, `"cosine"`, or `"correlation"`. Euclidean output is ordinary L2, not squared L2. Legacy aliases are rejected. |
| `tuning` | Tuning policy: `"auto"`, `"cache"`, `"pilot"`, `"fixed"`, `"off"`, or `"none"`. Automatic policies are selected in compiled code. |
| `target_recall` | Requested recall tier: exactly `0.9`, `0.95`, or `0.99`. Other values error; faissR does not round or interpolate. The tier is not a guarantee for a new dataset. |
| `cagra_implementation` | CUDA CAGRA provider for this call. `NULL` uses `options(faissR.cagra_implementation = ...)`; `"auto"` uses a deterministic shape-aware provider rule, selecting direct cuVS CAGRA for compact high-dimensional self-KNN and otherwise keeping FAISS GPU CAGRA as the default when both providers are available; `"faiss_gpu"` or `"cuvs"` force one provider for benchmark rows. This affects `backend = "cuda", method = "cagra"` and CUDA `method = "auto"` routes that select CAGRA. |
| `cagra_build_algo` | Direct RAPIDS cuVS CAGRA graph-build algorithm for this call. `NULL` uses `options(faissR.cuvs_cagra_build_algo = "auto")`; for direct cuVS CAGRA, `"auto"` applies faissR's deterministic shape-aware build rule, choosing iterative CAGRA construction for compact high-dimensional self-KNN cases and IVF-PQ construction otherwise. `"ivf_pq"` requests the IVF-PQ graph builder, `"nn_descent"` requests cuVS NN-descent graph construction, and `"iterative_cagra_search"` requests cuVS iterative CAGRA graph building. This is a CAGRA construction parameter, not a fallback to a different public method, and successful results record it in `route_parameters`. |
| `output` | Distance storage type: `"double"` returns the default R numeric matrix; `"float"` returns `distances` as a `float::fl()`/`float32` matrix and records `distance_type = "float32"` plus `attr(result, "distance_type") = "float32"`. Direct FAISS/cuVS float routes can construct float distances without first materializing an R double distance matrix, including CPU FAISS Flat/IVF/IVFPQ/FastScan, cached CPU FAISS fitted indexes, FAISS GPU Flat/IVF/IVFPQ, and direct Euclidean RAPIDS cuVS routes. Float32-route results expose `input_layout`, `input_owns_data`, and `float32_compatibility_conversion` so callers can distinguish direct float payload use from one-time double-to-float adaptation; unsupported native float32 routes error. The `float` package is optional and used only when this output is requested or a float32 input object is supplied. |
| `distances` | Optional alias for `output`; use `distances = "float"` when downstream code wants the returned distance matrix to remain float32. |
| `n_threads` | Number of CPU worker threads for CPU/FAISS CPU backends. GPU backends ignore this argument. |

For cosine and correlation, zero-normalized rows use the documented finite
edge-case convention on CPU, whereas explicit CUDA requests reject them rather
than applying CPU repair. All backends reject non-finite values. Call
`nn_metric_preflight(data, points, metric, backend)` to obtain the affected
one-based row indices and a stable action label before search. CUDA exact and
brute-force requests use direct cuVS brute
force when available, while FAISS GPU Flat remains the provider-backed
alternative.

For package-owned graph refinement, prefer `"nsg_style"`,
`"vamana_style"`, and `"nndescent_style"`. The shorter historical names remain
compatibility aliases. The `_style` suffix is a scope marker rather than an
algorithm name: package-owned routes are distinct derived graph-refinement
algorithms, while CUDA `nndescent_style` can resolve to direct cuVS
NN-descent. Package-owned implementations are experimental and excluded from
the publication's principal comparative performance claims. Result fields
`preferred_public_method`, `implementation_label`, `implementation_scope`,
`implementation_status`, `experimental`, and
`canonical_reimplementation` distinguish these derived algorithms from
direct external-provider routes.

Advanced tuning and cache knobs use `options(faissR.<name> = ...)`.
For cosine and correlation, FAISS/cuVS normalized routes store the transformed
row-normalized data as row-major float32 buffers in a small session cache keyed
by matrix contents, dimensions, and metric. The cache is enabled by default,
bounded by `options(faissR.cache_transformed_float32_max_entries = 4L)`, and can
be disabled with `options(faissR.cache_transformed_float32 = FALSE)`. Results
that use the cache report hit/miss metadata in
`attr(result, "faiss")$transform_cache`,
`attr(result, "cuvs")$transform_cache`, or
`attr(result, "approximation")$transform_cache`, depending on the route.

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

## `nn_gpu()`

```r
nn_gpu(data, points = data, k = NULL, exclude_self = FALSE,
       method = "auto", metric = "euclidean",
       tuning = "auto", target_recall = 0.99)
```

| Argument | Description |
| --- | --- |
| `data` | Numeric matrix, data frame, or optional `float::fl()`/float32 reference matrix. |
| `points` | Optional query matrix/data frame/float32 matrix with the same number of columns as `data`; defaults to `data` for self-search. |
| `k` | Number of neighbours. If `NULL`, faissR chooses an automatic neighbourhood size. |
| `exclude_self` | Logical; if `TRUE`, remove each query row from its own neighbour list in the CUDA kernel. This is valid only for self-query calls. |
| `method` | GPU-resident method selector. `"auto"` consults the compiled shape/k/metric/target-recall selector, but currently returns exact-family GPU-resident buffers; `"exact"`, `"flat"`, and `"bruteforce"` force the same resident exact search family. |
| `metric` | Canonical distance metric: `"euclidean"`, `"cosine"`, or `"correlation"`. Legacy aliases are rejected. |
| `tuning` | Tuning policy: `"auto"`, `"cache"`, `"pilot"`, `"fixed"`, `"off"`, or `"none"`. Automatic policies are selected in compiled code. |
| `target_recall` | Requested recall tier: exactly `0.9`, `0.95`, or `0.99`; no rounding or interpolation. |

`nn_gpu()` is for downstream CUDA consumers that need the KNN output to stay on
the GPU. It returns a `faissR_gpu_knn` object with an owning `handle` plus
non-owning `indices_ptr` and `distances_ptr` external pointers. The device
layout is column-major `n_query x k`, with 1-based int32 indices and float32
distances. Keep `handle` alive for as long as another package uses either
pointer.

The current GPU-resident route is exact search for `method = "auto"`,
`"exact"`, `"flat"`, or `"bruteforce"`. Euclidean inputs above three
dimensions use FAISS GPU direct distance search; 2D/3D Euclidean uses the
native direct-difference CUDA kernel.
Cosine and correlation use the native CUDA GPU-resident exact route.
`target_recall` is recorded for API symmetry. The route is exhaustive, but raw
set-overlap recall can differ from 1.0 at tied boundaries; publication analyses
therefore use the separate `exact_audited` criterion. With `method = "auto"`,
the object also records
`auto_preferred_backend`, `auto_preferred_method`, and
`auto_residency_constraint` when ordinary `nn()` would choose an approximate
CUDA backend that is not yet exposed as a persistent GPU-resident result. It
also includes the executed exact-family `execution_tuning` and, when available,
the compiled-policy `auto_preferred_tuning` for the preferred approximate CUDA
method.
Approximate FAISS GPU/cuVS methods still return host objects through `nn()` until their provider result buffers
are exposed with persistent GPU ownership.

Use `gpu_knn_to_host(x)` only when you explicitly want to inspect or test a
GPU-resident result in R; it copies both result buffers to host matrices.

### Ownership and CUDA execution contract

- `handle` is the sole owner of both CUDA allocations. `indices_ptr` and
  `distances_ptr` are non-owning views. Retain the complete result (or at least
  its owning handle) until every downstream CUDA operation has completed.
- The handle finalizer releases both allocations when the owner becomes
  unreachable. Garbage collection is not a synchronization mechanism. An
  asynchronous consumer must preserve the owner and synchronize its stream or
  record an equivalent completion event before allowing finalization.
- The producing route calls `cudaDeviceSynchronize()` before returning. No
  faissR stream or provider-resource handle is exported. A consumer may select
  its own stream after return, but must first select the recorded `device` and
  must not assume that its library-specific FAISS/cuVS resource object is the
  same object used by faissR.
- The pointers are ordinary same-process CUDA allocations on one device. They
  are not CUDA IPC handles, are not valid on another device or process, and
  require a CUDA-runtime-compatible consuming package. Different CUDA resource
  managers may use the buffers after the synchronized handoff; they do not
  share allocator or stream ownership.
- External-pointer GPU results and fitted indexes are session objects. Saving
  them with `saveRDS()` does not create a reusable GPU/index serialization.
  `serialization_supported = FALSE` and
  `interprocess_sharing_supported = FALSE` make this explicit.
- `gpu_knn_to_host()` copies the two complete buffers but does not invalidate
  the original GPU object. It remains resident until its owner is finalized.

## `gpu_knn_to_host()`

```r
gpu_knn_to_host(x)
```

| Argument | Description |
| --- | --- |
| `x` | A `faissR_gpu_knn` object returned by `nn_gpu()` or the C-callable GPU-result API. |

This helper explicitly copies a GPU-resident KNN result back to ordinary R
matrices. It is intended for diagnostics, tests, or handoff to code that cannot
consume CUDA device pointers. It is never called automatically by `nn_gpu()`.

### Nearest-Neighbour Methods

| `method` | Description |
| --- | --- |
| `"auto"` | Shape-aware selector for the chosen backend. CPU auto is a calibration-informed experimental static policy and returns `auto_policy_status = "calibration_informed_not_independently_validated"`; it uses exact search for tiny data, grid search for large 2D/3D self-search, FAISS HNSW for most medium/high-dimensional CPU self-KNN, FAISS IVF for selected large low-dimensional Euclidean rows, native CPU NSG/MRNG-derived refinement for selected larger non-Euclidean self-KNN cases, and native CPU NN-descent for other large self-KNN cases [1-2,5,21]. CUDA auto is also experimental: it is L40S-calibrated for cold full-self-search. It uses CUDA grid for large 2D/3D self-search; Euclidean non-grid self-KNN chooses exact Flat/brute force or IVF-Flat from the compiled shape/k/requested-recall-tier policy; non-grid cosine and correlation stay on exact FAISS GPU Flat or cuVS brute force when available [1-3,13-16]. Neither selector accepts a memory budget, latency/build deadline, or prohibition on exhaustive search. |
| `"grid"` | Native spatial grid search for 2D/3D Euclidean, cosine, and correlation self-KNN. Cosine/correlation use normalized Euclidean grid search. It is intended for low-dimensional spatial or simulated data and errors clearly outside supported dimensions. |

Use `nn(..., exclude_self = TRUE)` for embedding
workflows where each row should not list itself as its nearest neighbour.

All host and GPU KNN objects declare `distance_is_metric`,
`distance_semantics`, `distance_comparable_across_queries`, and
`distance_order`.

For CPU FAISS Flat/HNSW/IVF/IVFPQ/IVFPQ FastScan routes, raw `nn()` calls
reuse a bounded session-local fitted-index cache when the reference data and
method parameters match a previous call. This is especially useful for repeated
self-KNN, graph, and benchmark calls. Metadata reports `persistent_index_cache`
and `index_cache_hit`; use `options(faissR.cache_fitted_nn_indexes = FALSE)` to
disable the cache or `faissR.cache_fitted_nn_indexes_max_entries` to bound the
number of retained FAISS external pointers.

## `candidate_knn()`

```r
candidate_knn(data, candidates, points = data, k,
              backend = NULL, metric = "euclidean",
              n_threads = NULL, exclude_self = FALSE)
```

| Argument | Description |
| --- | --- |
| `data` | Numeric reference matrix with observations in rows. |
| `candidates` | Integer matrix of 1-based candidate reference row indices. It must have one row per query. Invalid, missing, zero, or out-of-range entries are ignored. |
| `points` | Optional query matrix. Defaults to `data` for self-query candidate scoring. |
| `k` | Number of best neighbours to keep from each candidate row. Must be no larger than `ncol(candidates)`. |
| `backend` | `"cpu"` for exact CPU scoring inside candidates or `"cuda"` for the native CUDA row-candidate kernel. `NULL` follows the package backend configuration. |
| `metric` | Canonical distance metric: `"euclidean"`, `"cosine"`, or `"correlation"`. Legacy aliases are rejected. |
| `n_threads` | CPU worker threads. |
| `exclude_self` | If `TRUE`, remove each row from its own candidate list. This requires `points = data`. |

This function does not generate candidates; it only reranks candidates supplied
by another method. Missing, invalid, and duplicate candidates do not expand
the candidate set. If fewer than `k` valid distinct candidates remain, both
backends return `NA` indices and `Inf` distances for the missing positions.
CUDA requires self-query scoring with `exclude_self = TRUE`, at least two
reference rows, and `k <= 256`; zero cosine rows and constant correlation rows
produce an explicit error rather than CPU-side repair.

## `fast_kmeans()`

```r
fast_kmeans(data, centers, backend = NULL,
            max_iter = "auto", n_init = "auto", tol = "auto",
            seed = 1L, n_threads = NULL,
            streaming_batch_size = 0L, init = "kmeans++",
            tuning = "auto")
```

| Argument | Description |
| --- | --- |
| `data` | Numeric matrix with observations in rows. |
| `centers` | Number of clusters. Must be between 1 and `nrow(data)`. |
| `backend` | `"cpu"` or `"cuda"`. `NULL` follows the package backend configuration. CUDA requests fail when no CUDA k-means provider is available [7-8]. |
| `max_iter` | Maximum number of Lloyd iterations, or `"auto"` for a deterministic shape-aware default computed by the compiled C++ tuning helper. |
| `n_init` | Number of random restarts where the selected backend supports it, or `"auto"` for a deterministic shape-aware default computed by the compiled C++ tuning helper. |
| `tol` | Non-negative convergence tolerance where supported, or `"auto"` for a deterministic shape-aware default computed by the compiled C++ tuning helper. |
| `seed` | Random seed for CPU/statistics and FAISS paths. The direct cuVS C API path currently does not expose an explicit seed in the stable params structure, so repeated cuVS runs should be interpreted as backend-controlled initialization. |
| `n_threads` | CPU worker threads for FAISS/statistics paths. |
| `streaming_batch_size` | cuVS host-data streaming batch size. Use `0` to let cuVS choose its default. |
| `init` | Initialization method: `"kmeans++"` or `"random"` where supported. |
| `tuning` | Tuning policy: `"auto"`, `"cache"`, `"pilot"`, `"fixed"`, `"off"`, or `"none"`. Automatic policies are selected in compiled code. |

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
When `centers = 1`, `fast_kmeans()` returns the exact column mean, records
`single_cluster_exact_mean`, and avoids iterative work on every backend.
The k-means automatic parameter rule is computed by
`kmeans_auto_params_cpp()` and records `tuning_source = "cpp"` in returned
metadata. `parameters$tuning$selection` records the explicit device, input
shape, runtime capability flags, and effective parameters.
`hit_max_iter` records whether the run reached the effective iteration cap; this
helps benchmark cycles identify fast settings that may be under-iterating.
`parameters$requested_backend` records the public backend argument,
`parameters$resolved_backend` records the selected device, and `backend`
records the implementation
that actually ran, such as `"faiss"`, `"cpu"`, `"cuda_faiss"`, or `"cuda_cuvs"`.
CUDA runs also record `parameters$cuda_provider_selection` as `"faiss_gpu"`,
`"direct_cuvs"`, or `"direct_cuvs_after_faiss_gpu_unavailable_or_failed"`;
`parameters$backend_resolution_note` describes the provider route, and
`parameters$faiss_gpu_error` is present when direct cuVS was used after a FAISS
GPU route was unavailable or failed.

## `knn()`

```r
model <- knn(Xtrain, Ytrain, backend = "cpu", method = "auto",
             tuning = "auto", target_recall = 0.99,
             cagra_implementation = NULL,
             cagra_build_algo = NULL, k = 15L)
pred  <- knn(Xtrain, Ytrain, Xtest, type = "response")
prob  <- knn(Xtrain, Ytrain, Xtest, type = "prob")
```

| Argument | Description |
| --- | --- |
| `Xtrain` | Numeric training matrix or optional `float::fl()`/`float32` matrix with observations in rows. Float32 training data is preserved for `nn()` methods with direct float32 adapters. |
| `Ytrain` | Training labels for classification or numeric response for regression. Must have one value per row of `Xtrain`. |
| `Xtest` | Optional query matrix. If supplied, `knn()` fits and predicts immediately; otherwise it returns a reusable model. |
| `backend` | Device backend passed to `nn()`: `"cpu"` or `"cuda"`. `NULL` follows the package backend configuration. |
| `method` | Nearest-neighbour algorithm selector passed to `nn()`. `"auto"` chooses the most appropriate method for the selected backend. |
| `metric` | Canonical distance metric: `"euclidean"`, `"cosine"`, or `"correlation"`. Legacy aliases are rejected. |
| `tuning` | Tuning policy: `"auto"`, `"cache"`, `"pilot"`, `"fixed"`, `"off"`, or `"none"`. Automatic policies are selected in compiled code. |
| `target_recall` | Requested recall tier: exactly `0.9`, `0.95`, or `0.99`; no rounding or interpolation. |
| `cagra_implementation` | CUDA CAGRA provider passed to `nn()` for fitting/immediate prediction. |
| `cagra_build_algo` | Direct RAPIDS cuVS CAGRA graph-build algorithm passed to `nn()` for direct cuVS CAGRA routes. |
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
CUDA FAISS/cuVS results additionally carry `attr(result, "gpu_residency")`,
with fields such as `gpu_provider`, `index_residency`,
`host_to_device_copies`, `query_reuses_device_data`, and `cpu_fallback`.
Direct cuVS self-search routes report whether the query reused the dataset
device buffer; FAISS GPU routes report FAISS-managed host/device transfers
because FAISS owns those internal copies.

For explicit CPU FAISS `method = "flat"`, `"hnsw"`, `"ivf"`, and `"ivfpq"`
models, the fitted model stores a session-local FAISS external pointer because
FAISS owns the indexed float32 vectors after `add()`. Flat uses exact
`IndexFlatL2`/`IndexFlatIP`; HNSW/IVF/IVFPQ use their corresponding FAISS CPU
index types. For IVF and IVFPQ, the trained coarse centroids and inverted-list
assignments are reused across compatible predictions; prediction can update
search-time `nprobe` for the requested `k` without retraining or re-adding the
training vectors. For IVFPQ, compatible predictions also reuse the trained
product-quantizer codebooks and compressed vector codes; metadata reports
`pq_codebooks_reused`, `pq_codes_reused`, and
`search_pq_train_call_count = 0`.
`predict()` reuses the fitted index when the requested backend, method, tuning,
and HNSW/IVF/IVFPQ `target_recall` requirements match the fitted model, and prediction
metadata reports `approximation$index_reused = TRUE`. If the model is saved and
reloaded in a later R session, or if prediction settings do not match,
`predict()` rebuilds the same route instead of switching algorithms.

Changing the recall tier bypasses an index built for the old tier and
recomputes its calibrated settings. The prediction records `query_source =
"nn"`; the original fitted object and its default tier are not modified.

## `predict()`

```r
predict(object, newdata, k = NULL,
        backend = NULL, tuning = "auto", target_recall = NULL,
        cagra_implementation = NULL, cagra_build_algo = NULL,
        vote = "majority", type = "response", ...)
```

| Argument | Description |
| --- | --- |
| `object` | A fitted model returned by `knn(Xtrain, Ytrain, ...)`. |
| `newdata` | Numeric query matrix or optional `float::fl()`/`float32` matrix with the same number of columns as the training matrix. Float32 query data is preserved for direct-adapter NN methods. |
| `k` | Number of neighbours for this prediction call. If `NULL`, uses the model default. |
| `backend` | Device backend for the prediction-time neighbour search: `"cpu"` or `"cuda"`. `NULL` follows the package backend configuration. The fitted model's method and metric are reused. |
| `tuning` | Tuning policy: `"auto"`, `"cache"`, `"pilot"`, `"fixed"`, `"off"`, or `"none"`. Automatic policies are selected in compiled code. |
| `target_recall` | Requested recall tier: exactly `0.9`, `0.95`, or `0.99`; no rounding or interpolation. |
| `cagra_implementation` | CUDA CAGRA provider for this prediction call. `NULL` reuses the fitted model setting, then the global option. |
| `cagra_build_algo` | Direct RAPIDS cuVS CAGRA graph-build algorithm for this prediction call. `NULL` reuses the fitted model setting, then the global option. |
| `vote` | `"majority"` for unweighted classification votes or regression means; `"weighted"` for inverse-distance weighting. |
| `type` | `"response"` for predicted labels/values or `"prob"` for classification probabilities. |
| `...` | Reserved for future options. |

For classification, use `predict(type = "prob")` to return class
probabilities. Prediction outputs carry the same `attr(result, "faissR_nn")`
route metadata, approximation parameters, and auto-selection metadata as
immediate `knn(..., Xtest)` predictions. The prediction-time neighbour search
is batched: `predict()` sends the full `newdata` matrix to the resolved
FAISS/cuVS/native NN route in one call, including when a reusable fitted FAISS
index is available. Metadata records `batch_query = TRUE`, `query_n`, and
`query_call_count = 1L` so benchmarks can verify that prediction did not query
one row at a time.

## Availability Helpers

```r
backend_info()
faiss_available()
faiss_gpu_available()
cuda_available()
cuvs_available()
```

| Function | Arguments | Description |
| --- | --- | --- |
| `backend_info()` | None. | Returns a data frame with backend availability, public call hints, public backend names, compact method/metric summaries, non-public implementation route labels, device/runtime hints, and notes. |
| `faiss_available()` | None. | Returns `TRUE` when faissR was compiled and linked against FAISS. |
| `faiss_gpu_available()` | None. | Returns `TRUE` when the linked FAISS build reports GPU support. |
| `cuda_available()` | None. | Returns `TRUE` when native CUDA support was compiled and a CUDA device/runtime is available. |
| `cuvs_available()` | None. | Returns `TRUE` when direct RAPIDS cuVS backends were compiled and can be loaded. |

## Typical Workflow

```r
library(faissR)

x <- scale(as.matrix(iris[, 1:4]))

nn_res <- nn(x, k = 15, exclude_self = TRUE,
             backend = "cpu", method = "auto", n_threads = 4)
nn_res$indices[1:3, 1:5]
```

For CUDA benchmarking on a GPU build:

```r
cuda_res <- nn(x, k = 15, exclude_self = TRUE,
               backend = "cuda", method = "auto")
cuda_device_res <- nn_gpu(x, k = 15, exclude_self = TRUE,
                          method = "exact")
```
