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
| `faiss_available()` | Logical check for compiled/linked FAISS CPU support. |
| `faiss_gpu_available()` | Logical check for FAISS GPU support in the linked FAISS build. |
| `cuda_available()` | Logical check for native CUDA support and an available CUDA runtime/device. |
| `cuvs_available()` | Logical check for direct RAPIDS cuVS support. |

## C/C++ Callable API

faissR registers a small stable ABI for downstream R packages that need to call
nearest-neighbour code from C/C++ without going through the R wrapper layer.
Retrieve the function pointer with `R_GetCCallable("faissR", "<name>")`.

| Name | Signature | Description |
| --- | --- | --- |
| `faissR_nn_float32_call` | `(SEXP x, SEXP k, SEXP backend, SEXP metric, SEXP include_self, SEXP n_threads)` | CPU FAISS Flat float32 KNN. Accepts ordinary R double matrices or optional `float::fl()`/float32 matrices. Returns the stable host KNN list with double distances. |
| `faissR_nn_float32_call_output` | `(SEXP x, SEXP k, SEXP backend, SEXP metric, SEXP include_self, SEXP n_threads, SEXP distances)` | Same CPU FAISS Flat float32 route, with `distances = "double"` or `"float"` for the returned host distance matrix. |
| `faissR_nn_cuda_tuned_gpu_call` | `(SEXP x, SEXP k, SEXP method, SEXP metric, SEXP include_self, SEXP target_recall)` | CUDA self-KNN route for `method = "auto"`, `"exact"`, `"flat"`, or `"bruteforce"`. Returns a `faissR_gpu_knn` object with CUDA-device `indices_ptr` and `distances_ptr`, `result_residency = "cuda"`, and `device_to_host_result_copies = 0`. |

The GPU-resident ABI is intentionally exact-family only at present. Approximate
CUDA methods such as IVF, CAGRA, HNSW, NN-descent, NSG, Vamana, and IVFPQ
FastScan are available through `nn()` where supported, but their provider result
buffers are not yet exposed through persistent GPU-resident ownership.

## `nn()`

```r
nn(data, points = data, k = NULL, exclude_self = FALSE, backend = "auto",
   method = "auto", metric = "euclidean", tuning = "auto",
   target_recall = 0.99,
   cagra_implementation = NULL, cagra_build_algo = NULL,
   output = "double", distances = NULL, n_threads = NULL)
```

| Argument | Description |
| --- | --- |
| `data` | Numeric matrix, data frame, or optional `float::fl()`/`float32` matrix with reference observations in rows and features in columns. FAISS CPU/GPU and RAPIDS cuVS nearest-neighbour routes use direct float-pointer input adapters without converting the float32 source object to an R double matrix. Resolved native routes without a direct float32 adapter fail clearly instead of silently converting benchmark input back to double. |
| `points` | Optional query matrix/data frame/float32 matrix with the same number of columns as `data`. Defaults to `data` for self-search. Float32 reference/query inputs can be mixed with ordinary R double matrices; direct FAISS/cuVS adapters convert only the double side once to row-major float32. |
| `k` | Number of neighbours to return. If `NULL`, faissR chooses an automatic neighbourhood size. |
| `exclude_self` | Logical; if `TRUE`, remove each query row from its own neighbour list. This is valid only for self-query calls where `points` is omitted or identical to `data`. The flag is passed into the compiled backend path, so self-neighbour removal is handled in C++/CUDA rather than by R-side row filtering. |
| `backend` | Device backend: `"auto"`, `"cpu"`, `"cuda"`, or `"metal"`. `"auto"` uses a validated CUDA route only when supported and otherwise resolves to CPU. Explicit `"cuda"` fails when CUDA support is unavailable. Explicit `"metal"` is a no-fallback Apple GPU route restricted to 2D/3D self-KNN with `method = "auto"` or `"grid"` and Euclidean, cosine, or correlation distance. |
| `method` | Algorithm selector: `"auto"`, `"exact"`, `"flat"`, `"bruteforce"`, `"grid"`, `"hnsw"`, `"ivf"`, `"ivfpq"`, `"vamana"`, `"nsg"`, `"nndescent"`, `"ivfpq_fastscan"`, or `"cagra"` [1-6,13-16,22-24,34]. These are canonical lowercase public labels; resolved implementation labels are metadata, not public values. `method = "grid"` maps to native CPU, CUDA, or Metal code according to `backend`. The Metal route supports no separate query, raw inner product, dimensions outside 2D/3D, or `k > 128`; unsupported combinations stop clearly. |
| `metric` | Distance metric: `"euclidean"`, `"cosine"`, `"correlation"`, or `"inner_product"`. Legacy aliases such as `"l2"`, `"cor"`, `"pearson"`, `"ip"`, and dot-product variants are rejected. Inner product is the raw dot product; cosine is the dot product after row L2 normalization; correlation is centered cosine similarity after subtracting each row mean and L2-normalizing each row. For `metric = "inner_product"`, neighbours are ranked by larger raw dot product, but returned `distances` keep faissR's smaller-is-better convention: within each query row the best returned dot product has distance `0`, and lower dot products have larger shifted distances. Euclidean is the validated high-performance route for approximate FAISS/CUDA/cuVS. Cosine and correlation use validated exact paths, FAISS CPU/GPU Flat, IVF-Flat, IVFPQ, CPU HNSW, CUDA cuVS HNSW, native CPU NSG/Vamana, and CUDA NSG/Vamana through normalized search. All-zero cosine rows and constant correlation rows are zero-normalized edge cases: faissR treats zero-vs-zero distance as `0` and zero-vs-nonzero distance as `1`; CPU FAISS Flat uses the exact CPU scorer for those rows to preserve deterministic small-`k` tie handling, while explicit CUDA routes error clearly rather than repairing those rows on CPU. Inner product is supported by native exact CPU scoring, FAISS Flat IP, FAISS IVF-Flat/IVFPQ IP, FAISS HNSW IP, native CPU NN-descent, native CPU NSG/Vamana, direct cuVS brute force through an exact MIPS-to-L2 transform, direct cuVS IVF/PQ through transformed approximate L2 indexes, CUDA CAGRA and CUDA cuVS HNSW through the same graph-search transform, CUDA NSG/Vamana self-KNN, and native CUDA NN-descent shifted-dot-product candidate refinement. |
| `tuning` | Tuning policy: `"auto"`, `"cache"`, `"pilot"`, `"fixed"`, `"off"`, or `"none"`. `"auto"` uses deterministic no-pilot defaults for the resolved method; these parameter rules are computed by C++ `nn_tune_*_cpp()` helpers and record `tuning_source = "cpp"` in result metadata. For CPU Euclidean/cosine/correlation/inner-product `method = "exact"`, `"flat"`, or `"bruteforce"`, the policy tunes FAISS Flat batch/cache behavior and records `exact_recall_by_construction = TRUE`. CUDA Euclidean/cosine/correlation/inner-product `method = "exact"` uses compiled FAISS GPU Flat query-batch/resource policies; inner-product rows use `benchmark_scripts/cuda_exact_inner_product_shape_tuning_defaults_from_seeded_euclidean_results.csv` and are marked validation-pending until the dedicated CUDA IP sweep replaces them. CUDA Euclidean/cosine/correlation/inner-product `method = "flat"` uses compiled FAISS GPU Flat query-batch/resource policies; inner-product rows use `benchmark_scripts/cuda_flat_inner_product_shape_tuning_defaults_from_seeded_euclidean_results.csv` and are marked validation-pending until `run_hpc_flat_tuning_cuda_inner_product.sh` replaces them. CUDA Euclidean/cosine/correlation/inner-product `method = "bruteforce"` uses compiled cuVS brute-force query-batch/resource policies; the correlation table is in `benchmark_scripts/cuda_bruteforce_correlation_shape_tuning_defaults_from_proxy_results.csv`, and the raw-inner-product table is in `benchmark_scripts/cuda_bruteforce_inner_product_shape_tuning_defaults_from_seeded_euclidean_results.csv`; both are seeded from measured Euclidean cuVS brute-force rows until corrected metric-specific sweeps replace them. For CPU Euclidean/cosine/correlation/inner-product `method = "hnsw"`, `"nsg"`, or `"vamana"`, CUDA Euclidean/cosine/correlation/inner-product `method = "hnsw"`, `"nsg"`, or `"vamana"`, it selects compiled graph-search tiers by shape, `k`, and `target_recall`; CUDA HNSW raw-inner-product rows use `benchmark_scripts/cuda_hnsw_inner_product_shape_tuning_defaults_from_seeded_euclidean_results.csv` and are marked validation-pending until `run_hpc_hnsw_tuning_cuda_inner_product.sh` replaces them, CUDA NSG correlation and raw-inner-product rows are seeded from `benchmark_scripts/cuda_nsg_correlation_shape_tuning_defaults_from_seeded_cosine_results.csv` and `benchmark_scripts/cuda_nsg_inner_product_shape_tuning_defaults_from_seeded_cosine_results.csv`, and CUDA Vamana correlation/raw-inner-product rows are seeded from `benchmark_scripts/cuda_vamana_correlation_shape_tuning_defaults_from_seeded_cosine_results.csv` and `benchmark_scripts/cuda_vamana_inner_product_shape_tuning_defaults_from_seeded_cosine_results.csv`. Seeded graph rows are flagged as validation-pending until rerun. `"cache"` and `"pilot"` opt into pilot tuning where implemented. |
| `target_recall` | Speed/recall tier. Use `0.9`, `0.95`, or `0.99`. CPU Euclidean/cosine/correlation/inner-product and CUDA Euclidean/cosine/correlation `method = "ivf"` use this value for compiled `nlist`/`nprobe` tiers; CPU Euclidean/cosine/correlation/inner-product and CUDA Euclidean/correlation `method = "ivfpq"` use it for compiled `nlist`/`nprobe`/`pq_m`/`pq_nbits` tiers. CPU IVF/IVFPQ and CUDA IVF/IVFPQ metadata records `tuning_benchmark_target_met` so best-available partial or below-target rows are not mistaken for guaranteed recall. CUDA `method = "auto"` uses this value when choosing between Flat/brute force and IVF-Flat for Euclidean self-KNN, and CPU/CUDA HNSW use it for graph-search tiers. CPU Euclidean/cosine/correlation/inner-product HNSW, NSG, and Vamana, plus CUDA Euclidean/cosine/correlation/inner-product NSG and Vamana, record `tuning_benchmark_target_met` for benchmark-derived or seeded rows; CPU Euclidean/cosine/correlation/inner-product and CUDA Euclidean/cosine/correlation/inner-product `method = "exact"` record the requested tier in exact tuning metadata, CPU Euclidean/cosine/correlation/inner-product and CUDA Euclidean/cosine/correlation/inner-product `method = "flat"` record it in flat-tuning metadata, and CPU Euclidean/cosine/correlation/inner-product plus CUDA Euclidean/cosine/correlation/inner-product `method = "bruteforce"` records it in bruteforce-tuning metadata. Exact, Flat, and brute-force recall is 1.0 by construction. |
| `cagra_implementation` | CUDA CAGRA provider for this call. `NULL` uses `options(faissR.cagra_implementation = ...)`; `"auto"` uses a deterministic shape-aware provider rule, selecting direct cuVS CAGRA for compact high-dimensional self-KNN and otherwise keeping FAISS GPU CAGRA as the default when both providers are available; `"faiss_gpu"` or `"cuvs"` force one provider for benchmark rows. This affects `backend = "cuda", method = "cagra"` and CUDA-auto routes that select CAGRA. |
| `cagra_build_algo` | Direct RAPIDS cuVS CAGRA graph-build algorithm for this call. `NULL` uses `options(faissR.cuvs_cagra_build_algo = "auto")`; for direct cuVS CAGRA, `"auto"` applies faissR's deterministic shape-aware build rule, choosing iterative CAGRA construction for compact high-dimensional self-KNN cases and IVF-PQ construction otherwise. `"ivf_pq"` requests the IVF-PQ graph builder, `"nn_descent"` requests cuVS NN-descent graph construction, and `"iterative_cagra_search"` requests cuVS iterative CAGRA graph building. This is a CAGRA construction parameter, not a fallback to a different public method, and successful results record it in `route_parameters`. |
| `output` | Distance storage type: `"double"` returns the default R numeric matrix; `"float"` returns `distances` as a `float::fl()`/`float32` matrix and records `distance_type = "float32"` plus `attr(result, "distance_type") = "float32"`. Direct FAISS/cuVS float routes can construct float distances without first materializing an R double distance matrix, including CPU FAISS Flat/IVF/IVFPQ/FastScan, cached CPU FAISS fitted indexes, FAISS GPU Flat/IVF/IVFPQ, and direct Euclidean RAPIDS cuVS routes. Float32-route results expose `input_layout`, `input_owns_data`, and `float32_compatibility_conversion` so callers can distinguish direct float payload use from one-time double-to-float adaptation; unsupported native float32 routes error. The `float` package is optional and used only when this output is requested or a float32 input object is supplied. |
| `distances` | Optional alias for `output`; use `distances = "float"` when downstream code wants the returned distance matrix to remain float32. |
| `n_threads` | Number of CPU worker threads for CPU/FAISS CPU backends. GPU backends ignore this argument. |

For CUDA `method = "cagra"`, Euclidean uses measured CAGRA shape/k/target
defaults. Cosine, correlation, and raw inner product currently use
validation-pending defaults seeded from the Euclidean CAGRA sweep; raw inner
product is stored in
`benchmark_scripts/cuda_cagra_inner_product_shape_tuning_defaults_from_seeded_euclidean_results.csv`
and is replaced by `run_hpc_cagra_tuning_cuda_inner_product.sh` when the
metric-specific sweep is rerun.

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
| `metric` | `"euclidean"`, `"cosine"`, `"correlation"`, or `"inner_product"`. Euclidean uses FAISS GPU direct `bfKnn` and requests FAISS/cuVS dispatch in cuVS builds; cosine/correlation are normalized in C++; raw inner product uses a MIPS-to-L2 transform and device-side shifted-distance conversion. |
| `tuning` | Tuning label to record. The current GPU-resident route is exact, so this does not change approximation parameters. |
| `target_recall` | Target-recall label to record for API symmetry; exact search has recall 1 by construction. |

`nn_gpu()` is for downstream CUDA consumers that need the KNN output to stay on
the GPU. It returns a `faissR_gpu_knn` object with an owning `handle` plus
non-owning `indices_ptr` and `distances_ptr` external pointers. The device
layout is column-major `n_query x k`, with 1-based int32 indices and float32
distances. Keep `handle` alive for as long as another package uses either
pointer.

The current GPU-resident route is exact search for `method = "auto"`,
`"exact"`, `"flat"`, or `"bruteforce"`. Euclidean and raw inner product use
FAISS GPU direct `bfKnn` when available; raw inner-product similarities are
converted on the CUDA device to faissR's shifted smaller-is-better distance.
Cosine and correlation use the native CUDA GPU-resident exact route.
`target_recall` is recorded for API symmetry but exact search has recall 1 by
construction. With `method = "auto"`, the object also records
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
| `"auto"` | Shape-aware selector for the chosen backend. CPU auto uses exact search for tiny data, grid search for large 2D/3D Euclidean/cosine/correlation self-search, FAISS HNSW for most medium/high-dimensional CPU self-KNN, FAISS IVF for selected large low-dimensional Euclidean rows, native CPU NSG-style refinement for selected larger non-Euclidean self-KNN cases, and native CPU NN-descent for other large self-KNN cases [1-2,5,21]. CUDA auto uses CUDA grid for large 2D/3D Euclidean/cosine/correlation self-search; Euclidean non-grid self-KNN chooses exact Flat/brute force or IVF-Flat from the compiled shape/k/target-recall policy; non-grid cosine/correlation/IP stays on exact FAISS GPU Flat/cuVS brute force when available [1-3,13-16]. |
| `"exact"` | Exact nearest-neighbour search. On CPU, Euclidean exact search resolves to FAISS Flat L2, cosine exact search resolves to normalized FAISS Flat cosine, correlation exact search resolves to centered/normalized FAISS Flat correlation, and raw inner-product exact search resolves to FAISS Flat IP; all four use compiled shape/k/target policies for query batching metadata, with fitted-index reuse where implemented. On CUDA it uses FAISS GPU Flat when FAISS reports GPU support; Euclidean, cosine, correlation, and raw-inner-product exact routes have compiled shape/k/target query-batch/resource policies. CUDA inner-product exact uses FAISS GPU Flat IP and currently seeds batch/resource choices from measured CUDA exact Euclidean rows until the metric-specific sweep replaces them. CUDA exact search can otherwise use direct cuVS brute force when available: Euclidean uses cuVS L2 directly, cosine/correlation use normalized Euclidean search, and inner product uses an exact maximum-inner-product-to-L2 transform [1-3,16]. |
| `"flat"` | FAISS Flat index route for exhaustive L2/IP search. CPU Euclidean, cosine, correlation, and inner-product Flat have compiled metric/shape/k/target-recall policies for query batching and fitted-index reuse, stored in `attr(result, "flat_tuning")`; CUDA Euclidean, cosine, and correlation Flat have compiled FAISS GPU Flat query-batch/resource policies, including the measured CUDA correlation sweep. Recall is exact by construction. Cosine/correlation use normalized Flat IP; CPU degenerate zero-normalized rows use exact CPU scoring to match deterministic exact tie semantics, while explicit CUDA calls error instead of repairing those rows on CPU [1-2,16]. |
| `"bruteforce"` | Brute-force exhaustive search. On CPU, Euclidean brute force uses FAISS Flat L2, cosine brute force uses normalized FAISS Flat cosine, correlation brute force uses centered/normalized FAISS Flat correlation, and raw inner-product brute force uses FAISS Flat IP; all four store compiled metric/shape/k/target-recall policies in `attr(result, "bruteforce_tuning")`. On CUDA, RAPIDS cuVS brute force is preferred when available; cosine/correlation use normalized Euclidean search and inner product uses an exact maximum-inner-product-to-L2 transform around the cuVS L2 kernel. CUDA Euclidean/cosine/correlation bruteforce stores compiled query-batch/resource policies in `attr(result, "bruteforce_tuning")` [1-3,16]. |
| `"grid"` | Native spatial grid search for 2D/3D Euclidean, cosine, and correlation self-KNN. Cosine/correlation use normalized Euclidean grid search. It is intended for low-dimensional spatial or simulated data and errors clearly outside supported dimensions. |
| `"hnsw"` | HNSW graph-search index. CPU uses FAISS HNSW. CPU Euclidean, cosine, correlation, and raw inner-product HNSW use compiled HPC-derived shape/k/target tiers for `M`, `efConstruction`, and `efSearch`; raw inner-product rows that did not meet the requested target report `tuning_benchmark_target_met = FALSE`. CUDA resolves to direct RAPIDS cuVS HNSW from a CAGRA seed graph with a cuVS CPU hierarchy and records `cuda_hnsw_design = "cuvs_hnsw_from_cagra_cpu_hierarchy"` because this is not a pure all-GPU HNSW path. CUDA Euclidean, cosine, correlation, and raw inner-product HNSW use separate compiled shape/k/target tables; cosine is searched as row-normalized float32 Euclidean graph search, correlation as centered row-normalized float32 Euclidean graph search, and raw inner product through a maximum-inner-product-to-L2 transform. HNSW reads `target_recall = 0.9`, `0.95`, or `0.99` to choose its compiled speed/recall tier [5,16,22-23]. |
| `"ivf"` | FAISS IVF-Flat inverted-file index. IVF partitions vectors into coarse lists and probes selected lists; it supports L2/IP plus normalized-IP cosine/correlation, trades exactness for speed/memory, and uses deterministic shape/k/metric-aware probe defaults for very large CPU/GPU searches. CPU Euclidean, cosine, correlation, and raw inner-product auto tuning use separate compiled shape/k/target `nlist`/`nprobe` tiers from CPU12 sweeps. CUDA Euclidean, cosine, and correlation IVF use separate compiled shape/k/target rows from GPU sweeps; raw inner product uses FAISS GPU IVF IP with a validation-pending seed table from the CUDA Euclidean IVF sweep until the dedicated IP sweep replaces it. Cosine is searched as row-normalized float32 IVF, correlation as centered row-normalized float32 IVF, and both are converted back to their public distances. Metadata records whether the requested benchmark target was actually met [1-2,16]. |
| `"ivfpq"` | FAISS IVF with product quantization. IVFPQ supports L2/IP plus normalized-IP cosine/correlation, compresses vectors, and is best treated as a memory-pressure method rather than an accuracy-first default. CPU Euclidean, cosine, correlation, and raw-inner-product auto tuning use separate compiled shape/k/target rows for `nlist`, `nprobe`, `pq_m`, and `pq_nbits`; CUDA Euclidean, correlation, and raw-inner-product IVFPQ use FAISS GPU IVF-PQ shape/k/target rows for the same parameters. CUDA raw inner product is validation-pending and seeded from the measured CUDA Euclidean IVFPQ table until `run_hpc_ivfpq_tuning_cuda_inner_product.sh` replaces it. Metadata records whether the requested benchmark target was actually met, and many IVFPQ correlation/raw-IP rows are explicitly best-available or seeded rather than target-reaching. Direct cuVS IVF-PQ and FAISS GPU IVFPQ use their own deterministic or CUDA-oriented tuning paths [1-3,6,16]. |
| `"ivfpq_fastscan"` | IVFPQ FastScan compressed-code search. CPU uses FAISS `IndexIVFPQFastScan` with 4-bit PQ lookup tables and optional Flat reranking; CPU Euclidean uses FastScan L2, raw inner product uses FastScan IP, cosine is implemented by row L2 normalization followed by FastScan L2 search and distance conversion to `1 - cosine`, and correlation subtracts row means before the same normalized-L2 FastScan route. CPU inner-product auto-tuning is currently seeded from the Euclidean FastScan table because the uploaded IP sweep was blocked by the old Euclidean-only guard; metadata reports `tuning_benchmark_target_met = FALSE` until the corrected sweep is rerun. CUDA uses direct cuVS IVF-PQ with 4-bit compressed codes for Euclidean, cosine, correlation, and raw-inner-product search; cosine row-normalizes to float32, correlation row-centers and row-normalizes to float32, and raw inner product applies the maximum-inner-product-to-L2 extra-dimension transform before cuVS L2 search. Distances are converted back to the public metric. CUDA reuses compatible fitted indexes, dataset device buffers, query device buffers, and cuVS resources through bounded session caches where possible. CUDA 4-bit IVF-PQ requires byte-aligned packed codes, so faissR repairs invalid `pq_dim` requests such as odd dimensions before calling cuVS and records the adjusted values in result metadata. CUDA cosine, correlation, and raw-inner-product FastScan tuning are seeded from the CUDA Euclidean FastScan table until the corrected metric-specific sweeps are rerun, and record `tuning_benchmark_target_met = FALSE`; CPU records `faiss_ivfpq_fastscan` and CUDA records `cuda_cuvs_ivfpq_fastscan` [6,34]. |
| `"vamana"` | DiskANN/Vamana-style robust-pruned candidate graph implemented in faissR. CPU refines candidate rows with native CPU scoring; CUDA refines them with the native CUDA row-candidate kernel. Large high-dimensional CPU inputs use deterministic HNSW seeding before robust pruning; smaller CPU inputs keep the exact seed. Robust pruning protects the first `k` seed neighbours before applying the Vamana rule. CPU Euclidean/cosine/correlation/inner-product and CUDA Euclidean/cosine/correlation/inner-product `tuning = "auto"` use compiled shape/k/target-recall defaults for `r`, `search_l`, and `alpha`; CUDA cosine rows come from measured normalized-float32 Vamana sweeps, while CUDA correlation and raw inner product currently seed the same parameters from those cosine rows. Seeded CUDA Vamana metric rows record `tuning_benchmark_source = "hpc_vamana_cuda_correlation_validation_pending_seeded_from_cosine_20260702_232209"` or `tuning_benchmark_source = "hpc_vamana_cuda_inner_product_validation_pending_seeded_from_cosine_20260702_232209"` and `tuning_benchmark_target_met = FALSE` until the dedicated sweeps are rerun. cuVS Vamana is acknowledged for GPU build/serialization, but current cuVS does not expose KNN search [3,5,24]. |
| `"nsg"` | Navigating Spreading-out Graph style approximate search. CPU uses faissR's native NSG-style self-KNN route for all public metrics so public calls avoid unsafe linked-FAISS graph construction. Large high-dimensional CPU inputs use deterministic HNSW seeding before NSG/MRNG-style pruning; smaller CPU inputs keep the exact seed. CUDA uses faissR's native NSG-style self-KNN route for all public metrics, with normalized cosine/correlation and shifted dot-product inner-product distances. Native NSG protects the first `k` seed neighbours before pruning. CPU Euclidean/cosine/correlation/inner-product and CUDA Euclidean/cosine/correlation/inner-product `tuning = "auto"` use compiled shape/k/target-recall defaults for `r` and `graph_k`; CUDA cosine rows come from measured normalized-float32 NSG sweeps, while CUDA correlation and raw inner product currently seed the same parameters from those cosine rows. Seeded CUDA NSG metric rows record `tuning_benchmark_source = "hpc_nsg_cuda_correlation_validation_pending_seeded_from_cosine_20260702_211910"` or `tuning_benchmark_source = "hpc_nsg_cuda_inner_product_validation_pending_seeded_from_cosine_20260702_211910"` and `tuning_benchmark_target_met = FALSE` until the dedicated sweeps are rerun. Manual overrides read `faissR.cpu_nsg_*` or `faissR.cuda_nsg_*` options [5,16,21,29]. |
| `"nndescent"` | NN-descent style approximate graph construction. CPU uses faissR's native NNDescent route by default for Euclidean/L2, normalized cosine/correlation, and raw inner product; `tuning = "auto"` selects CPU candidate pool, iteration, candidate breadth, and random-projection settings from metric-specific shape/k/target-recall tables. CUDA maps to direct cuVS NN-descent for Euclidean/L2 plus normalized cosine/correlation. CUDA Euclidean has measured shape/k/target defaults; CUDA cosine and correlation use transformed float32 search and seeded policies derived from the measured Euclidean table until the corrected metric-specific sweeps are rerun. Raw inner-product CUDA NN-descent uses faissR's native shifted-dot-product CUDA candidate refinement route because direct cuVS NN-descent does not expose raw IP. FAISS NNDescent is disabled by default because linked FAISS builds can abort during graph construction [3-4,16]. |
| `"cagra"` | CUDA-only graph-search method. faissR can use FAISS GPU CAGRA through the FAISS/cuVS integration or direct RAPIDS cuVS CAGRA. Use `cagra_implementation = "faiss_gpu"` or `"cuvs"` to force the provider for a call, or `options(faissR.cagra_implementation = ...)` to set a session default; `"auto"` applies the deterministic shape-aware provider rule. Direct cuVS CAGRA also exposes `cagra_build_algo` (`"auto"`, `"ivf_pq"`, `"nn_descent"`, or `"iterative_cagra_search"`) as an explicit graph-construction parameter; `cagra_build_algo = "auto"` is shape-aware rather than a silent provider fallback. It supports Euclidean/L2, cosine/correlation through normalized Euclidean graph search, and raw inner product through a maximum-inner-product-to-L2 extra-dimension transform. CUDA Euclidean `tuning = "auto"` uses measured shape/k/target rows from `benchmark_scripts/cuda_cagra_euclidean_shape_tuning_defaults_from_uploaded_results.csv`; CUDA cosine and correlation use validation-pending seeded tables and return `tuning_benchmark_target_met = FALSE` until the corrected metric-specific sweeps are rerun. The correlation seed is `benchmark_scripts/cuda_cagra_correlation_shape_tuning_defaults_from_seeded_euclidean_results.csv` [3,13-16]. |

Use `nn(..., exclude_self = TRUE)` for embedding
workflows where each row should not list itself as its nearest neighbour.

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
| `metric` | `"euclidean"`, `"cosine"`, `"correlation"`, or `"inner_product"`. Legacy aliases such as `"l2"`, `"cor"`, `"pearson"`, and `"ip"` are rejected. Correlation is centered cosine similarity, not raw inner product. Inner-product candidate scoring ranks by larger raw dot product, while returned `distances` are shifted within each query so the best returned dot product has distance `0`. CUDA candidate scoring supports Euclidean directly, cosine/correlation through normalized Euclidean scoring, and raw inner-product scoring through a dedicated CUDA kernel mode with the same shifted-distance convention. |
| `n_threads` | CPU worker threads. |
| `exclude_self` | If `TRUE`, remove each row from its own candidate list. This requires `points = data`. |

This function does not generate candidates; it only reranks candidates supplied
by another method.

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
| `max_iter` | Maximum number of Lloyd iterations, or `"auto"` for a deterministic shape-aware default computed by the compiled C++ tuning helper. |
| `n_init` | Number of random restarts where the selected backend supports it, or `"auto"` for a deterministic shape-aware default computed by the compiled C++ tuning helper. |
| `tol` | Non-negative convergence tolerance where supported, or `"auto"` for a deterministic shape-aware default computed by the compiled C++ tuning helper. |
| `seed` | Random seed for CPU/statistics and FAISS paths. The direct cuVS C API path currently does not expose an explicit seed in the stable params structure, so repeated cuVS runs should be interpreted as backend-controlled initialization. |
| `n_threads` | CPU worker threads for FAISS/statistics paths. |
| `streaming_batch_size` | cuVS host-data streaming batch size. Use `0` to let cuVS choose its default. |
| `init` | Initialization method: `"kmeans++"` or `"random"` where supported. |
| `tuning` | `"auto"` uses deterministic C++ rules based on `nrow(data)`, `ncol(data)`, `centers`, and `n / centers`; small many-cluster jobs can use extra restarts without pilot runs, while large/high-dimensional jobs use cheaper defaults. `centers = 1` uses the exact column-mean solution for every backend request and records `single_cluster_exact_mean`; `centers = nrow(data)` uses the exact singleton assignment for every backend request and records `singleton_exact_identity`. Explicit CUDA requests for these exact cases return the trivial result and do not launch GPU work. `"fixed"`, `"off"`, and `"none"` keep the historical defaults unless explicit parameter values are supplied. |

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
and auditing. The k-means auto parameter rule is computed by
`kmeans_auto_params_cpp()`, while the backend policy and final CUDA/CPU gate are
computed by `kmeans_auto_backend_policy_cpp()` and
`kmeans_auto_select_backend_cpp()`; they record `tuning_source = "cpp"` in
returned metadata. The CUDA auto gate can be adjusted for a benchmarked machine
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
CUDA runs also record `parameters$cuda_provider_selection` as `"faiss_gpu"`,
`"direct_cuvs"`, or `"direct_cuvs_after_faiss_gpu_unavailable_or_failed"`;
`parameters$backend_resolution_note` describes the provider route, and
`parameters$faiss_gpu_error` is present when direct cuVS was used after a FAISS
GPU route was unavailable or failed.

## `knn()`

```r
model <- knn(Xtrain, Ytrain, backend = "auto", method = "auto",
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
| `backend` | Device backend passed to `nn()`: `"auto"`, `"cpu"`, or `"cuda"`. `"auto"` follows `nn()` backend/method/metric resolution, using CUDA only for validated CUDA combinations when CUDA/cuVS runtime support is available, and CPU otherwise. |
| `method` | Nearest-neighbour algorithm selector passed to `nn()`. `"auto"` chooses the most appropriate method for the selected backend. |
| `metric` | Distance metric passed to `nn()`: `"euclidean"`, `"cosine"`, `"correlation"`, or `"inner_product"`. Legacy aliases such as `"l2"`, `"cor"`, `"pearson"`, and `"ip"` are rejected. Correlation is centered cosine similarity, not raw inner product. |
| `tuning` | Tuning policy passed to `nn()`. `"auto"` uses the deterministic default for the resolved method; `"cache"` and `"pilot"` opt into pilot tuning where implemented. CPU Euclidean/cosine/correlation/inner-product IVF, IVFPQ, NN-descent, NSG, and Vamana use compiled shape/k/target tiers where available; CUDA IVF has compiled Euclidean, cosine, correlation, and validation-pending raw-inner-product tiers; CUDA IVFPQ has compiled Euclidean, correlation, and validation-pending raw-inner-product tiers; CUDA CAGRA has a measured Euclidean table plus validation-pending cosine/correlation policies seeded from that table; CUDA NSG has compiled Euclidean/cosine tiers plus validation-pending correlation and raw-inner-product policies seeded from the measured CUDA cosine NSG table; CUDA Vamana has compiled Euclidean/cosine tiers plus validation-pending correlation and raw-inner-product policies seeded from the measured CUDA cosine Vamana table; CUDA NN-descent has a compiled Euclidean table plus cosine and correlation policies seeded from that table until the corrected metric-specific sweeps are rerun; CUDA IVFPQ FastScan has a compiled Euclidean table plus cosine, correlation, and raw-inner-product policies seeded from that table until the corrected metric-specific sweeps are rerun. FAISS GPU IVF pilot/cache tuning is Euclidean-only. |
| `target_recall` | Speed/recall tier passed to `nn()` for fitting and immediate prediction; affects CPU Euclidean/cosine/correlation/inner-product IVF, IVFPQ, NSG, and Vamana, CUDA auto Flat-vs-IVF selection, CUDA IVF probing, CUDA Euclidean/correlation/raw-inner-product IVFPQ probing/PQ tiers, CUDA NSG/Vamana graph tiers, and HNSW tiers. |
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
and HNSW `target_recall` requirements match the fitted model, and prediction
metadata reports `approximation$index_reused = TRUE`. If the model is saved and
reloaded in a later R session, or if prediction settings do not match,
`predict()` rebuilds the same route instead of switching algorithms.

## `predict()`

```r
predict(object, newdata, k = NULL,
        backend = "auto", tuning = "auto", target_recall = NULL,
        cagra_implementation = NULL, cagra_build_algo = NULL,
        vote = "majority", type = "response", ...)
```

| Argument | Description |
| --- | --- |
| `object` | A fitted model returned by `knn(Xtrain, Ytrain, ...)`. |
| `newdata` | Numeric query matrix or optional `float::fl()`/`float32` matrix with the same number of columns as the training matrix. Float32 query data is preserved for direct-adapter NN methods. |
| `k` | Number of neighbours for this prediction call. If `NULL`, uses the model default. |
| `backend` | Device backend for the prediction-time neighbour search: `"auto"`, `"cpu"`, or `"cuda"`. The fitted model's method and metric are reused. |
| `tuning` | Prediction-time tuning policy. `"auto"` uses the deterministic default for the resolved method; `"cache"` and `"pilot"` opt into pilot tuning. |
| `target_recall` | Optional speed/recall tier for prediction; `NULL` reuses the fitted model setting. It affects CPU Euclidean/cosine/correlation/inner-product IVF and IVFPQ probing/PQ tiers, CUDA auto Flat-vs-IVF selection, CUDA IVF probing, CUDA Euclidean/correlation/raw-inner-product IVFPQ probing/PQ tiers, and HNSW tiers when a new NN search is needed. |
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
