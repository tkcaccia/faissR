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

At runtime, availability helpers (`faiss_available()`, `faiss_gpu_available()`,
`cuda_available()`, `cuvs_available()`, and `cugraph_available()`) report
compiled/runtime support.
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
- FAISS CPU Flat, IVF-Flat, IVF-PQ, HNSW, NSG, and NN-Descent when present in
  the linked FAISS build [1-6,16];
- RcppHNSW as an optional CRAN-friendly fallback;
- exact 2D/3D grid routes for low-dimensional Euclidean, cosine, and
  correlation self-KNN;
  VP-tree also supports cosine/correlation through normalized Euclidean search
  when rows are nonzero/nonconstant.

Supported CUDA routes include:

- FAISS GPU Flat, IVF-Flat, IVF-PQ, and CAGRA when FAISS GPU support is built
  [1-2,13-16];
- direct RAPIDS cuVS brute force, CAGRA, IVF-Flat, IVF-PQ, and NN-Descent when
  cuVS is available [3,13-15];
- native CUDA 2D/3D grid search for low-dimensional Euclidean, cosine, and
  correlation self-KNN.

The validated high-performance metric is Euclidean/L2. Cosine and correlation
are exposed for exact CPU, FAISS CPU/GPU Flat, FAISS CPU/GPU IVF-Flat,
FAISS CPU/GPU IVFPQ, FAISS CPU HNSW, direct cuVS IVF/CAGRA/NN-Descent routes,
and RcppHNSW-compatible paths. FAISS IP-capable approximate routes implement cosine by row L2
normalizing the inputs before inner-product search, and correlation by row
centering plus L2 normalization before inner-product search; both routes return
`1 - similarity` distances. All-zero cosine rows and constant correlation rows
are zero-normalized edge cases. faissR treats two zero-normalized rows as
distance `0` and a zero-normalized row versus a nonzero row as distance `1`.
CPU FAISS Flat uses the exact CPU scorer for those rows to preserve
deterministic small-`k` tie handling; explicit CUDA routes remain on CUDA and
apply the normalized-distance repair without relabelling the backend.
Direct cuVS IVF/PQ, CAGRA, and NN-Descent use normalized Euclidean search for
cosine/correlation and keep raw inner product disabled because the direct cuVS
routes used by faissR are L2-based.
Graph-style routes that implement cosine/correlation through normalized
Euclidean search convert returned neighbour distances back to `1 - similarity`
with the stable formula
`normalized_euclidean_squared_over_2_to_1_minus_similarity`. Those results
record `metric_transform` and `attr(result, "distance_transform")`, and
approximate routes also copy the fields into `attr(result, "approximation")`,
so benchmark summaries can distinguish the search space from the reported
distance semantics.
Inner-product search is exposed for exact native CPU
scoring, FAISS Flat IP routes, FAISS IVF-Flat/IVFPQ IP, FAISS HNSW
IP, and the RcppHNSW/hnswlib IP
fallback. Approximate accelerator backends reject unsupported metric/backend
combinations instead of returning neighbours computed under a different metric
label.

### Result Metadata

All KNN routes return a `faissR_nn` object with:

- `indices`: an integer matrix of 1-based R row indices;
- `distances`: a numeric matrix aligned with `indices`;
- `attr(result, "backend")`: the public/resolved backend label returned to the
  user;
- `attr(result, "requested_backend")`: the public backend argument supplied to
  `nn()`/`nn_without_self()`;
- `attr(result, "requested_method")`: the public method argument after alias
  normalization;
- `attr(result, "resolved_backend")`: when relevant, the concrete backend chosen
  behind an alias such as `backend = "cuda"`;
- `attr(result, "tuning")`: the normalized tuning policy used by the public
  wrapper;
- `attr(result, "metric")`: the metric used;
- `attr(result, "exact")`: whether the route is exact by construction;
- `attr(result, "approximation")`: method-specific parameters for approximate
  routes, such as `nlist`, `nprobe`, HNSW `M`, CAGRA graph degree, or tuning
  metadata. Approximate selectors record deterministic no-pilot metadata such as
  `tuning_policy`, `tuning_rule`, and shape flags including high-dimensional,
  large-`n`, small-`k`, large-`k`, and non-Euclidean routing. For IVFPQ/PQ
  compression settings, PQ-specific fields are prefixed with `pq_`.
- `attr(result, "auto_selection")`: for requests involving
  `backend = "auto"` or `method = "auto"`, the static shape/k/metric decision
  record. It stores `policy = "static_shape_k_metric_selector"`, the predicted
  concrete backend, public method class, device class, the reason for the
  selection, explicit backend/method flags, backend/method decision reasons,
  `n`, `p`, query count, `k`, metric, work-size estimate, and `slow_tuning =
  FALSE`. This is a preflight record only; it does not run a pilot benchmark or
  build an index.

This metadata is intentionally simple because the same result object feeds graph
construction, clustering, benchmarking, and supervised prediction.

### Backend And Method Policy

The public KNN API separates device choice from algorithm choice:

- `backend = "auto"` uses a validated CUDA route only when the requested
  method/metric combination is supported and CUDA/cuVS runtime support is
  available, and otherwise resolves to CPU;
- `backend = "cpu"` forces CPU execution;
- `backend = "cuda"` forces CUDA execution and fails clearly if unavailable.

The public `method` argument selects the algorithm. `method = "auto"` is the
shape-aware selector for the chosen device. On CPU, it uses exact CPU for
small work, exact grid search for large 2D/3D Euclidean/cosine/correlation
self-KNN, FAISS IVF for million-row self-KNN where HNSW graph construction is
too memory-heavy, FAISS HNSW for large high-dimensional self-KNN across all
supported CPU metrics when FAISS is available, and FAISS Flat exact search for
larger cosine/correlation/inner-product query or exact workloads before falling
back to RcppHNSW/hnswlib for large non-Euclidean self-search when FAISS is
unavailable. If neither FAISS nor RcppHNSW is available, CPU auto can use
native CPU NSG-style candidate refinement for larger non-Euclidean self-KNN,
or native CPU NN-descent for other large self-KNN cases, instead of exact brute
force [1-2,5,21]. On CUDA, it uses CUDA grid search for large 2D/3D
Euclidean/cosine/correlation self-KNN, exact FAISS GPU Flat or cuVS brute force
for small and medium Euclidean searches, FAISS GPU CAGRA for very large
Euclidean self-KNN when FAISS GPU/cuVS integration is available, and FAISS GPU
Flat inner-product routes for cosine, correlation, and raw inner-product
searches when FAISS GPU Flat is available [13-15]. If CUDA/cuVS is present but
FAISS GPU Flat is not, `backend = "auto"` keeps non-grid non-Euclidean searches
on CPU instead of selecting an unavailable GPU index. The same rule applies
when `backend = "auto"` is combined with an explicit method such as `"flat"` or
`"ivf"`: the selected method/metric must have a runtime-capable CUDA route, or
auto uses the CPU route when that method/metric is supported on CPU.
The public `tuning` argument controls method-specific pilot tuning. The default
`tuning = "auto"` uses the recommended tuning policy for the resolved method;
`"cache"`, `"pilot"`, and `"fixed"` can be selected explicitly, and
`"off"`/`"none"` disables tuning.

Explicit methods map to the selected backend. For example,
`method = "grid", backend = "cpu"` resolves to the CPU grid implementation,
whereas `method = "grid", backend = "cuda"` resolves to the CUDA grid
implementation. Invalid combinations fail before computation; for example,
`method = "cagra", backend = "cpu"` errors because CAGRA is CUDA-only.
For CUDA CAGRA, `options(faissR.cagra_implementation = "auto")` keeps the
default FAISS GPU CAGRA first, direct cuVS fallback rule; `"faiss_gpu"` or
`"cuvs"` forces one provider for benchmark isolation. Runtime preflight and
availability checks respect the forced provider for Euclidean, cosine,
correlation, and inner-product CAGRA routes. Returned approximate NN objects
record `cagra_provider` (`"faiss_gpu"` or `"cuvs"`) and
`cagra_provider_option` in `attr(result, "approximation")`, so benchmark tables
can compare FAISS GPU CAGRA and direct RAPIDS cuVS CAGRA without parsing the
resolved backend string.

Direct cuVS CAGRA uses deterministic no-pilot defaults for `tuning = "auto"`.
If the user explicitly requests `tuning = "cache"` or `tuning = "pilot"`, faissR
runs recall tuning; if that pilot does not meet the configured recall target, the
function stops and recommends FAISS GPU CAGRA or cuVS brute force rather than
silently returning a poor graph-search result.

IVF probe defaults are conservative enough to avoid misleading speed-only
results. IVFPQ is treated as an explicit memory-pressure backend because product
quantization can reduce recall substantially [6].
FAISS HNSW uses no pilot tuning in the user call. Its default parameters are a
static shape/k/metric policy: a speed tier for lower-dimensional Euclidean
`k <= 10`, a balanced small-`k` tier for non-Euclidean metrics, a balanced tier
for general CPU HNSW, and a high-recall tier for large-`k` high-dimensional or
non-Euclidean searches.

### Method Families

The public `method` names are stable user-facing labels. Each algorithm family
has one public method name; implementation labels such as `faiss_hnsw`,
`faiss_ivf`, `faiss_gpu_ivf_flat`, or `cuda_cuvs_cagra` are reserved for
resolved backend metadata and internal benchmark diagnostics. Internally, the
public method names map to different concrete functions depending on `backend`.

| Method | CPU behavior | CUDA behavior | Notes |
| --- | --- | --- | --- |
| `auto` | Shape-aware exact/grid/FAISS IVF/FAISS HNSW selector. | Shape-aware CUDA grid, FAISS GPU Flat/cuVS brute force, cuVS HNSW, FAISS GPU CAGRA selector, or FAISS GPU Flat IP for exact inner-product routes. | Default for general use. |
| `exact` | Native exact CPU route. | FAISS GPU Flat when available; Euclidean can otherwise use cuVS brute force. | Accuracy-first baseline. |
| `flat` | FAISS Flat L2/IP index; cosine and correlation use normalized Flat IP, with exact CPU fallback for zero-normalized rows. | FAISS GPU Flat L2/IP; cosine and correlation use normalized Flat IP while staying on CUDA for explicit CUDA calls. | Exact FAISS route [1-2,16]. |
| `bruteforce` | Native exact CPU route. | Euclidean prefers direct cuVS brute force; non-Euclidean metrics use FAISS GPU Flat when available. | Useful for comparing direct cuVS against FAISS GPU Flat [1-3,16]. |
| `grid` | Native 2D/3D exact spatial grid. | CUDA 2D/3D grid. | Errors outside two or three columns. |
| `hnsw` | FAISS CPU HNSW. | RAPIDS cuVS HNSW from a CUDA CAGRA index for Euclidean/cosine/correlation and transformed raw inner product. | High-recall graph-search route; CUDA raw inner product uses a maximum-inner-product-to-L2 extra-dimension transform [3,5,16,22-23]. |
| `ivf` | FAISS CPU IVF-Flat L2/IP; cosine and correlation use normalized IVF IP. | FAISS GPU IVF-Flat L2/IP; cosine and correlation use normalized IVF IP. | Coarse-list approximate route [1-2,16]. |
| `ivfpq` | FAISS CPU IVF-PQ L2/IP; cosine and correlation use normalized IVFPQ IP. | FAISS GPU IVF-PQ L2/IP; cosine and correlation use normalized IVFPQ IP. | Compressed approximate route [6,16]. |
| `vamana` | Native DiskANN/Vamana-style robust-pruned candidate graph with CPU refinement. | Native DiskANN/Vamana-style robust-pruned candidate graph with CUDA row-candidate refinement. | Distinct pruned directed graph route implemented in faissR; cuVS Vamana currently provides build/serialization rather than KNN search [3,24]. |
| `nsg` | FAISS CPU NSG for Euclidean/L2; native CPU NSG-style self-KNN candidate graph for cosine, correlation, and inner product. | Native CUDA NSG-style self-KNN candidate graph for all public metrics. | Optional graph-search baseline; CPU non-L2 routes avoid unsafe linked-FAISS graph construction by using faissR-owned candidate pruning/refinement. Native CPU/CUDA NSG use backend-specific auto defaults and options (`faissR.cpu_nsg_*`, `faissR.cuda_nsg_*`) [16,21,29]. |
| `nndescent` | Native CPU NNDescent for Euclidean/L2, cosine, correlation, and raw inner product. | Direct cuVS NN-descent for Euclidean/L2, cosine, and correlation; faissR native CUDA candidate refinement for raw inner product. | Approximate KNN graph construction; cosine/correlation use normalized Euclidean search, CPU and native CUDA raw inner product use shifted dot-product distances, and FAISS NNDescent is disabled by default because linked FAISS builds can abort during graph construction [3-4,16]. |
| `cagra` | Unsupported. | FAISS GPU CAGRA preferred, direct cuVS CAGRA fallback; `faissR.cagra_implementation` can force `"faiss_gpu"` or `"cuvs"`. Cosine/correlation use normalized Euclidean graph search; raw inner product uses a MIPS-to-L2 extra-dimension transform. | CUDA-only FAISS/cuVS graph-search method [3,13-16]. |

Unsupported method/backend pairs stop before computation. This makes benchmark
failures interpretable: a row marked unavailable or unsupported means the
requested algorithm/device combination does not exist in the package, not that a
different algorithm was substituted.

### Automatic Tuning

The public `tuning` argument controls method-specific tuning for approximate GPU
routes. Its default, `tuning = "auto"`, means “use the appropriate default
policy for the resolved method.” Current policies are intentionally conservative:

- `auto`: use deterministic no-pilot defaults for the resolved method;
- `cache`: run a pilot tuning step when needed and reuse/store the selected
  parameters;
- `pilot`: run the pilot for this call without persisting the result;
- `fixed`: use fixed defaults but still record tuning metadata;
- `off`/`none`: disable tuning.

When explicitly requested, FAISS GPU IVF tuning tests candidate `nlist`/`nprobe`
settings on a sample and selects the fastest candidate that meets the recall
target when possible. Explicit cuVS CAGRA tuning tests graph/search parameters
and rejects the route if the pilot cannot meet the configured minimum recall.
This opt-in behavior exists because some direct cuVS CAGRA runs were fast but
produced untrustworthy recall on raw high-dimensional data.

## `nn_without_self()`

`nn_without_self()` wraps self-KNN and returns exactly `k` non-self neighbours.
It is used internally by graph construction, clustering, and benchmarks. The
function requests enough neighbours to remove the self-match safely and keeps the
same `faissR_nn` result shape as `nn()`.

## `candidate_knn()`

`candidate_knn()` ranks a supplied candidate set for each query row. It avoids a
full all-pairs search when another algorithm has already proposed likely
neighbours. CPU candidate ranking is exact for all public metrics. CUDA
candidate ranking is used only when compiled and explicitly requested; it scores
Euclidean candidates directly, cosine/correlation candidates after the same
row-normalized Euclidean transform used by graph-search methods, and raw
inner-product candidates through a dedicated CUDA kernel mode. CUDA and CPU
inner-product candidate scoring both return shifted smaller-is-better
distances where the best dot product in each query row has distance `0`. The
output can feed graph construction, prediction, or diagnostic code.

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
target community count, and KNN backend/method/metric/tuning metadata. It also
keeps compact-relevant KNN result attributes such as FAISS/cuVS approximation
parameters, spatial-index metadata, auto-selection metadata, and normalized
metric transform metadata so graph benchmarks can report which tuned KNN route
and distance semantics produced a graph. It does not require `igraph`.
Inner-product graph construction inherits the `nn()` metric contract:
neighbours are ranked by larger raw dot product, while edge weighting receives
shifted smaller-is-better distances.

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
for Louvain and Leiden. The target belongs to `graph_cluster()`, not
`knn_graph()`, so a precomputed graph can be reused with different target
counts or with an explicit `resolution`. If `n_clusters` is supplied and
`method` is omitted, faissR uses Louvain as the target-count clustering method;
passing `n_clusters` to explicit random-walking remains an error. The graph is
built once, then faissR evaluates a bounded deterministic grid of resolution
values centered from the supplied `resolution` and, when graph size is known, a
no-pilot shape heuristic based on `n_clusters / sqrt(n_vertices)`. The
candidate width is shape-aware: small graphs keep the wide `2^-4` to `2^4`
grid around the center, medium graphs use `2^-3` to `2^3`, and large graphs
use `2^-2` to `2^2` so target-count searches do not repeat full Louvain/Leiden
passes more than needed. It keeps the result whose number of communities is
closest to `m`, breaking ties by modularity. The selected resolution, candidate
center, and search table are returned in the result metadata as
`selected_resolution`, `resolution_selection`, and `resolution_search`, with the
requested target stored as `target_n_clusters`.
The selected row is marked in `resolution_search$selected`; `target_gap`
records the final absolute difference from the requested community count, and
`resolution_selection` records the deterministic rule
`closest_n_communities_then_highest_modularity`. The graph-clustering
benchmark flattens these diagnostics into selected-candidate, candidate-count,
minimum-gap, and selected-is-min-gap columns for cycle-level comparison.
The target must be a positive integer and
cannot exceed the graph vertex count; fractional targets and impossible targets
fail before the resolution-search loop.

Native graph clustering does not depend on `igraph`. This keeps the graph API
under faissR's control and avoids a large mandatory graph dependency. The
returned object includes membership, modularity, backend, parameters, graph edge
list, and source acknowledgements. As with the nearest-neighbour and k-means
APIs, `backend` records the implementation that actually ran, while
`parameters$requested_backend` and `parameters$resolved_backend` preserve the
public backend request and the resolved device policy for benchmark auditing.
When `graph_cluster()` builds the graph internally or receives a `faissR_graph`,
it records
`parameters$graph_backend`, `parameters$graph_requested_backend`, and
`parameters$graph_resolved_backend` so benchmark outputs can distinguish the
concrete KNN implementation from the public graph-backend request and resolved
KNN route. Precomputed `nn()` inputs propagate their resolved KNN backend into
graph metadata.
`parameters$n_vertices` and `parameters$n_edges` record the clustered graph
size directly so benchmark summaries do not need to inspect the embedded graph
edge list.
Direct graph builds also preserve compact KNN route metadata in
`parameters$nn_approximation`, `parameters$nn_faiss`, `parameters$nn_cuvs`,
`parameters$nn_spatial_index`, and
`parameters$nn_auto_selection` when those fields are attached to the internal
`nn()` result.

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
`small_many_centers`/`few_points_many_centers`. The stable `rule` label groups
comparable tuning tiers, while `rule_detail` preserves the exact `n`, `p`,
`centers`, `n_per_center`, and work estimate that produced the rule. Small
many-cluster problems can keep 100 Lloyd iterations and use three restarts when
the job is cheap enough; this includes both well-populated clusters and
few-points-per-center jobs where a small multistart budget improves stability
without running pilot searches. Genuinely large or high-dimensional problems
keep cheaper settings. For benchmark-like shapes, MNIST70k with 10 centres uses the
`medium_single_start` rule with 75 iterations and one restart, million-row
low-dimensional data uses `large_fast_convergence` with 50 iterations, one
restart, and `tol = 1e-3`, while a small 50,000 x 10 / 100-cluster job uses
`small_many_centers_multistart` with 100 iterations and three restarts. These
are fixed rules, not pilot benchmark loops. When `centers = 1`,
`fast_kmeans()` uses the exact CPU column mean for `backend = "auto"` and
`"cpu"` and records `single_cluster_exact_mean`, because iterative k-means
cannot improve that solution on any backend. When `centers = nrow(data)`,
`fast_kmeans()` uses the exact singleton assignment for `backend = "auto"` and
`"cpu"` and records `singleton_exact_identity`, because every observation is
already its own optimal cluster. The
`resolved_from` field records whether `max_iter`, `n_init`, and `tol` were
selected by auto/default rules or supplied explicitly.
`result$parameters$tuning$effective` records the final values used after
explicit overrides and `"auto"` defaults have been resolved, so benchmark code
can summarize the effective k-means run without comparing multiple parameter
fields. The flat aliases `effective_max_iter`, `effective_n_init`, and
`effective_tol` expose the same values for simple CSV summaries.

The public backend policy follows the KNN device contract but adds a
k-means-specific shape gate: `backend = "auto"` uses CUDA only when CUDA plus
FAISS GPU k-means or direct cuVS k-means is compiled and available and the
estimated work is large enough to justify GPU launch and host/device copy
overhead. Small jobs resolve to CPU even on CUDA-capable machines;
`backend = "cpu"` forces the CPU route; `backend = "cuda"` requires an
accelerated route and errors if unavailable. This makes k-means behavior
consistent with `nn()`, `knn_graph()`, and `graph_cluster()`.
The gate is deterministic and recorded in
`result$parameters$tuning$backend_policy`, with a `reason` such as
`"small_cpu_preferred"`, `"few_points_per_center_cpu_preferred"`,
`"work_at_least_1e8"`, `"input_at_least_256MiB"`,
`"large_high_dimensional_input"`, `"single_cluster_exact_mean"`, or
`"singleton_exact_identity"` so benchmark summaries can audit why
`backend = "auto"` selected CPU or CUDA without running extra tuning jobs.
Benchmark-derived threshold refinements can be applied without changing package
code by setting
`options(faissR.kmeans_cuda_work_threshold = ...)`,
`options(faissR.kmeans_cuda_nbytes_threshold = ...)`,
`options(faissR.kmeans_cuda_large_n_threshold = ...)`, or
`options(faissR.kmeans_cuda_large_p_threshold = ...)`, or
`options(faissR.kmeans_cuda_min_n_per_center = ...)`; invalid option values
fall back to the documented defaults. `result$parameters$tuning$selection` is the compact
no-pilot audit record: it stores the requested and predicted backends,
runtime capability flags, shape/work estimates, effective `max_iter`, `n_init`,
and `tol`, `explicit_backend`, `backend_decision`, and `slow_tuning = FALSE`.
The `backend_decision` field is the shape-policy reason for auto requests and
`explicit_cpu`/`explicit_cuda` for explicit device requests, so benchmark
summaries do not confuse a forced backend with an automatic choice.
The returned object also records `hit_max_iter` and `converged`, derived from
the effective iteration cap and the backend-reported iteration count. These
flags are conservative convergence diagnostics for benchmark tuning: they do
not run extra checks, but they make repeated under-iteration visible in cycle
summaries.
`seed` controls CPU/statistics and FAISS k-means paths. The direct cuVS C API
path currently does not expose an explicit seed in the stable params structure,
so repeated direct-cuVS runs use backend-controlled initialization and benchmark
cycle summaries should rely on the observed ARI/withinss stability columns.

## `knn()` And `predict()`

`knn()` is the high-level supervised kNN API. It uses one public model/predict
interface for CPU, FAISS, CUDA, and cuVS-backed neighbour search.

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
returns class probabilities for classification. Prediction outputs carry
`attr(result, "faissR_nn")` metadata from the underlying `nn()` call, including
requested backend/method/tuning, resolved backend, metric, `k`, and whether the
route was exact.

The supervised API intentionally reuses `nn()` rather than creating independent
FAISS/cuVS model classes. That keeps method selection, backend validation,
tuning, and result semantics identical between unsupervised KNN and supervised
kNN prediction.

## Availability Helpers

- `faiss_available()` reports whether FAISS support was compiled and linked.
- `faiss_gpu_available()` reports whether the linked FAISS build reports GPU
  support.
- `cuda_available()` reports CUDA build/runtime availability.
- `cuvs_available()` reports direct RAPIDS cuVS availability.
- `cugraph_available()` reports RAPIDS libcugraph graph-clustering support.
- `backend_info()` returns a data frame summarizing compiled/runtime backends,
  public call hints, non-public implementation route labels, devices, and
  notes.

These helpers are informational; explicit backend calls still validate
availability at execution time.

## Memory And Data Representation

R stores ordinary numeric matrices as double precision. FAISS and cuVS commonly
operate on float data internally. Most accelerated routes therefore need at
least one conversion/copy from R's column-major double matrix to backend-friendly
buffers. For moderate datasets this overhead is small relative to KNN search;
for very large datasets it can dominate memory use.

The core FAISS KNN implementation is being moved toward float-pointer entry
points. The first public slice accepts optional `float::fl()`/`float32` matrices
in `nn()` and `nn_without_self()` for the CPU FAISS Flat route across the four
public metrics: Euclidean, cosine, correlation, and inner product. Cosine and
correlation are normalized in the float32 row-major buffer before FAISS
`IndexFlatIP` search, with the same zero-row semantics as the double-precision
routes. Float objects are read from their float32 payload and copied only once
into FAISS's row-major `float*` layout, avoiding the previous
float32-to-R-double-to-float32 path. Ordinary R double matrices still work and
are converted once to float32 internally. When an ordinary R double input uses
a CPU FAISS Flat-style request with `output = "float"`, it also enters this
float-pointer route so FAISS does not produce an intermediate R double distance
matrix.

The float32 adapter records the route it used in the returned KNN object.
`input_layout` distinguishes ordinary R double conversion, float32 payload
transpose, mixed reference/query adapters, and the direct row-compatible
float32 payload route used for one-row or one-column `float::fl()` matrices.
`input_owns_data` records whether FAISS consumed an owned adapter buffer. For
cosine and correlation, direct float32 payloads still make one owned copy before
normalization because those transforms are applied in-place before FAISS search.

Distance output remains an ordinary R numeric matrix by default. Calling
`nn(..., output = "float")` or `nn_without_self(..., output = "float")` stores
`distances` as a `float::fl()`/`float32` object and records both
`distance_type = "float32"` and `attr(result, "distance_type") = "float32"`.
On the float32 FAISS Flat route, faissR constructs the returned float32
distance payload directly from FAISS's float results, so the common Euclidean,
inner-product, cosine, and correlation cases avoid an intermediate R double
distance matrix. The only fallback is the deterministic zero-row correction for
cosine/correlation inputs with all-zero normalized rows, where faissR first
repairs and sorts the double distance matrix before converting it to float32.
KNN results also expose stable list fields for downstream packages:
`index_base`, `metric`, `backend_used`, and, on the float32 route,
`input_layout`/`input_owns_data`. The `float` package is in `Suggests`; faissR
does not require it unless a user supplies a float32 object or requests float32
distance output.

faissR also registers a C-callable entry point named
`faissR_nn_float32_call` during package initialization with
`R_RegisterCCallable()`. Downstream packages can retrieve it with
`R_GetCCallable("faissR", "faissR_nn_float32_call")` and call the CPU FAISS
Flat float32 route without going through the R wrapper layer. The callable
accepts `backend = "auto"` or CPU/FAISS Flat aliases, including
`"cpu_faiss_flat"`, `"faiss_flat"`, and `"faiss_flat_l2"`; `"auto"` currently
resolves to the CPU FAISS Flat float32 route. It accepts either a
`float::fl()`/`float32`
object or an ordinary R double matrix; both are adapted once into the row-major
`float*` buffers consumed by FAISS.

For callers that need float32 distances, faissR also registers
`faissR_nn_float32_call_output`. It has the same first six arguments plus a
seventh string argument, `distances`, with values `"double"` or `"float"`.
The original six-argument callable is kept as the stable double-output ABI.
Both callables return a stable KNN list with `indices`, `distances`,
`index_base = 1L`, `distance_type`, `metric`, and `backend_used`. When
`distances = "float"`, the optional `float` package is required at runtime and
the returned `distances` component is a `float::fl()`/`float32` matrix.

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
implementation step is to avoid loading the dataset as a double data frame.
Using a matrix, `float::fl()` object, or on-disk float32 representation removes
one large conversion copy and makes FAISS/cuVS index construction more
practical.

## Licensing And Acknowledgement

faissR is released under the MIT license. The implementation is inspired by and
links against external work including FAISS, FAISS GPU/cuVS integration, RAPIDS
cuVS, RAPIDS libcugraph, HNSW, NN-Descent, IVF, product quantization, k-means,
Louvain, Leiden, random-walk clustering, NSG, DiskANN/Vamana, and related ANN
designs such as VP trees, GGNN, SONG, BANG, and PilotANN [1-29]. See
[References](references.md) for papers, software projects, and acknowledgements.
External libraries remain separate system dependencies with their own licenses.
