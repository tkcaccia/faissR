# Backend Capabilities

[Home](../README.md) |
[Installation](installation.md) |
[Implementation](implementation.md) |
[Examples](examples.md) |
[Benchmarks](benchmarks.md) |
[API](usage-api.md) |
[NN Methods](nn-methods.md) |
**Backends** |
[References](references.md)

`faissR` separates the public device selector from the algorithm selector:

- `backend = "auto"` uses CUDA only for validated CUDA method/metric
  combinations when CUDA/cuVS runtime support is available, otherwise CPU.
- `backend = "cpu"` forces CPU execution.
- `backend = "cuda"` forces CUDA execution and errors if no compatible CUDA
  backend is available.
- `method` selects one canonical lowercase public algorithm family, for example
  `"auto"`, `"flat"`, `"hnsw"`, `"ivf"`, `"cagra"`, or `"grid"`.
  Resolved implementation labels such as `faiss_hnsw` or `cuda_cuvs_cagra`
  are backend metadata, not additional public method names.

FAISS is the required compiled vector-search dependency. CUDA, FAISS GPU,
RAPIDS cuVS, and RAPIDS libcugraph are optional compiled/runtime capabilities
[1-3,12-16]. The package does not call Python and does not silently replace an
explicit CUDA request with CPU work.

## Public Backend Policy

| Public backend | Meaning | Failure behavior |
| --- | --- | --- |
| `"auto"` | Prefer CUDA/cuVS for validated CUDA method/metric combinations when CUDA/cuVS runtime support is available; otherwise use CPU. With an explicit method, the chosen method/metric must have a runtime-capable CUDA route before auto selects CUDA. | Falls back to CPU only because the user requested automatic device selection. |
| `"cpu"` | Use CPU/native/FAISS CPU routes. | Errors for CUDA-only methods such as `method = "cagra"`. |
| `"cuda"` | Use CUDA/FAISS GPU/cuVS routes. | Errors if CUDA/cuVS support is unavailable or if the selected method is CPU-only. |

For explicit public methods under `backend = "auto"`, the selector checks the
method/metric CUDA route before choosing a device. For example,
`method = "hnsw"` selects the cuVS HNSW route when RAPIDS cuVS HNSW is
available. Euclidean uses the cuVS L2 route directly, cosine/correlation use
normalized Euclidean graph search, and raw inner product uses the same
maximum-inner-product-to-L2 transform used by faissR's other cuVS L2 graph
routes.

The public request is stored in `attr(result, "requested_backend")` and
`attr(result, "requested_method")`; the normalized tuning policy is stored in
`attr(result, "tuning")`. The backend that ran is stored in
`attr(result, "backend")`. Some routes also store
`attr(result, "resolved_backend")` and an `attr(result, "approximation")` list
with method-specific parameters.

## Nearest-Neighbour Method Mapping

| `method` | CPU route | CUDA route | Main use |
| --- | --- | --- | --- |
| `"auto"` | Shape-aware exact/grid/FAISS IVF/FAISS HNSW selector. | Shape-aware CUDA grid for Euclidean/cosine/correlation 2D/3D self-KNN; FAISS GPU Flat/cuVS brute force, cuVS HNSW, or FAISS GPU CAGRA where available; raw inner-product auto uses FAISS GPU Flat IP when available or faissR native CUDA candidate refinement for large self-KNN. Explicit CUDA exact/brute-force calls can use transformed cuVS brute force. | Default general-purpose choice. |
| `"exact"` | Native exact CPU KNN. | FAISS GPU Flat if available; otherwise direct cuVS brute force for exact metric-aware search when cuVS is available. | Exact/high-recall baseline [1-3,16]. |
| `"flat"` | FAISS Flat L2/IP; cosine and correlation use normalized Flat IP, with exact CPU fallback for zero-normalized rows. | FAISS GPU Flat L2/IP; cosine and correlation use normalized Flat IP while explicit CUDA calls remain on CUDA. | Exact FAISS exhaustive search [1-2,16]. |
| `"bruteforce"` | Native exact CPU KNN. | Direct RAPIDS cuVS brute force when available; cosine/correlation use normalized Euclidean search and inner product uses a maximum-inner-product-to-L2 transform around the cuVS L2 kernel. | Exhaustive route, useful for FAISS/cuVS comparisons [1-3,16]. |
| `"grid"` | Native exact 2D/3D grid for Euclidean, cosine, and correlation. | Native CUDA 2D/3D grid for Euclidean, cosine, and correlation. | Low-dimensional spatial or simulated data; cosine/correlation use normalized Euclidean grid search; explicit grid requests error outside two or three columns. |
| `"hnsw"` | FAISS CPU HNSW for all four public metrics when FAISS is available; RcppHNSW/hnswlib fallback otherwise. | RAPIDS cuVS HNSW for Euclidean/L2, normalized cosine/correlation, and transformed raw inner product. | High-recall graph search. CUDA HNSW is built from a CUDA CAGRA index and searched through the cuVS HNSW wrapper; raw inner product uses a maximum-inner-product-to-L2 extra-dimension transform and returns shifted inner-product distances [3,5,16,22-23]. |
| `"ivf"` | FAISS CPU IVF-Flat L2/IP; cosine and correlation use normalized IVF IP with metric-aware default probing. | FAISS GPU IVF-Flat L2/IP; cosine and correlation use normalized IVF IP with metric-aware deterministic defaults. | Large approximate search with coarse-list probing [1-2,16]. |
| `"ivfpq"` | FAISS CPU IVF-PQ L2/IP; cosine and correlation use normalized IVFPQ IP with metric-aware IVF probing. | FAISS GPU IVF-PQ L2/IP; cosine and correlation use normalized IVFPQ IP with metric-aware IVF probing. | Compressed-memory approximate search; CPU IVFPQ requires at least 624 training rows and auto-selects 4-bit PQ below 9,984 rows [6,16]. |
| `"vamana"` | Native DiskANN/Vamana-style robust-pruned candidate graph with CPU candidate refinement. | Native DiskANN/Vamana-style robust-pruned candidate graph with CUDA row-candidate refinement. | Distinct pruned directed graph route inspired by DiskANN/Vamana; large high-dimensional CPU inputs use deterministic HNSW seed neighbours before robust pruning, while smaller CPU inputs keep exact seed neighbours. Robust pruning protects the first `k` seed neighbours before applying the Vamana rule; cuVS Vamana is acknowledged for GPU build/serialization, but faissR performs KNN candidate refinement because cuVS Vamana search is not exposed yet [3,5,24]. |
| `"nsg"` | Native CPU NSG-style candidate graph for Euclidean, cosine, correlation, and inner product self-KNN. | Native CUDA NSG-style candidate graph for Euclidean, cosine, correlation, and inner product self-KNN. | Optional graph-search baseline. Public CPU NSG avoids unsafe linked-FAISS graph construction and uses faissR-owned NSG-style candidate refinement for all metrics; large high-dimensional CPU inputs use deterministic HNSW seed neighbours before NSG/MRNG-style pruning, while smaller CPU inputs and CUDA keep exact seed neighbours. The first `k` seed neighbours are protected before pruning; cosine/correlation use normalized Euclidean search and inner product uses shifted dot-product distances [5,16,21,29]. |
| `"nndescent"` | Native CPU NNDescent for Euclidean/L2, cosine, correlation, and raw inner product. | Direct RAPIDS cuVS NN-descent for Euclidean/L2, cosine, and correlation; faissR native CUDA candidate refinement for raw inner product. | Approximate KNN graph construction; cosine/correlation use normalized Euclidean search, CPU and native CUDA raw inner product use shifted dot-product distances, and FAISS NNDescent is disabled by default because linked FAISS builds can abort during graph construction [3-4,16]. |
| `"cagra"` | Unsupported. | FAISS GPU CAGRA or direct RAPIDS cuVS CAGRA. `options(faissR.cagra_implementation = "faiss_gpu")` or `"cuvs"` forces one provider; `"auto"` uses a deterministic shape rule that selects direct cuVS CAGRA for compact high-dimensional self-KNN and otherwise keeps FAISS GPU CAGRA as the default when both providers are available. Direct cuVS CAGRA additionally exposes `cagra_build_algo` / `options(faissR.cuvs_cagra_build_algo = ...)` for the cuVS graph builder (`"auto"`, `"ivf_pq"`, `"nn_descent"`, or `"iterative_cagra_search"`); `cagra_build_algo = "auto"` is faissR's shape-aware rule and uses iterative CAGRA construction for compact high-dimensional self-KNN cases. Availability checks respect the forced provider for supported metrics. Approximation metadata records `cagra_provider` (`"faiss_gpu"` or `"cuvs"`), `cagra_provider_option`, and direct-cuVS `cagra_build_algo`. Cosine/correlation use normalized Euclidean graph search; raw inner product uses a maximum-inner-product-to-L2 extra-dimension transform and returns shifted inner-product distances. | CUDA graph-search method via FAISS/cuVS CAGRA [3,13-16]. |

Unsupported combinations fail before computation. For example,
`nn(x, backend = "cpu", method = "cagra")` errors because CAGRA is CUDA-only,
and `nn(x, backend = "cuda", method = "grid", metric = "inner_product")`
errors because the grid route is geometric Euclidean/cosine/correlation search.

## Compiled Backend Families

| Backend family | CPU | CUDA | Notes |
| --- | --- | --- | --- |
| Native faissR dense exact | yes | no | CRAN-friendly exact CPU baseline. |
| Native faissR grid | yes | optional CUDA | 2D/3D Euclidean, cosine, and correlation self-KNN only. |
| FAISS Flat | yes | yes, if FAISS GPU is built | Exact L2 search [1-2,16]. |
| FAISS IVF-Flat | yes | yes, if FAISS GPU is built | Inverted-file approximate L2/IP search; cosine/correlation use normalized IP [1-2,16]. |
| FAISS IVF-PQ | yes | yes, if FAISS GPU is built | Product-quantized approximate L2/IP search; cosine/correlation use normalized IP [6,16]. |
| FAISS HNSW | yes, if exposed by FAISS | no | Approximate CPU graph-search index with L2/IP and normalized-IP metric transforms [5,16]. |
| FAISS NSG | yes, if exposed by FAISS | no | Optional internal CPU graph-search index for Euclidean/L2 only; public CPU NSG requests use faissR's native NSG-style route instead [16,21,29]. |
| Native Vamana | yes | optional CUDA refinement | DiskANN/Vamana-style robust-pruned candidate graph with candidate refinement; large high-dimensional CPU inputs use deterministic HNSW seeding before robust pruning, and CUDA route refines rows with the native CUDA candidate kernel [5,24]. |
| Native CUDA NSG | no | yes, if native CUDA is built | NSG-style self-KNN candidate graph with CUDA candidate refinement for all public metrics; cosine/correlation use row transforms and raw inner product uses the CUDA row-candidate kernel [21]. |
| FAISS NNDescent | experimental opt-in | no | Disabled by default because linked FAISS builds can abort during graph construction; public CPU `method = "nndescent"` uses the native implementation [4,16]. |
| FAISS GPU CAGRA/cuVS integration | no | yes, if FAISS GPU/cuVS integration is built | Uses FAISS GPU indexes backed by NVIDIA cuVS where available; cosine/correlation use normalized Euclidean search and raw inner product uses a maximum-inner-product-to-L2 transform [13-15]. |
| RAPIDS cuVS brute force | no | yes, if cuVS is built | Exact direct cuVS route for Euclidean/L2 plus transformed cosine, correlation, and raw inner-product exhaustive search [1-3,16]. |
| RAPIDS cuVS CAGRA | no | yes, if cuVS is built | Direct CUDA graph-search route with deterministic no-pilot defaults; explicit `tuning = "cache"` or `"pilot"` can run recall-guarded pilot tuning. Cosine/correlation use normalized Euclidean search and raw inner product uses a maximum-inner-product-to-L2 transform [3]. |
| RAPIDS cuVS IVF/PQ | no | yes, if cuVS is built | Direct cuVS approximate routes for Euclidean/L2 plus transformed cosine, correlation, and raw inner product. Cosine/correlation use normalized Euclidean search; raw inner product uses the maximum-inner-product-to-L2 extra-dimension transform before the cuVS L2 index is built. Direct cuVS IVF-PQ requests 4-bit PQ below 9,984 training rows unless the user sets cuVS PQ bits. Public CUDA `method = "ivf"`/`"ivfpq"` still resolves to FAISS GPU IVF/IVFPQ; direct cuVS IVF/PQ appears in resolved backend metadata and benchmark diagnostics [3,6,16]. |
| RAPIDS cuVS NN-descent | no | yes, if cuVS is built | CUDA NN-descent route for Euclidean/L2 plus normalized cosine/correlation; public CUDA raw inner-product NN-descent uses faissR's native CUDA candidate-refinement route because direct cuVS NN-descent does not expose raw IP [3-4]. |
| RAPIDS cuVS HNSW | no | yes, if cuVS is built with HNSW headers | CUDA-built CAGRA index converted to a base-layer cuVS HNSW index (`hnsw_hierarchy = "none"`) and searched through the cuVS HNSW wrapper; supports Euclidean/L2, normalized cosine/correlation, and transformed raw inner product in faissR. The internal CAGRA build uses the same shape-aware `cagra_build_algo = "auto"` rule as direct cuVS CAGRA, and compact high-dimensional self-KNN records `iterative_cagra_search` in approximation metadata [3,5,22-23]. |

## Graph, Clustering, And Model Functions

| Function | CPU | CUDA | Notes |
| --- | --- | --- | --- |
| `candidate_knn()` | yes | optional CUDA candidate ranking where compiled | Exact ranking inside supplied candidates; CUDA supports Euclidean, normalized cosine/correlation, and raw inner product. |
| `knn_graph()` | yes | uses CUDA KNN if generated/supplied KNN uses CUDA | Returns a native `faissR_graph` edge list without requiring `igraph`. |
| `graph_cluster()` | native random-walking, Louvain, Leiden | Louvain/Leiden with RAPIDS libcugraph when built | CUDA random-walking is not enabled yet. `backend = "auto"` is resolved by compiled `graph_cluster_auto_backend_cpp()` and records `parameters$backend_selection$tuning_source = "cpp"` [9-12,17-19]. |
| `fast_kmeans()` | native/FAISS CPU k-means | FAISS GPU or direct cuVS k-means where available | Uses `"auto"`, `"cpu"`, and `"cuda"` backend policy; auto selects CUDA only for CUDA-capable builds and sufficiently large shape/work estimates [7-8]. |
| `knn()` / `predict()` | yes | yes, through `nn()` | Supervised classifier/regressor API reuses `nn()` backend and method resolution. |

## Availability Helpers

Use these helpers to inspect the build/runtime state:

```r
backend_info()
nn_capabilities()
faiss_available()
faiss_gpu_available()
cuda_available()
cuvs_available()
cugraph_available()
```

`backend_info()` returns a data frame with compiled/runtime availability,
public call hints, public backend names, compact public method/metric summaries,
non-public implementation route labels, device/runtime hints, and notes. The
`supported_methods` and `supported_metrics` columns are summaries; use
`nn_capabilities()` for the full method/backend/metric matrix. The
`resolved_route` column is diagnostic metadata; values such as `faiss_hnsw` or
`cuda_cuvs_cagra` are implementation labels, not accepted public `method`
values. For public CAGRA calls, keep `method = "cagra"` and select the CUDA
provider with the per-call `cagra_implementation = "auto"`, `"faiss_gpu"`, or
`"cuvs"` argument when a benchmark must force FAISS GPU CAGRA or direct cuVS
CAGRA. The session option `options(faissR.cagra_implementation = ...)` remains
available as a default. The forced provider is respected for Euclidean, cosine,
correlation, and inner-product preflight checks. Returned approximate NN objects
record the resolved provider in `attr(result, "approximation")` as
`cagra_provider` and the normalized selector as `cagra_provider_option`.
The boolean helpers return a single
`TRUE`/`FALSE` value. They are useful for diagnostics and examples, but
explicit backend calls still validate availability at execution time.
`nn_capabilities()` returns a data frame with one row per public
method/backend/metric combination, including `backend = "auto"`, `"cpu"`, and
`"cuda"`, and marks unsupported combinations before a benchmark tries to run
them.

For benchmark launchers, `nn_capabilities(runtime = TRUE)` adds the
implementation route that the public API would request on the current machine,
whether that route is available in the installed build, a stable
`runtime_reason`, and human-readable `runtime_notes`. This separates
method/metric validity from local runtime availability, for example a valid
FAISS GPU Flat row on a CPU-only installation. Reason labels include
`available`, `unsupported_combination`, `missing_faiss`, `missing_faiss_gpu`,
`missing_cuda`, `missing_cuda_route`, and `missing_cuvs`, so benchmark scripts
do not need to parse prose.

For `metric = "inner_product"`, faissR ranks neighbours by larger raw dot
product but reports shifted smaller-is-better distances, with the best returned
dot product in each query row at distance `0`.

The capability table is design-level. Runtime auto-selection can still choose
CPU when the public CUDA design route needs a missing optional component. For
example, CUDA cosine and correlation auto routes can use CUDA grid for large
2D/3D self-KNN. Non-grid CUDA cosine and correlation auto routes use FAISS GPU
Flat for exact small/query workloads when available, and can select FAISS
GPU/direct cuVS graph routes for large self-KNN. CUDA inner-product auto uses
FAISS GPU Flat IP when available or faissR's native CUDA candidate-refinement
route when native CUDA kernels are available; explicit CUDA CAGRA and HNSW can
use the MIPS-to-L2 graph-search transform. On a cuVS-only runtime, CUDA auto
non-Euclidean capability rows are reported as shape-dependent rather than as a
promise that every metric/method has a CUDA route. The same check is applied to explicit
methods such as `"flat"`, `"ivf"`, and `"ivfpq"` under `backend = "auto"`.
FAISS CPU and FAISS GPU availability are checked separately at execution time:
explicit FAISS GPU Flat, IVF, IVFPQ, and CAGRA routes require a FAISS build
that reports GPU support, not only a CPU FAISS installation.

## Tuning And Approximation Metadata

Approximate GPU routes use deterministic no-pilot defaults for
`tuning = "auto"`. FAISS IVF records fixed shape/k/metric-aware
`nlist`/`nprobe` metadata, and cuVS CAGRA records fixed graph/search metadata.
The route selector and deterministic parameter selectors are compiled C++
policies: `nn_auto_select_backend_cpp()` chooses the backend route, and
`nn_tune_*_cpp()` helpers choose HNSW/IVF/PQ/CAGRA/NSG/Vamana/NN-descent
parameters. R wrappers read user options and pass values into C++, but do not
maintain a separate tuning implementation. Approximate
results record relevant parameters in `attr(result, "approximation")`.
Approximate selectors use deterministic no-pilot parameter rules unless the user
explicitly enables a pilot/cache policy for routes such as FAISS GPU IVF or cuVS
CAGRA. IVF, IVFPQ/PQ, NSG, NN-descent, CAGRA, and HNSW record
`tuning_policy`, `tuning_rule`, and relevant shape flags in approximation
metadata; IVF also records `tuning_metric`/`tuning_metric_aware`, and PQ
compression selectors use `pq_tuning_*` field names. Deterministic selectors
record `tuning_source = "cpp"`.

Exact routes mark `attr(result, "exact") = TRUE`. Approximate routes mark
`exact = FALSE`, and benchmark code should report recall or explicitly mark
quality as exact-assumed only when the route is exact by construction.

## Installation Implications

- A CPU-only installation still requires FAISS.
- CUDA/cuVS/cuGraph support is optional and enabled only when matching headers
  and libraries are available.
- Explicit CUDA requests fail clearly on CPU-only builds.
- The package does not vendor FAISS, cuVS, cuGraph, or CUDA. See
  [Installation](installation.md) for build variables and
  [References](references.md) for software acknowledgements.
