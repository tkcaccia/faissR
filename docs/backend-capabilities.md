# Backend Capabilities

For cosine and correlation, zero-normalized edge cases have explicit semantics:
two zero-normalized rows have distance `0`, and a zero-normalized row versus a
nonzero row has distance `1`. These are software conventions for otherwise
undefined cases. Explicit CUDA routes fail clearly when preserving them would
require hidden CPU repair. Unsupported method/backend/metric combinations fail
without changing the requested metric, method, or device; use
`nn_capabilities(runtime = TRUE)` for route preflight checks. Use
`nn_metric_preflight()` for the data-dependent check: it reports one-based
non-finite, zero-vector, and constant-row indices together with the requested
backend's action. All backends reject non-finite values.

CUDA exact and brute-force requests use direct cuVS brute force when available,
while FAISS GPU Flat remains the provider-backed alternative.

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

- `backend = "cpu"` forces CPU execution.
- `backend = "cuda"` forces CUDA execution and errors if no compatible CUDA
  backend is available.
- Omitting `backend` follows `options(faissR.backend)`, then `FAISSR_BACKEND`,
  and finally uses CPU. Device selection is never automatic.
- `method` selects one canonical lowercase public algorithm family, for example
  `"auto"`, `"flat"`, `"hnsw"`, `"ivf"`, `"ivfpq_fastscan"`, `"cagra"`, or `"grid"`.
  Resolved implementation labels such as `faiss_hnsw` or `cuda_cuvs_cagra`
  are backend metadata, not public `method` values.

For package-owned candidate-graph routes, the preferred aliases are
`"nsg_style"`, `"vamana_style"`, and `"nndescent_style"`. The historical
shorter names remain accepted. The suffix is a scope marker, not an algorithm
name: package-owned routes are distinct algorithms derived from selected ideas
in the named canonical algorithms. Results make the qualification explicit through
`preferred_public_method`, `implementation_label`, `implementation_scope`, and
`canonical_reimplementation = FALSE`; direct cuVS NN-descent is labelled as an
external-provider implementation instead.

Package-owned `*_style` implementations are experimental. Their result
metadata sets `implementation_status = "experimental"` and
`experimental = TRUE`; direct cuVS NN-descent is marked as an external-provider
implementation instead. Experimental routes remain available for explicit
evaluation but are excluded from the publication's principal comparative
performance claims.

FAISS is the required compiled vector-search dependency. CUDA, FAISS GPU, and
RAPIDS cuVS are optional for CPU-only builds [1-3,13-16].
For a NVIDIA GPU build, set `FAISSR_REQUIRE_CUDA=1` and, as needed,
`FAISSR_REQUIRE_CUVS=1`; then missing GPU libraries or an unusable CUDA device
are fatal at configure time. An intentional build-only host can set
`FAISSR_SKIP_CUDA_RUNTIME_CHECK=1`, but that does not validate CUDA execution.
With a visible GPU, configure automatically compiles native code and PTX for
the detected compute capabilities. The package does not call Python and does
not silently replace an explicit CUDA request with CPU work.
For Bioconductor, faissR declares the `GPU` biocView and opts into optional GPU
builders with `.BBSoptions` set to `GPU_reliance: optional`; this advertises GPU
capability without making NVIDIA libraries required for CPU-only builders.

## Public Backend Policy

| Public backend | Meaning | Failure behavior |
| --- | --- | --- |
| `"cpu"` | Use CPU/native/FAISS CPU routes. | Errors for CUDA-only methods such as `method = "cagra"`. |
| `"cuda"` | Use CUDA/FAISS GPU/cuVS routes. | Errors if CUDA/cuVS support is unavailable or if the selected method is CPU-only. |

For example, `method = "hnsw"` is available on CPU and CUDA when the
corresponding libraries are present. CUDA HNSW resolves to RAPIDS cuVS HNSW:
faissR builds a CUDA CAGRA
seed graph, converts it with `cuvsHnswFromCagraWithDataset`, and records
`cuda_hnsw_design = "cuvs_hnsw_from_cagra_cpu_hierarchy"` because this is not a
pure all-GPU HNSW search path. Use CUDA `method = "cagra"` for all-GPU graph
search.

The public request is stored in `attr(result, "requested_backend")` and
`attr(result, "requested_method")`; the normalized tuning policy is stored in
`attr(result, "tuning")`. The backend that ran is stored in
`attr(result, "backend")`. Some routes also store
`attr(result, "resolved_backend")` and an `attr(result, "approximation")` list
with method-specific parameters.

## Nearest-Neighbour Method Mapping

| `method` | CPU route | CUDA route | Main use |
| --- | --- | --- | --- |
| `"auto"` | Shape-aware exact/grid/FAISS HNSW/FAISS IVF selector. | Shape-aware CUDA grid for Euclidean/cosine/correlation 2D/3D self-KNN; Euclidean CUDA auto chooses exact Flat/brute force for measured small, medium, and high-dimensional accuracy-first shapes and IVF-Flat for large low-dimensional shapes; non-Euclidean auto keeps exact FAISS GPU Flat/cuVS brute force where available. Explicit CUDA exact/brute-force calls can use transformed cuVS brute force. | Default general-purpose choice. |
| `"grid"` | Native exact 2D/3D grid for Euclidean, cosine, and correlation. | Native CUDA 2D/3D grid for Euclidean, cosine, and correlation. | Low-dimensional spatial or simulated data; cosine/correlation use normalized Euclidean grid search; explicit grid requests error outside two or three columns. |
| `"vamana_style"` (`"vamana"` alias) | Package-owned DiskANN/Vamana-derived robust-pruned candidate graph with CPU candidate refinement. | Package-owned Vamana-derived graph with CUDA row-candidate refinement. | A distinct algorithm, not a feature-complete Vamana reproduction. Large high-dimensional CPU inputs use deterministic HNSW seed neighbours before robust pruning, while smaller CPU inputs keep exact seed neighbours. Robust pruning runs in compiled C++ over a compact candidate matrix, protects the first `k` seed neighbours, and then refines candidates on CPU or CUDA. CPU/CUDA tuning uses compiled shape/k/target-recall tables; seeded CUDA metric rows remain marked validation-pending [3,5,24]. |

Unsupported combinations fail before computation. For example,
`nn(x, backend = "cpu", method = "cagra")` errors because CAGRA is CUDA-only.
An explicit grid request also errors unless it is a two- or three-dimensional
self-search under Euclidean, cosine, or correlation distance.

## Compiled Backend Families

| Backend family | CPU | CUDA | Notes |
| --- | --- | --- | --- |
| Native faissR dense exact | yes | no | CRAN-friendly exact CPU baseline. |
| Native faissR grid | yes | optional CUDA | Exact 2D/3D Euclidean, cosine, and correlation self-KNN only. |
| FAISS Flat | yes | yes, if FAISS GPU is built | Exact L2 search [1-2,16]. |
| FAISS IVF-Flat | yes | yes, if FAISS GPU is built | Inverted-file approximate L2/IP search; cosine/correlation use normalized IP [1-2,16]. |
| FAISS IVF-PQ | yes | yes, if FAISS GPU is built | Product-quantized approximate L2/IP search; cosine/correlation use normalized IP [6,16]. |
| FAISS HNSW | yes, if exposed by FAISS | no | Approximate CPU graph-search index with L2/IP and normalized-IP metric transforms [5,16]. |
| FAISS NSG | yes, if exposed by FAISS | no | Optional internal CPU graph-search index for Euclidean/L2 only; public CPU NSG requests use faissR's distinct NSG/MRNG-derived algorithm instead [16,21,29]. |
| faissR Vamana-derived route | yes | optional CUDA refinement | Package-owned DiskANN/Vamana-derived robust-pruned candidate graph, not a feature-complete Vamana reproduction; large high-dimensional CPU inputs use deterministic HNSW seeding before robust pruning, and CUDA refines candidate rows [5,24]. |
| FAISS NNDescent | experimental opt-in | no | Disabled by default because linked FAISS builds can abort during graph construction; public CPU `method = "nndescent_style"` (or compatibility alias `"nndescent"`) uses the distinct package-owned NN-descent-derived algorithm [4,16]. |
| RAPIDS cuVS HNSW | no | yes, if cuVS is built with HNSW headers | Exposed as CUDA `method = "hnsw"`. The cuVS HNSW C API converts a CAGRA index to an HNSW wrapper and searches host-compatible tensors; faissR records this in metadata instead of benchmarking it as a pure all-GPU method [22]. |

Affected cuVS builds can fail in direct NN-descent on high-dimensional FP32 L2
inputs unless cuVS opts the L2-norm kernel into the required dynamic
shared-memory limit. faissR reports this case with a specific diagnostic and
does not replace an explicit CUDA/cuVS NN-descent request with another method.
See the copy-ready [cuVS issue report](cuvs-nndescent-shared-memory-issue.md).

## Model Functions

| Function | CPU | CUDA | Notes |
| --- | --- | --- | --- |
| `fast_kmeans()` | native/FAISS CPU k-means | FAISS GPU or direct cuVS k-means where available | Uses explicit `"cpu"` or `"cuda"` device selection [7-8]. |
| `knn()` / `predict()` | yes | yes, through `nn()` | Supervised classifier/regressor API reuses `nn()` backend and method resolution. |

## Availability Helpers

Use these helpers to inspect the build/runtime state:

```r
backend_info()
gpuinfo::hardware_info()$gpu
nn_capabilities()
faiss_available()
faiss_gpu_available()
cuda_available()
cuvs_available()
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
record the resolved provider in `attr(result, "approximation")` as
`cagra_provider` and the normalized selector as `cagra_provider_option`.
The boolean helpers return a single
`TRUE`/`FALSE` value. They are useful for diagnostics and examples, but
explicit backend calls still validate availability at execution time.
`nn_capabilities()` returns a data frame with one row per public
method/backend/metric combination for `"cpu"` and `"cuda"`, and marks
unsupported combinations before a benchmark tries to run them.

For benchmark launchers, `nn_capabilities(runtime = TRUE)` adds the
implementation route that the public API would request on the current machine,
whether that route is available in the installed build, a stable
`runtime_reason`, and human-readable `runtime_notes`. This separates
method/metric validity from local runtime availability, for example a valid
FAISS GPU Flat row on a CPU-only installation. Reason labels include
`available`, `unsupported_combination`, `missing_faiss`, `missing_faiss_gpu`,
`missing_cuda`, `missing_cuda_route`, and `missing_cuvs`, so benchmark scripts
do not need to parse prose.

The capability table is design-level. For example, CUDA Euclidean
`method = "auto"` routes use CUDA grid for large 2D/3D self-KNN and
otherwise choose between Flat/brute force and IVF-Flat by shape, `k`, and
`target_recall`. Non-grid CUDA cosine and correlation auto routes use FAISS GPU
Flat for exact small/query workloads when available, and can select validated
for exact small/query workloads when available, or a transformed validated graph
route for large self-KNN when available.
Explicit CUDA HNSW is routed to the
cuVS HNSW-from-CAGRA wrapper path and labelled as such in metadata. On a
cuVS-only runtime, CUDA non-Euclidean capability rows are reported as
shape-dependent rather than as a promise that every metric/method has a CUDA
route. In mixed FAISS/cuVS builds, the shape-dependent route can choose
transformed FAISS GPU/direct cuVS CAGRA for large self-KNN when those compiled
routes are available.
FAISS CPU and FAISS GPU availability are checked separately at execution time:
explicit FAISS GPU Flat, IVF, IVFPQ, and CAGRA routes require a FAISS build
that reports GPU support, not only a CPU FAISS installation.

## Tuning And Approximation Metadata

Approximate GPU routes use deterministic no-pilot defaults for
`tuning = "auto"`. FAISS IVF records fixed shape/k/metric-aware
`nlist`/`nprobe` metadata, and cuVS CAGRA records fixed graph/search metadata.
The route selector and deterministic parameter selectors are compiled C++
policies: `nn_auto_select_backend_cpp()` chooses the method route within the
requested device, and
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
quality as `exact-audited` only after the exhaustive-route and reference audit
passes. Tied kth-neighbor boundaries are audited using identifier equality or
equivalence of the sorted distance multiset within the frozen tolerance; raw
set overlap is diagnostic and cannot disqualify an audited exact route.

## Installation Implications

- A CPU-only installation still requires FAISS.
- CUDA/cuVS support is optional for CPU-only builds and enabled only
  when matching headers and libraries are available.
- NVIDIA GPU builds should use the strict `FAISSR_REQUIRE_*` configure
  variables so missing CUDA/cuVS libraries fail installation.
- Explicit CUDA requests fail clearly on CPU-only builds.
- The package does not vendor FAISS, cuVS, or CUDA. See
  [Installation](installation.md) for build variables and
  [References](references.md) for software acknowledgements.
CUDA exact and brute-force requests use direct cuVS brute force when available,
while FAISS GPU Flat remains the provider-backed alternative.
