# Nearest-Neighbour Methods

Cosine zero vectors and constant correlation rows are zero-normalized edge
cases with documented, backend-dependent behavior; `nn_metric_preflight()`
reports affected rows before search.

[Home](../README.md) |
[Installation](installation.md) |
[Implementation](implementation.md) |
[Examples](examples.md) |
[Benchmarks](benchmarks.md) |
[Autotuning](autotuning.md) |
[API](usage-api.md) |
**NN Methods** |
[Backends](backend-capabilities.md) |
[References](references.md)

This page describes the `method` argument used by `nn()`, `nn(..., exclude_self = TRUE)`,
and `knn()`. In faissR, `backend` chooses the device family (`"cpu"` or
`"cuda"`), while `method` chooses the nearest-neighbour algorithm family.
Distance choices belong in `metric`, not in `method`.

## Quick Selection Guide

| Goal | Suggested call |
| --- | --- |
| Let faissR choose a method on CPU | `nn(x, k, backend = "cpu", method = "auto")` |
| Exact CPU reference | `nn(x, k, backend = "cpu", method = "exact")` |
| Exact FAISS CPU/GPU route | `nn(x, k, backend = "cpu", method = "flat")` or `nn(x, k, backend = "cuda", method = "flat")` |
| Exact CUDA route through cuVS when available | `nn(x, k, backend = "cuda", method = "bruteforce")` |
| Large high-dimensional CPU approximate search | `nn(x, k, backend = "cpu", method = "hnsw")` |
| Large CUDA graph search | `nn(x, k, backend = "cuda", method = "cagra")` |
| Memory-pressure approximate search | `nn(x, k, backend = "cpu", method = "ivfpq")` or `nn(x, k, backend = "cuda", method = "ivfpq")` |
| IVFPQ FastScan 4-bit PQ scan | `nn(x, k, backend = "cpu", method = "ivfpq_fastscan")` or `nn(x, k, backend = "cuda", method = "ivfpq_fastscan")` |
| 2D/3D spatial self-KNN | `nn(x, k, backend = "cpu", method = "grid")` or `nn(x, k, backend = "cuda", method = "grid")` |

Use `backend_info()` to inspect which compiled CPU, FAISS, CUDA, cuVS, and
cuVS capabilities are available on a given machine.
Use `nn_capabilities()` to return the same method/backend/metric support matrix
as a data frame for benchmark preflight checks, including rows for
`backend = "cpu"` and `"cuda"`. Use
`nn_capabilities(runtime = TRUE)` when a benchmark script also needs the
current build/runtime status; it appends `resolved_backend`,
`runtime_available`, `runtime_reason`, and `runtime_notes` columns so FAISS
GPU, CUDA, and cuVS rows can be skipped or labelled before computation starts.
The `runtime_reason` labels are machine-readable, for example `available`,
`unsupported_combination`, `missing_faiss`, `missing_faiss_gpu`,
`missing_cuda`, `missing_cuda_route`, and `missing_cuvs`.

## Numerical Contract And Input Preflight

Returned Euclidean distances are ordinary L2 distances. Providers that compute
squared L2 internally are square-rooted before faissR returns the result.
Cosine normalization and correlation centering/normalization on FAISS and cuVS
routes produce row-major float32 coordinates; row sums and squared norms are
accumulated in double precision before the transformed coordinates are stored
as floats. Native double routes retain R double input. Passing an R double
matrix to a float32 provider therefore includes a double-to-float32 conversion.

All backends reject `NA`, `NaN`, `Inf`, and `-Inf`. On CPU, two all-zero cosine
rows or two constant correlation rows have package-defined distance `0`, while
a degenerate row versus a non-degenerate row has distance `1`. Explicit CUDA
routes reject those rows rather than silently repairing results on CPU. These
are intentionally backend-dependent conventions. Inspect them before search:

```r
check <- nn_metric_preflight(x, metric = "cosine", backend = "cuda")
check$data_rows
check$action
```

Native exact results sort ties by `(distance, reference row ID)`. External
providers may return a different member of a tied boundary set; exact audits
therefore compare sorted distances and admit tied boundary identifiers. The
publication's frozen float32 audit uses
`abs(observed - reference) <= 1e-5 + 1e-4 * abs(reference)`. This tolerance is
an exhaustive-route audit threshold chosen for float32 accumulation and
provider ordering differences; it is not used by search or approximate-recall
eligibility.

CUDA result cleanup is deliberately backend-owned. When a CUDA route returns
`attr(result, "gpu_residency")`, faissR treats the indices/distances as already
ordered and shaped by C++/CUDA, and does not remove self-neighbours or reshape
include-self columns in R. CUDA graph routes without compiled include-self
layout support require `exclude_self = TRUE` and report an error otherwise.

## Method Summary

| `method` | Exact? | CPU | CUDA | Main references |
| --- | --- | --- | --- | --- |
| `"auto"` | depends on selected route | yes | yes | FAISS/cuVS/HNSW/IVF/CAGRA as selected [1-6,13-16,22-23,34] |
| `"flat"` | yes | FAISS Flat | FAISS GPU Flat | FAISS [1-2,16] |
| `"grid"` | yes | native 2D/3D grid | native CUDA 2D/3D grid | native faissR implementation |
| `"hnsw"` | approximate | FAISS HNSW for all metrics | cuVS HNSW from CAGRA | HNSW/cuVS API note [5,16,22-23] |
| `"ivf"` | approximate | FAISS IVF-Flat | FAISS GPU IVF-Flat | FAISS IVF [1-2,16] |
| `"ivfpq"` | approximate | FAISS IVF-PQ | FAISS GPU IVF-PQ | product quantization [6,16] |
| `"ivfpq_fastscan"` | approximate | FAISS IVFPQ FastScan with Flat refinement | cuVS IVF-PQ with 4-bit compressed codes | 4-bit IVFPQ compressed-code scan [6,34] |
| `"vamana_style"` | approximate, experimental | package-owned Vamana-derived candidate graph | package-owned Vamana-derived candidate graph with CUDA refinement | DiskANN/Vamana [3,24] |
| `"nsg_style"` | approximate, experimental | package-owned CPU NSG/MRNG-derived candidate graph | package-owned CUDA NSG/MRNG-derived candidate graph | NSG/FAISS [16,21,29] |
| `"nndescent_style"` | approximate; native CPU route experimental | package-owned NN-descent-derived refinement | direct cuVS NN-descent provider | NN-descent/cuVS [4,13-16] |
| `"cagra"` | approximate | unsupported | FAISS GPU CAGRA or cuVS CAGRA | FAISS/cuVS CAGRA [3,13-16] |

For clarity, new code should use `"nsg_style"`, `"vamana_style"`, and
`"nndescent_style"` when requesting faissR's package-owned graph refinements.
The shorter strings in the table remain compatibility aliases and resolve to
the same internal method families. Native results record the preferred spelling
in `preferred_public_method`, set
`canonical_reimplementation = FALSE` and identify themselves as package-owned
derived graph-refinement algorithms. The `_style` suffix is a scope marker,
not an algorithm name. Package-owned style routes are experimental, carry
machine-readable status metadata, and are excluded from the publication's
principal performance claims. Direct cuVS NN-descent is an external-provider
route and is labeled separately.

### Package-owned experimental graph refinements

All three native routes are self-KNN candidate-refinement procedures. Cosine
input is row-normalized before Euclidean refinement; correlation input is
row-centered and then row-normalized. For a fixed input, seed, and parameter
set, the package-owned stages are deterministic. A provider used to construct
the seed graph can retain its own documented floating-point or threading
variation.

`nsg_style` starts from an ordered seed-neighbor matrix. For each row it keeps
the first `k` candidates, then accepts a later candidate only when no retained
candidate is closer to it than the query is. It fills unused degree slots from
the original order and exactly rescores the retained row candidates. This
borrows the relative-neighborhood pruning test used by NSG/MRNG constructions,
but omits canonical NSG navigation, medoid selection, connectivity repair, and
graph traversal.

`vamana_style` uses the same seed/refine framework with robust pruning: a
candidate at query distance `d(q, v)` is rejected when a retained candidate
`u` satisfies `alpha * d(u, v) <= d(q, v)`. It does not implement canonical
Vamana's incremental insertion order, medoid graph search, reciprocal edge
updates, reachability repair, or disk layout.

`nndescent_style` on CPU initializes each candidate pool from Gaussian random
projections plus seeded random fill. Each fixed refinement iteration uses a
frozen graph snapshot, bounded reverse-neighbor lists, and capped
neighbor-of-neighbor joins before retaining the best candidate pool. It omits
canonical NN-descent's new/old flags, sampled local joins, and convergence
stopping. On CUDA the same public name can resolve instead to direct cuVS
NN-descent; returned metadata distinguishes that external provider.

Let `n` be the number of rows, `p` the dimension, `s` the seed width, `r` the
retained degree, `C` the NN-descent candidate cap, `P` the projection count,
and `t` the number of refinement iterations. Exact seed construction dominates
NSG-/Vamana-style execution at `O(n^2 p)`; pruning is at most `O(n r^2 p)` and
row rescoring is `O(n r p)`, with `O(np + ns + nr)` memory. NN-descent-style
initialization costs approximately `O(Pnp + Pn log n + nLp)` and refinement is
bounded by `O(tnCp)`, with `O(np + n(L + C))` memory. These are implementation
bounds, not complexity claims for the canonical algorithms.

The native low-dimensional `grid` route is different: it is an exact 2D/3D
uniform-grid search with expanding cells and a distance lower-bound stopping
rule. Build cost and memory are `O(n + B^d)` for `B` cells per axis and
`d` in `{2, 3}`; query cost is occupancy-dependent and `O(n)` in the worst
case. IVF-PQ and FastScan are provider-backed FAISS/cuVS implementations, not
package-owned algorithms. They remain supported secondary routes, but the
current principal held-out comparison does not make route-specific performance
claims for them.

## Metric Support Matrix

faissR exposes three canonical metrics: `"euclidean"`, `"cosine"`, and
`"correlation"`. Correlation is centered cosine similarity. Legacy aliases are
rejected before dispatch. Cosine and correlation may use normalized
documented distances.

| Method | CPU metrics | CUDA metrics | Notes |
| --- | --- | --- | --- |
| `"auto"` | euclidean, cosine, correlation | euclidean, cosine, correlation | Shape-, k-, target-, and capability-aware selection. |
| `"exact"` | euclidean, cosine, correlation | euclidean, cosine, correlation | Exhaustive provider route. |
| `"flat"` | euclidean, cosine, correlation | euclidean, cosine, correlation | FAISS Flat route. |
| `"bruteforce"` | euclidean, cosine, correlation | euclidean, cosine, correlation | FAISS CPU Flat or direct CUDA/cuVS exhaustive route. |
| `"grid"` | euclidean, cosine, correlation | euclidean, cosine, correlation | 2D/3D self-KNN only. |
| `"hnsw"` | euclidean, cosine, correlation | euclidean, cosine, correlation | FAISS CPU HNSW or cuVS HNSW-from-CAGRA. |
| `"ivf"` | euclidean, cosine, correlation | euclidean, cosine, correlation | FAISS IVF-Flat. |
| `"ivfpq"` | euclidean, cosine, correlation | euclidean, cosine, correlation | FAISS IVF-PQ. |
| `"ivfpq_fastscan"` | euclidean, cosine, correlation | euclidean, cosine, correlation | FAISS FastScan on CPU; cuVS 4-bit IVF-PQ on CUDA. |
| `"vamana_style"` | euclidean, cosine, correlation | euclidean, cosine, correlation | Package-owned Vamana-derived refinement. |
| `"nsg_style"` | euclidean, cosine, correlation | euclidean, cosine, correlation | Package-owned NSG/MRNG-derived refinement. |
| `"nndescent_style"` | euclidean, cosine, correlation | euclidean, cosine, correlation | Package-owned CPU route; external-provider cuVS route on CUDA. |
| `"cagra"` | unsupported | euclidean, cosine, correlation | CUDA-only FAISS GPU/cuVS graph search. |

Programmatic form:

```r
nn_capabilities()
```

Benchmark scripts should treat `supported = FALSE` rows from this table as
expected skips, not algorithmic failures.

## `"auto"`

`method = "auto"` is the default. It chooses a route from the selected
`backend` and the data shape. The route decision is made by the compiled C++
`nn_auto_select_backend_cpp()` selector after the R wrapper has normalized
arguments and collected runtime capability flags:

- CPU auto uses exact CPU for small work, native grid for large 2D/3D
  Euclidean/cosine/correlation self-search, FAISS IVF for some million-row low-dimensional cases,
  FAISS HNSW for large high-dimensional self-search, including non-Euclidean
  HNSW when FAISS exposes it, FAISS Flat exact search for larger cosine,
  NSG/MRNG-derived route for selected larger non-Euclidean self-KNN cases, and
  native CPU NN-descent for other large self-KNN cases, instead of exact brute
  force [1-2,5,16,21].
- CUDA auto uses CUDA grid for large 2D/3D Euclidean/cosine/correlation
  self-search. For Euclidean non-grid self-KNN, the compiled selector chooses
  between exact Flat/brute force and IVF-Flat using dataset shape, `k`, and
  `target_recall`: IVF is preferred for COIL20-like, MNIST/Fashion-like,
  flow-like, and ImageNet-like shapes when the tuning evidence reaches the
  requested recall; Flat is kept for query searches, tiny matrices, very small
  `k`, and shape/target combinations where IVF did not meet the target.
  FAISS GPU Flat or validated graph routes when available [1-3,13-16].

`auto` is intended as a balanced default, not a guarantee of the fastest method
for every dataset. For benchmarking, report the requested backend/method/tuning
stored in `attr(result, "requested_backend")`,
`attr(result, "requested_method")`, and `attr(result, "tuning")` together with
the resolved backend in `attr(result, "resolved_backend")` and approximation
parameters. Auto requests also carry `attr(result, "auto_selection")`, a
compiled no-pilot record of the workload/shape/k/metric rule that predicted the concrete
route. The record stores
`policy = "cpp_static_shape_k_metric_selector"`, keeps the internal
`predicted_backend`, and also exposes `predicted_method` plus
`predicted_device`. It also records
`explicit_backend`, `explicit_method`, `backend_decision`, and
`method_decision`, so benchmark tables can distinguish a forced CPU/CUDA or
forced method request from an automatic shape-policy choice without parsing
implementation labels or rerunning any tuning logic inside `nn()`.

After the route is selected, deterministic `tuning = "auto"` parameters are
also computed by compiled C++ helpers. FAISS HNSW, IVF, IVFPQ/PQ, FAISS graph
indexes, cuVS CAGRA/NN-descent/IVFPQ, native NSG, Vamana, and native CUDA
NN-descent all return parameter metadata with
`tuning_source = "cpp"`. R options remain the user-facing way to override
defaults, but clipping, shape/k/metric tier labels, and requested/effective
values come from the C++ policy layer.

FAISS CPU HNSW adds method-specific no-pilot tuning metadata in
`attr(result, "approximation")`. The policy records `tuning_rule` plus
high-dimensional, large-`n`, small-`k`, large-`k`, and non-Euclidean flags, so
benchmarks can distinguish the low-dimensional small-`k` speed tier, the
small-`k` metric-aware tier, the general balanced tier, and the high-recall
large-`k`/high-dimensional tier [5].

## `"hnsw"` Metrics

CPU `method = "hnsw"` is metric-aware. faissR uses FAISS HNSW for
correlation is implemented by row centering plus L2 normalization followed by
distances to the package convention `best_dot - dot`, so the first returned
neighbour has distance zero.

CUDA `method = "hnsw"` uses RAPIDS cuVS HNSW when cuVS is available. The
resolved implementation label, such as `cuda_cuvs_hnsw`, is metadata only; it is
not a separate public method or backend option. RAPIDS cuVS HNSW is documented
as a CAGRA-to-HNSW wrapper that converts a CUDA CAGRA index and searches
host-compatible tensors, so faissR records
`cuda_hnsw_design = "cuvs_hnsw_from_cagra_cpu_hierarchy"` instead of presenting
it as a pure all-GPU HNSW implementation. Use CUDA `method = "cagra"` when the
goal is a fully GPU graph-search baseline. CUHNSW is acknowledged as related
Apache-2.0 CUDA HNSW prior software, but faissR does not vendor or copy CUHNSW
source [22-23].

## `"exact"`

`method = "exact"` requests exhaustive exact KNN.

- On CPU, Euclidean exact search uses FAISS Flat L2.
  Cosine exact search uses row L2 normalization followed by FAISS Flat
- CPU Euclidean, cosine, and correlation exact have compiled `tuning = "auto"` policies
  keyed by metric, shape group, `k` bucket, and `target_recall`. They tune FAISS
  query batch size and fitted Flat-index reuse where implemented, while result
  metadata records the legacy exhaustive-route fields
  `exact_recall_by_construction = TRUE` and `expected_recall_at_k = 1.0`.
  Publication analyses nevertheless use an `exact_audited` classification:
  identifier equality or tie-aware distance-multiset equivalence establishes
  exactness, while raw set-overlap recall remains diagnostic at tied boundaries.
- When `data` or `points` is a `float::fl()`/float32 matrix, this route reads
  the float32 payload directly or with one layout conversion instead of first
  expanding the input to an R double matrix.
- On CUDA, faissR uses FAISS GPU Flat only when the linked FAISS build reports
  GPU support, otherwise direct cuVS brute force when available [1-3,16].
  compiled shape/k/target policies for GPU query batching; correlation rows are
  derived from `faissR_EXACT_TUNING_CUDA_correlation_20260703_023519` and
  summarized in `benchmark_scripts/cuda_exact_correlation_shape_tuning_defaults_from_uploaded_results.csv`.
  as validation-pending batch/resource defaults until the dedicated sweep is
  uploaded.

Exact search is the best reference for recall and correctness checks. It can be
too slow or too memory-heavy for full all-pairs self-KNN on very large datasets.
FAISS exact/Flat searches are internally query-batched; use
`FAISSR_FAISS_QUERY_BATCH_SIZE` or `FAISSR_FAISS_GPU_QUERY_BATCH_SIZE` to tune
the batch size for CPU or FAISS GPU builds.

## `"flat"`

`method = "flat"` requests a FAISS Flat exhaustive index [1-2,16].

- `backend = "cpu"` maps to FAISS CPU Flat.
- `backend = "cuda"` maps to FAISS GPU Flat.
  by metric, shape group, `k` bucket, and `target_recall`, derived from
  `faissR_FLAT_TUNING_CPU12_euclidean_20260630_161409` and
  `faissR_FLAT_TUNING_CPU12_cosine_20260701_015607`, and
  `faissR_FLAT_TUNING_CPU12_correlation_20260701_090337`, and
  batch size and fitted-index reuse on CPU. CUDA Euclidean, cosine, correlation,
  the CUDA correlation rows come from
  `faissR_FLAT_TUNING_CUDA_correlation_20260703_062359` and are summarized in
  `benchmark_scripts/cuda_flat_correlation_shape_tuning_defaults_from_uploaded_results.csv`.
  seeded from measured CUDA Flat Euclidean rows until the dedicated
  The selected policy is stored in `attr(result, "flat_tuning")`;
  `tuning_benchmark_target_met = FALSE`.

Flat search is exact for L2/Euclidean search and is useful when you want FAISS
uses `IndexFlatIP`; `metric = "cosine"` uses row L2 normalization followed by
Flat IP; and `metric = "correlation"` uses row centering plus L2 normalization
followed by Flat IP. The cosine and correlation routes return
dot-product distances. Flat can be
faster than a generic R exact implementation because index construction, data
layout, and search are handled by FAISS. Float32 inputs are passed through the
C++ float32 adapter; ordinary R double matrices are converted once to the
row-major float32 layout that FAISS expects. FAISS GPU Flat reuses a
thread-local `StandardGpuResources` object across calls by default; set
`FAISSR_FAISS_GPU_REUSE_RESOURCES=0` only when debugging resource lifetime
issues.

## `"bruteforce"`

`method = "bruteforce"` requests exhaustive brute-force search.

- On CPU, Euclidean brute force uses the batched FAISS Flat L2 route, cosine
  brute force uses normalized FAISS Flat cosine, correlation brute force
  force uses FAISS Flat IP.
  by metric, shape group, `k` bucket, and `target_recall`, derived from
  `faissR_BRUTEFORCE_TUNING_CPU12_euclidean_20260630_161409`,
  `faissR_BRUTEFORCE_TUNING_CPU12_cosine_20260630_161535`, and
  `faissR_BRUTEFORCE_TUNING_CPU12_correlation_20260701_090337`, plus
  batch size and fitted-index reuse separately from `method = "exact"` and
  `method = "flat"` and stores the selected policy in
  large-dataset coverage are labelled as partial in that metadata.
- With float32 input and FAISS available, CPU brute-force requests avoid a
  float32-to-double expansion.
- On CUDA, RAPIDS cuVS brute force is preferred when available [3].
  query-batch and resource reuse policy in `attr(result, "bruteforce_tuning")`;
  cuVS brute-force defaults as proxy/seed rows until the metric-specific sweeps
  replace them.
- Cosine and correlation use row transforms followed by exact cuVS L2 search;
  before exact cuVS L2 search [1-3,16].

This method is useful for comparing FAISS GPU Flat with direct cuVS exhaustive
search. Both are exhaustive routes, but implementation details, transfer
costs, and batching can differ.

## `"grid"`

`method = "grid"` uses faissR's native spatial grid implementation.

- On CPU, it supports 2D/3D Euclidean, cosine, and correlation self-KNN.
- On CUDA, it supports the CUDA 2D/3D grid route when compiled.
- It errors for higher-dimensional data.
- Include-self results are finalized inside the C++/CUDA route, so the self
  column is not prepended by R-side matrix reshaping.

Grid search is intended for low-dimensional spatial data or simulated 2D/3D
benchmarks. It is not a general high-dimensional ANN algorithm.

## `"hnsw"`

`method = "hnsw"` requests an HNSW graph-based approximate nearest-neighbour
route based on Hierarchical Navigable Small World graphs [5,16].

- On CPU, faissR uses FAISS HNSW.
- On CUDA, faissR uses RAPIDS cuVS HNSW by building a CAGRA seed graph and
  converting it with `cuvsHnswFromCagraWithDataset`. The route records
  `cuda_hnsw_design = "cuvs_hnsw_from_cagra_cpu_hierarchy"` because it is not a
  pure all-GPU HNSW implementation [22].
- HNSW supports Euclidean, cosine, and correlation through
  the metric transforms described above.
- HNSW is often a strong default for large high-dimensional CPU self-KNN.
- Tuning parameters include HNSW `M`, construction effort, and search effort.
  compiled CPU12 HPC shape/k tiers by default, with k buckets 15, 30, 50, and
  100. CUDA HNSW uses separate Euclidean, cosine, and correlation tables for
  the cuVS HNSW-from-CAGRA route. Use
  `target_recall = 0.9`, `0.95`, or `0.99` to choose the speed/recall tier;
  result metadata records `tuning_benchmark_target_met` when a stored row is
  best-available rather than verified across every dataset in its shape group.
- In `knn()`, explicit CPU FAISS `method = "flat"`, `"hnsw"`, `"ivf"`, and
  `"ivfpq"` models store a session-local FAISS external pointer. Matching
  `predict()` calls reuse that fitted index and report
  `approximation$index_reused = TRUE`; IVF/IVFPQ metadata also reports
  `centroids_reused`, `inverted_lists_reused`, `build_nprobe`, and
  `search_nprobe`. IVFPQ metadata additionally reports `pq_codebooks_reused`,
  `pq_codes_reused`, and `search_pq_train_call_count = 0` for compatible
  predictions. Saved/reloaded models rebuild the same route because
  external pointers are session-local.

HNSW is approximate. It can give excellent recall/speed trade-offs, but recall
should be measured for new datasets when it is used for scientific conclusions.

## `"ivf"`

`method = "ivf"` requests an inverted-file Flat index [1-2,16].

- On CPU, it maps to FAISS CPU IVF-Flat.
- On CUDA, it maps to FAISS GPU IVF-Flat.
- The main parameters are the number of coarse lists (`nlist`) and searched
  lists (`nprobe`).
- On CPU, `tuning = "auto"` chooses `nlist` and `nprobe` from compiled
  shape/k/target-recall tables for Euclidean, cosine, and correlation. The
  selected row records `tuning_benchmark_target_met`, so best-available partial
  or below-target rows remain visible rather than being presented as guaranteed
  recall.
- On CUDA, `tuning = "auto"` chooses `nlist` and `nprobe` from compiled
  shape/k/target-recall policies for Euclidean, cosine, and correlation IVF.
  Euclidean rows come from
  `faissR_IVF_TUNING_CUDA_euclidean_20260702_001853`; cosine rows come from
  `faissR_IVF_TUNING_CUDA_cosine_20260702_192200` after row normalization;
  correlation rows come from
  `faissR_IVF_TUNING_CUDA_correlation_20260703_133655` after row centering and
  normalization. The policy separates compact high-dimensional data, medium
  image-like data, large low-dimensional flow-like data, and large
  high-dimensional ImageNet-like data. Use `target_recall = 0.9`, `0.95`, or
  `0.99` to pick the speed/accuracy tier; partial shape rows report
  `tuning_benchmark_target_met = FALSE`.
- Manual CUDA IVF overrides are available through `options(faissR.cuda_ivf_nlist
  = ..., faissR.cuda_ivf_nprobe = ...)`, with provider-specific aliases
  `cuvs_ivf_*` and `faiss_gpu_ivf_*`.
- `metric = "cosine"` uses row normalization followed by L2 search and returns
  `1 - similarity`.
- `metric = "correlation"` uses row centering plus L2 normalization followed by
  L2 search and returns `1 - similarity`.

IVF partitions the vector space into coarse cells and searches a subset of
cells. It is approximate unless `nprobe` approaches the number of lists. It is
useful for large datasets where exhaustive search is too expensive.
The direct diagnostic backend `cuda_cuvs_ivf_flat` uses a RAPIDS cuVS IVF-Flat
L2 index.

## `"ivfpq"`

`method = "ivfpq"` requests IVF with product quantization [6,16].

- On CPU, it maps to FAISS CPU IVF-PQ.
- On CUDA, it maps to FAISS GPU IVF-PQ.
- It compresses vectors using product quantization and searches compressed
  codes.
- `metric = "cosine"` and `"correlation"` use normalized L2 search after the
  corresponding row transformation.
- On CPU, `tuning = "auto"` chooses `nlist`, `nprobe`, `pq_m`, and `pq_nbits`
  from compiled shape/k/target-recall tables for Euclidean, cosine,
  and correlation. Rows labelled `best_available_partial_shape_datasets_*`
  are best measured
  compression settings, not guarantees that the target recall was reached.
- On CUDA, `tuning = "auto"` selects `nlist`, `nprobe`, `pq_m`, and `pq_nbits`
  from FAISS GPU IVFPQ shape/k/target
  tables. The correlation table is summarized in
  `benchmark_scripts/cuda_ivfpq_correlation_shape_tuning_defaults_from_uploaded_results.csv`
  from `faissR_IVFPQ_TUNING_CUDA_correlation_20260703_095008`; many rows are
  best-available or below-target and report `tuning_benchmark_target_met = FALSE`.

IVFPQ is a memory-pressure method. It can be fast and memory-efficient, but
recall can drop substantially. Treat it as explicit opt-in when memory matters,
not as the default accuracy-first method. The direct diagnostic backend
`cuda_cuvs_ivfpq` applies the same transformed cosine/correlation convention
to a cuVS L2/PQ index.

CPU IVFPQ requires at least 624 training rows. This deterministic guard avoids
FAISS training runs where even the smallest supported 4-bit product quantizer
has too few training examples per codeword; smaller datasets should use
`method = "ivf"`, `"hnsw"`, or `"flat"` instead.
For 624-9,983 rows, CPU FAISS auto tuning requests 4-bit PQ so the codebook
size remains compatible with FAISS' recommended training density; 8-bit PQ is
used once the training set is large enough unless the user overrides
`faissR.faiss_pq_nbits`. Direct cuVS IVF-PQ follows the same small-training
principle and requests 4-bit PQ below 9,984 rows unless
`faissR.cuvs_ivfpq_pq_bits` or `faissR.ivfpq_pq_bits` is set. FAISS GPU IVFPQ
is an explicit 8-bit route because FAISS' GPU IVFPQ implementation requires
8-bit product-quantizer codes.

When fitted through `knn(..., method = "ivfpq")`, the session-local model stores
the trained FAISS IVFPQ external pointer. Compatible `predict()` calls reuse the
coarse centroids, inverted lists, PQ codebooks, and compressed codes instead of
training IVFPQ again; metadata records `pq_codebooks_reused`,
`pq_codes_reused`, and `search_pq_train_call_count`.

## `"ivfpq_fastscan"`

`method = "ivfpq_fastscan"` requests an IVFPQ FastScan approximate compressed-code scan.
On CPU, faissR uses FAISS `IndexIVFPQFastScan`: 4-bit PQ/AQ codes are scanned using
lookup tables kept in SIMD registers, with optional refinement by a Flat index
for reranking [6,34].

- `backend = "cpu"` maps to `faiss_ivfpq_fastscan`, a FAISS
  `IndexIVFPQFastScan` route with 4-bit PQ, deterministic IVF
  `nlist`/`nprobe`, and optional `IndexRefineFlat` reranking. The route
  requires a linked FAISS build that exposes `faiss/IndexIVFPQFastScan.h`.
- `backend = "cuda"` maps to `cuda_cuvs_ivfpq_fastscan`, a direct RAPIDS cuVS IVF-PQ
  route with 4-bit compressed codes. This is the CUDA IVFPQ FastScan route in
  faissR; it does not silently fall back to CPU FAISS FastScan. Compatible raw
  `nn()` calls reuse a fitted cuVS IVF-PQ index, dataset device buffer, and cuVS
  resources so repeated queries do not retrain centroids or rebuild PQ codes.
  Self-query searches reuse the fitted dataset device buffer directly; repeated
  searches with the same separate query matrix can reuse one cached query device
  buffer when `options(faissR.cache_cuda_ivfpq_query_buffers = TRUE)`.
  Queries are submitted to cuVS in large batches, controlled by
  `FAISSR_CUVS_IVF_BATCH_SIZE` (default 32768). For multi-query calls, the C++
  backend clamps invalid tiny values so the search does not degrade into
  row-by-row cuVS calls.
- CUDA 4-bit IVF-PQ requires byte-aligned packed PQ codes. faissR therefore
  repairs invalid `pq_dim` values before calling cuVS, for example reducing an
  odd manual value to the nearest valid lower dimension, and reports the
  requested and actual PQ parameters in metadata. In CUDA tuning runs, smaller
  `pq_dim` and smaller `nprobe` generally improve speed but can lower recall;
  `nlist` controls the IVF build/search balance.
- The CPU public route supports `metric = "euclidean"`, `metric = "cosine"`,
  and `metric = "correlation"`. Cosine normalizes rows, searches the normalized
  vectors with FastScan L2, and converts squared normalized Euclidean distances
  to `1 - cosine`; correlation subtracts each row mean before the same
  normalization, search, and distance conversion.
- The CUDA public route supports `metric = "euclidean"`, `metric = "cosine"`,
  and `metric = "correlation"`. Cosine normalizes rows; correlation subtracts
  each row mean and then normalizes rows. These transformed metrics search with
  cuVS IVF-PQ L2 and convert distances back to the public convention.
- Metadata in `attr(result, "approximation")` records `ivfpq_fastscan`,
  `fastscan`, the IVF/PQ parameters, and whether the CPU route used Flat
  refinement.
- With `tuning = "auto"`, CPU Euclidean, cosine, and correlation use
  shape/k/target policies keyed by dataset shape, `k`, and `target_recall`.
  The Euclidean table is HPC-derived. CUDA cosine and correlation use separate
  seeded policy files,
  `cuda_ivfpq_fastscan_cosine_shape_tuning_defaults_from_seeded_euclidean_results.csv`,
  `cuda_ivfpq_fastscan_correlation_shape_tuning_defaults_from_seeded_euclidean_results.csv`.
  These are seed policies using the Euclidean FastScan settings where the
  uploaded metric-specific sweeps failed before reaching the backend; those rows
  therefore record `tuning_benchmark_target_met = FALSE` until the corrected
  benchmarks are rerun.
  The table separates small high-dimensional data from small lower-dimensional
  data because COIL20 and USPS required different `nlist`/`nprobe` settings.
- Users can override CPU FastScan details with
  `options(faissR.ivfpq_fastscan_pq_m = ..., faissR.ivfpq_fastscan_refine_factor = ...,
  faissR.ivfpq_fastscan_bbs = ...)`; defaults keep 4-bit PQ and a small Flat reranking
  factor for quality.
- Repeated compatible raw `nn()` calls reuse session-local fitted indexes.
  CPU reuses the fitted FAISS FastScan index, including trained IVF centroids,
  inverted lists, PQ codebooks/codes, and the optional Flat refinement wrapper.
  CUDA reuses the fitted cuVS IVF-PQ index, dataset device buffer, compressed
  codes, and cuVS resources. CUDA metadata reports `dataset_residency`,
  `query_residency`, `query_device_cache_status`, and
  `query_host_to_device_copies` so benchmark runs can separate search-kernel time
  from host-device traffic.
- The HPC tuning wrappers expose these FastScan grids through
  `IVFPQ_FASTSCAN_NLIST_MULTS`, `IVFPQ_FASTSCAN_NPROBE_MULTS`,
  `IVFPQ_FASTSCAN_PQ_DIMS` for CUDA, and `CUVS_IVF_BATCH_SIZES`. The R driver
  also accepts `--ivfpq_fastscan_nlist_multipliers`,
  `--ivfpq_fastscan_nprobe_multipliers`, `--ivfpq_fastscan_pq_dim_values`, and
  `--cuvs_ivf_batch_sizes`.
- The CPU route always enters the FAISS FastScan C++ adapter as `float*`.
  `float::fl()`/float32 inputs avoid R double expansion; ordinary R double
  inputs are adapted once to row-major float32. `output = "float"` controls
  only the returned distance container, and self-neighbour removal stays in C++.

Use this method when you want to test an IVFPQ FastScan compressed-code scan
separately from general `method = "ivfpq"`. It is approximate, so benchmark
reports should include recall or another quality measure.

## `"vamana"`

`method = "vamana"` requests a distinct robust-pruned graph algorithm
implemented inside faissR and derived from selected DiskANN/Vamana ideas [24].

- CPU `method = "vamana"` builds a candidate graph, applies Vamana-derived
  robust pruning controlled by `alpha`, and refines top-k neighbours inside
  each candidate row with faissR's CPU candidate KNN scorer. Large
  high-dimensional CPU inputs use deterministic HNSW seed neighbours before
  robust pruning; smaller CPU inputs keep exact seed neighbours.
- CUDA `method = "vamana"` uses the same candidate graph semantics and refines
  candidate rows with faissR's native CUDA row-candidate KNN kernel.
- The deterministic pruning step protects the first `k` seed neighbours before
  applying robust pruning, so small-`k` calls do not discard neighbours already
  found by the candidate generator.
- Candidate pruning runs in compiled C++ over a compact column-major candidate
  matrix, then the same matrix is passed directly to CPU or CUDA candidate
  refinement.
- With `metric = "euclidean"`, `metric = "cosine"`, or `metric = "correlation"`,
  CPU Vamana uses compiled HPC-derived tables keyed by dataset shape, `k`, and
  `target_recall`. The tables select Vamana `r`, `search_l`, and `alpha`; rows
  that were only best-available rather than verified target hits record
  `tuning_benchmark_target_met = FALSE`.
- CUDA uses shape/k/target tables for the same `r`, `search_l`, and `alpha`
  parameters. CUDA Euclidean/cosine rows are measured; CUDA correlation
  currently seeds those parameters from the measured CUDA cosine table
  in `benchmark_scripts/cuda_vamana_correlation_shape_tuning_defaults_from_seeded_cosine_results.csv`
  and records `tuning_benchmark_target_met = FALSE` until the
  dedicated sweeps are rerun. CUDA cosine row-normalizes the float32 input,
  CUDA correlation row-centers then row-normalizes the float32 input before
  refinement and distance conversion.
- Euclidean, cosine, and correlation are supported for self-KNN. Cosine and
  correlation use the normalized Euclidean Vamana route; CPU uses
  metric-specific sweeps.
- Deterministic defaults expose `options(faissR.vamana_r)`,
  `options(faissR.vamana_search_l)`, `options(faissR.vamana_alpha)`, and
  `options(faissR.vamana_prune_max_work)`.

cuVS exposes Vamana/DiskANN GPU index construction and DiskANN-compatible
serialization [3], but current cuVS documentation states that Vamana search is
not exposed yet. Therefore faissR does not pretend that this route is a direct
cuVS search method; it uses faissR-owned candidate refinement while recording
the DiskANN/Vamana inspiration and cuVS build path in documentation.

## `"nsg"`

`method = "nsg"` requests a distinct faissR-owned approximate candidate-graph
algorithm derived from selected NSG/MRNG construction and pruning ideas [21].

- CPU `method = "nsg"` uses faissR's native NSG/MRNG-derived self-KNN candidate graph
  for all public metrics so public calls do not enter the unsafe linked-FAISS
  NSG graph builder [16,21,29]. Large high-dimensional CPU inputs use
  deterministic HNSW seed neighbours before derived graph pruning; smaller
  CPU inputs keep exact seed neighbours.
- CUDA `method = "nsg"` uses faissR's native CUDA NSG/MRNG-derived self-KNN route. It
  builds a candidate graph, prunes candidates with a derived NSG/MRNG rule, and
  refines rows with the native CUDA row-candidate KNN kernel.
- The derived pruning step protects the first `k` seed neighbours before
  the MRNG-derived pruning pass. This keeps the route high-recall for small `k` while
  preserving the same candidate-graph/refinement implementation.
- Candidate pruning runs in compiled C++ over a compact column-major candidate
  matrix, then the same matrix is passed directly to CPU or CUDA candidate
  refinement.
- CPU and CUDA native NSG routes support Euclidean, cosine, and correlation.
  Cosine uses normalized Euclidean refinement; correlation centers and then
  normalizes rows before refinement.
- `tuning = "auto"` chooses backend-specific candidate-graph defaults from
  `nrow(data)`, `ncol(data)`, `k`, and `metric`. CPU native NSG reads
  `options(faissR.cpu_nsg_r = ...)` and `options(faissR.cpu_nsg_graph_k = ...)`
  and allows up to 512 candidate columns; CUDA native NSG reads the matching
  `faissR.cuda_nsg_*` options and caps candidate columns at 255 for the current
  CUDA row-candidate kernel.
  Both backends use compiled HPC-derived tables keyed by dataset shape, `k`, and
  `target_recall`. The tables select `r` and `graph_k`; rows that were only
  best-available rather than verified target hits record
  `tuning_benchmark_target_met = FALSE`. Cosine uses normalized Euclidean
  refinement and correlation uses row-centered, normalized Euclidean
  refinement; each metric selects parameters from its own table.
- With `metric = "euclidean"` or `"cosine"`, CUDA `tuning = "auto"` uses
  measured CUDA shape/k/target tables for the same `r` and `graph_k`
  parameters. CUDA cosine row-normalizes the float32 input, runs native CUDA
  NSG refinement, and converts normalized Euclidean distances back to cosine
  distance. Result metadata records the CUDA benchmark source and whether the
  selected row reached the requested recall target.
- Query-vs-reference native NSG graph traversal is not exposed yet; explicit
  native CPU/CUDA NSG currently requires self-KNN.

When using NSG, check whether the backend returns the requested number of
neighbours and measure recall on a representative subset.

## `"nndescent"`

`method = "nndescent"` requests either faissR's distinct NN-descent-derived
CPU graph-refinement algorithm or direct cuVS NN-descent on CUDA [4].

- On CPU, faissR uses its native CPU NNDescent implementation by default.
- On CUDA, faissR maps to direct RAPIDS cuVS NN-descent for Euclidean/L2
  plus normalized cosine/correlation when available [3].
- Native CPU NNDescent seeds neighbours with random-projection windows plus
  deterministic row fill, stores the working graph in flat row-major buffers,
  and stores reverse neighbours in fixed-width C++ arrays during candidate
  expansion.
- With `tuning = "auto"`, CPU Euclidean, cosine, and correlation use
  HPC-derived tables keyed by dataset shape, `k`, and `target_recall`. The
  tables select `pool_size`, `n_iters`, `max_candidates`, and
  `n_random_projections` from the corresponding CPU12 metric sweep. Rows that
  were only best-available rather than verified target hits record
  `tuning_benchmark_target_met = FALSE`.
- CPU and CUDA NNDescent support cosine/correlation by row normalization
  or row centering plus normalization followed by Euclidean graph search. CPU
  cosine and correlation keep the original metric in the tuning helper so each
  can use its own target-recall table rather than Euclidean defaults.
- CUDA Euclidean NNDescent uses measured shape/k/target-recall defaults from
  the CUDA cuVS sweep. CUDA cosine uses row-normalized float32 search, and CUDA
  correlation uses row-centered, row-normalized float32 search. Both transformed
  CUDA metrics currently use seeded tables derived from the CUDA Euclidean
  sweep; those rows record `tuning_benchmark_target_met = FALSE` until the
  metric-specific HPC sweeps are rerun. The correlation seed table is
  `benchmark_scripts/cuda_nndescent_correlation_shape_tuning_defaults_from_seeded_euclidean_results.csv`.
- FAISS NNDescent is disabled by default because linked FAISS builds could
  abort the R process during graph construction. The explicit FAISS backend is
  available only behind `options(faissR.enable_faiss_nndescent = TRUE)` for
  local experiments.
- Affected cuVS builds can fail on high-dimensional FP32 L2 data with
  `cudaErrorInvalidValue` during `cuvsNNDescentBuild`. On COIL20
  (`1440 x 16384`), the cuVS L2-norm kernel required about 64 KiB of dynamic
  shared memory, above CUDA's default per-block launch limit. Rebuilding cuVS
  with an opt-in call to
  `cudaFuncSetAttribute(cudaFuncAttributeMaxDynamicSharedMemorySize)` fixed the
  cuVS route on the test machine. faissR documents this as an upstream cuVS
  issue and reports a specific error; it does not silently switch an explicit
  cuVS request to CPU or another algorithm. See
  [the upstream issue report](cuvs-nndescent-shared-memory-issue.md).

NN-descent can be fast for building approximate KNN graphs, but recall and
runtime depend strongly on graph degree, iterations, data shape, and backend.
It is best benchmarked against exact or high-recall references before being used
as a default.

## `"cagra"`

`method = "cagra"` is CUDA-only. faissR can use FAISS GPU CAGRA through the
FAISS/cuVS integration or direct RAPIDS cuVS CAGRA [3,13-16]. Use
`options(faissR.cagra_implementation = "faiss_gpu")` to force the FAISS GPU
CAGRA provider, `"cuvs"` to force direct RAPIDS cuVS CAGRA, or `"auto"` to keep
the deterministic shape-aware provider rule. Under `"auto"`, compact
high-dimensional self-KNN selects direct cuVS CAGRA when both providers are
available; other shapes keep FAISS GPU CAGRA as the default when it is
available.

- `backend = "cpu", method = "cagra"` errors.
- `backend = "cuda", method = "cagra"` requires CUDA plus FAISS GPU CAGRA or
  cuVS CAGRA.
- `metric = "cosine"` and `metric = "correlation"` use normalized Euclidean
  graph search and return `1 - similarity` distances.
  transform before graph search and converts returned L2 distances back to
- Direct cuVS CAGRA uses deterministic no-pilot defaults for
  `tuning = "auto"`; explicit `tuning = "cache"` or `"pilot"` runs recall
  tuning.
- CUDA Euclidean `tuning = "auto"` uses measured CAGRA shape/k/target rows
  from `benchmark_scripts/cuda_cagra_euclidean_shape_tuning_defaults_from_uploaded_results.csv`.
  CUDA cosine normalizes rows to float32 and CUDA correlation row-centers then
  row-normalizes to float32; both use validation-pending tables seeded from the
  measured Euclidean CAGRA sweep until corrected metric-specific tuning jobs are
  rerun. The correlation seed table is
  `benchmark_scripts/cuda_cagra_correlation_shape_tuning_defaults_from_seeded_euclidean_results.csv`.
  Result metadata records `tuning_benchmark_target_met = FALSE` for those
  seeded rows.

CAGRA is an important CUDA graph-search method. In faissR, it should be treated
as an approximate route: report its parameters and recall, especially on raw
high-dimensional data.

For cuVS CAGRA built with NN-descent,
FAISSR_CUVS_CAGRA_NN_DESCENT_NITER sets the iteration limit.
The default is 20. The native result records the resolved limit as
nn_descent_niter, including for batched and sharded builds. Measure recall
against independent exact neighbors when tuning this value.

## Tuning And Quality Reporting

Approximate methods should be reported with:

- requested `backend` and `method`;
- `attr(result, "requested_backend")`, `attr(result, "requested_method")`,
  `attr(result, "tuning")`, and `attr(result, "resolved_backend")`;
- index/search parameters stored in `attr(result, "approximation")`;
- `k`, metric, and whether self-neighbours were excluded;
- recall@k or an explicit note that quality was not evaluated.

`tuning = "auto"` is the default and uses deterministic fixed rules. FAISS GPU
IVF and cuVS CAGRA can still run pilot tuning and cache selected parameters when
called explicitly with `tuning = "cache"` or `tuning = "pilot"`. `tuning = "off"`
disables optional tuning.

Advanced tuning and cache knobs use `options(faissR.<name> = ...)`.

Exact methods mark `attr(result, "exact") = TRUE`; approximate methods mark it
as `FALSE`.

## Related Pages

- [API](usage-api.md): function arguments and examples.
- [Backends](backend-capabilities.md): backend/device availability matrix.
- [Autotuning](autotuning.md): empirical defaults and guardrails.
- [References](references.md): papers and software acknowledgements.
