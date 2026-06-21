# Nearest-Neighbour Methods

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

This page describes the `method` argument used by `nn()`, `nn_without_self()`,
and `knn()`. In faissR, `backend` chooses the device family (`"auto"`, `"cpu"`,
or `"cuda"`), while `method` chooses the nearest-neighbour algorithm family.
Distance choices belong in `metric`, not in `method`.

## Quick Selection Guide

| Goal | Suggested call |
| --- | --- |
| Let faissR choose a balanced route | `nn(x, k, backend = "auto", method = "auto")` |
| Exact CPU reference | `nn(x, k, backend = "cpu", method = "exact")` |
| Exact FAISS CPU/GPU route | `nn(x, k, backend = "cpu", method = "flat")` or `nn(x, k, backend = "cuda", method = "flat")` |
| Exact CUDA route through cuVS when available | `nn(x, k, backend = "cuda", method = "bruteforce")` |
| Large high-dimensional CPU approximate search | `nn(x, k, backend = "cpu", method = "hnsw")` |
| Large CUDA graph search | `nn(x, k, backend = "cuda", method = "cagra")` |
| Memory-pressure approximate search | `nn(x, k, backend = "cpu", method = "ivfpq")` or `nn(x, k, backend = "cuda", method = "ivfpq")` |
| 2D/3D spatial self-KNN | `nn(x, k, backend = "cpu", method = "grid")` or `nn(x, k, backend = "cuda", method = "grid")` |
| Sparse matrix input | `nn(x, k, backend = "cpu", method = "sparse")` |

Use `backend_info()` to inspect which compiled CPU, FAISS, CUDA, cuVS, and
cuGraph capabilities are available on a given machine.
Use `nn_capabilities()` to return the same method/backend/metric support matrix
as a data frame for benchmark preflight checks.

## Method Summary

| `method` | Exact? | CPU | CUDA | Main references |
| --- | --- | --- | --- | --- |
| `"auto"` | depends on selected route | yes | yes | FAISS/cuVS/HNSW/IVF/CAGRA as selected [1-6,13-16] |
| `"exact"` | yes | native CPU exact | FAISS GPU Flat or cuVS brute force | FAISS/cuVS [1-3,16] |
| `"flat"` | yes | FAISS Flat | FAISS GPU Flat | FAISS [1-2,16] |
| `"bruteforce"` | yes | native CPU exact | cuVS brute force | cuVS [3] |
| `"grid"` | yes | native 2D/3D grid | native CUDA 2D/3D grid | native faissR implementation |
| `"vptree"` | yes | native CPU VP-tree | unsupported | VP-tree/metric search [20] |
| `"sparse"` | yes | native sparse CPU | unsupported | native faissR sparse implementation |
| `"hnsw"` | approximate | FAISS HNSW for all metrics when FAISS is available; RcppHNSW/hnswlib fallback | unsupported | HNSW [5,16] |
| `"ivf"` | approximate | FAISS IVF-Flat | FAISS GPU IVF-Flat | FAISS IVF [1-2,16] |
| `"ivfpq"` | approximate | FAISS IVF-PQ | FAISS GPU IVF-PQ | product quantization [6,16] |
| `"nsg"` | approximate | FAISS NSG when exposed | unsupported | NSG/FAISS [16,21] |
| `"nndescent"` | approximate | native CPU NNDescent | cuVS NN-descent | NN-descent/cuVS [3-4,16] |
| `"cagra"` | approximate | unsupported | FAISS GPU CAGRA or cuVS CAGRA | FAISS/cuVS CAGRA [3,13-16] |

## Metric Support Matrix

faissR intentionally exposes four public metrics for nearest-neighbour search:
`"euclidean"`, `"cosine"`, `"correlation"`, and `"inner_product"`. Correlation
is not the same as inner product: correlation is centered cosine similarity,
whereas inner product is the raw dot product. The package only reports a metric
as supported for a method when that route computes neighbours under that metric
rather than silently falling back to Euclidean search.
Common aliases are accepted at the API boundary and canonicalized in result
attributes: `"l2"` maps to `"euclidean"`, `"cor"`/`"pearson"` map to
`"correlation"`, and `"ip"` maps to `"inner_product"`.

| Method | CPU metrics | CUDA metrics | Notes |
| --- | --- | --- | --- |
| `"auto"` | euclidean, cosine, correlation, inner_product | euclidean, cosine, correlation, inner_product | CUDA auto uses FAISS GPU Flat for cosine/correlation/IP and shape-aware CUDA routes for Euclidean. |
| `"exact"` | euclidean, cosine, correlation, inner_product | euclidean, cosine, correlation, inner_product | CUDA cosine/correlation/IP use FAISS GPU Flat variants when available. |
| `"flat"` | euclidean, cosine, correlation, inner_product | euclidean, cosine, correlation, inner_product | FAISS Flat L2/IP plus normalized Flat IP transforms. |
| `"bruteforce"` | euclidean, cosine, correlation, inner_product | euclidean, cosine, correlation, inner_product | CUDA Euclidean can use cuVS brute force; non-Euclidean routes use FAISS GPU Flat. |
| `"grid"` | euclidean | euclidean | 2D/3D self-KNN only. |
| `"vptree"` | euclidean, cosine, correlation | unsupported | Inner product is not a metric for VP-tree pruning. |
| `"sparse"` | euclidean, cosine, correlation, inner_product | unsupported | Exact sparse CPU route for `Matrix` inputs. |
| `"hnsw"` | euclidean, cosine, correlation, inner_product | unsupported | FAISS HNSW is used for all metrics when available; cosine/correlation use normalized inner-product search. |
| `"ivf"` | euclidean, cosine, correlation, inner_product | euclidean, cosine, correlation, inner_product | FAISS IVF-Flat supports L2/IP; cosine/correlation use normalized IVF IP. |
| `"ivfpq"` | euclidean, cosine, correlation, inner_product | euclidean, cosine, correlation, inner_product | FAISS IVFPQ supports L2/IP; cosine/correlation use normalized IVFPQ IP. |
| `"nsg"` | euclidean | unsupported | CPU FAISS NSG is Euclidean/L2-only in faissR because this linked FAISS graph builder can abort for non-L2 construction. |
| `"nndescent"` | euclidean, cosine, correlation | euclidean, cosine, correlation | Native CPU/cuVS NN-descent supports cosine/correlation by normalized Euclidean graph search; raw inner product is not exposed. FAISS NNDescent is experimental opt-in because linked FAISS builds can abort during graph construction. |
| `"cagra"` | unsupported | euclidean, cosine, correlation | CUDA-only FAISS/cuVS graph search; cosine/correlation use normalized Euclidean graph search. |

Programmatic form:

```r
nn_capabilities()
```

Benchmark scripts should treat `supported = FALSE` rows from this table as
expected skips, not algorithmic failures.

## `"auto"`

`method = "auto"` is the default. It chooses a route from the selected
`backend` and the data shape:

- `backend = "auto"` first resolves the device family: CUDA/cuVS only when the
  selected method and metric have a validated CUDA route, CPU otherwise.
- CPU auto uses exact CPU for small work, native grid for large 2D/3D
  Euclidean self-search, FAISS IVF for some million-row low-dimensional cases,
  FAISS HNSW for large high-dimensional self-search, including non-Euclidean
  HNSW when FAISS exposes it, FAISS Flat exact search for larger cosine,
  correlation, or inner-product query/exact workloads, and RcppHNSW/hnswlib as
  the large non-Euclidean self-search fallback when FAISS is unavailable
  [1-2,5,16].
- CUDA auto uses CUDA grid for large 2D/3D Euclidean self-search, exact FAISS
  GPU Flat or cuVS brute force for small/medium Euclidean searches, FAISS GPU
  CAGRA for very large Euclidean self-search when available, and FAISS GPU
  Flat IP routes for cosine, correlation, and inner-product searches
  [1-3,13-16].

`auto` is intended as a balanced default, not a guarantee of the fastest method
for every dataset. For benchmarking, report the resolved backend stored in the
result attributes.

## `"hnsw"` Metrics

CPU `method = "hnsw"` is metric-aware. When FAISS is available, faissR uses
FAISS HNSW for Euclidean/L2 and raw inner-product search. Cosine is implemented
by row L2 normalization followed by FAISS HNSW inner-product search, and
correlation is implemented by row centering plus L2 normalization followed by
FAISS HNSW inner-product search [5,16]. Inner-product HNSW normalizes returned
distances to the package convention `best_dot - dot`, so the first returned
neighbour has distance zero. If FAISS is unavailable, faissR falls back to
RcppHNSW/hnswlib for the HNSW route.

## `"exact"`

`method = "exact"` requests exhaustive exact KNN.

- On CPU, faissR uses the native exact CPU route.
- On CUDA, faissR uses FAISS GPU Flat when available, otherwise direct cuVS
  brute force when available [1-3,16].

Exact search is the best reference for recall and correctness checks. It can be
too slow or too memory-heavy for full all-pairs self-KNN on very large datasets.

## `"flat"`

`method = "flat"` requests a FAISS Flat exhaustive index [1-2,16].

- `backend = "cpu"` maps to FAISS CPU Flat.
- `backend = "cuda"` maps to FAISS GPU Flat.

Flat search is exact for L2/Euclidean search and is useful when you want FAISS
semantics specifically. On CPU and FAISS GPU Flat, `metric = "inner_product"`
uses `IndexFlatIP`; `metric = "cosine"` uses row L2 normalization followed by
Flat IP; and `metric = "correlation"` uses row centering plus L2 normalization
followed by Flat IP. The cosine and correlation routes return
`1 - similarity` distances, not shifted inner-product scores. Flat can be
faster than a generic R exact implementation because index construction, data
layout, and search are handled by FAISS.

## `"bruteforce"`

`method = "bruteforce"` requests exhaustive brute-force search.

- On CPU, it maps to the native exact CPU route.
- On CUDA, it prefers direct RAPIDS cuVS brute force [3].

This method is useful for comparing FAISS GPU Flat with direct cuVS exhaustive
search. Both are exact-style routes, but implementation details, transfer
costs, and batching can differ.

## `"grid"`

`method = "grid"` uses faissR's native spatial grid implementation.

- On CPU, it supports 2D/3D Euclidean self-KNN.
- On CUDA, it supports the CUDA 2D/3D grid route when compiled.
- It errors for higher-dimensional data.

Grid search is intended for low-dimensional spatial data or simulated 2D/3D
benchmarks. It is not a general high-dimensional ANN algorithm.

## `"vptree"`

`method = "vptree"` uses a native exact CPU vantage-point tree inspired by
metric-space nearest-neighbour search [20].

- It is CPU-only.
- It supports Euclidean directly.
- It supports cosine and correlation by normalizing rows, running the Euclidean
  VP-tree, and converting chord distances back to `1 - similarity`.
- If zero or constant rows make that normalized tree transform unsafe, faissR
  falls back to the native exact CPU scorer and records the fallback in
  `attr(result, "spatial_index")`.
- It does not support raw inner-product search because inner product is not a
  metric for VP-tree pruning.
- It is mainly useful for low-dimensional data where tree pruning reduces work.

For high-dimensional image or cytometry-style data, FAISS HNSW, IVF, or exact
GPU routes are usually more relevant.

## `"sparse"`

`method = "sparse"` uses faissR's native exact sparse CPU route for sparse
`Matrix` inputs.

- It avoids densifying `dgCMatrix` input.
- It is CPU-only.
- It is an exact route for the supported sparse metrics.

Use this method when preserving sparse representation matters more than using
FAISS/CUDA acceleration.

## `"hnsw"`

`method = "hnsw"` requests FAISS CPU HNSW, a graph-based approximate nearest
neighbour index based on Hierarchical Navigable Small World graphs [5,16].

- It is CPU-only in faissR.
- It is often a strong default for large high-dimensional CPU self-KNN.
- Tuning parameters include the graph degree `M`, construction effort, and
  search effort.

HNSW is approximate. It can give excellent recall/speed trade-offs, but recall
should be measured for new datasets when it is used for scientific conclusions.

## `"ivf"`

`method = "ivf"` requests an inverted-file Flat index [1-2,16].

- On CPU, it maps to FAISS CPU IVF-Flat.
- On CUDA, it maps to FAISS GPU IVF-Flat.
- The main parameters are the number of coarse lists (`nlist`) and searched
  lists (`nprobe`).
- `metric = "inner_product"` uses FAISS IVF-Flat with `METRIC_INNER_PRODUCT`.
- `metric = "cosine"` uses row L2 normalization followed by IVF inner-product
  search and returns `1 - similarity`.
- `metric = "correlation"` uses row centering plus L2 normalization followed by
  IVF inner-product search and returns `1 - similarity`.

IVF partitions the vector space into coarse cells and searches a subset of
cells. It is approximate unless `nprobe` approaches the number of lists. It is
useful for large datasets where exhaustive search is too expensive.

## `"ivfpq"`

`method = "ivfpq"` requests IVF with product quantization [6,16].

- On CPU, it maps to FAISS CPU IVF-PQ.
- On CUDA, it maps to FAISS GPU IVF-PQ.
- It compresses vectors using product quantization and searches compressed
  codes.
- `metric = "inner_product"` uses FAISS IVFPQ with `METRIC_INNER_PRODUCT`.
- `metric = "cosine"` and `"correlation"` use the same normalized
  inner-product transforms as IVF-Flat.

IVFPQ is a memory-pressure method. It can be fast and memory-efficient, but
recall can drop substantially. Treat it as explicit opt-in when memory matters,
not as the default accuracy-first method.

## `"nsg"`

`method = "nsg"` requests FAISS CPU NSG if the linked FAISS build exposes it.
NSG is a graph-based approximate nearest-neighbour method designed to build a
navigating graph with a sparse, search-friendly structure [16,21].

- It is CPU-only in faissR.
- Availability depends on the linked FAISS build.
- It is kept as an optional graph-search baseline.
- It supports Euclidean/L2 directly.
- Cosine, correlation, and raw inner-product NSG are not exposed because some
  linked FAISS graph builders can abort during non-L2 construction.

When using NSG, check whether the backend returns the requested number of
neighbours and measure recall on a representative subset.

## `"nndescent"`

`method = "nndescent"` requests NN-descent style approximate KNN graph
construction [4].

- On CPU, faissR uses its native CPU NNDescent implementation by default.
- On CUDA, faissR maps to direct RAPIDS cuVS NN-descent when available [3].
- Public NNDescent routes support Euclidean/L2 directly and cosine/correlation
  by row normalization followed by Euclidean graph search. Raw inner-product
  NNDescent is not exposed because dot-product search is not equivalent to this
  metric-preserving transform.
- FAISS NNDescent is disabled by default because linked FAISS builds could
  abort the R process during graph construction. The explicit FAISS backend is
  available only behind `options(faissR.enable_faiss_nndescent = TRUE)` for
  local experiments.

NN-descent can be fast for building approximate KNN graphs, but recall and
runtime depend strongly on graph degree, iterations, data shape, and backend.
It is best benchmarked against exact or high-recall references before being used
as a default.

## `"cagra"`

`method = "cagra"` is CUDA-only. faissR prefers FAISS GPU CAGRA when the linked
FAISS GPU build provides NVIDIA cuVS integration; otherwise it uses direct
RAPIDS cuVS CAGRA when available [3,13-16].

- `backend = "cpu", method = "cagra"` errors.
- `backend = "cuda", method = "cagra"` requires CUDA plus FAISS GPU CAGRA or
  cuVS CAGRA.
- `metric = "cosine"` and `metric = "correlation"` use normalized Euclidean
  graph search and return `1 - similarity` distances.
- Raw inner-product CAGRA is not exposed.
- Direct cuVS CAGRA is guarded by pilot recall tuning.

CAGRA is an important CUDA graph-search method. In faissR, it should be treated
as an approximate route: report its parameters and recall, especially on raw
high-dimensional data.

## Tuning And Quality Reporting

Approximate methods should be reported with:

- requested `backend` and `method`;
- resolved backend attributes;
- index/search parameters stored in `attr(result, "approximation")`;
- `k`, metric, and whether self-neighbours were excluded;
- recall@k or an explicit note that quality was not evaluated.

`tuning = "auto"` is the default. FAISS GPU IVF and cuVS CAGRA can run pilot
tuning and cache selected parameters. `tuning = "off"` disables this behavior
when deterministic fixed-parameter runs are required.

Advanced tuning and cache knobs use `options(faissR.<name> = ...)`. Legacy
`fastEmbedR.<name>` options remain compatibility fallbacks; if both are set,
`faissR.*` wins.

Exact methods mark `attr(result, "exact") = TRUE`; approximate methods mark it
as `FALSE`.

## Related Pages

- [API](usage-api.md): function arguments and examples.
- [Backends](backend-capabilities.md): backend/device availability matrix.
- [Autotuning](autotuning.md): empirical defaults and guardrails.
- [References](references.md): papers and software acknowledgements.
