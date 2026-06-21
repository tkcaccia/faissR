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
| Large high-dimensional CPU approximate search | `nn(x, k, backend = "cpu", method = "HNSW")` |
| Large CUDA graph search | `nn(x, k, backend = "cuda", method = "CAGRA")` |
| Memory-pressure approximate search | `nn(x, k, backend = "cpu", method = "IVFPQ")` or `nn(x, k, backend = "cuda", method = "IVFPQ")` |
| 2D/3D spatial self-KNN | `nn(x, k, backend = "cpu", method = "grid")` or `nn(x, k, backend = "cuda", method = "grid")` |
| Sparse matrix input | `nn(x, k, backend = "cpu", method = "sparse")` |

Use `backend_info()` to inspect which compiled CPU, FAISS, CUDA, cuVS, and
cuGraph capabilities are available on a given machine.

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
| `"HNSW"` | approximate | FAISS HNSW for Euclidean; RcppHNSW/hnswlib for cosine, correlation, and inner product | unsupported | HNSW [5,16] |
| `"IVF"` | approximate | FAISS IVF-Flat | FAISS GPU IVF-Flat | FAISS IVF [1-2,16] |
| `"IVFPQ"` | approximate | FAISS IVF-PQ | FAISS GPU IVF-PQ | product quantization [6,16] |
| `"NSG"` | approximate | FAISS NSG when exposed | unsupported | NSG/FAISS [16,21] |
| `"NNDescent"` | approximate | FAISS NNDescent when exposed | cuVS NN-descent | NN-descent/cuVS [3-4,16] |
| `"CAGRA"` | approximate | unsupported | FAISS GPU CAGRA or cuVS CAGRA | FAISS/cuVS CAGRA [3,13-16] |

## `"auto"`

`method = "auto"` is the default. It chooses a route from the selected
`backend` and the data shape:

- `backend = "auto"` first resolves the device family: CUDA/cuVS when available
  for validated CUDA metrics, CPU otherwise.
- CPU auto uses exact CPU for small work, native grid for large 2D/3D
  Euclidean self-search, FAISS IVF for some million-row low-dimensional cases,
  FAISS HNSW for large high-dimensional Euclidean self-search when FAISS
  exposes it, FAISS Flat exact search for larger cosine, correlation, or
  inner-product query/exact workloads, and RcppHNSW/hnswlib for large
  non-Euclidean self-search when available [1-2,5,16].
- CUDA auto uses CUDA grid for large 2D/3D Euclidean self-search, exact FAISS
  GPU Flat or cuVS brute force for small/medium searches, and FAISS GPU CAGRA
  for very large self-search when available [1-3,13-16].

`auto` is intended as a balanced default, not a guarantee of the fastest method
for every dataset. For benchmarking, report the resolved backend stored in the
result attributes.

## `"HNSW"` Metrics

CPU `method = "HNSW"` is metric-aware. For Euclidean/L2 search, faissR uses the
FAISS HNSW implementation when FAISS is available. For cosine, correlation, and
inner-product search, faissR routes to RcppHNSW/hnswlib because hnswlib exposes
cosine and IP spaces directly [5]. Correlation is implemented as cosine search
after row centering and L2 normalization. Inner-product HNSW uses hnswlib's
`ip` space and faissR normalizes returned distances to the package convention
`best_dot - dot`, so the first returned neighbour has distance zero.

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

## `"HNSW"`

`method = "HNSW"` requests FAISS CPU HNSW, a graph-based approximate nearest
neighbour index based on Hierarchical Navigable Small World graphs [5,16].

- It is CPU-only in faissR.
- It is often a strong default for large high-dimensional CPU self-KNN.
- Tuning parameters include the graph degree `M`, construction effort, and
  search effort.

HNSW is approximate. It can give excellent recall/speed trade-offs, but recall
should be measured for new datasets when it is used for scientific conclusions.

## `"IVF"`

`method = "IVF"` requests an inverted-file Flat index [1-2,16].

- On CPU, it maps to FAISS CPU IVF-Flat.
- On CUDA, it maps to FAISS GPU IVF-Flat.
- The main parameters are the number of coarse lists (`nlist`) and searched
  lists (`nprobe`).

IVF partitions the vector space into coarse cells and searches a subset of
cells. It is approximate unless `nprobe` approaches the number of lists. It is
useful for large datasets where exhaustive search is too expensive.

## `"IVFPQ"`

`method = "IVFPQ"` requests IVF with product quantization [6,16].

- On CPU, it maps to FAISS CPU IVF-PQ.
- On CUDA, it maps to FAISS GPU IVF-PQ.
- It compresses vectors using product quantization and searches compressed
  codes.

IVFPQ is a memory-pressure method. It can be fast and memory-efficient, but
recall can drop substantially. Treat it as explicit opt-in when memory matters,
not as the default accuracy-first method.

## `"NSG"`

`method = "NSG"` requests FAISS CPU NSG if the linked FAISS build exposes it.
NSG is a graph-based approximate nearest-neighbour method designed to build a
navigating graph with a sparse, search-friendly structure [16,21].

- It is CPU-only in faissR.
- Availability depends on the linked FAISS build.
- It is kept as an optional graph-search baseline.

When using NSG, check whether the backend returns the requested number of
neighbours and measure recall on a representative subset.

## `"NNDescent"`

`method = "NNDescent"` requests NN-descent style approximate KNN graph
construction [4].

- On CPU, faissR uses FAISS NNDescent when exposed by the linked FAISS build
  [16].
- On CUDA, faissR maps to direct RAPIDS cuVS NN-descent when available [3].

NN-descent can be fast for building approximate KNN graphs, but recall and
runtime depend strongly on graph degree, iterations, data shape, and backend.
It is best benchmarked against exact or high-recall references before being used
as a default.

## `"CAGRA"`

`method = "CAGRA"` is CUDA-only. faissR prefers FAISS GPU CAGRA when the linked
FAISS GPU build provides NVIDIA cuVS integration; otherwise it uses direct
RAPIDS cuVS CAGRA when available [3,13-16].

- `backend = "cpu", method = "CAGRA"` errors.
- `backend = "cuda", method = "CAGRA"` requires CUDA plus FAISS GPU CAGRA or
  cuVS CAGRA.
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

Exact methods mark `attr(result, "exact") = TRUE`; approximate methods mark it
as `FALSE`.

## Related Pages

- [API](usage-api.md): function arguments and examples.
- [Backends](backend-capabilities.md): backend/device availability matrix.
- [Autotuning](autotuning.md): empirical defaults and guardrails.
- [References](references.md): papers and software acknowledgements.
