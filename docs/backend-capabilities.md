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

- `backend = "auto"` uses CUDA when CUDA/cuVS support is available, otherwise
  CPU.
- `backend = "cpu"` forces CPU execution.
- `backend = "cuda"` forces CUDA execution and errors if no compatible CUDA
  backend is available.
- `method` selects the algorithm family, for example `"auto"`, `"flat"`,
  `"HNSW"`, `"IVF"`, `"CAGRA"`, or `"grid"`.

FAISS is the required compiled vector-search dependency. CUDA, FAISS GPU,
RAPIDS cuVS, and RAPIDS libcugraph are optional compiled/runtime capabilities
[1-3,12-16]. The package does not call Python and does not silently replace an
explicit CUDA request with CPU work.

## Public Backend Policy

| Public backend | Meaning | Failure behavior |
| --- | --- | --- |
| `"auto"` | Prefer CUDA/cuVS when available; otherwise use CPU. | Falls back to CPU only because the user requested automatic device selection. |
| `"cpu"` | Use CPU/native/FAISS CPU routes. | Errors for CUDA-only methods such as `method = "CAGRA"`. |
| `"cuda"` | Use CUDA/FAISS GPU/cuVS routes. | Errors if CUDA/cuVS support is unavailable or if the selected method is CPU-only. |

The resolved backend is stored in `attr(result, "backend")`. Some routes also
store `attr(result, "resolved_backend")` and an `attr(result, "approximation")`
list with method-specific parameters.

## Nearest-Neighbour Method Mapping

| `method` | CPU route | CUDA route | Main use |
| --- | --- | --- | --- |
| `"auto"` | Shape-aware exact/grid/FAISS IVF/FAISS HNSW selector. | Shape-aware CUDA grid, FAISS GPU Flat/cuVS brute force, or FAISS GPU CAGRA selector. | Default general-purpose choice. |
| `"exact"` | Native exact CPU KNN. | FAISS GPU Flat if available, otherwise direct cuVS brute force. | Exact/high-recall baseline [1-3,16]. |
| `"flat"` | FAISS Flat L2/IP; cosine and correlation use normalized Flat IP. | FAISS GPU Flat L2/IP; cosine and correlation use normalized Flat IP. | Exact FAISS exhaustive search [1-2,16]. |
| `"bruteforce"` | Native exact CPU KNN. | Direct RAPIDS cuVS brute force. | Direct exhaustive route, useful for FAISS/cuVS comparisons [3]. |
| `"grid"` | Native exact 2D/3D Euclidean grid. | Native CUDA 2D/3D Euclidean grid. | Low-dimensional spatial or simulated data. |
| `"vptree"` | Native exact CPU vantage-point tree for Euclidean, cosine, and correlation; zero-normalized non-Euclidean rows use exact CPU fallback. | Unsupported. | Low-dimensional CPU searches. |
| `"sparse"` | Native exact sparse `dgCMatrix` CPU route. | Unsupported. | Sparse matrices without densifying. |
| `"HNSW"` | FAISS CPU HNSW. | Unsupported. | High-recall approximate CPU graph search [5,16]. |
| `"IVF"` | FAISS CPU IVF-Flat. | FAISS GPU IVF-Flat. | Large approximate search with coarse-list probing [1-2,16]. |
| `"IVFPQ"` | FAISS CPU IVF-PQ. | FAISS GPU IVF-PQ. | Compressed-memory approximate search [6,16]. |
| `"NSG"` | FAISS CPU NSG when exposed by FAISS. | Unsupported. | Optional CPU graph-search baseline [16]. |
| `"NNDescent"` | FAISS CPU NNDescent when exposed by FAISS. | Direct RAPIDS cuVS NN-descent. | Approximate KNN graph construction [3-4,16]. |
| `"CAGRA"` | Unsupported. | FAISS GPU CAGRA preferred; direct RAPIDS cuVS CAGRA when available. | CUDA graph-search method [3,13-16]. |

Unsupported combinations fail before computation. For example,
`nn(x, backend = "cpu", method = "CAGRA")` errors because CAGRA is CUDA-only,
and `nn(x, backend = "cuda", method = "HNSW")` errors because HNSW is currently
a CPU FAISS route in faissR.

## Compiled Backend Families

| Backend family | CPU | CUDA | Notes |
| --- | --- | --- | --- |
| Native faissR dense exact | yes | no | CRAN-friendly exact CPU baseline. |
| Native faissR sparse exact | yes | no | Uses sparse `Matrix` input without densifying. |
| Native faissR grid | yes | optional CUDA | 2D/3D Euclidean self-KNN only. |
| FAISS Flat | yes | yes, if FAISS GPU is built | Exact L2 search [1-2,16]. |
| FAISS IVF-Flat | yes | yes, if FAISS GPU is built | Inverted-file approximate search [1-2,16]. |
| FAISS IVF-PQ | yes | yes, if FAISS GPU is built | Product-quantized approximate search [6,16]. |
| FAISS HNSW | yes, if exposed by FAISS | no | Approximate CPU graph-search index [5,16]. |
| FAISS NSG | yes, if exposed by FAISS | no | Optional CPU graph-search index [16]. |
| FAISS NNDescent | yes, if exposed by FAISS | no | Optional CPU NN-descent index [4,16]. |
| FAISS GPU CAGRA/cuVS integration | no | yes, if FAISS GPU/cuVS integration is built | Uses FAISS GPU indexes backed by NVIDIA cuVS where available [13-15]. |
| RAPIDS cuVS brute force | no | yes, if cuVS is built | Exact direct cuVS route [3]. |
| RAPIDS cuVS CAGRA | no | yes, if cuVS is built | Direct CUDA graph-search route, guarded by pilot tuning [3]. |
| RAPIDS cuVS IVF/PQ | no | yes, if cuVS is built | Direct cuVS approximate indexes [3,6]. |
| RAPIDS cuVS NN-descent | no | yes, if cuVS is built | CUDA NN-descent route [3-4]. |

## Graph, Clustering, And Model Functions

| Function | CPU | CUDA | Notes |
| --- | --- | --- | --- |
| `candidate_knn()` | yes | optional CUDA candidate ranking where compiled | Exact ranking inside supplied candidates; it does not generate candidates. |
| `knn_graph()` | yes | uses CUDA KNN if generated/supplied KNN uses CUDA | Returns a native `faissR_graph` edge list without requiring `igraph`. |
| `graph_cluster()` | native random-walking, Louvain, Leiden | Louvain/Leiden with RAPIDS libcugraph when built | CUDA random-walking is not enabled yet [9-12,17-19]. |
| `fast_kmeans()` | native/FAISS CPU k-means | FAISS GPU or direct cuVS k-means where available | Uses the same `"auto"`, `"cpu"`, `"cuda"` backend policy [7-8]. |
| `knn()` / `predict()` | yes | yes, through `nn()` | Supervised classifier/regressor API reuses `nn()` backend and method resolution. |

## Availability Helpers

Use these helpers to inspect the build/runtime state:

```r
backend_info()
faiss_available()
cuda_available()
cuvs_available()
cugraph_available()
```

`backend_info()` returns a data frame with compiled/runtime availability,
explicit backend labels, device/runtime hints, and notes. The boolean helpers
return a single `TRUE`/`FALSE` value. They are useful for diagnostics and
examples, but explicit backend calls still validate availability at execution
time.

## Tuning And Approximation Metadata

Approximate GPU routes can use `tuning = "auto"` to select the package's
recommended method-specific policy. FAISS GPU IVF can tune `nlist` and
`nprobe`; cuVS CAGRA can tune graph/search parameters and stop if pilot recall
does not meet the target. Approximate results record relevant parameters in
`attr(result, "approximation")`.

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
