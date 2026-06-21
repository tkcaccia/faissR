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
- `method` selects the algorithm family, for example `"auto"`, `"flat"`,
  `"hnsw"`, `"ivf"`, `"cagra"`, or `"grid"`.

FAISS is the required compiled vector-search dependency. CUDA, FAISS GPU,
RAPIDS cuVS, and RAPIDS libcugraph are optional compiled/runtime capabilities
[1-3,12-16]. The package does not call Python and does not silently replace an
explicit CUDA request with CPU work.

## Public Backend Policy

| Public backend | Meaning | Failure behavior |
| --- | --- | --- |
| `"auto"` | Prefer CUDA/cuVS for validated CUDA method/metric combinations when CUDA/cuVS runtime support is available; otherwise use CPU. | Falls back to CPU only because the user requested automatic device selection. |
| `"cpu"` | Use CPU/native/FAISS CPU routes. | Errors for CUDA-only methods such as `method = "cagra"`. |
| `"cuda"` | Use CUDA/FAISS GPU/cuVS routes. | Errors if CUDA/cuVS support is unavailable or if the selected method is CPU-only. |

The resolved backend is stored in `attr(result, "backend")`. Some routes also
store `attr(result, "resolved_backend")` and an `attr(result, "approximation")`
list with method-specific parameters.

## Nearest-Neighbour Method Mapping

| `method` | CPU route | CUDA route | Main use |
| --- | --- | --- | --- |
| `"auto"` | Shape-aware exact/grid/FAISS IVF/FAISS HNSW selector. | Shape-aware CUDA grid, FAISS GPU Flat/cuVS brute force, or FAISS GPU CAGRA for Euclidean; FAISS GPU Flat IP routes for cosine, correlation, and inner product when FAISS GPU Flat is available. | Default general-purpose choice. |
| `"exact"` | Native exact CPU KNN. | FAISS GPU Flat if available, otherwise direct cuVS brute force. | Exact/high-recall baseline [1-3,16]. |
| `"flat"` | FAISS Flat L2/IP; cosine and correlation use normalized Flat IP. | FAISS GPU Flat L2/IP; cosine and correlation use normalized Flat IP. | Exact FAISS exhaustive search [1-2,16]. |
| `"bruteforce"` | Native exact CPU KNN. | Direct RAPIDS cuVS brute force. | Direct exhaustive route, useful for FAISS/cuVS comparisons [3]. |
| `"grid"` | Native exact 2D/3D Euclidean grid. | Native CUDA 2D/3D Euclidean grid. | Low-dimensional spatial or simulated data. |
| `"vptree"` | Native exact CPU vantage-point tree for Euclidean, cosine, and correlation; zero-normalized non-Euclidean rows use exact CPU fallback. | Unsupported. | Low-dimensional CPU searches. |
| `"sparse"` | Native exact sparse `dgCMatrix` CPU route. | Unsupported. | Sparse matrices without densifying. |
| `"hnsw"` | FAISS CPU HNSW for all four public metrics when FAISS is available; RcppHNSW/hnswlib fallback otherwise. | Unsupported. | High-recall approximate CPU graph search [5,16]. |
| `"ivf"` | FAISS CPU IVF-Flat L2/IP; cosine and correlation use normalized IVF IP. | FAISS GPU IVF-Flat L2/IP; cosine and correlation use normalized IVF IP. | Large approximate search with coarse-list probing [1-2,16]. |
| `"ivfpq"` | FAISS CPU IVF-PQ L2/IP; cosine and correlation use normalized IVFPQ IP. | FAISS GPU IVF-PQ L2/IP; cosine and correlation use normalized IVFPQ IP. | Compressed-memory approximate search [6,16]. |
| `"nsg"` | FAISS CPU NSG for Euclidean/L2 only. | Unsupported. | Optional CPU graph-search baseline; cosine, correlation, and raw inner-product NSG remain disabled because the linked FAISS graph builder can abort during non-L2 construction [16]. |
| `"nndescent"` | Native CPU NNDescent for Euclidean/L2, cosine, and correlation. | Direct RAPIDS cuVS NN-descent for Euclidean/L2, cosine, and correlation. | Approximate KNN graph construction; cosine/correlation use normalized Euclidean search, raw inner product is not exposed, and FAISS NNDescent is disabled by default because linked FAISS builds can abort during graph construction [3-4,16]. |
| `"cagra"` | Unsupported. | FAISS GPU CAGRA preferred; direct RAPIDS cuVS CAGRA when available. Cosine/correlation use normalized Euclidean graph search. | CUDA graph-search method; raw inner-product search is not exposed [3,13-16]. |

Unsupported combinations fail before computation. For example,
`nn(x, backend = "cpu", method = "cagra")` errors because CAGRA is CUDA-only,
and `nn(x, backend = "cuda", method = "hnsw")` errors because HNSW is currently
a CPU FAISS route in faissR.

## Compiled Backend Families

| Backend family | CPU | CUDA | Notes |
| --- | --- | --- | --- |
| Native faissR dense exact | yes | no | CRAN-friendly exact CPU baseline. |
| Native faissR sparse exact | yes | no | Uses sparse `Matrix` input without densifying. |
| Native faissR grid | yes | optional CUDA | 2D/3D Euclidean self-KNN only. |
| FAISS Flat | yes | yes, if FAISS GPU is built | Exact L2 search [1-2,16]. |
| FAISS IVF-Flat | yes | yes, if FAISS GPU is built | Inverted-file approximate L2/IP search; cosine/correlation use normalized IP [1-2,16]. |
| FAISS IVF-PQ | yes | yes, if FAISS GPU is built | Product-quantized approximate L2/IP search; cosine/correlation use normalized IP [6,16]. |
| FAISS HNSW | yes, if exposed by FAISS | no | Approximate CPU graph-search index with L2/IP and normalized-IP metric transforms [5,16]. |
| FAISS NSG | yes, if exposed by FAISS | no | Optional CPU graph-search index for Euclidean/L2 only; non-L2 routes are guarded off in this linked FAISS build [16]. |
| FAISS NNDescent | experimental opt-in | no | Disabled by default because linked FAISS builds can abort during graph construction; public CPU `method = "nndescent"` uses the native implementation [4,16]. |
| FAISS GPU CAGRA/cuVS integration | no | yes, if FAISS GPU/cuVS integration is built | Uses FAISS GPU indexes backed by NVIDIA cuVS where available; cosine/correlation use normalized Euclidean search [13-15]. |
| RAPIDS cuVS brute force | no | yes, if cuVS is built | Exact direct cuVS route [3]. |
| RAPIDS cuVS CAGRA | no | yes, if cuVS is built | Direct CUDA graph-search route, guarded by pilot tuning; cosine/correlation use normalized Euclidean search [3]. |
| RAPIDS cuVS IVF/PQ | no | yes, if cuVS is built | Direct cuVS approximate Euclidean/L2 benchmark routes; use public FAISS GPU `method = "ivf"`/`"ivfpq"` for metric-aware IVF/IP/cosine/correlation search [3,6]. |
| RAPIDS cuVS NN-descent | no | yes, if cuVS is built | CUDA NN-descent route for Euclidean/L2 plus normalized cosine/correlation [3-4]. |

## Graph, Clustering, And Model Functions

| Function | CPU | CUDA | Notes |
| --- | --- | --- | --- |
| `candidate_knn()` | yes | optional CUDA candidate ranking where compiled | Exact ranking inside supplied candidates; CUDA supports Euclidean plus normalized cosine/correlation, but not raw inner product. |
| `knn_graph()` | yes | uses CUDA KNN if generated/supplied KNN uses CUDA | Returns a native `faissR_graph` edge list without requiring `igraph`. |
| `graph_cluster()` | native random-walking, Louvain, Leiden | Louvain/Leiden with RAPIDS libcugraph when built | CUDA random-walking is not enabled yet [9-12,17-19]. |
| `fast_kmeans()` | native/FAISS CPU k-means | FAISS GPU or direct cuVS k-means where available | Uses the same `"auto"`, `"cpu"`, `"cuda"` backend policy [7-8]. |
| `knn()` / `predict()` | yes | yes, through `nn()` | Supervised classifier/regressor API reuses `nn()` backend and method resolution. |

## Availability Helpers

Use these helpers to inspect the build/runtime state:

```r
backend_info()
nn_capabilities()
faiss_available()
cuda_available()
cuvs_available()
cugraph_available()
```

`backend_info()` returns a data frame with compiled/runtime availability,
public call hints, resolved/internal route labels, device/runtime hints, and
notes. The boolean helpers return a single `TRUE`/`FALSE` value. They are
useful for diagnostics and examples, but explicit backend calls still validate
availability at execution time.
`nn_capabilities()` returns a data frame with one row per public
method/backend/metric combination and marks unsupported combinations before a
benchmark tries to run them.

The capability table is design-level. Runtime auto-selection can still choose
CPU when the public CUDA design route needs a missing optional component. For
example, CUDA cosine, correlation, and inner-product auto routes require FAISS
GPU Flat; on a cuVS-only runtime, `backend = "auto"` keeps those metrics on CPU
instead of selecting an unavailable FAISS GPU index.

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
