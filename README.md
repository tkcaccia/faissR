# faissR

**Home** |
[Installation](docs/installation.md) |
[Implementation](docs/implementation.md) |
[Examples](docs/examples.md) |
[Benchmarks](docs/benchmarks.md) |
[Autotuning](docs/autotuning.md) |
[API](docs/usage-api.md) |
[NN Methods](docs/nn-methods.md) |
[Backends](docs/backend-capabilities.md) |
[References](docs/references.md)

Numbered citations in this README refer to the bibliography in
[References](docs/references.md).

`faissR` provides native nearest-neighbour search, graph construction, graph
clustering, kNN models, and k-means for R workflows that need mandatory
[FAISS](https://faiss.ai/index.html) support and optional NVIDIA CUDA/RAPIDS
acceleration [1-3,12-16]. The package is intended for CRAN-style source
installation: FAISS is required, while CUDA, RAPIDS cuVS, and RAPIDS libcugraph
are optional build features. A machine without CUDA can still install the
package from source and use the CPU/FAISS functionality.

The package code does not depend on Python or conda. Conda/mamba environments
can be useful for development or benchmarking because they provide compatible
FAISS/RAPIDS libraries, but CRAN and source builds should use normal system
headers and libraries discovered by `configure`.

## Main Features

- `nn()` for native CPU references, FAISS CPU indexes, FAISS GPU indexes, and
  optional direct RAPIDS cuVS/CUDA indexes, including an IVFPQ FastScan
  `method = "ivfpq_fastscan"` route through FAISS FastScan on CPU and cuVS 4-bit IVF-PQ
  on CUDA [1-6,13-16,22-23,34].
- Optional float32 KNN data flow: `nn()` accepts
  `float::fl()` matrices. FAISS CPU/GPU and RAPIDS cuVS NN routes consume
  float32 input through direct C++ adapters, and unsupported native routes now
  fail clearly instead of silently converting benchmark input back to R double.
  `output = "float"` returns float32 distance matrices when the optional
  `float` package is installed. The float32 FAISS routes construct returned
  float distances directly instead of materializing an intermediate R double
  matrix for CPU FAISS Flat/IVF/IVFPQ/FastScan, cached CPU FAISS fitted
  indexes, FAISS GPU Flat/IVF/IVFPQ, and direct Euclidean RAPIDS cuVS routes.
  Cosine and correlation transforms are cached as row-major float32 buffers
  inside the R session, so repeated FAISS/cuVS normalized searches can
  reuse the transformed data instead of normalizing on every call. A versioned
  C-callable entry point is also registered so downstream C++ packages can
  request the same float32 KNN result format without routing through the R
  wrappers.
- Raw `nn()` calls reuse a bounded session-local CPU
  FAISS fitted-index cache for matching Flat, HNSW, IVF, IVFPQ, and IVFPQ
  FastScan requests.
  This avoids rebuilding FAISS indexes across repeated calls; result metadata
  reports `persistent_index_cache` and `index_cache_hit`. Use
  `options(faissR.cache_fitted_nn_indexes = FALSE)` to disable the cache or
  `faissR.cache_fitted_nn_indexes_max_entries` to bound memory.
- `candidate_knn()` for exact top-k ranking inside supplied candidate rows.
- `knn_graph()` for native weighted KNN graph construction without requiring
  `igraph`.
- `graph_cluster()` for native C++/OpenMP random-walk, Louvain, and
  Leiden-style clustering [9-11], including an optional `n_clusters` target
  that searches a bounded deterministic resolution grid for Louvain/Leiden.
  CUDA Louvain and Leiden use RAPIDS libcugraph when faissR is built with
  libcugraph [12]; CUDA random-walking is not enabled yet.
- `fast_kmeans()` for CPU, FAISS CPU/GPU, and optional cuVS k-means [7-8],
  with deterministic shape-aware defaults for `max_iter`, `n_init`, and `tol`
  when `tuning = "auto"`, including no-pilot multistart tiers for cheap
  many-cluster jobs.
- `knn()` and `predict()` for kNN classification/regression, including
  immediate prediction with `knn(Xtrain, Ytrain, Xtest)`, class probabilities
  with `predict(type = "prob")`, and preserved `float::fl()`/`float32`
  training/query matrices for NN methods with direct float32 adapters. Explicit
  CPU FAISS Flat, HNSW, IVF, and IVFPQ models cache a session-local fitted
  index for matching `predict()` calls, so repeated predictions can reuse the
  indexed float32 vectors instead of rebuilding the FAISS index. IVF and IVFPQ
  predictions reuse trained centroids and inverted lists; IVFPQ additionally
  reuses trained product-quantizer codebooks and compressed codes. Prediction
  can adjust search-time `nprobe` for a requested `k` without retraining and
  sends the full `Xtest`/`newdata` matrix to the resolved NN backend in one
  batched search call, recording `batch_query`, `query_n`, and
  `query_call_count` in
  prediction metadata.
- CUDA FAISS/cuVS NN results record `attr(result, "gpu_residency")`, including
  the GPU provider, transient versus persistent index residency, host/device
  transfer strategy, whether a self-query reused the dataset device buffer, and
  whether any CPU fallback or CPU-side result repair occurred.
- `backend_info()`, `faiss_available()`, `faiss_gpu_available()`,
  `cuda_available()`, `cuvs_available()`, and `cugraph_available()` to report
  compiled/runtime backend support.
- `nn_capabilities()` to report supported nearest-neighbour
  method/backend/metric combinations for benchmark preflight checks.

Explicit GPU requests are honest: if a CUDA/cuVS/cuGraph backend is requested
and was not compiled or is not available at runtime, faissR reports an error
instead of silently running CPU code and labelling it as GPU.

For public nearest-neighbour APIs, `backend` selects the device family:
`"auto"`, `"cpu"`, or `"cuda"`. The `method` argument selects the algorithm,
for example `method = "grid"`, `method = "ivfpq_fastscan"`, or `method = "cagra"`. Thus
`nn(x, backend = "cuda", method = "grid")` uses the CUDA grid route, while
`nn(x, backend = "cpu", method = "cagra")` stops because CAGRA is CUDA-only.
`method = "ivfpq_fastscan"` is Euclidean-only and resolves to FAISS FastScan on CPU or
direct cuVS 4-bit IVF-PQ on CUDA, with no silent CPU fallback for explicit CUDA
requests.
With the default `method = "auto"`, faissR chooses the most appropriate method
for the selected backend. With `tuning = "auto"`, approximate methods use
deterministic defaults identified for the resolved method; pilot/cache tuning is
opt-in with `tuning = "cache"` or `tuning = "pilot"`. The route selector and
deterministic approximate-method tuning rules live in C++; R reads user options
and passes them to the compiled policy layer, and results report
`tuning_source = "cpp"` when those rules set method parameters.
The public nearest-neighbour metrics are `"euclidean"`, `"cosine"`,
`"correlation"`, and `"inner_product"`. Correlation is centered cosine
similarity, whereas inner product is the raw dot product; distance choices
belong in `metric`, not in separate method names. For inner-product searches,
neighbours are ranked by larger raw dot product, but returned `distances` keep
faissR's smaller-is-better convention: the best returned dot product in each
query row has distance `0`, and lower dot products have larger shifted
distances. For normalized cosine and
correlation routes, all-zero cosine rows and constant correlation rows are
handled explicitly: two zero-normalized rows have distance `0`, while a
zero-normalized row versus a nonzero row has distance `1`. CPU FAISS Flat uses
the exact CPU scorer for this degenerate case to preserve deterministic
small-`k` tie handling; explicit CUDA routes do not perform CPU repair and
therefore error clearly for these degenerate normalized rows.
The [NN methods guide](docs/nn-methods.md) describes each nearest-neighbour
method and cites the relevant algorithm/software references.

## Installation

```r
install.packages("remotes")
remotes::install_github("tkcaccia/faissR")
```

FAISS is required and is not vendored. `faissR` compiles with C++20 because
recent FAISS headers use C++20 syntax. On systems where FAISS is not visible
through `pkg-config` or standard compiler paths, set `FAISS_HOME`:

```sh
FAISS_HOME=/path/to/faiss R CMD INSTALL .
```

Optional CUDA/cuVS builds are enabled only when requested or auto-detected:

```sh
CUDA_HOME=/path/to/cuda CUVS_HOME=/path/to/cuvs \
FAISSR_USE_CUDA=1 FAISSR_USE_CUVS=1 R CMD INSTALL .
```

Optional CUDA graph clustering uses native RAPIDS libcugraph when available:

```sh
CUDA_HOME=/path/to/cuda CUGRAPH_HOME=/path/to/cugraph \
FAISSR_USE_CUDA=1 FAISSR_USE_CUGRAPH=1 R CMD INSTALL .
```

See [Installation](docs/installation.md) for CRAN/source-build details.

## FAISS GPU With cuVS

`faissR` distinguishes two GPU/cuVS routes [13-15]:

- FAISS GPU indexes with NVIDIA cuVS integration, exposed through FAISS-backed
  backends such as `faiss_gpu_ivf_flat`, `faiss_gpu_ivfpq`, and
  `faiss_gpu_cagra`. When the linked FAISS library was built with cuVS support,
  these paths report backend labels such as `GpuIndexIVFFlat_cuVS`,
  `GpuIndexIVFPQ_cuVS`, and `GpuIndexCagra_cuVS`.
- Direct RAPIDS cuVS calls, exposed through explicit backends such as
  `cuda_cuvs_cagra`, `cuda_cuvs_hnsw`, `cuda_cuvs_nndescent`,
  `cuda_cuvs_bruteforce`, `cuda_cuvs_ivf_flat`, `cuda_cuvs_ivfpq`, and
  `cuda_cuvs_ivfpq_fastscan`.
  The HNSW route builds a CUDA CAGRA seed graph and converts it with
  `cuvsHnswFromCagraWithDataset`, supports `target_recall = 0.9`, `0.95`, or `0.99`
  speed/recall tiers, and records `cuda_hnsw_design =
  "cuvs_hnsw_from_cagra_cpu_hierarchy"` because this is a cuVS wrapper route,
  not vendored CUDA code or a pure all-GPU HNSW implementation [3,22-23].

Use `backend_info()` and the attributes returned by `nn()` to confirm which
route and parameters a result used.
Use `nn_capabilities()` to inspect which public `method`, `backend`, and
`metric` combinations are supported before launching a large benchmark.

### cuVS NN-Descent Shared-Memory Note

On some RAPIDS cuVS builds, direct CUDA NN-descent can fail on
high-dimensional FP32 L2 inputs with `cudaErrorInvalidValue` during
`cuvsNNDescentBuild`. We traced this to the cuVS L2-norm kernel requesting more
than CUDA's default dynamic shared memory per block without opting into the
larger device-supported limit. A local cuVS patch that calls
`cudaFuncSetAttribute(cudaFuncAttributeMaxDynamicSharedMemorySize)` before the
kernel launch fixed the full COIL20 `1440 x 16384` case and MNIST70k on the
test machine. faissR does not vendor this cuVS patch or silently fall back to
CPU; affected users should update to a cuVS release containing the fix or build
cuVS with the patch. See the copy-ready upstream report in
[docs/cuvs-nndescent-shared-memory-issue.md](docs/cuvs-nndescent-shared-memory-issue.md).

## Quick Example

```r
library(faissR)

x <- scale(as.matrix(iris[, 1:4]))
nn_res <- nn(x, k = 15, backend = "auto", metric = "euclidean", n_threads = 4)

graph <- knn_graph(nn_res, weight = "snn")
cl <- graph_cluster(graph, method = "leiden", backend = "cpu",
                    n_clusters = 3, n_runs = 2, n_threads = 2)
table(cl$membership)
cl$selected_resolution
```

## License

`faissR` is released under the MIT license. External libraries such as FAISS,
RAPIDS cuVS, and RAPIDS cuGraph are linked as system dependencies and are not
vendored into the R package [1-3,12-16].
