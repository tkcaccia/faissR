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
installation: FAISS is required for all builds, while CUDA, RAPIDS cuVS, and
RAPIDS libcugraph are optional for CPU-only builds. A machine without CUDA can
still install the package from source and use the CPU/FAISS functionality. For
NVIDIA GPU users, the GPU stack should be requested explicitly so missing CUDA
or RAPIDS libraries are fatal rather than silently producing a CPU-only build.

The package code does not depend on Python or conda. Conda/mamba environments
can be useful for development or benchmarking because they provide compatible
FAISS/RAPIDS libraries, but CRAN and source builds should use normal system
headers and libraries discovered by `configure`.

## Main Features

- `nn()` for native CPU references, FAISS CPU indexes, FAISS GPU indexes, and
  optional direct RAPIDS cuVS/CUDA indexes, including an IVFPQ FastScan
  `method = "ivfpq_fastscan"` route through FAISS FastScan on CPU and cuVS
  4-bit IVF-PQ on CUDA. Repeated raw CUDA FastScan calls reuse a fitted cuVS
  IVF-PQ index and cuVS resources in a bounded session cache [1-6,13-16,22-23,34].
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
- GPU-resident exact KNN output for downstream CUDA consumers:
  `nn_gpu()` returns a `faissR_gpu_knn` object whose `indices_ptr` and
  `distances_ptr` remain on the CUDA device as int32 and float32 buffers.
  This is separate from `nn(..., output = "float")`, which still returns an R
  object on the host. `gpu_knn_to_host()` is the explicit diagnostic helper for
  copying a GPU-resident result back to ordinary R matrices. The registered
  C-callable `faissR_nn_cuda_tuned_gpu_call` exposes the same self-KNN route to
  downstream C/C++ packages. The first GPU-resident route is exact CUDA search
  for `method = "auto"`, `"exact"`, `"flat"`, or `"bruteforce"` across
  Euclidean, cosine, correlation, and raw inner-product metrics; approximate
  FAISS GPU/cuVS provider routes still return host objects until their provider
  result buffers are made persistent.
- CUDA exact metric tuning:
  `method = "exact", backend = "cuda"` uses FAISS GPU Flat query-batch/resource
  policies for Euclidean, cosine, correlation, and raw inner product when
  `tuning = "auto"`. The correlation table is measured in
  `benchmark_scripts/cuda_exact_correlation_shape_tuning_defaults_from_uploaded_results.csv`.
  The inner-product table is currently
  `benchmark_scripts/cuda_exact_inner_product_shape_tuning_defaults_from_seeded_euclidean_results.csv`,
  seeded from the measured CUDA exact Euclidean sweep and marked
  validation-pending until `run_hpc_exact_tuning_cuda_inner_product.sh` replaces
  it with measured IP rows. Exact recall is recorded by construction, and
  benchmark provenance remains visible through `tuning_benchmark_target_met`.
- CUDA Flat metric tuning:
  `method = "flat", backend = "cuda"` uses FAISS GPU Flat query-batch/resource
  policies for Euclidean, cosine, correlation, and raw inner product when
  `tuning = "auto"`. The correlation table is measured in
  `benchmark_scripts/cuda_flat_correlation_shape_tuning_defaults_from_uploaded_results.csv`.
  The inner-product table is currently
  `benchmark_scripts/cuda_flat_inner_product_shape_tuning_defaults_from_seeded_euclidean_results.csv`,
  seeded from the measured CUDA Flat Euclidean sweep and marked
  validation-pending until `run_hpc_flat_tuning_cuda_inner_product.sh` replaces
  it with measured IP rows. The selected row is stored in
  `attr(result, "flat_tuning")`.
- CUDA bruteforce metric tuning:
  `method = "bruteforce", backend = "cuda"` uses direct cuVS brute-force search
  for Euclidean and transformed exact routes for cosine, correlation, and raw
  inner product. Its `tuning = "auto"` tables store cuVS query-batch/resource
  defaults by shape, `k`, and target recall. Correlation uses
  `benchmark_scripts/cuda_bruteforce_correlation_shape_tuning_defaults_from_proxy_results.csv`;
  raw inner product currently uses
  `benchmark_scripts/cuda_bruteforce_inner_product_shape_tuning_defaults_from_seeded_euclidean_results.csv`,
  seeded from measured CUDA bruteforce Euclidean rows until
  `run_hpc_bruteforce_tuning_cuda_inner_product.sh` replaces it with measured
  IP rows.
- CUDA HNSW metric tuning:
  `method = "hnsw", backend = "cuda"` uses RAPIDS cuVS HNSW from a CAGRA seed
  graph. Correlation is searched on centered and row-normalized float32 data;
  raw inner product is searched through the same maximum-inner-product-to-L2
  transform used by the CUDA exact/brute-force routes. `tuning = "auto"` uses
  measured Euclidean/cosine/correlation shape/k/target tables. Raw inner
  product currently uses
  `benchmark_scripts/cuda_hnsw_inner_product_shape_tuning_defaults_from_seeded_euclidean_results.csv`,
  seeded from the measured CUDA HNSW Euclidean table and marked
  validation-pending until `run_hpc_hnsw_tuning_cuda_inner_product.sh`
  replaces it with measured IP rows.
- CUDA IVF metric tuning:
  `method = "ivf", backend = "cuda"` resolves to FAISS GPU IVF-Flat L2/IP.
  Euclidean, cosine, and correlation use measured CUDA IVF shape/k/target
  `nlist`/`nprobe` tables; raw inner product currently uses
  `benchmark_scripts/cuda_ivf_inner_product_shape_tuning_defaults_from_seeded_euclidean_results.csv`,
  seeded from the measured CUDA IVF Euclidean table and marked
  validation-pending until `run_hpc_ivf_tuning_cuda_inner_product.sh` replaces
  it with measured IP rows.
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
- CUDA nearest-neighbour routes return backend-shaped KNN matrices: self-neighbour
  removal, row ordering, and final output layout are handled in C++/CUDA rather
  than by R-side cleanup. CUDA graph routes that do not yet expose compiled
  include-self shaping fail clearly instead of repairing the output in R.
- `backend_info()`, `faiss_available()`, `faiss_gpu_available()`,
  `cuda_available()`, `cuvs_available()`, and `cugraph_available()` to report
  compiled/runtime backend support.
- `nn_capabilities()` to report supported nearest-neighbour
  method/backend/metric combinations for benchmark preflight checks.
- Benchmark #1 comparison launchers for Euclidean speed tests are split into
  CPU and CUDA runs:
  `benchmark_scripts/run_benchmark1_compare_cpu_euclidean.sh` compares faissR
  CPU methods with CPU external R KNN packages, while
  `benchmark_scripts/run_benchmark1_compare_cuda_euclidean.sh` compares faissR
  CUDA/FAISS-GPU/cuVS methods with CUDA-capable external packages such as
  `cuda.ml` when available. HPC/SLURM equivalents are
  `benchmark_scripts/run_hpc_benchmark1_compare_cpu12_euclidean.sh` and
  `benchmark_scripts/run_hpc_benchmark1_compare_cuda_euclidean.sh`; both force
  Euclidean distance and write `benchmark1_faissr_vs_external_speed.csv`.

## Available Functions

The table below summarizes the public R functions shown on GitHub. Full
argument-level details are in the [API reference](docs/usage-api.md), and the
method/backend/metric matrix is in the
[backend-capabilities page](docs/backend-capabilities.md).

| Function | Main use | Backends | Return |
| --- | --- | --- | --- |
| `nn()` | General nearest-neighbour search over reference/query matrices. Supports `backend = "auto"`, `"cpu"`, or `"cuda"`; `method = "auto"`, `"exact"`, `"flat"`, `"bruteforce"`, `"grid"`, `"hnsw"`, `"ivf"`, `"ivfpq"`, `"ivfpq_fastscan"`, `"nndescent"`, `"nsg"`, `"vamana"`, or `"cagra"`; and `metric = "euclidean"`, `"cosine"`, `"correlation"`, or `"inner_product"`. `exclude_self = TRUE` removes self-neighbours in compiled code for self-KNN. `tuning = "auto"` uses C++ shape/k/target-recall policies. | CPU, FAISS CPU, FAISS GPU, native CUDA, direct RAPIDS cuVS where compiled. | A `faissR_nn` list with 1-based `indices`, `distances`, `index_base`, `distance_type`, `metric`, `backend_used`, and route/tuning metadata. `output = "float"` can return float32 distances when the optional `float` package is available. |
| `nn_gpu()` | GPU-resident exact-family KNN for downstream CUDA packages. It is narrower than `nn()` and is intended when another package needs device pointers instead of R matrices. | CUDA exact-family routes for `method = "auto"`, `"exact"`, `"flat"`, or `"bruteforce"`. Euclidean and raw inner product use FAISS GPU direct `bfKnn` when available; cosine/correlation use native CUDA exact transforms. | A `faissR_gpu_knn` object with an owning `handle`, CUDA-device `indices_ptr` and `distances_ptr`, `result_residency = "cuda"`, `indices_type = "int32"`, `distance_type = "float32"`, `device_to_host_result_copies = 0`, plus exact-family `execution_tuning` and, for `method = "auto"`, the compiled-policy `auto_preferred_tuning` when applicable. |
| `gpu_knn_to_host()` | Explicit diagnostic conversion of a GPU-resident KNN result to ordinary R matrices. It is never called automatically by `nn_gpu()`. | Uses the CUDA result handle returned by `nn_gpu()` or the C-callable GPU API. | A host-side `faissR_nn` list with copied integer indices and numeric distances. |
| `candidate_knn()` | Exact top-k reranking inside a user-supplied candidate-neighbour matrix. This is useful when another algorithm proposes candidates and faissR should compute the final ordered neighbours. | CPU compiled scorer with the public metrics. | A `faissR_nn` list restricted to the supplied candidates. |
| `knn_graph()` | Build weighted nearest-neighbour graphs from data, an embedding, or an existing `faissR_nn` result. It can compute KNN internally through `nn(..., exclude_self = TRUE)`. | CPU and CUDA KNN backends through `nn()`, then native graph construction. | A `faissR_graph` object with edge indices, weights, graph parameters, and KNN metadata. |
| `graph_cluster()` | Cluster a graph or data-derived KNN graph with random-walking, Louvain, or Leiden-style algorithms. `n_clusters` is an alternative target to `resolution` for Louvain/Leiden. | Native CPU/OpenMP clustering; optional RAPIDS libcugraph CUDA Louvain/Leiden when compiled. | A `faissR_graph_cluster` object with `membership`, modularity/quality fields, method/backend metadata, and graph-building metadata when applicable. |
| `fast_kmeans()` | Fast k-means-style clustering with CPU, FAISS, FAISS GPU, or cuVS routes. `tuning = "auto"` selects deterministic shape-aware defaults for iteration count, starts, and tolerances. | CPU/statistics, FAISS CPU/GPU, direct cuVS where compiled. | A `faissR_kmeans` object with cluster assignments, centers, within-cluster summaries, backend/tuning metadata, and convergence diagnostics. |
| `knn()` | Fit a reusable kNN classifier/regressor, or fit and predict immediately with `knn(Xtrain, Ytrain, Xtest)`. It reuses `nn()` for neighbour search and can preserve float32 training/query data for supported routes. | Same device and method family as `nn()` for the selected training/prediction route. | Without `Xtest`, a `faissR_knn_model`; with `Xtest`, predictions or probabilities depending on `type`. |
| `predict()` | S3 method for `faissR_knn_model` objects. It predicts labels, numeric responses, or class probabilities with `type = "response"` or `"prob"`. Prediction sends the full query matrix in one batched NN call. | Same fitted route where compatible; otherwise rebuilds the same requested route rather than silently switching algorithms. | A vector/data frame of predictions or a probability matrix, with the underlying `nn()` metadata attached. |
| `backend_info()` | Inspect compiled/runtime backend support and implementation notes. | All compiled backend families. | A data frame of public backend names, implementation labels, runtime availability, and notes. |
| `nn_capabilities()` | Preflight check for supported public `method`/`backend`/`metric` combinations. `runtime = TRUE` adds current-machine availability information. | CPU, CUDA, FAISS, cuVS, and cuGraph capability checks. | A data frame suitable for benchmark filtering before a large run. |
| `faiss_available()` | Check whether faissR was compiled and linked against FAISS. | FAISS CPU. | A single logical value. |
| `faiss_gpu_available()` | Check whether the linked FAISS build reports GPU support. | FAISS GPU. | A single logical value. |
| `cuda_available()` | Check whether native CUDA support was compiled and a CUDA device/runtime is available. | Native CUDA. | A single logical value. |
| `cuvs_available()` | Check whether direct RAPIDS cuVS support was compiled and can be loaded. | RAPIDS cuVS. | A single logical value. |
| `cugraph_available()` | Check whether RAPIDS libcugraph graph clustering support was compiled and can be loaded. | RAPIDS libcugraph. | A single logical value. |

### C/C++ Callable Entry Points

faissR also registers stable C-callable entry points with
`R_RegisterCCallable()`. These are for downstream R packages with C/C++ code and
are retrieved with `R_GetCCallable("faissR", "<name>")`.

| C-callable name | ABI | Purpose |
| --- | --- | --- |
| `faissR_nn_float32_call` | `(x, k, backend, metric, include_self, n_threads)` | CPU FAISS Flat float32 KNN route. It accepts ordinary R double matrices or optional `float::fl()`/float32 matrices and returns the stable host `faissR_nn` list with double distances. |
| `faissR_nn_float32_call_output` | `(x, k, backend, metric, include_self, n_threads, distances)` | Same CPU FAISS Flat float32 route, with `distances = "double"` or `"float"` to request host distance storage type. |
| `faissR_nn_cuda_tuned_gpu_call` | `(x, k, method, metric, include_self, target_recall)` | CUDA self-KNN route that keeps result buffers on the GPU for `method = "auto"`, `"exact"`, `"flat"`, or `"bruteforce"`. It returns the same `faissR_gpu_knn` object shape as `nn_gpu()`, including CUDA device pointers and `device_to_host_result_copies = 0`. |

Explicit GPU requests are honest: if a CUDA/cuVS/cuGraph backend is requested
and was not compiled or is not available at runtime, faissR reports an error
instead of silently running CPU code and labelling it as GPU.

For public nearest-neighbour APIs, `backend` selects the device family:
`"auto"`, `"cpu"`, or `"cuda"`. The `method` argument selects the algorithm,
for example `method = "grid"`, `method = "ivfpq_fastscan"`, or `method = "cagra"`. Thus
`nn(x, backend = "cuda", method = "grid")` uses the CUDA grid route, while
`nn(x, backend = "cpu", method = "cagra")` stops because CAGRA is CUDA-only.
`method = "ivfpq_fastscan"` resolves to FAISS FastScan on CPU and direct cuVS
4-bit IVF-PQ on CUDA. CPU supports Euclidean, cosine, correlation, and raw
inner product; CUDA supports Euclidean, cosine, and correlation, with cosine
handled by row-normalized float32 L2 search and correlation by row-centering
plus row-normalized float32 L2 search before distance conversion. Explicit CUDA
requests never silently fall back to CPU.
With the default `method = "auto"`, faissR chooses the most appropriate method
for the selected backend. With `tuning = "auto"`, approximate methods use
deterministic defaults identified for the resolved method; pilot/cache tuning is
opt-in with `tuning = "cache"` or `tuning = "pilot"`. The route selector and
deterministic approximate-method tuning rules live in C++; R reads user options
and passes them to the compiled policy layer, and results report
`tuning_source = "cpp"` when those rules set method parameters.
For CUDA Euclidean self-KNN, the current benchmark-derived auto policy uses IVF
for large low-dimensional data, exact FAISS GPU Flat/brute force for the
measured small, medium, and high-dimensional accuracy-first shapes, and IVF for
very large high-dimensional data only at lower target-recall tiers. CPU
Euclidean auto uses exact search for tiny data, FAISS HNSW for most medium and
high-dimensional self-KNN, and FAISS IVF for selected large low-dimensional
datasets where the tuning sweep showed better speed at the requested recall.
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
The [Autotuning guide](docs/autotuning.md) explains how the HPC target-recall
sweeps convert speed, recall, failure, and shape-summary tables into
deterministic C++ defaults for each method and backend.

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

Automated Bioconductor/r-universe Windows binary builds are skipped with
`OS_type: unix` because FAISS is mandatory and those builders do not provide a
compatible FAISS development library. Automated Bioconductor macOS binary builds
are also marked unsupported until the Bioconductor/r-universe macOS
system-library bundle provides FAISS. User macOS source installs remain
supported with Homebrew or an active conda/mamba environment. Windows users
should use WSL2 for the supported Linux path, or provide a native
Rtools-compatible FAISS build and install from source manually.

On macOS with Homebrew, install FAISS and the OpenMP runtime first:

```sh
brew install faiss libomp
```

or explicitly allow the GitHub install to call Homebrew for you:

```r
Sys.setenv(FAISSR_AUTO_INSTALL_FAISS = "1")
remotes::install_github("tkcaccia/faissR")
```

The automatic Homebrew step is opt-in for ordinary user installs. On ordinary
macOS GitHub Actions workers, `configure` may run `brew install faiss libomp`
automatically because FAISS is mandatory. Bioconductor/r-universe macOS binary
workers deliberately remove Homebrew and do not currently provide FAISS, so
those builds are marked unsupported rather than using a hidden dependency
manager. Set `FAISSR_AUTO_INSTALL_FAISS=0` to suppress the Homebrew convenience
path.

If Homebrew is not available on a user macOS machine, an already-active
conda/mamba environment is also supported:

```sh
conda install -c conda-forge faiss-cpu libomp
export FAISS_HOME="$CONDA_PREFIX"
export LIBOMP_HOME="$CONDA_PREFIX"
R CMD INSTALL .
```

`configure` detects `CONDA_PREFIX` passively when FAISS and libomp are already
installed there. It does not install conda automatically, which keeps
Bioconductor and shared-machine builds explicit.

Optional CUDA/cuVS builds are enabled only when requested or auto-detected:

```sh
CUDA_HOME=/path/to/cuda CUVS_HOME=/path/to/cuvs \
FAISSR_REQUIRE_CUDA=1 FAISSR_REQUIRE_CUVS=1 R CMD INSTALL .
```

Optional CUDA graph clustering uses native RAPIDS libcugraph when available:

```sh
CUDA_HOME=/path/to/cuda CUGRAPH_HOME=/path/to/cugraph \
FAISSR_REQUIRE_CUDA=1 FAISSR_REQUIRE_CUGRAPH=1 R CMD INSTALL .
```

See [Installation](docs/installation.md) for CRAN/source-build details.

## Bioconductor Readiness

`faissR` includes the `GPU` `biocViews` term, a `BiocStyle` vignette, `NEWS.md`,
a standard `License: MIT + file LICENSE` declaration, and a top-level
`.BBSoptions` file with `GPU_reliance: optional`. This opts the package into
Bioconductor GPU builders without making NVIDIA libraries mandatory for the
regular CPU/FAISS build. Local submission checks should be run from a source
tarball:

```sh
R CMD build .
R CMD check faissR_0.99.12.tar.gz
```

and then:

```r
BiocCheck::BiocCheckGitClone(".")
BiocCheck::BiocCheck("faissR_0.99.12.tar.gz", `new-package` = TRUE)
```

FAISS is a required external system dependency. CUDA, cuVS, and libcugraph are
optional for CPU-only Bioconductor builds and must not be required there.
NVIDIA GPU builds should use `FAISSR_REQUIRE_CUDA=1` and, as needed,
`FAISSR_REQUIRE_CUVS=1` or `FAISSR_REQUIRE_CUGRAPH=1` so missing GPU libraries
fail during configuration. Maintainer Support Site registration and bioc-devel
subscription are external submission steps.

On Debian/Ubuntu CPU builders, FAISS should be supplied by the FAISS
development package, typically `libfaiss-dev`. If a BiocStaging/r-universe log
installs NVIDIA CUDA packages but not `libfaiss-dev`, the failing step is the
automated system-requirements resolution: FAISS is mandatory for faissR, while
CUDA/RAPIDS libraries are optional unless a GPU build is explicitly requested.
Until the upstream r-universe resolver includes FAISS, the repository-level
`.prepare` hook installs `libfaiss-dev` for r-universe source builds and is
excluded from the package tarball.

For macOS r-universe/BiocStaging binary builds, FAISS is not currently available
in the worker system-library bundle and Homebrew is deliberately removed before
package installation. Those automated macOS binary builds are therefore marked
unsupported for real FAISS execution until FAISS is provided by the builder.
Because the r-universe workflow may still launch the macOS binary job, the
configure script builds diagnostic stubs only for that worker when FAISS is
absent. `backend_info()` then reports FAISS as unavailable with reason
`runiverse_macos_diagnostic_stub_no_faiss`. Linux builds and ordinary user
macOS source installs still require real FAISS.

For r-universe/WebAssembly, `configure` detects the
`wasm32-unknown-emscripten` target and builds diagnostic stubs rather than
using host `/usr/include` FAISS headers inside the Emscripten sysroot. The WASM
artifact can report backend availability, but FAISS/CUDA/cuVS methods are not
available because those native libraries are not webR system libraries.
Supported Linux and macOS builds still require real FAISS.

## FAISS GPU With cuVS

`faissR` distinguishes two GPU/cuVS routes [13-15]:

- FAISS GPU indexes with NVIDIA cuVS integration, exposed through FAISS-backed
  backends such as `faiss_gpu_ivf_flat`, `faiss_gpu_ivfpq`, and
  `faiss_gpu_cagra`. When the linked FAISS library was built with cuVS support,
  these paths report backend labels such as `GpuIndexIVFFlat_cuVS`,
  `GpuIndexIVFPQ_cuVS`, and `GpuIndexCagra_cuVS`.
  CUDA IVF-Flat auto tuning now selects `nlist`/`nprobe` from a compiled
  shape/k/target-recall policy derived from the float32 CUDA IVF HPC sweeps
  for Euclidean, cosine, and correlation metrics. Raw inner product uses the
  same compiled selector with a validation-pending seed table until the
  dedicated IP sweep replaces it.
  CUDA IVFPQ auto tuning likewise uses compiled `nlist`/`nprobe`/`pq_m`/`pq_nbits`
  policies for Euclidean, correlation, and raw-inner-product FAISS GPU IVF-PQ
  searches. Raw inner product uses
  `benchmark_scripts/cuda_ivfpq_inner_product_shape_tuning_defaults_from_seeded_euclidean_results.csv`,
  seeded from measured CUDA Euclidean IVFPQ rows until the dedicated IP sweep
  replaces it; seeded or below-target rows record
  `tuning_benchmark_target_met = FALSE`.
- Direct RAPIDS cuVS calls selected through the public
  `backend = "cuda"` plus `method = ...` API. Concrete labels such as
  `cuda_cuvs_cagra`, `cuda_cuvs_hnsw`, `cuda_cuvs_nndescent`,
  `cuda_cuvs_bruteforce`, `cuda_cuvs_ivf_flat`, `cuda_cuvs_ivfpq`, and
  `cuda_cuvs_ivfpq_fastscan` are recorded as resolved backend metadata for
  diagnostics and benchmarks; they are not hidden public backend or method
  options.
  CUDA CAGRA supports cosine by row-normalizing float32 input, correlation by
  row-centering plus row-normalizing float32 input, and raw inner product by a
  maximum-inner-product-to-L2 extra-dimension transform before CAGRA search,
  then converting distances back to the public metric convention. Euclidean
  CAGRA `tuning = "auto"` uses measured shape/k/target rows from
  `benchmark_scripts/cuda_cagra_euclidean_shape_tuning_defaults_from_uploaded_results.csv`;
  cosine, correlation, and raw inner product currently use validation-pending
  tables seeded from those Euclidean rows and record
  `tuning_benchmark_target_met = FALSE` until the corrected metric-specific
  tuning sweeps are rerun. The raw inner-product seed table is
  `benchmark_scripts/cuda_cagra_inner_product_shape_tuning_defaults_from_seeded_euclidean_results.csv`.
  `cuda_cuvs_ivfpq_fastscan` keeps the trained cuVS IVF-PQ index, compressed
  codes, dataset device buffer, and cuVS resources in a bounded session-local
  cache for compatible repeated `nn()` calls. Self-query searches reuse the
  fitted dataset device buffer directly, and repeated searches with the same
  separate query matrix can reuse one cached query device buffer via
  `options(faissR.cache_cuda_ivfpq_query_buffers = TRUE)`.
  The CUDA HPC FastScan wrapper tunes `nlist`, `nprobe`, and byte-aligned 4-bit
  `pq_dim` through `IVFPQ_FASTSCAN_NLIST_MULTS`,
  `IVFPQ_FASTSCAN_NPROBE_MULTS`, and `IVFPQ_FASTSCAN_PQ_DIMS`.
  CUDA cosine, correlation, and raw-inner-product `tuning = "auto"` currently use policies seeded
  from the CUDA Euclidean FastScan sweep and mark
  `tuning_benchmark_target_met = FALSE` until the corrected metric-specific
  sweeps are rerun. The seeded correlation table is
  `benchmark_scripts/cuda_ivfpq_fastscan_correlation_shape_tuning_defaults_from_seeded_euclidean_results.csv`
  and the seeded raw-inner-product table is
  `benchmark_scripts/cuda_ivfpq_fastscan_inner_product_shape_tuning_defaults_from_seeded_euclidean_results.csv`.
  Raw inner product applies the maximum-inner-product-to-L2 extra-dimension
  transform before the same cuVS 4-bit IVF-PQ search. The uploaded correlation sweep
  `faissR_IVFPQ_FASTSCAN_TUNING_CUDA_correlation_20260703_083505` failed
  before reaching cuVS under the old Euclidean-only guard.
  `cuda_cuvs_nndescent` supports Euclidean/L2 and normalized cosine/correlation
  graph construction. CUDA NN-descent Euclidean auto tuning uses the measured
  CUDA sweep; CUDA cosine and correlation auto tuning currently use policies
  seeded from that Euclidean table and mark
  `tuning_benchmark_target_met = FALSE` until the metric-specific HPC wrappers
  are rerun. The CUDA correlation seed is
  `benchmark_scripts/cuda_nndescent_correlation_shape_tuning_defaults_from_seeded_euclidean_results.csv`.
  The HNSW route builds a CUDA CAGRA seed graph and converts it with
  `cuvsHnswFromCagraWithDataset`, supports `target_recall = 0.9`, `0.95`, or `0.99`
  speed/recall tiers for Euclidean/cosine/correlation/inner-product requests,
  and records `cuda_hnsw_design =
  "cuvs_hnsw_from_cagra_cpu_hierarchy"` because this is a cuVS wrapper route,
  not vendored CUDA code or a pure all-GPU HNSW implementation [3,22-23].
- Native CUDA graph routes such as `method = "nsg"` and `method = "vamana"`
  use faissR-owned candidate pruning plus CUDA row-candidate refinement. CUDA
  Euclidean/cosine/correlation/inner-product NSG and Euclidean/cosine/correlation Vamana
  `tuning = "auto"` now select graph parameters from compiled
  shape/k/target-recall tables; cosine is searched as
  row-normalized float32 Euclidean graph refinement and converted back to
  cosine distance. CUDA NSG correlation and Vamana correlation use the same
  centered-normalized CUDA graph routes, and CUDA NSG/Vamana raw inner product
  uses shifted dot-product ordering. CUDA NSG correlation/raw-inner-product and
  Vamana correlation/raw-inner-product currently seed their graph parameters
  from the corresponding measured CUDA cosine tables, reporting
  `tuning_benchmark_target_met = FALSE` until the dedicated metric sweeps are
  rerun [16,21,24,29].

Use `backend_info()` and the attributes returned by `nn()` to confirm which
route and parameters a result used.
Use `nn_capabilities()` to inspect which public `method`, `backend`, and
`metric` combinations are supported before launching a large benchmark.

## GPU-Resident Output For Downstream Packages

Most R users should call `nn()`, which returns ordinary R matrices. Packages
that already run CUDA code can instead call `nn_gpu()` when the KNN output must
stay on the GPU:

```r
res <- nn_gpu(
  x,
  k = 15,
  exclude_self = TRUE,
  method = "auto",
  metric = "euclidean",
  target_recall = 0.99
)
```

The returned `faissR_gpu_knn` object owns CUDA device buffers through an
external pointer and reports:

- `indices_ptr`: CUDA device pointer to a column-major `n_query x k` int32
  neighbour matrix using 1-based R indices.
- `distances_ptr`: CUDA device pointer to a column-major `n_query x k` float32
  distance matrix.
- `result_residency = "cuda"`, `index_base = 1L`,
  `indices_type = "int32"`, `distance_type = "float32"`, `metric`, and
  `backend_used` metadata.

The object is intentionally not converted to host memory by default. Use
`gpu_knn_to_host(res)` only when explicitly inspecting or testing the result in
R.

Other packages can also bypass the R wrapper and retrieve the C-callable entry
point registered by faissR:

```cpp
typedef SEXP (*faissR_nn_cuda_tuned_gpu_fun)(
  SEXP x,
  SEXP k,
  SEXP method,
  SEXP metric,
  SEXP include_self,
  SEXP target_recall
);

auto fn = reinterpret_cast<faissR_nn_cuda_tuned_gpu_fun>(
  R_GetCCallable("faissR", "faissR_nn_cuda_tuned_gpu_call")
);
```

The current GPU-resident route supports exact KNN for
`method = "auto"`, `"exact"`, `"flat"`, or `"bruteforce"`. Euclidean and raw
inner-product search use FAISS GPU direct `bfKnn` when available and request
FAISS/cuVS dispatch when faissR was built with cuVS; raw inner-product
similarities are converted on the CUDA device to faissR's shifted
smaller-is-better distance. Cosine and correlation use the native CUDA
GPU-resident exact route with the same output contract.
With `method = "auto"`, `nn_gpu()` records the same compiled auto-selection
metadata as `nn()`. If that policy would prefer an approximate method such as
IVF for ordinary `nn()` but that provider cannot yet expose persistent
GPU-resident result buffers, `nn_gpu()` keeps the exact-family GPU-resident
route and records `auto_preferred_backend`, `auto_preferred_method`, and
`auto_residency_constraint`. It also records the exact-family
`execution_tuning` used by the GPU-resident route and the
`auto_preferred_tuning` row for the preferred approximate CUDA method,
including the raw inner-product CAGRA/IVF/graph settings when those rows are
available.
Approximate FAISS GPU/cuVS methods still use `nn()` and return host objects
until those provider result buffers are exposed as persistent GPU-resident
objects.

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
