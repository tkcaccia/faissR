# faissR

**Home** |
[Installation](docs/installation.md) |
[Implementation](docs/implementation.md) |
[Examples](docs/examples.md) |
[Benchmarks](docs/benchmarks.md) |
[API](docs/usage-api.md) |
[Backends](docs/backend-capabilities.md) |
[References](docs/references.md)

`faissR` is an R package for fast nearest-neighbour search, graph construction,
kNN prediction, and k-means. It is designed as the nearest-neighbour companion
package for `fastEmbedR`, but the package can also be used directly whenever an
R workflow needs explicit CPU, FAISS, CUDA, or cuVS-backed KNN methods.

The package builds around [FAISS](https://faiss.ai/index.html), Meta FAIR's
C++ library for efficient similarity search and clustering of dense vectors.
FAISS provides exact and approximate indexes, GPU implementations, batched
search, Euclidean/L2 search, and maximum inner-product search. `faissR` exposes
those capabilities through R-friendly return objects while keeping explicit
backend requests honest: if a GPU backend is requested and unavailable, the call
fails instead of silently falling back to CPU.

The package focuses on:

- `nn()` for FAISS CPU and optional FAISS/cuVS CUDA nearest-neighbour search;
- `candidate_knn()` for top-k ranking inside supplied candidate sets;
- `knn_graph()` for `igraph` graph construction from data, KNN output, or an
  embedding;
- `fast_kmeans()` for FAISS/cuVS-backed k-means;
- `knn_fit()`, `faiss.fit()`, `cuvs.fit()`, `predict()`, and `predict_proba()`
  for kNN classification and regression;
- explicit backend reporting, with no silent CPU fallback labelled as GPU.

`fastEmbedR` calls `faissR::nn()` internally for one-call UMAP/openTSNE, and
advanced users can compute KNN once with `faissR` and reuse it across multiple
embedding or clustering workflows.

## Installation

FAISS is a mandatory system dependency for `faissR`. RAPIDS cuVS/CUDA is
optional and only needed for NVIDIA GPU backends.

```r
install.packages("remotes")
remotes::install_github("tkcaccia/faissR")
```

See [Installation](docs/installation.md) for macOS, Linux, conda, FAISS, CUDA,
and cuVS requirements.

## FAISS GPU With cuVS

`faissR` distinguishes two GPU/cuVS routes. This distinction matters for
benchmarking and methods sections, because FAISS can call cuVS internally while
the package can also call RAPIDS cuVS directly.

- FAISS GPU indexes with NVIDIA cuVS integration, exposed through FAISS-backed
  backends such as `faiss_gpu_ivf_flat`, `faiss_gpu_ivfpq`, and
  `faiss_gpu_cagra`. When the linked FAISS library was built with cuVS support,
  these paths report backend labels such as `GpuIndexIVFFlat_cuVS`,
  `GpuIndexIVFPQ_cuVS`, and `GpuIndexCagra_cuVS`.
- Direct RAPIDS cuVS calls, exposed through explicit backends such as
  `cuda_cuvs_cagra`, `cuda_cuvs_nndescent`, `cuda_cuvs_bruteforce`,
  `cuda_cuvs_ivf_flat`, and `cuda_cuvs_ivfpq`.

Use `backend_info()` and the `backend` attribute returned by `nn()` to confirm
which route a result used.

The FAISS/cuVS route follows the direction described by Meta and NVIDIA for
FAISS 1.10 and later: IVF-Flat, IVF-PQ, and CAGRA are exposed as FAISS GPU
indexes while cuVS provides accelerated GPU implementations underneath. The
direct cuVS route is useful for testing the RAPIDS C API directly, but paper
benchmarks should label it separately from FAISS-integrated cuVS results.

Common explicit backends:

| Backend | Route | Notes |
| --- | --- | --- |
| `faiss_flat_l2` | FAISS CPU | Exact L2 search. |
| `faiss_ivf` | FAISS CPU | Inverted-file approximate L2 search. |
| `faiss_ivfpq` | FAISS CPU | IVF with product quantization. |
| `faiss_hnsw`, `faiss_nsg`, `faiss_nndescent` | FAISS CPU | Graph/ANN indexes when supported by the linked FAISS build. |
| `faiss_gpu_flat_l2` | FAISS GPU | Exact GPU L2 search. |
| `faiss_gpu_ivf_flat` | FAISS GPU + cuVS when available | FAISS-owned IVF-Flat GPU index. |
| `faiss_gpu_ivfpq` | FAISS GPU + cuVS when available | FAISS-owned IVF-PQ GPU index. |
| `faiss_gpu_cagra` | FAISS GPU + cuVS when available | FAISS-owned CAGRA graph index. |
| `cuda_cuvs_bruteforce` | Direct RAPIDS cuVS | Exact cuVS L2 search. |
| `cuda_cuvs_ivf_flat`, `cuda_cuvs_ivfpq`, `cuda_cuvs_nndescent` | Direct RAPIDS cuVS | Explicit direct-cuVS comparison paths. |

For no-inner-product benchmarks, use L2/cosine-capable methods only. FAISS
inner-product indexes solve a different objective, maximum dot-product search,
and should be benchmarked separately.

## Quick Example

```r
library(faissR)

x <- scale(as.matrix(iris[, 1:4]))
knn <- nn(x, k = 15, backend = "auto", metric = "euclidean", n_threads = 4)

str(knn)
```

Build a shared-nearest-neighbour graph:

```r
if (requireNamespace("igraph", quietly = TRUE)) {
  g <- knn_graph(knn, k = 15, weight = "snn")
  cl <- igraph::cluster_louvain(g, weights = igraph::E(g)$weight)
  table(igraph::membership(cl))
}
```

Use the same KNN output in `fastEmbedR`:

```r
library(fastEmbedR)

y_tsne <- fastEmbedR::opentsne_knn(knn, init_data = x, backend = "cpu")
y_umap <- fastEmbedR::umap_knn(knn, backend = "cpu", graph_mode = "fuzzy")
```

## Backend Check

```r
library(faissR)

backend_info()
faiss_available()
cuda_available()
cuvs_available()
```

An explicit GPU backend request must resolve to a real GPU implementation.
Otherwise the call fails clearly and reports the missing dependency.

## Documentation

- [Installation](docs/installation.md): FAISS, CUDA, cuVS, and environment
  variables.
- [Implementation](docs/implementation.md): algorithmic details and design
  rationale.
- [Examples](docs/examples.md): small reproducible examples.
- [Benchmarks](docs/benchmarks.md): recommended benchmark design and output.
- [API](docs/usage-api.md): user-facing functions and return objects.
- [Backends](docs/backend-capabilities.md): CPU/CUDA capability table.
- [References](docs/references.md): AACR-style literature and software
  references.

## Acknowledgements

`faissR` is an R interface layer and benchmarking companion around major open
source systems:

- [FAISS](https://faiss.ai/index.html), developed primarily by Meta FAIR, is the
  required C++ similarity-search and clustering library used by the FAISS
  backends.
- [RAPIDS cuVS](https://github.com/rapidsai/cuvs) and NVIDIA CUDA provide the
  optional direct GPU vector-search backends.
- Meta and NVIDIA's FAISS/cuVS integration work explains why IVF-Flat, IVF-PQ,
  and CAGRA should be treated as important GPU methods in FAISS-backed
  benchmarks: [Accelerating GPU indexes in Faiss with NVIDIA cuVS](https://engineering.fb.com/2025/05/08/data-infrastructure/accelerating-gpu-indexes-in-faiss-with-nvidia-cuvs/).

Please cite and acknowledge FAISS, RAPIDS cuVS/NVIDIA CUDA, and the relevant
nearest-neighbour methods when publishing results obtained with `faissR`.
