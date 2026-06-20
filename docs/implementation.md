# Implementation

[Home](../README.md) |
[Installation](installation.md) |
**Implementation** |
[Examples](examples.md) |
[Benchmarks](benchmarks.md) |
[API](usage-api.md) |
[Backends](backend-capabilities.md) |
[References](references.md)

`faissR` is designed as a standalone nearest-neighbour and vector-search layer.
The package keeps FAISS and cuVS logic out of `fastEmbedR`, while exposing
plain R objects that can be reused by dimensionality reduction, graph
clustering, and supervised kNN workflows.

## Design Principles

1. FAISS is the required CPU vector-search dependency.
2. RAPIDS cuVS and CUDA are optional accelerated backends.
3. Backend labels are strict: explicit GPU requests never silently run on CPU.
4. Returned KNN objects use simple `indices` and `distances` matrices so they
   are easy to save, benchmark, and pass to other packages.
5. Expensive KNN work should be computed once and reused.

## Nearest Neighbours

`nn()` supports dense numeric matrices and selected sparse inputs. The public
distance choices are intentionally small:

- `metric = "euclidean"` maps to L2 search;
- `metric = "cosine"` normalizes rows and uses inner-product search where the
  backend supports it.

FAISS provides exact and approximate CPU indexes, including flat search,
inverted-file search, product-quantized variants, HNSW/NSG-style graph indexes
where available in the linked FAISS build, and CPU NN-descent where available.
CUDA builds can use FAISS GPU flat/IVF indexes and RAPIDS cuVS graph indexes
such as CAGRA and NN-descent. The auto router prefers high-recall GPU graph
search when CUDA/cuVS is available, then FAISS CPU indexes otherwise.

## Candidate KNN

`candidate_knn()` solves a frequent optimization problem: a previous algorithm
has already generated a candidate set per query row, and only the best `k`
distances must be retained. The package ranks the candidates without building
a full all-pairs distance matrix. This is useful for projection, refinement,
and approximate-search diagnostics.

## Graph Construction

`knn_graph()` converts KNN output into a native `faissR_graph` edge-list object. The supported weights
are:

- `weight = "distance"`: closer neighbours receive stronger edge weights based
  on the neighbour distances;
- `weight = "snn"`: edges are weighted by shared-nearest-neighbour overlap.

The function accepts an existing KNN object directly. This avoids repeating
nearest-neighbour search when the same graph is needed for clustering or
embedding diagnostics. `graph_cluster()` can then run native random-walking,
Louvain, or Leiden-style clustering without depending on `igraph`. CUDA Louvain
and Leiden use RAPIDS libcugraph when faissR is built with libcugraph; CUDA
random-walking is not enabled yet.

## kNN Models

`knn_fit()`, `faiss.fit()`, and `cuvs.fit()` build a reusable kNN classifier or
regressor. Prediction uses majority vote or distance-weighted vote for labels,
and `predict(type = "prob")` returns class probabilities from neighbour votes.

This model API is useful when a FAISS/cuVS index should be fitted once and
queried repeatedly, for example with ImageNet-like feature matrices.

## k-means

`fast_kmeans()` provides FAISS/cuVS-backed clustering when the corresponding
backend is available. It is intended as a practical high-throughput alternative
to base R k-means for large matrices, while still returning R-friendly
centroids, assignments, and objective summaries.

## Relationship To fastEmbedR

`fastEmbedR` imports `faissR` and calls `faissR::nn()` for matrix-input
`opentsne()` and `umap()`. The embedding package does not re-export the KNN
functions. Users who want explicit control over vector search should load
`faissR` directly, compute KNN once, and pass the result to
`fastEmbedR::opentsne_knn()` or `fastEmbedR::umap_knn()`.

## Licensing And Acknowledgement

FAISS, RAPIDS cuVS, RAPIDS cuGraph, HNSW, NN-descent, IVF, product
quantization, graph clustering, and k-means literature informed the
implementation choices [1-12]. External libraries are linked as system
dependencies and are not vendored into the package. faissR is released under
the MIT license.
