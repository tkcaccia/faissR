# Backend Capabilities

[Home](../README.md) |
[Installation](installation.md) |
[Implementation](implementation.md) |
[Examples](examples.md) |
[Benchmarks](benchmarks.md) |
[API](usage-api.md) |
**Backends** |
[References](references.md)

`faissR` can be compiled as CPU-only with FAISS, or as CPU+CUDA with optional
FAISS GPU, RAPIDS cuVS, and RAPIDS libcugraph support.

## KNN Backends

| Backend family | CPU | CUDA | Notes |
| --- | --- | --- | --- |
| FAISS Flat | yes | yes, if FAISS GPU is built | Exact L2 search. |
| FAISS IVF-Flat | yes | yes, if FAISS GPU is built | Inverted-file approximate search. |
| FAISS IVF-PQ | yes | yes, if FAISS GPU is built | Product-quantized approximate search; useful for memory reduction. |
| FAISS HNSW / NSG | depends on FAISS build | no package default | CPU graph-search baselines where supported. |
| FAISS NN-descent | depends on FAISS build | no package default | Approximate CPU graph construction where supported. |
| RAPIDS cuVS brute force | no | yes, if cuVS is built | Exact direct cuVS route. |
| RAPIDS cuVS CAGRA | no | yes, if cuVS is built | CUDA graph-search backend. |
| RAPIDS cuVS NN-descent | no | yes, if cuVS is built | CUDA NN-descent backend. |
| RAPIDS cuVS IVF/PQ | no | yes, if cuVS is built | Direct cuVS approximate indexes. |
| Grid 2D/3D | yes | optional CUDA | Specialized low-dimensional path. |

## Graph, Clustering, And Model Functions

| Function | CPU | CUDA | Notes |
| --- | --- | --- | --- |
| `knn_graph()` | yes | uses CUDA KNN if the supplied/generated KNN uses CUDA | Returns a native `faissR_graph` edge list. |
| `graph_cluster()` | yes | Louvain/Leiden with libcugraph when built | CPU random-walking/Louvain/Leiden use native C++/OpenMP. CUDA random-walking is not enabled yet. |
| `candidate_knn()` | yes | optional CUDA candidate ranking where compiled | Useful for projection/refinement candidate sets. |
| `knn()` / `predict()` | yes | yes, if CUDA backend requested and available | Stores training data/index metadata for repeated prediction, or predicts immediately when `Xtest` is supplied. |
| `fast_kmeans()` | yes | yes, where FAISS/cuVS k-means is available | Use `backend_info()` to confirm. |

## Availability Helpers

Use these helpers before benchmarking or selecting explicit backends:

```r
backend_info()
faiss_available()
cuda_available()
cuvs_available()
cugraph_available()
```

## No Silent Fallback

If an explicit GPU backend is requested but the library is unavailable,
`faissR` reports an error/status. It does not run CPU code and label the result
as CUDA.
