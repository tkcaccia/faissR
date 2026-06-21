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
FAISS GPU, RAPIDS cuVS, and RAPIDS libcugraph support [1-3,12-16].

## KNN Backends

| Backend family | CPU | CUDA | Notes |
| --- | --- | --- | --- |
| FAISS Flat | yes | yes, if FAISS GPU is built | Exact L2 search [1-2,16]. |
| FAISS IVF-Flat | yes | yes, if FAISS GPU is built | Inverted-file approximate search [1-2,16]. |
| FAISS IVF-PQ | yes | yes, if FAISS GPU is built | Product-quantized approximate search; useful for memory reduction [6,16]. |
| FAISS HNSW / NSG | depends on FAISS build | no package default | CPU graph-search baselines where supported [5,16]. |
| FAISS NN-descent | depends on FAISS build | no package default | Approximate CPU graph construction where supported [4,16]. |
| RAPIDS cuVS brute force | no | yes, if cuVS is built | Exact direct cuVS route [3]. |
| RAPIDS cuVS CAGRA | no | yes, if cuVS is built | CUDA graph-search backend [3,13-15]. |
| RAPIDS cuVS NN-descent | no | yes, if cuVS is built | CUDA NN-descent backend [3-4]. |
| RAPIDS cuVS IVF/PQ | no | yes, if cuVS is built | Direct cuVS approximate indexes [3,6]. |
| Grid 2D/3D | yes | optional CUDA | Specialized low-dimensional path. |

## Graph, Clustering, And Model Functions

| Function | CPU | CUDA | Notes |
| --- | --- | --- | --- |
| `knn_graph()` | yes | uses CUDA KNN if the supplied/generated KNN uses CUDA | Returns a native `faissR_graph` edge list. |
| `graph_cluster()` | yes | Louvain/Leiden with libcugraph when built | CPU random-walking/Louvain/Leiden use native C++/OpenMP [9-11]. CUDA random-walking is not enabled yet. |
| `candidate_knn()` | yes | optional CUDA candidate ranking where compiled | Useful for projection/refinement candidate sets. |
| `knn()` / `predict()` | yes | yes, if CUDA backend requested and available | Stores training data/index metadata for repeated prediction, or predicts immediately when `Xtest` is supplied. |
| `fast_kmeans()` | yes | yes, where FAISS/cuVS k-means is available | Use `backend_info()` to confirm [7-8]. |

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
