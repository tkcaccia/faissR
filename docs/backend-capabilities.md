# Backend Capabilities

[Home](../README.md) |
[Installation](installation.md) |
[Implementation](implementation.md) |
[Examples](examples.md) |
[Benchmarks](benchmarks.md) |
[API](usage-api.md) |
**Backends** |
[References](references.md)

`faissR` can be compiled as CPU-only or CPU+CUDA/cuVS.

## KNN Backends

| Backend family | CPU | CUDA | Notes |
| --- | --- | --- | --- |
| FAISS Flat | yes | yes, if FAISS GPU is built | Exact reference for medium datasets. |
| FAISS IVF-Flat | yes | yes, if FAISS GPU is built | Good speed/recall balance. |
| FAISS IVF-PQ | yes | yes, if FAISS GPU is built | Memory-saving approximate search. |
| FAISS HNSW / NSG | depends on FAISS build | no direct package default | Useful CPU graph-search baselines. |
| FAISS NN-descent | depends on FAISS build | no direct package default | Approximate CPU graph construction when available. |
| RAPIDS cuVS CAGRA | no | yes, if cuVS is built | Preferred CUDA graph-search backend when available. |
| RAPIDS cuVS NN-descent | no | yes, if cuVS is built | CUDA NN-descent backend. |
| RAPIDS cuVS IVF/PQ | no | yes, if cuVS is built | Direct cuVS approximate indexes where available. |
| Grid 2D/3D | yes | optional CUDA | Specialized low-dimensional uniform-data path. |

## Graph And Model Functions

| Function | CPU | CUDA | Notes |
| --- | --- | --- | --- |
| `knn_graph()` | yes | uses CUDA KNN if the supplied/generated KNN uses CUDA | Graph construction returns an R `igraph` object. |
| `candidate_knn()` | yes | optional CUDA candidate ranking where compiled | Useful for projection/refinement candidate sets. |
| `knn_fit()` | yes | yes, if CUDA backend requested and available | Stores training data/index metadata for repeated prediction. |
| `fast_kmeans()` | yes | yes, where FAISS/cuVS k-means is available | Use `backend_info()` to confirm. |

## No Silent Fallback

If `backend = "cuda_cuvs"` or another explicit GPU backend is requested but the
library is unavailable, `faissR` reports an error/status. It does not run CPU
code and label the result as CUDA.
