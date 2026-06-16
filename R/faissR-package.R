#' faissR: fast native nearest neighbours and graph utilities
#'
#' `faissR` contains the neighbour-search side of the fast embedding workflow:
#' [nn()], [candidate_knn()], [knn_graph()], [fast_kmeans()], and kNN
#' classifier/regressor helpers. `fastEmbedR` now uses this package for KNN and
#' keeps the UMAP/openTSNE embedding optimizers.
#'
#' FAISS is a required system dependency and provides the core C++ similarity
#' search and clustering indexes. RAPIDS cuVS/CUDA is optional, so CPU-only
#' machines can compile and use FAISS CPU indexes without NVIDIA libraries.
#' When FAISS itself is built with CUDA/cuVS support, `faissR` can use
#' FAISS-owned GPU IVF and CAGRA indexes backed by cuVS. Direct RAPIDS cuVS
#' backends are exposed separately for explicit comparisons. Explicit CUDA/cuVS
#' requests fail clearly when those optional libraries are unavailable.
#'
#' `faissR` acknowledges the FAISS project, Meta FAIR, RAPIDS cuVS, and NVIDIA
#' CUDA as the underlying systems that make the high-performance CPU/GPU
#' backends possible. See the package README and `citation("faissR")` for
#' suggested acknowledgements.
#'
#' @keywords internal
"_PACKAGE"
