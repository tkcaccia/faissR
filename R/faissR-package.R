#' faissR: fast native nearest neighbours and graph utilities
#'
#' `faissR` contains the neighbour-search side of the fast embedding workflow:
#' [nn()], [candidate_knn()], [knn_graph()], [fast_kmeans()], and kNN
#' classifier/regressor helpers. `fastEmbedR` now uses this package for KNN and
#' keeps the UMAP/openTSNE embedding optimizers.
#'
#' FAISS is a required system dependency. RAPIDS cuVS/CUDA is optional, so
#' CPU-only machines can compile and use FAISS CPU indexes without NVIDIA
#' libraries. Explicit CUDA/cuVS requests fail clearly when those optional
#' libraries are unavailable.
#'
#' @keywords internal
"_PACKAGE"
