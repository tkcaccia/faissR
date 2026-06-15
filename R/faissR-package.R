#' faissR: fast native nearest neighbours and graph utilities
#'
#' `faissR` contains the neighbour-search side of the fast embedding workflow:
#' [nn()], [candidate_knn()], [knn_graph()], [fast_kmeans()], and kNN
#' classifier/regressor helpers. `fastEmbedR` now uses this package for KNN and
#' keeps the UMAP/openTSNE embedding optimizers.
#'
#' Optional FAISS and RAPIDS cuVS backends are detected at build time. If they
#' are not available, explicit requests fail clearly and the package remains
#' usable through native CPU and optional RcppHNSW paths.
#'
#' @keywords internal
"_PACKAGE"
