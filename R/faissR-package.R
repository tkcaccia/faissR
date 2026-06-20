#' faissR: FAISS-backed nearest neighbours, graphs, and kNN utilities
#'
#' `faissR` contains FAISS-backed neighbour search, graph construction, graph clustering, k-means,
#' and kNN classifier/regressor helpers. The main public entry points are
#' [nn()], [candidate_knn()], [knn_graph()], [graph_cluster()], [fast_kmeans()], [knn_fit()],
#' [faiss.fit()], [cuvs.fit()], and [predict()]. Classification probabilities
#' are returned with `predict(type = "prob")`.
#'
#' FAISS is a required system dependency. RAPIDS cuVS/CUDA is optional, so
#' CPU-only machines can compile and use FAISS CPU indexes without NVIDIA
#' libraries. FAISS GPU indexes can use NVIDIA cuVS integration when linked
#' against a cuVS-enabled FAISS build; direct RAPIDS cuVS backends are also
#' available when requested at build time. Explicit CUDA/cuVS requests fail
#' clearly when those optional libraries are unavailable. CPU graph clustering uses
#' `igraph`; CUDA graph clustering is reserved for a future RAPIDS cuGraph
#' binding for Louvain, Leiden, and random-walk algorithms.
#'
#' @keywords internal
"_PACKAGE"
