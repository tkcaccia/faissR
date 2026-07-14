#' faissR: FAISS-backed nearest neighbours and kNN utilities
#'
#' `faissR` contains FAISS-backed neighbour search, k-means,
#' and kNN classifier/regressor helpers. The main public entry points are
#' `nn()`, `candidate_knn()`, `fast_kmeans()`,
#' `knn()`, `predict()`, `backend_info()`, and `nn_capabilities()`.
#' Classification probabilities
#' are returned with `predict(type = "prob")`.
#'
#' FAISS is a required system dependency for all builds. RAPIDS cuVS/CUDA is
#' optional for CPU-only builds, so CPU-only machines can compile and use FAISS
#' CPU indexes without NVIDIA libraries. For NVIDIA GPU builds, users should
#' request the GPU features explicitly so missing CUDA/cuVS libraries
#' are fatal at configure time. FAISS GPU indexes can use NVIDIA cuVS
#' integration when linked against a cuVS-enabled FAISS build; direct RAPIDS
#' cuVS backends are also available when requested at build time. Explicit
#' CUDA/cuVS requests fail clearly when those optional libraries are
#' unavailable. Apple Metal provides native exact float32 2D/3D grid KNN. No
#' Python bridge is used.
#'
#' @keywords internal
"_PACKAGE"
