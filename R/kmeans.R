#' Fast k-means clustering
#'
#' Run k-means using the fastest available backend. CPU acceleration uses
#' FAISS when fastEmbedR was built with FAISS; CUDA acceleration uses RAPIDS
#' cuVS when fastEmbedR was built with cuVS. If an optional backend is not
#' available, explicit requests fail clearly instead of silently changing
#' backend.
#'
#' @param data Numeric matrix with observations in rows.
#' @param centers Number of clusters.
#' @param backend One of `"auto"`, `"cpu"`, `"faiss"`, `"cuda"`,
#'   `"cuda_cuvs"`, or `"cuvs"`.
#' @param max_iter Maximum number of Lloyd iterations.
#' @param n_init Number of random restarts where supported.
#' @param tol Relative convergence tolerance where supported.
#' @param seed Random seed for CPU/statistics and FAISS paths. The current cuVS
#'   C API does not expose an explicit seed in the stable params structure.
#' @param n_threads Number of CPU threads for FAISS/statistics paths.
#' @param streaming_batch_size cuVS host-data streaming batch size. `0` lets
#'   cuVS choose its default.
#' @param init Initialization method, `"kmeans++"` or `"random"` where
#'   supported.
#' @return A list with `cluster`, `centers`, `withinss`, `tot.withinss`,
#'   `size`, `iter`, `backend`, and `parameters`.
#' @examples
#' x <- scale(as.matrix(iris[, 1:4]))
#' fit <- fast_kmeans(x, centers = 3, backend = "cpu", n_threads = 2)
#' table(fit$cluster)
#' @export
fast_kmeans <- function(data,
                        centers,
                        backend = c("auto", "cpu", "faiss", "cuda", "cuda_cuvs", "cuvs"),
                        max_iter = 100L,
                        n_init = 1L,
                        tol = 1e-4,
                        seed = 1L,
                        n_threads = NULL,
                        streaming_batch_size = 0L,
                        init = c("kmeans++", "random")) {
  backend <- match.arg(backend)
  init <- match.arg(init)
  x <- as.matrix(data)
  storage.mode(x) <- "double"
  if (nrow(x) < 1L || ncol(x) < 1L) {
    stop("`data` must have at least one row and one column.", call. = FALSE)
  }
  if (!all(is.finite(x))) {
    stop("`data` must contain only finite values.", call. = FALSE)
  }
  centers <- suppressWarnings(as.integer(centers))
  if (length(centers) != 1L || is.na(centers) || !is.finite(centers) ||
      centers < 1L || centers > nrow(x)) {
    stop("`centers` must be an integer in [1, nrow(data)].", call. = FALSE)
  }
  max_iter <- normalize_positive_int(max_iter, 100L)
  n_init <- normalize_positive_int(n_init, 1L)
  n_threads <- normalize_nn_threads(n_threads)
  seed <- suppressWarnings(as.integer(seed))
  if (length(seed) != 1L || is.na(seed) || !is.finite(seed)) seed <- 1L
  tol <- suppressWarnings(as.numeric(tol))
  if (length(tol) != 1L || is.na(tol) || !is.finite(tol) || tol < 0) tol <- 1e-4
  streaming_batch_size <- suppressWarnings(as.integer(streaming_batch_size))
  if (length(streaming_batch_size) != 1L || is.na(streaming_batch_size) ||
      !is.finite(streaming_batch_size) || streaming_batch_size < 0L) {
    streaming_batch_size <- 0L
  }

  if (identical(backend, "auto")) {
    backend <- if (isTRUE(cuvs_available()) && isTRUE(cuda_available())) {
      "cuda_cuvs"
    } else if (isTRUE(faiss_available())) {
      "faiss"
    } else {
      "cpu"
    }
  }
  if (identical(backend, "cuda")) backend <- "cuda_cuvs"
  if (identical(backend, "cuvs")) backend <- "cuda_cuvs"

  if (identical(backend, "faiss")) {
    if (!isTRUE(faiss_available())) {
      stop(
        "FAISS k-means is not available. Reinstall fastEmbedR with ",
        "FASTEMBEDR_USE_FAISS=1 and FAISS_HOME=/path/to/faiss.",
        call. = FALSE
      )
    }
    out <- kmeans_faiss_cpp(
      x,
      as.integer(centers),
      as.integer(max_iter),
      as.integer(n_init),
      as.numeric(tol),
      as.integer(seed),
      as.integer(n_threads),
      identical(init, "kmeans++")
    )
    return(finish_fast_kmeans(out, backend = "faiss", init = init))
  }

  if (identical(backend, "cuda_cuvs")) {
    require_cuvs_backend("cuVS k-means")
    out <- kmeans_cuvs_cpp(
      x,
      as.integer(centers),
      as.integer(max_iter),
      as.integer(n_init),
      as.numeric(tol),
      as.integer(streaming_batch_size),
      identical(init, "kmeans++")
    )
    return(finish_fast_kmeans(out, backend = "cuda_cuvs", init = init))
  }

  if (identical(backend, "cpu") && isTRUE(faiss_available())) {
    out <- kmeans_faiss_cpp(
      x,
      as.integer(centers),
      as.integer(max_iter),
      as.integer(n_init),
      as.numeric(tol),
      as.integer(seed),
      as.integer(n_threads),
      identical(init, "kmeans++")
    )
    return(finish_fast_kmeans(out, backend = "faiss", init = init))
  }

  set.seed(seed)
  stats_fit <- stats::kmeans(
    x,
    centers = centers,
    iter.max = max_iter,
    nstart = n_init,
    algorithm = "Lloyd"
  )
  out <- list(
    cluster = as.integer(stats_fit$cluster),
    centers = unname(as.matrix(stats_fit$centers)),
    withinss = as.numeric(stats_fit$withinss),
    tot.withinss = as.numeric(stats_fit$tot.withinss),
    size = as.integer(stats_fit$size),
    iter = as.integer(stats_fit$iter),
    backend = "cpu",
    backend_library = "stats",
    parameters = list(
      centers = as.integer(centers),
      max_iter = as.integer(max_iter),
      n_init = as.integer(n_init),
      tol = as.numeric(tol),
      seed = as.integer(seed),
      n_threads = as.integer(n_threads),
      init = "stats_default"
    )
  )
  class(out) <- c("faissR_kmeans", "fastEmbedR_kmeans", "kmeans")
  out
}

finish_fast_kmeans <- function(out, backend, init) {
  out$cluster <- as.integer(out$cluster)
  out$centers <- unname(as.matrix(out$centers))
  out$withinss <- as.numeric(out$withinss)
  out$tot.withinss <- as.numeric(out$tot.withinss)
  out$size <- as.integer(out$size)
  out$iter <- as.integer(out$iter)
  out$backend <- backend
  if (is.null(out$parameters)) out$parameters <- list()
  out$parameters$init <- init
  class(out) <- c("faissR_kmeans", "fastEmbedR_kmeans", "kmeans")
  out
}

normalize_positive_int <- function(x, fallback) {
  x <- suppressWarnings(as.integer(x))
  if (length(x) != 1L || is.na(x) || !is.finite(x) || x < 1L) {
    x <- fallback
  }
  as.integer(x)
}
