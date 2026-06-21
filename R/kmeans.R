#' Fast k-means clustering
#'
#' Run k-means using the fastest available backend. CPU acceleration uses
#' FAISS when faissR was built with FAISS. CUDA acceleration first tries FAISS
#' GPU, which can use NVIDIA/cuVS-enabled FAISS builds, then direct RAPIDS cuVS
#' when available. Explicit GPU requests fail clearly instead of silently
#' changing to CPU.
#'
#' @param data Numeric matrix with observations in rows.
#' @param centers Number of clusters.
#' @param backend Device backend: `"auto"`, `"cpu"`, or `"cuda"`. `"auto"`
#'   uses CUDA when CUDA/cuVS k-means is available and CPU otherwise.
#' @param max_iter Maximum number of Lloyd iterations.
#' @param n_init Number of random restarts where supported.
#' @param tol Relative convergence tolerance where supported.
#' @param seed Random seed for CPU/statistics and FAISS paths. The current
#'   direct cuVS C API path does not expose an explicit seed in the stable
#'   params structure.
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
                        backend = c("auto", "cpu", "cuda"),
                        max_iter = 100L,
                        n_init = 1L,
                        tol = 1e-4,
                        seed = 1L,
                        n_threads = NULL,
                        streaming_batch_size = 0L,
                        init = c("kmeans++", "random")) {
  backend <- as.character(backend)[1L]
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

  if (backend %in% c("faiss")) {
    backend <- "cpu"
  } else if (backend %in% c("cuda_faiss", "faiss_gpu", "cuda_cuvs", "cuvs")) {
    backend <- "cuda"
  } else {
    backend <- normalize_public_compute_backend(backend)
  }

  if (identical(backend, "cuda")) {
    return(run_cuda_kmeans(
      x = x,
      centers = centers,
      max_iter = max_iter,
      n_init = n_init,
      tol = tol,
      seed = seed,
      streaming_batch_size = streaming_batch_size,
      init = init,
      allow_cuvs_fallback = TRUE
    ))
  }

  if (identical(backend, "cuda_faiss")) {
    out <- run_faiss_gpu_kmeans(
      x = x,
      centers = centers,
      max_iter = max_iter,
      n_init = n_init,
      tol = tol,
      seed = seed,
      init = init
    )
    return(finish_fast_kmeans(out, backend = "cuda_faiss", init = init))
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

run_cuda_kmeans <- function(x,
                            centers,
                            max_iter,
                            n_init,
                            tol,
                            seed,
                            streaming_batch_size,
                            init,
                            allow_cuvs_fallback = TRUE) {
  faiss_error <- NULL
  if (isTRUE(faiss_available())) {
    out <- tryCatch(
      run_faiss_gpu_kmeans(
        x = x,
        centers = centers,
        max_iter = max_iter,
        n_init = n_init,
        tol = tol,
        seed = seed,
        init = init
      ),
      error = function(e) {
        faiss_error <<- conditionMessage(e)
        NULL
      }
    )
    if (!is.null(out)) {
      return(finish_fast_kmeans(out, backend = "cuda_faiss", init = init))
    }
  }

  if (allow_cuvs_fallback && isTRUE(cuvs_available()) && isTRUE(cuda_available())) {
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

  cuvs_note <- if (isTRUE(cuvs_available()) && isTRUE(cuda_available())) {
    "direct cuVS is available but was not used"
  } else {
    "direct cuVS is unavailable"
  }
  if (is.null(faiss_error)) {
    faiss_error <- "FAISS GPU k-means was not attempted because FAISS is unavailable"
  }
  stop(
    "CUDA k-means is not available. FAISS GPU status: ",
    faiss_error,
    "; cuVS status: ",
    cuvs_note,
    ".",
    call. = FALSE
  )
}

run_faiss_gpu_kmeans <- function(x,
                                 centers,
                                 max_iter,
                                 n_init,
                                 tol,
                                 seed,
                                 init) {
  if (!isTRUE(faiss_available())) {
    stop(
      "FAISS GPU k-means requires faissR to be built with FAISS.",
      call. = FALSE
    )
  }
  kmeans_faiss_gpu_cpp(
    x,
    as.integer(centers),
    as.integer(max_iter),
    as.integer(n_init),
    as.numeric(tol),
    as.integer(seed),
    identical(init, "kmeans++")
  )
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
