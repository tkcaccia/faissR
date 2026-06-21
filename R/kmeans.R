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
#'   uses CUDA/cuVS k-means when that backend is compiled and available, and
#'   otherwise resolves to CPU.
#' @param max_iter Maximum number of Lloyd iterations, or `"auto"` for a
#'   deterministic shape-aware default.
#' @param n_init Number of random restarts where supported, or `"auto"` for a
#'   deterministic shape-aware default.
#' @param tol Relative convergence tolerance where supported, or `"auto"` for a
#'   deterministic shape-aware default.
#' @param seed Random seed for CPU/statistics and FAISS paths. The current
#'   direct cuVS C API path does not expose an explicit seed in the stable
#'   params structure.
#' @param n_threads Number of CPU threads for FAISS/statistics paths.
#' @param streaming_batch_size cuVS host-data streaming batch size. `0` lets
#'   cuVS choose its default.
#' @param init Initialization method, `"kmeans++"` or `"random"` where
#'   supported.
#' @param tuning Tuning policy. `"auto"` uses deterministic defaults based on
#'   `nrow(data)`, `ncol(data)`, and `centers` without running pilot searches.
#'   Small many-cluster jobs can use extra restarts when `n / centers` remains
#'   large enough; large or high-dimensional jobs use cheaper iteration and
#'   tolerance defaults.
#'   `"fixed"`, `"off"`, and `"none"` use the historical fixed defaults unless
#'   `max_iter`, `n_init`, or `tol` are explicitly supplied.
#' @return A list with `cluster`, `centers`, `withinss`, `tot.withinss`,
#'   `size`, `iter`, `backend`, and `parameters`. `parameters$tuning`
#'   records the deterministic k-means policy, shape metadata, and whether
#'   `max_iter`, `n_init`, and `tol` were auto-selected or supplied explicitly.
#' @examples
#' x <- scale(as.matrix(iris[, 1:4]))
#' fit <- fast_kmeans(x, centers = 3, backend = "cpu", n_threads = 2)
#' table(fit$cluster)
#' @export
fast_kmeans <- function(data,
                        centers,
                        backend = c("auto", "cpu", "cuda"),
                        max_iter = "auto",
                        n_init = "auto",
                        tol = "auto",
                        seed = 1L,
                        n_threads = NULL,
                        streaming_batch_size = 0L,
                        init = c("kmeans++", "random"),
                        tuning = c("auto", "fixed", "off", "none")) {
  backend <- normalize_public_backend_arg(backend)
  init <- match.arg(init)
  tuning <- normalize_kmeans_tuning(tuning)
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
  auto_params <- kmeans_auto_params(
    n = nrow(x),
    p = ncol(x),
    centers = centers,
    tuning = tuning
  )
  auto_params$resolved_from <- list(
    max_iter = kmeans_value_source(max_iter),
    n_init = kmeans_value_source(n_init),
    tol = kmeans_value_source(tol)
  )
  max_iter <- normalize_kmeans_positive_int(max_iter, auto_params$max_iter)
  n_init <- normalize_kmeans_positive_int(n_init, auto_params$n_init)
  n_threads <- normalize_nn_threads(n_threads)
  seed <- suppressWarnings(as.integer(seed))
  if (length(seed) != 1L || is.na(seed) || !is.finite(seed)) seed <- 1L
  tol <- normalize_kmeans_tol(tol, auto_params$tol)
  streaming_batch_size <- suppressWarnings(as.integer(streaming_batch_size))
  if (length(streaming_batch_size) != 1L || is.na(streaming_batch_size) ||
      !is.finite(streaming_batch_size) || streaming_batch_size < 0L) {
    streaming_batch_size <- 0L
  }

  backend <- normalize_public_compute_backend(backend)

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
      tuning_metadata = auto_params,
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
      init = init,
      tuning_metadata = auto_params
    )
    return(finish_fast_kmeans(out, backend = "cuda_faiss", init = init, tuning_metadata = auto_params))
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
    return(finish_fast_kmeans(out, backend = "cuda_cuvs", init = init, tuning_metadata = auto_params))
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
    return(finish_fast_kmeans(out, backend = "faiss", init = init, tuning_metadata = auto_params))
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
      init = "stats_default",
      tuning = auto_params
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
                            tuning_metadata,
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
        init = init,
        tuning_metadata = tuning_metadata
      ),
      error = function(e) {
        faiss_error <<- conditionMessage(e)
        NULL
      }
    )
    if (!is.null(out)) {
      return(finish_fast_kmeans(out, backend = "cuda_faiss", init = init, tuning_metadata = tuning_metadata))
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
    return(finish_fast_kmeans(out, backend = "cuda_cuvs", init = init, tuning_metadata = tuning_metadata))
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
                                 init,
                                 tuning_metadata = NULL) {
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

finish_fast_kmeans <- function(out, backend, init, tuning_metadata = NULL) {
  out$cluster <- as.integer(out$cluster)
  out$centers <- unname(as.matrix(out$centers))
  out$withinss <- as.numeric(out$withinss)
  out$tot.withinss <- as.numeric(out$tot.withinss)
  out$size <- as.integer(out$size)
  out$iter <- as.integer(out$iter)
  out$backend <- backend
  if (is.null(out$parameters)) out$parameters <- list()
  out$parameters$init <- init
  if (!is.null(tuning_metadata)) out$parameters$tuning <- tuning_metadata
  class(out) <- c("faissR_kmeans", "fastEmbedR_kmeans", "kmeans")
  out
}

normalize_kmeans_tuning <- function(tuning) {
  tuning <- as.character(tuning)[1L]
  if (is.na(tuning) || !nzchar(tuning)) tuning <- "auto"
  tuning <- tolower(tuning)
  if (!tuning %in% c("auto", "fixed", "off", "none")) {
    stop("`tuning` must be one of \"auto\", \"fixed\", \"off\", or \"none\".", call. = FALSE)
  }
  tuning
}

kmeans_auto_params <- function(n, p, centers, tuning = "auto") {
  if (!identical(tuning, "auto")) {
    return(list(
      policy = tuning,
      max_iter = 100L,
      n_init = 1L,
      tol = 1e-4,
      rule = "fixed_defaults"
    ))
  }
  work <- as.double(n) * as.double(p) * as.double(centers)
  high_dim <- p >= 256L
  large_n <- n >= 100000L
  many_centers <- centers >= 100L
  n_per_center <- as.double(n) / as.double(centers)
  small_many_centers <- many_centers && n <= 50000L && work <= 2e8 && n_per_center >= 20
  max_iter <- if (large_n || work >= 5e9) {
    50L
  } else if (high_dim || (many_centers && !small_many_centers) || work >= 5e8) {
    75L
  } else {
    100L
  }
  n_init <- if (n <= 50000L && centers <= 20L && work <= 2e8) {
    5L
  } else if (small_many_centers) {
    3L
  } else if (n <= 100000L && centers <= 50L && work <= 5e8) {
    3L
  } else {
    1L
  }
  tol <- if (large_n || work >= 5e9) {
    1e-3
  } else {
    1e-4
  }
  list(
    policy = "auto",
    max_iter = as.integer(max_iter),
    n_init = as.integer(n_init),
    tol = as.numeric(tol),
    work = as.numeric(work),
    n_per_center = as.numeric(n_per_center),
    high_dim = isTRUE(high_dim),
    large_n = isTRUE(large_n),
    many_centers = isTRUE(many_centers),
    small_many_centers = isTRUE(small_many_centers),
    rule = paste(
      "shape",
      paste0("n=", n),
      paste0("p=", p),
      paste0("centers=", centers),
      paste0("n_per_center=", formatC(n_per_center, digits = 4, format = "fg")),
      paste0("work=", format(work, scientific = TRUE)),
      sep = ";"
    )
  )
}

normalize_kmeans_positive_int <- function(x, fallback) {
  if (is.character(x) && length(x) == 1L && identical(tolower(x), "auto")) {
    return(as.integer(fallback))
  }
  normalize_positive_int(x, fallback)
}

normalize_kmeans_tol <- function(x, fallback) {
  if (is.character(x) && length(x) == 1L && identical(tolower(x), "auto")) {
    return(as.numeric(fallback))
  }
  x <- suppressWarnings(as.numeric(x))
  if (length(x) != 1L || is.na(x) || !is.finite(x) || x < 0) x <- fallback
  as.numeric(x)
}

normalize_positive_int <- function(x, fallback) {
  x <- suppressWarnings(as.integer(x))
  if (length(x) != 1L || is.na(x) || !is.finite(x) || x < 1L) {
    x <- fallback
  }
  as.integer(x)
}

kmeans_value_source <- function(x) {
  if (is.character(x) && length(x) == 1L && identical(tolower(x), "auto")) {
    return("auto")
  }
  "explicit"
}
