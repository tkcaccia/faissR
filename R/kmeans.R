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
#'   uses CUDA only when CUDA plus FAISS GPU k-means or direct cuVS k-means is
#'   compiled and available and the deterministic shape rule estimates enough
#'   work to offset GPU launch and host/device copy overhead; otherwise it
#'   resolves to CPU.
#' @param max_iter Maximum number of Lloyd iterations, or `"auto"` for a
#'   deterministic shape-aware default.
#' @param n_init Number of random restarts where supported, or `"auto"` for a
#'   deterministic shape-aware default.
#' @param tol Single non-negative finite relative convergence tolerance where
#'   supported, or `"auto"` for a deterministic shape-aware default.
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
#'   `size`, `iter`, `converged`, `hit_max_iter`, `backend`, and
#'   `parameters`. `backend` records the
#'   implementation that actually ran, while `parameters$requested_backend` and
#'   `parameters$resolved_backend` record the public backend request and device
#'   policy result. `parameters$tuning` records the deterministic k-means policy,
#'   shape metadata, and whether `max_iter`, `n_init`, and `tol` were
#'   auto-selected or supplied explicitly. `parameters$tuning$effective` records
#'   the final values used after explicit overrides and `"auto"` defaults have
#'   been resolved; `parameters$tuning$effective_max_iter`,
#'   `parameters$tuning$effective_n_init`, and
#'   `parameters$tuning$effective_tol` expose the same values as flat fields for
#'   benchmark summaries. `parameters$tuning$backend_policy` records the
#'   deterministic shape rule used by `backend = "auto"` to decide whether CUDA
#'   has enough estimated work or input size to offset transfer overhead.
#'   `parameters$tuning$selection` stores the static no-pilot backend and
#'   effective-parameter decision used for benchmark auditing. `hit_max_iter`
#'   records whether the run reached the effective iteration cap, and
#'   `converged` is the corresponding conservative convergence flag used by
#'   benchmark summaries.
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
  requested_backend <- normalize_public_backend_arg(backend)
  backend <- requested_backend
  init <- normalize_kmeans_init(init)
  tuning <- normalize_kmeans_tuning(tuning)
  x <- as.matrix(data)
  storage.mode(x) <- "double"
  if (nrow(x) < 1L || ncol(x) < 1L) {
    stop("`data` must have at least one row and one column.", call. = FALSE)
  }
  if (!all(is.finite(x))) {
    stop("`data` must contain only finite values.", call. = FALSE)
  }
  centers <- normalize_kmeans_whole_number(centers, "centers", min_value = 1L)
  if (centers > nrow(x)) {
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
  max_iter <- normalize_kmeans_positive_int(max_iter, auto_params$max_iter, "max_iter")
  n_init <- normalize_kmeans_positive_int(n_init, auto_params$n_init, "n_init")
  n_threads <- normalize_nn_threads(n_threads)
  seed <- normalize_kmeans_seed(seed)
  tol <- normalize_kmeans_tol(tol, auto_params$tol)
  auto_params$effective <- list(
    max_iter = as.integer(max_iter),
    n_init = as.integer(n_init),
    tol = as.numeric(tol)
  )
  auto_params$effective_max_iter <- as.integer(max_iter)
  auto_params$effective_n_init <- as.integer(n_init)
  auto_params$effective_tol <- as.numeric(tol)
  streaming_batch_size <- normalize_kmeans_streaming_batch_size(streaming_batch_size)
  auto_params$backend_policy <- kmeans_auto_backend_policy(
    n = nrow(x),
    p = ncol(x),
    centers = centers
  )

  backend <- resolve_fast_kmeans_backend(
    backend,
    n = nrow(x),
    p = ncol(x),
    centers = centers
  )
  auto_params$selection <- kmeans_selection_metadata(
    requested_backend = requested_backend,
    resolved_backend = backend,
    n = nrow(x),
    p = ncol(x),
    centers = centers,
    effective = auto_params$effective,
    backend_policy = auto_params$backend_policy,
    tuning = tuning
  )

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
      allow_cuvs_fallback = TRUE,
      requested_backend = requested_backend,
      resolved_backend = backend
    ))
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
    return(finish_fast_kmeans(
      out,
      backend = "faiss",
      init = init,
      tuning_metadata = auto_params,
      requested_backend = requested_backend,
      resolved_backend = backend
    ))
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
    hit_max_iter = kmeans_hit_max_iter(stats_fit$iter, max_iter),
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
      requested_backend = requested_backend,
      resolved_backend = backend,
      tuning = auto_params
    )
  )
  out$converged <- if (is.na(out$hit_max_iter)) NA else !isTRUE(out$hit_max_iter)
  out$parameters$hit_max_iter <- out$hit_max_iter
  out$parameters$converged <- out$converged
  class(out) <- c("faissR_kmeans", "kmeans")
  out
}

resolve_fast_kmeans_backend <- function(backend,
                                        n = NULL,
                                        p = NULL,
                                        centers = NULL,
                                        cuda_available_value = cuda_available(),
                                        faiss_gpu_available_value = faiss_gpu_available(),
                                        cuvs_available_value = cuvs_available()) {
  backend <- normalize_public_backend_arg(backend)
  if (identical(backend, "auto")) {
    if (isTRUE(cuda_available_value) &&
        (isTRUE(faiss_gpu_available_value) || isTRUE(cuvs_available_value)) &&
        isTRUE(kmeans_auto_prefers_cuda(n, p, centers))) {
      return("cuda")
    }
    return("cpu")
  }
  backend
}

kmeans_auto_prefers_cuda <- function(n, p, centers) {
  isTRUE(kmeans_auto_backend_policy(n, p, centers)$prefer_cuda)
}

kmeans_selection_metadata <- function(requested_backend,
                                      resolved_backend,
                                      n,
                                      p,
                                      centers,
                                      effective,
                                      backend_policy,
                                      tuning = "auto",
                                      cuda_available_value = cuda_available(),
                                      faiss_gpu_available_value = faiss_gpu_available(),
                                      cuvs_available_value = cuvs_available()) {
  effective <- effective %||% list()
  backend_policy <- backend_policy %||% kmeans_auto_backend_policy(n, p, centers)
  list(
    policy = "static_shape_center_backend_selector",
    slow_tuning = FALSE,
    requested_backend = requested_backend,
    predicted_backend = resolved_backend,
    resolved_backend = resolved_backend,
    n = as.integer(n),
    p = as.integer(p),
    centers = as.integer(centers),
    work = as.numeric(backend_policy$work %||% (as.double(n) * as.double(p) * as.double(centers))),
    nbytes = as.numeric(backend_policy$nbytes %||% (as.double(n) * as.double(p) * 8)),
    n_per_center = as.numeric(backend_policy$n_per_center %||% (as.double(n) / as.double(centers))),
    backend_policy_reason = backend_policy$reason %||% NA_character_,
    backend_policy_prefer_cuda = isTRUE(backend_policy$prefer_cuda),
    cuda_available = isTRUE(cuda_available_value),
    faiss_gpu_available = isTRUE(faiss_gpu_available_value),
    cuvs_available = isTRUE(cuvs_available_value),
    effective_max_iter = as.integer(effective$max_iter %||% NA_integer_),
    effective_n_init = as.integer(effective$n_init %||% NA_integer_),
    effective_tol = as.numeric(effective$tol %||% NA_real_),
    tuning = normalize_kmeans_tuning(tuning)
  )
}

kmeans_auto_backend_policy <- function(n, p, centers) {
  work_threshold <- 1e8
  nbytes_threshold <- 256 * 1024^2
  large_n_threshold <- 50000
  large_p_threshold <- 128
  if (is.null(n) || is.null(p) || is.null(centers)) {
    return(list(
      prefer_cuda = TRUE,
      reason = "unknown_shape",
      work = NA_real_,
      nbytes = NA_real_,
      n_per_center = NA_real_,
      work_threshold = work_threshold,
      nbytes_threshold = nbytes_threshold,
      large_n_threshold = large_n_threshold,
      large_p_threshold = large_p_threshold
    ))
  }
  n <- suppressWarnings(as.double(n))
  p <- suppressWarnings(as.double(p))
  centers <- suppressWarnings(as.double(centers))
  if (length(n) != 1L || length(p) != 1L || length(centers) != 1L ||
      !is.finite(n) || !is.finite(p) || !is.finite(centers) ||
      n <= 0 || p <= 0 || centers <= 0) {
    return(list(
      prefer_cuda = TRUE,
      reason = "invalid_shape_assume_cuda_capable",
      work = NA_real_,
      nbytes = NA_real_,
      n_per_center = NA_real_,
      work_threshold = work_threshold,
      nbytes_threshold = nbytes_threshold,
      large_n_threshold = large_n_threshold,
      large_p_threshold = large_p_threshold
    ))
  }
  work <- n * p * centers
  nbytes <- n * p * 8
  n_per_center <- n / centers
  prefer <- work >= work_threshold ||
    nbytes >= nbytes_threshold ||
    (n >= large_n_threshold && p >= large_p_threshold)
  reason <- if (work >= work_threshold) {
    "work_at_least_1e8"
  } else if (nbytes >= nbytes_threshold) {
    "input_at_least_256MiB"
  } else if (n >= large_n_threshold && p >= large_p_threshold) {
    "large_high_dimensional_input"
  } else {
    "small_cpu_preferred"
  }
  list(
    prefer_cuda = isTRUE(prefer),
    reason = reason,
    work = as.numeric(work),
    nbytes = as.numeric(nbytes),
    n_per_center = as.numeric(n_per_center),
    work_threshold = work_threshold,
    nbytes_threshold = nbytes_threshold,
    large_n_threshold = large_n_threshold,
    large_p_threshold = large_p_threshold
  )
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
                            allow_cuvs_fallback = TRUE,
                            requested_backend = "cuda",
                            resolved_backend = "cuda") {
  faiss_error <- NULL
  if (isTRUE(faiss_gpu_available())) {
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
      return(finish_fast_kmeans(
        out,
        backend = "cuda_faiss",
        init = init,
        tuning_metadata = tuning_metadata,
        requested_backend = requested_backend,
        resolved_backend = resolved_backend
      ))
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
    return(finish_fast_kmeans(
      out,
      backend = "cuda_cuvs",
      init = init,
      tuning_metadata = tuning_metadata,
      requested_backend = requested_backend,
      resolved_backend = resolved_backend
    ))
  }

  cuvs_note <- if (isTRUE(cuvs_available()) && isTRUE(cuda_available())) {
    "direct cuVS is available but was not used"
  } else {
    "direct cuVS is unavailable"
  }
  if (is.null(faiss_error)) {
    faiss_error <- "FAISS GPU k-means was not attempted because FAISS GPU support is unavailable"
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
  if (!isTRUE(faiss_gpu_available())) {
    stop(
      "FAISS GPU k-means requires faissR to be built with FAISS GPU headers.",
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

finish_fast_kmeans <- function(out,
                               backend,
                               init,
                               tuning_metadata = NULL,
                               requested_backend = NULL,
                               resolved_backend = NULL) {
  out$cluster <- as.integer(out$cluster)
  out$centers <- unname(as.matrix(out$centers))
  out$withinss <- as.numeric(out$withinss)
  out$tot.withinss <- as.numeric(out$tot.withinss)
  out$size <- as.integer(out$size)
  out$iter <- as.integer(out$iter)
  out$backend <- backend
  if (is.null(out$parameters)) out$parameters <- list()
  out$parameters$init <- init
  if (!is.null(requested_backend)) out$parameters$requested_backend <- requested_backend
  if (!is.null(resolved_backend)) out$parameters$resolved_backend <- resolved_backend
  if (!is.null(tuning_metadata)) out$parameters$tuning <- tuning_metadata
  effective_max_iter <- out$parameters$max_iter %||%
    out$parameters$tuning$effective$max_iter %||%
    out$parameters$tuning$effective_max_iter %||%
    NA_integer_
  out$hit_max_iter <- kmeans_hit_max_iter(out$iter, effective_max_iter)
  out$converged <- if (is.na(out$hit_max_iter)) NA else !isTRUE(out$hit_max_iter)
  out$parameters$hit_max_iter <- out$hit_max_iter
  out$parameters$converged <- out$converged
  class(out) <- c("faissR_kmeans", "kmeans")
  out
}

kmeans_hit_max_iter <- function(iter, max_iter) {
  iter <- suppressWarnings(as.integer(iter))
  max_iter <- suppressWarnings(as.integer(max_iter))
  if (length(iter) != 1L || length(max_iter) != 1L ||
      is.na(iter) || is.na(max_iter) || max_iter < 1L) {
    return(NA)
  }
  iter >= max_iter
}

#' @export
print.faissR_kmeans <- function(x, ...) {
  params <- x$parameters %||% list()
  tuning <- params$tuning %||% list()
  effective <- tuning$effective %||% list()
  cat("faissR k-means\n")
  cat("  backend: ", x$backend %||% NA_character_, "\n", sep = "")
  if (!is.null(params$requested_backend)) {
    cat("  requested backend: ", params$requested_backend, "\n", sep = "")
  }
  if (!is.null(params$resolved_backend)) {
    cat("  resolved backend: ", params$resolved_backend, "\n", sep = "")
  }
  cat("  clusters: ", length(x$size), "\n", sep = "")
  cat("  observations: ", length(x$cluster), "\n", sep = "")
  cat("  iterations: ", x$iter %||% NA_integer_, "\n", sep = "")
  if (!is.null(x$converged) && !is.na(x$converged)) {
    cat("  converged before max_iter: ", if (isTRUE(x$converged)) "yes" else "no", "\n", sep = "")
  }
  if (!is.null(x$tot.withinss)) {
    cat("  total withinss: ", format(x$tot.withinss, digits = 4), "\n", sep = "")
  }
  if (!is.null(tuning$policy)) {
    cat("  tuning: ", tuning$policy, "\n", sep = "")
  }
  max_iter <- effective$max_iter %||% params$max_iter
  n_init <- effective$n_init %||% params$n_init
  tol <- effective$tol %||% params$tol
  if (!is.null(max_iter) || !is.null(n_init) || !is.null(tol)) {
    cat(
      "  effective: max_iter=",
      max_iter %||% NA_integer_,
      ", n_init=",
      n_init %||% NA_integer_,
      ", tol=",
      format(tol %||% NA_real_, digits = 4),
      "\n",
      sep = ""
    )
  }
  invisible(x)
}

normalize_kmeans_tuning <- function(tuning) {
  tuning <- normalize_scalar_choice_arg(
    tuning,
    arg = "tuning",
    default = "auto",
    formal_choices = c("auto", "fixed", "off", "none")
  )
  if (is.na(tuning) || !nzchar(tuning)) tuning <- "auto"
  tuning <- tolower(tuning)
  if (!tuning %in% c("auto", "fixed", "off", "none")) {
    stop("`tuning` must be one of \"auto\", \"fixed\", \"off\", or \"none\".", call. = FALSE)
  }
  tuning
}

normalize_kmeans_init <- function(init) {
  init <- normalize_scalar_choice_arg(
    init,
    arg = "init",
    default = "kmeans++",
    formal_choices = c("kmeans++", "random")
  )
  if (is.na(init) || !nzchar(init)) init <- "kmeans++"
  init <- trimws(init)
  if (!init %in% c("kmeans++", "random")) {
    stop("`init` must be one of \"kmeans++\" or \"random\".", call. = FALSE)
  }
  init
}

kmeans_auto_params <- function(n, p, centers, tuning = "auto") {
  tuning <- normalize_kmeans_tuning(tuning)
  work <- as.double(n) * as.double(p) * as.double(centers)
  high_dim <- p >= 256L
  large_n <- n >= 100000L
  many_centers <- centers >= 100L
  n_per_center <- as.double(n) / as.double(centers)
  small_many_centers <- many_centers && n <= 50000L && work <= 2e8 && n_per_center >= 20
  if (!identical(tuning, "auto")) {
    return(list(
      policy = tuning,
      max_iter = 100L,
      n_init = 1L,
      tol = 1e-4,
      work = as.numeric(work),
      n_per_center = as.numeric(n_per_center),
      high_dim = isTRUE(high_dim),
      large_n = isTRUE(large_n),
      many_centers = isTRUE(many_centers),
      small_many_centers = isTRUE(small_many_centers),
      rule = "fixed_defaults"
    ))
  }
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

normalize_kmeans_positive_int <- function(x, fallback, arg = "value") {
  if (is.character(x) && length(x) == 1L && identical(tolower(x), "auto")) {
    return(as.integer(fallback))
  }
  normalize_positive_int(x, fallback, arg = arg)
}

normalize_kmeans_seed <- function(seed) {
  seed <- suppressWarnings(as.numeric(seed))
  if (length(seed) != 1L || is.na(seed) || !is.finite(seed) ||
      abs(seed - round(seed)) > sqrt(.Machine$double.eps)) {
    stop("`seed` must be a single finite integer.", call. = FALSE)
  }
  as.integer(round(seed))
}

normalize_kmeans_streaming_batch_size <- function(streaming_batch_size) {
  normalize_kmeans_whole_number(
    streaming_batch_size,
    "streaming_batch_size",
    min_value = 0L,
    message = "`streaming_batch_size` must be a single non-negative integer."
  )
}

normalize_kmeans_tol <- function(x, fallback) {
  if (is.character(x) && length(x) == 1L && identical(tolower(x), "auto")) {
    return(as.numeric(fallback))
  }
  x <- suppressWarnings(as.numeric(x))
  if (length(x) != 1L || is.na(x) || !is.finite(x) || x < 0) {
    stop("`tol` must be `auto` or a single non-negative finite number.", call. = FALSE)
  }
  as.numeric(x)
}

normalize_positive_int <- function(x, fallback, arg = "value") {
  normalize_kmeans_whole_number(x, arg, min_value = 1L)
}

normalize_kmeans_whole_number <- function(x,
                                          arg,
                                          min_value = 1L,
                                          message = NULL) {
  value <- suppressWarnings(as.numeric(x))
  if (length(value) != 1L || is.na(value) || !is.finite(value) ||
      value < min_value || abs(value - round(value)) > sqrt(.Machine$double.eps)) {
    if (is.null(message)) {
      if (min_value <= 0L) {
        message <- paste0("`", arg, "` must be a single non-negative integer.")
      } else {
        message <- paste0("`", arg, "` must be a single positive integer.")
      }
    }
    stop(message, call. = FALSE)
  }
  as.integer(round(value))
}

kmeans_value_source <- function(x) {
  if (is.character(x) && length(x) == 1L && identical(tolower(x), "auto")) {
    return("auto")
  }
  "explicit"
}
