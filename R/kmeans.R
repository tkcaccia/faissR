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
#'   deterministic shape-aware default computed by the compiled C++ tuning
#'   helper.
#' @param n_init Number of random restarts where supported, or `"auto"` for a
#'   deterministic shape-aware default computed by the compiled C++ tuning
#'   helper.
#' @param tol Single non-negative finite relative convergence tolerance where
#'   supported, or `"auto"` for a deterministic shape-aware default computed by
#'   the compiled C++ tuning helper.
#' @param seed Random seed for CPU/statistics and FAISS paths. The current
#'   direct cuVS C API path does not expose an explicit seed in the stable
#'   params structure.
#' @param n_threads Number of CPU threads for FAISS/statistics paths.
#' @param streaming_batch_size cuVS host-data streaming batch size. `0` lets
#'   cuVS choose its default.
#' @param init Initialization method, `"kmeans++"` or `"random"` where
#'   supported.
#' @param tuning Tuning policy. `"auto"` uses deterministic C++ rules based on
#'   `nrow(data)`, `ncol(data)`, and `centers` without running pilot searches.
#'   Small many-cluster jobs can use extra restarts when `n / centers` remains
#'   large enough, and cheap many-cluster jobs with few observations per center
#'   also use a small multistart budget for stability; large or
#'   high-dimensional jobs use cheaper iteration and tolerance defaults.
#'   `centers = 1` uses the exact column-mean solution for every backend request
#'   with `max_iter = 1`, `n_init = 1`, and `tol = 0`, records
#'   `single_cluster_exact_mean`, and stays on CPU because no iterative
#'   k-means backend can improve that solution. `centers = nrow(data)` uses the
#'   exact singleton assignment for every backend request and records
#'   `singleton_exact_identity`; explicit CUDA requests are recorded as
#'   resolved by the exact faissR trivial route rather than launching GPU work.
#'   `"fixed"`, `"off"`, and `"none"` use the historical fixed defaults unless
#'   `max_iter`, `n_init`, or `tol` are explicitly supplied.
#' @return A list with `cluster`, `centers`, `withinss`, `tot.withinss`,
#'   `size`, `iter`, `converged`, `hit_max_iter`, `backend`, and
#'   `parameters`. `backend` records the
#'   implementation that actually ran, while `parameters$requested_backend` and
#'   `parameters$resolved_backend` record the public backend request and device
#'   policy result. `parameters$tuning` records the deterministic k-means policy,
#'   stable `rule` label, shape metadata, and whether `max_iter`, `n_init`, and
#'   `tol` were auto-selected or supplied explicitly. The auto parameter,
#'   backend-policy, and final auto backend-selection rules are computed by
#'   compiled C++ helpers and record `tuning_source = "cpp"`.
#'   `parameters$tuning$rule_detail` records the exact
#'   `n`/`p`/`centers`/work values used to choose the rule.
#'   `parameters$tuning$effective` records
#'   the final values used after explicit overrides and `"auto"` defaults have
#'   been resolved; `parameters$tuning$effective_max_iter`,
#'   `parameters$tuning$effective_n_init`, and
#'   `parameters$tuning$effective_tol` expose the same values as flat fields for
#'   benchmark summaries. `parameters$tuning$backend_policy` records the
#'   deterministic shape rule used by `backend = "auto"` to decide whether CUDA
#'   has enough estimated work or float32 transfer size to offset transfer
#'   overhead. The policy keeps `nbytes` as the ordinary R double input
#'   footprint and records `gpu_transfer_nbytes` for the float32 data passed to
#'   FAISS/cuVS. The default thresholds can be overridden without adding pilot
#'   work by setting
#'   `options(faissR.kmeans_cuda_work_threshold = ...)`,
#'   `options(faissR.kmeans_cuda_nbytes_threshold = ...)`,
#'   `options(faissR.kmeans_cuda_large_n_threshold = ...)`, or
#'   `options(faissR.kmeans_cuda_large_p_threshold = ...)`, or
#'   `options(faissR.kmeans_cuda_min_n_per_center = ...)`.
#'   `parameters$tuning$selection` stores the static no-pilot backend and
#'   effective-parameter decision used for benchmark auditing, including
#'   `explicit_backend` and `backend_decision` fields that distinguish an
#'   explicit `"cpu"`/`"cuda"` request from an automatic shape-policy choice,
#'   plus `runtime_decision` and CUDA k-means capability flags from the C++
#'   selector.
#'   CUDA runs also record `parameters$cuda_provider_selection` as `"faiss_gpu"`,
#'   `"direct_cuvs"`, or `"direct_cuvs_after_faiss_gpu_unavailable_or_failed"`;
#'   `parameters$backend_resolution_note` describes the provider route, and
#'   `parameters$faiss_gpu_error` is present when direct cuVS was used after a
#'   FAISS GPU route was unavailable or failed.
#'   `hit_max_iter`
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
  auto_params$selection <- kmeans_selection_metadata(
    requested_backend = requested_backend,
    resolved_backend = NULL,
    n = nrow(x),
    p = ncol(x),
    centers = centers,
    effective = auto_params$effective,
    backend_policy = NULL,
    tuning = tuning
  )
  auto_params$backend_policy <- auto_params$selection$backend_policy
  auto_params$selection$backend_policy <- NULL
  backend <- auto_params$selection$resolved_backend

  if (centers == 1L) {
    return(finish_trivial_one_cluster_kmeans(
      x = x,
      tuning_metadata = auto_params,
      requested_backend = requested_backend,
      resolved_backend = "trivial"
    ))
  }
  if (centers == nrow(x)) {
    return(finish_trivial_singleton_kmeans(
      x = x,
      tuning_metadata = auto_params,
      requested_backend = requested_backend,
      resolved_backend = "trivial"
    ))
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

  set_rng_seed(seed)
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
  kmeans_auto_backend_selection(
    requested_backend = backend,
    n = n,
    p = p,
    centers = centers,
    cuda_available_value = cuda_available_value,
    faiss_gpu_available_value = faiss_gpu_available_value,
    cuvs_available_value = cuvs_available_value
  )$resolved_backend
}

kmeans_auto_prefers_cuda <- function(n, p, centers) {
  isTRUE(kmeans_auto_backend_policy(n, p, centers)$prefer_cuda)
}

kmeans_shape_int <- function(x) {
  if (is.null(x)) {
    return(NA_integer_)
  }
  x <- suppressWarnings(as.integer(x[1L]))
  if (length(x) != 1L || is.na(x)) {
    return(NA_integer_)
  }
  x
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
  kmeans_auto_backend_selection(
    requested_backend = requested_backend,
    n = n,
    p = p,
    centers = centers,
    effective = effective,
    tuning = tuning,
    cuda_available_value = cuda_available_value,
    faiss_gpu_available_value = faiss_gpu_available_value,
    cuvs_available_value = cuvs_available_value
  )
}

kmeans_auto_backend_policy <- function(n, p, centers) {
  kmeans_auto_backend_policy_options(n, p, centers)$policy
}

kmeans_auto_backend_policy_options <- function(n, p, centers) {
  work_threshold <- kmeans_option_number("kmeans_cuda_work_threshold", 1e8, min_value = 1)
  nbytes_threshold <- kmeans_option_number("kmeans_cuda_nbytes_threshold", 256 * 1024^2, min_value = 1)
  large_n_threshold <- kmeans_option_integer("kmeans_cuda_large_n_threshold", 50000L, min_value = 1L)
  large_p_threshold <- kmeans_option_integer("kmeans_cuda_large_p_threshold", 128L, min_value = 1L)
  min_n_per_center <- kmeans_option_number("kmeans_cuda_min_n_per_center", 20, min_value = 1)
  list(
    n = kmeans_shape_int(n),
    p = kmeans_shape_int(p),
    centers = kmeans_shape_int(centers),
    work_threshold = work_threshold,
    nbytes_threshold = nbytes_threshold,
    large_n_threshold = large_n_threshold,
    large_p_threshold = large_p_threshold,
    min_n_per_center = min_n_per_center,
    policy = kmeans_auto_backend_policy_cpp(
      kmeans_shape_int(n),
      kmeans_shape_int(p),
      kmeans_shape_int(centers),
      work_threshold,
      nbytes_threshold,
      large_n_threshold,
      large_p_threshold,
      min_n_per_center
    )
  )
}

kmeans_auto_backend_selection <- function(requested_backend,
                                          n,
                                          p,
                                          centers,
                                          effective = NULL,
                                          tuning = "auto",
                                          cuda_available_value = cuda_available(),
                                          faiss_gpu_available_value = faiss_gpu_available(),
                                          cuvs_available_value = cuvs_available()) {
  requested_backend <- normalize_public_backend_arg(requested_backend)
  effective <- effective %||% list()
  opts <- kmeans_auto_backend_policy_options(n, p, centers)
  kmeans_auto_select_backend_cpp(
    requested_backend,
    opts$n,
    opts$p,
    opts$centers,
    opts$work_threshold,
    opts$nbytes_threshold,
    opts$large_n_threshold,
    opts$large_p_threshold,
    opts$min_n_per_center,
    isTRUE(cuda_available_value),
    isTRUE(faiss_gpu_available_value),
    isTRUE(cuvs_available_value),
    as.integer(effective$max_iter %||% NA_integer_),
    as.integer(effective$n_init %||% NA_integer_),
    as.numeric(effective$tol %||% NA_real_),
    normalize_kmeans_tuning(tuning)
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
    attempt <- tryCatch(
      list(result = run_faiss_gpu_kmeans(
        x = x,
        centers = centers,
        max_iter = max_iter,
        n_init = n_init,
        tol = tol,
        seed = seed,
        init = init,
        tuning_metadata = tuning_metadata
      ), error = NULL),
      error = function(e) {
        list(result = NULL, error = conditionMessage(e))
      }
    )
    out <- attempt$result
    faiss_error <- attempt$error
    if (!is.null(out)) {
      out <- finish_fast_kmeans(
        out,
        backend = "cuda_faiss",
        init = init,
        tuning_metadata = tuning_metadata,
        requested_backend = requested_backend,
        resolved_backend = resolved_backend
      )
      out$parameters$cuda_provider_selection <- "faiss_gpu"
      out$parameters$backend_resolution_note <- "FAISS GPU k-means was used within the requested CUDA backend."
      return(out)
    }
  }

  if (allow_cuvs_fallback && isTRUE(cuvs_available()) && isTRUE(cuda_available())) {
    if (is.null(faiss_error) && !isTRUE(faiss_gpu_available())) {
      faiss_error <- "FAISS GPU k-means was not attempted because FAISS GPU support is unavailable"
    }
    out <- kmeans_cuvs_cpp(
      x,
      as.integer(centers),
      as.integer(max_iter),
      as.integer(n_init),
      as.numeric(tol),
      as.integer(streaming_batch_size),
      identical(init, "kmeans++")
    )
    out <- finish_fast_kmeans(
      out,
      backend = "cuda_cuvs",
      init = init,
      tuning_metadata = tuning_metadata,
      requested_backend = requested_backend,
      resolved_backend = resolved_backend
    )
    out$parameters$cuda_provider_selection <- if (is.null(faiss_error)) {
      "direct_cuvs"
    } else {
      "direct_cuvs_after_faiss_gpu_unavailable_or_failed"
    }
    if (!is.null(faiss_error)) {
      out$parameters$faiss_gpu_error <- faiss_error
      out$parameters$backend_resolution_note <- paste(
        "Direct cuVS k-means was used within the requested CUDA backend.",
        "FAISS GPU status:",
        faiss_error
      )
    }
    return(out)
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

finish_trivial_one_cluster_kmeans <- function(x,
                                              tuning_metadata = NULL,
                                              requested_backend = "auto",
                                              resolved_backend = "cpu") {
  center <- matrix(colMeans(x), nrow = 1L)
  row_offsets <- sweep(x, 2L, center[1L, ], "-")
  within <- sum(row_offsets * row_offsets)
  max_iter <- tuning_metadata$effective$max_iter %||%
    tuning_metadata$effective_max_iter %||%
    NA_integer_
  n_init <- tuning_metadata$effective$n_init %||%
    tuning_metadata$effective_n_init %||%
    NA_integer_
  tol <- tuning_metadata$effective$tol %||%
    tuning_metadata$effective_tol %||%
    NA_real_
  out <- list(
    cluster = rep.int(1L, nrow(x)),
    centers = center,
    withinss = as.numeric(within),
    tot.withinss = as.numeric(within),
    size = as.integer(nrow(x)),
    iter = 0L,
    hit_max_iter = FALSE,
    backend = "trivial",
    backend_library = "faissR",
    parameters = list(
      centers = 1L,
      max_iter = as.integer(max_iter),
      n_init = as.integer(n_init),
      tol = as.numeric(tol),
      seed = NA_integer_,
      n_threads = 1L,
      init = "exact_mean",
      requested_backend = requested_backend,
      resolved_backend = resolved_backend,
      backend_resolution_note = "Exact one-cluster solution; no iterative CPU or CUDA backend was launched.",
      tuning = tuning_metadata,
      exact_trivial_solution = TRUE
    )
  )
  out$converged <- TRUE
  out$parameters$hit_max_iter <- out$hit_max_iter
  out$parameters$converged <- out$converged
  class(out) <- c("faissR_kmeans", "kmeans")
  out
}

finish_trivial_singleton_kmeans <- function(x,
                                            tuning_metadata = NULL,
                                            requested_backend = "auto",
                                            resolved_backend = "cpu") {
  max_iter <- tuning_metadata$effective$max_iter %||%
    tuning_metadata$effective_max_iter %||%
    NA_integer_
  n_init <- tuning_metadata$effective$n_init %||%
    tuning_metadata$effective_n_init %||%
    NA_integer_
  tol <- tuning_metadata$effective$tol %||%
    tuning_metadata$effective_tol %||%
    NA_real_
  out <- list(
    cluster = seq_len(nrow(x)),
    centers = unname(as.matrix(x)),
    withinss = rep.int(0, nrow(x)),
    tot.withinss = 0,
    size = rep.int(1L, nrow(x)),
    iter = 0L,
    hit_max_iter = FALSE,
    backend = "trivial",
    backend_library = "faissR",
    parameters = list(
      centers = as.integer(nrow(x)),
      max_iter = as.integer(max_iter),
      n_init = as.integer(n_init),
      tol = as.numeric(tol),
      seed = NA_integer_,
      n_threads = 1L,
      init = "exact_singletons",
      requested_backend = requested_backend,
      resolved_backend = resolved_backend,
      backend_resolution_note = "Exact singleton solution; no iterative CPU or CUDA backend was launched.",
      tuning = tuning_metadata,
      exact_trivial_solution = TRUE
    )
  )
  out$converged <- TRUE
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
  kmeans_auto_params_cpp(
    kmeans_shape_int(n),
    kmeans_shape_int(p),
    kmeans_shape_int(centers),
    tuning
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

kmeans_option_number <- function(name, default, min_value = -Inf) {
  value <- suppressWarnings(as.numeric(faissr_option(name, default)))
  if (length(value) != 1L || is.na(value) || !is.finite(value) || value < min_value) {
    return(default)
  }
  as.numeric(value)
}

kmeans_option_integer <- function(name, default, min_value = 1L) {
  value <- kmeans_option_number(name, default, min_value = min_value)
  as.integer(round(value))
}
