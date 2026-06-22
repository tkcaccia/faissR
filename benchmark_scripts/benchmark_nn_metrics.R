#!/usr/bin/env Rscript

`%||%` <- function(x, y) if (is.null(x) || length(x) == 0L || is.na(x[[1L]])) y else x

parse_args <- function(args = commandArgs(trailingOnly = TRUE)) {
  out <- list()
  for (arg in args) {
    if (!startsWith(arg, "--")) next
    kv <- strsplit(sub("^--", "", arg), "=", fixed = TRUE)[[1L]]
    key <- kv[[1L]]
    value <- if (length(kv) > 1L) paste(kv[-1L], collapse = "=") else "TRUE"
    out[[key]] <- value
  }
  out
}

split_arg <- function(x, default) {
  trimws(strsplit(x %||% default, ",", fixed = TRUE)[[1L]])
}

canonical_metric_key <- function(metric) {
  aliases <- c(
    euclidean = "euclidean",
    l2 = "euclidean",
    cosine = "cosine",
    cos = "cosine",
    correlation = "correlation",
    cor = "correlation",
    pearson = "correlation",
    inner_product = "inner_product",
    innerproduct = "inner_product",
    ip = "inner_product",
    dot = "inner_product",
    dot_product = "inner_product",
    dotproduct = "inner_product"
  )
  key <- tolower(trimws(as.character(metric)))
  key <- gsub("[[:space:]-]+", "_", key)
  out <- unname(aliases[key])
  out[is.na(out)] <- key[is.na(out)]
  out
}

canonical_metric_values <- function(metrics) {
  metrics <- canonical_metric_key(metrics)
  metrics <- metrics[metrics %in% c("euclidean", "cosine", "correlation", "inner_product")]
  unique(metrics)
}

validate_metric_values <- function(metrics, arg_name = "metrics") {
  raw <- trimws(as.character(metrics))
  raw <- raw[nzchar(raw)]
  metrics <- unique(canonical_metric_key(raw))
  valid <- c("euclidean", "cosine", "correlation", "inner_product")
  invalid <- metrics[!metrics %in% valid]
  if (length(invalid)) {
    stop(
      "`", arg_name, "` must contain only faissR public metrics: ",
      paste(valid, collapse = ", "),
      ". Invalid value(s): ",
      paste(invalid, collapse = ", "),
      ".",
      call. = FALSE
    )
  }
  if (!length(metrics)) {
    stop("`", arg_name, "` must contain at least one metric.", call. = FALSE)
  }
  metrics
}

default_nn_metric_values <- function() {
  c("euclidean", "cosine", "correlation", "inner_product")
}

default_nn_method_values <- function() {
  c(
    "auto", "exact", "flat", "bruteforce", "grid", "vptree", "sparse",
    "hnsw", "ivf", "ivfpq", "nsg", "nndescent", "cagra"
  )
}

default_nn_backend_values <- function() {
  c("auto", "cpu", "cuda")
}

validate_backend_values <- function(backends, arg_name = "backends") {
  backends <- unique(trimws(as.character(backends)))
  backends <- backends[nzchar(backends)]
  valid <- default_nn_backend_values()
  invalid <- backends[!backends %in% valid]
  if (length(invalid)) {
    stop(
      "`", arg_name, "` must contain only: ",
      paste(valid, collapse = ", "),
      ". Invalid value(s): ",
      paste(invalid, collapse = ", "),
      ".",
      call. = FALSE
    )
  }
  if (!length(backends)) {
    stop("`", arg_name, "` must contain at least one backend.", call. = FALSE)
  }
  backends
}

validate_dataset_values <- function(datasets, valid_datasets, arg_name = "datasets") {
  datasets <- unique(trimws(as.character(datasets)))
  datasets <- datasets[nzchar(datasets)]
  valid_datasets <- unique(trimws(as.character(valid_datasets)))
  valid_datasets <- valid_datasets[nzchar(valid_datasets)]
  invalid <- datasets[!datasets %in% valid_datasets]
  if (length(invalid)) {
    stop(
      "`", arg_name, "` must contain only available benchmark datasets: ",
      paste(valid_datasets, collapse = ", "),
      ". Invalid value(s): ",
      paste(invalid, collapse = ", "),
      ".",
      call. = FALSE
    )
  }
  if (!length(datasets)) {
    stop("`", arg_name, "` must contain at least one dataset.", call. = FALSE)
  }
  datasets
}

default_nn_k_values <- function() {
  c(5L, 10L, 15L, 50L, 100L)
}

as_int_vec_arg <- function(x, default) {
  value <- suppressWarnings(as.integer(x %||% default))
  value <- value[!is.na(value) & value > 0L]
  if (!length(value)) suppressWarnings(as.integer(default)) else unique(value)
}

required_positive_int_values <- function(x, arg) {
  raw <- trimws(as.character(x))
  raw <- raw[nzchar(raw)]
  value <- suppressWarnings(as.numeric(raw))
  invalid <- raw[
    is.na(value) | !is.finite(value) | value < 1L |
      abs(value - round(value)) > sqrt(.Machine$double.eps)
  ]
  if (length(invalid)) {
    stop(
      "`", arg, "` must contain only positive integers. Invalid value(s): ",
      paste(invalid, collapse = ", "),
      ".",
      call. = FALSE
    )
  }
  value <- unique(as.integer(round(value)))
  if (!length(value)) {
    stop("`", arg, "` must contain at least one positive integer.", call. = FALSE)
  }
  value
}

required_positive_int_arg <- function(x, arg) {
  value <- suppressWarnings(as.numeric(x))
  if (length(value) != 1L || is.na(value) || !is.finite(value) || value < 1L ||
      abs(value - round(value)) > sqrt(.Machine$double.eps)) {
    stop("`", arg, "` must be a positive integer.", call. = FALSE)
  }
  as.integer(round(value))
}

required_positive_numeric_arg <- function(x, arg) {
  value <- suppressWarnings(as.numeric(x))
  if (length(value) != 1L || is.na(value) || !is.finite(value) || value <= 0) {
    stop("`", arg, "` must be a positive numeric value.", call. = FALSE)
  }
  value
}

required_probability_arg <- function(x, arg) {
  value <- suppressWarnings(as.numeric(x))
  if (length(value) != 1L || is.na(value) || !is.finite(value) ||
      value < 0 || value > 1) {
    stop("`", arg, "` must be a numeric value between 0 and 1.", call. = FALSE)
  }
  value
}

configure_threads <- function(n_threads) {
  value <- as.character(as.integer(n_threads))
  Sys.setenv(
    OMP_NUM_THREADS = value,
    OPENBLAS_NUM_THREADS = value,
    MKL_NUM_THREADS = value,
    VECLIB_MAXIMUM_THREADS = value,
    NUMEXPR_NUM_THREADS = value,
    RCPP_PARALLEL_NUM_THREADS = value
  )
  options(Ncpus = as.integer(n_threads))
}

configure_native_libs <- function() {
  env_dir <- Sys.getenv("FAISSR_ENV_DIR", unset = "")
  cuda_home <- Sys.getenv("FAISSR_CUDA_LIB_DIR", Sys.getenv("CUDA_HOME", unset = ""))
  cuda_lib <- if (nzchar(cuda_home) && !grepl("/lib$", cuda_home)) {
    file.path(cuda_home, "targets/x86_64-linux/lib")
  } else {
    cuda_home
  }
  pieces <- c(
    if (nzchar(env_dir)) file.path(env_dir, "lib") else "",
    if (nzchar(env_dir)) file.path(env_dir, "targets/x86_64-linux/lib") else "",
    cuda_lib,
    Sys.getenv("LD_LIBRARY_PATH", unset = "")
  )
  pieces <- unique(pieces[nzchar(pieces)])
  if (nzchar(env_dir)) Sys.setenv(CONDA_PREFIX = env_dir)
  if (length(pieces)) Sys.setenv(LD_LIBRARY_PATH = paste(pieces, collapse = ":"))
}

read_peak_rss_gb <- function() {
  status <- "/proc/self/status"
  if (!file.exists(status)) return(NA_real_)
  line <- grep("^VmHWM:", readLines(status, warn = FALSE), value = TRUE)
  if (!length(line)) return(NA_real_)
  kb <- suppressWarnings(as.numeric(gsub("[^0-9.]", "", line[[1L]])))
  kb / 1024^2
}

coerce_matrix <- function(x) {
  if (inherits(x, "Matrix")) x <- as.matrix(x)
  if (is.data.frame(x)) x <- data.matrix(x)
  if (!is.matrix(x)) x <- as.matrix(x)
  storage.mode(x) <- "double"
  x
}

dataset_index <- function(data_root) {
  data.frame(
    dataset = c(
      "COIL20",
      "USPS",
      "FashionMNIST",
      "FlowRepository_FR-FCM-ZYRM_files",
      "flow18",
      "MNIST",
      "imagenet",
      "MetRef",
      "mass41"
    ),
    path = file.path(
      data_root,
      c(
        "COIL20/COIL20.RData",
        "USPS/USPS.RData",
        "FashionMNIST/FashionMNIST.RData",
        "FlowRepository_FR-FCM-ZYRM_files/van_unen_FR-FCM-ZYRM.RData",
        "flow18/flow18.RData",
        "MNIST/MNIST.RData",
        "imagenet/imagenet.RData",
        "MetRef/MetRef.RData",
        "mass41/mass41.RData"
      )
    ),
    stringsAsFactors = FALSE
  )
}

make_simulated_dataset <- function(name, seed) {
  set.seed(seed)
  if (identical(name, "SimulatedUniform2D")) {
    return(list(data = matrix(runif(2000000), ncol = 2), labels = NULL))
  }
  if (identical(name, "SimulatedUniform3D")) {
    return(list(data = matrix(runif(3000000), ncol = 3), labels = NULL))
  }
  if (identical(name, "SimulatedTiny3Clusters")) {
    centers <- rbind(c(-3, -3, 0, 0), c(0, 3, 3, 0), c(3, -2, 0, 3))
    n_each <- 80L
    data <- do.call(rbind, lapply(seq_len(nrow(centers)), function(i) {
      matrix(rnorm(n_each * ncol(centers), sd = 0.25), ncol = ncol(centers)) +
        matrix(centers[i, ], nrow = n_each, ncol = ncol(centers), byrow = TRUE)
    }))
    return(list(data = data, labels = rep(seq_len(nrow(centers)), each = n_each)))
  }
  NULL
}

load_dataset <- function(name, data_root, seed) {
  simulated <- make_simulated_dataset(name, seed)
  if (!is.null(simulated)) {
    simulated$data <- coerce_matrix(simulated$data)
    return(simulated)
  }
  index <- dataset_index(data_root)
  hit <- index[index$dataset == name, , drop = FALSE]
  if (!nrow(hit)) stop("Unknown dataset `", name, "`.", call. = FALSE)
  if (!file.exists(hit$path[[1L]])) stop("Dataset file does not exist: ", hit$path[[1L]], call. = FALSE)
  env <- new.env(parent = emptyenv())
  load(hit$path[[1L]], envir = env)
  if (!exists("dataset", envir = env, inherits = FALSE)) {
    stop("Dataset file must contain an object named `dataset`: ", hit$path[[1L]], call. = FALSE)
  }
  dataset <- get("dataset", envir = env, inherits = FALSE)
  if (!is.list(dataset) || is.null(dataset$data)) {
    stop("`dataset` must be a list containing `data`: ", hit$path[[1L]], call. = FALSE)
  }
  list(data = coerce_matrix(dataset$data), labels = if (is.null(dataset$labels)) NULL else dataset$labels)
}

with_elapsed_limit <- function(expr, timeout) {
  timeout <- suppressWarnings(as.numeric(timeout))
  if (length(timeout) == 1L && is.finite(timeout) && timeout > 0) {
    setTimeLimit(elapsed = timeout, transient = TRUE)
    on.exit(setTimeLimit(cpu = Inf, elapsed = Inf, transient = FALSE), add = TRUE)
  }
  force(expr)
}

metric_reference <- function(x, k, metric, quality_n, quality_max_ops, n_threads, seed) {
  n <- nrow(x)
  p <- ncol(x)
  full_ops <- as.double(n) * as.double(n) * as.double(p)
  if (n <= quality_n && full_ops <= quality_max_ops) {
    return(list(
      knn = faissR::nn(
        x,
        k = k,
        backend = "cpu",
        method = "exact",
        metric = metric,
        n_threads = n_threads
      ),
      rows = NULL,
      mode = "full"
    ))
  }
  sample_n <- min(as.integer(quality_n), n)
  sample_ops <- as.double(sample_n) * as.double(n) * as.double(p)
  if (sample_n < 1L || sample_ops > quality_max_ops) return(NULL)
  set.seed(seed + as.integer(k) + match(metric, c("euclidean", "cosine", "correlation", "inner_product")))
  rows <- sort(sample.int(n, sample_n))
  list(
    knn = faissR::nn(
      x,
      points = x[rows, , drop = FALSE],
      k = k,
      backend = "cpu",
      method = "exact",
      metric = metric,
      n_threads = n_threads
    ),
    rows = rows,
    mode = "sample"
  )
}

result_row <- function(dataset, n, p, backend, method, metric, k, cycle, n_threads,
                       status, error = NA_character_, elapsed_sec = NA_real_,
                       peak_rss_gb = NA_real_,
                       result_backend = NA_character_,
                       resolved_backend = NA_character_,
                       implementation_backend = NA_character_,
                       preflight_route = NA_character_,
                       exact = NA, recall_at_k = NA_real_,
                       median_recall_at_k = NA_real_,
                       min_recall_at_k = NA_real_,
                       recall_reference = NA_character_,
                       recall_query_n = NA_integer_,
                       expected_skip = FALSE,
                       capability_notes = NA_character_) {
  data.frame(
    dataset = dataset,
    n = as.integer(n),
    p = as.integer(p),
    backend = backend,
    method = method,
    metric = metric,
    k = as.integer(k),
    cycle = as.integer(cycle),
    n_threads = as.integer(n_threads),
    status = status,
    error = error,
    elapsed_sec = elapsed_sec,
    peak_rss_gb = peak_rss_gb,
    result_backend = result_backend,
    resolved_backend = resolved_backend,
    implementation_backend = implementation_backend,
    preflight_route = preflight_route,
    exact = exact,
    recall_at_k = recall_at_k,
    median_recall_at_k = median_recall_at_k,
    min_recall_at_k = min_recall_at_k,
    recall_reference = recall_reference,
    recall_query_n = as.integer(recall_query_n),
    expected_skip = isTRUE(expected_skip),
    capability_notes = capability_notes,
    stringsAsFactors = FALSE
  )
}

nn_implementation_backend <- function(out) {
  approx <- attr(out, "approximation") %||% list()
  cuvs <- attr(out, "cuvs") %||% list()
  faiss <- attr(out, "faiss") %||% list()
  attr(out, "resolved_backend") %||%
    approx$backend %||%
    cuvs$resolved_backend %||%
    faiss$backend %||%
    attr(out, "backend") %||%
    NA_character_
}

dominant_value <- function(x) {
  x <- as.character(x)
  x <- x[!is.na(x) & nzchar(x)]
  if (!length(x)) return(NA_character_)
  names(sort(table(x), decreasing = TRUE))[[1L]]
}

finite_median <- function(x) {
  x <- suppressWarnings(as.numeric(x))
  x <- x[is.finite(x)]
  if (!length(x)) return(NA_real_)
  stats::median(x)
}

finite_min <- function(x) {
  x <- suppressWarnings(as.numeric(x))
  x <- x[is.finite(x)]
  if (!length(x)) return(NA_real_)
  min(x)
}

finite_max <- function(x) {
  x <- suppressWarnings(as.numeric(x))
  x <- x[is.finite(x)]
  if (!length(x)) return(NA_real_)
  max(x)
}

summarize_nn_cycles <- function(ok) {
  parts <- split(ok, paste(ok$dataset, ok$backend, ok$method, ok$metric, ok$k, sep = "__"))
  summary <- lapply(parts, function(x) {
    data.frame(
      dataset = x$dataset[[1L]],
      backend = x$backend[[1L]],
      method = x$method[[1L]],
      metric = x$metric[[1L]],
      k = as.integer(x$k[[1L]]),
      n = as.integer(x$n[[1L]]),
      p = as.integer(x$p[[1L]]),
      n_threads = as.integer(x$n_threads[[1L]]),
      success_cycles = length(unique(x$cycle)),
      success_rows = nrow(x),
      median_elapsed_sec = finite_median(x$elapsed_sec),
      min_elapsed_sec = finite_min(x$elapsed_sec),
      max_elapsed_sec = finite_max(x$elapsed_sec),
      median_recall_at_k = finite_median(x$recall_at_k),
      min_recall_at_k = finite_min(x$recall_at_k),
      median_min_recall_at_k = finite_median(x$min_recall_at_k),
      min_min_recall_at_k = finite_min(x$min_recall_at_k),
      median_recall_query_n = finite_median(x$recall_query_n),
      exact = any(as.logical(x$exact), na.rm = TRUE),
      result_backend = dominant_value(x$result_backend),
      resolved_backend = dominant_value(x$resolved_backend),
      implementation_backend = dominant_value(x$implementation_backend),
      recall_reference = dominant_value(x$recall_reference),
      stringsAsFactors = FALSE
    )
  })
  out <- do.call(rbind, summary)
  out[order(out$dataset, out$backend, out$metric, out$k, out$median_elapsed_sec), , drop = FALSE]
}

recommend_nn_methods <- function(cycle_summary, recall_threshold) {
  out <- do.call(rbind, lapply(
    split(cycle_summary, paste(cycle_summary$dataset, cycle_summary$backend, cycle_summary$metric, cycle_summary$k, sep = "__")),
    function(x) {
      has_recall <- is.finite(x$median_recall_at_k)
      if (any(has_recall)) {
        candidates <- x[has_recall & x$median_recall_at_k >= recall_threshold, , drop = FALSE]
        if (nrow(candidates)) {
          candidates <- candidates[order(
            candidates$median_elapsed_sec,
            -ifelse(is.finite(candidates$median_recall_at_k), candidates$median_recall_at_k, -Inf),
            -ifelse(is.finite(candidates$min_recall_at_k), candidates$min_recall_at_k, -Inf),
            -ifelse(is.finite(candidates$median_min_recall_at_k), candidates$median_min_recall_at_k, -Inf)
          ), , drop = FALSE]
          candidates$recommendation_basis <- "fastest_at_recall_threshold"
          return(candidates[1L, , drop = FALSE])
        }
        candidates <- x[has_recall, , drop = FALSE]
        candidates <- candidates[order(
          -candidates$median_recall_at_k,
          -ifelse(is.finite(candidates$min_recall_at_k), candidates$min_recall_at_k, -Inf),
          -ifelse(is.finite(candidates$median_min_recall_at_k), candidates$median_min_recall_at_k, -Inf),
          candidates$median_elapsed_sec
        ), , drop = FALSE]
        candidates$recommendation_basis <- "best_recall_below_threshold"
        return(candidates[1L, , drop = FALSE])
      }
      candidates <- x[order(x$median_elapsed_sec), , drop = FALSE]
      candidates$recommendation_basis <- "speed_only_no_recall"
      candidates[1L, , drop = FALSE]
    }
  ))
  row.names(out) <- NULL
  out[order(out$dataset, out$backend, out$metric, out$k), , drop = FALSE]
}

compare_auto_to_recommendations <- function(cycle_summary, recommendations) {
  if (!nrow(recommendations)) return(recommendations)
  auto <- cycle_summary[cycle_summary$method == "auto", , drop = FALSE]
  if (!nrow(auto)) return(data.frame())
  keys <- c("dataset", "backend", "metric", "k")
  auto_keep <- c(
    keys, "method", "result_backend", "resolved_backend", "implementation_backend",
    "success_cycles", "median_elapsed_sec", "median_recall_at_k", "min_recall_at_k",
    "median_min_recall_at_k", "recall_reference", "median_recall_query_n"
  )
  rec_keep <- c(auto_keep, "recommendation_basis")
  auto <- auto[, auto_keep, drop = FALSE]
  recommendations <- recommendations[, rec_keep, drop = FALSE]
  names(auto)[match(auto_keep[-seq_along(keys)], names(auto))] <- paste0("auto_", auto_keep[-seq_along(keys)])
  names(recommendations)[match(rec_keep[-seq_along(keys)], names(recommendations))] <- paste0("recommended_", rec_keep[-seq_along(keys)])
  comparison <- merge(auto, recommendations, by = keys, all = FALSE)
  if (!nrow(comparison)) return(comparison)
  comparison$auto_is_recommended_method <- comparison$auto_method == comparison$recommended_method
  comparison$auto_uses_recommended_result_backend <- comparison$auto_result_backend == comparison$recommended_result_backend
  comparison$auto_uses_recommended_resolved_backend <- comparison$auto_resolved_backend == comparison$recommended_resolved_backend
  comparison$auto_uses_recommended_implementation <- comparison$auto_implementation_backend == comparison$recommended_implementation_backend
  comparison$auto_median_speed_ratio <- safe_positive_ratio(
    comparison$auto_median_elapsed_sec,
    comparison$recommended_median_elapsed_sec
  )
  comparison$auto_median_recall_gap <- safe_difference(
    comparison$recommended_median_recall_at_k,
    comparison$auto_median_recall_at_k
  )
  comparison[order(comparison$dataset, comparison$backend, comparison$metric, comparison$k), , drop = FALSE]
}

compare_auto_to_fastest <- function(ok, fastest) {
  auto_rows <- ok[ok$method == "auto", , drop = FALSE]
  if (!nrow(auto_rows) || is.null(fastest) || !nrow(fastest)) return(data.frame())
  keys <- c("dataset", "backend", "metric", "k", "cycle")
  auto_keep <- c(
    keys, "result_backend", "resolved_backend", "implementation_backend",
    "elapsed_sec", "recall_at_k", "recall_reference", "recall_query_n"
  )
  fastest_keep <- c(
    keys, "method", "result_backend", "resolved_backend", "implementation_backend",
    "elapsed_sec", "recall_at_k", "recall_reference", "recall_query_n"
  )
  auto_rows <- auto_rows[, auto_keep, drop = FALSE]
  fastest <- fastest[, fastest_keep, drop = FALSE]
  names(auto_rows)[match(auto_keep[-seq_along(keys)], names(auto_rows))] <- paste0(
    "auto_",
    auto_keep[-seq_along(keys)]
  )
  names(fastest)[match(fastest_keep[-seq_along(keys)], names(fastest))] <- paste0(
    "fastest_",
    fastest_keep[-seq_along(keys)]
  )
  comparison <- merge(
    auto_rows[, c(keys, paste0("auto_", auto_keep[-seq_along(keys)])), drop = FALSE],
    fastest[, c(keys, paste0("fastest_", fastest_keep[-seq_along(keys)])), drop = FALSE],
    by = keys,
    all = FALSE
  )
  if (!nrow(comparison)) return(comparison)
  comparison$auto_is_fastest_method <- comparison$fastest_method == "auto"
  comparison$auto_uses_fastest_result_backend <- comparison$auto_result_backend == comparison$fastest_result_backend
  comparison$auto_uses_fastest_resolved_backend <- comparison$auto_resolved_backend == comparison$fastest_resolved_backend
  comparison$auto_uses_fastest_implementation <- comparison$auto_implementation_backend == comparison$fastest_implementation_backend
  comparison$auto_speed_ratio <- safe_positive_ratio(comparison$auto_elapsed_sec, comparison$fastest_elapsed_sec)
  comparison$auto_recall_gap <- safe_difference(comparison$fastest_recall_at_k, comparison$auto_recall_at_k)
  comparison[order(comparison$dataset, comparison$backend, comparison$metric, comparison$k, comparison$cycle), , drop = FALSE]
}

safe_positive_ratio <- function(numerator, denominator) {
  numerator <- suppressWarnings(as.numeric(numerator))
  denominator <- suppressWarnings(as.numeric(denominator))
  out <- rep(NA_real_, max(length(numerator), length(denominator)))
  numerator <- rep(numerator, length.out = length(out))
  denominator <- rep(denominator, length.out = length(out))
  ok <- is.finite(numerator) & is.finite(denominator) & denominator > 0
  out[ok] <- numerator[ok] / denominator[ok]
  out
}

safe_difference <- function(left, right) {
  left <- suppressWarnings(as.numeric(left))
  right <- suppressWarnings(as.numeric(right))
  out <- rep(NA_real_, max(length(left), length(right)))
  left <- rep(left, length.out = length(out))
  right <- rep(right, length.out = length(out))
  ok <- is.finite(left) & is.finite(right)
  out[ok] <- left[ok] - right[ok]
  out
}

nn_metric_rank_value <- function(data, column, default, higher_is_better = FALSE) {
  value <- if (column %in% names(data)) data[[column]] else rep(default, nrow(data))
  value <- suppressWarnings(as.numeric(value))
  value[!is.finite(value)] <- default
  if (higher_is_better) -value else value
}

rank_nn_metric_success <- function(ok, include_cycle = FALSE) {
  if (!nrow(ok)) return(ok)
  order_args <- list(
    ok$dataset,
    ok$backend,
    ok$metric,
    ok$k
  )
  if (isTRUE(include_cycle)) {
    order_args <- c(order_args, list(ok$cycle))
  }
  order_args <- c(order_args, list(
    nn_metric_rank_value(ok, "recall_at_k", -Inf, higher_is_better = TRUE),
    nn_metric_rank_value(ok, "min_recall_at_k", -Inf, higher_is_better = TRUE),
    nn_metric_rank_value(ok, "elapsed_sec", Inf)
  ))
  ok[do.call(order, order_args), , drop = FALSE]
}

select_nn_metric_best_rows <- function(ok, recall_threshold, include_cycle = FALSE) {
  if (!nrow(ok)) return(ok)
  keys <- c("dataset", "backend", "metric", "k")
  if (isTRUE(include_cycle)) keys <- c(keys, "cycle")
  parts <- split(ok, do.call(paste, c(ok[keys], sep = "__")))
  best <- lapply(parts, function(x) {
    has_recall <- is.finite(x$recall_at_k)
    if (any(has_recall)) {
      above <- x[has_recall & x$recall_at_k >= recall_threshold, , drop = FALSE]
      if (nrow(above)) {
        ranked <- above[order(
          nn_metric_rank_value(above, "elapsed_sec", Inf),
          nn_metric_rank_value(above, "recall_at_k", -Inf, higher_is_better = TRUE),
          nn_metric_rank_value(above, "min_recall_at_k", -Inf, higher_is_better = TRUE)
        ), , drop = FALSE]
        return(ranked[1L, , drop = FALSE])
      }
      ranked <- x[has_recall, , drop = FALSE]
      ranked <- ranked[order(
        nn_metric_rank_value(ranked, "recall_at_k", -Inf, higher_is_better = TRUE),
        nn_metric_rank_value(ranked, "min_recall_at_k", -Inf, higher_is_better = TRUE),
        nn_metric_rank_value(ranked, "elapsed_sec", Inf)
      ), , drop = FALSE]
      return(ranked[1L, , drop = FALSE])
    }
    ranked <- x[order(nn_metric_rank_value(x, "elapsed_sec", Inf)), , drop = FALSE]
    ranked[1L, , drop = FALSE]
  })
  out <- do.call(rbind, best)
  row.names(out) <- NULL
  out[do.call(order, out[keys]), , drop = FALSE]
}

canonical_method_key <- function(method) {
  trimws(as.character(method))
}

canonical_method_values <- function(methods) {
  methods <- unique(canonical_method_key(methods))
  methods <- methods[nzchar(methods)]
  invalid <- methods[!methods %in% default_nn_method_values()]
  if (length(invalid)) {
    stop(
      "`methods` must use canonical lowercase public method labels. ",
      "Invalid value(s): ",
      paste(invalid, collapse = ", "),
      ".",
      call. = FALSE
    )
  }
  if (!length(methods)) {
    stop("`methods` must contain at least one method.", call. = FALSE)
  }
  methods
}

capability_row <- function(caps, backend, method, metric) {
  backend <- tolower(as.character(backend)[1L])
  metric <- canonical_metric_key(metric)[[1L]]
  method_key <- canonical_method_key(method)
  hit <- caps[caps$backend == backend & caps$method == method_key & caps$metric == metric, , drop = FALSE]
  if (!nrow(hit)) return(NULL)
  hit[1L, , drop = FALSE]
}

capability_status <- function(caps, backend, method, metric) {
  cap <- capability_row(caps, backend, method, metric)
  if (is.null(cap)) {
    return(list(
      listed = FALSE,
      supported = FALSE,
      notes = sprintf(
        "%s/%s/%s is not listed in faissR::nn_capabilities().",
        backend, method, metric
      )
    ))
  }
  list(
    listed = TRUE,
    supported = isTRUE(cap$supported[[1L]]),
    notes = cap$notes[[1L]] %||% "Unsupported by faissR::nn_capabilities()."
  )
}

faiss_gpu_available_runtime <- function() {
  helper <- tryCatch(
    getFromNamespace("faiss_gpu_available", "faissR"),
    error = function(e) NULL
  )
  is.function(helper) && isTRUE(helper()) && isTRUE(faissR::cuda_available())
}

resolve_public_route <- function(backend, method, metric) {
  helper <- tryCatch(
    getFromNamespace("resolve_public_nn_backend", "faissR"),
    error = function(e) NULL
  )
  if (!is.function(helper)) {
    stop("faissR internal backend resolver is unavailable.", call. = FALSE)
  }
  helper(backend, method, metric)
}

route_runtime_skip <- function(backend, method, metric) {
  route <- tryCatch(
    resolve_public_route(backend, method, metric),
    error = function(e) e
  )
  if (inherits(route, "error")) {
    return(list(
      skip = TRUE,
      route = NA_character_,
      notes = paste(
        "faissR backend resolver rejected this method/backend/metric combination:",
        conditionMessage(route)
      )
    ))
  }
  route <- as.character(route)[1L]
  if (route %in% c("faiss_gpu_flat_l2", "faiss_gpu_flat_ip",
                   "faiss_gpu_flat_cosine", "faiss_gpu_flat_correlation",
                   "faiss_gpu_ivf_flat", "faiss_gpu_ivfpq", "faiss_gpu_cagra")) {
    if (!isTRUE(faiss_gpu_available_runtime())) {
      return(list(
        skip = TRUE,
        route = route,
        notes = paste(
          "Resolved route `", route, "` requires FAISS GPU support and a CUDA device, ",
          "but that runtime is unavailable.", sep = ""
        )
      ))
    }
  } else if (startsWith(route, "cuda_cuvs")) {
    if (!isTRUE(faissR::cuvs_available())) {
      return(list(
        skip = TRUE,
        route = route,
        notes = paste(
          "Resolved route `", route, "` requires RAPIDS cuVS, ",
          "but cuVS is unavailable in the current runtime.", sep = ""
        )
      ))
    }
  } else if (route %in% c("cuda", "cuda_grid")) {
    if (!isTRUE(faissR::cuda_available())) {
      return(list(
        skip = TRUE,
        route = route,
        notes = paste(
          "Resolved route `", route, "` requires a CUDA device, ",
          "but CUDA is unavailable in the current runtime.", sep = ""
        )
      ))
    }
  } else if (startsWith(route, "faiss_")) {
    if (!isTRUE(faissR::faiss_available())) {
      return(list(
        skip = TRUE,
        route = route,
        notes = paste(
          "Resolved route `", route, "` requires FAISS, ",
          "but FAISS is unavailable in the current runtime.", sep = ""
        )
      ))
    }
  }
  NULL
}

auto_expected_skip <- function(caps, method, metric) {
  auto <- capability_status(caps, "auto", method, metric)
  if (!isTRUE(auto$supported)) {
    return(list(skip = TRUE, notes = auto$notes))
  }
  runtime <- route_runtime_skip("auto", method, metric)
  if (!is.null(runtime)) return(runtime)
  NULL
}

is_expected_skip <- function(caps, backend, method, metric) {
  backend <- tolower(as.character(backend)[1L])
  if (identical(backend, "auto")) {
    return(auto_expected_skip(caps, method, metric))
  }
  if (!backend %in% c("cpu", "cuda")) return(NULL)
  cap <- capability_status(caps, backend, method, metric)
  if (!isTRUE(cap$supported)) return(list(skip = TRUE, notes = cap$notes))
  runtime <- route_runtime_skip(backend, method, metric)
  if (!is.null(runtime)) return(runtime)
  if (identical(backend, "cuda") &&
      !isTRUE(faissR::cuda_available()) &&
      !isTRUE(faissR::cuvs_available())) {
    return(list(
      skip = TRUE,
      notes = paste(
        "backend = \"cuda\" is supported by design for this method/metric,",
        "but CUDA/cuVS is unavailable in the current runtime."
      )
    ))
  }
  NULL
}

nn_data_expected_skip <- function(x, method) {
  method <- canonical_method_key(method)[[1L]]
  if (identical(method, "grid")) {
    p <- ncol(x)
    if (length(p) != 1L || is.na(p) || !p %in% c(2L, 3L)) {
      return(list(
        skip = TRUE,
        route = NA_character_,
        notes = sprintf(
          paste(
            "`method = \"grid\"` supports only two- or three-column matrices.",
            "This dataset has %s columns, so grid is recorded as an expected",
            "skip instead of a method failure."
          ),
          if (length(p) == 1L && !is.na(p)) as.character(p) else "an unknown number of"
        )
      ))
    }
    return(NULL)
  }
  if (!identical(method, "sparse")) return(NULL)
  if (inherits(x, "sparseMatrix") || inherits(x, "dgCMatrix")) return(NULL)
  list(
    skip = TRUE,
    route = NA_character_,
    notes = paste(
      "`method = \"sparse\"` is a sparse Matrix route. The benchmark datasets",
      "loaded by this script are dense matrices, so sparse is recorded as an",
      "expected skip to avoid converting dense data into a sparse representation."
    )
  )
}

run_one <- function(x, dataset_name, backend, method, metric, k, cycle, n_threads,
                    timeout, reference, seed) {
  started <- proc.time()[["elapsed"]]
  old_options <- options(
    faissR.approx_knn_seed = as.integer(seed),
    faissR.faiss_gpu_ivf_tune_seed = as.integer(seed + 11L),
    faissR.cuvs_cagra_tune_seed = as.integer(seed + 23L)
  )
  on.exit(options(old_options), add = TRUE)
  set.seed(as.integer(seed))
  preflight_route <- tryCatch(
    as.character(resolve_public_route(backend, method, metric))[1L],
    error = function(e) NA_character_
  )
  tryCatch({
    out <- with_elapsed_limit({
      faissR::nn(
        x,
        k = k,
        backend = backend,
        method = method,
        metric = metric,
        n_threads = n_threads
      )
    }, timeout)
    elapsed <- proc.time()[["elapsed"]] - started
    recall <- if (is.null(reference)) {
      data.frame(recall_at_k = NA_real_, median_recall_at_k = NA_real_, min_recall_at_k = NA_real_)
    } else if (is.null(reference$rows)) {
      benchmark_knn_recall(out, reference$knn, k = k)
    } else {
      sampled <- list(
        indices = out$indices[reference$rows, , drop = FALSE],
        distances = out$distances[reference$rows, , drop = FALSE]
      )
      benchmark_knn_recall(sampled, reference$knn, k = k)
    }
    result_row(
      dataset = dataset_name,
      n = nrow(x),
      p = ncol(x),
      backend = backend,
      method = method,
      metric = metric,
      k = k,
      cycle = cycle,
      n_threads = n_threads,
      status = "success",
      elapsed_sec = elapsed,
      peak_rss_gb = read_peak_rss_gb(),
      result_backend = attr(out, "backend") %||% NA_character_,
      resolved_backend = attr(out, "resolved_backend") %||% attr(out, "backend") %||% NA_character_,
      implementation_backend = nn_implementation_backend(out),
      preflight_route = preflight_route,
      exact = isTRUE(attr(out, "exact")),
      recall_at_k = recall$recall_at_k[[1L]],
      median_recall_at_k = recall$median_recall_at_k[[1L]],
      min_recall_at_k = recall$min_recall_at_k[[1L]],
      recall_reference = if (is.null(reference)) NA_character_ else reference$mode,
      recall_query_n = if (is.null(reference)) NA_integer_ else if (is.null(reference$rows)) nrow(x) else length(reference$rows)
    )
  }, error = function(e) {
    result_row(
      dataset = dataset_name,
      n = nrow(x),
      p = ncol(x),
      backend = backend,
      method = method,
      metric = metric,
      k = k,
      cycle = cycle,
      n_threads = n_threads,
      status = "failed",
      error = conditionMessage(e),
      elapsed_sec = proc.time()[["elapsed"]] - started,
      peak_rss_gb = read_peak_rss_gb(),
      preflight_route = preflight_route
    )
  })
}

args <- parse_args()
configure_native_libs()

cmd_args <- commandArgs(FALSE)
file_arg <- cmd_args[grep("^--file=", cmd_args)[1L]] %||% "benchmark_scripts/benchmark_nn_metrics.R"
script_dir <- dirname(normalizePath(sub("^--file=", "", file_arg), mustWork = FALSE))
helper <- file.path(script_dir, "source.R")
if (!file.exists(helper)) helper <- file.path(getwd(), "benchmark_scripts/source.R")
source(helper)

data_root <- args$data_root %||% Sys.getenv("FAISSR_BENCHMARK_DATA", unset = file.path(getwd(), "Data"))
out_dir <- args$out_dir %||% Sys.getenv("FAISSR_BENCHMARK_OUT", unset = file.path(getwd(), paste0("faissR_NN_METRICS_", format(Sys.time(), "%Y%m%d_%H%M%S"))))
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

n_threads <- required_positive_int_arg(args$threads %||% 4L, "threads")
configure_threads(n_threads)
seed <- required_positive_int_arg(args$seed %||% 1L, "seed")
timeout <- required_positive_int_arg(args$timeout %||% 600L, "timeout")
cycles <- required_positive_int_arg(args$cycles %||% 1L, "cycles")
quality_n <- required_positive_int_arg(args$quality_n %||% 512L, "quality_n")
quality_max_ops <- required_positive_numeric_arg(args$quality_max_ops %||% "5e9", "quality_max_ops")
recall_threshold <- required_probability_arg(args$recall_threshold %||% "0.98", "recall_threshold")

available_datasets <- c(dataset_index(data_root)$dataset, "SimulatedUniform2D", "SimulatedUniform3D")
datasets <- validate_dataset_values(
  split_arg(args$datasets, paste(available_datasets, collapse = ",")),
  available_datasets
)
backends <- validate_backend_values(split_arg(args$backends, paste(default_nn_backend_values(), collapse = ",")))
methods <- canonical_method_values(split_arg(args$methods, paste(default_nn_method_values(), collapse = ",")))
metrics <- validate_metric_values(split_arg(args$metrics, paste(default_nn_metric_values(), collapse = ",")))
k_values <- required_positive_int_values(
  split_arg(args$k_values, paste(default_nn_k_values(), collapse = ",")),
  "k_values"
)

suppressPackageStartupMessages(library(faissR))
capabilities <- faissR::nn_capabilities()

config <- data.frame(
  key = c("data_root", "out_dir", "available_datasets", "datasets", "backends",
          "methods", "metrics", "k_values", "threads", "timeout", "cycles",
          "quality_n", "quality_max_ops", "recall_threshold", "seed"),
  value = c(
    data_root, out_dir, paste(available_datasets, collapse = ","),
    paste(datasets, collapse = ","), paste(backends, collapse = ","),
    paste(methods, collapse = ","), paste(metrics, collapse = ","),
    paste(k_values, collapse = ","), n_threads, timeout, cycles, quality_n,
    format(quality_max_ops, scientific = TRUE), recall_threshold, seed
  ),
  stringsAsFactors = FALSE
)
utils::write.csv(config, file.path(out_dir, "nn_metric_benchmark_config.csv"), row.names = FALSE)
utils::write.csv(capabilities, file.path(out_dir, "nn_metric_capabilities.csv"), row.names = FALSE)

results <- list()
row_id <- 0L
for (dataset_name in datasets) {
  loaded <- tryCatch(load_dataset(dataset_name, data_root, seed), error = identity)
  if (inherits(loaded, "error")) {
    row_id <- row_id + 1L
    results[[row_id]] <- result_row(
      dataset = dataset_name,
      n = NA_integer_,
      p = NA_integer_,
      backend = NA_character_,
      method = NA_character_,
      metric = NA_character_,
      k = NA_integer_,
      cycle = NA_integer_,
      n_threads = n_threads,
      status = "failed",
      error = conditionMessage(loaded)
    )
    next
  }
  x <- loaded$data
  references <- new.env(parent = emptyenv())
  for (metric in metrics) {
    for (k in k_values) {
      ref_key <- paste(metric, k, sep = "__")
      references[[ref_key]] <- tryCatch(
        metric_reference(x, k, metric, quality_n, quality_max_ops, n_threads, seed),
        error = function(e) NULL
      )
      for (cycle in seq_len(cycles)) {
        cycle_seed <- seed + (cycle - 1L) * 1000003L
        for (backend in backends) {
          for (method in methods) {
            row_id <- row_id + 1L
            expected <- is_expected_skip(capabilities, backend, method, metric)
            if (is.null(expected)) expected <- nn_data_expected_skip(x, method)
            if (!is.null(expected)) {
              row <- result_row(
                dataset = dataset_name,
                n = nrow(x),
                p = ncol(x),
                backend = backend,
                method = method,
                metric = metric,
                k = k,
                cycle = cycle,
                n_threads = n_threads,
                status = "expected_skip",
                error = expected$notes,
                expected_skip = TRUE,
                capability_notes = expected$notes,
                preflight_route = expected$route %||% NA_character_
              )
            } else {
              row <- run_one(
                x = x,
                dataset_name = dataset_name,
                backend = backend,
                method = method,
                metric = metric,
                k = k,
                cycle = cycle,
                n_threads = n_threads,
                timeout = timeout,
                reference = references[[ref_key]],
                seed = cycle_seed
              )
            }
            results[[row_id]] <- row
            utils::write.csv(
              do.call(rbind, results),
              file.path(out_dir, "nn_metric_benchmark_results.csv"),
              row.names = FALSE
            )
            cat(sprintf(
              "[%s] dataset=%s cycle=%s backend=%s method=%s metric=%s k=%s status=%s elapsed=%.3f\n",
              format(Sys.time(), "%Y-%m-%d %H:%M:%S"),
              dataset_name, cycle, backend, method, metric, k, row$status,
              ifelse(is.na(row$elapsed_sec), 0, row$elapsed_sec)
            ))
            flush.console()
          }
        }
      }
    }
  }
  rm(x, loaded, references)
  gc()
}

results_df <- do.call(rbind, results)
utils::write.csv(results_df, file.path(out_dir, "nn_metric_benchmark_results.csv"), row.names = FALSE)

ok <- results_df[results_df$status == "success", , drop = FALSE]
if (nrow(ok)) {
  best <- select_nn_metric_best_rows(ok, recall_threshold, include_cycle = FALSE)
  utils::write.csv(best, file.path(out_dir, "nn_metric_best_by_dataset_backend_metric_k.csv"), row.names = FALSE)

  best_cycle <- select_nn_metric_best_rows(ok, recall_threshold, include_cycle = TRUE)
  utils::write.csv(best_cycle, file.path(out_dir, "nn_metric_best_by_dataset_backend_metric_k_cycle.csv"), row.names = FALSE)

  cycle_summary <- summarize_nn_cycles(ok)
  utils::write.csv(
    cycle_summary,
    file.path(out_dir, "nn_metric_cycle_summary.csv"),
    row.names = FALSE
  )

  recommendations <- recommend_nn_methods(cycle_summary, recall_threshold)
  if (nrow(recommendations)) {
    utils::write.csv(
      recommendations,
      file.path(out_dir, "nn_metric_recommendations_from_cycles.csv"),
      row.names = FALSE
    )
  }

  aggregate_auto <- compare_auto_to_recommendations(cycle_summary, recommendations)
  if (nrow(aggregate_auto)) {
    utils::write.csv(
      aggregate_auto,
      file.path(out_dir, "nn_metric_auto_vs_cycle_recommendation.csv"),
      row.names = FALSE
    )
  }

  fastest <- NULL
  tunable <- ok[!is.na(ok$recall_at_k) & ok$recall_at_k >= recall_threshold, , drop = FALSE]
  if (nrow(tunable)) {
    tunable <- tunable[order(tunable$dataset, tunable$backend, tunable$metric, tunable$k, tunable$cycle, tunable$elapsed_sec), , drop = FALSE]
    fastest <- do.call(rbind, lapply(
      split(tunable, paste(tunable$dataset, tunable$backend, tunable$metric, tunable$k, tunable$cycle, sep = "__")),
      function(x) x[1L, , drop = FALSE]
    ))
    fastest$quality_score <- NULL
    utils::write.csv(
      fastest,
      file.path(out_dir, "nn_metric_fastest_at_recall_threshold.csv"),
      row.names = FALSE
    )
  }

  comparison <- compare_auto_to_fastest(ok, fastest)
  if (nrow(comparison)) {
    utils::write.csv(
      comparison,
      file.path(out_dir, "nn_metric_auto_vs_fastest.csv"),
      row.names = FALSE
    )
  }
}

materials <- c(
  "# NN Metric Benchmark",
  "",
  "This benchmark exercises public faissR nearest-neighbour methods across device backends, metrics, and k values.",
  "",
  sprintf("- Output directory: `%s`", out_dir),
  sprintf("- Data root: `%s`", data_root),
  sprintf("- Default real datasets: `%s`", paste(dataset_index(data_root)$dataset, collapse = "`, `")),
  "- Default simulated datasets: `SimulatedUniform2D`, `SimulatedUniform3D`",
  sprintf("- Backends: `%s`", paste(backends, collapse = "`, `")),
  sprintf("- Methods: `%s`", paste(methods, collapse = "`, `")),
  sprintf("- Metrics: `%s`", paste(metrics, collapse = "`, `")),
  sprintf("- k values: `%s`", paste(k_values, collapse = "`, `")),
  sprintf("- CPU thread cap: `%s`", n_threads),
  sprintf("- Timeout per combination: `%s` seconds", timeout),
  sprintf("- Cycles: `%s`", cycles),
  sprintf("- Fastest-method recall threshold: `%s`", recall_threshold),
  "",
  "Unsupported method/backend/metric combinations are preflighted with `faissR::nn_capabilities()` and the public backend resolver, then recorded as `status = \"expected_skip\"` with `expected_skip = TRUE`.",
  "`method = \"sparse\"` is included in the default public method list but is recorded as an expected skip for dense benchmark datasets, because it is intended for sparse `Matrix` inputs and should not force dense data through a sparse conversion.",
  "`method = \"grid\"` is included in the default public method list but is recorded as an expected skip for datasets outside two or three columns, because it is a native low-dimensional spatial search route.",
  "`nn_metric_benchmark_config.csv` records the run configuration, including the available real plus simulated dataset names accepted by the dataset selector. `nn_metric_benchmark_results.csv` is the raw row-level result table, including successes, failures, expected skips, timings, memory, recall metadata, and resolved backend fields.",
  "`nn_metric_capabilities.csv` stores the design-level capability table used for that preflight. Runtime expected skips also record when a resolved route requires unavailable FAISS, FAISS GPU, CUDA, or RAPIDS cuVS support.",
  "`preflight_route` records the route selected by the public backend resolver before runtime availability checks. `result_backend`, `resolved_backend`, and `implementation_backend` separate the result-facing backend label from the concrete FAISS/cuVS/native implementation label.",
  "Recall is computed against exact CPU references. Small datasets use a full exact self-KNN reference; larger datasets use a deterministic sample of query rows when `quality_n * nrow(data) * ncol(data)` is within `quality_max_ops`. The `recall_reference` and `recall_query_n` columns record which reference mode was used. The same reference is reused across cycles for the same dataset/metric/k.",
  "`nn_metric_fastest_at_recall_threshold.csv` records the fastest successful method per dataset/backend/metric/k/cycle whose recall is at least `recall_threshold`.",
  "`nn_metric_auto_vs_fastest.csv` compares `method = \"auto\"` against that fastest high-recall row within the same cycle and records speed ratio, recall gap, whether auto itself was the fastest high-recall method, whether the result-facing backend matches, and whether the concrete implementation backend matches. Speed ratios and recall gaps are reported as `NA` when the required timing or recall values are missing or invalid.",
  "`nn_metric_cycle_summary.csv` aggregates successful rows across cycles by dataset/backend/method/metric/k and reports success counts, median/min/max elapsed time, recall stability, and the dominant implementation backend.",
  "`nn_metric_recommendations_from_cycles.csv` selects one method per dataset/backend/metric/k. When recall is available, it selects the fastest method whose median recall is at least `recall_threshold`; tied median times are broken by higher median recall, minimum recall, and median minimum recall. If no method reaches the threshold it selects the best-recall row and marks it as below threshold, breaking tied median recall by minimum recall, median minimum recall, and then speed. When recall is unavailable for the group, it selects the fastest successful row and marks the recommendation as speed-only.",
  "`nn_metric_auto_vs_cycle_recommendation.csv` compares aggregate `method = \"auto\"` rows with those cycle-summary recommendations and reports the recommendation basis, median speed ratio, median recall gap, and backend/implementation agreement. Speed ratios and recall gaps are `NA` when the required timing or recall values are unavailable or invalid.",
  "`nn_metric_best_by_dataset_backend_metric_k_cycle.csv` stores the best row within each cycle using the same recall-threshold rule as the cycle recommendations: fastest above threshold, best recall below threshold, and fastest when recall is unavailable; `nn_metric_best_by_dataset_backend_metric_k.csv` keeps the overall best row across cycles with the same rule for backward-compatible summaries.",
  "The script does not add benchmark-only helpers to the package API."
)
writeLines(materials, file.path(out_dir, "MATERIALS_AND_METHODS_nn_metrics.md"))

cat("Saved NN metric benchmark files in: ", out_dir, "\n", sep = "")
