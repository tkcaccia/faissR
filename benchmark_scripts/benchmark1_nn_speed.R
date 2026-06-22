#!/usr/bin/env Rscript

parse_args <- function(args) {
  out <- list()
  for (arg in args) {
    if (grepl("^--", arg)) {
      kv <- strsplit(sub("^--", "", arg), "=", fixed = TRUE)[[1L]]
      key <- kv[[1L]]
      value <- if (length(kv) > 1L) paste(kv[-1L], collapse = "=") else TRUE
      out[[key]] <- value
    }
  }
  out
}

`%||%` <- function(x, y) if (is.null(x) || length(x) == 0L || is.na(x)) y else x

find_benchmark_script <- function(path = "benchmark_scripts/benchmark1_nn_speed.R") {
  candidates <- unique(c(
    path,
    file.path("..", path),
    file.path("..", "..", path)
  ))
  hit <- candidates[file.exists(candidates)]
  if (length(hit)) return(normalizePath(hit[[1L]], mustWork = TRUE))
  normalizePath(path, mustWork = TRUE)
}

script_arg <- grep("^--file=", commandArgs(FALSE), value = TRUE)
script_file <- if (length(script_arg)) sub("^--file=", "", script_arg[[1L]]) else find_benchmark_script()
script_dir <- dirname(normalizePath(script_file, mustWork = FALSE))
source(file.path(script_dir, "source.R"))

args <- parse_args(commandArgs(trailingOnly = TRUE))

faiss_env_dir <- Sys.getenv("FAISSR_ENV_DIR", "")
cuda_lib_dir <- Sys.getenv("FAISSR_CUDA_LIB_DIR", Sys.getenv("CUDA_HOME", ""))
cuda_lib_dir <- if (nzchar(cuda_lib_dir)) file.path(cuda_lib_dir, "targets/x86_64-linux/lib") else ""
library_candidates <- c(
  if (nzchar(faiss_env_dir)) file.path(faiss_env_dir, "lib") else "",
  if (nzchar(faiss_env_dir)) file.path(faiss_env_dir, "targets/x86_64-linux/lib") else "",
  cuda_lib_dir,
  Sys.getenv("LD_LIBRARY_PATH")
)
faiss_library_path <- paste(library_candidates[nzchar(library_candidates)], collapse = ":")
preload_candidates <- c(
  Sys.getenv("FAISSR_LD_PRELOAD"),
  if (nzchar(faiss_env_dir)) file.path(faiss_env_dir, "lib/libstdc++.so.6") else "",
  Sys.getenv("LD_PRELOAD")
)
faiss_preload <- paste(preload_candidates[nzchar(preload_candidates)], collapse = ":")
env_updates <- list()
if (nzchar(faiss_env_dir)) env_updates$CONDA_PREFIX <- faiss_env_dir
if (nzchar(faiss_library_path)) env_updates$LD_LIBRARY_PATH <- faiss_library_path
if (nzchar(faiss_preload)) env_updates$LD_PRELOAD <- faiss_preload
if (length(env_updates)) do.call(Sys.setenv, env_updates)

benchmark_env <- function() {
  env <- character()
  if (nzchar(faiss_env_dir)) env <- c(env, paste0("FAISSR_ENV_DIR=", faiss_env_dir))
  if (nzchar(faiss_library_path)) env <- c(env, paste0("LD_LIBRARY_PATH=", faiss_library_path))
  if (nzchar(faiss_preload)) env <- c(env, paste0("LD_PRELOAD=", faiss_preload))
  env
}

benchmark1_metric_value <- function(metric = NULL, arg_name = "metric") {
  raw <- metric %||% "l2"
  if (length(raw) != 1L || is.na(raw) || !nzchar(raw)) {
    stop("`", arg_name, "` must contain exactly one metric.", call. = FALSE)
  }
  value <- tolower(trimws(raw))
  if (value %in% c("euclidean", "l2")) return("l2")
  if (value %in% c("ip", "innerproduct", "inner_product")) return("inner_product")
  if (value %in% c("cor", "pearson")) return("correlation")
  if (value %in% c("cosine", "correlation")) return(value)
  stop(
    "`", arg_name, "` must be one of: l2, cosine, correlation, inner_product. ",
    "Invalid value(s): ", value, ".",
    call. = FALSE
  )
}

benchmark1_metric_values <- function(metrics = NULL,
                                     env_metrics = Sys.getenv("FAISSR_BENCHMARK1_METRICS", unset = NA_character_)) {
  raw <- metrics %||% env_metrics
  if (length(raw) != 1L || is.na(raw)) {
    raw <- "l2,cosine,correlation,inner_product"
  }
  if (!nzchar(raw)) {
    stop("`metrics` must contain at least one metric.", call. = FALSE)
  }
  values <- trimws(strsplit(raw, ",", fixed = TRUE)[[1L]])
  values <- unique(tolower(values[nzchar(values)]))
  values <- vapply(values, benchmark1_metric_value, character(1L), arg_name = "metrics")
  if (!length(values)) {
    stop("`metrics` must contain at least one metric.", call. = FALSE)
  }
  unique(unname(values))
}

data_root <- args$data_root %||% Sys.getenv("FAISSR_BENCHMARK_DATA", file.path(getwd(), "Data"))
out_dir <- args$out_dir %||% Sys.getenv(
  "FAISSR_BENCHMARK_OUT",
  file.path(getwd(), paste0("faissR_BENCHMARK1_", format(Sys.time(), "%Y%m%d_%H%M%S")))
)
metric <- benchmark1_metric_value(args$metric %||% "l2")

benchmark1_k_values <- function(k_values = NULL,
                                env_k_values = Sys.getenv("FAISSR_BENCHMARK1_K_VALUES", unset = NA_character_)) {
  raw <- k_values %||% env_k_values
  if (length(raw) != 1L || is.na(raw)) {
    raw <- "5,10,15,50,100"
  }
  values <- trimws(strsplit(raw, ",", fixed = TRUE)[[1L]])
  values <- values[nzchar(values)]
  parsed <- suppressWarnings(as.numeric(values))
  invalid <- values[
    is.na(parsed) | !is.finite(parsed) | parsed < 1L |
      abs(parsed - round(parsed)) > sqrt(.Machine$double.eps)
  ]
  if (length(invalid)) {
    stop(
      "`k_values` must contain only positive integers. Invalid value(s): ",
      paste(invalid, collapse = ", "),
      ".",
      call. = FALSE
    )
  }
  parsed <- unique(as.integer(round(parsed)))
  if (!length(parsed)) {
    stop("`k_values` must contain at least one positive integer.", call. = FALSE)
  }
  parsed
}

benchmark1_positive_int_arg <- function(value, arg, default = NULL) {
  raw <- value
  if (is.null(raw) || length(raw) != 1L || is.na(raw)) raw <- default
  parsed <- suppressWarnings(as.numeric(raw))
  if (length(parsed) != 1L || is.na(parsed) || !is.finite(parsed) || parsed < 1L ||
      abs(parsed - round(parsed)) > sqrt(.Machine$double.eps)) {
    stop("`", arg, "` must be a positive integer.", call. = FALSE)
  }
  as.integer(round(parsed))
}

benchmark1_positive_numeric_arg <- function(value, arg, default = NULL) {
  raw <- value
  if (is.null(raw) || length(raw) != 1L || is.na(raw)) raw <- default
  parsed <- suppressWarnings(as.numeric(raw))
  if (length(parsed) != 1L || is.na(parsed) || !is.finite(parsed) || parsed <= 0) {
    stop("`", arg, "` must be a positive numeric value.", call. = FALSE)
  }
  parsed
}

timeout_sec <- benchmark1_positive_int_arg(args$timeout, "timeout", "600")
worker <- isTRUE(as.logical(args$worker %||% FALSE))
quality_eval_max_n <- benchmark1_positive_int_arg(
  args$quality_n %||% Sys.getenv("FAISSR_BENCHMARK1_QUALITY_N", unset = NA_character_),
  "quality_n",
  "512"
)
quality_eval_max_ops <- benchmark1_positive_numeric_arg(
  args$quality_max_ops %||% Sys.getenv("FAISSR_BENCHMARK1_QUALITY_MAX_OPS", unset = NA_character_),
  "quality_max_ops",
  "5e9"
)
n_threads <- benchmark1_positive_int_arg(args$threads, "threads", "4")
k <- benchmark1_positive_int_arg(args$k, "k", "50")

configure_cpu_threads <- function(n_threads) {
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
  invisible(as.integer(n_threads))
}
configure_cpu_threads(n_threads)

dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

log_msg <- function(...) {
  cat(format(Sys.time(), "%Y-%m-%d %H:%M:%S"), " ", sprintf(...), "\n", sep = "")
  flush.console()
}

write_csv_one <- function(path, row) {
  utils::write.csv(row, path, row.names = FALSE)
}

read_result_csvs <- function(files) {
  rows <- lapply(files, read.csv, stringsAsFactors = FALSE)
  cols <- unique(unlist(lapply(rows, names), use.names = FALSE))
  rows <- lapply(rows, function(x) {
    missing <- setdiff(cols, names(x))
    for (col in missing) x[[col]] <- NA
    x[, cols, drop = FALSE]
  })
  do.call(rbind, rows)
}

benchmark1_rank_value <- function(data, column, default, higher_is_better = FALSE) {
  value <- if (column %in% names(data)) data[[column]] else rep(default, nrow(data))
  value <- suppressWarnings(as.numeric(value))
  value[!is.finite(value)] <- default
  if (higher_is_better) -value else value
}

rank_benchmark1_success <- function(success) {
  if (!nrow(success)) return(success)
  success[order(
    success$dataset,
    success$metric,
    success$k,
    benchmark1_rank_value(success, "recall_at_k", -Inf, higher_is_better = TRUE),
    benchmark1_rank_value(success, "rank_correlation", -Inf, higher_is_better = TRUE),
    benchmark1_rank_value(success, "mean_relative_distance_error", Inf),
    benchmark1_rank_value(success, "time_sec", Inf),
    benchmark1_rank_value(success, "peak_rss_gb", Inf)
  ), , drop = FALSE]
}

benchmark1_finite_mean <- function(x) {
  x <- suppressWarnings(as.numeric(x))
  x <- x[is.finite(x)]
  if (!length(x)) return(NA_real_)
  mean(x)
}

read_peak_rss_gb <- function() {
  status <- "/proc/self/status"
  if (!file.exists(status)) return(NA_real_)
  x <- readLines(status, warn = FALSE)
  v <- x[grepl("^VmHWM:", x)]
  if (!length(v)) return(NA_real_)
  kb <- suppressWarnings(as.numeric(gsub("[^0-9]", "", v[[1L]])))
  if (!is.finite(kb)) NA_real_ else kb / 1024^2
}

available_pkg <- function(pkg) requireNamespace(pkg, quietly = TRUE)

metric_arg_for_label <- function(metric) {
  switch(
    metric,
    l2 = "euclidean",
    cosine = "cosine",
    correlation = "correlation",
    inner_product = "inner_product",
    "euclidean"
  )
}

rcpphnsw_distance_arg <- function(metric) {
  switch(
    metric,
    cosine = "cosine",
    inner_product = "ip",
    "euclidean"
  )
}

method_metric_applicable <- function(method, metric) {
  ip_methods <- c(
    "faissR_cpu_exact",
    "faissR_rcpphnsw",
    "faissR_faiss_flat_l2",
    "faissR_faiss_gpu_flat_l2",
    "faissR_faiss_ivf",
    "faissR_faiss_ivfpq",
    "faissR_faiss_gpu_ivf_flat",
    "faissR_faiss_gpu_ivfpq",
    "faissR_faiss_hnsw",
    "faissR_cpu_nndescent",
    "RcppHNSW_hnsw"
  )
  if (grepl("_ip$", method) && !identical(metric, "inner_product")) {
    return(list(ok = FALSE, reason = "inner-product FAISS Flat methods are benchmarked only with `metric = inner_product`"))
  }
  if (identical(metric, "l2")) return(list(ok = TRUE, reason = ""))
  if (identical(metric, "inner_product")) {
    if (method %in% ip_methods) return(list(ok = TRUE, reason = ""))
    return(list(ok = FALSE, reason = "inner-product search is benchmarked only with exact/IP-capable methods"))
  }
  non_euclidean_methods <- c(
    "faissR_cpu_exact",
    "faissR_rcpphnsw",
    "faissR_faiss_flat_l2",
    "faissR_faiss_gpu_flat_l2",
    "faissR_faiss_ivf",
    "faissR_faiss_ivfpq",
    "faissR_faiss_gpu_ivf_flat",
    "faissR_faiss_gpu_ivfpq",
    "faissR_faiss_hnsw",
    "faissR_cpu_nndescent",
    "faissR_cuda_cuvs_nndescent",
    "RcppHNSW_hnsw",
    "BiocNeighbors_hnsw",
    "BiocNeighbors_annoy",
    "uwot_similarity_graph_fnn",
    "uwot_similarity_graph_annoy",
    "uwot_similarity_graph_hnsw",
    "uwot_similarity_graph_nndescent"
  )
  if (identical(metric, "correlation")) {
    non_euclidean_methods <- c(
      "faissR_cpu_exact",
      "faissR_rcpphnsw",
      "faissR_faiss_flat_l2",
      "faissR_faiss_gpu_flat_l2",
      "faissR_faiss_ivf",
      "faissR_faiss_ivfpq",
      "faissR_faiss_gpu_ivf_flat",
      "faissR_faiss_gpu_ivfpq",
      "faissR_faiss_hnsw",
      "faissR_cpu_nndescent",
      "faissR_cuda_cuvs_nndescent"
    )
  }
  if (method %in% non_euclidean_methods) return(list(ok = TRUE, reason = ""))
  list(ok = FALSE, reason = paste0("method `", method, "` does not expose a validated ", metric, " mode in this benchmark"))
}

method_dataset_applicable <- function(method, dataset) {
  validation_pending <- c(
    "faissR_cuda_cuvs_cagra",
    "cuda_ml_knn",
    "RANN_bd",
    "rnndescent_rpf",
    "rnndescent_rnnd",
    "rnndescent_nnd",
    "rnndescent_bruteforce",
    "uwot_similarity_graph_fnn",
    "uwot_similarity_graph_annoy",
    "uwot_similarity_graph_hnsw",
    "uwot_similarity_graph_nndescent"
  )
  if (method %in% validation_pending) {
    return(list(
      ok = FALSE,
      reason = paste0("method `", method, "` is marked unavailable pending wrapper/accuracy validation")
    ))
  }
  grid_methods <- c("faissR_cpu_grid", "faissR_cuda_grid_auto")
  grid_datasets <- c("SimulatedUniform2D", "SimulatedUniform3D")
  if (method %in% grid_methods && !dataset %in% grid_datasets) {
    return(list(ok = FALSE, reason = "grid methods are benchmarked only on the simulated 2D/3D datasets"))
  }
  list(ok = TRUE, reason = "")
}

method_is_exact <- function(method, metric) {
  if (identical(metric, "inner_product")) {
    return(method %in% c("faissR_cpu_exact", "faissR_faiss_flat_l2", "faissR_faiss_gpu_flat_l2"))
  }
  method %in% c(
    "faissR_cpu_exact",
    "faissR_faiss_flat_l2",
    "faissR_faiss_gpu_flat_l2",
    "faissR_cuda_exact",
    "faissR_cuda_cuvs_bruteforce",
    "Rnanoflann_standard",
    "RANN_kd",
    "RANN_bd",
    "rnndescent_bruteforce",
    "BiocNeighbors_exhaustive"
  )
}

coerce_matrix <- function(x) {
  if (inherits(x, "Matrix")) x <- as.matrix(x)
  if (is.data.frame(x)) x <- as.matrix(x)
  if (!is.matrix(x)) x <- as.matrix(x)
  storage.mode(x) <- "double"
  x
}

load_dataset <- function(dataset, data_path) {
  if (identical(dataset, "SimulatedUniform2D")) {
    set.seed(1)
    data <- matrix(runif(2000000), ncol = 2)
    colnames(data) <- c("x", "y")
    return(list(data = data, labels = NULL, source = "simulated matrix(runif(2000000), ncol = 2)"))
  }
  if (identical(dataset, "SimulatedUniform3D")) {
    set.seed(2)
    data <- matrix(runif(3000000), ncol = 3)
    colnames(data) <- c("x", "y", "z")
    return(list(data = data, labels = NULL, source = "simulated matrix(runif(3000000), ncol = 3)"))
  }
  env <- new.env(parent = emptyenv())
  load(data_path, envir = env)
  if (!exists("dataset", envir = env, inherits = FALSE)) {
    stop("No object named `dataset` in ", data_path)
  }
  ds <- get("dataset", envir = env, inherits = FALSE)
  list(data = coerce_matrix(ds$data), labels = ds$labels, source = data_path)
}

standardize_knn <- function(obj) {
  if (is.null(obj)) return(list(indices = NULL, distances = NULL))
  if (!is.null(obj$indices) && !is.null(obj$distances)) {
    return(list(indices = obj$indices, distances = obj$distances))
  }
  if (!is.null(obj$idx) && !is.null(obj$dist)) {
    return(list(indices = obj$idx, distances = obj$dist))
  }
  if (!is.null(obj$nn.idx) && !is.null(obj$nn.dists)) {
    return(list(indices = obj$nn.idx, distances = obj$nn.dists))
  }
  if (!is.null(obj$index) && !is.null(obj$distance)) {
    return(list(indices = obj$index, distances = obj$distance))
  }
  list(indices = NULL, distances = NULL)
}

choose_quality_rows <- function(n, p) {
  if (n < 2L) return(integer())
  by_ops <- floor(quality_eval_max_ops / max(1, as.double(n) * as.double(p)))
  size <- min(quality_eval_max_n, n, max(16L, as.integer(by_ops)))
  if (!is.finite(size) || size < 1L) return(integer())
  set.seed(20260615 + n + p)
  sort(sample.int(n, size))
}

exact_subset_knn <- function(x, rows, k, metric) {
  n <- nrow(x)
  k <- min(k, n - 1L)
  idx <- matrix(NA_integer_, length(rows), k)
  dst <- matrix(NA_real_, length(rows), k)
  if (identical(metric, "cosine")) {
    norms <- sqrt(rowSums(x * x))
    norms[!is.finite(norms) | norms <= 0] <- 1
    z <- x / norms
  } else if (identical(metric, "correlation")) {
    z <- x - rowMeans(x)
    norms <- sqrt(rowSums(z * z))
    norms[!is.finite(norms) | norms <= 0] <- 1
    z <- z / norms
  } else {
    z <- x
  }
  for (ii in seq_along(rows)) {
    r <- rows[[ii]]
    if (metric %in% c("cosine", "correlation")) {
      score <- drop(z %*% z[r, ])
      dist <- 1 - score
      dist[r] <- Inf
      ord <- order(dist, decreasing = FALSE)[seq_len(k)]
      idx[ii, ] <- ord
      dst[ii, ] <- dist[ord]
    } else if (identical(metric, "inner_product")) {
      score <- drop(z %*% z[r, ])
      score[r] <- -Inf
      ord <- order(score, decreasing = TRUE)[seq_len(k)]
      idx[ii, ] <- ord
      best <- score[ord[[1L]]]
      dst[ii, ] <- best - score[ord]
    } else {
      diff <- sweep(z, 2L, z[r, ], FUN = "-")
      dist2 <- rowSums(diff * diff)
      dist2[r] <- Inf
      ord <- order(dist2, decreasing = FALSE)[seq_len(k)]
      idx[ii, ] <- ord
      dst[ii, ] <- sqrt(pmax(0, dist2[ord]))
    }
  }
  list(indices = idx, distances = dst)
}

knn_rank_correlation <- function(candidate, reference, k) {
  vals <- numeric(nrow(reference$indices))
  vals[] <- NA_real_
  for (i in seq_len(nrow(reference$indices))) {
    a <- candidate$indices[i, seq_len(k)]
    b <- reference$indices[i, seq_len(k)]
    a <- a[!is.na(a) & is.finite(a)]
    b <- b[!is.na(b) & is.finite(b)]
    if (!length(a) || !length(b)) next
    universe <- unique(c(a, b))
    ra <- match(universe, a)
    rb <- match(universe, b)
    ra[is.na(ra)] <- k + 1L
    rb[is.na(rb)] <- k + 1L
    if (length(unique(ra)) > 1L && length(unique(rb)) > 1L) {
      vals[[i]] <- suppressWarnings(stats::cor(ra, rb, method = "spearman"))
    }
  }
  benchmark1_finite_mean(vals)
}

evaluate_knn_quality <- function(x, obj, k, metric, exact) {
  empty <- list(
    recall_at_k = NA_real_,
    median_recall_at_k = NA_real_,
    min_recall_at_k = NA_real_,
    mean_relative_distance_error = NA_real_,
    rank_correlation = NA_real_,
    quality_eval_n = 0L,
    quality_exact_sec = NA_real_,
    quality_status = "not_evaluated",
    quality_error = ""
  )
  sx <- standardize_knn(obj)
  if (is.null(sx$indices) || is.null(sx$distances)) {
    empty$quality_error <- "method did not return a KNN index/distance matrix"
    return(empty)
  }
  if (isTRUE(exact)) {
    empty$recall_at_k <- 1
    empty$median_recall_at_k <- 1
    empty$min_recall_at_k <- 1
    empty$mean_relative_distance_error <- 0
    empty$rank_correlation <- 1
    empty$quality_status <- "exact_assumed"
    empty$quality_error <- "exact backend; recall is assumed rather than recomputed"
    return(empty)
  }
  rows <- choose_quality_rows(nrow(x), ncol(x))
  if (!length(rows)) {
    empty$quality_error <- "quality subset is empty"
    return(empty)
  }
  t0 <- proc.time()[["elapsed"]]
  ref <- tryCatch(exact_subset_knn(x, rows, k, metric), error = function(e) e)
  empty$quality_exact_sec <- proc.time()[["elapsed"]] - t0
  empty$quality_eval_n <- length(rows)
  if (inherits(ref, "error")) {
    empty$quality_status <- "failed"
    empty$quality_error <- conditionMessage(ref)
    return(empty)
  }
  kk <- min(k, ncol(sx$indices), ncol(sx$distances), ncol(ref$indices), ncol(ref$distances))
  if (!is.finite(kk) || kk < 1L) {
    empty$quality_status <- "failed"
    empty$quality_error <- "method returned no usable neighbour columns"
    return(empty)
  }
  cand <- list(
    indices = as.matrix(sx$indices[rows, seq_len(kk), drop = FALSE]),
    distances = as.matrix(sx$distances[rows, seq_len(kk), drop = FALSE])
  )
  ref_eval <- list(
    indices = as.matrix(ref$indices[, seq_len(kk), drop = FALSE]),
    distances = as.matrix(ref$distances[, seq_len(kk), drop = FALSE])
  )
  rec <- benchmark_knn_recall(cand, ref_eval, k = kk)
  abs_ref <- benchmark1_finite_mean(abs(ref_eval$distances))
  abs_err <- benchmark1_finite_mean(abs(cand$distances - ref_eval$distances))
  empty$recall_at_k <- rec$recall_at_k[[1L]]
  empty$median_recall_at_k <- rec$median_recall_at_k[[1L]]
  empty$min_recall_at_k <- rec$min_recall_at_k[[1L]]
  empty$mean_relative_distance_error <- if (is.finite(abs_ref) && abs_ref > 0 && is.finite(abs_err)) {
    abs_err / abs_ref
  } else {
    NA_real_
  }
  empty$rank_correlation <- knn_rank_correlation(cand, ref_eval, kk)
  empty$quality_status <- "success"
  empty
}

drop_self_if_first <- function(indices, distances, target_k) {
  if (is.null(indices) || is.null(distances)) return(list(indices = indices, distances = distances))
  if (ncol(indices) > target_k) {
    self_first <- all(indices[, 1L] == seq_len(nrow(indices)))
    zero_first <- all(abs(distances[, 1L]) < 1e-12)
    if (isTRUE(self_first) || isTRUE(zero_first)) {
      indices <- indices[, -1L, drop = FALSE]
      distances <- distances[, -1L, drop = FALSE]
    }
  }
  if (ncol(indices) > target_k) {
    indices <- indices[, seq_len(target_k), drop = FALSE]
    distances <- distances[, seq_len(target_k), drop = FALSE]
  }
  list(indices = indices, distances = distances)
}

save_cuvs_knn <- function(obj, dataset, out_dir) {
  knn_dir <- file.path(out_dir, "knn_cuvs_nndescent")
  dir.create(knn_dir, recursive = TRUE, showWarnings = FALSE)
  nn_cuvs_nndescent <- obj
  save(
    nn_cuvs_nndescent,
    file = file.path(knn_dir, paste0(dataset, "_cuvs_nndescent_k", k, ".RData")),
    compress = "gzip"
  )
}

annoy_knn <- function(x, k, n_trees = 50L, n_threads = 1L) {
  if (!available_pkg("RcppAnnoy")) stop("RcppAnnoy unavailable")
  p <- ncol(x)
  index <- new(RcppAnnoy::AnnoyEuclidean, p)
  for (i in seq_len(nrow(x))) index$addItem(i - 1L, x[i, ])
  index$build(as.integer(n_trees))
  query_one <- function(i) {
    ans <- index$getNNsByVectorList(x[i, ], k + 1L, search_k = -1L, include_distances = TRUE)
    ii <- as.integer(ans$item + 1L)
    dd <- as.numeric(ans$distance)
    keep <- ii != i
    ii <- ii[keep]
    dd <- dd[keep]
    if (length(ii) < k) {
      ii <- c(ii, rep(NA_integer_, k - length(ii)))
      dd <- c(dd, rep(NA_real_, k - length(dd)))
    }
    list(indices = ii[seq_len(k)], distances = dd[seq_len(k)])
  }
  n_threads <- as.integer(max(1L, n_threads))
  rows <- seq_len(nrow(x))
  if (n_threads > 1L && .Platform$OS.type != "windows") {
    chunks <- split(rows, cut(rows, breaks = min(n_threads, length(rows)), labels = FALSE))
    partial <- parallel::mclapply(
      chunks,
      function(ii) lapply(ii, query_one),
      mc.cores = min(n_threads, length(chunks))
    )
    rows_out <- unlist(partial, recursive = FALSE, use.names = FALSE)
  } else {
    rows_out <- lapply(rows, query_one)
  }
  idx <- do.call(rbind, lapply(rows_out, `[[`, "indices"))
  dst <- do.call(rbind, lapply(rows_out, `[[`, "distances"))
  list(indices = idx, distances = dst)
}

faissr_benchmark_route <- function(method) {
  key <- sub("^faissR_", "", method)
  route <- switch(
    key,
    cpu_exact = list(execution_backend = "cpu", public_backend = "cpu", public_method = "exact"),
    rcpphnsw = list(execution_backend = "hnsw", public_backend = "cpu", public_method = "hnsw"),
    faiss_flat_l2 = list(execution_backend = "faiss_flat_l2", public_backend = "cpu", public_method = "flat"),
    faiss_gpu_flat_l2 = list(execution_backend = "faiss_gpu_flat_l2", public_backend = "cuda", public_method = "flat"),
    faiss_ivf = list(execution_backend = "faiss_ivf", public_backend = "cpu", public_method = "ivf"),
    faiss_ivfpq = list(execution_backend = "faiss_ivfpq", public_backend = "cpu", public_method = "ivfpq"),
    faiss_gpu_ivf_flat = list(execution_backend = "faiss_gpu_ivf_flat", public_backend = "cuda", public_method = "ivf"),
    faiss_gpu_ivfpq = list(execution_backend = "faiss_gpu_ivfpq", public_backend = "cuda", public_method = "ivfpq"),
    faiss_gpu_cagra = list(execution_backend = "faiss_gpu_cagra", public_backend = "cuda", public_method = "cagra"),
    faiss_hnsw = list(execution_backend = "faiss_hnsw", public_backend = "cpu", public_method = "hnsw"),
    faiss_nsg = list(execution_backend = "faiss_nsg", public_backend = "cpu", public_method = "nsg"),
    cpu_nndescent = list(execution_backend = "cpu_nndescent", public_backend = "cpu", public_method = "nndescent"),
    cpu_grid = list(execution_backend = "cpu_grid", public_backend = "cpu", public_method = "grid"),
    cuda_exact = list(execution_backend = "cuda_cuvs_bruteforce", public_backend = "cuda", public_method = "bruteforce"),
    cuda_ivf = list(execution_backend = "cuda_ivf", public_backend = "cuda", public_method = "ivf"),
    cuda_grid_auto = list(execution_backend = "cuda_grid_auto", public_backend = "cuda", public_method = "grid"),
    cuda_cuvs_ivf_flat = list(execution_backend = "cuda_cuvs_ivf_flat", public_backend = "cuda", public_method = "ivf"),
    cuda_cuvs_ivfpq = list(execution_backend = "cuda_cuvs_ivfpq", public_backend = "cuda", public_method = "ivfpq"),
    cuda_cuvs_bruteforce = list(execution_backend = "cuda_cuvs_bruteforce", public_backend = "cuda", public_method = "bruteforce"),
    cuda_cuvs_cagra = list(execution_backend = "cuda_cuvs_cagra", public_backend = "cuda", public_method = "cagra"),
    cuda_cuvs_nndescent = list(execution_backend = "cuda_cuvs_nndescent", public_backend = "cuda", public_method = "nndescent"),
    NULL
  )
  if (is.null(route)) {
    route <- list(execution_backend = key, public_backend = NA_character_, public_method = NA_character_)
  }
  route
}

benchmark1_execution_backend_status <- function(execution_backend) {
  backend <- as.character(execution_backend)[1L]
  if (is.na(backend) || !nzchar(backend)) {
    return(list(runtime_available = NA, runtime_reason = NA_character_, runtime_notes = NA_character_))
  }
  if (backend %in% c("cpu", "cpu_nndescent", "cpu_grid")) {
    return(list(runtime_available = TRUE, runtime_reason = "available", runtime_notes = "Native CPU faissR route is available."))
  }
  if (identical(backend, "hnsw")) {
    ok <- available_pkg("RcppHNSW")
    return(list(
      runtime_available = ok,
      runtime_reason = if (ok) "available" else "missing_rcpphnsw",
      runtime_notes = if (ok) "RcppHNSW fallback route is available." else "RcppHNSW is not installed."
    ))
  }
  if (startsWith(backend, "faiss_gpu")) {
    ok <- isTRUE(faissR::faiss_gpu_available())
    return(list(
      runtime_available = ok,
      runtime_reason = if (ok) "available" else "missing_faiss_gpu",
      runtime_notes = if (ok) "FAISS GPU route is available." else "FAISS GPU support is not available in this build."
    ))
  }
  if (startsWith(backend, "cuda_cuvs")) {
    ok <- isTRUE(faissR::cuvs_available())
    return(list(
      runtime_available = ok,
      runtime_reason = if (ok) "available" else "missing_cuvs",
      runtime_notes = if (ok) "Direct cuVS route is available." else "RAPIDS cuVS support is not available in this build."
    ))
  }
  if (startsWith(backend, "cuda")) {
    ok <- isTRUE(faissR::cuda_available())
    return(list(
      runtime_available = ok,
      runtime_reason = if (ok) "available" else "missing_cuda",
      runtime_notes = if (ok) "Native CUDA route is available." else "Native CUDA support is not available in this build."
    ))
  }
  if (startsWith(backend, "faiss")) {
    ok <- isTRUE(faissR::faiss_available())
    return(list(
      runtime_available = ok,
      runtime_reason = if (ok) "available" else "missing_faiss",
      runtime_notes = if (ok) "FAISS CPU route is available." else "FAISS CPU support is not available in this build."
    ))
  }
  list(runtime_available = TRUE, runtime_reason = "available", runtime_notes = "No faissR runtime dependency detected.")
}

benchmark1_runtime_capabilities <- function(methods, metrics = c("l2", "cosine", "correlation", "inner_product")) {
  metrics <- benchmark1_metric_values(paste(metrics, collapse = ","))
  rows <- vector("list", nrow(methods) * length(metrics))
  r <- 0L
  nn_caps <- tryCatch(faissR::nn_capabilities(runtime = TRUE), error = function(e) NULL)
  for (i in seq_len(nrow(methods))) {
    method <- methods$method[[i]]
    for (metric in metrics) {
      r <- r + 1L
      applicable <- method_metric_applicable(method, metric)
      public_metric <- metric_arg_for_label(metric)
      public_cap <- NULL
      if (!is.null(nn_caps) &&
          !is.na(methods$public_backend[[i]]) &&
          !is.na(methods$public_method[[i]])) {
        public_cap <- nn_caps[
          nn_caps$backend == methods$public_backend[[i]] &
            nn_caps$method == methods$public_method[[i]] &
            nn_caps$metric == public_metric,
          ,
          drop = FALSE
        ]
        if (nrow(public_cap)) public_cap <- public_cap[1L, , drop = FALSE] else public_cap <- NULL
      }
      runtime <- if (startsWith(method, "faissR_")) {
        benchmark1_execution_backend_status(methods$execution_backend[[i]])
      } else {
        list(runtime_available = NA, runtime_reason = NA_character_, runtime_notes = NA_character_)
      }
      rows[[r]] <- data.frame(
        method = method,
        metric = metric,
        implementation = methods$implementation[[i]],
        backend = methods$backend[[i]],
        backend_detail = methods$backend_detail[[i]],
        execution_backend = methods$execution_backend[[i]],
        public_backend = methods$public_backend[[i]],
        public_method = methods$public_method[[i]],
        public_metric = public_metric,
        metric_supported = isTRUE(applicable$ok),
        metric_notes = if (isTRUE(applicable$ok)) "" else applicable$reason,
        public_supported = if (!is.null(public_cap)) isTRUE(public_cap$supported[[1L]]) else NA,
        public_resolved_backend = if (!is.null(public_cap) && "resolved_backend" %in% names(public_cap)) public_cap$resolved_backend[[1L]] else NA_character_,
        public_runtime_reason = if (!is.null(public_cap) && "runtime_reason" %in% names(public_cap)) public_cap$runtime_reason[[1L]] else NA_character_,
        runtime_available = runtime$runtime_available,
        runtime_reason = runtime$runtime_reason,
        runtime_notes = runtime$runtime_notes,
        stringsAsFactors = FALSE
      )
    }
  }
  do.call(rbind, rows)
}

benchmark1_runtime_skip <- function(method, metric) {
  if (!startsWith(method, "faissR_")) return(NULL)
  applicable <- method_metric_applicable(method, metric)
  if (!isTRUE(applicable$ok)) return(NULL)
  route <- faissr_benchmark_route(method)
  status <- benchmark1_execution_backend_status(route$execution_backend)
  if (isTRUE(status$runtime_available) || is.na(status$runtime_available)) return(NULL)
  status$runtime_notes %||% paste("Resolved Benchmark #1 route", route$execution_backend, "is unavailable in the current runtime.")
}

run_method <- function(method, x, k, n_threads, dataset, out_dir, metric) {
  configure_cpu_threads(n_threads)
  if (startsWith(method, "faissR_")) {
    if (!available_pkg("faissR")) stop("faissR unavailable")
    route <- faissr_benchmark_route(method)
    obj <- faissR:::nn_compute(
      data = x,
      points = x,
      k = k,
      backend = route$execution_backend,
      points_missing = TRUE,
      exclude_self = TRUE,
      n_threads = n_threads,
      metric = metric_arg_for_label(metric),
      tuning = "auto"
    )
    if (identical(method, "faissR_cuda_cuvs_nndescent") && identical(metric, "l2")) {
      save_cuvs_knn(obj, dataset, out_dir)
    }
    return(obj)
  }

  switch(
    method,
    Rnanoflann_standard = {
      if (!available_pkg("Rnanoflann")) stop("Rnanoflann unavailable")
      out <- Rnanoflann::nn(x, x, k + 1L, parallel = TRUE, cores = n_threads, sorted = TRUE)
      keep <- drop_self_if_first(out$indices, out$distances, k)
      list(indices = keep$indices, distances = keep$distances)
    },
    RANN_kd = {
      if (!available_pkg("RANN")) stop("RANN unavailable")
      # RANN does not expose a thread argument; benchmark records requested threads,
      # but this call remains single-threaded inside RANN.
      out <- RANN::nn2(x, x, k = k + 1L, treetype = "kd")
      keep <- drop_self_if_first(out$nn.idx, out$nn.dists, k)
      list(indices = keep$indices, distances = keep$distances)
    },
    RANN_bd = {
      if (!available_pkg("RANN")) stop("RANN unavailable")
      # RANN does not expose a thread argument; benchmark records requested threads,
      # but this call remains single-threaded inside RANN.
      out <- RANN::nn2(x, x, k = k + 1L, treetype = "bd")
      keep <- drop_self_if_first(out$nn.idx, out$nn.dists, k)
      list(indices = keep$indices, distances = keep$distances)
    },
    rnndescent_rpf = {
      if (!available_pkg("rnndescent")) stop("rnndescent unavailable")
      rnndescent::rpf_knn(x, k = k, n_threads = n_threads, include_self = FALSE, progress = "none")
    },
    rnndescent_rnnd = {
      if (!available_pkg("rnndescent")) stop("rnndescent unavailable")
      rnndescent::rnnd_knn(x, k = k, n_threads = n_threads, progress = "none")
    },
    rnndescent_nnd = {
      if (!available_pkg("rnndescent")) stop("rnndescent unavailable")
      rnndescent::nnd_knn(x, k = k, n_threads = n_threads, progress = "none")
    },
    rnndescent_bruteforce = {
      if (!available_pkg("rnndescent")) stop("rnndescent unavailable")
      rnndescent::brute_force_knn(x, k = k, n_threads = n_threads)
    },
    RcppHNSW_hnsw = {
      if (!available_pkg("RcppHNSW")) stop("RcppHNSW unavailable")
      RcppHNSW::hnsw_knn(x, k = k, distance = rcpphnsw_distance_arg(metric), M = 16, ef_construction = 200, ef = max(50, 3 * k), n_threads = n_threads, progress = "none")
    },
    RcppAnnoy_euclidean = annoy_knn(x, k, n_threads = n_threads),
    BiocNeighbors_vptree = {
      if (!available_pkg("BiocNeighbors")) stop("BiocNeighbors unavailable")
      BiocNeighbors::findKNN(x, k = k, BNPARAM = BiocNeighbors::VptreeParam(distance = "Euclidean"), num.threads = n_threads)
    },
    BiocNeighbors_hnsw = {
      if (!available_pkg("BiocNeighbors")) stop("BiocNeighbors unavailable")
      BiocNeighbors::findKNN(x, k = k, BNPARAM = BiocNeighbors::HnswParam(distance = if (identical(metric, "cosine")) "Cosine" else "Euclidean", nlinks = 16, ef.construction = 200, ef.search = max(50, 3 * k)), num.threads = n_threads)
    },
    BiocNeighbors_annoy = {
      if (!available_pkg("BiocNeighbors")) stop("BiocNeighbors unavailable")
      BiocNeighbors::findKNN(x, k = k, BNPARAM = BiocNeighbors::AnnoyParam(distance = if (identical(metric, "cosine")) "Cosine" else "Euclidean", ntrees = 50), num.threads = n_threads)
    },
    uwot_similarity_graph_fnn = {
      if (!available_pkg("uwot")) stop("uwot unavailable")
      uwot::similarity_graph(x, n_neighbors = k, metric = metric_arg_for_label(metric), nn_method = "fnn", n_threads = n_threads, verbose = FALSE)
    },
    uwot_similarity_graph_annoy = {
      if (!available_pkg("uwot")) stop("uwot unavailable")
      uwot::similarity_graph(x, n_neighbors = k, metric = metric_arg_for_label(metric), nn_method = "annoy", n_threads = n_threads, verbose = FALSE)
    },
    uwot_similarity_graph_hnsw = {
      if (!available_pkg("uwot")) stop("uwot unavailable")
      uwot::similarity_graph(x, n_neighbors = k, metric = metric_arg_for_label(metric), nn_method = "hnsw", n_threads = n_threads, verbose = FALSE)
    },
    uwot_similarity_graph_nndescent = {
      if (!available_pkg("uwot")) stop("uwot unavailable")
      uwot::similarity_graph(x, n_neighbors = k, metric = metric_arg_for_label(metric), nn_method = "nndescent", n_threads = n_threads, verbose = FALSE)
    },
    cuda_ml_knn = {
      if (!available_pkg("cuda.ml")) stop("cuda.ml unavailable")
      exports <- getNamespaceExports("cuda.ml")
      candidate <- intersect(c("knn", "nearest_neighbors", "cuda_ml_knn"), exports)
      if (!length(candidate)) stop("cuda.ml is installed but no recognised KNN export was found")
      fn <- get(candidate[[1L]], envir = asNamespace("cuda.ml"))
      out <- fn(x, k = k)
      standardize_knn(out)
    },
    umap_umap_knn_from_cuvs = {
      if (!available_pkg("umap")) stop("umap unavailable")
      if (!available_pkg("faissR")) stop("faissR unavailable")
      knn <- faissR:::nn_compute(
        data = x,
        points = x,
        k = k,
        backend = "cuda_cuvs_nndescent",
        points_missing = TRUE,
        exclude_self = TRUE,
        n_threads = n_threads,
        metric = "euclidean",
        tuning = "auto"
      )
      sx <- standardize_knn(knn)
      umap::umap.knn(sx$indices, sx$distances)
    },
    Rtsne_neighbors = {
      stop("Rtsne::Rtsne_neighbors consumes precomputed neighbours and optimizes t-SNE; it is not a standalone KNN search method.")
    },
    stop("Unknown method: ", method)
  )
}

method_table <- function() {
  methods <- data.frame(
    method = c(
      "faissR_cpu_exact",
      "faissR_rcpphnsw",
      "faissR_faiss_flat_l2",
      "faissR_faiss_gpu_flat_l2",
      "faissR_faiss_ivf",
      "faissR_faiss_ivfpq",
      "faissR_faiss_gpu_ivf_flat",
      "faissR_faiss_gpu_ivfpq",
      "faissR_faiss_gpu_cagra",
      "faissR_faiss_hnsw",
      "faissR_faiss_nsg",
      "faissR_cpu_nndescent",
      "faissR_cpu_grid",
      "faissR_cuda_exact",
      "faissR_cuda_grid_auto",
      "faissR_cuda_cuvs_ivf_flat",
      "faissR_cuda_cuvs_ivfpq",
      "faissR_cuda_cuvs_bruteforce",
      "faissR_cuda_cuvs_cagra",
      "faissR_cuda_cuvs_nndescent",
      "Rnanoflann_standard",
      "RANN_kd",
      "RANN_bd",
      "rnndescent_rpf",
      "rnndescent_rnnd",
      "rnndescent_nnd",
      "rnndescent_bruteforce",
      "RcppHNSW_hnsw",
      "RcppAnnoy_euclidean",
      "BiocNeighbors_vptree",
      "BiocNeighbors_hnsw",
      "BiocNeighbors_annoy",
      "uwot_similarity_graph_fnn",
      "uwot_similarity_graph_annoy",
      "uwot_similarity_graph_hnsw",
      "uwot_similarity_graph_nndescent",
      "cuda_ml_knn",
      "umap_umap_knn_from_cuvs",
      "Rtsne_neighbors"
    ),
    implementation = c(
      rep("faissR", 20),
      "Rnanoflann", "RANN", "RANN",
      "rnndescent", "rnndescent", "rnndescent", "rnndescent",
      "RcppHNSW", "RcppAnnoy",
      "BiocNeighbors", "BiocNeighbors", "BiocNeighbors",
      "uwot", "uwot", "uwot", "uwot",
      "cuda.ml",
      "umap", "Rtsne"
    ),
    backend = c(
      "CPU", "CPU", "CPU", "CUDA", "CPU", "CPU", "CUDA", "CUDA", "CUDA", "CPU", "CPU", "CPU",
      "CPU", "CUDA", "CUDA", "CUDA", "CUDA", "CUDA", "CUDA", "CUDA",
      rep("CPU", 16),
      "CUDA",
      "CPU", "CPU"
    ),
    kind = c(
      rep("knn_search", 37),
      "knn_consumer",
      "not_applicable"
    ),
    stringsAsFactors = FALSE
  )
  methods$backend_detail <- methods$backend
  methods$backend_detail[methods$method == "faissR_faiss_gpu_ivf_flat"] <- "FAISS GPU + cuVS integrated IVF-Flat"
  methods$backend_detail[methods$method == "faissR_faiss_gpu_ivfpq"] <- "FAISS GPU + cuVS integrated IVF-PQ"
  methods$backend_detail[methods$method == "faissR_faiss_gpu_cagra"] <- "FAISS GPU + cuVS integrated CAGRA"
  methods$backend_detail[methods$method %in% c(
    "faissR_cuda_cuvs_ivf_flat",
    "faissR_cuda_cuvs_ivfpq",
    "faissR_cuda_cuvs_bruteforce",
    "faissR_cuda_cuvs_cagra",
    "faissR_cuda_cuvs_nndescent"
  )] <- "Direct RAPIDS cuVS"
  methods$backend_detail[methods$method %in% c(
    "faissR_faiss_gpu_flat_l2"
  )] <- "FAISS GPU Flat"
  route_rows <- lapply(methods$method, function(method) {
    if (startsWith(method, "faissR_")) {
      faissr_benchmark_route(method)
    } else {
      list(
        execution_backend = NA_character_,
        public_backend = NA_character_,
        public_method = NA_character_
      )
    }
  })
  methods$execution_backend <- vapply(route_rows, `[[`, character(1L), "execution_backend")
  methods$public_backend <- vapply(route_rows, `[[`, character(1L), "public_backend")
  methods$public_method <- vapply(route_rows, `[[`, character(1L), "public_method")
  methods
}

benchmark_method_aliases <- function(methods) {
  aliases <- c(
    auto = "faissR_faiss_hnsw",
    exact = "faissR_cpu_exact",
    flat = "faissR_faiss_flat_l2",
    bruteforce = "faissR_cpu_exact",
    grid = "faissR_cpu_grid",
    hnsw = "faissR_faiss_hnsw",
    ivf = "faissR_faiss_ivf",
    ivfpq = "faissR_faiss_ivfpq",
    nsg = "faissR_faiss_nsg",
    nndescent = "faissR_cpu_nndescent",
    cagra = "faissR_faiss_gpu_cagra",
    cuda_ivf = "faissR_cuda_cuvs_ivf_flat",
    faissR_cuda_ivf = "faissR_cuda_cuvs_ivf_flat"
  )
  out <- trimws(methods)
  mapped <- aliases[out]
  out[!is.na(mapped)] <- unname(mapped[!is.na(mapped)])
  unique(out[nzchar(out)])
}

benchmark1_method_values <- function(methods, valid_methods = method_table()$method) {
  values <- benchmark_method_aliases(strsplit(methods %||% "", ",", fixed = TRUE)[[1L]])
  if (!length(values)) {
    stop("`methods` must contain at least one method.", call. = FALSE)
  }
  invalid <- values[!values %in% valid_methods]
  if (length(invalid)) {
    stop(
      "`methods` contains invalid Benchmark #1 method value(s): ",
      paste(invalid, collapse = ", "),
      ".",
      call. = FALSE
    )
  }
  values
}

invalid_worker_method_row <- function(dataset, method, k, metric, n_threads) {
  data.frame(
    dataset = dataset %||% NA_character_,
    method = method %||% NA_character_,
    implementation = NA_character_,
    backend = NA_character_,
    backend_detail = NA_character_,
    execution_backend = NA_character_,
    public_backend = NA_character_,
    public_method = NA_character_,
    kind = NA_character_,
    n = NA_integer_,
    p = NA_integer_,
    k = k,
    metric = metric,
    n_threads = n_threads,
    status = "failed",
    time_sec = NA_real_,
    load_sec = NA_real_,
    peak_rss_gb = NA_real_,
    recall_at_k = NA_real_,
    median_recall_at_k = NA_real_,
    min_recall_at_k = NA_real_,
    mean_relative_distance_error = NA_real_,
    rank_correlation = NA_real_,
    quality_eval_n = NA_integer_,
    quality_exact_sec = NA_real_,
    quality_status = "failed",
    quality_error = "invalid Benchmark #1 method",
    output_rows = NA_integer_,
    output_cols = NA_integer_,
    error = paste0("invalid Benchmark #1 method: ", method %||% NA_character_),
    stringsAsFactors = FALSE
  )
}

if (worker) {
  dataset <- args$dataset
  data_path <- args$data_path
  method <- args$method
  result_path <- args$result_path
  dir.create(dirname(result_path), recursive = TRUE, showWarnings = FALSE)
  meta <- method_table()
  method_match <- match(method, meta$method)
  if (is.na(method_match)) {
    write_csv_one(result_path, invalid_worker_method_row(dataset, method, k, metric, n_threads))
    quit(status = 0L)
  }
  mm <- meta[method_match, , drop = FALSE]
  started_total <- proc.time()[["elapsed"]]
  row <- data.frame(
    dataset = dataset,
    method = method,
    implementation = mm$implementation %||% NA_character_,
    backend = mm$backend %||% NA_character_,
    backend_detail = mm$backend_detail %||% NA_character_,
    execution_backend = mm$execution_backend %||% NA_character_,
    public_backend = mm$public_backend %||% NA_character_,
    public_method = mm$public_method %||% NA_character_,
    kind = mm$kind %||% NA_character_,
    n = NA_integer_,
    p = NA_integer_,
    k = k,
    metric = metric,
    n_threads = n_threads,
    status = "failed",
    time_sec = NA_real_,
    load_sec = NA_real_,
    peak_rss_gb = NA_real_,
    recall_at_k = NA_real_,
    median_recall_at_k = NA_real_,
    min_recall_at_k = NA_real_,
    mean_relative_distance_error = NA_real_,
    rank_correlation = NA_real_,
    quality_eval_n = NA_integer_,
    quality_exact_sec = NA_real_,
    quality_status = NA_character_,
    quality_error = "",
    output_rows = NA_integer_,
    output_cols = NA_integer_,
    error = "",
    stringsAsFactors = FALSE
  )
  tryCatch({
    if (identical(mm$kind, "not_applicable")) {
      row$status <- "not_applicable"
      row$error <- "Rtsne::Rtsne_neighbors is not a standalone KNN search method."
      write_csv_one(result_path, row)
      quit(status = 0L)
    }
    applicable <- method_metric_applicable(method, metric)
    if (!isTRUE(applicable$ok)) {
      row$status <- "skipped"
      row$error <- applicable$reason
      row$quality_status <- "skipped"
      row$quality_error <- applicable$reason
      write_csv_one(result_path, row)
      quit(status = 0L)
    }
    runtime_skip <- benchmark1_runtime_skip(method, metric)
    if (!is.null(runtime_skip)) {
      row$status <- "skipped"
      row$error <- runtime_skip
      row$quality_status <- "skipped"
      row$quality_error <- runtime_skip
      write_csv_one(result_path, row)
      quit(status = 0L)
    }
    applicable <- method_dataset_applicable(method, dataset)
    if (!isTRUE(applicable$ok)) {
      row$status <- "skipped"
      row$error <- applicable$reason
      row$quality_status <- "skipped"
      row$quality_error <- applicable$reason
      write_csv_one(result_path, row)
      quit(status = 0L)
    }
    load_start <- proc.time()[["elapsed"]]
    ds <- load_dataset(dataset, data_path)
    x <- ds$data
    row$n <- nrow(x)
    row$p <- ncol(x)
    row$load_sec <- proc.time()[["elapsed"]] - load_start
    gc()
    start <- proc.time()[["elapsed"]]
    obj <- run_method(method, x, k, n_threads, dataset, out_dir, metric)
    row$time_sec <- proc.time()[["elapsed"]] - start
    sx <- standardize_knn(obj)
    if (!is.null(sx$indices)) {
      row$output_rows <- nrow(sx$indices)
      row$output_cols <- ncol(sx$indices)
    }
    quality <- evaluate_knn_quality(
      x,
      obj,
      k,
      metric,
      exact = method_is_exact(method, metric)
    )
    row$recall_at_k <- quality$recall_at_k
    row$median_recall_at_k <- quality$median_recall_at_k
    row$min_recall_at_k <- quality$min_recall_at_k
    row$mean_relative_distance_error <- quality$mean_relative_distance_error
    row$rank_correlation <- quality$rank_correlation
    row$quality_eval_n <- quality$quality_eval_n
    row$quality_exact_sec <- quality$quality_exact_sec
    row$quality_status <- quality$quality_status
    row$quality_error <- quality$quality_error
    row$status <- "success"
    row$peak_rss_gb <- read_peak_rss_gb()
  }, error = function(e) {
    row$status <- "failed"
    msg <- conditionMessage(e)
    if (!nzchar(msg)) {
      msg <- paste("native backend failed without an R condition message:", method)
    }
    row$error <- msg
    row$time_sec <- proc.time()[["elapsed"]] - started_total
    row$peak_rss_gb <- read_peak_rss_gb()
  })
  write_csv_one(result_path, row)
  quit(status = 0L)
}

manifest_path <- file.path(data_root, "dataset_manifest.csv")
required_datasets <- c(
  "COIL20",
  "USPS",
  "FashionMNIST",
  "FlowRepository_FR-FCM-ZYRM_files",
  "flow18",
  "MNIST",
  "imagenet",
  "MetRef",
  "mass41"
)
default_dataset_paths <- data.frame(
  dataset = required_datasets,
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
manifest <- if (file.exists(manifest_path)) {
  read.csv(manifest_path, stringsAsFactors = FALSE)
} else {
  default_dataset_paths
}
if (!"path" %in% names(manifest)) {
  if ("relative_path" %in% names(manifest)) {
    manifest$path <- file.path(data_root, manifest$relative_path)
  } else {
    manifest$path <- NA_character_
  }
}
manifest <- merge(default_dataset_paths, manifest, by = "dataset", all.x = TRUE, suffixes = c("_default", "_manifest"), sort = FALSE)
manifest$path <- ifelse(!is.na(manifest$path_manifest) & file.exists(manifest$path_manifest), manifest$path_manifest, manifest$path_default)
datasets <- manifest[, c("dataset", "path")]
datasets$n <- NA_integer_
datasets$p <- NA_integer_
datasets <- rbind(
  datasets,
  data.frame(dataset = "SimulatedUniform2D", path = "SIMULATED_2D", n = 1000000L, p = 2L),
  data.frame(dataset = "SimulatedUniform3D", path = "SIMULATED_3D", n = 1000000L, p = 3L)
)
missing_datasets <- datasets[!startsWith(datasets$path, "SIMULATED") & !file.exists(datasets$path), , drop = FALSE]
if (nrow(missing_datasets)) {
  stop("Missing required dataset files: ", paste(paste0(missing_datasets$dataset, "=", missing_datasets$path), collapse = "; "))
}
if (!is.null(args$datasets)) {
  wanted <- strsplit(args$datasets, ",", fixed = TRUE)[[1L]]
  datasets <- datasets[datasets$dataset %in% wanted, , drop = FALSE]
}

methods <- method_table()
if (!is.null(args$methods)) {
  wanted_methods <- benchmark1_method_values(args$methods, methods$method)
  methods <- methods[methods$method %in% wanted_methods, , drop = FALSE]
}

dir.create(file.path(out_dir, "worker_results"), recursive = TRUE, showWarnings = FALSE)

utils::write.csv(datasets, file.path(out_dir, "benchmark1_datasets.csv"), row.names = FALSE)
utils::write.csv(methods, file.path(out_dir, "benchmark1_methods.csv"), row.names = FALSE)
k_values <- benchmark1_k_values(args$k_values)
metric_values <- benchmark1_metric_values(args$metrics)
utils::write.csv(
  expand.grid(k = k_values, metric = metric_values, stringsAsFactors = FALSE),
  file.path(out_dir, "benchmark1_parameter_grid.csv"),
  row.names = FALSE
)
utils::write.csv(
  benchmark1_runtime_capabilities(methods, metric_values),
  file.path(out_dir, "benchmark1_runtime_capabilities.csv"),
  row.names = FALSE
)

cmdline <- commandArgs(FALSE)
file_arg <- grep("^--file=", cmdline, value = TRUE)
script <- if (length(file_arg)) sub("^--file=", "", file_arg[[1L]]) else find_benchmark_script()
if (!file.exists(script)) {
  script <- find_benchmark_script()
} else {
  script <- normalizePath(script, mustWork = TRUE)
}

results <- list()
job_id <- 0L
for (di in seq_len(nrow(datasets))) {
  for (mi in seq_len(nrow(methods))) {
    for (kk in k_values) {
      for (metric_i in metric_values) {
        job_id <- job_id + 1L
        dataset <- datasets$dataset[[di]]
        method <- methods$method[[mi]]
        result_path <- file.path(out_dir, "worker_results", sprintf("%03d_%s__%s__k%d__%s.csv", job_id, dataset, method, kk, metric_i))
        if (file.exists(result_path)) {
          log_msg("Skipping existing %s / %s / k=%d / metric=%s", dataset, method, kk, metric_i)
          next
        }
        log_msg("[%03d/%03d] %s / %s / k=%d / metric=%s", job_id, nrow(datasets) * nrow(methods) * length(k_values) * length(metric_values), dataset, method, kk, metric_i)
        cmd_args <- c(
          as.character(timeout_sec),
          "Rscript",
          script,
          "--worker=TRUE",
          paste0("--dataset=", dataset),
          paste0("--data_path=", datasets$path[[di]]),
          paste0("--method=", method),
          paste0("--result_path=", result_path),
          paste0("--out_dir=", out_dir),
          paste0("--k=", kk),
          paste0("--metric=", metric_i),
          paste0("--threads=", n_threads)
        )
        status <- system2("timeout", cmd_args, stdout = file.path(out_dir, "benchmark1_worker_stdout.log"), stderr = file.path(out_dir, "benchmark1_worker_stderr.log"), env = benchmark_env())
        if (!file.exists(result_path)) {
          timeout_row <- data.frame(
            dataset = dataset,
            method = method,
            implementation = methods$implementation[[mi]],
            backend = methods$backend[[mi]],
            backend_detail = methods$backend_detail[[mi]],
            execution_backend = methods$execution_backend[[mi]],
            public_backend = methods$public_backend[[mi]],
            public_method = methods$public_method[[mi]],
            kind = methods$kind[[mi]],
            n = datasets$n[[di]],
            p = datasets$p[[di]],
            k = kk,
            metric = metric_i,
            n_threads = n_threads,
            status = if (identical(status, 124L)) "timeout" else "failed",
            time_sec = if (identical(status, 124L)) timeout_sec else NA_real_,
            load_sec = NA_real_,
            peak_rss_gb = NA_real_,
            recall_at_k = NA_real_,
            median_recall_at_k = NA_real_,
            min_recall_at_k = NA_real_,
            mean_relative_distance_error = NA_real_,
            rank_correlation = NA_real_,
            quality_eval_n = NA_integer_,
            quality_exact_sec = NA_real_,
            quality_status = "timeout",
            quality_error = paste("worker did not produce result; exit status", status),
            output_rows = NA_integer_,
            output_cols = NA_integer_,
            error = paste("worker did not produce result; exit status", status),
            stringsAsFactors = FALSE
          )
          write_csv_one(result_path, timeout_row)
        }
      }
    }
  }
}

files <- list.files(file.path(out_dir, "worker_results"), pattern = "[.]csv$", full.names = TRUE)
results <- read_result_csvs(files)
results <- results[order(results$dataset, results$backend, results$implementation, results$method), ]
utils::write.csv(results, file.path(out_dir, "benchmark1_nn_speed_results.csv"), row.names = FALSE)

success <- results[results$status == "success" & results$kind == "knn_search", , drop = FALSE]
ranked_quality <- rank_benchmark1_success(success)
best <- ranked_quality[!duplicated(paste(ranked_quality$dataset, ranked_quality$metric, ranked_quality$k, sep = "\r")), ]
utils::write.csv(best, file.path(out_dir, "benchmark1_best_by_dataset.csv"), row.names = FALSE)
if (nrow(success)) {
  utils::write.csv(ranked_quality, file.path(out_dir, "benchmark1_ranked_speed_quality_memory.csv"), row.names = FALSE)
}

png(file.path(out_dir, "benchmark1_nn_speed_barplot.png"), width = 2200, height = 1400, res = 160)
op <- par(mar = c(12, 5, 4, 2), mfrow = c(ceiling(length(unique(results$dataset)) / 2), 2))
for (dataset in unique(results$dataset)) {
  sub <- results[results$dataset == dataset & results$kind != "not_applicable", , drop = FALSE]
  sub$plot_time <- ifelse(sub$status == "success", sub$time_sec, timeout_sec)
  sub <- sub[order(sub$plot_time), ]
  cols <- ifelse(sub$status == "success", ifelse(sub$backend == "CUDA", "#2b8cbe", "#7bccc4"), "#d95f0e")
  barplot(
    sub$plot_time,
    names.arg = sub$method,
    las = 2,
    col = cols,
    main = dataset,
    ylab = "seconds (timeouts shown at cap)",
    cex.names = 0.45
  )
  legend("topright", fill = c("#7bccc4", "#2b8cbe", "#d95f0e"), legend = c("CPU success", "CUDA success", "failed/timeout"), cex = 0.7, bty = "n")
}
par(op)
dev.off()

materials <- c(
  "# BENCHMARK #1 Materials and Methods",
  "",
  "Benchmark #1 measures nearest-neighbour construction speed across faissR native/FAISS backends, optional CUDA/cuVS backends, and external R package implementations.",
  "",
  paste0("Datasets were read from `", data_root, "`. The required datasets were COIL20, USPS, FashionMNIST, FlowRepository_FR-FCM-ZYRM_files, flow18, MNIST, imagenet, MetRef, and mass41. Each dataset file was loaded from its dataset folder as an `.RData` object named `dataset` containing `dataset$data` and `dataset$labels`. Two simulated reference datasets were generated as `matrix(runif(2000000), ncol = 2)` and `matrix(runif(3000000), ncol = 3)`, giving 1,000,000 observations with 2 and 3 variables, respectively."),
  paste0("All methods were tested over k = ", paste(k_values, collapse = ", "), " and metrics = ", paste(metric_values, collapse = ", "), ". CPU methods were run with n_threads/cores = ", n_threads, " when the package exposed a thread argument. Each dataset-method-parameter combination was executed in a separate R process with GNU `timeout` set to ", timeout_sec, " seconds."),
    "Workers were launched with the configured FAISS/cuVS/CUDA library paths before system library paths when those variables were supplied.",
  paste0("Nearest-neighbour quality was evaluated against an exact subset reference where feasible. The reference subset used at most ", quality_eval_max_n, " rows and was automatically reduced when the estimated operation count exceeded ", format(quality_eval_max_ops, scientific = TRUE), ". Reported quality metrics are recall@k, median recall@k, minimum recall@k, mean relative distance error, and Spearman rank correlation of neighbour ranks. Invalid or non-finite distance/rank quality summaries are recorded as `NA`."),
  "`benchmark1_best_by_dataset.csv` and `benchmark1_ranked_speed_quality_memory.csv` rank successful KNN-search rows by recall@k, neighbour-rank correlation, mean relative distance error, elapsed time, and peak memory. This keeps fast but low-recall rows from being reported as the best method.",
  "The faissR CUDA/cuVS NN-descent output was saved for every dataset where the method completed successfully.",
  "",
  "faissR methods tested: exact CPU, RcppHNSW wrapper, FAISS Flat, FAISS CPU IVF/IVF-Flat, FAISS CPU IVFPQ, FAISS GPU Flat, FAISS GPU IVF-Flat with NVIDIA cuVS integration, FAISS GPU IVF-PQ with NVIDIA cuVS integration, FAISS HNSW, FAISS NSG, native CPU NNDescent, CPU grid on simulated 2D/3D only, native CUDA exact, native CUDA IVF, CUDA grid on simulated 2D/3D only, direct RAPIDS cuVS IVF-Flat, direct RAPIDS cuVS IVF-PQ, direct cuVS brute force, direct cuVS CAGRA, and direct cuVS NN-descent.",
  "Native CPU NNDescent is benchmarked for Euclidean, cosine, correlation, and raw inner-product metrics. Direct CUDA/cuVS NN-descent is benchmarked for Euclidean, cosine, and correlation; raw inner-product CUDA/cuVS NN-descent is recorded as unsupported.",
  "The Flat rows use the public `method = \"flat\"` route. When `metric = \"inner_product\"` is explicitly requested, faissR dispatches the same public Flat rows to the appropriate FAISS inner-product index internally instead of listing duplicate Flat-IP methods.",
  "For faissR rows, `execution_backend` records the internal backend label used by `nn_compute()`, while `public_backend` and `public_method` record the equivalent public `nn(..., backend = , method = )` route. This separates legacy benchmark labels from the public API.",
  "The benchmark result table includes `backend_detail` to distinguish FAISS GPU indexes that use NVIDIA cuVS internally from direct RAPIDS cuVS API calls.",
  "`benchmark1_runtime_capabilities.csv` records the faissR Benchmark #1 method/metric preflight table, including legacy Benchmark #1 method labels, equivalent public `nn()` routes where available, execution backends, metric support, `public_runtime_reason`, `runtime_available`, `runtime_reason`, and current runtime availability notes.",
  "External R package methods tested: Rnanoflann, RANN kd-tree and bd-tree, rnndescent RPF/RNND/NND/brute-force, RcppHNSW, RcppAnnoy, BiocNeighbors VP-tree/HNSW/Annoy, uwot::similarity_graph with nn_method = fnn, annoy, hnsw, and nndescent, and cuda.ml KNN if an installed cuda.ml package exposes a recognised KNN routine. External RcppHNSW rows use Euclidean, cosine, or inner-product (`distance = \"ip\"`) modes when those metrics are requested; correlation is recorded as unavailable for the external RcppHNSW row because Benchmark #1 does not row-center data before calling that package.",
  "umap::umap.knn was included as a precomputed-neighbour consumer test, not as a standalone KNN search algorithm. Rtsne::Rtsne_neighbors was marked not applicable because it consumes precomputed neighbours and optimizes t-SNE rather than exporting a standalone KNN search.",
  "",
  "The benchmark records elapsed method time, load/conversion time, peak resident memory when available from `/proc/self/status`, output dimensions where an index matrix is returned, quality metrics, status, and error messages."
)
writeLines(materials, file.path(out_dir, "BENCHMARK1_MATERIALS_AND_METHODS.md"))

summary_lines <- c(
  "# BENCHMARK #1 Results Summary",
  "",
  paste0("Run directory: `", out_dir, "`"),
  "",
  "## Best Successful KNN Search Per Dataset, Metric, And k",
  "",
  paste(capture.output(print(best[, c("dataset", "metric", "k", "method", "implementation", "backend", "time_sec", "status")], row.names = FALSE)), collapse = "\n"),
  "",
  "## Comments",
  "",
  "This benchmark separates pure KNN search methods from graph/consumer functions. The fastest method can differ by dataset shape: low-dimensional simulated data favours tree/grid-like methods, while high-dimensional image matrices favour approximate graph or GPU methods. Exact brute-force methods are included as references but are expected to time out or be uncompetitive on the largest datasets. cuVS NN-descent outputs are saved to allow later embedding benchmarks to reuse the same neighbour graph rather than recomputing KNN."
)
writeLines(summary_lines, file.path(out_dir, "BENCHMARK1_RESULTS_SUMMARY.md"))

log_msg("DONE: %s", out_dir)
