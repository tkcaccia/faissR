#!/usr/bin/env Rscript

`%||%` <- function(x, y) {
  if (is.null(x) || length(x) == 0L || is.na(x[[1L]])) y else x
}

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

split_arg <- function(value, default) {
  value <- value %||% default
  out <- trimws(strsplit(value, ",", fixed = TRUE)[[1L]])
  out[nzchar(out)]
}

logical_arg <- function(value, default = FALSE) {
  if (is.null(value) || length(value) == 0L || is.na(value[[1L]])) return(isTRUE(default))
  key <- tolower(trimws(as.character(value[[1L]])))
  if (key %in% c("true", "t", "1", "yes", "y", "on")) return(TRUE)
  if (key %in% c("false", "f", "0", "no", "n", "off")) return(FALSE)
  stop("Logical argument must be true or false.", call. = FALSE)
}

positive_int <- function(value, default, name) {
  value <- suppressWarnings(as.numeric(value %||% default))
  if (length(value) != 1L || is.na(value) || !is.finite(value) ||
      value < 1L || abs(value - round(value)) > sqrt(.Machine$double.eps)) {
    stop("`", name, "` must be a positive integer.", call. = FALSE)
  }
  as.integer(round(value))
}

positive_num <- function(value, default, name) {
  value <- suppressWarnings(as.numeric(value %||% default))
  if (length(value) != 1L || is.na(value) || !is.finite(value) || value <= 0) {
    stop("`", name, "` must be a positive number.", call. = FALSE)
  }
  value
}

script_path <- function() {
  args <- commandArgs(FALSE)
  file_arg <- grep("^--file=", args, value = TRUE)
  if (length(file_arg)) return(normalizePath(sub("^--file=", "", file_arg[[1L]]), mustWork = TRUE))
  script_args <- args[file.exists(args) & grepl("[.]R$", args)]
  if (length(script_args)) {
    return(normalizePath(script_args[[length(script_args)]], mustWork = TRUE))
  }
  normalizePath(file.path("benchmark_scripts", "benchmark_nndescent_tuning_grid.R"), mustWork = FALSE)
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

append_csv <- function(row, path) {
  dir.create(dirname(path), recursive = TRUE, showWarnings = FALSE)
  write.table(
    row,
    file = path,
    sep = ",",
    row.names = FALSE,
    col.names = !file.exists(path),
    append = file.exists(path),
    quote = TRUE,
    na = ""
  )
}

read_peak_rss_gb <- function() {
  status <- "/proc/self/status"
  if (!file.exists(status)) return(NA_real_)
  line <- grep("^VmHWM:", readLines(status, warn = FALSE), value = TRUE)
  if (!length(line)) return(NA_real_)
  kb <- suppressWarnings(as.numeric(gsub("[^0-9.]", "", line[[1L]])))
  kb / 1024^2
}

classify_error <- function(message, timed_out = FALSE) {
  if (isTRUE(timed_out)) return("timeout")
  msg <- tolower(as.character(message %||% ""))
  if (grepl("not available|unavailable|not built|requires|no .*support|without cuda|without faiss|without cuvs", msg)) {
    return("unavailable")
  }
  if (grepl("not support|does not support|only supports|only available", msg)) {
    return("unsupported")
  }
  "failed"
}

load_faissR <- function() {
  source_dir <- Sys.getenv("FAISSR_SOURCE_DIR", unset = "")
  if (nzchar(source_dir) && dir.exists(source_dir)) {
    if (!requireNamespace("pkgload", quietly = TRUE)) {
      stop("FAISSR_SOURCE_DIR was set, but the pkgload package is not installed.", call. = FALSE)
    }
    pkgload::load_all(source_dir, quiet = TRUE)
  } else {
    library(faissR)
  }
  if (!"exclude_self" %in% names(formals(faissR::nn))) {
    stop("The loaded faissR package is too old for merged `nn(exclude_self=)` API.", call. = FALSE)
  }
  invisible(TRUE)
}

load_float_dataset <- function(path) {
  env <- new.env(parent = emptyenv())
  load(path, envir = env)
  obj <- if (exists("dataset", envir = env, inherits = FALSE)) {
    get("dataset", envir = env, inherits = FALSE)
  } else {
    NULL
  }
  if (is.list(obj) && !is.null(obj$data)) return(obj$data)
  for (name in ls(env)) {
    value <- get(name, envir = env, inherits = FALSE)
    if (is.list(value) && !is.null(value$data)) return(value$data)
  }
  for (name in ls(env)) {
    value <- get(name, envir = env, inherits = FALSE)
    if (inherits(value, "float32") || is.matrix(value)) return(value)
  }
  stop("Dataset file must contain a matrix or a list with `$data`: ", path, call. = FALSE)
}

subset_rows <- function(x, rows) {
  x[rows, , drop = FALSE]
}

remove_self_from_query_knn <- function(indices, rows, k) {
  out <- matrix(NA_integer_, nrow = length(rows), ncol = k)
  for (i in seq_along(rows)) {
    keep <- which(indices[i, ] != rows[[i]])
    keep <- keep[seq_len(min(length(keep), k))]
    if (length(keep)) out[i, seq_along(keep)] <- indices[i, keep]
  }
  out
}

recall_summary <- function(actual_indices, reference_indices) {
  if (is.null(reference_indices) || !length(reference_indices)) {
    return(list(recall_at_k = NA_real_, median_recall_at_k = NA_real_, min_recall_at_k = NA_real_))
  }
  n <- min(nrow(actual_indices), nrow(reference_indices))
  vals <- numeric(n)
  for (i in seq_len(n)) {
    ref <- reference_indices[i, ]
    ref <- ref[!is.na(ref)]
    got <- actual_indices[i, ]
    got <- got[!is.na(got)]
    vals[[i]] <- if (length(ref)) sum(got %in% ref) / length(ref) else NA_real_
  }
  vals <- vals[is.finite(vals)]
  list(
    recall_at_k = if (length(vals)) mean(vals) else NA_real_,
    median_recall_at_k = if (length(vals)) stats::median(vals) else NA_real_,
    min_recall_at_k = if (length(vals)) min(vals) else NA_real_
  )
}

shape_group <- function(n, p) {
  if (!is.finite(n) || !is.finite(p)) return("unknown")
  if (n < 50000L) return("small_n")
  if (n >= 5000000L && p <= 64L) return("huge_low_dim")
  if (n >= 50000L && n < 500000L && p <= 64L) return("medium_low_dim")
  if (n >= 500000L && p <= 64L) return("large_low_dim")
  if (n >= 50000L && p >= 256L) return("large_high_dim")
  "other"
}

normalize_manifest <- function(manifest) {
  if (!"dataset" %in% names(manifest)) {
    stop("Manifest must contain a `dataset` column.", call. = FALSE)
  }
  if (!"path" %in% names(manifest)) {
    path_candidates <- intersect(c("output", "file", "file_path", "rdata_path"), names(manifest))
    if (!length(path_candidates)) {
      stop("Manifest must contain a `path` column or an equivalent file path column.", call. = FALSE)
    }
    manifest$path <- manifest[[path_candidates[[1L]]]]
  }
  if (!"status" %in% names(manifest)) manifest$status <- "success"
  if (!"error" %in% names(manifest)) manifest$error <- NA_character_
  if (!all(c("n", "p") %in% names(manifest))) {
    stop("Manifest must contain `n` and `p` columns.", call. = FALSE)
  }
  manifest
}

manifest_success <- function(status) {
  tolower(as.character(status %||% "success")) %in% c("success", "ok", "available", "found", "true", "1")
}

cpu_candidates <- function(k, grid_level = "standard") {
  base <- data.frame(
    candidate_id = "auto",
    candidate_kind = "auto",
    pool_size = NA_integer_,
    n_iters = NA_integer_,
    max_candidates = NA_integer_,
    n_random_projections = NA_integer_,
    graph_degree = NA_integer_,
    intermediate_graph_degree = NA_integer_,
    max_iterations = NA_integer_,
    stringsAsFactors = FALSE
  )
  specs <- data.frame(
    label = c("very_fast", "fast", "balanced", "balanced_plus", "recall", "recall_plus"),
    pool_mult = c(1.25, 1.5, 2.0, 2.5, 3.0, 4.0),
    pool_min = c(24L, 32L, 48L, 80L, 120L, 160L),
    iters = c(1L, 2L, 3L, 4L, 5L, 6L),
    candidate_mult = c(2L, 3L, 4L, 4L, 5L, 6L),
    projections = c(3L, 4L, 6L, 8L, 10L, 12L),
    stringsAsFactors = FALSE
  )
  if (identical(grid_level, "compact")) specs <- specs[c(2L, 3L, 5L), , drop = FALSE]
  if (identical(grid_level, "wide")) {
    specs <- rbind(
      specs,
      data.frame(
        label = c("recall_wide", "recall_max"),
        pool_mult = c(5.0, 6.0),
        pool_min = c(220L, 320L),
        iters = c(8L, 10L),
        candidate_mult = c(6L, 8L),
        projections = c(16L, 24L),
        stringsAsFactors = FALSE
      )
    )
  }
  manual <- do.call(rbind, lapply(seq_len(nrow(specs)), function(i) {
    pool <- as.integer(max(k, ceiling(specs$pool_mult[[i]] * k), specs$pool_min[[i]]))
    data.frame(
      candidate_id = paste0("cpu_", specs$label[[i]], "_pool", pool, "_it", specs$iters[[i]]),
      candidate_kind = "manual",
      pool_size = pool,
      n_iters = as.integer(specs$iters[[i]]),
      max_candidates = as.integer(max(pool, pool * specs$candidate_mult[[i]])),
      n_random_projections = as.integer(specs$projections[[i]]),
      graph_degree = NA_integer_,
      intermediate_graph_degree = NA_integer_,
      max_iterations = NA_integer_,
      stringsAsFactors = FALSE
    )
  }))
  rbind(base, manual)
}

cuda_candidates <- function(k, grid_level = "standard") {
  base <- data.frame(
    candidate_id = "auto",
    candidate_kind = "auto",
    pool_size = NA_integer_,
    n_iters = NA_integer_,
    max_candidates = NA_integer_,
    n_random_projections = NA_integer_,
    graph_degree = NA_integer_,
    intermediate_graph_degree = NA_integer_,
    max_iterations = NA_integer_,
    stringsAsFactors = FALSE
  )
  specs <- data.frame(
    label = c("fast", "balanced", "balanced_plus", "recall", "recall_plus"),
    graph_mult = c(1.0, 1.0, 1.12, 1.28, 1.5),
    graph_min = c(k, 32L, 48L, 64L, 96L),
    intermediate_mult = c(1.0, 1.25, 1.5, 1.75, 2.0),
    iterations = c(5L, 8L, 12L, 16L, 24L),
    stringsAsFactors = FALSE
  )
  if (identical(grid_level, "compact")) specs <- specs[c(1L, 3L, 4L), , drop = FALSE]
  if (identical(grid_level, "wide")) {
    specs <- rbind(
      specs,
      data.frame(
        label = c("recall_wide", "recall_max"),
        graph_mult = c(4.0, 5.0),
        graph_min = c(128L, 192L),
        intermediate_mult = c(2.5, 3.0),
        iterations = c(32L, 48L),
        stringsAsFactors = FALSE
      )
    )
  }
  manual <- do.call(rbind, lapply(seq_len(nrow(specs)), function(i) {
    graph_degree <- as.integer(max(k, ceiling(specs$graph_mult[[i]] * k), specs$graph_min[[i]]))
    intermediate <- as.integer(max(graph_degree, ceiling(graph_degree * specs$intermediate_mult[[i]])))
    data.frame(
      candidate_id = paste0("cuda_", specs$label[[i]], "_gd", graph_degree, "_it", specs$iterations[[i]]),
      candidate_kind = "manual",
      pool_size = NA_integer_,
      n_iters = NA_integer_,
      max_candidates = NA_integer_,
      n_random_projections = NA_integer_,
      graph_degree = graph_degree,
      intermediate_graph_degree = intermediate,
      max_iterations = as.integer(specs$iterations[[i]]),
      stringsAsFactors = FALSE
    )
  }))
  rbind(base, manual)
}

candidate_grid <- function(backend, k, grid_level) {
  if (identical(backend, "cpu")) cpu_candidates(k, grid_level) else cuda_candidates(k, grid_level)
}

apply_candidate_options <- function(candidate, backend) {
  if (identical(backend, "cpu")) {
    opts <- list(
      faissR.cpu_nndescent_prefer_faiss = FALSE,
      faissR.cpu_nndescent_pool_size = NULL,
      faissR.cpu_nndescent_n_iters = NULL,
      faissR.cpu_nndescent_max_candidates = NULL,
      faissR.cpu_nndescent_n_random_projections = NULL
    )
    if (!identical(candidate$candidate_kind, "auto")) {
      opts$faissR.cpu_nndescent_pool_size <- as.integer(candidate$pool_size)
      opts$faissR.cpu_nndescent_n_iters <- as.integer(candidate$n_iters)
      opts$faissR.cpu_nndescent_max_candidates <- as.integer(candidate$max_candidates)
      opts$faissR.cpu_nndescent_n_random_projections <- as.integer(candidate$n_random_projections)
    }
    do.call(options, opts)
  } else {
    opts <- list(
      faissR.cuvs_nndescent_graph_degree = NULL,
      faissR.cuvs_nndescent_intermediate_graph_degree = NULL,
      faissR.cuvs_nndescent_max_iterations = NULL
    )
    if (!identical(candidate$candidate_kind, "auto")) {
      opts$faissR.cuvs_nndescent_graph_degree <- as.integer(candidate$graph_degree)
      opts$faissR.cuvs_nndescent_intermediate_graph_degree <- as.integer(candidate$intermediate_graph_degree)
      opts$faissR.cuvs_nndescent_max_iterations <- as.integer(candidate$max_iterations)
    }
    do.call(options, opts)
  }
  invisible(TRUE)
}

estimate_working_gb <- function(n, candidate, backend) {
  n <- as.numeric(n)
  if (identical(candidate$candidate_kind, "auto")) return(NA_real_)
  if (identical(backend, "cpu")) {
    pool <- as.numeric(candidate$pool_size)
    max_candidates <- as.numeric(candidate$max_candidates)
    return(n * (8 * pool + 4 * max_candidates + pool) / 1024^3)
  }
  graph_degree <- as.numeric(candidate$graph_degree)
  intermediate <- as.numeric(candidate$intermediate_graph_degree)
  n * (8 * graph_degree + 4 * intermediate) / 1024^3
}

base_row <- function(config, status = "success", error = NA_character_) {
  candidate <- config$candidate
  data.frame(
    dataset = config$dataset,
    n = as.integer(config$n),
    p = as.integer(config$p),
    shape_group = shape_group(config$n, config$p),
    backend = config$backend,
    method = "nndescent",
    metric = "euclidean",
    k = as.integer(config$k),
    candidate_id = candidate$candidate_id,
    candidate_kind = candidate$candidate_kind,
    n_threads = as.integer(config$n_threads),
    output = config$output,
    status = status,
    elapsed_sec = NA_real_,
    peak_rss_gb = NA_real_,
    recall_at_k = NA_real_,
    median_recall_at_k = NA_real_,
    min_recall_at_k = NA_real_,
    reference_status = config$reference_status %||% NA_character_,
    reference_backend = config$reference_backend %||% NA_character_,
    reference_query_n = as.integer(config$reference_query_n %||% NA_integer_),
    estimated_working_gb = as.numeric(config$estimated_working_gb %||% NA_real_),
    cpu_pool_size = as.integer(candidate$pool_size),
    cpu_n_iters = as.integer(candidate$n_iters),
    cpu_max_candidates = as.integer(candidate$max_candidates),
    cpu_n_random_projections = as.integer(candidate$n_random_projections),
    cuda_graph_degree = as.integer(candidate$graph_degree),
    cuda_intermediate_graph_degree = as.integer(candidate$intermediate_graph_degree),
    cuda_max_iterations = as.integer(candidate$max_iterations),
    result_backend = NA_character_,
    implementation_backend = NA_character_,
    resolved_backend = NA_character_,
    distance_type = NA_character_,
    input_type = NA_character_,
    input_layout = NA_character_,
    tuning_rule = NA_character_,
    result_pool_size = NA_integer_,
    result_n_iters = NA_integer_,
    result_max_candidates = NA_integer_,
    result_n_random_projections = NA_integer_,
    result_graph_degree = NA_integer_,
    result_intermediate_graph_degree = NA_integer_,
    result_max_iterations = NA_integer_,
    error = error,
    stringsAsFactors = FALSE
  )
}

run_reference <- function(config) {
  configure_native_libs()
  configure_threads(config$n_threads)
  load_faissR()
  x <- load_float_dataset(config$path)
  rows <- config$quality_rows
  k <- as.integer(config$k)
  started <- proc.time()[["elapsed"]]
  ref <- faissR::nn(
    x,
    points = subset_rows(x, rows),
    k = min(k + 1L, nrow(x)),
    backend = config$reference_backend,
    method = "exact",
    metric = "euclidean",
    n_threads = config$n_threads
  )
  elapsed <- proc.time()[["elapsed"]] - started
  list(
    status = "success",
    elapsed_sec = as.numeric(elapsed),
    peak_rss_gb = read_peak_rss_gb(),
    rows = rows,
    indices = remove_self_from_query_knn(ref$indices, rows, k),
    backend = attr(ref, "backend") %||% config$reference_backend
  )
}

run_method <- function(config) {
  configure_native_libs()
  configure_threads(config$n_threads)
  load_faissR()
  x <- load_float_dataset(config$path)
  apply_candidate_options(config$candidate, config$backend)
  started <- proc.time()[["elapsed"]]
  res <- faissR::nn(
    x,
    k = as.integer(config$k),
    exclude_self = TRUE,
    backend = config$backend,
    method = "nndescent",
    metric = "euclidean",
    output = config$output,
    n_threads = config$n_threads
  )
  elapsed <- proc.time()[["elapsed"]] - started
  approx <- attr(res, "approximation", exact = TRUE)
  row <- base_row(config, status = "success")
  recall <- recall_summary(res$indices[config$quality_rows, , drop = FALSE], config$reference_indices)
  row$elapsed_sec <- as.numeric(elapsed)
  row$peak_rss_gb <- read_peak_rss_gb()
  row$recall_at_k <- recall$recall_at_k
  row$median_recall_at_k <- recall$median_recall_at_k
  row$min_recall_at_k <- recall$min_recall_at_k
  row$result_backend <- attr(res, "backend") %||% NA_character_
  row$implementation_backend <- attr(res, "implementation_backend") %||% NA_character_
  row$resolved_backend <- attr(res, "resolved_backend") %||% row$result_backend
  row$distance_type <- res$distance_type %||% attr(res, "distance_type") %||% "double"
  row$input_type <- res$input_type %||% NA_character_
  row$input_layout <- res$input_layout %||% NA_character_
  row$tuning_rule <- approx$tuning_rule %||% NA_character_
  row$result_pool_size <- as.integer(approx$pool_size %||% NA_integer_)
  row$result_n_iters <- as.integer(approx$n_iters %||% NA_integer_)
  row$result_max_candidates <- as.integer(approx$max_candidates %||% NA_integer_)
  row$result_n_random_projections <- as.integer(approx$n_random_projections %||% NA_integer_)
  row$result_graph_degree <- as.integer(approx$graph_degree %||% NA_integer_)
  row$result_intermediate_graph_degree <- as.integer(approx$intermediate_graph_degree %||% NA_integer_)
  row$result_max_iterations <- as.integer(approx$max_iterations %||% NA_integer_)
  row
}

run_child_task <- function() {
  args <- parse_args()
  config <- readRDS(args$config)
  result <- tryCatch(
    if (identical(args$task, "reference")) run_reference(config) else run_method(config),
    error = function(e) {
      if (identical(args$task, "reference")) {
        list(status = classify_error(conditionMessage(e)), error = conditionMessage(e))
      } else {
        base_row(config, status = classify_error(conditionMessage(e)), error = conditionMessage(e))
      }
    }
  )
  saveRDS(result, args$result)
}

run_rscript_task <- function(task, config, timeout, bench_script) {
  config_path <- tempfile("faissR_nndescent_config_", fileext = ".rds")
  result_path <- tempfile("faissR_nndescent_result_", fileext = ".rds")
  saveRDS(config, config_path)
  on.exit(unlink(c(config_path, result_path)), add = TRUE)
  rscript <- Sys.getenv("R_BIN", "Rscript")
  timeout_bin <- Sys.which("timeout")
  cmd <- c(rscript, "--vanilla", bench_script, paste0("--task=", task),
           paste0("--config=", config_path), paste0("--result=", result_path))
  if (nzchar(timeout_bin)) {
    cmd <- c(timeout_bin, as.character(as.integer(ceiling(timeout))), cmd)
  }
  shell_cmd <- paste(shQuote(cmd), collapse = " ")
  shell_cmd <- paste(shell_cmd, "2>&1")
  status <- system(shell_cmd, intern = TRUE)
  exit_status <- attr(status, "status") %||% 0L
  timed_out <- nzchar(timeout_bin) && identical(as.integer(exit_status), 124L)
  if (file.exists(result_path) && !timed_out) {
    return(readRDS(result_path))
  }
  if (identical(task, "reference")) {
    return(list(
      status = if (timed_out) "timeout" else classify_error(paste(status, collapse = "\n")),
      error = if (timed_out) sprintf("reference timed out after %s seconds", timeout) else paste(status, collapse = "\n")
    ))
  }
  row <- base_row(
    config,
    status = if (timed_out) "timeout" else classify_error(paste(status, collapse = "\n")),
    error = if (timed_out) sprintf("child process timed out after %s seconds", timeout) else paste(status, collapse = "\n")
  )
  row$elapsed_sec <- if (timed_out) as.numeric(timeout) else NA_real_
  row
}

reference_for_dataset <- function(dataset_row, backend, k, n_threads, output,
                                  quality_n, seed, timeout, bench_script,
                                  reference_backend) {
  n <- as.integer(dataset_row$n)
  set.seed(seed + k + nchar(dataset_row$dataset))
  rows <- sort(sample(seq_len(n), min(quality_n, n)))
  cfg <- list(
    dataset = dataset_row$dataset,
    path = dataset_row$path,
    n = n,
    p = as.integer(dataset_row$p),
    backend = backend,
    reference_backend = reference_backend,
    k = as.integer(k),
    quality_rows = rows,
    n_threads = as.integer(n_threads),
    output = output,
    candidate = candidate_grid(backend, k, "compact")[1L, , drop = FALSE],
    reference_query_n = length(rows)
  )
  run_rscript_task("reference", cfg, timeout, bench_script)
}

row_key <- function(dataset, backend, k, candidate_id) {
  paste(dataset, backend, as.integer(k), candidate_id, sep = "\r")
}

completed_keys <- function(path) {
  if (!file.exists(path)) return(character())
  x <- tryCatch(read.csv(path, stringsAsFactors = FALSE), error = function(e) NULL)
  if (is.null(x) || !nrow(x) || !all(c("dataset", "backend", "k", "candidate_id") %in% names(x))) {
    return(character())
  }
  row_key(x$dataset, x$backend, x$k, x$candidate_id)
}

write_missing_rows <- function(dataset_row, backend, k, candidates, reason, results_path) {
  for (i in seq_len(nrow(candidates))) {
    cfg <- list(
      dataset = dataset_row$dataset,
      n = as.integer(dataset_row$n %||% NA_integer_),
      p = as.integer(dataset_row$p %||% NA_integer_),
      backend = backend,
      k = as.integer(k),
      n_threads = NA_integer_,
      output = NA_character_,
      candidate = candidates[i, , drop = FALSE],
      reference_status = "missing_dataset",
      reference_backend = NA_character_,
      reference_query_n = NA_integer_
    )
    append_csv(base_row(cfg, status = "missing_dataset", error = reason), results_path)
  }
}

summarize_results <- function(out_dir, target_recalls) {
  results_path <- file.path(out_dir, "nndescent_tuning_results.csv")
  if (!file.exists(results_path)) return(invisible(NULL))
  x <- read.csv(results_path, stringsAsFactors = FALSE)
  success <- x[x$status == "success" & is.finite(x$recall_at_k) & is.finite(x$elapsed_sec), , drop = FALSE]
  rec_rows <- list()
  idx <- 0L
  groups <- unique(success[c("dataset", "backend", "k")])
  for (g in seq_len(nrow(groups))) {
    part0 <- success[
      success$dataset == groups$dataset[[g]] &
        success$backend == groups$backend[[g]] &
        success$k == groups$k[[g]], , drop = FALSE
    ]
    for (target in target_recalls) {
      part <- part0
      part$meets_target <- part$recall_at_k >= target
      qualified <- part[part$meets_target, , drop = FALSE]
      if (nrow(qualified)) {
        qualified <- qualified[order(qualified$elapsed_sec, -qualified$recall_at_k), , drop = FALSE]
        best <- qualified[1L, , drop = FALSE]
        basis <- "fastest_meeting_target"
      } else {
        part <- part[order(-part$recall_at_k, part$elapsed_sec), , drop = FALSE]
        best <- part[1L, , drop = FALSE]
        basis <- "best_recall_below_target"
      }
      idx <- idx + 1L
      rec_rows[[idx]] <- cbind(
        target_recall_threshold = target,
        recommendation_basis = basis,
        best
      )
    }
  }
  recommendations <- if (length(rec_rows)) do.call(rbind, rec_rows) else data.frame()
  write.csv(recommendations, file.path(out_dir, "nndescent_tuning_recommendations.csv"), row.names = FALSE)

  if (nrow(success)) {
    shape_rows <- list()
    idx <- 0L
    for (target in target_recalls) {
      success$meets_target <- success$recall_at_k >= target
      keys <- c("shape_group", "backend", "k", "candidate_id")
      split_key <- interaction(success[keys], drop = TRUE, lex.order = TRUE)
      for (part in split(success, split_key)) {
        idx <- idx + 1L
        shape_rows[[idx]] <- data.frame(
          shape_group = part$shape_group[[1L]],
          backend = part$backend[[1L]],
          k = part$k[[1L]],
          target_recall_threshold = target,
          candidate_id = part$candidate_id[[1L]],
          candidate_kind = part$candidate_kind[[1L]],
          n_success = nrow(part),
          meet_rate = mean(part$meets_target),
          median_elapsed_sec = stats::median(part$elapsed_sec),
          median_recall_at_k = stats::median(part$recall_at_k),
          min_recall_at_k = min(part$recall_at_k),
          stringsAsFactors = FALSE
        )
      }
    }
    shape_candidates <- do.call(rbind, shape_rows)
    shape_candidates <- shape_candidates[order(
      shape_candidates$shape_group,
      shape_candidates$backend,
      shape_candidates$k,
      shape_candidates$target_recall_threshold,
      -shape_candidates$meet_rate,
      shape_candidates$median_elapsed_sec,
      -shape_candidates$min_recall_at_k
    ), , drop = FALSE]
    write.csv(shape_candidates, file.path(out_dir, "nndescent_tuning_shape_candidates.csv"), row.names = FALSE)
    shape_best <- do.call(rbind, lapply(
      split(shape_candidates, interaction(shape_candidates[c("shape_group", "backend", "k", "target_recall_threshold")], drop = TRUE, lex.order = TRUE)),
      function(part) part[1L, , drop = FALSE]
    ))
    row.names(shape_best) <- NULL
    write.csv(shape_best, file.path(out_dir, "nndescent_tuning_shape_recommendations.csv"), row.names = FALSE)
  }
  write_report(out_dir, x, recommendations)
  invisible(NULL)
}

write_report <- function(out_dir, results, recommendations) {
  lines <- c(
    "# NN-descent Tuning Report",
    "",
    sprintf("Generated: %s", format(Sys.time(), "%Y-%m-%d %H:%M:%S %Z")),
    "",
    "This benchmark evaluates explicit `method = \"nndescent\"` settings for",
    "Euclidean self-KNN. CPU rows use faissR's native CPU NN-descent route;",
    "CUDA rows use RAPIDS cuVS NN-descent. Recommendation tables select the",
    "fastest successful candidate reaching recall thresholds 0.90, 0.95, and",
    "0.99, or the highest-recall candidate when no row reaches the threshold.",
    "",
    "## Status Counts",
    "",
    paste(capture.output(print(table(results$status, useNA = "ifany"))), collapse = "\n"),
    "",
    "## Output Files",
    "",
    "- `nndescent_tuning_candidate_grid.csv`",
    "- `nndescent_tuning_results.csv`",
    "- `nndescent_tuning_recommendations.csv`",
    "- `nndescent_tuning_shape_candidates.csv`",
    "- `nndescent_tuning_shape_recommendations.csv`"
  )
  if (nrow(recommendations)) {
    top <- recommendations[seq_len(min(20L, nrow(recommendations))), c(
      "dataset", "backend", "k", "target_recall_threshold",
      "candidate_id", "elapsed_sec", "recall_at_k", "recommendation_basis"
    ), drop = FALSE]
    lines <- c(lines, "", "## First Recommendations", "", paste(capture.output(print(top, row.names = FALSE)), collapse = "\n"))
  }
  writeLines(lines, file.path(out_dir, "nndescent_tuning_report.md"))
}

main <- function() {
  args <- parse_args()
  if (!is.null(args$task)) return(run_child_task())

  configure_native_libs()
  backend <- match.arg(tolower(args$backend %||% "cpu"), c("cpu", "cuda"))
  manifest_path <- args$manifest %||% stop("`--manifest` is required.", call. = FALSE)
  out_dir <- args$out_dir %||% file.path(getwd(), paste0("faissR_NNDESCENT_TUNING_", toupper(backend), "_", format(Sys.time(), "%Y%m%d_%H%M%S")))
  dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
  results_path <- file.path(out_dir, "nndescent_tuning_results.csv")
  candidate_path <- file.path(out_dir, "nndescent_tuning_candidate_grid.csv")
  bench_script <- normalizePath(script_path(), mustWork = TRUE)
  datasets <- split_arg(args$datasets, "")
  k_values <- as.integer(split_arg(args$k_values, "10,15,50,100"))
  target_recalls <- as.numeric(split_arg(args$target_recalls, "0.9,0.95,0.99"))
  n_threads <- positive_int(args$threads, if (backend == "cpu") 12 else 2, "threads")
  timeout <- positive_num(args$timeout, 2000, "timeout")
  quality_n <- positive_int(args$quality_n, 256, "quality_n")
  seed <- positive_int(args$seed, 4, "seed")
  output <- args$output %||% if (backend == "cuda") "float" else "double"
  grid_level <- match.arg(args$grid_level %||% "standard", c("compact", "standard", "wide"))
  reference_backend <- match.arg(tolower(args$reference_backend %||% backend), c("cpu", "cuda"))
  resume <- logical_arg(args$resume, TRUE)
  max_working_gb <- positive_num(args$max_working_gb, if (backend == "cuda") 40 else 96, "max_working_gb")

  manifest <- normalize_manifest(read.csv(manifest_path, stringsAsFactors = FALSE))
  if (length(datasets) && nzchar(datasets[[1L]])) {
    manifest <- manifest[manifest$dataset %in% datasets, , drop = FALSE]
  }
  candidate_rows <- do.call(rbind, lapply(k_values, function(k) {
    x <- candidate_grid(backend, k, grid_level)
    x$backend <- backend
    x$k <- as.integer(k)
    x
  }))
  write.csv(candidate_rows, candidate_path, row.names = FALSE)
  write.csv(
    data.frame(
      key = c("backend", "manifest", "datasets", "k_values", "target_recalls",
              "threads", "timeout", "quality_n", "output", "grid_level",
              "reference_backend", "max_working_gb"),
      value = c(backend, manifest_path, paste(datasets, collapse = ","),
                paste(k_values, collapse = ","), paste(target_recalls, collapse = ","),
                n_threads, timeout, quality_n, output, grid_level,
                reference_backend, max_working_gb)
    ),
    file.path(out_dir, "nndescent_tuning_config.csv"),
    row.names = FALSE
  )

  keys_done <- if (resume) completed_keys(results_path) else character()
  for (i in seq_len(nrow(manifest))) {
    ds <- manifest[i, , drop = FALSE]
    dataset <- ds$dataset[[1L]]
    if (!manifest_success(ds$status[[1L]]) || !file.exists(ds$path[[1L]])) {
      for (k in k_values) {
        write_missing_rows(ds, backend, k, candidate_grid(backend, k, grid_level), ds$error[[1L]] %||% "missing dataset", results_path)
      }
      next
    }
    for (k in k_values) {
      candidates <- candidate_grid(backend, k, grid_level)
      ref <- reference_for_dataset(
        ds, backend, k, n_threads, output, quality_n, seed, timeout,
        bench_script, reference_backend
      )
      if (!identical(ref$status, "success")) {
        for (j in seq_len(nrow(candidates))) {
          cfg <- list(
            dataset = dataset,
            n = as.integer(ds$n),
            p = as.integer(ds$p),
            backend = backend,
            k = as.integer(k),
            n_threads = as.integer(n_threads),
            output = output,
            candidate = candidates[j, , drop = FALSE],
            reference_status = ref$status,
            reference_backend = reference_backend,
            reference_query_n = quality_n
          )
          append_csv(base_row(cfg, status = "reference_failed", error = ref$error %||% "reference failed"), results_path)
        }
        next
      }
      for (j in seq_len(nrow(candidates))) {
        candidate <- candidates[j, , drop = FALSE]
        key <- row_key(dataset, backend, k, candidate$candidate_id)
        if (key %in% keys_done) {
          message(sprintf("[%s] skip completed dataset=%s backend=%s k=%s candidate=%s",
                          Sys.time(), dataset, backend, k, candidate$candidate_id))
          next
        }
        estimated_gb <- estimate_working_gb(as.integer(ds$n), candidate, backend)
        cfg <- list(
          dataset = dataset,
          path = ds$path[[1L]],
          n = as.integer(ds$n),
          p = as.integer(ds$p),
          backend = backend,
          k = as.integer(k),
          quality_rows = ref$rows,
          reference_indices = ref$indices,
          reference_status = ref$status,
          reference_backend = ref$backend %||% reference_backend,
          reference_query_n = length(ref$rows),
          n_threads = as.integer(n_threads),
          output = output,
          candidate = candidate,
          estimated_working_gb = estimated_gb
        )
        if (is.finite(estimated_gb) && estimated_gb > max_working_gb) {
          append_csv(
            base_row(cfg, status = "expected_skip", error = sprintf("estimated working memory %.2f GB exceeds max_working_gb %.2f", estimated_gb, max_working_gb)),
            results_path
          )
          next
        }
        message(sprintf("[%s] run dataset=%s backend=%s k=%s candidate=%s",
                        Sys.time(), dataset, backend, k, candidate$candidate_id))
        append_csv(run_rscript_task("method", cfg, timeout, bench_script), results_path)
      }
      summarize_results(out_dir, target_recalls)
    }
  }
  summarize_results(out_dir, target_recalls)
  message("DONE: ", out_dir)
}

main()
