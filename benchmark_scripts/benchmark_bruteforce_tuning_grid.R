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
  normalizePath(file.path("benchmark_scripts", "benchmark_bruteforce_tuning_grid.R"), mustWork = FALSE)
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

clean_int_values <- function(values, default) {
  out <- suppressWarnings(as.integer(split_arg(values, default)))
  out <- out[is.finite(out) & out > 0L]
  unique(out)
}

clean_output_values <- function(values, default = "float,double") {
  out <- tolower(split_arg(values, default))
  out <- out[out %in% c("float", "double")]
  unique(out)
}

candidate_grid <- function(backend, k, thread_values, output_values) {
  grid <- expand.grid(
    n_threads = thread_values,
    output = output_values,
    KEEP.OUT.ATTRS = FALSE,
    stringsAsFactors = FALSE
  )
  grid$device <- backend
  grid$provider <- if (identical(backend, "cuda")) "cuda_bruteforce" else "cpu_bruteforce"
  grid$public_backend <- backend
  grid$public_method <- "bruteforce"
  grid$faiss_query_batch_size <- NA_integer_
  grid$faiss_gpu_query_batch_size <- NA_integer_
  grid$faiss_gpu_reuse_resources <- NA
  grid$cache_fitted_indexes <- NA
  grid$candidate_id <- sprintf(
    "%s_t%d_%s",
    grid$provider,
    grid$n_threads,
    grid$output
  )
  grid$candidate_kind <- "manual"
  grid$k <- as.integer(k)
  grid
}

apply_candidate_runtime <- function(candidate) {
  Sys.unsetenv(c(
    "FAISSR_FAISS_QUERY_BATCH_SIZE",
    "FAISSR_FAISS_GPU_QUERY_BATCH_SIZE",
    "FAISSR_FAISS_GPU_REUSE_RESOURCES"
  ))
  if (!is.na(candidate$faiss_query_batch_size)) {
    Sys.setenv(FAISSR_FAISS_QUERY_BATCH_SIZE = as.integer(candidate$faiss_query_batch_size))
  }
  if (!is.na(candidate$faiss_gpu_query_batch_size)) {
    Sys.setenv(FAISSR_FAISS_GPU_QUERY_BATCH_SIZE = as.integer(candidate$faiss_gpu_query_batch_size))
  }
  if (!is.na(candidate$faiss_gpu_reuse_resources)) {
    Sys.setenv(FAISSR_FAISS_GPU_REUSE_RESOURCES = if (isTRUE(candidate$faiss_gpu_reuse_resources)) "1" else "0")
  }
  invisible(TRUE)
}

estimate_working_gb <- function(n, p, k, candidate) {
  n <- as.numeric(n)
  p <- as.numeric(p)
  search_k <- as.numeric(k) + 1
  data_gb <- n * p * 4 / 1024^3
  cpp_result_gb <- n * search_k * 12 / 1024^3
  r_result_gb <- n * as.numeric(k) * (4 + if (identical(candidate$output, "float")) 4 else 8) / 1024^3
  gpu_extra <- if (identical(candidate$device, "cuda")) data_gb else 0
  data_gb + cpp_result_gb + r_result_gb + gpu_extra
}

base_row <- function(config, status = "success", error = NA_character_) {
  candidate <- config$candidate
  data.frame(
    dataset = config$dataset,
    n = as.integer(config$n),
    p = as.integer(config$p),
    shape_group = shape_group(config$n, config$p),
    device = candidate$device,
    provider = candidate$provider,
    public_backend = candidate$public_backend,
    public_method = candidate$public_method,
    method_family = "bruteforce",
    metric = "euclidean",
    k = as.integer(config$k),
    candidate_id = candidate$candidate_id,
    candidate_kind = candidate$candidate_kind,
    n_threads = as.integer(candidate$n_threads),
    output = candidate$output,
    status = status,
    elapsed_sec = NA_real_,
    peak_rss_gb = NA_real_,
    recall_at_k = NA_real_,
    median_recall_at_k = NA_real_,
    min_recall_at_k = NA_real_,
    quality_status = config$quality_status %||% NA_character_,
    reference_status = config$reference_status %||% NA_character_,
    reference_backend = config$reference_backend %||% NA_character_,
    reference_query_n = as.integer(config$reference_query_n %||% NA_integer_),
    estimated_working_gb = as.numeric(config$estimated_working_gb %||% NA_real_),
    requested_faiss_query_batch_size = as.integer(candidate$faiss_query_batch_size),
    requested_faiss_gpu_query_batch_size = as.integer(candidate$faiss_gpu_query_batch_size),
    requested_faiss_gpu_reuse_resources = candidate$faiss_gpu_reuse_resources,
    requested_cache_fitted_indexes = candidate$cache_fitted_indexes,
    result_exact = NA,
    result_backend = NA_character_,
    implementation_backend = NA_character_,
    resolved_backend = NA_character_,
    index_type = NA_character_,
    distance_type = NA_character_,
    input_type = NA_character_,
    input_layout = NA_character_,
    search_batch_size = NA_integer_,
    search_batches = NA_integer_,
    query_call_count = NA_integer_,
    persistent_index_cache = NA,
    index_cache_hit = NA,
    gpu_provider = NA_character_,
    gpu_resources_reused = NA,
    device_residency = NA_character_,
    index_residency = NA_character_,
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
    output = config$output,
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
  configure_threads(config$candidate$n_threads)
  load_faissR()
  x <- load_float_dataset(config$path)
  candidate <- config$candidate
  apply_candidate_runtime(candidate)
  started <- proc.time()[["elapsed"]]
  res <- faissR::nn(
    x,
    k = as.integer(config$k),
    exclude_self = TRUE,
    backend = as.character(candidate$public_backend),
    method = as.character(candidate$public_method),
    metric = "euclidean",
    output = as.character(candidate$output),
    n_threads = as.integer(candidate$n_threads),
    tuning = "fixed"
  )
  elapsed <- proc.time()[["elapsed"]] - started
  row <- base_row(config, status = "success")
  result_exact <- isTRUE(attr(res, "exact"))
  if (!is.null(config$reference_indices) && length(config$reference_indices)) {
    recall <- recall_summary(res$indices[config$quality_rows, , drop = FALSE], config$reference_indices)
    row$recall_at_k <- recall$recall_at_k
    row$median_recall_at_k <- recall$median_recall_at_k
    row$min_recall_at_k <- recall$min_recall_at_k
    row$quality_status <- "sample_exact_reference"
  } else if (isTRUE(result_exact) && isTRUE(config$assume_exact_recall)) {
    row$recall_at_k <- 1
    row$median_recall_at_k <- 1
    row$min_recall_at_k <- 1
    row$quality_status <- "exact_assumed"
  } else {
    row$quality_status <- "not_available"
  }
  faiss <- attr(res, "faiss", exact = TRUE)
  cuvs <- attr(res, "cuvs", exact = TRUE)
  gpu <- attr(res, "gpu_residency", exact = TRUE)
  row$elapsed_sec <- as.numeric(elapsed)
  row$peak_rss_gb <- read_peak_rss_gb()
  row$result_exact <- result_exact
  row$result_backend <- res$backend_used %||% attr(res, "backend") %||% NA_character_
  row$implementation_backend <- attr(res, "implementation_backend") %||% NA_character_
  row$resolved_backend <- attr(res, "resolved_backend") %||% row$result_backend
  row$index_type <- res$index_type %||% faiss$index_type %||% cuvs$index_type %||% NA_character_
  row$distance_type <- res$distance_type %||% attr(res, "distance_type") %||% "double"
  row$input_type <- res$input_type %||% attr(res, "input_type") %||% faiss$input_type %||% cuvs$input_type %||% NA_character_
  row$input_layout <- res$input_layout %||% attr(res, "input_layout") %||% NA_character_
  row$search_batch_size <- as.integer(res$search_batch_size %||% faiss$search_batch_size %||% NA_integer_)
  row$search_batches <- as.integer(res$search_batches %||% faiss$search_batches %||% NA_integer_)
  row$query_call_count <- as.integer(res$query_call_count %||% faiss$query_call_count %||% NA_integer_)
  row$persistent_index_cache <- faiss$persistent_index_cache %||% NA
  row$index_cache_hit <- faiss$index_cache_hit %||% NA
  row$gpu_provider <- res$gpu_provider %||% gpu$gpu_provider %||% NA_character_
  row$gpu_resources_reused <- res$gpu_resources_reused %||% gpu$gpu_resources_reused %||% NA
  row$device_residency <- res$device_residency %||% gpu$device_residency %||% NA_character_
  row$index_residency <- res$index_residency %||% gpu$index_residency %||% NA_character_
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
  config_path <- tempfile("faissR_bruteforce_config_", fileext = ".rds")
  result_path <- tempfile("faissR_bruteforce_result_", fileext = ".rds")
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

reference_for_dataset <- function(dataset_row, k, n_threads, output,
                                  quality_n, seed, timeout, bench_script,
                                  reference_backend, quality_mode) {
  n <- as.integer(dataset_row$n)
  set.seed(seed + k + nchar(dataset_row$dataset))
  rows <- sort(sample(seq_len(n), min(quality_n, n)))
  if (identical(quality_mode, "assume")) {
    return(list(
      status = "exact_assumed",
      rows = rows,
      indices = NULL,
      backend = reference_backend,
      query_n = length(rows),
      error = NA_character_
    ))
  }
  cfg <- list(
    dataset = dataset_row$dataset,
    path = dataset_row$path,
    n = n,
    p = as.integer(dataset_row$p),
    reference_backend = reference_backend,
    k = as.integer(k),
    quality_rows = rows,
    n_threads = as.integer(n_threads),
    output = output,
    candidate = data.frame(
      device = reference_backend,
      provider = "reference",
      public_backend = reference_backend,
      public_method = "exact",
      candidate_id = "reference",
      candidate_kind = "reference",
      n_threads = as.integer(n_threads),
      output = output,
      faiss_query_batch_size = NA_integer_,
      faiss_gpu_query_batch_size = NA_integer_,
      faiss_gpu_reuse_resources = NA,
      cache_fitted_indexes = NA,
      stringsAsFactors = FALSE
    ),
    reference_query_n = length(rows)
  )
  ref <- run_rscript_task("reference", cfg, timeout, bench_script)
  ref$query_n <- length(rows)
  ref
}

row_key <- function(dataset, device, provider, k, candidate_id) {
  paste(dataset, device, provider, as.integer(k), candidate_id, sep = "\r")
}

completed_keys <- function(path) {
  if (!file.exists(path)) return(character())
  x <- tryCatch(read.csv(path, stringsAsFactors = FALSE), error = function(e) NULL)
  required <- c("dataset", "device", "provider", "k", "candidate_id")
  if (is.null(x) || !nrow(x) || !all(required %in% names(x))) {
    return(character())
  }
  row_key(x$dataset, x$device, x$provider, x$k, x$candidate_id)
}

write_missing_rows <- function(dataset_row, k, candidates, reason, results_path) {
  for (i in seq_len(nrow(candidates))) {
    cfg <- list(
      dataset = dataset_row$dataset,
      n = as.integer(dataset_row$n %||% NA_integer_),
      p = as.integer(dataset_row$p %||% NA_integer_),
      k = as.integer(k),
      candidate = candidates[i, , drop = FALSE],
      reference_status = "missing_dataset",
      reference_backend = NA_character_,
      reference_query_n = NA_integer_
    )
    append_csv(base_row(cfg, status = "missing_dataset", error = reason), results_path)
  }
}

empty_recommendations <- function(template) {
  cbind(
    target_recall_threshold = numeric(0),
    recommendation_basis = character(0),
    template[FALSE, , drop = FALSE]
  )
}

select_best_rows <- function(success, target_recalls, group_cols, template) {
  rows <- list()
  idx <- 0L
  if (!nrow(success)) return(empty_recommendations(template))
  groups <- unique(success[group_cols])
  for (g in seq_len(nrow(groups))) {
    keep <- rep(TRUE, nrow(success))
    for (col in group_cols) keep <- keep & success[[col]] == groups[[col]][[g]]
    part0 <- success[keep, , drop = FALSE]
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
      rows[[idx]] <- cbind(
        target_recall_threshold = target,
        recommendation_basis = basis,
        best
      )
    }
  }
  if (length(rows)) do.call(rbind, rows) else empty_recommendations(template)
}

summarize_results <- function(out_dir, target_recalls) {
  results_path <- file.path(out_dir, "bruteforce_tuning_results.csv")
  if (!file.exists(results_path)) return(invisible(NULL))
  x <- read.csv(results_path, stringsAsFactors = FALSE)
  result_exact <- if ("result_exact" %in% names(x)) {
    as.logical(x$result_exact)
  } else {
    rep(FALSE, nrow(x))
  }
  result_exact[is.na(result_exact)] <- FALSE
  success <- x[
    x$status == "success" &
      result_exact &
      is.finite(x$recall_at_k) &
      is.finite(x$elapsed_sec), ,
    drop = FALSE
  ]
  recommendations <- select_best_rows(success, target_recalls, c("dataset", "device", "k"), x)
  provider_recommendations <- select_best_rows(success, target_recalls, c("dataset", "device", "provider", "k"), x)
  write.csv(recommendations, file.path(out_dir, "bruteforce_tuning_recommendations.csv"), row.names = FALSE)
  write.csv(provider_recommendations, file.path(out_dir, "bruteforce_tuning_provider_recommendations.csv"), row.names = FALSE)

  empty_shape_candidates <- data.frame(
    shape_group = character(0),
    device = character(0),
    provider = character(0),
    k = integer(0),
    target_recall_threshold = numeric(0),
    candidate_id = character(0),
    candidate_kind = character(0),
    n_success = integer(0),
    meet_rate = numeric(0),
    median_elapsed_sec = numeric(0),
    median_recall_at_k = numeric(0),
    min_recall_at_k = numeric(0),
    stringsAsFactors = FALSE
  )
  shape_candidates <- empty_shape_candidates
  shape_best <- empty_shape_candidates
  if (nrow(success)) {
    shape_rows <- list()
    idx <- 0L
    for (target in target_recalls) {
      success$meets_target <- success$recall_at_k >= target
      keys <- c("shape_group", "device", "provider", "k", "candidate_id")
      split_key <- interaction(success[keys], drop = TRUE, lex.order = TRUE)
      for (part in split(success, split_key)) {
        idx <- idx + 1L
        shape_rows[[idx]] <- data.frame(
          shape_group = part$shape_group[[1L]],
          device = part$device[[1L]],
          provider = part$provider[[1L]],
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
      shape_candidates$device,
      shape_candidates$provider,
      shape_candidates$k,
      shape_candidates$target_recall_threshold,
      -shape_candidates$meet_rate,
      shape_candidates$median_elapsed_sec,
      -shape_candidates$min_recall_at_k
    ), , drop = FALSE]
    shape_best <- do.call(rbind, lapply(
      split(shape_candidates, interaction(shape_candidates[c("shape_group", "device", "k", "target_recall_threshold")], drop = TRUE, lex.order = TRUE)),
      function(part) part[1L, , drop = FALSE]
    ))
    row.names(shape_best) <- NULL
  }
  write.csv(shape_candidates, file.path(out_dir, "bruteforce_tuning_shape_candidates.csv"), row.names = FALSE)
  write.csv(shape_best, file.path(out_dir, "bruteforce_tuning_shape_recommendations.csv"), row.names = FALSE)
  write_report(out_dir, x, recommendations, provider_recommendations)
  invisible(NULL)
}

write_report <- function(out_dir, results, recommendations, provider_recommendations) {
  lines <- c(
    "# Brute Force Tuning Report",
    "",
    sprintf("Generated: %s", format(Sys.time(), "%Y-%m-%d %H:%M:%S %Z")),
    "",
    "This benchmark evaluates public `method = \"bruteforce\"` exhaustive",
    "nearest-neighbour search for Euclidean self-KNN. Brute force has no",
    "approximation recall parameter; recall",
    "thresholds 0.90, 0.95, and 0.99 are used as correctness gates. Rows either",
    "measure recall against sampled exact references or mark recall as",
    "`exact_assumed` when the row reports `exact = TRUE` and reference sampling",
    "is skipped or unavailable.",
    "",
    "## Tuned Parameters",
    "",
    "- CPU: worker thread count and output distance storage (`float` or `double`).",
    "- CUDA: public CUDA brute-force route with output distance storage (`float` or `double`).",
    "- CUDA rows record the resolved backend, so cuVS brute force and any fallback route are auditable.",
    "",
    "## Status Counts",
    "",
    paste(capture.output(print(table(results$status, useNA = "ifany"))), collapse = "\n"),
    "",
    "## Output Files",
    "",
    "- `bruteforce_tuning_candidate_grid.csv`",
    "- `bruteforce_tuning_results.csv`",
    "- `bruteforce_tuning_recommendations.csv`",
    "- `bruteforce_tuning_provider_recommendations.csv`",
    "- `bruteforce_tuning_shape_candidates.csv`",
    "- `bruteforce_tuning_shape_recommendations.csv`"
  )
  if (nrow(recommendations)) {
    top <- recommendations[seq_len(min(20L, nrow(recommendations))), c(
      "dataset", "device", "k", "target_recall_threshold",
      "provider", "candidate_id", "elapsed_sec", "recall_at_k",
      "quality_status", "recommendation_basis"
    ), drop = FALSE]
    lines <- c(lines, "", "## First Device Recommendations", "", paste(capture.output(print(top, row.names = FALSE)), collapse = "\n"))
  }
  if (nrow(provider_recommendations)) {
    top <- provider_recommendations[seq_len(min(20L, nrow(provider_recommendations))), c(
      "dataset", "device", "provider", "k", "target_recall_threshold",
      "candidate_id", "elapsed_sec", "recall_at_k", "quality_status",
      "recommendation_basis"
    ), drop = FALSE]
    lines <- c(lines, "", "## First Provider Recommendations", "", paste(capture.output(print(top, row.names = FALSE)), collapse = "\n"))
  }
  writeLines(lines, file.path(out_dir, "bruteforce_tuning_report.md"))
}

main <- function() {
  args <- parse_args()
  if (!is.null(args$task)) return(run_child_task())

  configure_native_libs()
  backend <- match.arg(tolower(args$backend %||% "cpu"), c("cpu", "cuda"))
  manifest_path <- args$manifest %||% stop("`--manifest` is required.", call. = FALSE)
  out_dir <- args$out_dir %||% file.path(getwd(), paste0("faissR_BRUTEFORCE_TUNING_", toupper(backend), "_", format(Sys.time(), "%Y%m%d_%H%M%S")))
  dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
  results_path <- file.path(out_dir, "bruteforce_tuning_results.csv")
  candidate_path <- file.path(out_dir, "bruteforce_tuning_candidate_grid.csv")
  bench_script <- normalizePath(script_path(), mustWork = TRUE)
  datasets <- split_arg(args$datasets, "")
  k_values <- as.integer(split_arg(args$k_values, "10,15,50,100"))
  target_recalls <- as.numeric(split_arg(args$target_recalls, "0.9,0.95,0.99"))
  thread_values <- clean_int_values(args$thread_values, if (backend == "cpu") "1,2,4,8,12" else "12")
  if (!length(thread_values)) thread_values <- if (backend == "cpu") c(1L, 2L, 4L, 8L, 12L) else 12L
  n_threads_reference <- positive_int(args$reference_threads, max(thread_values), "reference_threads")
  timeout <- positive_num(args$timeout, 600, "timeout")
  quality_n <- positive_int(args$quality_n, 256, "quality_n")
  seed <- positive_int(args$seed, 4, "seed")
  output_values <- clean_output_values(args$output_values, "float,double")
  reference_backend <- match.arg(tolower(args$reference_backend %||% backend), c("cpu", "cuda"))
  quality_mode <- match.arg(tolower(args$quality_mode %||% "sample_or_assume"), c("assume", "sample", "sample_or_assume"))
  assume_exact_recall <- quality_mode %in% c("assume", "sample_or_assume")
  resume <- logical_arg(args$resume, TRUE)
  max_working_gb <- positive_num(args$max_working_gb, if (backend == "cuda") 40 else 96, "max_working_gb")

  manifest <- normalize_manifest(read.csv(manifest_path, stringsAsFactors = FALSE))
  if (length(datasets) && nzchar(datasets[[1L]])) {
    manifest <- manifest[manifest$dataset %in% datasets, , drop = FALSE]
  }
  candidate_rows <- do.call(rbind, lapply(k_values, function(k) {
    candidate_grid(
      backend,
      k,
      thread_values,
      output_values
    )
  }))
  write.csv(candidate_rows, candidate_path, row.names = FALSE)
  write.csv(
    data.frame(
      key = c("backend", "manifest", "datasets", "k_values", "target_recalls",
              "thread_values", "reference_threads", "timeout", "quality_n",
              "quality_mode", "output_values", "reference_backend",
              "max_working_gb"),
      value = c(backend, manifest_path, paste(datasets, collapse = ","),
                paste(k_values, collapse = ","), paste(target_recalls, collapse = ","),
                paste(thread_values, collapse = ","), n_threads_reference,
                timeout, quality_n, quality_mode, paste(output_values, collapse = ","),
                reference_backend, max_working_gb)
    ),
    file.path(out_dir, "bruteforce_tuning_config.csv"),
    row.names = FALSE
  )

  keys_done <- if (resume) completed_keys(results_path) else character()
  for (i in seq_len(nrow(manifest))) {
    ds <- manifest[i, , drop = FALSE]
    dataset <- ds$dataset[[1L]]
    if (!manifest_success(ds$status[[1L]]) || !file.exists(ds$path[[1L]])) {
      for (k in k_values) {
        candidates <- candidate_grid(backend, k, thread_values, output_values)
        write_missing_rows(ds, k, candidates, ds$error[[1L]] %||% "missing dataset", results_path)
      }
      next
    }
    for (k in k_values) {
      candidates <- candidate_grid(backend, k, thread_values, output_values)
      ref <- reference_for_dataset(
        ds, k, n_threads_reference, output_values[[1L]], quality_n, seed,
        timeout, bench_script, reference_backend, quality_mode
      )
      reference_indices <- if (identical(ref$status, "success")) ref$indices else NULL
      reference_status <- ref$status %||% "not_run"
      for (j in seq_len(nrow(candidates))) {
        candidate <- candidates[j, , drop = FALSE]
        key <- row_key(dataset, candidate$device, candidate$provider, k, candidate$candidate_id)
        if (key %in% keys_done) {
          message(sprintf("[%s] skip completed dataset=%s device=%s k=%s candidate=%s",
                          Sys.time(), dataset, backend, k, candidate$candidate_id))
          next
        }
        estimated_gb <- estimate_working_gb(as.integer(ds$n), as.integer(ds$p), k, candidate)
        cfg <- list(
          dataset = dataset,
          path = ds$path[[1L]],
          n = as.integer(ds$n),
          p = as.integer(ds$p),
          k = as.integer(k),
          quality_rows = ref$rows,
          reference_indices = reference_indices,
          reference_status = reference_status,
          reference_backend = ref$backend %||% reference_backend,
          reference_query_n = ref$query_n %||% length(ref$rows),
          quality_status = if (identical(reference_status, "success")) "sample_exact_reference" else if (isTRUE(assume_exact_recall)) "exact_assumed" else "not_available",
          assume_exact_recall = assume_exact_recall,
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
        message(sprintf("[%s] run dataset=%s device=%s provider=%s k=%s candidate=%s",
                        Sys.time(), dataset, backend, candidate$provider, k, candidate$candidate_id))
        append_csv(run_rscript_task("method", cfg, timeout, bench_script), results_path)
        keys_done <- c(keys_done, key)
      }
      summarize_results(out_dir, target_recalls)
    }
  }
  summarize_results(out_dir, target_recalls)
  message("DONE: ", out_dir)
}

main()
