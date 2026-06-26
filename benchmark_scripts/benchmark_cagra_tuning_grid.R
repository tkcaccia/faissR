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
  normalizePath(file.path("benchmark_scripts", "benchmark_cagra_tuning_grid.R"), mustWork = FALSE)
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

normalize_provider <- function(provider) {
  key <- tolower(gsub("[[:space:]_-]+", "", as.character(provider)))
  aliases <- c(
    cuvs = "cuvs",
    rapids = "cuvs",
    directcuvs = "cuvs",
    faiss = "faiss_gpu",
    faissgpu = "faiss_gpu",
    faissgpucagra = "faiss_gpu"
  )
  out <- unname(aliases[key])
  out[is.na(out)] <- NA_character_
  out
}

cagra_backend_for_provider <- function(provider) {
  if (identical(provider, "cuvs")) return("cuda_cuvs_cagra")
  if (identical(provider, "faiss_gpu")) return("faiss_gpu_cagra")
  stop("Unsupported CAGRA provider: ", provider, call. = FALSE)
}

provider_argument <- function(provider) {
  if (identical(provider, "cuvs")) return("cuvs")
  if (identical(provider, "faiss_gpu")) return("faiss_gpu")
  NULL
}

cagra_candidates_for_provider <- function(k,
                                          provider,
                                          cuvs_build_algos,
                                          cuvs_nndescent_niters,
                                          cagra_batch_sizes,
                                          grid_level = "standard") {
  auto <- data.frame(
    candidate_id = paste0(provider, "_auto"),
    candidate_kind = "auto",
    provider = provider,
    backend = cagra_backend_for_provider(provider),
    graph_degree = NA_integer_,
    intermediate_graph_degree = NA_integer_,
    search_width = NA_integer_,
    itopk_size = NA_integer_,
    cagra_build_algo = NA_character_,
    cagra_nndescent_niter = NA_integer_,
    cagra_batch_size = NA_integer_,
    stringsAsFactors = FALSE
  )
  specs <- data.frame(
    label = c("speed", "speed_plus", "balanced", "balanced_plus", "recall", "recall_plus"),
    graph_min = c(32L, 48L, 64L, 80L, 96L, 128L),
    graph_mult = c(1.0, 1.1, 1.25, 1.5, 2.0, 2.5),
    intermediate_mult = c(2L, 2L, 2L, 3L, 3L, 4L),
    intermediate_min = c(64L, 96L, 128L, 192L, 256L, 384L),
    search_width = c(0L, 1L, 2L, 4L, 8L, 16L),
    itopk_min = c(64L, 96L, 128L, 192L, 256L, 512L),
    itopk_mult = c(1.0, 1.25, 1.5, 2.0, 3.0, 4.0),
    stringsAsFactors = FALSE
  )
  if (identical(grid_level, "compact")) specs <- specs[c(1L, 3L, 5L), , drop = FALSE]
  if (identical(grid_level, "wide")) {
    specs <- rbind(
      specs,
      data.frame(
        label = c("recall_wide", "recall_max"),
        graph_min = c(192L, 256L),
        graph_mult = c(3.0, 4.0),
        intermediate_mult = c(4L, 4L),
        intermediate_min = c(512L, 768L),
        search_width = c(32L, 64L),
        itopk_min = c(768L, 1024L),
        itopk_mult = c(5.0, 6.0),
        stringsAsFactors = FALSE
      )
    )
  }
  manual <- do.call(rbind, lapply(seq_len(nrow(specs)), function(i) {
    graph_degree <- as.integer(max(k + 1L, ceiling(specs$graph_mult[[i]] * k), specs$graph_min[[i]]))
    intermediate <- as.integer(max(graph_degree, graph_degree * specs$intermediate_mult[[i]], specs$intermediate_min[[i]]))
    itopk <- as.integer(max(k, graph_degree, ceiling(specs$itopk_mult[[i]] * k), specs$itopk_min[[i]]))
    data.frame(
      candidate_id = sprintf(
        "%s_%s_gd%d_igd%d_sw%d_top%d",
        provider, specs$label[[i]], graph_degree, intermediate,
        specs$search_width[[i]], itopk
      ),
      candidate_kind = "manual",
      provider = provider,
      backend = cagra_backend_for_provider(provider),
      graph_degree = graph_degree,
      intermediate_graph_degree = intermediate,
      search_width = as.integer(specs$search_width[[i]]),
      itopk_size = itopk,
      cagra_build_algo = NA_character_,
      cagra_nndescent_niter = NA_integer_,
      cagra_batch_size = NA_integer_,
      stringsAsFactors = FALSE
    )
  }))
  if (identical(provider, "cuvs")) {
    build_algos <- split_arg(cuvs_build_algos, "auto,ivf_pq,nn_descent,iterative_cagra_search")
    nndescent_niters <- as.integer(split_arg(cuvs_nndescent_niters, "20"))
    nndescent_niters <- nndescent_niters[is.finite(nndescent_niters) & nndescent_niters > 0L]
    if (!length(nndescent_niters)) nndescent_niters <- 20L
    batch_sizes <- as.integer(split_arg(cagra_batch_sizes, "8192"))
    batch_sizes <- batch_sizes[is.finite(batch_sizes) & batch_sizes > 0L]
    if (!length(batch_sizes)) batch_sizes <- 8192L
    manual <- do.call(rbind, lapply(build_algos, function(algo) {
      algo_key <- gsub("[^A-Za-z0-9]+", "", algo)
      niters <- if (tolower(gsub("[[:space:]_-]+", "", algo)) %in% c("nndescent")) nndescent_niters else NA_integer_
      do.call(rbind, lapply(batch_sizes, function(batch_size) {
        do.call(rbind, lapply(niters, function(niter) {
          x <- manual
          x$cagra_build_algo <- algo
          x$cagra_nndescent_niter <- as.integer(niter)
          x$cagra_batch_size <- as.integer(batch_size)
          niter_label <- if (is.na(niter)) "" else paste0("_niter", niter)
          x$candidate_id <- paste0(x$candidate_id, "_", algo_key, niter_label, "_bs", batch_size)
          x
        }))
      }))
    }))
  }
  unique(rbind(auto, manual))
}

candidate_grid <- function(k,
                           providers,
                           cuvs_build_algos,
                           cuvs_nndescent_niters,
                           cagra_batch_sizes,
                           grid_level) {
  rows <- lapply(providers, function(provider) {
    cagra_candidates_for_provider(
      k,
      provider,
      cuvs_build_algos,
      cuvs_nndescent_niters,
      cagra_batch_sizes,
      grid_level
    )
  })
  do.call(rbind, rows)
}

apply_candidate_options <- function(candidate) {
  options(
    faissR.cuvs_cagra_tune = FALSE,
    faissR.cuvs_graph_degree = NULL,
    faissR.cuvs_intermediate_graph_degree = NULL,
    faissR.cuvs_search_width = NULL,
    faissR.cuvs_itopk_size = NULL,
    faissR.cuvs_cagra_build_algo = NULL
  )
  if (!identical(candidate$candidate_kind, "auto")) {
    options(
      faissR.cuvs_graph_degree = as.integer(candidate$graph_degree),
      faissR.cuvs_intermediate_graph_degree = as.integer(candidate$intermediate_graph_degree),
      faissR.cuvs_search_width = as.integer(candidate$search_width),
      faissR.cuvs_itopk_size = as.integer(candidate$itopk_size)
    )
    if (identical(candidate$provider, "cuvs") && !is.na(candidate$cagra_build_algo)) {
      options(faissR.cuvs_cagra_build_algo = as.character(candidate$cagra_build_algo))
    }
  }
  invisible(TRUE)
}

estimate_working_gb <- function(n, p, candidate) {
  n <- as.numeric(n)
  p <- as.numeric(p)
  if (identical(candidate$candidate_kind, "auto")) return(NA_real_)
  graph_degree <- as.numeric(candidate$graph_degree)
  intermediate <- as.numeric(candidate$intermediate_graph_degree)
  data_gb <- n * p * 4 / 1024^3
  graph_gb <- n * graph_degree * 8 / 1024^3
  intermediate_gb <- n * intermediate * 4 / 1024^3
  data_gb + graph_gb + intermediate_gb
}

base_row <- function(config, status = "success", error = NA_character_) {
  candidate <- config$candidate
  data.frame(
    dataset = config$dataset,
    n = as.integer(config$n),
    p = as.integer(config$p),
    shape_group = shape_group(config$n, config$p),
    backend = candidate$backend,
    method = "cagra",
    metric = "euclidean",
    k = as.integer(config$k),
    provider = candidate$provider,
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
    requested_graph_degree = as.integer(candidate$graph_degree),
    requested_intermediate_graph_degree = as.integer(candidate$intermediate_graph_degree),
    requested_search_width = as.integer(candidate$search_width),
    requested_itopk_size = as.integer(candidate$itopk_size),
    requested_cagra_build_algo = candidate$cagra_build_algo %||% NA_character_,
    requested_cagra_nndescent_niter = as.integer(candidate$cagra_nndescent_niter),
    requested_cagra_batch_size = as.integer(candidate$cagra_batch_size),
    result_backend = NA_character_,
    implementation_backend = NA_character_,
    resolved_backend = NA_character_,
    distance_type = NA_character_,
    input_type = NA_character_,
    input_layout = NA_character_,
    cagra_provider = NA_character_,
    cagra_provider_option = NA_character_,
    tuning_policy = NA_character_,
    tuning_rule = NA_character_,
    result_graph_degree = NA_integer_,
    result_intermediate_graph_degree = NA_integer_,
    result_search_width = NA_integer_,
    result_itopk_size = NA_integer_,
    result_cagra_build_algo = NA_character_,
    result_nn_descent_niter = NA_integer_,
    result_search_batch_size = NA_integer_,
    cagra_parameters_adjusted = NA,
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
  configure_threads(config$n_threads)
  load_faissR()
  x <- load_float_dataset(config$path)
  candidate <- config$candidate
  apply_candidate_options(candidate)
  if (!is.na(candidate$cagra_nndescent_niter)) {
    Sys.setenv(FAISSR_CUVS_CAGRA_NN_DESCENT_NITER = as.integer(candidate$cagra_nndescent_niter))
  }
  if (!is.na(candidate$cagra_batch_size)) {
    Sys.setenv(FAISSR_CUVS_CAGRA_BATCH_SIZE = as.integer(candidate$cagra_batch_size))
  }
  cagra_build_algo <- if (identical(candidate$provider, "cuvs") && !is.na(candidate$cagra_build_algo)) {
    as.character(candidate$cagra_build_algo)
  } else {
    NULL
  }
  started <- proc.time()[["elapsed"]]
  res <- faissR::nn(
    x,
    k = as.integer(config$k),
    exclude_self = TRUE,
    backend = as.character(candidate$backend),
    method = "cagra",
    metric = "euclidean",
    output = config$output,
    n_threads = config$n_threads,
    tuning = "fixed",
    cagra_implementation = provider_argument(candidate$provider),
    cagra_build_algo = cagra_build_algo
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
  row$result_backend <- res$backend_used %||% attr(res, "backend") %||% NA_character_
  row$implementation_backend <- attr(res, "implementation_backend") %||% NA_character_
  row$resolved_backend <- attr(res, "resolved_backend") %||% row$result_backend
  row$distance_type <- res$distance_type %||% attr(res, "distance_type") %||% "double"
  row$input_type <- res$input_type %||% attr(res, "input_type") %||% NA_character_
  row$input_layout <- res$input_layout %||% attr(res, "input_layout") %||% NA_character_
  row$cagra_provider <- approx$cagra_provider %||% candidate$provider
  row$cagra_provider_option <- approx$cagra_provider_option %||% provider_argument(candidate$provider)
  row$tuning_policy <- approx$tuning_policy %||% NA_character_
  row$tuning_rule <- approx$tuning_rule %||% NA_character_
  row$result_graph_degree <- as.integer(approx$graph_degree %||% NA_integer_)
  row$result_intermediate_graph_degree <- as.integer(approx$intermediate_graph_degree %||% NA_integer_)
  row$result_search_width <- as.integer(approx$search_width %||% NA_integer_)
  row$result_itopk_size <- as.integer(approx$itopk_size %||% NA_integer_)
  row$result_cagra_build_algo <- approx$cagra_build_algo %||% NA_character_
  row$result_nn_descent_niter <- as.integer(approx$nn_descent_niter %||% NA_integer_)
  row$result_search_batch_size <- as.integer(approx$search_batch_size %||% NA_integer_)
  row$cagra_parameters_adjusted <- isTRUE(approx$cagra_parameters_adjusted)
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
  config_path <- tempfile("faissR_cagra_config_", fileext = ".rds")
  result_path <- tempfile("faissR_cagra_result_", fileext = ".rds")
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
                                  reference_backend) {
  n <- as.integer(dataset_row$n)
  set.seed(seed + k + nchar(dataset_row$dataset))
  rows <- sort(sample(seq_len(n), min(quality_n, n)))
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
      candidate_id = "reference",
      candidate_kind = "reference",
      provider = "reference",
      backend = reference_backend,
      graph_degree = NA_integer_,
      intermediate_graph_degree = NA_integer_,
      search_width = NA_integer_,
      itopk_size = NA_integer_,
      cagra_build_algo = NA_character_,
      cagra_nndescent_niter = NA_integer_,
      cagra_batch_size = NA_integer_,
      stringsAsFactors = FALSE
    ),
    reference_query_n = length(rows)
  )
  run_rscript_task("reference", cfg, timeout, bench_script)
}

row_key <- function(dataset, backend, provider, k, candidate_id) {
  paste(dataset, backend, provider, as.integer(k), candidate_id, sep = "\r")
}

completed_keys <- function(path) {
  if (!file.exists(path)) return(character())
  x <- tryCatch(read.csv(path, stringsAsFactors = FALSE), error = function(e) NULL)
  required <- c("dataset", "backend", "provider", "k", "candidate_id")
  if (is.null(x) || !nrow(x) || !all(required %in% names(x))) {
    return(character())
  }
  row_key(x$dataset, x$backend, x$provider, x$k, x$candidate_id)
}

write_missing_rows <- function(dataset_row, k, candidates, reason, results_path) {
  for (i in seq_len(nrow(candidates))) {
    cfg <- list(
      dataset = dataset_row$dataset,
      n = as.integer(dataset_row$n %||% NA_integer_),
      p = as.integer(dataset_row$p %||% NA_integer_),
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
  results_path <- file.path(out_dir, "cagra_tuning_results.csv")
  if (!file.exists(results_path)) return(invisible(NULL))
  x <- read.csv(results_path, stringsAsFactors = FALSE)
  success <- x[x$status == "success" & is.finite(x$recall_at_k) & is.finite(x$elapsed_sec), , drop = FALSE]
  rec_rows <- list()
  idx <- 0L
  groups <- unique(success[c("dataset", "provider", "backend", "k")])
  for (g in seq_len(nrow(groups))) {
    part0 <- success[
      success$dataset == groups$dataset[[g]] &
        success$provider == groups$provider[[g]] &
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
  write.csv(recommendations, file.path(out_dir, "cagra_tuning_recommendations.csv"), row.names = FALSE)

  if (nrow(success)) {
    shape_rows <- list()
    idx <- 0L
    for (target in target_recalls) {
      success$meets_target <- success$recall_at_k >= target
      keys <- c("shape_group", "provider", "backend", "k", "candidate_id")
      split_key <- interaction(success[keys], drop = TRUE, lex.order = TRUE)
      for (part in split(success, split_key)) {
        idx <- idx + 1L
        shape_rows[[idx]] <- data.frame(
          shape_group = part$shape_group[[1L]],
          provider = part$provider[[1L]],
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
      shape_candidates$provider,
      shape_candidates$backend,
      shape_candidates$k,
      shape_candidates$target_recall_threshold,
      -shape_candidates$meet_rate,
      shape_candidates$median_elapsed_sec,
      -shape_candidates$min_recall_at_k
    ), , drop = FALSE]
    write.csv(shape_candidates, file.path(out_dir, "cagra_tuning_shape_candidates.csv"), row.names = FALSE)
    shape_best <- do.call(rbind, lapply(
      split(shape_candidates, interaction(shape_candidates[c("shape_group", "provider", "backend", "k", "target_recall_threshold")], drop = TRUE, lex.order = TRUE)),
      function(part) part[1L, , drop = FALSE]
    ))
    row.names(shape_best) <- NULL
    write.csv(shape_best, file.path(out_dir, "cagra_tuning_shape_recommendations.csv"), row.names = FALSE)
  }
  write_report(out_dir, x, recommendations)
  invisible(NULL)
}

write_report <- function(out_dir, results, recommendations) {
  lines <- c(
    "# CAGRA Tuning Report",
    "",
    sprintf("Generated: %s", format(Sys.time(), "%Y-%m-%d %H:%M:%S %Z")),
    "",
    "This benchmark evaluates explicit CUDA CAGRA settings for Euclidean self-KNN.",
    "Provider rows are separated into direct RAPIDS cuVS CAGRA and FAISS GPU",
    "CAGRA backed by cuVS. Recommendation tables select the fastest successful",
    "candidate reaching recall thresholds 0.90, 0.95, and 0.99, or the",
    "highest-recall candidate when no row reaches the threshold.",
    "",
    "## Tuned Parameters",
    "",
    "- `graph_degree`",
    "- `intermediate_graph_degree`",
    "- `search_width`",
    "- `itopk_size`",
    "- `cagra_build_algo` for direct cuVS CAGRA only",
    "- `FAISSR_CUVS_CAGRA_NN_DESCENT_NITER` when `cagra_build_algo = \"nn_descent\"`",
    "- `FAISSR_CUVS_CAGRA_BATCH_SIZE` for direct cuVS CAGRA batching",
    "",
    "## Status Counts",
    "",
    paste(capture.output(print(table(results$status, useNA = "ifany"))), collapse = "\n"),
    "",
    "## Output Files",
    "",
    "- `cagra_tuning_candidate_grid.csv`",
    "- `cagra_tuning_results.csv`",
    "- `cagra_tuning_recommendations.csv`",
    "- `cagra_tuning_shape_candidates.csv`",
    "- `cagra_tuning_shape_recommendations.csv`"
  )
  if (nrow(recommendations)) {
    top <- recommendations[seq_len(min(20L, nrow(recommendations))), c(
      "dataset", "provider", "backend", "k", "target_recall_threshold",
      "candidate_id", "elapsed_sec", "recall_at_k", "recommendation_basis"
    ), drop = FALSE]
    lines <- c(lines, "", "## First Recommendations", "", paste(capture.output(print(top, row.names = FALSE)), collapse = "\n"))
  }
  writeLines(lines, file.path(out_dir, "cagra_tuning_report.md"))
}

main <- function() {
  args <- parse_args()
  if (!is.null(args$task)) return(run_child_task())

  configure_native_libs()
  manifest_path <- args$manifest %||% stop("`--manifest` is required.", call. = FALSE)
  out_dir <- args$out_dir %||% file.path(getwd(), paste0("faissR_CAGRA_TUNING_CUDA_", format(Sys.time(), "%Y%m%d_%H%M%S")))
  dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
  results_path <- file.path(out_dir, "cagra_tuning_results.csv")
  candidate_path <- file.path(out_dir, "cagra_tuning_candidate_grid.csv")
  bench_script <- normalizePath(script_path(), mustWork = TRUE)
  datasets <- split_arg(args$datasets, "")
  k_values <- as.integer(split_arg(args$k_values, "10,15,50,100"))
  target_recalls <- as.numeric(split_arg(args$target_recalls, "0.9,0.95,0.99"))
  n_threads <- positive_int(args$threads, 12, "threads")
  timeout <- positive_num(args$timeout, 2000, "timeout")
  quality_n <- positive_int(args$quality_n, 256, "quality_n")
  seed <- positive_int(args$seed, 4, "seed")
  output <- args$output %||% "float"
  grid_level <- match.arg(args$grid_level %||% "standard", c("compact", "standard", "wide"))
  reference_backend <- match.arg(tolower(args$reference_backend %||% "cuda"), c("cpu", "cuda"))
  resume <- logical_arg(args$resume, TRUE)
  max_working_gb <- positive_num(args$max_working_gb, 40, "max_working_gb")
  providers <- normalize_provider(split_arg(args$providers, "cuvs,faiss_gpu"))
  if (any(is.na(providers)) || !length(providers)) {
    stop("`--providers` must contain `cuvs`, `faiss_gpu`, or both.", call. = FALSE)
  }
  providers <- unique(providers)
  cuvs_build_algos <- paste(split_arg(args$cuvs_build_algos, "auto,ivf_pq,nn_descent,iterative_cagra_search"), collapse = ",")
  cuvs_nndescent_niters <- paste(split_arg(args$cuvs_nndescent_niters, "10,20,40"), collapse = ",")
  cagra_batch_sizes <- paste(split_arg(args$cagra_batch_sizes, "8192"), collapse = ",")

  manifest <- normalize_manifest(read.csv(manifest_path, stringsAsFactors = FALSE))
  if (length(datasets) && nzchar(datasets[[1L]])) {
    manifest <- manifest[manifest$dataset %in% datasets, , drop = FALSE]
  }
  candidate_rows <- do.call(rbind, lapply(k_values, function(k) {
    x <- candidate_grid(k, providers, cuvs_build_algos, cuvs_nndescent_niters, cagra_batch_sizes, grid_level)
    x$k <- as.integer(k)
    x
  }))
  write.csv(candidate_rows, candidate_path, row.names = FALSE)
  write.csv(
    data.frame(
      key = c("manifest", "datasets", "providers", "cuvs_build_algos",
              "cuvs_nndescent_niters", "cagra_batch_sizes",
              "k_values", "target_recalls", "threads", "timeout",
              "quality_n", "output", "grid_level", "reference_backend",
              "max_working_gb"),
      value = c(manifest_path, paste(datasets, collapse = ","),
                paste(providers, collapse = ","), cuvs_build_algos,
                cuvs_nndescent_niters, cagra_batch_sizes,
                paste(k_values, collapse = ","), paste(target_recalls, collapse = ","),
                n_threads, timeout, quality_n, output, grid_level,
                reference_backend, max_working_gb)
    ),
    file.path(out_dir, "cagra_tuning_config.csv"),
    row.names = FALSE
  )

  keys_done <- if (resume) completed_keys(results_path) else character()
  for (i in seq_len(nrow(manifest))) {
    ds <- manifest[i, , drop = FALSE]
    dataset <- ds$dataset[[1L]]
    if (!manifest_success(ds$status[[1L]]) || !file.exists(ds$path[[1L]])) {
      for (k in k_values) {
        write_missing_rows(
          ds,
          k,
          candidate_grid(k, providers, cuvs_build_algos, cuvs_nndescent_niters, cagra_batch_sizes, grid_level),
          ds$error[[1L]] %||% "missing dataset",
          results_path
        )
      }
      next
    }
    for (k in k_values) {
      candidates <- candidate_grid(k, providers, cuvs_build_algos, cuvs_nndescent_niters, cagra_batch_sizes, grid_level)
      ref <- reference_for_dataset(
        ds, k, n_threads, output, quality_n, seed, timeout,
        bench_script, reference_backend
      )
      if (!identical(ref$status, "success")) {
        for (j in seq_len(nrow(candidates))) {
          cfg <- list(
            dataset = dataset,
            n = as.integer(ds$n),
            p = as.integer(ds$p),
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
        key <- row_key(dataset, candidate$backend, candidate$provider, k, candidate$candidate_id)
        if (key %in% keys_done) {
          message(sprintf("[%s] skip completed dataset=%s provider=%s k=%s candidate=%s",
                          Sys.time(), dataset, candidate$provider, k, candidate$candidate_id))
          next
        }
        estimated_gb <- estimate_working_gb(as.integer(ds$n), as.integer(ds$p), candidate)
        cfg <- list(
          dataset = dataset,
          path = ds$path[[1L]],
          n = as.integer(ds$n),
          p = as.integer(ds$p),
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
        message(sprintf("[%s] run dataset=%s provider=%s backend=%s k=%s candidate=%s",
                        Sys.time(), dataset, candidate$provider, candidate$backend, k, candidate$candidate_id))
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
