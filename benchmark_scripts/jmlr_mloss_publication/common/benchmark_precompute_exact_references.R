#!/usr/bin/env Rscript

`%||%` <- function(x, y) {
  if (is.null(x) || length(x) == 0L) return(y)
  first <- tryCatch(x[[1L]], error = function(e) NULL)
  if (is.null(first) || length(first) == 0L || isTRUE(is.na(first))) y else x
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

normalize_metric_arg <- function(value) {
  key <- tolower(trimws(as.character(value %||% "euclidean")[[1L]]))
  valid <- c("euclidean", "cosine", "correlation", "inner_product")
  if (!key %in% valid) {
    stop(
      "`metrics` must contain only euclidean, cosine, correlation, or inner_product.",
      call. = FALSE
    )
  }
  key
}

metric_values_arg <- function(value) {
  unique(vapply(split_arg(value, "euclidean"), normalize_metric_arg, character(1L)))
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

script_path <- function() {
  args <- commandArgs(FALSE)
  file_arg <- grep("^--file=", args, value = TRUE)
  if (length(file_arg)) return(normalizePath(sub("^--file=", "", file_arg[[1L]]), mustWork = TRUE))
  normalizePath(file.path("benchmark_scripts", "benchmark_precompute_exact_references.R"), mustWork = FALSE)
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

load_faissR <- function() {
  source_dir <- Sys.getenv("FAISSR_SOURCE_DIR", unset = "")
  if (nzchar(source_dir) && dir.exists(source_dir)) {
    if (!requireNamespace("pkgload", quietly = TRUE)) {
      stop("FAISSR_SOURCE_DIR was set, but pkgload is not installed.", call. = FALSE)
    }
    pkgload::load_all(source_dir, quiet = TRUE)
  } else {
    library(faissR)
  }
  invisible(TRUE)
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

load_dataset_matrix <- function(path) {
  env <- new.env(parent = emptyenv())
  load(path, envir = env)
  if (exists("dataset", envir = env, inherits = FALSE)) {
    dataset <- get("dataset", envir = env, inherits = FALSE)
    if (is.list(dataset) && !is.null(dataset$data)) return(dataset$data)
  }
  for (name in ls(env)) {
    value <- get(name, envir = env, inherits = FALSE)
    if (is.list(value) && !is.null(value$data)) return(value$data)
  }
  for (name in ls(env)) {
    value <- get(name, envir = env, inherits = FALSE)
    if (inherits(value, "float32") || is.matrix(value)) return(value)
  }
  stop("Dataset file must contain a matrix or list with `$data`: ", path, call. = FALSE)
}

matrix_dims <- function(x, label = "dataset") {
  dims <- tryCatch(dim(x), error = function(e) NULL)
  if (!is.null(dims) && length(dims) >= 2L &&
      all(is.finite(suppressWarnings(as.numeric(dims[1:2]))))) {
    return(as.integer(dims[1:2]))
  }
  nr <- tryCatch(NROW(x), error = function(e) NA_integer_)
  nc <- tryCatch(NCOL(x), error = function(e) NA_integer_)
  nr <- suppressWarnings(as.integer(nr))
  nc <- suppressWarnings(as.integer(nc))
  if (length(nr) == 1L && length(nc) == 1L &&
      is.finite(nr) && is.finite(nc) && nr > 0L && nc > 0L) {
    return(c(nr, nc))
  }
  stop("Could not determine matrix dimensions for ", label, ".", call. = FALSE)
}

dataset_path_column <- function(manifest) {
  candidates <- intersect(c("path", "output", "file", "file_path", "rdata_path"), names(manifest))
  if (!length(candidates)) stop("Manifest must contain a dataset file path column.", call. = FALSE)
  candidates[[1L]]
}

reference_file <- function(dataset_path, k, quality_n, seed, metric = "euclidean") {
  metric <- normalize_metric_arg(metric)
  file.path(
    dirname(dataset_path),
    sprintf(
      "faissR_exact_reference_%s_k%d_q%d_seed%d.RData",
      metric, as.integer(k), as.integer(quality_n), as.integer(seed)
    )
  )
}

reference_is_valid <- function(path, k) {
  if (!file.exists(path)) return(FALSE)
  env <- new.env(parent = emptyenv())
  ok <- tryCatch({
    load(path, envir = env)
    if (!exists("faissR_reference", envir = env, inherits = FALSE)) return(FALSE)
    ref <- get("faissR_reference", envir = env, inherits = FALSE)
    is.list(ref) &&
      identical(ref$status %||% "success", "success") &&
      is.matrix(ref$indices) &&
      ncol(ref$indices) >= as.integer(k)
  }, error = function(e) FALSE)
  isTRUE(ok)
}

remove_self <- function(indices, distances, rows, k) {
  out_indices <- matrix(NA_integer_, nrow = length(rows), ncol = k)
  out_distances <- matrix(NA_real_, nrow = length(rows), ncol = k)
  for (i in seq_along(rows)) {
    keep <- which(!is.na(indices[i, ]) & indices[i, ] != rows[[i]])
    keep <- keep[seq_len(min(length(keep), k))]
    if (length(keep)) {
      out_indices[i, seq_along(keep)] <- indices[i, keep]
      out_distances[i, seq_along(keep)] <- distances[i, keep]
    }
  }
  list(indices = out_indices, distances = out_distances)
}

classify_error <- function(message, timed_out = FALSE) {
  if (isTRUE(timed_out)) return("timeout")
  msg <- tolower(as.character(message %||% ""))
  if (grepl("not available|unavailable|not built|requires|without cuda|without faiss|without cuvs", msg)) return("unavailable")
  if (grepl("not support|does not support|only supports|only available", msg)) return("unsupported")
  "failed"
}

reference_methods_arg <- function(value) {
  methods <- tolower(split_arg(value, "flat,exact"))
  allowed <- c("flat", "exact", "bruteforce")
  bad <- setdiff(methods, allowed)
  if (length(bad)) {
    stop(
      "`--reference_methods` must contain only exact CPU routes: ",
      paste(allowed, collapse = ", "),
      call. = FALSE
    )
  }
  unique(methods)
}

compute_reference_nn <- function(x, rows, k, threads, methods, metric) {
  errors <- character()
  for (method in methods) {
    result <- tryCatch(
      faissR::nn(
        x,
        points = x[rows, , drop = FALSE],
        k = k,
        exclude_self = FALSE,
        backend = "cpu",
        method = method,
        metric = metric,
        output = "double",
        n_threads = threads
      ),
      error = function(e) e
    )
    if (!inherits(result, "error")) {
      return(list(result = result, method = method, errors = errors))
    }
    errors <- c(errors, sprintf("%s: %s", method, conditionMessage(result)))
  }
  stop("All exact reference methods failed: ", paste(errors, collapse = " | "), call. = FALSE)
}

compute_reference <- function(config) {
  configure_threads(config$threads)
  load_faissR()
  x <- load_dataset_matrix(config$dataset_path)
  dims <- matrix_dims(x, config$dataset)
  n <- dims[[1L]]
  p <- dims[[2L]]
  reference_k <- as.integer(config$k)
  if (!is.finite(reference_k) || reference_k < 1L) {
    stop("Reference `k` must be a positive integer.", call. = FALSE)
  }
  if (n < 2L) {
    stop("Reference calculation requires at least two rows.", call. = FALSE)
  }
  set.seed(config$seed + nchar(config$dataset) + reference_k)
  rows <- if (n <= config$quality_n) seq_len(n) else sort(sample.int(n, config$quality_n))
  started <- proc.time()[["elapsed"]]
  methods <- config$reference_methods %||% c("flat", "exact")
  ref_run <- compute_reference_nn(
    x,
    rows = rows,
    k = min(reference_k + 1L, n),
    threads = config$threads,
    methods = methods,
    metric = config$metric
  )
  ref <- ref_run$result
  reference_knn <- remove_self(ref$indices, ref$distances, rows, reference_k)
  elapsed <- proc.time()[["elapsed"]] - started
  faissR_reference <- list(
    status = "success",
    dataset = config$dataset,
    dataset_path = config$dataset_path,
    n = as.integer(n),
    p = as.integer(p),
    metric = config$metric,
    backend = "cpu",
    method = ref_run$method,
    exact = TRUE,
    backend_used = ref$backend_used %||% attr(ref, "backend") %||% NA_character_,
    resolved_backend = attr(ref, "resolved_backend") %||% NA_character_,
    implementation_backend = attr(ref, "implementation_backend") %||% NA_character_,
    attempted_methods = methods,
    failed_attempts = ref_run$errors,
    k = reference_k,
    max_k = reference_k,
    quality_n = as.integer(config$quality_n),
    seed = as.integer(config$seed),
    rows = rows,
    indices = reference_knn$indices,
    distances = reference_knn$distances,
    elapsed_sec = as.numeric(elapsed),
    peak_rss_gb = read_peak_rss_gb(),
    created_at = format(Sys.time(), "%Y-%m-%d %H:%M:%S %Z")
  )
  save(faissR_reference, file = config$output_path, compress = "xz")
  data.frame(
    dataset = config$dataset,
    metric = config$metric,
    k = reference_k,
    status = "success",
    reference_path = config$output_path,
    reference_method = ref_run$method,
    reference_backend_used = faissR_reference$backend_used,
    query_n = length(rows),
    elapsed_sec = as.numeric(elapsed),
    peak_rss_gb = faissR_reference$peak_rss_gb,
    error = NA_character_,
    stringsAsFactors = FALSE
  )
}

run_child <- function() {
  args <- parse_args()
  config <- readRDS(args$config)
  row <- tryCatch(
    compute_reference(config),
    error = function(e) {
      data.frame(
        dataset = config$dataset,
        metric = config$metric,
        k = as.integer(config$k),
        status = classify_error(conditionMessage(e)),
        reference_path = config$output_path,
        reference_method = NA_character_,
        reference_backend_used = NA_character_,
        query_n = NA_integer_,
        elapsed_sec = NA_real_,
        peak_rss_gb = NA_real_,
        error = conditionMessage(e),
        stringsAsFactors = FALSE
      )
    }
  )
  saveRDS(row, args$result)
}

run_task <- function(config, timeout, bench_script) {
  cfg <- tempfile("faissR_ref_cfg_", fileext = ".rds")
  out <- tempfile("faissR_ref_out_", fileext = ".rds")
  saveRDS(config, cfg)
  on.exit(unlink(c(cfg, out)), add = TRUE)
  cmd <- c(Sys.getenv("R_BIN", "Rscript"), "--vanilla", bench_script,
           "--child=TRUE", paste0("--config=", cfg), paste0("--result=", out))
  timeout_bin <- Sys.which("timeout")
  if (nzchar(timeout_bin)) cmd <- c(timeout_bin, as.character(as.integer(ceiling(timeout))), cmd)
  status <- system(paste(shQuote(cmd), collapse = " "), intern = TRUE)
  exit_status <- attr(status, "status") %||% 0L
  if (file.exists(out) && !identical(as.integer(exit_status), 124L)) return(readRDS(out))
  data.frame(
    dataset = config$dataset,
    metric = config$metric,
    k = as.integer(config$k),
    status = if (identical(as.integer(exit_status), 124L)) "timeout" else classify_error(paste(status, collapse = "\n")),
    reference_path = config$output_path,
    reference_method = NA_character_,
    reference_backend_used = NA_character_,
    query_n = NA_integer_,
    elapsed_sec = if (identical(as.integer(exit_status), 124L)) timeout else NA_real_,
    peak_rss_gb = NA_real_,
    error = if (identical(as.integer(exit_status), 124L)) sprintf("timed out after %s seconds", timeout) else paste(status, collapse = "\n"),
    stringsAsFactors = FALSE
  )
}

main <- function() {
  args <- parse_args()
  if (!is.null(args$child)) return(run_child())

  manifest_path <- args$manifest %||% stop("`--manifest` is required.", call. = FALSE)
  out_dir <- args$out_dir %||% dirname(manifest_path)
  dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
  results_path <- file.path(out_dir, "faissR_exact_reference_precompute_results.csv")
  manifest <- read.csv(manifest_path, stringsAsFactors = FALSE)
  path_col <- dataset_path_column(manifest)
  datasets <- split_arg(args$datasets, paste(manifest$dataset, collapse = ","))
  manifest <- manifest[manifest$dataset %in% datasets, , drop = FALSE]
  k_values <- unique(as.integer(split_arg(args$k_values, "15,30,50,100")))
  k_values <- k_values[is.finite(k_values) & k_values > 0L]
  if (!length(k_values)) stop("`--k_values` must contain at least one positive integer.", call. = FALSE)
  reference_k_default <- max(c(100L, k_values))
  reference_k <- positive_int(args$reference_k, reference_k_default, "reference_k")
  if (any(k_values > reference_k)) {
    stop("`--reference_k` must be at least as large as every value in `--k_values`.", call. = FALSE)
  }
  threads <- positive_int(args$threads, 12L, "threads")
  timeout <- positive_int(args$timeout, 1800L, "timeout")
  quality_n <- positive_int(args$quality_n, 256L, "quality_n")
  seed <- positive_int(args$seed, 4L, "seed")
  reference_methods <- reference_methods_arg(args$reference_methods)
  metrics <- metric_values_arg(args$metrics %||% args$metric %||% "euclidean")
  resume <- logical_arg(args$resume, TRUE)
  bench_script <- normalizePath(script_path(), mustWork = TRUE)

  for (i in seq_len(nrow(manifest))) {
    ds <- manifest[i, , drop = FALSE]
    dataset_path <- ds[[path_col]][[1L]]
    for (metric in metrics) {
      ref_path <- reference_file(dataset_path, reference_k, quality_n, seed, metric = metric)
      if (isTRUE(resume) && reference_is_valid(ref_path, reference_k)) {
        append_csv(data.frame(
          dataset = ds$dataset[[1L]], metric = metric, k = as.integer(reference_k),
          status = "already_exists",
          reference_path = ref_path,
          reference_method = NA_character_, reference_backend_used = NA_character_,
          query_n = NA_integer_, elapsed_sec = NA_real_,
          peak_rss_gb = NA_real_, error = NA_character_, stringsAsFactors = FALSE
        ), results_path)
        next
      }
      message(sprintf(
        "[%s] reference dataset=%s metric=%s reference_k=%s requested_k_values=%s",
        Sys.time(), ds$dataset[[1L]], metric, reference_k, paste(k_values, collapse = ",")
      ))
      row <- run_task(list(
        dataset = ds$dataset[[1L]],
        dataset_path = dataset_path,
        metric = metric,
        k = as.integer(reference_k),
        requested_k_values = k_values,
        quality_n = quality_n,
        seed = seed,
        threads = threads,
        reference_methods = reference_methods,
        output_path = ref_path
      ), timeout, bench_script)
      append_csv(row, results_path)
    }
  }
  message("DONE: ", results_path)
}

main()
