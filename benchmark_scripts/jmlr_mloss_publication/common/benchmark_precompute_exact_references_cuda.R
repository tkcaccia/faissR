#!/usr/bin/env Rscript

`%||%` <- function(x, y) {
  if (is.null(x) || !length(x)) return(y)
  first <- tryCatch(x[[1L]], error = function(e) NULL)
  if (is.null(first) || !length(first) || isTRUE(is.na(first))) y else x
}

parse_args <- function(args = commandArgs(trailingOnly = TRUE)) {
  out <- list()
  for (arg in args) {
    if (!startsWith(arg, "--")) next
    bits <- strsplit(sub("^--", "", arg), "=", fixed = TRUE)[[1L]]
    out[[bits[[1L]]]] <- if (length(bits) > 1L) paste(bits[-1L], collapse = "=") else "TRUE"
  }
  out
}

split_arg <- function(value, default = "") {
  value <- value %||% default
  if (!nzchar(value)) return(character())
  out <- trimws(strsplit(value, ",", fixed = TRUE)[[1L]])
  out[nzchar(out)]
}

logical_arg <- function(value, default = FALSE) {
  key <- tolower(as.character(value %||% default)[[1L]])
  if (key %in% c("true", "t", "1", "yes", "on")) return(TRUE)
  if (key %in% c("false", "f", "0", "no", "off")) return(FALSE)
  stop("Invalid logical value: ", key, call. = FALSE)
}

positive_int <- function(value, default, name) {
  out <- suppressWarnings(as.integer(value %||% default))
  if (length(out) != 1L || is.na(out) || out < 1L) {
    stop("`", name, "` must be a positive integer.", call. = FALSE)
  }
  out
}

numeric_arg <- function(value, default, name) {
  out <- suppressWarnings(as.numeric(value %||% default))
  if (length(out) != 1L || is.na(out) || !is.finite(out) || out < 0) {
    stop("`", name, "` must be a finite non-negative number.", call. = FALSE)
  }
  out
}

normalize_metrics <- function(value) {
  out <- tolower(split_arg(value, "euclidean"))
  valid <- c("euclidean", "cosine", "correlation", "inner_product")
  bad <- setdiff(out, valid)
  if (length(bad)) stop("Unsupported metric(s): ", paste(bad, collapse = ", "), call. = FALSE)
  unique(out)
}

script_path <- function() {
  file_arg <- grep("^--file=", commandArgs(FALSE), value = TRUE)
  if (length(file_arg)) return(normalizePath(sub("^--file=", "", file_arg[[1L]]), mustWork = TRUE))
  normalizePath("benchmark_precompute_exact_references_cuda.R", mustWork = TRUE)
}

dataset_path_column <- function(manifest) {
  found <- intersect(c("path", "output", "file", "file_path", "rdata_path"), names(manifest))
  if (!length(found)) stop("Manifest has no dataset path column.", call. = FALSE)
  found[[1L]]
}

load_dataset_matrix <- function(path) {
  env <- new.env(parent = emptyenv())
  load(path, envir = env)
  if (exists("dataset", envir = env, inherits = FALSE)) {
    value <- get("dataset", envir = env, inherits = FALSE)
    if (is.list(value) && !is.null(value$data)) return(value$data)
  }
  for (name in ls(env)) {
    value <- get(name, envir = env, inherits = FALSE)
    if (is.list(value) && !is.null(value$data)) return(value$data)
  }
  stop("No list containing `$data` was found in ", path, call. = FALSE)
}

matrix_dims <- function(x) {
  dims <- dim(x)
  if (is.null(dims) && methods::is(x, "float32")) dims <- dim(methods::slot(x, "Data"))
  if (is.null(dims) || length(dims) != 2L) stop("Dataset is not a matrix.", call. = FALSE)
  as.integer(dims)
}

configure_threads <- function(n) {
  value <- as.character(as.integer(n))
  Sys.setenv(
    OMP_NUM_THREADS = value,
    OPENBLAS_NUM_THREADS = value,
    MKL_NUM_THREADS = value,
    VECLIB_MAXIMUM_THREADS = value,
    RCPP_PARALLEL_NUM_THREADS = value
  )
}

standardize_knn <- function(x) {
  indices <- x$indices %||% x$idx %||% x$nn.idx
  distances <- x$distances %||% x$dist %||% x$nn.dists
  if (is.null(indices) || is.null(distances)) stop("KNN result has no indices/distances.", call. = FALSE)
  list(indices = as.matrix(indices), distances = as.matrix(distances))
}

remove_self <- function(x, original_rows, k) {
  x <- standardize_knn(x)
  out_i <- matrix(NA_integer_, nrow = length(original_rows), ncol = k)
  out_d <- matrix(NA_real_, nrow = length(original_rows), ncol = k)
  for (i in seq_along(original_rows)) {
    keep <- which(!is.na(x$indices[i, ]) & x$indices[i, ] != original_rows[[i]])
    take <- head(keep, k)
    if (length(take)) {
      out_i[i, seq_along(take)] <- as.integer(x$indices[i, take])
      out_d[i, seq_along(take)] <- as.numeric(x$distances[i, take])
    }
  }
  list(indices = out_i, distances = out_d)
}

reference_file <- function(dataset_path, metric, k, quality_n, seed) {
  file.path(
    dirname(dataset_path),
    sprintf("faissR_exact_reference_%s_k%d_q%d_seed%d.RData", metric, k, quality_n, seed)
  )
}

cuda_reference_is_valid <- function(path, k) {
  if (!file.exists(path)) return(FALSE)
  env <- new.env(parent = emptyenv())
  tryCatch({
    load(path, envir = env)
    ref <- get("faissR_reference", envir = env, inherits = FALSE)
    identical(ref$status, "success") && identical(ref$backend, "cuda") &&
      isTRUE(ref$cpu_audit_pass) && is.matrix(ref$indices) && ncol(ref$indices) >= k
  }, error = function(e) FALSE)
}

audit_reference <- function(x, rows, gpu_ref, k, metric, threads, audit_n, atol, rtol) {
  audit_positions <- unique(round(seq(1, length(rows), length.out = min(audit_n, length(rows)))))
  audit_rows <- rows[audit_positions]
  queries <- x[audit_rows, , drop = FALSE]
  cpu_raw <- faissR::nn(
    x,
    points = queries,
    k = min(k + 1L, matrix_dims(x)[[1L]]),
    exclude_self = FALSE,
    backend = "cpu",
    method = "flat",
    metric = metric,
    tuning = "auto",
    n_threads = threads,
    output = "double"
  )
  cpu_ref <- remove_self(cpu_raw, audit_rows, k)
  gpu_i <- gpu_ref$indices[audit_positions, , drop = FALSE]
  gpu_d <- gpu_ref$distances[audit_positions, , drop = FALSE]

  recalls <- numeric(length(audit_positions))
  distance_ok <- logical(length(audit_positions))
  max_errors <- numeric(length(audit_positions))
  for (i in seq_along(audit_positions)) {
    cpu_ids <- cpu_ref$indices[i, ]
    gpu_ids <- gpu_i[i, ]
    cpu_ids <- cpu_ids[is.finite(cpu_ids)]
    gpu_ids <- gpu_ids[is.finite(gpu_ids)]
    recalls[[i]] <- if (length(cpu_ids)) sum(gpu_ids %in% cpu_ids) / length(cpu_ids) else NA_real_

    cpu_dist <- sort(as.numeric(cpu_ref$distances[i, ]))
    gpu_dist <- sort(as.numeric(gpu_d[i, ]))
    finite <- is.finite(cpu_dist) & is.finite(gpu_dist)
    if (!any(finite)) {
      distance_ok[[i]] <- FALSE
      max_errors[[i]] <- Inf
    } else {
      err <- abs(cpu_dist[finite] - gpu_dist[finite])
      tolerance <- atol + rtol * pmax(abs(cpu_dist[finite]), abs(gpu_dist[finite]))
      distance_ok[[i]] <- length(cpu_dist) == length(gpu_dist) && all(err <= tolerance)
      max_errors[[i]] <- max(err)
    }
  }

  tie_aware_pass <- all(is.finite(recalls)) && all((recalls == 1) | distance_ok)
  list(
    pass = isTRUE(tie_aware_pass),
    n = length(audit_positions),
    rows = audit_rows,
    mean_recall = mean(recalls),
    min_recall = min(recalls),
    max_distance_error = max(max_errors),
    distance_pass_fraction = mean(distance_ok),
    atol = atol,
    rtol = rtol,
    cpu_backend_used = attr(cpu_raw, "backend") %||% cpu_raw$backend_used %||% NA_character_
  )
}

compute_one <- function(config) {
  configure_threads(config$threads)
  suppressPackageStartupMessages(library(faissR))
  if (!isTRUE(faissR::cuda_available()) || !isTRUE(faissR::faiss_gpu_available())) {
    stop("CUDA and FAISS GPU are required for CUDA exact references.", call. = FALSE)
  }

  x <- load_dataset_matrix(config$dataset_path)
  dims <- matrix_dims(x)
  n <- dims[[1L]]
  p <- dims[[2L]]
  if (n < 2L) stop("At least two rows are required.", call. = FALSE)
  set.seed(config$seed + nchar(config$dataset) + config$k)
  rows <- if (n <= config$quality_n) seq_len(n) else sort(sample.int(n, config$quality_n))
  queries <- x[rows, , drop = FALSE]
  audit_budget_n <- floor(config$audit_max_ops / max(1, as.double(n) * as.double(p)))
  effective_audit_n <- as.integer(min(config$audit_n, max(4, audit_budget_n)))

  started <- proc.time()[["elapsed"]]
  gpu_raw <- faissR::nn_gpu(
    x,
    points = queries,
    k = min(config$k + 1L, n),
    exclude_self = FALSE,
    method = "exact",
    metric = config$metric,
    tuning = "auto",
    target_recall = 0.99
  )
  gpu_backend_used <- gpu_raw$backend_used %||% attr(gpu_raw, "backend") %||% NA_character_
  gpu_residency <- gpu_raw$result_residency %||% attr(gpu_raw, "result_residency") %||% "cuda"
  host_started <- proc.time()[["elapsed"]]
  host_raw <- faissR::gpu_knn_to_host(gpu_raw)
  host_copy_sec <- proc.time()[["elapsed"]] - host_started
  gpu_ref <- remove_self(host_raw, rows, config$k)
  gpu_elapsed <- proc.time()[["elapsed"]] - started

  audit <- audit_reference(
    x, rows, gpu_ref, config$k, config$metric, config$threads,
    effective_audit_n, config$audit_atol, config$audit_rtol
  )
  if (!isTRUE(audit$pass)) {
    stop(
      sprintf(
        "CPU audit failed: mean recall %.6f, min recall %.6f, max distance error %.6g",
        audit$mean_recall, audit$min_recall, audit$max_distance_error
      ),
      call. = FALSE
    )
  }

  faissR_reference <- list(
    status = "success",
    dataset = config$dataset,
    dataset_path = config$dataset_path,
    n = n,
    p = p,
    metric = config$metric,
    backend = "cuda",
    method = "exact",
    exact = TRUE,
    backend_used = gpu_backend_used,
    result_residency_during_search = gpu_residency,
    device_to_host_result_copies = 1L,
    host_copy_sec = host_copy_sec,
    k = config$k,
    max_k = config$k,
    quality_n = config$quality_n,
    seed = config$seed,
    rows = rows,
    indices = gpu_ref$indices,
    distances = gpu_ref$distances,
    elapsed_sec = gpu_elapsed,
    cpu_audit_pass = audit$pass,
    cpu_audit_n_requested = config$audit_n,
    cpu_audit_n = audit$n,
    cpu_audit_rows = audit$rows,
    cpu_audit_mean_recall = audit$mean_recall,
    cpu_audit_min_recall = audit$min_recall,
    cpu_audit_max_distance_error = audit$max_distance_error,
    cpu_audit_distance_pass_fraction = audit$distance_pass_fraction,
    cpu_audit_atol = audit$atol,
    cpu_audit_rtol = audit$rtol,
    cpu_audit_backend_used = audit$cpu_backend_used,
    created_at = format(Sys.time(), "%Y-%m-%d %H:%M:%S %Z")
  )
  save(faissR_reference, file = config$output_path, compress = "xz")
  data.frame(
    dataset = config$dataset,
    metric = config$metric,
    seed = config$seed,
    status = "success",
    reference_path = config$output_path,
    reference_backend_used = gpu_backend_used,
    elapsed_sec = gpu_elapsed,
    host_copy_sec = host_copy_sec,
    cpu_audit_n = audit$n,
    cpu_audit_mean_recall = audit$mean_recall,
    cpu_audit_min_recall = audit$min_recall,
    cpu_audit_max_distance_error = audit$max_distance_error,
    cpu_audit_pass = audit$pass,
    error = NA_character_,
    stringsAsFactors = FALSE
  )
}

run_child <- function(args) {
  config <- readRDS(args$config)
  row <- tryCatch(
    compute_one(config),
    error = function(e) data.frame(
      dataset = config$dataset,
      metric = config$metric,
      seed = config$seed,
      status = "failed",
      reference_path = config$output_path,
      reference_backend_used = NA_character_,
      elapsed_sec = NA_real_,
      host_copy_sec = NA_real_,
      cpu_audit_n = config$audit_n,
      cpu_audit_mean_recall = NA_real_,
      cpu_audit_min_recall = NA_real_,
      cpu_audit_max_distance_error = NA_real_,
      cpu_audit_pass = FALSE,
      error = conditionMessage(e),
      stringsAsFactors = FALSE
    )
  )
  saveRDS(row, args$result)
}

run_task <- function(config, timeout, script) {
  cfg <- tempfile("faissR_cuda_ref_", fileext = ".rds")
  result <- tempfile("faissR_cuda_ref_result_", fileext = ".rds")
  saveRDS(config, cfg)
  on.exit(unlink(c(cfg, result)), add = TRUE)
  r_bin <- Sys.getenv("R_BIN", unset = "Rscript")
  command <- c(r_bin, "--vanilla", script, "--child=TRUE", paste0("--config=", cfg), paste0("--result=", result))
  timeout_bin <- Sys.which("timeout")
  if (nzchar(timeout_bin)) command <- c(timeout_bin, as.character(timeout), command)
  status <- system2(command[[1L]], command[-1L])
  if (file.exists(result)) return(readRDS(result))
  data.frame(
    dataset = config$dataset,
    metric = config$metric,
    seed = config$seed,
    status = if (identical(status, 124L)) "timeout" else "failed",
    reference_path = config$output_path,
    reference_backend_used = NA_character_,
    elapsed_sec = if (identical(status, 124L)) timeout else NA_real_,
    host_copy_sec = NA_real_,
    cpu_audit_n = config$audit_n,
    cpu_audit_mean_recall = NA_real_,
    cpu_audit_min_recall = NA_real_,
    cpu_audit_max_distance_error = NA_real_,
    cpu_audit_pass = FALSE,
    error = paste("child process exited with status", status),
    stringsAsFactors = FALSE
  )
}

append_csv <- function(row, path) {
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

main <- function() {
  args <- parse_args()
  if (logical_arg(args$child, FALSE)) return(run_child(args))

  manifest_path <- normalizePath(args$manifest %||% stop("`--manifest` is required."), mustWork = TRUE)
  manifest <- read.csv(manifest_path, stringsAsFactors = FALSE)
  path_col <- dataset_path_column(manifest)
  datasets <- split_arg(args$datasets, paste(manifest$dataset, collapse = ","))
  manifest <- manifest[manifest$dataset %in% datasets, , drop = FALSE]
  metrics <- normalize_metrics(args$metrics)
  seeds <- as.integer(split_arg(args$seeds, "4,20260706,20260807"))
  k <- positive_int(args$reference_k, 100L, "reference_k")
  quality_n <- positive_int(args$quality_n, 1024L, "quality_n")
  audit_n <- positive_int(args$audit_n, 64L, "audit_n")
  audit_max_ops <- numeric_arg(args$audit_max_ops, 5e9, "audit_max_ops")
  threads <- positive_int(args$threads, 2L, "threads")
  timeout <- positive_int(args$timeout, 2000L, "timeout")
  audit_atol <- numeric_arg(args$audit_atol, 1e-5, "audit_atol")
  audit_rtol <- numeric_arg(args$audit_rtol, 1e-4, "audit_rtol")
  resume <- logical_arg(args$resume, TRUE)
  out_dir <- args$out_dir %||% dirname(manifest_path)
  dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
  results_path <- file.path(out_dir, "faissR_cuda_exact_reference_results.csv")
  script <- script_path()

  for (i in seq_len(nrow(manifest))) {
    dataset_path <- manifest[[path_col]][[i]]
    for (metric in metrics) {
      for (seed in seeds) {
        output_path <- reference_file(dataset_path, metric, k, quality_n, seed)
        if (resume && cuda_reference_is_valid(output_path, k)) {
          append_csv(data.frame(
            dataset = manifest$dataset[[i]], metric = metric, seed = seed,
            status = "already_exists", reference_path = output_path,
            reference_backend_used = NA_character_, elapsed_sec = NA_real_,
            host_copy_sec = NA_real_, cpu_audit_n = audit_n,
            cpu_audit_mean_recall = NA_real_, cpu_audit_min_recall = NA_real_,
            cpu_audit_max_distance_error = NA_real_, cpu_audit_pass = TRUE,
            error = NA_character_, stringsAsFactors = FALSE
          ), results_path)
          next
        }
        message(sprintf("CUDA exact reference: %s / %s / seed %d", manifest$dataset[[i]], metric, seed))
        row <- run_task(list(
          dataset = manifest$dataset[[i]],
          dataset_path = dataset_path,
          metric = metric,
          seed = seed,
          k = k,
          quality_n = quality_n,
          audit_n = audit_n,
          audit_max_ops = audit_max_ops,
          audit_atol = audit_atol,
          audit_rtol = audit_rtol,
          threads = threads,
          output_path = output_path
        ), timeout, script)
        append_csv(row, results_path)
      }
    }
  }
  message("DONE: ", results_path)
}

if (!identical(Sys.getenv("FAISSR_CUDA_REFERENCE_SOURCE_ONLY", unset = ""), "true")) {
  main()
}
