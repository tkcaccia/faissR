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
  frames <- sys.frames()
  ofiles <- vapply(frames, function(frame) {
    value <- frame$ofile
    if (is.null(value) || !length(value)) NA_character_ else as.character(value[[1L]])
  }, character(1))
  ofiles <- ofiles[!is.na(ofiles) & nzchar(ofiles) & file.exists(ofiles)]
  if (length(ofiles)) return(normalizePath(ofiles[[length(ofiles)]], mustWork = TRUE))
  ofile <- tryCatch(sys.frame(1)$ofile, error = function(e) NULL)
  normalizePath(ofile %||% file.path("benchmark_scripts", "benchmark_hnsw_tuning_grid.R"), mustWork = FALSE)
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

safe_first <- function(x, default = NA_character_) {
  if (is.null(x) || !length(x)) return(default)
  as.character(x[[1L]])
}

safe_bool <- function(x) {
  if (is.null(x) || !length(x) || is.na(x[[1L]])) return(NA)
  isTRUE(x[[1L]])
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
  if (!"target_recall" %in% names(formals(faissR::nn))) {
    stop(
      "The loaded faissR package is too old for HNSW target-recall tuning. ",
      "Install the current faissR source or set FAISSR_SOURCE_DIR to the package checkout.",
      call. = FALSE
    )
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
    if (inherits(value, "float32")) return(value)
  }
  stop("Dataset file must contain a float32 matrix or a list with `$data`: ", path, call. = FALSE)
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

cpu_candidates <- function(k, target_recalls, grid_level = "standard") {
  auto <- data.frame(
    candidate_id = paste0("auto_", gsub("[.]", "", sprintf("%.2f", target_recalls))),
    candidate_kind = "auto",
    candidate_target_recall = target_recalls,
    m = NA_integer_,
    ef_construction = NA_integer_,
    ef_search = NA_integer_,
    graph_degree = NA_integer_,
    intermediate_graph_degree = NA_integer_,
    ef = NA_integer_,
    cagra_build_algo = NA_character_,
    stringsAsFactors = FALSE
  )
  manual <- data.frame(
    m = c(8L, 8L, 12L, 12L, 16L, 16L, 24L, 32L),
    ef_construction = c(30L, 40L, 40L, 60L, 80L, 80L, 120L, 160L),
    ef_search = pmax(as.integer(k), c(15L, 25L, 35L, 50L, 75L, 100L, 150L, 220L))
  )
  if (identical(grid_level, "compact")) manual <- manual[c(1L, 3L, 5L, 7L), , drop = FALSE]
  if (identical(grid_level, "wide")) {
    manual <- unique(rbind(
      manual,
      data.frame(
        m = c(12L, 16L, 24L, 32L, 48L),
        ef_construction = c(80L, 120L, 160L, 220L, 300L),
        ef_search = pmax(as.integer(k), c(120L, 160L, 220L, 300L, 400L))
      )
    ))
  }
  manual$candidate_id <- sprintf(
    "m%02d_ec%03d_es%03d",
    manual$m,
    manual$ef_construction,
    manual$ef_search
  )
  manual$candidate_kind <- "manual"
  manual$candidate_target_recall <- NA_real_
  manual$graph_degree <- NA_integer_
  manual$intermediate_graph_degree <- NA_integer_
  manual$ef <- NA_integer_
  manual$cagra_build_algo <- NA_character_
  manual <- manual[, names(auto), drop = FALSE]
  unique(rbind(auto, manual))
}

cuda_candidates <- function(k, target_recalls, build_algos = "auto", grid_level = "standard") {
  auto <- data.frame(
    candidate_id = paste0("auto_", gsub("[.]", "", sprintf("%.2f", target_recalls))),
    candidate_kind = "auto",
    candidate_target_recall = target_recalls,
    m = NA_integer_,
    ef_construction = NA_integer_,
    ef_search = NA_integer_,
    graph_degree = NA_integer_,
    intermediate_graph_degree = NA_integer_,
    ef = NA_integer_,
    cagra_build_algo = NA_character_,
    stringsAsFactors = FALSE
  )
  manual <- data.frame(
    graph_degree = c(16L, 24L, 32L, 32L, 48L, 48L, 64L, 64L),
    intermediate_graph_degree = c(32L, 48L, 64L, 96L, 96L, 128L, 128L, 192L),
    ef = pmax(as.integer(k), c(32L, 48L, 64L, 96L, 96L, 128L, 160L, 256L))
  )
  if (identical(grid_level, "compact")) manual <- manual[c(1L, 3L, 5L, 7L), , drop = FALSE]
  if (identical(grid_level, "wide")) {
    manual <- unique(rbind(
      manual,
      data.frame(
        graph_degree = c(80L, 96L),
        intermediate_graph_degree = c(192L, 256L),
        ef = pmax(as.integer(k), c(320L, 400L))
      )
    ))
  }
  build_algos <- split_arg(build_algos, "auto")
  manual <- do.call(rbind, lapply(build_algos, function(algo) {
    x <- manual
    x$cagra_build_algo <- algo
    x
  }))
  manual$candidate_id <- sprintf(
    "gd%03d_igd%03d_ef%03d_%s",
    manual$graph_degree,
    manual$intermediate_graph_degree,
    manual$ef,
    gsub("[^A-Za-z0-9]+", "", manual$cagra_build_algo)
  )
  manual$candidate_kind <- "manual"
  manual$candidate_target_recall <- NA_real_
  manual$m <- NA_integer_
  manual$ef_construction <- NA_integer_
  manual$ef_search <- NA_integer_
  manual <- manual[, names(auto), drop = FALSE]
  unique(rbind(auto, manual))
}

candidate_grid <- function(backend, k, target_recalls, grid_level, cuda_build_algos) {
  if (identical(backend, "cpu")) {
    cpu_candidates(k, target_recalls, grid_level = grid_level)
  } else {
    cuda_candidates(k, target_recalls, build_algos = cuda_build_algos, grid_level = grid_level)
  }
}

apply_candidate_options <- function(candidate, backend) {
  if (identical(candidate$candidate_kind, "auto")) {
    options(
      faissR.faiss_hnsw_m = NULL,
      faissR.faiss_hnsw_ef_construction = NULL,
      faissR.faiss_hnsw_ef_search = NULL,
      faissR.cuvs_graph_degree = NULL,
      faissR.cuvs_intermediate_graph_degree = NULL,
      faissR.cuvs_hnsw_ef = NULL,
      faissR.cuvs_cagra_build_algo = NULL
    )
    return(invisible(NULL))
  }
  if (identical(backend, "cpu")) {
    options(
      faissR.faiss_hnsw_m = as.integer(candidate$m),
      faissR.faiss_hnsw_ef_construction = as.integer(candidate$ef_construction),
      faissR.faiss_hnsw_ef_search = as.integer(candidate$ef_search)
    )
  } else {
    options(
      faissR.cuvs_graph_degree = as.integer(candidate$graph_degree),
      faissR.cuvs_intermediate_graph_degree = as.integer(candidate$intermediate_graph_degree),
      faissR.cuvs_hnsw_ef = as.integer(candidate$ef),
      faissR.cuvs_cagra_build_algo = as.character(candidate$cagra_build_algo)
    )
  }
  invisible(NULL)
}

empty_row <- function(config) {
  candidate <- config$candidate
  data.frame(
    dataset = config$dataset,
    n = as.integer(config$n),
    p = as.integer(config$p),
    shape_group = shape_group(config$n, config$p),
    backend = config$backend,
    method = "hnsw",
    metric = "euclidean",
    k = as.integer(config$k),
    candidate_id = candidate$candidate_id,
    candidate_kind = candidate$candidate_kind,
    candidate_target_recall = as.numeric(candidate$candidate_target_recall),
    n_threads = as.integer(config$n_threads),
    output = config$output,
    status = NA_character_,
    elapsed_sec = NA_real_,
    peak_rss_gb = NA_real_,
    recall_at_k = NA_real_,
    median_recall_at_k = NA_real_,
    min_recall_at_k = NA_real_,
    reference_status = config$reference_status %||% NA_character_,
    reference_backend = config$reference_backend %||% NA_character_,
    reference_query_n = as.integer(config$reference_query_n %||% NA_integer_),
    cpu_m = as.integer(candidate$m),
    cpu_ef_construction = as.integer(candidate$ef_construction),
    cpu_ef_search = as.integer(candidate$ef_search),
    cuda_graph_degree = as.integer(candidate$graph_degree),
    cuda_intermediate_graph_degree = as.integer(candidate$intermediate_graph_degree),
    cuda_ef = as.integer(candidate$ef),
    cagra_build_algo = candidate$cagra_build_algo %||% NA_character_,
    result_backend = NA_character_,
    implementation_backend = NA_character_,
    resolved_backend = NA_character_,
    distance_type = NA_character_,
    input_type = NA_character_,
    input_layout = NA_character_,
    tuning_rule = NA_character_,
    result_m = NA_integer_,
    result_ef_construction = NA_integer_,
    result_ef_search = NA_integer_,
    result_graph_degree = NA_integer_,
    result_intermediate_graph_degree = NA_integer_,
    result_ef = NA_integer_,
    result_hnsw_m = NA_integer_,
    result_hnsw_ef_construction = NA_integer_,
    error = NA_character_,
    stringsAsFactors = FALSE
  )
}

run_reference_child <- function(config) {
  configure_threads(config$n_threads)
  configure_native_libs()
  load_faissR()
  x <- load_float_dataset(config$dataset_path)
  rows <- config$rows
  started <- proc.time()[["elapsed"]]
  out <- faissR::nn(
    x,
    points = subset_rows(x, rows),
    k = min(as.integer(config$k) + 1L, nrow(x)),
    backend = config$backend,
    method = "exact",
    metric = "euclidean",
    output = config$output,
    n_threads = config$n_threads
  )
  saveRDS(
    list(
      status = "success",
      backend = config$backend,
      rows = rows,
      indices = remove_self_from_query_knn(out$indices, rows, config$k),
      elapsed_sec = proc.time()[["elapsed"]] - started,
      peak_rss_gb = read_peak_rss_gb(),
      error = NA_character_
    ),
    config$output_path
  )
}

run_method_child <- function(config) {
  configure_threads(config$n_threads)
  configure_native_libs()
  load_faissR()
  x <- load_float_dataset(config$dataset_path)
  candidate <- config$candidate
  apply_candidate_options(candidate, config$backend)
  target <- if (identical(candidate$candidate_kind, "auto")) {
    as.numeric(candidate$candidate_target_recall)
  } else {
    0.99
  }
  row <- empty_row(config)
  started <- proc.time()[["elapsed"]]
  out <- faissR::nn(exclude_self = TRUE,
    x,
    k = config$k,
    backend = config$backend,
    method = "hnsw",
    metric = "euclidean",
    target_recall = target,
    output = config$output,
    n_threads = config$n_threads
  )
  elapsed <- proc.time()[["elapsed"]] - started
  quality <- recall_summary(out$indices[config$rows, , drop = FALSE], config$reference_indices)
  approx <- attr(out, "approximation") %||% list()
  row$status <- "success"
  row$elapsed_sec <- elapsed
  row$peak_rss_gb <- read_peak_rss_gb()
  row$recall_at_k <- quality$recall_at_k
  row$median_recall_at_k <- quality$median_recall_at_k
  row$min_recall_at_k <- quality$min_recall_at_k
  row$result_backend <- safe_first(out$backend_used)
  row$implementation_backend <- safe_first(attr(out, "implementation_backend"))
  row$resolved_backend <- safe_first(attr(out, "resolved_backend"))
  row$distance_type <- safe_first(attr(out, "distance_type") %||% out$distance_type)
  row$input_type <- safe_first(attr(out, "input_type") %||% out$input_type)
  row$input_layout <- safe_first(attr(out, "input_layout") %||% out$input_layout)
  row$tuning_rule <- safe_first(approx$tuning_rule)
  row$result_m <- suppressWarnings(as.integer(approx$m %||% NA_integer_))
  row$result_ef_construction <- suppressWarnings(as.integer(approx$ef_construction %||% NA_integer_))
  row$result_ef_search <- suppressWarnings(as.integer(approx$ef_search %||% NA_integer_))
  row$result_graph_degree <- suppressWarnings(as.integer(approx$graph_degree %||% NA_integer_))
  row$result_intermediate_graph_degree <- suppressWarnings(as.integer(approx$intermediate_graph_degree %||% NA_integer_))
  row$result_ef <- suppressWarnings(as.integer(approx$ef %||% NA_integer_))
  row$result_hnsw_m <- suppressWarnings(as.integer(approx$hnsw_m %||% NA_integer_))
  row$result_hnsw_ef_construction <- suppressWarnings(as.integer(approx$hnsw_ef_construction %||% NA_integer_))
  saveRDS(row, config$output_path)
}

run_child_mode <- function(args) {
  config <- readRDS(args$config)
  tryCatch({
    if (identical(args$child_task, "reference")) {
      run_reference_child(config)
    } else if (identical(args$child_task, "method")) {
      run_method_child(config)
    } else {
      stop("Unknown child task.", call. = FALSE)
    }
  }, error = function(e) {
    if (identical(args$child_task, "method")) {
      row <- empty_row(config)
      row$status <- classify_error(conditionMessage(e))
      row$error <- conditionMessage(e)
      row$peak_rss_gb <- read_peak_rss_gb()
      saveRDS(row, config$output_path)
    } else {
      saveRDS(
        list(
          status = classify_error(conditionMessage(e)),
          backend = config$backend,
          rows = config$rows,
          indices = NULL,
          elapsed_sec = NA_real_,
          peak_rss_gb = read_peak_rss_gb(),
          error = conditionMessage(e)
        ),
        config$output_path
      )
    }
  })
}

run_rscript_task <- function(task, config, timeout, bench_script) {
  cfg <- tempfile("faissR_hnsw_tune_cfg_", fileext = ".rds")
  out <- tempfile("faissR_hnsw_tune_out_", fileext = ".rds")
  on.exit(unlink(c(cfg, out), force = TRUE), add = TRUE)
  config$output_path <- out
  saveRDS(config, cfg)
  rscript <- file.path(R.home("bin"), "Rscript")
  timeout_bin <- Sys.which("timeout")
  if (nzchar(timeout_bin)) {
    parts <- c(
      timeout_bin,
      as.character(as.integer(ceiling(timeout))),
      rscript,
      bench_script,
      paste0("--child_task=", task),
      paste0("--config=", cfg)
    )
  } else {
    parts <- c(
      rscript,
      bench_script,
      paste0("--child_task=", task),
      paste0("--config=", cfg)
    )
  }
  cmd <- paste(c(shQuote(parts), "2>&1"), collapse = " ")
  status <- system(cmd, intern = TRUE)
  exit_status <- attr(status, "status") %||% 0L
  timed_out <- nzchar(timeout_bin) && identical(as.integer(exit_status), 124L)
  if (file.exists(out)) return(readRDS(out))
  if (identical(task, "method")) {
    row <- empty_row(config)
    row$status <- classify_error(paste(status, collapse = "\n"), timed_out = timed_out)
    row$error <- if (timed_out) {
      sprintf("child process timed out after %s seconds", timeout)
    } else {
      paste(status, collapse = "\n")
    }
    row$elapsed_sec <- if (timed_out) as.numeric(timeout) else NA_real_
    return(row)
  }
  list(
    status = classify_error(paste(status, collapse = "\n"), timed_out = timed_out),
    backend = config$backend,
    rows = config$rows,
    indices = NULL,
    elapsed_sec = if (timed_out) as.numeric(timeout) else NA_real_,
    peak_rss_gb = NA_real_,
    error = if (timed_out) sprintf("reference timed out after %s seconds", timeout) else paste(status, collapse = "\n")
  )
}

make_reference <- function(dataset, dataset_path, n, k, backend, n_threads,
                           output, quality_n, seed, timeout, bench_script,
                           reference_backend = backend) {
  set.seed(seed + nchar(dataset) + as.integer(k))
  rows <- if (n <= quality_n) seq_len(n) else sort(sample.int(n, quality_n))
  backends <- switch(
    reference_backend,
    cpu = "cpu",
    cuda = c("cuda", "cpu"),
    auto = if (identical(backend, "cuda")) c("cuda", "cpu") else "cpu",
    c(reference_backend, "cpu")
  )
  errors <- character()
  for (ref_backend in unique(backends)) {
    cfg <- list(
      dataset = dataset,
      dataset_path = dataset_path,
      rows = rows,
      k = k,
      backend = ref_backend,
      n_threads = n_threads,
      output = output
    )
    ref <- run_rscript_task("reference", cfg, timeout, bench_script)
    if (identical(ref$status, "success") && !is.null(ref$indices)) {
      ref$query_n <- length(rows)
      return(ref)
    }
    errors <- c(errors, sprintf(
      "%s: %s",
      ref_backend,
      ref$error %||% ref$status %||% "reference failed"
    ))
  }
  list(
    status = "unavailable",
    backend = NA_character_,
    rows = rows,
    indices = NULL,
    query_n = length(rows),
    elapsed_sec = NA_real_,
    peak_rss_gb = NA_real_,
    error = paste(c("No exact reference completed within timeout.", errors), collapse = " | ")
  )
}

row_key <- function(dataset, backend, k, candidate_id) {
  paste(dataset, backend, as.integer(k), candidate_id, sep = "\r")
}

existing_keys <- function(results_path) {
  if (!file.exists(results_path)) return(character())
  x <- read.csv(results_path, stringsAsFactors = FALSE)
  if (!nrow(x) || !all(c("dataset", "backend", "k", "candidate_id") %in% names(x))) {
    return(character())
  }
  row_key(x$dataset, x$backend, x$k, x$candidate_id)
}

safe_median <- function(x) {
  x <- suppressWarnings(as.numeric(x))
  x <- x[is.finite(x)]
  if (!length(x)) return(NA_real_)
  stats::median(x)
}

safe_min <- function(x) {
  x <- suppressWarnings(as.numeric(x))
  x <- x[is.finite(x)]
  if (!length(x)) return(NA_real_)
  min(x)
}

write_tuning_summaries <- function(out_dir, results_path, target_recalls) {
  if (!file.exists(results_path)) return(invisible(NULL))
  x <- read.csv(results_path, stringsAsFactors = FALSE)
  if (!nrow(x)) return(invisible(NULL))
  x$recall_at_k <- suppressWarnings(as.numeric(x$recall_at_k))
  x$elapsed_sec <- suppressWarnings(as.numeric(x$elapsed_sec))
  rec_rows <- list()
  pos <- 0L
  for (target in target_recalls) {
    parts <- split(x, interaction(x[c("dataset", "backend", "k")], drop = TRUE, lex.order = TRUE))
    for (part in parts) {
      ok <- part[part$status == "success" & is.finite(part$recall_at_k), , drop = FALSE]
      if (nrow(ok)) {
        eligible <- ok[ok$recall_at_k >= target, , drop = FALSE]
        chosen <- if (nrow(eligible)) {
          eligible[order(eligible$elapsed_sec, -eligible$recall_at_k), , drop = FALSE][1L, , drop = FALSE]
        } else {
          ok[order(-ok$recall_at_k, ok$elapsed_sec), , drop = FALSE][1L, , drop = FALSE]
        }
      } else {
        chosen <- part[1L, , drop = FALSE]
      }
      chosen$target_recall_threshold <- target
      chosen$meets_target <- chosen$status == "success" &&
        is.finite(chosen$recall_at_k) && chosen$recall_at_k >= target
      pos <- pos + 1L
      rec_rows[[pos]] <- chosen
    }
  }
  recommendations <- do.call(rbind, rec_rows)
  row.names(recommendations) <- NULL
  recommendations <- recommendations[order(
    recommendations$dataset,
    recommendations$backend,
    recommendations$k,
    recommendations$target_recall_threshold
  ), , drop = FALSE]
  write.csv(recommendations, file.path(out_dir, "hnsw_tuning_recommendations.csv"), row.names = FALSE)

  aggregate_rows <- list()
  pos <- 0L
  for (target in target_recalls) {
    y <- x
    y$target_recall_threshold <- target
    y$meets_target <- y$status == "success" & is.finite(y$recall_at_k) & y$recall_at_k >= target
    keys <- c("shape_group", "backend", "k", "target_recall_threshold", "candidate_id")
    parts <- split(y, interaction(y[keys], drop = TRUE, lex.order = TRUE))
    for (part in parts) {
      pos <- pos + 1L
      aggregate_rows[[pos]] <- data.frame(
        part[1L, keys, drop = FALSE],
        candidate_kind = part$candidate_kind[[1L]],
        datasets = length(unique(part$dataset)),
        success = sum(part$status == "success"),
        meets_target = sum(part$meets_target, na.rm = TRUE),
        meet_rate = mean(part$meets_target, na.rm = TRUE),
        median_elapsed_sec = safe_median(part$elapsed_sec[part$status == "success"]),
        median_recall_at_k = safe_median(part$recall_at_k[part$status == "success"]),
        min_recall_at_k = safe_min(part$recall_at_k[part$status == "success"]),
        cpu_m = part$cpu_m[[1L]],
        cpu_ef_construction = part$cpu_ef_construction[[1L]],
        cpu_ef_search = part$cpu_ef_search[[1L]],
        cuda_graph_degree = part$cuda_graph_degree[[1L]],
        cuda_intermediate_graph_degree = part$cuda_intermediate_graph_degree[[1L]],
        cuda_ef = part$cuda_ef[[1L]],
        cagra_build_algo = part$cagra_build_algo[[1L]],
        stringsAsFactors = FALSE
      )
    }
  }
  shape_candidates <- do.call(rbind, aggregate_rows)
  row.names(shape_candidates) <- NULL
  shape_candidates <- shape_candidates[order(
    shape_candidates$shape_group,
    shape_candidates$backend,
    shape_candidates$k,
    shape_candidates$target_recall_threshold,
    -shape_candidates$meet_rate,
    shape_candidates$median_elapsed_sec,
    -shape_candidates$min_recall_at_k
  ), , drop = FALSE]
  write.csv(shape_candidates, file.path(out_dir, "hnsw_tuning_shape_candidates.csv"), row.names = FALSE)

  best_shape <- do.call(rbind, lapply(
    split(shape_candidates, interaction(shape_candidates[c("shape_group", "backend", "k", "target_recall_threshold")], drop = TRUE, lex.order = TRUE)),
    function(part) {
      part[order(
        -part$meet_rate,
        -part$success,
        part$median_elapsed_sec,
        -part$min_recall_at_k
      ), , drop = FALSE][1L, , drop = FALSE]
    }
  ))
  row.names(best_shape) <- NULL
  write.csv(best_shape, file.path(out_dir, "hnsw_tuning_shape_recommendations.csv"), row.names = FALSE)
  write_report(out_dir, x, recommendations, best_shape)
  invisible(recommendations)
}

md_table <- function(x, cols = names(x), max_rows = 40L, digits = 4L) {
  if (!nrow(x)) return("_No rows._")
  cols <- intersect(cols, names(x))
  x <- x[seq_len(min(nrow(x), max_rows)), cols, drop = FALSE]
  for (name in names(x)) {
    if (is.numeric(x[[name]])) {
      x[[name]] <- ifelse(is.na(x[[name]]), "", format(round(x[[name]], digits), trim = TRUE))
    } else {
      x[[name]] <- ifelse(is.na(x[[name]]), "", as.character(x[[name]]))
    }
  }
  c(
    paste0("| ", paste(names(x), collapse = " | "), " |"),
    paste0("| ", paste(rep("---", ncol(x)), collapse = " | "), " |"),
    apply(x, 1L, function(row) paste0("| ", paste(row, collapse = " | "), " |"))
  )
}

write_report <- function(out_dir, results, recommendations, best_shape) {
  lines <- c(
    "# HNSW Tuning Grid Report",
    "",
    "- Method: `hnsw`.",
    "- Metric: Euclidean/L2.",
    "- Backends are evaluated separately; `backend = \"auto\"` is not used.",
    "- Inputs are float32 `.RData` datasets from the generated manifest.",
    "- Recall is sampled against exact KNN reference rows.",
    "",
    "## Recommended Rows",
    "",
    md_table(
      recommendations,
      cols = c(
        "dataset", "shape_group", "backend", "k", "target_recall_threshold",
        "candidate_id", "status", "elapsed_sec", "recall_at_k", "meets_target",
        "cpu_m", "cpu_ef_construction", "cpu_ef_search",
        "cuda_graph_degree", "cuda_intermediate_graph_degree", "cuda_ef",
        "cagra_build_algo"
      ),
      max_rows = 120L
    ),
    "",
    "## Shape-Level Candidates",
    "",
    md_table(
      best_shape,
      cols = c(
        "shape_group", "backend", "k", "target_recall_threshold", "candidate_id",
        "datasets", "success", "meets_target", "meet_rate", "median_elapsed_sec",
        "median_recall_at_k", "min_recall_at_k",
        "cpu_m", "cpu_ef_construction", "cpu_ef_search",
        "cuda_graph_degree", "cuda_intermediate_graph_degree", "cuda_ef",
        "cagra_build_algo"
      ),
      max_rows = 160L
    )
  )
  writeLines(lines, file.path(out_dir, "hnsw_tuning_report.md"))
}

write_missing_rows <- function(dataset, backend, k_values, candidates, reason, results_path) {
  for (k in k_values) {
    for (i in seq_len(nrow(candidates))) {
      cfg <- list(
        dataset = dataset$dataset,
        n = as.integer(dataset$n %||% NA_integer_),
        p = as.integer(dataset$p %||% NA_integer_),
        backend = backend,
        k = as.integer(k),
        n_threads = NA_integer_,
        output = NA_character_,
        reference_status = "not_run",
        reference_backend = NA_character_,
        reference_query_n = NA_integer_,
        candidate = candidates[i, , drop = FALSE]
      )
      row <- empty_row(cfg)
      row$status <- "missing_dataset"
      row$error <- reason
      append_csv(row, results_path)
    }
  }
}

main <- function() {
  args <- parse_args()
  if (!is.null(args$child_task)) return(run_child_mode(args))

  manifest_path <- normalizePath(args$manifest, mustWork = TRUE)
  out_dir <- normalizePath(args$out_dir %||% file.path(getwd(), paste0("faissR_HNSW_TUNING_", format(Sys.time(), "%Y%m%d_%H%M%S"))), mustWork = FALSE)
  dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
  results_path <- file.path(out_dir, "hnsw_tuning_results.csv")
  config_path <- file.path(out_dir, "hnsw_tuning_config.csv")
  candidate_path <- file.path(out_dir, "hnsw_tuning_candidate_grid.csv")

  backend <- tolower(args$backend %||% "cpu")
  if (!backend %in% c("cpu", "cuda")) stop("`backend` must be `cpu` or `cuda`.", call. = FALSE)
  if (identical(backend, "cuda")) {
    stop(
      "CUDA HNSW tuning is disabled: RAPIDS cuVS HNSW is a ",
      "CAGRA-to-hnswlib host-wrapper path, not a pure all-GPU HNSW ",
      "implementation. Use CUDA method=\"cagra\" benchmarks for GPU graph ",
      "search, or run this HNSW tuning grid with `--backend=cpu`.",
      call. = FALSE
    )
  }
  k_values <- as.integer(split_arg(args$k_values %||% args$k, "10,15,50,100"))
  target_recalls <- suppressWarnings(as.numeric(split_arg(args$target_recalls, "0.9,0.95,0.99")))
  if (!all(target_recalls %in% c(0.9, 0.95, 0.99))) {
    stop("`target_recalls` must contain only 0.9, 0.95, and/or 0.99.", call. = FALSE)
  }
  n_threads <- positive_int(args$threads, 12L, "threads")
  timeout <- positive_num(args$timeout, 600, "timeout")
  quality_n <- positive_int(args$quality_n, 256L, "quality_n")
  seed <- positive_int(args$seed, 20260624L, "seed")
  output <- args$output %||% "float"
  if (!output %in% c("float", "double")) stop("`output` must be `float` or `double`.", call. = FALSE)
  grid_level <- args$grid_level %||% "standard"
  if (!grid_level %in% c("compact", "standard", "wide")) {
    stop("`grid_level` must be compact, standard, or wide.", call. = FALSE)
  }
  cuda_build_algos <- args$cuda_build_algos %||% "auto"
  resume <- logical_arg(args$resume, TRUE)
  reference_backend <- args$reference_backend %||% backend

  manifest <- read.csv(manifest_path, stringsAsFactors = FALSE)
  datasets <- split_arg(args$datasets, paste(manifest$dataset, collapse = ","))
  missing_names <- setdiff(datasets, manifest$dataset)
  if (length(missing_names)) stop("Unknown dataset(s): ", paste(missing_names, collapse = ", "), call. = FALSE)
  manifest <- manifest[match(datasets, manifest$dataset), , drop = FALSE]

  candidate_rows <- do.call(rbind, lapply(k_values, function(k) {
    x <- candidate_grid(backend, k, target_recalls, grid_level, cuda_build_algos)
    x$backend <- backend
    x$k <- as.integer(k)
    x
  }))
  write.csv(candidate_rows, candidate_path, row.names = FALSE)
  write.csv(
    data.frame(
      key = c("manifest", "out_dir", "datasets", "backend", "method", "metric",
              "k_values", "target_recalls", "threads", "timeout", "quality_n",
              "seed", "output", "grid_level", "cuda_build_algos", "resume",
              "reference_backend"),
      value = c(manifest_path, out_dir, paste(datasets, collapse = ","), backend,
                "hnsw", "euclidean", paste(k_values, collapse = ","),
                paste(target_recalls, collapse = ","), n_threads, timeout,
                quality_n, seed, output, grid_level, cuda_build_algos, resume,
                reference_backend),
      stringsAsFactors = FALSE
    ),
    config_path,
    row.names = FALSE
  )

  done <- if (resume) existing_keys(results_path) else character()
  bench_script <- script_path()
  if (!file.exists(bench_script)) {
    fallback_script <- file.path(getwd(), "benchmark_scripts", "benchmark_hnsw_tuning_grid.R")
    if (file.exists(fallback_script)) {
      bench_script <- normalizePath(fallback_script, mustWork = TRUE)
    }
  }
  if (!file.exists(bench_script)) {
    stop("Cannot resolve benchmark script path for child Rscript calls: ", bench_script, call. = FALSE)
  }
  message("Child benchmark script: ", bench_script)
  for (i in seq_len(nrow(manifest))) {
    ds <- manifest[i, , drop = FALSE]
    dataset <- ds$dataset[[1L]]
    if (!identical(ds$status[[1L]], "success")) {
      for (k in k_values) {
        candidates <- candidate_grid(backend, k, target_recalls, grid_level, cuda_build_algos)
        write_missing_rows(ds, backend, k, candidates, ds$error[[1L]] %||% ds$status[[1L]], results_path)
      }
      next
    }
    dataset_path <- ds$output[[1L]]
    n <- as.integer(ds$n[[1L]])
    p <- as.integer(ds$p[[1L]])
    for (k in k_values) {
      candidates <- candidate_grid(backend, k, target_recalls, grid_level, cuda_build_algos)
      message(sprintf("[%s] reference dataset=%s backend=%s k=%s", Sys.time(), dataset, backend, k))
      reference <- make_reference(
        dataset = dataset,
        dataset_path = dataset_path,
        n = n,
        k = k,
        backend = backend,
        n_threads = n_threads,
        output = output,
        quality_n = quality_n,
        seed = seed,
        timeout = timeout,
        bench_script = bench_script,
        reference_backend = reference_backend
      )
      message(sprintf(
        "[%s] reference status=%s backend=%s rows=%s",
        Sys.time(), reference$status, reference$backend %||% NA_character_, reference$query_n
      ))
      for (j in seq_len(nrow(candidates))) {
        candidate <- candidates[j, , drop = FALSE]
        key <- row_key(dataset, backend, k, candidate$candidate_id)
        if (key %in% done) {
          message(sprintf("[%s] skip completed dataset=%s backend=%s k=%s candidate=%s",
                          Sys.time(), dataset, backend, k, candidate$candidate_id))
          next
        }
        cfg <- list(
          dataset = dataset,
          dataset_path = dataset_path,
          n = n,
          p = p,
          rows = reference$rows,
          reference_indices = reference$indices,
          reference_status = reference$status,
          reference_backend = reference$backend,
          reference_query_n = reference$query_n,
          backend = backend,
          k = as.integer(k),
          n_threads = n_threads,
          output = output,
          candidate = candidate
        )
        message(sprintf("[%s] run dataset=%s backend=%s k=%s candidate=%s",
                        Sys.time(), dataset, backend, k, candidate$candidate_id))
        row <- if (identical(reference$status, "success")) {
          run_rscript_task("method", cfg, timeout, bench_script)
        } else {
          out <- empty_row(cfg)
          out$status <- "reference_failed"
          out$error <- reference$error %||% "reference failed"
          out
        }
        append_csv(row, results_path)
        done <- c(done, key)
        message(sprintf("[%s] -> status=%s elapsed=%s recall=%s",
                        Sys.time(), row$status[[1L]],
                        format(row$elapsed_sec[[1L]], digits = 4),
                        format(row$recall_at_k[[1L]], digits = 4)))
        write_tuning_summaries(out_dir, results_path, target_recalls)
      }
    }
  }
  write_tuning_summaries(out_dir, results_path, target_recalls)
  message("DONE: ", out_dir)
}

main()
