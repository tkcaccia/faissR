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

split_arg <- function(x, default) {
  trimws(strsplit(x %||% default, ",", fixed = TRUE)[[1L]])
}

logical_arg <- function(x, default = FALSE) {
  if (is.null(x) || length(x) == 0L || is.na(x[[1L]])) return(isTRUE(default))
  key <- tolower(trimws(as.character(x[[1L]])))
  if (key %in% c("true", "t", "1", "yes", "y", "on")) return(TRUE)
  if (key %in% c("false", "f", "0", "no", "n", "off")) return(FALSE)
  stop("Logical argument must be true or false.", call. = FALSE)
}

positive_int <- function(x, default, arg) {
  value <- suppressWarnings(as.numeric(x %||% default))
  if (length(value) != 1L || is.na(value) || !is.finite(value) ||
      value < 1L || abs(value - round(value)) > sqrt(.Machine$double.eps)) {
    stop("`", arg, "` must be a positive integer.", call. = FALSE)
  }
  as.integer(round(value))
}

positive_num <- function(x, default, arg) {
  value <- suppressWarnings(as.numeric(x %||% default))
  if (length(value) != 1L || is.na(value) || !is.finite(value) || value <= 0) {
    stop("`", arg, "` must be a positive number.", call. = FALSE)
  }
  value
}

script_path <- function() {
  args <- commandArgs(FALSE)
  file_arg <- grep("^--file=", args, value = TRUE)
  if (length(file_arg)) return(normalizePath(sub("^--file=", "", file_arg[[1L]]), mustWork = TRUE))
  normalizePath(sys.frame(1)$ofile %||% "benchmark_nn_float32_euclidean.R", mustWork = FALSE)
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

classify_error <- function(message, timed_out = FALSE) {
  if (isTRUE(timed_out)) return("timeout")
  msg <- tolower(as.character(message %||% ""))
  if (grepl("not available|unavailable|not built|requires|no .*support|without cuda|without faiss|without cuvs", msg)) {
    return("unavailable")
  }
  if (grepl("not support|does not support|only supports|only available|must be 2|must be 3|grid", msg)) {
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

load_float_dataset <- function(path) {
  env <- new.env(parent = emptyenv())
  load(path, envir = env)
  obj <- if (exists("dataset", envir = env, inherits = FALSE)) {
    get("dataset", envir = env, inherits = FALSE)
  } else {
    candidates <- ls(env)
    hit <- NULL
    for (name in candidates) {
      value <- get(name, envir = env, inherits = FALSE)
      if (is.list(value) && !is.null(value$data)) {
        hit <- value
        break
      }
    }
    hit
  }
  if (!is.list(obj) || is.null(obj$data)) {
    stop("Dataset file must contain a list with `$data`: ", path, call. = FALSE)
  }
  x <- obj$data
  if (!inherits(x, "float32")) {
    warning("Loaded data is not a float::fl()/float32 matrix: ", path, call. = FALSE)
  }
  x
}

subset_rows <- function(x, rows) {
  x[rows, , drop = FALSE]
}

remove_self_from_query_knn <- function(indices, distances, rows, k) {
  out_i <- matrix(NA_integer_, nrow = length(rows), ncol = k)
  out_d <- matrix(NA_real_, nrow = length(rows), ncol = k)
  for (i in seq_along(rows)) {
    keep <- which(indices[i, ] != rows[[i]])
    keep <- keep[seq_len(min(length(keep), k))]
    if (length(keep)) {
      out_i[i, seq_along(keep)] <- indices[i, keep]
      if (!is.null(distances)) out_d[i, seq_along(keep)] <- as.numeric(distances[i, keep])
    }
  }
  list(indices = out_i, distances = out_d)
}

recall_summary <- function(actual_indices, reference_indices) {
  if (is.null(reference_indices) || !length(reference_indices)) {
    return(list(recall_at_k = NA_real_, median_recall_at_k = NA_real_, min_recall_at_k = NA_real_))
  }
  n <- min(nrow(actual_indices), nrow(reference_indices))
  if (!is.finite(n) || n < 1L) {
    return(list(recall_at_k = NA_real_, median_recall_at_k = NA_real_, min_recall_at_k = NA_real_))
  }
  vals <- numeric(n)
  for (i in seq_len(n)) {
    ref <- reference_indices[i, ]
    ref <- ref[!is.na(ref)]
    got <- actual_indices[i, ]
    got <- got[!is.na(got)]
    vals[[i]] <- if (length(ref)) sum(got %in% ref) / length(ref) else NA_real_
  }
  list(
    recall_at_k = mean(vals, na.rm = TRUE),
    median_recall_at_k = stats::median(vals, na.rm = TRUE),
    min_recall_at_k = min(vals, na.rm = TRUE)
  )
}

empty_result_row <- function(config) {
  data.frame(
    dataset = config$dataset,
    n = as.integer(config$n),
    p = as.integer(config$p),
    backend = config$backend %||% NA_character_,
    method = config$method %||% NA_character_,
    metric = config$metric,
    k = as.integer(config$k),
    target_recall_requested = as.numeric(config$target_recall %||% NA_real_),
    target_recall_actual = NA_real_,
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
    reference_mode = config$reference_mode %||% NA_character_,
    reference_query_n = as.integer(config$reference_query_n %||% NA_integer_),
    result_backend = NA_character_,
    resolved_backend = NA_character_,
    implementation_backend = NA_character_,
    requested_backend = NA_character_,
    requested_method = NA_character_,
    exact = NA,
    distance_type = NA_character_,
    input_type = NA_character_,
    input_layout = NA_character_,
    float32_compatibility_conversion = NA,
    tuning_rule = NA_character_,
    hnsw_m = NA_integer_,
    hnsw_ef_construction = NA_integer_,
    hnsw_ef_search = NA_integer_,
    hnsw_graph_degree = NA_integer_,
    hnsw_intermediate_graph_degree = NA_integer_,
    hnsw_ef = NA_integer_,
    tuning_low_dim = NA,
    tuning_high_dim = NA,
    tuning_medium_n = NA,
    tuning_huge_low_dim = NA,
    tuning_runtime_guard = NA,
    error = NA_character_,
    stringsAsFactors = FALSE
  )
}

run_reference_child <- function(config) {
  configure_threads(config$n_threads)
  configure_native_libs()
  library(faissR)
  x <- load_float_dataset(config$dataset_path)
  rows <- config$rows
  started <- proc.time()[["elapsed"]]
  out <- faissR::nn(
    x,
    points = subset_rows(x, rows),
    k = min(as.integer(config$k) + 1L, nrow(x)),
    backend = config$backend,
    method = "exact",
    metric = config$metric,
    output = config$output,
    n_threads = config$n_threads
  )
  ref <- remove_self_from_query_knn(out$indices, out$distances, rows, config$k)
  saveRDS(
    list(
      status = "success",
      backend = config$backend,
      rows = rows,
      indices = ref$indices,
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
  library(faissR)
  x <- load_float_dataset(config$dataset_path)
  row <- empty_result_row(config)
  started <- proc.time()[["elapsed"]]
  out <- faissR::nn(exclude_self = TRUE,
    x,
    k = config$k,
    backend = config$backend,
    method = config$method,
    metric = config$metric,
    target_recall = config$target_recall,
    output = config$output,
    n_threads = config$n_threads
  )
  elapsed <- proc.time()[["elapsed"]] - started
  quality <- recall_summary(out$indices[config$rows, , drop = FALSE], config$reference_indices)
  row$status <- "success"
  row$elapsed_sec <- elapsed
  row$peak_rss_gb <- read_peak_rss_gb()
  row$recall_at_k <- quality$recall_at_k
  row$median_recall_at_k <- quality$median_recall_at_k
  row$min_recall_at_k <- quality$min_recall_at_k
  row$result_backend <- safe_first(out$backend_used)
  row$resolved_backend <- safe_first(attr(out, "resolved_backend"))
  row$implementation_backend <- safe_first(attr(out, "implementation_backend"))
  row$requested_backend <- safe_first(attr(out, "requested_backend"))
  row$requested_method <- safe_first(attr(out, "requested_method"))
  row$exact <- safe_bool(attr(out, "exact") %||% out$exact)
  row$distance_type <- safe_first(attr(out, "distance_type") %||% out$distance_type)
  row$input_type <- safe_first(attr(out, "input_type") %||% out$input_type)
  row$input_layout <- safe_first(attr(out, "input_layout") %||% out$input_layout)
  row$float32_compatibility_conversion <- safe_bool(
    attr(out, "float32_compatibility_conversion") %||% out$float32_compatibility_conversion
  )
  approx <- attr(out, "approximation") %||% list()
  row$target_recall_actual <- suppressWarnings(as.numeric(approx$target_recall %||% attr(out, "target_recall") %||% config$target_recall))
  row$tuning_rule <- safe_first(approx$tuning_rule)
  row$hnsw_m <- suppressWarnings(as.integer(approx$m %||% NA_integer_))
  row$hnsw_ef_construction <- suppressWarnings(as.integer(approx$ef_construction %||% NA_integer_))
  row$hnsw_ef_search <- suppressWarnings(as.integer(approx$ef_search %||% NA_integer_))
  row$hnsw_graph_degree <- suppressWarnings(as.integer(approx$graph_degree %||% NA_integer_))
  row$hnsw_intermediate_graph_degree <- suppressWarnings(as.integer(approx$intermediate_graph_degree %||% NA_integer_))
  row$hnsw_ef <- suppressWarnings(as.integer(approx$ef %||% NA_integer_))
  row$tuning_low_dim <- safe_bool(approx$tuning_low_dim)
  row$tuning_high_dim <- safe_bool(approx$tuning_high_dim)
  row$tuning_medium_n <- safe_bool(approx$tuning_medium_n)
  row$tuning_huge_low_dim <- safe_bool(approx$tuning_huge_low_dim)
  row$tuning_runtime_guard <- safe_bool(approx$tuning_runtime_guard)
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
      row <- empty_result_row(config)
      row$status <- classify_error(conditionMessage(e))
      row$error <- conditionMessage(e)
      row$elapsed_sec <- NA_real_
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
  cfg <- tempfile("faissR_float32_cfg_", fileext = ".rds")
  out <- tempfile("faissR_float32_out_", fileext = ".rds")
  on.exit(unlink(c(cfg, out), force = TRUE), add = TRUE)
  config$output_path <- out
  saveRDS(config, cfg)
  args <- c(
    as.character(as.integer(ceiling(timeout))),
    file.path(R.home("bin"), "Rscript"),
    bench_script,
    paste0("--child_task=", task),
    paste0("--config=", cfg)
  )
  status <- system2("timeout", args = args, stdout = TRUE, stderr = TRUE)
  exit_status <- attr(status, "status") %||% 0L
  timed_out <- identical(as.integer(exit_status), 124L)
  if (file.exists(out)) {
    return(readRDS(out))
  }
  if (identical(task, "method")) {
    row <- empty_result_row(config)
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

make_reference <- function(dataset, dataset_path, n, p, k, metric, n_threads,
                           output, quality_n, seed, timeout, bench_script) {
  set.seed(seed + nchar(dataset) + as.integer(k))
  rows <- if (n <= quality_n) seq_len(n) else sort(sample.int(n, quality_n))
  mode <- if (length(rows) == n) "full_rows" else "sample_rows"
  for (backend in c("cuda", "cpu")) {
    cfg <- list(
      dataset = dataset,
      dataset_path = dataset_path,
      n = n,
      p = p,
      rows = rows,
      k = k,
      metric = metric,
      backend = backend,
      n_threads = n_threads,
      output = output
    )
    ref <- run_rscript_task("reference", cfg, timeout, bench_script)
    if (identical(ref$status, "success") && !is.null(ref$indices)) {
      ref$mode <- mode
      ref$query_n <- length(rows)
      return(ref)
    }
  }
  list(
    status = "unavailable",
    backend = NA_character_,
    rows = rows,
    indices = NULL,
    mode = mode,
    query_n = length(rows),
    elapsed_sec = NA_real_,
    peak_rss_gb = NA_real_,
    error = "No exact reference completed within timeout."
  )
}

write_summary_files <- function(results_path, out_dir) {
  if (!file.exists(results_path)) return(invisible(NULL))
  x <- read.csv(results_path, stringsAsFactors = FALSE)
  ok <- x[x$status == "success", , drop = FALSE]
  if (!nrow(ok)) return(invisible(NULL))
  key <- c("dataset", "backend", "method", "metric", "k", "target_recall_requested")
  parts <- split(ok, interaction(ok[key], drop = TRUE, lex.order = TRUE))
  summary <- do.call(rbind, lapply(parts, function(part) {
    data.frame(
      dataset = part$dataset[[1L]],
      backend = part$backend[[1L]],
      method = part$method[[1L]],
      metric = part$metric[[1L]],
      k = part$k[[1L]],
      target_recall_requested = part$target_recall_requested[[1L]],
      target_recall_actual = part$target_recall_actual[[1L]],
      n = part$n[[1L]],
      p = part$p[[1L]],
      status = "success",
      elapsed_sec = stats::median(part$elapsed_sec, na.rm = TRUE),
      peak_rss_gb = suppressWarnings(max(part$peak_rss_gb, na.rm = TRUE)),
      recall_at_k = stats::median(part$recall_at_k, na.rm = TRUE),
      min_recall_at_k = suppressWarnings(min(part$min_recall_at_k, na.rm = TRUE)),
      reference_backend = part$reference_backend[[1L]],
      reference_mode = part$reference_mode[[1L]],
      reference_query_n = part$reference_query_n[[1L]],
      implementation_backend = part$implementation_backend[[1L]],
      distance_type = part$distance_type[[1L]],
      input_type = part$input_type[[1L]],
      input_layout = part$input_layout[[1L]],
      float32_compatibility_conversion = part$float32_compatibility_conversion[[1L]],
      tuning_rule = part$tuning_rule[[1L]],
      hnsw_m = part$hnsw_m[[1L]],
      hnsw_ef_construction = part$hnsw_ef_construction[[1L]],
      hnsw_ef_search = part$hnsw_ef_search[[1L]],
      hnsw_graph_degree = part$hnsw_graph_degree[[1L]],
      hnsw_intermediate_graph_degree = part$hnsw_intermediate_graph_degree[[1L]],
      hnsw_ef = part$hnsw_ef[[1L]],
      tuning_low_dim = part$tuning_low_dim[[1L]],
      tuning_high_dim = part$tuning_high_dim[[1L]],
      tuning_medium_n = part$tuning_medium_n[[1L]],
      tuning_huge_low_dim = part$tuning_huge_low_dim[[1L]],
      tuning_runtime_guard = part$tuning_runtime_guard[[1L]],
      stringsAsFactors = FALSE
    )
  }))
  row.names(summary) <- NULL
  write.csv(summary, file.path(out_dir, "float32_nn_benchmark_summary.csv"), row.names = FALSE)

  best <- do.call(rbind, lapply(split(ok, paste(ok$dataset, ok$backend, ok$target_recall_requested, sep = "\r")), function(part) {
    has_recall <- is.finite(part$recall_at_k)
    candidates <- if (any(has_recall)) {
      part[has_recall, , drop = FALSE]
    } else {
      part
    }
    ord <- order(
      candidates$elapsed_sec,
      -ifelse(is.finite(candidates$recall_at_k), candidates$recall_at_k, -Inf),
      candidates$method
    )
    candidates[ord[[1L]], , drop = FALSE]
  }))
  row.names(best) <- NULL
  write.csv(best, file.path(out_dir, "float32_nn_benchmark_best_by_dataset_backend.csv"), row.names = FALSE)
  invisible(summary)
}

main <- function() {
  args <- parse_args()
  if (!is.null(args$child_task)) return(run_child_mode(args))

  manifest_path <- normalizePath(args$manifest %||% "float32_dataset_manifest.csv", mustWork = TRUE)
  out_dir <- normalizePath(args$out_dir %||% file.path(getwd(), paste0("faissR_FLOAT32_NN_", format(Sys.time(), "%Y%m%d_%H%M%S"))), mustWork = FALSE)
  dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
  results_path <- file.path(out_dir, "float32_nn_benchmark_results.csv")
  config_path <- file.path(out_dir, "float32_nn_benchmark_config.csv")
  manifest <- read.csv(manifest_path, stringsAsFactors = FALSE)
  manifest <- manifest[manifest$status == "success", , drop = FALSE]
  datasets <- split_arg(args$datasets, paste(manifest$dataset, collapse = ","))
  missing <- setdiff(datasets, manifest$dataset)
  if (length(missing)) stop("Unknown dataset(s): ", paste(missing, collapse = ", "), call. = FALSE)
  manifest <- manifest[match(datasets, manifest$dataset), , drop = FALSE]
  methods <- split_arg(
    args$methods,
    "exact,flat,bruteforce,grid,hnsw,ivf,ivfpq,vamana,nsg,nndescent,cagra"
  )
  methods <- setdiff(unique(methods), "auto")
  backends <- split_arg(args$backends, "cpu,cuda")
  backends <- setdiff(unique(backends), "auto")
  if (!all(backends %in% c("cpu", "cuda"))) {
    stop("This benchmark accepts only explicit `cpu` and `cuda` backends.", call. = FALSE)
  }
  if (!length(backends)) stop("At least one of `cpu` or `cuda` is required.", call. = FALSE)
  k <- positive_int(args$k %||% args$k_values, 50L, "k")
  n_threads <- positive_int(args$threads, 4L, "threads")
  timeout <- positive_num(args$timeout, 600, "timeout")
  quality_n <- positive_int(args$quality_n, 128L, "quality_n")
  seed <- positive_int(args$seed, 20260624L, "seed")
  output <- args$output %||% "float"
  if (!output %in% c("float", "double")) stop("`output` must be `float` or `double`.", call. = FALSE)
  metric <- "euclidean"
  target_recall <- suppressWarnings(as.numeric(args$target_recall %||% 0.99))
  if (length(target_recall) != 1L || is.na(target_recall) || !is.finite(target_recall) ||
      !any(abs(target_recall - c(0.90, 0.95, 0.99)) < 1e-8)) {
    stop("`target_recall` must be one of 0.9, 0.95, or 0.99.", call. = FALSE)
  }

  write.csv(
    data.frame(
      key = c("manifest", "out_dir", "datasets", "methods", "backends", "metric", "k",
              "target_recall", "threads", "timeout", "quality_n", "seed", "output"),
      value = c(manifest_path, out_dir, paste(datasets, collapse = ","), paste(methods, collapse = ","),
                paste(backends, collapse = ","), metric, k, target_recall, n_threads, timeout, quality_n, seed, output),
      stringsAsFactors = FALSE
    ),
    config_path,
    row.names = FALSE
  )

  bench_script <- script_path()
  for (i in seq_len(nrow(manifest))) {
    ds <- manifest[i, , drop = FALSE]
    dataset <- ds$dataset[[1L]]
    dataset_path <- ds$output[[1L]]
    n <- as.integer(ds$n[[1L]])
    p <- as.integer(ds$p[[1L]])
    message(sprintf("[%s] loading reference rows for %s (%s x %s)", Sys.time(), dataset, n, p))
    reference <- make_reference(
      dataset = dataset,
      dataset_path = dataset_path,
      n = n,
      p = p,
      k = k,
      metric = metric,
      n_threads = n_threads,
      output = output,
      quality_n = quality_n,
      seed = seed,
      timeout = timeout,
      bench_script = bench_script
    )
    message(sprintf(
      "[%s] reference for %s: status=%s backend=%s rows=%s",
      Sys.time(), dataset, reference$status, reference$backend %||% NA_character_, reference$query_n
    ))
    for (backend in backends) {
      for (method in methods) {
        cfg <- list(
          dataset = dataset,
          dataset_path = dataset_path,
          n = n,
          p = p,
          rows = reference$rows,
          reference_indices = reference$indices,
          reference_status = reference$status,
          reference_backend = reference$backend,
          reference_mode = reference$mode,
          reference_query_n = reference$query_n,
          backend = backend,
          method = method,
          metric = metric,
          k = k,
          target_recall = target_recall,
          n_threads = n_threads,
          output = output
        )
        message(sprintf("[%s] %s backend=%s method=%s", Sys.time(), dataset, backend, method))
        row <- run_rscript_task("method", cfg, timeout, bench_script)
        append_csv(row, results_path)
        message(sprintf(
          "[%s] -> status=%s elapsed=%s recall=%s",
          Sys.time(), row$status[[1L]], format(row$elapsed_sec[[1L]], digits = 4),
          format(row$recall_at_k[[1L]], digits = 4)
        ))
        write_summary_files(results_path, out_dir)
      }
    }
  }
  write_summary_files(results_path, out_dir)
  message("DONE: ", out_dir)
}

main()
