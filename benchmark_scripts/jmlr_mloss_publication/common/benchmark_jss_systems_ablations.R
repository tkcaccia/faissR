#!/usr/bin/env Rscript

`%||%` <- function(x, y) {
  if (is.null(x) || !length(x) || is.na(x[[1L]])) y else x
}

parse_args <- function(args = commandArgs(trailingOnly = TRUE)) {
  out <- list()
  for (arg in args) {
    if (!startsWith(arg, "--")) next
    parts <- strsplit(sub("^--", "", arg), "=", fixed = TRUE)[[1L]]
    out[[parts[[1L]]]] <- if (length(parts) > 1L) paste(parts[-1L], collapse = "=") else "TRUE"
  }
  out
}

split_values <- function(x, default) trimws(strsplit(x %||% default, ",", fixed = TRUE)[[1L]])

logical_value <- function(x, default = FALSE) {
  if (is.null(x)) return(default)
  tolower(x) %in% c("true", "t", "1", "yes")
}

find_data <- function(path) {
  env <- new.env(parent = emptyenv())
  load(path, envir = env)
  if (exists("dataset", env, inherits = FALSE)) {
    value <- get("dataset", env)
    if (is.list(value) && !is.null(value$data)) return(value$data)
  }
  for (name in ls(env)) {
    value <- get(name, env)
    if (is.list(value) && !is.null(value$data)) return(value$data)
  }
  stop("No list object containing `$data` was found in ", path, call. = FALSE)
}

is_float32 <- function(x) inherits(x, "float32") || inherits(x, "float")

prepare_input <- function(x, input_type) {
  if (input_type == "float32") {
    if (!requireNamespace("float", quietly = TRUE)) stop("The float package is required.", call. = FALSE)
    if (is_float32(x)) return(x)
    return(float::fl(as.matrix(x)))
  }
  if (is_float32(x)) {
    if (!requireNamespace("float", quietly = TRUE)) stop("The float package is required.", call. = FALSE)
    x <- float::dbl(x)
  }
  if (!is.matrix(x)) x <- as.matrix(x)
  storage.mode(x) <- "double"
  x
}

clear_faissr_caches <- function() {
  ns <- asNamespace("faissR")
  for (name in c(
    ".faissR_fitted_nn_index_cache",
    ".faissR_cuvs_ivfpq_index_cache",
    ".faissR_transformed_float32_cache"
  )) {
    if (!exists(name, ns, inherits = FALSE)) next
    env <- get(name, ns)
    objects <- setdiff(ls(env, all.names = TRUE), ".keys")
    if (length(objects)) rm(list = objects, envir = env)
    env$.keys <- character()
  }
  invisible(gc())
}

peak_rss_gb <- function() {
  path <- "/proc/self/status"
  if (!file.exists(path)) return(NA_real_)
  line <- grep("^VmHWM:", readLines(path, warn = FALSE), value = TRUE)
  if (!length(line)) return(NA_real_)
  as.numeric(gsub("[^0-9]", "", line[[1L]])) / 1024^2
}

result_meta <- function(x) {
  approx <- attr(x, "approximation", exact = TRUE) %||% list()
  faiss <- attr(x, "faiss", exact = TRUE) %||% list()
  cuvs <- attr(x, "cuvs", exact = TRUE) %||% list()
  data.frame(
    resolved_backend = x$backend_used %||% attr(x, "backend_used") %||% NA_character_,
    resolved_method = x$method %||% attr(x, "method") %||% NA_character_,
    input_type_reported = x$input_type %||% attr(x, "input_type") %||% NA_character_,
    index_cache_hit = approx$index_cache_hit %||% faiss$index_cache_hit %||%
      cuvs$index_cache_hit %||% NA,
    persistent_index_cache = approx$persistent_index_cache %||%
      faiss$persistent_index_cache %||% cuvs$persistent_index_cache %||% NA,
    result_residency = x$result_residency %||% attr(x, "result_residency") %||% NA_character_,
    device_to_host_result_copies = x$device_to_host_result_copies %||%
      attr(x, "device_to_host_result_copies") %||% NA,
    stringsAsFactors = FALSE
  )
}

signature <- function(x) {
  if (inherits(x, "faissR_gpu_knn") || is.null(x$indices)) return(NA_real_)
  idx <- as.integer(x$indices)
  sum(head(idx, 1000L), na.rm = TRUE)
}

run_nn <- function(x, backend, method, metric, k, threads, exclude_self = TRUE) {
  faissR::nn(
    x, k = k, exclude_self = exclude_self,
    backend = backend, method = method, metric = metric,
    tuning = "auto", target_recall = 0.99,
    output = "double", n_threads = threads
  )
}

r_remove_self <- function(result, k) {
  idx <- as.matrix(result$indices)
  dst <- as.matrix(result$distances)
  out_i <- matrix(NA_integer_, nrow(idx), k)
  out_d <- matrix(NA_real_, nrow(idx), k)
  for (i in seq_len(nrow(idx))) {
    keep <- which(!is.na(idx[i, ]) & idx[i, ] != i)
    take <- head(keep, k)
    if (length(take)) {
      out_i[i, seq_along(take)] <- idx[i, take]
      out_d[i, seq_along(take)] <- dst[i, take]
    }
  }
  list(indices = out_i, distances = out_d)
}

base_row <- function(args, x) data.frame(
  dataset = args$dataset,
  data_path = args$data_path,
  n = nrow(x), p = ncol(x),
  backend = args$backend,
  method = args$method,
  metric = args$metric,
  k = as.integer(args$k),
  input_type = args$input_type,
  experiment = args$experiment,
  phase = NA_character_,
  repeat_id = NA_integer_,
  elapsed_sec = NA_real_,
  host_copy_sec = NA_real_,
  r_postprocess_sec = NA_real_,
  peak_rss_gb = NA_real_,
  result_signature = NA_real_,
  status = "failed",
  error = "",
  stringsAsFactors = FALSE
)

timed_call <- function(expr) {
  start <- proc.time()[["elapsed"]]
  value <- force(expr)
  list(value = value, elapsed = proc.time()[["elapsed"]] - start)
}

worker_input_cache <- function(x, args) {
  rows <- list()
  threads <- as.integer(args$threads)
  repeats <- as.integer(args$repeats)
  call <- function(phase, repeat_id) {
    row <- base_row(args, x)
    row$phase <- phase
    row$repeat_id <- repeat_id
    tryCatch({
      measured <- timed_call(run_nn(x, args$backend, args$method, args$metric,
                                    as.integer(args$k), threads, TRUE))
      row$elapsed_sec <- measured$elapsed
      row$result_signature <- signature(measured$value)
      row$peak_rss_gb <- peak_rss_gb()
      row$status <- "success"
      cbind(row, result_meta(measured$value))
    }, error = function(e) {
      row$error <- conditionMessage(e)
      row$peak_rss_gb <- peak_rss_gb()
      cbind(row, result_meta(list()))
    })
  }

  old <- options(
    faissR.cache_fitted_nn_indexes = FALSE,
    faissR.cache_transformed_float32 = FALSE
  )
  on.exit(options(old), add = TRUE)
  clear_faissr_caches()
  for (i in seq_len(repeats)) {
    clear_faissr_caches()
    rows[[length(rows) + 1L]] <- call("cache_disabled", i)
  }
  options(faissR.cache_fitted_nn_indexes = TRUE, faissR.cache_transformed_float32 = TRUE)
  clear_faissr_caches()
  rows[[length(rows) + 1L]] <- call("cache_enabled_cold", 1L)
  for (i in seq_len(repeats)) rows[[length(rows) + 1L]] <- call("cache_enabled_warm", i)
  do.call(rbind, rows)
}

worker_self_processing <- function(x, args) {
  rows <- list()
  threads <- as.integer(args$threads)
  k <- as.integer(args$k)
  old <- options(faissR.cache_fitted_nn_indexes = FALSE)
  on.exit(options(old), add = TRUE)

  clear_faissr_caches()
  compiled <- base_row(args, x)
  compiled$phase <- "compiled_self_removal"
  rows[[1L]] <- tryCatch({
    measured <- timed_call(run_nn(x, args$backend, args$method, args$metric, k, threads, TRUE))
    compiled$elapsed_sec <- measured$elapsed
    compiled$result_signature <- signature(measured$value)
    compiled$peak_rss_gb <- peak_rss_gb()
    compiled$status <- "success"
    cbind(compiled, result_meta(measured$value))
  }, error = function(e) {
    compiled$error <- conditionMessage(e)
    cbind(compiled, result_meta(list()))
  })

  clear_faissr_caches()
  r_side <- base_row(args, x)
  r_side$phase <- "r_self_removal"
  rows[[2L]] <- tryCatch({
    search <- timed_call(run_nn(x, args$backend, args$method, args$metric, k + 1L, threads, FALSE))
    processed <- timed_call(r_remove_self(search$value, k))
    r_side$elapsed_sec <- search$elapsed + processed$elapsed
    r_side$r_postprocess_sec <- processed$elapsed
    r_side$result_signature <- signature(processed$value)
    r_side$peak_rss_gb <- peak_rss_gb()
    r_side$status <- "success"
    cbind(r_side, result_meta(search$value))
  }, error = function(e) {
    r_side$error <- conditionMessage(e)
    cbind(r_side, result_meta(list()))
  })
  do.call(rbind, rows)
}

worker_gpu_residency <- function(x, args) {
  row <- base_row(args, x)
  row$phase <- "gpu_resident_then_explicit_host_copy"
  tryCatch({
    search <- timed_call(faissR::nn_gpu(
      x, k = as.integer(args$k), exclude_self = TRUE,
      method = args$method, metric = args$metric,
      tuning = "auto", target_recall = 0.99
    ))
    copied <- timed_call(faissR::gpu_knn_to_host(search$value))
    row$elapsed_sec <- search$elapsed
    row$host_copy_sec <- copied$elapsed
    row$result_signature <- signature(copied$value)
    row$peak_rss_gb <- peak_rss_gb()
    row$status <- "success"
    cbind(row, result_meta(search$value))
  }, error = function(e) {
    row$error <- conditionMessage(e)
    row$peak_rss_gb <- peak_rss_gb()
    cbind(row, result_meta(list()))
  })
}

worker_main <- function(args) {
  tryCatch(
    loadNamespace("faissR"),
    error = function(e) {
      stop("faissR cannot be loaded: ", conditionMessage(e), call. = FALSE)
    }
  )
  threads <- as.integer(args$threads)
  value <- as.character(threads)
  Sys.setenv(OMP_NUM_THREADS = value, OPENBLAS_NUM_THREADS = value,
             MKL_NUM_THREADS = value, RCPP_PARALLEL_NUM_THREADS = value)
  x <- prepare_input(find_data(args$data_path), args$input_type)
  result <- switch(
    args$experiment,
    input_cache = worker_input_cache(x, args),
    self_processing = worker_self_processing(x, args),
    gpu_residency = worker_gpu_residency(x, args),
    stop("Unknown experiment: ", args$experiment, call. = FALSE)
  )
  write.csv(result, args$result_path, row.names = FALSE)
}

read_union <- function(files) {
  rows <- lapply(files, read.csv, stringsAsFactors = FALSE)
  columns <- unique(unlist(lapply(rows, names), use.names = FALSE))
  rows <- lapply(rows, function(x) {
    for (name in setdiff(columns, names(x))) x[[name]] <- NA
    x[, columns, drop = FALSE]
  })
  do.call(rbind, rows)
}

driver_main <- function(args) {
  backend <- tolower(args$backend %||% "cpu")
  if (!backend %in% c("cpu", "cuda")) stop("`backend` must be cpu or cuda.", call. = FALSE)
  manifest <- read.csv(normalizePath(args$manifest, mustWork = TRUE), stringsAsFactors = FALSE)
  if (!"path" %in% names(manifest) && "output" %in% names(manifest)) manifest$path <- manifest$output
  datasets <- split_values(args$datasets, "COIL20,MNIST,TabulaMuris")
  manifest <- manifest[manifest$dataset %in% datasets & file.exists(manifest$path), , drop = FALSE]
  if (!nrow(manifest)) stop("No selected datasets are available.", call. = FALSE)
  methods <- split_values(args$methods, if (backend == "cpu") "flat,hnsw,ivf" else "flat,cagra,hnsw,ivf")
  input_types <- split_values(args$input_types, "float32,double")
  out_dir <- normalizePath(args$out_dir %||% file.path(getwd(), paste0("faissR_JSS_ABLATION_", toupper(backend))), mustWork = FALSE)
  dir.create(file.path(out_dir, "worker_results"), recursive = TRUE, showWarnings = FALSE)
  dir.create(file.path(out_dir, "worker_logs"), recursive = TRUE, showWarnings = FALSE)
  script <- normalizePath(sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE)[[1L]]), mustWork = TRUE)
  threads <- as.integer(args$threads %||% if (backend == "cpu") 12L else 2L)
  timeout <- as.integer(args$timeout %||% 2000L)
  repeats <- as.integer(args$repeats %||% 3L)
  k <- as.integer(args$k %||% 30L)
  metric <- args$metric %||% "euclidean"
  total <- 0L

  jobs <- list()
  for (di in seq_len(nrow(manifest))) {
    for (method in methods) {
      for (input_type in input_types) {
        jobs[[length(jobs) + 1L]] <- c(dataset = manifest$dataset[[di]], data_path = manifest$path[[di]],
          method = method, input_type = input_type, experiment = "input_cache")
      }
    }
    for (input_type in input_types) {
      jobs[[length(jobs) + 1L]] <- c(dataset = manifest$dataset[[di]], data_path = manifest$path[[di]],
        method = "flat", input_type = input_type, experiment = "self_processing")
      if (backend == "cuda") jobs[[length(jobs) + 1L]] <- c(
        dataset = manifest$dataset[[di]], data_path = manifest$path[[di]],
        method = "exact", input_type = input_type, experiment = "gpu_residency"
      )
    }
  }

  for (job in jobs) {
    total <- total + 1L
    result_path <- file.path(out_dir, "worker_results", sprintf(
      "%03d_%s_%s_%s_%s.csv", total, job[["dataset"]], job[["method"]],
      job[["input_type"]], job[["experiment"]]
    ))
    command <- c(
      as.character(timeout), "Rscript", script, "--worker=TRUE",
      paste0("--result_path=", result_path), paste0("--dataset=", job[["dataset"]]),
      paste0("--data_path=", job[["data_path"]]), paste0("--backend=", backend),
      paste0("--method=", job[["method"]]), paste0("--input_type=", job[["input_type"]]),
      paste0("--experiment=", job[["experiment"]]), paste0("--threads=", threads),
      paste0("--repeats=", repeats), paste0("--k=", k), paste0("--metric=", metric)
    )
    log_stem <- sub("[.]csv$", "", basename(result_path))
    stdout_path <- file.path(out_dir, "worker_logs", paste0(log_stem, ".out"))
    stderr_path <- file.path(out_dir, "worker_logs", paste0(log_stem, ".err"))
    status <- system2("timeout", command, stdout = stdout_path, stderr = stderr_path)
    if (!file.exists(result_path)) {
      detail <- if (file.exists(stderr_path)) {
        paste(tail(readLines(stderr_path, warn = FALSE), 20L), collapse = " | ")
      } else {
        ""
      }
      write.csv(data.frame(
        dataset = job[["dataset"]], backend = backend, method = job[["method"]],
        input_type = job[["input_type"]], experiment = job[["experiment"]],
        status = if (identical(status, 124L)) "timeout" else "failed",
        error = paste0("worker exit status ", status,
                       if (nzchar(detail)) paste0(": ", detail) else ""),
        stdout_path = stdout_path, stderr_path = stderr_path,
        stringsAsFactors = FALSE
      ), result_path, row.names = FALSE)
    }
  }
  files <- list.files(file.path(out_dir, "worker_results"), pattern = "[.]csv$", full.names = TRUE)
  result <- read_union(files)
  write.csv(result, file.path(out_dir, "jss_systems_ablation_results.csv"), row.names = FALSE)
  write.csv(result[result$status != "success", , drop = FALSE], file.path(out_dir, "jss_systems_ablation_failures.csv"), row.names = FALSE)
  writeLines(capture.output(sessionInfo()), file.path(out_dir, "sessionInfo.txt"))
  cat("Wrote systems ablations to ", out_dir, "\n", sep = "")
}

args <- parse_args()
if (logical_value(args$worker, FALSE)) {
  worker_main(args)
} else if (!identical(Sys.getenv("FAISSR_JSS_ABLATION_SOURCE_ONLY", unset = ""), "true")) {
  driver_main(args)
}
