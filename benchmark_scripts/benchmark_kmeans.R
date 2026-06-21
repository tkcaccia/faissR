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

as_int_arg <- function(x, default) {
  value <- suppressWarnings(as.integer(x %||% default))
  if (length(value) != 1L || is.na(value) || value < 1L) as.integer(default) else value
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
  if (requireNamespace("RhpcBLASctl", quietly = TRUE)) {
    try(RhpcBLASctl::blas_set_num_threads(as.integer(n_threads)), silent = TRUE)
    try(RhpcBLASctl::omp_set_num_threads(as.integer(n_threads)), silent = TRUE)
  }
}

configure_native_libs <- function() {
  env_dir <- Sys.getenv("FAISSR_ENV_DIR", Sys.getenv("CONDA_PREFIX", unset = ""))
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

label_center_count <- function(labels, fallback) {
  if (is.null(labels)) return(as.integer(fallback))
  labels <- labels[!is.na(labels)]
  n <- length(unique(labels))
  if (!is.finite(n) || n < 1L) as.integer(fallback) else as.integer(n)
}

kmeans_auto_params <- function(n, p, centers, tuning = "auto") {
  if (!identical(tuning, "auto")) {
    return(list(max_iter = 100L, n_init = 1L, tol = 1e-4, policy = tuning))
  }
  work <- as.double(n) * as.double(p) * as.double(centers)
  n_per_center <- as.double(n) / as.double(centers)
  small_many_centers <- centers >= 100L && n <= 50000L && work <= 2e8 && n_per_center >= 20
  max_iter <- if (n >= 100000L || work >= 5e9) {
    50L
  } else if (p >= 256L || (centers >= 100L && !small_many_centers) || work >= 5e8) {
    75L
  } else {
    100L
  }
  n_init <- if (n <= 50000L && centers <= 20L && work <= 2e8) {
    5L
  } else if (small_many_centers) {
    3L
  } else if (n <= 100000L && centers <= 50L && work <= 5e8) {
    3L
  } else {
    1L
  }
  tol <- if (n >= 100000L || work >= 5e9) 1e-3 else 1e-4
  list(max_iter = as.integer(max_iter), n_init = as.integer(n_init), tol = as.numeric(tol), policy = "auto")
}

resolve_kmeans_int <- function(x, fallback) {
  if (is.character(x) && length(x) == 1L && identical(tolower(x), "auto")) return(as.integer(fallback))
  out <- suppressWarnings(as.integer(x))
  if (length(out) != 1L || is.na(out) || out < 1L) as.integer(fallback) else out
}

resolve_kmeans_tol <- function(x, fallback) {
  if (is.character(x) && length(x) == 1L && identical(tolower(x), "auto")) return(as.numeric(fallback))
  out <- suppressWarnings(as.numeric(x))
  if (length(out) != 1L || is.na(out) || !is.finite(out) || out < 0) as.numeric(fallback) else out
}

result_row <- function(dataset, n, p, method, backend, centers, n_threads,
                       status, error = NA_character_, elapsed_sec = NA_real_,
                       peak_rss_gb = NA_real_, backend_used = NA_character_,
                       iter = NA_integer_, tot_withinss = NA_real_,
                       ari = NA_real_, max_iter = NA_integer_,
                       n_init = NA_integer_, tol = NA_real_,
                       tuning_policy = NA_character_,
                       expected_skip = FALSE) {
  data.frame(
    dataset = dataset,
    n = as.integer(n),
    p = as.integer(p),
    method = method,
    backend = backend,
    centers = as.integer(centers),
    n_threads = as.integer(n_threads),
    status = status,
    error = error,
    elapsed_sec = elapsed_sec,
    peak_rss_gb = peak_rss_gb,
    backend_used = backend_used,
    iter = as.integer(iter),
    tot_withinss = tot_withinss,
    ari = ari,
    max_iter = as.integer(max_iter),
    n_init = as.integer(n_init),
    tol = tol,
    tuning_policy = tuning_policy,
    expected_skip = isTRUE(expected_skip),
    stringsAsFactors = FALSE
  )
}

kmeans_expected_skip <- function(method, backend) {
  method <- tolower(as.character(method)[1L])
  backend <- tolower(as.character(backend)[1L])
  if (!identical(method, "fast_kmeans")) return(NULL)
  if (backend %in% c("cuda", "cuda_faiss", "faiss_gpu", "cuda_cuvs", "cuvs") &&
      !isTRUE(faissR::cuda_available()) &&
      !isTRUE(faissR::cuvs_available())) {
    return("CUDA k-means is unavailable in this faissR build/runtime; explicit CUDA requests are expected skips.")
  }
  NULL
}

run_one <- function(x, labels, dataset_name, method, backend, centers,
                    n_threads, seed, timeout, max_iter, n_init, tol, tuning) {
  started <- proc.time()[["elapsed"]]
  tryCatch({
    auto_params <- kmeans_auto_params(nrow(x), ncol(x), centers, tuning)
    resolved_max_iter <- resolve_kmeans_int(max_iter, auto_params$max_iter)
    resolved_n_init <- resolve_kmeans_int(n_init, auto_params$n_init)
    resolved_tol <- resolve_kmeans_tol(tol, auto_params$tol)
    fit <- with_elapsed_limit({
      if (identical(method, "stats")) {
        stats::kmeans(
          x,
          centers = centers,
          iter.max = resolved_max_iter,
          nstart = resolved_n_init,
          algorithm = "Lloyd"
        )
      } else {
        faissR::fast_kmeans(
          x,
          centers = centers,
          backend = backend,
          max_iter = max_iter,
          n_init = n_init,
          tol = tol,
          seed = seed,
          n_threads = n_threads,
          tuning = tuning
        )
      }
    }, timeout)
    elapsed <- proc.time()[["elapsed"]] - started
    params <- fit$parameters %||% list()
    result_row(
      dataset = dataset_name,
      n = nrow(x),
      p = ncol(x),
      method = method,
      backend = backend,
      centers = centers,
      n_threads = n_threads,
      status = "success",
      elapsed_sec = elapsed,
      peak_rss_gb = read_peak_rss_gb(),
      backend_used = fit$backend %||% if (identical(method, "stats")) "stats" else NA_character_,
      iter = fit$iter %||% NA_integer_,
      tot_withinss = fit$tot.withinss %||% NA_real_,
      ari = benchmark_adjusted_rand_index(labels, fit$cluster),
      max_iter = params$max_iter %||% resolved_max_iter,
      n_init = params$n_init %||% resolved_n_init,
      tol = params$tol %||% resolved_tol,
      tuning_policy = params$tuning$policy %||% if (identical(method, "stats")) "stats" else NA_character_
    )
  }, error = function(e) {
    result_row(
      dataset = dataset_name,
      n = nrow(x),
      p = ncol(x),
      method = method,
      backend = backend,
      centers = centers,
      n_threads = n_threads,
      status = "failed",
      error = conditionMessage(e),
      elapsed_sec = proc.time()[["elapsed"]] - started,
      peak_rss_gb = read_peak_rss_gb()
    )
  })
}

args <- parse_args()
configure_native_libs()

cmd_args <- commandArgs(FALSE)
file_arg <- cmd_args[grep("^--file=", cmd_args)[1L]] %||% "benchmark_scripts/benchmark_kmeans.R"
script_dir <- dirname(normalizePath(sub("^--file=", "", file_arg), mustWork = FALSE))
helper <- file.path(script_dir, "source.R")
if (!file.exists(helper)) helper <- file.path(getwd(), "benchmark_scripts/source.R")
source(helper)

data_root <- args$data_root %||% Sys.getenv("FAISSR_BENCHMARK_DATA", unset = file.path(getwd(), "Data"))
out_dir <- args$out_dir %||% Sys.getenv("FAISSR_BENCHMARK_OUT", unset = file.path(getwd(), paste0("faissR_KMEANS_", format(Sys.time(), "%Y%m%d_%H%M%S"))))
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

n_threads <- as_int_arg(args$threads, 4L)
configure_threads(n_threads)
seed <- as_int_arg(args$seed, 1L)
timeout <- as_int_arg(args$timeout, 600L)
fallback_centers <- as_int_arg(args$centers, 10L)
max_iter <- args$max_iter %||% "auto"
n_init <- args$n_init %||% "auto"
tol <- args$tol %||% "auto"
tuning <- args$tuning %||% "auto"

datasets <- split_arg(args$datasets, paste(c(dataset_index(data_root)$dataset, "SimulatedTiny3Clusters"), collapse = ","))
methods <- split_arg(args$methods, "fast_kmeans,stats")
backends <- split_arg(args$backends, "cpu,cuda")

suppressPackageStartupMessages(library(faissR))

config <- data.frame(
  key = c("data_root", "out_dir", "datasets", "methods", "backends", "centers",
          "threads", "timeout", "max_iter", "n_init", "tol", "tuning", "seed"),
  value = c(data_root, out_dir, paste(datasets, collapse = ","), paste(methods, collapse = ","),
            paste(backends, collapse = ","), fallback_centers, n_threads, timeout,
            max_iter, n_init, tol, tuning, seed),
  stringsAsFactors = FALSE
)
utils::write.csv(config, file.path(out_dir, "kmeans_benchmark_config.csv"), row.names = FALSE)

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
      method = NA_character_,
      backend = NA_character_,
      centers = fallback_centers,
      n_threads = n_threads,
      status = "failed",
      error = conditionMessage(loaded)
    )
    next
  }
  x <- loaded$data
  centers <- label_center_count(loaded$labels, fallback_centers)
  for (method in methods) {
    method_backends <- if (identical(method, "stats")) "stats" else backends
    for (backend in method_backends) {
      row_id <- row_id + 1L
      skip_reason <- kmeans_expected_skip(method, backend)
      if (!is.null(skip_reason)) {
        auto_params <- kmeans_auto_params(nrow(x), ncol(x), centers, tuning)
        row <- result_row(
          dataset = dataset_name,
          n = nrow(x),
          p = ncol(x),
          method = method,
          backend = backend,
          centers = centers,
          n_threads = n_threads,
          status = "expected_skip",
          error = skip_reason,
          max_iter = resolve_kmeans_int(max_iter, auto_params$max_iter),
          n_init = resolve_kmeans_int(n_init, auto_params$n_init),
          tol = resolve_kmeans_tol(tol, auto_params$tol),
          tuning_policy = auto_params$policy,
          expected_skip = TRUE
        )
      } else {
        row <- run_one(
          x = x,
          labels = loaded$labels,
          dataset_name = dataset_name,
          method = method,
          backend = backend,
          centers = centers,
          n_threads = n_threads,
          seed = seed,
          timeout = timeout,
          max_iter = max_iter,
          n_init = n_init,
          tol = tol,
          tuning = tuning
        )
      }
      results[[row_id]] <- row
      utils::write.csv(do.call(rbind, results), file.path(out_dir, "kmeans_benchmark_results.csv"), row.names = FALSE)
      cat(sprintf(
        "[%s] dataset=%s method=%s backend=%s centers=%s status=%s elapsed=%.3f\n",
        format(Sys.time(), "%Y-%m-%d %H:%M:%S"),
        dataset_name, method, backend, centers, row$status, row$elapsed_sec
      ))
      flush.console()
    }
  }
  rm(x, loaded)
  gc()
}

results_df <- do.call(rbind, results)
utils::write.csv(results_df, file.path(out_dir, "kmeans_benchmark_results.csv"), row.names = FALSE)

ok <- results_df[results_df$status == "success", , drop = FALSE]
if (nrow(ok)) {
  ok$quality_score <- ifelse(is.na(ok$ari), -Inf, ok$ari)
  best <- ok[order(ok$dataset, -ok$quality_score, ok$elapsed_sec), , drop = FALSE]
  best <- do.call(rbind, lapply(split(best, best$dataset), function(x) x[1L, , drop = FALSE]))
  best$quality_score <- NULL
  utils::write.csv(best, file.path(out_dir, "kmeans_best_by_dataset.csv"), row.names = FALSE)

  fast_rows <- ok[ok$method == "fast_kmeans", , drop = FALSE]
  stats_rows <- ok[ok$method == "stats", , drop = FALSE]
  if (nrow(fast_rows) && nrow(stats_rows)) {
    comparison <- merge(
      fast_rows,
      stats_rows[, c("dataset", "centers", "elapsed_sec", "tot_withinss", "ari", "iter"), drop = FALSE],
      by = c("dataset", "centers"),
      suffixes = c("_fast", "_stats"),
      all = FALSE
    )
    if (nrow(comparison)) {
      comparison$speedup_vs_stats <- ifelse(
        comparison$elapsed_sec_fast > 0,
        comparison$elapsed_sec_stats / comparison$elapsed_sec_fast,
        ifelse(comparison$elapsed_sec_stats == 0, 1, Inf)
      )
      comparison$ari_delta_vs_stats <- comparison$ari_fast - comparison$ari_stats
      comparison$withinss_ratio_vs_stats <- comparison$tot_withinss_fast / comparison$tot_withinss_stats
      comparison <- comparison[order(comparison$dataset, comparison$backend), , drop = FALSE]
      utils::write.csv(
        comparison,
        file.path(out_dir, "kmeans_fast_vs_stats.csv"),
        row.names = FALSE
      )
    }
  }
}

materials <- c(
  "# K-Means Benchmark",
  "",
  "This benchmark compares faissR `fast_kmeans()` backends with base `stats::kmeans`.",
  "",
  sprintf("- Output directory: `%s`", out_dir),
  sprintf("- Data root: `%s`", data_root),
  sprintf("- Methods: `%s`", paste(methods, collapse = "`, `")),
  sprintf("- Backends: `%s`", paste(backends, collapse = "`, `")),
  sprintf("- CPU thread cap: `%s`", n_threads),
  sprintf("- Timeout per combination: `%s` seconds", timeout),
  sprintf("- Requested centers fallback: `%s`; labels override this when available", fallback_centers),
  "",
  "The result table records elapsed time, peak resident memory when available, backend used, total within-cluster sum of squares, iterations, selected k-means parameters, tuning policy, and ARI against dataset labels when labels are available.",
  "`kmeans_fast_vs_stats.csv` compares successful `fast_kmeans()` rows with successful `stats::kmeans` rows for the same dataset and number of centers, recording speedup, ARI delta, and withinss ratio.",
  "Unsupported CUDA or library combinations known before execution are recorded as `status = \"expected_skip\"` with `expected_skip = TRUE`. Unexpected runtime errors remain failed rows rather than being replaced with CPU timings."
)
writeLines(materials, file.path(out_dir, "MATERIALS_AND_METHODS_kmeans.md"))

cat("Saved k-means benchmark files in: ", out_dir, "\n", sep = "")
