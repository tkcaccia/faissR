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

as_int_vec_arg <- function(x, default) {
  value <- suppressWarnings(as.integer(x %||% default))
  value <- value[!is.na(value) & value > 0L]
  if (!length(value)) suppressWarnings(as.integer(default)) else unique(value)
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
  if (identical(name, "SimulatedUniform2D")) {
    return(list(data = matrix(runif(2000000), ncol = 2), labels = NULL))
  }
  if (identical(name, "SimulatedUniform3D")) {
    return(list(data = matrix(runif(3000000), ncol = 3), labels = NULL))
  }
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

metric_reference <- function(x, k, metric, quality_n, quality_max_ops, n_threads, seed) {
  n <- nrow(x)
  p <- ncol(x)
  full_ops <- as.double(n) * as.double(n) * as.double(p)
  if (n <= quality_n && full_ops <= quality_max_ops) {
    return(list(
      knn = faissR::nn(
        x,
        k = k,
        backend = "cpu",
        method = "exact",
        metric = metric,
        n_threads = n_threads
      ),
      rows = NULL,
      mode = "full"
    ))
  }
  sample_n <- min(as.integer(quality_n), n)
  sample_ops <- as.double(sample_n) * as.double(n) * as.double(p)
  if (sample_n < 1L || sample_ops > quality_max_ops) return(NULL)
  set.seed(seed + as.integer(k) + match(metric, c("euclidean", "cosine", "correlation", "inner_product")))
  rows <- sort(sample.int(n, sample_n))
  list(
    knn = faissR::nn(
      x,
      points = x[rows, , drop = FALSE],
      k = k,
      backend = "cpu",
      method = "exact",
      metric = metric,
      n_threads = n_threads
    ),
    rows = rows,
    mode = "sample"
  )
}

result_row <- function(dataset, n, p, backend, method, metric, k, cycle, n_threads,
                       status, error = NA_character_, elapsed_sec = NA_real_,
                       peak_rss_gb = NA_real_,
                       result_backend = NA_character_,
                       resolved_backend = NA_character_,
                       implementation_backend = NA_character_,
                       exact = NA, recall_at_k = NA_real_,
                       median_recall_at_k = NA_real_,
                       min_recall_at_k = NA_real_,
                       recall_reference = NA_character_,
                       recall_query_n = NA_integer_,
                       expected_skip = FALSE,
                       capability_notes = NA_character_) {
  data.frame(
    dataset = dataset,
    n = as.integer(n),
    p = as.integer(p),
    backend = backend,
    method = method,
    metric = metric,
    k = as.integer(k),
    cycle = as.integer(cycle),
    n_threads = as.integer(n_threads),
    status = status,
    error = error,
    elapsed_sec = elapsed_sec,
    peak_rss_gb = peak_rss_gb,
    result_backend = result_backend,
    resolved_backend = resolved_backend,
    implementation_backend = implementation_backend,
    exact = exact,
    recall_at_k = recall_at_k,
    median_recall_at_k = median_recall_at_k,
    min_recall_at_k = min_recall_at_k,
    recall_reference = recall_reference,
    recall_query_n = as.integer(recall_query_n),
    expected_skip = isTRUE(expected_skip),
    capability_notes = capability_notes,
    stringsAsFactors = FALSE
  )
}

nn_implementation_backend <- function(out) {
  approx <- attr(out, "approximation") %||% list()
  cuvs <- attr(out, "cuvs") %||% list()
  faiss <- attr(out, "faiss") %||% list()
  attr(out, "resolved_backend") %||%
    approx$backend %||%
    cuvs$resolved_backend %||%
    faiss$backend %||%
    attr(out, "backend") %||%
    NA_character_
}

canonical_method_key <- function(method) {
  key <- tolower(gsub("[[:space:]_-]+", "", as.character(method)))
  aliases <- c(
    auto = "auto",
    exact = "exact",
    flat = "flat",
    bruteforce = "bruteforce",
    grid = "grid",
    vptree = "vptree",
    sparse = "sparse",
    hnsw = "hnsw",
    ivf = "ivf",
    ivfpq = "ivfpq",
    nsg = "nsg",
    nndescent = "nndescent",
    cagra = "cagra"
  )
  out <- unname(aliases[key])
  out[is.na(out)] <- key[is.na(out)]
  out
}

capability_row <- function(caps, backend, method, metric) {
  backend <- tolower(as.character(backend)[1L])
  metric <- tolower(as.character(metric)[1L])
  method_key <- canonical_method_key(method)
  caps_key <- canonical_method_key(caps$method)
  hit <- caps[caps$backend == backend & caps_key == method_key & caps$metric == metric, , drop = FALSE]
  if (!nrow(hit)) return(NULL)
  hit[1L, , drop = FALSE]
}

capability_status <- function(caps, backend, method, metric) {
  cap <- capability_row(caps, backend, method, metric)
  if (is.null(cap)) {
    return(list(
      listed = FALSE,
      supported = FALSE,
      notes = sprintf(
        "%s/%s/%s is not listed in faissR::nn_capabilities().",
        backend, method, metric
      )
    ))
  }
  list(
    listed = TRUE,
    supported = isTRUE(cap$supported[[1L]]),
    notes = cap$notes[[1L]] %||% "Unsupported by faissR::nn_capabilities()."
  )
}

auto_expected_skip <- function(caps, method, metric) {
  cpu <- capability_status(caps, "cpu", method, metric)
  cuda <- capability_status(caps, "cuda", method, metric)
  cuda_runtime <- isTRUE(faissR::cuda_available()) || isTRUE(faissR::cuvs_available())
  if (isTRUE(cpu$supported) || (cuda_runtime && isTRUE(cuda$supported))) {
    return(NULL)
  }
  if (!cuda_runtime && isTRUE(cuda$supported) && !isTRUE(cpu$supported)) {
    return(list(
      skip = TRUE,
      notes = paste(
        "backend = \"auto\" would need CUDA/cuVS for this method/metric,",
        "but CUDA/cuVS is unavailable in the current runtime.",
        sprintf("CPU: %s CUDA: %s", cpu$notes, cuda$notes)
      )
    ))
  }
  list(
    skip = TRUE,
    notes = paste(
      "backend = \"auto\" has no supported route for this method/metric in the current runtime.",
      sprintf("CPU: %s CUDA: %s", cpu$notes, cuda$notes)
    )
  )
}

is_expected_skip <- function(caps, backend, method, metric) {
  backend <- tolower(as.character(backend)[1L])
  if (identical(backend, "auto")) {
    return(auto_expected_skip(caps, method, metric))
  }
  if (!backend %in% c("cpu", "cuda")) return(NULL)
  cap <- capability_status(caps, backend, method, metric)
  if (!isTRUE(cap$supported)) return(list(skip = TRUE, notes = cap$notes))
  NULL
}

run_one <- function(x, dataset_name, backend, method, metric, k, cycle, n_threads,
                    timeout, reference, seed) {
  started <- proc.time()[["elapsed"]]
  old_options <- options(
    fastEmbedR.approx_knn_seed = as.integer(seed),
    fastEmbedR.faiss_gpu_ivf_tune_seed = as.integer(seed + 11L),
    fastEmbedR.cuvs_cagra_tune_seed = as.integer(seed + 23L)
  )
  on.exit(options(old_options), add = TRUE)
  set.seed(as.integer(seed))
  tryCatch({
    out <- with_elapsed_limit({
      faissR::nn(
        x,
        k = k,
        backend = backend,
        method = method,
        metric = metric,
        n_threads = n_threads
      )
    }, timeout)
    elapsed <- proc.time()[["elapsed"]] - started
    recall <- if (is.null(reference)) {
      data.frame(recall_at_k = NA_real_, median_recall_at_k = NA_real_, min_recall_at_k = NA_real_)
    } else if (is.null(reference$rows)) {
      benchmark_knn_recall(out, reference$knn, k = k)
    } else {
      sampled <- list(
        indices = out$indices[reference$rows, , drop = FALSE],
        distances = out$distances[reference$rows, , drop = FALSE]
      )
      benchmark_knn_recall(sampled, reference$knn, k = k)
    }
    result_row(
      dataset = dataset_name,
      n = nrow(x),
      p = ncol(x),
      backend = backend,
      method = method,
      metric = metric,
      k = k,
      cycle = cycle,
      n_threads = n_threads,
      status = "success",
      elapsed_sec = elapsed,
      peak_rss_gb = read_peak_rss_gb(),
      result_backend = attr(out, "backend") %||% NA_character_,
      resolved_backend = attr(out, "resolved_backend") %||% attr(out, "backend") %||% NA_character_,
      implementation_backend = nn_implementation_backend(out),
      exact = isTRUE(attr(out, "exact")),
      recall_at_k = recall$recall_at_k[[1L]],
      median_recall_at_k = recall$median_recall_at_k[[1L]],
      min_recall_at_k = recall$min_recall_at_k[[1L]],
      recall_reference = if (is.null(reference)) NA_character_ else reference$mode,
      recall_query_n = if (is.null(reference)) NA_integer_ else if (is.null(reference$rows)) nrow(x) else length(reference$rows)
    )
  }, error = function(e) {
    result_row(
      dataset = dataset_name,
      n = nrow(x),
      p = ncol(x),
      backend = backend,
      method = method,
      metric = metric,
      k = k,
      cycle = cycle,
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
file_arg <- cmd_args[grep("^--file=", cmd_args)[1L]] %||% "benchmark_scripts/benchmark_nn_metrics.R"
script_dir <- dirname(normalizePath(sub("^--file=", "", file_arg), mustWork = FALSE))
helper <- file.path(script_dir, "source.R")
if (!file.exists(helper)) helper <- file.path(getwd(), "benchmark_scripts/source.R")
source(helper)

data_root <- args$data_root %||% Sys.getenv("FAISSR_BENCHMARK_DATA", unset = file.path(getwd(), "Data"))
out_dir <- args$out_dir %||% Sys.getenv("FAISSR_BENCHMARK_OUT", unset = file.path(getwd(), paste0("faissR_NN_METRICS_", format(Sys.time(), "%Y%m%d_%H%M%S"))))
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

n_threads <- as_int_arg(args$threads, 4L)
configure_threads(n_threads)
seed <- as_int_arg(args$seed, 1L)
timeout <- as_int_arg(args$timeout, 600L)
cycles <- as_int_arg(args$cycles, 1L)
quality_n <- as_int_arg(args$quality_n, 512L)
quality_max_ops <- suppressWarnings(as.numeric(args$quality_max_ops %||% "5e9"))
if (length(quality_max_ops) != 1L || is.na(quality_max_ops) || !is.finite(quality_max_ops)) quality_max_ops <- 5e9
recall_threshold <- suppressWarnings(as.numeric(args$recall_threshold %||% "0.98"))
if (length(recall_threshold) != 1L || is.na(recall_threshold) || !is.finite(recall_threshold) ||
    recall_threshold < 0 || recall_threshold > 1) {
  recall_threshold <- 0.98
}

datasets <- split_arg(args$datasets, paste(c(dataset_index(data_root)$dataset, "SimulatedUniform2D", "SimulatedUniform3D"), collapse = ","))
backends <- split_arg(args$backends, "auto,cpu,cuda")
methods <- split_arg(args$methods, "auto,exact,flat,bruteforce,grid,vptree,HNSW,IVF,IVFPQ,NSG,NNDescent,CAGRA")
metrics <- split_arg(args$metrics, "euclidean,cosine,correlation,inner_product")
k_values <- as_int_vec_arg(split_arg(args$k_values, "5,10,15,50,100"), c(5L, 10L, 15L, 50L, 100L))

suppressPackageStartupMessages(library(faissR))
capabilities <- faissR::nn_capabilities()

config <- data.frame(
  key = c("data_root", "out_dir", "datasets", "backends", "methods", "metrics",
          "k_values", "threads", "timeout", "cycles", "quality_n", "quality_max_ops",
          "recall_threshold", "seed"),
  value = c(
    data_root, out_dir, paste(datasets, collapse = ","), paste(backends, collapse = ","),
    paste(methods, collapse = ","), paste(metrics, collapse = ","),
    paste(k_values, collapse = ","), n_threads, timeout, cycles, quality_n,
    format(quality_max_ops, scientific = TRUE), recall_threshold, seed
  ),
  stringsAsFactors = FALSE
)
utils::write.csv(config, file.path(out_dir, "nn_metric_benchmark_config.csv"), row.names = FALSE)
utils::write.csv(capabilities, file.path(out_dir, "nn_metric_capabilities.csv"), row.names = FALSE)

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
      backend = NA_character_,
      method = NA_character_,
      metric = NA_character_,
      k = NA_integer_,
      cycle = NA_integer_,
      n_threads = n_threads,
      status = "failed",
      error = conditionMessage(loaded)
    )
    next
  }
  x <- loaded$data
  references <- new.env(parent = emptyenv())
  for (metric in metrics) {
    for (k in k_values) {
      ref_key <- paste(metric, k, sep = "__")
      references[[ref_key]] <- tryCatch(
        metric_reference(x, k, metric, quality_n, quality_max_ops, n_threads, seed),
        error = function(e) NULL
      )
      for (cycle in seq_len(cycles)) {
        cycle_seed <- seed + (cycle - 1L) * 1000003L
        for (backend in backends) {
          for (method in methods) {
            row_id <- row_id + 1L
            expected <- is_expected_skip(capabilities, backend, method, metric)
            if (!is.null(expected)) {
              row <- result_row(
                dataset = dataset_name,
                n = nrow(x),
                p = ncol(x),
                backend = backend,
                method = method,
                metric = metric,
                k = k,
                cycle = cycle,
                n_threads = n_threads,
                status = "expected_skip",
                error = expected$notes,
                expected_skip = TRUE,
                capability_notes = expected$notes
              )
            } else {
              row <- run_one(
                x = x,
                dataset_name = dataset_name,
                backend = backend,
                method = method,
                metric = metric,
                k = k,
                cycle = cycle,
                n_threads = n_threads,
                timeout = timeout,
                reference = references[[ref_key]],
                seed = cycle_seed
              )
            }
            results[[row_id]] <- row
            utils::write.csv(
              do.call(rbind, results),
              file.path(out_dir, "nn_metric_benchmark_results.csv"),
              row.names = FALSE
            )
            cat(sprintf(
              "[%s] dataset=%s cycle=%s backend=%s method=%s metric=%s k=%s status=%s elapsed=%.3f\n",
              format(Sys.time(), "%Y-%m-%d %H:%M:%S"),
              dataset_name, cycle, backend, method, metric, k, row$status,
              ifelse(is.na(row$elapsed_sec), 0, row$elapsed_sec)
            ))
            flush.console()
          }
        }
      }
    }
  }
  rm(x, loaded, references)
  gc()
}

results_df <- do.call(rbind, results)
utils::write.csv(results_df, file.path(out_dir, "nn_metric_benchmark_results.csv"), row.names = FALSE)

ok <- results_df[results_df$status == "success", , drop = FALSE]
if (nrow(ok)) {
  ok$quality_score <- ifelse(is.na(ok$recall_at_k), -Inf, ok$recall_at_k)
  ok <- ok[order(ok$dataset, ok$backend, ok$metric, ok$k, -ok$quality_score, ok$elapsed_sec), , drop = FALSE]
  best <- do.call(rbind, lapply(split(ok, paste(ok$dataset, ok$backend, ok$metric, ok$k, sep = "__")), function(x) x[1L, , drop = FALSE]))
  best$quality_score <- NULL
  utils::write.csv(best, file.path(out_dir, "nn_metric_best_by_dataset_backend_metric_k.csv"), row.names = FALSE)

  ok_cycle <- ok[order(ok$dataset, ok$backend, ok$metric, ok$k, ok$cycle, -ok$quality_score, ok$elapsed_sec), , drop = FALSE]
  best_cycle <- do.call(rbind, lapply(split(ok_cycle, paste(ok_cycle$dataset, ok_cycle$backend, ok_cycle$metric, ok_cycle$k, ok_cycle$cycle, sep = "__")), function(x) x[1L, , drop = FALSE]))
  best_cycle$quality_score <- NULL
  utils::write.csv(best_cycle, file.path(out_dir, "nn_metric_best_by_dataset_backend_metric_k_cycle.csv"), row.names = FALSE)

  fastest <- NULL
  tunable <- ok[!is.na(ok$recall_at_k) & ok$recall_at_k >= recall_threshold, , drop = FALSE]
  if (nrow(tunable)) {
    tunable <- tunable[order(tunable$dataset, tunable$backend, tunable$metric, tunable$k, tunable$cycle, tunable$elapsed_sec), , drop = FALSE]
    fastest <- do.call(rbind, lapply(
      split(tunable, paste(tunable$dataset, tunable$backend, tunable$metric, tunable$k, tunable$cycle, sep = "__")),
      function(x) x[1L, , drop = FALSE]
    ))
    fastest$quality_score <- NULL
    utils::write.csv(
      fastest,
      file.path(out_dir, "nn_metric_fastest_at_recall_threshold.csv"),
      row.names = FALSE
    )
  }

  auto_rows <- ok[ok$method == "auto", , drop = FALSE]
  if (nrow(auto_rows) && !is.null(fastest) && nrow(fastest)) {
    keys <- c("dataset", "backend", "metric", "k", "cycle")
    auto_keep <- c(keys, "result_backend", "resolved_backend", "implementation_backend", "elapsed_sec", "recall_at_k", "recall_reference", "recall_query_n")
    fastest_keep <- c(keys, "method", "result_backend", "resolved_backend", "implementation_backend", "elapsed_sec", "recall_at_k", "recall_reference", "recall_query_n")
    names(auto_rows)[match(auto_keep[-seq_along(keys)], names(auto_rows))] <- paste0("auto_", auto_keep[-seq_along(keys)])
    names(fastest)[match(fastest_keep[-seq_along(keys)], names(fastest))] <- paste0("fastest_", fastest_keep[-seq_along(keys)])
    comparison <- merge(
      auto_rows[, c(keys, paste0("auto_", auto_keep[-seq_along(keys)])), drop = FALSE],
      fastest[, c(keys, paste0("fastest_", fastest_keep[-seq_along(keys)])), drop = FALSE],
      by = keys,
      all = FALSE
    )
    if (nrow(comparison)) {
      comparison$auto_is_fastest_method <- comparison$fastest_method == "auto"
      comparison$auto_uses_fastest_result_backend <- comparison$auto_result_backend == comparison$fastest_result_backend
      comparison$auto_uses_fastest_resolved_backend <- comparison$auto_resolved_backend == comparison$fastest_resolved_backend
      comparison$auto_uses_fastest_implementation <- comparison$auto_implementation_backend == comparison$fastest_implementation_backend
      comparison$auto_speed_ratio <- ifelse(
        comparison$fastest_elapsed_sec > 0,
        comparison$auto_elapsed_sec / comparison$fastest_elapsed_sec,
        ifelse(comparison$auto_elapsed_sec == 0, 1, Inf)
      )
      comparison$auto_recall_gap <- comparison$fastest_recall_at_k - comparison$auto_recall_at_k
      comparison <- comparison[order(comparison$dataset, comparison$backend, comparison$metric, comparison$k), , drop = FALSE]
      utils::write.csv(
        comparison,
        file.path(out_dir, "nn_metric_auto_vs_fastest.csv"),
        row.names = FALSE
      )
    }
  }
}

materials <- c(
  "# NN Metric Benchmark",
  "",
  "This benchmark exercises public faissR nearest-neighbour methods across device backends, metrics, and k values.",
  "",
  sprintf("- Output directory: `%s`", out_dir),
  sprintf("- Data root: `%s`", data_root),
  sprintf("- Backends: `%s`", paste(backends, collapse = "`, `")),
  sprintf("- Methods: `%s`", paste(methods, collapse = "`, `")),
  sprintf("- Metrics: `%s`", paste(metrics, collapse = "`, `")),
  sprintf("- k values: `%s`", paste(k_values, collapse = "`, `")),
  sprintf("- CPU thread cap: `%s`", n_threads),
  sprintf("- Timeout per combination: `%s` seconds", timeout),
  sprintf("- Cycles: `%s`", cycles),
  sprintf("- Fastest-method recall threshold: `%s`", recall_threshold),
  "",
  "Unsupported method/backend/metric combinations are preflighted with `faissR::nn_capabilities()` and recorded as `status = \"expected_skip\"` with `expected_skip = TRUE`.",
  "`nn_metric_capabilities.csv` stores the capability table used for that preflight.",
  "`result_backend`, `resolved_backend`, and `implementation_backend` separate the result-facing backend label from the concrete FAISS/cuVS/native implementation label.",
  "Recall is computed against exact CPU references. Small datasets use a full exact self-KNN reference; larger datasets use a deterministic sample of query rows when `quality_n * nrow(data) * ncol(data)` is within `quality_max_ops`. The `recall_reference` and `recall_query_n` columns record which reference mode was used. The same reference is reused across cycles for the same dataset/metric/k.",
  "`nn_metric_fastest_at_recall_threshold.csv` records the fastest successful method per dataset/backend/metric/k/cycle whose recall is at least `recall_threshold`.",
  "`nn_metric_auto_vs_fastest.csv` compares `method = \"auto\"` against that fastest high-recall row within the same cycle and records speed ratio, recall gap, whether auto itself was the fastest high-recall method, whether the result-facing backend matches, and whether the concrete implementation backend matches.",
  "`nn_metric_best_by_dataset_backend_metric_k_cycle.csv` stores the best row within each cycle; `nn_metric_best_by_dataset_backend_metric_k.csv` keeps the overall best row across cycles for backward-compatible summaries.",
  "The script does not add benchmark-only helpers to the package API."
)
writeLines(materials, file.path(out_dir, "MATERIALS_AND_METHODS_nn_metrics.md"))

cat("Saved NN metric benchmark files in: ", out_dir, "\n", sep = "")
