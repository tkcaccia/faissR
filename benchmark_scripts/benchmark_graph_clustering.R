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
  x <- x %||% default
  trimws(strsplit(x, ",", fixed = TRUE)[[1L]])
}

as_int_arg <- function(x, default) {
  value <- suppressWarnings(as.integer(x %||% default))
  if (length(value) != 1L || is.na(value) || value < 1L) as.integer(default) else value
}

as_int_vec_arg <- function(x, default) {
  value <- suppressWarnings(as.integer(x %||% default))
  value <- value[!is.na(value) & value > 0L]
  if (!length(value)) suppressWarnings(as.integer(default)) else value
}

configure_threads <- function(n_threads) {
  vars <- c(
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS"
  )
  for (var in vars) do.call(Sys.setenv, structure(list(as.character(n_threads)), names = var))
}

configure_native_libs <- function() {
  env_dir <- Sys.getenv("FAISSR_ENV_DIR", unset = "")
  cuda_lib <- Sys.getenv("FAISSR_CUDA_LIB_DIR", unset = "/usr/local/cuda/targets/x86_64-linux/lib")
  pieces <- character()
  if (nzchar(env_dir)) {
    Sys.setenv(CONDA_PREFIX = env_dir)
    pieces <- c(
      file.path(env_dir, "lib"),
      file.path(env_dir, "targets/x86_64-linux/lib")
    )
  }
  if (nzchar(cuda_lib)) pieces <- c(pieces, cuda_lib)
  old <- Sys.getenv("LD_LIBRARY_PATH", unset = "")
  pieces <- c(pieces, if (nzchar(old)) old else character())
  pieces <- unique(pieces[nzchar(pieces)])
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
  if (!file.exists(hit$path[[1L]])) {
    stop("Dataset file does not exist: ", hit$path[[1L]], call. = FALSE)
  }
  env <- new.env(parent = emptyenv())
  load(hit$path[[1L]], envir = env)
  if (!exists("dataset", envir = env, inherits = FALSE)) {
    stop("Dataset file must contain an object named `dataset`: ", hit$path[[1L]], call. = FALSE)
  }
  dataset <- get("dataset", envir = env, inherits = FALSE)
  if (!is.list(dataset) || is.null(dataset$data)) {
    stop("`dataset` must be a list containing `data`: ", hit$path[[1L]], call. = FALSE)
  }
  list(
    data = coerce_matrix(dataset$data),
    labels = if (is.null(dataset$labels)) NULL else dataset$labels
  )
}

with_elapsed_limit <- function(expr, timeout) {
  timeout <- suppressWarnings(as.numeric(timeout))
  if (length(timeout) == 1L && is.finite(timeout) && timeout > 0) {
    setTimeLimit(elapsed = timeout, transient = TRUE)
    on.exit(setTimeLimit(cpu = Inf, elapsed = Inf, transient = FALSE), add = TRUE)
  }
  force(expr)
}

result_row <- function(dataset, n, p, k, graph_backend, cluster_backend, method,
                       weight, n_clusters, n_threads, status, error = NA_character_,
                       load_sec = NA_real_, graph_sec = NA_real_,
                       cluster_sec = NA_real_, total_sec = NA_real_,
                       peak_rss_gb = NA_real_, n_edges = NA_integer_,
                       n_communities = NA_integer_, modularity = NA_real_,
                       ari = NA_real_, selected_resolution = NA_real_,
                       graph_resolved_backend = NA_character_,
                       cluster_resolved_backend = NA_character_,
                       graph_cached = NA,
                       expected_skip = FALSE) {
  data.frame(
    dataset = dataset,
    n = as.integer(n),
    p = as.integer(p),
    k = as.integer(k),
    graph_backend = graph_backend,
    graph_resolved_backend = graph_resolved_backend,
    cluster_backend = cluster_backend,
    cluster_resolved_backend = cluster_resolved_backend,
    method = method,
    weight = weight,
    n_clusters_requested = if (is.null(n_clusters)) NA_integer_ else as.integer(n_clusters),
    n_threads = as.integer(n_threads),
    status = status,
    error = error,
    load_sec = load_sec,
    graph_sec = graph_sec,
    cluster_sec = cluster_sec,
    total_sec = total_sec,
    peak_rss_gb = peak_rss_gb,
    n_edges = as.integer(n_edges),
    n_communities = as.integer(n_communities),
    modularity = modularity,
    ari = ari,
    selected_resolution = selected_resolution,
    graph_cached = if (is.na(graph_cached)) NA else isTRUE(graph_cached),
    expected_skip = isTRUE(expected_skip),
    stringsAsFactors = FALSE
  )
}

graph_cluster_expected_skip <- function(cluster_backend, method) {
  cluster_backend <- tolower(as.character(cluster_backend)[1L])
  method <- tolower(as.character(method)[1L])
  if (identical(cluster_backend, "cuda") && identical(method, "random_walking")) {
    return("CUDA random_walking is not enabled; graph_cluster(method = \"random_walking\", backend = \"auto\") stays on CPU.")
  }
  if (identical(cluster_backend, "cuda") &&
      method %in% c("louvain", "leiden") &&
      !isTRUE(faissR::cugraph_available())) {
    return("CUDA graph clustering requires faissR built with RAPIDS libcugraph; explicit CUDA clustering requests are expected skips.")
  }
  NULL
}

all_graph_cluster_expected_skips <- function(cluster_backends, methods) {
  checks <- unlist(lapply(cluster_backends, function(cluster_backend) {
    vapply(methods, function(method) {
      !is.null(graph_cluster_expected_skip(cluster_backend, method))
    }, logical(1))
  }), use.names = FALSE)
  length(checks) > 0L && all(checks)
}

graph_build_expected_skip <- function(graph_backend) {
  graph_backend <- tolower(as.character(graph_backend)[1L])
  if (identical(graph_backend, "cuda") &&
      !isTRUE(faissR::cuda_available()) &&
      !isTRUE(faissR::cuvs_available())) {
    return("CUDA graph construction requires a CUDA-capable faissR runtime; explicit CUDA graph requests are expected skips.")
  }
  NULL
}

build_graph_once <- function(data_obj, dataset_name, k, graph_backend, weight,
                             n_threads, timeout) {
  n <- nrow(data_obj$data)
  p <- ncol(data_obj$data)
  started <- proc.time()[["elapsed"]]
  tryCatch({
    graph <- with_elapsed_limit({
      faissR::knn_graph(
        data_obj$data,
        k = k,
        backend = graph_backend,
        weight = weight,
        n_threads = n_threads
      )
    }, timeout)
    elapsed <- proc.time()[["elapsed"]] - started
    graph_meta <- attr(graph, "faissR_graph") %||% list()
    list(
      status = "success",
      graph = graph,
      graph_sec = elapsed,
      n_edges = graph$n_edges %||% NA_integer_,
      graph_resolved_backend = graph_meta$resolved_backend %||% graph_backend,
      weight = attr(graph, "faissR_graph")$weight %||% weight,
      error = NA_character_
    )
  }, error = function(e) {
    list(
      status = "failed",
      graph = NULL,
      graph_sec = proc.time()[["elapsed"]] - started,
      n_edges = NA_integer_,
      weight = weight,
      error = conditionMessage(e),
      row = result_row(
        dataset = dataset_name,
        n = n,
        p = p,
        k = k,
        graph_backend = graph_backend,
        cluster_backend = NA_character_,
        method = NA_character_,
        weight = weight,
        n_clusters = NULL,
        n_threads = n_threads,
        status = "failed",
        error = conditionMessage(e),
        graph_sec = proc.time()[["elapsed"]] - started,
        peak_rss_gb = read_peak_rss_gb(),
        graph_cached = FALSE
      )
    )
  })
}

run_cluster_one <- function(data_obj, dataset_name, graph_obj, graph_sec,
                            graph_cached, n_edges, k, graph_backend,
                            cluster_backend, method, weight, n_threads,
                            target_mode, seed, timeout) {
  n <- nrow(data_obj$data)
  p <- ncol(data_obj$data)
  labels <- data_obj$labels
  label_count <- if (is.null(labels)) NA_integer_ else length(unique(labels[!is.na(labels)]))
  n_clusters <- NULL
  if (!identical(method, "random_walking") &&
      identical(target_mode, "labels") &&
      is.finite(label_count) &&
      label_count > 1L) {
    n_clusters <- as.integer(label_count)
  }

  started <- proc.time()[["elapsed"]]
  tryCatch({
    cluster <- with_elapsed_limit({
      cluster_started <- proc.time()[["elapsed"]]
      out <- faissR::graph_cluster(
        graph_obj,
        method = method,
        backend = cluster_backend,
        n_threads = n_threads,
        n_clusters = n_clusters,
        seed = seed
      )
      attr(out, "cluster_sec") <- proc.time()[["elapsed"]] - cluster_started
      out
    }, timeout)

    cluster_wall <- proc.time()[["elapsed"]] - started
    cluster_sec <- attr(cluster, "cluster_sec") %||% cluster_wall
    total_sec <- graph_sec + cluster_wall
    cluster_params <- cluster$parameters %||% list()
    result_row(
      dataset = dataset_name,
      n = n,
      p = p,
      k = k,
      graph_backend = graph_backend,
      cluster_backend = cluster_backend,
      method = method,
      weight = weight,
      n_clusters = n_clusters,
      n_threads = n_threads,
      status = "success",
      graph_sec = graph_sec,
      cluster_sec = cluster_sec,
      total_sec = total_sec,
      peak_rss_gb = read_peak_rss_gb(),
      n_edges = n_edges,
      n_communities = cluster$n_communities %||% length(unique(cluster$membership)),
      modularity = cluster$modularity %||% NA_real_,
      ari = benchmark_adjusted_rand_index(labels, cluster$membership),
      selected_resolution = cluster$selected_resolution %||% NA_real_,
      graph_resolved_backend = attr(graph_obj, "faissR_graph")$resolved_backend %||% graph_backend,
      cluster_resolved_backend = cluster_params$resolved_backend %||% (cluster$backend %||% cluster_backend),
      graph_cached = graph_cached
    )
  }, error = function(e) {
    result_row(
      dataset = dataset_name,
      n = n,
      p = p,
      k = k,
      graph_backend = graph_backend,
      cluster_backend = cluster_backend,
      method = method,
      weight = weight,
      n_clusters = n_clusters,
      n_threads = n_threads,
      status = "failed",
      error = conditionMessage(e),
      graph_sec = graph_sec,
      total_sec = graph_sec + (proc.time()[["elapsed"]] - started),
      peak_rss_gb = read_peak_rss_gb(),
      n_edges = n_edges,
      graph_cached = graph_cached
    )
  })
}

args <- parse_args()
configure_native_libs()

data_root <- args$data_root %||% Sys.getenv("FAISSR_BENCHMARK_DATA", unset = file.path(getwd(), "Data"))
out_dir <- args$out_dir %||% Sys.getenv("FAISSR_BENCHMARK_OUT", unset = file.path(getwd(), paste0("faissR_GRAPH_CLUSTER_", format(Sys.time(), "%Y%m%d_%H%M%S"))))
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

cmd_args <- commandArgs(FALSE)
file_arg <- cmd_args[grep("^--file=", cmd_args)[1L]] %||% "benchmark_scripts/benchmark_graph_clustering.R"
script_dir <- dirname(normalizePath(sub("^--file=", "", file_arg), mustWork = FALSE))
helper <- file.path(script_dir, "source.R")
if (!file.exists(helper)) helper <- file.path(getwd(), "benchmark_scripts/source.R")
source(helper)

n_threads <- as_int_arg(args$threads, 2L)
configure_threads(n_threads)
seed <- as_int_arg(args$seed, 1L)
timeout <- as_int_arg(args$timeout, 600L)
k_values <- as_int_vec_arg(split_arg(args$k_values, "15,50,100"), 50L)
datasets <- split_arg(args$datasets, paste(c(
  dataset_index(data_root)$dataset,
  "SimulatedUniform2D",
  "SimulatedUniform3D"
), collapse = ","))
methods <- split_arg(args$methods, "random_walking,louvain,leiden")
graph_backends <- split_arg(args$graph_backends, "auto,cpu,cuda")
cluster_backends <- split_arg(args$cluster_backends, "auto,cpu,cuda")
weight <- args$weight %||% "auto"
target_mode <- args$target_clusters %||% "labels"

suppressPackageStartupMessages(library(faissR))

config <- data.frame(
  key = c("data_root", "out_dir", "datasets", "methods", "graph_backends",
          "cluster_backends", "k_values", "threads", "timeout", "weight",
          "target_clusters", "seed"),
  value = c(data_root, out_dir, paste(datasets, collapse = ","), paste(methods, collapse = ","),
            paste(graph_backends, collapse = ","), paste(cluster_backends, collapse = ","),
            paste(k_values, collapse = ","), n_threads, timeout, weight, target_mode, seed),
  stringsAsFactors = FALSE
)
utils::write.csv(config, file.path(out_dir, "graph_cluster_benchmark_config.csv"), row.names = FALSE)

results <- list()
row_id <- 0L
for (dataset_name in datasets) {
  load_started <- proc.time()[["elapsed"]]
  loaded <- tryCatch(load_dataset(dataset_name, data_root, seed), error = identity)
  load_sec <- proc.time()[["elapsed"]] - load_started
  if (inherits(loaded, "error")) {
    row_id <- row_id + 1L
    results[[row_id]] <- result_row(
      dataset = dataset_name,
      n = NA_integer_,
      p = NA_integer_,
      k = NA_integer_,
      graph_backend = NA_character_,
      cluster_backend = NA_character_,
      method = NA_character_,
      weight = weight,
      n_clusters = NULL,
      n_threads = n_threads,
      status = "failed",
      error = conditionMessage(loaded),
      load_sec = load_sec
    )
    next
  }
  for (k in k_values) {
    for (graph_backend in graph_backends) {
      graph_skip_reason <- graph_build_expected_skip(graph_backend)
      if (!is.null(graph_skip_reason)) {
        graph_build <- list(
          status = "expected_skip",
          graph = NULL,
          graph_sec = NA_real_,
          n_edges = NA_integer_,
          weight = weight,
          error = graph_skip_reason
        )
      } else if (all_graph_cluster_expected_skips(cluster_backends, methods)) {
        graph_build <- list(
          status = "skipped",
          graph = NULL,
          graph_sec = NA_real_,
          n_edges = NA_integer_,
          weight = weight,
          error = "Graph construction skipped because every clustering row in this block is an expected skip."
        )
      } else {
        graph_build <- build_graph_once(
          data_obj = loaded,
          dataset_name = dataset_name,
          k = k,
          graph_backend = graph_backend,
          weight = weight,
          n_threads = n_threads,
          timeout = timeout
        )
      }
      for (cluster_backend in cluster_backends) {
        for (method in methods) {
          row_id <- row_id + 1L
          skip_reason <- graph_cluster_expected_skip(cluster_backend, method)
          if (!is.null(skip_reason)) {
            row <- result_row(
              dataset = dataset_name,
              n = nrow(loaded$data),
              p = ncol(loaded$data),
              k = k,
              graph_backend = graph_backend,
              cluster_backend = cluster_backend,
              method = method,
              weight = if (identical(graph_build$status, "success")) graph_build$weight else weight,
              n_clusters = NULL,
              n_threads = n_threads,
              status = "expected_skip",
              error = skip_reason,
              graph_sec = if (identical(graph_build$status, "success")) graph_build$graph_sec else NA_real_,
              total_sec = if (identical(graph_build$status, "success")) graph_build$graph_sec else NA_real_,
              peak_rss_gb = read_peak_rss_gb(),
              n_edges = if (identical(graph_build$status, "success")) graph_build$n_edges else NA_integer_,
              graph_resolved_backend = if (identical(graph_build$status, "success")) graph_build$graph_resolved_backend else NA_character_,
              graph_cached = identical(graph_build$status, "success"),
              expected_skip = TRUE
            )
          } else if (!identical(graph_build$status, "success")) {
            row <- result_row(
              dataset = dataset_name,
              n = nrow(loaded$data),
              p = ncol(loaded$data),
              k = k,
              graph_backend = graph_backend,
              cluster_backend = cluster_backend,
              method = method,
              weight = weight,
              n_clusters = NULL,
              n_threads = n_threads,
              status = if (identical(graph_build$status, "expected_skip")) "expected_skip" else "failed",
              error = graph_build$error,
              graph_sec = graph_build$graph_sec,
              total_sec = graph_build$graph_sec,
              peak_rss_gb = read_peak_rss_gb(),
              graph_resolved_backend = if (identical(graph_build$status, "success")) graph_build$graph_resolved_backend else NA_character_,
              graph_cached = FALSE,
              expected_skip = identical(graph_build$status, "expected_skip")
            )
          } else {
            row <- run_cluster_one(
              data_obj = loaded,
              dataset_name = dataset_name,
              graph_obj = graph_build$graph,
              graph_sec = graph_build$graph_sec,
              graph_cached = TRUE,
              n_edges = graph_build$n_edges,
              k = k,
              graph_backend = graph_backend,
              cluster_backend = cluster_backend,
              method = method,
              weight = graph_build$weight,
              n_threads = n_threads,
              target_mode = target_mode,
              seed = seed,
              timeout = timeout
            )
          }
          row$load_sec <- load_sec
          results[[row_id]] <- row
          utils::write.csv(
            do.call(rbind, results),
            file.path(out_dir, "graph_cluster_benchmark_results.csv"),
            row.names = FALSE
          )
          cat(
            sprintf(
              "[%s] dataset=%s k=%s graph=%s cluster=%s method=%s status=%s total=%.3f\n",
              format(Sys.time(), "%Y-%m-%d %H:%M:%S"),
              dataset_name, k, graph_backend, cluster_backend, method,
              row$status, row$total_sec
            )
          )
        }
      }
    }
  }
  rm(loaded)
  gc()
}

results_df <- do.call(rbind, results)
utils::write.csv(results_df, file.path(out_dir, "graph_cluster_benchmark_results.csv"), row.names = FALSE)

ok <- results_df[results_df$status == "success", , drop = FALSE]
if (nrow(ok)) {
  ok$order_score <- ifelse(is.na(ok$ari), -Inf, ok$ari)
  ok <- ok[order(ok$dataset, -ok$order_score, ok$total_sec), , drop = FALSE]
  best <- do.call(rbind, lapply(split(ok, ok$dataset), function(x) x[1L, , drop = FALSE]))
  best$order_score <- NULL
  utils::write.csv(best, file.path(out_dir, "graph_cluster_best_by_dataset.csv"), row.names = FALSE)
}

materials <- c(
  "# Graph Clustering Benchmark",
  "",
  "This benchmark evaluates faissR `knn_graph()` plus `graph_cluster()` on labelled benchmark datasets and simulated datasets.",
  "",
  sprintf("- Output directory: `%s`", out_dir),
  sprintf("- Data root: `%s`", data_root),
  sprintf("- Methods: `%s`", paste(methods, collapse = "`, `")),
  sprintf("- Graph backends: `%s`", paste(graph_backends, collapse = "`, `")),
  sprintf("- Clustering backends: `%s`", paste(cluster_backends, collapse = "`, `")),
  sprintf("- k values: `%s`", paste(k_values, collapse = "`, `")),
  sprintf("- CPU thread cap: `%s`", n_threads),
  "",
  "ARI is computed in `benchmark_scripts/source.R` from labels stored in each dataset object. ARI is `NA` when labels are unavailable.",
  "When `target_clusters = \"labels\"`, Louvain and Leiden receive `n_clusters = length(unique(labels))`; random-walking is benchmarked without a cluster-count target because the public API does not support that option for random-walking.",
  "Each KNN graph is built once per dataset/k/graph-backend/weight combination and reused across clustering methods and clustering backends. The `graph_cached` column records this reuse; `graph_sec` is the graph construction time for the shared graph, `cluster_sec` is the clustering-only time, and `total_sec` is `graph_sec + cluster_sec`.",
  "`graph_backend` and `cluster_backend` record the requested public backends. `graph_resolved_backend` and `cluster_resolved_backend` record the resolved public device policy after `auto` selection, so CPU/CUDA rows can be audited without opening the R objects.",
  "Unsupported graph-clustering combinations known from the public API, such as CUDA random_walking, are recorded as `status = \"expected_skip\"` with `expected_skip = TRUE`. If every row in a graph-build block is an expected skip, graph construction is skipped and graph timing/edge columns remain `NA`.",
  "CUDA rows are recorded as failed when faissR was not built with the required CUDA/cuGraph support; the benchmark does not silently replace CUDA clustering with CPU clustering."
)
writeLines(materials, file.path(out_dir, "MATERIALS_AND_METHODS_graph_clustering.md"))

cat("Saved graph clustering benchmark files in: ", out_dir, "\n", sep = "")
