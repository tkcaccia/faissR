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

required_nonnegative_numeric_arg <- function(x, arg) {
  value <- suppressWarnings(as.numeric(x))
  if (length(value) != 1L || is.na(value) || !is.finite(value) || value < 0) {
    stop("`", arg, "` must be a non-negative numeric value.", call. = FALSE)
  }
  value
}

default_graph_k_values <- function() {
  c(15L, 50L, 100L)
}

default_graph_cluster_methods <- function() {
  c("random_walking", "louvain", "leiden")
}

default_graph_backends <- function() {
  c("auto", "cpu", "cuda")
}

validate_choice_values <- function(values, valid, arg_name) {
  values <- unique(trimws(as.character(values)))
  values <- values[nzchar(values)]
  invalid <- values[!values %in% valid]
  if (length(invalid)) {
    stop(
      "`", arg_name, "` must contain only: ",
      paste(valid, collapse = ", "),
      ". Invalid value(s): ",
      paste(invalid, collapse = ", "),
      ".",
      call. = FALSE
    )
  }
  if (!length(values)) {
    stop("`", arg_name, "` must contain at least one value.", call. = FALSE)
  }
  values
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

result_row <- function(dataset, n, p, cycle, k, graph_backend, cluster_backend, method,
                       weight, n_clusters, n_threads, status, error = NA_character_,
                       n_clusters_source = NA_character_,
                       load_sec = NA_real_, graph_sec = NA_real_,
                       cluster_sec = NA_real_, total_sec = NA_real_,
                       peak_rss_gb = NA_real_, n_edges = NA_integer_,
                       n_communities = NA_integer_, modularity = NA_real_,
                       ari = NA_real_, selected_resolution = NA_real_,
                       graph_resolved_backend = NA_character_,
                       cluster_resolved_backend = NA_character_,
                       graph_preflight_route = NA_character_,
                       cluster_preflight_route = NA_character_,
                       graph_cached = NA,
                       expected_skip = FALSE) {
  data.frame(
    dataset = dataset,
    n = as.integer(n),
    p = as.integer(p),
    cycle = as.integer(cycle),
    k = as.integer(k),
    graph_backend = graph_backend,
    graph_resolved_backend = graph_resolved_backend,
    graph_preflight_route = graph_preflight_route,
    cluster_backend = cluster_backend,
    cluster_resolved_backend = cluster_resolved_backend,
    cluster_preflight_route = cluster_preflight_route,
    method = method,
    weight = weight,
    n_clusters_requested = if (is.null(n_clusters)) NA_integer_ else as.integer(n_clusters),
    n_clusters_source = n_clusters_source,
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

dominant_value <- function(x) {
  x <- as.character(x)
  x <- x[!is.na(x) & nzchar(x)]
  if (!length(x)) return(NA_character_)
  names(sort(table(x), decreasing = TRUE))[[1L]]
}

dominant_integer <- function(x) {
  x <- suppressWarnings(as.integer(x))
  x <- x[!is.na(x) & is.finite(x)]
  if (!length(x)) return(NA_integer_)
  as.integer(names(sort(table(x), decreasing = TRUE))[[1L]])
}

finite_median <- function(x) {
  x <- suppressWarnings(as.numeric(x))
  x <- x[is.finite(x)]
  if (!length(x)) return(NA_real_)
  stats::median(x)
}

finite_min <- function(x) {
  x <- suppressWarnings(as.numeric(x))
  x <- x[is.finite(x)]
  if (!length(x)) return(NA_real_)
  min(x)
}

finite_max <- function(x) {
  x <- suppressWarnings(as.numeric(x))
  x <- x[is.finite(x)]
  if (!length(x)) return(NA_real_)
  max(x)
}

summarize_graph_cycles <- function(ok) {
  cluster_key <- ifelse(is.na(ok$n_clusters_requested), "NA", as.character(ok$n_clusters_requested))
  parts <- split(ok, paste(ok$dataset, ok$k, ok$graph_backend, ok$cluster_backend, ok$method, ok$weight, cluster_key, sep = "__"))
  summary <- lapply(parts, function(x) {
    data.frame(
      dataset = x$dataset[[1L]],
      k = as.integer(x$k[[1L]]),
      graph_backend = x$graph_backend[[1L]],
      graph_resolved_backend = dominant_value(x$graph_resolved_backend),
      cluster_backend = x$cluster_backend[[1L]],
      cluster_resolved_backend = dominant_value(x$cluster_resolved_backend),
      method = x$method[[1L]],
      weight = x$weight[[1L]],
      n = as.integer(x$n[[1L]]),
      p = as.integer(x$p[[1L]]),
      n_threads = as.integer(x$n_threads[[1L]]),
      success_cycles = length(unique(x$cycle)),
      success_rows = nrow(x),
      median_load_sec = finite_median(x$load_sec),
      median_graph_sec = finite_median(x$graph_sec),
      min_graph_sec = finite_min(x$graph_sec),
      max_graph_sec = finite_max(x$graph_sec),
      median_cluster_sec = finite_median(x$cluster_sec),
      min_cluster_sec = finite_min(x$cluster_sec),
      max_cluster_sec = finite_max(x$cluster_sec),
      median_total_sec = finite_median(x$total_sec),
      min_total_sec = finite_min(x$total_sec),
      max_total_sec = finite_max(x$total_sec),
      median_ari = finite_median(x$ari),
      min_ari = finite_min(x$ari),
      max_ari = finite_max(x$ari),
      median_modularity = finite_median(x$modularity),
      min_modularity = finite_min(x$modularity),
      max_modularity = finite_max(x$modularity),
      median_n_edges = finite_median(x$n_edges),
      median_n_communities = finite_median(x$n_communities),
      median_selected_resolution = finite_median(x$selected_resolution),
      n_clusters_requested = dominant_integer(x$n_clusters_requested),
      n_clusters_source = dominant_value(x$n_clusters_source),
      graph_cached = any(as.logical(x$graph_cached), na.rm = TRUE),
      stringsAsFactors = FALSE
    )
  })
  out <- do.call(rbind, summary)
  out[order(out$dataset, -out$median_ari, out$median_total_sec), , drop = FALSE]
}

recommend_graph_cluster_methods <- function(cycle_summary, ari_tolerance) {
  cluster_key <- ifelse(is.na(cycle_summary$n_clusters_requested), "NA", as.character(cycle_summary$n_clusters_requested))
  parts <- split(cycle_summary, paste(cycle_summary$dataset, cycle_summary$k, cluster_key, sep = "__"))
  recommendations <- lapply(parts, function(x) {
    has_ari <- is.finite(x$median_ari)
    candidates <- if (any(has_ari)) {
      best_ari <- max(x$median_ari[has_ari])
      out <- x[has_ari & x$median_ari >= best_ari - ari_tolerance, , drop = FALSE]
      out$recommendation_basis <- "fastest_within_ari_tolerance"
      out
    } else {
      out <- x
      out$recommendation_basis <- "speed_only_no_ari"
      out
    }
    candidates <- candidates[order(
      candidates$median_total_sec,
      -ifelse(is.finite(candidates$median_ari), candidates$median_ari, -Inf),
      -ifelse(is.finite(candidates$median_modularity), candidates$median_modularity, -Inf)
    ), , drop = FALSE]
    candidates[1L, , drop = FALSE]
  })
  out <- do.call(rbind, recommendations)
  row.names(out) <- NULL
  out[order(out$dataset, out$k, out$n_clusters_requested), , drop = FALSE]
}

compare_auto_graph_to_recommendations <- function(cycle_summary, recommendations) {
  if (!nrow(recommendations)) return(recommendations)
  auto <- cycle_summary[
    cycle_summary$graph_backend == "auto" | cycle_summary$cluster_backend == "auto",
    ,
    drop = FALSE
  ]
  if (!nrow(auto)) return(data.frame())
  keys <- c("dataset", "k", "n_clusters_requested")
  keep <- c(
    keys, "graph_backend", "graph_resolved_backend", "cluster_backend",
    "cluster_resolved_backend", "method", "weight", "success_cycles",
    "median_graph_sec", "median_cluster_sec", "median_total_sec",
    "median_ari", "min_ari", "median_modularity", "median_n_communities",
    "median_selected_resolution", "n_clusters_source", "graph_cached"
  )
  rec_keep <- c(keep, "recommendation_basis")
  auto <- auto[, keep, drop = FALSE]
  recommendations <- recommendations[, rec_keep, drop = FALSE]
  names(auto)[match(keep[-seq_along(keys)], names(auto))] <- paste0("auto_", keep[-seq_along(keys)])
  names(recommendations)[match(rec_keep[-seq_along(keys)], names(recommendations))] <- paste0("recommended_", rec_keep[-seq_along(keys)])
  comparison <- merge(auto, recommendations, by = keys, all = FALSE)
  if (!nrow(comparison)) return(comparison)
  comparison$auto_uses_recommended_graph_backend <- comparison$auto_graph_backend == comparison$recommended_graph_backend
  comparison$auto_uses_recommended_cluster_backend <- comparison$auto_cluster_backend == comparison$recommended_cluster_backend
  comparison$auto_uses_recommended_method <- comparison$auto_method == comparison$recommended_method
  comparison$auto_median_speed_ratio <- safe_positive_ratio(
    comparison$auto_median_total_sec,
    comparison$recommended_median_total_sec
  )
  comparison$auto_median_ari_gap <- safe_difference(
    comparison$recommended_median_ari,
    comparison$auto_median_ari
  )
  comparison$auto_modularity_gap <- safe_difference(
    comparison$recommended_median_modularity,
    comparison$auto_median_modularity
  )
  comparison[order(comparison$dataset, comparison$k, comparison$n_clusters_requested, comparison$auto_graph_backend, comparison$auto_cluster_backend, comparison$auto_method), , drop = FALSE]
}

safe_positive_ratio <- function(numerator, denominator) {
  numerator <- suppressWarnings(as.numeric(numerator))
  denominator <- suppressWarnings(as.numeric(denominator))
  out <- rep(NA_real_, max(length(numerator), length(denominator)))
  numerator <- rep(numerator, length.out = length(out))
  denominator <- rep(denominator, length.out = length(out))
  ok <- is.finite(numerator) & is.finite(denominator) & denominator > 0
  out[ok] <- numerator[ok] / denominator[ok]
  out
}

safe_difference <- function(left, right) {
  left <- suppressWarnings(as.numeric(left))
  right <- suppressWarnings(as.numeric(right))
  out <- rep(NA_real_, max(length(left), length(right)))
  left <- rep(left, length.out = length(out))
  right <- rep(right, length.out = length(out))
  ok <- is.finite(left) & is.finite(right)
  out[ok] <- left[ok] - right[ok]
  out
}

graph_rank_value <- function(data, column, default, higher_is_better = FALSE) {
  value <- if (column %in% names(data)) data[[column]] else rep(default, nrow(data))
  value <- suppressWarnings(as.numeric(value))
  value[!is.finite(value)] <- default
  if (higher_is_better) -value else value
}

rank_graph_cluster_success <- function(ok) {
  if (!nrow(ok)) return(ok)
  ok[order(
    ok$dataset,
    graph_rank_value(ok, "ari", -Inf, higher_is_better = TRUE),
    graph_rank_value(ok, "modularity", -Inf, higher_is_better = TRUE),
    graph_rank_value(ok, "total_sec", Inf)
  ), , drop = FALSE]
}

select_graph_best_rows <- function(ok, group_cols = c("dataset")) {
  ranked <- rank_graph_cluster_success(ok)
  if (!nrow(ranked)) return(ranked)
  key <- do.call(
    paste,
    c(
      lapply(group_cols, function(col) {
        value <- ranked[[col]]
        ifelse(is.na(value), "NA", as.character(value))
      }),
      sep = "\r"
    )
  )
  ranked[!duplicated(key), , drop = FALSE]
}

graph_cluster_expected_skip <- function(cluster_backend, method) {
  cluster_backend <- tolower(as.character(cluster_backend)[1L])
  method <- tolower(as.character(method)[1L])
  if (identical(cluster_backend, "cuda") && identical(method, "random_walking")) {
    return("CUDA random_walking is not enabled; graph_cluster(method = \"random_walking\", backend = \"cuda\") is an expected skip. With backend = \"auto\", random_walking stays on CPU.")
  }
  if (identical(cluster_backend, "cuda") &&
      method %in% c("louvain", "leiden") &&
      !isTRUE(faissR::cugraph_available())) {
    return("CUDA graph clustering requires faissR built with RAPIDS libcugraph; explicit CUDA clustering requests are expected skips.")
  }
  NULL
}

graph_build_preflight_route <- function(graph_backend) {
  helper <- tryCatch(getFromNamespace("resolve_knn_graph_backend", "faissR"), error = function(e) NULL)
  if (!is.function(helper)) return(NA_character_)
  tryCatch(as.character(helper(graph_backend))[1L], error = function(e) NA_character_)
}

graph_cluster_preflight_route <- function(cluster_backend) {
  helper <- tryCatch(getFromNamespace("resolve_graph_cluster_backend", "faissR"), error = function(e) NULL)
  if (!is.function(helper)) return(NA_character_)
  tryCatch(as.character(helper(cluster_backend))[1L], error = function(e) NA_character_)
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

normalize_target_clusters_mode <- function(target_mode) {
  target_mode <- tolower(trimws(as.character(target_mode)[1L] %||% "labels"))
  aliases <- c(
    labels = "labels",
    label = "labels",
    dataset_labels = "labels",
    none = "none",
    off = "none",
    false = "none",
    no = "none"
  )
  if (!target_mode %in% names(aliases)) {
    stop(
      "`target_clusters` must be one of \"labels\" or \"none\".",
      call. = FALSE
    )
  }
  unname(aliases[[target_mode]])
}

label_target_clusters <- function(labels, target_mode) {
  if (!identical(target_mode, "labels") || is.null(labels)) return(NULL)
  label_count <- length(unique(labels[!is.na(labels)]))
  if (is.finite(label_count) && label_count > 1L) as.integer(label_count) else NULL
}

build_graph_once <- function(data_obj, dataset_name, k, graph_backend, weight,
                             n_clusters, n_threads, timeout, cycle = 1L) {
  n <- nrow(data_obj$data)
  p <- ncol(data_obj$data)
  started <- proc.time()[["elapsed"]]
  graph_preflight_route <- graph_build_preflight_route(graph_backend)
  tryCatch({
    graph <- with_elapsed_limit({
      faissR::knn_graph(
        data_obj$data,
        k = k,
        backend = graph_backend,
        weight = weight,
        n_clusters = n_clusters,
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
      graph_preflight_route = graph_preflight_route,
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
        cycle = cycle,
        k = k,
        graph_backend = graph_backend,
        cluster_backend = NA_character_,
        method = NA_character_,
        weight = weight,
        n_clusters = n_clusters,
        n_threads = n_threads,
        status = "failed",
        error = conditionMessage(e),
        graph_sec = proc.time()[["elapsed"]] - started,
        peak_rss_gb = read_peak_rss_gb(),
        graph_preflight_route = graph_preflight_route,
        graph_cached = FALSE
      )
    )
  })
}

run_cluster_one <- function(data_obj, dataset_name, graph_obj, graph_sec,
                            graph_cached, n_edges, k, graph_backend,
                            cluster_backend, method, weight, n_threads,
                            target_mode, seed, timeout, cycle = 1L) {
  n <- nrow(data_obj$data)
  p <- ncol(data_obj$data)
  labels <- data_obj$labels
  graph_meta <- attr(graph_obj, "faissR_graph") %||% list()
  graph_preflight_route <- graph_build_preflight_route(graph_backend)
  cluster_preflight_route <- graph_cluster_preflight_route(cluster_backend)
  graph_target <- graph_meta$target_n_clusters %||% NULL
  n_clusters <- NULL
  n_clusters_source <- NA_character_
  if (!identical(method, "random_walking")) {
    if (is.null(graph_target)) {
      n_clusters <- label_target_clusters(labels, target_mode)
      if (!is.null(n_clusters)) n_clusters_source <- target_mode
    } else {
      n_clusters <- NULL
      n_clusters_source <- "stored_graph_target"
    }
  }
  n_clusters_requested <- graph_target %||% n_clusters

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
      cycle = cycle,
      k = k,
      graph_backend = graph_backend,
      cluster_backend = cluster_backend,
      method = method,
      weight = weight,
      n_clusters = n_clusters_requested,
      n_clusters_source = n_clusters_source,
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
      graph_preflight_route = graph_meta$resolved_backend %||% graph_preflight_route,
      cluster_preflight_route = cluster_preflight_route,
      graph_cached = graph_cached
    )
  }, error = function(e) {
    result_row(
      dataset = dataset_name,
      n = n,
      p = p,
      cycle = cycle,
      k = k,
      graph_backend = graph_backend,
      cluster_backend = cluster_backend,
      method = method,
      weight = weight,
      n_clusters = n_clusters_requested,
      n_clusters_source = n_clusters_source,
      n_threads = n_threads,
      status = "failed",
      error = conditionMessage(e),
      graph_sec = graph_sec,
      total_sec = graph_sec + (proc.time()[["elapsed"]] - started),
      peak_rss_gb = read_peak_rss_gb(),
      n_edges = n_edges,
      graph_preflight_route = graph_meta$resolved_backend %||% graph_preflight_route,
      cluster_preflight_route = cluster_preflight_route,
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
cycles <- as_int_arg(args$cycles, 1L)
ari_tolerance <- required_nonnegative_numeric_arg(args$ari_tolerance %||% "0.01", "ari_tolerance")
k_values <- as_int_vec_arg(
  split_arg(args$k_values, paste(default_graph_k_values(), collapse = ",")),
  default_graph_k_values()
)
datasets <- split_arg(args$datasets, paste(c(
  dataset_index(data_root)$dataset,
  "SimulatedUniform2D",
  "SimulatedUniform3D"
), collapse = ","))
methods <- validate_choice_values(
  split_arg(args$methods, paste(default_graph_cluster_methods(), collapse = ",")),
  default_graph_cluster_methods(),
  "methods"
)
graph_backends <- validate_choice_values(
  split_arg(args$graph_backends, paste(default_graph_backends(), collapse = ",")),
  default_graph_backends(),
  "graph_backends"
)
cluster_backends <- validate_choice_values(
  split_arg(args$cluster_backends, paste(default_graph_backends(), collapse = ",")),
  default_graph_backends(),
  "cluster_backends"
)
weight <- args$weight %||% "auto"
target_mode <- normalize_target_clusters_mode(args$target_clusters %||% "labels")

suppressPackageStartupMessages(library(faissR))

config <- data.frame(
  key = c("data_root", "out_dir", "datasets", "methods", "graph_backends",
          "cluster_backends", "k_values", "threads", "timeout", "weight",
          "target_clusters", "cycles", "ari_tolerance", "seed"),
  value = c(data_root, out_dir, paste(datasets, collapse = ","), paste(methods, collapse = ","),
            paste(graph_backends, collapse = ","), paste(cluster_backends, collapse = ","),
            paste(k_values, collapse = ","), n_threads, timeout, weight, target_mode,
            cycles, ari_tolerance, seed),
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
      cycle = NA_integer_,
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
      for (cycle in seq_len(cycles)) {
        cycle_seed <- seed + (cycle - 1L) * 1000003L
        graph_target_clusters <- NULL
        if (!"random_walking" %in% methods) {
          graph_target_clusters <- label_target_clusters(loaded$labels, target_mode)
        }
        graph_preflight_route <- graph_build_preflight_route(graph_backend)
        graph_skip_reason <- graph_build_expected_skip(graph_backend)
        if (!is.null(graph_skip_reason)) {
          graph_build <- list(
            status = "expected_skip",
            graph = NULL,
            graph_sec = NA_real_,
            n_edges = NA_integer_,
            graph_preflight_route = graph_preflight_route,
            weight = weight,
            error = graph_skip_reason
          )
        } else if (all_graph_cluster_expected_skips(cluster_backends, methods)) {
          graph_build <- list(
            status = "skipped",
            graph = NULL,
            graph_sec = NA_real_,
            n_edges = NA_integer_,
            graph_preflight_route = graph_preflight_route,
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
            n_clusters = graph_target_clusters,
            n_threads = n_threads,
            timeout = timeout,
            cycle = cycle
          )
        }
        for (cluster_backend in cluster_backends) {
          for (method in methods) {
            row_id <- row_id + 1L
            cluster_preflight_route <- graph_cluster_preflight_route(cluster_backend)
            skip_reason <- graph_cluster_expected_skip(cluster_backend, method)
            row_target_clusters <- if (identical(method, "random_walking")) {
              NULL
            } else {
              graph_target_clusters %||% label_target_clusters(loaded$labels, target_mode)
            }
            row_target_source <- if (identical(method, "random_walking") || is.null(row_target_clusters)) {
              NA_character_
            } else if (!is.null(graph_target_clusters)) {
              "stored_graph_target"
            } else {
              target_mode
            }
            if (!is.null(skip_reason)) {
              row <- result_row(
                dataset = dataset_name,
                n = nrow(loaded$data),
                p = ncol(loaded$data),
                cycle = cycle,
                k = k,
                graph_backend = graph_backend,
                cluster_backend = cluster_backend,
                method = method,
                weight = if (identical(graph_build$status, "success")) graph_build$weight else weight,
                n_clusters = row_target_clusters,
                n_clusters_source = row_target_source,
                n_threads = n_threads,
                status = "expected_skip",
                error = skip_reason,
                graph_sec = if (identical(graph_build$status, "success")) graph_build$graph_sec else NA_real_,
                total_sec = if (identical(graph_build$status, "success")) graph_build$graph_sec else NA_real_,
                peak_rss_gb = read_peak_rss_gb(),
                n_edges = if (identical(graph_build$status, "success")) graph_build$n_edges else NA_integer_,
                graph_resolved_backend = if (identical(graph_build$status, "success")) graph_build$graph_resolved_backend else NA_character_,
                graph_preflight_route = graph_build$graph_preflight_route %||% graph_preflight_route,
                cluster_preflight_route = cluster_preflight_route,
                graph_cached = identical(graph_build$status, "success"),
                expected_skip = TRUE
              )
            } else if (!identical(graph_build$status, "success")) {
              row <- result_row(
                dataset = dataset_name,
                n = nrow(loaded$data),
                p = ncol(loaded$data),
                cycle = cycle,
                k = k,
                graph_backend = graph_backend,
                cluster_backend = cluster_backend,
                method = method,
                weight = weight,
                n_clusters = row_target_clusters,
                n_clusters_source = row_target_source,
                n_threads = n_threads,
                status = if (identical(graph_build$status, "expected_skip")) "expected_skip" else "failed",
                error = graph_build$error,
                graph_sec = graph_build$graph_sec,
                total_sec = graph_build$graph_sec,
                peak_rss_gb = read_peak_rss_gb(),
                graph_resolved_backend = if (identical(graph_build$status, "success")) graph_build$graph_resolved_backend else NA_character_,
                graph_preflight_route = graph_build$graph_preflight_route %||% graph_preflight_route,
                cluster_preflight_route = cluster_preflight_route,
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
                seed = cycle_seed,
                timeout = timeout,
                cycle = cycle
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
                "[%s] dataset=%s cycle=%s k=%s graph=%s cluster=%s method=%s status=%s total=%.3f\n",
                format(Sys.time(), "%Y-%m-%d %H:%M:%S"),
                dataset_name, cycle, k, graph_backend, cluster_backend, method,
                row$status, row$total_sec
              )
            )
          }
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
  best <- select_graph_best_rows(ok, group_cols = "dataset")
  utils::write.csv(best, file.path(out_dir, "graph_cluster_best_by_dataset.csv"), row.names = FALSE)
  best_by_target <- select_graph_best_rows(ok, group_cols = c("dataset", "k", "n_clusters_requested"))
  utils::write.csv(
    best_by_target,
    file.path(out_dir, "graph_cluster_best_by_dataset_k_target.csv"),
    row.names = FALSE
  )

  cycle_summary <- summarize_graph_cycles(ok)
  utils::write.csv(
    cycle_summary,
    file.path(out_dir, "graph_cluster_cycle_summary.csv"),
    row.names = FALSE
  )

  recommendations <- recommend_graph_cluster_methods(cycle_summary, ari_tolerance)
  if (nrow(recommendations)) {
    utils::write.csv(
      recommendations,
      file.path(out_dir, "graph_cluster_recommendations_from_cycles.csv"),
      row.names = FALSE
    )
  }

  auto_comparison <- compare_auto_graph_to_recommendations(cycle_summary, recommendations)
  if (nrow(auto_comparison)) {
    utils::write.csv(
      auto_comparison,
      file.path(out_dir, "graph_cluster_auto_vs_cycle_recommendation.csv"),
      row.names = FALSE
    )
  }
}

materials <- c(
  "# Graph Clustering Benchmark",
  "",
  "This benchmark evaluates faissR `knn_graph()` plus `graph_cluster()` on labelled benchmark datasets and simulated datasets.",
  "",
  sprintf("- Output directory: `%s`", out_dir),
  sprintf("- Data root: `%s`", data_root),
  sprintf("- Default real datasets: `%s`", paste(dataset_index(data_root)$dataset, collapse = "`, `")),
  "- Default simulated datasets: `SimulatedUniform2D`, `SimulatedUniform3D`",
  sprintf("- Methods: `%s`", paste(methods, collapse = "`, `")),
  sprintf("- Graph backends: `%s`", paste(graph_backends, collapse = "`, `")),
  sprintf("- Clustering backends: `%s`", paste(cluster_backends, collapse = "`, `")),
  sprintf("- k values: `%s`", paste(k_values, collapse = "`, `")),
  sprintf("- Cycles: `%s`", cycles),
  sprintf("- ARI tolerance for cycle recommendations: `%s`", ari_tolerance),
  sprintf("- CPU thread cap: `%s`", n_threads),
  sprintf("- Target clusters mode: `%s`", target_mode),
  "",
  "ARI is computed in `benchmark_scripts/source.R` from labels stored in each dataset object. ARI is `NA` when labels are unavailable.",
  "`graph_cluster_benchmark_config.csv` records the run configuration. `graph_cluster_benchmark_results.csv` is the raw row-level result table, including successes, failures, expected skips, graph timings, clustering timings, memory, ARI, modularity, and backend metadata.",
  "`target_clusters` is normalized to either `\"labels\"` or `\"none\"`; invalid values stop before the benchmark starts. When `target_clusters = \"labels\"`, Louvain and Leiden use `n_clusters = length(unique(labels))`. If a benchmark block contains only Louvain/Leiden, this target is stored on the graph with `knn_graph(n_clusters = ...)` and reused by `graph_cluster()`; mixed blocks that include random-walking pass the target only to Louvain/Leiden rows because random-walking intentionally has no cluster-count target.",
  "`n_clusters_requested` records the target community count used for Louvain/Leiden rows. `n_clusters_source` records whether that target came from dataset labels, a stored `knn_graph(n_clusters = ...)` target, or no target.",
  "Each KNN graph is built once per dataset/cycle/k/graph-backend/weight combination and reused across clustering methods and clustering backends within that cycle. The `cycle` column supports repeated benchmark cycles such as `--cycles=10`; `graph_cached` records reuse within a cycle, `graph_sec` is the graph construction time for the shared graph, `cluster_sec` is the clustering-only time, and `total_sec` is `graph_sec + cluster_sec`.",
  "`graph_cluster_best_by_dataset.csv` stores the best successful row per dataset after ranking by ARI, modularity, and total time for a compact backwards-compatible summary. `graph_cluster_best_by_dataset_k_target.csv` keeps the best successful row per dataset/k/target-cluster-count combination so different neighbourhood sizes and Louvain/Leiden target counts remain auditable.",
  "`graph_cluster_cycle_summary.csv` aggregates successful rows across cycles by dataset/k/graph-backend/cluster-backend/method/weight and reports success counts, median/min/max graph, clustering, and total time, ARI stability, modularity stability, community counts, and resolved backend metadata.",
  "`graph_cluster_recommendations_from_cycles.csv` selects the fastest graph/clustering/backend/method row within `ari_tolerance` of the best median ARI for each dataset/k/target-cluster-count combination and marks `recommendation_basis = \"fastest_within_ari_tolerance\"`; tied median total times are broken by higher median ARI and then higher median modularity. When ARI is unavailable it selects the fastest median total-time row and marks `recommendation_basis = \"speed_only_no_ari\"`.",
  "`graph_cluster_auto_vs_cycle_recommendation.csv` compares aggregate rows where graph or clustering backend was `auto` with those cycle-summary recommendations and reports the recommendation basis, median speed ratio, median ARI gap, modularity gap, and backend/method agreement. Speed ratios and quality gaps are `NA` when the required timing, ARI, or modularity values are unavailable or invalid.",
  "`graph_backend` and `cluster_backend` record the requested public backends. `graph_preflight_route` and `cluster_preflight_route` record the public resolver decision before runtime availability checks; `graph_resolved_backend` and `cluster_resolved_backend` record the resolved device policy from successful result objects, so CPU/CUDA rows can be audited without opening the R objects.",
  "Unsupported graph-clustering combinations known from the public API, such as CUDA random_walking, are recorded as `status = \"expected_skip\"` with `expected_skip = TRUE`. If every row in a graph-build block is an expected skip, graph construction is skipped and graph timing/edge columns remain `NA`.",
  "CUDA rows are recorded as expected skips when faissR was not built with the required CUDA/cuGraph support; unexpected CUDA runtime errors remain failed rows. The benchmark does not silently replace explicit CUDA clustering with CPU clustering."
)
writeLines(materials, file.path(out_dir, "MATERIALS_AND_METHODS_graph_clustering.md"))

cat("Saved graph clustering benchmark files in: ", out_dir, "\n", sep = "")
