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

required_positive_int_arg <- function(x, arg) {
  value <- suppressWarnings(as.numeric(x))
  if (length(value) != 1L || is.na(value) || !is.finite(value) || value < 1L ||
      abs(value - round(value)) > sqrt(.Machine$double.eps)) {
    stop("`", arg, "` must be a positive integer.", call. = FALSE)
  }
  as.integer(round(value))
}

required_nonnegative_numeric_arg <- function(x, arg) {
  value <- suppressWarnings(as.numeric(x))
  if (length(value) != 1L || is.na(value) || !is.finite(value) || value < 0) {
    stop("`", arg, "` must be a non-negative numeric value.", call. = FALSE)
  }
  value
}

default_kmeans_method_values <- function() {
  c("fast_kmeans", "stats")
}

default_kmeans_backend_values <- function() {
  c("auto", "cpu", "cuda")
}

valid_kmeans_tuning_values <- function() {
  c("auto", "fixed", "off", "none")
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

validate_kmeans_tuning_value <- function(value, arg_name = "tuning") {
  value <- trimws(as.character(value %||% "auto"))
  if (length(value) != 1L || is.na(value) || !nzchar(value)) {
    stop("`", arg_name, "` must be one of: ", paste(valid_kmeans_tuning_values(), collapse = ", "), ".", call. = FALSE)
  }
  value <- tolower(value)
  if (!value %in% valid_kmeans_tuning_values()) {
    stop(
      "`", arg_name, "` must be one of: ",
      paste(valid_kmeans_tuning_values(), collapse = ", "),
      ". Invalid value: ",
      value,
      ".",
      call. = FALSE
    )
  }
  value
}

validate_dataset_values <- function(datasets, valid_datasets, arg_name = "datasets") {
  datasets <- unique(trimws(as.character(datasets)))
  datasets <- datasets[nzchar(datasets)]
  valid_datasets <- unique(trimws(as.character(valid_datasets)))
  valid_datasets <- valid_datasets[nzchar(valid_datasets)]
  invalid <- datasets[!datasets %in% valid_datasets]
  if (length(invalid)) {
    stop(
      "`", arg_name, "` must contain only available benchmark datasets: ",
      paste(valid_datasets, collapse = ", "),
      ". Invalid value(s): ",
      paste(invalid, collapse = ", "),
      ".",
      call. = FALSE
    )
  }
  if (!length(datasets)) {
    stop("`", arg_name, "` must contain at least one dataset.", call. = FALSE)
  }
  datasets
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
  tuning <- validate_kmeans_tuning_value(tuning)
  helper <- tryCatch(
    getFromNamespace("kmeans_auto_params", "faissR"),
    error = function(e) NULL
  )
  if (is.function(helper)) {
    return(helper(n = n, p = p, centers = centers, tuning = tuning))
  }
  work <- as.double(n) * as.double(p) * as.double(centers)
  backend_policy <- kmeans_auto_backend_policy(n, p, centers)
  high_dim <- p >= 256L
  large_n <- n >= 100000L
  many_centers <- centers >= 100L
  n_per_center <- as.double(n) / as.double(centers)
  small_many_centers <- many_centers && n <= 50000L && work <= 2e8 && n_per_center >= 20
  few_points_many_centers <- many_centers && n <= 50000L && work <= 2e8 && n_per_center < 20
  rule_detail <- paste(
    paste0("n=", n),
    paste0("p=", p),
    paste0("centers=", centers),
    paste0("n_per_center=", formatC(n_per_center, digits = 4, format = "fg")),
    paste0("work=", format(work, scientific = TRUE)),
    sep = ";"
  )
  if (!identical(tuning, "auto")) {
    return(list(
      policy = tuning,
      max_iter = 100L,
      n_init = 1L,
      tol = 1e-4,
      work = as.numeric(work),
      n_per_center = as.numeric(n_per_center),
      high_dim = isTRUE(p >= 256L),
      large_n = isTRUE(n >= 100000L),
      many_centers = isTRUE(centers >= 100L),
      small_many_centers = isTRUE(small_many_centers),
      few_points_many_centers = isTRUE(few_points_many_centers),
      backend_policy = backend_policy,
      rule = "fixed_defaults",
      rule_detail = rule_detail
    ))
  }
  if (centers == 1L) {
    return(list(
      policy = "auto",
      max_iter = 1L,
      n_init = 1L,
      tol = 0,
      work = as.numeric(work),
      n_per_center = as.numeric(n_per_center),
      high_dim = isTRUE(high_dim),
      large_n = isTRUE(large_n),
      many_centers = FALSE,
      small_many_centers = FALSE,
      few_points_many_centers = FALSE,
      backend_policy = backend_policy,
      rule = "single_cluster_exact_mean",
      rule_detail = rule_detail
    ))
  }
  max_iter <- if (large_n || work >= 5e9) {
    50L
  } else if (high_dim || (many_centers && !small_many_centers && !few_points_many_centers) || work >= 5e8) {
    75L
  } else {
    100L
  }
  n_init <- if (n <= 50000L && centers <= 20L && work <= 2e8) {
    5L
  } else if (small_many_centers) {
    3L
  } else if (few_points_many_centers) {
    3L
  } else if (n <= 100000L && centers <= 50L && work <= 5e8) {
    3L
  } else {
    1L
  }
  tol <- if (large_n || work >= 5e9) 1e-3 else 1e-4
  rule <- if (large_n || work >= 5e9) {
    "large_fast_convergence"
  } else if (small_many_centers) {
    "small_many_centers_multistart"
  } else if (few_points_many_centers) {
    "few_points_many_centers_multistart"
  } else if (n <= 50000L && centers <= 20L && work <= 2e8) {
    "small_low_work_multistart"
  } else if (n <= 100000L && centers <= 50L && work <= 5e8) {
    "medium_multistart"
  } else if (high_dim || many_centers || work >= 5e8) {
    "medium_single_start"
  } else {
    "small_single_start"
  }
  list(
    policy = "auto",
    max_iter = as.integer(max_iter),
    n_init = as.integer(n_init),
    tol = as.numeric(tol),
    work = as.numeric(work),
    n_per_center = as.numeric(n_per_center),
    high_dim = isTRUE(high_dim),
    large_n = isTRUE(large_n),
    many_centers = isTRUE(many_centers),
    small_many_centers = isTRUE(small_many_centers),
    few_points_many_centers = isTRUE(few_points_many_centers),
    backend_policy = backend_policy,
    rule = rule,
    rule_detail = rule_detail
  )
}

kmeans_auto_backend_policy <- function(n, p, centers) {
  work_threshold <- 1e8
  nbytes_threshold <- 256 * 1024^2
  large_n_threshold <- 50000
  large_p_threshold <- 128
  helper <- tryCatch(
    getFromNamespace("kmeans_auto_backend_policy", "faissR"),
    error = function(e) NULL
  )
  if (is.function(helper)) {
    return(helper(n = n, p = p, centers = centers))
  }
  if (is.null(n) || is.null(p) || is.null(centers)) {
    return(list(
      prefer_cuda = TRUE,
      reason = "unknown_shape",
      work = NA_real_,
      nbytes = NA_real_,
      n_per_center = NA_real_,
      work_threshold = work_threshold,
      nbytes_threshold = nbytes_threshold,
      large_n_threshold = large_n_threshold,
      large_p_threshold = large_p_threshold
    ))
  }
  n <- suppressWarnings(as.double(n))
  p <- suppressWarnings(as.double(p))
  centers <- suppressWarnings(as.double(centers))
  if (length(n) != 1L || length(p) != 1L || length(centers) != 1L ||
      !is.finite(n) || !is.finite(p) || !is.finite(centers) ||
      n <= 0 || p <= 0 || centers <= 0) {
    return(list(
      prefer_cuda = TRUE,
      reason = "invalid_shape_assume_cuda_capable",
      work = NA_real_,
      nbytes = NA_real_,
      n_per_center = NA_real_,
      work_threshold = work_threshold,
      nbytes_threshold = nbytes_threshold,
      large_n_threshold = large_n_threshold,
      large_p_threshold = large_p_threshold
    ))
  }
  work <- n * p * centers
  nbytes <- n * p * 8
  n_per_center <- n / centers
  if (centers == 1) {
    return(list(
      prefer_cuda = FALSE,
      reason = "single_cluster_exact_mean",
      work = as.numeric(work),
      nbytes = as.numeric(nbytes),
      n_per_center = as.numeric(n_per_center),
      work_threshold = work_threshold,
      nbytes_threshold = nbytes_threshold,
      large_n_threshold = large_n_threshold,
      large_p_threshold = large_p_threshold
    ))
  }
  prefer <- work >= work_threshold ||
    nbytes >= nbytes_threshold ||
    (n >= large_n_threshold && p >= large_p_threshold)
  reason <- if (work >= work_threshold) {
    "work_at_least_1e8"
  } else if (nbytes >= nbytes_threshold) {
    "input_at_least_256MiB"
  } else if (n >= large_n_threshold && p >= large_p_threshold) {
    "large_high_dimensional_input"
  } else {
    "small_cpu_preferred"
  }
  list(
    prefer_cuda = isTRUE(prefer),
    reason = reason,
    work = as.numeric(work),
    nbytes = as.numeric(nbytes),
    n_per_center = as.numeric(n_per_center),
    work_threshold = work_threshold,
    nbytes_threshold = nbytes_threshold,
    large_n_threshold = large_n_threshold,
    large_p_threshold = large_p_threshold
  )
}

kmeans_auto_prefers_cuda <- function(n, p, centers) {
  helper <- tryCatch(
    getFromNamespace("kmeans_auto_prefers_cuda", "faissR"),
    error = function(e) NULL
  )
  if (is.function(helper)) {
    return(isTRUE(helper(n = n, p = p, centers = centers)))
  }
  isTRUE(kmeans_auto_backend_policy(n, p, centers)$prefer_cuda)
}

resolve_kmeans_int <- function(x, fallback) {
  if (is.character(x) && length(x) == 1L && identical(tolower(x), "auto")) return(as.integer(fallback))
  out <- suppressWarnings(as.numeric(x))
  if (length(out) != 1L || is.na(out) || !is.finite(out) || out < 1L ||
      abs(out - round(out)) > sqrt(.Machine$double.eps)) {
    stop("k-means integer tuning values must be positive integers or `auto`.", call. = FALSE)
  }
  as.integer(round(out))
}

resolve_kmeans_tol <- function(x, fallback) {
  if (is.character(x) && length(x) == 1L && identical(tolower(x), "auto")) return(as.numeric(fallback))
  out <- suppressWarnings(as.numeric(x))
  if (length(out) != 1L || is.na(out) || !is.finite(out) || out < 0) {
    stop("k-means tolerance must be a non-negative numeric value or `auto`.", call. = FALSE)
  }
  as.numeric(out)
}

kmeans_hit_max_iter <- function(iter, max_iter) {
  helper <- tryCatch(
    getFromNamespace("kmeans_hit_max_iter", "faissR"),
    error = function(e) NULL
  )
  if (is.function(helper)) return(helper(iter, max_iter))
  iter <- suppressWarnings(as.integer(iter))
  max_iter <- suppressWarnings(as.integer(max_iter))
  if (length(iter) != 1L || length(max_iter) != 1L ||
      is.na(iter) || is.na(max_iter) || max_iter < 1L) {
    return(NA)
  }
  iter >= max_iter
}

infer_kmeans_resolved_backend <- function(backend, backend_used, method) {
  if (identical(method, "stats")) return("stats")
  backend <- tolower(as.character(backend)[1L])
  backend_used <- tolower(as.character(backend_used)[1L])
  if (backend %in% c("cpu", "cuda")) return(backend)
  if (backend_used %in% c("cuda_faiss", "cuda_cuvs")) return("cuda")
  if (backend_used %in% c("faiss", "cpu")) return("cpu")
  NA_character_
}

result_row <- function(dataset, n, p, method, backend, centers, cycle, n_threads,
                       status, error = NA_character_, elapsed_sec = NA_real_,
                       peak_rss_gb = NA_real_, backend_used = NA_character_,
                       requested_backend = NA_character_,
                       resolved_backend = NA_character_,
                       iter = NA_integer_, tot_withinss = NA_real_,
                       ari = NA_real_, max_iter = NA_integer_,
                       converged = NA, hit_max_iter = NA,
                       n_init = NA_integer_, tol = NA_real_,
                       tuning_policy = NA_character_,
                       tuning_rule = NA_character_,
                       tuning_rule_detail = NA_character_,
                       tuning_work = NA_real_,
                       tuning_n_per_center = NA_real_,
                       tuning_high_dim = NA,
                       tuning_large_n = NA,
                       tuning_many_centers = NA,
                       tuning_small_many_centers = NA,
                       tuning_few_points_many_centers = NA,
                       selection_policy = NA_character_,
                       selection_slow_tuning = NA,
                       selection_predicted_backend = NA_character_,
                       selection_reason = NA_character_,
                       selection_work = NA_real_,
                       selection_nbytes = NA_real_,
                       selection_n_per_center = NA_real_,
                       selection_cuda_available = NA,
                       selection_faiss_gpu_available = NA,
                       selection_cuvs_available = NA,
                       expected_skip = FALSE) {
  data.frame(
    dataset = dataset,
    n = as.integer(n),
    p = as.integer(p),
    method = method,
    backend = backend,
    centers = as.integer(centers),
    cycle = as.integer(cycle),
    n_threads = as.integer(n_threads),
    status = status,
    error = error,
    elapsed_sec = elapsed_sec,
    peak_rss_gb = peak_rss_gb,
    backend_used = backend_used,
    requested_backend = requested_backend,
    resolved_backend = resolved_backend,
    iter = as.integer(iter),
    tot_withinss = tot_withinss,
    ari = ari,
    max_iter = as.integer(max_iter),
    converged = as.logical(converged),
    hit_max_iter = as.logical(hit_max_iter),
    n_init = as.integer(n_init),
    tol = tol,
    tuning_policy = tuning_policy,
    tuning_rule = tuning_rule,
    tuning_rule_detail = tuning_rule_detail,
    tuning_work = tuning_work,
    tuning_n_per_center = tuning_n_per_center,
    tuning_high_dim = as.logical(tuning_high_dim),
    tuning_large_n = as.logical(tuning_large_n),
    tuning_many_centers = as.logical(tuning_many_centers),
    tuning_small_many_centers = as.logical(tuning_small_many_centers),
    tuning_few_points_many_centers = as.logical(tuning_few_points_many_centers),
    selection_policy = selection_policy,
    selection_slow_tuning = as.logical(selection_slow_tuning),
    selection_predicted_backend = selection_predicted_backend,
    selection_reason = selection_reason,
    selection_work = selection_work,
    selection_nbytes = selection_nbytes,
    selection_n_per_center = selection_n_per_center,
    selection_cuda_available = as.logical(selection_cuda_available),
    selection_faiss_gpu_available = as.logical(selection_faiss_gpu_available),
    selection_cuvs_available = as.logical(selection_cuvs_available),
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

any_true_or_na <- function(x) {
  x <- suppressWarnings(as.logical(x))
  x <- x[!is.na(x)]
  if (!length(x)) return(NA)
  any(x)
}

all_true_or_na <- function(x) {
  x <- suppressWarnings(as.logical(x))
  x <- x[!is.na(x)]
  if (!length(x)) return(NA)
  all(x)
}

summarize_kmeans_cycles <- function(ok) {
  if (!"hit_max_iter" %in% names(ok)) ok$hit_max_iter <- NA
  if (!"converged" %in% names(ok)) ok$converged <- NA
  parts <- split(ok, paste(ok$dataset, ok$method, ok$backend, ok$centers, sep = "__"))
  summary <- lapply(parts, function(x) {
    data.frame(
      dataset = x$dataset[[1L]],
      method = x$method[[1L]],
      backend = x$backend[[1L]],
      centers = as.integer(x$centers[[1L]]),
      n = as.integer(x$n[[1L]]),
      p = as.integer(x$p[[1L]]),
      n_threads = as.integer(x$n_threads[[1L]]),
      success_cycles = length(unique(x$cycle)),
      success_rows = nrow(x),
      median_elapsed_sec = finite_median(x$elapsed_sec),
      min_elapsed_sec = finite_min(x$elapsed_sec),
      max_elapsed_sec = finite_max(x$elapsed_sec),
      median_ari = finite_median(x$ari),
      min_ari = finite_min(x$ari),
      max_ari = finite_max(x$ari),
      median_tot_withinss = finite_median(x$tot_withinss),
      min_tot_withinss = finite_min(x$tot_withinss),
      max_tot_withinss = finite_max(x$tot_withinss),
      median_iter = finite_median(x$iter),
      median_max_iter = finite_median(x$max_iter),
      any_hit_max_iter = any_true_or_na(x$hit_max_iter),
      all_converged = all_true_or_na(x$converged),
      median_n_init = finite_median(x$n_init),
      median_tol = finite_median(x$tol),
      median_tuning_work = finite_median(x$tuning_work),
      median_tuning_n_per_center = finite_median(x$tuning_n_per_center),
      backend_used = dominant_value(x$backend_used),
      requested_backend = dominant_value(x$requested_backend),
      resolved_backend = dominant_value(x$resolved_backend),
      tuning_policy = dominant_value(x$tuning_policy),
      tuning_rule = dominant_value(x$tuning_rule),
      tuning_rule_detail = dominant_value(x$tuning_rule_detail),
      tuning_high_dim = any(x$tuning_high_dim %in% TRUE, na.rm = TRUE),
      tuning_large_n = any(x$tuning_large_n %in% TRUE, na.rm = TRUE),
      tuning_many_centers = any(x$tuning_many_centers %in% TRUE, na.rm = TRUE),
      tuning_small_many_centers = any(x$tuning_small_many_centers %in% TRUE, na.rm = TRUE),
      tuning_few_points_many_centers = any(x$tuning_few_points_many_centers %in% TRUE, na.rm = TRUE),
      selection_policy = dominant_value(x$selection_policy),
      selection_slow_tuning = any(x$selection_slow_tuning %in% TRUE, na.rm = TRUE),
      selection_predicted_backend = dominant_value(x$selection_predicted_backend),
      selection_reason = dominant_value(x$selection_reason),
      median_selection_work = finite_median(x$selection_work),
      median_selection_nbytes = finite_median(x$selection_nbytes),
      median_selection_n_per_center = finite_median(x$selection_n_per_center),
      selection_cuda_available = any(x$selection_cuda_available %in% TRUE, na.rm = TRUE),
      selection_faiss_gpu_available = any(x$selection_faiss_gpu_available %in% TRUE, na.rm = TRUE),
      selection_cuvs_available = any(x$selection_cuvs_available %in% TRUE, na.rm = TRUE),
      stringsAsFactors = FALSE
    )
  })
  out <- do.call(rbind, summary)
  out[order(out$dataset, -out$median_ari, out$median_elapsed_sec), , drop = FALSE]
}

recommend_kmeans_methods <- function(cycle_summary, ari_tolerance, group_cols = c("dataset", "centers")) {
  missing_cols <- setdiff(group_cols, names(cycle_summary))
  if (length(missing_cols)) {
    stop("Missing k-means recommendation grouping columns: ", paste(missing_cols, collapse = ", "), call. = FALSE)
  }
  group_key <- do.call(
    paste,
    c(
      lapply(cycle_summary[, group_cols, drop = FALSE], function(x) {
        x <- as.character(x)
        x[is.na(x)] <- "NA"
        x
      }),
      sep = "__"
    )
  )
  parts <- split(cycle_summary, group_key)
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
    withinss <- if ("median_tot_withinss" %in% names(candidates)) {
      candidates$median_tot_withinss
    } else {
      rep(Inf, nrow(candidates))
    }
    candidates <- candidates[order(
      candidates$median_elapsed_sec,
      -ifelse(is.finite(candidates$median_ari), candidates$median_ari, -Inf),
      ifelse(is.finite(withinss), withinss, Inf)
    ), , drop = FALSE]
    candidates[1L, , drop = FALSE]
  })
  out <- do.call(rbind, recommendations)
  row.names(out) <- NULL
  order_cols <- group_cols[group_cols %in% names(out)]
  out[do.call(order, out[, order_cols, drop = FALSE]), , drop = FALSE]
}

compare_fast_kmeans_to_recommendations <- function(cycle_summary, recommendations) {
  if (!nrow(recommendations)) return(recommendations)
  fast <- cycle_summary[cycle_summary$method == "fast_kmeans", , drop = FALSE]
  if (!nrow(fast)) return(data.frame())
  keys <- c("dataset", "centers")
  keep <- c(
    keys, "method", "backend", "backend_used", "requested_backend",
    "resolved_backend", "n_threads", "success_cycles", "median_elapsed_sec",
    "median_ari", "min_ari", "median_tot_withinss", "median_iter",
    "median_max_iter", "any_hit_max_iter", "all_converged",
    "median_n_init", "median_tol", "tuning_policy",
    "tuning_rule", "tuning_rule_detail",
    "median_tuning_work", "median_tuning_n_per_center",
    "tuning_high_dim", "tuning_large_n", "tuning_many_centers",
    "tuning_small_many_centers", "tuning_few_points_many_centers",
    "selection_policy", "selection_slow_tuning",
    "selection_predicted_backend", "selection_reason", "median_selection_work",
    "median_selection_nbytes", "median_selection_n_per_center",
    "selection_cuda_available", "selection_faiss_gpu_available",
    "selection_cuvs_available"
  )
  keep <- keep[keep %in% names(cycle_summary)]
  rec_keep <- c(keep, "recommendation_basis")
  fast <- fast[, keep, drop = FALSE]
  recommendations <- recommendations[, rec_keep, drop = FALSE]
  names(fast)[match(keep[-seq_along(keys)], names(fast))] <- paste0("fast_", keep[-seq_along(keys)])
  names(recommendations)[match(rec_keep[-seq_along(keys)], names(recommendations))] <- paste0("recommended_", rec_keep[-seq_along(keys)])
  comparison <- merge(fast, recommendations, by = keys, all = FALSE)
  if (!nrow(comparison)) return(comparison)
  comparison$fast_is_recommended_method <- comparison$fast_method == comparison$recommended_method
  comparison$fast_uses_recommended_backend <- comparison$fast_backend == comparison$recommended_backend
  comparison$fast_uses_recommended_implementation <- comparison$fast_backend_used == comparison$recommended_backend_used
  comparison$fast_median_speed_ratio <- safe_positive_ratio(
    comparison$fast_median_elapsed_sec,
    comparison$recommended_median_elapsed_sec
  )
  comparison$fast_median_ari_gap <- safe_difference(
    comparison$recommended_median_ari,
    comparison$fast_median_ari
  )
  comparison$fast_withinss_ratio <- safe_positive_ratio(
    comparison$fast_median_tot_withinss,
    comparison$recommended_median_tot_withinss
  )
  comparison[order(comparison$dataset, comparison$centers, comparison$fast_backend), , drop = FALSE]
}

compare_auto_kmeans_to_recommendations <- function(cycle_summary, recommendations) {
  if (!nrow(recommendations)) return(recommendations)
  auto <- cycle_summary[
    cycle_summary$method == "fast_kmeans" & cycle_summary$backend == "auto",
    ,
    drop = FALSE
  ]
  if (!nrow(auto)) return(data.frame())
  keys <- c("dataset", "centers")
  keep <- c(
    keys, "method", "backend", "backend_used", "requested_backend",
    "resolved_backend", "n_threads", "success_cycles", "median_elapsed_sec",
    "median_ari", "min_ari", "median_tot_withinss", "median_iter",
    "median_max_iter", "any_hit_max_iter", "all_converged",
    "median_n_init", "median_tol", "tuning_policy",
    "tuning_rule", "tuning_rule_detail",
    "median_tuning_work", "median_tuning_n_per_center",
    "tuning_high_dim", "tuning_large_n", "tuning_many_centers",
    "tuning_small_many_centers", "tuning_few_points_many_centers",
    "selection_policy", "selection_slow_tuning",
    "selection_predicted_backend", "selection_reason", "median_selection_work",
    "median_selection_nbytes", "median_selection_n_per_center",
    "selection_cuda_available", "selection_faiss_gpu_available",
    "selection_cuvs_available"
  )
  keep <- keep[keep %in% names(cycle_summary)]
  rec_keep <- c(keep, "recommendation_basis")
  auto <- auto[, keep, drop = FALSE]
  recommendations <- recommendations[, rec_keep, drop = FALSE]
  names(auto)[match(keep[-seq_along(keys)], names(auto))] <- paste0("auto_", keep[-seq_along(keys)])
  names(recommendations)[match(rec_keep[-seq_along(keys)], names(recommendations))] <- paste0("recommended_", rec_keep[-seq_along(keys)])
  comparison <- merge(auto, recommendations, by = keys, all = FALSE)
  if (!nrow(comparison)) return(comparison)
  comparison$auto_is_recommended_method <- comparison$auto_method == comparison$recommended_method
  comparison$auto_uses_recommended_requested_backend <- comparison$auto_backend == comparison$recommended_backend
  comparison$auto_uses_recommended_resolved_backend <- comparison$auto_resolved_backend == comparison$recommended_resolved_backend
  comparison$auto_uses_recommended_implementation <- comparison$auto_backend_used == comparison$recommended_backend_used
  comparison$auto_median_speed_ratio <- safe_positive_ratio(
    comparison$auto_median_elapsed_sec,
    comparison$recommended_median_elapsed_sec
  )
  comparison$auto_median_ari_gap <- safe_difference(
    comparison$recommended_median_ari,
    comparison$auto_median_ari
  )
  comparison$auto_withinss_ratio <- safe_positive_ratio(
    comparison$auto_median_tot_withinss,
    comparison$recommended_median_tot_withinss
  )
  comparison[order(comparison$dataset, comparison$centers), , drop = FALSE]
}

compare_fast_kmeans_to_stats <- function(ok) {
  fast_rows <- ok[ok$method == "fast_kmeans", , drop = FALSE]
  stats_rows <- ok[ok$method == "stats", , drop = FALSE]
  if (!nrow(fast_rows) || !nrow(stats_rows)) return(data.frame())
  comparison <- merge(
    fast_rows,
    stats_rows[, c("dataset", "centers", "cycle", "elapsed_sec", "tot_withinss", "ari", "iter"), drop = FALSE],
    by = c("dataset", "centers", "cycle"),
    suffixes = c("_fast", "_stats"),
    all = FALSE
  )
  if (!nrow(comparison)) return(comparison)
  comparison$speedup_vs_stats <- safe_positive_ratio(
    comparison$elapsed_sec_stats,
    comparison$elapsed_sec_fast
  )
  comparison$ari_delta_vs_stats <- safe_difference(comparison$ari_fast, comparison$ari_stats)
  comparison$withinss_ratio_vs_stats <- safe_positive_ratio(
    comparison$tot_withinss_fast,
    comparison$tot_withinss_stats
  )
  comparison[order(comparison$dataset, comparison$backend), , drop = FALSE]
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

kmeans_rank_value <- function(data, column, default, higher_is_better = FALSE) {
  value <- if (column %in% names(data)) data[[column]] else rep(default, nrow(data))
  value <- suppressWarnings(as.numeric(value))
  value[!is.finite(value)] <- default
  if (higher_is_better) -value else value
}

rank_kmeans_success <- function(ok) {
  if (!nrow(ok)) return(ok)
  ok[order(
    ok$dataset,
    kmeans_rank_value(ok, "ari", -Inf, higher_is_better = TRUE),
    kmeans_rank_value(ok, "elapsed_sec", Inf),
    kmeans_rank_value(ok, "tot_withinss", Inf)
  ), , drop = FALSE]
}

select_kmeans_best_rows <- function(ok, group_cols = c("dataset")) {
  ranked <- rank_kmeans_success(ok)
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

kmeans_faiss_gpu_available <- function() {
  helper <- tryCatch(
    getFromNamespace("faiss_gpu_available", "faissR"),
    error = function(e) NULL
  )
  is.function(helper) && isTRUE(helper())
}

kmeans_cuda_runtime_reason <- function(cuda_available_value = faissR::cuda_available(),
                                       faiss_gpu_available_value = kmeans_faiss_gpu_available(),
                                       cuvs_available_value = faissR::cuvs_available()) {
  if (!isTRUE(cuda_available_value)) {
    return("missing_cuda_runtime")
  }
  if (!isTRUE(faiss_gpu_available_value) && !isTRUE(cuvs_available_value)) {
    return("missing_gpu_kmeans_backend")
  }
  "available"
}

kmeans_cuda_runtime_notes <- function(reason) {
  reason <- as.character(reason)[1L]
  switch(
    reason,
    available = "CUDA k-means route is available.",
    missing_cuda_runtime = "CUDA k-means requires a CUDA runtime/device; CUDA is unavailable in the current runtime.",
    missing_gpu_kmeans_backend = "CUDA k-means requires FAISS GPU k-means or direct cuVS k-means; neither route is available in this build/runtime.",
    "CUDA k-means requires a CUDA runtime plus FAISS GPU k-means or direct cuVS k-means."
  )
}

kmeans_runtime_capabilities <- function() {
  cuda_runtime <- isTRUE(faissR::cuda_available())
  faiss_gpu <- isTRUE(kmeans_faiss_gpu_available())
  cuvs <- isTRUE(faissR::cuvs_available())
  cuda_reason <- kmeans_cuda_runtime_reason(
    cuda_available_value = cuda_runtime,
    faiss_gpu_available_value = faiss_gpu,
    cuvs_available_value = cuvs
  )
  cuda_ok <- identical(cuda_reason, "available")
  data.frame(
    method = c("fast_kmeans", "fast_kmeans", "fast_kmeans", "stats"),
    backend = c("auto", "cpu", "cuda", "stats"),
    supported = c(TRUE, TRUE, TRUE, TRUE),
    runtime_available = c(TRUE, TRUE, cuda_ok, TRUE),
    resolved_backend = c(
      if (cuda_ok) "shape_aware_auto_cpu_or_cuda" else "cpu",
      "cpu",
      "cuda",
      "stats"
    ),
    implementation = c(
      if (cuda_ok) "faiss CPU/FAISS GPU/cuVS selected by shape gate" else "faiss/native CPU",
      "faiss/native CPU",
      "FAISS GPU k-means or direct cuVS k-means",
      "stats::kmeans"
    ),
    runtime_reason = c(
      if (cuda_ok) "auto_shape_gate_cuda_available" else "auto_resolves_cpu_no_cuda_kmeans",
      "available",
      cuda_reason,
      "available"
    ),
    runtime_notes = c(
      if (cuda_ok) {
        "Auto may select CUDA for sufficiently large shape/work estimates."
      } else {
        "Auto resolves to CPU because no k-means-capable CUDA route is available."
      },
      "CPU k-means route is available.",
      kmeans_cuda_runtime_notes(cuda_reason),
      "Base stats::kmeans is available."
    ),
    cuda_available = c(cuda_runtime, cuda_runtime, cuda_runtime, cuda_runtime),
    faiss_gpu_available = c(faiss_gpu, faiss_gpu, faiss_gpu, faiss_gpu),
    cuvs_available = c(cuvs, cuvs, cuvs, cuvs),
    stringsAsFactors = FALSE
  )
}

kmeans_runtime_status <- function(method, backend, caps = kmeans_runtime_capabilities()) {
  method <- tolower(as.character(method)[1L])
  backend <- tolower(as.character(backend)[1L])
  hit <- caps[caps$method == method & caps$backend == backend, , drop = FALSE]
  if (!nrow(hit)) return(NULL)
  hit[1L, , drop = FALSE]
}

kmeans_expected_skip <- function(method, backend) {
  method <- tolower(as.character(method)[1L])
  backend <- tolower(as.character(backend)[1L])
  if (!identical(method, "fast_kmeans")) return(NULL)
  status <- kmeans_runtime_status(method, backend)
  if (!is.null(status) && !isTRUE(status$runtime_available[[1L]])) {
    return(status$runtime_notes[[1L]] %||% paste(
      "CUDA k-means requires a CUDA runtime plus FAISS GPU k-means or direct cuVS k-means;",
      "explicit CUDA requests are expected skips in this faissR build/runtime."
    ))
  }
  NULL
}

run_one <- function(x, labels, dataset_name, method, backend, centers,
                    cycle, n_threads, seed, timeout, max_iter, n_init, tol, tuning) {
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
    tuning_meta <- params$tuning %||% auto_params
    selection <- tuning_meta$selection %||% list()
    backend_used <- fit$backend %||% if (identical(method, "stats")) "stats" else NA_character_
    result_row(
      dataset = dataset_name,
      n = nrow(x),
      p = ncol(x),
      method = method,
      backend = backend,
      centers = centers,
      cycle = cycle,
      n_threads = n_threads,
      status = "success",
      elapsed_sec = elapsed,
      peak_rss_gb = read_peak_rss_gb(),
      backend_used = backend_used,
      requested_backend = params$requested_backend %||% if (identical(method, "stats")) "stats" else backend,
      resolved_backend = params$resolved_backend %||% infer_kmeans_resolved_backend(backend, backend_used, method),
      iter = fit$iter %||% NA_integer_,
      tot_withinss = fit$tot.withinss %||% NA_real_,
      ari = benchmark_adjusted_rand_index(labels, fit$cluster),
      max_iter = params$max_iter %||% resolved_max_iter,
      converged = params$converged %||% fit$converged %||% {
        hit <- kmeans_hit_max_iter(fit$iter %||% NA_integer_, params$max_iter %||% resolved_max_iter)
        if (is.na(hit)) NA else !isTRUE(hit)
      },
      hit_max_iter = params$hit_max_iter %||% fit$hit_max_iter %||%
        kmeans_hit_max_iter(fit$iter %||% NA_integer_, params$max_iter %||% resolved_max_iter),
      n_init = params$n_init %||% resolved_n_init,
      tol = params$tol %||% resolved_tol,
      tuning_policy = tuning_meta$policy %||% if (identical(method, "stats")) "stats" else NA_character_,
      tuning_rule = tuning_meta$rule %||% if (identical(method, "stats")) "stats_kmeans" else NA_character_,
      tuning_rule_detail = tuning_meta$rule_detail %||% if (identical(method, "stats")) "stats_kmeans" else NA_character_,
      tuning_work = tuning_meta$work %||% NA_real_,
      tuning_n_per_center = tuning_meta$n_per_center %||% NA_real_,
      tuning_high_dim = tuning_meta$high_dim %||% NA,
      tuning_large_n = tuning_meta$large_n %||% NA,
      tuning_many_centers = tuning_meta$many_centers %||% NA,
      tuning_small_many_centers = tuning_meta$small_many_centers %||% NA,
      tuning_few_points_many_centers = tuning_meta$few_points_many_centers %||% NA,
      selection_policy = selection$policy %||% if (identical(method, "stats")) "stats" else NA_character_,
      selection_slow_tuning = selection$slow_tuning %||% if (identical(method, "stats")) FALSE else NA,
      selection_predicted_backend = selection$predicted_backend %||% if (identical(method, "stats")) "stats" else NA_character_,
      selection_reason = selection$backend_policy_reason %||% if (identical(method, "stats")) "stats_kmeans" else NA_character_,
      selection_work = selection$work %||% NA_real_,
      selection_nbytes = selection$nbytes %||% NA_real_,
      selection_n_per_center = selection$n_per_center %||% NA_real_,
      selection_cuda_available = selection$cuda_available %||% NA,
      selection_faiss_gpu_available = selection$faiss_gpu_available %||% NA,
      selection_cuvs_available = selection$cuvs_available %||% NA
    )
  }, error = function(e) {
    result_row(
      dataset = dataset_name,
      n = nrow(x),
      p = ncol(x),
      method = method,
      backend = backend,
      centers = centers,
      cycle = cycle,
      n_threads = n_threads,
      status = "failed",
      error = conditionMessage(e),
      elapsed_sec = proc.time()[["elapsed"]] - started,
      peak_rss_gb = read_peak_rss_gb(),
      requested_backend = if (identical(method, "stats")) "stats" else backend,
      resolved_backend = if (identical(method, "stats")) "stats" else NA_character_
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

n_threads <- required_positive_int_arg(args$threads %||% 4L, "threads")
configure_threads(n_threads)
seed <- required_positive_int_arg(args$seed %||% 1L, "seed")
timeout <- required_positive_int_arg(args$timeout %||% 600L, "timeout")
fallback_centers <- required_positive_int_arg(args$centers %||% 10L, "centers")
cycles <- required_positive_int_arg(args$cycles %||% 1L, "cycles")
ari_tolerance <- required_nonnegative_numeric_arg(args$ari_tolerance %||% "0.01", "ari_tolerance")
max_iter <- args$max_iter %||% "auto"
n_init <- args$n_init %||% "auto"
tol <- args$tol %||% "auto"
tuning <- validate_kmeans_tuning_value(args$tuning %||% "auto")

available_datasets <- c(dataset_index(data_root)$dataset, "SimulatedTiny3Clusters")
datasets <- validate_dataset_values(
  split_arg(args$datasets, paste(available_datasets, collapse = ",")),
  available_datasets
)
methods <- validate_choice_values(
  split_arg(args$methods, paste(default_kmeans_method_values(), collapse = ",")),
  default_kmeans_method_values(),
  "methods"
)
backends <- validate_choice_values(
  split_arg(args$backends, paste(default_kmeans_backend_values(), collapse = ",")),
  default_kmeans_backend_values(),
  "backends"
)

suppressPackageStartupMessages(library(faissR))

config <- data.frame(
  key = c("data_root", "out_dir", "available_datasets", "datasets", "methods",
          "backends", "centers", "threads", "timeout", "cycles", "ari_tolerance",
          "max_iter", "n_init", "tol", "tuning", "seed"),
  value = c(data_root, out_dir, paste(available_datasets, collapse = ","),
            paste(datasets, collapse = ","), paste(methods, collapse = ","),
            paste(backends, collapse = ","), fallback_centers, n_threads, timeout,
            cycles, ari_tolerance, max_iter, n_init, tol, tuning, seed),
  stringsAsFactors = FALSE
)
utils::write.csv(config, file.path(out_dir, "kmeans_benchmark_config.csv"), row.names = FALSE)
kmeans_capabilities <- kmeans_runtime_capabilities()
utils::write.csv(kmeans_capabilities, file.path(out_dir, "kmeans_runtime_capabilities.csv"), row.names = FALSE)

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
      cycle = NA_integer_,
      n_threads = n_threads,
      status = "failed",
      error = conditionMessage(loaded)
    )
    next
  }
  x <- loaded$data
  centers <- label_center_count(loaded$labels, fallback_centers)
  for (cycle in seq_len(cycles)) {
    cycle_seed <- seed + (cycle - 1L) * 1000003L
    for (method in methods) {
      method_backends <- if (identical(method, "stats")) "stats" else backends
      for (backend in method_backends) {
        row_id <- row_id + 1L
        skip_reason <- kmeans_expected_skip(method, backend)
        if (!is.null(skip_reason)) {
          auto_params <- kmeans_auto_params(nrow(x), ncol(x), centers, tuning)
          backend_policy <- kmeans_auto_backend_policy(nrow(x), ncol(x), centers)
          row <- result_row(
            dataset = dataset_name,
            n = nrow(x),
            p = ncol(x),
            method = method,
            backend = backend,
            centers = centers,
            cycle = cycle,
            n_threads = n_threads,
            status = "expected_skip",
            error = skip_reason,
            max_iter = resolve_kmeans_int(max_iter, auto_params$max_iter),
            n_init = resolve_kmeans_int(n_init, auto_params$n_init),
            tol = resolve_kmeans_tol(tol, auto_params$tol),
            tuning_policy = auto_params$policy,
            tuning_rule = auto_params$rule,
            tuning_rule_detail = auto_params$rule_detail,
            tuning_work = auto_params$work,
            tuning_n_per_center = auto_params$n_per_center,
            tuning_high_dim = auto_params$high_dim,
            tuning_large_n = auto_params$large_n,
            tuning_many_centers = auto_params$many_centers,
            tuning_small_many_centers = auto_params$small_many_centers,
            tuning_few_points_many_centers = auto_params$few_points_many_centers,
            selection_policy = "static_shape_center_backend_selector",
            selection_slow_tuning = FALSE,
            selection_predicted_backend = backend,
            selection_reason = backend_policy$reason %||% NA_character_,
            selection_work = backend_policy$work %||% auto_params$work,
            selection_nbytes = backend_policy$nbytes %||% NA_real_,
            selection_n_per_center = backend_policy$n_per_center %||% auto_params$n_per_center,
            selection_cuda_available = cuda_available(),
            selection_faiss_gpu_available = kmeans_faiss_gpu_available(),
            selection_cuvs_available = cuvs_available(),
            requested_backend = backend,
            resolved_backend = backend,
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
            cycle = cycle,
            n_threads = n_threads,
            seed = cycle_seed,
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
          "[%s] dataset=%s cycle=%s method=%s backend=%s centers=%s status=%s elapsed=%.3f\n",
          format(Sys.time(), "%Y-%m-%d %H:%M:%S"),
          dataset_name, cycle, method, backend, centers, row$status, row$elapsed_sec
        ))
        flush.console()
      }
    }
  }
  rm(x, loaded)
  gc()
}

results_df <- do.call(rbind, results)
utils::write.csv(results_df, file.path(out_dir, "kmeans_benchmark_results.csv"), row.names = FALSE)

ok <- results_df[results_df$status == "success", , drop = FALSE]
if (nrow(ok)) {
  best <- select_kmeans_best_rows(ok, group_cols = "dataset")
  utils::write.csv(best, file.path(out_dir, "kmeans_best_by_dataset.csv"), row.names = FALSE)
  best_by_centers <- select_kmeans_best_rows(ok, group_cols = c("dataset", "centers"))
  utils::write.csv(
    best_by_centers,
    file.path(out_dir, "kmeans_best_by_dataset_centers.csv"),
    row.names = FALSE
  )

  cycle_summary <- summarize_kmeans_cycles(ok)
  utils::write.csv(
    cycle_summary,
    file.path(out_dir, "kmeans_cycle_summary.csv"),
    row.names = FALSE
  )

  recommendations <- recommend_kmeans_methods(cycle_summary, ari_tolerance)
  if (nrow(recommendations)) {
    utils::write.csv(
      recommendations,
      file.path(out_dir, "kmeans_recommendations_from_cycles.csv"),
      row.names = FALSE
    )
  }
  backend_recommendations <- recommend_kmeans_methods(
    cycle_summary,
    ari_tolerance,
    group_cols = c("dataset", "centers", "backend")
  )
  if (nrow(backend_recommendations)) {
    utils::write.csv(
      backend_recommendations,
      file.path(out_dir, "kmeans_backend_recommendations_from_cycles.csv"),
      row.names = FALSE
    )
  }

  aggregate_fast <- compare_fast_kmeans_to_recommendations(cycle_summary, recommendations)
  if (nrow(aggregate_fast)) {
    utils::write.csv(
      aggregate_fast,
      file.path(out_dir, "kmeans_fast_vs_cycle_recommendation.csv"),
      row.names = FALSE
    )
  }

  auto_fast <- compare_auto_kmeans_to_recommendations(cycle_summary, recommendations)
  if (nrow(auto_fast)) {
    utils::write.csv(
      auto_fast,
      file.path(out_dir, "kmeans_auto_vs_global_recommendation.csv"),
      row.names = FALSE
    )
  }

  comparison <- compare_fast_kmeans_to_stats(ok)
  if (nrow(comparison)) {
    utils::write.csv(
      comparison,
      file.path(out_dir, "kmeans_fast_vs_stats.csv"),
      row.names = FALSE
    )
  }
}

materials <- c(
  "# K-Means Benchmark",
  "",
  "This benchmark compares faissR `fast_kmeans()` backends with base `stats::kmeans`.",
  "",
  sprintf("- Output directory: `%s`", out_dir),
  sprintf("- Data root: `%s`", data_root),
  sprintf("- Default real datasets: `%s`", paste(dataset_index(data_root)$dataset, collapse = "`, `")),
  "- Default simulated datasets: `SimulatedTiny3Clusters`",
  sprintf("- Methods: `%s`", paste(methods, collapse = "`, `")),
  sprintf("- Backends: `%s`", paste(backends, collapse = "`, `")),
  sprintf("- CPU thread cap: `%s`", n_threads),
  sprintf("- Timeout per combination: `%s` seconds", timeout),
  sprintf("- Cycles: `%s`", cycles),
  sprintf("- ARI tolerance for cycle recommendations: `%s`", ari_tolerance),
  sprintf("- Requested centers fallback: `%s`; `--centers` must be a positive integer and labels override this fallback when available", fallback_centers),
  "",
  "`kmeans_benchmark_config.csv` records the run configuration, including the available real plus simulated dataset names accepted by the dataset selector. `kmeans_benchmark_results.csv` is the raw row-level result table, including successes, failures, expected skips, timings, memory, selected parameters, convergence flags (`converged`, `hit_max_iter`), ARI, within-cluster sums of squares, backend metadata, static selection metadata, categorical `tuning_rule`, and detailed `tuning_rule_detail` shape metadata.",
  "`kmeans_runtime_capabilities.csv` records the runtime availability table used for k-means preflight, including CUDA, FAISS GPU, and cuVS availability, `runtime_reason`, human-readable `runtime_notes`, and whether explicit CUDA k-means requests can run in the current build. The `runtime_reason` field distinguishes available routes from `missing_cuda_runtime` and `missing_gpu_kmeans_backend` preflight skips.",
  "The result table records cycle, elapsed time, peak resident memory when available, requested backend, resolved backend, implementation backend used, total within-cluster sum of squares, iterations, selected k-means parameters, deterministic tuning policy/rule/shape metadata, static `selection_*` no-pilot backend decision metadata, and ARI against dataset labels when labels are available. `tuning_rule` is a stable grouping label such as `small_low_work_multistart`, while `tuning_rule_detail` preserves the exact shape/work values that produced the rule.",
  "`kmeans_best_by_dataset.csv` stores the best successful row per dataset after ranking by ARI, elapsed time, and total within-cluster sum of squares for a compact backwards-compatible summary. `kmeans_best_by_dataset_centers.csv` keeps the best successful row per dataset/centers combination so different requested cluster counts remain auditable.",
  "`kmeans_fast_vs_stats.csv` compares successful `fast_kmeans()` rows with successful `stats::kmeans` rows for the same dataset, cycle, and number of centers, recording speedup, ARI delta, and withinss ratio. Speedups, ARI deltas, and withinss ratios are `NA` when the required timing or quality values are missing or invalid. The `cycle` column supports repeated benchmark cycles such as `--cycles=10` for speed/ARI tuning.",
  "`kmeans_cycle_summary.csv` aggregates successful rows across cycles by dataset/method/backend/centers and reports success counts, median/min/max elapsed time, ARI stability, withinss stability, iteration counts, whether any cycle hit `max_iter`, whether all cycles converged before the iteration cap, selected parameter medians, deterministic tuning rule/shape metadata, static selection metadata, and resolved backend metadata.",
  "`kmeans_recommendations_from_cycles.csv` selects the fastest row within `ari_tolerance` of the best median ARI for each dataset/centers combination and marks `recommendation_basis = \"fastest_within_ari_tolerance\"`; tied median times are broken by higher median ARI and then lower median total within-cluster sum of squares. When ARI is unavailable it selects the fastest median-time row and marks `recommendation_basis = \"speed_only_no_ari\"`.",
  "`kmeans_backend_recommendations_from_cycles.csv` applies the same rule within each dataset/centers/backend group, so CPU, CUDA, auto, and stats rows can be tuned or reported separately without changing the overall recommendation file.",
  "`kmeans_fast_vs_cycle_recommendation.csv` compares aggregate `fast_kmeans()` rows with those cycle-summary recommendations and reports the recommendation basis, median speed ratio, median ARI gap, withinss ratio, selected tuning metadata, requested/resolved backend metadata, CPU thread count, static selection metadata, and backend/implementation agreement. Speed ratios, ARI gaps, and withinss ratios are `NA` when the required timing or quality values are missing or invalid.",
  "`kmeans_auto_vs_global_recommendation.csv` filters that comparison to aggregate `fast_kmeans(backend = \"auto\")` rows and compares them with the pooled global recommendation for the same dataset/centers combination. It records requested-backend, resolved-backend, implementation, speed, ARI, withinss, deterministic tuning, and static no-pilot backend-selection agreement so the k-means auto backend selector can be refined from benchmark evidence.",
  "Explicit CUDA requests whose required CUDA, FAISS GPU, or cuVS k-means runtime is unavailable are recorded as `status = \"expected_skip\"` with `expected_skip = TRUE`; `resolved_backend` remains `cuda` so the skipped public device request is auditable. `backend = \"auto\"` resolves to CPU instead of becoming an expected skip when no k-means-capable CUDA route is available, and also resolves to CPU for small k-means shapes where the deterministic shape gate estimates that GPU launch/copy overhead would dominate. `centers = 1` is resolved to the exact CPU column-mean solution and records `single_cluster_exact_mean`, even for large matrices, because no iterative CPU or CUDA k-means backend can improve that objective. Unexpected runtime errors remain failed rows rather than being replaced with CPU timings."
)
writeLines(materials, file.path(out_dir, "MATERIALS_AND_METHODS_kmeans.md"))

cat("Saved k-means benchmark files in: ", out_dir, "\n", sep = "")
