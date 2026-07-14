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

positive_int_values <- function(value, default = NULL, name) {
  if (is.null(value) || length(value) == 0L || is.na(value[[1L]]) || !nzchar(trimws(as.character(value[[1L]])))) {
    return(default)
  }
  values <- suppressWarnings(as.numeric(split_arg(value, "")))
  values <- values[is.finite(values)]
  if (!length(values) ||
      any(values < 1L | abs(values - round(values)) > sqrt(.Machine$double.eps))) {
    stop("`", name, "` must contain positive integer values.", call. = FALSE)
  }
  unique(as.integer(round(values)))
}

positive_num_values <- function(value, default = NULL, name) {
  if (is.null(value) || length(value) == 0L || is.na(value[[1L]]) || !nzchar(trimws(as.character(value[[1L]])))) {
    return(default)
  }
  values <- suppressWarnings(as.numeric(split_arg(value, "")))
  values <- values[is.finite(values)]
  if (!length(values) || any(values <= 0)) {
    stop("`", name, "` must contain positive numeric values.", call. = FALSE)
  }
  unique(values)
}

script_path <- function() {
  args <- commandArgs(FALSE)
  file_arg <- grep("^--file=", args, value = TRUE)
  if (length(file_arg)) return(normalizePath(sub("^--file=", "", file_arg[[1L]]), mustWork = TRUE))
  normalizePath(file.path("benchmark_scripts", "benchmark_method_tuning_from_reference.R"), mustWork = FALSE)
}

safe_timeout_bin <- local({
  cached <- NULL
  function() {
    if (!is.null(cached)) return(cached)
    env <- Sys.getenv("FAISSR_TIMEOUT_BIN", unset = "")
    if (nzchar(env)) {
      cached <<- if (tolower(env) %in% c("none", "false", "0", "off")) "" else env
      return(cached)
    }
    found <- tryCatch(unname(Sys.which("timeout")), error = function(e) "")
    if (!nzchar(found)) {
      for (candidate in c("/usr/bin/timeout", "/bin/timeout")) {
        if (file.exists(candidate) && file.access(candidate, 1L) == 0L) {
          found <- candidate
          break
        }
      }
    }
    cached <<- found %||% ""
    cached
  }
})

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
    row, file = path, sep = ",", row.names = FALSE,
    col.names = !file.exists(path), append = file.exists(path),
    quote = TRUE, na = ""
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
  if (grepl("not available|unavailable|not built|requires|without cuda|without faiss|without cuvs", msg)) return("unavailable")
  if (grepl("not support|does not support|only supports|only available", msg)) return("unsupported")
  "failed"
}

dataset_path_column <- function(manifest) {
  candidates <- intersect(c("path", "output", "file", "file_path", "rdata_path"), names(manifest))
  if (!length(candidates)) stop("Manifest must contain a dataset file path column.", call. = FALSE)
  candidates[[1L]]
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

reference_file <- function(dataset_path, k, quality_n, seed, metric = "euclidean") {
  metric <- normalize_metric_arg(metric)
  file.path(dirname(dataset_path), sprintf(
    "faissR_exact_reference_%s_k%d_q%d_seed%d.RData",
    metric, as.integer(k), as.integer(quality_n), as.integer(seed)
  ))
}

load_reference <- function(dataset_path, k, reference_k, quality_n, seed, metric = "euclidean") {
  metric <- normalize_metric_arg(metric)
  paths <- reference_file(dataset_path, reference_k, quality_n, seed, metric = metric)
  if (!identical(as.integer(k), as.integer(reference_k))) {
    paths <- c(paths, reference_file(dataset_path, k, quality_n, seed, metric = metric))
  }
  if (identical(metric, "euclidean")) {
    legacy <- file.path(dirname(dataset_path), sprintf(
      "faissR_exact_reference_euclidean_k%d_q%d_seed%d.RData",
      as.integer(reference_k), as.integer(quality_n), as.integer(seed)
    ))
    paths <- unique(c(paths, legacy))
  }
  existing <- paths[file.exists(paths)]
  path <- if (length(existing)) existing[[1L]] else paths[[1L]]
  if (!file.exists(path)) {
    return(list(
      status = "missing_reference",
      path = path,
      error = sprintf(
        "precomputed max-k reference file not found; expected %s",
        basename(paths[[1L]])
      )
    ))
  }
  env <- new.env(parent = emptyenv())
  load(path, envir = env)
  if (!exists("faissR_reference", envir = env, inherits = FALSE)) {
    return(list(status = "bad_reference", path = path, error = "object `faissR_reference` not found"))
  }
  ref <- get("faissR_reference", envir = env, inherits = FALSE)
  if (is.null(ref$indices) || !is.matrix(ref$indices)) {
    return(list(status = "bad_reference", path = path, error = "reference object has no `indices` matrix"))
  }
  if (ncol(ref$indices) < as.integer(k)) {
    return(list(
      status = "bad_reference",
      path = path,
      error = sprintf("reference file contains %d neighbours, but k=%d was requested", ncol(ref$indices), as.integer(k))
    ))
  }
  ref$source_k <- as.integer(ref$k %||% ref$max_k %||% ncol(ref$indices))
  ref$requested_k <- as.integer(k)
  ref$indices <- ref$indices[, seq_len(as.integer(k)), drop = FALSE]
  ref$k <- as.integer(k)
  ref$path <- path
  ref
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
  if (n < 50000L) return("small_n")
  if (n >= 50000L && n < 500000L && p <= 64L) return("medium_low_dim")
  if (n >= 500000L && p <= 64L) return("large_low_dim")
  if (n >= 50000L && p >= 256L) return("large_high_dim")
  "other"
}

all_option_columns <- c(
  "candidate_target_recall", "faiss_query_batch_size", "faiss_gpu_query_batch_size",
  "cuvs_ivf_batch_size", "faiss_gpu_reuse_resources", "cache_fitted_indexes",
  "hnsw_m", "hnsw_ef_construction", "hnsw_ef_search",
  "ivf_nlist", "ivf_nprobe", "pq_m", "pq_nbits", "pq_dim",
  "ivfpq_fastscan_refine_factor", "ivfpq_fastscan_bbs",
  "cagra_graph_degree", "cagra_intermediate_graph_degree",
  "cagra_search_width", "cagra_itopk_size", "cagra_build_algo",
  "nndescent_pool_size", "nndescent_n_iters", "nndescent_max_candidates",
  "nndescent_n_random_projections", "nndescent_graph_degree",
  "nndescent_intermediate_graph_degree", "nndescent_max_iterations",
  "nsg_r", "nsg_graph_k", "vamana_r", "vamana_search_l", "vamana_alpha"
)

fill_candidate <- function(x) {
  for (col in all_option_columns) if (!col %in% names(x)) x[[col]] <- NA
  x[, c("candidate_id", "candidate_kind", "method", "backend", "n_threads", "output", all_option_columns), drop = FALSE]
}

base_candidates <- function(method, backend, k, thread_values, output_values, ids = "default") {
  x <- expand.grid(
    n_threads = thread_values,
    output = output_values,
    candidate_id = ids,
    KEEP.OUT.ATTRS = FALSE,
    stringsAsFactors = FALSE
  )
  x$candidate_kind <- ifelse(x$candidate_id == "auto", "auto", "manual")
  x$method <- method
  x$backend <- backend
  x
}

ivf_nlist <- function(n, k) {
  as.integer(max(4L, min(1024L, n, max(16L, ceiling(sqrt(n)), ceiling(n / max(50L, 20L * k))))))
}

ivf_nprobe <- function(nlist, k) {
  as.integer(max(1L, min(nlist, max(16L, ceiling(sqrt(nlist)), ceiling(k / 3)))))
}

divisor_at_most <- function(p, target) {
  vals <- which(p %% seq_len(p) == 0L)
  vals[max(which(vals <= target))] %||% 1L
}

cuvs_byte_aligned_pq_dim <- function(p, pq_dim, pq_bits = 4L) {
  p <- as.integer(max(1L, p))
  pq_dim <- as.integer(max(0L, min(p, pq_dim)))
  pq_bits <- as.integer(max(4L, min(8L, pq_bits)))
  effective_dim <- if (pq_dim > 0L) pq_dim else p
  if ((pq_bits * effective_dim) %% 8L == 0L) return(pq_dim)
  gcd_int <- function(a, b) {
    a <- abs(as.integer(a)); b <- abs(as.integer(b))
    while (b != 0L) {
      r <- a %% b; a <- b; b <- r
    }
    if (a == 0L) 1L else a
  }
  step <- as.integer(8L / gcd_int(pq_bits, 8L))
  effective_dim <- as.integer(max(1L, min(p, effective_dim)))
  effective_dim <- as.integer(effective_dim - effective_dim %% step)
  if (effective_dim >= 1L) effective_dim else pq_dim
}

grid_number_label <- function(prefix, value) {
  txt <- format(as.numeric(value), trim = TRUE, scientific = FALSE)
  txt <- gsub("[^0-9A-Za-z]+", "p", txt)
  paste0(prefix, txt)
}

ivfpq_fastscan_pq_dim_candidates <- function(p, values = NULL, backend = "cpu") {
  p <- as.integer(max(1L, p))
  if (!identical(backend, "cuda")) {
    return(divisor_at_most(p, 16L))
  }
  if (length(values)) {
    out <- unique(vapply(values, function(value) {
      cuvs_byte_aligned_pq_dim(p, as.integer(round(value)), pq_bits = 4L)
    }, integer(1L)))
  } else {
    out <- unique(vapply(c(8L, 16L, 32L, 64L), function(target) {
      cuvs_byte_aligned_pq_dim(p, divisor_at_most(p, target), pq_bits = 4L)
    }, integer(1L)))
  }
  out <- out[is.finite(out) & out >= 1L & out <= p & (4L * out) %% 8L == 0L]
  if (length(out)) unique(as.integer(out)) else cuvs_byte_aligned_pq_dim(p, divisor_at_most(p, 32L), pq_bits = 4L)
}

nndescent_cuda_candidate_values <- function(n,
                                            k,
                                            grid_level,
                                            graph_degrees = NULL,
                                            intermediate_graph_degrees = NULL,
                                            max_iterations = NULL) {
  k <- as.integer(k)
  manual <- length(graph_degrees) || length(intermediate_graph_degrees) || length(max_iterations)
  if (manual) {
    graph <- as.integer(graph_degrees %||% c(32L, 48L, 64L, 96L))
    intermediate <- as.integer(intermediate_graph_degrees %||% ceiling(1.5 * pmax(k, graph)))
    iterations <- as.integer(max_iterations %||% c(5L, 8L, 12L, 16L))
    lens <- c(length(graph), length(intermediate), length(iterations))
    if (length(unique(lens)) == 1L) {
      vals <- data.frame(
        nndescent_graph_degree = graph,
        nndescent_intermediate_graph_degree = intermediate,
        nndescent_max_iterations = iterations
      )
    } else {
      vals <- expand.grid(
        nndescent_graph_degree = graph,
        nndescent_intermediate_graph_degree = intermediate,
        nndescent_max_iterations = iterations,
        KEEP.OUT.ATTRS = FALSE,
        stringsAsFactors = FALSE
      )
    }
  } else if (n >= 1000000L && k >= 100L) {
    vals <- data.frame(
      nndescent_graph_degree = as.integer(c(k, k, ceiling(112L * k / 100), ceiling(128L * k / 100), ceiling(150L * k / 100), ceiling(180L * k / 100))),
      nndescent_intermediate_graph_degree = as.integer(c(k, ceiling(125L * k / 100), ceiling(150L * k / 100), ceiling(175L * k / 100), ceiling(225L * k / 100), ceiling(275L * k / 100))),
      nndescent_max_iterations = c(5L, 8L, 12L, 16L, 24L, 32L)
    )
  } else {
    vals <- data.frame(
      nndescent_graph_degree = c(16L, 24L, 32L, 48L, 64L, 96L, 128L, 160L, 192L),
      nndescent_intermediate_graph_degree = c(32L, 48L, 64L, 96L, 128L, 192L, 256L, 320L, 384L),
      nndescent_max_iterations = c(4L, 5L, 8L, 12L, 16L, 24L, 32L, 40L, 48L)
    )
    if (identical(grid_level, "compact")) vals <- vals[c(1L, 2L, 3L, 4L), , drop = FALSE]
  }
  if (identical(grid_level, "wide")) {
    vals <- rbind(vals, data.frame(
      nndescent_graph_degree = c(224L, 256L, 320L, 384L),
      nndescent_intermediate_graph_degree = c(448L, 512L, 640L, 768L),
      nndescent_max_iterations = c(56L, 64L, 72L, 80L)
    ))
  }
  vals$nndescent_graph_degree <- as.integer(pmax(k, vals$nndescent_graph_degree))
  vals$nndescent_intermediate_graph_degree <- as.integer(pmax(vals$nndescent_graph_degree, vals$nndescent_intermediate_graph_degree))
  vals <- unique(vals)
  row.names(vals) <- NULL
  vals
}

candidate_grid <- function(method, backend, n, p, k, metric, target_recalls, thread_values, output_values, grid_level,
                           ivfpq_fastscan_refine_factors = NULL,
                           ivfpq_fastscan_nlist_multipliers = NULL,
                           ivfpq_fastscan_nprobe_multipliers = NULL,
                           ivfpq_fastscan_pq_dim_values = NULL,
                           ivfpq_fastscan_pq_m_values = NULL,
                           ivfpq_fastscan_bbs_values = NULL,
                           cuvs_ivf_batch_sizes = NULL,
                           nndescent_cuda_graph_degrees = NULL,
                           nndescent_cuda_intermediate_graph_degrees = NULL,
                           nndescent_cuda_max_iterations = NULL) {
  metric <- normalize_metric_arg(metric)
  inner_product_grid <- identical(metric, "inner_product")
  rows <- list()
  add <- function(x) {
    rows[[length(rows) + 1L]] <<- fill_candidate(x)
  }
  if (method %in% c("exact", "bruteforce")) {
    if (backend == "cpu") {
      x <- expand.grid(
        n_threads = thread_values, output = output_values,
        faiss_query_batch_size = c(2048L, 8192L, 32768L, 65536L, 131072L),
        cache_fitted_indexes = c(FALSE, TRUE),
        KEEP.OUT.ATTRS = FALSE, stringsAsFactors = FALSE
      )
      x$candidate_id <- sprintf("%s_t%d_b%d_%s_cache%s", method, x$n_threads, x$faiss_query_batch_size, x$output, ifelse(x$cache_fitted_indexes, "on", "off"))
    } else {
      x <- expand.grid(
        n_threads = thread_values, output = output_values,
        faiss_gpu_query_batch_size = c(1024L, 4096L, 8192L, 16384L, 32768L, 65536L),
        faiss_gpu_reuse_resources = c(TRUE, FALSE),
        KEEP.OUT.ATTRS = FALSE, stringsAsFactors = FALSE
      )
      x$candidate_id <- sprintf("%s_t%d_b%d_%s_reuse%s", method, x$n_threads, x$faiss_gpu_query_batch_size, x$output, ifelse(x$faiss_gpu_reuse_resources, "on", "off"))
    }
    x$candidate_kind <- "manual"; x$method <- method; x$backend <- backend; add(x)
  } else if (method == "flat") {
    if (backend == "cpu") {
      x <- expand.grid(
        n_threads = thread_values, output = output_values,
        faiss_query_batch_size = c(2048L, 8192L, 32768L, 65536L, 131072L),
        cache_fitted_indexes = c(FALSE, TRUE),
        KEEP.OUT.ATTRS = FALSE, stringsAsFactors = FALSE
      )
      x$candidate_id <- sprintf("flat_t%d_b%d_%s_cache%s", x$n_threads, x$faiss_query_batch_size, x$output, ifelse(x$cache_fitted_indexes, "on", "off"))
    } else {
      x <- expand.grid(
        n_threads = thread_values, output = output_values,
        faiss_gpu_query_batch_size = c(1024L, 4096L, 8192L, 16384L, 32768L, 65536L),
        faiss_gpu_reuse_resources = c(TRUE, FALSE),
        KEEP.OUT.ATTRS = FALSE, stringsAsFactors = FALSE
      )
      x$candidate_id <- sprintf("flat_t%d_b%d_%s_reuse%s", x$n_threads, x$faiss_gpu_query_batch_size, x$output, ifelse(x$faiss_gpu_reuse_resources, "on", "off"))
    }
    x$candidate_kind <- "manual"; x$method <- method; x$backend <- backend; add(x)
  } else if (method == "hnsw") {
    x <- base_candidates(method, backend, k, thread_values, output_values, paste0("auto_", target_recalls))
    x$candidate_kind <- "auto"; x$candidate_target_recall <- rep(target_recalls, each = length(thread_values) * length(output_values)); add(x)
    if (backend == "cpu") {
      manual <- data.frame(
        hnsw_m = c(6L, 8L, 8L, 10L, 12L, 12L, 16L, 24L, 32L, 48L, 64L),
        hnsw_ef_construction = c(30L, 30L, 40L, 50L, 60L, 80L, 100L, 160L, 240L, 320L, 480L),
        hnsw_ef_search = pmax(k, c(15L, 20L, 25L, 35L, 45L, 60L, 80L, 120L, 220L, 400L, 720L))
      )
      if (inner_product_grid) {
        manual <- rbind(manual, data.frame(
          hnsw_m = c(80L, 96L, 128L, 128L),
          hnsw_ef_construction = c(640L, 800L, 960L, 1280L),
          hnsw_ef_search = pmax(k, c(1024L, 2048L, 3072L, 4096L))
        ))
      }
    } else {
      manual <- data.frame(
        cagra_graph_degree = c(8L, 12L, 16L, 24L, 32L, 48L, 64L, 96L, 128L),
        cagra_intermediate_graph_degree = c(16L, 24L, 32L, 48L, 64L, 128L, 192L, 320L, 512L),
        hnsw_ef_search = pmax(k, c(24L, 32L, 48L, 64L, 96L, 128L, 256L, 480L, 768L))
      )
      if (inner_product_grid) {
        manual <- rbind(manual, data.frame(
          cagra_graph_degree = c(160L, 192L, 256L),
          cagra_intermediate_graph_degree = c(640L, 768L, 1024L),
          hnsw_ef_search = pmax(k, c(1024L, 2048L, 4096L))
        ))
      }
    }
    if (identical(grid_level, "compact")) manual <- manual[seq_len(min(5L, nrow(manual))), , drop = FALSE]
    for (i in seq_len(nrow(manual))) {
      x <- base_candidates(method, backend, k, thread_values, output_values, paste(names(manual), manual[i, ], sep = "", collapse = "_"))
      for (nm in names(manual)) x[[nm]] <- manual[[nm]][[i]]
      add(x)
    }
  } else if (method %in% c("ivf", "ivfpq", "ivfpq_fastscan")) {
    base_nlist <- ivf_nlist(n, k)
    custom_ivfpq_fastscan_grid <- method == "ivfpq_fastscan" &&
      (length(ivfpq_fastscan_nlist_multipliers) || length(ivfpq_fastscan_nprobe_multipliers))
    if (inner_product_grid && !custom_ivfpq_fastscan_grid) {
      # Raw MIPS can require substantially broader coarse-list coverage than
      # L2/cosine.  These anchors include exact-recall recovery points while
      # respecting the FAISS GPU nprobe ceiling of 2048.
      specs <- data.frame(
        label = c("ip_speed", "ip_balanced", "ip_recall", "ip_recall_plus",
                  "ip_near_exact", "ip_full_256", "ip_full_512",
                  "ip_full_1024", "ip_full_2048"),
        nlist_abs = c(1024L, 1024L, 1024L, 1024L, 1024L,
                      256L, 512L, 1024L, 2048L),
        probe_fraction = c(1 / 16, 1 / 8, 1 / 4, 1 / 2, 3 / 4,
                           1, 1, 1, 1),
        full_probe = c(FALSE, FALSE, FALSE, FALSE, FALSE,
                       TRUE, TRUE, TRUE, TRUE)
      )
      if (identical(grid_level, "compact")) {
        specs <- specs[c(1L, 3L, 5L, 6L, 8L), , drop = FALSE]
      }
    } else if (custom_ivfpq_fastscan_grid) {
      nlist_mults <- ivfpq_fastscan_nlist_multipliers
      if (!length(nlist_mults)) nlist_mults <- c(0.5, 1, 2, 4)
      nprobe_mults <- ivfpq_fastscan_nprobe_multipliers
      if (!length(nprobe_mults)) nprobe_mults <- c(0.5, 1, 2, 3)
      specs <- expand.grid(
        nlist_mult = unique(as.numeric(nlist_mults)),
        nprobe_mult = unique(as.numeric(nprobe_mults)),
        KEEP.OUT.ATTRS = FALSE,
        stringsAsFactors = FALSE
      )
      specs$label <- paste(
        grid_number_label("nlx", specs$nlist_mult),
        grid_number_label("npx", specs$nprobe_mult),
        sep = "_"
      )
      specs <- specs[, c("label", "nlist_mult", "nprobe_mult"), drop = FALSE]
    } else {
      specs <- data.frame(
        label = c("ultra_speed", "speed", "speed_plus", "balanced", "balanced_plus", "recall", "recall_plus", "recall_max", "full_probe"),
        nlist_mult = c(0.5, 0.5, 1, 1, 2, 2, 4, 6, 8),
        nprobe_mult = c(0.25, 0.5, 0.5, 1, 1.5, 2, 3, 4, 1),
        full_probe = c(FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, TRUE)
      )
      if (identical(grid_level, "compact")) specs <- specs[c(1L, 2L, 4L, 6L), , drop = FALSE]
    }
    if (!"full_probe" %in% names(specs)) specs$full_probe <- FALSE
    for (i in seq_len(nrow(specs))) {
      nlist <- if ("nlist_abs" %in% names(specs)) {
        as.integer(max(1L, min(n, specs$nlist_abs[[i]], if (backend == "cuda") 2048L else n)))
      } else {
        as.integer(max(1L, min(n, round(base_nlist * specs$nlist_mult[[i]]))))
      }
      base_probe <- ivf_nprobe(nlist, k)
      nprobe_values <- if ("probe_fraction" %in% names(specs)) {
        as.integer(max(1L, min(nlist, ceiling(nlist * specs$probe_fraction[[i]]))))
      } else if (isTRUE(specs$full_probe[[i]])) {
        as.integer(nlist)
      } else {
        as.integer(max(1L, min(nlist, ceiling(base_probe * specs$nprobe_mult[[i]]))))
      }
      if (method == "ivfpq_fastscan" && identical(backend, "cpu") &&
          length(ivfpq_fastscan_nprobe_multipliers) && !isTRUE(custom_ivfpq_fastscan_grid)) {
        nprobe_values <- unique(as.integer(pmax(1L, pmin(nlist, ceiling(base_probe * ivfpq_fastscan_nprobe_multipliers)))))
      }
      x <- base_candidates(method, backend, k, thread_values, output_values, sprintf("%s_nl%d_np%d", specs$label[[i]], nlist, nprobe_values[[1L]]))
      x$ivf_nlist <- nlist; x$ivf_nprobe <- nprobe_values[[1L]]
      if (method == "ivfpq") {
        pq_targets <- if (inner_product_grid) {
          if (backend == "cuda") c(16L, 32L, 64L, 96L, 128L, 192L, 256L) else c(16L, 32L, 64L, 96L, 128L, 192L, 256L)
        } else if (backend == "cuda") {
          c(16L, 32L, 48L, 64L)
        } else {
          c(16L, 32L, 48L, 64L, 96L)
        }
        pq_m_values <- unique(vapply(pq_targets, function(target) {
          divisor_at_most(p, target)
        }, integer(1L)))
        pq_m_values <- pq_m_values[is.finite(pq_m_values) & pq_m_values >= 1L]
        if (!length(pq_m_values)) pq_m_values <- divisor_at_most(p, if (backend == "cuda") 32L else 48L)
        for (pq_m in pq_m_values) {
          y <- x
          y$pq_m <- as.integer(pq_m)
          y$pq_nbits <- 8L
          y$candidate_id <- sprintf("%s_nl%d_np%d_m%d_b8", specs$label[[i]], nlist, nprobe_values[[1L]], as.integer(pq_m))
          add(y)
        }
        next
      }
      if (method == "ivfpq_fastscan") {
        pq_m_values <- divisor_at_most(p, 32L)
        if (identical(backend, "cpu") && length(ivfpq_fastscan_pq_m_values)) {
          pq_m_values <- unique(vapply(ivfpq_fastscan_pq_m_values, function(value) {
            divisor_at_most(p, as.integer(round(value)))
          }, integer(1L)))
        }
        pq_dim_values <- ivfpq_fastscan_pq_dim_candidates(
          p,
          values = ivfpq_fastscan_pq_dim_values,
          backend = backend
        )
        if (method == "ivfpq_fastscan" && identical(backend, "cuda")) {
          batch_values <- unique(as.integer(cuvs_ivf_batch_sizes %||% 32768L))
          batch_values <- batch_values[is.finite(batch_values) & batch_values > 0L]
          if (!length(batch_values)) batch_values <- 32768L
          for (nprobe in nprobe_values) {
            for (pq_dim in pq_dim_values) {
              for (batch_size in batch_values) {
                y <- x
                y$ivf_nprobe <- as.integer(nprobe)
                y$pq_m <- NA_integer_
                y$pq_dim <- as.integer(pq_dim)
                y$pq_nbits <- 4L
                y$ivfpq_fastscan_refine_factor <- NA_integer_
                y$ivfpq_fastscan_bbs <- NA_integer_
                y$cuvs_ivf_batch_size <- as.integer(batch_size)
                y$candidate_id <- sprintf(
                  "%s_nl%d_np%d_pqdim%d_bs%d",
                  specs$label[[i]], nlist, as.integer(nprobe),
                  as.integer(pq_dim), as.integer(batch_size)
                )
                add(y)
              }
            }
          }
          next
        }
        refine_by_spec <- if (inner_product_grid) {
          c(2L, 4L, 8L, 16L, 32L, 32L, 64L, 128L, 256L)
        } else {
          c(1L, 2L, 4L, 4L, 8L, 12L, 16L, 24L, 48L)
        }
        refine_values <- refine_by_spec[[min(i, length(refine_by_spec))]]
        if (identical(backend, "cpu") && length(ivfpq_fastscan_refine_factors)) {
          refine_values <- ivfpq_fastscan_refine_factors
        }
        bbs_values <- c(32L, 64L)
        if (identical(backend, "cpu") && length(ivfpq_fastscan_bbs_values)) {
          bbs_values <- unique(as.integer(ivfpq_fastscan_bbs_values))
        }
        for (nprobe in nprobe_values) {
          for (pq_m in pq_m_values) {
            for (bbs in bbs_values) {
              for (refine_factor in refine_values) {
                y <- x
                y$ivf_nprobe <- as.integer(nprobe)
                y$pq_m <- as.integer(pq_m)
                y$pq_dim <- pq_dim_values[[1L]]
                y$pq_nbits <- 4L
                y$ivfpq_fastscan_refine_factor <- as.integer(refine_factor)
                y$ivfpq_fastscan_bbs <- as.integer(bbs)
                y$candidate_id <- sprintf(
                  "%s_nl%d_np%d_m%d_pqdim%d_rf%d_bbs%d",
                  specs$label[[i]], nlist, as.integer(nprobe),
                  as.integer(pq_m), as.integer(pq_dim_values[[1L]]), as.integer(refine_factor),
                  as.integer(bbs)
                )
                add(y)
              }
            }
          }
        }
        next
      }
      add(x)
    }
  } else if (method == "cagra") {
    vals <- expand.grid(
      cagra_build_algo = c("auto", "ivf_pq", "nn_descent", "iterative_cagra_search"),
      setting_id = seq_len(8L),
      KEEP.OUT.ATTRS = FALSE,
      stringsAsFactors = FALSE
    )
    vals$cagra_graph_degree <- c(8L, 16L, 24L, 32L, 48L, 64L, 96L, 128L)[vals$setting_id]
    vals$cagra_intermediate_graph_degree <- c(16L, 32L, 48L, 64L, 128L, 192L, 320L, 512L)[vals$setting_id]
    vals$cagra_search_width <- c(1L, 1L, 2L, 2L, 4L, 8L, 12L, 16L)[vals$setting_id]
    vals$cagra_itopk_size <- pmax(k, c(16L, 32L, 64L, 64L, 128L, 256L, 512L, 768L)[vals$setting_id])
    vals$setting_id <- NULL
    if (inner_product_grid) {
      vals <- rbind(vals, data.frame(
        cagra_build_algo = rep("auto", 3L),
        cagra_graph_degree = c(160L, 192L, 256L),
        cagra_intermediate_graph_degree = c(640L, 768L, 1024L),
        cagra_search_width = c(24L, 32L, 48L),
        cagra_itopk_size = pmax(k, c(1024L, 1536L, 2048L))
      ))
    }
    if (identical(grid_level, "compact")) vals <- vals[vals$cagra_graph_degree <= 64L, , drop = FALSE]
    for (i in seq_len(nrow(vals))) {
      x <- base_candidates(method, backend, k, thread_values, output_values, sprintf("%s_gd%d_igd%d_sw%d_it%d", vals$cagra_build_algo[[i]], vals$cagra_graph_degree[[i]], vals$cagra_intermediate_graph_degree[[i]], vals$cagra_search_width[[i]], vals$cagra_itopk_size[[i]]))
      for (nm in names(vals)) x[[nm]] <- vals[[nm]][[i]]
      add(x)
    }
  } else if (method == "nndescent") {
    if (backend == "cpu") {
      vals <- data.frame(
        nndescent_pool_size = pmax(k, c(16L, 20L, 30L, 40L, 50L, 60L, 80L, 120L, 160L, 220L)),
        nndescent_n_iters = c(4L, 5L, 7L, 10L, 10L, 12L, 16L, 20L, 24L, 32L),
        nndescent_max_candidates = c(48L, 60L, 90L, 120L, 150L, 180L, 240L, 360L, 480L, 660L),
        nndescent_n_random_projections = c(3L, 4L, 6L, 8L, 10L, 12L, 16L, 24L, 32L, 40L)
      )
      vals$nndescent_max_candidates <- as.integer(pmax(vals$nndescent_max_candidates, 3L * vals$nndescent_pool_size))
      if (inner_product_grid) {
        vals <- rbind(vals, data.frame(
          nndescent_pool_size = pmax(k, c(256L, 384L, 512L, 768L)),
          nndescent_n_iters = c(40L, 48L, 64L, 80L),
          nndescent_max_candidates = c(1024L, 1536L, 2048L, 3072L),
          nndescent_n_random_projections = c(48L, 64L, 96L, 128L)
        ))
      }
      if (identical(grid_level, "compact")) vals <- vals[seq_len(min(4L, nrow(vals))), , drop = FALSE]
    } else {
      vals <- nndescent_cuda_candidate_values(
        n = n,
        k = k,
        grid_level = grid_level,
        graph_degrees = nndescent_cuda_graph_degrees,
        intermediate_graph_degrees = nndescent_cuda_intermediate_graph_degrees,
        max_iterations = nndescent_cuda_max_iterations
      )
    }
    for (i in seq_len(nrow(vals))) {
      x <- base_candidates(method, backend, k, thread_values, output_values, paste(names(vals), vals[i, ], sep = "", collapse = "_"))
      for (nm in names(vals)) x[[nm]] <- vals[[nm]][[i]]
      add(x)
    }
  } else if (method == "nsg") {
    vals <- data.frame(
      nsg_r = c(4L, 6L, 8L, 12L, 16L, 24L, 32L, 48L, 64L, 96L),
      nsg_graph_k = pmax(k, c(12L, 16L, 20L, 30L, 40L, 60L, 80L, 120L, 180L, 256L))
    )
    if (inner_product_grid) {
      vals <- rbind(vals, data.frame(
        nsg_r = c(128L, 192L, 256L, 384L),
        nsg_graph_k = pmax(k, c(384L, 512L, 768L, 1024L))
      ))
    }
    if (identical(grid_level, "compact")) vals <- vals[seq_len(min(4L, nrow(vals))), , drop = FALSE]
    for (i in seq_len(nrow(vals))) {
      x <- base_candidates(method, backend, k, thread_values, output_values, sprintf("r%d_gk%d", vals$nsg_r[[i]], vals$nsg_graph_k[[i]]))
      x$nsg_r <- vals$nsg_r[[i]]; x$nsg_graph_k <- vals$nsg_graph_k[[i]]; add(x)
    }
  } else if (method == "vamana") {
    vals <- data.frame(
      vamana_r = c(4L, 6L, 8L, 12L, 16L, 24L, 32L, 48L, 64L, 96L),
      vamana_search_l = pmax(k, c(12L, 16L, 20L, 30L, 40L, 60L, 100L, 160L, 240L, 400L)),
      vamana_alpha = c(1.0, 1.0, 1.0, 1.0, 1.0, 1.05, 1.1, 1.2, 1.35, 1.5)
    )
    if (inner_product_grid) {
      vals <- rbind(vals, data.frame(
        vamana_r = c(128L, 192L, 256L, 384L),
        vamana_search_l = pmax(k, c(512L, 768L, 1024L, 1536L)),
        vamana_alpha = c(1.1, 1.2, 1.3, 1.4)
      ))
    }
    if (identical(grid_level, "compact")) vals <- vals[seq_len(min(4L, nrow(vals))), , drop = FALSE]
    for (i in seq_len(nrow(vals))) {
      x <- base_candidates(method, backend, k, thread_values, output_values, sprintf("r%d_l%d_a%s", vals$vamana_r[[i]], vals$vamana_search_l[[i]], vals$vamana_alpha[[i]]))
      x$vamana_r <- vals$vamana_r[[i]]; x$vamana_search_l <- vals$vamana_search_l[[i]]; x$vamana_alpha <- vals$vamana_alpha[[i]]; add(x)
    }
  } else {
    stop("Unsupported method: ", method, call. = FALSE)
  }
  out <- do.call(rbind, rows)
  row.names(out) <- NULL
  out
}

apply_candidate <- function(candidate) {
  Sys.unsetenv(c(
    "FAISSR_FAISS_QUERY_BATCH_SIZE", "FAISSR_FAISS_GPU_QUERY_BATCH_SIZE",
    "FAISSR_FAISS_GPU_REUSE_RESOURCES", "FAISSR_CUVS_IVF_BATCH_SIZE"
  ))
  options(
    faissR.faiss_nlist = NULL, faissR.ivf_nlist = NULL,
    faissR.faiss_nprobe = NULL, faissR.ivf_nprobe = NULL,
    faissR.faiss_pq_m = NULL, faissR.faiss_pq_nbits = NULL,
    faissR.cuvs_ivfpq_pq_dim = NULL, faissR.ivfpq_pq_dim = NULL,
    faissR.cuvs_ivfpq_pq_bits = NULL, faissR.ivfpq_pq_bits = NULL,
    faissR.ivfpq_fastscan_pq_m = NULL, faissR.ivfpq_fastscan_refine_factor = NULL, faissR.ivfpq_fastscan_bbs = NULL,
    faissR.faiss_hnsw_m = NULL, faissR.faiss_hnsw_ef_construction = NULL, faissR.faiss_hnsw_ef_search = NULL,
    faissR.cuvs_graph_degree = NULL, faissR.cuvs_intermediate_graph_degree = NULL,
    faissR.cuvs_search_width = NULL, faissR.cuvs_itopk_size = NULL, faissR.cuvs_cagra_build_algo = NULL,
    faissR.cuvs_hnsw_ef = NULL,
    faissR.cpu_nndescent_pool_size = NULL, faissR.cpu_nndescent_n_iters = NULL,
    faissR.cpu_nndescent_max_candidates = NULL, faissR.cpu_nndescent_n_random_projections = NULL,
    faissR.cuvs_nndescent_graph_degree = NULL, faissR.cuvs_nndescent_intermediate_graph_degree = NULL,
    faissR.cuvs_nndescent_max_iterations = NULL,
    faissR.cpu_nsg_r = NULL, faissR.cpu_nsg_graph_k = NULL,
    faissR.cuda_nsg_r = NULL, faissR.cuda_nsg_graph_k = NULL,
    faissR.faiss_vamana_r = NULL, faissR.faiss_vamana_search_l = NULL, faissR.vamana_alpha = NULL
  )
  set_opt <- function(name, value) if (!is.null(value) && length(value) && !is.na(value[[1L]])) options(stats::setNames(list(value[[1L]]), paste0("faissR.", name)))
  if (!is.na(candidate$faiss_query_batch_size)) Sys.setenv(FAISSR_FAISS_QUERY_BATCH_SIZE = as.integer(candidate$faiss_query_batch_size))
  if (!is.na(candidate$faiss_gpu_query_batch_size)) Sys.setenv(FAISSR_FAISS_GPU_QUERY_BATCH_SIZE = as.integer(candidate$faiss_gpu_query_batch_size))
  if (!is.na(candidate$cuvs_ivf_batch_size)) Sys.setenv(FAISSR_CUVS_IVF_BATCH_SIZE = as.integer(candidate$cuvs_ivf_batch_size))
  if (!is.na(candidate$faiss_gpu_reuse_resources)) Sys.setenv(FAISSR_FAISS_GPU_REUSE_RESOURCES = if (isTRUE(candidate$faiss_gpu_reuse_resources)) "1" else "0")
  if (!is.na(candidate$cache_fitted_indexes)) options(faissR.cache_fitted_nn_indexes = isTRUE(candidate$cache_fitted_indexes), faissR.cache_fitted_nn_indexes_max_entries = 1L)
  set_opt("ivf_nlist", candidate$ivf_nlist); set_opt("faiss_nlist", candidate$ivf_nlist)
  set_opt("ivf_nprobe", candidate$ivf_nprobe); set_opt("faiss_nprobe", candidate$ivf_nprobe)
  set_opt("faiss_pq_m", candidate$pq_m); set_opt("faiss_pq_nbits", candidate$pq_nbits)
  set_opt("cuvs_ivfpq_pq_dim", candidate$pq_dim); set_opt("ivfpq_pq_dim", candidate$pq_dim)
  set_opt("cuvs_ivfpq_pq_bits", candidate$pq_nbits); set_opt("ivfpq_pq_bits", candidate$pq_nbits)
  set_opt("ivfpq_fastscan_pq_m", candidate$pq_m); set_opt("ivfpq_fastscan_refine_factor", candidate$ivfpq_fastscan_refine_factor); set_opt("ivfpq_fastscan_bbs", candidate$ivfpq_fastscan_bbs)
  set_opt("faiss_hnsw_m", candidate$hnsw_m); set_opt("faiss_hnsw_ef_construction", candidate$hnsw_ef_construction); set_opt("faiss_hnsw_ef_search", candidate$hnsw_ef_search)
  set_opt("cuvs_graph_degree", candidate$cagra_graph_degree); set_opt("cuvs_intermediate_graph_degree", candidate$cagra_intermediate_graph_degree)
  set_opt("cuvs_search_width", candidate$cagra_search_width); set_opt("cuvs_itopk_size", candidate$cagra_itopk_size); set_opt("cuvs_hnsw_ef", candidate$hnsw_ef_search)
  if (!is.na(candidate$cagra_build_algo)) options(faissR.cuvs_cagra_build_algo = as.character(candidate$cagra_build_algo))
  set_opt("cpu_nndescent_pool_size", candidate$nndescent_pool_size); set_opt("cpu_nndescent_n_iters", candidate$nndescent_n_iters)
  set_opt("cpu_nndescent_max_candidates", candidate$nndescent_max_candidates); set_opt("cpu_nndescent_n_random_projections", candidate$nndescent_n_random_projections)
  set_opt("cuvs_nndescent_graph_degree", candidate$nndescent_graph_degree); set_opt("cuvs_nndescent_intermediate_graph_degree", candidate$nndescent_intermediate_graph_degree); set_opt("cuvs_nndescent_max_iterations", candidate$nndescent_max_iterations)
  set_opt("cpu_nsg_r", candidate$nsg_r); set_opt("cpu_nsg_graph_k", candidate$nsg_graph_k); set_opt("cuda_nsg_r", candidate$nsg_r); set_opt("cuda_nsg_graph_k", candidate$nsg_graph_k)
  set_opt("faiss_vamana_r", candidate$vamana_r); set_opt("faiss_vamana_search_l", candidate$vamana_search_l); set_opt("vamana_alpha", candidate$vamana_alpha)
  invisible(TRUE)
}

base_row <- function(config, status = "success", error = NA_character_) {
  candidate <- config$candidate
  out <- data.frame(
    dataset = config$dataset, n = as.integer(config$n), p = as.integer(config$p),
    shape_group = shape_group(config$n, config$p), backend = config$backend,
    method = config$method, metric = config$metric, k = as.integer(config$k),
    candidate_id = candidate$candidate_id, candidate_kind = candidate$candidate_kind,
    n_threads = as.integer(candidate$n_threads), output = candidate$output,
    status = status, elapsed_sec = NA_real_, peak_rss_gb = NA_real_,
    recall_at_k = NA_real_, median_recall_at_k = NA_real_, min_recall_at_k = NA_real_,
    reference_status = config$reference_status %||% NA_character_,
    reference_path = config$reference_path %||% NA_character_,
    reference_query_n = as.integer(config$reference_query_n %||% NA_integer_),
    result_backend = NA_character_, resolved_backend = NA_character_,
    implementation_backend = NA_character_, distance_type = NA_character_,
    input_type = NA_character_, input_layout = NA_character_,
    search_batch_size = NA_integer_, requested_search_batch_size = NA_integer_,
    search_batch_policy = NA_character_, query_call_count = NA_integer_,
    dataset_residency = NA_character_, query_residency = NA_character_,
    query_uses_index_dataset_buffer = NA, query_device_buffer_cached = NA,
    query_device_buffer_reused = NA, query_device_cache_status = NA_character_,
    query_host_to_device_copies = NA_integer_,
    index_build_host_to_device_copies = NA_integer_,
    query_upload_count = NA_integer_, query_cache_hit_count = NA_integer_,
    host_device_traffic_policy = NA_character_,
    exact = NA, error = error, stringsAsFactors = FALSE
  )
  cbind(out, candidate[all_option_columns])
}

run_method <- function(config) {
  configure_threads(config$candidate$n_threads)
  load_faissR()
  x <- load_dataset_matrix(config$dataset_path)
  apply_candidate(config$candidate)
  started <- proc.time()[["elapsed"]]
  target <- suppressWarnings(as.numeric(config$candidate$candidate_target_recall))
  if (!is.finite(target)) target <- max(config$target_recalls)
  run_nn <- function(input) {
    faissR::nn(
      input, k = as.integer(config$k), exclude_self = TRUE,
      backend = config$backend, method = config$method, metric = config$metric,
      tuning = if (identical(config$candidate$candidate_kind, "auto")) "auto" else "fixed",
      target_recall = target, output = as.character(config$candidate$output),
      n_threads = as.integer(config$candidate$n_threads)
    )
  }
  res <- tryCatch(run_nn(x), error = function(e) e)
  if (inherits(res, "error")) {
    stop(res)
  }
  elapsed <- proc.time()[["elapsed"]] - started
  row <- base_row(config, "success")
  quality <- recall_summary(res$indices[config$reference_rows, , drop = FALSE], config$reference_indices)
  row$elapsed_sec <- as.numeric(elapsed)
  row$peak_rss_gb <- read_peak_rss_gb()
  row$recall_at_k <- quality$recall_at_k
  row$median_recall_at_k <- quality$median_recall_at_k
  row$min_recall_at_k <- quality$min_recall_at_k
  row$result_backend <- res$backend_used %||% attr(res, "backend") %||% NA_character_
  row$resolved_backend <- attr(res, "resolved_backend") %||% row$result_backend
  row$implementation_backend <- attr(res, "implementation_backend") %||% NA_character_
  row$distance_type <- res$distance_type %||% attr(res, "distance_type") %||% NA_character_
  row$input_type <- res$input_type %||% attr(res, "input_type") %||% NA_character_
  row$input_layout <- res$input_layout %||% attr(res, "input_layout") %||% NA_character_
  approx <- attr(res, "approximation") %||% list()
  row$search_batch_size <- as.integer(res$search_batch_size %||% approx$search_batch_size %||% NA_integer_)
  row$requested_search_batch_size <- as.integer(
    res$requested_search_batch_size %||% approx$requested_search_batch_size %||% NA_integer_
  )
  row$search_batch_policy <- as.character(
    res$search_batch_policy %||% approx$search_batch_policy %||% NA_character_
  )
  row$query_call_count <- as.integer(res$query_call_count %||% approx$query_call_count %||% NA_integer_)
  gpu <- attr(res, "gpu_residency") %||% list()
  row$dataset_residency <- as.character(
    res$dataset_residency %||% approx$dataset_residency %||% gpu$dataset_residency %||% NA_character_
  )
  row$query_residency <- as.character(
    res$query_residency %||% approx$query_residency %||% gpu$query_residency %||% NA_character_
  )
  row$query_uses_index_dataset_buffer <- as.logical(
    res$query_uses_index_dataset_buffer %||% approx$query_uses_index_dataset_buffer %||%
      gpu$query_uses_index_dataset_buffer %||% NA
  )
  row$query_device_buffer_cached <- as.logical(
    res$query_device_buffer_cached %||% approx$query_device_buffer_cached %||%
      gpu$query_device_buffer_cached %||% NA
  )
  row$query_device_buffer_reused <- as.logical(
    res$query_device_buffer_reused %||% approx$query_device_buffer_reused %||%
      gpu$query_device_buffer_reused %||% NA
  )
  row$query_device_cache_status <- as.character(
    res$query_device_cache_status %||% approx$query_device_cache_status %||%
      gpu$query_device_cache_status %||% NA_character_
  )
  row$query_host_to_device_copies <- as.integer(
    res$query_host_to_device_copies %||% approx$query_host_to_device_copies %||%
      gpu$query_host_to_device_copies %||% NA_integer_
  )
  row$index_build_host_to_device_copies <- as.integer(
    res$index_build_host_to_device_copies %||% approx$index_build_host_to_device_copies %||%
      gpu$index_build_host_to_device_copies %||% NA_integer_
  )
  row$query_upload_count <- as.integer(
    res$query_upload_count %||% approx$query_upload_count %||% gpu$query_upload_count %||% NA_integer_
  )
  row$query_cache_hit_count <- as.integer(
    res$query_cache_hit_count %||% approx$query_cache_hit_count %||%
      gpu$query_cache_hit_count %||% NA_integer_
  )
  row$host_device_traffic_policy <- as.character(
    res$host_device_traffic_policy %||% approx$host_device_traffic_policy %||%
      gpu$host_device_traffic_policy %||% NA_character_
  )
  row$exact <- isTRUE(attr(res, "exact"))
  row
}

run_child <- function() {
  args <- parse_args()
  config <- readRDS(args$config)
  row <- tryCatch(
    run_method(config),
    error = function(e) base_row(config, classify_error(conditionMessage(e)), conditionMessage(e))
  )
  saveRDS(row, args$result)
}

run_task <- function(config, timeout, bench_script) {
  cfg <- tempfile("faissR_method_cfg_", fileext = ".rds")
  out <- tempfile("faissR_method_out_", fileext = ".rds")
  saveRDS(config, cfg)
  on.exit(unlink(c(cfg, out)), add = TRUE)
  cmd <- c(Sys.getenv("R_BIN", "Rscript"), "--vanilla", bench_script,
           "--child=TRUE", paste0("--config=", cfg), paste0("--result=", out))
  timeout_bin <- safe_timeout_bin()
  if (nzchar(timeout_bin)) cmd <- c(timeout_bin, as.character(as.integer(ceiling(timeout))), cmd)
  status <- system(paste(shQuote(cmd), collapse = " "), intern = TRUE)
  exit_status <- attr(status, "status") %||% 0L
  if (file.exists(out) && !identical(as.integer(exit_status), 124L)) return(readRDS(out))
  row <- base_row(config, if (identical(as.integer(exit_status), 124L)) "timeout" else classify_error(paste(status, collapse = "\n")), paste(status, collapse = "\n"))
  row$elapsed_sec <- if (identical(as.integer(exit_status), 124L)) timeout else NA_real_
  row
}

completed_keys <- function(path, resume_statuses = c("success", "unsupported", "unavailable", "timeout")) {
  if (!file.exists(path)) return(character())
  x <- tryCatch(read.csv(path, stringsAsFactors = FALSE), error = function(e) NULL)
  if (is.null(x) || !nrow(x)) return(character())
  if ("status" %in% names(x)) {
    x <- x[x$status %in% resume_statuses, , drop = FALSE]
  }
  if (!nrow(x)) return(character())
  metric <- if ("metric" %in% names(x)) x$metric else rep("euclidean", nrow(x))
  paste(x$dataset, x$backend, x$method, metric, x$k, x$candidate_id, sep = "\r")
}

timeout_skip_keys <- function(paths) {
  paths <- unique(paths[nzchar(paths)])
  if (!length(paths)) return(character())
  files <- character()
  for (path in paths) {
    if (dir.exists(path)) {
      files <- c(files, list.files(path, pattern = "_tuning_results[.]csv$", recursive = TRUE, full.names = TRUE))
    } else if (file.exists(path)) {
      files <- c(files, path)
    }
  }
  files <- unique(files[file.exists(files)])
  if (!length(files)) return(character())
  keys <- character()
  for (file in files) {
    x <- tryCatch(read.csv(file, stringsAsFactors = FALSE), error = function(e) NULL)
    if (is.null(x) || !nrow(x)) next
    status_col <- if ("status" %in% names(x)) "status" else if ("previous_status" %in% names(x)) "previous_status" else NA_character_
    if (is.na(status_col)) next
    needed <- c("dataset", "backend", "method", "k", "candidate_id")
    if (!all(needed %in% names(x))) next
    x <- x[x[[status_col]] == "timeout", , drop = FALSE]
    if (!nrow(x)) next
    keys <- c(keys, paste(x$dataset, x$backend, x$method, x$k, x$candidate_id, sep = "\r"))
  }
  unique(keys)
}

summarize_results <- function(out_dir, results_path, target_recalls, method) {
  if (!file.exists(results_path)) return(invisible(NULL))
  x <- read.csv(results_path, stringsAsFactors = FALSE)
  success <- x[x$status == "success" & is.finite(x$recall_at_k) & is.finite(x$elapsed_sec), , drop = FALSE]
  rows <- list(); idx <- 0L
  groups <- unique(success[c("dataset", "backend", "method", "metric", "k")])
  for (g in seq_len(nrow(groups))) {
    part0 <- success[
      success$dataset == groups$dataset[[g]] &
        success$backend == groups$backend[[g]] &
        success$method == groups$method[[g]] &
        success$metric == groups$metric[[g]] &
        success$k == groups$k[[g]],
      , drop = FALSE
    ]
    for (target in target_recalls) {
      ok <- part0[part0$recall_at_k >= target, , drop = FALSE]
      best <- if (nrow(ok)) ok[order(ok$elapsed_sec, -ok$recall_at_k), , drop = FALSE][1L, , drop = FALSE] else part0[order(-part0$recall_at_k, part0$elapsed_sec), , drop = FALSE][1L, , drop = FALSE]
      idx <- idx + 1L
      rows[[idx]] <- cbind(target_recall_threshold = target, recommendation_basis = if (nrow(ok)) "fastest_meeting_target" else "best_recall_below_target", best)
    }
  }
  rec <- if (length(rows)) do.call(rbind, rows) else cbind(target_recall_threshold = numeric(0), recommendation_basis = character(0), x[FALSE, , drop = FALSE])
  write.csv(rec, file.path(out_dir, sprintf("%s_tuning_recommendations.csv", method)), row.names = FALSE)
  method_notes <- character()
  if (identical(method, "ivf")) {
    method_notes <- c(
      "",
      "## IVF Notes",
      "",
      "- Rows tune FAISS IVF-Flat `ivf_nlist` and `ivf_nprobe`; Euclidean and raw inner product use native FAISS IVF L2/IP, while cosine/correlation use normalized FAISS IVF searches.",
      "- CUDA rows exercise the public `backend = \"cuda\", method = \"ivf\"` route, which resolves to FAISS GPU IVF-Flat with cuVS-enabled FAISS builds when available.",
      "- CUDA correlation rows call the public `backend = \"cuda\", method = \"ivf\", metric = \"correlation\"` route, which row-centers and row-normalizes float32 data before FAISS GPU IVF search and converts normalized-search distances back to correlation distance.",
      "- CUDA IVF correlation shape/k/target defaults are summarized in `benchmark_scripts/cuda_ivf_correlation_shape_tuning_defaults_from_uploaded_results.csv` after aggregating the measured `faissR_IVF_TUNING_CUDA_correlation_20260703_133655` sweep.",
      "- CUDA IVF raw-inner-product defaults are initially summarized in `benchmark_scripts/cuda_ivf_inner_product_shape_tuning_defaults_from_seeded_euclidean_results.csv`, seeded from the measured CUDA IVF Euclidean rows and marked validation-pending until the dedicated inner-product sweep replaces them.",
      "- Shape-level defaults for `tuning = \"auto\"` are selected from the fastest candidate that reaches the requested recall target when available; otherwise the highest-coverage best-available candidate is recorded with `tuning_benchmark_target_met = FALSE`."
    )
  }
  if (identical(method, "ivfpq")) {
    method_notes <- c(
      "",
      "## IVFPQ Notes",
      "",
      "- Rows tune FAISS IVF-PQ `ivf_nlist`, `ivf_nprobe`, `pq_m`, and fixed 8-bit PQ codes; Euclidean and raw inner product use native FAISS IVF-PQ L2/IP, while cosine/correlation use normalized L2 transforms.",
      "- CUDA rows exercise the public `backend = \"cuda\", method = \"ivfpq\"` route, which resolves to FAISS GPU IVF-PQ with cuVS-enabled FAISS builds when available.",
      "- CUDA correlation rows call the public `backend = \"cuda\", method = \"ivfpq\", metric = \"correlation\"` route, which row-centers and row-normalizes float32 data before FAISS GPU IVF-PQ search and converts normalized-search distances back to correlation distance.",
      "- CUDA IVFPQ correlation shape/k/target defaults are summarized in `benchmark_scripts/cuda_ivfpq_correlation_shape_tuning_defaults_from_uploaded_results.csv` after aggregating the measured `faissR_IVFPQ_TUNING_CUDA_correlation_20260703_095008` sweep.",
      "- CUDA IVFPQ raw-inner-product defaults are initially summarized in `benchmark_scripts/cuda_ivfpq_inner_product_shape_tuning_defaults_from_seeded_euclidean_results.csv`, seeded from the measured CUDA IVFPQ Euclidean rows and marked validation-pending until the dedicated inner-product sweep replaces them.",
      "- Shape-level defaults for `tuning = \"auto\"` are selected from the fastest candidate that reaches the requested recall target when available; otherwise the highest-recall candidate below target is recorded as a best-available setting.",
      "- `nlist` controls the number of coarse IVF lists, `nprobe` controls how many lists are searched, and `pq_m` controls product-quantizer subdivision. Larger `nprobe` and larger feasible `pq_m` generally improve recall but can increase build/search time and GPU memory traffic."
    )
  }
  if (identical(method, "ivfpq_fastscan")) {
    method_notes <- c(
      "",
      "## IVFPQ FastScan Notes",
      "",
      "- CPU rows tune FAISS `IndexIVFPQFastScan` `ivf_nlist`, `ivf_nprobe`, `pq_m`, fixed 4-bit PQ, `ivfpq_fastscan_refine_factor`, and `ivfpq_fastscan_bbs`; Euclidean and raw inner product use native FastScan L2/IP, while cosine/correlation use normalized L2 transforms.",
      "- CUDA rows tune `ivf_nlist`, `ivf_nprobe`, byte-aligned 4-bit `cuvs_ivfpq_pq_dim`, and `FAISSR_CUVS_IVF_BATCH_SIZE`.",
      "- CUDA cosine rows call the public `backend = \"cuda\", method = \"ivfpq_fastscan\", metric = \"cosine\"` route, which row-normalizes to float32 before cuVS IVF-PQ L2 search and converts distances back to cosine distance.",
      "- CUDA correlation rows call the public `backend = \"cuda\", method = \"ivfpq_fastscan\", metric = \"correlation\"` route, which row-centers and row-normalizes to float32 before cuVS IVF-PQ L2 search and converts distances back to correlation distance.",
      "- CUDA raw-inner-product rows call the public `backend = \"cuda\", method = \"ivfpq_fastscan\", metric = \"inner_product\"` route, which applies the maximum-inner-product-to-L2 extra-dimension transform before cuVS IVF-PQ L2 search and converts distances back to shifted inner-product distances.",
      "- CUDA IVFPQ FastScan correlation shape/k/target defaults are summarized in `benchmark_scripts/cuda_ivfpq_fastscan_correlation_shape_tuning_defaults_from_seeded_euclidean_results.csv`; these rows are validation-pending until the corrected correlation sweep replaces the prior run that failed before reaching cuVS.",
      "- CUDA IVFPQ FastScan raw-inner-product shape/k/target defaults are summarized in `benchmark_scripts/cuda_ivfpq_fastscan_inner_product_shape_tuning_defaults_from_seeded_euclidean_results.csv`; these rows are validation-pending until this metric-specific sweep replaces the seeded Euclidean policy.",
      "- For cuVS 4-bit IVF-PQ, `pq_dim` is repaired to a byte-aligned value when needed; smaller `pq_dim` and smaller `nprobe` are expected to be faster but can reduce recall.",
      "- `nlist` controls the IVF build/search balance; too few lists can hurt recall, while too many lists can increase build and coarse-search overhead.",
      "- `FAISSR_CUVS_IVF_BATCH_SIZE` changes query batching and GPU memory use, not the IVF-PQ recall target directly.",
      "- faissR submits multi-query cuVS IVF-PQ/FastScan searches in batches and prevents row-by-row search for multi-query calls.",
      "- CUDA host-device traffic columns (`dataset_residency`, `query_residency`, `query_device_cache_status`, `query_host_to_device_copies`) show whether searches reused GPU-resident buffers or uploaded query data."
    )
  }
  if (identical(method, "flat")) {
    method_notes <- c(
      "",
      "## Flat Notes",
      "",
      "- Rows tune exhaustive FAISS Flat search metadata rather than an approximate recall/speed trade-off; recall should be exact apart from metric transforms and numerical precision.",
      "- CPU rows tune FAISS Flat query batching and fitted-index reuse for Euclidean, cosine, correlation, and raw inner-product routes.",
      "- CUDA rows tune FAISS GPU Flat query batch size, resource reuse, and float32 output handling for Euclidean, cosine, and correlation routes.",
      "- CUDA correlation rows call the public `backend = \"cuda\", method = \"flat\", metric = \"correlation\"` route, which row-centers and row-normalizes float32 data before FAISS GPU Flat L2 search and converts normalized Euclidean distances back to correlation distance.",
      "- Shape-level defaults for `tuning = \"auto\"` are selected from the fastest measured candidate for each shape/k/target cell; exact methods record `tuning_benchmark_target_met` so any numerical shortfall is visible.",
      "- CUDA Flat correlation shape/k/target defaults are summarized in `benchmark_scripts/cuda_flat_correlation_shape_tuning_defaults_from_uploaded_results.csv` after aggregating the measured FAISS GPU Flat correlation sweep."
    )
  }
  if (identical(method, "bruteforce")) {
    method_notes <- c(
      "",
      "## Bruteforce Notes",
      "",
      "- Rows tune exhaustive brute-force search metadata rather than an approximate recall/speed trade-off; recall should be exact apart from metric transforms and numerical precision.",
      "- CPU rows tune FAISS Flat query batching and fitted-index reuse for Euclidean, cosine, correlation, and raw inner-product bruteforce routes.",
      "- CUDA rows tune cuVS brute-force query batch size, resource reuse, and float32 output handling for Euclidean, cosine, and correlation routes.",
      "- CUDA correlation rows call the public `backend = \"cuda\", method = \"bruteforce\", metric = \"correlation\"` route, which row-centers and row-normalizes float32 data before cuVS L2 brute-force search and converts normalized Euclidean distances back to correlation distance.",
      "- CUDA Bruteforce correlation shape/k/target defaults are summarized in `benchmark_scripts/cuda_bruteforce_correlation_shape_tuning_defaults_from_proxy_results.csv`; these rows initially reuse measured Euclidean cuVS brute-force batch/resource choices because the earlier uploaded correlation sweep failed before reaching the backend."
    )
  }
  if (identical(method, "hnsw")) {
    method_notes <- c(
      "",
      "## HNSW Notes",
      "",
      "- CPU rows tune FAISS HNSW `M`, `efConstruction`, and `efSearch` for Euclidean, cosine, correlation, and raw inner-product routes.",
      "- CUDA rows tune the cuVS HNSW-from-CAGRA route through `cagra_graph_degree`, `cagra_intermediate_graph_degree`, and `hnsw_ef_search`.",
      "- CUDA correlation rows call the public `backend = \"cuda\", method = \"hnsw\", metric = \"correlation\"` route, which row-centers and row-normalizes float32 data before cuVS HNSW graph search and converts normalized Euclidean distances back to correlation distance.",
      "- CUDA raw-inner-product rows call the public `backend = \"cuda\", method = \"hnsw\", metric = \"inner_product\"` route, which applies the maximum-inner-product-to-L2 transform before cuVS HNSW graph search and converts distances back to raw inner-product scores.",
      "- CUDA HNSW correlation shape/k/target defaults are summarized in `benchmark_scripts/cuda_hnsw_correlation_shape_tuning_defaults_from_uploaded_results.csv` after aggregating the measured `faissR_HNSW_TUNING_CUDA_correlation_20260703_070901` sweep.",
      "- CUDA HNSW raw-inner-product defaults are initially seeded from `benchmark_scripts/cuda_hnsw_inner_product_shape_tuning_defaults_from_seeded_euclidean_results.csv`; this wrapper replaces those validation-pending rows with measured metric-specific settings.",
      "- Shape-level defaults for `tuning = \"auto\"` are selected from the fastest candidate that reaches the requested recall target when available; otherwise the highest-recall candidate below target is recorded as a best-available setting."
    )
  }
  if (identical(method, "nndescent")) {
    method_notes <- c(
      "",
      "## NN-descent Notes",
      "",
      "- CPU rows tune native NN-descent candidate pool size, iteration count, maximum candidate breadth, and random-projection seed count.",
      "- CUDA rows tune direct cuVS NN-descent `graph_degree`, `intermediate_graph_degree`, and `max_iterations`.",
      "- CUDA cosine rows call the public `backend = \"cuda\", method = \"nndescent\", metric = \"cosine\"` route, which row-normalizes to float32 before cuVS NN-descent and converts normalized Euclidean distances back to cosine distance.",
      "- CUDA correlation rows call the public `backend = \"cuda\", method = \"nndescent\", metric = \"correlation\"` route, which row-centers and row-normalizes to float32 before cuVS NN-descent and converts normalized Euclidean distances back to correlation distance.",
      "- CUDA NN-descent correlation shape/k/target defaults are summarized in `benchmark_scripts/cuda_nndescent_correlation_shape_tuning_defaults_from_seeded_euclidean_results.csv`; these rows are validation-pending until a metric-specific CUDA correlation sweep replaces the Euclidean-seeded policy.",
      "- Shape-level defaults for `tuning = \"auto\"` are selected from the fastest candidate that reaches the requested recall target when available; otherwise the highest-recall candidate below target is recorded as a best-available setting."
    )
  }
  if (identical(method, "cagra")) {
    method_notes <- c(
      "",
      "## CAGRA Notes",
      "",
      "- CUDA rows tune CAGRA graph construction and search parameters: `cagra_build_algo`, `cagra_graph_degree`, `cagra_intermediate_graph_degree`, `cagra_search_width`, and `cagra_itopk_size`.",
      "- `cagra_build_algo` is a cuVS CAGRA builder choice (`auto`, `ivf_pq`, `nn_descent`, or `iterative_cagra_search`), not a fallback to a different public faissR method.",
      "- CUDA Euclidean shape/k/target defaults are summarized in `benchmark_scripts/cuda_cagra_euclidean_shape_tuning_defaults_from_uploaded_results.csv` after aggregating the measured CAGRA sweep.",
      "- CUDA cosine rows call the public `backend = \"cuda\", method = \"cagra\", metric = \"cosine\"` route, which row-normalizes the float32 input, runs Euclidean CAGRA graph search, and converts normalized Euclidean distances back to cosine distance.",
      "- CUDA correlation rows call the public `backend = \"cuda\", method = \"cagra\", metric = \"correlation\"` route, which row-centers and row-normalizes the float32 input, runs Euclidean CAGRA graph search, and converts normalized Euclidean distances back to correlation distance.",
      "- CUDA raw-inner-product rows call the public `backend = \"cuda\", method = \"cagra\", metric = \"inner_product\"` route, which applies the maximum-inner-product-to-L2 extra-dimension transform, runs CAGRA graph search, and converts distances back to faissR's shifted inner-product convention.",
      "- CUDA CAGRA correlation shape/k/target defaults are summarized in `benchmark_scripts/cuda_cagra_correlation_shape_tuning_defaults_from_seeded_euclidean_results.csv`; these rows are validation-pending until a metric-specific CUDA correlation sweep replaces the Euclidean-seeded policy.",
      "- CUDA CAGRA raw-inner-product shape/k/target defaults are summarized in `benchmark_scripts/cuda_cagra_inner_product_shape_tuning_defaults_from_seeded_euclidean_results.csv`; these rows are validation-pending until `run_hpc_cagra_tuning_cuda_inner_product.sh` replaces the Euclidean-seeded policy.",
      "- Shape-level defaults for `tuning = \"auto\"` are selected from the fastest candidate that reaches the requested recall target when available; otherwise the highest-recall candidate below target is recorded as a best-available setting.",
      "- If a metric-specific sweep failed before reaching the backend, any seeded defaults must be marked with `tuning_benchmark_target_met = FALSE` until the corrected sweep is rerun."
    )
  }
  if (identical(method, "nsg")) {
    method_notes <- c(
      "",
      "## NSG Notes",
      "",
      "- CPU and CUDA rows tune the native faissR NSG-style pruning degree `nsg_r` and seed/candidate graph width `nsg_graph_k`.",
      "- CUDA rows call the public `backend = \"cuda\", method = \"nsg\"` route, which keeps the NSG pruning rule and uses the native CUDA row-candidate refinement kernel.",
      "- CUDA cosine rows row-normalize the float32 input, run normalized Euclidean NSG refinement, and convert distances back to cosine distance.",
      "- CUDA correlation rows row-center and row-normalize the float32 input before the same CUDA NSG refinement; CUDA raw-inner-product rows use shifted dot-product ordering. Current package defaults for both CUDA correlation and CUDA raw inner product are seeded from the measured CUDA cosine NSG table until their metric-specific sweeps are rerun.",
      "- Shape-level defaults for `tuning = \"auto\"` are selected from the fastest candidate that reaches the requested recall target when available; otherwise the highest-recall candidate below target is recorded as a best-available setting."
    )
  }
  if (identical(method, "vamana")) {
    method_notes <- c(
      "",
      "## Vamana Notes",
      "",
      "- CPU and CUDA rows tune the native faissR Vamana-style robust-pruning degree `vamana_r`, search breadth `vamana_search_l`, and robust-pruning `vamana_alpha`.",
      "- CUDA rows call the public `backend = \"cuda\", method = \"vamana\"` route, which keeps the Vamana pruning rule and uses the native CUDA row-candidate refinement kernel.",
      "- CUDA cosine rows row-normalize the float32 input, run normalized Euclidean Vamana refinement, and convert distances back to cosine distance.",
      "- CUDA correlation rows row-center and row-normalize the float32 input before the same CUDA Vamana refinement; CUDA raw-inner-product rows use shifted dot-product ordering. Current package defaults for both CUDA correlation and CUDA raw inner product are seeded from the measured CUDA cosine Vamana table until their metric-specific sweeps are rerun.",
      "- Shape-level defaults for `tuning = \"auto\"` are selected from the fastest candidate that reaches the requested recall target when available; otherwise the highest-recall candidate below target is recorded as a best-available setting."
    )
  }
  writeLines(c(
    sprintf("# %s Tuning Report", method),
    "",
    sprintf("Generated: %s", format(Sys.time(), "%Y-%m-%d %H:%M:%S %Z")),
    "",
    "References are loaded from max-k dataset-folder files created by benchmark_precompute_exact_references.R for each metric and cropped to the requested k.",
    method_notes,
    "",
    "## Status Counts",
    "",
    paste(capture.output(print(table(x$status, useNA = "ifany"))), collapse = "\n")
  ), file.path(out_dir, sprintf("%s_tuning_report.md", method)))
}

main <- function(args = commandArgs(trailingOnly = TRUE)) {
  args <- parse_args(args)
  if (!is.null(args$child)) return(run_child())
  method <- tolower(args$method %||% Sys.getenv("FAISSR_TUNING_METHOD", unset = ""))
  if (!nzchar(method)) stop("`--method` is required.", call. = FALSE)
  if (identical(method, "scann")) {
    stop("Use `--method=ivfpq_fastscan`; `scann` is not a public faissR method.", call. = FALSE)
  }
  backend <- match.arg(tolower(args$backend %||% "cpu"), c("cpu", "cuda"))
  if (method == "cagra" && backend == "cpu") stop("CAGRA is CUDA-only.", call. = FALSE)
  manifest_path <- normalizePath(args$manifest %||% stop("`--manifest` is required.", call. = FALSE), mustWork = TRUE)
  out_dir <- args$out_dir %||% file.path(getwd(), sprintf("faissR_%s_TUNING_%s_%s", toupper(method), toupper(backend), format(Sys.time(), "%Y%m%d_%H%M%S")))
  dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
  results_path <- file.path(out_dir, sprintf("%s_tuning_results.csv", method))
  candidate_path <- file.path(out_dir, sprintf("%s_tuning_candidate_grid.csv", method))
  bench_script <- normalizePath(script_path(), mustWork = TRUE)
  manifest <- read.csv(manifest_path, stringsAsFactors = FALSE)
  path_col <- dataset_path_column(manifest)
  datasets <- split_arg(args$datasets, paste(manifest$dataset, collapse = ","))
  manifest <- manifest[manifest$dataset %in% datasets, , drop = FALSE]
  k_values <- as.integer(split_arg(args$k_values, "15,30,50,100"))
  k_values <- k_values[is.finite(k_values) & k_values > 0L]
  if (!length(k_values)) stop("`--k_values` must contain at least one positive integer.", call. = FALSE)
  reference_k <- positive_int(args$reference_k, max(c(100L, k_values)), "reference_k")
  if (any(k_values > reference_k)) {
    stop("`--reference_k` must be at least as large as every value in `--k_values`.", call. = FALSE)
  }
  target_recalls <- as.numeric(split_arg(args$target_recalls, "0.9,0.95,0.99"))
  metrics <- metric_values_arg(args$metrics %||% args$metric %||% "euclidean")
  threads <- positive_int(args$threads, 12L, "threads")
  thread_values <- as.integer(split_arg(args$thread_values, if (backend == "cpu") "12" else "12"))
  output_values <- split_arg(args$output_values, args$output %||% "float")
  timeout <- positive_int(args$timeout, 2000L, "timeout")
  quality_n <- positive_int(args$quality_n, 256L, "quality_n")
  seed <- positive_int(args$seed, 4L, "seed")
  grid_level <- args$grid_level %||% "standard"
  ivfpq_fastscan_refine_factors <- positive_int_values(
    args$ivfpq_fastscan_refine_factors,
    default = NULL,
    name = "ivfpq_fastscan_refine_factors"
  )
  ivfpq_fastscan_nlist_multipliers <- positive_num_values(
    args$ivfpq_fastscan_nlist_multipliers,
    default = NULL,
    name = "ivfpq_fastscan_nlist_multipliers"
  )
  ivfpq_fastscan_nprobe_multipliers <- positive_num_values(
    args$ivfpq_fastscan_nprobe_multipliers,
    default = NULL,
    name = "ivfpq_fastscan_nprobe_multipliers"
  )
  ivfpq_fastscan_pq_dim_values <- positive_int_values(
    args$ivfpq_fastscan_pq_dim_values,
    default = NULL,
    name = "ivfpq_fastscan_pq_dim_values"
  )
  ivfpq_fastscan_pq_m_values <- positive_int_values(
    args$ivfpq_fastscan_pq_m_values,
    default = NULL,
    name = "ivfpq_fastscan_pq_m_values"
  )
  ivfpq_fastscan_bbs_values <- positive_int_values(
    args$ivfpq_fastscan_bbs_values,
    default = NULL,
    name = "ivfpq_fastscan_bbs_values"
  )
  cuvs_ivf_batch_sizes <- positive_int_values(
    args$cuvs_ivf_batch_sizes,
    default = if (identical(method, "ivfpq_fastscan") && identical(backend, "cuda")) {
      c(8192L, 16384L, 32768L, 65536L)
    } else {
      NULL
    },
    name = "cuvs_ivf_batch_sizes"
  )
  nndescent_cuda_graph_degrees <- positive_int_values(
    args$nndescent_cuda_graph_degrees,
    default = NULL,
    name = "nndescent_cuda_graph_degrees"
  )
  nndescent_cuda_intermediate_graph_degrees <- positive_int_values(
    args$nndescent_cuda_intermediate_graph_degrees,
    default = NULL,
    name = "nndescent_cuda_intermediate_graph_degrees"
  )
  nndescent_cuda_max_iterations <- positive_int_values(
    args$nndescent_cuda_max_iterations,
    default = NULL,
    name = "nndescent_cuda_max_iterations"
  )
  resume <- logical_arg(args$resume, TRUE)
  resume_statuses <- split_arg(args$resume_statuses, "success,unsupported,unavailable,timeout,skipped_previous_timeout")
  skip_previous_timeouts <- logical_arg(args$skip_previous_timeouts, FALSE)
  skip_timeouts_from <- split_arg(args$skip_timeouts_from, "")
  timeout_keys <- if (isTRUE(skip_previous_timeouts)) timeout_skip_keys(skip_timeouts_from) else character()
  all_candidates <- do.call(rbind, lapply(seq_len(nrow(manifest)), function(i) {
    ds <- manifest[i, , drop = FALSE]
    do.call(rbind, lapply(k_values, function(k) {
      do.call(rbind, lapply(metrics, function(metric) {
        y <- candidate_grid(
          method, backend, as.integer(ds$n), as.integer(ds$p), k, metric,
          target_recalls, thread_values, output_values, grid_level,
          ivfpq_fastscan_refine_factors = ivfpq_fastscan_refine_factors,
          ivfpq_fastscan_nlist_multipliers = ivfpq_fastscan_nlist_multipliers,
          ivfpq_fastscan_nprobe_multipliers = ivfpq_fastscan_nprobe_multipliers,
          ivfpq_fastscan_pq_dim_values = ivfpq_fastscan_pq_dim_values,
          ivfpq_fastscan_pq_m_values = ivfpq_fastscan_pq_m_values,
          ivfpq_fastscan_bbs_values = ivfpq_fastscan_bbs_values,
          cuvs_ivf_batch_sizes = cuvs_ivf_batch_sizes,
          nndescent_cuda_graph_degrees = nndescent_cuda_graph_degrees,
          nndescent_cuda_intermediate_graph_degrees = nndescent_cuda_intermediate_graph_degrees,
          nndescent_cuda_max_iterations = nndescent_cuda_max_iterations
        )
        y$dataset <- ds$dataset[[1L]]
        y$n <- as.integer(ds$n)
        y$p <- as.integer(ds$p)
        y$metric <- metric
        y$k <- as.integer(k)
        y
      }))
    }))
  }))
  write.csv(all_candidates, candidate_path, row.names = FALSE)
  done <- if (resume) completed_keys(results_path, resume_statuses) else character()
  for (i in seq_len(nrow(manifest))) {
    ds <- manifest[i, , drop = FALSE]
    dataset_path <- ds[[path_col]][[1L]]
    for (k in k_values) {
      for (metric in metrics) {
        ref <- load_reference(dataset_path, k, reference_k, quality_n, seed, metric = metric)
        candidates <- candidate_grid(
          method, backend, as.integer(ds$n), as.integer(ds$p), k, metric,
          target_recalls, thread_values, output_values, grid_level,
          ivfpq_fastscan_refine_factors = ivfpq_fastscan_refine_factors,
          ivfpq_fastscan_nlist_multipliers = ivfpq_fastscan_nlist_multipliers,
          ivfpq_fastscan_nprobe_multipliers = ivfpq_fastscan_nprobe_multipliers,
          ivfpq_fastscan_pq_dim_values = ivfpq_fastscan_pq_dim_values,
          ivfpq_fastscan_pq_m_values = ivfpq_fastscan_pq_m_values,
          ivfpq_fastscan_bbs_values = ivfpq_fastscan_bbs_values,
          cuvs_ivf_batch_sizes = cuvs_ivf_batch_sizes,
          nndescent_cuda_graph_degrees = nndescent_cuda_graph_degrees,
          nndescent_cuda_intermediate_graph_degrees = nndescent_cuda_intermediate_graph_degrees,
          nndescent_cuda_max_iterations = nndescent_cuda_max_iterations
        )
        for (j in seq_len(nrow(candidates))) {
          cand <- candidates[j, , drop = FALSE]
          key <- paste(ds$dataset[[1L]], backend, method, metric, as.integer(k), cand$candidate_id, sep = "\r")
          timeout_key <- paste(ds$dataset[[1L]], backend, method, as.integer(k), cand$candidate_id, sep = "\r")
          if (key %in% done) next
          cfg <- list(
            dataset = ds$dataset[[1L]], dataset_path = dataset_path,
            n = as.integer(ds$n), p = as.integer(ds$p), backend = backend,
            method = method, metric = metric, k = as.integer(k),
            target_recalls = target_recalls,
            reference_rows = ref$rows, reference_indices = ref$indices,
            reference_status = ref$status %||% NA_character_,
            reference_path = ref$path %||% NA_character_,
            reference_query_n = length(ref$rows %||% integer()),
            candidate = cand
          )
          row <- if (timeout_key %in% timeout_keys) {
            base_row(cfg, "skipped_previous_timeout", "Skipped because this method/dataset/k/candidate timed out in an earlier tuning run.")
          } else if (identical(ref$status, "success")) {
            run_task(cfg, timeout, bench_script)
          } else {
            base_row(cfg, "missing_reference", ref$error %||% "missing reference")
          }
          append_csv(row, results_path)
          done <- c(done, key)
        }
        summarize_results(out_dir, results_path, target_recalls, method)
      }
    }
  }
  summarize_results(out_dir, results_path, target_recalls, method)
  message("DONE: ", out_dir)
}

if (!identical(Sys.getenv("FAISSR_TUNING_SOURCE_ONLY", unset = ""), "true")) {
  main()
}
