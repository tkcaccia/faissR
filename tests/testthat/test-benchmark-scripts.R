source_benchmark_helpers <- function(path, stop_marker) {
  if (!file.exists(path)) {
    testthat::skip("Benchmark scripts are not available in this installed-package test context.")
  }
  old_out <- Sys.getenv("FAISSR_BENCHMARK_OUT", unset = NA_character_)
  tmp_out <- tempfile("faissR_benchmark_test_")
  Sys.setenv(FAISSR_BENCHMARK_OUT = tmp_out)
  on.exit({
    if (is.na(old_out)) {
      Sys.unsetenv("FAISSR_BENCHMARK_OUT")
    } else {
      Sys.setenv(FAISSR_BENCHMARK_OUT = old_out)
    }
    unlink(tmp_out, recursive = TRUE, force = TRUE)
  }, add = TRUE)
  lines <- readLines(path, warn = FALSE)
  stop_at <- grep(stop_marker, lines, fixed = TRUE)[1L] - 1L
  env <- new.env(parent = globalenv())
  conn <- textConnection(lines[seq_len(stop_at)])
  on.exit(close(conn), add = TRUE)
  source(conn, local = env)
  env
}

test_that("benchmark materials document key row-level and summary outputs", {
  files <- list(
    nn = c(
      "nn_metric_benchmark_config.csv",
      "nn_metric_benchmark_results.csv",
      "nn_metric_capabilities.csv",
      "nn_metric_cycle_summary.csv",
      "nn_metric_recommendations_from_cycles.csv",
      "nn_metric_auto_vs_cycle_recommendation.csv",
      "nn_metric_global_recommendations_from_cycles.csv",
      "nn_metric_auto_vs_global_recommendation.csv",
      "nn_metric_best_by_dataset_backend_metric_k.csv",
      "nn_metric_best_by_dataset_backend_metric_k_cycle.csv",
      "nn_metric_fastest_at_recall_threshold.csv",
      "nn_metric_auto_vs_fastest.csv",
      "MATERIALS_AND_METHODS_nn_metrics.md"
    ),
    graph = c(
      "graph_cluster_benchmark_config.csv",
      "graph_cluster_nn_capabilities.csv",
      "graph_cluster_benchmark_results.csv",
      "graph_cluster_best_by_dataset.csv",
      "graph_cluster_best_by_dataset_k_target.csv",
      "graph_cluster_cycle_summary.csv",
      "graph_cluster_recommendations_from_cycles.csv",
      "graph_cluster_auto_vs_cycle_recommendation.csv",
      "graph_cluster_global_recommendations_from_cycles.csv",
      "graph_cluster_auto_vs_global_recommendation.csv",
      "MATERIALS_AND_METHODS_graph_clustering.md"
    ),
    kmeans = c(
      "kmeans_benchmark_config.csv",
      "kmeans_runtime_capabilities.csv",
      "kmeans_benchmark_results.csv",
      "kmeans_best_by_dataset.csv",
      "kmeans_best_by_dataset_centers.csv",
      "kmeans_cycle_summary.csv",
      "kmeans_recommendations_from_cycles.csv",
      "kmeans_backend_recommendations_from_cycles.csv",
      "kmeans_fast_vs_cycle_recommendation.csv",
      "kmeans_auto_vs_global_recommendation.csv",
      "kmeans_fast_vs_stats.csv",
      "MATERIALS_AND_METHODS_kmeans.md"
    )
  )
  scripts <- c(
    nn = test_path("../../benchmark_scripts/benchmark_nn_metrics.R"),
    graph = test_path("../../benchmark_scripts/benchmark_graph_clustering.R"),
    kmeans = test_path("../../benchmark_scripts/benchmark_kmeans.R")
  )
  docs_file <- test_path("../../docs/benchmarks.md")
  if (!all(file.exists(scripts)) || !file.exists(docs_file)) {
    skip("Benchmark scripts or GitHub benchmark documentation are not available in this installed-package test context.")
  }

  docs <- paste(readLines(docs_file, warn = FALSE), collapse = "\n")
  for (name in names(files)) {
    script <- paste(readLines(scripts[[name]], warn = FALSE), collapse = "\n")
    for (file in files[[name]]) {
      expect_true(grepl(file, script, fixed = TRUE), info = scripts[[name]])
      expect_true(grepl(file, docs, fixed = TRUE), info = docs_file)
    }
  }
})

test_that("benchmark scripts parse before long-running execution", {
  scripts <- test_path("../../benchmark_scripts", c(
    "benchmark_nn_metrics.R",
    "benchmark_graph_clustering.R",
    "benchmark_kmeans.R",
    "benchmark1_nn_speed.R"
  ))
  missing <- scripts[!file.exists(scripts)]
  if (length(missing)) {
    skip("Benchmark scripts are not available in this installed-package test context.")
  }

  for (script in scripts) {
    expect_error(parse(script), NA, info = basename(script))
  }
})

test_that("benchmark dataset defaults use the requested real and simulated datasets", {
  real_datasets <- c(
    "COIL20",
    "USPS",
    "FashionMNIST",
    "FlowRepository_FR-FCM-ZYRM_files",
    "flow18",
    "MNIST",
    "imagenet",
    "MetRef",
    "mass41"
  )
  scripts <- list(
    nn = list(
      path = test_path("../../benchmark_scripts/benchmark_nn_metrics.R"),
      stop = "args <- parse_args()",
      simulated = c("SimulatedUniform2D", "SimulatedUniform3D")
    ),
    graph = list(
      path = test_path("../../benchmark_scripts/benchmark_graph_clustering.R"),
      stop = "args <- parse_args()",
      simulated = c("SimulatedUniform2D", "SimulatedUniform3D", "SimulatedTiny3Clusters")
    ),
    kmeans = list(
      path = test_path("../../benchmark_scripts/benchmark_kmeans.R"),
      stop = "args <- parse_args()",
      simulated = "SimulatedTiny3Clusters"
    )
  )

  for (name in names(scripts)) {
    script <- scripts[[name]]
    env <- source_benchmark_helpers(script$path, script$stop)
    script_text <- paste(readLines(script$path, warn = FALSE), collapse = "\n")
    expect_equal(env$dataset_index("/data")$dataset, real_datasets, info = name)
    expect_true(grepl("- Default real datasets:", script_text, fixed = TRUE), info = name)
    expect_true(grepl("- Default simulated datasets:", script_text, fixed = TRUE), info = name)
    default_datasets <- env$split_arg(
      NULL,
      paste(c(env$dataset_index("/data")$dataset, script$simulated), collapse = ",")
    )
    expect_equal(default_datasets, c(real_datasets, script$simulated), info = name)
    expect_true(grepl("\"available_datasets\"", script_text, fixed = TRUE), info = name)
    expect_true(grepl("accepted by the dataset selector", script_text, fixed = TRUE), info = name)
  }
})

test_that("NN metric benchmark preflights unsupported NSG metrics as expected skips", {
  env <- source_benchmark_helpers(
    test_path("../../benchmark_scripts/benchmark_nn_metrics.R"),
    "args <- parse_args()"
  )
  caps <- nn_capabilities()

  for (backend in c("auto", "cpu")) {
    for (metric in c("cosine", "correlation", "inner_product")) {
      skip <- env$is_expected_skip(caps, backend, "nsg", metric)
      expect_type(skip, "list")
      expect_true(isTRUE(skip$skip))
      expect_true(nzchar(skip$reason))
      expect_match(skip$notes, "euclidean|non-Euclidean|unsupported|rejected|No CPU or CUDA route", ignore.case = TRUE)
    }
  }
})

test_that("NN metric benchmark preflights auto rows from nn_capabilities", {
  env <- source_benchmark_helpers(
    test_path("../../benchmark_scripts/benchmark_nn_metrics.R"),
    "args <- parse_args()"
  )
  caps <- nn_capabilities()

  skip <- env$is_expected_skip(caps, "auto", "nsg", "inner_product")
  expect_type(skip, "list")
  expect_true(isTRUE(skip$skip))
  expect_true(nzchar(skip$reason))
  expect_match(skip$notes, "No CPU or CUDA route|unsupported|not exposed", ignore.case = TRUE)

  auto_cap <- env$capability_status(caps, "auto", "flat", "inner_product")
  expect_true(isTRUE(auto_cap$supported))
  skip <- env$is_expected_skip(caps, "auto", "flat", "inner_product")
  if (!is.null(skip)) {
    expect_true(isTRUE(skip$skip))
    expect_match(skip$notes, "requires|unavailable|resolver", ignore.case = TRUE)
  }
})

test_that("NN metric benchmark preflights every unsupported capability row", {
  env <- source_benchmark_helpers(
    test_path("../../benchmark_scripts/benchmark_nn_metrics.R"),
    "args <- parse_args()"
  )
  caps <- nn_capabilities()
  unsupported <- caps[!caps$supported, , drop = FALSE]
  expect_gt(nrow(unsupported), 0L)

  for (i in seq_len(nrow(unsupported))) {
    row <- unsupported[i, , drop = FALSE]
    label <- sprintf("%s/%s/%s", row$backend, row$method, row$metric)
    skip <- env$is_expected_skip(caps, row$backend, row$method, row$metric)
    expect_type(skip, "list")
    expect_true(
      isTRUE(skip$skip),
      info = label
    )
    expect_true(
      nzchar(skip$notes),
      info = label
    )
  }
})

test_that("NN metric benchmark preflights supported rows as runnable or runtime skips", {
  env <- source_benchmark_helpers(
    test_path("../../benchmark_scripts/benchmark_nn_metrics.R"),
    "args <- parse_args()"
  )
  caps <- nn_capabilities()
  supported <- caps[caps$supported, , drop = FALSE]
  expect_gt(nrow(supported), 0L)

  for (i in seq_len(nrow(supported))) {
    row <- supported[i, , drop = FALSE]
    label <- sprintf("%s/%s/%s", row$backend, row$method, row$metric)
    skip <- env$is_expected_skip(caps, row$backend, row$method, row$metric)
    if (is.null(skip)) {
      resolved <- faissR:::resolve_public_nn_backend(row$backend, row$method, row$metric)
      expect_type(resolved, "character")
      expect_true(nzchar(resolved), info = label)
    } else {
      expect_type(skip, "list")
      expect_true(isTRUE(skip$skip), info = label)
      expect_true(nzchar(skip$notes), info = label)
    }
  }
})

test_that("NN metric benchmark consumes runtime capability columns when present", {
  env <- source_benchmark_helpers(
    test_path("../../benchmark_scripts/benchmark_nn_metrics.R"),
    "args <- parse_args()"
  )
  caps <- nn_capabilities(runtime = TRUE)

  cpu_flat <- env$capability_status(caps, "cpu", "flat", "euclidean")
  expect_true(isTRUE(cpu_flat$supported))
  expect_equal(cpu_flat$resolved_backend, "faiss_flat_l2")
  expect_equal(cpu_flat$runtime_available, faiss_available())
  expect_equal(cpu_flat$runtime_reason, if (faiss_available()) "available" else "missing_faiss")
  expect_true(nzchar(cpu_flat$runtime_notes))

  cuda_flat <- env$is_expected_skip(caps, "cuda", "flat", "euclidean")
  if (isTRUE(faiss_gpu_available())) {
    expect_null(cuda_flat)
  } else {
    expect_type(cuda_flat, "list")
    expect_true(isTRUE(cuda_flat$skip))
    expect_equal(cuda_flat$route, "faiss_gpu_flat_l2")
    expect_match(cuda_flat$notes, "FAISS GPU|not available|unavailable", ignore.case = TRUE)
  }
})

test_that("NN metric benchmark recommendations are grouped by backend metric and k", {
  env <- source_benchmark_helpers(
    test_path("../../benchmark_scripts/benchmark_nn_metrics.R"),
    "args <- parse_args()"
  )
  cycle_summary <- data.frame(
    dataset = c("A", "A", "A", "A", "A", "A", "B", "B", "C", "C"),
    backend = c("cpu", "cpu", "cpu", "cpu", "cuda", "cuda", "cpu", "cpu", "cpu", "cpu"),
    method = c("exact", "hnsw", "exact", "hnsw", "flat", "cagra", "ivf", "hnsw", "flat", "ivf"),
    metric = c("euclidean", "euclidean", "cosine", "cosine", "euclidean", "euclidean", "euclidean", "euclidean", "euclidean", "euclidean"),
    k = c(15L, 15L, 15L, 15L, 50L, 50L, 15L, 15L, 100L, 100L),
    median_elapsed_sec = c(10, 2, 8, 3, 4, 1, 1, 3, 8, 2),
    median_recall_at_k = c(1.00, 0.99, 1.00, 0.97, 1.00, 0.99, 0.94, 0.96, NA, NA),
    min_recall_at_k = c(1.00, 0.98, 1.00, 0.96, 1.00, 0.98, 0.93, 0.95, NA, NA),
    median_min_recall_at_k = c(1.00, 0.98, 1.00, 0.96, 1.00, 0.98, 0.93, 0.95, NA, NA),
    recall_reference = c("exact", "exact", "exact", "exact", "exact", "exact", "exact", "exact", NA, NA),
    median_recall_query_n = c(100, 100, 100, 100, 100, 100, 100, 100, NA, NA),
    result_backend = c("cpu", "faiss_hnsw", "cpu", "faiss_hnsw", "faiss_gpu_flat_l2", "faiss_gpu_cagra", "faiss_ivf", "faiss_hnsw", "faiss_flat_l2", "faiss_ivf"),
    resolved_backend = c("cpu", "faiss_hnsw", "cpu", "faiss_hnsw", "faiss_gpu_flat_l2", "faiss_gpu_cagra", "faiss_ivf", "faiss_hnsw", "faiss_flat_l2", "faiss_ivf"),
    implementation_backend = c("cpu", "faiss_hnsw", "cpu", "faiss_hnsw", "faiss_gpu_flat_l2", "faiss_gpu_cagra", "faiss_ivf", "faiss_hnsw", "faiss_flat_l2", "faiss_ivf"),
    success_cycles = rep(2L, 10)
  )

  out <- env$recommend_nn_methods(cycle_summary, recall_threshold = 0.98)
  expect_equal(nrow(out), 5L)
  expect_equal(out$dataset, c("A", "A", "A", "B", "C"))
  expect_equal(out$backend, c("cpu", "cpu", "cuda", "cpu", "cpu"))
  expect_equal(out$metric, c("cosine", "euclidean", "euclidean", "euclidean", "euclidean"))
  expect_equal(as.integer(out$k), c(15L, 15L, 50L, 15L, 100L))
  expect_equal(out$method, c("exact", "hnsw", "cagra", "hnsw", "ivf"))
  expect_equal(
    out$recommendation_basis,
    c(
      "fastest_at_recall_threshold",
      "fastest_at_recall_threshold",
      "fastest_at_recall_threshold",
      "best_recall_below_threshold",
      "speed_only_no_recall"
    )
  )
})

test_that("NN metric benchmark recommendation ties prefer stronger recall stability", {
  env <- source_benchmark_helpers(
    test_path("../../benchmark_scripts/benchmark_nn_metrics.R"),
    "args <- parse_args()"
  )
  cycle_summary <- data.frame(
    dataset = c("A", "A", "B", "B"),
    backend = c("cpu", "cpu", "cpu", "cpu"),
    method = c("lower_recall", "higher_recall", "lower_min_recall", "higher_min_recall"),
    metric = c("euclidean", "euclidean", "cosine", "cosine"),
    k = c(15L, 15L, 15L, 15L),
    median_elapsed_sec = c(1, 1, 1, 1),
    median_recall_at_k = c(0.985, 0.990, 0.960, 0.960),
    min_recall_at_k = c(0.980, 0.980, 0.940, 0.950),
    median_min_recall_at_k = c(0.980, 0.980, 0.940, 0.950),
    recall_reference = c("exact", "exact", "exact", "exact"),
    median_recall_query_n = c(100, 100, 100, 100),
    result_backend = c("cpu", "cpu", "cpu", "cpu"),
    resolved_backend = c("cpu", "cpu", "cpu", "cpu"),
    implementation_backend = c("cpu", "cpu", "cpu", "cpu"),
    success_cycles = c(2L, 2L, 2L, 2L)
  )

  out <- env$recommend_nn_methods(cycle_summary, recall_threshold = 0.98)
  expect_equal(out$method, c("higher_recall", "higher_min_recall"))
  expect_equal(
    out$recommendation_basis,
    c("fastest_at_recall_threshold", "best_recall_below_threshold")
  )
})

test_that("NN metric cycle summaries preserve route and thread metadata", {
  env <- source_benchmark_helpers(
    test_path("../../benchmark_scripts/benchmark_nn_metrics.R"),
    "args <- parse_args()"
  )
  ok <- data.frame(
    dataset = c("A", "A"),
    backend = c("cpu", "cpu"),
    method = c("auto", "auto"),
    metric = c("euclidean", "euclidean"),
    k = c(50L, 50L),
    n = c(100L, 100L),
    p = c(10L, 10L),
    n_threads = c(2L, 2L),
    cycle = c(1L, 2L),
    elapsed_sec = c(1, 2),
    recall_at_k = c(0.99, 0.98),
    min_recall_at_k = c(0.97, 0.96),
    recall_query_n = c(100L, 100L),
    exact = c(FALSE, FALSE),
    result_backend = c("faiss_hnsw", "faiss_hnsw"),
    result_requested_backend = c("auto", "auto"),
    result_requested_method = c("auto", "auto"),
    result_tuning = c("auto", "auto"),
    resolved_backend = c("faiss_hnsw", "faiss_hnsw"),
    implementation_backend = c("faiss_hnsw", "faiss_hnsw"),
    preflight_route = c("faiss_hnsw", "faiss_hnsw"),
    recall_reference = c("sample", "sample")
  )

  out <- env$summarize_nn_cycles(ok)
  expect_equal(out$n_threads, 2L)
  expect_equal(out$result_requested_backend, "auto")
  expect_equal(out$result_requested_method, "auto")
  expect_equal(out$result_tuning, "auto")
  expect_equal(out$preflight_route, "faiss_hnsw")
  expect_equal(out$success_cycles, 2L)
})

test_that("NN metric best-row ranking prefers recall stability before speed", {
  env <- source_benchmark_helpers(
    test_path("../../benchmark_scripts/benchmark_nn_metrics.R"),
    "args <- parse_args()"
  )
  ok <- data.frame(
    dataset = c("A", "A", "A", "B", "B"),
    backend = c("cpu", "cpu", "cpu", "cuda", "cuda"),
    method = c("fast_unstable", "slow_stable", "lower_recall", "fast_missing_recall", "slow_missing_recall"),
    metric = c("euclidean", "euclidean", "euclidean", "cosine", "cosine"),
    k = c(50L, 50L, 50L, 15L, 15L),
    cycle = c(1L, 1L, 1L, 1L, 1L),
    recall_at_k = c(0.99, 0.99, 0.98, NA, NA),
    min_recall_at_k = c(0.80, 0.95, 0.98, NA, NA),
    elapsed_sec = c(1, 2, 0.5, 1, 2),
    stringsAsFactors = FALSE
  )

  ranked <- env$rank_nn_metric_success(ok)
  best <- ranked[!duplicated(paste(ranked$dataset, ranked$backend, ranked$metric, ranked$k, sep = "__")), ]
  expect_equal(best$method, c("slow_stable", "fast_missing_recall"))

  ranked_cycle <- env$rank_nn_metric_success(ok, include_cycle = TRUE)
  best_cycle <- ranked_cycle[!duplicated(paste(ranked_cycle$dataset, ranked_cycle$backend, ranked_cycle$metric, ranked_cycle$k, ranked_cycle$cycle, sep = "__")), ]
  expect_equal(best_cycle$method, c("slow_stable", "fast_missing_recall"))
})

test_that("NN metric best rows use threshold-aware speed and recall rules", {
  env <- source_benchmark_helpers(
    test_path("../../benchmark_scripts/benchmark_nn_metrics.R"),
    "args <- parse_args()"
  )
  ok <- data.frame(
    dataset = c("A", "A", "B", "B", "C", "C", "D", "D"),
    backend = c("cpu", "cpu", "cpu", "cpu", "cuda", "cuda", "cuda", "cuda"),
    method = c(
      "slow_perfect", "fast_good",
      "lower_recall_fast", "higher_recall_slow",
      "fast_missing_recall", "slow_missing_recall",
      "cycle1_fast_good", "cycle2_fast_good"
    ),
    metric = c("euclidean", "euclidean", "cosine", "cosine", "correlation", "correlation", "euclidean", "euclidean"),
    k = c(50L, 50L, 15L, 15L, 10L, 10L, 5L, 5L),
    cycle = c(1L, 1L, 1L, 1L, 1L, 1L, 1L, 2L),
    recall_at_k = c(1.00, 0.99, 0.94, 0.96, NA, NA, 0.99, 0.99),
    min_recall_at_k = c(1.00, 0.98, 0.93, 0.95, NA, NA, 0.98, 0.98),
    elapsed_sec = c(4, 1, 1, 3, 1, 2, 2, 1),
    stringsAsFactors = FALSE
  )

  best <- env$select_nn_metric_best_rows(ok, recall_threshold = 0.98)
  expect_equal(best$method, c("fast_good", "higher_recall_slow", "fast_missing_recall", "cycle2_fast_good"))

  best_cycle <- env$select_nn_metric_best_rows(ok, recall_threshold = 0.98, include_cycle = TRUE)
  expect_equal(best_cycle$method, c("fast_good", "higher_recall_slow", "fast_missing_recall", "cycle1_fast_good", "cycle2_fast_good"))
})

test_that("NN metric benchmark canonicalizes metric aliases before preflight", {
  env <- source_benchmark_helpers(
    test_path("../../benchmark_scripts/benchmark_nn_metrics.R"),
    "args <- parse_args()"
  )
  caps <- nn_capabilities()

  expect_equal(
    env$canonical_metric_values(c("l2", "pearson", "ip", "dot-product", "unknown")),
    c("euclidean", "correlation", "inner_product")
  )
  expect_equal(
    env$validate_metric_values(c("l2", "pearson", "ip", "dot-product")),
    c("euclidean", "correlation", "inner_product")
  )
  expect_error(
    env$validate_metric_values(c("euclidean", "manhattan")),
    "Invalid value\\(s\\): manhattan"
  )
  expect_error(
    env$validate_metric_values(character()),
    "at least one metric"
  )
  expect_true(isTRUE(env$capability_status(caps, "cpu", "flat", "l2")$supported))
  expect_true(isTRUE(env$capability_status(caps, "cpu", "flat", "pearson")$supported))
  expect_true(isTRUE(env$capability_status(caps, "cpu", "flat", "ip")$supported))
  expect_false(isTRUE(env$capability_status(caps, "cpu", "nsg", "ip")$supported))
})

test_that("NN metric benchmark validates public backend labels", {
  env <- source_benchmark_helpers(
    test_path("../../benchmark_scripts/benchmark_nn_metrics.R"),
    "args <- parse_args()"
  )

  expect_equal(
    env$validate_backend_values(c("auto", "cpu", "cuda", "cpu")),
    c("auto", "cpu", "cuda")
  )
  expect_error(
    env$validate_backend_values(c("cpu", "gpu")),
    "Invalid value\\(s\\): gpu"
  )
  expect_error(
    env$validate_backend_values(character()),
    "at least one backend"
  )
})

test_that("NN metric benchmark requires canonical public method labels", {
  env <- source_benchmark_helpers(
    test_path("../../benchmark_scripts/benchmark_nn_metrics.R"),
    "args <- parse_args()"
  )
  caps <- nn_capabilities()

  expect_equal(
    env$canonical_method_values(c("exact", "hnsw", "cagra")),
    c("exact", "hnsw", "cagra")
  )
  expect_error(
    env$canonical_method_values(c("HNSW", "faiss_hnsw")),
    "canonical lowercase"
  )
  expect_error(
    env$canonical_method_values(c("HNSW", "faiss_hnsw")),
    "Valid value\\(s\\): auto, exact, flat, bruteforce, grid, vptree, sparse, hnsw, ivf, ivfpq, nsg, nndescent, cagra"
  )
  expect_error(
    env$canonical_method_values(character()),
    "at least one method"
  )
  expect_false(isTRUE(env$capability_status(caps, "cpu", "HNSW", "euclidean")$supported))
  expect_false(isTRUE(env$capability_status(caps, "cpu", "faiss_hnsw", "euclidean")$supported))
})

test_that("NN metric benchmark validates dataset selectors", {
  env <- source_benchmark_helpers(
    test_path("../../benchmark_scripts/benchmark_nn_metrics.R"),
    "args <- parse_args()"
  )
  valid <- c("COIL20", "USPS", "SimulatedUniform2D", "SimulatedUniform3D", "SimulatedTiny3Clusters")

  expect_equal(
    env$validate_dataset_values(c("COIL20", "USPS", "COIL20"), valid),
    c("COIL20", "USPS")
  )
  expect_error(
    env$validate_dataset_values(c("COIL20", "bad_dataset"), valid),
    "Invalid value\\(s\\): bad_dataset"
  )
  expect_error(
    env$validate_dataset_values(character(), valid),
    "at least one dataset"
  )
})

test_that("NN metric benchmark defaults cover requested metrics and k grid", {
  env <- source_benchmark_helpers(
    test_path("../../benchmark_scripts/benchmark_nn_metrics.R"),
    "args <- parse_args()"
  )

  expect_equal(
    env$default_nn_method_values(),
    faissR:::nn_method_labels()
  )
  expect_equal(
    env$default_nn_backend_values(),
    c("auto", "cpu", "cuda")
  )
  expect_equal(
    env$default_nn_metric_values(),
    c("euclidean", "cosine", "correlation", "inner_product")
  )
  expect_equal(
    env$default_nn_k_values(),
    c(5L, 10L, 15L, 50L, 100L)
  )
  expect_equal(env$default_nn_cycles(), 10L)
  expect_equal(
    env$canonical_metric_values(c("unknown")),
    character()
  )
  expect_equal(
    env$as_int_vec_arg(c("unknown"), env$default_nn_k_values()),
    env$default_nn_k_values()
  )
  expect_equal(
    env$required_positive_int_values(c("5", "10", "10"), "k_values"),
    c(5L, 10L)
  )
  expect_error(
    env$required_positive_int_values(c("5", "zero"), "k_values"),
    "Invalid value\\(s\\): zero"
  )
  expect_error(
    env$required_positive_int_values(c("0"), "k_values"),
    "Invalid value\\(s\\): 0"
  )
  expect_error(
    env$required_positive_int_values(c("15.5"), "k_values"),
    "Invalid value\\(s\\): 15.5"
  )
  expect_error(
    env$required_positive_int_values(character(), "k_values"),
    "at least one positive integer"
  )
  expect_equal(env$required_positive_int_arg("12", "threads"), 12L)
  expect_equal(env$required_positive_int_arg("600", "timeout"), 600L)
  expect_equal(env$required_positive_int_arg("10", "cycles"), 10L)
  expect_equal(env$required_positive_int_arg("512", "quality_n"), 512L)
  expect_equal(env$required_positive_int_arg("42", "seed"), 42L)
  expect_error(env$required_positive_int_arg("many", "cycles"), "positive integer")
  expect_error(env$required_positive_int_arg("1.5", "cycles"), "positive integer")
  expect_error(env$required_positive_int_arg(0, "quality_n"), "positive integer")
  expect_error(env$required_positive_int_arg("many", "seed"), "positive integer")
  expect_error(env$required_positive_int_arg("1.5", "seed"), "positive integer")
  expect_error(env$required_positive_int_arg(0, "seed"), "positive integer")
  expect_equal(env$required_positive_numeric_arg("5e9", "quality_max_ops"), 5e9)
  expect_equal(env$required_positive_numeric_arg("1000", "quality_max_ops"), 1000)
  expect_error(
    env$required_positive_numeric_arg("many", "quality_max_ops"),
    "positive numeric"
  )
  expect_error(
    env$required_positive_numeric_arg(0, "quality_max_ops"),
    "positive numeric"
  )
  expect_equal(env$required_probability_arg("0.98", "recall_threshold"), 0.98)
  expect_equal(env$required_probability_arg(0, "recall_threshold"), 0)
  expect_equal(env$required_probability_arg(1, "recall_threshold"), 1)
  expect_error(
    env$required_probability_arg("high", "recall_threshold"),
    "between 0 and 1"
  )
  expect_error(
    env$required_probability_arg(1.1, "recall_threshold"),
    "between 0 and 1"
  )
})

test_that("NN metric auto comparison preserves recommendation basis", {
  env <- source_benchmark_helpers(
    test_path("../../benchmark_scripts/benchmark_nn_metrics.R"),
    "args <- parse_args()"
  )
  cycle_summary <- data.frame(
    dataset = c("A", "A"),
    backend = c("cpu", "cpu"),
    method = c("auto", "hnsw"),
    metric = c("euclidean", "euclidean"),
    k = c(50L, 50L),
    result_backend = c("faiss_hnsw", "faiss_hnsw"),
    result_requested_backend = c("auto", "cpu"),
    result_requested_method = c("auto", "hnsw"),
    result_tuning = c("auto", "auto"),
    auto_predicted_method = c("hnsw", NA_character_),
    auto_predicted_device = c("cpu", NA_character_),
    auto_explicit_backend = c(FALSE, NA),
    auto_explicit_method = c(FALSE, NA),
    auto_backend_decision = c("auto_cpu_fallback", NA_character_),
    auto_method_decision = c("auto_cpu_fallback", NA_character_),
    resolved_backend = c("faiss_hnsw", "faiss_hnsw"),
    implementation_backend = c("faiss_hnsw", "faiss_hnsw"),
    preflight_route = c("faiss_hnsw", "faiss_hnsw"),
    route_parameters = c("approximation.m=16;approximation.ef_search=150", "approximation.m=16;approximation.ef_search=150"),
    tuning_status = c(NA_character_, NA_character_),
    n_threads = c(2L, 2L),
    success_cycles = c(2L, 2L),
    median_elapsed_sec = c(5, 4),
    median_recall_at_k = c(0.99, 0.99),
    min_recall_at_k = c(0.98, 0.98),
    median_min_recall_at_k = c(0.98, 0.98),
    recall_reference = c("sample", "sample"),
    median_recall_query_n = c(512, 512)
  )
  recommendations <- cycle_summary[2, , drop = FALSE]
  recommendations$recommendation_basis <- "fastest_at_recall_threshold"

  out <- env$compare_auto_to_recommendations(cycle_summary, recommendations)

  expect_equal(anyDuplicated(names(out)), 0L)
  expect_equal(out$recommended_recommendation_basis, "fastest_at_recall_threshold")
  expect_true("recommended_recommendation_basis" %in% names(out))
  expect_equal(out$auto_result_requested_backend, "auto")
  expect_equal(out$recommended_result_requested_backend, "cpu")
  expect_equal(out$auto_result_requested_method, "auto")
  expect_equal(out$recommended_result_requested_method, "hnsw")
  expect_equal(out$auto_result_tuning, "auto")
  expect_equal(out$auto_auto_predicted_method, "hnsw")
  expect_equal(out$auto_auto_predicted_device, "cpu")
  expect_false(out$auto_auto_explicit_backend)
  expect_false(out$auto_auto_explicit_method)
  expect_equal(out$auto_auto_backend_decision, "auto_cpu_fallback")
  expect_equal(out$auto_auto_method_decision, "auto_cpu_fallback")
  expect_equal(out$auto_preflight_route, "faiss_hnsw")
  expect_equal(out$recommended_preflight_route, "faiss_hnsw")
  expect_equal(out$auto_route_parameters, "approximation.m=16;approximation.ef_search=150")
  expect_equal(out$recommended_route_parameters, "approximation.m=16;approximation.ef_search=150")
  expect_equal(out$auto_n_threads, 2L)
  expect_equal(out$recommended_n_threads, 2L)
})

test_that("NN metric global recommendations pool across requested backends", {
  env <- source_benchmark_helpers(
    test_path("../../benchmark_scripts/benchmark_nn_metrics.R"),
    "args <- parse_args()"
  )
  cycle_summary <- data.frame(
    dataset = c("A", "A", "A", "A"),
    backend = c("auto", "cpu", "cuda", "cuda"),
    method = c("auto", "hnsw", "flat", "ivf"),
    metric = c("euclidean", "euclidean", "euclidean", "euclidean"),
    k = c(50L, 50L, 50L, 50L),
    result_backend = c("faiss_hnsw", "faiss_hnsw", "faiss_gpu_flat_l2", "faiss_gpu_ivf_flat"),
    result_requested_backend = c("auto", "cpu", "cuda", "cuda"),
    result_requested_method = c("auto", "hnsw", "flat", "ivf"),
    result_tuning = c("auto", "auto", "auto", "auto"),
    auto_predicted_method = c("hnsw", NA_character_, NA_character_, NA_character_),
    auto_predicted_device = c("cpu", NA_character_, NA_character_, NA_character_),
    resolved_backend = c("faiss_hnsw", "faiss_hnsw", "faiss_gpu_flat_l2", "faiss_gpu_ivf_flat"),
    implementation_backend = c("faiss_hnsw", "faiss_hnsw", "faiss_gpu_flat_l2", "faiss_gpu_ivf_flat"),
    preflight_route = c("faiss_hnsw", "faiss_hnsw", "faiss_gpu_flat_l2", "faiss_gpu_ivf_flat"),
    route_parameters = c("hnsw", "hnsw", "flat", "ivf"),
    tuning_status = c("balanced", "balanced", NA_character_, "fixed"),
    n_threads = c(2L, 2L, 2L, 2L),
    success_cycles = c(2L, 2L, 2L, 2L),
    median_elapsed_sec = c(4, 5, 2, 3),
    median_recall_at_k = c(0.99, 0.995, 0.99, 0.96),
    min_recall_at_k = c(0.98, 0.99, 0.98, 0.95),
    median_min_recall_at_k = c(0.98, 0.99, 0.98, 0.95),
    recall_reference = c("sample", "sample", "sample", "sample"),
    median_recall_query_n = c(512, 512, 512, 512)
  )

  rec <- env$recommend_nn_global_methods(cycle_summary, recall_threshold = 0.98)
  expect_equal(nrow(rec), 1L)
  expect_equal(rec$backend, "cuda")
  expect_equal(rec$method, "flat")
  expect_equal(rec$recommendation_basis, "global_fastest_at_recall_threshold")

  out <- env$compare_auto_to_global_recommendations(cycle_summary, rec)
  expect_equal(anyDuplicated(names(out)), 0L)
  expect_true("auto_backend" %in% names(out))
  expect_true("recommended_backend" %in% names(out))
  auto_row <- out[out$auto_method == "auto", , drop = FALSE]
  expect_equal(nrow(auto_row), 1L)
  expect_equal(auto_row$recommended_backend, "cuda")
  expect_false(auto_row$auto_uses_recommended_requested_backend)
  expect_false(auto_row$auto_uses_recommended_implementation)
  expect_equal(auto_row$auto_median_speed_ratio, 2)
  expect_equal(auto_row$auto_median_recall_gap, 0)
})

test_that("NN metric benchmark extracts compact backend route parameters", {
  env <- source_benchmark_helpers(
    test_path("../../benchmark_scripts/benchmark_nn_metrics.R"),
    "args <- parse_args()"
  )
  out <- list(indices = matrix(1L, 1L, 1L), distances = matrix(0, 1L, 1L))
  attr(out, "approximation") <- list(
    strategy = "faiss_IndexHNSWFlat",
    backend = "faiss_hnsw",
    library = "faiss",
    m = 16L,
    ef_search = 150L,
    tuning_policy = "auto_shape_metric",
    tuning_rule = "balanced_shape_metric",
    tuning_high_dim = TRUE,
    tuning_large_k = FALSE,
    pq_tuning_policy = "auto_pq_shape_metric",
    pq_tuning_rule = "pq_high_dim_shape",
    pq_tuning_high_dim = TRUE,
    tuning = list(status = "target_met", results = data.frame(recall = 0.99))
  )
  attr(out, "spatial_index") <- list(strategy = "ignored_grid", bins_per_dim = 32L)
  attr(out, "auto_selection") <- list(
    policy = "static_shape_k_metric_selector",
    explicit_backend = TRUE,
    explicit_method = FALSE,
    backend_decision = "explicit_cpu",
    method_decision = "cpu_auto_shape_selector",
    predicted_backend = "faiss_hnsw",
    predicted_method = "hnsw",
    predicted_device = "cpu",
    reason = "cpu_auto_shape_selector",
    slow_tuning = FALSE
  )
  attr(out, "metric_transform") <- "row_l2_normalize_then_euclidean_graph_search"
  attr(out, "distance_transform") <- "normalized_euclidean_squared_over_2_to_1_minus_similarity"

  params <- env$nn_route_parameters(out)
  expect_match(params, "approximation.strategy=faiss_IndexHNSWFlat", fixed = TRUE)
  expect_match(params, "approximation.m=16", fixed = TRUE)
  expect_match(params, "approximation.ef_search=150", fixed = TRUE)
  expect_match(params, "approximation.tuning_policy=auto_shape_metric", fixed = TRUE)
  expect_match(params, "approximation.tuning_rule=balanced_shape_metric", fixed = TRUE)
  expect_match(params, "approximation.tuning_high_dim=TRUE", fixed = TRUE)
  expect_match(params, "approximation.tuning_large_k=FALSE", fixed = TRUE)
  expect_match(params, "approximation.pq_tuning_policy=auto_pq_shape_metric", fixed = TRUE)
  expect_match(params, "approximation.pq_tuning_rule=pq_high_dim_shape", fixed = TRUE)
  expect_match(params, "approximation.pq_tuning_high_dim=TRUE", fixed = TRUE)
  expect_match(params, "spatial_index.bins_per_dim=32", fixed = TRUE)
  expect_match(params, "auto_selection.explicit_backend=TRUE", fixed = TRUE)
  expect_match(params, "auto_selection.explicit_method=FALSE", fixed = TRUE)
  expect_match(params, "auto_selection.backend_decision=explicit_cpu", fixed = TRUE)
  expect_match(params, "auto_selection.method_decision=cpu_auto_shape_selector", fixed = TRUE)
  expect_match(params, "auto_selection.predicted_method=hnsw", fixed = TRUE)
  expect_match(params, "auto_selection.predicted_device=cpu", fixed = TRUE)
  expect_match(params, "auto_selection.reason=cpu_auto_shape_selector", fixed = TRUE)
  expect_match(params, "nn_metric.metric_transform=row_l2_normalize_then_euclidean_graph_search", fixed = TRUE)
  expect_match(params, "nn_metric.distance_transform=normalized_euclidean_squared_over_2_to_1_minus_similarity", fixed = TRUE)
  expect_equal(env$nn_tuning_status(out), "target_met")

  attr(out, "approximation")$tuning <- NULL
  expect_equal(env$nn_tuning_status(out), "balanced_shape_metric")
})

test_that("NN metric auto comparison guards speed ratios and recall gaps", {
  env <- source_benchmark_helpers(
    test_path("../../benchmark_scripts/benchmark_nn_metrics.R"),
    "args <- parse_args()"
  )
  cycle_summary <- data.frame(
    dataset = c("A", "A", "B", "B"),
    backend = c("cpu", "cpu", "cuda", "cuda"),
    method = c("auto", "hnsw", "auto", "flat"),
    metric = c("euclidean", "euclidean", "cosine", "cosine"),
    k = c(50L, 50L, 15L, 15L),
    result_backend = c("faiss_hnsw", "faiss_hnsw", "faiss_gpu_flat_cosine", "faiss_gpu_flat_cosine"),
    result_requested_backend = c("auto", "cpu", "auto", "cuda"),
    result_requested_method = c("auto", "hnsw", "auto", "flat"),
    result_tuning = c("auto", "auto", "auto", "auto"),
    resolved_backend = c("faiss_hnsw", "faiss_hnsw", "faiss_gpu_flat_cosine", "faiss_gpu_flat_cosine"),
    implementation_backend = c("faiss_hnsw", "faiss_hnsw", "faiss_gpu_flat_cosine", "faiss_gpu_flat_cosine"),
    preflight_route = c("faiss_hnsw", "faiss_hnsw", "faiss_gpu_flat_cosine", "faiss_gpu_flat_cosine"),
    n_threads = c(2L, 2L, 2L, 2L),
    success_cycles = c(1L, 1L, 1L, 1L),
    median_elapsed_sec = c(1, 0, 1, 2),
    median_recall_at_k = c(0.99, 0.99, NA, 0.98),
    min_recall_at_k = c(0.98, 0.98, NA, 0.97),
    median_min_recall_at_k = c(0.98, 0.98, NA, 0.97),
    recall_reference = c("sample", "sample", NA, "sample"),
    median_recall_query_n = c(512, 512, NA, 512)
  )
  recommendations <- cycle_summary[c(2, 4), , drop = FALSE]
  recommendations$recommendation_basis <- c("fastest_at_recall_threshold", "fastest_at_recall_threshold")

  out <- env$compare_auto_to_recommendations(cycle_summary, recommendations)
  expect_true(is.na(out$auto_median_speed_ratio[out$dataset == "A"]))
  expect_true(is.na(out$auto_median_recall_gap[out$dataset == "B"]))
  expect_true(is.finite(out$auto_median_speed_ratio[out$dataset == "B"]))
})

test_that("NN metric auto-vs-fastest guards speed ratios and recall gaps", {
  env <- source_benchmark_helpers(
    test_path("../../benchmark_scripts/benchmark_nn_metrics.R"),
    "args <- parse_args()"
  )
  ok <- data.frame(
    dataset = c("A", "B"),
    backend = c("cpu", "cuda"),
    method = c("auto", "auto"),
    metric = c("euclidean", "cosine"),
    k = c(50L, 15L),
    cycle = c(1L, 1L),
    result_backend = c("faiss_hnsw", "faiss_gpu_flat_cosine"),
    result_requested_backend = c("auto", "auto"),
    result_requested_method = c("auto", "auto"),
    result_tuning = c("auto", "auto"),
    resolved_backend = c("faiss_hnsw", "faiss_gpu_flat_cosine"),
    implementation_backend = c("faiss_hnsw", "faiss_gpu_flat_cosine"),
    elapsed_sec = c(1, 1),
    recall_at_k = c(0.99, NA),
    recall_reference = c("sample", NA),
    recall_query_n = c(512, NA)
  )
  fastest <- data.frame(
    dataset = c("A", "B"),
    backend = c("cpu", "cuda"),
    method = c("hnsw", "flat"),
    metric = c("euclidean", "cosine"),
    k = c(50L, 15L),
    cycle = c(1L, 1L),
    result_backend = c("faiss_hnsw", "faiss_gpu_flat_cosine"),
    result_requested_backend = c("cpu", "cuda"),
    result_requested_method = c("hnsw", "flat"),
    result_tuning = c("auto", "auto"),
    resolved_backend = c("faiss_hnsw", "faiss_gpu_flat_cosine"),
    implementation_backend = c("faiss_hnsw", "faiss_gpu_flat_cosine"),
    elapsed_sec = c(0, 2),
    recall_at_k = c(0.99, 0.98),
    recall_reference = c("sample", "sample"),
    recall_query_n = c(512, 512)
  )

  out <- env$compare_auto_to_fastest(ok, fastest)

  expect_equal(out$fastest_method, c("hnsw", "flat"))
  expect_equal(out$auto_result_requested_backend, c("auto", "auto"))
  expect_equal(out$fastest_result_requested_backend, c("cpu", "cuda"))
  expect_equal(out$auto_result_requested_method, c("auto", "auto"))
  expect_equal(out$fastest_result_requested_method, c("hnsw", "flat"))
  expect_true(is.na(out$auto_speed_ratio[out$dataset == "A"]))
  expect_true(is.na(out$auto_recall_gap[out$dataset == "B"]))
  expect_true(is.finite(out$auto_speed_ratio[out$dataset == "B"]))
})

test_that("NN metric benchmark accounts for data-shaped method skips", {
  env <- source_benchmark_helpers(
    test_path("../../benchmark_scripts/benchmark_nn_metrics.R"),
    "args <- parse_args()"
  )

  default_methods <- env$default_nn_method_values()
  expect_true("sparse" %in% default_methods)
  expect_true("grid" %in% default_methods)

  dense <- matrix(rnorm(20), ncol = 4)
  sparse_skip <- env$nn_data_expected_skip(dense, "sparse")
  expect_type(sparse_skip, "list")
  expect_true(isTRUE(sparse_skip$skip))
  expect_equal(sparse_skip$reason, "unsupported_input_type")
  expect_match(sparse_skip$notes, "sparse Matrix")

  grid_skip <- env$nn_data_expected_skip(dense, "grid")
  expect_type(grid_skip, "list")
  expect_true(isTRUE(grid_skip$skip))
  expect_equal(grid_skip$reason, "unsupported_shape")
  expect_match(grid_skip$notes, "two- or three-column")
  expect_match(grid_skip$notes, "4 columns")
  expect_null(env$nn_data_expected_skip(matrix(rnorm(20), ncol = 2), "grid"))
  expect_null(env$nn_data_expected_skip(matrix(rnorm(30), ncol = 3), "grid"))
  nsg_skip <- env$nn_data_expected_skip(matrix(rnorm(80 * 4), ncol = 4), "nsg")
  expect_type(nsg_skip, "list")
  expect_true(isTRUE(nsg_skip$skip))
  expect_equal(nsg_skip$reason, "insufficient_training_rows")
  expect_match(nsg_skip$notes, "more than 100 training rows")
  expect_match(nsg_skip$notes, "80 rows")
  expect_null(env$nn_data_expected_skip(matrix(rnorm(120 * 4), ncol = 4), "nsg"))
  expect_null(env$nn_data_expected_skip(matrix(rnorm(20), ncol = 4), "flat"))
})

test_that("NN metric benchmark raw rows record expected-skip reason labels", {
  env <- source_benchmark_helpers(
    test_path("../../benchmark_scripts/benchmark_nn_metrics.R"),
    "args <- parse_args()"
  )
  row <- env$result_row(
    dataset = "A",
    n = 100L,
    p = 4L,
    backend = "cpu",
    method = "grid",
    metric = "euclidean",
    k = 15L,
    cycle = 1L,
    n_threads = 2L,
    status = "expected_skip",
    error = "grid requires 2D/3D data",
    expected_skip = TRUE,
    expected_skip_reason = "unsupported_shape",
    capability_notes = "grid requires 2D/3D data"
  )

  expect_true("expected_skip_reason" %in% names(row))
  expect_true(isTRUE(row$expected_skip))
  expect_equal(row$expected_skip_reason, "unsupported_shape")
})

test_that("graph benchmark raw rows record expected-skip reason labels", {
  env <- source_benchmark_helpers(
    test_path("../../benchmark_scripts/benchmark_graph_clustering.R"),
    "args <- parse_args()"
  )
  row <- env$result_row(
    dataset = "A",
    n = 100L,
    p = 4L,
    cycle = 1L,
    k = 15L,
    graph_backend = "cuda",
    graph_method = "nndescent",
    metric = "euclidean",
    cluster_backend = "cuda",
    method = "leiden",
    weight = "distance",
    n_clusters = 10L,
    n_threads = 2L,
    status = "expected_skip",
    error = "CUDA graph clustering requires libcugraph",
    expected_skip = TRUE,
    expected_skip_reason = "missing_cugraph"
  )

  expect_true("expected_skip_reason" %in% names(row))
  expect_true(isTRUE(row$expected_skip))
  expect_equal(row$expected_skip_reason, "missing_cugraph")
})

test_that("legacy Benchmark #1 uses canonical Flat rows for inner product", {
  env <- source_benchmark_helpers(
    test_path("../../benchmark_scripts/benchmark1_nn_speed.R"),
    "if (worker)"
  )

  methods <- env$method_table()
  expect_false(any(methods$method %in% c("faissR_faiss_flat_ip", "faissR_faiss_gpu_flat_ip")))
  expect_true(all(c("faissR_faiss_flat_l2", "faissR_faiss_gpu_flat_l2") %in% methods$method))

  expect_equal(
    env$benchmark_method_aliases(c("flat")),
    "faissR_faiss_flat_l2"
  )
  expect_equal(
    env$benchmark1_method_values("flat", methods$method),
    "faissR_faiss_flat_l2"
  )
  expect_error(
    env$benchmark1_method_values("faissR_faiss_flat_ip", methods$method),
    "invalid Benchmark #1 method"
  )
  expect_error(
    env$benchmark1_method_values("faissR_faiss_gpu_flat_ip", methods$method),
    "invalid Benchmark #1 method"
  )
  expect_error(
    env$benchmark1_method_values("flat,unknown_method", methods$method),
    "invalid Benchmark #1 method"
  )
  expect_error(
    env$benchmark1_method_values("", methods$method),
    "at least one method"
  )

  expect_true(isTRUE(env$method_metric_applicable("faissR_faiss_flat_l2", "inner_product")$ok))
  expect_true(isTRUE(env$method_metric_applicable("faissR_faiss_gpu_flat_l2", "inner_product")$ok))
  expect_true(isTRUE(env$method_metric_applicable("RcppHNSW_hnsw", "inner_product")$ok))
  expect_false(isTRUE(env$method_metric_applicable("RcppHNSW_hnsw", "correlation")$ok))
  expect_true(isTRUE(env$method_is_exact("faissR_faiss_flat_l2", "inner_product")))
  expect_true(isTRUE(env$method_is_exact("faissR_faiss_gpu_flat_l2", "inner_product")))
  expect_true(isTRUE(env$method_is_exact("faissR_cuda_cuvs_bruteforce", "l2")))
  expect_false(isTRUE(env$method_is_exact("faissR_cuda_cuvs_bruteforce", "inner_product")))
  expect_equal(env$rcpphnsw_distance_arg("l2"), "euclidean")
  expect_equal(env$rcpphnsw_distance_arg("cosine"), "cosine")
  expect_equal(env$rcpphnsw_distance_arg("inner_product"), "ip")

  removed_cpu_route <- env$faissr_benchmark_route("faissR_faiss_flat_ip")
  removed_gpu_route <- env$faissr_benchmark_route("faissR_faiss_gpu_flat_ip")
  expect_equal(removed_cpu_route$execution_backend, "faiss_flat_ip")
  expect_true(is.na(removed_cpu_route$public_backend))
  expect_true(is.na(removed_cpu_route$public_method))
  expect_equal(removed_gpu_route$execution_backend, "faiss_gpu_flat_ip")
  expect_true(is.na(removed_gpu_route$public_backend))
  expect_true(is.na(removed_gpu_route$public_method))

  invalid_row <- env$invalid_worker_method_row(
    dataset = "COIL20",
    method = "faissR_faiss_flat_ip",
    k = 50L,
    metric = "inner_product",
    n_threads = 12L
  )
  expect_equal(invalid_row$status, "failed")
  expect_equal(invalid_row$quality_status, "failed")
  expect_match(invalid_row$error, "invalid Benchmark #1 method")
  expect_true(is.na(invalid_row$public_method))
  expect_true(is.na(invalid_row$route_parameters))
  expect_true(is.na(invalid_row$tuning_status))
})

test_that("legacy Benchmark #1 extracts faissR result route and tuning metadata", {
  env <- source_benchmark_helpers(
    test_path("../../benchmark_scripts/benchmark1_nn_speed.R"),
    "if (worker)"
  )
  out <- list(indices = matrix(1L, 1L, 1L), distances = matrix(0, 1L, 1L))
  attr(out, "backend") <- "cpu"
  attr(out, "requested_backend") <- "cpu"
  attr(out, "requested_method") <- "faiss_hnsw"
  attr(out, "resolved_backend") <- "faiss_hnsw"
  attr(out, "tuning") <- "auto"
  attr(out, "approximation") <- list(
    strategy = "faiss_IndexHNSWFlat",
    backend = "faiss_hnsw",
    m = 16L,
    ef_search = 150L,
    tuning_policy = "auto_shape_metric",
    tuning_rule = "balanced_shape_metric",
    pq_tuning_rule = "pq_high_dim_shape"
  )
  attr(out, "auto_selection") <- list(
    explicit_backend = TRUE,
    explicit_method = FALSE,
    backend_decision = "explicit_cpu",
    method_decision = "cpu_auto_shape_selector",
    predicted_method = "faiss_hnsw",
    predicted_device = "cpu",
    reason = "cpu_auto_shape_selector"
  )

  params <- env$benchmark1_route_parameters(out)
  expect_equal(env$benchmark1_implementation_backend(out), "faiss_hnsw")
  expect_match(params, "approximation.strategy=faiss_IndexHNSWFlat", fixed = TRUE)
  expect_match(params, "approximation.tuning_rule=balanced_shape_metric", fixed = TRUE)
  expect_match(params, "approximation.pq_tuning_rule=pq_high_dim_shape", fixed = TRUE)
  expect_match(params, "auto_selection.explicit_backend=TRUE", fixed = TRUE)
  expect_match(params, "auto_selection.backend_decision=explicit_cpu", fixed = TRUE)
  expect_match(params, "auto_selection.predicted_method=faiss_hnsw", fixed = TRUE)
  expect_match(params, "auto_selection.predicted_device=cpu", fixed = TRUE)
  expect_equal(env$benchmark1_tuning_status(out), "balanced_shape_metric")
})

test_that("legacy Benchmark #1 records faissR runtime capability preflight", {
  env <- source_benchmark_helpers(
    test_path("../../benchmark_scripts/benchmark1_nn_speed.R"),
    "if (worker)"
  )
  methods <- env$method_table()
  caps <- env$benchmark1_runtime_capabilities(methods, c("l2", "inner_product"))

  expect_s3_class(caps, "data.frame")
  expect_true(all(c(
    "method", "metric", "execution_backend", "public_backend",
    "public_method", "public_metric", "metric_supported",
    "public_runtime_reason", "runtime_available", "runtime_reason", "runtime_notes"
  ) %in% names(caps)))
  expect_true("faissR_faiss_gpu_flat_l2" %in% caps$method)

  cpu_flat <- caps[
    caps$method == "faissR_faiss_flat_l2" & caps$metric == "l2",
    ,
    drop = FALSE
  ]
  expect_equal(cpu_flat$execution_backend, "faiss_flat_l2")
  expect_equal(cpu_flat$public_backend, "cpu")
  expect_equal(cpu_flat$public_method, "flat")
  expect_equal(cpu_flat$public_metric, "euclidean")
  expect_equal(cpu_flat$runtime_available, faiss_available())
  expect_equal(cpu_flat$runtime_reason, if (faiss_available()) "available" else "missing_faiss")

  cpu_nnd_ip <- caps[
    caps$method == "faissR_cpu_nndescent" & caps$metric == "inner_product",
    ,
    drop = FALSE
  ]
  expect_equal(cpu_nnd_ip$execution_backend, "cpu_nndescent")
  expect_equal(cpu_nnd_ip$public_backend, "cpu")
  expect_equal(cpu_nnd_ip$public_method, "nndescent")
  expect_equal(cpu_nnd_ip$public_metric, "inner_product")
  expect_true(isTRUE(cpu_nnd_ip$metric_supported))
  expect_true(isTRUE(cpu_nnd_ip$public_supported))
  expect_equal(cpu_nnd_ip$public_resolved_backend, "cpu_nndescent")
  expect_true(isTRUE(cpu_nnd_ip$runtime_available))

  gpu_skip <- env$benchmark1_runtime_skip("faissR_faiss_gpu_flat_l2", "l2")
  if (isTRUE(faiss_gpu_available())) {
    expect_null(gpu_skip)
  } else {
    expect_match(gpu_skip, "FAISS GPU|not available|unavailable", ignore.case = TRUE)
  }
})

test_that("legacy Benchmark #1 uses canonical direct cuVS IVF row", {
  env <- source_benchmark_helpers(
    test_path("../../benchmark_scripts/benchmark1_nn_speed.R"),
    "if (worker)"
  )

  methods <- env$method_table()
  expect_false("faissR_cuda_ivf" %in% methods$method)
  expect_true("faissR_cuda_cuvs_ivf_flat" %in% methods$method)

  aliases <- env$benchmark_method_aliases(c("cuda_ivf", "faissR_cuda_ivf", "faissR_cuda_cuvs_ivf_flat"))
  expect_equal(aliases, "faissR_cuda_cuvs_ivf_flat")
})

test_that("legacy Benchmark #1 defaults to all four public metrics", {
  env <- source_benchmark_helpers(
    test_path("../../benchmark_scripts/benchmark1_nn_speed.R"),
    "if (worker)"
  )

  expect_equal(
    env$benchmark1_metric_values(metrics = NULL, env_metrics = NA_character_),
    c("l2", "cosine", "correlation", "inner_product")
  )
  expect_equal(env$benchmark1_metric_value("euclidean"), "l2")
  expect_equal(env$benchmark1_metric_value("pearson"), "correlation")
  expect_equal(env$benchmark1_metric_value("ip"), "inner_product")
  expect_error(
    env$benchmark1_metric_value("manhattan"),
    "Invalid value\\(s\\): manhattan"
  )
  expect_equal(
    env$benchmark1_metric_values("euclidean,pearson,ip", env_metrics = NA_character_),
    c("l2", "correlation", "inner_product")
  )
  expect_equal(
    env$benchmark1_metric_values(metrics = NULL, env_metrics = "cosine,innerproduct"),
    c("cosine", "inner_product")
  )
  expect_error(
    env$benchmark1_metric_values("unknown", env_metrics = NA_character_),
    "Invalid value\\(s\\): unknown"
  )
  expect_error(
    env$benchmark1_metric_values("", env_metrics = ""),
    "at least one metric"
  )
})

test_that("legacy Benchmark #1 validates k-value grids", {
  env <- source_benchmark_helpers(
    test_path("../../benchmark_scripts/benchmark1_nn_speed.R"),
    "if (worker)"
  )

  expect_equal(
    env$benchmark1_k_values(k_values = NULL, env_k_values = NA_character_),
    c(5L, 10L, 15L, 50L, 100L)
  )
  expect_equal(
    env$benchmark1_k_values("5,10,10", env_k_values = NA_character_),
    c(5L, 10L)
  )
  expect_equal(
    env$benchmark1_k_values(k_values = NULL, env_k_values = "15,50"),
    c(15L, 50L)
  )
  expect_error(
    env$benchmark1_k_values("5,bad", env_k_values = NA_character_),
    "Invalid value\\(s\\): bad"
  )
  expect_error(
    env$benchmark1_k_values("0", env_k_values = NA_character_),
    "Invalid value\\(s\\): 0"
  )
  expect_error(
    env$benchmark1_k_values("15.5", env_k_values = NA_character_),
    "Invalid value\\(s\\): 15.5"
  )
  expect_error(
    env$benchmark1_k_values("", env_k_values = ""),
    "at least one positive integer"
  )
})

test_that("legacy Benchmark #1 validates scalar numeric controls", {
  env <- source_benchmark_helpers(
    test_path("../../benchmark_scripts/benchmark1_nn_speed.R"),
    "if (worker)"
  )

  expect_equal(env$benchmark1_positive_int_arg(NULL, "threads", "4"), 4L)
  expect_equal(env$benchmark1_positive_int_arg("12", "threads", "4"), 12L)
  expect_equal(env$benchmark1_positive_int_arg("50", "k", "15"), 50L)
  expect_equal(env$benchmark1_positive_int_arg("600", "timeout", "60"), 600L)
  expect_error(
    env$benchmark1_positive_int_arg("zero", "threads", "4"),
    "positive integer"
  )
  expect_error(
    env$benchmark1_positive_int_arg("bad", "k", "15"),
    "positive integer"
  )
  expect_error(
    env$benchmark1_positive_int_arg("15.5", "k", "15"),
    "positive integer"
  )
  expect_error(
    env$benchmark1_positive_int_arg(0, "timeout", "60"),
    "positive integer"
  )
  expect_equal(
    env$benchmark1_positive_numeric_arg(NULL, "quality_max_ops", "5e9"),
    5e9
  )
  expect_equal(
    env$benchmark1_positive_numeric_arg("1e6", "quality_max_ops", "5e9"),
    1e6
  )
  expect_error(
    env$benchmark1_positive_numeric_arg("many", "quality_max_ops", "5e9"),
    "positive numeric"
  )
  expect_error(
    env$benchmark1_positive_numeric_arg(0, "quality_max_ops", "5e9"),
    "positive numeric"
  )
})

test_that("legacy Benchmark #1 exposes faissR NNDescent metric support", {
  env <- source_benchmark_helpers(
    test_path("../../benchmark_scripts/benchmark1_nn_speed.R"),
    "if (worker)"
  )

  for (method in c("faissR_cpu_nndescent", "faissR_cuda_cuvs_nndescent")) {
    expect_true(isTRUE(env$method_metric_applicable(method, "l2")$ok))
    expect_true(isTRUE(env$method_metric_applicable(method, "cosine")$ok))
    expect_true(isTRUE(env$method_metric_applicable(method, "correlation")$ok))
  }
  expect_true(isTRUE(env$method_metric_applicable("faissR_cpu_nndescent", "inner_product")$ok))
  expect_false(isTRUE(env$method_metric_applicable("faissR_cuda_cuvs_nndescent", "inner_product")$ok))
})

test_that("legacy Benchmark #1 best ranking is quality-aware before speed", {
  env <- source_benchmark_helpers(
    test_path("../../benchmark_scripts/benchmark1_nn_speed.R"),
    "if (worker)"
  )

  success <- data.frame(
    dataset = c("A", "A", "A", "B", "B"),
    metric = c("l2", "l2", "l2", "cosine", "cosine"),
    k = c(15L, 15L, 15L, 50L, 50L),
    method = c(
      "fast_low_recall",
      "slow_high_recall",
      "same_recall_better_rank",
      "fast_missing_quality",
      "slow_missing_quality"
    ),
    time_sec = c(1, 5, 4, 1, 2),
    peak_rss_gb = c(2, 1, 1, 1, 1),
    recall_at_k = c(0.80, 0.95, 0.95, NA, NA),
    rank_correlation = c(0.80, 0.90, 0.92, NA, NA),
    mean_relative_distance_error = c(0.20, 0.10, 0.08, NA, NA),
    stringsAsFactors = FALSE
  )

  ranked <- env$rank_benchmark1_success(success)
  best <- ranked[!duplicated(paste(ranked$dataset, ranked$metric, ranked$k, sep = "\r")), ]

  expect_equal(best$method, c("same_recall_better_rank", "fast_missing_quality"))
})

test_that("legacy Benchmark #1 quality metrics guard invalid finite means", {
  env <- source_benchmark_helpers(
    test_path("../../benchmark_scripts/benchmark1_nn_speed.R"),
    "if (worker)"
  )

  candidate <- list(
    indices = matrix(c(NA_integer_, NA_integer_), nrow = 1),
    distances = matrix(c(Inf, NA_real_), nrow = 1)
  )
  reference <- list(
    indices = matrix(c(1L, 2L), nrow = 1),
    distances = matrix(c(Inf, NA_real_), nrow = 1)
  )

  expect_true(is.na(env$benchmark1_finite_mean(c(Inf, NA_real_, NaN))))
  expect_true(is.na(env$knn_rank_correlation(candidate, reference, k = 2L)))
})

test_that("legacy Benchmark #1 inner-product reference uses distance convention", {
  env <- source_benchmark_helpers(
    test_path("../../benchmark_scripts/benchmark1_nn_speed.R"),
    "if (worker)"
  )
  x <- matrix(c(
    2, 0,
    0, 3,
    1, 1
  ), ncol = 2, byrow = TRUE)

  ref <- env$exact_subset_knn(x, rows = 1L, k = 2L, metric = "inner_product")

  expect_equal(ref$indices[1L, ], c(3L, 2L))
  expect_equal(ref$distances[1L, ], c(0, 2), tolerance = 1e-12)
})

test_that("benchmark KNN recall ignores missing neighbour padding", {
  source_file <- test_path("../../benchmark_scripts/source.R")
  if (!file.exists(source_file)) {
    skip("Benchmark scripts are not available in this installed-package test context.")
  }
  env <- new.env(parent = globalenv())
  source(source_file, local = env)

  approx <- list(indices = matrix(
    c(1L, NA_integer_, 2L, NA_integer_),
    nrow = 2L,
    byrow = TRUE
  ))
  exact <- list(indices = matrix(
    c(1L, NA_integer_, 3L, NA_integer_),
    nrow = 2L,
    byrow = TRUE
  ))

  out <- env$benchmark_knn_recall(approx, exact, k = 2L)

  expect_equal(out$recall_at_k, 0.5)
  expect_equal(out$median_recall_at_k, 0.5)
  expect_equal(out$min_recall_at_k, 0)
  expect_error(
    env$benchmark_knn_recall(approx, exact, k = 1.5),
    "`k` must be a positive integer"
  )
  expect_error(
    env$benchmark_knn_recall(
      matrix(integer(), nrow = 2L, ncol = 0L),
      matrix(integer(), nrow = 2L, ncol = 0L)
    ),
    "at least one neighbour column"
  )
})

test_that("legacy Benchmark #1 quality evaluation handles short KNN outputs", {
  env <- source_benchmark_helpers(
    test_path("../../benchmark_scripts/benchmark1_nn_speed.R"),
    "if (worker)"
  )
  x <- matrix(seq_len(20), nrow = 5)
  obj <- list(
    indices = matrix(c(2L, 1L, 4L, 5L, 3L), ncol = 1),
    distances = matrix(rep(1, 5), ncol = 1)
  )

  out <- env$evaluate_knn_quality(x, obj, k = 3L, metric = "l2", exact = FALSE)

  expect_equal(out$quality_status, "success")
  expect_equal(out$quality_eval_n, nrow(x))
  expect_true(is.finite(out$recall_at_k))
})

test_that("k-means benchmark defaults cover fast_kmeans stats and public backends", {
  env <- source_benchmark_helpers(
    test_path("../../benchmark_scripts/benchmark_kmeans.R"),
    "args <- parse_args()"
  )

  expect_equal(
    env$default_kmeans_method_values(),
    c("fast_kmeans", "stats")
  )
  expect_equal(
    env$default_kmeans_backend_values(),
    c("auto", "cpu", "cuda")
  )
  expect_equal(env$default_kmeans_cycles(), 10L)
  expect_equal(
    env$valid_kmeans_tuning_values(),
    c("auto", "fixed", "off", "none")
  )
})

test_that("k-means benchmark records runtime capability preflight", {
  env <- source_benchmark_helpers(
    test_path("../../benchmark_scripts/benchmark_kmeans.R"),
    "args <- parse_args()"
  )
  caps <- env$kmeans_runtime_capabilities()

  expect_s3_class(caps, "data.frame")
  expect_true(all(c(
    "method", "backend", "runtime_available", "resolved_backend",
    "runtime_reason", "runtime_notes", "cuda_available",
    "faiss_gpu_available", "cuvs_available"
  ) %in% names(caps)))
  expect_equal(caps$backend, c("auto", "cpu", "cuda", "stats"))

  cuda <- env$kmeans_runtime_status("fast_kmeans", "cuda", caps)
  expect_equal(cuda$resolved_backend, "cuda")
  expect_equal(
    cuda$runtime_reason,
    env$kmeans_cuda_runtime_reason(
      cuda_available_value = faissR::cuda_available(),
      faiss_gpu_available_value = env$kmeans_faiss_gpu_available(),
      cuvs_available_value = faissR::cuvs_available()
    )
  )
  expect_equal(
    cuda$runtime_available,
    isTRUE(faissR::cuda_available()) &&
      (isTRUE(faissR::faiss_gpu_available()) || isTRUE(faissR::cuvs_available()))
  )

  skip_reason <- env$kmeans_expected_skip("fast_kmeans", "cuda")
  if (isTRUE(cuda$runtime_available)) {
    expect_null(skip_reason)
  } else {
    expect_match(skip_reason, "CUDA k-means")
  }

  expect_equal(
    env$kmeans_cuda_runtime_reason(
      cuda_available_value = FALSE,
      faiss_gpu_available_value = TRUE,
      cuvs_available_value = TRUE
    ),
    "missing_cuda_runtime"
  )
  expect_match(
    env$kmeans_cuda_runtime_notes("missing_cuda_runtime"),
    "CUDA runtime"
  )
  expect_equal(
    env$kmeans_cuda_runtime_reason(
      cuda_available_value = TRUE,
      faiss_gpu_available_value = FALSE,
      cuvs_available_value = FALSE
    ),
    "missing_gpu_kmeans_backend"
  )
  expect_match(
    env$kmeans_cuda_runtime_notes("missing_gpu_kmeans_backend"),
    "FAISS GPU k-means|cuVS k-means"
  )
})

test_that("k-means benchmark validates method and backend selectors", {
  env <- source_benchmark_helpers(
    test_path("../../benchmark_scripts/benchmark_kmeans.R"),
    "args <- parse_args()"
  )

  expect_equal(
    env$validate_choice_values(
      c("fast_kmeans", "stats", "fast_kmeans"),
      env$default_kmeans_method_values(),
      "methods"
    ),
    c("fast_kmeans", "stats")
  )
  expect_equal(
    env$validate_choice_values(
      c("auto", "cpu", "cuda"),
      env$default_kmeans_backend_values(),
      "backends"
    ),
    c("auto", "cpu", "cuda")
  )
  expect_error(
    env$validate_choice_values(c("fast_kmeans", "kmeanspp"), env$default_kmeans_method_values(), "methods"),
    "Invalid value\\(s\\): kmeanspp"
  )
  expect_error(
    env$validate_choice_values(c("gpu"), env$default_kmeans_backend_values(), "backends"),
    "Invalid value\\(s\\): gpu"
  )
  expect_error(
    env$validate_choice_values(character(), env$default_kmeans_backend_values(), "backends"),
    "at least one value"
  )
  expect_equal(env$validate_kmeans_tuning_value(" auto "), "auto")
  expect_equal(env$validate_kmeans_tuning_value("FIXED"), "fixed")
  expect_equal(env$validate_kmeans_tuning_value("off"), "off")
  expect_equal(env$validate_kmeans_tuning_value("none"), "none")
  expect_error(
    env$validate_kmeans_tuning_value("pilot"),
    "Invalid value: pilot"
  )
  expect_error(
    env$validate_kmeans_tuning_value(""),
    "must be one of"
  )
})

test_that("k-means benchmark validates dataset selectors", {
  env <- source_benchmark_helpers(
    test_path("../../benchmark_scripts/benchmark_kmeans.R"),
    "args <- parse_args()"
  )
  valid <- c("COIL20", "USPS", "SimulatedTiny3Clusters")

  expect_equal(
    env$validate_dataset_values(c("COIL20", "USPS", "COIL20"), valid),
    c("COIL20", "USPS")
  )
  expect_error(
    env$validate_dataset_values(c("COIL20", "bad_dataset"), valid),
    "Invalid value\\(s\\): bad_dataset"
  )
  expect_error(
    env$validate_dataset_values(character(), valid),
    "at least one dataset"
  )
})

test_that("k-means benchmark centers argument is explicit", {
  env <- source_benchmark_helpers(
    test_path("../../benchmark_scripts/benchmark_kmeans.R"),
    "args <- parse_args()"
  )

  expect_equal(env$required_positive_int_arg(10L, "centers"), 10L)
  expect_equal(env$required_positive_int_arg("3", "centers"), 3L)
  expect_error(env$required_positive_int_arg("auto", "centers"), "positive integer")
  expect_error(env$required_positive_int_arg(0L, "centers"), "positive integer")
  expect_equal(env$required_positive_int_arg("12", "threads"), 12L)
  expect_equal(env$required_positive_int_arg("600", "timeout"), 600L)
  expect_equal(env$required_positive_int_arg("10", "cycles"), 10L)
  expect_error(env$required_positive_int_arg("many", "cycles"), "positive integer")
  expect_error(env$required_positive_int_arg("1.5", "cycles"), "positive integer")
  expect_equal(env$required_positive_int_arg("42", "seed"), 42L)
  expect_error(env$required_positive_int_arg("many", "seed"), "positive integer")
  expect_error(env$required_positive_int_arg(0, "seed"), "positive integer")
  expect_equal(env$required_nonnegative_numeric_arg("0.01", "ari_tolerance"), 0.01)
  expect_equal(env$required_nonnegative_numeric_arg(0, "ari_tolerance"), 0)
  expect_error(
    env$required_nonnegative_numeric_arg("auto", "ari_tolerance"),
    "non-negative numeric"
  )
  expect_error(
    env$required_nonnegative_numeric_arg(-0.1, "ari_tolerance"),
    "non-negative numeric"
  )
  expect_equal(env$resolve_kmeans_int("auto", 25L), 25L)
  expect_equal(env$resolve_kmeans_int("25", 100L), 25L)
  expect_error(env$resolve_kmeans_int("25.5", 100L), "positive integers or `auto`")
  expect_error(env$resolve_kmeans_int("many", 100L), "positive integers or `auto`")
  expect_equal(env$resolve_kmeans_tol("auto", 1e-4), 1e-4)
  expect_equal(env$resolve_kmeans_tol("0.001", 1e-4), 0.001)
  expect_error(env$resolve_kmeans_tol("many", 1e-4), "non-negative numeric")
  expect_error(env$resolve_kmeans_tol("-0.1", 1e-4), "non-negative numeric")
  expect_true(env$kmeans_hit_max_iter(25L, 25L))
  expect_false(env$kmeans_hit_max_iter(24L, 25L))
  expect_true(is.na(env$kmeans_hit_max_iter(NA_integer_, 25L)))
})

test_that("k-means benchmark mirrors fast_kmeans auto CUDA shape gate", {
  env <- source_benchmark_helpers(
    test_path("../../benchmark_scripts/benchmark_kmeans.R"),
    "args <- parse_args()"
  )

  expect_false(env$kmeans_auto_prefers_cuda(n = 120L, p = 4L, centers = 3L))
  expect_true(env$kmeans_auto_prefers_cuda(n = 70000L, p = 784L, centers = 10L))
  expect_true(env$kmeans_auto_prefers_cuda(n = 500000L, p = 32L, centers = 10L))
  expect_true(env$kmeans_auto_prefers_cuda(n = NULL, p = NULL, centers = NULL))
  expect_false(env$kmeans_auto_prefers_cuda(n = 1000000L, p = 784L, centers = 1L))

  small_policy <- env$kmeans_auto_backend_policy(n = 120L, p = 4L, centers = 3L)
  expect_false(small_policy$prefer_cuda)
  expect_equal(small_policy$reason, "small_cpu_preferred")
  expect_equal(small_policy$work_threshold, 1e8)
  expect_equal(small_policy$nbytes_threshold, 256 * 1024^2)
  expect_equal(small_policy$large_n_threshold, 50000)
  expect_equal(small_policy$large_p_threshold, 128)

  work_policy <- env$kmeans_auto_backend_policy(n = 70000L, p = 784L, centers = 10L)
  expect_true(work_policy$prefer_cuda)
  expect_equal(work_policy$reason, "work_at_least_1e8")
  expect_true(work_policy$work >= work_policy$work_threshold)

  one_cluster <- env$kmeans_auto_backend_policy(n = 1000000L, p = 784L, centers = 1L)
  expect_false(one_cluster$prefer_cuda)
  expect_equal(one_cluster$reason, "single_cluster_exact_mean")
  expect_equal(one_cluster$n_per_center, 1000000)
})

test_that("k-means benchmark fallback auto params mirror package metadata", {
  env <- source_benchmark_helpers(
    test_path("../../benchmark_scripts/benchmark_kmeans.R"),
    "args <- parse_args()"
  )
  env$getFromNamespace <- function(...) stop("simulate unavailable package helper")

  shapes <- list(
    list(n = 70000L, p = 784L, centers = 1L, tuning = "auto"),
    list(n = 70000L, p = 784L, centers = 10L, tuning = " Auto "),
    list(n = 50000L, p = 10L, centers = 100L, tuning = "auto"),
    list(n = 1000L, p = 10L, centers = 100L, tuning = "auto"),
    list(n = 200000L, p = 50L, centers = 100L, tuning = "FIXED"),
    list(n = 200000L, p = 50L, centers = 100L, tuning = "none")
  )
  for (shape in shapes) {
    fallback <- do.call(env$kmeans_auto_params, shape)
    expected <- do.call(faissR:::kmeans_auto_params, shape)
    expect_equal(fallback[names(expected)], expected, info = paste(shape, collapse = ","))
  }
})

test_that("k-means benchmark records static selection metadata", {
  env <- source_benchmark_helpers(
    test_path("../../benchmark_scripts/benchmark_kmeans.R"),
    "args <- parse_args()"
  )
  row <- env$result_row(
    dataset = "A",
    n = 100L,
    p = 4L,
    method = "fast_kmeans",
    backend = "auto",
    centers = 3L,
    cycle = 1L,
    n_threads = 2L,
    status = "success",
    elapsed_sec = 1,
    backend_used = "faiss",
    requested_backend = "auto",
    resolved_backend = "cpu",
    iter = 5L,
    tot_withinss = 10,
    ari = 0.9,
    max_iter = 100L,
    converged = TRUE,
    hit_max_iter = FALSE,
    n_init = 5L,
    tol = 1e-4,
    tuning_policy = "auto",
    tuning_rule = "small_low_work_multistart",
    tuning_rule_detail = "n=100;p=4;centers=3;n_per_center=33.33;work=1.2e+03",
    tuning_work = 1200,
    tuning_n_per_center = 33.3,
    tuning_few_points_many_centers = FALSE,
    selection_policy = "static_shape_center_backend_selector",
    selection_slow_tuning = FALSE,
    selection_predicted_backend = "cpu",
    selection_reason = "small_cpu_preferred",
    selection_explicit_backend = TRUE,
    selection_backend_decision = "explicit_cpu",
    selection_work = 1200,
    selection_nbytes = 3200,
    selection_n_per_center = 33.3,
    selection_cuda_available = FALSE,
    selection_faiss_gpu_available = FALSE,
    selection_cuvs_available = FALSE
  )
  expect_equal(row$selection_policy, "static_shape_center_backend_selector")
  expect_false(row$selection_slow_tuning)
  expect_equal(row$selection_predicted_backend, "cpu")
  expect_equal(row$selection_reason, "small_cpu_preferred")
  expect_true(row$selection_explicit_backend)
  expect_equal(row$selection_backend_decision, "explicit_cpu")
  expect_true(row$converged)
  expect_false(row$hit_max_iter)

  summary <- env$summarize_kmeans_cycles(row)
  expect_equal(summary$tuning_rule, "small_low_work_multistart")
  expect_equal(summary$tuning_rule_detail, "n=100;p=4;centers=3;n_per_center=33.33;work=1.2e+03")
  expect_false(summary$tuning_few_points_many_centers)
  expect_equal(summary$selection_policy, "static_shape_center_backend_selector")
  expect_false(summary$selection_slow_tuning)
  expect_equal(summary$selection_predicted_backend, "cpu")
  expect_equal(summary$selection_reason, "small_cpu_preferred")
  expect_true(summary$selection_explicit_backend)
  expect_equal(summary$selection_backend_decision, "explicit_cpu")
  expect_equal(summary$median_selection_work, 1200)
  expect_equal(summary$median_selection_nbytes, 3200)
  expect_equal(summary$median_selection_n_per_center, 33.3)
  expect_false(summary$any_hit_max_iter)
  expect_true(summary$all_converged)
})

test_that("k-means benchmark recommendations are grouped by dataset and centers", {
  env <- source_benchmark_helpers(
    test_path("../../benchmark_scripts/benchmark_kmeans.R"),
    "args <- parse_args()"
  )
  cycle_summary <- data.frame(
    dataset = c("A", "A", "A", "A", "B", "B"),
    centers = c(2L, 2L, 3L, 3L, 2L, 2L),
    method = c("m1", "m2", "m1", "m2", "m1", "m2"),
    backend = c("cpu", "cpu", "cpu", "cpu", "cpu", "cpu"),
    metric = c("euclidean", "euclidean", "euclidean", "euclidean", "euclidean", "euclidean"),
    median_ari = c(0.90, 0.89, 0.70, 0.69, NA, NA),
    median_elapsed_sec = c(2, 1, 5, 3, 4, 2)
  )

  out <- env$recommend_kmeans_methods(cycle_summary, ari_tolerance = 0.02)
  expect_equal(nrow(out), 3L)
  expect_equal(out$dataset, c("A", "A", "B"))
  expect_equal(as.integer(out$centers), c(2L, 3L, 2L))
  expect_equal(out$method, c("m2", "m2", "m2"))
  expect_equal(
    out$recommendation_basis,
    c("fastest_within_ari_tolerance", "fastest_within_ari_tolerance", "speed_only_no_ari")
  )
})

test_that("k-means benchmark can recommend within backend groups", {
  env <- source_benchmark_helpers(
    test_path("../../benchmark_scripts/benchmark_kmeans.R"),
    "args <- parse_args()"
  )
  cycle_summary <- data.frame(
    dataset = c("A", "A", "A", "A"),
    centers = c(2L, 2L, 2L, 2L),
    method = c("fast_kmeans", "fast_kmeans", "stats", "stats"),
    backend = c("cpu", "cuda", "stats", "stats"),
    metric = rep("euclidean", 4L),
    median_ari = c(0.91, 0.90, 0.92, 0.89),
    median_elapsed_sec = c(3, 1, 5, 4),
    median_tot_withinss = c(10, 11, 9, 12)
  )

  overall <- env$recommend_kmeans_methods(cycle_summary, ari_tolerance = 0.02)
  by_backend <- env$recommend_kmeans_methods(
    cycle_summary,
    ari_tolerance = 0.02,
    group_cols = c("dataset", "centers", "backend")
  )

  expect_equal(nrow(overall), 1L)
  expect_equal(overall$backend, "cuda")
  expect_equal(nrow(by_backend), 3L)
  expect_equal(by_backend$backend, c("cpu", "cuda", "stats"))
  expect_equal(by_backend$method, c("fast_kmeans", "fast_kmeans", "stats"))
})

test_that("k-means benchmark recommendation ties prefer higher ARI then lower withinss", {
  env <- source_benchmark_helpers(
    test_path("../../benchmark_scripts/benchmark_kmeans.R"),
    "args <- parse_args()"
  )
  cycle_summary <- data.frame(
    dataset = c("A", "A", "B", "B"),
    centers = c(2L, 2L, 2L, 2L),
    method = c("lower_ari", "higher_ari", "higher_withinss", "lower_withinss"),
    backend = c("cpu", "cpu", "cpu", "cpu"),
    metric = c("euclidean", "euclidean", "euclidean", "euclidean"),
    median_ari = c(0.89, 0.90, 0.90, 0.90),
    median_elapsed_sec = c(1, 1, 1, 1),
    median_tot_withinss = c(10, 12, 20, 15)
  )

  out <- env$recommend_kmeans_methods(cycle_summary, ari_tolerance = 0.02)
  expect_equal(out$method, c("higher_ari", "lower_withinss"))
})

test_that("k-means benchmark best-row ranking uses withinss after quality and speed", {
  env <- source_benchmark_helpers(
    test_path("../../benchmark_scripts/benchmark_kmeans.R"),
    "args <- parse_args()"
  )
  ok <- data.frame(
    dataset = c("A", "A", "A", "B", "B"),
    method = c(
      "fast_high_withinss", "fast_low_withinss", "lower_ari",
      "fast_missing_quality", "slow_missing_quality"
    ),
    ari = c(0.95, 0.95, 0.94, NA, NA),
    elapsed_sec = c(1, 1, 0.5, 1, 2),
    tot_withinss = c(20, 10, 1, NA, NA),
    stringsAsFactors = FALSE
  )

  ranked <- env$rank_kmeans_success(ok)
  best <- ranked[!duplicated(ranked$dataset), , drop = FALSE]
  expect_equal(best$method, c("fast_low_withinss", "fast_missing_quality"))
})

test_that("k-means benchmark best rows can preserve centers dimension", {
  env <- source_benchmark_helpers(
    test_path("../../benchmark_scripts/benchmark_kmeans.R"),
    "args <- parse_args()"
  )
  ok <- data.frame(
    dataset = c("A", "A", "A", "B"),
    centers = c(2L, 2L, 3L, 2L),
    method = c("fast_low_ari", "slow_high_ari", "centers3", "b_best"),
    ari = c(0.80, 0.90, 0.70, 0.95),
    elapsed_sec = c(1, 2, 1, 1),
    tot_withinss = c(20, 10, 5, 1),
    stringsAsFactors = FALSE
  )

  compact <- env$select_kmeans_best_rows(ok, group_cols = "dataset")
  by_centers <- env$select_kmeans_best_rows(ok, group_cols = c("dataset", "centers"))

  expect_equal(compact$method, c("slow_high_ari", "b_best"))
  expect_setequal(by_centers$method, c("slow_high_ari", "centers3", "b_best"))
  expect_equal(
    nrow(unique(by_centers[, c("dataset", "centers"), drop = FALSE])),
    3L
  )
})

test_that("k-means fast comparison is ordered by dataset centers and backend", {
  env <- source_benchmark_helpers(
    test_path("../../benchmark_scripts/benchmark_kmeans.R"),
    "args <- parse_args()"
  )
  cycle_summary <- data.frame(
    dataset = c("A", "A", "A", "A", "A", "A"),
    method = c("fast_kmeans", "fast_kmeans", "fast_kmeans", "fast_kmeans", "stats", "stats"),
    backend = c("cuda", "cpu", "cuda", "cpu", "stats", "stats"),
    backend_used = c("cuda_faiss", "faiss", "cuda_faiss", "faiss", "stats", "stats"),
    requested_backend = c("cuda", "cpu", "cuda", "cpu", "stats", "stats"),
    resolved_backend = c("cuda", "cpu", "cuda", "cpu", "stats", "stats"),
    n_threads = c(2L, 2L, 2L, 2L, 2L, 2L),
    centers = c(3L, 3L, 5L, 5L, 3L, 5L),
    success_cycles = c(1L, 1L, 1L, 1L, 1L, 1L),
    median_elapsed_sec = c(3, 4, 5, 6, 7, 8),
    median_ari = c(0.9, 0.91, 0.8, 0.81, 0.89, 0.79),
    min_ari = c(0.9, 0.91, 0.8, 0.81, 0.89, 0.79),
    median_tot_withinss = c(10, 11, 20, 21, 12, 22),
    median_iter = c(5, 5, 5, 5, 5, 5),
    median_max_iter = c(100, 100, 100, 100, 100, 100),
    median_n_init = c(3, 3, 3, 3, 1, 1),
    median_tol = c(1e-4, 1e-4, 1e-4, 1e-4, 1e-4, 1e-4),
    tuning_policy = c("auto", "auto", "auto", "auto", "stats", "stats"),
    tuning_rule = c("medium", "medium", "medium", "medium", "stats_kmeans", "stats_kmeans"),
    tuning_rule_detail = c("fast3", "fast3", "fast5", "fast5", "stats3", "stats5"),
    median_tuning_work = c(300, 300, 500, 500, NA, NA),
    median_tuning_n_per_center = c(10, 10, 10, 10, NA, NA),
    tuning_high_dim = c(FALSE, FALSE, FALSE, FALSE, NA, NA),
    tuning_large_n = c(FALSE, FALSE, FALSE, FALSE, NA, NA),
    tuning_many_centers = c(FALSE, FALSE, FALSE, FALSE, NA, NA),
    tuning_small_many_centers = c(FALSE, FALSE, FALSE, FALSE, NA, NA),
    tuning_few_points_many_centers = c(FALSE, TRUE, FALSE, FALSE, NA, NA),
    selection_policy = c(
      rep("static_shape_center_backend_selector", 4),
      rep("stats", 2)
    ),
    selection_slow_tuning = c(FALSE, FALSE, FALSE, FALSE, FALSE, FALSE),
    selection_predicted_backend = c("cuda", "cpu", "cuda", "cpu", "stats", "stats"),
    selection_reason = c("work_at_least_1e8", "small_cpu_preferred", "work_at_least_1e8", "small_cpu_preferred", "stats_kmeans", "stats_kmeans"),
    selection_explicit_backend = c(TRUE, TRUE, TRUE, TRUE, TRUE, TRUE),
    selection_backend_decision = c("explicit_cuda", "explicit_cpu", "explicit_cuda", "explicit_cpu", "stats_kmeans", "stats_kmeans"),
    median_selection_work = c(300, 300, 500, 500, NA, NA),
    median_selection_nbytes = c(2400, 2400, 4000, 4000, NA, NA),
    median_selection_n_per_center = c(10, 10, 10, 10, NA, NA),
    selection_cuda_available = c(TRUE, TRUE, TRUE, TRUE, NA, NA),
    selection_faiss_gpu_available = c(TRUE, TRUE, TRUE, TRUE, NA, NA),
    selection_cuvs_available = c(FALSE, FALSE, FALSE, FALSE, NA, NA)
  )
  recommendations <- cycle_summary[cycle_summary$method == "stats", , drop = FALSE]
  recommendations$recommendation_basis <- "fastest_within_ari_tolerance"

  out <- env$compare_fast_kmeans_to_recommendations(cycle_summary, recommendations)
  expect_equal(anyDuplicated(names(out)), 0L)
  expect_equal(as.integer(out$centers), c(3L, 3L, 5L, 5L))
  expect_equal(out$fast_backend, c("cpu", "cuda", "cpu", "cuda"))
  expect_equal(out$fast_requested_backend, c("cpu", "cuda", "cpu", "cuda"))
  expect_equal(out$recommended_requested_backend, c("stats", "stats", "stats", "stats"))
  expect_equal(out$fast_n_threads, c(2L, 2L, 2L, 2L))
  expect_equal(out$recommended_n_threads, c(2L, 2L, 2L, 2L))
  expect_equal(out$fast_tuning_rule, rep("medium", 4))
  expect_equal(out$recommended_tuning_rule, rep("stats_kmeans", 4))
  expect_equal(out$fast_tuning_rule_detail, c("fast3", "fast3", "fast5", "fast5"))
  expect_equal(out$recommended_tuning_rule_detail, c("stats3", "stats3", "stats5", "stats5"))
  expect_equal(out$fast_tuning_few_points_many_centers, c(TRUE, FALSE, FALSE, FALSE))
  expect_true(all(is.na(out$recommended_tuning_few_points_many_centers)))
  expect_equal(out$fast_selection_policy, rep("static_shape_center_backend_selector", 4))
  expect_equal(out$recommended_selection_policy, rep("stats", 4))
  expect_equal(out$fast_selection_predicted_backend, c("cpu", "cuda", "cpu", "cuda"))
  expect_equal(out$fast_selection_backend_decision, c("explicit_cpu", "explicit_cuda", "explicit_cpu", "explicit_cuda"))
  expect_true(all(out$fast_selection_explicit_backend))
  expect_equal(unique(out$recommended_recommendation_basis), "fastest_within_ari_tolerance")
})

test_that("k-means auto comparison audits global recommendation", {
  env <- source_benchmark_helpers(
    test_path("../../benchmark_scripts/benchmark_kmeans.R"),
    "args <- parse_args()"
  )
  cycle_summary <- data.frame(
    dataset = c("A", "A", "A"),
    method = c("fast_kmeans", "fast_kmeans", "stats"),
    backend = c("auto", "cuda", "stats"),
    backend_used = c("faiss", "cuda_faiss", "stats"),
    requested_backend = c("auto", "cuda", "stats"),
    resolved_backend = c("cpu", "cuda", "stats"),
    n_threads = c(2L, 2L, 2L),
    centers = c(3L, 3L, 3L),
    success_cycles = c(2L, 2L, 2L),
    median_elapsed_sec = c(4, 2, 6),
    median_ari = c(0.90, 0.90, 0.89),
    min_ari = c(0.89, 0.89, 0.88),
    median_tot_withinss = c(10, 10, 11),
    median_iter = c(5, 5, 5),
    median_max_iter = c(100, 100, 100),
    any_hit_max_iter = c(FALSE, FALSE, FALSE),
    all_converged = c(TRUE, TRUE, TRUE),
    median_n_init = c(3, 3, 1),
    median_tol = c(1e-4, 1e-4, 1e-4),
    tuning_policy = c("auto", "auto", "stats"),
    tuning_rule = c("medium", "medium", "stats_kmeans"),
    tuning_rule_detail = c("auto_cpu", "cuda_fast", "stats"),
    median_tuning_work = c(300, 300, NA),
    median_tuning_n_per_center = c(10, 10, NA),
    tuning_high_dim = c(FALSE, FALSE, NA),
    tuning_large_n = c(FALSE, FALSE, NA),
    tuning_many_centers = c(FALSE, FALSE, NA),
    tuning_small_many_centers = c(FALSE, FALSE, NA),
    tuning_few_points_many_centers = c(FALSE, FALSE, NA),
    selection_policy = c("static_shape_center_backend_selector", "explicit_backend", "stats"),
    selection_slow_tuning = c(FALSE, FALSE, FALSE),
    selection_predicted_backend = c("cpu", NA_character_, "stats"),
    selection_reason = c("small_cpu_preferred", "explicit_cuda", "stats_kmeans"),
    selection_explicit_backend = c(FALSE, TRUE, TRUE),
    selection_backend_decision = c("small_cpu_preferred", "explicit_cuda", "stats_kmeans"),
    median_selection_work = c(300, 300, NA),
    median_selection_nbytes = c(2400, 2400, NA),
    median_selection_n_per_center = c(10, 10, NA),
    selection_cuda_available = c(TRUE, TRUE, NA),
    selection_faiss_gpu_available = c(TRUE, TRUE, NA),
    selection_cuvs_available = c(FALSE, FALSE, NA)
  )
  recommendations <- cycle_summary[2, , drop = FALSE]
  recommendations$recommendation_basis <- "fastest_within_ari_tolerance"

  out <- env$compare_auto_kmeans_to_recommendations(cycle_summary, recommendations)
  expect_equal(anyDuplicated(names(out)), 0L)
  expect_equal(nrow(out), 1L)
  expect_equal(out$recommended_backend, "cuda")
  expect_false(out$auto_uses_recommended_requested_backend)
  expect_false(out$auto_uses_recommended_resolved_backend)
  expect_false(out$auto_uses_recommended_implementation)
  expect_equal(out$auto_median_speed_ratio, 2)
  expect_equal(out$auto_median_ari_gap, 0)
  expect_equal(out$auto_selection_predicted_backend, "cpu")
  expect_false(out$auto_selection_explicit_backend)
  expect_equal(out$recommended_selection_backend_decision, "explicit_cuda")
  expect_equal(out$recommended_selection_reason, "explicit_cuda")
})

test_that("k-means fast comparison guards withinss ratios", {
  env <- source_benchmark_helpers(
    test_path("../../benchmark_scripts/benchmark_kmeans.R"),
    "args <- parse_args()"
  )
  cycle_summary <- data.frame(
    dataset = c("A", "A", "B", "B"),
    method = c("fast_kmeans", "stats", "fast_kmeans", "stats"),
    backend = c("cpu", "stats", "cpu", "stats"),
    backend_used = c("faiss", "stats", "faiss", "stats"),
    requested_backend = c("cpu", "stats", "cpu", "stats"),
    resolved_backend = c("cpu", "stats", "cpu", "stats"),
    n_threads = c(2L, 2L, 2L, 2L),
    centers = c(3L, 3L, 3L, 3L),
    success_cycles = c(1L, 1L, 1L, 1L),
    median_elapsed_sec = c(1, 1, 1, 1),
    median_ari = c(0.9, 0.9, 0.9, 0.9),
    min_ari = c(0.9, 0.9, 0.9, 0.9),
    median_tot_withinss = c(10, 0, 10, NA),
    median_iter = c(5, 5, 5, 5),
    median_max_iter = c(100, 100, 100, 100),
    median_n_init = c(1, 1, 1, 1),
    median_tol = c(1e-4, 1e-4, 1e-4, 1e-4),
    tuning_policy = c("auto", "stats", "auto", "stats")
  )
  recommendations <- cycle_summary[cycle_summary$method == "stats", , drop = FALSE]
  recommendations$recommendation_basis <- "fastest_within_ari_tolerance"

  out <- env$compare_fast_kmeans_to_recommendations(cycle_summary, recommendations)
  expect_true(all(is.na(out$fast_withinss_ratio)))
  expect_true(all(is.finite(out$fast_median_speed_ratio)))
})

test_that("k-means fast comparison guards speed ratios and ARI gaps", {
  env <- source_benchmark_helpers(
    test_path("../../benchmark_scripts/benchmark_kmeans.R"),
    "args <- parse_args()"
  )
  cycle_summary <- data.frame(
    dataset = c("A", "A", "B", "B"),
    method = c("fast_kmeans", "stats", "fast_kmeans", "stats"),
    backend = c("cpu", "stats", "cpu", "stats"),
    backend_used = c("faiss", "stats", "faiss", "stats"),
    requested_backend = c("cpu", "stats", "cpu", "stats"),
    resolved_backend = c("cpu", "stats", "cpu", "stats"),
    n_threads = c(2L, 2L, 2L, 2L),
    centers = c(3L, 3L, 3L, 3L),
    success_cycles = c(1L, 1L, 1L, 1L),
    median_elapsed_sec = c(1, 0, 1, 1),
    median_ari = c(0.9, 0.9, NA, 0.9),
    min_ari = c(0.9, 0.9, NA, 0.9),
    median_tot_withinss = c(10, 10, 10, 10),
    median_iter = c(5, 5, 5, 5),
    median_max_iter = c(100, 100, 100, 100),
    median_n_init = c(1, 1, 1, 1),
    median_tol = c(1e-4, 1e-4, 1e-4, 1e-4),
    tuning_policy = c("auto", "stats", "auto", "stats")
  )
  recommendations <- cycle_summary[cycle_summary$method == "stats", , drop = FALSE]
  recommendations$recommendation_basis <- "fastest_within_ari_tolerance"

  out <- env$compare_fast_kmeans_to_recommendations(cycle_summary, recommendations)
  expect_true(is.na(out$fast_median_speed_ratio[out$dataset == "A"]))
  expect_true(is.na(out$fast_median_ari_gap[out$dataset == "B"]))
  expect_true(is.finite(out$fast_median_speed_ratio[out$dataset == "B"]))
})

test_that("k-means fast-vs-stats comparison guards derived metrics", {
  env <- source_benchmark_helpers(
    test_path("../../benchmark_scripts/benchmark_kmeans.R"),
    "args <- parse_args()"
  )
  ok <- data.frame(
    dataset = c("A", "A", "B", "B"),
    method = c("fast_kmeans", "stats", "fast_kmeans", "stats"),
    backend = c("cpu", "stats", "cuda", "stats"),
    backend_used = c("faiss", "stats", "cuda", "stats"),
    resolved_backend = c("cpu", "stats", "cuda", "stats"),
    centers = c(3L, 3L, 4L, 4L),
    cycle = c(1L, 1L, 1L, 1L),
    elapsed_sec = c(0, 1, 2, 4),
    tot_withinss = c(10, 0, 10, 20),
    ari = c(0.9, 0.9, NA, 0.8),
    iter = c(5L, 5L, 5L, 5L),
    status = "success"
  )

  out <- env$compare_fast_kmeans_to_stats(ok)

  expect_true(is.na(out$speedup_vs_stats[out$dataset == "A"]))
  expect_true(is.na(out$withinss_ratio_vs_stats[out$dataset == "A"]))
  expect_true(is.na(out$ari_delta_vs_stats[out$dataset == "B"]))
  expect_equal(out$speedup_vs_stats[out$dataset == "B"], 2)
  expect_equal(out$withinss_ratio_vs_stats[out$dataset == "B"], 0.5)
})

test_that("graph benchmark defaults cover requested methods backends and k grid", {
  env <- source_benchmark_helpers(
    test_path("../../benchmark_scripts/benchmark_graph_clustering.R"),
    "args <- parse_args()"
  )

  expect_equal(
    env$default_graph_cluster_methods(),
    c("random_walking", "louvain", "leiden")
  )
  expect_equal(
    env$default_graph_backends(),
    c("auto", "cpu", "cuda")
  )
  expect_equal(
    env$default_graph_k_values(),
    c(5L, 10L, 15L, 50L, 100L)
  )
  expect_equal(env$default_graph_cycles(), 10L)
  expect_equal(
    env$as_int_vec_arg(c("unknown"), env$default_graph_k_values()),
    env$default_graph_k_values()
  )
  expect_equal(
    env$required_positive_int_values(c("15", "50", "50"), "k_values"),
    c(15L, 50L)
  )
  expect_error(
    env$required_positive_int_values(c("15", "many"), "k_values"),
    "Invalid value\\(s\\): many"
  )
  expect_error(
    env$required_positive_int_values(c("-1"), "k_values"),
    "Invalid value\\(s\\): -1"
  )
  expect_error(
    env$required_positive_int_values(c("15.5"), "k_values"),
    "Invalid value\\(s\\): 15.5"
  )
  expect_error(
    env$required_positive_int_values(character(), "k_values"),
    "at least one positive integer"
  )
})

test_that("graph benchmark validates method and backend selectors", {
  env <- source_benchmark_helpers(
    test_path("../../benchmark_scripts/benchmark_graph_clustering.R"),
    "args <- parse_args()"
  )

  expect_equal(
    env$validate_choice_values(
      c("random_walking", "louvain", "leiden", "louvain"),
      env$default_graph_cluster_methods(),
      "methods"
    ),
    c("random_walking", "louvain", "leiden")
  )
  expect_equal(
    env$validate_choice_values(
      c("auto", "cpu", "cuda"),
      env$default_graph_backends(),
      "graph_backends"
    ),
    c("auto", "cpu", "cuda")
  )
  expect_error(
    env$validate_choice_values(c("walktrap"), env$default_graph_cluster_methods(), "methods"),
    "Invalid value\\(s\\): walktrap"
  )
  expect_error(
    env$validate_choice_values(c("gpu"), env$default_graph_backends(), "cluster_backends"),
    "Invalid value\\(s\\): gpu"
  )
  expect_error(
    env$validate_choice_values(character(), env$default_graph_backends(), "graph_backends"),
    "at least one value"
  )
})

test_that("graph benchmark validates dataset selectors", {
  env <- source_benchmark_helpers(
    test_path("../../benchmark_scripts/benchmark_graph_clustering.R"),
    "args <- parse_args()"
  )
  valid <- c("COIL20", "USPS", "SimulatedUniform2D", "SimulatedUniform3D", "SimulatedTiny3Clusters")

  expect_equal(
    env$validate_dataset_values(c("COIL20", "USPS", "SimulatedTiny3Clusters", "COIL20"), valid),
    c("COIL20", "USPS", "SimulatedTiny3Clusters")
  )
  expect_error(
    env$validate_dataset_values(c("COIL20", "bad_dataset"), valid),
    "Invalid value\\(s\\): bad_dataset"
  )
  expect_error(
    env$validate_dataset_values(character(), valid),
    "at least one dataset"
  )
})

test_that("graph benchmark target cluster mode is explicit", {
  env <- source_benchmark_helpers(
    test_path("../../benchmark_scripts/benchmark_graph_clustering.R"),
    "args <- parse_args()"
  )

  expect_equal(env$normalize_target_clusters_mode("labels"), "labels")
  expect_equal(env$normalize_target_clusters_mode("dataset_labels"), "labels")
  expect_equal(env$normalize_target_clusters_mode("none"), "none")
  expect_equal(env$normalize_target_clusters_mode("off"), "none")
  expect_null(env$label_target_clusters(rep(letters[1:3], each = 2L), "none"))
  expect_equal(env$label_target_clusters(rep(letters[1:3], each = 2L), "labels"), 3L)
  expect_error(env$normalize_target_clusters_mode("labelled"), "target_clusters")
  expect_equal(env$required_positive_int_arg("12", "threads"), 12L)
  expect_equal(env$required_positive_int_arg("600", "timeout"), 600L)
  expect_equal(env$required_positive_int_arg("10", "cycles"), 10L)
  expect_error(env$required_positive_int_arg("many", "cycles"), "positive integer")
  expect_error(env$required_positive_int_arg("1.5", "cycles"), "positive integer")
  expect_equal(env$required_positive_int_arg("42", "seed"), 42L)
  expect_error(env$required_positive_int_arg("many", "seed"), "positive integer")
  expect_error(env$required_positive_int_arg(0, "seed"), "positive integer")
  expect_equal(env$required_nonnegative_numeric_arg("0.01", "ari_tolerance"), 0.01)
  expect_equal(env$required_nonnegative_numeric_arg(0, "ari_tolerance"), 0)
  expect_error(
    env$required_nonnegative_numeric_arg("auto", "ari_tolerance"),
    "non-negative numeric"
  )
  expect_error(
    env$required_nonnegative_numeric_arg(-0.1, "ari_tolerance"),
    "non-negative numeric"
  )
  expect_equal(env$default_graph_nn_methods(), "auto")
  expect_true(all(c("auto", "hnsw", "ivf", "grid", "cagra") %in% env$valid_graph_nn_methods()))
  expect_equal(
    env$default_graph_metrics(),
    c("euclidean", "cosine", "correlation", "inner_product")
  )
  expect_equal(
    env$validate_choice_values(c("euclidean", "cosine"), env$valid_graph_metrics(), "metrics"),
    c("euclidean", "cosine")
  )
  expect_equal(
    env$canonical_metric_key(c("l2", "pearson", "ip", "dot-product")),
    c("euclidean", "correlation", "inner_product", "inner_product")
  )
  expect_equal(
    env$validate_metric_values(c("l2", "pearson", "ip", "dot-product")),
    c("euclidean", "correlation", "inner_product")
  )
  expect_error(
    env$validate_metric_values("manhattan"),
    "Invalid value\\(s\\): manhattan"
  )
  expect_error(
    env$validate_metric_values(character()),
    "at least one metric"
  )
  expect_error(
    env$validate_choice_values("manhattan", env$valid_graph_metrics(), "metrics"),
    "Invalid value\\(s\\): manhattan"
  )
  expect_error(
    env$validate_choice_values("faiss_hnsw", env$valid_graph_nn_methods(), "graph_methods"),
    "Invalid value\\(s\\): faiss_hnsw"
  )
})

test_that("graph benchmark preflights graph method and metric skips", {
  env <- source_benchmark_helpers(
    test_path("../../benchmark_scripts/benchmark_graph_clustering.R"),
    "args <- parse_args()"
  )
  dense <- matrix(rnorm(40), ncol = 4)

  grid_skip <- env$graph_build_expected_skip("cpu", graph_method = "grid", metric = "euclidean", x = dense)
  expect_equal(grid_skip$reason, "unsupported_shape")
  expect_match(grid_skip$notes, "two- or three-column")
  expect_match(grid_skip$notes, "4 columns")
  expect_null(env$graph_build_expected_skip("cpu", graph_method = "grid", metric = "euclidean", x = dense[, 1:2]))

  sparse_skip <- env$graph_build_expected_skip("cpu", graph_method = "sparse", metric = "euclidean", x = dense)
  expect_equal(sparse_skip$reason, "unsupported_input_type")
  expect_match(sparse_skip$notes, "sparse Matrix")

  expect_null(env$graph_build_expected_skip("cpu", graph_method = "nndescent", metric = "inner_product", x = dense))
  nnd_cuda_ip <- env$graph_build_expected_skip("cuda", graph_method = "nndescent", metric = "inner_product", x = dense)
  expect_true(nzchar(nnd_cuda_ip$reason))
  expect_match(nnd_cuda_ip$notes, "inner-product|inner_product")

  nsg_skip <- env$graph_build_expected_skip("cpu", graph_method = "nsg", metric = "euclidean", x = matrix(rnorm(80 * 4), ncol = 4))
  expect_equal(nsg_skip$reason, "insufficient_training_rows")
  expect_match(nsg_skip$notes, "more than 100 training rows")
  expect_match(nsg_skip$notes, "80 rows")
  expect_null(env$graph_build_expected_skip("cpu", graph_method = "nsg", metric = "euclidean", x = matrix(rnorm(120 * 4), ncol = 4)))
})

test_that("graph benchmark preflights resolved route runtime dependencies", {
  env <- source_benchmark_helpers(
    test_path("../../benchmark_scripts/benchmark_graph_clustering.R"),
    "args <- parse_args()"
  )

  faiss_gpu_skip <- env$graph_route_runtime_skip(
    "faiss_gpu_flat_l2",
    cuda_available_value = FALSE,
    faiss_gpu_available_value = TRUE,
    cuvs_available_value = TRUE
  )
  expect_equal(faiss_gpu_skip$reason, "missing_faiss_gpu")
  expect_match(faiss_gpu_skip$notes, "FAISS GPU")
  expect_match(faiss_gpu_skip$notes, "CUDA device")

  expect_null(env$graph_route_runtime_skip(
    "faiss_gpu_flat_l2",
    cuda_available_value = TRUE,
    faiss_gpu_available_value = TRUE,
    cuvs_available_value = FALSE
  ))

  cuvs_skip <- env$graph_route_runtime_skip(
    "cuda_cuvs_nndescent",
    cuda_available_value = TRUE,
    faiss_gpu_available_value = FALSE,
    cuvs_available_value = FALSE
  )
  expect_equal(cuvs_skip$reason, "missing_cuvs")
  expect_match(cuvs_skip$notes, "cuVS")

  cuda_skip <- env$graph_route_runtime_skip(
    "cuda_grid",
    cuda_available_value = FALSE,
    faiss_gpu_available_value = TRUE,
    cuvs_available_value = TRUE
  )
  expect_equal(cuda_skip$reason, "missing_cuda")
  expect_match(cuda_skip$notes, "CUDA")
})

test_that("graph benchmark preflights runtime-unavailable graph routes", {
  env <- source_benchmark_helpers(
    test_path("../../benchmark_scripts/benchmark_graph_clustering.R"),
    "args <- parse_args()"
  )
  dense <- matrix(rnorm(120 * 4), ncol = 4)
  caps <- env$graph_nn_capabilities()

  expect_true(all(c(
    "resolved_backend", "runtime_available", "runtime_reason", "runtime_notes"
  ) %in% names(caps)))

  cuda_skip <- env$graph_build_expected_skip(
    "cuda",
    graph_method = "flat",
    metric = "euclidean",
    x = dense
  )
  if (isTRUE(faiss_gpu_available()) && isTRUE(cuda_available())) {
    expect_null(cuda_skip)
  } else {
    expect_true(nzchar(cuda_skip$reason))
    expect_match(cuda_skip$notes, "faiss_gpu_flat_l2|FAISS GPU|unavailable", ignore.case = TRUE)
  }
})

test_that("graph benchmark recommendations are grouped by backend and target cluster count", {
  env <- source_benchmark_helpers(
    test_path("../../benchmark_scripts/benchmark_graph_clustering.R"),
    "args <- parse_args()"
  )
  cycle_summary <- data.frame(
    dataset = c("A", "A", "A", "A", "A", "A", "B", "B"),
    k = c(15L, 15L, 15L, 15L, 15L, 15L, 15L, 15L),
    graph_backend = c("cpu", "cpu", "cpu", "cpu", "cuda", "cuda", "cpu", "cpu"),
    graph_resolved_backend = c("cpu", "cpu", "cpu", "cpu", "cuda", "cuda", "cpu", "cpu"),
    cluster_backend = c("cpu", "cpu", "cpu", "cpu", "cuda", "cuda", "cpu", "cpu"),
    cluster_resolved_backend = c("cpu", "cpu", "cpu", "cpu", "cuda", "cuda", "cpu", "cpu"),
    method = c("louvain", "leiden", "louvain", "leiden", "louvain", "leiden", "louvain", "leiden"),
    weight = c("snn", "snn", "snn", "snn", "snn", "snn", "snn", "snn"),
    success_cycles = c(1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L),
    median_graph_sec = c(1, 1, 1, 1, 1, 1, 1, 1),
    median_cluster_sec = c(4, 2, 3, 1, 8, 6, 4, 2),
    median_total_sec = c(5, 3, 4, 2, 9, 7, 5, 3),
    median_ari = c(0.91, 0.90, 0.72, 0.71, 0.94, 0.93, NA, NA),
    min_ari = c(0.91, 0.90, 0.72, 0.71, 0.94, 0.93, NA, NA),
    median_modularity = c(0.4, 0.39, 0.3, 0.29, 0.5, 0.49, 0.2, 0.19),
    median_n_communities = c(3, 3, 5, 5, 3, 3, 3, 3),
    median_selected_resolution = c(1, 1, 2, 2, 1, 1, 1, 1),
    n_clusters_requested = c(3L, 3L, 5L, 5L, 3L, 3L, 3L, 3L),
    n_clusters_source = c("labels", "labels", "labels", "labels", "labels", "labels", "labels", "labels"),
    graph_cached = c(TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE)
  )

  out <- env$recommend_graph_cluster_methods(cycle_summary, ari_tolerance = 0.02)
  expect_equal(nrow(out), 4L)
  expect_equal(out$dataset, c("A", "A", "A", "B"))
  expect_equal(out$graph_backend, c("cpu", "cpu", "cuda", "cpu"))
  expect_equal(out$cluster_backend, c("cpu", "cpu", "cuda", "cpu"))
  expect_equal(as.integer(out$n_clusters_requested), c(3L, 5L, 3L, 3L))
  expect_equal(out$method, c("leiden", "leiden", "leiden", "leiden"))
  expect_equal(
    out$recommendation_basis,
    c(
      "fastest_within_ari_tolerance", "fastest_within_ari_tolerance",
      "fastest_within_ari_tolerance", "speed_only_no_ari"
    )
  )
})

test_that("graph benchmark recommendations stay separated by graph method and metric", {
  env <- source_benchmark_helpers(
    test_path("../../benchmark_scripts/benchmark_graph_clustering.R"),
    "args <- parse_args()"
  )
  cycle_summary <- data.frame(
    dataset = rep("A", 8L),
    k = rep(15L, 8L),
    graph_backend = rep("cpu", 8L),
    graph_method = rep(c("auto", "hnsw", "auto", "hnsw"), each = 2L),
    metric = rep(c("euclidean", "euclidean", "cosine", "cosine"), each = 2L),
    graph_resolved_backend = rep("cpu", 8L),
    graph_preflight_route = rep("cpu", 8L),
    graph_route_parameters = rep(NA_character_, 8L),
    cluster_backend = rep("cpu", 8L),
    cluster_resolved_backend = rep("cpu", 8L),
    cluster_preflight_route = rep("cpu", 8L),
    method = rep(c("louvain", "leiden"), 4L),
    weight = rep("snn", 8L),
    n = rep(100L, 8L),
    p = rep(4L, 8L),
    n_threads = rep(2L, 8L),
    success_cycles = rep(2L, 8L),
    median_graph_sec = rep(1, 8L),
    median_cluster_sec = rep(c(4, 1), 4L),
    median_total_sec = rep(c(5, 2), 4L),
    median_ari = c(0.90, 0.90, 0.91, 0.91, 0.80, 0.80, 0.81, 0.81),
    min_ari = c(0.90, 0.90, 0.91, 0.91, 0.80, 0.80, 0.81, 0.81),
    median_modularity = rep(0.4, 8L),
    median_n_communities = rep(3, 8L),
    median_selected_resolution = rep(1, 8L),
    n_clusters_requested = rep(3L, 8L),
    n_clusters_source = rep("labels", 8L),
    graph_cached = rep(TRUE, 8L)
  )

  local <- env$recommend_graph_cluster_methods(cycle_summary, ari_tolerance = 0.01)
  global <- env$recommend_graph_cluster_global_methods(cycle_summary, ari_tolerance = 0.01)

  expect_equal(nrow(local), 4L)
  expect_equal(nrow(global), 4L)
  expect_setequal(
    paste(local$graph_method, local$metric, sep = "/"),
    c("auto/euclidean", "hnsw/euclidean", "auto/cosine", "hnsw/cosine")
  )
  expect_setequal(
    paste(global$graph_method, global$metric, sep = "/"),
    c("auto/euclidean", "hnsw/euclidean", "auto/cosine", "hnsw/cosine")
  )
  expect_equal(local$method, rep("leiden", 4L))
  expect_equal(global$method, rep("leiden", 4L))
})

test_that("graph benchmark recommendation ties prefer higher ARI then modularity", {
  env <- source_benchmark_helpers(
    test_path("../../benchmark_scripts/benchmark_graph_clustering.R"),
    "args <- parse_args()"
  )
  cycle_summary <- data.frame(
    dataset = c("A", "A", "B", "B"),
    k = c(15L, 15L, 15L, 15L),
    graph_backend = c("cpu", "cpu", "cpu", "cpu"),
    graph_resolved_backend = c("cpu", "cpu", "cpu", "cpu"),
    cluster_backend = c("cpu", "cpu", "cpu", "cpu"),
    cluster_resolved_backend = c("cpu", "cpu", "cpu", "cpu"),
    method = c("lower_ari", "higher_ari", "lower_modularity", "higher_modularity"),
    weight = c("snn", "snn", "snn", "snn"),
    success_cycles = c(1L, 1L, 1L, 1L),
    median_graph_sec = c(0.5, 0.5, 0.5, 0.5),
    median_cluster_sec = c(0.5, 0.5, 0.5, 0.5),
    median_total_sec = c(1, 1, 1, 1),
    median_ari = c(0.89, 0.90, 0.90, 0.90),
    min_ari = c(0.89, 0.90, 0.90, 0.90),
    median_modularity = c(0.5, 0.4, 0.3, 0.4),
    median_n_communities = c(3, 3, 3, 3),
    median_selected_resolution = c(1, 1, 1, 1),
    n_clusters_requested = c(3L, 3L, 3L, 3L),
    n_clusters_source = c("labels", "labels", "labels", "labels"),
    graph_cached = c(TRUE, TRUE, TRUE, TRUE)
  )

  out <- env$recommend_graph_cluster_methods(cycle_summary, ari_tolerance = 0.02)
  expect_equal(out$method, c("higher_ari", "higher_modularity"))
})

test_that("graph benchmark best-row ranking uses modularity before speed", {
  env <- source_benchmark_helpers(
    test_path("../../benchmark_scripts/benchmark_graph_clustering.R"),
    "args <- parse_args()"
  )
  ok <- data.frame(
    dataset = c("A", "A", "A", "B", "B"),
    method = c(
      "fast_low_modularity", "slow_high_modularity", "lower_ari",
      "fast_missing_quality", "slow_missing_quality"
    ),
    ari = c(0.95, 0.95, 0.94, NA, NA),
    modularity = c(0.20, 0.40, 0.99, NA, NA),
    total_sec = c(1, 2, 0.5, 1, 2),
    stringsAsFactors = FALSE
  )

  ranked <- env$rank_graph_cluster_success(ok)
  best <- ranked[!duplicated(ranked$dataset), , drop = FALSE]
  expect_equal(best$method, c("slow_high_modularity", "fast_missing_quality"))
})

test_that("graph benchmark best rows can preserve k and target dimensions", {
  env <- source_benchmark_helpers(
    test_path("../../benchmark_scripts/benchmark_graph_clustering.R"),
    "args <- parse_args()"
  )
  ok <- data.frame(
    dataset = c("A", "A", "A", "A"),
    k = c(15L, 15L, 50L, 50L),
    n_clusters_requested = c(3L, 3L, 3L, 5L),
    method = c("fast_low_ari", "slow_high_ari", "k50", "target5"),
    ari = c(0.8, 0.9, 0.7, 0.95),
    modularity = c(0.1, 0.1, 0.2, 0.3),
    total_sec = c(1, 2, 1, 1),
    stringsAsFactors = FALSE
  )

  compact <- env$select_graph_best_rows(ok, group_cols = "dataset")
  by_target <- env$select_graph_best_rows(ok, group_cols = c("dataset", "k", "n_clusters_requested"))

  expect_equal(compact$method, "target5")
  expect_setequal(by_target$method, c("slow_high_ari", "k50", "target5"))
  expect_equal(
    nrow(unique(by_target[, c("dataset", "k", "n_clusters_requested"), drop = FALSE])),
    3L
  )
})

test_that("graph benchmark cycle summaries preserve target cluster count", {
  env <- source_benchmark_helpers(
    test_path("../../benchmark_scripts/benchmark_graph_clustering.R"),
    "args <- parse_args()"
  )
  ok <- data.frame(
    dataset = c("A", "A"),
    n = c(100L, 100L),
    p = c(4L, 4L),
    cycle = c(1L, 1L),
    k = c(15L, 15L),
    graph_backend = c("cpu", "cpu"),
    graph_resolved_backend = c("cpu", "cpu"),
    graph_preflight_route = c("cpu", "cpu"),
    graph_route_parameters = c(
      "nn_approximation.strategy=faiss_IndexHNSWFlat;nn_approximation.tuning_rule=balanced_shape_metric",
      "nn_approximation.strategy=faiss_IndexHNSWFlat;nn_approximation.tuning_rule=balanced_shape_metric"
    ),
    cluster_backend = c("cpu", "cpu"),
    cluster_resolved_backend = c("cpu", "cpu"),
    cluster_preflight_route = c("cpu", "cpu"),
    method = c("louvain", "louvain"),
    weight = c("snn", "snn"),
    n_clusters_requested = c(3L, 5L),
    n_clusters_source = c("stored_graph_target", "labels"),
    n_threads = c(2L, 2L),
    status = c("success", "success"),
    error = c(NA_character_, NA_character_),
    load_sec = c(0.1, 0.1),
    graph_sec = c(1, 1),
    cluster_sec = c(2, 3),
    total_sec = c(3, 4),
    peak_rss_gb = c(1, 1),
    graph_n_vertices = c(100L, 100L),
    n_edges = c(500L, 500L),
    n_communities = c(3L, 5L),
    modularity = c(0.4, 0.35),
    ari = c(0.9, 0.8),
    selected_resolution = c(1, 2),
    target_gap = c(0L, 0L),
    resolution_selection = c(
      "closest_n_communities_then_highest_modularity",
      "closest_n_communities_then_highest_modularity"
    ),
    resolution_selected_candidate = c(4L, 5L),
    resolution_candidates = c(7L, 7L),
    resolution_min_target_gap = c(0L, 0L),
    resolution_selected_is_min_gap = c(TRUE, TRUE),
    graph_cached = c(TRUE, TRUE),
    expected_skip = c(FALSE, FALSE)
  )

  out <- env$summarize_graph_cycles(ok)
  expect_equal(nrow(out), 2L)
  expect_type(out$n_clusters_requested, "integer")
  expect_equal(sort(as.integer(out$n_clusters_requested)), c(3L, 5L))
  expect_equal(sort(out$n_clusters_source), c("labels", "stored_graph_target"))
  expect_equal(out$median_graph_n_vertices, c(100, 100))
  expect_equal(out$graph_preflight_route, c("cpu", "cpu"))
  expect_equal(
    out$graph_route_parameters,
    rep("nn_approximation.strategy=faiss_IndexHNSWFlat;nn_approximation.tuning_rule=balanced_shape_metric", 2L)
  )
  expect_equal(out$cluster_preflight_route, c("cpu", "cpu"))
  expect_equal(out$n_threads, c(2L, 2L))
  expect_equal(out$median_target_gap, c(0, 0))
  expect_equal(out$median_resolution_selected_candidate, c(4, 5))
  expect_equal(out$median_resolution_candidates, c(7, 7))
  expect_equal(out$median_resolution_min_target_gap, c(0, 0))
  expect_true(all(out$resolution_selected_is_min_gap))
  expect_equal(
    out$resolution_selection,
    rep("closest_n_communities_then_highest_modularity", 2L)
  )
})

test_that("graph benchmark extracts compact graph KNN route parameters", {
  env <- source_benchmark_helpers(
    test_path("../../benchmark_scripts/benchmark_graph_clustering.R"),
    "args <- parse_args()"
  )
  graph <- list(from = c(1L, 2L), to = c(2L, 3L), weight = c(1, 1), n_vertices = 3L, n_edges = 2L)
  attr(graph, "faissR_graph") <- list(
    nn_approximation = list(
      strategy = "faiss_IndexHNSWFlat",
      backend = "faiss_hnsw",
      m = 32L,
      ef_search = 150L,
      tuning_rule = "balanced_shape_metric",
      tuning_large_k = FALSE
    ),
    nn_auto_selection = list(
      policy = "static_shape_k_metric_selector",
      explicit_backend = TRUE,
      explicit_method = FALSE,
      backend_decision = "explicit_cpu",
      method_decision = "cpu_auto_shape_selector",
      predicted_backend = "faiss_hnsw",
      predicted_method = "hnsw",
      predicted_device = "cpu",
      slow_tuning = FALSE
    ),
    nn_metric_transform = "row_l2_normalize_then_euclidean_graph_search",
    nn_distance_transform = "normalized_euclidean_squared_over_2_to_1_minus_similarity"
  )

  params <- env$graph_route_parameters(graph)
  expect_match(params, "nn_approximation.strategy=faiss_IndexHNSWFlat", fixed = TRUE)
  expect_match(params, "nn_approximation.m=32", fixed = TRUE)
  expect_match(params, "nn_approximation.ef_search=150", fixed = TRUE)
  expect_match(params, "nn_approximation.tuning_rule=balanced_shape_metric", fixed = TRUE)
  expect_match(params, "nn_approximation.tuning_large_k=FALSE", fixed = TRUE)
  expect_match(params, "nn_auto_selection.policy=static_shape_k_metric_selector", fixed = TRUE)
  expect_match(params, "nn_auto_selection.explicit_backend=TRUE", fixed = TRUE)
  expect_match(params, "nn_auto_selection.explicit_method=FALSE", fixed = TRUE)
  expect_match(params, "nn_auto_selection.backend_decision=explicit_cpu", fixed = TRUE)
  expect_match(params, "nn_auto_selection.method_decision=cpu_auto_shape_selector", fixed = TRUE)
  expect_match(params, "nn_auto_selection.predicted_method=hnsw", fixed = TRUE)
  expect_match(params, "nn_auto_selection.predicted_device=cpu", fixed = TRUE)
  expect_match(params, "nn_auto_selection.slow_tuning=FALSE", fixed = TRUE)
  expect_match(params, "nn_metric.metric_transform=row_l2_normalize_then_euclidean_graph_search", fixed = TRUE)
  expect_match(params, "nn_metric.distance_transform=normalized_euclidean_squared_over_2_to_1_minus_similarity", fixed = TRUE)
})

test_that("graph benchmark summarizes graph_cluster resolution search diagnostics", {
  env <- source_benchmark_helpers(
    test_path("../../benchmark_scripts/benchmark_graph_clustering.R"),
    "args <- parse_args()"
  )
  search <- data.frame(
    candidate = 1:3,
    resolution = c(0.5, 1, 2),
    n_communities = c(2L, 3L, 5L),
    modularity = c(0.3, 0.4, 0.35),
    target_gap = c(1L, 0L, 2L),
    selected = c(FALSE, TRUE, FALSE)
  )
  cluster <- list(
    resolution_search = search,
    resolution_selection = list(
      criterion = "closest_n_communities_then_highest_modularity",
      selected_candidate = 2L
    )
  )

  out <- env$resolution_search_summary(cluster)
  expect_equal(out$selected_candidate, 2L)
  expect_equal(out$candidates, 3L)
  expect_equal(out$min_target_gap, 0L)
  expect_true(out$selected_is_min_gap)
})

test_that("graph benchmark success rows keep preflight and resolved routes separate", {
  env <- source_benchmark_helpers(
    test_path("../../benchmark_scripts/benchmark_graph_clustering.R"),
    "args <- parse_args()"
  )
  set.seed(1771)
  x <- rbind(
    matrix(rnorm(40, -2, 0.2), ncol = 4),
    matrix(rnorm(40, 2, 0.2), ncol = 4)
  )
  labels <- rep(c("a", "b"), each = 10L)
  graph_obj <- knn_graph(x, k = 4L, backend = "cpu", weight = "distance")
  graph_meta <- attr(graph_obj, "faissR_graph")
  graph_meta$resolved_backend <- "faiss_hnsw"
  attr(graph_obj, "faissR_graph") <- graph_meta

  row <- env$run_cluster_one(
    data_obj = list(data = x, labels = labels),
    dataset_name = "A",
    graph_obj = graph_obj,
    graph_sec = 0.1,
    graph_cached = TRUE,
    graph_n_vertices = graph_obj$n_vertices,
    n_edges = graph_obj$n_edges,
    k = 4L,
    graph_backend = "auto",
    cluster_backend = "cpu",
    method = "louvain",
    weight = "distance",
    n_threads = 2L,
    target_mode = "none",
    seed = 1L,
    timeout = 60L,
    cycle = 1L
  )

  expect_equal(row$status, "success")
  expect_equal(row$graph_resolved_backend, "faiss_hnsw")
  expect_equal(row$graph_preflight_route, env$graph_build_preflight_route("auto"))
  expect_false(identical(row$graph_preflight_route, row$graph_resolved_backend))
})

test_that("graph benchmark recommendations preserve integer target counts", {
  env <- source_benchmark_helpers(
    test_path("../../benchmark_scripts/benchmark_graph_clustering.R"),
    "args <- parse_args()"
  )
  cycle_summary <- data.frame(
    dataset = c("A", "A"),
    k = c(15L, 15L),
    graph_backend = c("cpu", "cpu"),
    graph_resolved_backend = c("cpu", "cpu"),
    cluster_backend = c("cpu", "cpu"),
    cluster_resolved_backend = c("cpu", "cpu"),
    method = c("louvain", "leiden"),
    weight = c("snn", "snn"),
    success_cycles = c(1L, 1L),
    median_graph_sec = c(1, 1),
    median_cluster_sec = c(2, 1),
    median_total_sec = c(3, 2),
    median_ari = c(0.9, 0.9),
    min_ari = c(0.9, 0.9),
    median_modularity = c(0.4, 0.4),
    median_n_communities = c(3, 3),
    median_selected_resolution = c(1, 1),
    n_clusters_requested = c(3L, 3L),
    n_clusters_source = c("labels", "labels"),
    graph_cached = c(TRUE, TRUE)
  )

  out <- env$recommend_graph_cluster_methods(cycle_summary, ari_tolerance = 0.01)
  expect_type(out$n_clusters_requested, "integer")
  expect_equal(out$n_clusters_requested, 3L)
})

test_that("graph benchmark auto comparison has unique schema columns", {
  env <- source_benchmark_helpers(
    test_path("../../benchmark_scripts/benchmark_graph_clustering.R"),
    "args <- parse_args()"
  )
  cycle_summary <- data.frame(
    dataset = c("A", "A", "A"),
    k = c(15L, 15L, 15L),
    n_clusters_requested = c(3L, 3L, 3L),
    graph_backend = c("auto", "auto", "cpu"),
    cluster_backend = c("auto", "auto", "cpu"),
    graph_resolved_backend = c("cpu", "cpu", "cpu"),
    graph_preflight_route = c("cpu", "cpu", "cpu"),
    cluster_resolved_backend = c("cpu", "cpu", "cpu"),
    cluster_preflight_route = c("cpu", "cpu", "cpu"),
    method = c("louvain", "leiden", "leiden"),
    weight = c("snn", "snn", "snn"),
    n_threads = c(2L, 2L, 2L),
    success_cycles = c(2L, 2L, 2L),
    median_graph_sec = c(1, 1, 1),
    median_cluster_sec = c(3, 2, 2),
    median_total_sec = c(4, 3, 3),
    median_ari = c(0.90, 0.91, 0.91),
    min_ari = c(0.89, 0.90, 0.90),
    median_modularity = c(0.4, 0.41, 0.41),
    median_n_communities = c(3, 3, 3),
    median_selected_resolution = c(1, 1, 1),
    n_clusters_source = c("labels", "labels", "labels"),
    graph_cached = c(TRUE, TRUE, TRUE)
  )
  recommendations <- cycle_summary[2, , drop = FALSE]
  recommendations$recommendation_basis <- "fastest_within_ari_tolerance"

  out <- env$compare_auto_graph_to_recommendations(cycle_summary, recommendations)

  expect_equal(anyDuplicated(names(out)), 0L)
  expect_true("n_clusters_requested" %in% names(out))
  expect_true("graph_backend" %in% names(out))
  expect_true("cluster_backend" %in% names(out))
  expect_false("auto_n_clusters_requested" %in% names(out))
  expect_false("recommended_n_clusters_requested" %in% names(out))
  expect_false("auto_graph_backend" %in% names(out))
  expect_false("recommended_graph_backend" %in% names(out))
  expect_true(all(out$auto_uses_recommended_graph_backend))
  expect_true(all(out$auto_uses_recommended_cluster_backend))
  expect_true(all(out$auto_uses_recommended_graph_resolved_backend))
  expect_true(all(out$auto_uses_recommended_cluster_resolved_backend))
  expect_equal(out$recommended_recommendation_basis, rep("fastest_within_ari_tolerance", nrow(out)))
  louvain_row <- out[out$auto_method == "louvain", , drop = FALSE]
  expect_equal(nrow(louvain_row), 1L)
  expect_equal(louvain_row$recommended_method, "leiden")
  expect_equal(louvain_row$auto_graph_preflight_route, "cpu")
  expect_equal(louvain_row$recommended_cluster_preflight_route, "cpu")
  expect_equal(louvain_row$auto_n_threads, 2L)
  expect_equal(louvain_row$recommended_n_threads, 2L)
})

test_that("graph benchmark global recommendations pool across requested backends", {
  env <- source_benchmark_helpers(
    test_path("../../benchmark_scripts/benchmark_graph_clustering.R"),
    "args <- parse_args()"
  )
  cycle_summary <- data.frame(
    dataset = c("A", "A", "A", "A"),
    k = c(15L, 15L, 15L, 15L),
    graph_method = c("auto", "auto", "auto", "auto"),
    metric = c("euclidean", "euclidean", "euclidean", "euclidean"),
    n_clusters_requested = c(3L, 3L, 3L, 3L),
    graph_backend = c("auto", "cpu", "cuda", "cuda"),
    cluster_backend = c("auto", "cpu", "cuda", "cuda"),
    graph_resolved_backend = c("cpu", "cpu", "cuda", "cuda"),
    graph_preflight_route = c("cpu", "cpu", "cuda", "cuda"),
    cluster_resolved_backend = c("cpu", "cpu", "cuda", "cuda"),
    cluster_preflight_route = c("cpu", "cpu", "cuda", "cuda"),
    method = c("louvain", "leiden", "louvain", "leiden"),
    weight = c("snn", "snn", "snn", "snn"),
    n_threads = c(2L, 2L, 2L, 2L),
    success_cycles = c(2L, 2L, 2L, 2L),
    median_graph_sec = c(1, 1, 1, 1),
    median_cluster_sec = c(3, 4, 1, 2),
    median_total_sec = c(4, 5, 2, 3),
    median_ari = c(0.94, 0.95, 0.94, 0.90),
    min_ari = c(0.93, 0.94, 0.93, 0.89),
    median_modularity = c(0.40, 0.42, 0.41, 0.39),
    median_n_communities = c(3, 3, 3, 3),
    median_selected_resolution = c(1, 1, 1, 1),
    n_clusters_source = c("labels", "labels", "labels", "labels"),
    graph_cached = c(TRUE, TRUE, TRUE, TRUE)
  )

  rec <- env$recommend_graph_cluster_global_methods(cycle_summary, ari_tolerance = 0.02)
  expect_equal(nrow(rec), 1L)
  expect_equal(rec$graph_backend, "cuda")
  expect_equal(rec$cluster_backend, "cuda")
  expect_equal(rec$method, "louvain")
  expect_equal(rec$recommendation_basis, "global_fastest_within_ari_tolerance")

  comparison <- env$compare_auto_graph_to_global_recommendations(cycle_summary, rec)
  expect_equal(anyDuplicated(names(comparison)), 0L)
  expect_true("auto_graph_backend" %in% names(comparison))
  expect_true("recommended_graph_backend" %in% names(comparison))
  auto_louvain <- comparison[comparison$auto_method == "louvain", , drop = FALSE]
  expect_equal(nrow(auto_louvain), 1L)
  expect_equal(auto_louvain$recommended_graph_backend, "cuda")
  expect_false(auto_louvain$auto_uses_recommended_requested_graph_backend)
  expect_false(auto_louvain$auto_uses_recommended_graph_resolved_backend)
  expect_equal(auto_louvain$auto_median_speed_ratio, 2)
  expect_equal(auto_louvain$auto_median_ari_gap, 0)
})

test_that("graph benchmark auto comparison guards speed and quality gaps", {
  env <- source_benchmark_helpers(
    test_path("../../benchmark_scripts/benchmark_graph_clustering.R"),
    "args <- parse_args()"
  )
  cycle_summary <- data.frame(
    dataset = c("A", "A", "B", "B"),
    k = c(15L, 15L, 15L, 15L),
    n_clusters_requested = c(3L, 3L, 3L, 3L),
    graph_backend = c("auto", "auto", "auto", "auto"),
    graph_resolved_backend = c("cpu", "cpu", "cpu", "cpu"),
    graph_preflight_route = c("cpu", "cpu", "cpu", "cpu"),
    cluster_backend = c("auto", "auto", "auto", "auto"),
    cluster_resolved_backend = c("cpu", "cpu", "cpu", "cpu"),
    cluster_preflight_route = c("cpu", "cpu", "cpu", "cpu"),
    method = c("louvain", "leiden", "louvain", "leiden"),
    weight = c("snn", "snn", "snn", "snn"),
    n_threads = c(2L, 2L, 2L, 2L),
    success_cycles = c(1L, 1L, 1L, 1L),
    median_graph_sec = c(1, 1, 1, 1),
    median_cluster_sec = c(1, 0, 1, 1),
    median_total_sec = c(2, 0, 2, 3),
    median_ari = c(0.90, 0.91, NA, 0.91),
    min_ari = c(0.89, 0.90, NA, 0.90),
    median_modularity = c(0.40, 0.41, NA, 0.41),
    median_n_communities = c(3, 3, 3, 3),
    median_selected_resolution = c(1, 1, 1, 1),
    n_clusters_source = c("labels", "labels", "labels", "labels"),
    graph_cached = c(TRUE, TRUE, TRUE, TRUE)
  )
  recommendations <- cycle_summary[c(2, 4), , drop = FALSE]
  recommendations$recommendation_basis <- "fastest_within_ari_tolerance"

  out <- env$compare_auto_graph_to_recommendations(cycle_summary, recommendations)
  expect_true(all(is.na(out$auto_median_speed_ratio[out$dataset == "A"])))
  b_louvain <- out[out$dataset == "B" & out$auto_method == "louvain", , drop = FALSE]
  expect_equal(nrow(b_louvain), 1L)
  expect_true(is.na(b_louvain$auto_median_ari_gap))
  expect_true(is.na(b_louvain$auto_modularity_gap))
  expect_true(is.finite(b_louvain$auto_median_speed_ratio))
})
