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
      expect_match(skip$notes, "euclidean|non-Euclidean|unsupported|rejected", ignore.case = TRUE)
    }
  }
})

test_that("NN metric benchmark recommendations are grouped by backend metric and k", {
  env <- source_benchmark_helpers(
    test_path("../../benchmark_scripts/benchmark_nn_metrics.R"),
    "args <- parse_args()"
  )
  cycle_summary <- data.frame(
    dataset = c("A", "A", "A", "A", "A", "A"),
    backend = c("cpu", "cpu", "cpu", "cpu", "cuda", "cuda"),
    method = c("exact", "hnsw", "exact", "hnsw", "flat", "cagra"),
    metric = c("euclidean", "euclidean", "cosine", "cosine", "euclidean", "euclidean"),
    k = c(15L, 15L, 15L, 15L, 50L, 50L),
    median_elapsed_sec = c(10, 2, 8, 3, 4, 1),
    median_recall_at_k = c(1.00, 0.99, 1.00, 0.97, 1.00, 0.99),
    min_recall_at_k = c(1.00, 0.98, 1.00, 0.96, 1.00, 0.98),
    median_min_recall_at_k = c(1.00, 0.98, 1.00, 0.96, 1.00, 0.98),
    recall_reference = c("exact", "exact", "exact", "exact", "exact", "exact"),
    median_recall_query_n = c(100, 100, 100, 100, 100, 100),
    result_backend = c("cpu", "faiss_hnsw", "cpu", "faiss_hnsw", "faiss_gpu_flat_l2", "faiss_gpu_cagra"),
    resolved_backend = c("cpu", "faiss_hnsw", "cpu", "faiss_hnsw", "faiss_gpu_flat_l2", "faiss_gpu_cagra"),
    implementation_backend = c("cpu", "faiss_hnsw", "cpu", "faiss_hnsw", "faiss_gpu_flat_l2", "faiss_gpu_cagra"),
    success_cycles = c(2L, 2L, 2L, 2L, 2L, 2L)
  )

  out <- env$recommend_nn_methods(cycle_summary, recall_threshold = 0.98)
  expect_equal(nrow(out), 3L)
  expect_equal(out$backend, c("cpu", "cpu", "cuda"))
  expect_equal(out$metric, c("cosine", "euclidean", "euclidean"))
  expect_equal(as.integer(out$k), c(15L, 15L, 50L))
  expect_equal(out$method, c("exact", "hnsw", "cagra"))
})

test_that("legacy Benchmark #1 uses canonical Flat rows for inner product", {
  env <- source_benchmark_helpers(
    test_path("../../benchmark_scripts/benchmark1_nn_speed.R"),
    "if (worker)"
  )

  methods <- env$method_table()
  expect_false(any(methods$method %in% c("faissR_faiss_flat_ip", "faissR_faiss_gpu_flat_ip")))
  expect_true(all(c("faissR_faiss_flat_l2", "faissR_faiss_gpu_flat_l2") %in% methods$method))

  aliases <- env$benchmark_method_aliases(c("flat", "faissR_faiss_flat_ip", "faissR_faiss_gpu_flat_ip"))
  expect_equal(aliases, c("faissR_faiss_flat_l2", "faissR_faiss_gpu_flat_l2"))

  expect_true(isTRUE(env$method_metric_applicable("faissR_faiss_flat_l2", "inner_product")$ok))
  expect_true(isTRUE(env$method_metric_applicable("faissR_faiss_gpu_flat_l2", "inner_product")$ok))
  expect_true(isTRUE(env$method_is_exact("faissR_faiss_flat_l2", "inner_product")))
  expect_true(isTRUE(env$method_is_exact("faissR_faiss_gpu_flat_l2", "inner_product")))
})

test_that("k-means benchmark recommendations are grouped by dataset and centers", {
  env <- source_benchmark_helpers(
    test_path("../../benchmark_scripts/benchmark_kmeans.R"),
    "args <- parse_args()"
  )
  cycle_summary <- data.frame(
    dataset = c("A", "A", "A", "A"),
    centers = c(2L, 2L, 3L, 3L),
    method = c("m1", "m2", "m1", "m2"),
    backend = c("cpu", "cpu", "cpu", "cpu"),
    metric = c("euclidean", "euclidean", "euclidean", "euclidean"),
    median_ari = c(0.90, 0.89, 0.70, 0.69),
    median_elapsed_sec = c(2, 1, 5, 3)
  )

  out <- env$recommend_kmeans_methods(cycle_summary, ari_tolerance = 0.02)
  expect_equal(nrow(out), 2L)
  expect_equal(as.integer(out$centers), c(2L, 3L))
  expect_equal(out$method, c("m2", "m2"))
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
    resolved_backend = c("cuda", "cpu", "cuda", "cpu", "stats", "stats"),
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
    tuning_policy = c("auto", "auto", "auto", "auto", "stats", "stats")
  )
  recommendations <- cycle_summary[cycle_summary$method == "stats", , drop = FALSE]

  out <- env$compare_fast_kmeans_to_recommendations(cycle_summary, recommendations)
  expect_equal(as.integer(out$centers), c(3L, 3L, 5L, 5L))
  expect_equal(out$fast_backend, c("cpu", "cuda", "cpu", "cuda"))
})

test_that("graph benchmark recommendations are grouped by target cluster count", {
  env <- source_benchmark_helpers(
    test_path("../../benchmark_scripts/benchmark_graph_clustering.R"),
    "args <- parse_args()"
  )
  cycle_summary <- data.frame(
    dataset = c("A", "A", "A", "A"),
    k = c(15L, 15L, 15L, 15L),
    graph_backend = c("cpu", "cpu", "cpu", "cpu"),
    graph_resolved_backend = c("cpu", "cpu", "cpu", "cpu"),
    cluster_backend = c("cpu", "cpu", "cpu", "cpu"),
    cluster_resolved_backend = c("cpu", "cpu", "cpu", "cpu"),
    method = c("louvain", "leiden", "louvain", "leiden"),
    weight = c("snn", "snn", "snn", "snn"),
    success_cycles = c(1L, 1L, 1L, 1L),
    median_graph_sec = c(1, 1, 1, 1),
    median_cluster_sec = c(4, 2, 3, 1),
    median_total_sec = c(5, 3, 4, 2),
    median_ari = c(0.91, 0.90, 0.72, 0.71),
    min_ari = c(0.91, 0.90, 0.72, 0.71),
    median_modularity = c(0.4, 0.39, 0.3, 0.29),
    median_n_communities = c(3, 3, 5, 5),
    median_selected_resolution = c(1, 1, 2, 2),
    n_clusters_requested = c(3L, 3L, 5L, 5L),
    graph_cached = c(TRUE, TRUE, TRUE, TRUE)
  )

  out <- env$recommend_graph_cluster_methods(cycle_summary, ari_tolerance = 0.02)
  expect_equal(nrow(out), 2L)
  expect_equal(as.integer(out$n_clusters_requested), c(3L, 5L))
  expect_equal(out$method, c("leiden", "leiden"))
})
