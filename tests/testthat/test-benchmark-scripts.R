source_benchmark_helpers <- function(path, stop_marker) {
  if (!file.exists(path)) {
    testthat::skip("Benchmark scripts are not available in this installed-package test context.")
  }
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
