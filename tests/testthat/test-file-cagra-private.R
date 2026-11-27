test_that("file CAGRA queries an external row view with shard-local IDs", {
    skip_if_not(isTRUE(cuvs_available()))
    skip_if_not(isTRUE(cuda_available()))
    withr::local_envvar(c(FAISSR_CUVS_CAGRA_NN_DESCENT_NITER = "12"))
    set.seed(42L)
    reference <- matrix(runif(768L * 4L), ncol = 4L)
    queries <- reference[c(60L, 73L, 91L, 254L, 401L), , drop = FALSE]
    path <- tempfile(fileext = ".fbin")
    con <- file(path, "wb")
    writeBin(c(nrow(reference), ncol(reference)), con, endian = "little")
    writeBin(as.numeric(t(reference)), con, size = 4L, endian = "little")
    close(con)
    on.exit(unlink(path))
    source <- list(format = "fbin", path = path, nrow = 768L, ncol = 4L)
    shard <- list(format = "view", source = source, first = 50L, nrow = 500L)
    index <- file_cagra_build_cpp(shard, 8L, 16L, 2^30, 4L, 64L,
        TRUE, 32L, 64L)
    on.exit(file_cagra_release_cpp(index$pointer), add = TRUE)
    expect_identical(index$nn_descent_niter, 12L)

    raw <- tempfile(fileext = ".f32")
    writeBin(as.numeric(t(queries)), raw, size = 4L, endian = "little")
    on.exit(unlink(raw), add = TRUE)
    external <- list(format = "f32", path = raw, nrow = 5L, ncol = 4L)
    result <- file_cagra_query_cpp(index$pointer, external, 2L, 3L)
    expect_identical(result$backend_used, "cuda_cuvs_cagra_file_query")
    expect_true(result$index_reused)
    expect_identical(dim(result$indices), c(3L, 9L))
    expect_equal(as.integer(result$indices[, 1L]), c(24L, 42L, 205L))
    expect_equal(as.numeric(result$distances[, 1L]), rep(0, 3L),
        tolerance = 1e-3)
    expect_true(all(result$indices >= 1L & result$indices <= 500L))
    expect_error(file_cagra_query_cpp(index$pointer, external, 5L, 2L),
        "outside its file view")
    expect_identical(file_cagra_query_cpp(index$pointer, external, 2L, 3L),
        result)
    expect_error(file_cagra_set_search_cpp(index$pointer, 2L, 8L),
        "Invalid CUDA CAGRA file search")
    expect_identical(file_cagra_set_search_cpp(index$pointer, 2L, 32L),
        list(width = 2L, queue = 32L))
    changed <- file_cagra_query_cpp(index$pointer, external, 2L, 3L)
    expect_equal(changed$search_width, 2)
    expect_equal(changed$search_queue, 32)
    expect_true(changed$index_reused)
})

test_that("file CAGRA only advances verified speed and recall improvements", {
    incumbent <- list(width = 6L, queue = 384L, batch = 256L)
    grid <- cuvs_cagra_file_search_candidates(128L, incumbent, 768L)
    expect_true(all(grid$queue > 128L))
    expect_equal(sort(unique(grid$width)), c(3L, 6L))
    expect_equal(sort(unique(grid$batch)), c(256L, 768L))
    wide <- cuvs_cagra_file_search_candidates(128L, incumbent, 4096L)
    expect_equal(sort(unique(wide$batch)), c(256L, 4096L))
    expect_error(cuvs_cagra_file_search_candidates(128L, incumbent, 4097L))

    rows <- data.frame(width = c(6L, 3L, 3L, 3L),
        queue = c(384L, 256L, 192L, 129L), batch = c(256L, 768L, 768L, 768L),
        status = c("success", "success", "success", "failed"),
        seconds = c(2, 1.5, 1, 0.5),
        calibration_recall = c(.999, .995, .997, .999),
        holdout_recall = c(.998, .994, .990, .999))
    tuned <- cuvs_cagra_select_file_search(rows, 128L, incumbent, .99)
    expect_identical(tuned$status, "target_met")
    expect_identical(tuned$params, list(width = 3L, queue = 256L, batch = 768L))
    expect_identical(cuvs_cagra_select_file_search(rows, 128L, incumbent, .95)
        $params$queue, 192L)
    expect_identical(cuvs_cagra_select_file_search(rows, 128L, incumbent, .99,
        minimum_speedup = 1.5)$status, "incumbent_kept")
    expect_identical(cuvs_cagra_select_file_search(rows[-1L, ], 128L,
        incumbent, .99)$status, "baseline_unverified")
    expect_error(cuvs_cagra_file_search(list(), 128L,
        list(width = 3L, queue = 128L, batch = 768L)), "Invalid CUDA CAGRA")
})

test_that("multi-device file CAGRA uses the admitted profile", {
    observed <- NULL
    testthat::local_mocked_bindings(file_cagra_multi_build_cpp = function(
            source, k, batch, memory_limit, width, queue, degree,
            intermediate, devices, merge_workers) {
        observed <<- list(batch, memory_limit, degree, intermediate,
            devices, merge_workers)
        list(shard_rows = c(50001L, 50000L), nn_descent_niter = 12L)
    }, .package = "faissR")
    withr::local_options(list(faissR.cuvs_cagra_file_search = NULL))
    profile <- list(graph_degree = 64L, intermediate_degree = 96L,
        batch_rows = 256L, memory_limit_bytes = 8 * 2^30,
        builder = "nn_descent")
    index <- faissR:::cuvs_cagra_wide_file_multi(
        list(nrow = 100001L, ncol = 96L), 30L, 1024L, 16 * 2^30,
        0.99, TRUE, 0:1, merge_workers = 2L, profile = profile)
    expect_identical(observed, list(256L, 8 * 2^30, 64L, 96L, 0:1, 2L))
    expect_identical(index$params$shard_rows, c(50001L, 50000L))
    expect_identical(index$params$nn_descent_niter, 12L)
})
