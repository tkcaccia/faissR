test_that("large IVF candidates remain bounded", {
    grid <- faissR:::nn_cuda_ivf_large_candidates(1e8, 30L)
    expect_true(nrow(grid) > 0L)
    expect_true(all(grid$nlist >= 1L & grid$nlist <= 65536L))
    expect_true(all(grid$nprobe >= 1L & grid$nprobe <= grid$nlist))
    expect_true(any(grid$nlist == 4096L & grid$nprobe == 512L))
    ten_million <- faissR:::nn_cuda_ivf_large_candidates(1e7, 30L)
    expect_true(any(ten_million$nlist == 4096L &
                    ten_million$nprobe == 512L))
    expect_error(faissR:::nn_cuda_ivf_large_candidates(1e7 + 0.5, 30L),
                 "Invalid n or k")
    expect_error(faissR:::nn_cuda_ivf_large_candidates(1e7, 30.5),
                 "Invalid n or k")
    expect_error(faissR:::nn_cuda_ivf_large_candidates(
        1e7, 30L, probe_fraction = numeric()),
        "Invalid IVF candidate multipliers")
})

test_that("untested large IVF defaults do not claim audited recall", {
    for (n in c(1e7, 1e8, 1e9)) {
        plan <- faissR:::cuda_ivf_params(
            n, 96L, 30L, "euclidean", 0.99)
        expect_identical(plan$tuning_benchmark_source,
                         "heuristic_fallback")
        expect_false(plan$tuning_benchmark_target_met)
        expect_lte(plan$nprobe, plan$nlist)
    }
})

test_that("medium low dimension CUDA IVF distinguishes recall tiers", {
    low <- faissR:::cuda_ivf_params(
        100000L, 96L, 30L, "euclidean", 0.90)
    mid <- faissR:::cuda_ivf_params(
        100000L, 96L, 30L, "euclidean", 0.95)
    high <- faissR:::cuda_ivf_params(
        100000L, 96L, 30L, "euclidean", 0.99)
    expect_equal(c(low$nprobe, mid$nprobe, high$nprobe),
                 c(99L, 99L, 192L))
    expect_equal(low$nlist, high$nlist)
    expect_identical(high$tuning_benchmark_source,
                     "heuristic_fallback")
    expect_false(high$tuning_benchmark_target_met)
})

test_that("billion-row projection requires audited matching anchors", {
    observations <- data.frame(
        n = c(1e7, 1e7, 1e8, 1e8),
        p = 96L, k = 30L, gpu_count = 4L, mode = "shard",
        nlist = c(4096L, 4096L, 16384L, 16384L),
        nprobe = c(512L, 1024L, 2048L, 4096L),
        recall = c(0.985, 0.995, 0.991, 0.998),
        knn_seconds = c(100, 120, 900, 1300),
        index_build_seconds = c(90, 105, 800, 1100),
        persistent_search_seconds_median = c(10, 15, 100, 200),
        device_memory_bytes = c(2e9, 2e9, 15e9, 15e9),
        peak_rss_bytes = c(6e9, 6e9, 60e9, 60e9),
        full_self_performed = FALSE,
        audit_pass = TRUE,
        reference_scope = "full_database_exact",
        timing_scope = "index_build_plus_persistent_search_256_queries",
        query_count = 256L,
        source_sha = paste(rep("a", 40L), collapse = ""),
        input_sha256 = paste(rep("b", 64L), collapse = ""),
        query_rows_sha256 = paste(rep("c", 64L), collapse = ""),
        query_rows_max = 9000000L,
        database_selection = "prefix_first_n_rows",
        stringsAsFactors = FALSE
    )
    plan <- faissR:::nn_cuda_ivf_large_extrapolate(
        observations, 1e9, 96L, 30L, 4L, "shard", 96e9
    )
    expect_identical(plan$status, "extrapolated_not_validated")
    expect_equal(plan$anchor_rows$n, c(1e7, 1e8))
    expect_true(plan$nprobe <= plan$nlist)
    expect_false(plan$full_self_output_supported)
    expect_true(is.na(plan$recall_at_target))
    expect_true(plan$projected_peak_host_bytes > 0)
    expect_true(is.na(plan$projected_knn_seconds))
    expect_true(is.na(plan$projected_full_self_seconds))
    expect_identical(plan$full_self_time_status,
                     "unvalidated_query_scaling")
    expect_equal(plan$full_self_output_bytes, 12e9 * 30)
    expect_identical(plan$time_projection_status,
                     "device_memory_exceeded")
    expect_identical(plan$device_memory_admission,
                     "projection_exceeds_capacity")
    expect_gte(plan$projected_peak_device_bytes,
               plan$raw_index_bytes_per_device)
    falling <- observations
    falling$nprobe <- c(512L, 1024L, 256L, 512L)
    conservative <- faissR:::nn_cuda_ivf_large_extrapolate(
        falling, 1e9, 96L, 30L, 4L, "shard", 1e12
    )
    expect_gte(conservative$nprobe, max(conservative$anchor_rows$nprobe))
    admitted <- faissR:::nn_cuda_ivf_large_extrapolate(
        observations, 1e9, 96L, 30L, 4L, "shard", 1e12
    )
    expect_true(is.finite(admitted$projected_knn_seconds))
    expect_true(is.finite(admitted$projected_index_build_seconds))
    expect_true(is.finite(admitted$projected_256_query_search_seconds))
    expect_true(is.na(admitted$projected_full_self_seconds))
    inconsistent <- observations
    inconsistent$index_build_seconds[2L] <- 1
    expect_error(
        faissR:::nn_cuda_ivf_large_extrapolate(
            inconsistent, 1e9, 96L, 30L, 4L, "shard", 1e12
        ), "consistent 256-query stage timings"
    )
    expect_identical(admitted$time_projection_status,
                     "conditional_extrapolation")
    expect_identical(plan$input_sha256, observations$input_sha256[[1L]])
    expect_identical(plan$query_rows_sha256,
                     observations$query_rows_sha256[[1L]])
    expect_error(
        faissR:::nn_cuda_ivf_large_extrapolate(
            observations, 1e9 + 0.5, 96L, 30L, 4L, "shard", 96e9
        ), "Invalid large IVF extrapolation inputs"
    )
    expect_error(
        faissR:::nn_cuda_ivf_large_extrapolate(
            observations, 1e9, 96L, 30L, 1.5, "shard", 96e9
        ), "Invalid large IVF extrapolation inputs"
    )
    expect_error(
        faissR:::nn_cuda_ivf_large_extrapolate(
            observations, 1e9, 96L, 30L, 4L, "shard", 96e9, 0.975
        ), "Invalid large IVF extrapolation inputs"
    )
    invalid_ivf <- observations
    invalid_ivf$nprobe[invalid_ivf$n == 1e8] <- 20000L
    expect_error(
        faissR:::nn_cuda_ivf_large_extrapolate(
            invalid_ivf, 1e9, 96L, 30L, 4L, "shard", 96e9
        ), "10M and 100M"
    )
    invalid_ivf <- observations
    invalid_ivf$nlist[invalid_ivf$n == 1e8] <- 10000.5
    expect_error(
        faissR:::nn_cuda_ivf_large_extrapolate(
            invalid_ivf, 1e9, 96L, 30L, 4L, "shard", 96e9
        ), "10M and 100M"
    )
    mixed <- observations
    mixed$input_sha256[mixed$n == 1e8] <- paste(rep("c", 64L), collapse = "")
    expect_error(
        faissR:::nn_cuda_ivf_large_extrapolate(
            mixed, 1e9, 96L, 30L, 4L, "shard", 96e9
        ),
        "input-locked"
    )
    mixed$input_sha256 <- observations$input_sha256
    mixed$query_rows_sha256[mixed$n == 1e8] <-
        paste(rep("d", 64L), collapse = "")
    expect_error(
        faissR:::nn_cuda_ivf_large_extrapolate(
            mixed, 1e9, 96L, 30L, 4L, "shard", 96e9
        ),
        "identical query rows"
    )
    mixed$query_rows_sha256 <- observations$query_rows_sha256
    mixed$query_rows_max[mixed$n == 1e8] <- 11000000L
    expect_error(
        faissR:::nn_cuda_ivf_large_extrapolate(
            mixed, 1e9, 96L, 30L, 4L, "shard", 96e9
        ),
        "nested input prefixes"
    )
    mixed$query_rows_max <- observations$query_rows_max
    mixed$full_self_performed[mixed$n == 1e8] <- TRUE
    expect_error(
        faissR:::nn_cuda_ivf_large_extrapolate(
            mixed, 1e9, 96L, 30L, 4L, "shard", 96e9
        ),
        "10M and 100M"
    )
    observations$audit_pass[observations$n == 1e8] <- FALSE
    expect_error(
        faissR:::nn_cuda_ivf_large_extrapolate(
            observations, 1e9, 96L, 30L, 4L, "shard", 96e9
        ),
        "10M and 100M"
    )
})

test_that("IVF projections check a held-out size without fitting to it", {
    n <- c(2e6, 8e6, 4e6, 4e6)
    rows <- data.frame(
        n = n, p = 192L, k = 40L, gpu_count = 2L, mode = "shard",
        nlist = c(4096L, 4096L, 4096L, 4096L),
        nprobe = c(128L, 128L, 128L, 256L),
        recall = c(0.991, 0.992, 0.993, 1),
        index_build_seconds = n / 1e5,
        persistent_search_seconds_median = n / 1e7,
        device_memory_bytes = n * 400, peak_rss_bytes = n * 1600,
        full_self_performed = FALSE, audit_pass = TRUE,
        reference_scope = "full_database_exact",
        timing_scope = "index_build_plus_persistent_search_256_queries",
        query_count = 256L,
        source_sha = paste(rep("a", 40L), collapse = ""),
        input_sha256 = paste(rep("b", 64L), collapse = ""),
        query_rows_sha256 = paste(rep("c", 64L), collapse = ""),
        query_rows_max = 1500000L,
        database_selection = "prefix_first_n_rows"
    )
    rows$knn_seconds <- rows$index_build_seconds +
        rows$persistent_search_seconds_median
    plan <- function(observed = rows, fit = c(2e6, 8e6), holdout = 4e6) {
        faissR:::nn_cuda_ivf_large_extrapolate(
            observed, 16e6, 192L, 40L, 2L, "shard", 20e9,
            fit_sizes = fit, holdout_n = holdout
        )
    }
    selected <- plan()
    expect_identical(selected$holdout$status, "interpolation_pass")
    expect_equal(selected$anchor_rows$n, c(2e6, 8e6))
    expect_identical(selected$holdout$nprobe, 128L)
    expect_equal(selected$holdout$measured_recall, 0.993)
    expect_true(is.na(selected$recall_at_target))
    expect_equal(selected$projected_256_query_search_seconds, 1.6)
    expect_equal(unname(selected$holdout$relative_error), rep(0, 4),
                 tolerance = 1e-10)
    expect_identical(plan(rows[-3L, ])$holdout$status,
                     "predicted_setting_not_audited")
    weak <- rows
    weak$recall[3L] <- 0.95
    expect_identical(plan(weak)$holdout$status, "recall_miss")
    slow <- rows
    slow$index_build_seconds[3L] <- 100
    slow$knn_seconds[3L] <- 100.4
    expect_identical(plan(slow)$holdout$status, "prediction_error")
    expect_error(plan(holdout = 2e6), "distinct")
    expect_error(plan(holdout = 1e6), "must lie between")
    expect_error(plan(fit = c(2e6, 6e6, 8e6)), "declared fit size")
})
