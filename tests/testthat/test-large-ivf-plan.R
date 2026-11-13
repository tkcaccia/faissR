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

test_that("billion-row projection requires audited matching anchors", {
    observations <- data.frame(
        n = c(1e7, 1e7, 1e8, 1e8),
        p = 96L, k = 30L, gpu_count = 4L, mode = "shard",
        nlist = c(4096L, 4096L, 16384L, 16384L),
        nprobe = c(512L, 1024L, 2048L, 4096L),
        recall = c(0.985, 0.995, 0.991, 0.998),
        knn_seconds = c(100, 120, 900, 1300),
        device_memory_bytes = c(2e9, 2e9, 15e9, 15e9),
        peak_rss_bytes = c(6e9, 6e9, 60e9, 60e9),
        full_self_performed = FALSE,
        audit_pass = TRUE,
        reference_scope = "full_database_exact",
        timing_scope = "fixed_query_private_r_call",
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
