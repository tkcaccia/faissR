test_that("large IVF candidates remain bounded", {
    grid <- faissR:::nn_cuda_ivf_large_candidates(1e8, 30L)
    expect_true(nrow(grid) > 0L)
    expect_true(all(grid$nlist >= 1L & grid$nlist <= 65536L))
    expect_true(all(grid$nprobe >= 1L & grid$nprobe <= grid$nlist))
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
        audit_pass = TRUE,
        reference_scope = "full_database_exact",
        timing_scope = "fixed_query_private_r_call",
        query_count = 256L,
        source_sha = paste(rep("a", 40L), collapse = ""),
        input_sha256 = paste(rep("b", 64L), collapse = ""),
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
    expect_identical(plan$input_sha256, observations$input_sha256[[1L]])
    mixed <- observations
    mixed$input_sha256[mixed$n == 1e8] <- paste(rep("c", 64L), collapse = "")
    expect_error(
        faissR:::nn_cuda_ivf_large_extrapolate(
            mixed, 1e9, 96L, 30L, 4L, "shard", 96e9
        ),
        "input-locked"
    )
    observations$audit_pass[observations$n == 1e8] <- FALSE
    expect_error(
        faissR:::nn_cuda_ivf_large_extrapolate(
            observations, 1e9, 96L, 30L, 4L, "shard", 96e9
        ),
        "10M and 100M"
    )
})
