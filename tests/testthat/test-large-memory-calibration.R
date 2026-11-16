memory_rows <- function() {
    data.frame(
        role = c("fit", "holdout", "fit"),
        n = c(1e7, 5e7, 1e8),
        p = 96L, k = 30L, gpu_count = 2L,
        mode = "shard", nlist = 4096L, nprobe = 128L,
        peak_device_bytes = c(
            5195694080, 16354639872, 27717009408
        ),
        source_sha = rep(
            "dcbb0edfebdb0b2e059b01616b1f5e692d1df0f2", 3L
        ),
        input_sha256 = rep(
            paste(rep("a", 64L), collapse = ""), 3L
        ),
        query_rows_sha256 = rep(
            paste(rep("b", 64L), collapse = ""), 3L
        ),
        audit_pass = TRUE,
        reference_scope = "full_database_exact"
    )
}

test_that("held-out memory evidence changes the raw 1B device floor", {
    plan <- faissR:::nn_large_knn_capacity_from_audits(
        1e9, 96L, 30L, "cuda", "resident", 1.2e12,
        47667740672, 16L, memory_rows()
    )
    expect_equal(plan$minimum_cuda_gpus_with_headroom, 11)
    expect_equal(plan$minimum_cuda_gpus_affine_peak, 15)
    expect_equal(plan$minimum_cuda_gpus_affine_with_holdout_error, 16)
    expect_true(plan$requested_gpu_count_meets_calibrated_headroom)
    expect_equal(plan$holdout_memory_relative_error, 0.075597477529941,
                 tolerance = 1e-12)
    expect_true(is.na(plan$measured_recall))
    expect_match(plan$calibration_status, "not_1b_route_validation")
    too_few <- faissR:::nn_large_knn_capacity_from_audits(
        1e9, 96L, 30L, "cuda", "resident", 1.2e12,
        47667740672, 15L, memory_rows()
    )
    expect_false(too_few$requested_gpu_count_meets_calibrated_headroom)
})

test_that("capacity calibration rejects failed or mixed audits", {
    rows <- memory_rows()
    rows$audit_pass[[2L]] <- FALSE
    expect_error(faissR:::nn_large_knn_capacity_from_audits(
        1e9, 96L, 30L, "cuda", "resident", 1.2e12,
        47667740672, 16L, rows
    ), "Complete audited")
    rows <- memory_rows()
    rows$source_sha[[2L]] <- paste(rep("c", 40L), collapse = "")
    expect_error(faissR:::nn_large_knn_capacity_from_audits(
        1e9, 96L, 30L, "cuda", "resident", 1.2e12,
        47667740672, 16L, rows
    ), "mix source")
    rows <- memory_rows()
    rows$peak_device_bytes[[2L]] <- 25e9
    expect_error(faissR:::nn_large_knn_capacity_from_audits(
        1e9, 96L, 30L, "cuda", "resident", 1.2e12,
        47667740672, 16L, rows
    ), "calibration failed")
    expect_error(faissR:::nn_large_knn_capacity_from_audits(
        1e9, 96L, 30L, "metal", "resident", 1.2e12,
        47667740672, 1L, memory_rows()
    ), "resident CUDA")
})
