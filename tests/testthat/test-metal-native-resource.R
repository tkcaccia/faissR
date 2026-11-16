test_that("private Metal KNN reports scoped in-call allocation samples", {
    skip_if_not(isTRUE(faissR:::nn_metal_available_cpp()))
    set.seed(20261116)
    data <- matrix(rnorm(32L * 6L), nrow = 32L)
    self <- faissR:::nn_metal_cpp(data, 4L, "exact")
    query <- faissR:::nn_metal_query_cpp(
        data, data[1:5, , drop = FALSE], 4L, "exact"
    )
    for (result in list(self, query)) {
        expect_identical(result$accelerator, "metal")
        expect_true(is.finite(result$metal_allocated_before_bytes))
        expect_true(is.finite(result$metal_allocated_after_search_bytes))
        expect_gte(result$metal_allocated_after_search_bytes,
                   result$metal_allocated_before_bytes)
        expect_gt(result$metal_live_buffer_bytes_after_search, 0)
        expect_identical(result$device_memory_measurement_scope,
            "process_current_allocated_after_search_before_output_release")
        expect_false(result$device_memory_peak_measured)
        expect_identical(result$index_build_seconds, 0)
        expect_identical(result$index_build_seconds_scope,
            "no_index_exact_search")
        expect_true(is.finite(result$search_seconds))
        expect_gte(result$search_seconds, 0)
    }
    ivf_self <- faissR:::nn_metal_cpp(
        data, 4L, "ivf", target_recall = 0.90
    )
    ivf_query <- faissR:::nn_metal_query_cpp(
        data, data[1:5, , drop = FALSE], 4L, "ivf",
        target_recall = 0.90
    )
    for (result in list(ivf_self, ivf_query)) {
        expect_identical(result$accelerator, "metal")
        expect_true(is.finite(result$metal_allocated_after_search_bytes))
        expect_gte(result$metal_allocated_after_search_bytes,
                   result$metal_allocated_before_bytes)
        expect_gt(result$metal_live_buffer_bytes_after_search, 0)
        expect_false(result$device_memory_peak_measured)
        expect_true(is.finite(result$index_build_seconds))
        expect_gt(result$index_build_seconds, 0)
        expect_identical(result$index_build_seconds_scope,
            "metal_projection_training_and_list_pack")
        expect_true(is.finite(result$search_seconds))
        expect_identical(result$search_seconds_scope, "metal_full_search")
    }
})
