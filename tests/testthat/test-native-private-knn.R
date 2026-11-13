test_that("private Metal KNN has a truthful capability gate", {
    available <- faissR:::nn_metal_available_cpp()
    expect_type(available, "logical")
    x <- matrix(seq_len(80L), nrow = 20L)
    if (!isTRUE(available)) {
        expect_error(
            faissR:::nn_metal_cpp(x, 4L, "exact"),
            "Metal-enabled macOS"
        )
    } else {
        observed <- faissR:::nn_metal_cpp(x, 4L, "exact")
        reference <- faissR::nn(
            x, k = 4L, exclude_self = TRUE,
            backend = "cpu", method = "flat", n_threads = 1L
        )
        expect_equal(observed$indices, reference$indices)
        expect_equal(observed$distances, reference$distances,
                     tolerance = 1e-4)
        expect_identical(observed$backend_used,
                         "faissr_native_metal_exact")
        expect_identical(observed$result_residency, "host")
        expect_identical(observed$index_base, 1L)
        query <- faissR:::nn_metal_query_cpp(
            x, x[1:3, , drop = FALSE], 4L, "exact"
        )
        expect_identical(dim(query$indices), c(3L, 4L))
        expect_identical(query$backend_used,
                         "faissr_native_metal_exact_query")
    }
})

test_that("private multi-GPU KNN rejects unsupported device requests", {
    x <- matrix(seq_len(80L), nrow = 20L)
    expect_error(
        faissR:::nn_faiss_gpu_multi_cpp(
            x, 4L, "ivf", c(0L, 0L), 4L, 2L
        ),
        "multi-GPU|device|unavailable"
    )
    expect_error(
        faissR:::nn_faiss_gpu_multi_query_cpp(
            x, x[1:3, , drop = FALSE], 4L, "ivf",
            c(0L, 0L), 4L, 2L
        ),
        "multi-GPU|device|unavailable"
    )
})

test_that("private CUDA fixed-query KNN matches the exact CPU reference", {
    skip_if_not(isTRUE(faissR::faiss_gpu_available()))
    set.seed(611L)
    x <- matrix(rnorm(100L * 8L), nrow = 100L)
    query <- x[c(2L, 13L, 71L), , drop = FALSE]
    observed <- faissR:::nn_faiss_gpu_multi_query_cpp(
        x, query, 6L, "flat", 0L, 1L, 1L,
        distance_storage = "double"
    )
    reference <- faissR::nn(
        x, query, k = 6L, backend = "cpu", method = "flat",
        n_threads = 1L
    )
    expect_equal(observed$indices, reference$indices)
    expect_equal(observed$distances, reference$distances,
                 tolerance = 1e-4)
    expect_identical(observed$query_count, 3L)
    expect_identical(observed$multi_gpu_mode, "single")
    self <- faissR:::nn_faiss_gpu_multi_cpp(
        x, 6L, "flat", 0L, 1L, 1L,
        distance_storage = "double"
    )
    self_reference <- faissR::nn(
        x, k = 6L, exclude_self = TRUE,
        backend = "cpu", method = "flat", n_threads = 1L
    )
    expect_equal(self$indices, self_reference$indices)
    expect_equal(self$distances, self_reference$distances,
                 tolerance = 1e-4)
    expect_identical(self$backend_used, "faiss_gpu_single_self")
})
