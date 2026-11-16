test_that("resident CPU FAISS reports actual index and search boundaries", {
    skip_if_not(faiss_available())
    set.seed(4)
    data <- matrix(rnorm(600), nrow = 100)
    for (method in c("flat", "hnsw")) {
        result <- nn(
            data, k = 5L, backend = "cpu", method = method,
            exclude_self = TRUE, n_threads = 1L
        )
        expect_identical(dim(result$indices), c(100L, 5L))
        expect_true(all(is.finite(result$distances)))
        expect_true(is.finite(result$index_build_seconds))
        expect_gte(result$index_build_seconds, 0)
        expect_true(is.finite(result$search_seconds))
        expect_gte(result$search_seconds, 0)
        expect_identical(result$index_build_seconds_scope,
            "fitted_index_build_call_including_marshalling")
        expect_identical(result$search_seconds_scope,
            "fitted_index_search_call_including_marshalling")
        cached <- nn(
            data, k = 5L, backend = "cpu", method = method,
            exclude_self = TRUE, n_threads = 1L
        )
        expect_identical(cached$index_build_seconds, 0)
        expect_identical(cached$index_build_seconds_scope,
            "cached_index_reused")
        expect_equal(cached$indices, result$indices)
    }
})

test_that("float32 resident Flat search retains stage boundaries", {
    skip_if_not(faiss_available())
    set.seed(5)
    data <- float::fl(matrix(rnorm(240), nrow = 40))
    result <- nn(
        data, k = 3L, backend = "cpu", method = "flat",
        exclude_self = TRUE, n_threads = 1L
    )
    expect_identical(dim(result$indices), c(40L, 3L))
    expect_true(is.finite(result$index_build_seconds))
    expect_true(is.finite(result$search_seconds))
    expect_identical(result$index_build_seconds_scope,
        "fitted_index_build_call_including_marshalling")
    expect_identical(result$search_seconds_scope,
        "fitted_index_search_call_including_marshalling")
})

test_that("one-shot native Flat search measures train/add and query search", {
    skip_if_not(faiss_available())
    set.seed(6)
    data <- matrix(rnorm(120), nrow = 20)
    direct <- getFromNamespace("nn_faiss_flat_cpp", "faissR")(
        data, data, 3L, TRUE, 1L
    )
    expect_identical(dim(direct$indices), c(20L, 3L))
    expect_true(is.finite(direct$index_build_seconds))
    expect_true(is.finite(direct$search_seconds))
    expect_identical(direct$index_build_seconds_scope,
        "faiss_index_train_add")
    expect_identical(direct$search_seconds_scope, "faiss_index_search")
})

test_that("uncached public Flat search exposes native stage times", {
    skip_if_not(faiss_available())
    old <- options(faissR.cache_fitted_nn_indexes = FALSE)
    on.exit(options(old), add = TRUE)
    set.seed(7)
    data <- matrix(rnorm(150), nrow = 25)
    result <- nn(
        data, k = 3L, backend = "cpu", method = "flat",
        exclude_self = TRUE, n_threads = 1L
    )
    expect_true(is.finite(result$index_build_seconds))
    expect_true(is.finite(result$search_seconds))
    expect_identical(result$index_build_seconds_scope,
        "faiss_index_train_add")
    expect_identical(result$search_seconds_scope,
        "faiss_index_search")
})

test_that("opt-in public resource samples state their measurement scope", {
    skip_if_not(faiss_available())
    old <- options(
        faissR.measure_resources = TRUE,
        faissR.measure_device_memory = FALSE
    )
    on.exit(options(old), add = TRUE)
    data <- matrix(seq_len(120) / 13, nrow = 20)
    result <- nn(
        data, k = 3L, backend = "cpu", method = "flat",
        exclude_self = TRUE, n_threads = 1L
    )
    memory <- result$resource_measurement
    expect_identical(memory$scope, "process_before_after_public_nn_call")
    expect_false(memory$peak_during_call_measured)
    expect_true(is.finite(memory$before$process_resident_memory_bytes))
    expect_true(is.finite(memory$after$process_resident_memory_bytes))
    expect_null(memory$before$device_sample)
    expect_null(memory$after$device_sample)
})
