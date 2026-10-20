test_that("normalization preserves distances across finite input scales", {
    skip_if_not(faiss_available())
    set.seed(44)
    x <- matrix(rnorm(120), 30, 4)
    for (metric in c("cosine", "correlation")) {
        for (method in c("flat", "hnsw", "nndescent_style")) {
            search <- function(z) {
                nn(
                    z,
                    k = 3,
                    backend = "cpu",
                    method = method,
                    metric = metric,
                    exclude_self = TRUE,
                    n_threads = 1
                )
            }
            expected <- search(x)
            for (scale in c(1e-300, 1e-40, 1e40, 1e160, 1e300)) {
                actual <- search(x * scale)
                expect_true(all(is.finite(actual$distances)))
                expect_equal(actual$indices, expected$indices)
                expect_equal(
                    actual$distances,
                    expected$distances,
                    tolerance = 2e-6
                )
            }
        }
    }
})

test_that("scaled normalization retains genuine degenerate rows", {
    skip_if_not(faiss_available())
    set.seed(45)
    for (metric in c("cosine", "correlation")) {
        x <- rbind(matrix(rnorm(24), 6, 4), rep(0, 4), rep(0, 4))
        if (metric == "correlation") {
            x[7:8, ] <- 2
        }
        expected <- nn(
            x,
            k = 3,
            method = "flat",
            metric = metric,
            backend = "cpu",
            exclude_self = TRUE
        )
        actual <- nn(
            x * 1e160,
            k = 3,
            method = "flat",
            metric = metric,
            backend = "cpu",
            exclude_self = TRUE
        )
        expect_equal(actual$distances, expected$distances, tolerance = 2e-6)
        expect_true(all(is.finite(actual$distances)))
    }
})

test_that("direct float32 normalization and conversion are checked", {
    skip_if_not(faiss_available())
    x <- matrix(c(1, 2, 4, 1, -2, 3, -1, 3, 2), 3, 3)
    for (metric in c("cosine", "correlation")) {
        expected <- nn_faiss_flat_float32_cpp(
            x,
            x,
            2L,
            TRUE,
            1L,
            metric,
            "double"
        )
        for (z in list(x * 1e300, x * 1e-300, float::fl(x * 1e-40))) {
            actual <- nn_faiss_flat_float32_cpp(
                z,
                z,
                2L,
                TRUE,
                1L,
                metric,
                "double"
            )
            expect_equal(actual$indices, expected$indices)
            expect_equal(actual$distances, expected$distances, tolerance = 5e-5)
        }
    }
    expect_error(
        nn(x * 1e40, k = 2, method = "flat", backend = "cpu"),
        "finite values.*float32"
    )
})

test_that("fitted IVF predictions honor an explicitly changed recall tier", {
    skip_if_not(faiss_available())
    set.seed(501)
    x <- matrix(rnorm(50000 * 4), 50000, 4)
    fit <- knn(
        x,
        x[, 1],
        method = "ivf",
        backend = "cpu",
        k = 15,
        n_threads = 1,
        target_recall = 0.90
    )
    query <- x[1:20, , drop = FALSE]
    old <- attr(predict(fit, query, backend = "cpu"), "faissR_nn")
    new <- attr(
        predict(fit, query, backend = "cpu", target_recall = 0.99),
        "faissR_nn"
    )
    expected <- faiss_ivf_params(
        50000,
        15,
        metric = "euclidean",
        p = 4,
        target_recall = 0.99
    )
    expect_equal(old$query_source, "fitted_index")
    expect_equal(new$query_source, "nn")
    expect_equal(new$target_recall, 0.99)
    expect_equal(new$approximation$nprobe, expected$nprobe)
    expect_equal(fit$target_recall, 0.90)
    for (stored in c("faiss_ivf", "faiss_ivfpq")) {
        expect_false(knn_fitted_faiss_settings_match(
            fit,
            stored,
            15,
            "cpu",
            "auto",
            0.99
        ))
    }
})

test_that("CUDA candidate scoring keeps insufficient candidate sets restricted", {
    skip_if_not(cuda_available())
    x <- matrix(c(1, 2, 4, 8, 4, 1, 2, 3, 1, 3, 2, 5), 4, 3)
    candidates <- rbind(
        c(2L, 2L, NA, 0L),
        c(1L, 99L, 1L, 0L),
        c(NA, NA, 0L, -1L),
        c(1L, 4L, 4L, NA)
    )
    for (metric in c("euclidean", "cosine", "correlation")) {
        for (k in c(2L, 4L)) {
            cpu <- candidate_knn(
                x,
                candidates,
                k = k,
                backend = "cpu",
                metric = metric,
                exclude_self = TRUE
            )
            gpu <- candidate_knn(
                x,
                candidates,
                k = k,
                backend = "cuda",
                metric = metric,
                exclude_self = TRUE
            )
            expect_equal(gpu$indices, cpu$indices)
            expect_equal(gpu$distances, cpu$distances, tolerance = 2e-6)
        }
    }
    x[1, ] <- 0
    expect_error(
        candidate_knn(
            x,
            candidates,
            k = 2,
            backend = "cuda",
            metric = "cosine",
            exclude_self = TRUE
        ),
        "all-zero"
    )
})

test_that("GPU-resident normalization accepts finite extreme scales", {
    skip_if_not(cuda_available())
    x <- matrix(c(1, 2, 4, 8, 4, 1, 2, 3, 1, 3, 2, 5), 4, 3)
    for (metric in c("cosine", "correlation")) {
        expected <- nn(
            x,
            k = 2,
            method = "flat",
            backend = "cpu",
            metric = metric,
            exclude_self = TRUE
        )
        for (scale in c(1e-300, 1e300)) {
            actual <- gpu_knn_to_host(nn_gpu(
                x * scale,
                k = 2,
                method = "exact",
                metric = metric,
                exclude_self = TRUE
            ))
            expect_equal(actual$indices, expected$indices)
            expect_equal(actual$distances, expected$distances, tolerance = 2e-6)
        }
    }
})

test_that("matrix size helpers check bounds without allocating matrices", {
    header <- test_path("../../src/faissr_size_utils.hpp")
    if (!file.exists(header)) {
        skip("Native header requires source-tree tests")
    }
    Rcpp::cppFunction(
        '
        bool checked_matrix_bounds() {
            try {
                faissr::matrix_element_count(INT_MAX, INT_MAX);
            } catch (const std::length_error&) {
                return faissr::matrix_element_count(10, 20) == 200 &&
                    faissr::matrix_element_count_wide(INT_MAX, 2) == 4294967294LL;
            }
            return false;
        }',
        includes = paste0('#include "', normalizePath(header), '"')
    )
    expect_true(checked_matrix_bounds())
})
