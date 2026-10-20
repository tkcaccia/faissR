test_that("search entry points reject implicit sparse materialization", {
    skip_if_not_installed("Matrix")
    sparse <- Matrix::Matrix(diag(4), sparse = TRUE)
    expected <- "not converted implicitly"

    expect_error(nn(sparse, k = 1L, n_threads = 1L), expected)
    expect_error(nn_gpu(sparse, k = 1L), expected)
    expect_error(nn_metric_preflight(sparse), expected)
    expect_error(fast_kmeans(sparse, centers = 2L, n_threads = 1L), expected)
    expect_error(
        candidate_knn(sparse, matrix(1:4, ncol = 1L), k = 1L, n_threads = 1L),
        expected
    )
    expect_error(knn(sparse, factor(rep(c("a", "b"), 2L)), k = 1L, n_threads = 1L), expected)
})

test_that("dense numeric matrices and data frames remain accepted", {
    x <- matrix(seq_len(12), nrow = 4L)
    expect_invisible(faissR:::validate_dense_matrix_input(x, "x"))
    expect_invisible(
        faissR:::validate_dense_matrix_input(as.data.frame(x), "x")
    )
    expect_error(
        faissR:::validate_dense_matrix_input(
            data.frame(a = 1:2, b = letters[1:2]),
            "x"
        ),
        "Every column"
    )
})
