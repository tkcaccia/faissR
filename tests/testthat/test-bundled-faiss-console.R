test_that("bundled FAISS diagnostics use R's console", {
    skip_if_not(identical(Sys.getenv("FAISSR_FAISS_SOURCE"), "bundled"))
    skip_if_not(faiss_available())
    set.seed(123)
    x <- matrix(rnorm(80 * 4), 80, 4)
    text <- capture.output(
        result <- nn(x, k = 5, method = "ivf", backend = "cpu", n_threads = 2),
        type = "message"
    )
    expect_match(paste(text, collapse = "\n"), "clustering")
    expect_identical(dim(result$indices), c(80L, 5L))
    expect_true(all(is.finite(result$distances)))
})
