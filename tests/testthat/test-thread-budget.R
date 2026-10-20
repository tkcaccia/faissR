test_that("one-thread searches preserve numerical results for both inputs", {
    skip_if_not(faiss_available())
    set.seed(25)
    x <- matrix(rnorm(120), 30, 4)
    for (metric in c("euclidean", "cosine", "correlation")) {
        reference <- nn(x, k = 3, method = "flat", backend = "cpu",
            metric = metric, n_threads = 1L)
        for (input in list(x, float::fl(x))) {
            actual <- nn(input, k = 3, method = "flat", backend = "cpu",
                metric = metric, n_threads = 1L)
            expect_equal(actual$indices, reference$indices)
            expect_equal(actual$distances, reference$distances,
                tolerance = 2e-6)
        }
    }
})

test_that("input preparation has an explicit bounded OpenMP budget", {
    root <- test_path("../../src")
    header <- file.path(root, "faissr_size_utils.hpp")
    if (!file.exists(header)) skip("Native source is not installed.")
    text <- readLines(header, warn = FALSE)
    pragmas <- grep("#pragma omp parallel", text, value = TRUE)
    expect_length(pragmas, 1L)
    expect_match(pragmas, "num_threads(n_threads", fixed = TRUE)
    expect_match(pragmas, "elements >= 65536", fixed = TRUE)
    for (name in c("nn_faiss_impl.cpp", "nn_cuvs_impl.cpp")) {
        text <- readLines(file.path(root, name), warn = FALSE)
        pragmas <- grep("#pragma omp parallel", text, value = TRUE)
        expect_true(all(grepl("num_threads(", pragmas, fixed = TRUE)))
    }
})

test_that("recall diagnostics do not request all machine cores", {
    set.seed(26)
    x <- matrix(rnorm(120), 30, 4)
    result <- nn(x, k = 3, backend = "cpu", n_threads = 1L,
        exclude_self = TRUE)
    compute <- faissR:::nn_compute
    requested <- NULL
    local_mocked_bindings(nn_compute = function(..., n_threads = NULL) {
        requested <<- n_threads
        compute(..., n_threads = n_threads)
    }, .package = "faissR")
    actual <- faissR:::attach_knn_recall_subset(result, x, 3L, TRUE, 26L)
    expect_identical(requested, 1L)
    expect_equal(attr(actual, "recall")$recall_at_k, 1)
})
