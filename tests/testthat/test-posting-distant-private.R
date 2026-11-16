test_that("private posting audit selects distant rows exactly", {
    skip_if_not_installed("Rcpp")
    expect_false("faissR_posting_distant_rows_v1" %in%
        getNamespaceExports("faissR"))
    header <- system.file("include", "faissR_api_v1.h",
        package = "faissR")
    Rcpp::cppFunction(
        code = '
        SEXP faissR_test_posting_distant(SEXP arguments) {
          return faissR_get_posting_distant_rows_v1()(arguments);
        }',
        includes = paste0('#include "', header, '"')
    )
    x <- cbind(c(0, 1, 4, 5, 10, 11, 14, 15), rep(0, 8L))
    centers <- rbind(c(0, 0), c(10, 0))
    path <- tempfile(fileext = ".f32")
    on.exit(unlink(path), add = TRUE)
    writeBin(as.numeric(t(x)), path, size = 4L, endian = "little")
    args <- list(spec = list(format = "f32", path = path,
        nrow = nrow(x), ncol = ncol(x)), centers = centers,
        sample = 3L, chunk_rows = 3L, workers = 2L,
        candidate_rows = 0L)
    observed <- faissR_test_posting_distant(args)
    distances <- vapply(seq_len(nrow(x)), function(i) {
        min(rowSums((centers - matrix(x[i, ], 2L, 2L,
            byrow = TRUE))^2))
    }, numeric(1L))
    expected <- order(distances, seq_along(distances),
        decreasing = TRUE)[seq_len(args$sample)]
    expect_equal(observed$indices, expected)
    expect_identical(observed$provider, "faissR")
    expect_identical(observed$backend_used,
        "faissr_cpu_posting_audit")
    expect_identical(observed$method, "distant_center_rows")
    sampled <- args
    sampled$sample <- 2L
    sampled$candidate_rows <- 4L
    expect_equal(faissR_test_posting_distant(sampled)$indices,
        c(8L, 4L))
    synthetic <- sampled
    synthetic$spec <- list(format = "synthetic", nrow = 1e9,
        ncol = 2L)
    synthetic$sample <- 4L
    synthetic$candidate_rows <- 16L
    candidates <- floor(((seq_len(16L) - 1L) * 2 + 1) *
        1e9 / 32) + 1
    values <- cbind((candidates - 1) %% 10007 / 10007,
        (candidates - 1) %% 10007 / 10007 + 0.01)
    distances <- vapply(seq_len(16L), function(i) {
        min(rowSums((centers - matrix(values[i, ], 2L, 2L,
            byrow = TRUE))^2))
    }, numeric(1L))
    ranked <- order(distances, candidates, decreasing = TRUE)
    expect_equal(faissR_test_posting_distant(synthetic)$indices,
        as.integer(candidates[ranked[seq_len(4L)]]))
    bad <- args
    bad$sample <- 9L
    expect_error(faissR_test_posting_distant(bad),
        "Posting audit dimensions")
    bad <- args
    bad$centers[1L, 1L] <- Inf
    expect_error(faissR_test_posting_distant(bad),
        "finite float32")
    bad <- args
    bad$spec <- list(format = "memory", data = x,
        nrow = nrow(x), ncol = ncol(x))
    expect_error(faissR_test_posting_distant(bad),
        "Posting audit dimensions")
})
