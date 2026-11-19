test_that("private file-backed Metal IVF reads float32 once", {
    skip_if_not(faissR:::nn_metal_available_cpp(), "Metal is required")
    skip_if_not_installed("Rcpp")
    expect_false("faissR_metal_file_ivf_v1" %in%
        getNamespaceExports("faissR"))
    header <- system.file("include", "faissR_api_v1.h",
        package = "faissR")
    Rcpp::cppFunction(
        code = '
        SEXP faissR_test_metal_file_ivf(SEXP source, int k,
             double target, double limit) {
          faissR_metal_file_ivf_v1_fun fn =
            faissR_get_metal_file_ivf_v1();
          return fn(source, Rcpp::wrap(k), Rcpp::wrap(target),
            Rcpp::wrap(limit));
        }',
        includes = paste0('#include "', header, '"')
    )
    Rcpp::cppFunction(
        code = '
        SEXP faissR_test_metal_file_plan(SEXP source, int k,
             double limit) {
          faissR_metal_file_ivf_plan_v1_fun fn =
            faissR_get_metal_file_ivf_plan_v1();
          return fn(source, Rcpp::wrap(k), Rcpp::wrap(limit));
        }',
        includes = paste0('#include "', header, '"')
    )
    set.seed(42)
    x <- matrix(rnorm(4096L * 3L), ncol = 3L)
    path <- tempfile(fileext = ".f32")
    on.exit(unlink(path), add = TRUE)
    writeBin(as.numeric(t(x)), path, size = 4L, endian = "little")
    source <- list(format = "f32", path = path,
        nrow = nrow(x), ncol = ncol(x))
    expect_false(faissR_test_metal_file_plan(source, 5L, 1)$fits)
    expect_true(faissR_test_metal_file_plan(
        source, 5L, 512 * 1024^2)$fits)
    expect_error(faissR_test_metal_file_ivf(
        source, 5L, 0.9, 1), "budget")
    found <- faissR_test_metal_file_ivf(
        source, 5L, 0.9, 512 * 1024^2)
    expect_identical(found$backend_used,
        "faissr_native_metal_ivf_file")
    expect_identical(found$index_residency,
        "transient_metal_buffers")
    expect_false(found$exact)
    expect_true(found$target_met)
    expect_identical(dim(found$indices), c(4096L, 5L))
    expect_identical(dim(found$distances), c(4096L, 5L))
    expect_gt(found$source_read_seconds, 0)
    expect_gt(found$metal_live_buffer_bytes_after_search, 0)
    rows <- seq.int(1L, nrow(x), length.out = 32L)
    recall <- vapply(as.integer(rows), function(row) {
        distance <- rowSums(sweep(x, 2L, x[row, ])^2)
        distance[row] <- Inf
        truth <- order(distance)[seq_len(5L)]
        mean(found$indices[row, ] %in% truth)
    }, numeric(1L))
    expect_gte(mean(recall), 0.9)
})
