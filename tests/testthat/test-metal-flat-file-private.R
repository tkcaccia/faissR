test_that("a native Metal Flat shard serves bounded file query views", {
    skip_if_not(faissR:::nn_metal_available_cpp(), "Metal is required")
    header <- system.file("include", "faissR_api_v1.h", package = "faissR")
    Rcpp::cppFunction(
        code = '
        SEXP metal_flat_build(SEXP source, double limit) {
          return faissR_get_metal_file_flat_build_v1()(
            source, Rcpp::wrap(3), Rcpp::wrap(3), Rcpp::wrap(limit));
        }',
        includes = paste0('#include "', header, '"')
    )
    Rcpp::cppFunction(
        code = '
        SEXP metal_flat_search(SEXP pointer, SEXP query) {
          return faissR_get_metal_file_flat_search_v1()(
            pointer, query, Rcpp::wrap(3));
        }',
        includes = paste0('#include "', header, '"')
    )
    set.seed(39)
    x <- matrix(rnorm(19L * 11L), nrow = 19L)
    path <- tempfile(fileext = ".f32")
    on.exit(unlink(path), add = TRUE)
    writeBin(as.numeric(t(x)), path, size = 4L, endian = "little")
    source <- list(path = path, format = "f32", nrow = 19L, ncol = 11L)
    expect_error(metal_flat_build(source, 1), "memory_limit")
    built <- metal_flat_build(source, 128 * 1024^2)
    expect_identical(built$backend_used, "faissr_native_metal_flat_index")
    expect_equal(built$reference_bytes, length(x) * 4)
    expect_gt(built$owned_buffer_bytes, 0)
    expect_error(metal_flat_search(built$pointer, source), "capacity")
    for (rows in list(4:6, 17:18)) {
        query <- list(format = "view", source = source, first = min(rows),
            nrow = length(rows), ncol = 11L)
        found <- metal_flat_search(built$pointer, query)
        expected <- t(vapply(rows, function(row) {
            order(rowSums(sweep(x, 2L, x[row, ], "-") ^ 2))[1:3]
        }, integer(3L)))
        expect_identical(found$indices, expected)
        expect_true(found$index_reused)
    }
    Sys.setFileTime(path, Sys.time() - 60)
    expect_error(metal_flat_search(built$pointer, query), "reference changed")
})
