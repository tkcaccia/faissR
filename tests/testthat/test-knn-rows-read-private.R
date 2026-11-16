test_that("private KNN row reader checks IDs, bounds and files", {
    skip_if_not_installed("Rcpp")
    expect_false("faissR_knn_rows_read_v1" %in%
        getNamespaceExports("faissR"))
    header <- system.file("include", "faissR_api_v1.h",
        package = "faissR")
    Rcpp::cppFunction(
        code = '
        SEXP faissR_test_knn_rows_read(SEXP arguments) {
          return faissR_get_knn_rows_read_v1()(arguments);
        }',
        includes = paste0('#include "', header, '"')
    )
    indices <- rbind(c(2L, 3L), c(1L, 3L), c(2L, 1L))
    distances <- rbind(c(1, 2), c(1.5, 2.5), c(3, 4))
    ids_path <- tempfile(fileext = ".u32")
    distances_path <- tempfile(fileext = ".f32")
    on.exit(unlink(c(ids_path, distances_path)), add = TRUE)
    writeBin(as.integer(t(indices)), ids_path, size = 4L,
        endian = "little")
    writeBin(as.numeric(t(distances)), distances_path, size = 4L,
        endian = "little")
    args <- list(indices_path = ids_path,
        distances_path = distances_path, n_rows = 3,
        k = 2L, n_reference = 3, first_row = 2,
        count = 2L, max_bytes = 1000)
    observed <- faissR_test_knn_rows_read(args)
    expect_identical(observed$provider, "faissR")
    expect_identical(observed$backend_used,
        "faissr_cpu_knn_rows_read")
    expect_identical(observed$method, "bounded_file_rows")
    expect_identical(observed$indices, indices[2:3, , drop = FALSE])
    expect_equal(observed$distances,
        distances[2:3, , drop = FALSE], tolerance = 1e-6)
    args$max_bytes <- 1
    expect_error(faissR_test_knn_rows_read(args), "byte limit")
    args$max_bytes <- 1000
    args$first_row <- 3
    expect_error(faissR_test_knn_rows_read(args), "range")
    args$first_row <- 2
    writeBin(as.integer(c(2, 3, 0, 1, 3, 2)), ids_path,
        size = 4L, endian = "little")
    expect_error(faissR_test_knn_rows_read(args), "invalid values")
    writeBin(as.integer(t(indices)), ids_path, size = 4L,
        endian = "little")
    writeBin(as.numeric(1), distances_path, size = 4L,
        endian = "little")
    expect_error(faissR_test_knn_rows_read(args), "size mismatch")
})
