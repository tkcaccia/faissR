test_that("private file-backed CPU KNN agrees with exact FAISS", {
    skip_if_not_installed("Rcpp")
    skip_if_not(faiss_available(), "FAISS is required")
    expect_false("faissR_file_knn_v1" %in% getNamespaceExports("faissR"))
    header <- system.file("include", "faissR_api_v1.h", package = "faissR")
    Rcpp::cppFunction(
        code = '
        SEXP faissR_test_file_knn(SEXP query, SEXP reference,
                                  int k, std::string output,
                                  std::string backend, double memory_limit) {
          faissR_file_knn_v1_fun fn = faissR_get_file_knn_v1();
          return fn(query, reference, Rcpp::wrap(k), Rcpp::wrap(output),
            Rcpp::wrap(backend), Rcpp::wrap("stream_exact"),
            Rcpp::wrap(2), Rcpp::wrap(3), Rcpp::wrap(memory_limit),
            Rcpp::wrap(1));
        }',
        includes = paste0('#include "', header, '"')
    )
    x <- matrix(c(0, 0, 1, 2, 3, 1, 5, 4, 7, 2, 9, 8),
        ncol = 2L, byrow = TRUE)
    query <- matrix(c(0.5, 0.3, 6, 3), ncol = 2L, byrow = TRUE)
    directory <- tempfile("faissr-file-knn-")
    dir.create(directory)
    on.exit(unlink(directory, recursive = TRUE), add = TRUE)
    data_path <- file.path(directory, "reference.f32")
    query_path <- file.path(directory, "query.f32")
    writeBin(as.numeric(t(x)), data_path, size = 4L, endian = "little")
    writeBin(as.numeric(t(query)), query_path,
        size = 4L, endian = "little")
    reference_spec <- list(format = "f32", path = data_path,
        nrow = nrow(x), ncol = ncol(x))
    query_spec <- list(format = "f32", path = query_path,
        nrow = nrow(query), ncol = ncol(query))
    prefix <- file.path(directory, "result")
    observed <- faissR_test_file_knn(query_spec, reference_spec, 3L,
        prefix, "cpu", 128 * 1024^2)
    ids <- matrix(readBin(observed$indices_path, integer(), 6L,
        size = 4L, endian = "little"),
        nrow = 2L, byrow = TRUE)
    distances <- matrix(readBin(observed$distances_path, numeric(), 6L,
        size = 4L, endian = "little"), nrow = 2L, byrow = TRUE)
    expected <- nn(x, query, k = 3L, backend = "cpu",
        method = "flat", n_threads = 1L)
    expect_equal(ids, expected$indices)
    expect_equal(distances, expected$distances, tolerance = 1e-5)
    expect_identical(observed$backend_used, "faiss_cpu_stream_exact")
    expect_true(observed$exact)
    fbin_path <- file.path(directory, "reference.fbin")
    connection <- file(fbin_path, "wb")
    writeBin(as.integer(dim(x)), connection, size = 4L,
        endian = "little")
    writeBin(as.numeric(t(x)), connection, size = 4L,
        endian = "little")
    close(connection)
    fbin_spec <- list(format = "fbin", path = fbin_path,
        nrow = nrow(x), ncol = ncol(x))
    view_spec <- list(format = "view", source = query_spec,
        first = 2L, nrow = 1L, ncol = ncol(query))
    view <- faissR_test_file_knn(view_spec, fbin_spec, 3L,
        file.path(directory, "view"), "cpu", 128 * 1024^2)
    view_ids <- readBin(view$indices_path, integer(), 3L,
        size = 4L, endian = "little")
    expect_equal(view_ids, as.integer(expected$indices[2L, ]))
    expect_error(faissR_test_file_knn(query_spec, reference_spec, 3L,
        file.path(directory, "cuda"), "cuda", 128 * 1024^2),
        "unavailable")
    expect_error(faissR_test_file_knn(query_spec, reference_spec, 3L,
        file.path(directory, "low-memory"), "cpu", 1),
        "memory_limit")
})
