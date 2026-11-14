test_that("private bounded KNN preserves ties and self exclusion", {
    skip_if_not_installed("Rcpp")
    skip_if_not(faiss_available(), "FAISS is required")
    expect_false("faissR_matrix_knn_batch_v1" %in%
        getNamespaceExports("faissR"))
    header <- system.file("include", "faissR_api_v1.h", package = "faissR")
    Rcpp::cppFunction(
        code = '
        SEXP faissR_test_file_batch(SEXP query, SEXP reference,
             SEXP rows, int k, std::string backend, bool exclude_self,
             double memory_limit) {
          faissR_matrix_knn_batch_v1_fun fn =
            faissR_get_matrix_knn_batch_v1();
          return fn(query, reference, rows, Rcpp::wrap(k),
            Rcpp::wrap(backend), Rcpp::wrap("stream_exact"),
            Rcpp::wrap(2), Rcpp::wrap(memory_limit), Rcpp::wrap(2),
            Rcpp::wrap(exclude_self));
        }',
        includes = paste0('#include "', header, '"')
    )
    x <- matrix(c(0, 0, 0, 0, 1, 0, 1, 0, 2, 0),
        ncol = 2L, byrow = TRUE)
    directory <- tempfile("faissr-file-batch-")
    dir.create(directory)
    on.exit(unlink(directory, recursive = TRUE), add = TRUE)
    path <- file.path(directory, "ties.f32")
    writeBin(as.numeric(t(x)), path, size = 4L, endian = "little")
    spec <- list(format = "f32", path = path, nrow = nrow(x),
        ncol = ncol(x))
    observed <- faissR_test_file_batch(spec, spec, c(1, 3, 5),
        3L, "cpu", TRUE, 128 * 1024^2)
    for (i in seq_along(c(1, 3, 5))) {
        row <- c(1, 3, 5)[i]
        distance <- sqrt(rowSums((sweep(x, 2L, x[row, ]))^2))
        distance[row] <- Inf
        expected <- order(distance, seq_len(nrow(x)))[seq_len(3L)]
        expect_equal(unname(observed$indices[i, ]), expected)
        expect_equal(unname(observed$distances[i, ]),
            distance[expected], tolerance = 1e-6)
    }
    expect_identical(observed$backend_used, "faiss_cpu_stream_exact")
    expect_identical(observed$provider, "faissR")
    expect_true(observed$exact)
    expect_equal(length(observed$device_ids), 0L)
    resident <- list(format = "memory", data = x,
        nrow = nrow(x), ncol = ncol(x))
    resident_result <- faissR_test_file_batch(resident, resident,
        c(1, 3, 5), 3L, "cpu", TRUE, 128 * 1024^2)
    expect_identical(resident_result$residency, "resident")
    expect_identical(resident_result$indices, observed$indices)
    expect_equal(resident_result$distances, observed$distances)
    mixed <- faissR_test_file_batch(resident, spec, 1, 3L,
        "cpu", FALSE, 128 * 1024^2)
    expect_identical(mixed$residency, "mixed")
    expect_equal(unname(mixed$indices[1L, ]), c(1L, 2L, 3L))
    fbin_path <- file.path(directory, "ties.fbin")
    connection <- file(fbin_path, "wb")
    writeBin(as.integer(dim(x)), connection, size = 4L,
        endian = "little")
    writeBin(as.numeric(t(x)), connection, size = 4L,
        endian = "little")
    close(connection)
    fbin <- list(format = "fbin", path = fbin_path,
        nrow = nrow(x), ncol = ncol(x))
    view <- list(format = "view", source = spec, first = 2L,
        nrow = 2L, ncol = ncol(x))
    viewed <- faissR_test_file_batch(view, fbin, c(1, 2), 3L,
        "cpu", FALSE, 128 * 1024^2)
    expect_identical(viewed$residency, "file-backed")
    expect_equal(unname(viewed$indices[1L, ]), c(1L, 2L, 3L))
    expect_equal(unname(viewed$indices[2L, ]), c(3L, 4L, 1L))
    expect_error(faissR_test_file_batch(spec, spec, 1, 2L,
        "cuda", TRUE, 128 * 1024^2), "unavailable")
    expect_error(faissR_test_file_batch(resident, spec, 1, 2L,
        "cpu", TRUE, 128 * 1024^2), "dimensions or settings")
    expect_error(faissR_test_file_batch(spec, spec, 6, 2L,
        "cpu", TRUE, 128 * 1024^2), "row_ids")
    expect_error(faissR_test_file_batch(spec, spec, 1, 2L,
        "cpu", TRUE, 1), "memory_limit")
})
