test_that("private file-backed HNSW build streams and searches correctly", {
    skip_if_not_installed("Rcpp")
    skip_if_not_installed("float")
    skip_if_not(faiss_available(), "FAISS is required")
    expect_false("faissR_hnsw_file_index_build_v1" %in%
        getNamespaceExports("faissR"))
    header <- system.file("include", "faissR_api_v1.h", package = "faissR")
    Rcpp::cppFunction(code = '
        SEXP faissR_test_file_hnsw_build(SEXP spec, int m,
            int ef_construction, int threads, int chunk_rows) {
          return faissR_get_hnsw_file_index_build_v1()(
            spec, Rcpp::wrap(m), Rcpp::wrap(ef_construction),
            Rcpp::wrap(threads), Rcpp::wrap(chunk_rows));
        }', includes = paste0('#include "', header, '"'))
    Rcpp::cppFunction(code = '
        SEXP faissR_test_file_hnsw_search(SEXP index, SEXP query,
            int k, int ef_search) {
          return faissR_get_hnsw_index_search_v1()(
            index, query, Rcpp::wrap(k), Rcpp::wrap(ef_search),
            Rcpp::wrap(2));
        }', includes = paste0('#include "', header, '"'))
    set.seed(20261116)
    x <- matrix(rnorm(300L * 8L), nrow = 300L)
    directory <- tempfile("faissr-hnsw-file-")
    dir.create(directory)
    on.exit(unlink(directory, recursive = TRUE), add = TRUE)
    f32 <- file.path(directory, "sample.f32")
    fbin <- file.path(directory, "sample.fbin")
    values <- as.numeric(t(x))
    writeBin(values, f32, size = 4L, endian = "little")
    con <- file(fbin, "wb")
    writeBin(as.integer(c(nrow(x), ncol(x))), con,
             size = 4L, endian = "little")
    writeBin(values, con, size = 4L, endian = "little")
    close(con)
    raw_spec <- list(format = "f32", path = f32,
                     nrow = nrow(x), ncol = ncol(x))
    fbin_spec <- list(format = "fbin", path = fbin,
                      nrow = nrow(x), ncol = ncol(x))
    index <- faissR_test_file_hnsw_build(raw_spec, 16L, 80L, 1L, 13L)
    index_fbin <- faissR_test_file_hnsw_build(
        fbin_spec, 16L, 80L, 1L, 300L)
    expect_identical(attr(index, "residency"),
                     "file_backed_reference_resident_index")
    expect_identical(attr(index, "backend_used"),
                     "faiss_cpu_hnsw_file_index")
    expect_identical(attr(index, "n"), 300L)
    expect_identical(attr(index, "p"), 8L)
    expect_identical(attr(index, "reference_chunk_rows"), 13L)
    expect_identical(attr(index, "reference_add_batches"), 1L)
    expect_identical(attr(index_fbin, "reference_chunk_rows"), 300L)
    expect_identical(attr(index_fbin, "reference_add_batches"), 1L)
    two_reads <- faissR_test_file_hnsw_build(raw_spec, 16L, 80L, 1L, 151L)
    expect_identical(attr(two_reads, "reference_add_batches"), 2L)
    expect_gte(attr(index, "reference_read_seconds"), 0)
    expect_identical(attr(index, "reference_read_scope"),
                     "buffered_file_read_plus_finite_validation")
    expect_gt(attr(index, "index_build_seconds"), 0)
    rows <- c(2L, 31L, 116L, 251L)
    query <- float::fl(x[rows, , drop = FALSE])
    out <- faissR_test_file_hnsw_search(index, query, 5L, 300L)
    out_fbin <- faissR_test_file_hnsw_search(
        index_fbin, query, 5L, 300L)
    expect_identical(dim(out$indices), c(4L, 5L))
    expect_identical(out$provider, "faissR")
    expect_identical(out$backend_used, "faiss_cpu_hnsw_file_index")
    expect_identical(out$reference_residency, "file_backed")
    expect_identical(out$index_residency, "resident_cpu")
    expect_length(out$device_ids, 0L)
    expect_gte(out$search_seconds, 0)
    expect_gt(out$index_build_seconds, 0)
    expect_identical(out$indices, out_fbin$indices)
    expect_equal(out$distances, out_fbin$distances,
                 tolerance = 1e-6)
    low_effort <- faissR_test_file_hnsw_search(
        index, float::fl(x), 5L, 8L)
    low_effort_fbin <- faissR_test_file_hnsw_search(
        index_fbin, float::fl(x), 5L, 8L)
    expect_identical(low_effort$indices, low_effort_fbin$indices)
    expect_identical(low_effort$distances, low_effort_fbin$distances)
    for (i in seq_along(rows)) {
        distance <- sqrt(rowSums(sweep(x, 2L, x[rows[[i]], ])^2))
        expected <- order(distance, seq_len(nrow(x)))[seq_len(5L)]
        expect_setequal(out$indices[i, ], expected)
        expect_equal(sort(out$distances[i, ]), sort(distance[expected]),
                     tolerance = 1e-4)
    }
    view <- list(format = "view", source = raw_spec,
                 first = 11L, nrow = 100L)
    viewed <- faissR_test_file_hnsw_build(view, 16L, 80L, 2L, 17L)
    expect_identical(attr(viewed, "n"), 100L)
    tiny_view <- within(view, nrow <- 7L)
    tiny <- faissR_test_file_hnsw_build(
        tiny_view, 16L, 80L, 2L, 7L)
    expect_identical(attr(tiny, "m"), 7L)
    expect_identical(attr(tiny, "requested_m"), 16L)
    expect_true(attr(tiny, "hnsw_parameters_adjusted"))
    tiny_found <- faissR_test_file_hnsw_search(
        tiny, float::fl(x[11L, , drop = FALSE]), 3L, 7L)
    expect_identical(tiny_found$m, 7L)
    expect_identical(tiny_found$requested_m, 16L)
    expect_error(faissR_test_file_hnsw_build(
        within(view, first <- 250L), 16L, 80L, 2L, 17L),
        "view exceeds")
    expect_error(faissR_test_file_hnsw_build(
        raw_spec, 16L, 80L, 2L, 0L), "Invalid HNSW")
    bad <- raw_spec
    bad$nrow <- 301L
    expect_error(faissR_test_file_hnsw_build(
        bad, 16L, 80L, 2L, 13L), "size differs")
    bad_header <- file.path(directory, "bad-header.fbin")
    con <- file(bad_header, "wb")
    writeBin(as.integer(c(299L, 8L)), con,
             size = 4L, endian = "little")
    writeBin(values, con, size = 4L, endian = "little")
    close(con)
    bad <- fbin_spec
    bad$path <- bad_header
    expect_error(faissR_test_file_hnsw_build(
        bad, 16L, 80L, 2L, 13L), "header differs")
    nonfinite <- file.path(directory, "nonfinite.f32")
    values[[1L]] <- NaN
    writeBin(values, nonfinite, size = 4L, endian = "little")
    bad <- raw_spec
    bad$path <- nonfinite
    expect_error(faissR_test_file_hnsw_build(
        bad, 16L, 80L, 2L, 13L), "nonfinite")
})
