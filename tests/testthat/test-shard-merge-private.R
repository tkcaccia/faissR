test_that("private shard merge preserves exact IDs and distances", {
    skip_if_not_installed("Rcpp")
    expect_false("faissR_merge_knn_shard_v1" %in%
        getNamespaceExports("faissR"))
    header <- system.file("include", "faissR_api_v1.h",
        package = "faissR")
    Rcpp::cppFunction(
        code = '
        SEXP faissR_test_merge_shard(SEXP arguments) {
          return faissR_get_merge_knn_shard_v1()(arguments);
        }',
        includes = paste0('#include "', header, '"')
    )
    paths <- paste0(tempfile(), c(".indices.u32", ".distances.f32"))
    on.exit(unlink(paths), add = TRUE)
    first <- list(indices_path = paths[[1L]],
        distances_path = paths[[2L]],
        indices = matrix(c(1L, 2L, 2L, 1L), nrow = 2L,
            byrow = TRUE),
        distances = matrix(c(0, 5, 0, 4), nrow = 2L,
            byrow = TRUE),
        query_first = 1, shard_first = 1, shard_rows = 2,
        n_vertices = 4, k = 2L, first_shard = TRUE,
        n_queries = 2, exclude_self = FALSE)
    identity <- faissR_test_merge_shard(first)
    expect_identical(identity$provider, "faissR")
    expect_identical(identity$method, "shard_merge")
    second <- first
    second$indices <- matrix(c(1L, 2L, 1L, 2L), nrow = 2L,
        byrow = TRUE)
    second$distances <- matrix(c(2, 3, 2, 3), nrow = 2L,
        byrow = TRUE)
    second$shard_first <- 3
    second$first_shard <- FALSE
    faissR_test_merge_shard(second)
    ids <- matrix(readBin(paths[[1L]], integer(), n = 4L,
        size = 4L, endian = "little"), nrow = 2L, byrow = TRUE)
    distances <- matrix(readBin(paths[[2L]], numeric(), n = 4L,
        size = 4L, endian = "little"), nrow = 2L, byrow = TRUE)
    expect_equal(ids, rbind(c(1L, 3L), c(2L, 3L)))
    expect_equal(distances, rbind(c(0, 2), c(0, 2)))
    bad <- second
    bad$indices[1L, 1L] <- 3L
    expect_error(faissR_test_merge_shard(bad),
        "invalid neighbor")
    bad <- second
    bad$n_queries <- 3
    expect_error(faissR_test_merge_shard(bad),
        "unexpected size")
    self_paths <- paste0(tempfile(),
        c(".indices.u32", ".distances.f32"))
    on.exit(unlink(self_paths), add = TRUE)
    self_first <- first
    self_first$indices_path <- self_paths[[1L]]
    self_first$distances_path <- self_paths[[2L]]
    self_first$exclude_self <- TRUE
    faissR_test_merge_shard(self_first)
    self_second <- second
    self_second$indices_path <- self_paths[[1L]]
    self_second$distances_path <- self_paths[[2L]]
    self_second$exclude_self <- TRUE
    faissR_test_merge_shard(self_second)
    self_ids <- matrix(readBin(self_paths[[1L]], integer(), n = 4L,
        size = 4L, endian = "little"), nrow = 2L, byrow = TRUE)
    expect_equal(self_ids, rbind(c(3L, 4L), c(3L, 4L)))
})
