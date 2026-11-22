test_that("file CAGRA queries an external row view with shard-local IDs", {
    skip_if_not(isTRUE(cuvs_available()))
    skip_if_not(isTRUE(cuda_available()))
    set.seed(42L)
    reference <- matrix(runif(768L * 4L), ncol = 4L)
    queries <- reference[c(60L, 73L, 91L, 254L, 401L), , drop = FALSE]
    path <- tempfile(fileext = ".fbin")
    con <- file(path, "wb")
    writeBin(c(nrow(reference), ncol(reference)), con, endian = "little")
    writeBin(as.numeric(t(reference)), con, size = 4L, endian = "little")
    close(con)
    on.exit(unlink(path))
    source <- list(format = "fbin", path = path, nrow = 768L, ncol = 4L)
    shard <- list(format = "view", source = source, first = 50L, nrow = 500L)
    index <- file_cagra_build_cpp(shard, 8L, 16L, 2^30, 4L, 64L,
        TRUE, 32L, 64L)
    on.exit(file_cagra_release_cpp(index$pointer), add = TRUE)

    raw <- tempfile(fileext = ".f32")
    writeBin(as.numeric(t(queries)), raw, size = 4L, endian = "little")
    on.exit(unlink(raw), add = TRUE)
    external <- list(format = "f32", path = raw, nrow = 5L, ncol = 4L)
    result <- file_cagra_query_cpp(index$pointer, external, 2L, 3L)
    expect_identical(result$backend_used, "cuda_cuvs_cagra_file_query")
    expect_true(result$index_reused)
    expect_identical(dim(result$indices), c(3L, 9L))
    expect_equal(as.integer(result$indices[, 1L]), c(24L, 42L, 205L))
    expect_equal(as.numeric(result$distances[, 1L]), rep(0, 3L),
        tolerance = 1e-3)
    expect_true(all(result$indices >= 1L & result$indices <= 500L))
    expect_error(file_cagra_query_cpp(index$pointer, external, 5L, 2L),
        "outside its file view")
    expect_identical(file_cagra_query_cpp(index$pointer, external, 2L, 3L),
        result)
})
