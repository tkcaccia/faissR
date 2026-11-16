test_that("private posting graph reorder restores and validates rows", {
    skip_if_not_installed("Rcpp")
    expect_false("faissR_reorder_posting_graph_v1" %in%
        getNamespaceExports("faissR"))
    header <- system.file("include", "faissR_api_v1.h",
        package = "faissR")
    Rcpp::cppFunction(
        code = '
        SEXP faissR_test_graph_reorder(SEXP arguments) {
          return faissR_get_reorder_posting_graph_v1()(arguments);
        }',
        includes = paste0('#include "', header, '"')
    )
    n <- 17L
    k <- 2L
    order <- c(8L, 1L, 17L, 4L, 11L, 2L, 15L, 6L,
        13L, 3L, 12L, 5L, 16L, 7L, 14L, 9L, 10L)
    indices <- cbind(order %% n + 1L,
        (order + 1L) %% n + 1L)
    distances <- cbind(order / 10, order / 5)
    root <- tempfile()
    dir.create(root)
    on.exit(unlink(root, recursive = TRUE), add = TRUE)
    paths <- file.path(root, c("rows.u32", "grouped.ids.u32",
        "grouped.dist.f32", "output.ids.u32", "output.dist.f32"))
    write_files <- function(row_ids = order, neighbors = indices) {
        writeBin(as.integer(row_ids), paths[[1L]], size = 4L,
            endian = "little")
        writeBin(as.integer(t(neighbors)), paths[[2L]], size = 4L,
            endian = "little")
        writeBin(as.numeric(t(distances)), paths[[3L]], size = 4L,
            endian = "little")
    }
    args <- list(posting_ids_path = paths[[1L]],
        grouped_indices_path = paths[[2L]],
        grouped_distances_path = paths[[3L]],
        output_indices_path = paths[[4L]],
        output_distances_path = paths[[5L]],
        n_vertices = n, k = k, bucket_rows = 2L,
        read_rows = 3L, restart_work = FALSE)
    write_files()
    found <- faissR_test_graph_reorder(args)
    expect_identical(found$provider, "faissR")
    expect_identical(found$backend_used,
        "faissr_cpu_knn_reorder")
    expect_identical(found$method, "posting_graph_reorder")
    expect_identical(found$n_vertices, as.numeric(n))
    expect_identical(found$k, k)
    restored <- base::order(order)
    observed <- readBin(paths[[4L]], integer(), n * k,
        size = 4L, endian = "little")
    expect_identical(observed,
        as.integer(t(indices[restored, , drop = FALSE])))
    expect_false(file.exists(paste0(paths[[4L]], ".reorder-work")))
    bad <- order
    bad[[1L]] <- bad[[2L]]
    write_files(row_ids = bad)
    expect_error(faissR_test_graph_reorder(args),
        "missing or duplicated")
    broken <- indices
    broken[1L, 1L] <- order[[1L]]
    write_files(neighbors = broken)
    expect_error(faissR_test_graph_reorder(args),
        "invalid neighbors")
})
