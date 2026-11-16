test_that("private posting builder groups rows and preserves IDs", {
    skip_if_not_installed("Rcpp")
    expect_false("faissR_posting_build_v1" %in%
        getNamespaceExports("faissR"))
    header <- system.file("include", "faissR_api_v1.h",
        package = "faissR")
    Rcpp::cppFunction(
        code = '
        SEXP faissR_test_posting_build(SEXP arguments) {
          return faissR_get_posting_build_v1()(arguments);
        }',
        includes = paste0('#include "', header, '"')
    )
    x <- rbind(c(-5, 0), c(4, 0), c(-4, 1),
        c(5, 1), c(-3, 2), c(6, 2))
    centers <- rbind(c(-4, 0), c(5, 0))
    input <- tempfile(fileext = ".f32")
    on.exit(unlink(input), add = TRUE)
    writeBin(as.numeric(t(x)), input, size = 4L, endian = "little")
    run <- function(access) {
        prefix <- tempfile()
        on.exit(unlink(paste0(prefix, c(".features.f32",
            ".ids.u32", ".offsets.u64"))), add = TRUE)
        seen <- numeric()
        args <- list(spec = list(format = "f32", path = input,
            nrow = nrow(x), ncol = ncol(x), access = access),
            centers = centers, prefix = prefix, chunk_rows = 2L,
            workers = 2L, resume_state = NULL,
            checkpoint_callback = function(done, counts, positions) {
                seen <<- c(seen, done)
            }, checkpoint_every = 1L)
        found <- faissR_test_posting_build(args)
        expect_identical(found$provider, "faissR")
        expect_identical(found$backend_used,
            "faissr_cpu_posting_build")
        expect_identical(found$method,
            "nearest_center_physical_postings")
        expect_equal(found$counts, c(3, 3))
        expect_equal(found$offsets, c(0, 3, 6))
        expect_equal(seen, c(0, 2, 4, 6))
        ids <- readBin(paste0(prefix, ".ids.u32"), integer(),
            n = nrow(x), size = 4L, endian = "little")
        expect_identical(ids, c(1L, 3L, 5L, 2L, 4L, 6L))
        features <- readBin(paste0(prefix, ".features.f32"),
            numeric(), n = length(x), size = 4L,
            endian = "little")
        expect_equal(matrix(features, ncol = ncol(x), byrow = TRUE),
            x[ids, , drop = FALSE], tolerance = 1e-6)
        args$resume_state <- list(completed_rows = nrow(x),
            counts = found$counts, positions = found$offsets[-1L])
        expect_equal(faissR_test_posting_build(args)$offsets,
            found$offsets)
    }
    run("stream")
    if (.Platform$OS.type != "windows") run("mmap")
})
