test_that("Metal NN-descent builds the same CAGRA graph from both sources", {
    skip_if_not(faissR:::nn_metal_available_cpp(), "Metal is required")
    previous <- options(faissR.metal_cagra_nn_descent = c(2L, 8L, 4L),
        faissR.metal_cagra_search_profile = NULL,
        faissR.metal_cagra_partition_seed = FALSE)
    on.exit(options(previous), add = TRUE)

    set.seed(91L)
    n <- 273L
    k <- 10L
    path <- tempfile(fileext = ".f32")
    on.exit(unlink(path), add = TRUE)
    writeBin(rnorm(n * 12L), path, size = 4L, endian = "little")
    x <- matrix(readBin(path, "numeric", n * 12L, size = 4L,
        endian = "little"), nrow = n, byrow = TRUE)
    source <- list(format = "f32", path = path, nrow = n, ncol = ncol(x))
    budget <- 512 * 1024^2

    resident <- faissR:::nn_metal_cpp(x, k, "cagra")
    expect_error(faissR:::file_metal_cagra_build_cpp(source, k, 1, 0.99),
        "budget")
    file <- faissR:::file_metal_cagra_build_cpp(source, k, budget, 0.99)
    on.exit(faissR:::file_metal_cagra_release_cpp(file$pointer), add = TRUE)
    batch <- faissR:::file_metal_cagra_search_cpp(file$pointer, 1L, n)

    expect_identical(resident$cagra_settings$builder, "metal_nn_descent")
    expect_identical(resident$cagra_settings$reverse_edges_backend, "metal")
    expect_identical(resident$cagra_settings$nn_descent_retained_membership,
        "candidate_hash")
    expect_identical(resident$cagra_settings$nn_descent_threadgroup_workspace,
        "shape_sized")
    expect_identical(resident$cagra_settings, file$params)
    expect_identical(file$backend_used, "faissr_native_metal_cagra_file")
    expect_gt(resident$build_timing$nn_descent_refinement_seconds, 0)
    expect_gt(file$observed_metal_owned_buffer_peak_bytes, 0)
    estimate <- 4 * n * (ncol(x) + 4 * file$params$graph_degree + 4) +
        128 * 1024^2
    expect_equal(file$device_bytes_estimate, estimate)
    expect_identical(resident$indices, batch$indices)
    expect_equal(resident$distances, batch$distances, tolerance = 1e-6)
    expect_true(all(is.finite(resident$distances)))
    expect_false(any(resident$indices == row(resident$indices)))
    expect_true(all(apply(resident$indices, 1L, anyDuplicated) == 0))
    exact <- lapply(seq_len(n), function(i) {
        d <- rowSums(sweep(x, 2L, x[i, ])^2)
        d[i] <- Inf
        order(d)[seq_len(k)]
    })
    recall <- function(indices) mean(vapply(seq_len(n), function(i)
        mean(indices[i, ] %in% exact[[i]]), numeric(1L)))
    expect_gte(recall(resident$indices), 0.99)

    options(faissR.metal_cagra_search_profile = c(512L, 52L))
    wider_batch <- faissR:::file_metal_cagra_search_cpp(file$pointer, 1L, n)
    wider_resident <- faissR:::nn_metal_cpp(x, k, "cagra")
    expect_equal(c(wider_batch$queue, wider_batch$steps), c(512L, 52L))
    expect_equal(c(wider_resident$cagra_settings$queue,
        wider_resident$cagra_settings$steps), c(512L, 52L))
    expect_identical(wider_resident$indices, wider_batch$indices)
    expect_equal(wider_resident$distances, wider_batch$distances, tolerance = 1e-6)

    for (invalid in list(c(513L, 52L), c(k, 52L), c(512L, 200L),
            c(512, Inf), c(64, 1.5), "512,52")) {
        options(faissR.metal_cagra_search_profile = invalid)
        expect_error(faissR:::file_metal_cagra_search_cpp(file$pointer, 1L, n),
            "search profile")
    }
    options(faissR.metal_cagra_search_profile = NULL)
    restored <- faissR:::file_metal_cagra_search_cpp(file$pointer, 1L, n)
    expect_identical(restored$queue, batch$queue)
    expect_identical(restored$steps, batch$steps)
    expect_identical(restored$indices, batch$indices)

    options(faissR.metal_cagra_nn_descent = c(2L, 17L, 4L))
    expect_error(faissR:::nn_metal_cpp(x, k, "cagra"), "profile")
    options(faissR.metal_cagra_nn_descent = FALSE)
    incumbent <- faissR:::nn_metal_cpp(x, k, "cagra")
    expect_identical(incumbent$cagra_settings$builder, "metal_ivf")

    options(faissR.metal_cagra_nn_descent = TRUE)
    for (tier in list(c(.90, 4L, 8L, 6L, 1L, 128L, 36L),
            c(.95, 4L, 12L, 10L, 2L, 256L, 48L))) {
        target <- tier[[1L]]
        resident_tier <- faissR:::nn_metal_cpp(x, k, "cagra", "euclidean", target)
        staged <- faissR:::file_metal_cagra_build_cpp(source, k, budget, target)
        file_tier <- tryCatch(
            faissR:::file_metal_cagra_search_cpp(staged$pointer, 1L, n),
            finally = faissR:::file_metal_cagra_release_cpp(staged$pointer))
        settings <- resident_tier$cagra_settings
        fields <- c("nn_descent_seed_passes", "nn_descent_pivots",
            "nn_descent_iterations", "width", "queue", "steps")
        expect_equal(unname(unlist(settings[fields])), tier[-1L])
        expect_identical(settings, staged$params)
        expect_identical(resident_tier$indices, file_tier$indices)
        expect_equal(resident_tier$distances, file_tier$distances, tolerance = 1e-6)
        expect_gte(recall(file_tier$indices), target)
    }

    options(faissR.metal_cagra_nn_descent = c(2L, 8L, 4L),
        faissR.metal_cagra_partition_seed = TRUE)
    partitioned_resident <- faissR:::nn_metal_cpp(x, k, "cagra")
    partitioned <- faissR:::file_metal_cagra_build_cpp(source, k, budget, 0.99)
    partitioned_file <- tryCatch(
        faissR:::file_metal_cagra_search_cpp(partitioned$pointer, 1L, n),
        finally = faissR:::file_metal_cagra_release_cpp(partitioned$pointer))
    expect_identical(partitioned$params$nn_descent_seed_order, "projection_hash")
    expect_identical(partitioned_resident$cagra_settings, partitioned$params)
    expect_gt(partitioned$build_timing$seed_partition_seconds, 0)
    expect_equal(partitioned$device_bytes_estimate, estimate + 12 * n)
    expect_identical(partitioned_resident$indices, partitioned_file$indices)
    expect_equal(partitioned_resident$distances, partitioned_file$distances, tolerance = 1e-6)
    expect_gte(recall(partitioned_file$indices), 0.99)
})
