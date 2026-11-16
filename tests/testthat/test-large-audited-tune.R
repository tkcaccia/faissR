test_that("audited large IVF tuning keeps target distinct from recall", {
    grid <- data.frame(
        n = 1e7, p = 96L, k = 30L, gpu_count = 2L,
        mode = "shard", nlist = 4096L,
        nprobe = c(32L, 64L, 128L),
        measured_recall = c(0.951, 0.979, 0.994),
        search_seconds_median = c(0.007, 0.012, 0.022),
        index_build_seconds = 211,
        query_count = 256L, search_repetitions = 3L,
        source_sha = paste(rep("a", 40L), collapse = ""),
        input_sha256 = paste(rep("b", 64L), collapse = ""),
        query_rows_sha256 = paste(rep("c", 64L), collapse = ""),
        audit_pass = TRUE, reference_scope = "full_database_exact",
        timing_scope =
            "index_build_plus_persistent_search_256_queries",
        job_id = "1627802", stringsAsFactors = FALSE
    )
    request <- list(n = 1e7, p = 96L, k = 30L,
        backend = "cuda", storage_mode = "resident",
        host_memory_bytes = 256 * 1024^3,
        device_memory_bytes = 48 * 1024^3,
        gpu_count = 2L, audited_probes = grid,
        source_sha = grid$source_sha[[1L]],
        input_sha256 = grid$input_sha256[[1L]])
    low <- do.call(faissR:::nn_large_knn_tune_audited,
                   c(request, list(target_recall = 0.95)))
    high <- do.call(faissR:::nn_large_knn_tune_audited,
                    c(request, list(target_recall = 0.99)))
    expect_identical(low$selected_nprobe, 32L)
    expect_identical(high$selected_nprobe, 128L)
    expect_identical(high$target_recall, 0.99)
    expect_equal(high$measured_recall, 0.994)
    expect_identical(high$calibration_query_count, 256L)
    expect_identical(high$calibration_attempted_cells, 3L)
    expect_identical(high$tuning_status,
        "measured_fixed_query_candidate_requires_live_recall_check")

    grid$audit_pass[3L] <- FALSE
    request$audited_probes <- grid
    missed <- do.call(faissR:::nn_large_knn_tune_audited,
                      c(request, list(target_recall = 0.99)))
    expect_true(is.na(missed$selected_nprobe))
    expect_identical(missed$tuning_status,
                     "audited_grid_has_no_passing_cell")
    expect_identical(missed$calibration_attempted_cells, 3L)

    grid$measured_recall[3L] <- NA_real_
    grid$search_seconds_median[3L] <- NA_real_
    grid$index_build_seconds[3L] <- NA_real_
    grid$query_count[3L] <- NA_integer_
    grid$search_repetitions[3L] <- 0L
    request$audited_probes <- grid
    failed <- do.call(faissR:::nn_large_knn_tune_audited,
                      c(request, list(target_recall = 0.95)))
    expect_identical(failed$calibration_attempted_cells, 3L)
    expect_identical(failed$calibration_passing_cells, 2L)
    expect_identical(failed$selected_nprobe, 32L)
    grid$audit_pass[3L] <- TRUE
    request$audited_probes <- grid
    expect_error(do.call(faissR:::nn_large_knn_tune_audited,
                         c(request, list(target_recall = 0.95))),
                 "incomplete or mixes scopes")
})

test_that("large IVF planner rejects mixed or unverified grids", {
    grid <- data.frame(
        n = c(1e7, 1e7), p = 96L, k = 30L,
        gpu_count = 2L, mode = "shard", nlist = 4096L,
        nprobe = c(32L, 128L), measured_recall = c(0.95, 0.995),
        search_seconds_median = c(0.01, 0.02),
        index_build_seconds = 200,
        query_count = 256L, search_repetitions = 3L,
        source_sha = paste(rep("a", 40L), collapse = ""),
        input_sha256 = paste(rep("b", 64L), collapse = ""),
        query_rows_sha256 = paste(rep("c", 64L), collapse = ""),
        audit_pass = TRUE, reference_scope = "full_database_exact",
        timing_scope =
            "index_build_plus_persistent_search_256_queries",
        job_id = "1627802", stringsAsFactors = FALSE
    )
    request <- list(n = 1e7, p = 96L, k = 30L,
        backend = "cuda", storage_mode = "resident",
        host_memory_bytes = 256 * 1024^3,
        device_memory_bytes = 48 * 1024^3,
        gpu_count = 2L, target_recall = 0.99,
        audited_probes = grid,
        source_sha = grid$source_sha[[1L]],
        input_sha256 = grid$input_sha256[[1L]])
    grid$query_rows_sha256[[2L]] <- paste(rep("d", 64L), collapse = "")
    request$audited_probes <- grid
    expect_error(do.call(faissR:::nn_large_knn_tune_audited, request),
                 "mixes scopes")
    grid$query_rows_sha256[[2L]] <- grid$query_rows_sha256[[1L]]
    request$audited_probes <- grid
    request$source_sha <- paste(rep("e", 40L), collapse = "")
    absent <- do.call(faissR:::nn_large_knn_tune_audited, request)
    expect_identical(absent$tuning_status, "no_audited_matching_grid")
    expect_true(is.na(absent$selected_nprobe))
    request$source_sha <- grid$source_sha[[1L]]
    request$backend <- "metal"
    request$gpu_count <- 1L
    expect_error(do.call(faissR:::nn_large_knn_tune_audited, request),
                 "resident CUDA")
})
