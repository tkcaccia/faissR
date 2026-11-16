cohort_grid <- function() {
    expand <- expand.grid(cohort = c("first", "heldout"),
                          nprobe = c(64L, 128L, 256L),
                          KEEP.OUT.ATTRS = FALSE,
                          stringsAsFactors = FALSE)
    expand$n <- 1e7
    expand$p <- 96L
    expand$k <- 10L
    expand$backend <- "cuda"
    expand$storage_mode <- "resident"
    expand$gpu_count <- 1L
    expand$mode <- "single"
    expand$nlist <- 1024L
    expand$query_rows_sha256 <- ifelse(expand$cohort == "first",
        paste(rep("c", 64L), collapse = ""),
        paste(rep("d", 64L), collapse = ""))
    expand$source_archive_sha256 <- paste(rep("a", 64L), collapse = "")
    expand$input_sha256 <- paste(rep("b", 64L), collapse = "")
    expand$audit_pass <- TRUE
    expand$reference_scope <- "full_database_exact"
    expand$recall_scope <- "nonself_strict_id_at_k"
    expand$timing_scope <- "persistent_search_only"
    expand$query_count <- 256L
    expand$search_repetitions <- 3L
    expand$measured_recall <- c(0.995, 0.996, 0.9992, 0.9991, 1, 1)
    expand$minimum_tie_aware_row_recall <- c(0.8, 0.9, 0.9, 0.9, 1, 1)
    expand$search_seconds_median <- c(0.15, 0.16, 0.29, 0.30, 0.56, 0.57)
    expand$index_build_seconds <- 44
    expand$backend_used <- "faiss_gpu_persistent_query"
    expand$device_count <- 1L
    expand
}

cohort_request <- function(grid) {
    list(n = 1e7, p = 96L, k = 10L, backend = "cuda",
         storage_mode = "resident", host_memory_bytes = 16 * 1024^3,
         device_memory_bytes = 16 * 1024^3, gpu_count = 1L,
         target_recall = 0.99, audited_cohorts = grid,
         source_archive_sha256 = grid$source_archive_sha256[[1L]],
         input_sha256 = grid$input_sha256[[1L]])
}

test_that("large IVF cohort tuning requires all cohorts and row floor", {
    grid <- cohort_grid()
    plan <- do.call(faissR:::nn_large_knn_tune_cohorts,
                    cohort_request(grid))
    expect_identical(plan$selected_nprobe, 128L)
    expect_equal(plan$measured_recall, 0.9991)
    expect_equal(plan$measured_minimum_row_recall, 0.9)
    expect_identical(plan$calibration_cohort_count, 2L)
    expect_identical(plan$calibration_recall_scope,
                     "nonself_strict_id_at_k")
    expect_identical(plan$calibration_attempted_cells, 6L)
    expect_identical(plan$calibration_passing_probes, c(128L, 256L))
    expect_identical(plan$tuning_status,
        "measured_fixed_query_candidate_requires_live_graph_audit")

    grid$audit_pass[grid$cohort == "heldout" &
                    grid$nprobe == 128L] <- FALSE
    grid$measured_recall[!grid$audit_pass] <- NA_real_
    grid$search_seconds_median[!grid$audit_pass] <- NA_real_
    failed <- do.call(faissR:::nn_large_knn_tune_cohorts,
                      cohort_request(grid))
    expect_identical(failed$selected_nprobe, 256L)
    expect_identical(failed$calibration_failed_cells, 1L)

    grid <- cohort_grid()
    grid$measured_recall[grid$cohort == "heldout" &
                         grid$nprobe == 256L] <- 0.98
    fallback <- do.call(faissR:::nn_large_knn_tune_cohorts,
                        cohort_request(grid))
    expect_identical(fallback$selected_nprobe, 128L)
    grid$measured_recall[grid$cohort == "heldout" &
                         grid$nprobe == 128L] <- 0.98
    none <- do.call(faissR:::nn_large_knn_tune_cohorts,
                    cohort_request(grid))
    expect_true(is.na(none$selected_nprobe))
    expect_identical(none$tuning_status,
        "audited_cohorts_have_no_passing_probe")
})

test_that("large IVF cohort tuning accepts a frozen tie-aware scope", {
    grid <- cohort_grid()
    grid$recall_scope <- "nonself_distance_threshold_at_k"
    grid$measured_recall <- c(0.998, 0.998, 0.999, 0.999,
                              1, 1)
    plan <- do.call(faissR:::nn_large_knn_tune_cohorts,
                    cohort_request(grid))
    expect_identical(plan$calibration_recall_scope,
                     "nonself_distance_threshold_at_k")
    expect_identical(plan$selected_nprobe, 128L)
    expect_equal(plan$measured_recall, 0.999)

    grid$recall_scope[[2L]] <- "nonself_strict_id_at_k"
    expect_error(do.call(faissR:::nn_large_knn_tune_cohorts,
                         cohort_request(grid)), "mixes scopes")
})

test_that("large IVF cohort tuning rejects mixed scopes and missing cells", {
    grid <- cohort_grid()
    grid$audit_pass[[1L]] <- NA
    expect_error(do.call(faissR:::nn_large_knn_tune_cohorts,
                         cohort_request(grid)), "incomplete or mixes scopes")
    grid <- cohort_grid()
    one <- grid[grid$cohort == "first", ]
    unreplicated <- do.call(faissR:::nn_large_knn_tune_cohorts,
                            cohort_request(one))
    expect_identical(unreplicated$tuning_status,
                     "insufficient_independent_cohorts")
    expect_identical(unreplicated$calibration_cohort_count, 1L)
    expect_identical(unreplicated$calibration_attempted_cells, 3L)
    expect_true(is.na(unreplicated$selected_nprobe))

    request <- cohort_request(grid)
    request$source_archive_sha256 <- paste(rep("e", 64L), collapse = "")
    absent <- do.call(faissR:::nn_large_knn_tune_cohorts, request)
    expect_identical(absent$tuning_status, "no_matching_audited_cohorts")

    request$minimum_row_recall <- 1.1
    expect_error(do.call(faissR:::nn_large_knn_tune_cohorts, request),
                 "Invalid minimum row recall")

    request <- cohort_request(grid[-1L, ])
    expect_error(do.call(faissR:::nn_large_knn_tune_cohorts, request),
                 "Every probe")

    grid$recall_scope[[2L]] <- "self_inclusive"
    expect_error(do.call(faissR:::nn_large_knn_tune_cohorts,
                         cohort_request(grid)), "mixes scopes")
    grid <- cohort_grid()
    grid$query_rows_sha256[[2L]] <- paste(rep("e", 64L), collapse = "")
    expect_error(do.call(faissR:::nn_large_knn_tune_cohorts,
                         cohort_request(grid)), "query or backend")
    grid <- cohort_grid()
    grid$query_rows_sha256[grid$cohort == "heldout"] <-
        grid$query_rows_sha256[[1L]]
    expect_error(do.call(faissR:::nn_large_knn_tune_cohorts,
                         cohort_request(grid)), "distinct row identities")
    grid <- cohort_grid()
    grid$storage_mode <- "file_backed"
    fileback <- do.call(faissR:::nn_large_knn_tune_cohorts,
                        cohort_request(grid) |>
                            within(storage_mode <- "file_backed"))
    expect_identical(fileback$storage_mode, "file_backed")
    expect_identical(fileback$selected_nprobe, 128L)
})
