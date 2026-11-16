make_hnsw_audit_case <- function(n, build, peak, ef, recall) {
    list(passed = TRUE, n = n, p = 96L, k = 30L, threads = 16L,
         scope = "resident_hnsw_build_calibration_sampled_query_exact_audit",
         full_graph_performed = FALSE, hnsw_build_seconds = build,
         peak_memory_footprint_bytes = peak,
         reference_scope = "full_database_exact", query_count = 256L,
         pilot_rows = 128L, holdout_rows = 128L,
         hnsw_m = 32L, ef_construction = 200L,
         cells = Map(function(tier, effort, measured) {
             list(target_recall = tier, ef_search = effort,
                  measured_id_recall = measured)
         }, c(0.90, 0.95, 0.99), ef, recall))
}

test_that("source-locked HNSW holdout yields conditional 1B ranges", {
    cases <- list(
        make_hnsw_audit_case(2e7, 839.421, 33001561848,
                             c(120, 120, 480),
                             c(0.978125, 0.978125, 0.997395833333333)),
        make_hnsw_audit_case(5e7, 2420.433, 82188150912,
                             c(120, 240, 960),
                             c(0.9578125, 0.984375, 0.998958333333333)),
        make_hnsw_audit_case(1e8, 5267.396, 164255157688,
                             c(120, 240, 960),
                             c(0.960416666666667, 0.986979166666667,
                               0.9984375))
    )
    cases[[2L]]$prediction_passed <- TRUE
    cases[[2L]]$build_prediction_relative_error <- 0.013471801560307028
    cases[[2L]]$peak_memory_prediction_relative_error <-
        0.0012309199941961113
    for (i in seq_along(cases)) {
        cases[[i]]$source_archive_sha256 <- strrep("a", 64L)
        cases[[i]]$input_sha256 <- strrep("b", 64L)
        cases[[i]]$audit_sha256 <- strrep(as.character(i), 64L)
        cases[[i]]$query_rows_sha256 <- strrep(as.character(i + 3L), 64L)
    }
    request <- list(n = 1e9, p = 96L, k = 30L,
        host_memory_bytes = 3e12, target_recall = 0.99,
        cases = cases,
        source_archive_sha256 = paste(rep("a", 64L), collapse = ""),
        input_sha256 = paste(rep("b", 64L), collapse = ""),
        audit_sha256 = vapply(cases, function(x) x$audit_sha256,
                             character(1L)))
    plan <- do.call(faissR:::nn_large_hnsw_project_audited, request)
    expect_identical(plan$index_candidate, "hnsw")
    expect_identical(plan$storage_mode, "resident")
    expect_identical(plan$reference_residency, "resident_cpu")
    expect_identical(plan$index_residency, "resident_cpu")
    expect_identical(plan$selected_ef_search, 960L)
    expect_identical(plan$calibration_query_count, 256L)
    expect_equal(plan$conditional_index_build_seconds_range,
                 c(69730.17171816045, 77186.09988197369),
                 tolerance = 1e-8)
    expect_equal(plan$conditional_build_peak_bytes_range,
                 c(1623284627465.5024, 1638533661329.0159),
                 tolerance = 1e-8)
    expect_equal(plan$index_build_capacity_budget_bytes,
                 1638533661329.0159 / 0.8, tolerance = 1e-8)
    expect_identical(plan$tuning_status,
        "conditional_extrapolation_requires_live_recall_and_hardware_validation")
    expect_false(plan$r_full_output_vector_supported)
    request$target_recall <- 0.95
    expect_identical(do.call(faissR:::nn_large_hnsw_project_audited,
                             request)$selected_ef_search, 240L)
    request$target_recall <- 0.90
    expect_identical(do.call(faissR:::nn_large_hnsw_project_audited,
                             request)$selected_ef_search, 120L)

    request$cases[[2L]]$prediction_passed <- FALSE
    expect_error(do.call(faissR:::nn_large_hnsw_project_audited, request),
                 "holdout did not pass")
    request$cases[[2L]]$prediction_passed <- TRUE
    request$cases[[2L]]$build_prediction_relative_error <- 0
    expect_error(do.call(faissR:::nn_large_hnsw_project_audited, request),
                 "holdout did not pass")
    request$cases[[2L]]$build_prediction_relative_error <-
        cases[[2L]]$build_prediction_relative_error
    request$cases[[2L]]$cells[[3L]]$measured_id_recall <- 0.98
    expect_error(do.call(faissR:::nn_large_hnsw_project_audited, request),
                 "recall cells")
    request$cases[[2L]]$cells[[3L]]$measured_id_recall <-
        cases[[2L]]$cells[[3L]]$measured_id_recall
    request$cases[[3L]]$p <- 95L
    expect_error(do.call(faissR:::nn_large_hnsw_project_audited, request),
                 "mixed scope")
    request$cases <- cases
    request$cases[[3L]]$query_rows_sha256 <-
        request$cases[[1L]]$query_rows_sha256
    expect_error(do.call(faissR:::nn_large_hnsw_project_audited, request),
                 "mix source, input, query")
    request$cases <- cases
    request$cases[[3L]]$source_archive_sha256 <- strrep("f", 64L)
    expect_error(do.call(faissR:::nn_large_hnsw_project_audited, request),
                 "mix source, input, query")
    request$cases <- cases
    request$cases[[2L]]$query_count <- 128L
    expect_error(do.call(faissR:::nn_large_hnsw_project_audited, request),
                 "mix source, input, query")
    request$cases <- cases
    request$audit_sha256[[3L]] <- request$audit_sha256[[1L]]
    expect_error(do.call(faissR:::nn_large_hnsw_project_audited, request),
                 "Three source-locked")
})

test_that("matched 512-query HNSW audits preserve independent holdout", {
    cases <- list(
        make_hnsw_audit_case(1e7, 390, 10e9, c(32, 64, 256),
                             c(0.94, 0.97, 0.995)),
        make_hnsw_audit_case(5e7, 2244.96, 50e9, c(32, 64, 256),
                             c(0.93, 0.96, 0.992)),
        make_hnsw_audit_case(1e8, 5200, 100e9, c(32, 64, 256),
                             c(0.92, 0.95, 0.991))
    )
    for (i in seq_along(cases)) {
        cases[[i]]$k <- 10L
        cases[[i]]$query_count <- 512L
        cases[[i]]$pilot_rows <- 256L
        cases[[i]]$holdout_rows <- 256L
        cases[[i]]$fixed_query_search_seconds_by_ef <-
            setNames(c(1, 2, 4) * c(1, 5, 10)[[i]],
                     c("32", "512", "2048"))
        cases[[i]]$source_archive_sha256 <- strrep("a", 64L)
        cases[[i]]$input_sha256 <- strrep("b", 64L)
        cases[[i]]$audit_sha256 <- strrep(as.character(i), 64L)
        cases[[i]]$query_rows_sha256 <- strrep(as.character(i + 3L), 64L)
    }
    predicted <- 390 * 5^(log(5200 / 390) / log(10))
    cases[[2L]]$prediction_passed <- TRUE
    cases[[2L]]$build_prediction_relative_error <- abs(2244.96 / predicted - 1)
    cases[[2L]]$peak_memory_prediction_relative_error <- 0
    cases[[2L]]$fixed_query_prediction_passed <- TRUE
    cases[[2L]]$fixed_query_prediction_relative_error_by_ef <-
        setNames(c(0, 0, 0), c("32", "512", "2048"))
    request <- list(n = 1e9, p = 96L, k = 10L,
        host_memory_bytes = 3e12, target_recall = 0.99,
        cases = cases, source_archive_sha256 = strrep("a", 64L),
        input_sha256 = strrep("b", 64L),
        audit_sha256 = vapply(cases, function(x) x$audit_sha256,
                             character(1L)))
    plan <- do.call(faissR:::nn_large_hnsw_project_audited, request)
    expect_identical(plan$calibration_query_count, 512L)
    expect_identical(plan$calibration_pilot_rows, 256L)
    expect_identical(plan$calibration_holdout_rows, 256L)
    expect_identical(plan$fixed_query_timing_status,
                     "conditional_fixed_query_only_not_full_self_search")
    expect_equal(plan$conditional_fixed_query_search_seconds_range_by_ef[[
        "512"]], c(200, 200))
    expect_true(all(is.na(plan$conditional_full_self_search_seconds_range)))
    request$cases[[2L]]$pilot_rows <- 128L
    expect_error(do.call(faissR:::nn_large_hnsw_project_audited, request),
                 "mix source, input, query")
    request$cases <- cases
    request$cases[[2L]]$fixed_query_search_seconds_by_ef[["512"]] <- 20
    expect_error(do.call(faissR:::nn_large_hnsw_project_audited, request),
                 "fixed-query timing holdout")
    request$cases <- cases
    request$cases[[3L]]$fixed_query_search_seconds_by_ef <- NULL
    expect_error(do.call(faissR:::nn_large_hnsw_project_audited, request),
                 "timing evidence is incomplete")
})

test_that("file-input HNSW projection preserves reference residency", {
    cases <- list(
        make_hnsw_audit_case(1e7, 390, 10e9, c(32, 64, 256),
                             c(0.94, 0.97, 0.995)),
        make_hnsw_audit_case(5e7, 2244.96, 50e9,
                             c(32, 64, 256), c(0.93, 0.96, 0.992)),
        make_hnsw_audit_case(1e8, 5200, 100e9,
                             c(32, 64, 256), c(0.92, 0.95, 0.991))
    )
    for (i in seq_along(cases)) {
        cases[[i]]$scope <-
            "file_input_resident_hnsw_build_calibration_sampled_query_exact_audit"
        cases[[i]]$source_archive_sha256 <- strrep("a", 64L)
        cases[[i]]$input_sha256 <- strrep("b", 64L)
        cases[[i]]$audit_sha256 <- strrep(as.character(i), 64L)
        cases[[i]]$query_rows_sha256 <- strrep(as.character(i + 3L), 64L)
    }
    predicted <- 390 * 5^(log(5200 / 390) / log(10))
    cases[[2L]]$prediction_passed <- TRUE
    cases[[2L]]$build_prediction_relative_error <- abs(2244.96 / predicted - 1)
    cases[[2L]]$peak_memory_prediction_relative_error <- 0
    request <- list(n = 1e9, p = 96L, k = 30L,
        host_memory_bytes = 3e12, target_recall = 0.99,
        cases = cases, storage_mode = "file_backed",
        source_archive_sha256 = paste(rep("a", 64L), collapse = ""),
        input_sha256 = paste(rep("b", 64L), collapse = ""),
        audit_sha256 = vapply(cases, function(x) x$audit_sha256,
                             character(1L)))
    plan <- do.call(faissR:::nn_large_hnsw_project_audited, request)
    expect_identical(plan$storage_mode, "file_backed")
    expect_identical(plan$reference_residency, "file_backed")
    expect_identical(plan$index_residency, "resident_cpu")
    expect_true(plan$file_backed_output_format_addressable)
    request$cases[[2L]]$hnsw_build_seconds <- predicted * 1.25
    request$cases[[2L]]$build_prediction_relative_error <- 0.25
    expect_error(do.call(faissR:::nn_large_hnsw_project_audited, request),
                 "holdout did not pass")
    request$cases <- cases
    request$storage_mode <- "resident"
    expect_error(do.call(faissR:::nn_large_hnsw_project_audited, request),
                 "mixed scope")
})
