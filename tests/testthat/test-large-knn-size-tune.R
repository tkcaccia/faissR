size_tuning_fixture <- function(backend = "cpu", storage = "resident",
                                p = 96L, profile = strrep("a", 64L),
                                layout = "single_logical_index") {
    n <- rep(c(1e7, 5e7, 1e8), each = 3L)
    choice <- rep(1:3, 3L)
    # First choice is slow to build but fast to search. Third misses recall.
    data.frame(
        n, p, k = 30L, backend, storage_mode = storage,
        gpu_count = as.integer(backend != "cpu"), profile_sha256 = profile,
        layout, tuning_rows_sha256 = strrep("b", 64L),
        audit_rows_sha256 = strrep("c", 64L),
        reference_scope = "full_database_exact",
        timing_scope = "index_build_plus_fixed_query", query_count = 256L,
        mean_recall = c(0.998, 0.995, 0.975)[choice],
        min_row_recall = c(0.95, 0.93, 0.91)[choice], audit_pass = TRUE,
        build_seconds = n / 1e6 * c(2, 1, 1)[choice],
        search_seconds = n / 1e8 * c(1, 2, 0.5)[choice],
        peak_host_bytes = n * (4 * p + 512),
        peak_device_bytes = n * (4 * p + 80),
        m = c(32L, 16L, 16L)[choice], ef_construction = 200L,
        ef_search = c(128L, 256L, 64L)[choice],
        nlist = c(1024L, 512L, 512L)[choice],
        nprobe = c(512L, 128L, 64L)[choice]
    )
}

size_tuning_test_plan <- function(rows, backend = "cpu", storage = "resident",
                                  p = 96L, profile = strrep("a", 64L),
                                  ...) {
    faissR:::nn_large_knn_size_tune(
        1e9, p, 30L, backend, storage, 10e12, rows, profile,
        fit_sizes = c(1e7, 1e8), holdout_n = 5e7,
        device_memory_bytes = if (backend == "cpu") NA_real_ else 10e12,
        gpu_count = as.integer(backend != "cpu"), ...
    )
}

test_that("shape and host specific profiles can plan all supported routes", {
    routes <- data.frame(
        backend = c("cpu", "cuda", "cpu", "metal"),
        host = c("linux", "linux", "mac", "mac"))
    for (i in seq_len(nrow(routes))) {
        backend <- routes$backend[[i]]
        for (storage in c("resident", "file_backed")) {
            for (p in c(11L, 96L, 1024L)) {
                profile <- strrep(as.character(i), 64L)
                rows <- size_tuning_fixture(backend, storage, p, profile)
                plan <- size_tuning_test_plan(
                    rows, backend, storage, p, profile)
                expect_identical(plan$holdout$status, "interpolation_pass")
                expect_false(plan$settings_applied_automatically)
                expect_true(is.na(plan$recall_at_target))
                expect_true(is.na(plan$projected_full_self_search_seconds))
                expect_false(plan$r_full_output_vector_supported)
                if (backend == "cpu") {
                    expect_equal(plan$settings$ef_search, 256L)
                } else {
                    expect_equal(plan$settings$nprobe, 128L)
                    expect_equal(plan$settings$nlist, 512L)
                }
            }
        }
    }
})

test_that("search reuse, recall tier, and hardware cap affect selection", {
    rows <- size_tuning_fixture("metal", "file_backed")
    plan <- size_tuning_test_plan(rows, "metal", "file_backed",
        expected_query_batches = 200)
    expect_equal(plan$settings, list(nlist = 1024L, nprobe = 512L))
    expect_true(plan$parameter_limits_reached)
    relaxed <- size_tuning_test_plan(rows, "metal", "file_backed",
        target_recall = 0.95)
    expect_equal(relaxed$settings$nprobe, 64L)
    expect_equal(relaxed$recall_miss_cells, 0)
    cpu <- size_tuning_fixture()
    selected <- size_tuning_test_plan(cpu, expected_query_batches = 200)
    expect_equal(selected$settings$m, 32L)
    expect_equal(selected$settings$ef_search, 128L)
    cpu <- cpu[!(cpu$n == 1e8 & cpu$ef_search == 256L), ]
    selected <- size_tuning_test_plan(cpu)
    expect_equal(selected$settings$m, 32L)
})

test_that("missing or failed exact holdouts never produce settings", {
    rows <- size_tuning_fixture()
    inconsistent <- rows
    inconsistent$audit_rows_sha256[[2L]] <- strrep("d", 64L)
    expect_error(size_tuning_test_plan(inconsistent), "invalid timings")
    wrong <- size_tuning_test_plan(rows, profile = strrep("d", 64L))
    expect_identical(wrong$tuning_status, "no_matching_calibration")
    kept <- rows$n == 5e7 & rows$ef_search == 256L
    no_holdout <- size_tuning_test_plan(rows[!kept, ])
    expect_identical(no_holdout$tuning_status,
                     "predicted_holdout_setting_not_audited")
    expect_null(no_holdout$settings)
    slow <- rows
    slow$search_seconds[kept] <- slow$search_seconds[kept] * 2
    expect_identical(size_tuning_test_plan(slow)$tuning_status,
                     "holdout_prediction_error")
    rows$min_row_recall[kept] <- 0.80
    expect_identical(size_tuning_test_plan(rows)$tuning_status,
                     "holdout_recall_miss")
    rows$audit_pass[kept] <- FALSE
    denied <- size_tuning_test_plan(rows)
    expect_identical(denied$tuning_status, "holdout_audit_failed")
    expect_equal(denied$failed_calibration_cells, 1L)
})

test_that("shard projections use reference rows and enforce capacity", {
    rows <- size_tuning_fixture("cuda", "file_backed",
                               layout = "reference_shard")
    plan <- faissR:::nn_large_knn_size_tune(
        1e9, 96L, 30L, "cuda", "file_backed", 192e9, rows, strrep("a", 64L),
        fit_sizes = c(1e7, 1e8), holdout_n = 5e7,
        gpu_count = 1L, device_memory_bytes = 16e9, disk_free_bytes = 24e9,
        reference_shard_rows = 2e8
    )
    expect_equal(plan$tuned_reference_rows, 2e8)
    expect_equal(plan$reference_shards, 5)
    expect_false(plan$device_projection_fits)
    expect_false(plan$file_backed_output_floor_fits)
    expect_true(is.na(plan$projected_fixed_query_seconds))
    expect_equal(plan$settings$nprobe, 128L)
    wrong_layout <- size_tuning_test_plan(rows, "cuda", "file_backed")
    expect_identical(wrong_layout$tuning_status, "no_matching_calibration")
    unsupported <- size_tuning_test_plan(NULL, "metal", p = 1025L)
    expect_identical(unsupported$tuning_status, "shape_outside_metal_ivf_limits")
    expect_error(size_tuning_test_plan(rows, "cuda", "file_backed",
        reference_shard_rows = 1e10), "Invalid large KNN tuning request")
})

test_that("CUDA projections respect the declared native probe limit", {
    rows <- size_tuning_fixture("cuda")
    rows$nlist <- rows$nlist * 4L
    rows$nprobe <- rows$nprobe * 4L
    expect_error(size_tuning_test_plan(rows, "cuda", cuda_probe_limit = 1024L),
                 "invalid timings, settings, or scope")
    valid <- rows[rows$nprobe <= 1024L, ]
    plan <- size_tuning_test_plan(valid, "cuda", cuda_probe_limit = 1024L)
    expect_equal(plan$settings$nprobe, 512L)
    expect_equal(plan$ivf_limits[["probes"]], 1024L)
    expect_equal(size_tuning_test_plan(rows, "cuda")$ivf_limits[["probes"]],
                 2048L)
})
