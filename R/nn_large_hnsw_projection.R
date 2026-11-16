# Private conditional CPU HNSW projection from three source-locked audits.
# The middle case must be a prospective held-out interpolation pass.
nn_large_hnsw_projection_metric <- function(cases, field, target_n) {
    sizes <- vapply(cases, function(x) as.double(x$n), numeric(1L))
    values <- vapply(cases, function(x) as.double(x[[field]]),
                     numeric(1L))
    pairs <- list(c(1L, 2L), c(2L, 3L), c(1L, 3L))
    estimate <- vapply(pairs, function(pair) {
        low <- pair[[1L]]
        high <- pair[[2L]]
        slope <- log(values[[high]] / values[[low]]) /
            log(sizes[[high]] / sizes[[low]])
        values[[high]] * (target_n / sizes[[high]])^slope
    }, numeric(1L))
    c(min(estimate), max(estimate))
}

nn_large_hnsw_project_audited <- function(
    n, p, k, host_memory_bytes, target_recall,
    cases, source_archive_sha256, input_sha256, audit_sha256,
    cpu_threads = 16L, headroom_fraction = 0.20,
    storage_mode = "resident"
) {
    base <- nn_large_knn_resource_plan(
        n, p, k, "cpu", storage_mode, host_memory_bytes,
        cpu_threads_available = cpu_threads,
        target_recall = target_recall,
        device_headroom_fraction = headroom_fraction
    )
    hash_ok <- function(x) is.character(x) && length(x) == 1L &&
        !is.na(x) && grepl("^[0-9a-f]{64}$", x)
    if (!is.list(cases) || length(cases) != 3L ||
        !hash_ok(source_archive_sha256) || !hash_ok(input_sha256) ||
        length(audit_sha256) != 3L ||
        !all(vapply(audit_sha256, hash_ok, logical(1L)))) {
        stop("Three source-locked HNSW audits are required.", call. = FALSE)
    }
    sizes <- vapply(cases, function(x) as.double(x$n), numeric(1L))
    expected_scope <- if (storage_mode == "file_backed") {
        "file_input_resident_hnsw_build_calibration_sampled_query_exact_audit"
    } else {
        "resident_hnsw_build_calibration_sampled_query_exact_audit"
    }
    if (any(!is.finite(sizes)) || any(diff(sizes) <= 0) ||
        n <= sizes[[3L]] ||
        any(!vapply(cases, function(x) {
            isTRUE(x$passed) && identical(x$scope, expected_scope) &&
                identical(x$full_graph_performed, FALSE) &&
                length(x$p) == 1L && is.finite(x$p) && x$p == p &&
                length(x$k) == 1L && is.finite(x$k) && x$k == k &&
                length(x$threads) == 1L && is.finite(x$threads) &&
                x$threads == cpu_threads &&
                is.finite(x$hnsw_build_seconds) &&
                x$hnsw_build_seconds > 0 &&
                is.finite(x$peak_memory_footprint_bytes) &&
                x$peak_memory_footprint_bytes > 0
        }, logical(1L)))) {
        stop("HNSW audits have mixed scope or invalid measurements.",
             call. = FALSE)
    }
    heldout <- cases[[2L]]
    interpolate <- function(field) {
        lower <- cases[[1L]][[field]]
        upper <- cases[[3L]][[field]]
        power <- log(upper / lower) / log(sizes[[3L]] / sizes[[1L]])
        lower * (sizes[[2L]] / sizes[[1L]])^power
    }
    build_error <- abs(heldout$hnsw_build_seconds /
        interpolate("hnsw_build_seconds") - 1)
    memory_error <- abs(heldout$peak_memory_footprint_bytes /
        interpolate("peak_memory_footprint_bytes") - 1)
    if (!isTRUE(heldout$prediction_passed) ||
        !is.finite(heldout$build_prediction_relative_error) ||
        !is.finite(heldout$peak_memory_prediction_relative_error) ||
        abs(heldout$build_prediction_relative_error - build_error) > 1e-6 ||
        abs(heldout$peak_memory_prediction_relative_error -
            memory_error) > 1e-6 ||
        build_error > 0.20 || memory_error > 0.25) {
        stop("Prospective HNSW interpolation holdout did not pass.",
             call. = FALSE)
    }
    cells_ok <- vapply(cases, function(x) {
        length(x$cells) == 3L &&
            identical(vapply(x$cells, function(cell)
                as.double(cell$target_recall), numeric(1L)),
                      c(0.90, 0.95, 0.99)) &&
            all(vapply(x$cells, function(cell) {
                is.finite(cell$measured_id_recall) &&
                    cell$measured_id_recall >= cell$target_recall &&
                    is.finite(cell$ef_search) &&
                    cell$ef_search >= k + 1 &&
                    cell$ef_search == floor(cell$ef_search)
            }, logical(1L)))
    }, logical(1L))
    if (!all(cells_ok)) stop("HNSW recall cells are incomplete or missed.",
                             call. = FALSE)
    build <- nn_large_hnsw_projection_metric(cases,
        "hnsw_build_seconds", n)
    peak <- nn_large_hnsw_projection_metric(cases,
        "peak_memory_footprint_bytes", n)
    tier <- match(target_recall, c(0.90, 0.95, 0.99))
    base$index_candidate <- "hnsw"
    base$index_residency <- "resident_cpu"
    base$reference_residency <- if (storage_mode == "file_backed") {
        "file_backed"
    } else "resident_cpu"
    base$selected_ef_search <- as.integer(max(vapply(cases, function(x)
        as.double(x$cells[[tier]]$ef_search), numeric(1L))))
    base$conditional_index_build_seconds_range <- build
    base$conditional_build_peak_bytes_range <- peak
    base$index_build_capacity_budget_bytes <-
        peak[[2L]] / (1 - headroom_fraction)
    base$source_archive_sha256 <- source_archive_sha256
    base$input_sha256 <- input_sha256
    base$audit_sha256 <- audit_sha256
    base$calibration_rows <- sizes
    base$calibration_threads <- cpu_threads
    base$heldout_build_relative_error <-
        build_error
    base$heldout_memory_relative_error <-
        memory_error
    base$tuning_status <-
        "conditional_extrapolation_requires_live_recall_and_hardware_validation"
    base
}
