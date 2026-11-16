# Private resident CUDA capacity estimate from matched, audited IVF peaks.
# It projects memory only. A held-out size checks interpolation, not 1B recall,
# topology, full-output support, or a completed index build.
nn_large_knn_capacity_from_audits <- function(
    n, p, k, backend, storage_mode, host_memory_bytes,
    device_memory_bytes, gpu_count, observations,
    target_recall = 0.99, device_headroom_fraction = 0.20,
    max_holdout_relative_error = 0.25
) {
    plan <- nn_large_knn_resource_plan(
        n = n, p = p, k = k, backend = backend,
        storage_mode = storage_mode, host_memory_bytes = host_memory_bytes,
        device_memory_bytes = device_memory_bytes, gpu_count = gpu_count,
        target_recall = target_recall,
        device_headroom_fraction = device_headroom_fraction
    )
    if (backend != "cuda" || storage_mode != "resident") {
        stop("Audited capacity model requires resident CUDA.", call. = FALSE)
    }
    fields <- c(
        "role", "n", "p", "k", "gpu_count", "mode", "nlist",
        "nprobe", "peak_device_bytes", "source_sha", "input_sha256",
        "query_rows_sha256", "audit_pass", "reference_scope"
    )
    if (!is.data.frame(observations) || nrow(observations) != 3L ||
        !all(fields %in% names(observations)) ||
        !is.numeric(max_holdout_relative_error) ||
        length(max_holdout_relative_error) != 1L ||
        !is.finite(max_holdout_relative_error) ||
        max_holdout_relative_error < 0 ||
        max_holdout_relative_error >= 1 ||
        !is.logical(observations$audit_pass) ||
        anyNA(observations[fields]) ||
        !all(observations$audit_pass) ||
        !identical(sort(as.character(observations$role)),
                   c("fit", "fit", "holdout"))) {
        stop("Complete audited fit and holdout memory rows are required.",
             call. = FALSE)
    }
    numeric_fields <- c("n", "p", "k", "gpu_count", "nlist",
                        "nprobe", "peak_device_bytes")
    if (any(!vapply(observations[numeric_fields], is.numeric,
                    logical(1L))) ||
        any(!vapply(observations[numeric_fields], function(value)
            all(is.finite(value)), logical(1L))) ||
        any(observations$n < 2 | observations$n != floor(observations$n)) ||
        any(observations$peak_device_bytes <= 0) ||
        any(observations$p != p | observations$k != k) ||
        length(unique(observations$gpu_count)) != 1L ||
        observations$gpu_count[[1L]] < 1 ||
        any(observations$gpu_count != floor(observations$gpu_count)) ||
        any(observations$mode != "shard") ||
        length(unique(observations$nlist)) != 1L ||
        length(unique(observations$nprobe)) != 1L ||
        any(observations$nlist < 1 | observations$nlist > observations$n) ||
        any(observations$nprobe < 1 |
            observations$nprobe > observations$nlist) ||
        any(observations$reference_scope != "full_database_exact") ||
        length(unique(observations$source_sha)) != 1L ||
        length(unique(observations$input_sha256)) != 1L ||
        length(unique(observations$query_rows_sha256)) != 1L ||
        !grepl("^[0-9a-f]{40}$", observations$source_sha[[1L]]) ||
        !grepl("^[0-9a-f]{64}$", observations$input_sha256[[1L]]) ||
        !grepl("^[0-9a-f]{64}$", observations$query_rows_sha256[[1L]])) {
        stop("Audited memory rows mix source, settings, or scope.",
             call. = FALSE)
    }
    fit <- observations[observations$role == "fit", , drop = FALSE]
    fit <- fit[order(fit$n), , drop = FALSE]
    held <- observations[observations$role == "holdout", , drop = FALSE]
    if (fit$n[[1L]] == fit$n[[2L]] ||
        held$n[[1L]] <= fit$n[[1L]] ||
        held$n[[1L]] >= fit$n[[2L]] ||
        n <= fit$n[[2L]]) {
        stop("Distinct bracketed fit sizes and a larger target are required.",
             call. = FALSE)
    }
    slope <- diff(fit$peak_device_bytes) / diff(fit$n)
    intercept <- fit$peak_device_bytes[[1L]] - slope * fit$n[[1L]]
    predicted_holdout <- intercept + slope * held$n[[1L]]
    error <- abs(held$peak_device_bytes[[1L]] - predicted_holdout) /
        predicted_holdout
    if (!is.finite(slope) || slope <= 0 ||
        !is.finite(intercept) || intercept < 0 ||
        !is.finite(error) || error > max_holdout_relative_error) {
        stop("Held-out memory calibration failed.", call. = FALSE)
    }
    reference_gpus <- fit$gpu_count[[1L]]
    capacity <- device_memory_bytes * (1 - device_headroom_fraction)
    minimum_gpus <- function(factor) {
        remaining <- capacity / factor - intercept
        if (remaining <= 0) return(Inf)
        ceiling(slope * n * reference_gpus / remaining)
    }
    peak <- intercept + slope * n * reference_gpus / gpu_count
    plan$calibration_status <-
        "holdout_checked_memory_scenario_not_1b_route_validation"
    plan$calibration_source_sha <- observations$source_sha[[1L]]
    plan$calibration_input_sha256 <- observations$input_sha256[[1L]]
    plan$calibration_query_rows_sha256 <-
        observations$query_rows_sha256[[1L]]
    plan$calibration_nlist <- as.integer(fit$nlist[[1L]])
    plan$calibration_nprobe <- as.integer(fit$nprobe[[1L]])
    plan$calibration_reference_gpu_count <-
        as.integer(reference_gpus)
    plan$holdout_memory_relative_error <- error
    plan$estimated_peak_device_bytes <- peak
    plan$minimum_cuda_gpus_affine_peak <- minimum_gpus(1)
    plan$minimum_cuda_gpus_affine_with_holdout_error <-
        minimum_gpus(1 + error)
    plan$requested_gpu_count_meets_calibrated_headroom <-
        peak * (1 + error) <= capacity
    plan
}
