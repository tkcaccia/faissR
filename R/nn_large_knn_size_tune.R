# Private planning from one hardware, shape, and index-layout calibration.
nn_large_knn_size_tune <- function(
    n, p, k, backend, storage_mode, host_memory_bytes, observations,
    profile_sha256, fit_sizes, holdout_n, target_recall = 0.99,
    minimum_row_recall = 0.90, expected_query_batches = 1,
    device_memory_bytes = NA_real_, gpu_count = 0L,
    disk_free_bytes = NA_real_, reference_shard_rows = NULL,
    cuda_probe_limit = 2048L
) {
    plan <- nn_large_knn_resource_plan(
        n, p, k, backend, storage_mode, host_memory_bytes,
        device_memory_bytes = device_memory_bytes, gpu_count = gpu_count,
        target_recall = target_recall, disk_free_bytes = disk_free_bytes
    )
    index_n <- if (is.null(reference_shard_rows)) n else reference_shard_rows
    if (!is.numeric(index_n) || length(index_n) != 1L ||
        !is.finite(index_n) || index_n != floor(index_n) ||
        index_n > n || index_n <= k ||
        (!is.null(reference_shard_rows) && storage_mode != "file_backed") ||
        !is.numeric(expected_query_batches) ||
        length(expected_query_batches) != 1L ||
        !is.finite(expected_query_batches) || expected_query_batches < 1 ||
        expected_query_batches != floor(expected_query_batches) ||
        !is.numeric(minimum_row_recall) || length(minimum_row_recall) != 1L ||
        !is.finite(minimum_row_recall) ||
        minimum_row_recall < 0 || minimum_row_recall > 1 ||
        !nn_large_knn_hash(profile_sha256)) {
        stop("Invalid large KNN tuning request.", call. = FALSE)
    }
    nn_cuda_ivf_validate_sizes(index_n, fit_sizes, holdout_n)
    if (is.null(fit_sizes) || is.null(holdout_n)) {
        stop("Declare fit sizes and a separate held-out size.",
             call. = FALSE)
    }
    nn_cuda_ivf_validate_anchors(fit_sizes, index_n, fit_sizes, holdout_n)
    if (!is.numeric(cuda_probe_limit) || length(cuda_probe_limit) != 1L ||
        !is.finite(cuda_probe_limit) || cuda_probe_limit < 1L ||
        cuda_probe_limit > 65536L ||
        cuda_probe_limit != floor(cuda_probe_limit)) {
        stop("Invalid CUDA probe limit.", call. = FALSE)
    }
    plan$tuning_status <- "no_matching_calibration"
    plan$profile_sha256 <- profile_sha256
    plan$tuned_reference_rows <- index_n
    plan$reference_shards <- ceiling(n / index_n)
    plan$expected_query_batches <- expected_query_batches
    plan$minimum_row_recall <- minimum_row_recall
    plan$recall_at_target <- NA_real_
    plan$projected_full_self_search_seconds <- NA_real_
    plan$settings <- NULL
    plan$index_candidate <- if (backend == "cpu") "hnsw" else "ivf_flat"
    plan$ivf_limits <- if (backend == "metal") {
        c(lists = 1024L, probes = 512L)
    } else if (backend == "cuda") {
        c(lists = 65536L, probes = as.integer(cuda_probe_limit))
    } else NULL
    if (backend == "metal" && (p > 1024 || k > 128)) {
        plan$tuning_status <- "shape_outside_metal_ivf_limits"
        return(plan)
    }
    rows <- nn_large_knn_size_rows(observations, plan, fit_sizes, holdout_n)
    if (!nrow(rows)) return(plan)
    nn_large_knn_size_select(plan, rows, fit_sizes, holdout_n)
}

nn_large_knn_hash <- function(x) {
    is.character(x) && length(x) == 1L && !is.na(x) &&
        grepl("^[0-9a-f]{64}$", x)
}

nn_large_knn_size_rows <- function(observations, plan, fit_sizes, holdout_n) {
    if (is.null(observations)) return(data.frame())
    identity <- c("p", "k", "backend", "storage_mode", "gpu_count",
                  "profile_sha256", "layout")
    design <- c("tuning_rows_sha256", "audit_rows_sha256", "reference_scope",
                "timing_scope", "query_count")
    measured <- c("mean_recall", "min_row_recall", "build_seconds",
                  "search_seconds", "peak_host_bytes")
    settings <- if (plan$backend == "cpu") {
        c("m", "ef_construction", "ef_search")
    } else c("nlist", "nprobe")
    if (plan$backend != "cpu") measured <- c(measured, "peak_device_bytes")
    required <- c(identity, design, "n", "audit_pass", measured, settings)
    if (!is.data.frame(observations) || !is.numeric(observations$n) ||
        !all(required %in% names(observations))) {
        stop("Missing route calibration columns.", call. = FALSE)
    }
    layout <- if (plan$reference_shards > 1) {
        "reference_shard"
    } else "single_logical_index"
    match <- observations$layout == layout &
        observations$n %in% c(fit_sizes, holdout_n)
    for (field in setdiff(identity, "layout")) {
        match <- match & observations[[field]] == plan[[field]]
    }
    rows <- observations[which(match), required, drop = FALSE]
    if (!nrow(rows)) return(rows)
    pass <- rows$audit_pass
    valid_design <- nn_large_knn_size_design(rows, design)
    if (!valid_design || !is.logical(pass) || anyNA(pass) ||
        anyDuplicated(rows[c("n", settings)]) ||
        !all(vapply(rows[c(measured, settings)], is.numeric, logical(1L))) ||
        any(!is.finite(as.matrix(rows[pass, measured, drop = FALSE]))) ||
        any(as.matrix(rows[pass, setdiff(measured,
            c("mean_recall", "min_row_recall")), drop = FALSE]) <= 0) ||
        any(rows$mean_recall[pass] < 0 | rows$mean_recall[pass] > 1 |
            rows$min_row_recall[pass] < 0 |
            rows$min_row_recall[pass] > rows$mean_recall[pass]) ||
        !nn_large_knn_size_settings_valid(rows, plan, settings)) {
        stop("Calibration rows have invalid timings, settings, or scope.",
             call. = FALSE)
    }
    rows
}

nn_large_knn_size_design <- function(rows, design) {
    !anyNA(rows[design]) &&
        nrow(unique(rows[design])) == 1L &&
        nn_large_knn_hash(rows$tuning_rows_sha256[[1L]]) &&
        nn_large_knn_hash(rows$audit_rows_sha256[[1L]]) &&
        rows$tuning_rows_sha256[[1L]] != rows$audit_rows_sha256[[1L]] &&
        rows$reference_scope[[1L]] == "full_database_exact" &&
        rows$timing_scope[[1L]] == "index_build_plus_fixed_query" &&
        is.numeric(rows$query_count) && is.finite(rows$query_count[[1L]]) &&
        rows$query_count[[1L]] == floor(rows$query_count[[1L]]) &&
        rows$query_count[[1L]] >= 2L
}

nn_large_knn_size_settings_valid <- function(rows, plan, settings) {
    params <- as.matrix(rows[settings])
    if (any(!is.finite(params) | params != floor(params) |
            params < 1 | params > .Machine$integer.max)) return(FALSE)
    if (plan$backend == "cpu") {
        return(all(rows$m >= 2 & rows$ef_construction >= rows$m &
                   rows$ef_search >= plan$k + 1))
    }
    cap <- plan$ivf_limits
    all(rows$nlist <= pmin(rows$n, cap[["lists"]]) &
        rows$nprobe <= pmin(rows$nlist, cap[["probes"]]))
}

nn_large_knn_size_select <- function(plan, rows, fit_sizes, holdout_n) {
    good <- rows$audit_pass & !is.na(rows$mean_recall) &
        rows$mean_recall >= plan$target_recall &
        rows$min_row_recall >= plan$minimum_row_recall
    plan$attempted_calibration_cells <- nrow(rows)
    plan$failed_calibration_cells <- sum(!rows$audit_pass)
    plan$recall_miss_cells <- sum(rows$audit_pass & !good)
    plan$query_count <- rows$query_count[[1L]]
    fit <- rows[good & rows$n %in% fit_sizes, , drop = FALSE]
    plan$tuning_status <- "no_passing_setting_at_every_fit_size"
    if (!setequal(fit$n, fit_sizes)) return(plan)
    chosen <- nn_large_knn_size_anchors(fit, fit_sizes, plan)
    if (is.null(chosen)) {
        plan$tuning_status <- "no_shared_hnsw_build_settings_at_fit_sizes"
        return(plan)
    }
    plan$anchor_rows <- chosen
    predicted <- nn_large_knn_size_settings(chosen, holdout_n, plan)
    plan$holdout <- nn_large_knn_size_holdout(
        plan, rows, chosen, holdout_n, predicted)
    plan$tuning_status <- plan$holdout$status
    if (plan$holdout$status != "interpolation_pass") return(plan)
    plan$settings <- nn_large_knn_size_settings(
        chosen, plan$tuned_reference_rows, plan)
    plan$settings_applied_automatically <- FALSE
    plan$parameter_limits_reached <- if (plan$backend == "cpu") {
        plan$settings$ef_search == plan$tuned_reference_rows
    } else {
        cap <- plan$ivf_limits
        plan$settings$nlist == cap[["lists"]] ||
            plan$settings$nprobe == cap[["probes"]]
    }
    plan$tuning_status <- "conditional_settings_require_live_recall_audit"
    nn_large_knn_size_capacity(plan, chosen)
}

nn_large_knn_size_anchors <- function(fit, fit_sizes, plan) {
    fit$selection_seconds <- fit$build_seconds +
        plan$expected_query_batches * fit$search_seconds
    fit <- fit[order(fit$n, fit$selection_seconds, -fit$min_row_recall,
                     -fit$mean_recall), , drop = FALSE]
    if (plan$backend != "cpu") return(fit[!duplicated(fit$n), ])
    groups <- split(fit, list(fit$m, fit$ef_construction), drop = TRUE)
    groups <- Filter(function(group) setequal(group$n, fit_sizes), groups)
    if (!length(groups)) return(NULL)
    fastest <- fit$selection_seconds[match(sort(fit_sizes), fit$n)]
    anchors <- lapply(groups, function(group) group[!duplicated(group$n), ])
    scores <- vapply(anchors, function(group) {
        mean(log(group$selection_seconds / fastest))
    }, numeric(1L))
    anchors[[which.min(scores)]]
}

nn_large_knn_size_settings <- function(chosen, n, plan) {
    if (plan$backend == "cpu") {
        value <- nn_cuda_ivf_project_value(chosen$n, chosen$ef_search, n)
        if (n > max(chosen$n)) value <- max(value, max(chosen$ef_search))
        return(list(m = chosen$m[[1L]],
            ef_construction = chosen$ef_construction[[1L]],
            ef_search = as.integer(min(n, max(plan$k + 1, ceiling(value))))))
    }
    cap <- plan$ivf_limits
    nn_cuda_ivf_project_settings(chosen, n, cap[["lists"]], cap[["probes"]])
}

nn_large_knn_size_holdout <- function(plan, rows, chosen, n, settings) {
    held <- rows[rows$n == n, , drop = FALSE]
    for (field in names(settings)) {
        held <- held[held[[field]] == settings[[field]], , drop = FALSE]
    }
    result <- list(n = n, settings = settings,
                   status = "predicted_holdout_setting_not_audited")
    if (!nrow(held)) return(result)
    result$status <- "holdout_audit_failed"
    if (!held$audit_pass) return(result)
    result$mean_recall <- held$mean_recall
    result$min_row_recall <- held$min_row_recall
    result$status <- "holdout_recall_miss"
    if (held$mean_recall < plan$target_recall ||
        held$min_row_recall < plan$minimum_row_recall) return(result)
    tolerance <- c(build_seconds = 0.30, search_seconds = 0.30,
                   peak_host_bytes = 0.25)
    if (plan$backend != "cpu") tolerance["peak_device_bytes"] <- 0.25
    result$relative_error <- vapply(names(tolerance), function(field) {
        estimate <- nn_cuda_ivf_project_value(chosen$n, chosen[[field]], n)
        abs(estimate / held[[field]] - 1)
    }, numeric(1L))
    result$relative_error_tolerance <- tolerance
    result$status <- if (all(result$relative_error <= tolerance)) {
        "interpolation_pass"
    } else "holdout_prediction_error"
    result
}

nn_large_knn_size_capacity <- function(plan, chosen) {
    n <- plan$tuned_reference_rows
    predict <- function(field) {
        nn_cuda_ivf_project_value(chosen$n, chosen[[field]], n)
    }
    plan$peak_host_bytes_projection <- predict("peak_host_bytes")
    if (plan$storage_mode == "resident") {
        plan$peak_host_bytes_projection <- max(
            plan$peak_host_bytes_projection, plan$input_float32_bytes)
    }
    plan$host_projection_fits <- plan$peak_host_bytes_projection <=
        plan$host_memory_bytes * (1 - plan$device_headroom_fraction)
    plan$peak_device_bytes_projection <- NA_real_
    plan$device_projection_fits <- NA
    if (plan$backend != "cpu") {
        plan$peak_device_bytes_projection <- predict("peak_device_bytes")
        if (plan$backend == "cuda") {
            raw <- (4 * plan$p + 8) * n / plan$gpu_count
            plan$peak_device_bytes_projection <- max(
                plan$peak_device_bytes_projection, raw)
        }
        plan$device_projection_fits <- plan$peak_device_bytes_projection <=
            plan$device_memory_bytes * (1 - plan$device_headroom_fraction)
    }
    # Bounds apply to the declared index layout and fixed-query workload.
    plan$fixed_query_projection_admitted <- !any(c(
        plan$host_projection_fits, plan$device_projection_fits) == FALSE,
        na.rm = TRUE)
    plan$projected_index_build_seconds <- if (
        plan$fixed_query_projection_admitted) predict("build_seconds")
        else NA_real_
    plan$projected_fixed_query_seconds <- if (
        plan$fixed_query_projection_admitted) predict("search_seconds")
        else NA_real_
    plan
}
