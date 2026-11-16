normalize_public_nn_options <- function(
    exclude_self, backend, method, metric, tuning, target_recall, output,
    distances
) {
    exclude_self <- normalize_scalar_logical_arg(
        exclude_self,
        "exclude_self",
        default = FALSE
    )
    backend <- normalize_nn_backend_arg(backend)
    method <- normalize_nn_method(method)
    tuning <- normalize_nn_tuning(tuning)
    target_recall <- normalize_hnsw_target_recall(target_recall)
    metric <- normalize_nn_metric(metric)
    output <- resolve_nn_output(output, distances)
    list(
        exclude_self = exclude_self, backend = backend, method = method,
        metric = metric, tuning = tuning, target_recall = target_recall,
        output = output
    )
}

prepare_public_nn_request <- function(
    data, points, k, exclude_self, backend, method, metric, tuning,
    target_recall, output, distances, n_threads, points_missing,
    requested_method_input
) {
    validate_dense_matrix_input(data, "data")
    if (!points_missing) {
        validate_dense_matrix_input(points, "points")
    }
    options <- normalize_public_nn_options(
        exclude_self, backend, method, metric, tuning, target_recall, output,
        distances
    )
    validate_public_nn_method_shape(data, options$method)
    self_query <- points_missing || identical(data, points)
    data_dim <- if (is_float32_matrix_input(data)) {
        float32_matrix_dims(data, "data")
    } else {
        dim(data)
    }
    resolved_backend <- resolve_public_nn_backend(
        options$backend,
        options$method,
        options$metric,
        n = data_dim[[1L]],
        p = data_dim[[2L]],
        k = k,
        self_query = self_query
    )
    c(list(
        data = data, points = points, k = k, n_threads = n_threads,
        points_missing = points_missing, self_query = self_query,
        resolved_backend = resolved_backend,
        requested_method_input = requested_method_input
    ), options)
}

public_nn_auto_selection <- function(request) {
    selection <- nn_auto_selection_metadata(
        data = request$data,
        points = request$points,
        points_missing = request$points_missing,
        k = request$k,
        requested_backend = request$backend,
        requested_method = request$method,
        resolved_backend = request$resolved_backend,
        metric = request$metric,
        tuning = request$tuning,
        exclude_self = request$exclude_self,
        target_recall = request$target_recall
    )
    if (is.null(selection) || !identical(request$method, "auto")) {
        return(selection)
    }
    if (identical(selection$predicted_device %||% NA_character_, "cpu")) {
        selection$auto_policy_status <-
            "calibration_informed_not_independently_validated"
        selection$auto_policy_evidence_scope <-
            "cpu_static_policy_experimental"
    } else if (
        identical(selection$predicted_device %||% NA_character_, "cuda")
    ) {
        selection$auto_policy_status <-
            "experimental_l40s_calibrated_cold_full_self_search"
        selection$auto_policy_evidence_scope <-
            "cuda_l40s_cold_full_self_search_experimental"
    }
    selection
}

nn_hardware_warning_enabled <- function() {
    value <- faissr_option("warn_hardware_extrapolation", TRUE)
    isTRUE(value)
}

nn_maybe_warn_auto_hardware <- function(selection, requested_method = "auto") {
    if (
        is.null(selection) ||
            !identical(normalize_nn_method(requested_method), "auto") ||
            !identical(selection$predicted_device %||% NA_character_, "cuda") ||
            identical(
                selection$hardware_evidence %||% NA_character_,
                "calibration_hardware_matched"
            ) ||
            !nn_hardware_warning_enabled()
    ) {
        return(invisible(FALSE))
    }
    runtime <- as.character(
        selection$runtime_hardware_model %||% "unidentified CUDA device"
    )[1L]
    if (is.na(runtime) || !nzchar(runtime)) {
        runtime <- "unidentified CUDA device"
    }
    key <- paste0("hardware_warning:", nn_normalize_hardware_name(runtime))
    if (exists(key, envir = .faissR_auto_hardware_cache, inherits = FALSE)) {
        return(invisible(FALSE))
    }
    assign(key, TRUE, envir = .faissR_auto_hardware_cache)
    warning(
        "faissR CUDA `method = \"auto\"` is applying its NVIDIA L40S-",
        "informed static route policy on ", runtime, ". Search results retain ",
        "the documented method semantics, but the selected route is not ",
        "validated as locally fastest on this hardware/provider stack. ",
        "Inspect `attr(result, \"auto_selection\")`, request an explicit ",
        "method, or use `tuning = \"pilot\"` for local parameter tuning ",
        "within the resolved method. Full local cross-method policy learning ",
        "is not implemented. Disable hardware extrapolation notices with ",
        "the corresponding `faissR` session option.",
        call. = FALSE
    )
    invisible(TRUE)
}

decorate_public_nn_result <- function(result, request, auto_selection) {
    attr(result, "requested_backend") <- request$backend
    attr(result, "requested_method") <- public_nn_method_label(request$method)
    result$requested_method_input <- request$requested_method_input
    attr(result, "requested_method_input") <- request$requested_method_input
    attr(result, "tuning") <- request$tuning
    attr(result, "target_recall") <- request$target_recall
    result$exclude_self <- request$exclude_self
    attr(result, "exclude_self") <- request$exclude_self
    result <- attach_nn_method_implementation_contract(
        result,
        requested_method = request$method,
        backend_used = result$backend_used %||%
            attr(result, "backend_used") %||%
            attr(result, "resolved_backend")
    )
    if (!is.null(auto_selection)) {
        attr(result, "auto_selection") <- auto_selection
        attr(result, "auto_policy_status") <-
            auto_selection$auto_policy_status %||% NA_character_
        attr(result, "auto_policy_evidence_scope") <-
            auto_selection$auto_policy_evidence_scope %||% NA_character_
    }
    finalize_nn_output(result, request$output)
}

execute_public_nn_request <- function(request) {
    auto_selection <- public_nn_auto_selection(request)
    nn_maybe_warn_auto_hardware(auto_selection, request$method)
    measure_resources <- isTRUE(faissr_option("measure_resources", FALSE))
    before <- if (measure_resources) nn_resource_snapshot() else NULL
    result <- nn_compute(
        request$data,
        request$points,
        request$k,
        request$resolved_backend,
        request$points_missing,
        exclude_self = request$exclude_self,
        n_threads = request$n_threads,
        metric = request$metric,
        tuning = request$tuning,
        target_recall = request$target_recall,
        output = request$output,
        auto_selection = auto_selection,
        requested_method = request$method
    )
    result <- decorate_public_nn_result(result, request, auto_selection)
    if (measure_resources) {
        result$resource_measurement <- list(
            scope = "process_before_after_public_nn_call",
            before = before,
            after = nn_resource_snapshot(),
            peak_during_call_measured = FALSE
        )
    }
    result
}

nn_resource_snapshot <- function() {
    ram <- tryCatch(
        gpuinfo::system_memory_info(),
        error = function(error) list(
            process_resident_memory_bytes = NA_real_,
            process_peak_resident_memory_bytes = NA_real_,
            status = "unavailable",
            reason_codes = conditionMessage(error)
        )
    )
    device <- if (isTRUE(faissr_option("measure_device_memory", FALSE))) {
        tryCatch(
            gpuinfo::gpu_memory_sample(),
            error = function(error) structure(
                list(reason = conditionMessage(error)),
                class = "faissR_device_sample_error"
            )
        )
    } else NULL
    list(
        process_resident_memory_bytes =
            ram$process_resident_memory_bytes %||% NA_real_,
        process_lifetime_peak_resident_memory_bytes =
            ram$process_peak_resident_memory_bytes %||% NA_real_,
        ram_status = ram$status %||% "unavailable",
        ram_reason_codes = ram$reason_codes %||% character(),
        device_sample = device
    )
}
