finish_float32_direct_result <- function(result, out) {
    result$input_type <- out$input_type %||% "float32"
    result$input_layout <- out$input_layout %||% NA_character_
    result$input_owns_data <- isTRUE(out$input_owns_data)
    result$float32_compatibility_conversion <- isTRUE(
        out$float32_compatibility_conversion %||% FALSE
    )
    attr(result, "input_type") <- result$input_type
    attr(result, "input_layout") <- result$input_layout
    attr(result, "input_owns_data") <- result$input_owns_data
    attr(
        result,
        "float32_compatibility_conversion"
    ) <- result$float32_compatibility_conversion
    attach_gpu_residency_metadata(result, out)
}

.faissR_fitted_nn_index_cache <- new.env(parent = emptyenv())
.faissR_fitted_nn_index_cache$.keys <- character()
.faissR_auto_hardware_cache <- new.env(parent = emptyenv())

.normalized_similarity_distance_transform <- paste0(
    "normalized_euclidean_squared_over_2_",
    "to_1_minus_similarity"
)
.fastscan_normalized_strategy <- paste0(
    "faiss_IndexIVFPQFastScan_RefineFlat_",
    "normalized_L2"
)
.validation_replicate_rule <- paste0(
    "all_prespecified_validation_",
    "replicates"
)
.gpu_exact_residency_constraint <- paste0(
    "gpu_resident_output_currently_",
    "exact_family_only"
)

fitted_nn_index_cache_enabled <- function() {
    isTRUE(faissr_option("cache_fitted_nn_indexes", TRUE)) &&
        isTRUE(faiss_available())
}

with_faiss_query_batch_size <- function(params, expr) {
    size <- faissr_quiet_warning(as.integer(
        params$faiss_query_batch_size %||% NA_integer_
    ))
    if (length(size) != 1L || is.na(size) || !is.finite(size) || size < 1L) {
        return(force(expr))
    }
    env_name <- "FAISSR_FAISS_QUERY_BATCH_SIZE"
    old <- Sys.getenv(env_name, unset = NA_character_)
    if (is.na(old)) {
        on.exit(Sys.unsetenv(env_name), add = TRUE)
    } else {
        on.exit(
            do.call(Sys.setenv, stats::setNames(list(old), env_name)),
            add = TRUE
        )
    }
    do.call(Sys.setenv, stats::setNames(list(as.character(size)), env_name))
    force(expr)
}

with_faiss_gpu_runtime <- function(params, expr) {
    if (!is.list(params)) {
        return(force(expr))
    }
    old <- Sys.getenv(
        c(
            "FAISSR_FAISS_GPU_QUERY_BATCH_SIZE",
            "FAISSR_FAISS_GPU_REUSE_RESOURCES"
        ),
        unset = NA_character_
    )
    on.exit(
        {
            for (name in names(old)) {
                if (is.na(old[[name]])) {
                    Sys.unsetenv(name)
                } else {
                    do.call(
                        Sys.setenv,
                        stats::setNames(list(old[[name]]), name)
                    )
                }
            }
        },
        add = TRUE
    )
    size <- faissr_quiet_warning(as.integer(
        params$faiss_gpu_query_batch_size %||% NA_integer_
    ))
    if (length(size) == 1L && !is.na(size) && is.finite(size) && size >= 1L) {
        set_env_var("FAISSR_FAISS_GPU_QUERY_BATCH_SIZE", size)
    }
    reuse <- params$faiss_gpu_reuse_resources %||% NA
    if (!is.na(reuse)) {
        set_env_var(
            "FAISSR_FAISS_GPU_REUSE_RESOURCES",
            if (isTRUE(reuse)) "1" else "0"
        )
    }
    force(expr)
}

with_cuvs_ivf_batch_size <- function(params, expr) {
    if (!is.list(params)) {
        return(force(expr))
    }
    size <- params$cuvs_ivf_batch_size %||%
        params$tuning$cuvs_ivf_batch_size %||%
        params$ivf$cuvs_ivf_batch_size %||%
        NA_integer_
    size <- faissr_quiet_warning(as.integer(size))
    if (length(size) != 1L || is.na(size) || !is.finite(size) || size < 1L) {
        return(force(expr))
    }
    old <- Sys.getenv("FAISSR_CUVS_IVF_BATCH_SIZE", unset = NA_character_)
    on.exit(
        {
            if (is.na(old)) {
                Sys.unsetenv("FAISSR_CUVS_IVF_BATCH_SIZE")
            } else {
                set_env_var("FAISSR_CUVS_IVF_BATCH_SIZE", old)
            }
        },
        add = TRUE
    )
    set_env_var("FAISSR_CUVS_IVF_BATCH_SIZE", size)
    force(expr)
}

cpu_exact_params <- function(
    n,
    p,
    k,
    metric = "euclidean",
    target_recall = 0.99
) {
    nn_tune_cpu_exact_cpp(
        as.integer(n),
        faissr_quiet_warning(as.integer(p)),
        as.integer(k),
        normalize_nn_metric(metric),
        as.numeric(normalize_hnsw_target_recall(target_recall))
    )
}

cuda_exact_params <- function(
    n,
    p,
    k,
    metric = "euclidean",
    target_recall = 0.99
) {
    nn_tune_cuda_exact_cpp(
        as.integer(n),
        faissr_quiet_warning(as.integer(p)),
        as.integer(k),
        normalize_nn_metric(metric),
        as.numeric(normalize_hnsw_target_recall(target_recall))
    )
}

cuda_flat_params <- function(
    n,
    p,
    k,
    metric = "euclidean",
    target_recall = 0.99
) {
    nn_tune_cuda_flat_cpp(
        as.integer(n),
        faissr_quiet_warning(as.integer(p)),
        as.integer(k),
        normalize_nn_metric(metric),
        as.numeric(normalize_hnsw_target_recall(target_recall))
    )
}

cuda_bruteforce_params <- function(
    n,
    p,
    k,
    metric = "euclidean",
    target_recall = 0.99
) {
    nn_tune_cuda_bruteforce_cpp(
        as.integer(n),
        faissr_quiet_warning(as.integer(p)),
        as.integer(k),
        normalize_nn_metric(metric),
        as.numeric(normalize_hnsw_target_recall(target_recall))
    )
}

cpu_flat_params <- function(
    n,
    p,
    k,
    metric = "euclidean",
    target_recall = 0.99
) {
    nn_tune_cpu_flat_cpp(
        as.integer(n),
        faissr_quiet_warning(as.integer(p)),
        as.integer(k),
        normalize_nn_metric(metric),
        as.numeric(normalize_hnsw_target_recall(target_recall))
    )
}

cpu_bruteforce_params <- function(
    n,
    p,
    k,
    metric = "euclidean",
    target_recall = 0.99
) {
    nn_tune_cpu_bruteforce_cpp(
        as.integer(n),
        faissr_quiet_warning(as.integer(p)),
        as.integer(k),
        normalize_nn_metric(metric),
        as.numeric(normalize_hnsw_target_recall(target_recall))
    )
}

cpu_flatlike_params <- function(
    n,
    p,
    k,
    metric = "euclidean",
    target_recall = 0.99,
    requested_method = NULL
) {
    method <- public_nn_method_label(requested_method %||% "exact")
    if (identical(method, "flat")) {
        return(cpu_flat_params(
            n,
            p,
            k,
            metric = metric,
            target_recall = target_recall
        ))
    }
    if (identical(method, "bruteforce")) {
        return(cpu_bruteforce_params(
            n,
            p,
            k,
            metric = metric,
            target_recall = target_recall
        ))
    }
    cpu_exact_params(n, p, k, metric = metric, target_recall = target_recall)
}

cuda_flatlike_params <- function(
    n,
    p,
    k,
    metric = "euclidean",
    target_recall = 0.99,
    requested_method = NULL
) {
    method <- public_nn_method_label(requested_method %||% "exact")
    if (identical(method, "flat")) {
        return(cuda_flat_params(
            n,
            p,
            k,
            metric = metric,
            target_recall = target_recall
        ))
    }
    if (identical(method, "bruteforce")) {
        return(cuda_bruteforce_params(
            n,
            p,
            k,
            metric = metric,
            target_recall = target_recall
        ))
    }
    cuda_exact_params(n, p, k, metric = metric, target_recall = target_recall)
}

nn_gpu_tuning_params_for_method <- function(
    n,
    p,
    k,
    method,
    metric = "euclidean",
    target_recall = 0.99
) {
    method <- public_nn_method_label(method %||% "exact")
    metric <- normalize_nn_metric(metric)
    target <- normalize_hnsw_target_recall(target_recall)
    out <- tryCatch(
        select_gpu_tuning_params(n, p, k, method, metric, target),
        error = function(error) list(tuning_error = conditionMessage(error))
    )
    if (!is.list(out)) {
        return(out)
    }
    out$tuning_backend <- out$tuning_backend %||% "cuda"
    out$tuning_method <- out$tuning_method %||% method
    out$tuning_metric <- out$tuning_metric %||% metric
    out$target_recall <- out$target_recall %||% target
    out
}

select_gpu_tuning_params <- function(n, p, k, method, metric, target) {
    switch(
        method,
        exact = cuda_exact_params(n, p, k, metric, target),
        flat = cuda_flat_params(n, p, k, metric, target),
        bruteforce = cuda_bruteforce_params(n, p, k, metric, target),
        cagra = cuvs_cagra_params(n, k, p, metric, target),
        hnsw = cuvs_hnsw_params(n, k, p, metric, target),
        ivf = cuda_ivf_params(n, p, k, metric, target),
        ivfpq = cuda_ivfpq_tuning_metadata(n, p, k, metric, target),
        ivfpq_fastscan = ivfpq_fastscan_cuda_params(
            n,
            p,
            k,
            metric = metric,
            target_recall = target
        ),
        nndescent = cuvs_nndescent_params(n, p, k, metric, target),
        nsg = cuda_nsg_params(n, p, k, metric, target),
        vamana = vamana_params(n, p, k, metric, "cuda", target),
        NULL
    )
}

cuda_ivfpq_tuning_metadata <- function(n, p, k, metric, target) {
    ivf <- cuda_ivf_params(n, p, k, metric, target)
    pq <- cuvs_ivfpq_params(p, n = n)
    list(
        tuning_backend = "cuda",
        tuning_method = "ivfpq",
        tuning_metric = metric,
        target_recall = target,
        ivf = ivf,
        pq = pq,
        tuning_rule = ivf$tuning_rule %||%
            pq$tuning_rule %||%
            "cuda_ivfpq_auto",
        tuning_shape_group = ivf$tuning_shape_group %||% NA_character_,
        tuning_k_bucket = ivf$tuning_k_bucket %||% NA_integer_,
        tuning_target_recall_code = ivf$tuning_target_recall_code %||%
            NA_integer_,
        tuning_benchmark_basis = ivf$tuning_benchmark_basis %||%
            NA_character_,
        tuning_benchmark_target_met = ivf$tuning_benchmark_target_met %||%
            FALSE,
        tuning_benchmark_source = ivf$tuning_benchmark_source %||%
            NA_character_,
        tuning_source = ivf$tuning_source %||% "cpp"
    )
}

attach_cpu_exact_tuning <- function(
    result,
    params,
    output,
    n_threads,
    extra = NULL
) {
    if (!is.list(params)) {
        return(result)
    }
    meta <- exact_tuning_metadata(
        params, output, n_threads, accelerator = NULL, extra = extra
    )
    attach_exact_tuning_metadata(result, params, meta, "faiss")
}

attach_cuda_exact_tuning <- function(
    result,
    params,
    output,
    n_threads,
    extra = NULL
) {
    if (!is.list(params)) {
        return(result)
    }
    meta <- exact_tuning_metadata(
        params, output, n_threads, accelerator = "cuda", extra = extra
    )
    library <- if (identical(params$tuning_method, "bruteforce")) {
        "cuvs"
    } else {
        "faiss"
    }
    attach_exact_tuning_metadata(result, params, meta, library)
}

exact_tuning_metadata <- function(
    params, output, n_threads, accelerator = NULL, extra = NULL
) {
    method <- params$tuning_method %||% "exact"
    backend_label <- params$result_backend %||%
        params$resolved_backend %||%
        if (is.null(accelerator)) "faiss_flat_l2" else "faiss_gpu_flat_l2"
    resolved_default <- if (identical(method, "bruteforce") &&
        !is.null(accelerator)) {
        "cuda_cuvs_bruteforce"
    } else if (is.null(accelerator)) {
        "faiss_flat_l2"
    } else {
        "faiss_gpu_flat_l2"
    }
    batch_field <- if (is.null(accelerator)) {
        "faiss_query_batch_size"
    } else {
        "faiss_gpu_query_batch_size"
    }
    base <- list(
            strategy = paste0(backend_label, "_", method),
            backend = backend_label,
            resolved_backend = params$resolved_backend %||% resolved_default,
            method = method,
            exact = TRUE,
            exact_recall_by_construction = TRUE,
            expected_recall_at_k = as.numeric(params$expected_recall_at_k %||%
                1),

            metric = params$tuning_metric %||% NA_character_,
            actual_n_threads = as.integer(n_threads),
            recommended_n_threads = as.integer(params$recommended_n_threads %||%
                NA_integer_),
            actual_output = output,
            recommended_output = params$recommended_output %||% NA_character_,
            batch_size = as.integer(params[[batch_field]] %||% NA_integer_),
            cache_fitted_indexes = if (is.null(accelerator)) {
                isTRUE(params$cache_fitted_indexes)
            } else FALSE
    )
    if (!is.null(accelerator)) {
        base$accelerator <- accelerator
        base$faiss_gpu_reuse_resources <- isTRUE(
            params$faiss_gpu_reuse_resources)
    }
    c(base, nn_tuning_metadata(params), extra %||% list())
}

attach_exact_tuning_metadata <- function(result, params, meta, library) {
    method <- params$tuning_method %||% "exact"
    attr_name <- switch(
        method, flat = "flat_tuning", bruteforce = "bruteforce_tuning",
        "exact_tuning"
    )
    meta$library <- library
    batch_name <- if (identical(meta$accelerator %||% NULL, "cuda")) {
        "faiss_gpu_query_batch_size"
    } else {
        "faiss_query_batch_size"
    }
    meta[[batch_name]] <- meta$batch_size
    meta$batch_size <- NULL
    attr(result, attr_name) <- meta
    provider <- attr(result, "faiss", exact = TRUE) %||% list()
    attr(result, "faiss") <- c(
        provider, meta[setdiff(names(meta), names(provider))]
    )
    result
}

fitted_nn_index_cache_limit <- function() {
    value <- faissr_quiet_warning(as.integer(faissr_option(
        "cache_fitted_nn_indexes_max_entries",
        2L
    )))
    if (
        length(value) != 1L || is.na(value) || !is.finite(value) || value < 0L
    ) {
        return(2L)
    }
    value
}

fitted_nn_index_cache_prune <- function() {
    limit <- fitted_nn_index_cache_limit()
    keys <- .faissR_fitted_nn_index_cache$.keys
    if (limit < 1L) {
        rm(
            list = setdiff(
                ls(.faissR_fitted_nn_index_cache, all.names = TRUE),
                ".keys"
            ),
            envir = .faissR_fitted_nn_index_cache
        )
        .faissR_fitted_nn_index_cache$.keys <- character()
        return(invisible(NULL))
    }
    while (length(keys) > limit) {
        old <- keys[[1L]]
        if (
            exists(old, envir = .faissR_fitted_nn_index_cache, inherits = FALSE)
        ) {
            rm(list = old, envir = .faissR_fitted_nn_index_cache)
        }
        keys <- keys[-1L]
    }
    .faissR_fitted_nn_index_cache$.keys <- keys
    invisible(NULL)
}

fitted_nn_index_cache_drop <- function(key) {
    if (exists(key, envir = .faissR_fitted_nn_index_cache, inherits = FALSE)) {
        rm(list = key, envir = .faissR_fitted_nn_index_cache)
    }
    .faissR_fitted_nn_index_cache$.keys <- setdiff(
        .faissR_fitted_nn_index_cache$.keys,
        key
    )
    invisible(NULL)
}

fitted_nn_index_cache_get <- function(key) {
    if (!exists(key, envir = .faissR_fitted_nn_index_cache, inherits = FALSE)) {
        return(NULL)
    }
    entry <- get(key, envir = .faissR_fitted_nn_index_cache, inherits = FALSE)
    entry$cache_hit <- TRUE
    keys <- .faissR_fitted_nn_index_cache$.keys
    .faissR_fitted_nn_index_cache$.keys <- c(setdiff(keys, key), key)
    entry
}

fitted_nn_index_cache_set <- function(key, entry) {
    if (fitted_nn_index_cache_limit() < 1L) {
        return(invisible(NULL))
    }
    assign(key, entry, envir = .faissR_fitted_nn_index_cache)
    keys <- .faissR_fitted_nn_index_cache$.keys
    .faissR_fitted_nn_index_cache$.keys <- c(setdiff(keys, key), key)
    fitted_nn_index_cache_prune()
    invisible(NULL)
}

fitted_nn_index_dims <- function(x) {
    if (is_float32_matrix_input(x)) {
        float32_matrix_dims(x, "data")
    } else {
        dim(x)
    }
}

fitted_nn_index_param_text <- function(...) {
    values <- unlist(list(...), recursive = TRUE, use.names = TRUE)
    if (!length(values)) {
        return("")
    }
    names(values) <- names(values) %||% rep("", length(values))
    values <- values[order(names(values), as.character(values))]
    paste(names(values), as.character(values), sep = "=", collapse = ";")
}

fitted_nn_index_cache_key <- function(
    data,
    kind,
    metric,
    n_threads,
    distance_output,
    params = NULL,
    pq = NULL
) {
    dims <- fitted_nn_index_dims(data)
    fingerprint <- matrix_fingerprint_cpp(data)
    paste(
        "faiss-fitted",
        kind,
        metric,
        paste(as.integer(dims), collapse = "x"),
        as.integer(n_threads),
        distance_output,
        fingerprint,
        fitted_nn_index_param_text(params = params, pq = pq),
        sep = "|"
    )
}

fitted_nn_index_kind <- function(backend, metric) {
    if (backend %in% c("faiss_flat_l2", "faiss_flat_ip")) {
        return("flat")
    }
    if (identical(backend, "faiss_hnsw")) {
        return("hnsw")
    }
    if (identical(backend, "faiss_ivf")) {
        return("ivf")
    }
    if (identical(backend, "faiss_ivfpq")) {
        return("ivfpq")
    }
    if (identical(backend, "faiss_ivfpq_fastscan")) {
        return("ivfpq_fastscan")
    }
    NA_character_
}

fitted_nn_index_build <- function(
    data,
    kind,
    metric,
    n_threads,
    params = NULL,
    pq = NULL
) {
    distance_output <- faiss_metric_distance_output_arg(metric)
    if (identical(kind, "hnsw")) {
        return(nn_faiss_hnsw_index_build_float32_cpp(
            data,
            as.integer(params$m),
            as.integer(params$ef_construction),
            as.integer(params$ef_search),
            faiss_metric_search_arg(metric),
            distance_output,
            as.integer(n_threads)
        ))
    }
    if (identical(kind, "ivfpq_fastscan")) {
        return(fitted_fastscan_index_build(
            data, kind, metric, n_threads, params, pq, distance_output
        ))
    }
    nn_faiss_index_build_float32_cpp(
        data,
        kind,
        as.integer(params$nlist %||% NA_integer_),
        as.integer(params$nprobe %||% NA_integer_),
        as.integer(pq$m %||% NA_integer_),
        as.integer(pq$nbits %||% NA_integer_),
        NA_integer_,
        NA_integer_,
        NA_integer_,
        NA_integer_,
        faiss_metric_search_arg(metric),
        distance_output,
        as.integer(n_threads)
    )
}

fitted_fastscan_index_build <- function(
    data, kind, metric, n_threads, params, pq, distance_output
) {
    nn_faiss_index_build_float32_cpp(
        data, kind,
        as.integer(params$nlist %||% NA_integer_),
        as.integer(params$nprobe %||% NA_integer_),
        as.integer(pq$m %||% NA_integer_),
        as.integer(pq$nbits %||% 4L),
        as.integer(params$refine_factor %||% NA_integer_),
        as.integer(params$bbs %||% NA_integer_),
        NA_integer_, NA_integer_, faiss_metric_search_arg(metric),
        distance_output, as.integer(n_threads)
    )
}

fitted_nn_index_get_or_build <- function(
    data,
    kind,
    metric,
    n_threads,
    params = NULL,
    pq = NULL
) {
    distance_output <- faiss_metric_distance_output_arg(metric)
    key <- fitted_nn_index_cache_key(
        data = data,
        kind = kind,
        metric = metric,
        n_threads = n_threads,
        distance_output = distance_output,
        params = params,
        pq = pq
    )
    entry <- fitted_nn_index_cache_get(key)
    if (!is.null(entry)) {
        return(entry)
    }
    entry <- list(
        index = fitted_nn_index_build(
            data,
            kind,
            metric,
            n_threads,
            params,
            pq
        ),
        kind = kind,
        metric = metric,
        params = params,
        pq = pq,
        cache_key = key,
        cache_hit = FALSE
    )
    fitted_nn_index_cache_set(key, entry)
    entry
}

fitted_nn_index_search_width <- function(kind, params, k) {
    if (kind %in% c("ivf", "ivfpq", "ivfpq_fastscan")) {
        return(as.integer(params$nprobe %||% NA_integer_))
    }
    NA_integer_
}

ivfpq_fastscan_fitted_params <- function(params) {
    c(
        params$ivf,
        list(
            refine_factor = as.integer(params$refine_factor),
            requested_refine_factor = as.integer(params$refine_factor),
            bbs = as.integer(params$bbs),
            requested_bbs = as.integer(params$bbs)
        ),
        params$tuning
    )
}

fitted_nn_index_result <- function(
    data, points, k, backend, result_backend = backend, self_query,
    exclude_self, metric, n_threads, output, params = NULL, pq = NULL,
    target_recall = 0.99, use_cache = NULL
) {
    if (!fitted_nn_cache_requested(use_cache)) return(NULL)
    kind <- fitted_nn_index_kind(backend, metric)
    if (is.na(kind) || !metric %in% c("euclidean",
        "inner_product")) return(NULL)
    entry <- fitted_nn_index_get_or_build(
        data, kind, metric, n_threads, params, pq
    )
    out <- search_fitted_nn_index(
        entry, points, k, exclude_self, kind, params, n_threads, output
    )
    if (is.null(out)) return(NULL)
    result <- finish_nn_result(
        out, result_backend, k, self_query,
        exact = identical(kind, "flat"), metric = metric
    )
    cache_meta <- fitted_nn_cache_metadata(out, entry, points)
    result <- finish_fitted_nn_kind(
        result, out, kind, metric, params, pq, target_recall,
        output, n_threads, cache_meta
    )
    finish_float32_direct_result(result, out)
}

fitted_nn_cache_requested <- function(use_cache) {
    if (is.null(use_cache)) fitted_nn_index_cache_enabled() else {
        isTRUE(use_cache) && fitted_nn_index_cache_enabled()
    }
}

search_fitted_nn_index <- function(
    entry, points, k, exclude_self, kind, params, n_threads, output
) {
    tryCatch(with_faiss_query_batch_size(params, {
        if (identical(kind, "hnsw")) {
            nn_faiss_hnsw_index_search_float32_cpp(
                entry$index, points, as.integer(k), isTRUE(exclude_self),
                as.integer(params$ef_search), as.integer(n_threads), output
            )
        } else {
            nn_faiss_index_search_float32_cpp(
                entry$index, points, as.integer(k), isTRUE(exclude_self),
                fitted_nn_index_search_width(kind, params, k),
                as.integer(n_threads), output
            )
        }
    }), error = function(error) {
        if (grepl(
            "pointer|externalptr|not valid|null", conditionMessage(error),
            ignore.case = TRUE
        )) {
            fitted_nn_index_cache_drop(entry$cache_key)
            return(NULL)
        }
        stop(error)
    })
}

fitted_nn_cache_metadata <- function(out, entry, points) {
    list(
        persistent_index_cache = TRUE,
        index_cache_hit = isTRUE(entry$cache_hit),
        index_cache_key = entry$cache_key,
        index_reused = TRUE,
        batch_query = isTRUE(out$batch_query),
        query_n = as.integer(
            out$query_n %||% fitted_nn_index_dims(points)[[1L]]
        ),
        query_call_count = as.integer(out$query_call_count %||% 1L)
    )
}

finish_fitted_nn_kind <- function(
    result, out, kind, metric, params, pq, target_recall,
    output, n_threads, cache_meta
) {
    switch(kind,
        flat = finish_fitted_flat(
            result, out, metric, params, output, n_threads, cache_meta
        ),
        hnsw = finish_fitted_hnsw(
            result, out, metric, params, target_recall, cache_meta
        ),
        ivf = finish_fitted_ivf(result, out, metric, params, cache_meta),
        ivfpq_fastscan = finish_fitted_ivfpq(
            result, out, metric, params, pq, cache_meta, TRUE
        ),
        finish_fitted_ivfpq(result, out, metric, params, pq, cache_meta, FALSE)
    )
}

finish_fitted_flat <- function(
    result, out, metric, params, output, n_threads, cache_meta
) {
    type <- if (identical(metric, "inner_product")) {
        "IndexFlatIPExternalPtr"
    } else "IndexFlatL2ExternalPtr"
    attr(result, "faiss") <- c(list(
        index_type = out$index_type %||% type, library = "faiss",
        backend = "cpu", metric = metric, input_type = "float32", exact = TRUE
    ), cache_meta)
    attach_cpu_exact_tuning(result, params, output, n_threads, cache_meta)
}

finish_fitted_hnsw <- function(
    result, out, metric, params, target_recall, cache_meta
) {
    approximation <- list(
        strategy = "faiss_IndexHNSWFlat", backend = "faiss_hnsw",
        library = "faiss", metric = metric, input_type = "float32",
        fitted_index = TRUE, m = as.integer(out$m),
        ef_construction = as.integer(out$ef_construction),
        ef_search = as.integer(out$ef_search),
        requested_m = as.integer(out$requested_m),
        requested_ef_construction = as.integer(out$requested_ef_construction),
        requested_ef_search = as.integer(out$requested_ef_search),
        hnsw_parameters_adjusted = isTRUE(out$hnsw_parameters_adjusted)
    )
    attr(result, "approximation") <- c(
        approximation, hnsw_tuning_metadata(params, target_recall), cache_meta
    )
    attr(result, "faiss") <- c(list(
        index_type = out$index_type %||% "IndexHNSWFlatExternalPtr",
        library = "faiss", backend = "cpu", metric = metric, exact = FALSE
    ), cache_meta)
    append_nn_tuning_metadata(result, params)
}

fitted_ivf_reuse_metadata <- function(out) {
    list(
        index_trained = isTRUE(out$index_trained),
        index_training_reused = isTRUE(out$index_training_reused),
        centroids_reused = isTRUE(out$centroids_reused),
        inverted_lists_reused = isTRUE(out$inverted_lists_reused),
        vectors_reused = isTRUE(out$vectors_reused),
        build_train_call_count = as.integer(
            out$build_train_call_count %||% NA_integer_
        ),
        search_train_call_count = as.integer(
            out$search_train_call_count %||% NA_integer_
        )
    )
}

fitted_ivf_parameters_adjusted <- function(out, params) {
    isTRUE(out$ivf_parameters_adjusted) ||
        !identical(
            as.integer(params$requested_nlist %||% out$requested_nlist),
            as.integer(out$nlist)
        ) ||
        !identical(
            as.integer(params$requested_nprobe %||% out$requested_nprobe),
            as.integer(out$nprobe)
        )
}

fitted_ivf_base_metadata <- function(out, params) {
    c(list(
        nlist = as.integer(out$nlist), nprobe = as.integer(out$nprobe),
        requested_nlist = as.integer(
            params$requested_nlist %||% out$requested_nlist
        ),
        requested_nprobe = as.integer(
            params$requested_nprobe %||% out$requested_nprobe
        )
    ), fitted_ivf_reuse_metadata(out), list(
        ivf_parameters_adjusted = fitted_ivf_parameters_adjusted(out, params)
    ))
}

finish_fitted_ivf <- function(result, out, metric, params, cache_meta) {
    attr(result, "approximation") <- c(list(
        strategy = "faiss_IndexIVFFlat", backend = "faiss_ivf",
        library = "faiss", metric = metric, input_type = "float32",
        fitted_index = TRUE
    ), fitted_ivf_base_metadata(out, params), cache_meta)
    attr(result, "faiss") <- c(list(
        index_type = out$index_type %||% "IndexIVFFlatExternalPtr",
        library = "faiss", backend = "cpu", metric = metric, exact = FALSE
    ), fitted_ivf_reuse_metadata(out), cache_meta)
    append_nn_tuning_metadata(result, params)
}

fitted_pq_reuse_metadata <- function(out) {
    list(
        pq_m = as.integer(out$pq_m), pq_nbits = as.integer(out$pq_nbits),
        requested_pq_m = as.integer(out$requested_pq_m),
        requested_pq_nbits = as.integer(out$requested_pq_nbits),
        pq_codebooks_reused = isTRUE(out$pq_codebooks_reused),
        pq_codes_reused = isTRUE(out$pq_codes_reused),
        pq_training_reused = isTRUE(out$pq_training_reused),
        build_pq_train_call_count = as.integer(
            out$build_pq_train_call_count %||% NA_integer_
        ),
        search_pq_train_call_count = as.integer(
            out$search_pq_train_call_count %||% NA_integer_
        )
    )
}

fitted_fastscan_metadata <- function(out, params) {
    list(
        ivfpq_fastscan = TRUE, fastscan = isTRUE(out$fastscan),
        refine = isTRUE(out$refine),
        refine_factor = as.integer(out$refine_factor),
        requested_refine_factor = as.integer(out$requested_refine_factor),
        bbs = as.integer(out$bbs),
            requested_bbs = as.integer(out$requested_bbs),
        refine_parameters_adjusted = !identical(
            as.integer(params$requested_refine_factor %||%
                out$requested_refine_factor), as.integer(out$refine_factor)
        ) || !identical(
            as.integer(params$requested_bbs %||% out$requested_bbs),
            as.integer(out$bbs)
        )
    )
}

fitted_pq_faiss_metadata <- function(out, metric, cache_meta, fastscan) {
    type <- if (fastscan) {
        "IndexIVFPQFastScanExternalPtr"
    } else "IndexIVFPQExternalPtr"
    c(list(
        index_type = out$index_type %||% type, library = "faiss",
        backend = "cpu", metric = metric, exact = FALSE,
        fastscan = if (fastscan) isTRUE(out$fastscan) else NULL,
        pq_codebooks_reused = isTRUE(out$pq_codebooks_reused),
        pq_codes_reused = isTRUE(out$pq_codes_reused),
        pq_training_reused = isTRUE(out$pq_training_reused),
        search_pq_train_call_count = as.integer(
            out$search_pq_train_call_count %||% NA_integer_
        )
    ), cache_meta)
}

finish_fitted_ivfpq <- function(
    result, out, metric, params, pq, cache_meta, fastscan
) {
    strategy <- if (fastscan) {
        "faiss_IndexIVFPQFastScan_RefineFlat"
    } else "faiss_IndexIVFPQ"
    backend <- if (fastscan) "faiss_ivfpq_fastscan" else "faiss_ivfpq"
    metadata <- c(list(
        strategy = strategy, backend = backend, library = "faiss",
        metric = metric, input_type = "float32", fitted_index = TRUE
    ), fitted_ivf_base_metadata(out, params), fitted_pq_reuse_metadata(out),
    list(pq_parameters_adjusted = isTRUE(out$pq_parameters_adjusted)))
    if (fastscan) metadata <- c(metadata, fitted_fastscan_metadata(out, params))
    attr(result, "approximation") <- c(metadata, cache_meta)
    attr(result, "faiss") <- fitted_pq_faiss_metadata(
        out, metric, cache_meta, fastscan
    )
    if (fastscan) {
        append_nn_tuning_metadata(
            result, pq, params, .prefixes = list("pq_", "ivfpq_fastscan_")
        )
    } else {
        append_nn_tuning_metadata(
            result, params, pq, .prefixes = list(NULL, "pq_")
        )
    }
}

.faissR_cuvs_ivfpq_index_cache <- new.env(parent = emptyenv())
.faissR_cuvs_ivfpq_index_cache$.keys <- character()

cuvs_ivfpq_index_cache_enabled <- function() {
    isTRUE(faissr_option(
        "cache_fitted_cuda_ivfpq_indexes",
        faissr_option("cache_fitted_nn_indexes", TRUE)
    )) &&
        isTRUE(cuvs_available())
}

cuvs_ivfpq_index_cache_limit <- function() {
    value <- faissr_quiet_warning(as.integer(faissr_option(
        "cache_fitted_cuda_ivfpq_indexes_max_entries",
        1L
    )))
    if (
        length(value) != 1L || is.na(value) || !is.finite(value) || value < 0L
    ) {
        return(1L)
    }
    value
}

cuvs_ivfpq_index_cache_prune <- function() {
    limit <- cuvs_ivfpq_index_cache_limit()
    keys <- .faissR_cuvs_ivfpq_index_cache$.keys
    if (limit < 1L) {
        rm(
            list = setdiff(
                ls(.faissR_cuvs_ivfpq_index_cache, all.names = TRUE),
                ".keys"
            ),
            envir = .faissR_cuvs_ivfpq_index_cache
        )
        .faissR_cuvs_ivfpq_index_cache$.keys <- character()
        return(invisible(NULL))
    }
    while (length(keys) > limit) {
        old <- keys[[1L]]
        if (
            exists(
                old,
                envir = .faissR_cuvs_ivfpq_index_cache,
                inherits = FALSE
            )
        ) {
            rm(list = old, envir = .faissR_cuvs_ivfpq_index_cache)
        }
        keys <- keys[-1L]
    }
    .faissR_cuvs_ivfpq_index_cache$.keys <- keys
    invisible(NULL)
}

cuvs_ivfpq_index_cache_drop <- function(key) {
    if (exists(key, envir = .faissR_cuvs_ivfpq_index_cache, inherits = FALSE)) {
        rm(list = key, envir = .faissR_cuvs_ivfpq_index_cache)
    }
    .faissR_cuvs_ivfpq_index_cache$.keys <- setdiff(
        .faissR_cuvs_ivfpq_index_cache$.keys,
        key
    )
    invisible(NULL)
}

cuvs_ivfpq_index_cache_get <- function(key) {
    if (
        !exists(key, envir = .faissR_cuvs_ivfpq_index_cache, inherits = FALSE)
    ) {
        return(NULL)
    }
    entry <- get(key, envir = .faissR_cuvs_ivfpq_index_cache, inherits = FALSE)
    entry$cache_hit <- TRUE
    keys <- .faissR_cuvs_ivfpq_index_cache$.keys
    .faissR_cuvs_ivfpq_index_cache$.keys <- c(setdiff(keys, key), key)
    entry
}

cuvs_ivfpq_index_cache_set <- function(key, entry) {
    if (cuvs_ivfpq_index_cache_limit() < 1L) {
        return(invisible(NULL))
    }
    assign(key, entry, envir = .faissR_cuvs_ivfpq_index_cache)
    keys <- .faissR_cuvs_ivfpq_index_cache$.keys
    .faissR_cuvs_ivfpq_index_cache$.keys <- c(setdiff(keys, key), key)
    cuvs_ivfpq_index_cache_prune()
    invisible(NULL)
}

cuvs_ivfpq_index_cache_key <- function(data, params) {
    dims <- fitted_nn_index_dims(data)
    paste(
        "cuvs-fitted-ivfpq-fastscan",
        paste(as.integer(dims), collapse = "x"),
        matrix_fingerprint_cpp(data),
        fitted_nn_index_param_text(
            nlist = as.integer(params$ivf$nlist),
            pq_dim = as.integer(params$pq$pq_dim),
            pq_bits = as.integer(params$pq$pq_bits)
        ),
        sep = "|"
    )
}

cuvs_ivfpq_index_get_or_build <- function(data, params) {
    key <- cuvs_ivfpq_index_cache_key(data, params)
    entry <- cuvs_ivfpq_index_cache_get(key)
    if (!is.null(entry)) {
        return(entry)
    }
    entry <- list(
        index = nn_cuvs_ivf_pq_index_build_float32_cpp(
            data,
            as.integer(params$ivf$nlist),
            as.integer(params$ivf$nprobe),
            as.integer(params$pq$pq_dim),
            as.integer(params$pq$pq_bits)
        ),
        params = params,
        cache_key = key,
        cache_hit = FALSE
    )
    cuvs_ivfpq_index_cache_set(key, entry)
    entry
}

cuvs_ivfpq_fitted_search <- function(
    data,
    points,
    k,
    self_query,
    exclude_self,
    output,
    params
) {
    if (!cuvs_ivfpq_index_cache_enabled()) {
        return(NULL)
    }
    entry <- cuvs_ivfpq_index_get_or_build(data, params)
    cache_query <- isTRUE(faissr_option(
        "cache_cuda_ivfpq_query_buffers",
        TRUE
    ))
    key <- cuvs_ivfpq_query_cache_key(points, self_query, cache_query)
    out <- search_cuvs_ivfpq_fitted_index(
        entry,
        points,
        k,
        exclude_self,
        params,
        self_query,
        cache_query,
        key,
        output
    )
    if (is.null(out)) {
        return(NULL)
    }
    list(
        out = out,
        cache_meta = cuvs_ivfpq_fitted_cache_metadata(out, entry, points)
    )
}

cuvs_ivfpq_query_cache_key <- function(points, self_query, enabled) {
    if (isTRUE(self_query) || !enabled) {
        return("")
    }
    dims <- fitted_nn_index_dims(points)
    paste(
        "cuvs-query",
        paste(as.integer(dims), collapse = "x"),
        matrix_fingerprint_cpp(points),
        sep = "|"
    )
}

search_cuvs_ivfpq_fitted_index <- function(
    entry,
    points,
    k,
    exclude_self,
    params,
    self_query,
    cache_query,
    key,
    output
) {
    tryCatch(
        nn_cuvs_ivf_pq_index_search_float32_cpp(
            entry$index,
            points,
            as.integer(k),
            isTRUE(exclude_self),
            as.integer(params$ivf$nprobe),
            isTRUE(self_query),
            cache_query,
            key,
            output
        ),
        error = function(error) {
            message <- conditionMessage(error)
            if (
                grepl(
                    "pointer|externalptr|not valid|null",
                    message,
                    ignore.case = TRUE
                )
            ) {
                cuvs_ivfpq_index_cache_drop(entry$cache_key)
                return(NULL)
            }
            stop(error)
        }
    )
}

cuvs_ivfpq_fitted_cache_metadata <- function(out, entry, points) {
    c(
        cuvs_ivfpq_index_reuse_metadata(out, entry),
        cuvs_ivfpq_query_reuse_metadata(out, points)
    )
}

cuvs_ivfpq_index_reuse_metadata <- function(out, entry) {
    list(
        persistent_index_cache = TRUE,
        index_cache_hit = isTRUE(entry$cache_hit),
        index_cache_key = entry$cache_key,
        index_reused = TRUE,
        gpu_resources_reused = isTRUE(out$gpu_resources_reused),
        gpu_index_persistent = isTRUE(out$gpu_index_persistent),
        index_training_reused = isTRUE(out$index_training_reused),
        centroids_reused = isTRUE(out$centroids_reused),
        inverted_lists_reused = isTRUE(out$inverted_lists_reused),
        vectors_reused = isTRUE(out$vectors_reused),
        pq_codebooks_reused = isTRUE(out$pq_codebooks_reused),
        pq_codes_reused = isTRUE(out$pq_codes_reused),
        pq_training_reused = isTRUE(out$pq_training_reused),
        build_train_call_count = as.integer(
            out$build_train_call_count %||% NA_integer_
        ),
        search_train_call_count = as.integer(
            out$search_train_call_count %||% NA_integer_
        ),
        build_pq_train_call_count = as.integer(
            out$build_pq_train_call_count %||% NA_integer_
        ),
        search_pq_train_call_count = as.integer(
            out$search_pq_train_call_count %||% NA_integer_
        )
    )
}

cuvs_ivfpq_query_reuse_metadata <- function(out, points) {
    list(
        batch_query = isTRUE(out$batch_query),
        query_n = as.integer(
            out$query_n %||% fitted_nn_index_dims(points)[[1L]]
        ),
        query_call_count = as.integer(out$query_call_count %||% 1L),
        dataset_residency = out$dataset_residency %||% NA_character_,
        query_residency = out$query_residency %||% NA_character_,
        query_uses_index_dataset_buffer = isTRUE(
            out$query_uses_index_dataset_buffer
        ),
        query_device_buffer_cached = isTRUE(out$query_device_buffer_cached),
        query_device_buffer_reused = isTRUE(out$query_device_buffer_reused),
        query_device_cache_status = out$query_device_cache_status %||%
            NA_character_,
        query_host_to_device_copies = as.integer(
            out$query_host_to_device_copies %||% NA_integer_
        ),
        index_build_host_to_device_copies = as.integer(
            out$index_build_host_to_device_copies %||% NA_integer_
        ),
        query_upload_count = as.integer(
            out$query_upload_count %||% NA_integer_
        ),
        query_cache_hit_count = as.integer(
            out$query_cache_hit_count %||% NA_integer_
        ),
        host_device_traffic_policy = out$host_device_traffic_policy %||%
            NA_character_
    )
}

nn_compute <- function(
    data, points, k, backend, points_missing, exclude_self = FALSE,
    n_threads = NULL, metric = "euclidean", tuning = "auto",
    target_recall = 0.99, output = "double", auto_selection = NULL,
    requested_method = NULL
) {
    requested_backend <- backend
    requested_method <- public_nn_method_label(
        requested_method %||%
            nn_resolved_backend_public_method(backend) %||%
            "auto"
    )
    if (backend %in% c("hnsw", "cpu_hnsw") || grepl("^rcpp", backend)) {
        stop(
            "Legacy direct HNSW backend labels were removed. ",
            "Use `backend = \"cpu\", method = \"hnsw\"` for FAISS HNSW.",
            call. = FALSE
        )
    }
    tuning <- normalize_nn_tuning(tuning)
    target_recall <- normalize_hnsw_target_recall(target_recall)
    data_float32 <- is_float32_matrix_input(data)
    points_float32 <- if (isTRUE(points_missing)) {
        data_float32
    } else {
        is_float32_matrix_input(points)
    }
    args <- list(
        data = data, points = points, k = k, backend = backend,
        points_missing = points_missing, exclude_self = exclude_self,
        n_threads = n_threads, metric = metric, tuning = tuning,
        target_recall = target_recall, output = output,
        auto_selection = auto_selection, requested_method = requested_method,
        requested_backend = requested_backend
    )
    if (isTRUE(data_float32) || isTRUE(points_float32)) {
        args$data_float32 <- data_float32
        args$points_float32 <- points_float32
        return(do.call(nn_compute_float32, args))
    }
    do.call(nn_compute_double, args)
}

nn_compute_float32 <- function(
    data, points, k, backend, points_missing, exclude_self, n_threads,
    metric, tuning, target_recall, output, auto_selection,
    requested_method, requested_backend, data_float32, points_float32
) {
    context <- prepare_float32_nn_context(
        data, points, k, points_missing, exclude_self, n_threads, metric,
        data_float32, points_float32
    )
    args <- c(context, list(
        backend = backend, exclude_self = exclude_self, tuning = tuning,
        target_recall = target_recall, output = output,
        auto_selection = auto_selection, requested_method = requested_method,
        requested_backend = requested_backend
    ))
    do.call(nn_compute_float32_dispatch, args)
}

prepare_float32_nn_context <- function(
    data, points, k, points_missing, exclude_self, n_threads, metric,
    data_float32, points_float32
) {
    if (!isTRUE(data_float32)) {
        data <- as.matrix(data)
        storage.mode(data) <- "double"
    }
    if (!isTRUE(points_missing) && !isTRUE(points_float32)) {
        points <- as.matrix(points)
        storage.mode(points) <- "double"
    }
    data_dim <- if (isTRUE(data_float32)) {
        float32_matrix_dims(data, "data")
    } else dim(data)
    points_dim <- if (isTRUE(points_missing)) data_dim else if (
        isTRUE(points_float32)
    ) float32_matrix_dims(points, "points") else dim(points)
    if (!identical(data_dim[[2L]], points_dim[[2L]])) {
        stop("`data` and `points` must have the same number of columns.",
            call. = FALSE)
    }
    self_query <- isTRUE(points_missing) || identical(data, points)
    if (isTRUE(exclude_self) && !isTRUE(self_query)) {
        stop("Self-neighbor exclusion is valid only when `points` is `data`.",
            call. = FALSE)
    }
    if (is.null(k)) {
        k <- if (data_dim[[1L]] == 1L) 1L else min(
            data_dim[[1L]], auto_k(
                data_dim[[1L]],
                include_self = self_query && !isTRUE(exclude_self)
            )
        )
    }
    k <- normalize_nn_positive_integer(
        k, "k", "`k` must be NULL or a positive integer."
    )
    max_k <- data_dim[[1L]] - as.integer(isTRUE(exclude_self))
    if (k > max_k) stop(
        "`k` cannot be larger than the available neighbor count.",
        call. = FALSE
    )
    list(
        data = data, points = points, k = k, data_dim = data_dim,
        points_dim = points_dim, self_query = self_query,
        n_threads = normalize_nn_threads(n_threads),
        metric = normalize_nn_metric(metric)
    )
}

nn_compute_float32_dispatch <- function(
    data, points, k, data_dim, points_dim, self_query, n_threads, metric,
    backend, exclude_self, tuning, target_recall, output, auto_selection,
    requested_method, requested_backend
) {
    backend <- normalize_float32_nn_backend(
        backend, metric, data_dim, points_dim, k, self_query, exclude_self,
        tuning, target_recall, auto_selection
    )
    if (metric %in% c("cosine", "correlation")) {
        result <- float32_normalized_route(
            data, points, k, self_query, exclude_self, metric, backend,
            n_threads, output, target_recall, requested_method, data_dim
        )
        if (!is.null(result)) return(result)
    }
    result <- float32_faiss_cpu_route(
        data, points, k, self_query, exclude_self, metric, backend,
        n_threads, output, target_recall, data_dim
    )
    if (!is.null(result)) return(result)
    route_args <- list(
        data = data, points = points, k = k, data_dim = data_dim,
        points_dim = points_dim, self_query = self_query,
        n_threads = n_threads, metric = metric, backend = backend,
        exclude_self = exclude_self, tuning = tuning,
        target_recall = target_recall, output = output,
        auto_selection = auto_selection, requested_method = requested_method,
        requested_backend = requested_backend
    )
    result <- dispatch_float32_routes(route_args)
    if (!is.null(result)) return(result)
    validate_float32_direct_backend(backend)
    do.call(float32_route_faiss_flat, route_args)
}

float32_route_handlers <- function ()
{
    list(float32_route_cuvs_bruteforce, float32_route_cuvs_cagra,
        float32_route_faiss_fastscan, float32_route_faiss_ivfpq,
        float32_route_faiss_gpu_flat, float32_route_faiss_gpu_ivf,
        float32_route_faiss_gpu_ivfpq, float32_route_faiss_gpu_cagra,
        float32_route_grid, float32_route_cuvs_ivf, float32_route_cuvs_fastscan,
        float32_route_cuvs_ivfpq, float32_route_cuvs_nndescent,
        float32_route_faiss_hnsw, float32_route_cuvs_hnsw,
        float32_route_cpu_nndescent, float32_route_native_nsg,
        float32_route_native_vamana)
}

dispatch_float32_routes <- function(args) {
    for (handler in float32_route_handlers()) {
        result <- do.call(handler, args)
        if (!is.null(result)) return(result)
    }
    NULL
}

validate_float32_direct_backend <- function(backend) {
    supported <- c(
        "faiss", "cpu_faiss", "cpu_faiss_flat", "faiss_flat",
        "faiss_flat_l2", "faiss_flat_ip", "faiss_flat_cosine",
        "faiss_flat_correlation"
    )
    if (!backend %in% supported) {
        stop(
            "float32 input for backend `", backend,
            "` reached no direct float32 method handler.", call. = FALSE
        )
    }
    invisible(NULL)
}

float32_grid_execute <- function(
    data, k, bins, include_self, use_cuda, n_threads, data_dim
) {
    if (isTRUE(use_cuda)) {
        out <- cuda_grid_self_knn_float32_cpp(
            data, as.integer(k), as.integer(bins), isTRUE(include_self)
        )
        resolved <- if (data_dim[[2L]] == 3L) "cuda_grid3d" else "cuda_grid2d"
    } else if (data_dim[[2L]] == 3L) {
        out <- grid3d_self_knn_float32_cpp(
            data, as.integer(k), n_threads > 1L, as.integer(n_threads),
            as.integer(bins), isTRUE(include_self)
        )
        resolved <- "cpu_grid3d"
    } else {
        out <- grid2d_self_knn_float32_cpp(
            data, as.integer(k), n_threads > 1L, as.integer(n_threads),
            as.integer(bins), isTRUE(include_self)
        )
        resolved <- "cpu_grid2d"
    }
    list(out = out, resolved = resolved)
}

float32_route_grid <- function (data, points, k, data_dim, points_dim,
    self_query, n_threads, metric, backend, exclude_self, tuning, target_recall,
    output, auto_selection, requested_method, requested_backend)
{
    grid_backends <- c("grid", "cpu_grid", "grid2d", "cpu_grid2d", "grid3d",
        "cpu_grid3d", "cuda_grid", "cuda_grid_auto", "gpu_grid", "cuda_grid2d",
        "cuda_grid3d")
    if (!backend %in% grid_backends) return(NULL)
        if (!isTRUE(self_query)) {
            stop("Grid nearest-neighbour search is currently ",
                "available for self-KNN searches only.", call. = FALSE)
        }
        if (identical(metric, "inner_product")) {
            stop("Grid nearest-neighbour search does not ",
                "support `metric = \"inner_product\"`.", call. = FALSE)
        }
        if (!data_dim[[2]] %in% c(2, 3)) {
            stop("Grid nearest-neighbour search supports only ",
                "two- or three-column matrices.", call. = FALSE)
        }
        use_cuda <- backend %in% c("cuda_grid", "cuda_grid_auto", "gpu_grid",
            "cuda_grid2d", "cuda_grid3d")
        if (isTRUE(use_cuda) && !isTRUE(cuda_available())) {
            stop("No CUDA GPU backend is available on this machine.",
                call. = FALSE)
        }
        metric_inputs <- NULL
    float32_route_grid_continue(as.list(environment()))
}

float32_route_grid_continue <- function(context) {
    with(context, {
            search_data <- data
            if (metric %in% c("cosine", "correlation")) {
                metric_inputs <- normalized_euclidean_metric_inputs(data,
                    points,
                    self_query, metric, storage = "float")
                search_data <- metric_inputs$data
            }
            search_dim <- float32_matrix_dims(search_data, "data")
            include_self <- !isTRUE(exclude_self)
            nonself_k <- if (include_self)
                k - 1
            else k
            bins <- grid_bins_per_dim(search_dim[[1]], max(1, nonself_k),
                search_dim[[2]])
            search <- float32_grid_execute(
                search_data, k, bins, include_self, use_cuda, n_threads,
                    search_dim
            )
            out <- search$out
            resolved <- search$resolved
            result <- finish_nn_result(out, resolved, k, self_query,
                exact = TRUE,
                metric = metric)
            if (!is.null(metric_inputs)) {
                result <- finalize_normalized_euclidean_metric_result(result,
                    metric_inputs)
            }
            attr(result,
                "spatial_index") <- float32_route_grid_metadata_1(as.list(
                    environment()))
            return(finish_float32_direct_result(result, out))
    })
}

float32_route_faiss_flat <- function (data, points, k, data_dim, points_dim,
    self_query, n_threads, metric, backend, exclude_self, tuning, target_recall,
    output, auto_selection, requested_method, requested_backend)
{
    if (!backend %in% c("faiss", "cpu_faiss", "cpu_faiss_flat", "faiss_flat",
        "faiss_flat_l2", "faiss_flat_ip", "faiss_flat_cosine",
        "faiss_flat_correlation")) {
        stop("float32 input for backend `", backend, "` reached no direct ",
            "float32 method handler.", call. = FALSE)
    }
    if (!metric %in% c("euclidean", "cosine", "correlation", "inner_product")) {
        stop("float32 FAISS Flat input currently supports ",
            "`metric = \"euclidean\"`, ",
            "`\"cosine\"`, `\"correlation\"`, or `\"inner_product\"`.",
            call. = FALSE)
    }
    if (!isTRUE(faiss_available())) {
        stop("float32 FAISS Flat input requires faissR to ",
            "be built with FAISS.", call. = FALSE)
    }
    flat_backend <- switch(metric, inner_product = "faiss_flat_ip",
        "faiss_flat_l2")
    exact_params <- cpu_flatlike_params(data_dim[[1]], data_dim[[2]], k,
        metric = metric, target_recall = target_recall,
        requested_method = requested_method)
    if (metric %in% c("euclidean", "inner_product")) {
        cached <- fitted_nn_index_result(data = data, points = points, k = k,
            backend = flat_backend, result_backend = switch(metric,
            inner_product = "faiss_flat_ip", cosine = "faiss_flat_cosine",
            correlation = "faiss_flat_correlation", "faiss_flat_l2"),
            self_query = self_query, exclude_self = isTRUE(exclude_self),
            metric = metric, n_threads = n_threads, output = output,
            params = exact_params, target_recall = target_recall,
            use_cache = TRUE)
        if (!is.null(cached))
            return(cached)
    }
    out <- with_faiss_query_batch_size(exact_params, {
        nn_faiss_flat_float32_cpp(data, points, as.integer(k),
            isTRUE(exclude_self), as.integer(n_threads), metric, output)
    })
    result <- finish_nn_result(out, switch(metric,
        inner_product = "faiss_flat_ip", cosine = "faiss_flat_cosine",
        correlation = "faiss_flat_correlation", "faiss_flat_l2"), k, self_query,
        exact = TRUE, metric = metric)
    result <- attach_cpu_exact_tuning(result, exact_params, output, n_threads)
    return(finish_float32_direct_result(result, out))
    NULL
}

normalize_float32_nn_backend <- function(
    backend, metric, data_dim, points_dim, k, self_query, exclude_self,
    tuning, target_recall, auto_selection
) {
    if (backend %in% c("cuvs_ivf", "cuda_cuvs_ivf")) {
        backend <- "cuda_cuvs_ivf_flat"
    }
    if (backend %in% c("cuda_auto", "gpu_auto")) {
        route <- auto_selection %||% nn_auto_selection_for_backend(
            backend = "cuda_auto", self_query = self_query,
            n = data_dim[[1L]], p = data_dim[[2L]],
            n_points = points_dim[[1L]], k = k,
            work_size = prod(as.double(c(data_dim[[1L]], points_dim[[1L]],
                data_dim[[2L]]))), metric = metric,
            exclude_self = isTRUE(exclude_self), tuning = tuning,
            target_recall = target_recall
        )
        backend <- nn_auto_selected_backend(route, "faiss_gpu_ivf_flat")
        if (backend %in% c("cuda", "cuda_auto", "gpu_auto")) {
            backend <- float32_flat_backend(metric, gpu = TRUE)
        }
    }
    gpu_flat <- c("faiss_gpu_flat", "faiss_gpu_flat_l2", "cuda_faiss_flat_l2")
    cpu_flat <- c(
        "auto", "cpu", "cpu_auto", "faiss", "cpu_faiss",
        "cpu_faiss_flat", "faiss_flat", "faiss_flat_l2"
    )
    if (backend %in% gpu_flat && metric != "euclidean") {
        backend <- float32_flat_backend(metric, gpu = TRUE)
    }
    if (backend %in% cpu_flat) backend <- float32_flat_backend(metric, FALSE)
    backend
}

float32_flat_backend <- function(metric, gpu = FALSE) {
    prefix <- if (gpu) "faiss_gpu_flat_" else "faiss_flat_"
    suffix <- switch(
        metric, inner_product = "ip", cosine = "cosine",
        correlation = "correlation", "l2"
    )
    paste0(prefix, suffix)
}

float32_normalized_route <- function(
    data, points, k, self_query, exclude_self, metric, backend,
    n_threads, output, target_recall, requested_method, data_dim
) {
    if (backend %in% c("faiss_flat_cosine", "faiss_flat_correlation")) {
        return(float32_normalized_flat_route(
            data, points, k, self_query, exclude_self, metric, backend,
            n_threads, output, target_recall, requested_method, FALSE
        ))
    }
    if (backend %in% c(
        "faiss_gpu_flat_cosine", "cuda_faiss_flat_cosine",
        "faiss_gpu_flat_correlation", "cuda_faiss_flat_correlation"
    )) return(float32_normalized_flat_route(
        data, points, k, self_query, exclude_self, metric, backend,
        n_threads, output, target_recall, requested_method, TRUE
    ))
    if (backend %in% c("faiss_ivf", "cpu_faiss_index_ivf", "faiss_ivf_flat")) {
        return(float32_normalized_ivf_route(
            data, points, k, self_query, exclude_self, metric,
            n_threads, target_recall, data_dim, FALSE
        ))
    }
    if (backend %in% c(
        "faiss_gpu_ivf", "faiss_gpu_ivf_flat", "cuda_faiss_ivf_flat"
    )) return(float32_normalized_ivf_route(
        data, points, k, self_query, exclude_self, metric,
        n_threads, target_recall, data_dim, TRUE
    ))
    if (backend %in% c("faiss_ivfpq", "faiss_gpu_ivfpq", "cuda_faiss_ivfpq")) {
        return(float32_normalized_ivfpq_route(
            data, points, k, self_query, exclude_self, metric, backend,
            n_threads, target_recall, data_dim
        ))
    }
    if (identical(backend, "faiss_hnsw")) return(float32_normalized_hnsw_route(
        data, points, k, self_query, exclude_self, metric,
        n_threads, target_recall
    ))
    NULL
}

require_faiss_route <- function(label, gpu = FALSE) {
    available <- if (gpu) faiss_gpu_available() else faiss_available()
    if (!isTRUE(available)) stop(
        "float32 ", label, " input requires faissR to be built with ",
        if (gpu) "FAISS GPU." else "FAISS.", call. = FALSE
    )
}

float32_normalized_flat_route <- function(
    data, points, k, self_query, exclude_self, metric, backend,
    n_threads, output, target_recall, requested_method, gpu
) {
    require_faiss_route("normalized FAISS Flat", gpu)
    faiss_flat_normalized_metric_result(
        data, points, k, self_query, isTRUE(exclude_self), metric,
        backend = if (gpu) float32_flat_backend(metric, TRUE) else backend,
        accelerator = if (gpu) "cuda" else NULL, n_threads = n_threads,
        output = output, target_recall = target_recall,
        requested_method = requested_method
    )
}

float32_normalized_ivf_route <- function(
    data, points, k, self_query, exclude_self, metric,
    n_threads, target_recall, data_dim, gpu
) {
    require_faiss_route("normalized FAISS IVF", gpu)
    params <- if (gpu) cuda_ivf_params(
        data_dim[[1L]], data_dim[[2L]], k, metric, target_recall
    ) else faiss_ivf_params(
        data_dim[[1L]], k, metric, data_dim[[2L]],
        target_recall = target_recall
    )
    faiss_ivf_normalized_metric_result(
        data, points, k, self_query, isTRUE(exclude_self), metric,
        backend = if (gpu) "faiss_gpu_ivf_flat" else "faiss_ivf",
        accelerator = if (gpu) "cuda" else NULL,
        n_threads = n_threads, params = params
    )
}

float32_normalized_ivfpq_route <- function(
    data, points, k, self_query, exclude_self, metric, backend,
    n_threads, target_recall, data_dim
) {
    gpu <- backend %in% c("faiss_gpu_ivfpq", "cuda_faiss_ivfpq")
    require_faiss_route("normalized FAISS IVF-PQ", gpu)
    if (!gpu) validate_faiss_cpu_ivfpq_training_size(data_dim[[1L]])
    params <- faiss_ivf_params(
        data_dim[[1L]], k, metric, data_dim[[2L]],
        backend = if (gpu) "cuda" else "cpu", method = "ivfpq",
        target_recall = target_recall
    )
    pq <- faiss_ivfpq_pq_params(
        data_dim[[2L]], n = data_dim[[1L]], ivf_params = params
    )
    faiss_ivfpq_normalized_metric_result(
        data, points, k, self_query, isTRUE(exclude_self), metric,
        backend = if (gpu) "faiss_gpu_ivfpq" else "faiss_ivfpq",
        accelerator = if (gpu) "cuda" else NULL, n_threads = n_threads,
        params = params, pq = pq
    )
}

float32_normalized_hnsw_route <- function(
    data, points, k, self_query, exclude_self, metric,
    n_threads, target_recall
) {
    require_faiss_route("normalized FAISS HNSW")
    faiss_hnsw_normalized_metric_result(
        data, points, k, self_query, isTRUE(exclude_self), metric,
        n_threads, target_recall
    )
}

float32_faiss_cpu_route <- function(
    data, points, k, self_query, exclude_self, metric, backend,
    n_threads, output, target_recall, data_dim
) {
    if (backend %in% c("faiss_ivf", "cpu_faiss_index_ivf", "faiss_ivf_flat")) {
        return(float32_faiss_ivf_route(
            data, points, k, self_query, exclude_self, metric,
            n_threads, output, target_recall, data_dim
        ))
    }
    if (identical(backend, "faiss_nsg")) return(float32_faiss_nsg_route(
        data, points, k, self_query, exclude_self, metric, n_threads, output
    ))
    if (identical(backend, "faiss_nndescent")) {
        return(float32_faiss_nndescent_route(
            data, points, k, self_query, exclude_self, metric, n_threads, output
        ))
    }
    NULL
}

float32_faiss_ivf_route <- function(
    data, points, k, self_query, exclude_self, metric,
    n_threads, output, target_recall, data_dim
) {
    if (!metric %in% c("euclidean", "inner_product")) stop(
        "float32 FAISS IVF input supports euclidean or inner_product.",
        call. = FALSE
    )
    require_faiss_route("FAISS IVF")
    params <- faiss_ivf_params(
        data_dim[[1L]], k, metric, data_dim[[2L]],
        target_recall = target_recall
    )
    cached <- fitted_nn_index_result(
        data, points, k, "faiss_ivf", "faiss_ivf", self_query,
        isTRUE(exclude_self), metric, n_threads, output, params,
        target_recall = target_recall
    )
    if (!is.null(cached)) return(cached)
    out <- nn_faiss_ivf_float32_cpp(
        data, points, as.integer(k), as.integer(params$nlist),
        as.integer(params$nprobe), faiss_metric_search_arg(metric),
        faiss_metric_distance_output_arg(metric), isTRUE(exclude_self),
        as.integer(n_threads), output
    )
    result <- finish_nn_result(
        out, "faiss_ivf", k, self_query, exact = FALSE, metric = metric
    )
    attr(result, "approximation") <- float32_ivf_metadata(out, params, metric)
    result <- append_nn_tuning_metadata(result, params)
    finish_float32_direct_result(result, out)
}

float32_ivf_metadata <- function(out, params, metric) {
    list(
        strategy = "faiss_IndexIVFFlat", backend = "faiss_ivf",
        library = "faiss", metric = metric, input_type = "float32",
        nlist = as.integer(out$nlist), nprobe = as.integer(out$nprobe),
        requested_nlist = as.integer(params$requested_nlist),
        requested_nprobe = as.integer(params$requested_nprobe),
        ivf_parameters_adjusted = !identical(
            as.integer(params$requested_nlist), as.integer(out$nlist)
        ) || !identical(
            as.integer(params$requested_nprobe), as.integer(out$nprobe)
        )
    )
}

float32_faiss_nsg_route <- function(
    data, points, k, self_query, exclude_self, metric, n_threads, output
) {
    if (!identical(metric, "euclidean")) stop(
        "float32 FAISS NSG input supports only euclidean.", call. = FALSE
    )
    require_faiss_route("FAISS NSG")
    params <- faiss_nsg_params(k)
    out <- nn_faiss_nsg_float32_cpp(
        data, points, as.integer(k), as.integer(params$r),
        as.integer(params$search_l), as.integer(params$build_type),
        "euclidean", "euclidean", isTRUE(exclude_self),
        as.integer(n_threads), output
    )
    result <- finish_nn_result(
        out, "faiss_nsg", k, self_query, exact = FALSE, metric = metric
    )
    attr(result, "approximation") <- list(
        strategy = "faiss_IndexNSGFlat", backend = "faiss_nsg",
        library = "faiss", metric = metric, input_type = "float32",
        r = as.integer(out$r), search_l = as.integer(out$search_l),
        build_type = as.integer(out$build_type), gk = as.integer(out$gk),
        requested_r = as.integer(out$requested_r),
        requested_search_l = as.integer(out$requested_search_l),
        requested_build_type = as.integer(out$requested_build_type),
        nsg_parameters_adjusted = isTRUE(out$nsg_parameters_adjusted)
    )
    result <- append_nn_tuning_metadata(result, params)
    finish_float32_direct_result(result, out)
}

float32_faiss_nndescent_route <- function(
    data, points, k, self_query, exclude_self, metric, n_threads, output
) {
    validate_faiss_nndescent_self(metric)
    require_faiss_route("FAISS NNDescent")
    params <- faiss_nndescent_params(k)
    out <- nn_faiss_nndescent_float32_cpp(
        data, points, as.integer(k), as.integer(params$graph_k),
        as.integer(params$n_iter), as.integer(params$search_l),
        "euclidean", "euclidean", isTRUE(exclude_self),
        as.integer(n_threads), output
    )
    result <- finish_nn_result(
        out, "faiss_nndescent", k, self_query, exact = FALSE, metric = metric
    )
    attr(result, "approximation") <- list(
        strategy = "faiss_IndexNNDescentFlat", backend = "faiss_nndescent",
        library = "faiss", metric = metric, input_type = "float32",
        graph_k = as.integer(out$graph_k), n_iter = as.integer(out$n_iter),
        search_l = as.integer(out$search_l),
        requested_graph_k = as.integer(out$requested_graph_k),
        requested_n_iter = as.integer(out$requested_n_iter),
        requested_search_l = as.integer(out$requested_search_l),
        nndescent_parameters_adjusted = isTRUE(
            out$nndescent_parameters_adjusted)
    )
    result <- append_nn_tuning_metadata(result, params)
    finish_float32_direct_result(result, out)
}

float32_route_cuvs_bruteforce <- function (data, points, k, data_dim,
    points_dim, self_query, n_threads, metric, backend, exclude_self, tuning,
    target_recall, output, auto_selection, requested_method, requested_backend)
{
    if (!backend %in% c("cuvs_bruteforce", "cuda_cuvs_bruteforce",
        "cuda_cuvs_exact"))
        return(NULL)
    if (!metric %in% c("euclidean", "cosine", "correlation", "inner_product")) {
        stop("float32 cuVS brute-force input currently ",
            "supports `metric = \"euclidean\"`, ",
            "`metric = \"cosine\"`, `metric = \"correlation\"`, or ",
            "`metric = \"inner_product\"`.", call. = FALSE)
    }
    require_cuvs_backend("cuVS brute-force")
    brute_params <- cuda_bruteforce_params(data_dim[[1]], data_dim[[2]], k,
        metric = metric, target_recall = target_recall)
    metric_inputs <- NULL
    search_data <- data
    search_points <- points
    if (metric %in% c("cosine", "correlation")) {
        metric_inputs <- normalized_euclidean_metric_inputs(data, points,
            self_query, metric, storage = "float")
        search_data <- metric_inputs$data
        search_points <- metric_inputs$points
    }
    else if (identical(metric, "inner_product")) {
        metric_inputs <- mips_l2_metric_inputs(data, points, self_query)
        search_data <- metric_inputs$data
        search_points <- metric_inputs$points
    }
    float32_route_cuvs_bruteforce_continue(as.list(environment()))
}

float32_route_cuvs_bruteforce_continue <- function(context) {
    with(context, {
        brute_distance_output <- if (is.null(metric_inputs)) {
            output
        }
        else {
            "double"
        }
        out <- with_faiss_gpu_runtime(brute_params, {
            nn_cuvs_bruteforce_float32_cpp(search_data, search_points,
                as.integer(k), isTRUE(exclude_self), brute_distance_output)
        })
        resolved_backend <- "cuda_cuvs_bruteforce"
        result_backend <- if (requested_backend %in% c("cuda", "gpu")) {
            requested_backend
        } else resolved_backend
        result <- finish_nn_result(out, result_backend, k, self_query,
            exact = TRUE,
            metric = metric)
        if (!is.null(metric_inputs)) {
            result <- finalize_graph_metric_result(result, metric_inputs)
        }
        if (!identical(result_backend, resolved_backend)) {
            attr(result, "resolved_backend") <- resolved_backend
        }
        attr(result, "cuvs") <- float32_route_cuvs_bruteforce_metadata_2(
            as.list(environment())
        )
        result <- finish_float32_direct_result(result, out)
        result <- attach_cuda_exact_tuning(result, brute_params,
            brute_distance_output, n_threads)
        return(result)
    })
}

float32_route_cuvs_cagra <- function (data, points, k, data_dim, points_dim,
    self_query, n_threads, metric, backend, exclude_self, tuning, target_recall,
    output, auto_selection, requested_method, requested_backend)
{
    if (!backend %in% c("cuda_cuvs_cagra", "cuda_cagra", "gpu_cagra"))
        return(NULL)
    require_cuvs_backend("cuVS CAGRA")
    metric_inputs <- NULL
    search_data <- data
    search_points <- points
    if (metric %in% c("cosine", "correlation")) {
        metric_inputs <- normalized_euclidean_metric_inputs(data, points,
            self_query, metric, storage = "float")
        search_data <- metric_inputs$data
        search_points <- metric_inputs$points
    }
    else if (identical(metric, "inner_product")) {
        metric_inputs <- mips_l2_metric_inputs(data, points, self_query)
        search_data <- metric_inputs$data
        search_points <- metric_inputs$points
    }
    use_float32_transform <- identical(metric_inputs$transform_storage %||%
        "double",
        "float32")
    use_float32_input <- is.null(metric_inputs) || isTRUE(use_float32_transform)
    distance_output <- if (is.null(metric_inputs)) output else "double"
    params <- cuvs_cagra_params(data_dim[[1]], k, p = data_dim[[2]],
        metric = metric, target_recall = target_recall)
    build_algo <- cuvs_cagra_build_algo_for(search_data, k, self_query, params)
    float32_route_cuvs_cagra_continue(as.list(environment()))
}

float32_route_cuvs_cagra_continue <- function(context) {
    with(context, {
        out <- if (isTRUE(use_float32_input)) {
            nn_cuvs_cagra_float32_cpp(search_data, search_points, as.integer(k),
                isTRUE(exclude_self), as.integer(params$graph_degree),
                as.integer(params$intermediate_graph_degree),
                as.integer(params$search_width), as.integer(params$itopk_size),
                build_algo, distance_output)
        }
        else {
            nn_cuvs_cagra_cpp(search_data, search_points, as.integer(k),
                isTRUE(exclude_self), as.integer(params$graph_degree),
                as.integer(params$intermediate_graph_degree),
                as.integer(params$search_width), as.integer(params$itopk_size),
                build_algo)
        }
        resolved_backend <- "cuda_cuvs_cagra"
        result_backend <- if (requested_backend %in% c("cuda", "gpu")) {
            requested_backend
        } else resolved_backend
        result <- finish_nn_result(out, result_backend, k, self_query,
            exact = FALSE, metric = metric)
        if (!identical(result_backend, resolved_backend)) {
            attr(result, "resolved_backend") <- resolved_backend
        }
        if (!is.null(metric_inputs)) {
            result <- finalize_graph_metric_result(result, metric_inputs)
        }
        if (isTRUE(use_float32_input)) {
            result <- finish_float32_direct_result(result, out)
        }
        attr(result,
            "approximation") <- float32_route_cuvs_cagra_metadata_3(as.list(
                environment()))
        result <- append_nn_tuning_metadata(result, params)
        return(result)
    })
}

float32_route_faiss_fastscan <- function (data, points, k, data_dim, points_dim,
    self_query, n_threads, metric, backend, exclude_self, tuning, target_recall,
    output, auto_selection, requested_method, requested_backend)
{
    if (!identical(backend, "faiss_ivfpq_fastscan"))
        return(NULL)
    if (!metric %in% c("euclidean", "cosine", "correlation", "inner_product")) {
        stop("float32 FAISS IVFPQ FastScan input currently ",
            "supports `metric = \"euclidean\"`, ",
            "`\"cosine\"`, `\"correlation\"`, or ", "`\"inner_product\"`.",
            call. = FALSE)
    }
    if (!isTRUE(faiss_fastscan_available())) {
        stop("float32 FAISS IVFPQ FastScan input requires faissR to be ",
            "built with FAISS FastScan support ",
            "(`faiss/IndexIVFPQFastScan.h`).", call. = FALSE)
    }
    validate_faiss_cpu_ivfpq_training_size(data_dim[[1]])
    params <- ivfpq_fastscan_cpu_params(data_dim[[1]], data_dim[[2]], k,
        target_recall = target_recall, metric = metric)
    if (metric %in% c("cosine", "correlation")) {
        return(faiss_ivfpq_fastscan_normalized_metric_result(data = data,
            points = points, k = k, self_query = self_query,
            exclude_self = isTRUE(exclude_self), metric = metric,
            n_threads = n_threads, params = params))
    }
    cached <- fitted_nn_index_result(data = data, points = points, k = k,
        backend = "faiss_ivfpq_fastscan",
        result_backend = "faiss_ivfpq_fastscan", self_query = self_query,
        exclude_self = isTRUE(exclude_self), metric = metric,
        n_threads = n_threads, output = output,
        params = ivfpq_fastscan_fitted_params(params), pq = params$pq,
        target_recall = target_recall)
    float32_route_faiss_fastscan_continue(as.list(environment()))
}

float32_route_faiss_fastscan_continue <- function(context) {
    with(context, {
        if (!is.null(cached)) {
            return(cached)
        }
        out <- nn_faiss_ivfpq_fastscan_float32_cpp(data, points, as.integer(k),
            as.integer(params$ivf$nlist), as.integer(params$ivf$nprobe),
            as.integer(params$pq$m), faiss_metric_search_arg(metric),
            faiss_metric_distance_output_arg(metric),
            as.integer(params$refine_factor), as.integer(params$bbs),
            isTRUE(exclude_self), as.integer(n_threads), output)
        result <- finish_nn_result(out, "faiss_ivfpq_fastscan", k, self_query,
            exact = FALSE, metric = metric)
        attr(result,
            "approximation") <- list(
                strategy = "faiss_IndexIVFPQFastScan_RefineFlat",
            backend = "faiss_ivfpq_fastscan", library = "faiss",
                metric = metric,
            input_type = "float32", ivfpq_fastscan = TRUE, fastscan = TRUE,
            nlist = as.integer(out$nlist), nprobe = as.integer(out$nprobe),
            requested_nlist = as.integer(params$ivf$requested_nlist),
            requested_nprobe = as.integer(params$ivf$requested_nprobe),
            pq_m = as.integer(out$pq_m), pq_nbits = as.integer(out$pq_nbits),
            requested_pq_m = as.integer(out$requested_pq_m),
            requested_pq_nbits = as.integer(out$requested_pq_nbits),
                refine = isTRUE(out$refine),
                refine_factor = as.integer(out$refine_factor),
                requested_refine_factor = as.integer(
                    out$requested_refine_factor
                ),
                bbs = as.integer(out$bbs),
                requested_bbs = as.integer(out$requested_bbs),
                ivf_parameters_adjusted = !identical(as.integer(
                    params$ivf$requested_nlist),
                as.integer(out$nlist)) || !identical(as.integer(
                    params$ivf$requested_nprobe),
                as.integer(out$nprobe)),
                pq_parameters_adjusted = isTRUE(out$pq_parameters_adjusted))
        result <- append_nn_tuning_metadata(result, params$ivf, params$pq,
            params$tuning, .prefixes = list(NULL, "pq_", "ivfpq_fastscan_"))
        return(finish_float32_direct_result(result, out))
    })
}

float32_route_faiss_ivfpq <- function (data, points, k, data_dim, points_dim,
    self_query, n_threads, metric, backend, exclude_self, tuning, target_recall,
    output, auto_selection, requested_method, requested_backend)
{
    if (!identical(backend, "faiss_ivfpq"))
        return(NULL)
    if (!metric %in% c("euclidean", "inner_product")) {
        stop("float32 FAISS IVF-PQ input currently supports ",
            "`metric = \"euclidean\"` or ", "`\"inner_product\"`.",
            call. = FALSE)
    }
    if (!isTRUE(faiss_available())) {
        stop("float32 FAISS IVF-PQ input requires faissR to ",
            "be built with FAISS.", call. = FALSE)
    }
    validate_faiss_cpu_ivfpq_training_size(data_dim[[1]])
    params <- faiss_ivf_params(data_dim[[1]], k, metric = metric,
        p = data_dim[[2]], method = "ivfpq", target_recall = target_recall)
    pq <- faiss_ivfpq_pq_params(data_dim[[2]], n = data_dim[[1]],
        ivf_params = params)
    cached <- fitted_nn_index_result(data = data, points = points, k = k,
        backend = "faiss_ivfpq", result_backend = "faiss_ivfpq",
        self_query = self_query, exclude_self = isTRUE(exclude_self),
        metric = metric, n_threads = n_threads, output = output,
        params = params, pq = pq, target_recall = target_recall)
    if (!is.null(cached)) {
        return(cached)
    }
    out <- nn_faiss_ivfpq_float32_cpp(data, points, as.integer(k),
        as.integer(params$nlist), as.integer(params$nprobe), as.integer(pq$m),
        as.integer(pq$nbits), faiss_metric_search_arg(metric),
        faiss_metric_distance_output_arg(metric), isTRUE(exclude_self),
        as.integer(n_threads), output)
    result <- finish_nn_result(out, "faiss_ivfpq", k, self_query, exact = FALSE,
        metric = metric)
    attr(result, "approximation") <- list(strategy = "faiss_IndexIVFPQ",
        backend = "faiss_ivfpq", library = "faiss", metric = metric,
        input_type = "float32", nlist = as.integer(out$nlist),
        nprobe = as.integer(out$nprobe),
        requested_nlist = as.integer(params$requested_nlist),
        requested_nprobe = as.integer(params$requested_nprobe),
        pq_m = as.integer(out$pq_m), pq_nbits = as.integer(out$pq_nbits),
        requested_pq_m = as.integer(out$requested_pq_m),
        requested_pq_nbits = as.integer(out$requested_pq_nbits),
        pq_parameters_adjusted = isTRUE(out$pq_parameters_adjusted))
    result <- append_nn_tuning_metadata(result, params, pq,
        .prefixes = list(NULL, "pq_"))
    return(finish_float32_direct_result(result, out))
}

float32_route_faiss_gpu_flat <- function (data, points, k, data_dim, points_dim,
    self_query, n_threads, metric, backend, exclude_self, tuning, target_recall,
    output, auto_selection, requested_method, requested_backend)
{
    if (!backend %in% c("faiss_gpu_flat", "faiss_gpu_flat_l2",
        "cuda_faiss_flat_l2", "faiss_gpu_flat_ip", "cuda_faiss_flat_ip"))
        return(NULL)
    if (!metric %in% c("euclidean", "inner_product")) {
        stop("float32 FAISS GPU Flat input currently ",
            "supports `metric = \"euclidean\"` or ", "`\"inner_product\"`.",
            call. = FALSE)
    }
    if (!isTRUE(faiss_gpu_available())) {
        stop("float32 FAISS GPU Flat input requires faissR ",
            "to be built with FAISS GPU.", call. = FALSE)
    }
    flatlike_params <- if (metric %in% c("euclidean", "inner_product")) {
        cuda_flatlike_params(data_dim[[1]], data_dim[[2]], k, metric = metric,
            target_recall = target_recall, requested_method = requested_method)
    }
    else {
        NULL
    }
    out <- with_faiss_gpu_runtime(flatlike_params %||% list(), {
        nn_faiss_gpu_flat_float32_cpp(data, points, as.integer(k),
            isTRUE(exclude_self), faiss_metric_search_arg(metric),
            faiss_metric_distance_output_arg(metric), output)
    })
    result <- finish_nn_result(out, if (identical(metric, "inner_product")) {
        "faiss_gpu_flat_ip"
    }
    else {
        "faiss_gpu_flat_l2"
    }, k, self_query, exact = TRUE, metric = metric)
    attr(result, "faiss") <- list(index_type = as.character(out$index_type),
        library = "faiss", backend = "cuda", accelerator = "cuda",
        metric = as.character(out$metric %||% metric), input_type = "float32")
    result <- finish_float32_direct_result(result, out)
    if (!is.null(flatlike_params)) {
        result <- attach_cuda_exact_tuning(result, flatlike_params, output,
            n_threads)
    }
    return(result)
}

float32_route_faiss_gpu_ivf <- function (data, points, k, data_dim, points_dim,
    self_query, n_threads, metric, backend, exclude_self, tuning, target_recall,
    output, auto_selection, requested_method, requested_backend)
{
    if (!backend %in% c("faiss_gpu_ivf", "faiss_gpu_ivf_flat",
        "cuda_faiss_ivf_flat"))
        return(NULL)
    if (!metric %in% c("euclidean", "inner_product")) {
        stop("float32 FAISS GPU IVF-Flat input currently ",
            "supports `metric = \"euclidean\"` or ", "`\"inner_product\"`.",
            call. = FALSE)
    }
    if (!isTRUE(faiss_gpu_available())) {
        stop("float32 FAISS GPU IVF-Flat input requires ",
            "faissR to be built with FAISS GPU.", call. = FALSE)
    }
    params <- cuda_ivf_params(data_dim[[1]], data_dim[[2]], k, metric = metric,
        target_recall = target_recall)
    out <- nn_faiss_gpu_ivf_flat_float32_cpp(data, points, as.integer(k),
        as.integer(params$nlist), as.integer(params$nprobe),
        faiss_metric_search_arg(metric),
        faiss_metric_distance_output_arg(metric), isTRUE(exclude_self), output)
    result <- finish_nn_result(out, "faiss_gpu_ivf_flat", k, self_query,
        exact = FALSE, metric = metric)
    attr(result,
        "approximation") <- list(strategy = "faiss_gpu_IndexIVFFlat_cuVS",
        backend = "faiss_gpu_ivf_flat", library = "faiss", accelerator = "cuda",
        metric = metric, input_type = "float32", nlist = as.integer(out$nlist),
        nprobe = as.integer(out$nprobe),
        requested_nlist = as.integer(params$requested_nlist),
        requested_nprobe = as.integer(params$requested_nprobe),
        ivf_parameters_adjusted = !identical(as.integer(params$requested_nlist),
        as.integer(out$nlist)) || !identical(as.integer(
            params$requested_nprobe),

        as.integer(out$nprobe)))
    result <- append_nn_tuning_metadata(result, params)
    return(finish_float32_direct_result(result, out))
}

float32_route_faiss_gpu_ivfpq <- function (data, points, k, data_dim,
    points_dim, self_query, n_threads, metric, backend, exclude_self, tuning,
    target_recall, output, auto_selection, requested_method, requested_backend)
{
    if (!backend %in% c("faiss_gpu_ivfpq", "cuda_faiss_ivfpq"))
        return(NULL)
    if (!metric %in% c("euclidean", "inner_product")) {
        stop("float32 FAISS GPU IVF-PQ input currently ",
            "supports `metric = \"euclidean\"` or ", "`\"inner_product\"`.",
            call. = FALSE)
    }
    if (!isTRUE(faiss_gpu_available())) {
        stop("float32 FAISS GPU IVF-PQ input requires ",
            "faissR to be built with FAISS GPU.", call. = FALSE)
    }
    params <- faiss_ivf_params(data_dim[[1]], k, metric = metric,
        p = data_dim[[2]], backend = "cuda", method = "ivfpq",
        target_recall = target_recall)
    pq <- faiss_ivfpq_pq_params(data_dim[[2]], n = data_dim[[1]],
        ivf_params = params)
    out <- nn_faiss_gpu_ivfpq_float32_cpp(data, points, as.integer(k),
        as.integer(params$nlist), as.integer(params$nprobe), as.integer(pq$m),
        as.integer(pq$nbits), faiss_metric_search_arg(metric),
        faiss_metric_distance_output_arg(metric), isTRUE(exclude_self), output)
    result <- finish_nn_result(out, "faiss_gpu_ivfpq", k, self_query,
        exact = FALSE, metric = metric)
    attr(result,
        "approximation") <- list(strategy = "faiss_gpu_IndexIVFPQ_cuVS",
        backend = "faiss_gpu_ivfpq", library = "faiss", accelerator = "cuda",
        metric = metric, input_type = "float32", nlist = as.integer(out$nlist),
        nprobe = as.integer(out$nprobe),
        requested_nlist = as.integer(params$requested_nlist),
        requested_nprobe = as.integer(params$requested_nprobe),
        pq_m = as.integer(out$pq_m), pq_nbits = as.integer(out$pq_nbits),
        requested_pq_m = as.integer(out$requested_pq_m),
        requested_pq_nbits = as.integer(out$requested_pq_nbits),
        pq_parameters_adjusted = isTRUE(out$pq_parameters_adjusted))
    result <- append_nn_tuning_metadata(result, params, pq,
        .prefixes = list(NULL, "pq_"))
    return(finish_float32_direct_result(result, out))
}

float32_route_faiss_gpu_cagra <- function (data, points, k, data_dim,
    points_dim, self_query, n_threads, metric, backend, exclude_self, tuning,
    target_recall, output, auto_selection, requested_method, requested_backend)
{
    if (!backend %in% c("faiss_gpu_cagra", "cuda_faiss_cagra"))
        return(NULL)
    if (!isTRUE(faiss_gpu_available())) {
        stop("float32 FAISS GPU CAGRA input requires faissR ",
            "to be built with FAISS GPU.", call. = FALSE)
    }
    metric_inputs <- NULL
    search_data <- data
    search_points <- points
    if (metric %in% c("cosine", "correlation")) {
        metric_inputs <- normalized_euclidean_metric_inputs(data, points,
            self_query, metric, storage = "float")
        search_data <- metric_inputs$data
        search_points <- metric_inputs$points
    }
    else if (identical(metric, "inner_product")) {
        metric_inputs <- mips_l2_metric_inputs(data, points, self_query)
        search_data <- metric_inputs$data
        search_points <- metric_inputs$points
    }
    use_float32_transform <- identical(metric_inputs$transform_storage %||%
        "double",
        "float32")
    use_float32_input <- is.null(metric_inputs) || isTRUE(use_float32_transform)
    distance_output <- if (is.null(metric_inputs))
        output
    else "double"
    float32_route_faiss_gpu_cagra_continue(as.list(environment()))
}

float32_route_faiss_gpu_cagra_continue <- function(context) {
    with(context, {
        params <- cuvs_cagra_params(data_dim[[1]], k, p = data_dim[[2]],
            metric = metric, target_recall = target_recall)
        out <- if (isTRUE(use_float32_input)) {
            nn_faiss_gpu_cagra_float32_cpp(search_data, search_points,
                as.integer(k), as.integer(params$graph_degree),
                as.integer(params$intermediate_graph_degree),
                as.integer(params$search_width), as.integer(params$itopk_size),
                isTRUE(exclude_self), distance_output)
        }
        else {
            nn_faiss_gpu_cagra_cpp(search_data, search_points, as.integer(k),
                as.integer(params$graph_degree),
                as.integer(params$intermediate_graph_degree),
                as.integer(params$search_width), as.integer(params$itopk_size),
                isTRUE(exclude_self))
        }
        result <- finish_nn_result(out, "faiss_gpu_cagra", k, self_query,
            exact = FALSE, metric = metric)
        if (!is.null(metric_inputs)) {
            result <- finalize_graph_metric_result(result, metric_inputs)
        }
        if (isTRUE(use_float32_input)) {
            result <- finish_float32_direct_result(result, out)
        }
        attr(result, "approximation") <-
            float32_route_faiss_gpu_cagra_metadata_4(as.list(environment()))
        result <- append_nn_tuning_metadata(result, params)
        return(result)
    })
}

float32_route_cuvs_ivf <- function (data, points, k, data_dim, points_dim,
    self_query, n_threads, metric, backend, exclude_self, tuning, target_recall,
    output, auto_selection, requested_method, requested_backend)
{
    if (!backend %in% c("cuvs_ivf_flat", "cuda_cuvs_ivf_flat"))
        return(NULL)
    if (!identical(metric, "euclidean")) {
        stop("float32 cuVS IVF-Flat input currently ",
            "supports `metric = \"euclidean\"`.", call. = FALSE)
    }
    require_cuvs_backend("cuVS IVF-Flat")
    params <- cuda_ivf_params(data_dim[[1]], data_dim[[2]], k, metric = metric,
        target_recall = target_recall)
    out <- nn_cuvs_ivf_flat_float32_cpp(data, points, as.integer(k),
        as.integer(params$nlist), as.integer(params$nprobe),
        isTRUE(exclude_self), output)
    result <- finish_nn_result(out, "cuda_cuvs_ivf_flat", k, self_query,
        exact = FALSE, metric = metric)
    attr(result, "approximation") <- list(strategy = "rapids_cuvs_ivf_flat",
        backend = "cuda_cuvs_ivf_flat", library = "cuvs", accelerator = "cuda",
        metric = metric, input_type = "float32", default_candidate = FALSE,
        nlist = as.integer(out$n_lists), nprobe = as.integer(out$n_probes),
        requested_nlist = as.integer(params$requested_nlist),
        requested_nprobe = as.integer(params$requested_nprobe),
        ivf_parameters_adjusted = !identical(as.integer(params$requested_nlist),
        as.integer(out$n_lists)) || !identical(as.integer(
            params$requested_nprobe),

        as.integer(out$n_probes)),
            search_batch_size = as.integer(out$search_batch_size))
    result <- append_nn_tuning_metadata(result, params)
    return(finish_float32_direct_result(result, out))
}

float32_route_cuvs_fastscan <- function (data, points, k, data_dim, points_dim,
    self_query, n_threads, metric, backend, exclude_self, tuning, target_recall,
    output, auto_selection, requested_method, requested_backend)
{
    if (!backend %in% c("cuda_cuvs_ivfpq_fastscan", "cuvs_ivfpq_fastscan"))
        return(NULL)
    float32_route_cuvs_fastscan_preflight(as.list(environment()))
    require_cuvs_backend("CUDA cuVS 4-bit IVF-PQ")
    metric_inputs <- NULL
    search_data <- data
    search_points <- points
    if (metric %in% c("cosine", "correlation")) {
        metric_inputs <- normalized_euclidean_metric_inputs(data, points,
            self_query, metric, storage = "float")
        search_data <- metric_inputs$data
        search_points <- metric_inputs$points
    }
    else if (identical(metric, "inner_product")) {
        metric_inputs <- mips_l2_metric_inputs(data, points, self_query)
        search_data <- metric_inputs$data
        search_points <- metric_inputs$points
    }
    use_float32_transform <- identical(metric_inputs$transform_storage %||%
        "double",
        "float32")
    use_float32_output <- identical(output, "float") && is.null(metric_inputs)
    params_p <- if (is.null(metric_inputs)) data_dim[[2]] else ncol(search_data)
    params <- ivfpq_fastscan_cuda_params(data_dim[[1]], params_p, k,
        target_recall = target_recall, metric = metric)
    float32_route_cuvs_fastscan_continue(as.list(environment()))
}

float32_route_cuvs_fastscan_continue <- function(context) {
    with(context, {
        cached <- with_cuvs_ivf_batch_size(params, {
            cuvs_ivfpq_fitted_search(search_data, search_points, k, self_query,
                exclude_self, if (isTRUE(use_float32_output))
                output
            else "double", params)
        })
        cache_meta <- list()
        if (is.null(cached)) {
            out <- with_cuvs_ivf_batch_size(params, {
                nn_cuvs_ivf_pq_float32_cpp(search_data, search_points,
                    as.integer(k), as.integer(params$ivf$nlist),
                    as.integer(params$ivf$nprobe), as.integer(params$pq$pq_dim),
                    as.integer(params$pq$pq_bits), isTRUE(exclude_self),
                    if (isTRUE(use_float32_output))
                    output
                else "double")
            })
        }
        else {
            out <- cached$out
            cache_meta <- cached$cache_meta
        }
        result <- finish_nn_result(out, "cuda_cuvs_ivfpq_fastscan", k,
            self_query,
            exact = FALSE, metric = metric)
        if (!is.null(metric_inputs)) {
            result <- finalize_graph_metric_result(result, metric_inputs)
        }
        attr(result,
            "approximation") <- float32_route_cuvs_fastscan_metadata_5(as.list(
                environment()))
        result <- append_nn_tuning_metadata(result, params$ivf, params$pq,
            params$tuning, .prefixes = list(NULL, "pq_", "ivfpq_fastscan_"))
        return(finish_float32_direct_result(result, out))
    })
}

float32_route_cuvs_ivfpq <- function (data, points, k, data_dim, points_dim,
    self_query, n_threads, metric, backend, exclude_self, tuning, target_recall,
    output, auto_selection, requested_method, requested_backend)
{
    if (!backend %in% c("cuvs_ivfpq", "cuda_cuvs_ivfpq", "cuvs_ivf_pq",
        "cuda_cuvs_ivf_pq"))
        return(NULL)
    if (!identical(metric, "euclidean")) {
        stop("float32 cuVS IVF-PQ input currently supports ",
            "`metric = \"euclidean\"`.", call. = FALSE)
    }
    require_cuvs_backend("cuVS IVF-PQ")
    params <- faiss_ivf_params(data_dim[[1]], k, metric = metric,
        p = data_dim[[2]], backend = "cuda", method = "ivfpq",
        target_recall = target_recall)
    pq <- cuvs_ivfpq_params(data_dim[[2]], n = data_dim[[1]])
    out <- nn_cuvs_ivf_pq_float32_cpp(data, points, as.integer(k),
        as.integer(params$nlist), as.integer(params$nprobe),
        as.integer(pq$pq_dim), as.integer(pq$pq_bits), isTRUE(exclude_self),
        output)
    result <- finish_nn_result(out, "cuda_cuvs_ivfpq", k, self_query,
        exact = FALSE, metric = metric)
    attr(result, "approximation") <- list(strategy = "rapids_cuvs_ivf_pq",
        backend = "cuda_cuvs_ivfpq", library = "cuvs", accelerator = "cuda",
        metric = metric, input_type = "float32",
        role = "explicit_memory_pressure_backend", default_candidate = FALSE,
        nlist = as.integer(out$n_lists), nprobe = as.integer(out$n_probes),
        requested_nlist = as.integer(params$requested_nlist),
        requested_nprobe = as.integer(params$requested_nprobe),
        pq_dim = as.integer(out$pq_dim), pq_bits = as.integer(out$pq_bits),
        requested_pq_dim = as.integer(pq$requested_pq_dim),
            requested_pq_bits = as.integer(pq$requested_pq_bits),
            pq_parameters_adjusted = isTRUE(out$pq_parameters_adjusted) ||
                !identical(as.integer(pq$requested_pq_dim),
            as.integer(out$pq_dim)) || !identical(as.integer(
                pq$requested_pq_bits),
            as.integer(out$pq_bits)),
            pq_alignment_adjusted = isTRUE(out$pq_alignment_adjusted) || isTRUE(
                pq$pq_alignment_adjusted),
            pq_alignment_rule = out$pq_alignment_rule %||%
                pq$pq_alignment_rule %||% NA_character_,

        search_batch_size = as.integer(out$search_batch_size))
    result <- append_nn_tuning_metadata(result, params, pq,
        .prefixes = list(NULL, "pq_"))
    return(finish_float32_direct_result(result, out))
}

float32_route_cuvs_nndescent <- function (data, points, k, data_dim, points_dim,
    self_query, n_threads, metric, backend, exclude_self, tuning, target_recall,
    output, auto_selection, requested_method, requested_backend)
{
    if (!backend %in% c("cuvs_nndescent", "cuda_cuvs_nndescent",
        "cuda_nndescent"))
        return(NULL)
    if (identical(metric, "inner_product")) {
        stop("cuVS NN-descent does not support raw ",
            "inner-product self-KNN: its ",
            "graph-construction API accepts one symmetric ",
            "L2 dataset, while exact ",
            "maximum-inner-product reduction requires ",
            "distinct reference and query transforms.", call. = FALSE)
    }
    require_cuvs_backend("cuVS NN-descent")
    if (!isTRUE(self_query)) {
        stop("`backend = \"cuda_cuvs_nndescent\"` is only ",
            "available for self-KNN searches.", call. = FALSE)
    }
    reject_cuda_r_side_output_cleanup("cuda_cuvs_nndescent", exclude_self)
    metric_inputs <- NULL
    search_data <- data
    if (metric %in% c("cosine", "correlation")) {
        metric_inputs <- normalized_euclidean_metric_inputs(data, points,
            self_query, metric, storage = "float")
        search_data <- metric_inputs$data
    }
    search_dim <- fitted_nn_index_dims(search_data)
    nonself_k <- if (isTRUE(exclude_self))
        k
    else max(0, k - 1)
    float32_route_cuvs_nndescent_continue(as.list(environment()))
}

float32_route_cuvs_nndescent_continue <- function(context) {
    with(context, {
        distance_storage <- if (is.null(metric_inputs))
            output
        else "double"
        if (nonself_k < 1) {
            out <- list(indices = matrix(seq_len(search_dim[[1]]),
                search_dim[[1]],
                1), distances = matrix(0, search_dim[[1]], 1),
                input_type = "float32", input_layout = "trivial_self",
                input_owns_data = FALSE,
                    float32_compatibility_conversion = FALSE)
            params <- NULL
        }
        else {
            params <- cuvs_nndescent_params(search_dim[[1]], search_dim[[2]],
                nonself_k, metric = metric, target_recall = target_recall)
            out <- nn_cuvs_nndescent_self_float32_cpp(search_data,
                as.integer(nonself_k), as.integer(params$graph_degree),
                as.integer(params$intermediate_graph_degree),
                as.integer(params$max_iterations), distance_storage)
        }
        result <- finish_nn_result(out, "cuda_cuvs_nndescent", k, self_query,
            exact = FALSE, metric = metric)
        if (!is.null(metric_inputs)) {
            result <- finalize_graph_metric_result(result, metric_inputs)
        }
        attr(result,
            "approximation") <- float32_route_cuvs_nndescent_metadata_6(as.list(
                environment()))
        if (!is.null(params)) {
            result <- append_nn_tuning_metadata(result, params)
        }
        return(finish_float32_direct_result(result, out))
    })
}

float32_route_faiss_hnsw <- function (data, points, k, data_dim, points_dim,
    self_query, n_threads, metric, backend, exclude_self, tuning, target_recall,
    output, auto_selection, requested_method, requested_backend)
{
    if (!identical(backend, "faiss_hnsw"))
        return(NULL)
    if (!metric %in% c("euclidean", "inner_product")) {
        stop("float32 FAISS HNSW input currently supports ",
            "`metric = \"euclidean\"` ", "or `\"inner_product\"`.",
            call. = FALSE)
    }
    if (!isTRUE(faiss_available())) {
        stop("float32 FAISS HNSW input requires faissR to ",
            "be built with FAISS.", call. = FALSE)
    }
    params <- faiss_hnsw_params(k, n = data_dim[[1]], p = data_dim[[2]],
        metric = metric, target_recall = target_recall)
    cached <- fitted_nn_index_result(data = data, points = points, k = k,
        backend = "faiss_hnsw", result_backend = "faiss_hnsw",
        self_query = self_query, exclude_self = isTRUE(exclude_self),
        metric = metric, n_threads = n_threads, output = output,
        params = params, target_recall = target_recall)
    if (!is.null(cached)) {
        return(cached)
    }
    out <- nn_faiss_hnsw_float32_cpp(data, points, as.integer(k),
        as.integer(params$m), as.integer(params$ef_construction),
        as.integer(params$ef_search), faiss_metric_search_arg(metric),
        faiss_metric_distance_output_arg(metric), isTRUE(exclude_self),
        as.integer(n_threads), output)
    result <- finish_nn_result(out, "faiss_hnsw", k, self_query, exact = FALSE,
        metric = metric)
    float32_route_faiss_hnsw_continue(as.list(environment()))
}

float32_route_faiss_hnsw_continue <- function(context) {
    with(context, {
        attr(result, "approximation") <- list(strategy = "faiss_IndexHNSWFlat",
            backend = "faiss_hnsw", library = "faiss", metric = metric,
            input_type = "float32", m = as.integer(out$m),
            ef_construction = as.integer(out$ef_construction),
            ef_search = as.integer(out$ef_search),
            requested_m = as.integer(out$requested_m),
            requested_ef_construction = as.integer(
                out$requested_ef_construction
            ),
            requested_ef_search = as.integer(out$requested_ef_search),
            hnsw_parameters_adjusted = isTRUE(out$hnsw_parameters_adjusted),
            tuning_policy = params$policy, tuning_rule = params$rule,
                target_recall = as.numeric(
                    params$target_recall %||% target_recall
                ),
                tuning_low_dim = isTRUE(params$low_dim),
                tuning_high_dim = isTRUE(params$high_dim),
                tuning_large_n = isTRUE(params$large_n),
                tuning_small_k = isTRUE(params$small_k),
                tuning_large_k = isTRUE(params$large_k),
                tuning_non_euclidean = isTRUE(params$non_euclidean),
                tuning_shape_group = params$tuning_shape_group %||%
                    params$shape_group %||% NA_character_,
                tuning_k_bucket = as.integer(params$tuning_k_bucket %||%
                params$k_bucket %||% NA_integer_),
                    tuning_target_recall_code = as.integer(
                        params$tuning_target_recall_code %||%
                        params$target_recall_code %||% NA_integer_),
                    tuning_benchmark_basis = params$tuning_benchmark_basis %||%
                        params$benchmark_basis %||% NA_character_,
                    tuning_benchmark_target_met = isTRUE(
                        params$tuning_benchmark_target_met),
                    tuning_benchmark_source =
                        params$tuning_benchmark_source %||%
                            params$benchmark_source %||% NA_character_,
                    tuning_source = params$tuning_source %||% "cpp")
        return(finish_float32_direct_result(result, out))
    })
}

float32_route_cuvs_hnsw <- function (data, points, k, data_dim, points_dim,
    self_query, n_threads, metric, backend, exclude_self, tuning, target_recall,
    output, auto_selection, requested_method, requested_backend)
{
    if (!backend %in% c("cuda_cuvs_hnsw", "cuvs_hnsw"))
        return(NULL)
    return(cuvs_hnsw_result(data = data, points = points, k = k,
        self_query = self_query, exclude_self = isTRUE(exclude_self),
        metric = metric, n_threads = n_threads, target_recall = target_recall,
        output = output, result_backend = "cuda_cuvs_hnsw"))
}

finish_float32_nndescent_metadata <- function(
    result, out, metric, metric_inputs
) {
    approximation <- attr(out, "approximation") %||% list()
    approximation$metric <- metric
    approximation$transform <- if (is.null(metric_inputs)) {
        NA_character_
    } else {
        metric_inputs$transform
    }
    approximation$input_type <- "float32"
    attr(result, "approximation") <- approximation
    result
}

float32_route_cpu_nndescent <- function (data, points, k, data_dim, points_dim,
    self_query, n_threads, metric, backend, exclude_self, tuning, target_recall,
    output, auto_selection, requested_method, requested_backend)
{
    if (!identical(backend, "cpu_nndescent"))
        return(NULL)
    float32_route_cpu_nndescent_preflight(as.list(environment()))
    metric_inputs <- NULL
    search_data <- data
    if (metric %in% c("cosine", "correlation")) {
        metric_inputs <- normalized_euclidean_metric_inputs(data, points,
            self_query, metric, storage = "float")
        search_data <- metric_inputs$data
    }
    search_dim <- fitted_nn_index_dims(search_data)
    nonself_k <- if (isTRUE(exclude_self))
        k
    else k - 1
    float32_route_cpu_nndescent_continue(as.list(environment()))
}

float32_route_cpu_nndescent_continue <- function(context) {
    with(context, {
        if (nonself_k < 1) {
            out <- list(indices = matrix(seq_len(search_dim[[1]]),
                search_dim[[1]],
                1), distances = matrix(0, search_dim[[1]], 1),
                input_type = "float32", input_layout = "trivial_self",
                input_owns_data = FALSE,
                    float32_compatibility_conversion = FALSE)
            attr(out, "approximation") <-
                float32_route_cpu_nndescent_metadata_7(as.list(environment()))
        }
        else {
            out <- nndescent_self_knn(search_data, k = nonself_k,
                seed = fast_knn_approx_seed(), n_threads = n_threads,
                metric = if (is.null(metric_inputs)) {
                metric
            }
            else {
                "euclidean"
            }, tuning_metric = metric, target_recall = target_recall)
            if (!isTRUE(exclude_self)) {
                out <- prepend_self_neighbor_column(out)
            }
        }
        result <- finish_nn_result(out, "cpu_nndescent", k, self_query,
            exact = FALSE, metric = metric)
        if (!is.null(metric_inputs)) {
            result <- finalize_normalized_euclidean_metric_result(result,
                metric_inputs)
        }
        result <- finish_float32_nndescent_metadata(
            result, out, metric, metric_inputs
        )
        return(finish_float32_direct_result(result, out))
    })
}

float32_route_native_nsg <- function (data, points, k, data_dim, points_dim,
    self_query, n_threads, metric, backend, exclude_self, tuning, target_recall,
    output, auto_selection, requested_method, requested_backend)
{
    if (!backend %in% c("cpu_nsg", "cuda_nsg"))
        return(NULL)
    float32_route_native_nsg_preflight(as.list(environment()))
    use_cuda <- identical(backend, "cuda_nsg")
    if (isTRUE(use_cuda)) {
        reject_cuda_r_side_output_cleanup(backend, exclude_self)
    }
    metric_inputs <- NULL
    search_data <- data
    refine_metric <- "euclidean"
    if (metric %in% c("cosine", "correlation")) {
        metric_inputs <- normalized_euclidean_metric_inputs(data, points,
            self_query, metric, storage = "float")
        search_data <- metric_inputs$data
    }
    else if (identical(metric, "inner_product")) {
        refine_metric <- "inner_product"
    }
    search_dim <- if (is_float32_matrix_input(search_data)) {
        float32_matrix_dims(search_data, "data")
    }
    else {
        dim(search_data)
    }
    params <- native_nsg_params(search_dim[[1]], search_dim[[2]],
        if (isTRUE(exclude_self))
        k
    else max(1, k - 1), metric = metric, backend = if (isTRUE(use_cuda))
        "cuda"
    else "cpu", target_recall = target_recall)
    float32_route_native_nsg_continue(as.list(environment()))
}

float32_route_native_nsg_continue <- function(context) {
    with(context, {
        nonself_k <- if (isTRUE(exclude_self)) k else max(0L, k - 1L)
        if (nonself_k < 1) {
            out <- list(indices = matrix(seq_len(search_dim[[1]]),
                search_dim[[1]],
                1), distances = matrix(0, search_dim[[1]], 1),
                input_type = "float32", input_layout = "trivial_self",
                input_owns_data = FALSE,
                    float32_compatibility_conversion = FALSE)
            attr(out,
                "approximation") <- float32_route_native_nsg_metadata_9(as.list(
                    environment()))
        }
        else {
            out <- native_nsg_self_knn(search_data, k = nonself_k, r = params$r,
                graph_k = params$graph_k, metric = refine_metric,
                use_cuda = use_cuda, n_threads = n_threads,
                seed_backend = params$seed_backend %||% "exact")
            if (!isTRUE(use_cuda) && !isTRUE(exclude_self)) {
                out <- prepend_self_neighbor_column(out)
            }
        }
        result <- finish_nn_result(out, backend, k, self_query, exact = FALSE,
            metric = metric)
        if (!is.null(metric_inputs)) {
            result <- finalize_normalized_euclidean_metric_result(result,
                metric_inputs)
        }
        context <- c(as.list(environment()), list(
            approx = attr(out, "approximation", exact = TRUE)))
        attr(result,
            "approximation") <- float32_route_native_nsg_metadata_10(context)
        return(finish_float32_direct_result(result, out))
    })
}

float32_route_native_vamana <- function (data, points, k, data_dim, points_dim,
    self_query, n_threads, metric, backend, exclude_self, tuning, target_recall,
    output, auto_selection, requested_method, requested_backend)
{
    if (!backend %in% c("cpu_vamana", "cuda_vamana"))
        return(NULL)
    float32_route_native_vamana_preflight(as.list(environment()))
    use_cuda <- identical(backend, "cuda_vamana")
    if (isTRUE(use_cuda)) {
        reject_cuda_r_side_output_cleanup(backend, exclude_self)
    }
    metric_inputs <- NULL
    search_data <- data
    refine_metric <- metric
    if (metric %in% c("cosine", "correlation")) {
        metric_inputs <- normalized_euclidean_metric_inputs(data, points,
            self_query, metric, storage = "float")
        search_data <- metric_inputs$data
        refine_metric <- "euclidean"
    }
    search_dim <- if (is_float32_matrix_input(search_data)) {
        float32_matrix_dims(search_data, "data")
    }
    else {
        dim(search_data)
    }
    nonself_k <- if (isTRUE(exclude_self))
        k
    else max(0, k - 1)
    float32_route_native_vamana_continue(as.list(environment()))
}

float32_route_native_vamana_continue <- function(context) {
    with(context, {
        params <- vamana_params(search_dim[[1]], search_dim[[2]],
            if (nonself_k < 1)
            1
        else nonself_k, metric = metric, backend = if (isTRUE(use_cuda))
            "cuda"
        else "cpu", target_recall = target_recall)
        if (nonself_k < 1) {
            out <- list(indices = matrix(seq_len(search_dim[[1]]),
                search_dim[[1]],
                1), distances = matrix(0, search_dim[[1]], 1),
                input_type = "float32", input_layout = "trivial_self",
                input_owns_data = FALSE,
                    float32_compatibility_conversion = FALSE)
            attr(out, "approximation") <-
                float32_route_native_vamana_metadata_11(as.list(environment()))
        }
        else {
            out <- vamana_self_knn(search_data, k = nonself_k, r = params$r,
                search_l = params$search_l, alpha = params$alpha,
                metric = refine_metric, use_cuda = use_cuda,
                    n_threads = n_threads,
                seed_backend = params$seed_backend %||% "exact")
            if (!isTRUE(use_cuda) && !isTRUE(exclude_self)) {
                out <- prepend_self_neighbor_column(out)
            }
        }
        result <- finish_nn_result(out, backend, k, self_query, exact = FALSE,
            metric = metric)
        if (!is.null(metric_inputs)) {
            result <- finalize_normalized_euclidean_metric_result(result,
                metric_inputs)
        }
        approx <- attr(out, "approximation", exact = TRUE)
        attr(result,
            "approximation") <- float32_route_native_vamana_metadata_12(as.list(
                environment()))
        return(finish_float32_direct_result(result, out))
    })
}

double_route_handlers <- function ()
{
    list(double_route_faiss_flat, double_route_faiss_flat_ip,
        double_route_faiss_normalized, double_route_faiss_gpu_flat,
        double_route_faiss_gpu_flat_ip, double_route_faiss_gpu_normalized,
        double_route_faiss_ivf, double_route_faiss_ivfpq,
        double_route_faiss_fastscan, double_route_faiss_gpu_ivf,
        double_route_faiss_gpu_ivfpq, double_route_faiss_gpu_cagra,
        double_route_faiss_hnsw, double_route_faiss_nsg,
        double_route_native_nsg, double_route_native_vamana,
        double_route_faiss_nndescent, double_route_cuvs_default,
        double_route_cuvs_cagra, double_route_cuvs_hnsw, double_route_cuvs_ivf,
            double_route_cuvs_fastscan, double_route_cuvs_ivfpq,
            double_route_cuvs_bruteforce, double_route_cuvs_nndescent,
            double_route_cpu_nndescent, double_route_cuda_grid,
            double_route_cpu_grid)
}

dispatch_double_routes <- function(args) {
    for (handler in double_route_handlers()) {
        result <- do.call(handler, args)
        if (!is.null(result)) return(result)
    }
    NULL
}

double_route_faiss_flat <- function (data, points, k, backend, exclude_self,
    n_threads, metric, tuning, target_recall, output, auto_selection,
    requested_method, requested_backend, self_query)
{
    if (!backend %in% c("faiss", "cpu_faiss", "cpu_faiss_flat", "faiss_flat",
        "faiss_flat_l2"))
        return(NULL)
    if (!isTRUE(faiss_available())) {
        stop("The real FAISS C++ backend is not available in this build. ",
            "Reinstall faissR with `FAISS_HOME` pointing ",
            "to a FAISS installation.", call. = FALSE)
    }
    exact_params <- cpu_flatlike_params(nrow(data), ncol(data), k,
        metric = "euclidean", target_recall = target_recall,
        requested_method = requested_method)
    if (!identical(backend, "faiss")) {
        cached <- fitted_nn_index_result(data = data, points = points, k = k,
            backend = "faiss_flat_l2", result_backend = "faiss_flat_l2",
            self_query = self_query, exclude_self = isTRUE(exclude_self),
            metric = "euclidean", n_threads = n_threads, output = output,
            params = exact_params, target_recall = target_recall,
            use_cache = TRUE)
        if (!is.null(cached))
            return(cached)
    }
    out <- with_faiss_query_batch_size(exact_params, {
        nn_faiss_flat_cpp(data, points, as.integer(k), isTRUE(exclude_self),
            as.integer(n_threads))
    })
    result_backend <- if (identical(backend, "faiss")) {
        "faiss"
    }
    else {
        "faiss_flat_l2"
    }
    result <- finish_nn_result(out, result_backend, k, self_query, exact = TRUE,
        metric = "euclidean")
    attr(result, "faiss") <- list(index_type = as.character(out$index_type),
        library = "faiss", backend = "cpu", metric = "euclidean")
    result <- attach_cpu_exact_tuning(result, exact_params, output, n_threads)
    return(result)
}

double_route_faiss_flat_ip <- function (data, points, k, backend, exclude_self,
    n_threads, metric, tuning, target_recall, output, auto_selection,
    requested_method, requested_backend, self_query)
{
    if (!identical(backend, "faiss_flat_ip"))
        return(NULL)
    if (!isTRUE(faiss_available())) {
        stop("The real FAISS C++ backend is not available in this build. ",
            "Reinstall faissR with `FAISS_HOME` pointing ",
            "to a FAISS installation.", call. = FALSE)
    }
    exact_params <- cpu_flatlike_params(nrow(data), ncol(data), k,
        metric = "inner_product", target_recall = target_recall,
        requested_method = requested_method)
    cached <- fitted_nn_index_result(data = data, points = points, k = k,
        backend = "faiss_flat_ip", result_backend = "faiss_flat_ip",
        self_query = self_query, exclude_self = isTRUE(exclude_self),
        metric = "inner_product", n_threads = n_threads, output = output,
        params = exact_params, target_recall = target_recall, use_cache = TRUE)
    if (!is.null(cached)) {
        return(cached)
    }
    out <- with_faiss_query_batch_size(exact_params, {
        nn_faiss_flat_ip_cpp(data, points, as.integer(k), isTRUE(exclude_self),
            as.integer(n_threads))
    })
    result <- finish_nn_result(out, "faiss_flat_ip", k, self_query,
        exact = TRUE, metric = "inner_product")
    attr(result, "faiss") <- list(index_type = as.character(out$index_type),
        library = "faiss", backend = "cpu", metric = as.character(out$metric))
    result <- attach_cpu_exact_tuning(result, exact_params, output, n_threads)
    return(result)
}

double_route_faiss_normalized <- function (data, points, k, backend,
    exclude_self, n_threads, metric, tuning, target_recall, output,
    auto_selection, requested_method, requested_backend, self_query)
{
    if (!backend %in% c("faiss_flat_cosine", "faiss_flat_correlation"))
        return(NULL)
    if (!isTRUE(faiss_available())) {
        stop("The real FAISS C++ backend is not available in this build. ",
            "Reinstall faissR with `FAISS_HOME` pointing ",
            "to a FAISS installation.", call. = FALSE)
    }
    metric_label <- if (identical(backend, "faiss_flat_correlation")) {
        "correlation"
    }
    else {
        "cosine"
    }
    return(faiss_flat_normalized_metric_result(data = data, points = points,
        k = k, self_query = self_query, exclude_self = isTRUE(exclude_self),
        metric = metric_label, backend = backend, accelerator = NULL,
        n_threads = n_threads, output = output, target_recall = target_recall,
        requested_method = requested_method))
}

double_route_faiss_gpu_flat <- function (data, points, k, backend, exclude_self,
    n_threads, metric, tuning, target_recall, output, auto_selection,
    requested_method, requested_backend, self_query)
{
    if (!backend %in% c("faiss_gpu_flat", "faiss_gpu_flat_l2",
        "cuda_faiss_flat_l2"))
        return(NULL)
    if (!isTRUE(faiss_gpu_available())) {
        stop("The real FAISS C++ GPU Flat L2 backend is not ",
            "available in this build. ",
            "Reinstall faissR with FAISS GPU/cuVS headers ",
            "available through `FAISS_HOME`.", call. = FALSE)
    }
    flatlike_params <- cuda_flatlike_params(nrow(data), ncol(data), k,
        metric = "euclidean", target_recall = target_recall,
        requested_method = requested_method)
    out <- with_faiss_gpu_runtime(flatlike_params, {
        if (identical(output, "float")) {
            nn_faiss_gpu_flat_float32_cpp(data, points, as.integer(k),
                isTRUE(exclude_self), "euclidean", "euclidean", output)
        }
        else {
            nn_faiss_gpu_flat_cpp(data, points, as.integer(k),
                isTRUE(exclude_self))
        }
    })
    result <- finish_nn_result(out, "faiss_gpu_flat_l2", k, self_query,
        exact = TRUE)
    attr(result, "faiss") <- list(index_type = as.character(out$index_type),
        library = "faiss", backend = "cuda", accelerator = "cuda",
        metric = as.character(out$metric),
        input_type = out$input_type %||% NULL)
    if (identical(output, "float")) {
        result <- finish_float32_direct_result(result, out)
    }
    result <- attach_cuda_exact_tuning(result, flatlike_params, output,
        n_threads)
    return(result)
}

double_route_faiss_gpu_flat_ip <- function (data, points, k, backend,
    exclude_self, n_threads, metric, tuning, target_recall, output,
    auto_selection, requested_method, requested_backend, self_query)
{
    if (!backend %in% c("faiss_gpu_flat_ip", "cuda_faiss_flat_ip"))
        return(NULL)
    if (!isTRUE(faiss_gpu_available())) {
        stop("The real FAISS C++ GPU Flat IP backend is not ",
            "available in this build. ",
            "Reinstall faissR with FAISS GPU/cuVS headers ",
            "available through `FAISS_HOME`.", call. = FALSE)
    }
    flatlike_params <- cuda_flatlike_params(nrow(data), ncol(data), k,
        metric = "inner_product", target_recall = target_recall,
        requested_method = requested_method)
    out <- with_faiss_gpu_runtime(flatlike_params, {
        if (identical(output, "float")) {
            nn_faiss_gpu_flat_float32_cpp(data, points, as.integer(k),
                isTRUE(exclude_self), "inner_product", "inner_product", output)
        }
        else {
            nn_faiss_gpu_flat_ip_cpp(data, points, as.integer(k),
                isTRUE(exclude_self))
        }
    })
    result <- finish_nn_result(out, "faiss_gpu_flat_ip", k, self_query,
        exact = TRUE, metric = "inner_product")
    attr(result, "faiss") <- list(index_type = as.character(out$index_type),
        library = "faiss", backend = "cuda", accelerator = "cuda",
        metric = as.character(out$metric),
        input_type = out$input_type %||% NULL)
    if (identical(output, "float")) {
        result <- finish_float32_direct_result(result, out)
    }
    result <- attach_cuda_exact_tuning(result, flatlike_params, output,
        n_threads)
    return(result)
}

double_route_faiss_gpu_normalized <- function (data, points, k, backend,
    exclude_self, n_threads, metric, tuning, target_recall, output,
    auto_selection, requested_method, requested_backend, self_query)
{
    if (!backend %in% c("faiss_gpu_flat_cosine", "cuda_faiss_flat_cosine",
        "faiss_gpu_flat_correlation", "cuda_faiss_flat_correlation"))
        return(NULL)
    if (!isTRUE(faiss_gpu_available())) {
        stop("The real FAISS C++ GPU Flat IP backend is not ",
            "available in this build. ",
            "Reinstall faissR with FAISS GPU/cuVS headers ",
            "available through `FAISS_HOME`.", call. = FALSE)
    }
    metric_label <- if (backend %in% c("faiss_gpu_flat_correlation",
        "cuda_faiss_flat_correlation")) {
        "correlation"
    }
    else {
        "cosine"
    }
    return(faiss_flat_normalized_metric_result(data = data, points = points,
        k = k, self_query = self_query, exclude_self = isTRUE(exclude_self),
        metric = metric_label, backend = if (identical(metric_label,
        "correlation")) {
        "faiss_gpu_flat_correlation"
    } else {
        "faiss_gpu_flat_cosine"
    }, accelerator = "cuda", n_threads = n_threads, output = output,
        target_recall = target_recall, requested_method = requested_method))
}

double_route_faiss_ivf <- function (data, points, k, backend, exclude_self,
    n_threads, metric, tuning, target_recall, output, auto_selection,
    requested_method, requested_backend, self_query)
{
    if (!backend %in% c("faiss_ivf", "cpu_faiss_index_ivf", "faiss_ivf_flat"))
        return(NULL)
    if (!isTRUE(faiss_available())) {
        stop("The real FAISS C++ IVF backend is not ",
            "available in this build. ",
            "Reinstall faissR with `FAISS_HOME` pointing ",
            "to a FAISS installation.", call. = FALSE)
    }
    params <- faiss_ivf_params(nrow(data), k, metric = metric, p = ncol(data),
        target_recall = target_recall)
    if (metric %in% c("cosine", "correlation")) {
        return(faiss_ivf_normalized_metric_result(data = data, points = points,
            k = k, self_query = self_query, exclude_self = isTRUE(exclude_self),
            metric = metric, backend = "faiss_ivf", accelerator = NULL,
            n_threads = n_threads, params = params))
    }
    cached <- fitted_nn_index_result(data = data, points = points, k = k,
        backend = "faiss_ivf", result_backend = "faiss_ivf",
        self_query = self_query, exclude_self = isTRUE(exclude_self),
        metric = metric, n_threads = n_threads, output = output,
        params = params, target_recall = target_recall)
    if (!is.null(cached)) {
        return(cached)
    }
    double_route_faiss_ivf_continue(as.list(environment()))
}

double_route_faiss_ivf_continue <- function(context) {
    with(context, {
        out <- if (identical(output, "float")) {
            nn_faiss_ivf_float32_cpp(data, points, as.integer(k),
                as.integer(params$nlist), as.integer(params$nprobe),
                faiss_metric_search_arg(metric),
                faiss_metric_distance_output_arg(metric), isTRUE(exclude_self),
                as.integer(n_threads), output)
        }
        else {
            nn_faiss_ivf_cpp(data, points, as.integer(k),
                as.integer(params$nlist),
                as.integer(params$nprobe), faiss_metric_search_arg(metric),
                faiss_metric_distance_output_arg(metric), isTRUE(exclude_self),
                as.integer(n_threads))
        }
        result <- finish_nn_result(out, "faiss_ivf", k, self_query,
            exact = FALSE,
            metric = metric)
        attr(result, "approximation") <- list(strategy = "faiss_IndexIVFFlat",
            backend = "faiss_ivf", library = "faiss", metric = metric,
            input_type = out$input_type %||% NULL,
                nlist = as.integer(out$nlist),
            nprobe = as.integer(out$nprobe),
            requested_nlist = as.integer(params$requested_nlist),
            requested_nprobe = as.integer(params$requested_nprobe),
            ivf_parameters_adjusted = !identical(
                as.integer(params$requested_nlist), as.integer(out$nlist)
            ) || !identical(as.integer(
                params$requested_nprobe),
            as.integer(out$nprobe)))
        result <- append_nn_tuning_metadata(result, params)
        if (identical(output, "float")) {
            result <- finish_float32_direct_result(result, out)
        }
        return(result)
    })
}

double_route_faiss_ivfpq <- function (data, points, k, backend, exclude_self,
    n_threads, metric, tuning, target_recall, output, auto_selection,
    requested_method, requested_backend, self_query)
{
    if (!identical(backend, "faiss_ivfpq"))
        return(NULL)
    if (!isTRUE(faiss_available())) {
        stop("The real FAISS C++ IVFPQ backend is not ",
            "available in this build. ",
            "Reinstall faissR with `FAISS_HOME` pointing ",
            "to a FAISS installation.", call. = FALSE)
    }
    validate_faiss_cpu_ivfpq_training_size(nrow(data))
    params <- faiss_ivf_params(nrow(data), k, metric = metric, p = ncol(data),
        method = "ivfpq", target_recall = target_recall)
    pq <- faiss_ivfpq_pq_params(ncol(data), n = nrow(data), ivf_params = params)
    if (metric %in% c("cosine", "correlation")) {
        return(faiss_ivfpq_normalized_metric_result(data = data,
            points = points, k = k, self_query = self_query,
            exclude_self = isTRUE(exclude_self), metric = metric,
            backend = "faiss_ivfpq", accelerator = NULL, n_threads = n_threads,
            params = params, pq = pq))
    }
    cached <- fitted_nn_index_result(data = data, points = points, k = k,
        backend = "faiss_ivfpq", result_backend = "faiss_ivfpq",
        self_query = self_query, exclude_self = isTRUE(exclude_self),
        metric = metric, n_threads = n_threads, output = output,
        params = params, pq = pq, target_recall = target_recall)
    if (!is.null(cached)) {
        return(cached)
    }
    double_route_faiss_ivfpq_continue(as.list(environment()))
}

double_route_faiss_ivfpq_continue <- function(context) {
    with(context, {
        out <- if (identical(output, "float")) {
            nn_faiss_ivfpq_float32_cpp(data, points, as.integer(k),
                as.integer(params$nlist), as.integer(params$nprobe),
                as.integer(pq$m), as.integer(pq$nbits),
                faiss_metric_search_arg(metric),
                faiss_metric_distance_output_arg(metric), isTRUE(exclude_self),
                as.integer(n_threads), output)
        }
        else {
            nn_faiss_ivfpq_cpp(data, points, as.integer(k),
                as.integer(params$nlist), as.integer(params$nprobe),
                as.integer(pq$m), as.integer(pq$nbits),
                faiss_metric_search_arg(metric),
                faiss_metric_distance_output_arg(metric), isTRUE(exclude_self),
                as.integer(n_threads))
        }
        result <- finish_nn_result(out, "faiss_ivfpq", k, self_query,
            exact = FALSE,
            metric = metric)
        attr(result, "approximation") <- list(strategy = "faiss_IndexIVFPQ",
            backend = "faiss_ivfpq", library = "faiss", metric = metric,
            input_type = out$input_type %||% NULL,
                nlist = as.integer(out$nlist),
            nprobe = as.integer(out$nprobe),
            requested_nlist = as.integer(params$requested_nlist),
            requested_nprobe = as.integer(params$requested_nprobe),
            ivf_parameters_adjusted = !identical(
                as.integer(params$requested_nlist), as.integer(out$nlist)
            ) || !identical(as.integer(
                params$requested_nprobe),
            as.integer(out$nprobe)),
            pq_m = as.integer(out$pq_m), pq_nbits = as.integer(out$pq_nbits),
                requested_pq_m = as.integer(out$requested_pq_m),
                requested_pq_nbits = as.integer(out$requested_pq_nbits),
                pq_parameters_adjusted = isTRUE(out$pq_parameters_adjusted))
        result <- append_nn_tuning_metadata(result, params, pq,
            .prefixes = list(NULL, "pq_"))
        if (identical(output, "float")) {
            result <- finish_float32_direct_result(result, out)
        }
        return(result)
    })
}

double_route_faiss_fastscan <- function (data, points, k, backend, exclude_self,
    n_threads, metric, tuning, target_recall, output, auto_selection,
    requested_method, requested_backend, self_query)
{
    if (!identical(backend, "faiss_ivfpq_fastscan"))
        return(NULL)
    if (!metric %in% c("euclidean", "cosine", "correlation", "inner_product")) {
        stop("FAISS IVFPQ FastScan input currently supports ",
            "`metric = \"euclidean\"`, `\"cosine\"`, ",
            "`\"correlation\"`, or `\"inner_product\"`.", call. = FALSE)
    }
    if (!isTRUE(faiss_fastscan_available())) {
        stop("The FAISS IVFPQ FastScan backend is not ",
            "available in this build. ",
            "Reinstall faissR with `FAISS_HOME` pointing ",
            "to a FAISS installation ",
            "that provides `faiss/IndexIVFPQFastScan.h`.", call. = FALSE)
    }
    validate_faiss_cpu_ivfpq_training_size(nrow(data))
    params <- ivfpq_fastscan_cpu_params(nrow(data), ncol(data), k,
        target_recall = target_recall, metric = metric)
    if (metric %in% c("cosine", "correlation")) {
        return(faiss_ivfpq_fastscan_normalized_metric_result(data = data,
            points = points, k = k, self_query = self_query,
            exclude_self = isTRUE(exclude_self), metric = metric,
            n_threads = n_threads, params = params))
    }
    cached <- fitted_nn_index_result(data = data, points = points, k = k,
        backend = "faiss_ivfpq_fastscan",
        result_backend = "faiss_ivfpq_fastscan", self_query = self_query,
        exclude_self = isTRUE(exclude_self), metric = metric,
        n_threads = n_threads, output = output,
        params = ivfpq_fastscan_fitted_params(params), pq = params$pq,
        target_recall = target_recall)
    double_route_faiss_fastscan_continue(as.list(environment()))
}

double_route_faiss_fastscan_continue <- function(context) {
    with(context, {
        if (!is.null(cached)) {
            return(cached)
        }
        out <- nn_faiss_ivfpq_fastscan_float32_cpp(data, points, as.integer(k),
            as.integer(params$ivf$nlist), as.integer(params$ivf$nprobe),
            as.integer(params$pq$m), faiss_metric_search_arg(metric),
            faiss_metric_distance_output_arg(metric),
            as.integer(params$refine_factor), as.integer(params$bbs),
            isTRUE(exclude_self), as.integer(n_threads), output)
        result <- finish_nn_result(out, "faiss_ivfpq_fastscan", k, self_query,
            exact = FALSE, metric = metric)
        attr(result,
            "approximation") <- list(
                strategy = "faiss_IndexIVFPQFastScan_RefineFlat",
            backend = "faiss_ivfpq_fastscan", library = "faiss",
                metric = metric,
            input_type = out$input_type %||% NULL, ivfpq_fastscan = TRUE,
            fastscan = TRUE, nlist = as.integer(out$nlist),
            nprobe = as.integer(out$nprobe),
            requested_nlist = as.integer(params$ivf$requested_nlist),
            requested_nprobe = as.integer(params$ivf$requested_nprobe),
            pq_m = as.integer(out$pq_m), pq_nbits = as.integer(out$pq_nbits),
            requested_pq_m = as.integer(out$requested_pq_m),
            requested_pq_nbits = as.integer(out$requested_pq_nbits),
                refine = isTRUE(out$refine),
                refine_factor = as.integer(out$refine_factor),
                requested_refine_factor = as.integer(
                    out$requested_refine_factor
                ),
                bbs = as.integer(out$bbs),
                requested_bbs = as.integer(out$requested_bbs),
                ivf_parameters_adjusted = !identical(as.integer(
                    params$ivf$requested_nlist),
                as.integer(out$nlist)) || !identical(as.integer(
                    params$ivf$requested_nprobe),
                as.integer(out$nprobe)),
                pq_parameters_adjusted = isTRUE(out$pq_parameters_adjusted))
        result <- append_nn_tuning_metadata(result, params$ivf, params$pq,
            params$tuning, .prefixes = list(NULL, "pq_", "ivfpq_fastscan_"))
        return(finish_float32_direct_result(result, out))
    })
}

double_route_faiss_gpu_ivf <- function (data, points, k, backend, exclude_self,
    n_threads, metric, tuning, target_recall, output, auto_selection,
    requested_method, requested_backend, self_query)
{
    if (!backend %in% c("faiss_gpu_ivf", "faiss_gpu_ivf_flat",
        "cuda_faiss_ivf_flat"))
        return(NULL)
    if (!isTRUE(faiss_gpu_available())) {
        stop("The real FAISS C++ GPU IVF Flat backend is ",
            "not available in this build. ",
            "Reinstall faissR with FAISS GPU/cuVS headers ",
            "available through `FAISS_HOME`.", call. = FALSE)
    }
    params <- cuda_ivf_params(nrow(data), ncol(data), k, metric = metric,
        target_recall = target_recall)
    tuning_metadata <- NULL
    if (isTRUE(faiss_gpu_ivf_should_tune(data, k, self_query, tuning = tuning,
        metric = metric))) {
        tuned <- faiss_gpu_ivf_tune_params(data, k, params, tuning = tuning)
        params <- tuned$params
        tuning_metadata <- tuned$tuning
    }
    if (metric %in% c("cosine", "correlation")) {
        return(faiss_ivf_normalized_metric_result(data = data, points = points,
            k = k, self_query = self_query, exclude_self = isTRUE(exclude_self),
            metric = metric, backend = "faiss_gpu_ivf_flat",
            accelerator = "cuda", n_threads = n_threads, params = params,
            tuning_metadata = tuning_metadata))
    }
    double_route_faiss_gpu_ivf_continue(as.list(environment()))
}

double_route_faiss_gpu_ivf_continue <- function(context) {
    with(context, {
        out <- if (identical(output, "float")) {
            nn_faiss_gpu_ivf_flat_float32_cpp(data, points, as.integer(k),
                as.integer(params$nlist), as.integer(params$nprobe),
                faiss_metric_search_arg(metric),
                faiss_metric_distance_output_arg(metric), isTRUE(exclude_self),
                output)
        }
        else {
            nn_faiss_gpu_ivf_flat_cpp(data, points, as.integer(k),
                as.integer(params$nlist), as.integer(params$nprobe),
                faiss_metric_search_arg(metric),
                faiss_metric_distance_output_arg(metric), isTRUE(exclude_self))
        }
        result <- finish_nn_result(out, "faiss_gpu_ivf_flat", k, self_query,
            exact = FALSE, metric = metric)
        attr(result,
            "approximation") <- list(strategy = "faiss_gpu_IndexIVFFlat_cuVS",
            backend = "faiss_gpu_ivf_flat", library = "faiss",
                accelerator = "cuda",
            metric = metric, input_type = out$input_type %||% NULL,
            nlist = as.integer(out$nlist), nprobe = as.integer(out$nprobe),
            requested_nlist = as.integer(params$requested_nlist),
            requested_nprobe = as.integer(params$requested_nprobe),
            ivf_parameters_adjusted = !identical(
                as.integer(params$requested_nlist), as.integer(out$nlist)
            ) || !identical(as.integer(
                params$requested_nprobe),

            as.integer(out$nprobe)), tuning = tuning_metadata)
        result <- append_nn_tuning_metadata(result, params)
        if (identical(output, "float")) {
            result <- finish_float32_direct_result(result, out)
        }
        return(result)
    })
}

double_route_faiss_gpu_ivfpq <- function (data, points, k, backend,
    exclude_self, n_threads, metric, tuning, target_recall, output,
    auto_selection, requested_method, requested_backend, self_query)
{
    if (!backend %in% c("faiss_gpu_ivfpq", "cuda_faiss_ivfpq"))
        return(NULL)
    if (!isTRUE(faiss_gpu_available())) {
        stop("The real FAISS C++ GPU IVF-PQ backend is not ",
            "available in this build. ",
            "Reinstall faissR with FAISS GPU/cuVS headers ",
            "available through `FAISS_HOME`.", call. = FALSE)
    }
    params <- faiss_ivf_params(nrow(data), k, metric = metric, p = ncol(data),
        backend = "cuda", method = "ivfpq", target_recall = target_recall)
    pq <- faiss_ivfpq_pq_params(ncol(data), n = nrow(data), ivf_params = params)
    if (metric %in% c("cosine", "correlation")) {
        return(faiss_ivfpq_normalized_metric_result(data = data,
            points = points, k = k, self_query = self_query,
            exclude_self = isTRUE(exclude_self), metric = metric,
            backend = "faiss_gpu_ivfpq", accelerator = "cuda",
            n_threads = n_threads, params = params, pq = pq))
    }
    out <- if (identical(output, "float")) {
        nn_faiss_gpu_ivfpq_float32_cpp(data, points, as.integer(k),
            as.integer(params$nlist), as.integer(params$nprobe),
            as.integer(pq$m), as.integer(pq$nbits),
            faiss_metric_search_arg(metric),
            faiss_metric_distance_output_arg(metric), isTRUE(exclude_self),
            output)
    }
    else {
        nn_faiss_gpu_ivfpq_cpp(data, points, as.integer(k),
            as.integer(params$nlist), as.integer(params$nprobe),
            as.integer(pq$m), as.integer(pq$nbits),
            faiss_metric_search_arg(metric),
            faiss_metric_distance_output_arg(metric), isTRUE(exclude_self))
    }
    double_route_faiss_gpu_ivfpq_continue(as.list(environment()))
}

double_route_faiss_gpu_ivfpq_continue <- function(context) {
    with(context, {
        result <- finish_nn_result(out, "faiss_gpu_ivfpq", k, self_query,
            exact = FALSE, metric = metric)
        attr(result,
            "approximation") <- list(strategy = "faiss_gpu_IndexIVFPQ_cuVS",
            backend = "faiss_gpu_ivfpq", library = "faiss",
                accelerator = "cuda",
            metric = metric, input_type = out$input_type %||% NULL,
            role = "explicit_memory_pressure_backend",
                default_candidate = FALSE,
            nlist = as.integer(out$nlist), nprobe = as.integer(out$nprobe),
            requested_nlist = as.integer(params$requested_nlist),
            requested_nprobe = as.integer(params$requested_nprobe),
            ivf_parameters_adjusted = !identical(
                as.integer(params$requested_nlist), as.integer(out$nlist)
            ) || !identical(as.integer(
                params$requested_nprobe),
                as.integer(out$nprobe)), pq_m = as.integer(out$pq_m),
                pq_nbits = as.integer(out$pq_nbits),
                requested_pq_m = as.integer(out$requested_pq_m),
                requested_pq_nbits = as.integer(out$requested_pq_nbits),
                pq_parameters_adjusted = isTRUE(out$pq_parameters_adjusted))
        result <- append_nn_tuning_metadata(result, params, pq,
            .prefixes = list(NULL, "pq_"))
        if (identical(output, "float")) {
            result <- finish_float32_direct_result(result, out)
        }
        return(result)
    })
}

prepare_cuda_metric_inputs <- function(data, points, self_query, metric) {
    metric_inputs <- NULL
    if (metric %in% c("cosine", "correlation")) {
        metric_inputs <- normalized_euclidean_metric_inputs(
            data, points, self_query, metric, storage = "float"
        )
    } else if (identical(metric, "inner_product")) {
        metric_inputs <- mips_l2_metric_inputs(data, points, self_query)
    }
    list(
        metric_inputs = metric_inputs,
        data = metric_inputs$data %||% data,
        points = metric_inputs$points %||% points,
        use_float32 = identical(
            metric_inputs$transform_storage %||% "double", "float32"
        )
    )
}

cagra_requested_parameters <- function(params, out) {
    fields <- c(
        "graph_degree", "intermediate_graph_degree", "search_width",
        "itopk_size"
    )
    stats::setNames(lapply(fields, function(field) {
        params[[paste0("requested_", field)]] %||%
            out[[paste0("requested_", field)]]
    }), fields)
}

validate_cuvs_cagra_tuning <- function(tuning) {
    if (!is.list(tuning) || identical(tuning$status, "target_met")) return()
    recall <- tuning$results$recall %||% numeric()
    best <- if (length(recall)) max(recall, na.rm = TRUE) else NA_real_
    threshold <- faissr_quiet_warning(as.numeric(faissr_option(
        "cuvs_cagra_tune_min_recall", tuning$target_recall
    )))
    if (length(threshold) != 1L || !is.finite(threshold)) threshold <- 0.985
    if (!is.finite(best) || best < threshold) {
        stop(
            "cuVS CAGRA pilot tuning did not meet the requested recall ",
            "target (best pilot recall = ",
            if (is.finite(best)) formatC(best, digits = 4L,
                format = "f") else "NA",
            "). Use FAISS GPU CAGRA, cuVS brute force, or disable tuning.",
            call. = FALSE
        )
    }
}

tune_cuvs_cagra_route <- function(data, k, self_query, params, tuning, algo) {
    metadata <- NULL
    if (isTRUE(cuvs_cagra_should_tune(data, k, self_query, tuning = tuning))) {
        tuned <- cuvs_cagra_tune_params(
            data, k, params, tuning = tuning, build_algo = algo
        )
        params <- tuned$params
        metadata <- tuned$tuning
        validate_cuvs_cagra_tuning(metadata)
    }
    list(params = params, metadata = metadata)
}

double_route_faiss_gpu_cagra <- function (data, points, k, backend,
    exclude_self, n_threads, metric, tuning, target_recall, output,
    auto_selection, requested_method, requested_backend, self_query)
{
    if (!backend %in% c("faiss_gpu_cagra", "cuda_faiss_cagra"))
        return(NULL)
    double_route_faiss_gpu_cagra_preflight(as.list(environment()))
    inputs <- prepare_cuda_metric_inputs(data, points, self_query, metric)
    metric_inputs <- inputs$metric_inputs
    search_data <- inputs$data
    search_points <- inputs$points
    params <- cuvs_cagra_params(nrow(data), k, p = ncol(data), metric = metric,
        target_recall = target_recall)
    use_float32_transform <- inputs$use_float32
    out <- if (isTRUE(use_float32_transform)) {
        nn_faiss_gpu_cagra_float32_cpp(search_data, search_points,
            as.integer(k), as.integer(params$graph_degree),
            as.integer(params$intermediate_graph_degree),
            as.integer(params$search_width), as.integer(params$itopk_size),
            isTRUE(exclude_self), "double")
    }
    else {
        nn_faiss_gpu_cagra_cpp(search_data, search_points, as.integer(k),
            as.integer(params$graph_degree),
            as.integer(params$intermediate_graph_degree),
            as.integer(params$search_width), as.integer(params$itopk_size),
            isTRUE(exclude_self))
    }
    requested <- cagra_requested_parameters(params, out)
    requested_graph_degree <- requested$graph_degree
    requested_intermediate_graph_degree <- requested$intermediate_graph_degree
    requested_search_width <- requested$search_width
    requested_itopk_size <- requested$itopk_size
    result <- finish_nn_result(out, "faiss_gpu_cagra", k, self_query,
        exact = FALSE, metric = metric)
    if (!is.null(metric_inputs)) {
        result <- finalize_graph_metric_result(result, metric_inputs)
    }
    if (isTRUE(use_float32_transform)) {
        result <- finish_float32_direct_result(result, out)
    }
    attr(result,
        "approximation") <- double_route_faiss_gpu_cagra_metadata_13(as.list(
            environment()))
    result <- append_nn_tuning_metadata(result, params)
    return(result)
}

double_route_faiss_hnsw <- function (data, points, k, backend, exclude_self,
    n_threads, metric, tuning, target_recall, output, auto_selection,
    requested_method, requested_backend, self_query)
{
    if (!identical(backend, "faiss_hnsw"))
        return(NULL)
    if (!isTRUE(faiss_available())) {
        stop("The real FAISS C++ HNSW backend is not ",
            "available in this build. ",
            "Reinstall faissR with `FAISS_HOME` pointing ",
            "to a FAISS installation.", call. = FALSE)
    }
    if (metric %in% c("cosine", "correlation")) {
        return(faiss_hnsw_normalized_metric_result(data = data, points = points,
            k = k, self_query = self_query, exclude_self = isTRUE(exclude_self),
            metric = metric, n_threads = n_threads,
            target_recall = target_recall))
    }
    params <- faiss_hnsw_params(k, n = nrow(data), p = ncol(data),
        metric = metric, target_recall = target_recall)
    cached <- fitted_nn_index_result(data = data, points = points, k = k,
        backend = "faiss_hnsw", result_backend = "faiss_hnsw",
        self_query = self_query, exclude_self = isTRUE(exclude_self),
        metric = metric, n_threads = n_threads, output = output,
        params = params, target_recall = target_recall)
    if (!is.null(cached)) {
        return(cached)
    }
    out <- nn_faiss_hnsw_cpp(data, points, as.integer(k), as.integer(params$m),
        as.integer(params$ef_construction), as.integer(params$ef_search),
        faiss_metric_search_arg(metric),
        faiss_metric_distance_output_arg(metric), isTRUE(exclude_self),
        as.integer(n_threads))
    double_route_faiss_hnsw_continue(as.list(environment()))
}

double_route_faiss_hnsw_continue <- function(context) {
    with(context, {
        result <- finish_nn_result(out, "faiss_hnsw", k, self_query,
            exact = FALSE,
            metric = metric)
        attr(result, "approximation") <- list(strategy = "faiss_IndexHNSWFlat",
            backend = "faiss_hnsw", library = "faiss", metric = metric,
            m = as.integer(out$m),
            ef_construction = as.integer(out$ef_construction),
            ef_search = as.integer(out$ef_search),
            requested_m = as.integer(out$requested_m),
            requested_ef_construction = as.integer(
                out$requested_ef_construction
            ),
            requested_ef_search = as.integer(out$requested_ef_search),
            hnsw_parameters_adjusted = isTRUE(out$hnsw_parameters_adjusted),
            tuning_policy = params$policy,
            tuning_rule = params$rule,
                target_recall = as.numeric(
                    params$target_recall %||% target_recall
                ),
                tuning_low_dim = isTRUE(params$low_dim),
                tuning_high_dim = isTRUE(params$high_dim),
                tuning_large_n = isTRUE(params$large_n),
                tuning_small_k = isTRUE(params$small_k),
                tuning_large_k = isTRUE(params$large_k),
                tuning_non_euclidean = isTRUE(params$non_euclidean),
                tuning_shape_group = params$tuning_shape_group %||%
                    params$shape_group %||% NA_character_,
                tuning_k_bucket = as.integer(params$tuning_k_bucket %||%
                params$k_bucket %||% NA_integer_),
                    tuning_target_recall_code = as.integer(
                        params$tuning_target_recall_code %||%
                        params$target_recall_code %||% NA_integer_),
                    tuning_benchmark_basis = params$tuning_benchmark_basis %||%
                        params$benchmark_basis %||% NA_character_,
                    tuning_benchmark_target_met = isTRUE(
                        params$tuning_benchmark_target_met),
                    tuning_benchmark_source =
                        params$tuning_benchmark_source %||%
                            params$benchmark_source %||% NA_character_,
                    tuning_source = params$tuning_source %||% "cpp")
        return(result)
    })
}

double_route_faiss_nsg <- function (data, points, k, backend, exclude_self,
    n_threads, metric, tuning, target_recall, output, auto_selection,
    requested_method, requested_backend, self_query)
{
    if (!identical(backend, "faiss_nsg"))
        return(NULL)
    if (!isTRUE(faiss_available())) {
        stop("The real FAISS C++ NSG backend is not ",
            "available in this build. ",
            "Reinstall faissR with `FAISS_HOME` pointing ",
            "to a FAISS installation.", call. = FALSE)
    }
    metric_inputs <- NULL
    search_data <- data
    search_points <- points
    if (metric %in% c("cosine", "correlation", "inner_product")) {
        stop("`backend = \"faiss_nsg\"` currently supports ",
            "only `metric = \"euclidean\"`. ",
            "FAISS NSG graph construction can abort the R ",
            "process for normalized ",
            "cosine/correlation or raw inner-product ",
            "routes in this linked FAISS build.", call. = FALSE)
    }
    params <- faiss_nsg_params(k)
    out <- nn_faiss_nsg_cpp(search_data, search_points, as.integer(k),
        as.integer(params$r), as.integer(params$search_l),
        as.integer(params$build_type), "euclidean", "euclidean",
        isTRUE(exclude_self), as.integer(n_threads))
    result <- finish_nn_result(out, "faiss_nsg", k, self_query, exact = FALSE,
        metric = metric)
    if (!is.null(metric_inputs)) {
        result <- finalize_normalized_euclidean_metric_result(result,
            metric_inputs)
    }
    attr(result, "approximation") <- list(strategy = "faiss_IndexNSGFlat",
        backend = "faiss_nsg", library = "faiss", metric = metric,
        transform = if (is.null(metric_inputs)) {
        NA_character_
    } else {
        metric_inputs$transform
    }, r = as.integer(out$r), search_l = as.integer(out$search_l),
        build_type = as.integer(out$build_type), gk = as.integer(out$gk),
        requested_r = as.integer(out$requested_r),
        requested_search_l = as.integer(out$requested_search_l),
        requested_build_type = as.integer(out$requested_build_type),
        nsg_parameters_adjusted = isTRUE(out$nsg_parameters_adjusted))
    result <- append_nn_tuning_metadata(result, params)
    return(result)
}

double_route_native_nsg <- function (data, points, k, backend, exclude_self,
    n_threads, metric, tuning, target_recall, output, auto_selection,
    requested_method, requested_backend, self_query)
{
    if (!backend %in% c("cpu_nsg", "cuda_nsg"))
        return(NULL)
    if (!isTRUE(self_query)) {
        stop("Native NSG is currently implemented for ",
            "self-KNN searches only.", call. = FALSE)
    }
    use_cuda <- identical(backend, "cuda_nsg")
    if (isTRUE(use_cuda) && !isTRUE(cuda_available())) {
        stop("No CUDA GPU backend is available on this machine.", call. = FALSE)
    }
    if (isTRUE(use_cuda)) {
        reject_cuda_r_side_output_cleanup(backend, exclude_self)
    }
    metric_inputs <- NULL
    search_data <- data
    refine_metric <- "euclidean"
    if (metric %in% c("cosine", "correlation")) {
        metric_inputs <- normalized_euclidean_metric_inputs(data, points,
            self_query, metric)
        search_data <- metric_inputs$data
    }
    else if (identical(metric, "inner_product")) {
        refine_metric <- "inner_product"
    }
    params <- native_nsg_params(nrow(search_data), ncol(search_data),
        if (isTRUE(exclude_self))
        k
    else max(1, k - 1), metric = metric, backend = if (isTRUE(use_cuda))
        "cuda"
    else "cpu", target_recall = target_recall)
    double_route_native_nsg_continue(as.list(environment()))
}

double_route_native_nsg_continue <- function(context) {
    with(context, {
        nonself_k <- if (isTRUE(exclude_self))
            k
        else max(0, k - 1)
        if (nonself_k < 1) {
            out <- list(indices = matrix(seq_len(nrow(search_data)),
                nrow(search_data), 1), distances = matrix(0, nrow(search_data),
                    1))
            attr(out,
                "approximation") <- double_route_native_nsg_metadata_14(as.list(
                    environment()))
        }
        else {
            out <- native_nsg_self_knn(search_data, k = nonself_k, r = params$r,
                graph_k = params$graph_k, metric = refine_metric,
                use_cuda = use_cuda, n_threads = n_threads,
                seed_backend = params$seed_backend %||% "exact")
            if (!isTRUE(use_cuda) && !isTRUE(exclude_self)) {
                out <- prepend_self_neighbor_column(out)
            }
        }
        result <- finish_nn_result(out, backend, k, self_query, exact = FALSE,
            metric = metric)
        if (!is.null(metric_inputs)) {
            result <- finalize_normalized_euclidean_metric_result(result,
                metric_inputs)
        }
        approx <- attr(out, "approximation", exact = TRUE)
        attr(result,
            "approximation") <- double_route_native_nsg_metadata_15(as.list(
                environment()))
        return(result)
    })
}

double_route_native_vamana <- function (data, points, k, backend, exclude_self,
    n_threads, metric, tuning, target_recall, output, auto_selection,
    requested_method, requested_backend, self_query)
{
    if (!backend %in% c("cpu_vamana", "cuda_vamana"))
        return(NULL)
    if (!isTRUE(self_query)) {
        stop("Vamana is currently implemented for self-KNN searches only.",
            call. = FALSE)
    }
    use_cuda <- identical(backend, "cuda_vamana")
    if (isTRUE(use_cuda) && !isTRUE(cuda_available())) {
        stop("No CUDA GPU backend is available on this machine.", call. = FALSE)
    }
    if (isTRUE(use_cuda)) {
        reject_cuda_r_side_output_cleanup(backend, exclude_self)
    }
    metric_inputs <- NULL
    search_data <- data
    refine_metric <- metric
    if (metric %in% c("cosine", "correlation")) {
        metric_inputs <- normalized_euclidean_metric_inputs(data, points,
            self_query, metric)
        search_data <- metric_inputs$data
        refine_metric <- "euclidean"
    }
    nonself_k <- if (isTRUE(exclude_self))
        k
    else max(0, k - 1)
    double_route_native_vamana_continue(as.list(environment()))
}

double_route_native_vamana_continue <- function(context) {
    with(context, {
        params <- vamana_params(nrow(search_data), ncol(search_data),
            if (nonself_k < 1)
            1
        else nonself_k, metric = metric, backend = if (isTRUE(use_cuda))
            "cuda"
        else "cpu", target_recall = target_recall)
        if (nonself_k < 1) {
            out <- list(indices = matrix(seq_len(nrow(search_data)),
                nrow(search_data), 1), distances = matrix(0, nrow(search_data),
                    1))
            attr(out, "approximation") <-
                double_route_native_vamana_metadata_16(as.list(environment()))
        }
        else {
            out <- vamana_self_knn(search_data, k = nonself_k, r = params$r,
                search_l = params$search_l, alpha = params$alpha,
                metric = refine_metric, use_cuda = use_cuda,
                    n_threads = n_threads,
                seed_backend = params$seed_backend %||% "exact")
            if (!isTRUE(use_cuda) && !isTRUE(exclude_self)) {
                out <- prepend_self_neighbor_column(out)
            }
        }
        result <- finish_nn_result(out, backend, k, self_query, exact = FALSE,
            metric = metric)
        if (!is.null(metric_inputs)) {
            result <- finalize_normalized_euclidean_metric_result(result,
                metric_inputs)
        }
        approx <- attr(out, "approximation", exact = TRUE)
        attr(result,
            "approximation") <- double_route_native_vamana_metadata_17(as.list(
                environment()))
        return(result)
    })
}

double_route_faiss_nndescent <- function (data, points, k, backend,
    exclude_self, n_threads, metric, tuning, target_recall, output,
    auto_selection, requested_method, requested_backend, self_query)
{
    if (!identical(backend, "faiss_nndescent"))
        return(NULL)
    if (!identical(metric, "euclidean")) {
        stop("`backend = \"faiss_nndescent\"` is currently ",
            "validated only for ",
            "`metric = \"euclidean\"` in this FAISS build.", call. = FALSE)
    }
    if (!isTRUE(faissr_option("enable_faiss_nndescent", FALSE))) {
        stop("FAISS NNDescent is disabled by default ",
            "because linked FAISS builds can ",
            "abort the R process during graph construction. Use public ",
            "`method = \"nndescent\"` for the native CPU route, or set ",
            "`options(faissR.enable_faiss_nndescent = ",
            "TRUE)` to opt into the ", "experimental FAISS backend.",
            call. = FALSE)
    }
    if (!isTRUE(faiss_available())) {
        stop("The real FAISS C++ NNDescent backend is not ",
            "available in this build. ",
            "Reinstall faissR with `FAISS_HOME` pointing ",
            "to a FAISS installation.", call. = FALSE)
    }
    params <- faiss_nndescent_params(k)
    out <- nn_faiss_nndescent_cpp(data, points, as.integer(k),
        as.integer(params$graph_k), as.integer(params$n_iter),
        as.integer(params$search_l), "euclidean", "euclidean",
        isTRUE(exclude_self), as.integer(n_threads))
    result <- finish_nn_result(out, "faiss_nndescent", k, self_query,
        exact = FALSE, metric = metric)
    attr(result, "approximation") <- list(strategy = "faiss_IndexNNDescentFlat",
        backend = "faiss_nndescent", library = "faiss", metric = metric,
        graph_k = as.integer(out$graph_k), n_iter = as.integer(out$n_iter),
        search_l = as.integer(out$search_l),
        requested_graph_k = as.integer(out$requested_graph_k),
        requested_n_iter = as.integer(out$requested_n_iter),
        requested_search_l = as.integer(out$requested_search_l),
        nndescent_parameters_adjusted = isTRUE(
            out$nndescent_parameters_adjusted))
    result <- append_nn_tuning_metadata(result, params)
    return(result)
}

double_route_cuvs_default <- function (data, points, k, backend, exclude_self,
    n_threads, metric, tuning, target_recall, output, auto_selection,
    requested_method, requested_backend, self_query)
{
    if (!backend %in% c("cuvs", "gpu_cuvs", "cuda_cuvs"))
        return(NULL)
    require_cuvs_backend("cuVS")
    work_size <- prod(as.double(c(nrow(data), nrow(points), ncol(data))))
    selected <- select_cuvs_auto_backend(
        self_query = self_query, n = nrow(data), p = ncol(data),
        n_points = nrow(points), k = k, work_size = work_size
    )
    nn_compute(
        data, points, k, selected, self_query, exclude_self, n_threads,
        metric, tuning, target_recall, output, auto_selection,
        requested_method
    )
}

double_route_cuvs_cagra <- function (data, points, k, backend, exclude_self,
    n_threads, metric, tuning, target_recall, output, auto_selection,
    requested_method, requested_backend, self_query)
{
    if (!backend %in% c("cuda_cuvs_cagra", "cuda_cagra", "gpu_cagra"))
        return(NULL)
    require_cuvs_backend("cuVS CAGRA")
    inputs <- prepare_cuda_metric_inputs(data, points, self_query, metric)
    metric_inputs <- inputs$metric_inputs
    search_data <- inputs$data
    search_points <- inputs$points
    use_float32_transform <- inputs$use_float32
    use_float32_output <- identical(output, "float") && is.null(metric_inputs)
    params <- cuvs_cagra_params(nrow(data), k, p = ncol(data), metric = metric,
        target_recall = target_recall)
    build_algo <- cuvs_cagra_build_algo_for(search_data, k, self_query, params)
    tuned <- tune_cuvs_cagra_route(
        search_data, k, self_query, params, tuning, build_algo
    )
    params <- tuned$params
    tuning_metadata <- tuned$metadata
    out <- if (isTRUE(use_float32_transform) || isTRUE(use_float32_output)) {
        nn_cuvs_cagra_float32_cpp(search_data, search_points, as.integer(k),
            isTRUE(exclude_self), as.integer(params$graph_degree),
            as.integer(params$intermediate_graph_degree),
            as.integer(params$search_width), as.integer(params$itopk_size),
            build_algo, if (isTRUE(use_float32_output))
            output
        else "double")
    }
    else {
        nn_cuvs_cagra_cpp(search_data, search_points, as.integer(k),
            isTRUE(exclude_self), as.integer(params$graph_degree),
            as.integer(params$intermediate_graph_degree),
            as.integer(params$search_width), as.integer(params$itopk_size),
            build_algo)
    }
    double_route_cuvs_cagra_continue(as.list(environment()))
}

double_route_cuvs_cagra_continue <- function(context) {
    with(context, {
        requested <- cagra_requested_parameters(params, out)
        requested_graph_degree <- requested$graph_degree
        requested_intermediate_graph_degree <-
            requested$intermediate_graph_degree
        requested_search_width <- requested$search_width
        requested_itopk_size <- requested$itopk_size
        resolved_backend <- "cuda_cuvs_cagra"
        result_backend <- if (requested_backend %in% c("cuda", "gpu")) {
            requested_backend
        } else resolved_backend
        result <- finish_nn_result(out, result_backend, k, self_query,
            exact = FALSE, metric = metric)
        if (!identical(result_backend, resolved_backend)) {
            attr(result, "resolved_backend") <- resolved_backend
        }
        if (!is.null(metric_inputs)) {
            result <- finalize_graph_metric_result(result, metric_inputs)
        }
        if (isTRUE(use_float32_transform) || isTRUE(use_float32_output)) {
            result <- finish_float32_direct_result(result, out)
        }
        attr(result,
            "approximation") <- double_route_cuvs_cagra_metadata_18(as.list(
                environment()))
        result <- append_nn_tuning_metadata(result, params)
        return(result)
    })
}

double_route_cuvs_hnsw <- function (data, points, k, backend, exclude_self,
    n_threads, metric, tuning, target_recall, output, auto_selection,
    requested_method, requested_backend, self_query)
{
    if (!backend %in% c("cuda_cuvs_hnsw", "cuvs_hnsw"))
        return(NULL)
    return(cuvs_hnsw_result(data = data, points = points, k = k,
        self_query = self_query, exclude_self = isTRUE(exclude_self),
        metric = metric, n_threads = n_threads, target_recall = target_recall,
        output = output, result_backend = "cuda_cuvs_hnsw"))
}

double_route_cuvs_ivf <- function (data, points, k, backend, exclude_self,
    n_threads, metric, tuning, target_recall, output, auto_selection,
    requested_method, requested_backend, self_query)
{
    if (!backend %in% c("cuvs_ivf_flat", "cuda_cuvs_ivf_flat"))
        return(NULL)
    require_cuvs_backend("cuVS IVF-Flat")
    metric_inputs <- NULL
    search_data <- data
    search_points <- points
    if (metric %in% c("cosine", "correlation")) {
        metric_inputs <- normalized_euclidean_metric_inputs(data, points,
            self_query, metric, storage = "float")
        search_data <- metric_inputs$data
        search_points <- metric_inputs$points
    }
    else if (identical(metric, "inner_product")) {
        metric_inputs <- mips_l2_metric_inputs(data, points, self_query)
        search_data <- metric_inputs$data
        search_points <- metric_inputs$points
    }
    use_float32_transform <- identical(metric_inputs$transform_storage %||%
        "double",
        "float32")
    use_float32_output <- identical(output, "float") && is.null(metric_inputs)
    double_route_cuvs_ivf_continue(as.list(environment()))
}

double_route_cuvs_ivf_continue <- function(context) {
    with(context, {
        params <- cuda_ivf_params(nrow(search_data), ncol(search_data), k,
            metric = metric, target_recall = target_recall)
        use_float_route <- isTRUE(use_float32_transform) ||
            isTRUE(use_float32_output)
        out <- if (use_float_route) {
            nn_cuvs_ivf_flat_float32_cpp(search_data, search_points,
                as.integer(k),
                as.integer(params$nlist), as.integer(params$nprobe),
                isTRUE(exclude_self), if (isTRUE(use_float32_output))
                output
            else "double")
        }
        else {
            nn_cuvs_ivf_flat_cpp(search_data, search_points, as.integer(k),
                as.integer(params$nlist), as.integer(params$nprobe),
                isTRUE(exclude_self))
        }
        result <- finish_nn_result(out, "cuda_cuvs_ivf_flat", k, self_query,
            exact = FALSE, metric = metric)
        if (!is.null(metric_inputs)) {
            result <- finalize_graph_metric_result(result, metric_inputs)
        }
        if (isTRUE(use_float32_transform) || isTRUE(use_float32_output)) {
            result <- finish_float32_direct_result(result, out)
        }
        attr(result,
            "approximation") <- double_route_cuvs_ivf_metadata_19(as.list(
                environment()))
        result <- append_nn_tuning_metadata(result, params)
        return(result)
    })
}

double_route_cuvs_fastscan <- function (data, points, k, backend, exclude_self,
    n_threads, metric, tuning, target_recall, output, auto_selection,
    requested_method, requested_backend, self_query)
{
    if (!backend %in% c("cuda_cuvs_ivfpq_fastscan", "cuvs_ivfpq_fastscan"))
        return(NULL)
    double_route_cuvs_fastscan_preflight(as.list(environment()))
    require_cuvs_backend("CUDA cuVS 4-bit IVF-PQ")
    inputs <- prepare_cuda_metric_inputs(data, points, self_query, metric)
    metric_inputs <- inputs$metric_inputs
    search_data <- inputs$data
    search_points <- inputs$points
    use_float32_transform <- inputs$use_float32
    use_float32_output <- identical(output, "float") && is.null(metric_inputs)
    params_p <- if (is.null(metric_inputs)) ncol(data) else ncol(search_data)
    params <- ivfpq_fastscan_cuda_params(nrow(data), params_p, k,
        target_recall = target_recall, metric = metric)
    cached <- with_cuvs_ivf_batch_size(params, {
        cuvs_ivfpq_fitted_search(search_data, search_points, k, self_query,
            exclude_self, if (isTRUE(use_float32_output))
            output
        else "double", params)
    })
    cache_meta <- list()
    double_route_cuvs_fastscan_continue(as.list(environment()))
}

double_route_cuvs_fastscan_continue <- function(context) {
    with(context, {
        if (is.null(cached)) {
            out <- with_cuvs_ivf_batch_size(params, {
                use_float_route <- isTRUE(use_float32_transform) ||
                    isTRUE(use_float32_output)
                if (use_float_route) {
                    nn_cuvs_ivf_pq_float32_cpp(search_data, search_points,
                        as.integer(k), as.integer(params$ivf$nlist),
                        as.integer(params$ivf$nprobe),
                            as.integer(params$pq$pq_dim),
                        as.integer(params$pq$pq_bits), isTRUE(exclude_self),
                        if (isTRUE(use_float32_output))
                    output
                    else "double")
                }
                else {
                    nn_cuvs_ivf_pq_cpp(search_data, search_points,
                        as.integer(k),
                        as.integer(params$ivf$nlist),
                            as.integer(params$ivf$nprobe),
                        as.integer(params$pq$pq_dim),
                            as.integer(params$pq$pq_bits),
                        isTRUE(exclude_self))
                }
            })
        }
        else {
            out <- cached$out
            cache_meta <- cached$cache_meta
        }
        result <- finish_nn_result(out, "cuda_cuvs_ivfpq_fastscan", k,
            self_query,
            exact = FALSE, metric = metric)
        if (!is.null(metric_inputs)) {
            result <- finalize_graph_metric_result(result, metric_inputs)
        }
        attr(result,
            "approximation") <- double_route_cuvs_fastscan_metadata_20(as.list(
                environment()))
        result <- append_nn_tuning_metadata(result, params$ivf, params$pq,
            params$tuning, .prefixes = list(NULL, "pq_", "ivfpq_fastscan_"))
        if (isTRUE(use_float32_transform) || isTRUE(use_float32_output)) {
            result <- finish_float32_direct_result(result, out)
        }
        return(result)
    })
}

double_route_cuvs_ivfpq <- function (data, points, k, backend, exclude_self,
    n_threads, metric, tuning, target_recall, output, auto_selection,
    requested_method, requested_backend, self_query)
{
    if (!backend %in% c("cuvs_ivfpq", "cuda_cuvs_ivfpq", "cuvs_ivf_pq",
        "cuda_cuvs_ivf_pq"))
        return(NULL)
    require_cuvs_backend("cuVS IVF-PQ")
    metric_inputs <- NULL
    search_data <- data
    search_points <- points
    if (metric %in% c("cosine", "correlation")) {
        metric_inputs <- normalized_euclidean_metric_inputs(data, points,
            self_query, metric, storage = "float")
        search_data <- metric_inputs$data
        search_points <- metric_inputs$points
    }
    else if (identical(metric, "inner_product")) {
        metric_inputs <- mips_l2_metric_inputs(data, points, self_query)
        search_data <- metric_inputs$data
        search_points <- metric_inputs$points
    }
    use_float32_transform <- identical(metric_inputs$transform_storage %||%
        "double",
        "float32")
    use_float32_output <- identical(output, "float") && is.null(metric_inputs)
    params <- faiss_ivf_params(nrow(data), k, metric = metric,
        p = ncol(search_data), backend = "cuda", method = "ivfpq",
        target_recall = target_recall)
    double_route_cuvs_ivfpq_continue(as.list(environment()))
}

double_route_cuvs_ivfpq_continue <- function(context) {
    with(context, {
        pq <- cuvs_ivfpq_params(ncol(search_data), n = nrow(search_data))
        use_float_route <- isTRUE(use_float32_transform) ||
            isTRUE(use_float32_output)
        out <- if (use_float_route) {
            nn_cuvs_ivf_pq_float32_cpp(search_data, search_points,
                as.integer(k),
                as.integer(params$nlist), as.integer(params$nprobe),
                as.integer(pq$pq_dim), as.integer(pq$pq_bits),
                    isTRUE(exclude_self),
                if (isTRUE(use_float32_output))
                output
            else "double")
        }
        else {
            nn_cuvs_ivf_pq_cpp(search_data, search_points, as.integer(k),
                as.integer(params$nlist), as.integer(params$nprobe),
                as.integer(pq$pq_dim), as.integer(pq$pq_bits),
                    isTRUE(exclude_self))
        }
        result <- finish_nn_result(out, "cuda_cuvs_ivfpq", k, self_query,
            exact = FALSE, metric = metric)
        if (!is.null(metric_inputs)) {
            result <- finalize_graph_metric_result(result, metric_inputs)
        }
        if (isTRUE(use_float32_transform) || isTRUE(use_float32_output)) {
            result <- finish_float32_direct_result(result, out)
        }
        attr(result,
            "approximation") <- double_route_cuvs_ivfpq_metadata_21(as.list(
                environment()))
        result <- append_nn_tuning_metadata(result, params, pq,
            .prefixes = list(NULL, "pq_"))
        return(result)
    })
}

double_route_cuvs_bruteforce <- function (data, points, k, backend,
    exclude_self, n_threads, metric, tuning, target_recall, output,
    auto_selection, requested_method, requested_backend, self_query)
{
    if (!backend %in% c("cuvs_bruteforce", "cuda_cuvs_bruteforce",
        "cuda_cuvs_exact"))
        return(NULL)
    require_cuvs_backend("cuVS brute-force")
    inputs <- prepare_cuda_metric_inputs(data, points, self_query, metric)
    metric_inputs <- inputs$metric_inputs
    search_data <- inputs$data
    search_points <- inputs$points
    brute_params <- if (metric %in% c("euclidean", "cosine", "correlation")) {
        cuda_bruteforce_params(nrow(data), ncol(data), k, metric = metric,
            target_recall = target_recall)
    }
    else {
        NULL
    }
    use_float32_transform <- inputs$use_float32
    use_float32_output <- identical(output, "float") && is.null(metric_inputs)
    brute_distance_output <- if (isTRUE(
        use_float32_output)) output else "double"
    out <- with_faiss_gpu_runtime(brute_params %||% list(), {
        if (isTRUE(use_float32_transform) || isTRUE(use_float32_output)) {
            nn_cuvs_bruteforce_float32_cpp(search_data, search_points,
                as.integer(k), isTRUE(exclude_self), brute_distance_output)
        }
        else {
            nn_cuvs_bruteforce_cpp(search_data, search_points, as.integer(k),
                isTRUE(exclude_self))
        }
    })
    double_route_cuvs_bruteforce_continue(as.list(environment()))
}

double_route_cuvs_bruteforce_continue <- function(context) {
    with(context, {
        resolved_backend <- "cuda_cuvs_bruteforce"
        result_backend <- if (requested_backend %in% c("cuda", "gpu")) {
            requested_backend
        }
        else {
            resolved_backend
        }
        result <- finish_nn_result(out, result_backend, k, self_query,
            exact = TRUE,
            metric = metric)
        if (!is.null(metric_inputs)) {
            result <- finalize_graph_metric_result(result, metric_inputs)
        }
        if (isTRUE(use_float32_transform) || isTRUE(use_float32_output)) {
            result <- finish_float32_direct_result(result, out)
        }
        if (!identical(result_backend, resolved_backend)) {
            attr(result, "resolved_backend") <- resolved_backend
        }
        attr(result, "cuvs") <- double_route_cuvs_bruteforce_metadata_22(
            as.list(environment())
        )
        if (!is.null(brute_params)) {
            result <- attach_cuda_exact_tuning(result, brute_params,
                brute_distance_output, n_threads)
        }
        return(result)
    })
}

double_route_cuvs_nndescent <- function (data, points, k, backend, exclude_self,
    n_threads, metric, tuning, target_recall, output, auto_selection,
    requested_method, requested_backend, self_query)
{
    if (!backend %in% c("cuvs_nndescent", "cuda_cuvs_nndescent",
        "cuda_nndescent"))
        return(NULL)
    double_route_cuvs_nndescent_preflight(as.list(environment()))
    require_cuvs_backend("cuVS NN-descent")
    reject_cuda_r_side_output_cleanup("cuda_cuvs_nndescent", exclude_self)
    metric_inputs <- NULL
    search_data <- data
    if (metric %in% c("cosine", "correlation")) {
        metric_inputs <- normalized_euclidean_metric_inputs(data, points,
            self_query, metric, storage = "float")
        search_data <- metric_inputs$data
    }
    use_float32_transform <- identical(metric_inputs$transform_storage %||%
        "double",
        "float32")
    nonself_k <- if (isTRUE(exclude_self))
        k
    else max(0, k - 1)
    params <- NULL
    double_route_cuvs_nndescent_continue(as.list(environment()))
}

double_route_cuvs_nndescent_continue <- function(context) {
    with(context, {
        if (nonself_k < 1) {
            out <- list(indices = matrix(seq_len(nrow(data)), nrow(data), 1),
                distances = matrix(0, nrow(data), 1))
        }
        else {
            search_dim <- if (is_float32_matrix_input(search_data)) {
                float32_matrix_dims(search_data, "data")
            }
            else {
                dim(search_data)
            }
            params <- cuvs_nndescent_params(search_dim[[1]], search_dim[[2]],
                nonself_k, metric = metric, target_recall = target_recall)
            out <- if (isTRUE(use_float32_transform)) {
                nn_cuvs_nndescent_self_float32_cpp(search_data,
                    as.integer(nonself_k), as.integer(params$graph_degree),
                    as.integer(params$intermediate_graph_degree),
                    as.integer(params$max_iterations), "double")
            }
            else {
                nn_cuvs_nndescent_self_cpp(search_data, as.integer(nonself_k),
                    as.integer(params$graph_degree),
                    as.integer(params$intermediate_graph_degree),
                    as.integer(params$max_iterations))
            }
        }
        result <- finish_nn_result(out, "cuda_cuvs_nndescent", k, self_query,
            exact = FALSE, metric = metric)
        if (!is.null(metric_inputs)) {
            result <- finalize_graph_metric_result(result, metric_inputs)
        }
        if (isTRUE(use_float32_transform)) {
            result <- finish_float32_direct_result(result, out)
        }
        attr(result,
            "approximation") <- double_route_cuvs_nndescent_metadata_23(as.list(
                environment()))
        result <- append_nn_tuning_metadata(result, params)
        return(result)
    })
}

double_route_cpu_nndescent <- function (data, points, k, backend, exclude_self,
    n_threads, metric, tuning, target_recall, output, auto_selection,
    requested_method, requested_backend, self_query)
{
    if (!identical(backend, "cpu_nndescent"))
        return(NULL)
    if (!isTRUE(self_query)) {
        stop("`method = \"nndescent\"` is only available ",
            "for self-KNN searches on CPU.", call. = FALSE)
    }
    metric_inputs <- NULL
    search_data <- data
    if (metric %in% c("cosine", "correlation")) {
        metric_inputs <- normalized_euclidean_metric_inputs(data, points,
            self_query, metric)
        search_data <- metric_inputs$data
    }
    nonself_k <- if (isTRUE(exclude_self))
        k
    else k - 1
    double_route_cpu_nndescent_continue(as.list(environment()))
}

double_route_cpu_nndescent_continue <- function(context) {
    with(context, {
        if (nonself_k < 1) {
            out <- list(indices = matrix(seq_len(nrow(data)), nrow(data), 1),
                distances = matrix(0, nrow(data), 1))
            attr(out,
                "approximation") <- list(
                    strategy = "native_cpu_nndescent_trivial_self",
                backend = "cpu")
        }
        else {
            out <- nndescent_self_knn(search_data, k = nonself_k,
                seed = fast_knn_approx_seed(), n_threads = n_threads,
                metric = if (is.null(metric_inputs))
                metric
            else "euclidean", tuning_metric = metric,
                target_recall = target_recall)
            if (!isTRUE(exclude_self)) {
                out <- prepend_self_neighbor_column(out)
            }
        }
        result <- finish_nn_result(out, "cpu_nndescent", k, self_query,
            exact = FALSE, metric = metric)
        if (!is.null(metric_inputs)) {
            result <- finalize_normalized_euclidean_metric_result(result,
                metric_inputs)
        }
        approximation <- attr(out, "approximation")
        if (is.null(approximation)) {
            approximation <- list()
        }
        approximation$metric <- metric
        approximation$transform <- if (is.null(metric_inputs)) {
            NA_character_
        }
        else {
            metric_inputs$transform
        }
        attr(result, "approximation") <- approximation
        return(result)
    })
}

double_route_cuda_grid <- function (data, points, k, backend, exclude_self,
    n_threads, metric, tuning, target_recall, output, auto_selection,
    requested_method, requested_backend, self_query)
{
    if (!backend %in% c("cuda_grid", "cuda_grid_auto", "gpu_grid",
        "cuda_grid2d", "cuda_grid3d"))
        return(NULL)
    if (!isTRUE(self_query)) {
        stop("`backend = \"cuda_grid_auto\"` is only ",
            "available for self-KNN searches.", call. = FALSE)
    }
    if (identical(metric, "inner_product")) {
        stop("`backend = \"cuda_grid_auto\"` does not ",
            "support inner-product search.", call. = FALSE)
    }
    if (!ncol(data) %in% c(2, 3)) {
        stop("`backend = \"cuda_grid_auto\"` supports only ",
            "two- or three-column matrices.", call. = FALSE)
    }
    if (!isTRUE(cuda_available())) {
        stop("No CUDA GPU backend is available on this machine.", call. = FALSE)
    }
    metric_inputs <- NULL
    search_data <- data
    if (metric %in% c("cosine", "correlation")) {
        metric_inputs <- normalized_euclidean_metric_inputs(data, points,
            self_query, metric)
        search_data <- metric_inputs$data
    }
    include_self <- !isTRUE(exclude_self)
    double_route_cuda_grid_continue(as.list(environment()))
}

double_route_cuda_grid_continue <- function(context) {
    with(context, {
        nonself_k <- if (include_self)
            k - 1
        else k
        bins <- grid_bins_per_dim(nrow(search_data), max(1, nonself_k),
            ncol(search_data))
        out <- cuda_grid_self_knn_cpp(search_data, as.integer(k),
            as.integer(bins),
            isTRUE(include_self))
        resolved <- if (ncol(data) == 3)
            "cuda_grid3d"
        else "cuda_grid2d"
        result <- finish_nn_result(out, resolved, k, self_query, exact = TRUE,
            metric = metric)
        if (!is.null(metric_inputs)) {
            result <- finalize_normalized_euclidean_metric_result(result,
                metric_inputs)
        }
        attr(result, "spatial_index") <- list(strategy = if (ncol(data) == 3) {
            "native_cuda_exact_uniform_grid_3d"
        } else {
            "native_cuda_exact_uniform_grid_2d"
        }, backend = resolved, exact = TRUE,
            metric_transform = if (is.null(metric_inputs)) {
            NA_character_
        } else {
            metric_inputs$transform
        }, bins_per_dim = as.integer(out$bins_per_dim),
            n_cells = as.integer(out$n_cells),
            self_column_included = isTRUE(out$self_column_included),
            output_layout = out$output_layout %||% "knn_matrix_final",
            r_side_reshaping = FALSE)
        return(result)
    })
}

double_route_cpu_grid <- function (data, points, k, backend, exclude_self,
    n_threads, metric, tuning, target_recall, output, auto_selection,
    requested_method, requested_backend, self_query)
{
    if (!backend %in% c("grid", "cpu_grid", "grid2d", "cpu_grid2d", "grid3d",
        "cpu_grid3d"))
        return(NULL)
    if (!isTRUE(self_query)) {
        stop("`backend = \"cpu_grid\"` is only available ",
            "for self-KNN searches.", call. = FALSE)
    }
    if (identical(metric, "inner_product")) {
        stop("`backend = \"cpu_grid\"` does not support ",
            "inner-product search.", call. = FALSE)
    }
    metric_inputs <- NULL
    search_data <- data
    if (metric %in% c("cosine", "correlation")) {
        metric_inputs <- normalized_euclidean_metric_inputs(data, points,
            self_query, metric)
        search_data <- metric_inputs$data
    }
    grid_backend <- backend
    if (!is.null(metric_inputs) && backend %in% c("grid", "cpu_grid")) {
        grid_backend <- if (ncol(search_data) == 3) {
            "cpu_grid3d"
        }
        else {
            "cpu_grid2d"
        }
    }
    out <- grid_self_knn(search_data, k = k, backend = grid_backend,
        exclude_self = isTRUE(exclude_self), n_threads = n_threads)
    result <- finish_nn_result(out, attr(out, "spatial_index")$backend, k,
        self_query, exact = TRUE, metric = metric)
    if (!is.null(metric_inputs)) {
        result <- finalize_normalized_euclidean_metric_result(result,
            metric_inputs)
    }
    attr(result, "spatial_index") <- attr(out, "spatial_index")
    attr(result,
        "spatial_index")$metric_transform <- if (is.null(metric_inputs)) {
        NA_character_
    }
    else {
        metric_inputs$transform
    }
    return(result)
}

float32_route_grid_metadata_1 <- function (context)
with(context,
    list(strategy = paste0(if (isTRUE(
        use_cuda)) "native_cuda" else "native_cpu",
    "_exact_uniform_grid_", search_dim[[2]], "d"), backend = resolved,
    exact = TRUE, metric_transform = if (is.null(metric_inputs)) {
    NA_character_
} else {
    metric_inputs$transform
}, bins_per_dim = as.integer(out$bins_per_dim),
    n_cells = as.integer(out$n_cells),
    self_column_included = isTRUE(out$self_column_included),
    output_layout = out$output_layout %||% "knn_matrix_final",
    r_side_reshaping = FALSE, input_type = "float32"))

float32_route_cuvs_bruteforce_metadata_2 <- function (context)
with(context, list(index_type = as.character(out$index_type), library = "cuvs",
    backend = "cuda", resolved_backend = resolved_backend, metric = metric,
    input_type = "float32", transform = if (is.null(metric_inputs)) {
    NA_character_
} else {
    metric_inputs$transform
}, distance_transform = if (is.null(metric_inputs)) {
    NA_character_
} else {
    (metric_inputs$distance_transform %||%
        .normalized_similarity_distance_transform)
}, transform_storage = metric_inputs$transform_storage %||% "float32",
    transform_cache = metric_inputs$transform_cache %||% NULL))

float32_route_cuvs_cagra_metadata_3 <- function (context)
with(context, list(strategy = "rapids_cuvs_cagra", backend = resolved_backend,
    library = "cuvs", accelerator = "cuda", cagra_provider = "cuvs",
    cagra_provider_option = cagra_implementation_preference(), metric = metric,
    transform = if (is.null(metric_inputs)) {
    NA_character_
} else {
    metric_inputs$transform
}, distance_transform = if (is.null(metric_inputs)) {
    NA_character_
} else {
    (metric_inputs$distance_transform %||%
        .normalized_similarity_distance_transform)
},
    input_type = out$input_type %||% if (isTRUE(
        use_float32_input)) "float32" else "double",
    graph_degree = as.integer(out$graph_degree),
    intermediate_graph_degree = as.integer(out$intermediate_graph_degree),
    search_width = as.integer(out$search_width),
    itopk_size = as.integer(out$itopk_size),
    cagra_build_algo = out$build_algo %||% build_algo,
    search_batch_size = as.integer(out$search_batch_size),
    transform_storage = metric_inputs$transform_storage %||% if (isTRUE(
        use_float32_input)) "float32" else "double",

    transform_cache = metric_inputs$transform_cache %||% NULL))

float32_route_faiss_gpu_cagra_metadata_4 <- function (context)
with(context, list(strategy = "faiss_gpu_GpuIndexCagra_cuVS",
    backend = "faiss_gpu_cagra", library = "faiss", accelerator = "cuda",
    cagra_provider = "faiss_gpu",
    cagra_provider_option = cagra_implementation_preference(), metric = metric,
    transform = if (is.null(metric_inputs)) {
    NA_character_
} else {
    metric_inputs$transform
}, metric_transform = if (is.null(metric_inputs)) {
    NA_character_
} else {
    metric_inputs$transform
}, distance_transform = if (is.null(metric_inputs)) {
    NA_character_
} else {
    (metric_inputs$distance_transform %||%
        .normalized_similarity_distance_transform)
},
    input_type = out$input_type %||% if (isTRUE(
        use_float32_input)) "float32" else "double",
    graph_degree = as.integer(out$graph_degree),
    intermediate_graph_degree = as.integer(out$intermediate_graph_degree),
    search_width = as.integer(out$search_width),
    itopk_size = as.integer(out$itopk_size),
    transform_storage = metric_inputs$transform_storage %||% if (isTRUE(
        use_float32_input)) "float32" else "double",
    transform_cache = metric_inputs$transform_cache %||% NULL))

cuvs_fastscan_metric_metadata <- function(context, input_type = NULL) {
    with(context, {
        transform <- if (is.null(metric_inputs)) {
            NA_character_
        } else {
            metric_inputs$transform
        }
        metadata <- list(
            strategy = "rapids_cuvs_ivf_pq_4bit",
            backend = "cuda_cuvs_ivfpq_fastscan",
            library = "cuvs",
            accelerator = "cuda",
            metric = metric,
            transform = transform,
            metric_transform = transform,
            distance_transform = if (is.null(metric_inputs)) {
                NA_character_
            } else {
                metric_inputs$distance_transform %||%
                    .normalized_similarity_distance_transform
            },
            ivfpq_fastscan = TRUE,
            fastscan = FALSE,
            note = cuvs_fastscan_note(metric)
        )
        if (!is.null(input_type)) metadata$input_type <- input_type
        metadata
    })
}

cuvs_fastscan_note <- function(metric) {
    switch(metric,
        cosine = paste0(
            "CUDA cosine FastScan route row-normalizes to float32, ",
            "then uses cuVS IVF-PQ with 4-bit compressed codes."
        ),
        correlation = paste0(
            "CUDA correlation FastScan route row-centers and ",
            "row-normalizes to float32, then uses cuVS IVF-PQ ",
            "with 4-bit compressed codes."
        ),
        inner_product = paste0(
            "CUDA raw-inner-product FastScan route applies the ",
            "maximum-inner-product-to-L2 extra-dimension transform, ",
            "then uses cuVS IVF-PQ with 4-bit compressed codes."
        ),
        paste0(
            "CUDA route uses cuVS IVF-PQ with 4-bit compressed codes; ",
            "FAISS FastScan is CPU-only in this package route."
        )
    )
}

cuvs_fastscan_parameter_metadata <- function(context) {
    with(context, list(
        nlist = as.integer(out$n_lists),
        nprobe = as.integer(out$n_probes),
        requested_nlist = as.integer(params$ivf$requested_nlist),
        requested_nprobe = as.integer(params$ivf$requested_nprobe),
        pq_dim = as.integer(out$pq_dim),
        pq_bits = as.integer(out$pq_bits),
        requested_pq_dim = as.integer(params$pq$requested_pq_dim),
        requested_pq_bits = as.integer(params$pq$requested_pq_bits),
        ivf_parameters_adjusted = !identical(
            as.integer(params$ivf$requested_nlist), as.integer(out$n_lists)
        ) || !identical(
            as.integer(params$ivf$requested_nprobe), as.integer(out$n_probes)
        ),
        pq_parameters_adjusted = isTRUE(out$pq_parameters_adjusted) ||
            !identical(
                as.integer(params$pq$requested_pq_dim), as.integer(out$pq_dim)
            ) || !identical(
                as.integer(params$pq$requested_pq_bits), as.integer(out$pq_bits)
            ),
        pq_alignment_adjusted = isTRUE(out$pq_alignment_adjusted) ||
            isTRUE(params$pq$pq_alignment_adjusted),
        pq_alignment_rule = out$pq_alignment_rule %||%
            params$pq$pq_alignment_rule %||% NA_character_,
        search_batch_size = as.integer(out$search_batch_size),
        transform_storage = metric_inputs$transform_storage %||% "double",
        transform_cache = metric_inputs$transform_cache %||% NULL
    ))
}

float32_route_cuvs_fastscan_metadata_5 <- function(context) {
    c(
        cuvs_fastscan_metric_metadata(context, "float32"),
        cuvs_fastscan_parameter_metadata(context),
        context$cache_meta
    )
}

float32_route_cuvs_nndescent_metadata_6 <- function (context)
with(context, list(strategy = "rapids_cuvs_nndescent",
    backend = "cuda_cuvs_nndescent", library = "cuvs", accelerator = "cuda",
    metric = metric, transform = if (is.null(metric_inputs)) {
    NA_character_
} else {
    metric_inputs$transform
}, distance_transform = if (is.null(metric_inputs)) {
    NA_character_
} else {
    (metric_inputs$distance_transform %||%
        .normalized_similarity_distance_transform)
}, transform_storage = if (is.null(metric_inputs)) {
    "float32"
} else {
    (metric_inputs$transform_storage %||% "float32")
}, transform_cache = if (is.null(metric_inputs)) {
    NULL
} else {
    (metric_inputs$transform_cache %||% NULL)
}, input_type = "float32",
    graph_degree = as.integer(out$graph_degree %||% NA_integer_),
    intermediate_graph_degree = as.integer(out$intermediate_graph_degree %||%
        NA_integer_),
    max_iterations = as.integer(out$max_iterations %||% NA_integer_),
    cuvs_nndescent_input_dtype = out$cuvs_nndescent_input_dtype %||% "float32",
    cuvs_nndescent_shared_memory_workaround = isTRUE(
        out$cuvs_nndescent_shared_memory_workaround),
    cuvs_nndescent_l2_norm_shared_bytes_fp32 = as.numeric(
        out$cuvs_nndescent_l2_norm_shared_bytes_fp32 %||%
    NA_real_),
        cuvs_nndescent_l2_norm_shared_bytes_used = as.numeric(
            out$cuvs_nndescent_l2_norm_shared_bytes_used %||% NA_real_)))

float32_route_cpu_nndescent_metadata_7 <- function (context)
with(context, list(strategy = "native_cpu_nndescent_trivial_self",
    backend = "cpu"))

float32_route_cpu_nndescent_metadata_8 <- function (context)
with(context, approximation)

float32_route_native_nsg_metadata_9 <- function (context)
with(context, list(seed_backend = "trivial_self", candidate_columns = 0,
    seed_graph_k = 0, protected_seed_neighbors = 0, exact_mrng_prune = TRUE))

float32_route_native_nsg_metadata_10 <- function (context)
with(context, c(list(strategy = if (isTRUE(use_cuda)) {
    "native_cuda_nsg_candidate_graph"
} else {
    "native_cpu_nsg_candidate_graph"
}, backend = backend, accelerator = if (isTRUE(use_cuda)) "cuda" else "cpu",
    metric = metric, input_type = "float32",
    transform = if (is.null(metric_inputs)) {
    NA_character_
} else {
    metric_inputs$transform
}, r = as.integer(params$r), graph_k = as.integer(params$graph_k),
    requested_r = as.integer(params$requested_r),
    requested_graph_k = as.integer(params$requested_graph_k),
    requested_seed_backend = params$seed_backend %||% NA_character_,
    seed_k = as.integer(params$seed_k %||% params$graph_k),
    graph_k_cap = as.integer(params$graph_k_cap),
    nsg_parameters_adjusted = !identical(as.integer(params$r),
    as.integer(params$requested_r)) || !identical(as.integer(params$graph_k),
    as.integer(params$requested_graph_k)),
    tuning_policy = params$tuning_policy, tuning_rule = params$tuning_rule,
        tuning_metric = params$tuning_metric %||% metric,
        tuning_shape_group = params$tuning_shape_group %||% NA_character_,
        tuning_k_bucket = as.integer(params$tuning_k_bucket %||% NA_integer_),
        tuning_target_recall_code = as.integer(
            params$tuning_target_recall_code %||% NA_integer_),
        tuning_benchmark_basis = params$tuning_benchmark_basis %||%
            NA_character_,
        tuning_benchmark_target_met = isTRUE(
            params$tuning_benchmark_target_met),

    tuning_benchmark_source = params$tuning_benchmark_source %||% NA_character_,
        target_recall = as.numeric(params$target_recall %||% NA_real_),
        requested_target_recall = as.numeric(params$requested_target_recall %||%
            NA_real_),
        tuning_large_k = isTRUE(params$tuning_large_k),
        tuning_high_dim = isTRUE(params$tuning_high_dim),
        tuning_source = params$tuning_source %||% "cpp"), approx))

float32_route_native_vamana_metadata_11 <- function (context)
with(context, list(seed_backend = "trivial_self", candidate_columns = 0,
    seed_search_l = 0, alpha = as.numeric(params$alpha),
    protected_seed_neighbors = 0, exact_robust_prune = TRUE,
    cuvs_vamana_note = paste0("cuVS Vamana currently builds/serializes ",
    "DiskANN-compatible graphs; faissR performs ",
    "KNN refinement inside the candidate graph.")))

float32_route_native_vamana_metadata_12 <- function (context)
with(context, c(list(strategy = if (isTRUE(use_cuda)) {
    "native_vamana_candidate_graph_cuda_refine"
} else {
    "native_vamana_candidate_graph"
}, backend = backend, accelerator = if (isTRUE(use_cuda)) "cuda" else "cpu",
    metric = metric, input_type = "float32",
    transform = if (is.null(metric_inputs)) {
    NA_character_
} else {
    metric_inputs$transform
}, r = as.integer(params$r), search_l = as.integer(params$search_l),
    alpha = as.numeric(params$alpha),
    requested_r = as.integer(params$requested_r),
    requested_search_l = as.integer(params$requested_search_l),
    requested_alpha = as.numeric(params$requested_alpha),
    requested_seed_backend = params$seed_backend %||% NA_character_,
    seed_k = as.integer(params$seed_k %||% params$search_l),
    tuning_policy = params$tuning_policy, tuning_rule = params$tuning_rule,
    tuning_metric = params$tuning_metric %||% metric,
    tuning_shape_group = params$tuning_shape_group %||% NA_character_,
        tuning_k_bucket = as.integer(params$tuning_k_bucket %||% NA_integer_),
        tuning_target_recall_code = as.integer(
            params$tuning_target_recall_code %||% NA_integer_),
        tuning_benchmark_basis = params$tuning_benchmark_basis %||%
            NA_character_,
        tuning_benchmark_target_met = isTRUE(
            params$tuning_benchmark_target_met),
        tuning_benchmark_source = params$tuning_benchmark_source %||%
            NA_character_,
        target_recall = as.numeric(params$target_recall %||%
        NA_real_),
            requested_target_recall = as.numeric(
                params$requested_target_recall %||% NA_real_),
            tuning_large_k = isTRUE(params$tuning_large_k),
            tuning_high_dim = isTRUE(params$tuning_high_dim),
            tuning_source = params$tuning_source %||% "cpp"), approx))

double_route_faiss_gpu_cagra_metadata_13 <- function (context)
with(context, list(strategy = "faiss_gpu_GpuIndexCagra_cuVS",
    backend = "faiss_gpu_cagra", library = "faiss", accelerator = "cuda",
    cagra_provider = "faiss_gpu",
    cagra_provider_option = cagra_implementation_preference(), metric = metric,
    transform = if (is.null(metric_inputs)) {
    NA_character_
} else {
    metric_inputs$transform
}, metric_transform = if (is.null(metric_inputs)) {
    NA_character_
} else {
    metric_inputs$transform
}, distance_transform = if (is.null(metric_inputs)) {
    NA_character_
} else {
    metric_inputs$distance_transform %||%
        .normalized_similarity_distance_transform
}, graph_degree = as.integer(out$graph_degree),
    intermediate_graph_degree = as.integer(out$intermediate_graph_degree),
    search_width = as.integer(out$search_width),
    itopk_size = as.integer(out$itopk_size),
    requested_graph_degree = as.integer(requested_graph_degree),
    requested_intermediate_graph_degree = as.integer(
        requested_intermediate_graph_degree),
    requested_search_width = as.integer(requested_search_width),
    requested_itopk_size = as.integer(requested_itopk_size),
    cagra_parameters_adjusted = isTRUE(out$cagra_parameters_adjusted) ||
    !identical(as.integer(requested_graph_degree),
        as.integer(out$graph_degree)) || !identical(as.integer(
            requested_intermediate_graph_degree),
        as.integer(out$intermediate_graph_degree)) || !identical(as.integer(
            requested_search_width),
        as.integer(out$search_width)) || !identical(as.integer(
            requested_itopk_size),
        as.integer(out$itopk_size)),
        transform_storage = metric_inputs$transform_storage %||% "double",
        transform_cache = metric_inputs$transform_cache %||% NULL))

double_route_native_nsg_metadata_14 <- function (context)
with(context, list(seed_backend = "trivial_self", candidate_columns = 0,
    seed_graph_k = 0, protected_seed_neighbors = 0, exact_mrng_prune = TRUE))

double_route_native_nsg_metadata_15 <- function (context)
with(context, c(list(strategy = if (isTRUE(use_cuda)) {
    "native_cuda_nsg_candidate_graph"
} else {
    "native_cpu_nsg_candidate_graph"
}, backend = backend, accelerator = if (isTRUE(use_cuda)) "cuda" else "cpu",
    metric = metric, transform = if (is.null(metric_inputs)) {
    NA_character_
} else {
    metric_inputs$transform
}, r = as.integer(params$r), graph_k = as.integer(params$graph_k),
    requested_r = as.integer(params$requested_r),
    requested_graph_k = as.integer(params$requested_graph_k),
    requested_seed_backend = params$seed_backend %||% NA_character_,
    seed_k = as.integer(params$seed_k %||% params$graph_k),
    graph_k_cap = as.integer(params$graph_k_cap),
    nsg_parameters_adjusted = !identical(as.integer(params$r),
    as.integer(params$requested_r)) || !identical(as.integer(params$graph_k),
    as.integer(params$requested_graph_k)),
    tuning_policy = params$tuning_policy, tuning_rule = params$tuning_rule,
        tuning_metric = params$tuning_metric %||% metric,
        tuning_shape_group = params$tuning_shape_group %||% NA_character_,
        tuning_k_bucket = as.integer(params$tuning_k_bucket %||% NA_integer_),
        tuning_target_recall_code = as.integer(
            params$tuning_target_recall_code %||% NA_integer_),
        tuning_benchmark_basis = params$tuning_benchmark_basis %||%
            NA_character_,
        tuning_benchmark_target_met = isTRUE(
            params$tuning_benchmark_target_met),

    tuning_benchmark_source = params$tuning_benchmark_source %||% NA_character_,
        target_recall = as.numeric(params$target_recall %||% NA_real_),
        requested_target_recall = as.numeric(params$requested_target_recall %||%
            NA_real_),
        tuning_large_k = isTRUE(params$tuning_large_k),
        tuning_high_dim = isTRUE(params$tuning_high_dim),
        tuning_source = params$tuning_source %||% "cpp"), approx))

double_route_native_vamana_metadata_16 <- function (context)
with(context, list(seed_backend = "trivial_self", candidate_columns = 0,
    seed_search_l = 0, alpha = as.numeric(params$alpha),
    protected_seed_neighbors = 0, exact_robust_prune = TRUE,
    cuvs_vamana_note = paste0("cuVS Vamana currently builds/serializes ",
    "DiskANN-compatible graphs; faissR performs ",
    "KNN refinement inside the candidate graph.")))

double_route_native_vamana_metadata_17 <- function (context)
with(context, c(list(strategy = if (isTRUE(use_cuda)) {
    "native_vamana_candidate_graph_cuda_refine"
} else {
    "native_vamana_candidate_graph"
}, backend = backend, accelerator = if (isTRUE(use_cuda)) "cuda" else "cpu",
    metric = metric, transform = if (is.null(metric_inputs)) {
    NA_character_
} else {
    metric_inputs$transform
}, r = as.integer(params$r), search_l = as.integer(params$search_l),
    alpha = as.numeric(params$alpha),
    requested_r = as.integer(params$requested_r),
    requested_search_l = as.integer(params$requested_search_l),
    requested_alpha = as.numeric(params$requested_alpha),
    requested_seed_backend = params$seed_backend %||% NA_character_,
    seed_k = as.integer(params$seed_k %||% params$search_l),
    tuning_policy = params$tuning_policy, tuning_rule = params$tuning_rule,
    tuning_metric = params$tuning_metric %||% metric,
    tuning_shape_group = params$tuning_shape_group %||% NA_character_,
        tuning_k_bucket = as.integer(params$tuning_k_bucket %||% NA_integer_),
        tuning_target_recall_code = as.integer(
            params$tuning_target_recall_code %||% NA_integer_),
        tuning_benchmark_basis = params$tuning_benchmark_basis %||%
            NA_character_,
        tuning_benchmark_target_met = isTRUE(
            params$tuning_benchmark_target_met),
        tuning_benchmark_source = params$tuning_benchmark_source %||%
            NA_character_,
        target_recall = as.numeric(params$target_recall %||%
        NA_real_),
            requested_target_recall = as.numeric(
                params$requested_target_recall %||% NA_real_),
            tuning_large_k = isTRUE(params$tuning_large_k),
            tuning_high_dim = isTRUE(params$tuning_high_dim),
            tuning_source = params$tuning_source %||% "cpp"), approx))

double_route_cuvs_cagra_metadata_18 <- function (context)
with(context, list(strategy = "rapids_cuvs_cagra", backend = resolved_backend,
    library = "cuvs", accelerator = "cuda", cagra_provider = "cuvs",
    cagra_provider_option = cagra_implementation_preference(), metric = metric,
    transform = if (is.null(metric_inputs)) {
    NA_character_
} else {
    metric_inputs$transform
}, distance_transform = if (is.null(metric_inputs)) {
    NA_character_
} else {
    metric_inputs$distance_transform %||%
        .normalized_similarity_distance_transform
}, graph_degree = as.integer(out$graph_degree),
    intermediate_graph_degree = as.integer(out$intermediate_graph_degree),
    search_width = as.integer(out$search_width),
    itopk_size = as.integer(out$itopk_size),
    cagra_build_algo = out$build_algo %||% cagra_build_algo_preference(),
    nn_descent_niter = as.integer(out$nn_descent_niter %||% NA_integer_),
    requested_graph_degree = as.integer(requested_graph_degree),
    requested_intermediate_graph_degree = as.integer(
        requested_intermediate_graph_degree),
    requested_search_width = as.integer(requested_search_width),
    requested_itopk_size = as.integer(requested_itopk_size),
        cagra_parameters_adjusted = isTRUE(out$cagra_parameters_adjusted) ||
            !identical(as.integer(requested_graph_degree),
        as.integer(out$graph_degree)) || !identical(as.integer(
            requested_intermediate_graph_degree),
        as.integer(out$intermediate_graph_degree)) || !identical(as.integer(
            requested_search_width),
        as.integer(out$search_width)) || !identical(as.integer(
            requested_itopk_size),
        as.integer(out$itopk_size)),
        search_batch_size = as.integer(out$search_batch_size),
    tuning = tuning_metadata,
        transform_storage = metric_inputs$transform_storage %||% "double",
        transform_cache = metric_inputs$transform_cache %||% NULL))

double_route_cuvs_ivf_metadata_19 <- function (context)
with(context, list(strategy = "rapids_cuvs_ivf_flat",
    backend = "cuda_cuvs_ivf_flat", library = "cuvs", accelerator = "cuda",
    metric = metric, transform = if (is.null(metric_inputs)) {
    NA_character_
} else {
    metric_inputs$transform
}, metric_transform = if (is.null(metric_inputs)) {
    NA_character_
} else {
    metric_inputs$transform
}, distance_transform = if (is.null(metric_inputs)) {
    NA_character_
} else {
    (metric_inputs$distance_transform %||%
        .normalized_similarity_distance_transform)
}, default_candidate = FALSE, nlist = as.integer(out$n_lists),
    nprobe = as.integer(out$n_probes),
    requested_nlist = as.integer(params$requested_nlist),
    requested_nprobe = as.integer(params$requested_nprobe),
    ivf_parameters_adjusted = !identical(as.integer(params$requested_nlist),
    as.integer(out$n_lists)) || !identical(as.integer(params$requested_nprobe),
    as.integer(out$n_probes)),
    search_batch_size = as.integer(out$search_batch_size),
    transform_storage = metric_inputs$transform_storage %||% "double",
    transform_cache = metric_inputs$transform_cache %||% NULL))

double_route_cuvs_fastscan_metadata_20 <- function(context) {
    c(
        cuvs_fastscan_metric_metadata(context),
        cuvs_fastscan_parameter_metadata(context),
        context$cache_meta
    )
}

double_route_cuvs_ivfpq_metadata_21 <- function (context)
with(context, list(strategy = "rapids_cuvs_ivf_pq", backend = "cuda_cuvs_ivfpq",
    library = "cuvs", accelerator = "cuda", metric = metric,
    transform = if (is.null(metric_inputs)) {
    NA_character_
} else {
    metric_inputs$transform
}, metric_transform = if (is.null(metric_inputs)) {
    NA_character_
} else {
    metric_inputs$transform
}, distance_transform = if (is.null(metric_inputs)) {
    NA_character_
} else {
    (metric_inputs$distance_transform %||%
        .normalized_similarity_distance_transform)
}, role = "explicit_memory_pressure_backend", default_candidate = FALSE,
    nlist = as.integer(out$n_lists), nprobe = as.integer(out$n_probes),
    requested_nlist = as.integer(params$requested_nlist),
    requested_nprobe = as.integer(params$requested_nprobe),
    ivf_parameters_adjusted = !identical(as.integer(params$requested_nlist),
    as.integer(out$n_lists)) || !identical(as.integer(params$requested_nprobe),
    as.integer(out$n_probes)), pq_dim = as.integer(out$pq_dim),
    pq_bits = as.integer(out$pq_bits),
    requested_pq_dim = as.integer(pq$requested_pq_dim),
    requested_pq_bits = as.integer(pq$requested_pq_bits),
        pq_parameters_adjusted = isTRUE(out$pq_parameters_adjusted) ||
            !identical(as.integer(pq$requested_pq_dim),
        as.integer(out$pq_dim)) || !identical(as.integer(pq$requested_pq_bits),
        as.integer(out$pq_bits)),
        pq_alignment_adjusted = isTRUE(out$pq_alignment_adjusted) || isTRUE(
            pq$pq_alignment_adjusted),
        pq_alignment_rule = out$pq_alignment_rule %||% pq$pq_alignment_rule %||%
            NA_character_,
        search_batch_size = as.integer(out$search_batch_size),
    transform_storage = metric_inputs$transform_storage %||% "double",
        transform_cache = metric_inputs$transform_cache %||% NULL))

double_route_cuvs_bruteforce_metadata_22 <- function (context)
with(context, list(index_type = as.character(out$index_type), library = "cuvs",
    backend = "cuda", resolved_backend = resolved_backend, metric = metric,
    transform = if (is.null(metric_inputs)) {
    NA_character_
} else {
    metric_inputs$transform
}, distance_transform = if (is.null(metric_inputs)) {
    NA_character_
} else {
    (metric_inputs$distance_transform %||%
        .normalized_similarity_distance_transform)
}, transform_storage = metric_inputs$transform_storage %||% "double",
    transform_cache = metric_inputs$transform_cache %||% NULL))

double_route_cuvs_nndescent_metadata_23 <- function (context)
with(context, list(strategy = "rapids_cuvs_nndescent",
    backend = "cuda_cuvs_nndescent", library = "cuvs", metric = metric,
    transform = if (is.null(metric_inputs)) {
    NA_character_
} else {
    metric_inputs$transform
}, graph_degree = as.integer(out$graph_degree),
    intermediate_graph_degree = as.integer(out$intermediate_graph_degree),
    max_iterations = as.integer(out$max_iterations),
    cuvs_nndescent_input_dtype = out$cuvs_nndescent_input_dtype %||% "float32",
    cuvs_nndescent_shared_memory_workaround = isTRUE(
        out$cuvs_nndescent_shared_memory_workaround),
    cuvs_nndescent_l2_norm_shared_bytes_fp32 = as.numeric(
        out$cuvs_nndescent_l2_norm_shared_bytes_fp32 %||% NA_real_),
    cuvs_nndescent_l2_norm_shared_bytes_used = as.numeric(
        out$cuvs_nndescent_l2_norm_shared_bytes_used %||%
    NA_real_),
        transform_storage = metric_inputs$transform_storage %||% "double",
        transform_cache = metric_inputs$transform_cache %||% NULL))

float32_route_cuvs_fastscan_preflight <- function (context)
with(context, {
    if (!metric %in% c("euclidean", "cosine", "correlation", "inner_product")) {
        stop("float32 CUDA cuVS IVFPQ FastScan input ",
            "currently supports `metric = \"euclidean\"`, ",
            "`\"cosine\"`, `\"correlation\"`, or ", "`\"inner_product\"`.",
            call. = FALSE)
    }
})

float32_route_cpu_nndescent_preflight <- function (context)
with(context, {
    if (!isTRUE(self_query)) {
        stop("`method = \"nndescent\"` is only available ",
            "for self-KNN searches on CPU.", call. = FALSE)
    }
})

float32_route_native_nsg_preflight <- function (context)
with(context, {
    if (!isTRUE(self_query)) {
        stop("Native NSG is currently implemented for ",
            "self-KNN searches only.", call. = FALSE)
    }
    if (isTRUE(use_cuda) && !isTRUE(cuda_available())) {
        stop("No CUDA GPU backend is available on this machine.", call. = FALSE)
    }
})

float32_route_native_vamana_preflight <- function (context)
with(context, {
    if (!isTRUE(self_query)) {
        stop("Vamana is currently implemented for self-KNN ", "searches only.",
            call. = FALSE)
    }
    if (isTRUE(use_cuda) && !isTRUE(cuda_available())) {
        stop("No CUDA GPU backend is available on this machine.", call. = FALSE)
    }
})

double_route_faiss_gpu_cagra_preflight <- function (context)
with(context, {
    if (!isTRUE(faiss_gpu_available())) {
        stop("The real FAISS GPU CAGRA backend is not ",
            "available in this build. ",
            "Reinstall faissR with FAISS GPU/cuVS headers ",
            "available through `FAISS_HOME`.", call. = FALSE)
    }
})

double_route_cuvs_fastscan_preflight <- function (context)
with(context, {
    if (!metric %in% c("euclidean", "cosine", "correlation", "inner_product")) {
        stop("CUDA cuVS IVFPQ FastScan input currently ",
            "supports `metric = \"euclidean\"`, ",
            "`\"cosine\"`, `\"correlation\"`, or ", "`\"inner_product\"`.",
            call. = FALSE)
    }
})

double_route_cuvs_nndescent_preflight <- function (context)
with(context, {
    if (identical(metric, "inner_product")) {
        stop("cuVS NN-descent does not support raw ",
            "inner-product self-KNN: its ",
            "graph-construction API accepts one symmetric ",
            "L2 dataset, while exact ",
            "maximum-inner-product reduction requires ",
            "distinct reference and query transforms.", call. = FALSE)
    }
    if (!isTRUE(self_query)) {
        stop("`backend = \"cuda_cuvs_nndescent\"` is only ",
            "available for self-KNN searches.", call. = FALSE)
    }
})


prepare_double_nn_input <- function(
    data, points, k, points_missing, exclude_self, n_threads, metric
) {
    data <- as.matrix(data)
    storage.mode(data) <- "double"
    if (isTRUE(points_missing)) {
        points <- data
    } else {
        points <- as.matrix(points)
        storage.mode(points) <- "double"
    }
    if (!identical(ncol(data), ncol(points))) {
        stop("`data` and `points` must have the same number of columns.",
            call. = FALSE)
    }
    if (nrow(data) < 1L || nrow(points) < 1L) {
        stop("`data` and `points` must have at least one row.", call. = FALSE)
    }
    self_query <- isTRUE(points_missing) || identical(data, points)
    if (isTRUE(exclude_self) && !isTRUE(self_query)) {
        stop("Self-neighbor exclusion is only valid when `points` is `data`.",
            call. = FALSE)
    }
    if (is.null(k)) {
        k <- nn_auto_default_k(nrow(data), self_query, exclude_self)
    }
    k <- normalize_nn_positive_integer(
        k, "k", "`k` must be NULL or a positive integer."
    )
    max_k <- nrow(data) - as.integer(isTRUE(exclude_self))
    if (k > max_k) {
        stop("`k` cannot be larger than the available neighbor count.",
            call. = FALSE)
    }
    if (!all(is.finite(data)) || !all(is.finite(points))) {
        stop("`data` and `points` must contain only finite values.",
            call. = FALSE)
    }
    list(
        data = data, points = points, k = k, self_query = self_query,
        n_threads = normalize_nn_threads(n_threads),
            metric = normalize_nn_metric(metric)
    )
}

metric_routable_backends <- function() {
    c(
        "auto", "cpu", "cpu_auto", "faiss_hnsw", "faiss_ivf",
        "faiss_ivf_flat", "faiss_ivfpq", "faiss_ivfpq_fastscan",
        "faiss_nsg", "faiss_nndescent", "cpu_nsg", "cpu_vamana",
        "cuda_vamana", "cuda_nsg", "cpu_nndescent", "cuda_nndescent",
        "cpu_faiss_index_ivf", "faiss_gpu_ivf", "faiss_gpu_ivf_flat",
        "cuda_faiss_ivf_flat", "faiss_gpu_ivfpq", "cuda_faiss_ivfpq",
        "faiss_gpu_cagra", "cuda_faiss_cagra", "cuda_cuvs_cagra",
        "cuda_cagra", "gpu_cagra", "cuvs_ivf", "cuda_cuvs_ivf",
        "cuvs_ivf_flat", "cuda_cuvs_ivf_flat", "cuvs_ivfpq",
        "cuda_cuvs_ivfpq", "cuvs_ivf_pq", "cuda_cuvs_ivf_pq",
        "cuda_cuvs_ivfpq_fastscan", "cuvs_ivfpq_fastscan",
        "cuvs_bruteforce", "cuda_cuvs_bruteforce", "cuda_cuvs_exact",
        "cuda_cuvs_nndescent", "cuvs_nndescent", "cuda_cuvs_hnsw",
        "cuvs_hnsw"
    )
}

grid_route_backends <- function() {
    c(
        "grid", "cpu_grid", "grid2d", "cpu_grid2d", "grid3d",
        "cpu_grid3d", "cuda_grid", "cuda_grid_auto", "gpu_grid",
        "cuda_grid2d", "cuda_grid3d"
    )
}

normalize_double_metric_backend <- function(metric, backend) {
    if (identical(metric, "euclidean")) return(backend)
    cpu_flat <- c("faiss_flat_l2", "faiss_flat", "cpu_faiss_flat")
    gpu_flat <- c("faiss_gpu_flat_l2", "faiss_gpu_flat", "cuda_faiss_flat_l2")
    if (identical(metric, "inner_product") && backend %in% cpu_flat) {
        return("faiss_flat_ip")
    }
    if (identical(metric, "inner_product") && backend %in% gpu_flat) {
        return("faiss_gpu_flat_ip")
    }
    if (identical(metric, "cosine") && backend %in% cpu_flat) {
        return("faiss_flat_cosine")
    }
    if (identical(metric, "correlation") && backend %in% cpu_flat) {
        return("faiss_flat_correlation")
    }
    if (identical(metric, "cosine") && backend %in% gpu_flat) {
        return("faiss_gpu_flat_cosine")
    }
    if (identical(metric, "correlation") && backend %in% gpu_flat) {
        return("faiss_gpu_flat_correlation")
    }
    exact_names <- c(
        "faiss_flat_ip", "faiss_gpu_flat_ip", "cuda_faiss_flat_ip",
        "faiss_flat_cosine", "faiss_flat_correlation",
        "faiss_gpu_flat_cosine", "faiss_gpu_flat_correlation"
    )
    if (backend %in% exact_names) return(backend)
    if (backend %in% c("cuda_auto", "gpu_auto")) return("cuda_auto")
    if (backend %in% grid_route_backends()) {
        if (identical(metric, "inner_product")) {
            stop("Grid nearest-neighbour search does not support ",
                "inner product.",
                call. = FALSE)
        }
        return(backend)
    }
    if (!backend %in% metric_routable_backends()) {
        stop(
            "The requested metric supports only CPU or a validated ",
            "metric-specific backend.", call. = FALSE
        )
    }
    backend
}

resolve_double_auto_backend <- function(
    backend, data, points, k, self_query, exclude_self, metric, tuning,
    target_recall, auto_selection
) {
    if (backend %in% c("cuvs_ivf", "cuda_cuvs_ivf")) {
        backend <- "cuda_cuvs_ivf_flat"
    }
    if (backend %in% c("cuda", "gpu") &&
            !isTRUE(cuda_available()) && !isTRUE(cuvs_available())) {
        stop("No CUDA GPU backend is available on this machine.", call. = FALSE)
    }
    if (backend %in% c("cpu_auto", "cuda_auto", "gpu_auto")) {
        route <- auto_selection %||% nn_auto_selection_for_backend(
            backend, self_query, nrow(data), ncol(data), nrow(points), k,
            prod(as.double(c(nrow(data), nrow(points), ncol(data)))),
            metric, exclude_self, tuning, target_recall
        )
        return(nn_auto_selected_backend(route, backend))
    }
    if (identical(backend, "cpu_approx")) {
        if (!isTRUE(self_query)) {
            stop("`backend = \"cpu_approx\"` requires self-KNN.", call. = FALSE)
        }
        return(select_cpu_approx_backend(nrow(data), ncol(data), k))
    }
    backend
}

double_float_output_cached <- function(
    data, points, k, metric, self_query, exclude_self, n_threads, output,
    params, target_recall
) {
    if (!metric %in% c("euclidean", "inner_product")) return(NULL)
    fitted_nn_index_result(
        data, points, k,
        if (identical(metric,
            "inner_product")) "faiss_flat_ip" else "faiss_flat_l2",
        switch(metric, inner_product = "faiss_flat_ip", "faiss_flat_l2"),
        self_query, exclude_self, metric, n_threads, output, params,
        target_recall = target_recall, use_cache = TRUE
    )
}

double_float_output_route <- function(
    data, points, k, backend, self_query, exclude_self, n_threads, metric,
    output, target_recall, requested_method
) {
    flat <- c(
        "faiss", "cpu_faiss", "cpu_faiss_flat", "faiss_flat",
        "faiss_flat_l2", "faiss_flat_ip", "faiss_flat_cosine",
        "faiss_flat_correlation"
    )
    if (!identical(output, "float") || !backend %in% flat) return(NULL)
    require_faiss_route("float-output FAISS Flat")
    label <- switch(
        metric, inner_product = "faiss_flat_ip", cosine = "faiss_flat_cosine",
        correlation = "faiss_flat_correlation", "faiss_flat_l2"
    )
    params <- cpu_flatlike_params(
        nrow(data), ncol(data), k, metric, target_recall, requested_method
    )
    cached <- double_float_output_cached(
        data, points, k, metric, self_query, exclude_self, n_threads,
        output, params, target_recall
    )
    if (!is.null(cached)) return(cached)
    out <- with_faiss_query_batch_size(params, {
        nn_faiss_flat_float32_cpp(
            data, points, as.integer(k), exclude_self,
            as.integer(n_threads), metric, output
        )
    })
    result <- finish_nn_result(out, label, k, self_query, TRUE, metric)
    attr(result, "faiss") <- list(
        index_type = as.character(out$index_type), library = "faiss",
        backend = "cpu", metric = metric, input_type = "float32"
    )
    result <- attach_cpu_exact_tuning(result, params, output, n_threads)
    finish_float32_direct_result(result, out)
}

double_cuda_exclude_self <- function(
    data, points, k, points_missing, n_threads, metric, tuning,
    target_recall, output, auto_selection
) {
    route <- if (isTRUE(cuvs_available())) {
        "cuda_cuvs_bruteforce"
    } else if (isTRUE(faiss_gpu_available())) {
        "faiss_gpu_flat_l2"
    } else {
        stop(
            "CUDA self-neighbor removal requires cuVS brute force or ",
            "FAISS GPU Flat.", call. = FALSE
        )
    }
    nn_compute(
        data, points, k, route, points_missing, TRUE, n_threads, metric,
        tuning, target_recall, output, auto_selection
    )
}

double_native_fallback <- function(
    data, points, k, backend, points_missing, exclude_self, n_threads,
    metric, tuning, target_recall, output, auto_selection, self_query
) {
    selected_gpu <- backend == "cuda" ||
        (backend == "gpu" && isTRUE(cuda_available()))
    if (isTRUE(selected_gpu)) {
        if (!isTRUE(cuda_available())) {
            stop("No CUDA GPU backend is available on this machine.",
                call. = FALSE)
        }
        if (isTRUE(exclude_self)) {
            return(double_cuda_exclude_self(
                data, points, k, points_missing, n_threads, metric, tuning,
                target_recall, output, auto_selection
            ))
        }
        if (k > 256L) {
            stop("Native GPU backends currently support `k <= 256`.",
                call. = FALSE)
        }
        out <- nn_cuda_cpp(data, points, as.integer(k), FALSE)
        return(finish_nn_result(out, "cuda", k, self_query, TRUE, metric))
    }
    if (backend == "gpu") {
        stop("No CUDA KNN backend is available on this machine.", call. = FALSE)
    }
    out <- nn_cpp(
        data, points, as.integer(k), metric, FALSE, FALSE, 0, TRUE,
        as.integer(n_threads), isTRUE(exclude_self)
    )
    finish_nn_result(out, "cpu", k, self_query, metric = metric)
}

nn_compute_double <- function(
    data, points, k, backend, points_missing, exclude_self, n_threads,
    metric, tuning, target_recall, output, auto_selection,
    requested_method, requested_backend
) {
    input <- prepare_double_nn_input(
        data, points, k, points_missing, exclude_self, n_threads, metric
    )
    data <- input$data
    points <- input$points
    k <- input$k
    self_query <- input$self_query
    n_threads <- input$n_threads
    metric <- input$metric
    backend <- normalize_double_metric_backend(metric, backend)
    backend <- resolve_double_auto_backend(
        backend, data, points, k, self_query, exclude_self, metric, tuning,
        target_recall, auto_selection
    )
    result <- double_float_output_route(
        data, points, k, backend, self_query, exclude_self, n_threads, metric,
        output, target_recall, requested_method
    )
    if (!is.null(result)) return(result)
    route_args <- list(
        data = data, points = points, k = k, backend = backend,
        exclude_self = exclude_self, n_threads = n_threads, metric = metric,
        tuning = tuning, target_recall = target_recall, output = output,
        auto_selection = auto_selection, requested_method = requested_method,
        requested_backend = requested_backend, self_query = self_query
    )
    result <- dispatch_double_routes(route_args)
    if (!is.null(result)) return(result)
    double_native_fallback(
        data, points, k, backend, points_missing, exclude_self, n_threads,
        metric, tuning, target_recall, output, auto_selection, self_query
    )
}


normalize_scalar_choice_arg <- function(
    x,
    arg,
    default,
    formal_choices = NULL
) {
    value <- trimws(as.character(x))
    value <- value[nzchar(value)]
    if (!length(value)) {
        return(default)
    }
    if (length(value) > 1L) {
        if (!is.null(formal_choices) && identical(value, formal_choices)) {
            return(default)
        }
        stop("`", arg, "` must be a single value.", call. = FALSE)
    }
    value[[1L]]
}

normalize_scalar_logical_arg <- function(x, arg, default = FALSE) {
    if (is.null(x) || !length(x)) {
        return(isTRUE(default))
    }
    if (length(x) != 1L || is.na(x)) {
        stop("`", arg, "` must be a single TRUE or FALSE value.", call. = FALSE)
    }
    if (!is.logical(x)) {
        stop("`", arg, "` must be a single TRUE or FALSE value.", call. = FALSE)
    }
    isTRUE(x)
}

normalize_public_backend_arg <- function(backend, arg = "backend") {
    resolve_faissr_environment_backend(backend)
}

normalize_nn_backend_arg <- function(backend, arg = "backend") {
    resolve_faissr_environment_backend(backend)
}

normalize_nn_method <- function(method) {
    if (
        length(method) == 1L && identical(trimws(as.character(method)), "scann")
    ) {
        stop(
            "`method = \"scann\"` is not a public faissR method. ",
            "Use `method = \"ivfpq_fastscan\"` for the ",
            "FAISS FastScan/cuVS 4-bit IVF-PQ route.",
            call. = FALSE
        )
    }
    method <- normalize_scalar_choice_arg(
        method,
        arg = "method",
        default = "auto",
        formal_choices = nn_method_request_labels()
    )
    if (is.na(method) || !nzchar(method)) {
        method <- "auto"
    }
    method <- trimws(method)
    style_aliases <- c(
        nsg_style = "nsg",
        vamana_style = "vamana",
        nndescent_style = "nndescent"
    )
    if (method %in% names(style_aliases)) {
        method <- unname(style_aliases[[method]])
    }
    labels <- nn_method_labels()
    if (!method %in% labels) {
        stop(
            "`method` must be one of \"auto\", \"exact\", ",
            "\"flat\", \"bruteforce\", ",
            "\"grid\", \"hnsw\", \"ivf\", \"ivfpq\", \"vamana_style\", ",
            "\"nsg_style\", \"nndescent_style\", ",
            "\"ivfpq_fastscan\", or \"cagra\". ",
            "The shorter \"vamana\", \"nsg\", and \"nndescent\" spellings are ",
            "retained as compatibility aliases.",
            " Use these canonical lowercase method labels; ",
            "internal backend route ",
            "labels such as \"faiss_hnsw\" are not public `method` values.",
            call. = FALSE
        )
    }
    method
}

validate_public_nn_method_shape <- function(data, method) {
    if (!identical(method, "grid")) {
        return(invisible(TRUE))
    }
    p <- faissr_quiet_warning(as.integer(ncol(data)))
    if (length(p) != 1L || is.na(p) || !p %in% c(2L, 3L)) {
        stop(
            "`method = \"grid\"` supports only two- or three-column matrices. ",
            "Use `method = \"auto\"` to let faissR select ",
            "a non-grid method for ",
            "higher-dimensional data.",
            call. = FALSE
        )
    }
    invisible(TRUE)
}

nn_metric_labels <- function() {
    c("euclidean", "cosine", "correlation")
}

nn_method_labels <- function() {
    c(
        "auto",
        "exact",
        "flat",
        "bruteforce",
        "grid",
        "hnsw",
        "ivf",
        "ivfpq",
        "vamana",
        "nsg",
        "nndescent",
        "ivfpq_fastscan",
        "cagra"
    )
}

nn_public_method_labels <- function() {
    c(
        "auto",
        "exact",
        "flat",
        "bruteforce",
        "grid",
        "hnsw",
        "ivf",
        "ivfpq",
        "vamana_style",
        "nsg_style",
        "nndescent_style",
        "ivfpq_fastscan",
        "cagra"
    )
}

nn_method_request_labels <- function() {
    unique(c(nn_public_method_labels(), nn_method_labels()))
}

faissr_option <- function(name, default = NULL) {
    name <- as.character(name)
    for (key in paste0("faissR.", name)) {
        value <- getOption(key, NULL)
        if (!is.null(value)) return(value)
    }
    default
}

normalize_cagra_implementation_value <- function(
    value,
    default = "auto",
    arg = NULL,
    strict = FALSE
) {
    if (is.null(value)) {
        value <- default
    }
    value <- tolower(gsub("[[:space:]_-]+", "", as.character(value)[1L]))
    if (length(value) != 1L || is.na(value) || !nzchar(value)) {
        value <- ""
    }
    aliases <- c(
        auto = "auto",
        default = "auto",
        faiss = "faiss_gpu",
        faissgpu = "faiss_gpu",
        gpu = "faiss_gpu",
        faisscuvs = "faiss_gpu",
        faissgpucagra = "faiss_gpu",
        cuvs = "cuvs",
        rapids = "cuvs",
        directcuvs = "cuvs",
        cudacuvs = "cuvs",
        cudacuvscagra = "cuvs"
    )
    if (!value %in% names(aliases)) {
        if (isTRUE(strict)) {
            arg <- arg %||% "cagra_implementation"
            stop(
                "`",
                arg,
                "` must be one of \"auto\", \"faiss_gpu\", or \"cuvs\".",
                call. = FALSE
            )
        }
        return(default)
    }
    unname(aliases[[value]])
}

cagra_implementation_preference <- function(default = "auto") {
    normalize_cagra_implementation_value(
        faissr_option("cagra_implementation", default),
        default = default
    )
}

normalize_cagra_implementation_arg <- function(value) {
    if (is.null(value)) {
        return(NULL)
    }
    normalize_cagra_implementation_value(
        value,
        default = "auto",
        arg = "cagra_implementation",
        strict = TRUE
    )
}

set_call_cagra_implementation <- function(value) {
    value <- normalize_cagra_implementation_arg(value)
    if (is.null(value)) {
        return(invisible(FALSE))
    }
    old <- getOption("faissR.cagra_implementation")
    options(faissR.cagra_implementation = value)
    parent <- parent.frame()
    do.call(
        on.exit,
        list(
            substitute(
                options(faissR.cagra_implementation = OLD),
                list(OLD = old)
            ),
            add = TRUE
        ),
        envir = parent
    )
    invisible(TRUE)
}

normalize_cagra_build_algo_value <- function(
    value,
    default = "auto",
    arg = NULL,
    strict = FALSE
) {
    if (is.null(value)) {
        value <- default
    }
    value <- tolower(gsub("[[:space:]-]+", "_", as.character(value)[1L]))
    value <- gsub("_+", "_", value)
    value <- gsub("^_|_$", "", value)
    if (length(value) != 1L || is.na(value) || !nzchar(value)) {
        value <- ""
    }
    aliases <- c(
        auto = "auto",
        default = "auto",
        auto_select = "auto",
        ivfpq = "ivf_pq",
        ivf_pq = "ivf_pq",
        nndescent = "nn_descent",
        nn_descent = "nn_descent",
        iterative = "iterative_cagra_search",
        iterative_cagra = "iterative_cagra_search",
        iterative_cagra_search = "iterative_cagra_search"
    )
    if (!value %in% names(aliases)) {
        if (isTRUE(strict)) {
            arg <- arg %||% "cagra_build_algo"
            stop(
                "`",
                arg,
                "` must be one of \"auto\", \"ivf_pq\", ",
                "\"nn_descent\", or \"iterative_cagra_search\".",
                call. = FALSE
            )
        }
        return(default)
    }
    unname(aliases[[value]])
}

cagra_build_algo_preference <- function(default = "auto") {
    normalize_cagra_build_algo_value(
        faissr_option("cuvs_cagra_build_algo", default),
        default = default
    )
}

cuvs_cagra_build_algo_for <- function(data, k, self_query, params = NULL) {
    cuvs_cagra_build_algo_for_shape(
        n = nrow(data),
        p = ncol(data),
        k = k,
        self_query = self_query,
        params = params
    )
}

cuvs_cagra_build_algo_for_shape <- function(
    n,
    p,
    k,
    self_query,
    params = NULL
) {
    requested <- cagra_build_algo_preference()
    if (identical(requested, "auto")) {
        table_algo <- as.character(params$cagra_build_algo %||% NA_character_)[[
            1L
        ]]
        if (
            !is.na(table_algo) &&
                nzchar(table_algo) &&
                !identical(table_algo, "auto")
        ) {
            return(table_algo)
        }
    }
    nn_tune_cuvs_cagra_build_algo_cpp(
        as.integer(n),
        faissr_quiet_warning(as.integer(p)),
        as.integer(k),
        isTRUE(self_query),
        isTRUE(params$tuning_compact_build %||% FALSE),
        requested
    )
}

normalize_cagra_build_algo_arg <- function(value) {
    if (is.null(value)) {
        return(NULL)
    }
    normalize_cagra_build_algo_value(
        value,
        default = "auto",
        arg = "cagra_build_algo",
        strict = TRUE
    )
}

set_call_cagra_build_algo <- function(value) {
    value <- normalize_cagra_build_algo_arg(value)
    if (is.null(value)) {
        return(invisible(FALSE))
    }
    old <- getOption("faissR.cuvs_cagra_build_algo")
    options(faissR.cuvs_cagra_build_algo = value)
    parent <- parent.frame()
    do.call(
        on.exit,
        list(
            substitute(
                options(faissR.cuvs_cagra_build_algo = OLD),
                list(OLD = old)
            ),
            add = TRUE
        ),
        envir = parent
    )
    invisible(TRUE)
}

cuda_cagra_auto_prefers_cuvs <- function(
    n = NULL,
    p = NULL,
    k = NULL,
    self_query = NULL
) {
    if (!isTRUE(self_query)) {
        return(FALSE)
    }
    if (is.null(n) || is.null(p) || is.null(k)) {
        return(FALSE)
    }
    n <- faissr_quiet_warning(as.numeric(n))
    p <- faissr_quiet_warning(as.numeric(p))
    k <- faissr_quiet_warning(as.numeric(k))
    if (length(n) != 1L || length(p) != 1L || length(k) != 1L) {
        return(FALSE)
    }
    if (!is.finite(n) || !is.finite(p) || !is.finite(k)) {
        return(FALSE)
    }
    compact_n <- faiss_option_int(
        "cuda_cagra_cuvs_compact_n",
        10000L,
        min_value = 100L,
        max_value = 1000000L
    )
    high_dim_p <- faiss_option_int(
        "cuda_cagra_cuvs_high_dim_p",
        1024L,
        min_value = 2L,
        max_value = 100000L
    )
    max_k <- faiss_option_int(
        "cuda_cagra_cuvs_compact_max_k",
        128L,
        min_value = 1L,
        max_value = 10000L
    )
    n <= compact_n && p >= high_dim_p && k <= max_k
}

resolve_cuda_cagra_backend <- function(
    faiss_gpu_available_value = faiss_gpu_available(),
    cuvs_available_value = cuvs_available(),
    n = NULL,
    p = NULL,
    k = NULL,
    self_query = NULL
) {
    preference <- cagra_implementation_preference()
    if (identical(preference, "faiss_gpu")) {
        return("faiss_gpu_cagra")
    }
    if (identical(preference, "cuvs")) {
        return("cuda_cuvs_cagra")
    }
    if (
        isTRUE(faiss_gpu_available_value) &&
            isTRUE(cuvs_available_value) &&
            isTRUE(cuda_cagra_auto_prefers_cuvs(
                n = n,
                p = p,
                k = k,
                self_query = self_query
            ))
    ) {
        return("cuda_cuvs_cagra")
    }
    if (isTRUE(faiss_gpu_available_value)) {
        "faiss_gpu_cagra"
    } else {
        "cuda_cuvs_cagra"
    }
}

cuda_cagra_route_available <- function(
    faiss_gpu_available_value = faiss_gpu_available(),
    cuvs_available_value = cuvs_available(),
    n = NULL,
    p = NULL,
    k = NULL,
    self_query = NULL
) {
    selected <- resolve_cuda_cagra_backend(
        faiss_gpu_available_value = faiss_gpu_available_value,
        cuvs_available_value = cuvs_available_value,
        n = n,
        p = p,
        k = k,
        self_query = self_query
    )
    if (identical(selected, "faiss_gpu_cagra")) {
        isTRUE(faiss_gpu_available_value)
    } else {
        isTRUE(cuvs_available_value)
    }
}

#' Nearest-neighbour method capabilities
#'
#' `nn_capabilities()` returns the public method/backend/metric support table
#' used by the nearest-neighbour API. It separates combinations that are
#' supported by design from combinations that should be treated as expected
#' skips in benchmarks.
#'
#' Public CUDA HNSW uses RAPIDS cuVS HNSW from
#' a CAGRA seed graph; metadata records that this is the cuVS wrapper design
#' rather than a pure all-GPU HNSW search implementation.
#' Public CUDA `method = "cagra"` can resolve to FAISS GPU CAGRA or direct cuVS
#' CAGRA; `options(faissR.cagra_implementation = "faiss_gpu")` or `"cuvs"`
#' forces one provider, while `"auto"` uses a deterministic shape rule: direct
#' cuVS CAGRA is selected for compact high-dimensional self-KNN, and FAISS GPU
#' CAGRA remains the default when both providers are available for other
#' shapes.
#' Availability preflights respect the forced provider for supported CAGRA
#' metrics, and returned approximate NN objects record `cagra_provider` plus
#' `cagra_provider_option`.
#'
#' @param runtime Logical; when `FALSE` (the default), report support by design
#'   without checking the current compiled/runtime libraries. When `TRUE`, add
#'   `resolved_backend`, `runtime_available`, `runtime_reason`, and
#'   `runtime_notes` columns for the current installation. `runtime_reason`
#'   uses stable labels such as `"available"`, `"unsupported_combination"`,
#'   `"missing_faiss"`, `"missing_faiss_gpu"`, `"missing_cuda"`,
#'   `"missing_cuda_route"`, and `"missing_cuvs"` for benchmark preflight
#'   tables.
#' @return A data frame with one row per public `method`, `backend` (`"cpu"`
#'   or `"cuda"`), and `metric` combination. Columns include
#'   `supported`, `exact`, `implementation`, `implementation_status`, and
#'   `notes`. Package-owned `*_style` routes are marked `"experimental"`;
#'   external-provider NN-descent is identified separately. If
#'   `runtime = TRUE`, runtime availability columns are appended.
#' @examples
#' caps <- nn_capabilities()
#' subset(caps, method == "flat" & supported)
#' @export
nn_capabilities <- function(runtime = FALSE) {
    runtime <- normalize_scalar_logical_arg(runtime, "runtime", default = FALSE)
    methods <- nn_public_method_labels()
    backends <- c("cpu", "cuda")
    metrics <- nn_metric_labels()
    rows <- vector("list", length(methods) * length(backends) * length(metrics))
    i <- 0L
    for (method in methods) {
        for (backend in backends) {
            for (metric in metrics) {
                i <- i + 1L
                rows[[i]] <- nn_capability_row(method, backend, metric)
            }
        }
    }
    out <- do.call(rbind.data.frame, rows)
    row.names(out) <- NULL
    if (isTRUE(runtime)) {
        runtime_rows <- lapply(seq_len(nrow(out)), function(i) {
            nn_capability_runtime_row(out[i, , drop = FALSE])
        })
        runtime_out <- do.call(rbind.data.frame, runtime_rows)
        out <- cbind(out, runtime_out, stringsAsFactors = FALSE)
    }
    out
}

nn_capability_runtime_row <- function(row) {
    if (!isTRUE(row$supported[[1L]])) {
        return(data.frame(
            resolved_backend = NA_character_,
            runtime_available = FALSE,
            runtime_reason = "unsupported_combination",
            runtime_notes = "Unsupported method/backend/metric combination.",
            stringsAsFactors = FALSE
        ))
    }
    resolved <- tryCatch(
        resolve_public_nn_backend(
            row$backend[[1L]],
            row$method[[1L]],
            row$metric[[1L]]
        ),
        error = identity
    )
    if (inherits(resolved, "error")) {
        return(data.frame(
            resolved_backend = NA_character_,
            runtime_available = FALSE,
            runtime_reason = "resolver_error",
            runtime_notes = conditionMessage(resolved),
            stringsAsFactors = FALSE
        ))
    }
    availability <- if (identical(resolved, "cuda_auto")) {
        nn_cuda_auto_runtime_available(row$metric[[1L]])
    } else {
        nn_resolved_backend_available(resolved)
    }
    data.frame(
        resolved_backend = resolved,
        runtime_available = isTRUE(availability$available),
        runtime_reason = availability$reason %||%
            if (isTRUE(availability$available)) {
                "available"
            } else {
                "unavailable_runtime"
            },
        runtime_notes = availability$notes,
        stringsAsFactors = FALSE
    )
}

nn_cuda_auto_runtime_available <- function(
    metric,
    cuda_available_value = cuda_available(),
    cuvs_available_value = cuvs_available(),
    faiss_gpu_available_value = faiss_gpu_available()
) {
    metric <- normalize_nn_metric(metric)
    if (identical(metric, "euclidean")) {
        return(nn_cuda_auto_euclidean_availability(
            cuda_available_value,
            cuvs_available_value,
            faiss_gpu_available_value
        ))
    }
    nn_cuda_auto_metric_availability(
        metric, cuda_available_value, cuvs_available_value,
        faiss_gpu_available_value
    )
}

nn_cuda_auto_euclidean_availability <- function(cuda, cuvs, faiss_gpu) {
    ok <- isTRUE(cuda) || isTRUE(cuvs) || isTRUE(faiss_gpu)
    list(
        available = ok,
        reason = if (ok) "available" else "missing_cuda_route",
        notes = if (ok) {
            paste0(
                "CUDA auto Euclidean route is available ",
                "through native CUDA, FAISS GPU, or cuVS."
            )
        } else {
            paste0(
                "CUDA auto Euclidean route requires native ",
                "CUDA, FAISS GPU, or cuVS support."
            )
        }
    )
}

nn_cuda_auto_metric_availability <- function(metric, cuda, cuvs, faiss_gpu) {
    ok <- isTRUE(faiss_gpu) || isTRUE(cuvs) || isTRUE(cuda)
    list(
        available = ok,
        reason = if (ok) "available" else "missing_cuda_route",
        notes = nn_cuda_auto_metric_notes(metric, cuda, cuvs, faiss_gpu)
    )
}

nn_cuda_auto_metric_notes <- function(metric, cuda, cuvs, faiss_gpu) {
    if (isTRUE(faiss_gpu)) {
            paste0(
                "CUDA auto non-Euclidean route is available ",
                "through FAISS GPU Flat metric-aware search."
            )
        } else if (
            isTRUE(cuvs) &&
                metric %in% c("cosine", "correlation")
        ) {
            paste(
                paste0(
                    "CUDA auto non-Euclidean route is ",
                    "shape-dependent on this runtime:"
                ),
                "large self-KNN graph searches can use cuVS graph routes, and",
                paste0(
                    "explicit exact/brute-force calls can use ",
                    "transformed cuVS brute force."
                )
            )
        } else if (
            isTRUE(cuda) &&
                metric %in% c("cosine", "correlation")
        ) {
            paste(
                paste0(
                    "CUDA auto non-Euclidean route is ",
                    "shape-dependent on this runtime:"
                ),
                paste0(
                    "native CUDA grid may apply to eligible 2D/3D ",
                    "self-search datasets,"
                ),
                paste0(
                    "while general exact non-Euclidean search ",
                    "still requires FAISS GPU Flat."
                )
            )
        } else {
            paste0(
                "CUDA auto non-Euclidean route requires FAISS ",
                "GPU Flat, cuVS graph support, or an eligible ",
                "native CUDA grid route."
            )
    }
}

nn_resolved_backend_available <- function(backend) {
    backend <- as.character(backend)[1L]
    if (is.na(backend) || !nzchar(backend)) {
        return(nn_runtime_status(
            FALSE,
            "missing_resolved_backend",
            "No resolved backend."
        ))
    }
    cpu_routes <- c(
        "auto",
        "cpu",
        "cpu_auto",
        "cpu_grid",
        "cpu_nndescent",
        "cpu_approx",
        "grid",
        "grid2d",
        "grid3d",
        "cpu_grid2d",
        "cpu_grid3d"
    )
    if (backend %in% cpu_routes) {
        return(nn_runtime_status(
            TRUE,
            "available",
            "Native CPU route is available."
        ))
    }
    special <- nn_special_backend_availability(backend)
    if (!is.null(special)) {
        return(special)
    }
    dependency <- nn_backend_dependency(backend)
    if (!is.null(dependency)) {
        return(nn_dependency_availability(
            dependency$check, dependency$reason, dependency$label
        ))
    }
    nn_runtime_status(
        TRUE,
        "available",
        "No additional runtime dependency detected."
    )
}

nn_backend_dependency <- function(backend) {
    if (startsWith(backend, "faiss_gpu")) {
        return(list(
            check = faiss_gpu_available, reason = "faiss_gpu",
                label = "FAISS GPU"
        ))
    }
    if (startsWith(backend, "cuda_cuvs") || startsWith(backend, "cuvs")) {
        return(list(check = cuvs_available, reason = "cuvs", label = "cuVS"))
    }
    if (startsWith(backend, "cuda")) {
        return(list(check = cuda_available, reason = "cuda",
            label = "Native CUDA"))
    }
    if (startsWith(backend, "faiss")) {
        return(list(check = faiss_available, reason = "faiss",
            label = "FAISS CPU"))
    }
    NULL
}

nn_runtime_status <- function(available, reason, notes) {
    list(available = isTRUE(available), reason = reason, notes = notes)
}

nn_dependency_availability <- function(check, reason_label, display_label) {
    available <- isTRUE(check())
    nn_runtime_status(
        available,
        if (available) "available" else paste0("missing_", reason_label),
        if (available) {
            paste(display_label, "route is available.")
        } else {
            paste(display_label, "support is not available in this build.")
        }
    )
}

nn_special_backend_availability <- function(backend) {
    if (backend %in% c("hnsw", "cpu_hnsw")) {
        return(nn_runtime_status(
            FALSE,
            "removed_legacy_hnsw_backend",
            paste(
                "The legacy direct HNSW backend label was removed; use",
                "backend = \"cpu\", method = \"hnsw\" for FAISS HNSW."
            )
        ))
    }
    if (backend %in% c("cuda_cuvs_hnsw", "cuvs_hnsw")) {
        return(nn_cuvs_hnsw_availability())
    }
    if (identical(backend, "faiss_ivfpq_fastscan")) {
        available <- isTRUE(faiss_fastscan_available())
        return(nn_runtime_status(
            available,
            if (available) "available" else "missing_faiss_fastscan",
            if (available) {
                "FAISS CPU FastScan route is available."
            } else {
                "FAISS FastScan support is not available in this build."
            }
        ))
    }
    NULL
}

nn_cuvs_hnsw_availability <- function() {
    available <- isTRUE(cuvs_available())
    notes <- if (available) {
        paste(
            "RAPIDS cuVS HNSW is available through the CAGRA-to-HNSW",
            "wrapper. This route uses a cuVS CPU hierarchy and is not a",
            "pure all-GPU HNSW search implementation."
        )
    } else {
        "RAPIDS cuVS support is not available in this build."
    }
    nn_runtime_status(
        available,
        if (available) "available_cuvs_hnsw_from_cagra" else "missing_cuvs",
        notes
    )
}

nn_capability_row <- function(method, backend, metric) {
    internal_method <- normalize_nn_method(method)
    capability <- nn_method_capability(internal_method, backend, metric)
    data.frame(
        method = method,
        backend = backend,
        metric = metric,
        supported = capability$supported,
        exact = capability$exact,
        implementation = capability$implementation,
        implementation_status = nn_capability_implementation_status(
            method,
            backend
        ),
        notes = capability$notes,
        stringsAsFactors = FALSE
    )
}

nn_capability_implementation_status <- function(method, backend) {
    method <- normalize_nn_method(method)
    backend <- normalize_nn_backend_arg(backend)
    if (method %in% c("nsg", "vamana")) {
        return("experimental")
    }
    if (identical(method, "nndescent")) {
        return(switch(
            backend,
            cpu = "experimental",
            cuda = "external_provider"
        ))
    }
    "supported"
}

normalize_nn_tuning <- function(tuning) {
    tuning <- normalize_scalar_choice_arg(
        tuning,
        arg = "tuning",
        default = "auto",
        formal_choices = c("auto", "cache", "pilot", "fixed", "off", "none")
    )
    if (is.na(tuning) || !nzchar(tuning)) {
        tuning <- "auto"
    }
    tuning <- tolower(gsub("[[:space:]_-]+", "", tuning))
    aliases <- c(
        auto = "auto",
        cache = "cache",
        cached = "cache",
        pilot = "pilot",
        fixed = "fixed",
        off = "off",
        none = "off",
        false = "off",
        no = "off"
    )
    if (!tuning %in% names(aliases)) {
        stop(
            "`tuning` must be one of \"auto\", \"cache\", \"pilot\", ",
            "\"fixed\", \"off\", or \"none\".",
            call. = FALSE
        )
    }
    unname(aliases[[tuning]])
}

normalize_hnsw_target_recall <- function(target_recall) {
    if (is.null(target_recall)) {
        target_recall <- faissr_option("hnsw_target_recall", 0.99)
    }
    value <- faissr_quiet_warning(as.numeric(target_recall))
    if (length(value) != 1L || is.na(value) || !is.finite(value)) {
        stop(
            "`target_recall` must be one of 0.9, 0.95, or 0.99.",
            call. = FALSE
        )
    }
    allowed <- c(0.90, 0.95, 0.99)
    match <- which(abs(value - allowed) < 1e-8)
    if (!length(match)) {
        stop(
            "`target_recall` must be one of 0.9, 0.95, or 0.99.",
            call. = FALSE
        )
    }
    allowed[[match[[1L]]]]
}

stop_cuda_hnsw_unavailable <- function() {
    stop(
        "CUDA `method = \"hnsw\"` requires RAPIDS cuVS HNSW support. ",
        "Install faissR with RAPIDS cuVS headers and libraries visible ",
        "so `cuda_cuvs_hnsw` can build a CAGRA seed graph and convert it ",
        "with `cuvsHnswFromCagraWithDataset`.",
        call. = FALSE
    )
}

resolve_public_nn_backend <- function(
    backend,
    method,
    metric = "euclidean",
    n = NULL,
    p = NULL,
    k = NULL,
    self_query = NULL
) {
    device <- normalize_nn_backend_arg(backend)
    method <- normalize_nn_method(method)
    metric <- normalize_nn_metric(metric)
    if (identical(method, "auto")) {
        return(resolve_public_auto_method_backend(device))
    }
    if (identical(device, "cpu")) {
        return(resolve_cpu_nn_backend(method, metric))
    }
    resolve_cuda_nn_backend(method, metric, n, p, k, self_query)
}

public_nn_method_label <- function(method) {
    if (missing(method) || is.null(method) || length(method) < 1L) {
        return(NA_character_)
    }
    method <- as.character(method)[1L]
    if (is.na(method) || !nzchar(method)) {
        return(NA_character_)
    }
    labels <- c(
        auto = "auto",
        exact = "exact",
        flat = "flat",
        bruteforce = "bruteforce",
        grid = "grid",
        hnsw = "hnsw",
        ivf = "ivf",
        ivfpq = "ivfpq",
        vamana = "vamana",
        nsg = "nsg",
        nndescent = "nndescent",
        ivfpq_fastscan = "ivfpq_fastscan",
        cagra = "cagra"
    )
    labels[[method]] %||% method
}

.nn_backend_method_groups <- list(
    auto = c("auto", "cpu_auto", "cuda_auto", "gpu_auto"),
    exact = c("cpu", "cuda"),
    bruteforce = "cuda_cuvs_bruteforce",
    flat = c(
        "faiss", "cpu_faiss", "cpu_faiss_flat", "faiss_flat",
        "faiss_flat_l2", "faiss_flat_ip", "faiss_flat_cosine",
        "faiss_flat_correlation", "faiss_gpu_flat", "faiss_gpu_flat_l2",
        "cuda_faiss_flat_l2", "faiss_gpu_flat_ip", "cuda_faiss_flat_ip",
        "faiss_gpu_flat_cosine", "cuda_faiss_flat_cosine",
        "faiss_gpu_flat_correlation", "cuda_faiss_flat_correlation"
    ),
    grid = c(
        "grid", "cpu_grid", "grid2d", "grid3d", "cpu_grid2d",
        "cpu_grid3d", "cuda_grid", "cuda_grid_auto", "gpu_grid",
        "cuda_grid2d", "cuda_grid3d"
    ),
    hnsw = c("faiss_hnsw", "cuda_cuvs_hnsw", "cuvs_hnsw"),
    ivf = c(
        "faiss_ivf", "cpu_faiss_index_ivf", "faiss_ivf_flat",
        "faiss_gpu_ivf", "faiss_gpu_ivf_flat", "cuda_faiss_ivf_flat",
        "cuvs_ivf", "cuda_cuvs_ivf", "cuvs_ivf_flat", "cuda_cuvs_ivf_flat"
    ),
    ivfpq = c(
        "faiss_ivfpq", "faiss_gpu_ivfpq", "cuda_faiss_ivfpq",
        "cuvs_ivfpq", "cuda_cuvs_ivfpq", "cuvs_ivf_pq",
        "cuda_cuvs_ivf_pq"
    ),
    ivfpq_fastscan = c(
        "faiss_ivfpq_fastscan", "cuda_cuvs_ivfpq_fastscan",
        "cuvs_ivfpq_fastscan"
    ),
    vamana = c("cpu_vamana", "cuda_vamana"),
    nsg = c("faiss_nsg", "cpu_nsg", "cuda_nsg"),
    nndescent = c(
        "cpu_nndescent", "cuda_nndescent", "faiss_nndescent",
        "cuda_cuvs_nndescent", "cuvs_nndescent"
    ),
    cagra = c(
        "faiss_gpu_cagra", "cuda_faiss_cagra", "cuda_cuvs_cagra",
        "cuda_cagra", "gpu_cagra"
    )
)

nn_resolved_backend_public_method <- function(backend) {
    backend <- as.character(backend)[1L]
    if (is.na(backend) || !nzchar(backend)) {
        return(NA_character_)
    }
    match <- vapply(
        .nn_backend_method_groups,
        function(routes) backend %in% routes,
        logical(1L)
    )
    if (any(match)) names(match)[which(match)[1L]] else NA_character_
}

nn_resolved_backend_device <- function(backend) {
    backend <- as.character(backend)[1L]
    if (is.na(backend) || !nzchar(backend)) {
        return(NA_character_)
    }
    if (identical(backend, "cpu_auto")) {
        return("cpu")
    }
    if (backend %in% c("cuda_auto", "gpu_auto")) {
        return("cuda")
    }
    if (
        startsWith(backend, "cuda") ||
            startsWith(backend, "gpu") ||
            startsWith(backend, "cuvs") ||
            startsWith(backend, "faiss_gpu")
    ) {
        return("cuda")
    }
    "cpu"
}

select_cuda_auto_backend <- function(
    self_query,
    n,
    p,
    n_points,
    k,
    work_size,
    metric = "euclidean"
) {
    route <- nn_auto_select_shape_cpp(
        resolved_backend = "cuda_auto",
        requested_backend = "cuda",
        requested_method = "auto",
        shape = list(
            n = as.integer(n),
            p = as.integer(p),
            n_points = as.integer(n_points),
            k = as.integer(k),
            metric = normalize_nn_metric(metric),
            self_query = isTRUE(self_query),
            exclude_self = FALSE,
            work_size = as.double(work_size)
        )
    )
    if (!is.na(route$error)) {
        stop(route$error, call. = FALSE)
    }
    route$selected_backend
}

select_cuvs_auto_backend <- function(
    self_query,
    n,
    p,
    n_points,
    k,
    work_size,
    cuda_available_value = cuda_available(),
    cuvs_available_value = cuvs_available()
) {
    route <- nn_auto_select_shape_cpp(
        resolved_backend = "cuda_auto",
        requested_backend = "cuda",
        requested_method = "auto",
        shape = list(
            n = as.integer(n),
            p = as.integer(p),
            n_points = as.integer(n_points),
            k = as.integer(k),
            metric = "euclidean",
            self_query = isTRUE(self_query),
            exclude_self = FALSE,
            work_size = as.double(work_size)
        ),
        cuda_available_value = cuda_available_value,
        cuvs_available_value = cuvs_available_value,
        faiss_gpu_available_value = FALSE
    )
    if (!is.na(route$error)) {
        stop(route$error, call. = FALSE)
    }
    route$selected_backend
}

native_nsg_option_int <- function(
    name,
    default,
    min_value = 1L,
    max_value = .Machine$integer.max
) {
    value <- faissr_option(name, NULL)
    value <- if (is.null(value)) {
        default
    } else {
        faissr_quiet_warning(as.integer(value))
    }
    if (length(value) != 1L || is.na(value) || !is.finite(value)) {
        value <- default
    }
    as.integer(max(min_value, min(max_value, value)))
}

select_cpu_auto_backend <- function(
    self_query,
    n,
    p,
    n_points,
    k,
    work_size,
    metric = "euclidean"
) {
    route <- nn_auto_select_shape_cpp(
        resolved_backend = "cpu_auto",
        requested_backend = "cpu",
        requested_method = "auto",
        shape = list(
            n = as.integer(n),
            p = as.integer(p),
            n_points = as.integer(n_points),
            k = as.integer(k),
            metric = normalize_nn_metric(metric),
            self_query = isTRUE(self_query),
            exclude_self = FALSE,
            work_size = as.double(work_size)
        )
    )
    route$selected_backend
}

cpu_auto_faiss_flat_work_threshold <- function() {
    value <- faissr_option("cpu_auto_faiss_flat_work", 5e7)
    value <- faissr_quiet_warning(as.numeric(value))
    if (length(value) != 1L || is.na(value) || !is.finite(value) || value < 0) {
        value <- 5e7
    }
    value
}

cpu_auto_metric_faiss_flat_backend <- function(metric) {
    metric <- normalize_nn_metric(metric)
    switch(
        metric,
        cosine = "faiss_flat_cosine",
        correlation = "faiss_flat_correlation",
        inner_product = "faiss_flat_ip",
        NA_character_
    )
}

public_nn_cpu_route_supported <- function(method, metric) {
    method <- normalize_nn_method(method)
    metric <- normalize_nn_metric(metric)
    all_metrics <- metric %in% nn_metric_labels()
    euclidean <- identical(metric, "euclidean")
    non_ip_metric <- metric %in% c("euclidean", "cosine", "correlation")
    switch(
        method,
        auto = TRUE,
        exact = all_metrics,
        bruteforce = all_metrics,
        flat = all_metrics,
        grid = non_ip_metric,
        hnsw = all_metrics,
        ivf = all_metrics,
        ivfpq = all_metrics,
        vamana = all_metrics,
        nsg = all_metrics,
        nndescent = all_metrics,
        ivfpq_fastscan = metric %in%
            c("euclidean", "cosine", "correlation", "inner_product"),
        cagra = FALSE,
        FALSE
    )
}

public_nn_cuda_route_available <- function(
    method, metric,
    cuda_available_value = cuda_available(),
    cuvs_available_value = cuvs_available(),
    faiss_gpu_available_value = faiss_gpu_available()
) {
    method <- normalize_nn_method(method)
    metric <- normalize_nn_metric(metric)
    if (identical(method, "auto")) {
        return(isTRUE(
            nn_cuda_auto_runtime_available(
                metric,
                cuda_available_value = cuda_available_value,
                cuvs_available_value = cuvs_available_value,
                faiss_gpu_available_value = faiss_gpu_available_value
            )$available
        ))
    }
    if (identical(method, "nsg")) {
        return(isTRUE(cuda_available_value))
    }
    if (identical(method, "vamana")) {
        return(isTRUE(cuda_available_value))
    }
    if (identical(method, "ivfpq_fastscan")) {
        return(
            metric %in%
                c("euclidean", "cosine", "correlation", "inner_product") &&
                isTRUE(cuvs_available_value)
        )
    }
    if (identical(metric, "inner_product")) {
        return(public_nn_cuda_inner_product_available(
            method, cuvs_available_value, faiss_gpu_available_value
        ))
    }
    if (identical(method, "hnsw")) {
        return(isTRUE(cuvs_available_value))
    }
    if (metric %in% c("cosine", "correlation")) {
        return(public_nn_cuda_normalized_available(
            method, cuda_available_value, cuvs_available_value,
            faiss_gpu_available_value
        ))
    }
    public_nn_cuda_euclidean_available(
        method, cuda_available_value, cuvs_available_value,
        faiss_gpu_available_value
    )
}

public_nn_cuda_euclidean_available <- function(method, cuda, cuvs, faiss_gpu) {
    switch(
        method,
        exact = isTRUE(faiss_gpu) || isTRUE(cuvs) || isTRUE(cuda),
        bruteforce = isTRUE(faiss_gpu) || isTRUE(cuvs) || isTRUE(cuda),
        flat = isTRUE(faiss_gpu),
        grid = isTRUE(cuda),
        hnsw = isTRUE(cuvs),
        ivf = isTRUE(faiss_gpu),
        ivfpq = isTRUE(faiss_gpu),
        ivfpq_fastscan = isTRUE(cuvs),
        nndescent = isTRUE(cuvs),
        cagra = cuda_cagra_route_available(
            faiss_gpu_available_value = faiss_gpu,
            cuvs_available_value = cuvs
        ),
        FALSE
    )
}

public_nn_cuda_inner_product_available <- function(method, cuvs, faiss_gpu) {
    if (identical(method, "nndescent")) return(FALSE)
    if (identical(method, "cagra")) {
        return(cuda_cagra_route_available(faiss_gpu, cuvs))
    }
    if (identical(method, "hnsw")) return(isTRUE(cuvs))
    if (method %in% c("exact", "bruteforce")) {
        return(isTRUE(faiss_gpu) || isTRUE(cuvs))
    }
    method %in% c("flat", "ivf", "ivfpq", "vamana") && isTRUE(faiss_gpu)
}

public_nn_cuda_normalized_available <- function(method, cuda, cuvs, faiss_gpu) {
    if (method %in% c("exact", "bruteforce")) {
        return(isTRUE(faiss_gpu) || isTRUE(cuvs))
    }
    if (method %in% c("flat", "ivf", "ivfpq")) return(isTRUE(faiss_gpu))
    if (identical(method, "grid")) return(isTRUE(cuda))
    if (method %in% c("hnsw", "nndescent")) return(isTRUE(cuvs))
    if (identical(method, "cagra")) {
        return(cuda_cagra_route_available(faiss_gpu, cuvs))
    }
    FALSE
}

select_cpu_approx_backend <- function(n, p, k) {
    if (
        should_use_grid2d_self_knn(
            self_query = TRUE,
            n = n,
            p = p,
            k = k,
            exclude_self = FALSE,
            metric = "euclidean"
        )
    ) {
        return("cpu_grid")
    }
    select_self_approx_backend(prefer_cuda = FALSE)
}

select_self_approx_backend <- function(prefer_cuda = FALSE) {
    if (isTRUE(prefer_cuda) && isTRUE(cuvs_available())) {
        return("cuda_cagra")
    }
    if (isTRUE(faiss_available())) {
        return("faiss_hnsw")
    }
    "cpu"
}

nn_auto_shape <- function(
    data,
    points,
    points_missing,
    k,
    metric = "euclidean",
    exclude_self = FALSE
) {
    data_dim <- nn_auto_object_dims(data, "data")
    points_dim <- if (isTRUE(points_missing)) {
        data_dim
    } else {
        nn_auto_object_dims(points, "points")
    }
    n <- as.integer(data_dim[[1L]])
    p <- as.integer(data_dim[[2L]])
    n_points <- as.integer(points_dim[[1L]])
    self_query <- isTRUE(points_missing) || identical(data, points)
    if (is.null(k)) {
        k <- nn_auto_default_k(n, self_query, exclude_self)
    }
    k <- normalize_nn_positive_integer(
        k,
        "k",
        "`k` must be NULL or a positive integer."
    )
    list(
        n = n,
        p = p,
        n_points = n_points,
        k = as.integer(k),
        metric = normalize_nn_metric(metric),
        self_query = isTRUE(self_query),
        exclude_self = isTRUE(exclude_self),
        work_size = as.double(n) * as.double(n_points) * as.double(p)
    )
}

nn_auto_object_dims <- function(x, name) {
    dims <- if (is_float32_matrix_input(x)) {
        float32_matrix_dims(x, name)
    } else {
        dim(x)
    }
    if (is.null(dims) || length(dims) != 2L) dim(as.matrix(x)) else dims
}

nn_auto_default_k <- function(n, self_query, exclude_self) {
    if (n == 1L) return(1L)
    min(
        n,
        auto_k(n, include_self = isTRUE(self_query) && !isTRUE(exclude_self))
    )
}

nn_auto_select_shape_cpp <- function(
    resolved_backend,
    requested_backend,
    requested_method = "auto",
    shape,
    tuning = "auto",
    target_recall = 0.99,
    cuda_available_value = cuda_available(),
    cuvs_available_value = cuvs_available(),
    faiss_available_value = faiss_available(),
    faiss_gpu_available_value = faiss_gpu_available()
) {
    args <- list(
        resolved_backend = resolved_backend,
        requested_backend = requested_backend,
        requested_method = public_nn_method_label(requested_method),
        metric = normalize_nn_metric(shape$metric),
        n = as.integer(shape$n),
        p = as.integer(shape$p),
        n_points = as.integer(shape$n_points),
        k = as.integer(shape$k),
        self_query = isTRUE(shape$self_query),
        exclude_self = isTRUE(shape$exclude_self),
        cuda_available = isTRUE(cuda_available_value),
        cuvs_available = isTRUE(cuvs_available_value),
        faiss_available = isTRUE(faiss_available_value),
        faiss_gpu_available = isTRUE(faiss_gpu_available_value),
        cagra_preference = cagra_implementation_preference()
    )
    controls <- nn_auto_selector_controls(target_recall, tuning)
    out <- do.call(nn_auto_select_backend_cpp, c(args, controls))
    out <- normalize_nn_auto_selection_fields(out)
    nn_auto_hardware_metadata(out)
}

nn_auto_selector_numeric_option <- function(name, default) {
    value <- faissr_quiet_warning(as.numeric(faissr_option(name, default)))
    if (length(value) != 1L || is.na(value) || !is.finite(
        value)) default else value
}

nn_auto_selector_controls <- function(target_recall, tuning) {
    c(nn_auto_selector_cuda_controls(), list(
        cuvs_bruteforce_work_threshold = nn_auto_selector_numeric_option(
            "cuvs_bruteforce_work_threshold", 5e12
        ),
        cpu_exact_work = nn_auto_selector_numeric_option(
            "cpu_auto_exact_work", 2e8
        ),
        cpu_faiss_flat_work = cpu_auto_faiss_flat_work_threshold(),
        target_recall_option = as.numeric(
            normalize_hnsw_target_recall(target_recall)
        ),
        tuning = normalize_nn_tuning(tuning)
    ))
}

nn_auto_selector_cuda_controls <- function() {
    list(
        cuda_exact_n = faiss_option_int(
            "cuda_auto_exact_n",
            100000L,
            min_value = 1000L,
            max_value = 10000000L
        ),
        cuda_exact_work = nn_auto_selector_numeric_option(
            "cuda_auto_exact_work", 5e12
        ),
        metric_graph_n = faiss_option_int(
            "cuda_auto_metric_graph_n",
            100000L,
            min_value = 1000L,
            max_value = 10000000L
        ),
        metric_graph_min_k = faiss_option_int(
            "cuda_auto_metric_graph_min_k",
            10L,
            min_value = 2L,
            max_value = 256L
        ),
        metric_graph_work = nn_auto_selector_numeric_option(
            "cuda_auto_metric_graph_work", 5e12
        ),
        cagra_compact_n = faiss_option_int(
            "cuda_cagra_cuvs_compact_n",
            10000L,
            min_value = 100L,
            max_value = 1000000L
        ),
        cagra_high_dim_p = faiss_option_int(
            "cuda_cagra_cuvs_high_dim_p",
            1024L,
            min_value = 2L,
            max_value = 100000L
        ),
        cagra_compact_max_k = faiss_option_int(
            "cuda_cagra_cuvs_compact_max_k",
            128L,
            min_value = 1L,
            max_value = 10000L
        )
    )
}

normalize_nn_auto_selection_fields <- function(out) {
    for (field in c(
        "selected_backend",
        "predicted_backend",
        "predicted_method",
        "predicted_device",
        "reason",
        "error"
    )) {
        if (is.null(out[[field]]) || !nzchar(as.character(out[[field]])[1L])) {
            out[[field]] <- NA_character_
        }
    }
    out
}

nn_runtime_cpu_model <- function() {
    override <- faissr_option("runtime_cpu_model", NA_character_)
    override <- as.character(override)[1L]
    if (!is.na(override) && nzchar(override)) {
        return(override)
    }
    if (
        exists(
            "cpu_model",
            envir = .faissR_auto_hardware_cache,
            inherits = FALSE
        )
    ) {
        return(.faissR_auto_hardware_cache$cpu_model)
    }
    value <- gpuinfo_cpu_model()
    if (length(value) != 1L || is.na(value) || !nzchar(value)) {
        value <- NA_character_
    }
    .faissR_auto_hardware_cache$cpu_model <- value
    value
}

nn_runtime_gpu_model <- function() {
    override <- faissr_option("runtime_gpu_model", NA_character_)
    override <- as.character(override)[1L]
    if (!is.na(override) && nzchar(override)) {
        return(override)
    }
    if (
        exists(
            "gpu_model",
            envir = .faissR_auto_hardware_cache,
            inherits = FALSE
        )
    ) {
        return(.faissR_auto_hardware_cache$gpu_model)
    }
    value <- gpuinfo_cuda_device()
    value <- as.character(value)[1L]
    if (is.na(value) || !nzchar(value)) {
        value <- NA_character_
    }
    .faissR_auto_hardware_cache$gpu_model <- value
    value
}

nn_normalize_hardware_name <- function(x) {
    x <- tolower(as.character(x)[1L])
    if (is.na(x) || !nzchar(x)) {
        return(NA_character_)
    }
    gsub("[^a-z0-9]+", "", x)
}

nn_auto_hardware_metadata <- function(route) {
    device <- as.character(route$predicted_device %||% NA_character_)[1L]
    models <- nn_auto_hardware_models(route, device)
    calibration_model <- models$calibration
    runtime_model <- models$runtime
    calibration_name <- nn_normalize_hardware_name(calibration_model)
    runtime_name <- nn_normalize_hardware_name(runtime_model)
    match_status <- if (is.na(calibration_name) || is.na(runtime_name)) {
        "unknown"
    } else if (identical(calibration_name, runtime_name)) {
        "matched"
    } else {
        "mismatch"
    }
    route$runtime_hardware_device <- device
    route$runtime_hardware_model <- runtime_model
    cores <- parallel::detectCores(logical = TRUE)
    route$runtime_logical_cores <- faissr_quiet_warning(
        as.integer(cores)
    )
    route$hardware_match_status <- match_status
    route$hardware_evidence <- if (identical(match_status, "matched")) {
        "calibration_hardware_matched"
    } else if (identical(match_status, "unknown")) {
        "hardware_unidentified"
    } else {
        "hardware_extrapolated_unvalidated"
    }
    route$hardware_policy_action <- paste0(
        "static_policy_retained_",
        "no_hardware_fallback"
    )
    route$hardware_conservative_fallback <- FALSE
    route$hardware_evidence_note <- nn_auto_hardware_evidence_note(match_status)
    route
}

nn_auto_hardware_models <- function(route, device) {
    if (identical(device, "cuda")) {
        return(list(
            calibration = route$calibration_gpu_model %||% NA_character_,
            runtime = nn_runtime_gpu_model()
        ))
    }
    list(
        calibration = route$calibration_cpu_model %||% NA_character_,
        runtime = nn_runtime_cpu_model()
    )
}

nn_auto_hardware_evidence_note <- function(match_status) {
    if (identical(match_status, "matched")) {
        return(paste0(
            "Runtime accelerator model matches the frozen ",
            "calibration profile; provider versions and ",
            "data geometry remain relevant."
        ))
    }
    if (identical(match_status, "unknown")) {
        return(paste0(
            "Runtime hardware identity could not be confirmed. ",
            "Method selection is unchanged; hardware-specific ",
            "target and timing evidence is unavailable."
        ))
    }
    paste0(
        "The compiled policy is being applied outside ",
        "a confirmed hardware match. Method selection ",
        "is unchanged; target and timing evidence are ",
        "calibration-informed, not validated on this ",
        "machine."
    )
}

nn_auto_selection_for_backend <- function(
    backend,
    self_query,
    n,
    p,
    n_points,
    k,
    work_size,
    metric = "euclidean",
    exclude_self = FALSE,
    tuning = "auto",
    target_recall = 0.99
) {
    requested_backend <- switch(
        backend,
        cpu_auto = "cpu",
        cuda_auto = "cuda",
        gpu_auto = "cuda",
        nn_resolved_backend_device(backend)
    )
    nn_auto_select_shape_cpp(
        resolved_backend = backend,
        requested_backend = requested_backend,
        requested_method = "auto",
        shape = list(
            n = as.integer(n),
            p = as.integer(p),
            n_points = as.integer(n_points),
            k = as.integer(k),
            metric = normalize_nn_metric(metric),
            self_query = isTRUE(self_query),
            exclude_self = isTRUE(exclude_self),
            work_size = as.double(work_size)
        ),
        tuning = tuning,
        target_recall = target_recall
    )
}

nn_auto_selected_backend <- function(route, fallback_backend) {
    if (is.null(route)) {
        return(fallback_backend)
    }
    error <- route$error %||% NA_character_
    error <- as.character(error)[1L]
    if (!is.na(error) && nzchar(error)) {
        stop(error, call. = FALSE)
    }
    selected <- route$selected_backend %||%
        route$predicted_backend %||%
        fallback_backend
    selected <- as.character(selected)[1L]
    if (is.na(selected) || !nzchar(selected)) fallback_backend else selected
}

nn_auto_selection_metadata <- function(
    data,
    points,
    points_missing,
    k,
    requested_backend,
    requested_method,
    resolved_backend,
    metric = "euclidean",
    tuning = "auto",
    exclude_self = FALSE,
    target_recall = 0.99
) {
    explicit_backend <- TRUE
    explicit_method <- !identical(requested_method, "auto")
    if (
        explicit_backend &&
            explicit_method &&
            !resolved_backend %in% c("cpu_auto", "cuda_auto", "gpu_auto")
    ) {
        return(NULL)
    }
    shape <- nn_auto_shape(
        data = data,
        points = points,
        points_missing = points_missing,
        k = k,
        metric = metric,
        exclude_self = exclude_self
    )
    nn_auto_select_shape_cpp(
        resolved_backend = resolved_backend,
        requested_backend = requested_backend,
        requested_method = requested_method,
        shape = shape,
        tuning = tuning,
        target_recall = target_recall
    )
}

normalize_nn_threads <- function(n_threads) {
    if (is.null(n_threads)) {
        n_threads <- faissr_quiet_warning(parallel::detectCores(
            logical = FALSE
        ))
        n_threads <- faissr_quiet_warning(as.integer(n_threads))
        if (
            length(n_threads) != 1L ||
                is.na(n_threads) ||
                !is.finite(n_threads) ||
                n_threads < 1L
        ) {
            n_threads <- 1L
        }
        return(as.integer(max(1L, min(64L, n_threads))))
    }
    n_threads <- normalize_nn_positive_integer(
        n_threads,
        "n_threads",
        "`n_threads` must be NULL or a single positive integer."
    )
    as.integer(max(1L, min(64L, n_threads)))
}

normalize_nn_positive_integer <- function(x, arg, message) {
    value <- faissr_quiet_warning(as.numeric(x))
    if (
        length(value) != 1L ||
            is.na(value) ||
            !is.finite(value) ||
            value < 1L ||
            abs(value - round(value)) > sqrt(.Machine$double.eps)
    ) {
        stop(message, call. = FALSE)
    }
    as.integer(round(value))
}

normalize_nn_metric <- function(metric) {
    valid <- nn_metric_labels()
    metric <- normalize_scalar_choice_arg(
        metric,
        arg = "metric",
        default = "euclidean",
        formal_choices = valid
    )
    key <- tolower(trimws(metric))
    if (!key %in% valid) {
        stop(
            "`metric` must be one of \"euclidean\", \"cosine\", or ",
            "\"correlation\".",
            call. = FALSE
        )
    }
    key
}

faiss_metric_search_arg <- function(metric) {
    metric <- normalize_nn_metric(metric)
    if (identical(metric, "inner_product")) "inner_product" else "euclidean"
}

faiss_metric_distance_output_arg <- function(metric) {
    metric <- normalize_nn_metric(metric)
    if (identical(metric, "inner_product")) "inner_product" else "euclidean"
}

should_use_grid2d_self_knn <- function(
    self_query,
    n,
    p,
    k,
    exclude_self,
    metric
) {
    if (!isTRUE(self_query)) {
        return(FALSE)
    }
    metric <- normalize_nn_metric(metric)
    if (!metric %in% c("euclidean", "cosine", "correlation")) {
        return(FALSE)
    }
    if (!as.integer(p) %in% c(2L, 3L)) {
        return(FALSE)
    }
    if (as.integer(n) < 10000L) {
        return(FALSE)
    }
    nonself_k <- if (isTRUE(exclude_self)) as.integer(k) else as.integer(k) - 1L
    is.finite(nonself_k) && !is.na(nonself_k) && nonself_k >= 1L
}

grid_bins_per_dim <- function(n, k, p) {
    p <- as.integer(p)
    configured <- configured_grid_bins(p)
    if (!is.na(configured)) return(configured)
    target_occupancy <- grid_target_occupancy(p, k)
    bins <- if (identical(p, 3L)) {
        as.integer(ceiling((as.numeric(n) / target_occupancy)^(1 / 3)))
    } else {
        as.integer(ceiling(sqrt(as.numeric(n) / target_occupancy)))
    }
    as.integer(max(4L, min(4096L, bins)))
}

configured_grid_bins <- function(p) {
    value <- faissr_option(sprintf("grid%dd_bins_per_dim", p), NULL) %||%
        faissr_option("grid_bins_per_dim", NULL)
    if (is.null(value)) return(NA_integer_)
    value <- faissr_quiet_warning(as.integer(value))
    if (length(value) == 1L && is.finite(value) && !is.na(value) &&
        value > 0L) {
        as.integer(value)
    } else {
        NA_integer_
    }
}

grid_target_occupancy <- function(p, k) {
    target_occupancy <- faissr_option(
        sprintf("grid%dd_target_occupancy", p),
        NULL
    )
    if (is.null(target_occupancy)) {
        target_occupancy <- faissr_option("grid_target_occupancy", NULL)
    }
    if (is.null(target_occupancy)) {
        target_occupancy <- if (identical(p, 3L)) {
            max(1.5, min(8, as.numeric(k) / 25))
        } else {
            max(4, min(16, as.numeric(k) / 10))
        }
    }
    target_occupancy <- faissr_quiet_warning(as.numeric(target_occupancy))
    if (
        length(target_occupancy) != 1L ||
            !is.finite(target_occupancy) ||
            is.na(target_occupancy) ||
            target_occupancy <= 0
    ) {
        target_occupancy <- if (identical(p, 3L)) {
            max(1.5, min(8, as.numeric(k) / 25))
        } else {
            max(4, min(16, as.numeric(k) / 10))
        }
    }
    target_occupancy
}

grid2d_bins_per_dim <- function(n, k) grid_bins_per_dim(n, k, 2L)
grid3d_bins_per_dim <- function(n, k) grid_bins_per_dim(n, k, 3L)

select_cpu_spatial_backend <- function(data, k, exclude_self = TRUE) {
    p <- ncol(data)
    if (!(p %in% c(2L, 3L))) {
        stop(
            "`method = \"grid\"` supports only two- or three-column matrices.",
            call. = FALSE
        )
    }
    out <- if (p == 3L) "cpu_grid3d" else "cpu_grid2d"
    if (!isTRUE(faissr_option("cpu_spatial_auto", TRUE))) return(out)
    n <- nrow(data)
    sample_n <- min(n, as.integer(faissr_option("cpu_spatial_sample", 4096L)))
    if (sample_n < 512L) return(out)
    rows <- unique(as.integer(round(seq.int(1L, n, length.out = sample_n))))
    xs <- data[rows, , drop = FALSE]
    unique_sample <- nrow(unique(round(xs, digits = 12L)))
    duplicate_ratio <- unique_sample / sample_n
    duplicate_threshold <- as.numeric(faissr_option(
        "cpu_spatial_duplicate_threshold",
        0.05
    ))
    if (!is.finite(duplicate_threshold) || duplicate_threshold <= 0) {
        duplicate_threshold <- 0.05
    }
    if (is.finite(duplicate_ratio) && duplicate_ratio <= duplicate_threshold) {
        attr(out, "reason") <- sprintf(
            "duplicate_heavy_sample_unique_ratio_%.4g",
            duplicate_ratio
        )
    }
    out
}

row_center_l2_normalize <- function(x) {
    row_normalize_cpp(x, TRUE)
}

row_l2_normalize <- function(x) {
    row_normalize_cpp(x, FALSE)
}

normalized_euclidean_metric_inputs <- function(
    data,
    points,
    self_query,
    metric,
    storage = c("double", "float")
) {
    metric <- normalize_nn_metric(metric)
    storage <- match.arg(storage)
    if (!metric %in% c("cosine", "correlation")) {
        stop(
            "Normalized Euclidean metric transforms ",
            "require cosine or correlation.",
            call. = FALSE
        )
    }
    if (identical(storage, "float")) {
        float_result <- normalized_float32_metric_inputs(
            data, points, self_query, metric
        )
        if (!is.null(float_result)) return(float_result)
    }
    normalized_double_metric_inputs(data, points, self_query, metric)
}

normalized_float32_metric_inputs <- function(data, points, self_query, metric) {
    data_metric <- normalized_float32_transform_cached(data, metric,
        role = "data")
    if (is.null(data_metric)) return(NULL)
    points_metric <- if (isTRUE(self_query)) data_metric else {
        normalized_float32_transform_cached(points, metric, role = "points")
    }
    list(
        data = data_metric$data,
        points = points_metric$data,
        metric = metric,
        data_zero = data_metric$zero,
        points_zero = if (isTRUE(
            self_query)) data_metric$zero else points_metric$zero,

        transform = normalized_metric_transform_label(metric),
        transform_storage = "float32",
        transform_cache = normalized_float32_cache_metadata(
            data_metric, points_metric, self_query
        )
    )
}

normalized_float32_cache_metadata <- function(data, points, self_query) {
    list(
        enabled = isTRUE(data$cache_enabled),
        data_hit = isTRUE(data$cache_hit),
        points_hit = if (isTRUE(self_query)) {
            isTRUE(data$cache_hit)
        } else {
            isTRUE(points$cache_hit)
        },
        data_key = data$cache_key,
        points_key = if (isTRUE(
            self_query)) data$cache_key else points$cache_key,

        row_major = TRUE
    )
}

normalized_metric_transform_label <- function(metric) {
    if (identical(metric, "correlation")) {
        "row_center_l2_normalize_then_euclidean_graph_search"
    } else {
        "row_l2_normalize_then_euclidean_graph_search"
    }
}

normalized_double_metric_inputs <- function(data, points, self_query, metric) {
    data_metric <- if (identical(metric, "correlation")) {
        row_center_l2_normalize(data)
    } else {
        row_l2_normalize(data)
    }
    points_metric <- if (isTRUE(self_query)) {
        data_metric
    } else if (identical(metric, "correlation")) {
        row_center_l2_normalize(points)
    } else {
        row_l2_normalize(points)
    }
    list(
        data = data_metric,
        points = points_metric,
        metric = metric,
        data_zero = rowSums(data_metric * data_metric) <= 0,
        points_zero = if (isTRUE(self_query)) {
            rowSums(data_metric * data_metric) <= 0
        } else {
            rowSums(points_metric * points_metric) <= 0
        },
        transform = normalized_metric_transform_label(metric)
    )
}

row_inner_product_norm2 <- function(x) {
    x <- as.matrix(x)
    storage.mode(x) <- "double"
    rowSums(x * x)
}

mips_l2_metric_inputs <- function(data, points, self_query) {
    if (is_float32_matrix_input(data) || is_float32_matrix_input(points)) {
        transformed <- mips_l2_float32_transform_cpp(
            data,
            points,
            isTRUE(self_query)
        )
        return(list(
            data = transformed$data,
            points = transformed$points,
            radius2 = as.numeric(transformed$radius2),
            points_norm2 = as.numeric(transformed$points_norm2),
            transform = "maximum_inner_product_to_l2_extra_dimension",
            distance_transform = "mips_l2_to_shifted_inner_product_distance",
            transform_storage = "float32",
            transform_layout = "row_major",
            transform_cache = NULL
        ))
    }
    data <- as.matrix(data)
    storage.mode(data) <- "double"
    points <- as.matrix(points)
    storage.mode(points) <- "double"
    data_norm2 <- row_inner_product_norm2(data)
    points_norm2 <- if (isTRUE(self_query)) {
        data_norm2
    } else {
        row_inner_product_norm2(points)
    }
    radius2 <- faissr_quiet_warning(max(data_norm2, 0, na.rm = TRUE))
    if (!is.finite(radius2) || radius2 < 0) {
        radius2 <- 0
    }
    extra <- sqrt(pmax(0, radius2 - data_norm2))
    data_metric <- cbind(data, extra)
    points_metric <- cbind(points, rep(0, nrow(points)))
    list(
        data = data_metric,
        points = points_metric,
        radius2 = radius2,
        points_norm2 = points_norm2,
        transform = "maximum_inner_product_to_l2_extra_dimension",
        distance_transform = "mips_l2_to_shifted_inner_product_distance",
        transform_storage = "double",
        transform_layout = "column_major",
        transform_cache = NULL
    )
}

finalize_graph_metric_result <- function(result, inputs) {
    if (
        identical(
            inputs$distance_transform,
            "mips_l2_to_shifted_inner_product_distance"
        )
    ) {
        return(finalize_mips_l2_metric_result(result, inputs))
    }
    finalize_normalized_euclidean_metric_result(result, inputs)
}

knn_result_has_gpu_residency <- function(result) {
    !is.null(attr(result, "gpu_residency", exact = TRUE)) ||
        is.list(result$gpu_residency) ||
        identical(result$accelerator %||% NA_character_, "cuda")
}

reject_cuda_r_side_output_cleanup <- function(
    backend,
    exclude_self,
    cleanup = "include_self_output"
) {
    if (isTRUE(exclude_self)) {
        return(invisible(FALSE))
    }
    stop(
        "`backend = \"",
        backend,
        "\"` currently requires `exclude_self = TRUE` ",
        "for this CUDA graph route because ",
        "include-self output shaping must be ",
        "implemented by the CUDA/C++ backend, not by R-side cleanup.",
        call. = FALSE
    )
}

finalize_normalized_euclidean_metric_result <- function(result, inputs) {
    result <- normalized_euclidean_to_similarity_distance(
        result,
        data_zero = inputs$data_zero,
        points_zero = inputs$points_zero
    )
    result$metric_transform <- inputs$transform
    result$metric <- inputs$metric %||% result$metric
    attr(result, "metric") <- result$metric
    attr(result, "metric_transform") <- inputs$transform
    attr(
        result,
        "distance_transform"
    ) <- .normalized_similarity_distance_transform
    approximation <- attr(result, "approximation")
    if (!is.null(approximation)) {
        approximation$metric_transform <- inputs$transform
        approximation$distance_transform <-
            .normalized_similarity_distance_transform
        attr(result, "approximation") <- approximation
    }
    if (knn_result_has_gpu_residency(result)) {
        return(result)
    }
    sort_knn_rows_by_distance_index(result)
}

finalize_mips_l2_metric_result <- function(result, inputs) {
    if (ncol(result$indices) > 0L) {
        result$distances <- mips_l2_to_shifted_inner_product_distance_cpp(
            result$distances,
            as.numeric(inputs$points_norm2),
            as.numeric(inputs$radius2)
        )
    }
    result$metric_transform <- inputs$transform
    attr(result, "metric_transform") <- inputs$transform
    attr(result, "distance_transform") <- inputs$distance_transform
    approximation <- attr(result, "approximation")
    if (!is.null(approximation)) {
        approximation$metric_transform <- inputs$transform
        approximation$distance_transform <- inputs$distance_transform
        attr(result, "approximation") <- approximation
    }
    if (knn_result_has_gpu_residency(result)) {
        return(result)
    }
    sort_knn_rows_by_distance_index(result)
}

reject_cuda_normalized_cpu_repair <- function(
    accelerator,
    metric,
    data_zero,
    points_zero,
    backend
) {
    if (!identical(accelerator, "cuda")) {
        return(invisible(FALSE))
    }
    if (!identical(metric, "cosine") && !identical(metric, "correlation")) {
        return(invisible(FALSE))
    }
    if (!any(data_zero) && !any(points_zero)) {
        return(invisible(FALSE))
    }
    stop(
        "`backend = \"",
        backend,
        "\"`, `metric = \"",
        metric,
        "\"` contains ",
        if (identical(metric, "correlation")) "constant" else "all-zero",
        " rows. faissR does not repair zero-normalized CUDA results on CPU; ",
        "remove or perturb those rows, use `backend = \"cpu\"`, or use ",
        "`metric = \"euclidean\"` for this CUDA run.",
        call. = FALSE
    )
}

faiss_flat_normalized_metric_result <- function(
    data, points, k, self_query, exclude_self, metric, backend,
    accelerator = NULL, n_threads = NULL, output = "double",
    target_recall = 0.99, requested_method = NULL
) {
    data_dim <- if (is_float32_matrix_input(data)) {
        float32_matrix_dims(data, "data")
    } else {
        dim(data)
    }
    metric <- normalize_nn_metric(metric)
    exact_params <- normalized_flat_exact_params(
        data_dim, k, metric, target_recall, requested_method, accelerator
    )
    route <- prepare_normalized_faiss_route(
        data, points, self_query, metric, accelerator, backend
    )
    if (is.null(accelerator) && normalized_route_has_zero_rows(route)) {
        return(normalized_flat_zero_fallback(
            data, points, k, self_query, exclude_self, metric, backend,
            n_threads, output, exact_params, route
        ))
    }
    out <- execute_normalized_faiss_flat(
        route, k, exclude_self, n_threads, accelerator, exact_params
    )
    result <- finish_nn_result(
        out, backend, k, self_query, exact = TRUE, metric = metric
    )
    if (!identical(accelerator, "cuda")) {
        result <- restore_zero_normalized_ip_distances(
            result, route$inputs$data_zero, route$inputs$points_zero,
            isTRUE(exclude_self)
        )
        result <- sort_knn_rows_by_distance_index(result)
    }
    attr(result, "faiss") <- normalized_flat_metadata(
        out, route, accelerator
    )
    if (is.null(accelerator)) {
        result <- attach_cpu_exact_tuning(result, exact_params, output,
            n_threads)
    }
    if (isTRUE(route$float32)) result <- finish_float32_direct_result(result,
        out)
    result
}

normalized_flat_exact_params <- function(
    data_dim, k, metric, target_recall, requested_method, accelerator
) {
    if (!is.null(accelerator)) return(NULL)
    cpu_flatlike_params(
        data_dim[[1L]], data_dim[[2L]], k, metric = metric,
        target_recall = target_recall, requested_method = requested_method
    )
}

normalized_route_has_zero_rows <- function(route) {
    any(route$inputs$data_zero) || any(route$inputs$points_zero)
}

normalized_flat_zero_fallback <- function(
    data, points, k, self_query, exclude_self, metric, backend,
    n_threads, output, exact_params, route
) {
    inputs <- normalized_double_metric_inputs(data, points, self_query, metric)
    out <- nn_cpp(
        inputs$data, inputs$points, as.integer(k), "cosine", FALSE, TRUE,
        0, TRUE, as.integer(normalize_nn_threads(n_threads)),
        isTRUE(exclude_self)
    )
    result <- finish_nn_result(
        out, backend, k, self_query, exact = TRUE, metric = metric
    )
    metadata <- normalized_flat_metadata(out, route, NULL)
    metadata$index_type <- "IndexFlatIP"
    metadata$zero_row_exact_fallback <- TRUE
    attr(result, "faiss") <- metadata
    attach_cpu_exact_tuning(result, exact_params, output, n_threads)
}

execute_normalized_faiss_flat <- function(
    route, k, exclude_self, n_threads, accelerator, exact_params
) {
    with_faiss_query_batch_size(exact_params %||% list(), {
        if (isTRUE(route$float32) && identical(accelerator, "cuda")) {
            nn_faiss_gpu_flat_float32_cpp(
                route$data, route$points, as.integer(k), isTRUE(exclude_self),
                "inner_product", "one_minus_inner_product", "double"
            )
        } else if (isTRUE(route$float32)) {
            nn_faiss_flat_pretransformed_float32_cpp(
                route$data, route$points, as.integer(k), isTRUE(exclude_self),
                as.integer(normalize_nn_threads(n_threads)), "double"
            )
        } else if (identical(accelerator, "cuda")) {
            nn_faiss_gpu_flat_normalized_ip_distance_cpp(
                route$data, route$points, as.integer(k), isTRUE(exclude_self)
            )
        } else {
            nn_faiss_flat_normalized_ip_distance_cpp(
                route$data, route$points, as.integer(k), isTRUE(exclude_self),
                as.integer(normalize_nn_threads(n_threads))
            )
        }
    })
}

normalized_flat_metadata <- function(out, route, accelerator) {
    metadata <- list(
        index_type = as.character(out$index_type), library = "faiss",
        backend = if (identical(accelerator, "cuda")) "cuda" else "cpu",
        metric = route$metric,
        transform = if (identical(route$metric, "correlation")) {
            "row_center_l2_normalize_then_IndexFlatIP"
        } else "row_l2_normalize_then_IndexFlatIP",
        transform_storage = route$inputs$transform_storage %||% "double",
        transform_cache = route$inputs$transform_cache %||% NULL
    )
    if (!is.null(accelerator)) metadata$accelerator <- accelerator
    metadata
}

faiss_ivf_normalized_metric_result <- function(
    data,
    points,
    k,
    self_query,
    exclude_self,
    metric,
    backend,
    accelerator = NULL,
    n_threads = NULL,
    params,
    tuning_metadata = NULL
) {
    route <- prepare_normalized_faiss_route(
        data, points, self_query, metric, accelerator, backend
    )
    out <- execute_normalized_faiss_ivf(
        route, k, exclude_self, params, n_threads, accelerator
    )
    result <- finish_normalized_faiss_approximation(
        out, route, backend, k, self_query, exclude_self, accelerator
    )
    attr(result, "approximation") <- normalized_ivf_metadata(
        out, route, backend, params, accelerator, tuning_metadata
    )
    result <- append_nn_tuning_metadata(result, params)
    if (isTRUE(route$float32)) result <- finish_float32_direct_result(result,
        out)
    result
}

prepare_normalized_faiss_route <- function(
    data, points, self_query, metric, accelerator, backend
) {
    metric <- normalize_nn_metric(metric)
    inputs <- normalized_euclidean_metric_inputs(
        data, points, self_query, metric, storage = "float"
    )
    reject_cuda_normalized_cpu_repair(
        accelerator, metric, inputs$data_zero, inputs$points_zero, backend
    )
    list(
        data = inputs$data, points = inputs$points, inputs = inputs,
        metric = metric,
        float32 = identical(inputs$transform_storage %||% "double", "float32")
    )
}

execute_normalized_faiss_ivf <- function(
    route, k, exclude_self, params, n_threads, accelerator
) {
    if (isTRUE(route$float32) && identical(accelerator, "cuda")) {
        nn_faiss_gpu_ivf_flat_float32_cpp(
            route$data, route$points,
            as.integer(k),
            as.integer(params$nlist),
            as.integer(params$nprobe),
            "inner_product",
            "one_minus_inner_product",
            isTRUE(exclude_self),
            "double"
        )
    } else if (isTRUE(route$float32)) {
        nn_faiss_ivf_float32_cpp(
            route$data, route$points,
            as.integer(k),
            as.integer(params$nlist),
            as.integer(params$nprobe),
            "inner_product",
            "one_minus_inner_product",
            isTRUE(exclude_self),
            as.integer(normalize_nn_threads(n_threads)),
            "double"
        )
    } else if (identical(accelerator, "cuda")) {
        nn_faiss_gpu_ivf_flat_cpp(
            route$data, route$points,
            as.integer(k),
            as.integer(params$nlist),
            as.integer(params$nprobe),
            "inner_product",
            "one_minus_inner_product",
            isTRUE(exclude_self)
        )
    } else {
        nn_faiss_ivf_cpp(
            route$data, route$points,
            as.integer(k),
            as.integer(params$nlist),
            as.integer(params$nprobe),
            "inner_product",
            "one_minus_inner_product",
            isTRUE(exclude_self),
            as.integer(normalize_nn_threads(n_threads))
        )
    }
}

finish_normalized_faiss_approximation <- function(
    out, route, backend, k, self_query, exclude_self, accelerator
) {
    result <- finish_nn_result(
        out, backend, k, self_query, exact = FALSE, metric = route$metric
    )
    if (!identical(accelerator, "cuda")) {
        result <- restore_zero_normalized_ip_distances(
            result,
            data_zero = route$inputs$data_zero,
            points_zero = route$inputs$points_zero,
            exclude_self = isTRUE(exclude_self)
        )
        result <- sort_knn_rows_by_distance_index(result)
    }
    result
}

normalized_ivf_metadata <- function(
    out, route, backend, params, accelerator, tuning_metadata
) {
    metadata <- list(
        strategy = if (identical(accelerator, "cuda")) {
            "faiss_gpu_IndexIVFFlat_cuVS"
        } else {
            "faiss_IndexIVFFlat"
        },
        backend = backend,
        library = "faiss",
        accelerator = accelerator,
        metric = route$metric,
        transform = if (identical(route$metric, "correlation")) {
            "row_center_l2_normalize_then_IndexIVFFlat_METRIC_INNER_PRODUCT"
        } else {
            "row_l2_normalize_then_IndexIVFFlat_METRIC_INNER_PRODUCT"
        },
        nlist = as.integer(out$nlist),
        nprobe = as.integer(out$nprobe),
        requested_nlist = as.integer(params$requested_nlist),
        requested_nprobe = as.integer(params$requested_nprobe),
        ivf_parameters_adjusted = !identical(
            as.integer(params$requested_nlist),
            as.integer(out$nlist)
        ) ||
            !identical(
                as.integer(params$requested_nprobe),
                as.integer(out$nprobe)
        ),
        tuning = tuning_metadata,
        transform_storage = route$inputs$transform_storage %||% "double",
        transform_cache = route$inputs$transform_cache %||% NULL
    )
    if (is.null(accelerator)) metadata$accelerator <- NULL
    metadata
}

faiss_ivfpq_normalized_metric_result <- function(
    data,
    points,
    k,
    self_query,
    exclude_self,
    metric,
    backend,
    accelerator = NULL,
    n_threads = NULL,
    params,
    pq
) {
    route <- prepare_normalized_faiss_route(
        data, points, self_query, metric, accelerator, backend
    )
    out <- execute_normalized_faiss_ivfpq(
        route, k, exclude_self, params, pq, n_threads, accelerator
    )
    result <- finish_normalized_faiss_approximation(
        out, route, backend, k, self_query, exclude_self, accelerator
    )
    attr(result, "approximation") <- normalized_ivfpq_metadata(
        out, route, backend, params, accelerator
    )
    result <- append_nn_tuning_metadata(
        result, params, pq, .prefixes = list(NULL, "pq_")
    )
    if (isTRUE(route$float32)) result <- finish_float32_direct_result(result,
        out)
    result
}

execute_normalized_faiss_ivfpq <- function(
    route, k, exclude_self, params, pq, n_threads, accelerator
) {
    if (isTRUE(route$float32) && identical(accelerator, "cuda")) {
        nn_faiss_gpu_ivfpq_float32_cpp(
            route$data, route$points,
            as.integer(k), as.integer(params$nlist),
            as.integer(params$nprobe), as.integer(pq$m),
            as.integer(pq$nbits), "inner_product", "one_minus_inner_product",
            isTRUE(exclude_self),
            "double"
        )
    } else if (isTRUE(route$float32)) {
        nn_faiss_ivfpq_float32_cpp(
            route$data, route$points,
            as.integer(k), as.integer(params$nlist),
            as.integer(params$nprobe), as.integer(pq$m),
            as.integer(pq$nbits), "inner_product", "one_minus_inner_product",
            isTRUE(exclude_self),
            as.integer(normalize_nn_threads(n_threads)),
            "double"
        )
    } else if (identical(accelerator, "cuda")) {
        nn_faiss_gpu_ivfpq_cpp(
            route$data, route$points,
            as.integer(k), as.integer(params$nlist),
            as.integer(params$nprobe), as.integer(pq$m),
            as.integer(pq$nbits), "inner_product", "one_minus_inner_product",
            isTRUE(exclude_self)
        )
    } else {
        nn_faiss_ivfpq_cpp(
            route$data, route$points,
            as.integer(k), as.integer(params$nlist),
            as.integer(params$nprobe), as.integer(pq$m),
            as.integer(pq$nbits), "inner_product", "one_minus_inner_product",
            isTRUE(exclude_self),
            as.integer(normalize_nn_threads(n_threads))
        )
    }
}

normalized_ivfpq_metadata <- function(
    out, route, backend, params, accelerator
) {
    metadata <- list(
        strategy = if (identical(accelerator, "cuda")) {
            "faiss_gpu_IndexIVFPQ_cuVS"
        } else {
            "faiss_IndexIVFPQ"
        },
        backend = backend,
        library = "faiss",
        accelerator = accelerator,
        metric = route$metric,
        transform = if (identical(route$metric, "correlation")) {
            "row_center_l2_normalize_then_IndexIVFPQ_METRIC_INNER_PRODUCT"
        } else {
            "row_l2_normalize_then_IndexIVFPQ_METRIC_INNER_PRODUCT"
        },
        role = if (identical(accelerator, "cuda")) {
            "explicit_memory_pressure_backend"
        } else {
            NULL
        },
        default_candidate = if (identical(accelerator, "cuda")) FALSE else NULL,
        nlist = as.integer(out$nlist),
        nprobe = as.integer(out$nprobe),
        requested_nlist = as.integer(params$requested_nlist),
        requested_nprobe = as.integer(params$requested_nprobe),
        ivf_parameters_adjusted = !identical(
            as.integer(params$requested_nlist),
            as.integer(out$nlist)
        ) ||
            !identical(
                as.integer(params$requested_nprobe),
                as.integer(out$nprobe)
            ),
        pq_m = as.integer(out$pq_m),
        pq_nbits = as.integer(out$pq_nbits),
        requested_pq_m = as.integer(out$requested_pq_m),
        requested_pq_nbits = as.integer(out$requested_pq_nbits),
        pq_parameters_adjusted = isTRUE(out$pq_parameters_adjusted),
        transform_storage = route$inputs$transform_storage %||% "double",
        transform_cache = route$inputs$transform_cache %||% NULL
    )
    for (name in c("accelerator", "role", "default_candidate")) {
        if (is.null(metadata[[name]])) metadata[[name]] <- NULL
    }
    metadata
}

faiss_ivfpq_fastscan_normalized_metric_result <- function(
    data, points, k, self_query, exclude_self, metric,
    n_threads = NULL, params
) {
    metric <- normalize_nn_metric(metric)
    if (!metric %in% c("cosine", "correlation")) {
        stop(
            "CPU IVFPQ FastScan normalized route currently ",
            "supports cosine and correlation only.",
            call. = FALSE
        )
    }
    route <- prepare_normalized_faiss_route(
        data, points, self_query, metric, NULL, "faiss_ivfpq_fastscan"
    )
    cached <- normalized_fastscan_cached_result(
        route, k, self_query, exclude_self, n_threads, params
    )
    if (!is.null(cached)) return(cached)
    out <- execute_normalized_fastscan(route, k, exclude_self, n_threads,
        params)
    result <- finish_nn_result(
        out, "faiss_ivfpq_fastscan", k, self_query,
        exact = FALSE, metric = metric
    )
    result <- finalize_normalized_euclidean_metric_result(result, route$inputs)
    attr(result, "approximation") <- normalized_fastscan_metadata(
        out, route, params
    )
    result <- append_fastscan_tuning_metadata(result, params)
    if (isTRUE(route$float32)) result <- finish_float32_direct_result(result,
        out)
    result
}

normalized_fastscan_cached_result <- function(
    route, k, self_query, exclude_self, n_threads, params
) {
    result <- fitted_nn_index_result(
        data = route$data, points = route$points, k = k,
        backend = "faiss_ivfpq_fastscan",
        result_backend = "faiss_ivfpq_fastscan", self_query = self_query,
        exclude_self = isTRUE(exclude_self), metric = "euclidean",
        n_threads = n_threads, output = "double",
        params = ivfpq_fastscan_fitted_params(params), pq = params$pq,
        target_recall = params$ivf$target_recall %||% 0.99
    )
    if (is.null(result)) return(NULL)
    result <- finalize_normalized_euclidean_metric_result(result, route$inputs)
    approximation <- attr(result, "approximation", exact = TRUE) %||% list()
    approximation <- utils::modifyList(
        approximation, normalized_fastscan_transform_metadata(route)
    )
    attr(result, "approximation") <- approximation
    append_fastscan_tuning_metadata(result, params)
}

normalized_fastscan_transform_metadata <- function(route) {
    list(
        strategy = .fastscan_normalized_strategy, metric = route$metric,
        transform = if (identical(route$metric, "correlation")) {
            "row_center_l2_normalize_then_IndexIVFPQFastScan_METRIC_L2"
        } else "row_l2_normalize_then_IndexIVFPQFastScan_METRIC_L2",
        metric_transform = route$inputs$transform,
        distance_transform = .normalized_similarity_distance_transform,
        transform_storage = route$inputs$transform_storage %||% "double",
        transform_cache = route$inputs$transform_cache %||% NULL
    )
}

execute_normalized_fastscan <- function(
    route, k, exclude_self, n_threads, params
) {
    out <- nn_faiss_ivfpq_fastscan_float32_cpp(
        route$data, route$points,
        as.integer(k),
        as.integer(params$ivf$nlist),
        as.integer(params$ivf$nprobe),
        as.integer(params$pq$m),
        "euclidean",
        "euclidean",
        as.integer(params$refine_factor),
        as.integer(params$bbs),
        isTRUE(exclude_self),
        as.integer(normalize_nn_threads(n_threads)),
        "double"
    )
    out
}

normalized_fastscan_metadata <- function(out, route, params) {
    c(normalized_fastscan_transform_metadata(route), list(
        backend = "faiss_ivfpq_fastscan",
        library = "faiss",
        accelerator = NULL,
        input_type = out$input_type %||% "float32",
        nlist = as.integer(out$nlist),
        nprobe = as.integer(out$nprobe),
        requested_nlist = as.integer(params$ivf$requested_nlist),
        requested_nprobe = as.integer(params$ivf$requested_nprobe),
        pq_m = as.integer(out$pq_m),
        pq_nbits = as.integer(out$pq_nbits),
        requested_pq_m = as.integer(out$requested_pq_m),
        requested_pq_nbits = as.integer(out$requested_pq_nbits),
        refine = isTRUE(out$refine),
        refine_factor = as.integer(out$refine_factor),
        requested_refine_factor = as.integer(out$requested_refine_factor),
        bbs = as.integer(out$bbs),
        requested_bbs = as.integer(out$requested_bbs),
        ivf_parameters_adjusted = !identical(
            as.integer(params$ivf$requested_nlist),
            as.integer(out$nlist)
        ) ||
            !identical(
                as.integer(params$ivf$requested_nprobe),
                as.integer(out$nprobe)
            ),
        pq_parameters_adjusted = isTRUE(out$pq_parameters_adjusted),
        ivfpq_fastscan = TRUE,
        fastscan = TRUE
    ))
}

append_fastscan_tuning_metadata <- function(result, params) {
    append_nn_tuning_metadata(
        result,
        params$ivf,
        params$pq,
        params$tuning,
        .prefixes = list(NULL, "pq_", "ivfpq_fastscan_")
    )
}

faiss_hnsw_normalized_metric_result <- function(
    data,
    points,
    k,
    self_query,
    exclude_self,
    metric,
    n_threads = NULL,
    target_recall = 0.99
) {
    metric <- normalize_nn_metric(metric)
    target_recall <- normalize_hnsw_target_recall(target_recall)
    params <- faiss_hnsw_params(
        k,
        n = nrow(data),
        p = ncol(data),
        metric = metric,
        target_recall = target_recall
    )
    route <- prepare_normalized_faiss_route(
        data, points, self_query, metric, NULL, "faiss_hnsw"
    )
    out <- execute_normalized_faiss_hnsw(
        route, k, exclude_self, params, n_threads
    )
    result <- finish_normalized_faiss_approximation(
        out, route, "faiss_hnsw", k, self_query, exclude_self, NULL
    )
    attr(result, "approximation") <- normalized_hnsw_metadata(
        out, route, params, target_recall
    )
    if (isTRUE(route$float32)) result <- finish_float32_direct_result(result,
        out)
    result
}

execute_normalized_faiss_hnsw <- function(
    route, k, exclude_self, params, n_threads
) {
    if (isTRUE(route$float32)) {
        nn_faiss_hnsw_float32_cpp(
            route$data, route$points,
            as.integer(k),
            as.integer(params$m),
            as.integer(params$ef_construction),
            as.integer(params$ef_search),
            "inner_product",
            "one_minus_inner_product",
            isTRUE(exclude_self),
            as.integer(normalize_nn_threads(n_threads)),
            "double"
        )
    } else {
        nn_faiss_hnsw_cpp(
            route$data, route$points,
            as.integer(k),
            as.integer(params$m),
            as.integer(params$ef_construction),
            as.integer(params$ef_search),
            "inner_product",
            "one_minus_inner_product",
            isTRUE(exclude_self),
            as.integer(normalize_nn_threads(n_threads))
        )
    }
}

normalized_hnsw_metadata <- function(out, route, params, target_recall) {
    metadata <- list(
        strategy = "faiss_IndexHNSWFlat",
        backend = "faiss_hnsw",
        library = "faiss",
        metric = route$metric,
        transform = if (identical(route$metric, "correlation")) {
            "row_center_l2_normalize_then_IndexHNSWFlat_METRIC_INNER_PRODUCT"
        } else {
            "row_l2_normalize_then_IndexHNSWFlat_METRIC_INNER_PRODUCT"
        },
        m = as.integer(out$m),
        ef_construction = as.integer(out$ef_construction),
        ef_search = as.integer(out$ef_search),
        requested_m = as.integer(out$requested_m),
        requested_ef_construction = as.integer(out$requested_ef_construction),
        requested_ef_search = as.integer(out$requested_ef_search),
        hnsw_parameters_adjusted = isTRUE(out$hnsw_parameters_adjusted),
        transform_storage = route$inputs$transform_storage %||% "double",
        transform_cache = route$inputs$transform_cache %||% NULL
    )
    c(metadata, hnsw_tuning_metadata(params, target_recall))
}

restore_zero_normalized_ip_distances <- function(
    result,
    data_zero,
    points_zero,
    exclude_self = FALSE
) {
    if (!any(points_zero) || !any(data_zero) || ncol(result$indices) < 1L) {
        return(result)
    }
    restored <- restore_zero_normalized_ip_distances_cpp(
        result$indices,
        result$distances,
        data_zero,
        points_zero,
        isTRUE(attr(result, "self_query")),
        isTRUE(exclude_self)
    )
    result$indices <- restored$indices
    result$distances <- restored$distances
    result
}

sort_knn_rows_by_distance_index <- function(result) {
    if (ncol(result$indices) < 2L) {
        return(result)
    }
    sorted <- sort_knn_rows_cpp(result$indices, result$distances)
    result$indices <- sorted$indices
    result$distances <- sorted$distances
    result
}

.nn_tuning_metadata_fields <- c(
        "tuning_policy",
        "tuning_rule",
        "tuning_low_dim",
        "tuning_high_dim",
        "tuning_medium_n",
        "tuning_huge_low_dim",
        "tuning_runtime_guard",
        "tuning_large_n",
        "tuning_small_k",
        "tuning_large_k",
        "tuning_non_euclidean",
        "tuning_metric",
        "tuning_metric_aware",
        "target_recall",
        "requested_target_recall",
        "tuning_shape_group",
        "tuning_cuda_shape_group",
        "tuning_k_bucket",
        "tuning_rule_basis",
        "tuning_target_recall_code",
        "tuning_benchmark_basis",
        "tuning_benchmark_target_met",
        "tuning_benchmark_source",
        "tuning_source",
        "expected_recall_at_k",
        "exact_recall_by_construction",
        "recommended_n_threads",
        "faiss_query_batch_size",
        "cache_fitted_indexes",
        "recommended_output",
        "result_backend",
        "resolved_backend",
        "distance_type",
        "input_type",
        "input_layout"
)

nn_tuning_metadata <- function(params, prefix = NULL) {
    if (!is.list(params)) {
        return(list())
    }
    fields <- .nn_tuning_metadata_fields[
        .nn_tuning_metadata_fields %in% names(params)
    ]
    out <- params[fields]
    out <- c(
        out,
        list(
            target_recall_statistic = "mean_query_recall_at_k",
            target_recall_replicate_rule = .validation_replicate_rule,
            target_recall_min_query_role = "diagnostic_only"
        )
    )
    if (!is.null(prefix) && length(out)) {
        names(out) <- paste0(prefix, names(out))
    }
    out
}

append_nn_tuning_metadata <- function(result, ..., .prefixes = NULL) {
    params <- list(...)
    if (!length(params)) {
        return(result)
    }
    result$target_recall_statistic <- "mean_query_recall_at_k"
    result$target_recall_replicate_rule <-
        .validation_replicate_rule
    result$target_recall_min_query_role <- "diagnostic_only"
    approx <- attr(result, "approximation") %||% list()
    if (is.null(.prefixes)) {
        .prefixes <- rep(list(NULL), length(params))
    }
    for (i in seq_along(params)) {
        approx <- c(
            approx,
            nn_tuning_metadata(params[[i]], prefix = .prefixes[[i]])
        )
    }
    attr(result, "approximation") <- approx
    result
}

normalized_euclidean_to_similarity_distance <- function(
    result,
    data_zero,
    points_zero
) {
    if (ncol(result$indices) < 1L) {
        return(result)
    }
    result$distances <- normalized_euclidean_to_similarity_distance_cpp(
        result$indices,
        result$distances,
        data_zero,
        points_zero
    )
    result
}

should_use_clustered_self_knn <- function(
    backend,
    self_query,
    n,
    p,
    k,
    work_size
) {
    FALSE
}

fast_knn_approx_seed <- function() {
    value <- faissr_option("approx_knn_seed", 4L)
    value <- faissr_quiet_warning(as.integer(value))
    if (length(value) != 1L || is.na(value) || !is.finite(value)) 4L else value
}

native_nsg_params <- function(
    n,
    p,
    k,
    metric = "euclidean",
    backend = c("cpu", "cuda"),
    target_recall = 0.99
) {
    metric <- normalize_nn_metric(metric)
    backend <- match.arg(backend)
    target_recall <- normalize_hnsw_target_recall(target_recall)
    option_prefix <- if (identical(backend, "cuda")) "cuda_nsg" else "cpu_nsg"
    nn_tune_native_nsg_cpp(
        as.integer(n),
        as.integer(p),
        as.integer(k),
        metric,
        backend,
        as.numeric(target_recall),
        nn_option_int_or_na(paste0(option_prefix, "_r")),
        nn_option_int_or_na(paste0(option_prefix, "_graph_k"))
    )
}

cuda_nsg_params <- function(
    n,
    p,
    k,
    metric = "euclidean",
    target_recall = 0.99
) {
    native_nsg_params(
        n = n,
        p = p,
        k = k,
        metric = metric,
        backend = "cuda",
        target_recall = target_recall
    )
}

nsg_pair_score <- function(data, i, j, metric = "euclidean") {
    xi <- data[i, , drop = TRUE]
    xj <- data[j, , drop = TRUE]
    if (identical(metric, "inner_product")) {
        return(-sum(xi * xj))
    }
    sqrt(sum((xi - xj)^2))
}

candidate_graph_hnsw_seed_knn <- function(
    data,
    k,
    metric = "euclidean",
    n_threads = NULL
) {
    metric <- normalize_nn_metric(metric)
    if (isTRUE(faiss_available())) {
        out <- if (is_float32_matrix_input(data)) {
            nn_compute(
                data,
                data,
                k,
                "faiss_hnsw",
                points_missing = TRUE,
                exclude_self = TRUE,
                n_threads = n_threads,
                metric = metric
            )
        } else {
            faiss_self_knn(
                data,
                k = k,
                backend = "faiss_hnsw",
                metric = metric,
                n_threads = n_threads
            )
        }
        attr(out, "seed_backend") <- "faiss_hnsw"
        return(out)
    }
    NULL
}

nsg_prune_candidate_graph <- function(
    data,
    seed_indices,
    r,
    metric = "euclidean",
    protect_top = 0L
) {
    n <- nrow(seed_indices)
    r <- as.integer(min(max(1L, r), ncol(seed_indices), max(1L, n - 1L)))
    protect_top <- faissr_quiet_warning(as.integer(protect_top))
    if (
        length(protect_top) != 1L ||
            is.na(protect_top) ||
            !is.finite(protect_top)
    ) {
        protect_top <- 0L
    }
    protect_top <- as.integer(max(0L, min(protect_top, r, ncol(seed_indices))))
    max_exact_work <- faissr_option("cuda_nsg_prune_max_work", 2e8)
    max_exact_work <- faissr_quiet_warning(as.numeric(max_exact_work))
    if (
        length(max_exact_work) != 1L ||
            is.na(max_exact_work) ||
            !is.finite(max_exact_work)
    ) {
        max_exact_work <- 2e8
    }
    prune_fun <- if (is_float32_matrix_input(data)) {
        graph_prune_candidate_graph_float32_cpp
    } else {
        graph_prune_candidate_graph_cpp
    }
    prune_fun(
        data,
        seed_indices,
        as.integer(r),
        1,
        metric,
        as.integer(protect_top),
        max_exact_work,
        FALSE
    )
}

vamana_params <- function(
    n,
    p,
    k,
    metric = "euclidean",
    backend = c("cpu", "cuda"),
    target_recall = 0.99
) {
    metric <- normalize_nn_metric(metric)
    backend <- match.arg(backend)
    target_recall <- normalize_hnsw_target_recall(target_recall)
    option_prefix <- if (identical(backend, "cuda")) {
        "cuda_vamana"
    } else {
        "cpu_vamana"
    }
    out <- nn_tune_vamana_cpp(
        as.integer(n),
        as.integer(p),
        as.integer(k),
        metric,
        backend,
        as.numeric(target_recall),
        nn_option_int_or_na(c(paste0(option_prefix, "_r"), "faiss_vamana_r")),
        nn_option_int_or_na(c(
            paste0(option_prefix, "_search_l"),
            "faiss_vamana_search_l"
        )),
        vamana_alpha_option(option_prefix)
    )
    out$backend <- backend
    if (identical(backend, "cuda")) {
        out$seed_backend <- if (identical(metric, "euclidean")) {
            "cuda_exact"
        } else {
            "exact"
        }
    }
    out
}

vamana_alpha_option <- function(option_prefix) {
    backend_name <- paste0(option_prefix, "_alpha")
    if (!is.null(faissr_option(backend_name, NULL))) {
        return(nn_option_double_or_na(backend_name))
    }
    if (is.null(faissr_option("vamana_alpha", NULL))) NA_real_ else {
        nn_option_double_or_na("vamana_alpha")
    }
}

vamana_robust_prune_candidate_graph <- function(
    data,
    seed_indices,
    r,
    alpha = 1.2,
    metric = "euclidean",
    protect_top = 0L
) {
    n <- nrow(seed_indices)
    r <- as.integer(min(max(1L, r), ncol(seed_indices), max(1L, n - 1L)))
    protect_top <- faissr_quiet_warning(as.integer(protect_top))
    if (
        length(protect_top) != 1L ||
            is.na(protect_top) ||
            !is.finite(protect_top)
    ) {
        protect_top <- 0L
    }
    protect_top <- as.integer(max(0L, min(protect_top, r, ncol(seed_indices))))
    alpha <- as.numeric(alpha)
    if (length(alpha) != 1L || is.na(alpha) || !is.finite(alpha) || alpha < 1) {
        alpha <- 1.2
    }
    max_exact_work <- faissr_option("vamana_prune_max_work", 2e8)
    max_exact_work <- faissr_quiet_warning(as.numeric(max_exact_work))
    if (
        length(max_exact_work) != 1L ||
            is.na(max_exact_work) ||
            !is.finite(max_exact_work)
    ) {
        max_exact_work <- 2e8
    }
    prune_fun <- if (is_float32_matrix_input(data)) {
        graph_prune_candidate_graph_float32_cpp
    } else {
        graph_prune_candidate_graph_cpp
    }
    prune_fun(
        data,
        seed_indices,
        as.integer(r),
        alpha,
        metric,
        as.integer(protect_top),
        max_exact_work,
        TRUE
    )
}

vamana_self_knn <- function(
    data,
    k,
    r,
    search_l,
    alpha = 1.2,
    metric = "euclidean",
    use_cuda = FALSE,
    n_threads = NULL,
    seed_backend = "exact"
) {
    metric <- normalize_nn_metric(metric)
    if (!metric %in% c("euclidean", "inner_product")) {
        stop(
            "Vamana candidate refinement supports ",
            "Euclidean or inner-product scoring.",
            call. = FALSE
        )
    }
    if (nrow(data) < 2L) {
        stop("Vamana requires at least two rows.", call. = FALSE)
    }
    search_l <- as.integer(min(max(k, search_l), nrow(data) - 1L))
    r <- as.integer(min(max(k, r), search_l))
    seed <- candidate_graph_seed(
        data, search_l, metric, use_cuda, n_threads, seed_backend
    )
    seed_backend <- attr(seed, "resolved_seed_backend", exact = TRUE)
    graph <- vamana_robust_prune_candidate_graph(
        data,
        seed$indices,
        r = r,
        alpha = alpha,
        metric = metric,
        protect_top = k
    )
    out <- refine_candidate_graph(data, graph, k, metric, use_cuda, n_threads)
    attr(out, "approximation") <- vamana_result_metadata(
        graph, seed_backend, search_l, alpha
    )
    out
}

vamana_result_metadata <- function(graph, seed_backend, search_l, alpha) {
    list(
        seed_backend = seed_backend,
        candidate_columns = as.integer(ncol(graph)),
        candidate_layout = attr(graph, "candidate_layout", exact = TRUE) %||%
            NA_character_,
        pruning_rule = attr(graph, "pruning_rule", exact = TRUE) %||%
            NA_character_,
        seed_search_l = as.integer(search_l),
        alpha = as.numeric(alpha),
        protected_seed_neighbors = as.integer(
            attr(graph, "protected_top", exact = TRUE) %||% 0L
        ),
        exact_robust_prune = isTRUE(attr(graph, "exact_prune", exact = TRUE)),
        cuvs_vamana_note = paste0(
            "cuVS Vamana currently builds/serializes ",
            "DiskANN-compatible graphs; faissR performs ",
            "KNN refinement inside the candidate graph."
        )
    )
}

candidate_graph_seed <- function(
    data, graph_k, metric, use_cuda, n_threads, seed_backend
) {
    seed <- candidate_graph_requested_seed(
        data, graph_k, metric, use_cuda, n_threads, seed_backend
    )
    if (!is.null(seed)) {
        return(seed)
    }
    candidate_graph_fallback_seed(data, graph_k, metric, use_cuda, n_threads)
}

candidate_graph_requested_seed <- function(
    data, graph_k, metric, use_cuda, n_threads, seed_backend
) {
    if (isTRUE(use_cuda) || !identical(seed_backend, "faiss_hnsw")) {
        return(NULL)
    }
    out <- candidate_graph_hnsw_seed_knn(
        data, k = graph_k, metric = metric, n_threads = n_threads
    )
    if (!is.null(out)) {
        attr(out, "resolved_seed_backend") <- paste0(
            "native_", attr(out, "seed_backend", exact = TRUE), "_seed"
        )
    }
    out
}

candidate_graph_fallback_seed <- function(
    data, graph_k, metric, use_cuda, n_threads
) {
    backend <- candidate_graph_seed_backend(data, metric, use_cuda)
    if (!is.null(backend)) {
        out <- nn_compute(
            data, data, as.integer(graph_k), backend,
            points_missing = TRUE, exclude_self = TRUE,
            n_threads = n_threads, metric = metric
        )
        attr(out, "resolved_seed_backend") <- paste0(
            "native_", backend, "_seed"
        )
        return(out)
    }
    out <- nn_cpp(
        data, data, as.integer(graph_k), metric, FALSE, FALSE, 0, TRUE,
        as.integer(normalize_nn_threads(n_threads)), TRUE
    )
    attr(out, "resolved_seed_backend") <- if (
        identical(metric, "inner_product") && isTRUE(use_cuda)
    ) "native_cpu_inner_product_seed" else "native_cpu_exact_seed"
    out
}

candidate_graph_seed_backend <- function(data, metric, use_cuda) {
    if (isTRUE(use_cuda) && identical(metric, "euclidean")) {
        if (isTRUE(cuvs_available())) return("cuda_cuvs_bruteforce")
        if (isTRUE(faiss_gpu_available())) return("faiss_gpu_flat_l2")
    }
    if (is_float32_matrix_input(data)) {
        if (identical(metric, "inner_product")) "faiss_flat_ip" else {
            "faiss_flat_l2"
        }
    } else NULL
}

refine_candidate_graph <- function(
    data, graph, k, metric, use_cuda, n_threads
) {
    if (isTRUE(use_cuda)) {
        fun <- if (is_float32_matrix_input(data)) {
            row_candidate_knn_cuda_float32_cpp
        } else row_candidate_knn_cuda_cpp
        out <- fun(data, graph, as.integer(k), metric)
        attr(out, "cuda_kernel") <- "row_candidate_knn"
        return(out)
    }
    fun <- if (is_float32_matrix_input(data)) {
        candidate_knn_float32_cpp
    } else candidate_knn_cpp
    fun(
        data, data, graph, as.integer(k), metric, FALSE, TRUE, TRUE,
        as.integer(normalize_nn_threads(n_threads))
    )
}

native_nsg_self_knn <- function(
    data,
    k,
    r,
    graph_k,
    metric = "euclidean",
    use_cuda = FALSE,
    n_threads = NULL,
    seed_backend = "exact"
) {
    metric <- normalize_nn_metric(metric)
    if (!metric %in% c("euclidean", "inner_product")) {
        stop(
            "Native NSG candidate refinement supports ",
            "Euclidean or inner-product scoring.",
            call. = FALSE
        )
    }
    if (nrow(data) < 2L) {
        stop("Native NSG requires at least two rows.", call. = FALSE)
    }
    graph_k_cap <- if (isTRUE(use_cuda)) 255L else 512L
    graph_k <- as.integer(min(max(k, graph_k), nrow(data) - 1L, graph_k_cap))
    r <- as.integer(min(max(k, r), graph_k))
    seed <- candidate_graph_seed(
        data, graph_k, metric, use_cuda, n_threads, seed_backend
    )
    seed_backend <- attr(seed, "resolved_seed_backend", exact = TRUE)
    graph <- nsg_prune_candidate_graph(
        data,
        seed$indices,
        r = r,
        metric = metric,
        protect_top = k
    )
    out <- refine_candidate_graph(data, graph, k, metric, use_cuda, n_threads)
    attr(out, "approximation") <- nsg_result_metadata(
        graph, seed_backend, graph_k
    )
    out
}

nsg_result_metadata <- function(graph, seed_backend, graph_k) {
    list(
        seed_backend = seed_backend,
        candidate_columns = as.integer(ncol(graph)),
        candidate_layout = attr(graph, "candidate_layout", exact = TRUE) %||%
            NA_character_,
        pruning_rule = attr(graph, "pruning_rule", exact = TRUE) %||%
            NA_character_,
        seed_graph_k = as.integer(graph_k),
        protected_seed_neighbors = as.integer(
            attr(graph, "protected_top", exact = TRUE) %||% 0L
        ),
        exact_mrng_prune = isTRUE(attr(graph, "exact_prune", exact = TRUE))
    )
}

knn_recall_subset_size <- function(n) {
    value <- faissr_option("approx_recall_sample", 512L)
    value <- faissr_quiet_warning(as.integer(value))
    if (
        length(value) != 1L || is.na(value) || !is.finite(value) || value < 1L
    ) {
        value <- 512L
    }
    as.integer(min(n, value))
}

attach_knn_recall_subset <- function(result, data, k, exclude_self, seed) {
    n <- nrow(data)
    compare_k <- if (isTRUE(exclude_self)) k else k - 1L
    if (compare_k < 1L || n < 2L) {
        attr(result, "recall") <- data.frame(
            k = compare_k,
            recall_at_k = NA_real_,
            median_recall_at_k = NA_real_,
            min_recall_at_k = NA_real_,
            sample_size = 0L,
            stringsAsFactors = FALSE
        )
        return(result)
    }
    sample_size <- knn_recall_subset_size(n)
    rows <- with_rng_seed(
        as.integer(seed) + 1009L,
        sort(sample.int(n, sample_size))
    )
    exact_raw <- nn_compute(
        data,
        data[rows, , drop = FALSE],
        k = min(n, compare_k + 1L),
        backend = "cpu",
        points_missing = FALSE,
        exclude_self = FALSE
    )
    exact_idx <- matrix(0L, nrow = sample_size, ncol = compare_k)
    for (i in seq_along(rows)) {
        keep <- exact_raw$indices[i, ] != rows[i]
        row_idx <- exact_raw$indices[i, keep]
        if (length(row_idx) < compare_k) {
            row_idx <- exact_raw$indices[i, seq_len(ncol(exact_raw$indices))]
        }
        exact_idx[i, ] <- row_idx[seq_len(compare_k)]
    }
    approx_idx <- if (isTRUE(exclude_self)) {
        result$indices[rows, seq_len(compare_k), drop = FALSE]
    } else {
        result$indices[rows, 1L + seq_len(compare_k), drop = FALSE]
    }
    recall <- .knn_recall_summary(
        list(indices = approx_idx),
        list(indices = exact_idx),
        k = compare_k
    )
    recall$sample_size <- as.integer(sample_size)
    attr(result, "recall") <- recall
    result
}

should_use_nndescent_self_knn <- function(
    backend,
    self_query,
    n,
    p,
    k,
    exclude_self,
    work_size
) {
    if (!identical(backend, "cpu_nndescent")) {
        return(FALSE)
    }
    if (!isTRUE(self_query)) {
        return(FALSE)
    }
    if (n < 10000L || k < 10L || p < 2L) {
        return(FALSE)
    }
    if (work_size < 5e8) {
        return(FALSE)
    }
    nonself_k <- if (isTRUE(exclude_self)) k else k - 1L
    nonself_k >= 1L
}

cpu_nndescent_prefer_faiss <- function() {
    isTRUE(faissr_option("cpu_nndescent_prefer_faiss", FALSE)) &&
        isTRUE(faiss_available())
}

cpu_nndescent_faiss_index <- function() {
    value <- tolower(as.character(faissr_option(
        "cpu_nndescent_faiss_index",
        "hnsw"
    ))[1L])
    if (!value %in% c("hnsw", "ivf", "flat", "nsg", "nndescent")) {
        warning(
            "Option `faissR.cpu_nndescent_faiss_index` must be one of ",
            "\"hnsw\", \"ivf\", \"flat\", \"nsg\", or ",
            "\"nndescent\"; using \"hnsw\".",
            call. = FALSE
        )
        value <- "hnsw"
    }
    value
}

faiss_self_knn <- function(
    data, k, backend = "faiss_ivf", exact = FALSE, seed = 4L,
    metric = "euclidean", target_recall = 0.99, n_threads = NULL
) {
    n <- nrow(data)
    k <- as.integer(k)
    if (length(k) != 1L || is.na(k) || !is.finite(k) || k < 1L || k >= n) {
        stop("`k` must be in [1, nrow(data) - 1].", call. = FALSE)
    }
    if (!isTRUE(faiss_available())) {
        stop(
            "The real FAISS C++ backend is not available in this build. ",
            "Reinstall faissR with `FAISS_HOME` pointing ",
            "to a FAISS installation.",
            call. = FALSE
        )
    }
    n_threads <- normalize_nn_threads(n_threads)
    metric <- normalize_nn_metric(metric)
    target_recall <- normalize_hnsw_target_recall(target_recall)
    kind <- faiss_self_backend_kind(backend, exact)
    switch(
        kind,
        flat = faiss_flat_self_result(data, k, metric, n_threads, seed),
        hnsw = faiss_hnsw_self_result(
            data, k, metric, n_threads, seed, backend, target_recall
        ),
        nsg = faiss_nsg_self_result(data, k, metric, n_threads, seed, backend),
        nndescent = faiss_nndescent_self_result(
            data, k, metric, n_threads, seed, backend
        ),
        faiss_ivf_self_result(data, k, metric, n_threads, seed, backend)
    )
}

faiss_self_backend_kind <- function(backend, exact) {
    if (isTRUE(exact) || backend %in% c(
        "faiss_flat", "cpu_nndescent_faiss_flat"
    )) return("flat")
    if (backend %in% c(
        "faiss_hnsw", "cpu_nndescent_faiss_hnsw"
    )) return("hnsw")
    if (backend %in% c("faiss_nsg", "cpu_nndescent_faiss_nsg")) return("nsg")
    if (backend %in% c(
        "faiss_nndescent", "cpu_nndescent_faiss_nndescent"
    )) return("nndescent")
    "ivf"
}

faiss_flat_self_result <- function(data, k, metric, n_threads, seed) {
    fun <- if (identical(metric, "inner_product")) {
        nn_faiss_flat_ip_cpp
    } else nn_faiss_flat_cpp
    out <- fun(data, data, as.integer(k), TRUE, as.integer(n_threads))
    attr(out, "approximation") <- list(
        strategy = if (identical(metric, "inner_product")) {
            "faiss_IndexFlatIP_self"
        } else "faiss_IndexFlatL2_self",
        backend = "faiss", library = "faiss", exact = TRUE,
        metric = metric, seed = as.integer(seed)
    )
    out
}

faiss_hnsw_self_result <- function(
    data, k, metric, n_threads, seed, backend, target_recall
) {
    params <- faiss_hnsw_params(
        k, nrow(data), ncol(data), metric, target_recall
    )
    out <- nn_faiss_hnsw_cpp(
        data, data, as.integer(k), as.integer(params$m),
        as.integer(params$ef_construction), as.integer(params$ef_search),
        faiss_metric_search_arg(metric),
            faiss_metric_distance_output_arg(metric),
        TRUE, as.integer(n_threads)
    )
    attr(out, "approximation") <- faiss_hnsw_self_metadata(
        out, params, backend, metric, seed, target_recall
    )
    out
}

faiss_hnsw_self_metadata <- function(
    out, params, backend, metric, seed, target_recall
) {
    metadata <- list(
        strategy = "faiss_IndexHNSWFlat_self", backend = backend,
        library = "faiss", exact = FALSE, metric = metric,
        m = as.integer(out$m),
            ef_construction = as.integer(out$ef_construction),
        ef_search = as.integer(out$ef_search),
        requested_m = as.integer(out$requested_m),
        requested_ef_construction = as.integer(out$requested_ef_construction),
        requested_ef_search = as.integer(out$requested_ef_search),
        hnsw_parameters_adjusted = isTRUE(out$hnsw_parameters_adjusted),
        seed = as.integer(seed)
    )
    c(metadata, hnsw_tuning_metadata(params, target_recall))
}

hnsw_tuning_metadata <- function(params, target_recall) {
    list(
        tuning_policy = params$policy, tuning_rule = params$rule,
        target_recall = as.numeric(params$target_recall %||% target_recall),
        tuning_low_dim = isTRUE(params$low_dim),
        tuning_high_dim = isTRUE(params$high_dim),
        tuning_large_n = isTRUE(params$large_n),
        tuning_small_k = isTRUE(params$small_k),
        tuning_large_k = isTRUE(params$large_k),
        tuning_non_euclidean = isTRUE(params$non_euclidean),
        tuning_shape_group = params$tuning_shape_group %||%
            params$shape_group %||% NA_character_,
        tuning_k_bucket = as.integer(
            params$tuning_k_bucket %||% params$k_bucket %||% NA_integer_
        ),
        tuning_target_recall_code = as.integer(
            params$tuning_target_recall_code %||%
                params$target_recall_code %||% NA_integer_
        ),
        tuning_benchmark_basis = params$tuning_benchmark_basis %||%
            params$benchmark_basis %||% NA_character_,
        tuning_benchmark_target_met = isTRUE(
            params$tuning_benchmark_target_met),

        tuning_benchmark_source = params$tuning_benchmark_source %||%
            params$benchmark_source %||% NA_character_,
        tuning_source = params$tuning_source %||% "cpp"
    )
}

validate_faiss_nsg_self_metric <- function(metric) {
    if (!identical(metric, "euclidean")) {
        stop(
            "`backend = \"faiss_nsg\"` currently supports only ",
            "`metric = \"euclidean\"`. FAISS NSG graph construction can ",
            "abort the R process for normalized or inner-product routes ",
            "in this linked FAISS build.", call. = FALSE
        )
    }
}

faiss_nsg_self_result <- function(data, k, metric, n_threads, seed, backend) {
    validate_faiss_nsg_self_metric(metric)
    params <- faiss_nsg_params(k)
    out <- nn_faiss_nsg_cpp(
        data, data, as.integer(k), as.integer(params$r),
        as.integer(params$search_l), as.integer(params$build_type),
        faiss_metric_search_arg(metric),
            faiss_metric_distance_output_arg(metric),
        TRUE, as.integer(n_threads)
    )
    attr(out, "approximation") <- list(
        strategy = "faiss_IndexNSGFlat_self", backend = backend,
        library = "faiss", exact = FALSE, r = as.integer(out$r),
        search_l = as.integer(out$search_l),
            build_type = as.integer(out$build_type),
        gk = as.integer(out$gk), requested_r = as.integer(out$requested_r),
        requested_search_l = as.integer(out$requested_search_l),
        requested_build_type = as.integer(out$requested_build_type),
        nsg_parameters_adjusted = isTRUE(out$nsg_parameters_adjusted),
        seed = as.integer(seed)
    )
    out
}

validate_faiss_nndescent_self <- function(metric) {
    if (!identical(metric, "euclidean")) {
        stop(
            "`backend = \"faiss_nndescent\"` is validated only for ",
            "`metric = \"euclidean\"` in this FAISS build.", call. = FALSE
        )
    }
    if (!isTRUE(faissr_option("enable_faiss_nndescent", FALSE))) {
        stop(
            "FAISS NNDescent is disabled by default because linked FAISS ",
            "builds can abort R during graph construction.", call. = FALSE
        )
    }
}

faiss_nndescent_self_result <- function(
    data, k, metric, n_threads, seed, backend
) {
    validate_faiss_nndescent_self(metric)
    params <- faiss_nndescent_params(k)
    out <- nn_faiss_nndescent_cpp(
        data, data, as.integer(k), as.integer(params$graph_k),
        as.integer(params$n_iter), as.integer(params$search_l),
        "euclidean", "euclidean", TRUE, as.integer(n_threads)
    )
    attr(out, "approximation") <- list(
        strategy = "faiss_IndexNNDescentFlat_self", backend = backend,
        library = "faiss", exact = FALSE, graph_k = as.integer(out$graph_k),
        n_iter = as.integer(out$n_iter), search_l = as.integer(out$search_l),
        requested_graph_k = as.integer(out$requested_graph_k),
        requested_n_iter = as.integer(out$requested_n_iter),
        requested_search_l = as.integer(out$requested_search_l),
        nndescent_parameters_adjusted = isTRUE(
            out$nndescent_parameters_adjusted),

        seed = as.integer(seed)
    )
    out
}

faiss_ivf_self_result <- function(data, k, metric, n_threads, seed, backend) {
    params <- faiss_ivf_params(nrow(data), k, metric, ncol(data))
    out <- nn_faiss_ivf_cpp(
        data, data, as.integer(k), as.integer(params$nlist),
        as.integer(params$nprobe), "euclidean", "euclidean", TRUE,
        as.integer(n_threads)
    )
    attr(out, "approximation") <- list(
        strategy = "faiss_IndexIVFFlat_self",
        backend = backend,
        library = "faiss",
        exact = FALSE,
        metric = metric,
        nlist = as.integer(out$nlist),
        nprobe = as.integer(out$nprobe),
        seed = as.integer(seed)
    )
    out
}

nndescent_pool_size <- function(
    n,
    k,
    p = NA_integer_,
    metric = "euclidean",
    target_recall = 0.99
) {
    as.integer(
        cpu_nndescent_params(
            n,
            k,
            p = p,
            metric = metric,
            target_recall = target_recall
        )$pool_size
    )
}

nndescent_iterations <- function(
    n,
    k,
    p = NA_integer_,
    metric = "euclidean",
    target_recall = 0.99
) {
    as.integer(
        cpu_nndescent_params(
            n,
            k,
            p = p,
            metric = metric,
            target_recall = target_recall
        )$n_iters
    )
}

cpu_nndescent_params <- function(
    n,
    k,
    p = NA_integer_,
    metric = "euclidean",
    target_recall = 0.99
) {
    n <- as.integer(n)
    k <- as.integer(k)
    p <- faissr_quiet_warning(as.integer(p))
    metric <- normalize_nn_metric(metric)
    target_recall <- normalize_hnsw_target_recall(target_recall)
    tune <- nn_tune_cpu_nndescent_cpp(
        n,
        p,
        k,
        metric,
        as.numeric(target_recall)
    )
    n_cap <- max(1L, n - 1L)
    requested <- cpu_nndescent_requested_params()
    tune <- apply_cpu_nndescent_requests(tune, requested, n_cap, k)
    if (isTRUE(requested$manual)) {
        tune$tuning_policy <- "manual_options"
        tune$tuning_rule <- "manual_cpu_nndescent"
        tune$tuning_benchmark_target_met <- FALSE
    }
    record_cpu_nndescent_requests(tune, requested)
}

cpu_nndescent_requested_params <- function() {
    names <- c(
        "cpu_nndescent_pool_size",
        "cpu_nndescent_n_iters",
        "cpu_nndescent_max_candidates",
        "cpu_nndescent_n_random_projections"
    )
    list(
        manual = nn_any_options(names),
        pool_size = nn_option_int_or_na(names[[1L]]),
        n_iters = nn_option_int_or_na(names[[2L]]),
        max_candidates = nn_option_int_or_na(names[[3L]]),
        n_random_projections = nn_option_int_or_na(names[[4L]])
    )
}

apply_cpu_nndescent_requests <- function(tune, requested, n_cap, k) {
    if (!is.na(requested$pool_size)) {
        tune$pool_size <- as.integer(max(k, min(n_cap, requested$pool_size)))
    }
    if (!is.na(requested$n_iters)) {
        tune$n_iters <- as.integer(max(0L, min(100L, requested$n_iters)))
    }
    if (!is.na(requested$max_candidates)) {
        tune$max_candidates <- as.integer(max(
            tune$pool_size, min(n_cap, requested$max_candidates)
        ))
    }
    if (!is.na(requested$n_random_projections)) {
        tune$n_random_projections <- as.integer(max(
            1L, min(256L, requested$n_random_projections)
        ))
    }
    tune
}

record_cpu_nndescent_requests <- function(tune, requested) {
    for (name in c("pool_size", "n_iters", "max_candidates",
        "n_random_projections")) {
        value <- requested[[name]]
        tune[[paste0("requested_", name)]] <- as.integer(
            if (is.na(value)) tune[[name]] else value
        )
    }
    tune
}

nndescent_self_knn <- function(
    data, k, seed = 4L, n_threads = NULL, metric = "euclidean",
    tuning_metric = metric, target_recall = 0.99
) {
    args <- prepare_nndescent_self_args(
        data, k, metric, tuning_metric, target_recall, n_threads
    )
    k <- args$k
    metric <- args$metric
    n_threads <- args$n_threads
    target_recall <- args$target_recall
    if (!metric %in% c("euclidean", "inner_product")) {
        stop(
            "Native CPU NN-descent expects raw euclidean ",
            "or inner_product input. ",
            "Cosine and correlation are normalized before ",
            "this helper is called.",
            call. = FALSE
        )
    }
    if (cpu_nndescent_prefer_faiss()) {
        faiss_index <- cpu_nndescent_faiss_index()
        return(faiss_self_knn(
            data,
            k = k,
            backend = paste0("cpu_nndescent_faiss_", faiss_index),
            exact = identical(faiss_index, "flat"),
            seed = seed,
            n_threads = n_threads,
            metric = metric
        ))
    }
    tune <- cpu_nndescent_params(
        args$n,
        k,
        p = ncol(data),
        metric = args$tuning_metric,
        target_recall = target_recall
    )
    out <- execute_nndescent_self(data, k, tune, seed, n_threads, metric)
    params <- nndescent_result_metadata(
        out, tune, metric, args$tuning_metric, target_recall
    )
    attr(out, "nndescent") <- params
    attr(out, "approximation") <- params
    out
}

prepare_nndescent_self_args <- function(
    data, k, metric, tuning_metric, target_recall, n_threads
) {
    n <- nrow(data)
    k <- as.integer(k)
    if (length(k) != 1L || is.na(k) || !is.finite(k) || k < 1L || k >= n) {
        stop("`k` must be in [1, nrow(data) - 1].", call. = FALSE)
    }
    if (is.null(n_threads)) {
        n_threads <- faissr_quiet_warning(parallel::detectCores(
            logical = FALSE))
        if (length(n_threads) != 1L || is.na(n_threads) || !is.finite(
            n_threads)) {
            n_threads <- 1L
        }
    }
    list(
        n = n, k = k, n_threads = n_threads,
        metric = normalize_nn_metric(metric),
        tuning_metric = normalize_nn_metric(tuning_metric),
        target_recall = normalize_hnsw_target_recall(target_recall)
    )
}

execute_nndescent_self <- function(data, k, tune, seed, n_threads, metric) {
    pool_size <- as.integer(tune$pool_size)
    n_iters <- as.integer(tune$n_iters)
    max_candidates <- as.integer(tune$max_candidates)
    n_random_projections <- as.integer(tune$n_random_projections)
    if (is_float32_matrix_input(data)) {
        nndescent_self_knn_float32_cpp(
            data,
            as.integer(k),
            as.integer(pool_size),
            as.integer(n_iters),
            as.integer(max_candidates),
            as.integer(n_random_projections),
            as.integer(seed),
            TRUE,
            as.integer(max(1L, min(8L, n_threads))),
            metric
        )
    } else {
        nndescent_self_knn_cpp(
            data,
            as.integer(k),
            as.integer(pool_size),
            as.integer(n_iters),
            as.integer(max_candidates),
            as.integer(n_random_projections),
            as.integer(seed),
            TRUE,
            as.integer(max(1L, min(8L, n_threads))),
            metric
        )
    }
}

nndescent_result_metadata <- function(
    out, tune, metric, tuning_metric, target_recall
) {
    list(
        strategy = "native_cpu_nndescent",
        backend = "cpu",
        pool_size = as.integer(tune$pool_size),
        n_iters = as.integer(tune$n_iters),
        max_candidates = as.integer(tune$max_candidates),
        n_random_projections = as.integer(tune$n_random_projections),
        requested_pool_size = as.integer(
            tune$requested_pool_size %||% tune$pool_size
        ),
        requested_n_iters = as.integer(tune$requested_n_iters %||%
            tune$n_iters),

        requested_max_candidates = as.integer(
            tune$requested_max_candidates %||% tune$max_candidates
        ),
        requested_n_random_projections = as.integer(
            tune$requested_n_random_projections %||% tune$n_random_projections
        ),
        reverse_candidates = "rank_ordered",
        seed_initialization = out$seed_initialization %||%
            "random_projection_window_plus_row_fill",
        candidate_layout = out$candidate_layout %||% "flat_row_major_graph",
        reverse_storage = out$reverse_storage %||% "flat_fixed_width",
        distance_snapshot_copy = isTRUE(out$distance_snapshot_copy %||% FALSE),
        metric = metric,
        tuning_metric = tune$tuning_metric %||% tuning_metric,
        target_recall = as.numeric(tune$target_recall %||% target_recall),
        requested_target_recall = as.numeric(
            tune$requested_target_recall %||% target_recall
        ),
        tuning_policy = tune$tuning_policy,
        tuning_rule = tune$tuning_rule,
        tuning_shape_group = tune$tuning_shape_group %||% NA_character_,
        tuning_k_bucket = as.integer(tune$tuning_k_bucket %||% NA_integer_),
        tuning_target_recall_code = as.integer(
            tune$tuning_target_recall_code %||% NA_integer_
        ),
        tuning_benchmark_basis = tune$tuning_benchmark_basis %||% NA_character_,
        tuning_benchmark_target_met = isTRUE(tune$tuning_benchmark_target_met),
        tuning_benchmark_source = tune$tuning_benchmark_source %||%
            NA_character_,
        tuning_large_n = isTRUE(tune$tuning_large_n),
        tuning_small_k = isTRUE(tune$tuning_small_k),
        tuning_source = tune$tuning_source %||% "cpp"
    )
}

clustered_knn_center_count <- function(n, k) {
    n <- as.integer(n)
    k <- as.integer(k)
    max_centers <- max(2L, n - 1L)
    count <- max(16L, ceiling(sqrt(n) * 2))
    count <- max(count, ceiling(n / max(250L, 10L * k)))
    as.integer(min(max_centers, count))
}

clustered_knn_projection_columns <- function(projection_k, k) {
    projection_k <- as.integer(max(1L, projection_k))
    k <- as.integer(max(1L, k))
    bucket_cols <- min(projection_k, max(2L, min(4L, ceiling(k / 10))))
    query_cols <- min(projection_k, max(bucket_cols, min(8L, ceiling(k / 5))))
    list(bucket_cols = bucket_cols, query_cols = query_cols)
}

clustered_self_knn <- function(
    data,
    k,
    exclude_self = TRUE,
    seed = 4L,
    n_threads = NULL
) {
    n <- nrow(data)
    k <- validate_clustered_knn_k(n, k)
    nonself_k <- if (isTRUE(exclude_self)) k else k - 1L
    if (nonself_k < 1L) {
        return(list(
            indices = matrix(seq_len(n), n, 1L),
            distances = matrix(0, n, 1L)
        ))
    }
    if (nonself_k >= n) {
        stop(
            "`k` cannot be larger than the available neighbor count.",
            call. = FALSE
        )
    }

    projection <- clustered_knn_projection(data, nonself_k, seed, n_threads)
    cols <- clustered_knn_projection_columns(
        ncol(projection$indices),
        nonself_k
    )
    n_threads <- normalize_nn_threads(n_threads)
    out <- landmark_candidate_knn_cpp(
        data,
        projection$indices,
        as.integer(nonself_k),
        as.integer(cols$bucket_cols),
        as.integer(cols$query_cols),
        TRUE,
        as.integer(max(1L, min(8L, n_threads)))
    )
    storage.mode(out$indices) <- "integer"
    if (!identical(typeof(out$distances), "double")) {
        storage.mode(out$distances) <- "double"
    }

    if (!isTRUE(exclude_self)) {
        out <- prepend_self_neighbor_column(out)
    }
    out
}

validate_clustered_knn_k <- function(n, k) {
    if (n < 2L) stop("`data` must have at least two rows.", call. = FALSE)
    k <- as.integer(k)
    if (length(k) != 1L || is.na(k) || !is.finite(k) || k < 1L) {
        stop("`k` must be a positive integer.", call. = FALSE)
    }
    k
}

clustered_knn_projection <- function(data, k, seed, n_threads) {
    n_centers <- clustered_knn_center_count(nrow(data), k)
    centers <- select_landmark_rows(data, n_centers, seed)
    projection_k <- min(n_centers, max(2L, min(12L, ceiling(k / 2))))
    nn_compute(
        data[centers, , drop = FALSE], data, k = projection_k,
        backend = "cpu", points_missing = FALSE, exclude_self = FALSE,
        n_threads = n_threads
    )
}

nn_option_int_or_na <- function(name) {
    value <- faissr_option(name, NULL)
    if (is.null(value)) {
        return(NA_integer_)
    }
    value <- faissr_quiet_warning(as.integer(value))
    if (length(value) != 1L || is.na(value) || !is.finite(value)) {
        return(NA_integer_)
    }
    as.integer(value)
}

nn_option_double_or_na <- function(name) {
    value <- faissr_option(name, NULL)
    if (is.null(value)) {
        return(NA_real_)
    }
    value <- faissr_quiet_warning(as.numeric(value))
    if (length(value) != 1L || is.na(value) || !is.finite(value)) {
        return(NA_real_)
    }
    as.numeric(value)
}

nn_any_options <- function(names) {
    any(vapply(
        names,
        function(name) !is.null(faissr_option(name, NULL)),
        logical(1)
    ))
}

ivf_list_count <- function(n, k) {
    n <- as.integer(n)
    k <- as.integer(k)
    count <- max(16L, ceiling(sqrt(n)))
    count <- min(count, ceiling(n / max(50L, 20L * k)))
    as.integer(max(4L, min(n, count, 1024L)))
}

ivf_probe_count <- function(nlist, k, metric = "euclidean") {
    nlist <- as.integer(nlist)
    k <- as.integer(k)
    metric <- normalize_nn_metric(metric)
    base <- max(16L, ceiling(sqrt(nlist)), ceiling(k / 3))
    if (!identical(metric, "euclidean")) {
        base <- max(base, ceiling(1.5 * base), ceiling(k / 2))
    }
    as.integer(max(1L, min(nlist, base)))
}

faiss_ivf_params <- function(
    n,
    k,
    metric = "euclidean",
    p = NA_integer_,
    backend = "cpu",
    method = "ivf",
    target_recall = 0.99
) {
    metric <- normalize_nn_metric(metric)
    target_recall <- normalize_hnsw_target_recall(target_recall)
    nn_tune_faiss_ivf_cpp(
        as.integer(n),
        faissr_quiet_warning(as.integer(p)),
        as.integer(k),
        metric,
        as.numeric(target_recall),
        as.character(backend)[1L],
        as.character(method)[1L],
        nn_option_int_or_na(c("faiss_nlist", "ivf_nlist")),
        nn_option_int_or_na(c("faiss_nprobe", "ivf_nprobe")),
        faiss_ivf_manual_params()
    )
}

cuda_ivf_params <- function(
    n,
    p,
    k,
    metric = "euclidean",
    target_recall = 0.99
) {
    metric <- normalize_nn_metric(metric)
    target_recall <- normalize_hnsw_target_recall(target_recall)
    nn_tune_cuda_ivf_cpp(
        as.integer(n),
        as.integer(p),
        as.integer(k),
        metric,
        as.numeric(target_recall),
        nn_option_int_or_na(c(
            "cuda_ivf_nlist",
            "cuvs_ivf_nlist",
            "faiss_gpu_ivf_nlist",
            "faiss_nlist",
            "ivf_nlist"
        )),
        nn_option_int_or_na(c(
            "cuda_ivf_nprobe",
            "cuvs_ivf_nprobe",
            "faiss_gpu_ivf_nprobe",
            "faiss_nprobe",
            "ivf_nprobe"
        )),
        cuda_ivf_manual_params()
    )
}

faiss_ivf_manual_params <- function() {
    any(vapply(
        c("faiss_nlist", "ivf_nlist", "faiss_nprobe", "ivf_nprobe"),
        function(name) !is.null(faissr_option(name, NULL)),
        logical(1)
    ))
}

cuda_ivf_manual_params <- function() {
    nn_any_options(c(
        "cuda_ivf_nlist",
        "cuvs_ivf_nlist",
        "faiss_gpu_ivf_nlist",
        "faiss_nlist",
        "ivf_nlist",
        "cuda_ivf_nprobe",
        "cuvs_ivf_nprobe",
        "faiss_gpu_ivf_nprobe",
        "faiss_nprobe",
        "ivf_nprobe"
    ))
}

cuvs_ivfpq_params <- function(p, n = NULL) {
    nn_tune_cuvs_ivfpq_cpp(
        as.integer(p),
        faissr_quiet_warning(as.integer(n %||% NA_integer_)),
        nn_option_int_or_na(c("cuvs_ivfpq_pq_dim", "ivfpq_pq_dim")),
        nn_option_int_or_na(c("cuvs_ivfpq_pq_bits", "ivfpq_pq_bits")),
        nn_any_options(c(
            "cuvs_ivfpq_pq_dim",
            "ivfpq_pq_dim",
            "cuvs_ivfpq_pq_bits",
            "ivfpq_pq_bits"
        ))
    )
}

cuvs_ivfpq_align_params <- function(pq, p) {
    p <- as.integer(max(1L, p))
    pq_dim <- faissr_quiet_warning(as.integer(pq$pq_dim %||% 0L))
    pq_bits <- faissr_quiet_warning(as.integer(pq$pq_bits %||% 8L))
    if (length(pq_dim) != 1L || is.na(pq_dim)) {
        pq_dim <- 0L
    }
    if (length(pq_bits) != 1L || is.na(pq_bits)) {
        pq_bits <- 8L
    }
    original_dim <- pq_dim
    original_bits <- pq_bits
    pq_dim <- as.integer(max(0L, min(p, pq_dim)))
    pq_bits <- as.integer(max(4L, min(8L, pq_bits)))
    rule <- "byte_aligned"
    if (!cuvs_pq_byte_aligned(pq_dim, pq_bits, p)) {
        aligned <- align_cuvs_pq_values(pq_dim, pq_bits, p)
        pq_dim <- aligned$dim
        pq_bits <- aligned$bits
        rule <- aligned$rule
    }
    adjusted <- !identical(as.integer(original_dim), as.integer(pq_dim)) ||
        !identical(as.integer(original_bits), as.integer(pq_bits))
    pq$pq_dim <- as.integer(pq_dim)
    pq$pq_bits <- as.integer(pq_bits)
    pq$pq_alignment_adjusted <- isTRUE(adjusted)
    pq$pq_alignment_rule <- rule
    if (
        isTRUE(adjusted) && !grepl(rule, pq$tuning_rule %||% "", fixed = TRUE)
    ) {
        pq$tuning_rule <- paste0(pq$tuning_rule %||% "cuvs_ivfpq", "_", rule)
    }
    pq
}

cuvs_pq_byte_aligned <- function(dim, bits, p) {
    effective_dim <- if (dim > 0L) dim else p
    (bits * effective_dim) %% 8L == 0L
}

integer_gcd <- function(a, b) {
    a <- abs(as.integer(a))
    b <- abs(as.integer(b))
    while (b != 0L) {
        remainder <- a %% b
        a <- b
        b <- remainder
    }
    if (a == 0L) 1L else a
}

align_cuvs_pq_values <- function(dim, bits, p) {
    step <- as.integer(8L / integer_gcd(bits, 8L))
    effective <- as.integer(max(1L, min(p, if (dim > 0L) dim else p)))
    reduced <- as.integer(effective - effective %% step)
    if (reduced >= 1L) {
        return(list(
            dim = reduced, bits = bits,
            rule = "pq_dim_reduced_for_byte_alignment"
        ))
    }
    list(
        dim = effective, bits = 8L,
        rule = "pq_bits_promoted_for_byte_alignment"
    )
}

ivfpq_fastscan_option_int <- function(
    name,
    default,
    min_value = 1L,
    max_value = .Machine$integer.max
) {
    value <- faissr_option(name, NULL)
    value <- if (is.null(value)) {
        default
    } else {
        faissr_quiet_warning(as.integer(value))
    }
    if (length(value) != 1L || is.na(value) || !is.finite(value)) {
        value <- default
    }
    as.integer(max(min_value, min(max_value, value)))
}

ivfpq_fastscan_default_int <- function(value, default) {
    value <- faissr_quiet_warning(as.integer(value %||% default))
    if (length(value) != 1L || is.na(value) || !is.finite(value)) {
        value <- default
    }
    as.integer(value)
}

ivfpq_fastscan_cpu_params <- function(
    n,
    p,
    k,
    target_recall = 0.99,
    metric = "euclidean"
) {
    metric <- normalize_nn_metric(metric)
    if (!metric %in% c("euclidean", "cosine", "correlation", "inner_product")) {
        stop(
            "CPU IVFPQ FastScan tuning currently supports ",
            "Euclidean, cosine, correlation, and ",
            "inner-product metrics.",
            call. = FALSE
        )
    }
    ivf <- faiss_ivf_params(
        n,
        k,
        metric = metric,
        p = p,
        method = "ivfpq_fastscan",
        target_recall = target_recall
    )
    pq <- configure_cpu_fastscan_pq(
        faiss_ivfpq_pq_params(p, n = n, ivf_params = ivf), ivf, p
    )
    default_refine_factor <- ivfpq_fastscan_default_int(
        ivf$ivfpq_fastscan_refine_factor, 8L
    )
    default_bbs <- ivfpq_fastscan_default_int(ivf$ivfpq_fastscan_bbs, 32L)
    list(
        ivf = ivf,
        pq = pq,
        refine_factor = ivfpq_fastscan_option_int(
            "ivfpq_fastscan_refine_factor", default_refine_factor,
            min_value = 1L, max_value = 128L
        ),
        bbs = ivfpq_fastscan_option_int(
            "ivfpq_fastscan_bbs", default_bbs,
            min_value = 32L, max_value = 4096L
        ),
        tuning = ivfpq_fastscan_tuning_metadata(
            ivf, "cpu", metric, default_rule = "hpc_cpu_ivfpq_fastscan"
        )
    )
}

configure_cpu_fastscan_pq <- function(pq, ivf, p) {
    pq$m <- ivfpq_fastscan_option_int(
        "ivfpq_fastscan_pq_m",
        pq$m,
        min_value = 1L,
        max_value = as.integer(p)
    )
    pq$requested_m <- as.integer(pq$m)
    pq$nbits <- 4L
    pq$requested_nbits <- 4L
    pq$tuning_policy <- "auto_shape_k_target_recall_ivfpq_fastscan"
    pq$tuning_rule <- ivf$tuning_rule %||% "hpc_cpu_ivfpq_fastscan"
    pq
}

ivfpq_fastscan_tuning_metadata <- function(
    ivf, backend, metric, batch_size = NULL,
    default_rule = paste0(backend, "_ivfpq_fastscan")
) {
    out <- list(
        tuning_policy = "auto_shape_k_target_recall_ivfpq_fastscan",
        tuning_rule = ivf$tuning_rule %||% default_rule,
        tuning_metric = ivf$tuning_metric %||% metric,
        tuning_backend = ivf$tuning_backend %||% backend,
        tuning_method = ivf$tuning_method %||% "ivfpq_fastscan",
        tuning_shape_group = ivf$tuning_shape_group %||% NA_character_,
        tuning_k_bucket = ivf$tuning_k_bucket %||% NA_integer_,
        tuning_target_recall_code = ivf$tuning_target_recall_code %||%
            NA_integer_,

        tuning_benchmark_basis = ivf$tuning_benchmark_basis %||% NA_character_,
        tuning_benchmark_target_met = ivf$tuning_benchmark_target_met %||%
            FALSE,

        tuning_benchmark_source = ivf$tuning_benchmark_source %||%
            NA_character_,

        tuning_source = ivf$tuning_source %||% "cpp",
        ivfpq_fastscan = TRUE
    )
    if (!is.null(batch_size)) out$cuvs_ivf_batch_size <- batch_size
    out
}

ivfpq_fastscan_cuda_params <- function(
    n,
    p,
    k,
    target_recall = 0.99,
    metric = "euclidean"
) {
    metric <- normalize_nn_metric(metric)
    if (!metric %in% c("euclidean", "cosine", "correlation", "inner_product")) {
        stop(
            "CUDA IVFPQ FastScan tuning currently supports ",
            "Euclidean, cosine, correlation, and ",
            "inner-product metrics.",
            call. = FALSE
        )
    }
    ivf <- faiss_ivf_params(
        n,
        k,
        metric = metric,
        p = p,
        backend = "cuda",
        method = "ivfpq_fastscan",
        target_recall = target_recall
    )
    pq <- cuvs_ivfpq_params(p, n = n)
    manual_pq <- nn_any_options(c(
        "cuvs_ivfpq_pq_dim",
        "ivfpq_pq_dim",
        "cuvs_ivfpq_pq_bits",
        "ivfpq_pq_bits"
    ))
    tuned_pq_dim <- faissr_quiet_warning(as.integer(ivf$pq_m %||% NA_integer_))
    tuned_pq_bits <- faissr_quiet_warning(as.integer(ivf$pq_nbits %||% 4L))
    pq <- configure_cuda_fastscan_pq(
        pq, ivf, manual_pq, tuned_pq_dim, tuned_pq_bits, p, metric
    )
    batch_size <- valid_positive_integer_or_na(ivf$cuvs_ivf_batch_size)
    list(
        ivf = ivf,
        pq = pq,
        tuning = ivfpq_fastscan_tuning_metadata(
            ivf, "cuda", metric, batch_size,
            default_rule = "cuda_cuvs_ivfpq_fastscan_4bit"
        )
    )
}

configure_cuda_fastscan_pq <- function(
    pq, ivf, manual_pq, tuned_pq_dim, tuned_pq_bits, p, metric
) {
    if (
        !manual_pq &&
            length(tuned_pq_dim) == 1L &&
            !is.na(tuned_pq_dim) &&
            is.finite(tuned_pq_dim) &&
            tuned_pq_dim > 0L
    ) {
        pq$pq_dim <- as.integer(tuned_pq_dim)
        pq$requested_pq_dim <- as.integer(tuned_pq_dim)
    }
    if (
        !manual_pq &&
            length(tuned_pq_bits) == 1L &&
            !is.na(tuned_pq_bits) &&
            is.finite(tuned_pq_bits) &&
            tuned_pq_bits > 0L
    ) {
        pq$pq_bits <- as.integer(tuned_pq_bits)
        pq$requested_pq_bits <- as.integer(tuned_pq_bits)
    } else if (!manual_pq) {
        pq$pq_bits <- 4L
        pq$requested_pq_bits <- 4L
    }
    pq$tuning_policy <- "auto_shape_k_target_recall_ivfpq_fastscan"
    pq$tuning_rule <- ivf$tuning_rule %||% "cuda_cuvs_ivfpq_fastscan_4bit"
    pq$tuning_metric <- metric
    pq$tuning_backend <- "cuda"
    pq$tuning_method <- "ivfpq_fastscan"
    pq$tuning_shape_group <- ivf$tuning_shape_group %||% NA_character_
    pq$tuning_k_bucket <- ivf$tuning_k_bucket %||% NA_integer_
    pq$tuning_target_recall_code <- ivf$tuning_target_recall_code %||%
        NA_integer_
    pq$tuning_benchmark_basis <- ivf$tuning_benchmark_basis %||% NA_character_
    pq$tuning_benchmark_target_met <- ivf$tuning_benchmark_target_met %||% FALSE
    pq$tuning_benchmark_source <- ivf$tuning_benchmark_source %||% NA_character_
    cuvs_ivfpq_align_params(pq, p)
}

valid_positive_integer_or_na <- function(value) {
    value <- faissr_quiet_warning(as.integer(value %||% NA_integer_))
    if (length(value) != 1L || is.na(value) || !is.finite(value) ||
        value < 1L) {
        NA_integer_
    } else {
        value
    }
}

.faiss_gpu_ivf_tune_cache <- new.env(parent = emptyenv())
.faiss_gpu_ivf_tune_disk_cache <- new.env(parent = emptyenv())
.faiss_gpu_ivf_tune_disk_cache$loaded <- FALSE
.faiss_gpu_ivf_tune_disk_cache$file <- NULL
.faiss_gpu_ivf_tune_disk_cache$entries <- list()

faiss_gpu_ivf_tune_policy <- function(tuning = "auto") {
    tuning <- normalize_nn_tuning(tuning)
    if (!identical(tuning, "auto")) {
        return(tuning)
    }
    policy <- faissr_option("faiss_gpu_ivf_tune_policy", "fixed")
    policy <- tolower(as.character(policy)[1L])
    if (!policy %in% c("cache", "pilot", "fixed", "off")) {
        policy <- "fixed"
    }
    policy
}

faiss_gpu_ivf_tune_cache_file <- function() {
    path <- faissr_option("faiss_gpu_ivf_tune_cache_file", NULL)
    if (!is.null(path)) {
        return(path)
    }
    root <- tryCatch(
        tools::R_user_dir("faissR", which = "cache"),
        error = function(e) file.path(tempdir(), "faissR-cache")
    )
    file.path(root, "faiss_gpu_ivf_tuning.rds")
}

faiss_gpu_ivf_should_tune <- function(
    data,
    k,
    self_query,
    tuning = "auto",
    metric = "euclidean"
) {
    tuning <- normalize_nn_tuning(tuning)
    metric <- normalize_nn_metric(metric)
    policy <- faiss_gpu_ivf_tune_policy(tuning)
    if (!policy %in% c("cache", "pilot")) {
        return(FALSE)
    }
    if (!identical(metric, "euclidean")) {
        return(FALSE)
    }
    if (!isTRUE(self_query)) {
        return(FALSE)
    }
    if (!isTRUE(faiss_gpu_available())) {
        return(FALSE)
    }
    if (isTRUE(cuda_ivf_manual_params())) {
        return(FALSE)
    }
    enabled <- faissr_option("faiss_gpu_ivf_tune", TRUE)
    if (!isTRUE(enabled)) {
        return(FALSE)
    }
    threshold <- faissr_option("faiss_gpu_ivf_tune_threshold", 20000L)
    threshold <- faissr_quiet_warning(as.integer(threshold))
    if (length(threshold) != 1L || is.na(threshold) || !is.finite(threshold)) {
        threshold <- 20000L
    }
    nrow(data) >= threshold && as.integer(k) >= 10L
}

faiss_gpu_ivf_tune_signature <- function(data, k, sample_size, target, seed) {
    rows <- unique(as.integer(round(seq(
        1L,
        nrow(data),
        length.out = min(17L, nrow(data))
    ))))
    cols <- seq_len(min(13L, ncol(data)))
    sample_sum <- sum(data[rows, cols, drop = FALSE])
    paste(
        "faiss_gpu_ivf_flat",
        nrow(data),
        ncol(data),
        as.integer(k),
        as.integer(sample_size),
        format(signif(target, 8L), scientific = TRUE),
        as.integer(seed),
        format(signif(sample_sum, 8L), scientific = TRUE),
        sep = ":"
    )
}

faiss_gpu_ivf_load_disk_cache <- function() {
    file <- faiss_gpu_ivf_tune_cache_file()
    if (
        isTRUE(.faiss_gpu_ivf_tune_disk_cache$loaded) &&
            identical(.faiss_gpu_ivf_tune_disk_cache$file, file)
    ) {
        return(invisible(.faiss_gpu_ivf_tune_disk_cache$entries))
    }
    .faiss_gpu_ivf_tune_disk_cache$loaded <- TRUE
    .faiss_gpu_ivf_tune_disk_cache$file <- file
    entries <- tryCatch(
        faissr_quiet_warning(readRDS(file)),
        error = function(e) list()
    )
    if (!is.list(entries)) {
        entries <- list()
    }
    .faiss_gpu_ivf_tune_disk_cache$entries <- entries
    invisible(entries)
}

faiss_gpu_ivf_get_cached_tuning <- function(key) {
    cached <- .faiss_gpu_ivf_tune_cache[[key]]
    if (!is.null(cached)) {
        if (is.list(cached$tuning)) {
            cached$tuning$cache <- "memory"
        }
        return(cached)
    }
    faiss_gpu_ivf_load_disk_cache()
    cached <- .faiss_gpu_ivf_tune_disk_cache$entries[[key]]
    if (!is.null(cached)) {
        if (is.list(cached$tuning)) {
            cached$tuning$cache <- "disk"
        }
        .faiss_gpu_ivf_tune_cache[[key]] <- cached
        return(cached)
    }
    NULL
}

faiss_gpu_ivf_set_cached_tuning <- function(key, value, persist = TRUE) {
    .faiss_gpu_ivf_tune_cache[[key]] <- value
    if (!isTRUE(persist)) {
        return(invisible(value))
    }
    faiss_gpu_ivf_load_disk_cache()
    .faiss_gpu_ivf_tune_disk_cache$entries[[key]] <- value
    file <- faiss_gpu_ivf_tune_cache_file()
    dir.create(dirname(file), recursive = TRUE, showWarnings = FALSE)
    tryCatch(
        saveRDS(.faiss_gpu_ivf_tune_disk_cache$entries, file),
        error = function(e) NULL
    )
    invisible(value)
}

faiss_gpu_ivf_candidate_params <- function(n, k, base_params) {
    n <- as.integer(n)
    k <- as.integer(k)
    base_nlist <- as.integer(base_params$nlist)
    base_nprobe <- as.integer(base_params$nprobe)
    max_nlist <- max(4L, min(n, floor(n / 40L)))
    nlist_values <- unique(as.integer(round(c(
        max(16L, base_nlist),
        max(16L, 2L * base_nlist),
        max(16L, 4L * base_nlist),
        max(16L, ceiling(sqrt(n))),
        max(16L, ceiling(2 * sqrt(n)))
    ))))
    nlist_values <- unique(pmax(1L, pmin(max_nlist, nlist_values)))
    rows <- vector("list", length(nlist_values) * 4L)
    pos <- 0L
    for (nlist in nlist_values) {
        probes <- unique(as.integer(ceiling(c(
            base_nprobe,
            sqrt(nlist),
            0.10 * nlist,
            0.20 * nlist
        ))))
        probes <- unique(pmax(1L, pmin(nlist, probes)))
        for (nprobe in probes) {
            pos <- pos + 1L
            rows[[pos]] <- data.frame(
                nlist = as.integer(nlist),
                nprobe = as.integer(nprobe),
                stringsAsFactors = FALSE
            )
        }
    }
    candidates <- do.call(rbind, rows[seq_len(pos)])
    candidates <- unique(candidates)
    candidates[order(candidates$nlist, candidates$nprobe), , drop = FALSE]
}

faiss_gpu_ivf_tune_params <- function(data, k, base_params, tuning = "auto") {
    policy <- faiss_gpu_ivf_tune_policy(tuning)
    if (identical(policy, "off")) {
        return(disabled_tuning_result(base_params, policy))
    }
    controls <- faiss_gpu_ivf_tune_controls(data)
    key <- faiss_gpu_ivf_tune_signature(
        data, k, controls$sample_size, controls$target, controls$seed
    )
    cached <- faiss_gpu_ivf_get_cached_tuning(key)
    if (!is.null(cached)) return(cached)
    if (identical(policy, "fixed")) {
        return(fixed_tuning_result(base_params, policy, controls))
    }
    sample <- faiss_gpu_ivf_tuning_sample(data, k, controls)
    if (inherits(sample$reference, "error")) {
        out <- failed_tuning_result(
            base_params, policy, controls, sample$reference
        )
    } else {
        results <- evaluate_faiss_gpu_ivf_candidates(sample, base_params)
        selected <- select_tuning_result(
            results, base_params, controls$target, c("nlist", "nprobe")
        )
        out <- completed_tuning_result(selected, results, policy, controls)
    }
    faiss_gpu_ivf_set_cached_tuning(
        key, out, persist = !identical(policy, "pilot")
    )
    out
}

normalize_tuning_control <- function(value, default, minimum, type) {
    value <- faissr_quiet_warning(switch(
        type, integer = as.integer(value), numeric = as.numeric(value)
    ))
    if (length(value) != 1L || is.na(value) || !is.finite(value) ||
        value < minimum) default else value
}

faiss_gpu_ivf_tune_controls <- function(data) {
    sample_size <- normalize_tuning_control(
        faissr_option("faiss_gpu_ivf_tune_sample", 10000L),
        10000L, 1000L, "integer"
    )
    seed <- normalize_tuning_control(
        faissr_option("faiss_gpu_ivf_tune_seed", 7L), 7L, -Inf, "integer"
    )
    target <- normalize_tuning_control(
        faissr_option("faiss_gpu_ivf_tune_recall", 0.985),
        0.985, -Inf, "numeric"
    )
    list(
        sample_size = as.integer(min(nrow(data), sample_size)),
        seed = as.integer(seed), target = max(0, min(1, target))
    )
}

disabled_tuning_result <- function(params, policy) {
    list(params = params, tuning = list(status = "disabled", policy = policy))
}

fixed_tuning_result <- function(params, policy, controls) {
    list(params = params, tuning = list(
        status = "fixed_default", policy = policy, cache = "miss",
        sample_size = controls$sample_size,
        target_recall = as.numeric(controls$target)
    ))
}

failed_tuning_result <- function(params, policy, controls, error) {
    list(params = params, tuning = list(
        status = "failed", policy = policy, cache = "miss",
        reason = conditionMessage(error), sample_size = controls$sample_size
    ))
}

faiss_gpu_ivf_tuning_sample <- function(data, k, controls) {
    with_rng_seed(controls$seed, {
        rows <- sort(sample.int(nrow(data), controls$sample_size))
        x <- data[rows, , drop = FALSE]
        compare_k <- as.integer(min(k, nrow(x)))
        reference <- tryCatch(
            nn_faiss_gpu_flat_cpp(x, x, compare_k, FALSE),
            error = function(e) e
        )
        list(data = x, k = compare_k, reference = reference)
    })
}

evaluate_faiss_gpu_ivf_candidates <- function(sample, base_params) {
    x <- sample$data
    compare_k <- sample$k
    candidates <- faiss_gpu_ivf_candidate_params(
        nrow(x), compare_k, base_params
    )
    rows <- lapply(seq_len(nrow(candidates)), function(i) {
        cand <- candidates[i, , drop = FALSE]
        timed_tuning_candidate(cand, function() {
            nn_faiss_gpu_ivf_flat_cpp(
                x,
                x,
                compare_k,
                as.integer(cand$nlist),
                as.integer(cand$nprobe),
                "euclidean",
                "euclidean",
                FALSE
            )
        }, sample$reference, compare_k)
    })
    do.call(rbind, rows)
}

timed_tuning_candidate <- function(candidate, executor, reference, k) {
    elapsed <- system.time({
        result <- tryCatch(executor(), error = function(e) e)
    })[["elapsed"]]
    failed <- inherits(result, "error")
    recall <- if (failed) NA_real_ else {
        .knn_recall_summary(result, reference, k = k)$recall_at_k
    }
    cbind(candidate, data.frame(
        seconds = as.numeric(elapsed), recall = as.numeric(recall),
        status = if (failed) "failed" else "success",
        error = if (failed) conditionMessage(result) else "",
        stringsAsFactors = FALSE
    ))
}

select_tuning_result <- function(results, defaults, target, fields) {
    success <- results[results$status == "success", , drop = FALSE]
    if (!nrow(success)) return(list(params = defaults, status = "failed"))
    eligible <- success[is.finite(success$recall) & success$recall >= target,
        , drop = FALSE]
    if (nrow(eligible)) {
        row <- eligible[order(eligible$seconds, -eligible$recall),
            , drop = FALSE][1L, , drop = FALSE]
        status <- "target_met"
    } else {
        row <- success[order(-success$recall, success$seconds),
            , drop = FALSE][1L, , drop = FALSE]
        status <- "best_available"
    }
    params <- lapply(fields, function(name) as.integer(row[[name]]))
    names(params) <- fields
    list(params = params, status = status)
}

completed_tuning_result <- function(selected, results, policy, controls) {
    list(params = selected$params, tuning = list(
        status = selected$status, policy = policy, cache = "miss",
        sample_size = as.integer(controls$sample_size),
        target_recall = as.numeric(controls$target),
        chosen = selected$params, results = results
    ))
}

faiss_option_int <- function(
    name,
    default,
    min_value = 1L,
    max_value = .Machine$integer.max
) {
    value <- faissr_option(paste0("faiss_", name), NULL)
    value <- if (is.null(value)) {
        default
    } else {
        faissr_quiet_warning(as.integer(value))
    }
    if (length(value) != 1L || is.na(value) || !is.finite(value)) {
        value <- default
    }
    as.integer(max(min_value, min(max_value, value)))
}

faiss_pq_default_m <- function(p) {
    p <- as.integer(p)
    candidates <- seq.int(min(p, 96L), 1L, by = -1L)
    candidates <- candidates[p %% candidates == 0L]
    if (length(candidates) == 0L) {
        return(1L)
    }
    as.integer(candidates[[1L]])
}

faiss_cpu_ivfpq_min_training_rows <- function() 624L

faiss_cpu_ivfpq_8bit_training_rows <- function() 9984L

validate_faiss_cpu_ivfpq_training_size <- function(n) {
    n <- faissr_quiet_warning(as.integer(n))
    min_n <- faiss_cpu_ivfpq_min_training_rows()
    if (length(n) != 1L || is.na(n) || n < min_n) {
        stop(
            "FAISS CPU IVFPQ requires at least ",
            min_n,
            " training rows for the smallest supported ",
            "4-bit product quantizer. ",
            "Use `method = \"ivf\"`, `\"hnsw\"`, or ",
            "`\"flat\"` for smaller datasets.",
            call. = FALSE
        )
    }
    invisible(TRUE)
}

faiss_pq_manual_params <- function() {
    nn_any_options(c("faiss_pq_m", "faiss_pq_nbits"))
}

faiss_pq_manual_nbits <- function() {
    !is.null(faissr_option("faiss_pq_nbits", NULL))
}

faiss_pq_params <- function(p, n = NULL) {
    nn_tune_faiss_pq_cpp(
        as.integer(p),
        faissr_quiet_warning(as.integer(n %||% NA_integer_)),
        nn_option_int_or_na("faiss_pq_m"),
        nn_option_int_or_na("faiss_pq_nbits"),
        faiss_pq_manual_params(),
        faiss_pq_manual_nbits()
    )
}

faiss_ivfpq_pq_params <- function(p, n = NULL, ivf_params = NULL) {
    pq <- faiss_pq_params(p, n = n)
    if (faiss_pq_manual_params() || !is.list(ivf_params)) {
        return(pq)
    }
    pq_m <- faissr_quiet_warning(as.integer(ivf_params$pq_m %||% NA_integer_))
    pq_nbits <- faissr_quiet_warning(as.integer(
        ivf_params$pq_nbits %||% NA_integer_
    ))
    if (length(pq_m) != 1L || is.na(pq_m) || pq_m < 1L) {
        return(pq)
    }
    if (length(pq_nbits) != 1L || is.na(pq_nbits) || pq_nbits < 1L) {
        return(pq)
    }
    pq$m <- pq_m
    pq$nbits <- pq_nbits
    pq$tuning_policy <- "auto_shape_k_target_recall_ivfpq"
    pq$tuning_rule <- ivf_params$tuning_rule %||% "hpc_cpu_ivfpq"
    pq$tuning_metric <- ivf_params$tuning_metric %||% "euclidean"
    pq$tuning_metric_aware <- ivf_params$tuning_metric_aware %||% FALSE
    pq$target_recall <- ivf_params$target_recall %||% NA_real_
    pq$requested_target_recall <- ivf_params$requested_target_recall %||%
        NA_real_
    pq$tuning_shape_group <- ivf_params$tuning_shape_group %||% NA_character_
    pq$tuning_k_bucket <- ivf_params$tuning_k_bucket %||% NA_integer_
    pq$tuning_target_recall_code <- ivf_params$tuning_target_recall_code %||%
        NA_integer_
    pq$tuning_benchmark_basis <- ivf_params$tuning_benchmark_basis %||%
        NA_character_
    pq$tuning_benchmark_target_met <-
        ivf_params$tuning_benchmark_target_met %||% FALSE
    pq$tuning_benchmark_source <- ivf_params$tuning_benchmark_source %||%
        NA_character_
    pq$tuning_source <- ivf_params$tuning_source %||% "cpp"
    pq
}

faiss_hnsw_auto_policy <- function(
    n = NULL,
    p = NULL,
    k,
    metric = "euclidean",
    target_recall = 0.99
) {
    target_recall <- normalize_hnsw_target_recall(target_recall)
    out <- nn_tune_faiss_hnsw_cpp(
        faissr_quiet_warning(as.integer(n %||% NA_integer_)),
        faissr_quiet_warning(as.integer(p %||% NA_integer_)),
        as.integer(k),
        normalize_nn_metric(metric),
        as.numeric(target_recall),
        NA_integer_,
        NA_integer_,
        NA_integer_,
        FALSE
    )
    out[c(
        "m",
        "ef_construction",
        "ef_search",
        "rule",
        "high_dim",
        "large_n",
        "small_k",
        "large_k",
        "non_euclidean",
        "target_recall",
        "shape_group",
        "k_bucket",
        "target_recall_code",
        "benchmark_basis",
        "benchmark_source"
    )]
}

faiss_hnsw_manual_params <- function() {
    any(vapply(
        c("hnsw_m", "hnsw_ef_construction", "hnsw_ef_search"),
        function(name) !is.null(faissr_option(paste0("faiss_", name), NULL)),
        logical(1)
    ))
}

faiss_hnsw_params <- function(
    k,
    n = NULL,
    p = NULL,
    metric = "euclidean",
    target_recall = 0.99
) {
    target_recall <- normalize_hnsw_target_recall(target_recall)
    nn_tune_faiss_hnsw_cpp(
        faissr_quiet_warning(as.integer(n %||% NA_integer_)),
        faissr_quiet_warning(as.integer(p %||% NA_integer_)),
        as.integer(k),
        normalize_nn_metric(metric),
        as.numeric(target_recall),
        nn_option_int_or_na("faiss_hnsw_m"),
        nn_option_int_or_na("faiss_hnsw_ef_construction"),
        nn_option_int_or_na("faiss_hnsw_ef_search"),
        faiss_hnsw_manual_params()
    )
}

faiss_nsg_params <- function(k) {
    nn_tune_faiss_nsg_cpp(
        as.integer(k),
        nn_option_int_or_na("faiss_nsg_r"),
        nn_option_int_or_na("faiss_nsg_search_l"),
        nn_option_int_or_na("faiss_nsg_build_type"),
        nn_any_options(c(
            "faiss_nsg_r",
            "faiss_nsg_search_l",
            "faiss_nsg_build_type"
        ))
    )
}

faiss_nndescent_params <- function(k) {
    nn_tune_faiss_nndescent_cpp(
        as.integer(k),
        nn_option_int_or_na("faiss_nndescent_graph_k"),
        nn_option_int_or_na("faiss_nndescent_iter"),
        nn_option_int_or_na("faiss_nndescent_search_l"),
        nn_any_options(c(
            "faiss_nndescent_graph_k",
            "faiss_nndescent_iter",
            "faiss_nndescent_search_l"
        ))
    )
}

cuvs_option_int <- function(
    name,
    default,
    min_value = 1L,
    max_value = .Machine$integer.max
) {
    value <- faissr_option(paste0("cuvs_", name), NULL)
    value <- if (is.null(value)) {
        default
    } else {
        faissr_quiet_warning(as.integer(value))
    }
    if (length(value) != 1L || is.na(value) || !is.finite(value)) {
        value <- default
    }
    as.integer(max(min_value, min(max_value, value)))
}

cuvs_requested_option_int <- function(name, default) {
    value <- faissr_option(paste0("cuvs_", name), NULL)
    value <- if (is.null(value)) {
        default
    } else {
        faissr_quiet_warning(as.integer(value))
    }
    if (length(value) != 1L || is.na(value) || !is.finite(value)) {
        value <- default
    }
    as.integer(value)
}

cuvs_cagra_params <- function(
    n,
    k,
    p = NA_integer_,
    metric = "euclidean",
    target_recall = 0.99
) {
    target_recall <- normalize_hnsw_target_recall(target_recall)
    metric <- normalize_nn_metric(metric)
    nn_tune_cuvs_cagra_cpp(
        as.integer(n),
        faissr_quiet_warning(as.integer(p)),
        as.integer(k),
        metric,
        as.numeric(target_recall),
        nn_option_int_or_na("cuvs_graph_degree"),
        nn_option_int_or_na("cuvs_intermediate_graph_degree"),
        nn_option_int_or_na("cuvs_search_width"),
        nn_option_int_or_na("cuvs_itopk_size"),
        cuvs_cagra_manual_params()
    )
}

cuvs_hnsw_params <- function(
    n,
    k,
    p = NA_integer_,
    n_threads = NULL,
    target_recall = 0.99,
    metric = "euclidean"
) {
    target_recall <- normalize_hnsw_target_recall(target_recall)
    metric <- normalize_nn_metric(metric)
    nn_tune_cuvs_hnsw_cpp(
        as.integer(n),
        faissr_quiet_warning(as.integer(p)),
        as.integer(k),
        as.integer(normalize_nn_threads(n_threads)),
        cagra_build_algo_preference(),
        as.numeric(target_recall),
        nn_option_int_or_na("cuvs_graph_degree"),
        nn_option_int_or_na("cuvs_intermediate_graph_degree"),
        nn_option_int_or_na("cuvs_search_width"),
        nn_option_int_or_na("cuvs_itopk_size"),
        nn_option_int_or_na("cuvs_hnsw_ef"),
        cuvs_cagra_manual_params(),
        metric
    )
}

cuvs_hnsw_result <- function(
    data, points, k, self_query, exclude_self, metric, n_threads = NULL,
    target_recall = 0.99, output = "double",
    result_backend = "cuda_cuvs_hnsw"
) {
    require_cuvs_backend("cuVS HNSW")
    metric <- normalize_nn_metric(metric)
    target_recall <- normalize_hnsw_target_recall(target_recall)
    route <- prepare_cuvs_hnsw_route(data, points, self_query, metric, output)
    params <- cuvs_hnsw_params(
        route$dims[[1L]], k, route$dims[[2L]], n_threads,
        target_recall, metric
    )
    out <- execute_cuvs_hnsw(route, k, exclude_self, params)
    result <- finish_nn_result(
        out, result_backend, k, self_query, exact = FALSE, metric = metric
    )
    if (!is.null(route$metric_inputs)) {
        result <- finalize_graph_metric_result(result, route$metric_inputs)
    }
    if (isTRUE(route$float32)) result <- finish_float32_direct_result(result,
        out)
    attr(result, "approximation") <- cuvs_hnsw_metadata(
        out, route, params, target_recall
    )
    append_nn_tuning_metadata(result, params)
}

prepare_cuvs_hnsw_route <- function(data, points, self_query, metric, output) {
    inputs <- NULL
    if (metric %in% c("cosine", "correlation")) {
        inputs <- normalized_euclidean_metric_inputs(
            data, points, self_query, metric, storage = "float"
        )
        reject_cuda_normalized_cpu_repair(
            "cuda", metric, inputs$data_zero, inputs$points_zero,
            "cuda_cuvs_hnsw"
        )
    } else if (identical(metric, "inner_product")) {
        inputs <- mips_l2_metric_inputs(data, points, self_query)
    }
    search_data <- if (is.null(inputs)) data else inputs$data
    search_points <- if (is.null(inputs)) points else inputs$points
    dims <- if (is_float32_matrix_input(search_data)) {
        float32_matrix_dims(search_data, "data")
    } else dim(search_data)
    float32 <- isTRUE(is.null(inputs) && (
        identical(output, "float") || is_float32_matrix_input(search_data) ||
            is_float32_matrix_input(search_points)
    )) || identical(inputs$transform_storage %||% "double", "float32")
    storage <- if (float32 && is.null(inputs) && output == "float") {
        "float"
    } else "double"
    list(
        data = search_data, points = search_points, dims = dims,
        metric = metric, metric_inputs = inputs, float32 = float32,
        distance_storage = storage
    )
}

execute_cuvs_hnsw <- function(route, k, exclude_self, params) {
    if (isTRUE(route$float32)) {
        nn_cuvs_hnsw_float32_cpp(
            route$data, route$points,
            as.integer(k),
            isTRUE(exclude_self),
            as.integer(params$graph_degree),
            as.integer(params$intermediate_graph_degree),
            as.integer(params$ef),
            as.integer(params$n_threads),
            as.character(params$cagra_build_algo),
            route$distance_storage
        )
    } else {
        nn_cuvs_hnsw_cpp(
            route$data, route$points,
            as.integer(k),
            isTRUE(exclude_self),
            as.integer(params$graph_degree),
            as.integer(params$intermediate_graph_degree),
            as.integer(params$ef),
            as.integer(params$n_threads),
            as.character(params$cagra_build_algo)
        )
    }
}

cuvs_hnsw_metric_metadata <- function(route) {
    inputs <- route$metric_inputs
    list(
        transform = if (is.null(inputs)) NA_character_ else inputs$transform,
        metric_transform = if (is.null(
            inputs)) NA_character_ else inputs$transform,

        distance_transform = if (is.null(inputs)) NA_character_ else {
            inputs$distance_transform %||%
                .normalized_similarity_distance_transform
        },
        transform_storage = inputs$transform_storage %||% "double",
        transform_cache = inputs$transform_cache %||% NULL
    )
}

cuvs_hnsw_metadata <- function(out, route, params, target_recall) {
    metadata <- list(
        strategy = "rapids_cuvs_hnsw_from_cagra",
        backend = "cuda_cuvs_hnsw",
        library = "cuvs",
        accelerator = "cuda",
        metric = route$metric,
        graph_degree = as.integer(out$graph_degree),
        intermediate_graph_degree = as.integer(out$intermediate_graph_degree),
        ef = as.integer(out$ef),
        num_threads = as.integer(out$num_threads),
        cagra_build_algo = out$cagra_build_algo %||% params$cagra_build_algo,
        hnsw_build_algo = out$hnsw_build_algo %||% "from_cagra",
        hnsw_hierarchy = out$hnsw_hierarchy %||% "cpu",
        hnsw_m = as.integer(out$hnsw_m %||% NA_integer_),
        hnsw_ef_construction = as.integer(
            out$hnsw_ef_construction %||% NA_integer_
        ),
        requested_graph_degree = as.integer(params$requested_graph_degree),
        requested_intermediate_graph_degree = as.integer(
            params$requested_intermediate_graph_degree
        ),
        requested_ef = as.integer(params$requested_ef),
        hnsw_parameters_adjusted = isTRUE(out$hnsw_parameters_adjusted),
        target_recall = as.numeric(params$target_recall %||% target_recall),
        tuning_policy = params$tuning_policy,
        tuning_rule = params$tuning_rule,
        tuning_source = params$tuning_source %||% "cpp",
        cuda_hnsw_design = out$cuda_hnsw_design %||%
            "cuvs_hnsw_from_cagra_cpu_hierarchy",
        cuda_hnsw_pure_gpu = isTRUE(out$cuda_hnsw_pure_gpu)
    )
    c(metadata, cuvs_hnsw_metric_metadata(route))
}

.cuvs_cagra_tune_cache <- new.env(parent = emptyenv())
.cuvs_cagra_tune_disk_cache <- new.env(parent = emptyenv())
.cuvs_cagra_tune_disk_cache$loaded <- FALSE
.cuvs_cagra_tune_disk_cache$file <- NULL
.cuvs_cagra_tune_disk_cache$entries <- list()

cuvs_cagra_manual_params <- function() {
    any(vapply(
        c(
            "graph_degree",
            "intermediate_graph_degree",
            "search_width",
            "itopk_size"
        ),
        function(name) !is.null(faissr_option(paste0("cuvs_", name), NULL)),
        logical(1)
    ))
}

cuvs_cagra_should_tune <- function(data, k, self_query, tuning = "auto") {
    tuning <- normalize_nn_tuning(tuning)
    policy <- cuvs_cagra_tune_policy(tuning)
    if (!policy %in% c("cache", "pilot")) {
        return(FALSE)
    }
    if (!isTRUE(self_query)) {
        return(FALSE)
    }
    if (!isTRUE(cuvs_available())) {
        return(FALSE)
    }
    if (isTRUE(cuvs_cagra_manual_params())) {
        return(FALSE)
    }
    enabled <- faissr_option("cuvs_cagra_tune", TRUE)
    if (!isTRUE(enabled)) {
        return(FALSE)
    }
    threshold <- faissr_option("cuvs_cagra_tune_threshold", 20000L)
    threshold <- faissr_quiet_warning(as.integer(threshold))
    if (length(threshold) != 1L || is.na(threshold) || !is.finite(threshold)) {
        threshold <- 20000L
    }
    nrow(data) >= threshold && as.integer(k) >= 10L
}

cuvs_cagra_tune_policy <- function(tuning = "auto") {
    tuning <- normalize_nn_tuning(tuning)
    if (!identical(tuning, "auto")) {
        return(tuning)
    }
    policy <- faissr_option("cuvs_cagra_tune_policy", "fixed")
    policy <- tolower(as.character(policy)[1L])
    if (!policy %in% c("cache", "pilot", "fixed", "off")) {
        policy <- "fixed"
    }
    policy
}

cuvs_cagra_tune_cache_file <- function() {
    path <- faissr_option("cuvs_cagra_tune_cache_file", NULL)
    if (!is.null(path)) {
        return(path)
    }
    root <- tryCatch(
        tools::R_user_dir("faissR", which = "cache"),
        error = function(e) file.path(tempdir(), "faissR-cache")
    )
    file.path(root, "cuvs_cagra_tuning.rds")
}

cuvs_cagra_tune_signature <- function(data, k, sample_size, target, seed) {
    rows <- unique(as.integer(round(seq(
        1L,
        nrow(data),
        length.out = min(17L, nrow(data))
    ))))
    cols <- seq_len(min(13L, ncol(data)))
    sample_sum <- sum(data[rows, cols, drop = FALSE])
    paste(
        nrow(data),
        ncol(data),
        as.integer(k),
        as.integer(sample_size),
        format(signif(target, 8L), scientific = TRUE),
        as.integer(seed),
        format(signif(sample_sum, 8L), scientific = TRUE),
        sep = ":"
    )
}

cuvs_cagra_load_disk_cache <- function() {
    file <- cuvs_cagra_tune_cache_file()
    if (
        isTRUE(.cuvs_cagra_tune_disk_cache$loaded) &&
            identical(.cuvs_cagra_tune_disk_cache$file, file)
    ) {
        return(invisible(.cuvs_cagra_tune_disk_cache$entries))
    }
    .cuvs_cagra_tune_disk_cache$loaded <- TRUE
    .cuvs_cagra_tune_disk_cache$file <- file
    entries <- tryCatch(
        faissr_quiet_warning(readRDS(file)),
        error = function(e) list()
    )
    if (!is.list(entries)) {
        entries <- list()
    }
    .cuvs_cagra_tune_disk_cache$entries <- entries
    invisible(entries)
}

cuvs_cagra_get_cached_tuning <- function(key) {
    cached <- .cuvs_cagra_tune_cache[[key]]
    if (!is.null(cached)) {
        if (is.list(cached$tuning)) {
            cached$tuning$cache <- "memory"
        }
        return(cached)
    }
    cuvs_cagra_load_disk_cache()
    cached <- .cuvs_cagra_tune_disk_cache$entries[[key]]
    if (!is.null(cached)) {
        if (is.list(cached$tuning)) {
            cached$tuning$cache <- "disk"
        }
        .cuvs_cagra_tune_cache[[key]] <- cached
        return(cached)
    }
    NULL
}

cuvs_cagra_set_cached_tuning <- function(key, value, persist = TRUE) {
    .cuvs_cagra_tune_cache[[key]] <- value
    if (!isTRUE(persist)) {
        return(invisible(value))
    }
    cuvs_cagra_load_disk_cache()
    .cuvs_cagra_tune_disk_cache$entries[[key]] <- value
    file <- cuvs_cagra_tune_cache_file()
    dir.create(dirname(file), recursive = TRUE, showWarnings = FALSE)
    tryCatch(
        saveRDS(.cuvs_cagra_tune_disk_cache$entries, file),
        error = function(e) NULL
    )
    invisible(value)
}

cuvs_cagra_candidate_params <- function(k, n) {
    k <- as.integer(k)
    n <- as.integer(n)
    base_degree <- max(64L, k + 1L)
    candidates <- data.frame(
        graph_degree = c(base_degree, max(96L, k + 1L), max(128L, k + 1L)),
        intermediate_graph_degree = c(
            max(128L, 2L * base_degree),
            max(192L, 2L * max(96L, k + 1L)),
            max(256L, 2L * max(128L, k + 1L))
        ),
        search_width = c(0L, 0L, 0L),
        itopk_size = c(max(64L, base_degree), max(96L, k), max(128L, k)),
        stringsAsFactors = FALSE
    )
    candidates$graph_degree <- pmin(candidates$graph_degree, max(1L, n - 1L))
    candidates$intermediate_graph_degree <- pmin(
        pmax(candidates$intermediate_graph_degree, candidates$graph_degree),
        max(1L, n - 1L)
    )
    candidates$itopk_size <- pmax(as.integer(k), candidates$itopk_size)
    unique(candidates)
}

cuvs_cagra_tune_params <- function(
    data, k, base_params, tuning = "auto", build_algo = "auto"
) {
    policy <- cuvs_cagra_tune_policy(tuning)
    if (identical(policy, "off")) {
        return(disabled_tuning_result(base_params, policy))
    }
    controls <- cuvs_cagra_tune_controls(data)
    key <- cuvs_cagra_tune_signature(
        data, k, controls$sample_size, controls$target, controls$seed
    )
    cached <- cuvs_cagra_get_cached_tuning(key)
    if (!is.null(cached)) return(cached)
    if (identical(policy, "fixed")) {
        return(fixed_tuning_result(base_params, policy, controls))
    }
    sample <- cuvs_cagra_tuning_sample(data, k, controls)
    if (inherits(sample$reference, "error")) {
        out <- failed_tuning_result(
            base_params, policy, controls, sample$reference
        )
    } else {
        results <- evaluate_cuvs_cagra_candidates(sample, build_algo)
        selected <- select_tuning_result(
            results, base_params, controls$target,
            c("graph_degree", "intermediate_graph_degree", "search_width",
                "itopk_size")
        )
        out <- completed_tuning_result(selected, results, policy, controls)
    }
    cuvs_cagra_set_cached_tuning(
        key, out, persist = !identical(policy, "pilot")
    )
    out
}

cuvs_cagra_tune_controls <- function(data) {
    sample_size <- normalize_tuning_control(
        faissr_option("cuvs_cagra_tune_sample", 2048L),
        2048L, 256L, "integer"
    )
    seed <- normalize_tuning_control(
        faissr_option("cuvs_cagra_tune_seed", 4L), 4L, -Inf, "integer"
    )
    target <- normalize_tuning_control(
        faissr_option("cuvs_cagra_tune_recall", 0.985),
        0.985, -Inf, "numeric"
    )
    list(
        sample_size = as.integer(min(nrow(data), sample_size)),
        seed = as.integer(seed), target = max(0, min(1, target))
    )
}

cuvs_cagra_tuning_sample <- function(data, k, controls) {
    with_rng_seed(controls$seed, {
        rows <- sort(sample.int(nrow(data), controls$sample_size))
        x <- data[rows, , drop = FALSE]
        compare_k <- as.integer(min(k, nrow(x)))
        reference <- tryCatch(
            nn_cuvs_bruteforce_cpp(x, x, compare_k, FALSE),
            error = function(e) e
        )
        list(data = x, k = compare_k, reference = reference)
    })
}

evaluate_cuvs_cagra_candidates <- function(sample, build_algo) {
    candidates <- cuvs_cagra_candidate_params(sample$k, nrow(sample$data))
    rows <- lapply(seq_len(nrow(candidates)), function(i) {
        candidate <- candidates[i, , drop = FALSE]
        timed_tuning_candidate(candidate, function() {
            nn_cuvs_cagra_cpp(
                sample$data, sample$data, sample$k, FALSE,
                as.integer(candidate$graph_degree),
                as.integer(candidate$intermediate_graph_degree),
                as.integer(candidate$search_width),
                as.integer(candidate$itopk_size), build_algo
            )
        }, sample$reference, sample$k)
    })
    do.call(rbind, rows)
}

cuvs_nndescent_params <- function(
    n,
    p,
    k,
    metric = "euclidean",
    target_recall = 0.99
) {
    nn_tune_cuvs_nndescent_cpp(
        as.integer(n),
        faissr_quiet_warning(as.integer(p)),
        as.integer(k),
        normalize_nn_metric(metric),
        as.numeric(normalize_hnsw_target_recall(target_recall)),
        nn_option_int_or_na("cuvs_nndescent_graph_degree"),
        nn_option_int_or_na("cuvs_nndescent_intermediate_graph_degree"),
        nn_option_int_or_na("cuvs_nndescent_max_iterations"),
        nn_any_options(c(
            "cuvs_nndescent_graph_degree",
            "cuvs_nndescent_intermediate_graph_degree",
            "cuvs_nndescent_max_iterations"
        ))
    )
}

cuvs_nndescent_threshold <- function() {
    value <- faissr_option("cuvs_nndescent_threshold", 50000L)
    value <- faissr_quiet_warning(as.integer(value))
    if (length(value) != 1L || is.na(value) || !is.finite(value)) {
        value <- 50000L
    }
    as.integer(max(2L, value))
}

cuvs_should_use_nndescent <- function(self_query, n) {
    isTRUE(self_query) && as.integer(n) >= cuvs_nndescent_threshold()
}

require_cuvs_backend <- function(label = "cuVS") {
    if (isTRUE(cuvs_available())) {
        return(invisible(TRUE))
    }
    info <- tryCatch(cuvs_info_json_cpp(), error = function(e) NA_character_)
    reason <- json_get_string(info, "reason")
    suffix <- if (!is.na(reason) && nzchar(reason)) {
        paste0(" Reason: ", reason, ".")
    } else {
        ""
    }
    stop(
        label,
        " backend is not available in this build or no CUDA device is visible.",
        suffix,
        " Reinstall faissR with `FAISSR_USE_CUDA=1 FAISSR_USE_CUVS=1` and ",
        "`CUVS_HOME` pointing to a RAPIDS cuVS installation.",
        call. = FALSE
    )
}

ivf_self_knn <- function(
    data,
    k,
    backend = "cpu_ivf",
    seed = 4L,
    n_threads = NULL
) {
    n <- nrow(data)
    k <- as.integer(k)
    if (length(k) != 1L || is.na(k) || !is.finite(k) || k < 1L || k >= n) {
        stop("`k` must be in [1, nrow(data) - 1].", call. = FALSE)
    }
    n_threads <- normalize_nn_threads(n_threads)
    params <- ivf_self_params(n, k)
    out <- ivf_self_knn_cpp(
        data, k, params$nlist, params$nprobe, as.integer(seed), TRUE,
        as.integer(max(1L, min(8L, n_threads)))
    )
    attr(out, "approximation") <- list(
        strategy = if (identical(backend, "cpu_faiss_ivf")) {
            "faiss_style_ivf_flat_native"
        } else {
            "ivf_flat_native"
        },
        backend = backend,
        nlist = as.integer(out$nlist),
        nprobe = as.integer(out$nprobe),
        seed = as.integer(seed)
    )
    out
}

ivf_self_params <- function(n, k) {
    nlist <- faissr_option("ivf_nlist", NULL)
    nprobe <- faissr_option("ivf_nprobe", NULL)
    nlist <- if (is.null(nlist)) ivf_list_count(n, k) else as.integer(nlist)
    if (length(nlist) != 1L || is.na(nlist) || !is.finite(nlist)) {
        nlist <- ivf_list_count(n, k)
    }
    nlist <- max(1L, min(as.integer(n), nlist))
    nprobe <- if (is.null(nprobe)) {
        ivf_probe_count(nlist, k)
    } else {
        as.integer(nprobe)
    }
    if (length(nprobe) != 1L || is.na(nprobe) || !is.finite(nprobe)) {
        nprobe <- ivf_probe_count(nlist, k)
    }
    nprobe <- max(1L, min(nlist, nprobe))
    list(nlist = as.integer(nlist), nprobe = as.integer(nprobe))
}

grid2d_self_knn <- function(data, k, exclude_self = TRUE, n_threads = NULL) {
    n <- nrow(data)
    p <- ncol(data)
    if (p != 2L) {
        stop(
            "`backend = \"cpu_grid2d\"` requires a two-column matrix.",
            call. = FALSE
        )
    }
    k <- as.integer(k)
    include_self <- !isTRUE(exclude_self)
    nonself_k <- if (include_self) k - 1L else k
    if (
        length(k) != 1L ||
            is.na(k) ||
            !is.finite(k) ||
            k < 1L ||
            k > n ||
            (!include_self && k >= n) ||
            nonself_k < 0L
    ) {
        stop(
            "`k` must be compatible with the requested self-neighbor policy.",
            call. = FALSE
        )
    }
    n_threads <- normalize_nn_threads(n_threads)
    bins <- grid2d_bins_per_dim(n, max(1L, nonself_k))
    out <- grid2d_self_knn_cpp(
        data,
        as.integer(k),
        TRUE,
        as.integer(n_threads),
        as.integer(bins),
        isTRUE(include_self)
    )
    attr(out, "spatial_index") <- list(
        strategy = "native_exact_uniform_grid_2d",
        backend = "cpu_grid2d",
        exact = TRUE,
        bins_per_dim = as.integer(out$bins_per_dim),
        n_cells = as.integer(out$n_cells),
        n_threads = as.integer(out$n_threads),
        self_column_included = isTRUE(out$self_column_included),
        output_layout = out$output_layout %||% "knn_matrix_final",
        r_side_reshaping = FALSE
    )
    out
}

grid3d_self_knn <- function(data, k, exclude_self = TRUE, n_threads = NULL) {
    n <- nrow(data)
    p <- ncol(data)
    if (p != 3L) {
        stop(
            "`backend = \"cpu_grid3d\"` requires a three-column matrix.",
            call. = FALSE
        )
    }
    k <- as.integer(k)
    include_self <- !isTRUE(exclude_self)
    nonself_k <- if (include_self) k - 1L else k
    if (
        length(k) != 1L ||
            is.na(k) ||
            !is.finite(k) ||
            k < 1L ||
            k > n ||
            (!include_self && k >= n) ||
            nonself_k < 0L
    ) {
        stop(
            "`k` must be compatible with the requested self-neighbor policy.",
            call. = FALSE
        )
    }
    n_threads <- normalize_nn_threads(n_threads)
    bins <- grid3d_bins_per_dim(n, max(1L, nonself_k))
    out <- grid3d_self_knn_cpp(
        data,
        as.integer(k),
        TRUE,
        as.integer(n_threads),
        as.integer(bins),
        isTRUE(include_self)
    )
    attr(out, "spatial_index") <- list(
        strategy = "native_exact_uniform_grid_3d",
        backend = "cpu_grid3d",
        exact = TRUE,
        bins_per_dim = as.integer(out$bins_per_dim),
        n_cells = as.integer(out$n_cells),
        n_threads = as.integer(out$n_threads),
        self_column_included = isTRUE(out$self_column_included),
        output_layout = out$output_layout %||% "knn_matrix_final",
        r_side_reshaping = FALSE
    )
    out
}

grid_self_knn <- function(
    data,
    k,
    backend = "cpu_grid",
    exclude_self = TRUE,
    n_threads = NULL
) {
    p <- ncol(data)
    if (backend %in% c("grid", "cpu_grid")) {
        backend <- select_cpu_spatial_backend(
            data,
            k = k,
            exclude_self = exclude_self
        )
    }
    reason <- attr(backend, "reason", exact = TRUE)
    if (backend %in% c("grid3d", "cpu_grid3d") || identical(p, 3L)) {
        out <- grid3d_self_knn(
            data,
            k,
            exclude_self = exclude_self,
            n_threads = n_threads
        )
        if (!is.null(reason)) {
            attr(out, "spatial_index")$reason <- reason
        }
        return(out)
    }
    if (
        backend %in%
            c("grid2d", "cpu_grid2d", "grid", "cpu_grid") ||
            identical(p, 2L)
    ) {
        out <- grid2d_self_knn(
            data,
            k,
            exclude_self = exclude_self,
            n_threads = n_threads
        )
        if (!is.null(reason)) {
            attr(out, "spatial_index")$reason <- reason
        }
        return(out)
    }
    stop(
        "`backend = \"cpu_grid\"` supports only two- or three-column matrices.",
        call. = FALSE
    )
}

#' Nearest neighbors from row-wise matrices
#'
#' `nn()` provides a package-native nearest-neighbor entry point compatible with
#' the common `nn(data, points, k)` use case. The public API separates device
#' selection from algorithm selection. `backend` is either `"cpu"` or
#' `"cuda"`; `method` chooses the algorithm. For example,
#' `backend = "cpu", method = "grid"` uses the CPU grid implementation, while
#' `backend = "cuda", method = "grid"` uses the CUDA grid implementation.
#' Invalid combinations stop clearly before computation; for example,
#' `backend = "cpu", method = "cagra"` errors because CAGRA is CUDA-only.
#'
#' @details
#' The public methods are:
#' \itemize{
#'   \item `"auto"`: deterministic selection from the requested backend,
#'   metric, data shape, `k`, and target recall. It does not run a timing pilot.
#'   The CPU policy is calibration-informed and experimental: it has not
#'   received the independent installed-policy validation reported for CUDA.
#'   \item `"exact"`: exact search using FAISS Flat or direct cuVS brute force.
#'   \item `"flat"`: an explicit FAISS exhaustive Flat index.
#'   \item `"bruteforce"`: exhaustive search, using FAISS on CPU and preferring
#'   direct cuVS on CUDA when available.
#'   \item `"grid"`: exact 2D/3D self-KNN for Euclidean, cosine, or correlation.
#'   \item `"hnsw"`: FAISS HNSW on CPU; CUDA uses the cuVS HNSW wrapper that
#'   converts a CAGRA seed graph and is not a pure all-GPU HNSW search.
#'   \item `"ivf"`: FAISS IVF-Flat approximate search with coarse-list probing.
#'   \item `"ivfpq"`: compressed FAISS IVF-PQ approximate search.
#'   \item `"ivfpq_fastscan"`: FAISS FastScan on CPU and the package's
#'   separately identified cuVS 4-bit IVF-PQ route on CUDA.
#'   \item `"vamana_style"` (`"vamana"` compatibility alias): an experimental
#'   package-owned
#'   robust-pruned candidate graph inspired by DiskANN/Vamana, followed by CPU
#'   or CUDA candidate refinement. It is not a feature-complete Vamana
#'   reproduction.
#'   \item `"nsg_style"` (`"nsg"` compatibility alias): a distinct
#'   experimental package-owned candidate-graph algorithm derived from selected
#'   NSG/MRNG
#'   pruning ideas, followed by CPU or CUDA candidate refinement. It is not a
#'   feature-complete NSG reproduction.
#'   \item `"nndescent_style"` (`"nndescent"` compatibility alias): a
#'   distinct experimental package-owned CPU graph-refinement algorithm derived
#'   from
#'   NN-descent ideas, or direct external-provider cuVS NN-descent on CUDA.
#'   \item `"cagra"`: CUDA-only FAISS GPU CAGRA or direct cuVS CAGRA. Use
#'   `cagra_implementation` to request a provider explicitly.
#' }
#'
#' Exact-family names are distinct public policies even when a backend resolves
#' more than one of them to the same exhaustive provider. Result metadata
#' records both the requested method and the concrete provider. Unsupported
#' combinations fail instead of silently changing method, metric, or device.
#'
#' @references
#' Johnson J, Douze M, Jegou H. Billion-scale similarity search with GPUs. IEEE
#' Transactions on Big Data. 2021;7:535-547.
#'
#' Douze M, Guzhva A, Deng C, Johnson J, Szilvasy G, Mazaré PE, et al. The
#' FAISS library. arXiv 2024. See also the FAISS C++ API documentation.
#'
#' RAPIDS Development Team. RAPIDS cuVS: GPU-accelerated vector search and
#' clustering. \url{https://github.com/NVIDIA/cuvs}.
#'
#' Dong W, Moses C, Li K. Efficient k-nearest neighbor graph construction for
#' generic similarity measures. WWW 2011:577-586.
#'
#' Malkov YA, Yashunin DA. Efficient and robust approximate nearest neighbor
#' search using hierarchical navigable small world graphs. IEEE TPAMI.
#' 2020;42:824-836.
#'
#' Jégou H, Douze M, Schmid C. Product quantization for nearest neighbor
#' search. IEEE TPAMI. 2011;33:117-128.
#'
#' NVIDIA, Meta, and FAISS documentation for FAISS GPU indexes backed by NVIDIA
#' cuVS, including IVF and CAGRA integration.
#'
#' @param data Numeric matrix/data frame or optional `float::fl()`/`float32`
#'   object of reference observations in rows. FAISS CPU/GPU and RAPIDS cuVS
#'   nearest-neighbour routes use direct float-pointer adapters for float32
#'   inputs. Native routes without a direct float32 adapter fail clearly instead
#'   of silently converting benchmark input back to R double. Sparse, delayed,
#'   file-backed, and other matrix-like objects are rejected before conversion
#'   to prevent accidental dense materialization; compute a dense
#'   low-dimensional representation or explicitly materialize a manageable
#'   block first.
#' @param points Numeric matrix/data frame or optional `float::fl()`/`float32`
#'   query object with observations in rows. Defaults to `data`. A float32
#'   query can be paired with an ordinary R double reference matrix on direct
#'   FAISS/cuVS float32 routes.
#' @param k Number of neighbors to return. `NULL` chooses the package's
#'   automatic neighborhood size and includes the self-neighbor when `points`
#'   is `data`.
#' @param exclude_self Logical; when `TRUE`, remove each query row from its own
#'   neighbour list. This is valid only for self-query calls where `points` is
#'   omitted or identical to `data`. Self-neighbour removal is passed to the
#'   compiled backend path rather than repaired by R-side post-processing. CUDA
#'   graph routes that do not yet expose compiled include-self output shaping
#'   require `exclude_self = TRUE` and fail clearly instead of reshaping in R.
#' @param backend Requested execution device: `"cpu"` or `"cuda"`. `NULL`
#'   follows `options(faissR.backend)`, then `FAISSR_BACKEND`, with CPU as the
#'   final default. The historical result field `backend_used` instead names
#'   the concrete resolved provider/route and is retained for API
#'   compatibility. CUDA requests fail clearly when CUDA support or the
#'   selected CUDA combination is unavailable.
#' @param method Algorithm selector. `"auto"` chooses a shape-aware default for
#'   the selected backend. Other values include `"exact"`, `"flat"`,
#'   `"bruteforce"`, `"grid"`, `"hnsw"`, `"ivf"`,
#'   `"ivfpq"`, `"vamana_style"`, `"nsg_style"`,
#'   `"nndescent_style"`, `"ivfpq_fastscan"`, and `"cagra"`. The historical
#'   shorter spellings `"vamana"`, `"nsg"`, and `"nndescent"` remain accepted
#'   compatibility aliases. Use these public lowercase method labels; resolved
#'   implementation labels such as
#'  `"faiss_hnsw"` or `"cuda_cuvs_cagra"` are not public `method` values.
#'  Unsupported
#'   backend/method combinations fail clearly; for example,
#'   `method = "cagra", backend = "cpu"` errors because CAGRA is CUDA-only,
#'   and CUDA `method = "ivfpq_fastscan"` accepts Euclidean/L2, cosine, and
#'  correlation. CPU FastScan accepts the same metrics; cosine is implemented
#'  by row
#'   L2 normalization, correlation by row centering plus L2 normalization
#'   followed by FastScan L2 search.
#'   The preferred names for package-owned graph refinement are
#'   `"nsg_style"`, `"vamana_style"`, and `"nndescent_style"`; the shorter
#'   names remain compatibility aliases. The `_style` suffix is a scope
#'   marker, not an algorithm name: native results explicitly report a distinct
#'   package-owned derived graph-refinement algorithm rather than a
#'   feature-complete canonical reproduction. CUDA cuVS NN-descent is
#'   identified separately as an external-provider implementation. Package-owned
#'   routes carry `implementation_status = "experimental"` and
#'   `experimental = TRUE` in their returned metadata. They are available for
#'   explicit evaluation but are excluded from the package's principal
#'   comparative performance claims.
#' @param metric Distance metric. The intentionally small public set is
#'   `"euclidean"`, `"cosine"`, and `"correlation"`.
#'   Euclidean results are ordinary L2 distances, not squared L2 values.
#'   Legacy aliases such as `"l2"`, `"cor"`, `"pearson"`, and `"ip"` are
#'   rejected; use the canonical metric names. `"cosine"` uses row L2
#'   normalization, and `"correlation"` is centered cosine
#'   similarity after subtracting each row mean and L2-normalizing each row.
#'   `"euclidean"` is the validated high-performance default. `"cosine"` and
#'   `"correlation"` are implemented for exact CPU KNN, native 2D/3D grid
#'   search, FAISS CPU/GPU Flat,
#'   FAISS CPU/GPU IVF-Flat, FAISS CPU/GPU IVFPQ, FAISS CPU FastScan,
#'   CUDA cuVS IVFPQ FastScan for cosine and correlation,
#'   FAISS CPU HNSW, and native graph-refinement routes. Similarity-capable
#'   routes use row L2 normalization
#'   for cosine and row centering plus L2 normalization for correlation before
#'   search; distances are returned as `1 - similarity`.
#'   All-zero cosine rows and constant correlation rows are zero-normalized
#'   edge cases: two zero-normalized rows have distance `0`, while a
#'   zero-normalized row versus a nonzero row has distance `1`. CPU FAISS Flat
#'   uses the exact CPU scorer for those rows to preserve deterministic
#'   small-`k` tie handling; explicit CUDA routes error clearly instead of
#'   repairing those rows on CPU. These finite values are software conventions
#'   for otherwise undefined cosine or correlation cases, not mathematical
#'   cosine similarities or Pearson correlations. All backends reject rows
#'   containing `NA`, `NaN`, `Inf`, or `-Inf`. Use
#'   [nn_metric_preflight()] to obtain affected one-based row indices and the
#'   requested backend's action before search. CUDA FAISS/cuVS results carry
#'   `attr(result, "gpu_residency")` metadata with provider, index residency,
#'   host/device transfer strategy, query device reuse, and CPU fallback flags.
#'   CPU `method = "auto"` can use FAISS Flat for larger exact non-Euclidean
#'   query workloads, FAISS HNSW for large non-Euclidean self-search, native
#'   CPU NSG/Vamana refinement for selected larger self-KNN cases, and native
#'   CPU NN-descent for other large self-KNN cases. CPU `method = "hnsw"` uses
#'   FAISS HNSW for all three metrics. CUDA HNSW metadata records the available
#'   cuVS HNSW wrapper design.
#'   Unsupported method/backend/metric combinations fail clearly instead of
#'   changing the requested metric, method, or device. Use
#'   `nn_capabilities(runtime = TRUE)` to preflight both design support and the
#'   libraries available in the current installation.
#' @param tuning Tuning policy. `"auto"` uses deterministic compiled defaults
#'   selected from backend, method, metric, data shape, `k`, and
#'   `target_recall`; it does not run a timing pilot. `"cache"` reuses or stores
#'   pilot results, `"pilot"` tunes for the current call, `"fixed"` uses
#'   untuned fixed defaults while retaining metadata, and `"off"`/`"none"`
#'   disables tuning. Result metadata records the selected parameters, their
#'   provenance, and `tuning_benchmark_target_met` when a benchmark-derived row
#'   did not attain the requested target. Exact methods record recall 1 by
#'   construction; their tuning controls batching, copies, and resource reuse.
#' @param target_recall Requested recall tier: `0.9`, `0.95`, or `0.99`
#'   (default). Approximate methods use it to select a speed/quality policy;
#'   `method = "auto"` may also use it when choosing a method. It is a target,
#'   not a guarantee: inspect `tuning_benchmark_target_met` and measured recall.
#'   Other numeric values are rejected; faissR does not round or interpolate
#'   between the three calibrated tiers.
#'   Exact methods remain exact and use this value only as policy metadata.
#' @param cagra_implementation CUDA CAGRA provider for this call. `NULL` uses
#'   the global `options(faissR.cagra_implementation = ...)` value. `"auto"`
#'   uses a deterministic provider rule: compact high-dimensional self-KNN
#'   selects direct RAPIDS cuVS CAGRA when both providers are available, while
#'   FAISS GPU CAGRA remains the default for other shapes. `"faiss_gpu"` and
#'   `"cuvs"` force one provider for benchmarking.
#'   This argument affects only public `backend = "cuda", method = "cagra"`
#'   requests and CUDA `method = "auto"` routes that select CAGRA.
#' @param cagra_build_algo Direct RAPIDS cuVS CAGRA graph-build algorithm for
#'   this call. `NULL` uses `options(faissR.cuvs_cagra_build_algo = "auto")`.
#'   For direct cuVS CAGRA, `"auto"` applies faissR's deterministic shape-aware
#'   CAGRA build rule, choosing iterative CAGRA construction for compact
#'   high-dimensional self-KNN cases and IVF-PQ construction otherwise.
#'   `"ivf_pq"` requests the IVF-PQ graph builder, `"nn_descent"` requests cuVS
#'   NN-descent graph construction, and `"iterative_cagra_search"` requests cuVS
#'   iterative CAGRA graph building.
#'   This is a CAGRA construction parameter, not a fallback to a different
#'   public method; successful results record the selected value in
#'   `attr(result, "approximation")$cagra_build_algo`.
#' @param output Distance storage type for the returned object. `"double"`
#'   returns the default R numeric distance matrix. `"float"` returns
#'   `distances` as a `float::fl()`/`float32` object and records
#'   `distance_type = "float32"` plus
#'   `attr(result, "distance_type") = "float32"`; this requires the optional
#'   `float` package. When either `data` or `points` is a `float::fl()` matrix,
#'  FAISS Flat/IVF/IVFPQ/HNSW/FastScan/NSG/NNDescent, FAISS GPU
#'  Flat/IVF/IVFPQ/CAGRA,
#'   and RAPIDS cuVS brute-force/CAGRA/IVF/IVFPQ/NN-descent routes consume
#'   float32 input directly through C++ adapters. Methods without a direct
#'   float32 adapter now error instead of silently converting the benchmark
#'   input back to R double. Ordinary R double inputs with `output = "float"`
#'   also enter float-pointer routes for CPU FAISS Flat/IVF/IVFPQ/FastScan,
#'   cached CPU FAISS fitted indexes, FAISS GPU Flat/IVF/IVFPQ, and direct
#'  Euclidean RAPIDS cuVS brute-force/CAGRA/IVF/IVFPQ and IVFPQ FastScan
#'  routes.
#'   On direct FAISS/cuVS float routes, float distance output is constructed
#'   directly from backend float results instead of first materializing an R
#'   double distance matrix, except for routes that need R-side metric
#'   transformation or zero-row cosine/correlation correction.
#' @param distances Optional alias for `output`, kept for callers that prefer
#'   `distances = "double"` or `distances = "float"` to describe the returned
#'   distance storage type.
#' @param n_threads Number of CPU worker threads for CPU backends. GPU backends
#'   ignore this argument.
#' @return A list with integer matrix `indices`, `distances`, and stable
#'   metadata fields `index_base`, `distance_type`, `metric`, and
#'   `backend_used`. Every result declares `distance_is_metric`,
#'   `distance_semantics`, `distance_comparable_across_queries`, and
#'   `distance_order`.
#'   Float32 routes also record `input_layout` and
#'   `input_owns_data` so downstream packages can distinguish direct float32
#'   payload use from one-time row-major conversion. Normalized Euclidean graph
#'   routes for cosine/correlation record `metric_transform` and
#'   `attr(result, "distance_transform")`. Indices are 1-based signed R
#'   integers; CUDA result identifiers are signed int32. The public identifier
#'   space is therefore limited to `.Machine$integer.max` reference rows,
#'   although provider and memory limits are usually much lower. The
#'   requested backend/method, tuning policy, resolved
#'   backend, metric, exact/approximate flag, and self-query flag are stored in
#'   attributes including `attr(result, "requested_backend")`,
#'   `attr(result, "requested_method")`, `attr(result, "tuning")`, and
#'   `attr(result, "resolved_backend")`. `method = "auto"` requests also include
#'   `attr(result, "auto_selection")`, a static workload/shape/k/metric decision
#'   record. It includes reference rows and variables, query-row count,
#'   self-query status, the `n * n_points * p` work estimate, and the predicted
#'   internal backend, public method class, device
#'   class, explicit backend/method flags, backend/method decision reasons, and
#'   hardware provenance. CPU `method = "auto"`
#'   records `auto_policy_status =
#'   "calibration_informed_not_independently_validated"` and
#'   `auto_policy_evidence_scope = "cpu_static_policy_experimental"`. CUDA
#'   `method = "auto"` is a separately evaluated but still experimental
#'   L40S-calibrated policy for cold full-self-search and records
#'   `auto_policy_status =
#'   "experimental_l40s_calibrated_cold_full_self_search"`; neither label
#'   implies general workload or hardware validation.
#'   Expected future query batches are not an input; use a
#'   fitted or cached index when construction will be amortized. A
#'   capability-compatible machine that differs from the calibration hardware
#'   keeps the compiled policy but is labelled
#'   `hardware_extrapolated_unvalidated`. If either hardware identity cannot be
#'   determined, the label is `hardware_unidentified`. Hardware identity alone
#'   never causes a silent method or device fallback. CUDA method selection
#'   emits one
#'   warning per confirmed, unmatched runtime GPU model unless
#'   `options(faissR.warn_hardware_extrapolation = FALSE)` is set. Pilot/cache
#'   tuning adjusts parameters within a method and does not install a new
#'   cross-method policy. The selector does not run pilot
#'   tuning. CPU FAISS Flat/HNSW/IVF/IVFPQ/FastScan routes use a
#'   bounded session-local fitted-index cache for repeated raw `nn()` calls
#'   with matching data and parameters. CUDA `method = "ivfpq_fastscan"` also
#'  reuses a fitted cuVS IVF-PQ index, dataset device buffer, and cuVS
#'  resources
#'   through a separate bounded cache. Self-query uses the fitted dataset device
#'  buffer directly, and repeated separate-query calls can reuse one cached
#'  query
#'   device buffer with
#'   `options(faissR.cache_cuda_ivfpq_query_buffers = TRUE)`. Metadata reports
#'   `persistent_index_cache`, `index_cache_hit`, `dataset_residency`,
#'   `query_residency`, and `query_host_to_device_copies`. Disable CPU and CUDA
#'   fitted-index caches with
#'   `options(faissR.cache_fitted_nn_indexes = FALSE)`, bound CPU memory with
#'   `options(faissR.cache_fitted_nn_indexes_max_entries = <n>)`, or bound CUDA
#'   FastScan GPU memory with
#'   `options(faissR.cache_fitted_cuda_ivfpq_indexes_max_entries = <n>)`.
#'   CPU Euclidean, cosine, and correlation
#'   `method = "ivfpq_fastscan"` resolve `tuning = "auto"` in C++ from
#'   shape/k/target-recall defaults for `nlist`, `nprobe`, `pq_m`,
#'   `refine_factor`, and FastScan block size. Cosine uses row L2 normalization
#'  and correlation uses row centering plus L2 normalization before FastScan
#'  L2.
#'  CUDA FastScan cosine and correlation auto policies are seeded from the CUDA
#'  Euclidean
#'   FastScan table until metric-specific HPC sweeps replace them, and those
#'   rows report `tuning_benchmark_target_met = FALSE`.
#' @examples
#' x <- scale(as.matrix(iris[, 1:4]))
#' knn_euclidean <- nn(x, k = 16, metric = "euclidean", backend = "cpu")
#' knn_cosine <- nn(x, k = 16, metric = "cosine", backend = "cpu")
#' knn_correlation <- nn(x, k = 16, metric = "correlation", backend = "cpu")
#'
#' if (faiss_available() && requireNamespace("Biobase", quietly = TRUE)) {
#'     data("sample.ExpressionSet", package = "Biobase")
#'     expression_data <- Biobase::exprs(sample.ExpressionSet)
#'     x_biobase <- scale(t(expression_data[seq_len(32L), , drop = FALSE]))
#'     expression_knn <- nn(
#'         x_biobase,
#'         k = 3,
#'         backend = "cpu",
#'         method = "exact",
#'         exclude_self = TRUE
#'     )
#'     dim(expression_knn$indices)
#' }
#' @export
nn <- function(
    data,
    points = data,
    k = NULL,
    exclude_self = FALSE,
    backend = NULL,
    method = c(
        "auto", "exact", "flat", "bruteforce", "grid", "hnsw", "ivf",
        "ivfpq", "vamana_style", "nsg_style", "nndescent_style",
        "ivfpq_fastscan", "cagra"
    ),
    metric = c("euclidean", "cosine", "correlation"),
    tuning = c("auto", "cache", "pilot", "fixed", "off", "none"),
    target_recall = 0.99,
    cagra_implementation = NULL,
    cagra_build_algo = NULL,
    output = c("double", "float"),
    distances = NULL,
    n_threads = NULL
) {
    if (missing(method)) method <- "auto"
    if (missing(metric)) metric <- "euclidean"
    if (missing(tuning)) tuning <- "auto"
    set_call_cagra_implementation(cagra_implementation)
    set_call_cagra_build_algo(cagra_build_algo)
    request <- prepare_public_nn_request(
        data, points, k, exclude_self, backend, method, metric, tuning,
        target_recall, output, distances, n_threads, missing(points),
        trimws(as.character(method)[1L])
    )
    execute_public_nn_request(request)
}

#' GPU-resident tuned nearest-neighbour search
#'
#' `nn_gpu()` runs a CUDA nearest-neighbour search and returns a
#' GPU-resident result object. Unlike `nn()`, the `indices` and `distances`
#' are not copied back into R matrices. The returned object stores owning and
#' non-owning external pointers so downstream C/C++ packages can consume the
#' KNN output on the CUDA device.
#'
#' @details
#' This is intentionally narrower than `nn()`: for Euclidean
#' `method = "auto"`, `"exact"`, `"flat"`, or `"bruteforce"`, `nn_gpu()`
#' uses a FAISS GPU direct `bfKnn` route and keeps the result buffers on the
#' CUDA device when FAISS GPU is available. Euclidean inputs with two or three
#' columns instead use the native direct-difference CUDA exact kernel. This
#' avoids cancellation in the dot-product L2 identity for nearly coincident
#' low-dimensional vectors while retaining GPU-resident results. When FAISS
#' was built with cuVS
#' support, FAISS may dispatch the brute-force GPU distance primitive through
#' cuVS internally.
#' Cosine and correlation currently keep using the native CUDA GPU-resident
#' exact route; they are transformed to normalized squared L2 on the C++ side
#' and stored as `1 - similarity`. Approximate cuVS/FAISS GPU methods currently
#' fail clearly here because those provider result buffers still need
#' provider-specific persistent GPU ownership.
#' For `method = "auto"`, `nn_gpu()` records the same C++ auto-selection
#' metadata
#' used by `nn()`. If that policy would prefer an approximate CUDA method whose
#' result buffers are not yet exposed as persistent GPU-resident objects,
#' `nn_gpu()` keeps the exact-family GPU-resident route and records
#' `auto_preferred_backend`, `auto_preferred_method`, and
#' `auto_residency_constraint`. It also records the actual exact-family
#' `execution_tuning` used by the GPU-resident route and the
#' `auto_preferred_tuning` row for the approximate method selected by the
#' compiled policy when available. That policy is an optional experimental
#' L40S-calibrated heuristic for cold full-self-search. Because it is
#' L40S-informed,
#' unmatched or unidentified CUDA hardware emits the same once-per-model
#' extrapolation warning as `nn()`; set
#' `options(faissR.warn_hardware_extrapolation = FALSE)` only after reviewing
#' the returned `auto_selection` metadata.
#'
#' @param data Dense numeric matrix/data frame or optional
#'   `float::fl()`/`float32` reference matrix. Sparse, delayed, file-backed, and
#'   other matrix-like objects are rejected before conversion.
#' @param points Optional query matrix. Defaults to `data`.
#' @param k Number of neighbours.
#' @param exclude_self Logical; remove each row from its own neighbour list for
#'   self-query calls.
#' @param method `"auto"`, `"exact"`, `"flat"`, or `"bruteforce"`. `"auto"`
#'   consults the compiled shape/k/metric/target-recall selector but currently
#'   returns GPU-resident exact-family buffers.
#' @param metric `"euclidean"`, `"cosine"`, or `"correlation"`.
#' @param tuning Tuning label to record. The current GPU-resident route is
#'   exact,
#'   so `target_recall` is metadata for the executed route; auto-selected
#'   approximate settings are recorded separately as `auto_preferred_tuning`.
#' @param target_recall Target recall label to record; use `0.9`, `0.95`, or
#'   `0.99`.
#' @return A `faissR_gpu_knn` list. It contains an owning `handle`, non-owning
#'   `indices_ptr` and `distances_ptr`, `n_query`, `k`, `index_base = 1L`,
#'   `indices_type = "int32"`, `distance_type = "float32"`, `layout`, `metric`,
#'   and `backend_used`. The object also declares the same distance-contract
#'   fields as `nn()`. With `tuning = "auto"`, it also includes
#'   `execution_tuning`; with `method = "auto"`, it includes
#'   `auto_preferred_tuning` when the compiled selector has a preferred CUDA
#'   method/tuning row. Ownership metadata states that `handle` owns both
#'   allocations, the buffer pointers are non-owning, the producing device is
#'   synchronized before return, and serialization/interprocess sharing are not
#'   supported.
#' @examples
#' if (cuda_available()) {
#'   x <- matrix(rnorm(200), ncol = 4)
#'   gpu_knn <- nn_gpu(x, k = 5, exclude_self = TRUE)
#'   print(gpu_knn)
#' }
#' @export
nn_gpu <- function(
    data,
    points = data,
    k = NULL,
    exclude_self = FALSE,
    method = c("auto", "exact", "flat", "bruteforce"),
    metric = c("euclidean", "cosine", "correlation"),
    tuning = c("auto", "cache", "pilot", "fixed", "off", "none"),
    target_recall = 0.99
) {
    if (missing(method)) {
        method <- "auto"
    }
    if (missing(metric)) {
        metric <- "euclidean"
    }
    if (missing(tuning)) {
        tuning <- "auto"
    }
    request <- prepare_nn_gpu_request(
        data = data,
        points = points,
        k = k,
        exclude_self = exclude_self,
        method = method,
        metric = metric,
        tuning = tuning,
        target_recall = target_recall,
        points_missing = missing(points)
    )
    plan <- resolve_nn_gpu_plan(request)
    finish_nn_gpu_result(execute_nn_gpu_plan(request, plan), request, plan)
}

#' Copy a GPU-resident KNN result to host matrices
#'
#' `gpu_knn_to_host()` is an explicit diagnostic/conversion helper for
#' `faissR_gpu_knn` objects. It copies device `indices` and `distances` into
#' ordinary R matrices. It is never called automatically by `nn_gpu()`.
#'
#' @param x A `faissR_gpu_knn` object.
#' @return A host `faissR_nn` list with integer `indices` and numeric
#'   `distances`. The distance-contract metadata from the GPU object is
#'   retained.
#' @examples
#' if (cuda_available()) {
#'   x <- matrix(rnorm(200), ncol = 4)
#'   gpu_knn <- nn_gpu(x, k = 5, exclude_self = TRUE)
#'   host_knn <- gpu_knn_to_host(gpu_knn)
#'   str(host_knn)
#' }
#' @export
gpu_knn_to_host <- function(x) {
    if (!inherits(x, "faissR_gpu_knn")) {
        stop("`x` must be a `faissR_gpu_knn` object.", call. = FALSE)
    }
    out <- gpu_knn_to_host_cpp(x)
    attach_nn_distance_contract(
        out,
        x$metric %||% attr(x, "metric") %||% "euclidean"
    )
}

#' @export
print.faissR_gpu_knn <- function(x, ...) {
    cat("<faissR_gpu_knn>\n")
    cat("  backend: ", x$backend_used %||% NA_character_, "\n", sep = "")
    cat("  metric:  ", x$metric %||% NA_character_, "\n", sep = "")
    cat(
        "  shape:   ",
        as.integer(x$n_query %||% NA_integer_),
        " x ",
        as.integer(x$k %||% NA_integer_),
        "\n",
        sep = ""
    )
    cat("  result:  CUDA device pointers (indices int32, distances float32)\n")
    if (identical(x$metric %||% attr(x, "metric"), "inner_product")) {
        cat(
            "  values:  query-specific shifted inner product; not a metric or ",
            "cross-query comparable\n",
            sep = ""
        )
    }
    invisible(x)
}

.knn_recall_summary <- function(approx, exact, k = NULL) {
    approx_idx <- if (is.list(approx)) approx$indices else approx
    exact_idx <- if (is.list(exact)) exact$indices else exact
    approx_idx <- as.matrix(approx_idx)
    exact_idx <- as.matrix(exact_idx)
    if (nrow(approx_idx) != nrow(exact_idx)) {
        stop(
            "Approximate and exact KNN must have the same number of rows.",
            call. = FALSE
        )
    }
    k_is_auto <- is.null(k)
    k <- if (k_is_auto) {
        min(ncol(approx_idx), ncol(exact_idx))
    } else {
        normalize_nn_positive_integer(k, "k", "`k` must be a positive integer.")
    }
    k <- min(k, ncol(approx_idx), ncol(exact_idx))
    if (k < 1L) {
        stop(
            "KNN matrices must have at least one neighbour column.",
            call. = FALSE
        )
    }
    recall <- numeric(nrow(approx_idx))
    for (i in seq_len(nrow(approx_idx))) {
        approx_row <- approx_idx[i, seq_len(k)]
        exact_row <- exact_idx[i, seq_len(k)]
        approx_row <- approx_row[!is.na(approx_row) & is.finite(approx_row)]
        exact_row <- exact_row[!is.na(exact_row) & is.finite(exact_row)]
        recall[[i]] <- if (length(exact_row)) {
            sum(approx_row %in% exact_row) / length(exact_row)
        } else {
            NA_real_
        }
    }
    recall <- recall[is.finite(recall)]
    data.frame(
        k = k,
        recall_at_k = if (length(recall)) mean(recall) else NA_real_,
        median_recall_at_k = if (length(recall)) median(recall) else NA_real_,
        min_recall_at_k = if (length(recall)) min(recall) else NA_real_,
        stringsAsFactors = FALSE
    )
}

drop_self_knn_result <- function(raw, k) {
    indices <- raw$indices
    distances <- raw$distances
    if (!is.matrix(indices)) {
        indices <- as.matrix(indices)
    }
    if (!is.matrix(distances)) {
        distances <- as.matrix(distances)
    }
    if (!is.integer(indices)) {
        storage.mode(indices) <- "integer"
    }
    if (!identical(typeof(distances), "double")) {
        storage.mode(distances) <- "double"
    }
    keep <- matrix(1L, nrow(indices), k)
    keep_dist <- matrix(0, nrow(indices), k)
    for (i in seq_len(nrow(indices))) {
        row_keep <- which(
            indices[i, ] != i | distances[i, ] > sqrt(.Machine$double.eps)
        )
        if (length(row_keep) < k) {
            row_keep <- seq_len(ncol(indices))
        }
        row_keep <- row_keep[seq_len(k)]
        keep[i, ] <- indices[i, row_keep]
        keep_dist[i, ] <- distances[i, row_keep]
    }
    list(indices = keep, distances = keep_dist)
}

#' @export
print.faissR_nn <- function(x, ...) {
    cat("faissR KNN\n")
    cat("  queries: ", nrow(x$indices), "\n", sep = "")
    cat("  neighbors: ", ncol(x$indices), "\n", sep = "")
    cat("  backend: ", attr(x, "backend"), "\n", sep = "")
    metric <- attr(x, "metric")
    if (!is.null(metric) && !is.na(metric)) {
        cat("  metric: ", metric, "\n", sep = "")
    }
    print_nn_auto_status(x)
    print_nn_implementation_status(x)
    print_nn_approximation_status(x)
    if (
        isTRUE(attr(x, "self_query")) &&
            !isTRUE(x$exclude_self %||% attr(x, "exclude_self"))
    ) {
        cat("  first column: self-neighbor\n")
    }
    invisible(x)
}

print_nn_auto_status <- function(x) {
    auto_status <- attr(x, "auto_policy_status")
    if (
        length(auto_status) == 1L &&
            !is.na(auto_status) &&
            nzchar(auto_status)
    ) {
        cat("  auto policy: ", auto_status, "\n", sep = "")
    }
    invisible(NULL)
}

print_nn_implementation_status <- function(x) {
    if (
        !is.null(x$implementation_label) &&
            identical(
                x$implementation_scope,
                "package_owned_style_implementation"
            )
    ) {
        cat("  implementation: ", x$implementation_label, "\n", sep = "")
        if (identical(x$implementation_status, "experimental")) {
            cat("  status: experimental\n")
        }
        cat("  canonical reproduction: no\n")
    }
    invisible(NULL)
}

print_nn_approximation_status <- function(x) {
    if (!isTRUE(attr(x, "exact"))) {
        cat("  exact: false\n")
        recall <- attr(x, "recall")
        if (
            is.data.frame(recall) &&
                nrow(recall) >= 1L &&
                is.finite(recall$recall_at_k[1L])
        ) {
            cat(
                "  recall@",
                recall$k[1L],
                " on exact subset: ",
                formatC(recall$recall_at_k[1L], digits = 3, format = "f"),
                " (n=",
                recall$sample_size[1L],
                ")\n",
                sep = ""
            )
        }
    }
    invisible(NULL)
}

#' Check whether the native CUDA backend is available
#'
#' @return `TRUE` when the package was built with CUDA support and the CUDA
#'   runtime reports at least one available device.
#' @examples
#' cuda_available()
#' @export
cuda_available <- function() {
    isTRUE(cuda_available_cpp())
}

#' Check whether the real FAISS C++ backend is available
#'
#' @return `TRUE` when faissR was compiled and linked against FAISS.
#' @examples
#' faiss_available()
#' @export
faiss_available <- function() {
    isTRUE(faiss_available_cpp())
}

faiss_fastscan_available <- function() {
    info <- tryCatch(faiss_info_json_cpp(), error = function(e) NA_character_)
    isTRUE(json_get_bool(info, "fastscan"))
}

#' Check whether the RAPIDS cuVS backend is available
#'
#' @return `TRUE` when faissR was compiled and linked against RAPIDS cuVS
#'   and the CUDA runtime reports at least one available device.
#' @examples
#' cuvs_available()
#' @export
cuvs_available <- function() {
    isTRUE(cuvs_available_cpp())
}
