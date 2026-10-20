coerce_knn_input <- function(indices, distances = NULL, arg_name = "indices") {
    input <- unpack_knn_input(indices, distances, arg_name)
    matrices <- normalize_knn_matrices(input$indices, input$distances)
    validate_knn_matrix_pair(matrices$indices, matrices$distances)
    stripped <- validated_nonself_knn(matrices$indices, matrices$distances)
    list(
        indices = stripped$indices,
        distances = stripped$distances,
        has_self = isTRUE(stripped$has_self),
        col_start = as.integer(stripped$col_start),
        n_neighbors = as.integer(stripped$n_neighbors),
        materialized = isTRUE(stripped$materialized),
        input_backend = if (is.null(input$backend)) {
            NA_character_
        } else {
            input$backend
        }
    )
}

unpack_knn_input <- function(indices, distances, arg_name) {
    if (!is.null(distances)) {
        return(list(
            indices = indices,
            distances = distances,
            backend = NA_character_
        ))
    }
    required <- is.list(indices) &&
        all(c("indices", "distances") %in% names(indices))
    if (!required) {
        stop(
            "`distances` is required unless `", arg_name,
            "` is an `nn()` result with `indices` and `distances`.",
            call. = FALSE
        )
    }
    list(
        indices = indices$indices,
        distances = indices$distances,
        backend = attr(indices, "resolved_backend") %||%
            attr(indices, "backend")
    )
}

normalize_knn_matrices <- function(indices, distances) {
    indices <- if (is.matrix(indices)) indices else as.matrix(indices)
    distances <- if (is.matrix(distances)) distances else as.matrix(distances)
    if (!is.integer(indices)) storage.mode(indices) <- "integer"
    if (!identical(typeof(distances), "double")) {
        storage.mode(distances) <- "double"
    }
    list(indices = indices, distances = distances)
}

validate_knn_matrix_pair <- function(indices, distances) {
    if (!identical(dim(indices), dim(distances))) {
        stop(
            "KNN `indices` and `distances` must have the same dimensions.",
            call. = FALSE
        )
    }
    if (nrow(indices) < 2L || ncol(indices) < 1L) {
        stop(
            "KNN input must have at least two rows and one neighbor column.",
            call. = FALSE
        )
    }
    invisible(TRUE)
}

validated_nonself_knn <- function(indices, distances) {
    stripped <- strip_self_neighbors_cpp(indices, distances)
    if (stripped$n_neighbors < 1L) {
        stop(
            "KNN input must contain at least one non-self neighbor.",
            call. = FALSE
        )
    }
    stripped
}

is_knn_input <- function(x) {
    is.list(x) && all(c("indices", "distances") %in% names(x))
}

knn_has_self_column <- function(indices, distances) {
    strip_self_neighbors_cpp(indices, distances)$has_self
}

prepend_self_neighbor_column <- function(out) {
    if (!is.list(out) || is.null(out$indices) || is.null(out$distances)) {
        stop(
            "KNN result must contain `indices` and `distances`.",
            call. = FALSE
        )
    }
    distances_are_float32 <- inherits(out$distances, "float32")
    distances <- if (isTRUE(distances_are_float32)) {
        float32_to_numeric_matrix(out$distances, "distances")
    } else {
        out$distances
    }
    prepended <- prepend_self_neighbors_cpp(out$indices, distances)
    out$indices <- prepended$indices
    out$distances <- if (isTRUE(distances_are_float32)) {
        as_float_distances(prepended$distances)
    } else {
        prepended$distances
    }
    out
}

set_embedding_colnames <- function(layout, prefix) {
    if (!is.matrix(layout)) {
        layout <- as.matrix(layout)
    }
    colnames(layout) <- paste0(prefix, seq_len(ncol(layout)))
    layout
}

validate_n_components <- function(n_components) {
    n_components <- as.integer(n_components)
    if (
        length(n_components) != 1L ||
            is.na(n_components) ||
            !is.finite(n_components) ||
            n_components < 1L
    ) {
        stop("`n_components` must be a positive integer.", call. = FALSE)
    }
    n_components
}

finish_nn_result <- function(
    out,
    backend,
    k,
    self_query,
    exact = TRUE,
    metric = "euclidean"
) {
    out$index_base <- out$index_base %||% 1L
    out$metric <- metric
    out$backend_used <- out$backend_used %||% backend
    out$exact <- isTRUE(exact)
    out$self_query <- isTRUE(self_query)
    attr(out, "backend") <- backend
    attr(out, "resolved_backend") <- backend
    attr(out, "k") <- as.integer(k)
    attr(out, "self_query") <- isTRUE(self_query)
    attr(out, "exact") <- out$exact
    attr(out, "metric") <- metric
    attr(out, "index_base") <- as.integer(out$index_base)
    attr(out, "backend_used") <- out$backend_used
    class(out) <- c("faissR_nn", "list")
    out <- attach_nn_distance_contract(out, metric)
    attach_gpu_residency_metadata(out, out)
}

nn_distance_contract <- function(metric) {
    metric <- normalize_nn_metric(metric)
    switch(
        metric,
        euclidean = list(
            distance_is_metric = TRUE,
            distance_semantics = "euclidean_distance",
            distance_comparable_across_queries = TRUE
        ),
        cosine = list(
            distance_is_metric = FALSE,
            distance_semantics = "one_minus_cosine_similarity",
            distance_comparable_across_queries = TRUE
        ),
        correlation = list(
            distance_is_metric = FALSE,
            distance_semantics = "one_minus_row_centered_cosine_similarity",
            distance_comparable_across_queries = TRUE
        )
    )
}

attach_nn_distance_contract <- function(result, metric) {
    contract <- nn_distance_contract(metric)
    for (name in names(contract)) {
        result[[name]] <- contract[[name]]
        attr(result, name) <- contract[[name]]
    }
    result$distance_order <- "smaller_is_better"
    attr(result, "distance_order") <- "smaller_is_better"
    result
}

nn_method_implementation_contract <- function(requested_method, backend_used) {
    requested_method <- normalize_nn_method(requested_method %||% "auto")
    backend_used <- as.character(backend_used %||% NA_character_)[1L]
    method <- if (identical(requested_method, "auto")) {
        nn_resolved_backend_public_method(backend_used)
    } else {
        requested_method
    }
    switch(
        method,
        nsg = nn_style_implementation_contract(
            "nsg",
            "nsg_style",
            "NSG/MRNG-derived refinement",
            "NSG"
        ),
        vamana = nn_style_implementation_contract(
            "vamana",
            "vamana_style",
            "Vamana-derived robust pruning",
            "DiskANN/Vamana"
        ),
        nndescent = nndescent_implementation_contract(backend_used),
        NULL
    )
}

nn_style_implementation_contract <- function(
    family,
    public_method,
    label,
    reference
) {
    list(
        method_family = family,
        preferred_public_method = public_method,
        implementation_label = paste("faissR package-owned", label),
        implementation_scope = "package_owned_style_implementation",
        implementation_status = "experimental",
        experimental = TRUE,
        principal_evidence_scope = "excluded_from_principal_performance_claims",
        canonical_reimplementation = FALSE,
        canonical_reference_name = reference
    )
}

nndescent_implementation_contract <- function(backend_used) {
    provider_routes <- c(
        "cuda_nndescent",
        "cuda_cuvs_nndescent",
        "cuvs_nndescent",
        "faiss_nndescent",
        "cpu_nndescent_faiss_nndescent"
    )
    if (backend_used %in% provider_routes) {
        provider <- if (grepl("cuvs|cuda", backend_used)) {
            "RAPIDS cuVS"
        } else {
            "FAISS"
        }
        return(list(
            method_family = "nndescent",
            preferred_public_method = "nndescent_style",
            implementation_label = paste(provider, "NN-descent provider route"),
            implementation_scope = "external_provider_implementation",
            implementation_status = "external_provider",
            experimental = FALSE,
            principal_evidence_scope = "provider_route_evidence_required",
            canonical_reimplementation = NA,
            canonical_reference_name = "NN-descent"
        ))
    }
    nn_style_implementation_contract(
        "nndescent",
        "nndescent_style",
        "NN-descent-derived graph construction",
        "NN-descent"
    )
}

attach_nn_method_implementation_contract <- function(
    result,
    requested_method,
    backend_used
) {
    contract <- nn_method_implementation_contract(
        requested_method,
        backend_used
    )
    if (is.null(contract)) {
        return(result)
    }
    for (name in names(contract)) {
        result[[name]] <- contract[[name]]
        attr(result, name) <- contract[[name]]
    }
    result
}

.nn_gpu_residency_fields <- c(
        "accelerator",
        "gpu_provider",
        "device_residency",
        "index_residency",
        "dataset_residency",
        "gpu_index_resident",
        "gpu_index_persistent",
        "gpu_resources_reused",
        "gpu_resource_scope",
        "host_to_device_transfer_strategy",
        "host_to_device_copies_known",
        "host_to_device_data_copies",
        "host_to_device_query_copies",
        "host_to_device_copies",
        "host_to_device_data_copies_minimum",
        "host_to_device_query_copies_minimum",
        "host_to_device_data_bytes",
        "host_to_device_query_bytes",
        "host_to_device_bytes",
        "host_to_device_data_bytes_minimum",
        "host_to_device_query_bytes_minimum",
        "host_to_device_bytes_minimum",
        "query_reuses_host_data",
        "query_reuses_device_data",
        "query_residency",
        "result_residency",
        "query_uses_index_dataset_buffer",
        "query_device_buffer_cached",
        "query_device_buffer_reused",
        "query_device_cache_status",
        "query_host_to_device_copies",
        "index_build_host_to_device_copies",
        "query_upload_count",
        "query_cache_hit_count",
        "host_device_traffic_policy",
        "device_to_host_result_copies_known",
        "device_to_host_result_copies",
        "device_to_host_result_bytes",
        "device_to_host_result_bytes_minimum",
        "cpu_fallback",
        "cpu_side_result_repair"
)

nn_gpu_residency_metadata <- function(out) {
    if (!is.list(out)) {
        return(NULL)
    }
    fields <- .nn_gpu_residency_fields[
        .nn_gpu_residency_fields %in% names(out)
    ]
    if (!length(fields)) {
        return(NULL)
    }
    out[fields]
}

attach_gpu_residency_metadata <- function(result, out) {
    gpu <- nn_gpu_residency_metadata(out)
    if (!length(gpu)) {
        return(result)
    }
    result$gpu_residency <- gpu
    attr(result, "gpu_residency") <- gpu
    for (attr_name in c("faiss", "cuvs", "approximation")) {
        metadata <- attr(result, attr_name, exact = TRUE)
        if (is.list(metadata)) {
            metadata <- c(metadata, gpu[setdiff(names(gpu), names(metadata))])
            attr(result, attr_name) <- metadata
        }
    }
    result
}

normalize_nn_output <- function(output) {
    output <- normalize_scalar_choice_arg(
        output,
        arg = "output",
        default = "double",
        formal_choices = c("double", "float")
    )
    if (is.na(output) || !nzchar(output)) {
        output <- "double"
    }
    output <- tolower(trimws(output))
    if (!output %in% c("double", "float")) {
        stop("`output` must be one of \"double\" or \"float\".", call. = FALSE)
    }
    output
}

resolve_nn_output <- function(output, distances = NULL) {
    output <- normalize_nn_output(output)
    if (is.null(distances)) {
        return(output)
    }
    distances <- normalize_scalar_choice_arg(
        distances,
        arg = "distances",
        default = "double",
        formal_choices = c("double", "float")
    )
    if (is.na(distances) || !nzchar(distances)) {
        distances <- "double"
    }
    distances <- tolower(trimws(distances))
    if (!distances %in% c("double", "float")) {
        stop(
            "`distances` must be one of \"double\" or \"float\".",
            call. = FALSE
        )
    }
    if (!identical(output, "double") && !identical(output, distances)) {
        stop(
            "`output` and `distances` request different ",
            "distance storage types.",
            call. = FALSE
        )
    }
    distances
}

is_float32_matrix_input <- function(x) {
    inherits(x, "float32")
}

validate_dense_matrix_input <- function(x, arg_name = "data") {
    if (is_float32_matrix_input(x) || is.matrix(x)) {
        if (!is_float32_matrix_input(x) && !is.numeric(x)) {
            stop(
                "`", arg_name, "` must be a dense numeric matrix.",
                call. = FALSE
            )
        }
        return(invisible(TRUE))
    }
    if (is.data.frame(x)) {
        numeric_columns <- vapply(x, is.numeric, logical(1L))
        if (!all(numeric_columns)) {
            stop(
                "Every column of `", arg_name,
                "` must be numeric.",
                call. = FALSE
            )
        }
        return(invisible(TRUE))
    }

    class_label <- paste(class(x), collapse = "/")
    if (!nzchar(class_label)) {
        class_label <- typeof(x)
    }
    stop(
        "`", arg_name, "` is a ", class_label,
        " object. faissR accepts dense numeric matrices, numeric data ",
        "frames, and float::fl() matrices only. Sparse, delayed, ",
        "and file-backed objects are not converted implicitly because ",
        "`as.matrix()` can allocate a dense copy. Compute a size-checked ",
        "dense representation such as PCA scores, or explicitly materialize ",
        "a manageable block before calling faissR.",
        call. = FALSE
    )
}

float32_matrix_dims <- function(x, arg_name = "data") {
    d <- dim(x)
    if ((is.null(d) || length(d) != 2L) && isS4(x)) {
        payload <- tryCatch(methods::slot(x, "Data"), error = function(e) NULL)
        if (!is.null(payload)) {
            d <- dim(payload)
        }
    }
    if (is.null(d) || length(d) != 2L) {
        stop(
            "`",
            arg_name,
            "` must be a two-dimensional float::fl()/float32 matrix.",
            call. = FALSE
        )
    }
    d <- as.integer(d)
    if (anyNA(d) || any(d < 1L)) {
        stop(
            "`",
            arg_name,
            "` must have at least one row and one column.",
            call. = FALSE
        )
    }
    d
}

float32_to_numeric_matrix <- function(x, arg_name = "data") {
    if (!is_float32_matrix_input(x)) {
        x <- as.matrix(x)
        storage.mode(x) <- "double"
        return(x)
    }
    out <- float::dbl(x)
    out <- as.matrix(out)
    storage.mode(out) <- "double"
    out
}

as_float_distances <- function(x) {
    float::fl(x)
}

.faissR_transformed_float32_cache <- new.env(parent = emptyenv())
.faissR_transformed_float32_cache$.keys <- character()

transformed_float32_cache_enabled <- function() {
    isTRUE(faissr_option("cache_transformed_float32", TRUE))
}

transformed_float32_cache_limit <- function() {
    value <- faissr_quiet_warning(as.integer(faissr_option(
        "cache_transformed_float32_max_entries",
        4L
    )))
    if (
        length(value) != 1L || is.na(value) || !is.finite(value) || value < 0L
    ) {
        return(4L)
    }
    value
}

transformed_float32_cache_transform_label <- function(metric) {
    if (identical(metric, "correlation")) {
        "row_center_l2_normalize_float32_cached"
    } else {
        "row_l2_normalize_float32_cached"
    }
}

transformed_float32_cache_prune <- function() {
    limit <- transformed_float32_cache_limit()
    keys <- .faissR_transformed_float32_cache$.keys
    if (limit < 1L) {
        rm(
            list = setdiff(
                ls(.faissR_transformed_float32_cache, all.names = TRUE),
                ".keys"
            ),
            envir = .faissR_transformed_float32_cache
        )
        .faissR_transformed_float32_cache$.keys <- character()
        return(invisible(NULL))
    }
    while (length(keys) > limit) {
        old <- keys[[1L]]
        if (
            exists(
                old,
                envir = .faissR_transformed_float32_cache,
                inherits = FALSE
            )
        ) {
            rm(list = old, envir = .faissR_transformed_float32_cache)
        }
        keys <- keys[-1L]
    }
    .faissR_transformed_float32_cache$.keys <- keys
    invisible(NULL)
}

normalized_float32_transform_cached <- function(x, metric, role = "data") {
    metric <- normalize_nn_metric(metric)
    if (!metric %in% c("cosine", "correlation")) {
        stop(
            "Cached normalized float32 transforms require ",
            "cosine or correlation.",
            call. = FALSE
        )
    }
    dims <- if (is_float32_matrix_input(x)) {
        float32_matrix_dims(x, role)
    } else {
        dim(x)
    }
    fingerprint <- matrix_fingerprint_cpp(x)
    key <- normalized_float32_cache_key(metric, dims, fingerprint)
    cache_enabled <- transformed_float32_cache_enabled()
    entry <- find_normalized_float32_cache_entry(key, cache_enabled)
    if (!is.null(entry)) {
        entry$cache_hit <- TRUE
        entry$role <- role
        return(entry)
    }
    transformed <- normalized_float32_transform_cpp(x, metric)
    entry <- new_normalized_float32_cache_entry(
        transformed, metric, key, fingerprint, cache_enabled, role
    )
    store_normalized_float32_cache_entry(key, entry, cache_enabled)
    entry
}

normalized_float32_cache_key <- function(metric, dims, fingerprint) {
    paste(
        metric,
        paste(as.integer(dims), collapse = "x"),
        fingerprint,
        sep = ":"
    )
}

find_normalized_float32_cache_entry <- function(key, enabled) {
    if (!isTRUE(enabled) || !exists(
        key,
        envir = .faissR_transformed_float32_cache,
        inherits = FALSE
    )) {
        return(NULL)
    }
    get(key, envir = .faissR_transformed_float32_cache, inherits = FALSE)
}

new_normalized_float32_cache_entry <- function(
    transformed,
    metric,
    key,
    fingerprint,
    cache_enabled,
    role
) {
    list(
        data = transformed$data,
        zero = as.logical(transformed$zero),
        transform = transformed_float32_cache_transform_label(metric),
        storage = "float32",
        row_major = TRUE,
        cache_key = key,
        fingerprint = fingerprint,
        cache_hit = FALSE,
        cache_enabled = isTRUE(cache_enabled),
        role = role
    )
}

store_normalized_float32_cache_entry <- function(key, entry, enabled) {
    if (isTRUE(enabled) && transformed_float32_cache_limit() > 0L) {
        assign(key, entry, envir = .faissR_transformed_float32_cache)
        keys <- .faissR_transformed_float32_cache$.keys
        .faissR_transformed_float32_cache$.keys <- c(setdiff(keys, key), key)
        transformed_float32_cache_prune()
    }
    invisible(NULL)
}

finalize_nn_output <- function(result, output = "double") {
    output <- normalize_nn_output(output)
    public_metric <- attr(result, "metric") %||% result$metric %||% "euclidean"
    if (identical(output, "float")) {
        if (!inherits(result$distances, "float32")) {
            result$distances <- as_float_distances(result$distances)
        }
        result$distance_type <- "float32"
        attr(result, "distance_type") <- "float32"
    } else {
        result$distance_type <- "double"
        attr(result, "distance_type") <- "double"
    }
    result$index_base <- result$index_base %||% 1L
    result$metric <- public_metric
    result$backend_used <- result$backend_used %||%
        attr(result, "resolved_backend") %||%
        attr(result, "backend") %||%
        NA_character_
    attr(result, "index_base") <- as.integer(result$index_base)
    attr(result, "metric") <- public_metric
    attr(result, "backend_used") <- result$backend_used
    result
}
