cuvs_cagra_wide_params <- function(target_recall, width4, nn_descent = FALSE) {
    target <- normalize_hnsw_target_recall(target_recall)
    if (isTRUE(width4) && (target == 0.99 || isTRUE(nn_descent))) {
        return(list(width = 4L, queue = 384L, builder = "nn_descent"))
    }
    if (isTRUE(width4)) {
        return(list(width = 4L, queue = 512L, builder = "ivf_pq"))
    }
    if (target == 0.90) {
        return(list(width = 2L, queue = 256L, builder = "ivf_pq"))
    }
    if (target == 0.95) {
        return(list(width = 2L, queue = 384L, builder = "ivf_pq"))
    }
    list(width = 2L, queue = 512L, builder = "nn_descent")
}

cuvs_cagra_wide_self <- function(data, k, target_recall, output,
        width4 = FALSE, profile = getOption("faissR.cuvs_cagra_profile")) {
    require_cuvs_backend("cuVS CAGRA")
    params <- cuvs_cagra_wide_params(target_recall, width4,
        nn_descent = identical(profile$builder, "nn_descent"))
    result <- if (is.null(profile)) {
        cuvs_cagra_wide_legacy_self(data, k, target_recall, output, params)
    } else {
        cuvs_cagra_profiled_self(data, k, output, params, profile)
    }
    engine <- attr(result, "resolved_backend")
    stopifnot(identical(engine, "cuda_cuvs_cagra"),
        identical(dim(result$indices), c(nrow(data), as.integer(k))))
    result$backend_used <- engine
    result$backend <- "cuda"
    result$accelerator <- "cuda"
    result$index_base <- 1L
    result$exclude_self <- TRUE
    result$knn_provider <- "faissR"
    result$target_recall <- target_recall
    result$target_met <- NA
    result$recall_audited <- FALSE
    result$recall_status <- "calibration_informed_not_runtime_audited"
    result
}

cuvs_validate_cagra_profile <- function(profile, n, k) {
    if (!is.list(profile) || (!is.null(profile$builder) &&
        !identical(profile$builder, "nn_descent"))) {
        stop("Invalid CUDA CAGRA profile.", call. = FALSE)
    }
    profile$builder <- NULL
    fields <- c("graph_degree", "intermediate_degree", "batch_rows",
        "memory_limit_bytes")
    if (!identical(sort(names(profile)), sort(fields)) ||
        !all(vapply(profile, function(value) {
            is.numeric(value) && length(value) == 1L &&
                is.finite(value) && value > 0 && value == floor(value)
        }, logical(1L))) ||
        profile$graph_degree <= k ||
        profile$intermediate_degree < profile$graph_degree ||
        profile$intermediate_degree >= n || profile$batch_rows > 1024L) {
        stop("Invalid CUDA CAGRA profile.", call. = FALSE)
    }
    profile
}

cuvs_cagra_profiled_self <- function(data, k, output, params, profile) {
    profile <- cuvs_validate_cagra_profile(profile, nrow(data), k)
    result <- nn_cuvs_cagra_batched_self_cpp(data, k,
        as.integer(profile$batch_rows), profile$memory_limit_bytes,
        params$width, params$queue, params$builder == "nn_descent",
        as.integer(profile$graph_degree),
        as.integer(profile$intermediate_degree), output)
    result$cagra_settings <- c(params, profile)
    result
}

cuvs_cagra_wide_legacy_self <- function(data, k, target_recall, output,
                                       params) {
    previous <- options(
        faissR.cuvs_graph_degree = 64L,
        faissR.cuvs_intermediate_graph_degree = 128L,
        faissR.cuvs_search_width = params$width,
        faissR.cuvs_itopk_size = params$queue
    )
    on.exit(options(previous), add = TRUE)
    batch <- Sys.getenv("FAISSR_CUVS_CAGRA_BATCH_SIZE", unset = NA_character_)
    on.exit({
        if (is.na(batch)) Sys.unsetenv("FAISSR_CUVS_CAGRA_BATCH_SIZE")
        else Sys.setenv(FAISSR_CUVS_CAGRA_BATCH_SIZE = batch)
    }, add = TRUE)
    Sys.setenv(FAISSR_CUVS_CAGRA_BATCH_SIZE = "1024")
    nn(data, k = k, backend = "cuda", method = "cagra",
        metric = "euclidean", exclude_self = TRUE, output = output,
        target_recall = target_recall, cagra_implementation = "cuvs",
        cagra_build_algo = params$builder, tuning = "fixed", n_threads = 1L)
}

cuvs_cagra_file_search <- function(params, k,
        search = getOption("faissR.cuvs_cagra_file_search")) {
    if (is.null(search)) return(params)
    fields <- c("queue", "width")
    if ("batch" %in% names(search)) fields <- c("batch", fields)
    if (!is.list(search) || !identical(sort(names(search)), fields) ||
        !all(vapply(search, function(x) {
            is.numeric(x) && length(x) == 1L && is.finite(x) && x == floor(x)
        }, logical(1L))) ||
        search$width < 1L || search$width > 64L ||
        search$queue <= k || search$queue > 4096L ||
        (!is.null(search$batch) && (search$batch < 1L || search$batch > 1024L))) {
        stop("Invalid CUDA CAGRA file search width or queue.", call. = FALSE)
    }
    params$width <- as.integer(search$width)
    params$queue <- as.integer(search$queue)
    if (!is.null(search$batch)) params$batch <- as.integer(search$batch)
    params
}

# The caller supplies the query batch capacity admitted by the memory planner.
cuvs_cagra_file_search_candidates <- function(k, incumbent, max_batch) {
    incumbent <- cuvs_cagra_file_search(list(), k, incumbent)
    stopifnot(length(max_batch) == 1L, is.finite(max_batch),
        max_batch == floor(max_batch), max_batch >= incumbent$batch,
        max_batch <= 1024L, length(incumbent$batch) == 1L)
    widths <- unique(c(incumbent$width, max(1L, incumbent$width %/% 2L)))
    queues <- unique(as.integer(pmax(k + 1L,
        ceiling(incumbent$queue * c(1, 2 / 3, 1 / 2, 1 / 3)))))
    batches <- unique(c(incumbent$batch, as.integer(max_batch)))
    expand.grid(width = widths, queue = queues, batch = batches,
        KEEP.OUT.ATTRS = FALSE)
}

# Each candidate is measured on the same complete reference index or shards.
# Calibration and holdout rows both use a separate exact reference.
cuvs_cagra_select_file_search <- function(results, k, incumbent, target_recall,
                                         minimum_speedup = 1.20) {
    fields <- c("width", "queue", "batch")
    incumbent <- cuvs_cagra_file_search(list(), k, incumbent)
    stopifnot(length(incumbent$batch) == 1L,
        length(minimum_speedup) == 1L, is.finite(minimum_speedup),
        minimum_speedup > 1, all(c(fields, "seconds", "status",
            "calibration_recall", "holdout_recall") %in% names(results)))
    target <- normalize_hnsw_target_recall(target_recall)
    calibration <- results$calibration_recall
    holdout <- results$holdout_recall
    valid <- which(results$status == "success" & is.finite(results$seconds) &
        results$seconds > 0 & is.finite(calibration) & calibration >= 0 &
        calibration <= 1 & is.finite(holdout) & holdout >= 0 & holdout <= 1)
    success <- results[valid, , drop = FALSE]
    same <- Reduce(`&`, lapply(fields, function(field)
        success[[field]] == incumbent[[field]]))
    baseline <- success[which(same), , drop = FALSE]
    if (nrow(baseline) != 1L) {
        return(list(params = incumbent, status = "baseline_unverified"))
    }
    success$recall <- pmin(success$calibration_recall, success$holdout_recall)
    qualifies <- success$recall > target &
        success$seconds <= baseline$seconds / minimum_speedup
    eligible <- success[qualifies, , drop = FALSE]
    if (!nrow(eligible)) {
        status <- if (min(baseline$calibration_recall,
            baseline$holdout_recall) > target) "incumbent_kept" else
            "target_unverified"
        return(list(params = incumbent, status = status))
    }
    result <- select_tuning_result(eligible, incumbent, target, fields)
    result$params <- cuvs_cagra_file_search(list(), k, result$params)
    result
}

cuvs_cagra_wide_file <- function(source, k, batch, memory_limit,
        target_recall, width4,
        profile = getOption("faissR.cuvs_cagra_profile")) {
    params <- cuvs_cagra_wide_params(target_recall, width4,
        nn_descent = identical(profile$builder, "nn_descent"))
    if (is.null(profile)) {
        profile <- list(graph_degree = 64L, intermediate_degree = 128L,
            batch_rows = batch, memory_limit_bytes = memory_limit)
    } else {
        profile <- cuvs_validate_cagra_profile(profile, source$nrow, k)
    }
    params <- cuvs_cagra_file_search(params, k)
    batch <- as.integer(min(batch, profile$batch_rows, params$batch %||% batch))
    params$batch <- NULL
    budget <- min(memory_limit, profile$memory_limit_bytes)
    index <- file_cagra_build_cpp(source, k, batch, budget,
        params$width, params$queue, params$builder == "nn_descent",
        as.integer(profile$graph_degree),
        as.integer(profile$intermediate_degree))
    pointer <- index$pointer
    index$search_batch <- function(first, count)
        file_cagra_search_cpp(pointer, first, count)
    index$query_batch <- function(query, first, count)
        file_cagra_query_cpp(pointer, query, first, count)
    index$set_search <- function(width, queue)
        file_cagra_set_search_cpp(pointer, width, queue)
    index$release <- function() file_cagra_release_cpp(pointer)
    index$params <- c(params, list(graph_degree = index$graph_degree,
        intermediate_degree = index$intermediate_graph_degree,
        batch_rows = index$query_batch_rows, memory_limit_bytes = budget))
    index
}

metal_cagra_file <- function(source, k, batch, memory_limit,
                             target_recall, width4) {
    stopifnot(batch == 1024L, !isTRUE(width4))
    index <- file_metal_cagra_build_cpp(source, k, memory_limit, target_recall)
    pointer <- index$pointer
    index$search_batch <- function(first, count)
        file_metal_cagra_search_cpp(pointer, first, count)
    index$release <- function() file_metal_cagra_release_cpp(pointer)
    index
}

cuvs_cagra_wide_file_multi <- function(source, k, batch, memory_limit,
        target_recall, width4, devices) {
    params <- cuvs_cagra_wide_params(target_recall, width4, nn_descent = TRUE)
    params <- cuvs_cagra_file_search(params, k)
    batch <- as.integer(min(batch, params$batch %||% batch))
    degree <- as.integer(max(if (source$ncol <= 128L) 32L else 64L, k + 1L))
    index <- file_cagra_multi_build_cpp(source, k, batch, memory_limit,
        params$width, params$queue, degree, 2L * degree, devices)
    pointer <- index$pointer
    index$search_batch <- function(first, count)
        file_cagra_multi_search_cpp(pointer, source, first, count)
    index$release <- function() file_cagra_multi_release_cpp(pointer)
    index$params <- list(width = params$width, queue = params$queue,
        builder = "nn_descent", graph_degree = degree,
        intermediate_degree = 2L * degree, batch_rows = batch,
        memory_limit_bytes = memory_limit, shard_rows = index$shard_rows)
    index
}
