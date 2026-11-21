cuvs_cagra_wide_params <- function(target_recall, width4) {
    target <- normalize_hnsw_target_recall(target_recall)
    if (isTRUE(width4) && target == 0.99) {
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
                                width4 = FALSE) {
    require_cuvs_backend("cuVS CAGRA")
    params <- cuvs_cagra_wide_params(target_recall, width4)
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
    result <- nn(data, k = k, backend = "cuda", method = "cagra",
        metric = "euclidean", exclude_self = TRUE, output = output,
        target_recall = target_recall, cagra_implementation = "cuvs",
        cagra_build_algo = params$builder, tuning = "fixed", n_threads = 1L)
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

cuvs_cagra_wide_file <- function(source, k, batch, memory_limit,
                                 target_recall, width4) {
    params <- cuvs_cagra_wide_params(target_recall, width4)
    index <- file_cagra_build_cpp(source, k, batch, memory_limit,
        params$width, params$queue, params$builder == "nn_descent")
    pointer <- index$pointer
    index$search_batch <- function(first, count)
        file_cagra_search_cpp(pointer, first, count)
    index$release <- function() file_cagra_release_cpp(pointer)
    index$params <- params
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
