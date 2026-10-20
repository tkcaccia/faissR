faiss_flat_backend <- function(metric, accelerator = NULL) {
    prefix <- if (identical(accelerator, "cuda")) {
        "faiss_gpu_flat_"
    } else {
        "faiss_flat_"
    }
    suffix <- switch(
        metric,
        inner_product = "ip",
        cosine = "cosine",
        correlation = "correlation",
        "l2"
    )
    paste0(prefix, suffix)
}

validate_cpu_nn_route <- function(method, metric) {
    if (identical(method, "grid") && identical(metric, "inner_product")) {
        stop(
            "CPU grid search does not support raw inner product.",
            call. = FALSE
        )
    }
    if (
        identical(method, "ivfpq_fastscan") &&
            !metric %in%
                c("euclidean", "cosine", "correlation", "inner_product")
    ) {
        stop(
            "CPU `method = \"ivfpq_fastscan\"` does not support this metric.",
            call. = FALSE
        )
    }
}

resolve_cpu_nn_backend <- function(method, metric) {
    validate_cpu_nn_route(method, metric)
    exact_backend <- if (isTRUE(faiss_available())) {
        faiss_flat_backend(metric)
    } else {
        "cpu"
    }
    switch(
        method,
        exact = exact_backend,
        bruteforce = exact_backend,
        flat = faiss_flat_backend(metric),
        grid = "cpu_grid",
        hnsw = "faiss_hnsw",
        ivf = "faiss_ivf",
        ivfpq = "faiss_ivfpq",
        vamana = "cpu_vamana",
        nsg = "cpu_nsg",
        nndescent = "cpu_nndescent",
        ivfpq_fastscan = "faiss_ivfpq_fastscan",
        cagra = stop(
            "`method = \"cagra\"` is only available with `backend = \"cuda\"`.",
            call. = FALSE
        ),
        stop("Unsupported CPU nearest-neighbour method.", call. = FALSE)
    )
}

resolve_cuda_exact_backend <- function(method, metric) {
    if (metric %in% c("cosine", "correlation", "inner_product")) {
        if (
            identical(method, "bruteforce") ||
                (identical(method, "exact") && !isTRUE(faiss_gpu_available()))
        ) {
            if (isTRUE(cuvs_available())) return("cuda_cuvs_bruteforce")
        }
        return(faiss_flat_backend(metric, accelerator = "cuda"))
    }
    if (identical(method, "exact")) {
        if (isTRUE(faiss_gpu_available())) {
            return("faiss_gpu_flat_l2")
        }
        if (isTRUE(cuvs_available())) {
            return("cuda_cuvs_bruteforce")
        }
        return("cuda")
    }
    if (identical(method, "bruteforce")) {
        if (isTRUE(cuvs_available())) {
            return("cuda_cuvs_bruteforce")
        }
        if (isTRUE(faiss_gpu_available())) {
            return("faiss_gpu_flat_l2")
        }
        return("cuda")
    }
    "faiss_gpu_flat_l2"
}

validate_cuda_nn_route <- function(method, metric) {
    if (
        identical(method, "ivfpq_fastscan") &&
            !metric %in%
                c("euclidean", "cosine", "correlation", "inner_product")
    ) {
        stop(
            "CUDA `method = \"ivfpq_fastscan\"` does not support this metric.",
            call. = FALSE
        )
    }
    if (identical(metric, "inner_product") && identical(method, "nndescent")) {
        stop(
            "CUDA `method = \"nndescent\"` does not support raw inner product.",
            call. = FALSE
        )
    }
    inner_product_methods <- c(
        "exact",
        "bruteforce",
        "flat",
        "ivf",
        "ivfpq",
        "ivfpq_fastscan",
        "nsg",
        "vamana",
        "cagra",
        "hnsw"
    )
    if (
        identical(metric, "inner_product") && !method %in% inner_product_methods
    ) {
        stop("Unsupported CUDA inner-product method.", call. = FALSE)
    }
}

resolve_cuda_nn_backend <- function(method, metric, n, p, k, self_query) {
    validate_cuda_nn_route(method, metric)
    if (method %in% c("exact", "bruteforce", "flat")) {
        return(resolve_cuda_exact_backend(method, metric))
    }
    switch(
        method,
        grid = "cuda_grid",
        hnsw = "cuda_cuvs_hnsw",
        ivf = "faiss_gpu_ivf_flat",
        ivfpq = "faiss_gpu_ivfpq",
        vamana = "cuda_vamana",
        nsg = "cuda_nsg",
        nndescent = "cuda_cuvs_nndescent",
        ivfpq_fastscan = "cuda_cuvs_ivfpq_fastscan",
        cagra = resolve_cuda_cagra_backend(
            n = n,
            p = p,
            k = k,
            self_query = self_query
        ),
        stop("Unsupported CUDA nearest-neighbour method.", call. = FALSE)
    )
}

resolve_public_auto_method_backend <- function(device) {
    if (identical(device, "cuda")) {
        return("cuda_auto")
    }
    "cpu_auto"
}
