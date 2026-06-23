nn_compute <- function(data,
                       points,
                       k,
                       backend,
                       points_missing,
                       exclude_self = FALSE,
                       n_threads = NULL,
                       metric = "euclidean",
                       tuning = "auto",
                       output = "double",
                       auto_selection = NULL) {
  requested_backend <- backend
  tuning <- normalize_nn_tuning(tuning)
  data_float32 <- is_float32_matrix_input(data)
  points_float32 <- if (isTRUE(points_missing)) data_float32 else is_float32_matrix_input(points)
  if (isTRUE(data_float32) || isTRUE(points_float32)) {
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
    } else {
      dim(data)
    }
    points_dim <- if (isTRUE(points_missing)) {
      data_dim
    } else if (isTRUE(points_float32)) {
      float32_matrix_dims(points, "points")
    } else {
      dim(points)
    }
    if (!identical(data_dim[[2L]], points_dim[[2L]])) {
      stop("`data` and `points` must have the same number of columns.", call. = FALSE)
    }
    self_query <- isTRUE(points_missing) || identical(data, points)
    if (isTRUE(exclude_self) && !isTRUE(self_query)) {
      stop("Self-neighbor exclusion is only valid when `points` is `data`.", call. = FALSE)
    }
    if (is.null(k)) {
      k <- if (data_dim[[1L]] == 1L) {
        1L
      } else {
        min(
          data_dim[[1L]],
          auto_k(data_dim[[1L]], include_self = isTRUE(self_query) && !isTRUE(exclude_self))
        )
      }
    }
    k <- normalize_nn_positive_integer(k, "k", "`k` must be NULL or a positive integer.")
    max_k <- if (isTRUE(exclude_self)) data_dim[[1L]] - 1L else data_dim[[1L]]
    if (k > max_k) {
      stop("`k` cannot be larger than the available neighbor count.", call. = FALSE)
    }
    n_threads <- normalize_nn_threads(n_threads)
    metric <- normalize_nn_metric(metric)
    if (backend %in% c("auto", "cpu", "cpu_auto", "faiss", "cpu_faiss",
                       "cpu_faiss_flat", "faiss_flat", "faiss_flat_l2")) {
      backend <- switch(metric,
        inner_product = "faiss_flat_ip",
        cosine = "faiss_flat_cosine",
        correlation = "faiss_flat_correlation",
        "faiss_flat_l2"
      )
    }
    if (!backend %in% c("faiss", "cpu_faiss", "cpu_faiss_flat", "faiss_flat",
                        "faiss_flat_l2", "faiss_flat_ip",
                        "faiss_flat_cosine", "faiss_flat_correlation")) {
      stop(
        "float32 input currently supports CPU FAISS Flat routes only. ",
        "Use `backend = \"cpu\"` with `method = \"auto\"`, `\"exact\"`, ",
        "`\"bruteforce\"`, or `\"flat\"`, or pass an ordinary R numeric ",
        "matrix for other methods.",
        call. = FALSE
      )
    }
    if (!metric %in% c("euclidean", "cosine", "correlation", "inner_product")) {
      stop(
        "float32 FAISS Flat input currently supports `metric = \"euclidean\"`, ",
        "`\"cosine\"`, `\"correlation\"`, or `\"inner_product\"`.",
        call. = FALSE
      )
    }
    if (!isTRUE(faiss_available())) {
      stop(
        "float32 FAISS Flat input requires faissR to be built with FAISS.",
        call. = FALSE
      )
    }
    out <- nn_faiss_flat_float32_cpp(
      data,
      points,
      as.integer(k),
      isTRUE(exclude_self),
      as.integer(n_threads),
      metric,
      output
    )
    return(finish_nn_result(
      out,
      switch(metric,
        inner_product = "faiss_flat_ip",
        cosine = "faiss_flat_cosine",
        correlation = "faiss_flat_correlation",
        "faiss_flat_l2"
      ),
      k,
      self_query,
      exact = TRUE,
      metric = metric
    ))
  }
  data <- as.matrix(data)
  storage.mode(data) <- "double"
  if (isTRUE(points_missing)) {
    points <- data
  } else {
    points <- as.matrix(points)
    storage.mode(points) <- "double"
  }

  if (!identical(ncol(data), ncol(points))) {
    stop("`data` and `points` must have the same number of columns.", call. = FALSE)
  }
  if (nrow(data) < 1L || nrow(points) < 1L) {
    stop("`data` and `points` must have at least one row.", call. = FALSE)
  }
  self_query <- isTRUE(points_missing) || (
    nrow(data) == nrow(points) &&
      ncol(data) == ncol(points) &&
      identical(data, points)
  )
  if (isTRUE(exclude_self) && !isTRUE(self_query)) {
    stop("Self-neighbor exclusion is only valid when `points` is `data`.", call. = FALSE)
  }
  if (is.null(k)) {
    k <- if (nrow(data) == 1L) {
      1L
    } else {
      min(
        nrow(data),
        auto_k(nrow(data), include_self = isTRUE(self_query) && !isTRUE(exclude_self))
      )
    }
  }
  k <- normalize_nn_positive_integer(k, "k", "`k` must be NULL or a positive integer.")
  max_k <- if (isTRUE(exclude_self)) nrow(data) - 1L else nrow(data)
  if (k > max_k) {
    stop("`k` cannot be larger than the available neighbor count.", call. = FALSE)
  }
  finite_input <- if (isTRUE(points_missing)) {
    all(is.finite(data))
  } else {
    all(is.finite(data)) && all(is.finite(points))
  }
  if (!isTRUE(finite_input)) {
    stop("`data` and `points` must contain only finite values.", call. = FALSE)
  }
  n_threads <- normalize_nn_threads(n_threads)
  metric <- normalize_nn_metric(metric)
  if (!identical(metric, "euclidean")) {
    if (identical(metric, "inner_product") &&
        backend %in% c("cuda_cuvs_hnsw", "cuvs_hnsw")) {
      stop(
        "CUDA cuVS HNSW is disabled for `metric = \"inner_product\"` because ",
        "the current cuVS HNSW route is built from a CAGRA L2 graph, and the ",
        "maximum-inner-product-to-L2 transform can trigger illegal CUDA memory ",
        "accesses. Use CPU HNSW for inner product, or a CUDA method with native ",
        "inner-product candidate refinement such as `method = \"nsg\"`, ",
        "`\"vamana\"`, or `\"nndescent\"`.",
        call. = FALSE
      )
    } else if (identical(metric, "inner_product") &&
               backend %in% c("faiss_flat_l2", "faiss_flat", "cpu_faiss_flat")) {
      backend <- "faiss_flat_ip"
    } else if (identical(metric, "inner_product") &&
               backend %in% c("faiss_gpu_flat_l2", "faiss_gpu_flat", "cuda_faiss_flat_l2")) {
      backend <- "faiss_gpu_flat_ip"
    } else if (identical(metric, "inner_product") &&
               backend %in% c("faiss_flat_ip", "faiss_gpu_flat_ip", "cuda_faiss_flat_ip")) {
      backend <- backend
    } else if (identical(metric, "cosine") &&
               backend %in% c("faiss_flat_l2", "faiss_flat", "cpu_faiss_flat")) {
      backend <- "faiss_flat_cosine"
    } else if (identical(metric, "correlation") &&
               backend %in% c("faiss_flat_l2", "faiss_flat", "cpu_faiss_flat")) {
      backend <- "faiss_flat_correlation"
    } else if (backend %in% c("faiss_flat_cosine", "faiss_flat_correlation")) {
      backend <- backend
    } else if (identical(metric, "cosine") &&
               backend %in% c("faiss_gpu_flat_l2", "faiss_gpu_flat", "cuda_faiss_flat_l2")) {
      backend <- "faiss_gpu_flat_cosine"
    } else if (identical(metric, "correlation") &&
               backend %in% c("faiss_gpu_flat_l2", "faiss_gpu_flat", "cuda_faiss_flat_l2")) {
      backend <- "faiss_gpu_flat_correlation"
    } else if (backend %in% c("faiss_gpu_flat_cosine", "faiss_gpu_flat_correlation")) {
      backend <- backend
    } else if (backend %in% c("cuda_auto", "gpu_auto")) {
      backend <- "cuda_auto"
    } else if (metric %in% c("cosine", "correlation") &&
               backend %in% c("grid", "cpu_grid", "grid2d", "cpu_grid2d",
                              "grid3d", "cpu_grid3d", "cuda_grid",
                              "cuda_grid_auto", "gpu_grid", "cuda_grid2d",
                              "cuda_grid3d")) {
      backend <- backend
    } else if (identical(metric, "inner_product") &&
               backend %in% c("grid", "cpu_grid", "grid2d", "cpu_grid2d",
                              "grid3d", "cpu_grid3d", "cuda_grid",
                              "cuda_grid_auto", "gpu_grid", "cuda_grid2d",
                              "cuda_grid3d")) {
      stop("Grid nearest-neighbour search does not support `metric = \"inner_product\"`.", call. = FALSE)
    } else if (!backend %in% c("auto", "cpu", "cpu_auto", "hnsw", "rcpphnsw", "cpu_hnsw",
                               "faiss_hnsw", "faiss_ivf", "faiss_ivf_flat",
                               "faiss_ivfpq", "faiss_nsg", "faiss_nndescent",
                               "cpu_nsg", "cpu_vamana", "cuda_vamana", "cuda_nsg",
                               "cpu_nndescent",
                               "cpu_faiss_index_ivf", "faiss_gpu_ivf",
                               "faiss_gpu_ivf_flat", "cuda_faiss_ivf_flat",
                               "faiss_gpu_ivfpq", "cuda_faiss_ivfpq",
                               "faiss_gpu_cagra", "cuda_faiss_cagra",
                               "cuda_cuvs_cagra", "cuda_cagra", "gpu_cagra",
                               "cuvs_ivf", "cuda_cuvs_ivf",
                               "cuvs_ivf_flat", "cuda_cuvs_ivf_flat",
                               "cuvs_ivfpq", "cuda_cuvs_ivfpq",
                               "cuvs_ivf_pq", "cuda_cuvs_ivf_pq",
                               "cuda_cuvs_nndescent", "cuvs_nndescent",
                               "cuda_native_nndescent",
                               "cuda_cuvs_hnsw", "cuvs_hnsw")) {
      stop(
        "`metric = \"", metric, "\"` currently supports only `backend = \"cpu\"` ",
        "or a validated metric-specific exact backend. ",
        "Approximate FAISS, CUDA, and cuVS KNN paths in this build ",
        "have validated Euclidean-distance semantics only unless explicitly routed.",
        call. = FALSE
      )
    }
  }

  if (backend %in% c("cuda_nndescent", "cuda_approx", "gpu_nndescent", "gpu_approx")) {
    backend <- "cuda_cuvs_nndescent"
  }
  if (backend %in% c("cuvs_ivf", "cuda_cuvs_ivf")) {
    backend <- "cuda_cuvs_ivf_flat"
  }

  work_size <- as.double(nrow(data)) * as.double(nrow(points)) * as.double(ncol(data))

  if (backend %in% c("cuda", "gpu") &&
      !isTRUE(cuda_available()) &&
      !isTRUE(cuvs_available())) {
    stop("No CUDA GPU backend is available on this machine.", call. = FALSE)
  }

  if (backend %in% c("auto", "cpu_auto", "cuda_auto", "gpu_auto")) {
    route <- auto_selection %||% nn_auto_selection_for_backend(
      backend = backend,
      self_query = self_query,
      n = nrow(data),
      p = ncol(data),
      n_points = nrow(points),
      k = k,
      work_size = work_size,
      metric = metric,
      exclude_self = isTRUE(exclude_self),
      tuning = tuning
    )
    backend <- nn_auto_selected_backend(route, backend)
  } else if (identical(backend, "cpu_approx")) {
    if (!isTRUE(self_query)) {
      stop("`backend = \"cpu_approx\"` is only available for self-KNN searches.", call. = FALSE)
    }
    backend <- select_cpu_approx_backend(nrow(data), ncol(data), k)
  }

  gpu_ivf <- resolve_gpu_ivf_backend(
    backend = backend,
    self_query = self_query,
    n = nrow(data),
    p = ncol(data),
    k = k,
    exclude_self = isTRUE(exclude_self)
  )
  if (!is.na(gpu_ivf$backend)) {
    return(gpu_ivf_self_knn(
      data,
      k = k,
      backend = gpu_ivf$backend,
      label = gpu_ivf$label,
      strategy = gpu_ivf$strategy,
      exclude_self = isTRUE(exclude_self),
      seed = fast_knn_approx_seed(),
      metric = metric
    ))
  }

  if (identical(output, "float") &&
      backend %in% c("faiss", "cpu_faiss", "cpu_faiss_flat", "faiss_flat",
                     "faiss_flat_l2", "faiss_flat_ip",
                     "faiss_flat_cosine", "faiss_flat_correlation")) {
    if (!isTRUE(faiss_available())) {
      stop(
        "The real FAISS C++ backend is not available in this build. ",
        "Reinstall faissR with `FAISS_HOME` pointing ",
        "to a FAISS installation.",
        call. = FALSE
      )
    }
    backend_label <- switch(metric,
      inner_product = "faiss_flat_ip",
      cosine = "faiss_flat_cosine",
      correlation = "faiss_flat_correlation",
      "faiss_flat_l2"
    )
    out <- nn_faiss_flat_float32_cpp(
      data,
      points,
      as.integer(k),
      isTRUE(exclude_self),
      as.integer(n_threads),
      metric,
      output
    )
    result <- finish_nn_result(
      out,
      backend_label,
      k,
      self_query,
      exact = TRUE,
      metric = metric
    )
    attr(result, "faiss") <- list(
      index_type = as.character(out$index_type),
      library = "faiss",
      backend = "cpu",
      metric = metric,
      input_type = "float32"
    )
    return(result)
  }

  if (backend %in% c("faiss", "cpu_faiss", "cpu_faiss_flat", "faiss_flat", "faiss_flat_l2")) {
    if (!isTRUE(faiss_available())) {
      stop(
        "The real FAISS C++ backend is not available in this build. ",
        "Reinstall faissR with `FAISS_HOME` pointing ",
        "to a FAISS installation.",
        call. = FALSE
      )
    }
    out <- nn_faiss_flat_cpp(
      data,
      points,
      as.integer(k),
      isTRUE(exclude_self),
      as.integer(n_threads)
    )
    result <- finish_nn_result(out, "faiss", k, self_query, exact = TRUE)
    attr(result, "faiss") <- list(
      index_type = as.character(out$index_type),
      library = "faiss",
      backend = "cpu"
    )
    return(result)
  }

  if (identical(backend, "faiss_flat_ip")) {
    if (!isTRUE(faiss_available())) {
      stop(
        "The real FAISS C++ backend is not available in this build. ",
        "Reinstall faissR with `FAISS_HOME` pointing ",
        "to a FAISS installation.",
        call. = FALSE
      )
    }
    out <- nn_faiss_flat_ip_cpp(
      data,
      points,
      as.integer(k),
      isTRUE(exclude_self),
      as.integer(n_threads)
    )
    result <- finish_nn_result(
      out,
      "faiss_flat_ip",
      k,
      self_query,
      exact = TRUE,
      metric = "inner_product"
    )
    attr(result, "faiss") <- list(
      index_type = as.character(out$index_type),
      library = "faiss",
      backend = "cpu",
      metric = as.character(out$metric)
    )
    return(result)
  }

  if (backend %in% c("faiss_flat_cosine", "faiss_flat_correlation")) {
    if (!isTRUE(faiss_available())) {
      stop(
        "The real FAISS C++ backend is not available in this build. ",
        "Reinstall faissR with `FAISS_HOME` pointing ",
        "to a FAISS installation.",
        call. = FALSE
      )
    }
    metric_label <- if (identical(backend, "faiss_flat_correlation")) "correlation" else "cosine"
    return(faiss_flat_normalized_metric_result(
      data = data,
      points = points,
      k = k,
      self_query = self_query,
      exclude_self = isTRUE(exclude_self),
      metric = metric_label,
      backend = backend,
      accelerator = NULL,
      n_threads = n_threads
    ))
  }

  if (backend %in% c("faiss_gpu_flat", "faiss_gpu_flat_l2", "cuda_faiss_flat_l2")) {
    if (!isTRUE(faiss_gpu_available())) {
      stop(
        "The real FAISS C++ GPU Flat L2 backend is not available in this build. ",
        "Reinstall faissR with FAISS GPU/cuVS headers ",
        "available through `FAISS_HOME`.",
        call. = FALSE
      )
    }
    out <- nn_faiss_gpu_flat_cpp(
      data,
      points,
      as.integer(k),
      isTRUE(exclude_self)
    )
    result <- finish_nn_result(out, "faiss_gpu_flat_l2", k, self_query, exact = TRUE)
    attr(result, "faiss") <- list(
      index_type = as.character(out$index_type),
      library = "faiss",
      backend = "cuda",
      accelerator = "cuda",
      metric = as.character(out$metric)
    )
    return(result)
  }

  if (backend %in% c("faiss_gpu_flat_ip", "cuda_faiss_flat_ip")) {
    if (!isTRUE(faiss_gpu_available())) {
      stop(
        "The real FAISS C++ GPU Flat IP backend is not available in this build. ",
        "Reinstall faissR with FAISS GPU/cuVS headers ",
        "available through `FAISS_HOME`.",
        call. = FALSE
      )
    }
    out <- nn_faiss_gpu_flat_ip_cpp(
      data,
      points,
      as.integer(k),
      isTRUE(exclude_self)
    )
    result <- finish_nn_result(
      out,
      "faiss_gpu_flat_ip",
      k,
      self_query,
      exact = TRUE,
      metric = "inner_product"
    )
    attr(result, "faiss") <- list(
      index_type = as.character(out$index_type),
      library = "faiss",
      backend = "cuda",
      accelerator = "cuda",
      metric = as.character(out$metric)
    )
    return(result)
  }

  if (backend %in% c("faiss_gpu_flat_cosine", "cuda_faiss_flat_cosine",
                     "faiss_gpu_flat_correlation", "cuda_faiss_flat_correlation")) {
    if (!isTRUE(faiss_gpu_available())) {
      stop(
        "The real FAISS C++ GPU Flat IP backend is not available in this build. ",
        "Reinstall faissR with FAISS GPU/cuVS headers ",
        "available through `FAISS_HOME`.",
        call. = FALSE
      )
    }
    metric_label <- if (backend %in% c("faiss_gpu_flat_correlation", "cuda_faiss_flat_correlation")) {
      "correlation"
    } else {
      "cosine"
    }
    return(faiss_flat_normalized_metric_result(
      data = data,
      points = points,
      k = k,
      self_query = self_query,
      exclude_self = isTRUE(exclude_self),
      metric = metric_label,
      backend = if (identical(metric_label, "correlation")) "faiss_gpu_flat_correlation" else "faiss_gpu_flat_cosine",
      accelerator = "cuda",
      n_threads = n_threads
    ))
  }

  if (backend %in% c("faiss_ivf", "cpu_faiss_index_ivf", "faiss_ivf_flat")) {
    if (!isTRUE(faiss_available())) {
      stop(
        "The real FAISS C++ IVF backend is not available in this build. ",
        "Reinstall faissR with `FAISS_HOME` pointing ",
        "to a FAISS installation.",
        call. = FALSE
      )
    }
    params <- faiss_ivf_params(nrow(data), k, metric = metric)
    if (metric %in% c("cosine", "correlation")) {
      return(faiss_ivf_normalized_metric_result(
        data = data,
        points = points,
        k = k,
        self_query = self_query,
        exclude_self = isTRUE(exclude_self),
        metric = metric,
        backend = "faiss_ivf",
        accelerator = NULL,
        n_threads = n_threads,
        params = params
      ))
    }
    out <- nn_faiss_ivf_cpp(
      data,
      points,
      as.integer(k),
      as.integer(params$nlist),
      as.integer(params$nprobe),
      faiss_metric_search_arg(metric),
      faiss_metric_distance_output_arg(metric),
      isTRUE(exclude_self),
      as.integer(n_threads)
    )
    result <- finish_nn_result(out, "faiss_ivf", k, self_query, exact = FALSE, metric = metric)
    attr(result, "approximation") <- list(
      strategy = "faiss_IndexIVFFlat",
      backend = "faiss_ivf",
      library = "faiss",
      metric = metric,
      nlist = as.integer(out$nlist),
      nprobe = as.integer(out$nprobe),
      requested_nlist = as.integer(params$requested_nlist),
      requested_nprobe = as.integer(params$requested_nprobe),
      ivf_parameters_adjusted = !identical(as.integer(params$requested_nlist), as.integer(out$nlist)) ||
        !identical(as.integer(params$requested_nprobe), as.integer(out$nprobe))
    )
    result <- append_nn_tuning_metadata(result, params)
    return(result)
  }

  if (identical(backend, "faiss_ivfpq")) {
    if (!isTRUE(faiss_available())) {
      stop(
        "The real FAISS C++ IVFPQ backend is not available in this build. ",
        "Reinstall faissR with `FAISS_HOME` pointing ",
        "to a FAISS installation.",
        call. = FALSE
      )
    }
    validate_faiss_cpu_ivfpq_training_size(nrow(data))
    params <- faiss_ivf_params(nrow(data), k, metric = metric)
    pq <- faiss_pq_params(ncol(data), n = nrow(data))
    if (metric %in% c("cosine", "correlation")) {
      return(faiss_ivfpq_normalized_metric_result(
        data = data,
        points = points,
        k = k,
        self_query = self_query,
        exclude_self = isTRUE(exclude_self),
        metric = metric,
        backend = "faiss_ivfpq",
        accelerator = NULL,
        n_threads = n_threads,
        params = params,
        pq = pq
      ))
    }
    out <- nn_faiss_ivfpq_cpp(
      data,
      points,
      as.integer(k),
      as.integer(params$nlist),
      as.integer(params$nprobe),
      as.integer(pq$m),
      as.integer(pq$nbits),
      faiss_metric_search_arg(metric),
      faiss_metric_distance_output_arg(metric),
      isTRUE(exclude_self),
      as.integer(n_threads)
    )
    result <- finish_nn_result(out, "faiss_ivfpq", k, self_query, exact = FALSE, metric = metric)
    attr(result, "approximation") <- list(
      strategy = "faiss_IndexIVFPQ",
      backend = "faiss_ivfpq",
      library = "faiss",
      metric = metric,
      nlist = as.integer(out$nlist),
      nprobe = as.integer(out$nprobe),
      requested_nlist = as.integer(params$requested_nlist),
      requested_nprobe = as.integer(params$requested_nprobe),
      ivf_parameters_adjusted = !identical(as.integer(params$requested_nlist), as.integer(out$nlist)) ||
        !identical(as.integer(params$requested_nprobe), as.integer(out$nprobe)),
      pq_m = as.integer(out$pq_m),
      pq_nbits = as.integer(out$pq_nbits),
      requested_pq_m = as.integer(out$requested_pq_m),
      requested_pq_nbits = as.integer(out$requested_pq_nbits),
      pq_parameters_adjusted = isTRUE(out$pq_parameters_adjusted)
    )
    result <- append_nn_tuning_metadata(result, params, pq, .prefixes = list(NULL, "pq_"))
    return(result)
  }

  if (backend %in% c("faiss_gpu_ivf", "faiss_gpu_ivf_flat", "cuda_faiss_ivf_flat")) {
    if (!isTRUE(faiss_gpu_available())) {
      stop(
        "The real FAISS C++ GPU IVF Flat backend is not available in this build. ",
        "Reinstall faissR with FAISS GPU/cuVS headers ",
        "available through `FAISS_HOME`.",
        call. = FALSE
      )
    }
    params <- faiss_ivf_params(nrow(data), k, metric = metric)
    tuning_metadata <- NULL
    if (isTRUE(faiss_gpu_ivf_should_tune(data, k, self_query, tuning = tuning, metric = metric))) {
      tuned <- faiss_gpu_ivf_tune_params(data, k, params, tuning = tuning)
      params <- tuned$params
      tuning_metadata <- tuned$tuning
    }
    if (metric %in% c("cosine", "correlation")) {
      return(faiss_ivf_normalized_metric_result(
        data = data,
        points = points,
        k = k,
        self_query = self_query,
        exclude_self = isTRUE(exclude_self),
        metric = metric,
        backend = "faiss_gpu_ivf_flat",
        accelerator = "cuda",
        n_threads = n_threads,
        params = params,
        tuning_metadata = tuning_metadata
      ))
    }
    out <- nn_faiss_gpu_ivf_flat_cpp(
      data,
      points,
      as.integer(k),
      as.integer(params$nlist),
      as.integer(params$nprobe),
      faiss_metric_search_arg(metric),
      faiss_metric_distance_output_arg(metric),
      isTRUE(exclude_self)
    )
    result <- finish_nn_result(out, "faiss_gpu_ivf_flat", k, self_query, exact = FALSE, metric = metric)
    attr(result, "approximation") <- list(
      strategy = "faiss_gpu_IndexIVFFlat_cuVS",
      backend = "faiss_gpu_ivf_flat",
      library = "faiss",
      accelerator = "cuda",
      metric = metric,
      nlist = as.integer(out$nlist),
      nprobe = as.integer(out$nprobe),
      requested_nlist = as.integer(params$requested_nlist),
      requested_nprobe = as.integer(params$requested_nprobe),
      ivf_parameters_adjusted = !identical(as.integer(params$requested_nlist), as.integer(out$nlist)) ||
        !identical(as.integer(params$requested_nprobe), as.integer(out$nprobe)),
      tuning = tuning_metadata
    )
    result <- append_nn_tuning_metadata(result, params)
    return(result)
  }

  if (backend %in% c("faiss_gpu_ivfpq", "cuda_faiss_ivfpq")) {
    if (!isTRUE(faiss_gpu_available())) {
      stop(
        "The real FAISS C++ GPU IVF-PQ backend is not available in this build. ",
        "Reinstall faissR with FAISS GPU/cuVS headers ",
        "available through `FAISS_HOME`.",
        call. = FALSE
      )
    }
    params <- faiss_ivf_params(nrow(data), k, metric = metric)
    pq <- faiss_pq_params(ncol(data))
    if (metric %in% c("cosine", "correlation")) {
      return(faiss_ivfpq_normalized_metric_result(
        data = data,
        points = points,
        k = k,
        self_query = self_query,
        exclude_self = isTRUE(exclude_self),
        metric = metric,
        backend = "faiss_gpu_ivfpq",
        accelerator = "cuda",
        n_threads = n_threads,
        params = params,
        pq = pq
      ))
    }
    out <- nn_faiss_gpu_ivfpq_cpp(
      data,
      points,
      as.integer(k),
      as.integer(params$nlist),
      as.integer(params$nprobe),
      as.integer(pq$m),
      as.integer(pq$nbits),
      faiss_metric_search_arg(metric),
      faiss_metric_distance_output_arg(metric),
      isTRUE(exclude_self)
    )
    result <- finish_nn_result(out, "faiss_gpu_ivfpq", k, self_query, exact = FALSE, metric = metric)
    attr(result, "approximation") <- list(
      strategy = "faiss_gpu_IndexIVFPQ_cuVS",
      backend = "faiss_gpu_ivfpq",
      library = "faiss",
      accelerator = "cuda",
      metric = metric,
      role = "explicit_memory_pressure_backend",
      default_candidate = FALSE,
      nlist = as.integer(out$nlist),
      nprobe = as.integer(out$nprobe),
      requested_nlist = as.integer(params$requested_nlist),
      requested_nprobe = as.integer(params$requested_nprobe),
      ivf_parameters_adjusted = !identical(as.integer(params$requested_nlist), as.integer(out$nlist)) ||
        !identical(as.integer(params$requested_nprobe), as.integer(out$nprobe)),
      pq_m = as.integer(out$pq_m),
      pq_nbits = as.integer(out$pq_nbits),
      requested_pq_m = as.integer(out$requested_pq_m),
      requested_pq_nbits = as.integer(out$requested_pq_nbits),
      pq_parameters_adjusted = isTRUE(out$pq_parameters_adjusted)
    )
    result <- append_nn_tuning_metadata(result, params, pq, .prefixes = list(NULL, "pq_"))
    return(result)
  }

  if (backend %in% c("faiss_gpu_cagra", "cuda_faiss_cagra")) {
    if (!isTRUE(faiss_gpu_available())) {
      stop(
        "The real FAISS GPU CAGRA backend is not available in this build. ",
        "Reinstall faissR with FAISS GPU/cuVS headers ",
        "available through `FAISS_HOME`.",
        call. = FALSE
      )
    }
    if (identical(metric, "inner_product")) {
      stop(
        "FAISS GPU CAGRA is disabled for `metric = \"inner_product\"` because ",
        "the current maximum-inner-product-to-L2 route is not reliable across ",
        "k values and can return incomplete results or trigger an illegal CUDA ",
        "memory access. Use a CUDA method with native inner-product support, ",
        "such as `method = \"flat\"`, `\"ivf\"`, `\"ivfpq\"`, `\"nsg\"`, ",
        "`\"vamana\"`, or `\"nndescent\"`.",
        call. = FALSE
      )
    }
    metric_inputs <- NULL
    search_data <- data
    search_points <- points
    if (metric %in% c("cosine", "correlation")) {
      metric_inputs <- normalized_euclidean_metric_inputs(data, points, self_query, metric)
      search_data <- metric_inputs$data
      search_points <- metric_inputs$points
    } else if (identical(metric, "inner_product")) {
      metric_inputs <- mips_l2_metric_inputs(data, points, self_query)
      search_data <- metric_inputs$data
      search_points <- metric_inputs$points
    }
    params <- cuvs_cagra_params(nrow(data), k, p = ncol(data))
    out <- nn_faiss_gpu_cagra_cpp(
      search_data,
      search_points,
      as.integer(k),
      as.integer(params$graph_degree),
      as.integer(params$intermediate_graph_degree),
      as.integer(params$search_width),
      as.integer(params$itopk_size),
      isTRUE(exclude_self)
    )
    requested_graph_degree <- if (is.null(params$requested_graph_degree)) out$requested_graph_degree else params$requested_graph_degree
    requested_intermediate_graph_degree <- if (is.null(params$requested_intermediate_graph_degree)) out$requested_intermediate_graph_degree else params$requested_intermediate_graph_degree
    requested_search_width <- if (is.null(params$requested_search_width)) out$requested_search_width else params$requested_search_width
    requested_itopk_size <- if (is.null(params$requested_itopk_size)) out$requested_itopk_size else params$requested_itopk_size
    result <- finish_nn_result(out, "faiss_gpu_cagra", k, self_query, exact = FALSE, metric = metric)
    if (!is.null(metric_inputs)) {
      result <- finalize_graph_metric_result(result, metric_inputs)
    }
    attr(result, "approximation") <- list(
      strategy = "faiss_gpu_GpuIndexCagra_cuVS",
      backend = "faiss_gpu_cagra",
      library = "faiss",
      accelerator = "cuda",
      cagra_provider = "faiss_gpu",
      cagra_provider_option = cagra_implementation_preference(),
      metric = metric,
      transform = if (is.null(metric_inputs)) NA_character_ else metric_inputs$transform,
      distance_transform = if (is.null(metric_inputs)) NA_character_ else metric_inputs$distance_transform %||% "normalized_euclidean_squared_over_2_to_1_minus_similarity",
      graph_degree = as.integer(out$graph_degree),
      intermediate_graph_degree = as.integer(out$intermediate_graph_degree),
      search_width = as.integer(out$search_width),
      itopk_size = as.integer(out$itopk_size),
      requested_graph_degree = as.integer(requested_graph_degree),
      requested_intermediate_graph_degree = as.integer(requested_intermediate_graph_degree),
      requested_search_width = as.integer(requested_search_width),
      requested_itopk_size = as.integer(requested_itopk_size),
      cagra_parameters_adjusted = isTRUE(out$cagra_parameters_adjusted) ||
        !identical(as.integer(requested_graph_degree), as.integer(out$graph_degree)) ||
        !identical(as.integer(requested_intermediate_graph_degree), as.integer(out$intermediate_graph_degree)) ||
        !identical(as.integer(requested_search_width), as.integer(out$search_width)) ||
        !identical(as.integer(requested_itopk_size), as.integer(out$itopk_size))
    )
    result <- append_nn_tuning_metadata(result, params)
    return(result)
  }

  if (identical(backend, "faiss_hnsw")) {
    if (!isTRUE(faiss_available())) {
      stop(
        "The real FAISS C++ HNSW backend is not available in this build. ",
        "Reinstall faissR with `FAISS_HOME` pointing ",
        "to a FAISS installation.",
        call. = FALSE
      )
    }
    if (metric %in% c("cosine", "correlation")) {
      return(faiss_hnsw_normalized_metric_result(
        data = data,
        points = points,
        k = k,
        self_query = self_query,
        exclude_self = isTRUE(exclude_self),
        metric = metric,
        n_threads = n_threads
      ))
    }
    params <- faiss_hnsw_params(
      k,
      n = nrow(data),
      p = ncol(data),
      metric = metric
    )
    out <- nn_faiss_hnsw_cpp(
      data,
      points,
      as.integer(k),
      as.integer(params$m),
      as.integer(params$ef_construction),
      as.integer(params$ef_search),
      faiss_metric_search_arg(metric),
      faiss_metric_distance_output_arg(metric),
      isTRUE(exclude_self),
      as.integer(n_threads)
    )
    result <- finish_nn_result(out, "faiss_hnsw", k, self_query, exact = FALSE, metric = metric)
    attr(result, "approximation") <- list(
      strategy = "faiss_IndexHNSWFlat",
      backend = "faiss_hnsw",
      library = "faiss",
      metric = metric,
      m = as.integer(out$m),
      ef_construction = as.integer(out$ef_construction),
      ef_search = as.integer(out$ef_search),
      requested_m = as.integer(out$requested_m),
      requested_ef_construction = as.integer(out$requested_ef_construction),
      requested_ef_search = as.integer(out$requested_ef_search),
      hnsw_parameters_adjusted = isTRUE(out$hnsw_parameters_adjusted),
      tuning_policy = params$policy,
      tuning_rule = params$rule,
      tuning_high_dim = isTRUE(params$high_dim),
      tuning_large_n = isTRUE(params$large_n),
      tuning_small_k = isTRUE(params$small_k),
      tuning_large_k = isTRUE(params$large_k),
      tuning_non_euclidean = isTRUE(params$non_euclidean),
      tuning_source = params$tuning_source %||% "cpp"
    )
    return(result)
  }

  if (identical(backend, "faiss_nsg")) {
    if (!isTRUE(faiss_available())) {
      stop(
        "The real FAISS C++ NSG backend is not available in this build. ",
        "Reinstall faissR with `FAISS_HOME` pointing ",
        "to a FAISS installation.",
        call. = FALSE
      )
    }
    metric_inputs <- NULL
    search_data <- data
    search_points <- points
    if (metric %in% c("cosine", "correlation", "inner_product")) {
      stop(
        "`backend = \"faiss_nsg\"` currently supports only `metric = \"euclidean\"`. ",
        "FAISS NSG graph construction can abort the R process for normalized ",
        "cosine/correlation or raw inner-product routes in this linked FAISS build.",
        call. = FALSE
      )
    }
    params <- faiss_nsg_params(k)
    out <- nn_faiss_nsg_cpp(
      search_data,
      search_points,
      as.integer(k),
      as.integer(params$r),
      as.integer(params$search_l),
      as.integer(params$build_type),
      "euclidean",
      "euclidean",
      isTRUE(exclude_self),
      as.integer(n_threads)
    )
    result <- finish_nn_result(out, "faiss_nsg", k, self_query, exact = FALSE, metric = metric)
    if (!is.null(metric_inputs)) {
      result <- finalize_normalized_euclidean_metric_result(result, metric_inputs)
    }
    attr(result, "approximation") <- list(
      strategy = "faiss_IndexNSGFlat",
      backend = "faiss_nsg",
      library = "faiss",
      metric = metric,
      transform = if (is.null(metric_inputs)) NA_character_ else metric_inputs$transform,
      r = as.integer(out$r),
      search_l = as.integer(out$search_l),
      build_type = as.integer(out$build_type),
      gk = as.integer(out$gk),
      requested_r = as.integer(out$requested_r),
      requested_search_l = as.integer(out$requested_search_l),
      requested_build_type = as.integer(out$requested_build_type),
      nsg_parameters_adjusted = isTRUE(out$nsg_parameters_adjusted)
    )
    result <- append_nn_tuning_metadata(result, params)
    return(result)
  }

  if (backend %in% c("cpu_nsg", "cuda_nsg")) {
    if (!isTRUE(self_query)) {
      stop("Native NSG is currently implemented for self-KNN searches only.", call. = FALSE)
    }
    use_cuda <- identical(backend, "cuda_nsg")
    if (isTRUE(use_cuda) && !isTRUE(cuda_available())) {
      stop("No CUDA GPU backend is available on this machine.", call. = FALSE)
    }
    metric_inputs <- NULL
    search_data <- data
    refine_metric <- "euclidean"
    if (metric %in% c("cosine", "correlation")) {
      metric_inputs <- normalized_euclidean_metric_inputs(data, points, self_query, metric)
      search_data <- metric_inputs$data
    } else if (identical(metric, "inner_product")) {
      refine_metric <- "inner_product"
    }
    params <- native_nsg_params(
      nrow(search_data),
      ncol(search_data),
      if (isTRUE(exclude_self)) k else max(1L, k - 1L),
      metric = metric,
      backend = if (isTRUE(use_cuda)) "cuda" else "cpu"
    )
    nonself_k <- if (isTRUE(exclude_self)) k else max(0L, k - 1L)
    if (nonself_k < 1L) {
      out <- list(
        indices = matrix(seq_len(nrow(search_data)), nrow(search_data), 1L),
        distances = matrix(0, nrow(search_data), 1L)
      )
      attr(out, "approximation") <- list(
        seed_backend = "trivial_self",
        candidate_columns = 0L,
        seed_graph_k = 0L,
        protected_seed_neighbors = 0L,
        exact_mrng_prune = TRUE
      )
    } else {
      out <- native_nsg_self_knn(
        search_data,
        k = nonself_k,
        r = params$r,
        graph_k = params$graph_k,
        metric = refine_metric,
        use_cuda = use_cuda,
        n_threads = n_threads,
        seed_backend = params$seed_backend %||% "exact"
      )
      if (!isTRUE(exclude_self)) {
        out$indices <- cbind(seq_len(nrow(search_data)), out$indices)
        out$distances <- cbind(rep(0, nrow(search_data)), out$distances)
      }
    }
    result <- finish_nn_result(out, backend, k, self_query, exact = FALSE, metric = metric)
    if (!is.null(metric_inputs)) {
      result <- finalize_normalized_euclidean_metric_result(result, metric_inputs)
    }
    approx <- attr(out, "approximation", exact = TRUE)
    attr(result, "approximation") <- c(
      list(
        strategy = if (isTRUE(use_cuda)) "native_cuda_nsg_candidate_graph" else "native_cpu_nsg_candidate_graph",
        backend = backend,
        accelerator = if (isTRUE(use_cuda)) "cuda" else "cpu",
        metric = metric,
        transform = if (is.null(metric_inputs)) NA_character_ else metric_inputs$transform,
        r = as.integer(params$r),
        graph_k = as.integer(params$graph_k),
        requested_r = as.integer(params$requested_r),
        requested_graph_k = as.integer(params$requested_graph_k),
        requested_seed_backend = params$seed_backend %||% NA_character_,
        seed_k = as.integer(params$seed_k %||% params$graph_k),
        graph_k_cap = as.integer(params$graph_k_cap),
        nsg_parameters_adjusted = !identical(as.integer(params$r), as.integer(params$requested_r)) ||
          !identical(as.integer(params$graph_k), as.integer(params$requested_graph_k)),
        tuning_policy = params$tuning_policy,
        tuning_rule = params$tuning_rule,
        tuning_large_k = isTRUE(params$tuning_large_k),
        tuning_high_dim = isTRUE(params$tuning_high_dim),
        tuning_source = params$tuning_source %||% "cpp"
      ),
      approx
    )
    return(result)
  }

  if (backend %in% c("cpu_vamana", "cuda_vamana")) {
    if (!isTRUE(self_query)) {
      stop("Vamana is currently implemented for self-KNN searches only.", call. = FALSE)
    }
    use_cuda <- identical(backend, "cuda_vamana")
    if (isTRUE(use_cuda) && !isTRUE(cuda_available())) {
      stop("No CUDA GPU backend is available on this machine.", call. = FALSE)
    }
    metric_inputs <- NULL
    search_data <- data
    refine_metric <- metric
    if (metric %in% c("cosine", "correlation")) {
      metric_inputs <- normalized_euclidean_metric_inputs(data, points, self_query, metric)
      search_data <- metric_inputs$data
      refine_metric <- "euclidean"
    }
    nonself_k <- if (isTRUE(exclude_self)) k else max(0L, k - 1L)
    params <- vamana_params(
      nrow(search_data),
      ncol(search_data),
      if (nonself_k < 1L) 1L else nonself_k,
      metric = metric,
      backend = if (isTRUE(use_cuda)) "cuda" else "cpu"
    )
    if (nonself_k < 1L) {
      out <- list(
        indices = matrix(seq_len(nrow(search_data)), nrow(search_data), 1L),
        distances = matrix(0, nrow(search_data), 1L)
      )
      attr(out, "approximation") <- list(
        seed_backend = "trivial_self",
        candidate_columns = 0L,
        seed_search_l = 0L,
        alpha = as.numeric(params$alpha),
        protected_seed_neighbors = 0L,
        exact_robust_prune = TRUE,
        cuvs_vamana_note = "cuVS Vamana currently builds/serializes DiskANN-compatible graphs; faissR performs KNN refinement inside the candidate graph."
      )
    } else {
      out <- vamana_self_knn(
        search_data,
        k = nonself_k,
        r = params$r,
        search_l = params$search_l,
        alpha = params$alpha,
        metric = refine_metric,
        use_cuda = use_cuda,
        n_threads = n_threads,
        seed_backend = params$seed_backend %||% "exact"
      )
      if (!isTRUE(exclude_self)) {
        out$indices <- cbind(seq_len(nrow(search_data)), out$indices)
        out$distances <- cbind(rep(0, nrow(search_data)), out$distances)
      }
    }
    result <- finish_nn_result(out, backend, k, self_query, exact = FALSE, metric = metric)
    if (!is.null(metric_inputs)) {
      result <- finalize_normalized_euclidean_metric_result(result, metric_inputs)
    }
    approx <- attr(out, "approximation", exact = TRUE)
    attr(result, "approximation") <- c(
      list(
        strategy = if (isTRUE(use_cuda)) "native_vamana_candidate_graph_cuda_refine" else "native_vamana_candidate_graph",
        backend = backend,
        accelerator = if (isTRUE(use_cuda)) "cuda" else "cpu",
        metric = metric,
        transform = if (is.null(metric_inputs)) NA_character_ else metric_inputs$transform,
        r = as.integer(params$r),
        search_l = as.integer(params$search_l),
        alpha = as.numeric(params$alpha),
        requested_r = as.integer(params$requested_r),
        requested_search_l = as.integer(params$requested_search_l),
        requested_alpha = as.numeric(params$requested_alpha),
        requested_seed_backend = params$seed_backend %||% NA_character_,
        seed_k = as.integer(params$seed_k %||% params$search_l),
        tuning_policy = params$tuning_policy,
        tuning_rule = params$tuning_rule,
        tuning_large_k = isTRUE(params$tuning_large_k),
        tuning_high_dim = isTRUE(params$tuning_high_dim),
        tuning_source = params$tuning_source %||% "cpp"
      ),
      approx
    )
    return(result)
  }

  if (identical(backend, "faiss_nndescent")) {
    if (!identical(metric, "euclidean")) {
      stop(
        "`backend = \"faiss_nndescent\"` is currently validated only for ",
        "`metric = \"euclidean\"` in this FAISS build.",
        call. = FALSE
      )
    }
    if (!isTRUE(faissr_option("enable_faiss_nndescent", FALSE))) {
      stop(
        "FAISS NNDescent is disabled by default because linked FAISS builds can ",
        "abort the R process during graph construction. Use public ",
        "`method = \"nndescent\"` for the native CPU route, or set ",
        "`options(faissR.enable_faiss_nndescent = TRUE)` to opt into the ",
        "experimental FAISS backend.",
        call. = FALSE
      )
    }
    if (!isTRUE(faiss_available())) {
      stop(
        "The real FAISS C++ NNDescent backend is not available in this build. ",
        "Reinstall faissR with `FAISS_HOME` pointing ",
        "to a FAISS installation.",
        call. = FALSE
      )
    }
    params <- faiss_nndescent_params(k)
    out <- nn_faiss_nndescent_cpp(
      data,
      points,
      as.integer(k),
      as.integer(params$graph_k),
      as.integer(params$n_iter),
      as.integer(params$search_l),
      "euclidean",
      "euclidean",
      isTRUE(exclude_self),
      as.integer(n_threads)
    )
    result <- finish_nn_result(out, "faiss_nndescent", k, self_query, exact = FALSE, metric = metric)
    attr(result, "approximation") <- list(
      strategy = "faiss_IndexNNDescentFlat",
      backend = "faiss_nndescent",
      library = "faiss",
      metric = metric,
      graph_k = as.integer(out$graph_k),
      n_iter = as.integer(out$n_iter),
      search_l = as.integer(out$search_l),
      requested_graph_k = as.integer(out$requested_graph_k),
      requested_n_iter = as.integer(out$requested_n_iter),
      requested_search_l = as.integer(out$requested_search_l),
      nndescent_parameters_adjusted = isTRUE(out$nndescent_parameters_adjusted)
    )
    result <- append_nn_tuning_metadata(result, params)
    return(result)
  }

  if (backend %in% c("cuvs", "gpu_cuvs", "cuda_cuvs")) {
    require_cuvs_backend("cuVS")
    backend <- select_cuvs_auto_backend(
      self_query = self_query,
      n = nrow(data),
      p = ncol(data),
      n_points = nrow(points),
      k = k,
      work_size = work_size
    )
  }

  if (backend %in% c("cuda_cuvs_cagra", "cuda_cagra", "gpu_cagra")) {
    require_cuvs_backend("cuVS CAGRA")
    if (identical(metric, "inner_product")) {
      stop(
        "Direct RAPIDS cuVS CAGRA is disabled for `metric = \"inner_product\"` ",
        "because the current cuVS MIPS-to-L2 route can trigger an illegal CUDA ",
        "memory access. Use `cagra_implementation = \"faiss_gpu\"` when FAISS GPU ",
        "CAGRA is available, or choose a CUDA method with native inner-product ",
        "candidate refinement such as `method = \"nsg\"`, `\"vamana\"`, or ",
        "`\"nndescent\"`.",
        call. = FALSE
      )
    }
    metric_inputs <- NULL
    search_data <- data
    search_points <- points
    if (metric %in% c("cosine", "correlation")) {
      metric_inputs <- normalized_euclidean_metric_inputs(data, points, self_query, metric)
      search_data <- metric_inputs$data
      search_points <- metric_inputs$points
    } else if (identical(metric, "inner_product")) {
      metric_inputs <- mips_l2_metric_inputs(data, points, self_query)
      search_data <- metric_inputs$data
      search_points <- metric_inputs$points
    }
    params <- cuvs_cagra_params(nrow(data), k, p = ncol(data))
    build_algo <- cuvs_cagra_build_algo_for(search_data, k, self_query, params)
    tuning_metadata <- NULL
    if (isTRUE(cuvs_cagra_should_tune(search_data, k, self_query, tuning = tuning))) {
      tuned <- cuvs_cagra_tune_params(search_data, k, params, tuning = tuning, build_algo = build_algo)
      params <- tuned$params
      tuning_metadata <- tuned$tuning
      if (is.list(tuning_metadata) && !identical(tuning_metadata$status, "target_met")) {
        best_recall <- if (is.data.frame(tuning_metadata$results) && "recall" %in% names(tuning_metadata$results)) {
          suppressWarnings(max(tuning_metadata$results$recall, na.rm = TRUE))
        } else {
          NA_real_
        }
        min_recall <- faissr_option("cuvs_cagra_tune_min_recall", tuning_metadata$target_recall)
        min_recall <- suppressWarnings(as.numeric(min_recall))
        if (length(min_recall) != 1L || is.na(min_recall) || !is.finite(min_recall)) {
          min_recall <- 0.985
        }
        if (!is.finite(best_recall) || best_recall < min_recall) {
          stop(
            "cuVS CAGRA pilot tuning did not meet the requested recall target ",
            "on this dataset (best pilot recall = ",
            if (is.finite(best_recall)) formatC(best_recall, digits = 4, format = "f") else "NA",
            "). Use `backend = \"faiss_gpu_cagra\"`, ",
            "`backend = \"cuda_cuvs_bruteforce\"`, or explicitly disable ",
            "`faissR.cuvs_cagra_tune = FALSE` to force cuVS CAGRA.",
            call. = FALSE
          )
        }
      }
    }
    out <- nn_cuvs_cagra_cpp(
      search_data,
      search_points,
      as.integer(k),
      isTRUE(exclude_self),
      as.integer(params$graph_degree),
      as.integer(params$intermediate_graph_degree),
      as.integer(params$search_width),
      as.integer(params$itopk_size),
      build_algo
    )
    requested_graph_degree <- if (is.null(params$requested_graph_degree)) out$requested_graph_degree else params$requested_graph_degree
    requested_intermediate_graph_degree <- if (is.null(params$requested_intermediate_graph_degree)) out$requested_intermediate_graph_degree else params$requested_intermediate_graph_degree
    requested_search_width <- if (is.null(params$requested_search_width)) out$requested_search_width else params$requested_search_width
    requested_itopk_size <- if (is.null(params$requested_itopk_size)) out$requested_itopk_size else params$requested_itopk_size
    resolved_backend <- "cuda_cuvs_cagra"
    result_backend <- if (requested_backend %in% c("cuda", "gpu")) requested_backend else resolved_backend
    result <- finish_nn_result(out, result_backend, k, self_query, exact = FALSE, metric = metric)
    if (!identical(result_backend, resolved_backend)) {
      attr(result, "resolved_backend") <- resolved_backend
    }
    if (!is.null(metric_inputs)) {
      result <- finalize_graph_metric_result(result, metric_inputs)
    }
    attr(result, "approximation") <- list(
      strategy = "rapids_cuvs_cagra",
      backend = resolved_backend,
      library = "cuvs",
      accelerator = "cuda",
      cagra_provider = "cuvs",
      cagra_provider_option = cagra_implementation_preference(),
      metric = metric,
      transform = if (is.null(metric_inputs)) NA_character_ else metric_inputs$transform,
      distance_transform = if (is.null(metric_inputs)) NA_character_ else metric_inputs$distance_transform %||% "normalized_euclidean_squared_over_2_to_1_minus_similarity",
      graph_degree = as.integer(out$graph_degree),
      intermediate_graph_degree = as.integer(out$intermediate_graph_degree),
      search_width = as.integer(out$search_width),
      itopk_size = as.integer(out$itopk_size),
      cagra_build_algo = out$build_algo %||% cagra_build_algo_preference(),
      nn_descent_niter = as.integer(out$nn_descent_niter %||% NA_integer_),
      requested_graph_degree = as.integer(requested_graph_degree),
      requested_intermediate_graph_degree = as.integer(requested_intermediate_graph_degree),
      requested_search_width = as.integer(requested_search_width),
      requested_itopk_size = as.integer(requested_itopk_size),
      cagra_parameters_adjusted = isTRUE(out$cagra_parameters_adjusted) ||
        !identical(as.integer(requested_graph_degree), as.integer(out$graph_degree)) ||
        !identical(as.integer(requested_intermediate_graph_degree), as.integer(out$intermediate_graph_degree)) ||
        !identical(as.integer(requested_search_width), as.integer(out$search_width)) ||
        !identical(as.integer(requested_itopk_size), as.integer(out$itopk_size)),
      search_batch_size = as.integer(out$search_batch_size),
      tuning = tuning_metadata
    )
    result <- append_nn_tuning_metadata(result, params)
    return(result)
  }

  if (backend %in% c("cuda_cuvs_hnsw", "cuvs_hnsw")) {
    require_cuvs_backend("cuVS HNSW")
    if (identical(metric, "inner_product")) {
      stop(
        "CUDA cuVS HNSW is disabled for `metric = \"inner_product\"` because ",
        "the current cuVS HNSW route is built from a CAGRA L2 graph, and the ",
        "maximum-inner-product-to-L2 transform can trigger illegal CUDA memory ",
        "accesses. Use CPU HNSW for inner product, or a CUDA method with native ",
        "inner-product candidate refinement such as `method = \"nsg\"`, ",
        "`\"vamana\"`, or `\"nndescent\"`.",
        call. = FALSE
      )
    }
    metric_inputs <- NULL
    search_data <- data
    search_points <- points
    if (metric %in% c("cosine", "correlation")) {
      metric_inputs <- normalized_euclidean_metric_inputs(data, points, self_query, metric)
      search_data <- metric_inputs$data
      search_points <- metric_inputs$points
    } else if (identical(metric, "inner_product")) {
      metric_inputs <- mips_l2_metric_inputs(data, points, self_query)
      search_data <- metric_inputs$data
      search_points <- metric_inputs$points
    }
    params <- cuvs_hnsw_params(nrow(search_data), k, p = ncol(search_data), n_threads = n_threads)
    out <- nn_cuvs_hnsw_cpp(
      search_data,
      search_points,
      as.integer(k),
      isTRUE(exclude_self),
      as.integer(params$graph_degree),
      as.integer(params$intermediate_graph_degree),
      as.integer(params$ef),
      as.integer(params$n_threads),
      params$cagra_build_algo
    )
    result <- finish_nn_result(out, "cuda_cuvs_hnsw", k, self_query, exact = FALSE, metric = metric)
    if (!is.null(metric_inputs)) {
      result <- finalize_graph_metric_result(result, metric_inputs)
    }
    attr(result, "approximation") <- list(
      strategy = "rapids_cuvs_hnsw_from_cagra",
      backend = "cuda_cuvs_hnsw",
      library = "cuvs",
      accelerator = "cuda",
      metric = metric,
      transform = if (is.null(metric_inputs)) NA_character_ else metric_inputs$transform,
      distance_transform = if (is.null(metric_inputs)) NA_character_ else metric_inputs$distance_transform %||% "normalized_euclidean_squared_over_2_to_1_minus_similarity",
      graph_degree = as.integer(out$graph_degree),
      intermediate_graph_degree = as.integer(out$intermediate_graph_degree),
      ef = as.integer(out$ef),
      n_threads = as.integer(out$num_threads),
      cagra_build_algo = out$cagra_build_algo %||% params$cagra_build_algo,
      hnsw_build_algo = out$hnsw_build_algo %||% "from_cagra",
      hnsw_hierarchy = out$hnsw_hierarchy %||% "none",
      hnsw_m = as.integer(out$hnsw_m %||% NA_integer_),
      hnsw_ef_construction = as.integer(out$hnsw_ef_construction %||% NA_integer_),
      requested_graph_degree = as.integer(params$requested_graph_degree),
      requested_intermediate_graph_degree = as.integer(params$requested_intermediate_graph_degree),
      requested_ef = as.integer(params$requested_ef),
      requested_n_threads = as.integer(params$requested_n_threads),
      hnsw_parameters_adjusted = isTRUE(out$hnsw_parameters_adjusted),
      note = "cuVS HNSW is built from a CUDA CAGRA index and searched through the cuVS HNSW wrapper."
    )
    result <- append_nn_tuning_metadata(result, params)
    return(result)
  }

  if (backend %in% c("cuvs_ivf_flat", "cuda_cuvs_ivf_flat")) {
    require_cuvs_backend("cuVS IVF-Flat")
    metric_inputs <- NULL
    search_data <- data
    search_points <- points
    if (metric %in% c("cosine", "correlation")) {
      metric_inputs <- normalized_euclidean_metric_inputs(data, points, self_query, metric)
      search_data <- metric_inputs$data
      search_points <- metric_inputs$points
    } else if (identical(metric, "inner_product")) {
      metric_inputs <- mips_l2_metric_inputs(data, points, self_query)
      search_data <- metric_inputs$data
      search_points <- metric_inputs$points
    }
    params <- faiss_ivf_params(nrow(data), k, metric = metric)
    out <- nn_cuvs_ivf_flat_cpp(
      search_data,
      search_points,
      as.integer(k),
      as.integer(params$nlist),
      as.integer(params$nprobe),
      isTRUE(exclude_self)
    )
    result <- finish_nn_result(out, "cuda_cuvs_ivf_flat", k, self_query, exact = FALSE, metric = metric)
    if (!is.null(metric_inputs)) {
      result <- finalize_graph_metric_result(result, metric_inputs)
    }
    attr(result, "approximation") <- list(
      strategy = "rapids_cuvs_ivf_flat",
      backend = "cuda_cuvs_ivf_flat",
      library = "cuvs",
      accelerator = "cuda",
      metric = metric,
      transform = if (is.null(metric_inputs)) NA_character_ else metric_inputs$transform,
      metric_transform = if (is.null(metric_inputs)) NA_character_ else metric_inputs$transform,
      distance_transform = if (is.null(metric_inputs)) NA_character_ else (
        metric_inputs$distance_transform %||%
          "normalized_euclidean_squared_over_2_to_1_minus_similarity"
      ),
      default_candidate = FALSE,
      nlist = as.integer(out$n_lists),
      nprobe = as.integer(out$n_probes),
      requested_nlist = as.integer(params$requested_nlist),
      requested_nprobe = as.integer(params$requested_nprobe),
      ivf_parameters_adjusted = !identical(as.integer(params$requested_nlist), as.integer(out$n_lists)) ||
        !identical(as.integer(params$requested_nprobe), as.integer(out$n_probes)),
      search_batch_size = as.integer(out$search_batch_size)
    )
    result <- append_nn_tuning_metadata(result, params)
    return(result)
  }

  if (backend %in% c("cuvs_ivfpq", "cuda_cuvs_ivfpq", "cuvs_ivf_pq", "cuda_cuvs_ivf_pq")) {
    require_cuvs_backend("cuVS IVF-PQ")
    metric_inputs <- NULL
    search_data <- data
    search_points <- points
    if (metric %in% c("cosine", "correlation")) {
      metric_inputs <- normalized_euclidean_metric_inputs(data, points, self_query, metric)
      search_data <- metric_inputs$data
      search_points <- metric_inputs$points
    } else if (identical(metric, "inner_product")) {
      metric_inputs <- mips_l2_metric_inputs(data, points, self_query)
      search_data <- metric_inputs$data
      search_points <- metric_inputs$points
    }
    params <- faiss_ivf_params(nrow(data), k, metric = metric)
    pq <- cuvs_ivfpq_params(ncol(search_data))
    out <- nn_cuvs_ivf_pq_cpp(
      search_data,
      search_points,
      as.integer(k),
      as.integer(params$nlist),
      as.integer(params$nprobe),
      as.integer(pq$pq_dim),
      as.integer(pq$pq_bits),
      isTRUE(exclude_self)
    )
    result <- finish_nn_result(out, "cuda_cuvs_ivfpq", k, self_query, exact = FALSE, metric = metric)
    if (!is.null(metric_inputs)) {
      result <- finalize_graph_metric_result(result, metric_inputs)
    }
    attr(result, "approximation") <- list(
      strategy = "rapids_cuvs_ivf_pq",
      backend = "cuda_cuvs_ivfpq",
      library = "cuvs",
      accelerator = "cuda",
      metric = metric,
      transform = if (is.null(metric_inputs)) NA_character_ else metric_inputs$transform,
      metric_transform = if (is.null(metric_inputs)) NA_character_ else metric_inputs$transform,
      distance_transform = if (is.null(metric_inputs)) NA_character_ else (
        metric_inputs$distance_transform %||%
          "normalized_euclidean_squared_over_2_to_1_minus_similarity"
      ),
      role = "explicit_memory_pressure_backend",
      default_candidate = FALSE,
      nlist = as.integer(out$n_lists),
      nprobe = as.integer(out$n_probes),
      requested_nlist = as.integer(params$requested_nlist),
      requested_nprobe = as.integer(params$requested_nprobe),
      ivf_parameters_adjusted = !identical(as.integer(params$requested_nlist), as.integer(out$n_lists)) ||
        !identical(as.integer(params$requested_nprobe), as.integer(out$n_probes)),
      pq_dim = as.integer(out$pq_dim),
      pq_bits = as.integer(out$pq_bits),
      requested_pq_dim = as.integer(pq$requested_pq_dim),
      requested_pq_bits = as.integer(pq$requested_pq_bits),
      pq_parameters_adjusted = isTRUE(out$pq_parameters_adjusted) ||
        !identical(as.integer(pq$requested_pq_dim), as.integer(out$pq_dim)) ||
        !identical(as.integer(pq$requested_pq_bits), as.integer(out$pq_bits)),
      search_batch_size = as.integer(out$search_batch_size)
    )
    result <- append_nn_tuning_metadata(result, params, pq, .prefixes = list(NULL, "pq_"))
    return(result)
  }

  if (backend %in% c("cuvs_bruteforce", "cuda_cuvs_bruteforce", "cuda_cuvs_exact")) {
    require_cuvs_backend("cuVS brute-force")
    metric_inputs <- NULL
    search_data <- data
    search_points <- points
    if (metric %in% c("cosine", "correlation")) {
      metric_inputs <- normalized_euclidean_metric_inputs(data, points, self_query, metric)
      search_data <- metric_inputs$data
      search_points <- metric_inputs$points
    } else if (identical(metric, "inner_product")) {
      metric_inputs <- mips_l2_metric_inputs(data, points, self_query)
      search_data <- metric_inputs$data
      search_points <- metric_inputs$points
    }
    out <- nn_cuvs_bruteforce_cpp(
      search_data,
      search_points,
      as.integer(k),
      isTRUE(exclude_self)
    )
    resolved_backend <- "cuda_cuvs_bruteforce"
    result_backend <- if (requested_backend %in% c("cuda", "gpu")) requested_backend else resolved_backend
    result <- finish_nn_result(out, result_backend, k, self_query, exact = TRUE, metric = metric)
    if (!is.null(metric_inputs)) {
      result <- finalize_graph_metric_result(result, metric_inputs)
    }
    if (!identical(result_backend, resolved_backend)) {
      attr(result, "resolved_backend") <- resolved_backend
    }
    attr(result, "cuvs") <- list(
      index_type = as.character(out$index_type),
      library = "cuvs",
      backend = "cuda",
      resolved_backend = resolved_backend,
      metric = metric,
      transform = if (is.null(metric_inputs)) NA_character_ else metric_inputs$transform,
      distance_transform = if (is.null(metric_inputs)) NA_character_ else (
        metric_inputs$distance_transform %||%
          "normalized_euclidean_squared_over_2_to_1_minus_similarity"
      )
    )
    return(result)
  }

  if (identical(backend, "cuda_native_nndescent")) {
    if (!identical(metric, "inner_product")) {
      stop("`backend = \"cuda_native_nndescent\"` is reserved for raw inner-product search.", call. = FALSE)
    }
    if (!isTRUE(self_query)) {
      stop("`backend = \"cuda_native_nndescent\"` is only available for self-KNN searches.", call. = FALSE)
    }
    nonself_k <- if (isTRUE(exclude_self)) k else max(0L, k - 1L)
    if (nonself_k < 1L) {
      out <- list(
        indices = matrix(seq_len(nrow(data)), nrow(data), 1L),
        distances = matrix(0, nrow(data), 1L)
      )
    } else {
      out <- gpu_nndescent_self_knn(
        data,
        k = nonself_k,
        backend = "cuda",
        seed = fast_knn_approx_seed(),
        metric = "inner_product"
      )
      if (!isTRUE(exclude_self)) {
        out$indices <- cbind(seq_len(nrow(data)), out$indices)
        out$distances <- cbind(rep(0, nrow(data)), out$distances)
      }
    }
    result <- finish_nn_result(out, "cuda_native_nndescent", k, self_query, exact = FALSE, metric = metric)
    attr(result, "approximation") <- c(
      list(
        strategy = "native_cuda_nndescent_candidate_refine",
        backend = "cuda_native_nndescent",
        accelerator = "cuda",
        metric = metric,
        metric_strategy = "euclidean_seed_inner_product_refine"
      ),
      attr(out, "approximation", exact = TRUE)
    )
    return(result)
  }

  if (backend %in% c("cuvs_nndescent", "cuda_cuvs_nndescent")) {
    if (identical(metric, "inner_product")) {
      stop("cuVS NN-descent does not support `metric = \"inner_product\"`.", call. = FALSE)
    }
    require_cuvs_backend("cuVS NN-descent")
    if (!isTRUE(self_query)) {
      stop("`backend = \"cuda_cuvs_nndescent\"` is only available for self-KNN searches.", call. = FALSE)
    }
    metric_inputs <- NULL
    search_data <- data
    if (metric %in% c("cosine", "correlation")) {
      metric_inputs <- normalized_euclidean_metric_inputs(data, points, self_query, metric)
      search_data <- metric_inputs$data
    }
    nonself_k <- if (isTRUE(exclude_self)) k else max(0L, k - 1L)
    params <- NULL
    if (nonself_k < 1L) {
      out <- list(
        indices = matrix(seq_len(nrow(data)), nrow(data), 1L),
        distances = matrix(0, nrow(data), 1L)
      )
    } else {
      params <- cuvs_nndescent_params(nrow(search_data), nonself_k)
      out <- nn_cuvs_nndescent_self_cpp(
        search_data,
        as.integer(nonself_k),
        as.integer(params$graph_degree),
        as.integer(params$intermediate_graph_degree),
        as.integer(params$max_iterations)
      )
      if (!isTRUE(exclude_self)) {
        out$indices <- cbind(seq_len(nrow(data)), out$indices)
        out$distances <- cbind(rep(0, nrow(data)), out$distances)
      }
    }
    result <- finish_nn_result(out, "cuda_cuvs_nndescent", k, self_query, exact = FALSE, metric = metric)
    if (!is.null(metric_inputs)) {
      result <- finalize_normalized_euclidean_metric_result(result, metric_inputs)
    }
    attr(result, "approximation") <- list(
      strategy = "rapids_cuvs_nndescent",
      backend = "cuda_cuvs_nndescent",
      library = "cuvs",
      metric = metric,
      transform = if (is.null(metric_inputs)) NA_character_ else metric_inputs$transform,
      graph_degree = as.integer(out$graph_degree),
      intermediate_graph_degree = as.integer(out$intermediate_graph_degree),
      max_iterations = as.integer(out$max_iterations)
    )
    result <- append_nn_tuning_metadata(result, params)
    return(result)
  }

  if (identical(backend, "cpu_nndescent")) {
    if (!isTRUE(self_query)) {
      stop("`method = \"nndescent\"` is only available for self-KNN searches on CPU.", call. = FALSE)
    }
    metric_inputs <- NULL
    search_data <- data
    if (metric %in% c("cosine", "correlation")) {
      metric_inputs <- normalized_euclidean_metric_inputs(data, points, self_query, metric)
      search_data <- metric_inputs$data
    }
    nonself_k <- if (isTRUE(exclude_self)) k else k - 1L
    if (nonself_k < 1L) {
      out <- list(
        indices = matrix(seq_len(nrow(data)), nrow(data), 1L),
        distances = matrix(0, nrow(data), 1L)
      )
      attr(out, "approximation") <- list(
        strategy = "native_cpu_nndescent_trivial_self",
        backend = "cpu"
      )
    } else {
      out <- nndescent_self_knn(
        search_data,
        k = nonself_k,
        seed = fast_knn_approx_seed(),
        n_threads = n_threads,
        metric = if (is.null(metric_inputs)) metric else "euclidean"
      )
      if (!isTRUE(exclude_self)) {
        out$indices <- cbind(seq_len(nrow(data)), out$indices)
        out$distances <- cbind(rep(0, nrow(data)), out$distances)
      }
    }
    result <- finish_nn_result(out, "cpu_nndescent", k, self_query, exact = FALSE, metric = metric)
    if (!is.null(metric_inputs)) {
      result <- finalize_normalized_euclidean_metric_result(result, metric_inputs)
    }
    approximation <- attr(out, "approximation")
    if (is.null(approximation)) approximation <- list()
    approximation$metric <- metric
    approximation$transform <- if (is.null(metric_inputs)) NA_character_ else metric_inputs$transform
    attr(result, "approximation") <- approximation
    return(result)
  }

  if (backend %in% c("hnsw", "rcpphnsw", "cpu_hnsw")) {
    out <- rcpphnsw_knn(
      data,
      points,
      k = k,
      self_query = self_query,
      exclude_self = isTRUE(exclude_self),
      metric = metric,
      n_threads = n_threads
    )
    result <- finish_nn_result(out, "hnsw", k, self_query, exact = FALSE, metric = metric)
    attr(result, "approximation") <- attr(out, "approximation")
    return(result)
  }


  if (backend %in% c("cuda_grid", "cuda_grid_auto", "gpu_grid",
                     "cuda_grid2d", "cuda_grid3d")) {
    if (!isTRUE(self_query)) {
      stop("`backend = \"cuda_grid_auto\"` is only available for self-KNN searches.", call. = FALSE)
    }
    if (identical(metric, "inner_product")) {
      stop("`backend = \"cuda_grid_auto\"` does not support inner-product search.", call. = FALSE)
    }
    if (!ncol(data) %in% c(2L, 3L)) {
      stop("`backend = \"cuda_grid_auto\"` supports only two- or three-column matrices.", call. = FALSE)
    }
    if (!isTRUE(cuda_available())) {
      stop("No CUDA GPU backend is available on this machine.", call. = FALSE)
    }
    metric_inputs <- NULL
    search_data <- data
    if (metric %in% c("cosine", "correlation")) {
      metric_inputs <- normalized_euclidean_metric_inputs(data, points, self_query, metric)
      search_data <- metric_inputs$data
    }
    nonself_k <- if (isTRUE(exclude_self)) k else k - 1L
    bins <- grid_bins_per_dim(nrow(search_data), nonself_k, ncol(search_data))
    out <- cuda_grid_self_knn_cpp(
      search_data,
      as.integer(nonself_k),
      as.integer(bins)
    )
    if (!isTRUE(exclude_self)) {
      out$indices <- cbind(seq_len(nrow(data)), out$indices)
      out$distances <- cbind(rep(0, nrow(data)), out$distances)
    }
    resolved <- if (ncol(data) == 3L) "cuda_grid3d" else "cuda_grid2d"
    result <- finish_nn_result(out, resolved, k, self_query, exact = TRUE, metric = metric)
    if (!is.null(metric_inputs)) {
      result <- finalize_normalized_euclidean_metric_result(result, metric_inputs)
    }
    attr(result, "spatial_index") <- list(
      strategy = if (ncol(data) == 3L) "native_cuda_exact_uniform_grid_3d" else "native_cuda_exact_uniform_grid_2d",
      backend = resolved,
      exact = TRUE,
      metric_transform = if (is.null(metric_inputs)) NA_character_ else metric_inputs$transform,
      bins_per_dim = as.integer(out$bins_per_dim),
      n_cells = as.integer(out$n_cells)
    )
    return(result)
  }

  if (backend %in% c("grid", "cpu_grid", "grid2d", "cpu_grid2d", "grid3d", "cpu_grid3d")) {
    if (!isTRUE(self_query)) {
      stop("`backend = \"cpu_grid\"` is only available for self-KNN searches.", call. = FALSE)
    }
    if (identical(metric, "inner_product")) {
      stop("`backend = \"cpu_grid\"` does not support inner-product search.", call. = FALSE)
    }
    metric_inputs <- NULL
    search_data <- data
    if (metric %in% c("cosine", "correlation")) {
      metric_inputs <- normalized_euclidean_metric_inputs(data, points, self_query, metric)
      search_data <- metric_inputs$data
    }
    grid_backend <- backend
    if (!is.null(metric_inputs) && backend %in% c("grid", "cpu_grid")) {
      grid_backend <- if (ncol(search_data) == 3L) "cpu_grid3d" else "cpu_grid2d"
    }
    out <- grid_self_knn(
      search_data,
      k = k,
      backend = grid_backend,
      exclude_self = isTRUE(exclude_self),
      n_threads = n_threads
    )
    result <- finish_nn_result(out, attr(out, "spatial_index")$backend, k, self_query, exact = TRUE, metric = metric)
    if (!is.null(metric_inputs)) {
      result <- finalize_normalized_euclidean_metric_result(result, metric_inputs)
    }
    attr(result, "spatial_index") <- attr(out, "spatial_index")
    attr(result, "spatial_index")$metric_transform <- if (is.null(metric_inputs)) {
      NA_character_
    } else {
      metric_inputs$transform
    }
    return(result)
  }

  selected_gpu <- NA_character_
  if (backend == "cuda") {
    selected_gpu <- "cuda"
  } else if (backend == "gpu") {
    selected_gpu <- if (isTRUE(cuda_available())) "cuda" else NA_character_
  }

  if (!is.na(selected_gpu)) {
    gpu_k <- if (isTRUE(exclude_self)) k + 1L else k
    if (gpu_k > 256L) {
      stop("Native GPU backends currently support `k <= 256`.", call. = FALSE)
    }
    if (selected_gpu == "cuda") {
      if (!isTRUE(cuda_available())) {
        stop("No CUDA GPU backend is available on this machine.", call. = FALSE)
      }
      out <- nn_cuda_cpp(data, points, as.integer(gpu_k), FALSE)
      if (isTRUE(exclude_self)) {
        out <- drop_self_knn_result(out, k)
      }
      return(finish_nn_result(out, "cuda", k, self_query, exact = TRUE, metric = metric))
    }
    stop("No CUDA KNN backend is available on this machine.", call. = FALSE)
  }

  out <- nn_cpp(
    data,
    points,
    as.integer(k),
    metric,
    FALSE,
    FALSE,
    0,
    TRUE,
    as.integer(n_threads),
    isTRUE(exclude_self)
  )
  finish_nn_result(out, "cpu", k, self_query, metric = metric)
}

normalize_scalar_choice_arg <- function(x, arg, default, formal_choices = NULL) {
  value <- trimws(as.character(x))
  value <- value[nzchar(value)]
  if (!length(value)) return(default)
  if (length(value) > 1L) {
    if (!is.null(formal_choices) && identical(value, formal_choices)) {
      return(default)
    }
    stop("`", arg, "` must be a single value.", call. = FALSE)
  }
  value[[1L]]
}

normalize_scalar_logical_arg <- function(x, arg, default = FALSE) {
  if (is.null(x) || !length(x)) return(isTRUE(default))
  if (length(x) != 1L || is.na(x)) {
    stop("`", arg, "` must be a single TRUE or FALSE value.", call. = FALSE)
  }
  if (!is.logical(x)) {
    stop("`", arg, "` must be a single TRUE or FALSE value.", call. = FALSE)
  }
  isTRUE(x)
}

normalize_public_compute_backend <- function(backend, arg = "backend") {
  backend <- normalize_scalar_choice_arg(
    backend,
    arg = arg,
    default = "auto",
    formal_choices = c("auto", "cpu", "cuda")
  )
  if (is.na(backend) || !nzchar(backend)) backend <- "auto"
  backend <- tolower(backend)
  if (!backend %in% c("auto", "cpu", "cuda")) {
    stop("`", arg, "` must be one of \"auto\", \"cpu\", or \"cuda\".", call. = FALSE)
  }
  if (identical(backend, "auto")) {
    if (isTRUE(cuda_available()) || isTRUE(cuvs_available())) {
      return("cuda")
    }
    return("cpu")
  }
  backend
}

normalize_public_backend_arg <- function(backend, arg = "backend") {
  backend <- normalize_scalar_choice_arg(
    backend,
    arg = arg,
    default = "auto",
    formal_choices = c("auto", "cpu", "cuda")
  )
  if (is.na(backend) || !nzchar(backend)) backend <- "auto"
  backend <- tolower(backend)
  if (!backend %in% c("auto", "cpu", "cuda")) {
    stop("`", arg, "` must be one of \"auto\", \"cpu\", or \"cuda\".", call. = FALSE)
  }
  backend
}

normalize_nn_method <- function(method) {
  method <- normalize_scalar_choice_arg(
    method,
    arg = "method",
    default = "auto",
    formal_choices = nn_method_labels()
  )
  if (is.na(method) || !nzchar(method)) method <- "auto"
  method <- trimws(method)
  labels <- nn_method_labels()
  if (!method %in% labels) {
    stop(
      "`method` must be one of \"auto\", \"exact\", \"flat\", \"bruteforce\", ",
      "\"grid\", \"hnsw\", \"ivf\", \"ivfpq\", \"vamana\", ",
      "\"nsg\", \"nndescent\", or \"cagra\".",
      " Use these canonical lowercase method labels; internal backend route ",
      "labels such as \"faiss_hnsw\" are not public `method` values.",
      call. = FALSE
    )
  }
  method
}

validate_public_nn_method_shape <- function(data, method) {
  if (!identical(method, "grid")) return(invisible(TRUE))
  p <- suppressWarnings(as.integer(ncol(data)))
  if (length(p) != 1L || is.na(p) || !p %in% c(2L, 3L)) {
    stop(
      "`method = \"grid\"` supports only two- or three-column matrices. ",
      "Use `method = \"auto\"` to let faissR select a non-grid method for ",
      "higher-dimensional data.",
      call. = FALSE
    )
  }
  invisible(TRUE)
}

nn_metric_labels <- function() {
  c("euclidean", "cosine", "correlation", "inner_product")
}

nn_method_labels <- function() {
  c(
    "auto", "exact", "flat", "bruteforce", "grid",
    "hnsw", "ivf", "ivfpq", "vamana", "nsg", "nndescent", "cagra"
  )
}

faissr_option <- function(name, default = NULL) {
  name <- as.character(name)
  for (key in paste0("faissR.", name)) {
    value <- getOption(key, NULL)
    if (!is.null(value)) return(value)
  }
  default
}

normalize_cagra_implementation_value <- function(value, default = "auto", arg = NULL, strict = FALSE) {
  if (is.null(value)) value <- default
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
        "`", arg, "` must be one of \"auto\", \"faiss_gpu\", or \"cuvs\".",
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
  if (is.null(value)) return(NULL)
  normalize_cagra_implementation_value(
    value,
    default = "auto",
    arg = "cagra_implementation",
    strict = TRUE
  )
}

set_call_cagra_implementation <- function(value) {
  value <- normalize_cagra_implementation_arg(value)
  if (is.null(value)) return(invisible(FALSE))
  old <- getOption("faissR.cagra_implementation")
  options(faissR.cagra_implementation = value)
  parent <- parent.frame()
  do.call(
    on.exit,
    list(substitute(options(faissR.cagra_implementation = OLD), list(OLD = old)), add = TRUE),
    envir = parent
  )
  invisible(TRUE)
}

normalize_cagra_build_algo_value <- function(value, default = "auto", arg = NULL, strict = FALSE) {
  if (is.null(value)) value <- default
  value <- tolower(gsub("[[:space:]-]+", "_", as.character(value)[1L]))
  value <- gsub("_+", "_", value)
  value <- gsub("^_|_$", "", value)
  if (length(value) != 1L || is.na(value) || !nzchar(value)) value <- ""
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
        "`", arg, "` must be one of \"auto\", \"ivf_pq\", ",
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

cuvs_cagra_build_algo_for_shape <- function(n, p, k, self_query, params = NULL) {
  requested <- cagra_build_algo_preference()
  nn_tune_cuvs_cagra_build_algo_cpp(
    as.integer(n),
    suppressWarnings(as.integer(p)),
    as.integer(k),
    isTRUE(self_query),
    isTRUE(params$tuning_compact_build %||% FALSE),
    requested
  )
}

normalize_cagra_build_algo_arg <- function(value) {
  if (is.null(value)) return(NULL)
  normalize_cagra_build_algo_value(
    value,
    default = "auto",
    arg = "cagra_build_algo",
    strict = TRUE
  )
}

set_call_cagra_build_algo <- function(value) {
  value <- normalize_cagra_build_algo_arg(value)
  if (is.null(value)) return(invisible(FALSE))
  old <- getOption("faissR.cuvs_cagra_build_algo")
  options(faissR.cuvs_cagra_build_algo = value)
  parent <- parent.frame()
  do.call(
    on.exit,
    list(substitute(options(faissR.cuvs_cagra_build_algo = OLD), list(OLD = old)), add = TRUE),
    envir = parent
  )
  invisible(TRUE)
}

cuda_cagra_auto_prefers_cuvs <- function(n = NULL,
                                         p = NULL,
                                         k = NULL,
                                         self_query = NULL) {
  if (!isTRUE(self_query)) return(FALSE)
  if (is.null(n) || is.null(p) || is.null(k)) return(FALSE)
  n <- suppressWarnings(as.numeric(n))
  p <- suppressWarnings(as.numeric(p))
  k <- suppressWarnings(as.numeric(k))
  if (length(n) != 1L || length(p) != 1L || length(k) != 1L) return(FALSE)
  if (!is.finite(n) || !is.finite(p) || !is.finite(k)) return(FALSE)
  compact_n <- faiss_option_int("cuda_cagra_cuvs_compact_n", 10000L, min_value = 100L, max_value = 1000000L)
  high_dim_p <- faiss_option_int("cuda_cagra_cuvs_high_dim_p", 1024L, min_value = 2L, max_value = 100000L)
  max_k <- faiss_option_int("cuda_cagra_cuvs_compact_max_k", 128L, min_value = 1L, max_value = 10000L)
  n <= compact_n && p >= high_dim_p && k <= max_k
}

resolve_cuda_cagra_backend <- function(faiss_gpu_available_value = faiss_gpu_available(),
                                       cuvs_available_value = cuvs_available(),
                                       n = NULL,
                                       p = NULL,
                                       k = NULL,
                                       self_query = NULL) {
  preference <- cagra_implementation_preference()
  if (identical(preference, "faiss_gpu")) {
    return("faiss_gpu_cagra")
  }
  if (identical(preference, "cuvs")) {
    return("cuda_cuvs_cagra")
  }
  if (isTRUE(faiss_gpu_available_value) && isTRUE(cuvs_available_value) &&
      isTRUE(cuda_cagra_auto_prefers_cuvs(n = n, p = p, k = k, self_query = self_query))) {
    return("cuda_cuvs_cagra")
  }
  if (isTRUE(faiss_gpu_available_value)) {
    "faiss_gpu_cagra"
  } else {
    "cuda_cuvs_cagra"
  }
}

cuda_cagra_route_available <- function(faiss_gpu_available_value = faiss_gpu_available(),
                                       cuvs_available_value = cuvs_available(),
                                       n = NULL,
                                       p = NULL,
                                       k = NULL,
                                       self_query = NULL) {
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
#' faissR treats `metric = "inner_product"` as raw-dot-product ranking while
#' keeping returned `distances` in the usual smaller-is-better orientation via
#' per-query shifted dot-product distances.
#' Direct cuVS NNDescent does not expose raw inner-product search; public CUDA
#' `method = "nndescent", metric = "inner_product"` uses faissR's native CUDA
#' candidate-refinement route. Public CUDA HNSW and CAGRA inner-product routes
#' are disabled because the available cuVS HNSW route is CAGRA-derived and the
#' transformed raw-inner-product graph route is not reliable across k values in
#' the current implementation.
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
#' @return A data frame with one row per public `method`, `backend` (`"auto"`,
#'   `"cpu"`, or `"cuda"`), and `metric` combination. Columns include
#'   `supported`, `exact`, `implementation`, and `notes`. If `runtime = TRUE`,
#'   runtime availability columns are appended.
#' @examples
#' caps <- nn_capabilities()
#' subset(caps, method == "flat" & supported)
#' @export
nn_capabilities <- function(runtime = FALSE) {
  runtime <- normalize_scalar_logical_arg(runtime, "runtime", default = FALSE)
  methods <- nn_method_labels()
  backends <- c("auto", "cpu", "cuda")
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
    resolve_public_nn_backend(row$backend[[1L]], row$method[[1L]], row$metric[[1L]]),
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
  if (identical(resolved, "cuda_cuvs_cagra") &&
      identical(normalize_nn_metric(row$metric[[1L]]), "inner_product")) {
    return(data.frame(
      resolved_backend = resolved,
      runtime_available = FALSE,
      runtime_reason = "unsupported_cuvs_cagra_inner_product",
      runtime_notes = paste(
        "Direct RAPIDS cuVS CAGRA is disabled for raw inner-product search",
        "because the current MIPS-to-L2 route can trigger an illegal CUDA",
        "memory access. Force FAISS GPU CAGRA when available or use a CUDA",
        "method with native inner-product candidate refinement."
      ),
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
    runtime_reason = availability$reason %||% if (isTRUE(availability$available)) "available" else "unavailable_runtime",
    runtime_notes = availability$notes,
    stringsAsFactors = FALSE
  )
}

nn_cuda_auto_runtime_available <- function(metric,
                                           cuda_available_value = cuda_available(),
                                           cuvs_available_value = cuvs_available(),
                                           faiss_gpu_available_value = faiss_gpu_available()) {
  metric <- normalize_nn_metric(metric)
  if (identical(metric, "euclidean")) {
    ok <- isTRUE(cuda_available_value) ||
      isTRUE(cuvs_available_value) ||
      isTRUE(faiss_gpu_available_value)
    return(list(
      available = ok,
      reason = if (ok) "available" else "missing_cuda_route",
      notes = if (ok) {
        "CUDA auto Euclidean route is available through native CUDA, FAISS GPU, or cuVS."
      } else {
        "CUDA auto Euclidean route requires native CUDA, FAISS GPU, or cuVS support."
      }
    ))
  }
  if (identical(metric, "inner_product")) {
    ok <- isTRUE(faiss_gpu_available_value) ||
      isTRUE(cuda_available_value)
    return(list(
      available = ok,
      reason = if (ok) "available" else "missing_cuda_route",
      notes = if (isTRUE(faiss_gpu_available_value)) {
        "CUDA auto raw-inner-product route is available through FAISS GPU Flat IP."
      } else if (isTRUE(cuda_available_value) && isTRUE(cuvs_available_value)) {
        paste(
          "CUDA auto raw-inner-product route is shape-dependent on this runtime:",
          "large self-KNN graph searches can use faissR's native CUDA",
          "candidate-refinement route, while explicit CUDA exact/brute-force",
          "calls can use transformed cuVS brute force."
        )
      } else if (isTRUE(cuda_available_value)) {
        paste(
          "CUDA auto raw-inner-product route is shape-dependent on this runtime:",
          "self-KNN graph searches can use faissR's native CUDA",
          "candidate-refinement route, while exact or query inner-product search",
          "requires FAISS GPU Flat or cuVS brute force."
        )
      } else if (isTRUE(cuvs_available_value)) {
        paste(
          "A cuVS runtime was detected, but CUDA auto raw-inner-product search",
          "does not yet select the transformed cuVS brute-force route.",
          "Explicit CUDA exact/brute-force calls can use transformed direct",
          "cuVS brute force; CAGRA/HNSW raw-inner-product graph routes are",
          "disabled in faissR."
        )
      } else {
        "CUDA auto raw-inner-product search requires FAISS GPU Flat IP, transformed cuVS brute force, or faissR's native CUDA self-KNN candidate-refinement route."
      }
    ))
  }
  ok <- isTRUE(faiss_gpu_available_value) ||
    isTRUE(cuvs_available_value) ||
    isTRUE(cuda_available_value)
  list(
    available = ok,
    reason = if (ok) "available" else "missing_cuda_route",
    notes = if (isTRUE(faiss_gpu_available_value)) {
      "CUDA auto non-Euclidean route is available through FAISS GPU Flat metric-aware search."
    } else if (isTRUE(cuvs_available_value) && metric %in% c("cosine", "correlation")) {
      paste(
        "CUDA auto non-Euclidean route is shape-dependent on this runtime:",
        "large self-KNN graph searches can use cuVS graph routes, and",
        "explicit exact/brute-force calls can use transformed cuVS brute force."
      )
    } else if (isTRUE(cuda_available_value) && metric %in% c("cosine", "correlation")) {
      paste(
        "CUDA auto non-Euclidean route is shape-dependent on this runtime:",
        "native CUDA grid may apply to eligible 2D/3D self-search datasets,",
        "while general exact non-Euclidean search still requires FAISS GPU Flat."
      )
    } else {
      "CUDA auto non-Euclidean route requires FAISS GPU Flat, cuVS graph support, or an eligible native CUDA grid route."
    }
  )
}

nn_resolved_backend_available <- function(backend) {
  backend <- as.character(backend)[1L]
  if (is.na(backend) || !nzchar(backend)) {
    return(list(available = FALSE, reason = "missing_resolved_backend", notes = "No resolved backend."))
  }
  if (backend %in% c(
    "auto", "cpu", "cpu_auto", "cpu_grid",
    "cpu_nndescent", "cpu_approx", "grid", "grid2d", "grid3d",
    "cpu_grid2d", "cpu_grid3d"
  )) {
    return(list(available = TRUE, reason = "available", notes = "Native CPU route is available."))
  }
  if (identical(backend, "hnsw")) {
    ok <- requireNamespace("RcppHNSW", quietly = TRUE)
    return(list(
      available = ok,
      reason = if (ok) "available" else "missing_rcpphnsw",
      notes = if (ok) "RcppHNSW fallback is available." else "RcppHNSW fallback is not installed."
    ))
  }
  if (startsWith(backend, "faiss_gpu")) {
    ok <- isTRUE(faiss_gpu_available())
    return(list(
      available = ok,
      reason = if (ok) "available" else "missing_faiss_gpu",
      notes = if (ok) "FAISS GPU route is available." else "FAISS GPU support is not available in this build."
    ))
  }
  if (startsWith(backend, "cuda_cuvs") || startsWith(backend, "cuvs")) {
    ok <- isTRUE(cuvs_available())
    return(list(
      available = ok,
      reason = if (ok) "available" else "missing_cuvs",
      notes = if (ok) "cuVS route is available." else "cuVS support is not available in this build."
    ))
  }
  if (startsWith(backend, "cuda")) {
    ok <- isTRUE(cuda_available())
    return(list(
      available = ok,
      reason = if (ok) "available" else "missing_cuda",
      notes = if (ok) "Native CUDA route is available." else "Native CUDA support is not available in this build."
    ))
  }
  if (startsWith(backend, "faiss")) {
    ok <- isTRUE(faiss_available())
    return(list(
      available = ok,
      reason = if (ok) "available" else "missing_faiss",
      notes = if (ok) "FAISS CPU route is available." else "FAISS CPU support is not available in this build."
    ))
  }
  list(available = TRUE, reason = "available", notes = "No additional runtime dependency detected.")
}

nn_capability_row <- function(method, backend, metric) {
  supported <- FALSE
  exact <- NA
  implementation <- NA_character_
  notes <- NA_character_

  all_metrics <- metric %in% nn_metric_labels()
  euclidean <- identical(metric, "euclidean")
  non_ip_metric <- metric %in% c("euclidean", "cosine", "correlation")

  if (identical(backend, "auto")) {
    cpu <- nn_capability_row(method, "cpu", metric)
    cuda <- nn_capability_row(method, "cuda", metric)
    supported <- isTRUE(cpu$supported[[1L]]) || isTRUE(cuda$supported[[1L]])
    exact_values <- c(
      if (isTRUE(cpu$supported[[1L]])) cpu$exact[[1L]] else NA,
      if (isTRUE(cuda$supported[[1L]])) cuda$exact[[1L]] else NA
    )
    exact_values <- exact_values[!is.na(exact_values)]
    exact <- if (length(exact_values)) all(as.logical(exact_values)) else NA
    implementation <- "runtime CPU/CUDA selector"
    notes <- if (supported) {
      paste(
        "Auto backend uses CUDA only when the selected method/metric has a",
        "validated CUDA route and the required runtime is available; otherwise",
        "it uses the CPU route when one exists."
      )
    } else {
      paste(
        "No CPU or CUDA route is exposed for this public method/metric",
        "combination."
      )
    }
  } else if (identical(method, "auto")) {
    if (identical(backend, "cpu")) {
      supported <- all_metrics
      exact <- NA
      implementation <- "shape-aware CPU selector"
      notes <- "Euclidean can resolve to exact, grid, FAISS IVF, FAISS HNSW, or native CPU NN-descent fallback; non-Euclidean resolves to exact, FAISS Flat, FAISS HNSW, RcppHNSW/hnswlib, or native CPU NN-descent fallback depending on shape and availability."
    } else if (all_metrics) {
      supported <- TRUE
      exact <- NA
      implementation <- "shape-aware CUDA selector"
      notes <- if (euclidean) {
        "Can resolve to CUDA grid, FAISS GPU Flat, cuVS brute force, FAISS GPU CAGRA, or cuVS approximate routes depending on shape and availability."
      } else if (metric %in% c("cosine", "correlation")) {
        "CUDA auto can resolve to CUDA grid for large 2D/3D self-search, FAISS GPU Flat normalized-IP search for exact small/query workloads, or cuVS HNSW/CAGRA graph routes for large self-KNN when available; cuVS-only runtimes are shape-dependent and keep small/query workloads on CPU."
      } else {
        "CUDA auto uses FAISS GPU Flat IP for exact small/query inner-product search when FAISS GPU Flat is available, or faissR's native CUDA candidate-refinement route for large self-KNN when native CUDA kernels are available. cuVS CAGRA-derived raw-inner-product graph routes are disabled."
      }
    }
  } else if (method %in% c("exact", "bruteforce")) {
    supported <- all_metrics
    exact <- TRUE
    if (identical(backend, "cpu")) {
      implementation <- "native CPU exact"
      notes <- "CPU exact scorer supports all public metrics."
    } else {
      if (euclidean) {
        implementation <- "FAISS GPU Flat or cuVS brute force"
        notes <- "Euclidean CUDA exact/brute-force search can use FAISS GPU Flat when available, otherwise direct cuVS brute force when cuVS is available."
      } else {
        implementation <- "FAISS GPU Flat or transformed cuVS brute force"
        notes <- "CUDA cosine, correlation, and inner-product exact/brute-force search can use FAISS GPU Flat metric-aware routes or direct cuVS brute force with exact metric transforms."
      }
    }
  } else if (identical(method, "flat")) {
    supported <- all_metrics
    exact <- TRUE
    implementation <- if (identical(backend, "cpu")) "FAISS CPU Flat" else "FAISS GPU Flat"
    notes <- "Cosine uses row L2 normalization plus Flat IP; correlation uses row centering plus L2 normalization plus Flat IP."
  } else if (identical(method, "grid")) {
    supported <- non_ip_metric
    exact <- if (supported) TRUE else NA
    implementation <- if (identical(backend, "cpu")) "native CPU 2D/3D grid" else "native CUDA 2D/3D grid"
    notes <- if (supported) {
      "Only valid for 2D/3D self-KNN. Cosine/correlation use normalized Euclidean grid search."
    } else {
      "Grid search does not expose raw inner-product search."
    }
  } else if (identical(method, "hnsw")) {
    supported <- if (identical(backend, "cpu")) all_metrics else non_ip_metric
    exact <- if (supported) FALSE else NA
    implementation <- if (identical(backend, "cpu")) {
      "FAISS HNSW or RcppHNSW/hnswlib"
    } else {
      "cuVS CUDA HNSW from CAGRA"
    }
    notes <- if (identical(backend, "cpu")) {
      "Uses FAISS HNSW for all metrics when available; cosine and correlation use normalized inner-product HNSW. Falls back to RcppHNSW/hnswlib when FAISS is unavailable."
    } else if (identical(metric, "inner_product")) {
      "CUDA cuVS HNSW is disabled for raw inner product until a reliable native or transformed route is available."
    } else if (supported) {
      "Uses RAPIDS cuVS HNSW converted from a CUDA CAGRA index; cosine/correlation use normalized Euclidean search."
    } else {
      "Unsupported CUDA HNSW metric."
    }
  } else if (method %in% c("ivf", "ivfpq")) {
    supported <- all_metrics
    exact <- if (supported) FALSE else NA
    implementation <- if (identical(backend, "cpu")) {
      if (identical(method, "ivf")) "FAISS CPU IVF-Flat" else "FAISS CPU IVF-PQ"
    } else {
      if (identical(method, "ivf")) "FAISS GPU IVF-Flat" else "FAISS GPU IVF-PQ"
    }
    notes <- if (identical(method, "ivf")) {
      "FAISS IVF-Flat supports Euclidean/L2 and raw inner product; cosine/correlation use row transforms followed by IVF inner-product search."
    } else {
      "FAISS IVFPQ supports Euclidean/L2 and raw inner product; cosine/correlation use row transforms followed by IVFPQ inner-product search."
    }
  } else if (identical(method, "nsg")) {
    supported <- all_metrics
    exact <- FALSE
    implementation <- if (identical(backend, "cpu")) {
      "native CPU NSG candidate graph"
    } else {
      "native CUDA NSG candidate graph"
    }
    notes <- if (identical(backend, "cpu")) {
      "Public CPU NSG uses faissR's native NSG-style candidate graph for all metrics to avoid unsafe linked-FAISS graph construction; large high-dimensional CPU inputs use a deterministic FAISS HNSW seed before NSG/MRNG-style pruning."
    } else if (supported) {
      "CUDA NSG builds an NSG-style candidate graph and refines candidates with the native CUDA row-candidate kernel; cosine/correlation use normalized Euclidean search and raw inner product uses shifted dot-product distances."
    } else {
      "Unsupported CUDA NSG metric."
    }
  } else if (identical(method, "vamana")) {
    supported <- all_metrics
    exact <- FALSE
    implementation <- if (identical(backend, "cpu")) {
      "native Vamana candidate graph"
    } else {
      "native Vamana candidate graph with CUDA refinement"
    }
    notes <- if (identical(backend, "cpu")) {
      "Builds a DiskANN/Vamana-style robust-pruned candidate graph and refines top-k within candidate rows on CPU; large high-dimensional CPU inputs use a deterministic FAISS HNSW seed before robust pruning. Cosine/correlation use normalized Euclidean search and raw inner product uses shifted dot-product distances."
    } else {
      "Builds a Vamana-style candidate graph and refines candidate rows with the native CUDA row-candidate kernel; cuVS Vamana currently builds/serializes DiskANN-compatible indexes but does not expose KNN search."
    }
  } else if (identical(method, "nndescent")) {
    supported <- all_metrics
    exact <- if (supported) FALSE else NA
    implementation <- if (identical(backend, "cpu")) {
      "native CPU NNDescent"
    } else if (identical(metric, "inner_product")) {
      "native CUDA NNDescent candidate refinement"
    } else {
      "cuVS CUDA NN-descent"
    }
    notes <- if (identical(backend, "cpu")) {
      "Native CPU NNDescent supports Euclidean/L2 and raw inner-product self-KNN; cosine/correlation use normalized Euclidean graph search."
    } else if (supported) {
      if (identical(metric, "inner_product")) {
        "Raw inner-product CUDA NNDescent uses faissR's native CUDA candidate-refinement kernel with an Euclidean IVF seed graph and inner-product top-k refinement."
      } else {
        "Validated for Euclidean/L2 self-KNN search through cuVS NN-descent; cosine/correlation use normalized Euclidean graph search."
      }
    } else {
      "CUDA NNDescent does not expose raw inner-product search."
    }
  } else if (identical(method, "cagra")) {
    supported <- identical(backend, "cuda") && non_ip_metric
    exact <- if (supported) FALSE else NA
    implementation <- if (identical(backend, "cuda")) "FAISS GPU CAGRA or cuVS CAGRA" else NA_character_
    notes <- if (!identical(backend, "cuda")) {
      "CAGRA is CUDA-only."
    } else if (identical(metric, "inner_product")) {
      "Raw inner-product CAGRA is disabled for both FAISS GPU CAGRA and direct cuVS CAGRA because transformed CAGRA inner-product search is not reliable across k values in the current implementation."
    } else if (identical(backend, "cuda")) {
      "CUDA-only approximate graph search; faissR.cagra_implementation selects FAISS GPU CAGRA, direct cuVS CAGRA, or a deterministic shape-aware auto provider rule."
    } else {
      "Unsupported CAGRA route."
    }
  }

  if (!isTRUE(supported)) {
    implementation <- NA_character_
    exact <- NA
  }

  data.frame(
    method = method,
    backend = backend,
    metric = metric,
    supported = isTRUE(supported),
    exact = if (is.na(exact)) NA else isTRUE(exact),
    implementation = implementation,
    notes = notes,
    stringsAsFactors = FALSE
  )
}

normalize_nn_tuning <- function(tuning) {
  tuning <- normalize_scalar_choice_arg(
    tuning,
    arg = "tuning",
    default = "auto",
    formal_choices = c("auto", "cache", "pilot", "fixed", "off", "none")
  )
  if (is.na(tuning) || !nzchar(tuning)) tuning <- "auto"
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

resolve_public_nn_backend <- function(backend,
                                      method,
                                      metric = "euclidean",
                                      n = NULL,
                                      p = NULL,
                                      k = NULL,
                                      self_query = NULL) {
  backend_label <- normalize_scalar_choice_arg(
    backend,
    arg = "backend",
    default = "auto",
    formal_choices = c("auto", "cpu", "cuda")
  )
  method_label <- normalize_nn_method(method)
  metric <- normalize_nn_metric(metric)
  if (!tolower(backend_label) %in% c("auto", "cpu", "cuda")) {
    stop("`backend` should be one of \"auto\", \"cpu\", or \"cuda\".", call. = FALSE)
  }
  requested_device <- tolower(backend_label)
  device <- normalize_public_compute_backend(backend)
  method <- method_label
  if (identical(requested_device, "auto") && !identical(method, "auto")) {
    device <- resolve_auto_public_nn_device(method, metric)
  }
  if (identical(method, "auto")) {
    if (identical(requested_device, "auto")) {
      return("auto")
    }
    if (identical(device, "cuda")) {
      return("cuda_auto")
    }
    return("cpu_auto")
  }
  if (!metric %in% c("euclidean", "inner_product") && identical(device, "cuda") &&
      !method %in% c("exact", "bruteforce", "flat", "grid", "hnsw", "ivf", "ivfpq", "vamana", "nsg", "nndescent", "cagra")) {
    if (identical(requested_device, "auto")) {
      device <- "cpu"
    } else {
      stop(
        "CUDA `method = \"", method, "\"` does not support ",
        "`metric = \"", metric, "\"`. Use `method = \"exact\"`, ",
        "`\"bruteforce\"`, `\"flat\"`, `\"grid\"`, `\"ivf\"`, ",
        "`\"hnsw\"`, `\"ivfpq\"`, `\"nsg\"`, `\"nndescent\"`, or `\"cagra\"` for validated ",
        "CUDA cosine/correlation search.",
        call. = FALSE
      )
    }
  }
  if (identical(device, "cpu")) {
    if (identical(method, "grid") && identical(metric, "inner_product")) {
      stop("CPU `method = \"grid\"` does not support `metric = \"inner_product\"`.", call. = FALSE)
    }
    switch(
      method,
      exact = "cpu",
      bruteforce = "cpu",
      flat = switch(
        metric,
        inner_product = "faiss_flat_ip",
        cosine = "faiss_flat_cosine",
        correlation = "faiss_flat_correlation",
        "faiss_flat_l2"
      ),
      grid = "cpu_grid",
      hnsw = if (isTRUE(faiss_available())) "faiss_hnsw" else "hnsw",
      ivf = "faiss_ivf",
      ivfpq = "faiss_ivfpq",
      vamana = "cpu_vamana",
      nsg = "cpu_nsg",
      nndescent = "cpu_nndescent",
      cagra = stop("`method = \"cagra\"` is only available with `backend = \"cuda\"`.", call. = FALSE),
      stop("Unsupported CPU nearest-neighbour method.", call. = FALSE)
    )
  } else {
    if (metric %in% c("cosine", "correlation") &&
        method %in% c("exact", "bruteforce", "flat")) {
      if (identical(method, "bruteforce") && isTRUE(cuvs_available())) {
        return("cuda_cuvs_bruteforce")
      }
      if (identical(method, "exact") && !isTRUE(faiss_gpu_available()) && isTRUE(cuvs_available())) {
        return("cuda_cuvs_bruteforce")
      }
      return(if (identical(metric, "correlation")) "faiss_gpu_flat_correlation" else "faiss_gpu_flat_cosine")
    }
    if (identical(metric, "inner_product") &&
        method %in% c("exact", "bruteforce", "flat")) {
      if (identical(method, "bruteforce") && isTRUE(cuvs_available())) {
        return("cuda_cuvs_bruteforce")
      }
      if (identical(method, "exact") && !isTRUE(faiss_gpu_available()) && isTRUE(cuvs_available())) {
        return("cuda_cuvs_bruteforce")
      }
      return("faiss_gpu_flat_ip")
    }
    if (identical(metric, "inner_product") && identical(method, "cagra")) {
      stop(
        "CUDA `method = \"cagra\"` does not currently support ",
        "`metric = \"inner_product\"` because transformed CAGRA inner-product ",
        "search is not reliable across k values. Use `method = \"flat\"`, ",
        "`\"ivf\"`, `\"ivfpq\"`, `\"nsg\"`, `\"vamana\"`, or ",
        "`\"nndescent\"` for CUDA inner-product search.",
        call. = FALSE
      )
    }
    if (identical(metric, "inner_product") && identical(method, "hnsw")) {
      stop(
        "CUDA `method = \"hnsw\"` does not currently support ",
        "`metric = \"inner_product\"` because the available cuVS HNSW route ",
        "is built from a CAGRA L2 graph. Use CPU HNSW for inner product, or ",
        "CUDA `method = \"nsg\"`, `\"vamana\"`, or `\"nndescent\"` for native ",
        "inner-product candidate refinement.",
        call. = FALSE
      )
    }
    if (identical(metric, "inner_product") && !method %in% c("ivf", "ivfpq", "nsg", "vamana", "nndescent")) {
      stop("CUDA `metric = \"inner_product\"` currently supports `method = \"exact\"`, `\"bruteforce\"`, `\"flat\"`, `\"ivf\"`, `\"ivfpq\"`, `\"nsg\"`, `\"vamana\"`, or `\"nndescent\"`.", call. = FALSE)
    }
    switch(
      method,
      exact = if (isTRUE(faiss_gpu_available())) "faiss_gpu_flat_l2" else if (isTRUE(cuvs_available())) "cuda_cuvs_bruteforce" else "cuda",
      bruteforce = if (isTRUE(cuvs_available())) "cuda_cuvs_bruteforce" else if (isTRUE(faiss_gpu_available())) "faiss_gpu_flat_l2" else "cuda",
      flat = "faiss_gpu_flat_l2",
      grid = "cuda_grid",
      hnsw = "cuda_cuvs_hnsw",
      ivf = "faiss_gpu_ivf_flat",
      ivfpq = "faiss_gpu_ivfpq",
      vamana = "cuda_vamana",
      nsg = "cuda_nsg",
      nndescent = if (!identical(metric, "inner_product")) {
        "cuda_cuvs_nndescent"
      } else {
        "cuda_native_nndescent"
      },
      cagra = resolve_cuda_cagra_backend(
        n = n,
        p = p,
        k = k,
        self_query = self_query
      ),
      stop("Unsupported CUDA nearest-neighbour method.", call. = FALSE)
    )
  }
}

public_nn_method_label <- function(method) {
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
    cagra = "cagra"
  )
  labels[[method]] %||% method
}

nn_resolved_backend_public_method <- function(backend) {
  backend <- as.character(backend)[1L]
  if (is.na(backend) || !nzchar(backend)) return(NA_character_)
  if (backend %in% c("auto", "cpu_auto", "cuda_auto", "gpu_auto")) return("auto")
  if (backend %in% c("cpu", "cuda")) return("exact")
  if (backend %in% c("cuda_cuvs_bruteforce")) return("bruteforce")
  if (backend %in% c(
    "faiss", "cpu_faiss", "cpu_faiss_flat", "faiss_flat", "faiss_flat_l2",
    "faiss_flat_ip", "faiss_flat_cosine", "faiss_flat_correlation",
    "faiss_gpu_flat", "faiss_gpu_flat_l2", "cuda_faiss_flat_l2",
    "faiss_gpu_flat_ip", "cuda_faiss_flat_ip",
    "faiss_gpu_flat_cosine", "cuda_faiss_flat_cosine",
    "faiss_gpu_flat_correlation", "cuda_faiss_flat_correlation"
  )) return("flat")
  if (backend %in% c("grid", "cpu_grid", "grid2d", "grid3d",
                     "cpu_grid2d", "cpu_grid3d", "cuda_grid",
                     "cuda_grid_auto", "gpu_grid", "cuda_grid2d",
                     "cuda_grid3d")) return("grid")
  if (backend %in% c("hnsw", "rcpphnsw", "cpu_hnsw", "faiss_hnsw",
                     "cuda_cuvs_hnsw", "cuvs_hnsw")) return("hnsw")
  if (backend %in% c("faiss_ivf", "cpu_faiss_index_ivf", "faiss_ivf_flat",
                     "faiss_gpu_ivf", "faiss_gpu_ivf_flat",
                     "cuda_faiss_ivf_flat", "cuvs_ivf",
                     "cuda_cuvs_ivf", "cuvs_ivf_flat",
                     "cuda_cuvs_ivf_flat")) return("ivf")
  if (backend %in% c("faiss_ivfpq", "faiss_gpu_ivfpq",
                     "cuda_faiss_ivfpq", "cuvs_ivfpq",
                     "cuda_cuvs_ivfpq", "cuvs_ivf_pq",
                     "cuda_cuvs_ivf_pq")) return("ivfpq")
  if (backend %in% c("cpu_vamana", "cuda_vamana")) return("vamana")
  if (backend %in% c("faiss_nsg", "cpu_nsg", "cuda_nsg")) return("nsg")
  if (backend %in% c("cpu_nndescent", "faiss_nndescent",
                     "cuda_cuvs_nndescent", "cuvs_nndescent",
                     "cuda_native_nndescent",
                     "cuda_nndescent", "cuda_approx",
                     "gpu_nndescent", "gpu_approx")) return("nndescent")
  if (backend %in% c("faiss_gpu_cagra", "cuda_faiss_cagra",
                     "cuda_cuvs_cagra", "cuda_cagra",
                     "gpu_cagra")) return("cagra")
  NA_character_
}

nn_resolved_backend_device <- function(backend) {
  backend <- as.character(backend)[1L]
  if (is.na(backend) || !nzchar(backend)) return(NA_character_)
  if (backend %in% c("auto", "cpu_auto", "cuda_auto", "gpu_auto")) return("auto")
  if (startsWith(backend, "cuda") ||
      startsWith(backend, "gpu") ||
      startsWith(backend, "cuvs") ||
      startsWith(backend, "faiss_gpu")) {
    return("cuda")
  }
  "cpu"
}

resolve_auto_knn_gpu_backend <- function(backend,
                                         self_query,
                                         n_points,
                                         n,
                                         p,
                                         k,
                                         work_size,
                                         metric = "euclidean",
                                         cuda_available_value = cuda_available(),
                                         cuvs_available_value = cuvs_available(),
                                         faiss_gpu_available_value = faiss_gpu_available()) {
  if (!identical(backend, "auto")) return(NA_character_)
  route <- nn_auto_select_shape_cpp(
    resolved_backend = "auto",
    requested_backend = "auto",
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
    ),
    cuda_available_value = cuda_available_value,
    cuvs_available_value = cuvs_available_value,
    faiss_gpu_available_value = faiss_gpu_available_value
  )
  if (identical(route$reason, "auto_cuda_preselector")) route$selected_backend else NA_character_
}

select_cuda_auto_backend <- function(self_query,
                                     n,
                                     p,
                                     n_points,
                                     k,
                                     work_size,
                                     metric = "euclidean") {
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

select_cuvs_auto_backend <- function(self_query,
                                     n,
                                     p,
                                     n_points,
                                     k,
                                     work_size,
                                     cuda_available_value = cuda_available(),
                                     cuvs_available_value = cuvs_available()) {
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
  if (!is.na(route$error)) stop(route$error, call. = FALSE)
  route$selected_backend
}

native_nsg_option_int <- function(name, default, min_value = 1L, max_value = .Machine$integer.max) {
  value <- faissr_option(name, NULL)
  value <- if (is.null(value)) default else suppressWarnings(as.integer(value))
  if (length(value) != 1L || is.na(value) || !is.finite(value)) value <- default
  as.integer(max(min_value, min(max_value, value)))
}

select_cpu_auto_backend <- function(self_query,
                                    n,
                                    p,
                                    n_points,
                                    k,
                                    work_size,
                                    metric = "euclidean") {
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
  value <- suppressWarnings(as.numeric(value))
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
    cagra = FALSE,
    FALSE
  )
}

public_nn_cuda_route_available <- function(method,
                                           metric,
                                           cuda_available_value = cuda_available(),
                                           cuvs_available_value = cuvs_available(),
                                           faiss_gpu_available_value = faiss_gpu_available()) {
  method <- normalize_nn_method(method)
  metric <- normalize_nn_metric(metric)
  if (identical(method, "auto")) {
    return(isTRUE(nn_cuda_auto_runtime_available(
      metric,
      cuda_available_value = cuda_available_value,
      cuvs_available_value = cuvs_available_value,
      faiss_gpu_available_value = faiss_gpu_available_value
    )$available))
  }
  if (identical(method, "nsg")) return(isTRUE(cuda_available_value))
  if (identical(method, "vamana")) return(isTRUE(cuda_available_value))
  if (identical(metric, "inner_product")) {
    if (identical(method, "nndescent")) return(isTRUE(cuda_available_value))
    if (identical(method, "cagra")) return(FALSE)
    if (identical(method, "hnsw")) return(FALSE)
    if (method %in% c("exact", "bruteforce")) {
      return(isTRUE(faiss_gpu_available_value) || isTRUE(cuvs_available_value))
    }
    return(method %in% c("flat", "ivf", "ivfpq", "vamana") &&
      isTRUE(faiss_gpu_available_value))
  }
  if (identical(method, "hnsw")) return(isTRUE(cuvs_available_value))
  if (metric %in% c("cosine", "correlation")) {
    if (method %in% c("exact", "bruteforce")) {
      return(isTRUE(faiss_gpu_available_value) || isTRUE(cuvs_available_value))
    }
    if (method %in% c("flat", "ivf", "ivfpq")) {
      return(isTRUE(faiss_gpu_available_value))
    }
    if (identical(method, "grid")) return(isTRUE(cuda_available_value))
    if (identical(method, "hnsw")) return(isTRUE(cuvs_available_value))
    if (identical(method, "nndescent")) return(isTRUE(cuvs_available_value))
    if (identical(method, "cagra")) {
      return(cuda_cagra_route_available(
        faiss_gpu_available_value = faiss_gpu_available_value,
        cuvs_available_value = cuvs_available_value
      ))
    }
    return(FALSE)
  }
  switch(
    method,
    exact = isTRUE(faiss_gpu_available_value) || isTRUE(cuvs_available_value) || isTRUE(cuda_available_value),
    bruteforce = isTRUE(faiss_gpu_available_value) || isTRUE(cuvs_available_value) || isTRUE(cuda_available_value),
    flat = isTRUE(faiss_gpu_available_value),
    grid = isTRUE(cuda_available_value),
    hnsw = isTRUE(cuvs_available_value),
    ivf = isTRUE(faiss_gpu_available_value),
    ivfpq = isTRUE(faiss_gpu_available_value),
    nndescent = isTRUE(cuvs_available_value),
    cagra = cuda_cagra_route_available(
      faiss_gpu_available_value = faiss_gpu_available_value,
      cuvs_available_value = cuvs_available_value
    ),
    FALSE
  )
}

resolve_auto_public_nn_device <- function(method,
                                          metric,
                                          cuda_available_value = cuda_available(),
                                          cuvs_available_value = cuvs_available(),
                                          faiss_gpu_available_value = faiss_gpu_available()) {
  method <- normalize_nn_method(method)
  metric <- normalize_nn_metric(metric)
  if (identical(method, "cagra")) return("cuda")
  if (identical(method, "nsg") && !public_nn_cpu_route_supported(method, metric)) {
    return("cuda")
  }
  if (public_nn_cuda_route_available(
    method,
    metric,
    cuda_available_value = cuda_available_value,
    cuvs_available_value = cuvs_available_value,
    faiss_gpu_available_value = faiss_gpu_available_value
  )) {
    return("cuda")
  }
  if (public_nn_cpu_route_supported(method, metric)) return("cpu")
  stop(
    "`backend = \"auto\"`, method = \"", method, "\", metric = \"", metric,
    "\" has no supported CPU route and no available CUDA route.",
    call. = FALSE
  )
}

select_cpu_approx_backend <- function(n, p, k) {
  if (should_use_grid2d_self_knn(
    self_query = TRUE,
    n = n,
    p = p,
    k = k,
    exclude_self = FALSE,
    metric = "euclidean"
  )) {
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
  if (isTRUE(requireNamespace("RcppHNSW", quietly = TRUE))) {
    return("hnsw")
  }
  "cpu"
}

nn_auto_shape <- function(data,
                          points,
                          points_missing,
                          k,
                          metric = "euclidean",
                          exclude_self = FALSE) {
  data_dim <- if (is_float32_matrix_input(data)) float32_matrix_dims(data, "data") else dim(data)
  if (is.null(data_dim) || length(data_dim) != 2L) {
    data_dim <- dim(as.matrix(data))
  }
  points_dim <- if (isTRUE(points_missing)) {
    data_dim
  } else if (is_float32_matrix_input(points)) {
    float32_matrix_dims(points, "points")
  } else {
    dim(points)
  }
  if (is.null(points_dim) || length(points_dim) != 2L) {
    points_dim <- dim(as.matrix(points))
  }
  n <- as.integer(data_dim[[1L]])
  p <- as.integer(data_dim[[2L]])
  n_points <- as.integer(points_dim[[1L]])
  self_query <- isTRUE(points_missing) || identical(data, points)
  if (is.null(k)) {
    k <- if (n == 1L) {
      1L
    } else {
      min(n, auto_k(n, include_self = isTRUE(self_query) && !isTRUE(exclude_self)))
    }
  }
  k <- normalize_nn_positive_integer(k, "k", "`k` must be NULL or a positive integer.")
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

nn_auto_select_shape_cpp <- function(resolved_backend,
                                     requested_backend = "auto",
                                     requested_method = "auto",
                                     shape,
                                     tuning = "auto",
                                     cuda_available_value = cuda_available(),
                                     cuvs_available_value = cuvs_available(),
                                     faiss_available_value = faiss_available(),
                                     faiss_gpu_available_value = faiss_gpu_available(),
                                     rcpphnsw_available_value = requireNamespace("RcppHNSW", quietly = TRUE)) {
  numeric_option <- function(name, default) {
    value <- suppressWarnings(as.numeric(faissr_option(name, default)))
    if (length(value) != 1L || is.na(value) || !is.finite(value)) default else value
  }
  out <- nn_auto_select_backend_cpp(
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
    rcpphnsw_available = isTRUE(rcpphnsw_available_value),
    cagra_preference = cagra_implementation_preference(),
    cuda_exact_n = faiss_option_int("cuda_auto_exact_n", 100000L, min_value = 1000L, max_value = 10000000L),
    cuda_exact_work = numeric_option("cuda_auto_exact_work", 5e12),
    metric_graph_n = faiss_option_int("cuda_auto_metric_graph_n", 100000L, min_value = 1000L, max_value = 10000000L),
    metric_graph_min_k = faiss_option_int("cuda_auto_metric_graph_min_k", 10L, min_value = 2L, max_value = 256L),
    metric_graph_work = numeric_option("cuda_auto_metric_graph_work", 5e12),
    cagra_compact_n = faiss_option_int("cuda_cagra_cuvs_compact_n", 10000L, min_value = 100L, max_value = 1000000L),
    cagra_high_dim_p = faiss_option_int("cuda_cagra_cuvs_high_dim_p", 1024L, min_value = 2L, max_value = 100000L),
    cagra_compact_max_k = faiss_option_int("cuda_cagra_cuvs_compact_max_k", 128L, min_value = 1L, max_value = 10000L),
    cuvs_bruteforce_work_threshold = numeric_option("cuvs_bruteforce_work_threshold", 5e12),
    cpu_exact_work = numeric_option("cpu_auto_exact_work", 2e8),
    cpu_faiss_flat_work = cpu_auto_faiss_flat_work_threshold(),
    tuning = normalize_nn_tuning(tuning)
  )
  for (field in c("selected_backend", "predicted_backend", "predicted_method",
                  "predicted_device", "reason", "error")) {
    if (is.null(out[[field]]) || !nzchar(as.character(out[[field]])[1L])) {
      out[[field]] <- NA_character_
    }
  }
  out
}

nn_auto_selection_for_backend <- function(backend,
                                          self_query,
                                          n,
                                          p,
                                          n_points,
                                          k,
                                          work_size,
                                          metric = "euclidean",
                                          exclude_self = FALSE,
                                          tuning = "auto") {
  requested_backend <- switch(
    backend,
    cpu_auto = "cpu",
    cuda_auto = "cuda",
    gpu_auto = "cuda",
    "auto"
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
    tuning = tuning
  )
}

nn_auto_selected_backend <- function(route, fallback_backend) {
  if (is.null(route)) return(fallback_backend)
  error <- route$error %||% NA_character_
  error <- as.character(error)[1L]
  if (!is.na(error) && nzchar(error)) {
    stop(error, call. = FALSE)
  }
  selected <- route$selected_backend %||% route$predicted_backend %||% fallback_backend
  selected <- as.character(selected)[1L]
  if (is.na(selected) || !nzchar(selected)) fallback_backend else selected
}

nn_auto_route_for_shape <- function(shape, resolved_backend) {
  route <- nn_auto_select_shape_cpp(
    resolved_backend = resolved_backend,
    requested_backend = "auto",
    requested_method = "auto",
    shape = shape
  )
  list(selected_backend = route$selected_backend, reason = route$reason, error = route$error)
}

nn_auto_selection_metadata <- function(data,
                                       points,
                                       points_missing,
                                       k,
                                       requested_backend,
                                       requested_method,
                                       resolved_backend,
                                       metric = "euclidean",
                                       tuning = "auto",
                                       exclude_self = FALSE) {
  explicit_backend <- !identical(requested_backend, "auto")
  explicit_method <- !identical(requested_method, "auto")
  if (explicit_backend &&
      explicit_method &&
      !resolved_backend %in% c("auto", "cpu_auto", "cuda_auto", "gpu_auto")) {
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
    tuning = tuning
  )
}

normalize_nn_threads <- function(n_threads) {
  if (is.null(n_threads)) {
    n_threads <- suppressWarnings(parallel::detectCores(logical = FALSE))
    n_threads <- suppressWarnings(as.integer(n_threads))
    if (length(n_threads) != 1L || is.na(n_threads) || !is.finite(n_threads) || n_threads < 1L) {
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
  value <- suppressWarnings(as.numeric(x))
  if (length(value) != 1L || is.na(value) || !is.finite(value) ||
      value < 1L || abs(value - round(value)) > sqrt(.Machine$double.eps)) {
    stop(message, call. = FALSE)
  }
  as.integer(round(value))
}

normalize_nn_metric <- function(metric) {
  aliases <- c(
    euclidean = "euclidean",
    l2 = "euclidean",
    cosine = "cosine",
    cos = "cosine",
    correlation = "correlation",
    cor = "correlation",
    pearson = "correlation",
    inner_product = "inner_product",
    innerproduct = "inner_product",
    ip = "inner_product",
    dot = "inner_product",
    dot_product = "inner_product",
    dotproduct = "inner_product"
  )
  metric <- normalize_scalar_choice_arg(
    metric,
    arg = "metric",
    default = "euclidean",
    formal_choices = nn_metric_labels()
  )
  key <- tolower(trimws(metric))
  key <- gsub("[[:space:]-]+", "_", key)
  if (!key %in% names(aliases)) {
    stop(
      "`metric` must be one of \"euclidean\", \"cosine\", ",
      "\"correlation\", or \"inner_product\".",
      call. = FALSE
    )
  }
  unname(aliases[[key]])
}

faiss_metric_search_arg <- function(metric) {
  metric <- normalize_nn_metric(metric)
  if (identical(metric, "inner_product")) "inner_product" else "euclidean"
}

faiss_metric_distance_output_arg <- function(metric) {
  metric <- normalize_nn_metric(metric)
  if (identical(metric, "inner_product")) "inner_product" else "euclidean"
}

should_use_grid2d_self_knn <- function(self_query,
                                       n,
                                       p,
                                       k,
                                       exclude_self,
                                       metric) {
  if (!isTRUE(self_query)) return(FALSE)
  metric <- normalize_nn_metric(metric)
  if (!metric %in% c("euclidean", "cosine", "correlation")) return(FALSE)
  if (!as.integer(p) %in% c(2L, 3L)) return(FALSE)
  if (as.integer(n) < 10000L) return(FALSE)
  nonself_k <- if (isTRUE(exclude_self)) as.integer(k) else as.integer(k) - 1L
  is.finite(nonself_k) && !is.na(nonself_k) && nonself_k >= 1L
}

grid_bins_per_dim <- function(n, k, p) {
  p <- as.integer(p)
  value <- faissr_option(sprintf("grid%dd_bins_per_dim", p), NULL)
  if (is.null(value)) value <- faissr_option("grid_bins_per_dim", NULL)
  if (!is.null(value)) {
    value <- suppressWarnings(as.integer(value))
    if (length(value) == 1L && is.finite(value) && !is.na(value) && value > 0L) {
      return(as.integer(value))
    }
  }
  target_occupancy <- faissr_option(sprintf("grid%dd_target_occupancy", p), NULL)
  if (is.null(target_occupancy)) target_occupancy <- faissr_option("grid_target_occupancy", NULL)
  if (is.null(target_occupancy)) {
    target_occupancy <- if (identical(p, 3L)) {
      max(1.5, min(8, as.numeric(k) / 25))
    } else {
      max(4, min(16, as.numeric(k) / 10))
    }
  }
  target_occupancy <- suppressWarnings(as.numeric(target_occupancy))
  if (length(target_occupancy) != 1L || !is.finite(target_occupancy) ||
      is.na(target_occupancy) || target_occupancy <= 0) {
    target_occupancy <- if (identical(p, 3L)) {
      max(1.5, min(8, as.numeric(k) / 25))
    } else {
      max(4, min(16, as.numeric(k) / 10))
    }
  }
  bins <- if (identical(p, 3L)) {
    as.integer(ceiling((as.numeric(n) / target_occupancy)^(1 / 3)))
  } else {
    as.integer(ceiling(sqrt(as.numeric(n) / target_occupancy)))
  }
  as.integer(max(4L, min(4096L, bins)))
}

grid2d_bins_per_dim <- function(n, k) grid_bins_per_dim(n, k, 2L)
grid3d_bins_per_dim <- function(n, k) grid_bins_per_dim(n, k, 3L)

select_cpu_spatial_backend <- function(data,
                                       k,
                                       exclude_self = TRUE) {
  p <- ncol(data)
  if (!(p %in% c(2L, 3L))) {
    stop("`method = \"grid\"` supports only two- or three-column matrices.", call. = FALSE)
  }
  if (!isTRUE(faissr_option("cpu_spatial_auto", TRUE))) {
    return(if (p == 3L) "cpu_grid3d" else "cpu_grid2d")
  }
  n <- nrow(data)
  sample_n <- min(n, as.integer(faissr_option("cpu_spatial_sample", 4096L)))
  if (sample_n < 512L) {
    return(if (p == 3L) "cpu_grid3d" else "cpu_grid2d")
  }
  rows <- unique(as.integer(round(seq.int(1L, n, length.out = sample_n))))
  xs <- data[rows, , drop = FALSE]
  sds <- apply(xs, 2L, stats::sd)
  finite_sds <- is.finite(sds) & sds > 0
  if (sum(finite_sds) < p) return(if (p == 3L) "cpu_grid3d" else "cpu_grid2d")
  anisotropy <- min(sds[finite_sds]) / max(sds[finite_sds])
  anisotropy_threshold <- as.numeric(faissr_option("cpu_spatial_anisotropy_threshold", 0.02))
  if (!is.finite(anisotropy_threshold) || anisotropy_threshold <= 0) anisotropy_threshold <- 0.02
  if (is.finite(anisotropy) && anisotropy < anisotropy_threshold) return(if (p == 3L) "cpu_grid3d" else "cpu_grid2d")

  unique_sample <- nrow(unique(round(xs, digits = 12L)))
  duplicate_ratio <- unique_sample / sample_n
  duplicate_threshold <- as.numeric(faissr_option("cpu_spatial_duplicate_threshold", 0.05))
  if (!is.finite(duplicate_threshold) || duplicate_threshold <= 0) duplicate_threshold <- 0.05
  if (is.finite(duplicate_ratio) && duplicate_ratio <= duplicate_threshold) {
    out <- if (p == 3L) "cpu_grid3d" else "cpu_grid2d"
    attr(out, "reason") <- sprintf("duplicate_heavy_sample_unique_ratio_%.4g", duplicate_ratio)
    return(out)
  }

  sample_bins <- as.integer(faissr_option("cpu_spatial_sample_bins", if (p == 3L) 16L else 32L))
  if (!is.finite(sample_bins) || is.na(sample_bins) || sample_bins < 4L) sample_bins <- if (p == 3L) 16L else 32L
  mins <- apply(xs, 2L, min)
  spans <- apply(xs, 2L, function(z) max(z) - min(z))
  spans[!is.finite(spans) | spans <= 0] <- 1
  coords <- lapply(seq_len(p), function(j) {
    coord <- floor((xs[, j] - mins[[j]]) / spans[[j]] * sample_bins)
    coord <- pmin(sample_bins - 1L, pmax(0L, as.integer(coord)))
    coord
  })
  cell <- coords[[1L]] + sample_bins * coords[[2L]]
  if (p == 3L) cell <- cell + sample_bins * sample_bins * coords[[3L]]
  counts <- tabulate(cell + 1L, nbins = sample_bins^p)
  mean_count <- sample_n / length(counts)
  imbalance <- max(counts) / max(mean_count, 1e-12)
  imbalance_threshold <- as.numeric(faissr_option("cpu_spatial_imbalance_threshold", 20))
  if (!is.finite(imbalance_threshold) || imbalance_threshold <= 0) imbalance_threshold <- 20
  if (is.finite(imbalance) && imbalance > imbalance_threshold) return(if (p == 3L) "cpu_grid3d" else "cpu_grid2d")
  if (p == 3L) "cpu_grid3d" else "cpu_grid2d"
}

row_center_l2_normalize <- function(x) {
  x <- as.matrix(x)
  storage.mode(x) <- "double"
  means <- rowMeans(x)
  x <- x - means
  norms <- sqrt(rowSums(x * x))
  keep <- is.finite(norms) & norms > 0
  if (any(keep)) {
    x[keep, ] <- x[keep, , drop = FALSE] / norms[keep]
  }
  x
}

row_l2_normalize <- function(x) {
  x <- as.matrix(x)
  storage.mode(x) <- "double"
  norms <- sqrt(rowSums(x * x))
  keep <- is.finite(norms) & norms > 0
  if (any(keep)) {
    x[keep, ] <- x[keep, , drop = FALSE] / norms[keep]
  }
  x
}

normalized_euclidean_metric_inputs <- function(data, points, self_query, metric) {
  metric <- normalize_nn_metric(metric)
  if (!metric %in% c("cosine", "correlation")) {
    stop("Normalized Euclidean metric transforms require cosine or correlation.", call. = FALSE)
  }
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
    data_zero = rowSums(data_metric * data_metric) <= 0,
    points_zero = if (isTRUE(self_query)) {
      rowSums(data_metric * data_metric) <= 0
    } else {
      rowSums(points_metric * points_metric) <= 0
    },
    transform = if (identical(metric, "correlation")) {
      "row_center_l2_normalize_then_euclidean_graph_search"
    } else {
      "row_l2_normalize_then_euclidean_graph_search"
    }
  )
}

row_inner_product_norm2 <- function(x) {
  x <- as.matrix(x)
  storage.mode(x) <- "double"
  rowSums(x * x)
}

mips_l2_metric_inputs <- function(data, points, self_query) {
  data <- as.matrix(data)
  storage.mode(data) <- "double"
  points <- as.matrix(points)
  storage.mode(points) <- "double"
  data_norm2 <- row_inner_product_norm2(data)
  points_norm2 <- if (isTRUE(self_query)) data_norm2 else row_inner_product_norm2(points)
  radius2 <- suppressWarnings(max(data_norm2, 0, na.rm = TRUE))
  if (!is.finite(radius2) || radius2 < 0) radius2 <- 0
  extra <- sqrt(pmax(0, radius2 - data_norm2))
  data_metric <- cbind(data, extra)
  points_metric <- cbind(points, rep(0, nrow(points)))
  list(
    data = data_metric,
    points = points_metric,
    radius2 = radius2,
    points_norm2 = points_norm2,
    transform = "maximum_inner_product_to_l2_extra_dimension",
    distance_transform = "mips_l2_to_shifted_inner_product_distance"
  )
}

finalize_graph_metric_result <- function(result, inputs) {
  if (identical(inputs$distance_transform, "mips_l2_to_shifted_inner_product_distance")) {
    return(finalize_mips_l2_metric_result(result, inputs))
  }
  finalize_normalized_euclidean_metric_result(result, inputs)
}

finalize_normalized_euclidean_metric_result <- function(result, inputs) {
  result <- normalized_euclidean_to_similarity_distance(
    result,
    data_zero = inputs$data_zero,
    points_zero = inputs$points_zero
  )
  result$metric_transform <- inputs$transform
  attr(result, "metric_transform") <- inputs$transform
  attr(result, "distance_transform") <- "normalized_euclidean_squared_over_2_to_1_minus_similarity"
  approximation <- attr(result, "approximation")
  if (!is.null(approximation)) {
    approximation$metric_transform <- inputs$transform
    approximation$distance_transform <- "normalized_euclidean_squared_over_2_to_1_minus_similarity"
    attr(result, "approximation") <- approximation
  }
  sort_knn_rows_by_distance_index(result)
}

finalize_mips_l2_metric_result <- function(result, inputs) {
  if (ncol(result$indices) > 0L) {
    q_norm2 <- rep(as.numeric(inputs$points_norm2), length.out = nrow(result$distances))
    l2_squared <- result$distances * result$distances
    scores <- (as.numeric(inputs$radius2) + q_norm2 - l2_squared) / 2
    row_best <- apply(scores, 1L, function(x) {
      finite <- x[is.finite(x)]
      if (!length(finite)) NA_real_ else max(finite)
    })
    result$distances <- row_best - scores
    result$distances[!is.finite(result$distances)] <- Inf
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
  sort_knn_rows_by_distance_index(result)
}

faiss_flat_normalized_metric_result <- function(data,
                                                points,
                                                k,
                                                self_query,
                                                exclude_self,
                                                metric,
                                                backend,
                                                accelerator = NULL,
                                                n_threads = NULL) {
  metric <- normalize_nn_metric(metric)
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
  data_zero <- rowSums(data_metric * data_metric) <= 0
  points_zero <- if (isTRUE(self_query)) data_zero else rowSums(points_metric * points_metric) <= 0
  if (is.null(accelerator) && (any(data_zero) || any(points_zero))) {
    out <- nn_cpp(
      data,
      points,
      as.integer(k),
      metric,
      FALSE,
      TRUE,
      0,
      TRUE,
      as.integer(normalize_nn_threads(n_threads)),
      isTRUE(exclude_self)
    )
    result <- finish_nn_result(out, backend, k, self_query, exact = TRUE, metric = metric)
    faiss_meta <- list(
      index_type = "IndexFlatIP",
      library = "faiss",
      backend = if (identical(accelerator, "cuda")) "cuda" else "cpu",
      metric = metric,
      zero_row_exact_fallback = TRUE,
      transform = if (identical(metric, "correlation")) {
        "row_center_l2_normalize_then_IndexFlatIP"
      } else {
        "row_l2_normalize_then_IndexFlatIP"
      }
    )
    if (!is.null(accelerator)) faiss_meta$accelerator <- accelerator
    attr(result, "faiss") <- faiss_meta
    return(result)
  }
  out <- if (identical(accelerator, "cuda")) {
    nn_faiss_gpu_flat_normalized_ip_distance_cpp(
      data_metric,
      points_metric,
      as.integer(k),
      isTRUE(exclude_self)
    )
  } else {
    nn_faiss_flat_normalized_ip_distance_cpp(
      data_metric,
      points_metric,
      as.integer(k),
      isTRUE(exclude_self),
      as.integer(normalize_nn_threads(n_threads))
    )
  }
  result <- finish_nn_result(
    out,
    backend,
    k,
    self_query,
    exact = TRUE,
    metric = metric
  )
  result <- restore_zero_normalized_ip_distances(
    result,
    data_zero = data_zero,
    points_zero = points_zero,
    exclude_self = isTRUE(exclude_self)
  )
  result <- sort_knn_rows_by_distance_index(result)
  faiss_meta <- list(
    index_type = as.character(out$index_type),
    library = "faiss",
    backend = if (identical(accelerator, "cuda")) "cuda" else "cpu",
    metric = metric,
    transform = if (identical(metric, "correlation")) {
      "row_center_l2_normalize_then_IndexFlatIP"
    } else {
      "row_l2_normalize_then_IndexFlatIP"
    }
  )
  if (!is.null(accelerator)) faiss_meta$accelerator <- accelerator
  attr(result, "faiss") <- faiss_meta
  result
}

faiss_ivf_normalized_metric_result <- function(data,
                                               points,
                                               k,
                                               self_query,
                                               exclude_self,
                                               metric,
                                               backend,
                                               accelerator = NULL,
                                               n_threads = NULL,
                                               params,
                                               tuning_metadata = NULL) {
  metric <- normalize_nn_metric(metric)
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
  out <- if (identical(accelerator, "cuda")) {
    nn_faiss_gpu_ivf_flat_cpp(
      data_metric,
      points_metric,
      as.integer(k),
      as.integer(params$nlist),
      as.integer(params$nprobe),
      "inner_product",
      "one_minus_inner_product",
      isTRUE(exclude_self)
    )
  } else {
    nn_faiss_ivf_cpp(
      data_metric,
      points_metric,
      as.integer(k),
      as.integer(params$nlist),
      as.integer(params$nprobe),
      "inner_product",
      "one_minus_inner_product",
      isTRUE(exclude_self),
      as.integer(normalize_nn_threads(n_threads))
    )
  }
  result <- finish_nn_result(out, backend, k, self_query, exact = FALSE, metric = metric)
  data_zero <- rowSums(data_metric * data_metric) <= 0
  points_zero <- if (isTRUE(self_query)) data_zero else rowSums(points_metric * points_metric) <= 0
  result <- restore_zero_normalized_ip_distances(
    result,
    data_zero = data_zero,
    points_zero = points_zero,
    exclude_self = isTRUE(exclude_self)
  )
  result <- sort_knn_rows_by_distance_index(result)
  attr(result, "approximation") <- list(
    strategy = if (identical(accelerator, "cuda")) {
      "faiss_gpu_IndexIVFFlat_cuVS"
    } else {
      "faiss_IndexIVFFlat"
    },
    backend = backend,
    library = "faiss",
    accelerator = accelerator,
    metric = metric,
    transform = if (identical(metric, "correlation")) {
      "row_center_l2_normalize_then_IndexIVFFlat_METRIC_INNER_PRODUCT"
    } else {
      "row_l2_normalize_then_IndexIVFFlat_METRIC_INNER_PRODUCT"
    },
    nlist = as.integer(out$nlist),
    nprobe = as.integer(out$nprobe),
    requested_nlist = as.integer(params$requested_nlist),
    requested_nprobe = as.integer(params$requested_nprobe),
    ivf_parameters_adjusted = !identical(as.integer(params$requested_nlist), as.integer(out$nlist)) ||
      !identical(as.integer(params$requested_nprobe), as.integer(out$nprobe)),
    tuning = tuning_metadata
  )
  if (is.null(accelerator)) attr(result, "approximation")$accelerator <- NULL
  result <- append_nn_tuning_metadata(result, params)
  result
}

faiss_ivfpq_normalized_metric_result <- function(data,
                                                 points,
                                                 k,
                                                 self_query,
                                                 exclude_self,
                                                 metric,
                                                 backend,
                                                 accelerator = NULL,
                                                 n_threads = NULL,
                                                 params,
                                                 pq) {
  metric <- normalize_nn_metric(metric)
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
  out <- if (identical(accelerator, "cuda")) {
    nn_faiss_gpu_ivfpq_cpp(
      data_metric,
      points_metric,
      as.integer(k),
      as.integer(params$nlist),
      as.integer(params$nprobe),
      as.integer(pq$m),
      as.integer(pq$nbits),
      "inner_product",
      "one_minus_inner_product",
      isTRUE(exclude_self)
    )
  } else {
    nn_faiss_ivfpq_cpp(
      data_metric,
      points_metric,
      as.integer(k),
      as.integer(params$nlist),
      as.integer(params$nprobe),
      as.integer(pq$m),
      as.integer(pq$nbits),
      "inner_product",
      "one_minus_inner_product",
      isTRUE(exclude_self),
      as.integer(normalize_nn_threads(n_threads))
    )
  }
  result <- finish_nn_result(out, backend, k, self_query, exact = FALSE, metric = metric)
  data_zero <- rowSums(data_metric * data_metric) <= 0
  points_zero <- if (isTRUE(self_query)) data_zero else rowSums(points_metric * points_metric) <= 0
  result <- restore_zero_normalized_ip_distances(
    result,
    data_zero = data_zero,
    points_zero = points_zero,
    exclude_self = isTRUE(exclude_self)
  )
  result <- sort_knn_rows_by_distance_index(result)
  attr(result, "approximation") <- list(
    strategy = if (identical(accelerator, "cuda")) {
      "faiss_gpu_IndexIVFPQ_cuVS"
    } else {
      "faiss_IndexIVFPQ"
    },
    backend = backend,
    library = "faiss",
    accelerator = accelerator,
    metric = metric,
    transform = if (identical(metric, "correlation")) {
      "row_center_l2_normalize_then_IndexIVFPQ_METRIC_INNER_PRODUCT"
    } else {
      "row_l2_normalize_then_IndexIVFPQ_METRIC_INNER_PRODUCT"
    },
    role = if (identical(accelerator, "cuda")) "explicit_memory_pressure_backend" else NULL,
    default_candidate = if (identical(accelerator, "cuda")) FALSE else NULL,
    nlist = as.integer(out$nlist),
    nprobe = as.integer(out$nprobe),
    requested_nlist = as.integer(params$requested_nlist),
    requested_nprobe = as.integer(params$requested_nprobe),
    ivf_parameters_adjusted = !identical(as.integer(params$requested_nlist), as.integer(out$nlist)) ||
      !identical(as.integer(params$requested_nprobe), as.integer(out$nprobe)),
    pq_m = as.integer(out$pq_m),
    pq_nbits = as.integer(out$pq_nbits),
    requested_pq_m = as.integer(out$requested_pq_m),
    requested_pq_nbits = as.integer(out$requested_pq_nbits),
    pq_parameters_adjusted = isTRUE(out$pq_parameters_adjusted)
  )
  if (is.null(accelerator)) attr(result, "approximation")$accelerator <- NULL
  if (is.null(attr(result, "approximation")$role)) attr(result, "approximation")$role <- NULL
  if (is.null(attr(result, "approximation")$default_candidate)) attr(result, "approximation")$default_candidate <- NULL
  result <- append_nn_tuning_metadata(result, params, pq, .prefixes = list(NULL, "pq_"))
  result
}

faiss_hnsw_normalized_metric_result <- function(data,
                                                points,
                                                k,
                                                self_query,
                                                exclude_self,
                                                metric,
                                                n_threads = NULL) {
  metric <- normalize_nn_metric(metric)
  params <- faiss_hnsw_params(
    k,
    n = nrow(data),
    p = ncol(data),
    metric = metric
  )
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
  out <- nn_faiss_hnsw_cpp(
    data_metric,
    points_metric,
    as.integer(k),
    as.integer(params$m),
    as.integer(params$ef_construction),
    as.integer(params$ef_search),
    "inner_product",
    "one_minus_inner_product",
    isTRUE(exclude_self),
    as.integer(normalize_nn_threads(n_threads))
  )
  result <- finish_nn_result(out, "faiss_hnsw", k, self_query, exact = FALSE, metric = metric)
  data_zero <- rowSums(data_metric * data_metric) <= 0
  points_zero <- if (isTRUE(self_query)) data_zero else rowSums(points_metric * points_metric) <= 0
  result <- restore_zero_normalized_ip_distances(
    result,
    data_zero = data_zero,
    points_zero = points_zero,
    exclude_self = isTRUE(exclude_self)
  )
  result <- sort_knn_rows_by_distance_index(result)
  attr(result, "approximation") <- list(
    strategy = "faiss_IndexHNSWFlat",
    backend = "faiss_hnsw",
    library = "faiss",
    metric = metric,
    transform = if (identical(metric, "correlation")) {
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
    tuning_policy = params$policy,
    tuning_rule = params$rule,
    tuning_high_dim = isTRUE(params$high_dim),
    tuning_large_n = isTRUE(params$large_n),
    tuning_small_k = isTRUE(params$small_k),
    tuning_large_k = isTRUE(params$large_k),
    tuning_non_euclidean = isTRUE(params$non_euclidean),
    tuning_source = params$tuning_source %||% "cpp"
  )
  result
}

restore_zero_normalized_ip_distances <- function(result,
                                                 data_zero,
                                                 points_zero,
                                                 exclude_self = FALSE) {
  if (!any(points_zero) || !any(data_zero) || ncol(result$indices) < 1L) {
    return(result)
  }
  k <- ncol(result$indices)
  n_data <- length(data_zero)
  data_indices <- seq_len(n_data)
  self_query <- isTRUE(attr(result, "self_query"))
  for (i in which(points_zero)) {
    zero_candidates <- data_indices[data_zero]
    if (isTRUE(self_query) && isTRUE(exclude_self) && i <= n_data) {
      zero_candidates <- zero_candidates[zero_candidates != i]
    }
    nonzero_candidates <- data_indices[!data_zero]
    ordered <- c(zero_candidates, nonzero_candidates)
    if (length(ordered) > 0L) {
      keep <- ordered[seq_len(min(length(ordered), k))]
      result$indices[i, seq_along(keep)] <- keep
      zero_count <- min(length(zero_candidates), length(keep))
      result$distances[i, seq_along(keep)] <- c(
        rep(0, zero_count),
        rep(1, length(keep) - zero_count)
      )
      if (length(keep) < k) {
        fill <- seq.int(length(keep) + 1L, k)
        result$indices[i, fill] <- NA_integer_
        result$distances[i, fill] <- Inf
      }
    }
  }
  result
}

sort_knn_rows_by_distance_index <- function(result) {
  if (ncol(result$indices) < 2L) return(result)
  for (i in seq_len(nrow(result$indices))) {
    ord <- order(result$distances[i, ], result$indices[i, ])
    result$indices[i, ] <- result$indices[i, ord]
    result$distances[i, ] <- result$distances[i, ord]
  }
  result
}

nn_tuning_metadata <- function(params, prefix = NULL) {
  if (!is.list(params)) return(list())
  fields <- c(
    "tuning_policy", "tuning_rule", "tuning_high_dim", "tuning_large_n",
    "tuning_small_k", "tuning_large_k", "tuning_non_euclidean",
    "tuning_metric", "tuning_metric_aware", "tuning_source"
  )
  fields <- fields[fields %in% names(params)]
  out <- params[fields]
  if (!is.null(prefix) && length(out)) {
    names(out) <- paste0(prefix, names(out))
  }
  out
}

append_nn_tuning_metadata <- function(result, ..., .prefixes = NULL) {
  params <- list(...)
  if (!length(params)) return(result)
  approx <- attr(result, "approximation") %||% list()
  if (is.null(.prefixes)) .prefixes <- rep(list(NULL), length(params))
  for (i in seq_along(params)) {
    approx <- c(approx, nn_tuning_metadata(params[[i]], prefix = .prefixes[[i]]))
  }
  attr(result, "approximation") <- approx
  result
}

normalized_euclidean_to_similarity_distance <- function(result, data_zero, points_zero) {
  if (ncol(result$indices) < 1L) return(result)
  for (i in seq_len(nrow(result$indices))) {
    neighbors <- result$indices[i, ]
    both_zero <- isTRUE(points_zero[[i]]) & data_zero[neighbors]
    one_zero <- xor(isTRUE(points_zero[[i]]), data_zero[neighbors])
    values <- (result$distances[i, ] * result$distances[i, ]) / 2
    values[values < 0 & values > -1e-8] <- 0
    values[values > 2 & values < 2 + 1e-8] <- 2
    if (any(both_zero)) values[both_zero] <- 0
    if (any(one_zero)) values[one_zero] <- 1
    result$distances[i, ] <- values
  }
  result
}

should_use_clustered_self_knn <- function(backend,
                                          self_query,
                                          n,
                                          p,
                                          k,
                                          work_size) {
  FALSE
}

resolve_gpu_ivf_backend <- function(backend,
                                    self_query,
                                    n,
                                    p,
                                    k,
                                    exclude_self) {
  out <- list(backend = NA_character_, label = NA_character_, strategy = NA_character_)
  explicit <- backend %in% c("cuda_ivf", "cuda_faiss")
  if (!explicit) return(out)
  if (!isTRUE(self_query)) {
    stop("GPU IVF/FAISS-style KNN currently supports self-KNN only.", call. = FALSE)
  }
  if (isTRUE(exclude_self) && n < 2L) return(out)
  nonself_k <- if (isTRUE(exclude_self)) k else k - 1L
  if (nonself_k < 1L) return(out)
  if (k > 256L) {
    stop("Native GPU IVF/FAISS-style KNN currently supports `k <= 256`.", call. = FALSE)
  }

  if (!isTRUE(cuda_available())) {
    stop("No CUDA GPU backend is available on this machine.", call. = FALSE)
  }
  selected <- "cuda"
  out$backend <- selected
  out$label <- switch(
    backend,
    cuda_faiss = "cuda_faiss_ivf",
    backend
  )
  out$strategy <- if (grepl("faiss", backend, fixed = TRUE)) {
    "faiss_style_ivf_flat_native"
  } else {
    "ivf_flat_native"
  }
  out
}

should_use_gpu_approx_self_knn <- function(backend,
                                           self_query,
                                           n,
                                           p,
                                           k,
                                           exclude_self,
                                           work_size) {
  if (!backend %in% c("cuda_approx")) return(FALSE)
  if (!isTRUE(self_query)) return(FALSE)
  if (p < 2L || k < 10L || k > 256L) return(FALSE)
  nonself_k <- if (isTRUE(exclude_self)) k else k - 1L
  if (nonself_k < 1L) return(FALSE)
  TRUE
}

fast_knn_approx_seed <- function() {
  value <- faissr_option("approx_knn_seed", 4L)
  value <- suppressWarnings(as.integer(value))
  if (length(value) != 1L || is.na(value) || !is.finite(value)) 4L else value
}

gpu_approx_params <- function(n,
                              k,
                              backend = NULL,
                              label = NULL) {
  n <- as.integer(n)
  k <- as.integer(k)
  anchors <- faissr_option("gpu_approx_anchors", NULL)
  if (is.null(anchors)) {
    anchors <- max(128L, ceiling(4 * sqrt(n)), ceiling(n / 500))
    anchors <- min(n - 1L, 4096L, as.integer(anchors))
  } else {
    anchors <- suppressWarnings(as.integer(anchors))
    if (length(anchors) != 1L || is.na(anchors) || !is.finite(anchors)) {
      anchors <- max(128L, ceiling(4 * sqrt(n)))
    }
    anchors <- max(2L, min(n - 1L, anchors))
  }
  projection_k <- faissr_option("gpu_approx_projection_k", NULL)
  if (is.null(projection_k)) {
    projection_k <- max(12L, ceiling(k / 2))
    projection_k <- min(anchors, as.integer(projection_k))
  } else {
    projection_k <- suppressWarnings(as.integer(projection_k))
    if (length(projection_k) != 1L || is.na(projection_k) || !is.finite(projection_k)) {
      projection_k <- max(12L, ceiling(k / 2))
    }
    projection_k <- max(1L, min(anchors, projection_k))
  }
  cols <- clustered_knn_graph_columns(projection_k, k)
  list(
    anchors = as.integer(anchors),
    projection_k = as.integer(projection_k),
    bucket_cols = as.integer(cols$bucket_cols),
    query_cols = as.integer(cols$query_cols)
  )
}

gpu_approx_self_knn <- function(data,
                                k,
                                backend,
                                exclude_self = FALSE,
                                seed = 4L,
                                label = NULL,
                                strategy = "anchor_projection_candidate_knn",
                                metric = "euclidean") {
  metric <- normalize_nn_metric(metric)
  if (!identical(metric, "euclidean")) {
    stop("Native GPU approximate KNN currently supports only `metric = \"euclidean\"`.", call. = FALSE)
  }
  n <- nrow(data)
  label <- if (is.null(label)) paste0(backend, "_approx") else as.character(label)
  nonself_k <- if (isTRUE(exclude_self)) k else k - 1L
  if (nonself_k < 1L) {
    out <- list(
      indices = matrix(seq_len(n), n, 1L),
      distances = matrix(0, n, 1L)
    )
    return(finish_nn_result(out, label, k, TRUE, exact = FALSE, metric = metric))
  }

  params <- gpu_approx_params(n, nonself_k, backend = backend, label = label)
  anchors <- select_landmark_rows(data, params$anchors, seed)
  projection <- nn_compute(
    data[anchors, , drop = FALSE],
    data,
    k = params$projection_k,
    backend = backend,
    points_missing = FALSE,
    exclude_self = FALSE,
    metric = metric
  )
  out <- if (identical(backend, "cuda")) {
    landmark_candidate_knn_cuda_cpp(
      data,
      projection$indices,
      as.integer(nonself_k),
      as.integer(params$bucket_cols),
      as.integer(params$query_cols)
    )
  } else {
    stop("Unsupported CUDA KNN backend.", call. = FALSE)
  }

  if (!isTRUE(exclude_self)) {
    out$indices <- cbind(seq_len(n), out$indices)
    out$distances <- cbind(rep(0, n), out$distances)
  }
  result <- finish_nn_result(out, label, k, TRUE, exact = FALSE, metric = metric)
  attr(result, "approximation") <- list(
    strategy = strategy,
    backend = backend,
    metric = metric,
    anchors = as.integer(params$anchors),
    projection_k = as.integer(params$projection_k),
    bucket_cols = as.integer(params$bucket_cols),
    query_cols = as.integer(params$query_cols),
    seed = as.integer(seed)
  )
  if (isTRUE(faissr_option("gpu_approx_recall", FALSE))) {
    result <- attach_knn_recall_subset(
      result,
      data = data,
      k = k,
      exclude_self = isTRUE(exclude_self),
      seed = seed
    )
  }
  result
}

gpu_ivf_self_knn <- function(data,
                             k,
                             backend,
                             label,
                             strategy,
                             exclude_self = FALSE,
                             seed = 4L,
                             metric = "euclidean") {
  gpu_approx_self_knn(
    data,
    k = k,
    backend = backend,
    exclude_self = exclude_self,
    seed = seed,
    label = label,
    strategy = strategy,
    metric = metric
  )
}

gpu_nndescent_option <- function(backend, name, default = NULL) {
  backend <- as.character(backend)
  keys <- c(
    sprintf("%s_nndescent_%s", backend, name),
    sprintf("gpu_nndescent_%s", name)
  )
  faissr_option(keys, default)
}

gpu_nndescent_graph_degree <- function(n, k, backend = "cuda") {
  graph_degree <- gpu_nndescent_option(backend, "graph_degree", NULL)
  if (is.null(graph_degree)) {
    graph_degree <- if (!is.null(n) && n >= 50000L && k <= 30L) {
      max(k, 64L)
    } else {
      k
    }
  }
  graph_degree <- suppressWarnings(as.integer(graph_degree))
  if (length(graph_degree) != 1L || is.na(graph_degree) || !is.finite(graph_degree)) {
    graph_degree <- k
  }
  if (!is.null(n)) graph_degree <- min(graph_degree, as.integer(n) - 1L)
  as.integer(max(k, min(256L, graph_degree)))
}

gpu_nndescent_params <- function(k, backend = "cuda", n = NULL) {
  backend <- match.arg(as.character(backend), c("cuda"))
  nn_tune_gpu_nndescent_cpp(
    suppressWarnings(as.integer(n %||% NA_integer_)),
    as.integer(k),
    backend,
    nn_option_int_or_na(c(
      sprintf("%s_nndescent_graph_degree", backend),
      "gpu_nndescent_graph_degree"
    )),
    nn_option_int_or_na(c(
      sprintf("%s_nndescent_iters", backend),
      "gpu_nndescent_iters"
    )),
    nn_option_int_or_na(c(
      sprintf("%s_nndescent_sources", backend),
      "gpu_nndescent_sources"
    )),
    nn_option_int_or_na(c(
      sprintf("%s_nndescent_neighbors", backend),
      "gpu_nndescent_neighbors"
    )),
    {
      value <- faissr_option(c(
        sprintf("%s_nndescent_delta", backend),
        "gpu_nndescent_delta"
      ), NULL)
      if (is.null(value)) 0.015 else nn_option_double_or_na(c(
        sprintf("%s_nndescent_delta", backend),
        "gpu_nndescent_delta"
      ))
    }
  )
}

gpu_nndescent_self_knn <- function(data,
                                   k,
                                   backend,
                                   seed = 4L,
                                   metric = "euclidean") {
  n <- nrow(data)
  k <- as.integer(k)
  backend <- match.arg(as.character(backend), c("cuda"))
  metric <- normalize_nn_metric(metric)
  if (!metric %in% c("euclidean", "inner_product")) {
    stop("Native GPU NN-descent supports Euclidean or inner-product candidate refinement.", call. = FALSE)
  }
  if (length(k) != 1L || is.na(k) || !is.finite(k) || k < 1L || k >= n) {
    stop("`k` must be in [1, nrow(data) - 1].", call. = FALSE)
  }
  if (k > 256L) {
    stop("Native GPU NN-descent currently supports `k <= 256`.", call. = FALSE)
  }
  if (identical(backend, "cuda") && !isTRUE(cuda_available())) {
    stop("No CUDA GPU backend is available on this machine.", call. = FALSE)
  }
  output_k <- k
  params <- gpu_nndescent_params(output_k, backend = backend, n = n)
  work_k <- params$graph_degree

  base <- gpu_ivf_self_knn(
    data,
    k = work_k,
    backend = backend,
    label = paste0(backend, "_ivf_seed"),
    strategy = "ivf_flat_native",
    exclude_self = TRUE,
    seed = seed,
    metric = "euclidean"
  )
  indices <- base$indices
  distances <- base$distances
  changes <- numeric(params$n_iters)
  used_iters <- 0L
  candidate_stats <- list()
  flags <- matrix(TRUE, nrow = nrow(indices), ncol = ncol(indices))
  update_frac <- 1
  candidate_attr <- function(x, name, default) {
    value <- attr(x, name, exact = TRUE)
    if (is.null(value)) default else value
  }

  for (iter in seq_len(params$n_iters)) {
    if (identical(backend, "cuda")) {
      # Native adaptive NN-descent schedule: expand aggressively while the graph
      # is moving, then use only NEW-neighbour sources and skip reverse
      # candidates near convergence.
      adaptive_scale <- min(1, sqrt(max(update_frac, 0)) * 3)
      min_sources <- min(work_k, 3L)
      iter_sources <- min(
        work_k,
        max(min_sources, as.integer(ceiling(params$sources * adaptive_scale)))
      )
      min_neighbors <- min(work_k, max(1L, as.integer(ceiling(work_k / 2))))
      iter_neighbors <- min(
        work_k,
        max(min_neighbors, as.integer(ceiling(params$neighbors * adaptive_scale)))
      )
      active_only <- isTRUE(update_frac < 0.5)
      use_reverse <- isTRUE(update_frac >= 0.10)
      query_rows <- NULL
      candidate_indices <- nndescent_candidate_matrix_adaptive_cpp(
        indices,
        flags,
        as.integer(iter_sources),
        as.integer(iter_neighbors),
        use_reverse = use_reverse,
        active_only = active_only
      )
    } else {
      query_rows <- NULL
      candidate_indices <- nndescent_candidate_matrix_cpp(
        indices,
        as.integer(params$sources),
        as.integer(params$neighbors)
      )
    }
    candidate_stats[[iter]] <- list(
      columns = as.integer(ncol(candidate_indices)),
      mean_unique = as.numeric(attr(candidate_indices, "mean_unique_candidates")),
      max_unique = as.integer(attr(candidate_indices, "max_unique_candidates")),
      raw_columns = as.integer(attr(candidate_indices, "raw_candidate_columns")),
      sources = as.integer(candidate_attr(candidate_indices, "sources", params$sources)),
      neighbors = as.integer(candidate_attr(candidate_indices, "neighbors", params$neighbors)),
      active_rows = as.integer(candidate_attr(candidate_indices, "active_rows", n)),
      use_reverse = isTRUE(candidate_attr(candidate_indices, "use_reverse", TRUE)),
      active_only = isTRUE(candidate_attr(candidate_indices, "active_only", FALSE))
    )

    refined <- row_candidate_knn_cuda_cpp(
      data,
      candidate_indices,
      as.integer(work_k),
      metric
    )
    new_indices <- refined$indices
    new_flags <- new_indices != indices
    changes[iter] <- mean(new_flags)
    flags <- new_flags
    update_frac <- changes[iter]
    indices <- new_indices
    distances <- refined$distances
    used_iters <- iter
    if (changes[iter] < params$delta) break
  }

  if (work_k > output_k) {
    keep <- seq_len(output_k)
    indices <- indices[, keep, drop = FALSE]
    distances <- distances[, keep, drop = FALSE]
  }
  if (identical(metric, "inner_product") && ncol(distances) > 0L) {
    distances <- distances - distances[, 1L]
  }

  out <- list(indices = indices, distances = distances)
  attr(out, "cuda_kernel") <- "row_candidate_knn"
  attr(out, "approximation") <- list(
    strategy = paste0("adaptive_seeded_nndescent_native_", backend),
    backend = backend,
    seed_backend = paste0(backend, "_ivf"),
    metric = metric,
    seed_metric = "euclidean",
    refine_metric = metric,
    metric_strategy = if (identical(metric, "inner_product")) {
      "euclidean_seed_inner_product_refine"
    } else {
      "euclidean_seed_euclidean_refine"
    },
    seed_bucket_cols = as.integer(attr(base, "approximation")$bucket_cols),
    seed_query_cols = as.integer(attr(base, "approximation")$query_cols),
    output_graph_degree = as.integer(output_k),
    graph_degree = as.integer(work_k),
    n_iters = as.integer(used_iters),
    changes = changes[seq_len(used_iters)],
    sources = as.integer(params$sources),
    neighbors = as.integer(params$neighbors),
    candidate_columns = vapply(candidate_stats[seq_len(used_iters)], `[[`, integer(1L), "columns"),
    mean_unique_candidates = vapply(candidate_stats[seq_len(used_iters)], `[[`, numeric(1L), "mean_unique"),
    raw_candidate_columns = vapply(candidate_stats[seq_len(used_iters)], `[[`, integer(1L), "raw_columns"),
    iteration_sources = vapply(candidate_stats[seq_len(used_iters)], `[[`, integer(1L), "sources"),
    iteration_neighbors = vapply(candidate_stats[seq_len(used_iters)], `[[`, integer(1L), "neighbors"),
    active_rows = vapply(candidate_stats[seq_len(used_iters)], `[[`, integer(1L), "active_rows"),
    use_reverse = vapply(candidate_stats[seq_len(used_iters)], `[[`, logical(1L), "use_reverse"),
    active_only = vapply(candidate_stats[seq_len(used_iters)], `[[`, logical(1L), "active_only"),
    delta = as.numeric(params$delta),
    seed = as.integer(seed)
  )
  out
}

native_nsg_params <- function(n, p, k, metric = "euclidean", backend = c("cpu", "cuda")) {
  metric <- normalize_nn_metric(metric)
  backend <- match.arg(backend)
  option_prefix <- if (identical(backend, "cuda")) "cuda_nsg" else "cpu_nsg"
  nn_tune_native_nsg_cpp(
    as.integer(n),
    as.integer(p),
    as.integer(k),
    metric,
    backend,
    nn_option_int_or_na(paste0(option_prefix, "_r")),
    nn_option_int_or_na(paste0(option_prefix, "_graph_k"))
  )
}

cuda_nsg_params <- function(n, p, k, metric = "euclidean") {
  native_nsg_params(n = n, p = p, k = k, metric = metric, backend = "cuda")
}

nsg_pair_score <- function(data, i, j, metric = "euclidean") {
  xi <- data[i, , drop = TRUE]
  xj <- data[j, , drop = TRUE]
  if (identical(metric, "inner_product")) {
    return(-sum(xi * xj))
  }
  sqrt(sum((xi - xj)^2))
}

candidate_graph_hnsw_seed_knn <- function(data,
                                          k,
                                          metric = "euclidean",
                                          n_threads = NULL) {
  metric <- normalize_nn_metric(metric)
  if (isTRUE(faiss_available())) {
    out <- faiss_self_knn(
      data,
      k = k,
      backend = "faiss_hnsw",
      metric = metric,
      n_threads = n_threads
    )
    attr(out, "seed_backend") <- "faiss_hnsw"
    return(out)
  }
  if (requireNamespace("RcppHNSW", quietly = TRUE)) {
    out <- rcpphnsw_knn(
      data,
      data,
      k = k,
      self_query = TRUE,
      exclude_self = TRUE,
      metric = metric,
      n_threads = n_threads
    )
    attr(out, "seed_backend") <- "rcpphnsw"
    return(out)
  }
  NULL
}

nsg_prune_candidate_graph <- function(data,
                                      seed_indices,
                                      r,
                                      metric = "euclidean",
                                      protect_top = 0L) {
  n <- nrow(seed_indices)
  r <- as.integer(min(max(1L, r), ncol(seed_indices), max(1L, n - 1L)))
  protect_top <- suppressWarnings(as.integer(protect_top))
  if (length(protect_top) != 1L || is.na(protect_top) || !is.finite(protect_top)) {
    protect_top <- 0L
  }
  protect_top <- as.integer(max(0L, min(protect_top, r, ncol(seed_indices))))
  out <- matrix(NA_integer_, n, r)
  max_exact_work <- faissr_option("cuda_nsg_prune_max_work", 2e8)
  max_exact_work <- suppressWarnings(as.numeric(max_exact_work))
  if (length(max_exact_work) != 1L || is.na(max_exact_work) || !is.finite(max_exact_work)) {
    max_exact_work <- 2e8
  }
  exact_prune <- as.double(n) * as.double(r) * as.double(r) * as.double(ncol(data)) <= max_exact_work

  for (i in seq_len(n)) {
    candidates <- unique(as.integer(seed_indices[i, ]))
    candidates <- candidates[!is.na(candidates) & candidates >= 1L & candidates <= n & candidates != i]
    if (!length(candidates)) {
      candidates <- setdiff(seq_len(n), i)
    }
    protected <- if (protect_top > 0L) {
      head(candidates, protect_top)
    } else {
      integer()
    }
    keep <- protected
    if (isTRUE(exact_prune)) {
      for (cand in candidates) {
        if (cand %in% protected) next
        if (length(keep) >= r) break
        di <- nsg_pair_score(data, i, cand, metric = metric)
        occluded <- FALSE
        for (kept in keep) {
          if (nsg_pair_score(data, kept, cand, metric = metric) < di) {
            occluded <- TRUE
            break
          }
        }
        if (!isTRUE(occluded)) keep <- c(keep, cand)
      }
    }
    if (length(keep) < r) {
      keep <- unique(c(keep, candidates))
    }
    if (length(keep) < r) {
      keep <- unique(c(keep, setdiff(seq_len(n), c(i, keep))))
    }
    out[i, ] <- keep[seq_len(r)]
  }
  attr(out, "exact_prune") <- isTRUE(exact_prune)
  attr(out, "protected_top") <- as.integer(protect_top)
  out
}

vamana_params <- function(n, p, k, metric = "euclidean", backend = c("cpu", "cuda")) {
  metric <- normalize_nn_metric(metric)
  backend <- match.arg(backend)
  out <- nn_tune_vamana_cpp(
    as.integer(n),
    as.integer(p),
    as.integer(k),
    metric,
    nn_option_int_or_na("faiss_vamana_r"),
    nn_option_int_or_na("faiss_vamana_search_l"),
    {
      value <- faissr_option("vamana_alpha", NULL)
      if (is.null(value)) 1.2 else nn_option_double_or_na("vamana_alpha")
    }
  )
  out$backend <- backend
  if (identical(backend, "cuda")) {
    out$seed_backend <- if (identical(metric, "euclidean")) "cuda_exact" else "exact"
  }
  out
}

vamana_robust_prune_candidate_graph <- function(data,
                                                seed_indices,
                                                r,
                                                alpha = 1.2,
                                                metric = "euclidean",
                                                protect_top = 0L) {
  n <- nrow(seed_indices)
  r <- as.integer(min(max(1L, r), ncol(seed_indices), max(1L, n - 1L)))
  protect_top <- suppressWarnings(as.integer(protect_top))
  if (length(protect_top) != 1L || is.na(protect_top) || !is.finite(protect_top)) {
    protect_top <- 0L
  }
  protect_top <- as.integer(max(0L, min(protect_top, r, ncol(seed_indices))))
  alpha <- as.numeric(alpha)
  if (length(alpha) != 1L || is.na(alpha) || !is.finite(alpha) || alpha < 1) alpha <- 1.2
  out <- matrix(NA_integer_, n, r)
  max_exact_work <- faissr_option("vamana_prune_max_work", 2e8)
  max_exact_work <- suppressWarnings(as.numeric(max_exact_work))
  if (length(max_exact_work) != 1L || is.na(max_exact_work) || !is.finite(max_exact_work)) {
    max_exact_work <- 2e8
  }
  exact_prune <- as.double(n) * as.double(r) * as.double(r) * as.double(ncol(data)) <= max_exact_work

  for (i in seq_len(n)) {
    candidates <- unique(as.integer(seed_indices[i, ]))
    candidates <- candidates[!is.na(candidates) & candidates >= 1L & candidates <= n & candidates != i]
    if (!length(candidates)) candidates <- setdiff(seq_len(n), i)
    protected <- if (protect_top > 0L) {
      head(candidates, protect_top)
    } else {
      integer()
    }
    keep <- protected
    if (isTRUE(exact_prune)) {
      for (cand in candidates) {
        if (cand %in% protected) next
        if (length(keep) >= r) break
        di <- nsg_pair_score(data, i, cand, metric = metric)
        pruned <- FALSE
        for (kept in keep) {
          if (alpha * nsg_pair_score(data, kept, cand, metric = metric) <= di) {
            pruned <- TRUE
            break
          }
        }
        if (!isTRUE(pruned)) keep <- c(keep, cand)
      }
    }
    if (length(keep) < r) keep <- unique(c(keep, candidates))
    if (length(keep) < r) keep <- unique(c(keep, setdiff(seq_len(n), c(i, keep))))
    out[i, ] <- keep[seq_len(r)]
  }
  attr(out, "exact_prune") <- isTRUE(exact_prune)
  attr(out, "protected_top") <- as.integer(protect_top)
  out
}

vamana_self_knn <- function(data,
                            k,
                            r,
                            search_l,
                            alpha = 1.2,
                            metric = "euclidean",
                            use_cuda = FALSE,
                            n_threads = NULL,
                            seed_backend = "exact") {
  metric <- normalize_nn_metric(metric)
  if (!metric %in% c("euclidean", "inner_product")) {
    stop("Vamana candidate refinement supports Euclidean or inner-product scoring.", call. = FALSE)
  }
  if (nrow(data) < 2L) {
    stop("Vamana requires at least two rows.", call. = FALSE)
  }
  search_l <- as.integer(min(max(k, search_l), nrow(data) - 1L))
  r <- as.integer(min(max(k, r), search_l))
  ann_seed <- NULL
  if (!isTRUE(use_cuda) && identical(seed_backend, "faiss_hnsw")) {
    ann_seed <- candidate_graph_hnsw_seed_knn(
      data,
      k = search_l,
      metric = metric,
      n_threads = n_threads
    )
  }
  if (!is.null(ann_seed)) {
    seed <- ann_seed
    seed_backend <- paste0("native_", attr(ann_seed, "seed_backend", exact = TRUE), "_seed")
  } else if (isTRUE(use_cuda) && identical(metric, "euclidean")) {
    raw <- nn_cuda_cpp(data, data, as.integer(min(search_l + 1L, nrow(data))), FALSE)
    seed <- drop_self_knn_result(raw, search_l)
    seed_backend <- "native_cuda_exact_seed"
  } else {
    seed <- nn_cpp(
      data,
      data,
      as.integer(search_l),
      metric,
      FALSE,
      FALSE,
      0,
      TRUE,
      as.integer(normalize_nn_threads(n_threads)),
      TRUE
    )
    seed_backend <- if (isTRUE(use_cuda)) "native_cpu_inner_product_seed" else "native_cpu_exact_seed"
  }
  graph <- vamana_robust_prune_candidate_graph(
    data,
    seed$indices,
    r = r,
    alpha = alpha,
    metric = metric,
    protect_top = k
  )
  if (isTRUE(use_cuda)) {
    refined <- row_candidate_knn_cuda_cpp(data, graph, as.integer(k), metric)
    out <- list(indices = refined$indices, distances = refined$distances)
    attr(out, "cuda_kernel") <- "row_candidate_knn"
  } else {
    out <- candidate_knn_cpp(
      data,
      data,
      graph,
      as.integer(k),
      metric,
      FALSE,
      TRUE,
      TRUE,
      as.integer(normalize_nn_threads(n_threads))
    )
  }
  attr(out, "approximation") <- list(
    seed_backend = seed_backend,
    candidate_columns = as.integer(ncol(graph)),
    seed_search_l = as.integer(search_l),
    alpha = as.numeric(alpha),
    protected_seed_neighbors = as.integer(attr(graph, "protected_top", exact = TRUE) %||% 0L),
    exact_robust_prune = isTRUE(attr(graph, "exact_prune", exact = TRUE)),
    cuvs_vamana_note = "cuVS Vamana currently builds/serializes DiskANN-compatible graphs; faissR performs KNN refinement inside the candidate graph."
  )
  out
}

native_nsg_self_knn <- function(data,
                                k,
                                r,
                                graph_k,
                                metric = "euclidean",
                                use_cuda = FALSE,
                                n_threads = NULL,
                                seed_backend = "exact") {
  metric <- normalize_nn_metric(metric)
  if (!metric %in% c("euclidean", "inner_product")) {
    stop("Native NSG candidate refinement supports Euclidean or inner-product scoring.", call. = FALSE)
  }
  if (nrow(data) < 2L) {
    stop("Native NSG requires at least two rows.", call. = FALSE)
  }
  graph_k_cap <- if (isTRUE(use_cuda)) 255L else 512L
  graph_k <- as.integer(min(max(k, graph_k), nrow(data) - 1L, graph_k_cap))
  r <- as.integer(min(max(k, r), graph_k))
  ann_seed <- NULL
  if (!isTRUE(use_cuda) && identical(seed_backend, "faiss_hnsw")) {
    ann_seed <- candidate_graph_hnsw_seed_knn(
      data,
      k = graph_k,
      metric = metric,
      n_threads = n_threads
    )
  }
  if (!is.null(ann_seed)) {
    seed <- ann_seed
    seed_backend <- paste0("native_", attr(ann_seed, "seed_backend", exact = TRUE), "_seed")
  } else if (isTRUE(use_cuda) && identical(metric, "euclidean")) {
    raw <- nn_cuda_cpp(data, data, as.integer(graph_k + 1L), FALSE)
    seed <- drop_self_knn_result(raw, graph_k)
    seed_backend <- "native_cuda_exact_seed"
  } else {
    seed <- nn_cpp(
      data,
      data,
      as.integer(graph_k),
      metric,
      FALSE,
      FALSE,
      0,
      TRUE,
      as.integer(normalize_nn_threads(n_threads)),
      TRUE
    )
    seed_backend <- if (identical(metric, "inner_product")) {
      "native_cpu_inner_product_seed"
    } else if (isTRUE(use_cuda)) {
      "native_cpu_exact_seed"
    } else {
      "native_cpu_exact_seed"
    }
  }
  graph <- nsg_prune_candidate_graph(
    data,
    seed$indices,
    r = r,
    metric = metric,
    protect_top = k
  )
  if (isTRUE(use_cuda)) {
    refined <- row_candidate_knn_cuda_cpp(
      data,
      graph,
      as.integer(k),
      metric
    )
    out <- list(indices = refined$indices, distances = refined$distances)
    attr(out, "cuda_kernel") <- "row_candidate_knn"
  } else {
    out <- candidate_knn_cpp(
      data,
      data,
      graph,
      as.integer(k),
      metric,
      FALSE,
      TRUE,
      TRUE,
      as.integer(normalize_nn_threads(n_threads))
    )
  }
  attr(out, "approximation") <- list(
    seed_backend = seed_backend,
    candidate_columns = as.integer(ncol(graph)),
    seed_graph_k = as.integer(graph_k),
    protected_seed_neighbors = as.integer(attr(graph, "protected_top", exact = TRUE) %||% 0L),
    exact_mrng_prune = isTRUE(attr(graph, "exact_prune", exact = TRUE))
  )
  out
}

cuda_nndescent_self_knn <- function(data,
                                    k,
                                    seed = 4L,
                                    metric = "euclidean") {
  gpu_nndescent_self_knn(data, k = k, backend = "cuda", seed = seed, metric = metric)
}

knn_recall_subset_size <- function(n) {
  value <- faissr_option("gpu_approx_recall_sample", 512L)
  value <- suppressWarnings(as.integer(value))
  if (length(value) != 1L || is.na(value) || !is.finite(value) || value < 1L) {
    value <- 512L
  }
  as.integer(min(n, value))
}

attach_knn_recall_subset <- function(result,
                                     data,
                                     k,
                                     exclude_self,
                                     seed) {
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
  set.seed(as.integer(seed) + 1009L)
  rows <- sort(sample.int(n, sample_size))
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

should_use_nndescent_self_knn <- function(backend,
                                          self_query,
                                          n,
                                          p,
                                          k,
                                          exclude_self,
                                          work_size) {
  if (!identical(backend, "cpu_nndescent")) return(FALSE)
  if (!isTRUE(self_query)) return(FALSE)
  if (n < 10000L || k < 10L || p < 2L) return(FALSE)
  if (work_size < 5e8) return(FALSE)
  nonself_k <- if (isTRUE(exclude_self)) k else k - 1L
  nonself_k >= 1L
}

cpu_nndescent_prefer_faiss <- function() {
  isTRUE(faissr_option("cpu_nndescent_prefer_faiss", FALSE)) &&
    isTRUE(faiss_available())
}

cpu_nndescent_faiss_index <- function() {
  value <- tolower(as.character(faissr_option("cpu_nndescent_faiss_index", "hnsw"))[1L])
  if (!value %in% c("hnsw", "ivf", "flat", "nsg", "nndescent")) {
    warning(
      "Option `faissR.cpu_nndescent_faiss_index` must be one of ",
      "\"hnsw\", \"ivf\", \"flat\", \"nsg\", or \"nndescent\"; using \"hnsw\".",
      call. = FALSE
    )
    value <- "hnsw"
  }
  value
}

faiss_self_knn <- function(data,
                           k,
                           backend = "faiss_ivf",
                           exact = FALSE,
                           seed = 4L,
                           metric = "euclidean",
                           n_threads = NULL) {
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
  if (isTRUE(exact) || identical(backend, "faiss_flat") ||
      identical(backend, "cpu_nndescent_faiss_flat")) {
    out <- if (identical(metric, "inner_product")) {
      nn_faiss_flat_ip_cpp(
        data,
        data,
        as.integer(k),
        TRUE,
        as.integer(n_threads)
      )
    } else {
      nn_faiss_flat_cpp(
        data,
        data,
        as.integer(k),
        TRUE,
        as.integer(n_threads)
      )
    }
    attr(out, "approximation") <- list(
      strategy = if (identical(metric, "inner_product")) "faiss_IndexFlatIP_self" else "faiss_IndexFlatL2_self",
      backend = "faiss",
      library = "faiss",
      exact = TRUE,
      metric = metric,
      seed = as.integer(seed)
    )
    return(out)
  }
  if (identical(backend, "faiss_hnsw") ||
      identical(backend, "cpu_nndescent_faiss_hnsw")) {
    params <- faiss_hnsw_params(
      k,
      n = nrow(data),
      p = ncol(data),
      metric = metric
    )
    out <- nn_faiss_hnsw_cpp(
      data,
      data,
      as.integer(k),
      as.integer(params$m),
      as.integer(params$ef_construction),
      as.integer(params$ef_search),
      faiss_metric_search_arg(metric),
      faiss_metric_distance_output_arg(metric),
      TRUE,
      as.integer(n_threads)
    )
    attr(out, "approximation") <- list(
      strategy = "faiss_IndexHNSWFlat_self",
      backend = backend,
      library = "faiss",
      exact = FALSE,
      metric = metric,
      m = as.integer(out$m),
      ef_construction = as.integer(out$ef_construction),
      ef_search = as.integer(out$ef_search),
      requested_m = as.integer(out$requested_m),
      requested_ef_construction = as.integer(out$requested_ef_construction),
      requested_ef_search = as.integer(out$requested_ef_search),
      hnsw_parameters_adjusted = isTRUE(out$hnsw_parameters_adjusted),
      tuning_policy = params$policy,
      tuning_rule = params$rule,
      tuning_high_dim = isTRUE(params$high_dim),
      tuning_large_n = isTRUE(params$large_n),
      tuning_small_k = isTRUE(params$small_k),
      tuning_large_k = isTRUE(params$large_k),
      tuning_non_euclidean = isTRUE(params$non_euclidean),
      tuning_source = params$tuning_source %||% "cpp",
      seed = as.integer(seed)
    )
    return(out)
  }
  if (identical(backend, "faiss_nsg") ||
      identical(backend, "cpu_nndescent_faiss_nsg")) {
    if (metric %in% c("cosine", "correlation", "inner_product")) {
      stop(
        "`backend = \"faiss_nsg\"` currently supports only `metric = \"euclidean\"`. ",
        "FAISS NSG graph construction can abort the R process for normalized ",
        "cosine/correlation or raw inner-product routes in this linked FAISS build.",
        call. = FALSE
      )
    }
    params <- faiss_nsg_params(k)
    out <- nn_faiss_nsg_cpp(
      data,
      data,
      as.integer(k),
      as.integer(params$r),
      as.integer(params$search_l),
      as.integer(params$build_type),
      faiss_metric_search_arg(metric),
      faiss_metric_distance_output_arg(metric),
      TRUE,
      as.integer(n_threads)
    )
    attr(out, "approximation") <- list(
      strategy = "faiss_IndexNSGFlat_self",
      backend = backend,
      library = "faiss",
      exact = FALSE,
      r = as.integer(out$r),
      search_l = as.integer(out$search_l),
      build_type = as.integer(out$build_type),
      gk = as.integer(out$gk),
      requested_r = as.integer(out$requested_r),
      requested_search_l = as.integer(out$requested_search_l),
      requested_build_type = as.integer(out$requested_build_type),
      nsg_parameters_adjusted = isTRUE(out$nsg_parameters_adjusted),
      seed = as.integer(seed)
    )
    return(out)
  }
  if (identical(backend, "faiss_nndescent") ||
      identical(backend, "cpu_nndescent_faiss_nndescent")) {
    if (!identical(metric, "euclidean")) {
      stop(
        "`backend = \"faiss_nndescent\"` is currently validated only for ",
        "`metric = \"euclidean\"` in this FAISS build.",
        call. = FALSE
      )
    }
    if (!isTRUE(faissr_option("enable_faiss_nndescent", FALSE))) {
      stop(
        "FAISS NNDescent is disabled by default because linked FAISS builds can ",
        "abort the R process during graph construction.",
        call. = FALSE
      )
    }
    params <- faiss_nndescent_params(k)
    out <- nn_faiss_nndescent_cpp(
      data,
      data,
      as.integer(k),
      as.integer(params$graph_k),
      as.integer(params$n_iter),
      as.integer(params$search_l),
      "euclidean",
      "euclidean",
      TRUE,
      as.integer(n_threads)
    )
    attr(out, "approximation") <- list(
      strategy = "faiss_IndexNNDescentFlat_self",
      backend = backend,
      library = "faiss",
      exact = FALSE,
      graph_k = as.integer(out$graph_k),
      n_iter = as.integer(out$n_iter),
      search_l = as.integer(out$search_l),
      requested_graph_k = as.integer(out$requested_graph_k),
      requested_n_iter = as.integer(out$requested_n_iter),
      requested_search_l = as.integer(out$requested_search_l),
      nndescent_parameters_adjusted = isTRUE(out$nndescent_parameters_adjusted),
      seed = as.integer(seed)
    )
    return(out)
  }
  params <- faiss_ivf_params(n, k, metric = metric)
  out <- nn_faiss_ivf_cpp(
    data,
    data,
    as.integer(k),
    as.integer(params$nlist),
    as.integer(params$nprobe),
    "euclidean",
    "euclidean",
    TRUE,
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

rcpphnsw_params <- function(k) {
  nn_tune_rcpphnsw_cpp(
    as.integer(k),
    nn_option_int_or_na("hnsw_m"),
    nn_option_int_or_na("hnsw_ef_construction"),
    nn_option_int_or_na("hnsw_ef")
  )
}

rcpphnsw_knn <- function(data,
                         points,
                         k,
                         self_query,
                         exclude_self,
                         metric = "euclidean",
                         n_threads = NULL) {
  if (!requireNamespace("RcppHNSW", quietly = TRUE)) {
    stop(
      "The RcppHNSW fallback backend is not installed. ",
      "Install it with `install.packages(\"RcppHNSW\")`, or use `backend = \"cpu\"`.",
      call. = FALSE
    )
  }
  metric <- normalize_nn_metric(metric)
  if (identical(metric, "correlation")) {
    data <- row_center_l2_normalize(data)
    if (isTRUE(self_query)) {
      points <- data
    } else {
      points <- row_center_l2_normalize(points)
    }
  }
  hnsw_metric <- switch(
    metric,
    euclidean = "euclidean",
    cosine = "cosine",
    correlation = "cosine",
    inner_product = "ip"
  )
  n_threads <- normalize_nn_threads(n_threads)
  params <- rcpphnsw_params(k)

  raw <- if (isTRUE(self_query)) {
    query_k <- if (isTRUE(exclude_self)) k + 1L else k
    RcppHNSW::hnsw_knn(
      data,
      k = as.integer(query_k),
      distance = hnsw_metric,
      M = as.integer(params$m),
      ef_construction = as.integer(params$ef_construction),
      ef = as.integer(max(params$ef, query_k)),
      verbose = FALSE,
      progress = "none",
      n_threads = as.integer(n_threads),
      byrow = TRUE,
      random_seed = as.integer(fast_knn_approx_seed())
    )
  } else {
    index <- RcppHNSW::hnsw_build(
      data,
      distance = hnsw_metric,
      M = as.integer(params$m),
      ef = as.integer(params$ef_construction),
      verbose = FALSE,
      progress = "none",
      n_threads = as.integer(n_threads),
      byrow = TRUE,
      random_seed = as.integer(fast_knn_approx_seed())
    )
    RcppHNSW::hnsw_search(
      points,
      index,
      k = as.integer(k),
      ef = as.integer(max(params$ef, k)),
      verbose = FALSE,
      progress = "none",
      n_threads = as.integer(n_threads),
      byrow = TRUE
    )
  }

  out <- list(indices = as.matrix(raw$idx), distances = as.matrix(raw$dist))
  storage.mode(out$indices) <- "integer"
  storage.mode(out$distances) <- "double"
  if (identical(metric, "inner_product") && ncol(out$distances) > 0L) {
    out$distances <- out$distances - out$distances[, 1L]
  }
  if (isTRUE(self_query) && isTRUE(exclude_self)) {
    out <- drop_self_knn_result(out, k)
  }
  attr(out, "approximation") <- list(
    strategy = "RcppHNSW_hnswlib",
    backend = "hnsw",
    library = "RcppHNSW",
    metric = metric,
    exact = FALSE,
    m = as.integer(params$m),
    ef_construction = as.integer(params$ef_construction),
    ef = as.integer(params$ef),
    n_threads = as.integer(n_threads),
    tuning_policy = params$tuning_policy,
    tuning_rule = params$tuning_rule,
    tuning_source = params$tuning_source %||% "cpp"
  )
  out
}

nndescent_pool_size <- function(n, k) {
  as.integer(nn_tune_cpu_nndescent_cpp(as.integer(n), as.integer(k))$pool_size)
}

nndescent_iterations <- function(n, k) {
  as.integer(nn_tune_cpu_nndescent_cpp(as.integer(n), as.integer(k))$n_iters)
}

nndescent_self_knn <- function(data,
                               k,
                               seed = 4L,
                               n_threads = NULL,
                               metric = "euclidean") {
  n <- nrow(data)
  k <- as.integer(k)
  metric <- normalize_nn_metric(metric)
  if (!metric %in% c("euclidean", "inner_product")) {
    stop(
      "Native CPU NN-descent expects raw euclidean or inner_product input. ",
      "Cosine and correlation are normalized before this helper is called.",
      call. = FALSE
    )
  }
  if (length(k) != 1L || is.na(k) || !is.finite(k) || k < 1L || k >= n) {
    stop("`k` must be in [1, nrow(data) - 1].", call. = FALSE)
  }
  if (is.null(n_threads)) {
    n_threads <- suppressWarnings(parallel::detectCores(logical = FALSE))
    if (length(n_threads) != 1L || is.na(n_threads) || !is.finite(n_threads)) {
      n_threads <- 1L
    }
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
  tune <- nn_tune_cpu_nndescent_cpp(as.integer(n), as.integer(k))
  pool_size <- as.integer(tune$pool_size)
  n_iters <- as.integer(tune$n_iters)
  max_candidates <- as.integer(tune$max_candidates)
  n_random_projections <- as.integer(tune$n_random_projections)
  out <- nndescent_self_knn_cpp(
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
  params <- list(
    strategy = "native_cpu_nndescent",
    backend = "cpu",
    pool_size = pool_size,
    n_iters = n_iters,
    max_candidates = max_candidates,
    n_random_projections = n_random_projections,
    reverse_candidates = "rank_ordered",
    metric = metric,
    tuning_policy = tune$tuning_policy,
    tuning_rule = tune$tuning_rule,
    tuning_large_n = isTRUE(tune$tuning_large_n),
    tuning_small_k = isTRUE(tune$tuning_small_k),
    tuning_source = tune$tuning_source %||% "cpp"
  )
  attr(out, "nndescent") <- params
  attr(out, "approximation") <- params
  out
}

clustered_knn_center_count <- function(n, k) {
  n <- as.integer(n)
  k <- as.integer(k)
  max_centers <- max(2L, n - 1L)
  count <- max(16L, ceiling(sqrt(n) * 2))
  count <- max(count, ceiling(n / max(250L, 10L * k)))
  as.integer(min(max_centers, count))
}

clustered_knn_graph_columns <- function(projection_k, k) {
  projection_k <- as.integer(max(1L, projection_k))
  k <- as.integer(max(1L, k))
  bucket_cols <- min(projection_k, max(2L, min(4L, ceiling(k / 10))))
  query_cols <- min(projection_k, max(bucket_cols, min(8L, ceiling(k / 5))))
  list(bucket_cols = bucket_cols, query_cols = query_cols)
}

clustered_self_knn <- function(data,
                               k,
                               exclude_self = TRUE,
                               seed = 4L,
                               n_threads = NULL) {
  n <- nrow(data)
  if (n < 2L) {
    stop("`data` must have at least two rows.", call. = FALSE)
  }
  k <- as.integer(k)
  if (length(k) != 1L || is.na(k) || !is.finite(k) || k < 1L) {
    stop("`k` must be a positive integer.", call. = FALSE)
  }

  nonself_k <- if (isTRUE(exclude_self)) k else k - 1L
  if (nonself_k < 1L) {
    return(list(
      indices = matrix(seq_len(n), n, 1L),
      distances = matrix(0, n, 1L)
    ))
  }
  if (nonself_k >= n) {
    stop("`k` cannot be larger than the available neighbor count.", call. = FALSE)
  }

  n_centers <- clustered_knn_center_count(n, nonself_k)
  centers <- select_landmark_rows(data, n_centers, seed)
  x_centers <- data[centers, , drop = FALSE]
  projection_k <- min(n_centers, max(2L, min(12L, ceiling(nonself_k / 2))))
  projection <- nn_compute(
    x_centers,
    data,
    k = projection_k,
    backend = "cpu",
    points_missing = FALSE,
    exclude_self = FALSE,
    n_threads = n_threads
  )
  cols <- clustered_knn_graph_columns(ncol(projection$indices), nonself_k)
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
  if (!identical(typeof(out$distances), "double")) storage.mode(out$distances) <- "double"

  if (!isTRUE(exclude_self)) {
    out$indices <- cbind(seq_len(n), out$indices)
    out$distances <- cbind(rep(0, n), out$distances)
  }
  out
}

nn_option_int_or_na <- function(name) {
  value <- faissr_option(name, NULL)
  if (is.null(value)) return(NA_integer_)
  value <- suppressWarnings(as.integer(value))
  if (length(value) != 1L || is.na(value) || !is.finite(value)) {
    return(NA_integer_)
  }
  as.integer(value)
}

nn_option_double_or_na <- function(name) {
  value <- faissr_option(name, NULL)
  if (is.null(value)) return(NA_real_)
  value <- suppressWarnings(as.numeric(value))
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

faiss_ivf_params <- function(n, k, metric = "euclidean") {
  metric <- normalize_nn_metric(metric)
  nn_tune_faiss_ivf_cpp(
    as.integer(n),
    as.integer(k),
    metric,
    nn_option_int_or_na(c("faiss_nlist", "ivf_nlist")),
    nn_option_int_or_na(c("faiss_nprobe", "ivf_nprobe")),
    faiss_ivf_manual_params()
  )
}

faiss_ivf_manual_params <- function() {
  any(vapply(
    c("faiss_nlist", "ivf_nlist", "faiss_nprobe", "ivf_nprobe"),
    function(name) !is.null(faissr_option(name, NULL)),
    logical(1)
  ))
}

cuvs_ivfpq_params <- function(p) {
  nn_tune_cuvs_ivfpq_cpp(
    as.integer(p),
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

.faiss_gpu_ivf_tune_cache <- new.env(parent = emptyenv())
.faiss_gpu_ivf_tune_disk_cache <- new.env(parent = emptyenv())
.faiss_gpu_ivf_tune_disk_cache$loaded <- FALSE
.faiss_gpu_ivf_tune_disk_cache$file <- NULL
.faiss_gpu_ivf_tune_disk_cache$entries <- list()

faiss_gpu_ivf_tune_policy <- function(tuning = "auto") {
  tuning <- normalize_nn_tuning(tuning)
  if (!identical(tuning, "auto")) return(tuning)
  policy <- faissr_option("faiss_gpu_ivf_tune_policy", "fixed")
  policy <- tolower(as.character(policy)[1L])
  if (!policy %in% c("cache", "pilot", "fixed", "off")) {
    policy <- "fixed"
  }
  policy
}

faiss_gpu_ivf_tune_cache_file <- function() {
  path <- faissr_option("faiss_gpu_ivf_tune_cache_file", NULL)
  if (!is.null(path)) return(path)
  root <- tryCatch(
    tools::R_user_dir("faissR", which = "cache"),
    error = function(e) file.path(tempdir(), "faissR-cache")
  )
  file.path(root, "faiss_gpu_ivf_tuning.rds")
}

faiss_gpu_ivf_should_tune <- function(data, k, self_query, tuning = "auto", metric = "euclidean") {
  tuning <- normalize_nn_tuning(tuning)
  metric <- normalize_nn_metric(metric)
  policy <- faiss_gpu_ivf_tune_policy(tuning)
  if (!policy %in% c("cache", "pilot")) return(FALSE)
  if (!identical(metric, "euclidean")) return(FALSE)
  if (!isTRUE(self_query)) return(FALSE)
  if (!isTRUE(faiss_gpu_available())) return(FALSE)
  if (isTRUE(faiss_ivf_manual_params())) return(FALSE)
  enabled <- faissr_option("faiss_gpu_ivf_tune", TRUE)
  if (!isTRUE(enabled)) return(FALSE)
  threshold <- faissr_option("faiss_gpu_ivf_tune_threshold", 20000L)
  threshold <- suppressWarnings(as.integer(threshold))
  if (length(threshold) != 1L || is.na(threshold) || !is.finite(threshold)) {
    threshold <- 20000L
  }
  nrow(data) >= threshold && as.integer(k) >= 10L
}

faiss_gpu_ivf_tune_signature <- function(data, k, sample_size, target, seed) {
  rows <- unique(as.integer(round(seq(1L, nrow(data), length.out = min(17L, nrow(data))))))
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
  if (isTRUE(.faiss_gpu_ivf_tune_disk_cache$loaded) &&
      identical(.faiss_gpu_ivf_tune_disk_cache$file, file)) {
    return(invisible(.faiss_gpu_ivf_tune_disk_cache$entries))
  }
  .faiss_gpu_ivf_tune_disk_cache$loaded <- TRUE
  .faiss_gpu_ivf_tune_disk_cache$file <- file
  entries <- tryCatch(suppressWarnings(readRDS(file)), error = function(e) list())
  if (!is.list(entries)) entries <- list()
  .faiss_gpu_ivf_tune_disk_cache$entries <- entries
  invisible(entries)
}

faiss_gpu_ivf_get_cached_tuning <- function(key) {
  cached <- .faiss_gpu_ivf_tune_cache[[key]]
  if (!is.null(cached)) {
    if (is.list(cached$tuning)) cached$tuning$cache <- "memory"
    return(cached)
  }
  faiss_gpu_ivf_load_disk_cache()
  cached <- .faiss_gpu_ivf_tune_disk_cache$entries[[key]]
  if (!is.null(cached)) {
    if (is.list(cached$tuning)) cached$tuning$cache <- "disk"
    .faiss_gpu_ivf_tune_cache[[key]] <- cached
    return(cached)
  }
  NULL
}

faiss_gpu_ivf_set_cached_tuning <- function(key, value, persist = TRUE) {
  .faiss_gpu_ivf_tune_cache[[key]] <- value
  if (!isTRUE(persist)) return(invisible(value))
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
    return(list(
      params = base_params,
      tuning = list(status = "disabled", policy = policy)
    ))
  }
  sample_size <- faissr_option("faiss_gpu_ivf_tune_sample", 10000L)
  sample_size <- suppressWarnings(as.integer(sample_size))
  if (length(sample_size) != 1L || is.na(sample_size) || !is.finite(sample_size) || sample_size < 1000L) {
    sample_size <- 10000L
  }
  sample_size <- as.integer(min(nrow(data), sample_size))
  seed <- faissr_option("faiss_gpu_ivf_tune_seed", 7L)
  seed <- suppressWarnings(as.integer(seed))
  if (length(seed) != 1L || is.na(seed) || !is.finite(seed)) seed <- 7L
  target <- faissr_option("faiss_gpu_ivf_tune_recall", 0.985)
  target <- suppressWarnings(as.numeric(target))
  if (length(target) != 1L || is.na(target) || !is.finite(target)) target <- 0.985
  target <- max(0, min(1, target))
  key <- faiss_gpu_ivf_tune_signature(data, k, sample_size, target, seed)
  cached <- faiss_gpu_ivf_get_cached_tuning(key)
  if (!is.null(cached)) return(cached)
  if (identical(policy, "fixed")) {
    return(list(
      params = base_params,
      tuning = list(
        status = "fixed_default",
        policy = policy,
        cache = "miss",
        sample_size = sample_size,
        target_recall = as.numeric(target)
      )
    ))
  }

  old_seed <- if (exists(".Random.seed", envir = .GlobalEnv, inherits = FALSE)) {
    get(".Random.seed", envir = .GlobalEnv, inherits = FALSE)
  } else {
    NULL
  }
  on.exit({
    if (is.null(old_seed)) {
      rm(".Random.seed", envir = .GlobalEnv)
    } else {
      assign(".Random.seed", old_seed, envir = .GlobalEnv)
    }
  }, add = TRUE)
  set.seed(seed)
  rows <- sort(sample.int(nrow(data), sample_size))
  x <- data[rows, , drop = FALSE]
  compare_k <- as.integer(min(k, nrow(x)))
  reference <- tryCatch(
    nn_faiss_gpu_flat_cpp(x, x, compare_k, FALSE),
    error = function(e) e
  )
  if (inherits(reference, "error")) {
    out <- list(
      params = base_params,
      tuning = list(
        status = "failed",
        policy = policy,
        cache = "miss",
        reason = conditionMessage(reference),
        sample_size = sample_size
      )
    )
    faiss_gpu_ivf_set_cached_tuning(key, out, persist = !identical(policy, "pilot"))
    return(out)
  }

  candidates <- faiss_gpu_ivf_candidate_params(nrow(x), compare_k, base_params)
  rows_out <- vector("list", nrow(candidates))
  for (i in seq_len(nrow(candidates))) {
    cand <- candidates[i, , drop = FALSE]
    elapsed <- system.time({
      approx <- tryCatch(
        nn_faiss_gpu_ivf_flat_cpp(
          x,
          x,
          compare_k,
          as.integer(cand$nlist),
          as.integer(cand$nprobe),
          "euclidean",
          "euclidean",
          FALSE
        ),
        error = function(e) e
      )
    })[["elapsed"]]
    if (inherits(approx, "error")) {
      rows_out[[i]] <- data.frame(
        nlist = cand$nlist,
        nprobe = cand$nprobe,
        seconds = as.numeric(elapsed),
        recall = NA_real_,
        status = "failed",
        error = conditionMessage(approx),
        stringsAsFactors = FALSE
      )
    } else {
      recall <- .knn_recall_summary(approx, reference, k = compare_k)$recall_at_k
      rows_out[[i]] <- data.frame(
        nlist = cand$nlist,
        nprobe = cand$nprobe,
        seconds = as.numeric(elapsed),
        recall = as.numeric(recall),
        status = "success",
        error = "",
        stringsAsFactors = FALSE
      )
    }
  }

  results <- do.call(rbind, rows_out)
  success <- results[results$status == "success", , drop = FALSE]
  if (nrow(success) < 1L) {
    chosen <- base_params
    status <- "failed"
  } else {
    eligible <- success[is.finite(success$recall) & success$recall >= target, , drop = FALSE]
    if (nrow(eligible) > 0L) {
      chosen_row <- eligible[order(eligible$seconds, -eligible$recall), , drop = FALSE][1L, , drop = FALSE]
      status <- "target_met"
    } else {
      chosen_row <- success[order(-success$recall, success$seconds), , drop = FALSE][1L, , drop = FALSE]
      status <- "best_available"
    }
    chosen <- list(
      nlist = as.integer(chosen_row$nlist),
      nprobe = as.integer(chosen_row$nprobe)
    )
  }
  tuning <- list(
    status = status,
    policy = policy,
    cache = "miss",
    sample_size = as.integer(sample_size),
    target_recall = as.numeric(target),
    chosen = chosen,
    results = results
  )
  out <- list(params = chosen, tuning = tuning)
  faiss_gpu_ivf_set_cached_tuning(key, out, persist = !identical(policy, "pilot"))
  out
}

faiss_option_int <- function(name, default, min_value = 1L, max_value = .Machine$integer.max) {
  value <- faissr_option(paste0("faiss_", name), NULL)
  value <- if (is.null(value)) default else suppressWarnings(as.integer(value))
  if (length(value) != 1L || is.na(value) || !is.finite(value)) value <- default
  as.integer(max(min_value, min(max_value, value)))
}

faiss_pq_default_m <- function(p) {
  p <- as.integer(p)
  candidates <- c(64L, 56L, 48L, 40L, 32L, 28L, 24L, 16L, 14L, 12L, 8L, 7L, 4L, 2L, 1L)
  candidates <- candidates[candidates <= p & p %% candidates == 0L]
  if (length(candidates) == 0L) return(1L)
  as.integer(candidates[[1L]])
}

faiss_cpu_ivfpq_min_training_rows <- function() 624L

faiss_cpu_ivfpq_8bit_training_rows <- function() 9984L

validate_faiss_cpu_ivfpq_training_size <- function(n) {
  n <- suppressWarnings(as.integer(n))
  min_n <- faiss_cpu_ivfpq_min_training_rows()
  if (length(n) != 1L || is.na(n) || n < min_n) {
    stop(
      "FAISS CPU IVFPQ requires at least ", min_n,
      " training rows for the smallest supported 4-bit product quantizer. ",
      "Use `method = \"ivf\"`, `\"hnsw\"`, or `\"flat\"` for smaller datasets.",
      call. = FALSE
    )
  }
  invisible(TRUE)
}

faiss_pq_params <- function(p, n = NULL) {
  nn_tune_faiss_pq_cpp(
    as.integer(p),
    suppressWarnings(as.integer(n %||% NA_integer_)),
    nn_option_int_or_na("faiss_pq_m"),
    nn_option_int_or_na("faiss_pq_nbits"),
    nn_any_options(c("faiss_pq_m", "faiss_pq_nbits")),
    !is.null(faissr_option("faiss_pq_nbits", NULL))
  )
}

faiss_hnsw_auto_policy <- function(n = NULL, p = NULL, k, metric = "euclidean") {
  out <- nn_tune_faiss_hnsw_cpp(
    suppressWarnings(as.integer(n %||% NA_integer_)),
    suppressWarnings(as.integer(p %||% NA_integer_)),
    as.integer(k),
    normalize_nn_metric(metric),
    NA_integer_,
    NA_integer_,
    NA_integer_,
    FALSE
  )
  out[c("m", "ef_construction", "ef_search", "rule", "high_dim",
        "large_n", "small_k", "large_k", "non_euclidean")]
}

faiss_hnsw_manual_params <- function() {
  any(vapply(
    c("hnsw_m", "hnsw_ef_construction", "hnsw_ef_search"),
    function(name) !is.null(faissr_option(paste0("faiss_", name), NULL)),
    logical(1)
  ))
}

faiss_hnsw_params <- function(k, n = NULL, p = NULL, metric = "euclidean") {
  nn_tune_faiss_hnsw_cpp(
    suppressWarnings(as.integer(n %||% NA_integer_)),
    suppressWarnings(as.integer(p %||% NA_integer_)),
    as.integer(k),
    normalize_nn_metric(metric),
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
    nn_any_options(c("faiss_nsg_r", "faiss_nsg_search_l", "faiss_nsg_build_type"))
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

cuvs_option_int <- function(name, default, min_value = 1L, max_value = .Machine$integer.max) {
  value <- faissr_option(paste0("cuvs_", name), NULL)
  value <- if (is.null(value)) default else suppressWarnings(as.integer(value))
  if (length(value) != 1L || is.na(value) || !is.finite(value)) value <- default
  as.integer(max(min_value, min(max_value, value)))
}

cuvs_requested_option_int <- function(name, default) {
  value <- faissr_option(paste0("cuvs_", name), NULL)
  value <- if (is.null(value)) default else suppressWarnings(as.integer(value))
  if (length(value) != 1L || is.na(value) || !is.finite(value)) value <- default
  as.integer(value)
}

cuvs_cagra_params <- function(n, k, p = NA_integer_) {
  nn_tune_cuvs_cagra_cpp(
    as.integer(n),
    suppressWarnings(as.integer(p)),
    as.integer(k),
    nn_option_int_or_na("cuvs_graph_degree"),
    nn_option_int_or_na("cuvs_intermediate_graph_degree"),
    nn_option_int_or_na("cuvs_search_width"),
    nn_option_int_or_na("cuvs_itopk_size"),
    cuvs_cagra_manual_params()
  )
}

cuvs_hnsw_params <- function(n, k, p = NA_integer_, n_threads = NULL) {
  nn_tune_cuvs_hnsw_cpp(
    as.integer(n),
    suppressWarnings(as.integer(p)),
    as.integer(k),
    as.integer(normalize_nn_threads(n_threads)),
    cagra_build_algo_preference(),
    nn_option_int_or_na("cuvs_graph_degree"),
    nn_option_int_or_na("cuvs_intermediate_graph_degree"),
    nn_option_int_or_na("cuvs_search_width"),
    nn_option_int_or_na("cuvs_itopk_size"),
    nn_option_int_or_na("cuvs_hnsw_ef"),
    cuvs_cagra_manual_params()
  )
}

.cuvs_cagra_tune_cache <- new.env(parent = emptyenv())
.cuvs_cagra_tune_disk_cache <- new.env(parent = emptyenv())
.cuvs_cagra_tune_disk_cache$loaded <- FALSE
.cuvs_cagra_tune_disk_cache$file <- NULL
.cuvs_cagra_tune_disk_cache$entries <- list()

cuvs_cagra_manual_params <- function() {
  any(vapply(
    c("graph_degree", "intermediate_graph_degree", "search_width", "itopk_size"),
    function(name) !is.null(faissr_option(paste0("cuvs_", name), NULL)),
    logical(1)
  ))
}

cuvs_cagra_should_tune <- function(data, k, self_query, tuning = "auto") {
  tuning <- normalize_nn_tuning(tuning)
  policy <- cuvs_cagra_tune_policy(tuning)
  if (!policy %in% c("cache", "pilot")) return(FALSE)
  if (!isTRUE(self_query)) return(FALSE)
  if (!isTRUE(cuvs_available())) return(FALSE)
  if (isTRUE(cuvs_cagra_manual_params())) return(FALSE)
  enabled <- faissr_option("cuvs_cagra_tune", TRUE)
  if (!isTRUE(enabled)) return(FALSE)
  threshold <- faissr_option("cuvs_cagra_tune_threshold", 20000L)
  threshold <- suppressWarnings(as.integer(threshold))
  if (length(threshold) != 1L || is.na(threshold) || !is.finite(threshold)) {
    threshold <- 20000L
  }
  nrow(data) >= threshold && as.integer(k) >= 10L
}

cuvs_cagra_tune_policy <- function(tuning = "auto") {
  tuning <- normalize_nn_tuning(tuning)
  if (!identical(tuning, "auto")) return(tuning)
  policy <- faissr_option("cuvs_cagra_tune_policy", "fixed")
  policy <- tolower(as.character(policy)[1L])
  if (!policy %in% c("cache", "pilot", "fixed", "off")) {
    policy <- "fixed"
  }
  policy
}

cuvs_cagra_tune_cache_file <- function() {
  path <- faissr_option("cuvs_cagra_tune_cache_file", NULL)
  if (!is.null(path)) return(path)
  root <- tryCatch(
    tools::R_user_dir("faissR", which = "cache"),
    error = function(e) file.path(tempdir(), "faissR-cache")
  )
  file.path(root, "cuvs_cagra_tuning.rds")
}

cuvs_cagra_tune_signature <- function(data, k, sample_size, target, seed) {
  rows <- unique(as.integer(round(seq(1L, nrow(data), length.out = min(17L, nrow(data))))))
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
  if (isTRUE(.cuvs_cagra_tune_disk_cache$loaded) &&
      identical(.cuvs_cagra_tune_disk_cache$file, file)) {
    return(invisible(.cuvs_cagra_tune_disk_cache$entries))
  }
  .cuvs_cagra_tune_disk_cache$loaded <- TRUE
  .cuvs_cagra_tune_disk_cache$file <- file
  entries <- tryCatch(suppressWarnings(readRDS(file)), error = function(e) list())
  if (!is.list(entries)) entries <- list()
  .cuvs_cagra_tune_disk_cache$entries <- entries
  invisible(entries)
}

cuvs_cagra_get_cached_tuning <- function(key) {
  cached <- .cuvs_cagra_tune_cache[[key]]
  if (!is.null(cached)) {
    if (is.list(cached$tuning)) cached$tuning$cache <- "memory"
    return(cached)
  }
  cuvs_cagra_load_disk_cache()
  cached <- .cuvs_cagra_tune_disk_cache$entries[[key]]
  if (!is.null(cached)) {
    if (is.list(cached$tuning)) cached$tuning$cache <- "disk"
    .cuvs_cagra_tune_cache[[key]] <- cached
    return(cached)
  }
  NULL
}

cuvs_cagra_set_cached_tuning <- function(key, value, persist = TRUE) {
  .cuvs_cagra_tune_cache[[key]] <- value
  if (!isTRUE(persist)) return(invisible(value))
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
    intermediate_graph_degree = c(max(128L, 2L * base_degree), max(192L, 2L * max(96L, k + 1L)), max(256L, 2L * max(128L, k + 1L))),
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

cuvs_cagra_tune_params <- function(data, k, base_params, tuning = "auto", build_algo = "auto") {
  policy <- cuvs_cagra_tune_policy(tuning)
  if (identical(policy, "off")) {
    return(list(
      params = base_params,
      tuning = list(status = "disabled", policy = policy)
    ))
  }
  sample_size <- faissr_option("cuvs_cagra_tune_sample", 2048L)
  sample_size <- suppressWarnings(as.integer(sample_size))
  if (length(sample_size) != 1L || is.na(sample_size) || !is.finite(sample_size) || sample_size < 256L) {
    sample_size <- 2048L
  }
  sample_size <- as.integer(min(nrow(data), sample_size))
  seed <- faissr_option("cuvs_cagra_tune_seed", 4L)
  seed <- suppressWarnings(as.integer(seed))
  if (length(seed) != 1L || is.na(seed) || !is.finite(seed)) seed <- 4L
  target <- faissr_option("cuvs_cagra_tune_recall", 0.985)
  target <- suppressWarnings(as.numeric(target))
  if (length(target) != 1L || is.na(target) || !is.finite(target)) target <- 0.985
  target <- max(0, min(1, target))
  key <- cuvs_cagra_tune_signature(data, k, sample_size, target, seed)
  cached <- cuvs_cagra_get_cached_tuning(key)
  if (!is.null(cached)) return(cached)
  if (identical(policy, "fixed")) {
    return(list(
      params = base_params,
      tuning = list(
        status = "fixed_default",
        policy = policy,
        cache = "miss",
        sample_size = sample_size,
        target_recall = as.numeric(target)
      )
    ))
  }
  old_seed <- if (exists(".Random.seed", envir = .GlobalEnv, inherits = FALSE)) {
    get(".Random.seed", envir = .GlobalEnv, inherits = FALSE)
  } else {
    NULL
  }
  on.exit({
    if (is.null(old_seed)) {
      rm(".Random.seed", envir = .GlobalEnv)
    } else {
      assign(".Random.seed", old_seed, envir = .GlobalEnv)
    }
  }, add = TRUE)
  set.seed(seed)
  rows <- sort(sample.int(nrow(data), sample_size))
  x <- data[rows, , drop = FALSE]
  compare_k <- as.integer(min(k, nrow(x)))
  reference <- tryCatch(
    nn_cuvs_bruteforce_cpp(x, x, compare_k, FALSE),
    error = function(e) e
  )
  if (inherits(reference, "error")) {
    out <- list(
      params = base_params,
      tuning = list(
        status = "failed",
        policy = policy,
        cache = "miss",
        reason = conditionMessage(reference),
        sample_size = sample_size
      )
    )
    cuvs_cagra_set_cached_tuning(key, out, persist = !identical(policy, "pilot"))
    return(out)
  }

  candidates <- cuvs_cagra_candidate_params(compare_k, nrow(x))

  rows_out <- vector("list", nrow(candidates))
  for (i in seq_len(nrow(candidates))) {
    cand <- candidates[i, , drop = FALSE]
    elapsed <- system.time({
      approx <- tryCatch(
        nn_cuvs_cagra_cpp(
          x,
          x,
          compare_k,
          FALSE,
          as.integer(cand$graph_degree),
          as.integer(cand$intermediate_graph_degree),
          as.integer(cand$search_width),
          as.integer(cand$itopk_size),
          build_algo
        ),
        error = function(e) e
      )
    })[["elapsed"]]
    if (inherits(approx, "error")) {
      rows_out[[i]] <- data.frame(
        graph_degree = cand$graph_degree,
        intermediate_graph_degree = cand$intermediate_graph_degree,
        search_width = cand$search_width,
        itopk_size = cand$itopk_size,
        seconds = as.numeric(elapsed),
        recall = NA_real_,
        status = "failed",
        error = conditionMessage(approx),
        stringsAsFactors = FALSE
      )
    } else {
      recall <- .knn_recall_summary(approx, reference, k = compare_k)$recall_at_k
      rows_out[[i]] <- data.frame(
        graph_degree = cand$graph_degree,
        intermediate_graph_degree = cand$intermediate_graph_degree,
        search_width = cand$search_width,
        itopk_size = cand$itopk_size,
        seconds = as.numeric(elapsed),
        recall = as.numeric(recall),
        status = "success",
        error = "",
        stringsAsFactors = FALSE
      )
    }
  }

  results <- do.call(rbind, rows_out)
  success <- results[results$status == "success", , drop = FALSE]
  if (nrow(success) < 1L) {
    chosen <- base_params
    status <- "failed"
  } else {
    eligible <- success[is.finite(success$recall) & success$recall >= target, , drop = FALSE]
    if (nrow(eligible) > 0L) {
      chosen_row <- eligible[order(eligible$seconds, -eligible$recall), , drop = FALSE][1L, , drop = FALSE]
      status <- "target_met"
    } else {
      chosen_row <- success[order(-success$recall, success$seconds), , drop = FALSE][1L, , drop = FALSE]
      status <- "best_available"
    }
    chosen <- list(
      graph_degree = as.integer(chosen_row$graph_degree),
      intermediate_graph_degree = as.integer(chosen_row$intermediate_graph_degree),
      search_width = as.integer(chosen_row$search_width),
      itopk_size = as.integer(chosen_row$itopk_size)
    )
  }
  tuning <- list(
    status = status,
    policy = policy,
    cache = "miss",
    sample_size = as.integer(sample_size),
    target_recall = as.numeric(target),
    chosen = chosen,
    results = results
  )
  out <- list(params = chosen, tuning = tuning)
  cuvs_cagra_set_cached_tuning(key, out, persist = !identical(policy, "pilot"))
  out
}

cuvs_nndescent_params <- function(n, k) {
  nn_tune_cuvs_nndescent_cpp(
    as.integer(n),
    as.integer(k),
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
  value <- suppressWarnings(as.integer(value))
  if (length(value) != 1L || is.na(value) || !is.finite(value)) value <- 50000L
  as.integer(max(2L, value))
}

cuvs_should_use_nndescent <- function(self_query, n) {
  isTRUE(self_query) && as.integer(n) >= cuvs_nndescent_threshold()
}

require_cuvs_backend <- function(label = "cuVS") {
  if (isTRUE(cuvs_available())) return(invisible(TRUE))
  info <- tryCatch(cuvs_info_json_cpp(), error = function(e) NA_character_)
  reason <- json_get_string(info, "reason")
  suffix <- if (!is.na(reason) && nzchar(reason)) paste0(" Reason: ", reason, ".") else ""
  stop(
    label,
    " backend is not available in this build or no CUDA device is visible.",
    suffix,
    " Reinstall faissR with `FAISSR_USE_CUDA=1 FAISSR_USE_CUVS=1` and ",
    "`CUVS_HOME` pointing to a RAPIDS cuVS installation.",
    call. = FALSE
  )
}

ivf_self_knn <- function(data,
                         k,
                         backend = "cpu_ivf",
                         seed = 4L,
                         n_threads = NULL) {
  n <- nrow(data)
  k <- as.integer(k)
  if (length(k) != 1L || is.na(k) || !is.finite(k) || k < 1L || k >= n) {
    stop("`k` must be in [1, nrow(data) - 1].", call. = FALSE)
  }
  n_threads <- normalize_nn_threads(n_threads)
  nlist <- faissr_option("ivf_nlist", NULL)
  nprobe <- faissr_option("ivf_nprobe", NULL)
  nlist <- if (is.null(nlist)) ivf_list_count(n, k) else as.integer(nlist)
  if (length(nlist) != 1L || is.na(nlist) || !is.finite(nlist)) {
    nlist <- ivf_list_count(n, k)
  }
  nlist <- max(1L, min(as.integer(n), nlist))
  nprobe <- if (is.null(nprobe)) ivf_probe_count(nlist, k) else as.integer(nprobe)
  if (length(nprobe) != 1L || is.na(nprobe) || !is.finite(nprobe)) {
    nprobe <- ivf_probe_count(nlist, k)
  }
  nprobe <- max(1L, min(nlist, nprobe))
  out <- ivf_self_knn_cpp(
    data,
    as.integer(k),
    as.integer(nlist),
    as.integer(nprobe),
    as.integer(seed),
    TRUE,
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

grid2d_self_knn <- function(data,
                            k,
                            exclude_self = TRUE,
                            n_threads = NULL) {
  n <- nrow(data)
  p <- ncol(data)
  if (p != 2L) {
    stop("`backend = \"cpu_grid2d\"` requires a two-column matrix.", call. = FALSE)
  }
  k <- as.integer(k)
  nonself_k <- if (isTRUE(exclude_self)) k else k - 1L
  if (length(nonself_k) != 1L || is.na(nonself_k) || !is.finite(nonself_k) ||
      nonself_k < 1L || nonself_k >= n) {
    stop("`k` must leave at least one non-self neighbour.", call. = FALSE)
  }
  n_threads <- normalize_nn_threads(n_threads)
  bins <- grid2d_bins_per_dim(n, nonself_k)
  out <- grid2d_self_knn_cpp(
    data,
    as.integer(nonself_k),
    TRUE,
    as.integer(n_threads),
    as.integer(bins)
  )
  if (!isTRUE(exclude_self)) {
    out$indices <- cbind(seq_len(n), out$indices)
    out$distances <- cbind(rep(0, n), out$distances)
  }
  attr(out, "spatial_index") <- list(
    strategy = "native_exact_uniform_grid_2d",
    backend = "cpu_grid2d",
    exact = TRUE,
    bins_per_dim = as.integer(out$bins_per_dim),
    n_cells = as.integer(out$n_cells),
    n_threads = as.integer(out$n_threads)
  )
  out
}

grid3d_self_knn <- function(data,
                            k,
                            exclude_self = TRUE,
                            n_threads = NULL) {
  n <- nrow(data)
  p <- ncol(data)
  if (p != 3L) {
    stop("`backend = \"cpu_grid3d\"` requires a three-column matrix.", call. = FALSE)
  }
  k <- as.integer(k)
  nonself_k <- if (isTRUE(exclude_self)) k else k - 1L
  if (length(nonself_k) != 1L || is.na(nonself_k) || !is.finite(nonself_k) ||
      nonself_k < 1L || nonself_k >= n) {
    stop("`k` must leave at least one non-self neighbour.", call. = FALSE)
  }
  n_threads <- normalize_nn_threads(n_threads)
  bins <- grid3d_bins_per_dim(n, nonself_k)
  out <- grid3d_self_knn_cpp(
    data,
    as.integer(nonself_k),
    TRUE,
    as.integer(n_threads),
    as.integer(bins)
  )
  if (!isTRUE(exclude_self)) {
    out$indices <- cbind(seq_len(n), out$indices)
    out$distances <- cbind(rep(0, n), out$distances)
  }
  attr(out, "spatial_index") <- list(
    strategy = "native_exact_uniform_grid_3d",
    backend = "cpu_grid3d",
    exact = TRUE,
    bins_per_dim = as.integer(out$bins_per_dim),
    n_cells = as.integer(out$n_cells),
    n_threads = as.integer(out$n_threads)
  )
  out
}

grid_self_knn <- function(data,
                          k,
                          backend = "cpu_grid",
                          exclude_self = TRUE,
                          n_threads = NULL) {
  p <- ncol(data)
  if (backend %in% c("grid", "cpu_grid")) {
    backend <- select_cpu_spatial_backend(data, k = k, exclude_self = exclude_self)
  }
  reason <- attr(backend, "reason", exact = TRUE)
  if (backend %in% c("grid3d", "cpu_grid3d") || identical(p, 3L)) {
    out <- grid3d_self_knn(data, k, exclude_self = exclude_self, n_threads = n_threads)
    if (!is.null(reason)) attr(out, "spatial_index")$reason <- reason
    return(out)
  }
  if (backend %in% c("grid2d", "cpu_grid2d", "grid", "cpu_grid") || identical(p, 2L)) {
    out <- grid2d_self_knn(data, k, exclude_self = exclude_self, n_threads = n_threads)
    if (!is.null(reason)) attr(out, "spatial_index")$reason <- reason
    return(out)
  }
  stop("`backend = \"cpu_grid\"` supports only two- or three-column matrices.", call. = FALSE)
}

#' Nearest neighbors from row-wise matrices
#'
#' `nn()` provides a package-native nearest-neighbor entry point compatible with
#' the common `nn(data, points, k)` use case. The public API separates device
#' selection from algorithm selection. `backend` is one of `"auto"`, `"cpu"`,
#' or `"cuda"`; `method` chooses the algorithm. For example,
#' `backend = "cpu", method = "grid"` uses the CPU grid implementation, while
#' `backend = "cuda", method = "grid"` uses the CUDA grid implementation.
#' Invalid combinations stop clearly before computation; for example,
#' `backend = "cpu", method = "cagra"` errors because CAGRA is CUDA-only.
#'
#' @details
#' Method descriptions:
#' \itemize{
#'   \item `"auto"`: shape-aware selector for the selected backend. CPU auto
#'   uses exact, grid, FAISS IVF, FAISS HNSW, or native CPU NN-descent fallback
#'   depending on data shape, size, and available libraries. CUDA auto uses CUDA
#'   grid for 2D/3D Euclidean/cosine/correlation
#'   self-KNN, exact FAISS GPU Flat/cuVS brute force or FAISS GPU CAGRA for
#'   Euclidean searches when appropriate, cuVS HNSW for explicit CUDA HNSW
#'   requests, FAISS GPU Flat IP routes for exact inner-product searches when
#'   FAISS GPU Flat is available, faissR native CUDA candidate refinement for
#'   large raw-inner-product self-KNN, and transformed direct cuVS brute force
#'   for explicit CUDA exact/brute-force non-Euclidean searches when cuVS is
#'   available
#'   [1-3,5,13-15,22-23]. When `backend = "auto"` is
#'   combined with an explicit method, faissR first checks whether that exact
#'   method/metric has a runtime-capable CUDA route; otherwise it uses the CPU
#'   route when that method/metric is supported on CPU.
#'   \item `"exact"`: exact nearest-neighbour search. CPU uses faissR's native
#'   exact route; CUDA uses FAISS GPU Flat when the linked FAISS build reports
#'   GPU support. CUDA exact search can otherwise use direct cuVS brute force
#'   when available: Euclidean uses cuVS L2 directly, cosine/correlation use
#'   normalized Euclidean search, and inner product uses an exact
#'   maximum-inner-product-to-L2 transform [1-3,16].
#'   \item `"flat"`: FAISS Flat exhaustive index. CPU and FAISS GPU support
#'   L2, IP, and normalized-IP cosine/correlation routes when available
#'   [1-2,16].
#'   \item `"bruteforce"`: exhaustive brute-force search. CPU maps to exact
#'   CPU search. On CUDA, RAPIDS cuVS brute force is preferred when available;
#'   cosine/correlation use normalized Euclidean search and inner product uses
#'   an exact maximum-inner-product-to-L2 transform around the cuVS L2 kernel
#'   [1-3,16].
#'   \item `"grid"`: native exact 2D/3D spatial grid search for Euclidean,
#'   cosine, and correlation self-KNN. Cosine/correlation use normalized
#'   Euclidean grid search. Explicit grid requests error for
#'   higher-dimensional matrices; use `"auto"` to let faissR choose a non-grid
#'   method when appropriate.
#'   \item `"hnsw"`: HNSW approximate graph-search index [3,5,16,22-23].
#'   CPU uses FAISS HNSW when available and RcppHNSW/hnswlib fallback
#'   otherwise. CUDA uses RAPIDS cuVS HNSW converted from a CUDA CAGRA index
#'   for Euclidean plus normalized cosine/correlation; raw inner product is
#'   disabled for CUDA HNSW until a reliable native or transformed route is
#'   available. Default parameters are selected by a deterministic
#'   shape/k/metric rule without pilot tuning; result approximation metadata
#'   records the selected `tuning_rule` and shape flags used.
#'   \item `"ivf"`: FAISS IVF-Flat inverted-file index, trading exhaustive
#'   search for coarse-list probing. It supports L2, raw IP, and normalized-IP
#'   cosine/correlation routes on CPU and FAISS GPU [1-2,16]. IVF records
#'   deterministic no-pilot `tuning_policy`, `tuning_rule`, shape/k flags,
#'   `tuning_metric`, and `tuning_metric_aware` in approximation metadata.
#'   \item `"ivfpq"`: FAISS inverted-file index with product quantization,
#'   mainly for compressed-memory approximate search. It supports L2, raw IP,
#'   and normalized-IP cosine/correlation routes on CPU and FAISS GPU [1-2,6,16].
#'   It reuses the metric-aware IVF probing defaults. IVF and PQ parameter
#'   selectors record deterministic tuning metadata; PQ fields use
#'   `pq_tuning_*` names. CPU IVFPQ requires at least 624 training rows; for
#'   624-9,983 rows, auto tuning uses 4-bit PQ rather than the 8-bit default.
#'   \item `"vamana"`: DiskANN/Vamana-style robust-pruned candidate graph
#'   implemented in faissR [24]. CPU refines top-k within candidate rows
#'   using native CPU scoring; CUDA refines candidates with faissR's native
#'   CUDA row-candidate kernel. Large high-dimensional CPU inputs use a
#'   deterministic FAISS HNSW seed before robust pruning; smaller inputs keep
#'   the exact seed. The first `k` seed neighbours are protected before robust
#'   pruning so pruning cannot discard neighbours already found by the seed
#'   generator. cuVS Vamana is acknowledged for GPU build/serialization, but
#'   current cuVS documentation does not expose KNN search for this index.
#'   \item `"nsg"`: Navigating Spreading-out Graph style approximate search
#'   [16,21,29]. CPU uses faissR's native NSG-style self-KNN candidate graph for
#'   all public metrics to avoid unsafe linked-FAISS graph construction. CUDA
#'   uses faissR's native NSG-style self-KNN candidate graph for all public
#'   metrics; cosine/correlation use normalized Euclidean search and raw inner
#'   product uses shifted dot-product distances. Large high-dimensional CPU
#'   inputs use a deterministic FAISS HNSW seed before NSG/MRNG-style pruning;
#'   smaller inputs keep the exact seed. The first `k` seed neighbours are
#'   protected before NSG/MRNG-style pruning.
#'   \item `"nndescent"`: NN-descent approximate graph construction via
#'   faissR's native CPU route, direct cuVS on CUDA for Euclidean/L2 plus
#'   normalized cosine/correlation, or faissR's native CUDA candidate-refinement
#'   route for raw inner-product self-KNN. FAISS NNDescent is experimental
#'   opt-in because linked FAISS builds can abort during graph construction
#'   [3-4,16].
#'   \item `"cagra"`: CUDA-only graph-search method via FAISS GPU CAGRA/cuVS
#'   integration or direct RAPIDS cuVS CAGRA. By default faissR chooses FAISS
#'   GPU CAGRA when that route is available and otherwise direct cuVS CAGRA;
#'   set `options(faissR.cagra_implementation = "faiss_gpu")` or `"cuvs"` to
#'   force one provider for benchmarking. Availability preflights respect this
#'   forced provider for supported metrics, and approximation metadata records
#'   `cagra_provider` plus `cagra_provider_option`. It supports Euclidean/L2,
#'   and cosine/correlation through normalized Euclidean graph search. Raw
#'   inner-product CAGRA is disabled for both FAISS GPU CAGRA and direct cuVS
#'   CAGRA until the transformed route is reliable across k values [3,13-16].
#' }
#'
#' References are numbered as in `docs/references.md` in the GitHub
#' repository.
#'
#' @references
#' Johnson J, Douze M, Jegou H. Billion-scale similarity search with GPUs. IEEE
#' Transactions on Big Data. 2021;7:535-547.
#'
#' Douze M, Guzhva A, Deng C, Johnson J, Szilvasy G, Mazaré PE, et al. The
#' FAISS library. arXiv 2024. See also the FAISS C++ API documentation.
#'
#' RAPIDS Development Team. RAPIDS cuVS: GPU-accelerated vector search and
#' clustering. https://github.com/rapidsai/cuvs.
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
#'   object of reference observations in rows. Float32 inputs use the CPU FAISS
#'   Flat float32 route for public
#'   `method = "auto"`, `"exact"`, `"bruteforce"`, or `"flat"` requests and
#'   can be paired with ordinary R double query matrices.
#' @param points Numeric matrix/data frame or optional `float::fl()`/`float32`
#'   query object with observations in rows. Defaults to `data`. A float32
#'   query can be paired with an ordinary R double
#'   reference matrix on the CPU FAISS Flat float32 route.
#' @param k Number of neighbors to return. `NULL` chooses the package's
#'   automatic neighborhood size and includes the self-neighbor when `points`
#'   is `data`.
#' @param backend Device backend: `"auto"`, `"cpu"`, or `"cuda"`. `"auto"`
#'   uses a validated CUDA route only when the requested method/metric
#'   combination is supported and CUDA/cuVS runtime support is available, and
#'   otherwise resolves to CPU. Explicit `"cuda"` fails clearly when CUDA
#'   support or the selected CUDA combination is unavailable.
#' @param method Algorithm selector. `"auto"` chooses a shape-aware default for
#'   the selected backend. Other values include `"exact"`, `"flat"`,
#'   `"bruteforce"`, `"grid"`, `"hnsw"`, `"ivf"`,
#'   `"ivfpq"`, `"vamana"`, `"nsg"`, `"nndescent"`, and `"cagra"`. Use these canonical
#'   lowercase method labels; resolved implementation labels such as
#'   `"faiss_hnsw"` are not public `method` values. Unsupported
#'   backend/method combinations fail clearly; for example,
#'   `method = "cagra", backend = "cpu"` errors because CAGRA is CUDA-only.
#' @param metric Distance metric. The intentionally small public set is
#'   `"euclidean"`, `"cosine"`, `"correlation"`, and `"inner_product"`;
#'   aliases such as `"l2"`, `"cor"`/`"pearson"`, and `"ip"` are accepted and
#'   stored as canonical metric labels.
#'   `"inner_product"` is the raw dot product, `"cosine"` is the dot product
#'   after row L2 normalization, and `"correlation"` is centered cosine
#'   similarity after subtracting each row mean and L2-normalizing each row.
#'   For `metric = "inner_product"`, neighbours are ranked by larger raw dot
#'   product, but returned `distances` keep faissR's smaller-is-better
#'   convention: within each query row the best returned dot product has
#'   distance `0`, and lower dot products have larger shifted distances.
#'   `"euclidean"` is the validated high-performance default. `"cosine"` and
#'   `"correlation"` are implemented for exact CPU KNN, native 2D/3D grid
#'   search, FAISS CPU/GPU Flat,
#'   FAISS CPU/GPU IVF-Flat, FAISS CPU/GPU IVFPQ, FAISS CPU HNSW,
#'   and RcppHNSW. FAISS approximate IP-capable routes use row L2 normalization
#'   for cosine and row centering plus L2 normalization for correlation before
#'   inner-product search; distances are returned as `1 - similarity`.
#'   All-zero cosine rows and constant correlation rows are zero-normalized
#'   edge cases: two zero-normalized rows have distance `0`, while a
#'   zero-normalized row versus a nonzero row has distance `1`. CPU FAISS Flat
#'   uses the exact CPU scorer for those rows to preserve deterministic
#'   small-`k` tie handling; explicit CUDA routes remain on CUDA.
#'   CPU `method = "auto"` can use FAISS Flat for larger exact non-Euclidean
#'   query workloads, FAISS HNSW for large non-Euclidean self-search when FAISS
#'   is available, and RcppHNSW/hnswlib only as the fallback when FAISS is
#'   unavailable. CPU `method = "hnsw"` uses FAISS HNSW for all metrics when
#'   available and RcppHNSW/hnswlib when FAISS is unavailable.
#'   `"inner_product"` is exact on native CPU routes and maps to FAISS Flat IP,
#'   FAISS IVF-Flat/IVFPQ IP, FAISS HNSW IP, native CPU NN-descent raw
#'   dot-product search, direct cuVS brute force through an exact MIPS-to-L2
#'   transform, direct cuVS IVF/PQ through transformed approximate L2 indexes,
#'   and native CUDA NN-descent candidate refinement.
#'   Direct cuVS NN-descent, CUDA cuVS HNSW, FAISS GPU CAGRA, and direct cuVS
#'   CAGRA do not expose a safe raw-inner-product route in faissR.
#'   Unsupported backend combinations fail clearly instead of returning neighbours
#'   computed under a different metric.
#' @param tuning Tuning policy for approximate methods. `"auto"` uses
#'   deterministic no-pilot defaults for the resolved method, `"cache"`
#'   reuses/stores pilot results, `"pilot"` tunes for this call without
#'   persisting, `"fixed"` uses fixed defaults with tuning metadata, and
#'   `"off"`/`"none"` disables tuning.
#'   FAISS CPU HNSW uses deterministic no-pilot defaults based on `n`, `p`,
#'   `k`, and `metric`, including separate small-`k` Euclidean,
#'   small-`k` metric-aware, balanced, and high-recall tiers; explicit
#'   `faissR.faiss_hnsw_*` options override those defaults. FAISS IVF/IVFPQ
#'   use deterministic shape/k/metric-aware `nlist`/`nprobe` defaults; optional
#'   FAISS GPU IVF `"cache"`/`"pilot"` tuning currently runs only for Euclidean
#'   IVF, while non-Euclidean IVF routes use deterministic metric-aware
#'   defaults. Deterministic approximate-method defaults are computed by C++
#'   `nn_tune_*_cpp()` helpers and record `tuning_source = "cpp"` in
#'   approximation metadata. Advanced tuning and cache knobs use
#'   `options(faissR.<name> = ...)`.
#' @param cagra_implementation CUDA CAGRA provider for this call. `NULL` uses
#'   the global `options(faissR.cagra_implementation = ...)` value. `"auto"`
#'   uses a deterministic provider rule: compact high-dimensional self-KNN
#'   selects direct RAPIDS cuVS CAGRA when both providers are available, while
#'   FAISS GPU CAGRA remains the default for other shapes. `"faiss_gpu"` and
#'   `"cuvs"` force one provider for benchmarking.
#'   This argument affects only public `backend = "cuda", method = "cagra"`
#'   requests and CUDA-auto routes that select CAGRA.
#' @param cagra_build_algo Direct RAPIDS cuVS CAGRA graph-build algorithm for
#'   this call. `NULL` uses `options(faissR.cuvs_cagra_build_algo = "auto")`.
#'   `"auto"` applies faissR's deterministic shape-aware CAGRA build rule,
#'   choosing iterative CAGRA construction for compact high-dimensional
#'   self-KNN cases and IVF-PQ construction otherwise. `"ivf_pq"` requests the
#'   IVF-PQ graph builder, `"nn_descent"` requests cuVS NN-descent graph
#'   construction, and `"iterative_cagra_search"` requests cuVS iterative CAGRA
#'   graph building.
#'   This is a CAGRA construction parameter, not a fallback to a different
#'   public method; successful results record the selected value in
#'   `attr(result, "approximation")$cagra_build_algo`.
#' @param output Distance storage type for the returned object. `"double"`
#'   returns the default R numeric distance matrix. `"float"` returns
#'   `distances` as a `float::fl()`/`float32` object and records
#'   `distance_type = "float32"` plus
#'   `attr(result, "distance_type") = "float32"`; this requires the optional
#'   `float` package. When either `data` or `points` is a `float::fl()` matrix,
#'   the current float32 input route uses CPU FAISS Flat for public
#'   `method = "auto"`, `"exact"`, `"bruteforce"`, or `"flat"` requests, with
#'   ordinary R double inputs converted once to float32 internally. Ordinary R
#'   double inputs with a CPU FAISS Flat-style request and `output = "float"`
#'   also use this float-pointer FAISS route. On that route, float distance
#'   output is constructed directly from FAISS float results instead of first
#'   materializing an R double distance matrix, except for zero-row
#'   cosine/correlation correction.
#' @param distances Optional alias for `output`, kept for callers that prefer
#'   `distances = "double"` or `distances = "float"` to describe the returned
#'   distance storage type.
#' @param n_threads Number of CPU worker threads for CPU backends. GPU backends
#'   ignore this argument.
#' @return A list with integer matrix `indices`, `distances`, and stable
#'   metadata fields `index_base`, `distance_type`, `metric`, and
#'   `backend_used`. Float32 routes also record `input_layout` and
#'   `input_owns_data` so downstream packages can distinguish direct float32
#'   payload use from one-time row-major conversion. Normalized Euclidean graph
#'   routes for cosine/correlation record `metric_transform` and
#'   `attr(result, "distance_transform")`. Indices are 1-based. The
#'   requested backend/method, tuning policy, resolved
#'   backend, metric, exact/approximate flag, and self-query flag are stored in
#'   attributes including `attr(result, "requested_backend")`,
#'   `attr(result, "requested_method")`, `attr(result, "tuning")`, and
#'   `attr(result, "resolved_backend")`. Auto requests also include
#'   `attr(result, "auto_selection")`, a static shape/k/metric decision record
#'   that records the predicted internal backend, public method class, device
#'   class, explicit backend/method flags, backend/method decision reasons, and
#'   does not run pilot tuning.
#' @examples
#' x <- scale(as.matrix(iris[, 1:4]))
#' knn_euclidean <- nn(x, k = 16, metric = "euclidean", backend = "cpu")
#' knn_cosine <- nn(x, k = 16, metric = "cosine", backend = "cpu")
#' knn_correlation <- nn(x, k = 16, metric = "correlation", backend = "cpu")
#' knn_ip <- nn(x, k = 16, metric = "inner_product", backend = "cpu")
#' @export
nn <- function(data,
               points = data,
               k = NULL,
               backend = c("auto", "cpu", "cuda"),
               method = c("auto", "exact", "flat", "bruteforce", "grid", "hnsw", "ivf", "ivfpq", "vamana", "nsg", "nndescent", "cagra"),
               metric = c("euclidean", "cosine", "correlation", "inner_product"),
               tuning = c("auto", "cache", "pilot", "fixed", "off", "none"),
               cagra_implementation = NULL,
               cagra_build_algo = NULL,
               output = c("double", "float"),
               distances = NULL,
               n_threads = NULL) {
  set_call_cagra_implementation(cagra_implementation)
  set_call_cagra_build_algo(cagra_build_algo)
  points_missing <- missing(points)
  backend <- normalize_public_backend_arg(backend)
  method <- normalize_nn_method(method)
  tuning <- normalize_nn_tuning(tuning)
  metric <- normalize_nn_metric(metric)
  output <- resolve_nn_output(output, distances)
  validate_public_nn_method_shape(data, method)
  resolved_backend <- resolve_public_nn_backend(
    backend,
    method,
    metric,
    n = nrow(data),
    p = ncol(data),
    k = k,
    self_query = points_missing
  )
  auto_selection <- nn_auto_selection_metadata(
    data = data,
    points = points,
    points_missing = points_missing,
    k = k,
    requested_backend = backend,
    requested_method = method,
    resolved_backend = resolved_backend,
    metric = metric,
    tuning = tuning,
    exclude_self = FALSE
  )
  result <- nn_compute(
    data,
    points,
    k,
    resolved_backend,
    points_missing,
    exclude_self = FALSE,
    n_threads = n_threads,
    metric = metric,
    tuning = tuning,
    output = output,
    auto_selection = auto_selection
  )
  attr(result, "requested_backend") <- backend
  attr(result, "requested_method") <- public_nn_method_label(method)
  attr(result, "tuning") <- tuning
  if (!is.null(auto_selection)) attr(result, "auto_selection") <- auto_selection
  finalize_nn_output(result, output)
}

#' Nearest neighbours excluding the self match
#'
#' `nn_without_self()` is a convenience wrapper around `nn()` for the common
#' case where the reference and query data are the same matrix and the
#' self-neighbour should be removed. It returns exactly `k` non-self neighbours
#' per observation.
#'
#' @param data Numeric matrix/data frame with
#'   observations in rows.
#' @param k Number of non-self neighbours to return.
#' @param backend Device backend: `"auto"`, `"cpu"`, or `"cuda"`. `"auto"`
#'   follows \code{\link{nn}()} backend/method/metric resolution, using CUDA
#'   only for validated CUDA combinations when CUDA/cuVS runtime support is
#'   available, and CPU otherwise.
#' @param method Algorithm selector passed through the same resolver as
#'   \code{\link{nn}()}. See \code{\link{nn}()} for method descriptions and
#'   references.
#' @param metric Distance metric: `"euclidean"`, `"cosine"`, `"correlation"`,
#'   or `"inner_product"`; aliases such as `"l2"`, `"cor"`/`"pearson"`, and
#'   `"ip"` are accepted. Correlation is centered cosine similarity, not raw
#'   inner product; see \code{\link{nn}()} for metric/backend support details,
#'   including metric-aware CPU HNSW routing.
#' @param tuning Tuning policy passed to \code{\link{nn}()}. `"auto"` uses the
#'   deterministic no-pilot default for the resolved method. FAISS IVF/IVFPQ
#'   routes use metric-aware defaults; optional FAISS GPU IVF pilot/cache tuning
#'   is Euclidean-only. Deterministic approximate-method defaults are computed
#'   by C++ `nn_tune_*_cpp()` helpers.
#' @param cagra_implementation CUDA CAGRA provider for this call. See
#'   \code{\link{nn}()}.
#' @param cagra_build_algo Direct RAPIDS cuVS CAGRA graph-build algorithm for
#'   this call. See \code{\link{nn}()}.
#' @param output Distance storage type: `"double"` for the default R numeric
#'   matrix or `"float"` for a `float::fl()`/`float32` distance matrix when the
#'   optional `float` package is installed. `float::fl()` input matrices use
#'   the same CPU FAISS Flat float32 route described in \code{\link{nn}()}.
#' @param distances Optional alias for `output`.
#' @param n_threads Number of CPU worker threads used by CPU backends.
#' @return A `faissR_nn` object with `indices`, `distances`, and stable
#'   metadata fields `index_base`, `distance_type`, `metric`, and
#'   `backend_used`. Float32 routes also record `input_layout` and
#'   `input_owns_data`. Normalized Euclidean graph routes for cosine/correlation
#'   record `metric_transform` and `attr(result, "distance_transform")`. Auto
#'   requests also include `attr(result, "auto_selection")`, a static
#'   shape/k/metric decision record that records the predicted internal backend,
#'   public method class, device class, explicit backend/method flags,
#'   backend/method decision reasons, and does not run pilot tuning.
#' @export
nn_without_self <- function(data,
                            k,
                            backend = c("auto", "cpu", "cuda"),
                            method = c("auto", "exact", "flat", "bruteforce", "grid", "hnsw", "ivf", "ivfpq", "vamana", "nsg", "nndescent", "cagra"),
                            metric = c("euclidean", "cosine", "correlation", "inner_product"),
                            tuning = c("auto", "cache", "pilot", "fixed", "off", "none"),
                            cagra_implementation = NULL,
                            cagra_build_algo = NULL,
                            output = c("double", "float"),
                            distances = NULL,
                            n_threads = NULL) {
  set_call_cagra_implementation(cagra_implementation)
  set_call_cagra_build_algo(cagra_build_algo)
  backend <- normalize_public_backend_arg(backend)
  method <- normalize_nn_method(method)
  tuning <- normalize_nn_tuning(tuning)
  metric <- normalize_nn_metric(metric)
  output <- resolve_nn_output(output, distances)
  validate_public_nn_method_shape(data, method)
  resolved_backend <- resolve_public_nn_backend(
    backend,
    method,
    metric,
    n = nrow(data),
    p = ncol(data),
    k = k,
    self_query = TRUE
  )
  auto_selection <- nn_auto_selection_metadata(
    data = data,
    points = data,
    points_missing = TRUE,
    k = k,
    requested_backend = backend,
    requested_method = method,
    resolved_backend = resolved_backend,
    metric = metric,
    tuning = tuning,
    exclude_self = TRUE
  )
  result <- nn_compute(
    data,
    data,
    k,
    resolved_backend,
    TRUE,
    exclude_self = TRUE,
    n_threads = n_threads,
    metric = metric,
    tuning = tuning,
    output = output,
    auto_selection = auto_selection
  )
  attr(result, "requested_backend") <- backend
  attr(result, "requested_method") <- public_nn_method_label(method)
  attr(result, "tuning") <- tuning
  if (!is.null(auto_selection)) attr(result, "auto_selection") <- auto_selection
  finalize_nn_output(result, output)
}

.knn_recall_summary <- function(approx, exact, k = NULL) {
  approx_idx <- if (is.list(approx)) approx$indices else approx
  exact_idx <- if (is.list(exact)) exact$indices else exact
  approx_idx <- as.matrix(approx_idx)
  exact_idx <- as.matrix(exact_idx)
  if (nrow(approx_idx) != nrow(exact_idx)) {
    stop("Approximate and exact KNN must have the same number of rows.", call. = FALSE)
  }
  k_is_auto <- is.null(k)
  k <- if (k_is_auto) {
    min(ncol(approx_idx), ncol(exact_idx))
  } else {
    normalize_nn_positive_integer(k, "k", "`k` must be a positive integer.")
  }
  k <- min(k, ncol(approx_idx), ncol(exact_idx))
  if (k < 1L) {
    stop("KNN matrices must have at least one neighbour column.", call. = FALSE)
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
  if (!is.matrix(indices)) indices <- as.matrix(indices)
  if (!is.matrix(distances)) distances <- as.matrix(distances)
  if (!is.integer(indices)) storage.mode(indices) <- "integer"
  if (!identical(typeof(distances), "double")) storage.mode(distances) <- "double"
  keep <- matrix(1L, nrow(indices), k)
  keep_dist <- matrix(0, nrow(indices), k)
  for (i in seq_len(nrow(indices))) {
    row_keep <- which(indices[i, ] != i | distances[i, ] > sqrt(.Machine$double.eps))
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
  if (!isTRUE(attr(x, "exact"))) {
    cat("  exact: false\n")
    recall <- attr(x, "recall")
    if (is.data.frame(recall) && nrow(recall) >= 1L && is.finite(recall$recall_at_k[1L])) {
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
  if (isTRUE(attr(x, "self_query"))) {
    cat("  first column: self-neighbor\n")
  }
  invisible(x)
}

#' Check whether the native CUDA backend is available
#'
#' @return `TRUE` when the package was built with CUDA support and the CUDA
#'   runtime reports at least one available device.
#' @export
cuda_available <- function() {
  isTRUE(cuda_available_cpp())
}

#' Check whether the real FAISS C++ backend is available
#'
#' @return `TRUE` when faissR was compiled and linked against FAISS.
#' @export
faiss_available <- function() {
  isTRUE(faiss_available_cpp())
}

#' Check whether the RAPIDS cuVS backend is available
#'
#' @return `TRUE` when faissR was compiled and linked against RAPIDS cuVS
#'   and the CUDA runtime reports at least one available device.
#' @export
cuvs_available <- function() {
  isTRUE(cuvs_available_cpp())
}

#' Check whether the RAPIDS libcugraph backend is available
#'
#' @return `TRUE` when faissR was compiled and linked against RAPIDS libcugraph.
#' @export
cugraph_available <- function() {
  isTRUE(cugraph_available_cpp())
}
