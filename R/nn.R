nn_compute <- function(data,
                       points,
                       k,
                       backend,
                       points_missing,
                       exclude_self = FALSE,
                       n_threads = NULL,
                       metric = "euclidean",
                       tuning = "auto") {
  requested_backend <- backend
  tuning <- normalize_nn_tuning(tuning)
  data_sparse <- is_sparse_matrix_input(data)
  points_sparse <- if (isTRUE(points_missing)) data_sparse else is_sparse_matrix_input(points)
  if (isTRUE(data_sparse) || isTRUE(points_sparse)) {
    sparse_native <- backend %in% c("auto", "cpu", "cpu_auto", "sparse", "cpu_sparse", "sparse_cpu")
    if (isTRUE(sparse_native)) {
      return(nn_sparse_compute(
        data = data,
        points = points,
        k = k,
        backend = backend,
        points_missing = points_missing,
        exclude_self = exclude_self,
        metric = metric
      ))
    }
    warning(
      "Sparse input is being converted to a dense matrix for backend `",
      backend,
      "`. Use `backend = \"cpu_sparse\"` to keep the exact sparse CPU path.",
      call. = FALSE
    )
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
  k <- as.integer(k)
  if (length(k) != 1L || is.na(k) || !is.finite(k) || k < 1L) {
    stop("`k` must be NULL or a positive integer.", call. = FALSE)
  }
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
  if (backend %in% c("sparse", "cpu_sparse", "sparse_cpu")) {
    return(nn_sparse_compute(
      data = data,
      points = points,
      k = k,
      backend = backend,
      points_missing = points_missing,
      exclude_self = exclude_self,
      metric = metric
    ))
  }
  if (!identical(metric, "euclidean")) {
    if (identical(backend, "auto")) {
      backend <- "cpu"
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
               backend %in% c("vptree", "cpu_vptree")) {
      backend <- "cpu_vptree"
    } else if (identical(metric, "inner_product") &&
               backend %in% c("vptree", "cpu_vptree")) {
      backend <- "cpu_vptree"
    } else if (!backend %in% c("cpu", "cpu_auto", "hnsw", "rcpphnsw", "cpu_hnsw",
                               "faiss_hnsw", "faiss_ivf", "faiss_ivf_flat",
                               "faiss_ivfpq", "faiss_nsg", "faiss_nndescent",
                               "cpu_nndescent",
                               "cpu_faiss_index_ivf", "faiss_gpu_ivf",
                               "faiss_gpu_ivf_flat", "cuda_faiss_ivf_flat",
                               "faiss_gpu_ivfpq", "cuda_faiss_ivfpq",
                               "faiss_gpu_cagra", "cuda_faiss_cagra",
                               "cuda_cuvs_cagra", "cuda_cagra", "gpu_cagra",
                               "cuda_cuvs_nndescent", "cuvs_nndescent",
                               "cpu_sparse", "sparse", "sparse_cpu")) {
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

  if (identical(backend, "cpu_auto")) {
    backend <- select_cpu_auto_backend(
      self_query = self_query,
      n = nrow(data),
      p = ncol(data),
      n_points = nrow(points),
      k = k,
      work_size = work_size,
      metric = metric
    )
  } else if (backend %in% c("cuda_auto", "gpu_auto")) {
    backend <- select_cuda_auto_backend(
      self_query = self_query,
      n = nrow(data),
      p = ncol(data),
      n_points = nrow(points),
      k = k,
      work_size = work_size,
      metric = metric
    )
  } else {
    auto_gpu <- resolve_auto_knn_gpu_backend(
      backend = backend,
      self_query = self_query,
      n_points = nrow(points),
      n = nrow(data),
      p = ncol(data),
      k = k,
      work_size = work_size,
      metric = metric
    )
    if (!is.na(auto_gpu)) {
      backend <- auto_gpu
    } else if (identical(backend, "auto") &&
               should_use_grid2d_self_knn(
                 self_query = self_query,
                 n = nrow(data),
                 p = ncol(data),
                 k = k,
                 exclude_self = isTRUE(exclude_self),
                 metric = metric
               )) {
      backend <- "cpu_grid"
    } else if (identical(backend, "auto") &&
               should_use_auto_cpu_approx_self_knn(
                 self_query = self_query,
                 n = nrow(data),
                 p = ncol(data),
                 k = k,
                 work_size = work_size
               )) {
      backend <- select_cpu_approx_backend(nrow(data), ncol(data), k)
    } else if (identical(backend, "cpu_approx")) {
      if (!isTRUE(self_query)) {
        stop("`backend = \"cpu_approx\"` is only available for self-KNN searches.", call. = FALSE)
      }
      backend <- select_cpu_approx_backend(nrow(data), ncol(data), k)
    }
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
      seed = fast_knn_approx_seed()
    ))
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
    if (!isTRUE(faiss_available())) {
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
    if (!isTRUE(faiss_available())) {
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
    if (!isTRUE(faiss_available())) {
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
    params <- faiss_ivf_params(nrow(data), k)
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
      if (identical(metric, "inner_product")) "inner_product" else "euclidean",
      if (identical(metric, "inner_product")) "inner_product" else "euclidean",
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
    params <- faiss_ivf_params(nrow(data), k)
    pq <- faiss_pq_params(ncol(data))
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
      if (identical(metric, "inner_product")) "inner_product" else "euclidean",
      if (identical(metric, "inner_product")) "inner_product" else "euclidean",
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
    return(result)
  }

  if (backend %in% c("faiss_gpu_ivf", "faiss_gpu_ivf_flat", "cuda_faiss_ivf_flat")) {
    if (!isTRUE(faiss_available())) {
      stop(
        "The real FAISS C++ GPU IVF Flat backend is not available in this build. ",
        "Reinstall faissR with FAISS GPU/cuVS headers ",
        "available through `FAISS_HOME`.",
        call. = FALSE
      )
    }
    params <- faiss_ivf_params(nrow(data), k)
    tuning_metadata <- NULL
    if (isTRUE(faiss_gpu_ivf_should_tune(data, k, self_query, tuning = tuning))) {
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
      if (identical(metric, "inner_product")) "inner_product" else "euclidean",
      if (identical(metric, "inner_product")) "inner_product" else "euclidean",
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
    return(result)
  }

  if (backend %in% c("faiss_gpu_ivfpq", "cuda_faiss_ivfpq")) {
    if (!isTRUE(faiss_available())) {
      stop(
        "The real FAISS C++ GPU IVF-PQ backend is not available in this build. ",
        "Reinstall faissR with FAISS GPU/cuVS headers ",
        "available through `FAISS_HOME`.",
        call. = FALSE
      )
    }
    params <- faiss_ivf_params(nrow(data), k)
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
      if (identical(metric, "inner_product")) "inner_product" else "euclidean",
      if (identical(metric, "inner_product")) "inner_product" else "euclidean",
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
    return(result)
  }

  if (backend %in% c("faiss_gpu_cagra", "cuda_faiss_cagra")) {
    if (!isTRUE(faiss_available())) {
      stop(
        "The real FAISS GPU CAGRA backend is not available in this build. ",
        "Reinstall faissR with FAISS GPU/cuVS headers ",
        "available through `FAISS_HOME`.",
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
      stop("FAISS GPU CAGRA does not support `metric = \"inner_product\"`.", call. = FALSE)
    }
    params <- cuvs_cagra_params(nrow(data), k)
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
      result <- finalize_normalized_euclidean_metric_result(result, metric_inputs)
    }
    attr(result, "approximation") <- list(
      strategy = "faiss_gpu_GpuIndexCagra_cuVS",
      backend = "faiss_gpu_cagra",
      library = "faiss",
      accelerator = "cuda",
      metric = metric,
      transform = if (is.null(metric_inputs)) NA_character_ else metric_inputs$transform,
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
    params <- faiss_hnsw_params(k)
    out <- nn_faiss_hnsw_cpp(
      data,
      points,
      as.integer(k),
      as.integer(params$m),
      as.integer(params$ef_construction),
      as.integer(params$ef_search),
      if (identical(metric, "inner_product")) "inner_product" else "euclidean",
      if (identical(metric, "inner_product")) "inner_product" else "euclidean",
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
      hnsw_parameters_adjusted = isTRUE(out$hnsw_parameters_adjusted)
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
    return(result)
  }

  if (identical(backend, "faiss_nndescent")) {
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
    if (!identical(metric, "euclidean")) {
      stop(
        "`backend = \"faiss_nndescent\"` is currently validated only for ",
        "`metric = \"euclidean\"` in this FAISS build.",
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
    metric_inputs <- NULL
    search_data <- data
    search_points <- points
    if (metric %in% c("cosine", "correlation")) {
      metric_inputs <- normalized_euclidean_metric_inputs(data, points, self_query, metric)
      search_data <- metric_inputs$data
      search_points <- metric_inputs$points
    } else if (identical(metric, "inner_product")) {
      stop("cuVS CAGRA does not support `metric = \"inner_product\"`.", call. = FALSE)
    }
    params <- cuvs_cagra_params(nrow(data), k)
    tuning_metadata <- NULL
    if (isTRUE(cuvs_cagra_should_tune(search_data, k, self_query, tuning = tuning))) {
      tuned <- cuvs_cagra_tune_params(search_data, k, params, tuning = tuning)
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
      as.integer(params$itopk_size)
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
      result <- finalize_normalized_euclidean_metric_result(result, metric_inputs)
    }
    attr(result, "approximation") <- list(
      strategy = "rapids_cuvs_cagra",
      backend = resolved_backend,
      library = "cuvs",
      metric = metric,
      transform = if (is.null(metric_inputs)) NA_character_ else metric_inputs$transform,
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
        !identical(as.integer(requested_itopk_size), as.integer(out$itopk_size)),
      search_batch_size = as.integer(out$search_batch_size),
      tuning = tuning_metadata
    )
    return(result)
  }

  if (backend %in% c("cuvs_ivf_flat", "cuda_cuvs_ivf_flat")) {
    require_cuvs_backend("cuVS IVF-Flat")
    params <- faiss_ivf_params(nrow(data), k)
    out <- nn_cuvs_ivf_flat_cpp(
      data,
      points,
      as.integer(k),
      as.integer(params$nlist),
      as.integer(params$nprobe),
      isTRUE(exclude_self)
    )
    result <- finish_nn_result(out, "cuda_cuvs_ivf_flat", k, self_query, exact = FALSE)
    attr(result, "approximation") <- list(
      strategy = "rapids_cuvs_ivf_flat",
      backend = "cuda_cuvs_ivf_flat",
      library = "cuvs",
      accelerator = "cuda",
      default_candidate = FALSE,
      nlist = as.integer(out$n_lists),
      nprobe = as.integer(out$n_probes),
      requested_nlist = as.integer(params$requested_nlist),
      requested_nprobe = as.integer(params$requested_nprobe),
      ivf_parameters_adjusted = !identical(as.integer(params$requested_nlist), as.integer(out$n_lists)) ||
        !identical(as.integer(params$requested_nprobe), as.integer(out$n_probes)),
      search_batch_size = as.integer(out$search_batch_size)
    )
    return(result)
  }

  if (backend %in% c("cuvs_ivfpq", "cuda_cuvs_ivfpq", "cuvs_ivf_pq", "cuda_cuvs_ivf_pq")) {
    require_cuvs_backend("cuVS IVF-PQ")
    params <- faiss_ivf_params(nrow(data), k)
    pq <- cuvs_ivfpq_params(ncol(data))
    out <- nn_cuvs_ivf_pq_cpp(
      data,
      points,
      as.integer(k),
      as.integer(params$nlist),
      as.integer(params$nprobe),
      as.integer(pq$pq_dim),
      as.integer(pq$pq_bits),
      isTRUE(exclude_self)
    )
    result <- finish_nn_result(out, "cuda_cuvs_ivfpq", k, self_query, exact = FALSE)
    attr(result, "approximation") <- list(
      strategy = "rapids_cuvs_ivf_pq",
      backend = "cuda_cuvs_ivfpq",
      library = "cuvs",
      accelerator = "cuda",
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
    return(result)
  }

  if (backend %in% c("cuvs_bruteforce", "cuda_cuvs_bruteforce", "cuda_cuvs_exact")) {
    require_cuvs_backend("cuVS brute-force")
    out <- nn_cuvs_bruteforce_cpp(
      data,
      points,
      as.integer(k),
      isTRUE(exclude_self)
    )
    resolved_backend <- "cuda_cuvs_bruteforce"
    result_backend <- if (requested_backend %in% c("cuda", "gpu")) requested_backend else resolved_backend
    result <- finish_nn_result(out, result_backend, k, self_query, exact = TRUE)
    if (!identical(result_backend, resolved_backend)) {
      attr(result, "resolved_backend") <- resolved_backend
    }
    attr(result, "cuvs") <- list(
      index_type = as.character(out$index_type),
      library = "cuvs",
      backend = "cuda",
      resolved_backend = resolved_backend
    )
    return(result)
  }

  if (backend %in% c("cuvs_nndescent", "cuda_cuvs_nndescent")) {
    if (!identical(metric, "euclidean")) {
      stop("cuVS NN-descent is currently validated only for `metric = \"euclidean\"`.", call. = FALSE)
    }
    require_cuvs_backend("cuVS NN-descent")
    if (!isTRUE(self_query)) {
      stop("`backend = \"cuda_cuvs_nndescent\"` is only available for self-KNN searches.", call. = FALSE)
    }
    nonself_k <- if (isTRUE(exclude_self)) k else max(0L, k - 1L)
    if (nonself_k < 1L) {
      out <- list(
        indices = matrix(seq_len(nrow(data)), nrow(data), 1L),
        distances = matrix(0, nrow(data), 1L)
      )
    } else {
      params <- cuvs_nndescent_params(nrow(data), nonself_k)
      out <- nn_cuvs_nndescent_self_cpp(
        data,
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
    attr(result, "approximation") <- list(
      strategy = "rapids_cuvs_nndescent",
      backend = "cuda_cuvs_nndescent",
      library = "cuvs",
      metric = metric,
      graph_degree = as.integer(out$graph_degree),
      intermediate_graph_degree = as.integer(out$intermediate_graph_degree),
      max_iterations = as.integer(out$max_iterations)
    )
    return(result)
  }

  if (identical(backend, "cpu_nndescent")) {
    if (!identical(metric, "euclidean")) {
      stop("CPU NN-descent is currently validated only for `metric = \"euclidean\"`.", call. = FALSE)
    }
    if (!isTRUE(self_query)) {
      stop("`method = \"nndescent\"` is only available for self-KNN searches on CPU.", call. = FALSE)
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
        data,
        k = nonself_k,
        seed = fast_knn_approx_seed(),
        n_threads = n_threads
      )
      if (!isTRUE(exclude_self)) {
        out$indices <- cbind(seq_len(nrow(data)), out$indices)
        out$distances <- cbind(rep(0, nrow(data)), out$distances)
      }
    }
    result <- finish_nn_result(out, "cpu_nndescent", k, self_query, exact = FALSE, metric = metric)
    attr(result, "approximation") <- attr(out, "approximation")
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

  if (backend %in% c("vptree", "cpu_vptree")) {
    if (identical(metric, "inner_product")) {
      stop("`backend = \"cpu_vptree\"` does not support inner-product search.", call. = FALSE)
    }
    tree_data <- data
    tree_points <- points
    data_zero <- NULL
    points_zero <- NULL
    if (identical(metric, "cosine")) {
      tree_data <- row_l2_normalize(data)
      tree_points <- if (isTRUE(self_query)) tree_data else row_l2_normalize(points)
      data_zero <- rowSums(tree_data * tree_data) <= 0
      points_zero <- if (isTRUE(self_query)) data_zero else rowSums(tree_points * tree_points) <= 0
    } else if (identical(metric, "correlation")) {
      tree_data <- row_center_l2_normalize(data)
      tree_points <- if (isTRUE(self_query)) tree_data else row_center_l2_normalize(points)
      data_zero <- rowSums(tree_data * tree_data) <= 0
      points_zero <- if (isTRUE(self_query)) data_zero else rowSums(tree_points * tree_points) <= 0
    }
    if (metric %in% c("cosine", "correlation") && (any(data_zero) || any(points_zero))) {
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
      result <- finish_nn_result(out, "cpu_vptree", k, self_query, exact = TRUE, metric = metric)
      attr(result, "spatial_index") <- list(
        strategy = "native_exact_cpu_zero_safe_vptree_fallback",
        backend = "cpu_vptree",
        exact = TRUE,
        metric_transform = if (identical(metric, "correlation")) {
          "row_center_l2_normalize_vptree_unavailable_for_zero_rows"
        } else {
          "row_l2_normalize_vptree_unavailable_for_zero_rows"
        },
        fallback = TRUE,
        n_threads = as.integer(normalize_nn_threads(n_threads))
      )
      return(result)
    }
    if (isTRUE(self_query)) {
      out <- vptree_self_knn(tree_data, k = if (isTRUE(exclude_self)) k else k - 1L, n_threads = n_threads)
      if (!isTRUE(exclude_self)) {
        out$indices <- cbind(seq_len(nrow(tree_data)), out$indices)
        out$distances <- cbind(rep(0, nrow(tree_data)), out$distances)
      }
    } else {
      out <- vptree_query_knn(tree_data, tree_points, k = k, n_threads = n_threads)
    }
    result <- finish_nn_result(out, "cpu_vptree", k, self_query, exact = TRUE, metric = metric)
    if (metric %in% c("cosine", "correlation")) {
      result <- normalized_euclidean_to_similarity_distance(result, data_zero, points_zero)
      result <- sort_knn_rows_by_distance_index(result)
    }
    attr(result, "spatial_index") <- list(
      strategy = "native_exact_vptree",
      backend = "cpu_vptree",
      exact = TRUE,
      metric_transform = if (metric %in% c("cosine", "correlation")) {
        if (identical(metric, "correlation")) "row_center_l2_normalize_then_vptree" else "row_l2_normalize_then_vptree"
      } else {
        NA_character_
      },
      nodes = as.integer(out$nodes),
      n_threads = as.integer(normalize_nn_threads(n_threads))
    )
    return(result)
  }

  if (backend %in% c("cuda_grid", "cuda_grid_auto", "gpu_grid",
                     "cuda_grid2d", "cuda_grid3d")) {
    if (!isTRUE(self_query)) {
      stop("`backend = \"cuda_grid_auto\"` is only available for self-KNN searches.", call. = FALSE)
    }
    if (!identical(metric, "euclidean")) {
      stop("`backend = \"cuda_grid_auto\"` supports only Euclidean distances.", call. = FALSE)
    }
    if (!ncol(data) %in% c(2L, 3L)) {
      stop("`backend = \"cuda_grid_auto\"` supports only two- or three-column matrices.", call. = FALSE)
    }
    if (!isTRUE(cuda_available())) {
      stop("No CUDA GPU backend is available on this machine.", call. = FALSE)
    }
    nonself_k <- if (isTRUE(exclude_self)) k else k - 1L
    bins <- grid_bins_per_dim(nrow(data), nonself_k, ncol(data))
    out <- cuda_grid_self_knn_cpp(
      data,
      as.integer(nonself_k),
      as.integer(bins)
    )
    if (!isTRUE(exclude_self)) {
      out$indices <- cbind(seq_len(nrow(data)), out$indices)
      out$distances <- cbind(rep(0, nrow(data)), out$distances)
    }
    resolved <- if (ncol(data) == 3L) "cuda_grid3d" else "cuda_grid2d"
    result <- finish_nn_result(out, resolved, k, self_query, exact = TRUE, metric = metric)
    attr(result, "spatial_index") <- list(
      strategy = if (ncol(data) == 3L) "native_cuda_exact_uniform_grid_3d" else "native_cuda_exact_uniform_grid_2d",
      backend = resolved,
      exact = TRUE,
      bins_per_dim = as.integer(out$bins_per_dim),
      n_cells = as.integer(out$n_cells)
    )
    return(result)
  }

  if (backend %in% c("grid", "cpu_grid", "grid2d", "cpu_grid2d", "grid3d", "cpu_grid3d")) {
    if (!isTRUE(self_query)) {
      stop("`backend = \"cpu_grid\"` is only available for self-KNN searches.", call. = FALSE)
    }
    if (!identical(metric, "euclidean")) {
      stop("`backend = \"cpu_grid\"` supports only Euclidean distances.", call. = FALSE)
    }
    out <- grid_self_knn(
      data,
      k = k,
      backend = backend,
      exclude_self = isTRUE(exclude_self),
      n_threads = n_threads
    )
    result <- finish_nn_result(out, attr(out, "spatial_index")$backend, k, self_query, exact = TRUE, metric = metric)
    attr(result, "spatial_index") <- attr(out, "spatial_index")
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
      return(finish_nn_result(out, "cuda", k, self_query))
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

normalize_public_compute_backend <- function(backend, arg = "backend") {
  backend <- as.character(backend)[1L]
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
  backend <- as.character(backend)[1L]
  if (is.na(backend) || !nzchar(backend)) backend <- "auto"
  backend <- tolower(backend)
  if (!backend %in% c("auto", "cpu", "cuda")) {
    stop("`", arg, "` must be one of \"auto\", \"cpu\", or \"cuda\".", call. = FALSE)
  }
  backend
}

normalize_nn_method <- function(method) {
  method <- as.character(method)[1L]
  if (is.na(method) || !nzchar(method)) method <- "auto"
  method <- tolower(gsub("[[:space:]_-]+", "", method))
  aliases <- c(
    exact = "exact",
    flat = "flat",
    bruteforce = "bruteforce",
    grid = "grid",
    vptree = "vptree",
    sparse = "sparse",
    hnsw = "hnsw",
    ivf = "ivf",
    ivfpq = "ivfpq",
    nsg = "nsg",
    nndescent = "nndescent",
    cagra = "cagra",
    auto = "auto"
  )
  if (!method %in% names(aliases)) {
    stop(
      "`method` must be one of \"auto\", \"exact\", \"flat\", \"bruteforce\", ",
      "\"grid\", \"vptree\", \"sparse\", \"hnsw\", \"ivf\", \"ivfpq\", ",
      "\"nsg\", \"nndescent\", or \"cagra\".",
      " It should be one of the supported method labels.",
      call. = FALSE
    )
  }
  unname(aliases[[method]])
}

nn_metric_labels <- function() {
  c("euclidean", "cosine", "correlation", "inner_product")
}

nn_method_labels <- function() {
  c(
    "auto", "exact", "flat", "bruteforce", "grid", "vptree", "sparse",
    "hnsw", "ivf", "ivfpq", "nsg", "nndescent", "cagra"
  )
}

faissr_option <- function(name, default = NULL) {
  name <- as.character(name)
  for (key in paste0("faissR.", name)) {
    value <- getOption(key, NULL)
    if (!is.null(value)) return(value)
  }
  for (key in paste0("fastEmbedR.", name)) {
    value <- getOption(key, NULL)
    if (!is.null(value)) return(value)
  }
  default
}

#' Nearest-neighbour method capabilities
#'
#' `nn_capabilities()` returns the public method/backend/metric support table
#' used by the nearest-neighbour API. It separates combinations that are
#' supported by design from combinations that should be treated as expected
#' skips in benchmarks.
#'
#' @return A data frame with one row per public `method`, `backend`, and
#'   `metric` combination. Columns include `supported`, `exact`,
#'   `implementation`, and `notes`.
#' @examples
#' caps <- nn_capabilities()
#' subset(caps, method == "flat" & supported)
#' @export
nn_capabilities <- function() {
  methods <- nn_method_labels()
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
  out
}

nn_capability_row <- function(method, backend, metric) {
  supported <- FALSE
  exact <- NA
  implementation <- NA_character_
  notes <- NA_character_

  all_metrics <- metric %in% nn_metric_labels()
  euclidean <- identical(metric, "euclidean")
  non_ip_metric <- metric %in% c("euclidean", "cosine", "correlation")

  if (identical(method, "auto")) {
    if (identical(backend, "cpu")) {
      supported <- all_metrics
      exact <- NA
      implementation <- "shape-aware CPU selector"
      notes <- "Euclidean can resolve to exact, grid, FAISS IVF, or FAISS HNSW; non-Euclidean resolves to exact, FAISS Flat, FAISS HNSW, or RcppHNSW/hnswlib fallback depending on shape and availability."
    } else if (all_metrics) {
      supported <- TRUE
      exact <- NA
      implementation <- "shape-aware CUDA selector"
      notes <- if (euclidean) {
        "Can resolve to CUDA grid, FAISS GPU Flat, cuVS brute force, FAISS GPU CAGRA, or cuVS approximate routes depending on shape and availability."
      } else if (metric %in% c("cosine", "correlation")) {
        "CUDA auto uses validated FAISS GPU Flat normalized-IP search for cosine/correlation."
      } else {
        "CUDA auto uses FAISS GPU Flat IP for inner-product search."
      }
    }
  } else if (method %in% c("exact", "bruteforce")) {
    supported <- all_metrics
    exact <- TRUE
    if (identical(backend, "cpu")) {
      implementation <- "native CPU exact"
      notes <- "CPU exact scorer supports all public metrics."
    } else {
      implementation <- "FAISS GPU Flat or cuVS brute force"
      notes <- "Cosine/correlation and inner-product route through FAISS GPU Flat; Euclidean brute force can use cuVS when available."
    }
  } else if (identical(method, "flat")) {
    supported <- all_metrics
    exact <- TRUE
    implementation <- if (identical(backend, "cpu")) "FAISS CPU Flat" else "FAISS GPU Flat"
    notes <- "Cosine uses row L2 normalization plus Flat IP; correlation uses row centering plus L2 normalization plus Flat IP."
  } else if (identical(method, "grid")) {
    supported <- euclidean
    exact <- if (supported) TRUE else NA
    implementation <- if (identical(backend, "cpu")) "native CPU 2D/3D grid" else "native CUDA 2D/3D grid"
    notes <- if (supported) "Only valid for 2D/3D self-KNN." else "Grid search supports only Euclidean distances."
  } else if (identical(method, "vptree")) {
    supported <- identical(backend, "cpu") && non_ip_metric
    exact <- if (supported) TRUE else NA
    implementation <- if (identical(backend, "cpu")) "native CPU VP-tree" else NA_character_
    notes <- if (identical(backend, "cpu")) {
      "Supports Euclidean directly and cosine/correlation by normalized Euclidean search; raw inner product is not available for VP-tree because it is not a metric distance for tree pruning."
    } else {
      "VP-tree is CPU-only."
    }
  } else if (identical(method, "sparse")) {
    supported <- identical(backend, "cpu") && all_metrics
    exact <- if (supported) TRUE else NA
    implementation <- if (identical(backend, "cpu")) "native sparse CPU exact" else NA_character_
    notes <- if (identical(backend, "cpu")) "Requires sparse Matrix input to avoid dense conversion." else "Sparse exact search is CPU-only."
  } else if (identical(method, "hnsw")) {
    supported <- identical(backend, "cpu") && all_metrics
    exact <- if (supported) FALSE else NA
    implementation <- if (identical(backend, "cpu")) "FAISS HNSW or RcppHNSW/hnswlib" else NA_character_
    notes <- if (identical(backend, "cpu")) {
      "Uses FAISS HNSW for all metrics when available; cosine and correlation use normalized inner-product HNSW. Falls back to RcppHNSW/hnswlib when FAISS is unavailable."
    } else {
      "HNSW is CPU-only in the public API."
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
    supported <- identical(backend, "cpu") && identical(metric, "euclidean")
    exact <- if (supported) FALSE else NA
    implementation <- if (identical(backend, "cpu")) "FAISS CPU NSG" else NA_character_
    notes <- if (identical(backend, "cpu")) {
      "Validated for Euclidean/L2 search only. Cosine, correlation, and raw inner-product NSG construction are disabled because this linked FAISS build can abort during graph construction."
    } else {
      "NSG is CPU-only in the public API."
    }
  } else if (identical(method, "nndescent")) {
    supported <- euclidean
    exact <- if (supported) FALSE else NA
    implementation <- if (identical(backend, "cpu")) "native CPU NNDescent" else "cuVS CUDA NN-descent"
    notes <- if (supported) {
      "Validated for Euclidean/L2 self-KNN search."
    } else {
      "NNDescent routes are kept Euclidean-only because linked FAISS builds can abort during non-Euclidean graph construction."
    }
  } else if (identical(method, "cagra")) {
    supported <- identical(backend, "cuda") && metric %in% c("euclidean", "cosine", "correlation")
    exact <- if (supported) FALSE else NA
    implementation <- if (identical(backend, "cuda")) "FAISS GPU CAGRA or cuVS CAGRA" else NA_character_
    notes <- if (identical(backend, "cuda")) "CUDA-only approximate graph search, validated for Euclidean/L2; cosine/correlation use normalized Euclidean search." else "CAGRA is CUDA-only."
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
  tuning <- as.character(tuning)[1L]
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

resolve_public_nn_backend <- function(backend, method, metric = "euclidean") {
  backend_label <- as.character(backend)[1L]
  method_label <- normalize_nn_method(method)
  metric <- normalize_nn_metric(metric)
  if (!tolower(backend_label) %in% c("auto", "cpu", "cuda")) {
    removed <- c("cpu_clustered", "clustered", "cpu_nndescent", "nndescent",
                 "cpu_ivf", "ivf", "cpu_annoy", "annoy")
    if (tolower(backend_label) %in% removed) {
      stop("`backend` should be one of \"auto\", \"cpu\", or \"cuda\".", call. = FALSE)
    }
    if (!identical(method_label, "auto")) {
      stop(
        "Legacy backend labels cannot be combined with `method`. ",
        "Use `backend = \"auto\"`, \"cpu\", or \"cuda\" with `method`, ",
        "or omit `method` when using a legacy backend label.",
        call. = FALSE
      )
    }
    return(backend_label)
  }
  requested_device <- tolower(backend_label)
  device <- normalize_public_compute_backend(backend)
  method <- method_label
  if (identical(requested_device, "auto") && !identical(method, "auto")) {
    if (method %in% c("vptree", "sparse", "hnsw", "nsg")) {
      device <- "cpu"
    } else if (identical(method, "cagra") &&
               (isTRUE(cuda_available()) || isTRUE(cuvs_available()))) {
      device <- "cuda"
    }
  }
  if (identical(method, "auto")) {
    if (identical(device, "cuda")) {
      return(switch(
        metric,
        cosine = "faiss_gpu_flat_cosine",
        correlation = "faiss_gpu_flat_correlation",
        inner_product = "faiss_gpu_flat_ip",
        "cuda_auto"
      ))
    }
    return("cpu_auto")
  }
    if (!metric %in% c("euclidean", "inner_product") && identical(device, "cuda") &&
      !method %in% c("exact", "bruteforce", "flat", "ivf", "ivfpq", "nndescent", "cagra")) {
    if (identical(requested_device, "auto")) {
      device <- "cpu"
    } else {
      stop(
        "CUDA nearest-neighbour approximate methods currently support only ",
        "`metric = \"euclidean\"` or `metric = \"inner_product\"`. ",
        "Use `method = \"flat\"`, `\"ivf\"`, `\"ivfpq\"`, `\"exact\"`, ",
        "or `\"bruteforce\"` for validated CUDA cosine/correlation search.",
        call. = FALSE
      )
    }
  }
  if (identical(device, "cpu")) {
    if (identical(method, "grid") && !identical(metric, "euclidean")) {
      stop("CPU `method = \"grid\"` supports only `metric = \"euclidean\"`.", call. = FALSE)
    }
    if (identical(method, "vptree") && identical(metric, "inner_product")) {
      stop("CPU `method = \"vptree\"` does not support `metric = \"inner_product\"`.", call. = FALSE)
    }
    if (identical(method, "nsg") && !identical(metric, "euclidean")) {
      stop(
        "CPU `method = \"nsg\"` currently supports only `metric = \"euclidean\"`.",
        call. = FALSE
      )
    }
    if (identical(method, "nndescent") && !identical(metric, "euclidean")) {
      stop(
        "CPU `method = \"nndescent\"` is currently validated only for `metric = \"euclidean\"`.",
        call. = FALSE
      )
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
      vptree = "cpu_vptree",
      sparse = "cpu_sparse",
      hnsw = if (isTRUE(faiss_available())) "faiss_hnsw" else "hnsw",
      ivf = "faiss_ivf",
      ivfpq = "faiss_ivfpq",
      nsg = "faiss_nsg",
      nndescent = "cpu_nndescent",
      cagra = stop("`method = \"cagra\"` is only available with `backend = \"cuda\"`.", call. = FALSE),
      stop("Unsupported CPU nearest-neighbour method.", call. = FALSE)
    )
  } else {
    if (metric %in% c("cosine", "correlation") &&
        method %in% c("exact", "bruteforce", "flat")) {
      return(if (identical(metric, "correlation")) "faiss_gpu_flat_correlation" else "faiss_gpu_flat_cosine")
    }
    if (identical(metric, "inner_product") &&
        method %in% c("exact", "bruteforce", "flat")) {
      return("faiss_gpu_flat_ip")
    }
    if (identical(metric, "inner_product") && !method %in% c("ivf", "ivfpq")) {
      stop("CUDA `metric = \"inner_product\"` currently supports `method = \"exact\"`, `\"bruteforce\"`, `\"flat\"`, `\"ivf\"`, or `\"ivfpq\"`.", call. = FALSE)
    }
    switch(
      method,
      exact = if (isTRUE(faiss_available())) "faiss_gpu_flat_l2" else if (isTRUE(cuvs_available())) "cuda_cuvs_bruteforce" else "cuda",
      bruteforce = if (isTRUE(cuvs_available())) "cuda_cuvs_bruteforce" else if (isTRUE(faiss_available())) "faiss_gpu_flat_l2" else "cuda",
      flat = "faiss_gpu_flat_l2",
      grid = "cuda_grid",
      ivf = "faiss_gpu_ivf_flat",
      ivfpq = "faiss_gpu_ivfpq",
      nndescent = if (identical(metric, "euclidean")) {
        "cuda_cuvs_nndescent"
      } else {
        stop(
          "CUDA `method = \"nndescent\"` is currently validated only for `metric = \"euclidean\"`.",
          call. = FALSE
        )
      },
      cagra = if (isTRUE(faiss_available())) "faiss_gpu_cagra" else "cuda_cuvs_cagra",
      hnsw = stop("`method = \"hnsw\"` is only available with `backend = \"cpu\"`.", call. = FALSE),
      nsg = stop("`method = \"nsg\"` is only available with `backend = \"cpu\"`.", call. = FALSE),
      sparse = stop("`method = \"sparse\"` is only available with `backend = \"cpu\"`.", call. = FALSE),
      vptree = stop("`method = \"vptree\"` is only available with `backend = \"cpu\"`.", call. = FALSE),
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
    vptree = "vptree",
    sparse = "sparse",
    hnsw = "hnsw",
    ivf = "ivf",
    ivfpq = "ivfpq",
    nsg = "nsg",
    nndescent = "nndescent",
    cagra = "cagra"
  )
  labels[[method]] %||% method
}

is_sparse_matrix_input <- function(x) {
  inherits(x, "sparseMatrix") || inherits(x, "dgCMatrix")
}

as_dgCMatrix_input <- function(x, name) {
  if (inherits(x, "dgCMatrix")) return(x)
  if (is_sparse_matrix_input(x)) {
    return(methods::as(x, "dgCMatrix"))
  }
  if (!is.matrix(x) && !is.data.frame(x)) {
    stop("`", name, "` must be a matrix, data frame, or sparse Matrix object.", call. = FALSE)
  }
  if (!requireNamespace("Matrix", quietly = TRUE)) {
    stop(
      "Dense-to-sparse conversion requires the Matrix package. ",
      "Install Matrix or pass a dgCMatrix object.",
      call. = FALSE
    )
  }
  x <- as.matrix(x)
  storage.mode(x) <- "double"
  Matrix::Matrix(x, sparse = TRUE)
}

nn_sparse_compute <- function(data,
                              points,
                              k,
                              backend,
                              points_missing,
                              exclude_self,
                              metric) {
  metric <- normalize_nn_metric(metric)
  data <- as_dgCMatrix_input(data, "data")
  points <- if (isTRUE(points_missing)) {
    data
  } else {
    as_dgCMatrix_input(points, "points")
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
  k <- as.integer(k)
  if (length(k) != 1L || is.na(k) || !is.finite(k) || k < 1L) {
    stop("`k` must be NULL or a positive integer.", call. = FALSE)
  }
  max_k <- if (isTRUE(exclude_self)) nrow(data) - 1L else nrow(data)
  if (k > max_k) {
    stop("`k` cannot be larger than the available neighbor count.", call. = FALSE)
  }
  if (!all(is.finite(data@x)) || !all(is.finite(points@x))) {
    stop("`data` and `points` must contain only finite values.", call. = FALSE)
  }
  out <- sparse_nn_cpp(
    data,
    points,
    as.integer(k),
    metric,
    isTRUE(exclude_self),
    isTRUE(self_query)
  )
  result <- finish_nn_result(out, "cpu_sparse", k, self_query, exact = TRUE, metric = metric)
  attr(result, "sparse") <- list(
    input = TRUE,
    backend = "native_dgCMatrix_exact",
    data_nnz = length(data@x),
    points_nnz = length(points@x),
    requested_backend = backend
  )
  result
}

resolve_auto_knn_gpu_backend <- function(backend,
                                         self_query,
                                         n_points,
                                         n,
                                         p,
                                         k,
                                         work_size,
                                         metric = "euclidean") {
  if (!identical(backend, "auto")) return(NA_character_)
  if (!isTRUE(self_query)) return(NA_character_)
  if (k > 256L) return(NA_character_)
  if (work_size < 5e8) return(NA_character_)
  if (!isTRUE(cuda_available()) && !isTRUE(cuvs_available()) && !isTRUE(faiss_available())) {
    return(NA_character_)
  }
  select_cuda_auto_backend(
    self_query = self_query,
    n = n,
    p = p,
    n_points = n_points,
    k = k,
    work_size = work_size,
    metric = metric
  )
}

select_cuda_auto_backend <- function(self_query,
                                     n,
                                     p,
                                     n_points,
                                     k,
                                     work_size,
                                     metric = "euclidean") {
  metric <- normalize_nn_metric(metric)
  if (k > 256L) {
    stop("CUDA auto backends currently support `k <= 256`.", call. = FALSE)
  }
  if (!isTRUE(cuda_available()) && !isTRUE(cuvs_available())) {
    stop("No CUDA GPU backend is available on this machine.", call. = FALSE)
  }
  if (!identical(metric, "euclidean")) {
    return(switch(
      metric,
      cosine = "faiss_gpu_flat_cosine",
      correlation = "faiss_gpu_flat_correlation",
      inner_product = "faiss_gpu_flat_ip"
    ))
  }
  if (isTRUE(self_query) && p %in% c(2L, 3L) && isTRUE(cuda_available()) && n >= 10000L) {
    return("cuda_grid")
  }
  if (!isTRUE(self_query)) {
    if (isTRUE(faiss_available())) return("faiss_gpu_flat_l2")
    if (isTRUE(cuvs_available())) return("cuda_cuvs_bruteforce")
    return("cuda")
  }
  exact_n <- faiss_option_int("cuda_auto_exact_n", 100000L, min_value = 1000L, max_value = 10000000L)
  exact_work <- faissr_option("cuda_auto_exact_work", 5e12)
  exact_work <- suppressWarnings(as.numeric(exact_work))
  if (length(exact_work) != 1L || is.na(exact_work) || !is.finite(exact_work)) {
    exact_work <- 5e12
  }
  if (n <= exact_n || work_size <= exact_work || k <= 8L) {
    if (isTRUE(faiss_available())) return("faiss_gpu_flat_l2")
    if (isTRUE(cuvs_available())) return("cuda_cuvs_bruteforce")
    return("cuda")
  }
  if (isTRUE(faiss_available())) {
    return("faiss_gpu_cagra")
  }
  if (isTRUE(cuvs_available())) {
    return(select_cuvs_auto_backend(
      self_query = self_query,
      n = n,
      p = p,
      n_points = n_points,
      k = k,
      work_size = work_size
    ))
  }
  "cuda"
}

select_cuvs_auto_backend <- function(self_query,
                                     n,
                                     p,
                                     n_points,
                                     k,
                                     work_size) {
  small_threshold <- faissr_option("cuvs_bruteforce_work_threshold", 5e12)
  small_threshold <- suppressWarnings(as.numeric(small_threshold))
  if (length(small_threshold) != 1L || is.na(small_threshold) || !is.finite(small_threshold)) {
    small_threshold <- 5e12
  }
  if (!isTRUE(self_query) || work_size <= small_threshold || k <= 8L || n <= 100000L || n_points <= 5000L) {
    return("cuda_cuvs_bruteforce")
  }
  if (p <= 64L) {
    return("cuda_cuvs_ivf_flat")
  }
  "cuda_cuvs_nndescent"
}

should_use_auto_cpu_approx_self_knn <- function(self_query,
                                                n,
                                                p,
                                                k,
                                                work_size) {
  if (!isTRUE(self_query)) return(FALSE)
  if (n < 5000L || k < 10L || p < 2L) return(FALSE)
  if (work_size < 5e8) return(FALSE)
  TRUE
}

select_cpu_auto_backend <- function(self_query,
                                    n,
                                    p,
                                    n_points,
                                    k,
                                    work_size,
                                    metric = "euclidean") {
  if (!identical(metric, "euclidean")) {
    exact_work <- faissr_option("cpu_auto_exact_work", 2e8)
    exact_work <- suppressWarnings(as.numeric(exact_work))
    if (length(exact_work) != 1L || is.na(exact_work) || !is.finite(exact_work)) {
      exact_work <- 2e8
    }
    faiss_flat_work <- cpu_auto_faiss_flat_work_threshold()
    faiss_flat_backend <- cpu_auto_metric_faiss_flat_backend(metric)
    if (!is.na(faiss_flat_backend) && isTRUE(faiss_available()) &&
        work_size >= faiss_flat_work &&
        (!isTRUE(self_query) || k < 10L || n < 5000L)) {
      return(faiss_flat_backend)
    }
    if (!isTRUE(self_query) || work_size <= exact_work || n < 5000L || k < 10L || p < 2L) {
      return("cpu")
    }
    if (isTRUE(faiss_available())) return("faiss_hnsw")
    if (isTRUE(requireNamespace("RcppHNSW", quietly = TRUE))) return("hnsw")
    if (!is.na(faiss_flat_backend) && isTRUE(faiss_available()) && work_size >= faiss_flat_work) {
      return(faiss_flat_backend)
    }
    return("cpu")
  }
  if (isTRUE(self_query) && should_use_grid2d_self_knn(
    self_query = TRUE,
    n = n,
    p = p,
    k = k,
    exclude_self = FALSE,
    metric = "euclidean"
  )) {
    return("cpu_grid")
  }
  exact_work <- faissr_option("cpu_auto_exact_work", 2e8)
  exact_work <- suppressWarnings(as.numeric(exact_work))
  if (length(exact_work) != 1L || is.na(exact_work) || !is.finite(exact_work)) {
    exact_work <- 2e8
  }
  if (work_size <= exact_work || n < 5000L || k < 10L || p < 2L) {
    return("cpu")
  }
  if (isTRUE(self_query) && n >= 1000000L && isTRUE(faiss_available())) {
    return("faiss_ivf")
  }
  if (isTRUE(faiss_available())) return("faiss_hnsw")
  if (isTRUE(requireNamespace("RcppHNSW", quietly = TRUE))) return("hnsw")
  "cpu"
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

normalize_nn_threads <- function(n_threads) {
  if (is.null(n_threads)) {
    n_threads <- suppressWarnings(parallel::detectCores(logical = FALSE))
  }
  n_threads <- suppressWarnings(as.integer(n_threads))
  if (length(n_threads) != 1L || is.na(n_threads) || !is.finite(n_threads) || n_threads < 1L) {
    n_threads <- 1L
  }
  as.integer(max(1L, min(64L, n_threads)))
}

normalize_nn_metric <- function(metric) {
  metric <- tolower(as.character(metric))
  match.arg(metric, c("euclidean", "cosine", "correlation", "inner_product"))
}

should_use_grid2d_self_knn <- function(self_query,
                                       n,
                                       p,
                                       k,
                                       exclude_self,
                                       metric) {
  if (!isTRUE(self_query)) return(FALSE)
  if (!identical(metric, "euclidean")) return(FALSE)
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
    return("cpu_vptree")
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
  if (sum(finite_sds) < p) {
    out <- "cpu_vptree"
    attr(out, "reason") <- "degenerate_or_duplicate_dimension"
    return(out)
  }
  anisotropy <- min(sds[finite_sds]) / max(sds[finite_sds])
  anisotropy_threshold <- as.numeric(faissr_option("cpu_spatial_anisotropy_threshold", 0.02))
  if (!is.finite(anisotropy_threshold) || anisotropy_threshold <= 0) anisotropy_threshold <- 0.02
  if (is.finite(anisotropy) && anisotropy < anisotropy_threshold) {
    out <- "cpu_vptree"
    attr(out, "reason") <- sprintf("anisotropic_sample_sd_ratio_%.4g", anisotropy)
    return(out)
  }

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
  if (is.finite(imbalance) && imbalance > imbalance_threshold) {
    out <- "cpu_vptree"
    attr(out, "reason") <- sprintf("sample_grid_imbalance_%.3g", imbalance)
    return(out)
  }
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

finalize_normalized_euclidean_metric_result <- function(result, inputs) {
  result <- normalized_euclidean_to_similarity_distance(
    result,
    data_zero = inputs$data_zero,
    points_zero = inputs$points_zero
  )
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
  data_zero <- rowSums(data_metric * data_metric) <= 0
  points_zero <- if (isTRUE(self_query)) data_zero else rowSums(points_metric * points_metric) <= 0
  result <- restore_zero_zero_normalized_ip_distances(result, data_zero, points_zero)
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
  result <- restore_zero_zero_normalized_ip_distances(result, data_zero, points_zero)
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
  result <- restore_zero_zero_normalized_ip_distances(result, data_zero, points_zero)
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
  params <- faiss_hnsw_params(k)
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
  result <- restore_zero_zero_normalized_ip_distances(result, data_zero, points_zero)
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
    hnsw_parameters_adjusted = isTRUE(out$hnsw_parameters_adjusted)
  )
  result
}

restore_zero_zero_normalized_ip_distances <- function(result, data_zero, points_zero) {
  if (!any(points_zero) || !any(data_zero) || ncol(result$indices) < 1L) {
    return(result)
  }
  for (i in which(points_zero)) {
    zero_neighbor <- data_zero[result$indices[i, ]]
    if (any(zero_neighbor)) {
      result$distances[i, zero_neighbor] <- 0
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
                                strategy = "anchor_projection_candidate_knn") {
  n <- nrow(data)
  label <- if (is.null(label)) paste0(backend, "_approx") else as.character(label)
  nonself_k <- if (isTRUE(exclude_self)) k else k - 1L
  if (nonself_k < 1L) {
    out <- list(
      indices = matrix(seq_len(n), n, 1L),
      distances = matrix(0, n, 1L)
    )
    return(finish_nn_result(out, label, k, TRUE, exact = FALSE))
  }

  params <- gpu_approx_params(n, nonself_k, backend = backend, label = label)
  anchors <- select_landmark_rows(data, params$anchors, seed)
  projection <- nn_compute(
    data[anchors, , drop = FALSE],
    data,
    k = params$projection_k,
    backend = backend,
    points_missing = FALSE,
    exclude_self = FALSE
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
  result <- finish_nn_result(out, label, k, TRUE, exact = FALSE)
  attr(result, "approximation") <- list(
    strategy = strategy,
    backend = backend,
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
                             seed = 4L) {
  gpu_approx_self_knn(
    data,
    k = k,
    backend = backend,
    exclude_self = exclude_self,
    seed = seed,
    label = label,
    strategy = strategy
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
  graph_degree <- gpu_nndescent_graph_degree(n, k, backend = backend)
  default_iters <- if (identical(backend, "cuda") && !is.null(n) && n >= 50000L) {
    3L
  } else {
    1L
  }
  n_iters <- gpu_nndescent_option(backend, "iters", default_iters)
  sources <- gpu_nndescent_option(backend, "sources", NULL)
  neighbors <- gpu_nndescent_option(backend, "neighbors", NULL)
  delta <- gpu_nndescent_option(backend, "delta", 0.015)

  n_iters <- suppressWarnings(as.integer(n_iters))
  if (length(n_iters) != 1L || is.na(n_iters) || !is.finite(n_iters)) n_iters <- 1L
  n_iters <- as.integer(max(1L, min(5L, n_iters)))

  if (is.null(sources)) {
    sources <- max(3L, min(graph_degree, 10L))
  } else {
    sources <- suppressWarnings(as.integer(sources))
    if (length(sources) != 1L || is.na(sources) || !is.finite(sources)) {
      sources <- max(3L, min(graph_degree, 10L))
    }
  }
  sources <- as.integer(max(1L, min(graph_degree, sources)))

  if (is.null(neighbors)) {
    neighbors <- max(5L, min(graph_degree, ceiling(graph_degree / 2)))
  } else {
    neighbors <- suppressWarnings(as.integer(neighbors))
    if (length(neighbors) != 1L || is.na(neighbors) || !is.finite(neighbors)) {
      neighbors <- max(5L, min(graph_degree, ceiling(graph_degree / 2)))
    }
  }
  neighbors <- as.integer(max(1L, min(graph_degree, neighbors)))

  delta <- suppressWarnings(as.numeric(delta))
  if (length(delta) != 1L || is.na(delta) || !is.finite(delta) || delta < 0) {
    delta <- 0.015
  }

  list(
    graph_degree = graph_degree,
    n_iters = n_iters,
    sources = sources,
    neighbors = neighbors,
    delta = delta
  )
}

gpu_nndescent_self_knn <- function(data,
                                   k,
                                   backend,
                                   seed = 4L) {
  n <- nrow(data)
  k <- as.integer(k)
  backend <- match.arg(as.character(backend), c("cuda"))
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
    seed = seed
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
      as.integer(work_k)
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

  out <- list(indices = indices, distances = distances)
  attr(out, "cuda_kernel") <- "row_candidate_knn"
  attr(out, "approximation") <- list(
    strategy = paste0("adaptive_seeded_nndescent_native_", backend),
    backend = backend,
    seed_backend = paste0(backend, "_ivf"),
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

cuda_nndescent_self_knn <- function(data,
                                    k,
                                    seed = 4L) {
  gpu_nndescent_self_knn(data, k = k, backend = "cuda", seed = seed)
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
  if (isTRUE(exact) || identical(backend, "faiss_flat") ||
      identical(backend, "cpu_nndescent_faiss_flat")) {
    out <- nn_faiss_flat_cpp(
      data,
      data,
      as.integer(k),
      TRUE,
      as.integer(n_threads)
    )
    attr(out, "approximation") <- list(
      strategy = "faiss_IndexFlatL2_self",
      backend = "faiss",
      library = "faiss",
      exact = TRUE,
      seed = as.integer(seed)
    )
    return(out)
  }
  if (identical(backend, "faiss_hnsw") ||
      identical(backend, "cpu_nndescent_faiss_hnsw")) {
    params <- faiss_hnsw_params(k)
    out <- nn_faiss_hnsw_cpp(
      data,
      data,
      as.integer(k),
      as.integer(params$m),
      as.integer(params$ef_construction),
      as.integer(params$ef_search),
      "euclidean",
      "euclidean",
      TRUE,
      as.integer(n_threads)
    )
    attr(out, "approximation") <- list(
      strategy = "faiss_IndexHNSWFlat_self",
      backend = backend,
      library = "faiss",
      exact = FALSE,
      m = as.integer(out$m),
      ef_construction = as.integer(out$ef_construction),
      ef_search = as.integer(out$ef_search),
      requested_m = as.integer(out$requested_m),
      requested_ef_construction = as.integer(out$requested_ef_construction),
      requested_ef_search = as.integer(out$requested_ef_search),
      hnsw_parameters_adjusted = isTRUE(out$hnsw_parameters_adjusted),
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
      "euclidean",
      "euclidean",
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
  params <- faiss_ivf_params(n, k)
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
    nlist = as.integer(out$nlist),
    nprobe = as.integer(out$nprobe),
    seed = as.integer(seed)
  )
  out
}

rcpphnsw_params <- function(k) {
  m <- faissr_option("hnsw_m", 16L)
  ef_construction <- faissr_option("hnsw_ef_construction", 200L)
  ef <- faissr_option("hnsw_ef", max(50L, 3L * as.integer(k)))
  m <- suppressWarnings(as.integer(m))
  ef_construction <- suppressWarnings(as.integer(ef_construction))
  ef <- suppressWarnings(as.integer(ef))
  if (length(m) != 1L || is.na(m) || !is.finite(m) || m < 2L) m <- 16L
  if (length(ef_construction) != 1L || is.na(ef_construction) ||
      !is.finite(ef_construction) || ef_construction < m) {
    ef_construction <- max(200L, m)
  }
  if (length(ef) != 1L || is.na(ef) || !is.finite(ef) || ef < k) {
    ef <- max(50L, 3L * as.integer(k))
  }
  list(
    m = as.integer(m),
    ef_construction = as.integer(ef_construction),
    ef = as.integer(ef)
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
    n_threads = as.integer(n_threads)
  )
  out
}

nndescent_pool_size <- function(n, k) {
  as.integer(min(n - 1L, max(k + 15L, min(160L, ceiling(2.5 * k)))))
}

nndescent_iterations <- function(n, k) {
  out <- if (n >= 50000L) 3L else 4L
  if (k < 30L) out <- out + 1L
  as.integer(out)
}

nndescent_self_knn <- function(data,
                               k,
                               seed = 4L,
                               n_threads = NULL) {
  n <- nrow(data)
  k <- as.integer(k)
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
      n_threads = n_threads
    ))
  }
  pool_size <- nndescent_pool_size(n, k)
  n_iters <- nndescent_iterations(n, k)
  max_candidates <- as.integer(min(n - 1L, max(pool_size * 4L, k * 12L)))
  n_random_projections <- if (n >= 50000L) 8L else 6L
  out <- nndescent_self_knn_cpp(
    data,
    as.integer(k),
    as.integer(pool_size),
    as.integer(n_iters),
    as.integer(max_candidates),
    as.integer(n_random_projections),
    as.integer(seed),
    TRUE,
    as.integer(max(1L, min(8L, n_threads)))
  )
  params <- list(
    strategy = "native_cpu_nndescent",
    backend = "cpu",
    pool_size = pool_size,
    n_iters = n_iters,
    max_candidates = max_candidates,
    n_random_projections = n_random_projections,
    reverse_candidates = "rank_ordered"
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

annoy_tree_count <- function(n, k) {
  value <- faissr_option("annoy_n_trees", NULL)
  if (is.null(value)) {
    value <- if (n >= 10000L) 50L else 24L
  }
  value <- suppressWarnings(as.integer(value))
  if (length(value) != 1L || is.na(value) || !is.finite(value)) value <- 24L
  as.integer(max(1L, min(128L, value)))
}

annoy_leaf_size <- function(k) {
  value <- faissr_option("annoy_leaf_size", NULL)
  if (is.null(value)) {
    value <- max(64L, min(256L, ceiling(2L * k)))
  }
  value <- suppressWarnings(as.integer(value))
  if (length(value) != 1L || is.na(value) || !is.finite(value)) value <- max(64L, 2L * k)
  as.integer(max(k + 1L, min(1024L, value)))
}

annoy_search_k <- function(k, n_trees, leaf_size) {
  value <- faissr_option("annoy_search_k", NULL)
  if (is.null(value)) {
    value <- max(n_trees * leaf_size, 50L * k)
  }
  value <- suppressWarnings(as.integer(value))
  if (length(value) != 1L || is.na(value) || !is.finite(value)) value <- n_trees * leaf_size
  as.integer(max(k, value))
}

annoy_self_knn <- function(data,
                           k,
                           seed = 4L,
                           n_threads = NULL) {
  n <- nrow(data)
  k <- as.integer(k)
  if (length(k) != 1L || is.na(k) || !is.finite(k) || k < 1L || k >= n) {
    stop("`k` must be in [1, nrow(data) - 1].", call. = FALSE)
  }
  n_threads <- normalize_nn_threads(n_threads)
  n_trees <- annoy_tree_count(n, k)
  leaf_size <- annoy_leaf_size(k)
  search_k <- annoy_search_k(k, n_trees, leaf_size)
  out <- annoy_self_knn_cpp(
    data,
    as.integer(k),
    as.integer(n_trees),
    as.integer(leaf_size),
    as.integer(search_k),
    as.integer(seed),
    TRUE,
    as.integer(max(1L, min(8L, n_threads)))
  )
  attr(out, "approximation") <- list(
    strategy = "annoy_style_random_projection_forest_native",
    backend = "cpu_annoy",
    n_trees = as.integer(out$n_trees),
    leaf_size = as.integer(out$leaf_size),
    search_k = as.integer(out$search_k),
    n_leaves = as.integer(out$n_leaves),
    seed = as.integer(seed)
  )
  out
}

ivf_list_count <- function(n, k) {
  n <- as.integer(n)
  k <- as.integer(k)
  count <- max(16L, ceiling(sqrt(n)))
  count <- min(count, ceiling(n / max(50L, 20L * k)))
  as.integer(max(4L, min(n, count, 1024L)))
}

ivf_probe_count <- function(nlist, k) {
  nlist <- as.integer(nlist)
  k <- as.integer(k)
  as.integer(max(1L, min(nlist, max(16L, ceiling(sqrt(nlist)), ceiling(k / 3)))))
}

faiss_ivf_params <- function(n, k) {
  n <- as.integer(n)
  k <- as.integer(k)
  nlist <- faissr_option(c("faiss_nlist", "ivf_nlist"), NULL)
  nprobe <- faissr_option(c("faiss_nprobe", "ivf_nprobe"), NULL)

  nlist <- if (is.null(nlist)) ivf_list_count(n, k) else suppressWarnings(as.integer(nlist))
  requested_nlist <- nlist
  if (length(nlist) != 1L || is.na(nlist) || !is.finite(nlist)) {
    nlist <- ivf_list_count(n, k)
  }
  if (length(requested_nlist) != 1L || is.na(requested_nlist) || !is.finite(requested_nlist)) {
    requested_nlist <- nlist
  }
  nlist <- max(1L, min(n, nlist))

  nprobe <- if (is.null(nprobe)) ivf_probe_count(nlist, k) else suppressWarnings(as.integer(nprobe))
  requested_nprobe <- nprobe
  if (length(nprobe) != 1L || is.na(nprobe) || !is.finite(nprobe)) {
    nprobe <- ivf_probe_count(nlist, k)
  }
  if (length(requested_nprobe) != 1L || is.na(requested_nprobe) || !is.finite(requested_nprobe)) {
    requested_nprobe <- nprobe
  }
  nprobe <- max(1L, min(nlist, nprobe))
  list(
    nlist = as.integer(nlist),
    nprobe = as.integer(nprobe),
    requested_nlist = as.integer(requested_nlist),
    requested_nprobe = as.integer(requested_nprobe)
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
  pq_dim <- faissr_option(c("cuvs_ivfpq_pq_dim", "ivfpq_pq_dim"), 0L)
  pq_dim <- suppressWarnings(as.integer(pq_dim))
  requested_pq_dim <- pq_dim
  if (length(pq_dim) != 1L || is.na(pq_dim) || !is.finite(pq_dim) || pq_dim < 0L) {
    pq_dim <- 0L
  }
  if (length(requested_pq_dim) != 1L || is.na(requested_pq_dim) || !is.finite(requested_pq_dim)) {
    requested_pq_dim <- pq_dim
  }

  pq_bits <- faissr_option(c("cuvs_ivfpq_pq_bits", "ivfpq_pq_bits"), 8L)
  pq_bits <- suppressWarnings(as.integer(pq_bits))
  requested_pq_bits <- pq_bits
  if (length(pq_bits) != 1L || is.na(pq_bits) || !is.finite(pq_bits)) {
    pq_bits <- 8L
  }
  if (length(requested_pq_bits) != 1L || is.na(requested_pq_bits) || !is.finite(requested_pq_bits)) {
    requested_pq_bits <- pq_bits
  }
  pq_bits <- as.integer(max(4L, min(8L, pq_bits)))

  list(
    pq_dim = as.integer(pq_dim),
    pq_bits = pq_bits,
    requested_pq_dim = as.integer(requested_pq_dim),
    requested_pq_bits = as.integer(requested_pq_bits)
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
  policy <- faissr_option("faiss_gpu_ivf_tune_policy", "cache")
  policy <- tolower(as.character(policy)[1L])
  if (!policy %in% c("cache", "pilot", "fixed", "off")) {
    policy <- "cache"
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

faiss_gpu_ivf_should_tune <- function(data, k, self_query, tuning = "auto") {
  tuning <- normalize_nn_tuning(tuning)
  if (identical(tuning, "off")) return(FALSE)
  if (!isTRUE(self_query)) return(FALSE)
  if (!isTRUE(faiss_available())) return(FALSE)
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

faiss_pq_params <- function(p) {
  p <- as.integer(p)
  m <- faiss_option_int("pq_m", faiss_pq_default_m(p), min_value = 1L, max_value = p)
  while (m > 1L && p %% m != 0L) m <- m - 1L
  nbits <- faiss_option_int("pq_nbits", 8L, min_value = 4L, max_value = 12L)
  list(m = as.integer(m), nbits = as.integer(nbits))
}

faiss_hnsw_params <- function(k) {
  k <- as.integer(k)
  m <- faiss_option_int("hnsw_m", 32L, min_value = 2L, max_value = 256L)
  ef_construction <- faiss_option_int(
    "hnsw_ef_construction",
    max(200L, 5L * m),
    min_value = m,
    max_value = 4096L
  )
  ef_search <- faiss_option_int(
    "hnsw_ef_search",
    max(150L, 3L * k),
    min_value = k,
    max_value = 4096L
  )
  list(m = as.integer(m), ef_construction = as.integer(ef_construction), ef_search = as.integer(ef_search))
}

faiss_nsg_params <- function(k) {
  k <- as.integer(k)
  r <- faiss_option_int("nsg_r", 48L, min_value = 2L, max_value = 512L)
  search_l <- faiss_option_int("nsg_search_l", max(200L, 4L * k), min_value = k, max_value = 4096L)
  build_type <- faiss_option_int("nsg_build_type", 1L, min_value = 0L, max_value = 1L)
  list(r = as.integer(r), search_l = as.integer(search_l), build_type = as.integer(build_type))
}

faiss_nndescent_params <- function(k) {
  k <- as.integer(k)
  graph_k <- faiss_option_int("nndescent_graph_k", max(100L, 2L * k), min_value = k, max_value = 1024L)
  n_iter <- faiss_option_int("nndescent_iter", 20L, min_value = 1L, max_value = 100L)
  search_l <- faiss_option_int("nndescent_search_l", max(graph_k, 2L * k), min_value = k, max_value = 4096L)
  list(graph_k = as.integer(graph_k), n_iter = as.integer(n_iter), search_l = as.integer(search_l))
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

cuvs_cagra_params <- function(n, k) {
  n <- as.integer(n)
  k <- as.integer(k)
  default_graph_degree <- max(64L, k + 1L)
  requested_graph_degree <- cuvs_requested_option_int("graph_degree", default_graph_degree)
  graph_degree <- cuvs_option_int(
    "graph_degree",
    default = default_graph_degree,
    min_value = k + 1L,
    max_value = max(1L, n - 1L)
  )
  default_intermediate_graph_degree <- max(128L, graph_degree * 2L)
  requested_intermediate_graph_degree <- cuvs_requested_option_int(
    "intermediate_graph_degree",
    default_intermediate_graph_degree
  )
  intermediate_graph_degree <- cuvs_option_int(
    "intermediate_graph_degree",
    default = default_intermediate_graph_degree,
    min_value = graph_degree,
    max_value = max(1L, n - 1L)
  )
  requested_search_width <- cuvs_requested_option_int("search_width", 0L)
  search_width <- cuvs_option_int(
    "search_width",
    default = 0L,
    min_value = 0L,
    max_value = 1024L
  )
  default_itopk_size <- max(64L, graph_degree)
  requested_itopk_size <- cuvs_requested_option_int("itopk_size", default_itopk_size)
  itopk_size <- cuvs_option_int(
    "itopk_size",
    default = default_itopk_size,
    min_value = k,
    max_value = 4096L
  )
  list(
    graph_degree = as.integer(graph_degree),
    intermediate_graph_degree = as.integer(intermediate_graph_degree),
    search_width = as.integer(search_width),
    itopk_size = as.integer(itopk_size),
    requested_graph_degree = as.integer(requested_graph_degree),
    requested_intermediate_graph_degree = as.integer(requested_intermediate_graph_degree),
    requested_search_width = as.integer(requested_search_width),
    requested_itopk_size = as.integer(requested_itopk_size)
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
  if (identical(tuning, "off")) return(FALSE)
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
  policy <- faissr_option("cuvs_cagra_tune_policy", "cache")
  policy <- tolower(as.character(policy)[1L])
  if (!policy %in% c("cache", "pilot", "fixed", "off")) {
    policy <- "cache"
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

cuvs_cagra_tune_params <- function(data, k, base_params, tuning = "auto") {
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
          as.integer(cand$itopk_size)
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
  n <- as.integer(n)
  k <- as.integer(k)
  graph_degree <- cuvs_option_int(
    "nndescent_graph_degree",
    default = k,
    min_value = k,
    max_value = max(1L, n - 1L)
  )
  intermediate_graph_degree <- cuvs_option_int(
    "nndescent_intermediate_graph_degree",
    default = max(graph_degree * 2L, graph_degree),
    min_value = graph_degree,
    max_value = max(1L, n - 1L)
  )
  max_iterations <- cuvs_option_int(
    "nndescent_max_iterations",
    default = 20L,
    min_value = 1L,
    max_value = 200L
  )
  list(
    graph_degree = as.integer(graph_degree),
    intermediate_graph_degree = as.integer(intermediate_graph_degree),
    max_iterations = as.integer(max_iterations)
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

vptree_self_knn <- function(data,
                            k,
                            n_threads = NULL) {
  n <- nrow(data)
  k <- as.integer(k)
  if (length(k) != 1L || is.na(k) || !is.finite(k) || k < 1L || k >= n) {
    stop("`k` must be in [1, nrow(data) - 1].", call. = FALSE)
  }
  n_threads <- normalize_nn_threads(n_threads)
  vptree_self_knn_cpp(
    data,
    as.integer(k),
    TRUE,
    as.integer(max(1L, min(8L, n_threads)))
  )
}

vptree_query_knn <- function(data,
                             points,
                             k,
                             n_threads = NULL) {
  n <- nrow(data)
  k <- as.integer(k)
  if (length(k) != 1L || is.na(k) || !is.finite(k) || k < 1L || k > n) {
    stop("`k` must be in [1, nrow(data)].", call. = FALSE)
  }
  n_threads <- normalize_nn_threads(n_threads)
  vptree_query_knn_cpp(
    data,
    points,
    as.integer(k),
    TRUE,
    as.integer(max(1L, min(8L, n_threads)))
  )
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
  if (backend %in% c("vptree", "cpu_vptree")) {
    out <- vptree_self_knn(data, k = if (isTRUE(exclude_self)) k else k - 1L, n_threads = n_threads)
    if (!isTRUE(exclude_self)) {
      out$indices <- cbind(seq_len(nrow(data)), out$indices)
      out$distances <- cbind(rep(0, nrow(data)), out$distances)
    }
    attr(out, "spatial_index") <- list(
      strategy = "native_exact_vptree",
      backend = "cpu_vptree",
      exact = TRUE,
      reason = if (is.null(reason)) "explicit" else reason,
      n_threads = as.integer(normalize_nn_threads(n_threads))
    )
    return(out)
  }
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
#'   uses exact, grid, FAISS IVF, or FAISS HNSW depending on data shape and
#'   size. CUDA auto uses CUDA grid, exact FAISS GPU Flat/cuVS brute force, or
#'   FAISS GPU CAGRA for Euclidean searches when appropriate, and FAISS GPU Flat
#'   IP routes for cosine, correlation, and inner-product searches
#'   [1-3,5,13-15].
#'   \item `"exact"`: exact nearest-neighbour search. CPU uses faissR's native
#'   exact route; CUDA uses FAISS GPU Flat or cuVS brute force when available
#'   [1-3,16].
#'   \item `"flat"`: FAISS Flat exhaustive index. CPU and FAISS GPU support
#'   L2, IP, and normalized-IP cosine/correlation routes when available
#'   [1-2,16].
#'   \item `"bruteforce"`: exhaustive brute-force search. CUDA prefers RAPIDS
#'   cuVS brute force; CPU maps to exact CPU search [3].
#'   \item `"grid"`: native exact 2D/3D Euclidean spatial grid search.
#'   \item `"vptree"`: native exact CPU vantage-point-tree search for
#'   Euclidean, cosine, and correlation. Cosine/correlation use normalized
#'   Euclidean tree search when rows are nonzero/nonconstant and exact CPU
#'   fallback otherwise. Raw inner product is not available for VP-tree because
#'   it is not a metric distance for tree pruning; use `"exact"`, `"flat"`, or
#'   `"hnsw"` for inner-product search.
#'   \item `"sparse"`: native exact sparse `dgCMatrix` CPU search.
#'   \item `"hnsw"`: FAISS CPU HNSW approximate graph-search index [5,16].
#'   \item `"ivf"`: FAISS IVF-Flat inverted-file index, trading exhaustive
#'   search for coarse-list probing. It supports L2, raw IP, and normalized-IP
#'   cosine/correlation routes on CPU and FAISS GPU [1-2,16].
#'   \item `"ivfpq"`: FAISS inverted-file index with product quantization,
#'   mainly for compressed-memory approximate search. It supports L2, raw IP,
#'   and normalized-IP cosine/correlation routes on CPU and FAISS GPU [1-2,6,16].
#'   \item `"nsg"`: FAISS CPU NSG graph-search index when exposed by the linked
#'   FAISS build. It is exposed for Euclidean/L2 only because linked FAISS
#'   graph builders can abort during normalized cosine/correlation or raw
#'   inner-product construction [16].
#'   \item `"nndescent"`: NN-descent approximate graph construction via
#'   faissR's native CPU route or cuVS on CUDA. It is kept Euclidean/L2-only;
#'   FAISS NNDescent is experimental opt-in because linked FAISS builds can
#'   abort during graph construction [3-4,16].
#'   \item `"cagra"`: CUDA-only graph-search method via FAISS GPU CAGRA/cuVS
#'   integration or direct RAPIDS cuVS CAGRA. It supports Euclidean/L2 plus
#'   cosine/correlation through normalized Euclidean graph search; raw inner
#'   product is not exposed [3,13-16].
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
#' @param data Numeric matrix/data frame or sparse `Matrix` object of reference
#'   observations in rows. Sparse inputs use the native exact `dgCMatrix`
#'   backend for `backend = "auto"` or `"cpu"`.
#' @param points Numeric matrix/data frame or sparse `Matrix` object of query
#'   observations in rows. Defaults to `data`.
#' @param k Number of neighbors to return. `NULL` chooses the package's
#'   automatic neighborhood size and includes the self-neighbor when `points`
#'   is `data`.
#' @param backend Device backend: `"auto"`, `"cpu"`, or `"cuda"`. `"auto"`
#'   uses a validated CUDA route only when the requested method/metric
#'   combination is supported, and otherwise resolves to CPU. Explicit
#'   `"cuda"` fails clearly when CUDA support or the selected CUDA combination
#'   is unavailable.
#' @param method Algorithm selector. `"auto"` chooses a shape-aware default for
#'   the selected backend. Other values include `"exact"`, `"flat"`,
#'   `"bruteforce"`, `"grid"`, `"vptree"`, `"sparse"`, `"hnsw"`, `"ivf"`,
#'   `"ivfpq"`, `"nsg"`, `"nndescent"`, and `"cagra"`. Older uppercase
#'   spellings are accepted as compatibility aliases but are not listed as
#'   separate methods. Unsupported backend/method combinations fail clearly;
#'   for example,
#'   `method = "cagra", backend = "cpu"` errors because CAGRA is CUDA-only.
#' @param metric Distance metric. The intentionally small public set is
#'   `"euclidean"`, `"cosine"`, `"correlation"`, and `"inner_product"`.
#'   `"euclidean"` is the validated high-performance default. `"cosine"` and
#'   `"correlation"` are implemented for exact CPU KNN, FAISS CPU/GPU Flat,
#'   FAISS CPU/GPU IVF-Flat, FAISS CPU/GPU IVFPQ, FAISS CPU HNSW,
#'   and RcppHNSW. FAISS approximate IP-capable routes use row L2 normalization
#'   for cosine and row centering plus L2 normalization for correlation before
#'   inner-product search; distances are returned as `1 - similarity`.
#'   CPU `method = "auto"` can use FAISS Flat for larger exact non-Euclidean
#'   query workloads, FAISS HNSW for large non-Euclidean self-search when FAISS
#'   is available, and RcppHNSW/hnswlib only as the fallback when FAISS is
#'   unavailable. CPU `method = "hnsw"` uses FAISS HNSW for all metrics when
#'   available and RcppHNSW/hnswlib when FAISS is unavailable.
#'   `"inner_product"` is exact on native CPU routes and maps to FAISS Flat IP,
#'   FAISS IVF-Flat/IVFPQ IP, and FAISS HNSW IP where available.
#'   Unsupported backend combinations fail clearly instead of returning neighbours
#'   computed under a different metric.
#' @param tuning Tuning policy for approximate GPU methods. `"auto"` uses the
#'   tuned default for the resolved method, `"cache"` reuses/stores pilot
#'   results, `"pilot"` tunes for this call without persisting, `"fixed"` uses
#'   fixed defaults with tuning metadata, and `"off"`/`"none"` disables tuning.
#'   Advanced tuning and cache knobs use `options(faissR.<name> = ...)`.
#'   Legacy `fastEmbedR.<name>` option keys are still accepted as fallbacks, but
#'   `faissR.*` takes precedence.
#' @param n_threads Number of CPU worker threads for CPU backends. GPU backends
#'   ignore this argument.
#' @return A list with integer matrix `indices` and numeric matrix `distances`.
#'   Indices are 1-based. The resolved backend, metric, exact/approximate flag,
#'   and self-query flag are stored as attributes.
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
               method = c("auto", "exact", "flat", "bruteforce", "grid", "vptree",
                          "sparse", "hnsw", "ivf", "ivfpq", "nsg", "nndescent", "cagra"),
               metric = c("euclidean", "cosine", "correlation", "inner_product"),
               tuning = c("auto", "cache", "pilot", "fixed", "off", "none"),
               n_threads = NULL) {
  backend <- normalize_public_backend_arg(backend)
  method <- as.character(method)[1L]
  tuning <- as.character(tuning)[1L]
  metric <- match.arg(metric)
  resolved_backend <- resolve_public_nn_backend(backend, method, metric)
  nn_compute(
    data,
    points,
    k,
    resolved_backend,
    missing(points),
    exclude_self = FALSE,
    n_threads = n_threads,
    metric = metric,
    tuning = tuning
  )
}

#' Nearest neighbours excluding the self match
#'
#' `nn_without_self()` is a convenience wrapper around `nn()` for the common
#' case where the reference and query data are the same matrix and the
#' self-neighbour should be removed. It returns exactly `k` non-self neighbours
#' per observation.
#'
#' @param data Numeric matrix/data frame or sparse `Matrix` object with
#'   observations in rows.
#' @param k Number of non-self neighbours to return.
#' @param backend Device backend: `"auto"`, `"cpu"`, or `"cuda"`. `"auto"`
#'   follows \code{\link{nn}()} backend/method/metric resolution, using CUDA only for
#'   validated CUDA combinations and CPU otherwise.
#' @param method Algorithm selector passed through the same resolver as
#'   \code{\link{nn}()}. See \code{\link{nn}()} for method descriptions and
#'   references.
#' @param metric Distance metric: `"euclidean"`, `"cosine"`, `"correlation"`,
#'   or `"inner_product"`. See \code{\link{nn}()} for metric/backend support
#'   details, including metric-aware CPU HNSW routing.
#' @param tuning Tuning policy passed to \code{\link{nn}()}. `"auto"` uses the
#'   tuned default for the resolved method.
#' @param n_threads Number of CPU worker threads used by CPU backends.
#' @return A `faissR_nn` object with `indices` and `distances` matrices.
#' @export
nn_without_self <- function(data,
                            k,
                            backend = c("auto", "cpu", "cuda"),
                            method = c("auto", "exact", "flat", "bruteforce", "grid", "vptree",
                                       "sparse", "hnsw", "ivf", "ivfpq", "nsg", "nndescent", "cagra"),
                            metric = c("euclidean", "cosine", "correlation", "inner_product"),
                            tuning = c("auto", "cache", "pilot", "fixed", "off", "none"),
                            n_threads = NULL) {
  backend <- normalize_public_backend_arg(backend)
  method <- as.character(method)[1L]
  tuning <- as.character(tuning)[1L]
  metric <- match.arg(metric)
  resolved_backend <- resolve_public_nn_backend(backend, method, metric)
  nn_compute(
    data,
    data,
    k,
    resolved_backend,
    TRUE,
    exclude_self = TRUE,
    n_threads = n_threads,
    metric = metric,
    tuning = tuning
  )
}

.knn_recall_summary <- function(approx, exact, k = NULL) {
  approx_idx <- if (is.list(approx)) approx$indices else approx
  exact_idx <- if (is.list(exact)) exact$indices else exact
  approx_idx <- as.matrix(approx_idx)
  exact_idx <- as.matrix(exact_idx)
  if (nrow(approx_idx) != nrow(exact_idx)) {
    stop("Approximate and exact KNN must have the same number of rows.", call. = FALSE)
  }
  k <- if (is.null(k)) min(ncol(approx_idx), ncol(exact_idx)) else as.integer(k)
  if (length(k) != 1L || is.na(k) || !is.finite(k) || k < 1L) {
    stop("`k` must be a positive integer.", call. = FALSE)
  }
  k <- min(k, ncol(approx_idx), ncol(exact_idx))
  recall <- numeric(nrow(approx_idx))
  for (i in seq_len(nrow(approx_idx))) {
    recall[[i]] <- mean(approx_idx[i, seq_len(k)] %in% exact_idx[i, seq_len(k)])
  }
  data.frame(
    k = k,
    recall_at_k = mean(recall),
    median_recall_at_k = median(recall),
    min_recall_at_k = min(recall),
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
