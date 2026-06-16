nn_compute <- function(data,
                       points,
                       k,
                       backend,
                       points_missing,
                       exclude_self = FALSE,
                       n_threads = NULL,
                       metric = "euclidean") {
  data_sparse <- is_sparse_matrix_input(data)
  points_sparse <- if (isTRUE(points_missing)) data_sparse else is_sparse_matrix_input(points)
  if (isTRUE(data_sparse) || isTRUE(points_sparse)) {
    sparse_native <- backend %in% c("auto", "cpu", "sparse", "cpu_sparse", "sparse_cpu")
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
  if (!identical(metric, "euclidean")) {
    if (identical(backend, "auto")) {
      backend <- "cpu"
    } else if (!backend %in% c("cpu", "hnsw", "rcpphnsw", "cpu_hnsw")) {
      stop(
        "`metric = \"", metric, "\"` currently supports only `backend = \"cpu\"` ",
        "or `backend = \"hnsw\"`. ",
        "The FAISS, CUDA, and cuVS KNN paths in this build ",
        "have validated Euclidean-distance semantics only.",
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

  auto_gpu <- resolve_auto_knn_gpu_backend(
    backend = backend,
    self_query = self_query,
    n_points = nrow(points),
    n = nrow(data),
    k = k,
    work_size = work_size
  )
  if (!is.na(auto_gpu)) {
    backend <- auto_gpu
  } else if (backend %in% c("cuda", "gpu") && isTRUE(self_query)) {
    backend <- select_cuda_auto_backend(
      self_query = self_query,
      n = nrow(data),
      n_points = nrow(points),
      k = k,
      work_size = work_size
    )
  } else if (backend %in% c("cuda", "gpu") && !isTRUE(self_query)) {
    backend <- select_cuda_auto_backend(
      self_query = self_query,
      n = nrow(data),
      n_points = nrow(points),
      k = k,
      work_size = work_size
    )
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
    out <- nn_faiss_ivf_cpp(
      data,
      points,
      as.integer(k),
      as.integer(params$nlist),
      as.integer(params$nprobe),
      isTRUE(exclude_self),
      as.integer(n_threads)
    )
    result <- finish_nn_result(out, "faiss_ivf", k, self_query, exact = FALSE)
    attr(result, "approximation") <- list(
      strategy = "faiss_IndexIVFFlat",
      backend = "faiss_ivf",
      library = "faiss",
      nlist = as.integer(out$nlist),
      nprobe = as.integer(out$nprobe)
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
    out <- nn_faiss_ivfpq_cpp(
      data,
      points,
      as.integer(k),
      as.integer(params$nlist),
      as.integer(params$nprobe),
      as.integer(pq$m),
      as.integer(pq$nbits),
      isTRUE(exclude_self),
      as.integer(n_threads)
    )
    result <- finish_nn_result(out, "faiss_ivfpq", k, self_query, exact = FALSE)
    attr(result, "approximation") <- list(
      strategy = "faiss_IndexIVFPQ",
      backend = "faiss_ivfpq",
      library = "faiss",
      nlist = as.integer(out$nlist),
      nprobe = as.integer(out$nprobe),
      pq_m = as.integer(out$pq_m),
      pq_nbits = as.integer(out$pq_nbits)
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
    tuning <- NULL
    if (isTRUE(faiss_gpu_ivf_should_tune(data, k, self_query))) {
      tuned <- faiss_gpu_ivf_tune_params(data, k, params)
      params <- tuned$params
      tuning <- tuned$tuning
    }
    out <- nn_faiss_gpu_ivf_flat_cpp(
      data,
      points,
      as.integer(k),
      as.integer(params$nlist),
      as.integer(params$nprobe),
      isTRUE(exclude_self)
    )
    result <- finish_nn_result(out, "faiss_gpu_ivf_flat", k, self_query, exact = FALSE)
    attr(result, "approximation") <- list(
      strategy = "faiss_gpu_IndexIVFFlat_cuVS",
      backend = "faiss_gpu_ivf_flat",
      library = "faiss",
      accelerator = "cuda",
      nlist = as.integer(out$nlist),
      nprobe = as.integer(out$nprobe),
      tuning = tuning
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
    out <- nn_faiss_gpu_ivfpq_cpp(
      data,
      points,
      as.integer(k),
      as.integer(params$nlist),
      as.integer(params$nprobe),
      as.integer(pq$m),
      as.integer(pq$nbits),
      isTRUE(exclude_self)
    )
    result <- finish_nn_result(out, "faiss_gpu_ivfpq", k, self_query, exact = FALSE)
    attr(result, "approximation") <- list(
      strategy = "faiss_gpu_IndexIVFPQ_cuVS",
      backend = "faiss_gpu_ivfpq",
      library = "faiss",
      accelerator = "cuda",
      role = "explicit_memory_pressure_backend",
      default_candidate = FALSE,
      nlist = as.integer(out$nlist),
      nprobe = as.integer(out$nprobe),
      pq_m = as.integer(out$pq_m),
      pq_nbits = as.integer(out$pq_nbits)
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
    params <- cuvs_cagra_params(nrow(data), k)
    out <- nn_faiss_gpu_cagra_cpp(
      data,
      points,
      as.integer(k),
      as.integer(params$graph_degree),
      as.integer(params$intermediate_graph_degree),
      as.integer(params$search_width),
      as.integer(params$itopk_size),
      isTRUE(exclude_self)
    )
    result <- finish_nn_result(out, "faiss_gpu_cagra", k, self_query, exact = FALSE)
    attr(result, "approximation") <- list(
      strategy = "faiss_gpu_GpuIndexCagra_cuVS",
      backend = "faiss_gpu_cagra",
      library = "faiss",
      accelerator = "cuda",
      graph_degree = as.integer(out$graph_degree),
      intermediate_graph_degree = as.integer(out$intermediate_graph_degree),
      search_width = as.integer(out$search_width),
      itopk_size = as.integer(out$itopk_size)
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
    params <- faiss_hnsw_params(k)
    out <- nn_faiss_hnsw_cpp(
      data,
      points,
      as.integer(k),
      as.integer(params$m),
      as.integer(params$ef_construction),
      as.integer(params$ef_search),
      isTRUE(exclude_self),
      as.integer(n_threads)
    )
    result <- finish_nn_result(out, "faiss_hnsw", k, self_query, exact = FALSE)
    attr(result, "approximation") <- list(
      strategy = "faiss_IndexHNSWFlat",
      backend = "faiss_hnsw",
      library = "faiss",
      m = as.integer(params$m),
      ef_construction = as.integer(params$ef_construction),
      ef_search = as.integer(params$ef_search)
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
    params <- faiss_nsg_params(k)
    out <- nn_faiss_nsg_cpp(
      data,
      points,
      as.integer(k),
      as.integer(params$r),
      as.integer(params$search_l),
      as.integer(params$build_type),
      isTRUE(exclude_self),
      as.integer(n_threads)
    )
    result <- finish_nn_result(out, "faiss_nsg", k, self_query, exact = FALSE)
    attr(result, "approximation") <- list(
      strategy = "faiss_IndexNSGFlat",
      backend = "faiss_nsg",
      library = "faiss",
      r = as.integer(params$r),
      search_l = as.integer(params$search_l),
      build_type = as.integer(params$build_type)
    )
    return(result)
  }

  if (identical(backend, "faiss_nndescent")) {
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
      isTRUE(exclude_self),
      as.integer(n_threads)
    )
    result <- finish_nn_result(out, "faiss_nndescent", k, self_query, exact = FALSE)
    attr(result, "approximation") <- list(
      strategy = "faiss_IndexNNDescentFlat",
      backend = "faiss_nndescent",
      library = "faiss",
      graph_k = as.integer(params$graph_k),
      n_iter = as.integer(params$n_iter),
      search_l = as.integer(params$search_l)
    )
    return(result)
  }

  if (backend %in% c("cuvs", "gpu_cuvs", "cuda_cuvs")) {
    require_cuvs_backend("cuVS")
    backend <- select_cuvs_auto_backend(
      self_query = self_query,
      n = nrow(data),
      n_points = nrow(points),
      k = k,
      work_size = work_size
    )
  }

  if (backend %in% c("cuda_cuvs_cagra", "cuda_cagra", "gpu_cagra")) {
    require_cuvs_backend("cuVS CAGRA")
    params <- cuvs_cagra_params(nrow(data), k)
    tuning <- NULL
    if (isTRUE(cuvs_cagra_should_tune(data, k, self_query))) {
      tuned <- cuvs_cagra_tune_params(data, k, params)
      params <- tuned$params
      tuning <- tuned$tuning
    }
    out <- nn_cuvs_cagra_cpp(
      data,
      points,
      as.integer(k),
      isTRUE(exclude_self),
      as.integer(params$graph_degree),
      as.integer(params$intermediate_graph_degree),
      as.integer(params$search_width),
      as.integer(params$itopk_size)
    )
    result <- finish_nn_result(out, "cuda_cuvs_cagra", k, self_query, exact = FALSE)
    attr(result, "approximation") <- list(
      strategy = "rapids_cuvs_cagra",
      backend = "cuda_cuvs_cagra",
      library = "cuvs",
      graph_degree = as.integer(out$graph_degree),
      intermediate_graph_degree = as.integer(out$intermediate_graph_degree),
      search_width = as.integer(out$search_width),
      itopk_size = as.integer(out$itopk_size),
      search_batch_size = as.integer(out$search_batch_size),
      tuning = tuning
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
      pq_dim = as.integer(out$pq_dim),
      pq_bits = as.integer(out$pq_bits),
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
    result <- finish_nn_result(out, "cuda_cuvs_bruteforce", k, self_query, exact = TRUE)
    attr(result, "cuvs") <- list(
      index_type = as.character(out$index_type),
      library = "cuvs",
      backend = "cuda"
    )
    return(result)
  }

  if (backend %in% c("cuvs_nndescent", "cuda_cuvs_nndescent")) {
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
    result <- finish_nn_result(out, "cuda_cuvs_nndescent", k, self_query, exact = FALSE)
    attr(result, "approximation") <- list(
      strategy = "rapids_cuvs_nndescent",
      backend = "cuda_cuvs_nndescent",
      library = "cuvs",
      graph_degree = as.integer(out$graph_degree),
      intermediate_graph_degree = as.integer(out$intermediate_graph_degree),
      max_iterations = as.integer(out$max_iterations)
    )
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
    if (!identical(metric, "euclidean")) {
      stop("`backend = \"cpu_vptree\"` supports only Euclidean distances.", call. = FALSE)
    }
    if (isTRUE(self_query)) {
      out <- vptree_self_knn(data, k = if (isTRUE(exclude_self)) k else k - 1L, n_threads = n_threads)
      if (!isTRUE(exclude_self)) {
        out$indices <- cbind(seq_len(nrow(data)), out$indices)
        out$distances <- cbind(rep(0, nrow(data)), out$distances)
      }
    } else {
      out <- vptree_query_knn(data, points, k = k, n_threads = n_threads)
    }
    result <- finish_nn_result(out, "cpu_vptree", k, self_query, exact = TRUE, metric = metric)
    attr(result, "spatial_index") <- list(
      strategy = "native_exact_vptree",
      backend = "cpu_vptree",
      exact = TRUE,
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
                                         k,
                                         work_size) {
  if (!identical(backend, "auto")) return(NA_character_)
  if (!isTRUE(self_query)) return(NA_character_)
  if (k > 256L) return(NA_character_)
  if (work_size < 5e8) return(NA_character_)
  if (isTRUE(cuvs_available())) {
    return(select_cuvs_auto_backend(
      self_query = self_query,
      n = n,
      n_points = n_points,
      k = k,
      work_size = work_size
    ))
  }
  NA_character_
}

select_cuda_auto_backend <- function(self_query,
                                     n,
                                     n_points,
                                     k,
                                     work_size) {
  if (isTRUE(cuvs_available())) {
    return(select_cuvs_auto_backend(
      self_query = self_query,
      n = n,
      n_points = n_points,
      k = k,
      work_size = work_size
    ))
  }
  if (!isTRUE(cuda_available())) {
    stop("No CUDA GPU backend is available on this machine.", call. = FALSE)
  }
  "cuda"
}

select_cuvs_auto_backend <- function(self_query,
                                     n,
                                     n_points,
                                     k,
                                     work_size) {
  small_threshold <- getOption("fastEmbedR.cuvs_bruteforce_work_threshold", 2e8)
  small_threshold <- suppressWarnings(as.numeric(small_threshold))
  if (length(small_threshold) != 1L || is.na(small_threshold) || !is.finite(small_threshold)) {
    small_threshold <- 2e8
  }
  if (work_size <= small_threshold || k <= 8L || n <= 5000L || n_points <= 5000L) {
    return("cuda_cuvs_bruteforce")
  }
  "cuda_cagra"
}

should_auto_use_exact_metal_knn <- function(n) {
  threshold <- getOption("fastEmbedR.metal_exact_auto_threshold", 15000L)
  threshold <- suppressWarnings(as.integer(threshold))
  if (length(threshold) != 1L || is.na(threshold) || !is.finite(threshold)) {
    threshold <- 15000L
  }
  as.integer(n) <= max(1L, threshold)
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
    return("faiss_nndescent")
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
  match.arg(metric, c("euclidean", "cosine", "correlation"))
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
  value <- getOption(sprintf("fastEmbedR.grid%dd_bins_per_dim", p), NULL)
  if (is.null(value)) value <- getOption("fastEmbedR.grid_bins_per_dim", NULL)
  if (!is.null(value)) {
    value <- suppressWarnings(as.integer(value))
    if (length(value) == 1L && is.finite(value) && !is.na(value) && value > 0L) {
      return(as.integer(value))
    }
  }
  target_occupancy <- getOption(sprintf("fastEmbedR.grid%dd_target_occupancy", p), NULL)
  if (is.null(target_occupancy)) target_occupancy <- getOption("fastEmbedR.grid_target_occupancy", NULL)
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
  if (!isTRUE(getOption("fastEmbedR.cpu_spatial_auto", TRUE))) {
    return(if (p == 3L) "cpu_grid3d" else "cpu_grid2d")
  }
  n <- nrow(data)
  sample_n <- min(n, as.integer(getOption("fastEmbedR.cpu_spatial_sample", 4096L)))
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
  anisotropy_threshold <- as.numeric(getOption("fastEmbedR.cpu_spatial_anisotropy_threshold", 0.02))
  if (!is.finite(anisotropy_threshold) || anisotropy_threshold <= 0) anisotropy_threshold <- 0.02
  if (is.finite(anisotropy) && anisotropy < anisotropy_threshold) {
    out <- "cpu_vptree"
    attr(out, "reason") <- sprintf("anisotropic_sample_sd_ratio_%.4g", anisotropy)
    return(out)
  }

  unique_sample <- nrow(unique(round(xs, digits = 12L)))
  duplicate_ratio <- unique_sample / sample_n
  duplicate_threshold <- as.numeric(getOption("fastEmbedR.cpu_spatial_duplicate_threshold", 0.05))
  if (!is.finite(duplicate_threshold) || duplicate_threshold <= 0) duplicate_threshold <- 0.05
  if (is.finite(duplicate_ratio) && duplicate_ratio <= duplicate_threshold) {
    out <- if (p == 3L) "cpu_grid3d" else "cpu_grid2d"
    attr(out, "reason") <- sprintf("duplicate_heavy_sample_unique_ratio_%.4g", duplicate_ratio)
    return(out)
  }

  sample_bins <- as.integer(getOption("fastEmbedR.cpu_spatial_sample_bins", if (p == 3L) 16L else 32L))
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
  imbalance_threshold <- as.numeric(getOption("fastEmbedR.cpu_spatial_imbalance_threshold", 20))
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
  value <- getOption("fastEmbedR.approx_knn_seed", 4L)
  value <- suppressWarnings(as.integer(value))
  if (length(value) != 1L || is.na(value) || !is.finite(value)) 4L else value
}

gpu_approx_params <- function(n,
                              k,
                              backend = NULL,
                              label = NULL) {
  n <- as.integer(n)
  k <- as.integer(k)
  anchors <- getOption("fastEmbedR.gpu_approx_anchors", NULL)
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
  projection_k <- getOption("fastEmbedR.gpu_approx_projection_k", NULL)
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
  if (isTRUE(getOption("fastEmbedR.gpu_approx_recall", FALSE))) {
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
    sprintf("fastEmbedR.%s_nndescent_%s", backend, name),
    sprintf("fastEmbedR.gpu_nndescent_%s", name)
  )
  for (key in keys) {
    value <- getOption(key, NULL)
    if (!is.null(value)) return(value)
  }
  default
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
      # Native port of the mlx-vis NN-descent schedule: expand aggressively
      # while the graph is moving, then use only NEW-neighbour sources and
      # skip reverse candidates near convergence.
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
      candidate_indices <- nndescent_candidate_matrix_mlx_cpp(
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
    strategy = paste0("mlx_vis_adaptive_seeded_nndescent_native_", backend),
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
  value <- getOption("fastEmbedR.gpu_approx_recall_sample", 512L)
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
  recall <- knn_recall(
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
  isTRUE(getOption("fastEmbedR.cpu_nndescent_prefer_faiss", TRUE)) &&
    isTRUE(faiss_available())
}

cpu_nndescent_faiss_index <- function() {
  value <- tolower(as.character(getOption("fastEmbedR.cpu_nndescent_faiss_index", "hnsw"))[1L])
  if (!value %in% c("hnsw", "ivf", "flat", "nsg", "nndescent")) {
    warning(
      "Option `fastEmbedR.cpu_nndescent_faiss_index` must be one of ",
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
      TRUE,
      as.integer(n_threads)
    )
    attr(out, "approximation") <- list(
      strategy = "faiss_IndexHNSWFlat_self",
      backend = backend,
      library = "faiss",
      exact = FALSE,
      m = as.integer(params$m),
      ef_construction = as.integer(params$ef_construction),
      ef_search = as.integer(params$ef_search),
      seed = as.integer(seed)
    )
    return(out)
  }
  if (identical(backend, "faiss_nsg") ||
      identical(backend, "cpu_nndescent_faiss_nsg")) {
    params <- faiss_nsg_params(k)
    out <- nn_faiss_nsg_cpp(
      data,
      data,
      as.integer(k),
      as.integer(params$r),
      as.integer(params$search_l),
      as.integer(params$build_type),
      TRUE,
      as.integer(n_threads)
    )
    attr(out, "approximation") <- list(
      strategy = "faiss_IndexNSGFlat_self",
      backend = backend,
      library = "faiss",
      exact = FALSE,
      r = as.integer(params$r),
      search_l = as.integer(params$search_l),
      build_type = as.integer(params$build_type),
      seed = as.integer(seed)
    )
    return(out)
  }
  if (identical(backend, "faiss_nndescent") ||
      identical(backend, "cpu_nndescent_faiss_nndescent")) {
    params <- faiss_nndescent_params(k)
    out <- nn_faiss_nndescent_cpp(
      data,
      data,
      as.integer(k),
      as.integer(params$graph_k),
      as.integer(params$n_iter),
      as.integer(params$search_l),
      TRUE,
      as.integer(n_threads)
    )
    attr(out, "approximation") <- list(
      strategy = "faiss_IndexNNDescentFlat_self",
      backend = backend,
      library = "faiss",
      exact = FALSE,
      graph_k = as.integer(params$graph_k),
      n_iter = as.integer(params$n_iter),
      search_l = as.integer(params$search_l),
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
  m <- getOption("fastEmbedR.hnsw_m", 16L)
  ef_construction <- getOption("fastEmbedR.hnsw_ef_construction", 200L)
  ef <- getOption("fastEmbedR.hnsw_ef", max(50L, 3L * as.integer(k)))
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
    correlation = "cosine"
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
  value <- getOption("fastEmbedR.annoy_n_trees", NULL)
  if (is.null(value)) {
    value <- if (n >= 10000L) 50L else 24L
  }
  value <- suppressWarnings(as.integer(value))
  if (length(value) != 1L || is.na(value) || !is.finite(value)) value <- 24L
  as.integer(max(1L, min(128L, value)))
}

annoy_leaf_size <- function(k) {
  value <- getOption("fastEmbedR.annoy_leaf_size", NULL)
  if (is.null(value)) {
    value <- max(64L, min(256L, ceiling(2L * k)))
  }
  value <- suppressWarnings(as.integer(value))
  if (length(value) != 1L || is.na(value) || !is.finite(value)) value <- max(64L, 2L * k)
  as.integer(max(k + 1L, min(1024L, value)))
}

annoy_search_k <- function(k, n_trees, leaf_size) {
  value <- getOption("fastEmbedR.annoy_search_k", NULL)
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
  as.integer(max(1L, min(nlist, max(4L, ceiling(sqrt(nlist)), ceiling(k / 5)))))
}

faiss_ivf_params <- function(n, k) {
  n <- as.integer(n)
  k <- as.integer(k)
  nlist <- getOption("fastEmbedR.faiss_nlist", NULL)
  if (is.null(nlist)) nlist <- getOption("fastEmbedR.ivf_nlist", NULL)
  nprobe <- getOption("fastEmbedR.faiss_nprobe", NULL)
  if (is.null(nprobe)) nprobe <- getOption("fastEmbedR.ivf_nprobe", NULL)

  nlist <- if (is.null(nlist)) ivf_list_count(n, k) else suppressWarnings(as.integer(nlist))
  if (length(nlist) != 1L || is.na(nlist) || !is.finite(nlist)) {
    nlist <- ivf_list_count(n, k)
  }
  nlist <- max(1L, min(n, nlist))

  nprobe <- if (is.null(nprobe)) ivf_probe_count(nlist, k) else suppressWarnings(as.integer(nprobe))
  if (length(nprobe) != 1L || is.na(nprobe) || !is.finite(nprobe)) {
    nprobe <- ivf_probe_count(nlist, k)
  }
  nprobe <- max(1L, min(nlist, nprobe))
  list(nlist = as.integer(nlist), nprobe = as.integer(nprobe))
}

faiss_ivf_manual_params <- function() {
  any(vapply(
    c("faiss_nlist", "ivf_nlist", "faiss_nprobe", "ivf_nprobe"),
    function(name) !is.null(getOption(paste0("fastEmbedR.", name), NULL)),
    logical(1)
  ))
}

cuvs_ivfpq_params <- function(p) {
  pq_dim <- getOption("fastEmbedR.cuvs_ivfpq_pq_dim", NULL)
  if (is.null(pq_dim)) pq_dim <- getOption("fastEmbedR.ivfpq_pq_dim", 0L)
  pq_dim <- suppressWarnings(as.integer(pq_dim))
  if (length(pq_dim) != 1L || is.na(pq_dim) || !is.finite(pq_dim) || pq_dim < 0L) {
    pq_dim <- 0L
  }

  pq_bits <- getOption("fastEmbedR.cuvs_ivfpq_pq_bits", NULL)
  if (is.null(pq_bits)) pq_bits <- getOption("fastEmbedR.ivfpq_pq_bits", 8L)
  pq_bits <- suppressWarnings(as.integer(pq_bits))
  if (length(pq_bits) != 1L || is.na(pq_bits) || !is.finite(pq_bits)) {
    pq_bits <- 8L
  }
  pq_bits <- as.integer(max(4L, min(8L, pq_bits)))

  list(pq_dim = as.integer(pq_dim), pq_bits = pq_bits)
}

.faiss_gpu_ivf_tune_cache <- new.env(parent = emptyenv())
.faiss_gpu_ivf_tune_disk_cache <- new.env(parent = emptyenv())
.faiss_gpu_ivf_tune_disk_cache$loaded <- FALSE
.faiss_gpu_ivf_tune_disk_cache$file <- NULL
.faiss_gpu_ivf_tune_disk_cache$entries <- list()

faiss_gpu_ivf_tune_policy <- function() {
  policy <- getOption("fastEmbedR.faiss_gpu_ivf_tune_policy", "cache")
  policy <- tolower(as.character(policy)[1L])
  if (!policy %in% c("cache", "pilot", "fixed", "off")) {
    policy <- "cache"
  }
  policy
}

faiss_gpu_ivf_tune_cache_file <- function() {
  path <- getOption("fastEmbedR.faiss_gpu_ivf_tune_cache_file", NULL)
  if (!is.null(path)) return(path)
  root <- tryCatch(
    tools::R_user_dir("fastEmbedR", which = "cache"),
    error = function(e) file.path(tempdir(), "fastEmbedR-cache")
  )
  file.path(root, "faiss_gpu_ivf_tuning.rds")
}

faiss_gpu_ivf_should_tune <- function(data, k, self_query) {
  if (!isTRUE(self_query)) return(FALSE)
  if (!isTRUE(faiss_available())) return(FALSE)
  if (isTRUE(faiss_ivf_manual_params())) return(FALSE)
  enabled <- getOption("fastEmbedR.faiss_gpu_ivf_tune", TRUE)
  if (!isTRUE(enabled)) return(FALSE)
  threshold <- getOption("fastEmbedR.faiss_gpu_ivf_tune_threshold", 20000L)
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

faiss_gpu_ivf_tune_params <- function(data, k, base_params) {
  policy <- faiss_gpu_ivf_tune_policy()
  if (identical(policy, "off")) {
    return(list(
      params = base_params,
      tuning = list(status = "disabled", policy = policy)
    ))
  }
  sample_size <- getOption("fastEmbedR.faiss_gpu_ivf_tune_sample", 10000L)
  sample_size <- suppressWarnings(as.integer(sample_size))
  if (length(sample_size) != 1L || is.na(sample_size) || !is.finite(sample_size) || sample_size < 1000L) {
    sample_size <- 10000L
  }
  sample_size <- as.integer(min(nrow(data), sample_size))
  seed <- getOption("fastEmbedR.faiss_gpu_ivf_tune_seed", 7L)
  seed <- suppressWarnings(as.integer(seed))
  if (length(seed) != 1L || is.na(seed) || !is.finite(seed)) seed <- 7L
  target <- getOption("fastEmbedR.faiss_gpu_ivf_tune_recall", 0.985)
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
      recall <- knn_recall(approx, reference, k = compare_k)$recall_at_k
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
  value <- getOption(paste0("fastEmbedR.faiss_", name), NULL)
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
    max(200L, 4L * m),
    min_value = m,
    max_value = 4096L
  )
  ef_search <- faiss_option_int(
    "hnsw_ef_search",
    max(100L, 2L * k),
    min_value = k,
    max_value = 4096L
  )
  list(m = as.integer(m), ef_construction = as.integer(ef_construction), ef_search = as.integer(ef_search))
}

faiss_nsg_params <- function(k) {
  k <- as.integer(k)
  r <- faiss_option_int("nsg_r", 32L, min_value = 2L, max_value = 512L)
  search_l <- faiss_option_int("nsg_search_l", max(100L, 2L * k), min_value = k, max_value = 4096L)
  build_type <- faiss_option_int("nsg_build_type", 1L, min_value = 0L, max_value = 1L)
  list(r = as.integer(r), search_l = as.integer(search_l), build_type = as.integer(build_type))
}

faiss_nndescent_params <- function(k) {
  k <- as.integer(k)
  graph_k <- faiss_option_int("nndescent_graph_k", max(k, 64L), min_value = k, max_value = 1024L)
  n_iter <- faiss_option_int("nndescent_iter", 10L, min_value = 1L, max_value = 100L)
  search_l <- faiss_option_int("nndescent_search_l", max(k, graph_k), min_value = k, max_value = 4096L)
  list(graph_k = as.integer(graph_k), n_iter = as.integer(n_iter), search_l = as.integer(search_l))
}

cuvs_option_int <- function(name, default, min_value = 1L, max_value = .Machine$integer.max) {
  value <- getOption(paste0("fastEmbedR.cuvs_", name), NULL)
  value <- if (is.null(value)) default else suppressWarnings(as.integer(value))
  if (length(value) != 1L || is.na(value) || !is.finite(value)) value <- default
  as.integer(max(min_value, min(max_value, value)))
}

cuvs_cagra_params <- function(n, k) {
  n <- as.integer(n)
  k <- as.integer(k)
  graph_degree <- cuvs_option_int(
    "graph_degree",
    default = max(64L, k + 1L),
    min_value = k + 1L,
    max_value = max(1L, n - 1L)
  )
  intermediate_graph_degree <- cuvs_option_int(
    "intermediate_graph_degree",
    default = max(128L, graph_degree * 2L),
    min_value = graph_degree,
    max_value = max(1L, n - 1L)
  )
  search_width <- cuvs_option_int(
    "search_width",
    default = 0L,
    min_value = 0L,
    max_value = 1024L
  )
  itopk_size <- cuvs_option_int(
    "itopk_size",
    default = max(64L, graph_degree),
    min_value = k,
    max_value = 4096L
  )
  list(
    graph_degree = as.integer(graph_degree),
    intermediate_graph_degree = as.integer(intermediate_graph_degree),
    search_width = as.integer(search_width),
    itopk_size = as.integer(itopk_size)
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
    function(name) !is.null(getOption(paste0("fastEmbedR.cuvs_", name), NULL)),
    logical(1)
  ))
}

cuvs_cagra_should_tune <- function(data, k, self_query) {
  if (!isTRUE(self_query)) return(FALSE)
  if (!isTRUE(cuvs_available())) return(FALSE)
  if (isTRUE(cuvs_cagra_manual_params())) return(FALSE)
  enabled <- getOption("fastEmbedR.cuvs_cagra_tune", TRUE)
  if (!isTRUE(enabled)) return(FALSE)
  threshold <- getOption("fastEmbedR.cuvs_cagra_tune_threshold", 20000L)
  threshold <- suppressWarnings(as.integer(threshold))
  if (length(threshold) != 1L || is.na(threshold) || !is.finite(threshold)) {
    threshold <- 20000L
  }
  nrow(data) >= threshold && as.integer(k) >= 10L
}

cuvs_cagra_tune_policy <- function() {
  policy <- getOption("fastEmbedR.cuvs_cagra_tune_policy", "cache")
  policy <- tolower(as.character(policy)[1L])
  if (!policy %in% c("cache", "pilot", "fixed", "off")) {
    policy <- "cache"
  }
  policy
}

cuvs_cagra_tune_cache_file <- function() {
  path <- getOption("fastEmbedR.cuvs_cagra_tune_cache_file", NULL)
  if (!is.null(path)) return(path)
  root <- tryCatch(
    tools::R_user_dir("fastEmbedR", which = "cache"),
    error = function(e) file.path(tempdir(), "fastEmbedR-cache")
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

cuvs_cagra_tune_params <- function(data, k, base_params) {
  policy <- cuvs_cagra_tune_policy()
  if (identical(policy, "off")) {
    return(list(
      params = base_params,
      tuning = list(status = "disabled", policy = policy)
    ))
  }
  sample_size <- getOption("fastEmbedR.cuvs_cagra_tune_sample", 2048L)
  sample_size <- suppressWarnings(as.integer(sample_size))
  if (length(sample_size) != 1L || is.na(sample_size) || !is.finite(sample_size) || sample_size < 256L) {
    sample_size <- 2048L
  }
  sample_size <- as.integer(min(nrow(data), sample_size))
  seed <- getOption("fastEmbedR.cuvs_cagra_tune_seed", 4L)
  seed <- suppressWarnings(as.integer(seed))
  if (length(seed) != 1L || is.na(seed) || !is.finite(seed)) seed <- 4L
  target <- getOption("fastEmbedR.cuvs_cagra_tune_recall", 0.985)
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
      recall <- knn_recall(approx, reference, k = compare_k)$recall_at_k
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
  value <- getOption("fastEmbedR.cuvs_nndescent_threshold", 50000L)
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
  nlist <- getOption("fastEmbedR.ivf_nlist", NULL)
  nprobe <- getOption("fastEmbedR.ivf_nprobe", NULL)
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
#' the common `nn(data, points, k)` use case. Explicit `backend = "cpu"`
#' performs exact brute-force search in C++ and is mainly a reference path for
#' small data or recall checks. `backend = "auto"` chooses a recorded fast path
#' for large self-KNN searches: CUDA cuVS NN-Descent when a CUDA backend is
#' requested and cuVS is available, otherwise FAISS NN-Descent when FAISS is
#' compiled in, otherwise the optional CRAN-friendly RcppHNSW backend when
#' installed, otherwise exact CPU. Use `backend = "cpu_approx"` to force the
#' non-CUDA part of that selector. `backend = "cpu_grid"` is a native exact
#' two- or three-dimensional Euclidean self-KNN spatial selector: it uses a
#' fast grid for uniform-like clouds and a VP-tree for thin or imbalanced
#' clouds. `backend = "vptree"`/`"cpu_vptree"` builds an exact VP-tree on
#' `data` and supports both self-KNN and `nn(data, points, k)` query searches.
#' `backend = "cuda_grid_auto"`/`"cuda_grid"` is the CUDA equivalent for exact
#' two- or three-dimensional Euclidean self-KNN: it chooses the native 2D or 3D
#' CUDA grid kernel and errors clearly when CUDA is unavailable.
#'
#' @param data Numeric matrix/data frame or sparse `Matrix` object of reference
#'   observations in rows. Sparse inputs use the native exact `dgCMatrix`
#'   backend for `backend = "auto"` or `"cpu"`.
#' @param points Numeric matrix/data frame or sparse `Matrix` object of query
#'   observations in rows. Defaults to `data`.
#' @param k Number of neighbors to return. `NULL` chooses the package's
#'   automatic neighborhood size and includes the self-neighbor when `points`
#'   is `data`.
#' @param backend Execution backend. `"auto"` records the selected fast backend
#'   in `attr(result, "backend")`. `"cpu"` always uses the exact C++ CPU path.
#'   `"cpu_approx"` chooses FAISS NN-Descent, RcppHNSW, or exact CPU depending
#'   on what is available; for large 2D Euclidean self-KNN it chooses
#'   `"cpu_grid2d"` and for large 3D Euclidean self-KNN it chooses
#'   `"cpu_grid"`. Explicit `"cpu_grid2d"`/`"cpu_grid3d"` force the exact
#'   grid implementation, while `"cpu_vptree"` forces the exact VP-tree for
#'   Euclidean self-KNN or reference/query KNN.
#'   `"cuda_grid_auto"`/`"cuda_grid"` chooses the native CUDA exact 2D or 3D
#'   grid implementation for Euclidean self-KNN only; it does not silently
#'   fall back to CPU.
#'   `"cpu_sparse"`/`"sparse"` forces the native exact sparse `dgCMatrix` CPU
#'   path for Euclidean, cosine, and correlation distances. Explicit dense
#'   accelerator backends convert sparse input to dense matrices because
#'   FAISS/cuVS/Metal kernels operate on dense row vectors.
#'   `"hnsw"`/`"rcpphnsw"` uses the optional CRAN package RcppHNSW.
#'   `"faiss"` uses the real FAISS C++ `IndexFlatL2` backend when faissR was
#'   built against FAISS. `"faiss_ivf"`, `"faiss_ivfpq"`, `"faiss_hnsw"`,
#'   `"faiss_nsg"`, and `"faiss_nndescent"` use the corresponding FAISS index
#'   types when FAISS is available at compile time. `"faiss_gpu_flat_l2"` and
#'   `"faiss_gpu_flat_ip"` use exact FAISS GPU Flat indexes as CUDA references.
#'   `"faiss_gpu_ivf_flat"` and `"faiss_gpu_ivfpq"` use FAISS GPU/cuVS IVF
#'   indexes when the package was built against FAISS GPU headers; they fail
#'   clearly otherwise. `"faiss_gpu_ivfpq"` is explicit-only: it is useful when
#'   GPU memory pressure matters, but it is not used by automatic routing
#'   because product quantization can reduce neighbor recall. For large
#'   self-KNN, `"faiss_gpu_ivf_flat"` uses
#'   recall-aware pilot tuning of `nlist`/`nprobe` against exact FAISS GPU Flat
#'   and caches the chosen values by dataset signature and `k`. Internal option
#'   `fastEmbedR.faiss_gpu_ivf_tune_policy = "fixed"` skips pilot tuning on a
#'   cache miss and uses the current deterministic IVF defaults directly.
#'   `"cuda_cuvs"` uses a RAPIDS-inspired cuVS policy: exact cuVS brute force
#'   for small searches and cuVS NN-descent for large self-KNN. Use
#'   `"cuda_cagra"`/`"cuda_cuvs_cagra"` to force cuVS CAGRA,
#'   `"cuda_cuvs_bruteforce"` for
#'   exact cuVS brute-force search, and `"cuda_cuvs_nndescent"` for cuVS
#'   NN-descent self-KNN. `"gpu"` is a CUDA-only convenience alias for KNN now.
#'   For large CUDA self-KNN, `"cuda"`/`"gpu"` prefer cuVS CAGRA with batched
#'   search and internal recall-aware pilot tuning. The CAGRA pilot is cached
#'   on disk by dataset signature and `k`, so repeated benchmark runs reuse the
#'   same tuned graph parameters instead of rerunning the pilot. Internal option
#'   `fastEmbedR.cuvs_cagra_tune_policy = "fixed"` skips a cache miss and uses
#'   the current default CAGRA policy directly. `"cuda_cuvs_ivf_flat"` and
#'   `"cuda_cuvs_ivfpq"` call RAPIDS cuVS IVF-Flat and IVF-PQ directly through
#'   the cuVS C API when those headers are available. They are explicit
#'   benchmarking/memory-control backends; automatic CUDA routing continues to
#'   prefer cuVS CAGRA/NN-descent. `"cuda_cuvs_ivfpq"` is explicit-only because
#'   product quantization can lower recall.
#'   `"cuda_nndescent"` and `"cuda_approx"` are
#'   compatibility aliases for
#'   `"cuda_cuvs_nndescent"`; CUDA approximate KNN is routed through RAPIDS
#'   cuVS NN-descent rather than the removed native CUDA NN-descent branch.
#'   `"cuda_ivf"` and `"cuda_faiss"` request the package's native CUDA
#'   IVF/FAISS-style IVF-flat search and fail clearly if CUDA is unavailable.
#' @param metric Distance metric. The intentionally small public set is
#'   `"euclidean"`, `"cosine"`, and `"correlation"`. `"euclidean"` is the
#'   validated high-performance default. `"cosine"` and `"correlation"` are
#'   implemented for exact CPU KNN and RcppHNSW; with `backend = "auto"` they
#'   select the exact CPU path. FAISS, CUDA, and cuVS backends fail clearly for
#'   non-Euclidean metrics instead of returning Euclidean neighbours under a
#'   different label.
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
#' @export
nn <- function(data,
               points = data,
               k = NULL,
               backend = c(
                 "auto", "cpu", "cpu_approx", "cpu_grid", "grid",
                 "cpu_sparse", "sparse", "sparse_cpu",
                 "cpu_grid2d", "grid2d", "cpu_grid3d", "grid3d",
                 "vptree", "cpu_vptree", "hnsw", "rcpphnsw", "cpu_hnsw",
                 "faiss", "cpu_faiss", "cpu_faiss_flat", "faiss_flat", "faiss_flat_l2",
                 "faiss_flat_ip", "faiss_ivf", "faiss_ivf_flat", "cpu_faiss_index_ivf",
                 "faiss_ivfpq", "faiss_hnsw", "faiss_nsg", "faiss_nndescent",
                 "faiss_gpu_flat", "faiss_gpu_flat_l2", "faiss_gpu_flat_ip",
                 "cuda_faiss_flat_l2", "cuda_faiss_flat_ip",
                 "faiss_gpu_ivf", "faiss_gpu_ivf_flat", "faiss_gpu_ivfpq",
                 "faiss_gpu_cagra", "cuda_faiss_cagra",
                 "cuda_faiss_ivf_flat", "cuda_faiss_ivfpq",
                 "cuvs", "gpu_cuvs", "cuda_cuvs", "cuda_cuvs_cagra",
                 "cuda_cagra", "gpu_cagra",
                 "cuvs_ivf", "cuda_cuvs_ivf", "cuvs_ivf_flat", "cuda_cuvs_ivf_flat",
                 "cuvs_ivfpq", "cuda_cuvs_ivfpq", "cuvs_ivf_pq", "cuda_cuvs_ivf_pq",
                 "cuvs_bruteforce", "cuda_cuvs_bruteforce", "cuda_cuvs_exact",
                 "cuvs_nndescent", "cuda_cuvs_nndescent",
                 "cuda_grid", "cuda_grid_auto", "gpu_grid",
                 "cuda_grid2d", "cuda_grid3d",
                 "gpu", "cuda", "cuda_approx", "cuda_nndescent",
                 "cuda_ivf", "cuda_faiss"
		               ),
               metric = c("euclidean", "cosine", "correlation"),
               n_threads = NULL) {
  backend <- match.arg(backend)
  metric <- match.arg(metric)
  nn_compute(
    data,
    points,
    k,
    backend,
    missing(points),
    exclude_self = FALSE,
    n_threads = n_threads,
    metric = metric
  )
}

#' Nearest neighbours excluding the self match
#'
#' `nn_without_self()` is a convenience wrapper around [nn()] for the common
#' case where the reference and query data are the same matrix and the
#' self-neighbour should be removed. It returns exactly `k` non-self neighbours
#' per observation.
#'
#' @param data Numeric matrix/data frame or sparse `Matrix` object with
#'   observations in rows.
#' @param k Number of non-self neighbours to return.
#' @param backend Nearest-neighbour backend. `"auto"` chooses CUDA/cuVS when
#'   available, then FAISS, then RcppHNSW/CPU fallbacks.
#' @param metric Distance metric: `"euclidean"`, `"cosine"`, or
#'   `"correlation"`.
#' @param n_threads Number of CPU worker threads used by CPU backends.
#' @return A `faissR_nn` object with `indices` and `distances` matrices.
#' @export
nn_without_self <- function(data,
                            k,
                            backend = c(
                              "auto", "cpu", "cpu_approx", "cpu_grid", "grid",
                              "cpu_sparse", "sparse", "sparse_cpu",
                              "cpu_grid2d", "grid2d", "cpu_grid3d", "grid3d",
                              "vptree", "cpu_vptree", "hnsw", "rcpphnsw", "cpu_hnsw",
                              "faiss", "cpu_faiss", "cpu_faiss_flat", "faiss_flat", "faiss_flat_l2",
                              "faiss_flat_ip", "faiss_ivf", "faiss_ivf_flat", "cpu_faiss_index_ivf",
                              "faiss_ivfpq", "faiss_hnsw", "faiss_nsg", "faiss_nndescent",
                              "faiss_gpu_flat", "faiss_gpu_flat_l2", "faiss_gpu_flat_ip",
                              "cuda_faiss_flat_l2", "cuda_faiss_flat_ip",
                              "faiss_gpu_ivf", "faiss_gpu_ivf_flat", "faiss_gpu_ivfpq",
                              "faiss_gpu_cagra", "cuda_faiss_cagra",
                              "cuda_faiss_ivf_flat", "cuda_faiss_ivfpq",
                              "cuvs", "gpu_cuvs", "cuda_cuvs", "cuda_cuvs_cagra",
                              "cuda_cagra", "gpu_cagra",
                              "cuvs_ivf", "cuda_cuvs_ivf", "cuvs_ivf_flat", "cuda_cuvs_ivf_flat",
                              "cuvs_ivfpq", "cuda_cuvs_ivfpq", "cuvs_ivf_pq", "cuda_cuvs_ivf_pq",
                              "cuvs_bruteforce", "cuda_cuvs_bruteforce", "cuda_cuvs_exact",
                              "cuvs_nndescent", "cuda_cuvs_nndescent",
                              "cuda_grid", "cuda_grid_auto", "gpu_grid",
                              "cuda_grid2d", "cuda_grid3d",
                              "gpu", "cuda", "cuda_approx", "cuda_nndescent",
                              "cuda_ivf", "cuda_faiss"
                            ),
                            metric = c("euclidean", "cosine", "correlation"),
                            n_threads = NULL) {
  backend <- match.arg(backend)
  metric <- match.arg(metric)
  nn_compute(
    data,
    data,
    k,
    backend,
    TRUE,
    exclude_self = TRUE,
    n_threads = n_threads,
    metric = metric
  )
}

#' Measure approximate KNN recall
#'
#' Compare an approximate nearest-neighbour result against an exact/reference
#' result by counting how many neighbours overlap in the first `k` columns.
#'
#' @param approx Approximate KNN result from [nn()] or an integer index matrix.
#' @param exact Reference KNN result from [nn()] or an integer index matrix.
#' @param k Number of neighbours to compare. Defaults to the shared number of
#'   columns in `approx` and `exact`.
#' @return A one-row data frame with mean, median, and minimum recall at `k`.
#' @export
knn_recall <- function(approx, exact, k = NULL) {
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
  values <- knn_recall_cpp(approx_idx, exact_idx, as.integer(k))
  data.frame(
    k = k,
    recall_at_k = unname(values["recall_at_k"]),
    median_recall_at_k = unname(values["median_recall_at_k"]),
    min_recall_at_k = unname(values["min_recall_at_k"]),
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

print.fastEmbedR_nn <- print.faissR_nn

#' Check whether the native Metal backend is available
#'
#' @return `TRUE` when a Metal device is available to the package.
#' @export
metal_available <- function() {
  isTRUE(metal_available_cpp())
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
