internal_nn <- function(data,
                        points = data,
                        k,
                        backend,
                        exclude_self = FALSE,
                        n_threads = NULL,
                        metric = "euclidean",
                        tuning = "auto") {
  faissR:::nn_compute(
    data = data,
    points = points,
    k = k,
    backend = backend,
    points_missing = missing(points),
    exclude_self = exclude_self,
    n_threads = n_threads,
    metric = metric,
    tuning = tuning
  )
}

internal_nn_without_self <- function(data,
                                     k,
                                     backend,
                                     n_threads = NULL,
                                     metric = "euclidean",
                                     tuning = "auto") {
  faissR:::nn_compute(
    data = data,
    points = data,
    k = k,
    backend = backend,
    points_missing = TRUE,
    exclude_self = TRUE,
    n_threads = n_threads,
    metric = metric,
    tuning = tuning
  )
}

test_that("public nearest-neighbour wrappers expose one canonical method and metric set", {
  canonical_methods <- c(
    "auto", "exact", "flat", "bruteforce", "grid",
    "hnsw", "ivf", "ivfpq", "vamana", "nsg", "nndescent", "cagra"
  )
  canonical_metrics <- c("euclidean", "cosine", "correlation", "inner_product")
  wrappers <- list(
    nn = formals(nn),
    nn_without_self = formals(nn_without_self),
    knn = formals(knn),
    knn_graph = formals(knn_graph),
    graph_cluster = formals(graph_cluster)
  )

  expect_equal(faissR:::nn_method_labels(), canonical_methods)
  expect_equal(faissR:::nn_metric_labels(), canonical_metrics)
  for (name in names(wrappers)) {
    f <- wrappers[[name]]
    method_arg <- if (identical(name, "knn_graph")) "nn_method" else if (identical(name, "graph_cluster")) "graph_method" else "method"
    expect_equal(eval(f[[method_arg]]), canonical_methods, info = name)
    expect_equal(eval(f$metric), canonical_metrics, info = name)
  }
})

test_that("public high-level APIs expose auto/cpu/cuda backend and auto tuning", {
  backend_choices <- c("auto", "cpu", "cuda")
  nn_tuning_choices <- c("auto", "cache", "pilot", "fixed", "off", "none")
  kmeans_tuning_choices <- c("auto", "fixed", "off", "none")
  wrappers <- list(
    nn = formals(nn),
    nn_without_self = formals(nn_without_self),
    knn = formals(knn),
    knn_graph = formals(knn_graph),
    graph_cluster = formals(graph_cluster),
    fast_kmeans = formals(fast_kmeans)
  )

  for (name in names(wrappers)) {
    f <- wrappers[[name]]
    expect_equal(eval(f$backend), backend_choices, info = name)
    expect_equal(eval(f$tuning), if (identical(name, "fast_kmeans")) kmeans_tuning_choices else nn_tuning_choices, info = name)
  }

  predict_formals <- formals(getS3method("predict", "faissR_knn_model"))
  expect_equal(eval(predict_formals$backend), backend_choices)
  expect_equal(eval(predict_formals$tuning), nn_tuning_choices)
  expect_equal(eval(formals(graph_cluster)$graph_backend), "auto")
  expect_equal(eval(formals(knn_graph)$nn_method), eval(formals(nn)$method))
  expect_equal(eval(formals(graph_cluster)$graph_method), eval(formals(nn)$method))
})

test_that("nn returns exact euclidean neighbors", {
  x <- matrix(c(
    0, 0,
    1, 0,
    0, 2,
    3, 0
  ), ncol = 2, byrow = TRUE)

  out <- nn(x, x, k = 3)
  expect_equal(dim(out$indices), c(4L, 3L))
  expect_equal(dim(out$distances), c(4L, 3L))
  expect_equal(out$indices[, 1], seq_len(nrow(x)))
  expect_equal(out$distances[, 1], rep(0, nrow(x)))

  d <- as.matrix(stats::dist(x))
  expected_idx <- t(apply(d, 1, order))[, 1:3]
  expected_dst <- matrix(0, nrow(x), 3)
  for (i in seq_len(nrow(x))) {
    expected_dst[i, ] <- d[i, expected_idx[i, ]]
  }
  expect_equal(unname(out$indices), unname(expected_idx))
  expect_equal(unname(out$distances), unname(expected_dst))
})

test_that("nn output distance storage can be requested explicitly", {
  x <- matrix(c(
    0, 0,
    1, 0,
    0, 1,
    2, 2
  ), ncol = 2, byrow = TRUE)

  out <- nn(x, x, k = 2L, backend = "cpu", method = "exact", output = "double")
  expect_true(is.matrix(out$distances))
  expect_equal(out$index_base, 1L)
  expect_equal(out$distance_type, "double")
  expect_equal(out$metric, "euclidean")
  expect_equal(out$backend_used, attr(out, "resolved_backend"))
  expect_equal(attr(out, "distance_type"), "double")

  if (requireNamespace("float", quietly = TRUE)) {
    fout <- nn(x, x, k = 2L, backend = "cpu", method = "exact", output = "float")
    expect_true(inherits(fout$distances, "float32"))
    expect_equal(fout$distance_type, "float32")
    expect_equal(fout$index_base, 1L)
    expect_equal(attr(fout, "distance_type"), "float32")

    dout <- nn_without_self(
      x,
      k = 2L,
      backend = "cpu",
      method = "flat",
      distances = "float"
    )
    expect_true(inherits(dout$distances, "float32"))
    expect_equal(dout$input_type, "float32")
    expect_equal(dout$input_layout, "r_double_column_major_to_row_major_float32")
    expect_true(dout$input_owns_data)
    expect_equal(dout$backend_used, "faiss_flat_l2")
    expect_equal(dout$distance_type, "float32")
    expect_equal(attr(dout, "distance_type"), "float32")
  } else {
    expect_error(
      nn(x, x, k = 2L, backend = "cpu", method = "exact", output = "float"),
      "requires the optional float package"
    )
  }

  expect_error(
    nn(x, x, k = 2L, backend = "cpu", method = "exact", output = "single"),
    "`output`"
  )
  expect_error(
    nn(x, x, k = 2L, backend = "cpu", method = "exact", output = "float", distances = "double"),
    "`output` and `distances`"
  )
})

test_that("float32 input routes through FAISS Flat when float is installed", {
  skip_if_not_installed("float")
  skip_if_not(faiss_available(), "FAISS is required for float32 input")

  x <- matrix(c(
    0, 0,
    1, 0,
    0, 1,
    2, 2,
    3, 3
  ), ncol = 2, byrow = TRUE)
  xf <- float::fl(x)

  ref <- nn_without_self(x, k = 2L, backend = "cpu", method = "flat", n_threads = 2L)
  out <- nn_without_self(xf, k = 2L, backend = "cpu", method = "flat", n_threads = 2L)
  auto <- nn_without_self(xf, k = 2L, backend = "cpu", method = "auto", n_threads = 2L)
  exact <- nn_without_self(xf, k = 2L, backend = "cpu", method = "exact", n_threads = 2L)
  fout <- nn_without_self(
    xf,
    k = 2L,
    backend = "cpu",
    method = "auto",
    output = "float",
    n_threads = 2L
  )

  expect_equal(out$indices, ref$indices)
  expect_equal(out$distances, ref$distances, tolerance = 1e-6)
  expect_equal(out$input_type, "float32")
  expect_equal(out$index_base, 1L)
  expect_equal(out$distance_type, "double")
  expect_equal(out$metric, "euclidean")
  expect_equal(out$backend_used, "faiss_flat_l2")
  expect_equal(out$input_layout, "float32_column_major_payload_to_row_major")
  expect_true(out$input_owns_data)
  expect_equal(attr(out, "distance_type"), "double")
  expect_equal(attr(out, "resolved_backend"), "faiss_flat_l2")
  expect_equal(auto$indices, ref$indices)
  expect_equal(auto$distances, ref$distances, tolerance = 1e-6)
  expect_equal(auto$backend_used, "faiss_flat_l2")
  expect_equal(attr(auto, "resolved_backend"), "faiss_flat_l2")
  expect_equal(exact$indices, ref$indices)
  expect_equal(exact$distances, ref$distances, tolerance = 1e-6)
  expect_equal(exact$backend_used, "faiss_flat_l2")
  expect_true(inherits(fout$distances, "float32"))
  expect_equal(fout$distance_type, "float32")
  expect_equal(attr(fout, "distance_type"), "float32")

  raw_float <- faissR:::nn_faiss_flat_float32_cpp(
    xf, xf, 2L, TRUE, 2L, "euclidean", "float"
  )
  expect_true(inherits(raw_float$distances, "float32"))
  expect_equal(raw_float$distance_type, "float32")
})

test_that("float32 FAISS Flat input accepts mixed double and float32 query matrices", {
  skip_if_not_installed("float")
  skip_if_not(faiss_available(), "FAISS is required for float32 input")

  x <- matrix(c(
    0.0, 0.0,
    1.0, 0.0,
    0.0, 1.0,
    2.0, 2.0,
    3.0, 3.0
  ), ncol = 2, byrow = TRUE)
  q <- matrix(c(
    0.1, 0.0,
    2.2, 2.0
  ), ncol = 2, byrow = TRUE)
  xf <- float::fl(x)
  qf <- float::fl(q)

  ref <- nn(x, q, k = 2L, backend = "cpu", method = "flat", n_threads = 2L)
  float_data <- nn(xf, q, k = 2L, backend = "cpu", method = "flat", n_threads = 2L)
  float_query <- nn(x, qf, k = 2L, backend = "cpu", method = "flat", n_threads = 2L)

  expect_equal(float_data$indices, ref$indices)
  expect_equal(float_data$distances, ref$distances, tolerance = 1e-6)
  expect_equal(float_data$input_type, "float32")
  expect_equal(
    float_data$input_layout,
    "data=float32_column_major_payload_to_row_major;points=r_double_column_major_to_row_major_float32"
  )
  expect_equal(float_data$backend_used, "faiss_flat_l2")

  expect_equal(float_query$indices, ref$indices)
  expect_equal(float_query$distances, ref$distances, tolerance = 1e-6)
  expect_equal(float_query$input_type, "float32")
  expect_equal(
    float_query$input_layout,
    "data=r_double_column_major_to_row_major_float32;points=float32_column_major_payload_to_row_major"
  )
  expect_equal(float_query$backend_used, "faiss_flat_l2")
})

test_that("row-compatible float32 inputs can use the direct payload route", {
  skip_if_not_installed("float")
  skip_if_not(faiss_available(), "FAISS is required for float32 input")

  x <- float::fl(matrix(c(0, 1, 3, 7, 8), ncol = 1))
  out <- nn_without_self(x, k = 2L, backend = "cpu", method = "flat", n_threads = 2L)

  expect_equal(out$input_type, "float32")
  expect_equal(out$input_layout, "float32_payload_direct_row_compatible")
  expect_false(out$input_owns_data)
  expect_equal(out$backend_used, "faiss_flat_l2")
})

test_that("float32 FAISS Flat input supports normalized metrics", {
  skip_if_not_installed("float")
  skip_if_not(faiss_available(), "FAISS is required for float32 input")

  x <- matrix(c(
    0.20,  1.10, -0.40,
    1.30, -0.20,  0.70,
   -0.80,  0.40,  1.50,
    2.10,  1.70, -1.20,
   -1.40,  2.20,  0.30,
    0.60, -1.30, -0.90
  ), ncol = 3, byrow = TRUE)
  xf <- float::fl(x)

  for (metric in c("cosine", "correlation")) {
    out <- nn_without_self(
      xf,
      k = 2L,
      backend = "cpu",
      method = "flat",
      metric = metric,
      n_threads = 2L
    )
    exact <- nn_without_self(
      x,
      k = 2L,
      backend = "cpu",
      method = "exact",
      metric = metric,
      n_threads = 2L
    )

    expect_equal(out$indices, exact$indices, info = metric)
    expect_equal(out$distances, exact$distances, tolerance = 1e-5, info = metric)
    expect_equal(out$input_type, "float32", info = metric)
    expect_equal(attr(out, "resolved_backend"), paste0("faiss_flat_", metric), info = metric)
  }
})

test_that("float32 C-callable entry point is registered", {
  skip_if_not_installed("Rcpp")
  Rcpp::cppFunction('
    #include <R_ext/Rdynload.h>
    bool faissR_test_float32_callable_registered() {
      DL_FUNC ptr = R_GetCCallable("faissR", "faissR_nn_float32_call");
      DL_FUNC ptr_output = R_GetCCallable("faissR", "faissR_nn_float32_call_output");
      return ptr != NULL && ptr_output != NULL;
    }
  ')
  expect_true(faissR_test_float32_callable_registered())
})

test_that("float32 C-callable returns stable KNN metadata", {
  skip_if_not_installed("Rcpp")
  skip_if_not_installed("float")
  skip_if_not(faiss_available(), "FAISS is required for float32 input")
  Rcpp::cppFunction(
    code = '
      SEXP faissR_test_float32_callable_run(SEXP x) {
      typedef SEXP (*faissR_nn_float32_fun)(SEXP, SEXP, SEXP, SEXP, SEXP, SEXP);
      faissR_nn_float32_fun fn = (faissR_nn_float32_fun)
        R_GetCCallable("faissR", "faissR_nn_float32_call");
      if (fn == NULL) Rcpp::stop("faissR_nn_float32_call is not registered");
      SEXP out = fn(
        x,
        Rcpp::wrap(2),
        Rcpp::wrap(std::string("faiss_flat_l2")),
        Rcpp::wrap(std::string("euclidean")),
        Rcpp::wrap(false),
        Rcpp::wrap(2)
      );
      return out;
    }',
    includes = "#include <R_ext/Rdynload.h>"
  )

  x <- float::fl(matrix(c(
    0, 0,
    1, 0,
    0, 2,
    3, 3
  ), ncol = 2, byrow = TRUE))
  out <- faissR_test_float32_callable_run(x)

  expect_s3_class(out, "faissR_nn")
  expect_equal(dim(out$indices), c(4L, 2L))
  expect_equal(out$index_base, 1L)
  expect_equal(out$distance_type, "double")
  expect_equal(out$metric, "euclidean")
  expect_equal(out$backend_used, "faiss_flat_l2")
  expect_equal(attr(out, "distance_type"), "double")
  expect_equal(attr(out, "metric"), "euclidean")
  expect_equal(attr(out, "backend_used"), "faiss_flat_l2")
  expect_equal(attr(out, "resolved_backend"), "faiss_flat_l2")

  xd <- matrix(c(
    0, 0,
    1, 0,
    0, 2,
    3, 3
  ), ncol = 2, byrow = TRUE)
  out_double <- faissR_test_float32_callable_run(xd)
  expect_equal(out_double$indices, out$indices)
  expect_equal(out_double$distances, out$distances, tolerance = 1e-6)
  expect_equal(out_double$input_type, "float32")
  expect_equal(out_double$input_layout, "r_double_column_major_to_row_major_float32")
})

test_that("float32 C-callable can return float distances", {
  skip_if_not_installed("Rcpp")
  skip_if_not_installed("float")
  skip_if_not(faiss_available(), "FAISS is required for float32 input")
  Rcpp::cppFunction(
    code = '
      SEXP faissR_test_float32_callable_run_output(SEXP x) {
      typedef SEXP (*faissR_nn_float32_fun)(SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP);
      faissR_nn_float32_fun fn = (faissR_nn_float32_fun)
        R_GetCCallable("faissR", "faissR_nn_float32_call_output");
      if (fn == NULL) Rcpp::stop("faissR_nn_float32_call_output is not registered");
      SEXP out = fn(
        x,
        Rcpp::wrap(2),
        Rcpp::wrap(std::string("cpu")),
        Rcpp::wrap(std::string("euclidean")),
        Rcpp::wrap(false),
        Rcpp::wrap(2),
        Rcpp::wrap(std::string("float"))
      );
      return out;
    }',
    includes = "#include <R_ext/Rdynload.h>"
  )

  x <- float::fl(matrix(c(
    0, 0,
    1, 0,
    0, 2,
    3, 3
  ), ncol = 2, byrow = TRUE))
  out <- faissR_test_float32_callable_run_output(x)

  expect_s3_class(out, "faissR_nn")
  expect_equal(dim(out$indices), c(4L, 2L))
  expect_true(inherits(out$distances, "float32"))
  expect_equal(out$index_base, 1L)
  expect_equal(out$distance_type, "float32")
  expect_equal(out$metric, "euclidean")
  expect_equal(out$backend_used, "faiss_flat_l2")
  expect_equal(attr(out, "distance_type"), "float32")
  expect_equal(attr(out, "metric"), "euclidean")
  expect_equal(attr(out, "backend_used"), "faiss_flat_l2")
  expect_equal(attr(out, "resolved_backend"), "faiss_flat_l2")
})

test_that("native exact KNN only accepts the documented metrics", {
  x <- matrix(c(
    0, 0,
    1, 0,
    0, 1
  ), ncol = 2, byrow = TRUE)

  expect_error(
    faissR:::nn_cpp(x, x, 2L, "invalid_metric", FALSE, TRUE, 2, FALSE, 1L, FALSE),
    "unsupported metric"
  )
  expect_error(
    faissR:::nn_cpp(x, x, 2L, "minkowski", FALSE, TRUE, 2, FALSE, 1L, FALSE),
    "unsupported metric"
  )
})

test_that("public NN APIs canonicalize common metric aliases", {
  x <- matrix(c(
    1, 0,
    0, 1,
    1, 1,
    2, 0
  ), ncol = 2, byrow = TRUE)

  l2 <- nn(x, k = 2L, backend = "cpu", metric = "l2")
  cor <- nn(x, k = 2L, backend = "cpu", metric = "pearson")
  ip <- nn(x, k = 2L, backend = "cpu", metric = "ip")
  without_self <- nn_without_self(x, k = 1L, backend = "cpu", metric = "cor")

  expect_equal(attr(l2, "metric"), "euclidean")
  expect_equal(attr(cor, "metric"), "correlation")
  expect_equal(attr(ip, "metric"), "inner_product")
  expect_equal(attr(without_self, "metric"), "correlation")
  expect_error(nn(x, k = 2L, backend = "cpu", metric = "manhattan"), "metric")
})

test_that("public NN results preserve requested and resolved routing metadata", {
  set.seed(1042)
  x <- matrix(rnorm(80), ncol = 4)

  out <- nn(
    x,
    k = 3L,
    backend = "auto",
    method = "exact",
    metric = "l2",
    tuning = "off",
    n_threads = 2L
  )
  without_self <- nn_without_self(
    x,
    k = 3L,
    backend = "auto",
    method = "exact",
    metric = "cos",
    tuning = "off",
    n_threads = 2L
  )

  expect_equal(attr(out, "requested_backend"), "auto")
  expect_equal(attr(out, "requested_method"), "exact")
  expect_equal(attr(out, "resolved_backend"), attr(out, "backend"))
  expect_equal(attr(out, "metric"), "euclidean")
  expect_equal(attr(out, "tuning"), "off")
  expect_equal(attr(without_self, "requested_backend"), "auto")
  expect_equal(attr(without_self, "requested_method"), "exact")
  expect_equal(attr(without_self, "resolved_backend"), attr(without_self, "backend"))
  expect_equal(attr(without_self, "metric"), "cosine")
  expect_equal(attr(without_self, "tuning"), "off")
  auto_meta <- attr(out, "auto_selection")
  expect_equal(auto_meta$policy, "static_shape_k_metric_selector")
  expect_false(auto_meta$slow_tuning)
  expect_equal(auto_meta$requested_backend, "auto")
  expect_equal(auto_meta$requested_method, "exact")
  expect_false(auto_meta$explicit_backend)
  expect_true(auto_meta$explicit_method)
  expect_equal(auto_meta$backend_decision, "explicit_route")
  expect_equal(auto_meta$method_decision, "explicit_exact")
  expect_equal(auto_meta$predicted_backend, attr(out, "backend"))
  expect_equal(auto_meta$predicted_method, "exact")
  expect_equal(auto_meta$predicted_device, "cpu")
  expect_equal(auto_meta$metric, "euclidean")
  expect_equal(auto_meta$k, 3L)
  expect_equal(auto_meta$n, nrow(x))
  expect_equal(auto_meta$p, ncol(x))
})

test_that("top-level auto reuses the metric-aware CPU auto selector after GPU decline", {
  skip_if_not(faiss_available())
  old <- options(
    faissR.cpu_auto_exact_work = 1,
    faissR.cpu_auto_faiss_flat_work = 1,
    faissR.cuda_auto_exact_work = 1e99
  )
  on.exit(options(old), add = TRUE)

  set.seed(1043)
  x <- matrix(rnorm(80), ncol = 4)
  q <- x[1:3, , drop = FALSE]

  cpu_auto <- nn(x, q, k = 3L, backend = "cpu", method = "auto", metric = "cosine")
  top_auto <- nn(x, q, k = 3L, backend = "auto", method = "auto", metric = "cosine")

  expect_equal(attr(cpu_auto, "backend"), "faiss_flat_cosine")
  expect_equal(attr(top_auto, "backend"), attr(cpu_auto, "backend"))
  expect_equal(attr(top_auto, "metric"), "cosine")
  expect_equal(attr(top_auto, "requested_backend"), "auto")
  expect_equal(attr(top_auto, "requested_method"), "auto")
  auto_meta <- attr(top_auto, "auto_selection")
  expect_equal(auto_meta$policy, "static_shape_k_metric_selector")
  expect_false(auto_meta$explicit_backend)
  expect_false(auto_meta$explicit_method)
  expect_equal(auto_meta$backend_decision, "auto_cpu_fallback")
  expect_equal(auto_meta$method_decision, "auto_cpu_fallback")
  expect_equal(auto_meta$predicted_backend, attr(top_auto, "backend"))
  expect_equal(auto_meta$predicted_method, "flat")
  expect_equal(auto_meta$predicted_device, "cpu")
  expect_equal(auto_meta$reason, "auto_cpu_fallback")
  expect_equal(auto_meta$metric, "cosine")
  expect_false(auto_meta$self_query)
  expect_false(auto_meta$slow_tuning)
})

test_that("explicit NN routes do not attach auto selection metadata", {
  set.seed(1044)
  x <- matrix(rnorm(80), ncol = 4)
  out <- nn(x, k = 3L, backend = "cpu", method = "exact", tuning = "off")

  expect_null(attr(out, "auto_selection"))
})

test_that("NN auto-selection metadata distinguishes explicit backend and method requests", {
  skip_if_not(faiss_available())
  set.seed(1045)
  x <- matrix(rnorm(80), ncol = 4)

  cpu_auto_method <- nn(x, k = 3L, backend = "cpu", method = "auto", tuning = "off")
  auto_meta <- attr(cpu_auto_method, "auto_selection")

  expect_true(auto_meta$explicit_backend)
  expect_false(auto_meta$explicit_method)
  expect_equal(auto_meta$backend_decision, "explicit_cpu")
  expect_equal(auto_meta$method_decision, "cpu_auto_shape_selector")
  expect_equal(auto_meta$requested_backend, "cpu")
  expect_equal(auto_meta$requested_method, "auto")
  expect_equal(auto_meta$predicted_backend, attr(cpu_auto_method, "backend"))
})

test_that("resolved backend labels map to stable auto-selection method and device metadata", {
  expect_equal(faissR:::nn_resolved_backend_public_method("faiss_hnsw"), "hnsw")
  expect_equal(faissR:::nn_resolved_backend_public_method("faiss_flat_cosine"), "flat")
  expect_equal(faissR:::nn_resolved_backend_public_method("cuda_cuvs_nndescent"), "nndescent")
  expect_equal(faissR:::nn_resolved_backend_public_method("cuda_native_nndescent"), "nndescent")
  expect_equal(faissR:::nn_resolved_backend_public_method("cuda_cuvs_hnsw"), "hnsw")
  expect_equal(faissR:::nn_resolved_backend_public_method("faiss_gpu_cagra"), "cagra")
  expect_equal(faissR:::nn_resolved_backend_public_method("cuda_cuvs_bruteforce"), "exact")
  expect_equal(faissR:::nn_resolved_backend_public_method("unknown_backend"), NA_character_)

  expect_equal(faissR:::nn_resolved_backend_device("faiss_hnsw"), "cpu")
  expect_equal(faissR:::nn_resolved_backend_device("faiss_gpu_ivf_flat"), "cuda")
  expect_equal(faissR:::nn_resolved_backend_device("cuda_cuvs_bruteforce"), "cuda")
  expect_equal(faissR:::nn_resolved_backend_device("cuda_cuvs_hnsw"), "cuda")
  expect_equal(faissR:::nn_resolved_backend_device("cuvs_ivfpq"), "cuda")
  expect_equal(faissR:::nn_resolved_backend_device("cpu_auto"), "auto")
})

test_that("public NN APIs require scalar backend method metric and tuning choices", {
  x <- matrix(rnorm(40), ncol = 4)
  y <- factor(rep(c("a", "b"), each = 5L))

  expect_no_error(nn(x, k = 2L))
  expect_no_error(nn_without_self(x, k = 2L))
  expect_error(nn(x, k = c(2L, 3L), backend = "cpu"), "`k` must be NULL or a positive integer")
  expect_error(nn(x, k = 2.5, backend = "cpu"), "`k` must be NULL or a positive integer")
  expect_error(nn_without_self(x, k = 1.5, backend = "cpu"), "`k` must be NULL or a positive integer")
  expect_error(nn(x, k = 2L, backend = "cpu", n_threads = c(1L, 2L)), "`n_threads` must be NULL or a single positive integer")
  expect_error(nn(x, k = 2L, backend = "cpu", n_threads = 1.5), "`n_threads` must be NULL or a single positive integer")
  expect_error(nn_without_self(x, k = 2L, backend = "cpu", n_threads = 0L), "`n_threads` must be NULL or a single positive integer")
  expect_equal(faissR:::normalize_nn_threads(128L), 64L)
  expect_error(nn(x, k = 2L, backend = c("cpu", "cuda")), "`backend` must be a single value")
  expect_error(nn(x, k = 2L, method = c("exact", "hnsw")), "`method` must be a single value")
  expect_error(nn(x, k = 2L, metric = c("cosine", "correlation")), "`metric` must be a single value")
  expect_error(nn(x, k = 2L, tuning = c("auto", "off")), "`tuning` must be a single value")
  internal_routes <- c("faiss_hnsw", "faiss_ivf", "faiss_gpu_cagra", "cuda_cuvs_nndescent", "cuda_cuvs_hnsw")
  for (route in internal_routes) {
    expect_error(nn(x, k = 2L, method = route), "canonical lowercase", info = route)
    expect_error(nn_without_self(x, k = 2L, method = route), "canonical lowercase", info = route)
    expect_error(knn(x, y, method = route), "canonical lowercase", info = route)
    expect_error(knn_graph(x, k = 2L, method = route), "canonical lowercase", info = route)
  }
  expect_error(
    faissR:::resolve_public_nn_backend(c("cpu", "cuda"), "exact", "euclidean"),
    "`backend` must be a single value"
  )
})

test_that("public NN APIs validate tuning at the exported boundary", {
  x <- matrix(rnorm(40), ncol = 4)

  expect_no_error(nn(x, k = 2L, backend = "cpu", method = "exact", tuning = "none"))
  expect_no_error(nn_without_self(x, k = 2L, backend = "cpu", method = "exact", tuning = "none"))
  expect_error(nn(x, k = 2L, backend = "cpu", tuning = "aggressive"), "tuning")
  expect_error(nn_without_self(x, k = 2L, backend = "cpu", tuning = "aggressive"), "tuning")
})

test_that("nn_capabilities documents the public method metric matrix", {
  caps <- nn_capabilities()
  methods <- c(
    "auto", "exact", "flat", "bruteforce", "grid",
    "hnsw", "ivf", "ivfpq", "vamana", "nsg", "nndescent", "cagra"
  )
  metrics <- c("euclidean", "cosine", "correlation", "inner_product")

  expect_s3_class(caps, "data.frame")
  expect_setequal(caps$method, methods)
  expect_setequal(caps$backend, c("auto", "cpu", "cuda"))
  expect_setequal(caps$metric, metrics)
  expect_identical(
    names(caps),
    c("method", "backend", "metric", "supported", "exact", "implementation", "notes")
  )
  expect_equal(nrow(caps), length(methods) * 3L * length(metrics))
  expect_equal(anyDuplicated(caps[c("method", "backend", "metric")]), 0L)

  expect_true(caps$supported[caps$method == "flat" & caps$metric == "correlation" & caps$backend == "cuda"])
  expect_true(all(caps$supported[caps$method == "auto" & caps$backend == "cuda"]))
  expect_true(all(!caps$supported[caps$method == "cagra" & caps$backend == "cpu"]))
  expect_true(all(caps$supported[
    caps$method == "hnsw" & caps$backend == "cuda"
  ]))
  hnsw_ip <- caps[
    caps$method == "hnsw" & caps$backend == "cuda" &
      caps$metric == "inner_product",
    ,
    drop = FALSE
  ]
  expect_true(hnsw_ip$supported)
  expect_match(hnsw_ip$notes, "maximum-inner-product-to-L2")
  cagra_ip <- caps[
    caps$method == "cagra" & caps$backend == "cuda" &
      caps$metric == "inner_product",
    ,
    drop = FALSE
  ]
  expect_true(cagra_ip$supported)
  expect_match(cagra_ip$notes, "maximum-inner-product-to-L2")
  expect_true(all(is.na(caps$implementation[!caps$supported])))
  expect_true(all(caps$supported[caps$method == "ivf"]))
  expect_true(all(caps$supported[caps$method == "ivfpq"]))
  expect_true(all(caps$supported[caps$method == "vamana"]))
  expect_true(all(caps$supported[
    caps$method == "grid" & caps$metric %in% c("euclidean", "cosine", "correlation")
  ]))
  expect_true(all(!caps$supported[
    caps$method == "grid" & caps$metric == "inner_product"
  ]))
  expect_true(all(caps$supported[caps$method == "nsg" & caps$backend == "cpu"]))
  expect_true(all(caps$supported[caps$method == "nsg" & caps$backend == "cuda"]))
  expect_true(all(caps$supported[
    caps$method == "nndescent" & caps$metric %in% c("euclidean", "cosine", "correlation")
  ]))
  expect_true(caps$supported[
    caps$method == "nndescent" & caps$backend == "cpu" & caps$metric == "inner_product"
  ])
  expect_true(caps$supported[
    caps$method == "nndescent" & caps$backend == "auto" & caps$metric == "inner_product"
  ])
  expect_true(caps$supported[
    caps$method == "nndescent" & caps$backend == "cuda" & caps$metric == "inner_product"
  ])
  expect_false("removed_method" %in% caps$method)

  cuda_bruteforce_l2 <- caps[
    caps$backend == "cuda" & caps$method == "bruteforce" & caps$metric == "euclidean",
    ,
    drop = FALSE
  ]
  cuda_bruteforce_cor <- caps[
    caps$backend == "cuda" & caps$method == "bruteforce" & caps$metric == "correlation",
    ,
    drop = FALSE
  ]
  expect_match(cuda_bruteforce_l2$implementation, "cuVS brute force")
  expect_match(cuda_bruteforce_l2$notes, "direct cuVS brute force")
  expect_equal(cuda_bruteforce_cor$implementation, "FAISS GPU Flat")
  expect_match(cuda_bruteforce_cor$notes, "FAISS GPU Flat")
  expect_match(cuda_bruteforce_cor$notes, "Euclidean/L2-only")

  cuda_auto_cor <- caps[
    caps$backend == "cuda" & caps$method == "auto" & caps$metric == "correlation",
    ,
    drop = FALSE
  ]
  expect_equal(nrow(cuda_auto_cor), 1L)
  expect_match(cuda_auto_cor$notes, "CUDA grid")
  expect_match(cuda_auto_cor$notes, "FAISS GPU Flat")
  expect_match(cuda_auto_cor$notes, "cuVS-only")
})

test_that("nn_capabilities agrees with public CPU/CUDA resolver support", {
  caps <- nn_capabilities()
  checked <- subset(caps, backend %in% c("cpu", "cuda") & method != "auto")

  for (i in seq_len(nrow(checked))) {
    row <- checked[i, , drop = FALSE]
    label <- paste(row$backend, row$method, row$metric)
    expr <- quote(faissR:::resolve_public_nn_backend(
      row$backend[[1L]],
      row$method[[1L]],
      row$metric[[1L]]
    ))
    if (isTRUE(row$supported[[1L]])) {
      expect_error(eval(expr), NA, info = label)
    } else {
      expect_error(eval(expr), info = label)
    }
  }
})

test_that("nn_capabilities can report current runtime availability", {
  caps <- nn_capabilities()
  runtime_caps <- nn_capabilities(runtime = TRUE)

  expect_identical(
    names(caps),
    c("method", "backend", "metric", "supported", "exact", "implementation", "notes")
  )
  expect_equal(nrow(runtime_caps), nrow(caps))
  expect_true(all(c(
    "resolved_backend", "runtime_available", "runtime_reason", "runtime_notes"
  ) %in% names(runtime_caps)))
  expect_type(runtime_caps$runtime_available, "logical")
  expect_true(all(!is.na(runtime_caps$runtime_reason)))
  expect_true(all(!is.na(runtime_caps$runtime_notes)))
  expect_false(any(runtime_caps$runtime_reason == "resolver_error"))

  cpu_exact <- runtime_caps[
    runtime_caps$backend == "cpu" &
      runtime_caps$method == "exact" &
      runtime_caps$metric == "euclidean",
    ,
    drop = FALSE
  ]
  expect_equal(cpu_exact$resolved_backend, "cpu")
  expect_true(cpu_exact$runtime_available)
  expect_equal(cpu_exact$runtime_reason, "available")

  cpu_flat <- runtime_caps[
    runtime_caps$backend == "cpu" &
      runtime_caps$method == "flat" &
      runtime_caps$metric == "euclidean",
    ,
    drop = FALSE
  ]
  expect_equal(cpu_flat$resolved_backend, "faiss_flat_l2")
  expect_equal(cpu_flat$runtime_available, faiss_available())
  expect_equal(cpu_flat$runtime_reason, if (faiss_available()) "available" else "missing_faiss")

  cuda_flat <- runtime_caps[
    runtime_caps$backend == "cuda" &
      runtime_caps$method == "flat" &
      runtime_caps$metric == "euclidean",
    ,
    drop = FALSE
  ]
  expect_equal(cuda_flat$resolved_backend, "faiss_gpu_flat_l2")
  expect_equal(cuda_flat$runtime_available, faiss_gpu_available())
  expect_equal(cuda_flat$runtime_reason, if (faiss_gpu_available()) "available" else "missing_faiss_gpu")

  unsupported <- runtime_caps[
    runtime_caps$backend == "cpu" &
      runtime_caps$method == "cagra" &
      runtime_caps$metric == "euclidean",
    ,
    drop = FALSE
  ]
  expect_false(unsupported$supported)
  expect_false(unsupported$runtime_available)
  expect_equal(unsupported$runtime_reason, "unsupported_combination")
  expect_true(is.na(unsupported$resolved_backend))
})

test_that("runtime-available CPU capability rows execute across public metrics", {
  skip_if_not_installed("Matrix")

  old_options <- options(
    faissR.faiss_pq_m = 1L,
    faissR.faiss_pq_nbits = 4L
  )
  on.exit(options(old_options), add = TRUE)

  set.seed(20260622)
  dense <- matrix(rnorm(180L * 8L), nrow = 180L)
  low_dim <- matrix(rnorm(120L * 2L), nrow = 120L)
  ivfpq_dense <- matrix(rnorm(700L * 4L), nrow = 700L)
  smoke_methods <- c(
    "exact", "flat", "bruteforce", "grid",
    "hnsw", "ivf", "ivfpq", "nndescent"
  )
  runtime_caps <- nn_capabilities(runtime = TRUE)
  rows <- runtime_caps[
    runtime_caps$backend == "cpu" &
      runtime_caps$method %in% smoke_methods &
      runtime_caps$supported &
      runtime_caps$runtime_available,
    ,
    drop = FALSE
  ]
  expect_gt(nrow(rows), 0L)
  expect_false("nsg" %in% rows$method)

  for (i in seq_len(nrow(rows))) {
    row <- rows[i, , drop = FALSE]
    x <- switch(
      row$method,
      grid = low_dim,
      ivfpq = ivfpq_dense,
      dense
    )
    label <- paste(row$method, row$metric, row$resolved_backend, sep = "/")
    out <- nn_without_self(
      x,
      k = 5L,
      backend = "cpu",
      method = row$method,
      metric = row$metric,
      tuning = "fixed",
      n_threads = 2L
    )

    expect_true(inherits(out, "faissR_nn"), info = label)
    expect_equal(dim(out$indices), c(nrow(x), 5L), info = label)
    expect_equal(dim(as.matrix(out$distances)), c(nrow(x), 5L), info = label)
    expect_equal(attr(out, "metric"), row$metric, info = label)
    expect_equal(attr(out, "requested_method"), row$method, info = label)
    expect_equal(attr(out, "requested_backend"), "cpu", info = label)
    expect_true(all(out$indices >= 1L & out$indices <= nrow(x)), info = label)
    expect_true(all(is.finite(as.matrix(out$distances))), info = label)
  }
})

test_that("runtime-available CPU core methods execute across benchmark k grid", {
  old_options <- options(
    faissR.faiss_nlist = 16L,
    faissR.faiss_nprobe = 16L
  )
  on.exit(options(old_options), add = TRUE)

  set.seed(20260623)
  x <- matrix(rnorm(240L * 8L), nrow = 240L)
  k_values <- c(5L, 10L, 15L, 50L, 100L)
  core_methods <- c("exact", "flat", "hnsw", "ivf", "nndescent")
  runtime_caps <- nn_capabilities(runtime = TRUE)
  rows <- runtime_caps[
    runtime_caps$backend == "cpu" &
      runtime_caps$method %in% core_methods &
      runtime_caps$supported &
      runtime_caps$runtime_available,
    ,
    drop = FALSE
  ]
  expect_gt(nrow(rows), 0L)

  for (i in seq_len(nrow(rows))) {
    row <- rows[i, , drop = FALSE]
    for (k in k_values) {
      label <- paste(row$method, row$metric, row$resolved_backend, paste0("k=", k), sep = "/")
      out <- nn_without_self(
        x,
        k = k,
        backend = "cpu",
        method = row$method,
        metric = row$metric,
        tuning = "fixed",
        n_threads = 2L
      )

      expect_true(inherits(out, "faissR_nn"), info = label)
      expect_equal(dim(out$indices), c(nrow(x), k), info = label)
      expect_equal(dim(as.matrix(out$distances)), c(nrow(x), k), info = label)
      expect_equal(attr(out, "metric"), row$metric, info = label)
      expect_equal(attr(out, "requested_method"), row$method, info = label)
      expect_equal(attr(out, "requested_backend"), "cpu", info = label)
      expect_equal(attr(out, "tuning"), "fixed", info = label)
      expect_true(all(out$indices >= 1L & out$indices <= nrow(x)), info = label)
      expect_true(all(is.finite(as.matrix(out$distances))), info = label)
    }
  }
})

test_that("cuda_auto runtime availability distinguishes cuVS and FAISS GPU metric routes", {
  euclidean_cuvs <- faissR:::nn_cuda_auto_runtime_available(
    "euclidean",
    cuda_available_value = FALSE,
    cuvs_available_value = TRUE,
    faiss_gpu_available_value = FALSE
  )
  expect_true(euclidean_cuvs$available)
  expect_match(euclidean_cuvs$notes, "cuVS")

  cosine_cuvs_only <- faissR:::nn_cuda_auto_runtime_available(
    "cosine",
    cuda_available_value = TRUE,
    cuvs_available_value = TRUE,
    faiss_gpu_available_value = FALSE
  )
  expect_false(cosine_cuvs_only$available)
  expect_match(cosine_cuvs_only$notes, "FAISS GPU Flat")
  expect_match(cosine_cuvs_only$notes, "2D/3D")

  ip_faiss_gpu <- faissR:::nn_cuda_auto_runtime_available(
    "inner_product",
    cuda_available_value = FALSE,
    cuvs_available_value = FALSE,
    faiss_gpu_available_value = TRUE
  )
  expect_true(ip_faiss_gpu$available)
  expect_match(ip_faiss_gpu$notes, "FAISS GPU Flat")
})

test_that("backend auto explicit methods require metric-capable CUDA routes before selecting CUDA", {
  expect_equal(
    faissR:::resolve_auto_public_nn_device(
      "flat",
      "cosine",
      cuda_available_value = TRUE,
      cuvs_available_value = TRUE,
      faiss_gpu_available_value = FALSE
    ),
    "cpu"
  )
  expect_equal(
    faissR:::resolve_auto_public_nn_device(
      "ivf",
      "inner_product",
      cuda_available_value = TRUE,
      cuvs_available_value = TRUE,
      faiss_gpu_available_value = FALSE
    ),
    "cpu"
  )
  expect_equal(
    faissR:::resolve_auto_public_nn_device(
      "bruteforce",
      "euclidean",
      cuda_available_value = FALSE,
      cuvs_available_value = TRUE,
      faiss_gpu_available_value = FALSE
    ),
    "cuda"
  )
  expect_equal(
    faissR:::resolve_auto_public_nn_device(
      "nndescent",
      "inner_product",
      cuda_available_value = TRUE,
      cuvs_available_value = TRUE,
      faiss_gpu_available_value = TRUE
    ),
    "cuda"
  )
  expect_equal(
    faissR:::resolve_auto_public_nn_device(
      "hnsw",
      "euclidean",
      cuda_available_value = FALSE,
      cuvs_available_value = TRUE,
      faiss_gpu_available_value = FALSE
    ),
    "cuda"
  )
  expect_equal(
    faissR:::resolve_auto_public_nn_device(
      "hnsw",
      "inner_product",
      cuda_available_value = TRUE,
      cuvs_available_value = TRUE,
      faiss_gpu_available_value = TRUE
    ),
    "cuda"
  )
  expect_equal(
    faissR:::resolve_auto_public_nn_device(
      "cagra",
      "euclidean",
      cuda_available_value = FALSE,
      cuvs_available_value = FALSE,
      faiss_gpu_available_value = FALSE
    ),
    "cuda"
  )
})

test_that("nn_capabilities agrees with public backend resolver", {
  caps <- nn_capabilities()
  expect_equal(sort(unique(caps$backend)), c("auto", "cpu", "cuda"))
  for (i in seq_len(nrow(caps))) {
    row <- caps[i, , drop = FALSE]
    resolved <- tryCatch(
      faissR:::resolve_public_nn_backend(row$backend, row$method, row$metric),
      error = identity
    )
    if (isTRUE(row$supported)) {
      expect_false(
        inherits(resolved, "error"),
        info = sprintf(
          "%s/%s/%s should resolve because nn_capabilities() marks it supported",
          row$backend, row$method, row$metric
        )
      )
    } else {
      expect_true(
        inherits(resolved, "error"),
        info = sprintf(
          "%s/%s/%s should error because nn_capabilities() marks it unsupported",
          row$backend, row$method, row$metric
        )
      )
    }
  }
})

test_that("nn_capabilities exposes auto backend as CPU/CUDA support union", {
  caps <- nn_capabilities()
  keys <- unique(caps[, c("method", "metric"), drop = FALSE])
  for (i in seq_len(nrow(keys))) {
    method <- keys$method[[i]]
    metric <- keys$metric[[i]]
    auto <- caps[caps$backend == "auto" & caps$method == method & caps$metric == metric, , drop = FALSE]
    cpu <- caps[caps$backend == "cpu" & caps$method == method & caps$metric == metric, , drop = FALSE]
    cuda <- caps[caps$backend == "cuda" & caps$method == method & caps$metric == metric, , drop = FALSE]
    expect_equal(nrow(auto), 1L)
    expect_equal(nrow(cpu), 1L)
    expect_equal(nrow(cuda), 1L)
    expect_equal(
      auto$supported[[1L]],
      isTRUE(cpu$supported[[1L]]) || isTRUE(cuda$supported[[1L]]),
      info = sprintf("%s/%s", method, metric)
    )
  }
})

test_that("faissR options use the faissR namespace only", {
  old <- options(
    faissR.cpu_auto_exact_work = NULL,
    faissR.faiss_nlist = NULL
  )
  on.exit(options(old), add = TRUE)

  default_route <- faissR:::select_cpu_auto_backend(
    self_query = TRUE,
    n = 10000L,
    p = 20L,
    n_points = 10000L,
    k = 50L,
    work_size = 100,
    metric = "euclidean"
  )
  expect_equal(default_route, "cpu")

  options(faissR.cpu_auto_exact_work = 1)
  current <- faissR:::select_cpu_auto_backend(
    self_query = TRUE,
    n = 10000L,
    p = 20L,
    n_points = 10000L,
    k = 50L,
    work_size = 100,
    metric = "euclidean"
  )
  expect_true(current %in% c("faiss_hnsw", "hnsw", "cpu"))

  expect_false(identical(faissR:::faiss_ivf_params(1000L, 10L)$requested_nlist, 32L))
  options(faissR.faiss_nlist = 32L)
  expect_equal(faissR:::faiss_ivf_params(1000L, 10L)$requested_nlist, 32L)
})


test_that("nn rejects sparse Matrix inputs after sparse method removal", {
  skip_if_not_installed("Matrix")
  sx <- Matrix::Matrix(matrix(0, nrow = 4L, ncol = 3L), sparse = TRUE)
  expect_error(nn(sx, k = 2L), "Sparse Matrix input is no longer supported")
  expect_error(nn_without_self(sx, k = 2L), "Sparse Matrix input is no longer supported")
})

test_that("nn returns exact cosine neighbors on CPU", {
  x <- matrix(c(
    1, 0,
    0, 1,
    -1, 0,
    1, 1
  ), ncol = 2, byrow = TRUE)

  out <- nn(x, k = 3L, backend = "cpu", metric = "cosine")

  expect_equal(attr(out, "backend"), "cpu")
  expect_equal(attr(out, "metric"), "cosine")
  expect_true(isTRUE(attr(out, "exact")))
  expect_equal(out$indices[1, ], c(1L, 4L, 2L))
  expect_equal(out$distances[1, ], c(0, 1 - 1 / sqrt(2), 1), tolerance = 1e-12)
})

test_that("nn returns exact correlation neighbors on CPU", {
  x <- matrix(c(
    1, 2, 3,
    1, 3, 5,
    3, 2, 1,
    5, 5, 5
  ), ncol = 3, byrow = TRUE)

  corr_dist <- function(a, b) {
    a <- a - mean(a)
    b <- b - mean(b)
    an <- sqrt(sum(a * a))
    bn <- sqrt(sum(b * b))
    if (an <= 0 && bn <= 0) return(0)
    if (an <= 0 || bn <= 0) return(1)
    1 - sum(a * b) / (an * bn)
  }
  expected <- outer(seq_len(nrow(x)), seq_len(nrow(x)), Vectorize(function(i, j) {
    corr_dist(x[i, ], x[j, ])
  }))

  out <- nn(x, k = 4L, backend = "cpu", metric = "correlation")

  expect_equal(attr(out, "backend"), "cpu")
  expect_equal(attr(out, "metric"), "correlation")
  expect_true(isTRUE(attr(out, "exact")))
  expect_equal(unname(out$indices), unname(t(apply(expected, 1, order))))
  for (i in seq_len(nrow(x))) {
    expect_equal(out$distances[i, ], expected[i, out$indices[i, ]], tolerance = 1e-12)
  }
})

test_that("nn returns exact inner-product neighbors on CPU", {
  x <- matrix(c(
    2, 0,
    0, 3,
    1, 1,
    -1, 0
  ), ncol = 2, byrow = TRUE)

  dots <- x %*% t(x)
  expected_idx <- t(apply(-dots, 1, order))[, 1:3]

  out <- nn(x, k = 3L, backend = "cpu", metric = "inner_product")

  expect_equal(attr(out, "backend"), "cpu")
  expect_equal(attr(out, "metric"), "inner_product")
  expect_true(isTRUE(attr(out, "exact")))
  expect_equal(unname(out$indices), unname(expected_idx))
  for (i in seq_len(nrow(x))) {
    best <- dots[i, expected_idx[i, 1L]]
    expect_equal(out$distances[i, ], best - dots[i, out$indices[i, ]], tolerance = 1e-12)
  }
})

test_that("non-euclidean metrics use only validated backend paths", {
  x <- scale(as.matrix(iris[1:20, 1:4]))

  auto <- nn(x, k = 4L, backend = "auto", metric = "cosine")
  expect_equal(attr(auto, "backend"), "cpu")
  expect_equal(attr(auto, "metric"), "cosine")

  auto_cor <- nn(x, k = 4L, backend = "auto", metric = "correlation")
  expect_equal(attr(auto_cor, "backend"), "cpu")
  expect_equal(attr(auto_cor, "metric"), "correlation")

  if (isTRUE(faiss_available())) {
    faiss_cos <- nn(x, k = 4L, backend = "cpu", method = "flat", metric = "cosine", n_threads = 2L)
    cpu_cos <- nn(x, k = 4L, backend = "cpu", method = "exact", metric = "cosine", n_threads = 2L)
    expect_equal(attr(faiss_cos, "backend"), "faiss_flat_cosine")
    expect_equal(attr(faiss_cos, "metric"), "cosine")
    expect_true(isTRUE(attr(faiss_cos, "exact")))
    expect_equal(unname(faiss_cos$indices), unname(cpu_cos$indices))
    expect_equal(unname(faiss_cos$distances), unname(cpu_cos$distances), tolerance = 1e-5)

    faiss_cor <- nn(x, k = 4L, backend = "cpu", method = "flat", metric = "correlation", n_threads = 2L)
    cpu_cor <- nn(x, k = 4L, backend = "cpu", method = "exact", metric = "correlation", n_threads = 2L)
    expect_equal(attr(faiss_cor, "backend"), "faiss_flat_correlation")
    expect_equal(attr(faiss_cor, "metric"), "correlation")
    expect_true(isTRUE(attr(faiss_cor, "exact")))
    expect_equal(unname(faiss_cor$indices), unname(cpu_cor$indices))
    expect_equal(unname(faiss_cor$distances), unname(cpu_cor$distances), tolerance = 1e-5)
  } else {
    expect_error(
      nn(x, k = 4L, backend = "cpu", method = "flat", metric = "cosine"),
      "FAISS"
    )
  }
  expect_error(
    internal_nn(x, k = 4L, backend = "cuda_cuvs_nndescent", metric = "inner_product"),
    "inner_product"
  )
  expect_error(
    internal_nn(x, k = 4L, backend = "cuda_cuvs_ivf_flat", metric = "inner_product"),
    "Direct cuVS IVF.*inner-product"
  )
  expect_error(
    internal_nn(x, k = 4L, backend = "cuda_cuvs_ivfpq", metric = "inner_product"),
    "Direct cuVS IVF.*inner-product"
  )
  expect_error(
    internal_nn(x, k = 4L, backend = "cuda_cuvs_bruteforce", metric = "correlation"),
    "Direct cuVS brute-force.*euclidean"
  )
  if (requireNamespace("RcppHNSW", quietly = TRUE)) {
    hnsw_ip <- internal_nn(x, k = 4L, backend = "hnsw", metric = "inner_product", n_threads = 2L)
    expect_equal(attr(hnsw_ip, "metric"), "inner_product")
    expect_equal(attr(hnsw_ip, "backend"), "hnsw")
  } else {
    expect_error(
      internal_nn(x, k = 4L, backend = "hnsw", metric = "inner_product"),
      "RcppHNSW"
    )
  }
})

test_that("FAISS Flat cosine and correlation preserve zero-row exact distances", {
  skip_if_not(isTRUE(faiss_available()), "FAISS is not available")
  x <- matrix(c(
    0, 0, 0,
    0, 0, 0,
    1, 0, 0,
    1, 2, 3,
    2, 4, 6
  ), ncol = 3, byrow = TRUE)

  for (metric in c("cosine", "correlation")) {
    flat <- nn(x, k = 5L, backend = "cpu", method = "flat", metric = metric, n_threads = 2L)
    exact <- nn(x, k = 5L, backend = "cpu", method = "exact", metric = metric, n_threads = 2L)
    expect_equal(unname(flat$indices), unname(exact$indices))
    expect_equal(unname(flat$distances), unname(exact$distances), tolerance = 1e-5)
  }
})

test_that("normalized IP routes rank zero-normalized rows before arbitrary IP ties", {
  out <- list(
    indices = matrix(c(
      3L, 4L,
      1L, 4L,
      1L, 2L,
      1L, 2L
    ), nrow = 4L, byrow = TRUE),
    distances = matrix(1, nrow = 4L, ncol = 2L)
  )
  attr(out, "self_query") <- TRUE
  fixed <- faissR:::restore_zero_normalized_ip_distances(
    out,
    data_zero = c(TRUE, TRUE, FALSE, FALSE),
    points_zero = c(TRUE, TRUE, FALSE, FALSE),
    exclude_self = FALSE
  )
  expect_equal(fixed$indices[1L, ], c(1L, 2L))
  expect_equal(fixed$distances[1L, ], c(0, 0))
  expect_equal(fixed$indices[2L, ], c(1L, 2L))
  expect_equal(fixed$distances[2L, ], c(0, 0))

  fixed_without_self <- faissR:::restore_zero_normalized_ip_distances(
    out,
    data_zero = c(TRUE, TRUE, FALSE, FALSE),
    points_zero = c(TRUE, TRUE, FALSE, FALSE),
    exclude_self = TRUE
  )
  expect_equal(fixed_without_self$indices[1L, ], c(2L, 3L))
  expect_equal(fixed_without_self$distances[1L, ], c(0, 1))
  expect_equal(fixed_without_self$indices[2L, ], c(1L, 3L))
  expect_equal(fixed_without_self$distances[2L, ], c(0, 1))
})

test_that("FAISS Flat zero-row normalized metrics match exact CPU at small k", {
  skip_if_not(isTRUE(faiss_available()), "FAISS is not available")
  x <- matrix(c(
    0, 0, 0,
    0, 0, 0,
    1, 0, 0,
    1, 2, 3,
    2, 4, 6
  ), ncol = 3, byrow = TRUE)

  for (metric in c("cosine", "correlation")) {
    flat <- nn(x, k = 2L, backend = "cpu", method = "flat", metric = metric, n_threads = 2L)
    exact <- nn(x, k = 2L, backend = "cpu", method = "exact", metric = metric, n_threads = 2L)
    expect_equal(unname(flat$indices), unname(exact$indices))
    expect_equal(unname(flat$distances), unname(exact$distances), tolerance = 1e-5)

    flat_without_self <- nn_without_self(
      x,
      k = 1L,
      backend = "cpu",
      method = "flat",
      metric = metric,
      n_threads = 2L
    )
    exact_without_self <- nn_without_self(
      x,
      k = 1L,
      backend = "cpu",
      method = "exact",
      metric = metric,
      n_threads = 2L
    )
    expect_equal(unname(flat_without_self$indices), unname(exact_without_self$indices))
    expect_equal(unname(flat_without_self$distances), unname(exact_without_self$distances), tolerance = 1e-5)
  }
})

test_that("nn chooses a practical default k and prints clearly", {
  set.seed(101)
  x <- matrix(rnorm(60), nrow = 20L)

  out <- nn(x, backend = "cpu")

  expect_s3_class(out, "faissR_nn")
  expect_equal(dim(out$indices), c(nrow(x), faissR:::auto_k(x, include_self = TRUE)))
  expect_equal(attr(out, "backend"), "cpu")
  expect_true(isTRUE(attr(out, "exact")))
  expect_true(isTRUE(attr(out, "self_query")))
  expect_output(print(out), "faissR KNN")
})

test_that("automatic nn is deterministic", {
  set.seed(12)
  data <- matrix(rnorm(200), ncol = 5)
  points <- matrix(rnorm(75), ncol = 5)

  first <- nn(data, points, k = 4)
  second <- nn(data, points, k = 4)

  expect_equal(second$indices, first$indices)
  expect_equal(second$distances, first$distances)
})

test_that("CPU nn handles non-small Euclidean work", {
  set.seed(121)
  data <- matrix(rnorm(200L * 30L), nrow = 200L)
  points <- matrix(rnorm(200L * 30L), nrow = 200L)

  out <- nn(data, points, k = 12L, backend = "cpu")

  expect_equal(dim(out$indices), c(nrow(points), 12L))
  expect_true(all(is.finite(out$distances)))
  expect_equal(attr(out, "backend"), "cpu")
  expect_true(isTRUE(attr(out, "exact")))
})

test_that("CPU nn row-major distance layout matches column-major fallback", {
  old_row_major <- Sys.getenv("FAISSR_NN_ROW_MAJOR", unset = NA_character_)
  old_fortran <- Sys.getenv("FAISSR_USE_FORTRAN_NN", unset = NA_character_)
  on.exit({
    if (is.na(old_row_major)) {
      Sys.unsetenv("FAISSR_NN_ROW_MAJOR")
    } else {
      Sys.setenv(FAISSR_NN_ROW_MAJOR = old_row_major)
    }
    if (is.na(old_fortran)) {
      Sys.unsetenv("FAISSR_USE_FORTRAN_NN")
    } else {
      Sys.setenv(FAISSR_USE_FORTRAN_NN = old_fortran)
    }
  }, add = TRUE)

  set.seed(1211)
  data <- matrix(rnorm(140L * 11L), nrow = 140L)
  points <- matrix(rnorm(65L * 11L), nrow = 65L)

  Sys.setenv(
    FAISSR_USE_FORTRAN_NN = "0",
    FAISSR_NN_ROW_MAJOR = "1"
  )
  row_major <- nn(data, points, k = 9L, backend = "cpu", n_threads = 2L)

  Sys.setenv(FAISSR_NN_ROW_MAJOR = "0")
  column_major <- nn(data, points, k = 9L, backend = "cpu", n_threads = 2L)

  expect_equal(attr(row_major, "memory_layout"), "row_major_contiguous")
  expect_equal(attr(column_major, "memory_layout"), "r_column_major")
  expect_true(isTRUE(attr(row_major, "row_major_copy")))
  expect_false(isTRUE(attr(column_major, "row_major_copy")))
  expect_equal(row_major$indices, column_major$indices)
  expect_equal(row_major$distances, column_major$distances, tolerance = 1e-12)
})

test_that("2D grid self KNN matches exact CPU neighbors", {
  set.seed(1212)
  x <- matrix(runif(6000L), ncol = 2L)
  k <- 16L

  exact <- nn(x, k = k, backend = "cpu", n_threads = 2L)
  grid <- internal_nn(x, k = k, backend = "cpu_grid2d", n_threads = 2L)

  expect_equal(attr(grid, "backend"), "cpu_grid2d")
  expect_true(isTRUE(attr(grid, "exact")))
  expect_equal(attr(grid, "spatial_index")$strategy, "native_exact_uniform_grid_2d")
  expect_equal(grid$indices, exact$indices)
  expect_equal(grid$distances, exact$distances, tolerance = 1e-12)
})

test_that("3D grid self KNN matches exact CPU neighbors", {
  set.seed(1213)
  x <- matrix(runif(7200L), ncol = 3L)
  k <- 14L

  exact <- nn(x, k = k, backend = "cpu", n_threads = 2L)
  grid <- internal_nn(x, k = k, backend = "cpu_grid", n_threads = 2L)

  expect_equal(attr(grid, "backend"), "cpu_grid3d")
  expect_true(isTRUE(attr(grid, "exact")))
  expect_equal(attr(grid, "spatial_index")$strategy, "native_exact_uniform_grid_3d")
  expect_equal(grid$indices, exact$indices)
  expect_equal(grid$distances, exact$distances, tolerance = 1e-12)
})

test_that("grid self KNN supports normalized cosine and correlation metrics", {
  set.seed(1214)
  x <- matrix(rnorm(9000L), ncol = 3L)
  k <- 10L

  for (metric in c("cosine", "correlation")) {
    exact <- nn(x, k = k, backend = "cpu", method = "exact", metric = metric, n_threads = 2L)
    grid <- internal_nn(x, k = k, backend = "cpu_grid", metric = metric, n_threads = 2L)

    expect_equal(attr(grid, "backend"), "cpu_grid3d", info = metric)
    expect_true(isTRUE(attr(grid, "exact")), info = metric)
    expect_equal(attr(grid, "metric"), metric)
    expect_match(grid$metric_transform, "normalize_then_euclidean", info = metric)
    expect_equal(attr(grid, "metric_transform"), grid$metric_transform, info = metric)
    expect_equal(
      attr(grid, "distance_transform"),
      "normalized_euclidean_squared_over_2_to_1_minus_similarity",
      info = metric
    )
    expect_match(attr(grid, "spatial_index")$metric_transform, "normalize_then_euclidean", info = metric)
    expect_equal(grid$indices, exact$indices, info = metric)
    expect_equal(grid$distances, exact$distances, tolerance = 1e-8, info = metric)
  }
})

test_that("public grid method rejects higher-dimensional data explicitly", {
  set.seed(12130)
  x <- matrix(runif(600L), nrow = 120L)

  expect_error(
    nn(x, k = 5L, backend = "cpu", method = "grid", n_threads = 2L),
    "two- or three-column"
  )
  expect_error(
    nn_without_self(x, k = 5L, backend = "cpu", method = "grid", n_threads = 2L),
    "two- or three-column"
  )
})

test_that("generic CPU spatial KNN keeps duplicate-heavy data on grid", {
  set.seed(1215)
  base <- matrix(runif(20L), ncol = 2L)
  x <- base[sample.int(nrow(base), 3000L, replace = TRUE), , drop = FALSE]
  k <- 12L

  exact <- nn_without_self(x, k = k, backend = "cpu", n_threads = 2L)
  spatial <- internal_nn_without_self(x, k = k, backend = "cpu_grid", n_threads = 2L)

  expect_equal(attr(spatial, "backend"), "cpu_grid2d")
  expect_equal(attr(spatial, "spatial_index")$strategy, "native_exact_uniform_grid_2d")
  expect_match(attr(spatial, "spatial_index")$reason, "duplicate_heavy_sample")
  spatial_dist <- t(apply(spatial$distances, 1L, sort))
  exact_dist <- t(apply(exact$distances, 1L, sort))
  expect_equal(spatial_dist, exact_dist, tolerance = 1e-12)
})

test_that("clustered self KNN reports approximation and preserves useful neighbors", {
  set.seed(122)
  n_per <- 80L
  labels <- rep(seq_len(3L), each = n_per)
  centers <- c(-4, 0, 4)
  x <- matrix(rnorm(length(labels) * 6L, sd = 0.45), ncol = 6L)
  x <- x + matrix(rep(centers[labels], 6L), ncol = 6L)
  k <- 12L

  exact <- faissR:::nn_without_self(x, k = k, backend = "cpu")
  clustered <- faissR:::clustered_self_knn(x, k = k, exclude_self = TRUE, seed = 122L)
  clustered <- faissR:::finish_nn_result(
    clustered,
    "cpu_clustered",
    k,
    TRUE,
    exact = FALSE
  )

  overlap <- mean(vapply(
    seq_len(nrow(x)),
    function(i) length(intersect(exact$indices[i, ], clustered$indices[i, ])) / k,
    numeric(1)
  ))

  expect_equal(dim(clustered$indices), c(nrow(x), k))
  expect_equal(dim(clustered$distances), c(nrow(x), k))
  expect_equal(attr(clustered, "backend"), "cpu_clustered")
  expect_false(isTRUE(attr(clustered, "exact")))
  expect_gt(overlap, 0.45)
  expect_output(print(clustered), "exact: false")
})

test_that("approximate landmark projection KNN matches exact when window covers all landmarks", {
  set.seed(124)
  landmarks <- matrix(rnorm(50L * 6L), nrow = 50L)
  queries <- matrix(rnorm(35L * 6L), nrow = 35L)
  k <- 7L

  exact <- nn(landmarks, queries, k = k, backend = "cpu")
  approx <- faissR:::landmark_projection_knn_approx_cpp(
    landmarks,
    queries,
    as.integer(k),
    4L,
    nrow(landmarks),
    124L,
    FALSE,
    1L
  )

  expect_equal(approx$indices, exact$indices)
  expect_equal(approx$distances, exact$distances, tolerance = 1e-6)
  expect_equal(approx$n_projections, 4L)
  expect_equal(approx$window, nrow(landmarks))
  expect_equal(approx$n_threads, 1L)
  expect_equal(approx$score_threads, 1L)
  expect_gt(approx$visited_stamp_mb_per_thread, 0)
})

test_that("subset landmark candidate KNN matches full candidate rows", {
  set.seed(125)
  x <- matrix(rnorm(48L * 5L), nrow = 48L, ncol = 5L)
  landmark_rows <- seq(1L, 48L, length.out = 12L)
  projection <- nn(x[landmark_rows, , drop = FALSE], x, k = 5L, backend = "cpu")
  rows <- c(3L, 11L, 24L, 39L)
  k <- 6L

  full <- faissR:::landmark_candidate_knn_cpp(
    x,
    projection$indices,
    as.integer(k),
    3L,
    4L,
    FALSE,
    1L
  )
  subset <- faissR:::landmark_candidate_knn_subset_cpp(
    x,
    projection$indices,
    as.integer(rows),
    as.integer(k),
    3L,
    4L,
    FALSE,
    1L
  )

  expect_equal(subset$row_ids, rows)
  expect_equal(subset$indices, full$indices[rows, , drop = FALSE])
  expect_equal(subset$distances, full$distances[rows, , drop = FALSE], tolerance = 1e-8)
})

test_that("clustered self KNN is not selected automatically", {
  expect_false(faissR:::should_use_clustered_self_knn(
    backend = "auto",
    self_query = TRUE,
    n = 6000L,
    p = 20L,
    k = 30L,
    work_size = 7.2e8
  ))
  expect_false(faissR:::should_use_clustered_self_knn(
    backend = "cpu",
    self_query = TRUE,
    n = 6000L,
    p = 20L,
    k = 30L,
    work_size = 7.2e8
  ))
  expect_false(faissR:::should_use_clustered_self_knn(
    backend = "auto",
    self_query = FALSE,
    n = 6000L,
    p = 20L,
    k = 30L,
    work_size = 7.2e8
  ))
})

test_that("removed CPU approximation backends are not public nn choices", {
  x <- matrix(rnorm(120L), nrow = 30L)
  expect_error(nn(x, k = 5L, backend = "cpu_clustered"), "must be one of")
  expect_error(nn(x, k = 5L, backend = "cpu_nndescent"), "must be one of")
  expect_error(nn(x, k = 5L, backend = "cpu_ivf"), "must be one of")
  expect_error(nn(x, k = 5L, backend = "cpu_annoy"), "must be one of")
})

test_that("CPU approximate selector chooses FAISS HNSW, RcppHNSW, or exact CPU", {
  selected <- faissR:::select_cpu_approx_backend(12000L, 30L, 30L)
  expect_true(selected %in% c("faiss_hnsw", "hnsw", "cpu"))
  if (faiss_available()) {
    expect_equal(selected, "faiss_hnsw")
  } else if (requireNamespace("RcppHNSW", quietly = TRUE)) {
    expect_equal(selected, "hnsw")
  }
  expect_true(faissR:::should_use_auto_cpu_approx_self_knn(
    self_query = TRUE,
    n = 12000L,
    p = 30L,
    k = 30L,
    work_size = 12000 * 12000 * 30
  ))
})

test_that("auto GPU preselector does not require CUDA when only CPU FAISS is available", {
  if (!cuda_available() && !cuvs_available()) {
    expect_equal(
      faissR:::resolve_auto_knn_gpu_backend(
        backend = "auto",
        self_query = TRUE,
        n_points = 20000L,
        n = 20000L,
        p = 50L,
        k = 50L,
        work_size = 20000 * 20000 * 50,
        metric = "euclidean"
      ),
      NA_character_
    )
  } else {
    skip("CUDA/cuVS runtime is available; CPU-only auto fallback is not exercised here.")
  }

  expect_equal(
    faissR:::resolve_auto_knn_gpu_backend(
      backend = "auto",
      self_query = TRUE,
      n_points = 400000L,
      n = 400000L,
      p = 50L,
      k = 50L,
      work_size = as.double(400000L) * as.double(400000L) * 50,
      metric = "cosine",
      cuda_available_value = TRUE,
      cuvs_available_value = TRUE,
      faiss_gpu_available_value = FALSE
    ),
    "cuda_cuvs_cagra"
  )
})

test_that("non-Euclidean CUDA auto requires FAISS GPU Flat support", {
  expect_equal(
    faissR:::cuda_auto_non_euclidean_backend(
      "cosine",
      requested_device = "auto",
      faiss_gpu_available_value = FALSE
    ),
    "cpu_auto"
  )
  expect_equal(
    faissR:::cuda_auto_non_euclidean_backend(
      "correlation",
      requested_device = "cuda",
      faiss_gpu_available_value = TRUE
    ),
    "faiss_gpu_flat_correlation"
  )
  expect_equal(
    faissR:::cuda_auto_non_euclidean_backend(
      "inner_product",
      requested_device = "cuda",
      faiss_gpu_available_value = TRUE
    ),
    "faiss_gpu_flat_ip"
  )
  expect_error(
    faissR:::cuda_auto_non_euclidean_backend(
      "cosine",
      requested_device = "cuda",
      faiss_gpu_available_value = FALSE,
      require_available = TRUE
    ),
    "FAISS GPU Flat"
  )
  expect_equal(
    faissR:::cuda_auto_non_euclidean_backend(
      "cosine",
      requested_device = "cuda",
      faiss_gpu_available_value = FALSE,
      require_available = FALSE
    ),
    "faiss_gpu_flat_cosine"
  )
})

test_that("explicit FAISS GPU NN backends require FAISS GPU availability", {
  if (faiss_gpu_available()) {
    skip("FAISS GPU is available; unavailable-runtime guard is not exercised here.")
  }

  x <- matrix(rnorm(80L), nrow = 20L)
  gpu_backends <- c(
    "faiss_gpu_flat_l2",
    "faiss_gpu_flat_ip",
    "faiss_gpu_flat_cosine",
    "faiss_gpu_flat_correlation",
    "faiss_gpu_ivf_flat",
    "faiss_gpu_ivfpq",
    "faiss_gpu_cagra"
  )

  for (backend in gpu_backends) {
    expect_error(
      internal_nn(x, k = 4L, backend = backend, n_threads = 2L),
      "FAISS.*GPU|GPU.*FAISS",
      info = backend
    )
  }
})

test_that("CUDA resolver uses FAISS GPU availability for routes with alternatives", {
  expected_exact <- if (faiss_gpu_available()) {
    "faiss_gpu_flat_l2"
  } else if (cuvs_available()) {
    "cuda_cuvs_bruteforce"
  } else {
    "cuda"
  }
  expected_bruteforce <- if (cuvs_available()) {
    "cuda_cuvs_bruteforce"
  } else if (faiss_gpu_available()) {
    "faiss_gpu_flat_l2"
  } else {
    "cuda"
  }
  expected_cagra <- if (faiss_gpu_available()) "faiss_gpu_cagra" else "cuda_cuvs_cagra"

  expect_equal(
    faissR:::resolve_public_nn_backend("cuda", "exact", "euclidean"),
    expected_exact
  )
  expect_equal(
    faissR:::resolve_public_nn_backend("cuda", "bruteforce", "euclidean"),
    expected_bruteforce
  )
  expect_equal(
    faissR:::resolve_public_nn_backend("cuda", "cagra", "euclidean"),
    expected_cagra
  )
})

test_that("CUDA CAGRA implementation can be selected by option", {
  old <- getOption("faissR.cagra_implementation")
  on.exit(options(faissR.cagra_implementation = old), add = TRUE)

  options(faissR.cagra_implementation = "faiss_gpu")
  expect_equal(
    faissR:::resolve_public_nn_backend("cuda", "cagra", "euclidean"),
    "faiss_gpu_cagra"
  )
  expect_false(faissR:::public_nn_cuda_route_available(
    "cagra",
    "euclidean",
    faiss_gpu_available_value = FALSE,
    cuvs_available_value = TRUE
  ))
  expect_true(faissR:::public_nn_cuda_route_available(
    "cagra",
    "euclidean",
    faiss_gpu_available_value = TRUE,
    cuvs_available_value = FALSE
  ))

  options(faissR.cagra_implementation = "cuvs")
  expect_equal(
    faissR:::resolve_public_nn_backend("cuda", "cagra", "euclidean"),
    "cuda_cuvs_cagra"
  )
  expect_true(faissR:::public_nn_cuda_route_available(
    "cagra",
    "euclidean",
    faiss_gpu_available_value = FALSE,
    cuvs_available_value = TRUE
  ))
  expect_false(faissR:::public_nn_cuda_route_available(
    "cagra",
    "euclidean",
    faiss_gpu_available_value = TRUE,
    cuvs_available_value = FALSE
  ))
  expect_true(faissR:::public_nn_cuda_route_available(
    "cagra",
    "cosine",
    faiss_gpu_available_value = FALSE,
    cuvs_available_value = TRUE
  ))
  expect_true(faissR:::public_nn_cuda_route_available(
    "cagra",
    "inner_product",
    faiss_gpu_available_value = FALSE,
    cuvs_available_value = TRUE
  ))

  options(faissR.cagra_implementation = "unknown")
  expect_equal(
    faissR:::resolve_cuda_cagra_backend(
      faiss_gpu_available_value = TRUE,
      cuvs_available_value = TRUE
    ),
    "faiss_gpu_cagra"
  )

  options(faissR.cagra_implementation = "faiss_gpu")
  expect_false(faissR:::public_nn_cuda_route_available(
    "cagra",
    "cosine",
    faiss_gpu_available_value = FALSE,
    cuvs_available_value = TRUE
  ))
  expect_false(faissR:::public_nn_cuda_route_available(
    "cagra",
    "inner_product",
    faiss_gpu_available_value = FALSE,
    cuvs_available_value = TRUE
  ))

  options(faissR.cagra_implementation = "cuvs")
  expect_false(faissR:::public_nn_cuda_route_available(
    "cagra",
    "correlation",
    faiss_gpu_available_value = TRUE,
    cuvs_available_value = FALSE
  ))
  expect_false(faissR:::public_nn_cuda_route_available(
    "cagra",
    "inner_product",
    faiss_gpu_available_value = TRUE,
    cuvs_available_value = FALSE
  ))
})

test_that("CUDA CAGRA implementation can be selected per call", {
  old <- getOption("faissR.cagra_implementation")
  on.exit(options(faissR.cagra_implementation = old), add = TRUE)

  options(faissR.cagra_implementation = "cuvs")
  local({
    faissR:::set_call_cagra_implementation("faiss_gpu")
    expect_equal(getOption("faissR.cagra_implementation"), "faiss_gpu")
    expect_equal(
      faissR:::resolve_public_nn_backend("cuda", "cagra", "euclidean"),
      "faiss_gpu_cagra"
    )
  })
  expect_equal(getOption("faissR.cagra_implementation"), "cuvs")

  expect_equal(faissR:::normalize_cagra_implementation_arg("direct-cuvs"), "cuvs")
  expect_equal(faissR:::normalize_cagra_implementation_arg("faiss"), "faiss_gpu")
  expect_error(
    nn_without_self(
      matrix(rnorm(20), ncol = 4),
      k = 2L,
      backend = "cpu",
      cagra_implementation = "metal"
    ),
    "cagra_implementation"
  )
})


test_that("public backend and method resolver maps device plus method", {
  expect_equal(
    faissR:::resolve_public_nn_backend("cpu", "auto", "euclidean"),
    "cpu_auto"
  )
  expect_equal(
    faissR:::resolve_public_nn_backend("auto", "auto", "cosine"),
    "auto"
  )
  expect_equal(
    faissR:::resolve_public_nn_backend("auto", "hnsw", "euclidean"),
    if (faiss_available()) "faiss_hnsw" else "hnsw"
  )
  expect_error(
    faissR:::resolve_public_nn_backend("auto", "removed_method", "cosine"),
    "`method` must be one of"
  )
  expect_error(
    faissR:::resolve_public_nn_backend("auto", "removed_method", "inner_product"),
    "`method` must be one of"
  )
  expect_equal(
    faissR:::resolve_public_nn_backend("auto", "nsg", "euclidean"),
    if (cuda_available()) "cuda_nsg" else "cpu_nsg"
  )
  expect_equal(
    faissR:::resolve_public_nn_backend("cpu", "grid", "euclidean"),
    "cpu_grid"
  )
  expect_equal(
    faissR:::resolve_public_nn_backend("cpu", "grid", "cosine"),
    "cpu_grid"
  )
  expect_error(
    faissR:::resolve_public_nn_backend("cpu", "grid", "inner_product"),
    "inner_product"
  )
  expect_equal(
    faissR:::resolve_public_nn_backend("cuda", "auto", "euclidean"),
    "cuda_auto"
  )
  expect_equal(
    faissR:::resolve_public_nn_backend("cuda", "auto", "cosine"),
    "cuda_auto"
  )
  expect_equal(
    faissR:::resolve_public_nn_backend("cuda", "auto", "correlation"),
    "cuda_auto"
  )
  expect_equal(
    faissR:::resolve_public_nn_backend("cuda", "auto", "inner_product"),
    "cuda_auto"
  )
  expect_equal(
    faissR:::resolve_public_nn_backend("cuda", "grid", "euclidean"),
    "cuda_grid"
  )
  expect_equal(
    faissR:::resolve_public_nn_backend("cuda", "grid", "correlation"),
    "cuda_grid"
  )
  expect_error(
    faissR:::resolve_public_nn_backend("cpu", "cagra", "euclidean"),
    "cagra.*only available.*cuda"
  )
  expect_equal(
    faissR:::resolve_public_nn_backend("cuda", "hnsw", "euclidean"),
    "cuda_cuvs_hnsw"
  )
  expect_equal(
    faissR:::resolve_public_nn_backend("cpu", "ivf", "euclidean"),
    "faiss_ivf"
  )
  expect_equal(
    faissR:::resolve_public_nn_backend("cpu", "hnsw", "euclidean"),
    if (faiss_available()) "faiss_hnsw" else "hnsw"
  )
  expect_equal(
    faissR:::resolve_public_nn_backend("cpu", "hnsw", "cosine"),
    if (faiss_available()) "faiss_hnsw" else "hnsw"
  )
  expect_equal(
    faissR:::resolve_public_nn_backend("cpu", "hnsw", "correlation"),
    if (faiss_available()) "faiss_hnsw" else "hnsw"
  )
  expect_equal(
    faissR:::resolve_public_nn_backend("cpu", "hnsw", "inner_product"),
    if (faiss_available()) "faiss_hnsw" else "hnsw"
  )
  expect_equal(
    faissR:::resolve_public_nn_backend("cpu", "hnsw", "euclidean"),
    if (faiss_available()) "faiss_hnsw" else "hnsw"
  )
  expect_error(
    faissR:::resolve_public_nn_backend("cpu", "HNSW", "euclidean"),
    "canonical lowercase"
  )
  expect_error(
    faissR:::resolve_public_nn_backend("cuda", "CAGRA", "euclidean"),
    "canonical lowercase"
  )
  if (faiss_available()) {
    x_hnsw <- matrix(rnorm(240L), nrow = 40L)
    for (metric in c("cosine", "correlation", "inner_product")) {
      out <- nn(x_hnsw, k = 5L, backend = "cpu", method = "hnsw", metric = metric, n_threads = 2L)
      expect_equal(attr(out, "backend"), "faiss_hnsw")
      expect_equal(attr(out, "metric"), metric)
      expect_equal(attr(out, "approximation")$library, "faiss")
    }
  }
  expect_equal(
    faissR:::resolve_public_nn_backend("cpu", "flat", "inner_product"),
    "faiss_flat_ip"
  )
  expect_equal(
    faissR:::resolve_public_nn_backend("cpu", "flat", "cosine"),
    "faiss_flat_cosine"
  )
  expect_equal(
    faissR:::resolve_public_nn_backend("cpu", "flat", "correlation"),
    "faiss_flat_correlation"
  )
  expect_equal(
    faissR:::resolve_public_nn_backend("cuda", "flat", "inner_product"),
    "faiss_gpu_flat_ip"
  )
  expect_equal(
    faissR:::resolve_public_nn_backend("cuda", "flat", "cosine"),
    "faiss_gpu_flat_cosine"
  )
  expect_equal(
    faissR:::resolve_public_nn_backend("cuda", "flat", "correlation"),
    "faiss_gpu_flat_correlation"
  )
  expect_equal(
    faissR:::resolve_public_nn_backend("cuda", "exact", "cosine"),
    "faiss_gpu_flat_cosine"
  )
  expect_equal(
    faissR:::resolve_public_nn_backend("cuda", "bruteforce", "correlation"),
    "faiss_gpu_flat_correlation"
  )
  expect_equal(
    faissR:::resolve_public_nn_backend("cpu", "ivf", "inner_product"),
    "faiss_ivf"
  )
  expect_equal(
    faissR:::resolve_public_nn_backend("cpu", "ivf", "cosine"),
    "faiss_ivf"
  )
  expect_equal(
    faissR:::resolve_public_nn_backend("cuda", "ivf", "inner_product"),
    "faiss_gpu_ivf_flat"
  )
  expect_equal(
    faissR:::resolve_public_nn_backend("cuda", "ivf", "correlation"),
    "faiss_gpu_ivf_flat"
  )
  expect_equal(
    faissR:::resolve_public_nn_backend("cpu", "ivfpq", "inner_product"),
    "faiss_ivfpq"
  )
  expect_equal(
    faissR:::resolve_public_nn_backend("cuda", "ivfpq", "cosine"),
    "faiss_gpu_ivfpq"
  )
  expect_equal(
    faissR:::resolve_public_nn_backend("cpu", "vamana", "euclidean"),
    "cpu_vamana"
  )
  expect_equal(
    faissR:::resolve_public_nn_backend("cuda", "vamana", "correlation"),
    "cuda_vamana"
  )
  expect_equal(
    faissR:::resolve_public_nn_backend("cuda", "vamana", "inner_product"),
    "cuda_vamana"
  )
  expect_equal(
    faissR:::resolve_public_nn_backend("cpu", "nsg", "euclidean"),
    "cpu_nsg"
  )
  expect_equal(
    faissR:::resolve_public_nn_backend("cpu", "nsg", "correlation"),
    "cpu_nsg"
  )
  expect_equal(
    faissR:::resolve_public_nn_backend("cpu", "nsg", "inner_product"),
    "cpu_nsg"
  )
  expect_equal(
    faissR:::resolve_public_nn_backend("cuda", "nsg", "euclidean"),
    "cuda_nsg"
  )
  expect_equal(
    faissR:::resolve_public_nn_backend("cuda", "nsg", "cosine"),
    "cuda_nsg"
  )
  expect_equal(
    faissR:::resolve_public_nn_backend("cuda", "nsg", "correlation"),
    "cuda_nsg"
  )
  expect_equal(
    faissR:::resolve_public_nn_backend("cuda", "nsg", "inner_product"),
    "cuda_nsg"
  )
  expect_equal(
    faissR:::resolve_public_nn_backend("cpu", "nndescent", "inner_product"),
    "cpu_nndescent"
  )
  expect_equal(
    faissR:::resolve_public_nn_backend("cuda", "nndescent", "correlation"),
    "cuda_cuvs_nndescent"
  )
  expect_equal(
    faissR:::resolve_public_nn_backend("cuda", "nndescent", "inner_product"),
    "cuda_native_nndescent"
  )
  expect_equal(
    faissR:::resolve_public_nn_backend("cuda", "hnsw", "euclidean"),
    "cuda_cuvs_hnsw"
  )
  expect_equal(
    faissR:::resolve_public_nn_backend("cuda", "hnsw", "cosine"),
    "cuda_cuvs_hnsw"
  )
  expect_equal(
    faissR:::resolve_public_nn_backend("cuda", "hnsw", "correlation"),
    "cuda_cuvs_hnsw"
  )
  expect_equal(
    faissR:::resolve_public_nn_backend("cuda", "hnsw", "inner_product"),
    "cuda_cuvs_hnsw"
  )
  expect_true(
    faissR:::resolve_public_nn_backend("cuda", "cagra", "inner_product") %in%
      c("faiss_gpu_cagra", "cuda_cuvs_cagra")
  )
  expect_true(
    faissR:::resolve_public_nn_backend("cuda", "cagra", "cosine") %in%
    c("faiss_gpu_cagra", "cuda_cuvs_cagra")
  )
  expect_error(
    faissR:::resolve_public_nn_backend("cpu", "faiss_hnsw", "euclidean"),
    "method.*must be one of"
  )
  expect_error(
    faissR:::resolve_public_nn_backend("cpu", "faiss_ivf", "euclidean"),
    "method.*must be one of"
  )
  expect_error(
    faissR:::resolve_public_nn_backend("cpu", "faiss_nndescent", "euclidean"),
    "method.*must be one of"
  )
  expect_error(
    faissR:::resolve_public_nn_backend("cuda", "faiss_gpu_ivf_flat", "euclidean"),
    "method.*must be one of"
  )
  expect_error(
    faissR:::resolve_public_nn_backend("cuda", "cuda_cuvs_nndescent", "euclidean"),
    "method.*must be one of"
  )
  expect_error(
    faissR:::resolve_public_nn_backend("cuda", "faiss_gpu_cagra", "euclidean"),
    "method.*must be one of"
  )
  expect_true(
    faissR:::resolve_public_nn_backend("cuda", "cagra", "euclidean") %in%
      c("faiss_gpu_cagra", "cuda_cuvs_cagra")
  )
})

test_that("nn_capabilities supported public rows resolve through the public router", {
  caps <- nn_capabilities()
  supported <- caps[caps$supported, c("backend", "method", "metric"), drop = FALSE]
  expect_gt(nrow(supported), 0L)

  for (i in seq_len(nrow(supported))) {
    row <- supported[i, , drop = FALSE]
    label <- paste(row$backend, row$method, row$metric, sep = "/")
    err <- tryCatch(
      {
        faissR:::resolve_public_nn_backend(row$backend, row$method, row$metric)
        NULL
      },
      error = identity
    )
    expect_null(err, label = label)
  }
})

test_that("CPU-supported public method/metric rows execute on smoke data", {
  set.seed(9091)
  dense <- matrix(rnorm(20L * 6L), nrow = 20L)
  spatial <- matrix(runif(40L * 2L), ncol = 2L)
  nsg_dense <- matrix(rnorm(200L * 8L), nrow = 200L)
  caps <- nn_capabilities()
  checked <- caps[
    caps$supported &
      caps$backend == "cpu" &
      caps$method != "auto",
    c("method", "metric"),
    drop = FALSE
  ]

  for (i in seq_len(nrow(checked))) {
    method <- checked$method[[i]]
    metric <- checked$metric[[i]]
    if (identical(method, "ivfpq")) {
      next
    }
    data <- switch(
      method,
      grid = spatial,
      nsg = nsg_dense,
      dense
    )
    label <- paste(method, metric)
    out <- nn_without_self(
      data,
      k = 3L,
      backend = "cpu",
      method = method,
      metric = metric,
      n_threads = 2L
    )
    expect_true(inherits(out, "faissR_nn"), info = label)
    expect_equal(dim(out$indices), c(nrow(data), 3L), info = label)
    expect_equal(dim(as.matrix(out$distances)), c(nrow(data), 3L), info = label)
    expect_equal(attr(out, "metric"), metric, info = label)
  }
})

test_that("native Vamana returns metric-aware CPU self-KNN results", {
  set.seed(240124)
  x <- matrix(rnorm(90L * 5L), nrow = 90L)

  for (metric in c("euclidean", "cosine", "correlation", "inner_product")) {
    out <- internal_nn_without_self(
      x,
      k = 6L,
      backend = "cpu_vamana",
      metric = metric,
      n_threads = 2L
    )
    approx <- attr(out, "approximation")
    expect_equal(dim(out$indices), c(nrow(x), 6L), info = metric)
    expect_equal(attr(out, "backend"), "cpu_vamana", info = metric)
    expect_equal(attr(out, "metric"), metric, info = metric)
    expect_equal(approx$backend, "cpu_vamana", info = metric)
    expect_equal(approx$strategy, "native_vamana_candidate_graph", info = metric)
    expect_true(all(is.finite(out$distances)), info = metric)
  }
})

test_that("native CPU NSG returns metric-aware self-KNN results", {
  set.seed(240125)
  x <- matrix(rnorm(90L * 5L), nrow = 90L)

  for (metric in c("euclidean", "cosine", "correlation", "inner_product")) {
    out <- internal_nn_without_self(
      x,
      k = 6L,
      backend = "cpu_nsg",
      metric = metric,
      n_threads = 2L
    )
    approx <- attr(out, "approximation")
    expect_equal(dim(out$indices), c(nrow(x), 6L), info = metric)
    expect_equal(dim(out$distances), c(nrow(x), 6L), info = metric)
    expect_equal(attr(out, "backend"), "cpu_nsg", info = metric)
    expect_equal(attr(out, "metric"), metric, info = metric)
    expect_equal(approx$backend, "cpu_nsg", info = metric)
    expect_equal(approx$strategy, "native_cpu_nsg_candidate_graph", info = metric)
    expect_equal(approx$accelerator, "cpu", info = metric)
    expect_equal(approx$graph_k_cap, 512L, info = metric)
    expect_match(approx$tuning_rule, "cpu_nsg", info = metric)
    expect_true(all(is.finite(out$distances)), info = metric)
    expect_true(all(out$indices >= 1L & out$indices <= nrow(x)), info = metric)
  }
})

test_that("public CPU NSG uses native route for euclidean instead of unsafe FAISS NSG", {
  set.seed(240126)
  x <- matrix(rnorm(120L * 8L), nrow = 120L)

  out <- nn_without_self(
    x,
    k = 5L,
    backend = "cpu",
    method = "nsg",
    metric = "euclidean",
    n_threads = 2L
  )

  expect_equal(attr(out, "backend"), "cpu_nsg")
  expect_equal(out$backend_used, "cpu_nsg")
  expect_equal(dim(out$indices), c(nrow(x), 5L))
  expect_equal(dim(out$distances), c(nrow(x), 5L))
  expect_equal(attr(out, "approximation")$strategy, "native_cpu_nsg_candidate_graph")
  expect_true(all(is.finite(out$distances)))
})

test_that("native NSG tuning is backend-specific", {
  old_options <- options(
    faissR.cpu_nsg_r = NULL,
    faissR.cpu_nsg_graph_k = NULL,
    faissR.cuda_nsg_r = NULL,
    faissR.cuda_nsg_graph_k = NULL
  )
  on.exit(options(old_options), add = TRUE)

  cpu <- faissR:::native_nsg_params(
    n = 70000L,
    p = 784L,
    k = 50L,
    metric = "cosine",
    backend = "cpu"
  )
  cuda <- faissR:::native_nsg_params(
    n = 70000L,
    p = 784L,
    k = 50L,
    metric = "cosine",
    backend = "cuda"
  )

  expect_equal(cpu$backend, "cpu")
  expect_equal(cuda$backend, "cuda")
  expect_equal(cpu$graph_k_cap, 512L)
  expect_equal(cuda$graph_k_cap, 255L)
  expect_match(cpu$tuning_rule, "cpu_nsg")
  expect_match(cuda$tuning_rule, "cuda_nsg")

  options(
    faissR.cpu_nsg_graph_k = 400L,
    faissR.cuda_nsg_graph_k = 400L
  )
  cpu_override <- faissR:::native_nsg_params(1000L, 256L, 50L, backend = "cpu")
  cuda_override <- faissR:::native_nsg_params(1000L, 256L, 50L, backend = "cuda")
  expect_equal(cpu_override$requested_graph_k, 400L)
  expect_equal(cpu_override$graph_k, 400L)
  expect_equal(cuda_override$requested_graph_k, 255L)
  expect_equal(cuda_override$graph_k, 255L)
})

test_that("nearest-neighbour results expose resolved backend metadata", {
  x <- matrix(rnorm(200), ncol = 4)
  out <- nn_without_self(x, k = 5L, backend = "cpu", method = "exact", n_threads = 2L)
  expect_equal(attr(out, "backend"), "cpu")
  expect_equal(attr(out, "resolved_backend"), "cpu")
})

test_that("public tuning policy normalizes and can override defaults", {
  expect_equal(faissR:::normalize_nn_tuning("auto"), "auto")
  expect_equal(faissR:::normalize_nn_tuning("none"), "off")
  expect_equal(faissR:::normalize_nn_tuning("pilot"), "pilot")
  expect_error(faissR:::normalize_nn_tuning("aggressive"), "tuning")

  old_ivf <- getOption("faissR.faiss_gpu_ivf_tune_policy")
  old_cagra <- getOption("faissR.cuvs_cagra_tune_policy")
  on.exit(options(
    faissR.faiss_gpu_ivf_tune_policy = old_ivf,
    faissR.cuvs_cagra_tune_policy = old_cagra
  ), add = TRUE)
  options(
    faissR.faiss_gpu_ivf_tune_policy = NULL,
    faissR.cuvs_cagra_tune_policy = NULL
  )
  expect_equal(faissR:::faiss_gpu_ivf_tune_policy("auto"), "fixed")
  expect_equal(faissR:::cuvs_cagra_tune_policy("auto"), "fixed")
  options(
    faissR.faiss_gpu_ivf_tune_policy = "not-a-policy",
    faissR.cuvs_cagra_tune_policy = "not-a-policy"
  )
  expect_equal(faissR:::faiss_gpu_ivf_tune_policy("auto"), "fixed")
  expect_equal(faissR:::cuvs_cagra_tune_policy("auto"), "fixed")
  options(
    faissR.faiss_gpu_ivf_tune_policy = "fixed",
    faissR.cuvs_cagra_tune_policy = "fixed"
  )

  expect_equal(faissR:::faiss_gpu_ivf_tune_policy("auto"), "fixed")
  expect_equal(faissR:::faiss_gpu_ivf_tune_policy("pilot"), "pilot")
  expect_false(faissR:::faiss_gpu_ivf_should_tune(
    matrix(0, nrow = 30000L, ncol = 2L),
    k = 50L,
    self_query = TRUE,
    tuning = "auto"
  ))
  expect_false(faissR:::faiss_gpu_ivf_should_tune(
    matrix(0, nrow = 30000L, ncol = 2L),
    k = 50L,
    self_query = TRUE,
    tuning = "off"
  ))
  expect_equal(faissR:::cuvs_cagra_tune_policy("auto"), "fixed")
  expect_equal(faissR:::cuvs_cagra_tune_policy("cache"), "cache")
  expect_false(faissR:::cuvs_cagra_should_tune(
    matrix(0, nrow = 30000L, ncol = 2L),
    k = 50L,
    self_query = TRUE,
    tuning = "auto"
  ))
})


test_that("CPU auto selector is shape-aware", {
  small <- faissR:::select_cpu_auto_backend(
    self_query = TRUE,
    n = 1000L,
    p = 20L,
    n_points = 1000L,
    k = 50L,
    work_size = 1000 * 1000 * 20
  )
  expect_equal(small, "cpu")

  low_dim <- faissR:::select_cpu_auto_backend(
    self_query = TRUE,
    n = 20000L,
    p = 2L,
    n_points = 20000L,
    k = 50L,
    work_size = 20000 * 20000 * 2
  )
  expect_equal(low_dim, "cpu_grid")

  low_dim_cosine <- faissR:::select_cpu_auto_backend(
    self_query = TRUE,
    n = 20000L,
    p = 2L,
    n_points = 20000L,
    k = 50L,
    work_size = 20000 * 20000 * 2,
    metric = "cosine"
  )
  expect_equal(low_dim_cosine, "cpu_grid")

  low_dim_correlation <- faissR:::select_cpu_auto_backend(
    self_query = TRUE,
    n = 20000L,
    p = 3L,
    n_points = 20000L,
    k = 50L,
    work_size = 20000 * 20000 * 3,
    metric = "correlation"
  )
  expect_equal(low_dim_correlation, "cpu_grid")

  low_dim_inner_product <- faissR:::select_cpu_auto_backend(
    self_query = TRUE,
    n = 20000L,
    p = 2L,
    n_points = 20000L,
    k = 50L,
    work_size = 20000 * 20000 * 2,
    metric = "inner_product"
  )
  expect_false(identical(low_dim_inner_product, "cpu_grid"))

  large <- faissR:::select_cpu_auto_backend(
    self_query = TRUE,
    n = 20000L,
    p = 100L,
    n_points = 20000L,
    k = 50L,
    work_size = 20000 * 20000 * 100
  )
  expect_true(large %in% c("faiss_hnsw", "hnsw", "cpu_nndescent"))
  if (faiss_available()) expect_equal(large, "faiss_hnsw")

  million_row <- faissR:::select_cpu_auto_backend(
    self_query = TRUE,
    n = 1200000L,
    p = 32L,
    n_points = 1200000L,
    k = 50L,
    work_size = 1200000 * 1200000 * 32
  )
  expect_true(million_row %in% c("faiss_ivf", "hnsw", "cpu_nndescent"))
  if (faiss_available()) expect_equal(million_row, "faiss_ivf")

  small_non_euclidean <- faissR:::select_cpu_auto_backend(
    self_query = TRUE,
    n = 1000L,
    p = 20L,
    n_points = 1000L,
    k = 50L,
    work_size = 1000 * 1000 * 20,
    metric = "cosine"
  )
  expect_equal(small_non_euclidean, "cpu")

  large_query_cosine <- faissR:::select_cpu_auto_backend(
    self_query = FALSE,
    n = 10000L,
    p = 100L,
    n_points = 5000L,
    k = 50L,
    work_size = 10000 * 5000 * 100,
    metric = "cosine"
  )
  expect_true(large_query_cosine %in% c("faiss_flat_cosine", "cpu"))
  if (faiss_available()) expect_equal(large_query_cosine, "faiss_flat_cosine")

  large_query_correlation <- faissR:::select_cpu_auto_backend(
    self_query = FALSE,
    n = 10000L,
    p = 100L,
    n_points = 5000L,
    k = 50L,
    work_size = 10000 * 5000 * 100,
    metric = "correlation"
  )
  expect_true(large_query_correlation %in% c("faiss_flat_correlation", "cpu"))
  if (faiss_available()) expect_equal(large_query_correlation, "faiss_flat_correlation")

  large_query_inner_product <- faissR:::select_cpu_auto_backend(
    self_query = FALSE,
    n = 10000L,
    p = 100L,
    n_points = 5000L,
    k = 50L,
    work_size = 10000 * 5000 * 100,
    metric = "inner_product"
  )
  expect_true(large_query_inner_product %in% c("faiss_flat_ip", "cpu"))
  if (faiss_available()) expect_equal(large_query_inner_product, "faiss_flat_ip")

  large_non_euclidean <- faissR:::select_cpu_auto_backend(
    self_query = TRUE,
    n = 20000L,
    p = 100L,
    n_points = 20000L,
    k = 50L,
    work_size = 20000 * 20000 * 100,
    metric = "cosine"
  )
  expect_true(large_non_euclidean %in% c("faiss_hnsw", "hnsw", "cpu_nsg", "cpu_nndescent"))
  if (faiss_available()) {
    expect_equal(large_non_euclidean, "faiss_hnsw")
  } else if (requireNamespace("RcppHNSW", quietly = TRUE)) {
    expect_equal(large_non_euclidean, "hnsw")
  } else {
    expect_equal(large_non_euclidean, "cpu_nsg")
  }
})

test_that("CPU auto selector is k and metric aware on benchmark k grid", {
  n <- 70000L
  p <- 784L
  work_size <- as.double(n) * as.double(n) * as.double(p)
  k_values <- c(5L, 10L, 15L, 50L, 100L)
  metrics <- c("euclidean", "cosine", "correlation", "inner_product")
  routes <- expand.grid(k = k_values, metric = metrics, stringsAsFactors = FALSE)
  routes$route <- vapply(seq_len(nrow(routes)), function(i) {
    faissR:::select_cpu_auto_backend(
      self_query = TRUE,
      n = n,
      p = p,
      n_points = n,
      k = routes$k[[i]],
      work_size = work_size,
      metric = routes$metric[[i]]
    )
  }, character(1L))

  expect_equal(
    routes$route[routes$metric == "euclidean"],
    rep(if (faiss_available()) "faiss_hnsw" else if (requireNamespace("RcppHNSW", quietly = TRUE)) "hnsw" else "cpu_nndescent", length(k_values))
  )

  for (metric in c("cosine", "correlation", "inner_product")) {
    k5_route <- routes$route[routes$metric == metric & routes$k == 5L]
    if (faiss_available()) {
      expect_equal(k5_route, switch(
        metric,
        cosine = "faiss_flat_cosine",
        correlation = "faiss_flat_correlation",
        inner_product = "faiss_flat_ip"
      ))
    } else {
      expect_equal(k5_route, "cpu")
    }
    expected_large_k <- if (faiss_available()) {
      rep("faiss_hnsw", 4L)
    } else if (requireNamespace("RcppHNSW", quietly = TRUE)) {
      rep("hnsw", 4L)
    } else {
      c("cpu_nndescent", "cpu_nndescent", "cpu_nsg", "cpu_nsg")
    }
    expect_equal(routes$route[routes$metric == metric & routes$k >= 10L], expected_large_k)
  }
})

test_that("FAISS HNSW defaults are shape, k, and metric aware without pilot tuning", {
  old_options <- options(
    faissR.faiss_hnsw_m = NULL,
    faissR.faiss_hnsw_ef_construction = NULL,
    faissR.faiss_hnsw_ef_search = NULL
  )
  on.exit(options(old_options), add = TRUE)

  speed <- faissR:::faiss_hnsw_params(
    k = 5L,
    n = 20000L,
    p = 32L,
    metric = "euclidean"
  )
  expect_equal(speed$policy, "auto_shape_metric")
  expect_equal(speed$rule, "small_k_speed")
  expect_equal(speed$m, 24L)
  expect_equal(speed$ef_construction, 120L)
  expect_equal(speed$ef_search, 80L)

  balanced <- faissR:::faiss_hnsw_params(
    k = 50L,
    n = 70000L,
    p = 784L,
    metric = "euclidean"
  )
  expect_equal(balanced$rule, "balanced_shape_metric")
  expect_equal(balanced$m, 32L)
  expect_equal(balanced$ef_construction, 200L)
  expect_equal(balanced$ef_search, 150L)
  expect_true(isTRUE(balanced$high_dim))
  expect_true(isTRUE(balanced$large_n))

  small_k_metric <- faissR:::faiss_hnsw_params(
    k = 5L,
    n = 70000L,
    p = 784L,
    metric = "cosine"
  )
  expect_equal(small_k_metric$rule, "balanced_small_k_metric")
  expect_equal(small_k_metric$m, 32L)
  expect_equal(small_k_metric$ef_construction, 160L)
  expect_equal(small_k_metric$ef_search, 120L)
  expect_true(isTRUE(small_k_metric$small_k))
  expect_false(isTRUE(small_k_metric$large_k))
  expect_true(isTRUE(small_k_metric$non_euclidean))
  expect_equal(anyDuplicated(names(small_k_metric)), 0L)

  high_recall <- faissR:::faiss_hnsw_params(
    k = 100L,
    n = 70000L,
    p = 784L,
    metric = "correlation"
  )
  expect_equal(high_recall$rule, "high_recall_shape_metric")
  expect_equal(high_recall$m, 48L)
  expect_equal(high_recall$ef_construction, 240L)
  expect_equal(high_recall$ef_search, 300L)
  expect_true(isTRUE(high_recall$non_euclidean))
  expect_true(isTRUE(high_recall$large_k))

  options(faissR.faiss_hnsw_m = 40L)
  manual <- faissR:::faiss_hnsw_params(
    k = 50L,
    n = 70000L,
    p = 784L,
    metric = "euclidean"
  )
  expect_equal(manual$policy, "manual_options")
  expect_equal(manual$m, 40L)
  expect_equal(manual$rule, "balanced_shape_metric")
})

test_that("approximate NN parameter selectors expose deterministic tuning metadata", {
  old_options <- options(
    faissR.faiss_nlist = NULL,
    faissR.ivf_nlist = NULL,
    faissR.faiss_nprobe = NULL,
    faissR.ivf_nprobe = NULL,
    faissR.faiss_pq_m = NULL,
    faissR.faiss_pq_nbits = NULL,
    faissR.faiss_nsg_r = NULL,
    faissR.faiss_nsg_search_l = NULL,
    faissR.faiss_nsg_build_type = NULL,
    faissR.faiss_nndescent_graph_k = NULL,
    faissR.faiss_nndescent_iter = NULL,
    faissR.faiss_nndescent_search_l = NULL,
    faissR.cuvs_graph_degree = NULL,
    faissR.cuvs_intermediate_graph_degree = NULL,
    faissR.cuvs_search_width = NULL,
    faissR.cuvs_itopk_size = NULL,
    faissR.cuvs_nndescent_graph_degree = NULL,
    faissR.cuvs_nndescent_intermediate_graph_degree = NULL,
    faissR.cuvs_nndescent_max_iterations = NULL
  )
  on.exit(options(old_options), add = TRUE)

  ivf <- faissR:::faiss_ivf_params(70000L, 50L)
  expect_equal(ivf$tuning_policy, "auto_shape_k")
  expect_equal(ivf$tuning_rule, "balanced_shape_k")
  expect_equal(ivf$tuning_metric, "euclidean")
  expect_false(isTRUE(ivf$tuning_metric_aware))
  expect_false(isTRUE(ivf$tuning_small_k))
  expect_false(isTRUE(ivf$tuning_large_k))
  expect_equal(anyDuplicated(names(ivf)), 0L)

  metric_ivf <- faissR:::faiss_ivf_params(70000L, 50L, metric = "correlation")
  expect_equal(metric_ivf$tuning_rule, "metric_balanced_shape_k")
  expect_equal(metric_ivf$tuning_metric, "correlation")
  expect_true(isTRUE(metric_ivf$tuning_metric_aware))
  expect_true(metric_ivf$nprobe >= ivf$nprobe)
  expect_equal(anyDuplicated(names(metric_ivf)), 0L)

  large_ivf <- faissR:::faiss_ivf_params(1000000L, 100L)
  expect_equal(large_ivf$tuning_rule, "large_n_coarse_quantizer")
  expect_true(isTRUE(large_ivf$tuning_large_n))
  expect_true(isTRUE(large_ivf$tuning_large_k))

  large_metric_ivf <- faissR:::faiss_ivf_params(1000000L, 100L, metric = "inner_product")
  expect_equal(large_metric_ivf$tuning_rule, "metric_large_n_coarse_quantizer")
  expect_equal(large_metric_ivf$tuning_metric, "inner_product")
  expect_true(isTRUE(large_metric_ivf$tuning_metric_aware))

  pq <- faissR:::faiss_pq_params(784L)
  expect_equal(pq$tuning_policy, "auto_dimension")
  expect_equal(pq$tuning_rule, "high_dim_largest_divisor_pq")
  expect_true(isTRUE(pq$tuning_high_dim))
  expect_false(isTRUE(pq$tuning_small_training))

  small_training_pq <- faissR:::faiss_pq_params(12L, n = 120L)
  expect_equal(small_training_pq$tuning_rule, "small_training_rows_minimum_pq")
  expect_true(isTRUE(small_training_pq$tuning_small_training))
  expect_equal(small_training_pq$min_training_rows, 624L)
  reduced_codebook_pq <- faissR:::faiss_pq_params(12L, n = 700L)
  expect_equal(reduced_codebook_pq$nbits, 4L)
  expect_equal(reduced_codebook_pq$tuning_rule, "training_rows_4bit_pq")
  expect_true(isTRUE(reduced_codebook_pq$tuning_reduced_codebook_training))
  expect_equal(reduced_codebook_pq$min_training_rows_8bit, 9984L)
  expect_error(
    faissR:::validate_faiss_cpu_ivfpq_training_size(120L),
    "at least 624 training rows"
  )
  expect_no_error(faissR:::validate_faiss_cpu_ivfpq_training_size(624L))

  nsg <- faissR:::faiss_nsg_params(100L)
  expect_equal(nsg$tuning_rule, "large_k_search_l")
  expect_true(isTRUE(nsg$tuning_large_k))

  nnd <- faissR:::faiss_nndescent_params(5L)
  expect_equal(nnd$tuning_rule, "small_k_speed")
  expect_true(isTRUE(nnd$tuning_small_k))

  cagra <- faissR:::cuvs_cagra_params(1000000L, 100L)
  expect_equal(cagra$tuning_policy, "auto_shape_k")
  expect_equal(cagra$tuning_rule, "large_n_large_k_graph_recall")
  expect_true(isTRUE(cagra$tuning_large_n))
  expect_true(isTRUE(cagra$tuning_large_k))

  cuvs_nnd <- faissR:::cuvs_nndescent_params(1000000L, 50L)
  expect_equal(cuvs_nnd$tuning_rule, "large_graph_search")
  expect_true(isTRUE(cuvs_nnd$tuning_large_n))
})

test_that("CPU auto selector can fall back to native NNDescent for large self-KNN", {
  expect_true(faissR:::should_use_native_nndescent_auto_fallback(
    self_query = TRUE,
    n = 70000L,
    p = 784L,
    k = 50L,
    work_size = as.double(70000L) * as.double(70000L) * as.double(784L)
  ))
  expect_false(faissR:::should_use_native_nndescent_auto_fallback(
    self_query = FALSE,
    n = 70000L,
    p = 784L,
    k = 50L,
    work_size = as.double(70000L) * as.double(70000L) * as.double(784L)
  ))
  expect_false(faissR:::should_use_native_nndescent_auto_fallback(
    self_query = TRUE,
    n = 70000L,
    p = 784L,
    k = 5L,
    work_size = as.double(70000L) * as.double(70000L) * as.double(784L)
  ))
})

test_that("CPU auto selector canonicalizes metric aliases", {
  work_size <- as.double(70000L) * as.double(70000L) * as.double(784L)
  expect_equal(
    faissR:::select_cpu_auto_backend(
      self_query = TRUE,
      n = 70000L,
      p = 784L,
      n_points = 70000L,
      k = 5L,
      work_size = work_size,
      metric = "pearson"
    ),
    faissR:::select_cpu_auto_backend(
      self_query = TRUE,
      n = 70000L,
      p = 784L,
      n_points = 70000L,
      k = 5L,
      work_size = work_size,
      metric = "correlation"
    )
  )
  expect_equal(
    faissR:::select_cpu_auto_backend(
      self_query = TRUE,
      n = 70000L,
      p = 784L,
      n_points = 70000L,
      k = 5L,
      work_size = work_size,
      metric = "dot-product"
    ),
    faissR:::select_cpu_auto_backend(
      self_query = TRUE,
      n = 70000L,
      p = 784L,
      n_points = 70000L,
      k = 5L,
      work_size = work_size,
      metric = "inner_product"
    )
  )
  expect_error(
    faissR:::select_cpu_auto_backend(
      self_query = TRUE,
      n = 70000L,
      p = 784L,
      n_points = 70000L,
      k = 5L,
      work_size = work_size,
      metric = "manhattan"
    ),
    "`metric` must be one of"
  )
})

test_that("CUDA auto selector is shape-aware", {
  skip_if_not(cuda_available() || cuvs_available())

  medium <- faissR:::select_cuda_auto_backend(
    self_query = TRUE,
    n = 50000L,
    p = 784L,
    n_points = 50000L,
    k = 50L,
    work_size = 50000 * 50000 * 784
  )
  expect_true(medium %in% c("faiss_gpu_flat_l2", "cuda_cuvs_bruteforce", "cuda"))

  low_dim <- faissR:::select_cuda_auto_backend(
    self_query = TRUE,
    n = 20000L,
    p = 3L,
    n_points = 20000L,
    k = 50L,
    work_size = 20000 * 20000 * 3
  )
  if (cuda_available()) expect_equal(low_dim, "cuda_grid")

  large <- faissR:::select_cuda_auto_backend(
    self_query = TRUE,
    n = 500000L,
    p = 512L,
    n_points = 500000L,
    k = 50L,
    work_size = 500000 * 500000 * 512
  )
  expect_true(large %in% c("faiss_gpu_cagra", "cuda_cuvs_nndescent", "cuda"))
  if (faiss_gpu_available()) expect_equal(large, "faiss_gpu_cagra")
})

test_that("CUDA auto selector has deterministic k and metric policy", {
  expect_equal(
    faissR:::cuda_auto_non_euclidean_backend(
      "cosine",
      requested_device = "cuda",
      faiss_gpu_available_value = TRUE,
      require_available = FALSE
    ),
    "faiss_gpu_flat_cosine"
  )
  expect_equal(
    faissR:::cuda_auto_non_euclidean_backend(
      "correlation",
      requested_device = "cuda",
      faiss_gpu_available_value = TRUE,
      require_available = FALSE
    ),
    "faiss_gpu_flat_correlation"
  )
  expect_equal(
    faissR:::cuda_auto_non_euclidean_backend(
      "inner_product",
      requested_device = "cuda",
      faiss_gpu_available_value = TRUE,
      require_available = FALSE
    ),
    "faiss_gpu_flat_ip"
  )
  expect_equal(
    faissR:::cuda_auto_non_euclidean_backend(
      "cosine",
      requested_device = "auto",
      faiss_gpu_available_value = FALSE
    ),
    "cpu_auto"
  )
  expect_error(
    faissR:::cuda_auto_non_euclidean_backend(
      "cosine",
      requested_device = "cuda",
      faiss_gpu_available_value = FALSE,
      require_available = TRUE
    ),
    "FAISS GPU Flat"
  )

  large_metric_work <- as.double(300000L) * as.double(300000L) * 64
  for (metric in c("cosine", "correlation", "inner_product")) {
    expect_equal(
      faissR:::cuda_auto_non_euclidean_backend(
        metric,
        requested_device = "cuda",
        self_query = TRUE,
        n = 300000L,
        p = 64L,
        n_points = 300000L,
        k = 50L,
        work_size = large_metric_work,
        cuda_available_value = TRUE,
        cuvs_available_value = TRUE,
        faiss_gpu_available_value = FALSE,
        require_available = TRUE
      ),
      "cuda_cuvs_cagra",
      info = metric
    )
    expect_equal(
      faissR:::cuda_auto_non_euclidean_backend(
        metric,
        requested_device = "cuda",
        self_query = TRUE,
        n = 300000L,
        p = 64L,
        n_points = 300000L,
        k = 50L,
        work_size = large_metric_work,
        cuda_available_value = TRUE,
        cuvs_available_value = FALSE,
        faiss_gpu_available_value = TRUE,
        require_available = TRUE
      ),
      "faiss_gpu_cagra",
      info = metric
    )
    expect_equal(
      faissR:::cuda_auto_non_euclidean_backend(
        metric,
        requested_device = "cuda",
        self_query = FALSE,
        n = 200000L,
        p = 64L,
        n_points = 1000L,
        k = 50L,
        work_size = as.double(200000L) * 1000 * 64,
        cuda_available_value = TRUE,
        cuvs_available_value = TRUE,
        faiss_gpu_available_value = TRUE,
        require_available = TRUE
      ),
      switch(
        metric,
        cosine = "faiss_gpu_flat_cosine",
        correlation = "faiss_gpu_flat_correlation",
        inner_product = "faiss_gpu_flat_ip"
      ),
      info = metric
    )
  }

  n <- 500000L
  p <- 32L
  work_size <- as.double(n) * as.double(n) * as.double(p)
  k_values <- c(5L, 10L, 15L, 50L, 100L)
  cuvs_routes <- vapply(k_values, function(k) {
    faissR:::select_cuvs_auto_backend(
      self_query = TRUE,
      n = n,
      p = p,
      n_points = n,
      k = k,
      work_size = work_size
    )
  }, character(1L))
  expect_equal(cuvs_routes[[1L]], "cuda_cuvs_bruteforce")
  expect_equal(cuvs_routes[-1L], rep("cuda_cuvs_ivf_flat", 4L))

  high_dim_route <- faissR:::select_cuvs_auto_backend(
    self_query = TRUE,
    n = n,
    p = 256L,
    n_points = n,
    k = 50L,
    work_size = as.double(n) * as.double(n) * 256
  )
  expect_equal(high_dim_route, "cuda_cuvs_nndescent")
})

test_that("RcppHNSW implementation backend is available when installed", {
  skip_if_not_installed("RcppHNSW")
  set.seed(127)
  x <- rbind(
    matrix(rnorm(400, -2, 0.4), ncol = 8),
    matrix(rnorm(400, 2, 0.4), ncol = 8)
  )
  out <- internal_nn(x, k = 10L, backend = "hnsw", n_threads = 2L)

  expect_equal(dim(out$indices), c(nrow(x), 10L))
  expect_equal(out$indices[, 1L], seq_len(nrow(x)))
  expect_equal(attr(out, "backend"), "hnsw")
  expect_false(isTRUE(attr(out, "exact")))
  expect_equal(attr(out, "approximation")$strategy, "RcppHNSW_hnswlib")
})

test_that("RcppHNSW backend supports correlation metric", {
  skip_if_not_installed("RcppHNSW")
  set.seed(128)
  x <- matrix(rnorm(80L * 10L), nrow = 80L)

  out <- internal_nn(x, k = 6L, backend = "hnsw", metric = "correlation", n_threads = 2L)

  expect_equal(dim(out$indices), c(nrow(x), 6L))
  expect_equal(attr(out, "backend"), "hnsw")
  expect_equal(attr(out, "metric"), "correlation")
  expect_equal(attr(out, "approximation")$metric, "correlation")
  expect_true(all(is.finite(out$distances)))
})

test_that("RcppHNSW backend supports inner-product metric", {
  skip_if_not_installed("RcppHNSW")
  x <- matrix(c(
    2, 0,
    0, 3,
    1, 1,
    -1, 0
  ), ncol = 2, byrow = TRUE)

  old_options <- options(
    faissR.hnsw_m = 8L,
    faissR.hnsw_ef_construction = 100L,
    faissR.hnsw_ef = 100L
  )
  on.exit(options(old_options), add = TRUE)

  out <- internal_nn(x, k = 3L, backend = "hnsw", metric = "inner_product", n_threads = 2L)
  public <- nn(x, k = 3L, backend = "cpu", method = "hnsw", metric = "inner_product", n_threads = 2L)

  expect_equal(dim(out$indices), c(nrow(x), 3L))
  expect_equal(attr(out, "backend"), "hnsw")
  expect_equal(attr(out, "metric"), "inner_product")
  expect_equal(attr(out, "approximation")$metric, "inner_product")
  expect_equal(attr(public, "backend"), if (faiss_available()) "faiss_hnsw" else "hnsw")
  expect_equal(attr(public, "metric"), "inner_product")
  expect_equal(out$indices[1, 1L], 1L)
  expect_equal(out$distances[, 1L], rep(0, nrow(x)), tolerance = 1e-12)
  expect_true(all(is.finite(out$distances)))
})

test_that("real FAISS C++ backend is either exact or clearly unavailable", {
  set.seed(137)
  x <- matrix(rnorm(120L * 6L), nrow = 120L)
  k <- 7L

  if (faiss_available()) {
    exact <- nn_without_self(x, k = k, backend = "cpu", n_threads = 2L)
    out <- internal_nn_without_self(x, k = k, backend = "faiss", n_threads = 2L)
    recall <- faissR:::.knn_recall_summary(out, exact, k = k)

    expect_equal(dim(out$indices), c(nrow(x), k))
    expect_equal(attr(out, "backend"), "faiss")
    expect_true(isTRUE(attr(out, "exact")))
    expect_equal(attr(out, "faiss")$index_type, "IndexFlatL2")
    expect_equal(recall$recall_at_k, 1)
  } else {
    expect_error(internal_nn(x, k = k + 1L, backend = "faiss"), "FAISS")
  }
})

test_that("real FAISS IVF backend records approximate index metadata", {
  set.seed(138)
  x <- matrix(rnorm(700L * 5L), nrow = 700L)
  k <- 8L

  if (faiss_available()) {
    old_options <- options(
      faissR.faiss_nlist = 16L,
      faissR.faiss_nprobe = 4L
    )
    on.exit(options(old_options), add = TRUE)

    out <- internal_nn(x, k = k + 1L, backend = "faiss_ivf", n_threads = 2L)
    expect_equal(dim(out$indices), c(nrow(x), k + 1L))
    expect_equal(out$indices[, 1L], seq_len(nrow(x)))
    expect_equal(attr(out, "backend"), "faiss_ivf")
    expect_false(isTRUE(attr(out, "exact")))
    expect_equal(attr(out, "approximation")$strategy, "faiss_IndexIVFFlat")
    expect_equal(attr(out, "approximation")$nlist, 16L)
    expect_equal(attr(out, "approximation")$nprobe, 4L)
    expect_equal(attr(out, "approximation")$requested_nlist, 16L)
    expect_equal(attr(out, "approximation")$requested_nprobe, 4L)
    expect_false(attr(out, "approximation")$ivf_parameters_adjusted)
    expect_equal(attr(out, "approximation")$tuning_policy, "manual_options")
    expect_equal(attr(out, "approximation")$tuning_rule, "small_k_speed")

    options(
      faissR.faiss_nlist = 16L,
      faissR.faiss_nprobe = 16L
    )
    for (metric in c("cosine", "correlation", "inner_product")) {
      metric_out <- internal_nn(
        x,
        k = k + 1L,
        backend = "faiss_ivf",
        metric = metric,
        n_threads = 2L
      )
      expect_equal(dim(metric_out$indices), c(nrow(x), k + 1L))
      expect_equal(attr(metric_out, "backend"), "faiss_ivf")
      expect_equal(attr(metric_out, "metric"), metric)
      expect_false(isTRUE(attr(metric_out, "exact")))
      expect_equal(attr(metric_out, "approximation")$metric, metric)
      expect_equal(attr(metric_out, "approximation")$tuning_policy, "manual_options")
      expect_equal(attr(metric_out, "approximation")$tuning_rule, "small_k_speed")
      expect_equal(attr(metric_out, "approximation")$tuning_metric, metric)
      expect_true(isTRUE(attr(metric_out, "approximation")$tuning_metric_aware))
      expect_true(all(is.finite(metric_out$distances)))
    }

    for (metric in c("cosine", "correlation", "inner_product")) {
      metric_out <- internal_nn(
        x,
        k = k + 1L,
        backend = "faiss_ivfpq",
        metric = metric,
        n_threads = 2L
      )
      expect_equal(dim(metric_out$indices), c(nrow(x), k + 1L))
      expect_equal(attr(metric_out, "backend"), "faiss_ivfpq")
      expect_equal(attr(metric_out, "metric"), metric)
      expect_false(isTRUE(attr(metric_out, "exact")))
      expect_equal(attr(metric_out, "approximation")$metric, metric)
      expect_equal(attr(metric_out, "approximation")$tuning_policy, "manual_options")
      expect_equal(attr(metric_out, "approximation")$tuning_rule, "small_k_speed")
      expect_equal(attr(metric_out, "approximation")$tuning_metric, metric)
      expect_true(isTRUE(attr(metric_out, "approximation")$tuning_metric_aware))
      expect_equal(attr(metric_out, "approximation")$pq_tuning_policy, "auto_dimension")
      expect_true(all(is.finite(metric_out$distances)))
    }

    options(
      faissR.faiss_nlist = 9999L,
      faissR.faiss_nprobe = 9999L
    )
    clamped <- internal_nn(x, k = k + 1L, backend = "faiss_ivf", n_threads = 2L)
    expect_equal(attr(clamped, "approximation")$nlist, nrow(x))
    expect_equal(attr(clamped, "approximation")$nprobe, nrow(x))
    expect_equal(attr(clamped, "approximation")$requested_nlist, 9999L)
    expect_equal(attr(clamped, "approximation")$requested_nprobe, 9999L)
    expect_true(attr(clamped, "approximation")$ivf_parameters_adjusted)
  } else {
    expect_error(internal_nn(x, k = k + 1L, backend = "faiss_ivf"), "FAISS")
  }
})

test_that("CPU IVFPQ rejects too-small training sets before FAISS warnings", {
  skip_if_not(faiss_available())
  set.seed(260215)
  x <- matrix(rnorm(120L * 8L), nrow = 120L)

  expect_error(
    nn_without_self(x, k = 5L, backend = "cpu", method = "ivfpq", n_threads = 2L),
    "at least 624 training rows"
  )
})

test_that("FAISS graph backends reject too-small training sets clearly", {
  set.seed(13815)
  x <- matrix(rnorm(80L * 8L), nrow = 80L)

  if (faiss_available()) {
    expect_error(
      internal_nn_without_self(x, k = 10L, backend = "faiss_nsg", n_threads = 2L),
      "more than 100 training rows"
    )
    expect_error(
      internal_nn_without_self(x, k = 10L, backend = "faiss_nndescent", n_threads = 2L),
      "disabled"
    )
  } else {
    expect_error(internal_nn(x, k = 10L, backend = "faiss_nsg"), "FAISS")
    expect_error(internal_nn(x, k = 10L, backend = "faiss_nndescent"), "FAISS")
  }
})

test_that("FAISS graph backends report actual and requested parameters", {
  set.seed(1382)
  x <- matrix(rnorm(200L * 8L), nrow = 200L)

  if (faiss_available()) {
    nsg <- internal_nn_without_self(x, k = 10L, backend = "faiss_nsg", n_threads = 2L)
    nsg_approx <- attr(nsg, "approximation")
    expect_equal(dim(nsg$indices), c(nrow(x), 10L))
    expect_equal(nsg_approx$r, 48L)
    expect_equal(nsg_approx$requested_r, 48L)
    expect_equal(nsg_approx$search_l, 200L)
    expect_equal(nsg_approx$requested_search_l, 200L)
    expect_equal(nsg_approx$gk, max(64L, 2L * 10L, 2L * nsg_approx$r))
    expect_false(nsg_approx$nsg_parameters_adjusted)

    nnd <- nn_without_self(x, k = 10L, backend = "cpu", method = "nndescent", n_threads = 2L)
    nnd_approx <- attr(nnd, "approximation")
    expect_equal(dim(nnd$indices), c(nrow(x), 10L))
    expect_equal(attr(nnd, "backend"), "cpu_nndescent")
    expect_equal(nnd_approx$strategy, "native_cpu_nndescent")
    expect_equal(nnd_approx$backend, "cpu")

    for (metric in c("cosine", "correlation")) {
      expect_error(
        internal_nn_without_self(
          x,
          k = 5L,
          backend = "faiss_nsg",
          metric = metric,
          n_threads = 2L
        ),
        "euclidean"
      )
      nnd_metric <- internal_nn_without_self(
        x,
        k = 5L,
        backend = "cpu_nndescent",
        metric = metric,
        n_threads = 2L
      )
      nnd_metric_approx <- attr(nnd_metric, "approximation")
      expect_equal(attr(nnd_metric, "backend"), "cpu_nndescent")
      expect_equal(attr(nnd_metric, "metric"), metric)
      expect_equal(dim(nnd_metric$indices), c(nrow(x), 5L))
      expect_match(nnd_metric_approx$transform, "normalize")
    }
    nnd_ip <- internal_nn_without_self(
      x,
      k = 5L,
      backend = "cpu_nndescent",
      metric = "inner_product",
      n_threads = 2L
    )
    nnd_ip_approx <- attr(nnd_ip, "approximation")
    expect_equal(attr(nnd_ip, "backend"), "cpu_nndescent")
    expect_equal(attr(nnd_ip, "metric"), "inner_product")
    expect_equal(dim(nnd_ip$indices), c(nrow(x), 5L))
    expect_equal(nnd_ip_approx$metric, "inner_product")
    expect_equal(unname(nnd_ip$distances[, 1L]), rep(0, nrow(x)))
    expect_true(all(is.finite(nnd_ip$distances)))
    expect_error(
      internal_nn_without_self(x, k = 5L, backend = "faiss_nsg", metric = "inner_product", n_threads = 2L),
      "euclidean"
    )
    expect_error(
      internal_nn_without_self(x, k = 5L, backend = "faiss_nndescent", metric = "inner_product", n_threads = 2L),
      "euclidean"
    )
    withr::with_options(
      list(faissR.enable_faiss_nndescent = TRUE),
      expect_error(
        internal_nn_without_self(x, k = 5L, backend = "faiss_nndescent", metric = "inner_product", n_threads = 2L),
        "euclidean"
      )
    )
    expect_error(
      internal_nn_without_self(x, k = 5L, backend = "faiss_nndescent", metric = "euclidean", n_threads = 2L),
      "disabled"
    )
  } else {
    expect_error(internal_nn(x, k = 10L, backend = "faiss_nsg"), "FAISS")
    expect_error(internal_nn(x, k = 10L, backend = "faiss_nndescent"), "FAISS")
  }
})

test_that("FAISS HNSW reports actual and requested parameters", {
  set.seed(1381)
  x <- matrix(rnorm(30L * 6L), nrow = 30L)

  if (faiss_available()) {
    old_options <- options(
      faissR.faiss_hnsw_m = 128L,
      faissR.faiss_hnsw_ef_construction = 2L,
      faissR.faiss_hnsw_ef_search = 2L
    )
    on.exit(options(old_options), add = TRUE)

    out <- internal_nn_without_self(x, k = 10L, backend = "faiss_hnsw", n_threads = 2L)
    approx <- attr(out, "approximation")
    expect_equal(dim(out$indices), c(nrow(x), 10L))
    expect_equal(approx$m, nrow(x))
    expect_equal(approx$requested_m, 128L)
    expect_equal(approx$ef_search, 10L)
    expect_equal(approx$requested_ef_search, 10L)
    expect_true(approx$hnsw_parameters_adjusted)
    expect_equal(approx$tuning_policy, "manual_options")
    expect_equal(approx$tuning_rule, "small_k_speed")
    expect_false(isTRUE(approx$tuning_high_dim))
    expect_false(isTRUE(approx$tuning_large_n))
    expect_false(isTRUE(approx$tuning_non_euclidean))
  } else {
    expect_error(internal_nn(x, k = 10L, backend = "faiss_hnsw"), "FAISS")
  }
})

test_that("FAISS GPU backends are explicit and do not fall back to CPU", {
  set.seed(139)
  x <- matrix(rnorm(96L * 6L), nrow = 96L)

  for (backend in c(
    "faiss_gpu_flat_l2",
    "faiss_gpu_flat_ip",
    "faiss_gpu_ivf_flat",
    "faiss_gpu_ivfpq"
  )) {
    out <- tryCatch(
      internal_nn(x, k = 8L, backend = backend),
      error = identity
    )
    if (inherits(out, "error")) {
      expect_match(conditionMessage(out), "FAISS.*GPU|GPU.*FAISS")
    } else {
      expect_equal(dim(out$indices), c(nrow(x), 8L))
      expect_equal(attr(out, "backend"), backend)
      if (grepl("flat", backend, fixed = TRUE) && !grepl("ivf", backend, fixed = TRUE)) {
        expect_true(isTRUE(attr(out, "exact")))
        expect_equal(attr(out, "faiss")$library, "faiss")
        expect_equal(attr(out, "faiss")$accelerator, "cuda")
      } else {
        expect_false(isTRUE(attr(out, "exact")))
        expect_equal(attr(out, "approximation")$library, "faiss")
        expect_equal(attr(out, "approximation")$accelerator, "cuda")
        if (identical(backend, "faiss_gpu_ivfpq")) {
          expect_equal(attr(out, "approximation")$role, "explicit_memory_pressure_backend")
          expect_false(isTRUE(attr(out, "approximation")$default_candidate))
        }
      }
    }
  }
})

test_that("GPU approximate KNN helpers require explicit backend requests", {
  expect_false(faissR:::should_use_gpu_approx_self_knn(
    backend = "auto",
    self_query = TRUE,
    n = 100000L,
    p = 20L,
    k = 30L,
    exclude_self = FALSE,
    work_size = 1e9
  ))
  expect_true(faissR:::should_use_gpu_approx_self_knn(
    backend = "cuda_approx",
    self_query = TRUE,
    n = 1000L,
    p = 20L,
    k = 30L,
    exclude_self = FALSE,
    work_size = 1e6
  ))
  expect_false(faissR:::should_use_gpu_approx_self_knn(
    backend = "cuda_approx",
    self_query = FALSE,
    n = 100000L,
    p = 20L,
    k = 30L,
    exclude_self = FALSE,
    work_size = 1e9
  ))
  params <- faissR:::gpu_approx_params(50000L, 30L)
  expect_gte(params$anchors, 128L)
  expect_gte(params$projection_k, 12L)
  expect_gte(params$query_cols, params$bucket_cols)
  expect_error(
    faissR:::gpu_approx_self_knn(
      matrix(rnorm(50), ncol = 5),
      k = 2L,
      backend = "cuda",
      metric = "cosine"
    ),
    "supports only"
  )
  trivial <- faissR:::gpu_approx_self_knn(
    matrix(rnorm(50), ncol = 5),
    k = 1L,
    backend = "cuda",
    exclude_self = FALSE,
    metric = "euclidean"
  )
  expect_equal(trivial$metric, "euclidean")
  expect_equal(attr(trivial, "metric"), "euclidean")
  expect_equal(trivial$backend_used, "cuda_approx")
})

test_that("approximate KNN recall metadata is attached against exact subset", {
  set.seed(131)
  x <- matrix(rnorm(80L * 5L), nrow = 80L)
  exact <- faissR:::nn_without_self(x, k = 6L, backend = "cpu")
  approx <- faissR:::finish_nn_result(exact, "test_approx", 6L, TRUE, exact = FALSE)
  approx <- faissR:::attach_knn_recall_subset(
    approx,
    data = x,
    k = 6L,
    exclude_self = TRUE,
    seed = 131L
  )
  recall <- attr(approx, "recall")
  expect_s3_class(approx, "faissR_nn")
  expect_true(is.data.frame(recall))
  expect_equal(recall$k, 6L)
  expect_equal(recall$recall_at_k, 1)
  expect_gt(recall$sample_size, 0L)
  expect_output(print(approx), "recall@6")
})

test_that("KNN recall summary ignores missing neighbour padding", {
  approx <- list(indices = matrix(
    c(1L, NA_integer_, 2L, NA_integer_),
    nrow = 2L,
    byrow = TRUE
  ))
  exact <- list(indices = matrix(
    c(1L, NA_integer_, 3L, NA_integer_),
    nrow = 2L,
    byrow = TRUE
  ))

  recall <- faissR:::.knn_recall_summary(approx, exact, k = 2L)

  expect_equal(recall$recall_at_k, 0.5)
  expect_equal(recall$median_recall_at_k, 0.5)
  expect_equal(recall$min_recall_at_k, 0)
  expect_error(
    faissR:::.knn_recall_summary(
      matrix(integer(), nrow = 2L, ncol = 0L),
      matrix(integer(), nrow = 2L, ncol = 0L)
    ),
    "at least one neighbour column"
  )
  expect_error(
    faissR:::.knn_recall_summary(approx, exact, k = 1.5),
    "`k` must be a positive integer"
  )
})

test_that("nn matches brute-force euclidean neighbors for query points", {
  set.seed(13)
  data <- matrix(rnorm(120), ncol = 6)
  points <- matrix(rnorm(60), ncol = 6)

  ours <- nn(data, points, k = 5)
  d <- matrix(0, nrow(points), nrow(data))
  for (i in seq_len(nrow(points))) {
    d[i, ] <- rowSums((data - matrix(points[i, ], nrow(data), ncol(data), byrow = TRUE))^2)
  }
  expected_idx <- t(apply(d, 1L, order))[, 1:5, drop = FALSE]
  expected_dst <- matrix(0, nrow(points), 5L)
  for (i in seq_len(nrow(points))) {
    expected_dst[i, ] <- sqrt(d[i, expected_idx[i, ]])
  }

  expect_equal(ours$indices, expected_idx)
  expect_equal(ours$distances, expected_dst, tolerance = 1e-12)
})

test_that("Fortran CPU nn path matches C++ fallback", {
  old <- Sys.getenv("FAISSR_USE_FORTRAN_NN", unset = NA_character_)
  on.exit({
    if (is.na(old)) {
      Sys.unsetenv("FAISSR_USE_FORTRAN_NN")
    } else {
      Sys.setenv(FAISSR_USE_FORTRAN_NN = old)
    }
  })

  set.seed(130)
  data <- matrix(rnorm(240), nrow = 40L)
  points <- matrix(rnorm(90), nrow = 15L)

  Sys.setenv(FAISSR_USE_FORTRAN_NN = "1")
  fortran <- nn(data, points, k = 7L, backend = "cpu")
  Sys.setenv(FAISSR_USE_FORTRAN_NN = "0")
  cpp <- nn(data, points, k = 7L, backend = "cpu")

  expect_equal(fortran$indices, cpp$indices)
  expect_equal(fortran$distances, cpp$distances, tolerance = 1e-12)

  Sys.setenv(FAISSR_USE_FORTRAN_NN = "1")
  fortran_self <- faissR:::nn_without_self(data, k = 6L, backend = "cpu")
  Sys.setenv(FAISSR_USE_FORTRAN_NN = "0")
  cpp_self <- faissR:::nn_without_self(data, k = 6L, backend = "cpu")

  expect_equal(fortran_self$indices, cpp_self$indices)
  expect_equal(fortran_self$distances, cpp_self$distances, tolerance = 1e-12)
})

test_that("removed nn compatibility options are not accepted", {
  data <- matrix(c(0, 0, 2, 0, 0, 3), ncol = 2, byrow = TRUE)
  point <- matrix(c(0, 0), nrow = 1)

  expect_error(nn(data, point, k = 3, square = TRUE), "unused")
  expect_error(nn(data, point, k = 3, method = "not_a_method"), "must be one of")
})

test_that("cuda availability helper returns a logical scalar", {
  expect_type(cuda_available(), "logical")
  expect_length(cuda_available(), 1L)
})

test_that("faiss availability helper returns a logical scalar", {
  expect_type(faiss_available(), "logical")
  expect_length(faiss_available(), 1L)
})

test_that("faiss GPU availability helper returns a logical scalar", {
  expect_type(faiss_gpu_available(), "logical")
  expect_length(faiss_gpu_available(), 1L)
})

test_that("cuvs availability helper returns a logical scalar", {
  expect_type(cuvs_available(), "logical")
  expect_length(cuvs_available(), 1L)
})

test_that("backend_info reports native availability without crashing", {
  info <- backend_info()
  expect_s3_class(info, "data.frame")
  expect_true(all(c("cpu", "faiss", "cuvs", "cuda") %in% info$backend))
  expect_true(all(c(
    "available",
    "knn_available",
    "public_backends",
    "supported_methods",
    "supported_metrics",
    "device",
    "runtime"
  ) %in% names(info)))
  expect_true(isTRUE(info$available[info$backend == "cpu"]))
  expect_false(any(is.na(info$available)))
  expect_false(any(grepl("faiss_|cuda_cuvs|cpu_", info$public_call)))
  expect_true(all(grepl("implementation", info$resolved_route)))
  expect_false(grepl("cagra", info$supported_methods[info$backend == "cpu"]))
  expect_match(info$supported_methods[info$backend == "cpu"], "vamana")
  expect_match(info$supported_methods[info$backend == "cpu"], "nsg")
  expect_match(info$supported_methods[info$backend == "faiss"], "hnsw")
  expect_match(info$supported_methods[info$backend == "faiss"], "cagra")
  expect_match(info$supported_methods[info$backend == "faiss_gpu_cuvs"], "cagra")
  expect_match(info$supported_methods[info$backend == "cuvs"], "hnsw")
  expect_match(info$supported_methods[info$backend == "cuvs"], "cagra")
  expect_match(info$supported_methods[info$backend == "cuda"], "grid")
  expect_match(info$supported_methods[info$backend == "cuda"], "hnsw")
  expect_match(info$supported_methods[info$backend == "cuda"], "vamana")
  expect_match(info$supported_methods[info$backend == "cuda"], "nsg")
  expect_match(info$supported_methods[info$backend == "cuda"], "ivfpq")
  expect_match(info$supported_methods[info$backend == "cuda"], "cagra")
  expect_match(info$supported_metrics[info$backend == "cpu"], "euclidean")
  expect_match(info$supported_metrics[info$backend == "cpu"], "cosine")
  expect_match(info$supported_metrics[info$backend == "cpu"], "correlation")
  expect_match(info$supported_metrics[info$backend == "cpu"], "inner_product")
  expect_match(info$supported_metrics[info$backend == "faiss_gpu_cuvs"], "CAGRA inner_product uses a MIPS-to-L2 transform")
  expect_match(info$supported_metrics[info$backend == "cuvs"], "public CUDA NN-descent inner_product uses native CUDA")

  cuda_info <- faissR:::cuda_device_info_json_cpp()
  expect_type(cuda_info, "character")
  expect_length(cuda_info, 1L)
  expect_match(cuda_info, "available")
})

test_that("nvidia-smi summary parses device, driver, and memory", {
  bin <- file.path(tempdir(), paste0("fake-nvidia-smi-", Sys.getpid()))
  dir.create(bin, showWarnings = FALSE, recursive = TRUE)
  smi <- file.path(bin, "nvidia-smi")
  writeLines(
    c(
      "#!/bin/sh",
      "printf 'NVIDIA L40S, 555.42, 46068\\n'"
    ),
    smi
  )
  Sys.chmod(smi, "0755")

  old_path <- Sys.getenv("PATH")
  on.exit(Sys.setenv(PATH = old_path), add = TRUE)
  Sys.setenv(PATH = paste(bin, old_path, sep = .Platform$path.sep))

  summary <- faissR:::nvidia_smi_summary()
  expect_equal(summary$device, "NVIDIA L40S")
  expect_match(summary$runtime, "driver 555.42")
  expect_match(summary$runtime, "46068 MiB")
})

test_that("CUDA grid auto does not silently fall back to CPU", {
  skip_if(cuda_available())

  x <- matrix(runif(40L), ncol = 2L)
  expect_error(
    internal_nn(x, k = 4L, backend = "cuda_grid_auto"),
    "No CUDA GPU backend is available"
  )
  expect_error(
    internal_nn(x, k = 4L, backend = "cuda_nsg"),
    "No CUDA GPU backend is available"
  )
  expect_error(
    internal_nn(x, k = 4L, backend = "cuda_vamana"),
    "No CUDA GPU backend is available"
  )
})

test_that("RcppHNSW implementation backend is available when the suggested package is installed", {
  skip_if_not_installed("RcppHNSW")

  set.seed(144)
  x <- matrix(rnorm(120L * 6L), nrow = 120L)
  out <- internal_nn(x, k = 8L, backend = "hnsw", n_threads = 2L)

  expect_equal(dim(out$indices), c(nrow(x), 8L))
  expect_equal(attr(out, "backend"), "hnsw")
  expect_false(isTRUE(attr(out, "exact")))
  expect_equal(attr(out, "approximation")$library, "RcppHNSW")
})

test_that("CUDA nn backend matches CPU euclidean results", {
  skip_if_not(cuda_available())

  set.seed(15)
  data <- matrix(rnorm(500), ncol = 10)
  points <- matrix(rnorm(230), ncol = 10)

  cpu <- nn(data, points, k = 6, backend = "cpu")
  gpu <- nn(data, points, k = 6, backend = "cuda")

  expect_equal(attr(gpu, "backend"), "cuda")
  expect_equal(gpu$indices, cpu$indices)
  expect_equal(gpu$distances, cpu$distances, tolerance = 1e-5)
})

test_that("CUDA grid auto matches exact CPU grid results", {
  skip_if_not(cuda_available())

  set.seed(151)
  x <- matrix(runif(3000L), ncol = 3L)
  cpu <- internal_nn(x, k = 8L, backend = "cpu_grid", n_threads = 2L)
  gpu <- internal_nn(x, k = 8L, backend = "cuda_grid_auto")

  expect_equal(attr(gpu, "backend"), "cuda_grid3d")
  expect_true(isTRUE(attr(gpu, "exact")))
  expect_equal(gpu$indices, cpu$indices)
  expect_equal(gpu$distances, cpu$distances, tolerance = 1e-5)
})

test_that("CUDA NSG returns metric-aware self-KNN results", {
  skip_if_not(cuda_available())

  set.seed(1449)
  x <- matrix(rnorm(120L * 8L), nrow = 120L)
  for (metric in c("euclidean", "cosine", "correlation", "inner_product")) {
    out <- internal_nn_without_self(
      x,
      k = 8L,
      backend = "cuda_nsg",
      metric = metric,
      n_threads = 2L
    )
    approx <- attr(out, "approximation")
    expect_equal(dim(out$indices), c(nrow(x), 8L), info = metric)
    expect_equal(attr(out, "backend"), "cuda_nsg", info = metric)
    expect_equal(attr(out, "metric"), metric, info = metric)
    expect_equal(approx$backend, "cuda_nsg", info = metric)
    expect_equal(approx$strategy, "native_cuda_nsg_candidate_graph", info = metric)
    expect_true(all(is.finite(out$distances)), info = metric)
  }
})

test_that("CUDA NN-descent requests use RAPIDS cuVS", {
  skip_if_not(cuvs_available())

  set.seed(136)
  x <- rbind(
    matrix(rnorm(300, -1.4, 0.35), ncol = 5),
    matrix(rnorm(300, 1.4, 0.35), ncol = 5)
  )
  k <- 8L
  exact <- nn_without_self(x, k = k, backend = "cpu", n_threads = 4L)
  refined <- internal_nn_without_self(x, k = k, backend = "cuda_cuvs_nndescent")
  recall <- faissR:::.knn_recall_summary(refined, exact, k)

  expect_equal(dim(refined$indices), c(nrow(x), k))
  expect_equal(attr(refined, "backend"), "cuda_cuvs_nndescent")
  expect_equal(
    attr(refined, "approximation")$strategy,
    "rapids_cuvs_nndescent"
  )
  expect_false(isTRUE(attr(refined, "exact")))
  expect_gt(recall$recall_at_k, 0.65)
})

test_that("cuVS brute force reports Euclidean metric metadata", {
  skip_if_not(cuvs_available())

  x <- matrix(rnorm(80L), nrow = 20L)
  out <- internal_nn(x, k = 4L, backend = "cuda_cuvs_bruteforce", metric = "euclidean")

  expect_equal(attr(out, "metric"), "euclidean")
  expect_equal(attr(out, "cuvs")$metric, "euclidean")
  expect_equal(attr(out, "cuvs")$resolved_backend, "cuda_cuvs_bruteforce")
})

test_that("direct cuVS IVF routes report metric metadata", {
  skip_if_not(cuvs_available())

  x <- matrix(rnorm(160L), nrow = 40L)
  for (backend in c("cuda_cuvs_ivf_flat", "cuda_cuvs_ivfpq")) {
    out <- internal_nn(x, k = 4L, backend = backend, metric = "euclidean")

    expect_equal(attr(out, "metric"), "euclidean", info = backend)
    expect_equal(attr(out, "approximation")$metric, "euclidean", info = backend)
    expect_equal(attr(out, "approximation")$library, "cuvs", info = backend)

    cos_out <- internal_nn(x, k = 4L, backend = backend, metric = "cosine")
    expect_equal(attr(cos_out, "metric"), "cosine", info = backend)
    expect_equal(attr(cos_out, "approximation")$metric, "cosine", info = backend)
    expect_match(
      attr(cos_out, "approximation")$transform,
      "row_l2_normalize",
      fixed = TRUE,
      info = backend
    )
  }
})

test_that("cuVS CAGRA reports actual and requested graph parameters", {
  skip_if_not(cuvs_available())

  set.seed(839)
  x <- matrix(rnorm(40L * 8L), nrow = 40L)
  old_options <- options(
    faissR.cuvs_graph_degree = 128L,
    faissR.cuvs_intermediate_graph_degree = 512L,
    faissR.cuvs_search_width = 0L,
    faissR.cuvs_itopk_size = 8L
  )
  on.exit(options(old_options), add = TRUE)

  out <- internal_nn_without_self(x, k = 10L, backend = "cuda_cuvs_cagra")
  approx <- attr(out, "approximation")
  expect_equal(dim(out$indices), c(nrow(x), 10L))
  expect_equal(approx$graph_degree, nrow(x) - 1L)
  expect_equal(approx$requested_graph_degree, 128L)
  expect_equal(approx$intermediate_graph_degree, nrow(x) - 1L)
  expect_equal(approx$requested_intermediate_graph_degree, 512L)
  expect_equal(approx$itopk_size, 10L)
  expect_equal(approx$requested_itopk_size, 8L)
  expect_true(approx$cagra_parameters_adjusted)
  expect_equal(approx$library, "cuvs")
  expect_equal(approx$accelerator, "cuda")
  expect_equal(approx$cagra_provider, "cuvs")
  expect_equal(approx$cagra_provider_option, "auto")
})

test_that("CUDA backend reports unavailable runtime clearly", {
  skip_if(cuda_available())

  x <- matrix(rnorm(30), ncol = 3)
  expect_error(nn(x, x, k = 2, backend = "cuda"), "No CUDA GPU backend")
  expect_error(nn(x, x, k = 2, backend = "gpu"), "must be one of")
  expect_error(nn(x, x, k = 2, backend = "cuda_ivf"), "must be one of")
  expect_error(nn(x, x, k = 2, backend = "cuda_faiss"), "must be one of")
})

test_that("CAGRA tuning cache can round-trip without CUDA", {
  cache_file <- tempfile(fileext = ".rds")
  old_file <- getOption("faissR.cuvs_cagra_tune_cache_file")
  on.exit(options(faissR.cuvs_cagra_tune_cache_file = old_file), add = TRUE)
  options(faissR.cuvs_cagra_tune_cache_file = cache_file)

  key <- "unit-test-cagra-cache"
  value <- list(
    params = list(
      graph_degree = 64L,
      intermediate_graph_degree = 128L,
      search_width = 0L,
      itopk_size = 64L
    ),
    tuning = list(status = "target_met")
  )
  faissR:::cuvs_cagra_set_cached_tuning(key, value)
  rm(list = key, envir = faissR:::.cuvs_cagra_tune_cache)

  cached <- faissR:::cuvs_cagra_get_cached_tuning(key)
  expect_equal(cached$params$graph_degree, 64L)
  expect_equal(cached$tuning$cache, "disk")
})

test_that("FAISS GPU IVF tuning cache can round-trip without CUDA", {
  cache_file <- tempfile(fileext = ".rds")
  old_file <- getOption("faissR.faiss_gpu_ivf_tune_cache_file")
  on.exit(options(faissR.faiss_gpu_ivf_tune_cache_file = old_file), add = TRUE)
  options(faissR.faiss_gpu_ivf_tune_cache_file = cache_file)

  key <- "unit-test-faiss-gpu-ivf-cache"
  value <- list(
    params = list(nlist = 128L, nprobe = 16L),
    tuning = list(status = "target_met")
  )
  faissR:::faiss_gpu_ivf_set_cached_tuning(key, value)
  rm(list = key, envir = faissR:::.faiss_gpu_ivf_tune_cache)

  cached <- faissR:::faiss_gpu_ivf_get_cached_tuning(key)
  expect_equal(cached$params$nlist, 128L)
  expect_equal(cached$params$nprobe, 16L)
  expect_equal(cached$tuning$cache, "disk")
})

test_that("FAISS GPU IVF tuner calls the current metric-aware C++ signature", {
  source_file <- test_path("../../R/nn.R")
  if (!file.exists(source_file)) {
    skip("Package source file is not available in this installed-package test context.")
  }
  source <- paste(readLines(source_file, warn = FALSE), collapse = "\n")

  expect_true(grepl(
    paste(
      "compare_k,",
      "          as.integer\\(cand\\$nlist\\),",
      "          as.integer\\(cand\\$nprobe\\),",
      "          \"euclidean\",",
      "          \"euclidean\",",
      "          FALSE",
      sep = "\n"
    ),
    source
  ))
  expect_false(grepl(
    "compare_k,\\n          as.integer\\(cand\\$nlist\\),\\n          as.integer\\(cand\\$nprobe\\),\\n          FALSE",
    source
  ))
})

test_that("cuVS backend reports unavailable runtime clearly", {
  skip_if(cuvs_available())

  x <- matrix(rnorm(30), ncol = 3)
  expect_error(internal_nn(x, x, k = 2, backend = "cuda_cuvs"), "cuVS")
  expect_error(internal_nn(x, x, k = 2, backend = "cuda_cagra"), "cuVS")
  expect_error(internal_nn(x, x, k = 2, backend = "gpu_cagra"), "cuVS")
  expect_error(internal_nn(x, x, k = 2, backend = "cuda_cuvs_ivf_flat"), "cuVS")
  expect_error(internal_nn(x, x, k = 2, backend = "cuda_cuvs_ivfpq"), "cuVS")
  expect_error(internal_nn(x, x, k = 2, backend = "cuda_cuvs_bruteforce"), "cuVS")
  expect_error(internal_nn(x, x, k = 2, backend = "cuda_cuvs_nndescent"), "cuVS")
  expect_error(internal_nn(x, x, k = 2, backend = "cuda_cuvs_hnsw"), "cuVS")
  expect_error(internal_nn(x, x, k = 2, backend = "cuda_approx"), "cuVS")
  expect_error(internal_nn(x, x, k = 2, backend = "cuda_nndescent"), "cuVS")
})
