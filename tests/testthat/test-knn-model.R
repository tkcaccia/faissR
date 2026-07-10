test_that("knn fits models and predicts classes and probabilities", {
  x <- matrix(c(
    0, 0,
    0, 1,
    5, 5,
    5, 6
  ), ncol = 2, byrow = TRUE)
  y <- factor(c("a", "a", "b", "b"))
  fit <- knn(x, y, backend = "cpu", k = 2L)

  pred <- predict(fit, matrix(c(0.1, 0.2, 5.2, 5.4), ncol = 2, byrow = TRUE))
  expect_s3_class(fit, "faissR_knn_model")
  expect_equal(anyDuplicated(names(fit)), 0L)
  expect_equal(as.character(pred), c("a", "b"))
  expect_equal(levels(pred), levels(y))

  proba <- predict(fit, matrix(c(0.1, 0.2, 5.2, 5.4), ncol = 2, byrow = TRUE), type = "prob")
  expect_equal(dim(proba), c(2L, 2L))
  expect_equal(colnames(proba), c("a", "b"))
  expect_equal(rowSums(proba), c(1, 1), tolerance = 1e-12)
  expect_gt(proba[1, "a"], proba[1, "b"])
  expect_gt(proba[2, "b"], proba[2, "a"])
})

test_that("predict preserves nearest-neighbour route metadata", {
  x <- matrix(c(
    0, 0,
    0, 1,
    5, 5,
    5, 6
  ), ncol = 2, byrow = TRUE)
  y <- factor(c("a", "a", "b", "b"))
  fit <- knn(x, y, backend = "auto", method = "exact", metric = "cosine", tuning = "off", k = 2L)

  pred <- predict(fit, x[1:2, , drop = FALSE], backend = "auto", tuning = "off")
  proba <- predict(fit, x[1:2, , drop = FALSE], backend = "auto", tuning = "off", type = "prob")

  pred_meta <- attr(pred, "faissR_nn")
  proba_meta <- attr(proba, "faissR_nn")
  expect_equal(pred_meta$requested_backend, "auto")
  expect_equal(pred_meta$requested_method, "exact")
  expect_equal(pred_meta$tuning, "off")
  expect_equal(pred_meta$metric, "cosine")
  expect_equal(pred_meta$k, 2L)
  expect_true(pred_meta$batch_query)
  expect_equal(pred_meta$query_n, 2L)
  expect_equal(pred_meta$query_call_count, 1L)
  expect_equal(pred_meta$query_source, "nn")
  expect_equal(proba_meta$requested_backend, "auto")
  expect_equal(proba_meta$requested_method, "exact")
  expect_equal(proba_meta$tuning, "off")
  expect_equal(proba_meta$metric, "cosine")
  expect_true(proba_meta$batch_query)
  expect_equal(proba_meta$query_n, 2L)
  expect_equal(proba_meta$query_call_count, 1L)
  expect_equal(proba_meta$query_source, "nn")
  expect_equal(proba_meta$resolved_backend, proba_meta$backend)
  expect_true(is.null(pred_meta$auto_selection) || is.list(pred_meta$auto_selection))
  expect_equal(pred_meta$distance_type, "double")

  reg <- knn(x, c(0, 0, 1, 1), backend = "auto", method = "exact", task = "regression", tuning = "off", k = 2L)
  reg_pred <- predict(reg, x[1:2, , drop = FALSE], backend = "auto", tuning = "off")
  reg_meta <- attr(reg_pred, "faissR_nn")
  expect_equal(reg_meta$requested_backend, "auto")
  expect_equal(reg_meta$requested_method, "exact")
  expect_equal(reg_meta$tuning, "off")
  expect_equal(reg_meta$metric, "euclidean")
  expect_true(reg_meta$batch_query)
  expect_equal(reg_meta$query_n, 2L)
  expect_equal(reg_meta$query_call_count, 1L)
  expect_equal(reg_meta$query_source, if (is.null(reg$nn_index)) "nn" else "fitted_index")
})

test_that("knn preserves float32 input for direct NN backends", {
  skip_if_not_installed("float")
  skip_if_not(faiss_available())

  set.seed(19)
  x_num <- matrix(runif(120), nrow = 20L, ncol = 6L)
  x <- float::fl(x_num)
  y <- factor(rep(c("a", "b"), length.out = nrow(x_num)))
  query <- float::fl(x_num[1:5, , drop = FALSE])

  fit <- knn(
    x,
    y,
    backend = "cpu",
    method = "hnsw",
    metric = "euclidean",
    target_recall = 0.95,
    k = 3L,
    n_threads = 2L
  )
  expect_s3_class(fit, "faissR_knn_model")
  expect_true(inherits(fit$Xtrain, "float32"))

  pred <- predict(fit, query, backend = "cpu")
  meta <- attr(pred, "faissR_nn")
  expect_s3_class(pred, "factor")
  expect_equal(length(pred), 5L)
  expect_equal(meta$resolved_backend, "faiss_hnsw")
  expect_equal(meta$requested_method, "hnsw")
  expect_equal(meta$input_type, "float32")
  expect_false(meta$float32_compatibility_conversion)
  expect_match(meta$approximation$tuning_policy, "auto_shape_metric")
})

test_that("knn reuses fitted FAISS HNSW index for matching predictions", {
  skip_if_not_installed("float")
  skip_if_not(faiss_available())

  set.seed(23)
  x_num <- matrix(runif(180), nrow = 30L, ncol = 6L)
  x <- float::fl(x_num)
  y <- factor(rep(c("a", "b", "c"), length.out = nrow(x_num)))

  fit <- knn(
    x,
    y,
    backend = "cpu",
    method = "hnsw",
    metric = "euclidean",
    target_recall = 0.95,
    k = 3L,
    n_threads = 2L
  )
  expect_equal(fit$nn_index_backend, "faiss_hnsw")
  expect_s3_class(fit$nn_index, "faissR_faiss_hnsw_index")

  pred <- predict(fit, x[1:5, , drop = FALSE], backend = "cpu", k = 3L)
  meta <- attr(pred, "faissR_nn")
  expect_equal(meta$resolved_backend, "faiss_hnsw")
  expect_equal(meta$requested_method, "hnsw")
  expect_true(meta$approximation$fitted_index)
  expect_true(meta$approximation$index_reused)
  expect_true(meta$faiss$index_reused)
  expect_true(meta$batch_query)
  expect_equal(meta$query_n, 5L)
  expect_equal(meta$query_call_count, 1L)
  expect_equal(meta$query_source, "fitted_index")
  expect_true(meta$approximation$batch_query)
  expect_equal(meta$approximation$query_n, 5L)
  expect_equal(meta$approximation$query_call_count, 1L)
  expect_true(meta$faiss$batch_query)
  expect_equal(meta$faiss$query_n, 5L)
  expect_equal(meta$input_type, "float32")
  expect_false(meta$float32_compatibility_conversion)

  pred_k5 <- predict(fit, x[1:5, , drop = FALSE], backend = "cpu", k = 5L)
  meta_k5 <- attr(pred_k5, "faissR_nn")
  expect_equal(meta_k5$resolved_backend, "faiss_hnsw")
  expect_true(meta_k5$approximation$index_reused)
  expect_equal(meta_k5$k, 5L)

  pred_double <- predict(fit, x_num[1:5, , drop = FALSE], backend = "cpu", k = 3L)
  meta_double <- attr(pred_double, "faissR_nn")
  expect_equal(meta_double$resolved_backend, "faiss_hnsw")
  expect_equal(meta_double$query_source, "fitted_index")
  expect_true(meta_double$faiss$index_reused)
  expect_true(meta_double$float32_compatibility_conversion)
})

test_that("knn reuses fitted FAISS Flat index for matching predictions", {
  skip_if_not_installed("float")
  skip_if_not(faiss_available())

  set.seed(25)
  x_num <- matrix(runif(160), nrow = 40L, ncol = 4L)
  x <- float::fl(x_num)
  y <- factor(rep(c("a", "b"), length.out = nrow(x_num)))

  fit <- knn(
    x,
    y,
    backend = "cpu",
    method = "flat",
    metric = "euclidean",
    k = 3L,
    n_threads = 2L
  )
  expect_equal(fit$nn_index_backend, "faiss_flat_l2")
  expect_s3_class(fit$nn_index, "faissR_faiss_flat_index")

  pred <- predict(fit, x[1:7, , drop = FALSE], backend = "cpu", k = 3L)
  meta <- attr(pred, "faissR_nn")
  expect_equal(length(pred), 7L)
  expect_equal(meta$resolved_backend, "faiss_flat_l2")
  expect_equal(meta$requested_method, "flat")
  expect_true(meta$exact)
  expect_true(meta$approximation$fitted_index)
  expect_true(meta$approximation$index_reused)
  expect_true(meta$approximation$exact)
  expect_equal(meta$approximation$index_type, "IndexFlatL2ExternalPtr")
  expect_true(meta$faiss$index_reused)
  expect_true(meta$faiss$exact)
  expect_equal(meta$faiss$index_type, "IndexFlatL2ExternalPtr")
  expect_true(meta$batch_query)
  expect_equal(meta$query_n, 7L)
  expect_equal(meta$query_call_count, 1L)
  expect_equal(meta$query_source, "fitted_index")
  expect_true(meta$approximation$batch_query)
  expect_equal(meta$approximation$query_n, 7L)
  expect_equal(meta$approximation$query_call_count, 1L)
  expect_true(meta$faiss$batch_query)
  expect_equal(meta$faiss$query_n, 7L)
  expect_equal(meta$input_type, "float32")
})

test_that("knn reuses fitted FAISS IVF index for repeated predictions", {
  skip_if_not(faiss_available())

  set.seed(24)
  x <- matrix(runif(320), nrow = 80L, ncol = 4L)
  y <- factor(rep(c("a", "b"), length.out = nrow(x)))

  fit <- knn(
    x,
    y,
    backend = "cpu",
    method = "ivf",
    metric = "euclidean",
    k = 3L,
    n_threads = 2L
  )
  expect_equal(fit$nn_index_backend, "faiss_ivf")
  expect_s3_class(fit$nn_index, "faissR_faiss_ivf_index")
  expect_true(isTRUE(attr(fit$nn_index, "index_trained", exact = TRUE)))
  expect_true(isTRUE(attr(fit$nn_index, "centroids_trained", exact = TRUE)))
  expect_true(isTRUE(attr(fit$nn_index, "inverted_lists_built", exact = TRUE)))

  pred <- predict(fit, x[1:6, , drop = FALSE], backend = "cpu", k = 4L)
  meta <- attr(pred, "faissR_nn")
  expect_equal(length(pred), 6L)
  expect_equal(meta$resolved_backend, "faiss_ivf")
  expect_equal(meta$requested_method, "ivf")
  expect_true(meta$approximation$fitted_index)
  expect_true(meta$approximation$index_reused)
  expect_true(meta$approximation$index_trained)
  expect_true(meta$approximation$index_training_reused)
  expect_true(meta$approximation$centroids_reused)
  expect_true(meta$approximation$inverted_lists_reused)
  expect_true(meta$approximation$vectors_reused)
  expect_equal(meta$approximation$search_train_call_count, 0L)
  expect_equal(meta$approximation$build_train_call_count, 1L)
  expect_equal(meta$approximation$search_nprobe, meta$approximation$nprobe)
  expect_equal(meta$approximation$tuning_query_k, 4L)
  expect_true(meta$faiss$index_reused)
  expect_true(meta$faiss$index_training_reused)
  expect_true(meta$faiss$centroids_reused)
  expect_true(meta$faiss$inverted_lists_reused)
  expect_equal(meta$faiss$search_train_call_count, 0L)
  expect_true(meta$batch_query)
  expect_equal(meta$query_n, 6L)
  expect_equal(meta$query_call_count, 1L)
  expect_equal(meta$query_source, "fitted_index")
  expect_true(meta$approximation$batch_query)
  expect_equal(meta$approximation$query_n, 6L)
  expect_equal(meta$approximation$query_call_count, 1L)
  expect_true(meta$faiss$batch_query)
  expect_equal(meta$faiss$query_n, 6L)
  expect_equal(meta$k, 4L)
  expect_equal(meta$input_type, "float32")
})

test_that("knn reuses fitted FAISS IVFPQ codebooks for repeated predictions", {
  skip_if_not(faiss_available())

  set.seed(26)
  x <- matrix(runif(700L * 8L), nrow = 700L, ncol = 8L)
  y <- factor(rep(c("a", "b", "c"), length.out = nrow(x)))

  fit <- knn(
    x,
    y,
    backend = "cpu",
    method = "ivfpq",
    metric = "euclidean",
    k = 5L,
    n_threads = 2L
  )
  expect_equal(fit$nn_index_backend, "faiss_ivfpq")
  expect_s3_class(fit$nn_index, "faissR_faiss_ivfpq_index")
  expect_true(isTRUE(attr(fit$nn_index, "index_trained", exact = TRUE)))
  expect_true(isTRUE(attr(fit$nn_index, "centroids_trained", exact = TRUE)))
  expect_true(isTRUE(attr(fit$nn_index, "inverted_lists_built", exact = TRUE)))
  expect_true(isTRUE(attr(fit$nn_index, "pq_codebooks_trained", exact = TRUE)))
  expect_true(isTRUE(attr(fit$nn_index, "pq_codes_built", exact = TRUE)))
  expect_equal(attr(fit$nn_index, "build_pq_train_call_count", exact = TRUE), 1L)

  pred <- predict(fit, x[1:8, , drop = FALSE], backend = "cpu", k = 6L)
  meta <- attr(pred, "faissR_nn")
  expect_equal(length(pred), 8L)
  expect_equal(meta$resolved_backend, "faiss_ivfpq")
  expect_equal(meta$requested_method, "ivfpq")
  expect_true(meta$approximation$fitted_index)
  expect_true(meta$approximation$index_reused)
  expect_true(meta$approximation$centroids_reused)
  expect_true(meta$approximation$inverted_lists_reused)
  expect_true(meta$approximation$pq_codebooks_reused)
  expect_true(meta$approximation$pq_codes_reused)
  expect_true(meta$approximation$pq_training_reused)
  expect_equal(meta$approximation$search_train_call_count, 0L)
  expect_equal(meta$approximation$search_pq_train_call_count, 0L)
  expect_equal(meta$approximation$build_pq_train_call_count, 1L)
  expect_equal(meta$approximation$search_nprobe, meta$approximation$nprobe)
  expect_equal(meta$approximation$tuning_query_k, 6L)
  expect_true(meta$faiss$index_reused)
  expect_true(meta$faiss$pq_codebooks_reused)
  expect_true(meta$faiss$pq_codes_reused)
  expect_true(meta$faiss$pq_training_reused)
  expect_equal(meta$faiss$search_pq_train_call_count, 0L)
  expect_equal(meta$query_source, "fitted_index")
  expect_true(meta$batch_query)
  expect_equal(meta$query_n, 8L)
  expect_equal(meta$query_call_count, 1L)
  expect_equal(meta$input_type, "float32")
})

test_that("predict carries NN auto-selection route metadata", {
  set.seed(22)
  x <- rbind(
    matrix(rnorm(40, -1, 0.2), ncol = 4),
    matrix(rnorm(40, 1, 0.2), ncol = 4)
  )
  y <- factor(rep(c("a", "b"), each = 10L))
  fit <- knn(x, y, backend = "cpu", method = "auto", metric = "cosine", k = 3L)

  pred <- predict(fit, x[1:2, , drop = FALSE], backend = "cpu", type = "prob")
  meta <- attr(pred, "faissR_nn")

  expect_equal(meta$requested_backend, "cpu")
  expect_equal(meta$requested_method, "auto")
  expect_equal(meta$metric, "cosine")
  expect_type(meta$auto_selection, "list")
  expect_true(meta$auto_selection$explicit_backend)
  expect_false(meta$auto_selection$explicit_method)
  expect_equal(meta$auto_selection$backend_decision, "explicit_cpu")
})

test_that("weighted votes use exact matches cleanly", {
  x <- matrix(c(
    0, 0,
    0, 2,
    4, 4
  ), ncol = 2, byrow = TRUE)
  fit <- knn(x, c("zero", "other", "other"), backend = "cpu", k = 3L)

  proba <- predict(fit, x[1, , drop = FALSE], k = 3L, vote = "weighted", type = "prob")
  expect_equal(unname(proba[1, "zero"]), 1)
  expect_equal(unname(proba[1, "other"]), 0)
})

test_that("knn supports regression", {
  x <- matrix(c(
    0, 0,
    1, 0,
    10, 0
  ), ncol = 2, byrow = TRUE)
  fit <- knn(x, c(0, 2, 10), backend = "cpu", task = "regression", k = 2L)

  pred <- predict(fit, matrix(c(0.2, 0), ncol = 2), k = 2L)
  pred_w <- predict(fit, matrix(c(0.2, 0), ncol = 2), k = 2L, vote = "weighted")
  expect_type(pred, "double")
  expect_equal(as.numeric(pred), 1, tolerance = 1e-12)
  expect_lt(pred_w, pred)
  expect_error(predict(fit, x, type = "prob"), "classification")
})

test_that("knn stores canonical metric labels and rejects legacy metric aliases", {
  x <- matrix(c(
    1, 0,
    0, 1,
    1, 1,
    2, 0
  ), ncol = 2, byrow = TRUE)
  y <- factor(c("a", "b", "a", "b"))

  fit <- knn(x, y, backend = "cpu", metric = "inner_product", k = 2L)
  pred <- predict(fit, x[1:2, , drop = FALSE])

  expect_equal(fit$metric, "inner_product")
  expect_s3_class(pred, "factor")
  expect_error(knn(x, y, backend = "cpu", metric = "ip", k = 2L), "metric")
})

test_that("knn validates method and tuning before returning a model", {
  x <- matrix(rnorm(40), ncol = 4)
  y <- rep(c("a", "b"), length.out = nrow(x))

  expect_error(knn(x, y, backend = "cpu", method = "faiss_hnsw", k = 3L), "method")
  expect_error(knn(x, y, backend = "cpu", tuning = "aggressive", k = 3L), "tuning")
  expect_error(knn(x, y, backend = "cpu", method = "grid", k = 3L), "grid")

  model <- knn(x, y, backend = "cpu", tuning = "none", k = 3L)
  expect_equal(model$tuning, "off")
  expect_error(
    predict(model, x[1:2, , drop = FALSE], tuning = "aggressive"),
    "tuning"
  )
})

test_that("knn and predict require canonical task vote and type labels", {
  x <- matrix(rnorm(40), ncol = 4)
  y <- rep(c("a", "b"), length.out = nrow(x))

  expect_equal(faissR:::normalize_knn_task(NULL), "auto")
  expect_equal(faissR:::normalize_knn_vote(NULL), "majority")
  expect_equal(faissR:::normalize_knn_type(NULL), "response")
  expect_no_error(knn(x, y, backend = "cpu", k = 3L))
  expect_error(faissR:::normalize_knn_task(c("classification", "regression")), "`task` must be a single value")
  expect_error(faissR:::normalize_knn_vote(c("weighted", "majority")), "`vote` must be a single value")
  expect_error(faissR:::normalize_knn_type(c("prob", "response")), "`type` must be a single value")
  expect_error(knn(x, y, backend = "cpu", task = "r", k = 3L), "task")
  expect_error(knn(x, y, x[1:2, , drop = FALSE], backend = "cpu", vote = "w", k = 3L), "vote")
  expect_error(knn(x, y, x[1:2, , drop = FALSE], backend = "cpu", type = "p", k = 3L), "type")
  expect_error(knn(x, y, backend = "cpu", k = c(2L, 3L)), "`k` must be a positive integer")
  expect_error(knn(x, y, backend = "cpu", k = 2.5), "`k` must be a positive integer")
  expect_error(knn(x, y, backend = "cpu", k = 3L, n_threads = c(1L, 2L)), "`n_threads` must be NULL or a single positive integer")

  model <- knn(x, y, backend = "cpu", k = 3L)
  expect_error(predict(model, x[1:2, , drop = FALSE], k = c(2L, 3L)), "`k` must be a positive integer")
  expect_error(predict(model, x[1:2, , drop = FALSE], k = 2.5), "`k` must be a positive integer")
  expect_error(predict(model, x[1:2, , drop = FALSE], vote = "w"), "vote")
  expect_error(predict(model, x[1:2, , drop = FALSE], type = "p"), "type")
})

test_that("knn rejects implementation backend labels", {
  x <- matrix(rnorm(30), ncol = 3)
  y <- rep(c("a", "b"), length.out = nrow(x))

  expect_error(knn(x, y, backend = "faiss", k = 3L), "must be one of")
  expect_error(knn(x, y, backend = "cuda_cuvs", k = 3L), "must be one of")

  model <- knn(x, y, backend = "cpu", k = 3L)
  expect_error(
    predict(model, x[1:2, , drop = FALSE], backend = "faiss"),
    "must be one of"
  )
})

test_that("predict uses public backend choices and reuses fitted method", {
  x <- rbind(
    matrix(c(0, 0, 0.2, 0.1, 0.3, 0.2), ncol = 2, byrow = TRUE),
    matrix(c(5, 5, 5.1, 5.2, 5.3, 5.1), ncol = 2, byrow = TRUE)
  )
  y <- factor(rep(c("a", "b"), each = 3L))

  model <- knn(x, y, backend = "cpu", method = "exact", k = 3L)
  expect_equal(model$backend, "cpu")
  expect_equal(model$method, "exact")

  implicit <- predict(model, x[1:2, , drop = FALSE])
  explicit <- predict(model, x[1:2, , drop = FALSE], backend = "cpu")

  expect_equal(as.character(implicit), as.character(explicit))
  expect_equal(attr(implicit, "faissR_nn")$requested_backend, "auto")
  expect_equal(attr(explicit, "faissR_nn")$requested_backend, "cpu")
  expect_error(predict(model, x[1:2, , drop = FALSE], backend = "faiss"), "must be one of")
})

test_that("knn immediately predicts when Xtest is supplied", {
  x <- matrix(c(
    0, 0,
    0, 1,
    5, 5,
    5, 6
  ), ncol = 2, byrow = TRUE)
  y <- factor(c("a", "a", "b", "b"))
  query <- matrix(c(0.1, 0.2, 5.2, 5.4), ncol = 2, byrow = TRUE)

  pred <- knn(x, y, query, backend = "cpu", k = 2L)
  expect_equal(as.character(pred), c("a", "b"))
  pred_meta <- attr(pred, "faissR_nn")
  expect_true(pred_meta$batch_query)
  expect_equal(pred_meta$query_n, 2L)
  expect_equal(pred_meta$query_call_count, 1L)
  expect_equal(pred_meta$query_source, "nn")

  proba <- knn(x, y, query, backend = "cpu", k = 2L, type = "prob")
  expect_equal(dim(proba), c(2L, 2L))
  expect_gt(proba[1, "a"], proba[1, "b"])
  expect_gt(proba[2, "b"], proba[2, "a"])
  proba_meta <- attr(proba, "faissR_nn")
  expect_true(proba_meta$batch_query)
  expect_equal(proba_meta$query_n, 2L)
  expect_equal(proba_meta$query_call_count, 1L)
  expect_equal(proba_meta$query_source, "nn")
})
