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
  expect_equal(proba_meta$requested_backend, "auto")
  expect_equal(proba_meta$requested_method, "exact")
  expect_equal(proba_meta$tuning, "off")
  expect_equal(proba_meta$metric, "cosine")
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

test_that("knn stores canonical metric labels for aliases", {
  x <- matrix(c(
    1, 0,
    0, 1,
    1, 1,
    2, 0
  ), ncol = 2, byrow = TRUE)
  y <- factor(c("a", "b", "a", "b"))

  fit <- knn(x, y, backend = "cpu", metric = "ip", k = 2L)
  pred <- predict(fit, x[1:2, , drop = FALSE])

  expect_equal(fit$metric, "inner_product")
  expect_s3_class(pred, "factor")
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

  proba <- knn(x, y, query, backend = "cpu", k = 2L, type = "prob")
  expect_equal(dim(proba), c(2L, 2L))
  expect_gt(proba[1, "a"], proba[1, "b"])
  expect_gt(proba[2, "b"], proba[2, "a"])
})
