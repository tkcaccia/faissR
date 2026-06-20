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
  expect_s3_class(fit, "fastEmbedR_knn_model")
  expect_equal(as.character(pred), c("a", "b"))
  expect_equal(levels(pred), levels(y))

  proba <- predict(fit, matrix(c(0.1, 0.2, 5.2, 5.4), ncol = 2, byrow = TRUE), type = "prob")
  expect_equal(dim(proba), c(2L, 2L))
  expect_equal(colnames(proba), c("a", "b"))
  expect_equal(rowSums(proba), c(1, 1), tolerance = 1e-12)
  expect_gt(proba[1, "a"], proba[1, "b"])
  expect_gt(proba[2, "b"], proba[2, "a"])
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
  expect_equal(pred, 1, tolerance = 1e-12)
  expect_lt(pred_w, pred)
  expect_error(predict(fit, x, type = "prob"), "classification")
})

test_that("knn preserves explicit backend requests", {
  x <- matrix(rnorm(30), ncol = 3)
  y <- rep(c("a", "b"), length.out = nrow(x))

  faiss_model <- knn(x, y, backend = "faiss", k = 3L)
  expect_equal(faiss_model$backend, "faiss")
  if (!faiss_available()) {
    expect_error(predict(faiss_model, x[1:2, , drop = FALSE]), "FAISS")
  }

  cuvs_model <- knn(x, y, backend = "cuda_cuvs", k = 3L)
  expect_equal(cuvs_model$backend, "cuda_cuvs")
  if (!cuvs_available()) {
    expect_error(predict(cuvs_model, x[1:2, , drop = FALSE]), "cuVS")
  }
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
