test_that("nearest-neighbor summaries expose stable metadata", {
  x <- matrix(c(0, 0, 1, 0, 0, 1, 2, 2), ncol = 2L, byrow = TRUE)
  result <- nn(
    x,
    k = 2L,
    exclude_self = TRUE,
    backend = "cpu",
    method = "exact",
      n_threads = 1L
  )

  info <- summary(result)
  expect_s3_class(info, "data.frame")
  expect_equal(info$queries, 4L)
  expect_equal(info$neighbors, 2L)
  expect_equal(info$metric, "euclidean")
  expect_true(info$exact)
  distance_summary <- unlist(
    info[c("min_distance", "median_distance", "max_distance")],
    use.names = FALSE
  )
  expect_true(all(is.finite(distance_summary)))
  expect_true(info$exclude_self)
  expect_identical(result$exact, attr(result, "exact"))
  expect_silent(capture.output(printed <- print(result)))
  expect_identical(printed, result)
  expect_false(any(grepl("first column: self-neighbor", capture.output(print(result)))))
})

test_that("fitted kNN models have print and summary methods", {
  x <- scale(as.matrix(iris[, 1:4]))
  model <- knn(
    x,
    iris$Species,
    k = 3L,
    backend = "cpu",
    method = "flat",
      n_threads = 1L
  )

  info <- summary(model)
  expect_s3_class(info, "data.frame")
  expect_equal(info$observations, nrow(x))
  expect_equal(info$features, ncol(x))
  expect_equal(info$task, "classification")
  expect_equal(info$method, "flat")
  expect_output(print(model), "faissR kNN model")
})

test_that("k-means summaries report fit dimensions", {
  fit <- fast_kmeans(
    as.matrix(iris[, 1:4]),
    centers = 3L,
    backend = "cpu",
    n_threads = 1L
  )
  info <- summary(fit)
  expect_equal(info$observations, nrow(iris))
  expect_equal(info$clusters, 3L)
  expect_true(is.finite(info$total_withinss))
})
