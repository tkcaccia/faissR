test_that("fast_kmeans works on the CPU path", {
  set.seed(201)
  x <- rbind(
    matrix(rnorm(80, -2), ncol = 4),
    matrix(rnorm(80, 2), ncol = 4)
  )
  fit <- fast_kmeans(
    x,
    centers = 2,
    backend = "cpu",
    max_iter = 20,
    n_init = 2,
    seed = 11,
    n_threads = 2
  )

  expect_s3_class(fit, "fastEmbedR_kmeans")
  expect_equal(length(fit$cluster), nrow(x))
  expect_equal(dim(fit$centers), c(2L, ncol(x)))
  expect_equal(length(fit$withinss), 2L)
  expect_equal(length(fit$size), 2L)
  expect_true(all(fit$cluster %in% 1:2))
  expect_equal(sum(fit$size), nrow(x))
  expect_true(is.finite(fit$tot.withinss))
  expect_true(fit$backend %in% c("cpu", "faiss"))
})

test_that("fast_kmeans explicit FAISS request fails clearly when unavailable", {
  skip_if(faiss_available())

  x <- matrix(rnorm(40), ncol = 4)
  expect_error(
    fast_kmeans(x, centers = 2, backend = "faiss"),
    "FAISS k-means"
  )
})

test_that("fast_kmeans explicit cuVS request fails clearly when unavailable", {
  skip_if(cuvs_available())

  x <- matrix(rnorm(40), ncol = 4)
  expect_error(
    fast_kmeans(x, centers = 2, backend = "cuda_cuvs"),
    "cuVS"
  )
})
