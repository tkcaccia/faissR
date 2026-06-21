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

test_that("fast_kmeans records deterministic auto tuning policy", {
  set.seed(202)
  x <- matrix(rnorm(400), ncol = 4)

  auto <- fast_kmeans(x, centers = 3, backend = "cpu", seed = 12, n_threads = 2)
  expect_equal(auto$parameters$tuning$policy, "auto")
  expect_equal(auto$parameters$tuning$max_iter, 100L)
  expect_equal(auto$parameters$tuning$n_init, 5L)
  expect_equal(auto$parameters$tuning$resolved_from$max_iter, "auto")
  expect_equal(auto$parameters$tuning$resolved_from$n_init, "auto")
  expect_equal(auto$parameters$tuning$resolved_from$tol, "auto")
  expect_equal(auto$parameters$max_iter, 100L)
  expect_equal(auto$parameters$n_init, 5L)

  fixed <- fast_kmeans(x, centers = 3, backend = "cpu", tuning = "fixed", seed = 12, n_threads = 2)
  expect_equal(fixed$parameters$tuning$policy, "fixed")
  expect_equal(fixed$parameters$max_iter, 100L)
  expect_equal(fixed$parameters$n_init, 1L)

  explicit <- fast_kmeans(
    x,
    centers = 3,
    backend = "cpu",
    max_iter = 7L,
    n_init = 2L,
    tol = 1e-5,
    seed = 12,
    n_threads = 2
  )
  expect_equal(explicit$parameters$max_iter, 7L)
  expect_equal(explicit$parameters$n_init, 2L)
  expect_equal(explicit$parameters$tol, 1e-5)
  expect_equal(explicit$parameters$tuning$resolved_from$max_iter, "explicit")
  expect_equal(explicit$parameters$tuning$resolved_from$n_init, "explicit")
  expect_equal(explicit$parameters$tuning$resolved_from$tol, "explicit")
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

test_that("fast_kmeans CUDA requests never silently use CPU", {
  x <- matrix(rnorm(80), ncol = 4)
  out <- tryCatch(
    fast_kmeans(x, centers = 2, backend = "cuda", max_iter = 2),
    error = identity
  )
  if (inherits(out, "error")) {
    expect_match(conditionMessage(out), "CUDA k-means|FAISS GPU|cuVS")
  } else {
    expect_true(out$backend %in% c("cuda_faiss", "cuda_cuvs"))
    expect_false(identical(out$backend, "cpu"))
    expect_false(identical(out$backend, "faiss"))
  }
})
