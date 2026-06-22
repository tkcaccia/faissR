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

  expect_s3_class(fit, "faissR_kmeans")
  expect_equal(length(fit$cluster), nrow(x))
  expect_equal(dim(fit$centers), c(2L, ncol(x)))
  expect_equal(length(fit$withinss), 2L)
  expect_equal(length(fit$size), 2L)
  expect_true(all(fit$cluster %in% 1:2))
  expect_equal(sum(fit$size), nrow(x))
  expect_true(is.finite(fit$tot.withinss))
  expect_true(fit$backend %in% c("cpu", "faiss"))
  expect_equal(fit$parameters$requested_backend, "cpu")
  expect_equal(fit$parameters$resolved_backend, "cpu")
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
  expect_equal(auto$parameters$tuning$small_many_centers, FALSE)
  expect_true(is.finite(auto$parameters$tuning$work))
  expect_true(is.finite(auto$parameters$tuning$n_per_center))

  auto_backend <- fast_kmeans(x, centers = 3, backend = "auto", seed = 12, n_threads = 2)
  expect_equal(auto_backend$parameters$requested_backend, "auto")
  expect_true(auto_backend$parameters$resolved_backend %in% c("cpu", "cuda"))
  expect_true(auto_backend$backend %in% c("cpu", "faiss", "cuda_faiss", "cuda_cuvs"))

  small_many <- faissR:::kmeans_auto_params(
    n = 5000L,
    p = 10L,
    centers = 100L,
    tuning = "auto"
  )
  expect_equal(small_many$max_iter, 100L)
  expect_equal(small_many$n_init, 3L)
  expect_true(isTRUE(small_many$many_centers))
  expect_true(isTRUE(small_many$small_many_centers))
  expect_equal(small_many$n_per_center, 50)

  large_many <- faissR:::kmeans_auto_params(
    n = 200000L,
    p = 50L,
    centers = 100L,
    tuning = "auto"
  )
  expect_equal(large_many$max_iter, 50L)
  expect_equal(large_many$n_init, 1L)
  expect_true(isTRUE(large_many$large_n))
  expect_false(isTRUE(large_many$small_many_centers))

  fixed <- fast_kmeans(x, centers = 3, backend = "cpu", tuning = "fixed", seed = 12, n_threads = 2)
  expect_equal(fixed$parameters$tuning$policy, "fixed")
  expect_equal(fixed$parameters$max_iter, 100L)
  expect_equal(fixed$parameters$n_init, 1L)
  expect_true(is.finite(fixed$parameters$tuning$work))
  expect_true(is.finite(fixed$parameters$tuning$n_per_center))
  expect_false(isTRUE(fixed$parameters$tuning$high_dim))
  expect_false(isTRUE(fixed$parameters$tuning$large_n))

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

test_that("fast_kmeans auto tuning is shape and center-count aware for benchmark shapes", {
  mnist10 <- faissR:::kmeans_auto_params(
    n = 70000L,
    p = 784L,
    centers = 10L,
    tuning = "auto"
  )
  expect_equal(mnist10$max_iter, 75L)
  expect_equal(mnist10$n_init, 1L)
  expect_equal(mnist10$tol, 1e-4)
  expect_true(isTRUE(mnist10$high_dim))
  expect_false(isTRUE(mnist10$large_n))
  expect_false(isTRUE(mnist10$many_centers))
  expect_equal(mnist10$n_per_center, 7000)

  flow10 <- faissR:::kmeans_auto_params(
    n = 1000000L,
    p = 32L,
    centers = 10L,
    tuning = "auto"
  )
  expect_equal(flow10$max_iter, 50L)
  expect_equal(flow10$n_init, 1L)
  expect_equal(flow10$tol, 1e-3)
  expect_true(isTRUE(flow10$large_n))
  expect_false(isTRUE(flow10$high_dim))
  expect_equal(flow10$n_per_center, 100000)

  small_many <- faissR:::kmeans_auto_params(
    n = 50000L,
    p = 10L,
    centers = 100L,
    tuning = "auto"
  )
  expect_equal(small_many$max_iter, 100L)
  expect_equal(small_many$n_init, 3L)
  expect_equal(small_many$tol, 1e-4)
  expect_true(isTRUE(small_many$many_centers))
  expect_true(isTRUE(small_many$small_many_centers))
  expect_equal(small_many$n_per_center, 500)

  highdim_many <- faissR:::kmeans_auto_params(
    n = 70000L,
    p = 784L,
    centers = 100L,
    tuning = "auto"
  )
  expect_equal(highdim_many$max_iter, 50L)
  expect_equal(highdim_many$n_init, 1L)
  expect_equal(highdim_many$tol, 1e-3)
  expect_true(isTRUE(highdim_many$high_dim))
  expect_true(isTRUE(highdim_many$many_centers))
  expect_false(isTRUE(highdim_many$small_many_centers))
})

test_that("fast_kmeans auto backend requires a k-means capable CUDA route", {
  expect_equal(
    faissR:::resolve_fast_kmeans_backend(
      "auto",
      cuda_available_value = FALSE,
      faiss_gpu_available_value = TRUE,
      cuvs_available_value = TRUE
    ),
    "cpu"
  )
  expect_equal(
    faissR:::resolve_fast_kmeans_backend(
      "auto",
      cuda_available_value = TRUE,
      faiss_gpu_available_value = FALSE,
      cuvs_available_value = FALSE
    ),
    "cpu"
  )
  expect_equal(
    faissR:::resolve_fast_kmeans_backend(
      "auto",
      cuda_available_value = TRUE,
      faiss_gpu_available_value = TRUE,
      cuvs_available_value = FALSE
    ),
    "cuda"
  )
  expect_equal(
    faissR:::resolve_fast_kmeans_backend(
      "auto",
      cuda_available_value = TRUE,
      faiss_gpu_available_value = FALSE,
      cuvs_available_value = TRUE
    ),
    "cuda"
  )
  expect_equal(
    faissR:::resolve_fast_kmeans_backend(
      "cuda",
      cuda_available_value = FALSE,
      faiss_gpu_available_value = FALSE,
      cuvs_available_value = FALSE
    ),
    "cuda"
  )
  expect_error(
    faissR:::resolve_fast_kmeans_backend("faiss"),
    "must be one of"
  )
})

test_that("fast_kmeans rejects implementation backend labels", {
  x <- matrix(rnorm(40), ncol = 4)
  expect_error(
    fast_kmeans(x, centers = 2, backend = "faiss"),
    "must be one of"
  )
  expect_error(
    fast_kmeans(x, centers = 2, backend = "cuda_cuvs"),
    "must be one of"
  )
})

test_that("fast_kmeans requires canonical initialization labels", {
  x <- matrix(rnorm(40), ncol = 4)

  expect_equal(faissR:::normalize_kmeans_init(NULL), "kmeans++")
  expect_equal(faissR:::normalize_kmeans_init("random"), "random")
  expect_error(faissR:::normalize_kmeans_init("r"), "init")
  expect_error(
    faissR:::normalize_kmeans_init(c("random", "kmeans++")),
    "`init` must be a single value"
  )
  expect_error(
    faissR:::normalize_kmeans_tuning(c("off", "auto")),
    "`tuning` must be a single value"
  )
  expect_no_error(fast_kmeans(x, centers = 2, backend = "cpu", n_threads = 2))
  expect_error(
    fast_kmeans(x, centers = 2, backend = "cpu", init = "r", n_threads = 2),
    "init"
  )
})

test_that("fast_kmeans validates scalar seed and streaming batch size", {
  x <- matrix(rnorm(40), ncol = 4)

  expect_equal(faissR:::normalize_kmeans_seed(12L), 12L)
  expect_equal(faissR:::normalize_kmeans_streaming_batch_size(0L), 0L)
  expect_equal(faissR:::normalize_kmeans_streaming_batch_size(128L), 128L)
  expect_error(faissR:::normalize_kmeans_seed(1.5), "single finite integer")
  expect_error(faissR:::normalize_kmeans_seed(c(1L, 2L)), "single finite integer")
  expect_error(faissR:::normalize_kmeans_seed(NA_integer_), "single finite integer")
  expect_error(
    faissR:::normalize_kmeans_streaming_batch_size(1.5),
    "single non-negative integer"
  )
  expect_error(
    faissR:::normalize_kmeans_streaming_batch_size(c(0L, 128L)),
    "single non-negative integer"
  )
  expect_error(
    faissR:::normalize_kmeans_streaming_batch_size(-1L),
    "single non-negative integer"
  )
  expect_error(
    fast_kmeans(x, centers = 2, backend = "cpu", seed = c(1L, 2L), n_threads = 2),
    "seed"
  )
  expect_error(
    fast_kmeans(x, centers = 2, backend = "cpu", streaming_batch_size = c(0L, 128L), n_threads = 2),
    "streaming_batch_size"
  )
})

test_that("fast_kmeans integer controls reject fractional values", {
  x <- matrix(rnorm(40), ncol = 4)

  expect_error(
    fast_kmeans(x, centers = 2.5, backend = "cpu", n_threads = 2),
    "`centers` must be a single positive integer"
  )
  expect_error(
    fast_kmeans(x, centers = 2, backend = "cpu", max_iter = 2.5, n_threads = 2),
    "`max_iter` must be a single positive integer"
  )
  expect_error(
    fast_kmeans(x, centers = 2, backend = "cpu", n_init = 1.5, n_threads = 2),
    "`n_init` must be a single positive integer"
  )
  expect_error(
    fast_kmeans(x, centers = 2, backend = "cpu", seed = 1.5, n_threads = 2),
    "`seed` must be a single finite integer"
  )
  expect_error(
    fast_kmeans(x, centers = 2, backend = "cpu", streaming_batch_size = 1.5, n_threads = 2),
    "`streaming_batch_size` must be a single non-negative integer"
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
