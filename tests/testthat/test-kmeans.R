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
  expect_equal(fit$hit_max_iter, fit$iter >= fit$parameters$max_iter)
  expect_equal(fit$converged, !isTRUE(fit$hit_max_iter))
  expect_equal(fit$parameters$hit_max_iter, fit$hit_max_iter)
  expect_equal(fit$parameters$converged, fit$converged)
  printed <- capture.output(print(fit))
  expect_true(any(grepl("faissR k-means", printed, fixed = TRUE)))
  expect_true(any(grepl("backend:", printed, fixed = TRUE)))
  expect_true(any(grepl("requested backend: cpu", printed, fixed = TRUE)))
  expect_true(any(grepl("resolved backend: cpu", printed, fixed = TRUE)))
  expect_true(any(grepl("converged before max_iter:", printed, fixed = TRUE)))
  expect_true(any(grepl("effective: max_iter=20", printed, fixed = TRUE)))
})

test_that("fast_kmeans records deterministic auto tuning policy", {
  set.seed(202)
  x <- matrix(rnorm(400), ncol = 4)

  auto <- fast_kmeans(x, centers = 3, backend = "cpu", seed = 12, n_threads = 2)
  expect_equal(auto$parameters$tuning$policy, "auto")
  expect_equal(auto$parameters$tuning$max_iter, 100L)
  expect_equal(auto$parameters$tuning$n_init, 5L)
  expect_equal(auto$parameters$tuning$rule, "small_low_work_multistart")
  expect_match(auto$parameters$tuning$rule_detail, "n=100", fixed = TRUE)
  expect_match(auto$parameters$tuning$rule_detail, "centers=3", fixed = TRUE)
  expect_equal(auto$parameters$tuning$resolved_from$max_iter, "auto")
  expect_equal(auto$parameters$tuning$resolved_from$n_init, "auto")
  expect_equal(auto$parameters$tuning$resolved_from$tol, "auto")
  expect_equal(auto$parameters$max_iter, 100L)
  expect_equal(auto$parameters$n_init, 5L)
  expect_equal(auto$parameters$tuning$effective$max_iter, auto$parameters$max_iter)
  expect_equal(auto$parameters$tuning$effective$n_init, auto$parameters$n_init)
  expect_equal(auto$parameters$tuning$effective$tol, auto$parameters$tol)
  expect_equal(auto$parameters$tuning$effective_max_iter, auto$parameters$max_iter)
  expect_equal(auto$parameters$tuning$effective_n_init, auto$parameters$n_init)
  expect_equal(auto$parameters$tuning$effective_tol, auto$parameters$tol)
  expect_equal(auto$parameters$tuning$small_many_centers, FALSE)
  expect_true(is.finite(auto$parameters$tuning$work))
  expect_true(is.finite(auto$parameters$tuning$n_per_center))
  expect_false(auto$parameters$tuning$backend_policy$prefer_cuda)
  expect_equal(auto$parameters$tuning$backend_policy$reason, "small_cpu_preferred")
  expect_true(is.finite(auto$parameters$tuning$backend_policy$work))
  expect_true(is.finite(auto$parameters$tuning$backend_policy$nbytes))
  expect_true(is.finite(auto$parameters$tuning$backend_policy$n_per_center))
  expect_equal(auto$parameters$tuning$backend_policy$work_threshold, 1e8)
  expect_equal(auto$parameters$tuning$backend_policy$nbytes_threshold, 256 * 1024^2)
  expect_equal(auto$parameters$tuning$backend_policy$large_n_threshold, 50000)
  expect_equal(auto$parameters$tuning$backend_policy$large_p_threshold, 128)
  expect_equal(auto$parameters$tuning$selection$policy, "static_shape_center_backend_selector")
  expect_false(auto$parameters$tuning$selection$slow_tuning)
  expect_equal(auto$parameters$tuning$selection$requested_backend, "cpu")
  expect_equal(auto$parameters$tuning$selection$predicted_backend, "cpu")
  expect_equal(auto$parameters$tuning$selection$resolved_backend, "cpu")
  expect_equal(auto$parameters$tuning$selection$n, nrow(x))
  expect_equal(auto$parameters$tuning$selection$p, ncol(x))
  expect_equal(auto$parameters$tuning$selection$centers, 3L)
  expect_equal(auto$parameters$tuning$selection$effective_max_iter, auto$parameters$max_iter)
  expect_equal(auto$parameters$tuning$selection$effective_n_init, auto$parameters$n_init)
  expect_equal(auto$parameters$tuning$selection$effective_tol, auto$parameters$tol)
  expect_equal(auto$parameters$tuning$selection$backend_policy_reason, "small_cpu_preferred")
  expect_true(auto$parameters$tuning$selection$explicit_backend)
  expect_equal(auto$parameters$tuning$selection$backend_decision, "explicit_cpu")
  expect_false(auto$parameters$tuning$selection$backend_policy_prefer_cuda)

  auto_backend <- fast_kmeans(x, centers = 3, backend = "auto", seed = 12, n_threads = 2)
  expect_equal(auto_backend$parameters$requested_backend, "auto")
  expect_true(auto_backend$parameters$resolved_backend %in% c("cpu", "cuda"))
  expect_true(auto_backend$backend %in% c("cpu", "faiss", "cuda_faiss", "cuda_cuvs"))
  expect_equal(auto_backend$parameters$tuning$selection$requested_backend, "auto")
  expect_equal(auto_backend$parameters$tuning$selection$predicted_backend, auto_backend$parameters$resolved_backend)
  expect_false(auto_backend$parameters$tuning$selection$explicit_backend)
  expect_equal(
    auto_backend$parameters$tuning$selection$backend_decision,
    auto_backend$parameters$tuning$selection$backend_policy_reason
  )

  small_many <- faissR:::kmeans_auto_params(
    n = 5000L,
    p = 10L,
    centers = 100L,
    tuning = "auto"
  )
  expect_equal(small_many$max_iter, 100L)
  expect_equal(small_many$n_init, 3L)
  expect_equal(small_many$rule, "small_many_centers_multistart")
  expect_match(small_many$rule_detail, "centers=100", fixed = TRUE)
  expect_true(isTRUE(small_many$many_centers))
  expect_true(isTRUE(small_many$small_many_centers))
  expect_equal(small_many$n_per_center, 50)

  few_points_many <- faissR:::kmeans_auto_params(
    n = 1000L,
    p = 10L,
    centers = 100L,
    tuning = "auto"
  )
  expect_equal(few_points_many$max_iter, 100L)
  expect_equal(few_points_many$n_init, 3L)
  expect_equal(few_points_many$tol, 1e-4)
  expect_equal(few_points_many$rule, "few_points_many_centers_multistart")
  expect_true(isTRUE(few_points_many$many_centers))
  expect_false(isTRUE(few_points_many$small_many_centers))
  expect_true(isTRUE(few_points_many$few_points_many_centers))
  expect_equal(few_points_many$n_per_center, 10)

  large_many <- faissR:::kmeans_auto_params(
    n = 200000L,
    p = 50L,
    centers = 100L,
    tuning = "auto"
  )
  expect_equal(large_many$max_iter, 50L)
  expect_equal(large_many$n_init, 1L)
  expect_equal(large_many$rule, "large_fast_convergence")
  expect_true(isTRUE(large_many$large_n))
  expect_false(isTRUE(large_many$small_many_centers))

  fixed <- fast_kmeans(x, centers = 3, backend = "cpu", tuning = "fixed", seed = 12, n_threads = 2)
  expect_equal(fixed$parameters$tuning$policy, "fixed")
  expect_equal(fixed$parameters$tuning$rule, "fixed_defaults")
  expect_match(fixed$parameters$tuning$rule_detail, "n=100", fixed = TRUE)
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
  expect_equal(explicit$parameters$tuning$effective$max_iter, 7L)
  expect_equal(explicit$parameters$tuning$effective$n_init, 2L)
  expect_equal(explicit$parameters$tuning$effective$tol, 1e-5)
})

test_that("fast_kmeans uses an exact trivial solution for one cluster on CPU and auto", {
  set.seed(2021)
  x <- matrix(rnorm(120), ncol = 6)
  expected_center <- matrix(colMeans(x), nrow = 1L)
  expected_within <- sum(sweep(x, 2L, expected_center[1L, ], "-")^2)

  fit <- fast_kmeans(x, centers = 1L, backend = "auto", seed = 99, n_threads = 2)
  expect_s3_class(fit, "faissR_kmeans")
  expect_equal(fit$backend, "trivial")
  expect_equal(fit$parameters$requested_backend, "auto")
  expect_equal(fit$parameters$resolved_backend, "trivial")
  expect_true(isTRUE(fit$parameters$exact_trivial_solution))
  expect_match(fit$parameters$backend_resolution_note, "no iterative CPU or CUDA backend was launched")
  expect_equal(fit$cluster, rep.int(1L, nrow(x)))
  expect_equal(fit$centers, expected_center)
  expect_equal(fit$withinss, expected_within)
  expect_equal(fit$tot.withinss, expected_within)
  expect_equal(fit$size, nrow(x))
  expect_equal(fit$iter, 0L)
  expect_false(fit$hit_max_iter)
  expect_true(fit$converged)
  expect_equal(fit$parameters$max_iter, 1L)
  expect_equal(fit$parameters$n_init, 1L)
  expect_equal(fit$parameters$tol, 0)
  expect_equal(fit$parameters$tuning$rule, "single_cluster_exact_mean")
  expect_equal(fit$parameters$tuning$backend_policy$reason, "single_cluster_exact_mean")
  expect_false(fit$parameters$tuning$backend_policy$prefer_cuda)
  expect_equal(fit$parameters$tuning$selection$predicted_backend, "cpu")

  cpu_fit <- fast_kmeans(x, centers = 1L, backend = "cpu", seed = 99, n_threads = 2)
  expect_equal(cpu_fit$backend, "trivial")
  expect_equal(cpu_fit$parameters$resolved_backend, "trivial")
  expect_equal(cpu_fit$centers, expected_center)
  expect_equal(cpu_fit$tot.withinss, expected_within)

  cuda_fit <- fast_kmeans(x, centers = 1L, backend = "cuda", seed = 99, n_threads = 2)
  expect_equal(cuda_fit$backend, "trivial")
  expect_equal(cuda_fit$parameters$requested_backend, "cuda")
  expect_equal(cuda_fit$parameters$resolved_backend, "trivial")
  expect_equal(cuda_fit$centers, expected_center)
  expect_equal(cuda_fit$tot.withinss, expected_within)
  expect_match(cuda_fit$parameters$backend_resolution_note, "Exact one-cluster solution")
})

test_that("fast_kmeans uses an exact trivial solution for singleton clusters on CPU and auto", {
  set.seed(2022)
  x <- matrix(rnorm(30), ncol = 3)

  fit <- fast_kmeans(x, centers = nrow(x), backend = "auto", seed = 99, n_threads = 2)
  expect_s3_class(fit, "faissR_kmeans")
  expect_equal(fit$backend, "trivial")
  expect_equal(fit$parameters$requested_backend, "auto")
  expect_equal(fit$parameters$resolved_backend, "trivial")
  expect_true(isTRUE(fit$parameters$exact_trivial_solution))
  expect_match(fit$parameters$backend_resolution_note, "no iterative CPU or CUDA backend was launched")
  expect_equal(fit$cluster, seq_len(nrow(x)))
  expect_equal(fit$centers, x)
  expect_equal(fit$withinss, rep.int(0, nrow(x)))
  expect_equal(fit$tot.withinss, 0)
  expect_equal(fit$size, rep.int(1L, nrow(x)))
  expect_equal(fit$iter, 0L)
  expect_false(fit$hit_max_iter)
  expect_true(fit$converged)
  expect_equal(fit$parameters$max_iter, 1L)
  expect_equal(fit$parameters$n_init, 1L)
  expect_equal(fit$parameters$tol, 0)
  expect_equal(fit$parameters$tuning$rule, "singleton_exact_identity")
  expect_equal(fit$parameters$tuning$backend_policy$reason, "singleton_exact_identity")
  expect_false(fit$parameters$tuning$backend_policy$prefer_cuda)
  expect_equal(fit$parameters$tuning$selection$predicted_backend, "cpu")

  cpu_fit <- fast_kmeans(x, centers = nrow(x), backend = "cpu", seed = 99, n_threads = 2)
  expect_equal(cpu_fit$backend, "trivial")
  expect_equal(cpu_fit$parameters$resolved_backend, "trivial")
  expect_equal(cpu_fit$centers, x)
  expect_equal(cpu_fit$tot.withinss, 0)

  cuda_fit <- fast_kmeans(x, centers = nrow(x), backend = "cuda", seed = 99, n_threads = 2)
  expect_equal(cuda_fit$backend, "trivial")
  expect_equal(cuda_fit$parameters$requested_backend, "cuda")
  expect_equal(cuda_fit$parameters$resolved_backend, "trivial")
  expect_equal(cuda_fit$centers, x)
  expect_equal(cuda_fit$tot.withinss, 0)
  expect_match(cuda_fit$parameters$backend_resolution_note, "Exact singleton solution")
})

test_that("kmeans auto parameter helper canonicalizes tuning labels", {
  auto <- faissR:::kmeans_auto_params(
    n = 100L,
    p = 4L,
    centers = 3L,
    tuning = " Auto "
  )
  expect_equal(auto$policy, "auto")
  expect_equal(auto$n_init, 5L)

  fixed <- faissR:::kmeans_auto_params(
    n = 100L,
    p = 4L,
    centers = 3L,
    tuning = "NONE"
  )
  expect_equal(fixed$policy, "none")
  expect_equal(fixed$n_init, 1L)
  expect_equal(fixed$rule, "fixed_defaults")

  expect_error(
    faissR:::kmeans_auto_params(
      n = 100L,
      p = 4L,
      centers = 3L,
      tuning = "pilot"
    ),
    "`tuning`"
  )
})

test_that("kmeans max-iteration helper is conservative for missing values", {
  expect_true(faissR:::kmeans_hit_max_iter(10L, 10L))
  expect_false(faissR:::kmeans_hit_max_iter(9L, 10L))
  expect_true(is.na(faissR:::kmeans_hit_max_iter(NA_integer_, 10L)))
  expect_true(is.na(faissR:::kmeans_hit_max_iter(10L, NA_integer_)))
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
  expect_equal(mnist10$rule, "medium_single_start")
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
  expect_equal(flow10$rule, "large_fast_convergence")
  expect_true(isTRUE(flow10$large_n))
  expect_false(isTRUE(flow10$high_dim))
  expect_equal(flow10$n_per_center, 100000)

  byte_policy <- faissR:::kmeans_auto_backend_policy(
    n = 300000L,
    p = 128L,
    centers = 2L
  )
  expect_true(byte_policy$prefer_cuda)
  expect_equal(byte_policy$reason, "large_high_dimensional_input")
  expect_equal(byte_policy$nbytes_threshold, 256 * 1024^2)
  expect_true(byte_policy$nbytes >= byte_policy$nbytes_threshold)
  expect_true(byte_policy$gpu_transfer_nbytes < byte_policy$nbytes_threshold)

  high_dim_policy <- faissR:::kmeans_auto_backend_policy(
    n = 50000L,
    p = 128L,
    centers = 2L
  )
  expect_true(high_dim_policy$prefer_cuda)
  expect_equal(high_dim_policy$reason, "large_high_dimensional_input")
  expect_equal(high_dim_policy$large_n_threshold, 50000)
  expect_equal(high_dim_policy$large_p_threshold, 128)

  small_many <- faissR:::kmeans_auto_params(
    n = 50000L,
    p = 10L,
    centers = 100L,
    tuning = "auto"
  )
  expect_equal(small_many$max_iter, 100L)
  expect_equal(small_many$n_init, 3L)
  expect_equal(small_many$tol, 1e-4)
  expect_equal(small_many$rule, "small_many_centers_multistart")
  expect_true(isTRUE(small_many$many_centers))
  expect_true(isTRUE(small_many$small_many_centers))
  expect_false(isTRUE(small_many$few_points_many_centers))
  expect_equal(small_many$n_per_center, 500)

  few_points_many <- faissR:::kmeans_auto_params(
    n = 1000L,
    p = 10L,
    centers = 100L,
    tuning = "auto"
  )
  expect_equal(few_points_many$max_iter, 100L)
  expect_equal(few_points_many$n_init, 3L)
  expect_equal(few_points_many$tol, 1e-4)
  expect_equal(few_points_many$rule, "few_points_many_centers_multistart")
  expect_true(isTRUE(few_points_many$many_centers))
  expect_false(isTRUE(few_points_many$small_many_centers))
  expect_true(isTRUE(few_points_many$few_points_many_centers))
  expect_equal(few_points_many$n_per_center, 10)

  highdim_many <- faissR:::kmeans_auto_params(
    n = 70000L,
    p = 784L,
    centers = 100L,
    tuning = "auto"
  )
  expect_equal(highdim_many$max_iter, 50L)
  expect_equal(highdim_many$n_init, 1L)
  expect_equal(highdim_many$tol, 1e-3)
  expect_equal(highdim_many$rule, "large_fast_convergence")
  expect_true(isTRUE(highdim_many$high_dim))
  expect_true(isTRUE(highdim_many$many_centers))
  expect_false(isTRUE(highdim_many$small_many_centers))
})

test_that("fast_kmeans auto backend requires a k-means capable CUDA route", {
  expect_equal(
    faissR:::resolve_fast_kmeans_backend(
      "auto",
      n = 100000L,
      p = 784L,
      centers = 10L,
      cuda_available_value = FALSE,
      faiss_gpu_available_value = TRUE,
      cuvs_available_value = TRUE
    ),
    "cpu"
  )
  expect_equal(
    faissR:::resolve_fast_kmeans_backend(
      "auto",
      n = 100000L,
      p = 784L,
      centers = 10L,
      cuda_available_value = TRUE,
      faiss_gpu_available_value = FALSE,
      cuvs_available_value = FALSE
    ),
    "cpu"
  )
  expect_equal(
    faissR:::resolve_fast_kmeans_backend(
      "auto",
      n = 100000L,
      p = 784L,
      centers = 10L,
      cuda_available_value = TRUE,
      faiss_gpu_available_value = TRUE,
      cuvs_available_value = FALSE
    ),
    "cuda"
  )
  expect_equal(
    faissR:::resolve_fast_kmeans_backend(
      "auto",
      n = 100000L,
      p = 784L,
      centers = 10L,
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

test_that("fast_kmeans auto backend is shape-aware", {
  expect_false(faissR:::kmeans_auto_prefers_cuda(n = 120L, p = 4L, centers = 3L))
  expect_false(faissR:::kmeans_auto_prefers_cuda(n = 1000000L, p = 784L, centers = 1L))
  expect_true(faissR:::kmeans_auto_prefers_cuda(n = 70000L, p = 784L, centers = 10L))
  expect_true(faissR:::kmeans_auto_prefers_cuda(n = 500000L, p = 32L, centers = 10L))
  expect_true(faissR:::kmeans_auto_prefers_cuda(n = NULL, p = NULL, centers = NULL))

  small_policy <- faissR:::kmeans_auto_backend_policy(n = 120L, p = 4L, centers = 3L)
  expect_false(small_policy$prefer_cuda)
  expect_equal(small_policy$reason, "small_cpu_preferred")
  expect_equal(small_policy$work, 1440)
  expect_equal(small_policy$nbytes, 3840)
  expect_equal(small_policy$input_nbytes, 3840)
  expect_equal(small_policy$gpu_transfer_nbytes, 1920)
  expect_equal(small_policy$n_per_center, 40)
  expect_equal(small_policy$work_threshold, 1e8)
  expect_equal(small_policy$nbytes_threshold, 256 * 1024^2)
  expect_equal(small_policy$large_n_threshold, 50000)
  expect_equal(small_policy$large_p_threshold, 128)
  expect_equal(small_policy$min_n_per_center, 20)

  withr::with_options(
    list(
      faissR.kmeans_cuda_work_threshold = 1000,
      faissR.kmeans_cuda_nbytes_threshold = 1e9,
      faissR.kmeans_cuda_large_n_threshold = 1e9,
      faissR.kmeans_cuda_large_p_threshold = 1e9,
      faissR.kmeans_cuda_min_n_per_center = 5
    ),
    {
      tuned_policy <- faissR:::kmeans_auto_backend_policy(n = 120L, p = 4L, centers = 3L)
      expect_true(tuned_policy$prefer_cuda)
      expect_equal(tuned_policy$reason, "work_at_least_1e8")
      expect_equal(tuned_policy$work_threshold, 1000)
      expect_equal(tuned_policy$nbytes_threshold, 1e9)
      expect_equal(tuned_policy$large_n_threshold, 1e9)
      expect_equal(tuned_policy$large_p_threshold, 1e9)
    }
  )

  withr::with_options(
    list(
      faissR.kmeans_cuda_work_threshold = -1,
      faissR.kmeans_cuda_nbytes_threshold = "bad",
      faissR.kmeans_cuda_large_n_threshold = 0,
      faissR.kmeans_cuda_large_p_threshold = NA,
      faissR.kmeans_cuda_min_n_per_center = 0
    ),
    {
      fallback_policy <- faissR:::kmeans_auto_backend_policy(n = 120L, p = 4L, centers = 3L)
      expect_equal(fallback_policy$work_threshold, 1e8)
      expect_equal(fallback_policy$nbytes_threshold, 256 * 1024^2)
      expect_equal(fallback_policy$large_n_threshold, 50000)
      expect_equal(fallback_policy$large_p_threshold, 128)
      expect_equal(fallback_policy$min_n_per_center, 20)
    }
  )

  work_policy <- faissR:::kmeans_auto_backend_policy(n = 70000L, p = 784L, centers = 10L)
  expect_true(work_policy$prefer_cuda)
  expect_equal(work_policy$reason, "work_at_least_1e8")

  cuda_selection <- faissR:::kmeans_selection_metadata(
    requested_backend = "auto",
    resolved_backend = "cuda",
    n = 70000L,
    p = 784L,
    centers = 10L,
    effective = list(max_iter = 75L, n_init = 1L, tol = 1e-4),
    backend_policy = work_policy,
    tuning = "auto",
    cuda_available_value = TRUE,
    faiss_gpu_available_value = TRUE,
    cuvs_available_value = FALSE
  )
  expect_equal(cuda_selection$policy, "static_shape_center_backend_selector")
  expect_equal(cuda_selection$predicted_backend, "cuda")
  expect_equal(cuda_selection$backend_policy_reason, "work_at_least_1e8")
  expect_false(cuda_selection$explicit_backend)
  expect_equal(cuda_selection$backend_decision, "work_at_least_1e8")
  expect_true(cuda_selection$backend_policy_prefer_cuda)
  expect_equal(cuda_selection$input_nbytes, work_policy$input_nbytes)
  expect_equal(cuda_selection$gpu_transfer_nbytes, work_policy$gpu_transfer_nbytes)
  expect_true(cuda_selection$cuda_available)
  expect_true(cuda_selection$faiss_gpu_available)
  expect_false(cuda_selection$cuvs_available)
  expect_equal(cuda_selection$effective_max_iter, 75L)
  expect_equal(cuda_selection$effective_n_init, 1L)
  expect_equal(cuda_selection$effective_tol, 1e-4)

  memory_policy <- faissR:::kmeans_auto_backend_policy(n = 9000000L, p = 4L, centers = 2L)
  expect_false(memory_policy$prefer_cuda)
  expect_equal(memory_policy$reason, "small_cpu_preferred")
  expect_true(memory_policy$nbytes >= memory_policy$nbytes_threshold)
  expect_true(memory_policy$gpu_transfer_nbytes < memory_policy$nbytes_threshold)

  withr::with_options(
    list(faissR.kmeans_cuda_work_threshold = 1e12),
    {
      transfer_policy <- faissR:::kmeans_auto_backend_policy(n = 20000000L, p = 4L, centers = 2L)
      expect_true(transfer_policy$prefer_cuda)
      expect_equal(transfer_policy$reason, "input_at_least_256MiB")
      expect_true(transfer_policy$gpu_transfer_nbytes >= transfer_policy$nbytes_threshold)
    }
  )

  high_dim_policy <- faissR:::kmeans_auto_backend_policy(n = 60000L, p = 128L, centers = 2L)
  expect_true(high_dim_policy$prefer_cuda)
  expect_equal(high_dim_policy$reason, "large_high_dimensional_input")

  few_points_policy <- faissR:::kmeans_auto_backend_policy(n = 5000L, p = 512L, centers = 1000L)
  expect_false(few_points_policy$prefer_cuda)
  expect_equal(few_points_policy$reason, "few_points_per_center_cpu_preferred")
  expect_equal(few_points_policy$n_per_center, 5)
  expect_equal(few_points_policy$min_n_per_center, 20)

  singleton_policy <- faissR:::kmeans_auto_backend_policy(n = 120L, p = 4L, centers = 120L)
  expect_false(singleton_policy$prefer_cuda)
  expect_equal(singleton_policy$reason, "singleton_exact_identity")
  expect_equal(singleton_policy$n_per_center, 1)

  unknown_policy <- faissR:::kmeans_auto_backend_policy(n = NULL, p = NULL, centers = NULL)
  expect_true(unknown_policy$prefer_cuda)
  expect_equal(unknown_policy$reason, "unknown_shape")

  expect_equal(
    faissR:::resolve_fast_kmeans_backend(
      "auto",
      n = 120L,
      p = 4L,
      centers = 3L,
      cuda_available_value = TRUE,
      faiss_gpu_available_value = TRUE,
      cuvs_available_value = TRUE
    ),
    "cpu"
  )
  expect_equal(
    faissR:::resolve_fast_kmeans_backend(
      "auto",
      n = 1000000L,
      p = 784L,
      centers = 1L,
      cuda_available_value = TRUE,
      faiss_gpu_available_value = TRUE,
      cuvs_available_value = TRUE
    ),
    "cpu"
  )
  expect_equal(
    faissR:::resolve_fast_kmeans_backend(
      "auto",
      n = 70000L,
      p = 784L,
      centers = 10L,
      cuda_available_value = TRUE,
      faiss_gpu_available_value = TRUE,
      cuvs_available_value = FALSE
    ),
    "cuda"
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

test_that("fast_kmeans rejects invalid explicit tolerance values", {
  x <- matrix(rnorm(40), ncol = 4)

  expect_error(
    fast_kmeans(x, centers = 2, backend = "cpu", tol = c(1e-4, 1e-5), n_threads = 2),
    "`tol` must be `auto` or a single non-negative finite number"
  )
  expect_error(
    fast_kmeans(x, centers = 2, backend = "cpu", tol = -1e-4, n_threads = 2),
    "`tol` must be `auto` or a single non-negative finite number"
  )
  expect_error(
    fast_kmeans(x, centers = 2, backend = "cpu", tol = "small", n_threads = 2),
    "`tol` must be `auto` or a single non-negative finite number"
  )
  expect_error(
    fast_kmeans(x, centers = 2, backend = "cpu", tol = NA_real_, n_threads = 2),
    "`tol` must be `auto` or a single non-negative finite number"
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

test_that("CUDA k-means records direct cuVS provider selection", {
  x <- matrix(rnorm(40), ncol = 4)
  fake_cuvs <- function(...) {
    list(
      cluster = rep(1:2, length.out = nrow(x)),
      centers = matrix(0, nrow = 2, ncol = ncol(x)),
      withinss = c(1, 2),
      tot.withinss = 3,
      size = c(5L, 5L),
      iter = 1L,
      parameters = list(max_iter = 2L, n_init = 1L, tol = 1e-4)
    )
  }

  testthat::local_mocked_bindings(
    faiss_gpu_available = function() FALSE,
    cuvs_available = function() TRUE,
    cuda_available = function() TRUE,
    kmeans_cuvs_cpp = fake_cuvs,
    .package = "faissR"
  )

  out <- faissR:::run_cuda_kmeans(
    x = x,
    centers = 2L,
    max_iter = 2L,
    n_init = 1L,
    tol = 1e-4,
    seed = 1L,
    streaming_batch_size = 0L,
    init = "kmeans++",
    tuning_metadata = list(effective = list(max_iter = 2L, n_init = 1L, tol = 1e-4)),
    requested_backend = "cuda",
    resolved_backend = "cuda"
  )

  expect_equal(out$backend, "cuda_cuvs")
  expect_equal(
    out$parameters$cuda_provider_selection,
    "direct_cuvs_after_faiss_gpu_unavailable_or_failed"
  )
  expect_match(out$parameters$faiss_gpu_error, "FAISS GPU support is unavailable", fixed = TRUE)
  expect_match(out$parameters$backend_resolution_note, "Direct cuVS k-means", fixed = TRUE)
  expect_equal(out$parameters$requested_backend, "cuda")
  expect_equal(out$parameters$resolved_backend, "cuda")
})

test_that("CUDA k-means records FAISS GPU provider selection", {
  x <- matrix(rnorm(40), ncol = 4)
  fake_faiss_gpu <- function(...) {
    list(
      cluster = rep(1:2, length.out = nrow(x)),
      centers = matrix(0, nrow = 2, ncol = ncol(x)),
      withinss = c(1, 2),
      tot.withinss = 3,
      size = c(5L, 5L),
      iter = 1L,
      parameters = list(max_iter = 2L, n_init = 1L, tol = 1e-4)
    )
  }

  testthat::local_mocked_bindings(
    faiss_gpu_available = function() TRUE,
    run_faiss_gpu_kmeans = fake_faiss_gpu,
    .package = "faissR"
  )

  out <- faissR:::run_cuda_kmeans(
    x = x,
    centers = 2L,
    max_iter = 2L,
    n_init = 1L,
    tol = 1e-4,
    seed = 1L,
    streaming_batch_size = 0L,
    init = "kmeans++",
    tuning_metadata = list(effective = list(max_iter = 2L, n_init = 1L, tol = 1e-4)),
    requested_backend = "cuda",
    resolved_backend = "cuda"
  )

  expect_equal(out$backend, "cuda_faiss")
  expect_equal(out$parameters$cuda_provider_selection, "faiss_gpu")
  expect_match(out$parameters$backend_resolution_note, "FAISS GPU k-means", fixed = TRUE)
  expect_null(out$parameters$faiss_gpu_error)
  expect_equal(out$parameters$requested_backend, "cuda")
  expect_equal(out$parameters$resolved_backend, "cuda")
})
