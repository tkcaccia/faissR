test_that("candidate_knn matches exact self-KNN with full candidate rows", {
  set.seed(301)
  x <- matrix(rnorm(30 * 4), nrow = 30)
  candidates <- matrix(rep(seq_len(nrow(x)), times = nrow(x)), nrow = nrow(x), byrow = TRUE)

  exact <- nn(x, k = 5L, backend = "cpu")
  cand <- candidate_knn(x, candidates, k = 5L, backend = "cpu", n_threads = 2L)

  expect_s3_class(cand, "faissR_nn")
  expect_equal(attr(cand, "backend"), "cpu_candidate")
  expect_false(isTRUE(attr(cand, "exact")))
  expect_true(isTRUE(attr(cand, "candidate_knn")$exact_within_candidates))
  expect_equal(cand$indices, exact$indices)
  expect_equal(cand$distances, exact$distances, tolerance = 1e-12)
})

test_that("candidate_knn supports query candidates", {
  data <- matrix(c(
    0, 0,
    1, 0,
    5, 0,
    9, 0
  ), ncol = 2, byrow = TRUE)
  points <- matrix(c(
    0.2, 0,
    8.8, 0
  ), ncol = 2, byrow = TRUE)
  candidates <- matrix(c(
    1L, 2L, 3L,
    2L, 3L, 4L
  ), nrow = 2, byrow = TRUE)

  out <- candidate_knn(data, candidates, points = points, k = 2L, backend = "cpu")

  expect_equal(out$indices[1, ], c(1L, 2L))
  expect_equal(out$indices[2, ], c(4L, 3L))
  expect_equal(out$distances[1, ], c(0.2, 0.8), tolerance = 1e-12)
  expect_equal(out$distances[2, ], c(0.2, 3.8), tolerance = 1e-12)
})

test_that("candidate_knn exclude_self matches exact CPU without self", {
  set.seed(302)
  x <- matrix(rnorm(25 * 3), nrow = 25)
  candidates <- matrix(rep(seq_len(nrow(x)), times = nrow(x)), nrow = nrow(x), byrow = TRUE)

  exact <- faissR:::nn(exclude_self = TRUE, x, k = 4L, backend = "cpu")
  cand <- candidate_knn(x, candidates, k = 4L, backend = "cpu", exclude_self = TRUE)

  expect_equal(cand$indices, exact$indices)
  expect_equal(cand$distances, exact$distances, tolerance = 1e-12)
})

test_that("candidate_knn ignores duplicate and invalid candidates", {
  x <- matrix(c(
    0, 0,
    1, 0,
    2, 0
  ), ncol = 2, byrow = TRUE)
  candidates <- matrix(c(
    1L, 1L, NA_integer_, 0L,
    2L, 3L, 3L, 99L,
    3L, 2L, 1L, 1L
  ), nrow = 3, byrow = TRUE)

  out <- candidate_knn(x, candidates, k = 2L, backend = "cpu")

  expect_equal(out$indices[1, ], c(1L, NA_integer_))
  expect_equal(out$distances[1, ], c(0, Inf))
  expect_equal(out$indices[2, ], c(2L, 3L))
  expect_equal(out$indices[3, ], c(3L, 2L))
})

test_that("candidate_knn supports CPU cosine candidates", {
  x <- matrix(c(
    1, 0,
    0, 1,
    1, 1
  ), ncol = 2, byrow = TRUE)
  candidates <- matrix(rep(seq_len(nrow(x)), times = nrow(x)), nrow = nrow(x), byrow = TRUE)

  out <- candidate_knn(x, candidates, k = 2L, backend = "cpu", metric = "cosine")

  expect_equal(attr(out, "metric"), "cosine")
  expect_equal(out$indices[1, ], c(1L, 3L))
  expect_equal(out$distances[1, ], c(0, 1 - 1 / sqrt(2)), tolerance = 1e-12)
})

test_that("candidate_knn canonicalizes metric aliases", {
  x <- matrix(c(
    1, 0,
    0, 1,
    1, 1
  ), ncol = 2, byrow = TRUE)
  candidates <- matrix(rep(seq_len(nrow(x)), times = nrow(x)), nrow = nrow(x), byrow = TRUE)

  out <- candidate_knn(x, candidates, k = 2L, backend = "cpu", metric = "ip")

  expect_s3_class(out, "faissR_nn")
  expect_equal(attr(out, "metric"), "inner_product")
})

test_that("candidate_knn uses strict public backend labels", {
  x <- matrix(rnorm(30), ncol = 3)
  candidates <- matrix(rep(seq_len(nrow(x)), times = nrow(x)), nrow = nrow(x), byrow = TRUE)

  auto <- candidate_knn(x, candidates, k = 2L, backend = "auto")
  expect_equal(attr(auto, "backend"), "cpu_candidate")
  expect_error(candidate_knn(x, candidates, k = 2L, backend = "a"), "backend")
  expect_error(candidate_knn(x, candidates, k = 2L, backend = "cuda_cuvs"), "backend")
})

test_that("candidate_knn validates k strictly", {
  x <- matrix(rnorm(30), ncol = 3)
  candidates <- matrix(rep(seq_len(nrow(x)), times = nrow(x)), nrow = nrow(x), byrow = TRUE)

  expect_error(candidate_knn(x, candidates, k = 2.5, backend = "cpu"), "integer in")
  expect_error(candidate_knn(x, candidates, k = c(1L, 2L), backend = "cpu"), "integer in")
  expect_error(candidate_knn(x, candidates, k = 0L, backend = "cpu"), "integer in")
  expect_error(candidate_knn(x, candidates, k = ncol(candidates) + 1L, backend = "cpu"), "integer in")
})

test_that("candidate_knn requires scalar logical exclude_self", {
  x <- matrix(rnorm(30), ncol = 3)
  candidates <- matrix(rep(seq_len(nrow(x)), times = nrow(x)), nrow = nrow(x), byrow = TRUE)

  expect_equal(faissR:::normalize_scalar_logical_arg(FALSE, "exclude_self"), FALSE)
  expect_equal(faissR:::normalize_scalar_logical_arg(TRUE, "exclude_self"), TRUE)
  expect_error(
    faissR:::normalize_scalar_logical_arg(c(TRUE, FALSE), "exclude_self"),
    "single TRUE or FALSE"
  )
  expect_error(
    faissR:::normalize_scalar_logical_arg("TRUE", "exclude_self"),
    "single TRUE or FALSE"
  )
  expect_error(
    candidate_knn(x, candidates, k = 2L, backend = "cpu", exclude_self = c(TRUE, FALSE)),
    "exclude_self"
  )
})

test_that("candidate_knn supports CPU correlation candidates", {
  x <- matrix(c(
    1, 2, 3,
    1, 3, 5,
    3, 2, 1
  ), ncol = 3, byrow = TRUE)
  candidates <- matrix(rep(seq_len(nrow(x)), times = nrow(x)), nrow = nrow(x), byrow = TRUE)

  out <- candidate_knn(x, candidates, k = 2L, backend = "cpu", metric = "correlation")

  expect_equal(attr(out, "metric"), "correlation")
  expect_equal(out$indices[1, ], c(1L, 2L))
  expect_equal(out$distances[1, ], c(0, 0), tolerance = 1e-12)
})

test_that("candidate_knn supports CPU inner-product candidates", {
  x <- matrix(c(
    2, 0,
    0, 3,
    1, 1
  ), ncol = 2, byrow = TRUE)
  candidates <- matrix(rep(seq_len(nrow(x)), times = nrow(x)), nrow = nrow(x), byrow = TRUE)

  out <- candidate_knn(x, candidates, k = 2L, backend = "cpu", metric = "inner_product")

  expect_equal(attr(out, "metric"), "inner_product")
  expect_equal(out$indices[1, ], c(1L, 3L))
  expect_equal(out$distances[1, ], c(0, 2), tolerance = 1e-12)
})

test_that("candidate_knn GPU requests do not silently fall back", {
  x <- matrix(rnorm(20), ncol = 2)
  candidates <- matrix(rep(seq_len(nrow(x)), times = nrow(x)), nrow = nrow(x), byrow = TRUE)

  if (cuda_available()) {
    expect_no_error(candidate_knn(x, candidates, k = 2L, backend = "cuda", exclude_self = TRUE))
  } else {
    expect_error(candidate_knn(x, candidates, k = 2L, backend = "cuda", exclude_self = TRUE), "CUDA")
  }
  if (!cuda_available()) {
    expect_error(
      candidate_knn(x, candidates, k = 2L, backend = "cuda", metric = "inner_product", exclude_self = TRUE),
      "CUDA"
    )
  }
})

test_that("candidate_knn CUDA supports normalized and inner-product candidates", {
  set.seed(931)
  x <- matrix(rnorm(36), nrow = 9)
  candidates <- matrix(rep(seq_len(nrow(x)), times = nrow(x)), nrow = nrow(x), byrow = TRUE)

  for (metric in c("cosine", "correlation", "inner_product")) {
    if (cuda_available()) {
      cpu <- candidate_knn(x, candidates, k = 3L, backend = "cpu", metric = metric, exclude_self = TRUE)
      cuda <- candidate_knn(x, candidates, k = 3L, backend = "cuda", metric = metric, exclude_self = TRUE)
      expect_equal(attr(cuda, "backend"), "cuda_candidate")
      expect_equal(attr(cuda, "metric"), metric)
      expect_equal(unname(cuda$indices), unname(cpu$indices))
      expect_equal(unname(cuda$distances), unname(cpu$distances), tolerance = 1e-6)
      if (metric %in% c("cosine", "correlation")) {
        expect_match(attr(cuda, "candidate_knn")$transform, "normalize")
        expect_equal(attr(cuda, "candidate_knn")$cuda_metric, "euclidean")
      } else {
        expect_true(is.na(attr(cuda, "candidate_knn")$transform))
        expect_equal(attr(cuda, "candidate_knn")$cuda_metric, "inner_product")
      }
    } else {
      expect_error(
        candidate_knn(x, candidates, k = 3L, backend = "cuda", metric = metric, exclude_self = TRUE),
        "CUDA"
      )
    }
  }
})
