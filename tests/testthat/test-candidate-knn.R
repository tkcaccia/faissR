test_that("candidate_knn matches exact self-KNN with full candidate rows", {
  set.seed(301)
  x <- matrix(rnorm(30 * 4), nrow = 30)
  candidates <- matrix(rep(seq_len(nrow(x)), times = nrow(x)), nrow = nrow(x), byrow = TRUE)
  exact <- nn(x, k = 5L, backend = "cpu", n_threads = 1L)
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

  out <- candidate_knn(data, candidates, points = points, k = 2L, backend = "cpu", n_threads = 1L)

  expect_equal(out$indices[1, ], c(1L, 2L))
  expect_equal(out$indices[2, ], c(4L, 3L))
  expect_equal(out$distances[1, ], c(0.2, 0.8), tolerance = 1e-12)
  expect_equal(out$distances[2, ], c(0.2, 3.8), tolerance = 1e-12)
})

test_that("candidate_knn exclude_self matches exact CPU without self", {
  set.seed(302)
  x <- matrix(rnorm(25 * 3), nrow = 25)
  candidates <- matrix(rep(seq_len(nrow(x)), times = nrow(x)), nrow = nrow(x), byrow = TRUE)

  exact <- faissR:::nn(exclude_self = TRUE, x, k = 4L, backend = "cpu", n_threads = 1L)
  cand <- candidate_knn(x, candidates, k = 4L, backend = "cpu", exclude_self = TRUE, n_threads = 1L)

  expect_equal(cand$indices, exact$indices)
  expect_equal(cand$distances, exact$distances, tolerance = 1e-12)
  expect_true(isTRUE(attr(cand, "exclude_self")))
  expect_true(isTRUE(summary(cand)$exclude_self))
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

  out <- candidate_knn(x, candidates, k = 2L, backend = "cpu", n_threads = 1L)

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

  out <- candidate_knn(x, candidates, k = 2L, backend = "cpu", metric = "cosine", n_threads = 1L)

  expect_equal(attr(out, "metric"), "cosine")
  expect_equal(out$indices[1, ], c(1L, 3L))
  expect_equal(out$distances[1, ], c(0, 1 - 1 / sqrt(2)), tolerance = 1e-12)
})
test_that("candidate_knn uses strict public backend labels", {
  x <- matrix(rnorm(30), ncol = 3)
  candidates <- matrix(rep(seq_len(nrow(x)), times = nrow(x)), nrow = nrow(x), byrow = TRUE)

  expect_error(candidate_knn(x, candidates, k = 2L, backend = "auto", n_threads = 1L), "backend")
  expect_error(candidate_knn(x, candidates, k = 2L, backend = "a", n_threads = 1L), "backend")
  expect_error(candidate_knn(x, candidates, k = 2L, backend = "cuda_cuvs", n_threads = 1L), "backend")
})

test_that("candidate_knn validates k strictly", {
  x <- matrix(rnorm(30), ncol = 3)
  candidates <- matrix(rep(seq_len(nrow(x)), times = nrow(x)), nrow = nrow(x), byrow = TRUE)

  expect_error(candidate_knn(x, candidates, k = 2.5, backend = "cpu", n_threads = 1L), "integer in")
  expect_error(candidate_knn(x, candidates, k = c(1L, 2L), backend = "cpu", n_threads = 1L), "integer in")
  expect_error(candidate_knn(x, candidates, k = 0L, backend = "cpu", n_threads = 1L), "integer in")
  expect_error(candidate_knn(x, candidates, k = ncol(candidates) + 1L, backend = "cpu", n_threads = 1L), "integer in")
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
    candidate_knn(x, candidates, k = 2L, backend = "cpu", exclude_self = c(TRUE, FALSE), n_threads = 1L),
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

  out <- candidate_knn(x, candidates, k = 2L, backend = "cpu", metric = "correlation", n_threads = 1L)

  expect_equal(attr(out, "metric"), "correlation")
  expect_equal(out$indices[1, ], c(1L, 2L))
  expect_equal(out$distances[1, ], c(0, 0), tolerance = 1e-12)
})
