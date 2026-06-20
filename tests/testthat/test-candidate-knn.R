test_that("candidate_knn matches exact self-KNN with full candidate rows", {
  set.seed(301)
  x <- matrix(rnorm(30 * 4), nrow = 30)
  candidates <- matrix(rep(seq_len(nrow(x)), times = nrow(x)), nrow = nrow(x), byrow = TRUE)

  exact <- nn(x, k = 5L, backend = "cpu")
  cand <- candidate_knn(x, candidates, k = 5L, backend = "cpu", n_threads = 2L)

  expect_s3_class(cand, "fastEmbedR_nn")
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

  exact <- faissR:::nn_without_self(x, k = 4L, backend = "cpu")
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

test_that("candidate_knn GPU requests do not silently fall back", {
  x <- matrix(rnorm(20), ncol = 2)
  candidates <- matrix(rep(seq_len(nrow(x)), times = nrow(x)), nrow = nrow(x), byrow = TRUE)

  if (cuda_available()) {
    expect_no_error(candidate_knn(x, candidates, k = 2L, backend = "cuda", exclude_self = TRUE))
  } else {
    expect_error(candidate_knn(x, candidates, k = 2L, backend = "cuda", exclude_self = TRUE), "CUDA")
  }
})
