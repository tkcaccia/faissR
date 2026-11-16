test_that("CUDA IVF query pilot raises probes to meet each recall tier", {
  skip_if_not(faiss_gpu_available())
  set.seed(151101)
  reference <- matrix(rnorm(100000L * 12L), nrow = 100000L)
  query <- matrix(rnorm(32L * 12L), nrow = 32L)
  k <- 10L
  exact <- lapply(seq_len(nrow(query)), function(i) {
    distance <- sqrt(rowSums((t(t(reference) - query[i, ]))^2))
    order(distance, method = "radix")[seq_len(k)]
  })
  for (tier in c(0.90, 0.95, 0.99)) {
    out <- nn(reference, query, k = k, backend = "cuda",
              method = "ivf", tuning = "pilot",
              target_recall = tier)
    recall <- mean(vapply(seq_len(nrow(query)), function(i)
      length(intersect(out$indices[i, ], exact[[i]])) / k,
      numeric(1L)))
    expect_identical(out$backend_used, "faiss_gpu_ivf_flat")
    expect_identical(attr(out, "target_recall"), tier)
    expect_true(isTRUE(out$target_met))
    expect_gte(out$pilot_recall, tier)
    expect_gte(recall, tier)
    expect_gt(out$nprobe, 9L)
    expect_identical(out$pilot_reference_backend,
                     "faiss_gpu_flat_exact")
  }
})

test_that("CUDA IVF query pilot rejects exact-reference boundary ties", {
  skip_if_not(faiss_gpu_available())
  reference <- matrix(0, nrow = 8L, ncol = 2L)
  query <- matrix(0, nrow = 2L, ncol = 2L)
  expect_error(faissR:::nn_faiss_gpu_ivf_query_pilot_cpp(
    reference, query, 1L, 2L, 1L, 0.90, "double"),
    "boundary distance ties")
})
