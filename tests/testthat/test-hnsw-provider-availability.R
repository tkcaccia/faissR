test_that("CPU HNSW reports a missing FAISS provider without legacy dispatch", {
    testthat::local_mocked_bindings(
        faiss_available = function() FALSE,
        .package = "faissR"
    )
    x <- matrix(seq_len(120), nrow = 20)
    for (metric in c("euclidean", "cosine", "correlation")) {
        expect_identical(
            faissR:::resolve_public_nn_backend("cpu", "hnsw", metric),
            "faiss_hnsw"
        )
        expect_error(
            nn(x, k = 3, backend = "cpu", method = "hnsw",
                metric = metric, tuning = "none", n_threads = 1L),
            "FAISS.*not.*available"
        )
    }

    caps <- subset(nn_capabilities(runtime = TRUE),
        backend == "cpu" & method == "hnsw")
    expect_true(all(caps$supported))
    expect_true(all(caps$resolved_backend == "faiss_hnsw"))
    expect_false(any(caps$runtime_available))
    expect_true(all(caps$runtime_reason == "missing_faiss"))

    expect_no_error(nn(x, k = 3, backend = "cpu", method = "exact", n_threads = 1L))
})
