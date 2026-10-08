test_that("HNSW tuning grids are bounded and include the static baseline", {
    baseline <- list(m = 16L, ef_construction = 100L, ef_search = 60L)
    build <- faissR:::hnsw_tune_default_build_grid(baseline)
    search <- faissR:::hnsw_tune_default_search(baseline)

    expect_lte(nrow(build), 3L)
    expect_true(any(build$m == 16L & build$ef_construction == 100L))
    expect_true(60L %in% search)
    expect_true(480L %in% search)
    expect_true(all(build$m > 0L))
    expect_true(all(build$ef_construction > 0L))
})

test_that("HNSW tuning validation rejects unsupported or unbounded requests", {
    skip_if_not_installed("float")
    skip_if_not(faiss_available())
    x <- matrix(stats::rnorm(300), nrow = 50L)

    expect_error(
        hnsw_tune(
            x,
            k = 5L,
            metric = "cosine",
            reference_size = 50L,
            tune_queries = 10L,
            holdout_queries = 10L
        ),
        "euclidean"
    )
    expect_error(
        hnsw_tune(
            x,
            k = 5L,
            reference_size = 50L,
            tune_queries = 10L,
            holdout_queries = 10L,
            build_grid = data.frame(
                m = c(4L, 8L),
                ef_construction = c(20L, 40L)
            ),
            max_builds = 1L
        ),
        "max_builds"
    )
})

test_that("data-aware HNSW tuning is deterministic and reuses its index", {
    skip_if_not_installed("float")
    skip_if_not(faiss_available())
    set.seed(20261008L)
    x <- matrix(stats::rnorm(180L * 6L), nrow = 180L, ncol = 6L)
    args <- list(
        data = x,
        k = 5L,
        target_recall = 0.9,
        reference_size = 180L,
        tune_queries = 30L,
        holdout_queries = 30L,
        seed = 19L,
        build_grid = data.frame(m = 12L, ef_construction = 80L),
        ef_search = 80L,
        safety_margin = 0,
        bootstrap_repetitions = 40L,
        timing_repetitions = 2L,
        n_threads = 1L,
        time_limit = 60,
        build_index = TRUE
    )

    first <- do.call(hnsw_tune, args)
    second <- do.call(hnsw_tune, args)

    expect_s3_class(first, "faissR_hnsw_tuning")
    expect_true(is.call(first$call))
    expect_identical(first$sampling$query_data_rows,
        second$sampling$query_data_rows)
    expect_equal(first$diagnostics$mean_recall,
        second$diagnostics$mean_recall)
    expect_equal(first$selected$m, 12L)
    expect_match(first$selected$selection_rule,
        "minimum_median_projected")
    expect_true(first$selected$projected_workload_seconds >= 0)
    expect_true(first$selected$holdout_confirmed)
    expect_identical(first$status, "confirmed_index_built")
    expect_type(first$index, "externalptr")
    expect_false(first$provenance$fallback_allowed)
    expect_identical(first$provenance$index_base, 1L)
    expect_identical(first$sampling$reference_sampling, "full_reference")
    expect_false(first$sampling$prefix_reference)
    expect_equal(first$workload$query_rows_per_batch, nrow(x))
    expect_equal(first$workload$expected_query_batches, 1L)
    expect_identical(first$workload$timing_statistic, "median")
    expect_true(first$tuning_exact_seconds >= 0)
    expect_true(first$tuning_build_seconds >= 0)
    expect_true(first$tuning_search_seconds >= 0)
    expect_true(first$tuning_total_seconds >= 0)
    expect_gte(first$workflow_total_seconds, first$tuning_total_seconds)

    result <- predict(first, x[1:8, , drop = FALSE], k = 4L)
    expect_s3_class(result, "faissR_nn")
    expect_equal(dim(result$indices), c(8L, 4L))
    expect_true(all(result$indices >= 1L & result$indices <= nrow(x)))
    expect_true(attr(result, "approximation")$index_reused)
    expect_true(attr(result, "approximation")$no_fallback)
    expect_true(attr(result, "final_search_seconds") >= 0)
    expect_gte(attr(result, "adaptive_end_to_end_seconds"),
        first$workflow_total_seconds)
    expect_equal(attr(result, "hnsw_tuning")$selected$m, 12L)
})

test_that("HNSW tuning does not deploy an incomplete pilot", {
    diagnostics <- data.frame(
        split = c("tune", "holdout"), status = "measured",
        lower_95_recall = c(1, 1)
    )
    selected <- faissR:::hnsw_tune_select(
        list(target_recall = 0.9, safety_margin = 0),
        list(complete = FALSE, diagnostics = diagnostics)
    )
    fit <- faissR:::hnsw_tune_final_index(list(), selected)

    expect_null(selected)
    expect_null(fit$index)
    expect_identical(fit$status, "no_confirmed_candidate")
    expect_match(faissR:::hnsw_tune_guidance(fit$status),
        "broader `ef_search`")
})

test_that("HNSW tuning selects the fastest recall-qualified workload", {
    diagnostics <- data.frame(
        split = rep(c("tune", "holdout"), each = 2L),
        status = "measured",
        m = rep(c(8L, 16L), 2L),
        ef_construction = rep(c(40L, 80L), 2L),
        ef_search = rep(c(40L, 80L), 2L),
        lower_95_recall = c(0.96, 0.97, 0.95, 0.96),
        projected_workload_seconds = rep(c(2, 1), 2L),
        projected_query_batch_seconds = rep(c(1.5, 0.6), 2L),
        projected_build_seconds = rep(c(0.5, 0.4), 2L)
    )
    selected <- faissR:::hnsw_tune_select(
        list(target_recall = 0.95, safety_margin = 0),
        list(complete = TRUE, diagnostics = diagnostics)
    )

    expect_equal(selected$m, 16L)
    expect_equal(selected$projected_workload_seconds, 1)
    expect_true(selected$holdout_confirmed)
    expect_match(selected$selection_rule, "projected_build")
})
