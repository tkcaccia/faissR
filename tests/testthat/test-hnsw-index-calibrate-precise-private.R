test_that("version-3 HNSW calibration matches double exact IDs", {
    skip_if_not_installed("Rcpp")
    skip_if_not_installed("float")
    skip_if_not(faiss_available(), "FAISS is required")
    header <- system.file("include", "faissR_api_v1.h", package = "faissR")
    Rcpp::cppFunction(
        code = '
    Rcpp::List faissR_test_precise_hnsw(SEXP data, SEXP query,
        Rcpp::IntegerVector ids) {
      Rcpp::Shield<SEXP> index(faissR_get_hnsw_index_build_v1()(
        data, Rcpp::wrap(8), Rcpp::wrap(40), Rcpp::wrap(1)));
      Rcpp::List pilot = faissR_get_hnsw_index_calibrate_v3()(
        index, query, ids, Rcpp::wrap(5), Rcpp::wrap(0.90),
        Rcpp::wrap(8), Rcpp::wrap(200), Rcpp::wrap(1));
      Rcpp::List found = faissR_get_hnsw_index_search_v1()(
        index, query, Rcpp::wrap(6), pilot["ef_search"], Rcpp::wrap(1));
      return Rcpp::List::create(Rcpp::Named("pilot") = pilot,
                                Rcpp::Named("found") = found);
    }
  ',
        includes = paste0('#include "', header, '"')
    )
    set.seed(20261117)
    x <- matrix(rnorm(400L * 8L), nrow = 400L)
    xf <- float::fl(x)
    x <- float::dbl(xf)
    rows <- seq.int(3L, 399L, by = 4L)
    out <- faissR_test_precise_hnsw(
        xf, float::fl(x[rows, , drop = FALSE]), rows)
    expect_identical(out$pilot$pilot_reference, "full_cpu_double_l2")
    expect_identical(out$pilot$pilot_rows, length(rows))
    expect_gte(out$pilot$pilot_recall, out$pilot$pilot_guard_recall)
    expect_true(out$pilot$pilot_near_boundary_rows >= 0L)
    actual <- out$found$indices
    recall <- vapply(seq_along(rows), function(i) {
        distances <- rowSums((t(t(x) - x[rows[i], ]))^2)
        distances[rows[i]] <- Inf
        wanted <- order(distances, seq_len(nrow(x)))[seq_len(5L)]
        found <- actual[i, actual[i, ] != rows[i]][seq_len(5L)]
        length(intersect(wanted, found)) / 5
    }, numeric(1L))
    expect_equal(out$pilot$pilot_recall, mean(recall), tolerance = 1e-12)
    query_error <- sd(recall) / sqrt(length(recall))
    nominal_error <- sqrt(0.9 * 0.1 / (length(recall) * 5))
    expect_equal(out$pilot$pilot_query_recall_se, query_error, tolerance = 1e-8)
    expect_equal(out$pilot$pilot_guard_recall,
        min(1, 0.9 + 2.58 * max(nominal_error, query_error)),
        tolerance = 1e-8)
    outside_shard <- faissR_test_precise_hnsw(
        xf, float::fl(x[rows, , drop = FALSE]),
        rep.int(0L, length(rows)))
    expect_identical(outside_shard$pilot$pilot_reference,
                     "full_cpu_double_l2")
    expect_gte(outside_shard$pilot$pilot_recall,
               outside_shard$pilot$pilot_guard_recall)
})

test_that("version-3 calibration detects a near-boundary row", {
    skip_if_not_installed("Rcpp")
    skip_if_not_installed("float")
    skip_if_not(faiss_available(), "FAISS is required")
    header <- system.file("include", "faissR_api_v1.h", package = "faissR")
    Rcpp::cppFunction(
        code = '
    Rcpp::List faissR_test_precise_tie(SEXP data, SEXP query,
        Rcpp::IntegerVector ids) {
      Rcpp::Shield<SEXP> index(faissR_get_hnsw_index_build_v1()(
        data, Rcpp::wrap(8), Rcpp::wrap(40), Rcpp::wrap(1)));
      return faissR_get_hnsw_index_calibrate_v3()(
        index, query, ids, Rcpp::wrap(1), Rcpp::wrap(0.90),
        Rcpp::wrap(8), Rcpp::wrap(200), Rcpp::wrap(1));
    }
  ',
        includes = paste0('#include "', header, '"')
    )
    set.seed(20261117)
    distant <- matrix(runif(400L * 2L, 10, 20), ncol = 2L)
    x <- rbind(c(0, 0), c(1, 0.0001), c(1, 0), distant)
    rows <- c(1L, seq.int(4L, 400L, by = 4L))
    pilot <- faissR_test_precise_tie(
        float::fl(x), float::fl(x[rows, , drop = FALSE]), rows)
    expect_identical(pilot$pilot_reference, "full_cpu_double_l2")
    expect_gte(pilot$pilot_near_boundary_rows, 1L)
    expect_gte(pilot$pilot_recall, pilot$pilot_guard_recall)
})

test_that("file HNSW requests broader pilots for large high-dimensional data", {
    tune <- function(n, p, target) {
        faissR:::nn_tune_faiss_hnsw_cpp(n, p, 30L, "euclidean",
            target)$file_pilot_offsets
    }
    expect_identical(tune(1281167L, 1024L, 0.99), c(0.125, 0.25, 0.625, 0.875))
    expect_identical(tune(10000000L, 96L, 0.99), 0.25)
    expect_identical(tune(1281167L, 1024L, 0.95), 0.25)
})
