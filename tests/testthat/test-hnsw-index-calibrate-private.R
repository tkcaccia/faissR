test_that("private fitted HNSW calibration audits the full reference", {
    skip_if_not_installed("Rcpp")
    skip_if_not_installed("float")
    skip_if_not(faiss_available(), "FAISS is required")
    header <- system.file("include", "faissR_api_v1.h", package = "faissR")
    Rcpp::cppFunction(
        code = '
    Rcpp::List faissR_test_fitted_hnsw(SEXP data, SEXP query,
        Rcpp::IntegerVector ids, double target) {
      Rcpp::Shield<SEXP> index(faissR_get_hnsw_index_build_v1()(
        data, Rcpp::wrap(8), Rcpp::wrap(40), Rcpp::wrap(1)));
      return faissR_get_hnsw_index_calibrate_v1()(
        index, query, ids, Rcpp::wrap(5), Rcpp::wrap(target),
        Rcpp::wrap(5), Rcpp::wrap(1));
    }
  ',
        includes = paste0('#include "', header, '"')
    )
    set.seed(20261115)
    x <- matrix(rnorm(400L * 8L), nrow = 400L)
    rows <- seq.int(3L, 399L, by = 4L)
    for (tier in c(0.90, 0.95, 0.99)) {
        out <- faissR_test_fitted_hnsw(
            float::fl(x), float::fl(x[rows, , drop = FALSE]),
            rows, tier)
        expect_gte(out$pilot_recall, out$pilot_guard_recall)
        expect_gte(out$ef_search, 5L)
        expect_identical(out$max_ef_search, 400L)
        expect_identical(out$pilot_rows, length(rows))
        expect_identical(out$pilot_reference, "full_cpu_flat")
        expect_true(out$target_met)
    }
    expect_error(faissR_test_fitted_hnsw(
        float::fl(x), float::fl(x[rows, , drop = FALSE]),
        rows, 0.92), "target_recall must be")
    expect_error(faissR_test_fitted_hnsw(
        float::fl(x), float::fl(x[rows, , drop = FALSE]),
        rows[-1L], 0.90), "shape mismatch")
})

test_that("private HNSW calibration can search beyond efSearch 1024", {
    skip_if_not_installed("Rcpp")
    skip_if_not_installed("float")
    skip_if_not(faiss_available(), "FAISS is required")
    header <- system.file("include", "faissR_api_v1.h", package = "faissR")
    Rcpp::cppFunction(
        code = '
    Rcpp::List faissR_test_extended_hnsw(SEXP data, SEXP query,
        Rcpp::IntegerVector ids, int initial, int maximum) {
      Rcpp::Shield<SEXP> index(faissR_get_hnsw_index_build_v1()(
        data, Rcpp::wrap(8), Rcpp::wrap(40), Rcpp::wrap(1)));
      return faissR_get_hnsw_index_calibrate_v2()(
        index, query, ids, Rcpp::wrap(5), Rcpp::wrap(0.99),
        Rcpp::wrap(initial), Rcpp::wrap(maximum), Rcpp::wrap(1));
    }
  ',
        includes = paste0('#include "', header, '"')
    )
    set.seed(20261116)
    x <- matrix(rnorm(1500L * 8L), nrow = 1500L)
    rows <- seq.int(3L, 1499L, by = 17L)
    out <- faissR_test_extended_hnsw(
        float::fl(x), float::fl(x[rows, , drop = FALSE]),
        rows, 1200L, 2048L)
    expect_gt(out$ef_search, 1024L)
    expect_identical(out$max_ef_search, 1500L)
    expect_identical(out$pilot_rows, length(rows))
    expect_gte(out$pilot_recall, out$pilot_guard_recall)
    expect_error(faissR_test_extended_hnsw(
        float::fl(x), float::fl(x[rows, , drop = FALSE]),
        rows, 1200L, 1000L), "max_ef must be")
})
