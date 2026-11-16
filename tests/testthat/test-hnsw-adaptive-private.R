test_that("private HNSW v2 audits a full-reference pilot", {
    skip_if_not_installed("Rcpp")
    skip_if_not(faiss_available(), "FAISS is required for HNSW")
    header <- system.file("include", "faissR_api_v1.h", package = "faissR")
    Rcpp::cppFunction(
        code = '
    SEXP faissR_test_hnsw_adaptive(SEXP data, double target) {
      faissR_hnsw_search_v2_fun fn = faissR_get_hnsw_search_v2();
      Rcpp::IntegerVector dims = Rcpp::as<Rcpp::IntegerVector>(
        Rf_getAttrib(data, R_DimSymbol));
      Rcpp::Shield<SEXP> n(Rcpp::wrap(dims[0]));
      Rcpp::Shield<SEXP> p(Rcpp::wrap(dims[1]));
      Rcpp::Shield<SEXP> k(Rcpp::wrap(5));
      Rcpp::Shield<SEXP> recall(Rcpp::wrap(target));
      Rcpp::Shield<SEXP> threads(Rcpp::wrap(2));
      Rcpp::Shield<SEXP> storage(Rcpp::wrap("double"));
      return fn(data, R_NilValue, n, p, k, recall, threads, storage);
    }
  ',
        includes = paste0('#include "', header, '"')
    )
    set.seed(20261122L)
    x <- matrix(rnorm(400L * 8L), nrow = 400L)
    for (tier in c(0.90, 0.95, 0.99)) {
        out <- faissR_test_hnsw_adaptive(x, tier)
        expect_identical(out$index_type, "IndexHNSWFlat")
        expect_false(out$exact)
        expect_true(out$recall_audited)
        expect_true(out$target_met)
        expect_equal(out$target_recall, tier)
        expect_gte(out$pilot_recall, out$pilot_guard_recall)
        expect_identical(out$pilot_reference, "full_cpu_flat")
        expect_length(unique(out$pilot_row_ids), 256L)
        expect_identical(dim(out$indices), c(400L, 5L))
        expect_false(any(out$indices == row(out$indices)))
        expect_true(all(is.finite(out$distances)))
    }
    expect_error(faissR_test_hnsw_adaptive(x, 0.92),
        "target_recall must be")
})
