test_that("private HNSW index survives a native FAISS file round trip", {
    skip_if_not_installed("Rcpp")
    skip_if_not_installed("float")
    skip_if_not(faiss_available(), "FAISS is required")
    header <- system.file("include", "faissR_api_v1.h", package = "faissR")
    Rcpp::cppFunction(
        code = '
    Rcpp::List faissR_test_hnsw_persistence(SEXP data, SEXP query,
                                             std::string path) {
      Rcpp::Shield<SEXP> index(faissR_get_hnsw_index_build_v1()(
        data, Rcpp::wrap(8), Rcpp::wrap(40), Rcpp::wrap(2)));
      Rcpp::List before(faissR_get_hnsw_index_search_v1()(
        index, query, Rcpp::wrap(5), Rcpp::wrap(80), Rcpp::wrap(2)));
      faissR_get_hnsw_index_save_v1()(index, Rcpp::wrap(path));
      Rcpp::Shield<SEXP> loaded(faissR_get_hnsw_index_load_v1()(
        Rcpp::wrap(path), Rcpp::wrap(2)));
      Rcpp::List after(faissR_get_hnsw_index_search_v1()(
        loaded, query, Rcpp::wrap(5), Rcpp::wrap(80), Rcpp::wrap(2)));
      return Rcpp::List::create(
        Rcpp::Named("before") = before,
        Rcpp::Named("after") = after,
        Rcpp::Named("n") = Rf_getAttrib(loaded, Rf_install("n")),
        Rcpp::Named("p") = Rf_getAttrib(loaded, Rf_install("p")),
        Rcpp::Named("m") = Rf_getAttrib(loaded, Rf_install("m")));
    }
  ',
        includes = paste0('#include "', header, '"')
    )
    set.seed(20261115)
    x <- float::fl(matrix(rnorm(400L * 8L), nrow = 400L))
    query <- float::fl(as.matrix(x)[seq.int(5L, 395L, by = 13L), , drop = FALSE])
    path <- tempfile(fileext = ".faiss")
    on.exit(unlink(path), add = TRUE)
    out <- faissR_test_hnsw_persistence(x, query, path)
    expect_gt(file.info(path)$size, 0)
    expect_identical(out$n, 400L)
    expect_identical(out$p, 8L)
    expect_identical(out$m, 8L)
    expect_identical(out$after$indices, out$before$indices)
    expect_equal(out$after$distances, out$before$distances, tolerance = 1e-7)
    expect_identical(out$after$backend_used, out$before$backend_used)
    expect_error(faissR_test_hnsw_persistence(
        x, query, file.path(tempdir(), "missing", "index.faiss")),
        "save failed")
})
