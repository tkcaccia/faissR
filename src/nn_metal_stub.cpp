#include <Rcpp.h>

#include <string>

bool faissr_metal_knn_available_impl() {
  return false;
}

Rcpp::List faissr_metal_knn_impl(SEXP,
                                  int,
                                  const std::string&,
                                  const std::string&,
                                  double) {
  Rcpp::stop("Metal KNN requires a Metal-enabled macOS faissR build.");
}

Rcpp::List faissr_metal_query_knn_impl(SEXP,
                                        SEXP,
                                        int,
                                        const std::string&,
                                        const std::string&,
                                        double) {
  Rcpp::stop("Metal query KNN requires a Metal-enabled macOS faissR build.");
}
