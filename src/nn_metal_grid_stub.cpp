#include <Rcpp.h>

bool metal_grid_available_impl() {
  return false;
}

Rcpp::List metal_grid_self_knn_impl(SEXP,
                                    int,
                                    int,
                                    bool) {
  Rcpp::stop(
    "Metal grid nearest-neighbour search is available only on macOS with a Metal device."
  );
}
