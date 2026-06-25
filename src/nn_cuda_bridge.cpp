#include <Rcpp.h>

#include <string>

using Rcpp::List;
using Rcpp::IntegerMatrix;
using Rcpp::NumericMatrix;

bool cuda_is_available_impl();
std::string cuda_device_info_json_impl();
List cuda_nn_impl(NumericMatrix data,
                  NumericMatrix points,
                  int k,
                  bool square);
List cuda_nn_float32_impl(SEXP data,
                          SEXP points,
                          int k,
                          bool square);
List cuda_row_candidate_knn_impl(NumericMatrix data,
                                 IntegerMatrix candidate_indices,
                                 int k,
                                 std::string metric);
List cuda_row_candidate_knn_float32_impl(SEXP data,
                                         IntegerMatrix candidate_indices,
                                         int k,
                                         std::string metric);
List cuda_grid_self_knn_impl(NumericMatrix data,
                             int k,
                             int bins_per_dim,
                             bool include_self);

// [[Rcpp::export]]
bool cuda_available_cpp() {
  return cuda_is_available_impl();
}

// [[Rcpp::export]]
std::string cuda_device_info_json_cpp() {
  return cuda_device_info_json_impl();
}

// [[Rcpp::export]]
List nn_cuda_cpp(NumericMatrix data,
                 NumericMatrix points,
                 int k,
                 bool square) {
  return cuda_nn_impl(data, points, k, square);
}

// [[Rcpp::export]]
List nn_cuda_float32_cpp(SEXP data,
                         SEXP points,
                         int k,
                         bool square) {
  return cuda_nn_float32_impl(data, points, k, square);
}

// [[Rcpp::export]]
List row_candidate_knn_cuda_cpp(NumericMatrix data,
                                IntegerMatrix candidate_indices,
                                int k,
                                std::string metric) {
  return cuda_row_candidate_knn_impl(data, candidate_indices, k, metric);
}

// [[Rcpp::export]]
List row_candidate_knn_cuda_float32_cpp(SEXP data,
                                        IntegerMatrix candidate_indices,
                                        int k,
                                        std::string metric) {
  return cuda_row_candidate_knn_float32_impl(data, candidate_indices, k, metric);
}

// [[Rcpp::export]]
List cuda_grid_self_knn_cpp(NumericMatrix data,
                            int k,
                            int bins_per_dim,
                            bool include_self) {
  return cuda_grid_self_knn_impl(data, k, bins_per_dim, include_self);
}
