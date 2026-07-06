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
List cuda_nn_float32_gpu_impl(SEXP data,
                              SEXP points,
                              int k,
                              bool exclude_self,
                              std::string metric,
                              std::string backend_used,
                              std::string method);
bool faiss_gpu_bfknn_float32_gpu_available_impl();
List faiss_gpu_bfknn_float32_gpu_impl(SEXP data,
                                      SEXP points,
                                      int k,
                                      bool exclude_self,
                                      std::string metric,
                                      std::string backend_used,
                                      std::string method);
List cuda_gpu_knn_to_host_impl(SEXP result);
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
List nn_cuda_float32_gpu_cpp(SEXP data,
                             SEXP points,
                             int k,
                             bool exclude_self,
                             std::string metric,
                             std::string backend_used,
                             std::string method) {
  return cuda_nn_float32_gpu_impl(
    data,
    points,
    k,
    exclude_self,
    metric,
    backend_used,
    method
  );
}

// [[Rcpp::export]]
List gpu_knn_to_host_cpp(SEXP result) {
  return cuda_gpu_knn_to_host_impl(result);
}

extern "C" SEXP faissR_nn_cuda_tuned_gpu_call(SEXP x,
                                              SEXP k,
                                              SEXP method,
                                              SEXP metric,
                                              SEXP include_self,
                                              SEXP target_recall) {
  BEGIN_RCPP
  const int kk = Rcpp::as<int>(k);
  const std::string method_value = Rcpp::as<std::string>(method);
  const std::string metric_value = Rcpp::as<std::string>(metric);
  const bool include_self_value = Rcpp::as<bool>(include_self);
  const double target_recall_value = Rcpp::as<double>(target_recall);
  if (method_value != "auto" &&
      method_value != "exact" &&
      method_value != "flat" &&
      method_value != "bruteforce") {
    Rcpp::stop(
      "faissR_nn_cuda_tuned_gpu_call currently keeps results on GPU only for "
      "CUDA exact/flat/bruteforce routes. Use nn() for host output or add a "
      "provider-specific GPU-result route for this method."
    );
  }
  const std::string resolved_method =
    method_value == "auto" ? "exact" : method_value;
  Rcpp::List out;
  if ((metric_value == "euclidean" || metric_value == "inner_product") &&
      faiss_gpu_bfknn_float32_gpu_available_impl()) {
    const std::string backend_used = metric_value == "inner_product" ?
      "faiss_gpu_flat_ip" : "faiss_gpu_bfknn_l2";
    out = faiss_gpu_bfknn_float32_gpu_impl(
      x,
      x,
      kk,
      !include_self_value,
      metric_value,
      backend_used,
      resolved_method
    );
  } else {
    out = cuda_nn_float32_gpu_impl(
      x,
      x,
      kk,
      !include_self_value,
      metric_value,
      "cuda_native_exact_gpu",
      resolved_method
    );
  }
  out["target_recall"] = target_recall_value;
  out["tuning"] = "auto";
  out["tuning_note"] =
    "exact GPU-resident route; target_recall is recorded but recall is exact by construction";
  out.attr("target_recall") = target_recall_value;
  return out;
  END_RCPP
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
