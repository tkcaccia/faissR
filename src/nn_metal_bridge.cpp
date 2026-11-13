#include <Rcpp.h>

#include <string>

bool faissr_metal_knn_available_impl();
Rcpp::List faissr_metal_knn_impl(SEXP data,
                                  int k,
                                  const std::string& method,
                                  const std::string& metric,
                                  double target_recall);
Rcpp::List faissr_metal_query_knn_impl(SEXP data,
                                        SEXP query,
                                        int k,
                                        const std::string& method,
                                        const std::string& metric,
                                        double target_recall);

namespace {

Rcpp::List finish_metal_result(Rcpp::List out) {
  const std::string method = Rcpp::as<std::string>(out["method"]);
  const bool exact = method.find("exact") != std::string::npos;
  out["backend_used"] = "faissr_" + method;
  out["accelerator"] = "metal";
  out["gpu_count"] = 1;
  out["index_base"] = 1;
  out["result_residency"] = "host";
  out["exact"] = exact;
  if (exact) out["target_met"] = true;
  return out;
}

} // namespace

// [[Rcpp::export]]
bool nn_metal_available_cpp() {
  return faissr_metal_knn_available_impl();
}

// [[Rcpp::export]]
Rcpp::List nn_metal_cpp(SEXP data,
                         int k,
                         std::string method = "auto",
                         std::string metric = "euclidean",
                         double target_recall = 0.99) {
  return finish_metal_result(faissr_metal_knn_impl(
    data, k, method, metric, target_recall
  ));
}

// [[Rcpp::export]]
Rcpp::List nn_metal_query_cpp(SEXP data,
                               SEXP query,
                               int k,
                               std::string method = "auto",
                               std::string metric = "euclidean",
                               double target_recall = 0.99) {
  return finish_metal_result(faissr_metal_query_knn_impl(
    data, query, k, method, metric, target_recall
  ));
}

extern "C" SEXP faissR_metal_knn_v1(SEXP data,
                                    SEXP k,
                                    SEXP method,
                                    SEXP metric,
                                    SEXP target_recall) {
  BEGIN_RCPP
  return finish_metal_result(faissr_metal_knn_impl(
    data, Rcpp::as<int>(k), Rcpp::as<std::string>(method),
    Rcpp::as<std::string>(metric), Rcpp::as<double>(target_recall)
  ));
  END_RCPP
}

extern "C" SEXP faissR_metal_query_knn_v1(SEXP data,
                                          SEXP query,
                                          SEXP k,
                                          SEXP method,
                                          SEXP metric,
                                          SEXP target_recall) {
  BEGIN_RCPP
  return finish_metal_result(faissr_metal_query_knn_impl(
    data, query, Rcpp::as<int>(k), Rcpp::as<std::string>(method),
    Rcpp::as<std::string>(metric), Rcpp::as<double>(target_recall)
  ));
  END_RCPP
}
