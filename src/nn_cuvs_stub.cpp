#include <Rcpp.h>

#include <cstdint>
#include <string>

using Rcpp::List;
using Rcpp::NumericMatrix;

bool cuvs_is_available_impl() {
  return false;
}

std::string cuvs_info_json_impl() {
  return "{\"available\":false,\"reason\":\"package_not_built_with_cuvs\"}";
}

List cuvs_bruteforce_knn_impl(NumericMatrix,
                              NumericMatrix,
                              int,
                              bool) {
  Rcpp::stop(
    "cuVS backend is not available. Reinstall faissR with RAPIDS cuVS "
    "visible to configure, for example FAISSR_USE_CUVS=1 and "
    "CUVS_HOME=/path/to/cuvs."
  );
}

List cuvs_cagra_knn_impl(NumericMatrix,
                         NumericMatrix,
                         int,
                         bool,
                         int,
                         int,
                         int,
                         int,
                         std::string) {
  Rcpp::stop(
    "cuVS CAGRA backend is not available. Reinstall faissR with RAPIDS "
    "cuVS visible to configure, for example FAISSR_USE_CUVS=1 and "
    "CUVS_HOME=/path/to/cuvs."
  );
}

List cuvs_nndescent_self_knn_impl(NumericMatrix,
                                  int,
                                  int,
                                  int,
                                  int) {
  Rcpp::stop(
    "cuVS NN-descent backend is not available. Reinstall faissR with "
    "RAPIDS cuVS visible to configure, for example FAISSR_USE_CUVS=1 "
    "and CUVS_HOME=/path/to/cuvs."
  );
}

List cuvs_hnsw_knn_impl(NumericMatrix,
                        NumericMatrix,
                        int,
                        bool,
                        int,
                        int,
                        int,
                        int,
                        std::string) {
  Rcpp::stop(
    "cuVS HNSW backend is not available. Reinstall faissR with RAPIDS "
    "cuVS visible to configure, for example FAISSR_USE_CUVS=1 and "
    "CUVS_HOME=/path/to/cuvs."
  );
}

List cuvs_hnsw_float32_knn_impl(SEXP,
                                SEXP,
                                int,
                                bool,
                                int,
                                int,
                                int,
                                int,
                                std::string) {
  Rcpp::stop(
    "cuVS HNSW backend is not available. Reinstall faissR with RAPIDS "
    "cuVS visible to configure, for example FAISSR_USE_CUVS=1 and "
    "CUVS_HOME=/path/to/cuvs."
  );
}

List cuvs_ivf_flat_knn_impl(NumericMatrix,
                            NumericMatrix,
                            int,
                            int,
                            int,
                            bool) {
  Rcpp::stop(
    "Direct cuVS IVF-Flat backend is not available. Reinstall faissR with "
    "RAPIDS cuVS visible to configure, for example FAISSR_USE_CUVS=1 "
    "and CUVS_HOME=/path/to/cuvs."
  );
}

List cuvs_ivf_pq_knn_impl(NumericMatrix,
                          NumericMatrix,
                          int,
                          int,
                          int,
                          int,
                          int,
                          bool) {
  Rcpp::stop(
    "Direct cuVS IVF-PQ backend is not available. Reinstall faissR with "
    "RAPIDS cuVS visible to configure, for example FAISSR_USE_CUVS=1 "
    "and CUVS_HOME=/path/to/cuvs. IVF-PQ is explicit-only and intended for "
    "GPU memory pressure."
  );
}

List cuvs_kmeans_impl(NumericMatrix,
                      int,
                      int,
                      int,
                      double,
                      int64_t,
                      bool) {
  Rcpp::stop(
    "cuVS k-means backend is not available. Reinstall faissR with RAPIDS "
    "cuVS visible to configure, for example FAISSR_USE_CUVS=1 and "
    "CUVS_HOME=/path/to/cuvs."
  );
}
