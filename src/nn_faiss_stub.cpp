#include <Rcpp.h>

#include <string>

using Rcpp::List;
using Rcpp::NumericMatrix;

bool faiss_is_available_impl() {
  return false;
}

std::string faiss_info_json_impl() {
  return "{\"available\":false,\"gpu\":false,\"gpu_cagra\":false,\"reason\":\"package_not_built_with_faiss\"}";
}

List faiss_flat_knn_impl(NumericMatrix,
                         NumericMatrix,
                         int,
                         bool,
                         int) {
  Rcpp::stop(
    "FAISS backend is not available. Reinstall faissR with a FAISS C++ "
    "library visible to configure, for example "
    "FAISS_HOME=/path/to/faiss."
  );
}

List faiss_ivf_knn_impl(NumericMatrix,
                        NumericMatrix,
                        int,
                        int,
                        int,
                        bool,
                        int) {
  Rcpp::stop(
    "FAISS IVF backend is not available. Reinstall faissR with a FAISS C++ "
    "library visible to configure, for example "
    "FAISS_HOME=/path/to/faiss."
  );
}

List faiss_flat_ip_knn_impl(NumericMatrix,
                            NumericMatrix,
                            int,
                            bool,
                            int) {
  Rcpp::stop(
    "FAISS IndexFlatIP backend is not available. Reinstall faissR with "
    "FAISS_HOME=/path/to/faiss."
  );
}

List faiss_gpu_flat_knn_impl(NumericMatrix,
                             NumericMatrix,
                             int,
                             bool) {
  Rcpp::stop(
    "FAISS GPU Flat L2 backend is not available. Reinstall faissR with "
    "a FAISS GPU/cuVS library visible to configure, for example "
    "FAISS_HOME=/path/to/faiss-gpu."
  );
}

List faiss_gpu_flat_ip_knn_impl(NumericMatrix,
                                NumericMatrix,
                                int,
                                bool) {
  Rcpp::stop(
    "FAISS GPU Flat IP backend is not available. Reinstall faissR with "
    "a FAISS GPU/cuVS library visible to configure, for example "
    "FAISS_HOME=/path/to/faiss-gpu."
  );
}

List faiss_ivfpq_knn_impl(NumericMatrix,
                          NumericMatrix,
                          int,
                          int,
                          int,
                          int,
                          int,
                          bool,
                          int) {
  Rcpp::stop(
    "FAISS IndexIVFPQ backend is not available. Reinstall faissR with "
    "FAISS_HOME=/path/to/faiss."
  );
}

List faiss_hnsw_knn_impl(NumericMatrix,
                         NumericMatrix,
                         int,
                         int,
                         int,
                         int,
                         std::string,
                         std::string,
                         bool,
                         int) {
  Rcpp::stop(
    "FAISS IndexHNSWFlat backend is not available. Reinstall faissR with "
    "FAISS_HOME=/path/to/faiss."
  );
}

List faiss_nsg_knn_impl(NumericMatrix,
                        NumericMatrix,
                        int,
                        int,
                        int,
                        int,
                        bool,
                        int) {
  Rcpp::stop(
    "FAISS IndexNSGFlat backend is not available. Reinstall faissR with "
    "FAISS_HOME=/path/to/faiss."
  );
}

List faiss_nndescent_knn_impl(NumericMatrix,
                              NumericMatrix,
                              int,
                              int,
                              int,
                              int,
                              bool,
                              int) {
  Rcpp::stop(
    "FAISS IndexNNDescentFlat backend is not available. Reinstall faissR "
    "with FAISS_HOME=/path/to/faiss."
  );
}

List faiss_gpu_ivf_flat_knn_impl(NumericMatrix,
                                 NumericMatrix,
                                 int,
                                 int,
                                 int,
                                 bool) {
  Rcpp::stop(
    "FAISS GPU IVF Flat backend is not available. Reinstall faissR with "
    "a FAISS GPU/cuVS library visible to configure, for example "
    "FAISS_HOME=/path/to/faiss-gpu."
  );
}

List faiss_gpu_ivfpq_knn_impl(NumericMatrix,
                              NumericMatrix,
                              int,
                              int,
                              int,
                              int,
                              int,
                              bool) {
  Rcpp::stop(
    "FAISS GPU IVF-PQ backend is not available. Reinstall faissR with "
    "a FAISS GPU/cuVS library visible to configure, for example "
    "FAISS_HOME=/path/to/faiss-gpu."
  );
}

List faiss_kmeans_impl(NumericMatrix,
                       int,
                       int,
                       int,
                       double,
                       int,
                       int,
                       bool) {
  Rcpp::stop(
    "FAISS k-means is not available. Reinstall faissR with "
    "FAISS_HOME=/path/to/faiss."
  );
}
