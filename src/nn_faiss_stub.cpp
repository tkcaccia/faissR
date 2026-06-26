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

List faiss_flat_float32_knn_impl(SEXP,
                                 SEXP,
                                 int,
                                 bool,
                                 int,
                                 std::string,
                                 std::string) {
  Rcpp::stop(
    "FAISS float32 Flat backend is not available. Reinstall faissR with "
    "a FAISS C++ library visible to configure, for example "
    "FAISS_HOME=/path/to/faiss."
  );
}

List faiss_ivf_knn_impl(NumericMatrix,
                        NumericMatrix,
                        int,
                        int,
                        int,
                        std::string,
                        std::string,
                        bool,
                        int) {
  Rcpp::stop(
    "FAISS IVF backend is not available. Reinstall faissR with a FAISS C++ "
    "library visible to configure, for example "
    "FAISS_HOME=/path/to/faiss."
  );
}

List faiss_ivf_float32_knn_impl(SEXP,
                                SEXP,
                                int,
                                int,
                                int,
                                std::string,
                                std::string,
                                bool,
                                int,
                                std::string) {
  Rcpp::stop(
    "FAISS float32 IVF backend is not available. Reinstall faissR with "
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

List faiss_flat_normalized_ip_distance_knn_impl(NumericMatrix,
                                                NumericMatrix,
                                                int,
                                                bool,
                                                int) {
  Rcpp::stop(
    "FAISS normalized Flat backend is not available. Reinstall faissR with "
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

List faiss_gpu_flat_float32_knn_impl(SEXP,
                                     SEXP,
                                     int,
                                     bool,
                                     std::string,
                                     std::string,
                                     std::string) {
  Rcpp::stop(
    "FAISS GPU float32 Flat backend is not available. Reinstall faissR with "
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

List faiss_gpu_flat_normalized_ip_distance_knn_impl(NumericMatrix,
                                                    NumericMatrix,
                                                    int,
                                                    bool) {
  Rcpp::stop(
    "FAISS GPU normalized Flat backend is not available. Reinstall faissR "
    "with a FAISS GPU/cuVS library visible to configure, for example "
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
                          std::string,
                          std::string,
                          bool,
                          int) {
  Rcpp::stop(
    "FAISS IndexIVFPQ backend is not available. Reinstall faissR with "
    "FAISS_HOME=/path/to/faiss."
  );
}

List faiss_ivfpq_float32_knn_impl(SEXP,
                                  SEXP,
                                  int,
                                  int,
                                  int,
                                  int,
                                  int,
                                  std::string,
                                  std::string,
                                  bool,
                                  int,
                                  std::string) {
  Rcpp::stop(
    "FAISS float32 IVF-PQ backend is not available. Reinstall faissR with "
    "FAISS_HOME=/path/to/faiss."
  );
}

List faiss_ivfpq_fastscan_knn_impl(NumericMatrix,
                          NumericMatrix,
                          int,
                          int,
                          int,
                          int,
                          int,
                          int,
                          bool,
                          int) {
  Rcpp::stop(
    "FAISS IVFPQ FastScan backend is not available. Reinstall "
    "faissR with FAISS_HOME pointing to a FAISS build that provides "
    "faiss/IndexIVFPQFastScan.h."
  );
}

List faiss_ivfpq_fastscan_float32_knn_impl(SEXP,
                                  SEXP,
                                  int,
                                  int,
                                  int,
                                  int,
                                  int,
                                  int,
                                  bool,
                                  int,
                                  std::string) {
  Rcpp::stop(
    "FAISS float32 IVFPQ FastScan backend is not available. "
    "Reinstall faissR with FAISS_HOME pointing to a FAISS build that "
    "provides faiss/IndexIVFPQFastScan.h."
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

List faiss_hnsw_float32_knn_impl(SEXP,
                                 SEXP,
                                 int,
                                 int,
                                 int,
                                 int,
                                 std::string,
                                 std::string,
                                 bool,
                                 int,
                                 std::string) {
  Rcpp::stop(
    "FAISS float32 HNSW backend is not available. Reinstall faissR with "
    "FAISS_HOME=/path/to/faiss."
  );
}

SEXP faiss_hnsw_index_build_float32_impl(SEXP,
                                         int,
                                         int,
                                         int,
                                         std::string,
                                         std::string,
                                         int) {
  Rcpp::stop(
    "FAISS float32 HNSW fitted-index backend is not available. Reinstall "
    "faissR with FAISS_HOME=/path/to/faiss."
  );
}

List faiss_hnsw_index_search_float32_impl(SEXP,
                                          SEXP,
                                          int,
                                          bool,
                                          int,
                                          int,
                                          std::string) {
  Rcpp::stop(
    "FAISS float32 HNSW fitted-index backend is not available. Reinstall "
    "faissR with FAISS_HOME=/path/to/faiss."
  );
}

SEXP faiss_index_build_float32_impl(SEXP,
                                    std::string,
                                    int,
                                    int,
                                    int,
                                    int,
                                    int,
                                    int,
                                    int,
                                    int,
                                    std::string,
                                    std::string,
                                    int) {
  Rcpp::stop(
    "FAISS float32 fitted-index backend is not available. Reinstall "
    "faissR with FAISS_HOME=/path/to/faiss."
  );
}

List faiss_index_search_float32_impl(SEXP,
                                     SEXP,
                                     int,
                                     bool,
                                     int,
                                     int,
                                     std::string) {
  Rcpp::stop(
    "FAISS float32 fitted-index backend is not available. Reinstall "
    "faissR with FAISS_HOME=/path/to/faiss."
  );
}

List faiss_nsg_knn_impl(NumericMatrix,
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
    "FAISS IndexNSGFlat backend is not available. Reinstall faissR with "
    "FAISS_HOME=/path/to/faiss."
  );
}

List faiss_nsg_float32_knn_impl(SEXP,
                                SEXP,
                                int,
                                int,
                                int,
                                int,
                                std::string,
                                std::string,
                                bool,
                                int,
                                std::string) {
  Rcpp::stop(
    "FAISS float32 NSG backend is not available. Reinstall faissR with "
    "FAISS_HOME=/path/to/faiss."
  );
}

List faiss_nndescent_knn_impl(NumericMatrix,
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
    "FAISS IndexNNDescentFlat backend is not available. Reinstall faissR "
    "with FAISS_HOME=/path/to/faiss."
  );
}

List faiss_nndescent_float32_knn_impl(SEXP,
                                      SEXP,
                                      int,
                                      int,
                                      int,
                                      int,
                                      std::string,
                                      std::string,
                                      bool,
                                      int,
                                      std::string) {
  Rcpp::stop(
    "FAISS float32 NNDescent backend is not available. Reinstall faissR "
    "with FAISS_HOME=/path/to/faiss."
  );
}

List faiss_gpu_ivf_flat_knn_impl(NumericMatrix,
                                 NumericMatrix,
                                 int,
                                 int,
                                 int,
                                 std::string,
                                 std::string,
                                 bool) {
  Rcpp::stop(
    "FAISS GPU IVF Flat backend is not available. Reinstall faissR with "
    "a FAISS GPU/cuVS library visible to configure, for example "
    "FAISS_HOME=/path/to/faiss-gpu."
  );
}

List faiss_gpu_ivf_flat_float32_knn_impl(SEXP,
                                         SEXP,
                                         int,
                                         int,
                                         int,
                                         std::string,
                                         std::string,
                                         bool,
                                         std::string) {
  Rcpp::stop(
    "FAISS GPU float32 IVF Flat backend is not available. Reinstall faissR "
    "with a FAISS GPU/cuVS library visible to configure, for example "
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
                              std::string,
                              std::string,
                              bool) {
  Rcpp::stop(
    "FAISS GPU IVF-PQ backend is not available. Reinstall faissR with "
    "a FAISS GPU/cuVS library visible to configure, for example "
    "FAISS_HOME=/path/to/faiss-gpu."
  );
}

List faiss_gpu_ivfpq_float32_knn_impl(SEXP,
                                      SEXP,
                                      int,
                                      int,
                                      int,
                                      int,
                                      int,
                                      std::string,
                                      std::string,
                                      bool,
                                      std::string) {
  Rcpp::stop(
    "FAISS GPU float32 IVF-PQ backend is not available. Reinstall faissR "
    "with a FAISS GPU/cuVS library visible to configure, for example "
    "FAISS_HOME=/path/to/faiss-gpu."
  );
}

List faiss_gpu_cagra_knn_impl(NumericMatrix,
                              NumericMatrix,
                              int,
                              int,
                              int,
                              int,
                              int,
                              bool) {
  Rcpp::stop(
    "FAISS GPU CAGRA backend is not available. Reinstall faissR with "
    "a FAISS GPU/cuVS library visible to configure, for example "
    "FAISS_HOME=/path/to/faiss-gpu."
  );
}

List faiss_gpu_cagra_float32_knn_impl(SEXP,
                                      SEXP,
                                      int,
                                      int,
                                      int,
                                      int,
                                      int,
                                      bool,
                                      std::string) {
  Rcpp::stop(
    "FAISS GPU float32 CAGRA backend is not available. Reinstall faissR "
    "with a FAISS GPU/cuVS library visible to configure, for example "
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

List faiss_gpu_kmeans_impl(NumericMatrix,
                           int,
                           int,
                           int,
                           double,
                           int,
                           bool) {
  Rcpp::stop(
    "FAISS GPU k-means is not available. Reinstall faissR with "
    "a FAISS GPU/cuVS library visible to configure, for example "
    "FAISS_HOME=/path/to/faiss-gpu."
  );
}
