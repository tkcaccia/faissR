#include <Rcpp.h>

#include <string>

using Rcpp::List;
using Rcpp::NumericMatrix;

bool cuvs_is_available_impl();
std::string cuvs_info_json_impl();
List cuvs_bruteforce_knn_impl(NumericMatrix data,
                              NumericMatrix points,
                              int k,
                              bool exclude_self);
List cuvs_bruteforce_float32_knn_impl(SEXP data,
                                      SEXP points,
                                      int k,
                                      bool exclude_self,
                                      std::string distance_storage);
List cuvs_cagra_knn_impl(NumericMatrix data,
                         NumericMatrix points,
                         int k,
                         bool exclude_self,
                         int graph_degree,
                         int intermediate_graph_degree,
                         int search_width,
                         int itopk_size,
                         std::string build_algo);
List cuvs_cagra_float32_knn_impl(SEXP data,
                                 SEXP points,
                                 int k,
                                 bool exclude_self,
                                 int graph_degree,
                                 int intermediate_graph_degree,
                                 int search_width,
                                 int itopk_size,
                                 std::string build_algo,
                                 std::string distance_storage);
List cuvs_nndescent_self_knn_impl(NumericMatrix data,
                                  int k,
                                  int graph_degree,
                                  int intermediate_graph_degree,
                                  int max_iterations);
List cuvs_nndescent_self_float32_knn_impl(SEXP data,
                                          int k,
                                          int graph_degree,
                                          int intermediate_graph_degree,
                                          int max_iterations,
                                          std::string distance_storage);
List cuvs_hnsw_knn_impl(NumericMatrix data,
                        NumericMatrix points,
                        int k,
                        bool exclude_self,
                        int graph_degree,
                        int intermediate_graph_degree,
                        int ef,
                        int n_threads,
                        std::string cagra_build_algo);
List cuvs_hnsw_float32_knn_impl(SEXP data,
                                SEXP points,
                                int k,
                                bool exclude_self,
                                int graph_degree,
                                int intermediate_graph_degree,
                                int ef,
                                int n_threads,
                                std::string cagra_build_algo,
                                std::string distance_storage);
List cuvs_ivf_flat_knn_impl(NumericMatrix data,
                            NumericMatrix points,
                            int k,
                            int n_lists,
                            int n_probes,
                            bool exclude_self);
List cuvs_ivf_flat_float32_knn_impl(SEXP data,
                                    SEXP points,
                                    int k,
                                    int n_lists,
                                    int n_probes,
                                    bool exclude_self,
                                    std::string distance_storage);
List cuvs_ivf_pq_knn_impl(NumericMatrix data,
                          NumericMatrix points,
                          int k,
                          int n_lists,
                          int n_probes,
                          int pq_dim,
                          int pq_bits,
                          bool exclude_self);
List cuvs_ivf_pq_float32_knn_impl(SEXP data,
                                  SEXP points,
                                  int k,
                                  int n_lists,
                                  int n_probes,
                                  int pq_dim,
                                  int pq_bits,
                                  bool exclude_self,
                                  std::string distance_storage);
SEXP cuvs_ivf_pq_index_build_float32_impl(SEXP data,
                                          int n_lists,
                                          int n_probes,
                                          int pq_dim,
                                          int pq_bits);
List cuvs_ivf_pq_index_search_float32_impl(SEXP index_ptr,
                                           SEXP points,
                                           int k,
                                           bool exclude_self,
                                           int n_probes,
                                           bool query_is_index_data,
                                           bool cache_query_device_buffer,
                                           std::string query_cache_key,
                                           std::string distance_storage);
List cuvs_kmeans_impl(NumericMatrix data,
                      int centers,
                      int max_iter,
                      int n_init,
                      double tol,
                      int64_t streaming_batch_size,
                      bool kmeans_plus_plus);

// [[Rcpp::export]]
bool cuvs_available_cpp() {
  return cuvs_is_available_impl();
}

// [[Rcpp::export]]
std::string cuvs_info_json_cpp() {
  return cuvs_info_json_impl();
}

// [[Rcpp::export]]
List nn_cuvs_bruteforce_cpp(NumericMatrix data,
                            NumericMatrix points,
                            int k,
                            bool exclude_self) {
  return cuvs_bruteforce_knn_impl(data, points, k, exclude_self);
}

// [[Rcpp::export]]
List nn_cuvs_bruteforce_float32_cpp(SEXP data,
                                    SEXP points,
                                    int k,
                                    bool exclude_self,
                                    std::string distance_storage) {
  return cuvs_bruteforce_float32_knn_impl(
    data,
    points,
    k,
    exclude_self,
    distance_storage
  );
}

// [[Rcpp::export]]
List nn_cuvs_cagra_cpp(NumericMatrix data,
                       NumericMatrix points,
                       int k,
                       bool exclude_self,
                       int graph_degree,
                       int intermediate_graph_degree,
                       int search_width,
                       int itopk_size,
                       std::string build_algo) {
  return cuvs_cagra_knn_impl(
    data,
    points,
    k,
    exclude_self,
    graph_degree,
    intermediate_graph_degree,
    search_width,
    itopk_size,
    build_algo
  );
}

// [[Rcpp::export]]
List nn_cuvs_cagra_float32_cpp(SEXP data,
                               SEXP points,
                               int k,
                               bool exclude_self,
                               int graph_degree,
                               int intermediate_graph_degree,
                               int search_width,
                               int itopk_size,
                               std::string build_algo,
                               std::string distance_storage) {
  return cuvs_cagra_float32_knn_impl(
    data,
    points,
    k,
    exclude_self,
    graph_degree,
    intermediate_graph_degree,
    search_width,
    itopk_size,
    build_algo,
    distance_storage
  );
}

// [[Rcpp::export]]
List nn_cuvs_nndescent_self_cpp(NumericMatrix data,
                                int k,
                                int graph_degree,
                                int intermediate_graph_degree,
                                int max_iterations) {
  return cuvs_nndescent_self_knn_impl(
    data,
    k,
    graph_degree,
    intermediate_graph_degree,
    max_iterations
  );
}

// [[Rcpp::export]]
List nn_cuvs_nndescent_self_float32_cpp(SEXP data,
                                        int k,
                                        int graph_degree,
                                        int intermediate_graph_degree,
                                        int max_iterations,
                                        std::string distance_storage) {
  return cuvs_nndescent_self_float32_knn_impl(
    data,
    k,
    graph_degree,
    intermediate_graph_degree,
    max_iterations,
    distance_storage
  );
}

// [[Rcpp::export]]
List nn_cuvs_hnsw_cpp(NumericMatrix data,
                      NumericMatrix points,
                      int k,
                      bool exclude_self,
                      int graph_degree,
                      int intermediate_graph_degree,
                      int ef,
                      int n_threads,
                      std::string cagra_build_algo) {
  return cuvs_hnsw_knn_impl(
    data,
    points,
    k,
    exclude_self,
    graph_degree,
    intermediate_graph_degree,
    ef,
    n_threads,
    cagra_build_algo
  );
}

// [[Rcpp::export]]
List nn_cuvs_hnsw_float32_cpp(SEXP data,
                              SEXP points,
                              int k,
                              bool exclude_self,
                              int graph_degree,
                              int intermediate_graph_degree,
                              int ef,
                              int n_threads,
                              std::string cagra_build_algo,
                              std::string distance_storage) {
  return cuvs_hnsw_float32_knn_impl(
    data,
    points,
    k,
    exclude_self,
    graph_degree,
    intermediate_graph_degree,
    ef,
    n_threads,
    cagra_build_algo,
    distance_storage
  );
}

// [[Rcpp::export]]
List nn_cuvs_ivf_flat_cpp(NumericMatrix data,
                          NumericMatrix points,
                          int k,
                          int n_lists,
                          int n_probes,
                          bool exclude_self) {
  return cuvs_ivf_flat_knn_impl(
    data,
    points,
    k,
    n_lists,
    n_probes,
    exclude_self
  );
}

// [[Rcpp::export]]
List nn_cuvs_ivf_flat_float32_cpp(SEXP data,
                                  SEXP points,
                                  int k,
                                  int n_lists,
                                  int n_probes,
                                  bool exclude_self,
                                  std::string distance_storage) {
  return cuvs_ivf_flat_float32_knn_impl(
    data,
    points,
    k,
    n_lists,
    n_probes,
    exclude_self,
    distance_storage
  );
}

// [[Rcpp::export]]
List nn_cuvs_ivf_pq_cpp(NumericMatrix data,
                        NumericMatrix points,
                        int k,
                        int n_lists,
                        int n_probes,
                        int pq_dim,
                        int pq_bits,
                        bool exclude_self) {
  return cuvs_ivf_pq_knn_impl(
    data,
    points,
    k,
    n_lists,
    n_probes,
    pq_dim,
    pq_bits,
    exclude_self
  );
}

// [[Rcpp::export]]
List nn_cuvs_ivf_pq_float32_cpp(SEXP data,
                                SEXP points,
                                int k,
                                int n_lists,
                                int n_probes,
                                int pq_dim,
                                int pq_bits,
                                bool exclude_self,
                                std::string distance_storage) {
  return cuvs_ivf_pq_float32_knn_impl(
    data,
    points,
    k,
    n_lists,
    n_probes,
    pq_dim,
    pq_bits,
    exclude_self,
    distance_storage
  );
}

// [[Rcpp::export]]
SEXP nn_cuvs_ivf_pq_index_build_float32_cpp(SEXP data,
                                            int n_lists,
                                            int n_probes,
                                            int pq_dim,
                                            int pq_bits) {
  return cuvs_ivf_pq_index_build_float32_impl(
    data,
    n_lists,
    n_probes,
    pq_dim,
    pq_bits
  );
}

// [[Rcpp::export]]
List nn_cuvs_ivf_pq_index_search_float32_cpp(SEXP index_ptr,
                                             SEXP points,
                                             int k,
                                             bool exclude_self,
                                             int n_probes,
                                             bool query_is_index_data,
                                             bool cache_query_device_buffer,
                                             std::string query_cache_key,
                                             std::string distance_storage) {
  return cuvs_ivf_pq_index_search_float32_impl(
    index_ptr,
    points,
    k,
    exclude_self,
    n_probes,
    query_is_index_data,
    cache_query_device_buffer,
    query_cache_key,
    distance_storage
  );
}

// [[Rcpp::export]]
List kmeans_cuvs_cpp(NumericMatrix data,
                     int centers,
                     int max_iter,
                     int n_init,
                     double tol,
                     int streaming_batch_size,
                     bool kmeans_plus_plus) {
  return cuvs_kmeans_impl(
    data,
    centers,
    max_iter,
    n_init,
    tol,
    static_cast<int64_t>(streaming_batch_size),
    kmeans_plus_plus
  );
}
