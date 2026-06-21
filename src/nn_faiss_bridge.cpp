#include <Rcpp.h>

#include <string>

using Rcpp::List;
using Rcpp::NumericMatrix;

bool faiss_is_available_impl();
std::string faiss_info_json_impl();
List faiss_flat_knn_impl(NumericMatrix data,
                         NumericMatrix points,
                         int k,
                         bool exclude_self,
                         int n_threads);
List faiss_ivf_knn_impl(NumericMatrix data,
                        NumericMatrix points,
                        int k,
                        int nlist,
                        int nprobe,
                        bool exclude_self,
                        int n_threads);
List faiss_flat_ip_knn_impl(NumericMatrix data,
                            NumericMatrix points,
                            int k,
                            bool exclude_self,
                            int n_threads);
List faiss_flat_normalized_ip_distance_knn_impl(NumericMatrix data,
                                                NumericMatrix points,
                                                int k,
                                                bool exclude_self,
                                                int n_threads);
List faiss_gpu_flat_knn_impl(NumericMatrix data,
                             NumericMatrix points,
                             int k,
                             bool exclude_self);
List faiss_gpu_flat_ip_knn_impl(NumericMatrix data,
                                NumericMatrix points,
                                int k,
                                bool exclude_self);
List faiss_gpu_flat_normalized_ip_distance_knn_impl(NumericMatrix data,
                                                    NumericMatrix points,
                                                    int k,
                                                    bool exclude_self);
List faiss_ivfpq_knn_impl(NumericMatrix data,
                          NumericMatrix points,
                          int k,
                          int nlist,
                          int nprobe,
                          int pq_m,
                          int pq_nbits,
                          bool exclude_self,
                          int n_threads);
List faiss_hnsw_knn_impl(NumericMatrix data,
                         NumericMatrix points,
                         int k,
                         int m,
                         int ef_construction,
                         int ef_search,
                         bool exclude_self,
                         int n_threads);
List faiss_nsg_knn_impl(NumericMatrix data,
                        NumericMatrix points,
                        int k,
                        int r,
                        int search_l,
                        int build_type,
                        bool exclude_self,
                        int n_threads);
List faiss_nndescent_knn_impl(NumericMatrix data,
                              NumericMatrix points,
                              int k,
                              int graph_k,
                              int n_iter,
                              int search_l,
                              bool exclude_self,
                              int n_threads);
List faiss_gpu_ivf_flat_knn_impl(NumericMatrix data,
                                 NumericMatrix points,
                                 int k,
                                 int nlist,
                                 int nprobe,
                                 bool exclude_self);
List faiss_gpu_ivfpq_knn_impl(NumericMatrix data,
                              NumericMatrix points,
                              int k,
                              int nlist,
                              int nprobe,
                              int pq_m,
                              int pq_nbits,
                              bool exclude_self);
List faiss_gpu_cagra_knn_impl(NumericMatrix data,
                              NumericMatrix points,
                              int k,
                              int graph_degree,
                              int intermediate_graph_degree,
                              int search_width,
                              int itopk_size,
                              bool exclude_self);
List faiss_kmeans_impl(NumericMatrix data,
                       int centers,
                       int max_iter,
                       int nredo,
                       double tol,
                       int seed,
                       int n_threads,
                       bool kmeans_plus_plus);
List faiss_gpu_kmeans_impl(NumericMatrix data,
                           int centers,
                           int max_iter,
                           int nredo,
                           double tol,
                           int seed,
                           bool kmeans_plus_plus);

// [[Rcpp::export]]
bool faiss_available_cpp() {
  return faiss_is_available_impl();
}

// [[Rcpp::export]]
std::string faiss_info_json_cpp() {
  return faiss_info_json_impl();
}

// [[Rcpp::export]]
List nn_faiss_flat_cpp(NumericMatrix data,
                       NumericMatrix points,
                       int k,
                       bool exclude_self,
                       int n_threads) {
  return faiss_flat_knn_impl(data, points, k, exclude_self, n_threads);
}

// [[Rcpp::export]]
List nn_faiss_ivf_cpp(NumericMatrix data,
                      NumericMatrix points,
                      int k,
                      int nlist,
                      int nprobe,
                      bool exclude_self,
                      int n_threads) {
  return faiss_ivf_knn_impl(
    data, points, k, nlist, nprobe, exclude_self, n_threads
  );
}

// [[Rcpp::export]]
List nn_faiss_flat_ip_cpp(NumericMatrix data,
                          NumericMatrix points,
                          int k,
                          bool exclude_self,
                          int n_threads) {
  return faiss_flat_ip_knn_impl(data, points, k, exclude_self, n_threads);
}

// [[Rcpp::export]]
List nn_faiss_flat_normalized_ip_distance_cpp(NumericMatrix data,
                                              NumericMatrix points,
                                              int k,
                                              bool exclude_self,
                                              int n_threads) {
  return faiss_flat_normalized_ip_distance_knn_impl(
    data, points, k, exclude_self, n_threads
  );
}

// [[Rcpp::export]]
List nn_faiss_gpu_flat_cpp(NumericMatrix data,
                           NumericMatrix points,
                           int k,
                           bool exclude_self) {
  return faiss_gpu_flat_knn_impl(data, points, k, exclude_self);
}

// [[Rcpp::export]]
List nn_faiss_gpu_flat_ip_cpp(NumericMatrix data,
                              NumericMatrix points,
                              int k,
                              bool exclude_self) {
  return faiss_gpu_flat_ip_knn_impl(data, points, k, exclude_self);
}

// [[Rcpp::export]]
List nn_faiss_gpu_flat_normalized_ip_distance_cpp(NumericMatrix data,
                                                  NumericMatrix points,
                                                  int k,
                                                  bool exclude_self) {
  return faiss_gpu_flat_normalized_ip_distance_knn_impl(
    data, points, k, exclude_self
  );
}

// [[Rcpp::export]]
List nn_faiss_ivfpq_cpp(NumericMatrix data,
                        NumericMatrix points,
                        int k,
                        int nlist,
                        int nprobe,
                        int pq_m,
                        int pq_nbits,
                        bool exclude_self,
                        int n_threads) {
  return faiss_ivfpq_knn_impl(
    data, points, k, nlist, nprobe, pq_m, pq_nbits, exclude_self, n_threads
  );
}

// [[Rcpp::export]]
List nn_faiss_hnsw_cpp(NumericMatrix data,
                       NumericMatrix points,
                       int k,
                       int m,
                       int ef_construction,
                       int ef_search,
                       bool exclude_self,
                       int n_threads) {
  return faiss_hnsw_knn_impl(
    data, points, k, m, ef_construction, ef_search, exclude_self, n_threads
  );
}

// [[Rcpp::export]]
List nn_faiss_nsg_cpp(NumericMatrix data,
                      NumericMatrix points,
                      int k,
                      int r,
                      int search_l,
                      int build_type,
                      bool exclude_self,
                      int n_threads) {
  return faiss_nsg_knn_impl(
    data, points, k, r, search_l, build_type, exclude_self, n_threads
  );
}

// [[Rcpp::export]]
List nn_faiss_nndescent_cpp(NumericMatrix data,
                            NumericMatrix points,
                            int k,
                            int graph_k,
                            int n_iter,
                            int search_l,
                            bool exclude_self,
                            int n_threads) {
  return faiss_nndescent_knn_impl(
    data, points, k, graph_k, n_iter, search_l, exclude_self, n_threads
  );
}

// [[Rcpp::export]]
List nn_faiss_gpu_ivf_flat_cpp(NumericMatrix data,
                               NumericMatrix points,
                               int k,
                               int nlist,
                               int nprobe,
                               bool exclude_self) {
  return faiss_gpu_ivf_flat_knn_impl(
    data, points, k, nlist, nprobe, exclude_self
  );
}

// [[Rcpp::export]]
List nn_faiss_gpu_ivfpq_cpp(NumericMatrix data,
                            NumericMatrix points,
                            int k,
                            int nlist,
                            int nprobe,
                            int pq_m,
                            int pq_nbits,
                            bool exclude_self) {
  return faiss_gpu_ivfpq_knn_impl(
    data, points, k, nlist, nprobe, pq_m, pq_nbits, exclude_self
  );
}

// [[Rcpp::export]]
List nn_faiss_gpu_cagra_cpp(NumericMatrix data,
                            NumericMatrix points,
                            int k,
                            int graph_degree,
                            int intermediate_graph_degree,
                            int search_width,
                            int itopk_size,
                            bool exclude_self) {
  return faiss_gpu_cagra_knn_impl(
    data,
    points,
    k,
    graph_degree,
    intermediate_graph_degree,
    search_width,
    itopk_size,
    exclude_self
  );
}

// [[Rcpp::export]]
List kmeans_faiss_cpp(NumericMatrix data,
                      int centers,
                      int max_iter,
                      int nredo,
                      double tol,
                      int seed,
                      int n_threads,
                      bool kmeans_plus_plus) {
  return faiss_kmeans_impl(
    data,
    centers,
    max_iter,
    nredo,
    tol,
    seed,
    n_threads,
    kmeans_plus_plus
  );
}

// [[Rcpp::export]]
List kmeans_faiss_gpu_cpp(NumericMatrix data,
                          int centers,
                          int max_iter,
                          int nredo,
                          double tol,
                          int seed,
                          bool kmeans_plus_plus) {
  return faiss_gpu_kmeans_impl(
    data,
    centers,
    max_iter,
    nredo,
    tol,
    seed,
    kmeans_plus_plus
  );
}
