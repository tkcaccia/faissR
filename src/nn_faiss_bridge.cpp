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
List faiss_flat_float32_knn_impl(SEXP data,
                                 SEXP points,
                                 int k,
                                 bool exclude_self,
                                 int n_threads,
                                 std::string metric,
                                 std::string distance_storage);
List faiss_flat_pretransformed_float32_knn_impl(SEXP data,
                                                SEXP points,
                                                int k,
                                                bool exclude_self,
                                                int n_threads,
                                                std::string distance_storage);
List faiss_ivf_knn_impl(NumericMatrix data,
                        NumericMatrix points,
                        int k,
                        int nlist,
                        int nprobe,
                        std::string metric,
                        std::string distance_output,
                        bool exclude_self,
                        int n_threads);
List faiss_ivf_float32_knn_impl(SEXP data,
                                SEXP points,
                                int k,
                                int nlist,
                                int nprobe,
                                std::string metric,
                                std::string distance_output,
                                bool exclude_self,
                                int n_threads,
                                std::string distance_storage);
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
List faiss_gpu_flat_float32_knn_impl(SEXP data,
                                     SEXP points,
                                     int k,
                                     bool exclude_self,
                                     std::string metric,
                                     std::string distance_output,
                                     std::string distance_storage);
List faiss_gpu_bfknn_float32_gpu_impl(SEXP data,
                                      SEXP points,
                                      int k,
                                      bool exclude_self,
                                      std::string metric,
                                      std::string backend_used,
                                      std::string method);
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
                          std::string metric,
                          std::string distance_output,
                          bool exclude_self,
                          int n_threads);
List faiss_ivfpq_float32_knn_impl(SEXP data,
                                  SEXP points,
                                  int k,
                                  int nlist,
                                  int nprobe,
                                  int pq_m,
                                  int pq_nbits,
                                  std::string metric,
                                  std::string distance_output,
                                  bool exclude_self,
                                  int n_threads,
                                  std::string distance_storage);
List faiss_ivfpq_fastscan_float32_knn_impl(SEXP data,
                                  SEXP points,
                                  int k,
                                  int nlist,
                                  int nprobe,
                                  int pq_m,
                                  std::string metric,
                                  std::string distance_output,
                                  int refine_factor,
                                  int bbs,
                                  bool exclude_self,
                                  int n_threads,
                                  std::string distance_storage);
List faiss_hnsw_knn_impl(NumericMatrix data,
                         NumericMatrix points,
                         int k,
                         int m,
                         int ef_construction,
                         int ef_search,
                         std::string metric,
                         std::string distance_output,
                         bool exclude_self,
                         int n_threads);
List faiss_hnsw_float32_knn_impl(SEXP data,
                                 SEXP points,
                                 int k,
                                 int m,
                                 int ef_construction,
                                 int ef_search,
                                 std::string metric,
                                 std::string distance_output,
                                 bool exclude_self,
                                 int n_threads,
                                 std::string distance_storage);
SEXP faiss_hnsw_index_build_float32_impl(SEXP data,
                                         int m,
                                         int ef_construction,
                                         int ef_search,
                                         std::string metric,
                                         std::string distance_output,
                                         int n_threads);
List faiss_hnsw_index_search_float32_impl(SEXP index_ptr,
                                          SEXP points,
                                          int k,
                                          bool exclude_self,
                                          int ef_search,
                                          int n_threads,
                                          std::string distance_storage);
SEXP faiss_index_build_float32_impl(SEXP data,
                                    std::string kind,
                                    int nlist,
                                    int nprobe,
                                    int pq_m,
                                    int pq_nbits,
                                    int graph_degree,
                                    int search_width,
                                    int build_type,
                                    int n_iter,
                                    std::string metric,
                                    std::string distance_output,
                                    int n_threads);
List faiss_index_search_float32_impl(SEXP index_ptr,
                                     SEXP points,
                                     int k,
                                     bool exclude_self,
                                     int search_width,
                                     int n_threads,
                                     std::string distance_storage);
List faiss_nsg_knn_impl(NumericMatrix data,
                        NumericMatrix points,
                        int k,
                        int r,
                        int search_l,
                        int build_type,
                        std::string metric,
                        std::string distance_output,
                        bool exclude_self,
                        int n_threads);
List faiss_nsg_float32_knn_impl(SEXP data,
                                SEXP points,
                                int k,
                                int r,
                                int search_l,
                                int build_type,
                                std::string metric,
                                std::string distance_output,
                                bool exclude_self,
                                int n_threads,
                                std::string distance_storage);
List faiss_nndescent_knn_impl(NumericMatrix data,
                              NumericMatrix points,
                              int k,
                              int graph_k,
                              int n_iter,
                              int search_l,
                              std::string metric,
                              std::string distance_output,
                              bool exclude_self,
                              int n_threads);
List faiss_nndescent_float32_knn_impl(SEXP data,
                                      SEXP points,
                                      int k,
                                      int graph_k,
                                      int n_iter,
                                      int search_l,
                                      std::string metric,
                                      std::string distance_output,
                                      bool exclude_self,
                                      int n_threads,
                                      std::string distance_storage);
List faiss_gpu_ivf_flat_knn_impl(NumericMatrix data,
                                 NumericMatrix points,
                                 int k,
                                 int nlist,
                                 int nprobe,
                                 std::string metric,
                                 std::string distance_output,
                                 bool exclude_self);
List faiss_gpu_ivf_flat_float32_knn_impl(SEXP data,
                                         SEXP points,
                                         int k,
                                         int nlist,
                                         int nprobe,
                                         std::string metric,
                                         std::string distance_output,
                                         bool exclude_self,
                                         std::string distance_storage);
List faiss_gpu_ivfpq_knn_impl(NumericMatrix data,
                              NumericMatrix points,
                              int k,
                              int nlist,
                              int nprobe,
                              int pq_m,
                              int pq_nbits,
                              std::string metric,
                              std::string distance_output,
                              bool exclude_self);
List faiss_gpu_ivfpq_float32_knn_impl(SEXP data,
                                      SEXP points,
                                      int k,
                                      int nlist,
                                      int nprobe,
                                      int pq_m,
                                      int pq_nbits,
                                      std::string metric,
                                      std::string distance_output,
                                      bool exclude_self,
                                      std::string distance_storage);
List faiss_gpu_cagra_knn_impl(NumericMatrix data,
                              NumericMatrix points,
                              int k,
                              int graph_degree,
                              int intermediate_graph_degree,
                              int search_width,
                              int itopk_size,
                              bool exclude_self);
List faiss_gpu_cagra_float32_knn_impl(SEXP data,
                                      SEXP points,
                                      int k,
                                      int graph_degree,
                                      int intermediate_graph_degree,
                                      int search_width,
                                      int itopk_size,
                                      bool exclude_self,
                                      std::string distance_storage);
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
List nn_faiss_flat_float32_cpp(SEXP data,
                               SEXP points,
                               int k,
                               bool exclude_self,
                               int n_threads,
                               std::string metric,
                               std::string distance_storage) {
  return faiss_flat_float32_knn_impl(
    data, points, k, exclude_self, n_threads, metric, distance_storage
  );
}

// [[Rcpp::export]]
List nn_faiss_flat_pretransformed_float32_cpp(SEXP data,
                                              SEXP points,
                                              int k,
                                              bool exclude_self,
                                              int n_threads,
                                              std::string distance_storage) {
  return faiss_flat_pretransformed_float32_knn_impl(
    data, points, k, exclude_self, n_threads, distance_storage
  );
}

namespace {

std::string float32_backend_used(const std::string& metric) {
  if (metric == "cosine") return "faiss_flat_cosine";
  if (metric == "correlation") return "faiss_flat_correlation";
  return "faiss_flat_l2";
}

void maybe_convert_float32_distances(List& out, const std::string& distances) {
  std::string value = distances;
  if (value == "float32") value = "float";
  if (value != "double" && value != "float") {
    Rcpp::stop("`distances` must be either \"double\" or \"float\"");
  }
  if (value == "float") {
    Rcpp::Environment base = Rcpp::Environment::namespace_env("base");
    Rcpp::Function require_namespace = base["requireNamespace"];
    const bool ok = Rcpp::as<bool>(
      require_namespace("float", Rcpp::Named("quietly") = true)
    );
    if (!ok) {
      Rcpp::stop(
        "`distances = \"float\"` requires the optional float package"
      );
    }
    Rcpp::Environment float_ns = Rcpp::Environment::namespace_env("float");
    Rcpp::Function fl = float_ns["fl"];
    if (!Rf_inherits(out["distances"], "float32")) {
      out["distances"] = fl(out["distances"]);
    }
    out["distance_type"] = "float32";
    out.attr("distance_type") = "float32";
  } else {
    out["distance_type"] = "double";
    out.attr("distance_type") = "double";
  }
}

List faissR_nn_float32_call_impl(SEXP x,
                                 SEXP k,
                                 SEXP backend,
                                 SEXP metric,
                                 SEXP include_self,
                                 SEXP n_threads,
                                 SEXP distances) {
  const int kk = Rcpp::as<int>(k);
  const std::string backend_value = Rcpp::as<std::string>(backend);
  const std::string metric_value = Rcpp::as<std::string>(metric);
  const bool include_self_value = Rcpp::as<bool>(include_self);
  const int n_threads_value = Rcpp::as<int>(n_threads);
  const std::string distances_value = Rcpp::as<std::string>(distances);
  if (metric_value != "euclidean" &&
      metric_value != "cosine" &&
      metric_value != "correlation") {
    Rcpp::stop(
      "`metric` must be one of \"euclidean\", \"cosine\", or \"correlation\""
    );
  }
  if (backend_value != "cpu" &&
      backend_value != "faiss" &&
      backend_value != "cpu_faiss" &&
      backend_value != "cpu_faiss_flat" &&
      backend_value != "flat" &&
      backend_value != "faiss_flat" &&
      backend_value != "faiss_flat_l2") {
    Rcpp::stop(
      "faissR_nn_float32_call currently exposes the CPU FAISS Flat float32 route"
    );
  }
  Rcpp::List out = faiss_flat_float32_knn_impl(
    x,
    x,
    kk,
    !include_self_value,
    n_threads_value,
    metric_value,
    distances_value
  );
  const std::string backend_used = float32_backend_used(metric_value);
  maybe_convert_float32_distances(out, distances_value);
  out["index_base"] = 1;
  out["metric"] = metric_value;
  out["backend_used"] = backend_used;
  out["distance_order"] = "smaller_is_better";
  out["similarity_materialized"] = false;
  if (metric_value == "euclidean") {
    out["distance_is_metric"] = true;
    out["distance_semantics"] = "euclidean_distance";
    out["distance_comparable_across_queries"] = true;
  } else if (metric_value == "cosine") {
    out["distance_is_metric"] = false;
    out["distance_semantics"] = "one_minus_cosine_similarity";
    out["distance_comparable_across_queries"] = true;
  } else if (metric_value == "correlation") {
    out["distance_is_metric"] = false;
    out["distance_semantics"] = "one_minus_row_centered_cosine_similarity";
    out["distance_comparable_across_queries"] = true;
  }
  // Adding list elements may reallocate the R vector, so attach metadata only
  // after the payload is complete.
  out.attr("index_base") = 1;
  out.attr("metric") = metric_value;
  out.attr("backend_used") = backend_used;
  out.attr("resolved_backend") = backend_used;
  out.attr("distance_type") = Rcpp::as<std::string>(out["distance_type"]);
  out.attr("distance_is_metric") = out["distance_is_metric"];
  out.attr("distance_semantics") = out["distance_semantics"];
  out.attr("distance_comparable_across_queries") =
    out["distance_comparable_across_queries"];
  out.attr("distance_order") = "smaller_is_better";
  out.attr("class") = Rcpp::CharacterVector::create("faissR_nn", "list");
  return out;
}

} // namespace

extern "C" SEXP faissR_nn_float32_call(SEXP x,
                                       SEXP k,
                                       SEXP backend,
                                       SEXP metric,
                                       SEXP include_self,
                                       SEXP n_threads) {
  BEGIN_RCPP
  return faissR_nn_float32_call_impl(
    x, k, backend, metric, include_self, n_threads, Rcpp::wrap("double")
  );
  END_RCPP
}

extern "C" SEXP faissR_nn_float32_call_output(SEXP x,
                                              SEXP k,
                                              SEXP backend,
                                              SEXP metric,
                                              SEXP include_self,
                                              SEXP n_threads,
                                              SEXP distances) {
  BEGIN_RCPP
  return faissR_nn_float32_call_impl(
    x, k, backend, metric, include_self, n_threads, distances
  );
  END_RCPP
}

extern "C" SEXP faissR_nn_cuda_tuned_gpu_call(SEXP x,
                                              SEXP k,
                                              SEXP method,
                                              SEXP metric,
                                              SEXP include_self,
                                              SEXP target_recall);

extern "C" SEXP faissR_hnsw_search_v1(SEXP data,
                                      SEXP query,
                                      SEXP n,
                                      SEXP p,
                                      SEXP k,
                                      SEXP target_recall,
                                      SEXP n_threads,
                                      SEXP distance_storage);

extern "C" int faissR_c_api_version_impl() {
  return 1;
}

// [[Rcpp::init]]
void register_faissR_ccallables(DllInfo *dll) {
  (void) dll;
  R_RegisterCCallable(
    "faissR",
    "faissR_c_api_version",
    (DL_FUNC) &faissR_c_api_version_impl
  );
  R_RegisterCCallable(
    "faissR",
    "faissR_nn_float32_call",
    (DL_FUNC) &faissR_nn_float32_call
  );
  R_RegisterCCallable(
    "faissR",
    "faissR_nn_float32_call_output",
    (DL_FUNC) &faissR_nn_float32_call_output
  );
  R_RegisterCCallable(
    "faissR",
    "faissR_nn_cuda_tuned_gpu_call",
    (DL_FUNC) &faissR_nn_cuda_tuned_gpu_call
  );
  R_RegisterCCallable(
    "faissR",
    "faissR_hnsw_search_v1",
    (DL_FUNC) &faissR_hnsw_search_v1
  );
}

// [[Rcpp::export]]
List nn_faiss_ivf_cpp(NumericMatrix data,
                      NumericMatrix points,
                      int k,
                      int nlist,
                      int nprobe,
                      std::string metric,
                      std::string distance_output,
                      bool exclude_self,
                      int n_threads) {
  return faiss_ivf_knn_impl(
    data, points, k, nlist, nprobe, metric, distance_output, exclude_self,
    n_threads
  );
}

// [[Rcpp::export]]
List nn_faiss_ivf_float32_cpp(SEXP data,
                              SEXP points,
                              int k,
                              int nlist,
                              int nprobe,
                              std::string metric,
                              std::string distance_output,
                              bool exclude_self,
                              int n_threads,
                              std::string distance_storage) {
  return faiss_ivf_float32_knn_impl(
    data, points, k, nlist, nprobe, metric, distance_output, exclude_self,
    n_threads, distance_storage
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
List nn_faiss_gpu_flat_float32_cpp(SEXP data,
                                   SEXP points,
                                   int k,
                                   bool exclude_self,
                                   std::string metric,
                                   std::string distance_output,
                                   std::string distance_storage) {
  return faiss_gpu_flat_float32_knn_impl(
    data, points, k, exclude_self, metric, distance_output, distance_storage
  );
}

// [[Rcpp::export]]
List nn_faiss_gpu_bfknn_float32_gpu_cpp(SEXP data,
                                        SEXP points,
                                        int k,
                                        bool exclude_self,
                                        std::string metric,
                                        std::string backend_used,
                                        std::string method) {
  return faiss_gpu_bfknn_float32_gpu_impl(
    data, points, k, exclude_self, metric, backend_used, method
  );
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
                        std::string metric,
                        std::string distance_output,
                        bool exclude_self,
                        int n_threads) {
  return faiss_ivfpq_knn_impl(
    data, points, k, nlist, nprobe, pq_m, pq_nbits, metric, distance_output,
    exclude_self, n_threads
  );
}

// [[Rcpp::export]]
List nn_faiss_ivfpq_float32_cpp(SEXP data,
                                SEXP points,
                                int k,
                                int nlist,
                                int nprobe,
                                int pq_m,
                                int pq_nbits,
                                std::string metric,
                                std::string distance_output,
                                bool exclude_self,
                                int n_threads,
                                std::string distance_storage) {
  return faiss_ivfpq_float32_knn_impl(
    data, points, k, nlist, nprobe, pq_m, pq_nbits, metric, distance_output,
    exclude_self, n_threads, distance_storage
  );
}

// [[Rcpp::export]]
List nn_faiss_ivfpq_fastscan_float32_cpp(SEXP data,
                                SEXP points,
                                int k,
                                int nlist,
                                int nprobe,
                                int pq_m,
                                std::string metric,
                                std::string distance_output,
                                int refine_factor,
                                int bbs,
                                bool exclude_self,
                                int n_threads,
                                std::string distance_storage) {
  return faiss_ivfpq_fastscan_float32_knn_impl(
    data, points, k, nlist, nprobe, pq_m, metric, distance_output,
    refine_factor, bbs, exclude_self, n_threads, distance_storage
  );
}

// [[Rcpp::export]]
List nn_faiss_hnsw_cpp(NumericMatrix data,
                       NumericMatrix points,
                       int k,
                       int m,
                       int ef_construction,
                       int ef_search,
                       std::string metric,
                       std::string distance_output,
                       bool exclude_self,
                       int n_threads) {
  return faiss_hnsw_knn_impl(
    data, points, k, m, ef_construction, ef_search, metric, distance_output,
    exclude_self, n_threads
  );
}

// [[Rcpp::export]]
List nn_faiss_hnsw_float32_cpp(SEXP data,
                               SEXP points,
                               int k,
                               int m,
                               int ef_construction,
                               int ef_search,
                               std::string metric,
                               std::string distance_output,
                               bool exclude_self,
                               int n_threads,
                               std::string distance_storage) {
  return faiss_hnsw_float32_knn_impl(
    data, points, k, m, ef_construction, ef_search, metric, distance_output,
    exclude_self, n_threads, distance_storage
  );
}

List nn_tune_faiss_hnsw_cpp(int n, int p, int k, std::string metric,
                            double target_recall, int m_option,
                            int ef_construction_option,
                            int ef_search_option, bool manual);

extern "C" SEXP faissR_hnsw_search_v1(SEXP data,
                                      SEXP query,
                                      SEXP n,
                                      SEXP p,
                                      SEXP k,
                                      SEXP target_recall,
                                      SEXP n_threads,
                                      SEXP distance_storage) {
  BEGIN_RCPP
  const bool self = Rf_isNull(query);
  if (self) query = data;
  const int neighbors = Rcpp::as<int>(k);
  const int threads = Rcpp::as<int>(n_threads);
  List params = nn_tune_faiss_hnsw_cpp(
    Rcpp::as<int>(n), Rcpp::as<int>(p), neighbors, "euclidean",
    Rcpp::as<double>(target_recall), NA_INTEGER, NA_INTEGER, NA_INTEGER,
    false
  );
  const int m = Rcpp::as<int>(params["m"]);
  const int ef_construction = Rcpp::as<int>(params["ef_construction"]);
  const int ef_search = Rcpp::as<int>(params["ef_search"]);
  List out;
  if (Rf_inherits(data, "float32") && Rf_inherits(query, "float32")) {
    out = nn_faiss_hnsw_float32_cpp(
      data, query, neighbors, m, ef_construction, ef_search,
      "euclidean", "euclidean", self, threads,
      Rcpp::as<std::string>(distance_storage)
    );
  } else {
    if (TYPEOF(data) != REALSXP || TYPEOF(query) != REALSXP) {
      Rcpp::stop("HNSW inputs must have matching double or float32 storage");
    }
    out = nn_faiss_hnsw_cpp(
      NumericMatrix(data), NumericMatrix(query), neighbors, m,
      ef_construction, ef_search, "euclidean", "euclidean", self,
      threads
    );
  }
  out["tuning_rule"] = params["rule"];
  return out;
  END_RCPP
}

// [[Rcpp::export]]
SEXP nn_faiss_hnsw_index_build_float32_cpp(SEXP data,
                                           int m,
                                           int ef_construction,
                                           int ef_search,
                                           std::string metric,
                                           std::string distance_output,
                                           int n_threads) {
  return faiss_hnsw_index_build_float32_impl(
    data, m, ef_construction, ef_search, metric, distance_output, n_threads
  );
}

// [[Rcpp::export]]
List nn_faiss_hnsw_index_search_float32_cpp(SEXP index_ptr,
                                            SEXP points,
                                            int k,
                                            bool exclude_self,
                                            int ef_search,
                                            int n_threads,
                                            std::string distance_storage) {
  return faiss_hnsw_index_search_float32_impl(
    index_ptr, points, k, exclude_self, ef_search, n_threads, distance_storage
  );
}

// [[Rcpp::export]]
SEXP nn_faiss_index_build_float32_cpp(SEXP data,
                                      std::string kind,
                                      int nlist,
                                      int nprobe,
                                      int pq_m,
                                      int pq_nbits,
                                      int graph_degree,
                                      int search_width,
                                      int build_type,
                                      int n_iter,
                                      std::string metric,
                                      std::string distance_output,
                                      int n_threads) {
  return faiss_index_build_float32_impl(
    data,
    kind,
    nlist,
    nprobe,
    pq_m,
    pq_nbits,
    graph_degree,
    search_width,
    build_type,
    n_iter,
    metric,
    distance_output,
    n_threads
  );
}

// [[Rcpp::export]]
List nn_faiss_index_search_float32_cpp(SEXP index_ptr,
                                       SEXP points,
                                       int k,
                                       bool exclude_self,
                                       int search_width,
                                       int n_threads,
                                       std::string distance_storage) {
  return faiss_index_search_float32_impl(
    index_ptr,
    points,
    k,
    exclude_self,
    search_width,
    n_threads,
    distance_storage
  );
}

// [[Rcpp::export]]
List nn_faiss_nsg_cpp(NumericMatrix data,
                      NumericMatrix points,
                      int k,
                      int r,
                      int search_l,
                      int build_type,
                      std::string metric,
                      std::string distance_output,
                      bool exclude_self,
                      int n_threads) {
  return faiss_nsg_knn_impl(
    data, points, k, r, search_l, build_type, metric, distance_output,
    exclude_self, n_threads
  );
}

// [[Rcpp::export]]
List nn_faiss_nsg_float32_cpp(SEXP data,
                              SEXP points,
                              int k,
                              int r,
                              int search_l,
                              int build_type,
                              std::string metric,
                              std::string distance_output,
                              bool exclude_self,
                              int n_threads,
                              std::string distance_storage) {
  return faiss_nsg_float32_knn_impl(
    data, points, k, r, search_l, build_type, metric, distance_output,
    exclude_self, n_threads, distance_storage
  );
}

// [[Rcpp::export]]
List nn_faiss_nndescent_cpp(NumericMatrix data,
                            NumericMatrix points,
                            int k,
                            int graph_k,
                            int n_iter,
                            int search_l,
                            std::string metric,
                            std::string distance_output,
                            bool exclude_self,
                            int n_threads) {
  return faiss_nndescent_knn_impl(
    data, points, k, graph_k, n_iter, search_l, metric, distance_output,
    exclude_self, n_threads
  );
}

// [[Rcpp::export]]
List nn_faiss_nndescent_float32_cpp(SEXP data,
                                    SEXP points,
                                    int k,
                                    int graph_k,
                                    int n_iter,
                                    int search_l,
                                    std::string metric,
                                    std::string distance_output,
                                    bool exclude_self,
                                    int n_threads,
                                    std::string distance_storage) {
  return faiss_nndescent_float32_knn_impl(
    data, points, k, graph_k, n_iter, search_l, metric, distance_output,
    exclude_self, n_threads, distance_storage
  );
}

// [[Rcpp::export]]
List nn_faiss_gpu_ivf_flat_cpp(NumericMatrix data,
                               NumericMatrix points,
                               int k,
                               int nlist,
                               int nprobe,
                               std::string metric,
                               std::string distance_output,
                               bool exclude_self) {
  return faiss_gpu_ivf_flat_knn_impl(
    data, points, k, nlist, nprobe, metric, distance_output, exclude_self
  );
}

// [[Rcpp::export]]
List nn_faiss_gpu_ivf_flat_float32_cpp(SEXP data,
                                       SEXP points,
                                       int k,
                                       int nlist,
                                       int nprobe,
                                       std::string metric,
                                       std::string distance_output,
                                       bool exclude_self,
                                       std::string distance_storage) {
  return faiss_gpu_ivf_flat_float32_knn_impl(
    data, points, k, nlist, nprobe, metric, distance_output, exclude_self,
    distance_storage
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
                            std::string metric,
                            std::string distance_output,
                            bool exclude_self) {
  return faiss_gpu_ivfpq_knn_impl(
    data, points, k, nlist, nprobe, pq_m, pq_nbits, metric, distance_output,
    exclude_self
  );
}

// [[Rcpp::export]]
List nn_faiss_gpu_ivfpq_float32_cpp(SEXP data,
                                    SEXP points,
                                    int k,
                                    int nlist,
                                    int nprobe,
                                    int pq_m,
                                    int pq_nbits,
                                    std::string metric,
                                    std::string distance_output,
                                    bool exclude_self,
                                    std::string distance_storage) {
  return faiss_gpu_ivfpq_float32_knn_impl(
    data, points, k, nlist, nprobe, pq_m, pq_nbits, metric, distance_output,
    exclude_self, distance_storage
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
List nn_faiss_gpu_cagra_float32_cpp(SEXP data,
                                    SEXP points,
                                    int k,
                                    int graph_degree,
                                    int intermediate_graph_degree,
                                    int search_width,
                                    int itopk_size,
                                    bool exclude_self,
                                    std::string distance_storage) {
  return faiss_gpu_cagra_float32_knn_impl(
    data, points, k, graph_degree, intermediate_graph_degree, search_width,
    itopk_size, exclude_self, distance_storage
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
