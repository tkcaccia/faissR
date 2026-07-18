#include <Rcpp.h>
#include <algorithm>
#include <cmath>
#include <sstream>
#include <string>

using namespace Rcpp;

namespace {

#include "nn_hpc_tuning_tables.hpp"

bool finite1(double x) {
  return std::isfinite(x);
}

int gcd_int(int a, int b) {
  a = std::abs(a);
  b = std::abs(b);
  while (b != 0) {
    const int r = a % b;
    a = b;
    b = r;
  }
  return a == 0 ? 1 : a;
}

struct CuvsIvfPqAlignment {
  int pq_dim;
  int pq_bits;
  bool adjusted;
  std::string rule;
};

bool cuvs_ivfpq_byte_aligned(int pq_dim, int pq_bits, int p) {
  const int effective_dim = pq_dim > 0 ? pq_dim : p;
  return effective_dim > 0 && ((pq_bits * effective_dim) % 8 == 0);
}

CuvsIvfPqAlignment repair_cuvs_ivfpq_alignment(int p,
                                               int pq_dim,
                                               int pq_bits) {
  if (p < 1) p = 1;
  const int original_dim = pq_dim;
  const int original_bits = pq_bits;
  pq_dim = std::max(0, pq_dim);
  pq_bits = std::max(4, std::min(8, pq_bits));
  if (pq_dim > p) pq_dim = p;

  if (cuvs_ivfpq_byte_aligned(pq_dim, pq_bits, p)) {
    return {pq_dim, pq_bits, original_dim != pq_dim || original_bits != pq_bits,
            "byte_aligned"};
  }

  const int step = 8 / gcd_int(pq_bits, 8);
  int effective_dim = pq_dim > 0 ? pq_dim : p;
  effective_dim = std::min(std::max(1, effective_dim), p);
  effective_dim -= effective_dim % step;

  if (effective_dim >= 1) {
    pq_dim = effective_dim;
  } else {
    // No positive dimension can satisfy the requested packing; use 8-bit PQ.
    pq_bits = 8;
    pq_dim = std::min(std::max(1, pq_dim > 0 ? pq_dim : p), p);
  }

  return {pq_dim, pq_bits, true,
          original_bits != pq_bits ? "pq_bits_promoted_for_byte_alignment" :
                                     "pq_dim_reduced_for_byte_alignment"};
}

bool grid_self_knn(bool self_query,
                   int n,
                   int p,
                   int k,
                   bool exclude_self,
                   const std::string& metric) {
  if (!self_query) return false;
  if (!(metric == "euclidean" || metric == "cosine" || metric == "correlation")) return false;
  if (!(p == 2 || p == 3)) return false;
  if (n < 10000) return false;
  const int nonself_k = exclude_self ? k : k - 1;
  return nonself_k >= 1;
}

bool auto_cpu_approx_self(bool self_query,
                          int n,
                          int p,
                          int k,
                          double work_size) {
  if (!self_query) return false;
  if (n < 5000 || k < 10 || p < 2) return false;
  if (!finite1(work_size) || work_size < 5e8) return false;
  return true;
}

bool native_nsg_fallback(bool self_query,
                         int n,
                         int p,
                         int k,
                         double work_size,
                         const std::string& metric) {
  if (!auto_cpu_approx_self(self_query, n, p, k, work_size)) return false;
  return metric != "euclidean" && (k >= 50 || p >= 128);
}

bool native_nndescent_fallback(bool self_query,
                               int n,
                               int p,
                               int k,
                               double work_size) {
  return auto_cpu_approx_self(self_query, n, p, k, work_size);
}

std::string cpu_metric_flat_backend(const std::string& metric) {
  if (metric == "cosine") return "faiss_flat_cosine";
  if (metric == "correlation") return "faiss_flat_correlation";
  if (metric == "inner_product") return "faiss_flat_ip";
  return "";
}

std::string public_method_from_backend(const std::string& backend) {
  if (backend == "auto" || backend == "cpu_auto" ||
      backend == "cuda_auto" || backend == "gpu_auto") return "auto";
  if (backend == "cpu" || backend == "cuda") return "exact";
  if (backend == "cuda_cuvs_bruteforce") return "bruteforce";
  if (backend == "faiss" || backend == "cpu_faiss" ||
      backend == "cpu_faiss_flat" || backend == "faiss_flat" ||
      backend == "faiss_flat_l2" || backend == "faiss_flat_ip" ||
      backend == "faiss_flat_cosine" || backend == "faiss_flat_correlation" ||
      backend == "faiss_gpu_flat" || backend == "faiss_gpu_flat_l2" ||
      backend == "cuda_faiss_flat_l2" || backend == "faiss_gpu_flat_ip" ||
      backend == "cuda_faiss_flat_ip" || backend == "faiss_gpu_flat_cosine" ||
      backend == "cuda_faiss_flat_cosine" ||
      backend == "faiss_gpu_flat_correlation" ||
      backend == "cuda_faiss_flat_correlation") return "flat";
  if (backend == "grid" || backend == "cpu_grid" ||
      backend == "grid2d" || backend == "grid3d" ||
      backend == "cpu_grid2d" || backend == "cpu_grid3d" ||
      backend == "cuda_grid" || backend == "cuda_grid_auto" ||
      backend == "gpu_grid" || backend == "cuda_grid2d" ||
      backend == "cuda_grid3d") return "grid";
  if (backend == "faiss_hnsw" ||
      backend == "cuda_cuvs_hnsw" || backend == "cuvs_hnsw") return "hnsw";
  if (backend == "faiss_ivf" || backend == "cpu_faiss_index_ivf" ||
      backend == "faiss_ivf_flat" || backend == "faiss_gpu_ivf" ||
      backend == "faiss_gpu_ivf_flat" || backend == "cuda_faiss_ivf_flat" ||
      backend == "cuvs_ivf" || backend == "cuda_cuvs_ivf" ||
      backend == "cuvs_ivf_flat" || backend == "cuda_cuvs_ivf_flat") return "ivf";
  if (backend == "faiss_ivfpq" || backend == "faiss_gpu_ivfpq" ||
      backend == "cuda_faiss_ivfpq" || backend == "cuvs_ivfpq" ||
      backend == "cuda_cuvs_ivfpq" || backend == "cuvs_ivf_pq" ||
      backend == "cuda_cuvs_ivf_pq") return "ivfpq";
  if (backend == "faiss_ivfpq_fastscan" || backend == "cuda_cuvs_ivfpq_fastscan" ||
      backend == "cuvs_ivfpq_fastscan") return "ivfpq_fastscan";
  if (backend == "cpu_vamana" || backend == "cuda_vamana") return "vamana";
  if (backend == "faiss_nsg" || backend == "cpu_nsg" ||
      backend == "cuda_nsg") return "nsg";
  if (backend == "cpu_nndescent" || backend == "faiss_nndescent" ||
      backend == "cuda_cuvs_nndescent" || backend == "cuvs_nndescent") {
    return "nndescent";
  }
  if (backend == "faiss_gpu_cagra" || backend == "cuda_faiss_cagra" ||
      backend == "cuda_cuvs_cagra" || backend == "cuda_cagra" ||
      backend == "gpu_cagra") return "cagra";
  return "";
}

std::string device_from_backend(const std::string& backend) {
  if (backend == "auto" || backend == "cpu_auto" ||
      backend == "cuda_auto" || backend == "gpu_auto") return "auto";
  if (backend.rfind("cuda", 0) == 0 ||
      backend.rfind("gpu", 0) == 0 ||
      backend.rfind("cuvs", 0) == 0 ||
      backend.rfind("faiss_gpu", 0) == 0) return "cuda";
  return "cpu";
}

std::string cuda_non_euclidean_backend(const std::string& metric,
                                       const std::string& requested_device,
                                       bool self_query,
                                       int n,
                                       int p,
                                       int n_points,
                                       int k,
                                       double work_size,
                                       bool cuda_available,
                                       bool cuvs_available,
                                       bool faiss_gpu_available,
                                       const std::string& cagra_preference,
                                       int graph_n,
                                       int graph_min_k,
                                       double graph_work,
                                       int cagra_compact_n,
                                       int cagra_high_dim_p,
                                       int cagra_compact_max_k) {
  if (metric == "euclidean") return "";
  (void)self_query;
  (void)n;
  (void)p;
  (void)n_points;
  (void)k;
  (void)work_size;
  (void)cuda_available;
  (void)graph_n;
  (void)graph_min_k;
  (void)graph_work;
  (void)cagra_preference;
  (void)cagra_compact_n;
  (void)cagra_high_dim_p;
  (void)cagra_compact_max_k;
  const std::string flat_backend =
    metric == "cosine" ? "faiss_gpu_flat_cosine" :
    metric == "correlation" ? "faiss_gpu_flat_correlation" :
    "faiss_gpu_flat_ip";
  if (faiss_gpu_available) return flat_backend;
  if (cuvs_available) return "cuda_cuvs_bruteforce";
  if (requested_device == "auto") return "cpu_auto";
  return "__cuda_non_euclidean_unavailable__";
}

int cuda_auto_k_bucket(int k) {
  if (k <= 15) return 15;
  if (k <= 30) return 30;
  if (k <= 50) return 50;
  return 100;
}

std::string cuda_auto_flat_ivf_shape_group(int n, int p) {
  n = std::max(1, n);
  p = std::max(1, p);
  if (n < 5000 && p >= 1024) return "small_n_very_high_dim";
  if (n < 5000 && p >= 128) return "small_n_high_dim";
  if (n < 20000 && p >= 128) return "small_high_dim";
  if (n >= 500000 && p >= 256) return "large_high_dim";
  if (n >= 500000 && p <= 64) return "large_low_dim";
  if (n >= 20000 && n < 200000 && p >= 256) return "medium_high_dim";
  if (n >= 20000 && n < 200000 && p <= 128) return "medium_low_dim";
  return "general";
}

std::string cuda_auto_exact_backend(bool faiss_gpu_available,
                                    bool cuvs_available,
                                    bool cuda_available) {
  if (faiss_gpu_available) return "faiss_gpu_flat_l2";
  if (cuvs_available) return "cuda_cuvs_bruteforce";
  return cuda_available ? "cuda" : "cuda_auto";
}

std::string cuda_auto_ivf_backend(bool faiss_gpu_available,
                                  bool cuvs_available) {
  if (faiss_gpu_available) return "faiss_gpu_ivf_flat";
  if (cuvs_available) return "cuda_cuvs_ivf_flat";
  return "";
}

bool cuda_auto_prefers_ivf(bool self_query,
                           int n,
                           int p,
                           int n_points,
                           int k,
                           double work_size,
                           int target_recall_code,
                           const std::string& shape_group) {
  if (!self_query || n != n_points) return false;
  if (k <= 8 || n < 1000) return false;
  const int k_bucket = cuda_auto_k_bucket(k);
  (void)k_bucket;
  if (shape_group == "large_low_dim") return true;
  if (shape_group == "large_high_dim" && target_recall_code <= 95) return true;
  if (shape_group == "small_n_very_high_dim" ||
      shape_group == "small_n_high_dim" ||
      shape_group == "small_high_dim" ||
      shape_group == "medium_high_dim" ||
      shape_group == "medium_low_dim" ||
      shape_group == "large_high_dim") {
    return false;
  }
  if (target_recall_code <= 95 && n >= 5000 &&
      finite1(work_size) && work_size >= 5e8) {
    return true;
  }
  return n >= 100000 && finite1(work_size) && work_size >= 1e10;
}

std::string select_cuda(bool self_query,
                        int n,
                        int p,
                        int n_points,
                        int k,
                        double work_size,
                        const std::string& metric,
                        bool cuda_available,
                        bool cuvs_available,
                        bool faiss_gpu_available,
                        const std::string& cagra_preference,
                        int cuda_exact_n,
                        double cuda_exact_work,
                        int metric_graph_n,
                        int metric_graph_min_k,
                        double metric_graph_work,
                        int cagra_compact_n,
                        int cagra_high_dim_p,
                        int cagra_compact_max_k,
                        double cuvs_bruteforce_work_threshold,
                        int target_recall_code,
                        std::string& auto_rule,
                        std::string& auto_shape_group,
                        std::string& error) {
  (void)cuvs_bruteforce_work_threshold;
  if (k > 256) {
    auto_rule = "cuda_auto_k_limit";
    error = "CUDA auto backends currently support `k <= 256`.";
    return "cuda_auto";
  }
  if (!cuda_available && !cuvs_available) {
    auto_rule = "cuda_auto_unavailable";
    error = "No CUDA GPU backend is available on this machine.";
    return "cuda_auto";
  }
  if (grid_self_knn(self_query, n, p, k, false, metric) && cuda_available) {
    auto_rule = "cuda_auto_grid";
    auto_shape_group = "grid_2d3d";
    return "cuda_grid";
  }
  const std::string metric_backend = cuda_non_euclidean_backend(
    metric, "cuda", self_query, n, p, n_points, k, work_size,
    cuda_available, cuvs_available, faiss_gpu_available, cagra_preference,
    metric_graph_n, metric_graph_min_k, metric_graph_work,
    cagra_compact_n, cagra_high_dim_p, cagra_compact_max_k
  );
  if (metric_backend == "__cuda_non_euclidean_unavailable__") {
    auto_rule = "cuda_auto_non_euclidean_unavailable";
    error = "CUDA auto for non-Euclidean metrics requires FAISS GPU Flat support "
      "for exact routes or CAGRA/cuVS support for large self-KNN graph routes. "
      "Use `backend = \"auto\"` to fall back to CPU, or rebuild faissR with "
      "FAISS GPU/cuVS support.";
    return "cuda_auto";
  }
  if (!metric_backend.empty()) {
    auto_rule = "cuda_auto_non_euclidean_metric_route";
    auto_shape_group = cuda_auto_flat_ivf_shape_group(n, p);
    return metric_backend;
  }
  if (!self_query) {
    auto_rule = "cuda_auto_query_flat";
    auto_shape_group = cuda_auto_flat_ivf_shape_group(n, p);
    return cuda_auto_exact_backend(faiss_gpu_available, cuvs_available, cuda_available);
  }
  auto_shape_group = cuda_auto_flat_ivf_shape_group(n, p);
  const std::string ivf_backend = cuda_auto_ivf_backend(faiss_gpu_available, cuvs_available);
  const bool prefer_ivf = !ivf_backend.empty() && cuda_auto_prefers_ivf(
    self_query, n, p, n_points, k, work_size, target_recall_code, auto_shape_group
  );
  if (prefer_ivf) {
    auto_rule = auto_shape_group == "large_low_dim" ?
      "cuda_auto_ivf_large_low_dim_measured_fastest" :
      "cuda_auto_ivf_large_high_dim_lower_recall_tier";
    return ivf_backend;
  }
  if (target_recall_code >= 99 && auto_shape_group == "large_high_dim") {
    auto_rule = "cuda_auto_exact_large_high_dim_accuracy_tier";
  } else if (auto_shape_group == "small_n_very_high_dim" ||
             auto_shape_group == "small_n_high_dim" ||
             auto_shape_group == "small_high_dim" ||
             auto_shape_group == "medium_high_dim" ||
             auto_shape_group == "medium_low_dim") {
    auto_rule = "cuda_auto_exact_measured_shape_fastest";
  } else if (k <= 8 || n <= cuda_exact_n || work_size <= cuda_exact_work) {
    auto_rule = "cuda_auto_flat_vs_ivf_select_flat_small_or_exact";
  } else {
    auto_rule = "cuda_auto_flat_vs_ivf_select_flat_no_ivf_route";
  }
  return cuda_auto_exact_backend(faiss_gpu_available, cuvs_available, cuda_available);
}

std::string select_cpu(bool self_query,
                       int n,
                       int p,
                       int n_points,
                       int k,
                       double work_size,
                       const std::string& metric,
                       bool faiss_available,
                       double cpu_exact_work,
                       double cpu_faiss_flat_work,
                       int target_recall_code) {
  if (metric != "euclidean") {
    if ((metric == "cosine" || metric == "correlation") &&
        grid_self_knn(true, n, p, k, false, metric)) {
      return "cpu_grid";
    }
    const std::string flat_backend = cpu_metric_flat_backend(metric);
    if (!flat_backend.empty() && faiss_available &&
        work_size >= cpu_faiss_flat_work &&
        (!self_query || k < 10 || n < 5000)) {
      return flat_backend;
    }
    if (!self_query || work_size <= cpu_exact_work ||
        n < 5000 || k < 10 || p < 2) {
      return "cpu";
    }
    if (faiss_available) return "faiss_hnsw";
    if (native_nsg_fallback(self_query, n, p, k, work_size, metric)) {
      return "cpu_nsg";
    }
    if (native_nndescent_fallback(self_query, n, p, k, work_size)) {
      return "cpu_nndescent";
    }
    if (!flat_backend.empty() && faiss_available && work_size >= cpu_faiss_flat_work) {
      return flat_backend;
    }
    return "cpu";
  }
  if (grid_self_knn(self_query, n, p, k, false, "euclidean")) {
    return "cpu_grid";
  }
  if (work_size <= cpu_exact_work || n < 5000 || p < 2) {
    return "cpu";
  }
  if (self_query && n >= 500000 && p <= 64 && faiss_available) {
    if (n >= 2000000 && target_recall_code >= 95) return "faiss_hnsw";
    return "faiss_ivf";
  }
  if (faiss_available) return "faiss_hnsw";
  if (native_nsg_fallback(self_query, n, p, k, work_size, metric)) {
    return "cpu_nsg";
  }
  if (native_nndescent_fallback(self_query, n, p, k, work_size)) {
    return "cpu_nndescent";
  }
  return "cpu";
}

bool valid_int(int x) {
  return x != NA_INTEGER;
}

bool valid_double(double x) {
  return !NumericVector::is_na(x) && std::isfinite(x);
}

int clamp_int(int value, int min_value, int max_value) {
  if (max_value < min_value) max_value = min_value;
  return std::max(min_value, std::min(max_value, value));
}

int option_int(int value, int fallback, int min_value, int max_value) {
  if (!valid_int(value)) value = fallback;
  return clamp_int(value, min_value, max_value);
}

int requested_int(int value, int fallback) {
  return valid_int(value) ? value : fallback;
}

int safe_n(int n) {
  return valid_int(n) ? std::max(1, n) : 1;
}

int safe_p(int p) {
  return valid_int(p) ? std::max(1, p) : 1;
}

int safe_k(int k) {
  return valid_int(k) ? std::max(1, k) : 1;
}

double hnsw_target_recall_cpp(double target_recall) {
  if (!valid_double(target_recall)) return 0.99;
  if (target_recall <= 0.925) return 0.90;
  if (target_recall <= 0.975) return 0.95;
  return 0.99;
}

int hnsw_target_code_cpp(double target_recall) {
  target_recall = hnsw_target_recall_cpp(target_recall);
  if (target_recall <= 0.90) return 90;
  if (target_recall <= 0.95) return 95;
  return 99;
}

const char* hnsw_target_label_cpp(int target_code) {
  if (target_code <= 90) return "recall90";
  if (target_code <= 95) return "recall95";
  return "recall99";
}

std::string hnsw_shape_group_cpp(int n, int p) {
  if (!valid_int(n) || !valid_int(p)) return "other";
  if (n < 50000) return "small_n";
  if (n >= 50000 && n < 500000 && p <= 64) return "medium_low_dim";
  if (n >= 500000 && p <= 64) return "large_low_dim";
  if (n >= 50000 && p >= 256) return "large_high_dim";
  return "other";
}

std::string ivfpq_fastscan_shape_group_cpp(int n, int p) {
  if (!valid_int(n) || !valid_int(p)) return "other";
  if (n < 5000 && p >= 1024) return "small_high_dim";
  if (n < 50000 && p < 1024) return "small_low_dim";
  if (n >= 50000 && n < 500000 && p <= 64) return "medium_low_dim";
  if (n >= 500000 && p <= 64) return "large_low_dim";
  if (n >= 50000 && p >= 256) return "large_high_dim";
  return hnsw_shape_group_cpp(n, p);
}

int hnsw_cpu_k_bucket_cpp(int k) {
  k = safe_k(k);
  if (k <= 15) return 15;
  if (k <= 30) return 30;
  if (k <= 50) return 50;
  return 100;
}

struct HnswCpuTuningSpec {
  const char* shape_group;
  int k_bucket;
  int target_code;
  int m;
  int ef_construction;
  int ef_search;
  const char* benchmark_basis;
};

const HnswCpuTuningSpec* hnsw_cpu_benchmark_spec(const std::string& shape_group,
                                                 int k_bucket,
                                                 int target_code) {
  static const HnswCpuTuningSpec specs[] = {
    {"large_high_dim", 15, 90, 12, 80, 60, "hit_all_shape_datasets"},
    {"large_high_dim", 15, 95, 16, 100, 80, "hit_all_shape_datasets"},
    {"large_high_dim", 15, 99, 24, 160, 120, "hit_all_shape_datasets"},
    {"large_high_dim", 30, 90, 12, 80, 60, "hit_all_shape_datasets"},
    {"large_high_dim", 30, 95, 16, 100, 80, "hit_all_shape_datasets"},
    {"large_high_dim", 30, 99, 24, 160, 120, "hit_all_shape_datasets"},
    {"large_high_dim", 50, 90, 12, 80, 60, "hit_all_shape_datasets"},
    {"large_high_dim", 50, 95, 16, 100, 80, "hit_all_shape_datasets"},
    {"large_high_dim", 50, 99, 24, 160, 120, "hit_all_shape_datasets"},
    {"large_high_dim", 100, 90, 12, 60, 100, "hit_all_shape_datasets"},
    {"large_high_dim", 100, 95, 16, 100, 100, "hit_all_shape_datasets"},
    {"large_high_dim", 100, 99, 24, 160, 120, "hit_all_shape_datasets"},

    {"large_low_dim", 15, 90, 12, 60, 45, "hit_all_shape_datasets"},
    {"large_low_dim", 15, 95, 12, 80, 60, "hit_all_shape_datasets"},
    {"large_low_dim", 15, 99, 24, 160, 120, "hit_all_shape_datasets"},
    {"large_low_dim", 30, 90, 12, 80, 60, "hit_all_shape_datasets"},
    {"large_low_dim", 30, 95, 16, 100, 80, "hit_all_shape_datasets"},
    {"large_low_dim", 30, 99, 24, 160, 120, "hit_all_shape_datasets"},
    {"large_low_dim", 50, 90, 12, 80, 60, "hit_all_shape_datasets"},
    {"large_low_dim", 50, 95, 16, 100, 80, "hit_all_shape_datasets"},
    {"large_low_dim", 50, 99, 24, 160, 120, "hit_all_shape_datasets"},
    {"large_low_dim", 100, 90, 12, 80, 100, "hit_all_shape_datasets"},
    {"large_low_dim", 100, 95, 24, 160, 120, "hit_all_shape_datasets"},
    {"large_low_dim", 100, 99, 32, 240, 220, "hit_all_shape_datasets"},

    {"medium_low_dim", 15, 90, 10, 50, 35, "hit_all_shape_datasets"},
    {"medium_low_dim", 15, 95, 10, 50, 35, "hit_all_shape_datasets"},
    {"medium_low_dim", 15, 99, 12, 80, 60, "hit_all_shape_datasets"},
    {"medium_low_dim", 30, 90, 10, 50, 35, "hit_all_shape_datasets"},
    {"medium_low_dim", 30, 95, 10, 50, 35, "hit_all_shape_datasets"},
    {"medium_low_dim", 30, 99, 12, 80, 60, "hit_all_shape_datasets"},
    {"medium_low_dim", 50, 90, 6, 30, 50, "hit_all_shape_datasets"},
    {"medium_low_dim", 50, 95, 12, 80, 60, "hit_all_shape_datasets"},
    {"medium_low_dim", 50, 99, 12, 80, 60, "hit_all_shape_datasets"},
    {"medium_low_dim", 100, 90, 8, 30, 100, "hit_all_shape_datasets"},
    {"medium_low_dim", 100, 95, 8, 30, 100, "hit_all_shape_datasets"},
    {"medium_low_dim", 100, 99, 12, 60, 100, "hit_all_shape_datasets"},

    {"small_n", 15, 90, 6, 30, 15, "hit_all_shape_datasets"},
    {"small_n", 15, 95, 8, 40, 25, "hit_all_shape_datasets"},
    {"small_n", 15, 99, 12, 60, 45, "hit_all_shape_datasets"},
    {"small_n", 30, 90, 6, 30, 30, "hit_all_shape_datasets"},
    {"small_n", 30, 95, 8, 40, 30, "hit_all_shape_datasets"},
    {"small_n", 30, 99, 12, 80, 60, "hit_all_shape_datasets"},
    {"small_n", 50, 90, 12, 60, 50, "hit_all_shape_datasets"},
    {"small_n", 50, 95, 12, 60, 50, "hit_all_shape_datasets"},
    {"small_n", 50, 99, 12, 80, 60, "hit_all_shape_datasets"},
    {"small_n", 100, 90, 12, 60, 100, "hit_all_shape_datasets"},
    {"small_n", 100, 95, 12, 60, 100, "hit_all_shape_datasets"},
    {"small_n", 100, 99, 12, 80, 100, "hit_all_shape_datasets"}
  };
  for (const auto& spec : specs) {
    if (shape_group == spec.shape_group &&
        k_bucket == spec.k_bucket &&
        target_code == spec.target_code) {
      return &spec;
    }
  }
  return nullptr;
}

const HnswCpuTuningSpec* hnsw_cpu_cosine_benchmark_spec(const std::string& shape_group,
                                                        int k_bucket,
                                                        int target_code) {
  static const HnswCpuTuningSpec specs[] = {
    {"large_high_dim", 15, 90, 16, 100, 80, "fastest_meeting_target_hnsw_recall90_coverage_3of3"},
    {"large_high_dim", 15, 95, 24, 160, 120, "fastest_meeting_target_hnsw_recall95_coverage_3of3"},
    {"large_high_dim", 15, 99, 32, 240, 220, "fastest_meeting_target_hnsw_recall99_coverage_3of3"},
    {"large_high_dim", 30, 90, 16, 100, 80, "fastest_meeting_target_hnsw_recall90_coverage_3of3"},
    {"large_high_dim", 30, 95, 24, 160, 120, "fastest_meeting_target_hnsw_recall95_coverage_3of3"},
    {"large_high_dim", 30, 99, 32, 240, 220, "fastest_meeting_target_hnsw_recall99_coverage_3of3"},
    {"large_high_dim", 50, 90, 16, 100, 80, "fastest_meeting_target_hnsw_recall90_coverage_3of3"},
    {"large_high_dim", 50, 95, 24, 160, 120, "fastest_meeting_target_hnsw_recall95_coverage_3of3"},
    {"large_high_dim", 50, 99, 32, 240, 220, "fastest_meeting_target_hnsw_recall99_coverage_3of3"},
    {"large_high_dim", 100, 90, 16, 100, 100, "fastest_meeting_target_hnsw_recall90_coverage_3of3"},
    {"large_high_dim", 100, 95, 24, 160, 120, "fastest_meeting_target_hnsw_recall95_coverage_3of3"},
    {"large_high_dim", 100, 99, 32, 240, 220, "fastest_meeting_target_hnsw_recall99_coverage_3of3"},

    {"large_low_dim", 15, 90, 12, 60, 45, "fastest_meeting_target_hnsw_recall90_coverage_3of3"},
    {"large_low_dim", 15, 95, 12, 80, 60, "fastest_meeting_target_hnsw_recall95_coverage_3of3"},
    {"large_low_dim", 15, 99, 24, 160, 120, "fastest_meeting_target_hnsw_recall99_coverage_3of3"},
    {"large_low_dim", 30, 90, 12, 80, 60, "fastest_meeting_target_hnsw_recall90_coverage_3of3"},
    {"large_low_dim", 30, 95, 16, 100, 80, "fastest_meeting_target_hnsw_recall95_coverage_3of3"},
    {"large_low_dim", 30, 99, 24, 160, 120, "fastest_meeting_target_hnsw_recall99_coverage_3of3"},
    {"large_low_dim", 50, 90, 12, 80, 60, "fastest_meeting_target_hnsw_recall90_coverage_3of3"},
    {"large_low_dim", 50, 95, 16, 100, 80, "fastest_meeting_target_hnsw_recall95_coverage_3of3"},
    {"large_low_dim", 50, 99, 24, 160, 120, "fastest_meeting_target_hnsw_recall99_coverage_3of3"},
    {"large_low_dim", 100, 90, 12, 80, 100, "fastest_meeting_target_hnsw_recall90_coverage_3of3"},
    {"large_low_dim", 100, 95, 24, 160, 120, "fastest_meeting_target_hnsw_recall95_coverage_3of3"},
    {"large_low_dim", 100, 99, 32, 240, 220, "fastest_meeting_target_hnsw_recall99_coverage_3of3"},

    {"medium_low_dim", 15, 90, 8, 30, 20, "fastest_meeting_target_hnsw_recall90_coverage_1of1"},
    {"medium_low_dim", 15, 95, 10, 50, 35, "fastest_meeting_target_hnsw_recall95_coverage_1of1"},
    {"medium_low_dim", 15, 99, 12, 80, 60, "fastest_meeting_target_hnsw_recall99_coverage_1of1"},
    {"medium_low_dim", 30, 90, 8, 40, 30, "fastest_meeting_target_hnsw_recall90_coverage_1of1"},
    {"medium_low_dim", 30, 95, 10, 50, 35, "fastest_meeting_target_hnsw_recall95_coverage_1of1"},
    {"medium_low_dim", 30, 99, 12, 80, 60, "fastest_meeting_target_hnsw_recall99_coverage_1of1"},
    {"medium_low_dim", 50, 90, 8, 40, 50, "fastest_meeting_target_hnsw_recall90_coverage_1of1"},
    {"medium_low_dim", 50, 95, 8, 40, 50, "fastest_meeting_target_hnsw_recall95_coverage_1of1"},
    {"medium_low_dim", 50, 99, 16, 100, 80, "fastest_meeting_target_hnsw_recall99_coverage_1of1"},
    {"medium_low_dim", 100, 90, 6, 30, 100, "fastest_meeting_target_hnsw_recall90_coverage_1of1"},
    {"medium_low_dim", 100, 95, 8, 30, 100, "fastest_meeting_target_hnsw_recall95_coverage_1of1"},
    {"medium_low_dim", 100, 99, 12, 80, 100, "fastest_meeting_target_hnsw_recall99_coverage_1of1"},

    {"small_n", 15, 90, 6, 30, 15, "fastest_meeting_target_hnsw_recall90_coverage_3of3"},
    {"small_n", 15, 95, 8, 30, 20, "fastest_meeting_target_hnsw_recall95_coverage_3of3"},
    {"small_n", 15, 99, 24, 160, 120, "best_available_partial_shape_datasets_hnsw_recall99_coverage_2of3"},
    {"small_n", 30, 90, 12, 60, 45, "fastest_meeting_target_hnsw_recall90_coverage_3of3"},
    {"small_n", 30, 95, 12, 60, 45, "fastest_meeting_target_hnsw_recall95_coverage_3of3"},
    {"small_n", 30, 99, 12, 80, 60, "fastest_meeting_target_hnsw_recall99_coverage_3of3"},
    {"small_n", 50, 90, 8, 30, 50, "fastest_meeting_target_hnsw_recall90_coverage_3of3"},
    {"small_n", 50, 95, 8, 30, 50, "fastest_meeting_target_hnsw_recall95_coverage_3of3"},
    {"small_n", 50, 99, 12, 80, 60, "fastest_meeting_target_hnsw_recall99_coverage_3of3"},
    {"small_n", 100, 90, 8, 30, 100, "fastest_meeting_target_hnsw_recall90_coverage_3of3"},
    {"small_n", 100, 95, 8, 30, 100, "fastest_meeting_target_hnsw_recall95_coverage_3of3"},
    {"small_n", 100, 99, 16, 100, 100, "fastest_meeting_target_hnsw_recall99_coverage_3of3"}
  };
  for (const auto& spec : specs) {
    if (shape_group == spec.shape_group &&
        k_bucket == spec.k_bucket &&
        target_code == spec.target_code) {
      return &spec;
    }
  }
  return nullptr;
}

const HnswCpuTuningSpec* hnsw_cpu_correlation_benchmark_spec(const std::string& shape_group,
                                                             int k_bucket,
                                                             int target_code) {
  static const HnswCpuTuningSpec specs[] = {
    {"large_high_dim", 15, 90, 16, 100, 80, "fastest_meeting_target_hnsw_recall90_coverage_3of3"},
    {"large_high_dim", 15, 95, 24, 160, 120, "fastest_meeting_target_hnsw_recall95_coverage_3of3"},
    {"large_high_dim", 15, 99, 32, 240, 220, "fastest_meeting_target_hnsw_recall99_coverage_3of3"},
    {"large_high_dim", 30, 90, 16, 100, 80, "fastest_meeting_target_hnsw_recall90_coverage_3of3"},
    {"large_high_dim", 30, 95, 24, 160, 120, "fastest_meeting_target_hnsw_recall95_coverage_3of3"},
    {"large_high_dim", 30, 99, 32, 240, 220, "fastest_meeting_target_hnsw_recall99_coverage_3of3"},
    {"large_high_dim", 50, 90, 16, 100, 80, "fastest_meeting_target_hnsw_recall90_coverage_3of3"},
    {"large_high_dim", 50, 95, 24, 160, 120, "fastest_meeting_target_hnsw_recall95_coverage_3of3"},
    {"large_high_dim", 50, 99, 32, 240, 220, "fastest_meeting_target_hnsw_recall99_coverage_3of3"},
    {"large_high_dim", 100, 90, 16, 100, 100, "fastest_meeting_target_hnsw_recall90_coverage_3of3"},
    {"large_high_dim", 100, 95, 24, 160, 120, "fastest_meeting_target_hnsw_recall95_coverage_3of3"},
    {"large_high_dim", 100, 99, 32, 240, 220, "fastest_meeting_target_hnsw_recall99_coverage_3of3"},

    {"large_low_dim", 15, 90, 12, 60, 45, "fastest_meeting_target_hnsw_recall90_coverage_3of3"},
    {"large_low_dim", 15, 95, 12, 80, 60, "fastest_meeting_target_hnsw_recall95_coverage_3of3"},
    {"large_low_dim", 15, 99, 24, 160, 120, "fastest_meeting_target_hnsw_recall99_coverage_3of3"},
    {"large_low_dim", 30, 90, 12, 80, 60, "fastest_meeting_target_hnsw_recall90_coverage_3of3"},
    {"large_low_dim", 30, 95, 16, 100, 80, "fastest_meeting_target_hnsw_recall95_coverage_3of3"},
    {"large_low_dim", 30, 99, 24, 160, 120, "fastest_meeting_target_hnsw_recall99_coverage_3of3"},
    {"large_low_dim", 50, 90, 12, 80, 60, "fastest_meeting_target_hnsw_recall90_coverage_3of3"},
    {"large_low_dim", 50, 95, 16, 100, 80, "fastest_meeting_target_hnsw_recall95_coverage_3of3"},
    {"large_low_dim", 50, 99, 24, 160, 120, "fastest_meeting_target_hnsw_recall99_coverage_3of3"},
    {"large_low_dim", 100, 90, 12, 60, 100, "fastest_meeting_target_hnsw_recall90_coverage_3of3"},
    {"large_low_dim", 100, 95, 24, 160, 120, "fastest_meeting_target_hnsw_recall95_coverage_3of3"},
    {"large_low_dim", 100, 99, 32, 240, 220, "fastest_meeting_target_hnsw_recall99_coverage_3of3"},

    {"medium_low_dim", 15, 90, 8, 30, 20, "fastest_meeting_target_hnsw_recall90_coverage_1of1"},
    {"medium_low_dim", 15, 95, 8, 40, 25, "fastest_meeting_target_hnsw_recall95_coverage_1of1"},
    {"medium_low_dim", 15, 99, 16, 100, 80, "fastest_meeting_target_hnsw_recall99_coverage_1of1"},
    {"medium_low_dim", 30, 90, 8, 30, 30, "fastest_meeting_target_hnsw_recall90_coverage_1of1"},
    {"medium_low_dim", 30, 95, 10, 50, 35, "fastest_meeting_target_hnsw_recall95_coverage_1of1"},
    {"medium_low_dim", 30, 99, 16, 100, 80, "fastest_meeting_target_hnsw_recall99_coverage_1of1"},
    {"medium_low_dim", 50, 90, 8, 30, 50, "fastest_meeting_target_hnsw_recall90_coverage_1of1"},
    {"medium_low_dim", 50, 95, 10, 50, 50, "fastest_meeting_target_hnsw_recall95_coverage_1of1"},
    {"medium_low_dim", 50, 99, 16, 100, 80, "fastest_meeting_target_hnsw_recall99_coverage_1of1"},
    {"medium_low_dim", 100, 90, 8, 40, 100, "fastest_meeting_target_hnsw_recall90_coverage_1of1"},
    {"medium_low_dim", 100, 95, 8, 40, 100, "fastest_meeting_target_hnsw_recall95_coverage_1of1"},
    {"medium_low_dim", 100, 99, 16, 100, 100, "fastest_meeting_target_hnsw_recall99_coverage_1of1"},

    {"small_n", 15, 90, 8, 30, 20, "fastest_meeting_target_hnsw_recall90_coverage_3of3"},
    {"small_n", 15, 95, 8, 30, 20, "fastest_meeting_target_hnsw_recall95_coverage_3of3"},
    {"small_n", 15, 99, 48, 320, 400, "best_recall_below_target_hnsw_recall99_coverage_3of3"},
    {"small_n", 30, 90, 8, 30, 30, "fastest_meeting_target_hnsw_recall90_coverage_3of3"},
    {"small_n", 30, 95, 8, 30, 30, "fastest_meeting_target_hnsw_recall95_coverage_3of3"},
    {"small_n", 30, 99, 16, 100, 80, "fastest_meeting_target_hnsw_recall99_coverage_3of3"},
    {"small_n", 50, 90, 6, 30, 50, "fastest_meeting_target_hnsw_recall90_coverage_3of3"},
    {"small_n", 50, 95, 6, 30, 50, "fastest_meeting_target_hnsw_recall95_coverage_3of3"},
    {"small_n", 50, 99, 12, 80, 60, "fastest_meeting_target_hnsw_recall99_coverage_3of3"},
    {"small_n", 100, 90, 10, 50, 100, "fastest_meeting_target_hnsw_recall90_coverage_3of3"},
    {"small_n", 100, 95, 10, 50, 100, "fastest_meeting_target_hnsw_recall95_coverage_3of3"},
    {"small_n", 100, 99, 12, 80, 100, "fastest_meeting_target_hnsw_recall99_coverage_3of3"}
  };
  for (const auto& spec : specs) {
    if (shape_group == spec.shape_group &&
        k_bucket == spec.k_bucket &&
        target_code == spec.target_code) {
      return &spec;
    }
  }
  return nullptr;
}

const HnswCpuTuningSpec* hnsw_cpu_inner_product_benchmark_spec(const std::string& shape_group,
                                                               int k_bucket,
                                                               int target_code) {
  if (const auto* override_spec = jmlr_match_shape_spec(
        jmlr_cpu_hnsw_inner_product_specs,
        shape_group,
        k_bucket,
        target_code)) {
    static thread_local HnswCpuTuningSpec adapted;
    adapted = {
      override_spec->shape_group,
      override_spec->k_bucket,
      override_spec->target_code,
      override_spec->m,
      override_spec->ef_construction,
      override_spec->ef_search,
      override_spec->basis
    };
    return &adapted;
  }
  static const HnswCpuTuningSpec specs[] = {
    {"large_high_dim", 15, 90, 48, 320, 400, "best_recall_below_target_hnsw_recall90_coverage_3of3"},
    {"large_high_dim", 15, 95, 48, 320, 400, "best_recall_below_target_hnsw_recall95_coverage_3of3"},
    {"large_high_dim", 15, 99, 48, 320, 400, "best_recall_below_target_hnsw_recall99_coverage_3of3"},
    {"large_high_dim", 30, 90, 48, 240, 220, "best_recall_below_target_hnsw_recall90_coverage_3of3"},
    {"large_high_dim", 30, 95, 48, 240, 220, "best_recall_below_target_hnsw_recall95_coverage_3of3"},
    {"large_high_dim", 30, 99, 48, 240, 220, "best_recall_below_target_hnsw_recall99_coverage_3of3"},
    {"large_high_dim", 50, 90, 48, 320, 400, "best_recall_below_target_hnsw_recall90_coverage_3of3"},
    {"large_high_dim", 50, 95, 48, 320, 400, "best_recall_below_target_hnsw_recall95_coverage_3of3"},
    {"large_high_dim", 50, 99, 48, 320, 400, "best_recall_below_target_hnsw_recall99_coverage_3of3"},
    {"large_high_dim", 100, 90, 48, 320, 400, "best_recall_below_target_hnsw_recall90_coverage_3of3"},
    {"large_high_dim", 100, 95, 48, 320, 400, "best_recall_below_target_hnsw_recall95_coverage_3of3"},
    {"large_high_dim", 100, 99, 48, 320, 400, "best_recall_below_target_hnsw_recall99_coverage_3of3"},

    {"large_low_dim", 15, 90, 48, 320, 400, "best_recall_below_target_hnsw_recall90_coverage_3of3"},
    {"large_low_dim", 15, 95, 48, 320, 400, "best_recall_below_target_hnsw_recall95_coverage_3of3"},
    {"large_low_dim", 15, 99, 48, 320, 400, "best_recall_below_target_hnsw_recall99_coverage_3of3"},
    {"large_low_dim", 30, 90, 48, 320, 400, "best_recall_below_target_hnsw_recall90_coverage_3of3"},
    {"large_low_dim", 30, 95, 48, 320, 400, "best_recall_below_target_hnsw_recall95_coverage_3of3"},
    {"large_low_dim", 30, 99, 48, 320, 400, "best_recall_below_target_hnsw_recall99_coverage_3of3"},
    {"large_low_dim", 50, 90, 48, 320, 400, "best_recall_below_target_hnsw_recall90_coverage_3of3"},
    {"large_low_dim", 50, 95, 48, 320, 400, "best_recall_below_target_hnsw_recall95_coverage_3of3"},
    {"large_low_dim", 50, 99, 48, 320, 400, "best_recall_below_target_hnsw_recall99_coverage_3of3"},
    {"large_low_dim", 100, 90, 48, 320, 400, "best_recall_below_target_hnsw_recall90_coverage_3of3"},
    {"large_low_dim", 100, 95, 48, 320, 400, "best_recall_below_target_hnsw_recall95_coverage_3of3"},
    {"large_low_dim", 100, 99, 48, 320, 400, "best_recall_below_target_hnsw_recall99_coverage_3of3"},

    {"medium_low_dim", 15, 90, 32, 160, 120, "fastest_meeting_target_hnsw_recall90_coverage_1of1"},
    {"medium_low_dim", 15, 95, 32, 240, 220, "fastest_meeting_target_hnsw_recall95_coverage_1of1"},
    {"medium_low_dim", 15, 99, 64, 480, 720, "best_recall_below_target_hnsw_recall99_coverage_1of1"},
    {"medium_low_dim", 30, 90, 32, 200, 150, "fastest_meeting_target_hnsw_recall90_coverage_1of1"},
    {"medium_low_dim", 30, 95, 48, 320, 400, "fastest_meeting_target_hnsw_recall95_coverage_1of1"},
    {"medium_low_dim", 30, 99, 64, 480, 720, "best_recall_below_target_hnsw_recall99_coverage_1of1"},
    {"medium_low_dim", 50, 90, 32, 240, 220, "fastest_meeting_target_hnsw_recall90_coverage_1of1"},
    {"medium_low_dim", 50, 95, 48, 320, 400, "fastest_meeting_target_hnsw_recall95_coverage_1of1"},
    {"medium_low_dim", 50, 99, 64, 480, 720, "best_recall_below_target_hnsw_recall99_coverage_1of1"},
    {"medium_low_dim", 100, 90, 48, 320, 400, "fastest_meeting_target_hnsw_recall90_coverage_1of1"},
    {"medium_low_dim", 100, 95, 64, 480, 720, "fastest_meeting_target_hnsw_recall95_coverage_1of1"},
    {"medium_low_dim", 100, 99, 64, 480, 720, "best_recall_below_target_hnsw_recall99_coverage_1of1"},

    {"small_n", 15, 90, 32, 240, 220, "fastest_meeting_target_hnsw_recall90_coverage_3of3"},
    {"small_n", 15, 95, 48, 320, 400, "fastest_meeting_target_hnsw_recall95_coverage_3of3"},
    {"small_n", 15, 99, 64, 480, 720, "best_recall_below_target_hnsw_recall99_coverage_3of3"},
    {"small_n", 30, 90, 48, 240, 220, "fastest_meeting_target_hnsw_recall90_coverage_3of3"},
    {"small_n", 30, 95, 64, 480, 720, "fastest_meeting_target_hnsw_recall95_coverage_3of3"},
    {"small_n", 30, 99, 64, 480, 720, "best_recall_below_target_hnsw_recall99_coverage_3of3"},
    {"small_n", 50, 90, 48, 320, 400, "fastest_meeting_target_hnsw_recall90_coverage_3of3"},
    {"small_n", 50, 95, 64, 480, 720, "best_recall_below_target_hnsw_recall95_coverage_3of3"},
    {"small_n", 50, 99, 64, 480, 720, "best_recall_below_target_hnsw_recall99_coverage_3of3"},
    {"small_n", 100, 90, 64, 480, 720, "fastest_meeting_target_hnsw_recall90_coverage_3of3"},
    {"small_n", 100, 95, 64, 480, 720, "best_recall_below_target_hnsw_recall95_coverage_3of3"},
    {"small_n", 100, 99, 64, 480, 720, "best_recall_below_target_hnsw_recall99_coverage_3of3"}
  };
  for (const auto& spec : specs) {
    if (shape_group == spec.shape_group &&
        k_bucket == spec.k_bucket &&
        target_code == spec.target_code) {
      return &spec;
    }
  }
  return nullptr;
}

std::string hnsw_cuda_shape_group_cpp(int n, int p) {
  return hnsw_shape_group_cpp(n, p);
}

struct HnswCudaTuningSpec {
  const char* shape_group;
  int k_bucket;
  int target_code;
  int graph_degree;
  int intermediate_graph_degree;
  int ef;
  const char* benchmark_basis;
};

const HnswCudaTuningSpec* hnsw_cuda_euclidean_benchmark_spec(const std::string& shape_group,
                                                             int k_bucket,
                                                             int target_code) {
  // Source: benchmark_scripts/cuda_hnsw_euclidean_shape_tuning_defaults_from_uploaded_results.csv.
  static const HnswCudaTuningSpec specs[] = {
    {"large_high_dim", 15, 90, 96, 320, 480, "best_available_all_shape_datasets"},
    {"large_high_dim", 15, 95, 96, 320, 480, "best_available_all_shape_datasets"},
    {"large_high_dim", 15, 99, 96, 320, 480, "best_available_all_shape_datasets"},
    {"large_high_dim", 30, 90, 96, 320, 480, "best_available_all_shape_datasets"},
    {"large_high_dim", 30, 95, 96, 320, 480, "best_available_all_shape_datasets"},
    {"large_high_dim", 30, 99, 96, 320, 480, "best_available_all_shape_datasets"},
    {"large_high_dim", 50, 90, 96, 320, 480, "best_available_all_shape_datasets"},
    {"large_high_dim", 50, 95, 96, 320, 480, "best_available_all_shape_datasets"},
    {"large_high_dim", 50, 99, 96, 320, 480, "best_available_all_shape_datasets"},
    {"large_high_dim", 100, 90, 96, 320, 480, "best_available_all_shape_datasets"},
    {"large_high_dim", 100, 95, 96, 320, 480, "best_available_all_shape_datasets"},
    {"large_high_dim", 100, 99, 96, 320, 480, "best_available_all_shape_datasets"},

    {"large_low_dim", 15, 90, 8, 16, 24, "fastest_meeting_target_all_shape_datasets"},
    {"large_low_dim", 15, 95, 24, 48, 64, "fastest_meeting_target_all_shape_datasets"},
    {"large_low_dim", 15, 99, 32, 64, 96, "fastest_meeting_target_all_shape_datasets"},
    {"large_low_dim", 30, 90, 8, 16, 30, "fastest_meeting_target_all_shape_datasets"},
    {"large_low_dim", 30, 95, 12, 24, 32, "fastest_meeting_target_all_shape_datasets"},
    {"large_low_dim", 30, 99, 32, 64, 96, "fastest_meeting_target_all_shape_datasets"},
    {"large_low_dim", 50, 90, 16, 32, 50, "fastest_meeting_target_all_shape_datasets"},
    {"large_low_dim", 50, 95, 16, 32, 50, "fastest_meeting_target_all_shape_datasets"},
    {"large_low_dim", 50, 99, 32, 64, 96, "fastest_meeting_target_all_shape_datasets"},
    {"large_low_dim", 100, 90, 12, 24, 100, "fastest_meeting_target_all_shape_datasets"},
    {"large_low_dim", 100, 95, 12, 24, 100, "fastest_meeting_target_all_shape_datasets"},
    {"large_low_dim", 100, 99, 12, 24, 100, "fastest_meeting_target_all_shape_datasets"},

    {"medium_low_dim", 15, 90, 8, 16, 24, "fastest_meeting_target_all_shape_datasets"},
    {"medium_low_dim", 15, 95, 8, 16, 24, "fastest_meeting_target_all_shape_datasets"},
    {"medium_low_dim", 15, 99, 24, 48, 64, "fastest_meeting_target_all_shape_datasets"},
    {"medium_low_dim", 30, 90, 8, 16, 30, "fastest_meeting_target_all_shape_datasets"},
    {"medium_low_dim", 30, 95, 8, 16, 30, "fastest_meeting_target_all_shape_datasets"},
    {"medium_low_dim", 30, 99, 16, 32, 48, "fastest_meeting_target_all_shape_datasets"},
    {"medium_low_dim", 50, 90, 16, 32, 50, "fastest_meeting_target_all_shape_datasets"},
    {"medium_low_dim", 50, 95, 16, 32, 50, "fastest_meeting_target_all_shape_datasets"},
    {"medium_low_dim", 50, 99, 16, 32, 50, "fastest_meeting_target_all_shape_datasets"},
    {"medium_low_dim", 100, 90, 8, 16, 100, "fastest_meeting_target_all_shape_datasets"},
    {"medium_low_dim", 100, 95, 8, 16, 100, "fastest_meeting_target_all_shape_datasets"},
    {"medium_low_dim", 100, 99, 8, 16, 100, "fastest_meeting_target_all_shape_datasets"},

    {"small_n", 15, 90, 128, 512, 768, "best_available_all_shape_datasets"},
    {"small_n", 15, 95, 128, 512, 768, "best_available_all_shape_datasets"},
    {"small_n", 15, 99, 128, 512, 768, "best_available_all_shape_datasets"},
    {"small_n", 30, 90, 128, 512, 768, "best_available_all_shape_datasets"},
    {"small_n", 30, 95, 128, 512, 768, "best_available_all_shape_datasets"},
    {"small_n", 30, 99, 128, 512, 768, "best_available_all_shape_datasets"},
    {"small_n", 50, 90, 128, 512, 768, "best_available_all_shape_datasets"},
    {"small_n", 50, 95, 128, 512, 768, "best_available_all_shape_datasets"},
    {"small_n", 50, 99, 128, 512, 768, "best_available_all_shape_datasets"},
    {"small_n", 100, 90, 128, 512, 768, "best_available_all_shape_datasets"},
    {"small_n", 100, 95, 128, 512, 768, "best_available_all_shape_datasets"},
    {"small_n", 100, 99, 128, 512, 768, "best_available_all_shape_datasets"}
  };
  for (const auto& spec : specs) {
    if (shape_group == spec.shape_group &&
        k_bucket == spec.k_bucket &&
        target_code == spec.target_code) {
      return &spec;
    }
  }
  return nullptr;
}

const HnswCudaTuningSpec* hnsw_cuda_cosine_benchmark_spec(const std::string& shape_group,
                                                         int k_bucket,
                                                         int target_code) {
  // Source: benchmark_scripts/cuda_hnsw_cosine_shape_tuning_defaults_from_uploaded_results.csv.
  static const HnswCudaTuningSpec specs[] = {
    {"large_high_dim", 15, 90, 48, 128, 128, "fastest_meeting_target_all_shape_datasets"},
    {"large_high_dim", 15, 95, 64, 192, 256, "fastest_meeting_target_all_shape_datasets"},
    {"large_high_dim", 15, 99, 12, 24, 32, "best_available_all_shape_datasets"},
    {"large_high_dim", 30, 90, 48, 128, 128, "fastest_meeting_target_all_shape_datasets"},
    {"large_high_dim", 30, 95, 64, 192, 256, "fastest_meeting_target_all_shape_datasets"},
    {"large_high_dim", 30, 99, 64, 192, 256, "fastest_meeting_target_all_shape_datasets"},
    {"large_high_dim", 50, 90, 32, 64, 96, "fastest_meeting_target_all_shape_datasets"},
    {"large_high_dim", 50, 95, 64, 192, 256, "fastest_meeting_target_all_shape_datasets"},
    {"large_high_dim", 50, 99, 96, 320, 480, "fastest_meeting_target_all_shape_datasets"},
    {"large_high_dim", 100, 90, 24, 48, 100, "fastest_meeting_target_all_shape_datasets"},
    {"large_high_dim", 100, 95, 64, 192, 256, "fastest_meeting_target_all_shape_datasets"},
    {"large_high_dim", 100, 99, 96, 320, 480, "fastest_meeting_target_all_shape_datasets"},

    {"large_low_dim", 15, 90, 8, 16, 24, "fastest_meeting_target_all_shape_datasets"},
    {"large_low_dim", 15, 95, 24, 48, 64, "fastest_meeting_target_all_shape_datasets"},
    {"large_low_dim", 15, 99, 64, 192, 256, "fastest_meeting_target_all_shape_datasets"},
    {"large_low_dim", 30, 90, 8, 16, 30, "fastest_meeting_target_all_shape_datasets"},
    {"large_low_dim", 30, 95, 8, 16, 30, "fastest_meeting_target_all_shape_datasets"},
    {"large_low_dim", 30, 99, 32, 64, 96, "fastest_meeting_target_all_shape_datasets"},
    {"large_low_dim", 50, 90, 12, 24, 50, "fastest_meeting_target_all_shape_datasets"},
    {"large_low_dim", 50, 95, 12, 24, 50, "fastest_meeting_target_all_shape_datasets"},
    {"large_low_dim", 50, 99, 48, 128, 128, "fastest_meeting_target_all_shape_datasets"},
    {"large_low_dim", 100, 90, 12, 24, 100, "fastest_meeting_target_all_shape_datasets"},
    {"large_low_dim", 100, 95, 12, 24, 100, "fastest_meeting_target_all_shape_datasets"},
    {"large_low_dim", 100, 99, 8, 16, 100, "fastest_meeting_target_all_shape_datasets"},

    {"medium_low_dim", 15, 90, 8, 16, 24, "fastest_meeting_target_all_shape_datasets"},
    {"medium_low_dim", 15, 95, 8, 16, 24, "fastest_meeting_target_all_shape_datasets"},
    {"medium_low_dim", 15, 99, 24, 48, 64, "fastest_meeting_target_all_shape_datasets"},
    {"medium_low_dim", 30, 90, 8, 16, 30, "fastest_meeting_target_all_shape_datasets"},
    {"medium_low_dim", 30, 95, 8, 16, 30, "fastest_meeting_target_all_shape_datasets"},
    {"medium_low_dim", 30, 99, 8, 16, 30, "fastest_meeting_target_all_shape_datasets"},
    {"medium_low_dim", 50, 90, 8, 16, 50, "fastest_meeting_target_all_shape_datasets"},
    {"medium_low_dim", 50, 95, 8, 16, 50, "fastest_meeting_target_all_shape_datasets"},
    {"medium_low_dim", 50, 99, 8, 16, 50, "fastest_meeting_target_all_shape_datasets"},
    {"medium_low_dim", 100, 90, 8, 16, 100, "fastest_meeting_target_all_shape_datasets"},
    {"medium_low_dim", 100, 95, 8, 16, 100, "fastest_meeting_target_all_shape_datasets"},
    {"medium_low_dim", 100, 99, 8, 16, 100, "fastest_meeting_target_all_shape_datasets"},

    {"small_n", 15, 90, 12, 24, 32, "fastest_meeting_target_all_shape_datasets"},
    {"small_n", 15, 95, 12, 24, 32, "fastest_meeting_target_all_shape_datasets"},
    {"small_n", 15, 99, 32, 64, 96, "best_available_all_shape_datasets"},
    {"small_n", 30, 90, 16, 32, 48, "fastest_meeting_target_all_shape_datasets"},
    {"small_n", 30, 95, 16, 32, 48, "fastest_meeting_target_all_shape_datasets"},
    {"small_n", 30, 99, 16, 32, 48, "fastest_meeting_target_all_shape_datasets"},
    {"small_n", 50, 90, 16, 32, 50, "fastest_meeting_target_all_shape_datasets"},
    {"small_n", 50, 95, 16, 32, 50, "fastest_meeting_target_all_shape_datasets"},
    {"small_n", 50, 99, 16, 32, 50, "fastest_meeting_target_all_shape_datasets"},
    {"small_n", 100, 90, 16, 32, 100, "fastest_meeting_target_all_shape_datasets"},
    {"small_n", 100, 95, 16, 32, 100, "fastest_meeting_target_all_shape_datasets"},
    {"small_n", 100, 99, 16, 32, 100, "fastest_meeting_target_all_shape_datasets"}
  };
  for (const auto& spec : specs) {
    if (shape_group == spec.shape_group &&
        k_bucket == spec.k_bucket &&
        target_code == spec.target_code) {
      return &spec;
    }
  }
  return nullptr;
}

const HnswCudaTuningSpec* hnsw_cuda_correlation_benchmark_spec(const std::string& shape_group,
                                                              int k_bucket,
                                                              int target_code) {
  // Source: benchmark_scripts/cuda_hnsw_correlation_shape_tuning_defaults_from_uploaded_results.csv.
  static const HnswCudaTuningSpec specs[] = {
    {"large_high_dim", 15, 90, 48, 128, 128, "fastest_meeting_target_all_shape_datasets"},
    {"large_high_dim", 15, 95, 64, 192, 256, "fastest_meeting_target_all_shape_datasets"},
    {"large_high_dim", 15, 99, 96, 320, 480, "best_available_all_shape_datasets"},
    {"large_high_dim", 30, 90, 48, 128, 128, "fastest_meeting_target_all_shape_datasets"},
    {"large_high_dim", 30, 95, 64, 192, 256, "fastest_meeting_target_all_shape_datasets"},
    {"large_high_dim", 30, 99, 96, 320, 480, "best_available_all_shape_datasets"},
    {"large_high_dim", 50, 90, 24, 48, 64, "fastest_meeting_target_all_shape_datasets"},
    {"large_high_dim", 50, 95, 64, 192, 256, "fastest_meeting_target_all_shape_datasets"},
    {"large_high_dim", 50, 99, 96, 320, 480, "best_available_all_shape_datasets"},
    {"large_high_dim", 100, 90, 16, 32, 100, "fastest_meeting_target_all_shape_datasets"},
    {"large_high_dim", 100, 95, 64, 192, 256, "fastest_meeting_target_all_shape_datasets"},
    {"large_high_dim", 100, 99, 96, 320, 480, "fastest_meeting_target_all_shape_datasets"},

    {"large_low_dim", 15, 90, 8, 16, 24, "fastest_meeting_target_all_shape_datasets"},
    {"large_low_dim", 15, 95, 24, 48, 64, "fastest_meeting_target_all_shape_datasets"},
    {"large_low_dim", 15, 99, 32, 64, 96, "fastest_meeting_target_all_shape_datasets"},
    {"large_low_dim", 30, 90, 8, 16, 30, "fastest_meeting_target_all_shape_datasets"},
    {"large_low_dim", 30, 95, 8, 16, 30, "fastest_meeting_target_all_shape_datasets"},
    {"large_low_dim", 30, 99, 32, 64, 96, "fastest_meeting_target_all_shape_datasets"},
    {"large_low_dim", 50, 90, 8, 16, 50, "fastest_meeting_target_all_shape_datasets"},
    {"large_low_dim", 50, 95, 8, 16, 50, "fastest_meeting_target_all_shape_datasets"},
    {"large_low_dim", 50, 99, 32, 64, 96, "fastest_meeting_target_all_shape_datasets"},
    {"large_low_dim", 100, 90, 8, 16, 100, "fastest_meeting_target_all_shape_datasets"},
    {"large_low_dim", 100, 95, 8, 16, 100, "fastest_meeting_target_all_shape_datasets"},
    {"large_low_dim", 100, 99, 8, 16, 100, "fastest_meeting_target_all_shape_datasets"},

    {"medium_low_dim", 15, 90, 8, 16, 24, "fastest_meeting_target_all_shape_datasets"},
    {"medium_low_dim", 15, 95, 8, 16, 24, "fastest_meeting_target_all_shape_datasets"},
    {"medium_low_dim", 15, 99, 16, 32, 48, "fastest_meeting_target_all_shape_datasets"},
    {"medium_low_dim", 30, 90, 12, 24, 32, "fastest_meeting_target_all_shape_datasets"},
    {"medium_low_dim", 30, 95, 12, 24, 32, "fastest_meeting_target_all_shape_datasets"},
    {"medium_low_dim", 30, 99, 8, 16, 30, "fastest_meeting_target_all_shape_datasets"},
    {"medium_low_dim", 50, 90, 16, 32, 50, "fastest_meeting_target_all_shape_datasets"},
    {"medium_low_dim", 50, 95, 16, 32, 50, "fastest_meeting_target_all_shape_datasets"},
    {"medium_low_dim", 50, 99, 16, 32, 50, "fastest_meeting_target_all_shape_datasets"},
    {"medium_low_dim", 100, 90, 12, 24, 100, "fastest_meeting_target_all_shape_datasets"},
    {"medium_low_dim", 100, 95, 12, 24, 100, "fastest_meeting_target_all_shape_datasets"},
    {"medium_low_dim", 100, 99, 12, 24, 100, "fastest_meeting_target_all_shape_datasets"},

    {"small_n", 15, 90, 8, 16, 24, "fastest_meeting_target_all_shape_datasets"},
    {"small_n", 15, 95, 12, 24, 32, "fastest_meeting_target_all_shape_datasets"},
    {"small_n", 15, 99, 24, 48, 64, "best_available_all_shape_datasets"},
    {"small_n", 30, 90, 12, 24, 32, "fastest_meeting_target_all_shape_datasets"},
    {"small_n", 30, 95, 12, 24, 32, "fastest_meeting_target_all_shape_datasets"},
    {"small_n", 30, 99, 12, 24, 32, "fastest_meeting_target_all_shape_datasets"},
    {"small_n", 50, 90, 16, 32, 50, "fastest_meeting_target_all_shape_datasets"},
    {"small_n", 50, 95, 16, 32, 50, "fastest_meeting_target_all_shape_datasets"},
    {"small_n", 50, 99, 16, 32, 50, "fastest_meeting_target_all_shape_datasets"},
    {"small_n", 100, 90, 16, 32, 100, "fastest_meeting_target_all_shape_datasets"},
    {"small_n", 100, 95, 16, 32, 100, "fastest_meeting_target_all_shape_datasets"},
    {"small_n", 100, 99, 16, 32, 100, "fastest_meeting_target_all_shape_datasets"}
  };
  for (const auto& spec : specs) {
    if (shape_group == spec.shape_group &&
        k_bucket == spec.k_bucket &&
        target_code == spec.target_code) {
      return &spec;
    }
  }
  return nullptr;
}

const HnswCudaTuningSpec* hnsw_cuda_inner_product_benchmark_spec(const std::string& shape_group,
                                                                int k_bucket,
                                                                int target_code) {
  // Source: benchmark_scripts/cuda_hnsw_inner_product_shape_tuning_defaults_from_seeded_euclidean_results.csv.
  static const HnswCudaTuningSpec specs[] = {
    {"large_high_dim", 15, 90, 96, 320, 480, "seeded_from_cuda_hnsw_euclidean_pending_inner_product_sweep"},
    {"large_high_dim", 15, 95, 96, 320, 480, "seeded_from_cuda_hnsw_euclidean_pending_inner_product_sweep"},
    {"large_high_dim", 15, 99, 96, 320, 480, "seeded_from_cuda_hnsw_euclidean_pending_inner_product_sweep"},
    {"large_high_dim", 30, 90, 96, 320, 480, "seeded_from_cuda_hnsw_euclidean_pending_inner_product_sweep"},
    {"large_high_dim", 30, 95, 96, 320, 480, "seeded_from_cuda_hnsw_euclidean_pending_inner_product_sweep"},
    {"large_high_dim", 30, 99, 96, 320, 480, "seeded_from_cuda_hnsw_euclidean_pending_inner_product_sweep"},
    {"large_high_dim", 50, 90, 96, 320, 480, "seeded_from_cuda_hnsw_euclidean_pending_inner_product_sweep"},
    {"large_high_dim", 50, 95, 96, 320, 480, "seeded_from_cuda_hnsw_euclidean_pending_inner_product_sweep"},
    {"large_high_dim", 50, 99, 96, 320, 480, "seeded_from_cuda_hnsw_euclidean_pending_inner_product_sweep"},
    {"large_high_dim", 100, 90, 96, 320, 480, "seeded_from_cuda_hnsw_euclidean_pending_inner_product_sweep"},
    {"large_high_dim", 100, 95, 96, 320, 480, "seeded_from_cuda_hnsw_euclidean_pending_inner_product_sweep"},
    {"large_high_dim", 100, 99, 96, 320, 480, "seeded_from_cuda_hnsw_euclidean_pending_inner_product_sweep"},

    {"large_low_dim", 15, 90, 8, 16, 24, "seeded_from_cuda_hnsw_euclidean_pending_inner_product_sweep"},
    {"large_low_dim", 15, 95, 24, 48, 64, "seeded_from_cuda_hnsw_euclidean_pending_inner_product_sweep"},
    {"large_low_dim", 15, 99, 32, 64, 96, "seeded_from_cuda_hnsw_euclidean_pending_inner_product_sweep"},
    {"large_low_dim", 30, 90, 8, 16, 30, "seeded_from_cuda_hnsw_euclidean_pending_inner_product_sweep"},
    {"large_low_dim", 30, 95, 12, 24, 32, "seeded_from_cuda_hnsw_euclidean_pending_inner_product_sweep"},
    {"large_low_dim", 30, 99, 32, 64, 96, "seeded_from_cuda_hnsw_euclidean_pending_inner_product_sweep"},
    {"large_low_dim", 50, 90, 16, 32, 50, "seeded_from_cuda_hnsw_euclidean_pending_inner_product_sweep"},
    {"large_low_dim", 50, 95, 16, 32, 50, "seeded_from_cuda_hnsw_euclidean_pending_inner_product_sweep"},
    {"large_low_dim", 50, 99, 32, 64, 96, "seeded_from_cuda_hnsw_euclidean_pending_inner_product_sweep"},
    {"large_low_dim", 100, 90, 12, 24, 100, "seeded_from_cuda_hnsw_euclidean_pending_inner_product_sweep"},
    {"large_low_dim", 100, 95, 12, 24, 100, "seeded_from_cuda_hnsw_euclidean_pending_inner_product_sweep"},
    {"large_low_dim", 100, 99, 12, 24, 100, "seeded_from_cuda_hnsw_euclidean_pending_inner_product_sweep"},

    {"medium_low_dim", 15, 90, 8, 16, 24, "seeded_from_cuda_hnsw_euclidean_pending_inner_product_sweep"},
    {"medium_low_dim", 15, 95, 8, 16, 24, "seeded_from_cuda_hnsw_euclidean_pending_inner_product_sweep"},
    {"medium_low_dim", 15, 99, 24, 48, 64, "seeded_from_cuda_hnsw_euclidean_pending_inner_product_sweep"},
    {"medium_low_dim", 30, 90, 8, 16, 30, "seeded_from_cuda_hnsw_euclidean_pending_inner_product_sweep"},
    {"medium_low_dim", 30, 95, 8, 16, 30, "seeded_from_cuda_hnsw_euclidean_pending_inner_product_sweep"},
    {"medium_low_dim", 30, 99, 16, 32, 48, "seeded_from_cuda_hnsw_euclidean_pending_inner_product_sweep"},
    {"medium_low_dim", 50, 90, 16, 32, 50, "seeded_from_cuda_hnsw_euclidean_pending_inner_product_sweep"},
    {"medium_low_dim", 50, 95, 16, 32, 50, "seeded_from_cuda_hnsw_euclidean_pending_inner_product_sweep"},
    {"medium_low_dim", 50, 99, 16, 32, 50, "seeded_from_cuda_hnsw_euclidean_pending_inner_product_sweep"},
    {"medium_low_dim", 100, 90, 8, 16, 100, "seeded_from_cuda_hnsw_euclidean_pending_inner_product_sweep"},
    {"medium_low_dim", 100, 95, 8, 16, 100, "seeded_from_cuda_hnsw_euclidean_pending_inner_product_sweep"},
    {"medium_low_dim", 100, 99, 8, 16, 100, "seeded_from_cuda_hnsw_euclidean_pending_inner_product_sweep"},

    {"small_n", 15, 90, 128, 512, 768, "seeded_from_cuda_hnsw_euclidean_pending_inner_product_sweep"},
    {"small_n", 15, 95, 128, 512, 768, "seeded_from_cuda_hnsw_euclidean_pending_inner_product_sweep"},
    {"small_n", 15, 99, 128, 512, 768, "seeded_from_cuda_hnsw_euclidean_pending_inner_product_sweep"},
    {"small_n", 30, 90, 128, 512, 768, "seeded_from_cuda_hnsw_euclidean_pending_inner_product_sweep"},
    {"small_n", 30, 95, 128, 512, 768, "seeded_from_cuda_hnsw_euclidean_pending_inner_product_sweep"},
    {"small_n", 30, 99, 128, 512, 768, "seeded_from_cuda_hnsw_euclidean_pending_inner_product_sweep"},
    {"small_n", 50, 90, 128, 512, 768, "seeded_from_cuda_hnsw_euclidean_pending_inner_product_sweep"},
    {"small_n", 50, 95, 128, 512, 768, "seeded_from_cuda_hnsw_euclidean_pending_inner_product_sweep"},
    {"small_n", 50, 99, 128, 512, 768, "seeded_from_cuda_hnsw_euclidean_pending_inner_product_sweep"},
    {"small_n", 100, 90, 128, 512, 768, "seeded_from_cuda_hnsw_euclidean_pending_inner_product_sweep"},
    {"small_n", 100, 95, 128, 512, 768, "seeded_from_cuda_hnsw_euclidean_pending_inner_product_sweep"},
    {"small_n", 100, 99, 128, 512, 768, "seeded_from_cuda_hnsw_euclidean_pending_inner_product_sweep"}
  };
  for (const auto& spec : specs) {
    if (shape_group == spec.shape_group &&
        k_bucket == spec.k_bucket &&
        target_code == spec.target_code) {
      return &spec;
    }
  }
  return nullptr;
}

const HnswCudaTuningSpec* hnsw_cuda_benchmark_spec(const std::string& metric,
                                                   const std::string& shape_group,
                                                   int k_bucket,
                                                   int target_code) {
  if (metric == "cosine") {
    return hnsw_cuda_cosine_benchmark_spec(shape_group, k_bucket, target_code);
  }
  if (metric == "correlation") {
    return hnsw_cuda_correlation_benchmark_spec(shape_group, k_bucket, target_code);
  }
  if (metric == "inner_product") {
    return hnsw_cuda_inner_product_benchmark_spec(shape_group, k_bucket, target_code);
  }
  return hnsw_cuda_euclidean_benchmark_spec(shape_group, k_bucket, target_code);
}

int ivf_list_count_cpp(int n, int k) {
  n = safe_n(n);
  k = safe_k(k);
  int count = std::max(16, static_cast<int>(std::ceil(std::sqrt(static_cast<double>(n)))));
  count = std::min(count, static_cast<int>(std::ceil(n / static_cast<double>(std::max(50, 20 * k)))));
  return clamp_int(count, 4, std::min(n, 1024));
}

int ivf_probe_count_cpp(int nlist, int k, const std::string& metric) {
  nlist = safe_n(nlist);
  k = safe_k(k);
  int base = std::max(std::max(16, static_cast<int>(std::ceil(std::sqrt(static_cast<double>(nlist))))),
                      static_cast<int>(std::ceil(k / 3.0)));
  if (metric != "euclidean") {
    base = std::max(std::max(base, static_cast<int>(std::ceil(1.5 * base))),
                    static_cast<int>(std::ceil(k / 2.0)));
  }
  return clamp_int(base, 1, nlist);
}

int faiss_pq_default_m_cpp(int p) {
  p = safe_p(p);
  const int max_m = std::min(p, 96);
  for (int candidate = max_m; candidate >= 1; --candidate) {
    if (p % candidate == 0) return candidate;
  }
  return 1;
}

List cuvs_cagra_params_core(int n,
                            int p,
                            int k,
                            std::string metric,
                            double target_recall_option,
                            int graph_degree_option,
                            int intermediate_graph_degree_option,
                            int search_width_option,
                            int itopk_size_option,
                            bool manual) {
  n = safe_n(n);
  p = safe_p(p);
  k = safe_k(k);
  if (metric.empty()) metric = "euclidean";
  const double target_recall = hnsw_target_recall_cpp(target_recall_option);
  const int target_code = hnsw_target_code_cpp(target_recall);
  const int k_bucket = hnsw_cpu_k_bucket_cpp(k);
  const std::string shape_group = hnsw_shape_group_cpp(n, p);
  const bool small_k = k <= 10;
  const bool large_k = k >= 100;
  const bool large_n = n >= 1000000;
  const bool small_n = n <= 5000;
  const bool high_dim = p >= 1024;
  const bool compact_build = small_k && (small_n || high_dim);
  std::string rule;
  if (large_n && large_k) {
    rule = "large_n_large_k_graph_recall";
  } else if (large_n) {
    rule = "large_n_graph_recall";
  } else if (compact_build) {
    rule = "small_k_compact_cagra_build";
  } else if (small_k) {
    rule = "small_k_graph_speed";
  } else {
    rule = "balanced_graph_search";
  }

  const int n_cap = std::max(1, n - 1);
  int default_graph_degree = compact_build ? std::max(32, k + 1) : std::max(64, k + 1);
  int default_intermediate_seed = compact_build ?
    std::max(32, default_graph_degree * 2) : std::max(128, default_graph_degree * 2);
  int default_search_width = 0;
  int default_itopk = compact_build ?
    std::max(std::max(32, default_graph_degree), k) : std::max(64, default_graph_degree);
  std::string default_build_algo = "auto";
  std::string benchmark_basis;
  std::string benchmark_source = "heuristic_fallback";
  bool benchmark_target_met = false;
  if (!manual) {
    const HpcCagraSpec* spec = nullptr;
    bool seeded_correlation_from_cosine = false;
    if (metric == "cosine") {
      spec = hpc_cagra_cosine_spec(shape_group, k_bucket, target_code);
    } else if (metric == "correlation") {
      spec = hpc_cagra_cosine_spec(shape_group, k_bucket, target_code);
      seeded_correlation_from_cosine = spec != nullptr;
    } else if (metric == "inner_product") {
      spec = hpc_cagra_inner_product_spec(shape_group, k_bucket, target_code);
    }
    if (spec == nullptr) {
      spec = hpc_cagra_spec(shape_group, k_bucket, target_code);
    }
    if (spec != nullptr) {
      default_graph_degree = spec->graph_degree;
      default_intermediate_seed = spec->intermediate_graph_degree;
      default_search_width = spec->search_width;
      default_itopk = spec->itopk_size;
      default_build_algo = spec->build_algo;
      benchmark_basis = spec->basis;
      if (seeded_correlation_from_cosine) {
        const std::string from = "cosine";
        const std::string to = "correlation";
        std::size_t pos = 0;
        while ((pos = benchmark_basis.find(from, pos)) != std::string::npos) {
          benchmark_basis.replace(pos, from.length(), to);
          pos += to.length();
        }
      }
      if (metric == "euclidean") {
        benchmark_source = "hpc_cagra_cuda_euclidean_20260628_054710";
        benchmark_target_met =
          benchmark_basis.find("fastest_meeting_target") != std::string::npos;
      } else if (metric == "cosine") {
        benchmark_source =
          "hpc_cagra_cuda_cosine_seeded_from_euclidean_20260628_054710";
        benchmark_target_met = false;
      } else if (metric == "correlation") {
        benchmark_source =
          "hpc_cagra_cuda_correlation_validation_pending_seeded_from_euclidean_20260628_054710";
        benchmark_target_met = false;
      } else if (metric == "inner_product") {
        benchmark_source =
          "hpc_cagra_cuda_inner_product_validation_pending_seeded_from_euclidean_20260628_054710";
        benchmark_target_met = false;
      } else {
        benchmark_source =
          "hpc_cagra_cuda_" + metric + "_validation_pending_seeded_from_euclidean_20260628_054710";
        benchmark_target_met = false;
      }
      rule = "hpc_cuda_cagra_" + metric + "_" + shape_group +
        "_k" + std::to_string(k_bucket) + "_" +
        hnsw_target_label_cpp(target_code);
    }
  }
  default_graph_degree = clamp_int(default_graph_degree, k + 1, n_cap);
  const int requested_graph_degree = requested_int(graph_degree_option, default_graph_degree);
  const int graph_degree = option_int(graph_degree_option, default_graph_degree, k + 1, n_cap);
  const int default_intermediate = std::max(graph_degree, default_intermediate_seed);
  const int requested_intermediate = requested_int(intermediate_graph_degree_option, default_intermediate);
  const int intermediate = option_int(
    intermediate_graph_degree_option,
    default_intermediate,
    graph_degree,
    n_cap
  );
  const int requested_search_width = requested_int(search_width_option, default_search_width);
  const int search_width = option_int(search_width_option, default_search_width, 0, 1024);
  default_itopk = std::max(default_itopk, k);
  const int requested_itopk = requested_int(itopk_size_option, default_itopk);
  const int itopk = option_int(itopk_size_option, default_itopk, k, 4096);

  return List::create(
    _["graph_degree"] = graph_degree,
    _["intermediate_graph_degree"] = intermediate,
    _["search_width"] = search_width,
    _["itopk_size"] = itopk,
    _["cagra_build_algo"] = default_build_algo,
    _["requested_graph_degree"] = requested_graph_degree,
    _["requested_intermediate_graph_degree"] = requested_intermediate,
    _["requested_search_width"] = requested_search_width,
    _["requested_itopk_size"] = requested_itopk,
    _["target_recall"] = target_recall,
    _["requested_target_recall"] = target_recall_option,
    _["tuning_policy"] = manual ? "manual_options" : "auto_shape_k_target_recall",
    _["tuning_rule"] = rule,
    _["tuning_metric"] = metric,
    _["tuning_metric_aware"] = metric != "euclidean",
    _["tuning_shape_group"] = shape_group,
    _["tuning_k_bucket"] = k_bucket,
    _["tuning_target_recall_code"] = target_code,
    _["tuning_benchmark_basis"] = benchmark_basis,
    _["tuning_benchmark_target_met"] = benchmark_target_met,
    _["tuning_benchmark_source"] = benchmark_source,
    _["tuning_large_n"] = large_n,
    _["tuning_small_n"] = small_n,
    _["tuning_high_dim"] = high_dim,
    _["tuning_compact_build"] = compact_build,
    _["tuning_small_k"] = small_k,
    _["tuning_large_k"] = large_k,
    _["tuning_source"] = "cpp"
  );
}

std::string cagra_build_algo_for_shape_core(int n,
                                            int p,
                                            int k,
                                            bool self_query,
                                            bool compact,
                                            const std::string& requested) {
  if (requested != "auto") return requested;
  n = safe_n(n);
  p = safe_p(p);
  k = safe_k(k);
  const bool small_n = n <= 5000;
  const bool high_dim = p >= 1024;
  const bool moderate_k = k <= 100;
  if (self_query && moderate_k && (compact || (small_n && high_dim))) {
    return "iterative_cagra_search";
  }
  return "ivf_pq";
}

} // namespace

// [[Rcpp::export]]
List nn_auto_select_backend_cpp(std::string resolved_backend,
                                std::string requested_backend,
                                std::string requested_method,
                                std::string metric,
                                int n,
                                int p,
                                int n_points,
                                int k,
                                bool self_query,
                                bool exclude_self,
                                bool cuda_available,
                                bool cuvs_available,
                                bool faiss_available,
                                bool faiss_gpu_available,
                                std::string cagra_preference,
                                int cuda_exact_n,
                                double cuda_exact_work,
                                int metric_graph_n,
                                int metric_graph_min_k,
                                double metric_graph_work,
                                int cagra_compact_n,
                                int cagra_high_dim_p,
                                int cagra_compact_max_k,
                                double cuvs_bruteforce_work_threshold,
                                double cpu_exact_work,
                                double cpu_faiss_flat_work,
                                double target_recall_option,
                                std::string tuning) {
  const double work_size = static_cast<double>(n) *
    static_cast<double>(n_points) * static_cast<double>(p);
  const double target_recall = hnsw_target_recall_cpp(target_recall_option);
  const int target_recall_code = hnsw_target_code_cpp(target_recall);
  std::string selected = resolved_backend;
  std::string reason = "explicit_route";
  std::string error;
  std::string cuda_auto_rule;
  std::string cuda_auto_shape_group;

  if (resolved_backend == "auto") {
    std::string cuda_error;
    std::string gpu;
    std::string gpu_rule;
    std::string gpu_shape_group;
    if (self_query && k <= 256 && work_size >= 5e8 &&
        (cuda_available || cuvs_available)) {
      gpu = select_cuda(
        self_query, n, p, n_points, k, work_size, metric,
        cuda_available, cuvs_available, faiss_gpu_available,
        cagra_preference, cuda_exact_n, cuda_exact_work,
        metric_graph_n, metric_graph_min_k, metric_graph_work,
        cagra_compact_n, cagra_high_dim_p, cagra_compact_max_k,
        cuvs_bruteforce_work_threshold, target_recall_code,
        gpu_rule, gpu_shape_group, cuda_error
      );
    }
    if (!gpu.empty() && cuda_error.empty() && gpu != "cpu_auto") {
      selected = gpu;
      reason = "auto_cuda_preselector";
      cuda_auto_rule = gpu_rule;
      cuda_auto_shape_group = gpu_shape_group;
    } else {
      selected = select_cpu(
        self_query, n, p, n_points, k, work_size, metric,
        faiss_available,
        cpu_exact_work, cpu_faiss_flat_work, target_recall_code
      );
      reason = "auto_cpu_fallback";
    }
  } else if (resolved_backend == "cpu_auto") {
    selected = select_cpu(
      self_query, n, p, n_points, k, work_size, metric,
      faiss_available,
      cpu_exact_work, cpu_faiss_flat_work, target_recall_code
    );
    reason = "cpu_auto_shape_selector";
  } else if (resolved_backend == "cuda_auto" || resolved_backend == "gpu_auto") {
    std::string cuda_error;
    selected = select_cuda(
      self_query, n, p, n_points, k, work_size, metric,
      cuda_available, cuvs_available, faiss_gpu_available,
      cagra_preference, cuda_exact_n, cuda_exact_work,
      metric_graph_n, metric_graph_min_k, metric_graph_work,
      cagra_compact_n, cagra_high_dim_p, cagra_compact_max_k,
      cuvs_bruteforce_work_threshold, target_recall_code,
      cuda_auto_rule, cuda_auto_shape_group, cuda_error
    );
    if (cuda_error.empty()) {
      reason = cuda_auto_rule.empty() ? "cuda_auto_shape_selector" : cuda_auto_rule;
    } else {
      reason = "cuda_auto_unavailable";
      error = cuda_error;
      selected = resolved_backend;
    }
  }

  const bool explicit_backend = requested_backend != "auto";
  const bool explicit_method = requested_method != "auto";
  const std::string backend_decision = explicit_backend ?
    ("explicit_" + requested_backend) : reason;
  const std::string method_decision = explicit_method ?
    ("explicit_" + requested_method) : reason;

  return List::create(
    _["policy"] = "cpp_static_shape_k_metric_selector",
    _["slow_tuning"] = false,
    _["requested_backend"] = requested_backend,
    _["requested_method"] = requested_method,
    _["resolved_public_backend"] = resolved_backend,
    _["explicit_backend"] = explicit_backend,
    _["explicit_method"] = explicit_method,
    _["backend_decision"] = backend_decision,
    _["method_decision"] = method_decision,
    _["selected_backend"] = selected,
    _["predicted_backend"] = selected,
    _["predicted_method"] = public_method_from_backend(selected),
    _["predicted_device"] = device_from_backend(selected),
    _["reason"] = reason,
    _["error"] = error,
    _["n"] = n,
    _["p"] = p,
    _["n_points"] = n_points,
    _["k"] = k,
    _["metric"] = metric,
    _["self_query"] = self_query,
    _["exclude_self"] = exclude_self,
    _["work_size"] = work_size,
    _["target_recall"] = target_recall,
    _["requested_target_recall"] = target_recall_option,
    _["target_recall_code"] = target_recall_code,
    _["cuda_auto_rule"] = cuda_auto_rule,
    _["cuda_auto_shape_group"] = cuda_auto_shape_group,
    _["auto_method_policy"] = cuda_auto_rule.empty() ? "" : "cuda_flat_ivf_shape_k_target_recall",
    _["tuning"] = tuning
  );
}

// [[Rcpp::export]]
List nn_tune_cpu_exact_cpp(int n,
                           int p,
                           int k,
                           std::string metric = "euclidean",
                           double target_recall_option = NA_REAL) {
  n = safe_n(n);
  p = valid_int(p) ? p : NA_INTEGER;
  k = safe_k(k);
  const double target_recall = hnsw_target_recall_cpp(target_recall_option);
  const int target_code = hnsw_target_code_cpp(target_recall);
  const int k_bucket = hnsw_cpu_k_bucket_cpp(k);
  const std::string shape_group = hnsw_shape_group_cpp(n, p);
  const bool euclidean = metric == "euclidean";
  const bool cosine = metric == "cosine";
  const bool correlation = metric == "correlation";
  const bool inner_product = metric == "inner_product";
  const HpcExactSpec* spec = euclidean ?
    hpc_exact_spec("cpu", shape_group, k_bucket, target_code) :
    (cosine ? hpc_exact_cosine_spec("cpu", shape_group, k_bucket, target_code) :
     (correlation ? hpc_exact_correlation_spec("cpu", shape_group, k_bucket, target_code) :
      (inner_product ? hpc_exact_inner_product_spec("cpu", shape_group, k_bucket, target_code) : nullptr)));

  int recommended_n_threads = 12;
  int faiss_query_batch_size = 16384;
  bool cache_fitted_indexes = false;
  std::string recommended_output = "float";
  std::string result_backend = euclidean ? "faiss_flat_l2" :
    (cosine ? "faiss_flat_cosine" :
     (correlation ? "faiss_flat_correlation" :
      (inner_product ? "faiss_flat_ip" : "faiss_flat_l2")));
  std::string resolved_backend = result_backend;
  std::string distance_type = "float32";
  std::string input_type = "float32";
  std::string input_layout = "float32_column_major_payload_to_row_major";
  std::string benchmark_basis;
  std::string benchmark_source = "heuristic_fallback";
  std::string rule = (euclidean || cosine || correlation || inner_product) ?
    ("cpu_exact_" + shape_group + "_k" + std::to_string(k_bucket) + "_" +
     hnsw_target_label_cpp(target_code)) :
    "cpu_exact_metric_fallback";
  bool benchmark_target_met = false;

  if (spec != nullptr) {
    recommended_n_threads = spec->n_threads;
    faiss_query_batch_size = spec->faiss_query_batch_size;
    cache_fitted_indexes = spec->cache_fitted_indexes;
    recommended_output = spec->output;
    result_backend = spec->result_backend;
    resolved_backend = spec->resolved_backend;
    distance_type = spec->distance_type;
    input_type = spec->input_type;
    input_layout = spec->input_layout;
    benchmark_basis = spec->basis;
    benchmark_source = inner_product ? "hpc_exact_cpu12_inner_product_20260630_161530" :
      (correlation ? "hpc_exact_cpu12_correlation_20260701_090337" :
       (cosine ? "hpc_exact_cpu12_cosine_20260630_161539" :
        "hpc_exact_cpu12_euclidean_20260630_161409"));
    rule = std::string("hpc_cpu_exact_") + metric + "_" + shape_group +
      "_k" + std::to_string(k_bucket) + "_" +
      hnsw_target_label_cpp(target_code);
    benchmark_target_met =
      benchmark_basis.find("fastest_meeting_target") != std::string::npos;
  }

  faiss_query_batch_size = clamp_int(faiss_query_batch_size, 1, std::max(1, n));

  return List::create(
    _["target_recall"] = target_recall,
    _["requested_target_recall"] = target_recall_option,
    _["expected_recall_at_k"] = 1.0,
    _["exact_recall_by_construction"] = true,
    _["tuning_policy"] = "auto_shape_k_metric_target_recall_exact",
    _["tuning_rule"] = rule,
    _["tuning_metric"] = metric,
    _["tuning_metric_aware"] = !euclidean,
    _["tuning_backend"] = "cpu",
    _["tuning_method"] = "exact",
    _["tuning_shape_group"] = shape_group,
    _["tuning_k_bucket"] = k_bucket,
    _["tuning_target_recall_code"] = target_code,
    _["tuning_benchmark_basis"] = benchmark_basis,
    _["tuning_benchmark_target_met"] = benchmark_target_met,
    _["tuning_benchmark_source"] = benchmark_source,
    _["recommended_n_threads"] = recommended_n_threads,
    _["faiss_query_batch_size"] = faiss_query_batch_size,
    _["cache_fitted_indexes"] = cache_fitted_indexes,
    _["recommended_output"] = recommended_output,
    _["result_backend"] = result_backend,
    _["resolved_backend"] = resolved_backend,
    _["distance_type"] = distance_type,
    _["input_type"] = input_type,
    _["input_layout"] = input_layout,
    _["tuning_large_n"] = n >= 50000,
    _["tuning_small_k"] = k <= 15,
    _["tuning_large_k"] = k >= 100,
    _["tuning_source"] = "cpp"
  );
}

// [[Rcpp::export]]
List nn_tune_cuda_exact_cpp(int n,
                            int p,
                            int k,
                            std::string metric = "euclidean",
                            double target_recall_option = NA_REAL) {
  n = safe_n(n);
  p = valid_int(p) ? p : NA_INTEGER;
  k = safe_k(k);
  const double target_recall = hnsw_target_recall_cpp(target_recall_option);
  const int target_code = hnsw_target_code_cpp(target_recall);
  const int k_bucket = hnsw_cpu_k_bucket_cpp(k);
  const std::string shape_group = hnsw_shape_group_cpp(n, p);
  const bool euclidean = metric == "euclidean";
  const bool cosine = metric == "cosine";
  const bool correlation = metric == "correlation";
  const bool inner_product = metric == "inner_product";
  const HpcCudaExactSpec* spec = euclidean ?
    hpc_cuda_exact_euclidean_spec("cuda", shape_group, k_bucket, target_code) :
    (cosine ?
     hpc_cuda_exact_cosine_spec("cuda", shape_group, k_bucket, target_code) :
	 (correlation ?
	  hpc_cuda_exact_correlation_spec("cuda", shape_group, k_bucket, target_code) :
	  (inner_product ?
	   hpc_cuda_exact_inner_product_spec("cuda", shape_group, k_bucket, target_code) :
	   nullptr)));

  int recommended_n_threads = 12;
  int faiss_gpu_query_batch_size = 8192;
  bool faiss_gpu_reuse_resources = true;
  std::string recommended_output = "float";
  std::string result_backend = euclidean ? "faiss_gpu_flat_l2" :
    (cosine ? "faiss_gpu_flat_cosine" :
     (correlation ? "faiss_gpu_flat_correlation" :
      (inner_product ? "faiss_gpu_flat_ip" : "faiss_gpu_flat_l2")));
  std::string resolved_backend = result_backend;
  std::string distance_type = "float32";
  std::string input_type = "float32";
  std::string input_layout = "float32_column_major_payload_to_row_major";
  std::string benchmark_basis;
  std::string benchmark_source = "heuristic_fallback";
  std::string rule = (euclidean || cosine || correlation || inner_product) ?
    ("cuda_exact_" + shape_group + "_k" + std::to_string(k_bucket) + "_" +
     hnsw_target_label_cpp(target_code)) :
    "cuda_exact_metric_fallback";
  bool benchmark_target_met = false;

  if (spec != nullptr) {
    recommended_n_threads = spec->n_threads;
    faiss_gpu_query_batch_size = spec->faiss_gpu_query_batch_size;
    faiss_gpu_reuse_resources = spec->faiss_gpu_reuse_resources;
    recommended_output = spec->output;
    result_backend = spec->result_backend;
    resolved_backend = spec->resolved_backend;
    distance_type = spec->distance_type;
    input_type = spec->input_type;
    input_layout = spec->input_layout;
    benchmark_basis = spec->basis;
	benchmark_source = inner_product && benchmark_basis.find("jmlr_mloss_inner_product") != std::string::npos ?
	  "hpc_jmlr_mloss_exact_cuda_inner_product_20260715" :
	  (euclidean ? "hpc_exact_cuda_euclidean_20260701_014100" :
	   (cosine ? "hpc_exact_cuda_cosine_20260702_110455" :
	    (correlation ? "hpc_exact_cuda_correlation_20260703_023519" :
	     "hpc_exact_cuda_inner_product_seeded_from_euclidean_pending")));
    rule = std::string("hpc_cuda_exact_") + metric + "_" + shape_group +
      "_k" + std::to_string(k_bucket) + "_" +
      hnsw_target_label_cpp(target_code);
    benchmark_target_met =
      benchmark_basis.find("fastest_meeting_target") != std::string::npos;
  }

  faiss_gpu_query_batch_size =
    clamp_int(faiss_gpu_query_batch_size, 1, std::max(1, n));

  return List::create(
    _["target_recall"] = target_recall,
    _["requested_target_recall"] = target_recall_option,
    _["expected_recall_at_k"] = 1.0,
    _["exact_recall_by_construction"] = true,
    _["tuning_policy"] = "auto_shape_k_metric_target_recall_cuda_exact",
    _["tuning_rule"] = rule,
    _["tuning_metric"] = metric,
    _["tuning_metric_aware"] = !euclidean,
    _["tuning_backend"] = "cuda",
    _["tuning_method"] = "exact",
    _["tuning_shape_group"] = shape_group,
    _["tuning_k_bucket"] = k_bucket,
    _["tuning_target_recall_code"] = target_code,
    _["tuning_benchmark_basis"] = benchmark_basis,
    _["tuning_benchmark_target_met"] = benchmark_target_met,
    _["tuning_benchmark_source"] = benchmark_source,
    _["recommended_n_threads"] = recommended_n_threads,
    _["faiss_gpu_query_batch_size"] = faiss_gpu_query_batch_size,
    _["faiss_gpu_reuse_resources"] = faiss_gpu_reuse_resources,
    _["cache_fitted_indexes"] = false,
    _["recommended_output"] = recommended_output,
    _["result_backend"] = result_backend,
    _["resolved_backend"] = resolved_backend,
    _["distance_type"] = distance_type,
    _["input_type"] = input_type,
    _["input_layout"] = input_layout,
    _["tuning_large_n"] = n >= 50000,
    _["tuning_small_k"] = k <= 15,
    _["tuning_large_k"] = k >= 100,
    _["tuning_source"] = "cpp"
  );
}

// [[Rcpp::export]]
List nn_tune_cuda_flat_cpp(int n,
                           int p,
                           int k,
                           std::string metric = "euclidean",
                           double target_recall_option = NA_REAL) {
  n = safe_n(n);
  p = valid_int(p) ? p : NA_INTEGER;
  k = safe_k(k);
  const double target_recall = hnsw_target_recall_cpp(target_recall_option);
  const int target_code = hnsw_target_code_cpp(target_recall);
  const int k_bucket = hnsw_cpu_k_bucket_cpp(k);
  const std::string shape_group = hnsw_shape_group_cpp(n, p);
  const bool euclidean = metric == "euclidean";
  const bool cosine = metric == "cosine";
  const bool correlation = metric == "correlation";
  const bool inner_product = metric == "inner_product";
  const HpcCudaFlatSpec* spec = euclidean ?
    hpc_cuda_flat_euclidean_spec("cuda", shape_group, k_bucket, target_code) :
    (cosine ?
     hpc_cuda_flat_cosine_spec("cuda", shape_group, k_bucket, target_code) :
     (correlation ?
      hpc_cuda_flat_correlation_spec("cuda", shape_group, k_bucket, target_code) :
      (inner_product ?
       hpc_cuda_flat_inner_product_spec("cuda", shape_group, k_bucket, target_code) :
       nullptr)));

  int recommended_n_threads = 12;
  int faiss_gpu_query_batch_size = 8192;
  bool faiss_gpu_reuse_resources = true;
  std::string recommended_output = "float";
  std::string result_backend = inner_product ? "faiss_gpu_flat_ip" :
    (cosine ? "faiss_gpu_flat_cosine" :
     (correlation ? "faiss_gpu_flat_correlation" : "faiss_gpu_flat_l2"));
  std::string resolved_backend = result_backend;
  std::string distance_type = "float32";
  std::string input_type = "float32";
  std::string input_layout = (cosine || correlation) ? "float32_payload_direct_row_major" :
    "float32_column_major_payload_to_row_major";
  std::string benchmark_basis;
  std::string benchmark_source = "heuristic_fallback";
  std::string rule = (euclidean || cosine || correlation || inner_product) ?
    ("cuda_flat_" + shape_group + "_k" + std::to_string(k_bucket) + "_" +
     hnsw_target_label_cpp(target_code)) :
    "cuda_flat_metric_fallback";
  bool benchmark_target_met = false;

  if (spec != nullptr) {
    recommended_n_threads = spec->n_threads;
    faiss_gpu_query_batch_size = spec->faiss_gpu_query_batch_size;
    faiss_gpu_reuse_resources = spec->faiss_gpu_reuse_resources;
    recommended_output = spec->output;
    result_backend = spec->result_backend;
    resolved_backend = spec->resolved_backend;
    distance_type = spec->distance_type;
    input_type = spec->input_type;
    input_layout = spec->input_layout;
    benchmark_basis = spec->basis;
    benchmark_source = inner_product && benchmark_basis.find("jmlr_mloss_inner_product") != std::string::npos ?
      "hpc_jmlr_mloss_flat_cuda_inner_product_20260715" :
      (euclidean ? "hpc_flat_cuda_euclidean_20260701_031219" :
       (cosine ? "hpc_flat_cuda_cosine_20260702_120850" :
        (correlation ? "hpc_flat_cuda_correlation_20260703_062359" :
         "hpc_flat_cuda_inner_product_seeded_from_euclidean_pending")));
    rule = std::string("hpc_cuda_flat_") + metric + "_" + shape_group +
      "_k" + std::to_string(k_bucket) + "_" +
      hnsw_target_label_cpp(target_code);
    benchmark_target_met =
      benchmark_basis.find("fastest_meeting_target") != std::string::npos;
  }

  faiss_gpu_query_batch_size =
    clamp_int(faiss_gpu_query_batch_size, 1, std::max(1, n));

  return List::create(
    _["target_recall"] = target_recall,
    _["requested_target_recall"] = target_recall_option,
    _["expected_recall_at_k"] = 1.0,
    _["exact_recall_by_construction"] = true,
    _["tuning_policy"] = "auto_shape_k_metric_target_recall_cuda_flat",
    _["tuning_rule"] = rule,
    _["tuning_metric"] = metric,
    _["tuning_metric_aware"] = !euclidean,
    _["tuning_backend"] = "cuda",
    _["tuning_method"] = "flat",
    _["tuning_shape_group"] = shape_group,
    _["tuning_k_bucket"] = k_bucket,
    _["tuning_target_recall_code"] = target_code,
    _["tuning_benchmark_basis"] = benchmark_basis,
    _["tuning_benchmark_target_met"] = benchmark_target_met,
    _["tuning_benchmark_source"] = benchmark_source,
    _["recommended_n_threads"] = recommended_n_threads,
    _["faiss_gpu_query_batch_size"] = faiss_gpu_query_batch_size,
    _["faiss_gpu_reuse_resources"] = faiss_gpu_reuse_resources,
    _["cache_fitted_indexes"] = false,
    _["recommended_output"] = recommended_output,
    _["result_backend"] = result_backend,
    _["resolved_backend"] = resolved_backend,
    _["distance_type"] = distance_type,
    _["input_type"] = input_type,
    _["input_layout"] = input_layout,
    _["tuning_large_n"] = n >= 50000,
    _["tuning_small_k"] = k <= 15,
    _["tuning_large_k"] = k >= 100,
    _["tuning_source"] = "cpp"
  );
}

// [[Rcpp::export]]
List nn_tune_cuda_bruteforce_cpp(int n,
                                 int p,
                                 int k,
                                 std::string metric = "euclidean",
                                 double target_recall_option = NA_REAL) {
  n = safe_n(n);
  p = valid_int(p) ? p : NA_INTEGER;
  k = safe_k(k);
  const double target_recall = hnsw_target_recall_cpp(target_recall_option);
  const int target_code = hnsw_target_code_cpp(target_recall);
  const int k_bucket = hnsw_cpu_k_bucket_cpp(k);
  const std::string shape_group = hnsw_shape_group_cpp(n, p);
  const bool euclidean = metric == "euclidean";
  const bool cosine = metric == "cosine";
  const bool correlation = metric == "correlation";
  const bool inner_product = metric == "inner_product";
  const HpcCudaBruteforceSpec* spec = nullptr;
  if (euclidean) {
    spec = hpc_cuda_bruteforce_euclidean_spec("cuda", shape_group, k_bucket, target_code);
  } else if (cosine) {
    spec = hpc_cuda_bruteforce_cosine_spec("cuda", shape_group, k_bucket, target_code);
  } else if (correlation) {
    spec = hpc_cuda_bruteforce_correlation_spec("cuda", shape_group, k_bucket, target_code);
  } else if (inner_product) {
    spec = hpc_cuda_bruteforce_inner_product_spec("cuda", shape_group, k_bucket, target_code);
  }

  int recommended_n_threads = 12;
  int faiss_gpu_query_batch_size = 8192;
  bool faiss_gpu_reuse_resources = true;
  std::string recommended_output = "float";
  std::string result_backend = "cuda_cuvs_bruteforce";
  std::string resolved_backend = result_backend;
  std::string distance_type = "float32";
  std::string input_type = "float32";
  std::string input_layout = (cosine || correlation) ?
    "float32_payload_direct_row_major" :
    "float32_column_major_payload_to_row_major";
  std::string benchmark_basis;
  std::string benchmark_source = "heuristic_fallback";
  std::string rule = (euclidean || cosine || correlation || inner_product) ?
    ("cuda_bruteforce_" + metric + "_" + shape_group +
     "_k" + std::to_string(k_bucket) + "_" +
     hnsw_target_label_cpp(target_code)) :
    "cuda_bruteforce_metric_fallback";
  bool benchmark_target_met = false;

  if (spec != nullptr) {
    recommended_n_threads = spec->n_threads;
    faiss_gpu_query_batch_size = spec->faiss_gpu_query_batch_size;
    faiss_gpu_reuse_resources = spec->faiss_gpu_reuse_resources;
    recommended_output = spec->output;
    result_backend = spec->result_backend;
    resolved_backend = spec->resolved_backend;
    distance_type = spec->distance_type;
    input_type = spec->input_type;
    input_layout = spec->input_layout;
    benchmark_basis = spec->basis;
    benchmark_source = euclidean ?
      "hpc_bruteforce_cuda_euclidean_20260630_181030" :
      (cosine ?
       "hpc_bruteforce_cuda_cosine_proxy_from_euclidean_20260630_181030" :
       (correlation ?
        "hpc_bruteforce_cuda_correlation_proxy_from_euclidean_20260630_181030" :
        "hpc_bruteforce_cuda_inner_product_seeded_from_euclidean_pending"));
    rule = std::string("hpc_cuda_bruteforce_") + metric + "_" + shape_group +
      "_k" + std::to_string(k_bucket) + "_" +
      hnsw_target_label_cpp(target_code);
    benchmark_target_met =
      benchmark_basis.find("fastest_meeting_target") != std::string::npos;
  }

  faiss_gpu_query_batch_size =
    clamp_int(faiss_gpu_query_batch_size, 1, std::max(1, n));

  return List::create(
    _["target_recall"] = target_recall,
    _["requested_target_recall"] = target_recall_option,
    _["expected_recall_at_k"] = 1.0,
    _["exact_recall_by_construction"] = true,
    _["tuning_policy"] = "auto_shape_k_metric_target_recall_cuda_bruteforce",
    _["tuning_rule"] = rule,
    _["tuning_metric"] = metric,
    _["tuning_metric_aware"] = !euclidean,
    _["tuning_backend"] = "cuda",
    _["tuning_method"] = "bruteforce",
    _["tuning_shape_group"] = shape_group,
    _["tuning_k_bucket"] = k_bucket,
    _["tuning_target_recall_code"] = target_code,
    _["tuning_benchmark_basis"] = benchmark_basis,
    _["tuning_benchmark_target_met"] = benchmark_target_met,
    _["tuning_benchmark_source"] = benchmark_source,
    _["recommended_n_threads"] = recommended_n_threads,
    _["faiss_gpu_query_batch_size"] = faiss_gpu_query_batch_size,
    _["faiss_gpu_reuse_resources"] = faiss_gpu_reuse_resources,
    _["cache_fitted_indexes"] = false,
    _["recommended_output"] = recommended_output,
    _["result_backend"] = result_backend,
    _["resolved_backend"] = resolved_backend,
    _["distance_type"] = distance_type,
    _["input_type"] = input_type,
    _["input_layout"] = input_layout,
    _["tuning_large_n"] = n >= 50000,
    _["tuning_small_k"] = k <= 15,
    _["tuning_large_k"] = k >= 100,
    _["tuning_source"] = "cpp"
  );
}

// [[Rcpp::export]]
List nn_tune_cpu_flat_cpp(int n,
                          int p,
                          int k,
                          std::string metric = "euclidean",
                          double target_recall_option = NA_REAL) {
  n = safe_n(n);
  p = valid_int(p) ? p : NA_INTEGER;
  k = safe_k(k);
  const double target_recall = hnsw_target_recall_cpp(target_recall_option);
  const int target_code = hnsw_target_code_cpp(target_recall);
  const int k_bucket = hnsw_cpu_k_bucket_cpp(k);
  const std::string shape_group = hnsw_shape_group_cpp(n, p);
  const bool euclidean = metric == "euclidean";
  const bool cosine = metric == "cosine";
  const bool correlation = metric == "correlation";
  const bool inner_product = metric == "inner_product";
  const HpcExactSpec* spec = euclidean ?
    hpc_flat_spec("cpu", shape_group, k_bucket, target_code) :
    (cosine ? hpc_flat_cosine_spec("cpu", shape_group, k_bucket, target_code) :
     (correlation ? hpc_flat_correlation_spec("cpu", shape_group, k_bucket, target_code) :
      (inner_product ? hpc_flat_inner_product_spec("cpu", shape_group, k_bucket, target_code) : nullptr)));

  int recommended_n_threads = 12;
  int faiss_query_batch_size = 16384;
  bool cache_fitted_indexes = false;
  std::string recommended_output = "float";
  std::string result_backend =
    metric == "inner_product" ? "faiss_flat_ip" :
    (metric == "cosine" ? "faiss_flat_cosine" :
     (metric == "correlation" ? "faiss_flat_correlation" : "faiss_flat_l2"));
  std::string resolved_backend = result_backend;
  std::string distance_type = "float32";
  std::string input_type = "float32";
  std::string input_layout = "float32_column_major_payload_to_row_major";
  std::string benchmark_basis;
  std::string benchmark_source = "heuristic_fallback";
  std::string rule = (euclidean || cosine || correlation || inner_product) ?
    ("cpu_flat_" + shape_group + "_k" + std::to_string(k_bucket) + "_" +
     hnsw_target_label_cpp(target_code)) :
    "cpu_flat_metric_fallback";
  bool benchmark_target_met = false;

  if (spec != nullptr) {
    recommended_n_threads = spec->n_threads;
    faiss_query_batch_size = spec->faiss_query_batch_size;
    cache_fitted_indexes = spec->cache_fitted_indexes;
    recommended_output = spec->output;
    result_backend = spec->result_backend;
    resolved_backend = spec->resolved_backend;
    distance_type = spec->distance_type;
    input_type = spec->input_type;
    input_layout = spec->input_layout;
    benchmark_basis = spec->basis;
    benchmark_source = inner_product ? "hpc_flat_cpu12_inner_product_20260630_161530" :
      (correlation ? "hpc_flat_cpu12_correlation_20260701_090337" :
       (cosine ? "hpc_flat_cpu12_cosine_20260701_015607" :
        "hpc_flat_cpu12_euclidean_20260630_161409"));
    rule = std::string("hpc_cpu_flat_") + metric + "_" + shape_group +
      "_k" + std::to_string(k_bucket) + "_" +
      hnsw_target_label_cpp(target_code);
    benchmark_target_met =
      benchmark_basis.find("fastest_meeting_target") != std::string::npos;
  }

  faiss_query_batch_size = clamp_int(faiss_query_batch_size, 1, std::max(1, n));

  return List::create(
    _["target_recall"] = target_recall,
    _["requested_target_recall"] = target_recall_option,
    _["expected_recall_at_k"] = 1.0,
    _["exact_recall_by_construction"] = true,
    _["tuning_policy"] = "auto_shape_k_metric_target_recall_flat",
    _["tuning_rule"] = rule,
    _["tuning_metric"] = metric,
    _["tuning_metric_aware"] = !euclidean,
    _["tuning_backend"] = "cpu",
    _["tuning_method"] = "flat",
    _["tuning_shape_group"] = shape_group,
    _["tuning_k_bucket"] = k_bucket,
    _["tuning_target_recall_code"] = target_code,
    _["tuning_benchmark_basis"] = benchmark_basis,
    _["tuning_benchmark_target_met"] = benchmark_target_met,
    _["tuning_benchmark_source"] = benchmark_source,
    _["recommended_n_threads"] = recommended_n_threads,
    _["faiss_query_batch_size"] = faiss_query_batch_size,
    _["cache_fitted_indexes"] = cache_fitted_indexes,
    _["recommended_output"] = recommended_output,
    _["result_backend"] = result_backend,
    _["resolved_backend"] = resolved_backend,
    _["distance_type"] = distance_type,
    _["input_type"] = input_type,
    _["input_layout"] = input_layout,
    _["tuning_large_n"] = n >= 50000,
    _["tuning_small_k"] = k <= 15,
    _["tuning_large_k"] = k >= 100,
    _["tuning_source"] = "cpp"
  );
}

// [[Rcpp::export]]
List nn_tune_cpu_bruteforce_cpp(int n,
                                int p,
                                int k,
                                std::string metric = "euclidean",
                                double target_recall_option = NA_REAL) {
  n = safe_n(n);
  p = valid_int(p) ? p : NA_INTEGER;
  k = safe_k(k);
  const double target_recall = hnsw_target_recall_cpp(target_recall_option);
  const int target_code = hnsw_target_code_cpp(target_recall);
  const int k_bucket = hnsw_cpu_k_bucket_cpp(k);
  const std::string shape_group = hnsw_shape_group_cpp(n, p);
  const bool euclidean = metric == "euclidean";
  const bool cosine = metric == "cosine";
  const bool correlation = metric == "correlation";
  const bool inner_product = metric == "inner_product";
  const HpcExactSpec* spec = euclidean ?
    hpc_bruteforce_spec("cpu", shape_group, k_bucket, target_code) :
    (cosine ? hpc_bruteforce_cosine_spec("cpu", shape_group, k_bucket, target_code) :
     (correlation ? hpc_bruteforce_correlation_spec("cpu", shape_group, k_bucket, target_code) :
      (inner_product ? hpc_bruteforce_inner_product_spec("cpu", shape_group, k_bucket, target_code) : nullptr)));

  int recommended_n_threads = 12;
  int faiss_query_batch_size = 16384;
  bool cache_fitted_indexes = false;
  std::string recommended_output = "float";
  std::string result_backend =
    metric == "inner_product" ? "faiss_flat_ip" :
    (metric == "cosine" ? "faiss_flat_cosine" :
     (metric == "correlation" ? "faiss_flat_correlation" : "faiss_flat_l2"));
  std::string resolved_backend = result_backend;
  std::string distance_type = "float32";
  std::string input_type = "float32";
  std::string input_layout = "float32_column_major_payload_to_row_major";
  std::string benchmark_basis;
  std::string benchmark_source = "heuristic_fallback";
  std::string rule = (euclidean || cosine || correlation || inner_product) ?
    ("cpu_bruteforce_" + shape_group + "_k" + std::to_string(k_bucket) + "_" +
     hnsw_target_label_cpp(target_code)) :
    "cpu_bruteforce_metric_fallback";
  bool benchmark_target_met = false;

  if (spec != nullptr) {
    recommended_n_threads = spec->n_threads;
    faiss_query_batch_size = spec->faiss_query_batch_size;
    cache_fitted_indexes = spec->cache_fitted_indexes;
    recommended_output = spec->output;
    result_backend = spec->result_backend;
    resolved_backend = spec->resolved_backend;
    distance_type = spec->distance_type;
    input_type = spec->input_type;
    input_layout = spec->input_layout;
    benchmark_basis = spec->basis;
    benchmark_source = inner_product ?
      "hpc_bruteforce_cpu12_inner_product_20260630_161530" :
      (correlation ?
      "hpc_bruteforce_cpu12_correlation_20260701_090337" :
      (cosine ? "hpc_bruteforce_cpu12_cosine_20260630_161535" :
       "hpc_bruteforce_cpu12_euclidean_20260630_161409"));
    rule = std::string("hpc_cpu_bruteforce_") + metric + "_" + shape_group +
      "_k" + std::to_string(k_bucket) + "_" +
      hnsw_target_label_cpp(target_code);
    benchmark_target_met =
      benchmark_basis.find("fastest_meeting_target") != std::string::npos;
  }

  faiss_query_batch_size = clamp_int(faiss_query_batch_size, 1, std::max(1, n));

  return List::create(
    _["target_recall"] = target_recall,
    _["requested_target_recall"] = target_recall_option,
    _["expected_recall_at_k"] = 1.0,
    _["exact_recall_by_construction"] = true,
    _["tuning_policy"] = "auto_shape_k_metric_target_recall_bruteforce",
    _["tuning_rule"] = rule,
    _["tuning_metric"] = metric,
    _["tuning_metric_aware"] = !euclidean,
    _["tuning_backend"] = "cpu",
    _["tuning_method"] = "bruteforce",
    _["tuning_shape_group"] = shape_group,
    _["tuning_k_bucket"] = k_bucket,
    _["tuning_target_recall_code"] = target_code,
    _["tuning_benchmark_basis"] = benchmark_basis,
    _["tuning_benchmark_target_met"] = benchmark_target_met,
    _["tuning_benchmark_source"] = benchmark_source,
    _["recommended_n_threads"] = recommended_n_threads,
    _["faiss_query_batch_size"] = faiss_query_batch_size,
    _["cache_fitted_indexes"] = cache_fitted_indexes,
    _["recommended_output"] = recommended_output,
    _["result_backend"] = result_backend,
    _["resolved_backend"] = resolved_backend,
    _["distance_type"] = distance_type,
    _["input_type"] = input_type,
    _["input_layout"] = input_layout,
    _["tuning_large_n"] = n >= 50000,
    _["tuning_small_k"] = k <= 15,
    _["tuning_large_k"] = k >= 100,
    _["tuning_source"] = "cpp"
  );
}

// [[Rcpp::export]]
List nn_tune_faiss_ivf_cpp(int n,
                           int p,
                           int k,
                           std::string metric,
                           double target_recall_option = NA_REAL,
                           std::string backend = "cpu",
                           std::string method = "ivf",
                           int nlist_option = NA_INTEGER,
                           int nprobe_option = NA_INTEGER,
                           bool manual = false) {
  n = safe_n(n);
  p = valid_int(p) ? p : NA_INTEGER;
  k = safe_k(k);
  const double target_recall = hnsw_target_recall_cpp(target_recall_option);
  const int target_code = hnsw_target_code_cpp(target_recall);
  const int k_bucket = hnsw_cpu_k_bucket_cpp(k);
  const std::string shape_group = method == "ivfpq_fastscan" ?
    ivfpq_fastscan_shape_group_cpp(n, p) : hnsw_shape_group_cpp(n, p);
  const bool small_k = k <= 10;
  const bool large_k = k >= 100;
  const bool large_n = n >= 1000000;
  const bool metric_aware = metric != "euclidean";
  const bool euclidean = metric == "euclidean";
  const bool cosine = metric == "cosine";
  const bool correlation = metric == "correlation";
  const bool inner_product = metric == "inner_product";
  std::string base_rule = large_n ? "large_n_coarse_quantizer" :
    (large_k ? "large_k_more_probe" : (small_k ? "small_k_speed" : "balanced_shape_k"));
  std::string rule = (metric_aware && !manual) ? ("metric_" + base_rule) : base_rule;

  int default_nlist = ivf_list_count_cpp(n, k);
  int default_nprobe = ivf_probe_count_cpp(default_nlist, k, metric);
  int default_pq_m = NA_INTEGER;
  int default_pq_nbits = NA_INTEGER;
  int default_fastscan_refine_factor = NA_INTEGER;
  int default_fastscan_bbs = NA_INTEGER;
  int default_cuvs_ivf_batch_size = NA_INTEGER;
  std::string benchmark_basis;
  std::string benchmark_source = "heuristic_fallback";
  bool benchmark_target_met = false;
  if (!manual && euclidean) {
    if (method == "ivfpq_fastscan") {
      if (const HpcIvfpqFastscanSpec* spec =
            hpc_ivfpq_fastscan_spec(backend, shape_group, k_bucket, target_code)) {
        default_nlist = spec->nlist;
        default_nprobe = spec->nprobe;
        default_pq_m = spec->pq_m;
        default_pq_nbits = spec->pq_nbits;
        default_fastscan_refine_factor = spec->refine_factor;
        default_fastscan_bbs = spec->bbs;
        benchmark_basis = spec->basis;
        benchmark_target_met =
          benchmark_basis.find("fastest_meeting_target") != std::string::npos;
        if (const HpcIvfpqFastscanBatchSpec* batch_spec =
              hpc_ivfpq_fastscan_batch_spec(backend, shape_group, k_bucket, target_code)) {
          default_cuvs_ivf_batch_size = batch_spec->cuvs_ivf_batch_size;
        }
        benchmark_source = backend == "cpu" ?
          "hpc_ivfpq_fastscan_cpu12_euclidean_20260630_161409" :
          "hpc_ivfpq_fastscan_cuda_euclidean_20260701_100837";
        rule = "hpc_" + backend + "_ivfpq_fastscan_" + shape_group +
          "_k" + std::to_string(k_bucket) + "_" +
          hnsw_target_label_cpp(target_code);
      }
    } else if (method == "ivfpq") {
      if (const HpcIvfpqSpec* spec =
            hpc_ivfpq_spec(backend, shape_group, k_bucket, target_code)) {
        default_nlist = spec->nlist;
        default_nprobe = spec->nprobe;
        default_pq_m = spec->pq_m;
        default_pq_nbits = spec->pq_nbits;
        benchmark_basis = spec->basis;
        benchmark_target_met =
          benchmark_basis.find("fastest_meeting_target") != std::string::npos;
        benchmark_source = backend == "cpu" ?
          "hpc_ivfpq_cpu12_euclidean_shape_defaults_20260630_161409" :
          "hpc_ivfpq_cuda_euclidean_20260701_194051";
        rule = "hpc_" + backend + "_ivfpq_" + shape_group +
          "_k" + std::to_string(k_bucket) + "_" +
          hnsw_target_label_cpp(target_code);
      }
    } else if (const HpcIvfSpec* spec =
                 hpc_ivf_spec(backend, shape_group, k_bucket, target_code)) {
      default_nlist = spec->nlist;
      default_nprobe = spec->nprobe;
      benchmark_basis = spec->basis;
      benchmark_target_met =
        benchmark_basis.find("fastest_meeting_target") != std::string::npos;
        benchmark_source = backend == "cpu" ?
          "hpc_ivf_cpu12_euclidean_20260630_161409" :
          "hpc_ivf_cuda_euclidean_20260702_001853";
      rule = "hpc_" + backend + "_ivf_" + shape_group +
        "_k" + std::to_string(k_bucket) + "_" +
        hnsw_target_label_cpp(target_code);
    }
  } else if (!manual && cosine && method == "ivfpq_fastscan" &&
             (backend == "cpu" || backend == "cuda")) {
    if (const HpcIvfpqFastscanSpec* spec =
          hpc_ivfpq_fastscan_cosine_seed_spec(backend, shape_group, k_bucket, target_code)) {
      default_nlist = spec->nlist;
      default_nprobe = spec->nprobe;
      default_pq_m = spec->pq_m;
      default_pq_nbits = spec->pq_nbits;
      benchmark_basis = backend == "cuda" ?
        "cosine_validation_pending_seeded_from_cuda_euclidean_fastscan" :
        "cosine_validation_pending_seeded_from_euclidean_fastscan";
      benchmark_target_met = false;
      benchmark_source = backend == "cuda" ?
        "hpc_ivfpq_fastscan_cuda_cosine_20260702_133619_failed_before_backend_seeded_from_euclidean_20260701_100837" :
        "hpc_ivfpq_fastscan_cpu12_cosine_20260701_090337_failed_before_backend_seeded_from_euclidean_20260630_161409";
      default_fastscan_refine_factor = spec->refine_factor;
      default_fastscan_bbs = spec->bbs;
      if (const HpcIvfpqFastscanBatchSpec* batch_spec =
            hpc_ivfpq_fastscan_batch_spec(backend, shape_group, k_bucket, target_code)) {
        default_cuvs_ivf_batch_size = batch_spec->cuvs_ivf_batch_size;
      }
      rule = "hpc_" + backend + "_ivfpq_fastscan_cosine_" + shape_group +
        "_k" + std::to_string(k_bucket) + "_" +
        hnsw_target_label_cpp(target_code);
    }
  } else if (!manual && correlation && method == "ivfpq_fastscan" &&
             (backend == "cpu" || backend == "cuda")) {
    if (const HpcIvfpqFastscanSpec* spec =
          hpc_ivfpq_fastscan_correlation_seed_spec(backend, shape_group, k_bucket, target_code)) {
      default_nlist = spec->nlist;
      default_nprobe = spec->nprobe;
      default_pq_m = spec->pq_m;
      default_pq_nbits = spec->pq_nbits;
      benchmark_basis = backend == "cuda" ?
        "correlation_validation_pending_seeded_from_cuda_euclidean_fastscan" :
        "correlation_validation_pending_seeded_from_euclidean_fastscan";
      benchmark_target_met = false;
      benchmark_source = backend == "cuda" ?
        "hpc_ivfpq_fastscan_cuda_correlation_20260703_083505_failed_before_backend_seeded_from_euclidean_20260701_100837" :
        "hpc_ivfpq_fastscan_cpu12_correlation_20260701_090337_failed_before_backend_seeded_from_euclidean_20260630_161409";
      default_fastscan_refine_factor = spec->refine_factor;
      default_fastscan_bbs = spec->bbs;
      if (const HpcIvfpqFastscanBatchSpec* batch_spec =
            hpc_ivfpq_fastscan_batch_spec(backend, shape_group, k_bucket, target_code)) {
        default_cuvs_ivf_batch_size = batch_spec->cuvs_ivf_batch_size;
      }
      rule = "hpc_" + backend + "_ivfpq_fastscan_correlation_" + shape_group +
        "_k" + std::to_string(k_bucket) + "_" +
        hnsw_target_label_cpp(target_code);
    }
  } else if (!manual && inner_product && method == "ivfpq_fastscan" &&
             (backend == "cpu" || backend == "cuda")) {
    if (const HpcIvfpqFastscanSpec* spec =
          hpc_ivfpq_fastscan_inner_product_seed_spec(backend, shape_group, k_bucket, target_code)) {
      default_nlist = spec->nlist;
      default_nprobe = spec->nprobe;
      default_pq_m = spec->pq_m;
      default_pq_nbits = spec->pq_nbits;
      benchmark_basis = backend == "cuda" ?
        "inner_product_validation_pending_seeded_from_cuda_euclidean_fastscan" :
        "inner_product_validation_pending_seeded_from_euclidean_fastscan";
      benchmark_target_met = false;
      benchmark_source = backend == "cuda" ?
        "hpc_ivfpq_fastscan_cuda_inner_product_seeded_from_euclidean_pending" :
        "hpc_ivfpq_fastscan_cpu12_inner_product_20260701_090337_failed_before_backend_seeded_from_euclidean_20260630_161409";
      default_fastscan_refine_factor = spec->refine_factor;
      default_fastscan_bbs = spec->bbs;
      if (const HpcIvfpqFastscanBatchSpec* batch_spec =
            hpc_ivfpq_fastscan_batch_spec(backend, shape_group, k_bucket, target_code)) {
        default_cuvs_ivf_batch_size = batch_spec->cuvs_ivf_batch_size;
      }
      rule = "hpc_" + backend + "_ivfpq_fastscan_inner_product_" + shape_group +
        "_k" + std::to_string(k_bucket) + "_" +
        hnsw_target_label_cpp(target_code);
    }
  } else if (!manual && cosine && method == "ivfpq" && backend == "cpu") {
    if (const HpcIvfpqSpec* spec =
          hpc_ivfpq_cosine_spec(backend, shape_group, k_bucket, target_code)) {
      default_nlist = spec->nlist;
      default_nprobe = spec->nprobe;
      default_pq_m = spec->pq_m;
      default_pq_nbits = spec->pq_nbits;
      benchmark_basis = spec->basis;
      benchmark_target_met =
        benchmark_basis.find("fastest_meeting_target") != std::string::npos;
      benchmark_source = "hpc_ivfpq_cpu12_cosine_20260701_090337";
      rule = "hpc_cpu_ivfpq_cosine_" + shape_group +
        "_k" + std::to_string(k_bucket) + "_" +
        hnsw_target_label_cpp(target_code);
    }
  } else if (!manual && correlation && method == "ivfpq" &&
             (backend == "cpu" || backend == "cuda")) {
    if (const HpcIvfpqSpec* spec =
          hpc_ivfpq_correlation_spec(backend, shape_group, k_bucket, target_code)) {
      default_nlist = spec->nlist;
      default_nprobe = spec->nprobe;
      default_pq_m = spec->pq_m;
      default_pq_nbits = spec->pq_nbits;
      benchmark_basis = spec->basis;
      benchmark_target_met =
        benchmark_basis.find("fastest_meeting_target") != std::string::npos;
      benchmark_source = backend == "cpu" ?
        "hpc_ivfpq_cpu12_correlation_20260701_090337" :
        "hpc_ivfpq_cuda_correlation_20260703_095008";
      rule = "hpc_" + backend + "_ivfpq_correlation_" + shape_group +
        "_k" + std::to_string(k_bucket) + "_" +
        hnsw_target_label_cpp(target_code);
    }
  } else if (!manual && inner_product && method == "ivfpq" &&
             (backend == "cpu" || backend == "cuda")) {
    if (const HpcIvfpqSpec* spec =
          hpc_ivfpq_inner_product_spec(backend, shape_group, k_bucket, target_code)) {
      default_nlist = spec->nlist;
      default_nprobe = spec->nprobe;
      default_pq_m = spec->pq_m;
      default_pq_nbits = spec->pq_nbits;
      benchmark_basis = spec->basis;
      benchmark_target_met =
        benchmark_basis.find("fastest_meeting_target") != std::string::npos;
      benchmark_source = benchmark_basis.find("jmlr_mloss_inner_product") != std::string::npos ?
        (backend == "cpu" ?
          "hpc_jmlr_mloss_ivfpq_cpu12_inner_product_20260715" :
          "hpc_jmlr_mloss_ivfpq_cuda_inner_product_20260715") :
        (backend == "cpu" ?
          "hpc_ivfpq_cpu12_inner_product_20260701_090337" :
          "hpc_ivfpq_cuda_inner_product_seeded_from_euclidean_pending");
      rule = "hpc_" + backend + "_ivfpq_inner_product_" + shape_group +
        "_k" + std::to_string(k_bucket) + "_" +
        hnsw_target_label_cpp(target_code);
    }
  } else if (!manual && cosine && method == "ivf" && backend == "cpu") {
    if (const HpcIvfSpec* spec =
          hpc_ivf_cosine_spec(backend, shape_group, k_bucket, target_code)) {
      default_nlist = spec->nlist;
      default_nprobe = spec->nprobe;
      benchmark_basis = spec->basis;
      benchmark_target_met =
        benchmark_basis.find("fastest_meeting_target") != std::string::npos;
      benchmark_source = "hpc_ivf_cpu12_cosine_20260701_090337";
      rule = "hpc_cpu_ivf_cosine_" + shape_group +
        "_k" + std::to_string(k_bucket) + "_" +
        hnsw_target_label_cpp(target_code);
    }
  } else if (!manual && correlation && method == "ivf" && backend == "cpu") {
    if (const HpcIvfSpec* spec =
          hpc_ivf_correlation_spec(backend, shape_group, k_bucket, target_code)) {
      default_nlist = spec->nlist;
      default_nprobe = spec->nprobe;
      benchmark_basis = spec->basis;
      benchmark_target_met =
        benchmark_basis.find("fastest_meeting_target") != std::string::npos;
      benchmark_source = "hpc_ivf_cpu12_correlation_20260701_090337";
      rule = "hpc_cpu_ivf_correlation_" + shape_group +
        "_k" + std::to_string(k_bucket) + "_" +
        hnsw_target_label_cpp(target_code);
    }
  } else if (!manual && inner_product && method == "ivf" && backend == "cpu") {
    if (const HpcIvfSpec* spec =
          hpc_ivf_inner_product_spec(backend, shape_group, k_bucket, target_code)) {
      default_nlist = spec->nlist;
      default_nprobe = spec->nprobe;
      benchmark_basis = spec->basis;
      benchmark_target_met =
        benchmark_basis.find("fastest_meeting_target") != std::string::npos;
      benchmark_source = benchmark_basis.find("jmlr_mloss_inner_product") != std::string::npos ?
        "hpc_jmlr_mloss_ivf_cpu12_inner_product_20260715" :
        "hpc_ivf_cpu12_inner_product_20260701_090337";
      rule = "hpc_cpu_ivf_inner_product_" + shape_group +
        "_k" + std::to_string(k_bucket) + "_" +
        hnsw_target_label_cpp(target_code);
    }
  }

  default_nlist = clamp_int(default_nlist, 1, n);
  const int requested_nlist = requested_int(nlist_option, default_nlist);
  const int nlist = option_int(nlist_option, default_nlist, 1, n);
  default_nprobe = clamp_int(default_nprobe, 1, nlist);
  const int requested_nprobe = requested_int(nprobe_option, default_nprobe);
  const int nprobe = option_int(nprobe_option, default_nprobe, 1, nlist);
  return List::create(
    _["nlist"] = nlist,
    _["nprobe"] = nprobe,
    _["requested_nlist"] = requested_nlist,
    _["requested_nprobe"] = requested_nprobe,
    _["pq_m"] = default_pq_m,
    _["pq_nbits"] = default_pq_nbits,
    _["ivfpq_fastscan_refine_factor"] = default_fastscan_refine_factor,
    _["ivfpq_fastscan_bbs"] = default_fastscan_bbs,
    _["cuvs_ivf_batch_size"] = default_cuvs_ivf_batch_size,
    _["target_recall"] = target_recall,
    _["requested_target_recall"] = target_recall_option,
    _["tuning_policy"] = manual ? "manual_options" : "auto_shape_k_target_recall",
    _["tuning_rule"] = rule,
    _["tuning_metric"] = metric,
    _["tuning_metric_aware"] = metric_aware,
    _["tuning_backend"] = backend,
    _["tuning_method"] = method,
    _["tuning_shape_group"] = shape_group,
    _["tuning_k_bucket"] = k_bucket,
    _["tuning_target_recall_code"] = target_code,
    _["tuning_benchmark_basis"] = benchmark_basis,
    _["tuning_benchmark_target_met"] = benchmark_target_met,
    _["tuning_benchmark_source"] = benchmark_source,
    _["tuning_large_n"] = large_n,
    _["tuning_small_k"] = small_k,
    _["tuning_large_k"] = large_k,
    _["tuning_source"] = "cpp"
  );
}

// [[Rcpp::export]]
List nn_tune_cuda_ivf_cpp(int n,
                          int p,
                          int k,
                          std::string metric,
                          double target_recall_option = NA_REAL,
                          int nlist_option = NA_INTEGER,
                          int nprobe_option = NA_INTEGER,
                          bool manual = false) {
  n = safe_n(n);
  p = safe_p(p);
  k = safe_k(k);

  const double target_recall = hnsw_target_recall_cpp(target_recall_option);
  const int target_code = hnsw_target_code_cpp(target_recall);
  const bool metric_aware = metric != "euclidean";
  const int k_bucket = hnsw_cpu_k_bucket_cpp(k);
  const int root = std::max(1, static_cast<int>(std::ceil(std::sqrt(static_cast<double>(n)))));
  const int half_root = std::max(16, static_cast<int>(std::floor(static_cast<double>(root) / 2.0)));

  const std::string hpc_shape = hnsw_shape_group_cpp(n, p);
  std::string shape = "general";
  if (n < 5000 && p >= 1024) {
    shape = "small_n_very_high_dim";
  } else if (n < 5000 && p >= 128) {
    shape = "small_n_high_dim";
  } else if (n < 20000 && p >= 128) {
    shape = "small_high_dim";
  } else if (n >= 500000 && p >= 256) {
    shape = "large_high_dim";
  } else if (n >= 500000 && p <= 64) {
    shape = "large_low_dim";
  } else if (n >= 20000 && n < 200000 && p >= 256) {
    shape = "medium_high_dim";
  } else if (n >= 20000 && n < 200000 && p <= 128) {
    shape = "medium_low_dim";
  }

  int default_nlist = ivf_list_count_cpp(n, k);
  int default_nprobe = ivf_probe_count_cpp(default_nlist, k, metric);
  std::string basis = "hpc_cuda_ivf_float32_euclidean_20260628";
  std::string rule_basis = "shape_k_target_recall";
  std::string benchmark_source = "heuristic_fallback";
  bool benchmark_target_met = true;
  bool hpc_metric_spec_used = false;

  if (shape == "small_n_very_high_dim") {
    default_nlist = half_root;
    if (k_bucket <= 15) {
      default_nprobe = 8;
    } else if (k_bucket <= 30) {
      default_nprobe = 4;
    } else if (k_bucket <= 50) {
      default_nprobe = 9;
    } else {
      default_nprobe = target_code >= 99 ? 10 : 5;
    }
  } else if (shape == "small_n_high_dim") {
    if (k_bucket <= 15) {
      default_nlist = root * 6;
      default_nprobe = 64;
    } else if (k_bucket <= 30) {
      default_nlist = root * 4;
      default_nprobe = 48;
    } else if (k_bucket <= 50) {
      default_nlist = root;
      default_nprobe = 17;
    } else {
      default_nlist = root;
      default_nprobe = root;
    }
  } else if (shape == "small_high_dim") {
    if (k_bucket <= 15) {
      default_nlist = target_code >= 99 ? root : root * 2;
      default_nprobe = target_code >= 99 ? 16 : 24;
      benchmark_target_met = target_code < 99;
      if (target_code >= 99) basis += "_best_available_below_target_for_small_high_dim_k15";
    } else if (k_bucket <= 30) {
      default_nlist = root * 4;
      default_nprobe = 63;
    } else if (k_bucket <= 50) {
      default_nlist = root;
      default_nprobe = 17;
    } else {
      default_nlist = root * 6;
      default_nprobe = 136;
    }
  } else if (shape == "medium_high_dim") {
    default_nlist = half_root;
    if (target_code >= 99) {
      if (k_bucket <= 15) {
        default_nprobe = 8;
      } else if (k_bucket <= 30) {
        default_nlist = half_root * 2;
        default_nprobe = 17;
      } else if (k_bucket <= 50) {
        default_nlist = half_root * 4;
        default_nprobe = 36;
      } else {
        default_nprobe = 17;
      }
    } else if (target_code >= 95) {
      if (k_bucket <= 15) {
        default_nprobe = 4;
      } else if (k_bucket <= 30) {
        default_nprobe = 8;
      } else if (k_bucket <= 50) {
        default_nprobe = 5;
      } else {
        default_nprobe = 9;
      }
    } else {
      if (k_bucket <= 30) {
        default_nprobe = 4;
      } else if (k_bucket <= 50) {
        default_nprobe = 5;
      } else {
        default_nprobe = 9;
      }
    }
  } else if (shape == "medium_low_dim") {
    if (k_bucket <= 15) {
      default_nlist = root;
      default_nprobe = 9;
    } else if (k_bucket <= 30) {
      default_nlist = root * 4;
      default_nprobe = 99;
    } else if (k_bucket <= 50) {
      default_nlist = half_root;
      default_nprobe = 9;
    } else {
      default_nlist = root;
      default_nprobe = 17;
    }
  } else if (shape == "large_low_dim") {
    if (target_code >= 99) {
      if (k_bucket <= 15) {
        default_nlist = n >= 1000000 ? 1024 : 512;
        default_nprobe = n >= 1000000 ? 16 : 12;
      } else if (k_bucket <= 30) {
        default_nlist = n >= 1000000 ? 512 : 1024;
        default_nprobe = n >= 1000000 ? 12 : 32;
      } else if (k_bucket <= 50) {
        if (n > 2000000) {
          default_nlist = 512;
          default_nprobe = 12;
        } else if (n >= 1000000) {
          default_nlist = root;
          default_nprobe = 16;
        } else {
          default_nlist = root;
          default_nprobe = 32;
        }
      } else {
        default_nlist = n > 2000000 ? 512 : (n >= 1000000 ? root : std::max(512, half_root));
        default_nprobe = 17;
      }
    } else if (target_code >= 95) {
      if (k_bucket <= 15) {
        default_nlist = 512;
        default_nprobe = 6;
      } else if (k_bucket <= 30) {
        default_nlist = 512;
        default_nprobe = n < 1000000 ? 12 : 6;
      } else if (k_bucket <= 50) {
        default_nlist = n < 1000000 ? root : 512;
        default_nprobe = n < 1000000 ? 16 : 6;
      } else {
        default_nlist = n > 2000000 ? 512 :
          (p <= 12 ? std::max(500, static_cast<int>(std::floor(static_cast<double>(root) / 2.0))) : root);
        default_nprobe = n < 1000000 ? 17 : 9;
      }
    } else {
      if (k_bucket >= 100) {
        default_nlist = n < 1000000 ? root :
          (p <= 12 ? std::max(500, static_cast<int>(std::floor(static_cast<double>(root) / 2.0))) : 512);
        default_nprobe = n < 1000000 ? 17 : 9;
      } else if (k_bucket >= 50 && n >= 1000000 && p <= 12) {
        default_nlist = std::max(500, static_cast<int>(std::floor(static_cast<double>(root) / 2.0)));
        default_nprobe = 6;
      } else {
        default_nlist = 512;
        default_nprobe = 6;
      }
    }
  } else if (shape == "large_high_dim") {
    if (target_code >= 99) {
      if (k_bucket >= 100) {
        default_nlist = 4096;
        default_nprobe = 192;
      } else {
        default_nlist = 2048;
        default_nprobe = 69;
      }
    } else {
      default_nlist = 1024;
      default_nprobe = k_bucket >= 100 ? 17 : 16;
    }
  }

  if (!manual) {
    const HpcIvfSpec* spec = nullptr;
    if (metric == "euclidean") {
      spec = hpc_ivf_spec("cuda", hpc_shape, k_bucket, target_code);
    } else if (metric == "cosine") {
      spec = hpc_ivf_cosine_spec("cuda", hpc_shape, k_bucket, target_code);
    } else if (metric == "correlation") {
      spec = hpc_ivf_correlation_spec("cuda", hpc_shape, k_bucket, target_code);
    } else if (metric == "inner_product") {
      spec = hpc_ivf_inner_product_spec("cuda", hpc_shape, k_bucket, target_code);
    }
    if (spec != nullptr) {
      default_nlist = spec->nlist;
      default_nprobe = spec->nprobe;
      basis = spec->basis;
      benchmark_target_met =
        basis.find("fastest_meeting_target") != std::string::npos;
      benchmark_source = metric == "inner_product" &&
        basis.find("jmlr_mloss_inner_product") != std::string::npos ?
        "hpc_jmlr_mloss_ivf_cuda_inner_product_20260716" :
        (metric == "cosine" ?
        "hpc_ivf_cuda_cosine_20260702_192200" :
        (metric == "correlation" ?
          "hpc_ivf_cuda_correlation_20260703_133655" :
          (metric == "inner_product" ?
            "hpc_ivf_cuda_inner_product_seeded_from_euclidean_pending" :
            "hpc_ivf_cuda_euclidean_20260702_001853")));
      rule_basis = "hpc_shape_k_target_recall";
      shape = hpc_shape;
      hpc_metric_spec_used = true;
    }
  }

  if (metric_aware && !manual && !hpc_metric_spec_used) {
    default_nprobe = std::max(default_nprobe,
                              static_cast<int>(std::ceil(static_cast<double>(default_nprobe) * 1.25)));
    rule_basis += "_metric_probe_increase";
  }

  default_nlist = clamp_int(default_nlist, 1, n);
  const int requested_nlist = requested_int(nlist_option, default_nlist);
  const int nlist = option_int(nlist_option, default_nlist, 1, n);
  default_nprobe = clamp_int(default_nprobe, 1, nlist);
  const int requested_nprobe = requested_int(nprobe_option, default_nprobe);
  const int nprobe = option_int(nprobe_option, default_nprobe, 1, nlist);
  const std::string tuning_rule_prefix = metric == "euclidean" ?
    "cuda_ivf_" : ("cuda_ivf_" + metric + "_");

  return List::create(
    _["nlist"] = nlist,
    _["nprobe"] = nprobe,
    _["requested_nlist"] = requested_nlist,
    _["requested_nprobe"] = requested_nprobe,
    _["target_recall"] = target_recall,
    _["requested_target_recall"] = target_recall_option,
    _["tuning_policy"] = manual ? "manual_options" : "cuda_ivf_shape_k_target_recall",
    _["tuning_rule"] = tuning_rule_prefix + shape + "_k" + std::to_string(k_bucket) +
      "_recall" + std::to_string(target_code),
    _["tuning_rule_basis"] = rule_basis,
    _["tuning_metric"] = metric,
    _["tuning_metric_aware"] = metric_aware,
    _["tuning_shape_group"] = shape,
    _["tuning_cuda_shape_group"] = shape,
    _["tuning_k_bucket"] = k_bucket,
    _["tuning_target_recall_code"] = target_code,
    _["tuning_benchmark_basis"] = basis,
    _["tuning_benchmark_target_met"] = benchmark_target_met,
    _["tuning_benchmark_source"] = benchmark_source,
    _["tuning_source"] = "cpp"
  );
}

// [[Rcpp::export]]
List nn_tune_faiss_pq_cpp(int p,
                          int n = NA_INTEGER,
                          int m_option = NA_INTEGER,
                          int nbits_option = NA_INTEGER,
                          bool manual = false,
                          bool manual_nbits = false) {
  p = safe_p(p);
  const bool n_known = valid_int(n);
  const int min_training = 624;
  const int min_training_8bit = 9984;
  const bool high_dim = p >= 256;
  const bool small_training = n_known && n < min_training;
  const bool reduced_codebook_training = n_known && n >= min_training && n < min_training_8bit;
  int m = option_int(m_option, faiss_pq_default_m_cpp(p), 1, p);
  while (m > 1 && p % m != 0) --m;
  const int nbits_default = (reduced_codebook_training && !manual_nbits) ? 4 : 8;
  const int nbits = option_int(nbits_option, nbits_default, 4, 12);
  std::string rule = small_training ? "small_training_rows_minimum_pq" :
    ((reduced_codebook_training && !manual_nbits) ? "training_rows_4bit_pq" :
       (high_dim ? "high_dim_largest_divisor_pq" : "dimension_largest_divisor_pq"));

  return List::create(
    _["m"] = m,
    _["nbits"] = nbits,
    _["tuning_policy"] = manual ? "manual_options" : "auto_dimension",
    _["tuning_rule"] = rule,
    _["tuning_high_dim"] = high_dim,
    _["tuning_small_training"] = small_training,
    _["tuning_reduced_codebook_training"] = reduced_codebook_training,
    _["min_training_rows"] = min_training,
    _["min_training_rows_8bit"] = min_training_8bit,
    _["tuning_source"] = "cpp"
  );
}

// [[Rcpp::export]]
List nn_tune_cuvs_ivfpq_cpp(int p,
                            int n = NA_INTEGER,
                            int pq_dim_option = NA_INTEGER,
                            int pq_bits_option = NA_INTEGER,
                            bool manual = false) {
  p = safe_p(p);
  const bool n_known = valid_int(n);
  const int min_training_8bit = 9984;
  const bool high_dim = p >= 256;
  const bool manual_bits = valid_int(pq_bits_option);
  const bool reduced_codebook_training = n_known && n < min_training_8bit;
  const int requested_pq_dim = requested_int(pq_dim_option, 0);
  int pq_dim = requested_int(pq_dim_option, 0);
  if (pq_dim < 0) pq_dim = 0;
  const int pq_bits_default = (reduced_codebook_training && !manual_bits) ? 4 : 8;
  const int requested_pq_bits = requested_int(pq_bits_option, pq_bits_default);
  int pq_bits = option_int(pq_bits_option, pq_bits_default, 4, 8);
  CuvsIvfPqAlignment aligned = repair_cuvs_ivfpq_alignment(p, pq_dim, pq_bits);
  pq_dim = aligned.pq_dim;
  pq_bits = aligned.pq_bits;
  std::string rule = (reduced_codebook_training && !manual_bits) ? "training_rows_4bit_pq" :
    (high_dim ? "high_dim_default_pq" : "dimension_default_pq");
  if (aligned.adjusted) {
    rule += "_";
    rule += aligned.rule;
  }
  return List::create(
    _["pq_dim"] = pq_dim,
    _["pq_bits"] = pq_bits,
    _["requested_pq_dim"] = requested_pq_dim,
    _["requested_pq_bits"] = requested_pq_bits,
    _["pq_alignment_adjusted"] = aligned.adjusted,
    _["pq_alignment_rule"] = aligned.rule,
    _["tuning_policy"] = manual ? "manual_options" : "auto_shape",
    _["tuning_rule"] = rule,
    _["tuning_high_dim"] = high_dim,
    _["tuning_reduced_codebook_training"] = reduced_codebook_training,
    _["min_training_rows_8bit"] = min_training_8bit,
    _["tuning_source"] = "cpp"
  );
}

// [[Rcpp::export]]
List nn_tune_faiss_hnsw_cpp(int n,
                            int p,
                            int k,
                            std::string metric,
                            double target_recall = 0.99,
                            int m_option = NA_INTEGER,
                            int ef_construction_option = NA_INTEGER,
                            int ef_search_option = NA_INTEGER,
                            bool manual = false) {
  n = valid_int(n) ? n : NA_INTEGER;
  p = valid_int(p) ? p : NA_INTEGER;
  k = safe_k(k);
  target_recall = hnsw_target_recall_cpp(target_recall);
  const bool euclidean = metric == "euclidean";
  const bool cosine = metric == "cosine";
  const bool correlation = metric == "correlation";
  const bool inner_product = metric == "inner_product";
  const bool low_dim = valid_int(p) && p <= 64;
  const bool high_dim = valid_int(p) && p >= 256;
  const bool large_n = valid_int(n) && n >= 50000;
  const bool very_large_high_dim = large_n && high_dim;
  const bool small_k = k <= 15;
  const bool large_k = k >= 100;
  const bool non_euclidean = !euclidean;
  const std::string shape_group = hnsw_shape_group_cpp(n, p);
  const int k_bucket = hnsw_cpu_k_bucket_cpp(k);
  const int target_code = hnsw_target_code_cpp(target_recall);
  const HnswCpuTuningSpec* benchmark_spec = euclidean ?
    hnsw_cpu_benchmark_spec(shape_group, k_bucket, target_code) :
    (cosine ? hnsw_cpu_cosine_benchmark_spec(shape_group, k_bucket, target_code) :
     (correlation ? hnsw_cpu_correlation_benchmark_spec(shape_group, k_bucket, target_code) :
      (inner_product ? hnsw_cpu_inner_product_benchmark_spec(shape_group, k_bucket, target_code) : nullptr)));
  const std::string benchmark_basis = benchmark_spec ? benchmark_spec->benchmark_basis : "";
  const std::string benchmark_source = benchmark_spec ?
    (inner_product && benchmark_basis.find("jmlr_mloss_inner_product") != std::string::npos ?
       "hpc_jmlr_mloss_hnsw_cpu12_inner_product_20260715" :
     (inner_product ? "hpc_hnsw_cpu12_inner_product_20260701_090337" :
     (correlation ? "hpc_hnsw_cpu12_correlation_20260701_090337" :
     (cosine ? "hpc_hnsw_cpu12_cosine_20260701_082849" :
      "hpc_hnsw_cpu12_euclidean_20260630_161409")))) :
    "heuristic_fallback";
  const bool benchmark_target_met = !manual && benchmark_spec != nullptr &&
    (benchmark_basis.find("fastest_meeting_target") != std::string::npos ||
     benchmark_basis.find("hit_all_shape_datasets") != std::string::npos);

  std::string rule;
  int default_m;
  int default_ef_construction;
  int default_ef_search;
  if (benchmark_spec != nullptr) {
    rule = std::string("hpc_cpu_hnsw_") +
      (inner_product ? "inner_product_" : (correlation ? "correlation_" : (cosine ? "cosine_" : ""))) + shape_group +
      "_k" + std::to_string(k_bucket) + "_" +
      hnsw_target_label_cpp(target_code);
    default_m = benchmark_spec->m;
    default_ef_construction = benchmark_spec->ef_construction;
    default_ef_search = std::max(k, benchmark_spec->ef_search);
  } else if (euclidean && large_n && low_dim) {
    if (target_recall <= 0.90 && small_k) {
      rule = "recall90_large_low_dim_small_k";
      default_m = 6;
      default_ef_construction = 30;
      default_ef_search = std::max(k, 15);
    } else if (target_recall <= 0.90 && k <= 50) {
      rule = "recall90_large_low_dim_mid_k";
      default_m = 8;
      default_ef_construction = 40;
      default_ef_search = std::max(k, 35);
    } else if (target_recall <= 0.90) {
      rule = "recall90_large_low_dim_large_k";
      default_m = 12;
      default_ef_construction = 60;
      default_ef_search = std::max(k, 75);
    } else if (target_recall <= 0.95 && small_k) {
      rule = "recall95_large_low_dim_small_k";
      default_m = 8;
      default_ef_construction = 40;
      default_ef_search = std::max(k, 20);
    } else if (target_recall <= 0.95 && k <= 50) {
      rule = "recall95_large_low_dim_mid_k";
      default_m = 12;
      default_ef_construction = 60;
      default_ef_search = std::max(k, 50);
    } else if (target_recall <= 0.95) {
      rule = "recall95_large_low_dim_large_k";
      default_m = 12;
      default_ef_construction = 60;
      default_ef_search = std::max(k, 100);
    } else if (small_k) {
      rule = "recall99_large_low_dim_small_k";
      default_m = 12;
      default_ef_construction = 60;
      default_ef_search = std::max(k, 30);
    } else if (k <= 50) {
      rule = "recall99_large_low_dim_mid_k";
      default_m = 16;
      default_ef_construction = 80;
      default_ef_search = std::max(k, 75);
    } else {
      rule = "recall99_large_low_dim_large_k";
      default_m = 16;
      default_ef_construction = 80;
      default_ef_search = std::max(k, 100);
    }
  } else if (euclidean && very_large_high_dim) {
    if (target_recall <= 0.90 && small_k) {
      rule = "recall90_large_high_dim_small_k";
      default_m = 8;
      default_ef_construction = 30;
      default_ef_search = std::max(k, 15);
    } else if (target_recall <= 0.90 && k <= 50) {
      rule = "recall90_large_high_dim_mid_k";
      default_m = 8;
      default_ef_construction = 40;
      default_ef_search = std::max(k, 35);
    } else if (target_recall <= 0.90) {
      rule = "recall90_large_high_dim_large_k";
      default_m = 12;
      default_ef_construction = 60;
      default_ef_search = std::max(k, 75);
    } else if (target_recall <= 0.95 && small_k) {
      rule = "recall95_large_high_dim_small_k";
      default_m = 8;
      default_ef_construction = 40;
      default_ef_search = std::max(k, 25);
    } else if (target_recall <= 0.95 && k <= 50) {
      rule = "recall95_large_high_dim_mid_k";
      default_m = 12;
      default_ef_construction = 60;
      default_ef_search = std::max(k, 50);
    } else if (target_recall <= 0.95) {
      rule = "recall95_large_high_dim_large_k";
      default_m = 16;
      default_ef_construction = 80;
      default_ef_search = std::max(k, 100);
    } else if (small_k) {
      rule = "recall99_large_high_dim_small_k";
      default_m = 12;
      default_ef_construction = 60;
      default_ef_search = std::max(k, 40);
    } else if (k <= 50) {
      rule = "recall99_large_high_dim_mid_k";
      default_m = 16;
      default_ef_construction = 80;
      default_ef_search = std::max(k, 75);
    } else {
      rule = "recall99_large_high_dim_large_k";
      default_m = 24;
      default_ef_construction = 120;
      default_ef_search = std::max(k, 150);
    }
  } else if (non_euclidean && small_k) {
    rule = "balanced_small_k_metric";
    default_m = 32;
    default_ef_construction = 160;
    default_ef_search = std::max(120, 4 * k);
  } else if ((very_large_high_dim && large_k) ||
             (non_euclidean && (large_k || high_dim))) {
    rule = "high_recall_shape_metric";
    default_m = 48;
    default_ef_construction = 240;
    default_ef_search = std::max(220, 3 * k);
  } else if (small_k && !high_dim && !non_euclidean) {
    rule = "small_k_speed";
    default_m = 24;
    default_ef_construction = 120;
    default_ef_search = std::max(80, 4 * k);
  } else {
    rule = "balanced_shape_metric";
    default_m = 32;
    default_ef_construction = 200;
    default_ef_search = std::max(150, 3 * k);
  }

  const int m = option_int(m_option, default_m, 2, 256);
  const int ef_construction = option_int(ef_construction_option, default_ef_construction, m, 4096);
  const int ef_search = option_int(ef_search_option, default_ef_search, k, 4096);

  return List::create(
    _["m"] = m,
    _["ef_construction"] = ef_construction,
    _["ef_search"] = ef_search,
    _["rule"] = rule,
    _["policy"] = manual ? "manual_options" :
      (benchmark_spec != nullptr ? "auto_shape_metric_target_recall" : "auto_shape_metric"),
    _["high_dim"] = high_dim,
    _["low_dim"] = low_dim,
    _["large_n"] = large_n,
    _["small_k"] = small_k,
    _["large_k"] = large_k,
    _["non_euclidean"] = non_euclidean,
    _["shape_group"] = shape_group,
    _["k_bucket"] = k_bucket,
    _["target_recall_code"] = target_code,
    _["benchmark_basis"] = benchmark_basis,
    _["benchmark_source"] = benchmark_source,
    _["tuning_shape_group"] = shape_group,
    _["tuning_k_bucket"] = k_bucket,
    _["tuning_target_recall_code"] = target_code,
    _["tuning_benchmark_basis"] = benchmark_basis,
    _["tuning_benchmark_target_met"] = benchmark_target_met,
    _["tuning_benchmark_source"] = benchmark_source,
    _["requested_m"] = default_m,
    _["requested_ef_construction"] = default_ef_construction,
    _["requested_ef_search"] = default_ef_search,
    _["target_recall"] = target_recall,
    _["tuning_source"] = "cpp"
  );
}

// [[Rcpp::export]]
List nn_tune_faiss_nsg_cpp(int k,
                           int r_option = NA_INTEGER,
                           int search_l_option = NA_INTEGER,
                           int build_type_option = NA_INTEGER,
                           bool manual = false) {
  k = safe_k(k);
  const bool small_k = k <= 10;
  const bool large_k = k >= 100;
  const int r = option_int(r_option, 48, 2, 512);
  const int search_l = option_int(search_l_option, std::max(200, 4 * k), k, 4096);
  const int build_type = option_int(build_type_option, 1, 0, 1);
  return List::create(
    _["r"] = r,
    _["search_l"] = search_l,
    _["build_type"] = build_type,
    _["tuning_policy"] = manual ? "manual_options" : "auto_k",
    _["tuning_rule"] = large_k ? "large_k_search_l" : (small_k ? "small_k_speed" : "balanced_k"),
    _["tuning_small_k"] = small_k,
    _["tuning_large_k"] = large_k,
    _["tuning_source"] = "cpp"
  );
}

// [[Rcpp::export]]
List nn_tune_faiss_nndescent_cpp(int k,
                                 int graph_k_option = NA_INTEGER,
                                 int n_iter_option = NA_INTEGER,
                                 int search_l_option = NA_INTEGER,
                                 bool manual = false) {
  k = safe_k(k);
  const bool small_k = k <= 10;
  const bool large_k = k >= 100;
  const int graph_k = option_int(graph_k_option, std::max(100, 2 * k), k, 1024);
  const int n_iter = option_int(n_iter_option, 20, 1, 100);
  const int search_l = option_int(search_l_option, std::max(graph_k, 2 * k), k, 4096);
  return List::create(
    _["graph_k"] = graph_k,
    _["n_iter"] = n_iter,
    _["search_l"] = search_l,
    _["tuning_policy"] = manual ? "manual_options" : "auto_k",
    _["tuning_rule"] = large_k ? "large_k_graph_search" : (small_k ? "small_k_speed" : "balanced_k"),
    _["tuning_small_k"] = small_k,
    _["tuning_large_k"] = large_k,
    _["tuning_source"] = "cpp"
  );
}

// [[Rcpp::export]]
List nn_tune_cpu_nndescent_cpp(int n,
                               int p,
                               int k,
                               std::string metric = "euclidean",
                               double target_recall_option = NA_REAL) {
  n = safe_n(n);
  p = valid_int(p) ? p : NA_INTEGER;
  k = safe_k(k);
  const double target_recall = hnsw_target_recall_cpp(target_recall_option);
  const int target_code = hnsw_target_code_cpp(target_recall);
  const int k_bucket = hnsw_cpu_k_bucket_cpp(k);
  const std::string shape_group = hnsw_shape_group_cpp(n, p);
  const int n_cap = std::max(1, n - 1);
  int pool_size = std::min(
    n_cap,
    std::max(k + 15, std::min(160, static_cast<int>(std::ceil(2.5 * k))))
  );
  int n_iters = n >= 50000 ? 3 : 4;
  if (k < 30) ++n_iters;
  int max_candidates = std::min(n_cap, std::max(pool_size * 4, k * 12));
  int n_random_projections = n >= 50000 ? 8 : 6;
  std::string benchmark_basis;
  std::string benchmark_source = "heuristic_fallback";
  bool benchmark_target_met = false;
  std::string rule = n >= 50000 ? "large_n_random_projection_seed" :
    "balanced_random_projection_seed";
  if (metric == "euclidean") {
    if (const HpcNndescentSpec* spec =
          hpc_cpu_nndescent_spec(shape_group, k_bucket, target_code)) {
      pool_size = spec->pool_size;
      n_iters = spec->n_iters;
      max_candidates = spec->max_candidates;
      n_random_projections = spec->n_random_projections;
      benchmark_basis = spec->basis;
      benchmark_target_met =
        benchmark_basis.find("fastest_meeting_target") != std::string::npos;
      benchmark_source = "hpc_nndescent_cpu12_euclidean_20260630_161409";
      rule = "hpc_cpu_nndescent_" + shape_group +
        "_k" + std::to_string(k_bucket) + "_" +
        hnsw_target_label_cpp(target_code);
    }
  } else if (metric == "cosine") {
    if (const HpcNndescentSpec* spec =
          hpc_cpu_nndescent_cosine_spec(shape_group, k_bucket, target_code)) {
      pool_size = spec->pool_size;
      n_iters = spec->n_iters;
      max_candidates = spec->max_candidates;
      n_random_projections = spec->n_random_projections;
      benchmark_basis = spec->basis;
      benchmark_target_met =
        benchmark_basis.find("fastest_meeting_target") != std::string::npos;
      benchmark_source = "hpc_nndescent_cpu12_cosine_20260701_090337";
      rule = "hpc_cpu_nndescent_cosine_" + shape_group +
        "_k" + std::to_string(k_bucket) + "_" +
        hnsw_target_label_cpp(target_code);
    }
  } else if (metric == "correlation") {
    if (const HpcNndescentSpec* spec =
          hpc_cpu_nndescent_correlation_spec(shape_group, k_bucket, target_code)) {
      pool_size = spec->pool_size;
      n_iters = spec->n_iters;
      max_candidates = spec->max_candidates;
      n_random_projections = spec->n_random_projections;
      benchmark_basis = spec->basis;
      benchmark_target_met =
        benchmark_basis.find("fastest_meeting_target") != std::string::npos;
      benchmark_source = "hpc_nndescent_cpu12_correlation_20260701_090337";
      rule = "hpc_cpu_nndescent_correlation_" + shape_group +
        "_k" + std::to_string(k_bucket) + "_" +
        hnsw_target_label_cpp(target_code);
    }
  } else if (metric == "inner_product") {
    if (const HpcNndescentSpec* spec =
          hpc_cpu_nndescent_inner_product_spec(shape_group, k_bucket, target_code)) {
      pool_size = spec->pool_size;
      n_iters = spec->n_iters;
      max_candidates = spec->max_candidates;
      n_random_projections = spec->n_random_projections;
      benchmark_basis = spec->basis;
      benchmark_target_met =
        benchmark_basis.find("fastest_meeting_target") != std::string::npos;
      benchmark_source = benchmark_basis.find("jmlr_mloss_inner_product") != std::string::npos ?
        "hpc_jmlr_mloss_nndescent_cpu12_inner_product_20260715" :
        "hpc_nndescent_cpu12_inner_product_20260701_090337";
      rule = "hpc_cpu_nndescent_inner_product_" + shape_group +
        "_k" + std::to_string(k_bucket) + "_" +
        hnsw_target_label_cpp(target_code);
    }
  }
  pool_size = clamp_int(pool_size, 1, n_cap);
  max_candidates = clamp_int(max_candidates, pool_size, n_cap);
  return List::create(
    _["pool_size"] = pool_size,
    _["n_iters"] = n_iters,
    _["max_candidates"] = max_candidates,
    _["n_random_projections"] = n_random_projections,
    _["target_recall"] = target_recall,
    _["requested_target_recall"] = target_recall_option,
    _["tuning_policy"] = "auto_shape_k_target_recall",
    _["tuning_rule"] = rule,
    _["tuning_metric"] = metric,
    _["tuning_shape_group"] = shape_group,
    _["tuning_k_bucket"] = k_bucket,
    _["tuning_target_recall_code"] = target_code,
    _["tuning_benchmark_basis"] = benchmark_basis,
    _["tuning_benchmark_target_met"] = benchmark_target_met,
    _["tuning_benchmark_source"] = benchmark_source,
    _["tuning_large_n"] = n >= 50000,
    _["tuning_small_k"] = k < 30,
    _["tuning_source"] = "cpp"
  );
}

// [[Rcpp::export]]
List nn_tune_cuvs_cagra_cpp(int n,
                            int p,
                            int k,
                            std::string metric = "euclidean",
                            double target_recall_option = NA_REAL,
                            int graph_degree_option = NA_INTEGER,
                            int intermediate_graph_degree_option = NA_INTEGER,
                            int search_width_option = NA_INTEGER,
                            int itopk_size_option = NA_INTEGER,
                            bool manual = false) {
  return cuvs_cagra_params_core(
    n, p, k, metric, target_recall_option,
    graph_degree_option, intermediate_graph_degree_option,
    search_width_option, itopk_size_option, manual
  );
}

// [[Rcpp::export]]
std::string nn_tune_cuvs_cagra_build_algo_cpp(int n,
                                              int p,
                                              int k,
                                              bool self_query,
                                              bool compact,
                                              std::string requested = "auto") {
  return cagra_build_algo_for_shape_core(n, p, k, self_query, compact, requested);
}

// [[Rcpp::export]]
List nn_tune_cuvs_hnsw_cpp(int n,
                           int p,
                           int k,
                           int n_threads,
                           std::string build_algo_preference = "auto",
                           double target_recall_option = 0.99,
                           int graph_degree_option = NA_INTEGER,
                           int intermediate_graph_degree_option = NA_INTEGER,
                           int search_width_option = NA_INTEGER,
                           int itopk_size_option = NA_INTEGER,
                           int ef_option = NA_INTEGER,
                           bool manual_cagra = false,
                           std::string metric = "euclidean") {
  List base = cuvs_cagra_params_core(
    n, p, k, metric, target_recall_option,
    graph_degree_option, intermediate_graph_degree_option,
    search_width_option, itopk_size_option, manual_cagra
  );
  n = safe_n(n);
  p = safe_p(p);
  k = safe_k(k);
  const double target_recall = hnsw_target_recall_cpp(target_recall_option);
  const bool low_dim = p <= 64;
  const bool high_dim = p >= 256;
  const bool large_n = n >= 50000;
  const bool large_k = k >= 100;
  const int n_cap = std::max(1, n - 1);
  const int min_degree = std::min(n_cap, 2);
  const std::string cuda_shape_group = hnsw_cuda_shape_group_cpp(n, p);
  const int k_bucket = hnsw_cpu_k_bucket_cpp(k);
  const int target_code = hnsw_target_code_cpp(target_recall);
  const HnswCudaTuningSpec* benchmark_spec =
    hnsw_cuda_benchmark_spec(metric, cuda_shape_group, k_bucket, target_code);
  const std::string benchmark_basis = benchmark_spec ? benchmark_spec->benchmark_basis : "";
  const std::string benchmark_source = benchmark_spec ?
    (metric == "cosine" ?
      "hpc_hnsw_cuda_cosine_20260702_123021" :
      (metric == "correlation" ?
        "hpc_hnsw_cuda_correlation_20260703_070901" :
        (metric == "inner_product" ?
          "hpc_hnsw_cuda_inner_product_seeded_from_euclidean_pending" :
          "hpc_hnsw_cuda_euclidean_20260701_083355"))) :
    "heuristic_fallback";
  const bool benchmark_target_met =
    benchmark_basis.find("fastest_meeting_target") != std::string::npos;

  int default_graph_degree = as<int>(base["graph_degree"]);
  int default_ef = std::max(k, 50);
  std::string rule = "balanced_cuvs_hnsw_from_cagra";
  if (benchmark_spec != nullptr) {
    default_graph_degree = benchmark_spec->graph_degree;
    default_ef = benchmark_spec->ef;
    const std::string rule_prefix = metric == "cosine" ?
      "hpc_cuda_hnsw_cosine_" :
      (metric == "correlation" ?
        "hpc_cuda_hnsw_correlation_" :
        (metric == "inner_product" ?
          "hpc_cuda_hnsw_inner_product_" :
          "hpc_cuda_hnsw_"));
    rule = rule_prefix + cuda_shape_group +
      "_k" + std::to_string(k_bucket) + "_" +
      hnsw_target_label_cpp(target_code);
  } else if (target_recall <= 0.90) {
    default_graph_degree = high_dim ? 16 : (low_dim && large_n ? 24 : 32);
    default_ef = std::max(k, large_k ? 100 : 64);
    rule = high_dim ? "recall90_high_dim_cuvs_hnsw_from_cagra" :
      "recall90_cuvs_hnsw_from_cagra";
  } else if (target_recall <= 0.95) {
    default_graph_degree = high_dim ? 24 : (low_dim && large_n ? 48 : 48);
    default_ef = std::max(k, large_k ? 150 : 96);
    rule = high_dim ? "recall95_high_dim_cuvs_hnsw_from_cagra" :
      "recall95_cuvs_hnsw_from_cagra";
  } else {
    default_graph_degree = high_dim ? 32 : (low_dim && large_n ? 96 : 64);
    default_ef = std::max(k, large_k ? 250 : 150);
    rule = high_dim ? "recall99_high_dim_cuvs_hnsw_from_cagra" :
      "recall99_cuvs_hnsw_from_cagra";
  }
  default_graph_degree = std::max(default_graph_degree, min_degree);
  default_graph_degree = std::min(default_graph_degree, n_cap);
  const int requested_graph_degree = requested_int(graph_degree_option, default_graph_degree);
  const int graph_degree = option_int(
    graph_degree_option,
    default_graph_degree,
    min_degree,
    n_cap
  );
  const int default_intermediate = benchmark_spec != nullptr ?
    std::max(graph_degree, benchmark_spec->intermediate_graph_degree) :
    std::max(
      graph_degree,
      std::max(as<int>(base["intermediate_graph_degree"]), 2 * graph_degree)
    );
  const int requested_intermediate = requested_int(
    intermediate_graph_degree_option,
    default_intermediate
  );
  const int intermediate = option_int(
    intermediate_graph_degree_option,
    default_intermediate,
    graph_degree,
    n_cap
  );
  const int requested_ef = requested_int(ef_option, default_ef);
  const int ef = option_int(ef_option, default_ef, k, 4096);
  const int threads = std::max(1, std::min(64, valid_int(n_threads) ? n_threads : 1));
  const bool compact = as<bool>(base["tuning_compact_build"]);
  const bool auto_build_algo = build_algo_preference == "auto";
  const std::string build_algo = auto_build_algo ?
    cagra_build_algo_for_shape_core(n, p, k, true, compact, "auto") :
    cagra_build_algo_for_shape_core(n, p, k, true, compact, build_algo_preference);

  return List::create(
    _["graph_degree"] = graph_degree,
    _["intermediate_graph_degree"] = intermediate,
    _["ef"] = ef,
    _["n_threads"] = threads,
    _["cagra_build_algo"] = build_algo,
    _["requested_graph_degree"] = requested_graph_degree,
    _["requested_intermediate_graph_degree"] = requested_intermediate,
    _["requested_ef"] = requested_ef,
    _["requested_n_threads"] = threads,
    _["requested_target_recall"] = target_recall,
    _["target_recall"] = target_recall,
    _["tuning_policy"] = manual_cagra ? "manual_options" : "auto_shape_k_recall",
    _["tuning_rule"] = auto_build_algo ? rule : (rule + "_manual_build_algo"),
    _["tuning_low_dim"] = low_dim,
    _["tuning_high_dim"] = high_dim,
    _["tuning_large_n"] = large_n,
    _["tuning_large_k"] = large_k,
    _["shape_group"] = cuda_shape_group,
    _["cuda_shape_group"] = cuda_shape_group,
    _["k_bucket"] = k_bucket,
    _["target_recall_code"] = target_code,
    _["tuning_metric"] = metric,
    _["benchmark_basis"] = benchmark_basis,
    _["benchmark_source"] = benchmark_source,
    _["tuning_shape_group"] = cuda_shape_group,
    _["tuning_cuda_shape_group"] = cuda_shape_group,
    _["tuning_k_bucket"] = k_bucket,
    _["tuning_target_recall_code"] = target_code,
    _["tuning_benchmark_basis"] = benchmark_basis,
    _["tuning_benchmark_source"] = benchmark_source,
    _["tuning_benchmark_target_met"] = benchmark_target_met,
    _["tuning_source"] = "cpp",
    _["cuda_hnsw_design"] = "cuvs_hnsw_from_cagra_cpu_hierarchy",
    _["cuda_hnsw_pure_gpu"] = false
  );
}

// [[Rcpp::export]]
List nn_tune_cuvs_nndescent_cpp(int n,
                                int p,
                                int k,
                                std::string metric = "euclidean",
                                double target_recall_option = NA_REAL,
                                int graph_degree_option = NA_INTEGER,
                                int intermediate_graph_degree_option = NA_INTEGER,
                                int max_iterations_option = NA_INTEGER,
                                bool manual = false) {
  n = safe_n(n);
  p = valid_int(p) ? p : NA_INTEGER;
  k = safe_k(k);
  const double target_recall = hnsw_target_recall_cpp(target_recall_option);
  const int target_code = hnsw_target_code_cpp(target_recall);
  const int k_bucket = hnsw_cpu_k_bucket_cpp(k);
  const std::string shape_group = hnsw_shape_group_cpp(n, p);
  const bool small_k = k <= 10;
  const bool large_k = k >= 100;
  const bool large_n = n >= 1000000;
  const int n_cap = std::max(1, n - 1);
  int default_graph_degree = k;
  int default_intermediate = std::max(
    static_cast<int>((3LL * default_graph_degree + 1) / 2),
    default_graph_degree
  );
  int default_max_iterations = 20;
  std::string benchmark_basis;
  std::string benchmark_source = "heuristic_fallback";
  bool benchmark_target_met = false;
  std::string rule = (large_n || large_k) ? "large_graph_search" :
    (small_k ? "small_k_speed" : "balanced_graph_search");
  if (!manual && metric == "euclidean") {
    if (const HpcCudaNndescentSpec* spec =
          hpc_cuda_nndescent_spec(shape_group, k_bucket, target_code)) {
      default_graph_degree = spec->graph_degree;
      default_intermediate = spec->intermediate_graph_degree;
      default_max_iterations = spec->max_iterations;
      benchmark_basis = spec->basis;
      benchmark_target_met =
        benchmark_basis.find("fastest_meeting_target") != std::string::npos;
      benchmark_source = "hpc_nndescent_cuda_euclidean_20260630_173056";
      rule = "hpc_cuda_nndescent_" + shape_group +
        "_k" + std::to_string(k_bucket) + "_" +
        hnsw_target_label_cpp(target_code);
    }
  } else if (!manual && metric == "cosine") {
    if (const HpcCudaNndescentSpec* spec =
          hpc_cuda_nndescent_cosine_spec(shape_group, k_bucket, target_code)) {
      default_graph_degree = spec->graph_degree;
      default_intermediate = spec->intermediate_graph_degree;
      default_max_iterations = spec->max_iterations;
      benchmark_basis = std::string(spec->basis) + "_seeded_from_euclidean";
      benchmark_target_met = false;
      benchmark_source =
        "hpc_nndescent_cuda_cosine_validation_pending_seeded_from_euclidean_20260630_173056";
      rule = "hpc_cuda_nndescent_cosine_" + shape_group +
        "_k" + std::to_string(k_bucket) + "_" +
        hnsw_target_label_cpp(target_code);
    }
  } else if (!manual && metric == "correlation") {
    if (const HpcCudaNndescentSpec* spec =
          hpc_cuda_nndescent_correlation_seed_spec(shape_group, k_bucket, target_code)) {
      default_graph_degree = spec->graph_degree;
      default_intermediate = spec->intermediate_graph_degree;
      default_max_iterations = spec->max_iterations;
      benchmark_basis = std::string(spec->basis) + "_seeded_from_euclidean";
      benchmark_target_met = false;
      benchmark_source =
        "hpc_nndescent_cuda_correlation_validation_pending_seeded_from_euclidean_20260630_173056";
      rule = "hpc_cuda_nndescent_correlation_" + shape_group +
        "_k" + std::to_string(k_bucket) + "_" +
        hnsw_target_label_cpp(target_code);
    }
  }
  const int requested_graph_degree = requested_int(graph_degree_option, default_graph_degree);
  const int graph_degree = option_int(graph_degree_option, default_graph_degree, k, n_cap);
  default_intermediate = std::max(default_intermediate, graph_degree);
  const int requested_intermediate = requested_int(
    intermediate_graph_degree_option,
    default_intermediate
  );
  const int intermediate = option_int(
    intermediate_graph_degree_option,
    default_intermediate,
    graph_degree,
    n_cap
  );
  const int requested_max_iterations = requested_int(
    max_iterations_option,
    default_max_iterations
  );
  const int max_iterations = option_int(max_iterations_option, default_max_iterations, 1, 200);
  return List::create(
    _["graph_degree"] = graph_degree,
    _["intermediate_graph_degree"] = intermediate,
    _["max_iterations"] = max_iterations,
    _["requested_graph_degree"] = requested_graph_degree,
    _["requested_intermediate_graph_degree"] = requested_intermediate,
    _["requested_max_iterations"] = requested_max_iterations,
    _["target_recall"] = target_recall,
    _["requested_target_recall"] = target_recall_option,
    _["tuning_policy"] = manual ? "manual_options" : "auto_shape_k_target_recall",
    _["tuning_rule"] = rule,
    _["tuning_metric"] = metric,
    _["tuning_backend"] = "cuda",
    _["tuning_method"] = "nndescent",
    _["tuning_shape_group"] = shape_group,
    _["tuning_k_bucket"] = k_bucket,
    _["tuning_target_recall_code"] = target_code,
    _["tuning_benchmark_basis"] = benchmark_basis,
    _["tuning_benchmark_source"] = benchmark_source,
    _["tuning_benchmark_target_met"] = benchmark_target_met,
    _["tuning_large_n"] = large_n,
    _["tuning_small_k"] = small_k,
    _["tuning_large_k"] = large_k,
    _["tuning_source"] = "cpp"
  );
}

// [[Rcpp::export]]
List nn_tune_native_nsg_cpp(int n,
                            int p,
                            int k,
                            std::string metric,
                            std::string backend,
                            double target_recall_option = NA_REAL,
                            int r_option = NA_INTEGER,
                            int graph_k_option = NA_INTEGER) {
  n = safe_n(n);
  p = safe_p(p);
  k = safe_k(k);
  const double target_recall = hnsw_target_recall_cpp(target_recall_option);
  const int target_code = hnsw_target_code_cpp(target_recall);
  const int k_bucket = hnsw_cpu_k_bucket_cpp(k);
  const std::string shape_group = hnsw_shape_group_cpp(n, p);
  const bool large_k = k >= 50;
  const bool high_dim = p >= 128;
  const bool large_n = n >= 50000;
  const bool inner_product = metric == "inner_product";
  const bool ann_seed = backend == "cpu" && large_n && high_dim;
  const bool manual = valid_int(r_option) || valid_int(graph_k_option);
  int default_r = ann_seed && !large_k ? 32 : ((large_k || high_dim || inner_product) ? 64 : 48);
  const int graph_k_cap = backend == "cuda" ? 255 : 512;
  int default_graph_k = 0;
  std::string benchmark_basis;
  std::string benchmark_source = "heuristic_fallback";
  bool benchmark_target_met = false;
  std::string rule = inner_product ? ("inner_product_" + backend + "_nsg_candidate_refine") :
    ((high_dim || large_k) ? ("high_recall_" + backend + "_nsg") : ("balanced_" + backend + "_nsg"));
  if (metric == "euclidean") {
    if (const HpcNsgSpec* spec =
          hpc_nsg_spec(backend, shape_group, k_bucket, target_code)) {
      default_r = spec->r;
      default_graph_k = spec->graph_k;
      benchmark_basis = spec->basis;
      benchmark_target_met =
        !manual && benchmark_basis.find("fastest_meeting_target") != std::string::npos;
      benchmark_source = backend == "cuda" ? "hpc_nsg_cuda_euclidean_20260702_013830" :
        "hpc_nsg_cpu12_euclidean_20260630_161409";
      rule = "hpc_" + backend + "_nsg_" + shape_group +
        "_k" + std::to_string(k_bucket) + "_" +
        hnsw_target_label_cpp(target_code);
    }
  } else if (metric == "cosine") {
    if (const HpcNsgSpec* spec =
          hpc_nsg_cosine_spec(backend, shape_group, k_bucket, target_code)) {
      default_r = spec->r;
      default_graph_k = spec->graph_k;
      benchmark_basis = spec->basis;
      benchmark_target_met =
        !manual && benchmark_basis.find("fastest_meeting_target") != std::string::npos;
      benchmark_source = backend == "cpu" ?
        "hpc_nsg_cpu12_cosine_20260701_090337" :
        "hpc_nsg_cuda_cosine_20260702_211910";
      rule = "hpc_" + backend + "_nsg_cosine_" + shape_group +
        "_k" + std::to_string(k_bucket) + "_" +
        hnsw_target_label_cpp(target_code);
    }
  } else if (metric == "correlation") {
    const HpcNsgSpec* spec =
      hpc_nsg_correlation_spec(backend, shape_group, k_bucket, target_code);
    const bool seeded_cuda_correlation = spec == nullptr && backend == "cuda";
    if (seeded_cuda_correlation) {
      spec = hpc_nsg_cosine_spec(backend, shape_group, k_bucket, target_code);
    }
    if (spec != nullptr) {
      default_r = spec->r;
      default_graph_k = spec->graph_k;
      benchmark_basis = spec->basis;
      if (seeded_cuda_correlation) benchmark_basis += "_seeded_from_cosine";
      benchmark_target_met =
        !manual && !seeded_cuda_correlation &&
        benchmark_basis.find("fastest_meeting_target") != std::string::npos;
      benchmark_source = seeded_cuda_correlation ?
        "hpc_nsg_cuda_correlation_validation_pending_seeded_from_cosine_20260702_211910" :
        (backend == "cpu" ?
           "hpc_nsg_cpu12_correlation_20260701_090337" :
           "heuristic_fallback");
      rule = "hpc_" + backend + "_nsg_correlation_" + shape_group +
        "_k" + std::to_string(k_bucket) + "_" +
        hnsw_target_label_cpp(target_code);
    }
  } else if (metric == "inner_product") {
    if (const HpcNsgSpec* spec =
          hpc_nsg_inner_product_spec(backend, shape_group, k_bucket, target_code)) {
      default_r = spec->r;
      default_graph_k = spec->graph_k;
      benchmark_basis = spec->basis;
      const bool seeded_cuda_inner_product =
        backend == "cuda" &&
        benchmark_basis.find("seeded_from_cosine") != std::string::npos;
      benchmark_target_met =
        !manual && !seeded_cuda_inner_product &&
        benchmark_basis.find("fastest_meeting_target") != std::string::npos;
      benchmark_source = benchmark_basis.find("jmlr_mloss_inner_product") != std::string::npos ?
        (backend == "cpu" ?
          "hpc_jmlr_mloss_nsg_cpu12_inner_product_20260715" :
          "hpc_jmlr_mloss_nsg_cuda_inner_product_20260716") :
        (seeded_cuda_inner_product ?
          "hpc_nsg_cuda_inner_product_validation_pending_seeded_from_cosine_20260702_211910" :
          (backend == "cpu" ?
             "hpc_nsg_cpu12_inner_product_20260701_090337" :
             "hpc_nsg_cuda_inner_product"));
      rule = "hpc_" + backend + "_nsg_inner_product_" + shape_group +
        "_k" + std::to_string(k_bucket) + "_" +
        hnsw_target_label_cpp(target_code);
    }
  }
  const int requested_r = option_int(r_option, default_r, 2, 256);
  int r = std::min(std::max(2, requested_r), std::max(1, n - 1));
  const int default_multiplier = (high_dim || inner_product) ? 3 : 2;
  if (default_graph_k <= 0) {
    default_graph_k = ann_seed ?
      std::max(std::max(k, 2 * r), large_k ? 96 : 64) :
      std::max(std::max(k, default_multiplier * r), 96);
  }
  const int requested_graph_k = option_int(graph_k_option, default_graph_k, k, graph_k_cap);
  int graph_k = std::min(std::min(std::max(k, requested_graph_k), std::max(1, n - 1)), graph_k_cap);
  if (r > graph_k) r = graph_k;
  return List::create(
    _["r"] = r,
    _["graph_k"] = graph_k,
    _["requested_r"] = requested_r,
    _["requested_graph_k"] = requested_graph_k,
    _["backend"] = backend,
    _["graph_k_cap"] = graph_k_cap,
    _["seed_backend"] = ann_seed ? "faiss_hnsw" : "exact",
    _["seed_k"] = graph_k,
    _["target_recall"] = target_recall,
    _["requested_target_recall"] = target_recall_option,
    _["tuning_policy"] = manual ? "manual_options" : "auto_shape_k_metric_target_recall",
    _["tuning_rule"] = manual ? ("manual_" + backend + "_nsg") : rule,
    _["tuning_metric"] = metric,
    _["tuning_shape_group"] = shape_group,
    _["tuning_k_bucket"] = k_bucket,
    _["tuning_target_recall_code"] = target_code,
    _["tuning_benchmark_basis"] = benchmark_basis,
    _["tuning_benchmark_target_met"] = benchmark_target_met,
    _["tuning_benchmark_source"] = benchmark_source,
    _["tuning_large_k"] = large_k,
    _["tuning_high_dim"] = high_dim,
    _["tuning_source"] = "cpp"
  );
}

// [[Rcpp::export]]
List nn_tune_vamana_cpp(int n,
                        int p,
                        int k,
                        std::string metric,
                        std::string backend = "cpu",
                        double target_recall_option = NA_REAL,
                        int r_option = NA_INTEGER,
                        int search_l_option = NA_INTEGER,
                        double alpha_option = NA_REAL) {
  n = safe_n(n);
  p = safe_p(p);
  k = safe_k(k);
  const double target_recall = hnsw_target_recall_cpp(target_recall_option);
  const int target_code = hnsw_target_code_cpp(target_recall);
  const int k_bucket = hnsw_cpu_k_bucket_cpp(k);
  const std::string shape_group = hnsw_shape_group_cpp(n, p);
  const bool large_k = k >= 50;
  const bool high_dim = p >= 128;
  const bool large_n = n >= 50000;
  const bool ann_seed = backend == "cpu" && large_n && high_dim;
  const bool manual = valid_int(r_option) || valid_int(search_l_option) || valid_double(alpha_option);
  int default_r = ann_seed && !large_k ? 32 : ((high_dim || large_k) ? 64 : 48);
  int default_search_l = 0;
  double default_alpha = 1.2;
  std::string benchmark_basis;
  std::string benchmark_source = "heuristic_fallback";
  bool benchmark_target_met = false;
  std::string rule = metric == "inner_product" ? "inner_product_vamana_candidate_refine" :
    ((high_dim || large_k) ? "high_recall_vamana" : "balanced_vamana");
  if (metric == "euclidean") {
    if (const HpcVamanaSpec* spec =
          hpc_vamana_spec(backend, shape_group, k_bucket, target_code)) {
      default_r = spec->r;
      default_search_l = spec->search_l;
      default_alpha = spec->alpha;
      benchmark_basis = spec->basis;
      benchmark_target_met =
        !manual && benchmark_basis.find("fastest_meeting_target") != std::string::npos;
      benchmark_source = backend == "cuda" ? "hpc_vamana_cuda_euclidean_20260702_042943" :
        "hpc_vamana_cpu12_euclidean_20260630_161409";
      rule = "hpc_" + backend + "_vamana_" + shape_group +
        "_k" + std::to_string(k_bucket) + "_" +
        hnsw_target_label_cpp(target_code);
    }
  } else if (metric == "cosine") {
    if (const HpcVamanaSpec* spec =
          hpc_vamana_cosine_spec(backend, shape_group, k_bucket, target_code)) {
      default_r = spec->r;
      default_search_l = spec->search_l;
      default_alpha = spec->alpha;
      benchmark_basis = spec->basis;
      benchmark_target_met =
        !manual && benchmark_basis.find("fastest_meeting_target") != std::string::npos;
      benchmark_source = backend == "cpu" ?
        "hpc_vamana_cpu12_cosine_20260701_090337" :
        "hpc_vamana_cuda_cosine_20260702_232209";
      rule = "hpc_" + backend + "_vamana_cosine_" + shape_group +
        "_k" + std::to_string(k_bucket) + "_" +
        hnsw_target_label_cpp(target_code);
    }
  } else if (metric == "correlation") {
    const HpcVamanaSpec* spec =
      hpc_vamana_correlation_spec(backend, shape_group, k_bucket, target_code);
    const bool seeded_cuda_correlation = spec == nullptr && backend == "cuda";
    if (seeded_cuda_correlation) {
      spec = hpc_vamana_cosine_spec(backend, shape_group, k_bucket, target_code);
    }
    if (spec != nullptr) {
      default_r = spec->r;
      default_search_l = spec->search_l;
      default_alpha = spec->alpha;
      benchmark_basis = spec->basis;
      if (seeded_cuda_correlation) benchmark_basis += "_seeded_from_cosine";
      benchmark_target_met =
        !manual && !seeded_cuda_correlation &&
        benchmark_basis.find("fastest_meeting_target") != std::string::npos;
      benchmark_source = seeded_cuda_correlation ?
        "hpc_vamana_cuda_correlation_validation_pending_seeded_from_cosine_20260702_232209" :
        (backend == "cpu" ?
           "hpc_vamana_cpu12_correlation_20260701_090337" :
           "heuristic_fallback");
      rule = "hpc_" + backend + "_vamana_correlation_" + shape_group +
        "_k" + std::to_string(k_bucket) + "_" +
        hnsw_target_label_cpp(target_code);
    }
  } else if (metric == "inner_product") {
    if (const HpcVamanaSpec* spec =
          hpc_vamana_inner_product_spec(backend, shape_group, k_bucket, target_code)) {
      default_r = spec->r;
      default_search_l = spec->search_l;
      default_alpha = spec->alpha;
      benchmark_basis = spec->basis;
      const bool seeded_cuda_inner_product =
        backend == "cuda" &&
        benchmark_basis.find("seeded_from_cosine") != std::string::npos;
      benchmark_target_met =
        !manual && !seeded_cuda_inner_product &&
        benchmark_basis.find("fastest_meeting_target") != std::string::npos;
      benchmark_source = benchmark_basis.find("jmlr_mloss_inner_product") != std::string::npos ?
        (backend == "cpu" ?
          "hpc_jmlr_mloss_vamana_cpu12_inner_product_20260715" :
          "hpc_jmlr_mloss_vamana_cuda_inner_product_20260716") :
        (seeded_cuda_inner_product ?
          "hpc_vamana_cuda_inner_product_validation_pending_seeded_from_cosine_20260702_232209" :
          (backend == "cpu" ?
             "hpc_vamana_cpu12_inner_product_20260701_090337" :
             "hpc_vamana_cuda_inner_product"));
      rule = "hpc_" + backend + "_vamana_inner_product_" + shape_group +
        "_k" + std::to_string(k_bucket) + "_" +
        hnsw_target_label_cpp(target_code);
    }
  }
  const int requested_r = option_int(r_option, default_r, 2, 256);
  int r = std::min(std::max(2, requested_r), std::max(1, n - 1));
  if (default_search_l <= 0) {
    default_search_l = ann_seed ?
      std::max(std::max(k, 2 * r), large_k ? 96 : 64) :
      std::max(std::max(k, 2 * r), 96);
  }
  const int requested_search_l = option_int(search_l_option, default_search_l, k, 512);
  int search_l = std::min(std::min(std::max(k, requested_search_l), std::max(1, n - 1)), 512);
  if (r > search_l) r = search_l;
  const double requested_alpha = valid_double(alpha_option) && alpha_option >= 1.0 ? alpha_option : default_alpha;
  const double alpha = std::max(1.0, std::min(2.0, requested_alpha));
  return List::create(
    _["r"] = r,
    _["search_l"] = search_l,
    _["alpha"] = alpha,
    _["requested_r"] = requested_r,
    _["requested_search_l"] = requested_search_l,
    _["requested_alpha"] = requested_alpha,
    _["seed_backend"] = ann_seed ? "faiss_hnsw" : "exact",
    _["seed_k"] = search_l,
    _["backend"] = backend,
    _["target_recall"] = target_recall,
    _["requested_target_recall"] = target_recall_option,
    _["tuning_policy"] = manual ? "manual_options" : "auto_shape_k_metric_target_recall",
    _["tuning_rule"] = manual ? ("manual_" + backend + "_vamana") : rule,
    _["tuning_metric"] = metric,
    _["tuning_shape_group"] = shape_group,
    _["tuning_k_bucket"] = k_bucket,
    _["tuning_target_recall_code"] = target_code,
    _["tuning_benchmark_basis"] = benchmark_basis,
    _["tuning_benchmark_target_met"] = benchmark_target_met,
    _["tuning_benchmark_source"] = benchmark_source,
    _["tuning_large_k"] = large_k,
    _["tuning_high_dim"] = high_dim,
    _["tuning_source"] = "cpp"
  );
}

std::string kmeans_rule_detail_cpp(int n,
                                   int p,
                                   int centers,
                                   double n_per_center,
                                   double work) {
  std::ostringstream out;
  out << "n=" << n
      << ";p=" << p
      << ";centers=" << centers
      << ";n_per_center=" << n_per_center
      << ";work=" << std::scientific << work;
  return out.str();
}

std::string kmeans_auto_rule_label_cpp(int n,
                                       int centers,
                                       bool large_n,
                                       bool high_dim,
                                       bool many_centers,
                                       bool small_many_centers,
                                       bool few_points_many_centers,
                                       double work) {
  if (centers == 1) return "single_cluster_exact_mean";
  if (centers == n) return "singleton_exact_identity";
  if (large_n || work >= 5e9) return "large_fast_convergence";
  if (small_many_centers) return "small_many_centers_multistart";
  if (few_points_many_centers) return "few_points_many_centers_multistart";
  if (n <= 50000 && centers <= 20 && work <= 2e8) {
    return "small_low_work_multistart";
  }
  if (n <= 100000 && centers <= 50 && work <= 5e8) {
    return "medium_multistart";
  }
  if (high_dim || many_centers || work >= 5e8) return "medium_single_start";
  return "small_single_start";
}

// [[Rcpp::export]]
List kmeans_auto_params_cpp(int n,
                            int p,
                            int centers,
                            std::string tuning) {
  n = safe_n(n);
  p = safe_p(p);
  centers = safe_k(centers);
  const double work = static_cast<double>(n) *
    static_cast<double>(p) * static_cast<double>(centers);
  const bool high_dim = p >= 256;
  const bool large_n = n >= 100000;
  const bool many_centers = centers >= 100;
  const double n_per_center = static_cast<double>(n) / static_cast<double>(centers);
  const bool small_many_centers = many_centers && n <= 50000 &&
    work <= 2e8 && n_per_center >= 20.0;
  const bool few_points_many_centers = many_centers && n <= 50000 &&
    work <= 2e8 && n_per_center < 20.0;
  const std::string rule_detail = kmeans_rule_detail_cpp(
    n, p, centers, n_per_center, work
  );

  if (tuning != "auto") {
    return List::create(
      _["policy"] = tuning,
      _["max_iter"] = 100,
      _["n_init"] = 1,
      _["tol"] = 1e-4,
      _["work"] = work,
      _["n_per_center"] = n_per_center,
      _["high_dim"] = high_dim,
      _["large_n"] = large_n,
      _["many_centers"] = many_centers,
      _["small_many_centers"] = small_many_centers,
      _["few_points_many_centers"] = few_points_many_centers,
      _["rule"] = "fixed_defaults",
      _["rule_detail"] = rule_detail,
      _["tuning_source"] = "cpp"
    );
  }

  if (centers == 1) {
    return List::create(
      _["policy"] = "auto",
      _["max_iter"] = 1,
      _["n_init"] = 1,
      _["tol"] = 0.0,
      _["work"] = work,
      _["n_per_center"] = n_per_center,
      _["high_dim"] = high_dim,
      _["large_n"] = large_n,
      _["many_centers"] = false,
      _["small_many_centers"] = false,
      _["few_points_many_centers"] = false,
      _["rule"] = "single_cluster_exact_mean",
      _["rule_detail"] = rule_detail,
      _["tuning_source"] = "cpp"
    );
  }

  if (centers == n) {
    return List::create(
      _["policy"] = "auto",
      _["max_iter"] = 1,
      _["n_init"] = 1,
      _["tol"] = 0.0,
      _["work"] = work,
      _["n_per_center"] = n_per_center,
      _["high_dim"] = high_dim,
      _["large_n"] = large_n,
      _["many_centers"] = true,
      _["small_many_centers"] = false,
      _["few_points_many_centers"] = true,
      _["rule"] = "singleton_exact_identity",
      _["rule_detail"] = rule_detail,
      _["tuning_source"] = "cpp"
    );
  }

  int max_iter = 100;
  if (large_n || work >= 5e9) {
    max_iter = 50;
  } else if (high_dim ||
             (many_centers && !small_many_centers && !few_points_many_centers) ||
             work >= 5e8) {
    max_iter = 75;
  }

  int n_init = 1;
  if (n <= 50000 && centers <= 20 && work <= 2e8) {
    n_init = 5;
  } else if (small_many_centers || few_points_many_centers) {
    n_init = 3;
  } else if (n <= 100000 && centers <= 50 && work <= 5e8) {
    n_init = 3;
  }

  const double tol = (large_n || work >= 5e9) ? 1e-3 : 1e-4;
  const std::string rule = kmeans_auto_rule_label_cpp(
    n, centers, large_n, high_dim, many_centers,
    small_many_centers, few_points_many_centers, work
  );

  return List::create(
    _["policy"] = "auto",
    _["max_iter"] = max_iter,
    _["n_init"] = n_init,
    _["tol"] = tol,
    _["work"] = work,
    _["n_per_center"] = n_per_center,
    _["high_dim"] = high_dim,
    _["large_n"] = large_n,
    _["many_centers"] = many_centers,
    _["small_many_centers"] = small_many_centers,
    _["few_points_many_centers"] = few_points_many_centers,
    _["rule"] = rule,
    _["rule_detail"] = rule_detail,
    _["tuning_source"] = "cpp"
  );
}

// [[Rcpp::export]]
List kmeans_auto_backend_policy_cpp(int n,
                                    int p,
                                    int centers,
                                    double work_threshold,
                                    double nbytes_threshold,
                                    int large_n_threshold,
                                    int large_p_threshold,
                                    double min_n_per_center) {
  if (!valid_double(work_threshold) || work_threshold < 1.0) work_threshold = 1e8;
  if (!valid_double(nbytes_threshold) || nbytes_threshold < 1.0) {
    nbytes_threshold = 256.0 * 1024.0 * 1024.0;
  }
  if (!valid_int(large_n_threshold) || large_n_threshold < 1) large_n_threshold = 50000;
  if (!valid_int(large_p_threshold) || large_p_threshold < 1) large_p_threshold = 128;
  if (!valid_double(min_n_per_center) || min_n_per_center < 1.0) min_n_per_center = 20.0;

  auto base = [&](bool prefer_cuda, const std::string& reason,
                  double work, double nbytes, double n_per_center) {
    const double gpu_transfer_nbytes = valid_double(nbytes) ? nbytes / 2.0 : NA_REAL;
    return List::create(
      _["prefer_cuda"] = prefer_cuda,
      _["reason"] = reason,
      _["work"] = work,
      _["nbytes"] = nbytes,
      _["input_nbytes"] = nbytes,
      _["gpu_transfer_nbytes"] = gpu_transfer_nbytes,
      _["n_per_center"] = n_per_center,
      _["work_threshold"] = work_threshold,
      _["nbytes_threshold"] = nbytes_threshold,
      _["large_n_threshold"] = large_n_threshold,
      _["large_p_threshold"] = large_p_threshold,
      _["min_n_per_center"] = min_n_per_center,
      _["tuning_source"] = "cpp"
    );
  };

  if (!valid_int(n) || !valid_int(p) || !valid_int(centers)) {
    return base(true, "unknown_shape", NA_REAL, NA_REAL, NA_REAL);
  }
  n = safe_n(n);
  p = safe_p(p);
  centers = safe_k(centers);
  const double work = static_cast<double>(n) *
    static_cast<double>(p) * static_cast<double>(centers);
  const double nbytes = static_cast<double>(n) * static_cast<double>(p) * 8.0;
  const double n_per_center = static_cast<double>(n) / static_cast<double>(centers);

  if (centers == 1) {
    return base(false, "single_cluster_exact_mean", work, nbytes, n_per_center);
  }
  if (centers == n) {
    return base(false, "singleton_exact_identity", work, nbytes, n_per_center);
  }
  if (n_per_center < min_n_per_center && (nbytes / 2.0) < nbytes_threshold) {
    return base(false, "few_points_per_center_cpu_preferred", work, nbytes, n_per_center);
  }

  bool prefer = work >= work_threshold ||
    (nbytes / 2.0) >= nbytes_threshold ||
    (n >= large_n_threshold && p >= large_p_threshold);
  std::string reason;
  if (work >= work_threshold) {
    reason = "work_at_least_1e8";
  } else if ((nbytes / 2.0) >= nbytes_threshold) {
    reason = "input_at_least_256MiB";
  } else if (n >= large_n_threshold && p >= large_p_threshold) {
    reason = "large_high_dimensional_input";
  } else {
    reason = "small_cpu_preferred";
  }
  return base(prefer, reason, work, nbytes, n_per_center);
}

// [[Rcpp::export]]
List kmeans_auto_select_backend_cpp(std::string requested_backend,
                                    int n,
                                    int p,
                                    int centers,
                                    double work_threshold,
                                    double nbytes_threshold,
                                    int large_n_threshold,
                                    int large_p_threshold,
                                    double min_n_per_center,
                                    bool cuda_available,
                                    bool faiss_gpu_available,
                                    bool cuvs_available,
                                    int effective_max_iter = NA_INTEGER,
                                    int effective_n_init = NA_INTEGER,
                                    double effective_tol = NA_REAL,
                                    std::string tuning = "auto") {
  if (!(requested_backend == "auto" ||
        requested_backend == "cpu" ||
        requested_backend == "cuda")) {
    stop("`requested_backend` must be one of \"auto\", \"cpu\", or \"cuda\".");
  }

  List policy = kmeans_auto_backend_policy_cpp(
    n, p, centers, work_threshold, nbytes_threshold,
    large_n_threshold, large_p_threshold, min_n_per_center
  );
  const bool prefer_cuda = as<bool>(policy["prefer_cuda"]);
  const std::string policy_reason = as<std::string>(policy["reason"]);
  const bool explicit_backend = requested_backend != "auto";
  const bool cuda_kmeans_route_available =
    cuda_available && (faiss_gpu_available || cuvs_available);

  std::string resolved_backend = requested_backend;
  std::string runtime_decision;
  if (explicit_backend) {
    runtime_decision = "explicit_backend_no_auto_fallback";
  } else if (prefer_cuda && cuda_kmeans_route_available) {
    resolved_backend = "cuda";
    runtime_decision = "cuda_kmeans_route_available";
  } else {
    resolved_backend = "cpu";
    if (!prefer_cuda) {
      runtime_decision = "cpu_preferred_by_shape";
    } else if (!cuda_available) {
      runtime_decision = "cuda_runtime_unavailable";
    } else {
      runtime_decision = "cuda_kmeans_provider_unavailable";
    }
  }

  const std::string backend_decision = explicit_backend ?
    ("explicit_" + requested_backend) : policy_reason;

  return List::create(
    _["policy"] = "static_shape_center_backend_selector",
    _["slow_tuning"] = false,
    _["requested_backend"] = requested_backend,
    _["predicted_backend"] = resolved_backend,
    _["resolved_backend"] = resolved_backend,
    _["n"] = n,
    _["p"] = p,
    _["centers"] = centers,
    _["work"] = policy["work"],
    _["nbytes"] = policy["nbytes"],
    _["input_nbytes"] = policy["input_nbytes"],
    _["gpu_transfer_nbytes"] = policy["gpu_transfer_nbytes"],
    _["n_per_center"] = policy["n_per_center"],
    _["backend_policy_reason"] = policy_reason,
    _["explicit_backend"] = explicit_backend,
    _["backend_decision"] = backend_decision,
    _["backend_policy_prefer_cuda"] = prefer_cuda,
    _["cuda_available"] = cuda_available,
    _["faiss_gpu_available"] = faiss_gpu_available,
    _["cuvs_available"] = cuvs_available,
    _["cuda_kmeans_route_available"] = cuda_kmeans_route_available,
    _["runtime_decision"] = runtime_decision,
    _["effective_max_iter"] = valid_int(effective_max_iter) ? effective_max_iter : NA_INTEGER,
    _["effective_n_init"] = valid_int(effective_n_init) ? effective_n_init : NA_INTEGER,
    _["effective_tol"] = valid_double(effective_tol) ? effective_tol : NA_REAL,
    _["tuning"] = tuning,
    _["tuning_source"] = "cpp",
    _["backend_policy"] = policy
  );
}
