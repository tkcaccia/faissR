#include <Rcpp.h>
#include <algorithm>
#include <cmath>
#include <sstream>
#include <string>

using namespace Rcpp;

namespace {

bool finite1(double x) {
  return std::isfinite(x);
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

std::string cagra_backend(bool faiss_gpu_available,
                          bool cuvs_available,
                          const std::string& preference,
                          int n,
                          int p,
                          int k,
                          bool self_query,
                          int compact_n,
                          int high_dim_p,
                          int compact_max_k) {
  if (preference == "faiss_gpu") return "faiss_gpu_cagra";
  if (preference == "cuvs") return "cuda_cuvs_cagra";
  const bool compact_high_dim = self_query &&
    n <= compact_n && p >= high_dim_p && k <= compact_max_k;
  if (faiss_gpu_available && cuvs_available && compact_high_dim) {
    return "cuda_cuvs_cagra";
  }
  if (faiss_gpu_available) return "faiss_gpu_cagra";
  return "cuda_cuvs_cagra";
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
  if (backend == "hnsw" || backend == "rcpphnsw" ||
      backend == "cpu_hnsw" || backend == "faiss_hnsw" ||
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
  if (backend == "cpu_vamana" || backend == "cuda_vamana") return "vamana";
  if (backend == "faiss_nsg" || backend == "cpu_nsg" ||
      backend == "cuda_nsg") return "nsg";
  if (backend == "cpu_nndescent" || backend == "faiss_nndescent" ||
      backend == "cuda_cuvs_nndescent" || backend == "cuvs_nndescent" ||
      backend == "cuda_native_nndescent" || backend == "cuda_nndescent" ||
      backend == "cuda_approx" || backend == "gpu_nndescent" ||
      backend == "gpu_approx") return "nndescent";
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
  const std::string flat_backend =
    metric == "cosine" ? "faiss_gpu_flat_cosine" :
    metric == "correlation" ? "faiss_gpu_flat_correlation" :
    "faiss_gpu_flat_ip";
  const bool cagra_available = faiss_gpu_available || cuvs_available;
  const bool large_self_graph = self_query &&
    n >= graph_n && n_points >= graph_n && k >= graph_min_k &&
    finite1(work_size) && work_size >= graph_work;
  if (large_self_graph && cagra_available) {
    return cagra_backend(
      faiss_gpu_available, cuvs_available, cagra_preference,
      n, p, k, self_query,
      cagra_compact_n, cagra_high_dim_p, cagra_compact_max_k
    );
  }
  if (faiss_gpu_available) return flat_backend;
  if (metric == "inner_product" && cuda_available && self_query) {
    return "cuda_native_nndescent";
  }
  if (requested_device == "auto") return "cpu_auto";
  return "__cuda_non_euclidean_unavailable__";
}

std::string select_cuvs_auto(bool self_query,
                             int n,
                             int p,
                             int n_points,
                             int k,
                             double work_size,
                             double brute_force_work_threshold) {
  if (!self_query || work_size <= brute_force_work_threshold ||
      k <= 8 || n <= 100000 || n_points <= 5000) {
    return "cuda_cuvs_bruteforce";
  }
  if (p <= 64) return "cuda_cuvs_ivf_flat";
  return "cuda_cuvs_nndescent";
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
                        std::string& error) {
  if (k > 256) {
    error = "CUDA auto backends currently support `k <= 256`.";
    return "cuda_auto";
  }
  if (!cuda_available && !cuvs_available) {
    error = "No CUDA GPU backend is available on this machine.";
    return "cuda_auto";
  }
  if (grid_self_knn(self_query, n, p, k, false, metric) && cuda_available) {
    return "cuda_grid";
  }
  const std::string metric_backend = cuda_non_euclidean_backend(
    metric, "cuda", self_query, n, p, n_points, k, work_size,
    cuda_available, cuvs_available, faiss_gpu_available, cagra_preference,
    metric_graph_n, metric_graph_min_k, metric_graph_work,
    cagra_compact_n, cagra_high_dim_p, cagra_compact_max_k
  );
  if (metric_backend == "__cuda_non_euclidean_unavailable__") {
    error = "CUDA auto for non-Euclidean metrics requires FAISS GPU Flat support "
      "for exact routes or CAGRA/cuVS support for large self-KNN graph routes. "
      "Use `backend = \"auto\"` to fall back to CPU, or rebuild faissR with "
      "FAISS GPU/cuVS support.";
    return "cuda_auto";
  }
  if (!metric_backend.empty()) return metric_backend;
  if (!self_query) {
    if (faiss_gpu_available) return "faiss_gpu_flat_l2";
    if (cuvs_available) return "cuda_cuvs_bruteforce";
    return "cuda";
  }
  const bool compact_high_dim_self =
    cuvs_available && n == n_points && n <= cagra_compact_n &&
    p >= cagra_high_dim_p && k <= cagra_compact_max_k;
  if (compact_high_dim_self) return "cuda_cuvs_bruteforce";
  const int cuvs_exact_high_dim_p = std::min(cagra_high_dim_p, 128);
  const bool compact_exact_self =
    cuvs_available && n == n_points && n <= cagra_compact_n &&
    p >= cuvs_exact_high_dim_p && k <= cagra_compact_max_k;
  if (compact_exact_self) return "cuda_cuvs_bruteforce";
  if (n <= cuda_exact_n || work_size <= cuda_exact_work || k <= 8) {
    if (faiss_gpu_available) return "faiss_gpu_flat_l2";
    if (cuvs_available) return "cuda_cuvs_bruteforce";
    return "cuda";
  }
  if (faiss_gpu_available) return "faiss_gpu_cagra";
  if (cuvs_available) {
    return select_cuvs_auto(
      self_query, n, p, n_points, k, work_size,
      cuvs_bruteforce_work_threshold
    );
  }
  return "cuda";
}

std::string select_cpu(bool self_query,
                       int n,
                       int p,
                       int n_points,
                       int k,
                       double work_size,
                       const std::string& metric,
                       bool faiss_available,
                       bool rcpphnsw_available,
                       double cpu_exact_work,
                       double cpu_faiss_flat_work) {
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
    if (rcpphnsw_available) return "hnsw";
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
  if (self_query && n >= 1000000 && faiss_available) return "faiss_ivf";
  if (faiss_available) return "faiss_hnsw";
  if (rcpphnsw_available) return "hnsw";
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

double option_double(double value,
                     double fallback,
                     double min_value,
                     double max_value) {
  if (!valid_double(value)) value = fallback;
  value = std::max(min_value, std::min(max_value, value));
  return value;
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
  const int candidates[] = {64, 56, 48, 40, 32, 28, 24, 16, 14, 12, 8, 7, 4, 2, 1};
  for (int candidate : candidates) {
    if (candidate <= p && p % candidate == 0) return candidate;
  }
  return 1;
}

List cuvs_cagra_params_core(int n,
                            int p,
                            int k,
                            int graph_degree_option,
                            int intermediate_graph_degree_option,
                            int search_width_option,
                            int itopk_size_option,
                            bool manual) {
  n = safe_n(n);
  p = safe_p(p);
  k = safe_k(k);
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
  const int default_graph_degree = compact_build ? std::max(32, k + 1) : std::max(64, k + 1);
  const int requested_graph_degree = requested_int(graph_degree_option, default_graph_degree);
  const int graph_degree = option_int(graph_degree_option, default_graph_degree, k + 1, n_cap);
  const int default_intermediate = compact_build ?
    std::max(32, graph_degree * 2) : std::max(128, graph_degree * 2);
  const int requested_intermediate = requested_int(intermediate_graph_degree_option, default_intermediate);
  const int intermediate = option_int(
    intermediate_graph_degree_option,
    default_intermediate,
    graph_degree,
    n_cap
  );
  const int requested_search_width = requested_int(search_width_option, 0);
  const int search_width = option_int(search_width_option, 0, 0, 1024);
  const int default_itopk = compact_build ?
    std::max(std::max(32, graph_degree), k) : std::max(64, graph_degree);
  const int requested_itopk = requested_int(itopk_size_option, default_itopk);
  const int itopk = option_int(itopk_size_option, default_itopk, k, 4096);

  return List::create(
    _["graph_degree"] = graph_degree,
    _["intermediate_graph_degree"] = intermediate,
    _["search_width"] = search_width,
    _["itopk_size"] = itopk,
    _["requested_graph_degree"] = requested_graph_degree,
    _["requested_intermediate_graph_degree"] = requested_intermediate,
    _["requested_search_width"] = requested_search_width,
    _["requested_itopk_size"] = requested_itopk,
    _["tuning_policy"] = manual ? "manual_options" : "auto_shape_k",
    _["tuning_rule"] = rule,
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
                                bool rcpphnsw_available,
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
                                std::string tuning) {
  const double work_size = static_cast<double>(n) *
    static_cast<double>(n_points) * static_cast<double>(p);
  std::string selected = resolved_backend;
  std::string reason = "explicit_route";
  std::string error;

  if (resolved_backend == "auto") {
    std::string cuda_error;
    std::string gpu;
    if (self_query && k <= 256 && work_size >= 5e8 &&
        (cuda_available || cuvs_available)) {
      gpu = select_cuda(
        self_query, n, p, n_points, k, work_size, metric,
        cuda_available, cuvs_available, faiss_gpu_available,
        cagra_preference, cuda_exact_n, cuda_exact_work,
        metric_graph_n, metric_graph_min_k, metric_graph_work,
        cagra_compact_n, cagra_high_dim_p, cagra_compact_max_k,
        cuvs_bruteforce_work_threshold, cuda_error
      );
    }
    if (!gpu.empty() && cuda_error.empty() && gpu != "cpu_auto") {
      selected = gpu;
      reason = "auto_cuda_preselector";
    } else {
      selected = select_cpu(
        self_query, n, p, n_points, k, work_size, metric,
        faiss_available, rcpphnsw_available,
        cpu_exact_work, cpu_faiss_flat_work
      );
      reason = "auto_cpu_fallback";
    }
  } else if (resolved_backend == "cpu_auto") {
    selected = select_cpu(
      self_query, n, p, n_points, k, work_size, metric,
      faiss_available, rcpphnsw_available,
      cpu_exact_work, cpu_faiss_flat_work
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
      cuvs_bruteforce_work_threshold, cuda_error
    );
    if (cuda_error.empty()) {
      reason = "cuda_auto_shape_selector";
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
    _["tuning"] = tuning
  );
}

// [[Rcpp::export]]
List nn_tune_faiss_ivf_cpp(int n,
                           int k,
                           std::string metric,
                           int nlist_option = NA_INTEGER,
                           int nprobe_option = NA_INTEGER,
                           bool manual = false) {
  n = safe_n(n);
  k = safe_k(k);
  const bool small_k = k <= 10;
  const bool large_k = k >= 100;
  const bool large_n = n >= 1000000;
  const bool metric_aware = metric != "euclidean";
  std::string base_rule = large_n ? "large_n_coarse_quantizer" :
    (large_k ? "large_k_more_probe" : (small_k ? "small_k_speed" : "balanced_shape_k"));
  std::string rule = (metric_aware && !manual) ? ("metric_" + base_rule) : base_rule;

  const int default_nlist = ivf_list_count_cpp(n, k);
  const int requested_nlist = requested_int(nlist_option, default_nlist);
  const int nlist = option_int(nlist_option, default_nlist, 1, n);
  const int default_nprobe = ivf_probe_count_cpp(nlist, k, metric);
  const int requested_nprobe = requested_int(nprobe_option, default_nprobe);
  const int nprobe = option_int(nprobe_option, default_nprobe, 1, nlist);

  return List::create(
    _["nlist"] = nlist,
    _["nprobe"] = nprobe,
    _["requested_nlist"] = requested_nlist,
    _["requested_nprobe"] = requested_nprobe,
    _["tuning_policy"] = manual ? "manual_options" : "auto_shape_k",
    _["tuning_rule"] = rule,
    _["tuning_metric"] = metric,
    _["tuning_metric_aware"] = metric_aware,
    _["tuning_large_n"] = large_n,
    _["tuning_small_k"] = small_k,
    _["tuning_large_k"] = large_k,
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
  const int pq_bits = option_int(pq_bits_option, pq_bits_default, 4, 8);
  std::string rule = (reduced_codebook_training && !manual_bits) ? "training_rows_4bit_pq" :
    (high_dim ? "high_dim_default_pq" : "dimension_default_pq");
  return List::create(
    _["pq_dim"] = pq_dim,
    _["pq_bits"] = pq_bits,
    _["requested_pq_dim"] = requested_pq_dim,
    _["requested_pq_bits"] = requested_pq_bits,
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
                            int m_option = NA_INTEGER,
                            int ef_construction_option = NA_INTEGER,
                            int ef_search_option = NA_INTEGER,
                            bool manual = false) {
  n = valid_int(n) ? n : NA_INTEGER;
  p = valid_int(p) ? p : NA_INTEGER;
  k = safe_k(k);
  const bool high_dim = valid_int(p) && p >= 256;
  const bool large_n = valid_int(n) && n >= 50000;
  const bool very_large_high_dim = large_n && high_dim;
  const bool small_k = k <= 10;
  const bool large_k = k >= 100;
  const bool non_euclidean = metric != "euclidean";

  std::string rule;
  int default_m;
  int default_ef_construction;
  int default_ef_search;
  if (non_euclidean && small_k) {
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
    _["policy"] = manual ? "manual_options" : "auto_shape_metric",
    _["high_dim"] = high_dim,
    _["large_n"] = large_n,
    _["small_k"] = small_k,
    _["large_k"] = large_k,
    _["non_euclidean"] = non_euclidean,
    _["requested_m"] = default_m,
    _["requested_ef_construction"] = default_ef_construction,
    _["requested_ef_search"] = default_ef_search,
    _["tuning_source"] = "cpp"
  );
}

// [[Rcpp::export]]
List nn_tune_rcpphnsw_cpp(int k,
                          int m_option = NA_INTEGER,
                          int ef_construction_option = NA_INTEGER,
                          int ef_option = NA_INTEGER) {
  k = safe_k(k);
  const int m = option_int(m_option, 16, 2, 256);
  const int ef_construction = option_int(
    ef_construction_option,
    std::max(200, m),
    m,
    4096
  );
  const int ef = option_int(
    ef_option,
    std::max(50, 3 * k),
    k,
    4096
  );
  return List::create(
    _["m"] = m,
    _["ef_construction"] = ef_construction,
    _["ef"] = ef,
    _["tuning_policy"] = "auto_k",
    _["tuning_rule"] = k >= 100 ? "large_k_hnswlib" : (k <= 10 ? "small_k_hnswlib" : "balanced_hnswlib"),
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
List nn_tune_cpu_nndescent_cpp(int n, int k) {
  n = safe_n(n);
  k = safe_k(k);
  const int n_cap = std::max(1, n - 1);
  const int pool_size = std::min(
    n_cap,
    std::max(k + 15, std::min(160, static_cast<int>(std::ceil(2.5 * k))))
  );
  int n_iters = n >= 50000 ? 3 : 4;
  if (k < 30) ++n_iters;
  const int max_candidates = std::min(n_cap, std::max(pool_size * 4, k * 12));
  const int n_random_projections = n >= 50000 ? 8 : 6;
  return List::create(
    _["pool_size"] = pool_size,
    _["n_iters"] = n_iters,
    _["max_candidates"] = max_candidates,
    _["n_random_projections"] = n_random_projections,
    _["tuning_policy"] = "auto_shape_k",
    _["tuning_rule"] = n >= 50000 ? "large_n_random_projection_seed" : "balanced_random_projection_seed",
    _["tuning_large_n"] = n >= 50000,
    _["tuning_small_k"] = k < 30,
    _["tuning_source"] = "cpp"
  );
}

// [[Rcpp::export]]
List nn_tune_cuvs_cagra_cpp(int n,
                            int p,
                            int k,
                            int graph_degree_option = NA_INTEGER,
                            int intermediate_graph_degree_option = NA_INTEGER,
                            int search_width_option = NA_INTEGER,
                            int itopk_size_option = NA_INTEGER,
                            bool manual = false) {
  return cuvs_cagra_params_core(
    n, p, k, graph_degree_option, intermediate_graph_degree_option,
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
                           int graph_degree_option = NA_INTEGER,
                           int intermediate_graph_degree_option = NA_INTEGER,
                           int search_width_option = NA_INTEGER,
                           int itopk_size_option = NA_INTEGER,
                           int ef_option = NA_INTEGER,
                           bool manual_cagra = false) {
  List base = cuvs_cagra_params_core(
    n, p, k, graph_degree_option, intermediate_graph_degree_option,
    search_width_option, itopk_size_option, manual_cagra
  );
  n = safe_n(n);
  p = safe_p(p);
  k = safe_k(k);
  const bool large_k = k >= 100;
  const bool large_n = n >= 1000000;
  const bool compact = as<bool>(base["tuning_compact_build"]);
  const bool auto_build_algo = build_algo_preference == "auto";
  const std::string build_algo = auto_build_algo ?
    "iterative_cagra_search" :
    cagra_build_algo_for_shape_core(n, p, k, true, compact, build_algo_preference);
  const int base_graph_degree = as<int>(base["graph_degree"]);
  const int base_intermediate_graph_degree = as<int>(base["intermediate_graph_degree"]);
  const int hnsw_graph_degree = manual_cagra ? base_graph_degree : std::max(2, k);
  const int hnsw_intermediate_graph_degree = manual_cagra ?
    base_intermediate_graph_degree :
    std::max(hnsw_graph_degree, 2 * hnsw_graph_degree);
  // Target about 0.99 recall by default. The CAGRA seed graph remains
  // high-quality; lowering HNSW ef trims search time without changing the
  // public algorithm. Users can raise faissR.cuvs_hnsw_ef for stricter recall.
  const int default_ef = std::max(50, k);
  const int requested_ef = requested_int(ef_option, default_ef);
  const int ef = option_int(ef_option, default_ef, k, 4096);
  const int threads = std::max(1, std::min(64, valid_int(n_threads) ? n_threads : 1));

  return List::create(
    _["graph_degree"] = hnsw_graph_degree,
    _["intermediate_graph_degree"] = hnsw_intermediate_graph_degree,
    _["ef"] = ef,
    _["n_threads"] = threads,
    _["cagra_build_algo"] = build_algo,
    _["requested_graph_degree"] = hnsw_graph_degree,
    _["requested_intermediate_graph_degree"] = hnsw_intermediate_graph_degree,
    _["requested_ef"] = requested_ef,
    _["requested_n_threads"] = threads,
    _["tuning_policy"] = as<std::string>(base["tuning_policy"]),
    _["tuning_rule"] = auto_build_algo ? "recall99_hnsw_from_iterative_cagra" :
      ((large_n && large_k) ? "recall99_large_n_large_k_hnsw_from_cagra" :
        (large_n ? "recall99_large_n_hnsw_from_cagra" :
          (large_k ? "recall99_large_k_hnsw_from_cagra" : "recall99_balanced_hnsw_from_cagra"))),
    _["target_recall"] = 0.99,
    _["tuning_large_n"] = large_n,
    _["tuning_large_k"] = large_k,
    _["tuning_small_k"] = as<bool>(base["tuning_small_k"]),
    _["tuning_source"] = "cpp"
  );
}

// [[Rcpp::export]]
List nn_tune_cuvs_nndescent_cpp(int n,
                                int k,
                                int graph_degree_option = NA_INTEGER,
                                int intermediate_graph_degree_option = NA_INTEGER,
                                int max_iterations_option = NA_INTEGER,
                                bool manual = false) {
  n = safe_n(n);
  k = safe_k(k);
  const bool small_k = k <= 10;
  const bool large_k = k >= 100;
  const bool large_n = n >= 1000000;
  const int n_cap = std::max(1, n - 1);
  const int graph_degree = option_int(graph_degree_option, k, k, n_cap);
  const int intermediate = option_int(
    intermediate_graph_degree_option,
    std::max(graph_degree * 2, graph_degree),
    graph_degree,
    n_cap
  );
  const int max_iterations = option_int(max_iterations_option, 20, 1, 200);
  return List::create(
    _["graph_degree"] = graph_degree,
    _["intermediate_graph_degree"] = intermediate,
    _["max_iterations"] = max_iterations,
    _["tuning_policy"] = manual ? "manual_options" : "auto_shape_k",
    _["tuning_rule"] = (large_n || large_k) ? "large_graph_search" :
      (small_k ? "small_k_speed" : "balanced_graph_search"),
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
                            int r_option = NA_INTEGER,
                            int graph_k_option = NA_INTEGER) {
  n = safe_n(n);
  p = safe_p(p);
  k = safe_k(k);
  const bool large_k = k >= 50;
  const bool high_dim = p >= 128;
  const bool large_n = n >= 50000;
  const bool inner_product = metric == "inner_product";
  const bool ann_seed = backend == "cpu" && large_n && high_dim;
  const int default_r = ann_seed && !large_k ? 32 : ((large_k || high_dim || inner_product) ? 64 : 48);
  const int graph_k_cap = backend == "cuda" ? 255 : 512;
  const int requested_r = option_int(r_option, default_r, 2, 256);
  int r = std::min(std::max(2, requested_r), std::max(1, n - 1));
  const int default_multiplier = (high_dim || inner_product) ? 3 : 2;
  const int default_graph_k = ann_seed ?
    std::max(std::max(k, 2 * r), large_k ? 96 : 64) :
    std::max(std::max(k, default_multiplier * r), 96);
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
    _["tuning_policy"] = "auto_shape_k_metric",
    _["tuning_rule"] = inner_product ? ("inner_product_" + backend + "_nsg_candidate_refine") :
      ((high_dim || large_k) ? ("high_recall_" + backend + "_nsg") : ("balanced_" + backend + "_nsg")),
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
                        int r_option = NA_INTEGER,
                        int search_l_option = NA_INTEGER,
                        double alpha_option = NA_REAL) {
  n = safe_n(n);
  p = safe_p(p);
  k = safe_k(k);
  const bool large_k = k >= 50;
  const bool high_dim = p >= 128;
  const bool large_n = n >= 50000;
  const bool ann_seed = large_n && high_dim;
  const int default_r = ann_seed && !large_k ? 32 : ((high_dim || large_k) ? 64 : 48);
  const int requested_r = option_int(r_option, default_r, 2, 256);
  int r = std::min(std::max(2, requested_r), std::max(1, n - 1));
  const int default_search_l = ann_seed ?
    std::max(std::max(k, 2 * r), large_k ? 96 : 64) :
    std::max(std::max(k, 2 * r), 96);
  const int requested_search_l = option_int(search_l_option, default_search_l, k, 512);
  int search_l = std::min(std::min(std::max(k, requested_search_l), std::max(1, n - 1)), 512);
  if (r > search_l) r = search_l;
  const double requested_alpha = valid_double(alpha_option) && alpha_option >= 1.0 ? alpha_option : 1.2;
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
    _["tuning_policy"] = "auto_shape_k_metric",
    _["tuning_rule"] = metric == "inner_product" ? "inner_product_vamana_candidate_refine" :
      ((high_dim || large_k) ? "high_recall_vamana" : "balanced_vamana"),
    _["tuning_large_k"] = large_k,
    _["tuning_high_dim"] = high_dim,
    _["tuning_source"] = "cpp"
  );
}

// [[Rcpp::export]]
List nn_tune_gpu_nndescent_cpp(int n,
                               int k,
                               std::string backend,
                               int graph_degree_option = NA_INTEGER,
                               int n_iters_option = NA_INTEGER,
                               int sources_option = NA_INTEGER,
                               int neighbors_option = NA_INTEGER,
                               double delta_option = NA_REAL) {
  n = valid_int(n) ? n : NA_INTEGER;
  k = safe_k(k);
  int graph_degree_default = (valid_int(n) && n >= 50000 && k <= 30) ? std::max(k, 64) : k;
  int graph_degree = valid_int(graph_degree_option) ? graph_degree_option : graph_degree_default;
  if (valid_int(n)) graph_degree = std::min(graph_degree, n - 1);
  graph_degree = clamp_int(graph_degree, k, 256);
  const int default_iters = (backend == "cuda" && valid_int(n) && n >= 50000) ? 3 : 1;
  const int n_iters = option_int(n_iters_option, default_iters, 1, 5);
  const int default_sources = std::max(3, std::min(graph_degree, 10));
  const int sources = option_int(sources_option, default_sources, 1, graph_degree);
  const int default_neighbors = std::max(5, std::min(graph_degree, static_cast<int>(std::ceil(graph_degree / 2.0))));
  const int neighbors = option_int(neighbors_option, default_neighbors, 1, graph_degree);
  const double delta = option_double(delta_option, 0.015, 0.0, R_PosInf);
  return List::create(
    _["graph_degree"] = graph_degree,
    _["n_iters"] = n_iters,
    _["sources"] = sources,
    _["neighbors"] = neighbors,
    _["delta"] = delta,
    _["tuning_policy"] = "auto_shape_k",
    _["tuning_rule"] = (valid_int(n) && n >= 50000) ? "large_graph_adaptive" : "balanced_adaptive",
    _["tuning_large_n"] = valid_int(n) && n >= 50000,
    _["tuning_small_k"] = k <= 10,
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
