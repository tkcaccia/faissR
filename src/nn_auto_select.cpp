#include <Rcpp.h>
#include <algorithm>
#include <cmath>
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
  const bool cagra_available = metric != "inner_product" &&
    (faiss_gpu_available || cuvs_available);
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
  if (metric == "inner_product" && cuda_available) return "cuda_native_nndescent";
  if (metric == "inner_product" && cuvs_available) return "cuda_cuvs_hnsw";
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
