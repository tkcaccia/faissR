#include <Rcpp.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

#include <faiss/IndexFlat.h>
#include <faiss/Clustering.h>
#include <faiss/IndexHNSW.h>
#include <faiss/IndexIVFFlat.h>
#include <faiss/IndexIVFPQ.h>
#include <faiss/IndexNNDescent.h>
#include <faiss/IndexNSG.h>

#if __has_include(<faiss/gpu/StandardGpuResources.h>) && \
    __has_include(<faiss/gpu/GpuIndexFlat.h>) && \
    __has_include(<faiss/gpu/GpuIndexIVFFlat.h>) && \
    __has_include(<faiss/gpu/GpuIndexIVFPQ.h>)
#define FASTEMBEDR_HAS_FAISS_GPU 1
#include <faiss/gpu/StandardGpuResources.h>
#include <faiss/gpu/GpuIndexFlat.h>
#include <faiss/gpu/GpuIndexIVFFlat.h>
#include <faiss/gpu/GpuIndexIVFPQ.h>
#endif

#if defined(FASTEMBEDR_HAS_FAISS_GPU) && __has_include(<faiss/gpu/GpuIndexCagra.h>)
#define FASTEMBEDR_HAS_FAISS_GPU_CAGRA 1
#include <faiss/gpu/GpuIndexCagra.h>
#endif

using Rcpp::IntegerMatrix;
using Rcpp::List;
using Rcpp::NumericMatrix;

namespace {

enum class DistanceOutput {
  L2Squared,
  InnerProduct,
  OneMinusInnerProduct
};

void validate_inputs(const NumericMatrix& data,
                     const NumericMatrix& points,
                     const int k,
                     const bool exclude_self) {
  if (data.nrow() < 1 || points.nrow() < 1) {
    Rcpp::stop("data and points must have at least one row");
  }
  if (data.ncol() != points.ncol()) {
    Rcpp::stop("data and points must have the same number of columns");
  }
  if (data.ncol() < 1) {
    Rcpp::stop("data and points must have at least one column");
  }
  if (k < 1 || k > data.nrow()) {
    Rcpp::stop("k must be in [1, nrow(data)]");
  }
  if (exclude_self && data.nrow() != points.nrow()) {
    Rcpp::stop("self-neighbor exclusion requires points to be data");
  }
  if (data.nrow() > std::numeric_limits<int>::max() ||
      points.nrow() > std::numeric_limits<int>::max() ||
      data.ncol() > std::numeric_limits<int>::max()) {
    Rcpp::stop("FAISS backend currently supports dimensions that fit in int");
  }
}

void copy_row_major_float(const NumericMatrix& src, std::vector<float>& dest) {
  const int nrow = src.nrow();
  const int ncol = src.ncol();
  dest.assign(static_cast<std::size_t>(nrow) * ncol, 0.0f);
  bool finite = true;
#ifdef _OPENMP
#pragma omp parallel for schedule(static) reduction(&& : finite)
#endif
  for (int r = 0; r < nrow; ++r) {
    for (int c = 0; c < ncol; ++c) {
      const double value = src(r, c);
      if (!std::isfinite(value)) {
        finite = false;
        continue;
      }
      dest[static_cast<std::size_t>(r) * ncol + c] =
        static_cast<float>(value);
    }
  }
  if (!finite) {
    Rcpp::stop("FAISS backend requires finite numeric input");
  }
}

bool same_matrix_storage(const NumericMatrix& data,
                         const NumericMatrix& points) {
  return data.nrow() == points.nrow() &&
    data.ncol() == points.ncol() &&
    data.begin() == points.begin();
}

class OmpThreadScope {
 public:
  explicit OmpThreadScope(const int n_threads) {
#ifdef _OPENMP
    previous_ = omp_get_max_threads();
    if (n_threads > 0) {
      omp_set_num_threads(std::max(1, n_threads));
    }
#else
    (void)n_threads;
#endif
  }

  ~OmpThreadScope() {
#ifdef _OPENMP
    if (previous_ > 0) {
      omp_set_num_threads(previous_);
    }
#endif
  }

 private:
#ifdef _OPENMP
  int previous_ = 0;
#endif
};

List format_faiss_result(const std::vector<faiss::idx_t>& labels,
                         const std::vector<float>& distances,
                         const int n_points,
                         const int search_k,
                         const int out_k,
                         const bool self_query,
                         const bool exclude_self,
                         const std::string& index_type,
                         const bool exact,
                         const DistanceOutput distance_output = DistanceOutput::L2Squared,
                         const int n_threads = 1,
                         const int nlist = NA_INTEGER,
                         const int nprobe = NA_INTEGER,
                         const int graph_degree = NA_INTEGER,
                         const int search_width = NA_INTEGER) {
  IntegerMatrix indices(n_points, out_k);
  NumericMatrix dists(n_points, out_k);
  int* indices_ptr = indices.begin();
  double* dists_ptr = dists.begin();
  const bool skip_self = exclude_self && self_query;
  const bool inner_product_output = distance_output == DistanceOutput::InnerProduct;
  const bool one_minus_ip_output = distance_output == DistanceOutput::OneMinusInnerProduct;
  bool complete = true;

#ifdef _OPENMP
#pragma omp parallel for num_threads(n_threads > 0 ? n_threads : 1) schedule(static) reduction(&& : complete)
#endif
  for (int i = 0; i < n_points; ++i) {
    const std::size_t row_offset = static_cast<std::size_t>(i) * search_k;
    double row_best_ip = -std::numeric_limits<double>::infinity();
    if (inner_product_output) {
      for (int j = 0; j < search_k; ++j) {
        const std::size_t result_offset = row_offset + j;
        const faiss::idx_t label = labels[result_offset];
        if (label < 0) continue;
        if (skip_self && label == i) continue;
        row_best_ip = std::max(
          row_best_ip,
          static_cast<double>(distances[result_offset])
        );
      }
      if (!std::isfinite(row_best_ip)) row_best_ip = 0.0;
    }
    int written = 0;
    for (int j = 0; j < search_k && written < out_k; ++j) {
      const std::size_t result_offset = row_offset + j;
      const faiss::idx_t label = labels[result_offset];
      if (label < 0) continue;
      if (skip_self && label == i) continue;
      const std::size_t output_offset = static_cast<std::size_t>(written) * n_points + i;
      indices_ptr[output_offset] = static_cast<int>(label) + 1;
      const float sq = distances[result_offset];
      double value;
      if (inner_product_output) {
        value = std::max(row_best_ip - static_cast<double>(sq), 0.0);
      } else if (one_minus_ip_output) {
        value = 1.0 - static_cast<double>(sq);
        if (value < 0.0 && value > -1e-6) value = 0.0;
        if (value > 2.0 && value < 2.0 + 1e-6) value = 2.0;
      } else {
        value = std::sqrt(std::max(static_cast<double>(sq), 0.0));
      }
      dists_ptr[output_offset] = value;
      ++written;
    }
    if (written < out_k) {
      complete = false;
    }
  }
  if (!complete) {
    Rcpp::stop("FAISS returned fewer neighbors than requested");
  }

  List out = List::create(
    Rcpp::Named("indices") = indices,
    Rcpp::Named("distances") = dists,
    Rcpp::Named("index_type") = index_type,
    Rcpp::Named("exact") = exact
  );
  if (nlist != NA_INTEGER) out["nlist"] = nlist;
  if (nprobe != NA_INTEGER) out["nprobe"] = nprobe;
  if (graph_degree != NA_INTEGER) out["graph_degree"] = graph_degree;
  if (search_width != NA_INTEGER) out["search_width"] = search_width;
  out["metric"] = distance_output == DistanceOutput::InnerProduct ?
    "inner_product_similarity_shifted_to_distance" :
    (distance_output == DistanceOutput::OneMinusInnerProduct ? "one_minus_inner_product" : "euclidean");
  return out;
}

List search_faiss_index(faiss::Index& index,
                        NumericMatrix data,
                        NumericMatrix points,
                        int k,
                        bool exclude_self,
                        int n_threads,
                        const std::string& index_type,
                        bool exact,
                        DistanceOutput distance_output,
                        int nlist = NA_INTEGER,
                        int nprobe = NA_INTEGER,
                        int graph_degree = NA_INTEGER,
                        int search_width = NA_INTEGER,
                        bool use_ivf_search_params = false) {
  validate_inputs(data, points, k, exclude_self);
  const bool same_storage = same_matrix_storage(data, points);
  const bool self_query = exclude_self || same_storage;
  const int n_data = data.nrow();
  const int n_points = points.nrow();
  const int search_k = exclude_self ? std::min(n_data, k + 1) : k;

  OmpThreadScope threads(n_threads);

  std::vector<float> xb;
  std::vector<float> xq;
  copy_row_major_float(data, xb);
  if (same_storage) {
    xq.clear();
  } else {
    copy_row_major_float(points, xq);
  }
  const float* query_ptr = same_storage ? xb.data() : xq.data();

  std::vector<float> distances(static_cast<std::size_t>(n_points) * search_k);
  std::vector<faiss::idx_t> labels(static_cast<std::size_t>(n_points) * search_k);

  try {
    if (!index.is_trained) {
      index.train(n_data, xb.data());
    }
    index.add(n_data, xb.data());
    if (use_ivf_search_params && nprobe != NA_INTEGER) {
      faiss::SearchParametersIVF params;
      params.nprobe = static_cast<std::size_t>(std::max(1, nprobe));
      index.search(
        n_points, query_ptr, search_k, distances.data(), labels.data(), &params
      );
    } else {
      index.search(n_points, query_ptr, search_k, distances.data(), labels.data());
    }
  } catch (const std::exception& e) {
    Rcpp::stop("FAISS %s search failed: %s", index_type.c_str(), e.what());
  }

  return format_faiss_result(
    labels, distances, n_points, search_k, k, self_query, exclude_self,
    index_type, exact, distance_output, n_threads, nlist, nprobe, graph_degree, search_width
  );
}

int clamp_positive(const int value, const int fallback, const int upper) {
  int out = value > 0 ? value : fallback;
  if (upper > 0) out = std::min(out, upper);
  return std::max(1, out);
}

bool faiss_gpu_supported_pq_code_size(const int code_size) {
  switch (code_size) {
    case 1:
    case 2:
    case 3:
    case 4:
    case 8:
    case 12:
    case 16:
    case 20:
    case 24:
    case 28:
    case 32:
    case 48:
    case 56:
    case 64:
    case 96:
      return true;
    default:
      return false;
  }
}

DistanceOutput parse_distance_output(const std::string& distance_output,
                                     const char* index_type) {
  if (distance_output == "inner_product") {
    return DistanceOutput::InnerProduct;
  }
  if (distance_output == "one_minus_inner_product") {
    return DistanceOutput::OneMinusInnerProduct;
  }
  if (distance_output == "euclidean") {
    return DistanceOutput::L2Squared;
  }
  Rcpp::stop("Unsupported FAISS %s distance output mode", index_type);
}

} // namespace

bool faiss_is_available_impl() {
  return true;
}

std::string faiss_info_json_impl() {
#ifdef FASTEMBEDR_HAS_FAISS_GPU
  const char* gpu = "true";
#else
  const char* gpu = "false";
#endif
#ifdef FASTEMBEDR_HAS_FAISS_GPU_CAGRA
  const char* gpu_cagra = "true";
#else
  const char* gpu_cagra = "false";
#endif
  return std::string("{\"available\":true,\"library\":\"faiss\",\"interface\":\"c++\",") +
    "\"gpu\":" + gpu + ",\"gpu_cagra\":" + gpu_cagra + "}";
}

List faiss_flat_knn_impl(NumericMatrix data,
                         NumericMatrix points,
                         int k,
                         bool exclude_self,
                         int n_threads) {
  const int n_features = data.ncol();
  faiss::IndexFlatL2 index(n_features);
  return search_faiss_index(
    index, data, points, k, exclude_self, n_threads,
    "IndexFlatL2", true, DistanceOutput::L2Squared
  );
}

List faiss_flat_ip_knn_impl(NumericMatrix data,
                            NumericMatrix points,
                            int k,
                            bool exclude_self,
                            int n_threads) {
  const int n_features = data.ncol();
  faiss::IndexFlatIP index(n_features);
  return search_faiss_index(
    index, data, points, k, exclude_self, n_threads,
    "IndexFlatIP", true, DistanceOutput::InnerProduct
  );
}

List faiss_flat_normalized_ip_distance_knn_impl(NumericMatrix data,
                                                NumericMatrix points,
                                                int k,
                                                bool exclude_self,
                                                int n_threads) {
  const int n_features = data.ncol();
  faiss::IndexFlatIP index(n_features);
  return search_faiss_index(
    index, data, points, k, exclude_self, n_threads,
    "IndexFlatIP", true, DistanceOutput::OneMinusInnerProduct
  );
}

List faiss_gpu_flat_knn_impl(NumericMatrix data,
                             NumericMatrix points,
                             int k,
                             bool exclude_self) {
#ifdef FASTEMBEDR_HAS_FAISS_GPU
  const int n_features = data.ncol();
  faiss::gpu::StandardGpuResources resources;
  faiss::gpu::GpuIndexFlatConfig config;
  config.device = 0;
  faiss::gpu::GpuIndexFlatL2 index(&resources, n_features, config);
  return search_faiss_index(
    index, data, points, k, exclude_self, 1,
    "GpuIndexFlatL2", true, DistanceOutput::L2Squared
  );
#else
  (void)data;
  (void)points;
  (void)k;
  (void)exclude_self;
  Rcpp::stop(
    "FAISS GPU Flat L2 backend is not available in this build. "
    "Install FAISS GPU/cuVS headers and rebuild faissR with FAISS_HOME set."
  );
#endif
}

List faiss_gpu_flat_ip_knn_impl(NumericMatrix data,
                                NumericMatrix points,
                                int k,
                                bool exclude_self) {
#ifdef FASTEMBEDR_HAS_FAISS_GPU
  const int n_features = data.ncol();
  faiss::gpu::StandardGpuResources resources;
  faiss::gpu::GpuIndexFlatConfig config;
  config.device = 0;
  faiss::gpu::GpuIndexFlatIP index(&resources, n_features, config);
  return search_faiss_index(
    index, data, points, k, exclude_self, 1,
    "GpuIndexFlatIP", true, DistanceOutput::InnerProduct
  );
#else
  (void)data;
  (void)points;
  (void)k;
  (void)exclude_self;
  Rcpp::stop(
    "FAISS GPU Flat IP backend is not available in this build. "
    "Install FAISS GPU/cuVS headers and rebuild faissR with FAISS_HOME set."
  );
#endif
}

List faiss_gpu_flat_normalized_ip_distance_knn_impl(NumericMatrix data,
                                                    NumericMatrix points,
                                                    int k,
                                                    bool exclude_self) {
#ifdef FASTEMBEDR_HAS_FAISS_GPU
  const int n_features = data.ncol();
  faiss::gpu::StandardGpuResources resources;
  faiss::gpu::GpuIndexFlatConfig config;
  config.device = 0;
  faiss::gpu::GpuIndexFlatIP index(&resources, n_features, config);
  return search_faiss_index(
    index, data, points, k, exclude_self, 1,
    "GpuIndexFlatIP", true, DistanceOutput::OneMinusInnerProduct
  );
#else
  (void)data;
  (void)points;
  (void)k;
  (void)exclude_self;
  Rcpp::stop(
    "FAISS GPU Flat IP backend is not available in this build. "
    "Install FAISS GPU/cuVS headers and rebuild faissR with FAISS_HOME set."
  );
#endif
}

List faiss_ivf_knn_impl(NumericMatrix data,
                        NumericMatrix points,
                        int k,
                        int nlist,
                        int nprobe,
                        std::string metric,
                        std::string distance_output,
                        bool exclude_self,
                        int n_threads) {
  const int n_data = data.nrow();
  const int n_features = data.ncol();
  nlist = std::max(1, std::min(nlist, n_data));
  nprobe = std::max(1, std::min(nprobe, nlist));
  faiss::MetricType faiss_metric = faiss::METRIC_L2;
  std::unique_ptr<faiss::Index> quantizer;
  if (metric == "inner_product") {
    faiss_metric = faiss::METRIC_INNER_PRODUCT;
    quantizer.reset(new faiss::IndexFlatIP(n_features));
  } else if (metric == "euclidean") {
    quantizer.reset(new faiss::IndexFlatL2(n_features));
  } else {
    Rcpp::stop("FAISS IVF supports metric = 'euclidean' or 'inner_product'");
  }
  const DistanceOutput output = parse_distance_output(distance_output, "IVF");
  faiss::IndexIVFFlat index(quantizer.get(), n_features, nlist, faiss_metric);
  index.nprobe = nprobe;
  return search_faiss_index(
    index, data, points, k, exclude_self, n_threads,
    metric == "inner_product" ? "IndexIVFFlatIP" : "IndexIVFFlat",
    false,
    output,
    nlist,
    nprobe
  );
}

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
                          int n_threads) {
  const int n_data = data.nrow();
  const int n_features = data.ncol();
  const int requested_pq_m = pq_m;
  const int requested_pq_nbits = pq_nbits;
  nlist = std::max(1, std::min(nlist, n_data));
  nprobe = std::max(1, std::min(nprobe, nlist));
  pq_m = clamp_positive(pq_m, 8, n_features);
  while (pq_m > 1 && (n_features % pq_m) != 0) --pq_m;
  pq_nbits = std::max(4, std::min(pq_nbits, 12));
  while (pq_nbits > 4 && (1 << pq_nbits) > n_data) {
    --pq_nbits;
  }
  faiss::MetricType faiss_metric = faiss::METRIC_L2;
  std::unique_ptr<faiss::Index> quantizer;
  if (metric == "inner_product") {
    faiss_metric = faiss::METRIC_INNER_PRODUCT;
    quantizer.reset(new faiss::IndexFlatIP(n_features));
  } else if (metric == "euclidean") {
    quantizer.reset(new faiss::IndexFlatL2(n_features));
  } else {
    Rcpp::stop("FAISS IVFPQ supports metric = 'euclidean' or 'inner_product'");
  }
  const DistanceOutput output = parse_distance_output(distance_output, "IVFPQ");
  faiss::IndexIVFPQ index(
    quantizer.get(), n_features, nlist, pq_m, pq_nbits, faiss_metric
  );
  index.nprobe = nprobe;
  List out = search_faiss_index(
    index, data, points, k, exclude_self, n_threads,
    metric == "inner_product" ? "IndexIVFPQIP" : "IndexIVFPQ",
    false,
    output,
    nlist,
    nprobe
  );
  out["pq_m"] = pq_m;
  out["pq_nbits"] = pq_nbits;
  out["requested_pq_m"] = requested_pq_m;
  out["requested_pq_nbits"] = requested_pq_nbits;
  out["pq_parameters_adjusted"] = requested_pq_m != pq_m || requested_pq_nbits != pq_nbits;
  return out;
}

List faiss_hnsw_knn_impl(NumericMatrix data,
                         NumericMatrix points,
                         int k,
                         int m,
                         int ef_construction,
                         int ef_search,
                         std::string metric,
                         std::string distance_output,
                         bool exclude_self,
                         int n_threads) {
  const int n_features = data.ncol();
  const int requested_m = m;
  const int requested_ef_construction = ef_construction;
  const int requested_ef_search = ef_search;
  m = clamp_positive(m, 32, data.nrow());
  ef_construction = std::max(ef_construction, m);
  ef_search = std::max(ef_search, k);
  faiss::MetricType faiss_metric = faiss::METRIC_L2;
  if (metric == "inner_product") {
    faiss_metric = faiss::METRIC_INNER_PRODUCT;
  } else if (metric != "euclidean") {
    Rcpp::stop("FAISS HNSW supports metric = 'euclidean' or 'inner_product'");
  }
  const DistanceOutput output = parse_distance_output(distance_output, "HNSW");
  faiss::IndexHNSWFlat index(n_features, m, faiss_metric);
  index.hnsw.efConstruction = ef_construction;
  index.hnsw.efSearch = ef_search;
  List out = search_faiss_index(
    index, data, points, k, exclude_self, n_threads,
    "IndexHNSWFlat", false, output,
    NA_INTEGER, NA_INTEGER, m, ef_search
  );
  out["m"] = m;
  out["ef_construction"] = ef_construction;
  out["ef_search"] = ef_search;
  out["requested_m"] = requested_m;
  out["requested_ef_construction"] = requested_ef_construction;
  out["requested_ef_search"] = requested_ef_search;
  out["hnsw_parameters_adjusted"] = requested_m != m ||
    requested_ef_construction != ef_construction || requested_ef_search != ef_search;
  return out;
}

List faiss_nsg_knn_impl(NumericMatrix data,
                        NumericMatrix points,
                        int k,
                        int r,
                        int search_l,
                        int build_type,
                        std::string metric,
                        std::string distance_output,
                        bool exclude_self,
                        int n_threads) {
  const int n_data = data.nrow();
  const int n_features = data.ncol();
  if (n_data <= 100) {
    Rcpp::stop("FAISS NSG requires more than 100 training rows in this FAISS build.");
  }
  const int requested_r = r;
  const int requested_search_l = search_l;
  const int requested_build_type = build_type;
  r = clamp_positive(r, 32, n_data);
  search_l = std::max(search_l, k);
  build_type = build_type == 1 ? 1 : 0;
  if (metric != "euclidean") {
    Rcpp::stop("FAISS NSG is currently validated only for metric = 'euclidean'");
  }
  const DistanceOutput output = parse_distance_output(distance_output, "NSG");
  faiss::IndexNSGFlat index(n_features, r, faiss::METRIC_L2);
  index.nsg.search_L = search_l;
  index.build_type = static_cast<char>(build_type);
  const int gk = std::max(64, std::max(2 * k, 2 * r));
  index.GK = gk;
  List out = search_faiss_index(
    index, data, points, k, exclude_self, n_threads,
    "IndexNSGFlat",
    false,
    output,
    NA_INTEGER, NA_INTEGER, r, search_l
  );
  out["r"] = r;
  out["search_l"] = search_l;
  out["build_type"] = build_type;
  out["gk"] = gk;
  out["requested_r"] = requested_r;
  out["requested_search_l"] = requested_search_l;
  out["requested_build_type"] = requested_build_type;
  out["nsg_parameters_adjusted"] = requested_r != r ||
    requested_search_l != search_l || requested_build_type != build_type;
  return out;
}

List faiss_nndescent_knn_impl(NumericMatrix data,
                              NumericMatrix points,
                              int k,
                              int graph_k,
                              int n_iter,
                              int search_l,
                              std::string metric,
                              std::string distance_output,
                              bool exclude_self,
                              int n_threads) {
  const int n_data = data.nrow();
  const int n_features = data.ncol();
  if (n_data <= 100) {
    Rcpp::stop("FAISS NN-Descent requires more than 100 training rows in this FAISS build.");
  }
  const int requested_graph_k = graph_k;
  const int requested_n_iter = n_iter;
  const int requested_search_l = search_l;
  graph_k = std::max(graph_k, k);
  n_iter = std::max(1, n_iter);
  search_l = std::max(search_l, k);
  if (metric != "euclidean") {
    Rcpp::stop("FAISS NNDescent is currently validated only for metric = 'euclidean'");
  }
  const DistanceOutput output = parse_distance_output(distance_output, "NNDescent");
  faiss::IndexNNDescentFlat index(n_features, graph_k, faiss::METRIC_L2);
  index.nndescent.iter = n_iter;
  index.nndescent.search_L = search_l;
  List out = search_faiss_index(
    index, data, points, k, exclude_self, n_threads,
    "IndexNNDescentFlat",
    false,
    output,
    NA_INTEGER, NA_INTEGER, graph_k, search_l
  );
  out["graph_k"] = graph_k;
  out["n_iter"] = n_iter;
  out["search_l"] = search_l;
  out["requested_graph_k"] = requested_graph_k;
  out["requested_n_iter"] = requested_n_iter;
  out["requested_search_l"] = requested_search_l;
  out["nndescent_parameters_adjusted"] = requested_graph_k != graph_k ||
    requested_n_iter != n_iter || requested_search_l != search_l;
  return out;
}

List faiss_gpu_ivf_flat_knn_impl(NumericMatrix data,
                                 NumericMatrix points,
                                 int k,
                                 int nlist,
                                 int nprobe,
                                 std::string metric,
                                 std::string distance_output,
                                 bool exclude_self) {
#ifdef FASTEMBEDR_HAS_FAISS_GPU
  const int n_data = data.nrow();
  const int n_features = data.ncol();
  nlist = std::max(1, std::min(nlist, n_data));
  nprobe = std::max(1, std::min(nprobe, nlist));
  faiss::MetricType faiss_metric = faiss::METRIC_L2;
  if (metric == "inner_product") {
    faiss_metric = faiss::METRIC_INNER_PRODUCT;
  } else if (metric != "euclidean") {
    Rcpp::stop("FAISS GPU IVF Flat supports metric = 'euclidean' or 'inner_product'");
  }
  const DistanceOutput output = parse_distance_output(distance_output, "GPU IVF Flat");
  faiss::gpu::StandardGpuResources resources;
  faiss::gpu::GpuIndexIVFFlatConfig config;
  config.device = 0;
  faiss::gpu::GpuIndexIVFFlat index(
    &resources,
    n_features,
    nlist,
    faiss_metric,
    config
  );
  return search_faiss_index(
    index, data, points, k, exclude_self, 1,
    metric == "inner_product" ? "GpuIndexIVFFlatIP_cuVS" : "GpuIndexIVFFlat_cuVS",
    false,
    output,
    nlist, nprobe, NA_INTEGER, NA_INTEGER, true
  );
#else
  (void)data;
  (void)points;
  (void)k;
  (void)nlist;
  (void)nprobe;
  (void)metric;
  (void)distance_output;
  (void)exclude_self;
  Rcpp::stop(
    "FAISS GPU IVF Flat backend is not available in this build. "
    "Install FAISS GPU/cuVS headers and rebuild faissR with FAISS_HOME set."
  );
#endif
}

List faiss_gpu_ivfpq_knn_impl(NumericMatrix data,
                              NumericMatrix points,
                              int k,
                              int nlist,
                              int nprobe,
                              int pq_m,
                              int pq_nbits,
                              std::string metric,
                              std::string distance_output,
                              bool exclude_self) {
#ifdef FASTEMBEDR_HAS_FAISS_GPU
  const int n_data = data.nrow();
  const int n_features = data.ncol();
  const int requested_pq_m = pq_m;
  const int requested_pq_nbits = pq_nbits;
  nlist = std::max(1, std::min(nlist, n_data));
  nprobe = std::max(1, std::min(nprobe, nlist));
  faiss::MetricType faiss_metric = faiss::METRIC_L2;
  if (metric == "inner_product") {
    faiss_metric = faiss::METRIC_INNER_PRODUCT;
  } else if (metric != "euclidean") {
    Rcpp::stop("FAISS GPU IVFPQ supports metric = 'euclidean' or 'inner_product'");
  }
  const DistanceOutput output = parse_distance_output(distance_output, "GPU IVFPQ");
  pq_m = clamp_positive(pq_m, 8, n_features);
  while (pq_m > 1 && (n_features % pq_m) != 0) --pq_m;
  if (n_data < 256) {
    Rcpp::stop(
      "FAISS GPU IVFPQ requires at least 256 training rows because "
      "GpuIndexIVFPQ supports 8-bit PQ codes only."
    );
  }
  pq_nbits = 8;
  const int max_full_precision_lut_entries = 49152 / (static_cast<int>(sizeof(float)) * (1 << pq_nbits));
  while (pq_m > 1 && pq_m > max_full_precision_lut_entries) --pq_m;
  while (pq_m > 1 && ((n_features % pq_m) != 0 || !faiss_gpu_supported_pq_code_size(pq_m))) {
    --pq_m;
  }
  faiss::gpu::StandardGpuResources resources;
  faiss::gpu::GpuIndexIVFPQConfig config;
  config.device = 0;
  // Full-precision lookup tables are safer for raw, unscaled benchmark data.
  config.useFloat16LookupTables = false;
  faiss::gpu::GpuIndexIVFPQ index(
    &resources,
    n_features,
    nlist,
    pq_m,
    pq_nbits,
    faiss_metric,
    config
  );
  List out = search_faiss_index(
    index, data, points, k, exclude_self, 1,
    metric == "inner_product" ? "GpuIndexIVFPQIP_cuVS" : "GpuIndexIVFPQ_cuVS",
    false,
    output,
    nlist, nprobe, NA_INTEGER, NA_INTEGER, true
  );
  out["pq_m"] = pq_m;
  out["pq_nbits"] = pq_nbits;
  out["requested_pq_m"] = requested_pq_m;
  out["requested_pq_nbits"] = requested_pq_nbits;
  out["pq_parameters_adjusted"] = requested_pq_m != pq_m || requested_pq_nbits != pq_nbits;
  return out;
#else
  (void)data;
  (void)points;
  (void)k;
  (void)nlist;
  (void)nprobe;
  (void)pq_m;
  (void)pq_nbits;
  (void)metric;
  (void)distance_output;
  (void)exclude_self;
  Rcpp::stop(
    "FAISS GPU IVF-PQ backend is not available in this build. "
    "Install FAISS GPU/cuVS headers and rebuild faissR with FAISS_HOME set."
  );
#endif
}

List faiss_gpu_cagra_knn_impl(NumericMatrix data,
                              NumericMatrix points,
                              int k,
                              int graph_degree,
                              int intermediate_graph_degree,
                              int search_width,
                              int itopk_size,
                              bool exclude_self) {
#ifdef FASTEMBEDR_HAS_FAISS_GPU_CAGRA
  validate_inputs(data, points, k, exclude_self);
  const bool same_storage = same_matrix_storage(data, points);
  const bool self_query = exclude_self || same_storage;
  const int n_data = data.nrow();
  const int n_points = points.nrow();
  const int n_features = data.ncol();
  const int search_k = exclude_self ? std::min(n_data, k + 1) : k;
  const int requested_graph_degree = graph_degree;
  const int requested_intermediate_graph_degree = intermediate_graph_degree;
  const int requested_search_width = search_width;
  const int requested_itopk_size = itopk_size;

  graph_degree = clamp_positive(graph_degree, std::max(64, k + 1), n_data - 1);
  intermediate_graph_degree = clamp_positive(
    intermediate_graph_degree,
    std::max(128, graph_degree * 2),
    n_data - 1
  );
  intermediate_graph_degree = std::max(intermediate_graph_degree, graph_degree);
  itopk_size = clamp_positive(itopk_size, std::max(64, graph_degree), 4096);
  itopk_size = std::max(itopk_size, search_k);
  search_width = std::max(1, search_width);

  std::vector<float> xb;
  std::vector<float> xq;
  copy_row_major_float(data, xb);
  if (!same_storage) {
    copy_row_major_float(points, xq);
  }
  const float* query_ptr = same_storage ? xb.data() : xq.data();

  faiss::gpu::StandardGpuResources resources;
  faiss::gpu::GpuIndexCagraConfig config;
  config.device = 0;
  config.graph_degree = static_cast<std::size_t>(graph_degree);
  config.intermediate_graph_degree = static_cast<std::size_t>(intermediate_graph_degree);
  config.store_dataset = true;
  faiss::gpu::GpuIndexCagra index(
    &resources,
    n_features,
    faiss::METRIC_L2,
    config
  );

  std::vector<float> distances(static_cast<std::size_t>(n_points) * search_k);
  std::vector<faiss::idx_t> labels(static_cast<std::size_t>(n_points) * search_k);
  try {
    index.train(n_data, xb.data());
    faiss::gpu::SearchParametersCagra params;
    params.itopk_size = static_cast<std::size_t>(itopk_size);
    params.search_width = static_cast<std::size_t>(search_width);
    index.search(
      n_points,
      query_ptr,
      search_k,
      distances.data(),
      labels.data(),
      &params
    );
  } catch (const std::exception& e) {
    Rcpp::stop("FAISS GpuIndexCagra search failed: %s", e.what());
  }

  List out = format_faiss_result(
    labels, distances, n_points, search_k, k, self_query, exclude_self,
    "GpuIndexCagra_cuVS", false, DistanceOutput::L2Squared,
    1, NA_INTEGER, NA_INTEGER, graph_degree, search_width
  );
  out["intermediate_graph_degree"] = intermediate_graph_degree;
  out["itopk_size"] = itopk_size;
  out["requested_graph_degree"] = requested_graph_degree;
  out["requested_intermediate_graph_degree"] = requested_intermediate_graph_degree;
  out["requested_search_width"] = requested_search_width;
  out["requested_itopk_size"] = requested_itopk_size;
  out["cagra_parameters_adjusted"] = requested_graph_degree != graph_degree ||
    requested_intermediate_graph_degree != intermediate_graph_degree ||
    requested_search_width != search_width || requested_itopk_size != itopk_size;
  return out;
#else
  (void)data;
  (void)points;
  (void)k;
  (void)graph_degree;
  (void)intermediate_graph_degree;
  (void)search_width;
  (void)itopk_size;
  (void)exclude_self;
  Rcpp::stop(
    "FAISS GPU CAGRA backend is not available in this build. "
    "Install FAISS GPU/cuVS headers with faiss/gpu/GpuIndexCagra.h and rebuild faissR."
  );
#endif
}

List faiss_kmeans_impl(NumericMatrix data,
                       int centers,
                       int max_iter,
                       int nredo,
                       double tol,
                       int seed,
                       int n_threads,
                       bool kmeans_plus_plus) {
  if (data.nrow() < 1 || data.ncol() < 1) {
    Rcpp::stop("data must have at least one row and one column");
  }
  const int n = data.nrow();
  const int p = data.ncol();
  if (centers < 1 || centers > n) {
    Rcpp::stop("centers must be in [1, nrow(data)]");
  }
  max_iter = std::max(1, max_iter);
  nredo = std::max(1, nredo);
  if (!std::isfinite(tol) || tol < 0.0) tol = 1e-4;

  std::vector<float> xb;
  copy_row_major_float(data, xb);

  faiss::ClusteringParameters cp;
  cp.niter = max_iter;
  cp.nredo = nredo;
  cp.verbose = false;
  cp.spherical = false;
  cp.seed = seed;
  cp.min_points_per_centroid = 1;
  cp.max_points_per_centroid = std::max(
    256,
    static_cast<int>((static_cast<long long>(n) + centers - 1) / centers)
  );
  cp.early_stop_threshold = tol;
  if (kmeans_plus_plus) {
    cp.init_method = faiss::ClusteringInitMethod::KMEANS_PLUS_PLUS;
  }

  faiss::IndexFlatL2 index(p);
  try {
    OmpThreadScope threads(n_threads);
    faiss::Clustering clustering(p, centers, cp);
    clustering.train(n, xb.data(), index);

    faiss::IndexFlatL2 assign_index(p);
    assign_index.add(centers, clustering.centroids.data());
    std::vector<float> distances(static_cast<std::size_t>(n));
    std::vector<faiss::idx_t> labels(static_cast<std::size_t>(n));
    assign_index.search(n, xb.data(), 1, distances.data(), labels.data());

    NumericMatrix center_matrix(centers, p);
    for (int c = 0; c < p; ++c) {
      for (int r = 0; r < centers; ++r) {
        center_matrix(r, c) =
          clustering.centroids[static_cast<std::size_t>(r) * p + c];
      }
    }

    Rcpp::IntegerVector cluster(n);
    Rcpp::IntegerVector size(centers);
    Rcpp::NumericVector withinss(centers);
    double total = 0.0;
    for (int i = 0; i < n; ++i) {
      const int label = static_cast<int>(labels[static_cast<std::size_t>(i)]);
      if (label < 0 || label >= centers) {
        Rcpp::stop("FAISS k-means returned an invalid cluster label");
      }
      cluster[i] = label + 1;
      size[label] += 1;
      const double d = std::max(
        0.0,
        static_cast<double>(distances[static_cast<std::size_t>(i)])
      );
      withinss[label] += d;
      total += d;
    }

    const int actual_iter = static_cast<int>(clustering.iteration_stats.size());
    return List::create(
      Rcpp::Named("cluster") = cluster,
      Rcpp::Named("centers") = center_matrix,
      Rcpp::Named("withinss") = withinss,
      Rcpp::Named("tot.withinss") = total,
      Rcpp::Named("size") = size,
      Rcpp::Named("iter") = actual_iter > 0 ? actual_iter : max_iter,
      Rcpp::Named("backend_library") = "faiss",
      Rcpp::Named("parameters") = List::create(
        Rcpp::Named("centers") = centers,
        Rcpp::Named("max_iter") = max_iter,
        Rcpp::Named("n_init") = nredo,
        Rcpp::Named("tol") = tol,
        Rcpp::Named("seed") = seed,
        Rcpp::Named("n_threads") = n_threads,
        Rcpp::Named("max_points_per_centroid") = cp.max_points_per_centroid
      )
    );
  } catch (const std::exception& e) {
    Rcpp::stop("FAISS k-means failed: %s", e.what());
  }
}

List faiss_gpu_kmeans_impl(NumericMatrix data,
                           int centers,
                           int max_iter,
                           int nredo,
                           double tol,
                           int seed,
                           bool kmeans_plus_plus) {
#ifdef FASTEMBEDR_HAS_FAISS_GPU
  if (data.nrow() < 1 || data.ncol() < 1) {
    Rcpp::stop("data must have at least one row and one column");
  }
  const int n = data.nrow();
  const int p = data.ncol();
  if (centers < 1 || centers > n) {
    Rcpp::stop("centers must be in [1, nrow(data)]");
  }
  max_iter = std::max(1, max_iter);
  nredo = std::max(1, nredo);
  if (!std::isfinite(tol) || tol < 0.0) tol = 1e-4;

  std::vector<float> xb;
  copy_row_major_float(data, xb);

  faiss::ClusteringParameters cp;
  cp.niter = max_iter;
  cp.nredo = nredo;
  cp.verbose = false;
  cp.spherical = false;
  cp.seed = seed;
  cp.min_points_per_centroid = 1;
  cp.max_points_per_centroid = std::max(
    256,
    static_cast<int>((static_cast<long long>(n) + centers - 1) / centers)
  );
  cp.early_stop_threshold = tol;
  if (kmeans_plus_plus) {
    cp.init_method = faiss::ClusteringInitMethod::KMEANS_PLUS_PLUS;
  }

  try {
    faiss::gpu::StandardGpuResources resources;
    faiss::gpu::GpuIndexFlatConfig config;
    config.device = 0;

    faiss::gpu::GpuIndexFlatL2 train_index(&resources, p, config);
    faiss::Clustering clustering(p, centers, cp);
    clustering.train(n, xb.data(), train_index);

    faiss::gpu::GpuIndexFlatL2 assign_index(&resources, p, config);
    assign_index.add(centers, clustering.centroids.data());
    std::vector<float> distances(static_cast<std::size_t>(n));
    std::vector<faiss::idx_t> labels(static_cast<std::size_t>(n));
    assign_index.search(n, xb.data(), 1, distances.data(), labels.data());

    NumericMatrix center_matrix(centers, p);
    for (int c = 0; c < p; ++c) {
      for (int r = 0; r < centers; ++r) {
        center_matrix(r, c) =
          clustering.centroids[static_cast<std::size_t>(r) * p + c];
      }
    }

    Rcpp::IntegerVector cluster(n);
    Rcpp::IntegerVector size(centers);
    Rcpp::NumericVector withinss(centers);
    double total = 0.0;
    for (int i = 0; i < n; ++i) {
      const int label = static_cast<int>(labels[static_cast<std::size_t>(i)]);
      if (label < 0 || label >= centers) {
        Rcpp::stop("FAISS GPU k-means returned an invalid cluster label");
      }
      cluster[i] = label + 1;
      size[label] += 1;
      const double d = std::max(
        0.0,
        static_cast<double>(distances[static_cast<std::size_t>(i)])
      );
      withinss[label] += d;
      total += d;
    }

    const int actual_iter = static_cast<int>(clustering.iteration_stats.size());
    return List::create(
      Rcpp::Named("cluster") = cluster,
      Rcpp::Named("centers") = center_matrix,
      Rcpp::Named("withinss") = withinss,
      Rcpp::Named("tot.withinss") = total,
      Rcpp::Named("size") = size,
      Rcpp::Named("iter") = actual_iter > 0 ? actual_iter : max_iter,
      Rcpp::Named("backend_library") = "faiss_gpu",
      Rcpp::Named("parameters") = List::create(
        Rcpp::Named("centers") = centers,
        Rcpp::Named("max_iter") = max_iter,
        Rcpp::Named("n_init") = nredo,
        Rcpp::Named("tol") = tol,
        Rcpp::Named("seed") = seed,
        Rcpp::Named("device") = 0,
        Rcpp::Named("max_points_per_centroid") = cp.max_points_per_centroid
      )
    );
  } catch (const std::exception& e) {
    Rcpp::stop("FAISS GPU k-means failed: %s", e.what());
  }
#else
  (void)data;
  (void)centers;
  (void)max_iter;
  (void)nredo;
  (void)tol;
  (void)seed;
  (void)kmeans_plus_plus;
  Rcpp::stop(
    "FAISS GPU k-means is not available in this build. "
    "Install FAISS GPU/cuVS headers and rebuild faissR with FAISS_HOME set."
  );
#endif
}
