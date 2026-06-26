#include <Rcpp.h>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
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
#include <faiss/IndexIVF.h>
#include <faiss/IndexIVFFlat.h>
#include <faiss/IndexIVFPQ.h>
#include <faiss/IndexNNDescent.h>
#include <faiss/IndexNSG.h>

#if __has_include(<faiss/IndexIVFPQFastScan.h>) && __has_include(<faiss/IndexRefine.h>)
#define FAISSR_HAS_FAISS_FASTSCAN 1
#include <faiss/IndexIVFPQFastScan.h>
#include <faiss/IndexRefine.h>
#endif

#if __has_include(<faiss/gpu/StandardGpuResources.h>) && \
    __has_include(<faiss/gpu/GpuIndexFlat.h>) && \
    __has_include(<faiss/gpu/GpuIndexIVFFlat.h>) && \
    __has_include(<faiss/gpu/GpuIndexIVFPQ.h>)
#define FAISSR_HAS_FAISS_GPU 1
#include <faiss/gpu/StandardGpuResources.h>
#include <faiss/gpu/GpuIndexFlat.h>
#include <faiss/gpu/GpuIndexIVFFlat.h>
#include <faiss/gpu/GpuIndexIVFPQ.h>
#endif

#if defined(FAISSR_HAS_FAISS_GPU) && __has_include(<faiss/gpu/GpuIndexCagra.h>)
#define FAISSR_HAS_FAISS_GPU_CAGRA 1
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

int env_positive_int(const char* name, const int fallback) {
  const char* value = std::getenv(name);
  if (value == nullptr || *value == '\0') return fallback;
  char* end = nullptr;
  const long parsed = std::strtol(value, &end, 10);
  if (end == value || parsed < 1L ||
      parsed > static_cast<long>(std::numeric_limits<int>::max())) {
    return fallback;
  }
  return static_cast<int>(parsed);
}

bool env_false(const char* name) {
  const char* value = std::getenv(name);
  if (value == nullptr) return false;
  const std::string text(value);
  return text == "0" || text == "false" || text == "FALSE" ||
    text == "no" || text == "NO" || text == "off" || text == "OFF";
}

int faiss_query_batch_size(const int n_points,
                           const int search_k,
                           const bool gpu_index) {
  if (n_points < 1) return 1;
  const int env_batch = env_positive_int(
    gpu_index ? "FAISSR_FAISS_GPU_QUERY_BATCH_SIZE" : "FAISSR_FAISS_QUERY_BATCH_SIZE",
    0
  );
  int batch = env_batch > 0 ? env_batch : (gpu_index ? 8192 : 16384);
  if (search_k >= 512 && env_batch <= 0) {
    batch = std::max(1024, batch / 2);
  }
  return std::max(1, std::min(batch, n_points));
}

#ifdef FAISSR_HAS_FAISS_GPU
faiss::gpu::StandardGpuResources& reusable_faiss_gpu_resources() {
  static thread_local std::unique_ptr<faiss::gpu::StandardGpuResources> resources;
  if (resources == nullptr || env_false("FAISSR_FAISS_GPU_REUSE_RESOURCES")) {
    resources.reset(new faiss::gpu::StandardGpuResources());
  }
  return *resources;
}
#endif

struct MatrixViewF32 {
  const float* data = nullptr;
  int nrow = 0;
  int ncol = 0;
  bool row_major = true;
  bool owns_data = false;
  bool compatibility_conversion = false;
  std::string layout = "unknown";
  std::vector<float> buffer;
};

struct FaissHnswIndexHandle {
  std::unique_ptr<faiss::IndexHNSWFlat> index;
  int n = 0;
  int p = 0;
  int m = 0;
  int ef_construction = 0;
  int ef_search = 0;
  int requested_m = 0;
  int requested_ef_construction = 0;
  int requested_ef_search = 0;
  int max_threads = 1;
  DistanceOutput distance_output = DistanceOutput::L2Squared;
  std::string metric = "euclidean";
  std::string input_layout = "unknown";
  bool input_owns_data = false;

  FaissHnswIndexHandle(std::unique_ptr<faiss::IndexHNSWFlat>&& index_,
                       const int n_,
                       const int p_,
                       const int m_,
                       const int ef_construction_,
                       const int ef_search_,
                       const int requested_m_,
                       const int requested_ef_construction_,
                       const int requested_ef_search_,
                       const int max_threads_,
                       const DistanceOutput distance_output_,
                       std::string metric_,
                       std::string input_layout_,
                       const bool input_owns_data_)
      : index(std::move(index_)),
        n(n_),
        p(p_),
        m(m_),
        ef_construction(ef_construction_),
        ef_search(ef_search_),
        requested_m(requested_m_),
        requested_ef_construction(requested_ef_construction_),
        requested_ef_search(requested_ef_search_),
        max_threads(std::max(1, max_threads_)),
        distance_output(distance_output_),
        metric(std::move(metric_)),
        input_layout(std::move(input_layout_)),
        input_owns_data(input_owns_data_) {}
};

struct FaissFittedIndexHandle {
  std::unique_ptr<faiss::Index> index;
  std::string kind;
  std::string index_type;
  int n = 0;
  int p = 0;
  int nlist = NA_INTEGER;
  int nprobe = NA_INTEGER;
  int requested_nlist = NA_INTEGER;
  int requested_nprobe = NA_INTEGER;
  int pq_m = NA_INTEGER;
  int pq_nbits = NA_INTEGER;
  int requested_pq_m = NA_INTEGER;
  int requested_pq_nbits = NA_INTEGER;
  int graph_degree = NA_INTEGER;
  int search_width = NA_INTEGER;
  int requested_graph_degree = NA_INTEGER;
  int requested_search_width = NA_INTEGER;
  int build_type = NA_INTEGER;
  int requested_build_type = NA_INTEGER;
  int n_iter = NA_INTEGER;
  int requested_n_iter = NA_INTEGER;
  int gk = NA_INTEGER;
  int max_threads = 1;
  DistanceOutput distance_output = DistanceOutput::L2Squared;
  std::string metric = "euclidean";
  std::string input_layout = "unknown";
  bool input_owns_data = false;
  bool index_trained = false;
  bool centroids_trained = false;
  bool inverted_lists_built = false;
  bool pq_codebooks_trained = false;
  bool pq_codes_built = false;
  int build_train_call_count = 0;
  int build_pq_train_call_count = 0;

  FaissFittedIndexHandle(std::unique_ptr<faiss::Index>&& index_,
                         std::string kind_,
                         std::string index_type_,
                         const int n_,
                         const int p_,
                         const int max_threads_,
                         const DistanceOutput distance_output_,
                         std::string metric_,
                         std::string input_layout_,
                         const bool input_owns_data_)
      : index(std::move(index_)),
        kind(std::move(kind_)),
        index_type(std::move(index_type_)),
        n(n_),
        p(p_),
        max_threads(std::max(1, max_threads_)),
        distance_output(distance_output_),
        metric(std::move(metric_)),
        input_layout(std::move(input_layout_)),
        input_owns_data(input_owns_data_) {}
};

Rcpp::IntegerVector matrix_dims_from_object(SEXP x, const char* name) {
  SEXP dim = Rf_getAttrib(x, R_DimSymbol);
  if (Rf_isNull(dim) && Rf_isS4(x)) {
    SEXP data_slot = R_do_slot(x, Rf_install("Data"));
    dim = Rf_getAttrib(data_slot, R_DimSymbol);
  }
  if (Rf_isNull(dim) || Rf_length(dim) != 2) {
    Rcpp::stop("%s must be a two-dimensional numeric or float32 matrix", name);
  }
  Rcpp::IntegerVector dims(dim);
  if (dims[0] < 1 || dims[1] < 1) {
    Rcpp::stop("%s must have at least one row and one column", name);
  }
  if (dims[0] > std::numeric_limits<int>::max() ||
      dims[1] > std::numeric_limits<int>::max()) {
    Rcpp::stop("%s dimensions exceed FAISS int limits", name);
  }
  return dims;
}

const float* float32_slot_ptr(SEXP slot, const int expected_length, const char* name) {
  if (TYPEOF(slot) == INTSXP) {
    if (Rf_length(slot) != expected_length) {
      Rcpp::stop("%s float32 payload length does not match its dimensions", name);
    }
    return reinterpret_cast<const float*>(INTEGER(slot));
  }
  if (TYPEOF(slot) == RAWSXP) {
    const R_xlen_t expected_bytes = static_cast<R_xlen_t>(expected_length) *
      static_cast<R_xlen_t>(sizeof(float));
    if (Rf_xlength(slot) != expected_bytes) {
      Rcpp::stop("%s float32 raw payload length does not match its dimensions", name);
    }
    return reinterpret_cast<const float*>(RAW(slot));
  }
  return nullptr;
}

bool finite_float32_payload(const float* ptr, const int length) {
  bool finite = true;
#ifdef _OPENMP
#pragma omp parallel for schedule(static) reduction(&& : finite)
#endif
  for (int i = 0; i < length; ++i) {
    if (!std::isfinite(ptr[i])) finite = false;
  }
  return finite;
}

MatrixViewF32 make_float32_matrix_view(SEXP x, const char* name) {
  Rcpp::IntegerVector dims = matrix_dims_from_object(x, name);
  MatrixViewF32 view;
  view.nrow = dims[0];
  view.ncol = dims[1];
  const int expected_length = view.nrow * view.ncol;

  bool finite = true;
  if (TYPEOF(x) == REALSXP) {
    view.buffer.assign(static_cast<std::size_t>(expected_length), 0.0f);
    view.owns_data = true;
    view.compatibility_conversion = true;
    view.row_major = true;
    view.layout = "r_double_column_major_to_row_major_float32";
    if (Rf_length(x) != expected_length) {
      Rcpp::stop("%s payload length does not match its dimensions", name);
    }
    const double* col_major_double = REAL(x);
#ifdef _OPENMP
#pragma omp parallel for schedule(static) reduction(&& : finite)
#endif
    for (int r = 0; r < view.nrow; ++r) {
      for (int c = 0; c < view.ncol; ++c) {
        const double value = col_major_double[r + view.nrow * c];
        if (!std::isfinite(value)) {
          finite = false;
          continue;
        }
        view.buffer[static_cast<std::size_t>(r) * view.ncol + c] =
          static_cast<float>(value);
      }
    }
  } else if (Rf_isS4(x)) {
    SEXP slot = R_do_slot(x, Rf_install("Data"));
    const float* col_major = float32_slot_ptr(slot, expected_length, name);
    if (col_major != nullptr) {
      finite = finite_float32_payload(col_major, expected_length);
      const bool row_major_payload =
        Rf_asLogical(Rf_getAttrib(x, Rf_install("faissR_row_major_float32"))) == TRUE;
      if (row_major_payload) {
        view.data = col_major;
        view.owns_data = false;
        view.row_major = true;
        view.layout = "float32_payload_direct_row_major";
      } else if (view.nrow == 1 || view.ncol == 1) {
        view.data = col_major;
        view.owns_data = false;
        view.row_major = true;
        view.layout = "float32_payload_direct_row_compatible";
      } else {
        view.buffer.assign(static_cast<std::size_t>(expected_length), 0.0f);
        view.owns_data = true;
        view.row_major = true;
        view.layout = "float32_column_major_payload_to_row_major";
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
        for (int r = 0; r < view.nrow; ++r) {
          for (int c = 0; c < view.ncol; ++c) {
            view.buffer[static_cast<std::size_t>(r) * view.ncol + c] =
              col_major[r + view.nrow * c];
          }
        }
      }
    } else if (TYPEOF(slot) == REALSXP) {
      view.buffer.assign(static_cast<std::size_t>(expected_length), 0.0f);
      view.owns_data = true;
      view.compatibility_conversion = true;
      view.row_major = true;
      view.layout = "s4_double_column_major_to_row_major_float32";
      const double* col_major_double = REAL(slot);
      if (Rf_length(slot) != expected_length) {
        Rcpp::stop("%s payload length does not match its dimensions", name);
      }
#ifdef _OPENMP
#pragma omp parallel for schedule(static) reduction(&& : finite)
#endif
      for (int r = 0; r < view.nrow; ++r) {
        for (int c = 0; c < view.ncol; ++c) {
          const double value = col_major_double[r + view.nrow * c];
          if (!std::isfinite(value)) {
            finite = false;
            continue;
          }
          view.buffer[static_cast<std::size_t>(r) * view.ncol + c] =
            static_cast<float>(value);
        }
      }
    } else {
      Rcpp::stop(
        "%s must be a float::fl()/float32 object with an integer or raw @Data payload",
        name
      );
    }
  } else {
    Rcpp::stop(
      "%s must be an ordinary R double matrix or a float::fl()/float32 object",
      name
    );
  }
  if (!finite) {
    Rcpp::stop("FAISS float32 input requires finite values");
  }
  if (view.data == nullptr) {
    view.data = view.buffer.data();
  }
  return view;
}

std::vector<char> normalize_float32_view(MatrixViewF32& view,
                                         const std::string& metric) {
  std::vector<char> zero(static_cast<std::size_t>(view.nrow), 0);
  if (metric != "cosine" && metric != "correlation") {
    return zero;
  }

  if (!view.owns_data || view.buffer.empty()) {
    const float* source = view.data;
    view.buffer.assign(static_cast<std::size_t>(view.nrow) * view.ncol, 0.0f);
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int r = 0; r < view.nrow; ++r) {
      for (int c = 0; c < view.ncol; ++c) {
        view.buffer[static_cast<std::size_t>(r) * view.ncol + c] =
          source[static_cast<std::size_t>(r) * view.ncol + c];
      }
    }
    view.data = view.buffer.data();
    view.owns_data = true;
    view.layout += "_normalized_copy";
  }

#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
  for (int r = 0; r < view.nrow; ++r) {
    float* row = view.buffer.data() + static_cast<std::size_t>(r) * view.ncol;
    double mean = 0.0;
    if (metric == "correlation") {
      for (int c = 0; c < view.ncol; ++c) {
        mean += static_cast<double>(row[c]);
      }
      mean /= static_cast<double>(view.ncol);
    }
    double norm2 = 0.0;
    for (int c = 0; c < view.ncol; ++c) {
      const double centered = static_cast<double>(row[c]) - mean;
      row[c] = static_cast<float>(centered);
      norm2 += centered * centered;
    }
    if (norm2 <= 0.0 || !std::isfinite(norm2)) {
      zero[static_cast<std::size_t>(r)] = 1;
      std::fill(row, row + view.ncol, 0.0f);
      continue;
    }
    const float inv_norm = static_cast<float>(1.0 / std::sqrt(norm2));
    for (int c = 0; c < view.ncol; ++c) {
      row[c] *= inv_norm;
    }
  }
  view.data = view.buffer.data();
  return zero;
}

bool same_float32_object(SEXP data, SEXP points) {
  return data == points;
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
                         const DistanceOutput distance_output,
                         const int n_threads,
                         const int nlist,
                         const int nprobe,
                         const int graph_degree,
                         const int search_width,
                         const bool float_distances = false);

struct FaissBatchSearchInfo {
  int batch_size = 1;
  int batches = 1;
};

FaissBatchSearchInfo search_faiss_queries(faiss::Index& index,
                                          const float* query_ptr,
                                          int n_points,
                                          int n_features,
                                          int search_k,
                                          bool use_ivf_search_params,
                                          int nprobe,
                                          bool gpu_index,
                                          std::vector<float>& distances,
                                          std::vector<faiss::idx_t>& labels);

List search_faiss_flat_float32_ptr(const float* data,
                                   const int n,
                                   const int p,
                                   const float* points,
                                   const int n_points,
                                   const int k,
                                   const bool exclude_self,
                                   const bool self_query,
                                   const int n_threads,
                                   const std::string& metric,
                                   const bool float_distances = false) {
  if (data == nullptr || points == nullptr) {
    Rcpp::stop("FAISS float32 Flat search received a null data pointer");
  }
  if (n < 1 || p < 1 || n_points < 1) {
    Rcpp::stop("FAISS float32 Flat search requires positive dimensions");
  }
  if (k < 1 || k > n) {
    Rcpp::stop("k must be in [1, nrow(data)]");
  }
  const bool normalized_metric = metric == "cosine" || metric == "correlation";
  const DistanceOutput output = metric == "inner_product" ?
    DistanceOutput::InnerProduct :
    (normalized_metric ? DistanceOutput::OneMinusInnerProduct : DistanceOutput::L2Squared);

  OmpThreadScope threads(n_threads);
  std::unique_ptr<faiss::Index> index;
  std::string index_type;
  if (metric == "inner_product" || normalized_metric) {
    index.reset(new faiss::IndexFlatIP(p));
    index_type = "IndexFlatIP";
  } else if (metric == "euclidean") {
    index.reset(new faiss::IndexFlatL2(p));
    index_type = "IndexFlatL2";
  } else {
    Rcpp::stop(
      "FAISS float32 Flat input supports metric = 'euclidean', "
      "'cosine', 'correlation', or 'inner_product'"
    );
  }

  const int search_k = exclude_self ? std::min(n, k + 1) : k;
  std::vector<float> distances(static_cast<std::size_t>(n_points) * search_k);
  std::vector<faiss::idx_t> labels(static_cast<std::size_t>(n_points) * search_k);
  FaissBatchSearchInfo batch_info;
  try {
    index->add(n, data);
    batch_info = search_faiss_queries(
      *index,
      points,
      n_points,
      p,
      search_k,
      false,
      NA_INTEGER,
      false,
      distances,
      labels
    );
  } catch (const std::exception& e) {
    Rcpp::stop("FAISS float32 Flat search failed: %s", e.what());
  }

  List out = format_faiss_result(
    labels,
    distances,
    n_points,
    search_k,
    k,
    self_query,
    exclude_self,
    index_type,
    true,
    output,
    n_threads,
    NA_INTEGER,
    NA_INTEGER,
    NA_INTEGER,
    NA_INTEGER,
    float_distances
  );
  out["search_batch_size"] = batch_info.batch_size;
  out["search_batches"] = batch_info.batches;
  out["batch_query"] = true;
  out["query_n"] = n_points;
  out["query_call_count"] = batch_info.batches;
  return out;
}

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
                         const int search_width = NA_INTEGER,
                         const bool float_distances) {
  IntegerMatrix indices(n_points, out_k);
  NumericMatrix dists;
  IntegerMatrix float_dists;
  if (float_distances) {
    float_dists = IntegerMatrix(n_points, out_k);
  } else {
    dists = NumericMatrix(n_points, out_k);
  }
  int* indices_ptr = indices.begin();
  double* dists_ptr = float_distances ? nullptr : dists.begin();
  int* float_dists_ptr = float_distances ? float_dists.begin() : nullptr;
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
      if (float_distances) {
        const float value_f = static_cast<float>(value);
        std::memcpy(
          float_dists_ptr + output_offset,
          &value_f,
          sizeof(float)
        );
      } else {
        dists_ptr[output_offset] = value;
      }
      ++written;
    }
    if (written < out_k) {
      complete = false;
    }
  }
  if (!complete) {
    Rcpp::stop("FAISS returned fewer neighbors than requested");
  }

  SEXP distance_sexp = dists;
  if (float_distances) {
    Rcpp::Environment base = Rcpp::Environment::namespace_env("base");
    Rcpp::Function require_namespace = base["requireNamespace"];
    const bool ok = Rcpp::as<bool>(
      require_namespace("float", Rcpp::Named("quietly") = true)
    );
    if (!ok) {
      Rcpp::stop("`output = \"float\"` requires the optional float package");
    }
    Rcpp::S4 float_matrix("float32");
    float_matrix.slot("Data") = float_dists;
    distance_sexp = float_matrix;
  }

  List out = List::create(
    Rcpp::Named("indices") = indices,
    Rcpp::Named("distances") = distance_sexp,
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
  if (float_distances) {
    out["distance_type"] = "float32";
    out.attr("distance_type") = "float32";
  }
  return out;
}

double float32_bytes(const int rows, const int cols) {
  return static_cast<double>(rows) * static_cast<double>(cols) *
    static_cast<double>(sizeof(float));
}

bool is_faiss_gpu_index_type(const std::string& index_type) {
  return index_type.rfind("GpuIndex", 0) == 0;
}

void annotate_faiss_gpu_residency(List& out,
                                  const bool same_storage,
                                  const int n_data,
                                  const int n_points,
                                  const int n_features,
                                  const int search_k) {
  const double data_bytes = float32_bytes(n_data, n_features);
  const double query_bytes = float32_bytes(n_points, n_features);
  const double result_bytes =
    static_cast<double>(n_points) * static_cast<double>(search_k) *
    static_cast<double>(sizeof(faiss::idx_t) + sizeof(float));
  out["accelerator"] = "cuda";
  out["gpu_provider"] = "faiss_gpu";
  out["device_residency"] = "cuda";
  out["index_residency"] = "gpu_transient";
  out["gpu_index_resident"] = true;
  out["gpu_index_persistent"] = false;
  out["gpu_resources_reused"] = !env_false("FAISSR_FAISS_GPU_REUSE_RESOURCES");
  out["gpu_resource_scope"] = "thread_local_standard_gpu_resources";
  out["host_to_device_transfer_strategy"] = "faiss_gpu_managed";
  out["host_to_device_copies_known"] = false;
  out["host_to_device_data_copies_minimum"] = 1;
  out["host_to_device_query_copies_minimum"] = 1;
  out["host_to_device_data_bytes_minimum"] = data_bytes;
  out["host_to_device_query_bytes_minimum"] = query_bytes;
  out["host_to_device_bytes_minimum"] = data_bytes + query_bytes;
  out["query_reuses_host_data"] = same_storage;
  out["query_reuses_device_data"] = false;
  out["device_to_host_result_copies_known"] = false;
  out["device_to_host_result_bytes_minimum"] = result_bytes;
  out["cpu_fallback"] = false;
  out["cpu_side_result_repair"] = false;
}

FaissBatchSearchInfo search_faiss_queries(faiss::Index& index,
                                          const float* query_ptr,
                                          const int n_points,
                                          const int n_features,
                                          const int search_k,
                                          const bool use_ivf_search_params,
                                          const int nprobe,
                                          const bool gpu_index,
                                          std::vector<float>& distances,
                                          std::vector<faiss::idx_t>& labels) {
  const int batch_size = faiss_query_batch_size(n_points, search_k, gpu_index);
  int batches = 0;
  faiss::SearchParametersIVF params;
  params.nprobe = static_cast<std::size_t>(std::max(1, nprobe));

  for (int offset = 0; offset < n_points; offset += batch_size) {
    const int current = std::min(batch_size, n_points - offset);
    const std::size_t output_offset =
      static_cast<std::size_t>(offset) * search_k;
    const float* batch_query = query_ptr +
      static_cast<std::size_t>(offset) * n_features;
    if (use_ivf_search_params && nprobe != NA_INTEGER) {
      index.search(
        current,
        batch_query,
        search_k,
        distances.data() + output_offset,
        labels.data() + output_offset,
        &params
      );
    } else {
      index.search(
        current,
        batch_query,
        search_k,
        distances.data() + output_offset,
        labels.data() + output_offset
      );
    }
    ++batches;
  }

  return FaissBatchSearchInfo{batch_size, batches};
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
  FaissBatchSearchInfo batch_info;

  try {
    if (!index.is_trained) {
      index.train(n_data, xb.data());
    }
    index.add(n_data, xb.data());
    batch_info = search_faiss_queries(
      index,
      query_ptr,
      n_points,
      data.ncol(),
      search_k,
      use_ivf_search_params,
      nprobe,
      is_faiss_gpu_index_type(index_type),
      distances,
      labels
    );
  } catch (const std::exception& e) {
    Rcpp::stop("FAISS %s search failed: %s", index_type.c_str(), e.what());
  }

  List out = format_faiss_result(
    labels, distances, n_points, search_k, k, self_query, exclude_self,
    index_type, exact, distance_output, n_threads, nlist, nprobe, graph_degree, search_width
  );
  out["search_batch_size"] = batch_info.batch_size;
  out["search_batches"] = batch_info.batches;
  out["batch_query"] = true;
  out["query_n"] = n_points;
  out["query_call_count"] = batch_info.batches;
  if (is_faiss_gpu_index_type(index_type)) {
    annotate_faiss_gpu_residency(
      out, same_storage, n_data, n_points, data.ncol(), search_k
    );
  }
  return out;
}

List search_faiss_index_float32(faiss::Index& index,
                                SEXP data,
                                SEXP points,
                                int k,
                                bool exclude_self,
                                int n_threads,
                                const std::string& index_type,
                                bool exact,
                                DistanceOutput distance_output,
                                const std::string& distance_storage,
                                int nlist = NA_INTEGER,
                                int nprobe = NA_INTEGER,
                                int graph_degree = NA_INTEGER,
                                int search_width = NA_INTEGER,
                                bool use_ivf_search_params = false) {
  MatrixViewF32 xb = make_float32_matrix_view(data, "data");
  const bool same_storage = same_float32_object(data, points);
  MatrixViewF32 xq = same_storage ? MatrixViewF32() :
    make_float32_matrix_view(points, "points");
  if (!same_storage && xq.ncol != xb.ncol) {
    Rcpp::stop("data and points must have the same number of columns");
  }
  if (k < 1 || k > xb.nrow) {
    Rcpp::stop("k must be in [1, nrow(data)]");
  }
  if (exclude_self && !same_storage) {
    Rcpp::stop("self-neighbor exclusion requires points to be data");
  }
  const int n_points = same_storage ? xb.nrow : xq.nrow;
  const int search_k = exclude_self ? std::min(xb.nrow, k + 1) : k;
  const float* query_ptr = same_storage ? xb.data : xq.data;
  const bool self_query = exclude_self || same_storage;
  const bool wants_float_distances = distance_storage == "float" ||
    distance_storage == "float32";
  if (!wants_float_distances && distance_storage != "double") {
    Rcpp::stop("`distance_storage` must be \"double\" or \"float\"");
  }

  OmpThreadScope threads(n_threads);
  std::vector<float> distances(static_cast<std::size_t>(n_points) * search_k);
  std::vector<faiss::idx_t> labels(static_cast<std::size_t>(n_points) * search_k);
  FaissBatchSearchInfo batch_info;

  try {
    if (!index.is_trained) {
      index.train(xb.nrow, xb.data);
    }
    index.add(xb.nrow, xb.data);
    batch_info = search_faiss_queries(
      index,
      query_ptr,
      n_points,
      xb.ncol,
      search_k,
      use_ivf_search_params,
      nprobe,
      is_faiss_gpu_index_type(index_type),
      distances,
      labels
    );
  } catch (const std::exception& e) {
    Rcpp::stop("FAISS %s float32 search failed: %s", index_type.c_str(), e.what());
  }

  List out = format_faiss_result(
    labels, distances, n_points, search_k, k, self_query, exclude_self,
    index_type, exact, distance_output, n_threads, nlist, nprobe,
    graph_degree, search_width, wants_float_distances
  );
  out["search_batch_size"] = batch_info.batch_size;
  out["search_batches"] = batch_info.batches;
  out["batch_query"] = true;
  out["query_n"] = n_points;
  out["query_call_count"] = batch_info.batches;
  out["input_type"] = "float32";
  out["input_layout"] = same_storage ?
    xb.layout :
    ("data=" + xb.layout + ";points=" + xq.layout);
  out["input_owns_data"] = same_storage ?
    xb.owns_data :
    (xb.owns_data || xq.owns_data);
  out["float32_compatibility_conversion"] = same_storage ?
    xb.compatibility_conversion :
    (xb.compatibility_conversion || xq.compatibility_conversion);
  if (is_faiss_gpu_index_type(index_type)) {
    annotate_faiss_gpu_residency(
      out, same_storage, xb.nrow, n_points, xb.ncol, search_k
    );
  }
  return out;
}

void sort_knn_result_rows(IntegerMatrix& indices, NumericMatrix& dists) {
  const int n_points = indices.nrow();
  const int k = indices.ncol();
  std::vector<int> order(static_cast<std::size_t>(k));
  for (int i = 0; i < n_points; ++i) {
    for (int j = 0; j < k; ++j) order[static_cast<std::size_t>(j)] = j;
    std::sort(order.begin(), order.end(), [&](const int a, const int b) {
      const double da = dists(i, a);
      const double db = dists(i, b);
      const bool a_na = indices(i, a) == NA_INTEGER || !std::isfinite(da);
      const bool b_na = indices(i, b) == NA_INTEGER || !std::isfinite(db);
      if (a_na != b_na) return !a_na;
      if (da != db) return da < db;
      return indices(i, a) < indices(i, b);
    });
    std::vector<int> idx(static_cast<std::size_t>(k));
    std::vector<double> dst(static_cast<std::size_t>(k));
    for (int j = 0; j < k; ++j) {
      const int source = order[static_cast<std::size_t>(j)];
      idx[static_cast<std::size_t>(j)] = indices(i, source);
      dst[static_cast<std::size_t>(j)] = dists(i, source);
    }
    for (int j = 0; j < k; ++j) {
      indices(i, j) = idx[static_cast<std::size_t>(j)];
      dists(i, j) = dst[static_cast<std::size_t>(j)];
    }
  }
}

void restore_zero_normalized_float32_rows(List& out,
                                          const std::vector<char>& data_zero,
                                          const std::vector<char>& points_zero,
                                          const bool self_query,
                                          const bool exclude_self) {
  if (data_zero.empty() || points_zero.empty()) return;
  bool any_data_zero = false;
  bool any_points_zero = false;
  for (char value : data_zero) any_data_zero = any_data_zero || value != 0;
  for (char value : points_zero) any_points_zero = any_points_zero || value != 0;
  if (!any_data_zero || !any_points_zero) return;

  IntegerMatrix indices = out["indices"];
  NumericMatrix dists = out["distances"];
  const int n_data = static_cast<int>(data_zero.size());
  const int n_points = indices.nrow();
  const int k = indices.ncol();
  std::vector<int> zero_candidates;
  std::vector<int> nonzero_candidates;
  zero_candidates.reserve(static_cast<std::size_t>(n_data));
  nonzero_candidates.reserve(static_cast<std::size_t>(n_data));
  for (int i = 0; i < n_data; ++i) {
    if (data_zero[static_cast<std::size_t>(i)] != 0) {
      zero_candidates.push_back(i + 1);
    } else {
      nonzero_candidates.push_back(i + 1);
    }
  }

  for (int i = 0; i < n_points; ++i) {
    if (points_zero[static_cast<std::size_t>(i)] == 0) continue;
    int written = 0;
    int zero_written = 0;
    for (int candidate : zero_candidates) {
      if (self_query && exclude_self && candidate == i + 1) continue;
      if (written >= k) break;
      indices(i, written) = candidate;
      dists(i, written) = 0.0;
      ++written;
      ++zero_written;
    }
    for (int candidate : nonzero_candidates) {
      if (written >= k) break;
      indices(i, written) = candidate;
      dists(i, written) = 1.0;
      ++written;
    }
    for (int j = written; j < k; ++j) {
      indices(i, j) = NA_INTEGER;
      dists(i, j) = R_PosInf;
    }
    (void)zero_written;
  }
  sort_knn_result_rows(indices, dists);
}

List search_faiss_flat_float32(SEXP data,
                               SEXP points,
                               int k,
                               bool exclude_self,
                               int n_threads,
                               const std::string& metric,
                               const std::string& distance_storage = "double") {
  MatrixViewF32 xb = make_float32_matrix_view(data, "data");
  MatrixViewF32 xq = same_float32_object(data, points) ?
    MatrixViewF32() :
    make_float32_matrix_view(points, "points");
  const bool same_storage = same_float32_object(data, points);
  const bool self_query = exclude_self || same_storage;
  if (xb.ncol < 1 || xb.nrow < 1) {
    Rcpp::stop("data must have at least one row and one column");
  }
  if (!same_storage && (xq.nrow < 1 || xq.ncol != xb.ncol)) {
    Rcpp::stop("data and points must have the same number of columns");
  }
  if (k < 1 || k > xb.nrow) {
    Rcpp::stop("k must be in [1, nrow(data)]");
  }
  if (exclude_self && !same_storage) {
    Rcpp::stop("self-neighbor exclusion requires points to be data");
  }

  std::vector<char> data_zero = normalize_float32_view(xb, metric);
  std::vector<char> points_zero = same_storage ?
    data_zero :
    normalize_float32_view(xq, metric);
  const bool normalized_metric = metric == "cosine" || metric == "correlation";
  const bool wants_float_distances = distance_storage == "float" ||
    distance_storage == "float32";
  if (!wants_float_distances && distance_storage != "double") {
    Rcpp::stop("`distance_storage` must be \"double\" or \"float\"");
  }
  auto has_zero = [](const std::vector<char>& values) {
    for (char value : values) {
      if (value != 0) return true;
    }
    return false;
  };
  const bool has_zero_normalized_rows = normalized_metric &&
    (has_zero(data_zero) || has_zero(points_zero));
  const bool direct_float_distances = wants_float_distances &&
    !has_zero_normalized_rows;

  const int n_points = same_storage ? xb.nrow : xq.nrow;
  const float* query_ptr = same_storage ? xb.data : xq.data;
  List out = search_faiss_flat_float32_ptr(
    xb.data,
    xb.nrow,
    xb.ncol,
    query_ptr,
    n_points,
    k,
    exclude_self,
    self_query,
    n_threads,
    metric,
    direct_float_distances
  );
  if (normalized_metric && !direct_float_distances) {
    restore_zero_normalized_float32_rows(
      out, data_zero, points_zero, self_query, exclude_self
    );
    IntegerMatrix indices = out["indices"];
    NumericMatrix dists = out["distances"];
    sort_knn_result_rows(indices, dists);
    out["metric"] = metric;
    out["metric_transform"] = metric == "correlation" ?
      "row_center_l2_normalize_then_IndexFlatIP" :
      "row_l2_normalize_then_IndexFlatIP";
  }
  out["input_type"] = "float32";
  out["input_layout"] = same_storage ?
    xb.layout :
    ("data=" + xb.layout + ";points=" + xq.layout);
  out["input_owns_data"] = same_storage ?
    xb.owns_data :
    (xb.owns_data || xq.owns_data);
  out["float32_compatibility_conversion"] = same_storage ?
    xb.compatibility_conversion :
    (xb.compatibility_conversion || xq.compatibility_conversion);
  if (direct_float_distances) {
    out["distance_type"] = "float32";
    out.attr("distance_type") = "float32";
  }
  return out;
}

int clamp_positive(const int value, const int fallback, const int upper) {
  int out = value > 0 ? value : fallback;
  if (upper > 0) out = std::min(out, upper);
  return std::max(1, out);
}

int adjust_fastscan_pq_m(const int requested, const int n_features) {
  int pq_m = clamp_positive(requested, std::min(32, n_features), n_features);
  while (pq_m > 1 && (n_features % pq_m) != 0) --pq_m;
  return std::max(1, pq_m);
}

int adjust_fastscan_bbs(const int requested) {
  if (requested <= 0) return 32;
  if (requested <= 32) return 32;
  const int rounded = (requested / 32) * 32;
  return std::max(32, rounded);
}

[[maybe_unused]] bool faiss_gpu_supported_pq_code_size(const int code_size) {
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
#ifdef FAISSR_HAS_FAISS_GPU
  const char* gpu = "true";
#else
  const char* gpu = "false";
#endif
#ifdef FAISSR_HAS_FAISS_GPU_CAGRA
  const char* gpu_cagra = "true";
#else
  const char* gpu_cagra = "false";
#endif
#ifdef FAISSR_HAS_FAISS_FASTSCAN
  const char* fastscan = "true";
#else
  const char* fastscan = "false";
#endif
  return std::string("{\"available\":true,\"library\":\"faiss\",\"interface\":\"c++\",") +
    "\"gpu\":" + gpu + ",\"gpu_cagra\":" + gpu_cagra +
    ",\"fastscan\":" + fastscan + "}";
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

List faiss_flat_float32_knn_impl(SEXP data,
                                 SEXP points,
                                 int k,
                                 bool exclude_self,
                                 int n_threads,
                                 std::string metric,
                                 std::string distance_storage) {
  return search_faiss_flat_float32(
    data,
    points,
    k,
    exclude_self,
    n_threads,
    metric,
    distance_storage
  );
}

List faiss_flat_pretransformed_float32_knn_impl(SEXP data,
                                                SEXP points,
                                                int k,
                                                bool exclude_self,
                                                int n_threads,
                                                std::string distance_storage) {
  MatrixViewF32 xb = make_float32_matrix_view(data, "data");
  MatrixViewF32 xq = same_float32_object(data, points) ?
    MatrixViewF32() :
    make_float32_matrix_view(points, "points");
  const bool same_storage = same_float32_object(data, points);
  const bool self_query = exclude_self || same_storage;
  if (!same_storage && (xq.nrow < 1 || xq.ncol != xb.ncol)) {
    Rcpp::stop("data and points must have the same number of columns");
  }
  const bool wants_float_distances = distance_storage == "float" ||
    distance_storage == "float32";
  if (!wants_float_distances && distance_storage != "double") {
    Rcpp::stop("`distance_storage` must be \"double\" or \"float\"");
  }
  const int n_points = same_storage ? xb.nrow : xq.nrow;
  const float* query_ptr = same_storage ? xb.data : xq.data;
  List out = search_faiss_flat_float32_ptr(
    xb.data,
    xb.nrow,
    xb.ncol,
    query_ptr,
    n_points,
    k,
    exclude_self,
    self_query,
    n_threads,
    "cosine",
    wants_float_distances
  );
  out["input_type"] = "float32";
  out["input_layout"] = same_storage ?
    xb.layout :
    ("data=" + xb.layout + ";points=" + xq.layout);
  out["input_owns_data"] = same_storage ?
    xb.owns_data :
    (xb.owns_data || xq.owns_data);
  out["float32_compatibility_conversion"] = same_storage ?
    xb.compatibility_conversion :
    (xb.compatibility_conversion || xq.compatibility_conversion);
  if (wants_float_distances) {
    out["distance_type"] = "float32";
    out.attr("distance_type") = "float32";
  }
  out["metric"] = "one_minus_inner_product";
  return out;
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
#ifdef FAISSR_HAS_FAISS_GPU
  const int n_features = data.ncol();
  faiss::gpu::StandardGpuResources& resources = reusable_faiss_gpu_resources();
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

List faiss_gpu_flat_float32_knn_impl(SEXP data,
                                     SEXP points,
                                     int k,
                                     bool exclude_self,
                                     std::string metric,
                                     std::string distance_output,
                                     std::string distance_storage) {
#ifdef FAISSR_HAS_FAISS_GPU
  Rcpp::IntegerVector dims = matrix_dims_from_object(data, "data");
  const int n_features = dims[1];
  faiss::gpu::StandardGpuResources& resources = reusable_faiss_gpu_resources();
  faiss::gpu::GpuIndexFlatConfig config;
  config.device = 0;
  const DistanceOutput output = parse_distance_output(distance_output, "GPU Flat");
  if (metric == "inner_product") {
    faiss::gpu::GpuIndexFlatIP index(&resources, n_features, config);
    return search_faiss_index_float32(
      index, data, points, k, exclude_self, 1,
      "GpuIndexFlatIP", true, output, distance_storage
    );
  }
  if (metric == "euclidean") {
    faiss::gpu::GpuIndexFlatL2 index(&resources, n_features, config);
    return search_faiss_index_float32(
      index, data, points, k, exclude_self, 1,
      "GpuIndexFlatL2", true, output, distance_storage
    );
  }
  Rcpp::stop("FAISS GPU Flat float32 supports metric = 'euclidean' or 'inner_product'");
#else
  (void)data;
  (void)points;
  (void)k;
  (void)exclude_self;
  (void)metric;
  (void)distance_output;
  (void)distance_storage;
  Rcpp::stop(
    "FAISS GPU Flat backend is not available in this build. "
    "Install FAISS GPU/cuVS headers and rebuild faissR with FAISS_HOME set."
  );
#endif
}

List faiss_gpu_flat_ip_knn_impl(NumericMatrix data,
                                NumericMatrix points,
                                int k,
                                bool exclude_self) {
#ifdef FAISSR_HAS_FAISS_GPU
  const int n_features = data.ncol();
  faiss::gpu::StandardGpuResources& resources = reusable_faiss_gpu_resources();
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
#ifdef FAISSR_HAS_FAISS_GPU
  const int n_features = data.ncol();
  faiss::gpu::StandardGpuResources& resources = reusable_faiss_gpu_resources();
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

List faiss_ivf_float32_knn_impl(SEXP data,
                                SEXP points,
                                int k,
                                int nlist,
                                int nprobe,
                                std::string metric,
                                std::string distance_output,
                                bool exclude_self,
                                int n_threads,
                                std::string distance_storage) {
  Rcpp::IntegerVector dims = matrix_dims_from_object(data, "data");
  const int n_data = dims[0];
  const int n_features = dims[1];
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
    Rcpp::stop("FAISS IVF float32 supports metric = 'euclidean' or 'inner_product'");
  }
  const DistanceOutput output = parse_distance_output(distance_output, "IVF");
  faiss::IndexIVFFlat index(quantizer.get(), n_features, nlist, faiss_metric);
  index.nprobe = nprobe;
  return search_faiss_index_float32(
    index, data, points, k, exclude_self, n_threads,
    metric == "inner_product" ? "IndexIVFFlatIP" : "IndexIVFFlat",
    false,
    output,
    distance_storage,
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
                                  std::string distance_storage) {
  Rcpp::IntegerVector dims = matrix_dims_from_object(data, "data");
  const int n_data = dims[0];
  const int n_features = dims[1];
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
    Rcpp::stop("FAISS IVFPQ float32 supports metric = 'euclidean' or 'inner_product'");
  }
  const DistanceOutput output = parse_distance_output(distance_output, "IVFPQ");
  faiss::IndexIVFPQ index(
    quantizer.get(), n_features, nlist, pq_m, pq_nbits, faiss_metric
  );
  index.nprobe = nprobe;
  List out = search_faiss_index_float32(
    index, data, points, k, exclude_self, n_threads,
    metric == "inner_product" ? "IndexIVFPQIP" : "IndexIVFPQ",
    false,
    output,
    distance_storage,
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

List faiss_ivfpq_fastscan_knn_impl(NumericMatrix data,
                          NumericMatrix points,
                          int k,
                          int nlist,
                          int nprobe,
                          int pq_m,
                          int refine_factor,
                          int bbs,
                          bool exclude_self,
                          int n_threads) {
#ifdef FAISSR_HAS_FAISS_FASTSCAN
  const int n_data = data.nrow();
  const int n_features = data.ncol();
  const int requested_nlist = nlist;
  const int requested_nprobe = nprobe;
  const int requested_pq_m = pq_m;
  const int requested_refine_factor = refine_factor;
  const int requested_bbs = bbs;
  nlist = std::max(1, std::min(nlist, n_data));
  nprobe = std::max(1, std::min(nprobe, nlist));
  pq_m = adjust_fastscan_pq_m(pq_m, n_features);
  refine_factor = std::max(1, refine_factor);
  bbs = adjust_fastscan_bbs(bbs);

  std::unique_ptr<faiss::Index> quantizer(new faiss::IndexFlatL2(n_features));
  std::unique_ptr<faiss::IndexIVFPQFastScan> base(
    new faiss::IndexIVFPQFastScan(
      quantizer.get(),
      n_features,
      nlist,
      pq_m,
      4,
      faiss::METRIC_L2,
      bbs
    )
  );
  base->nprobe = nprobe;

  std::unique_ptr<faiss::Index> index;
  const char* index_type = "IndexIVFPQFastScan";
  if (refine_factor > 1) {
    faiss::IndexIVFPQFastScan* base_ptr = base.release();
    std::unique_ptr<faiss::IndexRefineFlat> refine(
      new faiss::IndexRefineFlat(base_ptr)
    );
    refine->own_fields = true;
    refine->k_factor = static_cast<float>(refine_factor);
    index = std::move(refine);
    index_type = "IndexIVFPQFastScanRefineFlat";
  } else {
    index = std::move(base);
  }

  List out = search_faiss_index(
    *index, data, points, k, exclude_self, n_threads,
    index_type,
    false,
    DistanceOutput::L2Squared,
    nlist,
    nprobe
  );
  out["pq_m"] = pq_m;
  out["pq_nbits"] = 4;
  out["requested_nlist"] = requested_nlist;
  out["requested_nprobe"] = requested_nprobe;
  out["requested_pq_m"] = requested_pq_m;
  out["requested_pq_nbits"] = 4;
  out["pq_parameters_adjusted"] = requested_pq_m != pq_m;
  out["refine_factor"] = refine_factor;
  out["requested_refine_factor"] = requested_refine_factor;
  out["bbs"] = bbs;
  out["requested_bbs"] = requested_bbs;
  out["fastscan"] = true;
  out["ivfpq_fastscan"] = true;
  out["refine"] = refine_factor > 1;
  return out;
#else
  (void)data;
  (void)points;
  (void)k;
  (void)nlist;
  (void)nprobe;
  (void)pq_m;
  (void)refine_factor;
  (void)bbs;
  (void)exclude_self;
  (void)n_threads;
  Rcpp::stop(
    "FAISS FastScan is not available in this build. Rebuild faissR against "
    "a FAISS version that provides faiss/IndexIVFPQFastScan.h."
  );
#endif
}

List faiss_ivfpq_fastscan_float32_knn_impl(SEXP data,
                                  SEXP points,
                                  int k,
                                  int nlist,
                                  int nprobe,
                                  int pq_m,
                                  int refine_factor,
                                  int bbs,
                                  bool exclude_self,
                                  int n_threads,
                                  std::string distance_storage) {
#ifdef FAISSR_HAS_FAISS_FASTSCAN
  Rcpp::IntegerVector dims = matrix_dims_from_object(data, "data");
  const int n_data = dims[0];
  const int n_features = dims[1];
  const int requested_nlist = nlist;
  const int requested_nprobe = nprobe;
  const int requested_pq_m = pq_m;
  const int requested_refine_factor = refine_factor;
  const int requested_bbs = bbs;
  nlist = std::max(1, std::min(nlist, n_data));
  nprobe = std::max(1, std::min(nprobe, nlist));
  pq_m = adjust_fastscan_pq_m(pq_m, n_features);
  refine_factor = std::max(1, refine_factor);
  bbs = adjust_fastscan_bbs(bbs);

  std::unique_ptr<faiss::Index> quantizer(new faiss::IndexFlatL2(n_features));
  std::unique_ptr<faiss::IndexIVFPQFastScan> base(
    new faiss::IndexIVFPQFastScan(
      quantizer.get(),
      n_features,
      nlist,
      pq_m,
      4,
      faiss::METRIC_L2,
      bbs
    )
  );
  base->nprobe = nprobe;

  std::unique_ptr<faiss::Index> index;
  const char* index_type = "IndexIVFPQFastScan";
  if (refine_factor > 1) {
    faiss::IndexIVFPQFastScan* base_ptr = base.release();
    std::unique_ptr<faiss::IndexRefineFlat> refine(
      new faiss::IndexRefineFlat(base_ptr)
    );
    refine->own_fields = true;
    refine->k_factor = static_cast<float>(refine_factor);
    index = std::move(refine);
    index_type = "IndexIVFPQFastScanRefineFlat";
  } else {
    index = std::move(base);
  }

  List out = search_faiss_index_float32(
    *index, data, points, k, exclude_self, n_threads,
    index_type,
    false,
    DistanceOutput::L2Squared,
    distance_storage,
    nlist,
    nprobe
  );
  out["pq_m"] = pq_m;
  out["pq_nbits"] = 4;
  out["requested_nlist"] = requested_nlist;
  out["requested_nprobe"] = requested_nprobe;
  out["requested_pq_m"] = requested_pq_m;
  out["requested_pq_nbits"] = 4;
  out["pq_parameters_adjusted"] = requested_pq_m != pq_m;
  out["refine_factor"] = refine_factor;
  out["requested_refine_factor"] = requested_refine_factor;
  out["bbs"] = bbs;
  out["requested_bbs"] = requested_bbs;
  out["fastscan"] = true;
  out["ivfpq_fastscan"] = true;
  out["refine"] = refine_factor > 1;
  return out;
#else
  (void)data;
  (void)points;
  (void)k;
  (void)nlist;
  (void)nprobe;
  (void)pq_m;
  (void)refine_factor;
  (void)bbs;
  (void)exclude_self;
  (void)n_threads;
  (void)distance_storage;
  Rcpp::stop(
    "FAISS FastScan is not available in this build. Rebuild faissR against "
    "a FAISS version that provides faiss/IndexIVFPQFastScan.h."
  );
#endif
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
                                 std::string distance_storage) {
  Rcpp::IntegerVector dims = matrix_dims_from_object(data, "data");
  const int n_data = dims[0];
  const int n_features = dims[1];
  const int requested_m = m;
  const int requested_ef_construction = ef_construction;
  const int requested_ef_search = ef_search;
  m = clamp_positive(m, 32, n_data);
  ef_construction = std::max(ef_construction, m);
  ef_search = std::max(ef_search, k);
  faiss::MetricType faiss_metric = faiss::METRIC_L2;
  if (metric == "inner_product") {
    faiss_metric = faiss::METRIC_INNER_PRODUCT;
  } else if (metric != "euclidean") {
    Rcpp::stop("FAISS HNSW float32 supports metric = 'euclidean' or 'inner_product'");
  }
  const DistanceOutput output = parse_distance_output(distance_output, "HNSW");
  faiss::IndexHNSWFlat index(n_features, m, faiss_metric);
  index.hnsw.efConstruction = ef_construction;
  index.hnsw.efSearch = ef_search;
  List out = search_faiss_index_float32(
    index, data, points, k, exclude_self, n_threads,
    "IndexHNSWFlat", false, output, distance_storage,
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

SEXP faiss_hnsw_index_build_float32_impl(SEXP data,
                                         int m,
                                         int ef_construction,
                                         int ef_search,
                                         std::string metric,
                                         std::string distance_output,
                                         int n_threads) {
  Rcpp::IntegerVector dims = matrix_dims_from_object(data, "data");
  const int n_data = dims[0];
  const int n_features = dims[1];
  const int requested_m = m;
  const int requested_ef_construction = ef_construction;
  const int requested_ef_search = ef_search;
  n_threads = std::max(1, n_threads);
  m = clamp_positive(m, 32, n_data);
  ef_construction = std::max(ef_construction, m);
  ef_search = std::max(ef_search, 1);

  faiss::MetricType faiss_metric = faiss::METRIC_L2;
  if (metric == "inner_product") {
    faiss_metric = faiss::METRIC_INNER_PRODUCT;
  } else if (metric != "euclidean") {
    Rcpp::stop("FAISS HNSW fitted index supports metric = 'euclidean' or 'inner_product'");
  }
  const DistanceOutput output = parse_distance_output(distance_output, "HNSW");
  MatrixViewF32 xb = make_float32_matrix_view(data, "data");

  try {
    std::unique_ptr<faiss::IndexHNSWFlat> index(
      new faiss::IndexHNSWFlat(n_features, m, faiss_metric)
    );
    index->hnsw.efConstruction = ef_construction;
    index->hnsw.efSearch = ef_search;

    {
      OmpThreadScope threads(n_threads);
      index->add(n_data, xb.data);
    }

    FaissHnswIndexHandle* handle = new FaissHnswIndexHandle(
      std::move(index),
      n_data,
      n_features,
      m,
      ef_construction,
      ef_search,
      requested_m,
      requested_ef_construction,
      requested_ef_search,
      n_threads,
      output,
      metric,
      xb.layout,
      xb.owns_data
    );
    Rcpp::XPtr<FaissHnswIndexHandle> ptr(handle, true);
    ptr.attr("class") = "faissR_faiss_hnsw_index";
    ptr.attr("n") = n_data;
    ptr.attr("p") = n_features;
    ptr.attr("m") = m;
    ptr.attr("ef_construction") = ef_construction;
    ptr.attr("ef_search") = ef_search;
    ptr.attr("requested_m") = requested_m;
    ptr.attr("requested_ef_construction") = requested_ef_construction;
    ptr.attr("requested_ef_search") = requested_ef_search;
    ptr.attr("max_threads") = n_threads;
    ptr.attr("metric") = metric;
    ptr.attr("input_type") = "float32";
    ptr.attr("input_layout") = xb.layout;
    ptr.attr("input_owns_data") = xb.owns_data;
    ptr.attr("float32_compatibility_conversion") = xb.compatibility_conversion;
    return ptr;
  } catch (const std::exception& e) {
    Rcpp::stop("FAISS HNSW fitted-index build failed: %s", e.what());
  }
}

List faiss_hnsw_index_search_float32_impl(SEXP index_ptr,
                                          SEXP points,
                                          int k,
                                          bool exclude_self,
                                          int ef_search,
                                          int n_threads,
                                          std::string distance_storage) {
  if (k < 1) {
    Rcpp::stop("k must be positive");
  }
  Rcpp::XPtr<FaissHnswIndexHandle> handle(index_ptr);
  if (handle.get() == nullptr || handle->index.get() == nullptr) {
    Rcpp::stop("FAISS HNSW index pointer is not valid");
  }
  if (k > handle->n) {
    Rcpp::stop("k must not exceed the fitted FAISS HNSW index size");
  }
  n_threads = std::max(1, std::min(n_threads, handle->max_threads));
  ef_search = std::max(k, ef_search);
  const bool wants_float_distances = distance_storage == "float" ||
    distance_storage == "float32";
  if (!wants_float_distances && distance_storage != "double") {
    Rcpp::stop("`distance_storage` must be \"double\" or \"float\"");
  }

  MatrixViewF32 xq = make_float32_matrix_view(points, "points");
  if (xq.ncol != handle->p) {
    Rcpp::stop("points must have the same number of columns as the fitted FAISS HNSW index");
  }
  const int n_points = xq.nrow;
  const bool self_query = exclude_self;
  const int search_k = exclude_self ? std::min(handle->n, k + 1) : k;
  std::vector<float> distances(static_cast<std::size_t>(n_points) * search_k);
  std::vector<faiss::idx_t> labels(static_cast<std::size_t>(n_points) * search_k);

  try {
    OmpThreadScope threads(n_threads);
    handle->index->hnsw.efSearch = ef_search;
    handle->index->search(n_points, xq.data, search_k, distances.data(), labels.data());
  } catch (const std::exception& e) {
    Rcpp::stop("FAISS HNSW fitted-index search failed: %s", e.what());
  }

  List out = format_faiss_result(
    labels,
    distances,
    n_points,
    search_k,
    k,
    self_query,
    exclude_self,
    "IndexHNSWFlatExternalPtr",
    false,
    handle->distance_output,
    n_threads,
    NA_INTEGER,
    NA_INTEGER,
    handle->m,
    ef_search,
    wants_float_distances
  );
  out["m"] = handle->m;
  out["ef_construction"] = handle->ef_construction;
  out["ef_search"] = ef_search;
  out["requested_m"] = handle->requested_m;
  out["requested_ef_construction"] = handle->requested_ef_construction;
  out["requested_ef_search"] = handle->requested_ef_search;
  out["hnsw_parameters_adjusted"] = handle->requested_m != handle->m ||
    handle->requested_ef_construction != handle->ef_construction ||
    handle->requested_ef_search != ef_search;
  out["index_reused"] = true;
  out["index_n"] = handle->n;
  out["index_p"] = handle->p;
  out["query_n"] = n_points;
  out["batch_query"] = true;
  out["query_call_count"] = 1;
  out["input_type"] = "float32";
  out["input_layout"] = handle->input_layout + ";fitted_index_query:" + xq.layout;
  out["input_owns_data"] = xq.owns_data;
  out["index_input_owns_data"] = handle->input_owns_data;
  out["float32_compatibility_conversion"] = xq.compatibility_conversion;
  return out;
}

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
                                    int n_threads) {
  Rcpp::IntegerVector dims = matrix_dims_from_object(data, "data");
  const int n_data = dims[0];
  const int n_features = dims[1];
  n_threads = std::max(1, n_threads);
  std::transform(kind.begin(), kind.end(), kind.begin(),
                 [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  if (kind == "ivf_flat") kind = "ivf";
  if (kind == "ivf_pq") kind = "ivfpq";
  const DistanceOutput output = parse_distance_output(distance_output, "fitted FAISS index");
  MatrixViewF32 xb = make_float32_matrix_view(data, "data");

  std::unique_ptr<faiss::Index> index;
  std::string index_type;
  const int requested_nlist = nlist;
  const int requested_nprobe = nprobe;
  const int requested_pq_m = pq_m;
  const int requested_pq_nbits = pq_nbits;
  const int requested_graph_degree = graph_degree;
  const int requested_search_width = search_width;
  const int requested_build_type = build_type;
  const int requested_n_iter = n_iter;
  int gk = NA_INTEGER;

  try {
    if (kind == "flat") {
      if (metric == "inner_product") {
        index.reset(new faiss::IndexFlatIP(n_features));
        index_type = "IndexFlatIPExternalPtr";
      } else if (metric == "euclidean") {
        index.reset(new faiss::IndexFlatL2(n_features));
        index_type = "IndexFlatL2ExternalPtr";
      } else {
        Rcpp::stop("FAISS fitted Flat supports metric = 'euclidean' or 'inner_product'");
      }
    } else if (kind == "ivf") {
      nlist = std::max(1, std::min(nlist, n_data));
      nprobe = std::max(1, std::min(nprobe, nlist));
      faiss::MetricType faiss_metric = faiss::METRIC_L2;
      faiss::Index* quantizer = nullptr;
      if (metric == "inner_product") {
        faiss_metric = faiss::METRIC_INNER_PRODUCT;
        quantizer = new faiss::IndexFlatIP(n_features);
        index_type = "IndexIVFFlatIPExternalPtr";
      } else if (metric == "euclidean") {
        quantizer = new faiss::IndexFlatL2(n_features);
        index_type = "IndexIVFFlatExternalPtr";
      } else {
        Rcpp::stop("FAISS fitted IVF supports metric = 'euclidean' or 'inner_product'");
      }
      std::unique_ptr<faiss::IndexIVFFlat> ivf(
        new faiss::IndexIVFFlat(quantizer, n_features, nlist, faiss_metric)
      );
      ivf->own_fields = true;
      ivf->nprobe = nprobe;
      index = std::move(ivf);
    } else if (kind == "ivfpq") {
      nlist = std::max(1, std::min(nlist, n_data));
      nprobe = std::max(1, std::min(nprobe, nlist));
      pq_m = clamp_positive(pq_m, 8, n_features);
      while (pq_m > 1 && (n_features % pq_m) != 0) --pq_m;
      pq_nbits = std::max(4, std::min(pq_nbits, 12));
      while (pq_nbits > 4 && (1 << pq_nbits) > n_data) {
        --pq_nbits;
      }
      faiss::MetricType faiss_metric = faiss::METRIC_L2;
      faiss::Index* quantizer = nullptr;
      if (metric == "inner_product") {
        faiss_metric = faiss::METRIC_INNER_PRODUCT;
        quantizer = new faiss::IndexFlatIP(n_features);
        index_type = "IndexIVFPQIPExternalPtr";
      } else if (metric == "euclidean") {
        quantizer = new faiss::IndexFlatL2(n_features);
        index_type = "IndexIVFPQExternalPtr";
      } else {
        Rcpp::stop("FAISS fitted IVFPQ supports metric = 'euclidean' or 'inner_product'");
      }
      std::unique_ptr<faiss::IndexIVFPQ> ivfpq(
        new faiss::IndexIVFPQ(
          quantizer, n_features, nlist, pq_m, pq_nbits, faiss_metric
        )
      );
      ivfpq->own_fields = true;
      ivfpq->nprobe = nprobe;
      index = std::move(ivfpq);
    } else if (kind == "nsg") {
      if (n_data <= 100) {
        Rcpp::stop("FAISS NSG requires more than 100 training rows in this FAISS build.");
      }
      if (metric != "euclidean") {
        Rcpp::stop("FAISS fitted NSG is currently validated only for metric = 'euclidean'");
      }
      graph_degree = clamp_positive(graph_degree, 32, n_data);
      search_width = std::max(search_width, 1);
      build_type = build_type == 1 ? 1 : 0;
      gk = std::max(64, std::max(2 * graph_degree, 2 * search_width));
      std::unique_ptr<faiss::IndexNSGFlat> nsg(
        new faiss::IndexNSGFlat(n_features, graph_degree, faiss::METRIC_L2)
      );
      nsg->nsg.search_L = search_width;
      nsg->build_type = static_cast<char>(build_type);
      nsg->GK = gk;
      index_type = "IndexNSGFlatExternalPtr";
      index = std::move(nsg);
    } else if (kind == "nndescent") {
      if (n_data <= 100) {
        Rcpp::stop("FAISS NN-Descent requires more than 100 training rows in this FAISS build.");
      }
      if (metric != "euclidean") {
        Rcpp::stop("FAISS fitted NNDescent is currently validated only for metric = 'euclidean'");
      }
      graph_degree = std::max(1, graph_degree);
      n_iter = std::max(1, n_iter);
      search_width = std::max(search_width, 1);
      std::unique_ptr<faiss::IndexNNDescentFlat> nnd(
        new faiss::IndexNNDescentFlat(n_features, graph_degree, faiss::METRIC_L2)
      );
      nnd->nndescent.iter = n_iter;
      nnd->nndescent.search_L = search_width;
      index_type = "IndexNNDescentFlatExternalPtr";
      index = std::move(nnd);
    } else {
      Rcpp::stop("Unsupported fitted FAISS index kind: %s", kind);
    }

    int build_train_call_count = 0;
    {
      OmpThreadScope threads(n_threads);
      const bool needs_training = !index->is_trained;
      if (!index->is_trained) {
        index->train(n_data, xb.data);
      }
      index->add(n_data, xb.data);
      if (needs_training) {
        build_train_call_count = 1;
      }
    }

    FaissFittedIndexHandle* handle = new FaissFittedIndexHandle(
      std::move(index),
      kind,
      index_type,
      n_data,
      n_features,
      n_threads,
      output,
      metric,
      xb.layout,
      xb.owns_data
    );
    handle->nlist = nlist;
    handle->nprobe = nprobe;
    handle->requested_nlist = requested_nlist;
    handle->requested_nprobe = requested_nprobe;
    handle->pq_m = pq_m;
    handle->pq_nbits = pq_nbits;
    handle->requested_pq_m = requested_pq_m;
    handle->requested_pq_nbits = requested_pq_nbits;
    handle->graph_degree = graph_degree;
    handle->search_width = search_width;
    handle->requested_graph_degree = requested_graph_degree;
    handle->requested_search_width = requested_search_width;
    handle->build_type = build_type;
    handle->requested_build_type = requested_build_type;
    handle->n_iter = n_iter;
    handle->requested_n_iter = requested_n_iter;
    handle->gk = gk;
    handle->index_trained = handle->index->is_trained;
    handle->centroids_trained = kind == "ivf" || kind == "ivfpq";
    handle->inverted_lists_built = kind == "ivf" || kind == "ivfpq";
    handle->pq_codebooks_trained = kind == "ivfpq";
    handle->pq_codes_built = kind == "ivfpq";
    handle->build_train_call_count = handle->centroids_trained ? build_train_call_count : 0;
    handle->build_pq_train_call_count = handle->pq_codebooks_trained ? build_train_call_count : 0;

    Rcpp::XPtr<FaissFittedIndexHandle> ptr(handle, true);
    const std::string primary_class = "faissR_faiss_" + kind + "_index";
    ptr.attr("class") = Rcpp::CharacterVector::create(
      primary_class,
      "faissR_faiss_index"
    );
    ptr.attr("kind") = kind;
    ptr.attr("index_type") = index_type;
    ptr.attr("n") = n_data;
    ptr.attr("p") = n_features;
    ptr.attr("nlist") = nlist;
    ptr.attr("nprobe") = nprobe;
    ptr.attr("pq_m") = pq_m;
    ptr.attr("pq_nbits") = pq_nbits;
    ptr.attr("graph_degree") = graph_degree;
    ptr.attr("search_width") = search_width;
    ptr.attr("build_type") = build_type;
    ptr.attr("n_iter") = n_iter;
    ptr.attr("max_threads") = n_threads;
    ptr.attr("metric") = metric;
    ptr.attr("input_type") = "float32";
    ptr.attr("input_layout") = xb.layout;
    ptr.attr("input_owns_data") = xb.owns_data;
    ptr.attr("float32_compatibility_conversion") = xb.compatibility_conversion;
    ptr.attr("index_trained") = handle->index_trained;
    ptr.attr("centroids_trained") = handle->centroids_trained;
    ptr.attr("inverted_lists_built") = handle->inverted_lists_built;
    ptr.attr("build_train_call_count") = handle->build_train_call_count;
    ptr.attr("pq_codebooks_trained") = handle->pq_codebooks_trained;
    ptr.attr("pq_codes_built") = handle->pq_codes_built;
    ptr.attr("build_pq_train_call_count") = handle->build_pq_train_call_count;
    return ptr;
  } catch (const std::exception& e) {
    Rcpp::stop("FAISS fitted-index build failed: %s", e.what());
  }
}

List faiss_index_search_float32_impl(SEXP index_ptr,
                                     SEXP points,
                                     int k,
                                     bool exclude_self,
                                     int search_width,
                                     int n_threads,
                                     std::string distance_storage) {
  if (k < 1) {
    Rcpp::stop("k must be positive");
  }
  Rcpp::XPtr<FaissFittedIndexHandle> handle(index_ptr);
  if (handle.get() == nullptr || handle->index.get() == nullptr) {
    Rcpp::stop("FAISS index pointer is not valid");
  }
  if (k > handle->n) {
    Rcpp::stop("k must not exceed the fitted FAISS index size");
  }
  n_threads = std::max(1, std::min(n_threads, handle->max_threads));
  const bool wants_float_distances = distance_storage == "float" ||
    distance_storage == "float32";
  if (!wants_float_distances && distance_storage != "double") {
    Rcpp::stop("`distance_storage` must be \"double\" or \"float\"");
  }

  int effective_search_width = search_width;
  if (effective_search_width < 1) {
    effective_search_width = handle->search_width;
  }
  if (handle->kind == "ivf" || handle->kind == "ivfpq") {
    effective_search_width = std::max(1, std::min(effective_search_width, handle->nlist));
    faiss::IndexIVF* ivf = dynamic_cast<faiss::IndexIVF*>(handle->index.get());
    if (ivf == nullptr) {
      Rcpp::stop("Stored FAISS IVF index pointer has an unexpected type");
    }
    ivf->nprobe = effective_search_width;
  } else if (handle->kind == "nsg") {
    effective_search_width = std::max(k, effective_search_width);
    faiss::IndexNSGFlat* nsg = dynamic_cast<faiss::IndexNSGFlat*>(handle->index.get());
    if (nsg == nullptr) {
      Rcpp::stop("Stored FAISS NSG index pointer has an unexpected type");
    }
    nsg->nsg.search_L = effective_search_width;
  } else if (handle->kind == "nndescent") {
    effective_search_width = std::max(k, effective_search_width);
    faiss::IndexNNDescentFlat* nnd = dynamic_cast<faiss::IndexNNDescentFlat*>(handle->index.get());
    if (nnd == nullptr) {
      Rcpp::stop("Stored FAISS NNDescent index pointer has an unexpected type");
    }
    nnd->nndescent.search_L = effective_search_width;
  }

  MatrixViewF32 xq = make_float32_matrix_view(points, "points");
  if (xq.ncol != handle->p) {
    Rcpp::stop("points must have the same number of columns as the fitted FAISS index");
  }
  const int n_points = xq.nrow;
  const bool self_query = exclude_self;
  const int search_k = exclude_self ? std::min(handle->n, k + 1) : k;
  std::vector<float> distances(static_cast<std::size_t>(n_points) * search_k);
  std::vector<faiss::idx_t> labels(static_cast<std::size_t>(n_points) * search_k);

  try {
    OmpThreadScope threads(n_threads);
    handle->index->search(n_points, xq.data, search_k, distances.data(), labels.data());
  } catch (const std::exception& e) {
    Rcpp::stop("FAISS fitted-index search failed: %s", e.what());
  }

  const int out_nlist = (handle->kind == "ivf" || handle->kind == "ivfpq") ?
    handle->nlist : NA_INTEGER;
  const int out_nprobe = (handle->kind == "ivf" || handle->kind == "ivfpq") ?
    effective_search_width : NA_INTEGER;
  const int out_graph_degree = (handle->kind == "nsg" || handle->kind == "nndescent") ?
    handle->graph_degree : NA_INTEGER;
  const int out_search_width = (handle->kind == "nsg" || handle->kind == "nndescent") ?
    effective_search_width : NA_INTEGER;

  List out = format_faiss_result(
    labels,
    distances,
    n_points,
    search_k,
    k,
    self_query,
    exclude_self,
    handle->index_type,
    false,
    handle->distance_output,
    n_threads,
    out_nlist,
    out_nprobe,
    out_graph_degree,
    out_search_width,
    wants_float_distances
  );
  out["index_reused"] = true;
  out["index_n"] = handle->n;
  out["index_p"] = handle->p;
  out["index_trained"] = handle->index->is_trained;
  out["index_training_reused"] = handle->index->is_trained;
  out["build_train_call_count"] = handle->build_train_call_count;
  out["search_train_call_count"] = 0;
  out["vectors_reused"] = true;
  out["query_n"] = n_points;
  out["batch_query"] = true;
  out["query_call_count"] = 1;
  out["input_type"] = "float32";
  out["input_layout"] = handle->input_layout + ";fitted_index_query:" + xq.layout;
  out["input_owns_data"] = xq.owns_data;
  out["index_input_owns_data"] = handle->input_owns_data;
  out["float32_compatibility_conversion"] = xq.compatibility_conversion;
  out["kind"] = handle->kind;

  if (handle->kind == "ivf" || handle->kind == "ivfpq") {
    out["nlist"] = handle->nlist;
    out["nprobe"] = effective_search_width;
    out["build_nprobe"] = handle->nprobe;
    out["search_nprobe"] = effective_search_width;
    out["requested_nlist"] = handle->requested_nlist;
    out["requested_nprobe"] = handle->requested_nprobe;
    out["centroids_reused"] = handle->centroids_trained;
    out["inverted_lists_reused"] = handle->inverted_lists_built;
    out["ivf_parameters_adjusted"] = handle->requested_nlist != handle->nlist ||
      handle->requested_nprobe != effective_search_width;
  }
  if (handle->kind == "ivfpq") {
    out["pq_m"] = handle->pq_m;
    out["pq_nbits"] = handle->pq_nbits;
    out["requested_pq_m"] = handle->requested_pq_m;
    out["requested_pq_nbits"] = handle->requested_pq_nbits;
    out["pq_codebooks_reused"] = handle->pq_codebooks_trained;
    out["pq_codes_reused"] = handle->pq_codes_built;
    out["pq_training_reused"] = handle->pq_codebooks_trained;
    out["build_pq_train_call_count"] = handle->build_pq_train_call_count;
    out["search_pq_train_call_count"] = 0;
    out["pq_parameters_adjusted"] = handle->requested_pq_m != handle->pq_m ||
      handle->requested_pq_nbits != handle->pq_nbits;
  }
  if (handle->kind == "nsg") {
    out["r"] = handle->graph_degree;
    out["search_l"] = effective_search_width;
    out["build_type"] = handle->build_type;
    out["gk"] = handle->gk;
    out["requested_r"] = handle->requested_graph_degree;
    out["requested_search_l"] = handle->requested_search_width;
    out["requested_build_type"] = handle->requested_build_type;
    out["nsg_parameters_adjusted"] = handle->requested_graph_degree != handle->graph_degree ||
      handle->requested_search_width != effective_search_width ||
      handle->requested_build_type != handle->build_type;
  }
  if (handle->kind == "nndescent") {
    out["graph_k"] = handle->graph_degree;
    out["n_iter"] = handle->n_iter;
    out["search_l"] = effective_search_width;
    out["requested_graph_k"] = handle->requested_graph_degree;
    out["requested_n_iter"] = handle->requested_n_iter;
    out["requested_search_l"] = handle->requested_search_width;
    out["nndescent_parameters_adjusted"] = handle->requested_graph_degree != handle->graph_degree ||
      handle->requested_n_iter != handle->n_iter ||
      handle->requested_search_width != effective_search_width;
  }
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
                                std::string distance_storage) {
  Rcpp::IntegerVector dims = matrix_dims_from_object(data, "data");
  const int n_data = dims[0];
  const int n_features = dims[1];
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
    Rcpp::stop("FAISS NSG float32 is currently validated only for metric = 'euclidean'");
  }
  const DistanceOutput output = parse_distance_output(distance_output, "NSG");
  faiss::IndexNSGFlat index(n_features, r, faiss::METRIC_L2);
  index.nsg.search_L = search_l;
  index.build_type = static_cast<char>(build_type);
  const int gk = std::max(64, std::max(2 * k, 2 * r));
  index.GK = gk;
  List out = search_faiss_index_float32(
    index, data, points, k, exclude_self, n_threads,
    "IndexNSGFlat",
    false,
    output,
    distance_storage,
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
                                      std::string distance_storage) {
  Rcpp::IntegerVector dims = matrix_dims_from_object(data, "data");
  const int n_data = dims[0];
  const int n_features = dims[1];
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
    Rcpp::stop("FAISS NNDescent float32 is currently validated only for metric = 'euclidean'");
  }
  const DistanceOutput output = parse_distance_output(distance_output, "NNDescent");
  faiss::IndexNNDescentFlat index(n_features, graph_k, faiss::METRIC_L2);
  index.nndescent.iter = n_iter;
  index.nndescent.search_L = search_l;
  List out = search_faiss_index_float32(
    index, data, points, k, exclude_self, n_threads,
    "IndexNNDescentFlat",
    false,
    output,
    distance_storage,
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
#ifdef FAISSR_HAS_FAISS_GPU
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
  faiss::gpu::StandardGpuResources& resources = reusable_faiss_gpu_resources();
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

List faiss_gpu_ivf_flat_float32_knn_impl(SEXP data,
                                         SEXP points,
                                         int k,
                                         int nlist,
                                         int nprobe,
                                         std::string metric,
                                         std::string distance_output,
                                         bool exclude_self,
                                         std::string distance_storage) {
#ifdef FAISSR_HAS_FAISS_GPU
  Rcpp::IntegerVector dims = matrix_dims_from_object(data, "data");
  const int n_data = dims[0];
  const int n_features = dims[1];
  nlist = std::max(1, std::min(nlist, n_data));
  nprobe = std::max(1, std::min(nprobe, nlist));
  faiss::MetricType faiss_metric = faiss::METRIC_L2;
  if (metric == "inner_product") {
    faiss_metric = faiss::METRIC_INNER_PRODUCT;
  } else if (metric != "euclidean") {
    Rcpp::stop("FAISS GPU IVF Flat float32 supports metric = 'euclidean' or 'inner_product'");
  }
  const DistanceOutput output = parse_distance_output(distance_output, "GPU IVF Flat");
  faiss::gpu::StandardGpuResources& resources = reusable_faiss_gpu_resources();
  faiss::gpu::GpuIndexIVFFlatConfig config;
  config.device = 0;
  faiss::gpu::GpuIndexIVFFlat index(
    &resources,
    n_features,
    nlist,
    faiss_metric,
    config
  );
  return search_faiss_index_float32(
    index, data, points, k, exclude_self, 1,
    metric == "inner_product" ? "GpuIndexIVFFlatIP_cuVS" : "GpuIndexIVFFlat_cuVS",
    false,
    output,
    distance_storage,
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
  (void)distance_storage;
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
#ifdef FAISSR_HAS_FAISS_GPU
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
  faiss::gpu::StandardGpuResources& resources = reusable_faiss_gpu_resources();
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
                                      std::string distance_storage) {
#ifdef FAISSR_HAS_FAISS_GPU
  Rcpp::IntegerVector dims = matrix_dims_from_object(data, "data");
  const int n_data = dims[0];
  const int n_features = dims[1];
  const int requested_pq_m = pq_m;
  const int requested_pq_nbits = pq_nbits;
  nlist = std::max(1, std::min(nlist, n_data));
  nprobe = std::max(1, std::min(nprobe, nlist));
  faiss::MetricType faiss_metric = faiss::METRIC_L2;
  if (metric == "inner_product") {
    faiss_metric = faiss::METRIC_INNER_PRODUCT;
  } else if (metric != "euclidean") {
    Rcpp::stop("FAISS GPU IVFPQ float32 supports metric = 'euclidean' or 'inner_product'");
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
  faiss::gpu::StandardGpuResources& resources = reusable_faiss_gpu_resources();
  faiss::gpu::GpuIndexIVFPQConfig config;
  config.device = 0;
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
  List out = search_faiss_index_float32(
    index, data, points, k, exclude_self, 1,
    metric == "inner_product" ? "GpuIndexIVFPQIP_cuVS" : "GpuIndexIVFPQ_cuVS",
    false,
    output,
    distance_storage,
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
  (void)distance_storage;
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
#ifdef FAISSR_HAS_FAISS_GPU_CAGRA
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

  faiss::gpu::StandardGpuResources& resources = reusable_faiss_gpu_resources();
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
  annotate_faiss_gpu_residency(
    out, same_storage, n_data, n_points, n_features, search_k
  );
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

List faiss_gpu_cagra_float32_knn_impl(SEXP data,
                                      SEXP points,
                                      int k,
                                      int graph_degree,
                                      int intermediate_graph_degree,
                                      int search_width,
                                      int itopk_size,
                                      bool exclude_self,
                                      std::string distance_storage) {
#ifdef FAISSR_HAS_FAISS_GPU_CAGRA
  MatrixViewF32 xb = make_float32_matrix_view(data, "data");
  const bool same_storage = same_float32_object(data, points);
  MatrixViewF32 xq = same_storage ? MatrixViewF32() :
    make_float32_matrix_view(points, "points");
  if (!same_storage && xq.ncol != xb.ncol) {
    Rcpp::stop("data and points must have the same number of columns");
  }
  if (k < 1 || k > xb.nrow) {
    Rcpp::stop("k must be in [1, nrow(data)]");
  }
  if (exclude_self && !same_storage) {
    Rcpp::stop("self-neighbor exclusion requires points to be data");
  }
  const bool self_query = exclude_self || same_storage;
  const int n_data = xb.nrow;
  const int n_points = same_storage ? xb.nrow : xq.nrow;
  const int n_features = xb.ncol;
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

  const float* query_ptr = same_storage ? xb.data : xq.data;
  const bool wants_float_distances = distance_storage == "float" ||
    distance_storage == "float32";
  if (!wants_float_distances && distance_storage != "double") {
    Rcpp::stop("`distance_storage` must be \"double\" or \"float\"");
  }

  faiss::gpu::StandardGpuResources& resources = reusable_faiss_gpu_resources();
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
    index.train(n_data, xb.data);
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
    Rcpp::stop("FAISS GpuIndexCagra float32 search failed: %s", e.what());
  }

  List out = format_faiss_result(
    labels, distances, n_points, search_k, k, self_query, exclude_self,
    "GpuIndexCagra_cuVS", false, DistanceOutput::L2Squared,
    1, NA_INTEGER, NA_INTEGER, graph_degree, search_width,
    wants_float_distances
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
  annotate_faiss_gpu_residency(
    out, same_storage, n_data, n_points, n_features, search_k
  );
  out["input_type"] = "float32";
  out["input_layout"] = same_storage ?
    xb.layout :
    ("data=" + xb.layout + ";points=" + xq.layout);
  out["input_owns_data"] = same_storage ?
    xb.owns_data :
    (xb.owns_data || xq.owns_data);
  out["float32_compatibility_conversion"] = same_storage ?
    xb.compatibility_conversion :
    (xb.compatibility_conversion || xq.compatibility_conversion);
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
  (void)distance_storage;
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
#ifdef FAISSR_HAS_FAISS_GPU
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
    faiss::gpu::StandardGpuResources& resources = reusable_faiss_gpu_resources();
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
