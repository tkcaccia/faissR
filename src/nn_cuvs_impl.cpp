#include <Rcpp.h>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <limits>
#include <cstdlib>
#include <sstream>
#include <string>
#include <vector>

#include <cuda_runtime.h>
#include <cuvs/core/c_api.h>
#if defined(__has_include)
#  if __has_include(<cuvs/cluster/kmeans.h>)
#    include <cuvs/cluster/kmeans.h>
#    define FAISSR_HAS_CUVS_KMEANS 1
#  endif
#endif
#include <cuvs/distance/distance.h>
#include <cuvs/neighbors/brute_force.h>
#include <cuvs/neighbors/cagra.h>
#include <cuvs/neighbors/nn_descent.h>
#if defined(__has_include)
#  if __has_include(<cuvs/neighbors/ivf_flat.h>)
#    include <cuvs/neighbors/ivf_flat.h>
#    define FAISSR_HAS_CUVS_IVF_FLAT 1
#  endif
#  if __has_include(<cuvs/neighbors/ivf_pq.h>)
#    include <cuvs/neighbors/ivf_pq.h>
#    define FAISSR_HAS_CUVS_IVF_PQ 1
#  endif
#  if __has_include(<cuvs/neighbors/hnsw.h>)
#    include <cuvs/neighbors/hnsw.h>
#    define FAISSR_HAS_CUVS_HNSW 1
#  endif
#endif
#include <dlpack/dlpack.h>

using Rcpp::IntegerMatrix;
using Rcpp::List;
using Rcpp::NumericMatrix;

namespace {

void cuvs_check(const cuvsError_t status, const char* context) {
  if (status == CUVS_SUCCESS) return;
  const char* detail = cuvsGetLastErrorText();
  if (detail != nullptr && detail[0] != '\0') {
    Rcpp::stop("%s failed: %s", context, detail);
  }
  Rcpp::stop("%s failed.", context);
}

void cuda_check(const cudaError_t status, const char* context) {
  if (status == cudaSuccess) return;
  Rcpp::stop("%s failed: %s", context, cudaGetErrorString(status));
}

void cuda_sync(const char* context) {
  cuda_check(cudaDeviceSynchronize(), context);
}

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
    Rcpp::stop("cuVS backend currently supports dimensions that fit in int");
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
    Rcpp::stop("cuVS backend requires finite numeric input");
  }
}

bool same_matrix_storage(const NumericMatrix& data,
                         const NumericMatrix& points) {
  return data.nrow() == points.nrow() &&
    data.ncol() == points.ncol() &&
    data.begin() == points.begin();
}

int env_int(const char* name, const int default_value, const int min_value) {
  const char* raw = std::getenv(name);
  if (raw == nullptr || raw[0] == '\0') return default_value;
  char* end = nullptr;
  long value = std::strtol(raw, &end, 10);
  if (end == raw || value < min_value || value > std::numeric_limits<int>::max()) {
    return default_value;
  }
  return static_cast<int>(value);
}

bool env_flag(const char* name) {
  const char* raw = std::getenv(name);
  if (raw == nullptr || raw[0] == '\0') return false;
  return std::string(raw) != "0" && std::string(raw) != "false";
}

void cuvs_debug(const char* message) {
  if (env_flag("FAISSR_CUVS_DEBUG")) {
    Rcpp::Rcerr << "[faissR cuVS] " << message << "\n";
  }
}

DLManagedTensor make_tensor(void* data,
                            int64_t* shape,
                            const int ndim,
                            const DLDeviceType device_type,
                            const uint8_t code,
                            const uint8_t bits) {
  DLManagedTensor tensor{};
  tensor.dl_tensor.data = data;
  tensor.dl_tensor.device.device_type = device_type;
  tensor.dl_tensor.device.device_id = 0;
  tensor.dl_tensor.ndim = ndim;
  tensor.dl_tensor.dtype.code = code;
  tensor.dl_tensor.dtype.bits = bits;
  tensor.dl_tensor.dtype.lanes = 1;
  tensor.dl_tensor.shape = shape;
  tensor.dl_tensor.strides = nullptr;
  tensor.dl_tensor.byte_offset = 0;
  tensor.manager_ctx = nullptr;
  tensor.deleter = nullptr;
  return tensor;
}

class CuvsResources {
 public:
  CuvsResources() {
    cuvs_check(cuvsResourcesCreate(&res_), "cuvsResourcesCreate");
  }

  ~CuvsResources() {
    if (res_ != 0) {
      cuvsResourcesDestroy(res_);
    }
  }

  cuvsResources_t get() const { return res_; }

  CuvsResources(const CuvsResources&) = delete;
  CuvsResources& operator=(const CuvsResources&) = delete;

 private:
  cuvsResources_t res_ = 0;
};

class DeviceBuffer {
 public:
  DeviceBuffer() = default;

  DeviceBuffer(cuvsResources_t res, const std::size_t bytes) {
    reset(res, bytes);
  }

  ~DeviceBuffer() {
    if (ptr_ != nullptr) {
      cuvsRMMFree(res_, ptr_, bytes_);
    }
  }

  void reset(cuvsResources_t res, const std::size_t bytes) {
    if (ptr_ != nullptr) {
      cuvsRMMFree(res_, ptr_, bytes_);
      ptr_ = nullptr;
    }
    res_ = res;
    bytes_ = bytes;
    if (bytes_ > 0) {
      cuvs_check(cuvsRMMAlloc(res_, &ptr_, bytes_), "cuvsRMMAlloc");
    }
  }

  void* get() const { return ptr_; }

  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;

 private:
  cuvsResources_t res_ = 0;
  void* ptr_ = nullptr;
  std::size_t bytes_ = 0;
};

class BruteForceIndex {
 public:
  BruteForceIndex() {
    cuvs_check(cuvsBruteForceIndexCreate(&index_), "cuvsBruteForceIndexCreate");
  }
  ~BruteForceIndex() {
    if (index_ != nullptr) {
      cuvsBruteForceIndexDestroy(index_);
    }
  }
  cuvsBruteForceIndex_t get() const { return index_; }
  BruteForceIndex(const BruteForceIndex&) = delete;
  BruteForceIndex& operator=(const BruteForceIndex&) = delete;

 private:
  cuvsBruteForceIndex_t index_ = nullptr;
};

class CagraIndexParams {
 public:
  CagraIndexParams() {
    cuvs_check(cuvsCagraIndexParamsCreate(&params_), "cuvsCagraIndexParamsCreate");
  }
  ~CagraIndexParams() {
    if (params_ != nullptr) {
      cuvsCagraIndexParamsDestroy(params_);
    }
  }
  cuvsCagraIndexParams_t get() const { return params_; }
  CagraIndexParams(const CagraIndexParams&) = delete;
  CagraIndexParams& operator=(const CagraIndexParams&) = delete;

 private:
  cuvsCagraIndexParams_t params_ = nullptr;
};

class CagraSearchParams {
 public:
  CagraSearchParams() {
    cuvs_check(cuvsCagraSearchParamsCreate(&params_), "cuvsCagraSearchParamsCreate");
  }
  ~CagraSearchParams() {
    if (params_ != nullptr) {
      cuvsCagraSearchParamsDestroy(params_);
    }
  }
  cuvsCagraSearchParams_t get() const { return params_; }
  CagraSearchParams(const CagraSearchParams&) = delete;
  CagraSearchParams& operator=(const CagraSearchParams&) = delete;

 private:
  cuvsCagraSearchParams_t params_ = nullptr;
};

class CagraIndex {
 public:
  CagraIndex() {
    cuvs_check(cuvsCagraIndexCreate(&index_), "cuvsCagraIndexCreate");
  }
  ~CagraIndex() {
    reset();
  }
  void reset() {
    if (index_ != nullptr) {
      cuvsCagraIndexDestroy(index_);
      index_ = nullptr;
    }
  }
  cuvsCagraIndex_t get() const { return index_; }
  CagraIndex(const CagraIndex&) = delete;
  CagraIndex& operator=(const CagraIndex&) = delete;

 private:
  cuvsCagraIndex_t index_ = nullptr;
};

class NNDescentParams {
 public:
  NNDescentParams() {
    cuvs_check(
      cuvsNNDescentIndexParamsCreate(&params_),
      "cuvsNNDescentIndexParamsCreate"
    );
  }
  ~NNDescentParams() {
    if (params_ != nullptr) {
      cuvsNNDescentIndexParamsDestroy(params_);
    }
  }
  cuvsNNDescentIndexParams_t get() const { return params_; }
  NNDescentParams(const NNDescentParams&) = delete;
  NNDescentParams& operator=(const NNDescentParams&) = delete;

 private:
  cuvsNNDescentIndexParams_t params_ = nullptr;
};

class NNDescentIndex {
 public:
  NNDescentIndex() {
    cuvs_check(cuvsNNDescentIndexCreate(&index_), "cuvsNNDescentIndexCreate");
  }
  ~NNDescentIndex() {
    if (index_ != nullptr) {
      cuvsNNDescentIndexDestroy(index_);
    }
  }
  cuvsNNDescentIndex_t get() const { return index_; }
  NNDescentIndex(const NNDescentIndex&) = delete;
  NNDescentIndex& operator=(const NNDescentIndex&) = delete;

 private:
  cuvsNNDescentIndex_t index_ = nullptr;
};

#ifdef FAISSR_HAS_CUVS_KMEANS
class KMeansParams {
 public:
  KMeansParams() {
    cuvs_check(cuvsKMeansParamsCreate(&params_), "cuvsKMeansParamsCreate");
  }
  ~KMeansParams() {
    if (params_ != nullptr) {
      cuvsKMeansParamsDestroy(params_);
    }
  }
  cuvsKMeansParams_t get() const { return params_; }
  KMeansParams(const KMeansParams&) = delete;
  KMeansParams& operator=(const KMeansParams&) = delete;

 private:
  cuvsKMeansParams_t params_ = nullptr;
};
#endif

#ifdef FAISSR_HAS_CUVS_IVF_FLAT
class IvfFlatIndexParams {
 public:
  IvfFlatIndexParams() {
    cuvs_check(cuvsIvfFlatIndexParamsCreate(&params_), "cuvsIvfFlatIndexParamsCreate");
  }
  ~IvfFlatIndexParams() {
    if (params_ != nullptr) {
      cuvsIvfFlatIndexParamsDestroy(params_);
    }
  }
  cuvsIvfFlatIndexParams_t get() const { return params_; }
  IvfFlatIndexParams(const IvfFlatIndexParams&) = delete;
  IvfFlatIndexParams& operator=(const IvfFlatIndexParams&) = delete;

 private:
  cuvsIvfFlatIndexParams_t params_ = nullptr;
};

class IvfFlatSearchParams {
 public:
  IvfFlatSearchParams() {
    cuvs_check(cuvsIvfFlatSearchParamsCreate(&params_), "cuvsIvfFlatSearchParamsCreate");
  }
  ~IvfFlatSearchParams() {
    if (params_ != nullptr) {
      cuvsIvfFlatSearchParamsDestroy(params_);
    }
  }
  cuvsIvfFlatSearchParams_t get() const { return params_; }
  IvfFlatSearchParams(const IvfFlatSearchParams&) = delete;
  IvfFlatSearchParams& operator=(const IvfFlatSearchParams&) = delete;

 private:
  cuvsIvfFlatSearchParams_t params_ = nullptr;
};

class IvfFlatIndex {
 public:
  IvfFlatIndex() {
    cuvs_check(cuvsIvfFlatIndexCreate(&index_), "cuvsIvfFlatIndexCreate");
  }
  ~IvfFlatIndex() {
    if (index_ != nullptr) {
      cuvsIvfFlatIndexDestroy(index_);
    }
  }
  cuvsIvfFlatIndex_t get() const { return index_; }
  IvfFlatIndex(const IvfFlatIndex&) = delete;
  IvfFlatIndex& operator=(const IvfFlatIndex&) = delete;

 private:
  cuvsIvfFlatIndex_t index_ = nullptr;
};
#endif

#ifdef FAISSR_HAS_CUVS_IVF_PQ
class IvfPqIndexParams {
 public:
  IvfPqIndexParams() {
    cuvs_check(cuvsIvfPqIndexParamsCreate(&params_), "cuvsIvfPqIndexParamsCreate");
  }
  ~IvfPqIndexParams() {
    if (params_ != nullptr) {
      cuvsIvfPqIndexParamsDestroy(params_);
    }
  }
  cuvsIvfPqIndexParams_t get() const { return params_; }
  IvfPqIndexParams(const IvfPqIndexParams&) = delete;
  IvfPqIndexParams& operator=(const IvfPqIndexParams&) = delete;

 private:
  cuvsIvfPqIndexParams_t params_ = nullptr;
};

class IvfPqSearchParams {
 public:
  IvfPqSearchParams() {
    cuvs_check(cuvsIvfPqSearchParamsCreate(&params_), "cuvsIvfPqSearchParamsCreate");
  }
  ~IvfPqSearchParams() {
    if (params_ != nullptr) {
      cuvsIvfPqSearchParamsDestroy(params_);
    }
  }
  cuvsIvfPqSearchParams_t get() const { return params_; }
  IvfPqSearchParams(const IvfPqSearchParams&) = delete;
  IvfPqSearchParams& operator=(const IvfPqSearchParams&) = delete;

 private:
  cuvsIvfPqSearchParams_t params_ = nullptr;
};

class IvfPqIndex {
 public:
  IvfPqIndex() {
    cuvs_check(cuvsIvfPqIndexCreate(&index_), "cuvsIvfPqIndexCreate");
  }
  ~IvfPqIndex() {
    if (index_ != nullptr) {
      cuvsIvfPqIndexDestroy(index_);
    }
  }
  cuvsIvfPqIndex_t get() const { return index_; }
  IvfPqIndex(const IvfPqIndex&) = delete;
  IvfPqIndex& operator=(const IvfPqIndex&) = delete;

 private:
  cuvsIvfPqIndex_t index_ = nullptr;
};
#endif

#ifdef FAISSR_HAS_CUVS_HNSW
class HnswAceParams {
 public:
  HnswAceParams() {
    cuvs_check(cuvsHnswAceParamsCreate(&params_), "cuvsHnswAceParamsCreate");
  }
  ~HnswAceParams() {
    if (params_ != nullptr) {
      cuvsHnswAceParamsDestroy(params_);
    }
  }
  cuvsHnswAceParams_t get() const { return params_; }
  HnswAceParams(const HnswAceParams&) = delete;
  HnswAceParams& operator=(const HnswAceParams&) = delete;

 private:
  cuvsHnswAceParams_t params_ = nullptr;
};

class HnswIndexParams {
 public:
  HnswIndexParams() {
    cuvs_check(cuvsHnswIndexParamsCreate(&params_), "cuvsHnswIndexParamsCreate");
  }
  ~HnswIndexParams() {
    if (params_ != nullptr) {
      cuvsHnswIndexParamsDestroy(params_);
    }
  }
  cuvsHnswIndexParams_t get() const { return params_; }
  HnswIndexParams(const HnswIndexParams&) = delete;
  HnswIndexParams& operator=(const HnswIndexParams&) = delete;

 private:
  cuvsHnswIndexParams_t params_ = nullptr;
};

class HnswSearchParams {
 public:
  HnswSearchParams() {
    cuvs_check(cuvsHnswSearchParamsCreate(&params_), "cuvsHnswSearchParamsCreate");
  }
  ~HnswSearchParams() {
    if (params_ != nullptr) {
      cuvsHnswSearchParamsDestroy(params_);
    }
  }
  cuvsHnswSearchParams_t get() const { return params_; }
  HnswSearchParams(const HnswSearchParams&) = delete;
  HnswSearchParams& operator=(const HnswSearchParams&) = delete;

 private:
  cuvsHnswSearchParams_t params_ = nullptr;
};

class HnswIndex {
 public:
  HnswIndex() {
    cuvs_check(cuvsHnswIndexCreate(&index_), "cuvsHnswIndexCreate");
  }
  ~HnswIndex() {
    if (index_ != nullptr) {
      cuvsHnswIndexDestroy(index_);
    }
  }
  cuvsHnswIndex_t get() const { return index_; }
  HnswIndex(const HnswIndex&) = delete;
  HnswIndex& operator=(const HnswIndex&) = delete;

 private:
  cuvsHnswIndex_t index_ = nullptr;
};
#endif

cuvsFilter no_filter() {
  cuvsFilter filter;
  filter.type = NO_FILTER;
  filter.addr = static_cast<uintptr_t>(0);
  return filter;
}

List format_uint32_result(const std::vector<uint32_t>& labels,
                          const std::vector<float>& distances,
                          const int n_points,
                          const int search_k,
                          const int out_k,
                          const bool self_query,
                          const bool exclude_self,
                          const std::string& index_type,
                          const bool exact,
                          const bool already_sqrt = false) {
  IntegerMatrix indices(n_points, out_k);
  NumericMatrix dists(n_points, out_k);
  int* indices_ptr = indices.begin();
  double* dists_ptr = dists.begin();
  const bool skip_self = exclude_self && self_query;

  for (int i = 0; i < n_points; ++i) {
    const std::size_t row_offset = static_cast<std::size_t>(i) * search_k;
    int written = 0;
    for (int j = 0; j < search_k && written < out_k; ++j) {
      const std::size_t result_offset = row_offset + j;
      const uint32_t label = labels[result_offset];
      if (skip_self && label == static_cast<uint32_t>(i)) {
        continue;
      }
      const std::size_t output_offset = static_cast<std::size_t>(written) * n_points + i;
      indices_ptr[output_offset] = static_cast<int>(label) + 1;
      const float raw = distances[result_offset];
      dists_ptr[output_offset] = already_sqrt ? static_cast<double>(raw) :
        std::sqrt(std::max(static_cast<double>(raw), 0.0));
      ++written;
    }
    if (written < out_k) {
      Rcpp::stop("cuVS returned fewer neighbors than requested");
    }
  }

  return List::create(
    Rcpp::Named("indices") = indices,
    Rcpp::Named("distances") = dists,
    Rcpp::Named("index_type") = index_type,
    Rcpp::Named("exact") = exact
  );
}

List format_int64_result(const std::vector<int64_t>& labels,
                         const std::vector<float>& distances,
                         const int n_points,
                         const int search_k,
                         const int out_k,
                         const bool self_query,
                         const bool exclude_self,
                         const std::string& index_type,
                         const bool exact,
                         const bool already_sqrt = false) {
  IntegerMatrix indices(n_points, out_k);
  NumericMatrix dists(n_points, out_k);
  int* indices_ptr = indices.begin();
  double* dists_ptr = dists.begin();
  const bool skip_self = exclude_self && self_query;

  for (int i = 0; i < n_points; ++i) {
    const std::size_t row_offset = static_cast<std::size_t>(i) * search_k;
    int written = 0;
    for (int j = 0; j < search_k && written < out_k; ++j) {
      const std::size_t result_offset = row_offset + j;
      const int64_t label = labels[result_offset];
      if (label < 0) continue;
      if (skip_self && label == static_cast<int64_t>(i)) {
        continue;
      }
      if (label > std::numeric_limits<int>::max()) {
        Rcpp::stop("cuVS returned a neighbor index that does not fit in R integer");
      }
      const std::size_t output_offset = static_cast<std::size_t>(written) * n_points + i;
      indices_ptr[output_offset] = static_cast<int>(label) + 1;
      const float raw = distances[result_offset];
      dists_ptr[output_offset] = already_sqrt ? static_cast<double>(raw) :
        std::sqrt(std::max(static_cast<double>(raw), 0.0));
      ++written;
    }
    if (written < out_k) {
      Rcpp::stop("cuVS returned fewer neighbors than requested");
    }
  }

  return List::create(
    Rcpp::Named("indices") = indices,
    Rcpp::Named("distances") = dists,
    Rcpp::Named("index_type") = index_type,
    Rcpp::Named("exact") = exact
  );
}

List format_uint64_result(const std::vector<uint64_t>& labels,
                          const std::vector<float>& distances,
                          const int n_points,
                          const int search_k,
                          const int out_k,
                          const bool self_query,
                          const bool exclude_self,
                          const std::string& index_type,
                          const bool exact,
                          const bool already_sqrt = false) {
  IntegerMatrix indices(n_points, out_k);
  NumericMatrix dists(n_points, out_k);
  int* indices_ptr = indices.begin();
  double* dists_ptr = dists.begin();
  const bool skip_self = exclude_self && self_query;

  for (int i = 0; i < n_points; ++i) {
    const std::size_t row_offset = static_cast<std::size_t>(i) * search_k;
    int written = 0;
    for (int j = 0; j < search_k && written < out_k; ++j) {
      const std::size_t result_offset = row_offset + j;
      const uint64_t label = labels[result_offset];
      if (skip_self && label == static_cast<uint64_t>(i)) {
        continue;
      }
      if (label > static_cast<uint64_t>(std::numeric_limits<int>::max())) {
        Rcpp::stop("cuVS returned a neighbor index that does not fit in R integer");
      }
      const std::size_t output_offset = static_cast<std::size_t>(written) * n_points + i;
      indices_ptr[output_offset] = static_cast<int>(label) + 1;
      const float raw = distances[result_offset];
      dists_ptr[output_offset] = already_sqrt ? static_cast<double>(raw) :
        std::sqrt(std::max(static_cast<double>(raw), 0.0));
      ++written;
    }
    if (written < out_k) {
      Rcpp::stop("cuVS returned fewer neighbors than requested");
    }
  }

  return List::create(
    Rcpp::Named("indices") = indices,
    Rcpp::Named("distances") = dists,
    Rcpp::Named("index_type") = index_type,
    Rcpp::Named("exact") = exact
  );
}

std::string json_escape(const std::string& text) {
  std::ostringstream out;
  for (char ch : text) {
    switch (ch) {
      case '\\': out << "\\\\"; break;
      case '"': out << "\\\""; break;
      case '\n': out << "\\n"; break;
      case '\r': out << "\\r"; break;
      case '\t': out << "\\t"; break;
      default: out << ch; break;
    }
  }
  return out.str();
}

} // namespace

bool cuvs_is_available_impl() {
  int count = 0;
  return cudaGetDeviceCount(&count) == cudaSuccess && count > 0;
}

std::string cuvs_info_json_impl() {
  int count = 0;
  const cudaError_t status = cudaGetDeviceCount(&count);
  if (status != cudaSuccess) {
    return std::string("{\"available\":false,\"library\":\"cuvs\",\"reason\":\"") +
      json_escape(cudaGetErrorString(status)) + "\"}";
  }
  if (count < 1) {
    return "{\"available\":false,\"library\":\"cuvs\",\"reason\":\"no_cuda_device\"}";
  }

  std::ostringstream out;
  out << "{\"available\":true,\"library\":\"cuvs\",\"interface\":\"c_api\","
      << "\"device_count\":" << count
      << "}";
  return out.str();
}

List cuvs_bruteforce_knn_impl(NumericMatrix data,
                              NumericMatrix points,
                              int k,
                              bool exclude_self) {
  validate_inputs(data, points, k, exclude_self);
  const bool same_storage = same_matrix_storage(data, points);
  const bool self_query = exclude_self || same_storage;
  const int n_data = data.nrow();
  const int n_points = points.nrow();
  const int n_features = data.ncol();
  const int search_k = exclude_self ? std::min(n_data, k + 1) : k;

  std::vector<float> xb;
  std::vector<float> xq;
  copy_row_major_float(data, xb);
  if (!same_storage) {
    copy_row_major_float(points, xq);
  }

  CuvsResources res;
  const std::size_t data_bytes = xb.size() * sizeof(float);
  DeviceBuffer dataset_d(res.get(), data_bytes);
  cuda_check(
    cudaMemcpy(dataset_d.get(), xb.data(), data_bytes, cudaMemcpyHostToDevice),
    "cudaMemcpy(dataset)"
  );

  DeviceBuffer query_d;
  void* query_ptr = dataset_d.get();
  if (!same_storage) {
    const std::size_t query_bytes = xq.size() * sizeof(float);
    query_d.reset(res.get(), query_bytes);
    cuda_check(
      cudaMemcpy(query_d.get(), xq.data(), query_bytes, cudaMemcpyHostToDevice),
      "cudaMemcpy(queries)"
    );
    query_ptr = query_d.get();
  }

  std::vector<float>().swap(xb);
  if (!same_storage) {
    std::vector<float>().swap(xq);
  }

  DeviceBuffer neighbors_d(
    res.get(),
    static_cast<std::size_t>(n_points) * search_k * sizeof(int64_t)
  );
  DeviceBuffer distances_d(
    res.get(),
    static_cast<std::size_t>(n_points) * search_k * sizeof(float)
  );

  int64_t dataset_shape[2] = {n_data, n_features};
  int64_t query_shape[2] = {n_points, n_features};
  int64_t output_shape[2] = {n_points, search_k};
  DLManagedTensor dataset_tensor = make_tensor(
    dataset_d.get(), dataset_shape, 2, kDLCUDA, kDLFloat, 32
  );
  DLManagedTensor query_tensor = make_tensor(
    query_ptr, query_shape, 2, kDLCUDA, kDLFloat, 32
  );
  DLManagedTensor neighbors_tensor = make_tensor(
    neighbors_d.get(), output_shape, 2, kDLCUDA, kDLInt, 64
  );
  DLManagedTensor distances_tensor = make_tensor(
    distances_d.get(), output_shape, 2, kDLCUDA, kDLFloat, 32
  );

  BruteForceIndex index;
  cuvs_check(
    cuvsBruteForceBuild(res.get(), &dataset_tensor, L2Expanded, 0.0f, index.get()),
    "cuvsBruteForceBuild"
  );
  cuvs_check(
    cuvsBruteForceSearch(
      res.get(),
      index.get(),
      &query_tensor,
      &neighbors_tensor,
      &distances_tensor,
      no_filter()
    ),
    "cuvsBruteForceSearch"
  );
  cuda_sync("cudaDeviceSynchronize");

  std::vector<int64_t> labels(static_cast<std::size_t>(n_points) * search_k);
  std::vector<float> distances(static_cast<std::size_t>(n_points) * search_k);
  cuda_check(
    cudaMemcpy(labels.data(), neighbors_d.get(), labels.size() * sizeof(int64_t), cudaMemcpyDeviceToHost),
    "cudaMemcpy(neighbors)"
  );
  cuda_check(
    cudaMemcpy(distances.data(), distances_d.get(), distances.size() * sizeof(float), cudaMemcpyDeviceToHost),
    "cudaMemcpy(distances)"
  );

  return format_int64_result(
    labels,
    distances,
    n_points,
    search_k,
    k,
    self_query,
    exclude_self,
    "cuVS_BruteForce",
    true
  );
}

List cuvs_cagra_knn_impl(NumericMatrix data,
                         NumericMatrix points,
                         int k,
                         bool exclude_self,
                         int graph_degree,
                         int intermediate_graph_degree,
                         int search_width,
                         int itopk_size,
                         std::string build_algo) {
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

  graph_degree = std::max(search_k, graph_degree);
  graph_degree = std::min(graph_degree, std::max(1, n_data - 1));
  intermediate_graph_degree = std::max(
    intermediate_graph_degree,
    std::max(graph_degree, graph_degree * 2)
  );
  intermediate_graph_degree = std::min(
    intermediate_graph_degree,
    std::max(1, n_data - 1)
  );

  std::vector<float> xb;
  std::vector<float> xq;
  copy_row_major_float(data, xb);
  if (!same_storage) {
    copy_row_major_float(points, xq);
  }

  CuvsResources res;
  const std::size_t data_bytes = xb.size() * sizeof(float);
  DeviceBuffer dataset_d(res.get(), data_bytes);
  cuda_check(
    cudaMemcpy(dataset_d.get(), xb.data(), data_bytes, cudaMemcpyHostToDevice),
    "cudaMemcpy(dataset)"
  );

  DeviceBuffer query_d;
  void* query_ptr = dataset_d.get();
  if (!same_storage) {
    const std::size_t query_bytes = xq.size() * sizeof(float);
    query_d.reset(res.get(), query_bytes);
    cuda_check(
      cudaMemcpy(query_d.get(), xq.data(), query_bytes, cudaMemcpyHostToDevice),
      "cudaMemcpy(queries)"
    );
    query_ptr = query_d.get();
  }

  std::vector<float>().swap(xb);
  if (!same_storage) {
    std::vector<float>().swap(xq);
  }

  int64_t dataset_shape[2] = {n_data, n_features};
  int64_t query_shape[2] = {n_points, n_features};
  int64_t output_shape[2] = {n_points, search_k};
  DLManagedTensor dataset_tensor = make_tensor(
    dataset_d.get(), dataset_shape, 2, kDLCUDA, kDLFloat, 32
  );

  CagraIndexParams index_params;
  index_params.get()->metric = L2Expanded;
  index_params.get()->graph_degree = static_cast<std::size_t>(graph_degree);
  index_params.get()->intermediate_graph_degree =
    static_cast<std::size_t>(intermediate_graph_degree);
  std::string selected_build_algo = build_algo;
  std::transform(selected_build_algo.begin(), selected_build_algo.end(), selected_build_algo.begin(),
                 [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  if (selected_build_algo == "auto" || selected_build_algo == "auto_select") {
    index_params.get()->build_algo = AUTO_SELECT;
    selected_build_algo = "auto";
  } else if (selected_build_algo == "ivf_pq" || selected_build_algo == "ivfpq") {
    index_params.get()->build_algo = IVF_PQ;
    selected_build_algo = "ivf_pq";
  } else if (selected_build_algo == "nn_descent" || selected_build_algo == "nndescent") {
    index_params.get()->build_algo = NN_DESCENT;
    index_params.get()->nn_descent_niter =
      static_cast<std::size_t>(env_int("FAISSR_CUVS_CAGRA_NN_DESCENT_NITER", 20, 1));
    selected_build_algo = "nn_descent";
  } else if (selected_build_algo == "iterative" || selected_build_algo == "iterative_cagra_search") {
    index_params.get()->build_algo = ITERATIVE_CAGRA_SEARCH;
    selected_build_algo = "iterative_cagra_search";
  } else {
    Rcpp::stop("Unsupported cuVS CAGRA build algorithm: %s", build_algo);
  }

  CagraIndex index;
  cuvs_check(
    cuvsCagraBuild(res.get(), index_params.get(), &dataset_tensor, index.get()),
    "cuvsCagraBuild"
  );
  cuda_sync("cuvsCagraBuild synchronize");

  CagraSearchParams search_params;
  if (itopk_size > 0) {
    search_params.get()->itopk_size = static_cast<std::size_t>(itopk_size);
  }
  if (search_width > 0) {
    search_params.get()->search_width = static_cast<std::size_t>(search_width);
  }

  const int batch_size = std::min(
    n_points,
    env_int("FAISSR_CUVS_CAGRA_BATCH_SIZE", 8192, 1)
  );
  DeviceBuffer neighbors_d(
    res.get(),
    static_cast<std::size_t>(batch_size) * search_k * sizeof(int64_t)
  );
  DeviceBuffer distances_d(
    res.get(),
    static_cast<std::size_t>(batch_size) * search_k * sizeof(float)
  );

  std::vector<int64_t> labels(static_cast<std::size_t>(n_points) * search_k);
  std::vector<float> distances(static_cast<std::size_t>(n_points) * search_k);

  for (int offset = 0; offset < n_points; offset += batch_size) {
    const int current_batch = std::min(batch_size, n_points - offset);
    int64_t query_shape[2] = {current_batch, n_features};
    int64_t output_shape[2] = {current_batch, search_k};
    char* batch_query_ptr = static_cast<char*>(query_ptr) +
      static_cast<std::size_t>(offset) * n_features * sizeof(float);
    DLManagedTensor query_tensor = make_tensor(
      batch_query_ptr, query_shape, 2, kDLCUDA, kDLFloat, 32
    );
    DLManagedTensor neighbors_tensor = make_tensor(
      neighbors_d.get(), output_shape, 2, kDLCUDA, kDLInt, 64
    );
    DLManagedTensor distances_tensor = make_tensor(
      distances_d.get(), output_shape, 2, kDLCUDA, kDLFloat, 32
    );

    cuvs_check(
      cuvsCagraSearch(
        res.get(),
        search_params.get(),
        index.get(),
        &query_tensor,
        &neighbors_tensor,
        &distances_tensor,
        no_filter()
      ),
      "cuvsCagraSearch"
    );
    cuda_sync("cudaDeviceSynchronize");

    const std::size_t batch_values =
      static_cast<std::size_t>(current_batch) * search_k;
    const std::size_t out_offset =
      static_cast<std::size_t>(offset) * search_k;
    cuda_check(
      cudaMemcpy(
        labels.data() + out_offset,
        neighbors_d.get(),
        batch_values * sizeof(int64_t),
        cudaMemcpyDeviceToHost
      ),
      "cudaMemcpy(neighbors)"
    );
    cuda_check(
      cudaMemcpy(
        distances.data() + out_offset,
        distances_d.get(),
        batch_values * sizeof(float),
        cudaMemcpyDeviceToHost
      ),
      "cudaMemcpy(distances)"
    );
    cuda_sync("cuvsCagraSearch copy synchronize");
  }

  List out = format_int64_result(
    labels,
    distances,
    n_points,
    search_k,
    k,
    self_query,
    exclude_self,
    "cuVS_CAGRA",
    false
  );
  out["graph_degree"] = graph_degree;
  out["intermediate_graph_degree"] = intermediate_graph_degree;
  out["search_width"] = search_width;
  out["itopk_size"] = itopk_size;
  out["build_algo"] = selected_build_algo;
  out["nn_descent_niter"] = selected_build_algo == "nn_descent" ?
    static_cast<int>(index_params.get()->nn_descent_niter) : NA_INTEGER;
  out["requested_graph_degree"] = requested_graph_degree;
  out["requested_intermediate_graph_degree"] = requested_intermediate_graph_degree;
  out["requested_search_width"] = requested_search_width;
  out["requested_itopk_size"] = requested_itopk_size;
  out["cagra_parameters_adjusted"] = requested_graph_degree != graph_degree ||
    requested_intermediate_graph_degree != intermediate_graph_degree ||
    requested_search_width != search_width || requested_itopk_size != itopk_size;
  out["search_batch_size"] = batch_size;
  index.reset();
  cuda_sync("cuvsCagraIndexDestroy synchronize");
  neighbors_d.reset(res.get(), 0);
  distances_d.reset(res.get(), 0);
  query_d.reset(res.get(), 0);
  dataset_d.reset(res.get(), 0);
  cuda_sync("cuVS CAGRA device buffers release synchronize");
  return out;
}

List cuvs_hnsw_knn_impl(NumericMatrix data,
                        NumericMatrix points,
                        int k,
                        bool exclude_self,
                        int graph_degree,
                        int intermediate_graph_degree,
                        int ef,
                        int n_threads,
                        std::string cagra_build_algo) {
#ifndef FAISSR_HAS_CUVS_HNSW
  Rcpp::stop(
    "Direct cuVS HNSW is not available in this cuVS installation. "
    "Reinstall cuVS with cuvs/neighbors/hnsw.h."
  );
#else
  validate_inputs(data, points, k, exclude_self);
  const bool same_storage = same_matrix_storage(data, points);
  const bool self_query = exclude_self || same_storage;
  const int n_data = data.nrow();
  const int n_points = points.nrow();
  const int n_features = data.ncol();
  const int search_k = exclude_self ? std::min(n_data, k + 1) : k;
  const int requested_graph_degree = graph_degree;
  const int requested_intermediate_graph_degree = intermediate_graph_degree;
  const int requested_ef = ef;
  const int requested_n_threads = n_threads;

  graph_degree = std::max(search_k, graph_degree);
  graph_degree = std::min(graph_degree, std::max(1, n_data - 1));
  intermediate_graph_degree = std::max(
    intermediate_graph_degree,
    std::max(graph_degree, graph_degree * 2)
  );
  intermediate_graph_degree = std::min(
    intermediate_graph_degree,
    std::max(1, n_data - 1)
  );
  ef = std::max(search_k, ef);
  n_threads = std::max(1, n_threads);

  std::vector<float> xb;
  std::vector<float> xq;
  copy_row_major_float(data, xb);
  if (!same_storage) {
    copy_row_major_float(points, xq);
  }

  CuvsResources res;
  int64_t dataset_shape[2] = {n_data, n_features};
  DLManagedTensor dataset_tensor = make_tensor(
    xb.data(), dataset_shape, 2, kDLCPU, kDLFloat, 32
  );

  std::string selected_build_algo = cagra_build_algo;
  std::transform(selected_build_algo.begin(), selected_build_algo.end(), selected_build_algo.begin(),
                 [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  if (selected_build_algo == "auto" || selected_build_algo == "auto_select" ||
      selected_build_algo == "ivf_pq" || selected_build_algo == "ivfpq" ||
      selected_build_algo == "nn_descent" || selected_build_algo == "nndescent" ||
      selected_build_algo == "iterative" || selected_build_algo == "iterative_cagra_search") {
    selected_build_algo = "ace";
  } else {
    Rcpp::stop("Unsupported cuVS HNSW CAGRA build algorithm: %s", cagra_build_algo);
  }

  HnswAceParams ace_params;
  HnswIndexParams hnsw_params;
  hnsw_params.get()->hierarchy = GPU;
  hnsw_params.get()->ef_construction = std::max(ef, intermediate_graph_degree);
  hnsw_params.get()->num_threads = n_threads;
  hnsw_params.get()->M =
    static_cast<std::size_t>(std::max(2, graph_degree / 2));
  hnsw_params.get()->metric = L2Expanded;
  hnsw_params.get()->ace_params = ace_params.get();
  HnswIndex hnsw_index;
  cuvs_check(
    cuvsHnswBuild(res.get(), hnsw_params.get(), &dataset_tensor, hnsw_index.get()),
    "cuvsHnswBuild"
  );

  const float* query_ptr = same_storage ? xb.data() : xq.data();
  int64_t query_shape[2] = {n_points, n_features};
  int64_t output_shape[2] = {n_points, search_k};
  std::vector<uint64_t> labels(static_cast<std::size_t>(n_points) * search_k);
  std::vector<float> distances(static_cast<std::size_t>(n_points) * search_k);
  DLManagedTensor query_tensor = make_tensor(
    const_cast<float*>(query_ptr), query_shape, 2, kDLCPU, kDLFloat, 32
  );
  DLManagedTensor neighbors_tensor = make_tensor(
    labels.data(), output_shape, 2, kDLCPU, kDLUInt, 64
  );
  DLManagedTensor distances_tensor = make_tensor(
    distances.data(), output_shape, 2, kDLCPU, kDLFloat, 32
  );

  HnswSearchParams search_params;
  search_params.get()->ef = ef;
  search_params.get()->num_threads = n_threads;
  cuvs_check(
    cuvsHnswSearch(
      res.get(),
      search_params.get(),
      hnsw_index.get(),
      &query_tensor,
      &neighbors_tensor,
      &distances_tensor
    ),
    "cuvsHnswSearch"
  );

  List out = format_uint64_result(
    labels,
    distances,
    n_points,
    search_k,
    k,
    self_query,
    exclude_self,
    "cuVS_HNSW",
    false
  );
  out["graph_degree"] = graph_degree;
  out["intermediate_graph_degree"] = intermediate_graph_degree;
  out["ef"] = ef;
  out["num_threads"] = n_threads;
  out["requested_graph_degree"] = requested_graph_degree;
  out["requested_intermediate_graph_degree"] = requested_intermediate_graph_degree;
  out["requested_ef"] = requested_ef;
  out["requested_num_threads"] = requested_n_threads;
  out["cagra_build_algo"] = selected_build_algo;
  out["hnsw_build_algo"] = "ace";
  out["hnsw_hierarchy"] = "gpu";
  out["hnsw_m"] = static_cast<int>(hnsw_params.get()->M);
  out["hnsw_ef_construction"] = hnsw_params.get()->ef_construction;
  out["hnsw_parameters_adjusted"] = requested_graph_degree != graph_degree ||
    requested_intermediate_graph_degree != intermediate_graph_degree ||
    requested_ef != ef ||
    requested_n_threads != n_threads;
  return out;
#endif
}

List cuvs_ivf_flat_knn_impl(NumericMatrix data,
                            NumericMatrix points,
                            int k,
                            int n_lists,
                            int n_probes,
                            bool exclude_self) {
#ifndef FAISSR_HAS_CUVS_IVF_FLAT
  Rcpp::stop(
    "Direct cuVS IVF-Flat is not available in this cuVS installation. "
    "Use `backend = \"faiss_gpu_ivf_flat\"` or reinstall cuVS with "
    "cuvs/neighbors/ivf_flat.h."
  );
#else
  validate_inputs(data, points, k, exclude_self);
  const bool same_storage = same_matrix_storage(data, points);
  const bool self_query = exclude_self || same_storage;
  const int n_data = data.nrow();
  const int n_points = points.nrow();
  const int n_features = data.ncol();
  const int search_k = exclude_self ? std::min(n_data, k + 1) : k;
  n_lists = std::max(1, std::min(n_lists, n_data));
  n_probes = std::max(1, std::min(n_probes, n_lists));

  std::vector<float> xb;
  std::vector<float> xq;
  cuvs_debug("ivf_flat: copy input to row-major host buffers");
  copy_row_major_float(data, xb);
  if (!same_storage) {
    copy_row_major_float(points, xq);
  }

  CuvsResources res;
  const std::size_t data_bytes = xb.size() * sizeof(float);
  DeviceBuffer dataset_d(res.get(), data_bytes);
  cuvs_debug("ivf_flat: copy dataset to device");
  cuda_check(
    cudaMemcpy(dataset_d.get(), xb.data(), data_bytes, cudaMemcpyHostToDevice),
    "cudaMemcpy(dataset)"
  );

  DeviceBuffer query_d;
  void* query_ptr = dataset_d.get();
  if (!same_storage) {
    const std::size_t query_bytes = xq.size() * sizeof(float);
    query_d.reset(res.get(), query_bytes);
    cuvs_debug("ivf_flat: copy queries to separate device buffer");
    cuda_check(
      cudaMemcpy(query_d.get(), xq.data(), query_bytes, cudaMemcpyHostToDevice),
      "cudaMemcpy(queries)"
    );
    query_ptr = query_d.get();
  }

  std::vector<float>().swap(xb);
  if (!same_storage) {
    std::vector<float>().swap(xq);
  }

  int64_t dataset_shape[2] = {n_data, n_features};
  DLManagedTensor dataset_tensor = make_tensor(
    dataset_d.get(), dataset_shape, 2, kDLCUDA, kDLFloat, 32
  );

  IvfFlatIndexParams index_params;
  index_params.get()->metric = L2Expanded;
  index_params.get()->add_data_on_build = true;
  index_params.get()->n_lists = static_cast<uint32_t>(n_lists);
  const bool small_high_dim = n_data <= 5000 && n_features >= 4096;
  const int default_kmeans_iters = small_high_dim ? 1 : 20;
  const int kmeans_iters = env_int(
    "FAISSR_CUVS_IVF_FLAT_KMEANS_N_ITERS",
    default_kmeans_iters,
    1
  );
  const int train_percent = env_int(
    "FAISSR_CUVS_IVF_FLAT_TRAIN_PERCENT",
    100,
    1
  );
  index_params.get()->kmeans_n_iters = static_cast<uint32_t>(kmeans_iters);
  index_params.get()->kmeans_trainset_fraction = std::min(100, train_percent) / 100.0;
  index_params.get()->adaptive_centers = false;
  index_params.get()->conservative_memory_allocation = true;

  IvfFlatIndex index;
  cuvs_debug("ivf_flat: build index begin");
  cuvs_check(
    cuvsIvfFlatBuild(res.get(), index_params.get(), &dataset_tensor, index.get()),
    "cuvsIvfFlatBuild"
  );
  cuvs_debug("ivf_flat: build index done");
  IvfFlatSearchParams search_params;
  search_params.get()->n_probes = static_cast<uint32_t>(n_probes);

  const int batch_size = std::min(
    n_points,
    env_int("FAISSR_CUVS_IVF_BATCH_SIZE", 8192, 1)
  );
  DeviceBuffer neighbors_d(
    res.get(),
    static_cast<std::size_t>(batch_size) * search_k * sizeof(int64_t)
  );
  DeviceBuffer distances_d(
    res.get(),
    static_cast<std::size_t>(batch_size) * search_k * sizeof(float)
  );

  std::vector<int64_t> labels(static_cast<std::size_t>(n_points) * search_k);
  std::vector<float> distances(static_cast<std::size_t>(n_points) * search_k);

  for (int offset = 0; offset < n_points; offset += batch_size) {
    const int current_batch = std::min(batch_size, n_points - offset);
    int64_t query_shape[2] = {current_batch, n_features};
    int64_t output_shape[2] = {current_batch, search_k};
    char* batch_query_ptr = static_cast<char*>(query_ptr) +
      static_cast<std::size_t>(offset) * n_features * sizeof(float);
    DLManagedTensor query_tensor = make_tensor(
      batch_query_ptr, query_shape, 2, kDLCUDA, kDLFloat, 32
    );
    DLManagedTensor neighbors_tensor = make_tensor(
      neighbors_d.get(), output_shape, 2, kDLCUDA, kDLInt, 64
    );
    DLManagedTensor distances_tensor = make_tensor(
      distances_d.get(), output_shape, 2, kDLCUDA, kDLFloat, 32
    );

    if (offset == 0) cuvs_debug("ivf_flat: first search batch begin");
    cuvs_check(
      cuvsIvfFlatSearch(
        res.get(),
        search_params.get(),
        index.get(),
        &query_tensor,
        &neighbors_tensor,
        &distances_tensor,
        no_filter()
      ),
      "cuvsIvfFlatSearch"
    );
    cuda_sync("cudaDeviceSynchronize");
    if (offset == 0) cuvs_debug("ivf_flat: first search batch done");

    const std::size_t batch_values =
      static_cast<std::size_t>(current_batch) * search_k;
    const std::size_t out_offset =
      static_cast<std::size_t>(offset) * search_k;
    cuda_check(
      cudaMemcpy(
        labels.data() + out_offset,
        neighbors_d.get(),
        batch_values * sizeof(int64_t),
        cudaMemcpyDeviceToHost
      ),
      "cudaMemcpy(neighbors)"
    );
    cuda_check(
      cudaMemcpy(
        distances.data() + out_offset,
        distances_d.get(),
        batch_values * sizeof(float),
        cudaMemcpyDeviceToHost
      ),
      "cudaMemcpy(distances)"
    );
  }

  List out = format_int64_result(
    labels,
    distances,
    n_points,
    search_k,
    k,
    self_query,
    exclude_self,
    "cuVS_IVF_Flat",
    false
  );
  out["n_lists"] = n_lists;
  out["n_probes"] = n_probes;
  out["search_batch_size"] = batch_size;
  out["kmeans_n_iters"] = kmeans_iters;
  out["kmeans_trainset_fraction"] = std::min(100, train_percent) / 100.0;
  out["conservative_memory_allocation"] = true;
  return out;
#endif
}

List cuvs_ivf_pq_knn_impl(NumericMatrix data,
                          NumericMatrix points,
                          int k,
                          int n_lists,
                          int n_probes,
                          int pq_dim,
                          int pq_bits,
                          bool exclude_self) {
#ifndef FAISSR_HAS_CUVS_IVF_PQ
  Rcpp::stop(
    "Direct cuVS IVF-PQ is not available in this cuVS installation. "
    "Use `backend = \"faiss_gpu_ivfpq\"` or reinstall cuVS with "
    "cuvs/neighbors/ivf_pq.h."
  );
#else
  validate_inputs(data, points, k, exclude_self);
  const bool same_storage = same_matrix_storage(data, points);
  const bool self_query = exclude_self || same_storage;
  const int n_data = data.nrow();
  const int n_points = points.nrow();
  const int n_features = data.ncol();
  const int search_k = exclude_self ? std::min(n_data, k + 1) : k;
  const int requested_pq_dim = pq_dim;
  const int requested_pq_bits = pq_bits;
  n_lists = std::max(1, std::min(n_lists, n_data));
  n_probes = std::max(1, std::min(n_probes, n_lists));
  pq_dim = std::max(0, pq_dim);
  pq_bits = std::max(4, std::min(8, pq_bits));

  std::vector<float> xb;
  std::vector<float> xq;
  copy_row_major_float(data, xb);
  if (!same_storage) {
    copy_row_major_float(points, xq);
  }

  CuvsResources res;
  const std::size_t data_bytes = xb.size() * sizeof(float);
  DeviceBuffer dataset_d(res.get(), data_bytes);
  cuda_check(
    cudaMemcpy(dataset_d.get(), xb.data(), data_bytes, cudaMemcpyHostToDevice),
    "cudaMemcpy(dataset)"
  );

  DeviceBuffer query_d;
  void* query_ptr = dataset_d.get();
  if (!same_storage) {
    const std::size_t query_bytes = xq.size() * sizeof(float);
    query_d.reset(res.get(), query_bytes);
    cuda_check(
      cudaMemcpy(query_d.get(), xq.data(), query_bytes, cudaMemcpyHostToDevice),
      "cudaMemcpy(queries)"
    );
    query_ptr = query_d.get();
  }

  std::vector<float>().swap(xb);
  if (!same_storage) {
    std::vector<float>().swap(xq);
  }

  int64_t dataset_shape[2] = {n_data, n_features};
  DLManagedTensor dataset_tensor = make_tensor(
    dataset_d.get(), dataset_shape, 2, kDLCUDA, kDLFloat, 32
  );

  IvfPqIndexParams index_params;
  index_params.get()->metric = L2Expanded;
  index_params.get()->add_data_on_build = true;
  index_params.get()->n_lists = static_cast<uint32_t>(n_lists);
  index_params.get()->pq_bits = static_cast<uint32_t>(pq_bits);
  index_params.get()->pq_dim = static_cast<uint32_t>(pq_dim);
  index_params.get()->conservative_memory_allocation = true;

  IvfPqIndex index;
  cuvs_check(
    cuvsIvfPqBuild(res.get(), index_params.get(), &dataset_tensor, index.get()),
    "cuvsIvfPqBuild"
  );

  IvfPqSearchParams search_params;
  search_params.get()->n_probes = static_cast<uint32_t>(n_probes);

  const int batch_size = std::min(
    n_points,
    env_int("FAISSR_CUVS_IVF_BATCH_SIZE", 8192, 1)
  );
  DeviceBuffer neighbors_d(
    res.get(),
    static_cast<std::size_t>(batch_size) * search_k * sizeof(int64_t)
  );
  DeviceBuffer distances_d(
    res.get(),
    static_cast<std::size_t>(batch_size) * search_k * sizeof(float)
  );

  std::vector<int64_t> labels(static_cast<std::size_t>(n_points) * search_k);
  std::vector<float> distances(static_cast<std::size_t>(n_points) * search_k);

  for (int offset = 0; offset < n_points; offset += batch_size) {
    const int current_batch = std::min(batch_size, n_points - offset);
    int64_t query_shape[2] = {current_batch, n_features};
    int64_t output_shape[2] = {current_batch, search_k};
    char* batch_query_ptr = static_cast<char*>(query_ptr) +
      static_cast<std::size_t>(offset) * n_features * sizeof(float);
    DLManagedTensor query_tensor = make_tensor(
      batch_query_ptr, query_shape, 2, kDLCUDA, kDLFloat, 32
    );
    DLManagedTensor neighbors_tensor = make_tensor(
      neighbors_d.get(), output_shape, 2, kDLCUDA, kDLInt, 64
    );
    DLManagedTensor distances_tensor = make_tensor(
      distances_d.get(), output_shape, 2, kDLCUDA, kDLFloat, 32
    );

    cuvs_check(
      cuvsIvfPqSearch(
        res.get(),
        search_params.get(),
        index.get(),
        &query_tensor,
        &neighbors_tensor,
        &distances_tensor
      ),
      "cuvsIvfPqSearch"
    );
    cuda_sync("cudaDeviceSynchronize");

    const std::size_t batch_values =
      static_cast<std::size_t>(current_batch) * search_k;
    const std::size_t out_offset =
      static_cast<std::size_t>(offset) * search_k;
    cuda_check(
      cudaMemcpy(
        labels.data() + out_offset,
        neighbors_d.get(),
        batch_values * sizeof(int64_t),
        cudaMemcpyDeviceToHost
      ),
      "cudaMemcpy(neighbors)"
    );
    cuda_check(
      cudaMemcpy(
        distances.data() + out_offset,
        distances_d.get(),
        batch_values * sizeof(float),
        cudaMemcpyDeviceToHost
      ),
      "cudaMemcpy(distances)"
    );
  }

  List out = format_int64_result(
    labels,
    distances,
    n_points,
    search_k,
    k,
    self_query,
    exclude_self,
    "cuVS_IVF_PQ",
    false
  );
  int64_t actual_pq_dim = pq_dim;
  int64_t actual_pq_bits = pq_bits;
  cuvsIvfPqIndexGetPqDim(index.get(), &actual_pq_dim);
  cuvsIvfPqIndexGetPqBits(index.get(), &actual_pq_bits);
  out["n_lists"] = n_lists;
  out["n_probes"] = n_probes;
  out["pq_dim"] = static_cast<int>(actual_pq_dim);
  out["pq_bits"] = static_cast<int>(actual_pq_bits);
  out["requested_pq_dim"] = requested_pq_dim;
  out["requested_pq_bits"] = requested_pq_bits;
  out["pq_parameters_adjusted"] = requested_pq_dim != static_cast<int>(actual_pq_dim) ||
    requested_pq_bits != static_cast<int>(actual_pq_bits);
  out["search_batch_size"] = batch_size;
  return out;
#endif
}

List cuvs_kmeans_impl(NumericMatrix data,
                      int centers,
                      int max_iter,
                      int n_init,
                      double tol,
                      int64_t streaming_batch_size,
                      bool kmeans_plus_plus) {
#ifndef FAISSR_HAS_CUVS_KMEANS
  Rcpp::stop(
    "cuVS k-means is not available in this cuVS installation. Reinstall cuVS "
    "with cuvs/cluster/kmeans.h."
  );
#else
  if (data.nrow() < 1 || data.ncol() < 1) {
    Rcpp::stop("data must have at least one row and one column");
  }
  const int n = data.nrow();
  const int p = data.ncol();
  if (centers < 1 || centers > n) {
    Rcpp::stop("centers must be in [1, nrow(data)]");
  }
  max_iter = std::max(1, max_iter);
  n_init = std::max(1, n_init);
  if (!std::isfinite(tol) || tol < 0.0) tol = 1e-4;
  if (streaming_batch_size < 0) streaming_batch_size = 0;

  std::vector<float> xb;
  copy_row_major_float(data, xb);

  CuvsResources res;
  KMeansParams params;
  params.get()->metric = L2Expanded;
  params.get()->n_clusters = centers;
  params.get()->init = kmeans_plus_plus ? KMeansPlusPlus : Random;
  params.get()->max_iter = max_iter;
  params.get()->tol = tol;
  params.get()->n_init = n_init;
  params.get()->streaming_batch_size = streaming_batch_size;

  int64_t x_shape[2] = {n, p};
  DeviceBuffer x_d(
    res.get(),
    xb.size() * sizeof(float)
  );
  cuda_check(
    cudaMemcpy(
      x_d.get(),
      xb.data(),
      xb.size() * sizeof(float),
      cudaMemcpyHostToDevice
    ),
    "cudaMemcpy(kmeans data)"
  );
  DLManagedTensor x_device_tensor = make_tensor(
    x_d.get(), x_shape, 2, kDLCUDA, kDLFloat, 32
  );

  DeviceBuffer centroids_d(
    res.get(),
    static_cast<std::size_t>(centers) * p * sizeof(float)
  );
  cuda_check(
    cudaMemset(centroids_d.get(), 0, static_cast<std::size_t>(centers) * p * sizeof(float)),
    "cudaMemset(centroids)"
  );
  int64_t centroids_shape[2] = {centers, p};
  DLManagedTensor centroids_tensor = make_tensor(
    centroids_d.get(), centroids_shape, 2, kDLCUDA, kDLFloat, 32
  );

  double inertia = 0.0;
  int n_iter = 0;
  cuvs_check(
    cuvsKMeansFit(
      res.get(),
      params.get(),
      &x_device_tensor,
      nullptr,
      &centroids_tensor,
      &inertia,
      &n_iter
    ),
    "cuvsKMeansFit"
  );
  cuda_sync("cuvsKMeansFit");

  std::vector<float> centroids(
    static_cast<std::size_t>(centers) * p,
    0.0f
  );
  cuda_check(
    cudaMemcpy(
      centroids.data(),
      centroids_d.get(),
      centroids.size() * sizeof(float),
      cudaMemcpyDeviceToHost
    ),
    "cudaMemcpy(centroids)"
  );

  DeviceBuffer labels_d(res.get(), static_cast<std::size_t>(n) * sizeof(int));
  int64_t labels_shape[1] = {n};
  DLManagedTensor labels_tensor = make_tensor(
    labels_d.get(), labels_shape, 1, kDLCUDA, kDLInt, 32
  );
  double predict_inertia = 0.0;
  cuvs_check(
    cuvsKMeansPredict(
      res.get(),
      params.get(),
      &x_device_tensor,
      nullptr,
      &centroids_tensor,
      &labels_tensor,
      false,
      &predict_inertia
    ),
    "cuvsKMeansPredict"
  );
  cuda_sync("cuvsKMeansPredict");

  std::vector<int> labels(static_cast<std::size_t>(n), 0);
  cuda_check(
    cudaMemcpy(
      labels.data(),
      labels_d.get(),
      labels.size() * sizeof(int),
      cudaMemcpyDeviceToHost
    ),
    "cudaMemcpy(labels)"
  );

  NumericMatrix center_matrix(centers, p);
  for (int c = 0; c < p; ++c) {
    for (int r = 0; r < centers; ++r) {
      center_matrix(r, c) = centroids[static_cast<std::size_t>(r) * p + c];
    }
  }

  Rcpp::IntegerVector cluster(n);
  Rcpp::IntegerVector size(centers);
  Rcpp::NumericVector withinss(centers);
  double total = 0.0;
  for (int i = 0; i < n; ++i) {
    const int label = labels[static_cast<std::size_t>(i)];
    if (label < 0 || label >= centers) {
      Rcpp::stop("cuVS k-means returned an invalid cluster label");
    }
    cluster[i] = label + 1;
    size[label] += 1;
    double dist = 0.0;
    const std::size_t row_offset = static_cast<std::size_t>(i) * p;
    const std::size_t center_offset = static_cast<std::size_t>(label) * p;
    for (int j = 0; j < p; ++j) {
      const double diff = static_cast<double>(xb[row_offset + j]) -
        static_cast<double>(centroids[center_offset + j]);
      dist += diff * diff;
    }
    withinss[label] += dist;
    total += dist;
  }

  return List::create(
    Rcpp::Named("cluster") = cluster,
    Rcpp::Named("centers") = center_matrix,
    Rcpp::Named("withinss") = withinss,
    Rcpp::Named("tot.withinss") = std::isfinite(predict_inertia) ? predict_inertia : total,
    Rcpp::Named("size") = size,
    Rcpp::Named("iter") = n_iter,
    Rcpp::Named("backend_library") = "cuvs",
    Rcpp::Named("parameters") = List::create(
      Rcpp::Named("centers") = centers,
      Rcpp::Named("max_iter") = max_iter,
      Rcpp::Named("n_init") = n_init,
      Rcpp::Named("tol") = tol,
      Rcpp::Named("streaming_batch_size") = static_cast<double>(streaming_batch_size)
    )
  );
#endif
}

List cuvs_nndescent_self_knn_impl(NumericMatrix data,
                                  int k,
                                  int graph_degree,
                                  int intermediate_graph_degree,
                                  int max_iterations) {
  if (data.nrow() < 2) Rcpp::stop("data must have at least two rows");
  if (data.ncol() < 1) Rcpp::stop("data must have at least one column");
  if (k < 1 || k >= data.nrow()) {
    Rcpp::stop("k must be in [1, nrow(data) - 1]");
  }

  const int n_data = data.nrow();
  const int n_features = data.ncol();
  graph_degree = std::max(k, graph_degree);
  graph_degree = std::min(graph_degree, n_data - 1);
  intermediate_graph_degree = std::max(
    intermediate_graph_degree,
    std::max(graph_degree, graph_degree * 2)
  );
  intermediate_graph_degree = std::min(intermediate_graph_degree, n_data - 1);
  max_iterations = std::max(1, max_iterations);

  std::vector<float> xb;
  copy_row_major_float(data, xb);
  std::vector<uint32_t> graph(
    static_cast<std::size_t>(n_data) * graph_degree,
    0
  );
  std::vector<float> distances(
    static_cast<std::size_t>(n_data) * graph_degree,
    0.0f
  );

  CuvsResources res;
  int64_t dataset_shape[2] = {n_data, n_features};
  int64_t graph_shape[2] = {n_data, graph_degree};
  DLManagedTensor dataset_tensor = make_tensor(
    xb.data(), dataset_shape, 2, kDLCPU, kDLFloat, 32
  );
  DLManagedTensor graph_tensor = make_tensor(
    graph.data(), graph_shape, 2, kDLCPU, kDLUInt, 32
  );
  DLManagedTensor distances_tensor = make_tensor(
    distances.data(), graph_shape, 2, kDLCPU, kDLFloat, 32
  );

  NNDescentParams params;
  params.get()->metric = L2Expanded;
  params.get()->graph_degree = static_cast<std::size_t>(graph_degree);
  params.get()->intermediate_graph_degree =
    static_cast<std::size_t>(intermediate_graph_degree);
  params.get()->max_iterations = static_cast<std::size_t>(max_iterations);
  params.get()->return_distances = true;

  NNDescentIndex index;
  cuvs_check(
    cuvsNNDescentBuild(
      res.get(),
      params.get(),
      &dataset_tensor,
      nullptr,
      index.get()
    ),
    "cuvsNNDescentBuild"
  );
  cuvs_check(
    cuvsNNDescentIndexGetGraph(res.get(), index.get(), &graph_tensor),
    "cuvsNNDescentIndexGetGraph"
  );
  cuvs_check(
    cuvsNNDescentIndexGetDistances(res.get(), index.get(), &distances_tensor),
    "cuvsNNDescentIndexGetDistances"
  );
  cuda_sync("cudaDeviceSynchronize");

  List out = format_uint32_result(
    graph,
    distances,
    n_data,
    graph_degree,
    k,
    true,
    true,
    "cuVS_NNDescent",
    false
  );
  out["graph_degree"] = graph_degree;
  out["intermediate_graph_degree"] = intermediate_graph_degree;
  out["max_iterations"] = max_iterations;
  return out;
}
