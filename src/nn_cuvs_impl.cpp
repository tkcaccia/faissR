#include <Rcpp.h>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstring>
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
  std::string message(context);
  message += " failed";
  if (detail != nullptr && detail[0] != '\0') {
    message += ": ";
    message += detail;
  } else {
    message += ".";
  }
  std::string lower = message;
  std::transform(lower.begin(), lower.end(), lower.begin(),
                 [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); });
  if (std::string(context) == "cuvsNNDescentBuild" &&
      lower.find("invalid") != std::string::npos &&
      (lower.find("cudaerrorinvalidvalue") != std::string::npos ||
       lower.find("invalid argument") != std::string::npos)) {
    message +=
      "\n\nfaissR note: this matches a known RAPIDS cuVS NN-descent "
      "launch failure seen on high-dimensional FP32 L2 inputs. The cuVS "
      "L2-norm kernel can require more than CUDA's default dynamic shared "
      "memory per block and must opt in with "
      "cudaFuncSetAttribute(cudaFuncAttributeMaxDynamicSharedMemorySize). "
      "Update cuVS to a patched release or rebuild cuVS with that fix. "
      "faissR does not silently fall back to CPU or another algorithm for "
      "an explicit CUDA/cuVS NN-descent request.";
  }
  Rcpp::stop("%s", message.c_str());
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

struct MatrixViewF32 {
  const float* data = nullptr;
  int nrow = 0;
  int ncol = 0;
  bool row_major = true;
  bool owns_data = false;
  std::string layout = "unknown";
  std::vector<float> buffer;
};

std::size_t nndescent_l2_norm_shared_bytes(const int n_features,
                                           const std::size_t element_size) {
  if (n_features <= 0) return 0;
  const std::size_t rounded =
    ((static_cast<std::size_t>(n_features) + 31u) / 32u) * 32u;
  return rounded * element_size;
}

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
    Rcpp::stop("%s dimensions exceed cuVS int limits", name);
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
    Rcpp::stop("cuVS float32 input requires finite values");
  }
  if (view.data == nullptr) {
    view.data = view.buffer.data();
  }
  return view;
}

bool same_float32_object(SEXP data, SEXP points) {
  return data == points;
}

bool wants_float_distance_storage(const std::string& distance_storage) {
  if (distance_storage == "float" || distance_storage == "float32") return true;
  if (distance_storage == "double") return false;
  Rcpp::stop("`distance_storage` must be \"double\" or \"float\"");
  return false;
}

SEXP make_float32_distance_matrix(IntegerMatrix& float_dists) {
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
  return float_matrix;
}

void set_distance_value(double* dists_ptr,
                        int* float_dists_ptr,
                        const std::size_t output_offset,
                        const float raw,
                        const bool already_sqrt,
                        const bool float_distances) {
  const double value = already_sqrt ? static_cast<double>(raw) :
    std::sqrt(std::max(static_cast<double>(raw), 0.0));
  if (float_distances) {
    const float value_f = static_cast<float>(value);
    std::memcpy(float_dists_ptr + output_offset, &value_f, sizeof(float));
  } else {
    dists_ptr[output_offset] = value;
  }
}

void tag_float_distance_output(List& out, const bool float_distances) {
  if (float_distances) {
    out["distance_type"] = "float32";
    out.attr("distance_type") = "float32";
  }
}

void annotate_float32_input(List& out,
                            const MatrixViewF32& xb,
                            const MatrixViewF32& xq,
                            const bool same_storage) {
  out["input_type"] = "float32";
  out["input_layout"] = same_storage ?
    xb.layout :
    ("data=" + xb.layout + ";points=" + xq.layout);
  out["input_owns_data"] = same_storage ?
    xb.owns_data :
    (xb.owns_data || xq.owns_data);
  out["float32_compatibility_conversion"] = false;
}

double float32_bytes(const int rows, const int cols) {
  return static_cast<double>(rows) * static_cast<double>(cols) *
    static_cast<double>(sizeof(float));
}

void annotate_cuvs_gpu_residency(List& out,
                                 const std::string& index_residency,
                                 const bool same_storage,
                                 const int n_data,
                                 const int n_points,
                                 const int n_features,
                                 const int search_k,
                                 const bool query_on_device,
                                 const bool result_on_device,
                                 const int data_h2d_copies = 1,
                                 const bool gpu_index_resident = true) {
  const int query_h2d_copies = query_on_device && !same_storage ? 1 : 0;
  const double data_bytes = data_h2d_copies > 0 ?
    float32_bytes(n_data, n_features) : 0.0;
  const double query_bytes = query_h2d_copies > 0 ?
    float32_bytes(n_points, n_features) : 0.0;
  const double result_bytes = result_on_device ?
    static_cast<double>(n_points) * static_cast<double>(search_k) *
      static_cast<double>(sizeof(int64_t) + sizeof(float)) :
    0.0;
  out["accelerator"] = "cuda";
  out["gpu_provider"] = "cuvs";
  out["device_residency"] = "cuda";
  out["index_residency"] = index_residency;
  out["gpu_index_resident"] = gpu_index_resident;
  out["gpu_index_persistent"] = false;
  out["host_to_device_transfer_strategy"] = "explicit_cuvs_device_buffers";
  out["host_to_device_copies_known"] = true;
  out["host_to_device_data_copies"] = data_h2d_copies;
  out["host_to_device_query_copies"] = query_h2d_copies;
  out["host_to_device_copies"] = data_h2d_copies + query_h2d_copies;
  out["host_to_device_data_bytes"] = data_bytes;
  out["host_to_device_query_bytes"] = query_bytes;
  out["host_to_device_bytes"] = data_bytes + query_bytes;
  out["query_reuses_device_data"] = query_on_device && same_storage;
  out["query_residency"] = query_on_device ?
    (same_storage ? "dataset_device_buffer" : "query_device_buffer") :
    (same_storage ? "input_host_buffer" : "query_host_buffer");
  out["result_residency"] = result_on_device ?
    "device_result_buffer_then_R" : "host_result_tensor";
  out["device_to_host_result_copies_known"] = true;
  out["device_to_host_result_copies"] = result_on_device ? 2 : 0;
  out["device_to_host_result_bytes"] = result_bytes;
  out["cpu_fallback"] = false;
  out["cpu_side_result_repair"] = false;
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
                          const bool already_sqrt = false,
                          const bool float_distances = false) {
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
      set_distance_value(
        dists_ptr,
        float_dists_ptr,
        output_offset,
        raw,
        already_sqrt,
        float_distances
      );
      ++written;
    }
    if (written < out_k) {
      Rcpp::stop("cuVS returned fewer neighbors than requested");
    }
  }

  SEXP distance_sexp = float_distances ?
    make_float32_distance_matrix(float_dists) :
    static_cast<SEXP>(dists);
  List out = List::create(
    Rcpp::Named("indices") = indices,
    Rcpp::Named("distances") = distance_sexp,
    Rcpp::Named("index_type") = index_type,
    Rcpp::Named("exact") = exact
  );
  tag_float_distance_output(out, float_distances);
  return out;
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
                         const bool already_sqrt = false,
                         const bool float_distances = false) {
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
      set_distance_value(
        dists_ptr,
        float_dists_ptr,
        output_offset,
        raw,
        already_sqrt,
        float_distances
      );
      ++written;
    }
    if (written < out_k) {
      Rcpp::stop("cuVS returned fewer neighbors than requested");
    }
  }

  SEXP distance_sexp = float_distances ?
    make_float32_distance_matrix(float_dists) :
    static_cast<SEXP>(dists);
  List out = List::create(
    Rcpp::Named("indices") = indices,
    Rcpp::Named("distances") = distance_sexp,
    Rcpp::Named("index_type") = index_type,
    Rcpp::Named("exact") = exact
  );
  tag_float_distance_output(out, float_distances);
  return out;
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
                          const bool already_sqrt = false,
                          const bool float_distances = false) {
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
      set_distance_value(
        dists_ptr,
        float_dists_ptr,
        output_offset,
        raw,
        already_sqrt,
        float_distances
      );
      ++written;
    }
    if (written < out_k) {
      Rcpp::stop("cuVS returned fewer neighbors than requested");
    }
  }

  SEXP distance_sexp = float_distances ?
    make_float32_distance_matrix(float_dists) :
    static_cast<SEXP>(dists);
  List out = List::create(
    Rcpp::Named("indices") = indices,
    Rcpp::Named("distances") = distance_sexp,
    Rcpp::Named("index_type") = index_type,
    Rcpp::Named("exact") = exact
  );
  tag_float_distance_output(out, float_distances);
  return out;
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

List cuvs_bruteforce_float32_knn_impl(SEXP data,
                                      SEXP points,
                                      int k,
                                      bool exclude_self,
                                      std::string distance_storage) {
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
  const bool float_distances = wants_float_distance_storage(distance_storage);

  CuvsResources res;
  const std::size_t data_bytes =
    static_cast<std::size_t>(n_data) * n_features * sizeof(float);
  DeviceBuffer dataset_d(res.get(), data_bytes);
  cuda_check(
    cudaMemcpy(dataset_d.get(), xb.data, data_bytes, cudaMemcpyHostToDevice),
    "cudaMemcpy(dataset)"
  );

  DeviceBuffer query_d;
  void* query_ptr = dataset_d.get();
  if (!same_storage) {
    const std::size_t query_bytes =
      static_cast<std::size_t>(n_points) * n_features * sizeof(float);
    query_d.reset(res.get(), query_bytes);
    cuda_check(
      cudaMemcpy(query_d.get(), xq.data, query_bytes, cudaMemcpyHostToDevice),
      "cudaMemcpy(queries)"
    );
    query_ptr = query_d.get();
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

  List out = format_int64_result(
    labels,
    distances,
    n_points,
    search_k,
    k,
    self_query,
    exclude_self,
    "cuVS_BruteForce",
    true,
    false,
    float_distances
  );
  annotate_float32_input(out, xb, xq, same_storage);
  annotate_cuvs_gpu_residency(
    out, "gpu_transient", same_storage, n_data, n_points, n_features,
    search_k, true, true
  );
  return out;
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

List cuvs_cagra_float32_knn_impl(SEXP data,
                                 SEXP points,
                                 int k,
                                 bool exclude_self,
                                 int graph_degree,
                                 int intermediate_graph_degree,
                                 int search_width,
                                 int itopk_size,
                                 std::string build_algo,
                                 std::string distance_storage) {
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
  const bool float_distances = wants_float_distance_storage(distance_storage);

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

  CuvsResources res;
  const std::size_t data_bytes =
    static_cast<std::size_t>(n_data) * n_features * sizeof(float);
  DeviceBuffer dataset_d(res.get(), data_bytes);
  cuda_check(
    cudaMemcpy(dataset_d.get(), xb.data, data_bytes, cudaMemcpyHostToDevice),
    "cudaMemcpy(dataset)"
  );

  DeviceBuffer query_d;
  void* query_ptr = dataset_d.get();
  if (!same_storage) {
    const std::size_t query_bytes =
      static_cast<std::size_t>(n_points) * n_features * sizeof(float);
    query_d.reset(res.get(), query_bytes);
    cuda_check(
      cudaMemcpy(query_d.get(), xq.data, query_bytes, cudaMemcpyHostToDevice),
      "cudaMemcpy(queries)"
    );
    query_ptr = query_d.get();
  }

  int64_t dataset_shape[2] = {n_data, n_features};
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
    false,
    false,
    float_distances
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
  annotate_float32_input(out, xb, xq, same_storage);
  annotate_cuvs_gpu_residency(
    out, "gpu_transient", same_storage, n_data, n_points, n_features,
    search_k, true, true
  );
  index.reset();
  cuda_sync("cuvsCagraIndexDestroy synchronize");
  neighbors_d.reset(res.get(), 0);
  distances_d.reset(res.get(), 0);
  query_d.reset(res.get(), 0);
  dataset_d.reset(res.get(), 0);
  cuda_sync("cuVS CAGRA device buffers release synchronize");
  return out;
}

List cuvs_hnsw_knn_from_row_major(const float* xb,
                                  const float* xq,
                                  const bool same_storage,
                                  const int n_data,
                                  const int n_points,
                                  const int n_features,
                                  const int k,
                                  const bool exclude_self,
                                  int graph_degree,
                                  int intermediate_graph_degree,
                                  int ef,
                                  int n_threads,
                                  std::string cagra_build_algo,
                                  const bool float_distances = false) {
#ifndef FAISSR_HAS_CUVS_HNSW
  (void)float_distances;
  Rcpp::stop(
    "Direct cuVS HNSW is not available in this cuVS installation. "
    "Reinstall cuVS with cuvs/neighbors/hnsw.h."
  );
#else
  const bool self_query = exclude_self || same_storage;
  const int search_k = exclude_self ? std::min(n_data, k + 1) : k;
  const int requested_graph_degree = graph_degree;
  const int requested_intermediate_graph_degree = intermediate_graph_degree;
  const int requested_ef = ef;
  const int requested_n_threads = n_threads;

  graph_degree = std::max(2, graph_degree);
  graph_degree = std::min(graph_degree, std::max(1, n_data - 1));
  intermediate_graph_degree = std::max(
    intermediate_graph_degree,
    graph_degree
  );
  intermediate_graph_degree = std::min(
    intermediate_graph_degree,
    std::max(1, n_data - 1)
  );
  ef = std::max(search_k, ef);
  n_threads = std::max(1, n_threads);

  if (xb == nullptr || (!same_storage && xq == nullptr)) {
    Rcpp::stop("cuVS HNSW received a null float32 input pointer");
  }

  CuvsResources res;
  const std::size_t data_bytes =
    static_cast<std::size_t>(n_data) * n_features * sizeof(float);
  DeviceBuffer dataset_d(res.get(), data_bytes);
  cuda_check(
    cudaMemcpy(dataset_d.get(), xb, data_bytes, cudaMemcpyHostToDevice),
    "cudaMemcpy(dataset)"
  );

  int64_t dataset_shape[2] = {n_data, n_features};
  DLManagedTensor dataset_tensor = make_tensor(
    dataset_d.get(), dataset_shape, 2, kDLCUDA, kDLFloat, 32
  );

  CagraIndexParams cagra_params;
  cagra_params.get()->metric = L2Expanded;
  cagra_params.get()->graph_degree = static_cast<std::size_t>(graph_degree);
  cagra_params.get()->intermediate_graph_degree =
    static_cast<std::size_t>(intermediate_graph_degree);
  std::string selected_build_algo = cagra_build_algo;
  std::transform(selected_build_algo.begin(), selected_build_algo.end(), selected_build_algo.begin(),
                 [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  if (selected_build_algo == "auto" || selected_build_algo == "auto_select") {
    cagra_params.get()->build_algo = AUTO_SELECT;
    selected_build_algo = "auto";
  } else if (selected_build_algo == "ivf_pq" || selected_build_algo == "ivfpq") {
    cagra_params.get()->build_algo = IVF_PQ;
    selected_build_algo = "ivf_pq";
  } else if (selected_build_algo == "nn_descent" || selected_build_algo == "nndescent") {
    cagra_params.get()->build_algo = NN_DESCENT;
    cagra_params.get()->nn_descent_niter =
      static_cast<std::size_t>(env_int("FAISSR_CUVS_CAGRA_NN_DESCENT_NITER", 20, 1));
    selected_build_algo = "nn_descent";
  } else if (selected_build_algo == "iterative" || selected_build_algo == "iterative_cagra_search") {
    cagra_params.get()->build_algo = ITERATIVE_CAGRA_SEARCH;
    selected_build_algo = "iterative_cagra_search";
  } else {
    Rcpp::stop("Unsupported cuVS HNSW CAGRA build algorithm: %s", cagra_build_algo);
  }

  CagraIndex cagra_index;
  cuvs_check(
    cuvsCagraBuild(res.get(), cagra_params.get(), &dataset_tensor, cagra_index.get()),
    "cuvsCagraBuild"
  );

  int64_t host_dataset_shape[2] = {n_data, n_features};
  DLManagedTensor host_dataset_tensor = make_tensor(
    const_cast<float*>(xb), host_dataset_shape, 2, kDLCPU, kDLFloat, 32
  );

  HnswIndexParams hnsw_params;
  hnsw_params.get()->hierarchy = CPU;
  hnsw_params.get()->ef_construction = std::max(ef, intermediate_graph_degree);
  hnsw_params.get()->num_threads = n_threads;
  hnsw_params.get()->M =
    static_cast<std::size_t>(std::max(2, graph_degree / 2));
  hnsw_params.get()->metric = L2Expanded;
  HnswIndex hnsw_index;
  cuvs_check(
    cuvsHnswFromCagraWithDataset(
      res.get(),
      hnsw_params.get(),
      cagra_index.get(),
      hnsw_index.get(),
      &host_dataset_tensor
    ),
    "cuvsHnswFromCagraWithDataset"
  );

  const float* query_ptr = same_storage ? xb : xq;
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
    false,
    false,
    float_distances
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
  out["hnsw_build_algo"] = "from_cagra_with_dataset";
  out["hnsw_hierarchy"] = "cpu";
  out["hnsw_m"] = static_cast<int>(hnsw_params.get()->M);
  out["hnsw_ef_construction"] = hnsw_params.get()->ef_construction;
  out["hnsw_parameters_adjusted"] = requested_graph_degree != graph_degree ||
    requested_intermediate_graph_degree != intermediate_graph_degree ||
    requested_ef != ef ||
    requested_n_threads != n_threads;
  annotate_cuvs_gpu_residency(
    out, "hybrid_gpu_cagra_build_host_hnsw_search", same_storage, n_data,
    n_points, n_features, search_k, false, false, 1, false
  );
  out["cuda_hnsw_design"] = "cuvs_hnsw_from_cagra_cpu_hierarchy";
  out["cuda_hnsw_pure_gpu"] = false;
  return out;
#endif
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
  validate_inputs(data, points, k, exclude_self);
  const bool same_storage = same_matrix_storage(data, points);
  const int n_data = data.nrow();
  const int n_points = points.nrow();
  const int n_features = data.ncol();

  std::vector<float> xb;
  std::vector<float> xq;
  copy_row_major_float(data, xb);
  if (!same_storage) {
    copy_row_major_float(points, xq);
  }

  return cuvs_hnsw_knn_from_row_major(
    xb.data(),
    same_storage ? nullptr : xq.data(),
    same_storage,
    n_data,
    n_points,
    n_features,
    k,
    exclude_self,
    graph_degree,
    intermediate_graph_degree,
    ef,
    n_threads,
    cagra_build_algo
  );
}

List cuvs_hnsw_float32_knn_impl(SEXP data,
                                SEXP points,
                                int k,
                                bool exclude_self,
                                int graph_degree,
                                int intermediate_graph_degree,
                                int ef,
                                int n_threads,
                                std::string cagra_build_algo,
                                std::string distance_storage) {
#ifndef FAISSR_HAS_CUVS_HNSW
  (void)distance_storage;
  Rcpp::stop(
    "Direct cuVS HNSW is not available in this cuVS installation. "
    "Reinstall cuVS with cuvs/neighbors/hnsw.h."
  );
#else
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
  const bool float_distances = wants_float_distance_storage(distance_storage);

  List out = cuvs_hnsw_knn_from_row_major(
    xb.data,
    same_storage ? nullptr : xq.data,
    same_storage,
    xb.nrow,
    same_storage ? xb.nrow : xq.nrow,
    xb.ncol,
    k,
    exclude_self,
    graph_degree,
    intermediate_graph_degree,
    ef,
    n_threads,
    cagra_build_algo,
    float_distances
  );
  out["input_type"] = "float32";
  out["input_layout"] = same_storage ?
    xb.layout :
    ("data=" + xb.layout + ";points=" + xq.layout);
  out["input_owns_data"] = same_storage ?
    xb.owns_data :
    (xb.owns_data || xq.owns_data);
  out["float32_compatibility_conversion"] = false;
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
  annotate_cuvs_gpu_residency(
    out, "gpu_transient", same_storage, n_data, n_points, n_features,
    search_k, true, true
  );
  return out;
#endif
}

List cuvs_ivf_flat_float32_knn_impl(SEXP data,
                                    SEXP points,
                                    int k,
                                    int n_lists,
                                    int n_probes,
                                    bool exclude_self,
                                    std::string distance_storage) {
#ifndef FAISSR_HAS_CUVS_IVF_FLAT
  (void)distance_storage;
  Rcpp::stop(
    "Direct cuVS IVF-Flat is not available in this cuVS installation. "
    "Use `backend = \"faiss_gpu_ivf_flat\"` or reinstall cuVS with "
    "cuvs/neighbors/ivf_flat.h."
  );
#else
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
  const bool float_distances = wants_float_distance_storage(distance_storage);
  n_lists = std::max(1, std::min(n_lists, n_data));
  n_probes = std::max(1, std::min(n_probes, n_lists));

  CuvsResources res;
  const std::size_t data_bytes =
    static_cast<std::size_t>(n_data) * n_features * sizeof(float);
  DeviceBuffer dataset_d(res.get(), data_bytes);
  cuvs_debug("ivf_flat_float32: copy dataset to device");
  cuda_check(
    cudaMemcpy(dataset_d.get(), xb.data, data_bytes, cudaMemcpyHostToDevice),
    "cudaMemcpy(dataset)"
  );

  DeviceBuffer query_d;
  void* query_ptr = dataset_d.get();
  if (!same_storage) {
    const std::size_t query_bytes =
      static_cast<std::size_t>(n_points) * n_features * sizeof(float);
    query_d.reset(res.get(), query_bytes);
    cuda_check(
      cudaMemcpy(query_d.get(), xq.data, query_bytes, cudaMemcpyHostToDevice),
      "cudaMemcpy(queries)"
    );
    query_ptr = query_d.get();
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
  cuvs_debug("ivf_flat_float32: build index begin");
  cuvs_check(
    cuvsIvfFlatBuild(res.get(), index_params.get(), &dataset_tensor, index.get()),
    "cuvsIvfFlatBuild"
  );
  cuvs_debug("ivf_flat_float32: build index done");
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
    false,
    false,
    float_distances
  );
  out["n_lists"] = n_lists;
  out["n_probes"] = n_probes;
  out["search_batch_size"] = batch_size;
  out["kmeans_n_iters"] = kmeans_iters;
  out["kmeans_trainset_fraction"] = std::min(100, train_percent) / 100.0;
  out["conservative_memory_allocation"] = true;
  annotate_float32_input(out, xb, xq, same_storage);
  annotate_cuvs_gpu_residency(
    out, "gpu_transient", same_storage, n_data, n_points, n_features,
    search_k, true, true
  );
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
  annotate_cuvs_gpu_residency(
    out, "gpu_transient", same_storage, n_data, n_points, n_features,
    search_k, true, true
  );
  return out;
#endif
}

List cuvs_ivf_pq_float32_knn_impl(SEXP data,
                                  SEXP points,
                                  int k,
                                  int n_lists,
                                  int n_probes,
                                  int pq_dim,
                                  int pq_bits,
                                  bool exclude_self,
                                  std::string distance_storage) {
#ifndef FAISSR_HAS_CUVS_IVF_PQ
  (void)distance_storage;
  Rcpp::stop(
    "Direct cuVS IVF-PQ is not available in this cuVS installation. "
    "Use `backend = \"faiss_gpu_ivfpq\"` or reinstall cuVS with "
    "cuvs/neighbors/ivf_pq.h."
  );
#else
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
  const int requested_pq_dim = pq_dim;
  const int requested_pq_bits = pq_bits;
  const bool float_distances = wants_float_distance_storage(distance_storage);
  n_lists = std::max(1, std::min(n_lists, n_data));
  n_probes = std::max(1, std::min(n_probes, n_lists));
  pq_dim = std::max(0, pq_dim);
  pq_bits = std::max(4, std::min(8, pq_bits));

  CuvsResources res;
  const std::size_t data_bytes =
    static_cast<std::size_t>(n_data) * n_features * sizeof(float);
  DeviceBuffer dataset_d(res.get(), data_bytes);
  cuda_check(
    cudaMemcpy(dataset_d.get(), xb.data, data_bytes, cudaMemcpyHostToDevice),
    "cudaMemcpy(dataset)"
  );

  DeviceBuffer query_d;
  void* query_ptr = dataset_d.get();
  if (!same_storage) {
    const std::size_t query_bytes =
      static_cast<std::size_t>(n_points) * n_features * sizeof(float);
    query_d.reset(res.get(), query_bytes);
    cuda_check(
      cudaMemcpy(query_d.get(), xq.data, query_bytes, cudaMemcpyHostToDevice),
      "cudaMemcpy(queries)"
    );
    query_ptr = query_d.get();
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
    false,
    false,
    float_distances
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
  annotate_float32_input(out, xb, xq, same_storage);
  annotate_cuvs_gpu_residency(
    out, "gpu_transient", same_storage, n_data, n_points, n_features,
    search_k, true, true
  );
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
  out["cuvs_nndescent_input_dtype"] = "float32";
  out["cuvs_nndescent_shared_memory_workaround"] = false;
  out["cuvs_nndescent_l2_norm_shared_bytes_fp32"] =
    static_cast<double>(nndescent_l2_norm_shared_bytes(n_features, sizeof(float)));
  out["cuvs_nndescent_l2_norm_shared_bytes_used"] =
    static_cast<double>(nndescent_l2_norm_shared_bytes(n_features, sizeof(float)));
  return out;
}

List cuvs_nndescent_self_float32_knn_impl(SEXP data,
                                          int k,
                                          int graph_degree,
                                          int intermediate_graph_degree,
                                          int max_iterations,
                                          std::string distance_storage) {
  MatrixViewF32 xb = make_float32_matrix_view(data, "data");
  if (xb.nrow < 2) Rcpp::stop("data must have at least two rows");
  if (xb.ncol < 1) Rcpp::stop("data must have at least one column");
  if (k < 1 || k >= xb.nrow) {
    Rcpp::stop("k must be in [1, nrow(data) - 1]");
  }

  const int n_data = xb.nrow;
  const int n_features = xb.ncol;
  graph_degree = std::max(k, graph_degree);
  graph_degree = std::min(graph_degree, n_data - 1);
  intermediate_graph_degree = std::max(
    intermediate_graph_degree,
    std::max(graph_degree, graph_degree * 2)
  );
  intermediate_graph_degree = std::min(intermediate_graph_degree, n_data - 1);
  max_iterations = std::max(1, max_iterations);
  const bool float_distances = wants_float_distance_storage(distance_storage);

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
    const_cast<float*>(xb.data), dataset_shape, 2, kDLCPU, kDLFloat, 32
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
    false,
    false,
    float_distances
  );
  out["graph_degree"] = graph_degree;
  out["intermediate_graph_degree"] = intermediate_graph_degree;
  out["max_iterations"] = max_iterations;
  out["cuvs_nndescent_input_dtype"] = "float32";
  out["cuvs_nndescent_shared_memory_workaround"] = false;
  out["cuvs_nndescent_l2_norm_shared_bytes_fp32"] =
    static_cast<double>(nndescent_l2_norm_shared_bytes(n_features, sizeof(float)));
  out["cuvs_nndescent_l2_norm_shared_bytes_used"] =
    static_cast<double>(nndescent_l2_norm_shared_bytes(n_features, sizeof(float)));
  MatrixViewF32 empty_points;
  annotate_float32_input(out, xb, empty_points, true);
  annotate_cuvs_gpu_residency(
    out, "host_tensor_nndescent", true, n_data, n_data, n_features,
    graph_degree, false, false, 0, false
  );
  return out;
}
