#include <Rcpp.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

#include "faissr_size_utils.hpp"
#include "faissr_normalize.hpp"
#include "faissr_gpu_knn_handle.hpp"

#ifdef FAISSR_HAS_CUDA
#include <cuda_runtime_api.h>
#endif

using Rcpp::IntegerMatrix;
using Rcpp::IntegerVector;
using Rcpp::List;
using Rcpp::NumericMatrix;

extern "C" {
bool faissr_cuda_available();
const char* faissr_cuda_last_error();
const char* faissr_cuda_device_info_json();
int faissr_cuda_knn(const double* data,
                        const double* points,
                        int n_data,
                        int n_points,
                        int n_features,
                        int k,
                        int square,
                        int* out_indices,
                        double* out_distances);
int faissr_cuda_knn_float(const float* data,
                              const float* points,
                              int n_data,
                              int n_points,
                              int n_features,
                              int k,
                              int square,
                              int* out_indices,
                              double* out_distances);
int faissr_cuda_knn_float_gpu(const float* data,
                                  const float* points,
                                  int n_data,
                                  int n_points,
                                  int n_features,
                                  int k,
                                  int square,
                                  int exclude_self,
                                  int* out_max_batch,
                                  int** out_indices_device,
                                  float** out_distances_device);
int faissr_cuda_knn_gpu_postprocess_distances(float* distances,
                                                 int n_points,
                                                 int k,
                                                 float scale,
                                                 int subtract_first);
int faissr_cuda_knn_gpu_result_to_host(const int* indices_device,
                                          const float* distances_device,
                                          int n_points,
                                          int k,
                                          int* out_indices,
                                          double* out_distances);
void faissr_cuda_free_device(void* ptr);
int faissr_cuda_row_candidate_knn(const double* data,
                                      const int* candidate_indices,
                                      int n,
                                      int n_features,
                                      int n_candidates,
                                      int k,
                                      int metric_kind,
                                      int* out_indices,
                                      double* out_distances,
                                      int fill_missing);
int faissr_cuda_row_candidate_knn_float(const float* data,
                                            const int* candidate_indices,
                                            int n,
                                            int n_features,
                                            int n_candidates,
                                            int k,
                                            int metric_kind,
                                            int* out_indices,
                                            double* out_distances);
int faissr_cuda_grid_self_knn(const double* data,
                                  int n,
                                  int n_features,
                                  int k,
                                  int bins_per_dim,
                                  int* out_indices,
                                  double* out_distances,
                                  int* out_n_cells);
int faissr_cuda_grid_self_knn_float32(const float* data,
                                      int n,
                                      int n_features,
                                      int k,
                                      int bins_per_dim,
                                      int* out_indices,
                                      double* out_distances,
                                      int* out_n_cells);
}

namespace {

constexpr int kMaxCudaK = 256;

using CudaGpuKnnHandle = FaissRGpuKnnHandle;

struct CudaMatrixViewF32 {
  const float* data = nullptr;
  int nrow = 0;
  int ncol = 0;
  bool owns_data = false;
  bool compatibility_conversion = false;
  std::string layout;
  std::vector<float> buffer;
  std::vector<char> normalized_zero;
};

const char* cuda_error_message() {
  const char* msg = faissr_cuda_last_error();
  return msg == nullptr ? "unknown CUDA error" : msg;
}

void gpu_knn_handle_finalizer(SEXP ptr) {
  if (TYPEOF(ptr) != EXTPTRSXP) return;
  CudaGpuKnnHandle* handle = static_cast<CudaGpuKnnHandle*>(R_ExternalPtrAddr(ptr));
  if (handle == nullptr) return;
#ifdef FAISSR_HAS_CUDA
  int previous_device = -1;
  cudaGetDevice(&previous_device);
  cudaSetDevice(handle->device);
#endif
  faissr_cuda_free_device(handle->indices);
  faissr_cuda_free_device(handle->distances);
  faissr_cuda_free_device(handle->data);
#ifdef FAISSR_HAS_CUDA
  if (previous_device >= 0 && previous_device != handle->device) {
    cudaSetDevice(previous_device);
  }
#endif
  handle->indices = nullptr;
  handle->distances = nullptr;
  handle->data = nullptr;
  delete handle;
  R_ClearExternalPtr(ptr);
}

CudaGpuKnnHandle* gpu_knn_handle_from_result(SEXP x) {
  if (!Rf_isNewList(x)) {
    Rcpp::stop("GPU KNN result must be a faissR_gpu_knn list");
  }
  Rcpp::List result(x);
  if (!result.containsElementNamed("handle")) {
    Rcpp::stop("GPU KNN result is missing its owning handle");
  }
  SEXP ptr = result["handle"];
  if (TYPEOF(ptr) != EXTPTRSXP) {
    Rcpp::stop("GPU KNN handle is not an external pointer");
  }
  CudaGpuKnnHandle* handle = static_cast<CudaGpuKnnHandle*>(R_ExternalPtrAddr(ptr));
  if (handle == nullptr || handle->indices == nullptr || handle->distances == nullptr) {
    Rcpp::stop("GPU KNN handle is no longer valid");
  }
  return handle;
}

List make_gpu_knn_result(CudaGpuKnnHandle* handle,
                         const std::string& metric,
                         const std::string& backend_used,
                         const std::string& method,
                         bool exclude_self,
                         bool exact,
                         int max_batch,
                         const CudaMatrixViewF32& data_view,
                         const CudaMatrixViewF32& points_view) {
  SEXP owner = PROTECT(R_MakeExternalPtr(handle, R_NilValue, R_NilValue));
  R_RegisterCFinalizerEx(owner, gpu_knn_handle_finalizer, TRUE);
  SEXP indices_ptr = PROTECT(R_MakeExternalPtr(handle->indices, R_NilValue, owner));
  SEXP distances_ptr = PROTECT(R_MakeExternalPtr(handle->distances, R_NilValue, owner));
  Rcpp::List out = Rcpp::List::create(
    Rcpp::Named("handle") = owner,
    Rcpp::Named("indices_ptr") = indices_ptr,
    Rcpp::Named("distances_ptr") = distances_ptr,
    Rcpp::Named("n_query") = handle->n_points,
    Rcpp::Named("k") = handle->k,
    Rcpp::Named("index_base") = 1,
    Rcpp::Named("indices_type") = "int32",
    Rcpp::Named("distance_type") = "float32",
    Rcpp::Named("distance_residency") = "cuda_device",
    Rcpp::Named("indices_residency") = "cuda_device",
    Rcpp::Named("result_residency") = "cuda",
    Rcpp::Named("layout") = "column_major_query_by_k",
    Rcpp::Named("metric") = metric,
    Rcpp::Named("backend_used") = backend_used,
    Rcpp::Named("method") = method,
    Rcpp::Named("accelerator") = "cuda",
    Rcpp::Named("gpu_provider") = "native_cuda_exact",
    Rcpp::Named("device") = handle->device,
    Rcpp::Named("exact") = exact,
    Rcpp::Named("exclude_self") = exclude_self,
    Rcpp::Named("max_query_batch_size") = max_batch,
    Rcpp::Named("input_type") = "float32",
    Rcpp::Named("data_input_layout") = data_view.layout,
    Rcpp::Named("points_input_layout") = points_view.layout,
    Rcpp::Named("input_owns_data") = data_view.owns_data || points_view.owns_data,
    Rcpp::Named("float32_compatibility_conversion") =
      data_view.compatibility_conversion || points_view.compatibility_conversion,
    Rcpp::Named("device_to_host_result_copies") = 0,
    Rcpp::Named("device_to_host_result_copies_known") = true,
    Rcpp::Named("memory_owner") = "handle_externalptr",
    Rcpp::Named("buffer_ownership") =
      "indices_ptr_and_distances_ptr_are_non_owning",
    Rcpp::Named("owner_lifetime") =
      "retain_result_until_all_consumer_cuda_work_completes",
    Rcpp::Named("producer_synchronization") =
      "cuda_device_synchronized_before_return",
    Rcpp::Named("stream_semantics") =
      "no_stream_handle_exported;consumer_selects_stream_after_return",
    Rcpp::Named("consumer_device_requirement") =
      "select_recorded_device_before_pointer_use",
    Rcpp::Named("serialization_supported") = false,
    Rcpp::Named("interprocess_sharing_supported") = false,
    Rcpp::Named("cpu_fallback") = false
  );
  out["distance_order"] = "smaller_is_better";
  out["similarity_materialized"] = false;
  if (metric == "euclidean") {
    out["distance_is_metric"] = true;
    out["distance_semantics"] = "euclidean_distance";
    out["distance_comparable_across_queries"] = true;
  } else if (metric == "cosine") {
    out["distance_is_metric"] = false;
    out["distance_semantics"] = "one_minus_cosine_similarity";
    out["distance_comparable_across_queries"] = true;
  } else if (metric == "correlation") {
    out["distance_is_metric"] = false;
    out["distance_semantics"] = "one_minus_row_centered_cosine_similarity";
    out["distance_comparable_across_queries"] = true;
  } else {
    out["distance_is_metric"] = false;
    out["distance_semantics"] = "query_specific_shifted_inner_product";
    out["distance_comparable_across_queries"] = false;
    out["similarity_semantics"] = "raw_inner_product";
    out["similarity_order"] = "larger_is_better";
  }
  out.attr("class") = Rcpp::CharacterVector::create("faissR_gpu_knn", "list");
  out.attr("metric") = metric;
  out.attr("backend_used") = backend_used;
  out.attr("resolved_backend") = backend_used;
  out.attr("result_residency") = "cuda";
  out.attr("distance_is_metric") = out["distance_is_metric"];
  out.attr("distance_semantics") = out["distance_semantics"];
  out.attr("distance_comparable_across_queries") =
    out["distance_comparable_across_queries"];
  out.attr("distance_order") = "smaller_is_better";
  UNPROTECT(3);
  return out;
}

IntegerVector matrix_dims_from_object_f32(SEXP x, const char* name) {
  SEXP dim = Rf_getAttrib(x, R_DimSymbol);
  if (Rf_isNull(dim) && Rf_isS4(x)) {
    SEXP data_slot = R_do_slot(x, Rf_install("Data"));
    dim = Rf_getAttrib(data_slot, R_DimSymbol);
  }
  if (Rf_isNull(dim) || Rf_length(dim) != 2) {
    Rcpp::stop("%s must be a two-dimensional numeric or float32 matrix", name);
  }
  IntegerVector dims(dim);
  if (dims[0] < 1 || dims[1] < 1) {
    Rcpp::stop("%s must have at least one row and one column", name);
  }
  return dims;
}

const float* float32_slot_ptr_f32(SEXP slot,
                                  const R_xlen_t expected_length,
                                  const char* name) {
  if (TYPEOF(slot) == INTSXP) {
    if (Rf_xlength(slot) != expected_length) {
      Rcpp::stop("%s float32 payload length does not match its dimensions", name);
    }
    return reinterpret_cast<const float*>(INTEGER(slot));
  }
  if (TYPEOF(slot) == RAWSXP) {
    const R_xlen_t expected_bytes =
      faissr::float_payload_byte_count(expected_length, name);
    if (Rf_xlength(slot) != expected_bytes) {
      Rcpp::stop("%s float32 raw payload length does not match its dimensions", name);
    }
    return reinterpret_cast<const float*>(RAW(slot));
  }
  return nullptr;
}

bool finite_float32_payload_f32(const float* ptr, const R_xlen_t length) {
  for (R_xlen_t i = 0; i < length; ++i) {
    if (!std::isfinite(ptr[i])) return false;
  }
  return true;
}

CudaMatrixViewF32 make_column_major_float32_view(
    SEXP x, const char* name, const std::string& metric = "euclidean") {
  IntegerVector dims = matrix_dims_from_object_f32(x, name);
  CudaMatrixViewF32 view;
  view.nrow = dims[0];
  view.ncol = dims[1];
  const R_xlen_t expected_length =
    faissr::matrix_element_count(view.nrow, view.ncol);

  bool finite = true;
  if (TYPEOF(x) == REALSXP) {
    if (Rf_xlength(x) != expected_length) {
      Rcpp::stop("%s payload length does not match its dimensions", name);
    }
    view.buffer.assign(static_cast<std::size_t>(expected_length), 0.0f);
    view.owns_data = true;
    view.compatibility_conversion = true;
    view.layout = "r_double_column_major_to_column_major_float32";
    const double* src = REAL(x);
    if (metric == "cosine" || metric == "correlation") {
      view.normalized_zero = faissr::normalized_float_matrix(
        view.nrow, view.ncol, metric == "correlation", true,
        [&](int r, int c) { return src[static_cast<std::size_t>(c) * view.nrow + r]; },
        view.buffer
      );
    } else {
      for (R_xlen_t i = 0; i < expected_length; ++i) {
        const float value = static_cast<float>(src[i]);
        if (!std::isfinite(src[i]) || !std::isfinite(value)) finite = false;
        view.buffer[static_cast<std::size_t>(i)] = value;
      }
    }
  } else if (Rf_isS4(x)) {
    SEXP slot = R_do_slot(x, Rf_install("Data"));
    const float* payload = float32_slot_ptr_f32(slot, expected_length, name);
    if (payload != nullptr) {
      finite = finite_float32_payload_f32(payload, expected_length);
      const bool row_major_payload =
        Rf_asLogical(Rf_getAttrib(x, Rf_install("faissR_row_major_float32"))) == TRUE;
      if (!row_major_payload || view.nrow == 1 || view.ncol == 1) {
        view.data = payload;
        view.owns_data = false;
        view.layout = row_major_payload ?
          "float32_payload_direct_column_compatible" :
          "float32_payload_direct_column_major";
      } else {
        view.buffer.assign(static_cast<std::size_t>(expected_length), 0.0f);
        view.owns_data = true;
        view.layout = "float32_row_major_payload_to_column_major";
        for (int r = 0; r < view.nrow; ++r) {
          for (int c = 0; c < view.ncol; ++c) {
            view.buffer[static_cast<std::size_t>(c) * view.nrow + r] =
              payload[static_cast<std::size_t>(r) * view.ncol + c];
          }
        }
      }
    } else if (TYPEOF(slot) == REALSXP) {
      if (Rf_xlength(slot) != expected_length) {
        Rcpp::stop("%s payload length does not match its dimensions", name);
      }
      view.buffer.assign(static_cast<std::size_t>(expected_length), 0.0f);
      view.owns_data = true;
      view.compatibility_conversion = true;
      view.layout = "s4_double_column_major_to_column_major_float32";
      const double* src = REAL(slot);
      if (metric == "cosine" || metric == "correlation") {
        view.normalized_zero = faissr::normalized_float_matrix(
          view.nrow, view.ncol, metric == "correlation", true,
          [&](int r, int c) { return src[static_cast<std::size_t>(c) * view.nrow + r]; },
          view.buffer
        );
      } else {
        for (R_xlen_t i = 0; i < expected_length; ++i) {
          const float value = static_cast<float>(src[i]);
          if (!std::isfinite(src[i]) || !std::isfinite(value)) finite = false;
          view.buffer[static_cast<std::size_t>(i)] = value;
        }
      }
    } else {
      Rcpp::stop("%s must be a float::fl()/float32 object with an integer or raw @Data payload", name);
    }
  } else {
    Rcpp::stop("%s must be an ordinary R double matrix or float::fl()/float32 object", name);
  }
  if (!finite) Rcpp::stop("%s requires finite values representable in float32", name);
  if (view.data == nullptr) view.data = view.buffer.data();
  return view;
}

bool normalize_metric_view(CudaMatrixViewF32& view,
                           bool center_rows,
                           const char* name) {
  if (view.normalized_zero.empty()) {
    std::vector<float> transformed;
    view.normalized_zero = faissr::normalized_float_matrix(
      view.nrow, view.ncol, center_rows, true,
      [&](int r, int c) { return view.data[static_cast<std::size_t>(c) * view.nrow + r]; },
      transformed
    );
    view.buffer.swap(transformed);
  }
  const bool has_zero = std::find(
    view.normalized_zero.begin(), view.normalized_zero.end(), 1
  ) != view.normalized_zero.end();
  if (has_zero) {
    Rcpp::stop(
      "%s contains rows with zero norm after %s; GPU-resident cosine/correlation "
      "output does not perform CPU-side zero-row repair.",
      name,
      center_rows ? "centering" : "normalization"
    );
  }
  view.data = view.buffer.data();
  view.owns_data = true;
  view.layout = center_rows ?
    "column_major_float32_row_center_l2_normalized" :
    "column_major_float32_row_l2_normalized";
  return has_zero;
}

void mips_l2_transform_views(CudaMatrixViewF32& data_view,
                             CudaMatrixViewF32& points_view) {
  const int n_data = data_view.nrow;
  const int n_points = points_view.nrow;
  const int p = data_view.ncol;
  std::vector<double> data_norm2(static_cast<std::size_t>(n_data), 0.0);
  double radius2 = 0.0;
  for (int r = 0; r < n_data; ++r) {
    double norm2 = 0.0;
    for (int c = 0; c < p; ++c) {
      const double value =
        static_cast<double>(data_view.data[static_cast<std::size_t>(c) * n_data + r]);
      norm2 += value * value;
    }
    data_norm2[static_cast<std::size_t>(r)] = norm2;
    if (std::isfinite(norm2)) radius2 = std::max(radius2, norm2);
  }
  std::vector<float> data_transformed(
    static_cast<std::size_t>(n_data) * (p + 1),
    0.0f
  );
  for (int c = 0; c < p; ++c) {
    std::memcpy(
      data_transformed.data() + static_cast<std::size_t>(c) * n_data,
      data_view.data + static_cast<std::size_t>(c) * n_data,
      static_cast<std::size_t>(n_data) * sizeof(float)
    );
  }
  for (int r = 0; r < n_data; ++r) {
    const double extra2 = std::max(0.0, radius2 - data_norm2[static_cast<std::size_t>(r)]);
    data_transformed[static_cast<std::size_t>(p) * n_data + r] =
      static_cast<float>(std::sqrt(extra2));
  }

  std::vector<float> points_transformed(
    static_cast<std::size_t>(n_points) * (p + 1),
    0.0f
  );
  for (int c = 0; c < p; ++c) {
    std::memcpy(
      points_transformed.data() + static_cast<std::size_t>(c) * n_points,
      points_view.data + static_cast<std::size_t>(c) * n_points,
      static_cast<std::size_t>(n_points) * sizeof(float)
    );
  }

  data_view.buffer.swap(data_transformed);
  data_view.data = data_view.buffer.data();
  data_view.ncol = p + 1;
  data_view.owns_data = true;
  data_view.layout = "column_major_float32_mips_l2_data_transform";

  points_view.buffer.swap(points_transformed);
  points_view.data = points_view.buffer.data();
  points_view.ncol = p + 1;
  points_view.owns_data = true;
  points_view.layout = "column_major_float32_mips_l2_query_transform";
}

} // namespace

bool cuda_is_available_impl() {
  return faissr_cuda_available();
}

std::string cuda_device_info_json_impl() {
  const char* info = faissr_cuda_device_info_json();
  return info == nullptr ? std::string("{}") : std::string(info);
}

List cuda_nn_impl(NumericMatrix data,
                  NumericMatrix points,
                  int k,
                  bool square) {
  if (data.ncol() != points.ncol()) Rcpp::stop("data and points must have the same number of columns");
  if (k < 1 || k > data.nrow()) Rcpp::stop("k must be in [1, nrow(data)]");
  if (k > kMaxCudaK) Rcpp::stop("CUDA backend currently supports k <= %d", kMaxCudaK);
  if (!faissr_cuda_available()) Rcpp::stop("No CUDA device is available.");

  const int n_data = data.nrow();
  const int n_points = points.nrow();
  const int n_features = data.ncol();
  IntegerMatrix indices(n_points, k);
  NumericMatrix distances(n_points, k);

  const int status = faissr_cuda_knn(
    data.begin(),
    points.begin(),
    n_data,
    n_points,
    n_features,
    k,
    square ? 1 : 0,
    indices.begin(),
    distances.begin()
  );
  if (status != 0) {
    Rcpp::stop("CUDA KNN failed: %s", cuda_error_message());
  }

  return List::create(
    Rcpp::Named("indices") = indices,
    Rcpp::Named("distances") = distances
  );
}

List cuda_nn_float32_impl(SEXP data,
                          SEXP points,
                          int k,
                          bool square) {
  CudaMatrixViewF32 data_view = make_column_major_float32_view(data, "data");
  const bool same_object = data == points;
  CudaMatrixViewF32 points_view = same_object ?
    CudaMatrixViewF32() :
    make_column_major_float32_view(points, "points");
  if (same_object) {
    points_view.nrow = data_view.nrow;
    points_view.ncol = data_view.ncol;
    points_view.data = data_view.data;
    points_view.layout = data_view.layout;
  }
  if (data_view.ncol != points_view.ncol) {
    Rcpp::stop("data and points must have the same number of columns");
  }
  if (k < 1 || k > data_view.nrow) Rcpp::stop("k must be in [1, nrow(data)]");
  if (k > kMaxCudaK) Rcpp::stop("CUDA backend currently supports k <= %d", kMaxCudaK);
  if (!faissr_cuda_available()) Rcpp::stop("No CUDA device is available.");

  IntegerMatrix indices(points_view.nrow, k);
  NumericMatrix distances(points_view.nrow, k);
  const int status = faissr_cuda_knn_float(
    data_view.data,
    points_view.data,
    data_view.nrow,
    points_view.nrow,
    data_view.ncol,
    k,
    square ? 1 : 0,
    indices.begin(),
    distances.begin()
  );
  if (status != 0) {
    Rcpp::stop("CUDA float32 KNN failed: %s", cuda_error_message());
  }

  return List::create(
    Rcpp::Named("indices") = indices,
    Rcpp::Named("distances") = distances,
    Rcpp::Named("input_type") = "float32",
    Rcpp::Named("input_layout") = data_view.layout,
    Rcpp::Named("input_owns_data") = data_view.owns_data,
    Rcpp::Named("float32_compatibility_conversion") = data_view.compatibility_conversion
  );
}

List cuda_nn_float32_gpu_impl(SEXP data,
                              SEXP points,
                              int k,
                              bool exclude_self,
                              std::string metric,
                              std::string backend_used,
                              std::string method) {
  CudaMatrixViewF32 data_view = make_column_major_float32_view(data, "data", metric);
  const bool same_object = data == points;
  CudaMatrixViewF32 points_view = same_object ?
    CudaMatrixViewF32() :
    make_column_major_float32_view(points, "points", metric);
  if (same_object) {
    points_view.nrow = data_view.nrow;
    points_view.ncol = data_view.ncol;
    points_view.data = data_view.data;
    points_view.layout = data_view.layout;
  }
  if (data_view.ncol != points_view.ncol) {
    Rcpp::stop("data and points must have the same number of columns");
  }
  if (exclude_self && data_view.nrow != points_view.nrow) {
    Rcpp::stop("exclude_self is only valid for self-query GPU KNN calls");
  }
  if (k < 1 || k > data_view.nrow || (exclude_self && k >= data_view.nrow)) {
    Rcpp::stop("k is not compatible with the requested GPU KNN shape");
  }
  if (k > kMaxCudaK) Rcpp::stop("CUDA backend currently supports k <= %d", kMaxCudaK);
  if (!faissr_cuda_available()) Rcpp::stop("No CUDA device is available.");

  bool square = false;
  float distance_scale = 1.0f;
  bool subtract_first = false;
  std::string metric_transform = "none";
  if (metric == "cosine" || metric == "correlation") {
    const bool center_rows = metric == "correlation";
    normalize_metric_view(data_view, center_rows, "data");
    if (same_object) {
      points_view.nrow = data_view.nrow;
      points_view.ncol = data_view.ncol;
      points_view.data = data_view.data;
      points_view.layout = data_view.layout;
      points_view.owns_data = false;
      points_view.compatibility_conversion = data_view.compatibility_conversion;
    } else {
      normalize_metric_view(points_view, center_rows, "points");
    }
    square = true;
    distance_scale = 0.5f;
    metric_transform = center_rows ?
      "row_center_l2_normalize_then_squared_l2_over_2" :
      "row_l2_normalize_then_squared_l2_over_2";
  } else if (metric == "inner_product") {
    if (same_object) {
      points_view.nrow = data_view.nrow;
      points_view.ncol = data_view.ncol;
      points_view.data = data_view.data;
      points_view.layout = data_view.layout;
      points_view.owns_data = false;
      points_view.compatibility_conversion = data_view.compatibility_conversion;
    }
    mips_l2_transform_views(data_view, points_view);
    square = true;
    distance_scale = 0.5f;
    subtract_first = true;
    metric_transform = "maximum_inner_product_to_l2_then_shifted_distance";
  } else if (metric != "euclidean") {
    Rcpp::stop("metric must be euclidean, cosine, correlation, or inner_product");
  }

  int* d_indices = nullptr;
  float* d_distances = nullptr;
  int max_batch = 0;
  const int status = faissr_cuda_knn_float_gpu(
    data_view.data,
    points_view.data,
    data_view.nrow,
    points_view.nrow,
    data_view.ncol,
    k,
    square ? 1 : 0,
    exclude_self ? 1 : 0,
    &max_batch,
    &d_indices,
    &d_distances
  );
  if (status != 0) {
    Rcpp::stop("CUDA GPU-resident float32 KNN failed: %s", cuda_error_message());
  }
  if (distance_scale != 1.0f || subtract_first) {
    const int post_status = faissr_cuda_knn_gpu_postprocess_distances(
      d_distances,
      points_view.nrow,
      k,
      distance_scale,
      subtract_first ? 1 : 0
    );
    if (post_status != 0) {
      faissr_cuda_free_device(d_indices);
      faissr_cuda_free_device(d_distances);
      Rcpp::stop("CUDA GPU-resident distance post-processing failed: %s", cuda_error_message());
    }
  }

  CudaGpuKnnHandle* handle = new CudaGpuKnnHandle();
  handle->indices = d_indices;
  handle->distances = d_distances;
  handle->n_points = points_view.nrow;
  handle->k = k;
  handle->device = 0;
  List out = make_gpu_knn_result(
    handle,
    metric,
    backend_used,
    method,
    exclude_self,
    true,
    max_batch,
    data_view,
    points_view
  );
  out["metric_transform"] = metric_transform;
  out["distance_transform"] = metric_transform;
  out["host_to_device_data_copies"] = 1;
  out["host_to_device_query_copies"] = same_object ? 0 : 1;
  out["host_to_device_copies_known"] = true;
  out["index_residency"] = "transient_cuda_dataset_buffer";
  out["query_residency"] = same_object ? "dataset_device_buffer" : "transient_cuda_query_buffer";
  return out;
}

List cuda_gpu_knn_to_host_impl(SEXP result) {
  CudaGpuKnnHandle* handle = gpu_knn_handle_from_result(result);
  IntegerMatrix indices(handle->n_points, handle->k);
  NumericMatrix distances(handle->n_points, handle->k);
#ifdef FAISSR_HAS_CUDA
  int previous_device = -1;
  cudaGetDevice(&previous_device);
  if (cudaSetDevice(handle->device) != cudaSuccess) {
    Rcpp::stop("Could not select the GPU holding the KNN result.");
  }
#endif
  const int status = faissr_cuda_knn_gpu_result_to_host(
    handle->indices,
    handle->distances,
    handle->n_points,
    handle->k,
    indices.begin(),
    distances.begin()
  );
#ifdef FAISSR_HAS_CUDA
  if (previous_device >= 0 && previous_device != handle->device) {
    cudaSetDevice(previous_device);
  }
#endif
  if (status != 0) {
    Rcpp::stop("CUDA GPU KNN result copy failed: %s", cuda_error_message());
  }
  Rcpp::List src(result);
  const std::string metric = Rcpp::as<std::string>(src["metric"]);
  const std::string backend_used = Rcpp::as<std::string>(src["backend_used"]);
  List out = List::create(
    Rcpp::Named("indices") = indices,
    Rcpp::Named("distances") = distances,
    Rcpp::Named("index_base") = src["index_base"],
    Rcpp::Named("distance_type") = "double",
    Rcpp::Named("metric") = metric,
    Rcpp::Named("backend_used") = backend_used,
    Rcpp::Named("copied_from_residency") = "cuda"
  );
  out.attr("class") = Rcpp::CharacterVector::create("faissR_nn", "list");
  out.attr("metric") = metric;
  out.attr("backend_used") = backend_used;
  out.attr("result_residency") = "host";
  return out;
}

List cuda_gpu_knn_sample_to_host_impl(SEXP result,
                                      Rcpp::IntegerVector rows) {
  CudaGpuKnnHandle* handle = gpu_knn_handle_from_result(result);
  if (rows.size() < 1 || rows.size() > 256) {
    Rcpp::stop("GPU KNN sample requires 1 to 256 row IDs.");
  }
  for (int row : rows) {
    if (row == NA_INTEGER || row < 1 || row > handle->n_points) {
      Rcpp::stop("GPU KNN sample row ID is out of range.");
    }
  }
  IntegerMatrix indices(rows.size(), handle->k);
  NumericMatrix distances(rows.size(), handle->k);
#ifdef FAISSR_HAS_CUDA
  int previous_device = -1;
  cudaGetDevice(&previous_device);
  if (cudaSetDevice(handle->device) != cudaSuccess) {
    Rcpp::stop("Could not select the GPU holding the KNN sample.");
  }
  cudaError_t status = cudaSuccess;
  for (int column = 0; column < handle->k && status == cudaSuccess;
       ++column) {
    for (int sample = 0; sample < rows.size(); ++sample) {
      const std::size_t offset =
        static_cast<std::size_t>(column) * handle->n_points + rows[sample] - 1;
      int index = 0;
      float distance = 0.0f;
      status = cudaMemcpy(&index, handle->indices + offset, sizeof(index),
                          cudaMemcpyDeviceToHost);
      if (status != cudaSuccess) break;
      status = cudaMemcpy(&distance, handle->distances + offset,
                          sizeof(distance), cudaMemcpyDeviceToHost);
      if (status != cudaSuccess) break;
      indices(sample, column) = index;
      distances(sample, column) = distance;
    }
  }
  if (previous_device >= 0 && previous_device != handle->device) {
    cudaSetDevice(previous_device);
  }
  if (status != cudaSuccess) {
    Rcpp::stop("CUDA KNN sample copy failed: %s",
               cudaGetErrorString(status));
  }
#endif
  Rcpp::List source(result);
  List out = List::create(
    Rcpp::Named("indices") = indices,
    Rcpp::Named("distances") = distances,
    Rcpp::Named("rows") = rows,
    Rcpp::Named("index_base") = source["index_base"],
    Rcpp::Named("metric") = source["metric"],
    Rcpp::Named("backend_used") = source["backend_used"],
    Rcpp::Named("copied_from_residency") = "cuda"
  );
  out.attr("class") =
    Rcpp::CharacterVector::create("faissR_gpu_knn_sample", "list");
  return out;
}

List cuda_row_candidate_knn_impl(NumericMatrix data,
                                 IntegerMatrix candidate_indices,
                                 int k,
                                 std::string metric,
                                 bool fill_missing) {
  const int n = data.nrow();
  const int n_features = data.ncol();
  const int n_candidates = candidate_indices.ncol();
  if (n < 2) Rcpp::stop("data must have at least two rows");
  if (candidate_indices.nrow() != n) {
    Rcpp::stop("candidate_indices row count must match data");
  }
  if (n_candidates < 1) Rcpp::stop("candidate_indices must have at least one column");
  if (k < 1 || (fill_missing && k >= n)) {
    Rcpp::stop("k must be positive and smaller than nrow(data) when filling candidates");
  }
  if (k > kMaxCudaK) Rcpp::stop("CUDA backend currently supports k <= %d", kMaxCudaK);
  if (!faissr_cuda_available()) Rcpp::stop("No CUDA device is available.");
  int metric_kind = 0;
  if (metric == "inner_product") {
    metric_kind = 1;
  } else if (metric != "euclidean") {
    Rcpp::stop("CUDA row candidate KNN supports euclidean or inner_product scoring");
  }

  IntegerMatrix indices(n, k);
  NumericMatrix distances(n, k);
  const int status = faissr_cuda_row_candidate_knn(
    data.begin(),
    candidate_indices.begin(),
    n,
    n_features,
    n_candidates,
    k,
    metric_kind,
    indices.begin(),
    distances.begin(),
    fill_missing ? 1 : 0
  );
  if (status != 0) {
    Rcpp::stop("CUDA row candidate KNN failed: %s", cuda_error_message());
  }
  List result = List::create(
    Rcpp::Named("indices") = indices,
    Rcpp::Named("distances") = distances,
    Rcpp::Named("cuda_kernel") = "row_candidate_knn",
    Rcpp::Named("candidate_columns") = n_candidates,
    Rcpp::Named("metric_kind") = metric,
    Rcpp::Named("output_layout") = "knn_matrix_final",
    Rcpp::Named("r_side_reshaping") = false,
    Rcpp::Named("accelerator") = "cuda",
    Rcpp::Named("gpu_provider") = "native_cuda_row_candidate",
    Rcpp::Named("device_residency") = "cuda",
    Rcpp::Named("index_residency") = "candidate_graph_device_buffer",
    Rcpp::Named("gpu_index_resident") = true,
    Rcpp::Named("gpu_index_persistent") = false,
    Rcpp::Named("host_to_device_transfer_strategy") = "native_cuda_row_candidate_device_buffers",
    Rcpp::Named("host_to_device_data_copies") = 1,
    Rcpp::Named("host_to_device_query_copies") = 0,
    Rcpp::Named("index_build_host_to_device_copies") = 1,
    Rcpp::Named("host_to_device_copies") = 2,
    Rcpp::Named("host_to_device_copies_known") = true,
    Rcpp::Named("device_to_host_result_copies") = 2,
    Rcpp::Named("device_to_host_result_copies_known") = true,
    Rcpp::Named("query_reuses_device_data") = true,
    Rcpp::Named("query_residency") = "dataset_device_buffer",
    Rcpp::Named("result_residency") = "host",
    Rcpp::Named("cpu_fallback") = false,
    Rcpp::Named("cpu_side_result_repair") = false
  );
  result.attr("cuda_kernel") = "row_candidate_knn";
  result.attr("candidate_columns") = n_candidates;
  result.attr("metric_kind") = metric;
  return result;
}

List cuda_row_candidate_knn_float32_impl(SEXP data,
                                         IntegerMatrix candidate_indices,
                                         int k,
                                         std::string metric) {
  CudaMatrixViewF32 data_view = make_column_major_float32_view(data, "data");
  const int n = data_view.nrow;
  const int n_features = data_view.ncol;
  const int n_candidates = candidate_indices.ncol();
  if (n < 2) Rcpp::stop("data must have at least two rows");
  if (candidate_indices.nrow() != n) {
    Rcpp::stop("candidate_indices row count must match data");
  }
  if (n_candidates < 1) Rcpp::stop("candidate_indices must have at least one column");
  if (k < 1 || k >= n) Rcpp::stop("k must be in [1, nrow(data) - 1]");
  if (k > kMaxCudaK) Rcpp::stop("CUDA backend currently supports k <= %d", kMaxCudaK);
  if (!faissr_cuda_available()) Rcpp::stop("No CUDA device is available.");
  int metric_kind = 0;
  if (metric == "inner_product") {
    metric_kind = 1;
  } else if (metric != "euclidean") {
    Rcpp::stop("CUDA row candidate KNN supports euclidean or inner_product scoring");
  }

  IntegerMatrix indices(n, k);
  NumericMatrix distances(n, k);
  const int status = faissr_cuda_row_candidate_knn_float(
    data_view.data,
    candidate_indices.begin(),
    n,
    n_features,
    n_candidates,
    k,
    metric_kind,
    indices.begin(),
    distances.begin()
  );
  if (status != 0) {
    Rcpp::stop("CUDA float32 row candidate KNN failed: %s", cuda_error_message());
  }
  List result = List::create(
    Rcpp::Named("indices") = indices,
    Rcpp::Named("distances") = distances,
    Rcpp::Named("input_type") = "float32",
    Rcpp::Named("input_layout") = data_view.layout,
    Rcpp::Named("input_owns_data") = data_view.owns_data,
    Rcpp::Named("float32_compatibility_conversion") = data_view.compatibility_conversion,
    Rcpp::Named("cuda_kernel") = "row_candidate_knn",
    Rcpp::Named("candidate_columns") = n_candidates,
    Rcpp::Named("metric_kind") = metric,
    Rcpp::Named("output_layout") = "knn_matrix_final",
    Rcpp::Named("r_side_reshaping") = false,
    Rcpp::Named("accelerator") = "cuda",
    Rcpp::Named("gpu_provider") = "native_cuda_row_candidate",
    Rcpp::Named("device_residency") = "cuda",
    Rcpp::Named("index_residency") = "candidate_graph_device_buffer",
    Rcpp::Named("gpu_index_resident") = true,
    Rcpp::Named("gpu_index_persistent") = false,
    Rcpp::Named("host_to_device_transfer_strategy") = "native_cuda_row_candidate_device_buffers",
    Rcpp::Named("host_to_device_data_copies") = 1,
    Rcpp::Named("host_to_device_query_copies") = 0,
    Rcpp::Named("index_build_host_to_device_copies") = 1,
    Rcpp::Named("host_to_device_copies") = 2,
    Rcpp::Named("host_to_device_copies_known") = true,
    Rcpp::Named("device_to_host_result_copies") = 2,
    Rcpp::Named("device_to_host_result_copies_known") = true,
    Rcpp::Named("query_reuses_device_data") = true,
    Rcpp::Named("query_residency") = "dataset_device_buffer",
    Rcpp::Named("result_residency") = "host",
    Rcpp::Named("cpu_fallback") = false,
    Rcpp::Named("cpu_side_result_repair") = false
  );
  result.attr("cuda_kernel") = "row_candidate_knn";
  result.attr("candidate_columns") = n_candidates;
  result.attr("metric_kind") = metric;
  return result;
}

List cuda_grid_self_knn_impl(NumericMatrix data,
                             int k,
                             int bins_per_dim,
                             bool include_self) {
  const int n = data.nrow();
  const int n_features = data.ncol();
  if (n_features != 2 && n_features != 3) {
    Rcpp::stop("CUDA grid KNN requires a two- or three-column matrix");
  }
  if (n < 2) Rcpp::stop("data must have at least two rows");
  if (k < 1 || k > n) Rcpp::stop("k must be in [1, nrow(data)]");
  if (!include_self && k >= n) Rcpp::stop("k must be in [1, nrow(data) - 1] when excluding self");
  if (k > kMaxCudaK) Rcpp::stop("CUDA backend currently supports k <= %d", kMaxCudaK);
  if (bins_per_dim < 1) Rcpp::stop("bins_per_dim must be positive");
  if (!faissr_cuda_available()) Rcpp::stop("No CUDA device is available.");

  IntegerMatrix indices(n, k);
  NumericMatrix distances(n, k);
  int* indices_ptr = indices.begin();
  double* distances_ptr = distances.begin();
  if (include_self) {
    for (int i = 0; i < n; ++i) {
      indices_ptr[static_cast<std::size_t>(i)] = i + 1;
      distances_ptr[static_cast<std::size_t>(i)] = 0.0;
    }
  }

  const int search_k = include_self ? k - 1 : k;
  int n_cells = 0;
  if (search_k > 0) {
    IntegerMatrix search_indices(n, search_k);
    NumericMatrix search_distances(n, search_k);
    const int status = faissr_cuda_grid_self_knn(
      data.begin(),
      n,
      n_features,
      search_k,
      bins_per_dim,
      search_indices.begin(),
      search_distances.begin(),
      &n_cells
    );
    if (status != 0) {
      Rcpp::stop("CUDA grid KNN failed: %s", cuda_error_message());
    }
    const int col_offset = include_self ? 1 : 0;
    for (int col = 0; col < search_k; ++col) {
      for (int row = 0; row < n; ++row) {
        const std::size_t src = static_cast<std::size_t>(col) * n + row;
        const std::size_t dst = static_cast<std::size_t>(col + col_offset) * n + row;
        indices_ptr[dst] = search_indices.begin()[src];
        distances_ptr[dst] = search_distances.begin()[src];
      }
    }
  } else {
    const long long n_cells_ll = n_features == 3 ?
      static_cast<long long>(bins_per_dim) * bins_per_dim * bins_per_dim :
      static_cast<long long>(bins_per_dim) * bins_per_dim;
    if (n_cells_ll > 2147483647LL) {
      Rcpp::stop("CUDA grid KNN requested too many grid cells");
    }
    n_cells = static_cast<int>(n_cells_ll);
  }

  List result = List::create(
    Rcpp::Named("indices") = indices,
    Rcpp::Named("distances") = distances,
    Rcpp::Named("bins_per_dim") = bins_per_dim,
    Rcpp::Named("n_cells") = n_cells,
    Rcpp::Named("self_column_included") = include_self,
    Rcpp::Named("output_layout") = "knn_matrix_final",
    Rcpp::Named("r_side_reshaping") = false,
    Rcpp::Named("accelerator") = "cuda",
    Rcpp::Named("gpu_provider") = "native_cuda_grid",
    Rcpp::Named("device_residency") = "cuda",
    Rcpp::Named("index_residency") = "gpu_grid_offsets_rows",
    Rcpp::Named("gpu_index_resident") = true,
    Rcpp::Named("gpu_index_persistent") = false,
    Rcpp::Named("host_to_device_transfer_strategy") = "native_cuda_grid_device_buffers",
    Rcpp::Named("host_to_device_data_copies") = search_k > 0 ? 1 : 0,
    Rcpp::Named("host_to_device_copies_known") = true,
    Rcpp::Named("device_to_host_result_copies") = search_k > 0 ? 2 : 0,
    Rcpp::Named("device_to_host_result_copies_known") = true,
    Rcpp::Named("query_reuses_device_data") = true,
    Rcpp::Named("result_residency") = "host",
    Rcpp::Named("cpu_fallback") = false,
    Rcpp::Named("cpu_side_result_repair") = false
  );
  result.attr("cuda_kernel") = n_features == 3 ? "grid3d_self_knn" : "grid2d_self_knn";
  return result;
}

List cuda_grid_self_knn_float32_impl(SEXP data,
                                     int k,
                                     int bins_per_dim,
                                     bool include_self) {
  CudaMatrixViewF32 view = make_column_major_float32_view(data, "data");
  const int n = view.nrow;
  const int n_features = view.ncol;
  if (n_features != 2 && n_features != 3) {
    Rcpp::stop("CUDA grid KNN requires a two- or three-column matrix");
  }
  if (n < 2) Rcpp::stop("data must have at least two rows");
  if (k < 1 || k > n) Rcpp::stop("k must be in [1, nrow(data)]");
  if (!include_self && k >= n) {
    Rcpp::stop("k must be in [1, nrow(data) - 1] when excluding self");
  }
  if (k > kMaxCudaK) Rcpp::stop("CUDA backend currently supports k <= %d", kMaxCudaK);
  if (bins_per_dim < 1) Rcpp::stop("bins_per_dim must be positive");
  if (!faissr_cuda_available()) Rcpp::stop("No CUDA device is available.");

  IntegerMatrix indices(n, k);
  NumericMatrix distances(n, k);
  int* indices_ptr = indices.begin();
  double* distances_ptr = distances.begin();
  if (include_self) {
    for (int i = 0; i < n; ++i) {
      indices_ptr[static_cast<std::size_t>(i)] = i + 1;
      distances_ptr[static_cast<std::size_t>(i)] = 0.0;
    }
  }

  const int search_k = include_self ? k - 1 : k;
  int n_cells = 0;
  if (search_k > 0) {
    IntegerMatrix search_indices(n, search_k);
    NumericMatrix search_distances(n, search_k);
    const int status = faissr_cuda_grid_self_knn_float32(
      view.data,
      n,
      n_features,
      search_k,
      bins_per_dim,
      search_indices.begin(),
      search_distances.begin(),
      &n_cells
    );
    if (status != 0) {
      Rcpp::stop("CUDA float32 grid KNN failed: %s", cuda_error_message());
    }
    const int col_offset = include_self ? 1 : 0;
    for (int col = 0; col < search_k; ++col) {
      for (int row = 0; row < n; ++row) {
        const std::size_t src = static_cast<std::size_t>(col) * n + row;
        const std::size_t dst = static_cast<std::size_t>(col + col_offset) * n + row;
        indices_ptr[dst] = search_indices.begin()[src];
        distances_ptr[dst] = search_distances.begin()[src];
      }
    }
  } else {
    const long long n_cells_ll = n_features == 3 ?
      static_cast<long long>(bins_per_dim) * bins_per_dim * bins_per_dim :
      static_cast<long long>(bins_per_dim) * bins_per_dim;
    if (n_cells_ll > 2147483647LL) {
      Rcpp::stop("CUDA grid KNN requested too many grid cells");
    }
    n_cells = static_cast<int>(n_cells_ll);
  }

  List result = List::create(
    Rcpp::Named("indices") = indices,
    Rcpp::Named("distances") = distances,
    Rcpp::Named("bins_per_dim") = bins_per_dim,
    Rcpp::Named("n_cells") = n_cells,
    Rcpp::Named("self_column_included") = include_self,
    Rcpp::Named("output_layout") = "knn_matrix_final",
    Rcpp::Named("r_side_reshaping") = false,
    Rcpp::Named("input_type") = "float32",
    Rcpp::Named("input_layout") = view.layout,
    Rcpp::Named("input_owns_data") = view.owns_data,
    Rcpp::Named("float32_compatibility_conversion") = view.compatibility_conversion,
    Rcpp::Named("accelerator") = "cuda",
    Rcpp::Named("gpu_provider") = "native_cuda_grid",
    Rcpp::Named("device_residency") = "cuda",
    Rcpp::Named("index_residency") = "gpu_grid_offsets_rows",
    Rcpp::Named("gpu_index_resident") = true,
    Rcpp::Named("gpu_index_persistent") = false,
    Rcpp::Named("host_to_device_transfer_strategy") = "direct_float32_cuda_grid_device_buffers",
    Rcpp::Named("host_to_device_data_copies") = search_k > 0 ? 1 : 0,
    Rcpp::Named("host_to_device_copies_known") = true,
    Rcpp::Named("device_to_host_result_copies") = search_k > 0 ? 2 : 0,
    Rcpp::Named("device_to_host_result_copies_known") = true,
    Rcpp::Named("query_reuses_device_data") = true,
    Rcpp::Named("result_residency") = "host",
    Rcpp::Named("cpu_fallback") = false,
    Rcpp::Named("cpu_side_result_repair") = false
  );
  result.attr("cuda_kernel") = n_features == 3 ? "grid3d_self_knn" : "grid2d_self_knn";
  return result;
}
