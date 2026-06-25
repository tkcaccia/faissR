#include <Rcpp.h>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

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
int faissr_cuda_landmark_candidate_knn(const double* data,
                                           const int* projection_indices,
                                           int n,
                                           int n_features,
                                           int projection_k,
                                           int k,
                                           int bucket_cols,
                                           int query_cols,
                                           int* out_indices,
                                           double* out_distances);
int faissr_cuda_row_candidate_knn(const double* data,
                                      const int* candidate_indices,
                                      int n,
                                      int n_features,
                                      int n_candidates,
                                      int k,
                                      int metric_kind,
                                      int* out_indices,
                                      double* out_distances);
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
}

namespace {

constexpr int kMaxCudaK = 256;

struct CudaMatrixViewF32 {
  const float* data = nullptr;
  int nrow = 0;
  int ncol = 0;
  bool owns_data = false;
  bool compatibility_conversion = false;
  std::string layout;
  std::vector<float> buffer;
};

const char* cuda_error_message() {
  const char* msg = faissr_cuda_last_error();
  return msg == nullptr ? "unknown CUDA error" : msg;
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

const float* float32_slot_ptr_f32(SEXP slot, const int expected_length, const char* name) {
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

bool finite_float32_payload_f32(const float* ptr, const int length) {
  for (int i = 0; i < length; ++i) {
    if (!std::isfinite(ptr[i])) return false;
  }
  return true;
}

CudaMatrixViewF32 make_column_major_float32_view(SEXP x, const char* name) {
  IntegerVector dims = matrix_dims_from_object_f32(x, name);
  CudaMatrixViewF32 view;
  view.nrow = dims[0];
  view.ncol = dims[1];
  const int expected_length = view.nrow * view.ncol;

  bool finite = true;
  if (TYPEOF(x) == REALSXP) {
    if (Rf_length(x) != expected_length) {
      Rcpp::stop("%s payload length does not match its dimensions", name);
    }
    view.buffer.assign(static_cast<std::size_t>(expected_length), 0.0f);
    view.owns_data = true;
    view.compatibility_conversion = true;
    view.layout = "r_double_column_major_to_column_major_float32";
    const double* src = REAL(x);
    for (int i = 0; i < expected_length; ++i) {
      if (!std::isfinite(src[i])) finite = false;
      view.buffer[static_cast<std::size_t>(i)] = static_cast<float>(src[i]);
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
      if (Rf_length(slot) != expected_length) {
        Rcpp::stop("%s payload length does not match its dimensions", name);
      }
      view.buffer.assign(static_cast<std::size_t>(expected_length), 0.0f);
      view.owns_data = true;
      view.compatibility_conversion = true;
      view.layout = "s4_double_column_major_to_column_major_float32";
      const double* src = REAL(slot);
      for (int i = 0; i < expected_length; ++i) {
        if (!std::isfinite(src[i])) finite = false;
        view.buffer[static_cast<std::size_t>(i)] = static_cast<float>(src[i]);
      }
    } else {
      Rcpp::stop("%s must be a float::fl()/float32 object with an integer or raw @Data payload", name);
    }
  } else {
    Rcpp::stop("%s must be an ordinary R double matrix or float::fl()/float32 object", name);
  }
  if (!finite) Rcpp::stop("%s requires finite values", name);
  if (view.data == nullptr) view.data = view.buffer.data();
  return view;
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

List cuda_landmark_candidate_knn_impl(NumericMatrix data,
                                      IntegerMatrix projection_indices,
                                      int k,
                                      int bucket_cols,
                                      int query_cols) {
  const int n = data.nrow();
  const int n_features = data.ncol();
  const int projection_k = projection_indices.ncol();
  if (n < 2) Rcpp::stop("data must have at least two rows");
  if (projection_indices.nrow() != n) {
    Rcpp::stop("projection_indices row count must match data");
  }
  if (projection_k < 1) Rcpp::stop("projection_indices must have at least one column");
  if (k < 1 || k >= n) Rcpp::stop("k must be in [1, nrow(data) - 1]");
  if (k > kMaxCudaK) Rcpp::stop("CUDA backend currently supports k <= %d", kMaxCudaK);
  if (!faissr_cuda_available()) Rcpp::stop("No CUDA device is available.");

  for (int i = 0; i < n; ++i) {
    for (int c = 0; c < projection_k; ++c) {
      if (projection_indices(i, c) < 1) {
        Rcpp::stop("projection_indices must be 1-based positive integers");
      }
    }
  }

  IntegerMatrix indices(n, k);
  NumericMatrix distances(n, k);
  const int status = faissr_cuda_landmark_candidate_knn(
    data.begin(),
    projection_indices.begin(),
    n,
    n_features,
    projection_k,
    k,
    bucket_cols,
    query_cols,
    indices.begin(),
    distances.begin()
  );
  if (status != 0) {
    Rcpp::stop("CUDA landmark candidate KNN failed: %s", cuda_error_message());
  }
  return List::create(
    Rcpp::Named("indices") = indices,
    Rcpp::Named("distances") = distances
  );
}

List cuda_row_candidate_knn_impl(NumericMatrix data,
                                 IntegerMatrix candidate_indices,
                                 int k,
                                 std::string metric) {
  const int n = data.nrow();
  const int n_features = data.ncol();
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
  const int status = faissr_cuda_row_candidate_knn(
    data.begin(),
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
    Rcpp::stop("CUDA row candidate KNN failed: %s", cuda_error_message());
  }
  List result = List::create(
    Rcpp::Named("indices") = indices,
    Rcpp::Named("distances") = distances
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
    Rcpp::Named("float32_compatibility_conversion") = data_view.compatibility_conversion
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
