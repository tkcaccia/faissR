#include <Rcpp.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "vendor/usearch/index_dense.hpp"

using Rcpp::IntegerMatrix;
using Rcpp::List;
using Rcpp::NumericMatrix;

namespace {

struct MatrixViewF32 {
  const float* data = nullptr;
  int nrow = 0;
  int ncol = 0;
  bool owns_data = false;
  std::string layout = "unknown";
  std::vector<float> buffer;
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
    Rcpp::stop("%s dimensions exceed int limits", name);
  }
  return dims;
}

const float* float32_slot_ptr(SEXP slot,
                              const int expected_length,
                              const char* name) {
  if (TYPEOF(slot) == INTSXP) {
    if (Rf_length(slot) != expected_length) {
      Rcpp::stop("%s float32 payload length does not match its dimensions", name);
    }
    return reinterpret_cast<const float*>(INTEGER(slot));
  }
  if (TYPEOF(slot) == RAWSXP) {
    const R_xlen_t expected_bytes =
      static_cast<R_xlen_t>(expected_length) * static_cast<R_xlen_t>(sizeof(float));
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
    if (Rf_length(x) != expected_length) {
      Rcpp::stop("%s payload length does not match its dimensions", name);
    }
    const double* col_major = REAL(x);
    view.buffer.assign(static_cast<std::size_t>(expected_length), 0.0f);
    view.owns_data = true;
    view.layout = "r_double_column_major_to_row_major_float32";
#ifdef _OPENMP
#pragma omp parallel for schedule(static) reduction(&& : finite)
#endif
    for (int r = 0; r < view.nrow; ++r) {
      for (int c = 0; c < view.ncol; ++c) {
        const double value = col_major[r + view.nrow * c];
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
    const float* col_major_f32 = float32_slot_ptr(slot, expected_length, name);
    if (col_major_f32 != nullptr) {
      finite = finite_float32_payload(col_major_f32, expected_length);
      if (view.nrow == 1 || view.ncol == 1) {
        view.data = col_major_f32;
        view.owns_data = false;
        view.layout = "float32_payload_direct_row_compatible";
      } else {
        view.buffer.assign(static_cast<std::size_t>(expected_length), 0.0f);
        view.owns_data = true;
        view.layout = "float32_column_major_payload_to_row_major";
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
        for (int r = 0; r < view.nrow; ++r) {
          for (int c = 0; c < view.ncol; ++c) {
            view.buffer[static_cast<std::size_t>(r) * view.ncol + c] =
              col_major_f32[r + view.nrow * c];
          }
        }
      }
    } else if (TYPEOF(slot) == REALSXP) {
      if (Rf_length(slot) != expected_length) {
        Rcpp::stop("%s payload length does not match its dimensions", name);
      }
      const double* col_major = REAL(slot);
      view.buffer.assign(static_cast<std::size_t>(expected_length), 0.0f);
      view.owns_data = true;
      view.layout = "s4_double_column_major_to_row_major_float32";
#ifdef _OPENMP
#pragma omp parallel for schedule(static) reduction(&& : finite)
#endif
      for (int r = 0; r < view.nrow; ++r) {
        for (int c = 0; c < view.ncol; ++c) {
          const double value = col_major[r + view.nrow * c];
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
    Rcpp::stop("USEARCH input requires finite values");
  }
  if (view.data == nullptr) {
    view.data = view.buffer.data();
  }
  return view;
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

void require_float_namespace() {
  Rcpp::Environment base = Rcpp::Environment::namespace_env("base");
  Rcpp::Function require_namespace = base["requireNamespace"];
  const bool ok = Rcpp::as<bool>(
    require_namespace("float", Rcpp::Named("quietly") = true)
  );
  if (!ok) {
    Rcpp::stop("`output = \"float\"` requires the optional float package");
  }
}

List format_usearch_result(const std::vector<std::uint64_t>& labels,
                           const std::vector<float>& distances,
                           const int n_points,
                           const int search_k,
                           const int out_k,
                           const bool self_query,
                           const bool exclude_self,
                           const int n_threads,
                           const int connectivity,
                           const int expansion_add,
                           const int expansion_search,
                           const bool float_distances) {
  IntegerMatrix indices(n_points, out_k);
  NumericMatrix dists;
  IntegerMatrix float_dists;
  if (float_distances) {
    require_float_namespace();
    float_dists = IntegerMatrix(n_points, out_k);
  } else {
    dists = NumericMatrix(n_points, out_k);
  }

  int* indices_ptr = indices.begin();
  double* dists_ptr = float_distances ? nullptr : dists.begin();
  int* float_dists_ptr = float_distances ? float_dists.begin() : nullptr;
  const bool skip_self = exclude_self && self_query;
  bool complete = true;

#ifdef _OPENMP
#pragma omp parallel for num_threads(n_threads > 0 ? n_threads : 1) schedule(static) reduction(&& : complete)
#endif
  for (int i = 0; i < n_points; ++i) {
    const std::size_t row_offset = static_cast<std::size_t>(i) * search_k;
    int written = 0;
    for (int j = 0; j < search_k && written < out_k; ++j) {
      const std::uint64_t label = labels[row_offset + j];
      if (label > static_cast<std::uint64_t>(std::numeric_limits<int>::max())) {
        continue;
      }
      if (skip_self && label == static_cast<std::uint64_t>(i)) continue;
      const std::size_t output_offset =
        static_cast<std::size_t>(written) * n_points + i;
      indices_ptr[output_offset] = static_cast<int>(label) + 1;
      const double value =
        std::sqrt(std::max(static_cast<double>(distances[row_offset + j]), 0.0));
      if (float_distances) {
        const float value_f = static_cast<float>(value);
        std::memcpy(float_dists_ptr + output_offset, &value_f, sizeof(float));
      } else {
        dists_ptr[output_offset] = value;
      }
      ++written;
    }
    if (written < out_k) complete = false;
  }

  if (!complete) {
    Rcpp::stop("USEARCH returned fewer neighbours than requested");
  }

  SEXP distance_sexp = dists;
  if (float_distances) {
    Rcpp::S4 float_matrix("float32");
    float_matrix.slot("Data") = float_dists;
    distance_sexp = float_matrix;
  }

  List out = List::create(
    Rcpp::Named("indices") = indices,
    Rcpp::Named("distances") = distance_sexp,
    Rcpp::Named("index_type") = "USearchIndexDense",
    Rcpp::Named("exact") = false,
    Rcpp::Named("metric") = "euclidean",
    Rcpp::Named("connectivity") = connectivity,
    Rcpp::Named("expansion_add") = expansion_add,
    Rcpp::Named("expansion_search") = expansion_search,
    Rcpp::Named("n_threads") = n_threads
  );
  if (float_distances) {
    out["distance_type"] = "float32";
    out.attr("distance_type") = "float32";
  }
  return out;
}

} // namespace

// [[Rcpp::export]]
bool usearch_available_cpp() {
  return true;
}

// [[Rcpp::export]]
List nn_usearch_float32_cpp(SEXP data,
                            SEXP points,
                            int k,
                            int connectivity,
                            int expansion_add,
                            int expansion_search,
                            bool exclude_self,
                            int n_threads,
                            std::string distance_storage) {
  if (k < 1) {
    Rcpp::stop("k must be positive");
  }
  n_threads = std::max(1, n_threads);
  connectivity = std::max(2, connectivity);
  expansion_add = std::max(1, expansion_add);
  expansion_search = std::max(1, expansion_search);
  const bool float_distances = distance_storage == "float";
  if (!float_distances && distance_storage != "double") {
    Rcpp::stop("USEARCH distance storage must be 'double' or 'float'");
  }

  MatrixViewF32 xb = make_float32_matrix_view(data, "data");
  MatrixViewF32 xq = make_float32_matrix_view(points, "points");
  if (xb.ncol != xq.ncol) {
    Rcpp::stop("data and points must have the same number of columns");
  }
  const bool self_query = data == points;
  const int n = xb.nrow;
  const int n_points = xq.nrow;
  const int max_k = exclude_self ? n - 1 : n;
  if (k > max_k) {
    Rcpp::stop("k must not exceed the available neighbour count");
  }
  if (exclude_self && !self_query) {
    Rcpp::stop("self-neighbor exclusion requires points to be data");
  }

  const int search_k = exclude_self ? std::min(n, k + 1) : k;
  std::vector<std::uint64_t> labels(
    static_cast<std::size_t>(n_points) * search_k,
    std::numeric_limits<std::uint64_t>::max()
  );
  std::vector<float> distances(
    static_cast<std::size_t>(n_points) * search_k,
    std::numeric_limits<float>::infinity()
  );

  try {
    using namespace unum::usearch;
    metric_punned_t metric(
      static_cast<std::size_t>(xb.ncol),
      metric_kind_t::l2sq_k,
      scalar_kind_t::f32_k
    );
    index_dense_config_t config(
      static_cast<std::size_t>(connectivity),
      static_cast<std::size_t>(expansion_add),
      static_cast<std::size_t>(expansion_search)
    );
    config.enable_key_lookups = false;
    index_limits_t limits(static_cast<std::size_t>(n), static_cast<std::size_t>(n_threads));
    index_dense_t index = index_dense_t::make(metric, config, default_free_value<std::uint64_t>(), limits);

    OmpThreadScope threads(n_threads);
    bool add_ok = true;
#ifdef _OPENMP
#pragma omp parallel for num_threads(n_threads) schedule(static) reduction(&& : add_ok)
#endif
    for (int i = 0; i < n; ++i) {
      const std::size_t thread =
#ifdef _OPENMP
        static_cast<std::size_t>(omp_get_thread_num());
#else
        0;
#endif
      auto status = index.add(
        static_cast<std::uint64_t>(i),
        xb.data + static_cast<std::size_t>(i) * xb.ncol,
        thread,
        false
      );
      if (!status) {
        add_ok = false;
      }
    }
    if (!add_ok) {
      Rcpp::stop("USEARCH add failed");
    }

    bool search_ok = true;
#ifdef _OPENMP
#pragma omp parallel for num_threads(n_threads) schedule(static) reduction(&& : search_ok)
#endif
    for (int i = 0; i < n_points; ++i) {
      const std::size_t thread =
#ifdef _OPENMP
        static_cast<std::size_t>(omp_get_thread_num());
#else
        0;
#endif
      auto matches = index.search(
        xq.data + static_cast<std::size_t>(i) * xq.ncol,
        static_cast<std::size_t>(search_k),
        thread
      );
      if (!matches) {
        search_ok = false;
        continue;
      }
      const std::size_t row_offset = static_cast<std::size_t>(i) * search_k;
      const std::size_t count = matches.dump_to(
        labels.data() + row_offset,
        distances.data() + row_offset,
        static_cast<std::size_t>(search_k)
      );
      if (count < static_cast<std::size_t>(search_k)) {
        for (std::size_t j = count; j < static_cast<std::size_t>(search_k); ++j) {
          labels[row_offset + j] = std::numeric_limits<std::uint64_t>::max();
          distances[row_offset + j] = std::numeric_limits<float>::infinity();
        }
      }
    }
    if (!search_ok) {
      Rcpp::stop("USEARCH search failed");
    }
  } catch (const std::exception& e) {
    Rcpp::stop("USEARCH search failed: %s", e.what());
  }

  List out = format_usearch_result(
    labels,
    distances,
    n_points,
    search_k,
    k,
    self_query,
    exclude_self,
    n_threads,
    connectivity,
    expansion_add,
    expansion_search,
    float_distances
  );
  out["input_type"] = "float32";
  out["input_layout"] = self_query ? xb.layout : (xb.layout + ";" + xq.layout);
  out["input_owns_data"] = xb.owns_data || (!self_query && xq.owns_data);
  out["float32_compatibility_conversion"] = xb.owns_data || (!self_query && xq.owns_data);
  return out;
}
