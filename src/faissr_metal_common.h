#ifndef FAISSR_METAL_COMMON_H
#define FAISSR_METAL_COMMON_H

#include <Rcpp.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <string>
#include <utility>
#include <vector>

namespace faissr_metal {

constexpr int kMaxIvfDimensions = 1024;
constexpr std::uint32_t kCagraDegree = 64;
constexpr std::uint32_t kCagraBalancedDegree = 48;

inline bool cagra_balanced_shape(std::uint64_t rows, int columns,
                                 int k, double target) {
  return rows >= 5000000 && rows <= std::numeric_limits<int>::max() &&
    columns > 0 && columns <= 128 && k > 0 &&
    k < static_cast<int>(kCagraBalancedDegree) && target == 0.99;
}

inline double cagra_workspace_bytes(std::uint64_t rows, int columns,
    int projected, double shortlist_bytes, int degree) {
  // The seed and reverse buffers store two and half as many edges per row.
  constexpr auto row_auxiliary = 4;
  constexpr double reserve = 128.0 * 1024 * 1024;
  return 4.0 * rows * (columns + projected + 3.5 * degree + row_auxiliary) +
    shortlist_bytes + reserve;
}

inline int ivf_projection_dimensions(int columns, bool preserve_geometry) {
  int width = 1;
  while (width * 2 <= columns && width < 128) width *= 2;
  if (preserve_geometry && columns <= 128 && width < columns) width *= 2;
  return width;
}

inline bool ivf_shape_supported(std::uint64_t rows, int columns, int k) {
  // IVF's projection, reranking and result buffers use 32-bit offsets.
  return columns > 0 && columns <= kMaxIvfDimensions && k > 0 &&
    rows <= std::numeric_limits<std::uint32_t>::max() /
      static_cast<std::uint32_t>(std::max(columns, k));
}

inline bool cagra_shape_supported(std::uint64_t rows, int columns, int k) {
  return rows > 128 && rows <= std::numeric_limits<int>::max() &&
    columns > 0 && columns <= kMaxIvfDimensions && k > 0 && k <= 128;
}

inline std::uint32_t cagra_centroid_batch_rows(std::uint32_t rows,
    std::uint32_t projected) {
  constexpr std::uint32_t max_wide_batch = 1u << 20;
  const auto max_elements = std::numeric_limits<std::uint32_t>::max();
  return rows <= max_elements / projected ? rows :
    std::min(max_wide_batch, max_elements / projected);
}

enum class KnnMetric {
  Euclidean,
  Cosine,
  Correlation
};

inline int adaptive_worker_count(int requested, int work_items) {
  if (work_items <= 1) return 1;
  requested = std::max(1, requested);
  const int cap = work_items < 256 ? 1 :
    work_items < 2048 ? 2 : requested;
  return std::max(1, std::min(std::min(requested, cap), work_items));
}

inline KnnMetric parse_knn_metric(const std::string& metric) {
  if (metric == "euclidean") return KnnMetric::Euclidean;
  if (metric == "cosine") return KnnMetric::Cosine;
  if (metric == "correlation") return KnnMetric::Correlation;
  Rcpp::stop("Unsupported native KNN metric `%s`.", metric.c_str());
}

inline bool is_float32_matrix(SEXP data) {
  if (!Rf_isS4(data)) return false;
  Rcpp::S4 object(data);
  return object.is("float32");
}

inline float int_bits_to_float(int bits) {
  float value = 0.0f;
  static_assert(sizeof(value) == sizeof(bits), "float32 payload must use 32-bit storage");
  std::memcpy(&value, &bits, sizeof(value));
  return value;
}

struct FloatMatrix {
  std::vector<float> values;
  int nrow = 0;
  int ncol = 0;
  bool input_float32 = false;
};

inline void normalize_rows(FloatMatrix& matrix, bool center) {
  for (int i = 0; i < matrix.nrow; ++i) {
    float* row = matrix.values.data() + static_cast<std::size_t>(i) * matrix.ncol;
    double mean = 0.0;
    if (center) {
      for (int j = 0; j < matrix.ncol; ++j) mean += row[j];
      mean /= std::max(1, matrix.ncol);
    }
    double squared_norm = 0.0;
    for (int j = 0; j < matrix.ncol; ++j) {
      row[j] = static_cast<float>(row[j] - mean);
      squared_norm += static_cast<double>(row[j]) * row[j];
    }
    if (squared_norm <= 0.0) continue;
    const float inverse_norm = static_cast<float>(1.0 / std::sqrt(squared_norm));
    for (int j = 0; j < matrix.ncol; ++j) row[j] *= inverse_norm;
  }
}

template <typename Source, typename Convert>
inline void transpose_to_float(const Source* source,
                               FloatMatrix& result,
                               Convert convert) {
  constexpr int tile = 32;
  for (int row = 0; row < result.nrow; row += tile) {
    const int row_end = std::min(row + tile, result.nrow);
    for (int col = 0; col < result.ncol; col += tile) {
      const int col_end = std::min(col + tile, result.ncol);
      for (int j = col; j < col_end; ++j) {
        for (int i = row; i < row_end; ++i) {
          result.values[static_cast<std::size_t>(i) * result.ncol + j] =
            convert(source[i + static_cast<std::size_t>(j) * result.nrow]);
        }
      }
    }
  }
}

inline FloatMatrix matrix_to_row_major_float(SEXP data, KnnMetric metric) {
  FloatMatrix result;
  result.input_float32 = is_float32_matrix(data);
  if (result.input_float32) {
    Rcpp::S4 object(data);
    SEXP payload_sexp = object.slot("Data");
    if (TYPEOF(payload_sexp) != INTSXP || !Rf_isMatrix(payload_sexp)) {
      Rcpp::stop("Invalid float::float32 matrix payload.");
    }
    Rcpp::IntegerMatrix payload(payload_sexp);
    result.nrow = payload.nrow();
    result.ncol = payload.ncol();
    result.values.resize(static_cast<std::size_t>(result.nrow) * result.ncol);
    const int* source = INTEGER(payload);
    transpose_to_float(source, result, int_bits_to_float);
  } else {
    SEXP dims = Rf_getAttrib(data, R_DimSymbol);
    if (TYPEOF(dims) != INTSXP || Rf_length(dims) != 2) {
      Rcpp::stop("`data` must be an integer, numeric, or float::float32 matrix.");
    }
    result.nrow = INTEGER(dims)[0];
    result.ncol = INTEGER(dims)[1];
    result.values.resize(static_cast<std::size_t>(result.nrow) * result.ncol);
    if (TYPEOF(data) == INTSXP) {
      const int* source = INTEGER(data);
      transpose_to_float(source, result, [](int value) {
        return static_cast<float>(value);
      });
    } else if (TYPEOF(data) == REALSXP) {
      const double* source = REAL(data);
      transpose_to_float(source, result, [](double value) {
        return static_cast<float>(value);
      });
    } else {
      Rcpp::stop("`data` must be an integer, numeric, or float::float32 matrix.");
    }
  }
  if (metric == KnnMetric::Cosine) normalize_rows(result, false);
  if (metric == KnnMetric::Correlation) normalize_rows(result, true);
  return result;
}

inline void require_finite_matrix(const FloatMatrix& matrix) {
  for (float value : matrix.values) {
    if (!std::isfinite(value)) {
      Rcpp::stop("Native KNN requires finite input.");
    }
  }
}

inline float squared_l2_distance(const float* lhs,
                                 const float* rhs,
                                 int p) {
  float sum0 = 0.0f;
  float sum1 = 0.0f;
  float sum2 = 0.0f;
  float sum3 = 0.0f;
  int d = 0;
  for (; d + 15 < p; d += 16) {
    for (int offset = 0; offset < 16; offset += 4) {
      const float x0 = lhs[d + offset] - rhs[d + offset];
      const float x1 = lhs[d + offset + 1] - rhs[d + offset + 1];
      const float x2 = lhs[d + offset + 2] - rhs[d + offset + 2];
      const float x3 = lhs[d + offset + 3] - rhs[d + offset + 3];
      sum0 += x0 * x0;
      sum1 += x1 * x1;
      sum2 += x2 * x2;
      sum3 += x3 * x3;
    }
  }
  float sum = (sum0 + sum1) + (sum2 + sum3);
  for (; d < p; ++d) {
    const float delta = lhs[d] - rhs[d];
    sum += delta * delta;
  }
  return sum;
}

inline float output_distance(float internal_distance, KnnMetric metric) {
  if (metric == KnnMetric::Euclidean) {
    return std::sqrt(std::max(0.0f, internal_distance));
  }
  if (metric == KnnMetric::Cosine || metric == KnnMetric::Correlation) {
    return 0.5f * std::max(0.0f, internal_distance);
  }
  Rcpp::stop("Unsupported native KNN metric.");
}

} // namespace faissr_metal

#endif
