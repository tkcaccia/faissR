#ifndef FAISSR_SIZE_UTILS_HPP
#define FAISSR_SIZE_UTILS_HPP

#include <Rinternals.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace faissr {

constexpr long long matrix_element_count_wide(const int nrow, const int ncol) {
  return static_cast<long long>(nrow) * static_cast<long long>(ncol);
}

inline R_xlen_t matrix_element_count(const int nrow, const int ncol) {
  const long long elements = matrix_element_count_wide(nrow, ncol);
  if (nrow < 0 || ncol < 0 || elements > R_XLEN_T_MAX) {
    throw std::length_error("matrix dimensions exceed R vector limits");
  }
  return static_cast<R_xlen_t>(elements);
}

constexpr int chunk_bound(const int count,
                          const int part,
                          const int parts) {
  return static_cast<int>(
    static_cast<long long>(count) * static_cast<long long>(part) / parts
  );
}

inline R_xlen_t float_payload_byte_count(const R_xlen_t elements,
                                         const char* name) {
  constexpr R_xlen_t width = static_cast<R_xlen_t>(sizeof(float));
  if (elements < 0 || elements > R_XLEN_T_MAX / width) {
    throw std::length_error(
      std::string(name) + " float32 payload exceeds R vector limits"
    );
  }
  return elements * width;
}

// Pack an existing column-major row window into a caller-owned float32 buffer.
template <typename Source>
void copy_column_rows_to_row_major_float(const Source* source, float* destination,
    std::size_t stride, std::size_t columns, std::size_t first, std::size_t count) {
  constexpr std::size_t tile = 32;
  for (std::size_t row_begin = 0; row_begin < count; row_begin += tile) {
    const auto row_end = std::min(row_begin + tile, count);
    for (std::size_t col_begin = 0; col_begin < columns; col_begin += tile) {
      const auto col_end = std::min(col_begin + tile, columns);
      for (auto col = col_begin; col < col_end; ++col) {
        const auto* source_column = source + col * stride + first;
        for (auto row = row_begin; row < row_end; ++row) {
          destination[row * columns + col] = static_cast<float>(source_column[row]);
        }
      }
    }
  }
}

template <typename Source>
bool copy_column_major_to_row_major_float(
    const Source* source,
    std::vector<float>& destination,
    const int nrow,
    const int ncol,
    const bool check_finite = true,
    const int n_threads = 1) {
  constexpr long long tile = 32;
  const long long rows = nrow;
  const long long columns = ncol;
  const R_xlen_t elements = matrix_element_count(nrow, ncol);
  destination.assign(static_cast<std::size_t>(elements), 0.0f);
  bool finite = true;
#ifdef _OPENMP
#pragma omp parallel for num_threads(n_threads > 0 ? n_threads : 1) if(elements >= 65536 && n_threads > 1) schedule(static) reduction(&& : finite)
#else
  (void)n_threads;
#endif
  for (long long row_begin = 0; row_begin < rows; row_begin += tile) {
    const long long row_end =
      row_begin + tile < rows ? row_begin + tile : rows;
    for (long long column_begin = 0;
         column_begin < columns;
         column_begin += tile) {
      const long long column_end =
        column_begin + tile < columns ? column_begin + tile : columns;
      for (long long column = column_begin;
           column < column_end;
           ++column) {
        const Source* source_column = source +
          static_cast<std::size_t>(column) * nrow;
        for (long long row = row_begin; row < row_end; ++row) {
          const Source value = source_column[row];
          const float converted = static_cast<float>(value);
          if (check_finite && (!std::isfinite(value) || !std::isfinite(converted))) {
            finite = false;
            continue;
          }
          destination[
            static_cast<std::size_t>(row) * ncol + column
          ] = converted;
        }
      }
    }
  }
  return finite;
}

static_assert(
  matrix_element_count_wide(std::numeric_limits<int>::max(), 2) == 4294967294LL,
  "matrix element counts must use long-vector arithmetic"
);
static_assert(
  chunk_bound(std::numeric_limits<int>::max(), 64, 64) ==
    std::numeric_limits<int>::max(),
  "thread partition bounds must not overflow int multiplication"
);

}  // namespace faissr

#endif
