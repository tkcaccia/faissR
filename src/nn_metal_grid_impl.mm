#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <Rcpp.h>

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

namespace {

constexpr int kMaxMetalGridK = 128;

struct MetalGridParams {
  std::uint32_t n;
  std::uint32_t n_features;
  std::uint32_t k;
  std::uint32_t bins;
  float min_x;
  float min_y;
  float min_z;
  float cell_x;
  float cell_y;
  float cell_z;
};

struct MetalGridIndex {
  int n_features = 2;
  int bins = 1;
  int n_cells = 1;
  float min_x = 0.0f;
  float min_y = 0.0f;
  float min_z = 0.0f;
  float cell_x = 1.0f;
  float cell_y = 1.0f;
  float cell_z = 1.0f;
  std::vector<int> offsets;
  std::vector<int> rows;
};

struct MetalGridState {
  id<MTLDevice> device = nil;
  id<MTLLibrary> library = nil;
  id<MTLComputePipelineState> pipeline = nil;
  id<MTLCommandQueue> queue = nil;
};

const char* metal_grid_source() {
  return R"METAL(
#include <metal_stdlib>
using namespace metal;

constant uint max_grid_k = 128u;

struct GridParams {
  uint n;
  uint n_features;
  uint k;
  uint bins;
  float min_x;
  float min_y;
  float min_z;
  float cell_x;
  float cell_y;
  float cell_z;
};

inline int grid_coord(float value, float min_value, float cell_size, uint bins) {
  int out = int((value - min_value) / cell_size);
  if (out < 0) out = 0;
  if (out >= int(bins)) out = int(bins) - 1;
  return out;
}

inline int grid_cell_2d(int x, int y, uint bins) {
  return y * int(bins) + x;
}

inline int grid_cell_3d(int x, int y, int z, uint bins) {
  return (z * int(bins) + y) * int(bins) + x;
}

inline void insert_candidate(float distance,
                             int index,
                             thread float* best_distance,
                             thread int* best_index,
                             uint k) {
  if (distance > best_distance[k - 1u] ||
      (distance == best_distance[k - 1u] && index >= best_index[k - 1u])) {
    return;
  }
  int position = int(k) - 1;
  while (position > 0 &&
         (distance < best_distance[position - 1] ||
          (distance == best_distance[position - 1] &&
           index < best_index[position - 1]))) {
    best_distance[position] = best_distance[position - 1];
    best_index[position] = best_index[position - 1];
    --position;
  }
  best_distance[position] = distance;
  best_index[position] = index;
}

inline float lower_outside_2d(float x,
                              float y,
                              constant GridParams& params,
                              int x0,
                              int x1,
                              int y0,
                              int y1) {
  float best = MAXFLOAT;
  if (x0 > 0) {
    const float border = params.min_x + float(x0) * params.cell_x;
    const float delta = max(0.0f, x - border);
    best = min(best, delta * delta);
  }
  if (x1 + 1 < int(params.bins)) {
    const float border = params.min_x + float(x1 + 1) * params.cell_x;
    const float delta = max(0.0f, border - x);
    best = min(best, delta * delta);
  }
  if (y0 > 0) {
    const float border = params.min_y + float(y0) * params.cell_y;
    const float delta = max(0.0f, y - border);
    best = min(best, delta * delta);
  }
  if (y1 + 1 < int(params.bins)) {
    const float border = params.min_y + float(y1 + 1) * params.cell_y;
    const float delta = max(0.0f, border - y);
    best = min(best, delta * delta);
  }
  return best;
}

inline float lower_outside_3d(float x,
                              float y,
                              float z,
                              constant GridParams& params,
                              int x0,
                              int x1,
                              int y0,
                              int y1,
                              int z0,
                              int z1) {
  float best = lower_outside_2d(x, y, params, x0, x1, y0, y1);
  if (z0 > 0) {
    const float border = params.min_z + float(z0) * params.cell_z;
    const float delta = max(0.0f, z - border);
    best = min(best, delta * delta);
  }
  if (z1 + 1 < int(params.bins)) {
    const float border = params.min_z + float(z1 + 1) * params.cell_z;
    const float delta = max(0.0f, border - z);
    best = min(best, delta * delta);
  }
  return best;
}

inline void add_cell_2d(device const float* data,
                        device const int* offsets,
                        device const int* rows,
                        constant GridParams& params,
                        uint query,
                        int x,
                        int y,
                        thread float* best_distance,
                        thread int* best_index) {
  if (x < 0 || y < 0 || x >= int(params.bins) || y >= int(params.bins)) return;
  const int cell = grid_cell_2d(x, y, params.bins);
  const float qx = data[query];
  const float qy = data[params.n + query];
  for (int position = offsets[cell]; position < offsets[cell + 1]; ++position) {
    const int candidate = rows[position];
    if (candidate == int(query)) continue;
    const float dx = qx - data[candidate];
    const float dy = qy - data[params.n + uint(candidate)];
    insert_candidate(dx * dx + dy * dy, candidate,
                     best_distance, best_index, params.k);
  }
}

inline void add_cell_3d(device const float* data,
                        device const int* offsets,
                        device const int* rows,
                        constant GridParams& params,
                        uint query,
                        int x,
                        int y,
                        int z,
                        thread float* best_distance,
                        thread int* best_index) {
  if (x < 0 || y < 0 || z < 0 ||
      x >= int(params.bins) || y >= int(params.bins) || z >= int(params.bins)) return;
  const int cell = grid_cell_3d(x, y, z, params.bins);
  const float qx = data[query];
  const float qy = data[params.n + query];
  const float qz = data[2u * params.n + query];
  for (int position = offsets[cell]; position < offsets[cell + 1]; ++position) {
    const int candidate = rows[position];
    if (candidate == int(query)) continue;
    const uint candidate_u = uint(candidate);
    const float dx = qx - data[candidate_u];
    const float dy = qy - data[params.n + candidate_u];
    const float dz = qz - data[2u * params.n + candidate_u];
    insert_candidate(dx * dx + dy * dy + dz * dz, candidate,
                     best_distance, best_index, params.k);
  }
}

kernel void exact_grid_self_knn(
    device const float* data [[buffer(0)]],
    device const int* offsets [[buffer(1)]],
    device const int* rows [[buffer(2)]],
    device int* output_index [[buffer(3)]],
    device float* output_distance [[buffer(4)]],
    constant GridParams& params [[buffer(5)]],
    uint query [[thread_position_in_grid]]) {
  if (query >= params.n || params.k == 0u || params.k > max_grid_k) return;

  float best_distance[128];
  int best_index[128];
  for (uint j = 0u; j < params.k; ++j) {
    best_distance[j] = MAXFLOAT;
    best_index[j] = INT_MAX;
  }

  const float qx = data[query];
  const float qy = data[params.n + query];
  const int cx = grid_coord(qx, params.min_x, params.cell_x, params.bins);
  const int cy = grid_coord(qy, params.min_y, params.cell_y, params.bins);

  if (params.n_features == 2u) {
    for (int radius = 0; radius <= int(params.bins); ++radius) {
      const int raw_x0 = cx - radius;
      const int raw_x1 = cx + radius;
      const int raw_y0 = cy - radius;
      const int raw_y1 = cy + radius;
      const int x0 = max(0, raw_x0);
      const int x1 = min(int(params.bins) - 1, raw_x1);
      const int y0 = max(0, raw_y0);
      const int y1 = min(int(params.bins) - 1, raw_y1);
      if (radius == 0) {
        add_cell_2d(data, offsets, rows, params, query, cx, cy,
                    best_distance, best_index);
      } else {
        for (int x = raw_x0; x <= raw_x1; ++x) {
          if (x < 0 || x >= int(params.bins)) continue;
          if (raw_y0 >= 0 && raw_y0 < int(params.bins)) {
            add_cell_2d(data, offsets, rows, params, query, x, raw_y0,
                        best_distance, best_index);
          }
          if (raw_y1 != raw_y0 && raw_y1 >= 0 && raw_y1 < int(params.bins)) {
            add_cell_2d(data, offsets, rows, params, query, x, raw_y1,
                        best_distance, best_index);
          }
        }
        for (int y = raw_y0 + 1; y <= raw_y1 - 1; ++y) {
          if (y < 0 || y >= int(params.bins)) continue;
          if (raw_x0 >= 0 && raw_x0 < int(params.bins)) {
            add_cell_2d(data, offsets, rows, params, query, raw_x0, y,
                        best_distance, best_index);
          }
          if (raw_x1 != raw_x0 && raw_x1 >= 0 && raw_x1 < int(params.bins)) {
            add_cell_2d(data, offsets, rows, params, query, raw_x1, y,
                        best_distance, best_index);
          }
        }
      }
      if (best_index[params.k - 1u] != INT_MAX) {
        const float lower = lower_outside_2d(qx, qy, params, x0, x1, y0, y1);
        if (lower > best_distance[params.k - 1u]) break;
      }
    }
  } else {
    const float qz = data[2u * params.n + query];
    const int cz = grid_coord(qz, params.min_z, params.cell_z, params.bins);
    for (int radius = 0; radius <= int(params.bins); ++radius) {
      const int raw_x0 = cx - radius;
      const int raw_x1 = cx + radius;
      const int raw_y0 = cy - radius;
      const int raw_y1 = cy + radius;
      const int raw_z0 = cz - radius;
      const int raw_z1 = cz + radius;
      const int x0 = max(0, raw_x0);
      const int x1 = min(int(params.bins) - 1, raw_x1);
      const int y0 = max(0, raw_y0);
      const int y1 = min(int(params.bins) - 1, raw_y1);
      const int z0 = max(0, raw_z0);
      const int z1 = min(int(params.bins) - 1, raw_z1);
      if (radius == 0) {
        add_cell_3d(data, offsets, rows, params, query, cx, cy, cz,
                    best_distance, best_index);
      } else {
        for (int z = raw_z0; z <= raw_z1; ++z) {
          if (z < 0 || z >= int(params.bins)) continue;
          for (int y = raw_y0; y <= raw_y1; ++y) {
            if (y < 0 || y >= int(params.bins)) continue;
            for (int x = raw_x0; x <= raw_x1; ++x) {
              if (x < 0 || x >= int(params.bins)) continue;
              if (x != raw_x0 && x != raw_x1 &&
                  y != raw_y0 && y != raw_y1 &&
                  z != raw_z0 && z != raw_z1) continue;
              add_cell_3d(data, offsets, rows, params, query, x, y, z,
                          best_distance, best_index);
            }
          }
        }
      }
      if (best_index[params.k - 1u] != INT_MAX) {
        const float lower = lower_outside_3d(
          qx, qy, qz, params, x0, x1, y0, y1, z0, z1);
        if (lower > best_distance[params.k - 1u]) break;
      }
    }
  }

  for (uint j = 0u; j < params.k; ++j) {
    const ulong output = ulong(j) * ulong(params.n) + ulong(query);
    output_index[output] = best_index[j] + 1;
    output_distance[output] = sqrt(max(0.0f, best_distance[j]));
  }
}
)METAL";
}

std::string metal_error(NSError* error) {
  if (error == nil) return "unknown Metal error";
  NSString* description = [error localizedDescription];
  return description == nil ? "unknown Metal error" :
    std::string([description UTF8String]);
}

MetalGridState& metal_grid_state() {
  static MetalGridState state;
  if (state.device != nil && state.pipeline != nil && state.queue != nil) return state;
  state.device = MTLCreateSystemDefaultDevice();
  if (state.device == nil) Rcpp::stop("No Metal device is available.");
  NSError* error = nil;
  NSString* source = [NSString stringWithUTF8String:metal_grid_source()];
  state.library = [state.device newLibraryWithSource:source options:nil error:&error];
  if (state.library == nil) {
    Rcpp::stop("Failed to compile Metal grid KNN kernels: %s",
               metal_error(error).c_str());
  }
  id<MTLFunction> function = [state.library newFunctionWithName:@"exact_grid_self_knn"];
  if (function == nil) Rcpp::stop("Failed to load Metal grid KNN kernel.");
  state.pipeline = [state.device newComputePipelineStateWithFunction:function error:&error];
  [function release];
  if (state.pipeline == nil) {
    Rcpp::stop("Failed to create Metal grid KNN pipeline: %s",
               metal_error(error).c_str());
  }
  state.queue = [state.device newCommandQueue];
  if (state.queue == nil) Rcpp::stop("Failed to create Metal grid KNN command queue.");
  return state;
}

Rcpp::IntegerVector matrix_dimensions(SEXP x) {
  SEXP dimensions = Rf_getAttrib(x, R_DimSymbol);
  if (Rf_isNull(dimensions) && Rf_isS4(x)) {
    SEXP payload = R_do_slot(x, Rf_install("Data"));
    dimensions = Rf_getAttrib(payload, R_DimSymbol);
  }
  if (Rf_isNull(dimensions) || Rf_length(dimensions) != 2) {
    Rcpp::stop("data must be a two-dimensional numeric or float32 matrix");
  }
  return Rcpp::IntegerVector(dimensions);
}

std::vector<float> matrix_float32_column_major(SEXP x,
                                                int n,
                                                int p,
                                                std::string& input_type) {
  const std::size_t size = static_cast<std::size_t>(n) * p;
  std::vector<float> values(size);
  if (TYPEOF(x) == REALSXP) {
    if (Rf_xlength(x) != static_cast<R_xlen_t>(size)) {
      Rcpp::stop("data payload length does not match its dimensions");
    }
    const double* source = REAL(x);
    for (std::size_t i = 0; i < size; ++i) values[i] = static_cast<float>(source[i]);
    input_type = "double_to_float32";
  } else if (Rf_isS4(x)) {
    SEXP payload = R_do_slot(x, Rf_install("Data"));
    const float* source = nullptr;
    if (TYPEOF(payload) == INTSXP) {
      if (Rf_xlength(payload) != static_cast<R_xlen_t>(size)) {
        Rcpp::stop("float32 payload length does not match its dimensions");
      }
      source = reinterpret_cast<const float*>(INTEGER(payload));
    } else if (TYPEOF(payload) == RAWSXP) {
      if (Rf_xlength(payload) != static_cast<R_xlen_t>(size * sizeof(float))) {
        Rcpp::stop("float32 payload length does not match its dimensions");
      }
      source = reinterpret_cast<const float*>(RAW(payload));
    } else {
      Rcpp::stop("Unsupported float32 payload representation");
    }
    const bool row_major =
      Rf_asLogical(Rf_getAttrib(x, Rf_install("faissR_row_major_float32"))) == TRUE;
    if (row_major && n > 1 && p > 1) {
      for (int row = 0; row < n; ++row) {
        for (int column = 0; column < p; ++column) {
          values[static_cast<std::size_t>(column) * n + row] =
            source[static_cast<std::size_t>(row) * p + column];
        }
      }
      input_type = "float32_row_major_to_column_major";
    } else {
      std::memcpy(values.data(), source, size * sizeof(float));
      input_type = "float32";
    }
  } else {
    Rcpp::stop("data must be an ordinary R numeric matrix or float::float32 matrix");
  }
  for (const float value : values) {
    if (!std::isfinite(value)) Rcpp::stop("data must contain only finite values");
  }
  return values;
}

int grid_coordinate(float value, float minimum, float cell, int bins) {
  int coordinate = static_cast<int>((value - minimum) / cell);
  return std::max(0, std::min(bins - 1, coordinate));
}

MetalGridIndex build_grid(const std::vector<float>& data,
                          int n,
                          int p,
                          int bins) {
  MetalGridIndex grid;
  grid.n_features = p;
  grid.bins = bins;
  const long long cells = p == 3 ?
    static_cast<long long>(bins) * bins * bins :
    static_cast<long long>(bins) * bins;
  if (cells < 1 || cells > std::numeric_limits<int>::max() - 1LL) {
    Rcpp::stop("Metal grid KNN requested too many grid cells");
  }
  grid.n_cells = static_cast<int>(cells);

  auto coordinate_range = [&](int column) {
    const float* begin = data.data() + static_cast<std::size_t>(column) * n;
    return std::minmax_element(begin, begin + n);
  };
  const auto x_range = coordinate_range(0);
  const auto y_range = coordinate_range(1);
  grid.min_x = *x_range.first;
  grid.min_y = *y_range.first;
  grid.cell_x = std::nextafterf(
    std::max(*x_range.second - grid.min_x, FLT_EPSILON), FLT_MAX) / bins;
  grid.cell_y = std::nextafterf(
    std::max(*y_range.second - grid.min_y, FLT_EPSILON), FLT_MAX) / bins;
  if (p == 3) {
    const auto z_range = coordinate_range(2);
    grid.min_z = *z_range.first;
    grid.cell_z = std::nextafterf(
      std::max(*z_range.second - grid.min_z, FLT_EPSILON), FLT_MAX) / bins;
  }

  grid.offsets.assign(static_cast<std::size_t>(grid.n_cells + 1), 0);
  std::vector<int> cell_id(static_cast<std::size_t>(n));
  for (int row = 0; row < n; ++row) {
    const int x = grid_coordinate(data[row], grid.min_x, grid.cell_x, bins);
    const int y = grid_coordinate(data[static_cast<std::size_t>(n) + row],
                                  grid.min_y, grid.cell_y, bins);
    const int z = p == 3 ?
      grid_coordinate(data[static_cast<std::size_t>(2) * n + row],
                      grid.min_z, grid.cell_z, bins) : 0;
    const int cell = p == 3 ? (z * bins + y) * bins + x : y * bins + x;
    cell_id[static_cast<std::size_t>(row)] = cell;
    ++grid.offsets[static_cast<std::size_t>(cell + 1)];
  }
  for (int cell = 1; cell <= grid.n_cells; ++cell) {
    grid.offsets[static_cast<std::size_t>(cell)] +=
      grid.offsets[static_cast<std::size_t>(cell - 1)];
  }
  grid.rows.assign(static_cast<std::size_t>(n), 0);
  std::vector<int> cursor = grid.offsets;
  for (int row = 0; row < n; ++row) {
    const int cell = cell_id[static_cast<std::size_t>(row)];
    grid.rows[static_cast<std::size_t>(cursor[static_cast<std::size_t>(cell)]++)] = row;
  }
  return grid;
}

id<MTLBuffer> shared_buffer(id<MTLDevice> device,
                            const void* bytes,
                            std::size_t length,
                            const char* role) {
  id<MTLBuffer> buffer = [device newBufferWithLength:length
                                             options:MTLResourceStorageModeShared];
  if (buffer == nil) Rcpp::stop("Failed to allocate Metal %s buffer.", role);
  if (bytes != nullptr && length > 0) std::memcpy([buffer contents], bytes, length);
  return buffer;
}

void dispatch_grid(id<MTLComputeCommandEncoder> encoder,
                   id<MTLComputePipelineState> pipeline,
                   std::size_t count) {
  const NSUInteger width = std::min<NSUInteger>(
    pipeline.maxTotalThreadsPerThreadgroup,
    std::max<NSUInteger>(1, pipeline.threadExecutionWidth));
  [encoder setComputePipelineState:pipeline];
  [encoder dispatchThreads:MTLSizeMake(count, 1, 1)
      threadsPerThreadgroup:MTLSizeMake(width, 1, 1)];
}

}  // namespace

bool metal_grid_available_impl() {
  @autoreleasepool {
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    const bool available = device != nil;
    [device release];
    return available;
  }
}

Rcpp::List metal_grid_self_knn_impl(SEXP data,
                                    int k,
                                    int bins_per_dim,
                                    bool include_self) {
  @autoreleasepool {
    const Rcpp::IntegerVector dimensions = matrix_dimensions(data);
    const int n = dimensions[0];
    const int p = dimensions[1];
    if (p != 2 && p != 3) {
      Rcpp::stop("Metal grid KNN requires a two- or three-column matrix");
    }
    if (n < 2) Rcpp::stop("data must have at least two rows");
    if (k < 1 || k > n || (!include_self && k >= n)) {
      Rcpp::stop("k is incompatible with the requested self-neighbor policy");
    }
    const int search_k = include_self ? k - 1 : k;
    if (search_k > kMaxMetalGridK) {
      Rcpp::stop("Metal grid KNN currently supports at most %d non-self neighbors",
                 kMaxMetalGridK);
    }
    if (bins_per_dim < 1) Rcpp::stop("bins_per_dim must be positive");

    std::string input_type;
    std::vector<float> values = matrix_float32_column_major(
      data, n, p, input_type);
    MetalGridIndex grid = build_grid(values, n, p, bins_per_dim);

    Rcpp::IntegerMatrix indices(n, k);
    Rcpp::NumericMatrix distances(n, k);
    if (include_self) {
      for (int row = 0; row < n; ++row) {
        indices(row, 0) = row + 1;
        distances(row, 0) = 0.0;
      }
    }
    if (search_k == 0) {
      return Rcpp::List::create(
        Rcpp::Named("indices") = indices,
        Rcpp::Named("distances") = distances,
        Rcpp::Named("bins_per_dim") = bins_per_dim,
        Rcpp::Named("n_cells") = grid.n_cells,
        Rcpp::Named("self_column_included") = include_self,
        Rcpp::Named("input_type") = input_type,
        Rcpp::Named("accelerator") = "metal",
        Rcpp::Named("output_layout") = "knn_matrix_final",
        Rcpp::Named("r_side_reshaping") = false,
        Rcpp::Named("cpu_fallback") = false
      );
    }

    MetalGridState& state = metal_grid_state();
    const std::size_t output_size = static_cast<std::size_t>(n) * search_k;
    id<MTLBuffer> d_data = shared_buffer(
      state.device, values.data(), values.size() * sizeof(float), "grid data");
    id<MTLBuffer> d_offsets = shared_buffer(
      state.device, grid.offsets.data(), grid.offsets.size() * sizeof(int), "grid offsets");
    id<MTLBuffer> d_rows = shared_buffer(
      state.device, grid.rows.data(), grid.rows.size() * sizeof(int), "grid rows");
    id<MTLBuffer> d_indices = shared_buffer(
      state.device, nullptr, output_size * sizeof(int), "grid indices");
    id<MTLBuffer> d_distances = shared_buffer(
      state.device, nullptr, output_size * sizeof(float), "grid distances");

    const MetalGridParams params{
      static_cast<std::uint32_t>(n),
      static_cast<std::uint32_t>(p),
      static_cast<std::uint32_t>(search_k),
      static_cast<std::uint32_t>(bins_per_dim),
      grid.min_x,
      grid.min_y,
      grid.min_z,
      grid.cell_x,
      grid.cell_y,
      grid.cell_z
    };
    id<MTLCommandBuffer> command = [state.queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
    [encoder setBuffer:d_data offset:0 atIndex:0];
    [encoder setBuffer:d_offsets offset:0 atIndex:1];
    [encoder setBuffer:d_rows offset:0 atIndex:2];
    [encoder setBuffer:d_indices offset:0 atIndex:3];
    [encoder setBuffer:d_distances offset:0 atIndex:4];
    [encoder setBytes:&params length:sizeof(params) atIndex:5];
    dispatch_grid(encoder, state.pipeline, static_cast<std::size_t>(n));
    [encoder endEncoding];
    [command commit];
    [command waitUntilCompleted];
    if (command.status == MTLCommandBufferStatusError) {
      Rcpp::stop("Metal grid KNN failed: %s", metal_error(command.error).c_str());
    }

    const int* result_index = static_cast<const int*>([d_indices contents]);
    const float* result_distance = static_cast<const float*>([d_distances contents]);
    const int column_offset = include_self ? 1 : 0;
    for (int column = 0; column < search_k; ++column) {
      for (int row = 0; row < n; ++row) {
        const std::size_t source = static_cast<std::size_t>(column) * n + row;
        indices(row, column + column_offset) = result_index[source];
        distances(row, column + column_offset) = result_distance[source];
      }
    }

    [d_distances release];
    [d_indices release];
    [d_rows release];
    [d_offsets release];
    [d_data release];
    Rcpp::checkUserInterrupt();

    return Rcpp::List::create(
      Rcpp::Named("indices") = indices,
      Rcpp::Named("distances") = distances,
      Rcpp::Named("bins_per_dim") = bins_per_dim,
      Rcpp::Named("n_cells") = grid.n_cells,
      Rcpp::Named("self_column_included") = include_self,
      Rcpp::Named("input_type") = input_type,
      Rcpp::Named("input_layout") = "column_major_float32_shared",
      Rcpp::Named("input_owns_data") = true,
      Rcpp::Named("float32_compatibility_conversion") = input_type == "double_to_float32",
      Rcpp::Named("accelerator") = "metal",
      Rcpp::Named("device_residency") = "metal_shared_memory",
      Rcpp::Named("output_layout") = "knn_matrix_final",
      Rcpp::Named("r_side_reshaping") = false,
      Rcpp::Named("host_to_device_data_copies") = 1,
      Rcpp::Named("device_to_host_result_copies") = 1,
      Rcpp::Named("cpu_fallback") = false
    );
  }
}
