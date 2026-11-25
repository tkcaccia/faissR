#include <Rcpp.h>

#include <algorithm>
#include <chrono>
#include <climits>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <limits>
#include <memory>
#include <queue>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "faissr_metal_common.h"

#ifdef FAISSR_HAS_CUVS
Rcpp::List cuvs_file_cagra_build_impl(int n, int p, int k, int batch,
    double limit, int width, int queue, bool nn_descent,
    int graph_degree, int intermediate_degree,
    const std::function<void(int, int, float*)>& read);
Rcpp::List cuvs_file_cagra_search_impl(SEXP pointer, int first, int count);
Rcpp::List cuvs_file_cagra_query_impl(SEXP pointer, int features, int count,
    const std::function<void(float*)>& read);
Rcpp::List cuvs_file_cagra_set_search_impl(SEXP pointer, int width, int queue);
void cuvs_file_cagra_release_impl(SEXP pointer);
Rcpp::List cuvs_file_cagra_multi_build_impl(int n, int p, int k, int batch,
    double limit, int width, int queue, bool nn_descent,
    int graph_degree, int intermediate_degree,
    const std::vector<int>& devices, int merge_workers,
    const std::function<void(std::size_t, int, int, float*)>& read);
Rcpp::List cuvs_file_cagra_multi_search_impl(SEXP pointer, int features,
    int first, int count, const std::function<void(float*)>& read);
void cuvs_file_cagra_multi_release_impl(SEXP pointer);
#endif

#if defined(__unix__) || defined(__APPLE__)
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
#endif

#ifdef __APPLE__
struct FaissrMetalFlatBatch;
FaissrMetalFlatBatch* faissr_metal_flat_batch_prepare(
    const float* query, int n_query, int p, int reference_capacity, int max_k);
void faissr_metal_flat_batch_release(FaissrMetalFlatBatch* batch);
float* faissr_metal_flat_batch_reference(FaissrMetalFlatBatch& batch);
float* faissr_metal_flat_batch_query(FaissrMetalFlatBatch& batch);
double faissr_metal_flat_batch_owned_bytes(const FaissrMetalFlatBatch& batch);
void faissr_metal_flat_batch_search(FaissrMetalFlatBatch& batch,
    int n_query, int n_reference, int k,
    int* ids, float* distances, double* owned_buffer_bytes);
Rcpp::List faissr_metal_knn_file_float(std::vector<float> values,
    int rows, int columns, int k, double target);
bool faissr_metal_file_capacity(double bytes);
Rcpp::List faissr_metal_file_cagra_build_impl(int rows, int columns, int k,
    double target, double limit, const std::function<void(float*)>& read);
Rcpp::List faissr_metal_file_cagra_search_impl(SEXP pointer, int first, int count);
void faissr_metal_file_cagra_release_impl(SEXP pointer);
#endif

#ifdef _OPENMP
#include <omp.h>
#endif

#ifdef FAISSR_HAS_FAISS
#include <faiss/IndexFlat.h>
#include <faiss/utils/distances.h>
#endif

#if defined(FAISSR_HAS_CUDA) && !defined(FAISSR_FAISS_CPU_ONLY) && \
    __has_include(<faiss/gpu/GpuIndexFlat.h>) && \
    __has_include(<faiss/gpu/StandardGpuResources.h>)
#define FAISSR_FILEBACK_CUDA 1
#include <cuda_runtime_api.h>
#include <faiss/gpu/GpuIndexFlat.h>
#include <faiss/gpu/StandardGpuResources.h>
#if __has_include(<faiss/gpu/GpuCloner.h>)
#define FAISSR_FILEBACK_CUDA_MULTI 1
#include <faiss/IndexShards.h>
#include <faiss/gpu/GpuCloner.h>
#endif
#endif

namespace {

struct FileMatrix {
  std::string path;
  std::uint64_t rows;
  std::uint64_t columns;
  std::uint64_t first;
  std::uint64_t header;
};

struct BatchSource {
  FileMatrix file;
  Rcpp::NumericMatrix data;
  const int* float32_data = nullptr;
  std::uint64_t memory_rows = 0;
  SEXP owner = R_NilValue;
  bool memory = false;

  double resident_bytes() const {
    return memory_rows * static_cast<double>(file.columns) *
      (float32_data ? sizeof(float) : sizeof(double));
  }
};

std::uint64_t positive_integer(SEXP value, const char* name) {
  const double number = Rcpp::as<double>(value);
  if (!std::isfinite(number) || number < 1 ||
      number != std::floor(number) || number > 9007199254740991.0) {
    Rcpp::stop("%s must be an exact positive integer", name);
  }
  return static_cast<std::uint64_t>(number);
}

FileMatrix file_matrix(Rcpp::List spec) {
  const std::string format = Rcpp::as<std::string>(spec["format"]);
  if (format == "view") {
    FileMatrix source = file_matrix(Rcpp::as<Rcpp::List>(spec["source"]));
    const auto first = positive_integer(spec["first"], "view first");
    const auto rows = positive_integer(spec["nrow"], "view nrow");
    if (first > source.rows || rows > source.rows - first + 1) {
      Rcpp::stop("File-backed view exceeds its source");
    }
    source.first += first - 1;
    source.rows = rows;
    return source;
  }
  if (format != "f32" && format != "fbin") {
    Rcpp::stop("File-backed KNN requires f32 or fbin input");
  }
  FileMatrix result{
    Rcpp::as<std::string>(spec["path"]),
    positive_integer(spec["nrow"], "nrow"),
    positive_integer(spec["ncol"], "ncol"),
    0,
    format == "fbin" ? 8u : 0u
  };
  if (result.rows > std::numeric_limits<std::uint32_t>::max() ||
      result.columns > std::numeric_limits<int>::max() ||
      result.rows > (std::numeric_limits<std::uint64_t>::max() -
        result.header) / (4 * result.columns)) {
    Rcpp::stop("File-backed dimensions exceed native limits");
  }
  const std::uint64_t expected = result.header +
    result.rows * result.columns * 4;
  if (!std::filesystem::is_regular_file(result.path) ||
      std::filesystem::file_size(result.path) != expected) {
    Rcpp::stop("File-backed source size differs from descriptor");
  }
  if (format == "fbin") {
    std::ifstream input(result.path, std::ios::binary);
    std::uint32_t shape[2] = {0, 0};
    input.read(reinterpret_cast<char*>(shape), sizeof(shape));
    if (!input || shape[0] != result.rows ||
        shape[1] != result.columns) {
      Rcpp::stop("fbin header differs from descriptor");
    }
  }
  return result;
}

BatchSource batch_source(Rcpp::List spec) {
  const std::string format = Rcpp::as<std::string>(spec["format"]);
  if (format == "view") {
    BatchSource source = batch_source(
      Rcpp::as<Rcpp::List>(spec["source"]));
    const auto first = positive_integer(spec["first"], "view first");
    const auto rows = positive_integer(spec["nrow"], "view nrow");
    if (first > source.file.rows ||
        rows > source.file.rows - first + 1) {
      Rcpp::stop("KNN view exceeds its source");
    }
    source.file.first += first - 1;
    source.file.rows = rows;
    return source;
  }
  BatchSource source;
  if (format == "memory") {
    source.owner = spec["data"];
    source.file.rows = positive_integer(spec["nrow"], "nrow");
    source.file.columns = positive_integer(spec["ncol"], "ncol");
    if (Rf_isS4(source.owner) && Rf_inherits(source.owner, "float32")) {
      const Rcpp::S4 matrix(source.owner);
      const SEXP payload = matrix.slot("Data");
      const SEXP shape = Rf_getAttrib(payload, R_DimSymbol);
      if (TYPEOF(payload) != INTSXP || TYPEOF(shape) != INTSXP ||
          Rf_length(shape) != 2 || INTEGER(shape)[0] < 1 ||
          INTEGER(shape)[1] < 1 ||
          static_cast<std::uint64_t>(INTEGER(shape)[0]) != source.file.rows ||
          static_cast<std::uint64_t>(INTEGER(shape)[1]) != source.file.columns ||
          static_cast<std::uint64_t>(XLENGTH(payload)) !=
            source.file.rows * source.file.columns) {
        Rcpp::stop("Resident float32 KNN matrix differs from descriptor");
      }
      source.float32_data = INTEGER(payload);
    } else {
      source.data = Rcpp::as<Rcpp::NumericMatrix>(source.owner);
      if (source.data.nrow() != source.file.rows ||
          source.data.ncol() != source.file.columns) {
        Rcpp::stop("Resident KNN matrix differs from descriptor");
      }
    }
    source.file.first = 0;
    source.memory_rows = source.file.rows;
    source.memory = true;
  } else {
    source.file = file_matrix(spec);
  }
  return source;
}

void read_file_rows(std::ifstream& input, const FileMatrix& spec,
                    std::uint64_t first, std::uint64_t count, float* values) {
  const std::uint64_t items = count * spec.columns;
  const std::uint64_t offset = spec.header +
    (spec.first + first) * spec.columns * 4;
  input.seekg(static_cast<std::streamoff>(offset));
  input.read(reinterpret_cast<char*>(values),
    static_cast<std::streamsize>(items * 4));
  if (!input) throw std::runtime_error("Could not read file-backed rows");
  for (std::uint64_t i = 0; i < items; ++i) {
    if (!std::isfinite(values[i])) {
      throw std::runtime_error("File-backed KNN input contains nonfinite values");
    }
  }
}

void read_rows(std::ifstream& input, const FileMatrix& spec,
               std::uint64_t first, std::uint64_t count,
               std::vector<float>& values) {
  values.resize(static_cast<std::size_t>(count * spec.columns));
  read_file_rows(input, spec, first, count, values.data());
}

void read_float32_batch(const BatchSource& source, std::uint64_t first,
                        std::uint64_t count, float* values) {
  static_assert(sizeof(int) == sizeof(float), "float32 storage needs 32-bit ints");
  const auto columns = source.file.columns;
  constexpr std::uint64_t tile_rows = 64;
  for (std::uint64_t tile = 0; tile < count; tile += tile_rows) {
    const auto rows = std::min(tile_rows, count - tile);
    for (std::uint64_t col = 0; col < columns; ++col) {
      const int* input = source.float32_data + col * source.memory_rows +
        source.file.first + first + tile;
      for (std::uint64_t row = 0; row < rows; ++row) {
        float value;
        std::memcpy(&value, input + row, sizeof(value));
        if (!std::isfinite(value)) {
          Rcpp::stop("Resident KNN input contains nonfinite float32 values");
        }
        values[(tile + row) * columns + col] = value;
      }
    }
  }
}

void read_batch_rows_into(std::ifstream& input, const BatchSource& source,
                          std::uint64_t first, std::uint64_t count,
                          float* values) {
  if (!source.memory) {
    read_file_rows(input, source.file, first, count, values);
    return;
  }
  if (source.float32_data) {
    read_float32_batch(source, first, count, values);
    return;
  }
  for (std::uint64_t row = 0; row < count; ++row) {
    for (std::uint64_t col = 0; col < source.file.columns; ++col) {
      const float value = static_cast<float>(source.data(
        source.file.first + first + row, col));
      if (!std::isfinite(value)) {
        Rcpp::stop("Resident KNN input contains nonfinite float32 values");
      }
      values[row * source.file.columns + col] = value;
    }
  }
}

void read_batch_rows(std::ifstream& input, const BatchSource& source,
                     std::uint64_t first, std::uint64_t count,
                     std::vector<float>& values) {
  values.resize(static_cast<std::size_t>(count * source.file.columns));
  read_batch_rows_into(input, source, first, count, values.data());
}

using Candidate = std::pair<double, std::uint32_t>;

double precise_l2(const float* left, const float* right,
                  std::size_t columns) {
  double sum = 0;
  for (std::size_t col = 0; col < columns; ++col) {
    const double delta = static_cast<double>(left[col]) - right[col];
    sum += delta * delta;
  }
  return sum;
}

double bounded_l2(const float* left, const float* right,
                  std::size_t columns, double cutoff) {
  double sum = 0.0;
  for (std::size_t first = 0; first < columns; first += 16) {
    const auto end = std::min(first + 16, columns);
    for (std::size_t col = first; col < end; ++col) {
      const double delta = static_cast<double>(left[col]) - right[col];
      sum += delta * delta;
    }
    // Remaining squared differences can only increase the distance.
    if (sum > cutoff) break;
  }
  return sum;
}

struct ThreadLimit {
#ifdef _OPENMP
  int previous;
  explicit ThreadLimit(int requested) : previous(omp_get_max_threads()) {
    omp_set_num_threads(requested);
  }
  ~ThreadLimit() { omp_set_num_threads(previous); }
#else
  explicit ThreadLimit(int requested) { (void) requested; }
#endif
};

void add_candidate(std::priority_queue<Candidate>& heap,
                   Candidate candidate, std::size_t k) {
  if (heap.size() < k) {
    heap.push(candidate);
  } else if (candidate < heap.top()) {
    heap.pop();
    heap.push(candidate);
  }
}

#ifdef __APPLE__
void metal_candidates(FaissrMetalFlatBatch& batch, const float* reference,
                      const float* query,
                      std::uint64_t base, std::uint64_t rows,
                      const std::vector<std::uint64_t>& selected,
                      int k, bool exclude_self,
                      std::vector<std::priority_queue<Candidate>>& heaps,
                      std::size_t columns, double& owned_peak) {
  const int local_k = static_cast<int>(std::min<std::uint64_t>(
    rows, k + (exclude_self ? 1 : 0)));
  std::vector<int> indices(selected.size() * local_k);
  std::vector<float> distances(selected.size() * local_k);
  double owned_bytes = 0.0;
  faissr_metal_flat_batch_search(batch, static_cast<int>(selected.size()),
    static_cast<int>(rows), local_k,
    indices.data(), distances.data(), &owned_bytes);
  owned_peak = std::max(owned_peak, owned_bytes);
  for (std::size_t qi = 0; qi < selected.size(); ++qi) {
    for (int rank = 0; rank < local_k; ++rank) {
      const auto position = qi * local_k + rank;
      const int id = indices[position];
      const float distance = distances[position];
      if (id < 0 || static_cast<std::uint64_t>(id) >= rows ||
          !std::isfinite(distance) || distance < -1e-4f) {
        Rcpp::stop("faissR Metal file-backed batch returned invalid KNN");
      }
      const std::uint64_t global = base + id;
      if (exclude_self && selected[qi] == global) continue;
      const double checked_distance = precise_l2(
        query + qi * columns,
        reference + static_cast<std::size_t>(id) * columns,
        columns);
      if (!std::isfinite(checked_distance)) {
        Rcpp::stop("FAISS Metal file-backed distance overflowed float32");
      }
      add_candidate(heaps[qi], Candidate{
        checked_distance,
        static_cast<std::uint32_t>(global + 1)}, k);
    }
  }
}

struct MetalFlatFileReference {
  FileMatrix file;
  std::filesystem::file_time_type modified;
  std::uintmax_t bytes;
  int query_capacity;
  int k;
  std::unique_ptr<FaissrMetalFlatBatch,
    decltype(&faissr_metal_flat_batch_release)> batch;
};

void check_metal_reference(const MetalFlatFileReference& state) {
  if (std::filesystem::last_write_time(state.file.path) != state.modified ||
      std::filesystem::file_size(state.file.path) != state.bytes) {
    Rcpp::stop("Metal Flat reference changed during search");
  }
}
#endif

std::pair<Rcpp::IntegerMatrix, Rcpp::NumericMatrix> exact_knn_matrices(
    std::vector<std::priority_queue<Candidate>>& heaps, int k) {
  Rcpp::IntegerMatrix indices(heaps.size(), k);
  Rcpp::NumericMatrix distances(heaps.size(), k);
  for (std::size_t qi = 0; qi < heaps.size(); ++qi) {
    auto& heap = heaps[qi];
    if (heap.size() != static_cast<std::size_t>(k)) {
      Rcpp::stop("Incomplete file-backed KNN row");
    }
    for (int rank = k - 1; rank >= 0; --rank) {
      const Candidate candidate = heap.top();
      heap.pop();
      if (!std::isfinite(candidate.first)) {
        Rcpp::stop("File-backed KNN distance overflowed float32");
      }
      indices(qi, rank) = static_cast<int>(candidate.second);
      distances(qi, rank) = std::sqrt(std::max(0.0, candidate.first));
    }
  }
  return {indices, distances};
}

#ifdef FAISSR_FILEBACK_CUDA
struct CudaDeviceScope {
  int previous = -1;
  ~CudaDeviceScope() {
    if (previous >= 0) cudaSetDevice(previous);
  }
};

void cuda_candidates(faiss::Index& index,
                     const std::vector<float>& reference,
                     const std::vector<float>& query,
                     std::uint64_t base, std::uint64_t rows,
                     const std::vector<std::uint64_t>& selected,
                     int k, bool exclude_self,
                     std::vector<std::priority_queue<Candidate>>& heaps,
                     double& build_seconds, double& search_seconds) {
  const auto build_start = std::chrono::steady_clock::now();
  index.reset();
  index.add(static_cast<faiss::idx_t>(rows), reference.data());
  const auto search_start = std::chrono::steady_clock::now();
  build_seconds += std::chrono::duration<double>(
    search_start - build_start).count();
  const int local_k = static_cast<int>(std::min<std::uint64_t>(
    rows, k + (exclude_self ? 1 : 0)));
  std::vector<float> distances(selected.size() * local_k);
  std::vector<faiss::idx_t> labels(selected.size() * local_k);
  index.search(static_cast<faiss::idx_t>(selected.size()),
    query.data(), local_k, distances.data(), labels.data());
  search_seconds += std::chrono::duration<double>(
    std::chrono::steady_clock::now() - search_start).count();
  for (std::size_t qi = 0; qi < selected.size(); ++qi) {
    for (int rank = 0; rank < local_k; ++rank) {
      const auto offset = qi * local_k + rank;
      const auto id = labels[offset];
      const float distance = distances[offset];
      if (id < 0 || static_cast<std::uint64_t>(id) >= rows ||
          !std::isfinite(distance) || distance < -1e-4f) {
        Rcpp::stop("FAISS CUDA file-backed batch returned invalid KNN");
      }
      const std::uint64_t global = base + id;
      if (exclude_self && selected[qi] == global) continue;
      const float checked_distance = faiss::fvec_L2sqr(
        query.data() + qi * index.d,
        reference.data() + static_cast<std::size_t>(id) * index.d,
        static_cast<std::size_t>(index.d));
      if (!std::isfinite(checked_distance)) {
        Rcpp::stop("FAISS CUDA file-backed distance overflowed float32");
      }
      add_candidate(heaps[qi], Candidate{
        checked_distance,
        static_cast<std::uint32_t>(global + 1)}, k);
    }
  }
}
#endif

}  // namespace

extern "C" SEXP faissR_metal_file_ivf_plan_v1(SEXP source_spec,
    SEXP k_value, SEXP memory_limit_value) {
  BEGIN_RCPP
  const FileMatrix source = file_matrix(Rcpp::List(source_spec));
  const int k = Rcpp::as<int>(k_value);
  const double memory_limit = Rcpp::as<double>(memory_limit_value);
  const double n = static_cast<double>(source.rows);
  const double p = static_cast<double>(source.columns);
  const double q = p > 0 ? faissr_metal::ivf_projection_dimensions(
    source.columns, false) : 0.0;
  const double estimated_ram = 8.0 * n * p + 4.0 * n * q +
    20.0 * n * k + 256.0 * 1024 * 1024;
  const double estimated_device = 4.0 * n * p + 4.0 * n * q +
    8.0 * n * k + 128.0 * 1024 * 1024;
  const bool shape_ok = source.rows <= static_cast<std::uint64_t>(INT_MAX) &&
    faissr_metal::ivf_shape_supported(source.rows, source.columns, k) &&
    k <= 128 &&
    static_cast<std::uint64_t>(k) < source.rows;
  const bool ram_ok = std::isfinite(memory_limit) &&
    memory_limit >= estimated_ram;
  bool device_ok = false;
#ifdef __APPLE__
  if (shape_ok && ram_ok) {
    device_ok = faissr_metal_file_capacity(estimated_device);
  }
#endif
  return Rcpp::List::create(
    Rcpp::Named("fits") = shape_ok && ram_ok && device_ok,
    Rcpp::Named("shape_supported") = shape_ok,
    Rcpp::Named("ram_fits") = ram_ok,
    Rcpp::Named("metal_fits") = device_ok,
    Rcpp::Named("peak_ram_bytes_estimate") = estimated_ram,
    Rcpp::Named("peak_device_bytes_estimate") = estimated_device,
    Rcpp::Named("memory_budget_bytes") = memory_limit);
  END_RCPP
}

extern "C" SEXP faissR_metal_file_ivf_v1(SEXP source_spec,
    SEXP k_value, SEXP target_value, SEXP memory_limit_value) {
  BEGIN_RCPP
#ifndef __APPLE__
  Rcpp::stop("File-backed Metal IVF is unavailable on this platform");
#else
  const FileMatrix source = file_matrix(Rcpp::List(source_spec));
  const int k = Rcpp::as<int>(k_value);
  const double target = Rcpp::as<double>(target_value);
  const double memory_limit = Rcpp::as<double>(memory_limit_value);
  if (source.rows > static_cast<std::uint64_t>(INT_MAX) ||
      !faissr_metal::ivf_shape_supported(source.rows, source.columns, k) ||
      k > 128 ||
      static_cast<std::uint64_t>(k) >= source.rows ||
      (target != 0.9 && target != 0.95 && target != 0.99)) {
    Rcpp::stop("Invalid file-backed Metal IVF shape or recall target");
  }
  Rcpp::List plan(faissR_metal_file_ivf_plan_v1(
    source_spec, k_value, memory_limit_value));
  if (!Rcpp::as<bool>(plan["fits"])) {
    Rcpp::stop("File-backed Metal IVF exceeds RAM or Metal budget; "
      "select bounded exact search explicitly");
  }
  const auto started = std::chrono::steady_clock::now();
  const auto source_mtime = std::filesystem::last_write_time(source.path);
  std::ifstream input(source.path, std::ios::binary);
  if (!input) Rcpp::stop("Could not open file-backed Metal IVF input");
  std::vector<float> values;
  read_rows(input, source, 0, source.rows, values);
  input.close();
  const double read_seconds = std::chrono::duration<double>(
    std::chrono::steady_clock::now() - started).count();
  Rcpp::List result = faissr_metal_knn_file_float(std::move(values),
    static_cast<int>(source.rows), static_cast<int>(source.columns),
    k, target);
  if (std::filesystem::last_write_time(source.path) != source_mtime ||
      !Rcpp::as<bool>(result["target_met"])) {
    Rcpp::stop("File-backed Metal IVF input changed or pilot missed target");
  }
  result["provider"] = "faissR";
  result["backend_used"] = "faissr_native_metal_ivf_file";
  result["method"] = "metal_ivf";
  result["residency"] = "file-backed";
  result["index_residency"] = "transient_metal_buffers";
  result["accelerator"] = "metal";
  result["gpu_count"] = 1;
  result["index_base"] = 1;
  result["result_residency"] = "host";
  result["exact"] = false;
  result["source_read_seconds"] = read_seconds;
  result["peak_ram_bytes_estimate"] = plan["peak_ram_bytes_estimate"];
  result["peak_device_bytes_estimate"] = plan["peak_device_bytes_estimate"];
  result["memory_budget_bytes"] = memory_limit;
  result["device_ids"] = Rcpp::IntegerVector::create(0);
  result["total_seconds"] = std::chrono::duration<double>(
    std::chrono::steady_clock::now() - started).count();
  return result;
#endif
  END_RCPP
}

extern "C" SEXP faissR_metal_file_flat_build_v1(SEXP source_spec,
    SEXP query_capacity_value, SEXP k_value, SEXP memory_limit_value) {
  BEGIN_RCPP
#ifndef __APPLE__
  Rcpp::stop("Persistent file-backed Metal Flat is unavailable here");
#else
  const auto started = std::chrono::steady_clock::now();
  const FileMatrix file = file_matrix(Rcpp::List(source_spec));
  const int query_capacity = Rcpp::as<int>(query_capacity_value);
  const int k = Rcpp::as<int>(k_value);
  const double memory_limit = Rcpp::as<double>(memory_limit_value);
  if (file.rows > static_cast<std::uint64_t>(INT_MAX) ||
      file.columns > 16384 || k < 1 || k > 128 ||
      static_cast<std::uint64_t>(k) > file.rows ||
      query_capacity < 1 || query_capacity > 10000) {
    Rcpp::stop("Invalid file-backed Metal Flat shard dimensions");
  }
  const int max_tiles = query_capacity <= 1024 && file.rows >= 65536 ? 16 : 1;
  const double estimated = 64.0 * 1024 * 1024 +
    4.0 * (file.rows + query_capacity) * file.columns +
    (8.0 * max_tiles + 32.0) * query_capacity * k;
  if (!std::isfinite(memory_limit) || estimated > memory_limit) {
    Rcpp::stop("File-backed Metal Flat shard exceeds memory_limit");
  }
  std::unique_ptr<MetalFlatFileReference> state(new MetalFlatFileReference{
    file, std::filesystem::last_write_time(file.path),
    std::filesystem::file_size(file.path), query_capacity, k,
    {nullptr, faissr_metal_flat_batch_release}});
  state->batch.reset(faissr_metal_flat_batch_prepare(nullptr,
    query_capacity, static_cast<int>(file.columns),
    static_cast<int>(file.rows), k));
  std::ifstream input(file.path, std::ios::binary);
  if (!input) Rcpp::stop("Could not open Metal Flat shard");
  read_file_rows(input, file, 0, file.rows,
    faissr_metal_flat_batch_reference(*state->batch));
  check_metal_reference(*state);
  const double owned_bytes = faissr_metal_flat_batch_owned_bytes(*state->batch);
  Rcpp::XPtr<MetalFlatFileReference> pointer(state.release(), true,
    Rf_install("faissR_metal_file_flat_v1"));
  return Rcpp::List::create(Rcpp::Named("pointer") = pointer,
    Rcpp::Named("backend_used") = "faissr_native_metal_flat_index",
    Rcpp::Named("owned_buffer_bytes") = owned_bytes,
    Rcpp::Named("reference_bytes") = 4.0 * file.rows * file.columns,
    Rcpp::Named("build_seconds") = std::chrono::duration<double>(
      std::chrono::steady_clock::now() - started).count());
#endif
  END_RCPP
}

extern "C" SEXP faissR_metal_file_flat_search_v1(SEXP pointer, SEXP query_spec,
    SEXP k_value) {
  BEGIN_RCPP
#ifndef __APPLE__
  Rcpp::stop("Persistent file-backed Metal Flat is unavailable here");
#else
  if (TYPEOF(pointer) != EXTPTRSXP ||
      R_ExternalPtrTag(pointer) != Rf_install("faissR_metal_file_flat_v1") ||
      !R_ExternalPtrAddr(pointer)) Rcpp::stop("Invalid Metal Flat reference");
  Rcpp::XPtr<MetalFlatFileReference> state(pointer);
  const FileMatrix query = file_matrix(Rcpp::List(query_spec));
  const int k = Rcpp::as<int>(k_value);
  if (query.columns != state->file.columns || k < 1 || k > state->k ||
      query.rows > static_cast<std::uint64_t>(state->query_capacity)) {
    Rcpp::stop("Metal Flat query exceeds shard dimensions or capacity");
  }
  check_metal_reference(*state);
  const auto query_modified = std::filesystem::last_write_time(query.path);
  std::ifstream input(query.path, std::ios::binary);
  if (!input) Rcpp::stop("Could not open Metal Flat queries");
  float* query_values = faissr_metal_flat_batch_query(*state->batch);
  read_file_rows(input, query, 0, query.rows, query_values);
  std::vector<std::uint64_t> selected(query.rows);
  std::vector<std::priority_queue<Candidate>> heaps(query.rows);
  double owned_bytes = 0.0;
  metal_candidates(*state->batch,
    faissr_metal_flat_batch_reference(*state->batch), query_values, 0,
    state->file.rows, selected, k, false, heaps, query.columns, owned_bytes);
  check_metal_reference(*state);
  if (std::filesystem::last_write_time(query.path) != query_modified)
    Rcpp::stop("Metal Flat queries changed during search");
  const auto results = exact_knn_matrices(heaps, k);
  return Rcpp::List::create(Rcpp::Named("indices") = results.first,
    Rcpp::Named("distances") = results.second,
    Rcpp::Named("backend_used") = "faissr_native_metal_flat_query",
    Rcpp::Named("method") = "exact",
    Rcpp::Named("exact") = true, Rcpp::Named("index_reused") = true,
    Rcpp::Named("device_ids") = Rcpp::IntegerVector::create(0),
    Rcpp::Named("observed_metal_owned_live_buffer_bytes") = owned_bytes);
#endif
  END_RCPP
}

// [[Rcpp::export]]
Rcpp::List file_metal_cagra_build_cpp(SEXP source_spec, int k,
    double memory_limit, double target) {
#ifndef __APPLE__
  Rcpp::stop("File-backed Metal CAGRA requires Apple Metal support");
#else
  const FileMatrix file = file_matrix(Rcpp::List(source_spec));
  if (file.rows > static_cast<std::uint64_t>(INT_MAX)) {
    Rcpp::stop("File-backed Metal CAGRA supports at most INT_MAX rows");
  }
  const auto modified = std::filesystem::last_write_time(file.path);
  const auto bytes = std::filesystem::file_size(file.path);
  std::ifstream input(file.path, std::ios::binary);
  if (!input) Rcpp::stop("Could not open the file-backed Metal CAGRA source");
  auto read = [&](float* output) {
    for (std::uint64_t first = 0; first < file.rows;) {
      Rcpp::checkUserInterrupt();
      const auto count = std::min<std::uint64_t>(1024, file.rows - first);
      read_file_rows(input, file, first, count, output + first * file.columns);
      first += count;
    }
  };
  Rcpp::List index = faissr_metal_file_cagra_build_impl(
    static_cast<int>(file.rows), static_cast<int>(file.columns), k,
    target, memory_limit, read);
  if (std::filesystem::last_write_time(file.path) != modified ||
      std::filesystem::file_size(file.path) != bytes) {
    faissr_metal_file_cagra_release_impl(index["pointer"]);
    Rcpp::stop("File-backed Metal CAGRA source changed during index build");
  }
  return index;
#endif
}

// [[Rcpp::export]]
Rcpp::List file_metal_cagra_search_cpp(SEXP pointer, int first, int count) {
#ifndef __APPLE__
  Rcpp::stop("File-backed Metal CAGRA requires Apple Metal support");
#else
  return faissr_metal_file_cagra_search_impl(pointer, first, count);
#endif
}

// [[Rcpp::export]]
void file_metal_cagra_release_cpp(SEXP pointer) {
#ifndef __APPLE__
  Rcpp::stop("File-backed Metal CAGRA requires Apple Metal support");
#else
  faissr_metal_file_cagra_release_impl(pointer);
#endif
}

// [[Rcpp::export]]
Rcpp::List file_cagra_build_cpp(SEXP source_spec, int k, int batch,
    double memory_limit, int width, int queue, bool nn_descent,
    int graph_degree = 64, int intermediate_degree = 128) {
#ifndef FAISSR_HAS_CUVS
  Rcpp::stop("File-backed CUDA CAGRA requires cuVS support in faissR");
#else
  const FileMatrix file = file_matrix(Rcpp::List(source_spec));
  if (file.rows > static_cast<std::uint64_t>(INT_MAX)) {
    Rcpp::stop("File-backed CUDA CAGRA supports at most INT_MAX rows");
  }
  const auto modified = std::filesystem::last_write_time(file.path);
  const auto bytes = std::filesystem::file_size(file.path);
  std::ifstream input(file.path, std::ios::binary);
  if (!input) Rcpp::stop("Could not open file-backed CUDA CAGRA source");
  auto read = [&](int first, int count, float* output) {
    read_file_rows(input, file, first, count, output);
  };
  Rcpp::List index = cuvs_file_cagra_build_impl(static_cast<int>(file.rows),
    static_cast<int>(file.columns), k, batch, memory_limit,
    width, queue, nn_descent, graph_degree, intermediate_degree, read);
  if (std::filesystem::last_write_time(file.path) != modified ||
      std::filesystem::file_size(file.path) != bytes) {
    cuvs_file_cagra_release_impl(index["pointer"]);
    Rcpp::stop("File-backed CUDA CAGRA source changed during index build");
  }
  return index;
#endif
}

// [[Rcpp::export]]
Rcpp::List file_cagra_search_cpp(SEXP pointer, int first, int count) {
#ifndef FAISSR_HAS_CUVS
  Rcpp::stop("File-backed CUDA CAGRA requires cuVS support in faissR");
#else
  return cuvs_file_cagra_search_impl(pointer, first, count);
#endif
}

// [[Rcpp::export]]
Rcpp::List file_cagra_query_cpp(SEXP pointer, SEXP query_spec, int first, int count) {
#ifndef FAISSR_HAS_CUVS
  Rcpp::stop("File-backed CUDA CAGRA requires cuVS support in faissR");
#else
  const FileMatrix query = file_matrix(Rcpp::List(query_spec));
  if (first < 1 || count < 1 || static_cast<std::uint64_t>(first) > query.rows ||
      static_cast<std::uint64_t>(count) > query.rows - first + 1) {
    Rcpp::stop("External CUDA CAGRA query batch is outside its file view");
  }
  const auto modified = std::filesystem::last_write_time(query.path);
  std::ifstream input(query.path, std::ios::binary);
  if (!input) Rcpp::stop("Could not open external CUDA CAGRA query file");
  auto read = [&](float* output) {
    read_file_rows(input, query, first - 1, count, output);
    if (std::filesystem::last_write_time(query.path) != modified) {
      Rcpp::stop("External CUDA CAGRA query file changed during search");
    }
  };
  return cuvs_file_cagra_query_impl(pointer, static_cast<int>(query.columns), count, read);
#endif
}

// [[Rcpp::export]]
Rcpp::List file_cagra_set_search_cpp(SEXP pointer, int width, int queue) {
#ifndef FAISSR_HAS_CUVS
  Rcpp::stop("File-backed CUDA CAGRA requires cuVS support in faissR");
#else
  return cuvs_file_cagra_set_search_impl(pointer, width, queue);
#endif
}

// [[Rcpp::export]]
void file_cagra_release_cpp(SEXP pointer) {
#ifndef FAISSR_HAS_CUVS
  Rcpp::stop("File-backed CUDA CAGRA requires cuVS support in faissR");
#else
  cuvs_file_cagra_release_impl(pointer);
#endif
}

// [[Rcpp::export]]
Rcpp::List file_cagra_multi_build_cpp(SEXP source_spec, int k, int batch,
    double limit, int width, int queue, int degree, int intermediate,
    std::vector<int> devices, int merge_workers = 1) {
#ifndef FAISSR_HAS_CUVS
  Rcpp::stop("File-backed CUDA CAGRA requires cuVS support in faissR");
#else
  const FileMatrix file = file_matrix(Rcpp::List(source_spec));
  if (file.rows > static_cast<std::uint64_t>(INT_MAX)) {
    Rcpp::stop("File-backed CUDA CAGRA supports at most INT_MAX rows");
  }
  const auto modified = std::filesystem::last_write_time(file.path);
  const auto bytes = std::filesystem::file_size(file.path);
  std::vector<std::ifstream> inputs;
  for (std::size_t rank = 0; rank < devices.size(); ++rank) {
    inputs.emplace_back(file.path, std::ios::binary);
    if (!inputs.back()) Rcpp::stop("Cannot open a CUDA CAGRA shard source");
  }
  auto read = [&](std::size_t rank, int first, int count, float* output) {
    read_file_rows(inputs[rank], file, first, count, output);
  };
  Rcpp::List index = cuvs_file_cagra_multi_build_impl(
    static_cast<int>(file.rows), static_cast<int>(file.columns), k, batch,
    limit, width, queue, true, degree, intermediate, devices,
    merge_workers, read);
  if (std::filesystem::last_write_time(file.path) != modified ||
      std::filesystem::file_size(file.path) != bytes) {
    cuvs_file_cagra_multi_release_impl(index["pointer"]);
    Rcpp::stop("File-backed CUDA CAGRA source changed during index build");
  }
  return index;
#endif
}

// [[Rcpp::export]]
Rcpp::List file_cagra_multi_search_cpp(SEXP pointer, SEXP source_spec,
    int first, int count) {
#ifndef FAISSR_HAS_CUVS
  Rcpp::stop("File-backed CUDA CAGRA requires cuVS support in faissR");
#else
  const FileMatrix file = file_matrix(Rcpp::List(source_spec));
  if (first < 1 || count < 1 || static_cast<std::uint64_t>(first) > file.rows ||
      static_cast<std::uint64_t>(count) > file.rows - first + 1) {
    Rcpp::stop("CUDA CAGRA query batch is outside its file view");
  }
  const auto modified = std::filesystem::last_write_time(file.path);
  std::ifstream input(file.path, std::ios::binary);
  if (!input) Rcpp::stop("Cannot open the CUDA CAGRA query source");
  auto read = [&](float* output) {
    read_file_rows(input, file, first - 1, count, output);
    if (std::filesystem::last_write_time(file.path) != modified) {
      Rcpp::stop("CUDA CAGRA query file changed during search");
    }
  };
  return cuvs_file_cagra_multi_search_impl(pointer,
    static_cast<int>(file.columns), first, count, read);
#endif
}

// [[Rcpp::export]]
void file_cagra_multi_release_cpp(SEXP pointer) {
#ifndef FAISSR_HAS_CUVS
  Rcpp::stop("File-backed CUDA CAGRA requires cuVS support in faissR");
#else
  cuvs_file_cagra_multi_release_impl(pointer);
#endif
}

SEXP matrix_knn_batch_impl(SEXP query_spec,
    SEXP reference_spec, SEXP row_ids, SEXP k_value, SEXP backend,
    SEXP method, SEXP reference_chunk_value, SEXP memory_limit_value,
    SEXP n_threads_value, SEXP exclude_self_value, SEXP devices_value) {
  BEGIN_RCPP
#ifndef FAISSR_HAS_FAISS
  Rcpp::stop("File-backed KNN requires a functional FAISS build");
#else
  const std::string requested_backend = Rcpp::as<std::string>(backend);
  const bool metal = requested_backend == "metal";
  const bool cuda = requested_backend == "cuda";
  if ((requested_backend != "cpu" && !metal && !cuda) ||
      Rcpp::as<std::string>(method) != "stream_exact") {
    Rcpp::stop("Requested file-backed KNN backend/method is unavailable");
  }
#ifndef FAISSR_FILEBACK_CUDA
  if (cuda) Rcpp::stop("CUDA file-backed KNN is unavailable in this build");
#endif
#ifndef __APPLE__
  if (metal) Rcpp::stop("Metal-enabled file-backed KNN is unavailable here");
#endif
  std::vector<int> cuda_devices;
  if (devices_value != R_NilValue) {
    if (!cuda || TYPEOF(devices_value) != INTSXP) {
      Rcpp::stop("Explicit KNN devices require CUDA integer device IDs");
    }
    const Rcpp::IntegerVector requested_devices(devices_value);
    if (requested_devices.size() < 1 || requested_devices.size() > 16) {
      Rcpp::stop("CUDA KNN needs 1 to 16 selected devices");
    }
#ifdef FAISSR_FILEBACK_CUDA
    int available_devices = 0;
    if (cudaGetDeviceCount(&available_devices) != cudaSuccess) {
      Rcpp::stop("Could not enumerate CUDA devices");
    }
    for (int device : requested_devices) {
      if (device < 0 || device >= available_devices ||
          std::find(cuda_devices.begin(), cuda_devices.end(), device) !=
            cuda_devices.end()) {
        Rcpp::stop("CUDA KNN device IDs must be distinct and available");
      }
      cuda_devices.push_back(device);
    }
#endif
#ifndef FAISSR_FILEBACK_CUDA_MULTI
    if (requested_devices.size() > 1) {
      Rcpp::stop("Multi-GPU CUDA KNN is unavailable in this build");
    }
#endif
  }
  const BatchSource query = batch_source(Rcpp::as<Rcpp::List>(query_spec));
  const BatchSource reference = batch_source(
    Rcpp::as<Rcpp::List>(reference_spec));
  const Rcpp::NumericVector requested(row_ids);
  const int k = Rcpp::as<int>(k_value);
  const int reference_chunk = Rcpp::as<int>(reference_chunk_value);
  const int n_threads = Rcpp::as<int>(n_threads_value);
  const bool exclude_self = Rcpp::as<bool>(exclude_self_value);
  const double memory_limit = Rcpp::as<double>(memory_limit_value);
  const bool same_owner = query.memory == reference.memory &&
    (query.memory ? query.owner == reference.owner :
      query.file.path == reference.file.path);
  const bool same_source = same_owner &&
    query.file.first == reference.file.first &&
    query.file.rows == reference.file.rows;
  if (query.file.columns != reference.file.columns ||
      reference.file.rows > static_cast<std::uint64_t>(INT_MAX) ||
      requested.size() < 1 || requested.size() > 10000 ||
      k < 1 || k > 65536 ||
      static_cast<std::uint64_t>(k) > reference.file.rows -
        (exclude_self ? 1u : 0u) ||
      reference_chunk < 1 || n_threads < 1 ||
      (metal && (n_threads != 1 || k + (exclude_self ? 1 : 0) > 128 ||
        query.file.columns > 16384)) ||
      (cuda && (n_threads != 1 || k + (exclude_self ? 1 : 0) > 256)) ||
      (exclude_self && !same_source)) {
    Rcpp::stop("Invalid file-backed KNN batch dimensions or settings");
  }
  const std::size_t count = requested.size();
  const int metal_reference_rows = static_cast<int>(std::min<std::uint64_t>(
    reference.file.rows, reference_chunk));
  const double resident_bytes = query.resident_bytes() +
    (!same_owner ? reference.resident_bytes() : 0.0);
  const double metal_output_bytes = 8.0 * count *
    std::min(k + static_cast<int>(exclude_self), metal_reference_rows) *
    (count <= 1024 && metal_reference_rows >= 65536 ? 16 : 1);
  const double metal_bytes = 4.0 * (count + metal_reference_rows) *
    query.file.columns + metal_output_bytes;
  const double estimated = resident_bytes +
    4.0 * count * query.file.columns +
    (metal ? metal_bytes : 4.0 * reference_chunk * reference.file.columns) +
    (cuda ? 12.0 * count * std::min<int>(k + 1,
      reference_chunk) : 0.0) +
    32.0 * count * k + 64.0 * 1024 * 1024;
  if (!std::isfinite(memory_limit) || memory_limit < estimated) {
    Rcpp::stop("File-backed KNN batch buffers exceed memory_limit");
  }
  std::vector<std::uint64_t> selected(count);
  for (std::size_t i = 0; i < count; ++i) {
    const double value = requested[i];
    if (!std::isfinite(value) || value < 1 ||
        value != std::floor(value) || value > query.file.rows) {
      Rcpp::stop("File-backed KNN row_ids must be valid one-based rows");
    }
    selected[i] = static_cast<std::uint64_t>(value) - 1;
  }
  const auto started = std::chrono::steady_clock::now();
  const auto query_mtime = query.memory ?
    std::filesystem::file_time_type{} :
    std::filesystem::last_write_time(query.file.path);
  const auto reference_mtime = reference.memory ?
    std::filesystem::file_time_type{} :
    std::filesystem::last_write_time(reference.file.path);
  std::ifstream query_file;
  std::ifstream reference_file;
  if (!query.memory) query_file.open(query.file.path, std::ios::binary);
  if (!reference.memory)
    reference_file.open(reference.file.path, std::ios::binary);
  if ((!query.memory && !query_file) ||
      (!reference.memory && !reference_file)) {
    Rcpp::stop("Could not open file-backed KNN input");
  }
  std::vector<float> query_values(count * query.file.columns);
  std::vector<float> read_buffer;
  bool contiguous = true;
  for (std::size_t i = 1; i < count; ++i) {
    if (selected[i] != selected[i - 1] + 1) {
      contiguous = false;
      break;
    }
  }
  if (contiguous) {
    read_batch_rows(query_file, query, selected[0], count,
                    query_values);
  } else {
    for (std::size_t i = 0; i < count; ++i) {
      read_batch_rows(query_file, query, selected[i], 1, read_buffer);
      std::copy(read_buffer.begin(), read_buffer.end(),
        query_values.begin() + i * query.file.columns);
    }
  }
  const double query_read_seconds = std::chrono::duration<double>(
    std::chrono::steady_clock::now() - started).count();
  ThreadLimit threads(n_threads);
  std::vector<std::priority_queue<Candidate>> heaps(count);
  std::vector<float> reference_values;
  double build_seconds = 0.0;
  double search_seconds = 0.0;
  double device_bytes = 0.0;
  double metal_owned_peak = 0.0;
  double reference_read_seconds = 0.0;
  double candidate_processing_seconds = 0.0;
  double metal_staging_setup_seconds = 0.0;
#ifdef __APPLE__
  std::unique_ptr<FaissrMetalFlatBatch,
    decltype(&faissr_metal_flat_batch_release)>
    metal_batch(nullptr, faissr_metal_flat_batch_release);
  float* metal_reference = nullptr;
  if (metal) {
    const auto setup_started = std::chrono::steady_clock::now();
    const int local_k = std::min(k + static_cast<int>(exclude_self),
      metal_reference_rows);
    metal_batch.reset(faissr_metal_flat_batch_prepare(query_values.data(),
      static_cast<int>(count), static_cast<int>(query.file.columns),
      metal_reference_rows, local_k));
    metal_reference = faissr_metal_flat_batch_reference(*metal_batch);
    metal_staging_setup_seconds = std::chrono::duration<double>(
      std::chrono::steady_clock::now() - setup_started).count();
  }
#endif
#ifdef FAISSR_FILEBACK_CUDA
  int cuda_device = -1;
  CudaDeviceScope cuda_scope;
  std::vector<std::unique_ptr<faiss::gpu::StandardGpuResources>>
    cuda_resources;
  std::unique_ptr<faiss::Index> cuda_index;
  if (cuda) {
    if (cudaGetDevice(&cuda_scope.previous) != cudaSuccess) {
      Rcpp::stop("Could not identify the requested CUDA device");
    }
    if (cuda_devices.empty()) cuda_devices.push_back(
      cuda_scope.previous);
    cuda_device = cuda_devices.front();
    const double per_gpu_rows = std::ceil(
      static_cast<double>(reference_chunk) / cuda_devices.size());
    device_bytes = 4.0 * per_gpu_rows * query.file.columns +
      4.0 * count * query.file.columns +
      12.0 * count * std::min<int>(k + 1, reference_chunk) +
      64.0 * 1024 * 1024;
    for (int device : cuda_devices) {
      std::size_t free_bytes = 0;
      std::size_t total_bytes = 0;
      if (cudaSetDevice(device) != cudaSuccess ||
          cudaMemGetInfo(&free_bytes, &total_bytes) != cudaSuccess) {
        Rcpp::stop("Could not inspect selected CUDA KNN device");
      }
      if (free_bytes < device_bytes + 128.0 * 1024 * 1024) {
        Rcpp::stop("CUDA file-backed KNN exceeds free device memory");
      }
      cuda_resources.emplace_back(new faiss::gpu::StandardGpuResources());
      cuda_resources.back()->setTempMemory(64 * 1024 * 1024);
    }
    if (cuda_devices.size() == 1) {
      faiss::gpu::GpuIndexFlatConfig config;
      config.device = cuda_device;
      cuda_index.reset(new faiss::gpu::GpuIndexFlatL2(
        cuda_resources.front().get(), query.file.columns, config));
    } else {
#ifdef FAISSR_FILEBACK_CUDA_MULTI
      std::vector<faiss::gpu::GpuResourcesProvider*> providers;
      for (auto& owner : cuda_resources) providers.push_back(owner.get());
      faiss::IndexFlatL2 empty_index(query.file.columns);
      faiss::gpu::GpuMultipleClonerOptions options;
      options.shard = true;
      options.shard_type = 1;
      options.useFloat16 = false;
      cuda_index.reset(faiss::gpu::index_cpu_to_gpu_multiple(
        providers, cuda_devices, &empty_index, &options));
      if (dynamic_cast<faiss::IndexShards*>(cuda_index.get()) == nullptr) {
        Rcpp::stop("FAISS did not create a sharded GPU KNN index");
      }
#endif
    }
  }
#endif
  for (std::uint64_t base = 0; base < reference.file.rows;
       base += reference_chunk) {
    const auto rows = std::min<std::uint64_t>(reference_chunk,
                                              reference.file.rows - base);
    const auto read_started = std::chrono::steady_clock::now();
#ifdef __APPLE__
    if (metal) {
      read_batch_rows_into(reference_file, reference, base, rows,
                           metal_reference);
    } else
#endif
    {
      read_batch_rows(reference_file, reference, base, rows, reference_values);
    }
    reference_read_seconds += std::chrono::duration<double>(
      std::chrono::steady_clock::now() - read_started).count();
    const auto processing_started = std::chrono::steady_clock::now();
    if (metal) {
#ifdef __APPLE__
      metal_candidates(*metal_batch, metal_reference, query_values.data(),
        base, rows,
        selected, k, exclude_self, heaps, query.file.columns,
        metal_owned_peak);
#endif
    } else if (cuda) {
#ifdef FAISSR_FILEBACK_CUDA
      cuda_candidates(*cuda_index, reference_values, query_values,
        base, rows, selected, k, exclude_self, heaps,
        build_seconds, search_seconds);
#endif
    } else {
#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(n_threads)
#endif
    for (std::int64_t qi = 0;
         qi < static_cast<std::int64_t>(count); ++qi) {
      const float* point = query_values.data() + qi * query.file.columns;
      auto& heap = heaps[qi];
      for (std::uint64_t ri = 0; ri < rows; ++ri) {
        if (exclude_self && selected[qi] == base + ri) continue;
        const float* reference_row =
          reference_values.data() + ri * reference.file.columns;
        const auto columns = static_cast<std::size_t>(query.file.columns);
        const double distance = columns >= 32 &&
          heap.size() == static_cast<std::size_t>(k) ?
          bounded_l2(point, reference_row, columns, heap.top().first) :
          precise_l2(point, reference_row, columns);
        add_candidate(heap, Candidate{distance,
          static_cast<std::uint32_t>(base + ri + 1)}, k);
      }
    }
    }
    candidate_processing_seconds += std::chrono::duration<double>(
      std::chrono::steady_clock::now() - processing_started).count();
    Rcpp::checkUserInterrupt();
  }
  if ((!query.memory &&
       std::filesystem::last_write_time(query.file.path) != query_mtime) ||
      (!reference.memory &&
       std::filesystem::last_write_time(reference.file.path) !=
         reference_mtime)) {
    Rcpp::stop("File-backed KNN input changed during search");
  }
  const auto format_started = std::chrono::steady_clock::now();
  const auto results = exact_knn_matrices(heaps, k);
  const double result_format_seconds = std::chrono::duration<double>(
    std::chrono::steady_clock::now() - format_started).count();
  const double seconds = std::chrono::duration<double>(
    std::chrono::steady_clock::now() - started).count();
  Rcpp::IntegerVector device_ids(cuda_devices.size());
  std::copy(cuda_devices.begin(), cuda_devices.end(),
    device_ids.begin());
  if (metal) device_ids = Rcpp::IntegerVector::create(0);
  const bool file_io = !query.memory || !reference.memory;
  return Rcpp::List::create(
    Rcpp::Named("indices") = results.first,
    Rcpp::Named("distances") = results.second,
    Rcpp::Named("provider") = "faissR",
    Rcpp::Named("backend_used") = metal ?
      "faissr_native_metal_exact_file_batch" :
      (cuda ? (cuda_devices.size() > 1 ?
        "faiss_gpu_flat_file_shards" :
        "faiss_gpu_flat_file_batch") : "faiss_cpu_stream_exact"),
    Rcpp::Named("method") = "stream_exact",
    Rcpp::Named("residency") = query.memory && reference.memory ?
      "resident" : (query.memory || reference.memory ?
        "mixed" : "file-backed"),
    Rcpp::Named("device_ids") = device_ids,
    Rcpp::Named("exact") = true,
    Rcpp::Named("index_build_seconds") = build_seconds,
    Rcpp::Named("search_seconds") = cuda ? search_seconds : seconds,
    Rcpp::Named("search_seconds_scope") = cuda ?
      "faiss_index_search_including_internal_transfers" :
      (file_io ? "complete_batch_wall_including_file_io" :
        "complete_batch_wall_resident"),
    Rcpp::Named("total_seconds") = seconds,
    Rcpp::Named("query_read_seconds") = query_read_seconds,
    Rcpp::Named("reference_read_seconds") = reference_read_seconds,
    Rcpp::Named("candidate_processing_seconds") =
      candidate_processing_seconds,
    Rcpp::Named("result_format_seconds") = result_format_seconds,
    Rcpp::Named("timing_detail") =
      "read_and_processing_wall_times; CUDA build/search overlap processing",
    Rcpp::Named("peak_ram_bytes_estimate") = estimated,
    Rcpp::Named("peak_device_bytes") = metal ? metal_bytes :
      (cuda ? device_bytes : 0.0),
    Rcpp::Named("peak_device_bytes_by_id") = cuda ?
      Rcpp::NumericVector(cuda_devices.size(), device_bytes) :
      Rcpp::NumericVector(),
    Rcpp::Named("observed_metal_owned_live_buffer_bytes") =
      metal ? metal_owned_peak : 0.0,
    Rcpp::Named("metal_staging_setup_seconds") = metal_staging_setup_seconds,
    Rcpp::Named("metal_query_uploads") = metal ? 1 : 0,
    Rcpp::Named("metal_reference_shards") = metal ?
      static_cast<int>((reference.file.rows - 1) / reference_chunk + 1) : 0,
    Rcpp::Named("device_memory_model") = metal ?
      "unified_shared_estimate" :
      (cuda ? (cuda_devices.size() > 1 ?
        "faiss_gpu_flat_shard_estimate" :
        "faiss_gpu_flat_estimate") : "none"));
#endif
  END_RCPP
}

extern "C" SEXP faissR_matrix_knn_batch_v1(SEXP query_spec,
    SEXP reference_spec, SEXP row_ids, SEXP k_value, SEXP backend,
    SEXP method, SEXP reference_chunk_value, SEXP memory_limit_value,
    SEXP n_threads_value, SEXP exclude_self_value) {
  return matrix_knn_batch_impl(query_spec, reference_spec, row_ids,
    k_value, backend, method, reference_chunk_value,
    memory_limit_value, n_threads_value, exclude_self_value,
    R_NilValue);
}

extern "C" SEXP faissR_matrix_knn_batch_v2(SEXP query_spec,
    SEXP reference_spec, SEXP row_ids, SEXP k_value, SEXP backend,
    SEXP method, SEXP reference_chunk_value, SEXP memory_limit_value,
    SEXP n_threads_value, SEXP exclude_self_value, SEXP devices_value) {
  return matrix_knn_batch_impl(query_spec, reference_spec, row_ids,
    k_value, backend, method, reference_chunk_value,
    memory_limit_value, n_threads_value, exclude_self_value,
    devices_value);
}

namespace {

using PostingHit = std::pair<double, std::uint32_t>;

void add_posting_hit(std::priority_queue<PostingHit>& heap,
                     PostingHit candidate, std::size_t k) {
  if (heap.size() < k) {
    heap.push(candidate);
  } else if (candidate < heap.top()) {
    heap.pop();
    heap.push(candidate);
  }
}

std::vector<std::uint64_t> read_posting_offsets(
    const std::string& path, std::size_t lists, std::uint64_t rows) {
  if (!std::filesystem::is_regular_file(path) ||
      std::filesystem::file_size(path) != (lists + 1) * 8) {
    Rcpp::stop("Posting offsets file has the wrong size");
  }
  std::vector<std::uint64_t> offsets(lists + 1);
  std::ifstream input(path, std::ios::binary);
  input.read(reinterpret_cast<char*>(offsets.data()),
             static_cast<std::streamsize>(offsets.size() * 8));
  if (!input || offsets.front() != 0 || offsets.back() != rows) {
    Rcpp::stop("Posting offsets do not cover the source");
  }
  for (std::size_t i = 0; i < lists; ++i) {
    if (offsets[i] > offsets[i + 1]) {
      Rcpp::stop("Posting offsets are not ordered");
    }
  }
  return offsets;
}

double posting_distance(const float* left, const float* right,
                        std::size_t columns) {
  return precise_l2(left, right, columns);
}

Rcpp::List posting_search_impl(SEXP query_spec_value, SEXP feature_value,
    SEXP ids_value, SEXP offsets_value, SEXP centers_value,
    SEXP first_value, SEXP count_value, SEXP k_value, SEXP probes_value,
    SEXP chunk_value, SEXP workers_value, SEXP exclude_value,
    SEXP rows_value, SEXP verbose_value) {
  const Rcpp::List spec(query_spec_value);
  const BatchSource query = batch_source(spec);
  const std::string feature = Rcpp::as<std::string>(feature_value);
  const std::string ids = Rcpp::as<std::string>(ids_value);
  const std::string offsets_path = Rcpp::as<std::string>(offsets_value);
  const Rcpp::NumericMatrix centers(centers_value);
  const auto first = positive_integer(first_value, "first_row");
  const auto count = positive_integer(count_value, "count");
  const int k = Rcpp::as<int>(k_value);
  const int probes = Rcpp::as<int>(probes_value);
  const int chunk = Rcpp::as<int>(chunk_value);
  const int workers = Rcpp::as<int>(workers_value);
  const bool exclude_self = Rcpp::as<bool>(exclude_value);
  const auto rows = positive_integer(rows_value, "posting_rows");
  const bool verbose = Rcpp::as<bool>(verbose_value);
  const std::size_t lists = centers.nrow();
  const std::size_t columns = query.file.columns;
  if (rows > INT_MAX || lists < 2 || lists > 65536 ||
      centers.ncol() != static_cast<int>(columns) ||
      first > query.file.rows || count > query.file.rows - first + 1 ||
      count > 100000 || k < 1 || k > 65536 ||
      static_cast<std::uint64_t>(k) > rows - (exclude_self ? 1u : 0u) ||
      probes < 1 || probes > static_cast<int>(lists) ||
      chunk < 1 || workers < 1 ||
      count * columns * 4 > 256000000 ||
      static_cast<std::uint64_t>(chunk) * columns * 4 > 256000000 ||
      count * static_cast<std::uint64_t>(k) > 128000000 / 16) {
    Rcpp::stop("Posting search dimensions or buffers are invalid");
  }
  if (!std::filesystem::is_regular_file(feature) ||
      !std::filesystem::is_regular_file(ids) ||
      std::filesystem::file_size(feature) != rows * columns * 4 ||
      std::filesystem::file_size(ids) != rows * 4) {
    Rcpp::stop("Posting feature or ID file has the wrong size");
  }
  const auto offsets = read_posting_offsets(offsets_path, lists, rows);
  std::vector<float> center_values(lists * columns);
  for (std::size_t list = 0; list < lists; ++list) {
    for (std::size_t col = 0; col < columns; ++col) {
      const float value = static_cast<float>(centers(list, col));
      if (!std::isfinite(value)) Rcpp::stop("Posting center is nonfinite");
      center_values[list * columns + col] = value;
    }
  }
  std::ifstream query_file;
  if (!query.memory) query_file.open(query.file.path, std::ios::binary);
  std::vector<float> queries(count * columns);
  std::vector<std::uint32_t> query_ids(count);
  const bool selected = spec.containsElementNamed("selected_rows");
  const bool grouped = spec.containsElementNamed("query_ids_path");
  if (selected && (first != 1 || grouped)) {
    Rcpp::stop("Selected posting rows require first_row = 1");
  }
  if (selected) {
    const Rcpp::NumericVector requested(spec["selected_rows"]);
    if (requested.size() != static_cast<int>(count)) {
      Rcpp::stop("Selected posting row count differs from batch");
    }
    std::vector<float> one;
    for (std::size_t qi = 0; qi < count; ++qi) {
      const double value = requested[qi];
      if (!std::isfinite(value) || value < 1 ||
          value != std::floor(value) ||
          value > query.file.rows || value > rows) {
        Rcpp::stop("Selected query row is outside the source");
      }
      const auto id = static_cast<std::uint64_t>(value);
      query_ids[qi] = static_cast<std::uint32_t>(id);
      read_batch_rows(query_file, query, id - 1, 1, one);
      std::copy(one.begin(), one.end(), queries.begin() + qi * columns);
    }
  } else {
    read_batch_rows(query_file, query, first - 1, count, queries);
    if (grouped) {
      const std::string path = Rcpp::as<std::string>(
        spec["query_ids_path"]);
      if (!std::filesystem::is_regular_file(path) ||
          std::filesystem::file_size(path) != rows * 4) {
        Rcpp::stop("Posting query ID file has the wrong size");
      }
      std::ifstream input(path, std::ios::binary);
      input.seekg(static_cast<std::streamoff>((first - 1) * 4));
      input.read(reinterpret_cast<char*>(query_ids.data()),
                 static_cast<std::streamsize>(count * 4));
      if (!input) Rcpp::stop("Posting query ID read failed");
      for (const auto id : query_ids) {
        if (id < 1 || id > rows) Rcpp::stop("Invalid posting query ID");
      }
    } else {
      for (std::size_t qi = 0; qi < count; ++qi) {
        query_ids[qi] = static_cast<std::uint32_t>(first + qi);
      }
    }
  }
  std::vector<std::vector<int>> active(lists);
  for (std::size_t qi = 0; qi < count; ++qi) {
    std::vector<PostingHit> ranked(lists);
    for (std::size_t list = 0; list < lists; ++list) {
      ranked[list] = {posting_distance(
        queries.data() + qi * columns,
        center_values.data() + list * columns, columns),
        static_cast<std::uint32_t>(list)};
    }
    if (probes < static_cast<int>(lists)) {
      std::partial_sort(ranked.begin(), ranked.begin() + probes,
                        ranked.end());
    } else {
      std::sort(ranked.begin(), ranked.end());
    }
    for (int probe = 0; probe < probes; ++probe) {
      active[ranked[probe].second].push_back(static_cast<int>(qi));
    }
  }
  std::vector<std::priority_queue<PostingHit>> heaps(count);
  std::vector<std::uint64_t> candidates(count, 0);
  std::vector<float> block(static_cast<std::size_t>(chunk) * columns);
  std::vector<std::uint32_t> row_ids(chunk);
  std::ifstream features(feature, std::ios::binary);
  std::ifstream id_file(ids, std::ios::binary);
  if (!features || !id_file) Rcpp::stop("Cannot open posting files");
  const auto started = std::chrono::steady_clock::now();
  std::uint64_t posting_rows_read = 0;
  ThreadLimit thread_limit(workers);
  for (std::size_t list = 0; list < lists; ++list) {
    if (active[list].empty()) continue;
    posting_rows_read += offsets[list + 1] - offsets[list];
    for (auto base = offsets[list]; base < offsets[list + 1];) {
      const auto size = std::min<std::uint64_t>(chunk,
        offsets[list + 1] - base);
      features.seekg(static_cast<std::streamoff>(base * columns * 4));
      id_file.seekg(static_cast<std::streamoff>(base * 4));
      features.read(reinterpret_cast<char*>(block.data()),
                    static_cast<std::streamsize>(size * columns * 4));
      id_file.read(reinterpret_cast<char*>(row_ids.data()),
                   static_cast<std::streamsize>(size * 4));
      if (!features || !id_file) Rcpp::stop("Posting read failed");
      for (std::size_t ri = 0; ri < size; ++ri) {
        if (row_ids[ri] < 1 || row_ids[ri] > rows) {
          Rcpp::stop("Posting row ID is invalid");
        }
        for (std::size_t col = 0; col < columns; ++col) {
          if (!std::isfinite(block[ri * columns + col])) {
            Rcpp::stop("Posting feature is nonfinite");
          }
        }
      }
#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(workers)
#endif
      for (std::int64_t ai = 0;
           ai < static_cast<std::int64_t>(active[list].size()); ++ai) {
        const int qi = active[list][ai];
        auto& heap = heaps[qi];
        for (std::size_t ri = 0; ri < size; ++ri) {
          const auto id = row_ids[ri];
          if (exclude_self && id == query_ids[qi]) continue;
          const double distance = posting_distance(
            queries.data() + static_cast<std::size_t>(qi) * columns,
            block.data() + ri * columns, columns);
          add_posting_hit(heap, PostingHit{distance, id}, k);
          ++candidates[qi];
        }
      }
      base += size;
      Rcpp::checkUserInterrupt();
    }
    if (verbose && (list + 1) % std::max<std::size_t>(1, lists / 10) == 0) {
      Rcpp::Rcout << "Posting search: " << list + 1 << "/" << lists
                  << " lists\n";
    }
  }
  Rcpp::IntegerMatrix indices(count, k);
  Rcpp::NumericMatrix distances(count, k);
  Rcpp::NumericVector candidate_count(count);
  for (std::size_t qi = 0; qi < count; ++qi) {
    auto& heap = heaps[qi];
    if (heap.size() != static_cast<std::size_t>(k)) {
      Rcpp::stop("Probed postings contain fewer than k neighbors");
    }
    candidate_count[qi] = static_cast<double>(candidates[qi]);
    for (int rank = k - 1; rank >= 0; --rank) {
      const auto hit = heap.top();
      heap.pop();
      indices(qi, rank) = static_cast<int>(hit.second);
      distances(qi, rank) = std::sqrt(hit.first);
    }
  }
  const double seconds = std::chrono::duration<double>(
    std::chrono::steady_clock::now() - started).count();
  return Rcpp::List::create(
    Rcpp::Named("indices") = indices,
    Rcpp::Named("distances") = distances,
    Rcpp::Named("candidate_count") = candidate_count,
    Rcpp::Named("posting_read_bytes") = static_cast<double>(
      posting_rows_read * (columns * 4 + 4)),
    Rcpp::Named("provider") = "faissR",
    Rcpp::Named("backend_used") = "faissr_cpu_posting_scan",
    Rcpp::Named("method") = "physical_postings",
    Rcpp::Named("residency") = "file-backed",
    Rcpp::Named("device_ids") = Rcpp::IntegerVector(),
    Rcpp::Named("exact") = probes == static_cast<int>(lists),
    Rcpp::Named("index_build_seconds") = 0.0,
    Rcpp::Named("search_seconds") = seconds,
    Rcpp::Named("total_seconds") = seconds,
    Rcpp::Named("peak_ram_bytes_estimate") = static_cast<double>(
      64 * 1024 * 1024 + count * columns * 4 +
      static_cast<std::size_t>(chunk) * columns * 4 +
      count * static_cast<std::size_t>(k) * sizeof(PostingHit)),
    Rcpp::Named("peak_device_bytes") = 0.0);
}

double nearest_posting_distance(const float* row,
    const std::vector<float>& centers, std::size_t lists,
    std::size_t columns) {
  double best = std::numeric_limits<double>::infinity();
  for (std::size_t list = 0; list < lists; ++list) {
    double distance = 0;
    for (std::size_t col = 0; col < columns; ++col) {
      const double delta = static_cast<double>(row[col]) -
        centers[list * columns + col];
      distance += delta * delta;
    }
    if (distance < best) best = distance;
  }
  return best;
}

Rcpp::List posting_distant_rows_impl(SEXP arguments) {
  Rcpp::List args(arguments);
  Rcpp::List spec(args["spec"]);
  const bool synthetic = Rcpp::as<std::string>(spec["format"]) ==
    "synthetic";
  BatchSource source;
  if (synthetic) {
    source.file.rows = positive_integer(spec["nrow"], "nrow");
    source.file.columns = positive_integer(spec["ncol"], "ncol");
  } else {
    source = batch_source(spec);
  }
  Rcpp::NumericMatrix centers_r(args["centers"]);
  const auto rows = source.file.rows;
  const auto columns = source.file.columns;
  const auto lists = static_cast<std::size_t>(centers_r.nrow());
  const int sample = Rcpp::as<int>(args["sample"]);
  const int chunk_rows = Rcpp::as<int>(args["chunk_rows"]);
  const int workers = Rcpp::as<int>(args["workers"]);
  const int candidate_rows = Rcpp::as<int>(args["candidate_rows"]);
  if (source.memory || rows > static_cast<std::uint64_t>(INT_MAX) ||
      columns != static_cast<std::uint64_t>(centers_r.ncol()) ||
      lists < 2 || lists > 65536 || sample < 1 ||
      static_cast<std::uint64_t>(sample) > rows ||
      chunk_rows < 1 || workers < 1 || candidate_rows < 0 ||
      (candidate_rows > 0 &&
       (candidate_rows < sample ||
        static_cast<std::uint64_t>(candidate_rows) > rows))) {
    Rcpp::stop("Posting audit dimensions or controls are invalid.");
  }
  std::vector<float> centers(lists * columns);
  for (std::size_t list = 0; list < lists; ++list) {
    for (std::size_t col = 0; col < columns; ++col) {
      const double value = centers_r(list, col);
      if (!std::isfinite(value) ||
          std::abs(value) > std::numeric_limits<float>::max()) {
        Rcpp::stop("Posting centers must be finite float32 values.");
      }
      centers[list * columns + col] = static_cast<float>(value);
    }
  }
  using RankedRow = std::pair<double, int>;
  std::priority_queue<RankedRow, std::vector<RankedRow>,
                      std::greater<RankedRow>> farthest;
  auto consider = [&](std::uint64_t row, double distance) {
    if (!std::isfinite(distance)) Rcpp::stop(
      "Posting audit input has non-finite center distance.");
    const RankedRow candidate{distance, static_cast<int>(row + 1)};
    if (static_cast<int>(farthest.size()) < sample) {
      farthest.push(candidate);
    } else if (candidate > farthest.top()) {
      farthest.pop();
      farthest.push(candidate);
    }
  };
  std::ifstream input;
  if (!synthetic) {
    input.open(source.file.path, std::ios::binary);
    if (!input) Rcpp::stop("Cannot open posting audit input.");
  }
  std::vector<float> block;
  auto read = [&](std::uint64_t first, std::uint64_t count) {
    if (synthetic) {
      block.resize(count * columns);
      for (std::uint64_t row = 0; row < count; ++row) {
        for (std::uint64_t col = 0; col < columns; ++col) {
          block[row * columns + col] =
            static_cast<float>((first + row) % 10007) / 10007.0f +
            static_cast<float>(col) * 0.01f;
        }
      }
    } else {
      read_batch_rows(input, source, first, count, block);
    }
  };
  if (candidate_rows > 0) {
    for (int i = 0; i < candidate_rows; ++i) {
      const auto midpoint = 2 * static_cast<std::uint64_t>(i) + 1;
      const auto index = midpoint * rows /
        (2 * static_cast<std::uint64_t>(candidate_rows));
      read(index, 1);
      consider(index, nearest_posting_distance(
        block.data(), centers, lists, columns));
      if ((i & 255) == 0) Rcpp::checkUserInterrupt();
    }
  } else {
    std::vector<double> distances(chunk_rows);
    for (std::uint64_t first = 0; first < rows;) {
      const auto count = std::min<std::uint64_t>(chunk_rows, rows - first);
      read(first, count);
#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(workers)
#endif
      for (std::int64_t row = 0; row <
           static_cast<std::int64_t>(count); ++row) {
        distances[row] = nearest_posting_distance(
          block.data() + row * columns, centers, lists, columns);
      }
      for (std::uint64_t row = 0; row < count; ++row) {
        consider(first + row, distances[row]);
      }
      first += count;
      Rcpp::checkUserInterrupt();
    }
  }
  std::vector<RankedRow> ranked;
  ranked.reserve(sample);
  while (!farthest.empty()) {
    ranked.push_back(farthest.top());
    farthest.pop();
  }
  std::sort(ranked.rbegin(), ranked.rend());
  Rcpp::IntegerVector selected(sample);
  for (int i = 0; i < sample; ++i) selected[i] = ranked[i].second;
  return Rcpp::List::create(
    Rcpp::Named("indices") = selected,
    Rcpp::Named("provider") = "faissR",
    Rcpp::Named("backend_used") = "faissr_cpu_posting_audit",
    Rcpp::Named("method") = "distant_center_rows");
}

std::uint64_t posting_checked_product(std::uint64_t left,
                                      std::uint64_t right) {
  if (right != 0 && left >
      std::numeric_limits<std::uint64_t>::max() / right) {
    Rcpp::stop("Posting byte count exceeds native limits.");
  }
  return left * right;
}

std::uint64_t posting_whole_number(double value, const char* name) {
  if (!std::isfinite(value) || value < 0 ||
      value != std::floor(value) || value > 9007199254740991.0) {
    Rcpp::stop("%s must be an exact nonnegative integer.", name);
  }
  return static_cast<std::uint64_t>(value);
}

void posting_require_little_endian() {
  const std::uint16_t marker = 1;
  if (*reinterpret_cast<const unsigned char*>(&marker) != 1) {
    Rcpp::stop("Posting files require a little-endian host.");
  }
}

void report_graph_progress(const char* label,
                           std::uint64_t completed,
                           std::uint64_t total, int& next_percent) {
  const int percent = static_cast<int>(100.0 * completed / total);
  if (percent >= next_percent || completed == total) {
    Rcpp::Rcout << "  " << label << ": " << percent << "%\n";
    next_percent = percent + 10;
  }
}

Rcpp::List posting_root_spec(Rcpp::List spec) {
  while (Rcpp::as<std::string>(spec["format"]) == "view") {
    spec = Rcpp::as<Rcpp::List>(spec["source"]);
  }
  return spec;
}

class PostingReader {
 public:
  explicit PostingReader(Rcpp::List spec)
      : source_(batch_source(spec)) {
    if (source_.memory) Rcpp::stop(
      "Posting construction requires file-backed input.");
    Rcpp::List root = posting_root_spec(spec);
    const std::string access = Rcpp::as<std::string>(root["access"]);
    if (root.containsElementNamed("stamp")) {
      stamp_ = Rcpp::as<std::string>(root["stamp"]);
    }
    if (access == "stream") {
      input_.open(source_.file.path, std::ios::binary);
      if (!input_) Rcpp::stop("Cannot open posting input.");
    } else if (access == "mmap") {
#if defined(__unix__) || defined(__APPLE__)
      const auto bytes = std::filesystem::file_size(source_.file.path);
      if (bytes > std::numeric_limits<std::size_t>::max()) {
        Rcpp::stop("Posting input exceeds mmap address range.");
      }
      descriptor_ = open(source_.file.path.c_str(), O_RDONLY);
      if (descriptor_ < 0) Rcpp::stop("Cannot open posting input for mmap.");
      mapped_bytes_ = static_cast<std::size_t>(bytes);
      mapped_ = mmap(nullptr, mapped_bytes_, PROT_READ, MAP_PRIVATE,
                     descriptor_, 0);
      if (mapped_ == MAP_FAILED) {
        close(descriptor_);
        descriptor_ = -1;
        mapped_ = nullptr;
        Rcpp::stop("Cannot map posting input.");
      }
#else
      Rcpp::stop("Posting mmap input is unavailable on this platform.");
#endif
    } else {
      Rcpp::stop("Unknown posting input access mode.");
    }
  }

  ~PostingReader() {
#if defined(__unix__) || defined(__APPLE__)
    if (mapped_ != nullptr) munmap(mapped_, mapped_bytes_);
    if (descriptor_ >= 0) close(descriptor_);
#endif
  }

  std::uint64_t nrow() const { return source_.file.rows; }
  std::uint64_t ncol() const { return source_.file.columns; }

  void read_rows(std::uint64_t first, std::uint64_t count,
                 float* output) {
    if (first > nrow() || count > nrow() - first) {
      Rcpp::stop("Posting read exceeds source rows.");
    }
    verify_stamp();
    const auto items = posting_checked_product(count, ncol());
    const auto offset = source_.file.header +
      posting_checked_product(source_.file.first + first,
                              posting_checked_product(ncol(), 4));
    if (mapped_ != nullptr) {
      if (offset > mapped_bytes_ ||
          posting_checked_product(items, 4) > mapped_bytes_ - offset) {
        Rcpp::stop("Posting mmap read exceeds source bytes.");
      }
      const auto* start = static_cast<const unsigned char*>(mapped_) +
        offset;
      std::copy_n(reinterpret_cast<const float*>(start), items, output);
    } else {
      input_.seekg(static_cast<std::streamoff>(offset));
      input_.read(reinterpret_cast<char*>(output),
                  static_cast<std::streamsize>(items * 4));
      if (!input_) Rcpp::stop("Cannot read posting input rows.");
    }
    for (std::uint64_t item = 0; item < items; ++item) {
      if (!std::isfinite(output[item])) {
        Rcpp::stop("Posting input contains nonfinite float32 values.");
      }
    }
  }

 private:
  void verify_stamp() const {
    if (stamp_.empty()) return;
    std::error_code error;
    const auto modified = std::filesystem::last_write_time(
      source_.file.path, error);
    if (error || std::to_string(static_cast<long long>(
        modified.time_since_epoch().count())) != stamp_) {
      Rcpp::stop("Posting input changed after descriptor creation.");
    }
  }

  BatchSource source_;
  std::string stamp_;
  std::ifstream input_;
  void* mapped_ = nullptr;
  std::size_t mapped_bytes_ = 0;
  int descriptor_ = -1;
};

using Rcpp::List;
using Rcpp::NumericMatrix;
using Rcpp::NumericVector;

std::vector<float> posting_centers(NumericMatrix centers_r) {
  const auto lists = static_cast<std::size_t>(centers_r.nrow());
  const auto columns = static_cast<std::size_t>(centers_r.ncol());
  std::vector<float> centers(posting_checked_product(lists, columns));
  for (std::size_t list = 0; list < lists; ++list) {
    for (std::size_t col = 0; col < columns; ++col) {
      const double value = centers_r(list, col);
      if (!std::isfinite(value) ||
          std::abs(value) > std::numeric_limits<float>::max()) {
        Rcpp::stop("Posting centers must be finite float32 values.");
      }
      centers[list * columns + col] = static_cast<float>(value);
    }
  }
  return centers;
}

std::uint32_t coarse_posting(const float* row,
    const std::vector<float>& centers, std::size_t lists,
    std::size_t columns, double* nearest = nullptr) {
  double best = std::numeric_limits<double>::infinity();
  std::uint32_t selected = 0;
  for (std::size_t list = 0; list < lists; ++list) {
    double distance = 0;
    for (std::size_t col = 0; col < columns; ++col) {
      const double delta = static_cast<double>(row[col]) -
        centers[list * columns + col];
      distance += delta * delta;
    }
    if (distance < best) {
      best = distance;
      selected = static_cast<std::uint32_t>(list);
    }
  }
  if (nearest != nullptr) *nearest = best;
  return selected;
}

void assign_postings(const std::vector<float>& block,
    std::vector<std::uint32_t>& assignments,
    const std::vector<float>& centers, std::size_t lists,
    std::size_t columns, std::uint64_t count, int workers) {
#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(workers)
#endif
  for (std::int64_t row = 0;
       row < static_cast<std::int64_t>(count); ++row) {
    assignments[row] = coarse_posting(
      block.data() + row * columns, centers, lists, columns);
  }
}

void write_posting(std::fstream& stream, std::uint64_t offset,
    const char* data, std::uint64_t bytes) {
  if (offset > static_cast<std::uint64_t>(
      std::numeric_limits<std::streamoff>::max()) ||
      bytes > static_cast<std::uint64_t>(
      std::numeric_limits<std::streamsize>::max())) {
    Rcpp::stop("Posting offset exceeds stream limits.");
  }
  stream.seekp(static_cast<std::streamoff>(offset));
  stream.write(data, static_cast<std::streamsize>(bytes));
  if (!stream) Rcpp::stop("Posting output write failed.");
}

void verify_completed_posting_file(const std::string& path,
    std::uint64_t bytes) {
  const bool final = std::filesystem::exists(path);
  const bool part = std::filesystem::exists(path + ".part");
  if (final == part) Rcpp::stop(
    "Completed posting output has missing or duplicate files.");
  if (std::filesystem::file_size(final ? path : path + ".part") != bytes) {
    Rcpp::stop("Completed posting output has an invalid byte count.");
  }
}

void finish_completed_postings(const std::string& features,
    const std::string& ids, const std::string& offsets_path,
    const std::vector<std::uint64_t>& offsets,
    std::uint64_t feature_bytes, std::uint64_t id_bytes) {
  verify_completed_posting_file(features, feature_bytes);
  verify_completed_posting_file(ids, id_bytes);
  const auto part = offsets_path + ".part";
  if (!std::filesystem::exists(offsets_path) &&
      !std::filesystem::exists(part)) {
    std::ofstream output(part, std::ios::binary);
    output.write(reinterpret_cast<const char*>(offsets.data()),
      static_cast<std::streamsize>(offsets.size() * 8));
    output.close();
    if (!output) Rcpp::stop("Cannot write posting offsets.");
  } else {
    verify_completed_posting_file(offsets_path, offsets.size() * 8);
    const auto& existing = std::filesystem::exists(offsets_path) ?
      offsets_path : part;
    std::vector<std::uint64_t> saved(offsets.size());
    std::ifstream input(existing, std::ios::binary);
    input.read(reinterpret_cast<char*>(saved.data()),
      static_cast<std::streamsize>(saved.size() * 8));
    if (!input || saved != offsets) Rcpp::stop(
      "Completed posting offsets disagree with checkpoint.");
  }
  for (const auto& path : {features, ids, offsets_path}) {
    if (!std::filesystem::exists(path)) {
      std::filesystem::rename(path + ".part", path);
    }
  }
}


List posting_build_impl(List spec, NumericMatrix centers_r,
    std::string prefix, int chunk_rows, int workers,
    SEXP resume_state, SEXP checkpoint_callback,
    int checkpoint_every) {
  posting_require_little_endian();
  PostingReader source(spec);
  const auto rows = source.nrow();
  const auto columns = source.ncol();
  const auto lists = static_cast<std::size_t>(centers_r.nrow());
  if (rows > static_cast<std::uint64_t>(
      std::numeric_limits<int>::max()) ||
      columns != static_cast<std::uint64_t>(centers_r.ncol()) ||
      lists < 2 || lists > 65536 || chunk_rows < 1 || workers < 1 ||
      checkpoint_every < 0 ||
      (checkpoint_every == 0 && checkpoint_callback != R_NilValue) ||
      (checkpoint_every > 0 && checkpoint_callback == R_NilValue) ||
      (resume_state != R_NilValue && checkpoint_every == 0)) {
    Rcpp::stop("Posting dimensions or controls are invalid.");
  }
  const auto centers = posting_centers(centers_r);
  const std::string features = prefix + ".features.f32";
  const std::string ids = prefix + ".ids.u32";
  const std::string offsets_path = prefix + ".offsets.u64";
  const bool resume = resume_state != R_NilValue;
  const auto block_size = std::min<std::uint64_t>(rows, chunk_rows);
  std::vector<float> block(posting_checked_product(block_size, columns));
  std::vector<float> grouped(block.size());
  std::vector<std::uint32_t> assignments(block_size);
  std::vector<std::uint32_t> grouped_ids(block_size);
  std::vector<std::uint64_t> counts(lists, 0);
  std::uint64_t completed = 0;
  int next_percent = 10;
  if (resume) {
    List state(resume_state);
    NumericVector saved_counts = state["counts"];
    if (saved_counts.size() != static_cast<int>(lists)) {
      Rcpp::stop("Posting checkpoint count length is invalid.");
    }
    completed = posting_whole_number(Rcpp::as<double>(state["completed_rows"]),
                             "Posting completed rows");
    for (std::size_t list = 0; list < lists; ++list) {
      counts[list] = posting_whole_number(saved_counts[list], "Posting count");
    }
  }
  for (const auto& path : {features, ids, offsets_path}) {
    if ((std::filesystem::exists(path) &&
         (!resume || completed != rows)) ||
        (!resume && std::filesystem::exists(path + ".part"))) {
      Rcpp::stop("Posting output or partial file already exists.");
    }
  }
  if (!resume) {
    for (std::uint64_t first = 0; first < rows; first += block_size) {
      const auto count = std::min(block_size, rows - first);
      source.read_rows(first, count, block.data());
      assign_postings(block, assignments, centers, lists, columns,
                      count, workers);
      for (std::uint64_t row = 0; row < count; ++row) {
        ++counts[assignments[row]];
      }
      report_graph_progress("Posting count pass", first + count,
                            rows, next_percent);
      Rcpp::checkUserInterrupt();
    }
  }
  std::vector<std::uint64_t> offsets(lists + 1, 0);
  for (std::size_t list = 0; list < lists; ++list) {
    offsets[list + 1] = offsets[list] + counts[list];
  }
  if (offsets.back() != rows) Rcpp::stop(
    "Posting counts do not cover the source.");
  const auto feature_bytes =
    posting_checked_product(posting_checked_product(rows, columns), 4);
  const auto id_bytes = posting_checked_product(rows, 4);
  if (resume && completed == rows) {
    List state(resume_state);
    NumericVector saved_next = state["positions"];
    if (saved_next.size() != static_cast<int>(lists)) Rcpp::stop(
      "Posting checkpoint position is invalid.");
    for (std::size_t list = 0; list < lists; ++list) {
      if (posting_whole_number(saved_next[list], "Posting position") !=
          offsets[list + 1]) Rcpp::stop(
        "Posting checkpoint positions disagree with completed rows.");
    }
    finish_completed_postings(features, ids, offsets_path, offsets,
      feature_bytes, id_bytes);
    return List::create(
      Rcpp::Named("counts") = Rcpp::wrap(counts),
      Rcpp::Named("offsets") = Rcpp::wrap(offsets));
  }
  if (resume) {
    if (!std::filesystem::exists(features + ".part") ||
        !std::filesystem::exists(ids + ".part") ||
        std::filesystem::file_size(features + ".part") != feature_bytes ||
        std::filesystem::file_size(ids + ".part") != id_bytes) {
      Rcpp::stop("Posting partial files disagree with checkpoint.");
    }
  } else {
    std::ofstream(features + ".part", std::ios::binary).close();
    std::ofstream(ids + ".part", std::ios::binary).close();
    std::filesystem::resize_file(features + ".part", feature_bytes);
    std::filesystem::resize_file(ids + ".part", id_bytes);
  }
  std::fstream feature_stream(features + ".part",
    std::ios::in | std::ios::out | std::ios::binary);
  std::fstream id_stream(ids + ".part",
    std::ios::in | std::ios::out | std::ios::binary);
  if (!feature_stream || !id_stream) Rcpp::stop(
    "Cannot open posting outputs.");
  std::vector<std::uint64_t> next(offsets.begin(), offsets.end() - 1);
  if (resume) {
    List state(resume_state);
    NumericVector saved_next = state["positions"];
    if (saved_next.size() != static_cast<int>(lists) ||
        completed > rows ||
        (completed != rows && completed % block_size != 0)) {
      Rcpp::stop("Posting checkpoint position is invalid.");
    }
    std::uint64_t placed = 0;
    for (std::size_t list = 0; list < lists; ++list) {
      next[list] = posting_whole_number(saved_next[list], "Posting position");
      if (next[list] < offsets[list] ||
          next[list] > offsets[list + 1]) {
        Rcpp::stop("Posting checkpoint position is invalid.");
      }
      placed += next[list] - offsets[list];
    }
    if (placed != completed) Rcpp::stop(
      "Posting checkpoint positions disagree with completed rows.");
  }
  if (!resume && checkpoint_every > 0) {
    Rcpp::Function callback(checkpoint_callback);
    callback(0, Rcpp::wrap(counts), Rcpp::wrap(next));
  }
  std::vector<std::uint64_t> local(lists), cursor(lists);
  next_percent = 10;
  std::uint64_t blocks = completed / block_size;
  for (std::uint64_t first = completed; first < rows;
       first += block_size, ++blocks) {
    const auto count = std::min(block_size, rows - first);
    source.read_rows(first, count, block.data());
    assign_postings(block, assignments, centers, lists, columns,
                    count, workers);
    std::fill(local.begin(), local.end(), 0);
    for (std::uint64_t row = 0; row < count; ++row) {
      ++local[assignments[row]];
    }
    std::uint64_t placed = 0;
    for (std::size_t list = 0; list < lists; ++list) {
      cursor[list] = placed;
      placed += local[list];
    }
    for (std::uint64_t row = 0; row < count; ++row) {
      const auto position = cursor[assignments[row]]++;
      std::copy_n(block.data() + row * columns, columns,
                  grouped.data() + position * columns);
      grouped_ids[position] = static_cast<std::uint32_t>(first + row + 1);
    }
    placed = 0;
    for (std::size_t list = 0; list < lists; ++list) {
      if (local[list] != 0) {
        write_posting(feature_stream,
          posting_checked_product(
            posting_checked_product(next[list], columns), 4),
          reinterpret_cast<const char*>(
            grouped.data() + placed * columns),
          posting_checked_product(
            posting_checked_product(local[list], columns), 4));
        write_posting(id_stream, posting_checked_product(next[list], 4),
          reinterpret_cast<const char*>(grouped_ids.data() + placed),
          posting_checked_product(local[list], 4));
        next[list] += local[list];
      }
      placed += local[list];
    }
    report_graph_progress("Posting write pass", first + count,
                          rows, next_percent);
    if (checkpoint_every > 0 &&
        ((blocks + 1) % checkpoint_every == 0 ||
         first + count == rows)) {
      feature_stream.flush();
      id_stream.flush();
      if (!feature_stream || !id_stream) Rcpp::stop(
        "Cannot flush posting checkpoint outputs.");
      Rcpp::Function callback(checkpoint_callback);
      callback(static_cast<double>(first + count),
               Rcpp::wrap(counts), Rcpp::wrap(next));
    }
    Rcpp::checkUserInterrupt();
  }
  if (!std::equal(next.begin(), next.end(), offsets.begin() + 1)) {
    Rcpp::stop("Posting write counts do not match offsets.");
  }
  feature_stream.close();
  id_stream.close();
  std::ofstream offset_stream(offsets_path + ".part",
    std::ios::binary);
  offset_stream.write(reinterpret_cast<const char*>(offsets.data()),
    static_cast<std::streamsize>(offsets.size() * sizeof(offsets[0])));
  offset_stream.close();
  if (!offset_stream) Rcpp::stop("Cannot write posting offsets.");
  std::filesystem::rename(features + ".part", features);
  std::filesystem::rename(ids + ".part", ids);
  std::filesystem::rename(offsets_path + ".part", offsets_path);
  return List::create(
    Rcpp::Named("counts") = Rcpp::wrap(counts),
    Rcpp::Named("offsets") = Rcpp::wrap(offsets)
  );
}

List knn_rows_read_impl(List args) {
  posting_require_little_endian();
  const std::string indices_path = args["indices_path"];
  const std::string distances_path = args["distances_path"];
  const auto rows = posting_whole_number(
    Rcpp::as<double>(args["n_rows"]), "n_rows");
  const auto references = posting_whole_number(
    Rcpp::as<double>(args["n_reference"]), "n_reference");
  const auto first = posting_whole_number(
    Rcpp::as<double>(args["first_row"]), "first_row");
  const int k = Rcpp::as<int>(args["k"]);
  const int count = Rcpp::as<int>(args["count"]);
  const double max_bytes = Rcpp::as<double>(args["max_bytes"]);
  if (rows < 1 || references < 1 ||
      references > static_cast<std::uint64_t>(INT_MAX) ||
      k < 1 || count < 1 || first < 1 || first > rows ||
      static_cast<std::uint64_t>(count) > rows - first + 1) {
    Rcpp::stop("Invalid massive KNN row range or dimensions.");
  }
  const auto items = posting_checked_product(count, k);
  if (!std::isfinite(max_bytes) || max_bytes <= 0 ||
      items > static_cast<std::uint64_t>(std::floor(max_bytes / 20))) {
    Rcpp::stop("Massive KNN row read exceeds its byte limit.");
  }
  const auto expected = posting_checked_product(
    posting_checked_product(rows, k), 4);
  const auto offset = posting_checked_product(
    posting_checked_product(first - 1, k), 4);
  if (offset > static_cast<std::uint64_t>(
      std::numeric_limits<std::streamoff>::max())) {
    Rcpp::stop("Massive KNN file offset exceeds stream limits.");
  }
  std::error_code error;
  const auto index_size = std::filesystem::file_size(
    indices_path, error);
  if (error || index_size != expected) {
    Rcpp::stop("Massive KNN index file size mismatch.");
  }
  const auto distance_size = std::filesystem::file_size(
    distances_path, error);
  if (error || distance_size != expected) {
    Rcpp::stop("Massive KNN distance file size mismatch.");
  }
  std::ifstream indices_stream(indices_path, std::ios::binary);
  std::ifstream distances_stream(distances_path, std::ios::binary);
  indices_stream.seekg(static_cast<std::streamoff>(offset));
  distances_stream.seekg(static_cast<std::streamoff>(offset));
  std::vector<std::uint32_t> ids(items);
  std::vector<float> values(items);
  const auto bytes = static_cast<std::streamsize>(items * 4);
  indices_stream.read(reinterpret_cast<char*>(ids.data()), bytes);
  distances_stream.read(reinterpret_cast<char*>(values.data()), bytes);
  if (!indices_stream || !distances_stream) {
    Rcpp::stop("Cannot read massive KNN row range.");
  }
  Rcpp::IntegerMatrix indices(count, k);
  Rcpp::NumericMatrix distances(count, k);
  for (int row = 0; row < count; ++row) {
    for (int col = 0; col < k; ++col) {
      const auto position = static_cast<std::size_t>(row) * k + col;
      const auto id = ids[position];
      const auto distance = values[position];
      if (id < 1 || id > references ||
          !std::isfinite(distance) || distance < 0) {
        Rcpp::stop("Massive KNN row contains invalid values.");
      }
      indices(row, col) = static_cast<int>(id);
      distances(row, col) = distance;
    }
  }
  return List::create(
    Rcpp::Named("indices") = indices,
    Rcpp::Named("distances") = distances,
    Rcpp::Named("provider") = "faissR",
    Rcpp::Named("backend_used") = "faissr_cpu_knn_rows_read",
    Rcpp::Named("method") = "bounded_file_rows");
}


}  // namespace

extern "C" SEXP faissR_posting_search_v1(SEXP query_spec,
    SEXP feature, SEXP ids, SEXP offsets, SEXP centers, SEXP first,
    SEXP count, SEXP k, SEXP probes, SEXP chunk, SEXP workers,
    SEXP exclude_self, SEXP rows, SEXP verbose) {
  BEGIN_RCPP
  return posting_search_impl(query_spec, feature, ids, offsets, centers,
    first, count, k, probes, chunk, workers, exclude_self, rows, verbose);
  END_RCPP
}

extern "C" SEXP faissR_posting_distant_rows_v1(SEXP arguments) {
  BEGIN_RCPP
  return posting_distant_rows_impl(arguments);
  END_RCPP
}

extern "C" SEXP faissR_posting_build_v1(SEXP arguments) {
  BEGIN_RCPP
  Rcpp::List args(arguments);
  Rcpp::List result = posting_build_impl(
    Rcpp::as<Rcpp::List>(args["spec"]),
    Rcpp::as<Rcpp::NumericMatrix>(args["centers"]),
    Rcpp::as<std::string>(args["prefix"]),
    Rcpp::as<int>(args["chunk_rows"]),
    Rcpp::as<int>(args["workers"]),
    args["resume_state"], args["checkpoint_callback"],
    Rcpp::as<int>(args["checkpoint_every"]));
  result["provider"] = "faissR";
  result["backend_used"] = "faissr_cpu_posting_build";
  result["method"] = "nearest_center_physical_postings";
  return result;
  END_RCPP
}

extern "C" SEXP faissR_knn_rows_read_v1(SEXP arguments) {
  BEGIN_RCPP
  return knn_rows_read_impl(Rcpp::List(arguments));
  END_RCPP
}
