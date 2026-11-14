#include <Rcpp.h>

#include <algorithm>
#include <chrono>
#include <climits>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <limits>
#include <queue>
#include <string>
#include <utility>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

#ifdef FAISSR_HAS_FAISS
#include <faiss/utils/distances.h>
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
  SEXP owner = R_NilValue;
  bool memory = false;
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
    source.data = Rcpp::as<Rcpp::NumericMatrix>(source.owner);
    source.file.rows = positive_integer(spec["nrow"], "nrow");
    source.file.columns = positive_integer(spec["ncol"], "ncol");
    if (source.data.nrow() != source.file.rows ||
        source.data.ncol() != source.file.columns) {
      Rcpp::stop("Resident KNN matrix differs from descriptor");
    }
    source.file.first = 0;
    source.memory = true;
  } else {
    source.file = file_matrix(spec);
  }
  return source;
}

void read_rows(std::ifstream& input, const FileMatrix& spec,
               std::uint64_t first, std::uint64_t count,
               std::vector<float>& values) {
  const std::uint64_t items = count * spec.columns;
  const std::uint64_t offset = spec.header +
    (spec.first + first) * spec.columns * 4;
  values.resize(static_cast<std::size_t>(items));
  input.seekg(static_cast<std::streamoff>(offset));
  input.read(reinterpret_cast<char*>(values.data()),
    static_cast<std::streamsize>(items * 4));
  if (!input) Rcpp::stop("Could not read file-backed rows");
  for (float value : values) {
    if (!std::isfinite(value)) {
      Rcpp::stop("File-backed KNN input contains nonfinite values");
    }
  }
}

void read_batch_rows(std::ifstream& input, const BatchSource& source,
                     std::uint64_t first, std::uint64_t count,
                     std::vector<float>& values) {
  if (!source.memory) {
    read_rows(input, source.file, first, count, values);
    return;
  }
  values.resize(static_cast<std::size_t>(count * source.file.columns));
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

using Candidate = std::pair<float, std::uint32_t>;

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

}  // namespace

extern "C" SEXP faissR_matrix_knn_batch_v1(SEXP query_spec,
    SEXP reference_spec, SEXP row_ids, SEXP k_value, SEXP backend,
    SEXP method, SEXP reference_chunk_value, SEXP memory_limit_value,
    SEXP n_threads_value, SEXP exclude_self_value) {
  BEGIN_RCPP
#ifndef FAISSR_HAS_FAISS
  Rcpp::stop("File-backed KNN requires a functional FAISS build");
#else
  if (Rcpp::as<std::string>(backend) != "cpu" ||
      Rcpp::as<std::string>(method) != "stream_exact") {
    Rcpp::stop("Requested file-backed KNN backend/method is unavailable");
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
      (exclude_self && !same_source)) {
    Rcpp::stop("Invalid file-backed KNN batch dimensions or settings");
  }
  const std::size_t count = requested.size();
  const double resident_bytes =
    (query.memory ? 8.0 * query.data.nrow() * query.data.ncol() : 0.0) +
    (reference.memory && (!query.memory || !same_owner) ?
      8.0 * reference.data.nrow() * reference.data.ncol() : 0.0);
  const double estimated = resident_bytes +
    4.0 * count * query.file.columns +
    4.0 * reference_chunk * reference.file.columns +
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
  ThreadLimit threads(n_threads);
  std::vector<std::priority_queue<Candidate>> heaps(count);
  std::vector<float> reference_values;
  for (std::uint64_t base = 0; base < reference.file.rows;
       base += reference_chunk) {
    const auto rows = std::min<std::uint64_t>(reference_chunk,
                                              reference.file.rows - base);
    read_batch_rows(reference_file, reference, base, rows,
                    reference_values);
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (std::int64_t qi = 0;
         qi < static_cast<std::int64_t>(count); ++qi) {
      const float* point = query_values.data() + qi * query.file.columns;
      auto& heap = heaps[qi];
      for (std::uint64_t ri = 0; ri < rows; ++ri) {
        if (exclude_self && selected[qi] == base + ri) continue;
        const float distance = faiss::fvec_L2sqr(point,
          reference_values.data() + ri * reference.file.columns,
          static_cast<std::size_t>(query.file.columns));
        add_candidate(heap, Candidate{distance,
          static_cast<std::uint32_t>(base + ri + 1)}, k);
      }
    }
    Rcpp::checkUserInterrupt();
  }
  if ((!query.memory &&
       std::filesystem::last_write_time(query.file.path) != query_mtime) ||
      (!reference.memory &&
       std::filesystem::last_write_time(reference.file.path) !=
         reference_mtime)) {
    Rcpp::stop("File-backed KNN input changed during search");
  }
  Rcpp::IntegerMatrix indices(count, k);
  Rcpp::NumericMatrix distances(count, k);
  for (std::size_t qi = 0; qi < count; ++qi) {
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
      distances(qi, rank) = std::sqrt(std::max(0.0f,
        candidate.first));
    }
  }
  const double seconds = std::chrono::duration<double>(
    std::chrono::steady_clock::now() - started).count();
  return Rcpp::List::create(
    Rcpp::Named("indices") = indices,
    Rcpp::Named("distances") = distances,
    Rcpp::Named("provider") = "faissR",
    Rcpp::Named("backend_used") = "faiss_cpu_stream_exact",
    Rcpp::Named("method") = "stream_exact",
    Rcpp::Named("residency") = query.memory && reference.memory ?
      "resident" : (query.memory || reference.memory ?
        "mixed" : "file-backed"),
    Rcpp::Named("device_ids") = Rcpp::IntegerVector(),
    Rcpp::Named("exact") = true,
    Rcpp::Named("index_build_seconds") = 0.0,
    Rcpp::Named("search_seconds") = seconds,
    Rcpp::Named("peak_ram_bytes_estimate") = estimated,
    Rcpp::Named("peak_device_bytes") = 0.0);
#endif
  END_RCPP
}
