#include <Rcpp.h>

#include <algorithm>
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
#include <faiss/IndexFlat.h>
#endif

namespace {

struct FileMatrix {
  std::string path;
  std::uint64_t rows;
  std::uint64_t columns;
  std::uint64_t first;
  std::uint64_t header;
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

void write_batch(std::ofstream& indices, std::ofstream& distances,
                 std::vector<std::priority_queue<Candidate>>& heaps,
                 std::size_t k) {
  std::vector<std::uint32_t> ids(k);
  std::vector<float> values(k);
  for (auto& heap : heaps) {
    if (heap.size() != k) Rcpp::stop("Incomplete file-backed KNN row");
    for (std::size_t j = k; j > 0; --j) {
      const Candidate candidate = heap.top();
      heap.pop();
      ids[j - 1] = candidate.second;
      values[j - 1] = std::sqrt(std::max(0.0f, candidate.first));
    }
    indices.write(reinterpret_cast<const char*>(ids.data()),
                  static_cast<std::streamsize>(k * 4));
    distances.write(reinterpret_cast<const char*>(values.data()),
                    static_cast<std::streamsize>(k * 4));
    if (!indices || !distances) Rcpp::stop("KNN output write failed");
  }
}

Rcpp::List stream_exact(Rcpp::List query_spec, Rcpp::List reference_spec,
                        int k, const std::string& output,
                        int query_chunk, int reference_chunk,
                        double memory_limit, int n_threads) {
#ifndef FAISSR_HAS_FAISS
  Rcpp::stop("File-backed KNN requires a functional FAISS build");
#else
  const FileMatrix query = file_matrix(query_spec);
  const FileMatrix reference = file_matrix(reference_spec);
  if (query.columns != reference.columns || k < 1 ||
      static_cast<std::uint64_t>(k) > reference.rows ||
      query_chunk < 1 || reference_chunk < 1 || n_threads < 1) {
    Rcpp::stop("Invalid file-backed KNN dimensions or settings");
  }
  const double bytes = 4.0 * query_chunk * query.columns +
    4.0 * reference_chunk * reference.columns +
    32.0 * query_chunk * k + 64.0 * 1024 * 1024;
  if (!std::isfinite(memory_limit) || memory_limit < bytes) {
    Rcpp::stop("File-backed KNN buffers exceed memory_limit");
  }
  const std::string id_path = output + ".indices.u32";
  const std::string distance_path = output + ".distances.f32";
  const std::string id_part = id_path + ".part";
  const std::string distance_part = distance_path + ".part";
  const auto parent = std::filesystem::path(output).parent_path();
  const auto directory = parent.empty() ? "." : parent;
  const double output_bytes = 8.0 * query.rows * k;
  if (!std::filesystem::is_directory(directory) ||
      output_bytes > 0.9 * std::filesystem::space(directory).available) {
    Rcpp::stop("File-backed KNN output exceeds available disk space");
  }
  for (const auto& path : {id_path, distance_path, id_part,
                           distance_part}) {
    if (std::filesystem::exists(path)) {
      Rcpp::stop("File-backed KNN output already exists: %s",
                 path.c_str());
    }
  }
  std::ifstream query_file(query.path, std::ios::binary);
  std::ifstream reference_file(reference.path, std::ios::binary);
  std::ofstream ids(id_part, std::ios::binary | std::ios::trunc);
  std::ofstream distances(distance_part,
                          std::ios::binary | std::ios::trunc);
  if (!query_file || !reference_file || !ids || !distances) {
    Rcpp::stop("Could not open file-backed KNN input or output");
  }
  const auto query_mtime =
    std::filesystem::last_write_time(query.path);
  const auto reference_mtime =
    std::filesystem::last_write_time(reference.path);
  ThreadLimit threads(n_threads);
  std::vector<float> query_values;
  std::vector<float> reference_values;
  for (std::uint64_t first = 0; first < query.rows;
       first += query_chunk) {
    Rcpp::checkUserInterrupt();
    const auto rows = std::min<std::uint64_t>(query_chunk,
                                              query.rows - first);
    read_rows(query_file, query, first, rows, query_values);
    std::vector<std::priority_queue<Candidate>> heaps(rows);
    for (std::uint64_t base = 0; base < reference.rows;
         base += reference_chunk) {
      const auto count = std::min<std::uint64_t>(reference_chunk,
                                                reference.rows - base);
      read_rows(reference_file, reference, base, count,
                reference_values);
      faiss::IndexFlatL2 index(static_cast<int>(reference.columns));
      index.add(count, reference_values.data());
      const int width = static_cast<int>(std::min<std::uint64_t>(k,
                                                                 count));
      std::vector<float> found_distances(rows * width);
      std::vector<faiss::idx_t> found_ids(rows * width);
      index.search(rows, query_values.data(), width,
                   found_distances.data(), found_ids.data());
      for (std::uint64_t i = 0; i < rows; ++i) {
        for (int j = 0; j < width; ++j) {
          const std::size_t pos = i * width + j;
          add_candidate(heaps[i], Candidate{
            found_distances[pos], static_cast<std::uint32_t>(
              base + found_ids[pos] + 1)}, k);
        }
      }
    }
    write_batch(ids, distances, heaps, k);
  }
  ids.close();
  distances.close();
  if (std::filesystem::last_write_time(query.path) != query_mtime ||
      std::filesystem::last_write_time(reference.path) !=
        reference_mtime) {
    Rcpp::stop("File-backed KNN input changed during search");
  }
  const auto expected = query.rows * static_cast<std::uint64_t>(k) * 4;
  if (std::filesystem::file_size(id_part) != expected ||
      std::filesystem::file_size(distance_part) != expected) {
    Rcpp::stop("File-backed KNN output has the wrong size");
  }
  std::filesystem::rename(id_part, id_path);
  try {
    std::filesystem::rename(distance_part, distance_path);
  } catch (...) {
    std::filesystem::rename(id_path, id_part);
    throw;
  }
  return Rcpp::List::create(
    Rcpp::Named("indices_path") = id_path,
    Rcpp::Named("distances_path") = distance_path,
    Rcpp::Named("nrow") = static_cast<double>(query.rows),
    Rcpp::Named("ncol") = k,
    Rcpp::Named("backend_used") = "faiss_cpu_stream_exact",
    Rcpp::Named("exact") = true,
    Rcpp::Named("index_reused") = false,
    Rcpp::Named("query_chunk_rows") = query_chunk,
    Rcpp::Named("reference_chunk_rows") = reference_chunk);
#endif
}

}  // namespace

extern "C" SEXP faissR_file_knn_v1(SEXP query, SEXP reference,
    SEXP k, SEXP output, SEXP backend, SEXP method,
    SEXP query_chunk, SEXP reference_chunk, SEXP memory_limit,
    SEXP n_threads) {
  BEGIN_RCPP
  const std::string route = Rcpp::as<std::string>(backend);
  const std::string algorithm = Rcpp::as<std::string>(method);
  if (route != "cpu" || algorithm != "stream_exact") {
    Rcpp::stop("Requested file-backed KNN backend/method is unavailable");
  }
  return stream_exact(Rcpp::as<Rcpp::List>(query),
    Rcpp::as<Rcpp::List>(reference), Rcpp::as<int>(k),
    Rcpp::as<std::string>(output), Rcpp::as<int>(query_chunk),
    Rcpp::as<int>(reference_chunk), Rcpp::as<double>(memory_limit),
    Rcpp::as<int>(n_threads));
  END_RCPP
}
