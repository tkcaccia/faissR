#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef FAISSR_HAS_CUGRAPH
#define FALSE CUGRAPH_FALSE
#define TRUE CUGRAPH_TRUE
#include <cugraph_c/array.h>
#include <cugraph_c/community_algorithms.h>
#include <cugraph_c/error.h>
#include <cugraph_c/graph.h>
#include <cugraph_c/random.h>
#include <cugraph_c/resource_handle.h>
#undef FALSE
#undef TRUE
#endif

#include <Rcpp.h>

using Rcpp::IntegerVector;
using Rcpp::List;
using Rcpp::NumericVector;

#ifdef FAISSR_HAS_CUGRAPH
namespace {

std::string cugraph_error_string(cugraph_error_t* error) {
  std::string msg = error ? cugraph_error_message(error) : "unknown cuGraph error";
  if (error) cugraph_error_free(error);
  return msg;
}

void check_cugraph(cugraph_error_code_t code, cugraph_error_t* error, const char* what) {
  if (code != CUGRAPH_SUCCESS) {
    Rcpp::stop(std::string(what) + ": " + cugraph_error_string(error));
  }
  if (error) cugraph_error_free(error);
}

template <typename T>
struct DeviceArray {
  cugraph_type_erased_device_array_t* array = nullptr;
  cugraph_type_erased_device_array_view_t* view = nullptr;
  cugraph_data_type_id_t dtype;
  DeviceArray(const cugraph_resource_handle_t* handle,
              const std::vector<T>& host,
              cugraph_data_type_id_t dtype_) : dtype(dtype_) {
    cugraph_error_t* error = nullptr;
    check_cugraph(cugraph_type_erased_device_array_create(handle, host.size(), dtype, &array, &error), error, "cuGraph device array create");
    view = cugraph_type_erased_device_array_view(array);
    check_cugraph(cugraph_type_erased_device_array_view_copy_from_host(handle, view, reinterpret_cast<const byte_t*>(host.data()), &error), error, "cuGraph host-to-device copy");
  }
  ~DeviceArray() {
    if (view) cugraph_type_erased_device_array_view_free(view);
    if (array) cugraph_type_erased_device_array_free(array);
  }
  DeviceArray(const DeviceArray&) = delete;
  DeviceArray& operator=(const DeviceArray&) = delete;
};

struct GraphHandle {
  cugraph_graph_t* graph = nullptr;
  ~GraphHandle() {
    free();
  }
  void free() {
    if (graph) {
      cugraph_graph_free(graph);
      graph = nullptr;
    }
  }
  GraphHandle(const GraphHandle&) = delete;
  GraphHandle& operator=(const GraphHandle&) = delete;
};

std::vector<int32_t> copy_int32_result(const cugraph_resource_handle_t* handle,
                                       cugraph_type_erased_device_array_view_t* view) {
  const size_t n = cugraph_type_erased_device_array_view_size(view);
  std::vector<int32_t> out(n);
  cugraph_error_t* error = nullptr;
  check_cugraph(cugraph_type_erased_device_array_view_copy_to_host(handle, reinterpret_cast<byte_t*>(out.data()), view, &error), error, "cuGraph device-to-host copy");
  return out;
}

List run_cugraph_community(List edge_list,
                           const std::string& method,
                           int n_runs,
                           double resolution,
                           int n_iterations,
                           int seed) {
  IntegerVector from = edge_list["from"];
  IntegerVector to = edge_list["to"];
  NumericVector weight = edge_list["weight"];
  const int n_vertices = Rcpp::as<int>(edge_list["n_vertices"]);
  const int n_edges = from.size();
  if (n_edges <= 0) Rcpp::stop("cuGraph clustering requires a graph with at least one edge.");

  std::vector<int32_t> src(static_cast<std::size_t>(n_edges));
  std::vector<int32_t> dst(static_cast<std::size_t>(n_edges));
  std::vector<float> w(static_cast<std::size_t>(n_edges));
  for (int i = 0; i < n_edges; ++i) {
    src[static_cast<std::size_t>(i)] = static_cast<int32_t>(from[i] - 1);
    dst[static_cast<std::size_t>(i)] = static_cast<int32_t>(to[i] - 1);
    w[static_cast<std::size_t>(i)] = static_cast<float>(weight[i]);
  }
  std::vector<int32_t> vertices(static_cast<std::size_t>(n_vertices));
  for (int i = 0; i < n_vertices; ++i) vertices[static_cast<std::size_t>(i)] = static_cast<int32_t>(i);

  cugraph_resource_handle_t* handle = cugraph_create_resource_handle(nullptr);
  if (!handle) Rcpp::stop("cuGraph resource handle creation failed.");

  try {
    DeviceArray<int32_t> d_vertices(handle, vertices, INT32);
    DeviceArray<int32_t> d_src(handle, src, INT32);
    DeviceArray<int32_t> d_dst(handle, dst, INT32);
    DeviceArray<float> d_weight(handle, w, FLOAT32);

    cugraph_graph_properties_t props;
    props.is_symmetric = static_cast<bool_t>(1);
    props.is_multigraph = static_cast<bool_t>(0);
    GraphHandle graph;
    cugraph_error_t* error = nullptr;
    check_cugraph(cugraph_graph_create_sg(handle, &props, d_vertices.view, d_src.view, d_dst.view, d_weight.view,
                                          nullptr, nullptr, static_cast<bool_t>(0), static_cast<bool_t>(0), static_cast<bool_t>(1), static_cast<bool_t>(1), static_cast<bool_t>(0), static_cast<bool_t>(0), &graph.graph, &error),
                  error, "cuGraph graph create");

    std::vector<int32_t> best_membership;
    double best_modularity = -1e300;
    NumericVector all_modularity(n_runs);
    for (int run = 0; run < n_runs; ++run) {
      cugraph_hierarchical_clustering_result_t* result = nullptr;
      if (method == "louvain") {
        check_cugraph(cugraph_louvain(handle, graph.graph, static_cast<size_t>(std::max(1, n_iterations)), 1e-7, resolution, static_cast<bool_t>(0), &result, &error),
                      error, "cuGraph Louvain");
      } else if (method == "leiden") {
        cugraph_rng_state_t* rng = nullptr;
        check_cugraph(cugraph_rng_state_create(handle, static_cast<uint64_t>(seed + run * 104729), &rng, &error), error, "cuGraph RNG create");
        check_cugraph(cugraph_leiden(handle, rng, graph.graph, static_cast<size_t>(std::max(1, n_iterations)), resolution, 0.01, static_cast<bool_t>(0), &result, &error),
                      error, "cuGraph Leiden");
        cugraph_rng_state_free(rng);
      } else {
        Rcpp::stop("CUDA random_walking requires a dedicated cuGraph random-walk clustering adapter; use backend = 'cpu' for this method for now.");
      }

      const double modularity = cugraph_hierarchical_clustering_result_get_modularity(result);
      all_modularity[run] = modularity;
      if (modularity > best_modularity || best_membership.empty()) {
        cugraph_type_erased_device_array_view_t* vertices_view = cugraph_hierarchical_clustering_result_get_vertices(result);
        cugraph_type_erased_device_array_view_t* clusters_view = cugraph_hierarchical_clustering_result_get_clusters(result);
        std::vector<int32_t> result_vertices = copy_int32_result(handle, vertices_view);
        std::vector<int32_t> result_clusters = copy_int32_result(handle, clusters_view);
        best_membership.assign(static_cast<std::size_t>(n_vertices), 0);
        for (std::size_t i = 0; i < result_vertices.size(); ++i) {
          const int v = static_cast<int>(result_vertices[i]);
          if (v >= 0 && v < n_vertices) best_membership[static_cast<std::size_t>(v)] = result_clusters[i];
        }
        best_modularity = modularity;
      }
      cugraph_hierarchical_clustering_result_free(result);
    }
    graph.free();
    cugraph_free_resource_handle(handle);

    std::vector<int32_t> unique = best_membership;
    std::sort(unique.begin(), unique.end());
    unique.erase(std::unique(unique.begin(), unique.end()), unique.end());
    IntegerVector membership(n_vertices);
    for (int i = 0; i < n_vertices; ++i) {
      const auto it = std::lower_bound(unique.begin(), unique.end(), best_membership[static_cast<std::size_t>(i)]);
      membership[i] = static_cast<int>(std::distance(unique.begin(), it)) + 1;
    }
    return List::create(
      Rcpp::Named("membership") = membership,
      Rcpp::Named("modularity") = best_modularity,
      Rcpp::Named("n_communities") = static_cast<int>(unique.size()),
      Rcpp::Named("method") = method,
      Rcpp::Named("backend") = "cuda",
      Rcpp::Named("n_threads") = 1,
      Rcpp::Named("graph") = edge_list,
      Rcpp::Named("n_runs") = n_runs,
      Rcpp::Named("selected_run") = static_cast<int>(std::distance(all_modularity.begin(), std::max_element(all_modularity.begin(), all_modularity.end()))) + 1,
      Rcpp::Named("all_modularity") = all_modularity,
      Rcpp::Named("implementation") = "rapids_libcugraph"
    );
  } catch (...) {
    cugraph_free_resource_handle(handle);
    throw;
  }
}

} // namespace
#endif

List graph_cluster_cugraph_edges_cpp(List edge_list,
                                     std::string method,
                                     int n_runs,
                                     double resolution,
                                     int n_iterations,
                                     int steps,
                                     int seed) {
#ifdef FAISSR_HAS_CUGRAPH
  return run_cugraph_community(edge_list, method, std::max(1, n_runs), resolution, n_iterations, seed);
#else
  Rcpp::stop("CUDA graph clustering requires RAPIDS libcugraph at build time. Install libcugraph and rebuild faissR with FAISSR_USE_CUGRAPH=1, or use backend = 'cpu'.");
#endif
}
