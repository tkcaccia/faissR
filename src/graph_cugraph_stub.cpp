#include <Rcpp.h>
using Rcpp::List;

List graph_cluster_cugraph_edges_cpp(List edge_list,
                                     std::string method,
                                     int n_runs,
                                     double resolution,
                                     int n_iterations,
                                     int steps,
                                     int seed) {
  Rcpp::stop("CUDA graph clustering requires RAPIDS libcugraph at build time. Install libcugraph and rebuild faissR with FAISSR_USE_CUGRAPH=1, or use backend = 'cpu'.");
}
