#include <Rcpp.h>

#include <string>

using Rcpp::IntegerMatrix;
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

const char* cuda_error_message() {
  const char* msg = faissr_cuda_last_error();
  return msg == nullptr ? "unknown CUDA error" : msg;
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

List cuda_grid_self_knn_impl(NumericMatrix data,
                             int k,
                             int bins_per_dim) {
  const int n = data.nrow();
  const int n_features = data.ncol();
  if (n_features != 2 && n_features != 3) {
    Rcpp::stop("CUDA grid KNN requires a two- or three-column matrix");
  }
  if (n < 2) Rcpp::stop("data must have at least two rows");
  if (k < 1 || k >= n) Rcpp::stop("k must be in [1, nrow(data) - 1]");
  if (k > kMaxCudaK) Rcpp::stop("CUDA backend currently supports k <= %d", kMaxCudaK);
  if (bins_per_dim < 1) Rcpp::stop("bins_per_dim must be positive");
  if (!faissr_cuda_available()) Rcpp::stop("No CUDA device is available.");

  IntegerMatrix indices(n, k);
  NumericMatrix distances(n, k);
  int n_cells = 0;
  const int status = faissr_cuda_grid_self_knn(
    data.begin(),
    n,
    n_features,
    k,
    bins_per_dim,
    indices.begin(),
    distances.begin(),
    &n_cells
  );
  if (status != 0) {
    Rcpp::stop("CUDA grid KNN failed: %s", cuda_error_message());
  }

  List result = List::create(
    Rcpp::Named("indices") = indices,
    Rcpp::Named("distances") = distances,
    Rcpp::Named("bins_per_dim") = bins_per_dim,
    Rcpp::Named("n_cells") = n_cells
  );
  result.attr("cuda_kernel") = n_features == 3 ? "grid3d_self_knn" : "grid2d_self_knn";
  return result;
}
