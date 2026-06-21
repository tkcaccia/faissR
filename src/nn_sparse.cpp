#include <Rcpp.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <queue>
#include <string>
#include <utility>
#include <vector>

using Rcpp::IntegerMatrix;
using Rcpp::IntegerVector;
using Rcpp::List;
using Rcpp::NumericMatrix;
using Rcpp::NumericVector;
using Rcpp::S4;

namespace {

struct SparseRows {
  int nrow = 0;
  int ncol = 0;
  std::vector<int> row_ptr;
  std::vector<int> col_idx;
  std::vector<double> values;
  std::vector<double> sum;
  std::vector<double> norm_sq;
};

struct HeapNeighbor {
  double distance;
  int index;
};

struct WorstFirst {
  bool operator()(const HeapNeighbor& a, const HeapNeighbor& b) const {
    if (a.distance == b.distance) return a.index < b.index;
    return a.distance < b.distance;
  }
};

bool best_less(const HeapNeighbor& a, const HeapNeighbor& b) {
  if (a.distance == b.distance) return a.index < b.index;
  return a.distance < b.distance;
}

SparseRows dgC_to_rows(const S4& x) {
  IntegerVector dims = x.slot("Dim");
  IntegerVector i = x.slot("i");
  IntegerVector p = x.slot("p");
  NumericVector v = x.slot("x");
  const int nrow = dims[0];
  const int ncol = dims[1];
  if (nrow < 1 || ncol < 1) {
    Rcpp::stop("sparse matrices must have at least one row and one column");
  }
  if (p.size() != ncol + 1) {
    Rcpp::stop("invalid dgCMatrix column pointer");
  }
  SparseRows out;
  out.nrow = nrow;
  out.ncol = ncol;
  out.row_ptr.assign(static_cast<std::size_t>(nrow) + 1, 0);
  out.sum.assign(nrow, 0.0);
  out.norm_sq.assign(nrow, 0.0);

  for (int col = 0; col < ncol; ++col) {
    for (int pos = p[col]; pos < p[col + 1]; ++pos) {
      const int row = i[pos];
      if (row < 0 || row >= nrow) {
        Rcpp::stop("invalid dgCMatrix row index");
      }
      const double value = v[pos];
      if (!std::isfinite(value)) {
        Rcpp::stop("sparse matrices must contain only finite values");
      }
      ++out.row_ptr[static_cast<std::size_t>(row) + 1];
      out.sum[row] += value;
      out.norm_sq[row] += value * value;
    }
  }
  for (int row = 0; row < nrow; ++row) {
    out.row_ptr[static_cast<std::size_t>(row) + 1] += out.row_ptr[row];
  }
  out.col_idx.assign(out.row_ptr.back(), 0);
  out.values.assign(out.row_ptr.back(), 0.0);
  std::vector<int> cursor = out.row_ptr;
  for (int col = 0; col < ncol; ++col) {
    for (int pos = p[col]; pos < p[col + 1]; ++pos) {
      const int row = i[pos];
      const int dest = cursor[row]++;
      out.col_idx[dest] = col;
      out.values[dest] = v[pos];
    }
  }
  return out;
}

double sparse_dot(const SparseRows& a, const int row_a,
                  const SparseRows& b, const int row_b) {
  int pa = a.row_ptr[row_a];
  const int enda = a.row_ptr[row_a + 1];
  int pb = b.row_ptr[row_b];
  const int endb = b.row_ptr[row_b + 1];
  double acc = 0.0;
  while (pa < enda && pb < endb) {
    const int ca = a.col_idx[pa];
    const int cb = b.col_idx[pb];
    if (ca == cb) {
      acc += a.values[pa] * b.values[pb];
      ++pa;
      ++pb;
    } else if (ca < cb) {
      ++pa;
    } else {
      ++pb;
    }
  }
  return acc;
}

double sparse_distance(const SparseRows& data,
                       const SparseRows& points,
                       const int data_row,
                       const int point_row,
                       const std::string& metric) {
  const double dot = sparse_dot(data, data_row, points, point_row);
  if (metric == "euclidean") {
    const double sq = data.norm_sq[data_row] + points.norm_sq[point_row] - 2.0 * dot;
    return std::sqrt(std::max(0.0, sq));
  }
  if (metric == "cosine") {
    const double dn = std::sqrt(std::max(0.0, data.norm_sq[data_row]));
    const double qn = std::sqrt(std::max(0.0, points.norm_sq[point_row]));
    if (dn <= 0.0 && qn <= 0.0) return 0.0;
    if (dn <= 0.0 || qn <= 0.0) return 1.0;
    double sim = dot / (dn * qn);
    sim = std::max(-1.0, std::min(1.0, sim));
    return 1.0 - sim;
  }
  if (metric == "correlation") {
    const double p = static_cast<double>(data.ncol);
    const double centered_dot =
      dot - (data.sum[data_row] * points.sum[point_row] / p);
    const double data_centered =
      data.norm_sq[data_row] - (data.sum[data_row] * data.sum[data_row] / p);
    const double point_centered =
      points.norm_sq[point_row] - (points.sum[point_row] * points.sum[point_row] / p);
    const double dn = std::sqrt(std::max(0.0, data_centered));
    const double qn = std::sqrt(std::max(0.0, point_centered));
    if (dn <= 0.0 && qn <= 0.0) return 0.0;
    if (dn <= 0.0 || qn <= 0.0) return 1.0;
    double sim = centered_dot / (dn * qn);
    sim = std::max(-1.0, std::min(1.0, sim));
    return 1.0 - sim;
  }
  if (metric == "inner_product") {
    return -dot;
  }
  Rcpp::stop("unsupported sparse KNN metric");
}

} // namespace

// [[Rcpp::export]]
List sparse_nn_cpp(S4 data,
                   S4 points,
                   int k,
                   std::string metric,
                   bool exclude_self,
                   bool self_query) {
  SparseRows data_rows = dgC_to_rows(data);
  SparseRows point_rows = dgC_to_rows(points);
  if (data_rows.ncol != point_rows.ncol) {
    Rcpp::stop("`data` and `points` must have the same number of columns");
  }
  if (k < 1 || k > data_rows.nrow) {
    Rcpp::stop("`k` must be in [1, nrow(data)]");
  }
  if (exclude_self && !self_query) {
    Rcpp::stop("self-neighbor exclusion requires a self-KNN search");
  }
  if (metric != "euclidean" && metric != "cosine" &&
      metric != "correlation" && metric != "inner_product") {
    Rcpp::stop("unsupported sparse KNN metric");
  }

  const int n_points = point_rows.nrow;
  IntegerMatrix indices(n_points, k);
  NumericMatrix distances(n_points, k);

  for (int query = 0; query < n_points; ++query) {
    std::priority_queue<HeapNeighbor, std::vector<HeapNeighbor>, WorstFirst> heap;
    for (int ref = 0; ref < data_rows.nrow; ++ref) {
      if (exclude_self && self_query && ref == query) continue;
      const double dist = sparse_distance(data_rows, point_rows, ref, query, metric);
      if (static_cast<int>(heap.size()) < k) {
        heap.push({dist, ref});
      } else {
        const HeapNeighbor worst = heap.top();
        if (dist < worst.distance || (dist == worst.distance && ref < worst.index)) {
          heap.pop();
          heap.push({dist, ref});
        }
      }
    }
    if (static_cast<int>(heap.size()) < k) {
      Rcpp::stop("sparse KNN returned fewer neighbors than requested");
    }
    std::vector<HeapNeighbor> best;
    best.reserve(k);
    while (!heap.empty()) {
      best.push_back(heap.top());
      heap.pop();
    }
    std::sort(best.begin(), best.end(), best_less);
    for (int j = 0; j < k; ++j) {
      indices(query, j) = best[j].index + 1;
      distances(query, j) = metric == "inner_product" ?
        std::max(best[j].distance - best.front().distance, 0.0) :
        best[j].distance;
    }
  }

  return List::create(
    Rcpp::Named("indices") = indices,
    Rcpp::Named("distances") = distances,
    Rcpp::Named("sparse") = true
  );
}
