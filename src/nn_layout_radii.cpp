#include <Rcpp.h>

#include <algorithm>
#include <cmath>
#include <numeric>
#include <queue>
#include <vector>

namespace {

class LayoutRadii {
  const Rcpp::NumericMatrix& xy;
  std::vector<int> ids;

  void build(int begin, int end, int axis) {
    if (begin == end) return;
    const int mid = begin + (end - begin) / 2;
    std::nth_element(ids.begin() + begin, ids.begin() + mid,
      ids.begin() + end, [&](int a, int b) {
        return xy(a, axis) < xy(b, axis) ||
          (xy(a, axis) == xy(b, axis) && a < b);
      });
    build(begin, mid, 1 - axis);
    build(mid + 1, end, 1 - axis);
  }

  void search(int u, int begin, int end, int axis, int k,
      std::priority_queue<double>& nearest) const {
    if (begin == end) return;
    const int mid = begin + (end - begin) / 2;
    const int v = ids[mid];
    const double dx = xy(u, 0) - xy(v, 0);
    const double dy = xy(u, 1) - xy(v, 1);
    const double distance = dx * dx + dy * dy;
    if (v != u && (static_cast<int>(nearest.size()) < k ||
        distance < nearest.top())) {
      nearest.push(distance);
      if (static_cast<int>(nearest.size()) > k) nearest.pop();
    }
    const double delta = xy(u, axis) - xy(v, axis);
    if (delta < 0) search(u, begin, mid, 1 - axis, k, nearest);
    else search(u, mid + 1, end, 1 - axis, k, nearest);
    if (static_cast<int>(nearest.size()) < k ||
        delta * delta <= nearest.top()) {
      if (delta < 0) search(u, mid + 1, end, 1 - axis, k, nearest);
      else search(u, begin, mid, 1 - axis, k, nearest);
    }
  }

 public:
  explicit LayoutRadii(const Rcpp::NumericMatrix& layout) : xy(layout),
      ids(static_cast<std::size_t>(layout.nrow())) {
    std::iota(ids.begin(), ids.end(), 0);
    build(0, xy.nrow(), 0);
  }

  Rcpp::NumericVector radii(int k) const {
    Rcpp::NumericVector radius(xy.nrow());
    for (int u = 0; u < xy.nrow(); ++u) {
      if (u % 1024 == 0) Rcpp::checkUserInterrupt();
      std::priority_queue<double> nearest;
      search(u, 0, xy.nrow(), 0, k, nearest);
      radius[u] = std::sqrt(nearest.top());
      if (!(radius[u] > 0) || !std::isfinite(radius[u])) {
        Rcpp::stop("Local projection radii must be positive and finite.");
      }
    }
    return radius;
  }
};

}  // namespace

extern "C" SEXP faissR_layout_radii_v1(SEXP layout, SEXP neighbors) {
  Rcpp::NumericMatrix xy(layout);
  const int k = Rcpp::as<int>(neighbors);
  if (xy.ncol() != 2 || k < 1 || k >= xy.nrow()) {
    Rcpp::stop("Layout radii require an n-by-2 matrix and 1 <= k < n.");
  }
  for (double value : xy) {
    if (!std::isfinite(value)) Rcpp::stop("Layout must be finite.");
  }
  return LayoutRadii(xy).radii(k);
}
