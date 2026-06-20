#include <Rcpp.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

using Rcpp::IntegerMatrix;
using Rcpp::List;
using Rcpp::NumericMatrix;
using Rcpp::NumericVector;
using Rcpp::IntegerVector;

namespace {

std::uint64_t edge_key(const int a, const int b) {
  const std::uint32_t u = static_cast<std::uint32_t>(std::min(a, b));
  const std::uint32_t v = static_cast<std::uint32_t>(std::max(a, b));
  return (static_cast<std::uint64_t>(u) << 32) | static_cast<std::uint64_t>(v);
}

int edge_from_key(const std::uint64_t key) {
  return static_cast<int>(static_cast<std::uint32_t>(key >> 32));
}

int edge_to_key(const std::uint64_t key) {
  return static_cast<int>(static_cast<std::uint32_t>(key & 0xffffffffULL));
}

struct Edge {
  std::uint64_t key;
  double weight;
};

void push_edge(std::vector<Edge>& edges,
               const int i,
               const int j,
               const double weight) {
  if (i == j || !std::isfinite(weight) || weight <= 0.0) return;
  edges.push_back(Edge{edge_key(i, j), weight});
}

bool contains_neighbor(const IntegerMatrix& indices,
                       const int row,
                       const int target) {
  const int k = indices.ncol();
  for (int col = 0; col < k; ++col) {
    if (indices(row, col) == target) return true;
  }
  return false;
}

std::vector<Edge> build_full_snn_edges(const IntegerMatrix& indices,
                                       const double prune) {
  // Inspired by bluster::neighborsToSNNGraph() / scran_graph_cluster:
  // build an inverted neighbour index and count shared-neighbour
  // co-occurrences directly. This creates the standard full SNN graph
  // between all observations sharing at least one neighbour, not only
  // edges already present in the directed KNN graph.
  const int n = indices.nrow();
  const int k = indices.ncol();

  std::vector<int> valid_count(static_cast<std::size_t>(n), 0);
  std::vector<int> reverse_count(static_cast<std::size_t>(n) + 1U, 0);
  for (int row = 0; row < n; ++row) {
    const int self = row + 1;
    for (int col = 0; col < k; ++col) {
      const int idx = indices(row, col);
      if (idx >= 1 && idx <= n && idx != self) {
        ++valid_count[static_cast<std::size_t>(row)];
        ++reverse_count[static_cast<std::size_t>(idx)];
      }
    }
  }

  std::vector<int> reverse_ptr(static_cast<std::size_t>(n) + 2U, 0);
  for (int i = 1; i <= n; ++i) {
    reverse_ptr[static_cast<std::size_t>(i + 1)] =
      reverse_ptr[static_cast<std::size_t>(i)] +
      reverse_count[static_cast<std::size_t>(i)];
  }
  std::vector<int> reverse_rows(static_cast<std::size_t>(reverse_ptr[static_cast<std::size_t>(n + 1)]));
  std::fill(reverse_count.begin(), reverse_count.end(), 0);
  for (int row = 0; row < n; ++row) {
    const int self = row + 1;
    for (int col = 0; col < k; ++col) {
      const int idx = indices(row, col);
      if (idx >= 1 && idx <= n && idx != self) {
        const std::size_t offset =
          static_cast<std::size_t>(reverse_ptr[static_cast<std::size_t>(idx)] +
                                   reverse_count[static_cast<std::size_t>(idx)]++);
        reverse_rows[offset] = row;
      }
    }
  }

  std::vector<Edge> edges;
  edges.reserve(static_cast<std::size_t>(n) * static_cast<std::size_t>(k));
  std::vector<int> shared_counts(static_cast<std::size_t>(n), 0);
  std::vector<int> touched;
  touched.reserve(static_cast<std::size_t>(k) * static_cast<std::size_t>(k));

  for (int row = 0; row < n; ++row) {
    touched.clear();
    const int self = row + 1;
    for (int col = 0; col < k; ++col) {
      const int idx = indices(row, col);
      if (idx < 1 || idx > n || idx == self) continue;
      const int begin = reverse_ptr[static_cast<std::size_t>(idx)];
      const int end = reverse_ptr[static_cast<std::size_t>(idx + 1)];
      for (int pos = begin; pos < end; ++pos) {
        const int other = reverse_rows[static_cast<std::size_t>(pos)];
        if (other <= row) continue;
        int& counter = shared_counts[static_cast<std::size_t>(other)];
        if (counter == 0) touched.push_back(other);
        ++counter;
      }
    }

    for (const int other : touched) {
      const int shared = shared_counts[static_cast<std::size_t>(other)];
      shared_counts[static_cast<std::size_t>(other)] = 0;
      const int denom_int = valid_count[static_cast<std::size_t>(row)] +
        valid_count[static_cast<std::size_t>(other)] - shared;
      if (shared > 0 && denom_int > 0) {
        const double weight =
          static_cast<double>(shared) / static_cast<double>(denom_int);
        if (weight > prune) push_edge(edges, row + 1, other + 1, weight);
      }
    }
  }

  return edges;
}

std::vector<double> local_sigmas(const NumericMatrix& distances) {
  const int n = distances.nrow();
  const int k = distances.ncol();
  std::vector<double> sigma(static_cast<std::size_t>(n), 1.0);
  for (int row = 0; row < n; ++row) {
    double last = 0.0;
    double sum = 0.0;
    int count = 0;
    for (int col = 0; col < k; ++col) {
      const double d = distances(row, col);
      if (std::isfinite(d) && d > 0.0) {
        last = d;
        sum += d;
        ++count;
      }
    }
    if (last > 0.0) {
      sigma[static_cast<std::size_t>(row)] = last;
    } else if (count > 0 && sum > 0.0) {
      sigma[static_cast<std::size_t>(row)] = sum / static_cast<double>(count);
    }
  }
  return sigma;
}

} // namespace

// [[Rcpp::export]]
List knn_graph_edges_cpp(IntegerMatrix indices,
                         NumericMatrix distances,
                         std::string weight_type,
                         double prune,
                         bool mutual) {
  const int n = indices.nrow();
  const int k = indices.ncol();
  if (distances.nrow() != n || distances.ncol() != k) {
    Rcpp::stop("KNN indices and distances must have the same dimensions");
  }
  if (n < 2 || k < 1) {
    Rcpp::stop("KNN input must have at least two rows and one neighbor column");
  }
  if (weight_type != "snn" &&
      weight_type != "distance" &&
      weight_type != "adaptive" &&
      weight_type != "binary") {
    Rcpp::stop("unsupported graph weight type");
  }
  if (!std::isfinite(prune) || prune < 0.0) prune = 0.0;

  std::vector<Edge> edges;

  if (weight_type == "snn" && !mutual) {
    edges = build_full_snn_edges(indices, prune);
  } else if (weight_type == "snn") {
    edges.reserve(static_cast<std::size_t>(n) * static_cast<std::size_t>(k));
    std::vector<int> mark(static_cast<std::size_t>(n) + 1U, 0);
    int stamp = 1;
    for (int row = 0; row < n; ++row) {
      if (stamp == std::numeric_limits<int>::max()) {
        std::fill(mark.begin(), mark.end(), 0);
        stamp = 1;
      }
      const int i = row + 1;
      for (int col = 0; col < k; ++col) {
        const int idx = indices(row, col);
        if (idx >= 1 && idx <= n && idx != i) {
          mark[static_cast<std::size_t>(idx)] = stamp;
        }
      }

      for (int col = 0; col < k; ++col) {
        const int j = indices(row, col);
        if (j < 1 || j > n || j == i) continue;
        if (mutual && !contains_neighbor(indices, j - 1, i)) continue;
        int shared = 0;
        const int jrow = j - 1;
        for (int jcol = 0; jcol < k; ++jcol) {
          const int jj = indices(jrow, jcol);
          if (jj >= 1 && jj <= n && mark[static_cast<std::size_t>(jj)] == stamp) {
            ++shared;
          }
        }
        if (shared <= 0) continue;
        const double denom = static_cast<double>(2 * k - shared);
        const double weight = denom > 0.0 ? static_cast<double>(shared) / denom : 0.0;
        if (weight > prune) push_edge(edges, i, j, weight);
      }
      ++stamp;
    }
  } else {
    edges.reserve(static_cast<std::size_t>(n) * static_cast<std::size_t>(k));
    std::vector<double> sigma;
    if (weight_type == "adaptive") {
      sigma = local_sigmas(distances);
    }
    for (int row = 0; row < n; ++row) {
      const int i = row + 1;
      for (int col = 0; col < k; ++col) {
        const int j = indices(row, col);
        if (j < 1 || j > n || j == i) continue;
        if (mutual && !contains_neighbor(indices, j - 1, i)) continue;
        double weight = 1.0;
        if (weight_type == "distance") {
          const double d = distances(row, col);
          if (!std::isfinite(d) || d < 0.0) continue;
          weight = 1.0 / (1.0 + d);
        } else if (weight_type == "adaptive") {
          const double d = distances(row, col);
          if (!std::isfinite(d) || d < 0.0) continue;
          const double si = sigma[static_cast<std::size_t>(row)];
          const double sj = sigma[static_cast<std::size_t>(j - 1)];
          const double scale = std::max(si * sj, 1e-12);
          weight = std::exp(-(d * d) / scale);
        }
        if (weight > prune) push_edge(edges, i, j, weight);
      }
    }
  }

  std::sort(edges.begin(), edges.end(), [](const Edge& a, const Edge& b) {
    return a.key < b.key;
  });
  std::size_t unique_count = 0;
  for (std::size_t pos = 0; pos < edges.size();) {
    const std::uint64_t key = edges[pos].key;
    double max_weight = edges[pos].weight;
    ++pos;
    while (pos < edges.size() && edges[pos].key == key) {
      if (edges[pos].weight > max_weight) max_weight = edges[pos].weight;
      ++pos;
    }
    edges[unique_count++] = Edge{key, max_weight};
  }

  const int m = static_cast<int>(unique_count);
  IntegerVector from(m);
  IntegerVector to(m);
  NumericVector weight(m);
  for (int pos = 0; pos < m; ++pos) {
    from[pos] = edge_from_key(edges[static_cast<std::size_t>(pos)].key);
    to[pos] = edge_to_key(edges[static_cast<std::size_t>(pos)].key);
    weight[pos] = edges[static_cast<std::size_t>(pos)].weight;
  }

  return List::create(
    Rcpp::Named("from") = from,
    Rcpp::Named("to") = to,
    Rcpp::Named("weight") = weight,
    Rcpp::Named("n_vertices") = n,
    Rcpp::Named("n_edges") = m,
    Rcpp::Named("weight_type") = weight_type,
    Rcpp::Named("prune") = prune,
    Rcpp::Named("mutual") = mutual
  );
}

#include <deque>
#include <numeric>
#include <queue>
#include <unordered_map>
#ifdef _OPENMP
#include <omp.h>
#endif

namespace {

struct CsrGraph {
  int n = 0;
  std::vector<int> ptr;
  std::vector<int> to;
  std::vector<double> weight;
  std::vector<double> degree;
  double total_edge_weight = 0.0;
};

CsrGraph csr_from_edge_list(const List& edge_list) {
  IntegerVector from = edge_list["from"];
  IntegerVector to = edge_list["to"];
  NumericVector weight = edge_list["weight"];
  const int n = Rcpp::as<int>(edge_list["n_vertices"]);
  if (from.size() != to.size() || from.size() != weight.size()) {
    Rcpp::stop("invalid graph edge list");
  }

  CsrGraph g;
  g.n = n;
  g.ptr.assign(static_cast<std::size_t>(n) + 1U, 0);
  g.degree.assign(static_cast<std::size_t>(n), 0.0);

  const int m = from.size();
  for (int e = 0; e < m; ++e) {
    const int u = from[e] - 1;
    const int v = to[e] - 1;
    const double w = weight[e];
    if (u < 0 || u >= n || v < 0 || v >= n || u == v || !std::isfinite(w) || w <= 0.0) continue;
    ++g.ptr[static_cast<std::size_t>(u) + 1U];
    ++g.ptr[static_cast<std::size_t>(v) + 1U];
    g.degree[static_cast<std::size_t>(u)] += w;
    g.degree[static_cast<std::size_t>(v)] += w;
    g.total_edge_weight += w;
  }
  for (int i = 1; i <= n; ++i) {
    g.ptr[static_cast<std::size_t>(i)] += g.ptr[static_cast<std::size_t>(i - 1)];
  }
  g.to.assign(static_cast<std::size_t>(g.ptr[static_cast<std::size_t>(n)]), 0);
  g.weight.assign(g.to.size(), 0.0);
  std::vector<int> fill = g.ptr;
  for (int e = 0; e < m; ++e) {
    const int u = from[e] - 1;
    const int v = to[e] - 1;
    const double w = weight[e];
    if (u < 0 || u >= n || v < 0 || v >= n || u == v || !std::isfinite(w) || w <= 0.0) continue;
    int pos = fill[static_cast<std::size_t>(u)]++;
    g.to[static_cast<std::size_t>(pos)] = v;
    g.weight[static_cast<std::size_t>(pos)] = w;
    pos = fill[static_cast<std::size_t>(v)]++;
    g.to[static_cast<std::size_t>(pos)] = u;
    g.weight[static_cast<std::size_t>(pos)] = w;
  }
  return g;
}

std::vector<double> community_degree(const CsrGraph& g, const std::vector<int>& membership) {
  int max_comm = 0;
  for (const int c : membership) if (c > max_comm) max_comm = c;
  std::vector<double> deg(static_cast<std::size_t>(max_comm) + 1U, 0.0);
  for (int i = 0; i < g.n; ++i) {
    deg[static_cast<std::size_t>(membership[static_cast<std::size_t>(i)])] += g.degree[static_cast<std::size_t>(i)];
  }
  return deg;
}

std::vector<int> compact_membership(const std::vector<int>& membership) {
  std::unordered_map<int, int> map;
  map.reserve(membership.size());
  std::vector<int> out(membership.size());
  int next = 0;
  for (std::size_t i = 0; i < membership.size(); ++i) {
    const int c = membership[i];
    auto it = map.find(c);
    if (it == map.end()) {
      it = map.emplace(c, next++).first;
    }
    out[i] = it->second;
  }
  return out;
}

double modularity_score_native(const CsrGraph& g, const std::vector<int>& membership, double resolution) {
  if (g.total_edge_weight <= 0.0 || g.n == 0) return NA_REAL;
  const double two_m = 2.0 * g.total_edge_weight;
  double internal_twice = 0.0;
  for (int u = 0; u < g.n; ++u) {
    const int cu = membership[static_cast<std::size_t>(u)];
    for (int p = g.ptr[static_cast<std::size_t>(u)]; p < g.ptr[static_cast<std::size_t>(u + 1)]; ++p) {
      const int v = g.to[static_cast<std::size_t>(p)];
      if (cu == membership[static_cast<std::size_t>(v)]) {
        internal_twice += g.weight[static_cast<std::size_t>(p)];
      }
    }
  }
  const std::vector<double> cdeg = community_degree(g, membership);
  double expected = 0.0;
  for (const double d : cdeg) expected += d * d;
  return internal_twice / two_m - resolution * expected / (two_m * two_m);
}

std::vector<int> native_louvain_local_moving(const CsrGraph& g,
                                             int max_iter,
                                             double resolution,
                                             int n_threads,
                                             int seed) {
  std::vector<int> membership(static_cast<std::size_t>(g.n));
  std::iota(membership.begin(), membership.end(), 0);
  if (g.total_edge_weight <= 0.0) return membership;
  const double two_m = 2.0 * g.total_edge_weight;
  max_iter = std::max(1, max_iter);
  n_threads = std::max(1, n_threads);

  for (int iter = 0; iter < max_iter; ++iter) {
    const std::vector<double> cdeg = community_degree(g, membership);
    std::vector<int> proposed = membership;
    int changed = 0;

#ifdef _OPENMP
#pragma omp parallel for num_threads(n_threads) schedule(dynamic, 256) reduction(+:changed)
#endif
    for (int u = 0; u < g.n; ++u) {
      std::unordered_map<int, double> neigh;
      neigh.reserve(static_cast<std::size_t>(g.ptr[static_cast<std::size_t>(u + 1)] - g.ptr[static_cast<std::size_t>(u)] + 1));
      const int old_comm = membership[static_cast<std::size_t>(u)];
      for (int p = g.ptr[static_cast<std::size_t>(u)]; p < g.ptr[static_cast<std::size_t>(u + 1)]; ++p) {
        const int v = g.to[static_cast<std::size_t>(p)];
        const int cv = membership[static_cast<std::size_t>(v)];
        neigh[cv] += g.weight[static_cast<std::size_t>(p)];
      }
      int best_comm = old_comm;
      double best_gain = 0.0;
      const double k_i = g.degree[static_cast<std::size_t>(u)];
      for (const auto& kv : neigh) {
        const int cand = kv.first;
        double cand_deg = cdeg[static_cast<std::size_t>(cand)];
        if (cand == old_comm) cand_deg -= k_i;
        const double gain = kv.second - resolution * k_i * cand_deg / two_m;
        double old_weight = 0.0;
        const auto old_it = neigh.find(old_comm);
        if (old_it != neigh.end()) old_weight = old_it->second;
        const double stay_gain = old_weight - resolution * k_i * std::max(0.0, cdeg[static_cast<std::size_t>(old_comm)] - k_i) / two_m;
        const double delta = gain - stay_gain;
        if (delta > best_gain + 1e-12 ||
            (std::abs(delta - best_gain) <= 1e-12 && cand < best_comm && delta > 0.0)) {
          best_gain = delta;
          best_comm = cand;
        }
      }
      proposed[static_cast<std::size_t>(u)] = best_comm;
      if (best_comm != old_comm) ++changed;
    }

    membership.swap(proposed);
    membership = compact_membership(membership);
    if (changed == 0) break;
    if ((iter + seed) % 4 == 3) {
      // A deterministic small perturbation helps avoid two-community oscillation
      // in synchronous local moving while preserving reproducibility.
      membership = compact_membership(membership);
    }
  }
  return compact_membership(membership);
}

std::vector<int> split_disconnected_communities(const CsrGraph& g, const std::vector<int>& membership) {
  std::vector<int> refined(static_cast<std::size_t>(g.n), -1);
  std::vector<char> seen(static_cast<std::size_t>(g.n), 0);
  std::queue<int> q;
  int next_comm = 0;
  for (int start = 0; start < g.n; ++start) {
    if (seen[static_cast<std::size_t>(start)]) continue;
    const int comm = membership[static_cast<std::size_t>(start)];
    seen[static_cast<std::size_t>(start)] = 1;
    refined[static_cast<std::size_t>(start)] = next_comm;
    q.push(start);
    while (!q.empty()) {
      const int u = q.front();
      q.pop();
      for (int p = g.ptr[static_cast<std::size_t>(u)]; p < g.ptr[static_cast<std::size_t>(u + 1)]; ++p) {
        const int v = g.to[static_cast<std::size_t>(p)];
        if (!seen[static_cast<std::size_t>(v)] && membership[static_cast<std::size_t>(v)] == comm) {
          seen[static_cast<std::size_t>(v)] = 1;
          refined[static_cast<std::size_t>(v)] = next_comm;
          q.push(v);
        }
      }
    }
    ++next_comm;
  }
  return compact_membership(refined);
}

std::vector<int> native_leiden_refined(const CsrGraph& g,
                                       int n_iterations,
                                       double resolution,
                                       int n_threads,
                                       int seed) {
  // This native path follows the practical Leiden idea that communities should
  // be internally connected. It starts from modularity local moving and then
  // refines the partition by splitting disconnected pieces inside each
  // community. A full Leiden aggregation hierarchy can be added later without
  // changing the R API.
  std::vector<int> membership = native_louvain_local_moving(
    g, std::max(2, n_iterations), resolution, n_threads, seed);
  return compact_membership(split_disconnected_communities(g, membership));
}

std::vector<int> native_random_walk_cluster(const CsrGraph& g,
                                            int steps,
                                            int max_iter,
                                            int n_threads) {
  std::vector<int> labels(static_cast<std::size_t>(g.n));
  std::iota(labels.begin(), labels.end(), 0);
  steps = std::max(1, steps);
  max_iter = std::max(1, max_iter);
  n_threads = std::max(1, n_threads);
  if (g.total_edge_weight <= 0.0) return labels;

  for (int iter = 0; iter < max_iter; ++iter) {
    std::vector<int> proposed = labels;
    int changed = 0;
#ifdef _OPENMP
#pragma omp parallel for num_threads(n_threads) schedule(dynamic, 256) reduction(+:changed)
#endif
    for (int u = 0; u < g.n; ++u) {
      std::unordered_map<int, double> score;
      score.reserve(32);
      std::unordered_map<int, double> frontier;
      frontier.reserve(32);
      frontier[u] = 1.0;
      for (int step = 0; step < steps; ++step) {
        std::unordered_map<int, double> next;
        next.reserve(frontier.size() * 2U + 1U);
        for (const auto& item : frontier) {
          const int x = item.first;
          const double mass = item.second;
          const double deg = g.degree[static_cast<std::size_t>(x)];
          if (deg <= 0.0) continue;
          for (int p = g.ptr[static_cast<std::size_t>(x)]; p < g.ptr[static_cast<std::size_t>(x + 1)]; ++p) {
            const int v = g.to[static_cast<std::size_t>(p)];
            next[v] += mass * g.weight[static_cast<std::size_t>(p)] / deg;
          }
        }
        frontier.swap(next);
        for (const auto& item : frontier) {
          score[labels[static_cast<std::size_t>(item.first)]] += item.second;
        }
      }
      int best = labels[static_cast<std::size_t>(u)];
      double best_score = -1.0;
      for (const auto& kv : score) {
        if (kv.second > best_score + 1e-15 ||
            (std::abs(kv.second - best_score) <= 1e-15 && kv.first < best)) {
          best_score = kv.second;
          best = kv.first;
        }
      }
      proposed[static_cast<std::size_t>(u)] = best;
      if (best != labels[static_cast<std::size_t>(u)]) ++changed;
    }
    labels.swap(proposed);
    labels = compact_membership(labels);
    if (changed == 0) break;
  }
  return compact_membership(split_disconnected_communities(g, labels));
}

List make_cluster_result(const CsrGraph& g,
                         const std::vector<int>& membership0,
                         const std::string& method,
                         const std::string& backend,
                         double resolution,
                         int n_threads,
                         List edge_list) {
  IntegerVector membership(g.n);
  for (int i = 0; i < g.n; ++i) membership[i] = membership0[static_cast<std::size_t>(i)] + 1;
  const double modularity = modularity_score_native(g, membership0, resolution);
  int n_communities = 0;
  for (const int c : membership0) if (c + 1 > n_communities) n_communities = c + 1;
  return List::create(
    Rcpp::Named("membership") = membership,
    Rcpp::Named("modularity") = modularity,
    Rcpp::Named("n_communities") = n_communities,
    Rcpp::Named("method") = method,
    Rcpp::Named("backend") = backend,
    Rcpp::Named("n_threads") = n_threads,
    Rcpp::Named("graph") = edge_list
  );
}

} // namespace

// [[Rcpp::export]]
List graph_cluster_cpp(IntegerMatrix indices,
                       NumericMatrix distances,
                       std::string method,
                       std::string backend,
                       std::string weight_type,
                       double prune,
                       bool mutual,
                       int n_threads,
                       int n_runs,
                       double resolution,
                       int n_iterations,
                       int steps,
                       int seed) {
  if (backend == "cuda") {
    Rcpp::stop("Native CUDA graph clustering requires RAPIDS libcugraph headers and library at build time. This faissR build was not linked to libcugraph; no Python/cuGraph bridge is used.");
  }
  if (backend != "cpu") Rcpp::stop("unsupported graph clustering backend");
  if (method != "louvain" && method != "leiden" && method != "random_walking") {
    Rcpp::stop("unsupported graph clustering method");
  }
  n_threads = std::max(1, n_threads);
  n_runs = std::max(1, n_runs);
  n_iterations = std::max(1, n_iterations);
  steps = std::max(1, steps);
  if (!std::isfinite(resolution) || resolution <= 0.0) resolution = 1.0;

  List edge_list = knn_graph_edges_cpp(indices, distances, weight_type, prune, mutual);
  CsrGraph g = csr_from_edge_list(edge_list);
  if (g.n == 0) Rcpp::stop("empty graph");

  std::vector<int> best;
  double best_mod = -std::numeric_limits<double>::infinity();
  NumericVector all_modularity(n_runs);
  IntegerVector selected_threads(n_runs);

  for (int run = 0; run < n_runs; ++run) {
    std::vector<int> membership;
    const int run_seed = seed + run * 104729;
    if (method == "louvain") {
      membership = native_louvain_local_moving(g, n_iterations, resolution, n_threads, run_seed);
    } else if (method == "leiden") {
      membership = native_leiden_refined(g, n_iterations, resolution, n_threads, run_seed);
    } else {
      membership = native_random_walk_cluster(g, steps, n_iterations, n_threads);
    }
    const double mod = modularity_score_native(g, membership, resolution);
    all_modularity[run] = mod;
    selected_threads[run] = n_threads;
    if (best.empty() || (std::isfinite(mod) && mod > best_mod)) {
      best = membership;
      best_mod = mod;
    }
  }

  List out = make_cluster_result(g, best, method, backend, resolution, n_threads, edge_list);
  out["n_runs"] = n_runs;
  out["selected_run"] = static_cast<int>(std::distance(all_modularity.begin(), std::max_element(all_modularity.begin(), all_modularity.end()))) + 1;
  out["all_modularity"] = all_modularity;
  out["implementation"] = "native_cpp";
  return out;
}

// [[Rcpp::export]]
List graph_cluster_edges_cpp(List edge_list,
                             std::string method,
                             std::string backend,
                             int n_threads,
                             int n_runs,
                             double resolution,
                             int n_iterations,
                             int steps,
                             int seed) {
  if (backend == "cuda") {
    Rcpp::stop("Native CUDA graph clustering requires RAPIDS libcugraph headers and library at build time. This faissR build was not linked to libcugraph; no Python/cuGraph bridge is used.");
  }
  if (backend != "cpu") Rcpp::stop("unsupported graph clustering backend");
  if (method != "louvain" && method != "leiden" && method != "random_walking") {
    Rcpp::stop("unsupported graph clustering method");
  }
  n_threads = std::max(1, n_threads);
  n_runs = std::max(1, n_runs);
  n_iterations = std::max(1, n_iterations);
  steps = std::max(1, steps);
  if (!std::isfinite(resolution) || resolution <= 0.0) resolution = 1.0;

  CsrGraph g = csr_from_edge_list(edge_list);
  if (g.n == 0) Rcpp::stop("empty graph");
  std::vector<int> best;
  double best_mod = -std::numeric_limits<double>::infinity();
  NumericVector all_modularity(n_runs);
  for (int run = 0; run < n_runs; ++run) {
    const int run_seed = seed + run * 104729;
    std::vector<int> membership;
    if (method == "louvain") {
      membership = native_louvain_local_moving(g, n_iterations, resolution, n_threads, run_seed);
    } else if (method == "leiden") {
      membership = native_leiden_refined(g, n_iterations, resolution, n_threads, run_seed);
    } else {
      membership = native_random_walk_cluster(g, steps, n_iterations, n_threads);
    }
    const double mod = modularity_score_native(g, membership, resolution);
    all_modularity[run] = mod;
    if (best.empty() || (std::isfinite(mod) && mod > best_mod)) {
      best = membership;
      best_mod = mod;
    }
  }
  List out = make_cluster_result(g, best, method, backend, resolution, n_threads, edge_list);
  out["n_runs"] = n_runs;
  out["selected_run"] = static_cast<int>(std::distance(all_modularity.begin(), std::max_element(all_modularity.begin(), all_modularity.end()))) + 1;
  out["all_modularity"] = all_modularity;
  out["implementation"] = "native_cpp";
  return out;
}
