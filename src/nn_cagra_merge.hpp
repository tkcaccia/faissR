#ifndef FAISSR_NN_CAGRA_MERGE_HPP
#define FAISSR_NN_CAGRA_MERGE_HPP

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <utility>
#include <vector>

namespace faissr {

struct CagraBatch {
  std::vector<std::int64_t> ids;
  std::vector<float> distances;
};

struct CagraShard {
  int first;
  int rows;
};

// The shards cover disjoint global IDs. Shift local IDs before ranking.
template <typename Write>
inline void merge_cagra_candidates(const std::vector<CagraBatch>& batches,
    const std::vector<CagraShard>& shards, int first, int rows, int k,
    int stride, int output_first, Write write, int input_first = 0) {
  const int width = k + 1;
  using Candidate = std::pair<float, int>;
  std::vector<Candidate> candidates;
  candidates.reserve(batches.size() * width);
  std::vector<std::size_t> next(batches.size()), end(batches.size());
  std::vector<int> selected;
  selected.reserve(k);
  const auto none = batches.size();
  const bool use_tree = none >= 8;
  std::size_t base = 1;
  while (use_tree && base < none) base *= 2;
  std::vector<std::size_t> winners(use_tree ? 2 * base : 0, none);
  const auto earlier = [&](std::size_t a, std::size_t b) {
    if (a == none) return b;
    if (b == none) return a;
    return candidates[next[a]] < candidates[next[b]] ? a : b;
  };
  for (int row = 0; row < rows; ++row) {
    candidates.clear();
    selected.clear();
    const auto offset = (static_cast<std::size_t>(input_first) + row) * width;
    for (std::size_t shard = 0; shard < batches.size(); ++shard) {
      next[shard] = candidates.size();
      const auto& batch = batches[shard];
      const auto& range = shards[shard];
      for (int rank = 0; rank < width; ++rank) {
        const auto id = batch.ids[offset + rank];
        const auto value = batch.distances[offset + rank];
        if (id == -1) continue;
        if (id < 0 || id >= range.rows || !std::isfinite(value)) {
          throw std::runtime_error("Invalid CUDA CAGRA shard candidate");
        }
        const int global = range.first + static_cast<int>(id);
        if (global != first + row) {
          candidates.emplace_back(std::max(0.0f, value), global);
        }
      }
      end[shard] = candidates.size();
      const auto begin = candidates.begin() + next[shard];
      if (!std::is_sorted(begin, candidates.end())) {
        std::sort(begin, candidates.end());
      }
    }

    // For many shards, refresh only the winners along the path from
    // the shard just consumed instead of scanning every shard head.
    if (use_tree) {
      for (std::size_t rank = 0; rank < none; ++rank)
        winners[base + rank] = next[rank] < end[rank] ? rank : none;
      for (std::size_t node = base - 1; node > 0; --node)
        winners[node] = earlier(winners[2 * node], winners[2 * node + 1]);
    }

    // Each run is sorted, so only the next candidate per shard is needed.
    while (static_cast<int>(selected.size()) < k) {
      std::size_t best = none;
      if (use_tree) {
        best = winners[1];
      } else {
        for (std::size_t shard = 0; shard < none; ++shard) {
          if (next[shard] < end[shard] &&
              (best == none || candidates[next[shard]] <
                candidates[next[best]])) {
            best = shard;
          }
        }
      }
      if (best == none) break;
      const auto& item = candidates[next[best]++];
      if (use_tree) {
        std::size_t node = base + best;
        winners[node] = next[best] < end[best] ? best : none;
        for (node /= 2; node > 0; node /= 2)
          winners[node] = earlier(winners[2 * node], winners[2 * node + 1]);
      }
      if (std::find(selected.begin(), selected.end(), item.second) !=
          selected.end()) continue;
      const auto output = selected.size() * stride + output_first + row;
      write(output, item.second + 1, item.first);
      selected.push_back(item.second);
    }
    if (static_cast<int>(selected.size()) != k) {
      throw std::runtime_error("CUDA CAGRA shards returned too few neighbors");
    }
  }
}

}  // namespace faissr

#endif
