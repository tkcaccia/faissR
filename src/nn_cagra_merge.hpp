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
    int stride, int output_first, Write write) {
  const int width = k + 1;
  using Candidate = std::pair<float, int>;
  std::vector<Candidate> candidates;
  candidates.reserve(batches.size() * width);
  std::vector<std::size_t> next(batches.size()), end(batches.size());
  std::vector<int> selected;
  selected.reserve(k);
  for (int row = 0; row < rows; ++row) {
    candidates.clear();
    selected.clear();
    std::uint64_t seen = 0;
    const auto offset = static_cast<std::size_t>(row) * width;
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

    // Each run is sorted, so only the next candidate per shard is needed.
    while (static_cast<int>(selected.size()) < k) {
      std::size_t best = batches.size();
      for (std::size_t shard = 0; shard < batches.size(); ++shard) {
        if (next[shard] < end[shard] &&
            (best == batches.size() ||
             candidates[next[shard]] < candidates[next[best]])) {
          best = shard;
        }
      }
      if (best == batches.size()) break;
      const auto& item = candidates[next[best]++];
      // A clear bit proves the ID has not been emitted. Resolve collisions
      // with the exact check so ties and repeated IDs keep the same behavior.
      const auto bit = std::uint64_t{1} << (item.second % 64);
      if ((seen & bit) &&
          std::find(selected.begin(), selected.end(), item.second) !=
            selected.end()) continue;
      seen |= bit;
      const auto output = selected.size() * stride + output_first + row;
      write(output, item.second + 1, item.first);
      selected.push_back(item.second);
    }
    if (static_cast<int>(selected.size()) != k) {
      throw std::runtime_error("CUDA CAGRA shards returned too few neighbors");
    }
  }
}

inline void merge_cagra_batches(const std::vector<CagraBatch>& batches,
    const std::vector<CagraShard>& shards, int first, int rows, int k,
    int* indices, double* distances) {
  merge_cagra_candidates(batches, shards, first, rows, k, rows, 0,
    [&](std::size_t slot, int id, float squared) {
      indices[slot] = id;
      distances[slot] = std::sqrt(squared);
    });
}

}  // namespace faissr

#endif
