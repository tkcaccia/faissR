#ifndef FAISSR_NN_CAGRA_CAPACITY_HPP
#define FAISSR_NN_CAGRA_CAPACITY_HPP

#include <algorithm>

namespace faissr {

struct CagraCapacity {
  double reference_bytes;
  double device_bytes;
  double host_batch_bytes;
};

inline CagraCapacity cagra_capacity(int n, int p, int k, int batch,
                                    int graph_degree, int intermediate) {
  const double reference = 4.0 * n * p;
  const double upload = 4.0 * batch * p;
  const double output = 12.0 * batch * (k + 1.0);
  const double graph = 4.0 * n * (static_cast<double>(graph_degree) + intermediate);
  const double workspace = std::max(64.0 * 1024 * 1024, reference / 5);
  return {reference, 2 * reference + workspace + graph + output,
          upload + 4 * output};
}

}  // namespace faissr

#endif
