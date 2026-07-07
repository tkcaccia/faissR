# faissR 0.99.3

* Initial Bioconductor development release.
* Provides FAISS-backed nearest-neighbour search, graph construction,
  graph clustering, k-nearest-neighbour prediction, and k-means helpers.
* Adds optional CUDA, FAISS GPU, RAPIDS cuVS, and RAPIDS libcugraph routes
  where the corresponding system libraries are available.
* Supports shape-aware automatic tuning policies for Euclidean, cosine,
  correlation, and inner-product nearest-neighbour searches.
* Clarifies Bioconductor/r-universe system requirements: FAISS is the
  mandatory compiled dependency for all builds, while CUDA/RAPIDS libraries
  are optional and requested only for explicit GPU builds.
* Adds a repository-only r-universe `.prepare` hook to install the mandatory
  Debian/Ubuntu FAISS development package while the upstream sysreq database
  learns the FAISS rule, and detects `/usr` multiarch FAISS installs during
  configuration.
