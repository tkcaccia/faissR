# faissR 0.99.12

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
* Allows ordinary macOS GitHub Actions builders to install the mandatory
  Homebrew `faiss` and `libomp` dependencies during `configure`, while keeping
  ordinary interactive installs explicit or opt-in.
* Handles r-universe/WebAssembly cross-builds without leaking host Linux
  headers into the Emscripten sysroot. WebAssembly builds install diagnostic
  stubs because native FAISS/CUDA/cuVS libraries are not available in webR;
  supported native Linux/macOS builds still require FAISS.
* Detects already-active conda/mamba environments through `CONDA_PREFIX` as a
  passive macOS FAISS/libomp fallback without installing conda from
  `configure`.
* Marks automated Bioconductor macOS binary builds as unsupported for real
  FAISS execution until the Bioconductor/r-universe macOS system-library bundle
  provides FAISS. Because r-universe currently still launches the macOS binary
  job, that exact worker receives diagnostic stubs when FAISS is absent; user
  macOS source installs remain supported with Homebrew or an active conda/mamba
  environment and still require real FAISS.
* Keeps FAISS k-means compilation compatible with distro FAISS headers that do
  not expose newer clustering fields, and links macOS OpenMP through the exact
  detected `libomp` library path to avoid duplicate OpenMP runtimes when
  Homebrew FAISS is loaded.
