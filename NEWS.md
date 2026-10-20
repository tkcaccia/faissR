# faissR 0.1

## Initial CRAN submission

* Declare the roxygen2 generator and normalize source and vignette formatting.
* Provide FAISS-backed CPU nearest-neighbour search with optional CUDA and
  cuVS support, capability inspection, and explicit unsupported-route errors.
* Support Euclidean, cosine, and correlation search, float32 input,
  candidate ranking, reusable kNN prediction models, and k-means helpers.
* Provide GPU-resident nearest-neighbour results and a native C-callable
  interface for downstream packages.
* Add version-1 C callables for building an owned CPU FAISS HNSW index once
  and querying it repeatedly with explicit construction, search, and thread
  settings.
* Include reference documentation, installation guidance, and base R dataset
  examples in the vignette and help pages.
* Reject sparse, delayed, file-backed, and other matrix-like inputs before
  conversion, and demonstrate nearest-neighbor search on dense base R data.
* Reconcile the JSS manuscript and replication material with the completed
  comparison, tuned-HNSW, and query-workload audits.
