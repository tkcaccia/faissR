# faissR 0.1

* With cuVS, automatic CUDA Euclidean self-KNN selects CAGRA with NN-descent
  for at least one million rows, 65 to 1,024 features, and 9 to 128 neighbors.
  The downstream file-backed planner can fit smaller CAGRA indexes in
  sequence when a single index exceeds the memory budget.

* Batch persistent device-output CUDA searches and report the batch size
  and count. When CUDA runs out of temporary FAISS search memory, retry
  the same query rows in smaller batches.
* Allow private HNSW audit callers to retain an unmet pilot recall using
  `options(faissR.knn.report_recall_miss = TRUE)`. Return the measured
  recall and target status. The default continues to enforce the guard.

* Add private C-callable capacity planning and one-pass float32 file-backed
  Metal IVF search. The route reuses the native Metal index for all query
  rows, checks host and device budgets, and rejects missed recall pilots.
  Bounded exact file-backed Metal batches remain available.

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
