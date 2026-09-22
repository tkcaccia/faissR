# faissR 0.99.48

## Initial Bioconductor submission

* Declare the roxygen2 generator and normalize vignette continuation
    indentation for BiocCheck source-format validation.
* Provide FAISS-backed CPU nearest-neighbour search with optional CUDA and
  cuVS support, capability inspection, and explicit unsupported-route errors.
* Support Euclidean, cosine, and correlation search, optional float32 input,
  candidate ranking, reusable kNN prediction models, and k-means helpers.
* Provide GPU-resident nearest-neighbour results and a native C-callable
  interface for downstream packages.
* Include reference documentation, installation guidance, and Biobase dataset
  examples in the vignette and help pages.
* Reject sparse, delayed, file-backed, and other matrix-like inputs before
  conversion, and demonstrate nearest-neighbor search on a documented dense
  single-cell PCA representation.
* Reconcile the JSS manuscript and replication material with the completed
  comparison, tuned-HNSW, and query-workload audits.
