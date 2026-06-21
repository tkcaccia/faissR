# Benchmarks

[Home](../README.md) |
[Installation](installation.md) |
[Implementation](implementation.md) |
[Examples](examples.md) |
**Benchmarks** |
[Autotuning](autotuning.md) |
[API](usage-api.md) |
[NN Methods](nn-methods.md) |
[Backends](backend-capabilities.md) |
[References](references.md)

`faissR` benchmarks should separate vector-search quality from downstream
embedding or clustering quality.

## Recommended Measurements

For every KNN method record:

- dataset name, `n`, `p`, metric, and `k`;
- backend requested and backend used;
- build time, query time, and total time;
- peak RAM and GPU memory where measurable;
- recall@k against an exact reference on a reproducible subset;
- mean distance error and neighbour-rank agreement;
- downstream sanity checks such as openTSNE/UMAP plots when KNN is used for
  embeddings.

## Fair CPU And CUDA Runs

Use fixed CPU thread counts when comparing CPU algorithms:

```r
Sys.setenv(
  OMP_NUM_THREADS = "4",
  OPENBLAS_NUM_THREADS = "4",
  MKL_NUM_THREADS = "4"
)
```

For CUDA benchmarks, report the GPU model, driver, CUDA version, FAISS build,
and cuVS version [1-3,13-15]. Explicit CUDA failures should be recorded as
failures, not silently replaced with CPU timings.

## Reuse KNN

Large benchmarks should save KNN output once:

```r
knn <- nn(x, k = 100, backend = "auto", metric = "euclidean", n_threads = 4)
saveRDS(knn, "knn_k100.rds")
```

The same object can then feed graph construction, classifier tests, embedding
pipelines, and recall diagnostics without paying the KNN cost repeatedly.

## Benchmark #1

`benchmark_scripts/benchmark1_nn_speed.R` is the broad nearest-neighbour speed
benchmark that includes faissR implementation labels, external R KNN packages,
and selected KNN consumers. It defaults to `k = 5, 10, 15, 50, 100` and the
four public metrics L2/Euclidean, cosine, correlation, and inner product. Flat
inner-product searches are reported under the same public `method = "flat"` row
rather than duplicate Flat-IP rows. Implementation-specific faissR rows,
such as FAISS GPU IVF and direct cuVS rows, are timed through faissR's internal
benchmark route so the table can distinguish FAISS GPU indexes that use NVIDIA
cuVS internally from direct RAPIDS cuVS API calls.

If a non-standard runtime library directory is needed, set `FAISSR_ENV_DIR`
explicitly before launch. The script no longer treats an unrelated active
`CONDA_PREFIX` as a FAISS runtime, which avoids accidental library-path
pollution on local machines. CPU worker threads are controlled with environment
variables such as `OMP_NUM_THREADS`; the benchmark worker avoids loading
optional thread-control helper packages before FAISS/cuVS.

The same explicit-runtime convention is used by the NN metrics and k-means
benchmark scripts.

## NN Metric Cycles

`benchmark_scripts/benchmark_nn_metrics.R` focuses on faissR's public `nn()`
method matrix. It benchmarks `backend = "auto"`, `"cpu"`, and `"cuda"` across
the public methods, the four public metrics (`"euclidean"`, `"cosine"`,
`"correlation"`, and `"inner_product"`), and `k = 5, 10, 15, 50, 100` by
default. Metric aliases accepted by the API, such as `"l2"`, `"cor"`,
`"pearson"`, and `"ip"`, are canonicalized before preflight and reporting.
The public `method = "sparse"` route is included in the default method list,
but dense benchmark datasets record it as an expected skip because that route
is intended for sparse `Matrix` inputs.
Unsupported method/backend/metric combinations are preflighted with
`nn_capabilities()` and the public backend resolver, then written as expected
skips. Runtime expected skips also record when a resolved route requires
unavailable FAISS, FAISS GPU, CUDA, or RAPIDS cuVS support.

Use `--cycles=10` to repeat speed and recall measurements. The raw result table
contains one row per dataset/backend/method/metric/k/cycle combination.
`nn_metric_cycle_summary.csv` aggregates successful rows across cycles by
dataset/backend/method/metric/k and reports success counts, median/min/max
elapsed time, recall stability, and the dominant implementation backend.
`nn_metric_recommendations_from_cycles.csv` emits one row per
dataset/backend/metric/k. When recall is available, it selects the fastest
method whose median recall is at least the configured `recall_threshold`; if no
method reaches that threshold it selects the highest-recall row and marks
`recommendation_basis = "best_recall_below_threshold"`. When recall is
unavailable for the group, it selects the fastest successful row and marks
`recommendation_basis = "speed_only_no_recall"`.
`nn_metric_auto_vs_cycle_recommendation.csv` compares aggregate
`method = "auto"` rows with those recommendations and reports median speed
ratio, median recall gap, and backend/implementation agreement.
`nn_metric_best_by_dataset_backend_metric_k_cycle.csv` keeps the best row within
each cycle, while `nn_metric_best_by_dataset_backend_metric_k.csv` keeps the
overall best row across cycles for backward-compatible summaries.

Example CPU-focused metric run:

```sh
Rscript benchmark_scripts/benchmark_nn_metrics.R \
  --data_root=/path/to/Data \
  --out_dir=/path/to/faissR_NN_METRICS_CPU \
  --datasets=COIL20,USPS,FashionMNIST,MNIST \
  --backends=cpu \
  --methods=auto,exact,flat,hnsw,ivf,ivfpq,nsg,nndescent \
  --metrics=euclidean,cosine,correlation,inner_product \
  --k_values=5,10,15,50,100 \
  --threads=12 \
  --cycles=10
```

Example CUDA-focused metric run:

```sh
Rscript benchmark_scripts/benchmark_nn_metrics.R \
  --data_root=/path/to/Data \
  --out_dir=/path/to/faissR_NN_METRICS_CUDA \
  --datasets=COIL20,USPS,FashionMNIST,MNIST \
  --backends=cuda \
  --methods=auto,exact,flat,grid,ivf,ivfpq,nndescent,cagra \
  --metrics=euclidean,cosine,correlation,inner_product \
  --k_values=5,10,15,50,100 \
  --threads=2 \
  --cycles=10
```

## Graph Clustering

`benchmark_scripts/benchmark_graph_clustering.R` measures the two-stage graph
workflow:

1. `knn_graph()` builds a weighted nearest-neighbour graph.
2. `graph_cluster()` runs random-walking, Louvain, or Leiden clustering.

The script records graph-construction time, clustering time, total time, peak
resident memory when available, number of edges, number of communities,
modularity, and adjusted Rand index (ARI) against `dataset$labels` when labels
are present. ARI is computed by the benchmark helper in
`benchmark_scripts/source.R`; it is not part of the public package API.
For reproducibility and speed, each KNN graph is built once per
dataset/cycle/k/graph-backend/weight combination and reused across clustering
methods and clustering backends within that cycle. The `cycle` column supports
repeated benchmark cycles, for example `--cycles=10` for repeated speed/ARI
improvement runs. The `graph_cached` column records reuse within a cycle.
`graph_sec` is the shared graph-construction time, `cluster_sec` is
clustering-only time, and `total_sec` is `graph_sec + cluster_sec` for the
complete graph-plus-clustering workflow represented by the row.
`graph_cluster_cycle_summary.csv` aggregates successful rows across cycles by
dataset/k/graph-backend/cluster-backend/method/weight and reports success
counts, median/min/max graph, clustering, and total time, ARI stability,
modularity stability, community counts, and resolved backend metadata.
`graph_cluster_recommendations_from_cycles.csv` selects the fastest successful
graph/clustering/backend/method row within `ari_tolerance` of the best median
ARI for each dataset/k/target-cluster-count combination; when ARI is
unavailable it selects the fastest median total-time row. The
`recommendation_basis` column records whether the row was selected as
`"fastest_within_ari_tolerance"` or `"speed_only_no_ari"`.
`graph_cluster_auto_vs_cycle_recommendation.csv` compares aggregate rows where
the graph or clustering backend was `"auto"` against those recommendations and
reports median speed ratio, median ARI gap, modularity gap, and
backend/method agreement.
The result table stores both requested and resolved public backend metadata:
`graph_backend`/`cluster_backend` are the user requests, while
`graph_preflight_route`/`cluster_preflight_route` show the resolver decision
before runtime availability checks and
`graph_resolved_backend`/`cluster_resolved_backend` show the public device
policy recorded by successful result objects after `"auto"` resolution.
By default, graph construction and graph clustering are each tested with
`"auto"`, `"cpu"`, and `"cuda"` backends. `backend = "auto"` may resolve to CPU
when CUDA/cuGraph support is not available.
Known unsupported graph-clustering combinations from the public API, such as
CUDA random-walking or explicit CUDA clustering without libcugraph, are
recorded as `status = "expected_skip"` with `expected_skip = TRUE`; if every
row in a graph-build block is an expected skip, graph construction is skipped
and graph timing/edge columns remain `NA`. Explicit CUDA graph construction
without a CUDA-capable faissR runtime is also recorded as an expected skip.
Unexpected runtime errors remain failed rows.

Example CPU run:

```sh
Rscript benchmark_scripts/benchmark_graph_clustering.R \
  --data_root=/path/to/Data \
  --out_dir=/path/to/faissR_GRAPH_CLUSTER_CPU \
  --datasets=COIL20,USPS,FashionMNIST,MNIST \
  --k_values=15,50,100 \
  --cycles=10 \
  --ari_tolerance=0.01 \
  --graph_backends=cpu \
  --cluster_backends=cpu \
  --methods=random_walking,louvain,leiden \
  --threads=12
```

Example CUDA run:

```sh
Rscript benchmark_scripts/benchmark_graph_clustering.R \
  --data_root=/path/to/Data \
  --out_dir=/path/to/faissR_GRAPH_CLUSTER_CUDA \
  --datasets=COIL20,USPS,FashionMNIST,MNIST \
  --k_values=15,50,100 \
  --cycles=10 \
  --ari_tolerance=0.01 \
  --graph_backends=cuda \
  --cluster_backends=cuda \
  --methods=louvain,leiden \
  --threads=2
```

When `--target_clusters=labels` is used, Louvain and Leiden use
`n_clusters = length(unique(dataset$labels))`. If the selected method set
contains only Louvain and Leiden, the benchmark stores that target on the graph
with `knn_graph(n_clusters = ...)` and lets `graph_cluster()` reuse it. Mixed
method sets that include random-walking pass the target only to Louvain/Leiden
rows because the public API intentionally reserves `n_clusters` for Louvain and
Leiden. CUDA rows fail explicitly when faissR was not built with the required
CUDA/cuGraph support.

## NN Metrics File Layout

`benchmark_scripts/benchmark_nn_metrics.R` is a faissR-only nearest-neighbour
metric matrix. It runs public `nn()` combinations over:

- backends: `"auto"`, `"cpu"`, `"cuda"`, or any subset passed with
  `--backends`;
- methods: `"auto"`, `"exact"`, `"flat"`, `"bruteforce"`, `"grid"`,
  `"vptree"`, `"sparse"`, `"hnsw"`, `"ivf"`, `"ivfpq"`, `"nsg"`,
  `"nndescent"`, and `"cagra"`;
- metrics: `"euclidean"`, `"cosine"`, `"correlation"`, and
  `"inner_product"` after alias canonicalization;
- k values: `5`, `10`, `15`, `50`, and `100` by default.

Unsupported combinations are preflighted with `faissR::nn_capabilities()` and
the public backend resolver, then saved as `status = "expected_skip"` rows with
`expected_skip = TRUE`; the design-level capability table used for the run is
saved as `nn_metric_capabilities.csv`. For `backend = "auto"`, the preflight
checks the resolved CPU/CUDA route and records expected skips when that route
requires unavailable FAISS, FAISS GPU, CUDA, or RAPIDS cuVS support.
The sparse route is also recorded as an expected skip for dense benchmark
matrices so dense datasets are not converted just to exercise a sparse-specific
method.
Unexpected runtime errors remain ordinary failed rows. Recall is computed
against exact CPU references when feasible. Small datasets use a full exact
self-KNN reference; larger datasets use a deterministic sample of query rows
when `quality_n * nrow(data) * ncol(data)` fits `--quality_max_ops`. The
`recall_reference` and `recall_query_n` columns record whether recall used a
full or sampled exact reference. The script also writes
`nn_metric_fastest_at_recall_threshold.csv`, which records the fastest
successful method per dataset/backend/metric/k whose recall is at least
`--recall_threshold` when recall is available. When `method = "auto"` is part
of the run, `nn_metric_auto_vs_fastest.csv` compares auto against that fastest
high-recall row and reports speed ratio, recall gap, whether auto itself was
the fastest high-recall method, whether the result-facing backend matches, and
whether the concrete implementation backend matches. The result table separates
`result_backend`, `resolved_backend`, and `implementation_backend` so public
device labels such as `"cuda"` can be distinguished from concrete FAISS/cuVS
implementation labels such as `"faiss_gpu_cagra"` or `"cuda_cuvs_cagra"`.
The aggregate file `nn_metric_recommendations_from_cycles.csv` emits one row
per dataset/backend/metric/k: it chooses the fastest median row above the recall
threshold when possible, the best-recall row when all measured methods are below
threshold, and the fastest successful row when recall is unavailable. The
`recommendation_basis` column records which rule was used.

Example CPU run:

```sh
Rscript benchmark_scripts/benchmark_nn_metrics.R \
  --data_root=/path/to/Data \
  --out_dir=/path/to/faissR_NN_METRICS_CPU \
  --backends=cpu \
  --metrics=euclidean,cosine,correlation,inner_product \
  --k_values=5,10,15,50,100 \
  --recall_threshold=0.98 \
  --threads=12
```

Example CUDA run:

```sh
Rscript benchmark_scripts/benchmark_nn_metrics.R \
  --data_root=/path/to/Data \
  --out_dir=/path/to/faissR_NN_METRICS_CUDA \
  --backends=cuda \
  --metrics=euclidean,cosine,correlation,inner_product \
  --k_values=5,10,15,50,100 \
  --recall_threshold=0.98 \
  --threads=2
```

## K-Means

`benchmark_scripts/benchmark_kmeans.R` compares `fast_kmeans()` with
`backend = "auto"`, `"cpu"`, and `"cuda"` against base `stats::kmeans` by
default. It records elapsed time, peak resident memory when available, backend
used, total within-cluster sum of squares, iterations, selected k-means
parameters, tuning policy, benchmark cycle, and ARI against `dataset$labels` when labels are
available. The result table separates `requested_backend`, `resolved_backend`,
and `backend_used`, so `"auto"` device policy and the actual implementation
(`"faiss"`, `"cpu"`, `"cuda_faiss"`, `"cuda_cuvs"`, or `"stats"`) can be
audited directly. When `stats` is part of the run, `kmeans_fast_vs_stats.csv` compares
each successful `fast_kmeans()` row against `stats::kmeans` for the same
dataset, cycle, and number of centers, reporting speedup, ARI delta, and
withinss ratio. Use `--cycles=10` to repeat speed/ARI measurements without
hand-launching the same benchmark multiple times. `kmeans_cycle_summary.csv`
aggregates successful rows across cycles by dataset/method/backend/centers and
reports success counts, median/min/max elapsed time, ARI stability, withinss
stability, iteration counts, and resolved backend metadata.
`kmeans_recommendations_from_cycles.csv` selects the fastest row within
`ari_tolerance` of the best median ARI for each dataset/centers combination;
when ARI is unavailable it selects the fastest median-time row. The
`recommendation_basis` column records whether the row was selected as
`"fastest_within_ari_tolerance"` or `"speed_only_no_ari"`.
`kmeans_fast_vs_cycle_recommendation.csv` compares aggregate `fast_kmeans()`
rows with those recommendations and reports median speed ratio, median ARI gap,
withinss ratio, and backend/implementation agreement.
Explicit CUDA/library combinations that are known unavailable before execution
are recorded as `status = "expected_skip"` with `expected_skip = TRUE`, while
`resolved_backend` remains `"cuda"` so the skipped public device request is
auditable. `backend = "auto"` resolves to CPU instead of becoming an expected
skip when no k-means-capable CUDA route is available. Unexpected runtime errors
remain failed rows and are not replaced with CPU timings.

Example CPU run:

```sh
Rscript benchmark_scripts/benchmark_kmeans.R \
  --data_root=/path/to/Data \
  --out_dir=/path/to/faissR_KMEANS_CPU \
  --datasets=COIL20,USPS,FashionMNIST,MNIST \
  --methods=fast_kmeans,stats \
  --backends=cpu \
  --centers=10 \
  --threads=12 \
  --cycles=10 \
  --ari_tolerance=0.01
```

Example CUDA run:

```sh
Rscript benchmark_scripts/benchmark_kmeans.R \
  --data_root=/path/to/Data \
  --out_dir=/path/to/faissR_KMEANS_CUDA \
  --datasets=COIL20,USPS,FashionMNIST,MNIST \
  --methods=fast_kmeans \
  --backends=cuda \
  --centers=10 \
  --threads=2 \
  --cycles=10 \
  --ari_tolerance=0.01
```
