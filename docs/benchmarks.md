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

The benchmark scripts default to the real datasets `COIL20`, `USPS`,
`FashionMNIST`, `FlowRepository_FR-FCM-ZYRM_files`, `flow18`, `MNIST`,
`imagenet`, `MetRef`, and `mass41` from the configured `Data` directory.
NN metric benchmarks also include simulated uniform 2D and 3D datasets by
default. Graph-clustering benchmarks include those uniform datasets plus a
small labelled simulated three-cluster dataset for ARI sanity checks; the
k-means benchmark also includes that labelled three-cluster dataset.

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
four public metrics L2/Euclidean, cosine, correlation, and inner product.
Correlation is centered cosine similarity, while inner product is the raw dot
product, so benchmark rows for these metrics are not interchangeable. Flat
inner-product searches are reported under the same public `method = "flat"` row
rather than duplicate Flat-IP rows. Implementation-specific faissR rows,
such as FAISS GPU IVF and direct cuVS rows, are timed through faissR's internal
benchmark route so the table can distinguish FAISS GPU indexes that use NVIDIA
cuVS internally from direct RAPIDS cuVS API calls.
The file `benchmark1_runtime_capabilities.csv` records the faissR Benchmark #1
method/metric preflight table, including legacy Benchmark #1 method labels,
equivalent public `nn()` routes where available, execution backends, metric
support, `public_runtime_reason`, `runtime_available`, `runtime_reason`, and
current runtime availability notes. Runtime-unavailable faissR rows are
recorded as skipped before loading dataset matrices.

If a non-standard runtime library directory is needed, set `FAISSR_ENV_DIR`
explicitly before launch. The script no longer treats an unrelated active
`CONDA_PREFIX` as a FAISS runtime, which avoids accidental library-path
pollution on local machines. CPU worker threads are controlled with environment
variables such as `OMP_NUM_THREADS`; the benchmark worker avoids loading
optional thread-control helper packages before FAISS/cuVS.
Benchmark #1 accepts the public metric aliases `euclidean`, `pearson`, `cor`,
`ip`, and `innerproduct`, but unknown metric labels now stop the launcher before
workers are submitted. Numeric controls that define the timing and quality
envelope, including `--threads`, `--timeout`, `--quality_n`, and
`--quality_max_ops`, are validated before workers are submitted.

The same explicit-runtime convention is used by the NN metrics and k-means
benchmark scripts.

The legacy Benchmark #1 summary file `benchmark1_best_by_dataset.csv` is
quality-aware: within each dataset/metric/k group it ranks successful KNN rows
by recall@k, neighbour-rank correlation, mean relative distance error, elapsed
time, and peak memory. The companion
`benchmark1_ranked_speed_quality_memory.csv` preserves the same ordering for
all successful rows. This means the "best" row is not simply the fastest row
when a slower method has better measured nearest-neighbour quality. Invalid or
non-finite distance/rank quality summaries are recorded as `NA` and therefore
do not masquerade as successful quality measurements. Its `--k_values` grid
follows the same positive-integer validation as the newer NN metric benchmark.

## NN Metric Cycles

`benchmark_scripts/benchmark_nn_metrics.R` focuses on faissR's public `nn()`
method matrix. It benchmarks `backend = "auto"`, `"cpu"`, and `"cuda"` across
the public methods, the four public metrics (`"euclidean"`, `"cosine"`,
`"correlation"`, and `"inner_product"`), and `k = 5, 10, 15, 50, 100` by
default. Correlation and inner product keep their distinct public meanings:
centered cosine similarity versus raw dot product. Metric aliases accepted by the API, such as `"l2"`, `"cor"`,
`"pearson"`, and `"ip"`, are canonicalized before preflight and reporting.
Unknown metric names now stop the script instead of silently falling back to
the default metric set, so command-line typos cannot contaminate timing tables.
`--k_values` must contain one or more positive integers; malformed entries stop
the script before datasets are loaded.
The public `method = "sparse"` route is included in the default method list,
but dense benchmark datasets record it as an expected skip because that route
is intended for sparse `Matrix` inputs.
The public `method = "grid"` route is also recorded as an expected skip for
datasets that are not two- or three-dimensional, because that method is a
native low-dimensional spatial search route.
The public `method = "nsg"` route is recorded as an expected skip for datasets
with 100 or fewer rows, because linked FAISS NSG builds require more than 100
training rows before graph construction can start.
Unsupported method/backend/metric combinations are preflighted with
`nn_capabilities()` and the public backend resolver, then written as expected
skips. Runtime expected skips also record when a resolved route requires
unavailable FAISS, FAISS GPU, CUDA, or RAPIDS cuVS support.

Use `--cycles=10` to repeat speed and recall measurements. The raw result table
contains one row per dataset/backend/method/metric/k/cycle combination.
`--recall_threshold` must be a numeric value between 0 and 1; invalid values
stop before the benchmark starts instead of silently changing recommendation
rules. `--threads`, `--timeout`, `--cycles`, `--quality_n`, and
`--quality_max_ops` are also validated before datasets are loaded.
`nn_metric_cycle_summary.csv` aggregates successful rows across cycles by
dataset/backend/method/metric/k and reports success counts, median/min/max
elapsed time, recall stability, CPU thread count, preflight route, and the
dominant implementation backend. New runs also preserve the public request
stored on `nn()` results (`result_requested_backend`,
`result_requested_method`, and `result_tuning`), compact `route_parameters`
metadata from FAISS/cuVS/native result attributes, and `tuning_status` when a
backend reports tuning.
`nn_metric_recommendations_from_cycles.csv` emits one row per
dataset/backend/metric/k. When recall is available, it selects the fastest
method whose median recall is at least the configured `recall_threshold`; if no
method reaches that threshold it selects the highest-recall row and marks
`recommendation_basis = "best_recall_below_threshold"`. Above-threshold speed
ties are broken by higher median recall, minimum recall, and median minimum
recall; below-threshold median-recall ties are broken by minimum recall, median
minimum recall, and then speed. When recall is
unavailable for the group, it selects the fastest successful row and marks
`recommendation_basis = "speed_only_no_recall"`.
`nn_metric_auto_vs_cycle_recommendation.csv` compares aggregate
`method = "auto"` rows with those recommendations and reports median speed
ratio, median recall gap, CPU thread count, preflight route,
route-parameter/tuning metadata, backend/implementation agreement, and the
recommendation basis used for the recommended row. Speed ratios and recall gaps
are `NA` when the required timing or recall values are unavailable or invalid.
`nn_metric_best_by_dataset_backend_metric_k_cycle.csv` keeps the best row within
each cycle using the same recall-threshold rule: fastest above threshold,
best recall below threshold, and fastest when recall is unavailable.
`nn_metric_best_by_dataset_backend_metric_k.csv` keeps the overall best row
across cycles with the same rule for backward-compatible summaries.
`MATERIALS_AND_METHODS_nn_metrics.md` records the corresponding paper-ready
methods text, including the metric grid, k grid, recall rules, expected-skip
policy, and output-file definitions.

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
The run configuration is saved as `graph_cluster_benchmark_config.csv`, and
the raw row-level result table is saved as
`graph_cluster_benchmark_results.csv`, including graph vertex and edge counts.
Expected skips are marked with `expected_skip = TRUE` and a machine-readable
`expected_skip_reason`, so runtime, shape, and input-type skips can be grouped
without parsing the prose error message.
The config includes `available_datasets`, the validated real plus simulated
dataset names accepted by the `--datasets` selector, so subset runs remain
auditable.
`graph_cluster_cycle_summary.csv` aggregates successful rows across cycles by
dataset/k/graph-backend/cluster-backend/method/weight and reports success
counts, median/min/max graph, clustering, and total time, ARI stability,
modularity stability, graph size, community counts, CPU thread count, preflight
routes, and resolved backend metadata.
`graph_cluster_nn_capabilities.csv` stores the graph-construction
`nn_capabilities(runtime = TRUE)` table, including `runtime_reason` and
`runtime_notes` for runtime-unavailable KNN routes.
When `n_clusters` is used, graph-clustering result metadata also records
`target_gap`, `resolution_selection`, and a `resolution_search` table whose
selected row is marked with `selected = TRUE`, so target-count resolution
searches remain auditable in downstream summaries. The raw and cycle-summary
CSVs flatten the most important diagnostics as `resolution_selected_candidate`,
`resolution_candidates`, `resolution_min_target_gap`, and
`resolution_selected_is_min_gap`, making it possible to check whether the
selected resolution achieved the best observed target-count gap without opening
the R object.
`graph_cluster_best_by_dataset.csv` keeps a compact best successful row per
dataset after ranking by ARI, modularity, and total time for backward-compatible
summaries. `graph_cluster_best_by_dataset_k_target.csv` keeps the same
best-row ranking per dataset/k/target-cluster-count combination, which is the
safer table for comparing neighbourhood sizes and Louvain/Leiden target counts.
`graph_cluster_recommendations_from_cycles.csv` selects the fastest successful
graph/clustering method row within `ari_tolerance` of the best median ARI for
each dataset/k/graph-backend/cluster-backend/target-cluster-count combination;
when ARI is available and median total times tie, higher median ARI and then
higher median modularity break the tie. `--ari_tolerance` must be a
non-negative number and is validated before datasets are loaded. `--threads`,
`--timeout`, and `--cycles` must be positive integers and are also validated
before data loading.
When ARI is unavailable it selects the fastest median total-time row. The
`recommendation_basis` column records whether the row was selected as
`"fastest_within_ari_tolerance"` or `"speed_only_no_ari"`.
`graph_cluster_auto_vs_cycle_recommendation.csv` compares aggregate rows where
the graph or clustering backend was `"auto"` against recommendations from the
same requested graph-backend/cluster-backend group and reports the
recommendation basis, median speed ratio, median ARI gap, modularity gap,
method agreement, and resolved-backend agreement. Speed ratios and quality gaps
are `NA` when the required timing, ARI, or modularity values are unavailable or
invalid.
`MATERIALS_AND_METHODS_graph_clustering.md` records the corresponding
paper-ready methods text, including graph reuse, ARI/modularity reporting,
target-cluster handling, expected-skip policy, and output-file definitions.
The result table stores both requested and resolved public backend metadata:
`graph_backend`/`cluster_backend` are the user requests, while
`graph_preflight_route`/`cluster_preflight_route` show the resolver decision
before runtime availability checks and
`graph_resolved_backend`/`cluster_resolved_backend` show the public device
policy recorded by successful result objects after `"auto"` resolution. The
route columns and `n_threads` are also preserved in cycle summaries and
auto/recommendation comparisons.
By default, graph construction and graph clustering are each tested with
`"auto"`, `"cpu"`, and `"cuda"` backends. `backend = "auto"` may resolve to CPU
when CUDA/cuGraph support is not available.
Graph construction can also vary the nearest-neighbour route and metric with
`--graph_methods` and `--metrics`. The default uses
`--graph_methods=auto`,
`--metrics=euclidean,cosine,correlation,inner_product`, and
`--k_values=5,10,15,50,100` so benchmark rows cover the public metric surface
and the full requested graph-density grid. Expanded HPC runs can also use
public NN methods such as
`auto,hnsw,ivf,nndescent,grid` and metrics such as
`euclidean,cosine,correlation,inner_product` to evaluate how graph construction
affects clustering ARI and speed.
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
  --k_values=5,10,15,50,100 \
  --cycles=10 \
  --ari_tolerance=0.01 \
  --graph_backends=cpu \
  --graph_methods=auto,hnsw,ivf,nndescent \
  --metrics=euclidean,cosine,correlation,inner_product \
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
  --k_values=5,10,15,50,100 \
  --cycles=10 \
  --ari_tolerance=0.01 \
  --graph_backends=cuda \
  --graph_methods=auto,cagra,nndescent,ivf \
  --metrics=euclidean,cosine,correlation,inner_product \
  --cluster_backends=cuda \
  --methods=louvain,leiden \
  --threads=2
```

`--k_values` must contain one or more positive integers; malformed entries stop
the graph benchmark before datasets are loaded.
`--target_clusters` is normalized to either `labels` or `none`, and invalid
values stop before the benchmark starts. Method, graph-backend, and
cluster-backend selectors are also validated against the public benchmark
choices before any dataset is loaded. `--graph_methods` accepts the same public
NN method labels as `nn()` and `knn_graph()`, while `--metrics` accepts the four
public NN metrics and the same aliases as `nn()`, such as `l2`, `pearson`,
`cor`, `ip`, and `dot-product`; aliases are canonicalized before preflight and
reporting. Unsupported graph method/backend/metric combinations are
recorded as expected skips using `nn_capabilities(runtime = TRUE)` plus
data-shape checks such as the 2D/3D requirement for `method = "grid"` and the
more-than-100-row FAISS NSG construction requirement for `method = "nsg"`.
The runtime capability table is written to
`graph_cluster_nn_capabilities.csv`, so unavailable FAISS GPU, CUDA, and cuVS
graph-construction routes can be audited before any graph is built. When
`--target_clusters=labels` is
used, Louvain and Leiden use `n_clusters = length(unique(dataset$labels))`. If
the selected method set contains only Louvain and Leiden, the benchmark stores
that target on the graph with `knn_graph(n_clusters = ...)` and lets
`graph_cluster()` reuse it. Mixed method sets that include random-walking pass
the target only to Louvain/Leiden rows because the public API intentionally
reserves `n_clusters` for Louvain and Leiden. `n_clusters_requested` records
this requested target count, while `n_communities` records the actual
community count returned by clustering. The target is a convenience target, not
a hard guarantee. It is validated as a positive integer no larger than the
graph vertex count, so malformed label-derived targets fail clearly before
clustering. CUDA rows fail explicitly when faissR was not built with the
required CUDA/cuGraph support.

## NN Metrics File Layout

`benchmark_scripts/benchmark_nn_metrics.R` is a faissR-only nearest-neighbour
metric matrix. It runs public `nn()` combinations over:

- backends: `"auto"`, `"cpu"`, `"cuda"`, or any subset passed with
  `--backends`;
- methods: `"auto"`, `"exact"`, `"flat"`, `"bruteforce"`, `"grid"`,
  `"vptree"`, `"sparse"`, `"hnsw"`, `"ivf"`, `"ivfpq"`, `"nsg"`,
  `"nndescent"`, and `"cagra"`; these must be canonical lowercase public
  method labels, not resolved backend labels such as `faiss_hnsw`;
- metrics: `"euclidean"`, `"cosine"`, `"correlation"`, and
  `"inner_product"` after alias canonicalization; correlation is centered
  cosine similarity, while inner product ranks by larger raw dot product and
  reports shifted smaller-is-better distances;
- k values: `5`, `10`, `15`, `50`, and `100` by default.

Unsupported combinations are preflighted with `faissR::nn_capabilities(runtime = TRUE)` and
the public backend resolver, then saved as `status = "expected_skip"` rows with
`expected_skip = TRUE`; the raw result table also records
`expected_skip_reason` so runtime, shape, and input-type skips can be grouped
without parsing the prose error message. The run configuration is saved as
`nn_metric_benchmark_config.csv`, the raw row-level result table is saved as
`nn_metric_benchmark_results.csv`, and the runtime-aware capability table used
for the run is saved as `nn_metric_capabilities.csv`, including public
`backend = "auto"`, `"cpu"`, and `"cuda"` rows plus `resolved_backend`,
`runtime_available`, `runtime_reason`, and `runtime_notes`. For
`backend = "auto"`, the
preflight first checks the explicit auto capability row, then checks the
resolved CPU/CUDA route and records expected skips when that route requires
unavailable FAISS, FAISS GPU, CUDA, or RAPIDS cuVS support.
The config includes `available_datasets`, the validated real plus simulated
dataset names accepted by the `--datasets` selector, which makes partial or
subset reruns traceable to the full benchmark universe.
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
whether the concrete implementation backend matches. Speed ratios and recall
gaps are `NA` when the required timing or recall values are missing or invalid.
The result table separates `result_requested_backend`,
`result_requested_method`, `result_tuning`, `result_backend`,
`resolved_backend`, and `implementation_backend` so public device labels such as
`"cuda"` can be distinguished from concrete FAISS/cuVS implementation labels
such as `"faiss_gpu_cagra"` or `"cuda_cuvs_cagra"`.
The aggregate file `nn_metric_recommendations_from_cycles.csv` emits one row
per dataset/backend/metric/k: it chooses the fastest median row above the recall
threshold when possible, the best-recall row when all measured methods are below
threshold, and the fastest successful row when recall is unavailable. Ties are
resolved deterministically with minimum-recall stability before falling back to
speed for below-threshold groups. The
`recommendation_basis` column records which rule was used.
`nn_metric_auto_vs_cycle_recommendation.csv` carries this value as
`recommended_recommendation_basis` so auto comparisons can be interpreted as
recall-qualified, below-threshold, or speed-only comparisons. Cycle summaries
and auto comparisons also preserve `n_threads` and `preflight_route`, so CPU
threading and public route decisions remain auditable after aggregation.

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
used, total within-cluster sum of squares, iterations, `converged`,
`hit_max_iter`, selected k-means
parameters, tuning policy, benchmark cycle, and ARI against `dataset$labels` when labels are
available. The result table separates `requested_backend`, `resolved_backend`,
and `backend_used`, so `"auto"` device policy and the actual implementation
(`"faiss"`, `"cpu"`, `"cuda_faiss"`, `"cuda_cuvs"`, or `"stats"`) can be
audited directly. The run configuration is saved as
`kmeans_benchmark_config.csv`, and the raw row-level result table is saved as
`kmeans_benchmark_results.csv`. The runtime preflight table is saved as
`kmeans_runtime_capabilities.csv`, including CUDA, FAISS GPU, and cuVS
availability, `runtime_reason`, human-readable `runtime_notes`, and whether
explicit CUDA k-means requests are runnable in the current build. The
`runtime_reason` field distinguishes available routes from
`missing_cuda_runtime` and `missing_gpu_kmeans_backend` preflight skips.
`--centers` must be a positive integer; when
dataset labels are available, the benchmark uses the label-derived cluster
count for that dataset and otherwise uses the validated `--centers` fallback.
The config includes `available_datasets`, the validated real plus simulated
dataset names accepted by the `--datasets` selector.
Method and backend selectors are validated before loading datasets, so typos in
`--methods` or `--backends` stop the run instead of becoming failed benchmark
rows. `--threads`, `--timeout`, and `--cycles` must be positive integers and
are also validated before data loading.
When `stats` is part of the run,
`kmeans_fast_vs_stats.csv` compares
each successful `fast_kmeans()` row against `stats::kmeans` for the same
dataset, cycle, and number of centers, reporting speedup, ARI delta, and
withinss ratio. Speedups, ARI deltas, and withinss ratios are `NA` when the
required timing or quality values are missing or invalid. Use `--cycles=10` to
repeat speed/ARI measurements without hand-launching the same benchmark
multiple times. `kmeans_cycle_summary.csv`
aggregates successful rows across cycles by dataset/method/backend/centers and
reports success counts, median/min/max elapsed time, ARI stability, withinss
stability, iteration counts, whether any cycle hit `max_iter`, whether all
cycles converged before the iteration cap, selected parameter medians,
deterministic tuning rule/shape metadata, and resolved backend metadata.
`kmeans_best_by_dataset.csv` keeps a compact best successful row per dataset
after ranking by ARI, elapsed time, and total within-cluster sum of squares for
backward-compatible summaries. `kmeans_best_by_dataset_centers.csv` keeps the
same best-row ranking per dataset/centers combination, which is the safer table
for comparing different requested cluster counts.
`kmeans_recommendations_from_cycles.csv` selects the fastest row within
`ari_tolerance` of the best median ARI for each dataset/centers combination;
`--ari_tolerance` must be a non-negative number and is validated before
datasets are loaded.
When ARI is available and median times tie, higher median ARI and then lower
median total within-cluster sum of squares break the tie. When ARI is
unavailable it selects the fastest median-time row. The
`recommendation_basis` column records whether the row was selected as
`"fastest_within_ari_tolerance"` or `"speed_only_no_ari"`.
`kmeans_backend_recommendations_from_cycles.csv` applies the same rule within
each dataset/centers/backend group, so CPU, CUDA, auto, and stats results can
be tuned or reported separately without losing the overall recommendation.
`kmeans_fast_vs_cycle_recommendation.csv` compares aggregate `fast_kmeans()`
rows with those recommendations and reports median speed ratio, median ARI gap,
withinss ratio, selected tuning metadata, requested/resolved backend metadata,
CPU thread count, static no-pilot selection metadata, backend/implementation
agreement, and the recommendation basis used for the recommended row. Speed
ratios, ARI gaps, and withinss ratios are `NA` when the required timing or
quality values are missing or invalid.
`MATERIALS_AND_METHODS_kmeans.md` records the corresponding paper-ready
methods text, including centers selection, ARI/withinss reporting, tuning
policy, expected-skip policy, and output-file definitions.
Explicit CUDA/library combinations that are known unavailable before execution
are recorded as `status = "expected_skip"` with `expected_skip = TRUE`, while
`resolved_backend` remains `"cuda"` so the skipped public device request is
auditable. The skip decision is derived from `kmeans_runtime_capabilities.csv`.
`backend = "auto"` resolves to CPU instead of becoming an expected
skip when no k-means-capable CUDA route is available, and it can also resolve
to CPU for small k-means jobs where the deterministic shape gate estimates that
GPU launch/copy overhead would dominate. Unexpected runtime errors remain
failed rows and are not replaced with CPU timings.
The package records the same decision in
`parameters$tuning$backend_policy`, including a reason string such as
`small_cpu_preferred`, `work_at_least_1e8`, `input_at_least_256MiB`, or
`large_high_dimensional_input`, the estimated work and input bytes, and the
deterministic threshold values (`work_threshold`, `nbytes_threshold`,
`large_n_threshold`, and `large_p_threshold`) used for the CPU/CUDA decision.
Benchmark rows also record `selection_*` columns from
`parameters$tuning$selection`, including the predicted backend, backend-policy
reason, runtime capability flags, work/input-size estimates, and
`selection_slow_tuning = FALSE`. Benchmark summaries can therefore explain auto
CPU/CUDA selection without running extra pilot jobs.

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
