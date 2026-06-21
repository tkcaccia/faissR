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

The same object can then feed `fastEmbedR`, graph construction, classifier
tests, and recall diagnostics without paying the KNN cost repeatedly.

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
dataset/k/graph-backend/weight combination and reused across clustering
methods and clustering backends. The `graph_cached` column records this reuse.
`graph_sec` is the shared graph-construction time, `cluster_sec` is
clustering-only time, and `total_sec` is `graph_sec + cluster_sec` for the
complete graph-plus-clustering workflow represented by the row.
Known unsupported graph-clustering combinations from the public API, such as
CUDA random-walking, are recorded as `status = "expected_skip"` with
`expected_skip = TRUE`; unexpected runtime errors remain failed rows.

Example CPU run:

```sh
Rscript benchmark_scripts/benchmark_graph_clustering.R \
  --data_root=/path/to/Data \
  --out_dir=/path/to/faissR_GRAPH_CLUSTER_CPU \
  --datasets=COIL20,USPS,FashionMNIST,MNIST \
  --k_values=15,50,100 \
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
  --graph_backends=cuda \
  --cluster_backends=cuda \
  --methods=louvain,leiden \
  --threads=2
```

When `--target_clusters=labels` is used, Louvain and Leiden receive
`n_clusters = length(unique(dataset$labels))`. Random-walking is run without a
cluster-count target because the public API intentionally reserves
`n_clusters` for Louvain and Leiden. CUDA rows fail explicitly when faissR was
not built with the required CUDA/cuGraph support.

## NN Metrics

`benchmark_scripts/benchmark_nn_metrics.R` is a faissR-only nearest-neighbour
metric matrix. It runs public `nn()` combinations over:

- backends: `"cpu"`, `"cuda"`, or any subset passed with `--backends`;
- methods: `"auto"`, `"exact"`, `"flat"`, `"bruteforce"`, `"grid"`,
  `"vptree"`, `"HNSW"`, `"IVF"`, `"IVFPQ"`, `"NSG"`, `"NNDescent"`, and
  `"CAGRA"`;
- metrics: `"euclidean"`, `"cosine"`, `"correlation"`, and
  `"inner_product"`;
- k values: `5`, `10`, `15`, `50`, and `100` by default.

Unsupported combinations are preflighted with `faissR::nn_capabilities()` and
saved as `status = "expected_skip"` rows with `expected_skip = TRUE`; the
capability table used for the run is saved as `nn_metric_capabilities.csv`.
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
high-recall row and reports speed ratio, recall gap, and whether the resolved
backend matches.

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
  --metrics=euclidean,inner_product \
  --k_values=5,10,15,50,100 \
  --recall_threshold=0.98 \
  --threads=2
```

## K-Means

`benchmark_scripts/benchmark_kmeans.R` compares `fast_kmeans()` CPU/CUDA
backends with base `stats::kmeans`. It records elapsed time, peak resident
memory when available, backend used, total within-cluster sum of squares,
iterations, selected k-means parameters, tuning policy, and ARI against
`dataset$labels` when labels are available. When `stats` is part of the run,
`kmeans_fast_vs_stats.csv` compares each successful `fast_kmeans()` row against
`stats::kmeans` for the same dataset and number of centers, reporting speedup,
ARI delta, and withinss ratio.

Example CPU run:

```sh
Rscript benchmark_scripts/benchmark_kmeans.R \
  --data_root=/path/to/Data \
  --out_dir=/path/to/faissR_KMEANS_CPU \
  --datasets=COIL20,USPS,FashionMNIST,MNIST \
  --methods=fast_kmeans,stats \
  --backends=cpu \
  --centers=10 \
  --threads=12
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
  --threads=2
```
