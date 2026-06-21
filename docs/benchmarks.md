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

Unsupported combinations are saved as failed rows with the package error
message. Recall is computed against exact CPU references only when the full
dataset fits the configured `--quality_n` and `--quality_max_ops` limits; large
datasets still contribute speed, memory, and availability rows.

Example CPU run:

```sh
Rscript benchmark_scripts/benchmark_nn_metrics.R \
  --data_root=/path/to/Data \
  --out_dir=/path/to/faissR_NN_METRICS_CPU \
  --backends=cpu \
  --metrics=euclidean,cosine,correlation,inner_product \
  --k_values=5,10,15,50,100 \
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
  --threads=2
```
