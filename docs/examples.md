# Examples

[Home](../README.md) |
[Installation](installation.md) |
[Implementation](implementation.md) |
**Examples** |
[Benchmarks](benchmarks.md) |
[API](usage-api.md) |
[NN Methods](nn-methods.md) |
[Backends](backend-capabilities.md) |
[References](references.md)

## KNN On Iris

```r
library(faissR)

x <- scale(as.matrix(iris[, 1:4]))
labels <- iris$Species

nn_res <- nn(x, k = 15, backend = "auto", metric = "euclidean", n_threads = 4)
head(nn_res$indices)
head(nn_res$distances)
```

## Non-Euclidean Metrics

Cosine and correlation use validated CPU paths. Inner-product search is
available for exact CPU scoring and FAISS Flat IP routes where supported.

```r
knn_cos <- nn(x, k = 15, backend = "auto", metric = "cosine", n_threads = 4)
knn_ip <- nn(x, k = 15, backend = "cpu", method = "flat",
             metric = "inner_product", n_threads = 4)
```

## Shared Nearest-Neighbour Graph And Leiden Clustering

```r
g <- knn_graph(nn_res, k = 15, weight = "snn")

leiden <- graph_cluster(g, method = "leiden", backend = "cpu", n_threads = 4)
louvain <- graph_cluster(g, method = "louvain", backend = "cpu", n_threads = 4)
walk <- graph_cluster(g, method = "random_walking", backend = "cpu", n_threads = 4)

table(labels, leiden$membership)
table(labels, louvain$membership)
table(labels, walk$membership)
```

If faissR was compiled with RAPIDS libcugraph, Louvain and Leiden can run on
CUDA without changing the API [9,11-12]:

```r
if (isTRUE(cugraph_available())) {
  leiden_gpu <- graph_cluster(g, method = "leiden", backend = "cuda")
  table(labels, leiden_gpu$membership)
}
```

## kNN Classifier

```r
set.seed(1)
train <- sample(seq_len(nrow(x)), 100)
test <- setdiff(seq_len(nrow(x)), train)

fit <- knn(x[train, ], labels[train], backend = "auto", metric = "euclidean")
pred <- predict(fit, x[test, ], k = 5)
mean(pred == labels[test])

prob <- predict(fit, x[test, ], k = 5, type = "prob")
head(prob)

# Fit and predict in one call
pred2 <- knn(x[train, ], labels[train], x[test, ], backend = "auto", k = 5)
```

## k-means

`fast_kmeans()` exposes CPU/FAISS/CUDA/cuVS k-means-style clustering routes
[7-8].

```r
km <- fast_kmeans(x, centers = 3, backend = "auto", n_threads = 4)
table(km$cluster)
km$parameters$tuning
```

## Reuse KNN In fastEmbedR

```r
library(fastEmbedR)

y <- fastEmbedR::opentsne_knn(nn_res, init_data = x, backend = "cpu")
plot(y, pch = 21, bg = labels)
```
