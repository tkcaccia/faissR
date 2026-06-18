# Examples

[Home](../README.md) |
[Installation](installation.md) |
[Implementation](implementation.md) |
**Examples** |
[Benchmarks](benchmarks.md) |
[API](usage-api.md) |
[Backends](backend-capabilities.md) |
[References](references.md)

## KNN On Iris

```r
library(faissR)

x <- scale(as.matrix(iris[, 1:4]))
labels <- iris$Species

knn <- nn(x, k = 15, backend = "auto", metric = "euclidean", n_threads = 4)
head(knn$indices)
head(knn$distances)
```

## Cosine Search

Cosine search is implemented by row-normalizing the input and using
inner-product search where supported.

```r
knn_cos <- nn(x, k = 15, backend = "auto", metric = "cosine", n_threads = 4)
```

## Shared Nearest-Neighbour Graph

```r
if (requireNamespace("igraph", quietly = TRUE)) {
  g <- knn_graph(knn, k = 15, weight = "snn")
  cl <- igraph::cluster_louvain(g, weights = igraph::E(g)$weight)
  table(labels, igraph::membership(cl))
}
```

## kNN Classifier

```r
set.seed(1)
train <- sample(seq_len(nrow(x)), 100)
test <- setdiff(seq_len(nrow(x)), train)

fit <- knn_fit(x[train, ], labels[train], backend = "auto", metric = "euclidean")
pred <- predict(fit, x[test, ], k = 5)
mean(pred == labels[test])

prob <- predict_proba(fit, x[test, ], k = 5)
head(prob)
```

## k-means

```r
km <- fast_kmeans(x, centers = 3, backend = "auto", n_threads = 4)
table(km$cluster)
```

## Reuse KNN In fastEmbedR

```r
library(fastEmbedR)

y <- fastEmbedR::opentsne_knn(knn, init_data = x, backend = "cpu")
plot(y, pch = 21, bg = labels)
```
