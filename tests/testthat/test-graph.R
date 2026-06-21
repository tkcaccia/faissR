test_that("knn_graph builds a native graph from a KNN object", {
  set.seed(501)
  x <- rbind(
    matrix(rnorm(60, -2, 0.2), ncol = 4),
    matrix(rnorm(60, 2, 0.2), ncol = 4)
  )
  knn <- nn(x, k = 6L, backend = "cpu")

  g <- knn_graph(knn, k = 5L, weight = "snn")

  expect_s3_class(g, "faissR_graph")
  expect_equal(g$n_vertices, nrow(x))
  expect_gt(g$n_edges, 0L)
  expect_true(all(g$weight > 0))
  expect_equal(attr(g, "faissR_graph")$weight, "snn")
})

test_that("knn_graph accepts embedding objects as native layout graphs", {
  set.seed(503)
  x <- rbind(
    matrix(rnorm(80, -2, 0.2), ncol = 4),
    matrix(rnorm(80, 2, 0.2), ncol = 4)
  )
  layout <- x[, 1:2, drop = FALSE]
  fit <- list(
    layout = layout,
    method = "opentsne"
  )
  class(fit) <- "fastEmbedR_embedding"

  g_embedding <- knn_graph(fit, k = 5L, backend = "cpu")
  expect_s3_class(g_embedding, "faissR_graph")
  expect_equal(attr(g_embedding, "faissR_graph")$space, "embedding")
  expect_equal(attr(g_embedding, "faissR_graph")$weight, "distance")
  expect_equal(attr(g_embedding, "faissR_graph")$input_method, "opentsne")
})

test_that("knn_graph supports adaptive weights and mutual edges natively", {
  set.seed(504)
  x <- rbind(
    matrix(rnorm(80, -2, 0.2), ncol = 2),
    matrix(rnorm(80, 2, 0.2), ncol = 2)
  )
  g_union <- knn_graph(x, k = 8L, backend = "cpu", weight = "adaptive", mutual = FALSE)
  g_mutual <- knn_graph(x, k = 8L, backend = "cpu", weight = "adaptive", mutual = TRUE)

  expect_s3_class(g_union, "faissR_graph")
  expect_s3_class(g_mutual, "faissR_graph")
  expect_equal(attr(g_union, "faissR_graph")$weight, "adaptive")
  expect_true(attr(g_mutual, "faissR_graph")$mutual)
  expect_lte(g_mutual$n_edges, g_union$n_edges)
  expect_true(all(g_mutual$weight > 0))
})

test_that("knn_graph stores an optional cluster-count target for graph_cluster", {
  set.seed(509)
  x <- rbind(
    matrix(rnorm(80, -3, 0.15), ncol = 4),
    matrix(rnorm(80, 0, 0.15), ncol = 4),
    matrix(rnorm(80, 3, 0.15), ncol = 4)
  )

  g <- knn_graph(x, k = 8L, backend = "cpu", n_clusters = 3L)
  expect_s3_class(g, "faissR_graph")
  expect_equal(attr(g, "faissR_graph")$target_n_clusters, 3L)

  cl <- graph_cluster(g, method = "louvain", backend = "cpu", n_threads = 2L, seed = 1L)
  expect_s3_class(cl, "faissR_graph_cluster")
  expect_equal(cl$target_n_clusters, 3L)
  expect_equal(cl$parameters$n_clusters, 3L)
  expect_s3_class(cl$resolution_search, "data.frame")
  expect_lte(abs(cl$n_communities - 3L), 1L)

  expect_error(
    graph_cluster(g, method = "random_walking", backend = "cpu", n_threads = 2L),
    "n_clusters"
  )
})

test_that("graph_cluster runs native CPU random-walk and Louvain clustering without igraph", {
  set.seed(505)
  x <- rbind(
    matrix(rnorm(80, -2, 0.2), ncol = 4),
    matrix(rnorm(80, 2, 0.2), ncol = 4)
  )

  walk <- graph_cluster(x, method = "random_walking", backend = "cpu", k = 8L, graph_backend = "cpu", steps = 3)
  expect_s3_class(walk, "faissR_graph_cluster")
  expect_length(walk$membership, nrow(x))
  expect_equal(walk$method, "random_walking")
  expect_equal(walk$implementation, "native_cpp")
  expect_true("Pons and Latapy (2006) for random-walk walktrap clustering" %in% walk$sources)

  knn <- nn(x, k = 8L, backend = "cpu")
  louvain <- graph_cluster(knn, method = "louvain", backend = "cpu", n_runs = 2, n_threads = 2, seed = 1)
  expect_s3_class(louvain, "faissR_graph_cluster")
  expect_length(louvain$membership, nrow(x))
  expect_length(louvain$all_modularity, 2L)
  expect_gte(length(unique(louvain$membership)), 2L)
})

test_that("graph_cluster keeps random-walking on CPU", {
  set.seed(5051)
  x <- rbind(
    matrix(rnorm(60, -2, 0.2), ncol = 4),
    matrix(rnorm(60, 2, 0.2), ncol = 4)
  )
  g <- knn_graph(x, k = 6L, backend = "cpu")

  auto <- graph_cluster(g, method = "random_walking", backend = "auto", n_threads = 2L)
  expect_equal(auto$backend, "cpu")
  expect_equal(auto$method, "random_walking")

  expect_error(
    graph_cluster(g, method = "random_walking", backend = "cuda", n_threads = 2L),
    "CPU-only"
  )
})

test_that("graph_cluster runs native CPU Leiden-style refinement without igraph", {
  set.seed(506)
  x <- rbind(
    matrix(rnorm(80, -2, 0.2), ncol = 4),
    matrix(rnorm(80, 2, 0.2), ncol = 4)
  )

  leiden <- graph_cluster(x, method = "leiden", backend = "cpu", k = 8L, graph_backend = "cpu", n_iterations = 4)
  expect_s3_class(leiden, "faissR_graph_cluster")
  expect_length(leiden$membership, nrow(x))
  expect_gte(length(unique(leiden$membership)), 2L)
  expect_true(any(grepl("GVE-Leiden", leiden$sources, fixed = TRUE)))
})

test_that("graph_cluster can target a requested number of communities", {
  set.seed(508)
  x <- rbind(
    matrix(rnorm(80, -3, 0.15), ncol = 4),
    matrix(rnorm(80, 0, 0.15), ncol = 4),
    matrix(rnorm(80, 3, 0.15), ncol = 4)
  )

  cl <- graph_cluster(
    x,
    method = "louvain",
    backend = "cpu",
    graph_backend = "cpu",
    k = 8L,
    n_clusters = 3L,
    n_threads = 2L,
    seed = 1L
  )

  expect_s3_class(cl, "faissR_graph_cluster")
  expect_equal(cl$target_n_clusters, 3L)
  expect_s3_class(cl$resolution_search, "data.frame")
  expect_true(nrow(cl$resolution_search) > 1L)
  expect_equal(cl$parameters$n_clusters, 3L)
  expect_equal(cl$parameters$selected_resolution, cl$selected_resolution)
  expect_lte(abs(cl$n_communities - 3L), 1L)

  expect_error(
    graph_cluster(x, method = "random_walking", backend = "cpu", graph_backend = "cpu", k = 8L, n_clusters = 3L),
    "n_clusters"
  )
})

test_that("graph_cluster reports native CUDA graph backend as unavailable without libcugraph", {
  skip_if(isTRUE(cugraph_available()), "libcugraph is available")
  set.seed(507)
  x <- matrix(rnorm(80), ncol = 4)
  expect_error(
    graph_cluster(x, method = "louvain", backend = "cuda", k = 5L, graph_backend = "cpu"),
    "libcugraph"
  )
})
