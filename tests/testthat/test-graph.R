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

test_that("precomputed KNN graph metadata prefers resolved backend", {
  knn <- list(
    indices = matrix(c(
      2L, 3L,
      1L, 3L,
      2L, 1L
    ), nrow = 3L, byrow = TRUE),
    distances = matrix(c(
      1, 2,
      1, 1.5,
      1.5, 2
    ), nrow = 3L, byrow = TRUE)
  )
  class(knn) <- "faissR_nn"
  attr(knn, "backend") <- "cuda"
  attr(knn, "requested_backend") <- "cuda"
  attr(knn, "resolved_backend") <- "faiss_gpu_cagra"
  attr(knn, "metric") <- "euclidean"

  g <- knn_graph(knn, k = 2L, weight = "distance")
  cl <- graph_cluster(knn, method = "louvain", backend = "cpu", k = 2L, weight = "distance")

  expect_equal(attr(g, "faissR_graph")$nn_backend, "faiss_gpu_cagra")
  expect_equal(attr(g, "faissR_graph")$resolved_backend, "faiss_gpu_cagra")
  expect_equal(cl$parameters$graph_backend, "faiss_gpu_cagra")
  expect_equal(cl$parameters$graph_requested_backend, "cuda")
  expect_equal(cl$parameters$graph_resolved_backend, "faiss_gpu_cagra")
})

test_that("precomputed KNN graph metadata fills missing requested backend", {
  knn <- list(
    indices = matrix(c(
      2L, 3L,
      1L, 3L,
      2L, 1L
    ), nrow = 3L, byrow = TRUE),
    distances = matrix(c(
      1, 2,
      1, 1.5,
      1.5, 2
    ), nrow = 3L, byrow = TRUE)
  )
  class(knn) <- "faissR_nn"
  attr(knn, "backend") <- "faiss_hnsw"
  attr(knn, "metric") <- "euclidean"

  cl <- graph_cluster(knn, method = "louvain", backend = "cpu", k = 2L, weight = "distance")

  expect_equal(cl$parameters$graph_backend, "faiss_hnsw")
  expect_equal(cl$parameters$graph_requested_backend, "faiss_hnsw")
  expect_equal(cl$parameters$graph_resolved_backend, "faiss_hnsw")
})

test_that("knn_graph accepts embedding-layout objects as native layout graphs", {
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
  class(fit) <- "faissR_embedding_layout"

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

test_that("graph builders require canonical weight labels", {
  x <- matrix(rnorm(80), ncol = 4)

  expect_equal(faissR:::normalize_graph_weight(NULL), "auto")
  expect_equal(faissR:::normalize_graph_weight("adaptive"), "adaptive")
  expect_error(
    faissR:::normalize_graph_weight(c("distance", "binary")),
    "`weight` must be a single value"
  )
  expect_error(knn_graph(x, k = 5L, backend = "cpu", weight = "a"), "weight")
  expect_error(
    knn_graph(x, k = 5L, backend = c("cpu", "cuda")),
    "`backend` must be a single value"
  )
  expect_error(
    knn_graph(x, k = 5L, backend = "cpu", mutual = c(TRUE, FALSE)),
    "`mutual` must be a single TRUE or FALSE"
  )
  expect_error(
    knn_graph(x, k = 5L, backend = "cpu", mutual = "TRUE"),
    "`mutual` must be a single TRUE or FALSE"
  )
  expect_error(
    knn_graph(x, k = c(5L, 6L), backend = "cpu"),
    "`k` must be a single positive integer"
  )
  expect_error(
    knn_graph(x, k = 5.5, backend = "cpu"),
    "`k` must be a single positive integer"
  )
  expect_error(
    knn_graph(x, k = 5L, backend = "cpu", prune = c(0, 0.1)),
    "`prune` must be a non-negative number"
  )
  expect_error(
    knn_graph(x, k = 5L, backend = "cpu", prune = -0.1),
    "`prune` must be a non-negative number"
  )
  expect_error(
    graph_cluster(x, method = "louvain", backend = "cpu", graph_backend = "cpu", k = 5L, weight = "d"),
    "weight"
  )
})

test_that("knn_graph passes method metric and tuning to internal KNN", {
  set.seed(5041)
  x <- matrix(rnorm(120), ncol = 4)

  g <- knn_graph(
    x,
    k = 6L,
    backend = "cpu",
    nn_method = "exact",
    metric = "cor",
    tuning = "off",
    n_threads = 2L
  )
  meta <- attr(g, "faissR_graph")

  expect_s3_class(g, "faissR_graph")
  expect_equal(meta$nn_method, "exact")
  expect_equal(meta$metric, "correlation")
  expect_equal(meta$tuning, "off")
  expect_equal(meta$nn_backend, "cpu")
  expect_equal(meta$requested_backend, "cpu")
  expect_equal(meta$resolved_backend, "cpu")
})

test_that("knn_graph preserves normalized metric transform metadata", {
  set.seed(50412)
  x <- matrix(rnorm(180L), ncol = 3L)

  g <- knn_graph(
    x,
    k = 6L,
    backend = "cpu",
    method = "grid",
    metric = "cosine",
    weight = "distance",
    n_threads = 2L
  )
  meta <- attr(g, "faissR_graph")

  expect_s3_class(g, "faissR_graph")
  expect_match(meta$nn_metric_transform, "normalize_then_euclidean")
  expect_equal(
    meta$nn_distance_transform,
    "normalized_euclidean_squared_over_2_to_1_minus_similarity"
  )

  cl <- graph_cluster(g, method = "louvain", backend = "cpu", n_threads = 2L)
  expect_equal(cl$parameters$nn_metric_transform, meta$nn_metric_transform)
  expect_equal(cl$parameters$nn_distance_transform, meta$nn_distance_transform)
})

test_that("knn_graph stores KNN route metadata for benchmark auditing", {
  knn <- list(
    indices = matrix(c(
      2L, 3L,
      1L, 3L,
      1L, 2L
    ), nrow = 3L, byrow = TRUE),
    distances = matrix(c(
      0.1, 0.2,
      0.1, 0.3,
      0.2, 0.3
    ), nrow = 3L, byrow = TRUE)
  )
  attr(knn, "backend") <- "faiss_hnsw"
  attr(knn, "resolved_backend") <- "faiss_hnsw"
  attr(knn, "requested_backend") <- "cpu"
  attr(knn, "requested_method") <- "hnsw"
  attr(knn, "tuning") <- "auto"
  attr(knn, "metric") <- "euclidean"
  attr(knn, "approximation") <- list(
    strategy = "faiss_IndexHNSWFlat",
    backend = "faiss_hnsw",
    tuning_rule = "small_k_speed"
  )
  g <- knn_graph(
    knn,
    k = 2L,
    backend = "auto",
    method = "auto"
  )
  meta <- attr(g, "faissR_graph")

  expect_s3_class(g, "faissR_graph")
  expect_equal(meta$nn_requested_backend, "cpu")
  expect_equal(meta$nn_requested_method, "hnsw")
  expect_equal(meta$nn_tuning, "auto")
  expect_true(is.list(meta$nn_approximation))
  expect_equal(meta$nn_approximation$backend, "faiss_hnsw")
  expect_equal(meta$nn_approximation$tuning_rule, "small_k_speed")
})

test_that("knn_graph accepts method as an alias for nn_method", {
  set.seed(50411)
  x <- matrix(rnorm(80), ncol = 4)

  g <- knn_graph(
    x,
    k = 5L,
    backend = "cpu",
    method = "exact",
    metric = "euclidean",
    n_threads = 2L
  )
  expect_equal(attr(g, "faissR_graph")$nn_method, "exact")

  g_same <- knn_graph(
    x,
    k = 5L,
    backend = "cpu",
    method = "exact",
    nn_method = "exact",
    metric = "euclidean",
    n_threads = 2L
  )
  expect_equal(attr(g_same, "faissR_graph")$nn_method, "exact")

  expect_error(
    knn_graph(
      x,
      k = 5L,
      backend = "cpu",
      method = "exact",
      nn_method = "hnsw",
      n_threads = 2L
    ),
    "`method` and `nn_method`"
  )
})

test_that("knn_graph rejects removed NN methods", {
  set.seed(5042)
  x <- matrix(rnorm(160), ncol = 4)

  expect_error(knn_graph(
    x,
    k = 6L,
    backend = "auto",
    nn_method = "removed_method",
    metric = "cosine",
    n_threads = 2L
  ), "`method` must be one of")
})

test_that("graph construction rejects implementation backend labels", {
  x <- matrix(rnorm(80), ncol = 4)
  expect_error(
    knn_graph(x, k = 5L, backend = "faiss"),
    "must be one of"
  )
  expect_error(
    knn_graph(x, k = 5L, backend = "cuda_cuvs"),
    "must be one of"
  )
  expect_error(
    graph_cluster(
      x,
      method = "louvain",
      backend = "cu",
      graph_backend = "cpu",
      k = 5L
    ),
    "must be one of"
  )
  expect_error(
    graph_cluster(
      x,
      method = "louvain",
      backend = "cugraph",
      graph_backend = "cpu",
      k = 5L
    ),
    "must be one of"
  )
  expect_error(
    graph_cluster(
      x,
      method = "louvain",
      backend = "cpu",
      graph_backend = "faiss",
      k = 5L
    ),
    "must be one of"
  )
  expect_error(
    faissR:::resolve_knn_graph_backend("cpu_grid"),
    "must be one of"
  )
  expect_equal(faissR:::resolve_graph_cluster_backend("cpu"), "cpu")
  expect_equal(faissR:::resolve_graph_cluster_backend("cuda"), "cuda")
  expect_true(faissR:::resolve_graph_cluster_backend("auto") %in% c("cpu", "cuda"))
  expect_error(
    faissR:::resolve_graph_cluster_backend("cugraph"),
    "must be one of"
  )
})

test_that("graph_cluster owns target cluster count", {
  set.seed(509)
  x <- rbind(
    matrix(rnorm(80, -3, 0.15), ncol = 4),
    matrix(rnorm(80, 0, 0.15), ncol = 4),
    matrix(rnorm(80, 3, 0.15), ncol = 4)
  )

  g <- knn_graph(x, k = 8L, backend = "cpu")
  expect_s3_class(g, "faissR_graph")
  expect_null(attr(g, "faissR_graph")$target_n_clusters)

  cl <- graph_cluster(g, method = "louvain", backend = "cpu", n_clusters = 3L, n_threads = 2L, seed = 1L)
  expect_s3_class(cl, "faissR_graph_cluster")
  expect_equal(cl$target_n_clusters, 3L)
  expect_equal(cl$parameters$n_clusters, 3L)
  expect_equal(cl$parameters$resolution_source, "default")
  expect_equal(cl$parameters$requested_backend, "cpu")
  expect_equal(cl$parameters$resolved_backend, "cpu")
  expect_equal(cl$parameters$graph_backend, attr(g, "faissR_graph")$nn_backend)
  expect_equal(cl$parameters$graph_requested_backend, "cpu")
  expect_equal(cl$parameters$graph_resolved_backend, "cpu")
  expect_equal(cl$parameters$n_vertices, g$n_vertices)
  expect_equal(cl$parameters$n_edges, g$n_edges)
  expect_s3_class(cl$resolution_search, "data.frame")
  expect_true(all(c("candidate", "target_gap", "selected") %in% names(cl$resolution_search)))
  expect_equal(sum(cl$resolution_search$selected), 1L)
  expect_equal(
    cl$resolution_search$target_gap[cl$resolution_search$selected],
    min(cl$resolution_search$target_gap, na.rm = TRUE)
  )
  expect_equal(cl$target_gap, abs(cl$n_communities - 3L))
  expect_equal(cl$parameters$target_gap, cl$target_gap)
  expect_equal(
    cl$resolution_selection$criterion,
    "closest_n_communities_then_highest_modularity"
  )
  expect_true(is.numeric(cl$resolution_selection$candidate_center))
  expect_equal(cl$resolution_selection$n_vertices, g$n_vertices)
  expect_equal(cl$parameters$resolution_selection$selected_candidate, which(cl$resolution_search$selected))
  expect_lte(abs(cl$n_communities - 3L), 1L)

  override <- graph_cluster(
    g,
    method = "louvain",
    backend = "cpu",
    n_threads = 2L,
    seed = 1L,
    n_clusters = 2L
  )
  expect_equal(override$target_n_clusters, 2L)
  expect_equal(override$parameters$n_clusters, 2L)

  auto_resolution <- graph_cluster(
    g,
    method = "louvain",
    backend = "cpu",
    n_threads = 2L,
    seed = 1L,
    resolution = NULL,
    n_clusters = 3L
  )
  expect_equal(auto_resolution$target_n_clusters, 3L)
  expect_equal(auto_resolution$parameters$resolution, 1)
  expect_equal(auto_resolution$parameters$resolution_source, "target_auto")
  expect_s3_class(auto_resolution$resolution_search, "data.frame")

  precomputed <- nn_without_self(x, k = 8L, backend = "cpu", n_threads = 2L)
  g_knn <- knn_graph(precomputed, k = 8L)
  expect_s3_class(g_knn, "faissR_graph")
  expect_equal(class(g_knn), c("faissR_graph", "list"))
  expect_null(attr(g_knn, "faissR_graph")$target_n_clusters)
  leiden <- graph_cluster(g_knn, method = "leiden", backend = "cpu", n_clusters = 3L, n_threads = 2L, seed = 1L)
  expect_equal(leiden$target_n_clusters, 3L)
  expect_equal(leiden$parameters$n_clusters, 3L)

  walk <- graph_cluster(g, method = "random_walking", backend = "cpu", n_threads = 2L)
  expect_s3_class(walk, "faissR_graph_cluster")
  expect_null(walk$target_n_clusters)
  expect_null(walk$parameters$n_clusters)
  expect_error(
    graph_cluster(g, method = "random_walking", backend = "cpu", n_threads = 2L, n_clusters = 3L),
    "n_clusters"
  )
})

test_that("graph cluster-count targets are strict positive whole numbers", {
  set.seed(5091)
  x <- matrix(rnorm(40), ncol = 4)

  expect_error(
    knn_graph(x, k = 4L, backend = "cpu", n_clusters = 2L),
    "unused argument"
  )
  expect_error(
    graph_cluster(
      x,
      method = "louvain",
      backend = "cpu",
      graph_backend = "cpu",
      k = 4L,
      resolution = NULL
    ),
    "resolution"
  )
  null_resolution <- graph_cluster(
    x,
    method = "louvain",
    backend = "cpu",
    graph_backend = "cpu",
    k = 4L,
    resolution = NULL,
    n_clusters = 2L,
    n_threads = 2L
  )
  expect_equal(null_resolution$parameters$resolution_source, "target_auto")
  expect_equal(null_resolution$parameters$n_clusters, 2L)

  expect_error(
    graph_cluster(
      x,
      method = "louvain",
      backend = "cpu",
      graph_backend = "cpu",
      k = 4L,
      n_clusters = 2.5
    ),
    "positive integer"
  )
  expect_error(
    graph_cluster(
      x,
      method = "leiden",
      backend = "cpu",
      graph_backend = "cpu",
      k = 4L,
      n_clusters = nrow(x) + 1L
    ),
    "larger than the number of graph vertices"
  )
})

test_that("graph_cluster resolution changes Louvain partitions on separable graphs", {
  set.seed(5093)
  x <- rbind(
    matrix(rnorm(80, -4, 0.15), ncol = 4),
    matrix(rnorm(80, -1, 0.15), ncol = 4),
    matrix(rnorm(80, 1, 0.15), ncol = 4),
    matrix(rnorm(80, 4, 0.15), ncol = 4)
  )
  g <- knn_graph(x, k = 8L, backend = "cpu")

  low <- graph_cluster(g, method = "louvain", backend = "cpu", resolution = 0.05, n_threads = 2L, seed = 1L)
  high <- graph_cluster(g, method = "louvain", backend = "cpu", resolution = 16, n_threads = 2L, seed = 1L)

  expect_lt(low$n_communities, high$n_communities)
  expect_equal(low$parameters$resolution, 0.05)
  expect_equal(high$parameters$resolution, 16)
})

test_that("target cluster resolution candidates are bounded and deterministic", {
  candidates <- faissR:::graph_resolution_candidates(1, 3L)

  expect_equal(candidates, sort(candidates))
  expect_true(1 %in% candidates)
  expect_equal(min(candidates), 1 / 16)
  expect_equal(max(candidates), 16)
  expect_equal(length(candidates), 17L)
  expect_equal(faissR:::graph_resolution_candidates(0.5, NULL), 0.5)

  shape_center <- faissR:::graph_resolution_center(
    resolution = 1,
    n_clusters = 3L,
    n_vertices = 120L
  )
  shaped <- faissR:::graph_resolution_candidates(1, 3L, n_vertices = 120L)
  expect_equal(shape_center, sqrt(3 / sqrt(120)))
  expect_equal(shaped, sort(shaped))
  expect_true(1 %in% shaped)
  expect_true(shape_center %in% shaped)
  expect_lte(min(shaped), shape_center / 16)
  expect_gte(max(shaped), shape_center * 16)
  expect_lte(length(shaped), 18L)
  expect_equal(faissR:::graph_resolution_grid_exponents(120L), seq(-4, 4, by = 0.5))
  expect_equal(faissR:::graph_resolution_grid_exponents(10000L), seq(-3, 3, by = 0.5))
  expect_equal(faissR:::graph_resolution_grid_exponents(50000L), seq(-2, 2, by = 0.5))
  medium <- faissR:::graph_resolution_candidates(1, 10L, n_vertices = 10000L)
  large <- faissR:::graph_resolution_candidates(1, 10L, n_vertices = 70000L)
  expect_true(1 %in% medium)
  expect_true(1 %in% large)
  expect_lt(length(large), length(shaped))
  expect_lte(length(large), 10L)
  expect_equal(
    faissR:::graph_resolution_center(1, 3L, n_vertices = NULL),
    1
  )
})

test_that("graph runtime integer controls reject fractional values", {
  set.seed(5092)
  x <- matrix(rnorm(40), ncol = 4)

  expect_error(
    graph_cluster(x, method = "louvain", backend = "cpu", graph_backend = "cpu", k = 4.5),
    "`k` must be a single positive integer"
  )
  expect_error(
    graph_cluster(x, method = "louvain", backend = "cpu", graph_backend = "cpu", k = 4L, n_runs = 1.5),
    "`n_runs` must be a single positive integer"
  )
  expect_error(
    graph_cluster(x, method = "leiden", backend = "cpu", graph_backend = "cpu", k = 4L, n_iterations = 2.5),
    "`n_iterations` must be a single positive integer"
  )
  expect_error(
    graph_cluster(x, method = "random_walking", backend = "cpu", graph_backend = "cpu", k = 4L, steps = 2.5),
    "`steps` must be a single positive integer"
  )
  expect_error(
    graph_cluster(x, method = "louvain", backend = "cpu", graph_backend = "cpu", k = 4L, seed = 1.5),
    "`seed` must be a single positive integer"
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
  expect_true("Blondel et al. (2008) for Louvain modularity optimization" %in% louvain$sources)
  expect_false("Pons and Latapy (2006) for random-walk walktrap clustering" %in% louvain$sources)
})

test_that("graph_cluster requires canonical clustering method labels", {
  set.seed(50511)
  x <- matrix(rnorm(80), ncol = 4)

  expect_error(
    graph_cluster(x, method = "l", backend = "cpu", graph_backend = "cpu", k = 5L),
    "method"
  )
  expect_error(
    graph_cluster(x, method = "walktrap", backend = "cpu", graph_backend = "cpu", k = 5L),
    "method"
  )
  expect_equal(faissR:::normalize_graph_cluster_method(NULL), "random_walking")
  expect_equal(faissR:::normalize_graph_cluster_method("leiden"), "leiden")
  expect_error(
    faissR:::normalize_graph_cluster_method(c("leiden", "louvain")),
    "`method` must be a single value"
  )
})

test_that("graph_cluster requires canonical objective-function labels", {
  x <- matrix(rnorm(80), ncol = 4)

  expect_equal(faissR:::normalize_graph_objective_function(NULL), "modularity")
  expect_equal(faissR:::normalize_graph_objective_function("CPM"), "CPM")
  expect_error(
    faissR:::normalize_graph_objective_function(c("CPM", "modularity")),
    "`objective_function` must be a single value"
  )
  expect_error(
    graph_cluster(
      x,
      method = "leiden",
      backend = "cpu",
      graph_backend = "cpu",
      k = 5L,
      objective_function = "C"
    ),
    "objective_function"
  )
})

test_that("graph_cluster validates positive integer controls strictly", {
  x <- matrix(rnorm(80), ncol = 4)

  expect_equal(faissR:::normalize_graph_positive_int(5L, "k"), 5L)
  expect_error(faissR:::normalize_graph_positive_int(c(5L, 6L), "k"), "single positive integer")
  expect_error(faissR:::normalize_graph_positive_int("many", "n_runs"), "single positive integer")
  expect_error(
    graph_cluster(x, method = "louvain", backend = "cpu", graph_backend = "cpu", k = c(5L, 6L)),
    "`k` must be a single positive integer"
  )
  expect_error(
    graph_cluster(x, method = "louvain", backend = "cpu", graph_backend = "cpu", k = 5L, n_runs = "many"),
    "`n_runs` must be a single positive integer"
  )
  expect_error(
    graph_cluster(x, method = "louvain", backend = "cpu", graph_backend = "cpu", k = 5L, n_iterations = 0L),
    "`n_iterations` must be a single positive integer"
  )
  expect_error(
    graph_cluster(x, method = "random_walking", backend = "cpu", graph_backend = "cpu", k = 5L, steps = c(3L, 4L)),
    "`steps` must be a single positive integer"
  )
  expect_error(
    graph_cluster(x, method = "louvain", backend = "cpu", graph_backend = "cpu", k = 5L, seed = NA_integer_),
    "`seed` must be a single positive integer"
  )
})

test_that("graph_cluster passes method metric and tuning to internal KNN", {
  set.seed(5052)
  x <- rbind(
    matrix(rnorm(80, -2, 0.2), ncol = 4),
    matrix(rnorm(80, 2, 0.2), ncol = 4)
  )

  cl <- graph_cluster(
    x,
    method = "louvain",
    backend = "cpu",
    graph_backend = "cpu",
    graph_method = "exact",
    metric = "correlation",
    tuning = "off",
    k = 6L,
    n_threads = 2L
  )

  expect_s3_class(cl, "faissR_graph_cluster")
  expect_equal(cl$parameters$graph_method, "exact")
  expect_equal(cl$parameters$metric, "correlation")
  expect_equal(cl$parameters$tuning, "off")
  expect_equal(cl$parameters$graph_backend, "cpu")
})

test_that("graph_cluster preserves internal KNN route metadata when building a graph", {
  set.seed(50525)
  x <- rbind(
    matrix(rnorm(80, -2, 0.2), ncol = 4),
    matrix(rnorm(80, 2, 0.2), ncol = 4)
  )

  cl <- graph_cluster(
    x,
    method = "louvain",
    backend = "cpu",
    graph_backend = "cpu",
    graph_method = "auto",
    metric = "cosine",
    k = 6L,
    n_threads = 2L
  )

  expect_s3_class(cl, "faissR_graph_cluster")
  expect_equal(cl$parameters$nn_requested_backend, "cpu")
  expect_equal(cl$parameters$nn_requested_method, "auto")
  expect_equal(cl$parameters$nn_tuning, "auto")
  expect_type(cl$parameters$nn_auto_selection, "list")
  expect_true(cl$parameters$nn_auto_selection$explicit_backend)
  expect_false(cl$parameters$nn_auto_selection$explicit_method)
  expect_equal(cl$parameters$nn_auto_selection$backend_decision, "explicit_cpu")
  expect_equal(cl$parameters$nn_auto_selection$metric, "cosine")
})

test_that("graph_cluster lets graph_backend auto resolve CPU-only NN methods", {
  set.seed(5053)
  x <- rbind(
    matrix(rnorm(80, -2, 0.3), ncol = 4),
    matrix(rnorm(80, 2, 0.3), ncol = 4)
  )

  expect_error(graph_cluster(
    x,
    method = "louvain",
    backend = "cpu",
    graph_backend = "auto",
    graph_method = "removed_method",
    metric = "cosine",
    k = 6L,
    n_threads = 2L
  ), "`method` must be one of")
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
  expect_equal(auto$parameters$requested_backend, "auto")
  expect_equal(auto$parameters$resolved_backend, "cpu")

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
  expect_equal(cl$parameters$target_gap, cl$target_gap)
  expect_equal(
    cl$parameters$resolution_selection$criterion,
    "closest_n_communities_then_highest_modularity"
  )
  expect_equal(cl$parameters$requested_backend, "cpu")
  expect_equal(cl$parameters$resolved_backend, "cpu")
  expect_equal(cl$parameters$n_vertices, length(cl$membership))
  expect_equal(cl$parameters$n_edges, cl$graph$n_edges)
  expect_lte(abs(cl$n_communities - 3L), 1L)
  expect_equal(sum(cl$resolution_search$selected), 1L)
  expect_equal(
    cl$resolution_search$target_gap[cl$resolution_search$selected],
    min(cl$resolution_search$target_gap, na.rm = TRUE)
  )
  expect_equal(
    cl$resolution_search$candidate[cl$resolution_search$selected],
    cl$resolution_selection$selected_candidate
  )
  expect_equal(
    cl$resolution_search$target_gap[cl$resolution_search$selected],
    cl$target_gap
  )
  printed <- capture.output(print(cl))
  expect_true(any(grepl("target communities: 3", printed, fixed = TRUE)))
  expect_true(any(grepl("selected resolution:", printed, fixed = TRUE)))

  implicit <- graph_cluster(
    x,
    backend = "cpu",
    graph_backend = "cpu",
    k = 8L,
    n_clusters = 3L,
    n_threads = 2L,
    seed = 1L
  )
  expect_s3_class(implicit, "faissR_graph_cluster")
  expect_equal(implicit$method, "louvain")
  expect_equal(implicit$target_n_clusters, 3L)
  expect_equal(implicit$parameters$n_clusters, 3L)
  expect_equal(implicit$resolution_selection$criterion, "closest_n_communities_then_highest_modularity")

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
