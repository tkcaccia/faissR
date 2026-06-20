#' Build an igraph nearest-neighbour graph
#'
#' `knn_graph()` turns a data matrix, a precomputed [nn()] result, or a
#' `fastEmbedR_embedding` object returned by [opentsne()] or [umap()] into an
#' undirected `igraph` graph. This keeps the workflow simple: pass the [nn()]
#' result for a graph on the original data space, or pass the [opentsne()] /
#' [umap()] result for a graph on the visible embedding layout.
#'
#' @param data Numeric matrix/data frame, a KNN object returned by [nn()], or a
#'   `fastEmbedR_embedding` object returned by [opentsne()] or [umap()].
#' @param knn Optional precomputed KNN object returned by [nn()]. If supplied,
#'   `data` is ignored for neighbour search.
#' @param k Number of non-self neighbours used in the graph.
#' @param backend KNN backend passed to [nn()] when `knn` is not supplied.
#'   Use `"auto"` for the fastest graph-KNN default: cuVS NN-descent on CUDA
#'   when available, then FAISS NN-descent, then RcppHNSW, then exact CPU.
#' @param weight Graph weighting. `"auto"` uses SNN/Jaccard weights for input
#'   space and distance weights for embedding space. `"snn"` builds
#'   full shared-nearest-neighbour Jaccard weights between all rows sharing at
#'   least one neighbour. `"adaptive"` uses
#'   `exp(-d_ij^2 / (sigma_i * sigma_j))`, where each `sigma` is the local
#'   neighbourhood radius. `"distance"` uses `1 / (1 + distance)`. `"binary"`
#'   gives every edge weight 1.
#' @param mutual If `TRUE`, keep only reciprocal nearest-neighbour edges. This
#'   can sharpen cluster boundaries on embedding-layout graphs.
#' @param prune Drop edges with weight less than or equal to this value.
#' @param n_threads CPU threads passed to [nn()] when KNN is computed here.
#' @return An undirected `igraph` graph with edge attribute `weight`.
#' @examples
#' x <- scale(as.matrix(iris[, 1:4]))
#' if (requireNamespace("igraph", quietly = TRUE)) {
#'   g <- knn_graph(x, k = 15, backend = "cpu")
#'   cl <- igraph::cluster_louvain(g, weights = igraph::E(g)$weight)
#'   table(igraph::membership(cl))
#' }
#' @export
knn_graph <- function(data,
                      knn = NULL,
                      k = 50L,
                      backend = "auto",
                      weight = c("auto", "snn", "adaptive", "distance", "binary"),
                      mutual = FALSE,
                      prune = 0,
                      n_threads = NULL) {
  if (!requireNamespace("igraph", quietly = TRUE)) {
    stop(
      "`knn_graph()` requires the optional package `igraph`. ",
      "Install it with `install.packages(\"igraph\")`.",
      call. = FALSE
    )
  }
  weight <- match.arg(weight)
  k <- as.integer(k)
  if (length(k) != 1L || is.na(k) || !is.finite(k) || k < 1L) {
    stop("`k` must be a positive integer.", call. = FALSE)
  }
  mutual <- isTRUE(mutual)
  prune <- suppressWarnings(as.numeric(prune))
  if (length(prune) != 1L || is.na(prune) || !is.finite(prune) || prune < 0) {
    stop("`prune` must be a non-negative number.", call. = FALSE)
  }

  input_backend <- NA_character_
  graph_space <- "input"
  input_method <- NA_character_
  if (is.null(knn)) {
    if (missing(data) || is.null(data)) {
      stop("Provide either `data` or `knn`.", call. = FALSE)
    }
    if (is_knn_input(data)) {
      knn <- data
      graph_space <- "input"
      input_method <- "nn"
    } else if (is_fastembedr_embedding(data)) {
      input_method <- as.character(data$method %||% "embedding")[1L]
      if (!is.matrix(data$layout)) {
        stop("Embedding objects passed to `knn_graph()` must contain a matrix `layout`.", call. = FALSE)
      }
      graph_space <- "embedding"
      graph_backend <- resolve_knn_graph_backend(as.character(backend)[1L])
      knn <- nn_without_self(
        data$layout,
        k = k,
        backend = graph_backend,
        n_threads = n_threads
      )
      input_backend <- attr(knn, "backend") %||% graph_backend
    } else {
      graph_backend <- resolve_knn_graph_backend(as.character(backend)[1L])
      knn <- nn_without_self(
        data,
        k = k,
        backend = graph_backend,
        n_threads = n_threads
      )
      input_backend <- attr(knn, "backend") %||% graph_backend
    }
  }

  knn_input <- coerce_knn_input(knn, arg_name = "knn")
  if (identical(weight, "auto")) {
    weight <- if (identical(graph_space, "embedding")) "distance" else "snn"
  }
  if (!is.na(knn_input$input_backend)) input_backend <- knn_input$input_backend
  if (knn_input$n_neighbors < k) {
    k <- knn_input$n_neighbors
  }
  cols <- seq_len(k)
  indices <- knn_input$indices[, cols, drop = FALSE]
  distances <- knn_input$distances[, cols, drop = FALSE]

  edges <- knn_graph_edges_cpp(indices, distances, weight, prune, mutual)
  graph <- igraph::make_empty_graph(n = edges$n_vertices, directed = FALSE)
  if (length(edges$from) > 0L) {
    graph <- igraph::add_edges(graph, as.vector(rbind(edges$from, edges$to)))
    igraph::E(graph)$weight <- edges$weight
  }
  attr(graph, "fastEmbedR_graph") <- list(
    k = as.integer(k),
    space = graph_space,
    weight = weight,
    mutual = mutual,
    prune = prune,
    nn_backend = input_backend,
    input_method = input_method,
    n_vertices = igraph::vcount(graph),
    n_edges = igraph::ecount(graph)
  )
  graph
}

is_fastembedr_embedding <- function(x) {
  inherits(x, "fastEmbedR_embedding") ||
    (is.list(x) && is.matrix(x$layout) && !is.null(x$method))
}

resolve_knn_graph_backend <- function(backend) {
  backend <- as.character(backend)[1L]
  if (is.na(backend) || !nzchar(backend)) backend <- "auto"
  if (!identical(backend, "auto")) return(backend)
  if (isTRUE(cuvs_available())) return("cuda_cuvs_nndescent")
  if (isTRUE(faiss_available())) return("faiss_nndescent")
  if (isTRUE(requireNamespace("RcppHNSW", quietly = TRUE))) return("hnsw")
  "cpu"
}

#' Cluster a nearest-neighbour graph
#'
#' `graph_cluster()` runs community detection on an `igraph` graph or first
#' builds one with [knn_graph()]. The nearest-neighbour graph can therefore use
#' FAISS/cuVS backends, while CPU community detection uses `igraph` algorithms.
#' The CUDA backend is reserved for RAPIDS cuGraph Louvain, Leiden, and random
#' walk implementations and fails clearly when faissR was not built with a
#' cuGraph binding.
#'
#' @param graph An `igraph` graph, a matrix/data frame accepted by [knn_graph()],
#'   a KNN object returned by [nn()], or an embedding object accepted by
#'   [knn_graph()].
#' @param method One of `"random_walking"`, `"louvain"`, or `"leiden"`.
#'   `"random_walking"` uses igraph's walktrap/random-walk community method.
#' @param backend Community-detection backend. `"cpu"` uses `igraph`.
#'   `"cuda"` is reserved for RAPIDS cuGraph and currently fails clearly unless
#'   a future build links a CUDA graph backend.
#' @param k Number of neighbours when `graph` is not already an `igraph` graph.
#' @param graph_backend Backend passed to [knn_graph()] for neighbour search.
#' @param weight Weighting passed to [knn_graph()] when a graph must be built.
#' @param mutual,prune Graph-construction options passed to [knn_graph()].
#' @param n_threads CPU threads. Used by [knn_graph()] and by repeated CPU runs
#'   when `n_runs > 1` on Unix-like platforms.
#' @param n_runs Number of repeated CPU community-detection runs. The result with
#'   the largest modularity is returned. Values greater than one can use several
#'   CPU cores.
#' @param resolution Modularity resolution for Louvain/Leiden.
#' @param objective_function Leiden objective, `"modularity"` or `"CPM"`.
#' @param n_iterations Leiden iteration count.
#' @param steps Walk length used by igraph's walktrap implementation.
#' @param seed Optional seed for reproducible repeated CPU runs.
#' @param ... Reserved for future backend options.
#' @return A `faissR_graph_cluster` list with membership, modularity, method,
#'   backend, graph, and the backend-specific community object.
#' @references
#' Blondel VD, Guillaume JL, Lambiotte R, Lefebvre E. Fast unfolding of
#' communities in large networks. Journal of Statistical Mechanics: Theory and
#' Experiment. 2008;2008(10):P10008.
#'
#' Pons P, Latapy M. Computing communities in large networks using random walks.
#' Journal of Graph Algorithms and Applications. 2006;10(2):191-218.
#'
#' Traag VA, Waltman L, van Eck NJ. From Louvain to Leiden: guaranteeing
#' well-connected communities. Scientific Reports. 2019;9:5233.
#'
#' Sahu S. GVE-Leiden: Fast Leiden Algorithm for Community Detection in Shared
#' Memory Setting. arXiv:2312.13936.
#'
#' Sahu S. Heuristic-based Dynamic Leiden Algorithm for Efficient Tracking of
#' Communities on Evolving Graphs. arXiv:2410.15451.
#'
#' Kapralov M, Lattanzi S, Nouri N, Tardos J. Efficient and Local Parallel
#' Random Walks. arXiv:2112.00655.
#'
#' CPU implementations are provided through the igraph R package. The CUDA route
#' is designed for a future RAPIDS cuGraph binding; cuGraph provides GPU Louvain,
#' Leiden, and random-walk algorithms in its CUDA/C++/Python stack.
#' @export
graph_cluster <- function(graph,
                           method = c("random_walking", "louvain", "leiden"),
                           backend = c("cpu", "cuda"),
                           k = 50L,
                           graph_backend = "auto",
                           weight = c("auto", "snn", "adaptive", "distance", "binary"),
                           mutual = FALSE,
                           prune = 0,
                           n_threads = NULL,
                           n_runs = 1L,
                           resolution = 1,
                           objective_function = c("modularity", "CPM"),
                           n_iterations = 2L,
                           steps = 4L,
                           seed = NULL,
                           ...) {
  method <- match.arg(method)
  backend <- match.arg(backend)
  weight <- match.arg(weight)
  objective_function <- match.arg(objective_function)
  if (!requireNamespace("igraph", quietly = TRUE)) {
    stop(
      "`graph_cluster()` requires the optional package `igraph`. ",
      "Install it with `install.packages(\"igraph\")`.",
      call. = FALSE
    )
  }
  if (identical(backend, "cuda")) {
    stop(
      "`backend = \"cuda\"` for graph clustering requires a RAPIDS cuGraph ",
      "CUDA binding. This faissR build is not linked to cuGraph. ",
      "Use `backend = \"cpu\"`, or build a future faissR CUDA graph backend ",
      "against cuGraph for GPU Louvain, Leiden, and random-walk algorithms.",
      call. = FALSE
    )
  }
  n_threads <- normalize_nn_threads(n_threads)
  n_runs <- suppressWarnings(as.integer(n_runs))
  if (length(n_runs) != 1L || is.na(n_runs) || !is.finite(n_runs) || n_runs < 1L) {
    stop("`n_runs` must be a positive integer.", call. = FALSE)
  }
  resolution <- suppressWarnings(as.numeric(resolution))
  if (length(resolution) != 1L || is.na(resolution) || !is.finite(resolution) || resolution <= 0) {
    stop("`resolution` must be a positive number.", call. = FALSE)
  }
  n_iterations <- normalize_positive_int(n_iterations, 2L)
  steps <- normalize_positive_int(steps, 4L)

  g <- if (inherits(graph, "igraph")) {
    graph
  } else {
    knn_graph(
      graph,
      k = k,
      backend = graph_backend,
      weight = weight,
      mutual = mutual,
      prune = prune,
      n_threads = n_threads
    )
  }
  if (igraph::is_directed(g)) {
    g <- igraph::as_undirected(g, mode = "collapse", edge.attr.comb = list(weight = "sum", "ignore"))
  }
  weights <- graph_cluster_weights(g)
  run_ids <- seq_len(n_runs)
  run_one <- function(run_id) {
    if (!is.null(seed)) set.seed(as.integer(seed) + as.integer(run_id) - 1L)
    graph_cluster_cpu_once(
      g = g,
      method = method,
      weights = weights,
      resolution = resolution,
      objective_function = objective_function,
      n_iterations = n_iterations,
      steps = steps
    )
  }
  results <- if (n_runs > 1L && n_threads > 1L && .Platform$OS.type != "windows") {
    parallel::mclapply(run_ids, run_one, mc.cores = min(n_threads, n_runs))
  } else {
    lapply(run_ids, run_one)
  }
  modularity <- vapply(results, function(x) x$modularity, numeric(1))
  best <- results[[which.max(modularity)]]
  out <- list(
    membership = best$membership,
    modularity = best$modularity,
    communities = best$communities,
    graph = g,
    method = method,
    backend = backend,
    n_runs = as.integer(n_runs),
    selected_run = as.integer(which.max(modularity)),
    all_modularity = modularity,
    parameters = list(
      resolution = resolution,
      objective_function = objective_function,
      n_iterations = as.integer(n_iterations),
      steps = as.integer(steps),
      n_threads = as.integer(n_threads)
    ),
    sources = graph_cluster_sources(method, backend)
  )
  class(out) <- "faissR_graph_cluster"
  out
}

graph_cluster_cpu_once <- function(g,
                                   method,
                                   weights,
                                   resolution,
                                   objective_function,
                                   n_iterations,
                                   steps) {
  communities <- switch(
    method,
    random_walking = igraph::cluster_walktrap(
      g,
      weights = weights,
      steps = as.integer(steps),
      merges = TRUE,
      modularity = TRUE,
      membership = TRUE
    ),
    louvain = igraph::cluster_louvain(
      g,
      weights = weights,
      resolution = resolution
    ),
    leiden = {
      if (!"cluster_leiden" %in% getNamespaceExports("igraph")) {
        stop("`method = \"leiden\"` requires an igraph version with `cluster_leiden()`.", call. = FALSE)
      }
      igraph::cluster_leiden(
        g,
        objective_function = objective_function,
        weights = weights,
        resolution = resolution,
        n_iterations = as.integer(n_iterations)
      )
    }
  )
  membership <- as.integer(igraph::membership(communities))
  list(
    communities = communities,
    membership = membership,
    modularity = graph_cluster_modularity(g, membership, weights, resolution)
  )
}

graph_cluster_weights <- function(g) {
  if ("weight" %in% igraph::edge_attr_names(g)) {
    weights <- igraph::E(g)$weight
    if (!is.null(weights) && all(is.finite(weights))) return(as.numeric(weights))
  }
  NULL
}

graph_cluster_modularity <- function(g, membership, weights, resolution) {
  value <- tryCatch(
    igraph::modularity(g, membership = membership, weights = weights, resolution = resolution),
    error = function(e) NA_real_
  )
  as.numeric(value)
}

graph_cluster_sources <- function(method, backend) {
  base <- c(
    "igraph R package for CPU community detection",
    "Blondel et al. (2008) for Louvain modularity optimization",
    "Traag et al. (2019) for Leiden community detection",
    "Pons and Latapy (2006) for random-walk walktrap clustering"
  )
  if (identical(method, "leiden")) {
    base <- c(
      base,
      "Sahu (2024), GVE-Leiden/OpenMP, as multicore Leiden implementation inspiration",
      "Sahu (2024), heuristic dynamic Leiden, as dynamic Leiden inspiration"
    )
  }
  if (identical(method, "random_walking")) {
    base <- c(base, "Kapralov et al. (2021) for local parallel random-walk motivation")
  }
  if (identical(backend, "cuda")) {
    base <- c(base, "RAPIDS cuGraph for CUDA Louvain, Leiden, and random-walk algorithms")
  }
  unique(base)
}

#' @export
print.faissR_graph_cluster <- function(x, ...) {
  cat("faissR graph clustering\n")
  cat("  method: ", x$method, "\n", sep = "")
  cat("  backend: ", x$backend, "\n", sep = "")
  cat("  vertices: ", length(x$membership), "\n", sep = "")
  cat("  communities: ", length(unique(x$membership)), "\n", sep = "")
  cat("  modularity: ", format(x$modularity, digits = 4), "\n", sep = "")
  invisible(x)
}
