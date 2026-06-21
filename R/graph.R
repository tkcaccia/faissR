#' Build a native nearest-neighbour graph
#'
#' `knn_graph()` turns a data matrix, a precomputed [nn()] result, or a
#' `fastEmbedR_embedding` object into a weighted nearest-neighbour graph. It returns a native
#' `faissR_graph` edge-list object and does not require `igraph`.
#'
#' @param data Numeric matrix/data frame, a KNN object returned by [nn()], or a
#'   `fastEmbedR_embedding` object returned by [opentsne()] or [umap()].
#' @param knn Optional precomputed KNN object returned by [nn()]. If supplied,
#'   `data` is ignored for neighbour search.
#' @param k Number of non-self neighbours used in the graph.
#' @param backend Device backend passed to [nn_without_self()] when `knn` is
#'   not supplied: `"auto"`, `"cpu"`, or `"cuda"`.
#' @param weight Graph weighting. `"auto"` uses SNN/Jaccard weights for input
#'   space and distance weights for embedding space. `"snn"` builds full
#'   shared-nearest-neighbour Jaccard weights between all rows sharing at least
#'   one neighbour. `"adaptive"` uses `exp(-d_ij^2 / (sigma_i * sigma_j))`.
#'   `"distance"` uses `1 / (1 + distance)`. `"binary"` gives every edge weight 1.
#' @param mutual If `TRUE`, keep only reciprocal nearest-neighbour edges.
#' @param prune Drop edges with weight less than or equal to this value.
#' @param n_threads CPU threads passed to [nn()] when KNN is computed here.
#' @return A native `faissR_graph` edge-list object.
#' @examples
#' x <- scale(as.matrix(iris[, 1:4]))
#' g <- knn_graph(x, k = 15, backend = "cpu")
#' cl <- graph_cluster(g, method = "louvain", backend = "cpu")
#' table(cl$membership)
#' @export
knn_graph <- function(data,
                      knn = NULL,
                      k = 50L,
                      backend = c("auto", "cpu", "cuda"),
                      weight = c("auto", "snn", "adaptive", "distance", "binary"),
                      mutual = FALSE,
                      prune = 0,
                      n_threads = NULL) {
  backend <- as.character(backend)[1L]
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
      knn <- nn_without_self(data$layout, k = k, backend = graph_backend, n_threads = n_threads)
      input_backend <- attr(knn, "backend") %||% graph_backend
    } else {
      graph_backend <- resolve_knn_graph_backend(as.character(backend)[1L])
      knn <- nn_without_self(data, k = k, backend = graph_backend, n_threads = n_threads)
      input_backend <- attr(knn, "backend") %||% graph_backend
    }
  }

  knn_input <- coerce_knn_input(knn, arg_name = "knn")
  if (identical(weight, "auto")) {
    weight <- if (identical(graph_space, "embedding")) "distance" else "snn"
  }
  if (!is.na(knn_input$input_backend)) input_backend <- knn_input$input_backend
  if (knn_input$n_neighbors < k) k <- knn_input$n_neighbors
  cols <- seq_len(k)
  edges <- knn_graph_edges_cpp(
    knn_input$indices[, cols, drop = FALSE],
    knn_input$distances[, cols, drop = FALSE],
    weight,
    prune,
    mutual
  )
  metadata <- list(
    k = as.integer(k),
    space = graph_space,
    weight = weight,
    mutual = mutual,
    prune = prune,
    nn_backend = input_backend,
    input_method = input_method,
    n_vertices = edges$n_vertices,
    n_edges = edges$n_edges
  )
  attr(edges, "faissR_graph") <- metadata
  class(edges) <- c("faissR_graph", "list")
  edges
}

is_fastembedr_embedding <- function(x) {
  inherits(x, "fastEmbedR_embedding") ||
    (is.list(x) && is.matrix(x$layout) && !is.null(x$method))
}

resolve_knn_graph_backend <- function(backend) {
  backend_label <- as.character(backend)[1L]
  if (!tolower(backend_label) %in% c("auto", "cpu", "cuda")) {
    return(backend_label)
  }
  normalize_public_compute_backend(backend)
}

resolve_graph_cluster_backend <- function(backend) {
  backend <- as.character(backend)[1L]
  if (is.na(backend) || !nzchar(backend)) backend <- "auto"
  backend <- tolower(backend)
  if (!backend %in% c("auto", "cpu", "cuda")) {
    stop("`backend` must be one of \"auto\", \"cpu\", or \"cuda\".", call. = FALSE)
  }
  if (identical(backend, "auto")) {
    if (isTRUE(cuda_available()) && isTRUE(cugraph_available())) return("cuda")
    return("cpu")
  }
  backend
}

#' Cluster a nearest-neighbour graph without igraph
#'
#' `graph_cluster()` runs native faissR community detection on a KNN graph.
#' The graph can be supplied as a precomputed [nn()] result or built from a
#' matrix/data frame using any faissR nearest-neighbour backend, including FAISS
#' and cuVS KNN backends. The CPU clustering backend is implemented in C++ and
#' uses OpenMP when available. The CUDA clustering backend uses native RAPIDS libcugraph for
#' `method = "louvain"` and `method = "leiden"` when faissR is built
#' against libcugraph. `method = "random_walking"` currently remains CPU-only.
#' CUDA graph clustering never calls Python and never silently falls back to CPU.
#'
#' @param graph Numeric matrix/data frame, a KNN object returned by [nn()], or an
#'   embedding object with a matrix `layout`.
#' @param method One of `"random_walking"`, `"louvain"`, or `"leiden"`.
#'   `"random_walking"` uses a native random-walk label propagation pass inspired
#'   by walktrap/random-walk clustering. `"louvain"` uses native modularity local
#'   moving. `"leiden"` adds a native refinement pass that splits disconnected
#'   communities after local moving.
#' @param backend Community-detection backend. `"auto"` uses CUDA when
#'   libcugraph is available and CPU otherwise. `"cpu"` uses native C++/OpenMP.
#'   `"cuda"` uses native RAPIDS libcugraph for Louvain and Leiden when
#'   libcugraph was detected at build time; random-walking is CPU-only.
#' @param k Number of neighbours when `graph` is not already a KNN object.
#' @param graph_backend Backend passed to [nn_without_self()] for neighbour
#'   search when `graph` is a matrix or embedding.
#' @param weight KNN graph weighting. See [knn_graph()].
#' @param mutual,prune Graph-construction options.
#' @param n_threads CPU threads for KNN construction and native CPU clustering.
#' @param n_runs Number of independent native runs. The best modularity is kept.
#' @param resolution Modularity resolution for Louvain/Leiden-style scoring.
#' @param objective_function Reserved for Leiden-compatible APIs.
#' @param n_iterations Native clustering iterations.
#' @param steps Random-walk propagation depth.
#' @param seed Optional seed used to make repeated runs reproducible.
#' @param ... Reserved for future backend options.
#' @return A `faissR_graph_cluster` list with membership, modularity, method,
#'   backend, graph edge list, parameters, and source acknowledgements.
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
#' CPU implementations are native faissR C++/OpenMP code. CUDA Louvain and
#' Leiden use RAPIDS libcugraph when available at build time; RAPIDS cuGraph
#' also provides GPU random-walk primitives that may support a future native
#' random-walking adapter.
#' @export
graph_cluster <- function(graph,
                          method = c("random_walking", "louvain", "leiden"),
                          backend = c("auto", "cpu", "cuda"),
                          k = 50L,
                          graph_backend = "auto",
                          weight = c("auto", "snn", "adaptive", "distance", "binary"),
                          mutual = FALSE,
                          prune = 0,
                          n_threads = NULL,
                          n_runs = 1L,
                          resolution = 1,
                          objective_function = c("modularity", "CPM"),
                          n_iterations = 10L,
                          steps = 4L,
                          seed = NULL,
                          ...) {
  method <- match.arg(method)
  backend <- resolve_graph_cluster_backend(match.arg(backend))
  weight <- match.arg(weight)
  objective_function <- match.arg(objective_function)
  n_threads <- normalize_nn_threads(n_threads)
  k <- normalize_positive_int(k, 50L)
  n_runs <- normalize_positive_int(n_runs, 1L)
  n_iterations <- normalize_positive_int(n_iterations, 10L)
  steps <- normalize_positive_int(steps, 4L)
  resolution <- suppressWarnings(as.numeric(resolution))
  if (length(resolution) != 1L || is.na(resolution) || !is.finite(resolution) || resolution <= 0) {
    stop("`resolution` must be a positive number.", call. = FALSE)
  }
  prune <- suppressWarnings(as.numeric(prune))
  if (length(prune) != 1L || is.na(prune) || !is.finite(prune) || prune < 0) {
    stop("`prune` must be a non-negative number.", call. = FALSE)
  }
  seed <- if (is.null(seed)) 1L else normalize_positive_int(seed, 1L)


  graph_space <- "input"
  input_method <- NA_character_
  input_backend <- NA_character_
  if (inherits(graph, "faissR_graph")) {
    ans <- graph_cluster_edges_cpp(
      graph,
      method = method,
      backend = backend,
      n_threads = n_threads,
      n_runs = n_runs,
      resolution = resolution,
      n_iterations = n_iterations,
      steps = steps,
      seed = seed
    )
    meta <- attr(graph, "faissR_graph") %||% list()
    ans$parameters <- c(
      meta,
      list(
        resolution = resolution,
        objective_function = objective_function,
        n_iterations = as.integer(n_iterations),
        steps = as.integer(steps),
        n_threads = as.integer(n_threads)
      )
    )
    ans$sources <- graph_cluster_sources(method, backend)
    class(ans) <- "faissR_graph_cluster"
    return(ans)
  }
  if (is_knn_input(graph)) {
    knn <- graph
    input_method <- "nn"
  } else if (is_fastembedr_embedding(graph)) {
    if (!is.matrix(graph$layout)) {
      stop("Embedding objects passed to `graph_cluster()` must contain a matrix `layout`.", call. = FALSE)
    }
    graph_space <- "embedding"
    input_method <- as.character(graph$method %||% "embedding")[1L]
    resolved <- resolve_knn_graph_backend(as.character(graph_backend)[1L])
    knn <- nn_without_self(graph$layout, k = k, backend = resolved, n_threads = n_threads)
    input_backend <- attr(knn, "backend") %||% resolved
  } else {
    resolved <- resolve_knn_graph_backend(as.character(graph_backend)[1L])
    knn <- nn_without_self(graph, k = k, backend = resolved, n_threads = n_threads)
    input_backend <- attr(knn, "backend") %||% resolved
  }

  knn_input <- coerce_knn_input(knn, arg_name = "graph")
  if (!is.na(knn_input$input_backend)) input_backend <- knn_input$input_backend
  if (identical(weight, "auto")) {
    weight <- if (identical(graph_space, "embedding")) "distance" else "snn"
  }
  if (knn_input$n_neighbors < k) k <- knn_input$n_neighbors
  cols <- seq_len(k)
  ans <- graph_cluster_cpp(
    knn_input$indices[, cols, drop = FALSE],
    knn_input$distances[, cols, drop = FALSE],
    method = method,
    backend = backend,
    weight_type = weight,
    prune = prune,
    mutual = isTRUE(mutual),
    n_threads = n_threads,
    n_runs = n_runs,
    resolution = resolution,
    n_iterations = n_iterations,
    steps = steps,
    seed = seed
  )
  ans$parameters <- list(
    k = as.integer(k),
    graph_backend = input_backend,
    graph_space = graph_space,
    input_method = input_method,
    weight = weight,
    mutual = isTRUE(mutual),
    prune = prune,
    resolution = resolution,
    objective_function = objective_function,
    n_iterations = as.integer(n_iterations),
    steps = as.integer(steps),
    n_threads = as.integer(n_threads)
  )
  ans$sources <- graph_cluster_sources(method, backend)
  class(ans) <- "faissR_graph_cluster"
  ans
}

graph_cluster_sources <- function(method, backend) {
  base <- c(
    "faissR native C++/OpenMP implementation for CPU graph clustering",
    "Blondel et al. (2008) for Louvain modularity optimization",
    "Traag et al. (2019) for Leiden community detection",
    "Pons and Latapy (2006) for random-walk walktrap clustering"
  )
  if (identical(method, "leiden")) {
    base <- c(
      base,
      "Sahu (2024), GVE-Leiden/OpenMP, as multicore Leiden implementation inspiration",
      "Sahu (2024), heuristic dynamic Leiden, as dynamic Leiden inspiration",
      "https://github.com/puzzlef/leiden-communities-openmp",
      "https://github.com/puzzlef/leiden-communities-openmp-heuristic-dynamic"
    )
  }
  if (identical(method, "random_walking")) {
    base <- c(base, "Kapralov et al. (2021) for local parallel random-walk motivation")
  }
  if (identical(backend, "cuda")) {
    base <- c(base, "RAPIDS libcugraph/cuGraph for native CUDA Louvain, Leiden, and random-walk algorithms")
  }
  unique(base)
}

#' @export
print.faissR_graph_cluster <- function(x, ...) {
  cat("faissR graph clustering\n")
  cat("  method: ", x$method, "\n", sep = "")
  cat("  backend: ", x$backend, "\n", sep = "")
  cat("  implementation: ", x$implementation %||% "native_cpp", "\n", sep = "")
  cat("  vertices: ", length(x$membership), "\n", sep = "")
  cat("  communities: ", length(unique(x$membership)), "\n", sep = "")
  cat("  modularity: ", format(x$modularity, digits = 4), "\n", sep = "")
  invisible(x)
}
