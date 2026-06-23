#' Build a native nearest-neighbour graph
#'
#' `knn_graph()` turns a data matrix, a precomputed `nn()` result, or an
#' embedding object with a matrix `layout` into a weighted nearest-neighbour
#' graph. It returns a native `faissR_graph` edge-list object and does not
#' require `igraph`.
#'
#' @param data Numeric matrix/data frame, a KNN object returned by `nn()`, or an
#'   embedding object with a matrix `layout`.
#' @param knn Optional precomputed KNN object returned by `nn()`. If supplied,
#'   `data` is ignored for neighbour search.
#' @param k Number of non-self neighbours used in the graph.
#' @param backend Device backend passed to \code{\link{nn_without_self}()} when
#'   `knn` is not supplied: `"auto"`, `"cpu"`, or `"cuda"`.
#' @param nn_method Nearest-neighbour method passed to
#'   \code{\link{nn_without_self}()} when `knn` is not supplied. The default
#'   `"auto"` uses the shape-aware selector for the chosen backend.
#' @param method Backward-compatible alias for `nn_method`, provided so
#'   `knn_graph()` accepts the same nearest-neighbour method argument name as
#'   `nn()` and `knn()`. If both `method` and `nn_method` are supplied they must
#'   resolve to the same public method label.
#' @param metric Distance metric passed to \code{\link{nn_without_self}()} when
#'   `knn` is not supplied. Aliases such as `"l2"`, `"cor"`/`"pearson"`, and
#'   `"ip"` are accepted and stored as canonical metric labels. Correlation is
#'   centered cosine similarity, not raw inner product. Inner-product graph
#'   construction ranks neighbours by larger raw dot product while reusing
#'   faissR's shifted smaller-is-better distance convention from
#'   \code{\link{nn}()}.
#' @param tuning Tuning policy passed to \code{\link{nn_without_self}()} when
#'   `knn` is not supplied.
#' @param cagra_implementation CUDA CAGRA provider passed to
#'   \code{\link{nn_without_self}()} when KNN is computed here. `NULL` uses the
#'   global option; `"auto"` uses the same deterministic shape-aware provider
#'   rule as \code{\link{nn}()}, while `"faiss_gpu"` or `"cuvs"` force one
#'   provider.
#' @param cagra_build_algo Direct RAPIDS cuVS CAGRA graph-build algorithm passed
#'   to \code{\link{nn_without_self}()} when KNN is computed here. `NULL` uses
#'   the global `faissR.cuvs_cagra_build_algo` option. This setting applies to
#'   direct cuVS CAGRA and accepts `"auto"`, `"ivf_pq"`, `"nn_descent"`, or
#'   `"iterative_cagra_search"`.
#' @param weight Graph weighting. `"auto"` uses SNN/Jaccard weights for input
#'   space and distance weights for embedding space. `"snn"` builds full
#'   shared-nearest-neighbour Jaccard weights between all rows sharing at least
#'   one neighbour. `"adaptive"` uses `exp(-d_ij^2 / (sigma_i * sigma_j))`.
#'   `"distance"` uses `1 / (1 + distance)`. `"binary"` gives every edge weight 1.
#' @param mutual If `TRUE`, keep only reciprocal nearest-neighbour edges.
#' @param prune Drop edges with weight less than or equal to this value.
#' @param n_threads CPU threads passed to `nn()` when KNN is computed here.
#' @return A native `faissR_graph` edge-list object. The `faissR_graph`
#'   attribute stores graph-construction metadata: graph size, weighting,
#'   nearest-neighbour method, metric, tuning policy, requested/resolved KNN
#'   backends, and compact-relevant KNN result metadata such as `nn_approximation`,
#'   `nn_faiss`, `nn_cuvs`,
#'   `nn_spatial_index`, `nn_auto_selection`, `nn_metric_transform`, and
#'   `nn_distance_transform`. For precomputed KNN input,
#'   `nn_backend` prefers the KNN object's resolved backend when available, so
#'   benchmark metadata records concrete FAISS/cuVS routes rather than only the
#'   public requested backend.
#' @examples
#' x <- scale(as.matrix(iris[, 1:4]))
#' g <- knn_graph(x, k = 15, backend = "cpu")
#' cl <- graph_cluster(g, method = "louvain", backend = "cpu", n_clusters = 3)
#' table(cl$membership)
#' @export
knn_graph <- function(data,
                      knn = NULL,
                      k = 50L,
                      backend = c("auto", "cpu", "cuda"),
                      nn_method = c("auto", "exact", "flat", "bruteforce", "grid", "hnsw", "ivf", "ivfpq", "vamana", "nsg", "nndescent", "cagra"),
                      method = NULL,
                      metric = c("euclidean", "cosine", "correlation", "inner_product"),
                      tuning = c("auto", "cache", "pilot", "fixed", "off", "none"),
                      cagra_implementation = NULL,
                      cagra_build_algo = NULL,
                      weight = c("auto", "snn", "adaptive", "distance", "binary"),
                      mutual = FALSE,
                      prune = 0,
                      n_threads = NULL) {
  if (!is.null(method)) {
    method <- public_nn_method_label(normalize_nn_method(method))
    if (!missing(nn_method)) {
      nn_method_requested <- public_nn_method_label(normalize_nn_method(nn_method))
      if (!identical(method, nn_method_requested)) {
        stop("`method` and `nn_method` must resolve to the same nearest-neighbour method.", call. = FALSE)
      }
    }
    nn_method <- method
  } else {
    nn_method <- public_nn_method_label(normalize_nn_method(nn_method))
  }
  metric <- normalize_nn_metric(metric)
  tuning <- normalize_nn_tuning(tuning)
  cagra_implementation <- normalize_cagra_implementation_arg(cagra_implementation)
  cagra_build_algo <- normalize_cagra_build_algo_arg(cagra_build_algo)
  weight <- normalize_graph_weight(weight)
  k <- normalize_graph_positive_int(k, "k")
  mutual <- normalize_scalar_logical_arg(mutual, "mutual", default = FALSE)
  prune <- suppressWarnings(as.numeric(prune))
  if (length(prune) != 1L || is.na(prune) || !is.finite(prune) || prune < 0) {
    stop("`prune` must be a non-negative number.", call. = FALSE)
  }
  input_backend <- NA_character_
  requested_graph_backend <- normalize_public_backend_arg(backend)
  resolved_graph_backend <- NA_character_
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
    } else if (is_embedding_layout(data)) {
      input_method <- as.character(data$method %||% "embedding")[1L]
      if (!is.matrix(data$layout)) {
        stop("Embedding objects passed to `knn_graph()` must contain a matrix `layout`.", call. = FALSE)
      }
      graph_space <- "embedding"
      graph_backend <- requested_graph_backend
      knn <- nn_without_self(
        data$layout,
        k = k,
        backend = graph_backend,
        method = nn_method,
        metric = metric,
        tuning = tuning,
        cagra_implementation = cagra_implementation,
        cagra_build_algo = cagra_build_algo,
        n_threads = n_threads
      )
      resolved_graph_backend <- attr(knn, "resolved_backend") %||% attr(knn, "backend") %||% graph_backend
      input_backend <- attr(knn, "backend") %||% resolved_graph_backend
    } else {
      graph_backend <- requested_graph_backend
      knn <- nn_without_self(
        data,
        k = k,
        backend = graph_backend,
        method = nn_method,
        metric = metric,
        tuning = tuning,
        cagra_implementation = cagra_implementation,
        cagra_build_algo = cagra_build_algo,
        n_threads = n_threads
      )
      resolved_graph_backend <- attr(knn, "resolved_backend") %||% attr(knn, "backend") %||% graph_backend
      input_backend <- attr(knn, "backend") %||% resolved_graph_backend
    }
  }

  knn_input <- coerce_knn_input(knn, arg_name = "knn")
  if (identical(weight, "auto")) {
    weight <- if (identical(graph_space, "embedding")) "distance" else "snn"
  }
  if (!is.na(knn_input$input_backend)) input_backend <- knn_input$input_backend
  if ((length(resolved_graph_backend) != 1L || is.na(resolved_graph_backend)) &&
      !is.na(input_backend)) {
    resolved_graph_backend <- input_backend
  }
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
    requested_backend = requested_graph_backend,
    resolved_backend = resolved_graph_backend,
    nn_method = nn_method,
    metric = attr(knn, "metric") %||% metric,
    tuning = tuning,
    nn_requested_backend = attr(knn, "requested_backend") %||% requested_graph_backend,
    nn_requested_method = attr(knn, "requested_method") %||% nn_method,
    nn_tuning = attr(knn, "tuning") %||% tuning,
    nn_cagra_implementation = cagra_implementation %||% NA_character_,
    nn_cagra_build_algo = cagra_build_algo %||% NA_character_,
    nn_approximation = attr(knn, "approximation") %||% NULL,
    nn_faiss = attr(knn, "faiss") %||% NULL,
    nn_cuvs = attr(knn, "cuvs") %||% NULL,
    nn_spatial_index = attr(knn, "spatial_index") %||% NULL,
    nn_auto_selection = attr(knn, "auto_selection") %||% NULL,
    nn_metric_transform = attr(knn, "metric_transform") %||% knn$metric_transform %||% NULL,
    nn_distance_transform = attr(knn, "distance_transform") %||% NULL,
    input_method = input_method,
    n_vertices = edges$n_vertices,
    n_edges = edges$n_edges
  )
  attr(edges, "faissR_graph") <- metadata
  class(edges) <- c("faissR_graph", "list")
  edges
}

is_embedding_layout <- function(x) {
  is.list(x) && is.matrix(x$layout) && !is.null(x$method)
}

resolve_knn_graph_backend <- function(backend) {
  normalize_public_compute_backend(backend)
}

resolve_graph_cluster_backend <- function(backend) {
  graph_cluster_backend_selection(backend)$resolved_backend
}

graph_cluster_backend_selection <- function(backend,
                                            method = "louvain",
                                            cuda_available_value = cuda_available(),
                                            cugraph_available_value = cugraph_available()) {
  backend <- normalize_scalar_choice_arg(
    backend,
    arg = "backend",
    default = "auto",
    formal_choices = c("auto", "cpu", "cuda")
  )
  if (is.na(backend) || !nzchar(backend)) backend <- "auto"
  backend <- tolower(backend)
  if (!backend %in% c("auto", "cpu", "cuda")) {
    stop("`backend` must be one of \"auto\", \"cpu\", or \"cuda\".", call. = FALSE)
  }
  method <- if (is.null(method)) "louvain" else normalize_graph_cluster_method(method)
  graph_cluster_auto_backend_cpp(
    backend,
    method,
    isTRUE(cuda_available_value),
    isTRUE(cugraph_available_value)
  )
}

#' Cluster a nearest-neighbour graph without igraph
#'
#' `graph_cluster()` runs native faissR community detection on a KNN graph.
#' The graph can be supplied as a precomputed `nn()` result or built from a
#' matrix/data frame using any faissR nearest-neighbour backend, including FAISS
#' and cuVS KNN backends. The CPU clustering backend is implemented in C++ and
#' uses OpenMP when available. The CUDA clustering backend uses native RAPIDS libcugraph for
#' `method = "louvain"` and `method = "leiden"` when faissR is built
#' against libcugraph. `method = "random_walking"` currently remains CPU-only.
#' CUDA graph clustering never calls Python and never silently falls back to CPU.
#'
#' @param graph Numeric matrix/data frame, a KNN object returned by `nn()`, or an
#'   embedding object with a matrix `layout`.
#' @param method One of `"random_walking"`, `"louvain"`, or `"leiden"`.
#'   `"random_walking"` uses a native random-walk label propagation pass inspired
#'   by walktrap/random-walk clustering. `"louvain"` uses native modularity local
#'   moving. `"leiden"` adds a native refinement pass that splits disconnected
#'   communities after local moving.
#' @param backend Community-detection backend. `"auto"` uses CUDA when
#'   libcugraph is available for Louvain/Leiden and CPU otherwise; auto keeps
#'   `"random_walking"` on CPU. `"cpu"` uses native C++/OpenMP. `"cuda"` uses
#'   native RAPIDS libcugraph for Louvain and Leiden when libcugraph was
#'   detected at build time; random-walking is CPU-only. The `"auto"` backend
#'   decision is made by the compiled `graph_cluster_auto_backend_cpp()`
#'   selector and recorded in `parameters$backend_selection`.
#' @param k Number of neighbours when `graph` is not already a KNN object.
#' @param graph_backend Backend passed to \code{\link{nn_without_self}()} for
#'   neighbour search when `graph` is a matrix or embedding.
#' @param graph_method Nearest-neighbour method passed to
#'   \code{\link{nn_without_self}()} when `graph` is a matrix or embedding.
#' @param metric Distance metric passed to \code{\link{nn_without_self}()} when
#'   `graph` is a matrix or embedding. Aliases such as `"l2"`,
#'   `"cor"`/`"pearson"`, and `"ip"` are accepted and stored as canonical
#'   metric labels. Correlation is centered cosine similarity, not raw inner
#'   product. Inner-product graph construction ranks neighbours by larger raw
#'   dot product while reusing faissR's shifted smaller-is-better distance
#'   convention from \code{\link{nn}()}.
#' @param tuning Tuning policy passed to \code{\link{nn_without_self}()} when
#'   `graph` is a matrix or embedding.
#' @param cagra_implementation CUDA CAGRA provider passed to
#'   \code{\link{nn_without_self}()} when graph KNN is computed here. `NULL`
#'   uses the global option; `"auto"`, `"faiss_gpu"`, or `"cuvs"` select the
#'   CAGRA provider for CUDA `graph_method = "cagra"` and CUDA-auto CAGRA
#'   routes.
#' @param cagra_build_algo Direct RAPIDS cuVS CAGRA graph-build algorithm passed
#'   to \code{\link{nn_without_self}()} when graph KNN is computed here. `NULL`
#'   uses the global `faissR.cuvs_cagra_build_algo` option. This is a CAGRA
#'   construction parameter, not a fallback to a different public NN method.
#' @param weight KNN graph weighting. See \code{\link{knn_graph}()}.
#' @param mutual If `TRUE`, keep only reciprocal nearest-neighbour edges when a
#'   graph must be built from data or a KNN object.
#' @param prune Drop graph edges with weight less than or equal to this value
#'   when a graph must be built from data or a KNN object.
#' @param n_threads CPU threads for KNN construction and native CPU clustering.
#' @param n_runs Number of independent native runs. The best modularity is kept.
#' @param resolution Modularity resolution for Louvain/Leiden-style scoring.
#'   Use a positive number for direct resolution control. When `n_clusters` is
#'   supplied and `resolution` is omitted or `NULL`, faissR seeds the bounded
#'   deterministic grid from the no-pilot graph-shape heuristic instead of the
#'   numeric default.
#' @param n_clusters Optional target number of communities for Louvain/Leiden.
#'   If supplied, faissR evaluates a bounded deterministic resolution grid on
#'   the already-built graph and keeps the result whose community count is
#'   closest to `n_clusters`. The grid is centered from an explicitly requested
#'   `resolution`, or from an automatic seed when `resolution` is omitted or
#'   `NULL`; when graph size is known the automatic seed uses a no-pilot
#'   shape heuristic based on
#'   `n_clusters / sqrt(n_vertices)`. The search width is also
#'   shape-aware: small graphs use a wider grid, while large graphs use fewer
#'   deterministic candidates to avoid repeated full Louvain/Leiden passes.
#'   This is a convenience target, not a hard guarantee. If `n_clusters` is
#'   supplied and `method` is omitted, faissR uses `"louvain"` as the
#'   target-count clustering method. The target must be a positive integer and
#'   cannot exceed the number of graph vertices.
#' @param objective_function Community objective. Only `"modularity"` is
#'   currently implemented by the native CPU and CUDA clustering paths; CPM is
#'   not accepted until the implementation can optimize and report it honestly.
#' @param n_iterations Native clustering iterations.
#' @param steps Random-walk propagation depth.
#' @param seed Optional seed used to make repeated runs reproducible.
#' @param ... Reserved for future backend options.
#' @return A `faissR_graph_cluster` list with membership, modularity, method,
#'   backend, graph edge list, parameters, and source acknowledgements.
#'   `backend` records the clustering implementation that actually ran, while
#'   `parameters$requested_backend` and `parameters$resolved_backend` record the
#'   public backend request and the device policy after resolving `"auto"`.
#'   `parameters$backend_selection` records the compiled backend selector
#'   metadata, including `runtime_decision`, CUDA/libcugraph availability flags,
#'   and `tuning_source = "cpp"`.
#'   When `graph_cluster()` builds or receives a `faissR_graph`,
#'   `parameters$graph_backend`, `parameters$graph_requested_backend`, and
#'   `parameters$graph_resolved_backend` record the concrete KNN implementation,
#'   public graph backend request, and resolved KNN backend.
#'   `parameters$n_vertices` and `parameters$n_edges` record the clustered graph
#'   size for benchmark summaries. `parameters$nn_metric_transform` and
#'   `parameters$nn_distance_transform` preserve normalized metric conversion
#'   metadata from the KNN route that built the graph. `parameters$nn_approximation`,
#'   `parameters$nn_faiss`, `parameters$nn_cuvs`, `parameters$nn_spatial_index`,
#'   `parameters$nn_auto_selection` preserve compact
#'   KNN route metadata when the graph is built internally. When `n_clusters`
#'   is supplied,
#'   `target_n_clusters`, `selected_resolution`, `target_gap`,
#'   `resolution_selection`, and `resolution_search` record the requested
#'   target, selected resolution, final community-count gap, deterministic
#'   selection rule, and full resolution search table. The validated request is
#'   also stored as `parameters$n_clusters_requested` for benchmark summaries
#'   and as `parameters$n_clusters` for backward compatibility. The deterministic
#'   candidate center and bounded grid are computed by the compiled
#'   `graph_resolution_candidates_cpp()` helper and record
#'   `resolution_selection$tuning_source = "cpp"`. `parameters$resolution` is
#'   `NA` for target-auto searches where `n_clusters` is supplied and
#'   `resolution` is omitted or `NULL`; the actual automatic center is stored in
#'   `resolution_selection$candidate_center`.
#'   `parameters$resolution_source` is `"default"`, `"user"`, or
#'   `"target_auto"` depending on how the resolution-grid seed was chosen.
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
                          graph_method = c("auto", "exact", "flat", "bruteforce", "grid", "hnsw", "ivf", "ivfpq", "vamana", "nsg", "nndescent", "cagra"),
                          metric = c("euclidean", "cosine", "correlation", "inner_product"),
                          tuning = c("auto", "cache", "pilot", "fixed", "off", "none"),
                          cagra_implementation = NULL,
                          cagra_build_algo = NULL,
                          weight = c("auto", "snn", "adaptive", "distance", "binary"),
                          mutual = FALSE,
                          prune = 0,
                          n_threads = NULL,
                          n_runs = 1L,
                          resolution = 1,
                          n_clusters = NULL,
                          objective_function = c("modularity"),
                          n_iterations = 10L,
                          steps = 4L,
                          seed = NULL,
                          ...) {
  resolution_was_missing <- missing(resolution)
  if (missing(method) && !is.null(n_clusters)) {
    method <- "louvain"
  }
  method <- normalize_graph_cluster_method(method)
  requested_backend <- normalize_public_backend_arg(backend)
  backend_selection <- graph_cluster_backend_selection(requested_backend, method = method)
  backend <- backend_selection$resolved_backend
  graph_method <- public_nn_method_label(normalize_nn_method(graph_method))
  metric <- normalize_nn_metric(metric)
  tuning <- normalize_nn_tuning(tuning)
  cagra_implementation <- normalize_cagra_implementation_arg(cagra_implementation)
  cagra_build_algo <- normalize_cagra_build_algo_arg(cagra_build_algo)
  if (identical(method, "random_walking") && identical(backend, "cuda")) {
    if (identical(requested_backend, "auto")) {
      backend <- "cpu"
    } else {
      stop(
        "`method = \"random_walking\"` is currently CPU-only. ",
        "Use `backend = \"cpu\"` or `backend = \"auto\"`.",
        call. = FALSE
      )
    }
  }
  weight <- normalize_graph_weight(weight)
  objective_function <- normalize_graph_objective_function(objective_function)
  n_threads <- normalize_nn_threads(n_threads)
  k <- normalize_graph_positive_int(k, "k")
  n_runs <- normalize_graph_positive_int(n_runs, "n_runs")
  n_iterations <- normalize_graph_positive_int(n_iterations, "n_iterations")
  steps <- normalize_graph_positive_int(steps, "steps")
  n_clusters <- normalize_graph_target_clusters(n_clusters, method)
  if (!is.null(n_clusters) && isTRUE(resolution_was_missing)) {
    resolution <- NULL
  }
  resolution_info <- normalize_graph_resolution(
    resolution,
    has_target = !is.null(n_clusters),
    was_missing = resolution_was_missing
  )
  resolution <- resolution_info$value
  resolution_source <- resolution_info$source
  prune <- suppressWarnings(as.numeric(prune))
  if (length(prune) != 1L || is.na(prune) || !is.finite(prune) || prune < 0) {
    stop("`prune` must be a non-negative number.", call. = FALSE)
  }
  seed <- if (is.null(seed)) 1L else normalize_graph_positive_int(seed, "seed")


  graph_space <- "input"
  input_method <- NA_character_
  input_backend <- NA_character_
  graph_requested_backend <- NA_character_
  if (inherits(graph, "faissR_graph")) {
    meta <- attr(graph, "faissR_graph") %||% list()
    if (!is.null(meta$requested_backend)) {
      meta$graph_requested_backend <- meta$requested_backend
      meta$requested_backend <- NULL
    }
    if (!is.null(meta$resolved_backend)) {
      meta$graph_resolved_backend <- meta$resolved_backend
      meta$resolved_backend <- NULL
    }
    if (is.null(meta$graph_backend)) {
      meta$graph_backend <- meta$nn_backend %||%
        meta$graph_resolved_backend %||%
        meta$graph_requested_backend %||%
        NA_character_
    }
    graph_n_clusters <- normalize_graph_target_clusters(n_clusters, method)
    graph_n_clusters <- validate_graph_target_cluster_count(graph_n_clusters, graph$n_vertices)
    ans <- graph_cluster_edges_target(
      graph,
      method = method,
      backend = backend,
      n_threads = n_threads,
      n_runs = n_runs,
      resolution = resolution,
      n_clusters = graph_n_clusters,
      n_iterations = n_iterations,
      steps = steps,
      seed = seed
    )
    ans$parameters <- c(
      meta,
      list(
        resolution = resolution,
        resolution_source = resolution_source,
        n_clusters = graph_n_clusters,
        n_clusters_requested = graph_n_clusters,
        selected_resolution = ans$selected_resolution %||% resolution,
        target_gap = ans$target_gap %||% NA_integer_,
        resolution_selection = ans$resolution_selection %||% NULL,
        requested_backend = requested_backend,
        resolved_backend = backend,
        backend_selection = backend_selection,
        n_vertices = as.integer(graph$n_vertices),
        n_edges = as.integer(graph$n_edges),
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
    graph_requested_backend <- attr(knn, "requested_backend") %||% NA_character_
  } else if (is_embedding_layout(graph)) {
    if (!is.matrix(graph$layout)) {
      stop("Embedding objects passed to `graph_cluster()` must contain a matrix `layout`.", call. = FALSE)
    }
    graph_space <- "embedding"
    input_method <- as.character(graph$method %||% "embedding")[1L]
    resolved <- normalize_public_backend_arg(graph_backend, arg = "graph_backend")
    graph_requested_backend <- resolved
    knn <- nn_without_self(
      graph$layout,
      k = k,
      backend = resolved,
      method = graph_method,
      metric = metric,
      tuning = tuning,
      cagra_implementation = cagra_implementation,
      cagra_build_algo = cagra_build_algo,
      n_threads = n_threads
    )
    input_backend <- attr(knn, "resolved_backend") %||% attr(knn, "backend") %||% resolved
  } else {
    resolved <- normalize_public_backend_arg(graph_backend, arg = "graph_backend")
    graph_requested_backend <- resolved
    knn <- nn_without_self(
      graph,
      k = k,
      backend = resolved,
      method = graph_method,
      metric = metric,
      tuning = tuning,
      cagra_implementation = cagra_implementation,
      cagra_build_algo = cagra_build_algo,
      n_threads = n_threads
    )
    input_backend <- attr(knn, "resolved_backend") %||% attr(knn, "backend") %||% resolved
  }

  knn_input <- coerce_knn_input(knn, arg_name = "graph")
  if (!is.na(knn_input$input_backend)) input_backend <- knn_input$input_backend
  if ((length(graph_requested_backend) != 1L || is.na(graph_requested_backend)) &&
      !is.na(input_backend)) {
    graph_requested_backend <- input_backend
  }
  if (identical(weight, "auto")) {
    weight <- if (identical(graph_space, "embedding")) "distance" else "snn"
  }
  if (knn_input$n_neighbors < k) k <- knn_input$n_neighbors
  cols <- seq_len(k)
  graph_edges <- knn_graph_edges_cpp(
    knn_input$indices[, cols, drop = FALSE],
    knn_input$distances[, cols, drop = FALSE],
    weight_type = weight,
    prune = prune,
    mutual = mutual
  )
  n_clusters <- validate_graph_target_cluster_count(n_clusters, graph_edges$n_vertices)
  ans <- graph_cluster_edges_target(
    graph_edges,
    method = method,
    backend = backend,
    n_threads = n_threads,
    n_runs = n_runs,
    resolution = resolution,
    n_clusters = n_clusters,
    n_iterations = n_iterations,
    steps = steps,
    seed = seed
  )
  ans$parameters <- list(
    k = as.integer(k),
    graph_backend = input_backend,
    graph_requested_backend = graph_requested_backend,
    graph_resolved_backend = input_backend,
    graph_method = graph_method,
    metric = attr(knn, "metric") %||% metric,
    tuning = tuning,
    nn_requested_backend = attr(knn, "requested_backend") %||% graph_requested_backend,
    nn_requested_method = attr(knn, "requested_method") %||% graph_method,
    nn_tuning = attr(knn, "tuning") %||% tuning,
    nn_cagra_implementation = cagra_implementation %||% NA_character_,
    nn_cagra_build_algo = cagra_build_algo %||% NA_character_,
    nn_approximation = attr(knn, "approximation") %||% NULL,
    nn_faiss = attr(knn, "faiss") %||% NULL,
    nn_cuvs = attr(knn, "cuvs") %||% NULL,
    nn_spatial_index = attr(knn, "spatial_index") %||% NULL,
    nn_auto_selection = attr(knn, "auto_selection") %||% NULL,
    nn_metric_transform = attr(knn, "metric_transform") %||% knn$metric_transform %||% NULL,
    nn_distance_transform = attr(knn, "distance_transform") %||% NULL,
    graph_space = graph_space,
    input_method = input_method,
    weight = weight,
    mutual = mutual,
    prune = prune,
    resolution = resolution,
    resolution_source = resolution_source,
    n_clusters = n_clusters,
    n_clusters_requested = n_clusters,
    selected_resolution = ans$selected_resolution %||% resolution,
    target_gap = ans$target_gap %||% NA_integer_,
    resolution_selection = ans$resolution_selection %||% NULL,
    requested_backend = requested_backend,
    resolved_backend = backend,
    backend_selection = backend_selection,
    n_vertices = as.integer(graph_edges$n_vertices),
    n_edges = as.integer(graph_edges$n_edges),
    objective_function = objective_function,
    n_iterations = as.integer(n_iterations),
    steps = as.integer(steps),
    n_threads = as.integer(n_threads)
  )
  ans$sources <- graph_cluster_sources(method, backend)
  class(ans) <- "faissR_graph_cluster"
  ans
}

normalize_graph_cluster_method <- function(method) {
  method <- normalize_scalar_choice_arg(
    method,
    arg = "method",
    default = "random_walking",
    formal_choices = c("random_walking", "louvain", "leiden")
  )
  if (is.na(method) || !nzchar(method)) method <- "random_walking"
  method <- trimws(method)
  methods <- c("random_walking", "louvain", "leiden")
  if (!method %in% methods) {
    stop(
      "`method` must be one of \"random_walking\", \"louvain\", or \"leiden\".",
      call. = FALSE
    )
  }
  method
}

normalize_graph_weight <- function(weight) {
  weight <- normalize_scalar_choice_arg(
    weight,
    arg = "weight",
    default = "auto",
    formal_choices = c("auto", "snn", "adaptive", "distance", "binary")
  )
  if (is.na(weight) || !nzchar(weight)) weight <- "auto"
  weight <- trimws(weight)
  weights <- c("auto", "snn", "adaptive", "distance", "binary")
  if (!weight %in% weights) {
    stop(
      "`weight` must be one of \"auto\", \"snn\", \"adaptive\", \"distance\", or \"binary\".",
      call. = FALSE
    )
  }
  weight
}

normalize_graph_objective_function <- function(objective_function) {
  objective_function <- normalize_scalar_choice_arg(
    objective_function,
    arg = "objective_function",
    default = "modularity",
    formal_choices = c("modularity")
  )
  if (is.na(objective_function) || !nzchar(objective_function)) objective_function <- "modularity"
  objective_function <- trimws(objective_function)
  if (!objective_function %in% c("modularity")) {
    stop(
      "`objective_function` must be \"modularity\". CPM is not implemented ",
      "in the current native graph clustering backend.",
      call. = FALSE
    )
  }
  objective_function
}

normalize_graph_target_clusters <- function(n_clusters, method) {
  if (is.null(n_clusters)) return(NULL)
  if (!is.null(method) && identical(method, "random_walking")) {
    stop("`n_clusters` is available for `method = \"louvain\"` or `\"leiden\"`, not `\"random_walking\"`.", call. = FALSE)
  }
  value <- suppressWarnings(as.numeric(n_clusters))
  if (length(value) != 1L || is.na(value) || !is.finite(value) || value < 1L ||
      abs(value - round(value)) > sqrt(.Machine$double.eps)) {
    stop("`n_clusters` must be a positive integer.", call. = FALSE)
  }
  as.integer(round(value))
}

normalize_graph_positive_int <- function(x, arg) {
  value <- suppressWarnings(as.numeric(x))
  if (length(value) != 1L || is.na(value) || !is.finite(value) || value < 1L ||
      abs(value - round(value)) > sqrt(.Machine$double.eps)) {
    stop("`", arg, "` must be a single positive integer.", call. = FALSE)
  }
  as.integer(round(value))
}

validate_graph_target_cluster_count <- function(n_clusters, n_vertices) {
  if (is.null(n_clusters)) return(NULL)
  n_vertices <- suppressWarnings(as.integer(n_vertices))
  if (length(n_vertices) != 1L || is.na(n_vertices) || !is.finite(n_vertices) || n_vertices < 1L) {
    return(n_clusters)
  }
  if (n_clusters > n_vertices) {
    stop("`n_clusters` cannot be larger than the number of graph vertices.", call. = FALSE)
  }
  n_clusters
}

normalize_graph_resolution <- function(resolution, has_target = FALSE, was_missing = FALSE) {
  if (is.null(resolution)) {
    if (isTRUE(has_target)) {
      return(list(value = NA_real_, source = "target_auto"))
    }
    stop("`resolution` must be a positive number, or NULL when `n_clusters` is supplied.", call. = FALSE)
  }
  value <- suppressWarnings(as.numeric(resolution))
  if (length(value) != 1L || is.na(value) || !is.finite(value) || value <= 0) {
    stop("`resolution` must be a positive number.", call. = FALSE)
  }
  list(
    value = value,
    source = if (isTRUE(was_missing)) "default" else "user"
  )
}

graph_resolution_center <- function(resolution, n_clusters, n_vertices = NULL) {
  graph_resolution_candidate_info(resolution, n_clusters, n_vertices)$center
}

graph_resolution_grid_exponents <- function(n_vertices = NULL) {
  graph_resolution_candidate_info(1, 1L, n_vertices)$exponents
}

graph_resolution_candidates <- function(resolution, n_clusters, n_vertices = NULL) {
  if (is.null(n_clusters)) return(resolution)
  graph_resolution_candidate_info(resolution, n_clusters, n_vertices)$candidates
}

graph_resolution_candidate_info <- function(resolution, n_clusters, n_vertices = NULL) {
  as_int <- function(x) {
    if (is.null(x)) return(NA_integer_)
    x <- suppressWarnings(as.integer(x[1L]))
    if (length(x) != 1L || is.na(x)) NA_integer_ else x
  }
  resolution <- suppressWarnings(as.numeric(resolution))
  if (length(resolution) != 1L || is.na(resolution) || !is.finite(resolution)) {
    resolution <- NA_real_
  }
  graph_resolution_candidates_cpp(
    resolution,
    as_int(n_clusters),
    as_int(n_vertices)
  )
}

graph_cluster_edges_target <- function(edge_list,
                                       method,
                                       backend,
                                       n_threads,
                                       n_runs,
                                       resolution,
                                       n_clusters,
                                       n_iterations,
                                       steps,
                                       seed) {
  n_vertices <- edge_list$n_vertices %||% NA_integer_
  resolution_info <- graph_resolution_candidate_info(resolution, n_clusters, n_vertices)
  resolution_center <- resolution_info$center
  candidates <- if (is.null(n_clusters)) resolution else resolution_info$candidates
  best <- NULL
  best_index <- NA_integer_
  summary <- vector("list", length(candidates))
  for (i in seq_along(candidates)) {
    candidate_resolution <- candidates[[i]]
    ans <- graph_cluster_edges_cpp(
      edge_list,
      method = method,
      backend = backend,
      n_threads = n_threads,
      n_runs = n_runs,
      resolution = candidate_resolution,
      n_iterations = n_iterations,
      steps = steps,
      seed = seed
    )
    ans$selected_resolution <- candidate_resolution
    summary[[i]] <- data.frame(
      candidate = as.integer(i),
      resolution = candidate_resolution,
      n_communities = as.integer(ans$n_communities),
      modularity = as.numeric(ans$modularity),
      target_gap = if (is.null(n_clusters)) NA_integer_ else abs(as.integer(ans$n_communities) - n_clusters),
      selected = FALSE,
      stringsAsFactors = FALSE
    )
    if (is.null(best)) {
      best <- ans
      best_index <- i
    } else if (is.null(n_clusters)) {
      if (isTRUE(as.numeric(ans$modularity) > as.numeric(best$modularity))) {
        best <- ans
        best_index <- i
      }
    } else {
      current_gap <- abs(as.integer(ans$n_communities) - n_clusters)
      best_gap <- abs(as.integer(best$n_communities) - n_clusters)
      if (current_gap < best_gap ||
          (current_gap == best_gap && as.numeric(ans$modularity) > as.numeric(best$modularity))) {
        best <- ans
        best_index <- i
      }
    }
  }
  if (!is.null(n_clusters)) {
    best$target_n_clusters <- as.integer(n_clusters)
    search <- do.call(rbind, summary)
    if (!is.na(best_index) && best_index >= 1L && best_index <= nrow(search)) {
      search$selected[[best_index]] <- TRUE
    }
    best$resolution_search <- search
    best$target_gap <- abs(as.integer(best$n_communities) - as.integer(n_clusters))
    best$resolution_selection <- list(
      criterion = "closest_n_communities_then_highest_modularity",
      selected_candidate = as.integer(best_index),
      target_gap = as.integer(best$target_gap),
      candidate_center = as.numeric(resolution_center),
      n_vertices = as.integer(n_vertices),
      tuning_source = resolution_info$tuning_source %||% "cpp"
    )
    best$resolution_tuning_source <- resolution_info$tuning_source %||% "cpp"
  }
  best
}

graph_cluster_sources <- function(method, backend) {
  base <- c(
    "faissR native C++/OpenMP implementation for CPU graph clustering"
  )
  if (identical(method, "louvain")) {
    base <- c(base, "Blondel et al. (2008) for Louvain modularity optimization")
  }
  if (identical(method, "leiden")) {
    base <- c(
      base,
      "Blondel et al. (2008) for Louvain modularity optimization",
      "Traag et al. (2019) for Leiden community detection",
      "Sahu (2024), GVE-Leiden/OpenMP, as multicore Leiden implementation inspiration",
      "Sahu (2024), heuristic dynamic Leiden, as dynamic Leiden inspiration",
      "https://github.com/puzzlef/leiden-communities-openmp",
      "https://github.com/puzzlef/leiden-communities-openmp-heuristic-dynamic"
    )
  }
  if (identical(method, "random_walking")) {
    base <- c(
      base,
      "Pons and Latapy (2006) for random-walk walktrap clustering",
      "Kapralov et al. (2021) for local parallel random-walk motivation"
    )
  }
  if (identical(backend, "cuda")) {
    base <- c(base, "RAPIDS libcugraph/cuGraph for native CUDA Louvain and Leiden algorithms")
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
  if (!is.null(x$target_n_clusters)) {
    cat("  target communities: ", x$target_n_clusters, "\n", sep = "")
  }
  if (!is.null(x$selected_resolution)) {
    cat("  selected resolution: ", format(x$selected_resolution, digits = 4), "\n", sep = "")
  }
  cat("  modularity: ", format(x$modularity, digits = 4), "\n", sep = "")
  invisible(x)
}
