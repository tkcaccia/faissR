#' Select nearest neighbours from a candidate matrix
#'
#' `candidate_knn()` computes exact top-k neighbours restricted to a supplied
#' per-query candidate matrix. It is useful after an approximate candidate
#' generation step, for NN-descent refinement, graph refinement, or landmark
#' projection. The function does not generate candidates; it only scores and
#' ranks the candidates that you pass in.
#'
#' @param data Numeric reference matrix with observations in rows.
#' @param candidates Integer matrix of 1-based candidate reference row indices.
#'   It must have one row per query. Invalid, missing, zero, or out-of-range
#'   entries are ignored.
#' @param points Numeric query matrix with observations in rows. Defaults to
#'   `data`, i.e. self-query candidate KNN.
#' @param k Number of neighbours to return from each candidate row.
#' @param backend `"auto"`/`"cpu"` for the general CPU implementation,
#'   `"cuda"` for the native CUDA row-candidate kernel. GPU backends currently
#'   require self-query candidates with `exclude_self = TRUE`.
#' @param metric `"euclidean"`, `"cosine"`, `"correlation"`, or
#'   `"inner_product"`. Aliases such as `"l2"`, `"cor"`/`"pearson"`, and
#'   `"ip"` are accepted. CUDA candidate scoring supports Euclidean directly
#'   and cosine/correlation through normalized Euclidean scoring; raw
#'   inner-product CUDA candidate scoring is not exposed.
#' @param n_threads CPU threads for the CPU backend.
#' @param exclude_self If `TRUE`, remove each query row from its own candidate
#'   set. This is valid only for self-query candidate KNN.
#' @return A `faissR_nn` object with `indices` and `distances`. If a row
#'   has fewer than `k` valid unique candidates, remaining entries are `NA` and
#'   `Inf`.
#' @examples
#' x <- scale(as.matrix(iris[, 1:4]))
#' rough <- nn(x, k = 10, backend = "cpu")
#' refined <- candidate_knn(x, rough$indices, k = 5, exclude_self = TRUE)
#' refined
#' @export
candidate_knn <- function(data,
                          candidates,
                          points = data,
                          k,
                          backend = c("auto", "cpu", "cuda"),
                          metric = c("euclidean", "cosine", "correlation", "inner_product"),
                          n_threads = NULL,
                          exclude_self = FALSE) {
  backend <- normalize_public_backend_arg(backend)
  metric <- normalize_nn_metric(metric)
  exclude_self <- normalize_scalar_logical_arg(exclude_self, "exclude_self", default = FALSE)
  x <- as.matrix(data)
  storage.mode(x) <- "double"
  q <- as.matrix(points)
  storage.mode(q) <- "double"
  if (nrow(x) < 1L || ncol(x) < 1L || nrow(q) < 1L || ncol(q) != ncol(x)) {
    stop("`data` and `points` must have compatible positive dimensions.", call. = FALSE)
  }
  if (!all(is.finite(x)) || !all(is.finite(q))) {
    stop("`data` and `points` must contain only finite values.", call. = FALSE)
  }
  cand <- as.matrix(candidates)
  storage.mode(cand) <- "integer"
  if (nrow(cand) != nrow(q) || ncol(cand) < 1L) {
    stop("`candidates` must have one row per query and at least one column.", call. = FALSE)
  }
  k <- suppressWarnings(as.integer(k))
  if (length(k) != 1L || is.na(k) || !is.finite(k) || k < 1L || k > ncol(cand)) {
    stop("`k` must be an integer in [1, ncol(candidates)].", call. = FALSE)
  }
  n_threads <- normalize_nn_threads(n_threads)
  self_query <- nrow(x) == nrow(q) && ncol(x) == ncol(q) && identical(x, q)
  if (exclude_self && !isTRUE(self_query)) {
    stop("`exclude_self = TRUE` requires `points` to be `data`.", call. = FALSE)
  }
  if (exclude_self) {
    self_hits <- cand == row(cand)
    cand[self_hits] <- NA_integer_
  }

  if (identical(backend, "auto")) {
    backend <- "cpu"
  }

  if (identical(backend, "cuda")) {
    if (identical(metric, "inner_product")) {
      stop("CUDA candidate KNN does not support `metric = \"inner_product\"`.", call. = FALSE)
    }
    if (!exclude_self) {
      stop("CUDA candidate KNN currently requires `exclude_self = TRUE`.", call. = FALSE)
    }
    if (!isTRUE(self_query)) {
      stop("CUDA candidate KNN currently requires self-query candidates.", call. = FALSE)
    }
    if (!isTRUE(cuda_available())) {
      stop("No CUDA GPU backend is available on this machine.", call. = FALSE)
    }
    metric_inputs <- NULL
    search_x <- x
    if (metric %in% c("cosine", "correlation")) {
      metric_inputs <- normalized_euclidean_metric_inputs(x, q, self_query, metric)
      search_x <- metric_inputs$data
    }
    out <- row_candidate_knn_cuda_cpp(search_x, cand, as.integer(k))
    result <- finish_nn_result(out, "cuda_candidate", k, TRUE, exact = FALSE, metric = metric)
    if (!is.null(metric_inputs)) {
      result <- finalize_normalized_euclidean_metric_result(result, metric_inputs)
    }
    attr(result, "candidate_knn") <- list(
      candidate_columns = as.integer(ncol(cand)),
      exclude_self = exclude_self,
      exact_within_candidates = TRUE,
      transform = if (is.null(metric_inputs)) NA_character_ else metric_inputs$transform
    )
    return(result)
  }

  out <- candidate_knn_cpp(
    x,
    q,
    cand,
    as.integer(k),
    metric,
    FALSE,
    exclude_self,
    TRUE,
    as.integer(n_threads)
  )
  result <- finish_nn_result(out, "cpu_candidate", k, self_query, exact = FALSE, metric = metric)
  attr(result, "candidate_knn") <- list(
    candidate_columns = as.integer(ncol(cand)),
    exclude_self = exclude_self,
    exact_within_candidates = TRUE,
    n_threads = as.integer(out$n_threads)
  )
  result
}
