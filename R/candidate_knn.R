#' Select nearest neighbours from a candidate matrix
#'
#' `candidate_knn()` computes exact top-k neighbours restricted to a supplied
#' per-query candidate matrix. It is useful after an approximate candidate
#' generation step, for NN-descent refinement, graph refinement, or landmark
#' projection. The function does not generate candidates; it only scores and
#' ranks the candidates that you pass in.
#'
#' @param data Dense numeric reference matrix with observations in rows.
#'   Sparse, delayed, file-backed, and other matrix-like objects are rejected
#'   before conversion.
#' @param candidates Integer matrix of 1-based candidate reference row indices.
#'   It must have one row per query. Invalid, missing, zero, or out-of-range
#'   entries are ignored.
#' @param points Dense numeric query matrix with observations in rows. Defaults
#'   to `data`, i.e. self-query candidate KNN.
#' @param k Number of neighbours to return from each candidate row.
#' @param backend `"cpu"` for the general CPU implementation or `"cuda"` for
#'   the native CUDA row-candidate kernel. `NULL` follows the package backend
#'   configuration. GPU backends currently
#'   require self-query candidates with `exclude_self = TRUE` and `k <= 256`.
#' @param metric `"euclidean"`, `"cosine"`, or `"correlation"`. Legacy
#'   metric aliases such as `"l2"`, `"cor"`,
#'   `"pearson"`, and `"ip"` are rejected. Correlation is centered cosine
#'   similarity. CUDA candidate scoring supports Euclidean directly and
#'   cosine/correlation through normalized Euclidean scoring.
#'   CUDA rejects zero-vector cosine and constant-row correlation inputs;
#'   inspect affected rows with [nn_metric_preflight()].
#' @param n_threads CPU threads for the CPU backend.
#' @param exclude_self If `TRUE`, remove each query row from its own candidate
#'   set. This is valid only for self-query candidate KNN.
#' @return A `faissR_nn` object with `indices` and `distances`. If a row
#'   has fewer than `k` valid unique candidates, remaining entries are `NA` and
#'   `Inf` on both CPU and CUDA. No neighbors outside the supplied candidate
#'   set are added. Distance-contract metadata states whether the values are a
#'   metric and comparable across query rows.
#' @examples
#' x <- scale(as.matrix(iris[, 1:4]))
#' rough <- nn(x, k = 10, backend = "cpu", n_threads = 1)
#' refined <- candidate_knn(x, rough$indices, k = 5, exclude_self = TRUE,
#'     n_threads = 1)
#' refined
#' @export
candidate_knn <- function(
    data,
    candidates,
    points = data,
    k,
    backend = NULL,
    metric = c("euclidean", "cosine", "correlation"),
    n_threads = NULL,
    exclude_self = FALSE
) {
    if (missing(metric)) {
        metric <- "euclidean"
    }
    prepared <- prepare_candidate_knn_inputs(
        data,
        points,
        candidates,
        k,
        exclude_self
    )
    backend <- normalize_public_backend_arg(backend)
    metric <- normalize_nn_metric(metric)
    n_threads <- normalize_nn_threads(n_threads)

    if (identical(backend, "cuda")) {
        return(candidate_knn_cuda(prepared, metric))
    }
    candidate_knn_cpu(prepared, metric, n_threads)
}

prepare_candidate_knn_inputs <- function(
    data,
    points,
    candidates,
    k,
    exclude_self
) {
    exclude_self <- normalize_scalar_logical_arg(
        exclude_self,
        "exclude_self",
        default = FALSE
    )
    matrices <- prepare_candidate_matrices(data, points, candidates)
    x <- matrices$x
    q <- matrices$q
    cand <- matrices$candidates
    k_message <- "`k` must be an integer in [1, ncol(candidates)]."
    k <- normalize_nn_positive_integer(k, "k", k_message)
    if (k > ncol(cand)) {
        stop(k_message, call. = FALSE)
    }
    self_query <- identical(dim(x), dim(q)) && identical(x, q)
    if (exclude_self && !self_query) {
        stop(
            "`exclude_self = TRUE` requires `points` to be `data`.",
            call. = FALSE
        )
    }
    if (exclude_self) {
        cand[cand == row(cand)] <- NA_integer_
    }
    list(
        x = x,
        q = q,
        cand = cand,
        k = k,
        exclude_self = exclude_self,
        self_query = self_query
    )
}

prepare_candidate_matrices <- function(data, points, candidates) {
    validate_dense_matrix_input(data, "data")
    validate_dense_matrix_input(points, "points")
    validate_dense_matrix_input(candidates, "candidates")
    x <- as.matrix(data)
    q <- as.matrix(points)
    storage.mode(x) <- storage.mode(q) <- "double"
    valid_dims <- nrow(x) > 0L && ncol(x) > 0L &&
        nrow(q) > 0L && ncol(q) == ncol(x)
    if (!valid_dims) {
        stop(
            "`data` and `points` must have compatible positive dimensions.",
            call. = FALSE
        )
    }
    if (!all(is.finite(x)) || !all(is.finite(q))) {
        stop(
            "`data` and `points` must contain only finite values.",
            call. = FALSE
        )
    }
    candidates <- as.matrix(candidates)
    storage.mode(candidates) <- "integer"
    if (nrow(candidates) != nrow(q) || ncol(candidates) < 1L) {
        stop(
            "`candidates` must have one row per query and at least one column.",
            call. = FALSE
        )
    }
    list(x = x, q = q, candidates = candidates)
}

candidate_knn_cuda <- function(input, metric) {
    validate_candidate_knn_cuda(input)
    normalized <- metric %in% c("cosine", "correlation")
    metric_inputs <- if (normalized) {
        normalized_euclidean_metric_inputs(
            input$x,
            input$q,
            input$self_query,
            metric
        )
    }
    if (normalized) {
        reject_cuda_normalized_cpu_repair(
            "cuda",
            metric,
            metric_inputs$data_zero,
            metric_inputs$points_zero,
            "cuda_candidate"
        )
    }
    search_x <- if (normalized) metric_inputs$data else input$x
    out <- row_candidate_knn_cuda_cpp(
        search_x,
        input$cand,
        as.integer(input$k),
        "euclidean",
        fill_missing = FALSE
    )
    result <- finish_nn_result(
        out,
        "cuda_candidate",
        input$k,
        TRUE,
        exact = FALSE,
        metric = metric
    )
    attr(result, "exclude_self") <- input$exclude_self
    if (normalized) {
        result <- finalize_normalized_euclidean_metric_result(
            result,
            metric_inputs
        )
    }
    attr(result, "candidate_knn") <- candidate_knn_metadata(
        input,
        transform = if (normalized) metric_inputs$transform
    )
    result
}

validate_candidate_knn_cuda <- function(input) {
    if (!input$exclude_self) {
        stop(
            "CUDA candidate KNN currently requires `exclude_self = TRUE`.",
            call. = FALSE
        )
    }
    if (!input$self_query) {
        stop(
            "CUDA candidate KNN currently requires self-query candidates.",
            call. = FALSE
        )
    }
    if (!isTRUE(cuda_available())) {
        stop("No CUDA GPU backend is available on this machine.", call. = FALSE)
    }
    invisible(TRUE)
}

candidate_knn_cpu <- function(input, metric, n_threads) {
    out <- candidate_knn_cpp(
        input$x,
        input$q,
        input$cand,
        as.integer(input$k),
        metric,
        FALSE,
        input$exclude_self,
        TRUE,
        as.integer(n_threads)
    )
    result <- finish_nn_result(
        out,
        "cpu_candidate",
        input$k,
        input$self_query,
        exact = FALSE,
        metric = metric
    )
    attr(result, "exclude_self") <- input$exclude_self
    attr(result, "candidate_knn") <- candidate_knn_metadata(
        input,
        n_threads = as.integer(out$n_threads)
    )
    result
}

candidate_knn_metadata <- function(input, n_threads = NULL, transform = NULL) {
    metadata <- list(
        candidate_columns = as.integer(ncol(input$cand)),
        exclude_self = input$exclude_self,
        exact_within_candidates = TRUE
    )
    if (!is.null(n_threads)) {
        metadata$n_threads <- n_threads
    }
    if (!is.null(transform)) {
        metadata$cuda_metric <- "euclidean"
        metadata$transform <- transform
    }
    metadata
}
