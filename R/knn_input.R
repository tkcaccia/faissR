coerce_knn_input <- function(indices,
                             distances = NULL,
                             arg_name = "indices") {
  input_backend <- NA_character_
  if (is.null(distances)) {
    if (!is.list(indices) || !all(c("indices", "distances") %in% names(indices))) {
      stop(
        "`distances` is required unless `", arg_name,
        "` is a list returned by `nn()` with `indices` and `distances`.",
        call. = FALSE
      )
    }
    input_backend <- attr(indices, "resolved_backend") %||% attr(indices, "backend")
    distances <- indices$distances
    indices <- indices$indices
  }

  if (!is.matrix(indices)) indices <- as.matrix(indices)
  if (!is.matrix(distances)) distances <- as.matrix(distances)
  if (!is.integer(indices)) storage.mode(indices) <- "integer"
  if (!identical(typeof(distances), "double")) storage.mode(distances) <- "double"

  if (!identical(dim(indices), dim(distances))) {
    stop("KNN `indices` and `distances` must have the same dimensions.", call. = FALSE)
  }
  if (nrow(indices) < 2L || ncol(indices) < 1L) {
    stop("KNN input must have at least two rows and one neighbor column.", call. = FALSE)
  }

  stripped <- strip_self_neighbors_cpp(indices, distances)
  indices <- stripped$indices
  distances <- stripped$distances
  has_self <- isTRUE(stripped$has_self)
  col_start <- stripped$col_start
  n_neighbors <- stripped$n_neighbors
  if (n_neighbors < 1L) {
    stop("KNN input must contain at least one non-self neighbor.", call. = FALSE)
  }

  list(
    indices = indices,
    distances = distances,
    has_self = has_self,
    col_start = as.integer(col_start),
    n_neighbors = as.integer(n_neighbors),
    materialized = isTRUE(stripped$materialized),
    input_backend = if (is.null(input_backend)) NA_character_ else input_backend
  )
}

is_knn_input <- function(x) {
  is.list(x) && all(c("indices", "distances") %in% names(x))
}

knn_index_base <- function(indices, n = nrow(indices)) {
  min_idx <- suppressWarnings(min(indices, na.rm = TRUE))
  max_idx <- suppressWarnings(max(indices, na.rm = TRUE))
  if (is.finite(min_idx) && is.finite(max_idx) && min_idx >= 1L && max_idx <= n) {
    return("one")
  }
  "zero"
}

strip_self_neighbors <- function(indices, distances) {
  if (ncol(indices) < 1L) {
    return(list(
      indices = indices, distances = distances, has_self = FALSE,
      col_start = 0L, n_neighbors = 0L, materialized = FALSE
    ))
  }
  n <- nrow(indices)
  k <- ncol(indices)
  expected <- if (identical(knn_index_base(indices, n), "one")) seq_len(n) else seq_len(n) - 1L
  tolerance <- max(sqrt(.Machine$double.eps), 1e-12)

  first_self <- indices[, 1L] == expected & distances[, 1L] <= tolerance
  if (all(first_self)) {
    return(list(
      indices = indices,
      distances = distances,
      has_self = TRUE,
      col_start = 1L,
      n_neighbors = as.integer(k - 1L),
      materialized = FALSE
    ))
  }

  self_mask <- indices == expected & distances <= tolerance
  has_self <- all(rowSums(self_mask) > 0L)
  if (!has_self) {
    return(list(
      indices = indices, distances = distances, has_self = FALSE,
      col_start = 0L, n_neighbors = as.integer(k), materialized = FALSE
    ))
  }
  if (k == 1L) {
    return(list(
      indices = indices[, 0L, drop = FALSE],
      distances = distances[, 0L, drop = FALSE],
      has_self = TRUE,
      col_start = 0L,
      n_neighbors = 0L,
      materialized = TRUE
    ))
  }
  out_indices <- matrix(0L, nrow = n, ncol = k - 1L)
  out_distances <- matrix(0, nrow = n, ncol = k - 1L)
  storage.mode(out_indices) <- "integer"
  storage.mode(out_distances) <- "double"
  cols <- seq_len(k)
  for (i in seq_len(n)) {
    self_pos <- which(self_mask[i, ])[1L]
    keep <- cols[-self_pos]
    out_indices[i, ] <- indices[i, keep]
    out_distances[i, ] <- distances[i, keep]
  }
  list(
    indices = out_indices,
    distances = out_distances,
    has_self = TRUE,
    col_start = 0L,
    n_neighbors = as.integer(k - 1L),
    materialized = TRUE
  )
}

materialize_knn_range <- function(indices, distances, col_start = 0L, n_neighbors = ncol(indices) - col_start) {
  col_start <- as.integer(col_start)
  n_neighbors <- as.integer(n_neighbors)
  if (col_start == 0L && n_neighbors == ncol(indices)) {
    return(list(indices = indices, distances = distances))
  }
  cols <- seq.int(col_start + 1L, length.out = n_neighbors)
  list(
    indices = indices[, cols, drop = FALSE],
    distances = distances[, cols, drop = FALSE]
  )
}

knn_has_self_column <- function(indices, distances) {
  strip_self_neighbors(indices, distances)$has_self
}

set_embedding_colnames <- function(layout, prefix) {
  if (!is.matrix(layout)) layout <- as.matrix(layout)
  colnames(layout) <- paste0(prefix, seq_len(ncol(layout)))
  layout
}

validate_n_components <- function(n_components) {
  n_components <- as.integer(n_components)
  if (length(n_components) != 1L || is.na(n_components) || !is.finite(n_components) || n_components < 1L) {
    stop("`n_components` must be a positive integer.", call. = FALSE)
  }
  n_components
}

finish_nn_result <- function(out,
                             backend,
                             k,
                             self_query,
                             exact = TRUE,
                             metric = "euclidean") {
  out$index_base <- out$index_base %||% 1L
  out$metric <- metric
  out$backend_used <- out$backend_used %||% backend
  attr(out, "backend") <- backend
  attr(out, "resolved_backend") <- backend
  attr(out, "k") <- as.integer(k)
  attr(out, "self_query") <- isTRUE(self_query)
  attr(out, "exact") <- isTRUE(exact)
  attr(out, "metric") <- metric
  attr(out, "index_base") <- as.integer(out$index_base)
  attr(out, "backend_used") <- out$backend_used
  class(out) <- c("faissR_nn", "list")
  out
}

normalize_nn_output <- function(output) {
  output <- normalize_scalar_choice_arg(
    output,
    arg = "output",
    default = "double",
    formal_choices = c("double", "float")
  )
  if (is.na(output) || !nzchar(output)) output <- "double"
  output <- tolower(trimws(output))
  if (!output %in% c("double", "float")) {
    stop("`output` must be one of \"double\" or \"float\".", call. = FALSE)
  }
  output
}

resolve_nn_output <- function(output, distances = NULL) {
  output <- normalize_nn_output(output)
  if (is.null(distances)) {
    return(output)
  }
  distances <- normalize_scalar_choice_arg(
    distances,
    arg = "distances",
    default = "double",
    formal_choices = c("double", "float")
  )
  if (is.na(distances) || !nzchar(distances)) distances <- "double"
  distances <- tolower(trimws(distances))
  if (!distances %in% c("double", "float")) {
    stop("`distances` must be one of \"double\" or \"float\".", call. = FALSE)
  }
  if (!identical(output, "double") && !identical(output, distances)) {
    stop(
      "`output` and `distances` request different distance storage types.",
      call. = FALSE
    )
  }
  distances
}

is_float32_matrix_input <- function(x) {
  inherits(x, "float32")
}

float32_matrix_dims <- function(x, arg_name = "data") {
  d <- dim(x)
  if (is.null(d) || length(d) != 2L) {
    stop("`", arg_name, "` must be a two-dimensional float::fl()/float32 matrix.", call. = FALSE)
  }
  d <- as.integer(d)
  if (anyNA(d) || any(d < 1L)) {
    stop("`", arg_name, "` must have at least one row and one column.", call. = FALSE)
  }
  d
}

as_float_distances <- function(x) {
  if (!requireNamespace("float", quietly = TRUE)) {
    stop(
      "`output = \"float\"` requires the optional float package. ",
      "Install it with `install.packages(\"float\")`, or use ",
      "`output = \"double\"`.",
      call. = FALSE
    )
  }
  float::fl(x)
}

finalize_nn_output <- function(result, output = "double") {
  output <- normalize_nn_output(output)
  public_metric <- attr(result, "metric") %||% result$metric %||% "euclidean"
  if (identical(output, "float")) {
    if (!inherits(result$distances, "float32")) {
      result$distances <- as_float_distances(result$distances)
    }
    result$distance_type <- "float32"
    attr(result, "distance_type") <- "float32"
  } else {
    result$distance_type <- "double"
    attr(result, "distance_type") <- "double"
  }
  result$index_base <- result$index_base %||% 1L
  result$metric <- public_metric
  result$backend_used <- result$backend_used %||%
    attr(result, "resolved_backend") %||%
    attr(result, "backend") %||%
    NA_character_
  attr(result, "index_base") <- as.integer(result$index_base)
  attr(result, "metric") <- public_metric
  attr(result, "backend_used") <- result$backend_used
  result
}
