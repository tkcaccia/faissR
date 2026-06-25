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

knn_has_self_column <- function(indices, distances) {
  strip_self_neighbors_cpp(indices, distances)$has_self
}

prepend_self_neighbor_column <- function(out) {
  if (!is.list(out) || is.null(out$indices) || is.null(out$distances)) {
    stop("KNN result must contain `indices` and `distances`.", call. = FALSE)
  }
  distances_are_float32 <- inherits(out$distances, "float32")
  distances <- if (isTRUE(distances_are_float32)) {
    float32_to_numeric_matrix(out$distances, "distances")
  } else {
    out$distances
  }
  prepended <- prepend_self_neighbors_cpp(out$indices, distances)
  out$indices <- prepended$indices
  out$distances <- if (isTRUE(distances_are_float32)) {
    as_float_distances(prepended$distances)
  } else {
    prepended$distances
  }
  out
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
  attach_gpu_residency_metadata(out, out)
}

nn_gpu_residency_metadata <- function(out) {
  if (!is.list(out)) return(NULL)
  fields <- c(
    "accelerator", "gpu_provider", "device_residency", "index_residency",
    "gpu_index_resident", "gpu_index_persistent",
    "host_to_device_transfer_strategy", "host_to_device_copies_known",
    "host_to_device_data_copies", "host_to_device_query_copies",
    "host_to_device_copies", "host_to_device_data_copies_minimum",
    "host_to_device_query_copies_minimum", "host_to_device_data_bytes",
    "host_to_device_query_bytes", "host_to_device_bytes",
    "host_to_device_data_bytes_minimum",
    "host_to_device_query_bytes_minimum", "host_to_device_bytes_minimum",
    "query_reuses_host_data", "query_reuses_device_data",
    "query_residency", "result_residency",
    "device_to_host_result_copies_known", "device_to_host_result_copies",
    "device_to_host_result_bytes", "device_to_host_result_bytes_minimum",
    "cpu_fallback", "cpu_side_result_repair"
  )
  fields <- fields[fields %in% names(out)]
  if (!length(fields)) return(NULL)
  out[fields]
}

attach_gpu_residency_metadata <- function(result, out) {
  gpu <- nn_gpu_residency_metadata(out)
  if (!length(gpu)) return(result)
  result$gpu_residency <- gpu
  attr(result, "gpu_residency") <- gpu
  for (attr_name in c("faiss", "cuvs", "approximation")) {
    metadata <- attr(result, attr_name, exact = TRUE)
    if (is.list(metadata)) {
      metadata <- c(metadata, gpu[setdiff(names(gpu), names(metadata))])
      attr(result, attr_name) <- metadata
    }
  }
  result
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
  if ((is.null(d) || length(d) != 2L) && isS4(x)) {
    payload <- tryCatch(methods::slot(x, "Data"), error = function(e) NULL)
    if (!is.null(payload)) {
      d <- dim(payload)
    }
  }
  if (is.null(d) || length(d) != 2L) {
    stop("`", arg_name, "` must be a two-dimensional float::fl()/float32 matrix.", call. = FALSE)
  }
  d <- as.integer(d)
  if (anyNA(d) || any(d < 1L)) {
    stop("`", arg_name, "` must have at least one row and one column.", call. = FALSE)
  }
  d
}

float32_to_numeric_matrix <- function(x, arg_name = "data") {
  if (!is_float32_matrix_input(x)) {
    x <- as.matrix(x)
    storage.mode(x) <- "double"
    return(x)
  }
  if (!requireNamespace("float", quietly = TRUE)) {
    stop(
      "`", arg_name, "` is a float32 object but the optional float package ",
      "is not installed.",
      call. = FALSE
    )
  }
  out <- float::dbl(x)
  out <- as.matrix(out)
  storage.mode(out) <- "double"
  out
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

.faissR_transformed_float32_cache <- new.env(parent = emptyenv())
.faissR_transformed_float32_cache$.keys <- character()

transformed_float32_cache_enabled <- function() {
  isTRUE(faissr_option("cache_transformed_float32", TRUE)) &&
    requireNamespace("float", quietly = TRUE)
}

transformed_float32_cache_limit <- function() {
  value <- suppressWarnings(as.integer(faissr_option("cache_transformed_float32_max_entries", 4L)))
  if (length(value) != 1L || is.na(value) || !is.finite(value) || value < 0L) {
    return(4L)
  }
  value
}

transformed_float32_cache_transform_label <- function(metric) {
  if (identical(metric, "correlation")) {
    "row_center_l2_normalize_float32_cached"
  } else {
    "row_l2_normalize_float32_cached"
  }
}

transformed_float32_cache_prune <- function() {
  limit <- transformed_float32_cache_limit()
  keys <- .faissR_transformed_float32_cache$.keys
  if (limit < 1L) {
    rm(list = setdiff(ls(.faissR_transformed_float32_cache, all.names = TRUE), ".keys"),
       envir = .faissR_transformed_float32_cache)
    .faissR_transformed_float32_cache$.keys <- character()
    return(invisible(NULL))
  }
  while (length(keys) > limit) {
    old <- keys[[1L]]
    if (exists(old, envir = .faissR_transformed_float32_cache, inherits = FALSE)) {
      rm(list = old, envir = .faissR_transformed_float32_cache)
    }
    keys <- keys[-1L]
  }
  .faissR_transformed_float32_cache$.keys <- keys
  invisible(NULL)
}

normalized_float32_transform_cached <- function(x, metric, role = "data") {
  metric <- normalize_nn_metric(metric)
  if (!metric %in% c("cosine", "correlation")) {
    stop("Cached normalized float32 transforms require cosine or correlation.", call. = FALSE)
  }
  if (!requireNamespace("float", quietly = TRUE)) {
    return(NULL)
  }

  dims <- if (is_float32_matrix_input(x)) float32_matrix_dims(x, role) else dim(x)
  fingerprint <- matrix_fingerprint_cpp(x)
  key <- paste(metric, paste(as.integer(dims), collapse = "x"), fingerprint, sep = ":")
  cache_enabled <- transformed_float32_cache_enabled()
  if (isTRUE(cache_enabled) &&
      exists(key, envir = .faissR_transformed_float32_cache, inherits = FALSE)) {
    entry <- get(key, envir = .faissR_transformed_float32_cache, inherits = FALSE)
    entry$cache_hit <- TRUE
    entry$role <- role
    return(entry)
  }

  transformed <- normalized_float32_transform_cpp(x, metric)
  entry <- list(
    data = transformed$data,
    zero = as.logical(transformed$zero),
    transform = transformed_float32_cache_transform_label(metric),
    storage = "float32",
    row_major = TRUE,
    cache_key = key,
    fingerprint = fingerprint,
    cache_hit = FALSE,
    cache_enabled = isTRUE(cache_enabled),
    role = role
  )

  if (isTRUE(cache_enabled) && transformed_float32_cache_limit() > 0L) {
    assign(key, entry, envir = .faissR_transformed_float32_cache)
    keys <- .faissR_transformed_float32_cache$.keys
    .faissR_transformed_float32_cache$.keys <- c(setdiff(keys, key), key)
    transformed_float32_cache_prune()
  }
  entry
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
