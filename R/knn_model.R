#' Fit a k-nearest-neighbour classifier or regressor
#'
#' `knn_fit()` stores a reference matrix and response vector, then uses [nn()]
#' during prediction. This keeps the model API small while still using the
#' package's FAISS, cuVS/CUDA, HNSW, grid, VP-tree, or exact CPU neighbour
#' backends. Classification supports majority or inverse-distance weighted
#' votes. Regression supports ordinary or inverse-distance weighted neighbour
#' averages.
#'
#' @param X_train Numeric training matrix with observations in rows.
#' @param y_train Training labels or numeric response.
#' @param backend Neighbour backend passed to [nn()]. Use `"faiss"` for FAISS
#'   CPU search or `"cuda"`/`"cuda_cuvs"` for CUDA/cuVS when available.
#' @param metric Distance metric passed to [nn()].
#' @param task `"auto"`, `"classification"`, or `"regression"`. `"auto"` treats
#'   numeric responses as regression and other response types as classification.
#' @param k Default number of neighbours used by [predict()] and
#'   [predict()] with `type = "prob"` when they are called without `k`.
#' @param n_threads CPU threads passed to [nn()] for CPU backends.
#' @param ... Additional arguments passed to [knn_fit()] by the convenience
#'   wrappers [faiss.fit()] and [cuvs.fit()].
#' @return A fitted `faissR_knn_model` object.
#' @examples
#' x <- scale(as.matrix(iris[, 1:4]))
#' clf <- knn_fit(x, iris$Species, backend = "cpu", k = 5)
#' head(predict(clf, x, k = 5))
#' head(predict(clf, x, k = 5, type = "prob"))
#'
#' reg <- knn_fit(x, iris$Sepal.Length, backend = "cpu", task = "regression")
#' head(predict(reg, x, k = 5, vote = "weighted"))
#' @export
knn_fit <- function(X_train,
                    y_train,
                    backend = "auto",
                    metric = c("euclidean", "cosine", "correlation"),
                    task = c("auto", "classification", "regression"),
                    k = 15L,
                    n_threads = NULL) {
  metric <- match.arg(metric)
  task <- match.arg(task)
  x <- as.matrix(X_train)
  storage.mode(x) <- "double"
  if (nrow(x) < 1L || ncol(x) < 1L) {
    stop("`X_train` must have at least one row and one column.", call. = FALSE)
  }
  if (!all(is.finite(x))) {
    stop("`X_train` must contain only finite values.", call. = FALSE)
  }
  if (NROW(y_train) != nrow(x)) {
    stop("`y_train` must have one value for each row of `X_train`.", call. = FALSE)
  }
  if (anyNA(y_train)) {
    stop("`y_train` must not contain missing values.", call. = FALSE)
  }
  k <- normalize_knn_model_k(k, nrow(x))
  n_threads <- normalize_nn_threads(n_threads)

  if (identical(task, "auto")) {
    task <- if (is.numeric(y_train)) "regression" else "classification"
  }
  if (identical(task, "regression")) {
    y <- as.numeric(y_train)
    if (!all(is.finite(y))) {
      stop("Regression `y_train` must contain only finite numeric values.", call. = FALSE)
    }
    levels <- NULL
  } else {
    y <- as.factor(y_train)
    levels <- levels(y)
  }

  structure(
    list(
      X_train = x,
      y_train = y,
      task = task,
      levels = levels,
      backend = as.character(backend)[1L],
      metric = metric,
      k = k,
      n_threads = n_threads
    ),
    class = c("faissR_knn_model", "fastEmbedR_knn_model")
  )
}

#' @rdname knn_fit
#' @export
faiss.fit <- function(X_train, y_train, ...) {
  knn_fit(X_train, y_train, backend = "faiss", ...)
}

#' @rdname knn_fit
#' @export
cuvs.fit <- function(X_train, y_train, ...) {
  knn_fit(X_train, y_train, backend = "cuda_cuvs", ...)
}

#' Predict from a faissR kNN model
#'
#' @param object A model returned by [knn_fit()], [faiss.fit()], or
#'   [cuvs.fit()].
#' @param newdata Numeric query matrix with observations in rows.
#' @param k Number of neighbours.
#' @param vote `"majority"` or `"weighted"` for classification; `"majority"`
#'   means an unweighted neighbour mean for regression.
#' @param type `"response"` for class/regression predictions or `"prob"` for
#'   class probability matrices from classification models.
#' @param ... Reserved for future options.
#' @return A factor for classification or numeric vector for regression.
#' @export
predict.faissR_knn_model <- function(object,
                                      newdata,
                                      k = NULL,
                                      vote = c("majority", "weighted"),
                                      type = c("response", "prob"),
                                      ...) {
  vote <- match.arg(vote)
  type <- match.arg(type)
  query <- validate_knn_model_query(object, newdata)
  k <- normalize_knn_model_k(if (is.null(k)) object$k else k, nrow(object$X_train))
  neighbours <- nn(
    object$X_train,
    query,
    k = k,
    backend = object$backend,
    metric = object$metric,
    n_threads = object$n_threads
  )
  if (identical(object$task, "classification")) {
    proba <- class_vote_probabilities(
      labels = object$y_train,
      indices = neighbours$indices,
      distances = neighbours$distances,
      levels = object$levels,
      weighted = identical(vote, "weighted")
    )
    if (identical(type, "prob")) {
      return(proba)
    }
    best <- max.col(proba, ties.method = "first")
    return(factor(object$levels[best], levels = object$levels))
  }
  if (identical(type, "prob")) {
    stop("`type = \"prob\"` is only available for classification models.", call. = FALSE)
  }
  regression_vote(
    response = object$y_train,
    indices = neighbours$indices,
    distances = neighbours$distances,
    weighted = identical(vote, "weighted")
  )
}

#' @export
predict.fastEmbedR_knn_model <- predict.faissR_knn_model

validate_knn_model_query <- function(object, newdata) {
  if (!inherits(object, "faissR_knn_model") && !inherits(object, "fastEmbedR_knn_model")) {
    stop("`object` must be a faissR kNN model.", call. = FALSE)
  }
  query <- as.matrix(newdata)
  storage.mode(query) <- "double"
  if (nrow(query) < 1L || ncol(query) != ncol(object$X_train)) {
    stop("`newdata` must have at least one row and the same columns as training data.", call. = FALSE)
  }
  if (!all(is.finite(query))) {
    stop("`newdata` must contain only finite values.", call. = FALSE)
  }
  query
}


normalize_knn_model_k <- function(k, n_train) {
  k <- suppressWarnings(as.integer(k))
  if (length(k) != 1L || is.na(k) || !is.finite(k) || k < 1L) {
    stop("`k` must be a positive integer.", call. = FALSE)
  }
  if (k > n_train) {
    stop("`k` cannot be larger than the number of training rows.", call. = FALSE)
  }
  k
}

knn_vote_weights <- function(distances, weighted) {
  if (!isTRUE(weighted)) {
    return(matrix(1, nrow(distances), ncol(distances)))
  }
  eps <- sqrt(.Machine$double.eps)
  weights <- 1 / pmax(distances, eps)
  zero <- distances <= eps
  if (any(zero)) {
    for (i in which(rowSums(zero) > 0L)) {
      weights[i, ] <- 0
      weights[i, zero[i, ]] <- 1
    }
  }
  weights
}

class_vote_probabilities <- function(labels, indices, distances, levels, weighted) {
  labels <- as.factor(labels)
  if (!identical(levels(labels), levels)) {
    labels <- factor(as.character(labels), levels = levels)
  }
  indices <- as.matrix(indices)
  distances <- as.matrix(distances)
  weights <- knn_vote_weights(distances, weighted)
  out <- matrix(0, nrow(indices), length(levels))
  colnames(out) <- levels
  label_codes <- as.integer(labels)
  for (i in seq_len(nrow(indices))) {
    codes <- label_codes[indices[i, ]]
    row_weight <- weights[i, ]
    for (j in seq_along(codes)) {
      out[i, codes[j]] <- out[i, codes[j]] + row_weight[j]
    }
  }
  denom <- rowSums(out)
  bad <- denom <= 0 | !is.finite(denom)
  if (any(bad)) {
    out[bad, ] <- 1 / ncol(out)
    denom[bad] <- 1
  }
  out / denom
}

regression_vote <- function(response, indices, distances, weighted) {
  response <- as.numeric(response)
  indices <- as.matrix(indices)
  distances <- as.matrix(distances)
  weights <- knn_vote_weights(distances, weighted)
  out <- numeric(nrow(indices))
  for (i in seq_len(nrow(indices))) {
    values <- response[indices[i, ]]
    row_weight <- weights[i, ]
    denom <- sum(row_weight)
    out[i] <- if (denom > 0 && is.finite(denom)) {
      sum(values * row_weight) / denom
    } else {
      mean(values)
    }
  }
  out
}
