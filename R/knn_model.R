#' Fit or apply a k-nearest-neighbour classifier or regressor
#'
#' `knn()` is the high-level supervised kNN API. With `Xtrain` and `Ytrain`
#' only, it returns a reusable model. With `Xtest`, it immediately predicts the
#' query rows by fitting the model and calling `predict()` internally. The
#' low-level nearest-neighbour search API remains `nn()`.
#'
#' @param Xtrain Numeric training matrix with observations in rows.
#' @param Ytrain Training labels or numeric response.
#' @param Xtest Optional numeric query matrix. If supplied, `knn()` returns
#'   predictions for `Xtest`; otherwise it returns a fitted model.
#' @param backend Device backend passed to \code{\link{nn}()}: `"auto"`, `"cpu"`, or
#'   `"cuda"`. `"auto"` follows \code{\link{nn}()} backend/method/metric resolution,
#'   using CUDA only for validated CUDA combinations when CUDA/cuVS runtime
#'   support is available, and CPU otherwise.
#' @param method Nearest-neighbour algorithm selector passed to
#'   \code{\link{nn}()}. See \code{\link{nn}()} for method descriptions and
#'   references.
#' @param metric Distance metric passed to \code{\link{nn}()}. Aliases such as
#'   `"l2"`, `"cor"`/`"pearson"`, and `"ip"` are accepted and stored as
#'   canonical metric labels.
#' @param tuning Tuning policy passed to \code{\link{nn}()}. `"auto"` uses the
#'   tuned default for the resolved method.
#' @param task `"auto"`, `"classification"`, or `"regression"`. `"auto"` treats
#'   numeric responses as regression and other response types as classification.
#' @param k Default number of neighbours used by `predict()` and by immediate
#'   predictions when `Xtest` is supplied.
#' @param n_threads CPU threads passed to \code{\link{nn}()} for CPU backends.
#' @param vote `"majority"` or `"weighted"` for immediate predictions.
#' @param type `"response"` for class/regression predictions or `"prob"` for
#'   class probability matrices from classification models.
#' @param ... Reserved for future options.
#' @return If `Xtest` is not supplied, a fitted `faissR_knn_model` object. If
#'   `Xtest` is supplied, a factor for classification, a numeric vector for
#'   regression, or a numeric class-probability matrix when `type = "prob"`.
#' @examples
#' x <- scale(as.matrix(iris[, 1:4]))
#' model <- knn(x, iris$Species, backend = "cpu", k = 5)
#' head(predict(model, x, k = 5))
#' head(predict(model, x, k = 5, type = "prob"))
#'
#' pred <- knn(x, iris$Species, x, backend = "cpu", k = 5)
#' head(pred)
#' @export
knn <- function(Xtrain,
                Ytrain,
                Xtest = NULL,
                backend = c("auto", "cpu", "cuda"),
                method = c("auto", "exact", "flat", "bruteforce", "grid", "vptree",
                           "sparse", "hnsw", "ivf", "ivfpq", "nsg", "nndescent", "cagra"),
                metric = c("euclidean", "cosine", "correlation", "inner_product"),
                tuning = c("auto", "cache", "pilot", "fixed", "off", "none"),
                task = c("auto", "classification", "regression"),
                k = 15L,
                n_threads = NULL,
                vote = c("majority", "weighted"),
                type = c("response", "prob"),
                ...) {
  vote <- match.arg(vote)
  type <- match.arg(type)
  backend <- normalize_public_backend_arg(backend)
  method <- as.character(method)[1L]
  tuning <- as.character(tuning)[1L]
  model <- knn_model_fit(
    Xtrain = Xtrain,
    Ytrain = Ytrain,
    backend = backend,
    method = method,
    metric = metric,
    tuning = tuning,
    task = task,
    k = k,
    n_threads = n_threads
  )
  if (is.null(Xtest)) {
    return(model)
  }
  predict(model, Xtest, k = k, backend = backend, tuning = tuning, vote = vote, type = type, ...)
}

knn_model_fit <- function(Xtrain,
                          Ytrain,
                          backend = c("auto", "cpu", "cuda"),
                          method = c("auto", "exact", "flat", "bruteforce", "grid", "vptree",
                                     "sparse", "hnsw", "ivf", "ivfpq", "nsg", "nndescent", "cagra"),
                          metric = c("euclidean", "cosine", "correlation", "inner_product"),
                          tuning = c("auto", "cache", "pilot", "fixed", "off", "none"),
                          task = c("auto", "classification", "regression"),
                          k = 15L,
                          n_threads = NULL) {
  backend <- normalize_public_backend_arg(backend)
  method <- as.character(method)[1L]
  tuning <- as.character(tuning)[1L]
  metric <- normalize_nn_metric(metric)
  task <- match.arg(task)
  x <- as.matrix(Xtrain)
  storage.mode(x) <- "double"
  if (nrow(x) < 1L || ncol(x) < 1L) {
    stop("`Xtrain` must have at least one row and one column.", call. = FALSE)
  }
  if (!all(is.finite(x))) {
    stop("`Xtrain` must contain only finite values.", call. = FALSE)
  }
  if (NROW(Ytrain) != nrow(x)) {
    stop("`Ytrain` must have one value for each row of `Xtrain`.", call. = FALSE)
  }
  if (anyNA(Ytrain)) {
    stop("`Ytrain` must not contain missing values.", call. = FALSE)
  }
  k <- normalize_knn_model_k(k, nrow(x))
  n_threads <- normalize_nn_threads(n_threads)

  if (identical(task, "auto")) {
    task <- if (is.numeric(Ytrain)) "regression" else "classification"
  }
  if (identical(task, "regression")) {
    y <- as.numeric(Ytrain)
    if (!all(is.finite(y))) {
      stop("Regression `Ytrain` must contain only finite numeric values.", call. = FALSE)
    }
    levels <- NULL
  } else {
    y <- as.factor(Ytrain)
    levels <- levels(y)
  }

  structure(
    list(
      Xtrain = x,
      Ytrain = y,
      task = task,
      levels = levels,
      backend = as.character(backend)[1L],
      method = as.character(method)[1L],
      tuning = as.character(tuning)[1L],
      metric = metric,
      k = k,
      n_threads = n_threads
    ),
    class = "faissR_knn_model"
  )
}

#' Predict from a faissR kNN model
#'
#' @param object A model returned by \code{\link{knn}()}.
#' @param newdata Numeric query matrix with observations in rows.
#' @param k Number of neighbours.
#' @param backend Device backend used for this prediction call: `"auto"`,
#'   `"cpu"`, or `"cuda"`. `NULL` reuses the backend stored in the fitted
#'   model. The fitted model's method and metric are always reused.
#' @param tuning Tuning policy used for this prediction call. `"auto"` uses the
#'   tuned default for the resolved method.
#' @param vote `"majority"` or `"weighted"` for classification; `"majority"`
#'   means an unweighted neighbour mean for regression.
#' @param type `"response"` for class/regression predictions or `"prob"` for
#'   class probability matrices from classification models.
#' @param ... Reserved for future options.
#' @return A factor for classification, a numeric vector for regression, or a
#'   numeric class-probability matrix when `type = "prob"`.
#' @export
predict.faissR_knn_model <- function(object,
                                      newdata,
                                      k = NULL,
                                      backend = NULL,
                                      tuning = c("auto", "cache", "pilot", "fixed", "off", "none"),
                                      vote = c("majority", "weighted"),
                                      type = c("response", "prob"),
                                      ...) {
  backend <- if (is.null(backend)) {
    object$backend %||% "auto"
  } else {
    normalize_public_backend_arg(backend)
  }
  tuning <- as.character(tuning)[1L]
  vote <- match.arg(vote)
  type <- match.arg(type)
  query <- validate_knn_model_query(object, newdata)
  train <- model_Xtrain(object)
  response <- model_Ytrain(object)
  k <- normalize_knn_model_k(if (is.null(k)) object$k else k, nrow(train))
  neighbours <- nn(
    train,
    query,
    k = k,
    backend = backend,
    method = object$method %||% "auto",
    metric = object$metric,
    tuning = tuning,
    n_threads = object$n_threads
  )
  if (identical(object$task, "classification")) {
    proba <- class_vote_probabilities(
      labels = response,
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
    response = response,
    indices = neighbours$indices,
    distances = neighbours$distances,
    weighted = identical(vote, "weighted")
  )
}

validate_knn_model_query <- function(object, newdata) {
  if (!inherits(object, "faissR_knn_model")) {
    stop("`object` must be a faissR kNN model.", call. = FALSE)
  }
  query <- as.matrix(newdata)
  storage.mode(query) <- "double"
  train <- model_Xtrain(object)
  if (nrow(query) < 1L || ncol(query) != ncol(train)) {
    stop("`newdata` must have at least one row and the same columns as training data.", call. = FALSE)
  }
  if (!all(is.finite(query))) {
    stop("`newdata` must contain only finite values.", call. = FALSE)
  }
  query
}

model_Xtrain <- function(object) {
  x <- object$Xtrain
  if (is.null(x)) x <- object$X_train
  if (is.null(x)) stop("`object` does not contain training data.", call. = FALSE)
  x
}

model_Ytrain <- function(object) {
  y <- object$Ytrain
  if (is.null(y)) y <- object$y_train
  if (is.null(y)) stop("`object` does not contain training responses.", call. = FALSE)
  y
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
