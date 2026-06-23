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
#'   canonical metric labels. Correlation is centered cosine similarity, not
#'   raw inner product; see \code{\link{nn}()} for the metric transforms and
#'   backend support matrix.
#' @param tuning Tuning policy passed to \code{\link{nn}()}. `"auto"` uses the
#'   deterministic default for the resolved method; pilot/cache tuning is
#'   opt-in where implemented. FAISS GPU IVF pilot/cache tuning is
#'   Euclidean-only; non-Euclidean IVF routes use deterministic metric-aware
#'   defaults.
#' @param cagra_implementation CUDA CAGRA provider passed to \code{\link{nn}()}
#'   for `method = "cagra"` or CUDA-auto routes that select CAGRA. `NULL` uses
#'   the global `faissR.cagra_implementation` option; `"auto"` uses the same
#'   deterministic shape-aware provider rule as \code{\link{nn}()}, while
#'   `"faiss_gpu"` or `"cuvs"` force one provider.
#' @param cagra_build_algo Direct RAPIDS cuVS CAGRA graph-build algorithm passed
#'   to \code{\link{nn}()} for direct cuVS CAGRA routes. `NULL` uses the global
#'   `faissR.cuvs_cagra_build_algo` option.
#' @param task `"auto"`, `"classification"`, or `"regression"`. `"auto"` treats
#'   numeric responses as regression and other response types as classification.
#' @param k Default number of neighbours used by `predict()` and by immediate
#'   predictions when `Xtest` is supplied.
#' @param n_threads CPU threads passed to \code{\link{nn}()} for CPU backends.
#' @param vote `"majority"` or `"weighted"` for immediate predictions.
#' @param type `"response"` for class/regression predictions or `"prob"` for
#'   class probability matrices from classification models.
#' @param ... Reserved for future options.
#' @return If `Xtest` is not supplied, a fitted `faissR_knn_model` object that
#'   stores the training data, response, task, backend, method, metric, tuning,
#'   `k`, and CPU thread settings used by later \code{\link{predict}()} calls.
#'   If `Xtest` is supplied, a factor for classification, a numeric vector for
#'   regression, or a numeric class-probability matrix when `type = "prob"`;
#'   prediction outputs carry `attr(result, "faissR_nn")` route metadata,
#'   approximation parameters, and auto-selection metadata from the underlying
#'   \code{\link{nn}()} call.
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
                method = c("auto", "exact", "flat", "bruteforce", "grid", "hnsw", "ivf", "ivfpq", "vamana", "nsg", "nndescent", "cagra"),
                metric = c("euclidean", "cosine", "correlation", "inner_product"),
                tuning = c("auto", "cache", "pilot", "fixed", "off", "none"),
                cagra_implementation = NULL,
                cagra_build_algo = NULL,
                task = c("auto", "classification", "regression"),
                k = 15L,
                n_threads = NULL,
                vote = c("majority", "weighted"),
                type = c("response", "prob"),
                ...) {
  vote <- normalize_knn_vote(vote)
  type <- normalize_knn_type(type)
  backend <- normalize_public_backend_arg(backend)
  method <- normalize_nn_method(method)
  tuning <- normalize_nn_tuning(tuning)
  model <- knn_model_fit(
    Xtrain = Xtrain,
    Ytrain = Ytrain,
    backend = backend,
    method = method,
    metric = metric,
    tuning = tuning,
    cagra_implementation = cagra_implementation,
    cagra_build_algo = cagra_build_algo,
    task = task,
    k = k,
    n_threads = n_threads
  )
  if (is.null(Xtest)) {
    return(model)
  }
  predict(
    model,
    Xtest,
    k = k,
    backend = backend,
    tuning = tuning,
    cagra_implementation = cagra_implementation,
    cagra_build_algo = cagra_build_algo,
    vote = vote,
    type = type,
    ...
  )
}

knn_model_fit <- function(Xtrain,
                          Ytrain,
                          backend = c("auto", "cpu", "cuda"),
                          method = c("auto", "exact", "flat", "bruteforce", "grid", "hnsw", "ivf", "ivfpq", "vamana", "nsg", "nndescent", "cagra"),
                          metric = c("euclidean", "cosine", "correlation", "inner_product"),
                          tuning = c("auto", "cache", "pilot", "fixed", "off", "none"),
                          cagra_implementation = NULL,
                          cagra_build_algo = NULL,
                          task = c("auto", "classification", "regression"),
                          k = 15L,
                          n_threads = NULL) {
  backend <- normalize_public_backend_arg(backend)
  method <- normalize_nn_method(method)
  tuning <- normalize_nn_tuning(tuning)
  metric <- normalize_nn_metric(metric)
  cagra_implementation <- normalize_cagra_implementation_arg(cagra_implementation)
  cagra_build_algo <- normalize_cagra_build_algo_arg(cagra_build_algo)
  task <- normalize_knn_task(task)
  x <- as.matrix(Xtrain)
  storage.mode(x) <- "double"
  if (nrow(x) < 1L || ncol(x) < 1L) {
    stop("`Xtrain` must have at least one row and one column.", call. = FALSE)
  }
  if (!all(is.finite(x))) {
    stop("`Xtrain` must contain only finite values.", call. = FALSE)
  }
  validate_public_nn_method_shape(x, method)
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
      method = method,
      tuning = tuning,
      cagra_implementation = cagra_implementation,
      cagra_build_algo = cagra_build_algo,
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
#'   `"cpu"`, or `"cuda"`. The fitted model's method and metric are always
#'   reused.
#' @param tuning Tuning policy used for this prediction call. `"auto"` uses the
#'   deterministic default for the resolved method; pilot/cache tuning is
#'   opt-in where implemented. FAISS GPU IVF pilot/cache tuning is
#'   Euclidean-only; non-Euclidean IVF routes use deterministic metric-aware
#'   defaults.
#' @param cagra_implementation CUDA CAGRA provider for this prediction call.
#'   `NULL` reuses the fitted model's setting, then the global option.
#' @param cagra_build_algo Direct RAPIDS cuVS CAGRA graph-build algorithm for
#'   this prediction call. `NULL` reuses the fitted model's setting, then the
#'   global option.
#' @param vote `"majority"` or `"weighted"` for classification; `"majority"`
#'   means an unweighted neighbour mean for regression.
#' @param type `"response"` for class/regression predictions or `"prob"` for
#'   class probability matrices from classification models.
#' @param ... Reserved for future options.
#' @return A factor for classification, a numeric vector for regression, or a
#'   numeric class-probability matrix when `type = "prob"`. Outputs carry
#'   `attr(result, "faissR_nn")` route metadata, approximation parameters, and
#'   auto-selection metadata from the underlying \code{\link{nn}()} call.
#' @export
predict.faissR_knn_model <- function(object,
                                      newdata,
                                      k = NULL,
                                      backend = c("auto", "cpu", "cuda"),
                                      tuning = c("auto", "cache", "pilot", "fixed", "off", "none"),
                                      cagra_implementation = NULL,
                                      cagra_build_algo = NULL,
                                      vote = c("majority", "weighted"),
                                      type = c("response", "prob"),
                                      ...) {
  backend <- normalize_public_backend_arg(backend)
  tuning <- normalize_nn_tuning(tuning)
  if (is.null(cagra_implementation)) {
    cagra_implementation <- object$cagra_implementation %||% NULL
  }
  if (is.null(cagra_build_algo)) {
    cagra_build_algo <- object$cagra_build_algo %||% NULL
  }
  cagra_implementation <- normalize_cagra_implementation_arg(cagra_implementation)
  cagra_build_algo <- normalize_cagra_build_algo_arg(cagra_build_algo)
  vote <- normalize_knn_vote(vote)
  type <- normalize_knn_type(type)
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
    cagra_implementation = cagra_implementation,
    cagra_build_algo = cagra_build_algo,
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
    proba <- attach_knn_prediction_metadata(
      proba,
      neighbours,
      k,
      backend,
      object$method %||% "auto",
      tuning,
      cagra_implementation,
      cagra_build_algo
    )
    if (identical(type, "prob")) {
      return(proba)
    }
    best <- max.col(proba, ties.method = "first")
    pred <- factor(object$levels[best], levels = object$levels)
    return(attach_knn_prediction_metadata(
      pred,
      neighbours,
      k,
      backend,
      object$method %||% "auto",
      tuning,
      cagra_implementation,
      cagra_build_algo
    ))
  }
  if (identical(type, "prob")) {
    stop("`type = \"prob\"` is only available for classification models.", call. = FALSE)
  }
  pred <- regression_vote(
    response = response,
    indices = neighbours$indices,
    distances = neighbours$distances,
    weighted = identical(vote, "weighted")
  )
  attach_knn_prediction_metadata(
    pred,
    neighbours,
    k,
    backend,
    object$method %||% "auto",
    tuning,
    cagra_implementation,
    cagra_build_algo
  )
}

attach_knn_prediction_metadata <- function(out, neighbours, k, backend, method, tuning,
                                           cagra_implementation = NULL,
                                           cagra_build_algo = NULL) {
  distance_type <- neighbours$distance_type %||% NA_character_
  if (is.na(distance_type)) {
    distance_type <- if (inherits(neighbours$distances, "float32") ||
                         inherits(neighbours$distances, "fl")) {
      "float32"
    } else {
      "double"
    }
  }
  attr(out, "faissR_nn") <- list(
    k = as.integer(k),
    requested_backend = attr(neighbours, "requested_backend") %||% backend,
    requested_method = attr(neighbours, "requested_method") %||% public_nn_method_label(normalize_nn_method(method)),
    tuning = attr(neighbours, "tuning") %||% tuning,
    cagra_implementation = cagra_implementation %||% NA_character_,
    cagra_build_algo = cagra_build_algo %||% NA_character_,
    backend = attr(neighbours, "backend") %||% NA_character_,
    resolved_backend = attr(neighbours, "resolved_backend") %||% attr(neighbours, "backend") %||% NA_character_,
    metric = attr(neighbours, "metric") %||% NA_character_,
    exact = attr(neighbours, "exact") %||% NA,
    approximation = attr(neighbours, "approximation") %||% NULL,
    faiss = attr(neighbours, "faiss") %||% NULL,
    cuvs = attr(neighbours, "cuvs") %||% NULL,
    spatial_index = attr(neighbours, "spatial_index") %||% NULL,
    auto_selection = attr(neighbours, "auto_selection") %||% NULL,
    metric_transform = attr(neighbours, "metric_transform") %||% neighbours$metric_transform %||% NULL,
    distance_transform = attr(neighbours, "distance_transform") %||% NULL,
    distance_type = distance_type
  )
  out
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
  k <- suppressWarnings(as.numeric(k))
  if (length(k) != 1L || is.na(k) || !is.finite(k) || k < 1L ||
      abs(k - round(k)) > sqrt(.Machine$double.eps)) {
    stop("`k` must be a positive integer.", call. = FALSE)
  }
  k <- as.integer(round(k))
  if (k > n_train) {
    stop("`k` cannot be larger than the number of training rows.", call. = FALSE)
  }
  k
}

normalize_knn_task <- function(task) {
  task <- normalize_scalar_choice_arg(
    task,
    arg = "task",
    default = "auto",
    formal_choices = c("auto", "classification", "regression")
  )
  if (is.na(task) || !nzchar(task)) task <- "auto"
  task <- trimws(task)
  if (!task %in% c("auto", "classification", "regression")) {
    stop("`task` must be one of \"auto\", \"classification\", or \"regression\".", call. = FALSE)
  }
  task
}

normalize_knn_vote <- function(vote) {
  vote <- normalize_scalar_choice_arg(
    vote,
    arg = "vote",
    default = "majority",
    formal_choices = c("majority", "weighted")
  )
  if (is.na(vote) || !nzchar(vote)) vote <- "majority"
  vote <- trimws(vote)
  if (!vote %in% c("majority", "weighted")) {
    stop("`vote` must be one of \"majority\" or \"weighted\".", call. = FALSE)
  }
  vote
}

normalize_knn_type <- function(type) {
  type <- normalize_scalar_choice_arg(
    type,
    arg = "type",
    default = "response",
    formal_choices = c("response", "prob")
  )
  if (is.na(type) || !nzchar(type)) type <- "response"
  type <- trimws(type)
  if (!type %in% c("response", "prob")) {
    stop("`type` must be one of \"response\" or \"prob\".", call. = FALSE)
  }
  type
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
