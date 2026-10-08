#' Fit or apply a k-nearest-neighbour classifier or regressor
#'
#' `knn()` is the high-level supervised kNN API. With `Xtrain` and `Ytrain`
#' only, it returns a reusable model. With `Xtest`, it immediately predicts the
#' query rows by fitting the model and calling `predict()` internally. The
#' low-level nearest-neighbour search API remains `nn()`. For explicit CPU
#' FAISS-backed `method = "flat"`, `"hnsw"`, `"ivf"`, and `"ivfpq"` models, the
#' fitted object stores a session-local FAISS index and `predict()` reuses it
#' for
#' compatible prediction calls instead of rebuilding the index. For IVF/IVFPQ,
#' this reuses the trained centroids, inverted lists, and indexed vectors; a
#' prediction call may adjust the search-time `nprobe` for its requested `k`
#' without retraining. IVFPQ also reuses trained product-quantizer codebooks
#' and compressed codes.
#' Changing `target_recall` at prediction time bypasses an IVF/IVFPQ index
#' fitted for a different tier and performs a fresh route calculation with
#' the requested tier. The original fitted object is not modified. Inspect
#' `attr(prediction, "faissR_nn")$query_source` for `"fitted_index"` versus
#' `"nn"` and the effective parameters in `approximation`.
#'
#' @param Xtrain Dense numeric training matrix or optional
#'   `float::fl()`/`float32` matrix with observations in rows. Float32 inputs
#'   are preserved for \code{\link{nn}()} methods with direct float32 adapters.
#'   Sparse, delayed, file-backed, and other matrix-like objects are rejected
#'   before conversion.
#' @param Ytrain Training labels or numeric response.
#' @param Xtest Optional dense numeric or float32 query matrix. If supplied,
#'   `knn()` returns predictions for `Xtest`; otherwise it returns a fitted
#'   model.
#' @param backend Device backend passed to \code{\link{nn}()}: `"cpu"` or
#'   `"cuda"`. `NULL` follows the package backend configuration.
#' @param method Nearest-neighbour algorithm selector passed to
#'   \code{\link{nn}()}. See \code{\link{nn}()} for method descriptions and
#'   references.
#' @param metric Distance metric passed to \code{\link{nn}()}: `"euclidean"`,
#'   `"cosine"`, or `"correlation"`. Legacy metric aliases
#'   such as `"l2"`, `"cor"`, `"pearson"`, and `"ip"` are rejected.
#'   Correlation is centered cosine similarity; see
#'   \code{\link{nn}()} for the metric transforms and backend support matrix.
#' @param tuning Tuning policy passed to \code{\link{nn}()}. `"auto"` uses the
#'   deterministic default for the resolved method; pilot/cache tuning is
#'   opt-in where implemented. FAISS GPU IVF pilot/cache tuning is
#'   Euclidean-only; non-Euclidean IVF routes use deterministic metric-aware
#'   defaults.
#' @param target_recall Speed/recall tier passed to \code{\link{nn}()}.
#'   Use `0.9`, `0.95`, or `0.99`. CUDA `method = "auto"` uses it for
#'   Flat-vs-IVF selection, CUDA IVF uses it for probing defaults, and HNSW uses
#'   it for graph-search tiers. CUDA HNSW metadata records that the available
#'   cuVS route is a CAGRA-to-HNSW wrapper. For approximate methods this is a
#'   selection objective, not a guarantee; inspect the underlying NN metadata,
#'   including `tuning_benchmark_target_met`, and validate recall for the
#'   application.
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
#'   `k`, and CPU thread settings used by later
#'   \code{predict()} calls.
#'   Explicit CPU FAISS Flat/HNSW/IVF/IVFPQ models also store a session-local
#'   fitted index for compatible prediction calls; saved/reloaded models safely
#'   rebuild the same route when the external pointer is no longer valid. IVF
#'   metadata records whether trained centroids/inverted lists were reused and
#'   whether search used a query-specific `nprobe`; IVFPQ metadata also records
#'   product-quantizer codebook/code reuse.
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
knn <- function(
    Xtrain, Ytrain, Xtest = NULL,
    backend = NULL,
    method = NULL, metric = NULL,
    tuning = c("auto", "cache", "pilot", "fixed", "off", "none"),
    target_recall = 0.99,
    cagra_implementation = NULL, cagra_build_algo = NULL,
    task = c("auto", "classification", "regression"),
    k = 15L, n_threads = NULL,
    vote = c("majority", "weighted"),
    type = c("response", "prob"), ...
) {
    if (missing(method)) method <- "auto"
    if (missing(metric)) metric <- "euclidean"
    if (missing(tuning)) tuning <- "auto"
    args <- normalize_public_knn_args(
        backend,
        method,
        tuning,
        target_recall,
        vote,
        type
    )
    model <- fit_public_knn_model(
        Xtrain,
        Ytrain,
        metric,
        cagra_implementation,
        cagra_build_algo,
        task,
        k,
        n_threads,
        args
    )
    if (is.null(Xtest)) {
        return(model)
    }
    predict_public_knn_model(
        model,
        Xtest,
        k,
        cagra_implementation,
        cagra_build_algo,
        args,
        ...
    )
}

normalize_public_knn_args <- function(
    backend,
    method,
    tuning,
    target_recall,
    vote,
    type
) {
    list(
        backend = normalize_public_backend_arg(backend),
        method = normalize_nn_method(method),
        tuning = normalize_nn_tuning(tuning),
        target_recall = normalize_hnsw_target_recall(target_recall),
        vote = normalize_knn_vote(vote),
        type = normalize_knn_type(type)
    )
}

fit_public_knn_model <- function(
    Xtrain,
    Ytrain,
    metric,
    cagra_implementation,
    cagra_build_algo,
    task,
    k,
    n_threads,
    args
) {
    knn_model_fit(
        Xtrain = Xtrain,
        Ytrain = Ytrain,
        backend = args$backend,
        method = args$method,
        metric = metric,
        tuning = args$tuning,
        target_recall = args$target_recall,
        cagra_implementation = cagra_implementation,
        cagra_build_algo = cagra_build_algo,
        task = task,
        k = k,
        n_threads = n_threads
    )
}

predict_public_knn_model <- function(
    model,
    Xtest,
    k,
    cagra_implementation,
    cagra_build_algo,
    args,
    ...
) {
    predict(
        model,
        Xtest,
        k = k,
        backend = args$backend,
        tuning = args$tuning,
        target_recall = args$target_recall,
        cagra_implementation = cagra_implementation,
        cagra_build_algo = cagra_build_algo,
        vote = args$vote,
        type = args$type,
        ...
    )
}

knn_model_fit <- function(
    Xtrain, Ytrain,
    backend = NULL,
    method = NULL, metric = NULL, tuning = NULL,
    target_recall = 0.99,
    cagra_implementation = NULL, cagra_build_algo = NULL,
    task = c("auto", "classification", "regression"),
    k = 15L, n_threads = NULL
) {
    if (missing(method)) {
        method <- "auto"
    }
    if (missing(metric)) {
        metric <- "euclidean"
    }
    if (missing(tuning)) {
        tuning <- "auto"
    }
    args <- normalize_knn_model_fit_args(
        backend,
        method,
        metric,
        tuning,
        target_recall,
        cagra_implementation,
        cagra_build_algo,
        task,
        k,
        n_threads
    )
    x <- prepare_knn_model_matrix(Xtrain, "Xtrain")
    validate_public_nn_method_shape(x, args$method)
    response <- prepare_knn_model_response(Ytrain, x, args$task)
    args$task <- response$task
    args$k <- normalize_knn_model_k(args$k, nrow(x))
    index <- knn_build_fitted_nn_index(
        x = x,
        backend = args$backend,
        method = args$method,
        metric = args$metric,
        tuning = args$tuning,
        target_recall = args$target_recall,
        k = args$k,
        n_threads = args$n_threads
    )
    new_knn_model(x, response, args, index)
}

normalize_knn_model_fit_args <- function(
    backend,
    method,
    metric,
    tuning,
    target_recall,
    cagra_implementation,
    cagra_build_algo,
    task,
    k,
    n_threads
) {
    list(
        backend = normalize_public_backend_arg(backend),
        method = normalize_nn_method(method),
        metric = normalize_nn_metric(metric),
        tuning = normalize_nn_tuning(tuning),
        target_recall = normalize_hnsw_target_recall(target_recall),
        cagra_implementation = normalize_cagra_implementation_arg(
            cagra_implementation
        ),
        cagra_build_algo = normalize_cagra_build_algo_arg(cagra_build_algo),
        task = normalize_knn_task(task),
        k = k,
        n_threads = normalize_nn_threads(n_threads)
    )
}

prepare_knn_model_response <- function(y, x, task) {
    if (NROW(y) != nrow(x)) {
        stop(
            "`Ytrain` must have one value for each row of `Xtrain`.",
            call. = FALSE
        )
    }
    if (anyNA(y)) {
        stop("`Ytrain` must not contain missing values.", call. = FALSE)
    }
    if (identical(task, "auto")) {
        task <- if (is.numeric(y)) "regression" else "classification"
    }
    if (identical(task, "regression")) {
        y <- as.numeric(y)
        if (!all(is.finite(y))) {
            stop(
                "Regression `Ytrain` must contain only finite numeric values.",
                call. = FALSE
            )
        }
        return(list(y = y, levels = NULL, task = task))
    }
    y <- as.factor(y)
    list(y = y, levels = levels(y), task = task)
}

new_knn_model <- function(x, response, args, index) {
    structure(
        list(
            Xtrain = x,
            Ytrain = response$y,
            task = response$task,
            levels = response$levels,
            backend = as.character(args$backend)[1L],
            method = args$method,
            tuning = args$tuning,
            target_recall = args$target_recall,
            cagra_implementation = args$cagra_implementation,
            cagra_build_algo = args$cagra_build_algo,
            metric = args$metric,
            k = args$k,
            n_threads = args$n_threads,
            nn_index = index$index %||% NULL,
            nn_index_backend = index$backend %||% NA_character_,
            nn_index_method = index$method %||% NA_character_,
            nn_index_k = index$k %||% NA_integer_,
            nn_index_target_recall = index$target_recall %||% NA_real_,
            nn_index_params = index$params %||% NULL
        ),
        class = "faissR_knn_model"
    )
}

#' Predict from a faissR kNN model
#'
#' @param object A model returned by \code{\link{knn}()}.
#' @param newdata Numeric or optional `float::fl()`/`float32` query matrix with
#'   observations in rows. Float32 queries are preserved for methods with
#'   direct float32 adapters.
#' @param k Number of neighbours.
#' @param backend Device backend used for this prediction call: `"cpu"` or
#'   `"cuda"`. The fitted model's method and metric are always reused. `NULL`
#'   follows the package backend configuration.
#' @param tuning Tuning policy used for this prediction call. `"auto"` uses the
#'   deterministic default for the resolved method; pilot/cache tuning is
#'   opt-in where implemented. FAISS GPU IVF pilot/cache tuning is
#'   Euclidean-only; non-Euclidean IVF routes use deterministic metric-aware
#'   defaults.
#' @param target_recall Optional speed/recall tier for this prediction call.
#'   `NULL` reuses the fitted model's value; otherwise use `0.9`, `0.95`, or
#'   `0.99`. It affects CUDA auto Flat-vs-IVF selection, CUDA IVF probing, and
#'   HNSW graph-search tiers when prediction needs a new NN search. For
#'   approximate methods this is a selection objective rather than a recall
#'   guarantee.
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
#'   auto-selection metadata from the underlying \code{\link{nn}()} call. The
#'   prediction-time neighbour search is batched: the full `newdata` matrix is
#'   sent to the resolved FAISS/cuVS/native NN route in one call, and metadata
#'   records `batch_query`, `query_n`, and `query_call_count`.
#' @export
predict.faissR_knn_model <- function(
    object,
    newdata,
    k = NULL,
    backend = NULL,
    tuning = c("auto", "cache", "pilot", "fixed", "off", "none"),
    target_recall = NULL,
    cagra_implementation = NULL,
    cagra_build_algo = NULL,
    vote = c("majority", "weighted"),
    type = c("response", "prob"),
    ...
) {
    if (missing(tuning)) {
        tuning <- "auto"
    }
    args <- normalize_knn_prediction_args(
        object,
        k,
        backend,
        tuning,
        target_recall,
        cagra_implementation,
        cagra_build_algo,
        vote,
        type
    )
    query <- validate_knn_model_query(object, newdata)
    neighbours <- find_knn_model_neighbours(object, query, args)
    predict_from_knn_neighbours(object, neighbours, args)
}

normalize_knn_prediction_args <- function(
    object,
    k,
    backend,
    tuning,
    target_recall,
    cagra_implementation,
    cagra_build_algo,
    vote,
    type
) {
    train <- model_Xtrain(object)
    target_recall <- target_recall %||% object$target_recall %||% 0.99
    cagra_implementation <- cagra_implementation %||%
        object$cagra_implementation %||%
        NULL
    cagra_build_algo <- cagra_build_algo %||%
        object$cagra_build_algo %||%
        NULL
    list(
        k = normalize_knn_model_k(k %||% object$k, nrow(train)),
        backend = normalize_public_backend_arg(backend),
        tuning = normalize_nn_tuning(tuning),
        target_recall = normalize_hnsw_target_recall(target_recall),
        cagra_implementation = normalize_cagra_implementation_arg(
            cagra_implementation
        ),
        cagra_build_algo = normalize_cagra_build_algo_arg(cagra_build_algo),
        vote = normalize_knn_vote(vote),
        type = normalize_knn_type(type)
    )
}

find_knn_model_neighbours <- function(object, query, args) {
    neighbours <- knn_predict_with_fitted_nn_index(
        object = object,
        query = query,
        k = args$k,
        backend = args$backend,
        tuning = args$tuning,
        target_recall = args$target_recall
    )
    source <- "fitted_index"
    if (is.null(neighbours)) {
        source <- "nn"
        neighbours <- nn(
            model_Xtrain(object),
            query,
            k = args$k,
            backend = args$backend,
            method = object$method %||% "auto",
            metric = object$metric,
            tuning = args$tuning,
            target_recall = args$target_recall,
            cagra_implementation = args$cagra_implementation,
            cagra_build_algo = args$cagra_build_algo,
            n_threads = object$n_threads
        )
    }
    annotate_knn_prediction_batch(neighbours, query = query, source = source)
}

predict_from_knn_neighbours <- function(object, neighbours, args) {
    response <- model_Ytrain(object)
    if (identical(object$task, "classification")) {
        proba <- class_vote_probabilities(
            labels = response,
            indices = neighbours$indices,
            distances = neighbours$distances,
            levels = object$levels,
            weighted = identical(args$vote, "weighted")
        )
        proba <- attach_knn_model_prediction_metadata(
            proba,
            object,
            neighbours,
            args
        )
        if (identical(args$type, "prob")) {
            return(proba)
        }
        best <- max.col(proba, ties.method = "first")
        pred <- factor(object$levels[best], levels = object$levels)
        return(attach_knn_model_prediction_metadata(
            pred,
            object,
            neighbours,
            args
        ))
    }
    if (identical(args$type, "prob")) {
        stop(
            "`type = \"prob\"` is only available for classification models.",
            call. = FALSE
        )
    }
    pred <- regression_vote(
        response = response,
        indices = neighbours$indices,
        distances = neighbours$distances,
        weighted = identical(args$vote, "weighted")
    )
    attach_knn_model_prediction_metadata(pred, object, neighbours, args)
}

attach_knn_model_prediction_metadata <- function(
    out,
    object,
    neighbours,
    args
) {
    attach_knn_prediction_metadata(
        out,
        neighbours,
        args$k,
        args$backend,
        object$method %||% "auto",
        args$tuning,
        args$target_recall,
        args$cagra_implementation,
        args$cagra_build_algo
    )
}

attach_knn_prediction_metadata <- function(
    out,
    neighbours,
    k,
    backend,
    method,
    tuning,
    target_recall = attr(neighbours, "target_recall") %||% NA_real_,
    cagra_implementation = NULL,
    cagra_build_algo = NULL
) {
    metadata <- c(
        knn_prediction_core_metadata(
            neighbours,
            k,
            backend,
            method,
            tuning,
            target_recall,
            cagra_implementation,
            cagra_build_algo
        ),
        knn_prediction_distance_metadata(neighbours),
        knn_prediction_implementation_metadata(neighbours),
        knn_prediction_input_metadata(neighbours),
        knn_prediction_batch_metadata(neighbours)
    )
    attr(out, "faissR_nn") <- metadata
    out
}

knn_prediction_core_metadata <- function(
    neighbours,
    k,
    backend,
    method,
    tuning,
    target_recall,
    cagra_implementation,
    cagra_build_algo
) {
    list(
        k = as.integer(k),
        requested_backend = attr(neighbours, "requested_backend") %||% backend,
        requested_method = attr(neighbours, "requested_method") %||%
            public_nn_method_label(normalize_nn_method(method)),
        tuning = attr(neighbours, "tuning") %||% tuning,
        target_recall = attr(neighbours, "target_recall") %||% target_recall,
        cagra_implementation = cagra_implementation %||% NA_character_,
        cagra_build_algo = cagra_build_algo %||% NA_character_,
        backend = attr(neighbours, "backend") %||% NA_character_,
        resolved_backend = attr(neighbours, "resolved_backend") %||%
            attr(neighbours, "backend") %||%
            NA_character_,
        metric = attr(neighbours, "metric") %||% NA_character_,
        exact = attr(neighbours, "exact") %||% NA,
        approximation = attr(neighbours, "approximation") %||% NULL,
        faiss = attr(neighbours, "faiss") %||% NULL,
        cuvs = attr(neighbours, "cuvs") %||% NULL,
        spatial_index = attr(neighbours, "spatial_index") %||% NULL,
        auto_selection = attr(neighbours, "auto_selection") %||% NULL,
        metric_transform = attr(neighbours, "metric_transform") %||%
            neighbours$metric_transform %||%
            NULL
    )
}

knn_prediction_distance_metadata <- function(neighbours) {
    distance_type <- neighbours$distance_type %||% NA_character_
    if (is.na(distance_type)) {
        is_float <- inherits(neighbours$distances, "float32") ||
            inherits(neighbours$distances, "fl")
        distance_type <- if (is_float) "float32" else "double"
    }
    list(
        distance_transform = attr(neighbours, "distance_transform") %||% NULL,
        distance_is_metric = attr(neighbours, "distance_is_metric") %||%
            neighbours$distance_is_metric %||%
            NA,
        distance_semantics = attr(neighbours, "distance_semantics") %||%
            neighbours$distance_semantics %||%
            NA_character_,
        distance_comparable_across_queries = attr(
            neighbours,
            "distance_comparable_across_queries"
        ) %||%
            neighbours$distance_comparable_across_queries %||%
            NA,
        distance_order = attr(neighbours, "distance_order") %||%
            neighbours$distance_order %||%
            "smaller_is_better",
        distance_type = distance_type
    )
}

knn_prediction_implementation_metadata <- function(neighbours) {
    list(
        implementation_label = attr(neighbours, "implementation_label") %||%
            neighbours$implementation_label %||%
            NA_character_,
        implementation_scope = attr(neighbours, "implementation_scope") %||%
            neighbours$implementation_scope %||%
            NA_character_,
        preferred_public_method = attr(
            neighbours,
            "preferred_public_method"
        ) %||%
            neighbours$preferred_public_method %||%
            NA_character_,
        canonical_reimplementation = attr(
            neighbours,
            "canonical_reimplementation"
        ) %||%
            neighbours$canonical_reimplementation %||%
            NA
    )
}

knn_prediction_input_metadata <- function(neighbours) {
    list(
        input_type = attr(neighbours, "input_type") %||%
            neighbours$input_type %||%
            NA_character_,
        input_layout = attr(neighbours, "input_layout") %||%
            neighbours$input_layout %||%
            NA_character_,
        input_owns_data = attr(neighbours, "input_owns_data") %||%
            neighbours$input_owns_data %||%
            NA,
        float32_compatibility_conversion = attr(
            neighbours,
            "float32_compatibility_conversion"
        ) %||%
            neighbours$float32_compatibility_conversion %||%
            NA
    )
}

knn_prediction_batch_metadata <- function(neighbours) {
    list(
        batch_query = attr(neighbours, "batch_query") %||% NA,
        query_n = attr(neighbours, "query_n") %||% NA_integer_,
        query_call_count = attr(neighbours, "query_call_count") %||%
            NA_integer_,
        query_source = attr(neighbours, "query_source") %||% NA_character_
    )
}

annotate_knn_prediction_batch <- function(neighbours, query, source) {
    query_dims <- if (is_float32_matrix_input(query)) {
        float32_matrix_dims(query, "newdata")
    } else {
        dim(query)
    }
    query_n <- as.integer(query_dims[[1L]])
    query_call_count <- as.integer(neighbours$query_call_count %||% 1L)
    batch_query <- isTRUE(neighbours$batch_query %||% TRUE)

    neighbours$query_n <- as.integer(neighbours$query_n %||% query_n)
    neighbours$batch_query <- batch_query
    neighbours$query_call_count <- query_call_count
    neighbours$query_source <- source
    attr(neighbours, "query_n") <- as.integer(neighbours$query_n)
    attr(neighbours, "batch_query") <- batch_query
    attr(neighbours, "query_call_count") <- query_call_count
    attr(neighbours, "query_source") <- source

    for (metadata_name in c("approximation", "faiss", "cuvs")) {
        metadata <- attr(neighbours, metadata_name, exact = TRUE)
        if (is.list(metadata)) {
            metadata$batch_query <- batch_query
            metadata$query_n <- as.integer(neighbours$query_n)
            metadata$query_call_count <- query_call_count
            metadata$query_source <- source
            attr(neighbours, metadata_name) <- metadata
        }
    }
    neighbours
}

knn_build_fitted_nn_index <- function(
    x,
    backend,
    method,
    metric,
    tuning,
    target_recall,
    k,
    n_threads
) {
    spec <- knn_fitted_index_spec(x, backend, method, metric, k)
    if (is.null(spec)) {
        return(NULL)
    }
    args <- list(
        x = x,
        dims = spec$dims,
        metric = metric,
        k = k,
        target_recall = target_recall,
        tuning = tuning,
        n_threads = n_threads
    )
    switch(
        spec$backend,
        faiss_flat_l2 = build_knn_flat_index(args),
        faiss_hnsw = build_knn_hnsw_index(args),
        faiss_ivf = build_knn_ivf_index(args),
        faiss_ivfpq = build_knn_ivfpq_index(args),
        NULL
    )
}

knn_fitted_index_spec <- function(x, backend, method, metric, k) {
    if (
        !identical(backend, "cpu") ||
            !identical(metric, "euclidean") ||
            !isTRUE(faiss_available())
    ) {
        return(NULL)
    }
    dims <- if (is_float32_matrix_input(x)) {
        float32_matrix_dims(x, "Xtrain")
    } else {
        dim(x)
    }
    resolved <- tryCatch(
        resolve_public_nn_backend(
            backend,
            method,
            metric,
            n = dims[[1L]],
            p = dims[[2L]],
            k = k,
            self_query = FALSE
        ),
        error = function(e) NA_character_
    )
    supported <- c(
        "faiss_flat_l2",
        "faiss_hnsw",
        "faiss_ivf",
        "faiss_ivfpq"
    )
    if (!resolved %in% supported) {
        return(NULL)
    }
    list(backend = resolved, dims = dims)
}

new_knn_fitted_index <- function(index, backend, method, args, params) {
    list(
        index = index,
        backend = backend,
        method = method,
        k = as.integer(args$k),
        target_recall = as.numeric(args$target_recall),
        params = params,
        tuning = args$tuning
    )
}

build_knn_flat_index <- function(args) {
    index <- nn_faiss_index_build_float32_cpp(
        args$x,
        "flat",
        NA_integer_,
        NA_integer_,
        NA_integer_,
        NA_integer_,
        NA_integer_,
        NA_integer_,
        NA_integer_,
        NA_integer_,
        faiss_metric_search_arg(args$metric),
        faiss_metric_distance_output_arg(args$metric),
        as.integer(args$n_threads)
    )
    new_knn_fitted_index(
        index,
        "faiss_flat_l2",
        "flat",
        args,
        list(tuning_source = "none")
    )
}

build_knn_hnsw_index <- function(args) {
    params <- faiss_hnsw_params(
        args$k,
        n = args$dims[[1L]],
        p = args$dims[[2L]],
        metric = args$metric,
        target_recall = args$target_recall
    )
    index <- nn_faiss_hnsw_index_build_float32_cpp(
        args$x,
        as.integer(params$m),
        as.integer(params$ef_construction),
        as.integer(params$ef_search),
        faiss_metric_search_arg(args$metric),
        faiss_metric_distance_output_arg(args$metric),
        as.integer(args$n_threads)
    )
    new_knn_fitted_index(index, "faiss_hnsw", "hnsw", args, params)
}

build_knn_ivf_index <- function(args) {
    params <- faiss_ivf_params(
        args$dims[[1L]],
        args$k,
        metric = args$metric,
        p = args$dims[[2L]],
        target_recall = args$target_recall
    )
    index <- nn_faiss_index_build_float32_cpp(
        args$x,
        "ivf",
        as.integer(params$nlist),
        as.integer(params$nprobe),
        NA_integer_,
        NA_integer_,
        NA_integer_,
        NA_integer_,
        NA_integer_,
        NA_integer_,
        faiss_metric_search_arg(args$metric),
        faiss_metric_distance_output_arg(args$metric),
        as.integer(args$n_threads)
    )
    new_knn_fitted_index(index, "faiss_ivf", "ivf", args, params)
}

build_knn_ivfpq_index <- function(args) {
    validate_faiss_cpu_ivfpq_training_size(args$dims[[1L]])
    params <- faiss_ivf_params(
        args$dims[[1L]],
        args$k,
        metric = args$metric,
        p = args$dims[[2L]],
        method = "ivfpq",
        target_recall = args$target_recall
    )
    pq <- faiss_pq_params(args$dims[[2L]], n = args$dims[[1L]])
    index <- nn_faiss_index_build_float32_cpp(
        args$x,
        "ivfpq",
        as.integer(params$nlist),
        as.integer(params$nprobe),
        as.integer(pq$m),
        as.integer(pq$nbits),
        NA_integer_,
        NA_integer_,
        NA_integer_,
        NA_integer_,
        faiss_metric_search_arg(args$metric),
        faiss_metric_distance_output_arg(args$metric),
        as.integer(args$n_threads)
    )
    new_knn_fitted_index(
        index,
        "faiss_ivfpq",
        "ivfpq",
        args,
        c(params, pq)
    )
}

knn_predict_with_fitted_nn_index <- function(
    object,
    query,
    k,
    backend,
    tuning,
    target_recall
) {
    stored_backend <- object$nn_index_backend %||% NA_character_
    if (identical(stored_backend, "faiss_hnsw")) {
        return(knn_predict_with_fitted_faiss_hnsw_index(
            object = object,
            query = query,
            k = k,
            backend = backend,
            tuning = tuning,
            target_recall = target_recall
        ))
    }
    if (
        stored_backend %in%
            c(
                "faiss_flat_l2",
                "faiss_flat_ip",
                "faiss_ivf",
                "faiss_ivfpq",
                "faiss_nsg",
                "faiss_nndescent"
            )
    ) {
        return(knn_predict_with_fitted_faiss_index(
            object = object,
            query = query,
            k = k,
            backend = backend,
            tuning = tuning,
            target_recall = target_recall
        ))
    }
    NULL
}

knn_fitted_index_settings_match <- function(
    object,
    k,
    backend,
    tuning,
    target_recall
) {
    if (!identical(backend, "cpu")) {
        return(FALSE)
    }
    if (!identical(tuning, object$tuning %||% "auto")) {
        return(FALSE)
    }
    !is.null(object$nn_index)
}

knn_fitted_index_resolves_to_stored_backend <- function(object, backend, k) {
    train <- model_Xtrain(object)
    resolved_backend <- tryCatch(
        resolve_public_nn_backend(
            backend,
            object$method %||% "auto",
            object$metric %||% "euclidean",
            n = nrow(train),
            p = ncol(train),
            k = k,
            self_query = FALSE
        ),
        error = function(e) NA_character_
    )
    identical(resolved_backend, object$nn_index_backend %||% NA_character_)
}

knn_scalar_int <- function(value, default = NA_integer_) {
    value <- faissr_quiet_warning(as.integer(value))
    if (length(value) != 1L || is.na(value) || !is.finite(value)) {
        return(as.integer(default))
    }
    value
}

knn_fitted_index_attr_int <- function(index, name, default = NA_integer_) {
    knn_scalar_int(attr(index, name, exact = TRUE), default = default)
}

knn_fitted_faiss_search_params <- function(object, stored_backend, params, k) {
    search_width <- switch(
        stored_backend,
        faiss_ivf = params$nprobe %||% NA_integer_,
        faiss_ivfpq = params$nprobe %||% NA_integer_,
        faiss_nsg = params$search_l %||% NA_integer_,
        faiss_nndescent = params$search_l %||% NA_integer_,
        NA_integer_
    )
    search_params <- params
    if (stored_backend %in% c("faiss_ivf", "faiss_ivfpq")) {
        ivf <- knn_fitted_ivf_query_params(object, stored_backend, params, k)
        search_width <- ivf$search_width
        search_params <- ivf$params
    }
    list(search_width = search_width, params = search_params)
}

knn_fitted_ivf_query_params <- function(object, stored_backend, params, k) {
    train <- model_Xtrain(object)
    index_n <- knn_fitted_index_attr_int(object$nn_index, "n", nrow(train))
    index_nlist <- knn_fitted_index_attr_int(
        object$nn_index, "nlist", params$nlist %||% NA_integer_
    )
    query_params <- tryCatch(
        faiss_ivf_params(
            index_n, k, metric = object$metric %||% "euclidean",
            p = ncol(train),
            method = if (stored_backend == "faiss_ivfpq") "ivfpq" else "ivf",
            target_recall = object$target_recall %||% 0.99
        ),
        error = function(e) NULL
    )
    query_nprobe <- knn_scalar_int(
        query_params$nprobe %||% params$nprobe,
        default = params$nprobe %||% NA_integer_
    )
    if (!is.na(index_nlist)) {
        query_nprobe <- max(1L, min(query_nprobe, index_nlist))
    }
    params$build_nprobe <- knn_scalar_int(params$nprobe %||% NA_integer_)
    params$search_nprobe <- params$nprobe <- query_nprobe
    params$nlist <- index_nlist
    params$nprobe_recomputed_for_query <- !identical(
        as.integer(params$build_nprobe), as.integer(query_nprobe)
    )
    params$tuning_query_k <- as.integer(k)
    list(search_width = query_nprobe, params = params)
}

knn_predict_with_fitted_faiss_hnsw_index <- function(
    object,
    query,
    k,
    backend,
    tuning,
    target_recall
) {
    if (
        !knn_fitted_hnsw_settings_match(
            object,
            k,
            backend,
            tuning,
            target_recall
        )
    ) {
        return(NULL)
    }
    params <- object$nn_index_params
    out <- search_fitted_hnsw_index(object, query, k, params$ef_search)
    if (is.null(out)) {
        return(NULL)
    }
    finish_fitted_hnsw_prediction(
        out, params, object, k, backend, tuning, target_recall
    )
}

finish_fitted_hnsw_prediction <- function(
    out,
    params,
    object,
    k,
    backend,
    tuning,
    target_recall
) {
    metric <- object$metric %||% "euclidean"
    result <- finish_nn_result(
        out,
        "faiss_hnsw",
        k,
        self_query = FALSE,
        exact = FALSE,
        metric = metric
    )
    result <- annotate_fitted_search_request(
        result,
        backend,
        "hnsw",
        tuning,
        target_recall
    )
    attr(result, "approximation") <- knn_fitted_hnsw_approximation(
        out,
        params,
        metric,
        target_recall
    )
    attr(result, "faiss") <- list(
        index_type = out$index_type %||% "IndexHNSWFlatExternalPtr",
        backend = "cpu",
        library = "faiss",
        metric = metric,
        index_reused = TRUE
    )
    result <- append_nn_tuning_metadata(result, params)
    finish_float32_direct_result(result, out)
}

knn_fitted_hnsw_settings_match <- function(
    object,
    k,
    backend,
    tuning,
    target_recall
) {
    knn_fitted_index_settings_match(
        object,
        k,
        backend,
        tuning,
        target_recall
    ) &&
        isTRUE(all.equal(
            as.numeric(target_recall),
            as.numeric(object$nn_index_target_recall)
        )) &&
        identical(object$method %||% "auto", "hnsw") &&
        identical(object$metric, "euclidean") &&
        knn_fitted_index_resolves_to_stored_backend(object, backend, k)
}

search_fitted_hnsw_index <- function(object, query, k, ef_search) {
    tryCatch(
        nn_faiss_hnsw_index_search_float32_cpp(
            object$nn_index,
            query,
            as.integer(k),
            FALSE,
            as.integer(ef_search),
            as.integer(object$n_threads),
            "double"
        ),
        error = knn_invalid_fitted_pointer_handler
    )
}

knn_invalid_fitted_pointer_handler <- function(error) {
    message <- conditionMessage(error)
    if (
        grepl("pointer|externalptr|not valid|null", message, ignore.case = TRUE)
    ) {
        return(NULL)
    }
    stop(error)
}

annotate_fitted_search_request <- function(
    result,
    backend,
    method,
    tuning,
    target_recall
) {
    attr(result, "requested_backend") <- backend
    attr(result, "requested_method") <- method
    attr(result, "tuning") <- tuning
    attr(result, "target_recall") <- target_recall
    result
}

knn_fitted_hnsw_approximation <- function(out, params, metric, target_recall) {
    list(
        strategy = "faiss_IndexHNSWFlat",
        backend = "faiss_hnsw",
        library = "faiss",
        metric = metric,
        input_type = "float32",
        fitted_index = TRUE,
        index_reused = TRUE,
        m = as.integer(out$m),
        ef_construction = as.integer(out$ef_construction),
        ef_search = as.integer(out$ef_search),
        requested_m = as.integer(out$requested_m),
        requested_ef_construction = as.integer(out$requested_ef_construction),
        requested_ef_search = as.integer(out$requested_ef_search),
        hnsw_parameters_adjusted = isTRUE(out$hnsw_parameters_adjusted),
        tuning_policy = params$policy,
        tuning_rule = params$rule,
        target_recall = as.numeric(params$target_recall %||% target_recall),
        tuning_low_dim = isTRUE(params$low_dim),
        tuning_high_dim = isTRUE(params$high_dim),
        tuning_large_n = isTRUE(params$large_n),
        tuning_small_k = isTRUE(params$small_k),
        tuning_large_k = isTRUE(params$large_k),
        tuning_non_euclidean = isTRUE(params$non_euclidean),
        tuning_source = params$tuning_source %||% "cpp"
    )
}

knn_predict_with_fitted_faiss_index <- function(
    object,
    query,
    k,
    backend,
    tuning,
    target_recall
) {
    stored <- object$nn_index_backend %||% NA_character_
    if (
        !knn_fitted_faiss_settings_match(
            object,
            stored,
            k,
            backend,
            tuning,
            target_recall
        )
    ) {
        return(NULL)
    }
    params <- object$nn_index_params %||% list()
    search <- knn_fitted_faiss_search_params(object, stored, params, k)
    out <- search_fitted_faiss_index(
        object,
        query,
        k,
        search$search_width
    )
    if (is.null(out)) {
        return(NULL)
    }
    finish_fitted_faiss_prediction(
        out, search$params, object, stored, k,
        backend, tuning, target_recall
    )
}

finish_fitted_faiss_prediction <- function(
    out,
    params,
    object,
    stored,
    k,
    backend,
    tuning,
    target_recall
) {
    metric <- object$metric %||% "euclidean"
    exact <- stored %in% c("faiss_flat_l2", "faiss_flat_ip")
    result <- finish_nn_result(
        out,
        stored,
        k,
        self_query = FALSE,
        exact = exact,
        metric = metric
    )
    result <- annotate_fitted_search_request(
        result,
        backend,
        object$method %||% "auto",
        tuning,
        target_recall
    )
    attr(result, "approximation") <- knn_fitted_faiss_approximation(
        stored,
        out,
        params,
        metric,
        target_recall
    )
    attr(result, "faiss") <- knn_fitted_faiss_metadata(
        out,
        stored,
        metric,
        exact
    )
    result <- append_nn_tuning_metadata(
        result,
        object$nn_index_params %||% list()
    )
    finish_float32_direct_result(result, out)
}

knn_fitted_faiss_settings_match <- function(
    object,
    stored,
    k,
    backend,
    tuning,
    target_recall
) {
    if (
        stored %in%
            c("faiss_ivf", "faiss_ivfpq") &&
            !isTRUE(all.equal(
                as.numeric(target_recall),
                as.numeric(object$nn_index_target_recall)
            ))
    ) {
        return(FALSE)
    }
    supported <- c(
        "faiss_flat_l2",
        "faiss_flat_ip",
        "faiss_ivf",
        "faiss_ivfpq",
        "faiss_nsg",
        "faiss_nndescent"
    )
    stored %in%
        supported &&
        knn_fitted_index_settings_match(
            object,
            k,
            backend,
            tuning,
            target_recall
        ) &&
        knn_fitted_index_resolves_to_stored_backend(object, backend, k)
}

search_fitted_faiss_index <- function(object, query, k, search_width) {
    tryCatch(
        nn_faiss_index_search_float32_cpp(
            object$nn_index,
            query,
            as.integer(k),
            FALSE,
            faissr_quiet_warning(as.integer(search_width)),
            as.integer(object$n_threads),
            "double"
        ),
        error = knn_invalid_fitted_pointer_handler
    )
}

knn_fitted_faiss_metadata <- function(out, stored, metric, exact) {
    metadata <- list(
        index_type = out$index_type %||% paste0(stored, "_externalptr"),
        backend = "cpu",
        library = "faiss",
        metric = metric,
        exact = exact,
        index_reused = TRUE
    )
    fields <- c(
        "index_trained",
        "index_training_reused",
        "build_train_call_count",
        "search_train_call_count",
        "centroids_reused",
        "inverted_lists_reused",
        "vectors_reused",
        "build_nprobe",
        "search_nprobe",
        "pq_codebooks_reused",
        "pq_codes_reused",
        "pq_training_reused",
        "build_pq_train_call_count",
        "search_pq_train_call_count"
    )
    for (field in fields) {
        if (!is.null(out[[field]])) metadata[[field]] <- out[[field]]
    }
    metadata
}

knn_fitted_faiss_approximation <- function(
    backend,
    out,
    params,
    metric,
    target_recall
) {
    if (backend %in% c("faiss_flat_l2", "faiss_flat_ip")) {
        return(knn_fitted_flat_approximation(backend, out, params, metric))
    }
    if (backend %in% c("faiss_ivf", "faiss_ivfpq")) {
        metadata <- knn_fitted_ivf_approximation(
            backend,
            out,
            params,
            metric
        )
        if (identical(backend, "faiss_ivfpq")) {
            metadata <- c(
                metadata,
                knn_fitted_pq_approximation(out, params)
            )
        }
        return(metadata)
    }
    list(
        strategy = out$index_type %||% backend,
        backend = backend,
        library = "faiss",
        metric = metric,
        input_type = "float32",
        fitted_index = TRUE,
        index_reused = TRUE,
        target_recall = as.numeric(target_recall)
    )
}

knn_fitted_flat_approximation <- function(backend, out, params, metric) {
    is_ip <- identical(backend, "faiss_flat_ip")
    list(
        strategy = if (is_ip) "faiss_IndexFlatIP" else "faiss_IndexFlatL2",
        backend = backend,
        library = "faiss",
        metric = metric,
        input_type = "float32",
        fitted_index = TRUE,
        index_reused = TRUE,
        exact = TRUE,
        index_type = out$index_type %||%
            if (is_ip) "IndexFlatIPExternalPtr" else "IndexFlatL2ExternalPtr",
        tuning_source = params$tuning_source %||% "none"
    )
}

knn_fitted_ivf_approximation <- function(backend, out, params, metric) {
    list(
        strategy = if (identical(backend, "faiss_ivfpq")) {
            "faiss_IndexIVFPQ"
        } else {
            "faiss_IndexIVFFlat"
        },
        backend = backend,
        library = "faiss",
        metric = metric,
        input_type = "float32",
        fitted_index = TRUE,
        index_reused = TRUE,
        nlist = as.integer(out$nlist),
        nprobe = as.integer(out$nprobe),
        build_nprobe = as.integer(
            params$build_nprobe %||% out$build_nprobe %||% NA_integer_
        ),
        search_nprobe = as.integer(
            params$search_nprobe %||% out$search_nprobe %||% out$nprobe
        ),
        nprobe_recomputed_for_query = isTRUE(
            params$nprobe_recomputed_for_query
        ),
        tuning_query_k = as.integer(params$tuning_query_k %||% NA_integer_),
        requested_nlist = as.integer(
            params$requested_nlist %||% out$requested_nlist
        ),
        requested_nprobe = as.integer(
            params$requested_nprobe %||% out$requested_nprobe
        ),
        index_trained = isTRUE(out$index_trained),
        index_training_reused = isTRUE(out$index_training_reused),
        centroids_reused = isTRUE(out$centroids_reused),
        inverted_lists_reused = isTRUE(out$inverted_lists_reused),
        vectors_reused = isTRUE(out$vectors_reused),
        build_train_call_count = as.integer(
            out$build_train_call_count %||% NA_integer_
        ),
        search_train_call_count = as.integer(
            out$search_train_call_count %||% NA_integer_
        ),
        ivf_parameters_adjusted = isTRUE(out$ivf_parameters_adjusted)
    )
}

knn_fitted_pq_approximation <- function(out, params) {
    list(
        pq_m = as.integer(out$pq_m),
        pq_nbits = as.integer(out$pq_nbits),
        requested_pq_m = as.integer(
            params$requested_m %||%
                params$requested_pq_m %||%
                out$requested_pq_m
        ),
        requested_pq_nbits = as.integer(
            params$requested_nbits %||%
                params$requested_pq_nbits %||%
                out$requested_pq_nbits
        ),
        pq_codebooks_reused = isTRUE(out$pq_codebooks_reused),
        pq_codes_reused = isTRUE(out$pq_codes_reused),
        pq_training_reused = isTRUE(out$pq_training_reused),
        build_pq_train_call_count = as.integer(
            out$build_pq_train_call_count %||% NA_integer_
        ),
        search_pq_train_call_count = as.integer(
            out$search_pq_train_call_count %||% NA_integer_
        ),
        pq_parameters_adjusted = isTRUE(out$pq_parameters_adjusted)
    )
}

validate_knn_model_query <- function(object, newdata) {
    if (!inherits(object, "faissR_knn_model")) {
        stop("`object` must be a faissR kNN model.", call. = FALSE)
    }
    train <- model_Xtrain(object)
    prepare_knn_model_matrix(newdata, "newdata", expected_ncol = ncol(train))
}

prepare_knn_model_matrix <- function(
    x,
    arg_name = "Xtrain",
    expected_ncol = NULL
) {
    prepared <- coerce_knn_model_matrix(x, arg_name)
    out <- prepared$data
    dims <- prepared$dims
    if (anyNA(dims) || dims[1L] < 1L || dims[2L] < 1L) {
        stop(
            "`",
            arg_name,
            "` must have at least one row and one column.",
            call. = FALSE
        )
    }
    if (!is.null(expected_ncol) && dims[2L] != expected_ncol) {
        stop(
            "`",
            arg_name,
            "` must have at least one row and the same ",
            "columns as training data.",
            call. = FALSE
        )
    }
    if (!all(is.finite(out))) {
        stop("`", arg_name, "` must contain only finite values.", call. = FALSE)
    }
    out
}

coerce_knn_model_matrix <- function(x, arg_name) {
    validate_dense_matrix_input(x, arg_name)
    if (is_float32_matrix_input(x)) {
        if (!requireNamespace("float", quietly = TRUE)) {
            stop(
                "`", arg_name, "` is a float32 object but the optional ",
                "float package is not installed.", call. = FALSE
            )
        }
        out <- as.matrix(x)
        return(list(data = out, dims = float32_matrix_dims(out, arg_name)))
    }
    out <- as.matrix(x)
    storage.mode(out) <- "double"
    dims <- dim(out)
    if (is.null(dims) || length(dims) != 2L) {
        stop(
            "`", arg_name, "` must be a two-dimensional matrix-like object.",
            call. = FALSE
        )
    }
    list(data = out, dims = as.integer(dims))
}

model_Xtrain <- function(object) {
    x <- object$Xtrain
    if (is.null(x)) {
        x <- object$X_train
    }
    if (is.null(x)) {
        stop("`object` does not contain training data.", call. = FALSE)
    }
    x
}

model_Ytrain <- function(object) {
    y <- object$Ytrain
    if (is.null(y)) {
        y <- object$y_train
    }
    if (is.null(y)) {
        stop("`object` does not contain training responses.", call. = FALSE)
    }
    y
}

normalize_knn_model_k <- function(k, n_train) {
    k <- faissr_quiet_warning(as.numeric(k))
    if (
        length(k) != 1L ||
            is.na(k) ||
            !is.finite(k) ||
            k < 1L ||
            abs(k - round(k)) > sqrt(.Machine$double.eps)
    ) {
        stop("`k` must be a positive integer.", call. = FALSE)
    }
    k <- as.integer(round(k))
    if (k > n_train) {
        stop(
            "`k` cannot be larger than the number of training rows.",
            call. = FALSE
        )
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
    if (is.na(task) || !nzchar(task)) {
        task <- "auto"
    }
    task <- trimws(task)
    if (!task %in% c("auto", "classification", "regression")) {
        stop(
            "`task` must be one of \"auto\", ",
            "\"classification\", or \"regression\".",
            call. = FALSE
        )
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
    if (is.na(vote) || !nzchar(vote)) {
        vote <- "majority"
    }
    vote <- trimws(vote)
    if (!vote %in% c("majority", "weighted")) {
        stop(
            "`vote` must be one of \"majority\" or \"weighted\".",
            call. = FALSE
        )
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
    if (is.na(type) || !nzchar(type)) {
        type <- "response"
    }
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

class_vote_probabilities <- function(
    labels,
    indices,
    distances,
    levels,
    weighted
) {
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
