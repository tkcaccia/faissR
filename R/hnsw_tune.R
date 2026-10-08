#' Tune a reusable CPU HNSW index on local data
#'
#' `hnsw_tune()` is an opt-in, bounded tuning workflow for CPU FAISS HNSW.
#' It evaluates exact neighbours on disjoint tuning and holdout query samples
#' from a representative reference subset. The existing shape-based defaults
#' used by [nn()] are not changed.
#'
#' @param data Dense numeric reference matrix, numeric data frame, or
#'   `float::fl()` matrix. Sparse and delayed objects are rejected.
#' @param k Number of neighbours used to evaluate candidate settings.
#' @param target_recall Requested recall tier: `0.9`, `0.95`, or `0.99`.
#' @param metric Distance metric. The initial data-aware workflow supports
#'   Euclidean distance only.
#' @param reference_size Maximum number of rows in the representative tuning
#'   reference.
#' @param tune_queries,holdout_queries Numbers of disjoint query rows used for
#'   candidate selection and confirmation.
#' @param seed Integer seed used for row selection, candidate order, and the
#'   query bootstrap. The caller's random-number state is restored.
#' @param build_grid Optional data frame with integer columns `m` and
#'   `ef_construction`. `NULL` creates a bounded grid around the current
#'   shape-based setting.
#' @param ef_search Optional positive integer search-effort values. `NULL`
#'   creates a bounded grid around the current shape-based setting.
#' @param safety_margin Added to `target_recall` for tuning-set eligibility.
#'   The resulting threshold is capped at one.
#' @param bootstrap_repetitions Number of deterministic query-bootstrap
#'   resamples used for the empirical 95 percent lower bound.
#' @param timing_repetitions Number of same-process query timing repetitions.
#'   The median build and query times select among recall-qualified candidates.
#' @param workload_query_rows Expected query rows in each deployed query batch.
#'   `NULL` uses `nrow(data)`, representing a full self-search workload.
#' @param expected_query_batches Expected number of query batches served by one
#'   fitted index. Index construction is counted once and query time this many
#'   times in the projected workload objective.
#' @param n_threads Number of CPU threads.
#' @param max_builds Maximum permitted rows in `build_grid`.
#' @param time_limit Maximum elapsed seconds for the pilot sweep. A partial
#'   sweep is returned but cannot produce a deployable selection.
#' @param memory_budget_bytes Optional estimated byte budget for each pilot
#'   graph and the final graph. `Inf` disables this estimate-based filter.
#' @param build_index Logical; build an owned full-data index only after the
#'   selected setting also reaches the requested holdout lower bound.
#'
#' @return A `faissR_hnsw_tuning` object containing candidate diagnostics,
#'   the selected settings, sampling and provider provenance, and, when
#'   confirmed and requested, an owned persistent HNSW index. Use `predict()`
#'   for repeated external-query searches. Candidate and prediction identifiers
#'   are one-based.
#' @examples
#' if (faiss_available() && requireNamespace("float", quietly = TRUE)) {
#'     set.seed(1)
#'     x <- matrix(rnorm(600), nrow = 100)
#'     fit <- hnsw_tune(
#'         x,
#'         k = 5,
#'         target_recall = 0.9,
#'         reference_size = 100,
#'         tune_queries = 20,
#'         holdout_queries = 20,
#'         build_grid = data.frame(m = 8L, ef_construction = 40L),
#'         ef_search = 20L,
#'         bootstrap_repetitions = 25L,
#'         timing_repetitions = 1L
#'     )
#'     fit
#' }
#' @export
hnsw_tune <- function(
    data,
    k,
    target_recall = 0.99,
    metric = "euclidean",
    reference_size = 20000L,
    tune_queries = 512L,
    holdout_queries = 512L,
    seed = 1L,
    build_grid = NULL,
    ef_search = NULL,
    safety_margin = 0.002,
    bootstrap_repetitions = 500L,
    timing_repetitions = 3L,
    workload_query_rows = NULL,
    expected_query_batches = 1L,
    n_threads = NULL,
    max_builds = 8L,
    time_limit = 300,
    memory_budget_bytes = Inf,
    build_index = TRUE
) {
    call <- match.call()
    request <- hnsw_tune_request(
        data, k, target_recall, metric, reference_size, tune_queries,
        holdout_queries, seed, build_grid, ef_search, safety_margin,
        bootstrap_repetitions, timing_repetitions, workload_query_rows,
        expected_query_batches, n_threads, max_builds, time_limit,
        memory_budget_bytes, build_index
    )
    result <- hnsw_tune_execute(request)
    result$call <- call
    result
}

hnsw_tune_request <- function(
    data, k, target_recall, metric, reference_size, tune_queries,
    holdout_queries, seed, build_grid, ef_search, safety_margin,
    bootstrap_repetitions, timing_repetitions, workload_query_rows,
    expected_query_batches, n_threads, max_builds, time_limit,
    memory_budget_bytes, build_index
) {
    hnsw_tune_requirements(data, metric)
    dims <- if (is_float32_matrix_input(data)) {
        float32_matrix_dims(data, "data")
    } else dim(data)
    scalars <- hnsw_tune_scalars(
        dims, k, reference_size, tune_queries, holdout_queries, seed,
        safety_margin, bootstrap_repetitions, timing_repetitions,
        workload_query_rows, expected_query_batches, n_threads, max_builds,
        time_limit, memory_budget_bytes, build_index
    )
    baseline <- faiss_hnsw_params(
        scalars$k, dims[[1L]], dims[[2L]], "euclidean", target_recall
    )
    grid <- hnsw_tune_grid(build_grid, ef_search, baseline, scalars)
    c(list(
        data = data,
        dims = dims,
        target_recall = normalize_hnsw_target_recall(target_recall),
        baseline = baseline,
        grid = grid
    ), scalars)
}

hnsw_tune_requirements <- function(data, metric) {
    validate_dense_matrix_input(data, "data")
    metric <- normalize_nn_metric(metric)
    if (!identical(metric, "euclidean")) {
        stop(
            "`hnsw_tune()` currently supports `metric = \"euclidean\"` only.",
            call. = FALSE
        )
    }
    if (!isTRUE(faiss_available())) {
        stop("`hnsw_tune()` requires a functional CPU FAISS build.",
            call. = FALSE)
    }
    if (!requireNamespace("float", quietly = TRUE)) {
        stop("`hnsw_tune()` requires the optional `float` package.",
            call. = FALSE)
    }
    invisible(TRUE)
}

hnsw_tune_scalars <- function(
    dims, k, reference_size, tune_queries, holdout_queries, seed,
    safety_margin, bootstrap_repetitions, timing_repetitions,
    workload_query_rows, expected_query_batches, n_threads, max_builds,
    time_limit, memory_budget_bytes, build_index
) {
    k <- normalize_nn_positive_integer(k, "k", "`k` must be positive.")
    values <- list(
        reference_size = reference_size,
        tune_queries = tune_queries,
        holdout_queries = holdout_queries,
        bootstrap_repetitions = bootstrap_repetitions,
        timing_repetitions = timing_repetitions,
        workload_query_rows = if (is.null(workload_query_rows)) {
            dims[[1L]]
        } else workload_query_rows,
        expected_query_batches = expected_query_batches,
        max_builds = max_builds
    )
    values <- lapply(names(values), function(name) {
        normalize_nn_positive_integer(
            values[[name]], name, paste0("`", name, "` must be positive.")
        )
    })
    names(values) <- c(
        "reference_size", "tune_queries", "holdout_queries",
        "bootstrap_repetitions", "timing_repetitions",
        "workload_query_rows", "expected_query_batches", "max_builds"
    )
    hnsw_tune_validate_sizes(dims, k, values)
    c(list(
        k = k,
        seed = hnsw_tune_seed(seed),
        safety_margin = hnsw_tune_probability(safety_margin, "safety_margin"),
        n_threads = normalize_nn_threads(n_threads),
        time_limit = hnsw_tune_limit(time_limit, "time_limit"),
        memory_budget_bytes = hnsw_tune_limit(
            memory_budget_bytes, "memory_budget_bytes", allow_infinite = TRUE
        ),
        build_index = isTRUE(build_index)
    ), values)
}

hnsw_tune_validate_sizes <- function(dims, k, values) {
    reference_size <- min(as.integer(dims[[1L]]), values$reference_size)
    required <- values$tune_queries + values$holdout_queries
    if (k >= reference_size) {
        stop("`k` must be smaller than the tuning reference size.",
            call. = FALSE)
    }
    if (required > reference_size) {
        stop(
            "`tune_queries + holdout_queries` must not exceed the tuning ",
            "reference size.", call. = FALSE
        )
    }
    invisible(TRUE)
}

hnsw_tune_seed <- function(seed) {
    seed <- faissr_quiet_warning(as.integer(seed))
    if (length(seed) != 1L || is.na(seed)) {
        stop("`seed` must be one finite integer.", call. = FALSE)
    }
    seed
}

hnsw_tune_probability <- function(value, name) {
    value <- faissr_quiet_warning(as.numeric(value))
    if (length(value) != 1L || is.na(value) || !is.finite(value) ||
        value < 0 || value >= 1) {
        stop("`", name, "` must be in [0, 1).", call. = FALSE)
    }
    value
}

hnsw_tune_limit <- function(value, name, allow_infinite = FALSE) {
    value <- faissr_quiet_warning(as.numeric(value))
    valid_infinite <- isTRUE(allow_infinite) && identical(value, Inf)
    if (length(value) != 1L || is.na(value) ||
        (!is.finite(value) && !valid_infinite) || value <= 0) {
        stop("`", name, "` must be a positive number.", call. = FALSE)
    }
    value
}

hnsw_tune_grid <- function(build_grid, ef_search, baseline, scalars) {
    if (is.null(build_grid)) {
        build_grid <- hnsw_tune_default_build_grid(baseline)
    }
    required <- c("m", "ef_construction")
    if (!is.data.frame(build_grid) || !all(required %in% names(build_grid))) {
        stop("`build_grid` must contain `m` and `ef_construction` columns.",
            call. = FALSE)
    }
    build_grid <- unique(data.frame(
        m = hnsw_tune_positive_values(build_grid$m, "build_grid$m"),
        ef_construction = hnsw_tune_positive_values(
            build_grid$ef_construction, "build_grid$ef_construction"
        )
    ))
    if (nrow(build_grid) > scalars$max_builds) {
        stop("`build_grid` exceeds `max_builds`.", call. = FALSE)
    }
    if (is.null(ef_search)) ef_search <- hnsw_tune_default_search(baseline)
    search <- hnsw_tune_positive_vector(ef_search, "ef_search")
    list(build = build_grid, search = sort(unique(pmax(scalars$k, search))))
}

hnsw_tune_positive_vector <- function(value, name) {
    sort(unique(hnsw_tune_positive_values(value, name)))
}

hnsw_tune_positive_values <- function(value, name) {
    value <- faissr_quiet_warning(as.integer(value))
    if (!length(value) || anyNA(value) || any(value < 1L)) {
        stop("`", name, "` must contain positive integers.", call. = FALSE)
    }
    value
}

hnsw_tune_default_build_grid <- function(baseline) {
    m <- as.integer(baseline$m)
    efc <- as.integer(baseline$ef_construction)
    unique(data.frame(
        m = pmax(4L, c(m %/% 2L, m, m + m %/% 2L)),
        ef_construction = pmax(4L, c(efc %/% 2L, efc, efc * 2L))
    ))
}

hnsw_tune_default_search <- function(baseline) {
    value <- as.integer(baseline$ef_search)
    sort(unique(pmax(1L, c(
        value %/% 2L, value, value * 2L, value * 4L, value * 8L
    ))))
}

hnsw_tune_guidance <- function(status) {
    switch(status,
        no_confirmed_candidate = paste0(
            "No candidate completed the bounded sweep and met the tuning ",
            "criterion; inspect diagnostics and, if appropriate, supply a ",
            "broader `ef_search` or `build_grid`."
        ),
        holdout_not_confirmed = paste0(
            "The tuning candidate did not meet the confirmation criterion; ",
            "do not deploy it as target-attaining."
        ),
        confirmed_final_memory_budget_exceeded = paste0(
            "The confirmed setting exceeded the estimated full-index memory ",
            "budget; no full index was built."
        ),
        NULL
    )
}

hnsw_tune_execute <- function(request) {
    started <- proc.time()[[3L]]
    sample <- hnsw_tune_sample(request)
    sweep <- hnsw_tune_sweep(request, sample, started)
    selected <- hnsw_tune_select(request, sweep)
    tuning_seconds <- proc.time()[[3L]] - started
    fit <- hnsw_tune_final_index(request, selected)
    workflow_seconds <- proc.time()[[3L]] - started
    structure(list(
        status = fit$status,
        guidance = hnsw_tune_guidance(fit$status),
        selected = selected,
        diagnostics = sweep$diagnostics,
        index = fit$index,
        final_build_seconds = fit$seconds,
        tuning_exact_seconds = sample$exact_seconds,
        tuning_build_seconds = hnsw_tune_build_seconds(sweep$diagnostics),
        tuning_search_seconds = hnsw_tune_search_seconds(sweep$diagnostics),
        tuning_total_seconds = tuning_seconds,
        workflow_total_seconds = workflow_seconds,
        sampling = sample$metadata,
        budget = hnsw_tune_budget_metadata(request, sweep),
        provenance = hnsw_tune_provenance(request),
        workload = hnsw_tune_workload_metadata(request, sample),
        k = request$k,
        target_recall = request$target_recall,
        metric = "euclidean",
        n_threads = request$n_threads
    ), class = "faissR_hnsw_tuning")
}

hnsw_tune_sample <- function(request) {
    rows <- with_rng_seed(request$seed, sample.int(
        request$dims[[1L]], min(request$dims[[1L]], request$reference_size),
        replace = FALSE
    ))
    reference <- hnsw_tune_as_float(hnsw_tune_rows(request$data, rows))
    query_count <- request$tune_queries + request$holdout_queries
    query_rows <- with_rng_seed(request$seed + 1L, sample.int(
        length(rows), query_count, replace = FALSE
    ))
    query <- reference[query_rows, , drop = FALSE]
    started <- proc.time()[[3L]]
    truth <- hnsw_tune_exact(reference, query, query_rows, request)
    exact_seconds <- proc.time()[[3L]] - started
    scope <- if (length(rows) == request$dims[[1L]]) {
        "full_reference"
    } else "seeded_random_subset"
    tune_index <- seq_len(request$tune_queries)
    holdout_index <- request$tune_queries + seq_len(request$holdout_queries)
    list(
        reference = reference,
        query = query,
        query_rows = query_rows,
        truth = truth,
        exact_seconds = exact_seconds,
        split = rep(c("tune", "holdout"), c(
            request$tune_queries, request$holdout_queries
        )),
        metadata = list(
            reference_rows = rows,
            query_reference_rows = query_rows,
            query_data_rows = rows[query_rows],
            tune_query_data_rows = rows[query_rows[tune_index]],
            holdout_query_data_rows = rows[query_rows[holdout_index]],
            tune_queries = request$tune_queries,
            holdout_queries = request$holdout_queries,
            seed = request$seed,
            reference_size = length(rows),
            reference_sampling = scope,
            prefix_reference = FALSE
        )
    )
}

hnsw_tune_rows <- function(data, rows) {
    if (is.data.frame(data)) data <- as.matrix(data)
    data[rows, , drop = FALSE]
}

hnsw_tune_as_float <- function(data) {
    if (is_float32_matrix_input(data)) return(data)
    data <- as.matrix(data)
    storage.mode(data) <- "double"
    float::fl(data)
}

hnsw_tune_exact <- function(reference, query, query_rows, request) {
    out <- nn_faiss_flat_float32_cpp(
        reference, query, request$k + 1L, FALSE,
        request$n_threads, "euclidean", "double"
    )
    hnsw_tune_remove_self(out, query_rows, request$k)
}

hnsw_tune_remove_self <- function(result, self_rows, k) {
    ids <- matrix(NA_integer_, nrow(result$indices), k)
    distances <- matrix(NA_real_, nrow(result$indices), k)
    for (i in seq_len(nrow(result$indices))) {
        keep <- which(result$indices[i, ] != self_rows[[i]])
        if (length(keep) < k) stop("Self-neighbour removal failed.",
            call. = FALSE)
        keep <- keep[seq_len(k)]
        ids[i, ] <- result$indices[i, keep]
        distances[i, ] <- result$distances[i, keep]
    }
    list(indices = ids, distances = distances)
}

hnsw_tune_sweep <- function(request, sample, started) {
    diagnostics <- list()
    completed <- TRUE
    order <- with_rng_seed(request$seed + 2L, sample.int(
        nrow(request$grid$build)
    ))
    for (row in order) {
        if (proc.time()[[3L]] - started > request$time_limit) {
            completed <- FALSE
            break
        }
        evaluated <- hnsw_tune_build_case(request, sample, row)
        diagnostics <- c(diagnostics, evaluated)
    }
    list(
        diagnostics = hnsw_tune_bind(diagnostics),
        complete = completed,
        elapsed_seconds = proc.time()[[3L]] - started
    )
}

hnsw_tune_build_case <- function(request, sample, row) {
    setting <- request$grid$build[row, , drop = FALSE]
    estimate <- hnsw_tune_index_bytes(
        nrow(sample$reference), request$dims[[2L]], setting$m
    )
    if (estimate > request$memory_budget_bytes) {
        return(list(hnsw_tune_skipped_case(setting, estimate)))
    }
    built <- hnsw_tune_repeated_build(request, sample, setting)
    index <- built$index
    on.exit(rm(index), add = TRUE)
    order <- with_rng_seed(request$seed + 100L + row, sample(
        request$grid$search,
        length(request$grid$search),
        replace = FALSE
    ))
    lapply(order, function(ef) hnsw_tune_search_case(
        request, sample, index, setting, ef, built$times, estimate
    ))
}

hnsw_tune_repeated_build <- function(request, sample, setting) {
    times <- numeric(request$timing_repetitions)
    index <- NULL
    for (i in seq_along(times)) {
        if (!is.null(index)) {
            rm(index)
            gc(FALSE)
        }
        started <- proc.time()[[3L]]
        index <- nn_faiss_hnsw_index_build_float32_cpp(
            sample$reference, setting$m, setting$ef_construction, 1L,
            "euclidean", "euclidean", request$n_threads
        )
        times[[i]] <- proc.time()[[3L]] - started
    }
    list(index = index, times = times)
}

hnsw_tune_search_case <- function(
    request, sample, index, setting, ef, build_times, estimate
) {
    invisible(nn_faiss_hnsw_index_search_float32_cpp(
        index, sample$query[1:2, , drop = FALSE], request$k + 1L,
        FALSE, ef, request$n_threads, "double"
    ))
    times <- numeric(request$timing_repetitions)
    found <- NULL
    for (i in seq_along(times)) {
        started <- proc.time()[[3L]]
        current <- nn_faiss_hnsw_index_search_float32_cpp(
            index, sample$query, request$k + 1L, FALSE, ef,
            request$n_threads, "double"
        )
        times[[i]] <- proc.time()[[3L]] - started
        if (is.null(found)) found <- current
    }
    observed <- hnsw_tune_remove_self(found, sample$query_rows, request$k)
    recall <- hnsw_tune_recall(observed, sample$truth, request$k)
    hnsw_tune_case_rows(
        request, sample, setting, found, recall, times, build_times, estimate
    )
}

hnsw_tune_recall <- function(observed, truth, k) {
    vapply(seq_len(nrow(truth$indices)), function(i) {
        expected <- truth$indices[i, ]
        found <- observed$indices[i, ]
        overlap <- found %in% expected
        tolerance <- 1e-5 + 1e-4 * abs(truth$distances[i, k])
        tied <- !overlap & observed$distances[i, ] <=
            truth$distances[i, k] + tolerance
        min(k, sum(overlap) + sum(tied)) / k
    }, numeric(1L))
}

hnsw_tune_case_rows <- function(
    request, sample, setting, found, recall, times, build_times, estimate
) {
    timing <- hnsw_tune_projected_timing(
        request, sample, build_times, times
    )
    do.call(rbind, lapply(c("tune", "holdout"), function(split) {
        values <- recall[sample$split == split]
        summary <- hnsw_tune_recall_summary(
            values,
            request$bootstrap_repetitions,
            request$seed + if (identical(split, "tune")) 200L else 300L
        )
        data.frame(
            split = split,
            m = as.integer(found$m),
            ef_construction = as.integer(found$ef_construction),
            ef_search = as.integer(found$ef_search),
            mean_recall = summary$mean,
            lower_95_recall = summary$lower,
            p05_recall = summary$p05,
            p10_recall = summary$p10,
            min_recall = summary$min,
            build_seconds = timing$build_median_seconds,
            build_iqr_seconds = timing$build_iqr_seconds,
            build_total_seconds = timing$build_total_seconds,
            query_median_seconds = timing$query_median_seconds,
            query_iqr_seconds = timing$query_iqr_seconds,
            query_total_seconds = timing$query_total_seconds,
            projected_build_seconds = timing$projected_build_seconds,
            projected_query_batch_seconds =
                timing$projected_query_batch_seconds,
            projected_workload_seconds = timing$projected_workload_seconds,
            estimated_index_bytes = estimate,
            n_queries = length(values),
            status = "measured"
        )
    }))
}

hnsw_tune_projected_timing <- function(
    request, sample, build_times, query_times
) {
    reference_scale <- request$dims[[1L]] / nrow(sample$reference)
    query_scale <- request$workload_query_rows / nrow(sample$query)
    build_median <- stats::median(build_times)
    query_median <- stats::median(query_times)
    projected_build <- build_median * reference_scale
    projected_query <- query_median * query_scale
    list(
        build_median_seconds = build_median,
        build_iqr_seconds = stats::IQR(build_times),
        build_total_seconds = sum(build_times),
        query_median_seconds = query_median,
        query_iqr_seconds = stats::IQR(query_times),
        query_total_seconds = sum(query_times),
        projected_build_seconds = projected_build,
        projected_query_batch_seconds = projected_query,
        projected_workload_seconds = projected_build +
            request$expected_query_batches * projected_query
    )
}

hnsw_tune_recall_summary <- function(values, repetitions, seed) {
    means <- with_rng_seed(seed, replicate(
        repetitions, mean(sample(values, length(values), replace = TRUE))
    ))
    list(
        mean = mean(values),
        lower = unname(stats::quantile(means, 0.025, type = 8)),
        p05 = unname(stats::quantile(values, 0.05, type = 8)),
        p10 = unname(stats::quantile(values, 0.10, type = 8)),
        min = min(values)
    )
}

hnsw_tune_skipped_case <- function(setting, estimate) {
    data.frame(
        split = c("tune", "holdout"),
        m = setting$m,
        ef_construction = setting$ef_construction,
        ef_search = NA_integer_,
        mean_recall = NA_real_,
        lower_95_recall = NA_real_,
        p05_recall = NA_real_,
        p10_recall = NA_real_,
        min_recall = NA_real_,
        build_seconds = NA_real_,
        build_iqr_seconds = NA_real_,
        build_total_seconds = NA_real_,
        query_median_seconds = NA_real_,
        query_iqr_seconds = NA_real_,
        query_total_seconds = NA_real_,
        projected_build_seconds = NA_real_,
        projected_query_batch_seconds = NA_real_,
        projected_workload_seconds = NA_real_,
        estimated_index_bytes = estimate,
        n_queries = NA_integer_,
        status = "memory_budget_exceeded"
    )
}

hnsw_tune_bind <- function(rows) {
    if (!length(rows)) return(data.frame())
    do.call(rbind, rows)
}

hnsw_tune_build_seconds <- function(diagnostics) {
    measured <- diagnostics[diagnostics$status == "measured", , drop = FALSE]
    if (!nrow(measured)) return(0)
    key <- paste(measured$m, measured$ef_construction, sep = ":")
    sum(measured$build_total_seconds[!duplicated(key)])
}

hnsw_tune_search_seconds <- function(diagnostics) {
    measured <- diagnostics[
        diagnostics$status == "measured" & diagnostics$split == "tune",
        , drop = FALSE
    ]
    if (!nrow(measured)) return(0)
    sum(measured$query_total_seconds)
}

hnsw_tune_select <- function(request, sweep) {
    if (!isTRUE(sweep$complete) || !nrow(sweep$diagnostics)) return(NULL)
    tune <- sweep$diagnostics[
        sweep$diagnostics$split == "tune" &
            sweep$diagnostics$status == "measured",
        , drop = FALSE
    ]
    threshold <- min(1, request$target_recall + request$safety_margin)
    eligible <- tune[!is.na(tune$lower_95_recall) &
        tune$lower_95_recall >= threshold, , drop = FALSE]
    if (!nrow(eligible)) return(NULL)
    eligible <- eligible[order(
        eligible$projected_workload_seconds,
        eligible$projected_query_batch_seconds,
        eligible$projected_build_seconds,
        eligible$m, eligible$ef_construction, eligible$ef_search
    ), , drop = FALSE]
    selected <- eligible[1L, , drop = FALSE]
    holdout <- hnsw_tune_matching_holdout(sweep$diagnostics, selected)
    list(
        m = selected$m[[1L]],
        ef_construction = selected$ef_construction[[1L]],
        ef_search = selected$ef_search[[1L]],
        selection_rule = paste0(
            "minimum_median_projected_build_plus_",
            "expected_batches_times_query_meeting_tune_lcb"
        ),
        projected_workload_seconds =
            selected$projected_workload_seconds[[1L]],
        tuning_threshold = threshold,
        tune = selected,
        holdout = holdout,
        holdout_confirmed = nrow(holdout) == 1L &&
            holdout$lower_95_recall[[1L]] >= request$target_recall
    )
}

hnsw_tune_matching_holdout <- function(diagnostics, selected) {
    diagnostics[
        diagnostics$split == "holdout" &
            diagnostics$m == selected$m[[1L]] &
            diagnostics$ef_construction == selected$ef_construction[[1L]] &
            diagnostics$ef_search == selected$ef_search[[1L]],
        , drop = FALSE
    ]
}

hnsw_tune_final_index <- function(request, selected) {
    if (is.null(selected)) {
        return(list(status = "no_confirmed_candidate", index = NULL,
            seconds = NA_real_))
    }
    if (!isTRUE(selected$holdout_confirmed)) {
        return(list(status = "holdout_not_confirmed", index = NULL,
            seconds = NA_real_))
    }
    if (!isTRUE(request$build_index)) {
        return(list(status = "confirmed_not_built", index = NULL,
            seconds = NA_real_))
    }
    estimate <- hnsw_tune_index_bytes(
        request$dims[[1L]], request$dims[[2L]], selected$m
    )
    if (estimate > request$memory_budget_bytes) {
        return(list(status = "confirmed_final_memory_budget_exceeded",
            index = NULL, seconds = NA_real_))
    }
    full <- hnsw_tune_as_float(request$data)
    started <- proc.time()[[3L]]
    index <- nn_faiss_hnsw_index_build_float32_cpp(
        full, selected$m, selected$ef_construction, selected$ef_search,
        "euclidean", "euclidean", request$n_threads
    )
    list(status = "confirmed_index_built", index = index,
        seconds = proc.time()[[3L]] - started)
}

hnsw_tune_index_bytes <- function(n, p, m) {
    as.numeric(n) * (4 * as.numeric(p) + 16 * as.numeric(m) + 64)
}

hnsw_tune_budget_metadata <- function(request, sweep) {
    list(
        max_builds = request$max_builds,
        time_limit = request$time_limit,
        elapsed_seconds = sweep$elapsed_seconds,
        complete = sweep$complete,
        stopping_criterion = paste0(
            "complete_grid_or_", format(request$time_limit),
            "_second_pilot_limit"
        ),
        time_limit_check = "between_build_candidates",
        memory_budget_bytes = request$memory_budget_bytes,
        build_candidates = nrow(request$grid$build),
        search_candidates = length(request$grid$search)
    )
}

hnsw_tune_workload_metadata <- function(request, sample) {
    list(
        objective = paste0(
            "projected_build_plus_expected_batches_times_query"
        ),
        query_rows_per_batch = request$workload_query_rows,
        expected_query_batches = request$expected_query_batches,
        pilot_reference_rows = nrow(sample$reference),
        timed_query_rows = nrow(sample$query),
        full_reference_rows = request$dims[[1L]],
        build_projection = "pilot_median_times_full_to_pilot_row_ratio",
        query_projection = "pilot_median_times_workload_to_timed_row_ratio",
        timing_statistic = "median",
        timing_repetitions = request$timing_repetitions
    )
}

hnsw_tune_provenance <- function(request) {
    list(
        evidence = "local_data_query_sample",
        backend = "faiss_hnsw",
        exact_reference = "faiss_flat_l2",
        fallback_allowed = FALSE,
        hostname = Sys.info()[["nodename"]],
        faissR_version = as.character(utils::packageVersion("faissR")),
        faiss = faiss_info_json_cpp(),
        n = request$dims[[1L]],
        p = request$dims[[2L]],
        baseline = request$baseline,
        index_base = 1L,
        timing_scope = "same_process_same_host_workload_selection"
    )
}

#' Query a locally tuned HNSW index
#'
#' @param object A confirmed `faissR_hnsw_tuning` object returned by
#'   [hnsw_tune()] with `build_index = TRUE`.
#' @param newdata Dense query matrix or `float::fl()` matrix.
#' @param k Number of neighbours. Defaults to the tuning value.
#' @param ef_search Runtime search effort. Defaults to the selected value.
#' @param n_threads Number of CPU threads. Defaults to the tuning value.
#' @param exclude_self Logical; remove row `i` from query row `i`. Use only
#'   when `newdata` is the original reference in the same row order.
#' @param ... Reserved for the `predict()` generic.
#' @return A `faissR_nn` result with one-based identifiers and selected-setting
#'   provenance. The fitted pointer is reused and no fallback is attempted.
#' @export
predict.faissR_hnsw_tuning <- function(
    object, newdata, k = object$k,
    ef_search = object$selected$ef_search,
    n_threads = object$n_threads, exclude_self = FALSE, ...
) {
    if (is.null(object$index)) {
        stop("The tuning object does not contain a confirmed fitted index.",
            call. = FALSE)
    }
    validate_dense_matrix_input(newdata, "newdata")
    query <- hnsw_tune_as_float(newdata)
    started <- proc.time()[[3L]]
    out <- nn_faiss_hnsw_index_search_float32_cpp(
        object$index, query,
        normalize_nn_positive_integer(k, "k", "`k` must be positive."),
        isTRUE(exclude_self),
        normalize_nn_positive_integer(
            ef_search, "ef_search", "`ef_search` must be positive."
        ),
        normalize_nn_threads(n_threads),
        "double"
    )
    search_seconds <- proc.time()[[3L]] - started
    adaptive_seconds <- object$workflow_total_seconds + search_seconds
    result <- finish_nn_result(
        out, "faiss_hnsw", as.integer(k), isTRUE(exclude_self),
        exact = FALSE, metric = "euclidean"
    )
    attr(result, "approximation") <- c(
        list(
            strategy = "faiss_IndexHNSWFlat",
            local_data_tuning = TRUE,
            no_fallback = TRUE,
            index_reused = TRUE,
            final_search_seconds = search_seconds,
            adaptive_end_to_end_seconds = adaptive_seconds
        ),
        object$selected[c("m", "ef_construction", "ef_search")]
    )
    attr(result, "hnsw_tuning") <- object[c(
        "status", "selected", "sampling", "budget", "workload",
        "provenance", "tuning_exact_seconds", "tuning_build_seconds",
        "tuning_search_seconds", "tuning_total_seconds",
        "final_build_seconds", "workflow_total_seconds"
    )]
    attr(result, "final_search_seconds") <- search_seconds
    attr(result, "adaptive_end_to_end_seconds") <- adaptive_seconds
    finish_float32_direct_result(result, out)
}

#' @export
print.faissR_hnsw_tuning <- function(x, ...) {
    cat("<faissR_hnsw_tuning>\n")
    cat("  status: ", x$status, "\n", sep = "")
    cat("  target: ", format(x$target_recall), "\n", sep = "")
    if (!is.null(x$guidance)) {
        cat("  guidance: ", x$guidance, "\n", sep = "")
    }
    if (!is.null(x$selected)) {
        cat(
            "  selected: M=", x$selected$m,
            ", efConstruction=", x$selected$ef_construction,
            ", efSearch=", x$selected$ef_search, "\n", sep = ""
        )
        cat("  holdout confirmed: ", x$selected$holdout_confirmed,
            "\n", sep = "")
        cat("  projected workload seconds: ",
            format(x$selected$projected_workload_seconds), "\n", sep = "")
    }
    invisible(x)
}

#' @export
summary.faissR_hnsw_tuning <- function(object, ...) {
    object$diagnostics
}
