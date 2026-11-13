# Private planning helpers for measured, fixed-query CUDA IVF campaigns.
# They do not claim target recall without an exact-reference audit.

nn_cuda_ivf_large_candidates <- function(
    n, k, nlist_multiplier = c(0.5, 1, 2, 4, 8),
    probe_fraction = c(1 / 32, 1 / 16, 1 / 8, 1 / 4,
                       1 / 2, 3 / 4, 1)
) {
    if (length(n) != 1L || !is.finite(n) || n < 2 || n > .Machine$integer.max ||
        length(k) != 1L || !is.finite(k) || k < 1 || k >= n) {
        stop("Invalid n or k for CUDA IVF candidate planning.", call. = FALSE)
    }
    if (any(!is.finite(nlist_multiplier) | nlist_multiplier <= 0) ||
        any(!is.finite(probe_fraction) | probe_fraction <= 0 |
            probe_fraction > 1)) {
        stop("Invalid IVF candidate multipliers.", call. = FALSE)
    }
    max_lists <- min(n, 65536)
    lists <- unique(as.integer(pmax(1, pmin(max_lists,
        2^round(log2(sqrt(n) * nlist_multiplier))
    ))))
    grid <- expand.grid(
        nlist = lists, probe_fraction = probe_fraction,
        KEEP.OUT.ATTRS = FALSE
    )
    grid$nprobe <- as.integer(pmax(1, pmin(
        grid$nlist, ceiling(grid$nlist * grid$probe_fraction)
    )))
    grid <- unique(grid[c("nlist", "nprobe")])
    grid <- grid[order(grid$nlist, grid$nprobe), , drop = FALSE]
    rownames(grid) <- NULL
    grid
}

nn_cuda_ivf_large_extrapolate <- function(
    observations, target_n, p, k, gpu_count, mode,
    gpu_memory_bytes, target_recall = 0.99
) {
    required <- c(
        "n", "p", "k", "gpu_count", "mode", "nlist", "nprobe",
        "recall", "knn_seconds", "device_memory_bytes",
        "peak_rss_bytes", "full_self_performed", "audit_pass",
        "reference_scope", "timing_scope", "query_count", "source_sha",
        "input_sha256", "query_rows_sha256", "query_rows_max",
        "database_selection"
    )
    if (!is.data.frame(observations) ||
        !all(required %in% names(observations))) {
        stop("Missing audited IVF observation columns.", call. = FALSE)
    }
    if (length(target_n) != 1L || !is.finite(target_n) ||
        target_n < 1e8 || target_n > .Machine$integer.max ||
        length(p) != 1L || !is.finite(p) || p < 1 ||
        length(k) != 1L || !is.finite(k) || k < 1 ||
        length(gpu_count) != 1L || !is.finite(gpu_count) || gpu_count < 1 ||
        length(mode) != 1L || !mode %in% c("replicate", "shard") ||
        length(gpu_memory_bytes) != 1L ||
        !is.finite(gpu_memory_bytes) || gpu_memory_bytes <= 0 ||
        length(target_recall) != 1L || !is.finite(target_recall) ||
        target_recall <= 0 || target_recall > 1) {
        stop("Invalid large IVF extrapolation inputs.", call. = FALSE)
    }
    complete <- stats::complete.cases(observations[required])
    rows <- observations[
        complete & observations$audit_pass &
        observations$reference_scope == "full_database_exact" &
        observations$p == p & observations$k == k &
        observations$gpu_count == gpu_count & observations$mode == mode &
        is.finite(observations$recall) &
        observations$recall >= target_recall &
        is.finite(observations$knn_seconds) &
        observations$knn_seconds > 0 &
        is.finite(observations$device_memory_bytes) &
        observations$device_memory_bytes > 0 &
        is.finite(observations$peak_rss_bytes) &
        observations$peak_rss_bytes > 0 &
        !observations$full_self_performed &
        is.finite(observations$nlist) & observations$nlist >= 1 &
        is.finite(observations$nprobe) & observations$nprobe >= 1,
        , drop = FALSE
    ]
    if (!nrow(rows) || length(unique(rows$source_sha)) != 1L ||
        length(unique(rows$input_sha256)) != 1L ||
        !grepl("^[0-9a-f]{40}$", rows$source_sha[[1L]]) ||
        !grepl("^[0-9a-f]{64}$", rows$input_sha256[[1L]])) {
        stop("A single source- and input-locked recall cohort is required.",
             call. = FALSE)
    }
    if (length(unique(rows$query_rows_sha256)) != 1L ||
        !grepl("^[0-9a-f]{64}$", rows$query_rows_sha256[[1L]]) ||
        length(unique(rows$database_selection)) != 1L ||
        rows$database_selection[[1L]] != "prefix_first_n_rows" ||
        any(!is.finite(rows$query_rows_max)) ||
        any(rows$query_rows_max < 1) ||
        any(rows$query_rows_max > min(rows$n))) {
        stop(paste("Anchors require identical query rows within",
                   "nested input prefixes."), call. = FALSE)
    }
    if (length(unique(rows$timing_scope)) != 1L ||
        length(unique(rows$query_count)) != 1L ||
        rows$query_count[[1L]] < 1L) {
        stop("Anchors require one timing scope and fixed query count.",
             call. = FALSE)
    }
    by_n <- split(rows, rows$n)
    selected <- do.call(rbind, lapply(by_n, function(group) {
        group[which.min(group$knn_seconds), , drop = FALSE]
    }))
    selected <- selected[order(selected$n), , drop = FALSE]
    if (nrow(selected) < 2L || min(selected$n) > 1e7 ||
        max(selected$n) < 1e8 || target_n <= max(selected$n) ||
        target_n > 10 * max(selected$n)) {
        stop("Passing 10M and 100M anchors are required for extrapolation.",
             call. = FALSE)
    }
    # Log-linear power-law projection; no predicted recall is inferred.
    log_project <- function(values) {
        x <- log(selected$n)
        y <- log(values)
        coefficients <- stats::coef(stats::lm(y ~ x))
        exp(coefficients[[1L]] + coefficients[[2L]] * log(target_n))
    }
    nlist <- as.integer(max(1, min(65536, round(log_project(selected$nlist)))))
    fraction <- log_project(selected$nprobe / selected$nlist)
    nprobe <- as.integer(max(1, min(nlist, ceiling(nlist * fraction))))
    raw_bytes <- (4 * target_n * p + 8 * target_n) /
        if (identical(mode, "shard")) gpu_count else 1
    observed_raw <- (4 * selected$n * p + 8 * selected$n) /
        if (identical(mode, "shard")) gpu_count else 1
    peak_factor <- max(selected$device_memory_bytes / observed_raw)
    estimated_peak <- raw_bytes * peak_factor
    list(
        status = "extrapolated_not_validated",
        source_sha = selected$source_sha[[1L]],
        input_sha256 = selected$input_sha256[[1L]],
        query_rows_sha256 = selected$query_rows_sha256[[1L]],
        database_selection = selected$database_selection[[1L]],
        timing_scope = selected$timing_scope[[1L]],
        query_count = selected$query_count[[1L]],
        target_n = target_n, p = p, k = k,
        gpu_count = gpu_count, mode = mode,
        target_recall = target_recall,
        nlist = nlist, nprobe = nprobe,
        projected_knn_seconds = log_project(selected$knn_seconds),
        projected_peak_host_bytes = log_project(selected$peak_rss_bytes),
        raw_index_bytes_per_device = raw_bytes,
        projected_peak_device_bytes = estimated_peak,
        gpu_memory_bytes = gpu_memory_bytes,
        device_memory_admission = if (estimated_peak <= gpu_memory_bytes) {
            "projection_within_capacity"
        } else "projection_exceeds_capacity",
        full_self_output_supported = target_n * (k + 1) <=
            .Machine$integer.max,
        anchor_rows = selected[, c(
            "n", "nlist", "nprobe", "recall", "knn_seconds",
            "device_memory_bytes", "peak_rss_bytes"
        ), drop = FALSE],
        recall_at_target = NA_real_
    )
}
