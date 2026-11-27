# Private large-KNN planning from a source-locked, independently audited grid.
# A sampled-query recall pass is a candidate for a new live calibration, not
# a guarantee for different queries, sources, or a full self-KNN graph.
nn_large_knn_tune_audited <- function(
    n, p, k, backend, storage_mode, host_memory_bytes,
    device_memory_bytes = NA_real_, gpu_count = 0L,
    cpu_threads_available = 16L, target_recall = 0.99,
    audited_probes = NULL, source_sha = NULL,
    input_sha256 = NULL, multi_gpu_mode = NULL
) {
    plan <- nn_large_knn_resource_plan(
        n = n, p = p, k = k, backend = backend,
        storage_mode = storage_mode,
        host_memory_bytes = host_memory_bytes,
        device_memory_bytes = device_memory_bytes,
        gpu_count = gpu_count,
        cpu_threads_available = cpu_threads_available,
        target_recall = target_recall
    )
    plan$tuning_status <- "no_audited_matching_grid"
    plan$calibration_source_sha <- NA_character_
    plan$calibration_input_sha256 <- NA_character_
    plan$calibration_query_rows_sha256 <- NA_character_
    plan$calibration_reference_scope <- NA_character_
    plan$calibration_query_count <- NA_integer_
    plan$calibration_attempted_cells <- 0L
    plan$calibration_passing_cells <- 0L
    plan$measured_search_seconds_256_queries <- NA_real_
    plan$measured_index_build_seconds <- NA_real_
    plan$calibration_job_id <- NA_character_
    if (is.null(audited_probes)) return(plan)
    if (backend != "cuda" || storage_mode != "resident") {
        stop("Audited IVF probes require resident CUDA.", call. = FALSE)
    }
    if (is.null(multi_gpu_mode)) {
        multi_gpu_mode <- if (gpu_count > 1L) "shard" else "replicate"
    }
    if (length(multi_gpu_mode) != 1L ||
        !multi_gpu_mode %in% c("replicate", "shard") ||
        gpu_count == 1L && multi_gpu_mode == "shard") {
        stop("Invalid audited multi-GPU mode.", call. = FALSE)
    }
    hash_ok <- function(value, size) {
        is.character(value) && length(value) == 1L && !is.na(value) &&
            grepl(sprintf("^[0-9a-f]{%d}$", size), value)
    }
    if (!hash_ok(source_sha, 40L) || !hash_ok(input_sha256, 64L)) {
        stop("Source and input hashes are required for audited tuning.",
             call. = FALSE)
    }
    fields <- c("n", "p", "k", "gpu_count", "mode", "nlist",
                "nprobe", "measured_recall", "search_seconds_median",
                "index_build_seconds", "query_count",
                "search_repetitions", "source_sha", "input_sha256",
                "query_rows_sha256", "audit_pass", "reference_scope",
                "timing_scope", "job_id")
    if (!is.data.frame(audited_probes) ||
        !all(fields %in% names(audited_probes))) {
        stop("Missing audited probe-grid columns.", call. = FALSE)
    }
    matching <- stats::complete.cases(audited_probes[c(
        "n", "p", "k", "gpu_count", "mode", "source_sha",
        "input_sha256")])
    rows <- audited_probes[
        matching & audited_probes$n == n & audited_probes$p == p &
        audited_probes$k == k &
        audited_probes$gpu_count == gpu_count &
        audited_probes$mode == multi_gpu_mode &
        audited_probes$source_sha == source_sha &
        audited_probes$input_sha256 == input_sha256,
        , drop = FALSE
    ]
    if (!nrow(rows)) return(plan)
    numeric_fields <- c("nlist", "nprobe", "measured_recall",
                        "search_seconds_median", "index_build_seconds",
                        "query_count", "search_repetitions")
    identity_fields <- c("nlist", "nprobe", "query_rows_sha256",
                         "audit_pass", "reference_scope", "timing_scope",
                         "job_id")
    measured_fields <- c("measured_recall", "search_seconds_median",
                         "index_build_seconds", "query_count",
                         "search_repetitions")
    observed <- rows[measured_fields]
    if (!is.logical(rows$audit_pass) ||
        !all(stats::complete.cases(rows[identity_fields])) ||
        any(!vapply(rows[numeric_fields], is.numeric, logical(1L))) ||
        any(!vapply(rows[numeric_fields], function(value)
            all(is.finite(value[!is.na(value)])), logical(1L))) ||
        any(!stats::complete.cases(observed[rows$audit_pass, ,
                                             drop = FALSE])) ||
        any(rows$nlist != floor(rows$nlist) | rows$nlist < 1 |
            rows$nlist > n) ||
        any(rows$nprobe != floor(rows$nprobe) | rows$nprobe < 1 |
            rows$nprobe > rows$nlist) ||
        any(rows$measured_recall < 0 | rows$measured_recall > 1,
            na.rm = TRUE) ||
        any(rows$search_seconds_median <= 0, na.rm = TRUE) ||
        any(rows$index_build_seconds <= 0, na.rm = TRUE) ||
        any(rows$query_count != 256L, na.rm = TRUE) ||
        any(rows$search_repetitions[rows$audit_pass] < 3L) ||
        any(rows$search_repetitions[!rows$audit_pass] < 0L,
            na.rm = TRUE) ||
        any(rows$reference_scope != "full_database_exact") ||
        any(rows$timing_scope !=
            "index_build_plus_persistent_search_256_queries") ||
        length(unique(rows$query_rows_sha256)) != 1L ||
        !hash_ok(rows$query_rows_sha256[[1L]], 64L) ||
        length(unique(rows$job_id)) != 1L ||
        anyDuplicated(rows[c("nlist", "nprobe")])) {
        stop("Audited probe grid is incomplete or mixes scopes.",
             call. = FALSE)
    }
    plan$calibration_source_sha <- source_sha
    plan$calibration_input_sha256 <- input_sha256
    plan$calibration_query_rows_sha256 <- rows$query_rows_sha256[[1L]]
    plan$calibration_reference_scope <- "full_database_exact"
    plan$calibration_query_count <- 256L
    plan$calibration_attempted_cells <- nrow(rows)
    plan$calibration_passing_cells <- sum(rows$audit_pass &
        rows$measured_recall >= target_recall, na.rm = TRUE)
    passed <- rows[rows$audit_pass &
        rows$measured_recall >= target_recall, , drop = FALSE]
    if (!nrow(passed)) {
        plan$tuning_status <- "audited_grid_has_no_passing_cell"
        return(plan)
    }
    # Build time and 256-query persistent search have the same scope in every
    # row. This is a fixed-query cost choice, not a full graph time estimate.
    cost <- passed$index_build_seconds + passed$search_seconds_median
    selected <- passed[order(cost, passed$nprobe, passed$nlist)[[1L]],
                       , drop = FALSE]
    plan$tuning_status <-
        "measured_fixed_query_candidate_requires_live_recall_check"
    plan$selected_nlist <- as.integer(selected$nlist[[1L]])
    plan$selected_nprobe <- as.integer(selected$nprobe[[1L]])
    plan$measured_recall <- selected$measured_recall[[1L]]
    plan$measured_search_seconds_256_queries <-
        selected$search_seconds_median[[1L]]
    plan$measured_index_build_seconds <-
        selected$index_build_seconds[[1L]]
    plan$calibration_job_id <- as.character(selected$job_id[[1L]])
    plan
}
