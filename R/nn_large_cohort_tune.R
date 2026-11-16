# Private large-IVF tuning from independently audited, matched query cohorts.
# A selected setting remains a fixed-query candidate until full-graph audit.
nn_large_knn_tune_cohorts <- function(
    n, p, k, backend, storage_mode, host_memory_bytes,
    device_memory_bytes = NA_real_, gpu_count = 0L,
    cpu_threads_available = 16L, target_recall = 0.99,
    audited_cohorts = NULL, source_archive_sha256 = NULL,
    input_sha256 = NULL, multi_gpu_mode = NULL,
    minimum_row_recall = 0.90
) {
    plan <- nn_large_knn_resource_plan(
        n = n, p = p, k = k, backend = backend,
        storage_mode = storage_mode, host_memory_bytes = host_memory_bytes,
        device_memory_bytes = device_memory_bytes,
        gpu_count = gpu_count,
        cpu_threads_available = cpu_threads_available,
        target_recall = target_recall
    )
    plan$tuning_status <- "no_matching_audited_cohorts"
    plan$calibration_source_archive_sha256 <- NA_character_
    plan$calibration_input_sha256 <- NA_character_
    plan$calibration_cohort_count <- 0L
    plan$calibration_attempted_cells <- 0L
    plan$calibration_failed_cells <- 0L
    plan$calibration_passing_probes <- integer()
    plan$calibration_recall_scope <- NA_character_
    plan$measured_minimum_cohort_recall <- NA_real_
    plan$measured_minimum_row_recall <- NA_real_
    plan$measured_maximum_cohort_search_seconds <- NA_real_
    if (length(minimum_row_recall) != 1L ||
        !is.numeric(minimum_row_recall) ||
        !is.finite(minimum_row_recall) ||
        minimum_row_recall < 0 || minimum_row_recall > 1) {
        stop("Invalid minimum row recall.", call. = FALSE)
    }
    if (is.null(audited_cohorts)) return(plan)
    if (backend != "cuda") {
        stop("Audited IVF cohorts require CUDA.", call. = FALSE)
    }
    if (is.null(multi_gpu_mode)) {
        multi_gpu_mode <- if (gpu_count > 1L) "shard" else "single"
    }
    if (length(multi_gpu_mode) != 1L ||
        !multi_gpu_mode %in% c("single", "replicate", "shard") ||
        (gpu_count == 1L) != (multi_gpu_mode == "single")) {
        stop("Invalid audited multi-GPU mode.", call. = FALSE)
    }
    hash_ok <- function(value) {
        is.character(value) && length(value) == 1L && !is.na(value) &&
            grepl("^[0-9a-f]{64}$", value)
    }
    if (!hash_ok(source_archive_sha256) || !hash_ok(input_sha256)) {
        stop("Source archive and input SHA-256 are required.", call. = FALSE)
    }
    required <- c(
        "n", "p", "k", "backend", "storage_mode", "gpu_count", "mode",
        "nlist", "nprobe", "cohort", "query_rows_sha256",
        "source_archive_sha256", "input_sha256", "audit_pass",
        "reference_scope", "recall_scope", "timing_scope",
        "query_count", "search_repetitions", "measured_recall",
        "minimum_tie_aware_row_recall", "search_seconds_median",
        "index_build_seconds", "backend_used", "device_count"
    )
    if (!is.data.frame(audited_cohorts) ||
        !all(required %in% names(audited_cohorts))) {
        stop("Missing audited cohort-grid columns.", call. = FALSE)
    }
    identity <- c("n", "p", "k", "backend", "storage_mode",
                  "gpu_count", "mode", "source_archive_sha256",
                  "input_sha256")
    complete <- stats::complete.cases(audited_cohorts[identity])
    rows <- audited_cohorts[
        complete & audited_cohorts$n == n & audited_cohorts$p == p &
        audited_cohorts$k == k & audited_cohorts$backend == backend &
        audited_cohorts$storage_mode == storage_mode &
        audited_cohorts$gpu_count == gpu_count &
        audited_cohorts$mode == multi_gpu_mode &
        audited_cohorts$source_archive_sha256 == source_archive_sha256 &
        audited_cohorts$input_sha256 == input_sha256,
        , drop = FALSE
    ]
    if (!nrow(rows)) return(plan)
    numeric_fields <- c("nlist", "nprobe", "query_count",
                        "search_repetitions", "measured_recall",
                        "minimum_tie_aware_row_recall",
                        "search_seconds_median", "index_build_seconds",
                        "device_count")
    required_identity <- c("nlist", "nprobe", "cohort",
                           "query_rows_sha256", "audit_pass",
                           "reference_scope", "recall_scope",
                           "timing_scope", "backend_used", "device_count")
    observed <- c("measured_recall", "minimum_tie_aware_row_recall",
                  "search_seconds_median", "index_build_seconds",
                  "query_count", "search_repetitions")
    if (!is.logical(rows$audit_pass) || anyNA(rows$audit_pass) ||
        any(!stats::complete.cases(rows[required_identity])) ||
        any(!vapply(rows[numeric_fields], is.numeric, logical(1L))) ||
        any(!vapply(rows[numeric_fields], function(value)
            all(is.finite(value[!is.na(value)])), logical(1L))) ||
        any(!stats::complete.cases(rows[rows$audit_pass, observed,
                                             drop = FALSE])) ||
        any(rows$nlist != floor(rows$nlist) | rows$nlist < 1 |
            rows$nlist > n) ||
        any(rows$nprobe != floor(rows$nprobe) | rows$nprobe < 1 |
            rows$nprobe > rows$nlist) ||
        any(rows$device_count != gpu_count) ||
        any(rows$query_count[rows$audit_pass] < 1L) ||
        any(rows$search_repetitions[rows$audit_pass] < 3L) ||
        any(rows$measured_recall < 0 | rows$measured_recall > 1,
            na.rm = TRUE) ||
        any(rows$minimum_tie_aware_row_recall < 0 |
            rows$minimum_tie_aware_row_recall > 1, na.rm = TRUE) ||
        any(rows$search_seconds_median <= 0, na.rm = TRUE) ||
        any(rows$index_build_seconds <= 0, na.rm = TRUE) ||
        any(rows$reference_scope != "full_database_exact") ||
        any(!rows$recall_scope %in% c(
            "nonself_strict_id_at_k",
            "nonself_distance_threshold_at_k")) ||
        length(unique(rows$recall_scope)) != 1L ||
        any(rows$timing_scope != "persistent_search_only") ||
        any(!vapply(rows$query_rows_sha256, hash_ok, logical(1L))) ||
        any(!nzchar(rows$cohort)) ||
        anyDuplicated(rows[c("nlist", "nprobe", "cohort")])) {
        stop("Audited cohort grid is incomplete or mixes scopes.",
             call. = FALSE)
    }
    cohorts <- unique(as.character(rows$cohort))
    for (cohort in cohorts) {
        same <- rows[rows$cohort == cohort, , drop = FALSE]
        if (length(unique(same$query_rows_sha256)) != 1L ||
            length(unique(same$query_count)) != 1L ||
            length(unique(same$backend_used)) != 1L ||
            any(same$backend_used != "faiss_gpu_persistent_query")) {
            stop("A cohort changes query or backend identity.",
                 call. = FALSE)
        }
    }
    cohort_hashes <- vapply(cohorts, function(cohort)
        rows$query_rows_sha256[match(cohort, rows$cohort)],
        character(1L))
    if (anyDuplicated(cohort_hashes)) {
        stop("Query cohorts must have distinct row identities.",
             call. = FALSE)
    }
    probes <- unique(rows[c("nlist", "nprobe")])
    if (nrow(rows) != nrow(probes) * length(cohorts) ||
        any(vapply(seq_len(nrow(probes)), function(i) {
            selected <- rows[rows$nlist == probes$nlist[[i]] &
                rows$nprobe == probes$nprobe[[i]], , drop = FALSE]
            !setequal(selected$cohort, cohorts)
        }, logical(1L)))) {
        stop("Every probe must retain every attempted cohort.",
             call. = FALSE)
    }
    plan$calibration_source_archive_sha256 <- source_archive_sha256
    plan$calibration_input_sha256 <- input_sha256
    plan$calibration_cohort_count <- length(cohorts)
    plan$calibration_attempted_cells <- nrow(rows)
    plan$calibration_failed_cells <- sum(!rows$audit_pass)
    plan$calibration_recall_scope <- rows$recall_scope[[1L]]
    if (length(cohorts) < 2L) {
        plan$tuning_status <- "insufficient_independent_cohorts"
        return(plan)
    }
    eligible <- lapply(seq_len(nrow(probes)), function(i) {
        block <- rows[rows$nlist == probes$nlist[[i]] &
            rows$nprobe == probes$nprobe[[i]], , drop = FALSE]
        if (all(block$audit_pass) &&
            min(block$measured_recall) >= target_recall &&
            min(block$minimum_tie_aware_row_recall) >= minimum_row_recall) {
            data.frame(nlist = probes$nlist[[i]],
                       nprobe = probes$nprobe[[i]],
                       worst_recall = min(block$measured_recall),
                       worst_row_recall = min(
                           block$minimum_tie_aware_row_recall),
                       worst_search_seconds = max(
                           block$search_seconds_median),
                       worst_build_seconds = max(
                           block$index_build_seconds))
        } else NULL
    })
    eligible <- do.call(rbind, eligible)
    if (is.null(eligible)) {
        plan$tuning_status <- "audited_cohorts_have_no_passing_probe"
        return(plan)
    }
    plan$calibration_passing_probes <- as.integer(eligible$nprobe)
    selected <- eligible[order(eligible$worst_build_seconds +
        eligible$worst_search_seconds, eligible$nprobe,
        eligible$nlist)[[1L]], , drop = FALSE]
    plan$selected_nlist <- as.integer(selected$nlist[[1L]])
    plan$selected_nprobe <- as.integer(selected$nprobe[[1L]])
    plan$measured_recall <- selected$worst_recall[[1L]]
    plan$measured_minimum_cohort_recall <- selected$worst_recall[[1L]]
    plan$measured_minimum_row_recall <- selected$worst_row_recall[[1L]]
    plan$measured_maximum_cohort_search_seconds <-
        selected$worst_search_seconds[[1L]]
    plan$tuning_status <-
        "measured_fixed_query_candidate_requires_live_graph_audit"
    plan
}
