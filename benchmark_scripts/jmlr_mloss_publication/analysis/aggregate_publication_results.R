#!/usr/bin/env Rscript

`%||%` <- function(x, y) {
  if (is.null(x) || !length(x) || is.na(x[[1L]])) y else x
}

parse_args <- function(args = commandArgs(trailingOnly = TRUE)) {
  out <- list()
  for (arg in args) {
    if (!startsWith(arg, "--")) next
    parts <- strsplit(sub("^--", "", arg), "=", fixed = TRUE)[[1L]]
    out[[parts[[1L]]]] <- if (length(parts) > 1L) paste(parts[-1L], collapse = "=") else "TRUE"
  }
  out
}

split_values <- function(x, default) {
  trimws(strsplit(x %||% default, ",", fixed = TRUE)[[1L]])
}

read_union <- function(files) {
  tables <- lapply(files, function(path) {
    x <- read.csv(path, stringsAsFactors = FALSE, check.names = FALSE)
    x$source_file <- normalizePath(path, mustWork = TRUE)
    x$source_mtime <- as.numeric(file.info(path)$mtime)
    x$run_root <- dirname(dirname(path))
    x
  })
  columns <- unique(unlist(lapply(tables, names), use.names = FALSE))
  tables <- lapply(tables, function(x) {
    for (name in setdiff(columns, names(x))) x[[name]] <- NA
    x[, columns, drop = FALSE]
  })
  do.call(rbind, tables)
}

latest_method_runs <- function(x) {
  suite <- ifelse(is.na(x$dataset_suite) | !nzchar(x$dataset_suite), "real", x$dataset_suite)
  key <- paste(x$backend, x$method_id, suite, sep = "\r")
  selected <- unlist(lapply(split(seq_len(nrow(x)), key), function(ii) {
    roots <- unique(x$run_root[ii])
    root_time <- vapply(roots, function(root) max(x$source_mtime[ii][x$run_root[ii] == root]), numeric(1))
    ii[x$run_root[ii] == roots[[which.max(root_time)]]]
  }), use.names = FALSE)
  x[sort(selected), , drop = FALSE]
}

expand_external_targets <- function(x, targets) {
  external <- x$implementation != "faissR" | is.na(x$target_recall)
  fixed <- x[!external, , drop = FALSE]
  ext <- x[external, , drop = FALSE]
  if (!nrow(ext)) return(fixed)
  expanded <- do.call(rbind, lapply(targets, function(target) {
    out <- ext
    out$target_recall <- target
    out
  }))
  rbind(fixed, expanded)
}

group_apply <- function(x, columns, fun) {
  key_data <- lapply(x[columns], function(value) {
    value <- as.character(value)
    value[is.na(value)] <- "<NA>"
    value
  })
  key <- interaction(key_data, drop = TRUE, lex.order = TRUE)
  rows <- lapply(split(x, key), fun)
  out <- do.call(rbind, rows)
  row.names(out) <- NULL
  out
}

robust_summary <- function(x, expected_seeds, expected_repeats) {
  columns <- c(
    "dataset", "dataset_suite", "backend", "metric", "k", "target_recall",
    "implementation", "method_id", "public_method", "kind", "n_threads"
  )
  group_apply(x, columns, function(part) {
    success <- part$status == "success"
    times <- suppressWarnings(as.numeric(part$time_sec[success]))
    recalls <- suppressWarnings(as.numeric(part$recall_at_k[success]))
    memory <- suppressWarnings(as.numeric(part$peak_rss_gb[success]))
    copies <- suppressWarnings(as.numeric(part$host_copy_sec[success]))
    target <- suppressWarnings(as.numeric(part$target_recall[[1L]]))
    seeds <- unique(part$validation_seed[success])
    expected_runs <- expected_seeds * expected_repeats
    complete <- sum(success) >= expected_runs && length(seeds) >= expected_seeds
    data.frame(
      part[1L, columns, drop = FALSE],
      n_rows = nrow(part),
      n_success = sum(success),
      n_failed = sum(!success),
      n_validation_seeds = length(seeds),
      expected_runs = expected_runs,
      complete_validation = complete,
      median_time_sec = if (any(is.finite(times))) median(times[is.finite(times)]) else NA_real_,
      iqr_time_sec = if (sum(is.finite(times)) > 1L) IQR(times[is.finite(times)]) else NA_real_,
      median_peak_rss_gb = if (any(is.finite(memory))) median(memory[is.finite(memory)]) else NA_real_,
      median_host_copy_sec = if (any(is.finite(copies))) median(copies[is.finite(copies)]) else NA_real_,
      mean_recall_at_k = if (any(is.finite(recalls))) mean(recalls[is.finite(recalls)]) else NA_real_,
      min_recall_at_k = if (any(is.finite(recalls))) min(recalls[is.finite(recalls)]) else NA_real_,
      target_met_all_runs = complete && length(recalls) >= expected_runs &&
        all(is.finite(recalls) & recalls >= target),
      stringsAsFactors = FALSE
    )
  })
}

rank_qualifying <- function(summary) {
  keys <- c("dataset", "dataset_suite", "backend", "metric", "k", "target_recall")
  group_apply(summary, keys, function(part) {
    qualifying <- part[part$complete_validation & part$target_met_all_runs & is.finite(part$median_time_sec), , drop = FALSE]
    qualifying <- qualifying[order(qualifying$median_time_sec, qualifying$method_id), , drop = FALSE]
    exact <- qualifying[
      qualifying$public_method %in% c("exact", "flat", "bruteforce") |
        grepl("_(exact|flat|bruteforce)$", qualifying$method_id), , drop = FALSE
    ]
    auto <- qualifying[grepl("_auto$", qualifying$method_id), , drop = FALSE]
    oracle <- qualifying[!grepl("_auto$", qualifying$method_id), , drop = FALSE]
    pick <- function(tbl, i, column, default = NA) {
      if (nrow(tbl) < i) default else tbl[[column]][[i]]
    }
    data.frame(
      part[1L, keys, drop = FALSE],
      n_complete_methods = sum(part$complete_validation),
      n_qualifying_methods = nrow(qualifying),
      fastest_method = pick(oracle, 1L, "method_id", NA_character_),
      fastest_time_sec = pick(oracle, 1L, "median_time_sec", NA_real_),
      fastest_recall = pick(oracle, 1L, "min_recall_at_k", NA_real_),
      second_method = pick(oracle, 2L, "method_id", NA_character_),
      second_time_sec = pick(oracle, 2L, "median_time_sec", NA_real_),
      exact_baseline_method = pick(exact, 1L, "method_id", NA_character_),
      exact_baseline_time_sec = pick(exact, 1L, "median_time_sec", NA_real_),
      auto_method = pick(auto, 1L, "method_id", NA_character_),
      auto_time_sec = pick(auto, 1L, "median_time_sec", NA_real_),
      auto_recall = pick(auto, 1L, "min_recall_at_k", NA_real_),
      auto_over_oracle = if (nrow(auto) && nrow(oracle)) auto$median_time_sec[[1L]] / oracle$median_time_sec[[1L]] else NA_real_,
      stringsAsFactors = FALSE
    )
  })
}

route_audit <- function(x) {
  success <- x$status == "success"
  auditable <- success & x$implementation == "faissR" &
    !is.na(x$result_backend) & nzchar(x$result_backend)
  requested_backend_bad <- success & !is.na(x$requested_backend) &
    nzchar(x$requested_backend) & x$requested_backend != x$backend
  resolved <- tolower(as.character(x$result_backend))
  cpu_bad <- auditable & x$backend == "cpu" & grepl("cuda|gpu|cuvs", resolved)
  cuda_bad <- auditable & x$backend == "cuda" & !grepl("cuda|gpu|cuvs", resolved)
  requested_method_bad <- success & x$implementation == "faissR" &
    !is.na(x$public_method) & !is.na(x$requested_method) &
    nzchar(x$requested_method) & x$requested_method != x$public_method
  bad <- requested_backend_bad | cpu_bad | cuda_bad | requested_method_bad
  out <- x[bad, , drop = FALSE]
  if (nrow(out)) {
    out$route_audit_reason <- paste(
      ifelse(requested_backend_bad[bad], "requested_backend_mismatch", ""),
      ifelse(cpu_bad[bad] | cuda_bad[bad], "resolved_device_mismatch", ""),
      ifelse(requested_method_bad[bad], "requested_method_mismatch", ""),
      sep = ";"
    )
  }
  out
}

compliance_table <- function(summary) {
  columns <- c("backend", "metric", "target_recall")
  group_apply(summary, columns, function(part) data.frame(
    part[1L, columns, drop = FALSE],
    evaluated_cells = nrow(part),
    complete_cells = sum(part$complete_validation),
    target_met_cells = sum(part$complete_validation & part$target_met_all_runs),
    completion_fraction = mean(part$complete_validation),
    target_attainment_fraction = if (any(part$complete_validation))
      mean(part$target_met_all_runs[part$complete_validation]) else NA_real_,
    stringsAsFactors = FALSE
  ))
}

write_report <- function(out_dir, files, combined, summary, best, route_errors) {
  lines <- c(
    "# Held-out publication evidence",
    "",
    paste0("- Source result files: ", length(files), "."),
    paste0("- Selected result rows: ", nrow(combined), "."),
    paste0("- Robust method cells: ", nrow(summary), "."),
    paste0("- Complete validation cells: ", sum(summary$complete_validation), "."),
    paste0("- Cells meeting recall in every run: ", sum(summary$complete_validation & summary$target_met_all_runs), "."),
    paste0("- Successful route mismatches: ", nrow(route_errors), "."),
    "",
    "A result is publication-eligible only when it contains the requested number of independent validation seeds and repetitions and reaches the recall target in every successful run. Failed, timed-out, unsupported, incomplete, and route-mismatched rows remain in the archive.",
    "",
    "`jss_auto_vs_oracle.csv` compares method = auto with the fastest independently requested qualifying method. Values above one in `auto_over_oracle` quantify the remaining automatic-selection regret.",
    "",
    paste0("Complete fastest/second-fastest blocks: ", sum(!is.na(best$fastest_method)), " of ", nrow(best), ".")
  )
  writeLines(lines, file.path(out_dir, "JSS_EVIDENCE_REPORT.md"))
}

main <- function() {
  args <- parse_args()
  root <- normalizePath(args$results_root %||% ".", mustWork = TRUE)
  out_dir <- normalizePath(args$out_dir %||% file.path(root, "analysis"), mustWork = FALSE)
  dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
  backend <- tolower(args$backend %||% "all")
  if (!backend %in% c("all", "cpu", "cuda")) stop("`backend` must be all, cpu, or cuda.", call. = FALSE)
  targets <- as.numeric(split_values(args$target_recalls, "0.9,0.95,0.99"))
  expected_seeds <- as.integer(args$expected_seeds %||% 2L)
  expected_repeats <- as.integer(args$expected_repeats %||% 3L)

  files <- list.files(root, pattern = "^jmlr_tuned_benchmark_results[.]csv$", recursive = TRUE, full.names = TRUE)
  files <- files[!grepl("/calibration/|/analysis/", files)]
  if (!length(files)) stop("No held-out publication result files were found under `results_root`.", call. = FALSE)
  combined <- read_union(files)
  if (backend != "all") combined <- combined[combined$backend == backend, , drop = FALSE]
  if (!nrow(combined)) stop("No result rows match the requested backend.", call. = FALSE)
  combined <- latest_method_runs(combined)
  combined <- expand_external_targets(combined, targets)
  combined <- combined[order(combined$dataset, combined$backend, combined$metric, combined$k,
                             combined$target_recall, combined$method_id,
                             combined$validation_seed, combined$repeat_id), , drop = FALSE]
  summary <- robust_summary(combined, expected_seeds, expected_repeats)
  best <- rank_qualifying(summary)
  routes <- route_audit(combined)
  compliance <- compliance_table(summary)

  write.csv(combined, file.path(out_dir, "jss_publication_results_combined.csv"), row.names = FALSE)
  write.csv(combined[combined$status != "success", , drop = FALSE], file.path(out_dir, "jss_failures_and_unsupported.csv"), row.names = FALSE)
  write.csv(summary, file.path(out_dir, "jss_robust_method_summary.csv"), row.names = FALSE)
  write.csv(best, file.path(out_dir, "jss_fastest_second_exact_and_auto.csv"), row.names = FALSE)
  write.csv(best[, c("dataset", "dataset_suite", "backend", "metric", "k", "target_recall",
                     "fastest_method", "fastest_time_sec", "auto_method", "auto_time_sec",
                     "auto_recall", "auto_over_oracle"), drop = FALSE],
            file.path(out_dir, "jss_auto_vs_oracle.csv"), row.names = FALSE)
  write.csv(compliance, file.path(out_dir, "jss_recall_compliance_table.csv"), row.names = FALSE)
  write.csv(routes, file.path(out_dir, "jss_successful_route_mismatches.csv"), row.names = FALSE)
  focus <- best[best$k == 30L & abs(best$target_recall - 0.99) < 1e-12, , drop = FALSE]
  write.csv(focus, file.path(out_dir, "jss_main_table_k30_recall099.csv"), row.names = FALSE)
  write_report(out_dir, files, combined, summary, best, routes)
  writeLines(capture.output(sessionInfo()), file.path(out_dir, "sessionInfo.txt"))
  outputs <- list.files(out_dir, full.names = TRUE)
  checksums <- data.frame(file = basename(outputs), md5 = unname(tools::md5sum(outputs)), stringsAsFactors = FALSE)
  write.csv(checksums, file.path(out_dir, "checksums.csv"), row.names = FALSE)
  cat("Wrote held-out evidence to ", out_dir, "\n", sep = "")
}

if (!identical(Sys.getenv("FAISSR_JSS_AGGREGATE_SOURCE_ONLY", unset = ""), "true")) main()
