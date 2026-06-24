#!/usr/bin/env Rscript

`%||%` <- function(x, y) {
  if (is.null(x) || length(x) == 0L || is.na(x[[1L]])) y else x
}

parse_args <- function(args = commandArgs(trailingOnly = TRUE)) {
  out <- list()
  for (arg in args) {
    if (!startsWith(arg, "--")) next
    kv <- strsplit(sub("^--", "", arg), "=", fixed = TRUE)[[1L]]
    key <- kv[[1L]]
    value <- if (length(kv) > 1L) paste(kv[-1L], collapse = "=") else "TRUE"
    out[[key]] <- value
  }
  out
}

shape_group <- function(n, p) {
  out <- rep("other", length(n))
  out[n < 50000L] <- "small_n"
  out[n >= 5000000L & p <= 64L] <- "huge_low_dim"
  out[n >= 50000L & n < 500000L & p <= 64L] <- "medium_low_dim"
  out[n >= 500000L & p <= 64L & out == "other"] <- "large_low_dim"
  out[n >= 50000L & p >= 256L] <- "large_high_dim"
  out
}

safe_median <- function(x) {
  x <- suppressWarnings(as.numeric(x))
  x <- x[is.finite(x)]
  if (!length(x)) return(NA_real_)
  stats::median(x)
}

safe_min <- function(x) {
  x <- suppressWarnings(as.numeric(x))
  x <- x[is.finite(x)]
  if (!length(x)) return(NA_real_)
  min(x)
}

summarise_groups <- function(x, keys) {
  if (!nrow(x)) return(data.frame())
  pieces <- split(x, interaction(x[keys], drop = TRUE, lex.order = TRUE))
  out <- do.call(rbind, lapply(pieces, function(part) {
    data.frame(
      part[1L, keys, drop = FALSE],
      rows = nrow(part),
      success = sum(part$status == "success"),
      meets_target = sum(isTRUE(part$meets_target) | part$meets_target, na.rm = TRUE),
      timeout = sum(part$status == "timeout"),
      failed = sum(part$status == "failed"),
      unavailable = sum(part$status == "unavailable"),
      unsupported = sum(part$status == "unsupported"),
      median_elapsed_sec = safe_median(part$elapsed_sec[part$status == "success"]),
      median_recall_at_k = safe_median(part$recall_at_k[part$status == "success"]),
      min_recall_at_k = safe_min(part$recall_at_k[part$status == "success"]),
      stringsAsFactors = FALSE
    )
  }))
  row.names(out) <- NULL
  out[do.call(order, out[keys]), , drop = FALSE]
}

select_recommendation_rows <- function(x) {
  keys <- c("dataset", "backend", "k", "target_recall_requested")
  pieces <- split(x, interaction(x[keys], drop = TRUE, lex.order = TRUE))
  out <- do.call(rbind, lapply(pieces, function(part) {
    ok <- part[part$status == "success" & part$meets_target, , drop = FALSE]
    if (nrow(ok)) {
      ok[order(ok$elapsed_sec, -ok$recall_at_k), , drop = FALSE][1L, , drop = FALSE]
    } else {
      success <- part[part$status == "success", , drop = FALSE]
      if (nrow(success)) {
        success[order(-success$recall_at_k, success$elapsed_sec), , drop = FALSE][1L, , drop = FALSE]
      } else {
        part[1L, , drop = FALSE]
      }
    }
  }))
  row.names(out) <- NULL
  out
}

md_table <- function(x, cols = names(x), digits = 4L, max_rows = 30L) {
  if (!nrow(x)) return("_No rows._")
  cols <- intersect(cols, names(x))
  if (!length(cols)) return("_No requested columns were available._")
  x <- x[seq_len(min(nrow(x), max_rows)), cols, drop = FALSE]
  for (name in names(x)) {
    if (is.numeric(x[[name]])) {
      x[[name]] <- ifelse(is.na(x[[name]]), "", format(round(x[[name]], digits), trim = TRUE))
    } else {
      x[[name]] <- ifelse(is.na(x[[name]]), "", as.character(x[[name]]))
    }
  }
  header <- paste0("| ", paste(names(x), collapse = " | "), " |")
  sep <- paste0("| ", paste(rep("---", ncol(x)), collapse = " | "), " |")
  body <- apply(x, 1L, function(row) paste0("| ", paste(row, collapse = " | "), " |"))
  paste(c(header, sep, body), collapse = "\n")
}

write_report <- function(out_dir, x, backend_summary, shape_summary, below_target, recs) {
  report_path <- file.path(out_dir, "hnsw_target_recall_report.md")
  lines <- c(
    "# HNSW Target-Recall Benchmark Summary",
    "",
    sprintf("- Source results: `%s`", file.path(out_dir, "float32_nn_benchmark_results.csv")),
    "- Metric: Euclidean distance",
    "- Method: `hnsw`",
    "- Backends: explicit `cpu` and explicit `cuda`; `backend = \"auto\"` is not used.",
    "- Targets: `target_recall = 0.9`, `0.95`, and `0.99`.",
    "- k values: `10`, `15`, `50`, and `100`.",
    "- Input: float32 dataset manifest rows.",
    "- Quality: sampled recall against exact KNN reference rows.",
    "- Timeout: 600 seconds per row unless the launcher was overridden.",
    "",
    "## Backend Summary",
    "",
    md_table(
      backend_summary,
      cols = c("backend", "target_recall_requested", "k", "rows", "success",
               "meets_target", "timeout", "median_elapsed_sec",
               "median_recall_at_k", "min_recall_at_k"),
      max_rows = 200L
    ),
    "",
    "## Shape Summary",
    "",
    md_table(
      shape_summary,
      cols = c("shape_group", "backend", "target_recall_requested", "k", "rows",
               "success", "meets_target", "timeout", "median_elapsed_sec",
               "median_recall_at_k", "min_recall_at_k"),
      max_rows = 240L
    ),
    "",
    "## Rows Below Target",
    "",
    md_table(
      below_target,
      cols = c("dataset", "shape_group", "backend", "k", "target_recall_requested",
               "elapsed_sec", "recall_at_k", "status", "tuning_rule"),
      max_rows = 80L
    ),
    "",
    "## Recommendation Rows",
    "",
    "For each dataset/backend/k/target, the recommendation row is the fastest successful row meeting the requested target. If no row met target, it is the highest-recall successful row, or the failed row when no success exists.",
    "",
    md_table(
      recs,
      cols = c("dataset", "shape_group", "backend", "k", "target_recall_requested",
               "elapsed_sec", "recall_at_k", "status", "tuning_rule",
               "hnsw_m", "hnsw_ef_construction", "hnsw_ef_search",
               "hnsw_graph_degree", "hnsw_intermediate_graph_degree", "hnsw_ef"),
      max_rows = 120L
    )
  )
  writeLines(lines, report_path)
  report_path
}

main <- function() {
  args <- parse_args()
  out_dir <- normalizePath(args$out_dir %||% getwd(), mustWork = FALSE)
  results_path <- normalizePath(
    args$results %||% file.path(out_dir, "float32_nn_benchmark_results.csv"),
    mustWork = TRUE
  )
  x <- read.csv(results_path, stringsAsFactors = FALSE)
  if (!nrow(x)) stop("No benchmark rows found.", call. = FALSE)
  required <- c("dataset", "backend", "k", "target_recall_requested", "status", "recall_at_k")
  missing <- setdiff(required, names(x))
  if (length(missing)) {
    stop(
      "Results file is missing required columns: ",
      paste(missing, collapse = ", "),
      call. = FALSE
    )
  }
  x$target_recall_requested <- suppressWarnings(as.numeric(x$target_recall_requested))
  x$recall_at_k <- suppressWarnings(as.numeric(x$recall_at_k))
  x$elapsed_sec <- suppressWarnings(as.numeric(x$elapsed_sec))
  x$shape_group <- shape_group(as.integer(x$n), as.integer(x$p))
  x$meets_target <- x$status == "success" &
    is.finite(x$recall_at_k) &
    is.finite(x$target_recall_requested) &
    x$recall_at_k >= x$target_recall_requested

  row_summary <- x[, intersect(
    c(
      "dataset", "n", "p", "shape_group", "backend", "method", "metric", "k",
      "target_recall_requested", "target_recall_actual", "status", "elapsed_sec",
      "peak_rss_gb", "recall_at_k", "median_recall_at_k", "min_recall_at_k",
      "meets_target", "tuning_rule", "hnsw_m", "hnsw_ef_construction",
      "hnsw_ef_search", "hnsw_graph_degree", "hnsw_intermediate_graph_degree",
      "hnsw_ef", "tuning_low_dim", "tuning_high_dim", "tuning_medium_n",
      "tuning_huge_low_dim", "tuning_runtime_guard", "error"
    ),
    names(x)
  ), drop = FALSE]
  backend_summary <- summarise_groups(x, c("backend", "target_recall_requested", "k"))
  shape_summary <- summarise_groups(x, c("shape_group", "backend", "target_recall_requested", "k"))
  below_target <- x[x$status == "success" & !x$meets_target, , drop = FALSE]
  below_target <- below_target[order(
    below_target$backend,
    below_target$target_recall_requested,
    below_target$dataset,
    below_target$k
  ), , drop = FALSE]
  recs <- select_recommendation_rows(x)
  recs <- recs[order(recs$dataset, recs$backend, recs$k, recs$target_recall_requested), , drop = FALSE]

  write.csv(row_summary, file.path(out_dir, "hnsw_target_recall_rows.csv"), row.names = FALSE)
  write.csv(backend_summary, file.path(out_dir, "hnsw_target_recall_backend_summary.csv"), row.names = FALSE)
  write.csv(shape_summary, file.path(out_dir, "hnsw_target_recall_shape_summary.csv"), row.names = FALSE)
  write.csv(below_target, file.path(out_dir, "hnsw_target_recall_below_target.csv"), row.names = FALSE)
  write.csv(recs, file.path(out_dir, "hnsw_target_recall_recommendations.csv"), row.names = FALSE)
  report_path <- write_report(out_dir, x, backend_summary, shape_summary, below_target, recs)

  cat("Wrote HNSW target-recall summaries to ", out_dir, "\n", sep = "")
  cat("Report: ", report_path, "\n", sep = "")
}

main()
