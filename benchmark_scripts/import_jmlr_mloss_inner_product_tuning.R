#!/usr/bin/env Rscript

args <- commandArgs(trailingOnly = TRUE)
results_root <- if (length(args) >= 1L) args[[1L]] else Sys.getenv(
  "FAISSR_JMLR_MLOSS_RESULTS",
  unset = "faissR_JMLR_MLOSS/calibration"
)
output_file <- if (length(args) >= 2L) args[[2L]] else file.path(
  "benchmark_scripts",
  "jmlr_mloss_inner_product_shape_tuning_defaults.csv"
)
header_file <- if (length(args) >= 3L) args[[3L]] else file.path(
  "src",
  "nn_jmlr_inner_product_overrides.hpp"
)

if (!dir.exists(results_root)) {
  stop("Benchmark result directory does not exist: ", results_root, call. = FALSE)
}

report_files <- Sys.glob(file.path(results_root, "*", "faissR_*", "*_tuning_report.md"))
run_dirs <- unique(dirname(report_files))
result_files <- unlist(lapply(
  run_dirs,
  function(path) Sys.glob(file.path(path, "*_tuning_results.csv"))
), use.names = FALSE)
if (!length(result_files)) {
  stop("No report-backed tuning result files were found under: ", results_root, call. = FALSE)
}

read_result <- function(path) {
  out <- read.csv(path, check.names = FALSE, stringsAsFactors = FALSE)
  out$source_file <- normalizePath(path, winslash = "/", mustWork = TRUE)
  out
}

results <- do.call(rbind, lapply(result_files, read_result))
results <- results[
  results$metric == "inner_product" & results$candidate_kind == "manual",
  ,
  drop = FALSE
]
if (!nrow(results)) {
  stop("No manual inner-product candidates were found.", call. = FALSE)
}

expected_by_shape <- split(results$dataset, results$shape_group)
expected_by_shape <- lapply(expected_by_shape, function(x) sort(unique(x)))
targets <- c(0.90, 0.95, 0.99)

finite_number <- function(x, default = NA_real_) {
  x <- suppressWarnings(as.numeric(x))
  if (!length(x) || !is.finite(x[[1L]])) default else unname(x[[1L]])
}

first_value <- function(x, default = NA) {
  keep <- !is.na(x)
  if (is.character(x)) keep <- keep & nzchar(x)
  if (!any(keep)) default else unname(x[which(keep)[[1L]]])
}

candidate_summary <- function(part, expected_datasets) {
  successful <- part[
    part$status == "success" & is.finite(part$recall_at_k) &
      is.finite(part$elapsed_sec),
    ,
    drop = FALSE
  ]
  observed <- sort(unique(successful$dataset))
  complete <- identical(observed, expected_datasets)
  data.frame(
    candidate_id = first_value(part$candidate_id, "unknown"),
    successful_datasets = length(observed),
    expected_datasets = length(expected_datasets),
    complete_shape_coverage = complete,
    min_recall_at_k = if (nrow(successful)) min(successful$recall_at_k) else NA_real_,
    mean_recall_at_k = if (nrow(successful)) mean(successful$recall_at_k) else NA_real_,
    mean_elapsed_sec = if (nrow(successful)) mean(successful$elapsed_sec) else Inf,
    max_elapsed_sec = if (nrow(successful)) max(successful$elapsed_sec) else Inf,
    stringsAsFactors = FALSE
  )
}

selected <- list()
audit <- list()
group_keys <- unique(results[c("backend", "method", "shape_group", "k")])
for (group_index in seq_len(nrow(group_keys))) {
  key <- group_keys[group_index, , drop = FALSE]
  rownames(key) <- NULL
  part <- results[
    results$backend == key$backend & results$method == key$method &
      results$shape_group == key$shape_group & results$k == key$k,
    ,
    drop = FALSE
  ]
  expected_datasets <- expected_by_shape[[key$shape_group]]
  candidates <- split(part, part$candidate_id)
  summaries <- do.call(rbind, lapply(
    candidates,
    candidate_summary,
    expected_datasets = expected_datasets
  ))
  rownames(summaries) <- NULL

  for (target in targets) {
    summaries$target_recall <- target
    summaries$meets_target <- summaries$complete_shape_coverage &
      is.finite(summaries$min_recall_at_k) &
      summaries$min_recall_at_k + 1e-12 >= target
    audit[[length(audit) + 1L]] <- cbind(key, summaries)

    eligible <- summaries[summaries$meets_target, , drop = FALSE]
    if (nrow(eligible)) {
      eligible <- eligible[order(
        eligible$mean_elapsed_sec,
        eligible$max_elapsed_sec,
        -eligible$min_recall_at_k,
        eligible$candidate_id
      ), , drop = FALSE]
      chosen <- eligible[1L, , drop = FALSE]
      basis <- "fastest_meeting_target_complete_shape_coverage"
      target_met <- TRUE
    } else {
      eligible <- summaries[summaries$complete_shape_coverage, , drop = FALSE]
      if (!nrow(eligible)) next
      eligible <- eligible[order(
        -eligible$min_recall_at_k,
        eligible$mean_elapsed_sec,
        eligible$max_elapsed_sec,
        eligible$candidate_id
      ), , drop = FALSE]
      chosen <- eligible[1L, , drop = FALSE]
      basis <- "best_recall_below_target_complete_shape_coverage"
      target_met <- FALSE
    }

    source_rows <- candidates[[chosen$candidate_id]]
    source_rows <- source_rows[
      source_rows$status == "success" & source_rows$dataset %in% expected_datasets,
      ,
      drop = FALSE
    ]
    representative <- source_rows[which.min(source_rows$elapsed_sec), , drop = FALSE]
    selected[[length(selected) + 1L]] <- data.frame(
      backend = key$backend,
      method = key$method,
      metric = "inner_product",
      shape_group = key$shape_group,
      k_bucket = as.integer(key$k),
      target_recall = target,
      target_code = as.integer(round(100 * target)),
      candidate_id = unname(chosen$candidate_id),
      recommendation_basis = basis,
      benchmark_target_met = target_met,
      expected_dataset_count = length(expected_datasets),
      successful_dataset_count = chosen$successful_datasets,
      datasets = paste(expected_datasets, collapse = ";"),
      min_recall_at_k = chosen$min_recall_at_k,
      mean_recall_at_k = chosen$mean_recall_at_k,
      mean_elapsed_sec = chosen$mean_elapsed_sec,
      max_elapsed_sec = chosen$max_elapsed_sec,
      n_threads = first_value(source_rows$n_threads, NA_integer_),
      output = first_value(source_rows$output, NA_character_),
      result_backend = first_value(source_rows$result_backend, NA_character_),
      resolved_backend = first_value(source_rows$resolved_backend, NA_character_),
      distance_type = first_value(source_rows$distance_type, NA_character_),
      input_type = first_value(source_rows$input_type, NA_character_),
      input_layout = first_value(source_rows$input_layout, NA_character_),
      faiss_query_batch_size = first_value(source_rows$faiss_query_batch_size, NA_integer_),
      faiss_gpu_query_batch_size = first_value(source_rows$faiss_gpu_query_batch_size, NA_integer_),
      cuvs_ivf_batch_size = first_value(source_rows$cuvs_ivf_batch_size, NA_integer_),
      faiss_gpu_reuse_resources = first_value(source_rows$faiss_gpu_reuse_resources, NA),
      cache_fitted_indexes = first_value(source_rows$cache_fitted_indexes, NA),
      hnsw_m = first_value(source_rows$hnsw_m, NA_integer_),
      hnsw_ef_construction = first_value(source_rows$hnsw_ef_construction, NA_integer_),
      hnsw_ef_search = first_value(source_rows$hnsw_ef_search, NA_integer_),
      ivf_nlist = first_value(source_rows$ivf_nlist, NA_integer_),
      ivf_nprobe = first_value(source_rows$ivf_nprobe, NA_integer_),
      pq_m = first_value(source_rows$pq_m, NA_integer_),
      pq_nbits = first_value(source_rows$pq_nbits, NA_integer_),
      pq_dim = first_value(source_rows$pq_dim, NA_integer_),
      ivfpq_fastscan_refine_factor = first_value(source_rows$ivfpq_fastscan_refine_factor, NA_integer_),
      ivfpq_fastscan_bbs = first_value(source_rows$ivfpq_fastscan_bbs, NA_integer_),
      cagra_build_algo = first_value(source_rows$cagra_build_algo, NA_character_),
      cagra_graph_degree = first_value(source_rows$cagra_graph_degree, NA_integer_),
      cagra_intermediate_graph_degree = first_value(source_rows$cagra_intermediate_graph_degree, NA_integer_),
      cagra_search_width = first_value(source_rows$cagra_search_width, NA_integer_),
      cagra_itopk_size = first_value(source_rows$cagra_itopk_size, NA_integer_),
      nndescent_pool_size = first_value(source_rows$nndescent_pool_size, NA_integer_),
      nndescent_n_iters = first_value(source_rows$nndescent_n_iters, NA_integer_),
      nndescent_max_candidates = first_value(source_rows$nndescent_max_candidates, NA_integer_),
      nndescent_n_random_projections = first_value(source_rows$nndescent_n_random_projections, NA_integer_),
      nndescent_graph_degree = first_value(source_rows$nndescent_graph_degree, NA_integer_),
      nndescent_intermediate_graph_degree = first_value(source_rows$nndescent_intermediate_graph_degree, NA_integer_),
      nndescent_max_iterations = first_value(source_rows$nndescent_max_iterations, NA_integer_),
      nsg_r = first_value(source_rows$nsg_r, NA_integer_),
      nsg_graph_k = first_value(source_rows$nsg_graph_k, NA_integer_),
      vamana_r = first_value(source_rows$vamana_r, NA_integer_),
      vamana_search_l = first_value(source_rows$vamana_search_l, NA_integer_),
      vamana_alpha = first_value(source_rows$vamana_alpha, NA_real_),
      source_run = basename(dirname(first_value(source_rows$source_file, ""))),
      row.names = NULL,
      stringsAsFactors = FALSE
    )
  }
}

selected <- if (length(selected)) do.call(rbind, selected) else data.frame()
audit <- if (length(audit)) do.call(rbind, audit) else data.frame()
selected <- selected[order(
  selected$backend,
  selected$method,
  selected$shape_group,
  selected$k_bucket,
  selected$target_code
), , drop = FALSE]

dir.create(dirname(output_file), recursive = TRUE, showWarnings = FALSE)
write.csv(selected, output_file, row.names = FALSE, na = "")
audit_file <- sub("\\.csv$", "_candidate_audit.csv", output_file)
write.csv(audit, audit_file, row.names = FALSE, na = "")

cpp_string <- function(x, default = "") {
  x <- first_value(x, default)
  x <- gsub("\\\\", "\\\\\\\\", as.character(x))
  x <- gsub('"', '\\\\"', x, fixed = TRUE)
  paste0('"', x, '"')
}

cpp_int <- function(x, default = 0L) {
  value <- suppressWarnings(as.integer(first_value(x, default)))
  if (is.na(value)) value <- as.integer(default)
  as.character(value)
}

cpp_double <- function(x, default = 0) {
  value <- suppressWarnings(as.numeric(first_value(x, default)))
  if (!is.finite(value)) value <- default
  format(value, scientific = FALSE, trim = TRUE, digits = 12)
}

cpp_bool <- function(x, default = FALSE) {
  value <- first_value(x, default)
  value <- if (is.logical(value)) value else tolower(as.character(value)) %in% c("true", "1", "yes")
  if (isTRUE(value)) "true" else "false"
}

row_basis <- function(row) {
  paste(
    row$recommendation_basis,
    "jmlr_mloss_inner_product",
    row$source_run,
    paste0("coverage_", row$successful_dataset_count, "of", row$expected_dataset_count),
    sep = "_"
  )
}

emit_array <- function(type, name, rows, formatter) {
  if (!nrow(rows)) return(character())
  c(
    paste0("static const ", type, " ", name, "[] = {"),
    vapply(seq_len(nrow(rows)), function(i) {
      paste0("  {", formatter(rows[i, , drop = FALSE]), "},")
    }, character(1)),
    "};",
    ""
  )
}

subset_method <- function(backend, method) {
  selected[selected$backend == backend & selected$method == method, , drop = FALSE]
}

format_exact_cpu <- function(row) paste(
  cpp_string(row$backend), cpp_string(row$shape_group), cpp_int(row$k_bucket),
  cpp_int(row$target_code), cpp_int(row$n_threads, 12L),
  cpp_int(row$faiss_query_batch_size, 16384L), cpp_bool(row$cache_fitted_indexes),
  cpp_string(row$output, "double"), cpp_string(row$result_backend, "faiss_flat_ip"),
  cpp_string(row$resolved_backend, "faiss_flat_ip"), cpp_string(row$distance_type, "double"),
  cpp_string(row$input_type, "float32"), cpp_string(row$input_layout),
  cpp_string(row_basis(row)), sep = ", "
)
format_exact_cuda <- function(row) paste(
  cpp_string(row$backend), cpp_string(row$shape_group), cpp_int(row$k_bucket),
  cpp_int(row$target_code), cpp_int(row$n_threads, 2L),
  cpp_int(row$faiss_gpu_query_batch_size, 8192L), cpp_bool(row$faiss_gpu_reuse_resources, TRUE),
  cpp_string(row$output, "double"), cpp_string(row$result_backend, "faiss_gpu_flat_ip"),
  cpp_string(row$resolved_backend, "faiss_gpu_flat_ip"), cpp_string(row$distance_type, "double"),
  cpp_string(row$input_type, "float32"), cpp_string(row$input_layout),
  cpp_string(row_basis(row)), sep = ", "
)
format_ivf <- function(row) paste(
  cpp_string(row$backend), cpp_string(row$shape_group), cpp_int(row$k_bucket),
  cpp_int(row$target_code), cpp_int(row$ivf_nlist, 1L), cpp_int(row$ivf_nprobe, 1L),
  cpp_string(row_basis(row)), sep = ", "
)
format_ivfpq <- function(row) paste(
  cpp_string(row$backend), cpp_string(row$shape_group), cpp_int(row$k_bucket),
  cpp_int(row$target_code), cpp_int(row$ivf_nlist, 1L), cpp_int(row$ivf_nprobe, 1L),
  cpp_int(row$pq_m, 1L), cpp_int(row$pq_nbits, 8L), cpp_string(row_basis(row)),
  sep = ", "
)
format_nndescent_cpu <- function(row) paste(
  cpp_string(row$shape_group), cpp_int(row$k_bucket), cpp_int(row$target_code),
  cpp_int(row$nndescent_pool_size, 64L), cpp_int(row$nndescent_n_iters, 10L),
  cpp_int(row$nndescent_max_candidates, 64L),
  cpp_int(row$nndescent_n_random_projections, 8L), cpp_string(row_basis(row)),
  sep = ", "
)
format_nsg <- function(row) paste(
  cpp_string(row$backend), cpp_string(row$shape_group), cpp_int(row$k_bucket),
  cpp_int(row$target_code), cpp_int(row$nsg_r, 32L), cpp_int(row$nsg_graph_k, 64L),
  cpp_string(row_basis(row)), sep = ", "
)
format_vamana <- function(row) paste(
  cpp_string(row$backend), cpp_string(row$shape_group), cpp_int(row$k_bucket),
  cpp_int(row$target_code), cpp_int(row$vamana_r, 32L), cpp_int(row$vamana_search_l, 64L),
  cpp_double(row$vamana_alpha, 1.2), cpp_string(row_basis(row)), sep = ", "
)
format_hnsw_cpu <- function(row) paste(
  cpp_string(row$shape_group), cpp_int(row$k_bucket), cpp_int(row$target_code),
  cpp_int(row$hnsw_m, 32L), cpp_int(row$hnsw_ef_construction, 200L),
  cpp_int(row$hnsw_ef_search, 100L), cpp_string(row_basis(row)), sep = ", "
)

header <- c(
  "#pragma once",
  "",
  "// Generated by benchmark_scripts/import_jmlr_mloss_inner_product_tuning.R.",
  "// Only candidates with complete dataset coverage inside a shape class are included.",
  "",
  "struct JmlrHnswCpuSpec {",
  "  const char* shape_group;",
  "  int k_bucket;",
  "  int target_code;",
  "  int m;",
  "  int ef_construction;",
  "  int ef_search;",
  "  const char* basis;",
  "};",
  "",
  emit_array("HpcExactSpec", "jmlr_cpu_exact_inner_product_specs", subset_method("cpu", "exact"), format_exact_cpu),
  emit_array("HpcExactSpec", "jmlr_cpu_flat_inner_product_specs", subset_method("cpu", "flat"), format_exact_cpu),
  emit_array("HpcExactSpec", "jmlr_cpu_bruteforce_inner_product_specs", subset_method("cpu", "bruteforce"), format_exact_cpu),
  emit_array("HpcCudaExactSpec", "jmlr_cuda_exact_inner_product_specs", subset_method("cuda", "exact"), format_exact_cuda),
  emit_array("HpcCudaFlatSpec", "jmlr_cuda_flat_inner_product_specs", subset_method("cuda", "flat"), format_exact_cuda),
  emit_array("HpcIvfSpec", "jmlr_ivf_inner_product_specs", selected[selected$method == "ivf", , drop = FALSE], format_ivf),
  emit_array("HpcIvfpqSpec", "jmlr_ivfpq_inner_product_specs", selected[selected$method == "ivfpq", , drop = FALSE], format_ivfpq),
  emit_array("HpcNndescentSpec", "jmlr_cpu_nndescent_inner_product_specs", subset_method("cpu", "nndescent"), format_nndescent_cpu),
  emit_array("HpcNsgSpec", "jmlr_nsg_inner_product_specs", selected[selected$method == "nsg", , drop = FALSE], format_nsg),
  emit_array("HpcVamanaSpec", "jmlr_vamana_inner_product_specs", selected[selected$method == "vamana", , drop = FALSE], format_vamana),
  emit_array("JmlrHnswCpuSpec", "jmlr_cpu_hnsw_inner_product_specs", subset_method("cpu", "hnsw"), format_hnsw_cpu),
  "template <typename Spec, std::size_t N>",
  "inline const Spec* jmlr_match_spec(const Spec (&specs)[N], const std::string& backend, const std::string& shape_group, int k_bucket, int target_code) {",
  "  for (const auto& spec : specs) if (backend == spec.backend && shape_group == spec.shape_group && spec.k_bucket == k_bucket && spec.target_code == target_code) return &spec;",
  "  return nullptr;",
  "}",
  "",
  "template <typename Spec, std::size_t N>",
  "inline const Spec* jmlr_match_shape_spec(const Spec (&specs)[N], const std::string& shape_group, int k_bucket, int target_code) {",
  "  for (const auto& spec : specs) if (shape_group == spec.shape_group && spec.k_bucket == k_bucket && spec.target_code == target_code) return &spec;",
  "  return nullptr;",
  "}"
)
dir.create(dirname(header_file), recursive = TRUE, showWarnings = FALSE)
writeLines(header, header_file, useBytes = TRUE)

cat("Imported", nrow(selected), "complete-shape tuning rows from", length(result_files), "result files.\n")
cat("Defaults:", normalizePath(output_file, winslash = "/", mustWork = TRUE), "\n")
cat("Audit:", normalizePath(audit_file, winslash = "/", mustWork = TRUE), "\n")
cat("Header:", normalizePath(header_file, winslash = "/", mustWork = TRUE), "\n")
