#!/usr/bin/env Rscript

`%||%` <- function(x, y) {
  if (is.null(x) || length(x) == 0L || is.na(x[[1L]]) || !nzchar(x[[1L]])) y else x
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

script_path <- function() {
  args <- commandArgs(FALSE)
  file_arg <- grep("^--file=", args, value = TRUE)
  if (length(file_arg)) return(normalizePath(sub("^--file=", "", file_arg[[1L]]), mustWork = TRUE))
  normalizePath(file.path("benchmark_scripts", "generate_metric_specific_launchers.R"), mustWork = FALSE)
}

slug_job <- function(metric) {
  switch(
    metric,
    euclidean = "EUCL",
    cosine = "COS",
    correlation = "COR",
    inner_product = "IP",
    toupper(metric)
  )
}

rewrite_sbatch_line <- function(line, metric) {
  metric_file <- metric
  metric_job <- slug_job(metric)
  if (grepl("^#SBATCH[[:space:]]+--job-name=", line)) {
    value <- sub("^#SBATCH[[:space:]]+--job-name=", "", line)
    value <- gsub('^"|"$', "", value)
    return(sprintf('#SBATCH --job-name="%s_%s"', value, metric_job))
  }
  if (grepl("^#SBATCH[[:space:]]+--output=", line)) {
    path <- sub("^#SBATCH[[:space:]]+--output=", "", line)
    path <- sub("_%j\\.out$", paste0("_", metric_file, "_%j.out"), path)
    return(paste("#SBATCH --output=", path, sep = ""))
  }
  if (grepl("^#SBATCH[[:space:]]+--error=", line)) {
    path <- sub("^#SBATCH[[:space:]]+--error=", "", line)
    path <- sub("_%j\\.err$", paste0("_", metric_file, "_%j.err"), path)
    return(paste("#SBATCH --error=", path, sep = ""))
  }
  line
}

base_header <- function(path, metric) {
  lines <- readLines(path, warn = FALSE)
  shebang <- lines[[1L]]
  sbatch <- grep("^#SBATCH", lines, value = TRUE)
  c(shebang, "", vapply(sbatch, rewrite_sbatch_line, character(1L), metric = metric))
}

wrapper_body <- function(base_file, metric) {
  base_path <- file.path(script_dir_global(), base_file)
  prefix <- default_out_prefix(base_path)
  extra_exports <- character()
  if (identical(base_file, "run_hpc_nndescent_tuning_cuda.sh")) {
    extra_exports <- 'export ALLOW_CUDA_NNDESCENT_TUNING="${ALLOW_CUDA_NNDESCENT_TUNING:-TRUE}"'
  }
  c(
    "",
    "set -euo pipefail",
    "",
    sprintf("# Generated metric-specific wrapper for %s.", base_file),
    "# Submit this file directly with sbatch to run exactly one metric.",
    sprintf('export METRICS="%s"', metric),
    sprintf('export FAISSR_SINGLE_METRIC="%s"', metric),
    extra_exports,
    'export BASE_DIR="${BASE_DIR:-/scratch/firenze/NN}"',
    'if [[ -z "${OUT_DIR:-}" ]]; then',
    sprintf('  export OUT_DIR="${BASE_DIR}/%s_%s_$(date +%%Y%%m%%d_%%H%%M%%S)"', prefix, metric),
    "fi",
    "",
    'WRAPPER_SCRIPT="${BASH_SOURCE[0]:-$0}"',
    'if command -v readlink >/dev/null 2>&1; then',
    '  WRAPPER_SCRIPT="$(readlink -f "${WRAPPER_SCRIPT}" 2>/dev/null || printf \'%s\\n\' "${WRAPPER_SCRIPT}")"',
    "fi",
    'WRAPPER_SCRIPT_DIR="$(cd "$(dirname "${WRAPPER_SCRIPT}")" && pwd)"',
    'if [[ -z "${SCRIPT_DIR:-}" ]]; then',
    sprintf('  if [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "${SLURM_SUBMIT_DIR}/benchmark_scripts/%s" ]]; then', base_file),
    '    export SCRIPT_DIR="${SLURM_SUBMIT_DIR}/benchmark_scripts"',
    sprintf('  elif [[ -f "${BASE_DIR}/benchmark_scripts/%s" ]]; then', base_file),
    '    export SCRIPT_DIR="${BASE_DIR}/benchmark_scripts"',
    sprintf('  elif [[ -f "${WRAPPER_SCRIPT_DIR}/%s" ]]; then', base_file),
    '    export SCRIPT_DIR="${WRAPPER_SCRIPT_DIR}"',
    "  else",
    sprintf('    echo "Cannot locate base launcher %s. Set SCRIPT_DIR to the faissR benchmark_scripts folder." >&2', base_file),
    "    exit 1",
    "  fi",
    "fi",
    sprintf('exec bash "${SCRIPT_DIR}/%s"', base_file),
    ""
  )
}

script_dir_global <- local({
  value <- NULL
  function(new_value = NULL) {
    if (!is.null(new_value)) value <<- new_value
    value
  }
})

default_out_prefix <- function(base_path) {
  lines <- readLines(base_path, warn = FALSE)
  line <- grep('export OUT_DIR="\\$\\{OUT_DIR:-\\$\\{BASE_DIR\\}/', lines, value = TRUE)
  if (!length(line)) return("faissR_METRIC_TUNING")
  prefix <- sub('^.*\\$\\{BASE_DIR\\}/', "", line[[1L]])
  prefix <- sub('_\\$\\(date \\+%Y%m%d_%H%M%S\\).*$', "", prefix)
  if (!nzchar(prefix)) "faissR_METRIC_TUNING" else prefix
}

write_wrapper <- function(script_dir, base_file, metric) {
  base_path <- file.path(script_dir, base_file)
  out_file <- sub("\\.sh$", paste0("_", metric, ".sh"), base_file)
  out_path <- file.path(script_dir, out_file)
  content <- c(base_header(base_path, metric), wrapper_body(base_file, metric))
  writeLines(content, out_path, useBytes = TRUE)
  Sys.chmod(out_path, mode = "0755")
  out_path
}

main <- function() {
  args <- parse_args()
  default_dir <- dirname(script_path())
  script_dir <- normalizePath(args$script_dir %||% default_dir, mustWork = TRUE)
  script_dir_global(script_dir)
  metrics <- c("euclidean", "cosine", "correlation", "inner_product")

  launchers <- list.files(
    script_dir,
    pattern = "^run_hpc_.*_tuning_(cpu12|cuda)\\.sh$",
    full.names = FALSE
  )
  launchers <- sort(launchers)

  reference_launcher <- "run_hpc_precompute_exact_references_cpu12.sh"
  if (file.exists(file.path(script_dir, reference_launcher))) {
    launchers <- c(reference_launcher, launchers)
  }

  written <- character()
  for (base_file in launchers) {
    launcher_metrics <- if (identical(base_file, "run_hpc_nndescent_tuning_cuda.sh")) {
      c("euclidean", "cosine", "correlation")
    } else {
      metrics
    }
    for (metric in launcher_metrics) {
      written <- c(written, write_wrapper(script_dir, base_file, metric))
    }
  }

  message("Wrote ", length(written), " metric-specific launchers in ", script_dir)
  invisible(written)
}

main()
