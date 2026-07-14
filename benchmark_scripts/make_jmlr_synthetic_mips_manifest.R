#!/usr/bin/env Rscript

args <- commandArgs(trailingOnly = TRUE)
`%||%` <- function(x, y) if (is.null(x) || !length(x) || !nzchar(x[[1L]])) y else x
parse_args <- function(x) {
  out <- list()
  for (arg in x) {
    if (!startsWith(arg, "--")) next
    bits <- strsplit(sub("^--", "", arg), "=", fixed = TRUE)[[1L]]
    out[[bits[[1L]]]] <- paste(bits[-1L], collapse = "=")
  }
  out
}

opts <- parse_args(args)
out_dir <- normalizePath(opts$out_dir %||% file.path(getwd(), "Data", "JMLR_synthetic_MIPS"), mustWork = FALSE)
manifest_path <- normalizePath(opts$manifest %||% file.path(out_dir, "jmlr_synthetic_mips_manifest.csv"), mustWork = FALSE)
seed <- as.integer(opts$seed %||% 20260713L)

if (!requireNamespace("float", quietly = TRUE)) {
  stop("The publication synthetic-data generator requires the optional `float` package.", call. = FALSE)
}
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

specs <- data.frame(
  n = c(10000L, 10000L, 20000L, 70000L, 70000L, 200000L),
  p = c(2L, 3L, 32L, 128L, 512L, 64L),
  suite = c("spatial", "spatial", rep("mips", 4L)),
  stringsAsFactors = FALSE
)
norm_models <- c("unit", "lognormal", "pareto")
rows <- list()

for (i in seq_len(nrow(specs))) {
  models <- if (specs$suite[[i]] == "spatial") "unit" else norm_models
  for (norm_model in models) {
    set.seed(seed + 1000L * i + match(norm_model, norm_models, nomatch = 0L))
    n <- specs$n[[i]]
    p <- specs$p[[i]]
    x <- matrix(stats::rnorm(n * p), nrow = n, ncol = p)
    row_norm <- sqrt(rowSums(x * x))
    x <- x / pmax(row_norm, sqrt(.Machine$double.eps))
    scale <- switch(
      norm_model,
      unit = rep(1, n),
      lognormal = stats::rlnorm(n, meanlog = 0, sdlog = 1),
      pareto = (1 - stats::runif(n, min = 0, max = 1 - 1e-7))^(-1 / 2)
    )
    x <- x * scale
    dataset_name <- sprintf("synthetic_%s_n%d_p%d_%s", specs$suite[[i]], n, p, norm_model)
    path <- file.path(out_dir, paste0(dataset_name, "_float32.RData"))
    dataset <- list(data = float::fl(x), labels = factor(rep("synthetic", n)))
    save(dataset, file = path, compress = "xz")
    rows[[length(rows) + 1L]] <- data.frame(
      dataset = dataset_name,
      path = normalizePath(path, mustWork = TRUE),
      n = n,
      p = p,
      input_type = "float32",
      suite = specs$suite[[i]],
      norm_model = norm_model,
      norm_mean = mean(scale),
      norm_sd = stats::sd(scale),
      norm_cv = stats::sd(scale) / mean(scale),
      stringsAsFactors = FALSE
    )
    rm(x, dataset, row_norm, scale)
    gc()
  }
}

manifest <- do.call(rbind, rows)
utils::write.csv(manifest, manifest_path, row.names = FALSE)
cat(manifest_path, "\n", sep = "")
