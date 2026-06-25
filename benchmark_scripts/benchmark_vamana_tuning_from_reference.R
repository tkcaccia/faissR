#!/usr/bin/env Rscript
args0 <- commandArgs(FALSE)
file_arg <- grep("^--file=", args0, value = TRUE)
this_file <- if (length(file_arg)) sub("^--file=", "", file_arg[[1L]]) else file.path("benchmark_scripts", "benchmark_vamana_tuning_from_reference.R")
Sys.setenv(FAISSR_TUNING_METHOD = "vamana")
source(file.path(dirname(normalizePath(this_file, mustWork = FALSE)), "benchmark_method_tuning_from_reference.R"))
