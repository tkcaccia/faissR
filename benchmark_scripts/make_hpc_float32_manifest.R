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

split_arg <- function(value, default) {
  value <- value %||% default
  trimws(strsplit(value, ",", fixed = TRUE)[[1L]])
}

default_dataset_specs <- function() {
  data.frame(
    dataset = c(
      "COIL20",
      "USPS",
      "FashionMNIST",
      "FlowRepository_FR-FCM-ZYRM_files",
      "flow18",
      "MNIST",
      "imagenet",
      "MetRef",
      "mass41",
      "TabulaMuris"
    ),
    folder = c(
      "COIL20",
      "USPS",
      "FashionMNIST",
      "FlowRepository_FR-FCM-ZYRM_files",
      "flow18",
      "MNIST",
      "imagenet",
      "MetRef",
      "mass41",
      "TabulaMuris"
    ),
    preferred_file = c(
      "COIL20_float32.RData",
      "USPS_float32.RData",
      "FashionMNIST_float32.RData",
      "van_unen_FR-FCM-ZYRM_float32.RData",
      "flow18_float32.RData",
      "MNIST_float32.RData",
      "imagenet_float32.RData",
      "MetRef_float32.RData",
      "mass41_float32.RData",
      "TabulaMuris_float32.RData"
    ),
    stringsAsFactors = FALSE
  )
}

find_dataset_file <- function(data_root, folder, preferred_file) {
  preferred <- file.path(data_root, folder, preferred_file)
  if (file.exists(preferred)) return(normalizePath(preferred, mustWork = TRUE))
  folder_path <- file.path(data_root, folder)
  if (!dir.exists(folder_path)) return(preferred)
  candidates <- list.files(
    folder_path,
    pattern = "_float32[.]RData$",
    full.names = TRUE,
    ignore.case = FALSE
  )
  if (length(candidates)) return(normalizePath(candidates[[1L]], mustWork = TRUE))
  preferred
}

find_data_object <- function(env) {
  if (exists("dataset", envir = env, inherits = FALSE)) {
    value <- get("dataset", envir = env, inherits = FALSE)
    if (is.list(value) && !is.null(value$data)) {
      return(list(data = value$data, labels = value$labels %||% NULL, object_name = "dataset"))
    }
  }
  for (name in ls(env)) {
    value <- get(name, envir = env, inherits = FALSE)
    if (is.list(value) && !is.null(value$data)) {
      return(list(data = value$data, labels = value$labels %||% NULL, object_name = name))
    }
  }
  for (name in ls(env)) {
    value <- get(name, envir = env, inherits = FALSE)
    if (inherits(value, "float32") || is.matrix(value)) {
      return(list(data = value, labels = NULL, object_name = name))
    }
  }
  NULL
}

float_dims <- function(x) {
  dims <- dim(x)
  if (is.null(dims) && methods::is(x, "float32")) {
    dims <- dim(methods::slot(x, "Data"))
  }
  dims
}

inspect_dataset_file <- function(path) {
  if (!file.exists(path)) {
    return(list(status = "missing", error = "float32 RData file not found"))
  }
  env <- new.env(parent = emptyenv())
  loaded <- tryCatch(load(path, envir = env), error = function(e) e)
  if (inherits(loaded, "error")) {
    return(list(status = "load_failed", error = conditionMessage(loaded)))
  }
  found <- find_data_object(env)
  if (is.null(found)) {
    return(list(status = "invalid", error = "no matrix or list$data object found"))
  }
  dims <- float_dims(found$data)
  if (is.null(dims) || length(dims) != 2L) {
    return(list(status = "invalid", error = "data object is not two-dimensional"))
  }
  is_float <- inherits(found$data, "float32")
  list(
    status = if (is_float) "success" else "not_float32",
    error = if (is_float) NA_character_ else "data object is not float::fl()/float32",
    n = as.integer(dims[[1L]]),
    p = as.integer(dims[[2L]]),
    labels = !is.null(found$labels),
    data_class = paste(class(found$data), collapse = "|"),
    object_name = found$object_name,
    file_size_gb = file.info(path)$size / 1024^3,
    dataset_md5 = unname(tools::md5sum(path)[[1L]])
  )
}

main <- function() {
  args <- parse_args()
  data_root <- normalizePath(args$data_root %||% "/scratch/firenze/NN/Data", mustWork = FALSE)
  out <- normalizePath(
    args$out %||% file.path(data_root, "float32_dataset_manifest.csv"),
    mustWork = FALSE
  )
  specs <- default_dataset_specs()
  datasets <- split_arg(args$datasets, paste(specs$dataset, collapse = ","))
  unknown <- setdiff(datasets, specs$dataset)
  if (length(unknown)) {
    stop("Unknown dataset(s): ", paste(unknown, collapse = ", "), call. = FALSE)
  }
  specs <- specs[match(datasets, specs$dataset), , drop = FALSE]

  rows <- vector("list", nrow(specs))
  for (i in seq_len(nrow(specs))) {
    spec <- specs[i, , drop = FALSE]
    path <- find_dataset_file(data_root, spec$folder, spec$preferred_file)
    info <- inspect_dataset_file(path)
    rows[[i]] <- data.frame(
      dataset = spec$dataset,
      folder = spec$folder,
      preferred_file = spec$preferred_file,
      output = path,
      path = path,
      n = as.integer(info$n %||% NA_integer_),
      p = as.integer(info$p %||% NA_integer_),
      labels = isTRUE(info$labels),
      data_class = info$data_class %||% NA_character_,
      object_name = info$object_name %||% NA_character_,
      file_size_gb = as.numeric(info$file_size_gb %||% NA_real_),
      dataset_md5 = info$dataset_md5 %||% NA_character_,
      status = info$status %||% "failed",
      error = info$error %||% NA_character_,
      stringsAsFactors = FALSE
    )
  }
  manifest <- do.call(rbind, rows)
  dir.create(dirname(out), recursive = TRUE, showWarnings = FALSE)
  write.csv(manifest, out, row.names = FALSE)
  cat("Wrote float32 manifest: ", out, "\n", sep = "")
  print(manifest[, c("dataset", "n", "p", "labels", "status", "output")], row.names = FALSE)
  if (any(manifest$status != "success")) {
    warning(
      "Some datasets are not ready as float32 inputs: ",
      paste(manifest$dataset[manifest$status != "success"], collapse = ", "),
      call. = FALSE
    )
  }
}

main()
