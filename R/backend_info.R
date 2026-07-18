#' Summarize native neighbour-search backend availability
#'
#' `backend_info()` reports which `faissR` nearest-neighbour backends can
#' currently run. It never silently falls back from an explicit GPU request to
#' CPU; this table is informational only.
#'
#' @return A data frame with one row per compiled/runtime backend family and
#'   columns describing availability, public call hints, public backend names,
#'   supported public method/metric summaries, non-public implementation route
#'   labels, device/runtime hints, and a short note. Use
#'   \code{\link{nn_capabilities}()} for the full method/backend/metric matrix.
#' @examples
#' info <- backend_info()
#' info[, c("backend", "available", "public_backends")]
#' @export
backend_info <- function() {
  cuda_knn <- backend_flag(cuda_available)
  faiss_knn <- backend_flag(faiss_available)
  cuvs_knn <- backend_flag(cuvs_available)
  metal_knn <- backend_flag(metal_grid_available_cpp)
  cuda <- cuda_summary()
  faiss <- faiss_summary()
  cuvs <- cuvs_summary()

  data.frame(
    backend = c("cpu", "faiss", "faiss_gpu_cuvs", "cuvs", "cuda", "metal_grid"),
    available = c(TRUE, faiss_knn, faiss$gpu && cuda_knn, cuvs_knn, cuda_knn, metal_knn),
    knn_available = c(TRUE, faiss_knn, faiss$gpu && cuda_knn, cuvs_knn, cuda_knn, metal_knn),
    public_call = c(
      "backend = \"cpu\"",
      "backend = \"cpu\" or \"cuda\", method = \"flat\"/\"ivf\"/\"ivfpq\"/\"hnsw\"/\"ivfpq_fastscan\"/\"cagra\" as supported",
      "backend = \"cuda\", method = \"ivf\"/\"ivfpq\"/\"cagra\"",
      "backend = \"cuda\", method = \"bruteforce\"/\"hnsw\"/\"nndescent\"/\"ivfpq_fastscan\"/\"cagra\"",
      "backend = \"cuda\"",
      "backend = \"metal\", method = \"grid\""
    ),
    public_backends = c(
      "cpu",
      "cpu, cuda",
      "cuda",
      "cuda",
      "cuda",
      "metal"
    ),
    supported_methods = c(
      "auto, exact, flat, bruteforce, grid, hnsw, ivf, ivfpq, ivfpq_fastscan, vamana, nsg, nndescent",
      "flat, ivf, ivfpq, hnsw, ivfpq_fastscan, nsg; GPU flat/ivf/ivfpq/cagra when FAISS GPU is available",
      "ivf, ivfpq, cagra",
      "bruteforce, hnsw, nndescent, ivfpq_fastscan, cagra",
      "grid, flat, bruteforce, hnsw, ivf, ivfpq, ivfpq_fastscan, vamana, nsg, nndescent, cagra where compiled",
      "grid"
    ),
    supported_metrics = c(
      "euclidean, cosine, correlation, inner_product; method-specific exclusions in nn_capabilities()",
      "euclidean, cosine, correlation, inner_product for Flat/IVF/IVFPQ/HNSW and CPU IVFPQ FastScan where FastScan is available; public NSG uses the native CPU route for all metrics, with deterministic FAISS HNSW seeding on large high-dimensional CPU inputs, while explicit FAISS NSG is Euclidean-only",
      "euclidean, cosine, correlation, inner_product for IVF/IVFPQ and CAGRA; CAGRA inner_product uses a maximum-inner-product-to-L2 transform",
      "euclidean, cosine, correlation, inner_product for direct brute force, direct IVF/PQ, HNSW from CAGRA, and direct CAGRA using metric transforms where needed; euclidean plus normalized cosine/correlation for direct cuVS NN-descent; raw inner product is unsupported for CUDA NN-descent because its symmetric one-dataset graph API cannot use the asymmetric MIPS transform",
      "euclidean, cosine, correlation, inner_product where the selected CUDA method supports the metric",
      "euclidean, cosine, correlation for exact 2D/3D self-KNN"
    ),
    resolved_route = c(
      "implementation label: cpu",
      "implementation labels include faiss_flat_l2, faiss_ivf, faiss_hnsw, faiss_ivfpq_fastscan, and faiss_gpu_*",
      "implementation labels include faiss_gpu_ivf_flat, faiss_gpu_ivfpq, and faiss_gpu_cagra",
      "implementation labels include cuda_cuvs_bruteforce, cuda_cuvs_hnsw, cuda_cuvs_nndescent, cuda_cuvs_ivfpq_fastscan, and cuda_cuvs_cagra",
      "implementation labels include cuda_grid, cuda_vamana, and cuda_nsg; exact CUDA may report cuda",
      "implementation labels metal_grid2d and metal_grid3d"
    ),
    device = c(
      cpu_summary(),
      faiss$device,
      cuda$device,
      cuvs$device,
      cuda$device,
      if (metal_knn) "Apple Metal GPU" else "Metal unavailable"
    ),
    runtime = c(
      R.version$platform,
      faiss$runtime,
      if (isTRUE(faiss$gpu)) {
        combine_nonempty(faiss$runtime, "FAISS GPU IVF and CAGRA indexes backed by NVIDIA cuVS when FAISS is built with cuVS")
      } else {
        faiss$runtime
      },
      cuvs$runtime,
      cuda$runtime,
      if (metal_knn) "Apple Metal Shading Language exact float32 2D/3D grid KNN" else NA_character_
    ),
    note = c(
      "Native CPU path is always available.",
      if (faiss_knn) {
        "Real FAISS C++ KNN is available behind public CPU/CUDA method requests; IVFPQ FastScan CPU search requires linked FAISS FastScan support."
      } else {
        "Real FAISS C++ KNN is unavailable; FAISS-backed method requests will fail."
      },
      if (faiss$gpu && cuda_knn) {
        "FAISS GPU IVF-Flat, IVF-PQ, and CAGRA use FAISS GPU indexes with NVIDIA cuVS integration when the linked FAISS library provides it; result backends report GpuIndexIVFFlat_cuVS, GpuIndexIVFPQ_cuVS, or GpuIndexCagra_cuVS."
      } else {
        "FAISS GPU cuVS-integrated IVF/CAGRA requests are unavailable; public CUDA IVF/CAGRA method requests will fail unless another validated CUDA route is available."
      },
      if (cuvs_knn) {
        "RAPIDS cuVS CUDA KNN is available behind public CUDA method requests."
      } else {
        "RAPIDS cuVS CUDA KNN is unavailable; cuVS-backed public CUDA method requests will fail."
      },
      if (cuda_knn) {
        "Native CUDA KNN path is available for explicit CUDA requests."
      } else {
        "Native CUDA KNN path is unavailable; explicit CUDA requests will fail."
      },
      if (metal_knn) {
        "Native exact Metal grid KNN is available for two- and three-dimensional self-search; unsupported shapes and methods fail without CPU fallback."
      } else {
        "Native Metal grid KNN is unavailable; explicit Metal nearest-neighbour requests will fail."
      }
    ),
    stringsAsFactors = FALSE
  )
}

backend_flag <- function(fn) {
  tryCatch(isTRUE(fn()), error = function(e) FALSE)
}

cpu_summary <- function() {
  cores <- suppressWarnings(parallel::detectCores(logical = TRUE))
  if (length(cores) != 1L || is.na(cores) || !is.finite(cores)) {
    "CPU"
  } else {
    paste0("CPU (", cores, " logical cores)")
  }
}

nvidia_smi_summary <- function() {
  smi <- Sys.which("nvidia-smi")
  if (!nzchar(smi)) {
    return(list(device = NA_character_, runtime = NA_character_))
  }
  out <- tryCatch(
    system2(
      smi,
      c(
        "--query-gpu=name,driver_version,memory.total",
        "--format=csv,noheader,nounits"
      ),
      stdout = TRUE,
      stderr = FALSE
    ),
    error = function(e) character()
  )
  if (length(out) < 1L || !nzchar(out[1L])) {
    return(list(device = NA_character_, runtime = NA_character_))
  }
  parts <- trimws(strsplit(out[1L], ",", fixed = TRUE)[[1L]])
  device <- if (length(parts) >= 1L) parts[1L] else NA_character_
  driver <- if (length(parts) >= 2L) parts[2L] else NA_character_
  memory <- if (length(parts) >= 3L) parts[3L] else NA_character_
  runtime <- paste(
    c(
      if (!is.na(driver) && nzchar(driver)) paste0("driver ", driver) else NULL,
      if (!is.na(memory) && nzchar(memory)) paste0(memory, " MiB") else NULL
    ),
    collapse = ", "
  )
  if (!nzchar(runtime)) runtime <- NA_character_
  list(device = device, runtime = runtime)
}

cuda_summary <- function() {
  native <- cuda_native_summary()
  smi <- nvidia_smi_summary()

  device <- first_nonempty(native$device, smi$device)
  runtime <- combine_nonempty(native$runtime, smi$runtime)
  list(device = device, runtime = runtime)
}

cuda_native_summary <- function() {
  text <- tryCatch(
    cuda_device_info_json_cpp(),
    error = function(e) NA_character_
  )
  if (length(text) != 1L || is.na(text) || !nzchar(text)) {
    return(list(device = NA_character_, runtime = NA_character_))
  }

  available <- json_get_bool(text, "available")
  if (isTRUE(available)) {
    device <- json_get_string(text, "name")
    compute <- json_get_string(text, "compute_capability")
    total_memory <- json_get_number(text, "total_memory")
    free_memory <- json_get_number(text, "free_memory")
    memory <- cuda_memory_summary(free_memory, total_memory)
    runtime <- combine_nonempty(
      if (!is.na(compute)) paste0("compute capability ", compute) else NA_character_,
      memory
    )
    return(list(device = device, runtime = runtime))
  }

  reason <- json_get_string(text, "reason")
  list(device = NA_character_, runtime = reason)
}

faiss_summary <- function() {
  text <- tryCatch(
    faiss_info_json_cpp(),
    error = function(e) NA_character_
  )
  available <- json_get_bool(text, "available")
  gpu <- json_get_bool(text, "gpu")
  gpu_cagra <- json_get_bool(text, "gpu_cagra")
  fastscan <- json_get_bool(text, "fastscan")
  reason <- json_get_string(text, "reason")
  runtime <- if (isTRUE(available)) {
    combine_nonempty(
      "FAISS C++ library",
      if (isTRUE(gpu)) "FAISS GPU headers" else "CPU-only FAISS headers",
      if (isTRUE(gpu_cagra)) "GpuIndexCagra available" else NA_character_,
      if (isTRUE(fastscan)) "FastScan available" else NA_character_
    )
  } else if (!is.na(reason)) {
    reason
  } else {
    NA_character_
  }
  list(
    device = if (isTRUE(gpu)) "CPU/GPU depending on requested FAISS index" else "CPU",
    runtime = runtime,
    gpu = isTRUE(gpu),
    gpu_cagra = isTRUE(gpu_cagra),
    fastscan = isTRUE(fastscan)
  )
}

#' Check whether FAISS GPU support is available
#'
#' @return `TRUE` when faissR was compiled and linked against a FAISS build
#'   that reports GPU support.
#' @examples
#' faiss_gpu_available()
#' @export
faiss_gpu_available <- function() {
  text <- tryCatch(
    faiss_info_json_cpp(),
    error = function(e) NA_character_
  )
  isTRUE(json_get_bool(text, "available")) && isTRUE(json_get_bool(text, "gpu"))
}

cuvs_summary <- function() {
  text <- tryCatch(
    cuvs_info_json_cpp(),
    error = function(e) NA_character_
  )
  available <- json_get_bool(text, "available")
  reason <- json_get_string(text, "reason")
  device <- json_get_string(text, "device")
  compute <- json_get_string(text, "compute_capability")
  total_memory <- json_get_number(text, "total_memory")
  runtime <- if (isTRUE(available)) {
    combine_nonempty(
      "RAPIDS cuVS C API",
      if (!is.na(compute)) paste0("compute capability ", compute) else NA_character_,
      cuda_memory_summary(NA_real_, total_memory)
    )
  } else if (!is.na(reason)) {
    reason
  } else {
    NA_character_
  }
  list(device = if (!is.na(device)) device else "CUDA GPU", runtime = runtime)
}

json_get_bool <- function(text, key) {
  value <- json_capture(text, key, "(true|false)")
  if (is.na(value)) return(NA)
  identical(tolower(value), "true")
}

json_get_number <- function(text, key) {
  value <- json_capture(text, key, "([0-9]+(?:\\.[0-9]+)?)")
  if (is.na(value)) return(NA_real_)
  as.numeric(value)
}

json_get_string <- function(text, key) {
  value <- json_capture(text, key, "\"((?:\\\\.|[^\"\\\\])*)\"")
  if (is.na(value)) return(NA_character_)
  json_unescape(value)
}

json_capture <- function(text, key, value_pattern) {
  key_pattern <- paste0("\"", gsub("([\\W])", "\\\\\\1", key), "\"")
  pattern <- paste0(key_pattern, "\\s*:\\s*", value_pattern)
  match <- regexec(pattern, text, perl = TRUE)
  parts <- regmatches(text, match)[[1L]]
  if (length(parts) < 2L) NA_character_ else parts[2L]
}

json_unescape <- function(value) {
  value <- gsub("\\\\n", "\n", value)
  value <- gsub("\\\\r", "\r", value)
  value <- gsub("\\\\t", "\t", value)
  value <- gsub("\\\\\"", "\"", value)
  gsub("\\\\\\\\", "\\\\", value)
}

cuda_memory_summary <- function(free_memory, total_memory) {
  if (is.na(total_memory) || total_memory <= 0) {
    return(NA_character_)
  }
  total <- bytes_to_gib(total_memory)
  if (!is.na(free_memory) && free_memory >= 0) {
    return(paste0(bytes_to_gib(free_memory), " GiB free / ", total, " GiB total"))
  }
  paste0(total, " GiB total")
}

bytes_to_gib <- function(bytes) {
  format(round(bytes / 1024^3, 2), nsmall = 2L, trim = TRUE)
}

first_nonempty <- function(...) {
  values <- unlist(list(...), use.names = FALSE)
  values <- values[!is.na(values) & nzchar(values)]
  if (length(values) == 0L) NA_character_ else values[1L]
}

combine_nonempty <- function(...) {
  values <- unlist(list(...), use.names = FALSE)
  values <- values[!is.na(values) & nzchar(values)]
  values <- unique(values)
  if (length(values) == 0L) NA_character_ else paste(values, collapse = ", ")
}
