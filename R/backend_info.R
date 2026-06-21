#' Summarize native neighbour-search backend availability
#'
#' `backend_info()` reports which `faissR` nearest-neighbour backends can
#' currently run. It never silently falls back from an explicit GPU request to
#' CPU; this table is informational only.
#'
#' @return A data frame with one row per compiled/runtime backend family and
#'   columns describing availability, public call hints, resolved/internal route
#'   labels, device/runtime hints, and a short note.
#' @export
backend_info <- function() {
  cuda_knn <- backend_flag(cuda_available)
  faiss_knn <- backend_flag(faiss_available)
  cuvs_knn <- backend_flag(cuvs_available)
  cugraph_graph <- backend_flag(cugraph_available)
  cuda <- cuda_summary()
  faiss <- faiss_summary()
  cuvs <- cuvs_summary()

  data.frame(
    backend = c("cpu", "faiss", "faiss_gpu_cuvs", "cuvs", "cuda", "cugraph"),
    available = c(TRUE, faiss_knn, faiss_knn && cuda_knn, cuvs_knn, cuda_knn, cugraph_graph && cuda_knn),
    knn_available = c(TRUE, faiss_knn, faiss_knn && cuda_knn, cuvs_knn, cuda_knn, FALSE),
    public_call = c(
      "backend = \"cpu\"",
      "backend = \"cpu\" or \"cuda\", method = \"flat\"/\"IVF\"/\"HNSW\"/\"CAGRA\" as supported",
      "backend = \"cuda\", method = \"IVF\"/\"IVFPQ\"/\"CAGRA\"",
      "backend = \"cuda\", method = \"bruteforce\"/\"NNDescent\"/\"CAGRA\"",
      "backend = \"cuda\"",
      "graph_cluster(..., backend = \"cuda\")"
    ),
    resolved_route = c(
      "cpu",
      "faiss_flat_l2/faiss_ivf/faiss_hnsw/faiss_gpu_*",
      "faiss_gpu_ivf_flat/faiss_gpu_ivfpq/faiss_gpu_cagra",
      "cuda_cuvs_bruteforce/cuda_cuvs_nndescent/cuda_cuvs_cagra",
      "cuda",
      "graph_cluster(..., backend = \"cuda\")"
    ),
    device = c(
      cpu_summary(),
      faiss$device,
      cuda$device,
      cuvs$device,
      cuda$device,
      cuda$device
    ),
    runtime = c(
      R.version$platform,
      faiss$runtime,
      combine_nonempty(faiss$runtime, "FAISS GPU IVF and CAGRA indexes backed by NVIDIA cuVS when FAISS is built with cuVS"),
      cuvs$runtime,
      cuda$runtime,
      combine_nonempty(cuda$runtime, "RAPIDS libcugraph C API")
    ),
    note = c(
      "Native CPU path is always available.",
      if (faiss_knn) {
        "Real FAISS C++ KNN is available behind public CPU/CUDA method requests."
      } else {
        "Real FAISS C++ KNN is unavailable; FAISS-backed method requests will fail."
      },
      if (faiss_knn && cuda_knn) {
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
      if (cugraph_graph && cuda_knn) {
        "RAPIDS libcugraph graph clustering is available for explicit CUDA Louvain/Leiden requests; random_walking remains CPU-only."
      } else {
        "RAPIDS libcugraph graph clustering is unavailable; explicit CUDA graph-clustering requests will fail unless rebuilt with libcugraph."
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
  reason <- json_get_string(text, "reason")
  runtime <- if (isTRUE(available)) {
    "FAISS C++ library"
  } else if (!is.na(reason)) {
    reason
  } else {
    NA_character_
  }
  list(device = "CPU/GPU depending on requested FAISS index", runtime = runtime)
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
