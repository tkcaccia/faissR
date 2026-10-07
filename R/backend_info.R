#' Summarize native neighbour-search backend availability
#'
#' `backend_info()` reports which `faissR` nearest-neighbour backends can
#' currently run. It never silently falls back from an explicit GPU request to
#' CPU; this table is informational only.
#'
#' @section Hardware visibility and package capability:
#' A visible accelerator or installed driver does not establish that `faissR`
#' was compiled with a compatible provider. `backend_info()` reports
#' package-specific compiled and runtime capability; [cuda_available()] is
#' likewise specific to the current `faissR` installation. Framework-neutral
#' hardware discovery is supplied by the CRAN package `gpuinfo`. faissR uses
#' it for hardware, driver, memory, and device details, while its own native
#' probes remain authoritative for deciding whether a compiled provider can
#' execute.
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
    summaries <- backend_info_summaries()
    flags <- backend_info_flags(summaries)
    data.frame(
        backend = c("cpu", "faiss", "faiss_gpu_cuvs", "cuvs", "cuda"),
        available = flags$available,
        knn_available = flags$available,
        public_call = backend_info_public_calls(),
        public_backends = c("cpu", "cpu, cuda", "cuda", "cuda", "cuda"),
        supported_methods = backend_info_supported_methods(),
        supported_metrics = backend_info_supported_metrics(),
        resolved_route = backend_info_resolved_routes(),
        device = backend_info_devices(summaries),
        runtime = backend_info_runtimes(summaries),
        note = backend_info_notes(flags, summaries),
        stringsAsFactors = FALSE
    )
}

backend_info_flags <- function(summaries) {
    cuda <- backend_flag(cuda_available)
    faiss <- backend_flag(faiss_available)
    cuvs <- backend_flag(cuvs_available)
    list(
        cuda = cuda,
        faiss = faiss,
        cuvs = cuvs,
        available = c(
            TRUE,
            faiss,
            isTRUE(summaries$faiss$gpu) && cuda,
            cuvs,
            cuda
        )
    )
}

backend_info_summaries <- function() {
    hardware <- gpuinfo_hardware_info()
    list(
        hardware = hardware,
        cuda = cuda_summary(hardware),
        faiss = faiss_summary(),
        cuvs = cuvs_summary()
    )
}

backend_info_public_calls <- function() {
    c(
        "backend = \"cpu\"",
        paste0(
            "backend = \"cpu\" or \"cuda\", method = ",
            "\"flat\"/\"ivf\"/\"ivfpq\"/\"hnsw\"/",
            "\"ivfpq_fastscan\"/\"cagra\" as supported"
        ),
        "backend = \"cuda\", method = \"ivf\"/\"ivfpq\"/\"cagra\"",
        paste0(
            "backend = \"cuda\", method = \"bruteforce\"/\"hnsw\"/",
            "\"nndescent\"/\"ivfpq_fastscan\"/\"cagra\""
        ),
        "backend = \"cuda\""
    )
}

backend_info_supported_methods <- function() {
    c(
        paste(
            "auto, exact, flat, bruteforce, grid, hnsw, ivf, ivfpq,",
            "ivfpq_fastscan, vamana, nsg, nndescent"
        ),
        paste(
            "flat, ivf, ivfpq, hnsw, ivfpq_fastscan, nsg; GPU",
            "flat/ivf/ivfpq/cagra when FAISS GPU is available"
        ),
        "ivf, ivfpq, cagra",
        "bruteforce, hnsw, nndescent, ivfpq_fastscan, cagra",
        paste(
            "grid, flat, bruteforce, hnsw, ivf, ivfpq, ivfpq_fastscan,",
            "vamana, nsg, nndescent, cagra where compiled"
        )
    )
}

backend_info_supported_metrics <- function() {
    c(
        paste(
            "euclidean, cosine, correlation; method-specific exclusions",
            "in nn_capabilities()"
        ),
        paste(
            "euclidean, cosine, correlation for Flat/IVF/IVFPQ/HNSW and",
            "CPU IVFPQ FastScan where available; public NSG uses the native",
            "CPU route, with deterministic FAISS HNSW seeding on large",
            "high-dimensional CPU inputs; explicit FAISS NSG is Euclidean-only"
        ),
        "euclidean, cosine, correlation for IVF/IVFPQ and CAGRA",
        paste(
            "euclidean, cosine, correlation for direct brute force, direct",
            "IVF/PQ, HNSW from CAGRA, direct CAGRA, and direct cuVS",
            "NN-descent using metric transforms where needed"
        ),
        paste(
            "euclidean, cosine, correlation where the selected CUDA method",
            "supports the metric"
        )
    )
}

backend_info_resolved_routes <- function() {
    c(
        "implementation label: cpu",
        paste(
            "implementation labels include faiss_flat_l2, faiss_ivf,",
            "faiss_hnsw, faiss_ivfpq_fastscan, and faiss_gpu_*"
        ),
        paste(
            "implementation labels include faiss_gpu_ivf_flat,",
            "faiss_gpu_ivfpq, and faiss_gpu_cagra"
        ),
        paste(
            "implementation labels include cuda_cuvs_bruteforce,",
            "cuda_cuvs_hnsw, cuda_cuvs_nndescent,",
            "cuda_cuvs_ivfpq_fastscan, and cuda_cuvs_cagra"
        ),
        paste(
            "implementation labels include cuda_grid, cuda_vamana, and",
            "cuda_nsg; exact CUDA may report cuda"
        )
    )
}

backend_info_devices <- function(summaries) {
    with(summaries, {
        c(
            cpu_summary(hardware),
            faiss$device,
            cuda$device,
            cuda$device,
            cuda$device
        )
    })
}

backend_info_runtimes <- function(summaries) {
    faiss_gpu_runtime <- summaries$faiss$runtime
    if (isTRUE(summaries$faiss$gpu)) {
        faiss_gpu_runtime <- combine_nonempty(
            faiss_gpu_runtime,
            paste(
                "FAISS GPU IVF and CAGRA indexes backed by NVIDIA cuVS",
                "when FAISS is built with cuVS"
            )
        )
    }
    c(
        R.version$platform,
        summaries$faiss$runtime,
        faiss_gpu_runtime,
        summaries$cuvs$runtime,
        summaries$cuda$runtime
    )
}

backend_info_notes <- function(flags, summaries) {
    faiss_note <- if (flags$faiss) {
        paste(
            "Real FAISS C++ KNN is available behind public CPU/CUDA method",
            "requests; IVFPQ FastScan CPU search requires linked FAISS",
            "FastScan support."
        )
    } else {
        "Real FAISS C++ KNN is unavailable; FAISS method requests will fail."
    }
    c(
        "Native CPU path is always available.",
        faiss_note,
        backend_info_faiss_gpu_note(flags, summaries),
        backend_info_cuvs_note(flags$cuvs),
        backend_info_cuda_note(flags$cuda)
    )
}

backend_info_faiss_gpu_note <- function(flags, summaries) {
    if (isTRUE(summaries$faiss$gpu) && flags$cuda) {
        return(paste(
            "FAISS GPU IVF-Flat, IVF-PQ, and CAGRA use FAISS GPU indexes",
            "with NVIDIA cuVS integration when linked FAISS provides it;",
            "result backends identify the corresponding cuVS GPU index."
        ))
    }
    paste(
        "FAISS GPU cuVS-integrated IVF/CAGRA requests are unavailable;",
        "CUDA requests fail unless another validated route is available."
    )
}

backend_info_cuvs_note <- function(available) {
    if (available) {
        "RAPIDS cuVS CUDA KNN is available behind public CUDA requests."
    } else {
        paste(
            "RAPIDS cuVS CUDA KNN is unavailable; cuVS-backed public CUDA",
            "method requests will fail."
        )
    }
}

backend_info_cuda_note <- function(available) {
    if (available) {
        "Native CUDA KNN path is available for explicit CUDA requests."
    } else {
        "Native CUDA KNN is unavailable; explicit CUDA requests will fail."
    }
}

backend_flag <- function(fn) {
    tryCatch(isTRUE(fn()), error = function(e) FALSE)
}

gpuinfo_hardware_info <- function() {
    tryCatch(gpuinfo::hardware_info(), error = function(e) NULL)
}

gpuinfo_cpu_model <- function(info = gpuinfo_hardware_info()) {
    value <- if (is.list(info) && is.list(info$cpu)) {
        info$cpu$model
    } else {
        NA_character_
    }
    value <- as.character(value)[1L]
    if (is.na(value) || !nzchar(value)) NA_character_ else value
}

gpuinfo_cuda_device <- function(info = gpuinfo_hardware_info()) {
    if (!is.list(info) || !is.data.frame(info$gpu) || !nrow(info$gpu)) {
        return(NA_character_)
    }
    rows <- which(info$gpu$backend %in% "cuda")
    if (!length(rows)) {
        return(NA_character_)
    }
    value <- as.character(info$gpu$model[rows[1L]])
    if (is.na(value) || !nzchar(value)) NA_character_ else value
}

gpuinfo_numeric_scalar <- function(value) {
    if (!length(value)) {
        return(NA_real_)
    }
    value <- value[[1L]]
    if (is.numeric(value)) {
        return(as.numeric(value))
    }
    parsed <- utils::type.convert(as.character(value), as.is = TRUE)
    if (length(parsed) == 1L && is.numeric(parsed)) {
        as.numeric(parsed)
    } else {
        NA_real_
    }
}

cpu_summary <- function(info = gpuinfo_hardware_info()) {
    cores <- if (is.list(info) && is.list(info$cpu)) {
        gpuinfo_numeric_scalar(info$cpu$logical_cores)
    } else {
        NA_integer_
    }
    if (length(cores) != 1L || is.na(cores) || !is.finite(cores)) {
        "CPU"
    } else {
        paste0("CPU (", cores, " logical cores)")
    }
}

cuda_summary <- function(info = gpuinfo_hardware_info()) {
    package <- cuda_package_summary()
    hardware <- gpuinfo_cuda_summary(info)
    list(
        device = first_nonempty(
            hardware$device,
            if (isTRUE(package$available)) "CUDA GPU" else NA_character_
        ),
        runtime = combine_nonempty(package$runtime, hardware$runtime)
    )
}

cuda_package_summary <- function() {
    text <- tryCatch(
        cuda_device_info_json_cpp(),
        error = function(e) NA_character_
    )
    if (length(text) != 1L || is.na(text) || !nzchar(text)) {
        return(list(available = FALSE, runtime = NA_character_))
    }

    available <- json_get_bool(text, "available")
    if (isTRUE(available)) {
        compiled_toolkit <- json_get_string(text, "compiled_toolkit")
        runtime_version <- json_get_string(text, "runtime_version")
        driver_version <- json_get_string(text, "driver_version")
        runtime <- combine_nonempty(
            cuda_version_summary(
                compiled_toolkit,
                runtime_version,
                driver_version
            )
        )
        return(list(available = TRUE, runtime = runtime))
    }

    reason <- json_get_string(text, "reason")
    list(
        available = FALSE,
        runtime = combine_nonempty(
            cuda_version_summary(
                json_get_string(text, "compiled_toolkit"),
                json_get_string(text, "runtime_version"),
                json_get_string(text, "driver_version")
            ),
            reason
        )
    )
}

gpuinfo_cuda_summary <- function(info = gpuinfo_hardware_info()) {
    if (!is.list(info)) {
        return(list(device = NA_character_, runtime = NA_character_))
    }
    cuda <- if (is.list(info$cuda)) info$cuda else list()
    device <- gpuinfo_cuda_device(info)
    memory <- NA_real_
    if (is.data.frame(info$gpu) && nrow(info$gpu)) {
        rows <- which(info$gpu$backend %in% "cuda")
        if (length(rows)) {
            memory <- gpuinfo_numeric_scalar(
                info$gpu$memory_mb[rows[1L]]
            ) * 1024^2
        }
    }
    runtime <- combine_nonempty(
        gpuinfo_cuda_field(cuda$driver_version, "NVIDIA driver "),
        gpuinfo_cuda_field(
            cuda$compute_capability,
            "compute capability "
        ),
        cuda_memory_summary(NA_real_, memory)
    )
    list(device = device, runtime = runtime)
}

gpuinfo_cuda_field <- function(value, prefix) {
    value <- as.character(value)[1L]
    if (is.na(value) || !nzchar(value)) {
        NA_character_
    } else {
        paste0(prefix, value)
    }
}

cuda_version_summary <- function(compiled, runtime, driver) {
    fields <- c(
        if (!is.na(compiled) && compiled != "unknown") {
            paste0("compiled CUDA ", compiled)
        },
        if (!is.na(runtime) && runtime != "unknown") {
            paste0("runtime ", runtime)
        },
        if (!is.na(driver) && driver != "unknown") {
            paste0("driver API ", driver)
        }
    )
    if (!length(fields)) NA_character_ else paste(fields, collapse = ", ")
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
    version <- json_get_string(text, "version")
    reason <- json_get_string(text, "reason")
    runtime <- if (isTRUE(available)) {
        combine_nonempty(
            if (!is.na(version)) {
                paste0("FAISS C++ library ", version)
            } else {
                "FAISS C++ library"
            },
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
        device = if (isTRUE(gpu)) {
            "CPU/GPU depending on requested FAISS index"
        } else {
            "CPU"
        },
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
    isTRUE(json_get_bool(text, "available")) &&
        isTRUE(json_get_bool(text, "gpu"))
}

cuvs_summary <- function() {
    text <- tryCatch(
        cuvs_info_json_cpp(),
        error = function(e) NA_character_
    )
    available <- json_get_bool(text, "available")
    reason <- json_get_string(text, "reason")
    runtime <- if (isTRUE(available)) {
        "RAPIDS cuVS C API"
    } else if (!is.na(reason)) {
        reason
    } else {
        NA_character_
    }
    list(runtime = runtime)
}

json_get_bool <- function(text, key) {
    value <- json_capture(text, key, "(true|false)")
    if (is.na(value)) {
        return(NA)
    }
    identical(tolower(value), "true")
}

json_get_number <- function(text, key) {
    value <- json_capture(text, key, "([0-9]+(?:\\.[0-9]+)?)")
    if (is.na(value)) {
        return(NA_real_)
    }
    as.numeric(value)
}

json_get_string <- function(text, key) {
    value <- json_capture(text, key, "\"((?:\\\\.|[^\"\\\\])*)\"")
    if (is.na(value)) {
        return(NA_character_)
    }
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
        return(paste0(
            bytes_to_gib(free_memory),
            " GiB free / ",
            total,
            " GiB total"
        ))
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
