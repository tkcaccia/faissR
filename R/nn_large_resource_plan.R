# Private resource preflight for large KNN. Timing and recall need audits.
nn_large_knn_resource_plan <- function(
    n, p, k, backend, storage_mode, host_memory_bytes,
    device_memory_bytes = NA_real_, gpu_count = 0L,
    cpu_threads_available = 16L, target_recall = 0.99,
    device_headroom_fraction = 0.20
) {
    whole <- function(value, lower) {
        length(value) == 1L && is.numeric(value) &&
            is.finite(value) && value == floor(value) && value >= lower
    }
    if (!whole(n, 2) || n > .Machine$integer.max ||
        !whole(p, 1) || !whole(k, 1) || k >= n ||
        !whole(host_memory_bytes, 1) ||
        !whole(gpu_count, 0) || !whole(cpu_threads_available, 1) ||
        length(backend) != 1L ||
        !backend %in% c("cpu", "cuda", "metal") ||
        length(storage_mode) != 1L ||
        !storage_mode %in% c("resident", "file_backed") ||
        length(target_recall) != 1L ||
        !target_recall %in% c(0.90, 0.95, 0.99) ||
        length(device_headroom_fraction) != 1L ||
        !is.numeric(device_headroom_fraction) ||
        !is.finite(device_headroom_fraction) ||
        device_headroom_fraction < 0 ||
        device_headroom_fraction >= 1) {
        stop("Invalid large KNN resource request.", call. = FALSE)
    }
    if ((backend == "cpu" && gpu_count != 0L) ||
        (backend == "metal" && gpu_count != 1L) ||
        (backend == "cuda" && gpu_count < 1L) ||
        (backend != "cpu" && !whole(device_memory_bytes, 1))) {
        stop("Backend and device resources do not match.", call. = FALSE)
    }
    input_bytes <- 4 * n * p
    r_output_bytes <- 12 * n * k
    compact_output_bytes <- 8 * n * k
    r_output_supported <- as.double(n) * (as.double(k) + 1) <=
        .Machine$integer.max
    file_offset_addressable <- as.double(n) * as.double(k) <=
        floor((2^53 - 1) / 4)
    raw_ivf_index_bytes <- n * (4 * p + 8)
    minimum_cuda_gpus_raw <- if (backend == "cuda") {
        ceiling(raw_ivf_index_bytes / device_memory_bytes)
    } else NA_real_
    minimum_cuda_gpus_with_headroom <- if (backend == "cuda") {
        ceiling(raw_ivf_index_bytes /
                (device_memory_bytes * (1 - device_headroom_fraction)))
    } else NA_real_
    index_bytes_per_gpu <- if (backend == "cuda") {
        raw_ivf_index_bytes / gpu_count
    } else NA_real_
    file_backed_cuda <- backend == "cuda" &&
        storage_mode == "file_backed"
    shard_rows <- if (file_backed_cuda) {
        min(n, floor(gpu_count * device_memory_bytes *
            (1 - device_headroom_fraction) / (4 * p + 8)))
    } else NA_real_
    shard_count <- if (file_backed_cuda && shard_rows >= 1) {
        ceiling(n / shard_rows)
    } else NA_real_
    device_floor_pass <- if (backend == "cuda" &&
                             storage_mode == "resident") {
        index_bytes_per_gpu <= device_memory_bytes
    } else NA
    resident_floor_pass <- if (storage_mode == "resident") {
        input_bytes + r_output_bytes <= host_memory_bytes
    } else NA
    batch_budget <- if (backend == "cpu") host_memory_bytes else
        device_memory_bytes
    batch_candidate <- as.integer(max(1, min(65536,
        floor(0.05 * batch_budget / (4 * p + 12 * k)))))
    list(
        status = "resource_floor_only_not_recall_or_time_validated",
        n = n, p = p, k = k, backend = backend,
        storage_mode = storage_mode, gpu_count = gpu_count,
        target_recall = target_recall, measured_recall = NA_real_,
        index_candidate = if (backend == "cuda") "ivf_flat" else NA_character_,
        multi_gpu_mode_candidate = if (backend == "cuda" &&
                                       gpu_count > 1L) "shard" else "single",
        nlist_candidates = if (backend == "cuda") {
            unique(nn_cuda_ivf_large_candidates(n, k)$nlist)
        } else integer(),
        selected_nlist = NA_integer_, selected_nprobe = NA_integer_,
        query_batch_candidate = batch_candidate,
        cpu_threads_candidate = as.integer(min(16L, cpu_threads_available)),
        input_float32_bytes = input_bytes,
        compact_int32_float32_output_bytes = compact_output_bytes,
        r_int32_float64_output_bytes = r_output_bytes,
        minimum_one_pass_file_io_bytes = if (storage_mode == "file_backed") {
            input_bytes + compact_output_bytes
        } else NA_real_,
        file_backed_cuda_shard_rows_raw_candidate = shard_rows,
        file_backed_cuda_minimum_reference_shards_raw = shard_count,
        file_backed_cuda_shard_index_bytes_per_gpu = if (
            file_backed_cuda && shard_rows >= 1) {
            shard_rows * (4 * p + 8) / gpu_count
        } else NA_real_,
        file_backed_cuda_minimum_query_read_bytes = if (
            file_backed_cuda && !is.na(shard_count)) {
            input_bytes * shard_count
        } else NA_real_,
        raw_ivf_index_bytes_per_gpu = index_bytes_per_gpu,
        minimum_cuda_gpus_raw_ivf = minimum_cuda_gpus_raw,
        minimum_cuda_gpus_with_headroom =
            minimum_cuda_gpus_with_headroom,
        device_headroom_fraction = device_headroom_fraction,
        gpu_count_meets_headroom_floor = if (backend == "cuda" &&
            storage_mode == "resident") {
            gpu_count >= minimum_cuda_gpus_with_headroom
        } else NA,
        device_payload_model = if (file_backed_cuda) {
            "file_backed_cuda_ivf_shard_raw_floor_unvalidated"
        } else if (backend == "cuda") {
            "resident_cuda_ivf_flat_only"
        } else NA_character_,
        device_memory_bytes = device_memory_bytes,
        device_payload_floor_fits = device_floor_pass,
        host_memory_bytes = host_memory_bytes,
        resident_input_plus_output_fits = resident_floor_pass,
        resident_full_r_output_admitted = if (
            storage_mode == "resident") {
            resident_floor_pass && r_output_supported
        } else NA,
        r_full_output_vector_supported = r_output_supported,
        file_backed_output_format_addressable = if (
            storage_mode == "file_backed") {
            file_offset_addressable
        } else NA
    )
}
