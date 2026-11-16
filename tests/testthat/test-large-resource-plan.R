test_that("large resource preflight rejects an impossible resident index", {
    plan <- faissR:::nn_large_knn_resource_plan(
        1e9, 96L, 30L, "cuda", "resident",
        host_memory_bytes = 256 * 1024^3,
        device_memory_bytes = 48 * 1024^3, gpu_count = 2L
    )
    expect_false(plan$device_payload_floor_fits)
    expect_false(plan$resident_input_plus_output_fits)
    expect_false(plan$resident_full_r_output_admitted)
    expect_false(plan$r_full_output_vector_supported)
    expect_true(is.na(plan$file_backed_output_format_addressable))
    expect_equal(plan$raw_ivf_index_bytes_per_gpu, 196e9)
    expect_equal(plan$minimum_cuda_gpus_raw_ivf, 8)
    expect_equal(plan$minimum_cuda_gpus_with_headroom, 10)
    expect_false(plan$gpu_count_meets_headroom_floor)
    expect_true(is.na(plan$selected_nprobe))
    expect_true(is.na(plan$measured_recall))
})

test_that("file-backed resource floor includes I/O without a time claim", {
    plan <- faissR:::nn_large_knn_resource_plan(
        1e9, 96L, 30L, "cpu", "file_backed",
        host_memory_bytes = 128 * 1024^3,
        cpu_threads_available = 16L, target_recall = 0.95
    )
    expect_equal(plan$minimum_one_pass_file_io_bytes,
                 4e9 * 96 + 8e9 * 30)
    expect_identical(plan$cpu_threads_candidate, 16L)
    expect_true(is.na(plan$device_payload_floor_fits))
    expect_true(is.na(plan$resident_input_plus_output_fits))
    expect_true(is.na(plan$resident_full_r_output_admitted))
    expect_true(plan$file_backed_output_format_addressable)
    expect_false(plan$r_full_output_vector_supported)
    expect_true(is.na(plan$selected_nlist))
    expect_identical(plan$target_recall, 0.95)
    metal <- faissR:::nn_large_knn_resource_plan(
        1e7, 96L, 30L, "metal", "file_backed", 64e9,
        device_memory_bytes = 32e9, gpu_count = 1L
    )
    expect_true(is.na(metal$device_payload_floor_fits))
    expect_true(is.na(metal$raw_ivf_index_bytes_per_gpu))
    expect_true(is.na(metal$device_payload_model))
    expect_true(is.na(metal$minimum_cuda_gpus_raw_ivf))
    expect_true(metal$file_backed_output_format_addressable)
})

test_that("file-backed CUDA planning keeps full index and shard floors apart", {
    plan <- faissR:::nn_large_knn_resource_plan(
        1e9, 96L, 30L, "cuda", "file_backed",
        host_memory_bytes = 128 * 1024^3,
        device_memory_bytes = 48 * 1024^3, gpu_count = 1L
    )
    expected_rows <- floor(48 * 1024^3 * 0.8 / 392)
    expected_shards <- ceiling(1e9 / expected_rows)
    expect_true(is.na(plan$device_payload_floor_fits))
    expect_true(is.na(plan$gpu_count_meets_headroom_floor))
    expect_equal(plan$minimum_cuda_gpus_raw_ivf, 8)
    expect_equal(plan$file_backed_cuda_shard_rows_raw_candidate,
                 expected_rows)
    expect_equal(plan$file_backed_cuda_minimum_reference_shards_raw,
                 expected_shards)
    expect_equal(plan$file_backed_cuda_shard_index_bytes_per_gpu,
                 expected_rows * 392)
    expect_equal(plan$file_backed_cuda_minimum_query_read_bytes,
                 384e9 * expected_shards)
    expect_identical(plan$device_payload_model,
        "file_backed_cuda_ivf_shard_raw_floor_unvalidated")
    expect_true(plan$file_backed_output_format_addressable)
    expect_false(plan$r_full_output_vector_supported)
    two <- faissR:::nn_large_knn_resource_plan(
        1e9, 96L, 30L, "cuda", "file_backed",
        host_memory_bytes = 128 * 1024^3,
        device_memory_bytes = 48 * 1024^3, gpu_count = 2L
    )
    expect_gt(two$file_backed_cuda_shard_rows_raw_candidate,
              plan$file_backed_cuda_shard_rows_raw_candidate)
    expect_lt(two$file_backed_cuda_minimum_reference_shards_raw,
              plan$file_backed_cuda_minimum_reference_shards_raw)
    tiny <- faissR:::nn_large_knn_resource_plan(
        1e6, 96L, 10L, "cuda", "file_backed",
        host_memory_bytes = 1e9,
        device_memory_bytes = 400, gpu_count = 1L
    )
    expect_equal(tiny$file_backed_cuda_shard_rows_raw_candidate, 0)
    expect_true(is.na(tiny$file_backed_cuda_minimum_reference_shards_raw))
})

test_that("GPU-count floors depend on capacity and explicit headroom", {
    args <- list(1e9, 96L, 30L, "cuda", "resident", 1e12,
                 device_memory_bytes = 141e9, gpu_count = 4L)
    plan <- do.call(faissR:::nn_large_knn_resource_plan, args)
    expect_equal(plan$minimum_cuda_gpus_raw_ivf, 3)
    expect_equal(plan$minimum_cuda_gpus_with_headroom, 4)
    expect_true(plan$gpu_count_meets_headroom_floor)
    expect_true(plan$device_payload_floor_fits)
    expect_identical(plan$device_payload_model,
                     "resident_cuda_ivf_flat_only")
    args$device_headroom_fraction <- 0.35
    tight <- do.call(faissR:::nn_large_knn_resource_plan, args)
    expect_equal(tight$minimum_cuda_gpus_with_headroom, 5)
    expect_false(tight$gpu_count_meets_headroom_floor)
    expect_true(tight$device_payload_floor_fits)
})

test_that("resource request rejects inconsistent devices and recall", {
    expect_error(faissR:::nn_large_knn_resource_plan(
        1e7, 96L, 30L, "metal", "resident", 64e9,
        device_memory_bytes = 32e9, gpu_count = 2L),
        "Backend and device resources")
    expect_error(faissR:::nn_large_knn_resource_plan(
        1e7, 96L, 30L, "cuda", "resident", 64e9,
        device_memory_bytes = 32e9, gpu_count = 1L,
        target_recall = 0.97), "Invalid large KNN resource")
    expect_error(faissR:::nn_large_knn_resource_plan(
        1e7, 96L, 30L, "cuda", "resident", 64e9,
        device_memory_bytes = 32e9, gpu_count = 1L,
        device_headroom_fraction = 1), "Invalid large KNN resource")
})

test_that("100M integer dimensions do not overflow output planning", {
    plan <- faissR:::nn_large_knn_resource_plan(
        100000000L, 96L, 30L, "cuda", "resident",
        host_memory_bytes = 256 * 1024^3,
        device_memory_bytes = 48 * 1024^3, gpu_count = 2L
    )
    expect_false(plan$r_full_output_vector_supported)
    expect_false(plan$resident_full_r_output_admitted)
    backed <- faissR:::nn_large_knn_resource_plan(
        100000000L, 96L, 30L, "cpu", "file_backed",
        host_memory_bytes = 128 * 1024^3)
    expect_true(backed$file_backed_output_format_addressable)
})
