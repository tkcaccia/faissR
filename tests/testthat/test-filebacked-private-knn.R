test_that("private bounded KNN preserves ties and self exclusion", {
    skip_if_not_installed("Rcpp")
    skip_if_not(faiss_available(), "FAISS is required")
    expect_false("faissR_matrix_knn_batch_v1" %in%
        getNamespaceExports("faissR"))
    expect_false("faissR_matrix_knn_batch_v2" %in%
        getNamespaceExports("faissR"))
    header <- system.file("include", "faissR_api_v1.h", package = "faissR")
    Rcpp::cppFunction(
        code = '
        SEXP faissR_test_file_batch(SEXP query, SEXP reference,
             SEXP rows, int k, std::string backend, bool exclude_self,
             double memory_limit) {
          faissR_matrix_knn_batch_v1_fun fn =
            faissR_get_matrix_knn_batch_v1();
          return fn(query, reference, rows, Rcpp::wrap(k),
            Rcpp::wrap(backend), Rcpp::wrap("stream_exact"),
            Rcpp::wrap(2), Rcpp::wrap(memory_limit),
            Rcpp::wrap(backend == "cpu" ? 2 : 1),
            Rcpp::wrap(exclude_self));
        }',
        includes = paste0('#include "', header, '"')
    )
    Rcpp::cppFunction(
        code = '
        SEXP faissR_test_file_batch_v2(SEXP query, SEXP reference,
             SEXP rows, int k, std::string backend, bool exclude_self,
             double memory_limit, Rcpp::IntegerVector devices) {
          faissR_matrix_knn_batch_v2_fun fn =
            faissR_get_matrix_knn_batch_v2();
          return fn(query, reference, rows, Rcpp::wrap(k),
            Rcpp::wrap(backend), Rcpp::wrap("stream_exact"),
            Rcpp::wrap(2), Rcpp::wrap(memory_limit),
            Rcpp::wrap(1), Rcpp::wrap(exclude_self), devices);
        }',
        includes = paste0('#include "', header, '"')
    )
    x <- matrix(c(0, 0, 0, 0, 1, 0, 1, 0, 2, 0),
        ncol = 2L, byrow = TRUE)
    directory <- tempfile("faissr-file-batch-")
    dir.create(directory)
    on.exit(unlink(directory, recursive = TRUE), add = TRUE)
    path <- file.path(directory, "ties.f32")
    writeBin(as.numeric(t(x)), path, size = 4L, endian = "little")
    spec <- list(format = "f32", path = path, nrow = nrow(x),
        ncol = ncol(x))
    observed <- faissR_test_file_batch(spec, spec, c(1, 3, 5),
        3L, "cpu", TRUE, 128 * 1024^2)
    for (i in seq_along(c(1, 3, 5))) {
        row <- c(1, 3, 5)[i]
        distance <- sqrt(rowSums((sweep(x, 2L, x[row, ]))^2))
        distance[row] <- Inf
        expected <- order(distance, seq_len(nrow(x)))[seq_len(3L)]
        expect_equal(unname(observed$indices[i, ]), expected)
        expect_equal(unname(observed$distances[i, ]),
            distance[expected], tolerance = 1e-6)
    }
    expect_identical(observed$backend_used, "faiss_cpu_stream_exact")
    expect_identical(observed$provider, "faissR")
    expect_true(observed$exact)
    expect_equal(length(observed$device_ids), 0L)
    expect_equal(observed$observed_metal_owned_live_buffer_bytes, 0)
    expect_identical(observed$search_seconds_scope,
        "complete_batch_wall_including_file_io")
    stages <- unlist(observed[c("query_read_seconds",
        "reference_read_seconds", "candidate_processing_seconds",
        "result_format_seconds")], use.names = FALSE)
    expect_true(all(is.finite(stages) & stages >= 0))
    expect_lte(sum(stages), observed$total_seconds + 1e-6)
    resident <- list(format = "memory", data = x,
        nrow = nrow(x), ncol = ncol(x))
    resident_result <- faissR_test_file_batch(resident, resident,
        c(1, 3, 5), 3L, "cpu", TRUE, 128 * 1024^2)
    expect_identical(resident_result$residency, "resident")
    expect_identical(resident_result$search_seconds_scope,
        "complete_batch_wall_resident")
    expect_identical(resident_result$indices, observed$indices)
    expect_equal(resident_result$distances, observed$distances)
    f32_resident <- resident
    f32_resident$data <- float::fl(x)
    f32_result <- faissR_test_file_batch(f32_resident, f32_resident,
        c(1, 3, 5), 3L, "cpu", TRUE, 128 * 1024^2)
    expect_identical(f32_result$indices, resident_result$indices)
    expect_equal(f32_result$distances, resident_result$distances)
    expect_equal(resident_result$peak_ram_bytes_estimate -
        f32_result$peak_ram_bytes_estimate, 4 * length(x))
    view <- list(format = "view", source = f32_resident, first = 2,
                 nrow = 4, ncol = 2)
    view_result <- faissR_test_file_batch(view, view, c(1, 3), 2L, "cpu",
        TRUE, 128 * 1024^2)
    subset <- list(format = "memory", data = x[2:5, ], nrow = 4, ncol = 2)
    subset_result <- faissR_test_file_batch(subset, subset, c(1, 3), 2L,
        "cpu", TRUE, 128 * 1024^2)
    expect_identical(view_result$indices, subset_result$indices)
    expect_equal(view_result$distances, subset_result$distances)
    mixed <- faissR_test_file_batch(resident, spec, 1, 3L,
        "cpu", FALSE, 128 * 1024^2)
    expect_identical(mixed$residency, "mixed")
    expect_identical(mixed$search_seconds_scope,
        "complete_batch_wall_including_file_io")
    expect_equal(unname(mixed$indices[1L, ]), c(1L, 2L, 3L))
    fbin_path <- file.path(directory, "ties.fbin")
    connection <- file(fbin_path, "wb")
    writeBin(as.integer(dim(x)), connection, size = 4L,
        endian = "little")
    writeBin(as.numeric(t(x)), connection, size = 4L,
        endian = "little")
    close(connection)
    fbin <- list(format = "fbin", path = fbin_path,
        nrow = nrow(x), ncol = ncol(x))
    view <- list(format = "view", source = spec, first = 2L,
        nrow = 2L, ncol = ncol(x))
    viewed <- faissR_test_file_batch(view, fbin, c(1, 2), 3L,
        "cpu", FALSE, 128 * 1024^2)
    expect_identical(viewed$residency, "file-backed")
    expect_equal(unname(viewed$indices[1L, ]), c(1L, 2L, 3L))
    expect_equal(unname(viewed$indices[2L, ]), c(3L, 4L, 1L))
    if (cuda_available() && faiss_gpu_available()) {
        unique <- cbind(c(0, 1, 4, 9, 16), rep(0, 5L))
        unique_path <- file.path(directory, "unique.f32")
        writeBin(as.numeric(t(unique)), unique_path, size = 4L,
            endian = "little")
        unique_spec <- list(format = "f32", path = unique_path,
            nrow = 5L, ncol = 2L)
        cuda <- faissR_test_file_batch(unique_spec, unique_spec,
            c(1, 3, 5), 2L, "cuda", TRUE, 128 * 1024^2)
        for (i in seq_along(c(1, 3, 5))) {
            row <- c(1, 3, 5)[i]
            distance <- sqrt(rowSums(sweep(unique, 2L,
                unique[row, ])^2))
            distance[row] <- Inf
            expected <- order(distance)[seq_len(2L)]
            expect_equal(unname(cuda$indices[i, ]), expected)
            expect_equal(unname(cuda$distances[i, ]),
                distance[expected], tolerance = 1e-5)
        }
        expect_identical(cuda$backend_used,
            "faiss_gpu_flat_file_batch")
        expect_identical(cuda$provider, "faissR")
        expect_identical(cuda$residency, "file-backed")
        expect_equal(length(cuda$device_ids), 1L)
        expect_true(cuda$exact)
        expect_gt(cuda$peak_device_bytes, 0)
        expect_gte(cuda$index_build_seconds, 0)
        expect_gte(cuda$search_seconds, 0)
        expect_identical(cuda$search_seconds_scope,
            "faiss_index_search_including_internal_transfers")
        selected <- faissR_test_file_batch_v2(unique_spec,
            unique_spec, c(1, 3, 5), 2L, "cuda", TRUE,
            128 * 1024^2, 0L)
        expect_identical(selected$indices, cuda$indices)
        expect_equal(selected$distances, cuda$distances)
        expect_identical(selected$device_ids, 0L)
        expect_identical(selected$backend_used,
            "faiss_gpu_flat_file_batch")
        expect_equal(selected$peak_device_bytes_by_id,
            selected$peak_device_bytes)
        expect_error(faissR_test_file_batch_v2(unique_spec,
            unique_spec, 1, 2L, "cuda", TRUE,
            128 * 1024^2, c(0L, 0L)), "distinct")
        tied <- faissR_test_file_batch(spec, spec, c(1, 3, 5),
            3L, "cuda", TRUE, 128 * 1024^2)
        expect_identical(tied$indices, observed$indices)
        expect_equal(tied$distances, observed$distances,
            tolerance = 1e-5)
        for (i in seq_along(c(1, 3, 5))) {
            row <- c(1, 3, 5)[i]
            ids <- tied$indices[i, ]
            all_distance <- sqrt(rowSums(sweep(x, 2L,
                x[row, ])^2))
            all_distance[row] <- Inf
            cutoff <- sort(all_distance)[3L]
            expect_false(row %in% ids)
            expect_equal(length(unique(ids)), 3L)
            expect_true(all(all_distance[ids] <= cutoff + 1e-5))
            expect_equal(unname(tied$distances[i, ]),
                all_distance[ids], tolerance = 1e-5)
        }
    } else {
        expect_error(faissR_test_file_batch(spec, spec, 1, 2L,
            "cuda", TRUE, 128 * 1024^2), "unavailable")
        expect_error(faissR_test_file_batch_v2(spec, spec, 1, 2L,
            "cuda", TRUE, 128 * 1024^2, 0L), "unavailable")
    }
    if (faissR:::nn_metal_available_cpp()) {
        metal <- faissR_test_file_batch(spec, spec, c(1, 3, 5),
            3L, "metal", TRUE, 128 * 1024^2)
        expect_identical(metal$backend_used,
            "faissr_native_metal_exact_file_batch")
        expect_identical(metal$provider, "faissR")
        expect_identical(metal$residency, "file-backed")
        expect_identical(metal$device_ids, 0L)
        expect_identical(metal$device_memory_model,
            "unified_shared_estimate")
        expect_gt(metal$observed_metal_owned_live_buffer_bytes, 0)
        expect_identical(metal$search_seconds_scope,
            "complete_batch_wall_including_file_io")
        expect_identical(metal$indices, observed$indices)
        expect_equal(metal$distances, observed$distances,
            tolerance = 1e-5)
        metal_resident <- faissR_test_file_batch(resident, resident,
            c(1, 3, 5), 3L, "metal", TRUE, 128 * 1024^2)
        expect_identical(metal_resident$residency, "resident")
        expect_gt(metal_resident$observed_metal_owned_live_buffer_bytes, 0)
        expect_identical(metal_resident$search_seconds_scope,
            "complete_batch_wall_resident")
        expect_identical(metal_resident$indices, observed$indices)
        expect_equal(metal_resident$distances, observed$distances,
            tolerance = 1e-5)
    } else {
        expect_error(faissR_test_file_batch(spec, spec, 1, 2L,
            "metal", TRUE, 128 * 1024^2), "Metal-enabled")
    }
    expect_error(faissR_test_file_batch(resident, spec, 1, 2L,
        "cpu", TRUE, 128 * 1024^2), "dimensions or settings")
    expect_error(faissR_test_file_batch(spec, spec, 6, 2L,
        "cpu", TRUE, 128 * 1024^2), "row_ids")
    expect_error(faissR_test_file_batch(spec, spec, 1, 2L,
        "cpu", TRUE, 1), "memory_limit")
})
