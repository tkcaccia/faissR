test_that("native Metal selection respects IVF's index and width limits", {
    skip_if_not_installed("Rcpp")
    header <- system.file("include", "faissR_api_v1.h", package = "faissR")
    Rcpp::cppFunction(
        code = '
        SEXP select_metal_knn(SEXP spec) {
          return faissR_get_knn_select_v1()(spec);
        }',
        includes = paste0('#include "', header, '"')
    )
    choose <- function(rows, columns, k = 30L, query = rows,
                       self = TRUE, file = FALSE, recall = 0.95,
                       allow_ivf = TRUE, limit = Inf,
                       method = "auto") {
        select_metal_knn(list(backend = "metal", metric = "euclidean",
            n_reference = rows, n_features = columns, n_query = query,
            k = k, target_recall = recall, self_query = self,
            file_backed = file, allow_ivf = allow_ivf,
            memory_limit_bytes = limit, metal_method = method))
    }
    for (file in c(FALSE, TRUE)) {
        expect_identical(choose(50001L, 1024L, file = file)$method, "ivf")
        wide <- choose(50001L, 1025L, file = file)
        expect_identical(wide$method, "exact")
        expect_identical(wide$reason, "metal_ivf_shape_limit")
        expect_identical(choose(8388607L, 512L, file = file)$method, "ivf")
        expect_identical(choose(8388608L, 512L, file = file)$method, "exact")
    }
    expect_identical(choose(33554432L, 64L, 128L)$method, "exact")
    expect_identical(choose(4096L, 512L, query = 8388608L,
        self = FALSE)$method, "exact")
    for (file in c(FALSE, TRUE)) {
        limited <- choose(1281167L, 1024L, file = file, recall = 0.99,
            limit = 2^30)
        expect_identical(limited$method, "exact")
        skipped <- choose(1281167L, 1024L, file = file, recall = 0.99,
            allow_ivf = FALSE)
        expect_identical(skipped$method, "exact")
        expect_null(skipped$metal_cagra_admission)
        if (Sys.info()[["sysname"]] == "Darwin") {
            expect_false(limited$metal_cagra_admission$fits)
            expect_gt(limited$metal_cagra_admission$workspace_bytes, 2^30)
        }
    }
    for (rows in c(50000000L, 100000000L)) {
        for (file in c(FALSE, TRUE)) {
            cagra <- choose(rows, 96L, file = file, recall = 0.99,
                method = "cagra")
            expect_identical(cagra$method, "cagra")
            expect_identical(cagra$reason, "metal_cagra_requested")
            if (file) {
                expect_identical(choose(rows, 96L, file = file,
                    recall = 0.95)$reason, "metal_ivf_shape_limit")
            }
            if (Sys.info()[["sysname"]] == "Darwin") {
                plan <- choose(rows, 96L, file = file, recall = 0.99,
                    limit = 2^30)$metal_cagra_admission
                expect_false(plan$fits)
                expect_identical(plan$graph_degree, 48L)
                expect_identical(plan$projection_dim, 128L)
                expect_identical(plan$profile_source, "deep1b_10m_shape_profile")
                expect_gt(plan$workspace_bytes, 2^30)
            }
        }
    }
})

test_that("wide CAGRA supports NN-descent for automatic lower recall tiers", {
  preset <- faissR:::cuvs_cagra_wide_params
  expect_identical(preset(0.99, TRUE),
    list(width = 4L, queue = 384L, builder = "nn_descent"))
  expect_identical(preset(0.95, TRUE),
    list(width = 4L, queue = 512L, builder = "ivf_pq"))
  expect_identical(preset(0.90, TRUE), preset(0.95, TRUE))
  for (recall in c(0.90, 0.95)) {
    expect_identical(preset(recall, TRUE, nn_descent = TRUE), preset(0.99, TRUE))
  }
  expect_identical(preset(0.99, FALSE),
    list(width = 2L, queue = 512L, builder = "nn_descent"))
})

test_that("resident CUDA batches write compact host output with correct IDs", {
  skip_if_not_installed("float")
  skip_if_not(faissR::cuvs_available())
  set.seed(1607L)
  samples <- 513L
  k <- 8L
  x <- float::fl(matrix(rnorm(samples * 32L), samples))
  profile <- list(graph_degree = 32L, intermediate_degree = 64L,
    batch_rows = 128L, memory_limit_bytes = 2^30)
  prior <- options(faissR.cuvs_cagra_profile = profile)
  on.exit(options(prior))
  knn <- faissR:::cuvs_cagra_wide_self(x, k, 0.99, "float", TRUE)
  exact <- nn(x, k = k, backend = "cpu", method = "exact",
    exclude_self = TRUE, n_threads = 1L)
  expect_identical(knn$backend_used, "cuda_cuvs_cagra")
  expect_identical(knn$input_type, "float32")
  expect_identical(knn$full_host_input_copy, FALSE)
  expect_s4_class(knn$distances, "float32")
  expect_identical(dim(knn$indices), c(samples, k))
  expect_true(all(knn$indices >= 1L & knn$indices <= samples))
  expect_true(all(knn$indices != seq_len(samples)))
  expect_true(all(apply(knn$indices, 1L, anyDuplicated) == 0L))
  recall <- vapply(seq_len(samples), function(i) {
    mean(knn$indices[i, ] %in% exact$indices[i, ])
  }, numeric(1L))
  expect_gte(mean(recall), 0.99)
  profile$memory_limit_bytes <- 1024L
  options(faissR.cuvs_cagra_profile = profile)
  expect_error(faissR:::cuvs_cagra_wide_self(x, k, 0.99, "float", TRUE),
    "input and output exceed memory_limit")
})

test_that("explicit native CUDA CAGRA accepts large compact reference data", {
    skip_if_not_installed("Rcpp")
    header <- system.file("include", "faissR_api_v1.h", package = "faissR")
    Rcpp::cppFunction(
        code = '
        SEXP select_cuda_knn(SEXP spec) {
          return faissR_get_knn_select_v1()(spec);
        }',
        includes = paste0('#include "', header, '"')
    )
    choose <- function(method = "cagra_width4", rows = 10000000L,
                       metric = "euclidean", self = TRUE, file = FALSE) {
        select_cuda_knn(list(backend = "cuda", metric = metric,
            n_reference = rows, n_features = 96L, n_query = rows,
            k = 30L, target_recall = 0.99, self_query = self,
            file_backed = file, allow_ivf = TRUE, cuda_method = method))
    }
    for (file in c(FALSE, TRUE)) {
        choice <- choose(file = file)
        expect_identical(choice$method, "cagra_width4")
        expect_identical(choice$reason, "cuda_cagra_requested")
        expect_identical(choose("cagra", file = file)$method, "cagra")
        expect_identical(choose("auto", file = file)$method, "ivf")
    }
    expect_error(choose(rows = 499999L), "500,000 rows")
    expect_error(choose(metric = "cosine"), "Euclidean self KNN")
    expect_error(choose(self = FALSE), "Euclidean self KNN")
})

test_that("a shared CUDA profile bounds and records file-backed batches", {
    skip_if_not_installed("float")
    skip_if_not(faissR::cuvs_available())
    set.seed(1609L)
    rows <- 513L
    columns <- 32L
    k <- 8L
    x <- matrix(rnorm(rows * columns), rows)
    path <- tempfile(fileext = ".f32")
    on.exit(unlink(path), add = TRUE)
    writeBin(as.vector(t(x)), path, size = 4L, endian = "little")
    input <- list(format = "f32", path = path, nrow = rows, ncol = columns)
    profile <- list(graph_degree = 32L, intermediate_degree = 64L,
        batch_rows = 128L, memory_limit_bytes = 2^30)
    prior <- options(faissR.cuvs_cagra_profile = profile)
    on.exit(options(prior), add = TRUE)
    index <- faissR:::cuvs_cagra_wide_file(input, k, 256L, 2^30, .99, TRUE)
    release <- TRUE
    on.exit(if (release) index$release(), add = TRUE)
    expect_identical(index$backend_used, "cuda_cuvs_cagra_file")
    expect_identical(index$query_batch_rows, 128L)
    expect_identical(index$params$graph_degree, 32L)
    expect_identical(index$params$intermediate_degree, 64L)
    expect_lt(index$device_bytes_estimate, index$device_free_bytes_at_build)
    expect_error(index$search_batch(1L, 129L), "exceeds the index or capacity")
    ids <- index$search_batch(2L, 128L)$indices
    index$release()
    release <- FALSE
    exact <- nn(float::fl(x), k = k, backend = "cpu", method = "exact",
        exclude_self = TRUE, n_threads = 1L)
    expect_true(all(ids != 2:129))
    recall <- vapply(seq_len(nrow(ids)), function(i)
        mean(ids[i, ] %in% exact$indices[i + 1L, ]), numeric(1L))
    expect_gte(mean(recall), .99)
    expect_error(faissR:::cuvs_cagra_wide_file(input, k, 256L, 1024,
        .99, TRUE), "estimate exceeds memory_limit")
})
test_that("native file-backed HNSW respects the host memory and shard limits", {
    skip_if_not_installed("Rcpp")
    header <- system.file("include", "faissR_api_v1.h", package = "faissR")
    Rcpp::cppFunction(
        code = '
        SEXP select_file_hnsw(SEXP spec) {
          return faissR_get_knn_select_v1()(spec);
        }',
        includes = paste0('#include "', header, '"')
    )
    plan <- function(n, p, gib, ref = NULL, k = 30L, m = NULL) {
        select_file_hnsw(list(
            backend = "cpu", metric = "euclidean", cpu_method = "hnsw",
            n_reference = n, n_features = p, n_query = n, k = k,
            target_recall = .99, self_query = TRUE, file_backed = TRUE,
            allow_ivf = FALSE, memory_limit_bytes = gib * 2^30,
            reference_chunk_rows = ref, hnsw_build_m = m))
    }
    single <- plan(1281167L, 1024L, 22)
    expect_identical(single$reason, "cpu_hnsw_one_index_fits")
    expect_identical(single$resources$n_shards, 1L)
    sharded <- plan(1281167L, 1024L, 8)
    expect_identical(sharded$reason, "cpu_hnsw_balanced_shards_for_budget")
    expect_identical(sharded$resources$n_shards, 4L)
    expect_lt(sharded$resources$peak_ram_bytes, 0.7 * 8 * 2^30)
    deep <- plan(10000000L, 96L, 22)
    expect_identical(deep$resources$n_shards, 1L)
    expect_equal(deep$resources$reference_chunk_rows, 10000000)
    expect_gt(deep$resources$peak_ram_bytes, 10689652 * 1024)
    expect_lt(deep$resources$peak_ram_bytes, 0.7 * 22 * 2^30)
    expect_gt(plan(10000000L, 96L, 16)$resources$n_shards, 1L)
    expect_gt(plan(10000000L, 96L, 22, m = 128L)$resources$n_shards, 1L)
    expect_equal(plan(1e9, 96L, 128)$resources$output_bytes, 240e9)
    expect_error(plan(5L, 2L, 1, 2L, 2L), "Balanced HNSW shards exceed")
    expect_error(plan(5000L, 96L, .01), "reference shard exceeds")
})

test_that("ImageNet admits the validated width-four CUDA preset", {
    skip_if_not_installed("Rcpp")
    header <- system.file("include", "faissR_api_v1.h", package = "faissR")
    Rcpp::cppFunction(
        code = '
        SEXP select_calibrated_cuda(SEXP spec) {
          return faissR_get_knn_select_v1()(spec);
        }',
        includes = paste0('#include "', header, '"')
    )
    choose <- function(rows = 1281167L, columns = 1024L, recall = 0.99,
                       file = FALSE, free = 16, host = 24, limit = 26,
                       metric = "euclidean", devices = 1L) {
        select_calibrated_cuda(list(backend = "cuda", metric = metric,
            n_reference = rows, n_features = columns, n_query = rows,
            k = 30L, target_recall = recall, self_query = TRUE,
            file_backed = file, allow_ivf = TRUE,
            allow_cuda_profile = devices <= 1L, cuda_device_count = devices,
            cuda_free_bytes = free * 2^30, host_available_bytes = host * 2^30,
            memory_limit_bytes = limit * 2^30))
    }
    current <- choose()
    expect_identical(current$method, "cagra_width4")
    expect_identical(current$cuda_cagra_admission$profile_source,
                     "imagenet_full_runs")
    expect_equal(current$cuda_cagra_admission$batch_rows, 1024)
    expect_null(current$cuda_cagra_profile)
    for (limited in list(choose(free = 8), choose(host = 8),
                         choose(limit = 12))) {
        expect_identical(limited$method, "cagra_width4")
        expect_identical(limited$reason, "cuda_cagra_resident_capacity_exceeded")
        expect_false(limited$cuda_cagra_admission$fits)
    }
    file_backed <- choose(file = TRUE, host = 8, limit = 22)
    expect_identical(file_backed$method, "cagra_width4")
    expect_equal(file_backed$cuda_cagra_profile$graph_degree, 64)
    expect_equal(file_backed$cuda_cagra_profile$intermediate_degree, 128)
    expect_equal(file_backed$cuda_cagra_profile$batch_rows, 1024)
    expect_identical(choose(file = TRUE, host = 4)$method, "cagra_sharded")
    expect_identical(choose(file = TRUE, limit = 8)$method, "cagra_sharded")
    for (recall in c(0.90, 0.95)) {
        selected <- choose(recall = recall)
        expect_identical(selected$method, "cagra_width4")
        expect_identical(selected$cuda_cagra_profile$builder, "nn_descent")
    }
    expect_identical(choose(rows = 1281166L)$cuda_cagra_admission$profile_source,
                     "shape_extrapolation")
    expect_null(choose(rows = 999999L)$cuda_cagra_admission)
    expect_null(choose(metric = "cosine")$cuda_cagra_admission)
    compact <- choose(rows = 10000000L, columns = 96L, file = TRUE)
    expect_identical(compact$cuda_cagra_admission$profile_source,
                     "deep1b_10m_full_runs")
    expect_equal(compact$cuda_cagra_profile$graph_degree, 32)
    for (rows in c(50000000L, 100000000L, 1000000000L)) {
        args <- list(rows = rows, columns = 96L)
        resident <- do.call(choose, args)
        expect_identical(resident$method, "cagra_width4")
        expect_false(resident$cuda_cagra_admission$fits)
        sharded <- do.call(choose, c(args, list(file = TRUE, limit = 22)))
        expect_identical(sharded$method, "cagra_sharded")
        expect_equal(sharded$resources$n_shards, rows / 10000000)
        expect_identical(sharded$resources$builder, "nn_descent")
    }
    large <- choose(rows = 50000000L, columns = 96L, free = 96,
                    host = 192, limit = 192)
    expect_true(large$cuda_cagra_admission$fits)
    expect_identical(large$method, "cagra_width4")

    multi <- choose(free = 8, devices = 2L)
    expect_identical(multi$reason, "cuda_cagra_reference_shards_fit")
    expect_equal(multi$cuda_cagra_admission$largest_reference_shard_rows, 640584)
    expect_lt(multi$cuda_cagra_admission$device_bytes_estimate,
              current$cuda_cagra_admission$device_bytes_estimate)
    expect_identical(multi$cuda_cagra_profile$builder, "nn_descent")
    expect_error(choose(free = 2, devices = 2L), "sharded CUDA CAGRA exceeds")
    expect_error(choose(free = 8, host = 8, devices = 2L),
        "sharded CUDA CAGRA exceeds")
    even <- choose(rows = 10000000L, columns = 96L, free = 8, host = 32,
        limit = 32, file = TRUE, devices = 2L)
    expect_identical(even$method, "cagra_width4")
    expect_equal(even$cuda_cagra_admission$largest_reference_shard_rows, 5000000)
    expect_error(choose(free = 2, file = TRUE, devices = 2L),
        "sharded CUDA CAGRA exceeds")
    expect_error(choose(devices = 0L), "Invalid CUDA CAGRA device count")
})

test_that("sharded CAGRA plans bounded CUDA indexes for larger file inputs", {
    skip_if_not_installed("Rcpp")
    header <- system.file("include", "faissR_api_v1.h", package = "faissR")
    Rcpp::cppFunction(
        code = 'SEXP select_cagra_shards(SEXP spec) {
          return faissR_get_knn_select_v1()(spec);
        }', includes = paste0('#include "', header, '"'))
    previous <- options(faissR.cuvs_cagra_shard_rows = NULL)
    on.exit(options(previous))
    plan <- function(n, free = 16, file = TRUE, host = NULL, cap = 22) {
        request <- list(backend = "cuda", metric = "euclidean",
            cuda_method = "cagra_sharded", n_reference = n, n_query = n,
            n_features = 96L, k = 30L, target_recall = 0.99, self_query = TRUE,
            file_backed = file, allow_ivf = TRUE, cuda_free_bytes = free * 2^30)
        if (!is.null(host)) request$host_available_bytes <- host * 2^30
        if (!is.null(cap)) request$memory_limit_bytes <- cap * 2^30
        select_cagra_shards(request)
    }
    for (n in c(50000000L, 100000000L)) {
        selected <- plan(n)
        expect_identical(selected$method, "cagra_sharded")
        expect_equal(selected$resources$n_shards, n / 10000000)
        expect_equal(selected$resources$reference_chunk_rows, 10000000)
        expect_lte(selected$resources$device_bytes_estimate, 16 * 2^30)
        expect_equal(selected$resources$output_bytes, 8 * 30 * as.double(n))
    }
    expect_gt(plan(50000000L, free = 8)$resources$n_shards, 5L)
    automatic <- plan(50000000L, host = 22, cap = NULL)$resources
    expect_equal(automatic$n_shards, 5L)
    expect_equal(automatic$memory_limit_bytes, 22 * 2^30)
    expect_gt(plan(50000000L, host = 16, cap = NULL)$resources$n_shards, 5L)
    expect_gt(plan(50000000L, host = 22, cap = 8)$resources$n_shards, 5L)
    expect_error(plan(50000000L, file = FALSE), "file-backed CUDA")
    expect_error(plan(50000000L, free = 0.01), "cannot fit")
    tuned <- options(faissR.cuvs_cagra_file_search =
        list(width = 2L, queue = 192L, batch = 1024L))
    on.exit(options(tuned), add = TRUE)
    selected <- plan(50000000L)
    expect_equal(selected$resources$chunk_rows, 1024)
    expect_equal(selected$resources$n_shards, 5L)
    expect_gt(plan(50000000L, free = 8)$resources$n_shards, 5L)
    options(faissR.cuvs_cagra_file_search =
        list(width = 2L, queue = 192L, batch = 1025L))
    expect_error(plan(50000000L), "between 1 and 1024")
})
