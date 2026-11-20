/*
 * Copyright (c) 2026 Stefano Cacciatore
 * SPDX-License-Identifier: MIT
 */

#ifndef FAISSR_API_V1_H
#define FAISSR_API_V1_H

#include <R_ext/Rdynload.h>
#include <Rinternals.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef int (*faissR_c_api_version_fun)(void);

/* Choose a KNN algorithm for a specified shape, recall tier, and storage. */
typedef SEXP (*faissR_knn_select_v1_fun)(SEXP specification);


/* Exact double-precision kth non-self-neighbor radius of an n-by-2 layout. */
typedef SEXP (*faissR_layout_radii_v1_fun)(SEXP layout, SEXP k);

/*
 * The include_self arguments use the opposite polarity from the R-level
 * exclude_self argument: TRUE retains each query row in self-search output.
 * Host callables return one-based R indices.
 */
typedef SEXP (*faissR_nn_float32_fun)(
    SEXP x,
    SEXP k,
    SEXP backend,
    SEXP metric,
    SEXP include_self,
    SEXP n_threads);

typedef SEXP (*faissR_nn_float32_output_fun)(
    SEXP x,
    SEXP k,
    SEXP backend,
    SEXP metric,
    SEXP include_self,
    SEXP n_threads,
    SEXP distances);

typedef SEXP (*faissR_nn_cuda_tuned_gpu_fun)(
    SEXP x,
    SEXP k,
    SEXP method,
    SEXP metric,
    SEXP include_self,
    SEXP target_recall);

/*
 * Private multi-GPU FAISS self-KNN entry point. The host result contains
 * one-based indices, Euclidean distances, actual device IDs and mode.
 * `devices` contains one or more distinct visible CUDA ordinals. The
 * one-device route provides a source-matched scaling baseline.
 * `distance_storage = "gpu"` privately returns an owned CUDA result for
 * one device, with the query row excluded from its own neighbours. Multi-GPU
 * results currently require host output (`"float"` or `"double"`).
 */
typedef SEXP (*faissR_gpu_multi_knn_v1_fun)(
    SEXP data,
    SEXP k,
    SEXP method,
    SEXP devices,
    SEXP nlist,
    SEXP nprobe,
    SEXP shard,
    SEXP distance_storage);

/* Private one-GPU cosine/correlation IVF result retained on the device. */
typedef SEXP (*faissR_gpu_metric_ivf_knn_v1_fun)(
    SEXP data,
    SEXP k,
    SEXP metric,
    SEXP devices,
    SEXP nlist,
    SEXP nprobe);

typedef SEXP (*faissR_gpu_knn_retain_data_v1_fun)(
    SEXP result, SEXP data);

typedef SEXP (*faissR_gpu_multi_query_knn_v1_fun)(
    SEXP data,
    SEXP points,
    SEXP k,
    SEXP method,
    SEXP devices,
    SEXP nlist,
    SEXP nprobe,
    SEXP shard,
    SEXP distance_storage);

/*
 * Private persistent CUDA index for bounded query batches. The build result
 * owns the index; keep its `pointer` member reachable for every search. The
 * pointer is process-local, nonserializable, and not safe for concurrent calls.
 * Search accepts float32 matrices and returns one-based IDs on the host.
 * A search may request any positive k up to the index build's k limit;
 * this permits a k+1 pilot followed by k-neighbor self output on one index.
 * With one GPU, `distance_storage = "gpu"` returns an owned device result.
 * `"gpu_self_exclude"` additionally removes each query's same-row ID and
 * requires the query rows to correspond exactly to the indexed rows.
 * `"double_self_exclude"` and `"float_self_exclude"` produce host results
 * with the same identity rule. Build with capacity k+1 for these searches.
 */
typedef SEXP (*faissR_gpu_index_build_v1_fun)(
    SEXP data,
    SEXP k,
    SEXP method,
    SEXP devices,
    SEXP nlist,
    SEXP nprobe,
    SEXP shard);

typedef SEXP (*faissR_gpu_index_search_v1_fun)(
    SEXP index,
    SEXP points,
    SEXP k,
    SEXP distance_storage);

/* Change IVF search width on every resident GPU member without rebuilding. */
typedef SEXP (*faissR_gpu_index_set_nprobe_v1_fun)(
    SEXP index,
    SEXP nprobe);

/* Copy at most 256 selected one-based query rows from an owned CUDA result. */
typedef SEXP (*faissR_gpu_knn_sample_v1_fun)(SEXP result, SEXP rows);

typedef SEXP (*faissR_metal_knn_v1_fun)(
    SEXP data,
    SEXP k,
    SEXP method,
    SEXP metric,
    SEXP target_recall);

typedef SEXP (*faissR_metal_query_knn_v1_fun)(
    SEXP data,
    SEXP query,
    SEXP k,
    SEXP method,
    SEXP metric,
    SEXP target_recall);

typedef SEXP (*faissR_hnsw_search_v1_fun)(
    SEXP data,
    SEXP query,
    SEXP n,
    SEXP p,
    SEXP k,
    SEXP target_recall,
    SEXP n_threads,
    SEXP distance_storage);

/* Full-reference pilot audit and adaptive efSearch; same argument ABI. */
typedef faissR_hnsw_search_v1_fun faissR_hnsw_search_v2_fun;

/*
 * Return faissR's deterministic shape-based CPU HNSW policy without
 * constructing an index. The result contains M, efConstruction, efSearch,
 * and the rule and benchmark provenance used to choose them.
 */
typedef SEXP (*faissR_hnsw_tune_v1_fun)(
    SEXP n,
    SEXP p,
    SEXP k,
    SEXP metric,
    SEXP target_recall);

/*
 * Build and query inputs must be float::fl()/float32 matrices. The build
 * result owns a CPU FAISS HNSW index. Keep the external pointer reachable for
 * every search. It is not serializable or safe for concurrent calls. Search
 * returns one-based IDs, Euclidean distances, requested settings, and the
 * effective M, efConstruction, efSearch, and thread settings.
 */
typedef SEXP (*faissR_hnsw_index_build_v1_fun)(
    SEXP data,
    SEXP m,
    SEXP ef_construction,
    SEXP n_threads);

/* Build a resident CPU HNSW index by streaming f32/fbin reference chunks.
 * The reference descriptor may be a bounded view of a file descriptor.
 * The returned pointer is compatible with hnsw_index_search_v1 and must
 * remain owned by the caller. This does not stream the index itself. */
typedef SEXP (*faissR_hnsw_file_index_build_v1_fun)(
    SEXP file_spec, SEXP m, SEXP ef_construction,
    SEXP n_threads, SEXP chunk_rows);

typedef SEXP (*faissR_hnsw_index_search_v1_fun)(
    SEXP index,
    SEXP query,
    SEXP k,
    SEXP ef_search,
    SEXP n_threads);

/*
 * Save/load a fitted CPU HNSW index as a native FAISS file. The caller must
 * save to a temporary path, verify size/hash, and atomically publish it with
 * its source/settings identity. Verify that identity and hash before load.
 * These calls do not supply a checkpoint protocol or accept untrusted files.
 */
typedef SEXP (*faissR_hnsw_index_save_v1_fun)(SEXP index, SEXP path);
typedef SEXP (*faissR_hnsw_index_load_v1_fun)(SEXP path, SEXP n_threads);

/*
 * Calibrate a fitted CPU HNSW index against its full stored reference.
 * self_ids are one-based reference rows (0 when the query is not a reference
 * row). The float32 query and self_ids vectors must have equal row counts.
 */
typedef SEXP (*faissR_hnsw_index_calibrate_v1_fun)(
    SEXP index,
    SEXP query,
    SEXP self_ids,
    SEXP k,
    SEXP target_recall,
    SEXP initial_ef_search,
    SEXP n_threads);

/* Version 2 adds an explicit maximum efSearch for larger HNSW pilots. */
typedef SEXP (*faissR_hnsw_index_calibrate_v2_fun)(
    SEXP index,
    SEXP query,
    SEXP self_ids,
    SEXP k,
    SEXP target_recall,
    SEXP initial_ef_search,
    SEXP max_ef_search,
    SEXP n_threads);

/* Version 3 uses double-precision exact L2 on stored float32 values. */
typedef faissR_hnsw_index_calibrate_v2_fun
    faissR_hnsw_index_calibrate_v3_fun;

/* Private persistent CPU Flat index for bounded query batches. */
typedef SEXP (*faissR_flat_index_build_v1_fun)(
    SEXP data,
    SEXP n_threads);

typedef SEXP (*faissR_flat_index_search_v1_fun)(
    SEXP index,
    SEXP query,
    SEXP k,
    SEXP n_threads);

/*
 * Private bounded matrix KNN route; no faissR R-level export. CPU, Metal,
 * and CUDA support exact streamed reference batches. CPU search computes
 * distances in double precision from float32 inputs and breaks ties by
 * one-based reference ID. Metal uses the native
 * Metal query kernel on one GPU and requires n_threads = 1 and k <= 128
 * (k <= 127 when excluding self). CUDA uses FAISS GPU Flat on the current
 * CUDA device and requires n_threads = 1 and k <= 256 (255 excluding self).
 * Multi-GPU file-backed search is a separate capability, not this v1 call.
 */
typedef SEXP (*faissR_matrix_knn_batch_v1_fun)(
    SEXP query,
    SEXP reference,
    SEXP row_ids,
    SEXP k,
    SEXP backend,
    SEXP method,
    SEXP reference_chunk,
    SEXP memory_limit,
    SEXP n_threads,
    SEXP exclude_self);

/*
 * v2 selects explicit CUDA device IDs. One device uses GpuIndexFlatL2;
 * multiple devices require FAISS IndexShards backed by the selected GPUs.
 * CPU and Metal callers use v1. A missing multi-GPU capability fails.
 */
typedef SEXP (*faissR_matrix_knn_batch_v2_fun)(
    SEXP query,
    SEXP reference,
    SEXP row_ids,
    SEXP k,
    SEXP backend,
    SEXP method,
    SEXP reference_chunk,
    SEXP memory_limit,
    SEXP n_threads,
    SEXP exclude_self,
    SEXP devices);

/* One source read and one native Metal IVF build/search. The full graph is
 * returned in host matrices only when the conservative RAM and Metal
 * working-set budgets admit it. The input remains float32 throughout. */
/* Private Metal Flat reference shard held in shared float32 storage.
 * Search receives a bounded file-backed query view and returns shard-local
 * one-based IDs. The caller releases the handle before building a new shard.
 */
typedef SEXP (*faissR_metal_file_flat_build_v1_fun)(
    SEXP reference, SEXP query_capacity, SEXP k, SEXP memory_limit);
typedef SEXP (*faissR_metal_file_flat_search_v1_fun)(
    SEXP pointer, SEXP query, SEXP k);

typedef SEXP (*faissR_metal_file_ivf_v1_fun)(
    SEXP source_spec,
    SEXP k,
    SEXP target_recall,
    SEXP memory_limit);
typedef SEXP (*faissR_metal_file_ivf_plan_v1_fun)(
    SEXP source_spec,
    SEXP k,
    SEXP memory_limit);

/* Private CPU search over validated physical posting files. */
typedef SEXP (*faissR_posting_search_v1_fun)(
    SEXP query, SEXP feature_path, SEXP ids_path, SEXP offsets_path,
    SEXP centers, SEXP first_row, SEXP count, SEXP k, SEXP nprobe,
    SEXP reference_chunk, SEXP n_threads, SEXP exclude_self,
    SEXP posting_rows, SEXP verbose);

/* Private CPU selection of rows farthest from their assigned posting. */
typedef SEXP (*faissR_posting_distant_rows_v1_fun)(SEXP arguments);

/* Private file-backed construction of physical posting lists. */
typedef SEXP (*faissR_posting_build_v1_fun)(SEXP arguments);

/* Private bounded reader for file-backed KNN result rows. */
typedef SEXP (*faissR_knn_rows_read_v1_fun)(SEXP arguments);

/* Private reorder of posting-grouped KNN files into source-row order. */
typedef SEXP (*faissR_reorder_posting_graph_v1_fun)(SEXP arguments);

/*
 * Private file-backed KNN shard merge. `arguments` is a named list with
 * paths, one-based shard/query ranges, neighbor matrices, dimensions,
 * and self-exclusion controls. Returns provider and method identity.
 */
typedef SEXP (*faissR_merge_knn_shard_v1_fun)(SEXP arguments);

/*
 * The GPU-resident callable accepts method values "auto", "exact", "flat",
 * and "bruteforce". The returned owner object must remain reachable while
 * any child device pointer is in use.
 */

static inline int faissR_c_api_version(void) {
  faissR_c_api_version_fun fn =
      (faissR_c_api_version_fun) R_GetCCallable(
          "faissR", "faissR_c_api_version");
  return fn();
}

static inline faissR_nn_float32_fun faissR_get_nn_float32(void) {
  return (faissR_nn_float32_fun) R_GetCCallable(
      "faissR", "faissR_nn_float32_call");
}

static inline faissR_nn_float32_output_fun
faissR_get_nn_float32_output(void) {
  return (faissR_nn_float32_output_fun) R_GetCCallable(
      "faissR", "faissR_nn_float32_call_output");
}

static inline faissR_nn_cuda_tuned_gpu_fun
faissR_get_nn_cuda_tuned_gpu(void) {
  return (faissR_nn_cuda_tuned_gpu_fun) R_GetCCallable(
      "faissR", "faissR_nn_cuda_tuned_gpu_call");
}

static inline faissR_posting_search_v1_fun
faissR_get_posting_search_v1(void) {
  return (faissR_posting_search_v1_fun) R_GetCCallable(
      "faissR", "faissR_posting_search_v1");
}

static inline faissR_posting_distant_rows_v1_fun
faissR_get_posting_distant_rows_v1(void) {
  return (faissR_posting_distant_rows_v1_fun) R_GetCCallable(
      "faissR", "faissR_posting_distant_rows_v1");
}

static inline faissR_posting_build_v1_fun
faissR_get_posting_build_v1(void) {
  return (faissR_posting_build_v1_fun) R_GetCCallable(
      "faissR", "faissR_posting_build_v1");
}

static inline faissR_knn_rows_read_v1_fun
faissR_get_knn_rows_read_v1(void) {
  return (faissR_knn_rows_read_v1_fun) R_GetCCallable(
      "faissR", "faissR_knn_rows_read_v1");
}

static inline faissR_reorder_posting_graph_v1_fun
faissR_get_reorder_posting_graph_v1(void) {
  return (faissR_reorder_posting_graph_v1_fun) R_GetCCallable(
      "faissR", "faissR_reorder_posting_graph_v1");
}

static inline faissR_merge_knn_shard_v1_fun
faissR_get_merge_knn_shard_v1(void) {
  return (faissR_merge_knn_shard_v1_fun) R_GetCCallable(
      "faissR", "faissR_merge_knn_shard_v1");
}

static inline faissR_gpu_multi_knn_v1_fun
faissR_get_gpu_multi_knn_v1(void) {
  return (faissR_gpu_multi_knn_v1_fun) R_GetCCallable(
      "faissR", "faissR_gpu_multi_knn_v1");
}

static inline faissR_gpu_metric_ivf_knn_v1_fun
faissR_get_gpu_metric_ivf_knn_v1(void) {
  return (faissR_gpu_metric_ivf_knn_v1_fun) R_GetCCallable(
      "faissR", "faissR_gpu_metric_ivf_knn_v1");
}

static inline faissR_gpu_knn_retain_data_v1_fun
faissR_get_gpu_knn_retain_data_v1(void) {
  return (faissR_gpu_knn_retain_data_v1_fun) R_GetCCallable(
      "faissR", "faissR_gpu_knn_retain_data_v1");
}

static inline faissR_gpu_multi_query_knn_v1_fun
faissR_get_gpu_multi_query_knn_v1(void) {
  return (faissR_gpu_multi_query_knn_v1_fun) R_GetCCallable(
      "faissR", "faissR_gpu_multi_query_knn_v1");
}

static inline faissR_gpu_index_build_v1_fun
faissR_get_gpu_index_build_v1(void) {
  return (faissR_gpu_index_build_v1_fun) R_GetCCallable(
      "faissR", "faissR_gpu_index_build_v1");
}

static inline faissR_gpu_index_search_v1_fun
faissR_get_gpu_index_search_v1(void) {
  return (faissR_gpu_index_search_v1_fun) R_GetCCallable(
      "faissR", "faissR_gpu_index_search_v1");
}

static inline faissR_gpu_index_set_nprobe_v1_fun
faissR_get_gpu_index_set_nprobe_v1(void) {
  return (faissR_gpu_index_set_nprobe_v1_fun) R_GetCCallable(
      "faissR", "faissR_gpu_index_set_nprobe_v1");
}

static inline faissR_gpu_knn_sample_v1_fun
faissR_get_gpu_knn_sample_v1(void) {
  return (faissR_gpu_knn_sample_v1_fun) R_GetCCallable(
      "faissR", "faissR_gpu_knn_sample_v1");
}

static inline faissR_knn_select_v1_fun
faissR_get_knn_select_v1(void) {
  return (faissR_knn_select_v1_fun) R_GetCCallable(
      "faissR", "faissR_knn_select_v1");
}

static inline faissR_metal_knn_v1_fun
faissR_get_metal_knn_v1(void) {
  return (faissR_metal_knn_v1_fun) R_GetCCallable(
      "faissR", "faissR_metal_knn_v1");
}

static inline faissR_metal_query_knn_v1_fun
faissR_get_metal_query_knn_v1(void) {
  return (faissR_metal_query_knn_v1_fun) R_GetCCallable(
      "faissR", "faissR_metal_query_knn_v1");
}

static inline faissR_hnsw_search_v1_fun faissR_get_hnsw_search_v1(void) {
  return (faissR_hnsw_search_v1_fun) R_GetCCallable(
      "faissR", "faissR_hnsw_search_v1");
}

static inline faissR_hnsw_search_v2_fun faissR_get_hnsw_search_v2(void) {
  return (faissR_hnsw_search_v2_fun) R_GetCCallable(
      "faissR", "faissR_hnsw_search_v2");
}

static inline faissR_hnsw_tune_v1_fun faissR_get_hnsw_tune_v1(void) {
  return (faissR_hnsw_tune_v1_fun) R_GetCCallable(
      "faissR", "faissR_hnsw_tune_v1");
}

static inline faissR_hnsw_index_build_v1_fun
faissR_get_hnsw_index_build_v1(void) {
  return (faissR_hnsw_index_build_v1_fun) R_GetCCallable(
      "faissR", "faissR_hnsw_index_build_v1");
}

static inline faissR_hnsw_file_index_build_v1_fun
faissR_get_hnsw_file_index_build_v1(void) {
  return (faissR_hnsw_file_index_build_v1_fun) R_GetCCallable(
      "faissR", "faissR_hnsw_file_index_build_v1");
}

static inline faissR_hnsw_index_search_v1_fun
faissR_get_hnsw_index_search_v1(void) {
  return (faissR_hnsw_index_search_v1_fun) R_GetCCallable(
      "faissR", "faissR_hnsw_index_search_v1");
}

static inline faissR_hnsw_index_save_v1_fun
faissR_get_hnsw_index_save_v1(void) {
  return (faissR_hnsw_index_save_v1_fun) R_GetCCallable(
      "faissR", "faissR_hnsw_index_save_v1");
}

static inline faissR_hnsw_index_load_v1_fun
faissR_get_hnsw_index_load_v1(void) {
  return (faissR_hnsw_index_load_v1_fun) R_GetCCallable(
      "faissR", "faissR_hnsw_index_load_v1");
}

static inline faissR_hnsw_index_calibrate_v1_fun
faissR_get_hnsw_index_calibrate_v1(void) {
  return (faissR_hnsw_index_calibrate_v1_fun) R_GetCCallable(
      "faissR", "faissR_hnsw_index_calibrate_v1");
}

static inline faissR_hnsw_index_calibrate_v2_fun
faissR_get_hnsw_index_calibrate_v2(void) {
  return (faissR_hnsw_index_calibrate_v2_fun) R_GetCCallable(
      "faissR", "faissR_hnsw_index_calibrate_v2");
}

static inline faissR_hnsw_index_calibrate_v3_fun
faissR_get_hnsw_index_calibrate_v3(void) {
  return (faissR_hnsw_index_calibrate_v3_fun) R_GetCCallable(
      "faissR", "faissR_hnsw_index_calibrate_v3");
}

static inline faissR_flat_index_build_v1_fun
faissR_get_flat_index_build_v1(void) {
  return (faissR_flat_index_build_v1_fun) R_GetCCallable(
      "faissR", "faissR_flat_index_build_v1");
}

static inline faissR_flat_index_search_v1_fun
faissR_get_flat_index_search_v1(void) {
  return (faissR_flat_index_search_v1_fun) R_GetCCallable(
      "faissR", "faissR_flat_index_search_v1");
}

static inline faissR_matrix_knn_batch_v1_fun
faissR_get_matrix_knn_batch_v1(void) {
  return (faissR_matrix_knn_batch_v1_fun) R_GetCCallable(
      "faissR", "faissR_matrix_knn_batch_v1");
}

static inline faissR_matrix_knn_batch_v2_fun
faissR_get_matrix_knn_batch_v2(void) {
  return (faissR_matrix_knn_batch_v2_fun) R_GetCCallable(
      "faissR", "faissR_matrix_knn_batch_v2");
}

static inline faissR_metal_file_flat_build_v1_fun
faissR_get_metal_file_flat_build_v1(void) {
  return (faissR_metal_file_flat_build_v1_fun) R_GetCCallable(
      "faissR", "faissR_metal_file_flat_build_v1");
}

static inline faissR_metal_file_flat_search_v1_fun
faissR_get_metal_file_flat_search_v1(void) {
  return (faissR_metal_file_flat_search_v1_fun) R_GetCCallable(
      "faissR", "faissR_metal_file_flat_search_v1");
}

static inline faissR_metal_file_ivf_v1_fun
faissR_get_metal_file_ivf_v1(void) {
  return (faissR_metal_file_ivf_v1_fun) R_GetCCallable(
      "faissR", "faissR_metal_file_ivf_v1");
}

static inline faissR_metal_file_ivf_plan_v1_fun
faissR_get_metal_file_ivf_plan_v1(void) {
  return (faissR_metal_file_ivf_plan_v1_fun) R_GetCCallable(
      "faissR", "faissR_metal_file_ivf_plan_v1");
}

static inline faissR_layout_radii_v1_fun
faissR_get_layout_radii_v1(void) {
  return (faissR_layout_radii_v1_fun) R_GetCCallable(
      "faissR", "faissR_layout_radii_v1");
}

#ifdef __cplusplus
}
#endif

#endif
