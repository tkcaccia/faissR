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
 * With one GPU, `distance_storage = "gpu"` returns an owned device result.
 * `"gpu_self_exclude"` additionally removes each query's same-row ID and
 * requires the query rows to correspond exactly to the indexed rows.
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

typedef SEXP (*faissR_hnsw_index_search_v1_fun)(
    SEXP index,
    SEXP query,
    SEXP k,
    SEXP ef_search,
    SEXP n_threads);

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

static inline faissR_gpu_multi_knn_v1_fun
faissR_get_gpu_multi_knn_v1(void) {
  return (faissR_gpu_multi_knn_v1_fun) R_GetCCallable(
      "faissR", "faissR_gpu_multi_knn_v1");
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

static inline faissR_gpu_knn_sample_v1_fun
faissR_get_gpu_knn_sample_v1(void) {
  return (faissR_gpu_knn_sample_v1_fun) R_GetCCallable(
      "faissR", "faissR_gpu_knn_sample_v1");
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

static inline faissR_hnsw_tune_v1_fun faissR_get_hnsw_tune_v1(void) {
  return (faissR_hnsw_tune_v1_fun) R_GetCCallable(
      "faissR", "faissR_hnsw_tune_v1");
}

static inline faissR_hnsw_index_build_v1_fun
faissR_get_hnsw_index_build_v1(void) {
  return (faissR_hnsw_index_build_v1_fun) R_GetCCallable(
      "faissR", "faissR_hnsw_index_build_v1");
}

static inline faissR_hnsw_index_search_v1_fun
faissR_get_hnsw_index_search_v1(void) {
  return (faissR_hnsw_index_search_v1_fun) R_GetCCallable(
      "faissR", "faissR_hnsw_index_search_v1");
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

#ifdef __cplusplus
}
#endif

#endif
