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

static inline faissR_hnsw_search_v1_fun faissR_get_hnsw_search_v1(void) {
  return (faissR_hnsw_search_v1_fun) R_GetCCallable(
      "faissR", "faissR_hnsw_search_v1");
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

#ifdef __cplusplus
}
#endif

#endif
