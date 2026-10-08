#' Native C-callable nearest-neighbour API
#'
#' Downstream compiled R packages can retrieve faissR's registered C-callable
#' entry points by including `faissR_api.h` from the installed `include`
#' directory. The header contains the function-pointer types and inline lookup
#' helpers; these entry points are not called directly from R.
#'
#' @section ABI version:
#' Call `faissR_c_api_version()` before retrieving another entry point. The
#' current ABI version is 1. A downstream package should reject an unsupported
#' ABI version instead of assuming that signatures or object ownership remain
#' compatible.
#'
#' @section Registered callables:
#' `faissR_get_nn_float32()` retrieves the host-result self-KNN callable.
#' `faissR_get_nn_float32_output()` additionally accepts a distance-storage
#' selector. `faissR_get_nn_cuda_tuned_gpu()` retrieves the CUDA-resident
#' self-KNN callable for `"auto"`, `"exact"`, `"flat"`, and `"bruteforce"`.
#' `faissR_get_hnsw_index_build_v1()` builds an owned CPU FAISS HNSW index,
#' and `faissR_get_hnsw_index_search_v1()` queries that index repeatedly.
#' `faissR_get_hnsw_tune_v1()` returns the deterministic shape-based HNSW
#' settings without building an index. This policy lookup is distinct from
#' the opt-in, data-aware [hnsw_tune()] workflow.
#' The exact version-1 signatures are declared in `faissR_api_v1.h`;
#' `faissR_api.h` includes the current versioned header for source
#' compatibility.
#'
#' @section Persistent HNSW ownership:
#' The HNSW build callable accepts a float32 reference matrix plus explicit
#' `M`, `efConstruction`, and thread settings. Its external pointer owns the
#' native index and must remain reachable until all searches finish. The
#' search callable accepts explicit `k`, `efSearch`, and thread settings and
#' returns one-based identifiers, Euclidean distances, requested/effective
#' settings, and a cumulative query-call count. The pointer is neither
#' serializable nor safe for concurrent search calls. Its registered
#' finalizer releases the index when the pointer is garbage-collected.
#'
#' @section Self-neighbour polarity and indices:
#' The native API uses `include_self`: `TRUE` retains each query row in its own
#' self-search result and `FALSE` removes it. This is the opposite polarity of
#' the R-level `exclude_self` argument. Host results contain one-based R row
#' identifiers.
#'
#' @section GPU ownership:
#' A GPU-resident result owns its device allocations through an external
#' pointer. The owner object must remain reachable while a downstream consumer
#' uses either child device pointer. Consumers must not free those pointers.
#' Convert explicitly with [gpu_knn_to_host()] when host matrices are needed.
#'
#' @return The header accessor `faissR_c_api_version()` returns an integer ABI
#'     version. The `faissR_get_*()` accessors return typed C function pointers;
#'     the retrieved nearest-neighbour callables return R `SEXP` objects.
#'     Callers must protect returned objects before further R allocations
#'     (for example, with `PROTECT()` or `Rcpp::Shield<SEXP>`). Their structure
#'     and ownership rules are defined in
#'     `faissR_api.h`. This documentation topic does not create an R object.
#'
#' @name faissR-c-api
#' @keywords programming
NULL
