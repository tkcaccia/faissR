# Private native KNN entry points for fastEmbedR

This branch adds C callable entry points to `inst/include/faissR_api_v1.h`.
They are registered by `register_faissR_ccallables()` when faissR loads.
They do not extend `nn()`'s public backend choices. Consumer packages must
include the header, call `loadNamespace("faissR")`, check
`faissR_c_api_version() == 1`, and retain the faissR namespace while using
the callables. No Python code is used.

| Getter | Operation | Result |
| --- | --- | --- |
| `faissR_get_gpu_multi_knn_v1()` | FAISS GPU Flat or IVF-Flat self KNN on one or more distinct CUDA ordinals; `shard = FALSE` replicates the index, `TRUE` shards it for multiple devices | Host R list: one-based indices, ordinary Euclidean distances, device IDs, mode, FAISS backend |
| `faissR_get_gpu_multi_query_knn_v1()` | Search a fixed query matrix against the full multi GPU index, without self exclusion | Host R list with query count, stage times, device IDs, and one-based neighbors |
| `faissR_get_metal_knn_v1()` | Native Metal exact or IVF self KNN | Host R list with indices, distances, method and recall metadata |
| `faissR_get_metal_query_knn_v1()` | Native Metal fixed-reference query KNN | Host R list with indices, distances and method metadata |

The CUDA function currently accepts Euclidean data only and an explicit
`nlist`/`nprobe` for IVF. The consumer must use a frozen tuning policy and
audit recall. The fixed-query entry point permits bounded exact-reference
recall sweeps at 10–100 million database rows without allocating an all-row
KNN result for every candidate. GPU resources belong to the call and are
destroyed on return.

The private R helpers `nn_cuda_ivf_large_candidates()` and
`nn_cuda_ivf_large_extrapolate()` generate a bounded parameter grid and a
power-law projection from passing, source-matched 10M and 100M observations.
The extrapolator requires exact neighbors from the full database, records
one timing scope and query count, records device-memory admission, and leaves
projected recall unknown. It does not
turn an extrapolation into a verified billion-row capacity claim.
There is no persistent multi GPU output handle yet. The implementation has
compiled on CUDA 13/FAISS 1.14.3, but an actual two or eight GPU correctness
and speed run is still required before production use.

The Metal implementation was ported from fastEmbedR with its MIT and
Apache-2.0 notices and has a non-Mac stub. `faissR::nn(backend = "metal")`
continues to reject Metal; fastEmbedR uses the private callable. The same
source archive
built on Linux and an arm64 Mac. On generated 100-by-8 data, Metal exact
KNN matched CPU indices and had maximum distance error below `1e-6`.
On generated 1,000-by-32 data, Metal IVF had 0.9918 mean neighbor recall
against CPU exact at requested 0.99. The focused Mac test file passed
9 assertions. These small checks do not establish large-data performance.

For fastEmbedR, the intended transition is to call these entry points in its
KNN dispatch layer. Verify the returned backend, one-based IDs, dimensions,
distance semantics, recall status, and host residency. Do not reinterpret a
faissR host result as a fastEmbedR GPU resident pointer.
