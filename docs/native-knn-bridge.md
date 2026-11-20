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

Persistent single-device Euclidean IVF-Flat trains its coarse quantizer on
CUDA when FAISS is built without cuVS. Build metadata separates CUDA
training, CPU training, cloning, and index insertion. It retains float32
vectors and the caller's list and probe counts.

The private R helpers `nn_cuda_ivf_large_candidates()` and
`nn_cuda_ivf_large_extrapolate()` generate a bounded parameter grid and a
power-law projection from passing, source-matched 10M and 100M observations.
The extrapolator requires exact neighbors from the full database, records
one timing scope and query count, records device-memory admission, and leaves
projected recall unknown. Supply `fit_sizes` to declare other measured
anchors for a matched dataset and GPU configuration. Supply `holdout_n`
to leave an intermediate size out of the fit and check the projected
list and probe counts against the exact audited setting at that size.
The check reports recall misses, missing settings, and relative errors
for index build, fixed-query search, host RSS, and device memory.
A passing intermediate check supports interpolation at that size.
The larger projected setting still requires a live recall and timing audit.
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

## Route-specific size planning

`nn_large_knn_size_tune()` is a private offline tuner for Euclidean CPU
HNSW and CUDA or Metal IVF. Calibrate Linux and macOS CPU separately.
Give each source, hardware, thread count, index layout, input family, and
metric its own profile. Use different profiles for resident and file-backed
runs. A file-backed route with a resident index also needs a distinct
profile from a route that builds and searches one reference shard at a time.

Use nested reference selections at 10M and 100M, and keep a size such as 50M
out of the fit. Run a separate tuning query set and a disjoint audit query
set against exact neighbors in the full calibrated index. Record the
measured mean and worst-row recall, stages, peak host and device bytes,
and all attempted settings, including failures. Keep the query count and
timing scope fixed across the series. Estimate parameters separately on
narrow, medium, and wide inputs. Array dimensions alone cannot establish
that two data distributions have the same recall behavior.

Supply the following columns in `observations`:

- `n`, `p`, `k`, `backend`, `storage_mode`, `gpu_count`, `layout`,
  and `profile_sha256`. Use the SHA-256 of the source and experiment
  identity manifest as the profile identifier. Use `single_logical_index`
  for one index across the requested devices or `reference_shard`
  to model a file-backed route per reference shard.
- `tuning_rows_sha256`, `audit_rows_sha256`, `query_count`,
  `reference_scope = "full_database_exact"`, and
  `timing_scope = "index_build_plus_fixed_query"`. Rows and timing
  scope refer to the same fixed-query design at each size.
- `audit_pass`, `mean_recall`, `min_row_recall`, `build_seconds`,
  `search_seconds`, `peak_host_bytes`, and `peak_device_bytes` on GPU.
- CPU settings `m`, `ef_construction`, and `ef_search`, or GPU settings
  `nlist` and `nprobe`.

The tuner ranks passing anchor settings by build time plus the requested
number of identical fixed-query batches. The default is one batch. It
requires the predicted setting to pass the held-out size, including the
recall and timing tolerances, before it reports settings beyond the
largest anchor. With 100M as the largest anchor, the extrapolation range
ends at 1B. With `reference_shard_rows`, the settings and timing apply to
one reference shard. The planner reports host and device fit for a fixed
query batch separately from the disk and output checks for a full graph.

The projected settings still require live recall calibration. The tuner
leaves target-size recall and full self-search time unknown. Metal IVF
settings respect its 1,024-list and 512-probe limits. For CUDA, supply
the provider's probe limit if it differs from the default of 2,048.
The self-search callables still select settings internally. The private
planner does not add public API or change runtime defaults.
