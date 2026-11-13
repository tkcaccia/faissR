# faissR CUDA multi GPU nearest neighbor implementation handoff

Status: implementation notes and remaining validation for a private C++
entry point in this branch; no multi GPU speed result is claimed. Base source:
faissR `7a22de471eb6d83dd9d1873ef0f89401ea8f11f5`
and fastEmbedR `8c4fb388500a5a48f3708930c021d14cf7c6f33c` on 13 November 2026.
The branch includes `faissR_gpu_multi_knn_v1` in
`inst/include/faissR_api_v1.h`. It uses FAISS GPU `IndexReplicas` or
`IndexShards` and returns host matrices. The private C callable accepts data,
k, method, device ordinals, nlist, nprobe, shard mode, and distance storage.
The document below also records the remaining API and benchmark work.
Do not add Python to either package.

## Existing behavior and gap

`faissR::nn(data, k, exclude_self = TRUE, backend = "cuda",
method = "ivf")` executes the FAISS `GpuIndexIVFFlat` route and returns
host matrices. A generated 200 by 16 check returned
`backend_used = "faiss_gpu_ivf_flat"`, one based indices and Euclidean
distances. `method = "flat"` returned `faiss_gpu_flat_l2`.
`R/nn.R` already dispatches these routes. In `src/nn_faiss_impl.cpp`, both
`faiss_gpu_ivf_flat_knn_impl` and the float32 variant set
`GpuIndexIVFFlatConfig::device = 0`. There is no `devices` argument,
`GpuMultipleClonerOptions`, `IndexReplicas`, or `IndexShards` route in current
faissR. `nn_gpu()` supplies a GPU resident exact family result, but does not
expose GPU resident IVF. This blocks an honest resident 8 GPU embedding KNN.

## Private callable interface

The user requested an internal provider for fastEmbedR, with no new exported
R function. `faissR_gpu_multi_knn_v1` is registered as a private C callable.
It accepts unique visible CUDA device ordinals, Flat or IVF, Euclidean
distance, `nlist`, `nprobe`, and replica or shard mode. It rejects fewer than
two devices, duplicates, and unavailable devices before allocation. Existing
`nn()` behavior is unchanged. The consumer must load the faissR namespace and
use the versioned header in `inst/include/faissR_api_v1.h`. No CPU fallback is
permitted.

The default for multiple devices is `replicate`: each device holds the full
index and FAISS divides query rows among replicas. This is the design most
likely to accelerate the 10M all-row KNN workload when each H200 can hold
the index. `shard` partitions the database among GPUs and merges top-k on
host; it is for capacity and requires a separate quality and time audit.
Do not treat either mode as inherently faster. Record both actual mode and
device ordinals in returned metadata.

## Native implementation

1. Keep the multi GPU implementation alongside the current single GPU
   functions in `src/nn_faiss_impl.cpp`, with a private C callable and no
   `R/nn.R` dispatch. Preserve the existing single GPU ABI.
2. Reuse the existing matrix validation and row-major float32 conversion.
   Construct and train one CPU `faiss::IndexIVFFlat` with the selected
   `nlist` and `nprobe`. Clone the trained empty index, then add database
   rows through FAISS `IndexReplicas` or `IndexShards`; this avoids holding a
   second full CPU index. Compare its output with the current one GPU route
   because training and add policies can change neighbors.
3. Create one `faiss::gpu::StandardGpuResources` per selected device and keep
   them alive until all searches finish. Use FAISS
   `index_cpu_to_gpu_multiple` with `GpuMultipleClonerOptions`; set
   `shard = false` for replicas and `shard = true` for shards. Set
   `shard_type`, `useFloat16`, and `useFloat16CoarseQuantizer` explicitly and
   disclose them. Preserve IVF `nprobe` on the cloned GPU indexes. Check the
   dynamic index type and fail if FAISS substituted an unexpected provider.
4. Call the FAISS composite `search` on bounded batches. Let `IndexReplicas`
   divide query rows; do not add a competing R thread pool. For `IndexShards`,
   let FAISS merge shard candidates with global database IDs. Apply compiled
   self-neighbor removal by requesting enough neighbors and removing only the
   query's own ID. Preserve the current treatment of ties and missing IDs.
5. Return the existing one based integer index matrix and ordinary Euclidean
   (square rooted, not squared) distances. Keep `output = "float"` and
   `"double"` behavior. Reject an output whose dimensions exceed R's matrix
   and index representation limits before allocating. Avoid an unbounded
   duplicate of the full query or output matrix.
6. Use RAII for resources, indexes, buffers, and CUDA contexts. On errors,
   synchronize owned work, release every owned allocation, report the failed
   stage and device, and never reset a device. FAISS resources are not shared
   concurrently across CPU threads. Record process-owned device memory before
   and after in tests; an abnormal test exit is a failed lifecycle check.
7. Add stage timings: input conversion, CPU train, CPU add, GPU clone/transfer,
   query search, merge/self-removal, device-to-host output, and total public
   call. Mark timing scope explicitly. Report peak RSS and per-device peak
   memory when the benchmark harness can measure them; do not invent them in
   package metadata. Include `backend_used`, actual GPU count and ordinals,
   FAISS index type, mode, `nlist`, `nprobe`, input/output residency, recall
   audit status, and exact software versions in result metadata.

Use the FAISS C++ [multiple GPU example](https://github.com/facebookresearch/faiss/blob/main/tutorial/cpp/5-Multiple-GPUs.cpp),
[GPU cloner API](https://github.com/facebookresearch/faiss/blob/main/faiss/gpu/GpuCloner.h),
and [GPU guidance](https://github.com/facebookresearch/faiss/wiki/Faiss-on-the-GPU).
Pin the FAISS version used by faissR and compile against its exact C++ API.

## fastEmbedR integration after the faissR API lands

The fastEmbedR branch uses `faissR::nn()` for one CUDA GPU and the private C
callable for multiple GPUs. It passes the caller's device vector through the
embedding KNN path in `tsne()` and `umap()`. It validates the returned provider,
dimensions, one based indices, distances, device ordinals, and no CPU fallback.
Keep the returned KNN on host until faissR
offers an owning multi GPU resident output; report that host boundary in
fastEmbedR result metadata and time. If the current optimizer requires GPU
resident input, fail clearly rather than passing incompatible pointers.

The existing `fastembedr_gpu_knn_to_host()` refuses external GPU objects;
do not reinterpret faissR pointers as fastEmbedR native pointers. An eventual
resident bridge must use faissR's public C callable ABI with explicit buffer
ownership, per-device handles and synchronization. Preserve CPU and Metal
routes, and do not route file-backed data through `nn()`'s dense input API.
Remove the replaced native CUDA KNN path only when every public caller and
test has migrated; avoid maintaining two hidden CUDA providers.

## Required tests and acceptance

- Single selected nonzero GPU executes on that GPU; two and eight device
  requests use exactly those devices. An unavailable device or CPU only build
  errors before work. One device result matches the existing one device
  FAISS route for fixed training and tuning.
- Small generated data: compare flat to exact CPU indices and distances;
  compare IVF replicas and shards against one GPU with fixed seed; verify
  self exclusion, duplicates/ties, `k` limits, float32/double, batch tails,
  and 1 based IDs. Test zero rows before adding cosine/correlation.
- A source locked 10M Deep1B run at fixed `k`, preprocessing, seed and recall
  target reports all attempted 1/2/4/8 GPU cells, failed cells, full call
  and stage times, per-device memory, RSS, exact-reference neighbor recall
  on a fixed query sample, and embedding quality. Use identical device type
  and complete output scope. Replicas may fail memory admission; shards may
  differ in quality and must be labeled separately.
- Performance acceptance is measured, not assumed: eight GPUs must reduce
  the prespecified end-to-end KNN time versus one GPU while meeting the same
  recall threshold, with repeated clean runs. If only query search improves
  but train/clone or embedding dominates, report the negative end-to-end
  result and profile the stage rather than advertising a speedup.
- Build one source archive and verify its SHA256 across platforms using
  `docs/package-testing.md`. CPU tests cannot validate CUDA. Run package
  checks, a CUDA test, and cleanup checks before fastEmbedR integration.

No paid or HPC GPU work is authorized by this handoff. Observe the project's
GPU cleanup and spending gates before scheduling benchmarks.
