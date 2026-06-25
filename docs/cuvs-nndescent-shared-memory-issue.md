# cuVS NN-Descent Dynamic Shared-Memory Issue Report

[Home](../README.md) |
[Installation](installation.md) |
[Implementation](implementation.md) |
[Examples](examples.md) |
[Benchmarks](benchmarks.md) |
[Autotuning](autotuning.md) |
[API](usage-api.md) |
[NN Methods](nn-methods.md) |
[Backends](backend-capabilities.md) |
[References](references.md)

This page is written as a copy-ready upstream issue report for RAPIDS cuVS.
It is not a faissR workaround and does not imply that faissR vendors cuVS.

## Suggested Issue Title

`cuVS NN-descent FP32 L2 build can fail with cudaErrorInvalidValue for high-dimensional inputs unless the L2-norm kernel opts into larger dynamic shared memory`

## Summary

While testing direct RAPIDS cuVS NN-descent through the cuVS C API, a
high-dimensional FP32 Euclidean/L2 dataset failed during `cuvsNNDescentBuild`
with `cudaErrorInvalidValue: invalid argument`. The failure happened before any
nearest-neighbour result was produced.

The failing shape was COIL20:

```text
n = 1440
dim = 16384
k = 10
graph_degree = 10
intermediate_graph_degree = 20
max_iterations = 1
input dtype = float32
metric = L2Expanded
```

The same build path succeeds after patching cuVS to opt the
`compute_l2_norms_kernel` launch into the larger device-supported dynamic
shared-memory limit with
`cudaFuncSetAttribute(cudaFuncAttributeMaxDynamicSharedMemorySize, ...)`.

## Environment

Observed with:

```text
cuVS/libcuvs: 26.06.00
Source tag used for local rebuild: NVIDIA/cuVS v26.06.00
Source commit: 2bd7cd71d31e39eb9ee00c2a250085a1afb84977
CUDA runtime/toolkit family: CUDA 13.x
API path: cuVS C API, cuvsNNDescentBuild
GPU tested: NVIDIA GPU with cudaDevAttrMaxSharedMemoryPerBlockOptin = 101376
```

The failure is not specific to R. A standalone C/C++ repro using the cuVS C API
with the shape above produced the same cuVS error. faissR only exposed the
error because it calls the cuVS C API directly.

## Actual Error

The unpatched cuVS library failed with:

```text
cuvsNNDescentBuild failed: CUDA error encountered at:
cpp/src/neighbors/detail/nn_descent.cuh line=1564
call='cudaPeekAtLastError()'
Reason=cudaErrorInvalidValue: invalid argument
```

The line number can vary slightly by build, but the stack points to
`neighbors/detail/nn_descent.cuh` immediately after launching the L2-norm kernel.

## Expected Behavior

If the GPU supports enough opt-in dynamic shared memory, cuVS should set the
kernel attribute and launch successfully.

If the GPU cannot support the required dynamic shared memory, cuVS should fail
with a clear error explaining the required bytes and the device limit, rather
than surfacing a raw `cudaErrorInvalidValue` after launch.

## Diagnosis

For FP32 L2 input with `dim = 16384`, the L2-norm kernel dynamic shared-memory
request is:

```text
ceil(dim / warp_size) * warp_size * sizeof(float)
= ceil(16384 / 32) * 32 * 4
= 65536 bytes
```

That is larger than CUDA's usual default dynamic shared-memory launch limit
around 48 KiB per block. The test GPU reports an opt-in maximum of `101376`
bytes, so the launch is legal on the hardware if cuVS opts in before launching
the kernel.

The local patch confirmed that the missing opt-in is the issue.

## Local Patch That Fixed The Reproducer

In `cpp/src/neighbors/detail/nn_descent.cuh`, add a helper equivalent to:

```cpp
template <typename Data_t>
size_t compute_l2_norms_smem_size(size_t dim)
{
  return sizeof(Data_t) *
         raft::ceildiv(dim, static_cast<size_t>(raft::warp_size())) *
         static_cast<size_t>(raft::warp_size());
}

template <typename Data_t>
size_t configure_compute_l2_norms_kernel_smem(size_t dim)
{
  size_t smem = compute_l2_norms_smem_size<Data_t>(dim);

  int dev_id = 0;
  RAFT_CUDA_TRY(cudaGetDevice(&dev_id));

  int max_smem = 0;
  RAFT_CUDA_TRY(cudaDeviceGetAttribute(
    &max_smem,
    cudaDevAttrMaxSharedMemoryPerBlockOptin,
    dev_id
  ));

  RAFT_EXPECTS(
    smem <= static_cast<size_t>(max_smem),
    "NN_DESCENT L2 norm kernel requires %zu bytes of shared memory "
    "(dim=%zu), but the device supports at most %d bytes per block.",
    smem,
    dim,
    max_smem
  );

  auto kernel = compute_l2_norms_kernel<Data_t>;
  RAFT_CUDA_TRY(cudaFuncSetAttribute(
    kernel,
    cudaFuncAttributeMaxDynamicSharedMemorySize,
    static_cast<int>(smem)
  ));

  return smem;
}
```

Then replace the raw dynamic shared-memory expression used for
`compute_l2_norms_kernel<<<...>>>` with:

```cpp
auto smem =
  configure_compute_l2_norms_kernel_smem<float>(build_config_.dataset_dim);
```

or, in the templated direct-data path:

```cpp
auto smem =
  configure_compute_l2_norms_kernel_smem<input_t>(build_config_.dataset_dim);
```

The kernel launch should then use `smem` as the dynamic shared-memory argument.

## Validation After Patch

After rebuilding `libcuvs.so` and `libcuvs_c.so` with the local patch and
preloading the patched libraries before the conda-installed cuVS libraries, the
same calls succeeded:

```text
COIL20 full matrix: 1440 x 16384, k = 10
backend: cuda_cuvs_nndescent
elapsed: 0.696 s
result: OK, indices 1440 x 10, distances 1440 x 10

MNIST70k float32: 70000 x 784, k = 10
backend: cuda_cuvs_nndescent
elapsed: 1.162 s
result: OK, indices 70000 x 10, distances 70000 x 10
```

A standalone cuVS C/C++ reproducer for the COIL20-like shape also returned:

```text
shape=1440x16384 k=10 graph_degree=10 intermediate=20 iters=1
max_optin_smem=101376
OK
```

## Request

Please consider adding the dynamic shared-memory opt-in around the
NN-descent L2-norm kernel launch, or an equivalent implementation that:

1. Computes the dynamic shared-memory requirement from the dataset dimension and
   input type.
2. Queries `cudaDevAttrMaxSharedMemoryPerBlockOptin`.
3. Calls `cudaFuncSetAttribute(..., cudaFuncAttributeMaxDynamicSharedMemorySize, ...)`
   when the requirement exceeds the default limit but is within the opt-in
   device limit.
4. Raises a clear cuVS/RAFT error if the requirement exceeds the hardware limit.

This should make direct cuVS NN-descent reliable for high-dimensional FP32 L2
datasets on GPUs that support the required opt-in dynamic shared memory.
