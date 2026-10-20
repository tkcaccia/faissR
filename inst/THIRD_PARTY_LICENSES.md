# External native libraries

`faissR` is distributed under the Expat License. R package metadata identifies
this standard template as `MIT + file LICENSE`. It includes the FAISS 1.11.0
source archive for a CPU-only installation fallback. The upstream license is
installed as `FAISS-LICENSE`; `COPYRIGHTS` records attribution and patches.
cuVS, the CUDA toolkit, NVIDIA drivers and numerical libraries are not bundled.

| Component | License | Distribution relationship |
|---|---|---|
| FAISS | Expat ([license](https://github.com/facebookresearch/faiss/blob/v1.11.0/LICENSE)) | Compatible installed library, or static CPU library built from bundled source. |
| RAPIDS cuVS | Apache License 2.0 ([license](https://github.com/rapidsai/cuvs/blob/main/LICENSE)) | Optional system library supplied by the user or build environment. |
| NVIDIA CUDA toolkit and driver | NVIDIA CUDA SDK EULA ([terms](https://docs.nvidia.com/cuda/eula/index.html)) | Optional user-installed runtime/toolkit; not redistributed by `faissR`. |

The FAISS source retains its original license and file-level notices.
Users and binary distributors remain
responsible for satisfying the applicable external licenses and for using an
ABI-compatible compiler and library stack.
