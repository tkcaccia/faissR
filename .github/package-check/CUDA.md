# CUDA source-package test matrix

This matrix validates faissR against several CUDA user-space stacks while the
host's working NVIDIA driver remains unchanged. Singularity `--nv` exposes the
host driver and devices; every image supplies its own CUDA toolkit, R, and
provider libraries. Ubuntu 22.04, Ubuntu 24.04, and Debian 13 user spaces are
included. The definitions never install a driver on the host.

## Safety boundary

Do not install distribution `nvidia-cuda-toolkit`, NVIDIA `cuda`, or
`cuda-runtime-X-Y` metapackages on a shared host merely to run this matrix.
Build images from NVIDIA's toolkit containers. Driver administration remains a
separate system-owner task. Stop if `nvidia-smi` does not work before testing.

CUDA 12.4 is included as a deliberate compatibility test. It predates native
Blackwell code generation, so the image retains compute_80 PTX for driver JIT.
CUDA 12.8 and 13.2 generate native compute capability 12.0 code. Debian 13 is
paired with CUDA 13.2 because NVIDIA's 12.8 support table covers Debian 12,
whereas the 13.2 guide includes Debian 13. A failure in one row limits the
supported matrix; it must not be hidden by substituting CPU.

## Build images

Use a large secondary disk and preserve each image checksum and build log:

```sh
export PACKAGE_TEST_ROOT=/large-disk/r-package-test-lab
bash .github/package-check/linux/build-cuda-matrix.sh
```

The full cuVS image installs the RAPIDS 26.06 C library in an isolated prefix.
The Debian image obtains a versioned CUDA toolkit from isolated conda packages
rather than the Debian CUDA metapackage. The other images test native faissR
CUDA plus FAISS GPU without direct cuVS.

## Run the identical source archive

First verify that the host driver and GPU work:

```sh
nvidia-smi
singularity exec --nv \
  "$PACKAGE_TEST_ROOT/images/cuda-12.8-ubuntu24-faiss.sif" \
  nvidia-smi
```

Then run a source tarball with a new result directory:

```sh
export PACKAGE_TEST_COMMIT=$(git rev-parse HEAD)
bash .github/package-check/linux/run-cuda-matrix.sh \
  /absolute/path/faissR_VERSION.tar.gz \
  "$PACKAGE_TEST_ROOT/runs/cuda-UNIQUE-ID"
```

Each row performs a strict source installation, an independent-reference CUDA
smoke test, and `R CMD check --as-cran --no-manual`. `cuda-cuvs` additionally
requires the direct cuVS provider. Results record the source and image SHA256,
R session, `nvcc --version`, `nvidia-smi`, installation log, smoke log, and
check log. The matrix supplies explicit architecture overrides to exercise
native and PTX compatibility rows; normal installations detect visible GPU
architectures automatically. A CPU fallback is a failure.

Images with nonstandard prefixes can be tested without rewriting them. Set
`CONTAINER_R_BIN`, `CONTAINER_CUDA_HOME`, `CONTAINER_FAISS_HOME`, and
`CONTAINER_CUVS_HOME` before calling `run-cuda.sh`. Leave
`FAISSR_CUDA_ARCH` and `FAISSR_CUDA_PTX_ARCH` unset to test automatic
architecture detection; set them only for a deliberate compatibility row.

If a minimal image omits checking utilities, set
`CONTAINER_CHECK_TOOLS_ROOT` to a read-only tree containing
`usr/bin/checkbashisms`, `usr/bin/nm`, and any matching libraries under
`usr/lib/x86_64-linux-gnu`. This tree is test infrastructure, not a package or
runtime dependency.

These tests establish only the exact rows that pass. They do not imply that an
untested mixture of driver, toolkit, FAISS, cuVS, compiler, and GPU is supported.
