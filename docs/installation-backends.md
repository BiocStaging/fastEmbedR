# Installation And Backends

This page describes `fastEmbedR` embedding backends. The package contains its
own CPU HNSW and Metal exact/IVF search code. CUDA builds link directly to the
RAPIDS cuVS C API; they do not call `faissR` or Python for one-call KNN.

## Standard Installation

```r
install.packages("remotes")
remotes::install_github("tkcaccia/fastEmbedR")
```

Install `faissR` separately only when its reusable public KNN/classification
API is wanted. It is not a runtime dependency of one-call fastEmbedR
embeddings.

## Backend Rule

The public embedding backends are:

- `backend = "cpu"`;
- `backend = "metal"` on Apple Silicon when native Metal symbols are compiled;
- `backend = "cuda"` when native CUDA symbols are compiled.

Explicit unavailable GPU requests fail clearly. They are never reported as GPU
after running on CPU.

## Metal

On Apple Silicon, `umap_knn(..., backend = "metal")` uses the native
Objective-C++/Metal UMAP optimizer from the supplied KNN graph. It does not call
Python, `reticulate`, Torch, or MLX.

Native Metal openTSNE uses the package Metal FFT-grid path when compiled. The
validated default uses PCA initialization for matrix-input workflows and the
current Metal FFT-grid implementation for the negative-gradient approximation.

## CUDA

Native CUDA openTSNE is implemented with CUDA kernels and cuFFT when the CUDA
backend is compiled. Native CUDA UMAP uses the package CUDA optimizer when
available.

Compile the CUDA embedding backend with:

```sh
CUDA_HOME=/usr/local/cuda \
FASTEMBEDR_USE_CUDA=1 \
R CMD INSTALL /path/to/fastEmbedR
```

To compile the native cuVS KNN and RAFT TSVD initializer without requiring the
full cuML library, point the build at compatible cuVS, RAFT, RMM, and CCCL
prefixes:

```sh
CUDA_HOME=/usr/local/cuda \
FAISS_HOME=/path/to/faiss-gpu \
RAFT_HOME=/path/to/rapids \
RMM_HOME=/path/to/rapids \
RAPIDS_HOME=/path/to/rapids \
CUVS_HOME=/path/to/rapids \
CCCL_HOME=/path/to/compatible-cccl \
FASTEMBEDR_USE_CUDA=1 \
FASTEMBEDR_USE_FAISS_GPU=1 \
FASTEMBEDR_USE_CUVS=1 \
FASTEMBEDR_USE_RAFT=1 \
R CMD INSTALL /path/to/fastEmbedR
```

The RAFT and CCCL releases must be mutually compatible. For example, RAPIDS
26.06 requires CCCL 3.3 rather than the older CCCL headers bundled with some
CUDA toolkit installations. Set `FASTEMBEDR_CUDA_ARCH` to the target compute
capability when building for a specific GPU (for example, `75` for a Tesla
T4).

The FAISS build must provide `faiss/gpu/GpuDistance.h` and `libfaiss` with GPU
support. The cuVS build must provide `cuvs/core/c_api.h`, `libcuvs_c`, and `libcuvs`.
At run time those libraries must remain discoverable through the recorded
rpath or `LD_LIBRARY_PATH`. `FASTEMBEDR_USE_CUVS=1` is strict: configuration
fails if the headers or libraries are missing. `FASTEMBEDR_USE_FAISS_GPU=1` is
also strict and prevents an exact CUDA request from being replaced by a slower
provider.

## Diagnostics

```r
library(fastEmbedR)

fastEmbedR:::backend_info()
```

The diagnostic reports native KNN and embedding availability separately. If
an optional backend is unavailable, `fastEmbedR` reports the failure instead
of silently falling back to CPU.
