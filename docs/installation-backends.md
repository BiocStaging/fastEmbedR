# Installation And Native Compiler Configuration

[Home](../README.md) |
[Installation](installation.md) |
[Bioconductor](bioconductor.md) |
[Implementation](implementation.md) |
[Examples](examples.md) |
[Benchmarks](benchmarks.md) |
[API](usage-api.md) |
[Reproducibility](reproducibility.md) |
[References](references.md)

This page is the build contract for the native CPU, Apple Metal, and NVIDIA
CUDA backends. The public backend names are `"cpu"`, `"metal"`, and `"cuda"`.
An explicitly requested GPU backend fails if its native code or linked runtime
is unavailable; fastEmbedR never performs a CPU calculation and labels it as
Metal or CUDA.

## Compiler Policy

fastEmbedR deliberately keeps package flags small:

| Layer | Standard and package-owned flags | Source of optimization flags |
| --- | --- | --- |
| Portable C++ core | `C++17`; package adds `-pthread` | The active R installation's `CXX17` and `CXX17FLAGS` |
| Apple Metal | Objective-C++ plus `Foundation`, `Metal`, `MetalPerformanceShaders`, and `MetalPerformanceShadersGraph`; `PKG_OBJCXXFLAGS` inherits R's C++ release flags | Apple Clang/Xcode and the runtime Metal compiler |
| NVIDIA CUDA | `-std=c++17 -x cu --extended-lambda --expt-relaxed-constexpr -Xcompiler -fPIC` | NVCC plus optional user-supplied `FASTEMBEDR_CUDA_FLAGS` |

The CPU implementation uses a persistent `std::thread` worker team. It does
not require OpenMP, and the package does not add `-fopenmp`.

The package does **not** globally force `-march=native`, `-ffast-math`, or
`-O3`. R's platform defaults, normally a portable `-O2` release build, are
inherited. This policy has three reasons:

1. `-march=native` creates a binary that may fail on another CPU, including a
   compute node different from the build host.
2. relaxed floating-point transformations can change KNN tie decisions and
   stochastic embedding trajectories.
3. an experimental `-O3 -march=native` build did not improve the validated
   native HNSW benchmark, so it was not retained.

The native Metal KNN shader is the one narrow exception: it enables Apple's
runtime fast arithmetic for float32 candidate search. UMAP and openTSNE
embedding kernels do not obtain a global host-compiler fast-math flag.

Inspect the flags resolved by the active R installation before comparing
machines:

```sh
R CMD config CXX17
R CMD config CXX17FLAGS
R CMD config CPPFLAGS
R CMD config LDFLAGS
```

All packages in a runtime comparison should be built and run in the same R
environment. Compiler and BLAS differences can otherwise be mistaken for an
algorithmic speed difference.

## Standard CPU Installation

Requirements:

- R and its development headers;
- a compiler accepted by `R CMD config CXX17`;
- R package `Rcpp`;
- POSIX threads on Unix-like systems.

From a source checkout:

```sh
R CMD INSTALL --preclean /path/to/fastEmbedR
```

Or install the development source from R:

```r
install.packages("remotes")
# Use the exact release tag or full commit recorded in the build manifest.
ref <- "REPLACE_WITH_FROZEN_TAG_OR_COMMIT"
remotes::install_github(paste0("tkcaccia/fastEmbedR@", ref))
```

The configuration summary printed during installation must report
`C++17 (R CXX17/CXX17FLAGS)`. CPU HNSW, UMAP, openTSNE, PCA, transforms, and
clustering are package-native.

## Apple Metal Installation

The build-supported Metal target is Apple Silicon with macOS 14 or newer. The source
requires an SDK exposing the macOS 14 MPSGraph FFT declarations, so use a full
Xcode 15 or newer installation. Full performance benchmarking is limited to an
Apple M3 MacBook Pro, macOS 14.5, Xcode 16.2, and 8 GB unified memory. Other
Apple Silicon generations are compatibility targets until a strict real-device
artifact is archived, and no M3 speed claim transfers to them. Intel Macs are
not a supported Metal target and should use the CPU backend. Verify the
toolchain and SDK:

```sh
xcode-select -p
xcrun --find clang++
xcrun --find metal
xcrun metal --version
xcrun --sdk macosx --show-sdk-path
```

Then install with the normal source command:

```sh
SDKROOT="$(xcrun --sdk macosx --show-sdk-path)" \
R CMD INSTALL --preclean /path/to/fastEmbedR
```

`configure` defines `HAVE_METAL=1` and links the four Apple frameworks listed
in the compiler-policy table. Metal Shading Language source is compiled at
runtime with `newLibraryWithSource`, so the installed macOS SDK and the runtime
OS must support the selected Metal language version. The public Metal path
does not call Python, Torch, MLX, or `reticulate`.

Using an explicit `SDKROOT` also avoids stale Command Line Tools SDK paths
after an Xcode upgrade. If `xcrun --show-sdk-version` mentions a missing SDK
but `xcrun --sdk macosx --show-sdk-version` succeeds, use the explicit command
above rather than adding manual framework paths.

Verify the compiled symbols:

```r
library(fastEmbedR)
info <- fastEmbedR_capabilities()
print(info)
stopifnot(info$knn_available[info$backend == "metal"])
stopifnot(info$embedding_available[info$backend == "metal"])
```

## NVIDIA CUDA Installation

### Required components

A complete native CUDA build needs compatible installations of:

- a CUDA toolkit containing NVCC, CUDA runtime, cuFFT, cuBLAS, and cuSOLVER;
- FAISS built with GPU support for exact search;
- RAPIDS cuVS and its C API for recall-tuned IVF-Flat search;
- RAPIDS RAFT, RMM, and compatible CCCL headers for CUDA PCA/TSVD.

The CUDA PCA route uses RAFT TSVD directly and does not link or call cuML.

The CUDA host compiler must be supported by the selected CUDA toolkit and ABI
compatible with the compiler used to build R. Set `CUDAHOSTCXX` explicitly in
mixed Conda/system-toolchain environments.

### GPU architecture flags

`FASTEMBEDR_CUDA_ARCH` accepts a whitespace-, comma-, or semicolon-separated
list. Numeric (`89`), dotted (`8.9`), `sm_89`, and `compute_89` forms are
normalized by `configure`.

Examples:

```sh
# NVIDIA T4
export FASTEMBEDR_CUDA_ARCH="75"

# NVIDIA L40S
export FASTEMBEDR_CUDA_ARCH="89"

# One relocatable build for both machines
export FASTEMBEDR_CUDA_ARCH="75 89"
```

Query the installed device when `nvidia-smi` supports compute capability:

```sh
nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader
```

The architecture setting above controls only CUDA translation units compiled
by fastEmbedR. FAISS GPU, cuVS, RAFT, and every other linked CUDA library must
also contain kernels or PTX compatible with every deployment GPU. A package
binary containing `sm_89` cannot repair a cuVS library built only for `sm_75`.
The characteristic runtime error is:

```text
cudaErrorNoKernelImageForDevice: no kernel image is available for execution
```

### Strict complete build

```sh
export CUDA_HOME=/usr/local/cuda
export FAISS_HOME=/path/to/faiss-gpu
export CUVS_HOME=/path/to/rapids
export RAPIDS_HOME=/path/to/rapids
export RAFT_HOME=/path/to/rapids
export RMM_HOME=/path/to/rapids
export CCCL_HOME=/path/to/compatible-cccl
export CUDAHOSTCXX=/path/to/cuda-compatible-c++
export FASTEMBEDR_CUDA_ARCH="75 89"

FASTEMBEDR_USE_CUDA=1 \
FASTEMBEDR_USE_FAISS_GPU=1 \
FASTEMBEDR_USE_CUVS=1 \
FASTEMBEDR_USE_RAFT=1 \
R CMD INSTALL --preclean /path/to/fastEmbedR
```

The strict `=1` switches make configuration stop when a requested dependency
is missing. `auto` is convenient for development, but it is unsuitable for a
release image whose CUDA capabilities must be known.

Use `FASTEMBEDR_CUDA_FLAGS` only for a documented toolchain requirement, such
as an additional include directory. It is appended verbatim to the NVCC
command. It should not be used to hide an incompatible CUDA, CCCL, RAFT, or
host-compiler combination.

### Headers, libraries, and runtime resolution

The build checks for:

- `faiss/gpu/GpuDistance.h` and `libfaiss`;
- `cuvs/core/c_api.h`, `libcuvs_c`, and `libcuvs`;
- `raft/linalg/tsvd.cuh` when RAFT TSVD is requested.

RAPIDS and CCCL releases must be compatible. Runtime library lookup must prefer
the same CUDA/RAPIDS prefix used at compilation:

```sh
export LD_LIBRARY_PATH=/path/to/rapids/lib:/path/to/faiss/lib:/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}
```

fastEmbedR records an rpath for discovered FAISS/cuVS libraries. Nevertheless,
`LD_LIBRARY_PATH` is still important when cuVS transitively loads a particular
`libnvJitLink`. An error such as an undefined versioned
`__nvJitLink...` symbol normally means that headers/libraries from different
CUDA releases have been mixed.

### CUDA validation

```r
library(fastEmbedR)

info <- fastEmbedR_capabilities()
print(info)
stopifnot(info$knn_available[info$backend == "cuda"])
stopifnot(info$embedding_available[info$backend == "cuda"])

set.seed(1)
x <- matrix(runif(5000 * 32), nrow = 5000, ncol = 32)
knn <- precompute_knn(x, k = 15, backend = "cuda")
stopifnot(identical(attr(knn, "backend"), "cuda"))

u <- umap(x, n_neighbors = 15, backend = "cuda", graph_mode = "fuzzy")
t <- opentsne(x, perplexity = 15, backend = "cuda")
stopifnot(nrow(u$layout) == 5000L, nrow(t$layout) == 5000L)
```

Also run the smoke script supplied with the separate benchmark repository:

```sh
git clone https://github.com/tkcaccia/fastEmbedR-benchmark.git
cd fastEmbedR-benchmark
bash tools/run_cuda_smoke_test.sh
```

## Release And Check Procedure

Build from a clean source tree, then check the source tarball rather than an
in-place directory:

```sh
R CMD build .
R CMD check --as-cran fastEmbedR_0.99.6.tar.gz
```

For a manuscript or release benchmark, archive:

- output of all `R CMD config` commands shown above;
- compiler and linker versions;
- generated `src/Makevars`;
- `FASTEMBEDR_CUDA_ARCH`, `FASTEMBEDR_CUDA_FLAGS`, and `CUDAHOSTCXX`;
- Xcode/Metal version or CUDA/NVCC/driver/GPU versions;
- FAISS, cuVS, RAFT, RMM, CCCL, cuFFT, and cuBLAS versions;
- `sessionInfo()`, Git commit, seed, and thread environment.

Run
[`tools/write_manuscript_reproducibility.R`](https://github.com/tkcaccia/fastEmbedR-benchmark/blob/main/tools/write_manuscript_reproducibility.R)
from the separate benchmark repository to capture these fields in the
benchmark output directory.
