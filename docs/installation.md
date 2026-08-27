# Installation

[Home](../README.md) |
**Installation** |
[Bioconductor](bioconductor.md) |
[Implementation](implementation.md) |
[Examples](examples.md) |
[Benchmarks](benchmarks.md) |
[API](usage-api.md) |
[Reproducibility](reproducibility.md) |
[References](references.md)

`fastEmbedR` contains the compact native KNN routes required by its one-call
embeddings, UMAP, openTSNE-style t-SNE, landmark transforms, embedding metrics,
KNN graph construction, and native community detection.

## R Packages

```r
install.packages("remotes")
# Use the exact release tag or full commit recorded in the publication or
# analysis manifest; do not use a moving branch for a reproducible analysis.
ref <- "REPLACE_WITH_FROZEN_TAG_OR_COMMIT"
remotes::install_github(paste0("tkcaccia/fastEmbedR@", ref))
```

Suggested benchmark/reference packages:

```r
install.packages(c(
  "Rtsne", "uwot", "umap",
  "igraph", "jsonlite", "knitr", "rmarkdown", "float"
))
```

`Rtsne`, `uwot`, and `umap` are optional comparison packages. They are not
required by the core `fastEmbedR` embedding functions.

## Core Build Dependencies

`fastEmbedR` needs:

- R;
- the C++17 compiler configured for that R installation;
- `Rcpp`;
- Xcode/Apple Metal frameworks for native Metal kernels on Apple Silicon;
- CUDA toolkit plus RAPIDS cuVS/RAFT for optional native CUDA workflows.

`fastEmbedR` links directly to FAISS GPU and the RAPIDS cuVS C API for optional CUDA
one-call KNN. Follow the [backend build guide](installation-backends.md) for
the exact compiler, GPU-architecture, host-compiler, header, library, and
runtime requirements. CPU, Metal, and correctly compiled CUDA
`opentsne()`/`umap()` do not call another R package for neighbor search.

The portable C++ core inherits `CXX17` and `CXX17FLAGS` from R and adds only
`-pthread`. The package does not globally force `-march=native`,
`-ffast-math`, or `-O3`. This keeps release binaries portable and avoids
changing floating-point-sensitive KNN and embedding trajectories.

## CUDA Embedding Build

CUDA KNN uses direct FAISS GPU exact search and direct cuVS IVF-Flat linkage.
CUDA builds also compile the native UMAP/openTSNE kernels.

```sh
CUDA_HOME=/usr/local/cuda \
FAISS_HOME=/path/to/faiss-gpu \
CUVS_HOME=/path/to/rapids \
RAFT_HOME=/path/to/rapids \
RMM_HOME=/path/to/rapids \
CCCL_HOME=/path/to/compatible-cccl \
CUDAHOSTCXX=/path/to/cuda-compatible-c++ \
FASTEMBEDR_CUDA_ARCH="75 89" \
FASTEMBEDR_USE_CUDA=1 \
FASTEMBEDR_USE_FAISS_GPU=1 \
FASTEMBEDR_USE_CUVS=1 \
FASTEMBEDR_USE_RAFT=1 \
R CMD INSTALL /path/to/fastEmbedR
```

If CUDA is requested explicitly and unavailable, the embedding function fails
clearly. It does not run on CPU while reporting CUDA.

`FASTEMBEDR_CUDA_ARCH` must cover every deployment GPU, and linked FAISS/cuVS
libraries must be compiled for the same devices. For example, `75` covers a T4
and `89` covers an L40S.

## Apple Metal

The build-supported Metal target is Apple Silicon. The source requires an SDK that
exposes the macOS 14 MPSGraph FFT declarations, so use macOS 14 or newer and a
full Xcode 15 or newer toolchain. Full performance benchmarking is currently
limited to a MacBook Pro with an Apple M3, macOS 14.5, Xcode 16.2, and 8 GB
unified memory. Other Apple Silicon systems are compatibility targets, not
performance-validated configurations. Intel Macs are not a supported Metal
target; use `backend = "cpu"` on those systems.

On a build-supported Apple Silicon system, `fastEmbedR` builds native
Objective-C++/Metal embedding kernels for:

- exact and recall-tuned IVF-Flat KNN;
- UMAP layout optimization from KNN;
- openTSNE FFT-grid optimization;
- selected projection/refinement operations.

No Python, Torch, MLX, or `reticulate` call is required for the public Metal
embedding paths.

Metal feature parity is explicit rather than inferred from a backend label.
Core KNN, PCA, two-dimensional UMAP/openTSNE, transformation, and landmark
paths are native. `knn_graph()` uses Metal for neighbor search but constructs
the graph object on CPU; `evaluate_embedding()` computes metrics on CPU after
the final layout transfer; and Walktrap clustering is CPU-only. See the
[backend capability matrix](backend-capabilities.md) for dimensional and
metric limits.

## Backend Check

For CUDA dependency diagnostics after installation:

```r
library(fastEmbedR)
fastEmbedR_capabilities()
```

The diagnostic reports CUDA nearest-neighbor and embedding availability.
`fastEmbedR` checks CPU, Metal, and CUDA embedding backends when a function is
called with `backend = "cpu"`, `"metal"`, or `"cuda"`.

## Backend Rule

Backend labels are strict. An explicit GPU request must resolve to a real
native GPU backend. Otherwise the function errors and reports what dependency
is missing.
