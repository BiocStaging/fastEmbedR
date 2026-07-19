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
remotes::install_github("tkcaccia/fastEmbedR")
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
- a C++17 compiler;
- `Rcpp`;
- macOS Metal framework for native Metal embedding kernels on Apple Silicon;
- CUDA toolkit plus RAPIDS cuVS/RAFT for optional native CUDA workflows.

`fastEmbedR` links directly to FAISS GPU and the RAPIDS cuVS C API for optional CUDA
one-call KNN. Follow the [backend build guide](installation-backends.md) for
native CUDA. CPU, Metal, and correctly compiled CUDA `opentsne()`/`umap()` do
not call another R package for neighbour search.

## CUDA Embedding Build

CUDA KNN uses direct FAISS GPU exact search and direct cuVS IVF-Flat linkage.
CUDA builds also compile the native UMAP/openTSNE kernels.

```sh
CUDA_HOME=/usr/local/cuda \
FAISS_HOME=/path/to/faiss-gpu \
CUVS_HOME=/path/to/rapids \
FASTEMBEDR_USE_CUDA=1 \
FASTEMBEDR_USE_FAISS_GPU=1 \
FASTEMBEDR_USE_CUVS=1 \
R CMD INSTALL /path/to/fastEmbedR
```

If CUDA is requested explicitly and unavailable, the embedding function fails
clearly. It does not run on CPU while reporting CUDA.

## Apple Metal

On Apple Silicon, `fastEmbedR` builds native Objective-C++/Metal embedding
kernels for:

- exact and recall-tuned IVF-Flat KNN;
- UMAP layout optimization from KNN;
- openTSNE FFT-grid optimization;
- selected projection/refinement operations.

No Python, Torch, MLX, or `reticulate` call is required for the public Metal
embedding paths.

## Backend Check

For CUDA dependency diagnostics after installation:

```r
library(fastEmbedR)
fastEmbedR:::backend_info()
```

The diagnostic reports CUDA nearest-neighbour and embedding availability.
`fastEmbedR` checks CPU, Metal, and CUDA embedding backends when a function is
called with `backend = "cpu"`, `"metal"`, or `"cuda"`.

## Backend Rule

Backend labels are strict. An explicit GPU request must resolve to a real
native GPU backend. Otherwise the function errors and reports what dependency
is missing.
