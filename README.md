# fastEmbedR

**Home** |
[Installation](docs/installation.md) |
[Bioconductor](docs/bioconductor.md) |
[Implementation](docs/implementation.md) |
[Performance Engineering](docs/backend-performance-engineering.md) |
[Examples](docs/examples.md) |
[Benchmarks](docs/benchmarks.md) |
[API](docs/usage-api.md) |
[Reproducibility](docs/reproducibility.md) |
[References](docs/references.md)

`fastEmbedR` is a native R/C++ package for fast dimensionality reduction from
nearest-neighbour graphs. It focuses on:

- UMAP from KNN input;
- openTSNE-style t-SNE from KNN input;
- native CPU, Apple Metal, and CUDA embedding backends where available;
- float32 input/output support with float32 native optimizer buffers;
- explicit backend reporting, with no silent CPU fallback labelled as GPU;
- native CPU HNSW and Apple Metal exact/IVF-Flat KNN for one-call embeddings;
- optional GPU-resident CUDA KNN through direct FAISS GPU and RAPIDS cuVS APIs.

The intended workflow is:

1. call `opentsne()` or `umap()` and let fastEmbedR select its native KNN path,
   or compute a reusable graph with `faissR::nn()`;
2. reuse a supplied KNN object in `fastEmbedR::opentsne_knn()` or
   `fastEmbedR::umap_knn()`;
3. evaluate or plot the embedding.

For the one-call functions `opentsne()` and `umap()`, the embedding backend is
deliberately limited to `backend = "cpu"`, `"metal"`, or `"cuda"`. Internal
CPU one-call embeddings use the package-native float32 HNSW path. Metal uses
native exact search for small inputs and recall-tuned IVF-Flat for larger
inputs. CUDA uses direct FAISS GPU exact search below 100,000 rows and direct
cuVS IVF-Flat above that threshold, then passes package-owned device pointers
into UMAP or openTSNE. It does not call the faissR R API. No unavailable GPU
backend is silently relabelled as CPU.

## Quick Start

```r
library(fastEmbedR)

x <- scale(as.matrix(iris[, 1:4]))
labels <- iris$Species

knn <- faissR::nn(x, k = 15, backend = "auto", n_threads = 4)

y_tsne <- fastEmbedR::opentsne_knn(
  knn,
  init_data = x,
  backend = "cpu",
  seed = 1
)

y_umap <- fastEmbedR::umap_knn(
  knn,
  backend = "cpu",
  graph_mode = "fuzzy",
  seed = 1
)

plot(y_tsne, pch = 21, bg = labels)
plot(y_umap, pch = 21, bg = labels)
```

When a one-call function receives a `float::float32` matrix, its returned
`layout` remains float32 to reduce host memory. Plot the embedding object
directly with `plot(fit)`; fastEmbedR decodes the compact payload for graphics
and quality metrics without changing the stored layout.

## Main Functions

| Function | Purpose |
| --- | --- |
| `faissR::nn()` | Optional reusable FAISS/cuVS neighbour search supplied by the companion package. |
| `opentsne_knn()` | Native openTSNE-style t-SNE from a supplied KNN object. |
| `opentsne()` | One-call KNN plus openTSNE-style t-SNE. |
| `umap_knn()` | Native UMAP from a supplied KNN object. |
| `umap()` | One-call KNN plus UMAP. |
| `pca()` | Backend-native truncated PCA: CPU RSVD, resident Metal/MPS TSVD, or compiled CUDA RAFT TSVD. |
| `landmark_tsne()` / `landmark_umap()` | Landmark embedding and projection workflows. |
| `evaluate_embedding()` | Trustworthiness, neighbour preservation, label accuracy, and related metrics. |
| `faissR::backend_info()` | Report FAISS/cuVS neighbour-search availability. |

## Installation

For the development version:

```r
install.packages("remotes")
remotes::install_github("tkcaccia/fastEmbedR")
```

Install `faissR` separately only when its reusable public KNN API is needed.

See [Installation](docs/installation.md) for `fastEmbedR` CPU, Metal, and CUDA
embedding builds, including direct FAISS GPU and RAPIDS cuVS linkage for CUDA KNN. The
companion [`faissR`](https://github.com/tkcaccia/faissR) project documents its
broader reusable neighbour-search API.
See [Bioconductor](docs/bioconductor.md) for the dependency split used for
submission: native CPU/Metal code in `fastEmbedR`, optional direct FAISS/cuVS CUDA
KNN, and reference packages only in `Suggests`.

## License

`fastEmbedR` is distributed under the MIT license. GPL packages such as `uwot`
are used only as optional external benchmark/reference tools, not as required
runtime dependencies or vendored source. Native KNN derivatives and optional
linked libraries retain the FAISS MIT, Faiss-mlx Apache-2.0, and RAPIDS cuVS
Apache-2.0 notices under `inst/LICENSES/`.
