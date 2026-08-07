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
nearest-neighbor graphs. Its primary contributions are:

- UMAP from KNN input;
- openTSNE-style t-SNE from KNN input;
- native CPU, Apple Metal, and CUDA embedding backends where available;
- float32 input/output support with float32 native optimizer buffers;
- explicit backend reporting, with no silent CPU fallback labelled as GPU;
- native CPU HNSW and Apple Metal exact/IVF-Flat KNN for one-call embeddings;
- optional GPU-resident CUDA KNN through direct FAISS GPU and RAPIDS cuVS APIs.

Here, **openTSNE-style** describes algorithmic lineage: sparse perplexity
affinities, two-phase t-SNE optimization, FIt-SNE interpolation/FFT repulsion,
and fixed-reference transformation. It does not mean that fastEmbedR wraps,
ports, or is API-compatible with the Python `openTSNE` package. fastEmbedR
defines its own R API, defaults, objects, float32 storage, and native
CPU/Metal/CUDA kernels.

The intended workflow is:

1. call `opentsne()` or `umap()` and let fastEmbedR select its native KNN path,
   or call `precompute_knn()` explicitly with a CPU, Metal, or CUDA backend;
2. reuse that KNN object in `fastEmbedR::opentsne_knn()` or
   `fastEmbedR::umap_knn()`;
3. evaluate or plot the embedding.

For the one-call functions `opentsne()` and `umap()`, the embedding backend is
deliberately limited to `backend = "cpu"`, `"metal"`, or `"cuda"`. Internal
CPU one-call embeddings use the package-native float32 HNSW path. Metal uses
native exact search for small inputs and recall-tuned IVF-Flat for larger
inputs. CUDA uses direct FAISS GPU exact search below 100,000 rows and direct
cuVS IVF-Flat above that threshold, then passes package-owned device pointers
into UMAP or openTSNE. It does not call another R package for KNN. No
unavailable GPU backend is silently relabelled as CPU.

## Quick Start

```r
library(fastEmbedR)

x <- scale(as.matrix(iris[, 1:4]))
labels <- iris$Species

y_tsne <- fastEmbedR::opentsne(
  x,
  perplexity = 10,
  backend = "cpu",
  n.cores = 4,
  seed = 1
)

y_umap <- fastEmbedR::umap(
  x,
  n_neighbors = 15,
  backend = "cpu",
  n.cores = 4,
  seed = 1
)

plot(y_tsne, pch = 21, bg = labels)
plot(y_umap, pch = 21, bg = labels)

# Precompute once and reuse the identical neighbors.
knn <- fastEmbedR::precompute_knn(
  x, k = 15, backend = "cpu", n.cores = 4
)
y_from_knn <- fastEmbedR::umap_knn(knn, backend = "cpu", seed = 1)
```

`umap()` and `umap_knn()` use the standard fuzzy UMAP graph by default. Set
`graph_mode = "binary"` only for the explicit adjacency-only sensitivity mode.

When a one-call function receives a `float::float32` matrix, its returned
`layout` remains float32 to reduce host memory. Plot the embedding object
directly with `plot(fit)`; fastEmbedR decodes the compact payload for graphics
and quality metrics without changing the stored layout.

## Main Functions

| Function | Purpose |
| --- | --- |
| `precompute_knn()` | Native non-self KNN search on CPU, Metal, or CUDA, with backend-specific algorithm selection kept internal. |
| `opentsne_knn()` | Native openTSNE-style t-SNE from a supplied KNN object. |
| `opentsne()` | One-call KNN plus openTSNE-style t-SNE. |
| `umap_init()` | Build and retain a reusable UMAP graph plus its independent sparse initialization. |
| `umap_knn()` | Native UMAP from a supplied KNN object. |
| `umap()` | One-call KNN plus UMAP. |
| `pca()` | Backend-native truncated PCA; CPU calls expose `n.cores`, and `opentsne_init = TRUE` returns a ready-to-use openTSNE initialization. |
| `select_landmarks()` | Select and retain a reusable landmark/reference split. |
| `fit_landmark_model()` | Fit ordinary UMAP or openTSNE on the landmark reference. |
| `project_landmark_model()` | Project held-out or new observations into the fixed reference. |
| `landmark_tsne()` / `landmark_umap()` | One-call landmark embedding and projection workflows. |
| `evaluate_embedding()` | Trustworthiness, neighbor preservation, label accuracy, and related metrics. |

### Optional Downstream Graph Utilities

Clustering is a secondary downstream facility rather than the package's
principal contribution. An embedding or KNN result can be passed to
`knn_graph()`, followed by `graph_cluster()`. Louvain and Leiden have
package-native CPU, CUDA, and Metal backends; Walktrap is CPU-only.

| Function | Purpose |
| --- | --- |
| `knn_graph()` | Compact graph from data, an embedding, or supplied neighbors. |
| `graph_cluster()` | Native Louvain, Leiden, or Pons-Latapy Walktrap communities. |

## Installation

For the development version:

```r
install.packages("remotes")
remotes::install_github("tkcaccia/fastEmbedR")
```

See [Installation](docs/installation.md) for `fastEmbedR` CPU, Metal, and CUDA
embedding builds, including direct FAISS GPU and RAPIDS cuVS linkage for CUDA KNN.
See [Bioconductor](docs/bioconductor.md) for the dependency split used for
submission: native CPU/Metal code in `fastEmbedR`, optional direct FAISS/cuVS CUDA
KNN, and reference packages only in `Suggests`.

## License

`fastEmbedR` is distributed under the MIT license. GPL packages such as `uwot`
are used only as optional external benchmark/reference tools, not as required
runtime dependencies or vendored source. Native KNN derivatives and optional
linked libraries retain the FAISS MIT, Faiss-mlx Apache-2.0, and RAPIDS cuVS
Apache-2.0 notices under `inst/LICENSES/`.
