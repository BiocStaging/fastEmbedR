# Usage And API

[Home](../README.md) |
[Installation](installation.md) |
[Bioconductor](bioconductor.md) |
[Implementation](implementation.md) |
[Examples](examples.md) |
[Benchmarks](benchmarks.md) |
**API** |
[Reproducibility](reproducibility.md) |
[References](references.md)

This page gives the main KNN-first workflows and the public API.

## Which Function Should I Use?

| Situation | Use |
| --- | --- |
| You already computed nearest neighbours | `umap_knn()` or `opentsne_knn()` |
| You want one call from a data matrix | `umap()` or `opentsne()` |
| You want reusable PCA scores or t-SNE initialization | `pca()` or `opentsne_pca_init()` |
| You want to compare UMAP and openTSNE fairly | compute `knn <- faissR::nn(...)` once, then reuse it |
| You want Apple GPU | set `backend = "metal"` explicitly |
| You want NVIDIA GPU | build with CUDA/cuVS, then set embedding `backend = "cuda"` |
| You want a fast approximation for very large data | use `landmark_umap()` or `landmark_tsne()` and report it as landmarking |
| You want quality metrics | `evaluate_embedding(x, layout)` |

The recommended workflow is KNN first:

```r
knn <- faissR::nn(x, k = 50, exclude_self = TRUE, backend = "auto", n_threads = 4)
layout_umap <- umap_knn(knn, seed = 1)
layout_tsne <- opentsne_knn(knn, init_data = x, seed = 1)
```

This keeps nearest-neighbour time separate from embedding time and makes
benchmarks easier to interpret.

The one-call functions intentionally hide the KNN algorithm choice. For
`opentsne()` and `umap()`, `backend` accepts only `"cpu"`, `"metal"`, or
`"cuda"`. CPU KNN uses native HNSW; Metal uses native exact/IVF-Flat; CUDA
uses direct FAISS GPU exact search or RAPIDS cuVS IVF-Flat and keeps its output
resident on the device. To benchmark another KNN algorithm, compute it
explicitly with `faissR::nn()` or `faissR::nn_gpu()` and pass the result to
`opentsne_knn()` or `umap_knn()`.

## Distance Metrics In `faissR::nn()`

The default distance is Euclidean:

```r
knn <- faissR::nn(x, k = 50, exclude_self = TRUE, metric = "euclidean", backend = "auto", n_threads = 4)
```

Cosine distance is available through exact CPU KNN:

```r
knn_cosine <- faissR::nn(x, k = 50, exclude_self = TRUE, metric = "cosine", backend = "cpu", n_threads = 4)
layout <- umap_knn(knn_cosine, seed = 1)
```

Current metric support is deliberately explicit:

| metric | supported backends | notes |
| --- | --- | --- |
| `euclidean` | native CPU/Metal and optional direct CUDA/cuVS | Recommended default for large UMAP/openTSNE benchmarks. |
| `cosine` / inner product | FAISS/candidate paths where enabled by `faissR` | Use normalized rows when treating inner product as cosine similarity. |

## Basic KNN-First UMAP

```r
library(fastEmbedR)

set.seed(1)
x <- scale(as.matrix(iris[, 1:4]))
labels <- iris$Species

knn <- faissR::nn(x, k = 31, exclude_self = TRUE)
layout <- umap_knn(knn)

plot(layout, pch = 21, bg = labels)
```

The one-call interface computes KNN internally:

```r
fit <- umap(
  x,
  n_neighbors = 30,
  seed = 1
)
plot(fit)
```

## openTSNE From The Same KNN

```r
Y_init <- opentsne_pca_init(x, seed = 1)
layout_tsne <- opentsne_knn(
  knn,
  Y_init = Y_init,
  perplexity = 10,
  early_exaggeration_iter = 100,
  n_iter = 250
)

plot(layout_tsne, pch = 21, bg = labels)
```

`Y_init` can be computed once and reused across runs. `init_data` is still
available as a convenience; it is used only to compute PCA initialization for
KNN-input runs and is not used for neighbour search or optimization.

## PCA API

`fastEmbedR::pca()` exposes the backend-native truncated PCA used internally
for openTSNE initialization:

```r
pca_fit <- pca(x, ncomp = 2, backend = "cpu", seed = 1)
Y_init <- opentsne_pca_init(x, backend = "cpu", seed = 1)
```

The public `pca()` helper is intentionally simple: there is no `irlba` or
ARPACK method menu and no Python bridge. For openTSNE initialization, CUDA uses
native RAPIDS RAFT TSVD compiled into the package CUDA backend and fails loudly
if that support is unavailable. Metal uses a native float32 block-subspace TSVD
with MPS matrix products and a resident workspace. CPU uses the fastPLS-style
RSVD family available to the package build.

## Explicit GPU Use

GPU use is explicit. A request for Metal or CUDA must run that backend or fail
clearly.

```r
knn <- faissR::nn(x, k = 50, exclude_self = TRUE, backend = "metal")
layout <- opentsne_knn(knn, init_data = x, backend = "metal", seed = 1)
```

For CUDA builds with RAPIDS cuVS available:

```r
fit <- opentsne(x, perplexity = 50, backend = "cuda", seed = 1)
```

The package does not silently run these examples on CPU and report them as GPU
results.

## Landmark Workflow

```r
fit <- landmark_tsne(
  x,
  landmarks = 0.5,
  n_neighbors = 30,
  perplexity = 10,
  early_exaggeration_iter = 100,
  n_iter = 250,
  transform_iter = 100,
  seed = 1
)
plot(fit)
```

UMAP has the same landmark pattern:

```r
fit <- landmark_umap(
  x,
  landmarks = 0.5,
  n_neighbors = 30,
  backend = "cpu",
  seed = 1
)
plot(fit)
```

For landmark runs, `backend = "metal"` uses a fused native Metal projection
kernel that computes query-to-landmark KNN, interpolation, and projection
confidence in one pass before the fixed-reference transform. CPU runs use
exact multi-threaded projection KNN by default and switch to a native
projection-specific approximation only for large projections where the cheaper
candidate search is worthwhile.

## Automatic Parameters

`opentsne()` and `opentsne_knn()` use `auto_config = TRUE` by default. Missing
t-SNE settings are resolved in native C++ using the opt-SNE strategy:

- `"auto"` learning rate becomes `n / early_exaggeration`.
- Early exaggeration can stop at the local maximum of KLD relative change.
- The normal phase can stop when KLD improvement drops below the opt-SNE
  threshold.

The KLD monitor is enabled only where it is computationally honest: CPU/small
exact runs. Large FFT and GPU runs keep opt-SNE's learning-rate/default-limit
policy but do not perform a hidden CPU O(n^2) KLD poll or report it as GPU
work.

`umap()` and `umap_knn()` also choose internal defaults from the supplied KNN
distance profile in C++. This keeps the public API small while preserving the
supplied neighbour graph.

## Public API

| Function | Purpose |
| --- | --- |
| `faissR::nn()` | Companion-package KNN function for data/query matrices. |
| `umap_knn()` | UMAP from a supplied KNN object or matrices. |
| `umap()` | One-call preprocessing, KNN, and UMAP embedding. |
| `pca()` | Backend-native truncated PCA scores/loadings. |
| `embed_knn()` | KNN dispatcher; UMAP by default, openTSNE with `method = "opentsne"`. |
| `opentsne_knn()` | Direct native openTSNE-style optimizer from KNN. |
| `opentsne()` | One-call preprocessing, KNN, and openTSNE-style embedding. |
| `transform_tsne()` | Fixed-reference openTSNE-style transform for query points. |
| `landmark_tsne()` | Embed landmarks, then transform remaining rows. |
| `landmark_umap()` | Embed landmarks with UMAP, then project/refine remaining rows. |
| `evaluate_embedding()` | Embedding quality metrics. |
| `faissR::backend_info()` | FAISS/cuVS KNN backend detection without silent fallback. |
