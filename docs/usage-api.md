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
| You want to precompute fastEmbedR's native neighbors | `precompute_knn()` |
| You already computed nearest neighbors | `umap_knn()` or `opentsne_knn()` |
| You want one call from a data matrix | `umap()` or `opentsne()` |
| You want reusable PCA scores or t-SNE initialization | `pca()` or `opentsne_pca_init()` |
| You want to compare UMAP and openTSNE fairly | compute one host KNN list once, then reuse it |
| You want Apple GPU | set `backend = "metal"` explicitly |
| You want NVIDIA GPU | build with CUDA/cuVS, then set embedding `backend = "cuda"` |
| You want a fast approximation for very large data | use `landmark_umap()` or `landmark_tsne()` and report it as landmarking |
| You want quality metrics | `evaluate_embedding(x, layout)` |
| You want a clustering graph | `knn_graph()` |
| You want native graph communities | `graph_cluster(graph, method = "leiden")` |
| You want to inspect compiled backend support | `fastEmbedR_capabilities()` |

The recommended workflow is KNN first:

```r
knn <- precompute_knn(
  x,
  k = 50L,
  metric = "euclidean",
  backend = "cpu",
  n.cores = 4L
)
layout_umap <- umap_knn(knn, seed = 1)
layout_tsne <- opentsne_knn(knn, init_data = x, seed = 1)
```

This keeps nearest-neighbor time separate from embedding time and makes
benchmarks easier to interpret.

The one-call functions and `precompute_knn()` intentionally hide the KNN
algorithm choice. Their `backend` accepts only `"cpu"`, `"metal"`, or
`"cuda"`. CPU KNN uses native HNSW; Metal uses native exact/IVF-Flat; CUDA
uses direct FAISS GPU exact search or RAPIDS cuVS IVF-Flat and keeps its output
resident on the device. A CUDA KNN object should therefore be reused with a
CUDA embedding backend. A host KNN result from another tool may still be
supplied as a plain list containing `indices` and `distances`; fastEmbedR never
calls that tool itself.

## Distance Metrics

The default distance is Euclidean:

```r
fit <- umap(x, n_neighbors = 50, metric = "euclidean", n.cores = 4)
```

Cosine distance is available through exact CPU KNN:

```r
fit_cosine <- umap(x, n_neighbors = 50, metric = "cosine", n.cores = 4)
```

Current metric support is deliberately explicit:

| metric | supported backends | notes |
| --- | --- | --- |
| `euclidean` | native CPU/Metal and optional direct CUDA/cuVS | Recommended default for large UMAP/openTSNE benchmarks. |
| `cosine` | native CPU/Metal and compiled CUDA | Rows are normalized internally. |
| `correlation` | native CPU/Metal and compiled CUDA | Rows are centered and normalized internally. |
| `inner_product` | compiled CUDA only | Unsupported CPU/Metal requests fail explicitly. |

## Basic KNN-First UMAP

```r
library(fastEmbedR)

set.seed(1)
x <- scale(as.matrix(iris[, 1:4]))
labels <- iris$Species

fit <- umap(x, n_neighbors = 30, n.cores = 4)
layout <- fit$layout

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
KNN-input runs and is not used for neighbor search or optimization.

## PCA API

`fastEmbedR::pca()` exposes the backend-native truncated PCA used internally
for openTSNE initialization:

```r
pca_fit <- pca(
  x,
  ncomp = 2,
  backend = "cpu",
  n.cores = 4,
  seed = 1,
  opentsne_init = TRUE
)
Y_init <- pca_fit$opentsne_init
layout <- opentsne_knn(knn, Y_init = Y_init, perplexity = 30)
```

The public `pca()` helper is intentionally simple: there is no `irlba` or
ARPACK method menu and no Python bridge. For openTSNE initialization, CUDA uses
native RAPIDS RAFT TSVD compiled into the package CUDA backend and fails loudly
if that support is unavailable. Float32 CUDA input is passed to the native fit
without materializing an R double matrix, and float32 scores/loadings are
returned. Metal uses a native float32 block-subspace TSVD
with MPS matrix products and a resident workspace. CPU uses fastEmbedR's native
float32 blocked RSVD, with a seeded Gaussian sketch, quality-preserving
oversampling, and one or two subspace iterations according to requested rank.

For `backend = "cpu"`, `n.cores` is a positive integer that temporarily
sets the BLAS/OpenMP thread limit. The result records
`n.cores_requested`, `n.cores_effective`, and `core_control`.
Single-threaded BLAS builds cannot use additional cores and report one
effective thread. Metal and CUDA ignore this CPU-only argument.

The ordinary PCA scores are always retained in `pca_fit$scores`.
`opentsne_init = TRUE` adds a second matrix, centered and scaled so that its
largest component standard deviation is `1e-4`; no second decomposition is
performed.

Supplying `xtest` adds projected test coordinates in
`pca_fit$scores_test`. fastEmbedR intentionally omits an SVD method selector.

`irlba` is not imported, suggested, linked, or called by fastEmbedR. It is
used only by external benchmark scripts as a CPU reference.

## Explicit GPU Use

GPU use is explicit. A request for Metal or CUDA must run that backend or fail
clearly.

```r
fit <- opentsne(x, perplexity = 30, backend = "metal", seed = 1)
layout <- fit$layout
```

For CUDA builds with RAPIDS cuVS available:

```r
fit <- opentsne(x, perplexity = 50, backend = "cuda", seed = 1)
```

The package does not silently run these examples on CPU and report them as GPU
results.

## Graph Clustering

Build a graph from raw data, an embedding result, or a reusable KNN object:

```r
graph <- knn_graph(x, k = 20, weight = "snn", n.cores = 4)
communities <- graph_cluster(graph, method = "leiden", seed = 1)
table(communities$membership)
```

For an embedding-space graph, pass the fit directly:

```r
fit <- opentsne(x, perplexity = 15, backend = "cpu", seed = 1)
graph <- knn_graph(fit, k = 20, weight = "snn", n.cores = 4)
communities <- graph_cluster(graph, method = "leiden", seed = 1)
```

`knn_graph()` uses the selected backend only when it must compute neighbors.
`graph_cluster()` runs native CPU, CUDA, or Metal Louvain/Leiden without
calling igraph, cuGraph, Python, or another clustering routine. Walktrap is
CPU-only. An unavailable or unsupported GPU backend fails explicitly.
Walktrap is intended for small and moderate graphs because its transition
matrix is quadratic; use Leiden or Louvain for large graphs.

## Landmark Workflow

For reusable landmark models, keep the three stages explicit:

```r
selection <- select_landmarks(x, landmarks = 0.5, seed = 1)

model <- fit_landmark_model(
  x,
  selection,
  method = "umap",
  n_neighbors = 30,
  graph_mode = "fuzzy",
  backend = "cpu",
  n.cores = 4,
  seed = 1
)

fit <- project_landmark_model(
  model,
  x,
  refinement_epochs = 50,
  n.cores = 4
)
```

The reference fit uses the same `umap()` implementation, graph construction,
optimizer, and parameter values as an ordinary full UMAP run. For openTSNE,
set `method = "opentsne"` and pass `perplexity`. The resulting model can also
project a separate matrix of new observations in the same feature space.

`precompute_query_knn(reference, query, ...)` exposes the query-only search
used by the projection stage. It searches the fixed reference only and avoids
constructing unnecessary query-to-query neighbors.

The one-call wrapper remains available:

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
  graph_mode = "fuzzy",
  backend = "cpu",
  seed = 1
)
plot(fit)
```

CPU projection uses native recall-tuned HNSW. Metal uses native exact search
for small references and recall-tuned IVF-Flat for larger references, followed
by native fixed-reference transform kernels. CUDA uses native exact search for
smaller references and IVF-Flat for larger references; its KNN result remains
device resident for the projection and refinement stages.

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
supplied neighbor graph.

## Public API

| Function | Purpose |
| --- | --- |
| `precompute_knn()` | Package-native non-self KNN search with CPU, Metal, or CUDA backend. |
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
| `knn_graph()` | Compact graph from data, an embedding, or supplied KNN. |
| `graph_cluster()` | Native Louvain, Leiden, or Pons-Latapy Walktrap communities. |
