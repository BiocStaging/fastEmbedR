# Bioconductor Readiness

[Home](../README.md) |
[Installation](installation.md) |
**Bioconductor** |
[Implementation](implementation.md) |
[Examples](examples.md) |
[Benchmarks](benchmarks.md) |
[API](usage-api.md) |
[Reproducibility](reproducibility.md) |
[References](references.md)

This page records the dependency boundary used for a Bioconductor-friendly
submission of `fastEmbedR`.

## Dependency Classes

| Class | Dependency | Role | Required For Core Build |
| --- | --- | --- | --- |
| R package | `Rcpp` | R/C++ interface and exported native routines. | yes |
| R package | `BiocStyle` | Vignette rendering in Bioconductor style. | suggested |
| R package | `float` | Optional float32 R matrices and reduced host memory use. | suggested |
| R package | `jsonlite` | Optional benchmark and reproducibility metadata serialization. | suggested |
| R package | `knitr`, `rmarkdown` | Vignette and documentation rendering. | suggested |
| R package | `testthat` | Unit tests. | suggested |
| R package | `igraph` | Optional graph-clustering validation and examples. | suggested |
| R package | `Rtsne`, `uwot`, `umap` | Optional reference benchmarks only. | suggested |
| Companion package | `fastPLS >= 0.99.3` | Preferred optional randomized-SVD PCA provider for CPU; Metal and compiled CUDA initialization use native fastEmbedR backends. | optional enhancement |
| System library | C++17 compiler | Native CPU code and numerical helper compilation. | yes |
| System library | Apple Metal framework | Native Metal KNN and embedding backends on macOS. | optional |
| System library | CUDA Toolkit, FAISS GPU, cuFFT, cuBLAS, cuSOLVER, RAPIDS RAFT and cuVS C libraries | Native CUDA KNN, embedding backend, and CUDA TSVD initialization. | optional |

`fastEmbedR` does not vendor the full FAISS, cuVS, RAFT, or cuML libraries, or
`uwot`, `Rtsne`, or Python openTSNE source. Its compact FAISS-derived HNSW and
Faiss-mlx-informed Metal files retain their permissive licenses under
`inst/LICENSES/`.

## Native KNN Boundary

fastEmbedR owns the internal CPU/Metal KNN and direct FAISS/cuVS CUDA KNN used
by one-call embeddings. The following are available without another KNN R
package:

- `opentsne_knn()` works from supplied neighbor indices and distances;
- `umap_knn()` works from supplied neighbor indices and distances;
- `prepare_opentsne_knn()` and `prepare_umap_knn()` can prepare reusable
  native embedding inputs;
- `knn_graph()` and `graph_cluster()` provide native graph construction and
  community detection;
- `evaluate_embedding()` and plotting helpers use package-native routines.

CPU, Metal, and a correctly compiled CUDA one-call build are self-contained at
the R package level.
CUDA requests fail explicitly when cuVS is not linked.

## Backend Policy

The public embedding backend argument is intentionally small:

```r
backend = "cpu"
backend = "metal"
backend = "cuda"
```

An explicit GPU request must use the requested GPU backend. The package must
not run on CPU while reporting Metal or CUDA. Optional GPU code is compiled and
tested when the corresponding toolchain is available.

## Submission Checklist

- No `Remotes` field in `DESCRIPTION`.
- No vendored FAISS/cuVS/RAPIDS source or binary libraries in the R package.
- Large data, benchmark outputs, and container images are excluded by
  `.Rbuildignore`.
- Examples and vignettes use small built-in data or guard optional packages
  with `requireNamespace()`.
- Optional reference benchmarks (`Rtsne`, `uwot`, `umap`) are in `Suggests`,
  not `Imports`.
- CUDA and Metal failures are explicit, not silent CPU fallbacks.
- The maintainer email should be registered on the Bioconductor Support Site
  before submission.

## Minimal Bioconductor Check

A CPU-only check should be possible without FAISS/cuVS installed:

```sh
LC_ALL=C \
FASTEMBEDR_USE_CUDA=0 R CMD build .

LC_ALL=C \
FASTEMBEDR_USE_CUDA=0 R CMD check --as-cran fastEmbedR_0.99.0.tar.gz
```

GPU-enabled builds should be validated separately on machines with the relevant
toolchains, because Bioconductor build machines should not be assumed to have
CUDA or Apple Metal.

The local submission preflight used for this repository is:

```sh
LC_ALL=C \
FASTEMBEDR_USE_CUDA=0 \
R CMD check --no-manual --no-build-vignettes fastEmbedR_0.99.0.tar.gz

LC_ALL=C \
Rscript -e 'BiocCheck::BiocCheck("fastEmbedR_0.99.0.tar.gz", `quit-with-status`=FALSE)'
```

The `--no-build-vignettes` check mode is useful during development, but it
reports vignette-output warnings because `inst/doc` is intentionally not built.
The final submission should build vignettes.

Current Bioconductor-specific follow-up items:

- register and validate the maintainer email on the Bioconductor Support Site;
- consider whether the package should instead be submitted to CRAN if the
  Bioconductor review requires a runtime dependency on another Bioconductor
  package;
- add an ORCID to `Authors@R` when available;
- reduce or justify `set.seed()` usage in package code;
- gradually shorten very long R helper functions as maintenance work.
