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

## Current Review Status

`fastEmbedR` is under review in
[BiocContributions issue 142](https://github.com/Bioconductor/BiocContributions/issues/142).
The version 0.99.7 review build completed with 0 errors, 1 warning, and 5 notes.
The warning was:

> No Bioconductor dependencies detected. Note that some infrastructure
> packages may not have Bioconductor dependencies.

This is a submission-scope question rather than a compilation or correctness
failure. `fastEmbedR` is intended as infrastructure for dimensionality
reduction of large biological data, including single-cell, flow-cytometry,
gene-expression, and other high-dimensional assays. Its native CPU path is
deliberately standalone, however, and no Bioconductor package provides a
legitimate mandatory runtime API dependency. An artificial dependency will
therefore not be added merely to silence the check. Eligibility and any
required scope clarification should be decided with the assigned
Bioconductor reviewer. Version 0.99.9 therefore declares the accurate
`Infrastructure` biocView. BiocCheck treats dependency-independent
infrastructure packages as a note rather than a warning; no artificial package
dependency is added.

The accompanying notes are triaged as follows:

- a runnable example is provided for `fastEmbedR_backend()`;
- maintainer ORCID and funder metadata will be added only when verified;
- the suggested `ATACSeq` and `DNASeq` views are not used because fastEmbedR
  is not specific to either assay;
- condition handling is written without nonlocal assignment in the openTSNE
  PCA initialization path;
- remaining findings about long functions, line width, indentation,
  `suppressWarnings()`, and closure state are tracked as maintainability work.
  They should be changed incrementally with CPU, Metal, and CUDA regression
  tests rather than by a high-churn formatting or control-flow rewrite during
  review.

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
FASTEMBEDR_USE_CUDA=0 R CMD check --as-cran fastEmbedR_0.99.9.tar.gz
```

GPU-enabled builds should be validated separately on machines with the relevant
toolchains, because Bioconductor build machines should not be assumed to have
CUDA or Apple Metal.

The local submission preflight used for this repository is:

```sh
LC_ALL=C \
FASTEMBEDR_USE_CUDA=0 \
R CMD check --no-manual --no-build-vignettes fastEmbedR_0.99.9.tar.gz

LC_ALL=C \
Rscript -e 'BiocCheck::BiocCheck("fastEmbedR_0.99.9.tar.gz", `quit-with-status`=FALSE)'
```

The `--no-build-vignettes` check mode is useful during development, but it
reports vignette-output warnings because `inst/doc` is intentionally not built.
The final submission should build vignettes.

Current Bioconductor-specific follow-up items:

- register and validate the maintainer email on the Bioconductor Support Site;
- obtain the assigned reviewer's decision on eligibility without a mandatory
  Bioconductor dependency;
- add an ORCID to `Authors@R` when available;
- audit targeted warning suppression and closure state;
- gradually shorten very long R helper functions as maintenance work.
