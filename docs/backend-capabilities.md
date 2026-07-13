# Backend Capabilities

This page states what each backend does and what it does not do. The central
rule is simple: if a function is requested with `backend = "metal"` or
`backend = "cuda"`, it must run a real native GPU path or fail clearly.

## Capability Matrix

| Function | CPU | Metal | CUDA | Notes |
| --- | --- | --- | --- | --- |
| Internal one-call KNN | native float32 HNSW | native exact or recall-tuned IVF-Flat | direct FAISS GPU exact or RAPIDS cuVS IVF-Flat | KNN selection is internal; fastEmbedR does not export an `nn()` function. |
| `faissR::nn()` | optional public FAISS CPU search | not used for native Metal | optional RAPIDS cuVS / FAISS GPU search | Use faissR when a reusable public KNN object is required. |
| `umap_knn()` | native C++ CSR graph and optimizer | native Metal `atomic_inplace` optimizer | native CUDA pure-atomic optimizer | Metal/CUDA optimizers use the supplied graph; unavailable GPU backends fail clearly. |
| `umap()` | native HNSW, then `umap_knn()` | native exact/IVF-Flat, then native Metal UMAP | native FAISS/cuVS device KNN, then native CUDA UMAP | CUDA KNN is not copied through R. Metal IVF exact-reranks candidates in the original dimensions and records its pilot recall. |
| `opentsne_knn()` | native C++ FFT-grid optimizer | native Metal FFT-grid optimizer | native CUDA FFT-grid optimizer using cuFFT | Use `Y_init` or `init_data` for explicit PCA initialization. |
| `opentsne()` | native HNSW, then `opentsne_knn()` | native exact/IVF-Flat, then Metal openTSNE | native FAISS/cuVS device KNN, then CUDA openTSNE | The package does not call Python openTSNE in public functions. |
| `pca()` / openTSNE PCA init | fastPLS-style RSVD | native float32 MPS block-subspace TSVD | native RAPIDS RAFT TSVD when compiled | GPU requests do not route through Python or silently fall back to CPU. |
| `transform_tsne()` | native fixed-reference transform | native Metal projection/transform kernels where available | native CUDA projection/transform kernels where built | Used by openTSNE landmarking. |
| `landmark_umap()` | native landmark embed/project/refine | native Metal projection/refinement kernels where available | native CUDA projection/refinement kernels where built | Landmarking is an explicit approximation, not a replacement for full UMAP. |
| `landmark_tsne()` | native landmark embed plus transform | native Metal projection/transform kernels where available | native CUDA projection/transform kernels where built | Projection quality is tracked separately in benchmark plots. |
| `evaluate_embedding()` | native/R quality metrics | CPU metrics after final layout transfer | CPU metrics after final layout transfer | Metrics are not labelled as GPU work. |

## Distance Metrics

| Metric | CPU | Metal | CUDA | Notes |
| --- | --- | --- | --- | --- |
| `euclidean` | native HNSW | native exact/IVF-Flat | FAISS GPU exact or cuVS IVF-Flat | Validated default for UMAP/openTSNE. |
| `cosine` | row normalization plus native HNSW | row normalization plus native exact/IVF-Flat | row normalization plus native cuVS | Candidate selection is approximate; returned distances use the transformed full-dimensional vectors. |
| `correlation` | row centering/normalization plus native HNSW | row centering/normalization plus native exact/IVF-Flat | row centering/normalization plus native cuVS | Correlation is represented by Euclidean distance on centered unit rows. |
| `inner_product` | faissR when installed | not supported by native Metal KNN | native cuVS | Explicit unsupported Metal requests fail; they do not fall back to CPU. |

## Backend Labels

Every benchmark row should record the backend requested and the backend used.
The package avoids ambiguous GPU reporting:

- `backend_requested = "metal"` and `backend_used = "metal"` means a native
  Metal path ran.
- `backend_requested = "cuda"` and `backend_used = "cuda"` means a native CUDA
  path ran.
- If the GPU path is unavailable, status should be `backend_unavailable` or
  `not_supported`, not a hidden CPU result.

## CPU

CPU paths are native C++ and use `n_threads` where the operation is safe to
parallelize. Current CPU priorities are:

- reuse KNN graphs instead of recomputing them;
- keep graph data in compact integer/float arrays;
- use CSR graph storage for UMAP;
- keep openTSNE attractive affinities sparse;
- avoid unnecessary copies between R matrices and C++ buffers.

## Metal

Metal is implemented with Objective-C++ and Metal kernels. Public Metal and CUDA
UMAP/openTSNE paths do not call Python, Torch, MLX, or `reticulate`.

Metal one-call KNN uses exact search below the internal size threshold and
IVF-Flat above it. Large, high-dimensional IVF searches use a 128-dimensional
signed float32 projection, four native centroid passes, a four-SIMD-group
projected shortlist, and full-dimensional exact reranking. The shortlist grows
internally from 288 to 384 or 512 candidates when needed. Four deterministic
pilot strata select shortlist size and `nprobe`; direct Metal reranking is the
safety fallback. Failure to reach the requested pilot recall is reported
instead of silently switching to CPU.

The validated UMAP Metal path is `atomic_inplace`; other slower or distorted
Metal UMAP optimizer experiments were removed from the public API.

The validated openTSNE Metal path is the package-native FFT-grid path with PCA
initialization. Its PCA initialization uses a single resident float32 workspace,
MPS matrix products, a small float32 eigensolve, and one final Metal score
projection. Standalone MPSGraph FFT diagnostic helpers were removed after MNIST
tests did not justify exposing them as package features.

## CUDA

CUDA support is optional at build time. The package can use:

- direct FAISS GPU `bfKnn` for exact KNN and RAPIDS cuVS C API for IVF-Flat;
- native CUDA UMAP kernels;
- native CUDA FFT-grid openTSNE kernels with cuFFT.

FAISS, cuVS, CUDA, and cuFFT are not vendored into the package. They must be installed
on the CUDA machine and matched to the driver/toolkit stack. If CUDA is not
available, explicit CUDA requests fail clearly.

## External Libraries

`fastEmbedR` does not vendor the full FAISS or RAPIDS libraries. It contains a
small FAISS-derived HNSW implementation and a native Metal IVF design informed
by FAISS and Faiss-mlx; their permissive licenses and pinned source commits are
installed under `inst/LICENSES/`. Installed FAISS GPU and cuVS libraries are
linked directly by optional CUDA builds; fastEmbedR does not vendor them.
