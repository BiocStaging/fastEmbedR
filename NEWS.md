# fastEmbedR 0.99.1

- Moves publication benchmark and validation workflows to the separate
  `fastEmbedR-benchmark` repository, together with dataset acquisition and
  restricted-data instructions. Raw benchmark data and manuscript files are
  not distributed in either GitHub repository.
- Reduces the vignette-enabled source archive to approximately 1 MB by
  excluding benchmark, manuscript, container, and generated-result artifacts.
- Updates GitHub Pages and checkout workflows to Node 24-compatible action
  releases.

# fastEmbedR 0.99.0

- Makes the standard fuzzy UMAP graph the default for `umap()`, `umap_knn()`,
  UMAP initialization, and landmark UMAP. Binary weighting remains available
  only as the explicit `graph_mode = "binary"` sensitivity mode.
- Standardizes CPU parallelism controls on the public argument `n.cores`.
  Low-level `n_threads` names are now private implementation details; existing
  user scripts should replace `n_threads =` with `n.cores =`.
- Provides native UMAP and openTSNE-style optimizers for CPU, Apple Metal, and
  optional CUDA builds, with explicit backend reporting and no silent GPU to
  CPU fallback.
- Adds package-native Louvain and Leiden clustering for CPU, CUDA, and Metal.
  Accelerator local-moving and refinement phases use float32 compressed sparse
  row graphs and atomic community updates; package-owned C++ performs label
  compaction and graph coarsening between levels. No cuGraph source, library,
  Python module, or runtime symbol is required. Exact Pons-Latapy Walktrap
  remains CPU-only, and unsupported accelerator requests fail explicitly.
- Adds package-native float32 CPU HNSW, Apple Metal exact/recall-tuned IVF-Flat,
  direct FAISS GPU exact search, and direct RAPIDS cuVS CUDA IVF-Flat for one-call
  embeddings. CUDA results remain device-resident through graph or affinity
  construction and optimization; no faissR or CPU fallback is used.
- Distils the CUDA KNN adapter from the MIT-licensed faissR implementation into
  fastEmbedR. Exact search calls installed FAISS GPU directly; approximate
  search calls the installed Apache-2.0 cuVS C API. The package strips
  self-neighbours and packs int32/float32 output on device, and bounds IVF raw
  search storage to 32,768-query batches. `umap_knn()` and `opentsne_knn()`
  continue to accept reusable KNN results from any compatible provider.
- Validates CUDA IVF tuning against an exact cuVS pilot of evenly spaced rows
  and expands `nprobe` until the requested recall tier is reached; a failed
  target is never silently reported as tuned.
- Speeds the native Metal IVF path with a two-stage GPU search: a
  128-dimensional projected list scan builds an adaptive 288/384/512-candidate
  shortlist, followed by exact full-dimensional reranking. A four-stratum
  deterministic pilot selects both shortlist size and `nprobe`; direct
  reranking remains an internal safety fallback.
- Adds float32 input and output handling. Native graph weights, affinities,
  layouts, gradients, and optimizer buffers use float32; a double layout is
  returned only when the input was double.
- Replaces the R-orchestrated CPU and Metal RSVD initialization paths with
  package-native float32 implementations. CPU centers once into contiguous
  float32 storage, uses blocked SGEMM/subspace products, and retains only skinny
  QR and projected-SVD work at the ordinary R numerical boundary. Metal keeps
  the centered input, basis, projected block, loadings, and scores in one
  resident MPS workspace. Public CUDA `pca()` uses fastEmbedR's native RSVD;
  resident CUDA openTSNE initialization uses RAPIDS RAFT TSVD.
- Removes the optional fastPLS PCA delegation and `Enhances` dependency.
  CPU, Metal, and CUDA PCA ownership is now explicit and invariant to the
  packages installed in the user's library.
- Batches four openTSNE iterations per Metal command buffer, reducing command
  submission overhead without changing the optimization objective or schedule.
- Avoids allocating and uploading the padded `epochs_per_sample` buffer for
  the default Metal UMAP sampler, which does not read that schedule. The graph,
  edge sampling, random stream, and optimizer updates are unchanged.
- Corrects a race in a legacy multithreaded CPU UMAP optimizer path by using
  atomic coordinate updates.
- Replaces the former approximate structure score with the standard
  trustworthiness and continuity definitions, adds exact sampled neighbour-rank
  metrics, and samples before quadratic work to bound evaluation memory.
- Makes evaluation cache keys data-aware and removes self neighbours by row
  identity rather than assuming that self is always the first KNN column.
- Correctly decodes `float::float32` layouts before plotting and scoring.
- Preserves the caller's R random-number-generator state in seeded sampling,
  initialization, landmark selection, and evaluation helpers.
- Strengthens installed-package tests so optional CUDA tests skip individually
  while CPU, Metal, float32, and backend-contract tests continue to run.
