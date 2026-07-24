# Implementation

[Home](../README.md) |
[Installation](installation.md) |
[Bioconductor](bioconductor.md) |
**Implementation** |
[Performance Engineering](backend-performance-engineering.md) |
[Examples](examples.md) |
[Benchmarks](benchmarks.md) |
[API](usage-api.md) |
[Reproducibility](reproducibility.md) |
[References](references.md)

`fastEmbedR` implements two nonlinear embedding families: UMAP and an
openTSNE-style t-SNE. UMAP follows the fuzzy simplicial-set graph formulation
introduced by McInnes and colleagues [7,13]. The t-SNE path follows the
probabilistic neighbour-embedding objective of van der Maaten and Hinton [1],
with modern openTSNE/FIt-SNE-style optimization and interpolation ideas [3-4].
The package is intentionally KNN-first. fastEmbedR implements its CPU HNSW and
Apple Metal exact/IVF-Flat one-call KNN paths natively. CUDA builds link
directly to FAISS GPU for exact search and to the Apache-2.0 RAPIDS cuVS C API
for IVF-Flat; they do not call another R package, Python, or `reticulate`. Graph/affinity construction,
initialization, stochastic optimization, native fixed-reference transforms,
backend reporting, and quality metrics remain inside fastEmbedR.

The public package surface is deliberately small:

- `precompute_knn()` exposes the same native backend policy used by the
  one-call functions, while keeping algorithm and recall tuning internal.
- `opentsne_knn()` and `umap_knn()` consume a supplied KNN object.
- `opentsne()` and `umap()` select native CPU/Metal KNN or direct FAISS/cuVS CUDA KNN and
  then call the corresponding KNN entry point.
- `pca()` computes backend-native truncated PCA scores and loadings, including
  a resident float32 Metal/MPS TSVD path.
- `knn_graph()` builds one compact undirected graph from data, an embedding,
  or supplied neighbours; `graph_cluster()` applies native Louvain, Leiden,
  or Walktrap community detection.
- `backend` is limited to `"cpu"`, `"metal"`, and `"cuda"`.
- GPU requests fail clearly if native GPU code is unavailable. CPU fallback is
  never reported as Metal or CUDA work.

## Compilation And Numerical Contract

The portable numerical core requests C++17 and adds only `-pthread`; it
inherits the compiler command, optimization level, ABI, and linker flags from
the active R installation. This is part of the algorithmic contract, not only
an installation detail. HNSW construction, sparse graph creation, and
stochastic embedding contain floating-point comparisons whose tie decisions
can change under unsafe reassociation. fastEmbedR therefore does not impose
global `-ffast-math`, `-march=native`, or `-O3` flags.

On macOS, `configure` adds the Apple Foundation, Metal, MPS, and MPSGraph
frameworks and compiles the Objective-C++ bridge with the Xcode SDK. Metal
shader source is compiled by the runtime for the active GPU. The native Metal
KNN shader enables Apple's float32 fast arithmetic locally; this is not
propagated to the CPU core or to UMAP/openTSNE host code.

CUDA translation units are compiled by NVCC as C++17 with extended lambdas,
relaxed `constexpr`, and position-independent host code. The deployment
architectures are explicit through `FASTEMBEDR_CUDA_ARCH`. `CUDAHOSTCXX`
selects an R-ABI-compatible host compiler when CUDA and R come from different
toolchain prefixes. FAISS GPU, cuVS, RAFT, and their CUDA dependencies must be
built for the same devices; adding an architecture to fastEmbedR cannot add
missing kernels to a linked library.

The complete commands, architecture examples, diagnostics, and failure modes
are documented in
[Installation And Native Compiler Configuration](installation-backends.md).
The benchmark reproducibility bundle records the resolved `R CMD config`
values, generated `src/Makevars`, compiler versions, Xcode/Metal or
CUDA/NVCC versions, and relevant environment flags.

## Native Implementation Validation

Because fastEmbedR implements the embedding path natively rather than calling
Python openTSNE, the package includes a small reference-validation workflow.
The script
[`tools/validate_reference_implementations.R`](../tools/validate_reference_implementations.R)
uses exact KNN on iris, fixes the random seed, and compares package-native
outputs with established R references.

For t-SNE, the validation compares `opentsne_knn()` with
`Rtsne::Rtsne_neighbors()` from the same exact KNN matrix and PCA
initialization. It records trustworthiness, nearest-neighbour preservation,
embedding-space KNN label accuracy, the final KL/cost value where exposed, and
Procrustes-aligned similarity between the two embeddings [1-4,11]. On a small
dataset the script uses the exact negative-gradient diagnostic path rather
than the large-data FFT-grid approximation, because this isolates objective
correctness from interpolation-grid engineering.

For UMAP, the validation compares `umap_knn(..., graph_mode = "fuzzy")` with
`uwot::umap()` and compares the sparse graph returned by
`prepare_umap_knn()` with `uwot::similarity_graph()` [7,10,13]. The graph check
reports edge overlap and Spearman correlation of common-edge weights. These
checks are intended to show behavioural agreement, not bitwise identity:
stochastic optimizers, spectral initialization details, and parallel floating
point reductions can legitimately rotate, reflect, scale, or slightly deform
the final layout.

## Nearest-Neighbour Layer

`fastEmbedR` exports a focused precomputation boundary rather than a general
nearest-neighbour algorithm menu. `precompute_knn()` exposes `k`, metric,
backend, and CPU thread count; it applies the same internally selected search
policy as the one-call embedding functions. The KNN-input functions also
accept a plain host list of indices and distances from another implementation.

The CPU implementation distils the HNSW organization in FAISS 1.14.3 [8]:
exponentially sampled hierarchy levels, greedy descent through upper layers,
bounded `efConstruction` expansion at each insertion layer, diversity-aware
neighbour pruning, reciprocal graph links, and independent parallel queries.
All vectors, graph distances, and search buffers are float32; indices are
int32. A large/high-dimensional target-0.99 policy uses `M = 10`,
`efConstruction = 40`, and `efSearch = 40`, selected only after comparison
against the original FAISS HNSW output. Other shapes use a more conservative
graph. The package records the selected values and does not claim bitwise
identity with FAISS.

Construction uses one persistent worker team, reusable visit tables and
bounded heaps, early-exit squared-distance comparisons, and parallel
reciprocal-row updates. Temporary construction distances are released before
query. These changes preserve the selected graph and query output: exact KNN
indices and distances matched the pre-optimization implementation on
MNIST70k, Fashion-MNIST70k, USPS, and MetRef. On the validated four-thread
Linux environment, MNIST70k construction decreased from 23.730 to 17.345
seconds and complete KNN time from 27.653 to 21.288 seconds. A separate
`-O3 -march=native` build was slower and was rejected, which is why compiler
tuning is not presented as an algorithmic improvement.

The Metal implementation has two routes. Exact search assigns one SIMD group
to each candidate distance and merges per-group top-k lists on device. IVF-Flat
first forms a deterministic 128-dimensional signed float32 projection for
coarse routing. Four native centroid-assignment/update passes then pack the
inverted lists. For large, high-dimensional inputs, a four-SIMD-group kernel
scans selected lists in projected space and retains an internal shortlist. It
starts at 288 candidates and can expand to 384 or 512 candidates. A second
kernel exact-reranks only that shortlist using the original full-dimensional
vectors, in bounded query batches. Four deterministic 64-row pilot strata
spread across the dataset jointly select shortlist size and `nprobe`. If no
shortlist reaches the requested recall tier plus a generalization margin, the
same Metal backend uses direct exact reranking of selected lists. Candidate
routing is approximate; every returned neighbour and distance is based on
full-dimensional exact reranking. This fused list-scan/top-k organization was
informed by FAISS [8] and MLXPorts/Faiss-mlx. Their exact commits and
permissive licenses are retained under `inst/LICENSES/`.

For reproducibility, the one-call API fixes only the requested KNN device class
and a small/large data policy:

- CPU one-call embeddings use native HNSW. Metal uses native exact KNN below
  4,096 observations and recall-tuned IVF-Flat for larger inputs.
- CUDA one-call embeddings use direct FAISS GPU `bfKnn` below 100,000 samples
  and recall-tuned cuVS IVF-Flat at or above 100,000 samples. Exact float32
  Euclidean/IP input is uploaded in its existing R column-major layout because
  FAISS accepts column-major vectors; this removes a full host transpose and
  its temporary buffer without changing distances or neighbours. IVF starts from a
  deterministic shape rule, compares evenly spaced pilot queries against a
  cuVS exact oracle, and expands `nprobe` until it reaches the requested recall
  tier with a small safety margin. It records pilot recall, `nlist`, `nprobe`,
  and the number of tuning attempts.
- FAISS/cuVS write int64 row-major search output on the device. One package CUDA
  kernel removes self-neighbours, converts indices to int32/one-based form,
  transforms squared distances, and packs the result directly into the
  column-major device layout consumed by UMAP and openTSNE. No KNN matrix is
  materialized in R during a one-call CUDA embedding.
- KNN-input functions accept the index and distance matrices as already
  measured data and do not repeat neighbour search.

This separation makes benchmark timing interpretable: KNN time, affinity/graph
construction time, embedding time, and projection/transform time can be
reported separately.

## KNN Graphs And Community Detection

`knn_graph()` deliberately reuses the package's existing nearest-neighbour
boundary. It accepts raw data, a `fastEmbedR_embedding`, or supplied neighbour
indices and distances; it does not expose another KNN algorithm selector.
Graph construction is package-native C++ and produces one compact undirected
edge list. The supported weights are Jaccard shared-neighbour similarity on
observed KNN edges, inverse distance, and binary adjacency. Reciprocal-edge
filtering and a final weight threshold are optional. Directed duplicates are
collapsed once, so clustering does not repeat neighbour search or retain a
dense adjacency matrix.

`graph_cluster()` provides three CPU algorithms:

- multilevel Louvain local moving and graph aggregation [14];
- Leiden local moving, within-community refinement, and aggregation [15];
- the Pons-Latapy Walktrap random-walk distance and adjacent-community
  agglomeration [16].

The Leiden phase organization is informed by the MIT-licensed NetworKit
implementation [17], but fastEmbedR uses its own compact graph representation
and does not link to NetworKit. Walktrap is the published random-walk method,
not a sampled proximity approximation. Its exact transition storage is
quadratic, so the implementation stops before an unsafe allocation and directs
large graphs to Louvain or Leiden. `igraph` is used only as a guarded test
oracle. External clustering implementations are neither linked nor called.

The clustering backend is reported as CPU. CUDA or Metal can accelerate the
KNN stage when `knn_graph()` starts from data, but fastEmbedR does not label
that as GPU community detection.

## PCA And Truncated-SVD Initialization

`fastEmbedR::pca()` provides a small PCA API for reusable scores, loadings, and
openTSNE initialization. The API has no method-selection layer and does not
call `irlba`, ARPACK, Python, or `reticulate`.

The backend implementations share a randomized subspace objective but use
hardware-appropriate execution:

| Backend | PCA implementation |
| --- | --- |
| CPU | Native R/C++ RSVD matrix products using BLAS-backed `%*%`/`crossprod`. |
| Metal | Package-native float32 block-subspace TSVD using MPS matrix multiplication and a resident unified-memory workspace. |
| CUDA | Native C++/CUDA initialization through RAPIDS RAFT TSVD compiled into the CUDA backend. |

The Metal path is deliberately different from the former R-orchestrated RSVD.
It converts and centers the input once into a float32 buffer whose column-major
R layout is viewed as a row-major transposed matrix. A deterministic Gaussian
feature-space block is orthonormalized, two block power iterations are encoded
as MPS products, and only the small projected Gram matrix is returned to the
CPU for a float32 symmetric eigensolve. Final loadings and scores are projected
once on Metal. The input, projected block, basis, Gram matrix, loadings, and
scores remain allocated for the complete call, eliminating the repeated
full-matrix uploads and R intermediate matrices of the earlier Metal RSVD.

For openTSNE initialization, the input is mean-centered before decomposition
and the resulting scores are centered and scaled to the small t-SNE
initialization scale. CUDA acceleration for this step is native C++/CUDA
through RAPIDS RAFT TSVD compiled in the package CUDA translation unit. If
RAFT TSVD support is not compiled in, CUDA PCA initialization fails loudly
rather than falling back to a different implementation.

Set `opentsne_init = TRUE` in `pca()` to retain the ordinary PCA fit and add an
`opentsne_init` matrix derived from those same scores. The added matrix is
centered and rescaled so the maximum component standard deviation is `1e-4`,
matching the small-scale initialization expected by t-SNE/openTSNE optimizers
[1,3-4]. No second decomposition is performed. `opentsne_pca_init()` remains a
compact helper for users who need only the initialization matrix or an RDS
cache. Either result can be passed as `Y_init` to `opentsne_knn()` or
`opentsne()`.

## UMAP From KNN

UMAP is implemented as a sparse graph optimization from a supplied KNN matrix.
The implementation follows the UMAP fuzzy simplicial-set formulation [7,13]:
local bandwidths are estimated per observation, neighbour distances are
converted to directed membership strengths, the graph is symmetrized, and a
low-dimensional layout is optimized with attractive edge updates and sampled
repulsive updates.

### CPU Graph Construction

The CPU path is package-native C++:

1. Validate and normalize KNN indices and distances.
2. Estimate row-wise `rho` and `sigma` values by smooth KNN distance search.
3. Convert directed KNN distances to membership strengths.
4. Build the symmetric graph without dense intermediates.
5. Store compact edge arrays and epoch schedules for the optimizer.

The engineering goal is to preserve UMAP's graph mathematics while reducing
memory traffic. Internal distances and weights use `float` where safe; indices
are stored as integer buffers; and the implementation avoids duplicate graph
copies between R and C++.

### UMAP Optimizer

The optimizer uses stochastic edge sampling, negative sampling, a decaying
learning rate, and compact contiguous layout buffers, following the UMAP
objective rather than an exact copy of any R implementation [7,10,13]. The
current permissive implementation uses package-local samplers, random-number
generation, and update kernels rather than vendored `uwot` code. `uwot`
remains an external benchmark and behavioural reference, not a source
dependency [10].

UMAP exposes two graph modes:

| Mode | Meaning | Intended use |
| --- | --- | --- |
| `"fuzzy"` | Standard UMAP fuzzy graph weights. | Scientific comparison with the original UMAP model and `uwot`. |
| `"binary"` | Binary neighbour graph with the same optimizer. | Explicit approximation that may increase visible separation; always recorded in results. |

### Metal UMAP

The Metal backend is implemented in Objective-C++/Metal and uses the validated
atomic in-place edge-update kernel. It does not call Python, Torch, MLX, or
`reticulate`. The Metal optimizer consumes the same prepared graph as the CPU
path and returns only the final layout and metadata to R. The package-native
Metal FFT work used by openTSNE was informed by permissive Apple GPU FFT
engineering references [12].

### CUDA UMAP

The CUDA backend is compiled when `FASTEMBEDR_USE_CUDA=1` is enabled. The
public CUDA path uses the pure atomic optimizer. When
`FASTEMBEDR_USE_CUVS=1`, exact/IVF-Flat search is invoked through the native
cuVS C API, its KNN buffers remain on the selected CUDA device, and graph or
affinity construction consumes those pointers directly. The owning R object
contains external pointers with a finalizer; copying to ordinary R matrices is
an explicit diagnostic operation. No CPU or companion-package fallback is
used for a CUDA request.

## openTSNE-Style t-SNE From KNN

`opentsne_knn()` implements the t-SNE optimization structure used by modern
openTSNE/FIt-SNE workflows [3-4]:

1. Convert KNN distances to conditional probabilities by binary search on the
   Gaussian bandwidth for a target perplexity.
2. Symmetrize sparse probabilities and normalize the high-dimensional
   affinity matrix.
3. Initialize the embedding from the KNN-native default or an explicit
   user-supplied layout.
4. Run early exaggeration.
5. Run the normal optimization phase with adaptive gains, momentum, learning
   rate, update clipping, and recentering.
6. Return the layout with parameters, timing, and backend metadata.

The public `opentsne()` function is now only a convenience wrapper around this
KNN implementation. If a KNN object is supplied through `nn`, then
`opentsne(data, nn = knn)` calls the same path as `opentsne_knn(knn)` and
produces the same layout for the same seed and parameters. This mirrors the
KNN-input validation style used by R t-SNE tooling [11].

### Initialization

PCA initialization is explicit. Use:

```r
Y_init <- opentsne_pca_init(x, backend = "cpu")
y <- opentsne_knn(knn, Y_init = Y_init)
```

or pass `init_data` when a KNN-input run should compute PCA internally. The
old public `init = c("pca", "random")` argument was removed because hidden
initialization differences made KNN-input and one-call results difficult to
compare.

### Repulsive Force Approximation

The default large-data path uses an FFT-grid approximation inspired by
FIt-SNE [3]. Sparse attractive forces are evaluated from the KNN affinity
graph. The negative force is approximated by placing points on a
two-dimensional grid, convolving with the t-SNE kernel, and interpolating the
resulting force back to points [3-5]. The Barnes-Hut path is not part of the
public benchmark surface because the FFT-grid path is the intended standard
for MNIST70k-scale data; Barnes-Hut remains an important historical reference
for tree-based t-SNE acceleration [2].

### CPU, Metal, and CUDA

| Backend | Native implementation |
| --- | --- |
| CPU | C++ sparse affinities, FFT-grid repulsion, gains/momentum optimizer [3-4]. |
| Metal | Objective-C++/Metal scatter, FFT-grid convolution, gather, attractive-force, update, and centering kernels [3,12]. |
| CUDA | CUDA kernels with cuFFT for the FFT-grid convolution and device-side optimizer updates [3,5]. |

The Metal implementation includes package-native FFT kernels. Standalone
MPSGraph FFT diagnostics were tested and then removed because they did not
provide enough benefit to justify another user-facing backend.

The current CPU FFT-grid path reuses a scoped worker team, FFT bit-reversal and
root tables, and per-thread column scratch. The current Metal path encodes 16
unchanged optimization iterations per command buffer. The current CUDA path
captures chunks of 25 unchanged iterations in CUDA Graphs. These are
implementation-only optimizations: they do not reduce perplexity, grid size,
or iteration counts. Their profiling results, output-agreement gates, rejected
experiments, and exact benchmark artifacts are documented in
[Backend Performance Engineering](backend-performance-engineering.md).

## Landmarking

Landmarking is implemented as an explicit approximation. The package embeds a
subset, projects non-landmark observations using fixed-reference KNN
interpolation/transform steps, and records projection/transform time
separately. Landmarking is not used silently inside full `opentsne()` or
`umap()` calls.

## Autotuning

The API asks for few parameters, but selected defaults are saved in the output.
For openTSNE-style t-SNE, the native helper follows opt-SNE-inspired rules for
learning rate and iteration defaults [6]. For UMAP, the code uses the supplied
KNN and data size to choose internal initialization effort and optimizer
defaults without silently changing the supplied neighbour graph [7,13].

## License Boundary

The implementation is intended to remain compatible with the MIT license.
References such as `uwot`, `Rtsne`, openTSNE, FIt-SNE, t-SNE-CUDA, FAISS,
RAPIDS cuVS, and AppleSiliconFFT informed design and benchmarking decisions
[3-5,8-12]. GPL code is not vendored, linked, or required by the package.

See [References](references.md) for AACR-style citations and software
acknowledgements.
