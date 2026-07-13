# Performance-Engineering And Improvement Strategy

This text is the version-controlled source for the performance-engineering
revision inserted into the fastEmbedR software manuscript. It describes only
changes retained in the current implementation. Experimental branches that
were slower or changed the accepted embedding are identified explicitly and
are not exposed as package options.

## Methods: performance engineering and change acceptance

Optimization was performed as a sequence of controlled implementation changes
rather than by reducing the mathematical workload. Each candidate retained the
same observations, supplied KNN graph, distance metric, k or perplexity,
initialization, random seed, t-SNE iteration schedule, UMAP epoch schedule,
learning-rate and momentum rules, FFT-grid resolution, graph mode, and
negative-sampling policy. KNN and PCA initialization were cached when the
embedding kernel was the object of study. This design separates improvements
in data movement, memory layout, synchronization, and kernel scheduling from
changes that could make an algorithm appear faster by performing less work.

Candidate changes were accepted only after repeated wall-clock measurements
and three quality gates. First, deterministic CPU operations had to reproduce
the accepted coordinates when operation order was unchanged. Second, parallel
Metal and CUDA outputs had to remain within the variation observed between
repeated atomic GPU runs, evaluated by coordinate correlation,
Procrustes-aligned RMSD, neighbourhood overlap, and sampled distance
correlation. Third, label-coloured layouts from the complete dataset were
inspected visually for cluster splitting, collapse, artificial holes,
outliers, and loss of local organization. A slower candidate or a candidate
with weaker visual or numerical agreement was removed instead of becoming a
public option.

The CPU openTSNE path was improved by creating one reusable worker team per
embedding, caching the FFT plan, retaining per-worker column scratch, and
parallelizing independent grid-support operations. Point splatting retained
its stable accumulation order because private-grid reduction increased memory
traffic and was slower on full MNIST70k. UMAP uses compact sparse graph arrays
and precomputed edge schedules; CPU workers share immutable graph data and
contiguous float32 optimizer buffers.

The Metal path keeps layouts, gains, updates, grid fields, spectra, and force
buffers resident in unified GPU memory. Sixteen unchanged openTSNE iterations
are encoded per command buffer to reduce submission and synchronization
overhead. For PCA initialization, a float32 input matrix is copied directly
into Metal storage; a native reduction validates, centres, and optionally
scales each feature in place before MPS block-subspace TSVD matrix products.
Float32 input now returns actual float32 scores and loadings, and float32 PCA
scores are normalized to the t-SNE initialization scale without creating a
double score matrix.

The CUDA openTSNE implementation retains cuFFT plans, FFT work arrays,
affinities, coordinates, gains, updates, and reduction buffers on the device.
Twenty-five unchanged iterations are captured in CUDA Graph executables for
the early-exaggeration, normal, and tail phases. The CUDA UMAP and openTSNE
one-call paths can consume package-owned GPU KNN buffers directly, avoiding R
index and distance matrices. CUDA graph construction, scheduling, optimizer
state, and final layouts use compact integer and float32 device buffers, and
only the final embedding is returned to R.

## Results: implementation-level validation

On flattened MNIST70k with cached KNN and a fixed initialization, the retained
CPU openTSNE changes reduced median embedding time from 16.259 to 14.233
seconds (12.5%) with coordinate correlation 1.000 and neighbourhood overlap at
15 equal to 1.000. The retained Metal command batching reduced median
embedding time from 2.602 to 2.545 seconds (2.2%); its coordinate correlation,
normalized Procrustes RMSD, and neighbourhood overlap at 15 remained within
the variation of repeated atomic Metal runs. CUDA Graph capture reduced median
CUDA openTSNE embedding time from 3.293 to 2.534 seconds (23.1%) and median
full-call time from 5.234 to 4.534 seconds (13.4%), with coordinate correlation
0.999691 and normalized Procrustes RMSD 0.024861 against the accepted baseline.
CPU/Metal and CUDA candidates were measured in independent paired development
sessions. The inference is therefore the baseline-to-retained change within a
row; absolute times should not be compared across rows or with a separately
generated publication figure.

Moving float32 PCA preprocessing to Metal reduced warm median MNIST70k PCA
time from 0.429 to 0.104 seconds, a 75.8% reduction or 4.1-fold speed-up. The
first invocation, which includes Metal pipeline compilation, required 0.774
seconds and is reported separately. Against the CPU randomized-SVD reference,
the retained Metal loading subspace had minimum canonical correlation 0.9980,
maximum principal angle 3.65 degrees, and score-distance Pearson correlation
0.9977. The 70,000 by 2 float32 score object occupied approximately 0.56 MB,
half the numeric payload of its double counterpart.

Several plausible changes were tested and removed. CPU point splatting with
private per-thread grids increased full-MNIST median time to 14.897 seconds.
Fusing the Metal post-update centering stage increased median embedding time to
2.613 seconds. Moving the small PCA orthonormalization to a separate Metal
kernel increased median PCA time to 0.449 seconds. These negative results
support the package policy of retaining only measured improvements and keeping
failed experiments out of the user API.

## Discussion: interpretation and next steps

The measurements show that the most useful optimization depends on the
backend. CPU performance is constrained mainly by memory locality and the
parallel fraction of sparse and FFT-grid work; on the tested Apple M3,
openTSNE improved from 46.988 seconds with one thread to 14.304 seconds with
four threads, while additional threads produced a smaller gain because the
processor has four performance cores. Metal PCA benefited substantially from
eliminating a host-side scan of the full input matrix, whereas the already
GPU-resident Metal openTSNE loop showed a smaller gain from reduced command
submission. CUDA benefited most from repeated-kernel launch amortization and
device-resident data flow.

These implementation-level measurements are diagnostics, not replacements for
the manuscript's primary cross-package benchmark. Cross-package comparisons
continue to report total user-level elapsed time because reference packages do
not expose equivalent precomputed-KNN and embedding-only boundaries. The next
optimization targets are therefore selected by backend profiles: sparse-force
locality on CPU, sparse-attraction memory access on Metal, and graph/affinity
setup primitives on CUDA. Any future change remains subject to the same fixed-
workload, numerical-agreement, and full-layout visual-inspection gates.
