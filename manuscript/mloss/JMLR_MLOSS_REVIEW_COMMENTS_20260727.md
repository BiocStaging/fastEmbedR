# JMLR MLOSS Reviewer Comments

## Recommendation

**Minor revision before acceptance.**

`fastEmbedR` is a substantial software contribution rather than a thin
wrapper. The manuscript now distinguishes package-owned implementations from
external numerical primitives, documents the CPU, Metal, and CUDA execution
boundaries, and evaluates both total runtime and embedding quality. The main
text satisfies the four-page narrative limit; figures and references follow
those four pages.

## Principal Strengths

1. The software contribution is stated clearly. Native affinity/graph
   construction, optimization, float32 storage, backend validation,
   precomputed-KNN entry points, PCA initialization, and landmark transforms
   are separated from FAISS, cuVS, RAFT, cuFFT, and MPS primitives.
2. “openTSNE-style” is defined as algorithmic lineage rather than Python API,
   object, default, or coordinate compatibility.
3. Full-pipeline timing is used for cross-package claims, avoiding an unfair
   subtraction of nearest-neighbor or initialization stages that competing
   packages do not expose separately.
4. Runtime claims include dataset-level interquartile ranges and matched-pair
   rules. FlowRepository and ImageNet are explicitly excluded from unmatched
   comparative speedups.
5. Correctness validation now includes t-SNE affinity agreement and common-P
   KL divergence, UMAP graph-weight agreement, Procrustes alignment, and
   final-neighborhood agreement across CPU, Metal, CUDA, Python openTSNE, and
   uwot boundaries.
6. Binary UMAP is appropriately described as an adjacency-only sensitivity
   mode rather than standard UMAP or a guaranteed acceleration.
7. Landmarking is presented as a scalability approximation, with selection,
   interpolation, affine correction, refinement, timing, and quality changes
   reported explicitly.
8. The supplement provides method parameters, random seeds, initialization,
   metric, thread/device allocation, internal versus precomputed KNN
   boundaries, precision, hardware, compiler policy, and failure reporting.

## Required Before Final Submission

1. **Archive the release.** Replace the pending archival statement with a
   versioned release tag and Zenodo DOI. Deposit the main manuscript,
   supplement, source tarball, scripts, generated tables, and the
   machine-readable release-identity record.
2. **Close the historical provenance gap.** The numerical tables derive from
   archived `fastEmbedR` 0.99.0 runs whose manifests did not record a Git
   commit. The manuscript discloses this correctly, but the strongest final
   submission would rerun the comparative benchmark using the validated
   numerical tree (`58e39d5`) and image SHA-256
   `8a91df71be0dd4ab5a01caaccef159fa281a596d38127e62b22cd5cda846c6b9`.
   If a complete rerun is infeasible, preserve the limitation prominently and
   archive the original result bundle without implying that it was generated
   by the frozen image.
3. Replace placeholder JMLR metadata (`XX`, `MM/YY`, editor, and paper number)
   when assigned.

## Minor Editorial Checks

- Keep American spelling of “nearest-neighbor” throughout.
- Keep direct-Python fit time separate from R-mediated total-call time in
  captions, tables, and machine-readable outputs.
- Do not promote clustering to a principal contribution; its current
  supplementary placement is appropriate.
- Do not imply that backend-native PCA is always faster than `irlba`.
- Retain the statement that fixed seeds do not guarantee bitwise GPU
  reproducibility because atomic-update order can differ.

## Final Assessment

No further changes to the algorithm descriptions, central runtime claims,
quality tables, binary-UMAP interpretation, landmark discussion, or figure
organization are requested. Subject to the archival/provenance actions above,
the package and manuscript are suitable for JMLR MLOSS consideration.
