# Real-Hardware Backend Validation

GPU tests are skipped during ordinary CPU-only package checks. A release must
therefore archive separate evidence from machines that actually expose Apple
Metal and NVIDIA CUDA.

The hardware-validation driver is maintained in the separate
[`fastEmbedR-benchmark`](https://github.com/tkcaccia/fastEmbedR-benchmark)
repository.

## Evidence Contract

Each hardware run records:

- the exact Git commit;
- UTC timestamp and backend requested;
- `backend_info()` and `sessionInfo()`;
- whether the requested backend was available;
- the backend actually recorded by one-call UMAP and openTSNE results;
- elapsed smoke-test times; and
- the backend-specific `testthat` results.

Run from an installed copy of the candidate package:

```sh
git clone https://github.com/tkcaccia/fastEmbedR-benchmark.git
cd fastEmbedR-benchmark
Rscript tools/run_backend_hardware_validation.R \
  --backend=metal \
  --out-dir=docs/validation/metal_release
```

```sh
Rscript tools/run_backend_hardware_validation.R \
  --backend=cuda \
  --out-dir=docs/validation/cuda_release
```

The command fails if the requested accelerator is unavailable, if a result
records a different backend, or if a backend-specific expectation fails. This
prevents a CPU fallback from being archived as GPU evidence.

## Archived Results

The repository's benchmark archive contains successful CUDA and Metal
performance and agreement runs. These results are scientific benchmark
evidence, not substitutes for release-specific test-suite logs. Before tagging
a release, rerun the commands above against the release candidate and archive
the resulting directories under `docs/validation/`.

The current Mac hardware capture is stored under
`docs/validation/metal_current/`. Its summary records the source state,
backend identity, smoke-test timings, and Metal-specific test results. A
matching CUDA directory must be generated on CUDA hardware for the final
release tag.

Because GPU optimizers use asynchronous atomic updates, a fixed seed controls
the package random streams but does not guarantee bitwise-identical layouts
across runs or devices. Validation therefore checks backend identity,
objective behavior, neighborhood agreement, and numerical tolerances rather
than byte-for-byte coordinate identity.
