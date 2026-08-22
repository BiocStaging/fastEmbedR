# Real-Hardware Backend Validation

GPU tests are skipped during ordinary CPU-only package checks. A release must
therefore archive separate evidence from machines that actually expose Apple
Metal and NVIDIA CUDA. A macOS build alone is not accepted as Metal evidence:
the runtime capability checks and native kernels must execute successfully.

The package repository provides the manually dispatched **Real hardware
validation** GitHub Actions workflow. CPU uses a hosted Linux runner. Metal and
CUDA use explicitly labelled self-hosted runners:

- `[self-hosted, macOS, ARM64, fastembedr-metal]`;
- `[self-hosted, Linux, X64, fastembedr-cuda]`.

The project-owned labels ensure that accelerator jobs are not scheduled on a
machine without the requested hardware. Larger scientific benchmarks remain
in the separate
[`fastEmbedR-benchmark`](https://github.com/tkcaccia/fastEmbedR-benchmark)
repository.

Runner registration and custom labels follow GitHub's
[self-hosted runner guidance](https://docs.github.com/en/actions/how-tos/manage-runners/self-hosted-runners/use-in-a-workflow).

## Evidence Contract

Each hardware run records:

- the exact Git commit;
- UTC timestamp and backend requested;
- `fastEmbedR_capabilities()` and `sessionInfo()`;
- whether the requested backend was available;
- the backend actually recorded by PCA, KNN, one-call UMAP, openTSNE, and
  Leiden results;
- elapsed smoke-test times;
- the complete installed-package `testthat` results; and
- SHA-256 identities for the Git archive, source package, installed shared
  library, benchmark results, session information, hardware report, and logs.

From the GitHub Actions page, select **Real hardware validation**, choose
`all`, and dispatch the workflow. Each successful job uploads a commit-named
artifact such as `fastEmbedR-hardware-cuda-<commit>`. The same validation can
be run directly on a prepared hardware runner:

```sh
bash .github/scripts/run-hardware-validation.sh metal validation-artifacts/metal
```

```sh
bash .github/scripts/run-hardware-validation.sh cuda validation-artifacts/cuda
```

The command fails if the requested accelerator is unavailable, if a result
records a different backend, or if a backend-specific expectation fails. This
prevents a CPU fallback from being archived as GPU evidence.

## Archived Results

The workflow artifact and the corresponding long-term copy in the benchmark
archive must refer to the same clean Git commit. Evidence from a dirty working
tree, a different package version, or a run that skipped the requested backend
is not release evidence. GitHub artifact retention is finite, so accepted
release artifacts are copied without modification to the benchmark archive;
the included `SHA256SUMS` file makes that copy verifiable.

See [the validation evidence schema](validation/README.md) for the required
files and release checklist.

Because GPU optimizers use asynchronous atomic updates, a fixed seed controls
the package random streams but does not guarantee bitwise-identical layouts
across runs or devices. Validation therefore checks backend identity,
objective behavior, neighborhood agreement, and numerical tolerances rather
than byte-for-byte coordinate identity.
