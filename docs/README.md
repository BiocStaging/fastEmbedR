# fastEmbedR Documentation

[Home](../README.md) |
[Installation](installation.md) |
[Bioconductor](bioconductor.md) |
[Implementation](implementation.md) |
[Performance Engineering](backend-performance-engineering.md) |
[Examples](examples.md) |
[Benchmarks](benchmarks.md) |
[API](usage-api.md) |
[API Map](api-map.md) |
[Development](development.md) |
[References](references.md)

## Main Pages

- [Installation](installation.md): `fastEmbedR` CPU, Metal, and CUDA embedding
  builds, including optional direct FAISS GPU/cuVS linkage.
- [Implementation](implementation.md): how UMAP and openTSNE are implemented
  across CPU, Metal, and CUDA.
- [Backend performance engineering](backend-performance-engineering.md):
  profiler results, retained CPU/Metal/CUDA optimizations, rejected
  experiments, quality gates, and reproducible timing evidence.
- [Examples](examples.md): iris examples plus the MNIST70k benchmark command,
  figure, timing table, and machine specification.
- [Benchmarks](benchmarks.md): current MNIST benchmark summary and figures.
- [Usage and API](usage-api.md): function-level usage guide.
- [Public API map](api-map.md): canonical, advanced, diagnostic, secondary,
  and compatibility interfaces with classes, backends, residency, and methods.
- [Backend capabilities](backend-capabilities.md): what each backend can do.
- [Hardware evidence contract](backend-validation.md): the distinction between
  full benchmark validation, strict hardware smoke testing, and build-level
  architectural compatibility.
- [Development and software quality](development.md): CI, coverage,
  real-hardware validation, contribution, and release evidence.
- [References](references.md): AACR-style literature and software references.

## Repository-Level References

- [Benchmark summary](../BENCHMARK_SUMMARY.md)
- [License implications](../LICENSE-IMPLICATIONS.md)
- [Detailed provenance notice](../inst/NOTICE)
- [Algorithmic references](../inst/ALGORITHMIC_REFERENCES.md)
