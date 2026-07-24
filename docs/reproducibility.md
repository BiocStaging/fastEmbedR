# Reproducibility

[Home](../README.md) |
[Installation](installation.md) |
[Bioconductor](bioconductor.md) |
[Implementation](implementation.md) |
[Examples](examples.md) |
[Benchmarks](benchmarks.md) |
[API](usage-api.md) |
**Reproducibility** |
[References](references.md)

This page records the reproducibility contract for the manuscript benchmarks.
It is intentionally explicit because fastEmbedR benchmarks may involve R
packages, native C++ code, optional Metal kernels, optional CUDA kernels, FAISS,
and RAPIDS cuVS linked directly by optional CUDA builds.

## Manuscript Snapshot

The current working manuscript was generated from repository HEAD:

```text
57db7ea0527d8143e0ed6cd9906247f40c1208d4
```

At the time of this documentation update, the working tree also contained
manuscript-response edits. Before journal submission, create and archive a
clean release tag, for example:

```bash
git tag -a v0.99.0-manuscript -m "fastEmbedR manuscript benchmark snapshot"
git push origin v0.99.0-manuscript
```

The release tag should then be archived on Zenodo or an equivalent repository.
Record the minted DOI by setting:

```bash
export FASTEMBEDR_MANUSCRIPT_TAG=v0.99.0-manuscript
export FASTEMBEDR_ZENODO_DOI="10.xxxx/zenodo.xxxxxxx"
```

The reproducibility scripts write these values into the benchmark output
directory. Do not invent a DOI before the archive exists.

## Reproducibility Bundle

Run:

```bash
Rscript tools/write_manuscript_reproducibility.R \
  --out-dir=results/manuscript_reproducibility_current \
  --seed=4 \
  --k=30 \
  --perplexity=15 \
  --threads=12 \
  --timeout=10800
```

The script writes:

- `reproducibility_manifest.txt`;
- `reproducibility_manifest.json` when `jsonlite` is installed;
- `sessionInfo.txt`.

The manifest records:

- exact Git commit hash;
- `git describe --tags --always --dirty`;
- release tag and archival DOI fields;
- benchmark command lines;
- random seed, k, perplexity, timeout, and thread count;
- R version, package versions, and `sessionInfo()`;
- resolved `R CMD config` values for `CC`, `CFLAGS`, `CXX17`,
  `CXX17FLAGS`, `CPPFLAGS`, `LDFLAGS`, and `SHLIB_CXX17LD`;
- C/C++ compiler, Xcode, and Metal compiler versions where available;
- the generated package `src/Makevars` and compiler-related environment
  variables;
- hardware information from `Sys.info()`, `lscpu`, `free -h`, and
  `nvidia-smi` where available;
- CUDA information from `nvcc --version`, `CUDA_HOME`, `CUDAHOSTCXX`,
  `FASTEMBEDR_CUDA_ARCH`, `FASTEMBEDR_CUDA_FLAGS`, `LD_LIBRARY_PATH`, and
  fastEmbedR native-backend probes;
- directly linked FAISS GPU/cuVS availability recorded by fastEmbedR;
- paths to the benchmark driver and wrapper scripts.

The publication benchmark driver
[`tools/hpc_embeddings/benchmark_embeddings_float32_publication.R`](../tools/hpc_embeddings/benchmark_embeddings_float32_publication.R)
also writes this bundle automatically into every HPC benchmark output
directory.

## Benchmark Commands

Small reference validation:

```bash
Rscript tools/validate_reference_implementations.R \
  --out-dir=results/reference_validation_current \
  --threads=2 \
  --seed=4 \
  --perplexity=10 \
  --k=31
```

MNIST70k GitHub example:

```bash
Rscript tools/benchmark_github_mnist70k.R \
  --n=70000 \
  --k=30 \
  --perplexity=30 \
  --threads=4 \
  --run-metal=true \
  --run-cuda=false \
  --run-references=true \
  --out-dir=results/mnist70k_github_current
```

Publication CPU benchmark:

```bash
DATASETS=MNIST,FashionMNIST,flow18,mass41,imagenet,FlowRepository_FR-FCM-ZYRM_files,MetRef \
K=30 PERPLEXITY=15 TIMEOUT=10800 \
sbatch /scratch/firenze/NN/benchmark_embeddings_float32_cpu12.sh
```

Publication CUDA benchmark:

```bash
DATASETS=MNIST,FashionMNIST,flow18,mass41,imagenet,FlowRepository_FR-FCM-ZYRM_files,MetRef \
K=30 PERPLEXITY=15 TIMEOUT=10800 \
sbatch /scratch/firenze/NN/benchmark_embeddings_float32_cuda.sh
```

## Environment

The lightweight R-side environment file is:

```text
tools/reproducibility/benchmark_environment.yml
```

FAISS, RAPIDS cuVS, CUDA, cuFFT, and GPU driver versions are system-level
dependencies owned by the CUDA installation. They are therefore
not vendored into `fastEmbedR`; the exact versions actually used in a run are
captured in `reproducibility_manifest.txt` and `reproducibility_manifest.json`.

## Compiler Comparability

The package inherits optimization flags from R and adds only `-pthread` to the
portable C++ core. Benchmark reports must therefore include the resolved
compiler configuration, not only the package version. In particular, results
from a Conda R/GCC toolchain and a system R/GCC toolchain are not treated as a
controlled algorithm comparison even when they run on the same physical CPU.

The package does not require or recommend global `-march=native`,
`-ffast-math`, or `-O3`. CUDA deployment architectures are recorded
separately because package kernels and linked FAISS/cuVS/RAFT libraries must
all support the target GPU. See
[Installation And Native Compiler Configuration](installation-backends.md)
for the complete build contract.

## Seeds

The manuscript benchmark default seed is `4`. The small reference validation
also uses seed `4`. The GitHub MNIST example records its seed in the output
CSV and plot metadata.
