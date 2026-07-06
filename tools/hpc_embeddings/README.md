# HPC Embedding Benchmarks

These scripts run publication-style embedding benchmarks on the HPC datasets in
`/scratch/firenze/NN/Data`.

## Files

- `benchmark_embeddings_float32_publication.R`
  Main R driver. It runs each dataset/method in an isolated child R process,
  captures elapsed time, captures peak RSS memory through `/usr/bin/time -v`
  when available, saves layouts, saves per-method plots, computes embedding
  quality metrics, and continues after failed/OOM/timeout methods.

- `benchmark_python_direct.py`
  Native Python subprocess helper for Python reference methods. Rows ending in
  `_direct` use this file and report Python-side fit time separately from the
  R/reticulate-mediated rows.

- `benchmark_embeddings_float32_cpu12.sh`
  CPU-only Slurm wrapper using 12 CPU cores. It runs:
  `fastEmbedR_opentsne_cpu`, `fastEmbedR_umap_cpu_fuzzy`,
  `fastEmbedR_umap_cpu_binary`, `Rtsne_full`, `KlugerLab_FItSNE`,
  `python_opentsne_fft`, `python_opentsne_fft_direct`, `umap_package`,
  `uwot_default`, `uwot_fast_sgd`, `python_umap_learn`, and
  `python_umap_learn_direct`.

- `benchmark_embeddings_float32_cuda.sh`
  CUDA-only Slurm wrapper using one L40S GPU. It runs:
  `fastEmbedR_opentsne_cuda`, `fastEmbedR_umap_cuda_fuzzy`, and
  `fastEmbedR_umap_cuda_binary`, plus reticulate-mediated and native Python
  subprocess RAPIDS/cuML UMAP and t-SNE rows. It prints CUDA/faissR/fastEmbedR
  diagnostics before the benchmark so a missing CUDA backend is visible
  immediately.

## Input Rule

- fastEmbedR methods load each dataset's `*_float32.RData` file.
- Reference R packages load each dataset's standard `.RData` file.

## Copy To HPC Folder

From the local machine:

```bash
cp /Users/stefano/Documents/umap/tools/hpc_embeddings/benchmark_embeddings_float32_publication.R \
   /Users/stefano/HPC-firenze/NN/
cp /Users/stefano/Documents/umap/tools/hpc_embeddings/benchmark_python_direct.py \
   /Users/stefano/HPC-firenze/NN/
cp /Users/stefano/Documents/umap/tools/hpc_embeddings/benchmark_embeddings_float32_cpu12.sh \
   /Users/stefano/HPC-firenze/NN/
cp /Users/stefano/Documents/umap/tools/hpc_embeddings/benchmark_embeddings_float32_cuda.sh \
   /Users/stefano/HPC-firenze/NN/
```

If `dataset_input_audit.csv` reports missing standard `.RData` files for
reference packages, copy them into the local HPC mirror before syncing:

```bash
bash /Users/stefano/Documents/umap/tools/hpc_embeddings/sync_missing_standard_rdata.sh
```

Then sync `/Users/stefano/HPC-firenze/NN` to the HPC as usual.

## Submit On HPC

```bash
sbatch /scratch/firenze/NN/benchmark_embeddings_float32_cpu12.sh
sbatch /scratch/firenze/NN/benchmark_embeddings_float32_cuda.sh
```

Optional overrides:

```bash
DATASETS=MNIST,FashionMNIST K=30 PERPLEXITY=15 TIMEOUT=10800 \
sbatch /scratch/firenze/NN/benchmark_embeddings_float32_cpu12.sh

DATASETS=MNIST,FashionMNIST K=30 PERPLEXITY=15 TIMEOUT=10800 \
sbatch /scratch/firenze/NN/benchmark_embeddings_float32_cuda.sh
```

## Outputs

Each run creates a timestamped output directory containing:

- `embedding_benchmark_results.csv`
- `embedding_parameter_table.csv`
- `embedding_parameter_table.md`
- `embedding_quality_table.csv`
- `embedding_quality_table.md`
- `embedding_runtime_quality_pareto.csv`
- `embedding_runtime_quality_pareto.png`
- `embedding_time_barplot.png`
- `embedding_memory_barplot.png`
- `embedding_parameter_table.csv`
- `embedding_parameter_table.md`
- `benchmark_command_lines.txt`
- `sessionInfo.txt`
- `reproducibility_manifest.txt`
- `reproducibility_manifest.json` when `jsonlite` is installed
- `layouts/*.rds`
- `plots/*.png`
- `logs/*.log`
- `worker_results/*.csv`

The manuscript table is `embedding_quality_table.md`. It reports, for the key
datasets requested by the reviewer plus the explicit metabolomics benchmark
(`MNIST`, `FashionMNIST`, `flow18`, `mass41`, `imagenet`,
`FlowRepository_FR-FCM-ZYRM_files`, and `MetRef`): dataset, method, backend,
runtime, trustworthiness, nearest-neighbour preservation, silhouette score,
embedding-space KNN label accuracy, peak RSS memory, and status. The Pareto
figure plots runtime against trustworthiness for the same key datasets.

`MetRef` is the metabolomics dataset in the embedding benchmark. Simulated
matrices are used only in the separate nearest-neighbour stress benchmark and
should not be described as part of the UMAP/t-SNE embedding-quality panel.

The method-parameter table is `embedding_parameter_table.md`. It records the
settings needed to assess benchmark fairness: `n_neighbors`/`k`, perplexity,
iterations or epochs, early exaggeration, learning-rate policy, initialization,
distance metric, thread count, random seed, whether KNN was precomputed, and
whether the KNN path was approximate, exact, or package-internal. The benchmark
still compares total elapsed user-level runtime because the reference packages
do not expose identical KNN/affinity/optimization boundaries.

Python reference methods are reported in two timing modes. Method names without
the `_direct` suffix are called from R through `reticulate`. Method names with
the `_direct` suffix run inside a native Python subprocess using
`benchmark_python_direct.py`. For `_direct` rows, `elapsed_sec` is the
Python-side fit time measured by `time.perf_counter()`, while
`process_elapsed_sec` records the whole subprocess wall time including Python
startup and NPZ input loading.

The reproducibility manifest records the Git commit, release tag field,
archival DOI field, command lines, random seed, k/perplexity, thread count,
timeout, R `sessionInfo()`, package versions, hardware information,
`nvidia-smi` output when available, CUDA environment variables, and faissR
backend probes for FAISS/cuVS availability. The release tag and DOI are read
from `FASTEMBEDR_MANUSCRIPT_TAG` and `FASTEMBEDR_ZENODO_DOI` when those
environment variables are set.
