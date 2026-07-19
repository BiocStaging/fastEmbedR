# Reviewer Validation Benchmarks

This directory contains the publication-validation benchmark for fastEmbedR.
It runs the same analysis design on Linux CPU/CUDA and native macOS CPU/Metal,
keeps failures isolated, and records both computational cost and embedding
quality. The benchmark is intentionally separate from the installed R package.

## Analysis Cohort

The default dataset panel is:

`COIL20`, `USPS`, `FashionMNIST`, `FlowRepository_FR-FCM-ZYRM_files`,
`flow18`, `MNIST`, `imagenet`, `MetRef`, `mass41`, `TabulaMuris`, and
`Macosko2015_retina`.

The HPC launchers read these datasets from `/scratch/firenze/NN/Data`. The local
launcher reads them from `/Users/stefano/Documents/fastEmbedR/Data`. Each
dataset directory must contain a standard R dataset for reference packages.
fastEmbedR reads the corresponding `float::float32` file; when that file or a
precomputed object is missing, the driver creates it once under
`_fastEmbedR_precomputed` and reuses it across methods and seeds.

## Files

- `benchmark_reviewer_validation.R`: main benchmark and isolated worker.
- `publication_metrics.R`: exact/sampled quality and agreement metrics.
- `reference_opentsne_affinity.py`: independent Python openTSNE affinity oracle.
- `benchmark_worker_monitor.sh`: timeout, peak RSS, and GPU-memory monitor.
- `run_reviewer_dataset_job.sh`: shared execution logic for one dataset/backend.
- `jobs_by_dataset/run_<dataset>_cpu1.sh`: one-core Slurm job for one dataset.
- `jobs_by_dataset/run_<dataset>_cpu4.sh`: four-core Slurm job for one dataset.
- `jobs_by_dataset/run_<dataset>_cuda.sh`: one-GPU Slurm job for one dataset.
- `jobs_by_dataset/job_manifest.csv`: complete list of the 33 generated jobs.
- `generate_reviewer_dataset_jobs.R`: reproducibly regenerates those launchers.
- `submit_reviewer_dataset_jobs.sh`: submits each selected launcher as a
  separate Slurm job and records the returned job IDs.
- `run_reviewer_hpc_cpu.sh` and `run_reviewer_hpc_cuda.sh`: legacy aggregate
  launchers retained for compatibility.
- `run_reviewer_local_cpu_metal.sh`: native macOS CPU and Metal run.
- `combine_reviewer_benchmarks.R`: cross-machine CPU/Metal/CUDA agreement.
- `fastembedr_cuda_multiarch_cugraph.def`: reproducible Linux CUDA image.

## What Is Measured

Every embedding row records total elapsed time, peak process RAM, peak GPU
memory where measurable, trustworthiness, neighbourhood preservation at 15,
30, and 50 neighbours, silhouette, embedding-space KNN label accuracy, and a
sampled t-SNE KL divergence. GPU memory is reported as a device-wide delta and
is labelled as such because `nvidia-smi` cannot isolate every allocation made
through shared CUDA libraries.

Each method is repeated with seeds `4,17,42`. Summary files report median,
quartiles, IQR, standard deviation, minimum, and maximum. Seed stability is
reported using Procrustes-aligned embedding similarity and neighbourhood
stability.

Reviewer validation additionally reports:

- fastEmbedR PCA runtime by backend and agreement with `irlba` PCA;
- exact/backend KNN recall;
- fastEmbedR t-SNE affinity agreement with Python openTSNE;
- sampled t-SNE KL divergence;
- fastEmbedR UMAP graph-weight agreement with `uwot::similarity_graph`;
- fuzzy/binary graph agreement across CPU, Metal, and CUDA;
- Procrustes and KNN agreement across CPU, Metal, and CUDA and across seeds.
- four KODAMA workflows: PLS-LDA followed by openTSNE or UMAP, and KNN
  followed by openTSNE or UMAP.

The two KODAMA classifier fits are cached independently for every dataset,
backend, thread count, and seed. Each PLS-LDA fit is executed once and reused by
its openTSNE and UMAP rows; the same rule is applied to KNN. The publication
runtime for each workflow remains `KODAMA core + visualization`, while
`kodama_core_runs.csv` exposes the shared core cost separately. Peak memory for
each workflow is the maximum of its isolated core and visualization workers.
The current public kodamaR visualization API supports CPU and CUDA, so these
rows are not mislabelled as Metal runs.

All scientific plots contain points only: no title, axes, labels, ticks, legend,
or box. Labels are used only for point colours and numerical label-aware
metrics.

## Timing Boundaries

`full` rows include PCA initialization where applicable, nearest-neighbour
search, affinity/graph construction, and embedding optimization. `knn` rows
start from the same cached KNN result and measure graph/affinity construction
plus optimization. Cache generation is never charged to a `knn` row. This
allows a fair total-runtime comparison while also exposing optimizer-only
behaviour for implementations that accept precomputed neighbours.

fastEmbedR methods use float32 inputs and native float32 computational paths.
Reference R methods use the standard double-precision R dataset rather than a
converted float32 object.

## Run On HPC

Copy the benchmark files, the complete `jobs_by_dataset` directory, and the
image into `/scratch/firenze/NN`. Create the log directory before submission:

```bash
mkdir -p /scratch/firenze/NN/benchmark_logs

sbatch /scratch/firenze/NN/jobs_by_dataset/run_MetRef_cpu1.sh
sbatch /scratch/firenze/NN/jobs_by_dataset/run_MetRef_cpu4.sh
sbatch /scratch/firenze/NN/jobs_by_dataset/run_MetRef_cuda.sh
```

Submit all 33 jobs independently with:

```bash
bash /scratch/firenze/NN/submit_reviewer_dataset_jobs.sh
```

Submit one backend group only with `PROFILE=cpu1`, `PROFILE=cpu4`, or
`PROFILE=cuda`. Use `DRY_RUN=true` to inspect the commands without submitting:

```bash
PROFILE=cuda DRY_RUN=true \
  bash /scratch/firenze/NN/submit_reviewer_dataset_jobs.sh
```

Equivalent triples exist for every dataset in the analysis cohort. CPU-1 jobs
request `#SBATCH --ntasks=1`; CPU-4 jobs request `#SBATCH --ntasks=4`; CUDA jobs
request one L40S GPU. Each job processes exactly one dataset and writes to an
independent result and precomputation directory, so jobs may run concurrently
without cache collisions. All jobs default to `k = 30`, perplexity `30`, seeds
`4,17,42`, and a 3-hour per-method timeout. A failed, OOM-killed, or timed-out
method is recorded and the remaining methods continue.

The KODAMA defaults are `M = 100`, `Tcycle = 20`, at most 50 PLS components,
at most 10,000 landmarks, 100 retained graph neighbours, `k = 30`, 200 UMAP
epochs, and 500 openTSNE optimization iterations. They can be changed with the
`KODAMA_M`, `KODAMA_TCYCLE`, `KODAMA_NCOMP`, `KODAMA_LANDMARKS`,
`KODAMA_GRAPH_NEIGHBORS`, `KODAMA_N_EPOCHS`, and `KODAMA_N_ITER` environment
variables.

To add only the four KODAMA rows to an existing dataset/backend analysis, pass
the relevant method IDs through `BENCHMARK_METHODS`. For example:

```bash
export BENCHMARK_METHODS="KODAMA_plslda_opentsne_cuda,KODAMA_plslda_umap_cuda,KODAMA_knn_opentsne_cuda,KODAMA_knn_umap_cuda"
sbatch /scratch/firenze/NN/jobs_by_dataset/run_MetRef_cuda.sh
unset BENCHMARK_METHODS
```

Regenerate the launchers after changing the dataset panel with:

```bash
Rscript /scratch/firenze/NN/generate_reviewer_dataset_jobs.R \
  /scratch/firenze/NN/jobs_by_dataset
```

## Run On macOS

Metal must be tested with the native macOS package. A Linux Singularity image
cannot expose Apple Metal.

```bash
bash /Users/stefano/Documents/umap/tools/hpc_embeddings/run_reviewer_local_cpu_metal.sh
```

This executes CPU at one and four threads and the Metal backend against
`/Users/stefano/Documents/fastEmbedR/Data`.

## Combine Backends

After copying the HPC result directories to the Mac, combine them with the local
run:

```bash
Rscript /Users/stefano/Documents/umap/tools/hpc_embeddings/combine_reviewer_benchmarks.R \
  --cpu-dir=/path/to/benchmark_reviewer_CPU_* \
  --cuda-dir=/path/to/benchmark_reviewer_CUDA_* \
  --local-dir=/path/to/benchmark_reviewer_local_CPU_Metal_* \
  --out-dir=/path/to/combined_reviewer_results
```

The combined output includes cross-backend Procrustes similarity,
neighbourhood agreement, KNN agreement, t-SNE affinity agreement, and UMAP
fuzzy/binary graph-weight agreement.

## Main Outputs

- `benchmark_runs.csv`: one row per dataset/method/thread/seed.
- `kodama_core_runs.csv`: one isolated, reusable KODAMA classifier fit per
  dataset/classifier/backend/thread/seed.
- `benchmark_summary_median_variability.csv`: medians and variability across
  successful repeats.
- `stability_pairwise.csv`: within-method stability across seeds.
- `pca_vs_irlba_agreement.csv`: fastEmbedR PCA versus `irlba`.
- `tsne_affinity_agreement_vs_python_opentsne.csv`: fastEmbedR versus openTSNE.
- `umap_graph_agreement_vs_uwot.csv`: fastEmbedR versus uwot graph weights.
- `knn_affinity_umap_graph_agreement.csv`: within-run backend diagnostics.
- `parameter_table.csv`: complete benchmark settings and timing boundaries.
- `reproducibility_manifest.txt` and `sessionInfo.txt`.
- `layouts/`, `plots/`, `worker_results/`, and `logs/`.

## CUDA Image

Build and validate the CUDA image on chiamaka with:

```bash
STAMP=reviewer \
PKG_TARBALL=/Users/stefano/Documents/umap/fastEmbedR_0.99.0.tar.gz \
COPY_LOCAL=true \
LOCAL_COPY_DIR=/Users/stefano/Documents/umap/singularity \
bash /Users/stefano/Documents/umap/tools/build_chiamaka_singularity_cugraph.sh
```

CUDA availability is tested only after image construction using
`apptainer exec --nv`; a GPU is not visible during the definition `%post`
stage. Validation covers package-native CUDA KNN, PCA, openTSNE, fuzzy and
binary UMAP, Python openTSNE/umap-learn, RAPIDS cuML, and FIt-SNE. The copied
image is suitable for Linux CUDA/HPC reproduction, not native macOS Metal.
