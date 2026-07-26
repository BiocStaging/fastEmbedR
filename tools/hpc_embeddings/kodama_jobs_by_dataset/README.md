# KODAMA dataset jobs

This directory contains one Slurm launcher for each dataset and execution
profile. Each launcher runs only these four workflows:

- `KODAMA_plslda_opentsne_*`
- `KODAMA_plslda_umap_*`
- `KODAMA_knn_opentsne_*`
- `KODAMA_knn_umap_*`

The shared reviewer runner fits each KODAMA classifier only once per
dataset/backend/thread profile and reuses that fit for its UMAP and openTSNE
visualizations. Therefore the two visualizations for a classifier do not
duplicate the expensive KODAMA core calculation.

Profiles are:

- `cpu1`: one CPU task
- `cpu4`: four CPU tasks
- `cuda`: one L40S GPU task

The launchers expect the full benchmark support files under
`/scratch/firenze/NN`, including `run_reviewer_dataset_job.sh`,
`benchmark_reviewer_validation.R`, `publication_metrics.R`,
`benchmark_worker_monitor.sh`, `reference_opentsne_affinity.py`, and the
Singularity image.

Generate or regenerate the files with:

```bash
Rscript tools/hpc_embeddings/generate_kodama_dataset_jobs.R \
  tools/hpc_embeddings/kodama_jobs_by_dataset
```

After copying the generated directory and the shared runner files to the HPC:

```bash
PROFILE=cpu4 bash kodama_jobs_by_dataset/../submit_kodama_dataset_jobs.sh
PROFILE=cuda bash kodama_jobs_by_dataset/../submit_kodama_dataset_jobs.sh
```

The per-dataset output is written by the shared runner. Failed or timed-out
datasets do not prevent the other dataset jobs from running.
