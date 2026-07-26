# Dataset embedding jobs

Each `run_<dataset>_{cpu1,cpu4,cuda}.sh` launcher runs the generic embedding
and reference methods for one dataset. KODAMA is intentionally excluded from
these jobs and has its own launchers under `../kodama_jobs_by_dataset/`.

Use `../dataset_submitters/run_<dataset>_all_methods.sh` to submit all three
profiles for one dataset.
