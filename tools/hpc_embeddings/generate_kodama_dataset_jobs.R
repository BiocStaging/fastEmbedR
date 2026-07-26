#!/usr/bin/env Rscript

# Generate one Slurm launcher per dataset and execution profile for the
# KODAMA-only reviewer benchmark.  The shared reviewer runner performs one
# KODAMA fit per classifier and reuses it for UMAP and openTSNE.

args <- commandArgs(trailingOnly = TRUE)
script_arg <- grep("^--file=", commandArgs(FALSE), value = TRUE)
script_path <- if (length(script_arg)) {
  normalizePath(sub("^--file=", "", script_arg[[1L]]), mustWork = TRUE)
} else {
  normalizePath("generate_kodama_dataset_jobs.R", mustWork = TRUE)
}

output_dir <- if (length(args)) args[[1L]] else {
  file.path(dirname(script_path), "kodama_jobs_by_dataset")
}
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

datasets <- c(
  "COIL20", "USPS", "FashionMNIST", "FlowRepository_FR-FCM-ZYRM_files",
  "flow18", "MNIST", "imagenet", "MetRef", "mass41", "TabulaMuris",
  "Macosko2015_retina"
)

profiles <- list(
  cpu1 = list(
    backend = "cpu", threads = 1L, account = "immunology", partition = "ada",
    ntasks = 1L, gpu = character(), requeue = character()
  ),
  cpu4 = list(
    backend = "cpu", threads = 4L, account = "immunology", partition = "ada",
    ntasks = 4L, gpu = character(), requeue = character()
  ),
  cuda = list(
    backend = "cuda", threads = 1L, account = "l40sfree", partition = "l40s",
    ntasks = 1L, gpu = "#SBATCH --gres=gpu:l40s:1",
    requeue = "#SBATCH --requeue"
  )
)

safe_name <- function(x) gsub("[^A-Za-z0-9_.-]+", "_", x)
memory_gb <- function(dataset, backend) {
  if (identical(dataset, "imagenet")) return(if (backend == "cuda") 128L else 256L)
  if (identical(dataset, "FlowRepository_FR-FCM-ZYRM_files")) {
    return(if (backend == "cuda") 128L else 192L)
  }
  if (backend == "cuda") 64L else 64L
}

write_launcher <- function(dataset, profile_name, profile) {
  safe <- safe_name(dataset)
  path <- file.path(output_dir, sprintf("run_kodama_%s_%s.sh", safe, profile_name))
  requested_memory <- memory_gb(dataset, profile$backend)
  optional <- c(
    profile$gpu, sprintf("#SBATCH --mem=%dG", requested_memory), profile$requeue
  )
  optional <- optional[nzchar(optional)]
  lines <- c(
    "#!/usr/bin/env bash", "",
    sprintf("#SBATCH --account=%s", profile$account),
    sprintf("#SBATCH --partition=%s", profile$partition),
    "#SBATCH --nodes=1",
    sprintf("#SBATCH --ntasks=%d", profile$ntasks),
    optional,
    "#SBATCH --time=48:00:00",
    sprintf("#SBATCH --job-name=\"feR_KODAMA_%s_%s\"", substr(safe, 1L, 20L), profile_name),
    "#SBATCH --chdir=/scratch/firenze/NN",
    sprintf(
      "#SBATCH --output=/scratch/firenze/NN/benchmark_logs/fastEmbedR_KODAMA_%s_%s_%%j.out",
      safe, profile_name
    ),
    sprintf(
      "#SBATCH --error=/scratch/firenze/NN/benchmark_logs/fastEmbedR_KODAMA_%s_%s_%%j.err",
      safe, profile_name
    ),
    "", "set -euo pipefail", "",
    sprintf("export BENCHMARK_DATASET=\"%s\"", dataset),
    sprintf("export BENCHMARK_BACKEND_GROUP=\"%s\"", profile$backend),
    sprintf("export BENCHMARK_THREADS=\"%d\"", profile$threads),
    "export BASE_DIR=\"${BASE_DIR:-/scratch/firenze/NN}\"",
    "export BENCHMARK_SUITE=\"kodama\"",
    "export DATA_ROOT=\"${DATA_ROOT:-${BASE_DIR}/Data}\"",
    "export SEEDS=\"${SEEDS:-4}\"",
    "export K=\"${K:-30}\"",
    "export PERPLEXITY=\"${PERPLEXITY:-30}\"",
    "export TIMEOUT=\"${TIMEOUT:-172800}\"",
    "export REFERENCE_VALIDATIONS=\"${REFERENCE_VALIDATIONS:-FALSE}\"",
    "export FORCE=\"${FORCE:-FALSE}\"",
    "export KODAMA_M=\"${KODAMA_M:-100}\"",
    "export KODAMA_TCYCLE=\"${KODAMA_TCYCLE:-100}\"",
    "export KODAMA_NCOMP=\"${KODAMA_NCOMP:-50}\"",
    "export KODAMA_LANDMARKS=\"${KODAMA_LANDMARKS:-10000000}\"",
    "export KODAMA_GRAPH_NEIGHBORS=\"${KODAMA_GRAPH_NEIGHBORS:-100}\"",
    "export KODAMA_N_EPOCHS=\"${KODAMA_N_EPOCHS:-200}\"",
    "export KODAMA_N_ITER=\"${KODAMA_N_ITER:-500}\"",
    # Only the four KODAMA workflows are run.  The shared runner still caches
    # each classifier once and reuses it for its two visualizations.
    sprintf(
      "export BENCHMARK_METHODS=\"KODAMA_plslda_opentsne_%s,KODAMA_plslda_umap_%s,KODAMA_knn_opentsne_%s,KODAMA_knn_umap_%s\"",
      profile$backend, profile$backend, profile$backend, profile$backend
    ),
    "",
    "launcher_path=\"${BASH_SOURCE[0]:-$0}\"",
    "if command -v readlink >/dev/null 2>&1; then",
    "  launcher_path=\"$(readlink -f \"${launcher_path}\" 2>/dev/null || printf '%s\\n' \"${launcher_path}\")\"",
    "fi",
    "launcher_dir=\"$(cd \"$(dirname \"${launcher_path}\")\" && pwd)\"",
    "if [[ -f \"${launcher_dir}/../run_reviewer_dataset_job.sh\" ]]; then",
    "  runner=\"${launcher_dir}/../run_reviewer_dataset_job.sh\"",
    "elif [[ -f \"${BASE_DIR}/run_reviewer_dataset_job.sh\" ]]; then",
    "  runner=\"${BASE_DIR}/run_reviewer_dataset_job.sh\"",
    "else",
    "  echo \"Missing run_reviewer_dataset_job.sh\" >&2",
    "  exit 1",
    "fi",
    "",
    "echo \"KODAMA dataset benchmark\"",
    "echo \"  dataset: ${BENCHMARK_DATASET}\"",
    "echo \"  backend: ${BENCHMARK_BACKEND_GROUP}\"",
    "echo \"  threads: ${BENCHMARK_THREADS}\"",
    "echo \"  methods:  ${BENCHMARK_METHODS}\"",
    "exec bash \"${runner}\""
  )
  writeLines(lines, path, useBytes = TRUE)
  Sys.chmod(path, mode = "0755")
  data.frame(
    dataset = dataset, profile = profile_name, backend = profile$backend,
    threads = profile$threads, ntasks = profile$ntasks,
    memory_gb = requested_memory,
    methods = paste(
      sprintf("KODAMA_plslda_opentsne_%s", profile$backend),
      sprintf("KODAMA_plslda_umap_%s", profile$backend),
      sprintf("KODAMA_knn_opentsne_%s", profile$backend),
      sprintf("KODAMA_knn_umap_%s", profile$backend), sep = ","
    ),
    file = basename(path), stringsAsFactors = FALSE
  )
}

manifest <- do.call(rbind, unlist(lapply(datasets, function(dataset) {
  Map(
    function(name, profile) write_launcher(dataset, name, profile),
    names(profiles), profiles
  )
}), recursive = FALSE))

write.csv(manifest, file.path(output_dir, "job_manifest.csv"), row.names = FALSE)
cat(sprintf("Generated %d KODAMA launchers in %s\n", nrow(manifest), normalizePath(output_dir)))
