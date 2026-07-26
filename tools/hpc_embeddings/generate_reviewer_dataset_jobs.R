#!/usr/bin/env Rscript

args <- commandArgs(trailingOnly = TRUE)
script_arg <- grep("^--file=", commandArgs(FALSE), value = TRUE)
script_path <- if (length(script_arg)) {
  normalizePath(sub("^--file=", "", script_arg[[1L]]), mustWork = TRUE)
} else {
  normalizePath("generate_reviewer_dataset_jobs.R", mustWork = TRUE)
}
output_dir <- if (length(args)) args[[1L]] else file.path(dirname(script_path), "jobs_by_dataset")
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

datasets <- c(
  "COIL20", "USPS", "FashionMNIST", "FlowRepository_FR-FCM-ZYRM_files",
  "flow18", "MNIST", "imagenet", "MetRef", "mass41", "TabulaMuris",
  "Macosko2015_retina"
)

profiles <- list(
  cpu1 = list(
    backend = "cpu", threads = 1L, account = "immunology", partition = "ada",
    ntasks = 1L, cpus_per_task = 1L, gpu = character(), requeue = character()
  ),
  cpu4 = list(
    backend = "cpu", threads = 4L, account = "immunology", partition = "ada",
    ntasks = 1L, cpus_per_task = 4L, gpu = character(), requeue = character()
  ),
  cuda = list(
    backend = "cuda", threads = 1L, account = "l40sfree", partition = "l40s",
    ntasks = 1L, cpus_per_task = 1L, gpu = "#SBATCH --gres=gpu:l40s:1",
    requeue = "#SBATCH --requeue"
  )
)

safe_name <- function(x) gsub("[^A-Za-z0-9_.-]+", "_", x)
memory_gb <- function(dataset, backend) {
  if (identical(dataset, "imagenet")) return(if (backend == "cuda") 128L else 256L)
  if (identical(dataset, "FlowRepository_FR-FCM-ZYRM_files")) {
    return(if (backend == "cuda") 96L else 128L)
  }
  if (backend == "cuda") 64L else 32L
}
job_name <- function(dataset, profile) {
  paste0("feR_", substr(safe_name(dataset), 1L, 28L), "_", profile)
}

write_launcher <- function(dataset, profile_name, profile) {
  safe <- safe_name(dataset)
  path <- file.path(output_dir, sprintf("run_%s_%s.sh", safe, profile_name))
  requested_memory <- memory_gb(dataset, profile$backend)
  optional <- c(
    profile$gpu, sprintf("#SBATCH --mem=%dG", requested_memory), profile$requeue
  )
  optional <- optional[nzchar(optional)]
  lines <- c(
    "#!/usr/bin/env bash",
    "",
    sprintf("#SBATCH --account=%s", profile$account),
    sprintf("#SBATCH --partition=%s", profile$partition),
    "#SBATCH --nodes=1",
    sprintf("#SBATCH --ntasks=%d", profile$ntasks),
    sprintf("#SBATCH --cpus-per-task=%d", profile$cpus_per_task),
    optional,
    "#SBATCH --time=48:00:00",
    sprintf("#SBATCH --job-name=\"%s\"", job_name(dataset, profile_name)),
    "#SBATCH --chdir=/scratch/firenze/NN",
    sprintf(
      "#SBATCH --output=/scratch/firenze/NN/benchmark_logs/fastEmbedR_review_%s_%s_%%j.out",
      safe, profile_name
    ),
    sprintf(
      "#SBATCH --error=/scratch/firenze/NN/benchmark_logs/fastEmbedR_review_%s_%s_%%j.err",
      safe, profile_name
    ),
    "",
    "set -euo pipefail",
    "",
    sprintf("export BENCHMARK_DATASET=\"%s\"", dataset),
    sprintf("export BENCHMARK_BACKEND_GROUP=\"%s\"", profile$backend),
    sprintf("export BENCHMARK_THREADS=\"%d\"", profile$threads),
    "export BASE_DIR=\"${BASE_DIR:-/scratch/firenze/NN}\"",
    "",
    "launcher_path=\"${BASH_SOURCE[0]:-$0}\"",
    "if command -v readlink >/dev/null 2>&1; then",
    "  launcher_path=\"$(readlink -f \"${launcher_path}\" 2>/dev/null || printf '%s\\n' \"${launcher_path}\")\"",
    "fi",
    "launcher_dir=\"$(cd \"$(dirname \"${launcher_path}\")\" && pwd)\"",
    "if [[ -f \"${launcher_dir}/../run_reviewer_dataset_job.sh\" ]]; then",
    "  runner=\"${launcher_dir}/../run_reviewer_dataset_job.sh\"",
    "elif [[ -f \"${launcher_dir}/run_reviewer_dataset_job.sh\" ]]; then",
    "  runner=\"${launcher_dir}/run_reviewer_dataset_job.sh\"",
    "else",
    "  runner=\"${BASE_DIR}/run_reviewer_dataset_job.sh\"",
    "fi",
    "[[ -f \"${runner}\" ]] || { echo \"Missing run_reviewer_dataset_job.sh\" >&2; exit 1; }",
    "",
    "exec bash \"${runner}\""
  )
  writeLines(lines, path, useBytes = TRUE)
  Sys.chmod(path, mode = "0755")
  data.frame(
    dataset = dataset, profile = profile_name, backend = profile$backend,
    threads = profile$threads, ntasks = profile$ntasks,
    cpus_per_task = profile$cpus_per_task,
    memory_gb = requested_memory,
    file = basename(path), stringsAsFactors = FALSE
  )
}

manifest <- do.call(rbind, unlist(lapply(datasets, function(dataset) {
  Map(function(name, profile) write_launcher(dataset, name, profile), names(profiles), profiles)
}), recursive = FALSE))

write.csv(manifest, file.path(output_dir, "job_manifest.csv"), row.names = FALSE)
cat(sprintf("Generated %d launchers in %s\n", nrow(manifest), normalizePath(output_dir)))
