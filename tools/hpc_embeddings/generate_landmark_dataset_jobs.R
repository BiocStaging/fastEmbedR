#!/usr/bin/env Rscript

args <- commandArgs(trailingOnly = TRUE)
script_arg <- grep("^--file=", commandArgs(FALSE), value = TRUE)
script_path <- if (length(script_arg)) {
  normalizePath(sub("^--file=", "", script_arg[[1L]]), mustWork = TRUE)
} else {
  normalizePath("generate_landmark_dataset_jobs.R", mustWork = TRUE)
}
output_dir <- if (length(args)) args[[1L]] else {
  file.path(dirname(script_path), "landmark_jobs_by_dataset")
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
    ntasks = 1L, gpu = character(), memory = character()
  ),
  cpu4 = list(
    backend = "cpu", threads = 4L, account = "immunology", partition = "ada",
    ntasks = 4L, gpu = character(), memory = character()
  ),
  cuda = list(
    backend = "cuda", threads = 1L, account = "l40sfree", partition = "l40s",
    ntasks = 1L, gpu = "#SBATCH --gres=gpu:l40s:1", memory = "#SBATCH --mem=64G"
  )
)

safe_name <- function(x) gsub("[^A-Za-z0-9_.-]+", "_", x)

write_launcher <- function(dataset, profile_name, profile) {
  safe <- safe_name(dataset)
  path <- file.path(output_dir, sprintf("run_landmark_%s_%s.sh", safe, profile_name))
  optional <- c(profile$gpu, profile$memory)
  optional <- optional[nzchar(optional)]
  lines <- c(
    "#!/usr/bin/env bash", "",
    sprintf("#SBATCH --account=%s", profile$account),
    sprintf("#SBATCH --partition=%s", profile$partition),
    "#SBATCH --nodes=1",
    sprintf("#SBATCH --ntasks=%d", profile$ntasks),
    optional,
    "#SBATCH --time=48:00:00",
    sprintf("#SBATCH --job-name=\"feR_land_%s_%s\"", substr(safe, 1L, 24L), profile_name),
    "#SBATCH --chdir=/scratch/firenze/NN",
    sprintf(
      "#SBATCH --output=/scratch/firenze/NN/benchmark_logs/fastEmbedR_landmark_%s_%s_%%j.out",
      safe, profile_name
    ),
    sprintf(
      "#SBATCH --error=/scratch/firenze/NN/benchmark_logs/fastEmbedR_landmark_%s_%s_%%j.err",
      safe, profile_name
    ),
    "", "set -euo pipefail", "",
    sprintf("export BENCHMARK_DATASET=\"%s\"", dataset),
    sprintf("export BENCHMARK_BACKEND_GROUP=\"%s\"", profile$backend),
    sprintf("export BENCHMARK_THREADS=\"%d\"", profile$threads),
    "export LANDMARK_FRACTION=\"${LANDMARK_FRACTION:-0.5}\"",
    "export BASE_DIR=\"${BASE_DIR:-/scratch/firenze/NN}\"",
    "", "launcher_path=\"${BASH_SOURCE[0]:-$0}\"",
    "if command -v readlink >/dev/null 2>&1; then",
    "  launcher_path=\"$(readlink -f \"${launcher_path}\" 2>/dev/null || printf '%s\\n' \"${launcher_path}\")\"",
    "fi",
    "launcher_dir=\"$(cd \"$(dirname \"${launcher_path}\")\" && pwd)\"",
    "if [[ -f \"${launcher_dir}/../run_landmark_dataset_job.sh\" ]]; then",
    "  runner=\"${launcher_dir}/../run_landmark_dataset_job.sh\"",
    "elif [[ -f \"${BASE_DIR}/run_landmark_dataset_job.sh\" ]]; then",
    "  runner=\"${BASE_DIR}/run_landmark_dataset_job.sh\"",
    "else",
    "  echo \"Missing run_landmark_dataset_job.sh\" >&2",
    "  exit 1",
    "fi",
    "", "exec bash \"${runner}\""
  )
  writeLines(lines, path, useBytes = TRUE)
  Sys.chmod(path, mode = "0755")
  data.frame(
    dataset = dataset, profile = profile_name, backend = profile$backend,
    threads = profile$threads, ntasks = profile$ntasks,
    landmark_fraction = 0.5, file = basename(path), stringsAsFactors = FALSE
  )
}

manifest <- do.call(rbind, unlist(lapply(datasets, function(dataset) {
  Map(
    function(name, profile) write_launcher(dataset, name, profile),
    names(profiles), profiles
  )
}), recursive = FALSE))

write.csv(manifest, file.path(output_dir, "job_manifest.csv"), row.names = FALSE)
cat(sprintf("Generated %d landmark launchers in %s\n", nrow(manifest), normalizePath(output_dir)))
