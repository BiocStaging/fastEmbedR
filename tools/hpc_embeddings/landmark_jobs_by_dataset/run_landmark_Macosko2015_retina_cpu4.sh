#!/usr/bin/env bash

#SBATCH --account=immunology
#SBATCH --partition=ada
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --time=48:00:00
#SBATCH --job-name="feR_land_Macosko2015_retina_cpu4"
#SBATCH --chdir=/scratch/firenze/NN
#SBATCH --output=/scratch/firenze/NN/benchmark_logs/fastEmbedR_landmark_Macosko2015_retina_cpu4_%j.out
#SBATCH --error=/scratch/firenze/NN/benchmark_logs/fastEmbedR_landmark_Macosko2015_retina_cpu4_%j.err

set -euo pipefail

export BENCHMARK_DATASET="Macosko2015_retina"
export BENCHMARK_BACKEND_GROUP="cpu"
export BENCHMARK_THREADS="4"
export LANDMARK_FRACTION="${LANDMARK_FRACTION:-0.5}"
export BASE_DIR="${BASE_DIR:-/scratch/firenze/NN}"

launcher_path="${BASH_SOURCE[0]:-$0}"
if command -v readlink >/dev/null 2>&1; then
  launcher_path="$(readlink -f "${launcher_path}" 2>/dev/null || printf '%s\n' "${launcher_path}")"
fi
launcher_dir="$(cd "$(dirname "${launcher_path}")" && pwd)"
if [[ -f "${launcher_dir}/../run_landmark_dataset_job.sh" ]]; then
  runner="${launcher_dir}/../run_landmark_dataset_job.sh"
elif [[ -f "${BASE_DIR}/run_landmark_dataset_job.sh" ]]; then
  runner="${BASE_DIR}/run_landmark_dataset_job.sh"
else
  echo "Missing run_landmark_dataset_job.sh" >&2
  exit 1
fi

exec bash "${runner}"
