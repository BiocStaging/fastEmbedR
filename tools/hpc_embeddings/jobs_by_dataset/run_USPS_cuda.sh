#!/usr/bin/env bash

#SBATCH --account=l40sfree
#SBATCH --partition=l40s
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:l40s:1
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --job-name="feR_USPS_cuda"
#SBATCH --chdir=/scratch/firenze/NN
#SBATCH --output=/scratch/firenze/NN/benchmark_logs/fastEmbedR_review_USPS_cuda_%j.out
#SBATCH --error=/scratch/firenze/NN/benchmark_logs/fastEmbedR_review_USPS_cuda_%j.err

set -euo pipefail

export BENCHMARK_DATASET="USPS"
export BENCHMARK_BACKEND_GROUP="cuda"
export BENCHMARK_THREADS="1"
export BASE_DIR="${BASE_DIR:-/scratch/firenze/NN}"

launcher_path="${BASH_SOURCE[0]:-$0}"
if command -v readlink >/dev/null 2>&1; then
  launcher_path="$(readlink -f "${launcher_path}" 2>/dev/null || printf '%s\n' "${launcher_path}")"
fi
launcher_dir="$(cd "$(dirname "${launcher_path}")" && pwd)"
if [[ -f "${launcher_dir}/../run_reviewer_dataset_job.sh" ]]; then
  runner="${launcher_dir}/../run_reviewer_dataset_job.sh"
elif [[ -f "${launcher_dir}/run_reviewer_dataset_job.sh" ]]; then
  runner="${launcher_dir}/run_reviewer_dataset_job.sh"
else
  runner="${BASE_DIR}/run_reviewer_dataset_job.sh"
fi
[[ -f "${runner}" ]] || { echo "Missing run_reviewer_dataset_job.sh" >&2; exit 1; }

exec bash "${runner}"
