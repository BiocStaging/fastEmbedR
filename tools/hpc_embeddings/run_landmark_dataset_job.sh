#!/usr/bin/env bash

# Run matched full and 20% landmark fastEmbedR embeddings for one dataset.
# Dataset-specific Slurm launchers set BENCHMARK_DATASET,
# BENCHMARK_BACKEND_GROUP, and BENCHMARK_THREADS before invoking this file.

set -euo pipefail

: "${BENCHMARK_DATASET:?Set BENCHMARK_DATASET in the dataset launcher.}"
: "${BENCHMARK_BACKEND_GROUP:?Set BENCHMARK_BACKEND_GROUP to cpu or cuda.}"
: "${BENCHMARK_THREADS:?Set BENCHMARK_THREADS in the dataset launcher.}"

case "${BENCHMARK_BACKEND_GROUP}" in
  cpu)
    BENCHMARK_METHODS="fastEmbedR_opentsne_cpu_full,fastEmbedR_opentsne_cpu_landmark,fastEmbedR_umap_cpu_binary_full,fastEmbedR_umap_cpu_binary_landmark"
    ;;
  cuda)
    BENCHMARK_METHODS="fastEmbedR_opentsne_cuda_full,fastEmbedR_opentsne_cuda_landmark,fastEmbedR_umap_cuda_binary_full,fastEmbedR_umap_cuda_binary_landmark"
    ;;
  *)
    echo "Landmark HPC jobs support only cpu and cuda." >&2
    exit 2
    ;;
esac

BASE_DIR="${BASE_DIR:-/scratch/firenze/NN}"
LANDMARK_FRACTION="${LANDMARK_FRACTION:-0.2}"
K="${K:-30}"
PERPLEXITY="${PERPLEXITY:-30}"
if [[ "${K}" != "${PERPLEXITY}" ]]; then
  echo "Landmark validation requires K and PERPLEXITY to match." >&2
  exit 2
fi
REFERENCE_VALIDATIONS="FALSE"

script_path="${BASH_SOURCE[0]:-$0}"
if command -v readlink >/dev/null 2>&1; then
  script_path="$(readlink -f "${script_path}" 2>/dev/null || printf '%s\n' "${script_path}")"
fi
script_dir="$(cd "$(dirname "${script_path}")" && pwd)"
for candidate in \
  "${script_dir}/run_reviewer_dataset_job.sh" \
  "${script_dir}/../run_reviewer_dataset_job.sh" \
  "${BASE_DIR}/run_reviewer_dataset_job.sh"; do
  if [[ -f "${candidate}" ]]; then
    runner="${candidate}"
    break
  fi
done
[[ -n "${runner:-}" ]] || {
  echo "Cannot find run_reviewer_dataset_job.sh." >&2
  exit 1
}

export BASE_DIR LANDMARK_FRACTION K PERPLEXITY REFERENCE_VALIDATIONS
export BENCHMARK_SUITE="landmark"
export BENCHMARK_METHODS
exec bash "${runner}"
