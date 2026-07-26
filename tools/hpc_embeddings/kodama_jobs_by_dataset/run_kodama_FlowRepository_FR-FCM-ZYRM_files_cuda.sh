#!/usr/bin/env bash

#SBATCH --account=l40sfree
#SBATCH --partition=l40s
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:l40s:1
#SBATCH --mem=128G
#SBATCH --requeue
#SBATCH --time=48:00:00
#SBATCH --job-name="feR_KODAMA_FlowRepository_FR-FC_cuda"
#SBATCH --chdir=/scratch/firenze/NN
#SBATCH --output=/scratch/firenze/NN/benchmark_logs/fastEmbedR_KODAMA_FlowRepository_FR-FCM-ZYRM_files_cuda_%j.out
#SBATCH --error=/scratch/firenze/NN/benchmark_logs/fastEmbedR_KODAMA_FlowRepository_FR-FCM-ZYRM_files_cuda_%j.err

set -euo pipefail

export BENCHMARK_DATASET="FlowRepository_FR-FCM-ZYRM_files"
export BENCHMARK_BACKEND_GROUP="cuda"
export BENCHMARK_THREADS="1"
export BASE_DIR="${BASE_DIR:-/scratch/firenze/NN}"
export BENCHMARK_SUITE="kodama"
export DATA_ROOT="${DATA_ROOT:-${BASE_DIR}/Data}"
export SEEDS="${SEEDS:-4}"
export K="${K:-30}"
export PERPLEXITY="${PERPLEXITY:-30}"
export TIMEOUT="${TIMEOUT:-172800}"
export REFERENCE_VALIDATIONS="${REFERENCE_VALIDATIONS:-FALSE}"
export FORCE="${FORCE:-FALSE}"
export KODAMA_M="${KODAMA_M:-100}"
export KODAMA_TCYCLE="${KODAMA_TCYCLE:-100}"
export KODAMA_NCOMP="${KODAMA_NCOMP:-50}"
export KODAMA_LANDMARKS="${KODAMA_LANDMARKS:-10000000}"
export KODAMA_GRAPH_NEIGHBORS="${KODAMA_GRAPH_NEIGHBORS:-100}"
export KODAMA_N_EPOCHS="${KODAMA_N_EPOCHS:-200}"
export KODAMA_N_ITER="${KODAMA_N_ITER:-500}"
export BENCHMARK_METHODS="KODAMA_plslda_opentsne_cuda,KODAMA_plslda_umap_cuda,KODAMA_knn_opentsne_cuda,KODAMA_knn_umap_cuda"

launcher_path="${BASH_SOURCE[0]:-$0}"
if command -v readlink >/dev/null 2>&1; then
  launcher_path="$(readlink -f "${launcher_path}" 2>/dev/null || printf '%s\n' "${launcher_path}")"
fi
launcher_dir="$(cd "$(dirname "${launcher_path}")" && pwd)"
if [[ -f "${launcher_dir}/../run_reviewer_dataset_job.sh" ]]; then
  runner="${launcher_dir}/../run_reviewer_dataset_job.sh"
elif [[ -f "${BASE_DIR}/run_reviewer_dataset_job.sh" ]]; then
  runner="${BASE_DIR}/run_reviewer_dataset_job.sh"
else
  echo "Missing run_reviewer_dataset_job.sh" >&2
  exit 1
fi

echo "KODAMA dataset benchmark"
echo "  dataset: ${BENCHMARK_DATASET}"
echo "  backend: ${BENCHMARK_BACKEND_GROUP}"
echo "  threads: ${BENCHMARK_THREADS}"
echo "  methods:  ${BENCHMARK_METHODS}"
exec bash "${runner}"
