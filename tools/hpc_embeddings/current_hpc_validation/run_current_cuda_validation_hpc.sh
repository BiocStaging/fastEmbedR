#!/usr/bin/env bash

#SBATCH --account=l40sfree
#SBATCH --partition=l40s
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --gres=gpu:l40s:1
#SBATCH --mem=96G
#SBATCH --time=48:00:00
#SBATCH --job-name=feR_current_CUDA
#SBATCH --chdir=/scratch/firenze/NN
#SBATCH --output=/scratch/firenze/NN/benchmark_logs/feR_current_cuda_%j.out
#SBATCH --error=/scratch/firenze/NN/benchmark_logs/feR_current_cuda_%j.err

set -euo pipefail

BASE_DIR=/scratch/firenze/NN
BUNDLE_DIR="${BASE_DIR}/current_fastembedr_validation"
DATA_ROOT="${BASE_DIR}/Data"
IMAGE="${BASE_DIR}/singularity/fastembedr_cuda.sif"
RUN_ID="${FASTEMBEDR_RUN_ID:-current_$(date +%Y%m%d_%H%M%S)}"
OUT_DIR="${BASE_DIR}/fastEmbedR-results/${RUN_ID}/cuda"
CACHE_DIR="${BASE_DIR}/fastEmbedR-results/${RUN_ID}/cache_cuda"
GRAPH_DIR="${BASE_DIR}/fastEmbedR-results/${RUN_ID}/cpu_cuda_graph"

[[ -f "${BUNDLE_DIR}/INSTALL_OK" ]] || {
  echo "Current package installation is not validated." >&2
  exit 1
}

mkdir -p "${OUT_DIR}" "${CACHE_DIR}" "${GRAPH_DIR}" \
  "${BASE_DIR}/benchmark_logs"

CONTAINER="$(command -v apptainer || command -v singularity || true)"
[[ -n "${CONTAINER}" ]] || {
  echo "apptainer/singularity is unavailable" >&2
  exit 1
}

export APPTAINERENV_FASTEMBEDR_CURRENT_RLIB="${BUNDLE_DIR}/Rlib"
export APPTAINERENV_R_PROFILE_USER="${BUNDLE_DIR}/Rprofile.current"
export APPTAINERENV_OMP_NUM_THREADS=4
export APPTAINERENV_OPENBLAS_NUM_THREADS=4
export APPTAINERENV_MKL_NUM_THREADS=4
export APPTAINERENV_RCPP_PARALLEL_NUM_THREADS=4
export APPTAINERENV_LD_LIBRARY_PATH="/opt/rapids/lib:/opt/faiss/lib:/usr/local/cuda/lib64:/opt/conda/lib:/opt/conda/targets/x86_64-linux/lib"

"${CONTAINER}" exec --nv \
  --bind "${BASE_DIR}:${BASE_DIR}" \
  --pwd "${BUNDLE_DIR}" \
  "${IMAGE}" \
  /opt/conda/bin/Rscript "${BUNDLE_DIR}/benchmark_reviewer_validation.R" \
  --backend-group=cuda \
  --base-dir="${BASE_DIR}" \
  --data-root="${DATA_ROOT}" \
  --out-dir="${OUT_DIR}" \
  --cache-dir="${CACHE_DIR}" \
  --datasets=COIL20,USPS,FashionMNIST,FlowRepository_FR-FCM-ZYRM_files,flow18,MNIST,MetRef,mass41,TabulaMuris,Macosko2015_retina,imagenet \
  --threads-grid=4 \
  --seeds=4,17,42 \
  --k=30 \
  --perplexity=30 \
  --timeout=10800 \
  --quality-max-distance-ops=200000000 \
  --force=FALSE

"${CONTAINER}" exec --nv \
  --bind "${BASE_DIR}:${BASE_DIR}" \
  --pwd "${BUNDLE_DIR}" \
  "${IMAGE}" \
  /opt/conda/bin/Rscript "${BUNDLE_DIR}/benchmark_local_graph_clustering.R" \
  --data-root="${DATA_ROOT}" \
  --out-dir="${GRAPH_DIR}" \
  --datasets=MetRef,COIL20,USPS,Macosko2015_retina,FashionMNIST,MNIST,TabulaMuris \
  --backends=cpu,cuda \
  --threads-grid=4 \
  --seeds=4,17,42 \
  --k=30 \
  --timeout=10800 \
  --igraph-max-n=50000 \
  --walktrap-max-n=4000 \
  --force=FALSE

echo "CUDA benchmark complete: ${OUT_DIR}"
echo "CPU/CUDA graph validation complete: ${GRAPH_DIR}"
