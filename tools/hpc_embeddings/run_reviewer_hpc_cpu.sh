#!/usr/bin/env bash

#SBATCH --account=immunology
#SBATCH --partition=ada
#SBATCH --nodes=1
#SBATCH --ntasks=12
#SBATCH --time=48:00:00
#SBATCH --job-name="fastEmbedR_review_CPU"
#SBATCH --chdir=/scratch/firenze/NN
#SBATCH --output=/scratch/firenze/NN/benchmark_logs/fastEmbedR_review_cpu_%j.out
#SBATCH --error=/scratch/firenze/NN/benchmark_logs/fastEmbedR_review_cpu_%j.err

set -euo pipefail

BASE_DIR="${BASE_DIR:-/scratch/firenze/NN}"
DATA_ROOT="${DATA_ROOT:-${BASE_DIR}/Data}"
SCRIPT_DIR="${SCRIPT_DIR:-${BASE_DIR}}"
IMAGE="${SINGULARITY_IMAGE:-${BASE_DIR}/singularity/fastembedr_cuda.sif}"
OUT_DIR="${OUT_DIR:-${BASE_DIR}/benchmark_reviewer_CPU_$(date +%Y%m%d_%H%M%S)}"
CACHE_DIR="${CACHE_DIR:-${DATA_ROOT}/_fastEmbedR_precomputed}"
DATASETS="${DATASETS:-COIL20,USPS,FashionMNIST,FlowRepository_FR-FCM-ZYRM_files,flow18,MNIST,MetRef,mass41,TabulaMuris,Macosko2015_retina,imagenet}"
THREADS_GRID="${THREADS_GRID:-1,4}"
SEEDS="${SEEDS:-4,17,42}"
K="${K:-30}"
PERPLEXITY="${PERPLEXITY:-30}"
TIMEOUT="${TIMEOUT:-10800}"
QUALITY_MAX_DISTANCE_OPS="${QUALITY_MAX_DISTANCE_OPS:-200000000}"
FORCE="${FORCE:-FALSE}"

BENCH_R="${SCRIPT_DIR}/benchmark_reviewer_validation.R"
METRICS_R="${SCRIPT_DIR}/publication_metrics.R"
MONITOR_SH="${SCRIPT_DIR}/benchmark_worker_monitor.sh"
REFERENCE_PY="${SCRIPT_DIR}/reference_opentsne_affinity.py"
for required in "${BENCH_R}" "${METRICS_R}" "${MONITOR_SH}" "${REFERENCE_PY}" "${IMAGE}"; do
  [[ -f "${required}" ]] || { echo "Missing required file: ${required}" >&2; exit 1; }
done

CONTAINER="$(command -v apptainer || command -v singularity || true)"
[[ -n "${CONTAINER}" ]] || { echo "apptainer/singularity was not found" >&2; exit 1; }
mkdir -p "${BASE_DIR}/benchmark_logs" "${OUT_DIR}" "${CACHE_DIR}"

export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
export RCPP_PARALLEL_NUM_THREADS=4
export APPTAINERENV_OMP_NUM_THREADS=4
export APPTAINERENV_OPENBLAS_NUM_THREADS=4
export APPTAINERENV_MKL_NUM_THREADS=4
export APPTAINERENV_RCPP_PARALLEL_NUM_THREADS=4

echo "CPU reviewer benchmark"
echo "data=${DATA_ROOT}"
echo "output=${OUT_DIR}"
echo "datasets=${DATASETS}"
echo "threads=${THREADS_GRID}"
echo "seeds=${SEEDS}"

"${CONTAINER}" exec \
  --bind "${BASE_DIR}:${BASE_DIR}" \
  --pwd "${BASE_DIR}" \
  "${IMAGE}" \
  /opt/conda/bin/Rscript "${BENCH_R}" \
  --backend-group=cpu \
  --base-dir="${BASE_DIR}" \
  --data-root="${DATA_ROOT}" \
  --out-dir="${OUT_DIR}" \
  --cache-dir="${CACHE_DIR}" \
  --datasets="${DATASETS}" \
  --threads-grid="${THREADS_GRID}" \
  --seeds="${SEEDS}" \
  --k="${K}" \
  --perplexity="${PERPLEXITY}" \
  --timeout="${TIMEOUT}" \
  --quality-max-distance-ops="${QUALITY_MAX_DISTANCE_OPS}" \
  --force="${FORCE}"

echo "DONE: ${OUT_DIR}"
