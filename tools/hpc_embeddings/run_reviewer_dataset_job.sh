#!/usr/bin/env bash

# Shared runner used by the dataset-specific Slurm launchers. This file has no
# SBATCH header: submit one of jobs_by_dataset/run_*_{cpu1,cpu4,cuda}.sh.

set -euo pipefail

BENCHMARK_DATASET="${BENCHMARK_DATASET:?Set BENCHMARK_DATASET in the dataset launcher.}"
BENCHMARK_BACKEND_GROUP="${BENCHMARK_BACKEND_GROUP:?Set BENCHMARK_BACKEND_GROUP to cpu or cuda.}"
BENCHMARK_THREADS="${BENCHMARK_THREADS:?Set BENCHMARK_THREADS in the dataset launcher.}"

case "${BENCHMARK_BACKEND_GROUP}" in
  cpu|cuda) ;;
  *) echo "BENCHMARK_BACKEND_GROUP must be cpu or cuda." >&2; exit 2 ;;
esac
case "${BENCHMARK_THREADS}" in
  1|4) ;;
  *) echo "BENCHMARK_THREADS must be 1 or 4." >&2; exit 2 ;;
esac
if [[ "${BENCHMARK_BACKEND_GROUP}" == "cuda" && "${BENCHMARK_THREADS}" != "1" ]]; then
  echo "CUDA jobs use BENCHMARK_THREADS=1; GPU execution is not labelled as multi-CPU." >&2
  exit 2
fi

BASE_DIR="${BASE_DIR:-/scratch/firenze/NN}"
DATA_ROOT="${DATA_ROOT:-${BASE_DIR}/Data}"
IMAGE="${SINGULARITY_IMAGE:-${BASE_DIR}/singularity/fastembedr_cuda.sif}"
SEEDS="${SEEDS:-4,17,42}"
K="${K:-30}"
PERPLEXITY="${PERPLEXITY:-30}"
TIMEOUT="${TIMEOUT:-10800}"
QUALITY_MAX_DISTANCE_OPS="${QUALITY_MAX_DISTANCE_OPS:-200000000}"
BENCHMARK_METHODS="${BENCHMARK_METHODS:-}"
KODAMA_M="${KODAMA_M:-100}"
KODAMA_TCYCLE="${KODAMA_TCYCLE:-20}"
KODAMA_NCOMP="${KODAMA_NCOMP:-50}"
KODAMA_LANDMARKS="${KODAMA_LANDMARKS:-10000}"
KODAMA_GRAPH_NEIGHBORS="${KODAMA_GRAPH_NEIGHBORS:-100}"
KODAMA_N_EPOCHS="${KODAMA_N_EPOCHS:-200}"
KODAMA_N_ITER="${KODAMA_N_ITER:-500}"
FORCE="${FORCE:-FALSE}"
DRY_RUN="${DRY_RUN:-FALSE}"

safe_dataset="$(printf '%s' "${BENCHMARK_DATASET}" | tr -c 'A-Za-z0-9_.-' '_')"
profile="${BENCHMARK_BACKEND_GROUP}${BENCHMARK_THREADS}"
run_stamp="${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}"
job_suffix="${SLURM_JOB_ID:-manual}"
OUT_DIR="${OUT_DIR:-${BASE_DIR}/benchmark_reviewer_${safe_dataset}_${profile}_${run_stamp}_${job_suffix}}"
CACHE_DIR="${CACHE_DIR:-${DATA_ROOT}/_fastEmbedR_precomputed_jobs/${safe_dataset}/${profile}}"

runner_path="${BASH_SOURCE[0]:-$0}"
if command -v readlink >/dev/null 2>&1; then
  runner_path="$(readlink -f "${runner_path}" 2>/dev/null || printf '%s\n' "${runner_path}")"
fi
runner_dir="$(cd "$(dirname "${runner_path}")" && pwd)"

resolve_file() {
  local name="$1"
  local candidate
  for candidate in \
    "${runner_dir}/${name}" \
    "${SCRIPT_DIR:-}/${name}" \
    "${BASE_DIR}/${name}" \
    "${BASE_DIR}/benchmark_scripts/${name}"; do
    if [[ -n "${candidate}" && -f "${candidate}" ]]; then
      printf '%s\n' "${candidate}"
      return 0
    fi
  done
  echo "Cannot find ${name}. Copy the reviewer benchmark files together." >&2
  return 1
}

BENCH_R="$(resolve_file benchmark_reviewer_validation.R)"
METRICS_R="$(resolve_file publication_metrics.R)"
MONITOR_SH="$(resolve_file benchmark_worker_monitor.sh)"
REFERENCE_PY="$(resolve_file reference_opentsne_affinity.py)"

for required in "${BENCH_R}" "${METRICS_R}" "${MONITOR_SH}" "${REFERENCE_PY}" "${IMAGE}"; do
  [[ -f "${required}" ]] || { echo "Missing required file: ${required}" >&2; exit 1; }
done

CONTAINER="$(command -v apptainer || command -v singularity || true)"
[[ -n "${CONTAINER}" ]] || { echo "apptainer/singularity was not found" >&2; exit 1; }

mkdir -p "${BASE_DIR}/benchmark_logs" "${OUT_DIR}" "${CACHE_DIR}"

export OMP_NUM_THREADS="${BENCHMARK_THREADS}"
export OPENBLAS_NUM_THREADS="${BENCHMARK_THREADS}"
export MKL_NUM_THREADS="${BENCHMARK_THREADS}"
export VECLIB_MAXIMUM_THREADS="${BENCHMARK_THREADS}"
export RCPP_PARALLEL_NUM_THREADS="${BENCHMARK_THREADS}"
export APPTAINERENV_OMP_NUM_THREADS="${BENCHMARK_THREADS}"
export APPTAINERENV_OPENBLAS_NUM_THREADS="${BENCHMARK_THREADS}"
export APPTAINERENV_MKL_NUM_THREADS="${BENCHMARK_THREADS}"
export APPTAINERENV_RCPP_PARALLEL_NUM_THREADS="${BENCHMARK_THREADS}"

container_args=(exec)
if [[ "${BENCHMARK_BACKEND_GROUP}" == "cuda" ]]; then
  container_args+=(--nv)
  export APPTAINERENV_LD_LIBRARY_PATH="/opt/rapids/lib:/opt/faiss/lib:/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}"
fi
container_args+=(--bind "${BASE_DIR}:${BASE_DIR}" --pwd "${BASE_DIR}" "${IMAGE}")

r_args=(
  "${BENCH_R}"
  "--backend-group=${BENCHMARK_BACKEND_GROUP}"
  "--base-dir=${BASE_DIR}"
  "--data-root=${DATA_ROOT}"
  "--out-dir=${OUT_DIR}"
  "--cache-dir=${CACHE_DIR}"
  "--datasets=${BENCHMARK_DATASET}"
  "--threads-grid=${BENCHMARK_THREADS}"
  "--seeds=${SEEDS}"
  "--k=${K}"
  "--perplexity=${PERPLEXITY}"
  "--timeout=${TIMEOUT}"
  "--quality-max-distance-ops=${QUALITY_MAX_DISTANCE_OPS}"
  "--kodama-m=${KODAMA_M}"
  "--kodama-tcycle=${KODAMA_TCYCLE}"
  "--kodama-ncomp=${KODAMA_NCOMP}"
  "--kodama-landmarks=${KODAMA_LANDMARKS}"
  "--kodama-graph-neighbors=${KODAMA_GRAPH_NEIGHBORS}"
  "--kodama-n-epochs=${KODAMA_N_EPOCHS}"
  "--kodama-n-iter=${KODAMA_N_ITER}"
  "--force=${FORCE}"
)
if [[ -n "${BENCHMARK_METHODS}" ]]; then
  r_args+=("--methods=${BENCHMARK_METHODS}")
fi

echo "fastEmbedR reviewer dataset job"
echo "  dataset: ${BENCHMARK_DATASET}"
echo "  backend: ${BENCHMARK_BACKEND_GROUP}"
echo "  threads: ${BENCHMARK_THREADS}"
echo "  data:    ${DATA_ROOT}"
echo "  output:  ${OUT_DIR}"
echo "  cache:   ${CACHE_DIR}"
echo "  image:   ${IMAGE}"
echo "  KODAMA:  M=${KODAMA_M} Tcycle=${KODAMA_TCYCLE} ncomp=${KODAMA_NCOMP} landmarks=${KODAMA_LANDMARKS}"
if [[ -n "${BENCHMARK_METHODS}" ]]; then
  echo "  methods: ${BENCHMARK_METHODS}"
fi

case "$(printf '%s' "${DRY_RUN}" | tr '[:upper:]' '[:lower:]')" in
true|1|yes)
  printf 'DRY RUN:'
  printf ' %q' "${CONTAINER}" "${container_args[@]}" /opt/conda/bin/Rscript "${r_args[@]}"
  printf '\n'
  exit 0
  ;;
esac

"${CONTAINER}" "${container_args[@]}" /opt/conda/bin/Rscript "${r_args[@]}"

echo "DONE: ${OUT_DIR}"
