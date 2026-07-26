#!/usr/bin/env bash

# Shared runner used by the dataset-specific Slurm launchers. This file has no
# SBATCH header: submit one of jobs_by_dataset/run_*_{cpu1,cpu4,cuda}.sh.

set -euo pipefail

BENCHMARK_DATASET="${BENCHMARK_DATASET:?Set BENCHMARK_DATASET in the dataset launcher.}"
BENCHMARK_BACKEND_GROUP="${BENCHMARK_BACKEND_GROUP:?Set BENCHMARK_BACKEND_GROUP to cpu or cuda.}"
BENCHMARK_THREADS="${BENCHMARK_THREADS:?Set BENCHMARK_THREADS in the dataset launcher.}"
BENCHMARK_SUITE="${BENCHMARK_SUITE:-standard}"

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
case "${BENCHMARK_SUITE}" in
  standard|landmark|kodama) ;;
  *) echo "BENCHMARK_SUITE must be standard, landmark, or kodama." >&2; exit 2 ;;
esac

BASE_DIR="${BASE_DIR:-/scratch/firenze/NN}"
DATA_ROOT="${DATA_ROOT:-${BASE_DIR}/Data}"
IMAGE="${SINGULARITY_IMAGE:-${BASE_DIR}/singularity/fastembedr_cuda.sif}"
SEEDS="${SEEDS:-4,17,42}"
K="${K:-30}"
PERPLEXITY="${PERPLEXITY:-30}"
TIMEOUT="${TIMEOUT:-43200}"
QUALITY_MAX_DISTANCE_OPS="${QUALITY_MAX_DISTANCE_OPS:-200000000}"
LANDMARK_FRACTION="${LANDMARK_FRACTION:-0.2}"
REFERENCE_VALIDATIONS="${REFERENCE_VALIDATIONS:-TRUE}"
BENCHMARK_METHODS="${BENCHMARK_METHODS:-}"
KODAMA_M="${KODAMA_M:-100}"
KODAMA_TCYCLE="${KODAMA_TCYCLE:-20}"
KODAMA_NCOMP="${KODAMA_NCOMP:-50}"
KODAMA_LANDMARKS="${KODAMA_LANDMARKS:-10000000}"
KODAMA_GRAPH_NEIGHBORS="${KODAMA_GRAPH_NEIGHBORS:-100}"
KODAMA_N_EPOCHS="${KODAMA_N_EPOCHS:-200}"
KODAMA_N_ITER="${KODAMA_N_ITER:-500}"
FORCE="${FORCE:-FALSE}"
DRY_RUN="${DRY_RUN:-FALSE}"
CUDA_PREFLIGHT="${CUDA_PREFLIGHT:-TRUE}"

safe_dataset="$(printf '%s' "${BENCHMARK_DATASET}" | tr -c 'A-Za-z0-9_.-' '_')"
cache_profile="${BENCHMARK_BACKEND_GROUP}${BENCHMARK_THREADS}"
if [[ "${BENCHMARK_BACKEND_GROUP}" == "cuda" ]]; then
  profile="cuda"
else
  profile="cpu${BENCHMARK_THREADS}"
fi
run_stamp="${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}"
job_suffix="${SLURM_JOB_ID:-manual}"
RESULTS_ROOT="${RESULTS_ROOT:-${BASE_DIR}/fastEmbedR-results}"
LAYOUT_ROOT="${LAYOUT_ROOT:-${BASE_DIR}/fastEmbedR-rlayout}"
run_folder="${run_stamp}_${job_suffix}"
OUT_DIR="${OUT_DIR:-${RESULTS_ROOT}/${safe_dataset}/${BENCHMARK_SUITE}/${profile}/${run_folder}}"
LAYOUT_DIR="${LAYOUT_DIR:-${LAYOUT_ROOT}/${safe_dataset}/${BENCHMARK_SUITE}/${profile}/${run_folder}}"
CACHE_DIR="${CACHE_DIR:-${DATA_ROOT}/_fastEmbedR_precomputed_jobs/${safe_dataset}/${cache_profile}}"

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

image_identity="$(
  {
    "${CONTAINER}" inspect "${IMAGE}" 2>/dev/null || true
    if stat -c '%s:%Y' "${IMAGE}" >/dev/null 2>&1; then
      stat -c '%s:%Y' "${IMAGE}"
    else
      stat -f '%z:%m' "${IMAGE}"
    fi
  } | cksum | awk '{print $1}'
)"
if command -v readlink >/dev/null 2>&1; then
  image_resolved="$(readlink -f "${IMAGE}" 2>/dev/null || printf '%s\n' "${IMAGE}")"
else
  image_resolved="${IMAGE}"
fi
KODAMA_CACHE_TAG="${KODAMA_CACHE_TAG:-image_${image_identity}}"

mkdir -p \
  "${BASE_DIR}/benchmark_logs" "${RESULTS_ROOT}" "${LAYOUT_ROOT}" \
  "${OUT_DIR}" "${LAYOUT_DIR}" "${CACHE_DIR}"

export OMP_NUM_THREADS="${BENCHMARK_THREADS}"
export OPENBLAS_NUM_THREADS="${BENCHMARK_THREADS}"
export MKL_NUM_THREADS="${BENCHMARK_THREADS}"
export VECLIB_MAXIMUM_THREADS="${BENCHMARK_THREADS}"
export RCPP_PARALLEL_NUM_THREADS="${BENCHMARK_THREADS}"
export APPTAINERENV_OMP_NUM_THREADS="${BENCHMARK_THREADS}"
export APPTAINERENV_OPENBLAS_NUM_THREADS="${BENCHMARK_THREADS}"
export APPTAINERENV_MKL_NUM_THREADS="${BENCHMARK_THREADS}"
export APPTAINERENV_RCPP_PARALLEL_NUM_THREADS="${BENCHMARK_THREADS}"
export APPTAINERENV_FASTEMBEDR_IMAGE_PATH="${IMAGE}"
export APPTAINERENV_FASTEMBEDR_IMAGE_RESOLVED="${image_resolved}"
export APPTAINERENV_FASTEMBEDR_IMAGE_IDENTITY="${image_identity}"

container_args=(exec)
if [[ "${BENCHMARK_BACKEND_GROUP}" == "cuda" ]]; then
  container_args+=(--nv)
  export APPTAINERENV_LD_LIBRARY_PATH="/opt/rapids/lib:/opt/faiss/lib:/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}"
fi
container_args+=(--bind "${BASE_DIR}:${BASE_DIR}" --pwd "${BASE_DIR}" "${IMAGE}")

cuda_preflight() {
  local health_log="${OUT_DIR}/cuda_preflight.log"
  local ecc_output=""
  local allocation_output=""
  {
    echo "node=${SLURMD_NODENAME:-$(hostname)}"
    echo "timestamp=$(date '+%Y-%m-%dT%H:%M:%S%z')"
    nvidia-smi -L
  } >"${health_log}" 2>&1

  ecc_output="$(
    nvidia-smi \
      --query-gpu=ecc.errors.uncorrected.volatile.total \
      --format=csv,noheader,nounits 2>>"${health_log}" || true
  )"
  printf 'uncorrectable_ecc_volatile=%s\n' "${ecc_output}" >>"${health_log}"
  if printf '%s\n' "${ecc_output}" |
      awk '$1 ~ /^[0-9]+$/ && $1 > 0 { bad=1 } END { exit !bad }'; then
    echo "CUDA preflight failed: uncorrectable ECC errors are present on ${SLURMD_NODENAME:-$(hostname)}." >&2
    echo "This is a GPU/node health failure, not an embedding failure. See ${health_log}." >&2
    return 70
  fi

  if ! allocation_output="$(
    "${CONTAINER}" "${container_args[@]}" /opt/conda/bin/python -c \
      'import cupy as cp; x=cp.arange(4096,dtype=cp.float32); assert float(cp.sum(x).get()) > 0' \
      2>&1
  )"; then
    printf '%s\n' "${allocation_output}" >>"${health_log}"
    if printf '%s' "${allocation_output}" |
        grep -Eqi 'uncorrectable ECC|cudaErrorECCUncorrectable|GPU has fallen off the bus'; then
      echo "CUDA preflight failed: the device cannot complete a basic allocation/kernel test." >&2
      echo "This is a GPU/node health failure, not an embedding failure. See ${health_log}." >&2
      return 70
    fi
    echo "CUDA preflight warning: CuPy allocation test was unavailable; continuing after nvidia-smi health check." >&2
    printf 'preflight_warning=cupy_test_unavailable\n' >>"${health_log}"
  else
    printf 'cupy_allocation=success\n' >>"${health_log}"
  fi
}

r_args=(
  "${BENCH_R}"
  "--backend-group=${BENCHMARK_BACKEND_GROUP}"
  "--base-dir=${BASE_DIR}"
  "--data-root=${DATA_ROOT}"
  "--out-dir=${OUT_DIR}"
  "--layout-dir=${LAYOUT_DIR}"
  "--cache-dir=${CACHE_DIR}"
  "--datasets=${BENCHMARK_DATASET}"
  "--threads-grid=${BENCHMARK_THREADS}"
  "--seeds=${SEEDS}"
  "--k=${K}"
  "--perplexity=${PERPLEXITY}"
  "--timeout=${TIMEOUT}"
  "--quality-max-distance-ops=${QUALITY_MAX_DISTANCE_OPS}"
  "--landmark-fraction=${LANDMARK_FRACTION}"
  "--reference-validations=${REFERENCE_VALIDATIONS}"
  "--kodama-m=${KODAMA_M}"
  "--kodama-tcycle=${KODAMA_TCYCLE}"
  "--kodama-ncomp=${KODAMA_NCOMP}"
  "--kodama-landmarks=${KODAMA_LANDMARKS}"
  "--kodama-graph-neighbors=${KODAMA_GRAPH_NEIGHBORS}"
  "--kodama-n-epochs=${KODAMA_N_EPOCHS}"
  "--kodama-n-iter=${KODAMA_N_ITER}"
  "--kodama-cache-tag=${KODAMA_CACHE_TAG}"
  "--force=${FORCE}"
)
if [[ -n "${BENCHMARK_METHODS}" ]]; then
  r_args+=("--methods=${BENCHMARK_METHODS}")
fi

echo "fastEmbedR reviewer dataset job"
echo "  dataset: ${BENCHMARK_DATASET}"
echo "  suite:   ${BENCHMARK_SUITE}"
echo "  backend: ${BENCHMARK_BACKEND_GROUP}"
echo "  threads: ${BENCHMARK_THREADS}"
echo "  data:    ${DATA_ROOT}"
echo "  output:  ${OUT_DIR}"
echo "  layouts: ${LAYOUT_DIR}"
echo "  cache:   ${CACHE_DIR}"
echo "  image:   ${IMAGE}"
echo "  resolved image: ${image_resolved}"
echo "  image identity: ${image_identity}"
echo "  KODAMA cache tag: ${KODAMA_CACHE_TAG}"
echo "  landmark fraction: ${LANDMARK_FRACTION}"
echo "  reference validations: ${REFERENCE_VALIDATIONS}"
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

if [[ "${BENCHMARK_BACKEND_GROUP}" == "cuda" ]]; then
  case "$(printf '%s' "${CUDA_PREFLIGHT}" | tr '[:upper:]' '[:lower:]')" in
  true|1|yes)
    if ! cuda_preflight; then
      restart_count="${SLURM_RESTART_COUNT:-0}"
      if [[ -n "${SLURM_JOB_ID:-}" && "${restart_count}" -lt 2 ]] &&
          command -v scontrol >/dev/null 2>&1; then
        unhealthy_node="${SLURMD_NODENAME:-$(hostname)}"
        scontrol update JobId="${SLURM_JOB_ID}" ExcNodeList="${unhealthy_node}" \
          >/dev/null 2>&1 || true
        if scontrol requeue "${SLURM_JOB_ID}"; then
          echo "Requeued CUDA job ${SLURM_JOB_ID} away from ${unhealthy_node}."
          exit 0
        fi
      fi
      exit 70
    fi
    ;;
  esac
fi

"${CONTAINER}" "${container_args[@]}" /opt/conda/bin/Rscript "${r_args[@]}"

echo "DONE: ${OUT_DIR}"
