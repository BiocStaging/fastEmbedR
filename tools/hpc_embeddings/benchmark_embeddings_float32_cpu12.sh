#!/usr/bin/env bash

#SBATCH --account=immunology
#SBATCH --partition=ada
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --time=48:00:00
#SBATCH --job-name="fastEmbedR_emb_CPU1_12"
#SBATCH --chdir=/scratch/firenze/NN
#SBATCH --output=/scratch/firenze/NN/benchmark_logs/fastEmbedR_emb_cpu12_%j.out
#SBATCH --error=/scratch/firenze/NN/benchmark_logs/fastEmbedR_emb_cpu12_%j.err

set -euo pipefail

# CPU-only embedding benchmark for publication.
#
# fastEmbedR methods use the *_float32.RData files.
# Reference methods use the standard dataset .RData files.
# CPU methods are run with each value in CPU_THREADS_GRID, by default 1 and 12
# threads, so the same output folder contains the single-core and multicore
# comparison requested for the manuscript.
#
# Submit on the HPC with:
#   sbatch /scratch/firenze/NN/benchmark_embeddings_float32_cpu12.sh

export BASE_DIR="${BASE_DIR:-/scratch/firenze/NN}"
export DATA_ROOT="${DATA_ROOT:-${BASE_DIR}/Data}"
export SCRIPT_DIR="${SCRIPT_DIR:-${BASE_DIR}}"
export LOG_DIR="${LOG_DIR:-${BASE_DIR}/benchmark_logs}"
export OUT_DIR="${OUT_DIR:-${BASE_DIR}/benchmark_embeddings_float32_CPU1_12_$(date +%Y%m%d_%H%M%S)}"
export SINGULARITY_IMAGE="${SINGULARITY_IMAGE:-${BASE_DIR}/singularity/fastembedr_cuda.sif}"
export THREADS="${THREADS:-12}"
export CPU_THREADS_GRID="${CPU_THREADS_GRID:-1,12}"
export TIMEOUT="${TIMEOUT:-10800}"
export K="${K:-30}"
export PERPLEXITY="${PERPLEXITY:-15}"
export SEED="${SEED:-4}"
export FORCE="${FORCE:-TRUE}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${OUT_DIR}/.cache}"
export NUMBA_CACHE_DIR="${NUMBA_CACHE_DIR:-${OUT_DIR}/numba_cache}"
export CUPY_CACHE_DIR="${CUPY_CACHE_DIR:-${OUT_DIR}/cupy_cache}"

export DATASETS="${DATASETS:-COIL20,USPS,FashionMNIST,FlowRepository_FR-FCM-ZYRM_files,flow18,MNIST,imagenet,MetRef,mass41,TabulaMuris}"
export METHODS="${METHODS:-fastEmbedR_opentsne_cpu,fastEmbedR_umap_cpu_fuzzy,fastEmbedR_umap_cpu_binary,Rtsne_full,KlugerLab_FItSNE,python_opentsne_fft,umap_package,uwot_default,uwot_fast_sgd,python_umap_learn}"

export OMP_NUM_THREADS="${THREADS}"
export OPENBLAS_NUM_THREADS="${THREADS}"
export MKL_NUM_THREADS="${THREADS}"
export VECLIB_MAXIMUM_THREADS="${THREADS}"
export RCPP_PARALLEL_NUM_THREADS="${THREADS}"
export APPTAINERENV_OMP_NUM_THREADS="${THREADS}"
export APPTAINERENV_OPENBLAS_NUM_THREADS="${THREADS}"
export APPTAINERENV_MKL_NUM_THREADS="${THREADS}"
export APPTAINERENV_RCPP_PARALLEL_NUM_THREADS="${THREADS}"
export APPTAINERENV_XDG_CACHE_HOME="${XDG_CACHE_HOME}"
export APPTAINERENV_NUMBA_CACHE_DIR="${NUMBA_CACHE_DIR}"
export APPTAINERENV_CUPY_CACHE_DIR="${CUPY_CACHE_DIR}"
export SINGULARITYENV_OMP_NUM_THREADS="${THREADS}"
export SINGULARITYENV_OPENBLAS_NUM_THREADS="${THREADS}"
export SINGULARITYENV_MKL_NUM_THREADS="${THREADS}"
export SINGULARITYENV_RCPP_PARALLEL_NUM_THREADS="${THREADS}"
export SINGULARITYENV_XDG_CACHE_HOME="${XDG_CACHE_HOME}"
export SINGULARITYENV_NUMBA_CACHE_DIR="${NUMBA_CACHE_DIR}"
export SINGULARITYENV_CUPY_CACHE_DIR="${CUPY_CACHE_DIR}"

mkdir -p "${LOG_DIR}" "${OUT_DIR}" "${XDG_CACHE_HOME}" "${NUMBA_CACHE_DIR}" "${CUPY_CACHE_DIR}"
cd "${BASE_DIR}"

if [[ -f "${SCRIPT_DIR}/benchmark_embeddings_float32_publication.R" ]]; then
  BENCH_R="${SCRIPT_DIR}/benchmark_embeddings_float32_publication.R"
elif [[ -f "${BASE_DIR}/benchmark_embeddings_float32_publication.R" ]]; then
  BENCH_R="${BASE_DIR}/benchmark_embeddings_float32_publication.R"
else
  echo "Cannot find benchmark_embeddings_float32_publication.R in ${SCRIPT_DIR} or ${BASE_DIR}" >&2
  exit 1
fi

RUNNER=()
if [[ -n "${SINGULARITY_IMAGE}" && -f "${SINGULARITY_IMAGE}" ]]; then
  CONTAINER_BIN="${CONTAINER_BIN:-$(command -v apptainer || command -v singularity || true)}"
  if [[ -z "${CONTAINER_BIN}" ]]; then
    echo "Cannot find apptainer or singularity although SINGULARITY_IMAGE is set." >&2
    exit 1
  fi
  RUNNER=("${CONTAINER_BIN}" exec --bind "${BASE_DIR}:${BASE_DIR}" --pwd "${BASE_DIR}" "${SINGULARITY_IMAGE}")
fi

{
  echo "Starting CPU embedding benchmark"
  echo "BASE_DIR=${BASE_DIR}"
  echo "DATA_ROOT=${DATA_ROOT}"
  echo "OUT_DIR=${OUT_DIR}"
  echo "THREADS=${THREADS}"
  echo "CPU_THREADS_GRID=${CPU_THREADS_GRID}"
  echo "TIMEOUT=${TIMEOUT}"
  echo "FORCE=${FORCE}"
  echo "DATASETS=${DATASETS}"
  echo "METHODS=${METHODS}"
	  echo "Quality outputs will be written to:"
	  echo "  ${OUT_DIR}/embedding_parameter_table.csv"
	  echo "  ${OUT_DIR}/embedding_parameter_table.md"
	  echo "  ${OUT_DIR}/embedding_quality_table.csv"
	  echo "  ${OUT_DIR}/embedding_quality_table.md"
	  echo "  ${OUT_DIR}/embedding_runtime_quality_pareto.png"
  "${RUNNER[@]}" Rscript "${BENCH_R}" \
    --script="${BENCH_R}" \
    --backend_group=cpu \
    --base_dir="${BASE_DIR}" \
    --data_root="${DATA_ROOT}" \
    --out_dir="${OUT_DIR}" \
    --datasets="${DATASETS}" \
    --methods="${METHODS}" \
    --threads="${THREADS}" \
    --thread_grid="${CPU_THREADS_GRID}" \
    --timeout="${TIMEOUT}" \
    --k="${K}" \
    --perplexity="${PERPLEXITY}" \
    --seed="${SEED}" \
    --force="${FORCE}"
	  echo "Parameter table: ${OUT_DIR}/embedding_parameter_table.csv"
	  echo "Parameter manuscript table: ${OUT_DIR}/embedding_parameter_table.md"
	  echo "Quality table: ${OUT_DIR}/embedding_quality_table.csv"
  echo "Quality manuscript table: ${OUT_DIR}/embedding_quality_table.md"
  echo "Runtime-quality Pareto plot: ${OUT_DIR}/embedding_runtime_quality_pareto.png"
  echo "DONE: ${OUT_DIR}"
} 2>&1 | tee -a "${OUT_DIR}/benchmark_cpu12.log"
