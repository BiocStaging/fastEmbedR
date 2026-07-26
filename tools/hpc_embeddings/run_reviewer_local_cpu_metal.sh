#!/usr/bin/env bash

set -euo pipefail

REPO_DIR="${REPO_DIR:-/Users/stefano/Documents/umap}"
DATA_ROOT="${DATA_ROOT:-/Users/stefano/Documents/fastEmbedR/Data}"
RESULTS_ROOT="${RESULTS_ROOT:-${REPO_DIR}/results}"
OUT_DIR="${OUT_DIR:-${RESULTS_ROOT}/benchmark_reviewer_local_CPU_Metal_$(date +%Y%m%d_%H%M%S)}"
CACHE_DIR="${CACHE_DIR:-${RESULTS_ROOT}/reviewer_precomputed}"
DATASETS="${DATASETS:-COIL20,USPS,FashionMNIST,FlowRepository_FR-FCM-ZYRM_files,flow18,MNIST,MetRef,mass41,TabulaMuris,Macosko2015_retina,imagenet}"
METHODS="${METHODS:-fastEmbedR_pca_cpu,fastEmbedR_opentsne_cpu_full,fastEmbedR_opentsne_cpu_knn,KODAMA_plslda_opentsne_cpu,KODAMA_knn_opentsne_cpu,fastEmbedR_umap_cpu_fuzzy_full,fastEmbedR_umap_cpu_fuzzy_knn,fastEmbedR_umap_cpu_binary_full,fastEmbedR_umap_cpu_binary_knn,KODAMA_plslda_umap_cpu,KODAMA_knn_umap_cpu,fastEmbedR_pca_metal,fastEmbedR_opentsne_metal_full,fastEmbedR_opentsne_metal_knn,fastEmbedR_umap_metal_fuzzy_full,fastEmbedR_umap_metal_fuzzy_knn,fastEmbedR_umap_metal_binary_full,fastEmbedR_umap_metal_binary_knn}"
THREADS_GRID="${THREADS_GRID:-1,4}"
SEEDS="${SEEDS:-4,17,42}"
K="${K:-30}"
PERPLEXITY="${PERPLEXITY:-30}"
TIMEOUT="${TIMEOUT:-43200}"
QUALITY_MAX_DISTANCE_OPS="${QUALITY_MAX_DISTANCE_OPS:-200000000}"
FORCE="${FORCE:-FALSE}"
LOCAL_CPU_MAX_N="${LOCAL_CPU_MAX_N:-100000}"
LOCAL_CPU_EXCEPTIONS="${LOCAL_CPU_EXCEPTIONS:-TabulaMuris}"

BENCH_R="${REPO_DIR}/tools/hpc_embeddings/benchmark_reviewer_validation.R"
METRICS_R="${REPO_DIR}/tools/hpc_embeddings/publication_metrics.R"
MONITOR_SH="${REPO_DIR}/tools/hpc_embeddings/benchmark_worker_monitor.sh"
REFERENCE_PY="${REPO_DIR}/tools/hpc_embeddings/reference_opentsne_affinity.py"
for required in "${BENCH_R}" "${METRICS_R}" "${MONITOR_SH}" "${REFERENCE_PY}"; do
  [[ -f "${required}" ]] || { echo "Missing ${required}" >&2; exit 1; }
done
[[ -d "${DATA_ROOT}" ]] || { echo "Missing dataset root ${DATA_ROOT}" >&2; exit 1; }
mkdir -p "${OUT_DIR}" "${CACHE_DIR}"

export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 VECLIB_MAXIMUM_THREADS=4
export RCPP_PARALLEL_NUM_THREADS=4

Rscript "${BENCH_R}" \
  --backend-group=local \
  --base-dir="${REPO_DIR}" \
  --data-root="${DATA_ROOT}" \
  --out-dir="${OUT_DIR}" \
  --cache-dir="${CACHE_DIR}" \
  --datasets="${DATASETS}" \
  --methods="${METHODS}" \
  --threads-grid="${THREADS_GRID}" \
  --seeds="${SEEDS}" \
  --k="${K}" \
  --perplexity="${PERPLEXITY}" \
  --timeout="${TIMEOUT}" \
  --quality-max-distance-ops="${QUALITY_MAX_DISTANCE_OPS}" \
  --local-cpu-max-n="${LOCAL_CPU_MAX_N}" \
  --local-cpu-exceptions="${LOCAL_CPU_EXCEPTIONS}" \
  --force="${FORCE}"

echo "DONE: ${OUT_DIR}"
