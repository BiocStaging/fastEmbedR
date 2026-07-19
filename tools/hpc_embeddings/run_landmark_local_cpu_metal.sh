#!/usr/bin/env bash

# Matched full/50%-landmark validation on the local Mac. CPU is skipped above
# LOCAL_CPU_MAX_N except for named exceptions; Metal is evaluated on all data.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
DATA_ROOT="${DATA_ROOT:-/Users/stefano/Documents/fastEmbedR/Data}"
RESULTS_ROOT="${RESULTS_ROOT:-${REPO_DIR}/results}"
OUT_DIR="${OUT_DIR:-${RESULTS_ROOT}/benchmark_landmark_local_CPU_Metal_$(date +%Y%m%d_%H%M%S)}"
CACHE_DIR="${CACHE_DIR:-${RESULTS_ROOT}/reviewer_precomputed}"
DATASETS="${DATASETS:-COIL20,USPS,FashionMNIST,FlowRepository_FR-FCM-ZYRM_files,flow18,MNIST,imagenet,MetRef,mass41,TabulaMuris,Macosko2015_retina}"
METHODS="${METHODS:-fastEmbedR_opentsne_cpu_full,fastEmbedR_opentsne_cpu_landmark,fastEmbedR_umap_cpu_binary_full,fastEmbedR_umap_cpu_binary_landmark,fastEmbedR_opentsne_metal_full,fastEmbedR_opentsne_metal_landmark,fastEmbedR_umap_metal_binary_full,fastEmbedR_umap_metal_binary_landmark}"
THREADS_GRID="${THREADS_GRID:-1,4}"
SEEDS="${SEEDS:-4,17,42}"
LANDMARK_FRACTION="${LANDMARK_FRACTION:-0.5}"
LOCAL_CPU_MAX_N="${LOCAL_CPU_MAX_N:-100000}"
LOCAL_CPU_EXCEPTIONS="${LOCAL_CPU_EXCEPTIONS:-TabulaMuris}"

mkdir -p "${OUT_DIR}" "${CACHE_DIR}"

exec Rscript "${SCRIPT_DIR}/benchmark_reviewer_validation.R" \
  --backend-group=local \
  --base-dir="${REPO_DIR}" \
  --data-root="${DATA_ROOT}" \
  --out-dir="${OUT_DIR}" \
  --cache-dir="${CACHE_DIR}" \
  --datasets="${DATASETS}" \
  --methods="${METHODS}" \
  --threads-grid="${THREADS_GRID}" \
  --seeds="${SEEDS}" \
  --k=30 \
  --perplexity=30 \
  --landmark-fraction="${LANDMARK_FRACTION}" \
  --reference-validations=FALSE \
  --local-cpu-max-n="${LOCAL_CPU_MAX_N}" \
  --local-cpu-exceptions="${LOCAL_CPU_EXCEPTIONS}" \
  --timeout=10800
