#!/usr/bin/env bash

set -euo pipefail

REPO_DIR="${REPO_DIR:-/Users/stefano/Documents/umap}"
DATA_ROOT="${DATA_ROOT:-/Users/stefano/Documents/fastEmbedR/Data}"
OUT_DIR="${OUT_DIR:-${REPO_DIR}/results/local_graph_clustering_$(date +%Y%m%d_%H%M%S)}"
DATASETS="${DATASETS:-MetRef,COIL20,USPS,Macosko2015_retina,FashionMNIST,MNIST,TabulaMuris}"
BACKENDS="${BACKENDS:-cpu,metal}"
THREADS_GRID="${THREADS_GRID:-1,4}"
SEEDS="${SEEDS:-4,17,42}"
K="${K:-30}"
TIMEOUT="${TIMEOUT:-43200}"
IGRAPH_MAX_N="${IGRAPH_MAX_N:-50000}"
WALKTRAP_MAX_N="${WALKTRAP_MAX_N:-4000}"
FORCE="${FORCE:-FALSE}"

Rscript "${REPO_DIR}/tools/hpc_embeddings/benchmark_local_graph_clustering.R" \
  --data-root="${DATA_ROOT}" \
  --out-dir="${OUT_DIR}" \
  --datasets="${DATASETS}" \
  --backends="${BACKENDS}" \
  --threads-grid="${THREADS_GRID}" \
  --seeds="${SEEDS}" \
  --k="${K}" \
  --timeout="${TIMEOUT}" \
  --igraph-max-n="${IGRAPH_MAX_N}" \
  --walktrap-max-n="${WALKTRAP_MAX_N}" \
  --force="${FORCE}"

echo "DONE: ${OUT_DIR}"
