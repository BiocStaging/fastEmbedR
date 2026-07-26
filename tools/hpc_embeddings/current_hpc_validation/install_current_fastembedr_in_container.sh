#!/usr/bin/env bash

set -euo pipefail

BUNDLE_DIR="${BUNDLE_DIR:-/scratch/firenze/NN/current_fastembedr_validation}"
R_LIB="${FASTEMBEDR_CURRENT_RLIB:-${BUNDLE_DIR}/Rlib}"
TARBALL="${BUNDLE_DIR}/fastEmbedR_0.99.0.tar.gz"

[[ -f "${TARBALL}" ]] || {
  echo "Missing package tarball: ${TARBALL}" >&2
  exit 1
}

mkdir -p "${R_LIB}"

export CUDA_HOME=/usr/local/cuda
export FAISS_HOME=/opt/faiss
export CUVS_HOME=/opt/rapids
export RAPIDS_HOME=/opt/conda
export RAFT_HOME=/opt/conda
export RMM_HOME=/opt/conda
export CCCL_HOME=/opt/conda/include/rapids
export CUDAHOSTCXX=/opt/conda/bin/g++
export FASTEMBEDR_CUDA_ARCH=89
export FASTEMBEDR_USE_CUDA=1
export FASTEMBEDR_USE_FAISS_GPU=1
export FASTEMBEDR_USE_CUVS=1
export FASTEMBEDR_USE_RAFT=1
export FASTEMBEDR_USE_CUML=0
export LD_LIBRARY_PATH="/opt/rapids/lib:/opt/faiss/lib:/usr/local/cuda/lib64:/opt/conda/lib:/opt/conda/targets/x86_64-linux/lib:${LD_LIBRARY_PATH:-}"

/opt/conda/bin/R CMD INSTALL --preclean -l "${R_LIB}" "${TARBALL}"

FASTEMBEDR_CURRENT_RLIB="${R_LIB}" \
R_PROFILE_USER="${BUNDLE_DIR}/Rprofile.current" \
/opt/conda/bin/Rscript -e '
library(fastEmbedR)
cat("fastEmbedR version:", as.character(packageVersion("fastEmbedR")), "\n")
print(fastEmbedR:::backend_info())
stopifnot(fastEmbedR:::embedding_cuda_available_cpp())
stopifnot(fastEmbedR:::native_cuda_knn_available_cpp())
stopifnot("umap_init" %in% getNamespaceExports("fastEmbedR"))
'
