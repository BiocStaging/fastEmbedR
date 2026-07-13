#!/usr/bin/env bash

set -euo pipefail

ROOT=${FASTEMBEDR_LIGHTNING_ROOT:-$HOME/cuda_pca_test_20260711}
RAPIDS_ENV=${FASTEMBEDR_RAPIDS_ENV:-$HOME/.fastEmbedR/micromamba/envs/fastembedr-cuvs}
CCCL_HOME=${CCCL_HOME:-$HOME/.fastEmbedR/cccl33}
SOURCE_TARBALL=${1:-$ROOT/fastEmbedR_native_cuvs.tar.gz}
RUN_DIR=${2:-$ROOT/native_cuvs_validation}
VALIDATION_R=${3:-$ROOT/validate_native_cuda_knn.R}
R_LIBRARY=${R_LIBRARY:-$HOME/R/x86_64-pc-linux-gnu-library/4.3}

fail() {
  printf 'ERROR: %s\n' "$*" >&2
  exit 1
}

on_error() {
  status=$?
  printf 'Validation stopped at line %s with status %s.\n' "$1" "$status" >&2
  if [ -f "$RUN_DIR/install.log" ]; then
    printf '%s\n' '--- install.log tail ---' >&2
    tail -n 80 "$RUN_DIR/install.log" >&2
  fi
  exit "$status"
}
trap 'on_error $LINENO' ERR

[ -f "$SOURCE_TARBALL" ] || fail "source tarball not found: $SOURCE_TARBALL"
[ -f "$VALIDATION_R" ] || fail "validation R script not found: $VALIDATION_R"

if [ -x "$RAPIDS_ENV/bin/nvcc" ]; then
  NVCC_BIN="$RAPIDS_ENV/bin/nvcc"
  CUDA_PREFIX="$RAPIDS_ENV"
elif command -v nvcc >/dev/null 2>&1; then
  NVCC_BIN=$(command -v nvcc)
  CUDA_PREFIX=$(dirname "$(dirname "$NVCC_BIN")")
elif [ -x /usr/local/cuda/bin/nvcc ]; then
  NVCC_BIN=/usr/local/cuda/bin/nvcc
  CUDA_PREFIX=/usr/local/cuda
else
  fail "nvcc was not found in the RAPIDS environment, PATH, or /usr/local/cuda"
fi

CUVS_HEADER=$(find "$RAPIDS_ENV/include" -path '*/cuvs/core/c_api.h' -print -quit)
[ -n "$CUVS_HEADER" ] || fail "cuvs/core/c_api.h not found under $RAPIDS_ENV/include"
CUVS_LIBRARY=$(find "$RAPIDS_ENV" -maxdepth 4 -name 'libcuvs_c.so*' -print -quit)
[ -n "$CUVS_LIBRARY" ] || fail "libcuvs_c.so was not found under $RAPIDS_ENV"

printf 'CUDA compiler: %s\n' "$NVCC_BIN"
printf 'CUDA prefix:   %s\n' "$CUDA_PREFIX"
printf 'cuVS header:   %s\n' "$CUVS_HEADER"
printf 'cuVS library:  %s\n' "$CUVS_LIBRARY"

GPU_ARCH=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -n 1 | tr -d '.')
mkdir -p "$RUN_DIR" "$R_LIBRARY"
rm -rf "$RUN_DIR/source"
mkdir -p "$RUN_DIR/source"
tar -xzf "$SOURCE_TARBALL" -C "$RUN_DIR/source"
PACKAGE_DIR=$(find "$RUN_DIR/source" -mindepth 1 -maxdepth 1 -type d | head -n 1)

export PATH="$RAPIDS_ENV/bin:$PATH"
export LD_LIBRARY_PATH="$RAPIDS_ENV/lib:$RAPIDS_ENV/targets/x86_64-linux/lib:${LD_LIBRARY_PATH:-}"
export R_LIBS_USER="$R_LIBRARY"

CUDA_HOME="$CUDA_PREFIX" \
NVCC="$NVCC_BIN" \
FAISS_HOME="$RAPIDS_ENV" \
CUVS_HOME="$RAPIDS_ENV" \
RAFT_HOME="$RAPIDS_ENV" \
RMM_HOME="$RAPIDS_ENV" \
RAPIDS_HOME="$RAPIDS_ENV" \
CCCL_HOME="$CCCL_HOME" \
FASTEMBEDR_USE_CUDA=1 \
FASTEMBEDR_USE_FAISS_GPU=1 \
FASTEMBEDR_USE_CUVS=1 \
FASTEMBEDR_USE_RAFT=1 \
FASTEMBEDR_CUDA_ARCH="$GPU_ARCH" \
R CMD INSTALL --preclean -l "$R_LIBRARY" "$PACKAGE_DIR" \
  >"$RUN_DIR/install.log" 2>&1

Rscript "$VALIDATION_R" "$RUN_DIR/results" \
  >"$RUN_DIR/validation.log" 2>&1

cat "$RUN_DIR/results/native_cuda_knn_smoke.csv"
