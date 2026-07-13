#!/usr/bin/env bash

set -euo pipefail

ROOT=${FASTEMBEDR_LIGHTNING_ROOT:-$HOME/cuda_pca_test_20260711}
RAPIDS_ENV=${FASTEMBEDR_RAPIDS_ENV:-$HOME/.fastEmbedR/micromamba/envs/fastembedr-cuvs}

export PATH="$RAPIDS_ENV/bin:$PATH"
export LD_LIBRARY_PATH="$RAPIDS_ENV/lib:$RAPIDS_ENV/targets/x86_64-linux/lib:/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}"
export R_LIBS_USER=${R_LIBS_USER:-$HOME/R/x86_64-pc-linux-gnu-library/4.3}

exec Rscript \
  "$ROOT/validate_native_cuda_mnist70k.R" \
  "$ROOT/input/MNIST_float32.RData" \
  "$ROOT/native_cuda_mnist70k"
