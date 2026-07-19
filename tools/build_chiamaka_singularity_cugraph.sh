#!/usr/bin/env bash
set -euo pipefail

REMOTE="${REMOTE:-chiamaka@137.158.224.178}"
REMOTE_DIR="${REMOTE_DIR:-/mnt/sata_ssd/fastEmbedR/singularity}"
SSH_CMD="${SSH_CMD:-ssh}"
SCP_CMD="${SCP_CMD:-scp}"
REMOTE_EXEC="${REMOTE_EXEC:-singularity exec --nv}"
LOCAL_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PKG_TARBALL="${PKG_TARBALL:-}"
DEF_FILE="${LOCAL_ROOT}/tools/hpc_embeddings/fastembedr_cuda_multiarch_cugraph.def"
REMOTE_DEF="${REMOTE_DIR}/fastembedr_cuda_multiarch_cugraph.def"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
REMOTE_SIF="${REMOTE_SIF:-${REMOTE_DIR}/fastembedr_cuda_${STAMP}.sif}"
REMOTE_LOG="${REMOTE_LOG:-${REMOTE_DIR}/build_fastembedr_cuda_cugraph_${STAMP}.log}"
LOCAL_COPY_DIR="${LOCAL_COPY_DIR:-${LOCAL_ROOT}/singularity}"
COPY_LOCAL="${COPY_LOCAL:-false}"
BUILD_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/fastembedr-singularity-build.XXXXXX")"
trap 'rm -rf "${BUILD_ROOT}"' EXIT

if [[ -z "${PKG_TARBALL}" ]]; then
  echo "Building current fastEmbedR source archive"
  (
    cd "${BUILD_ROOT}"
    R CMD build --no-build-vignettes "${LOCAL_ROOT}"
  )
  PKG_TARBALL="$(find "${BUILD_ROOT}" -maxdepth 1 -name 'fastEmbedR_*.tar.gz' -print -quit)"
fi
[[ -s "${PKG_TARBALL}" ]] || { echo "Missing fastEmbedR archive" >&2; exit 1; }
if [[ ! -s "${DEF_FILE}" ]]; then
  echo "Missing Singularity definition: ${DEF_FILE}" >&2
  exit 1
fi

echo "Remote:       ${REMOTE}"
echo "Remote dir:   ${REMOTE_DIR}"
echo "Definition:   ${DEF_FILE}"
echo "Package:      ${PKG_TARBALL}"
echo "Output image: ${REMOTE_SIF}"
echo "Build log:    ${REMOTE_LOG}"

${SSH_CMD} "${REMOTE}" "mkdir -p '${REMOTE_DIR}' '${REMOTE_DIR}/patched_libs'"
${SCP_CMD} "${PKG_TARBALL}" "${REMOTE}:${REMOTE_DIR}/fastEmbedR_source.tar.gz"
${SCP_CMD} "${DEF_FILE}" "${REMOTE}:${REMOTE_DEF}"

${SSH_CMD} "${REMOTE}" "cd '${REMOTE_DIR}' && \
  (apptainer build --fakeroot --notest --force '${REMOTE_SIF}' '${REMOTE_DEF}' || singularity build --fakeroot --notest --force '${REMOTE_SIF}' '${REMOTE_DEF}') \
  2>&1 | tee '${REMOTE_LOG}'"

${SSH_CMD} "${REMOTE}" "${REMOTE_EXEC} '${REMOTE_SIF}' Rscript - <<'EOF_R'
library(float)
library(fastEmbedR)

cat('backend_info()\\n')
print(fastEmbedR:::backend_info())
stopifnot(isTRUE(fastEmbedR:::native_cuda_knn_available_cpp()))
stopifnot(isTRUE(fastEmbedR:::embedding_cuda_available_cpp()))

layout_dim <- function(x) {
  if (inherits(x, 'float32') || is.matrix(x)) return(dim(x))
  for (name in c('layout', 'embedding', 'Y')) {
    if (!is.null(x[[name]])) return(dim(x[[name]]))
  }
  NULL
}

set.seed(1)
x64 <- matrix(runif(5000 * 32), nrow = 5000)
x32 <- float::fl(x64)

cat('CPU float32 KNN smoke\\n')
knn_cpu <- fastEmbedR::precompute_knn(
  x32, k = 15, backend = 'cpu', n_threads = 4
)
stopifnot(nrow(knn_cpu\$indices) == 5000L, ncol(knn_cpu\$indices) == 15L)

cat('CUDA float32 KNN smoke\\n')
knn_cuda <- fastEmbedR::precompute_knn(
  x32, k = 15, backend = 'cuda', n_threads = 4
)
stopifnot(inherits(knn_cuda, 'fastEmbedR_gpu_knn'))

cat('fastEmbedR CUDA PCA float32 smoke\\n')
pca_cuda <- fastEmbedR::pca(
  x32, ncomp = 2, backend = 'cuda', opentsne_init = TRUE
)
stopifnot(all(dim(pca_cuda\$scores) == c(5000L, 2L)))

cat('fastEmbedR CPU openTSNE float32 smoke\\n')
y_cpu <- fastEmbedR::opentsne(x32, perplexity = 15, backend = 'cpu',
                              n_iter = 50, early_exaggeration_iter = 10,
                              n_threads = 4)
stopifnot(all(layout_dim(y_cpu) == c(5000L, 2L)))

cat('fastEmbedR CUDA openTSNE float32 smoke\\n')
y_cuda <- fastEmbedR::opentsne(x32, perplexity = 15, backend = 'cuda',
                               n_iter = 50, early_exaggeration_iter = 10,
                               n_threads = 4)
stopifnot(all(layout_dim(y_cuda) == c(5000L, 2L)))

cat('fastEmbedR CUDA KNN-input openTSNE float32 smoke\\n')
y_cuda_knn <- fastEmbedR::opentsne_knn(
  knn_cuda, perplexity = 15, backend = 'cuda',
  init_data = pca_cuda\$opentsne_init,
  n_iter = 50, early_exaggeration_iter = 10, n_threads = 4
)
stopifnot(all(layout_dim(y_cuda_knn) == c(5000L, 2L)))

cat('fastEmbedR CPU/CUDA UMAP float32 smoke\\n')
u_cpu <- fastEmbedR::umap(x32, n_neighbors = 15, backend = 'cpu',
                          graph_mode = 'fuzzy', n_threads = 4)
u_cuda <- fastEmbedR::umap(x32, n_neighbors = 15, backend = 'cuda',
                           graph_mode = 'fuzzy', n_threads = 4)
stopifnot(all(layout_dim(u_cpu) == c(5000L, 2L)))
stopifnot(all(layout_dim(u_cuda) == c(5000L, 2L)))

u_cuda_knn <- fastEmbedR::umap_knn(
  knn_cuda, backend = 'cuda', graph_mode = 'binary', n_threads = 4
)
stopifnot(all(layout_dim(u_cuda_knn) == c(5000L, 2L)))

cat('OK\\n')
EOF_R"

${SSH_CMD} "${REMOTE}" "${REMOTE_EXEC} '${REMOTE_SIF}' python - <<'EOF_PY'
import importlib
import numpy as np
from sklearn.datasets import load_iris
from openTSNE import TSNE
import umap

for name in ('openTSNE', 'umap', 'sklearn', 'numba', 'pynndescent', 'cuml'):
    if importlib.util.find_spec(name) is None:
        raise SystemExit(f'Missing Python module: {name}')

X = load_iris().data.astype(np.float32)
Y_tsne = TSNE(
    n_components=2,
    perplexity=5,
    n_iter=50,
    early_exaggeration_iter=10,
    initialization='pca',
    random_state=1,
    n_jobs=2,
).fit(X)
assert np.asarray(Y_tsne).shape == (150, 2)

Y_umap = umap.UMAP(
    n_neighbors=10,
    n_components=2,
    n_epochs=20,
    random_state=1,
).fit_transform(X)
assert Y_umap.shape == (150, 2)

import cuml
from cuml.manifold import UMAP as cuMLUMAP
from cuml.manifold import TSNE as cuMLTSNE
print('Python openTSNE, umap-learn, and cuML imports OK')
print('cuML version:', getattr(cuml, '__version__', 'unknown'))
print(cuMLUMAP, cuMLTSNE)
EOF_PY"

${SSH_CMD} "${REMOTE}" "${REMOTE_EXEC} '${REMOTE_SIF}' Rscript - <<'EOF_R'
source('/opt/fit-sne/bin/fast_tsne.R')
set.seed(1)
x <- matrix(rnorm(600), 150, 4)
y <- fftRtsne(
  X = x, dims = 2L, perplexity = 5, max_iter = 50L,
  stop_early_exag_iter = 10L, rand_seed = 1L, nthreads = 2L,
  fast_tsne_path = '/opt/fit-sne/bin/fast_tsne', fft_not_bh = TRUE
)
stopifnot(is.matrix(y), all(dim(y) == c(150L, 2L)))
cat('FIt-SNE binary and R wrapper OK\n')
EOF_R"

${SSH_CMD} "${REMOTE}" "cd '${REMOTE_DIR}' && ln -sfn '$(basename "${REMOTE_SIF}")' fastembedr_cuda_latest.sif"

if [[ "${COPY_LOCAL}" == "true" ]]; then
  mkdir -p "${LOCAL_COPY_DIR}"
  ${SCP_CMD} "${REMOTE}:${REMOTE_SIF}" "${LOCAL_COPY_DIR}/$(basename "${REMOTE_SIF}")"
  echo "Copied image to ${LOCAL_COPY_DIR}/$(basename "${REMOTE_SIF}")"
fi

echo "Validated image: ${REMOTE_SIF}"
echo "Latest symlink:  ${REMOTE_DIR}/fastembedr_cuda_latest.sif"
