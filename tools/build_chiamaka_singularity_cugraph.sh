#!/usr/bin/env bash
set -euo pipefail

REMOTE="${REMOTE:-chiamaka@137.158.224.178}"
REMOTE_DIR="${REMOTE_DIR:-/mnt/sata_ssd/fastEmbedR/singularity}"
SSH_CMD="${SSH_CMD:-ssh}"
SCP_CMD="${SCP_CMD:-scp}"
REMOTE_EXEC="${REMOTE_EXEC:-singularity exec --nv}"
LOCAL_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FAISSR_ROOT="${FAISSR_ROOT:-/Users/stefano/Documents/faissR 2}"
PKG_TARBALL="${LOCAL_ROOT}/fastEmbedR_0.1.0.tar.gz"
FAISSR_TARBALL="${FAISSR_TARBALL:-${FAISSR_ROOT}/faissR_0.1.0.tar.gz}"
DEF_FILE="${LOCAL_ROOT}/tools/hpc_embeddings/fastembedr_cuda_multiarch_cugraph.def"
REMOTE_DEF="${REMOTE_DIR}/fastembedr_cuda_multiarch_cugraph.def"
REMOTE_SIF="${REMOTE_DIR}/fastembedr_cuda.sif"
REMOTE_LOG="${REMOTE_DIR}/build_fastembedr_cuda_cugraph_$(date +%Y%m%d_%H%M%S).log"
LOCAL_COPY_DIR="${LOCAL_COPY_DIR:-${LOCAL_ROOT}/singularity}"

if [[ ! -s "${PKG_TARBALL}" ]]; then
  echo "Missing package tarball: ${PKG_TARBALL}" >&2
  echo "Run: R CMD build --no-build-vignettes ." >&2
  exit 1
fi
if [[ ! -s "${FAISSR_TARBALL}" ]]; then
  echo "Missing faissR package tarball: ${FAISSR_TARBALL}" >&2
  echo "Run: (cd '${FAISSR_ROOT}' && R CMD build --no-build-vignettes faissR_push)" >&2
  exit 1
fi
if [[ ! -s "${DEF_FILE}" ]]; then
  echo "Missing Singularity definition: ${DEF_FILE}" >&2
  exit 1
fi

echo "Remote:       ${REMOTE}"
echo "Remote dir:   ${REMOTE_DIR}"
echo "Definition:   ${DEF_FILE}"
echo "Package:      ${PKG_TARBALL}"
echo "faissR:       ${FAISSR_TARBALL}"
echo "Output image: ${REMOTE_SIF}"
echo "Build log:    ${REMOTE_LOG}"

${SSH_CMD} "${REMOTE}" "mkdir -p '${REMOTE_DIR}' '${REMOTE_DIR}/patched_libs'"
${SCP_CMD} "${PKG_TARBALL}" "${REMOTE}:${REMOTE_DIR}/fastEmbedR_0.1.0.tar.gz"
${SCP_CMD} "${FAISSR_TARBALL}" "${REMOTE}:${REMOTE_DIR}/faissR_0.1.0.tar.gz"
${SCP_CMD} "${DEF_FILE}" "${REMOTE}:${REMOTE_DEF}"

${SSH_CMD} "${REMOTE}" "cd '${REMOTE_DIR}' && \
  (apptainer build --fakeroot --notest --force '${REMOTE_SIF}' '${REMOTE_DEF}' || singularity build --fakeroot --notest --force '${REMOTE_SIF}' '${REMOTE_DEF}') \
  2>&1 | tee '${REMOTE_LOG}'"

${SSH_CMD} "${REMOTE}" "${REMOTE_EXEC} '${REMOTE_SIF}' Rscript - <<'EOF_R'
library(float)
library(faissR)
library(fastEmbedR)

cat('backend_info()\\n')
print(faissR::backend_info())
stopifnot(faissR::faiss_available())
stopifnot(faissR::cuda_available())
stopifnot(faissR::cuvs_available())
info <- faissR::backend_info()
if ('cugraph' %in% info\$backend) {
  stopifnot(isTRUE(info\$available[match('cugraph', info\$backend)]))
}

set.seed(1)
x64 <- matrix(runif(5000 * 32), nrow = 5000)
x32 <- float::fl(x64)

cat('CPU float32 KNN smoke\\n')
knn_cpu <- faissR::nn(x32, k = 15, backend = 'cpu', method = 'hnsw',
                      tuning = 'auto', target_recall = 0.99,
                      output = 'float', exclude_self = TRUE)
stopifnot(nrow(knn_cpu\$indices) == 5000L, ncol(knn_cpu\$indices) == 15L)

cat('CUDA float32 KNN smoke\\n')
knn_cuda <- faissR::nn(x32, k = 15, backend = 'cuda', method = 'auto',
                       tuning = 'auto', target_recall = 0.99,
                       output = 'float', exclude_self = TRUE)
stopifnot(nrow(knn_cuda\$indices) == 5000L, ncol(knn_cuda\$indices) == 15L)
stopifnot(inherits(knn_cuda\$distances, 'float32'))

cat('fastEmbedR CPU openTSNE float32 smoke\\n')
y_cpu <- fastEmbedR::opentsne(x32, perplexity = 15, backend = 'cpu',
                              n_iter = 50, early_exaggeration_iter = 10,
                              n_threads = 4)
stopifnot(all(dim(y_cpu\$layout) == c(5000L, 2L)))

cat('fastEmbedR CUDA openTSNE float32 smoke\\n')
y_cuda <- fastEmbedR::opentsne(x32, perplexity = 15, backend = 'cuda',
                               n_iter = 50, early_exaggeration_iter = 10,
                               n_threads = 4)
stopifnot(all(dim(y_cuda\$layout) == c(5000L, 2L)))

cat('fastEmbedR CPU/CUDA UMAP float32 smoke\\n')
u_cpu <- fastEmbedR::umap(x32, n_neighbors = 15, backend = 'cpu',
                          graph_mode = 'fuzzy', n_threads = 4)
u_cuda <- fastEmbedR::umap(x32, n_neighbors = 15, backend = 'cuda',
                           graph_mode = 'fuzzy', n_threads = 4)
stopifnot(all(dim(u_cpu\$layout) == c(5000L, 2L)))
stopifnot(all(dim(u_cuda\$layout) == c(5000L, 2L)))

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

mkdir -p "${LOCAL_COPY_DIR}"
${SCP_CMD} "${REMOTE}:${REMOTE_SIF}" "${LOCAL_COPY_DIR}/fastembedr_cuda.sif"
echo "Copied image to ${LOCAL_COPY_DIR}/fastembedr_cuda.sif"
