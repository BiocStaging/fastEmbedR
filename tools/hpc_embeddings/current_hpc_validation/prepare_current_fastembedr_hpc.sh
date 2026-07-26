#!/usr/bin/env bash

#SBATCH --account=immunology
#SBATCH --partition=ada
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --job-name=feR_current_build
#SBATCH --chdir=/scratch/firenze/NN
#SBATCH --output=/scratch/firenze/NN/benchmark_logs/feR_current_build_%j.out
#SBATCH --error=/scratch/firenze/NN/benchmark_logs/feR_current_build_%j.err

set -euo pipefail

BASE_DIR=/scratch/firenze/NN
BUNDLE_DIR="${BASE_DIR}/current_fastembedr_validation"
IMAGE="${BASE_DIR}/singularity/fastembedr_cuda.sif"
SOURCE_DIR="${BUNDLE_DIR}/source"
PACKAGE_SOURCE="${BUNDLE_DIR}/package_source"

[[ -f "${IMAGE}" ]] || { echo "Missing image: ${IMAGE}" >&2; exit 1; }
[[ -d "${SOURCE_DIR}/.git" ]] || {
  echo "Missing current source checkout: ${SOURCE_DIR}" >&2
  exit 1
}

mkdir -p "${BASE_DIR}/benchmark_logs" "${BUNDLE_DIR}/Rlib"

CONTAINER="$(command -v apptainer || command -v singularity || true)"
[[ -n "${CONTAINER}" ]] || {
  echo "apptainer/singularity is unavailable" >&2
  exit 1
}

rm -rf "${PACKAGE_SOURCE}"
mkdir -p "${PACKAGE_SOURCE}"
git -C "${SOURCE_DIR}" archive HEAD | tar -x -C "${PACKAGE_SOURCE}"

"${CONTAINER}" exec \
  --bind "${BASE_DIR}:${BASE_DIR}" \
  --pwd "${BUNDLE_DIR}" \
  "${IMAGE}" \
  /opt/conda/bin/R CMD build --no-build-vignettes "${PACKAGE_SOURCE}"

"${CONTAINER}" exec \
  --bind "${BASE_DIR}:${BASE_DIR}" \
  --pwd "${BUNDLE_DIR}" \
  "${IMAGE}" \
  /bin/bash "${BUNDLE_DIR}/install_current_fastembedr_in_container.sh"

touch "${BUNDLE_DIR}/INSTALL_OK"
echo "Current fastEmbedR installation validated: ${BUNDLE_DIR}/Rlib"
