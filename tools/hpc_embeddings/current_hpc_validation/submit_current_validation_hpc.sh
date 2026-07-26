#!/usr/bin/env bash

set -euo pipefail

BASE_DIR=/scratch/firenze/NN
BUNDLE_DIR="${BASE_DIR}/current_fastembedr_validation"
RUN_ID="current_$(date +%Y%m%d_%H%M%S)"

required=(
  "${BUNDLE_DIR}/source/.git/HEAD"
  "${BUNDLE_DIR}/Rprofile.current"
  "${BUNDLE_DIR}/install_current_fastembedr_in_container.sh"
  "${BUNDLE_DIR}/prepare_current_fastembedr_hpc.sh"
  "${BUNDLE_DIR}/run_current_cpu_validation_hpc.sh"
  "${BUNDLE_DIR}/run_current_cuda_validation_hpc.sh"
  "${BUNDLE_DIR}/benchmark_reviewer_validation.R"
  "${BUNDLE_DIR}/benchmark_local_graph_clustering.R"
  "${BUNDLE_DIR}/publication_metrics.R"
  "${BUNDLE_DIR}/benchmark_worker_monitor.sh"
  "${BUNDLE_DIR}/reference_opentsne_affinity.py"
  "${BASE_DIR}/singularity/fastembedr_cuda.sif"
)
for path in "${required[@]}"; do
  [[ -f "${path}" ]] || { echo "Missing required file: ${path}" >&2; exit 1; }
done

mkdir -p "${BASE_DIR}/benchmark_logs" \
  "${BASE_DIR}/fastEmbedR-results/${RUN_ID}"

prepare_job="$(sbatch --parsable "${BUNDLE_DIR}/prepare_current_fastembedr_hpc.sh")"
cpu_job="$(FASTEMBEDR_RUN_ID="${RUN_ID}" sbatch --parsable \
  --dependency="afterok:${prepare_job}" \
  --export="ALL,FASTEMBEDR_RUN_ID=${RUN_ID}" \
  "${BUNDLE_DIR}/run_current_cpu_validation_hpc.sh")"
cuda_job="$(FASTEMBEDR_RUN_ID="${RUN_ID}" sbatch --parsable \
  --dependency="afterok:${prepare_job}" \
  --export="ALL,FASTEMBEDR_RUN_ID=${RUN_ID}" \
  "${BUNDLE_DIR}/run_current_cuda_validation_hpc.sh")"

cat <<EOF
Submitted current fastEmbedR validation.
Run ID:      ${RUN_ID}
Prepare job: ${prepare_job}
CPU job:     ${cpu_job}
CUDA job:    ${cuda_job}
Results:     ${BASE_DIR}/fastEmbedR-results/${RUN_ID}

Monitor:
  squeue -j ${prepare_job},${cpu_job},${cuda_job}
  tail -f ${BASE_DIR}/benchmark_logs/feR_current_build_${prepare_job}.out
EOF
