#!/usr/bin/env bash

# Submit the independent landmark-validation jobs.
# Examples:
#   bash submit_landmark_dataset_jobs.sh
#   PROFILE=cpu4 bash submit_landmark_dataset_jobs.sh
#   PROFILE=cuda DRY_RUN=true bash submit_landmark_dataset_jobs.sh

set -uo pipefail

BASE_DIR="${BASE_DIR:-/scratch/firenze/NN}"
JOBS_DIR="${JOBS_DIR:-${BASE_DIR}/landmark_jobs_by_dataset}"
PROFILE="${PROFILE:-all}"
DRY_RUN="${DRY_RUN:-false}"
SUBMIT_SLEEP="${SUBMIT_SLEEP:-0.1}"
LOG_DIR="${LOG_DIR:-${BASE_DIR}/benchmark_logs}"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
SUBMISSION_LOG="${SUBMISSION_LOG:-${LOG_DIR}/landmark_job_submissions_${PROFILE}_${STAMP}.csv}"

case "${PROFILE}" in
  all)  scripts=("${JOBS_DIR}"/run_landmark_*.sh) ;;
  cpu1) scripts=("${JOBS_DIR}"/run_landmark_*_cpu1.sh) ;;
  cpu4) scripts=("${JOBS_DIR}"/run_landmark_*_cpu4.sh) ;;
  cuda) scripts=("${JOBS_DIR}"/run_landmark_*_cuda.sh) ;;
  *) echo "PROFILE must be one of: all, cpu1, cpu4, cuda" >&2; exit 2 ;;
esac

if [[ ! -d "${JOBS_DIR}" || ! -f "${scripts[0]}" ]]; then
  echo "No ${PROFILE} landmark launchers found under ${JOBS_DIR}" >&2
  exit 1
fi

dry_run="$(printf '%s' "${DRY_RUN}" | tr '[:upper:]' '[:lower:]')"
if [[ ! "${dry_run}" =~ ^(true|1|yes)$ ]] && ! command -v sbatch >/dev/null 2>&1; then
  echo "sbatch was not found." >&2
  exit 1
fi

mkdir -p "${LOG_DIR}"
printf 'submitted_at,profile,script,job_id,status,message\n' > "${SUBMISSION_LOG}"
submitted=0
failed=0
for script in "${scripts[@]}"; do
  name="$(basename "${script}")"
  if [[ "${dry_run}" =~ ^(true|1|yes)$ ]]; then
    echo "DRY RUN: sbatch ${script}"
    printf '%s,%s,%s,%s,%s,%s\n' \
      "$(date '+%Y-%m-%d %H:%M:%S')" "${PROFILE}" "${name}" "" \
      "dry_run" "not submitted" >> "${SUBMISSION_LOG}"
    continue
  fi
  echo "Submitting ${name}"
  output="$(sbatch --parsable "${script}" 2>&1)"
  status=$?
  if [[ ${status} -eq 0 ]]; then
    job_id="${output%%;*}"
    echo "  job ${job_id}"
    printf '%s,%s,%s,%s,%s,%s\n' \
      "$(date '+%Y-%m-%d %H:%M:%S')" "${PROFILE}" "${name}" \
      "${job_id}" "submitted" "" >> "${SUBMISSION_LOG}"
    submitted=$((submitted + 1))
  else
    message="$(printf '%s' "${output}" | tr ',\n' '; ')"
    echo "  FAILED: ${message}" >&2
    printf '%s,%s,%s,%s,%s,%s\n' \
      "$(date '+%Y-%m-%d %H:%M:%S')" "${PROFILE}" "${name}" "" \
      "failed" "${message}" >> "${SUBMISSION_LOG}"
    failed=$((failed + 1))
  fi
  sleep "${SUBMIT_SLEEP}"
done

echo "Submitted jobs: ${submitted}"
echo "Failed submissions: ${failed}"
echo "Submission log: ${SUBMISSION_LOG}"
[[ ${failed} -eq 0 ]]
