#!/usr/bin/env bash

# Submit each dataset/backend launcher as an independent Slurm job.
# Examples:
#   bash submit_reviewer_dataset_jobs.sh
#   PROFILE=cpu4 bash submit_reviewer_dataset_jobs.sh
#   PROFILE=cuda DRY_RUN=true bash submit_reviewer_dataset_jobs.sh

set -uo pipefail

BASE_DIR="${BASE_DIR:-/scratch/firenze/NN}"
JOBS_DIR="${JOBS_DIR:-${BASE_DIR}/jobs_by_dataset}"
PROFILE="${PROFILE:-all}"
DRY_RUN="${DRY_RUN:-false}"
SUBMIT_SLEEP="${SUBMIT_SLEEP:-0.1}"
LOG_DIR="${LOG_DIR:-${BASE_DIR}/benchmark_logs}"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
SUBMISSION_LOG="${SUBMISSION_LOG:-${LOG_DIR}/reviewer_job_submissions_${PROFILE}_${STAMP}.csv}"

case "${PROFILE}" in
  all)   scripts=("${JOBS_DIR}"/run_*.sh) ;;
  cpu1)  scripts=("${JOBS_DIR}"/run_*_cpu1.sh) ;;
  cpu4)  scripts=("${JOBS_DIR}"/run_*_cpu4.sh) ;;
  cuda)  scripts=("${JOBS_DIR}"/run_*_cuda.sh) ;;
  *)
    echo "PROFILE must be one of: all, cpu1, cpu4, cuda" >&2
    exit 2
    ;;
esac

if [[ ! -d "${JOBS_DIR}" || ! -f "${scripts[0]}" ]]; then
  echo "No ${PROFILE} launchers found under ${JOBS_DIR}" >&2
  exit 1
fi

dry_run_normalized="$(printf '%s' "${DRY_RUN}" | tr '[:upper:]' '[:lower:]')"
if [[ "${dry_run_normalized}" != "true" && "${dry_run_normalized}" != "1" &&
      "${dry_run_normalized}" != "yes" ]] && ! command -v sbatch >/dev/null 2>&1; then
  echo "sbatch was not found." >&2
  exit 1
fi

mkdir -p "${LOG_DIR}"
printf 'submitted_at,profile,script,job_id,status,message\n' > "${SUBMISSION_LOG}"

submitted=0
failed=0
for script in "${scripts[@]}"; do
  script_name="$(basename "${script}")"
  if [[ "${dry_run_normalized}" == "true" || "${dry_run_normalized}" == "1" ||
        "${dry_run_normalized}" == "yes" ]]; then
    echo "DRY RUN: sbatch ${script}"
    printf '%s,%s,%s,%s,%s,%s\n' \
      "$(date '+%Y-%m-%d %H:%M:%S')" "${PROFILE}" "${script_name}" "" \
      "dry_run" "not submitted" >> "${SUBMISSION_LOG}"
    continue
  fi

  echo "Submitting ${script_name}"
  output="$(sbatch --parsable "${script}" 2>&1)"
  status=$?
  if [[ ${status} -eq 0 ]]; then
    job_id="${output%%;*}"
    echo "  job ${job_id}"
    printf '%s,%s,%s,%s,%s,%s\n' \
      "$(date '+%Y-%m-%d %H:%M:%S')" "${PROFILE}" "${script_name}" \
      "${job_id}" "submitted" "" >> "${SUBMISSION_LOG}"
    submitted=$((submitted + 1))
  else
    message="$(printf '%s' "${output}" | tr ',\n' '; ')"
    echo "  FAILED: ${message}" >&2
    printf '%s,%s,%s,%s,%s,%s\n' \
      "$(date '+%Y-%m-%d %H:%M:%S')" "${PROFILE}" "${script_name}" "" \
      "failed" "${message}" >> "${SUBMISSION_LOG}"
    failed=$((failed + 1))
  fi
  sleep "${SUBMIT_SLEEP}"
done

echo "Submitted jobs: ${submitted}"
echo "Failed submissions: ${failed}"
echo "Submission log: ${SUBMISSION_LOG}"

if [[ ${failed} -gt 0 ]]; then
  exit 1
fi
