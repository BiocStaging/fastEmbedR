#!/usr/bin/env bash

# Run one isolated benchmark worker while recording process RAM and, on an
# NVIDIA host, whole-device GPU memory. The GPU measurement is intentionally
# labelled device-wide because process accounting is not reliable through all
# Apptainer PID namespaces. Publication jobs should request an exclusive GPU.

set -uo pipefail

if [[ "$#" -lt 4 ]]; then
  echo "usage: $0 TIME_FILE GPU_FILE TIMEOUT_SECONDS COMMAND [ARG ...]" >&2
  exit 64
fi

TIME_FILE="$1"
GPU_FILE="$2"
TIMEOUT_SECONDS="$3"
shift 3

mkdir -p "$(dirname "${TIME_FILE}")" "$(dirname "${GPU_FILE}")"
: > "${TIME_FILE}"
: > "${GPU_FILE}"

gpu_used_mb() {
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    return 1
  fi
  nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null \
    | awk 'NR == 1 { gsub(/ /, "", $1); print $1 }'
}

baseline_gpu="$(gpu_used_mb || true)"
peak_gpu="${baseline_gpu}"

run_command=()
if command -v timeout >/dev/null 2>&1; then
  run_command+=(timeout "${TIMEOUT_SECONDS}")
elif command -v gtimeout >/dev/null 2>&1; then
  run_command+=(gtimeout "${TIMEOUT_SECONDS}")
fi
run_command+=("$@")

native_time_file="${TIME_FILE}.native"
timed_command=()
if [[ -x /usr/bin/time ]]; then
  time_probe="${native_time_file}.probe"
  if [[ "$(uname -s 2>/dev/null || true)" == "Darwin" ]]; then
    if /usr/bin/time -l -o "${time_probe}" true >/dev/null 2>&1; then
      timed_command=(/usr/bin/time -l -o "${native_time_file}")
    fi
  else
    if /usr/bin/time -v -o "${time_probe}" true >/dev/null 2>&1; then
      timed_command=(/usr/bin/time -v -o "${native_time_file}")
    fi
  fi
  rm -f "${time_probe}"
fi
timed_command+=("${run_command[@]}")

rss_tree_proc_kb() {
  local root="$1"
  local -a queue=("${root}")
  local cursor=0
  local total=0
  local pid status rss children child

  while (( cursor < ${#queue[@]} )); do
    pid="${queue[cursor]}"
    cursor=$((cursor + 1))
    status="/proc/${pid}/status"
    if [[ -r "${status}" ]]; then
      rss="$(awk '/^VmRSS:/ { print $2; exit }' "${status}" 2>/dev/null || true)"
      if [[ "${rss}" =~ ^[0-9]+$ ]]; then
        total=$((total + rss))
      fi
    fi
    children="/proc/${pid}/task/${pid}/children"
    if [[ -r "${children}" ]]; then
      for child in $(<"${children}"); do
        [[ "${child}" =~ ^[0-9]+$ ]] && queue+=("${child}")
      done
    fi
  done
  echo "${total}"
}

rss_tree_ps_kb() {
  local root="$1"
  ps -axo pid=,ppid=,rss= 2>/dev/null | awk -v root="${root}" '
    { pid[NR] = $1; ppid[NR] = $2; rss[NR] = $3 }
    END {
      wanted[root] = 1
      changed = 1
      while (changed) {
        changed = 0
        for (i = 1; i <= NR; ++i) {
          if (wanted[ppid[i]] && !wanted[pid[i]]) {
            wanted[pid[i]] = 1
            changed = 1
          }
        }
      }
      total = 0
      for (i = 1; i <= NR; ++i) if (wanted[pid[i]]) total += rss[i]
      print total
    }'
}

rss_tree_kb() {
  local root="$1"
  if [[ -r "/proc/${root}/status" ]]; then
    rss_tree_proc_kb "${root}"
  else
    rss_tree_ps_kb "${root}"
  fi
}

start_epoch="$(date +%s)"
"${timed_command[@]}" &
worker_pid=$!
peak_rss_kb=0

while kill -0 "${worker_pid}" >/dev/null 2>&1; do
  current_rss="$(rss_tree_kb "${worker_pid}" || true)"
  if [[ "${current_rss}" =~ ^[0-9]+$ ]] && (( current_rss > peak_rss_kb )); then
    peak_rss_kb="${current_rss}"
  fi
  current="$(gpu_used_mb || true)"
  if [[ "${current}" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
    if [[ ! "${peak_gpu}" =~ ^[0-9]+([.][0-9]+)?$ ]] ||
       awk -v current="${current}" -v peak="${peak_gpu}" 'BEGIN { exit !(current > peak) }'; then
      peak_gpu="${current}"
    fi
  fi
  sleep 0.1
done

wait "${worker_pid}"
status=$?
end_epoch="$(date +%s)"
elapsed_sec=$(( end_epoch - start_epoch ))

if [[ -s "${native_time_file}" ]]; then
  native_rss_kb=""
  if grep -q "Maximum resident set size (kbytes)" "${native_time_file}" 2>/dev/null; then
    native_rss_kb="$(awk -F: '/Maximum resident set size \(kbytes\)/ {gsub(/ /, "", $2); print $2}' "${native_time_file}" | tail -n 1)"
  elif grep -q "maximum resident set size" "${native_time_file}" 2>/dev/null; then
    native_rss_bytes="$(awk '/maximum resident set size/ {print $1}' "${native_time_file}" | tail -n 1)"
    if [[ "${native_rss_bytes}" =~ ^[0-9]+$ ]]; then
      native_rss_kb=$(( native_rss_bytes / 1024 ))
    fi
  fi
  if [[ "${native_rss_kb}" =~ ^[0-9]+$ ]] && (( native_rss_kb > peak_rss_kb )); then
    peak_rss_kb="${native_rss_kb}"
  fi
fi

{
  echo "Maximum resident set size (kbytes): ${peak_rss_kb}"
  echo "Elapsed wall clock time (seconds): ${elapsed_sec}"
  echo "Exit status: ${status}"
} > "${TIME_FILE}"

delta_gpu="NA"
if [[ "${peak_gpu}" =~ ^[0-9]+([.][0-9]+)?$ ]] &&
   [[ "${baseline_gpu}" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
  delta_gpu="$(awk -v peak="${peak_gpu}" -v base="${baseline_gpu}" \
    'BEGIN { value = peak - base; if (value < 0) value = 0; printf "%.3f", value }')"
fi

{
  echo "gpu_memory_scope=device_wide"
  echo "gpu_baseline_mb=${baseline_gpu:-NA}"
  echo "gpu_peak_mb=${peak_gpu:-NA}"
  echo "gpu_peak_delta_mb=${delta_gpu}"
  echo "worker_exit_status=${status}"
} > "${GPU_FILE}"

exit "${status}"
