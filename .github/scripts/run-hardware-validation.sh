#!/usr/bin/env bash
set -euo pipefail

backend="${1:?backend is required}"
out_dir="${2:?output directory is required}"

case "$backend" in
  cpu|metal|cuda) ;;
  *) echo "backend must be cpu, metal, or cuda" >&2; exit 2 ;;
esac

root="$(git rev-parse --show-toplevel)"
out_dir="$(mkdir -p "$out_dir" && cd "$out_dir" && pwd)"
lib_dir="$out_dir/R-library"
mkdir -p "$lib_dir"

sha256_file() {
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$1" | awk '{print $1}'
  else
    shasum -a 256 "$1" | awk '{print $1}'
  fi
}

cd "$root"

git rev-parse HEAD > "$out_dir/git-commit.txt"
git status --porcelain=v1 > "$out_dir/git-status.txt"
if [[ -s "$out_dir/git-status.txt" ]]; then
  echo "Hardware validation requires a clean checkout." >&2
  cat "$out_dir/git-status.txt" >&2
  exit 3
fi

{
  echo "backend=$backend"
  echo "timestamp_utc=$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
  echo "runner_name=${RUNNER_NAME:-manual}"
  echo "runner_os=${RUNNER_OS:-$(uname -s)}"
  echo "runner_arch=${RUNNER_ARCH:-$(uname -m)}"
  uname -a
  R --version | head -n 1
  if [[ "$backend" == "metal" ]]; then
    xcrun --find metal
    xcrun metal --version
    system_profiler SPHardwareDataType SPDisplaysDataType
  elif [[ "$backend" == "cuda" ]]; then
    nvidia-smi
    "${NVCC:-${CUDA_HOME:-/usr/local/cuda}/bin/nvcc}" --version
  else
    command -v lscpu >/dev/null 2>&1 && lscpu || true
  fi
} > "$out_dir/hardware.txt" 2>&1

git archive --format=tar HEAD > "$out_dir/source.tar"
source_sha256="$(sha256_file "$out_dir/source.tar")"
rm "$out_dir/source.tar"

build_dir="$(mktemp -d "${TMPDIR:-/tmp}/fastembedr-hardware.XXXXXX")"
trap 'rm -rf "$build_dir" "$lib_dir"' EXIT

if [[ -n "${FASTEMBEDR_VALIDATION_LOCALE:-}" ]]; then
  build_locale="$FASTEMBEDR_VALIDATION_LOCALE"
elif [[ "$(uname -s)" == "Darwin" ]]; then
  build_locale="en_US.UTF-8"
else
  build_locale="C.UTF-8"
fi

(
  cd "$build_dir"
  LC_ALL="$build_locale" LANG="$build_locale" \
    R CMD build "$root" --no-build-vignettes --no-manual
) > "$out_dir/build.log" 2>&1
package_tar="$(find "$build_dir" -maxdepth 1 -name 'fastEmbedR_*.tar.gz' -print -quit)"
test -n "$package_tar"
package_sha256="$(sha256_file "$package_tar")"

if [[ "$backend" == "cpu" ]]; then
  export FASTEMBEDR_USE_CUDA=0
elif [[ "$backend" == "metal" ]]; then
  export FASTEMBEDR_USE_CUDA=0
  export SDKROOT="${SDKROOT:-$(xcrun --sdk macosx --show-sdk-path)}"
else
  export FASTEMBEDR_USE_CUDA=1
  export FASTEMBEDR_USE_CUVS="${FASTEMBEDR_USE_CUVS:-1}"
  export FASTEMBEDR_USE_FAISS_GPU="${FASTEMBEDR_USE_FAISS_GPU:-1}"
  export FASTEMBEDR_USE_RAFT="${FASTEMBEDR_USE_RAFT:-1}"
fi

R CMD INSTALL --preclean --library="$lib_dir" "$package_tar" \
  > "$out_dir/install.log" 2>&1

export R_LIBS="$lib_dir${R_LIBS:+:$R_LIBS}"
export FASTEMBEDR_VALIDATION_BACKEND="$backend"

dll_path="$(Rscript -e 'library(fastEmbedR); cat(getLoadedDLLs()[["fastEmbedR"]][["path"]])' \
  2> "$out_dir/dll-load.log")"
dll_sha256="$(sha256_file "$dll_path")"

cat > "$out_dir/identity.csv" <<EOF
"backend","git_commit","source_sha256","package_tar_sha256","installed_dll_sha256","installed_dll"
"$backend","$(cat "$out_dir/git-commit.txt")","$source_sha256","$package_sha256","$dll_sha256","$dll_path"
EOF

Rscript .github/scripts/validate-hardware.R "$backend" "$out_dir" \
  > "$out_dir/runtime-validation.log" 2>&1

Rscript -e 'testthat::test_dir("tests/testthat", reporter = "summary", package = "fastEmbedR", load_package = "installed", stop_on_failure = TRUE)' \
  > "$out_dir/testthat.log" 2>&1

git status --porcelain=v1 > "$out_dir/git-status-after.txt"
if [[ -s "$out_dir/git-status-after.txt" ]]; then
  echo "Validation modified the checkout." >&2
  cat "$out_dir/git-status-after.txt" >&2
  exit 4
fi

find "$out_dir" -type f ! -name SHA256SUMS ! -path '*/R-library/*' -print | LC_ALL=C sort | \
  while IFS= read -r file; do
    printf '%s  %s\n' "$(sha256_file "$file")" "${file#$out_dir/}"
  done > "$out_dir/SHA256SUMS"

echo "Validated fastEmbedR backend=$backend source_sha256=$source_sha256"
