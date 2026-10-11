#!/usr/bin/env bash
# Build the pinned GEMMul8 (the optional gemmul8 baseline of imd_bench) for one GPU architecture.
#   bench/build_gemmul8.sh <gemmul8 source dir> <arch: 89|90|120|...> <install dir>
# e.g.  bench/build_gemmul8.sh $GROUP/gemmul8-src 90 $TMPDIR/gemmul8-sm90
# The last line printed is the value to give bench/build.sh:  GEMMUL8_DIR=<install dir>
#
# What it builds: the GEMM family only, INT8 backend, profiling off, the STATIC library
# (make BACKEND=cuda OPS=gemm OZ2_BACKENDS=INT8 PROFILE=off lib/libgemmul8.a: the recipe the evaluation
# workspace used for the 4090 comparison campaigns). No GPU and no NVML is needed to build, so this can
# run on a CPU node; the architecture is the target GPU's compute capability without the dot
# (H100 / H200: 90, RTX PRO 6000 Blackwell: 120, RTX 4090: 89).
#
# The source must be GEMMul8 at the commit in bench/GEMMUL8_PIN ("name commit api"): a git checkout of
# that commit, or a source snapshot with a one-line file COMMIT holding it (what a login node can
# stage when the compute nodes have no network). Any other commit is refused unless GEMMUL8_ALLOW_UNPINNED=1
# (the records then still say the pinned name; do not use that for results).
# The source tree is never modified: the build runs in a copy under ${GEMMUL8_BUILD_TMP:-$TMPDIR:-/tmp}
# (node-local scratch on an HPC node), so builds for several architectures can share one source directory.
# Result in <install dir>: lib/libgemmul8.a, include/ (the headers), BUILD_INFO (name, commit, api, arch,
# nvcc, build seconds; read by bench/CMakeLists.txt).
#
# Toolchain: as bench/build.sh: the conda env IMMERDINK_ENV (default ~/miniforge3/envs/immerdink) if it has
# nvcc, else nvcc on PATH / CUDA_HOME. nvcc >= 12.8 is needed for 120. The build is CPU- and RAM-heavy: it
# takes the machine-wide benchmark lock, runs under nice inside tools/memcap.sh with at most 4 jobs.
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
root="$(dirname "$here")"
[ $# -eq 3 ] || { echo "usage: $0 <gemmul8 source dir> <arch: 89|90|120|...> <install dir>" >&2; exit 2; }
src="$(cd "$1" 2>/dev/null && pwd)" || { echo "no GEMMul8 source directory: $1" >&2; exit 2; }
arch="$2"
[[ "$arch" =~ ^[0-9]+$ ]] || { echo "architecture must be a number like 90 or 120 (got '$arch')" >&2; exit 2; }
inst="$(realpath -m -- "$3")"

read -r pin_name pin_commit pin_api < <(grep -v '^[[:space:]]*#' "$here/GEMMUL8_PIN" | grep -v '^[[:space:]]*$' | head -1)
[ -n "${pin_commit:-}" ] || { echo "$here/GEMMUL8_PIN has no 'name commit api' line" >&2; exit 2; }

# --- the source is the pinned commit
have=""
if command -v git >/dev/null && [ -e "$src/.git" ]; then have="$(git -C "$src" rev-parse HEAD 2>/dev/null || true)"; fi
[ -z "$have" ] && [ -f "$src/COMMIT" ] && have="$(tr -d '[:space:]' < "$src/COMMIT")"
if [ "$have" != "$pin_commit" ]; then
  if [ "${GEMMUL8_ALLOW_UNPINNED:-0}" = 1 ]; then
    echo "WARNING: $src is at '${have:-unknown}', not the pinned $pin_commit (GEMMUL8_ALLOW_UNPINNED=1)" >&2
  else
    echo "$src is at '${have:-unknown}', not the pinned $pin_commit ($pin_name, bench/GEMMUL8_PIN)." >&2
    echo "  git clone https://github.com/RIKEN-RCCS/GEMMul8.git <dir> && git -C <dir> checkout $pin_commit" >&2
    exit 2
  fi
fi
[ -f "$src/Makefile" ] && [ -d "$src/make" ] && [ -f "$src/include/gemmul8.hpp" ] \
  || { echo "$src does not look like a GEMMul8 v3.x source tree (Makefile, make/, include/gemmul8.hpp)" >&2; exit 2; }

# --- toolchain (same rule as bench/build.sh)
env_dir="${IMMERDINK_ENV:-$HOME/miniforge3/envs/immerdink}"
cuda_path=""
if [ -x "$env_dir/bin/nvcc" ]; then
  export PATH="$env_dir/bin:$PATH" CONDA_PREFIX="$env_dir"
  # a loaded gcc module exports GCC_EXEC_PREFIX / COMPILER_PATH; with them set the env's g++ looks for cc1plus in the
  # module's tree and fails ("cannot execute cc1plus"): the env toolchain must not see them
  unset GCC_EXEC_PREFIX COMPILER_PATH
  cuda_path="$env_dir"
  [ -x "$env_dir/bin/g++" ] && export NVCC_PREPEND_FLAGS="-ccbin $env_dir/bin/g++ ${NVCC_PREPEND_FLAGS:-}"
else
  nvcc="$(command -v nvcc || true)"
  [ -z "$nvcc" ] && [ -x "${CUDA_HOME:-/usr/local/cuda}/bin/nvcc" ] && nvcc="${CUDA_HOME:-/usr/local/cuda}/bin/nvcc"
  [ -n "$nvcc" ] || { echo "no nvcc: put one on PATH (module load cuda), set CUDA_HOME, or set IMMERDINK_ENV" >&2; exit 2; }
  cuda_path="$(cd "$(dirname "$nvcc")/.." && pwd)"
  echo "build_gemmul8: using $nvcc (no conda env at $env_dir)"
fi
command -v make >/dev/null || { echo "no make on PATH" >&2; exit 2; }
jobs=$(( ${JOBS:-4} < 4 ? ${JOBS:-4} : 4 ))

# --- build in a copy on node-local scratch
tmp_root="${GEMMUL8_BUILD_TMP:-${TMPDIR:-/tmp}}"
work="$(mktemp -d "$tmp_root/gemmul8-build.XXXXXX")"
trap 'rm -rf "$work"' EXIT
tar -C "$src" --exclude=.git --exclude=./build --exclude=./lib -cf - . | tar -C "$work" -xf -
mkdir -p "$inst/lib" "$inst/include"
log="$inst/build.log"

exec 9>"${TLC_LOCK:-${XDG_RUNTIME_DIR:-/tmp}/immerdink-tlc.lock}"
flock 9
t0=$(date +%s)
echo "build_gemmul8: $pin_name ($pin_commit) for sm_$arch, INT8 GEMM family, static library (log: $log)"
nice -n 19 "$root/tools/memcap.sh" make -C "$work" -j"$jobs" BACKEND=cuda CUDA_PATH="$cuda_path" GPU_ARCH="$arch" \
  OPS=gemm OZ2_BACKENDS=INT8 PROFILE=off lib/libgemmul8.a > "$log" 2>&1 \
  || { tail -30 "$log" >&2; echo "GEMMul8 build failed (see $log)" >&2; exit 1; }
flock -u 9
secs=$(( $(date +%s) - t0 ))
cp "$work/lib/libgemmul8.a" "$inst/lib/"
cp -r "$work/include/." "$inst/include/"
{
  echo "name=$pin_name"; echo "commit=$have"; echo "api=$pin_api"; echo "arch=$arch"; echo "build_s=$secs"; echo "jobs=$jobs"
  echo "nvcc=$(nvcc --version 2>/dev/null | tail -1 || echo unknown)"
  echo "note=make OPS=gemm OZ2_BACKENDS=INT8 PROFILE=off lib/libgemmul8.a"
} > "$inst/BUILD_INFO"
echo "built $pin_name for sm_$arch in ${secs}s: $(stat -c %s "$inst/lib/libgemmul8.a") bytes"
echo "GEMMUL8_DIR=$inst"
