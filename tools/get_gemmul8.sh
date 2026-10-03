#!/usr/bin/env bash
# Fetch and build GEMMul8, a benchmark-only baseline (never linked into libimmerdink).
#   tools/get_gemmul8.sh            clone (if needed) and build
#   tools/get_gemmul8.sh --print    print the built static library path and exit
# Pinned at 6a9eff8 (= v3.5.2 plus one README commit). Built with the immerdink conda env:
# INT8 backend, GEMM only, sm_89. Result: $GEMMUL8_DIR/lib/libgemmul8.{a,so}.
# Environment:
#   IMMERDINK_ENV  conda env (default ~/miniforge3/envs/immerdink)
#   GEMMUL8_DIR    checkout directory (default <repo>/bench/_deps/GEMMul8, git-ignored)
#   GPU_ARCH       default 89
#   JOBS           parallel compile jobs (default 8; the build runs under nice -n 19)
# The compile is CPU-heavy, so it takes the machine-wide benchmark lock (the one
# tools/run_bench.sh uses): it never overlaps a gated timing run.
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
root="$(dirname "$here")"
env_dir="${IMMERDINK_ENV:-$HOME/miniforge3/envs/immerdink}"
dir="${GEMMUL8_DIR:-$root/bench/_deps/GEMMul8}"
rev=6a9eff8bbd65fd98899d4eece5f305d33adc7eb4
url=https://github.com/RIKEN-RCCS/GEMMul8.git
arch="${GPU_ARCH:-89}"

if [ "${1:-}" = "--print" ]; then
  [ -f "$dir/lib/libgemmul8.a" ] || { echo "not built: $dir/lib/libgemmul8.a" >&2; exit 1; }
  echo "$dir/lib/libgemmul8.a"
  exit 0
fi
[ -x "$env_dir/bin/nvcc" ] || { echo "no nvcc in $env_dir (set IMMERDINK_ENV)" >&2; exit 2; }

if [ ! -d "$dir/.git" ]; then
  mkdir -p "$dir"
  git -C "$dir" init -q
  git -C "$dir" remote add origin "$url"
fi
if [ "$(git -C "$dir" rev-parse -q --verify HEAD 2>/dev/null || true)" != "$rev" ]; then
  git -C "$dir" fetch -q --depth 1 origin "$rev"
  git -C "$dir" checkout -q --detach "$rev"
fi
echo "GEMMul8 $(git -C "$dir" log --oneline -1) in $dir"

export PATH="$env_dir/bin:$PATH"
# libnvidia-ml and libcuda come from the driver, cuBLAS/cudart from the env.
libs="-L$env_dir/lib -L/usr/lib/x86_64-linux-gnu -lcublas -lcudart -lcublasLt -lcuda -lnvidia-ml -ldl"
exec 9>"${TLC_LOCK:-${XDG_RUNTIME_DIR:-/tmp}/immerdink-tlc.lock}"
flock 9
nice -n 19 make -C "$dir" -j"${JOBS:-8}" BACKEND=cuda CUDA_PATH="$env_dir" GPU_ARCH="$arch" \
  OPS=gemm OZ2_BACKENDS=INT8 PROFILE=off LIBS="$libs" CXX="$env_dir/bin/g++"
flock -u 9
ls -l "$dir/lib/"
echo "built: $dir/lib/libgemmul8.a (headers: $dir/include)"
