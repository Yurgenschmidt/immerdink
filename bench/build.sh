#!/usr/bin/env bash
# Configure and build the benchmark harness (and libimmerdink from this checkout) into
# build/bench, with the immerdink conda env.
#   bench/build.sh
# Environment:
#   IMMERDINK_ENV          conda env (default ~/miniforge3/envs/immerdink)
#   IMMERDINK_BENCH_BUILD  build directory (default <repo>/build/bench)
#   IMMERDINK_CUTLASS_DIR  CUTLASS v4.8.0 checkout (default: the third_party/cutlass submodule)
#   IMMERDINK_LIBRARY      link this prebuilt libimmerdink.so instead of building one
#   GEMMUL8_DIR            GEMMul8 built by tools/get_gemmul8.sh (default bench/_deps/GEMMul8)
# The build is CPU-heavy: it runs under nice and takes the machine-wide benchmark lock,
# so it never overlaps a gated timing run.
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
root="$(dirname "$here")"
env_dir="${IMMERDINK_ENV:-$HOME/miniforge3/envs/immerdink}"
build="${IMMERDINK_BENCH_BUILD:-$root/build/bench}"
[ -x "$env_dir/bin/nvcc" ] || { echo "no nvcc in $env_dir (set IMMERDINK_ENV)" >&2; exit 2; }
export PATH="$env_dir/bin:$PATH"
export CONDA_PREFIX="$env_dir"

cfg=(-S "$here" -B "$build" -G Ninja
     -DCMAKE_C_COMPILER="$env_dir/bin/gcc" -DCMAKE_CXX_COMPILER="$env_dir/bin/g++"
     -DCMAKE_CUDA_COMPILER="$env_dir/bin/nvcc" -DCMAKE_CUDA_HOST_COMPILER="$env_dir/bin/g++"
     -DPython3_EXECUTABLE="$env_dir/bin/python3")
[ -n "${IMMERDINK_CUTLASS_DIR:-}" ] && cfg+=(-DIMMERDINK_CUTLASS_DIR="$IMMERDINK_CUTLASS_DIR")
[ -n "${IMMERDINK_LIBRARY:-}" ] && cfg+=(-DIMMERDINK_LIBRARY="$IMMERDINK_LIBRARY")
[ -n "${GEMMUL8_DIR:-}" ] && cfg+=(-DGEMMUL8_DIR="$GEMMUL8_DIR")

exec 9>"${TLC_LOCK:-${XDG_RUNTIME_DIR:-/tmp}/immerdink-tlc.lock}"
flock 9
cmake "${cfg[@]}"
nice -n 19 cmake --build "$build" --target imd_bench -j "${JOBS:-8}"
echo "built: $build/imd_bench"
