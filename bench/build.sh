#!/usr/bin/env bash
# Configure and build the benchmark harness (and libimmerdink from this checkout) into
# build/bench, with the immerdink conda env.
#   bench/build.sh
# Environment:
#   IMMERDINK_ENV          conda env (default ~/miniforge3/envs/immerdink)
#   IMMERDINK_BENCH_BUILD  build directory (default <repo>/build/bench)
#   IMMERDINK_CUTLASS_DIR  CUTLASS v4.8.0 checkout (default: the third_party/cutlass submodule)
#   CMAKE_CUDA_ARCHITECTURES  GPU architecture(s) to build for (default 89 = sm_89; e.g. 120 for Blackwell
#                          workstation parts); changing it needs a fresh IMMERDINK_BENCH_BUILD directory
#   IMMERDINK_LIBRARY      link this prebuilt libimmerdink.so instead of building one
# The build is CPU-heavy: it runs under nice and takes the machine-wide benchmark lock,
# so it never overlaps a gated timing run; it runs inside tools/memcap.sh with at most 4 jobs.
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
[ -n "${CMAKE_CUDA_ARCHITECTURES:-}" ] && cfg+=(-DCMAKE_CUDA_ARCHITECTURES="$CMAKE_CUDA_ARCHITECTURES")
[ -n "${IMMERDINK_LIBRARY:-}" ] && cfg+=(-DIMMERDINK_LIBRARY="$IMMERDINK_LIBRARY")

exec 9>"${TLC_LOCK:-${XDG_RUNTIME_DIR:-/tmp}/immerdink-tlc.lock}"
flock 9
cmake "${cfg[@]}"
nice -n 19 "$root/tools/memcap.sh" cmake --build "$build" --target imd_bench -j "$(( ${JOBS:-4} < 4 ? ${JOBS:-4} : 4 ))"
echo "built: $build/imd_bench"
