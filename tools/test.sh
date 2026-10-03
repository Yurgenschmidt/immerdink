#!/usr/bin/env bash
# Configure, build and run every correctness test in one command.
#   tools/test.sh                      build/ in the repo, all tests
#   tools/test.sh -R workspace         extra arguments go to ctest
# Environment:
#   IMMERDINK_ENV         conda env with the toolchain (default ~/miniforge3/envs/immerdink)
#   IMMERDINK_BUILD_DIR   build directory (default <repo>/build)
#   IMMERDINK_CUTLASS_DIR CUTLASS checkout (default: the third_party/cutlass submodule)
#   IMMERDINK_WERROR      ON (default) / OFF: warnings in immerdink's own code are errors
# Correctness tests are pass/fail, so this is not gated on a quiet machine
# (tools/run_quiet.sh); it runs no timing benchmarks.
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
root="$(dirname "$here")"
env_dir="${IMMERDINK_ENV:-$HOME/miniforge3/envs/immerdink}"
build="${IMMERDINK_BUILD_DIR:-$root/build}"

[ -x "$env_dir/bin/nvcc" ] || { echo "no nvcc in $env_dir (set IMMERDINK_ENV)" >&2; exit 2; }
export PATH="$env_dir/bin:$PATH"

cfg=(-S "$root" -B "$build" -G Ninja
     -DCMAKE_C_COMPILER="$env_dir/bin/gcc" -DCMAKE_CXX_COMPILER="$env_dir/bin/g++"
     -DCMAKE_CUDA_COMPILER="$env_dir/bin/nvcc" -DCMAKE_CUDA_HOST_COMPILER="$env_dir/bin/g++"
     -DPython3_EXECUTABLE="$env_dir/bin/python3"
     -DIMMERDINK_WERROR="${IMMERDINK_WERROR:-ON}")
[ -n "${IMMERDINK_CUTLASS_DIR:-}" ] && cfg+=(-DIMMERDINK_CUTLASS_DIR="$IMMERDINK_CUTLASS_DIR")

cmake "${cfg[@]}"
cmake --build "$build"
ctest --test-dir "$build" --output-on-failure "$@"
