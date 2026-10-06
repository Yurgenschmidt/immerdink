#!/usr/bin/env bash
# Configure, build and run every correctness test in one command.
#   tools/test.sh                      build/ in the repo, all tests
#   tools/test.sh -R workspace         extra arguments go to ctest
# Environment:
#   IMMERDINK_ENV         conda env with the toolchain (default ~/miniforge3/envs/immerdink)
#   IMMERDINK_BUILD_DIR   build directory (default <repo>/build)
#   IMMERDINK_CUTLASS_DIR CUTLASS checkout (default: the third_party/cutlass submodule)
#   IMMERDINK_WERROR      ON (default) / OFF: warnings in immerdink's own code are errors
#   IMMERDINK_BUILD_JOBS  parallel compile jobs (default: min(nproc, MemAvailable / 3 GiB);
#                         one CUTLASS nvcc/cicc job can take ~2-3 GiB)
# Runs inside the shared immerdink memory budget (tools/memcap.sh).
# Correctness tests are pass/fail, so this is not gated on a quiet machine
# (tools/run_quiet.sh); it runs no timing benchmarks.
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
root="$(dirname "$here")"
grep -q "/immerdink.slice/" /proc/self/cgroup 2>/dev/null || exec "$here/memcap.sh" "$here/test.sh" "$@"
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
avail_gb=$(( $(awk '/^MemAvailable:/{print $2}' /proc/meminfo) / 1048576 ))
jobs="${IMMERDINK_BUILD_JOBS:-$(( avail_gb / 3 < $(nproc) ? avail_gb / 3 : $(nproc) ))}"
[ "$jobs" -ge 1 ] || jobs=1
cmake --build "$build" -j "$jobs"
# GPU tests share the machine with other projects: hold the advisory GPU lock for ctest
# (IMMERDINK_GPU_LOCK, same default as tools/run_quiet.sh; "none" disables).
gpu_lock="${IMMERDINK_GPU_LOCK:-/home/ef/repos/ehrenfest-ipdps/_wip/gpu.lock}"
[ "$gpu_lock" = none ] || [ -d "$(dirname "$gpu_lock")" ] || gpu_lock="${XDG_RUNTIME_DIR:-/tmp}/gpu.lock"
# CPU-only tests never touch the GPU, so they run first without the lock; only the GPU
# tests wait for it. A caller's own -R/-E filters apply to both passes.
cpu_only='^(host_c_api|host_python|host_perf|host_planner|host_accuracy|host_cuda_free|workspace_vs_params|params_check|constants_header_current)$'
if [ "$gpu_lock" = none ]; then ctest --test-dir "$build" --output-on-failure "$@"
else
  rc=0
  ctest --test-dir "$build" --output-on-failure -R "$cpu_only" "$@" || rc=$?
  flock "$gpu_lock" ctest --test-dir "$build" --output-on-failure -E "$cpu_only" "$@" || rc=$?
  exit $rc
fi
