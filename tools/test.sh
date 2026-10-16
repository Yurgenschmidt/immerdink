#!/usr/bin/env bash
# Configure, build and run every correctness test in one command.
#   tools/test.sh                      build/ in the repo, all tests
#   tools/test.sh -R workspace         extra arguments go to ctest (a test-selection option
#                                      runs the whole selection under the GPU lock, see below)
#   tools/test.sh --self-test          check the selection-option detection only (no build)
# Environment:
#   IMMERDINK_ENV         conda env with the toolchain (default ~/miniforge3/envs/immerdink)
#   IMMERDINK_BUILD_DIR   build directory (default <repo>/build)
#   IMMERDINK_CUTLASS_DIR CUTLASS checkout (default: the third_party/cutlass submodule)
#   IMMERDINK_COMPUTE_SANITIZER  compute-sanitizer for contract_initcheck_pad (default: next to nvcc)
#   IMMERDINK_WERROR      ON (default) / OFF: warnings in immerdink's own code are errors
#   IMMERDINK_BUILD_JOBS  parallel compile jobs (default: min(nproc, MemAvailable / 3 GiB);
#                         one CUTLASS nvcc/cicc job can take ~2-3 GiB)
# Runs inside the shared immerdink memory budget (tools/memcap.sh).
# Correctness tests are pass/fail, so this is not gated on a quiet machine
# (tools/run_quiet.sh); it runs no timing benchmarks.
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
root="$(dirname "$here")"

# 0 when "$@" holds a ctest test-selection option, exact or in its attached / --opt=value
# spelling. Such an option replaces or widens the CPU-only pass's own -R (ctest keeps the
# last -R; -U makes filters a union; -L/-I/--tests-from-file select independently), which
# would let GPU tests run WITHOUT the GPU lock; the caller then gets one locked pass.
has_selection() {
  local a
  for a in "$@"; do
    case "$a" in
      -R|-E|-L|-LE|-I|-U|-FA|-FS|-FC|--tests-regex|--exclude-regex|--label-regex| \
      --label-exclude|--tests-information|--union|--fixture-exclude-any| \
      --fixture-exclude-setup|--fixture-exclude-cleanup|--tests-from-file| \
      --exclude-from-file|--rerun-failed|-R?*|-E?*|-L?*|-I?*|--tests-regex=*| \
      --exclude-regex=*|--label-regex=*|--label-exclude=*|--tests-information=*| \
      --tests-from-file=*|--exclude-from-file=*)
        return 0 ;;
    esac
  done
  return 1
}

if [ "${1:-}" = --self-test ]; then
  fail=0
  for args in "-R x" "-Rx" "-E x" "-Ex" "-L gpu" "-LE gpu" "-I 1,3" "-I1,3" "-U" "-FA x" \
              "--tests-regex x" "--tests-regex=x" "--exclude-regex=x" "--label-regex=x" \
              "--tests-from-file f" "--rerun-failed" "-j 2 -R x" "--output-on-failure -E y"; do
    # shellcheck disable=SC2086
    has_selection $args || { echo "self-test: '$args' not detected as a selection" >&2; fail=1; }
  done
  for args in "" "-j 2" "-V" "--output-on-failure" "--timeout 60" "-N" "--repeat until-pass:2"; do
    # shellcheck disable=SC2086
    ! has_selection $args || { echo "self-test: '$args' wrongly detected as a selection" >&2; fail=1; }
  done
  [ "$fail" = 0 ] && echo "tools/test.sh self-test: OK"
  exit "$fail"
fi

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
[ -n "${IMMERDINK_COMPUTE_SANITIZER:-}" ] && cfg+=(-DIMMERDINK_COMPUTE_SANITIZER="$IMMERDINK_COMPUTE_SANITIZER")

cmake "${cfg[@]}"
avail_gb=$(( $(awk '/^MemAvailable:/{print $2}' /proc/meminfo) / 1048576 ))
jobs="${IMMERDINK_BUILD_JOBS:-$(( avail_gb / 3 < $(nproc) ? avail_gb / 3 : $(nproc) ))}"
[ "$jobs" -ge 1 ] || jobs=1
cmake --build "$build" -j "$jobs"

# contract_initcheck_pad (DESIGN v1.13 §4b.8) is registered only when CMake finds a
# compute-sanitizer; without it the one-sided-padding mutation is checked bit-exact only.
initcheck_warning=""
if ! ctest --test-dir "$build" -N -R '^contract_initcheck_pad$' 2>/dev/null \
     | grep -q 'contract_initcheck_pad'; then
  initcheck_warning="
******************************************************************************************
WARNING: contract_initcheck_pad is NOT registered (CMake found no compute-sanitizer), so the
one-sided K-padding mutation is NOT checked by initcheck in this run. To enable it:
  IMMERDINK_COMPUTE_SANITIZER=\$(ls -d ~/miniforge3/pkgs/cuda-sanitizer-api-*/bin/compute-sanitizer | head -1) \\
    tools/test.sh
******************************************************************************************"
  echo "$initcheck_warning" >&2
fi

# GPU tests share the machine with other projects: hold the advisory GPU lock for ctest
# (IMMERDINK_GPU_LOCK, same default as tools/run_quiet.sh; "none" disables).
gpu_lock="${IMMERDINK_GPU_LOCK:-/home/ef/repos/ehrenfest-ipdps/_wip/gpu.lock}"
[ "$gpu_lock" = none ] || [ -d "$(dirname "$gpu_lock")" ] || gpu_lock="${XDG_RUNTIME_DIR:-/tmp}/gpu.lock"
# Without a caller's selection option: CPU-only tests never touch the GPU, so they run first
# without the lock; only the GPU tests wait for it. With one (has_selection), the caller's
# selection runs in ONE pass under the lock, so no GPU test ever runs unlocked.
cpu_only='^(calib_strings_shipped|calib_strings_too_long|tools_test_sh_selftest|host_c_api|host_k_align|host_l2_knobs|bench_membw_selftest|host_python|host_devices|device_kernel_sites|host_perf|host_planner|host_accuracy|host_cuda_free|host_footprint_monotone|host_precision|c2_engine_selftest|sweep_disjoint_selftest|md2_time_selftest|campaign_[a-z]+_selftest|workspace_vs_params|params_check|constants_header_current)$'
rc=0
if [ "$gpu_lock" = none ]; then ctest --test-dir "$build" --output-on-failure "$@" || rc=$?
elif has_selection "$@"; then
  flock "$gpu_lock" ctest --test-dir "$build" --output-on-failure "$@" || rc=$?
else
  ctest --test-dir "$build" --output-on-failure -R "$cpu_only" "$@" || rc=$?
  flock "$gpu_lock" ctest --test-dir "$build" --output-on-failure -E "$cpu_only" "$@" || rc=$?
fi
# repeated at the end, where it is seen
[ -z "$initcheck_warning" ] || echo "$initcheck_warning" >&2
exit $rc
