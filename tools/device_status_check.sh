#!/usr/bin/env bash
# Build libimmerdink for one architecture list and print the device status it gives on THIS GPU
# (issue 18, docs/NOTES-18.md "Three 4090 builds"). Three builds answer "what happens on a card the
# binary was or was not made for":
#   tools/device_status_check.sh native    89-real      -> status native
#   tools/device_status_check.sh ptxonly   80-virtual   -> recompiled (driver as new as the toolkit), or
#                                                         unsupported with CUDA error 222 (older driver)
#   tools/device_status_check.sh foreign   90-real      -> unsupported, CUDA error 209
#   tools/device_status_check.sh all                    all three
# Build directories: $IMMERDINK_STATUS_BUILDS (default <repo>/build-status/<name>). Slow (a CUTLASS build
# each, 3-6 min with 4 jobs); not part of ctest. Uses the toolchain of tools/test.sh (IMMERDINK_ENV).
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
root="$(dirname "$here")"
[ -n "${IMMERDINK_MEMCAP:-}" ] || grep -q "/immerdink.slice/" /proc/self/cgroup 2>/dev/null || exec "$here/memcap.sh" "$0" "$@"
env_dir="${IMMERDINK_ENV:-$HOME/miniforge3/envs/immerdink}"
builds="${IMMERDINK_STATUS_BUILDS:-$root/build-status}"
[ -x "$env_dir/bin/nvcc" ] && export PATH="$env_dir/bin:$PATH"
run() {  # name archs expected-status expected-errors
  local name="$1" archs="$2" expect="$3" errs="$4" b="$builds/$1"
  echo "== $name: CMAKE_CUDA_ARCHITECTURES=$archs"
  local tc=()
  [ -x "$env_dir/bin/nvcc" ] && tc=(-DCMAKE_C_COMPILER="$env_dir/bin/gcc" -DCMAKE_CXX_COMPILER="$env_dir/bin/g++"
      -DCMAKE_CUDA_COMPILER="$env_dir/bin/nvcc" -DCMAKE_CUDA_HOST_COMPILER="$env_dir/bin/g++"
      -DPython3_EXECUTABLE="$env_dir/bin/python3")
  cmake -S "$root" -B "$b" -G Ninja "${tc[@]}" -DCMAKE_CUDA_ARCHITECTURES="$archs" >/dev/null
  nice -n 19 cmake --build "$b" --target immerdink -j "${JOBS:-4}" >/dev/null
  python3 "$root/tests/test_device_status.py" --lib "$b/libimmerdink.so" --expect "$expect" ${errs:+--expect-error "$errs"}
}
case "${1:-all}" in
  native)  run native 89-real native "" ;;
  ptxonly) run ptxonly 80-virtual recompiled,unsupported 222 ;;
  foreign) run foreign 90-real unsupported 209 ;;
  all) "$0" native; "$0" ptxonly; "$0" foreign ;;
  *) echo "usage: $0 native|ptxonly|foreign|all" >&2; exit 2 ;;
esac
