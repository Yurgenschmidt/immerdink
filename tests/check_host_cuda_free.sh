#!/usr/bin/env bash
# libimmerdink_host is CUDA-free (DESIGN §6a).
#
#   check_host_cuda_free.sh LIB
#       LIB (libimmerdink_host.so) has no CUDA / NVIDIA library among its DT_NEEDED entries
#       or its resolved dependencies (readelf, ldd), and no defined or undefined symbol that
#       mentions cuda (nm -D).
#   check_host_cuda_free.sh LIB --standalone SRC BUILD CXX PYTHON CMAKE
#       The above, then: in a clean environment (env -i, PATH=/usr/bin:/bin, so no nvcc and
#       no CUDA library path), configure SRC with -DIMMERDINK_HOST_ONLY=ON into BUILD using the
#       system C/C++ compiler (CXX if there is none), check that CMake found no CUDA compiler
#       or toolkit, build, run the host tests there (ctest, which reruns this check on the
#       standalone library), and check the standalone library like LIB.
set -euo pipefail

cuda_re='cuda|cudart|nvidia|nvrtc|cublas|cufft|curand|cusparse|cusolver|nvjitlink|libcu[a-z]*\.so'

check_lib() {
  local lib="$1" bad
  [ -f "$lib" ] || { echo "FAIL: no library at $lib"; return 1; }
  bad=$(readelf -d "$lib" | awk '/NEEDED/' | grep -Ei "$cuda_re" || true)
  [ -z "$bad" ] || { echo "FAIL: $lib needs a CUDA library:"; echo "$bad"; return 1; }
  # the library names only (field 1): a conda env called immerdink-cuda13 puts "cuda" in the PATH of libstdc++ (found on PACE)
  bad=$(ldd "$lib" | awk '{print $1}' | grep -Ei "$cuda_re" || true)
  [ -z "$bad" ] || { echo "FAIL: $lib resolves a CUDA library:"; echo "$bad"; return 1; }
  bad=$(nm -D "$lib" | grep -i cuda || true)
  [ -z "$bad" ] || { echo "FAIL: $lib has CUDA symbols:"; echo "$bad"; return 1; }
  echo "ok  $lib: DT_NEEDED $(readelf -d "$lib" | awk '/NEEDED/ {printf "%s ", $NF}')"
  echo "ok  $lib: no CUDA dependency (readelf, ldd) and no CUDA symbol (nm -D: $(nm -D "$lib" | wc -l) symbols)"
}

lib="$1"
shift
check_lib "$lib"
[ "${1:-}" = "--standalone" ] || exit 0
src="$2" build="$3" cxx="$4" python="$5" cmake="$6"
bindir="$(dirname "$cmake")"
cc_sys=/usr/bin/gcc cxx_sys=/usr/bin/g++
if [ ! -x "$cxx_sys" ]; then cxx_sys="$cxx"; cc_sys="$(dirname "$cxx")/gcc"; fi
rm -rf "$build"
clean=(env -i HOME="$HOME" PATH=/usr/bin:/bin LANG=C.UTF-8)
if "${clean[@]}" bash -c 'command -v nvcc' >/dev/null; then
  echo "FAIL: nvcc is on the clean PATH; this check cannot prove anything"; exit 1
fi
echo "ok  nvcc is not on PATH (/usr/bin:/bin)"
"${clean[@]}" "$cmake" -S "$src" -B "$build" -G "Unix Makefiles" -DIMMERDINK_HOST_ONLY=ON \
  -DCMAKE_C_COMPILER="$cc_sys" -DCMAKE_CXX_COMPILER="$cxx_sys" -DPython3_EXECUTABLE="$python" \
  -DIMMERDINK_WERROR=ON > "$build.configure.log" 2>&1 || { cat "$build.configure.log"; exit 1; }
if grep -Eq '^CMAKE_CUDA_COMPILER:[A-Z]+=.+' "$build/CMakeCache.txt" ||
   grep -Eq '^CUDAToolkit_[A-Za-z_]*:[A-Z]+=/' "$build/CMakeCache.txt"; then
  echo "FAIL: the host-only configure found CUDA:"; grep -E 'CUDA' "$build/CMakeCache.txt"; exit 1
fi
echo "ok  configured with $cxx_sys, no CUDA compiler or toolkit in the CMake cache"
"${clean[@]}" "$cmake" --build "$build" -j 2 > "$build.build.log" 2>&1 || { tail -50 "$build.build.log"; exit 1; }
echo "ok  built: $(cd "$build" && ls *.so tests/test_* 2>/dev/null | tr '\n' ' ')"
"${clean[@]}" "$bindir/ctest" --test-dir "$build" --output-on-failure
check_lib "$build/libimmerdink_host.so"
