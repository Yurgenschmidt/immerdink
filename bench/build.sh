#!/usr/bin/env bash
# Configure and build the benchmark harness (and libimmerdink from this checkout) into
# build/bench. Toolchain: the immerdink conda env if it exists, else whatever is on PATH
# (e.g. `module load cuda gcc cmake` on an HPC system): nvcc >= 12.8 (13.x preferred), a C++20
# host compiler (gcc >= 11), cmake >= 3.28, ninja or make.
#   bench/build.sh
# Environment:
#   IMMERDINK_ENV          conda env (default ~/miniforge3/envs/immerdink; used only if it has nvcc)
#   IMMERDINK_BENCH_BUILD  build directory (default <repo>/build/bench)
#   IMMERDINK_CUTLASS_DIR  CUTLASS v4.8.0 checkout (default: the third_party/cutlass submodule,
#                          fetched here if missing: needs network, so do it on a login node)
#   CMAKE_CUDA_ARCHITECTURES  GPU architecture(s) (default: detected from nvidia-smi, e.g. 120 for
#                          RTX PRO 6000 Blackwell, else 89); changing it needs a fresh build directory
#   CUTENSOR_ROOT          (or CUTENSOR_DIR / CUTENSOR_HOME) a cuTENSOR install (include/cutensor.h, lib/libcutensor.so);
#                          optional: enables the cutensor / cutensor-emu baselines (also found in the CUDA
#                          toolkit and the conda env)
#   IMMERDINK_LIBRARY      link this prebuilt libimmerdink.so instead of building one
# The build is CPU-heavy: it runs under nice and takes the machine-wide benchmark lock,
# so it never overlaps a gated timing run; it runs inside tools/memcap.sh with at most 4 jobs.
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
root="$(dirname "$here")"
env_dir="${IMMERDINK_ENV:-$HOME/miniforge3/envs/immerdink}"
build="${IMMERDINK_BENCH_BUILD:-$root/build/bench}"
cfg=(-S "$here" -B "$build")
if [ -x "$env_dir/bin/nvcc" ]; then
  export PATH="$env_dir/bin:$PATH" CONDA_PREFIX="$env_dir"
  cfg+=(-DCMAKE_C_COMPILER="$env_dir/bin/gcc" -DCMAKE_CXX_COMPILER="$env_dir/bin/g++"
        -DCMAKE_CUDA_COMPILER="$env_dir/bin/nvcc" -DCMAKE_CUDA_HOST_COMPILER="$env_dir/bin/g++"
        -DPython3_EXECUTABLE="$env_dir/bin/python3")
else
  nvcc="$(command -v nvcc || true)"
  [ -z "$nvcc" ] && [ -x "${CUDA_HOME:-/usr/local/cuda}/bin/nvcc" ] && nvcc="${CUDA_HOME:-/usr/local/cuda}/bin/nvcc"
  [ -n "$nvcc" ] || { echo "no nvcc: put one on PATH (module load cuda), set CUDA_HOME, or set IMMERDINK_ENV" >&2; exit 2; }
  echo "build: using $nvcc (no conda env at $env_dir)"
  cfg+=(-DCMAKE_CUDA_COMPILER="$nvcc")
fi
command -v cmake >/dev/null || { echo "no cmake on PATH (module load cmake)" >&2; exit 2; }
if command -v ninja >/dev/null; then cfg+=(-G Ninja); fi
if [ -z "${CMAKE_CUDA_ARCHITECTURES:-}" ] && command -v nvidia-smi >/dev/null; then
  cc="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d '. ')"
  [[ "$cc" =~ ^[0-9]+$ ]] && export CMAKE_CUDA_ARCHITECTURES="$cc" && echo "build: GPU architecture $cc (detected)"
fi
cutlass="${IMMERDINK_CUTLASS_DIR:-$root/third_party/cutlass}"
if [ ! -f "$cutlass/include/cutlass/cutlass.h" ] && [ -z "${IMMERDINK_CUTLASS_DIR:-}" ]; then
  echo "build: fetching the CUTLASS submodule (needs network)"
  git -C "$root" submodule update --init --depth 1 third_party/cutlass \
    || { echo "could not fetch CUTLASS; run 'git submodule update --init --depth 1 third_party/cutlass' on a machine with network, or set IMMERDINK_CUTLASS_DIR" >&2; exit 2; }
fi
[ -n "${IMMERDINK_CUTLASS_DIR:-}" ] && cfg+=(-DIMMERDINK_CUTLASS_DIR="$IMMERDINK_CUTLASS_DIR")
[ -n "${CMAKE_CUDA_ARCHITECTURES:-}" ] && cfg+=(-DCMAKE_CUDA_ARCHITECTURES="$CMAKE_CUDA_ARCHITECTURES")
[ -n "${IMMERDINK_LIBRARY:-}" ] && cfg+=(-DIMMERDINK_LIBRARY="$IMMERDINK_LIBRARY")

exec 9>"${TLC_LOCK:-${XDG_RUNTIME_DIR:-/tmp}/immerdink-tlc.lock}"
flock 9
cmake "${cfg[@]}"
nice -n 19 "$root/tools/memcap.sh" cmake --build "$build" --target imd_bench -j "$(( ${JOBS:-4} < 4 ? ${JOBS:-4} : 4 ))"
echo "built: $build/imd_bench"
