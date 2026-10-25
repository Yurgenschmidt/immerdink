#!/usr/bin/env bash
# Static SASS instruction counts of src/contract.cu per architecture, and the modulus-loop size of the pack kernels
# (docs/NOTES-21.md "Pack emitter and finish"). Builds only that translation unit, in three build directories (89 / 90 / 120),
# each configured once like tools/test.sh does (tests off; the SASS does not depend on them). nvcc and cuobjdump from the immerdink env.
#   tools/sass_seq_counts.sh [tag]      -> $out/sass_<tag>_<arch>.txt (counts) and sass_<tag>_<arch>_loops.txt (loop sizes)
# Environment: IMMERDINK_ENV (default ~/miniforge3/envs/immerdink), IMMERDINK_CUTLASS_DIR, SASS_OUT (default build-sass), SASS_JOBS (default 2).
set -euo pipefail
root="$(cd "$(dirname "$0")/.." && pwd)"
env_dir="${IMMERDINK_ENV:-$HOME/miniforge3/envs/immerdink}"
export PATH="$env_dir/bin:$PATH"
tag="${1:-now}"; out="${SASS_OUT:-$root/build-sass}"; mkdir -p "$out"
for a in 89 90 120; do
  b="$out/build-sm$a"
  [ -f "$b/build.ninja" ] || cmake -S "$root" -B "$b" -G Ninja -DCMAKE_C_COMPILER="$env_dir/bin/gcc" -DCMAKE_CXX_COMPILER="$env_dir/bin/g++" \
    -DCMAKE_CUDA_COMPILER="$env_dir/bin/nvcc" -DCMAKE_CUDA_HOST_COMPILER="$env_dir/bin/g++" -DPython3_EXECUTABLE="$env_dir/bin/python3" \
    -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=$a -DIMMERDINK_BUILD_TESTS=OFF -DIMMERDINK_WERROR=OFF \
    ${IMMERDINK_CUTLASS_DIR:+-DIMMERDINK_CUTLASS_DIR="$IMMERDINK_CUTLASS_DIR"} > /dev/null
  nice -n 19 ninja -C "$b" -j"${SASS_JOBS:-2}" CMakeFiles/immerdink.dir/src/contract.cu.o > /dev/null
  obj="$b/CMakeFiles/immerdink.dir/src/contract.cu.o"
  python3 -I "$root/tools/sass_counts.py" "$obj" | sort -k2 > "$out/sass_${tag}_$a.txt"
  python3 -I "$root/tools/sass_loops.py" "$obj" '^IDP' 'kl_scale_pack|k23_pack|kf_scale_pack' | sort -k3 > "$out/sass_${tag}_${a}_loops.txt"
  echo "sm_$a: $(wc -l < "$out/sass_${tag}_$a.txt") kernels -> $out/sass_${tag}_$a.txt"
done
