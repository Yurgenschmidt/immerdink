#!/usr/bin/env bash
# Build rung 0 of the Hopper K4 ladder (src/k4/hopper/rung0.cu) for sm_90a and, with `run`, run it.
#   tools/build_rung0.sh OUTDIR [run]
# Needs nvcc 12.x+ and CUTLASS 4.x (IMMERDINK_CUTLASS_DIR, default third_party/cutlass). No CMake: one nvcc call.
# Output: OUTDIR/imd_rung0; with `run`, OUTDIR/rung0.jsonl (one JSON line per variant and mode) and the exit status of the program.
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
root="$(dirname "$here")"
out="${1:?usage: build_rung0.sh OUTDIR [run]}"
mkdir -p "$out"
cutlass="${IMMERDINK_CUTLASS_DIR:-$root/third_party/cutlass}"
nvcc -std=c++17 -arch=sm_90a --expt-relaxed-constexpr -O2 \
  -I "$root/src" -I "$root/include" -I "$cutlass/include" \
  -o "$out/imd_rung0" "$root/src/k4/hopper/rung0.cu"
echo "built $out/imd_rung0"
if [ "${2:-}" = run ]; then
  set +e
  "$out/imd_rung0" > "$out/rung0.jsonl"
  rc=$?
  echo "imd_rung0 exit $rc"
  exit $rc
fi
