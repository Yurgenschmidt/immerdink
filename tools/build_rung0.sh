#!/usr/bin/env bash
# Build rung 0 of the Hopper K4 ladder (src/k4/hopper/rung0.cu) for sm_90a and, with `run`, run it.
#   tools/build_rung0.sh OUTDIR [run]
# Needs nvcc 12.x+ and CUTLASS 4.x (IMMERDINK_CUTLASS_DIR, default third_party/cutlass). No CMake: one nvcc call.
# -DNDEBUG (as build_rung1.sh): without it CUTLASS's assert leaves a call in the main loop, ptxas serialises the wgmma (C7510) and rung 0 would test different machine code from the timing run.
# `run` is limited to RUNG0_TIMEOUT seconds (default 600; timeout exits 124). It needs a Hopper GPU; take the GPU lock yourself on a shared box.
# Output: OUTDIR/imd_rung0; with `run`, OUTDIR/rung0.jsonl (one JSON line per variant and mode) and the exit status of the program.
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
root="$(dirname "$here")"
out="${1:?usage: build_rung0.sh OUTDIR [run]}"
mkdir -p "$out"
cutlass="${IMMERDINK_CUTLASS_DIR:-$root/third_party/cutlass}"
nvcc -std=c++17 -arch=sm_90a --expt-relaxed-constexpr -O2 -DNDEBUG \
  -I "$root/src" -I "$root/include" -I "$cutlass/include" \
  -o "$out/imd_rung0" "$root/src/k4/hopper/rung0.cu"
echo "built $out/imd_rung0"
if [ "${2:-}" = run ]; then
  set +e
  # a pipeline hang (these kernels have not run on a Hopper card before) must not hold the allocation: bounded, and on the dev box (no Hopper) it exits 3
  timeout "${RUNG0_TIMEOUT:-600}" "$out/imd_rung0" > "$out/rung0.jsonl"
  rc=$?
  echo "imd_rung0 exit $rc"
  exit $rc
fi
