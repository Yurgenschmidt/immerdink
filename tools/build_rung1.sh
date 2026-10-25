#!/usr/bin/env bash
# Build rung 1 of the Hopper K4 ladder (src/k4/hopper/rung1.cu, the timing harness, docs/NOTES-22.md) for sm_90a and, with `run`, run it.
#   tools/build_rung1.sh OUTDIR [run]
# Needs nvcc 12.x+ and CUTLASS 4.x (IMMERDINK_CUTLASS_DIR, default third_party/cutlass). No CMake: one nvcc call, -O2 -DNDEBUG (without NDEBUG
# CUTLASS's assert leaves a call in the main loop, ptxas prints C7510 and serialises the wgmma: the numbers would be wrong). Extra nvcc flags: IMD_NVCC_FLAGS
# (for example -Xptxas -v to see registers and spills). NVML is not linked (the program dlopens it if present).
# `run` is limited to RUNG1_TIMEOUT seconds (default 1800; timeout exits 124) and passes RUNG1_ARGS (for example "--quick") to the program. It needs a
# Hopper GPU; take the GPU lock yourself on a shared box. Without a Hopper the program exits 3 (message on stderr).
# Output: OUTDIR/imd_rung1; with `run`, OUTDIR/rung1.jsonl (one JSON line per variant, shape, batch, scheduler and mode; stderr has the progress) and the exit status.
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
root="$(dirname "$here")"
out="${1:?usage: build_rung1.sh OUTDIR [run]}"
mkdir -p "$out"
cutlass="${IMMERDINK_CUTLASS_DIR:-$root/third_party/cutlass}"
# shellcheck disable=SC2086
nvcc -std=c++17 -arch=sm_90a --expt-relaxed-constexpr -O2 -DNDEBUG ${IMD_NVCC_FLAGS:-} \
  -I "$root/src" -I "$root/include" -I "$cutlass/include" \
  -o "$out/imd_rung1" "$root/src/k4/hopper/rung1.cu" -ldl -Xcompiler -pthread
echo "built $out/imd_rung1"
if [ "${2:-}" = run ]; then
  set +e
  # a pipeline hang (these kernels have not run on a Hopper card before) must not hold the allocation: bounded; on a box without Hopper it exits 3
  # shellcheck disable=SC2086
  timeout "${RUNG1_TIMEOUT:-1800}" "$out/imd_rung1" ${RUNG1_ARGS:-} > "$out/rung1.jsonl"
  rc=$?
  echo "imd_rung1 exit $rc"
  exit $rc
fi
