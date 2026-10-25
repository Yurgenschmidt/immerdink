#!/usr/bin/env bash
# Build rung 1 of the Hopper K4 ladder (src/k4/hopper/rung1.cu, the timing harness, docs/NOTES-22.md) for sm_90a, check the build, and with `run` run it.
#   tools/build_rung1.sh OUTDIR [run]
# Needs nvcc 12.x+ and CUTLASS 4.x (IMMERDINK_CUTLASS_DIR, default third_party/cutlass). No CMake: one nvcc call, -O2 -DNDEBUG (without NDEBUG CUTLASS's
# assert leaves a call in the main loop, ptxas prints C7510 and serialises the wgmma: the numbers would be wrong). rung1.cu and src/k4/k4_cfg0.cu (the
# product's mma.sync K4, the baseline row) are the two sources of that one call. Extra nvcc flags: IMD_NVCC_FLAGS. NVML is not linked (dlopen).
# Checks after the build (OUTDIR/build.log = nvcc -Xptxas -v output; OUTDIR/sass_counts.txt): ptxas C7510 anywhere is FATAL (exit 6; set
# RUNG1_ALLOW_C7510=1 to run anyway), spills and the count of WARPGROUP.DEPBAR against IGMMA per kernel are printed (DEPBAR >= IGMMA in a wgmma kernel
# means the wgmma are serialised: a warning; the synthetic peak kernels of the p_ rows are listed as "peak wgmma-nN wgW stagesS", and a CALL in one of them is
# a warning too). The nvcc version goes to OUTDIR/nvcc_version.txt and, as IMD_NVCC_VERSION, into the program's meta record.
# `run` is limited to RUNG1_TIMEOUT seconds (default 1800; `timeout -k 30`: TERM, then KILL 30 s later; exit 124/137) and passes RUNG1_ARGS (for example
# "--quick") to the program. It needs a Hopper GPU; take the GPU lock yourself on a shared box. Without a Hopper the program exits 3 (message on stderr).
# Output: OUTDIR/imd_rung1; with `run`, OUTDIR/rung1.jsonl (meta record first, then one JSON line per variant, shape, batch, scheduler and mode;
# OUTDIR/rung1.err has the progress) and the exit status (program codes: header of rung1.cu; 5 = some error/skipped rows).
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
root="$(dirname "$here")"
out="${1:?usage: build_rung1.sh OUTDIR [run]}"
mkdir -p "$out"
cutlass="${IMMERDINK_CUTLASS_DIR:-$root/third_party/cutlass}"
[ -f "$cutlass/include/cutlass/cutlass.h" ] || { echo "build_rung1: no CUTLASS at $cutlass (include/cutlass/cutlass.h missing): set IMMERDINK_CUTLASS_DIR to a CUTLASS 4.x checkout" >&2; exit 2; }
command -v nvcc >/dev/null || { echo "build_rung1: nvcc not found on PATH (load the CUDA module / conda env)" >&2; exit 2; }
nvcc --version | tail -2 > "$out/nvcc_version.txt"
nvcc_ver="$(sed -n 's/.*release \([0-9.]*\).*/\1/p;s/^Build //p' "$out/nvcc_version.txt" | tr '\n' ' ')"
echo "nvcc: $nvcc_ver"
# shellcheck disable=SC2086
nvcc -std=c++17 -arch=sm_90a --expt-relaxed-constexpr -O2 -DNDEBUG -Xptxas -v ${IMD_NVCC_FLAGS:-} \
  -I "$root/src" -I "$root/include" -I "$cutlass/include" \
  -o "$out/imd_rung1" "$root/src/k4/hopper/rung1.cu" "$root/src/k4/k4_cfg0.cu" -ldl -Xcompiler -pthread > "$out/build.log" 2>&1 \
  || { tail -40 "$out/build.log" >&2; echo "build_rung1: nvcc failed (full log $out/build.log)" >&2; exit 2; }
echo "built $out/imd_rung1"
c7510=$(grep -c C7510 "$out/build.log" || true)
spills=$(grep -o '[0-9]* bytes spill stores' "$out/build.log" | awk '{s+=$1} END {print s+0}')
echo "ptxas: C7510 lines $c7510, spill-store bytes $spills (all kernels; build.log has the per-kernel report)"
if [ "$c7510" != 0 ]; then
  echo "build_rung1: !!! ptxas C7510: the wgmma are SERIALISED in this build (see build.log). Its timings are not the product's." >&2
  [ "${RUNG1_ALLOW_C7510:-0}" = 1 ] || { echo "build_rung1: refusing to run (RUNG1_ALLOW_C7510=1 overrides)" >&2; exit 6; }
fi
if command -v cuobjdump >/dev/null; then
  cuobjdump -sass "$out/imd_rung1" 2>/dev/null | awk '
    /Function : / { if (name != "") report(); name = $3; ig = 0; dep = 0; call = 0 }
    /IGMMA/ { ig++ } /WARPGROUP\.DEPBAR/ { dep++ } /CALL\./ { call++ }
    function report(  n, f, sc, t, a, st, lab) { if (ig > 0) {
      st = (dep >= ig ? "SERIAL?" : "ok")
      if (name ~ /k_wgmma_peak/) {  # the synthetic peak kernels (p_ rows): k_wgmma_peak<N, warpgroups, stages>; any CALL in them is a failure (the loop must be straight-line)
        match(name, /k_wgmma_peakILi[0-9]+ELi[0-9]+ELi[0-9]+E/); t = substr(name, RSTART, RLENGTH); gsub(/k_wgmma_peakILi|E$/, "", t); split(t, a, "ELi")
        if (call > 0) st = "CALL?"
        printf "%-8s IGMMA %3d  WARPGROUP.DEPBAR %3d  CALL %d  peak wgmma-n%s wg%s stages%s\n", st, ig, dep, call, a[1], a[2], a[3]
      } else {
        match(name, /MMA_64x[0-9]+x32/); n = substr(name, RSTART + 4, RLENGTH - 7); match(name, /K4ModEpilogueILb0ELb[01]/); f = substr(name, RSTART + RLENGTH - 1, 1) == "1" ? "fast" : "generic"; sc = (name ~ /Pingpong/) ? "ping" : "coop"
        printf "%-8s IGMMA %3d  WARPGROUP.DEPBAR %3d  CALL %d  %s wgmma-%s %s\n", st, ig, dep, call, sc, n, f } } }
    END { if (name != "") report() }' > "$out/sass_counts.txt"
  cat "$out/sass_counts.txt"
  if grep -q '^SERIAL?' "$out/sass_counts.txt"; then echo "build_rung1: WARNING: a wgmma kernel has as many DEPBAR as IGMMA (serialised); see sass_counts.txt" >&2; fi
  if grep -q '^CALL?' "$out/sass_counts.txt"; then echo "build_rung1: WARNING: a synthetic peak kernel (p_ rows) contains a CALL: its loop is not straight-line, its rate is not the instruction ceiling; see sass_counts.txt" >&2; fi
  if ! grep -q 'peak wgmma' "$out/sass_counts.txt"; then echo "build_rung1: WARNING: no synthetic peak kernel (k_wgmma_peak) found in the SASS" >&2; fi
else
  echo "build_rung1: no cuobjdump: the DEPBAR / IGMMA check was skipped" >&2
fi
if [ "${2:-}" = run ]; then
  set +e
  export IMD_NVCC_VERSION="$nvcc_ver"
  # a pipeline hang (these kernels have not run on a Hopper card before) must not hold the allocation: bounded; on a box without Hopper it exits 3
  # shellcheck disable=SC2086
  timeout -k 30 "${RUNG1_TIMEOUT:-1800}" "$out/imd_rung1" ${RUNG1_ARGS:-} > "$out/rung1.jsonl" 2> "$out/rung1.err"
  rc=$?
  tail -3 "$out/rung1.err"
  echo "imd_rung1 exit $rc"
  exit $rc
fi
