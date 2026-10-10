#!/usr/bin/env bash
# Recreate the performance metrics of an immerdink build on this machine's GPU: the throughput /
# accuracy sweep (bench/sweep_imd.txt), then the board-energy sweep (bench/sweep_energy.txt).
#   bench/run_metrics.sh <output directory>
# The output directory is required and must be OUTSIDE this repository (results are never kept in
# the repo tree). It receives:
#   sweep_imd.jsonl   throughput / accuracy records (+ a clock_state line after each case)
#   energy.jsonl      energy records (idle baseline + one record per case and N)
#   summary.md        bench/summarize.py tables of the throughput sweep
#   conditions.txt    commit, host, GPU, driver, the clock-lock check, the run dates
#   nvidia-smi-q.txt  nvidia-smi -q before the run
#   sweep.out, energy.out   the run logs
# Every case goes through tools/run_bench.sh (machine-wide lock, shared GPU lock, wait for a quiet
# machine; it never bypasses the gate), so it can wait, or stop with exit 3 if the machine never
# gets quiet. On a machine with no other GPU users the shared-GPU-lock part is switched off with
# IMMERDINK_GPU_LOCK=none (the quiet checks stay). Publication-grade numbers need locked clocks
# (docs/BENCHMARKING.md): `sudo nvidia-smi -lgc <f>,<f>` first, `-rgc` afterwards.
# Environment: what bench/build.sh, bench/run_sweep.sh and tools/run_bench.sh read (IMMERDINK_ENV,
# IMMERDINK_BENCH_BUILD, CMAKE_CUDA_ARCHITECTURES, IMMERDINK_GPU_LOCK, QUIET_*), plus
#   SWEEPS="imd energy"   which sweeps to run (default both)
set -uo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
root="$(dirname "$here")"
[ $# -eq 1 ] || { echo "usage: $0 <output directory outside the repository>" >&2; exit 2; }
outdir="$(realpath -m -- "$1")"
rootp="$(cd "$root" && pwd -P)"
case "$outdir/" in
  "$rootp"/*) echo "refusing: $outdir is inside the repository $rootp; give a directory outside it" >&2; exit 2 ;;
esac
mkdir -p "$outdir" || exit 2

commit="$(git -C "$root" rev-parse HEAD 2>/dev/null || echo unknown)"
dirty=""
git -C "$root" diff --quiet HEAD -- src include bench CMakeLists.txt 2>/dev/null \
  || dirty=" (dirty: uncommitted changes in src/include/bench)"
{
  echo "date: $(date -Is)"
  echo "host: $(hostname)  kernel: $(uname -sr)"
  echo "commit: $commit$dirty"
  echo "gpu: $(nvidia-smi --query-gpu=name,uuid,vbios_version,driver_version,power.limit,clocks.max.sm --format=csv,noheader 2>/dev/null)"
  echo "cuda_arch: ${CMAKE_CUDA_ARCHITECTURES:-89 (default)}"
  echo "clock_lock_check (sm MHz, max sm MHz, applications_clocks_setting): $(nvidia-smi --query-gpu=clocks.sm,clocks.max.sm,clocks_event_reasons.applications_clocks_setting --format=csv,noheader 2>/dev/null)"
  echo "  (an idle sm clock at the locked value means the clocks are locked; the clock_state lines in"
  echo "   sweep_imd.jsonl record the clock and the event reasons after each case's warm-up)"
} > "$outdir/conditions.txt"
nvidia-smi -q > "$outdir/nvidia-smi-q.txt" 2>&1 || true
echo "metrics -> $outdir (commit ${commit:0:7}$dirty)"
cat "$outdir/conditions.txt"

rc=0
for sw in ${SWEEPS:-imd energy}; do
  case "$sw" in
    imd)
      "$here/run_sweep.sh" "$here/sweep_imd.txt" "$outdir/sweep_imd.jsonl" 2>&1 | tee "$outdir/sweep.out"
      st=${PIPESTATUS[0]}; [ "$st" = 0 ] || rc=$st
      [ -s "$outdir/sweep_imd.jsonl" ] && python3 "$here/summarize.py" "$outdir/sweep_imd.jsonl" > "$outdir/summary.md" 2>/dev/null
      ;;
    energy)
      # one attempt per case: an energy case takes minutes and its spread is not the retry criterion
      BENCH_RETRIES=1 "$here/run_sweep.sh" "$here/sweep_energy.txt" "$outdir/energy.jsonl" 2>&1 | tee "$outdir/energy.out"
      st=${PIPESTATUS[0]}; [ "$st" = 0 ] || rc=$st
      ;;
    *) echo "unknown sweep '$sw' in SWEEPS (imd, energy)" >&2; exit 2 ;;
  esac
done
echo "finished: $(date -Is)" >> "$outdir/conditions.txt"
exit $rc
