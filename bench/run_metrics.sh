#!/usr/bin/env bash
# Recreate the performance metrics of an immerdink build on this machine's GPU: a GPU calibration
# (bench/calibrate.sh), the throughput / accuracy sweep (bench/sweep_imd.txt), the cuTENSOR native /
# emulated FP64 baselines (bench/sweep_baselines.txt, if imd_bench was built with cuTENSOR) and the
# board-energy sweep (bench/sweep_energy.txt: immerdink, and cuTENSOR where built), and the comparison
# sweeps: immerdink against GEMMul8, cuBLAS FP64 emulation, cuBLAS DGEMM and cuTENSOR native / emulated,
# throughput + accuracy (bench/sweep_compare.txt) and board energy (bench/sweep_compare_energy.txt); last,
# an nsys profile pass (bench/profile_nsys.sh): short runs of immerdink and the baselines under Nsight Systems,
# summarized by kernel category, with the check whether cuTENSOR's emulation really ran INT8 kernels.
#   bench/run_metrics.sh <output directory>
# The output directory is required and must be OUTSIDE this repository (results are never kept in
# the repo tree). It receives:
#   calibration.json  GPU calibration (stable clock / power / temperature, recommended lock)
#   sweep_imd.jsonl   throughput / accuracy records (+ a clock_state line after each case)
#   baselines.jsonl   cuTENSOR native / emulated throughput records
#   energy.jsonl      energy records (idle baseline + one record per case and N)
#   compare.jsonl     throughput / accuracy of immerdink and the baselines on the same inputs (sweep_compare.txt)
#   cutensor_ws.jsonl     cuTENSOR native vs emulated at workspace preference default and max (sweep_cutensor_ws.txt);
#                         cutensor_ws.md is its table (cutws_table.py)
#   compare_energy.jsonl  board-energy records of the same configurations (sweep_compare_energy.txt)
#   profile/          nsys pass: index.json, <tag>/summary.json (kernels by category, engaged_int8), <tag>/run.log;
#                     the .nsys-rep files only with PROFILE_KEEP=1 (bench/profile_nsys.sh, docs/BENCHMARKING.md)
#   summary.md        bench/summarize.py tables of the throughput sweeps (incl. cuTENSOR vs immerdink)
#   energy.md         bench/summarize.py table of the energy sweep
#   compare.md, compare_energy.md   the comparison tables at DGEMM-level accuracy (summarize.py --only-compare)
#   conditions.txt    commit, host, GPU, driver, the clock-lock check, the run dates
#   nvidia-smi-q.txt  nvidia-smi -q before the run
#   sweep.out, energy.out, profile.out   the run logs
# Every case goes through tools/run_bench.sh (machine-wide lock, shared GPU lock, wait for a quiet
# machine; it never bypasses the gate), so it can wait, or stop with exit 3 if the machine never
# gets quiet. On a machine with no other GPU users the shared-GPU-lock part is switched off with
# IMMERDINK_GPU_LOCK=none (the quiet checks stay). Publication-grade numbers need locked clocks
# (docs/BENCHMARKING.md): `sudo nvidia-smi -lgc <f>,<f>` first, `-rgc` afterwards.
# Environment: what bench/build.sh, bench/run_sweep.sh and tools/run_bench.sh read (IMMERDINK_ENV,
# IMMERDINK_BENCH_BUILD, CMAKE_CUDA_ARCHITECTURES, IMMERDINK_GPU_LOCK, QUIET_*), plus
#   SWEEPS="imd baseline cutws compare compare_energy energy profile"   which sweeps to run (the default, in this
#                         order; two more tokens exist that are NOT in the default: xduel (sweep_xduel.txt -> xduel.jsonl) and
#                         extra (sweep_r5.txt -> r5.jsonl: same-job new-vs-old and wgmma-vs-mma duels, sustained large squares, 16384);
#                         the rest of this description is the default list
#                         order; compare / compare_energy are the baselines of docs/BENCHMARKING.md Test 4;
#                         profile = the nsys pass, always last: it needs nsys, and skips itself with a
#                         note when there is none)
#   PROFILE_KEEP=1, PROFILE_TAGS=<regex>   keep the .nsys-rep files / profile only the matching runs (bench/profile_nsys.sh)
#   GEMMUL8_DIR           a GEMMul8 install (bench/build_gemmul8.sh): enables the gemmul8 baseline (bench/build.sh)
#   CALIBRATE=0           skip the GPU calibration (default: run it first, about a minute)
#   LOCK_CLOCKS=auto      if the calibration finds unlocked (or unclean) clocks and passwordless sudo works,
#                         lock the clock at the recommended value (nvidia-smi -lgc), re-calibrate, run the
#                         sweeps and restore the clocks (-rgc) on exit (also on Ctrl-C). Never done otherwise.
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


# --- GPU calibration ------------------------------------------------------------------------
jget() {  # jget <json file> <python expr on d>
  python3 -c 'import json,sys; d=json.load(open(sys.argv[1])); v=eval(sys.argv[2]); print("" if v is None else v)' "$1" "$2" 2>/dev/null
}
locked_by_us=""
gid="${CUDA_VISIBLE_DEVICES:-0}"; gid="${gid%%,*}"
restore_clocks() {
  if [ -n "$locked_by_us" ]; then
    echo "restoring clocks: sudo -n nvidia-smi -i $gid -rgc"
    sudo -n nvidia-smi -i "$gid" -rgc >/dev/null 2>&1 || echo "WARNING: could not restore the clocks; run: sudo nvidia-smi -i $gid -rgc" >&2
    locked_by_us=""
  fi
}
trap restore_clocks EXIT
trap 'exit 130' INT TERM
calib_state="not run (CALIBRATE=0 or no python3)"
if [ "${CALIBRATE:-1}" != 0 ] && command -v python3 >/dev/null; then
  "$here/build.sh" || exit 1
  echo "== GPU calibration =="
  pre="$outdir"
  [ "${LOCK_CLOCKS:-}" = auto ] && { pre="$outdir/calibration_pre"; mkdir -p "$pre"; }
  "$here/calibrate.sh" --out "$pre" 2>&1 | tee "$outdir/calibration.out"
  crc=${PIPESTATUS[0]}
  [ "$crc" = 3 ] && { echo "machine never got quiet; stopping" >&2; exit 3; }
  if [ "$crc" = 5 ]; then
    if [ "${QUIET_MODE:-gate}" = record ]; then
      echo "WARNING: the GPU is busy with another process; QUIET_MODE=record: continuing, the numbers will be contaminated" >&2
    else
      echo "the GPU is busy with another process (shared GPU?); stopping (QUIET_MODE=record continues anyway, contaminated)" >&2; exit 3
    fi
  fi
  if [ "${LOCK_CLOCKS:-}" = auto ] && [ -s "$pre/calibration.json" ]; then
    lock_mhz="$(jget "$pre/calibration.json" 'd.get("recommended_lock_mhz") if "lock_command" in d else ""')"
    if [ -n "$lock_mhz" ]; then
      if sudo -n true 2>/dev/null; then
        echo "LOCK_CLOCKS=auto: sudo nvidia-smi -i $gid -lgc $lock_mhz,$lock_mhz"
        if sudo -n nvidia-smi -i "$gid" -lgc "$lock_mhz,$lock_mhz"; then
          locked_by_us=1
          echo "== re-calibrating at the locked clock =="
          "$here/calibrate.sh" --out "$outdir" 2>&1 | tee -a "$outdir/calibration.out"
        else
          echo "WARNING: locking at $lock_mhz MHz failed (not a supported clock?): see nvidia-smi -q -d SUPPORTED_CLOCKS; continuing UNLOCKED" >&2
          cp "$pre/calibration.json" "$outdir/calibration.json"
        fi
      else
        echo "LOCK_CLOCKS=auto: no passwordless sudo; lock the clock yourself and re-run:" >&2
        jget "$pre/calibration.json" 'd.get("lock_command")' >&2
        cp "$pre/calibration.json" "$outdir/calibration.json"
      fi
    else
      cp "$pre/calibration.json" "$outdir/calibration.json"
    fi
  fi
  if [ -s "$outdir/calibration.json" ]; then
    calib_state="stable=$(jget "$outdir/calibration.json" 'd["stable"]') after $(jget "$outdir/calibration.json" 'd["seconds_to_stable"] and round(d["seconds_to_stable"])') s; steady SM $(jget "$outdir/calibration.json" 'd["steady_sm_mhz"]["median"]') MHz (min $(jget "$outdir/calibration.json" 'd["steady_sm_mhz"]["min"]'), max $(jget "$outdir/calibration.json" 'd["steady_sm_mhz"]["max"]')); power $(jget "$outdir/calibration.json" 'round(d["power_w"]["mean"])') W; clocks locked=$(jget "$outdir/calibration.json" 'd["locked"]'); $(jget "$outdir/calibration.json" 'd["verdict"]')"
    if [ "$(jget "$outdir/calibration.json" 'd["locked"]')" != True ]; then
      echo
      echo "*** The clocks are NOT locked: numbers will carry boost-clock drift (records say so). For publication-grade"
      echo "*** numbers stop here, run the command below, and start again:"
      echo "***   $(jget "$outdir/calibration.json" 'd.get("lock_command")')     (undo: $(jget "$outdir/calibration.json" 'd.get("unlock_command")'))"
      echo "*** or let this script do it with LOCK_CLOCKS=auto (needs passwordless sudo). Continuing unlocked."
      echo
    fi
  fi
fi
{
  echo "calibration: $calib_state"
  echo "clocks locked by this run (LOCK_CLOCKS=auto): $([ -n "$locked_by_us" ] && echo yes || echo no)"
  echo "toolchain: IMMERDINK_ENV=${IMMERDINK_ENV:-$HOME/miniforge3/envs/immerdink}; $( { command -v nvcc >/dev/null && nvcc --version | tail -1; } 2>/dev/null || echo 'nvcc not on PATH')"
  if [ -n "${GEMMUL8_DIR:-}" ]; then
    echo "gemmul8: $GEMMUL8_DIR ($(tr '\n' ' ' < "$GEMMUL8_DIR/BUILD_INFO" 2>/dev/null | cut -c1-300))"
  else
    echo "gemmul8: not built in (GEMMUL8_DIR unset): the gemmul8 baseline records an error"
  fi
} >> "$outdir/conditions.txt"

# --- sweeps ---------------------------------------------------------------------------------
build="${IMMERDINK_BENCH_BUILD:-$root/build/bench}"
rc=0
for sw in ${SWEEPS:-imd baseline cutws compare compare_energy energy profile}; do
  case "$sw" in
    imd)
      "$here/run_sweep.sh" "$here/sweep_imd.txt" "$outdir/sweep_imd.jsonl" 2>&1 | tee "$outdir/sweep.out"
      st=${PIPESTATUS[0]}; [ "$st" = 0 ] || rc=$st
      ;;
    baseline)
      if [ -x "$build/imd_bench" ] && command -v ldd >/dev/null && ! ldd "$build/imd_bench" 2>/dev/null | grep -q libcutensor; then
        echo "baseline sweep skipped: imd_bench was built without cuTENSOR (module load cutensor / set CUTENSOR_ROOT, rebuild in a fresh IMMERDINK_BENCH_BUILD)" | tee "$outdir/baselines.out"
      else
        "$here/run_sweep.sh" "$here/sweep_baselines.txt" "$outdir/baselines.jsonl" 2>&1 | tee "$outdir/baselines.out"
        st=${PIPESTATUS[0]}; [ "$st" = 0 ] || rc=$st
      fi
      ;;
    energy)
      # one attempt per case: an energy case takes minutes and its spread is not the retry criterion
      BENCH_RETRIES=1 "$here/run_sweep.sh" "$here/sweep_energy.txt" "$outdir/energy.jsonl" 2>&1 | tee "$outdir/energy.out"
      st=${PIPESTATUS[0]}; [ "$st" = 0 ] || rc=$st
      ;;
    cutws)
      # cuTENSOR emulation vs native planned at workspace preference DEFAULT and MAX (bench/sweep_cutensor_ws.txt):
      # is the emulation held back by the plan's workspace preference? -> cutensor_ws.jsonl, cutensor_ws.md
      "$here/run_sweep.sh" "$here/sweep_cutensor_ws.txt" "$outdir/cutensor_ws.jsonl" 2>&1 | tee "$outdir/cutensor_ws.out"
      st=${PIPESTATUS[0]}; [ "$st" = 0 ] || rc=$st
      ;;
    compare)
      # immerdink vs GEMMul8 / cuBLAS emulation / DGEMM / cuTENSOR native + emulated; a method that is not built in
      # (no GEMMUL8_DIR, no CUDA 13 cuBLAS, no cuTENSOR) records an error and the rest still runs
      "$here/run_sweep.sh" "$here/sweep_compare.txt" "$outdir/compare.jsonl" 2>&1 | tee "$outdir/compare.out"
      st=${PIPESTATUS[0]}; [ "$st" = 0 ] || rc=$st
      ;;
    compare_energy)
      BENCH_RETRIES=1 "$here/run_sweep.sh" "$here/sweep_compare_energy.txt" "$outdir/compare_energy.jsonl" 2>&1 | tee "$outdir/compare_energy.out"
      st=${PIPESTATUS[0]}; [ "$st" = 0 ] || rc=$st
      ;;
    xduel)
      # interleaved immerdink vs GEMMul8 fast (N=14 and 15) per case (bench/sweep_xduel.txt) -> xduel.jsonl; not in the default list
      "$here/run_sweep.sh" "$here/sweep_xduel.txt" "$outdir/xduel.jsonl" 2>&1 | tee "$outdir/xduel.out"
      st=${PIPESTATUS[0]}; [ "$st" = 0 ] || rc=$st
      ;;
    extra)
      # R5 additions (bench/sweep_r5.txt): same-job new-vs-old default and wgmma-vs-mma.sync duels, N=8 against GEMMul8, sustained
      # large squares, the 16384 square -> r5.jsonl; not in the default list
      "$here/run_sweep.sh" "$here/sweep_r5.txt" "$outdir/r5.jsonl" 2>&1 | tee "$outdir/r5.out"
      st=${PIPESTATUS[0]}; [ "$st" = 0 ] || rc=$st
      ;;
    profile)
      # short runs under nsys, summarized per kernel category (profile/index.json); a missing nsys is a note, not a failure.
      # It runs last in the default list (a profile changes nothing in the other results, but it is the longest to set up)
      "$here/profile_nsys.sh" "$outdir" 2>&1 | tee "$outdir/profile.out"
      st=${PIPESTATUS[0]}; [ "$st" = 0 ] || rc=$st
      ;;
    *) echo "unknown sweep '$sw' in SWEEPS (imd, baseline, cutws, compare, compare_energy, energy, xduel, extra, profile)" >&2; exit 2 ;;
  esac
done
if command -v python3 >/dev/null; then
  files=(); for f in sweep_imd.jsonl baselines.jsonl; do [ -s "$outdir/$f" ] && files+=("$outdir/$f"); done
  [ ${#files[@]} -gt 0 ] && python3 "$here/summarize.py" "${files[@]}" > "$outdir/summary.md" 2>/dev/null
  [ -s "$outdir/energy.jsonl" ] && python3 "$here/summarize.py" "$outdir/energy.jsonl" > "$outdir/energy.md" 2>/dev/null
  [ -s "$outdir/cutensor_ws.jsonl" ] && python3 "$here/cutws_table.py" "$outdir/cutensor_ws.jsonl" > "$outdir/cutensor_ws.md" 2>/dev/null
  [ -s "$outdir/compare.jsonl" ] && python3 "$here/summarize.py" --only-compare "$outdir/compare.jsonl" > "$outdir/compare.md" 2>/dev/null
  [ -s "$outdir/compare_energy.jsonl" ] && python3 "$here/summarize.py" --only-compare "$outdir/compare_energy.jsonl" > "$outdir/compare_energy.md" 2>/dev/null
fi
echo "finished: $(date -Is)" >> "$outdir/conditions.txt"
exit $rc
