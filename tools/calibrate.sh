#!/usr/bin/env bash
# Calibrate the performance model (DESIGN v1.8 §6c.2) on this machine's GPU.
#   tools/calibrate.sh [-o out.calib] [--validate] [--results DIR] [--k-align auto|16|64]
#                      [--sustained FILE] [--cache FILE] [--sustained-only]
# 1. builds the profiling imd_bench (bench/build.sh, via bench/run_sweep.sh);
# 2. runs the calibration sweep bench/sweep_calib.txt through bench/run_sweep.sh (quiet gate,
#    shared GPU lock, warm-up; ~12 min of GPU time on an RTX 4090) into DIR/calib-<stamp>.jsonl;
# 3. fits the parameters (bench/perf_fit.py, numpy) and writes the calibration file
#    (default: DIR/<gpu>.calib); load it with imd_perf_calib_load / host.Calibration.load, or
#    copy it over src/host/calib/rtx4090_sm89.calib to change the shipped one;
# 4. with --validate: runs the hold-out sweep bench/sweep_perf_validate.txt and prints the
#    error distribution against the §6c.5 targets (bench/perf_validate.py) and writes the
#    hold-out prediction bands into the calibration (v1.10).
# v1.10 also runs bench/sweep_sustained.txt (power model, ~4 min) and bench/sweep_cache.txt
# (COLD / WARM / REPEAT, ~1 min) and records the SM clock of every timed call.
# Environment: as bench/run_sweep.sh (IMMERDINK_ENV, IMMERDINK_CUTLASS_DIR, BENCH_RETRIES, ...);
# PERF_FIT_ARGS: extra bench/perf_fit.py arguments (a GPU other than the RTX 4090 needs its
# datasheet peaks: PERF_FIT_ARGS="--peak-dram 1.0e12 --peak-int8 6.6e14 --peak-fp64 1.3e12").
# --k-align (DESIGN v1.13 §4b.6): force the K alignment of every plan of the sweeps (imd_bench
# IMD_BENCH_K_ALIGN; default auto). A single sweep line can force it with "--k-align 16" too
# (the A = 16-forced cases at k > 96 the refit needs); every record carries k_align_hook and
# bench/perf_fit.py models each record at its alignment.
# --sustained / --cache: reuse the records of an earlier bench/sweep_sustained.txt /
# bench/sweep_cache.txt run instead of measuring them again (~13 min of GPU). Only valid when
# those sweeps' plans are unchanged since (v1.13 stage 2: every case there has k = 0 mod 64,
# so A = 16 and A = 64 give the same k_hat).
# --sustained-only (v1.15, DESIGN §6c.7): run only bench/sweep_sustained.txt (60 s pre-heat
# windows, ~17 min of GPU), print the records' path and exit (no fit; reuse them later with
# --sustained FILE).
# Results default to bench/results/ (git-ignored). Never changes GPU clocks.
set -uo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
root="$(dirname "$here")"
env_dir="${IMMERDINK_ENV:-$HOME/miniforge3/envs/immerdink}"
build="${IMMERDINK_BENCH_BUILD:-$root/build/bench}"
out="" validate=0 results="$root/bench/results" sdata="" cdata="" sonly=0
while [ $# -gt 0 ]; do
  case "$1" in
    -o) out="$2"; shift 2 ;;
    --validate) validate=1; shift ;;
    --results) results="$2"; shift 2 ;;
    --sustained) sdata="$2"; shift 2 ;;
    --cache) cdata="$2"; shift 2 ;;
    --sustained-only) sonly=1; shift ;;
    --k-align) case "$2" in auto|0|16|64) export IMD_BENCH_K_ALIGN="$2" ;;
               *) echo "--k-align auto|16|64" >&2; exit 2 ;; esac; shift 2 ;;
    *) echo "usage: tools/calibrate.sh [-o out.calib] [--validate] [--results DIR] [--k-align auto|16|64] [--sustained FILE] [--cache FILE] [--sustained-only]" >&2; exit 2 ;;
  esac
done
for f in "$sdata" "$cdata"; do
  [ -z "$f" ] || [ -s "$f" ] || { echo "calibrate: no records in $f" >&2; exit 2; }
done
mkdir -p "$results"
stamp="$(date +%Y%m%d-%H%M)"
export BENCH_RETRIES="${BENCH_RETRIES:-2}"
py="$env_dir/bin/python3"
hostlib="$build/immerdink/libimmerdink_host.so"

# v1.10: every timed record carries the SM clock of its calls (the per-call clock probe).
export IMD_BENCH_CLOCK=1
if [ "$sonly" = 1 ]; then
  [ -z "$sdata" ] || { echo "calibrate: --sustained-only measures; --sustained FILE reuses" >&2; exit 2; }
  sdata="$results/sustained-$stamp.jsonl"
  "$root/bench/run_sweep.sh" "$root/bench/sweep_sustained.txt" "$sdata" || exit $?
  echo "sustained records: $sdata"
  exit 0
fi
data="$results/calib-$stamp.jsonl"
"$root/bench/run_sweep.sh" "$root/bench/sweep_calib.txt" "$data" || exit $?
# v1.10 (DESIGN §6c.1): sustained windows (power model) and the cache-state pairs.
if [ -z "$sdata" ]; then
  sdata="$results/sustained-$stamp.jsonl"
  "$root/bench/run_sweep.sh" "$root/bench/sweep_sustained.txt" "$sdata" || exit $?
fi
if [ -z "$cdata" ]; then
  cdata="$results/cache-$stamp.jsonl"
  "$root/bench/run_sweep.sh" "$root/bench/sweep_cache.txt" "$cdata" || exit $?
fi
[ -n "$out" ] || out="$results/$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1 |
                                 tr ' ' '_' | tr -cd 'A-Za-z0-9_-')-$stamp.calib"
# shellcheck disable=SC2086
"$py" "$root/bench/perf_fit.py" "$data" "$sdata" "$cdata" -o "$out" --lib "$hostlib" ${PERF_FIT_ARGS:-} || exit $?
echo "calibration: $out"
if [ "$validate" = 1 ]; then
  vdata="$results/validate-$stamp.jsonl"
  "$root/bench/run_sweep.sh" "$root/bench/sweep_perf_validate.txt" "$vdata" || exit $?
  # v1.10: the prediction bands come from the hold-out residuals and are written into the
  # calibration (format 2); the coverage printed is 2-fold cross-validated over the cases.
  "$py" "$root/bench/perf_validate.py" "$vdata" --calib "$out" --lib "$hostlib" --phases \
    --bands "$out" --sustained "$sdata"
fi
