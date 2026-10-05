#!/usr/bin/env bash
# Calibrate the performance model (DESIGN v1.8 §6c.2) on this machine's GPU.
#   tools/calibrate.sh [-o out.calib] [--validate] [--results DIR]
# 1. builds the profiling imd_bench (bench/build.sh, via bench/run_sweep.sh);
# 2. runs the calibration sweep bench/sweep_calib.txt through bench/run_sweep.sh (quiet gate,
#    shared GPU lock, warm-up; ~10 min of GPU time on an RTX 4090) into DIR/calib-<stamp>.jsonl;
# 3. fits the parameters (bench/perf_fit.py, numpy) and writes the calibration file
#    (default: DIR/<gpu>.calib); load it with imd_perf_calib_load / host.Calibration.load, or
#    copy it over src/host/calib/rtx4090_sm89.calib to change the shipped one;
# 4. with --validate: runs the hold-out sweep bench/sweep_perf_validate.txt and prints the
#    error distribution against the §6c.5 targets (bench/perf_validate.py).
# Environment: as bench/run_sweep.sh (IMMERDINK_ENV, IMMERDINK_CUTLASS_DIR, BENCH_RETRIES, ...).
# Results default to bench/results/ (git-ignored). Never changes GPU clocks.
set -uo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
root="$(dirname "$here")"
env_dir="${IMMERDINK_ENV:-$HOME/miniforge3/envs/immerdink}"
build="${IMMERDINK_BENCH_BUILD:-$root/build/bench}"
out="" validate=0 results="$root/bench/results"
while [ $# -gt 0 ]; do
  case "$1" in
    -o) out="$2"; shift 2 ;;
    --validate) validate=1; shift ;;
    --results) results="$2"; shift 2 ;;
    *) echo "usage: tools/calibrate.sh [-o out.calib] [--validate] [--results DIR]" >&2; exit 2 ;;
  esac
done
mkdir -p "$results"
stamp="$(date +%Y%m%d-%H%M)"
export BENCH_RETRIES="${BENCH_RETRIES:-2}"
py="$env_dir/bin/python3"
hostlib="$build/immerdink/libimmerdink_host.so"

data="$results/calib-$stamp.jsonl"
"$root/bench/run_sweep.sh" "$root/bench/sweep_calib.txt" "$data" || exit $?
[ -n "$out" ] || out="$results/$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1 |
                                 tr ' ' '_' | tr -cd 'A-Za-z0-9_-')-$stamp.calib"
"$py" "$root/bench/perf_fit.py" "$data" -o "$out" --lib "$hostlib" || exit $?
echo "calibration: $out"
if [ "$validate" = 1 ]; then
  vdata="$results/validate-$stamp.jsonl"
  "$root/bench/run_sweep.sh" "$root/bench/sweep_perf_validate.txt" "$vdata" || exit $?
  "$py" "$root/bench/perf_validate.py" "$vdata" --calib "$out" --lib "$hostlib" --phases
fi
