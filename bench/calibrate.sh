#!/usr/bin/env bash
# Calibrate the GPU before measuring: a sustained dense load until the SM clock and temperature are
# stable, then the steady-state clock/power/temperature and a recommended clock lock.
#   bench/calibrate.sh [--max-seconds 180] [--out DIR]
# Writes DIR/calibration.json (default DIR: $TMPDIR or /tmp/imd-calibration) and prints a summary.
# Goes through tools/run_bench.sh like every timed run (machine-wide lock, quiet gate, shared GPU
# lock; exit 3 if the machine never gets quiet). Builds imd_bench first if needed (bench/build.sh).
# Needs python3. It never changes clocks; the summary prints the lock command to run (needs root).
# Exit: 0 done (also when not stable: read the summary), 1 error, 3 machine never quiet.
set -uo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
root="$(dirname "$here")"
max=180; out="${TMPDIR:-/tmp}/imd-calibration"
while [ $# -gt 0 ]; do
  case "$1" in
    --max-seconds) max="$2"; shift 2 ;;
    --out) out="$2"; shift 2 ;;
    -h|--help) sed -n '2,10p' "$0"; exit 0 ;;
    *) echo "usage: $0 [--max-seconds N] [--out DIR]" >&2; exit 2 ;;
  esac
done
command -v python3 >/dev/null || { echo "calibrate: python3 needed" >&2; exit 1; }
build="${IMMERDINK_BENCH_BUILD:-$root/build/bench}"
env_dir="${IMMERDINK_ENV:-$HOME/miniforge3/envs/immerdink}"
[ -d "$env_dir/lib" ] && export LD_LIBRARY_PATH="$env_dir/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export QUIET_GPU_IGNORE="${QUIET_GPU_IGNORE:-gnome-remote-desktop-daemon|Xorg|gnome-shell|claude-desktop --type=gpu-process}"
[ -x "$build/imd_bench" ] || "$here/build.sh" >&2 || exit 1
exec python3 "$here/calibrate.py" --bench "$build/imd_bench" --runner "$root/tools/run_bench.sh" \
  --out "$out" --max-seconds "$max"
