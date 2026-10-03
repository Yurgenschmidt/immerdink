#!/usr/bin/env bash
# Run a benchmark sweep: every case is one imd_bench process, run through
# tools/run_bench.sh (machine-wide lock + wait for a quiet machine + QUIET_LOG).
#   bench/run_sweep.sh [sweep file] [results.jsonl]
# Defaults: bench/sweep_m4.txt, bench/results/<date>-<commit>.jsonl. Builds first
# (bench/build.sh). Only cases whose name matches $CASES (a grep -E regex) run, if set.
# If the machine never gets quiet (run_bench.sh exit 3), the sweep stops there and says so;
# it never bypasses the gate.
# Environment: IMMERDINK_ENV, IMMERDINK_BENCH_BUILD, plus what bench/build.sh and
# tools/run_bench.sh read (QUIET_WAIT, QUIET_LOAD, QUIET_CPU, QUIET_GPU, QUIET_GPU_IGNORE).
set -uo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
root="$(dirname "$here")"
env_dir="${IMMERDINK_ENV:-$HOME/miniforge3/envs/immerdink}"
build="${IMMERDINK_BENCH_BUILD:-$root/build/bench}"
sweep="${1:-$here/sweep_m4.txt}"
commit="$(git -C "$root" rev-parse --short HEAD)"
git -C "$root" diff --quiet HEAD -- src include bench CMakeLists.txt || commit="$commit-dirty"
out="${2:-$here/results/$(date +%Y%m%d-%H%M)-$commit.jsonl}"
mkdir -p "$(dirname "$out")"

# The Claude desktop app keeps a GPU *renderer* process resident (like gnome-shell); it is
# not a compute load. GPU utilization is still gated by QUIET_GPU.
export QUIET_GPU_IGNORE="${QUIET_GPU_IGNORE:-gnome-remote-desktop-daemon|Xorg|gnome-shell|claude-desktop --type=gpu-process}"
export BENCH_COMMIT="$commit" BENCH_ENV="$(basename "$env_dir")"
export LD_LIBRARY_PATH="$env_dir/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

"$here/build.sh" || exit 1
logdir="$(mktemp -d "${TMPDIR:-/tmp}/imd-bench.XXXXXX")"
echo "sweep $sweep -> $out (commit $commit, env $BENCH_ENV)"

while IFS='|' read -r name spec ext methods Ns emu extra; do
  name="$(echo "$name" | xargs)"
  case "$name" in ''|'#'*) continue ;; esac
  if [ -n "${CASES:-}" ] && ! echo "$name" | grep -Eq "$CASES"; then continue; fi
  args=(--name "$name" --spec "$(echo "$spec" | xargs)" --ext "$(echo "$ext" | xargs)"
        --methods "$(echo "$methods" | xargs)" --out "$out")
  [ -n "$(echo "$Ns" | xargs)" ] && args+=(--N "$(echo "$Ns" | xargs)")
  [ -n "$(echo "$emu" | xargs)" ] && args+=(--emu "$(echo "$emu" | xargs)")
  # shellcheck disable=SC2206
  args+=($extra)
  echo "== $name $(date -Is)"
  QUIET_LOG="$logdir/$name.quiet" "$root/tools/run_bench.sh" "$here/warm_run.sh" "$build/imd_bench" "${args[@]}" </dev/null
  rc=$?
  if [ $rc = 3 ]; then
    echo "machine never got quiet; stopping the sweep before case $name" >&2
    printf '{"case": "%s", "status": "not run: machine not quiet (tools/run_bench.sh exit 3)", "commit": "%s"}\n' \
      "$name" "$commit" >> "$out"
    exit 3
  elif [ $rc != 0 ]; then
    echo "case $name failed (exit $rc); continuing" >&2
  fi
done < "$sweep"
echo "done: $out"
