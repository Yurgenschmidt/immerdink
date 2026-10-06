#!/usr/bin/env bash
# Same-session A/B of two imd_bench builds (before / after a change to immerdink),
# interleaved per case: for every case of a sweep file, the "before" binary runs the case,
# then the "after" binary runs it, each as its own quiet-gated process (tools/run_bench.sh:
# machine-wide lock, quiet machine, GPU lock; bench/warm_run.sh: warm-up, retries).
#   bench/run_ab.sh <before imd_bench> <after imd_bench> <sweep file> <before.jsonl> <after.jsonl>
# Methods: the case's (sweep file column 4) on both sides; AFTER_METHODS overrides them for the
# after binary only. Extra imd_bench arguments for the after binary only: $AFTER_ARGS
# (e.g. "--k4 rule,tune"). Summarize with bench/ab_ratio.py before.jsonl after.jsonl.
# Tiny calls (<~100 us) move +-10-20 % between per-process performance states (NOTES-A5
# §1.2), so a cross-process A/B there needs many timed calls (e.g. --reps 300 in the sweep's
# extra args) and repeated runs; within one build, compare configurations with
# `imd_bench --methods duel` instead.
# CASES (grep -E regex) selects cases; BEFORE_COMMIT / AFTER_COMMIT label the records.
# Stops (exit 3) if the machine never gets quiet.
set -uo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
root="$(dirname "$here")"
before="$1"; after="$2"; sweep="$3"; out_b="$4"; out_a="$5"
env_dir="${IMMERDINK_ENV:-$HOME/miniforge3/envs/immerdink}"
export QUIET_GPU_IGNORE="${QUIET_GPU_IGNORE:-gnome-remote-desktop-daemon|Xorg|gnome-shell|claude-desktop --type=gpu-process}"
export LD_LIBRARY_PATH="$env_dir/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
mkdir -p "$(dirname "$out_b")" "$(dirname "$out_a")"
while IFS='|' read -r name spec ext methods Ns extra; do
  name="$(echo "$name" | xargs)"
  case "$name" in ''|'#'*) continue ;; esac
  if [ -n "${CASES:-}" ] && ! echo "$name" | grep -Eq "$CASES"; then continue; fi
  common=(--name "$name" --spec "$(echo "$spec" | xargs)" --ext "$(echo "$ext" | xargs)")
  [ -n "$(echo "$Ns" | xargs)" ] && common+=(--N "$(echo "$Ns" | xargs)")
  # shellcheck disable=SC2206
  common+=($extra)
  echo "== $name $(date -Is)"
  for side in before after; do
    if [ "$side" = before ]; then
      bin="$before"; out="$out_b"; m="$(echo "$methods" | xargs)"; more=()
    else
      bin="$after"; out="$out_a"; m="${AFTER_METHODS:-$(echo "$methods" | xargs)}"
      # shellcheck disable=SC2206
      more=(${AFTER_ARGS:-})
    fi
    tag="BEFORE_COMMIT"; [ "$side" = after ] && tag="AFTER_COMMIT"
    BENCH_COMMIT="${!tag:-$side}" "$root/tools/run_bench.sh" "$root/bench/warm_run.sh" "$bin" "${common[@]}" --methods "$m" \
      "${more[@]}" --out "$out" </dev/null
    rc=$?
    if [ $rc = 3 ]; then echo "machine never got quiet; stopping before $name ($side)" >&2; exit 3; fi
    [ $rc != 0 ] && echo "case $name ($side) failed (exit $rc); continuing" >&2
  done
done < "$sweep"
echo "done: $out_b $out_a"
