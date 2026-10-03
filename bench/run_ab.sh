#!/usr/bin/env bash
# Same-session A/B of two imd_bench builds, interleaved per case (the A1/A2/A3 comparisons):
# for every case of a sweep file, the "before" binary runs the case's methods (e.g. gemmul8 and
# imd), then the "after" binary runs only imd, each as its own quiet-gated process
# (tools/run_bench.sh: machine-wide lock, quiet machine, GPU lock).
#   bench/run_ab.sh <before imd_bench> <after imd_bench> <sweep file> <before.jsonl> <after.jsonl>
# Extra imd_bench arguments for the after binary only: $AFTER_ARGS (e.g. "--k4 rule,tune").
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
while IFS='|' read -r name spec ext methods Ns emu extra; do
  name="$(echo "$name" | xargs)"
  case "$name" in ''|'#'*) continue ;; esac
  if [ -n "${CASES:-}" ] && ! echo "$name" | grep -Eq "$CASES"; then continue; fi
  common=(--name "$name" --spec "$(echo "$spec" | xargs)" --ext "$(echo "$ext" | xargs)")
  [ -n "$(echo "$Ns" | xargs)" ] && common+=(--N "$(echo "$Ns" | xargs)")
  [ -n "$(echo "$emu" | xargs)" ] && common+=(--emu "$(echo "$emu" | xargs)")
  # shellcheck disable=SC2206
  common+=($extra)
  echo "== $name $(date -Is)"
  for side in before after; do
    if [ "$side" = before ]; then
      bin="$before"; out="$out_b"; m="$(echo "$methods" | xargs)"; more=()
    else
      bin="$after"; out="$out_a"; m=imd
      # shellcheck disable=SC2206
      more=(${AFTER_ARGS:-})
    fi
    tag="BEFORE_COMMIT"; [ "$side" = after ] && tag="AFTER_COMMIT"
    BENCH_COMMIT="${!tag:-$side}" "$root/tools/run_bench.sh" "$bin" "${common[@]}" --methods "$m" \
      "${more[@]}" --out "$out" </dev/null
    rc=$?
    if [ $rc = 3 ]; then echo "machine never got quiet; stopping before $name ($side)" >&2; exit 3; fi
    [ $rc != 0 ] && echo "case $name ($side) failed (exit $rc); continuing" >&2
  done
done < "$sweep"
echo "done: $out_b $out_a"
