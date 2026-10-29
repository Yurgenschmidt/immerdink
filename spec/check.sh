#!/usr/bin/env bash
# Model-check the TLA+ specs.
#   spec/check.sh                        all models
#   spec/check.sh OzakiScaling           all models of one module
#   spec/check.sh OzakiCRT pos_3moduli   one model
#
# Conventions (flat directory):
#   <Module>.tla              the spec
#   MC_<Module>.tla           model values (EXTENDS <Module>)
#   MC_<Module>__pos_*.cfg    must pass: "No error has been found"
#   MC_<Module>__neg_*.cfg    must fail; the cfg carries a line
#                               \* EXPECT: <text>
#                             and TLC's output must contain <text> verbatim
#                             (e.g. "Invariant CRTExact is violated").
#   A cfg line "\* CHECK_DEADLOCK" turns deadlock checking on (off by default,
#   because the constant-level specs stutter forever by design).
#
# Model checking is a pass/fail correctness check (CPU contention cannot change
# its result), so it is never gated on a quiet machine (user decision,
# 2026-10-02; CHECK_QUIET is ignored). Knobs:
#   TLC_JOBS     models run concurrently (default: nproc/4, at least 1). Clamped
#                to what fits in memory: each job is budgeted 2*TLC_HEAP + 512 MiB
#                (heap + off-heap fingerprint set) against 80% of the smaller of
#                the immerdink memory cap and MemAvailable.
#   TLC_HEAP     JVM heap per model, MiB or with g/m suffix (default 2g); the
#                off-heap fingerprint set gets the same. Raise for a big model.
#   TLC_WORKERS  TLC worker threads per model (default 1; raise for big models)
#   TLC_TIMEOUT  seconds per model before it is reported as TIMEOUT (default 900)
# The whole run sits inside tools/memcap.sh's shared memory budget. Without a
# heap cap each JVM defaulted to 1/4 of RAM, and TLC_JOBS=16 crashed the machine
# (2026-10-03).
set -u
here="$(cd "$(dirname "$0")" && pwd)"; cd "$here"
. "$here/../tools/budget.sh"
immerdink_in_budget || exec "$here/../tools/memcap.sh" "$here/check.sh" "$@"
JAR="${TLA2TOOLS:-$(ls -d "$HOME"/.vscode/extensions/tlaplus.vscode-ide-*/tools/tla2tools.jar 2>/dev/null | tail -1)}"
[ -f "$JAR" ] || { echo "tla2tools.jar not found; set TLA2TOOLS" >&2; exit 2; }

shopt -s nullglob
cfgs=( MC_${1:-*}__${2:-*}.cfg )
[ ${#cfgs[@]} -gt 0 ] || { echo "no models match MC_${1:-*}__${2:-*}.cfg" >&2; exit 2; }
jobs_max="${TLC_JOBS:-$(( $(nproc) / 4 > 0 ? $(nproc) / 4 : 1 ))}"
heap="${TLC_HEAP:-2g}"
case "$heap" in *[gG]) heap_mb=$(( ${heap%[gG]} * 1024 ));; *[mM]) heap_mb=${heap%[mM]};; *) heap_mb=$heap;; esac
avail_mb=$(( $(awk '/^MemAvailable:/{print $2}' /proc/meminfo) / 1024 ))
cap_file="/sys/fs/cgroup$(cut -d: -f3 /proc/self/cgroup | xargs dirname)/memory.max"
cap_raw="$(cat "$cap_file" 2>/dev/null || echo max)"
[ "$cap_raw" != max ] && [ $(( cap_raw / 1048576 )) -lt "$avail_mb" ] && avail_mb=$(( cap_raw / 1048576 ))
mem_jobs=$(( avail_mb * 8 / 10 / (2 * heap_mb + 512) )); [ "$mem_jobs" -ge 1 ] || mem_jobs=1
if [ "$jobs_max" -gt "$mem_jobs" ]; then
  echo "check.sh: TLC_JOBS=$jobs_max does not fit in ${avail_mb} MiB at TLC_HEAP=$heap; using $mem_jobs" >&2
  jobs_max=$mem_jobs
fi
tmp="$(mktemp -d)"; trap 'rm -rf "$tmp"' EXIT

run_one() {  # cfg -> writes $tmp/<base>.verdict
  local cfg="$1" base="${1%.cfg}" module kind dl out t0 dt expect
  module="${base%%__*}"; kind="${base#*__}"
  dl="-deadlock"; grep -q '^\\\* CHECK_DEADLOCK' "$cfg" && dl=""
  t0=$SECONDS
  out="$(timeout "${TLC_TIMEOUT:-900}" java -Xmx"${heap_mb}m" -XX:MaxDirectMemorySize="${heap_mb}m" -XX:+UseParallelGC -XX:ParallelGCThreads=2 -cp "$JAR" tlc2.TLC -workers "${TLC_WORKERS:-1}" $dl \
         -noGenerateSpecTE -metadir "$tmp/meta-$base" -config "$cfg" "$module.tla" 2>&1)"
  local rc=$?
  dt=$((SECONDS - t0))
  if [ "$rc" = 124 ]; then echo "FAIL  $base  (TIMEOUT after ${dt}s)" > "$tmp/$base.verdict"; return; fi
  case "$kind" in
    pos_*)
      if grep -q 'No error has been found' <<<"$out"; then
        echo "PASS  $base  ($(grep -o '[0-9,]* distinct states found, 0 states left' <<<"$out" | tail -1 | sed 's/, 0 states left//'), ${dt}s)"
      else echo "FAIL  $base"; tail -30 <<<"$out"; fi ;;
    neg_*)
      expect="$(sed -n 's/^\\\* EXPECT: //p' "$cfg")"
      if [ -z "$expect" ]; then echo "FAIL  $base  (no '\\* EXPECT:' line)"
      elif grep -qF -- "$expect" <<<"$out"; then echo "PASS  $base  (expected: $expect, ${dt}s)"
      else echo "FAIL  $base  (expected: $expect)"; tail -30 <<<"$out"; fi ;;
    *) echo "FAIL  $base  (name must be MC_<Module>__pos_* or MC_<Module>__neg_*)" ;;
  esac > "$tmp/$base.verdict"
}

for cfg in "${cfgs[@]}"; do
  while [ "$(jobs -rp | wc -l)" -ge "$jobs_max" ]; do wait -n; done
  run_one "$cfg" &
done
wait

fail=0
for cfg in "${cfgs[@]}"; do
  v="$tmp/${cfg%.cfg}.verdict"
  cat "$v"
  grep -q '^FAIL' "$v" && fail=1
done
echo "${#cfgs[@]} model(s), $([ $fail = 0 ] && echo all as expected || echo FAILURES)"
exit $fail
