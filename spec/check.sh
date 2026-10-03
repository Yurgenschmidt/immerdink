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
# its result), so it is NOT gated on a quiet machine by default. Knobs:
#   TLC_JOBS     models run concurrently (default: nproc/4, at least 1)
#   TLC_WORKERS  TLC worker threads per model (default 1; raise for big models)
#   CHECK_QUIET=1  serialize machine-wide and wait for tools/quiet_check.sh first
#                  (QUIET_WAIT seconds, default 900), as benchmarks do.
set -u
here="$(cd "$(dirname "$0")" && pwd)"; cd "$here"
JAR="${TLA2TOOLS:-$(ls -d "$HOME"/.vscode/extensions/tlaplus.vscode-ide-*/tools/tla2tools.jar 2>/dev/null | tail -1)}"
[ -f "$JAR" ] || { echo "tla2tools.jar not found; set TLA2TOOLS" >&2; exit 2; }

if [ "${CHECK_QUIET:-0}" = 1 ]; then
  exec 9>"${TLC_LOCK:-${XDG_RUNTIME_DIR:-/tmp}/immerdink-tlc.lock}"
  flock 9
  waited=0
  until ../tools/quiet_check.sh >/dev/null 2>&1; do
    if [ "$waited" -ge "${QUIET_WAIT:-900}" ]; then
      echo "machine not quiet after ${waited}s; not running TLC" >&2
      ../tools/quiet_check.sh >&2; exit 3
    fi
    [ "$waited" = 0 ] && echo "waiting for a quiet machine..." >&2
    sleep 15; waited=$((waited + 15))
  done
fi

shopt -s nullglob
cfgs=( MC_${1:-*}__${2:-*}.cfg )
[ ${#cfgs[@]} -gt 0 ] || { echo "no models match MC_${1:-*}__${2:-*}.cfg" >&2; exit 2; }
jobs_max="${TLC_JOBS:-$(( $(nproc) / 4 > 0 ? $(nproc) / 4 : 1 ))}"
tmp="$(mktemp -d)"; trap 'rm -rf "$tmp"' EXIT

run_one() {  # cfg -> writes $tmp/<base>.verdict
  local cfg="$1" base="${1%.cfg}" module kind dl out t0 dt expect
  module="${base%%__*}"; kind="${base#*__}"
  dl="-deadlock"; grep -q '^\\\* CHECK_DEADLOCK' "$cfg" && dl=""
  t0=$SECONDS
  out="$(java -XX:+UseParallelGC -cp "$JAR" tlc2.TLC -workers "${TLC_WORKERS:-1}" $dl \
         -noGenerateSpecTE -metadir "$tmp/meta-$base" -config "$cfg" "$module.tla" 2>&1)"
  dt=$((SECONDS - t0))
  case "$kind" in
    pos_*)
      if grep -q 'No error has been found' <<<"$out"; then
        echo "PASS  $base  ($(grep -m1 -o '[0-9,]* distinct states found' <<<"$out"), ${dt}s)"
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
