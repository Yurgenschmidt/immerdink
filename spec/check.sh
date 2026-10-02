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
# TLC runs are serialized machine-wide with a lock, and the batch only starts
# once tools/quiet_check.sh reports QUIET (waits up to QUIET_WAIT seconds,
# default 900). TLC_WORKERS (default 1) sets TLC worker threads.
set -u
here="$(cd "$(dirname "$0")" && pwd)"; cd "$here"
JAR="${TLA2TOOLS:-$(ls -d "$HOME"/.vscode/extensions/tlaplus.vscode-ide-*/tools/tla2tools.jar 2>/dev/null | tail -1)}"
[ -f "$JAR" ] || { echo "tla2tools.jar not found; set TLA2TOOLS" >&2; exit 2; }

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

pattern="MC_${1:-*}__${2:-*}.cfg"
meta="$(mktemp -d)"; trap 'rm -rf "$meta"' EXIT
fail=0; n=0
for cfg in $pattern; do
  [ -e "$cfg" ] || { echo "no models match $pattern" >&2; exit 2; }
  base="${cfg%.cfg}"; module="${base%%__*}"; kind="${base#*__}"
  dl="-deadlock"; grep -q '^\\\* CHECK_DEADLOCK' "$cfg" && dl=""
  t0=$SECONDS
  out="$(java -XX:+UseParallelGC -cp "$JAR" tlc2.TLC -workers "${TLC_WORKERS:-1}" $dl \
         -noGenerateSpecTE -metadir "$meta/$base" -config "$cfg" "$module.tla" 2>&1)"
  dt=$((SECONDS - t0)); n=$((n + 1))
  case "$kind" in
    pos_*)
      if grep -q 'No error has been found' <<<"$out"; then
        echo "PASS  $base  ($(grep -m1 -o '[0-9,]* distinct states found' <<<"$out"), ${dt}s)"
      else echo "FAIL  $base"; tail -30 <<<"$out"; fail=1; fi ;;
    neg_*)
      expect="$(sed -n 's/^\\\* EXPECT: //p' "$cfg")"
      if [ -z "$expect" ]; then echo "FAIL  $base  (no '\\* EXPECT:' line)"; fail=1
      elif grep -qF -- "$expect" <<<"$out"; then echo "PASS  $base  (expected: $expect, ${dt}s)"
      else echo "FAIL  $base  (expected: $expect)"; tail -30 <<<"$out"; fail=1; fi ;;
    *) echo "FAIL  $base  (name must be MC_<Module>__pos_* or MC_<Module>__neg_*)"; fail=1 ;;
  esac
done
echo "$n model(s), $([ $fail = 0 ] && echo all as expected || echo FAILURES)"
exit $fail
