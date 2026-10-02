#!/usr/bin/env bash
# Model-check every spec/MC_*.cfg against MC_OzakiCRT.tla.
#   MC_pos_*.cfg must pass; MC_neg_*.cfg must report a violation of the one
#   invariant they list. Run tools/quiet_check.sh first (TLC is CPU-bound).
set -u
cd "$(dirname "$0")"
JAR="${TLA2TOOLS:-$(ls -d "$HOME"/.vscode/extensions/tlaplus.vscode-ide-*/tools/tla2tools.jar 2>/dev/null | tail -1)}"
[ -f "$JAR" ] || { echo "tla2tools.jar not found; set TLA2TOOLS" >&2; exit 2; }
meta="$(mktemp -d)"; trap 'rm -rf "$meta"' EXIT
fail=0
for cfg in MC_*.cfg; do
  name="${cfg%.cfg}"
  out="$(java -XX:+UseParallelGC -cp "$JAR" tlc2.TLC -workers 1 -deadlock \
         -metadir "$meta/$name" -config "$cfg" MC_OzakiCRT.tla 2>&1)"
  inv="$(sed -n 's/^INVARIANTS //p' "$cfg")"
  case "$name" in
    MC_pos_*) if grep -q 'No error has been found' <<<"$out"; then
                echo "PASS  $name  ($(grep -m1 -o '[0-9,]* distinct states found' <<<"$out"))"
              else echo "FAIL  $name"; echo "$out" | tail -25; fail=1; fi ;;
    MC_neg_*) if grep -q "Invariant $inv is violated" <<<"$out"; then
                echo "PASS  $name  (violates $inv as expected)"
              else echo "FAIL  $name  (expected $inv violation)"; echo "$out" | tail -25; fail=1; fi ;;
  esac
done
exit $fail
