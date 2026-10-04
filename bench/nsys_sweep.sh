#!/usr/bin/env bash
# nsys profiles of every case of a sweep file (A5), one process per (case, N), with the
# case's methods; then bench/nsys_calls.py on each. Each profile runs through
# bench/nsys_case.sh (quiet gate, GPU lock).
#   bench/nsys_sweep.sh <imd_bench> <sweep file> <out dir> [extra imd_bench args...]
# N list: $NS (default: the case's). CASES (grep -E regex) selects cases.
set -uo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
bench="$1"; sweep="$2"; out="$3"; shift 3
mkdir -p "$out"
while IFS='|' read -r name spec ext methods Ns _emu extra; do
  name="$(echo "$name" | xargs)"
  case "$name" in ''|'#'*) continue ;; esac
  if [ -n "${CASES:-}" ] && ! echo "$name" | grep -Eq "$CASES"; then continue; fi
  for N in $(echo "${NS:-$Ns}" | tr ',' ' '); do
    # shellcheck disable=SC2086
    "$here/nsys_case.sh" "$bench" "$out/$name-N$N" "$name" "$(echo "$spec" | xargs)" \
      "$(echo "$ext" | xargs)" --methods "$(echo "$methods" | xargs)" --N "$N" $extra "$@" \
      > "$out/$name-N$N.log" 2>&1 </dev/null
    echo "== $name N=$N"
    python3 "$here/nsys_calls.py" "$out/$name-N$N.sqlite" | tee "$out/$name-N$N.txt"
  done
done < "$sweep"
