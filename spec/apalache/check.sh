#!/usr/bin/env bash
# Real-width symbolic checks with Apalache (unbounded SMT integers).
#   spec/apalache/check.sh              all checks
#   spec/apalache/check.sh crt16        checks whose name starts with "crt16"
#
# Each check is one Apalache run; a claim proved by an inductive invariant
# takes three runs (Init => IndInv; IndInv /\ Next => IndInv'; IndInv =>
# Safety). Expected-violation checks (like TLC's neg_* models) must report
# a counterexample; its last state is printed.
#
# Apalache: $APALACHE_MC, else tools/apalache/bin/apalache-mc in this checkout
# or in the main checkout of this repository (tools/get_apalache.sh installs
# it; it needs Java >= 21). Knobs: APA_JOBS (checks run concurrently,
# default 1), APA_TIMEOUT (seconds per check, default 600), JVM_ARGS (passed
# to apalache-mc's JVM; default -Xmx4g). Z3 runs in the same process and uses
# native memory on top of the heap; each query here is small (well under
# 1 GB). The whole run sits inside tools/memcap.sh's shared memory budget.
set -u
here="$(cd "$(dirname "$0")" && pwd)"; cd "$here"
. "$here/../../tools/budget.sh"
immerdink_in_budget || exec "$here/../../tools/memcap.sh" "$here/check.sh" "$@"
export JVM_ARGS="${JVM_ARGS:--Xmx4g}"
root="$(cd "$here/../.." && pwd)"

APA="${APALACHE_MC:-}"
if [ -z "$APA" ]; then
  for c in "$root/tools/apalache/bin/apalache-mc" \
           "$(git -C "$root" rev-parse --path-format=absolute --git-common-dir 2>/dev/null | sed 's|/\.git$||')/tools/apalache/bin/apalache-mc"; do
    [ -x "$c" ] && { APA="$c"; break; }
  done
fi
[ -n "$APA" ] && [ -x "$APA" ] || { echo "apalache-mc not found; run tools/get_apalache.sh or set APALACHE_MC" >&2; exit 2; }

# The CRT modules are generated from spec/params_out.json; refuse stale ones.
python3 "$here/gen_crtword.py" --check || exit 1

# name | module | cinit | init | next | inv | length | expect (NoError / Error)
CHECKS='
int32_kc_init    |Int32Chunk|ConstInitKc     |Init  |Next     |IndInv        |0 |NoError
int32_kc_step    |Int32Chunk|ConstInitKc     |IndInv|Next     |IndInv        |1 |NoError
int32_kc_safe    |Int32Chunk|ConstInitKc     |IndInv|Next     |Safety        |0 |NoError
int32_kc1_unsafe |Int32Chunk|ConstInitKcPlus1|IndInv|Next     |Safety        |0 |Error
epi_init         |Epilogue  |ConstInit       |Init  |Next     |IndInv        |0 |NoError
epi_step         |Epilogue  |ConstInit       |IndInv|Next     |IndInv        |1 |NoError
epi_safe         |Epilogue  |ConstInit       |IndInv|Next     |Safety        |0 |NoError
epi_max_attained |Epilogue  |ConstInit       |IndInv|Next     |VMaxTight     |0 |Error
crt16_init       |CRTWord16 |                |Init  |Next     |IndInv        |0 |NoError
crt16_step       |CRTWord16 |                |IndInv|Next     |IndInv        |1 |NoError
crt16_safe       |CRTWord16 |                |IndInv|Next     |Safety        |0 |NoError
crt16_peak_max   |CRTWord16 |                |Init  |Next     |PeakAtMostCert|16|NoError
crt16_peak_hit   |CRTWord16 |                |Init  |Next     |PeakBelowCert |16|Error
crt16_nosub      |CRTWord16 |                |Init  |NextNoSub|XBelowP       |16|Error
crt20_init       |CRTWord20 |                |Init  |Next     |IndInv        |0 |NoError
crt20_step       |CRTWord20 |                |IndInv|Next     |IndInv        |1 |NoError
crt20_safe       |CRTWord20 |                |IndInv|Next     |Safety        |0 |NoError
crt20_peak_max   |CRTWord20 |                |Init  |Next     |PeakAtMostCert|20|NoError
crt20_peak_hit   |CRTWord20 |                |Init  |Next     |PeakBelowCert |20|Error
crt20_nosub      |CRTWord20 |                |Init  |NextNoSub|XBelowP       |20|Error
'

tmp="$(mktemp -d)"; trap 'rm -rf "$tmp"' EXIT
jobs_max="${APA_JOBS:-1}"

run_one() {  # fields of one CHECKS line -> $tmp/<name>.verdict
  local name="$1" mod="$2" cinit="$3" init="$4" next="$5" inv="$6" len="$7" expect="$8"
  local args=(check "--out-dir=$tmp/out-$name" "--init=$init" "--next=$next" "--inv=$inv" "--length=$len")
  [ -n "$cinit" ] && args+=("--cinit=$cinit")
  local t0=$SECONDS out rc dt outcome
  out="$(cd "$tmp" && timeout "${APA_TIMEOUT:-600}" "$APA" "${args[@]}" "$here/$mod.tla" 2>&1)"; rc=$?
  dt=$((SECONDS - t0))
  outcome="$(sed -n 's/^The outcome is: \([A-Za-z]*\).*/\1/p' <<<"$out" | tail -1)"
  {
    if [ "$rc" = 124 ]; then
      echo "FAIL  $name  (TIMEOUT after ${dt}s)"
    elif [ "$outcome" = "$expect" ]; then
      if [ "$expect" = Error ]; then
        local cex; cex="$(find "$tmp/out-$name" -name violation1.tla | head -1)"
        echo "PASS  $name  ($mod: $inv violated as expected, ${dt}s)"
        # the last state and the violated formula, each joined onto one line
        [ -n "$cex" ] && awk '
          /^(State[0-9]+|InvariantViolation) ==/ { if (b != "") blk[++n] = b; b = $0; next }
          /^$/ || /^====/ || /^\(\*/ { if (b != "") { blk[++n] = b; b = "" } next }
          b != "" { sub(/^ +/, " "); b = b $0 }
          END { if (b != "") blk[++n] = b; for (i = n - 1; i <= n; i++) if (i > 0) print "        " substr(blk[i], 1, 200) }
        ' "$cex"
      else
        echo "PASS  $name  ($mod: $init -> $inv, length $len, ${dt}s)"
      fi
    else
      echo "FAIL  $name  (expected $expect, got '${outcome:-none}', rc=$rc)"
      tail -20 <<<"$out"
    fi
  } > "$tmp/$name.verdict"
}

names=()
while IFS='|' read -r name mod cinit init next inv len expect; do
  name="$(echo $name)"; [ -z "$name" ] && continue
  case "$name" in "${1:-}"*) ;; *) continue ;; esac
  names+=("$name")
  while [ "$(jobs -rp | wc -l)" -ge "$jobs_max" ]; do wait -n; done
  run_one "$name" $mod "$(echo $cinit)" $init $next $inv $len $expect &
done <<<"$CHECKS"
wait

[ ${#names[@]} -gt 0 ] || { echo "no checks match '${1:-}'" >&2; exit 2; }
fail=0
for n in "${names[@]}"; do
  cat "$tmp/$n.verdict"
  grep -q '^FAIL' "$tmp/$n.verdict" && fail=1
done
echo "${#names[@]} Apalache check(s), $([ $fail = 0 ] && echo all as expected || echo FAILURES)"
exit $fail
