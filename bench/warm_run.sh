#!/usr/bin/env bash
# Run one benchmark case after a fixed GPU warm-up, then record the clock state.
#   bench/warm_run.sh <imd_bench> <case args...>     (called by run_sweep.sh inside run_bench.sh,
#                                                      so warm-up and case share one GPU-lock hold)
# GPU Boost picks the clock from power, temperature and recent load, so a process's clock
# depends on what ran before it (2535 vs 2820 MHz seen on the same case, 2026-10-03). A fixed
# warm-up (a dense int8-heavy imd call for BENCH_WARMUP_S seconds, default 3; 0 disables) puts
# every case in a similar power/thermal state. Afterwards one JSON line {"case", "clock_state"}
# is appended to the case's --out file: SM clock, power, temperature and the active clock-event
# reasons (applications_clocks_setting is how a `nvidia-smi -lgc` lock shows up; sw_power_cap
# is the 450 W limit). Summaries filter by "method", so these lines are ignored there.
set -u
bench="$1"; shift
out=""; name=""
args=("$@")
for ((i = 0; i < ${#args[@]}; i++)); do
  case "${args[$i]}" in --out) out="${args[$((i + 1))]}" ;; --name) name="${args[$((i + 1))]}" ;; esac
done

warm="${BENCH_WARMUP_S:-3}"
if [ "$warm" != 0 ]; then
  "$bench" --name warmup --spec 'ik,jk->ij' --ext i=4096,j=4096,k=4096 --methods imd --N 14 \
    --reps 1000 --warmup 1 --max-seconds "$warm" --phase-reps 0 --out /dev/null >/dev/null 2>&1 || true
fi

# Desktop renderers (the Claude app, gnome-shell) time-slice the GPU in ~2.2 ms slices when
# they have work queued, costing a running kernel 0.4-0.5 ms per switch (docs/NOTES-K5-bimodal.md).
# A clean run's timed-loop spread is small (<= ~2.3 % on medium/large shapes); a contaminated
# one is >= ~8.7 %. So each case runs into a scratch file and is retried (up to BENCH_RETRIES,
# default 3) while more than BENCH_MAX_BAD_FRAC (default 0.1) of its records have a spread_pct above
# BENCH_MAX_SPREAD (default 5): one noisy record among the 24 to 51 configurations of a compare case
# does not re-run the whole case. The attempt kept has the most status-ok records (a crashed attempt
# is a prefix of a full one and must not win), then the smaller fraction of noisy records, then the
# smaller worst spread. An attempt without any record (crash before the first result) is retried.
# Every record gets "attempts" and "contaminated": true for a record above the limit and, when the
# kept attempt still has more than BENCH_MAX_BAD_FRAC noisy records (the case as a whole is
# disturbed), for all of its records. Tiny shapes (tens of µs) can stay above the limit from jitter
# alone; read their flag with that in mind.
max_spread="${BENCH_MAX_SPREAD:-5}"; retries="${BENCH_RETRIES:-3}"; max_bad="${BENCH_MAX_BAD_FRAC:-0.1}"
if [ -z "$out" ]; then "$bench" "$@"; exit $?; fi
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
pick="$here/warm_run_pick.py"
tmpd="$(mktemp -d)"; best=""; best_key=""; best_bad=1; rc=0
for ((a = 1; a <= retries; a++)); do
  f="$tmpd/a$a.jsonl"; newargs=()
  for ((i = 0; i < ${#args[@]}; i++)); do
    if [ "${args[$i]}" = --out ]; then newargs+=(--out "$f"); i=$((i + 1)); else newargs+=("${args[$i]}"); fi
  done
  "$bench" "${newargs[@]}"; rc=$?
  read -r n_rec n_ok bad_frac w <<<"$(python3 -I "$pick" score "$f" "$max_spread" 2>/dev/null || echo '0 0 1 999')"
  cur="$n_ok $bad_frac $w"
  if [ -z "$best" ] || python3 -I "$pick" better "$cur" "$best_key"; then best="$f"; best_key="$cur"; best_bad="$bad_frac"; fi
  [ "$n_rec" -gt 0 ] && python3 -c "import sys; sys.exit(0 if float('$bad_frac') <= float('$max_bad') else 1)" && break
  echo "warm_run: $name attempt $a: $n_ok ok record(s), fraction of records above ${max_spread}% spread $bad_frac (worst ${w}%); retrying" >&2
done
attempts=$(( a > retries ? retries : a ))
if [ -f "$best" ]; then python3 -I "$pick" finalize "$best" "$out" "$attempts" "$best_bad" "$max_bad" "$max_spread"; fi
rm -rf "$tmpd"

if [ -n "$out" ] && command -v nvidia-smi >/dev/null; then
  q=clocks.sm,clocks.max.sm,power.draw,temperature.gpu,pstate,clocks_event_reasons.active,clocks_event_reasons.applications_clocks_setting,clocks_event_reasons.sw_power_cap,clocks_event_reasons.sw_thermal_slowdown,clocks_event_reasons.hw_slowdown
  IFS=',' read -r sm smmax pw temp ps act app pcap therm hw < <(nvidia-smi --query-gpu=$q --format=csv,noheader,nounits | head -1)
  t() { echo "$1" | xargs; }
  printf '{"case": "%s", "clock_state": {"warmup_s": %s, "sm_mhz": "%s", "sm_max_mhz": "%s", "power_w": "%s", "temp_c": "%s", "pstate": "%s", "event_reasons": "%s", "applications_clocks_setting": "%s", "sw_power_cap": "%s", "sw_thermal_slowdown": "%s", "hw_slowdown": "%s"}}\n' \
    "$name" "$warm" "$(t "$sm")" "$(t "$smmax")" "$(t "$pw")" "$(t "$temp")" "$(t "$ps")" "$(t "$act")" \
    "$(t "$app")" "$(t "$pcap")" "$(t "$therm")" "$(t "$hw")" >> "$out"
fi
exit $rc
