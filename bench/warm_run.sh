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

"$bench" "$@"
rc=$?

if [ -n "$out" ] && command -v nvidia-smi >/dev/null; then
  q=clocks.sm,clocks.max.sm,power.draw,temperature.gpu,pstate,clocks_event_reasons.active,clocks_event_reasons.applications_clocks_setting,clocks_event_reasons.sw_power_cap,clocks_event_reasons.sw_thermal_slowdown,clocks_event_reasons.hw_slowdown
  IFS=',' read -r sm smmax pw temp ps act app pcap therm hw < <(nvidia-smi --query-gpu=$q --format=csv,noheader,nounits | head -1)
  t() { echo "$1" | xargs; }
  printf '{"case": "%s", "clock_state": {"warmup_s": %s, "sm_mhz": "%s", "sm_max_mhz": "%s", "power_w": "%s", "temp_c": "%s", "pstate": "%s", "event_reasons": "%s", "applications_clocks_setting": "%s", "sw_power_cap": "%s", "sw_thermal_slowdown": "%s", "hw_slowdown": "%s"}}\n' \
    "$name" "$warm" "$(t "$sm")" "$(t "$smmax")" "$(t "$pw")" "$(t "$temp")" "$(t "$ps")" "$(t "$act")" \
    "$(t "$app")" "$(t "$pcap")" "$(t "$therm")" "$(t "$hw")" >> "$out"
fi
exit $rc
