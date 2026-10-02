#!/usr/bin/env bash
# Is the machine quiet enough to test/benchmark? Prints the evidence and exits
# non-zero if not. Thresholds are overridable:
#   QUIET_LOAD (1-min load average, default 1.0)
#   QUIET_CPU  (max %CPU of any single other process, default 20)
#   QUIET_GPU  (max GPU utilization %, default 5)
#   QUIET_GPU_IGNORE (regex of always-resident GPU processes to ignore)
set -u
load_max="${QUIET_LOAD:-1.0}"; cpu_max="${QUIET_CPU:-20}"; gpu_max="${QUIET_GPU:-5}"
gpu_ignore="${QUIET_GPU_IGNORE:-gnome-remote-desktop-daemon|Xorg|gnome-shell}"
noisy=0

read -r l1 l5 l15 _ < /proc/loadavg
echo "load average: $l1 $l5 $l15 (limit $load_max on 1-min)"
awk -v a="$l1" -v b="$load_max" 'BEGIN{exit !(a > b)}' && { echo "  NOISY: load"; noisy=1; }

echo "top processes by CPU:"
ps -eo pid,pcpu,pmem,comm --sort=-pcpu | sed -n "2,6p" | grep -v -w "$$" | sed 's/^/  /'
top_cpu="$(ps -eo pid,pcpu --sort=-pcpu | awk -v self=$$ 'NR>1 && $1!=self {print $2; exit}')"
awk -v a="${top_cpu:-0}" -v b="$cpu_max" 'BEGIN{exit !(a > b)}' && { echo "  NOISY: a process is using ${top_cpu}% CPU"; noisy=1; }

if command -v nvidia-smi >/dev/null; then
  echo "GPU:"
  nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,temperature.gpu,clocks.sm,pstate \
             --format=csv,noheader | sed 's/^/  /'
  while IFS=, read -r idx util; do
    [ "${util// /}" -gt "$gpu_max" ] && { echo "  NOISY: GPU $idx at ${util// /}%"; noisy=1; }
  done < <(nvidia-smi --query-gpu=index,utilization.gpu --format=csv,noheader,nounits)
  apps="$(nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader \
          | grep -Ev "$gpu_ignore")"
  if [ -n "$apps" ]; then echo "  NOISY: compute processes on GPU:"; echo "$apps" | sed 's/^/    /'; noisy=1; fi
fi

gov=/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor
[ -r "$gov" ] && echo "cpu governor: $(cat "$gov")"
nt=/sys/devices/system/cpu/intel_pstate/no_turbo
[ -r "$nt" ] && echo "turbo: $([ "$(cat "$nt")" = 0 ] && echo on || echo off)"

if [ $noisy = 0 ]; then echo "QUIET"; else echo "NOT QUIET"; fi
exit $noisy
