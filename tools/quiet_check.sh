#!/usr/bin/env bash
# Is the machine quiet enough to test/benchmark? Prints the evidence and exits
# non-zero if not. Thresholds are overridable:
#   QUIET_LOAD (1-min load average, default 2.0; 32 hardware threads, and a GPU benchmark
#              uses one core, so ~1 of desktop background load is not interference)
#   QUIET_CPU  (max live %CPU of any single other process, default 50, sampled by top over
#              0.5 s. ps %CPU is a lifetime average, and the check's own ps showed as
#              100%; desktop processes drawing an active window use 20-25% of one core.
#              Builds/TLC raise the load average, GPU jobs trip the GPU checks.)
#   QUIET_GPU  (max clock-weighted GPU load %, default 5: the MEAN over QUIET_GPU_SAMPLES
#              (10) samples 0.2 s apart of utilization.gpu x clocks.sm / clocks.max.sm.
#              utilization.gpu is only "fraction of time some kernel ran", so desktop
#              compositing at idle clocks (P8, ~400 of 3120 MHz) reads 10-20% while doing
#              ~2% of the GPU's work, and blocked runs on 2026-10-03; a benchmark or any
#              real CUDA load raises the clocks and reads near 100%)
#   QUIET_GPU_APP_MIB (compute processes using less GPU memory than this, default 256, are
#              listed but not blocking: desktop apps rendering through Vulkan/GL, e.g. an
#              image viewer, show up as compute apps with a few MiB; a CUDA benchmark
#              holds far more)
#   QUIET_GPU_IGNORE (regex of always-resident GPU processes to ignore)
set -u
load_max="${QUIET_LOAD:-2.0}"; cpu_max="${QUIET_CPU:-50}"; gpu_max="${QUIET_GPU:-5}"
gpu_samples="${QUIET_GPU_SAMPLES:-10}"; app_mib="${QUIET_GPU_APP_MIB:-256}"
gpu_ignore="${QUIET_GPU_IGNORE:-gnome-remote-desktop-daemon|Xorg|gnome-shell}"
noisy=0

read -r l1 l5 l15 _ < /proc/loadavg
echo "load average: $l1 $l5 $l15 (limit $load_max on 1-min)"
awk -v a="$l1" -v b="$load_max" 'BEGIN{exit !(a > b)}' && { echo "  NOISY: load"; noisy=1; }

echo "top processes by live CPU (top, 0.5 s):"
live="$(top -b -n 2 -d 0.5 -w 200 | awk '/^top -/{it++} it==2 && $1 ~ /^[0-9]+$/ {print $1, $9, $12}' \
        | awk -v self=$$ '$1 != self && $3 != "top" && $3 != "awk" && $3 != "quiet_check.sh"' \
        | sort -k2 -nr | head -5)"
echo "$live" | sed 's/^/    /'
top_cpu="$(awk 'NR==1{print $2}' <<<"$live")"; top_name="$(awk 'NR==1{print $3}' <<<"$live")"
awk -v a="${top_cpu:-0}" -v b="$cpu_max" 'BEGIN{exit !(a > b)}' && { echo "  NOISY: $top_name is using ${top_cpu}% CPU"; noisy=1; }

if command -v nvidia-smi >/dev/null; then
  echo "GPU:"
  nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,temperature.gpu,clocks.sm,pstate \
             --format=csv,noheader | sed 's/^/  /'
  util_samples="$(for _ in $(seq "$gpu_samples"); do
      nvidia-smi --query-gpu=index,utilization.gpu,clocks.sm,clocks.max.sm --format=csv,noheader,nounits
      sleep 0.2
    done)"
  while read -r idx mean peak eff; do
    echo "  GPU $idx: utilization mean ${mean}% (peak ${peak}%), clock-weighted load ${eff}% over $gpu_samples samples"
    awk -v a="$eff" -v b="$gpu_max" 'BEGIN{exit !(a > b)}' && { echo "  NOISY: GPU $idx clock-weighted load ${eff}%"; noisy=1; }
  done < <(awk -F', *' '{s[$1]+=$2; n[$1]++; if ($2>p[$1]) p[$1]=$2; if ($4>0) e[$1]+=$2*$3/$4}
                        END{for (i in s) printf "%s %.1f %d %.1f\n", i, s[i]/n[i], p[i], e[i]/n[i]}' <<<"$util_samples")
  apps="$(nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader,nounits \
          | grep -Ev "$gpu_ignore")"
  big="$(awk -F', *' -v t="$app_mib" '$3+0 >= t' <<<"$apps")"
  small="$(awk -F', *' -v t="$app_mib" 'NF && $3+0 < t' <<<"$apps")"
  [ -n "$small" ] && { echo "  small GPU clients (< ${app_mib} MiB, not blocking):"; echo "$small" | sed 's/^/    /'; }
  if [ -n "$big" ]; then echo "  NOISY: compute processes on GPU (MiB):"; echo "$big" | sed 's/^/    /'; noisy=1; fi
fi

gov=/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor
[ -r "$gov" ] && echo "cpu governor: $(cat "$gov")"
nt=/sys/devices/system/cpu/intel_pstate/no_turbo
[ -r "$nt" ] && echo "turbo: $([ "$(cat "$nt")" = 0 ] && echo on || echo off)"

if [ $noisy = 0 ]; then echo "QUIET"; else echo "NOT QUIET"; fi
exit $noisy
