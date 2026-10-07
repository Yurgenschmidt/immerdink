#!/usr/bin/env bash
# Run a command inside immerdink's shared memory budget.
#   tools/memcap.sh cmd [args...]
#
# Every heavy immerdink job (TLC, builds, tests, benchmarks, Lean) runs in one
# user cgroup slice, immerdink.slice, capped at IMMERDINK_MEM_MAX (default: half
# of physical RAM) with swap off. Jobs started concurrently (several agents, a
# parallel TLC run) share that one budget, so together they cannot take the
# desktop down: if they exceed it, the kernel OOM-kills inside the slice only.
# (2026-10-03: 17 TLC JVMs with default 16 GB heaps exhausted 61 GB and crashed
# the machine.) The cap is a runtime property; nothing persists past reboot.
#
# Already inside the slice (nested call): runs the command directly.
# No systemd user manager: warns and runs uncapped.
#
# A time limit goes INSIDE: `tools/memcap.sh timeout 600 cmd ...`. The command runs in a
# systemd scope, and `timeout 600 tools/memcap.sh cmd ...` did not stop it (2026-10-07: a
# timed-out study chunk kept running and writing records after the timeout fired).
set -u
[ $# -ge 1 ] || { echo "usage: $0 command [args...]" >&2; exit 2; }

slice=immerdink.slice
if grep -q "/$slice/" /proc/self/cgroup 2>/dev/null; then exec "$@"; fi

mem_total_kb=$(awk '/^MemTotal:/{print $2}' /proc/meminfo)
cap="${IMMERDINK_MEM_MAX:-$((mem_total_kb / 2 / 1024 / 1024))G}"

if ! systemctl --user set-property --runtime "$slice" MemoryMax="$cap" MemorySwapMax=0 2>/dev/null; then
  echo "memcap: no systemd user manager; running WITHOUT a memory cap: $*" >&2
  exec "$@"
fi
systemd-run --user --scope --quiet --collect --slice="$slice" -- "$@"
rc=$?
if [ "$rc" = 137 ]; then
  echo "memcap: killed (SIGKILL), most likely by the $cap immerdink memory cap: $*" >&2
fi
exit "$rc"
