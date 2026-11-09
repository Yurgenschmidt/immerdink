#!/usr/bin/env bash
# Run a correctness test command.
#   tools/run_quiet.sh python3 spec/test_oracle.py
# Correctness tests are pass/fail, so by default they are NOT gated on a quiet
# machine. With QUIET_GATE=1 (what tools/run_bench.sh sets) the command takes the
# machine-wide lock and waits (up to QUIET_WAIT seconds, default 900) for
# tools/quiet_check.sh to report QUIET; exit 3 if it never does.
# The command runs inside the shared immerdink memory budget (tools/memcap.sh).
set -u
here="$(cd "$(dirname "$0")" && pwd)"
[ $# -ge 1 ] || { echo "usage: $0 command [args...]" >&2; exit 2; }

if [ "${QUIET_GATE:-0}" = 1 ]; then
  exec 9>"${TLC_LOCK:-${XDG_RUNTIME_DIR:-/tmp}/immerdink-tlc.lock}"
  flock 9
  if [ "${QUIET_MODE:-gate}" = record ]; then
    # Opt-in: do not wait, only record the conditions (QUIET_LOG). For shared nodes where the quiet checks cannot
    # be satisfied; the numbers then include whatever else was running, and bench/warm_run.sh flags noisy cases.
    "$here/quiet_check.sh" > "${QUIET_LOG:-/dev/null}" 2>&1 || echo "QUIET_MODE=record: machine not quiet; running anyway (see ${QUIET_LOG:-quiet log})" >&2
    QUIET_WAIT=-1
  fi
  waited=0
  until [ "${QUIET_MODE:-gate}" = record ] || "$here/quiet_check.sh" >/dev/null 2>&1; do
    if [ "$waited" -ge "${QUIET_WAIT:-900}" ]; then
      echo "machine not quiet after ${waited}s; not running: $*" >&2
      "$here/quiet_check.sh" >&2; exit 3
    fi
    [ "$waited" = 0 ] && echo "waiting for a quiet machine..." >&2
    sleep 15; waited=$((waited + 15))
  done
  # Desktop renderers (the Claude app, gnome-shell) time-slice the GPU whenever they draw,
  # even at idle clocks (docs/NOTES-K5-bimodal.md), which the clock-weighted check above
  # lets through. Prefer a moment when raw utilization is ~0: wait up to QUIET_RAW_WAIT
  # seconds (default 300) for its mean over 2 s to be <= QUIET_GPU_RAW (default 1 %). If it
  # never gets there, run anyway; bench/warm_run.sh then flags the records "contaminated".
  rawt=0; [ "${QUIET_MODE:-gate}" = record ] && QUIET_RAW_WAIT=0
  while [ "$rawt" -lt "${QUIET_RAW_WAIT:-300}" ]; do
    raw="$(for _ in $(seq 10); do nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits | head -1; sleep 0.2; done | awk '{s+=$1} END{printf "%.1f", s/NR}')"
    awk -v a="$raw" -v b="${QUIET_GPU_RAW:-1}" 'BEGIN{exit !(a <= b)}' && break
    [ "$rawt" = 0 ] && echo "waiting for the desktop to stop drawing on the GPU (raw util ${raw}%)..." >&2
    sleep 5; rawt=$((rawt + 7))
  done
  [ "${QUIET_MODE:-gate}" = record ] || "$here/quiet_check.sh" > "${QUIET_LOG:-/dev/null}" 2>&1 || true   # record conditions
fi
# GPU work on this box is shared with other projects (ehrenfest): every gated run also
# takes the machine-wide advisory GPU lock, IMMERDINK_GPU_LOCK (default: ehrenfest's
# ~/.cache/ehrenfest/gpu.lock, the one canonical lock (Eric, 2026-11-08), if present, else $XDG_RUNTIME_DIR/gpu.lock; "none" disables).
gpu_lock="${IMMERDINK_GPU_LOCK:-}"
if [ -z "$gpu_lock" ]; then
  gpu_lock="$HOME/.cache/ehrenfest/gpu.lock"
  [ -d "$(dirname "$gpu_lock")" ] || gpu_lock="${XDG_RUNTIME_DIR:-/tmp}/gpu.lock"
fi
if [ "$gpu_lock" != none ] && [ "${QUIET_GATE:-0}" = 1 ]; then
  IMMERDINK_GPU_LOCK_HELD="$gpu_lock" exec flock "$gpu_lock" "$here/memcap.sh" "$@"   # a tool started below must not take the lock again (flock would wait for itself)
fi
"$here/memcap.sh" "$@"
