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
  waited=0
  until "$here/quiet_check.sh" >/dev/null 2>&1; do
    if [ "$waited" -ge "${QUIET_WAIT:-900}" ]; then
      echo "machine not quiet after ${waited}s; not running: $*" >&2
      "$here/quiet_check.sh" >&2; exit 3
    fi
    [ "$waited" = 0 ] && echo "waiting for a quiet machine..." >&2
    sleep 15; waited=$((waited + 15))
  done
  "$here/quiet_check.sh" > "${QUIET_LOG:-/dev/null}" 2>&1 || true   # record conditions
fi
"$here/memcap.sh" "$@"
