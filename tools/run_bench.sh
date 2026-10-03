#!/usr/bin/env bash
# Run a benchmark / timing-sensitive command: always serialized machine-wide and
# only on a quiet machine (see tools/run_quiet.sh). QUIET_LOG=<file> records the
# machine conditions right before the run.
QUIET_GATE=1 exec "$(cd "$(dirname "$0")" && pwd)/run_quiet.sh" "$@"
