#!/usr/bin/env bash
# Run a command while logging the GPU clocks every 20 ms (A5 diagnostic: DVFS states of
# small-shape benchmarks). Log: $CLOCKLOG (default build/nsys/clocks.csv).
#   bench/clocklog.sh <command...>
log="${CLOCKLOG:-build/nsys/clocks.csv}"
nvidia-smi --query-gpu=timestamp,clocks.sm,clocks.mem,pstate,power.draw,utilization.gpu \
  --format=csv,noheader -lms 20 > "$log" &
pid=$!
"$@"
rc=$?
kill "$pid"
exit $rc
