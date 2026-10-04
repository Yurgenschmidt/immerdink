#!/usr/bin/env bash
# nsys profile of one bench case (A5): CUDA API + kernels + GPU context switches, exported to
# sqlite for bench/nsys_calls.py. Runs through tools/run_bench.sh (quiet gate, GPU lock).
#   bench/nsys_case.sh <imd_bench> <out prefix> <name> <spec> <ext> [imd_bench args...]
# e.g. bench/nsys_case.sh build/bench/imd_bench /tmp/a5/base-s256 s256 'ik,jk->ij' \
#        i=256,j=256,k=256 --methods imd --N 8 --reps 300
# (--methods duel profiles two configurations interleaved; nsys_calls.py separates them by
# their kernel sequences.)
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
root="$(dirname "$here")"
bench="$1"; out="$2"; name="$3"; spec="$4"; ext="$5"; shift 5
env_dir="${IMMERDINK_ENV:-$HOME/miniforge3/envs/immerdink}"
nsys="${NSYS:-$HOME/nsight-systems-2026.5.1/target-linux-x64/nsys}"
export LD_LIBRARY_PATH="$env_dir/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
exec "$root/tools/run_bench.sh" "$nsys" profile -t cuda --gpuctxsw=true -o "$out" \
  --force-overwrite true --export sqlite "$bench" --name "$name" --spec "$spec" --ext "$ext" \
  --warmup 20 --max-seconds 5 --phase-reps 0 "$@"
