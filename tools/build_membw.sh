#!/usr/bin/env bash
# Build the bandwidth microbenchmark (bench/membw/imd_membw.cu) for this machine's GPU architecture and run it.
#   tools/build_membw.sh OUTDIR [run [extra imd_membw args]]
# One nvcc call, no CMake, no libraries. Output: OUTDIR/imd_membw and, with `run`, OUTDIR/membw.jsonl (rewritten by each run).
# The run takes the shared GPU lock (IMMERDINK_GPU_LOCK, as tools/run_quiet.sh; "none" disables) when the lock's folder exists, as on the dev box, unless a
# caller (tools/run_bench.sh) already holds it (IMMERDINK_GPU_LOCK_HELD = the lock's path: taking it twice would block for ever);
# on a cluster node there is none and it runs directly. It is a bandwidth measurement: run it on an otherwise idle GPU.
# Architecture: $CMAKE_CUDA_ARCHITECTURES (the cluster jobs export it) or detected with nvidia-smi.
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
root="$(dirname "$here")"
out="${1:?usage: build_membw.sh OUTDIR [run [args]]}"
shift
mkdir -p "$out"
arch="${CMAKE_CUDA_ARCHITECTURES:-}"
[ -n "$arch" ] || arch="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d '[:space:].')"
case "$arch" in *[\;,\ ]*|native|all|all-major) echo "build_membw.sh: CMAKE_CUDA_ARCHITECTURES=$arch is a list or a keyword; give one architecture, e.g. 90" >&2; exit 2;; esac
nvcc -O3 -std=c++17 -arch="sm_${arch%%-*}" "$root/bench/membw/imd_membw.cu" -o "$out/imd_membw"
echo "built $out/imd_membw (sm_${arch%%-*})"
if [ "${1:-}" = run ]; then
  shift
  gpu_lock="${IMMERDINK_GPU_LOCK:-$HOME/.cache/ehrenfest/gpu.lock}"
  if [ "$gpu_lock" != none ] && [ "${IMMERDINK_GPU_LOCK_HELD:-}" != "$gpu_lock" ] && [ -d "$(dirname "$gpu_lock")" ]; then   # tools/run_bench.sh already holds it
    flock "$gpu_lock" "$out/imd_membw" --out "$out/membw.jsonl" "$@"
  else
    "$out/imd_membw" --out "$out/membw.jsonl" "$@"
  fi
fi
