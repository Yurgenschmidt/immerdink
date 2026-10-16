#!/usr/bin/env bash
# Build the bandwidth microbenchmark (bench/membw/imd_membw.cu) for this machine's GPU architecture and run it.
#   tools/build_membw.sh OUTDIR [run [extra imd_membw args]]
# One nvcc call, no CMake, no libraries. Output: OUTDIR/imd_membw and, with `run`, OUTDIR/membw.jsonl (appended).
# Architecture: $CMAKE_CUDA_ARCHITECTURES (the cluster jobs export it) or detected with nvidia-smi.
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
root="$(dirname "$here")"
out="${1:?usage: build_membw.sh OUTDIR [run [args]]}"
shift
mkdir -p "$out"
arch="${CMAKE_CUDA_ARCHITECTURES:-}"
[ -n "$arch" ] || arch="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d '[:space:].')"
nvcc -O3 -std=c++17 -arch="sm_${arch%%-*}" "$root/bench/membw/imd_membw.cu" -o "$out/imd_membw"
echo "built $out/imd_membw (sm_${arch%%-*})"
if [ "${1:-}" = run ]; then
  shift
  "$out/imd_membw" --out "$out/membw.jsonl" "$@"
fi
