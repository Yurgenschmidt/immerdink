#!/usr/bin/env bash
# Compile src/contract.cu alone for sm_89 and for sm_90a (a cross-compile, nothing runs) and write the normalised per-kernel SASS hashes
# (tools/sass_hash.py) and ptxas resource usage (cuobjdump --dump-resource-usage) to $out/<tag>_<arch>.{hash,res}. Compare two tags with
#   diff <(grep -v '^TOTAL' a.hash | cut -d' ' -f1,2- ) ...   or just: tools/sass_ident.sh --cmp tagA tagB
# Used for the issue 23 claim "the knob off changes no shipped kernel's SASS" (docs/NOTES-23.md).
#   tools/sass_ident.sh <tag> [extra nvcc flags...]      env: SASS_OUT (default <repo>/build-sass23), IMMERDINK_ENV, IMMERDINK_CUTLASS_DIR, SASS_JOBS (unused)
set -euo pipefail
root="$(cd "$(dirname "$0")/.." && pwd)"
env_dir="${IMMERDINK_ENV:-$HOME/miniforge3/envs/immerdink}"
export PATH="$env_dir/bin:$PATH"
out="${SASS_OUT:-$root/build-sass23}"; mkdir -p "$out"
if [ "${1:-}" = --cmp ]; then
  for a in 89 90a 120; do
    ta="$out/$2_$a.hash"; tb="$out/$3_$a.hash"
    if diff <(grep -v '^TOTAL' "$ta" | cut -d' ' -f1 | sort) <(grep -v '^TOTAL' "$tb" | cut -d' ' -f1 | sort) > /dev/null; then
      echo "sm_$a: IDENTICAL  $(grep '^TOTAL' "$ta")  /  $(grep '^TOTAL' "$tb")"
    else echo "sm_$a: DIFFERENT  $(grep '^TOTAL' "$ta")  /  $(grep '^TOTAL' "$tb")"; fi
  done
  exit 0
fi
tag="$1"; shift
cutlass="${IMMERDINK_CUTLASS_DIR:-/home/ef/repos/immerdink/third_party/cutlass}"
gen="${IMD_GENERATED_INCLUDE:-/home/ef/repos/immerdink/build/generated/include}"
one() {
  local a="$1"; shift
  local gc="--generate-code=arch=compute_$a,code=sm_$a"
  nice -n 19 nvcc -forward-unknown-to-host-compiler -ccbin="$env_dir/bin/g++" -I"$root/include" -I"$root/src" -I"$root/src/host" -I"$gen" \
    -isystem "$env_dir/targets/x86_64-linux/include" -isystem "$env_dir/targets/x86_64-linux/include/cccl" -isystem "$cutlass/include" \
    -O3 -DNDEBUG -std=c++17 "$gc" -Xcompiler=-fPIC --expt-relaxed-constexpr -diag-suppress=20012 -Xptxas -v "$@" \
    -x cu -c "$root/src/contract.cu" -o "$out/${tag}_$a.o" > "$out/${tag}_$a.ptxas" 2>&1
  python3 -I "$root/tools/sass_hash.py" "$out/${tag}_$a.o" > "$out/${tag}_$a.hash"
  cuobjdump --dump-resource-usage "$out/${tag}_$a.o" > "$out/${tag}_$a.res" 2>&1
  echo "sm_$a: $(grep '^TOTAL' "$out/${tag}_$a.hash")"
}
one 89 "$@" &
one 90a "$@" &
one 120 "$@" &
wait
