#!/usr/bin/env bash
# Cross-compile every variant of the Hopper K4 prototype (src/k4/hopper/k4_hopper_variants.def) for sm_90a, one
# translation unit each, and check the machine code (issue 17, docs/NOTES-17.md). Nothing runs: sm_90a code does not run
# on the RTX 4090. Per variant it prints ptxas' registers per thread and spills, and the number of wgmma (IGMMA) and TMA
# (UTMALDG) instructions in the SASS, and it fails on any spill, on a variant without IGMMA or UTMALDG, or on a compile
# error (a pingpong 128 x 128 tile prints ok*: an 8-byte spill that CUTLASS's own epilogue has too). Slow (about 15 s per variant, JOBS in parallel); not part of ctest.
#   tools/check_hopper_variants.sh [variant-name-substring]
#   env: IMMERDINK_ENV (conda env with nvcc), JOBS (default 4), OUT (default <repo>/build-hopper-variants)
set -uo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
root="$(dirname "$here")"
[ -n "${IMMERDINK_MEMCAP:-}" ] || grep -q "/immerdink.slice/" /proc/self/cgroup 2>/dev/null || exec "$here/memcap.sh" "$0" "$@"
env_dir="${IMMERDINK_ENV:-$HOME/miniforge3/envs/immerdink}"
[ -x "$env_dir/bin/nvcc" ] && export PATH="$env_dir/bin:$PATH"
command -v nvcc >/dev/null || { echo "no nvcc (set IMMERDINK_ENV)" >&2; exit 2; }
cutlass="${IMMERDINK_CUTLASS_DIR:-$root/third_party/cutlass}"
out="${OUT:-$root/build-hopper-variants}"
mkdir -p "$out"
def="$root/src/k4/hopper/k4_hopper_variants.def"
mapfile -t names < <(grep -o '^K4H_VARIANT([a-zA-Z0-9_]*' "$def" | sed 's/K4H_VARIANT(//')
filter="${1:-}"

one() {  # index name
  local i="$1" name="$2" o="$out/$2"
  nvcc -std=c++17 -arch=sm_90a -c -DK4H_SELECT="$i" --expt-relaxed-constexpr -Xptxas -v \
       -I "$root/src" -I "$cutlass/include" -I "$cutlass/tools/util/include" -I "${IMD_GENERATED_INCLUDE:-$root/build/generated/include}" \
       -o "$o.o" "$root/src/k4/hopper/k4_hopper_proto.cu" > "$o.log" 2>&1 || { echo "FAIL $name: compile error (see $o.log)"; return; }
  local regs spills igmma tma
  regs=$(grep -o 'Used [0-9]* registers' "$o.log" | head -1 | grep -o '[0-9]*')
  spills=$(grep -o '[0-9]* bytes spill stores' "$o.log" | awk '{s+=$1} END {print s+0}')
  spills=$((spills + $(grep -o '[0-9]* bytes spill loads' "$o.log" | awk '{s+=$1} END {print s+0}')))
  cuobjdump -sass "$o.o" > "$o.sass" 2>/dev/null
  igmma=$(grep -c 'IGMMA' "$o.sass"); tma=$(grep -c 'UTMALDG' "$o.sass")
  local verdict=ok
  # A pingpong 128 x 128 kernel spills one 8-byte value (24 bytes in the counts: one store, two loads) with CUTLASS's own TMA
  # epilogue too (checked 2026-10-15, CUTLASS 4.8.0, ptxas 13.4): a property of the stock kernel at this tile, not of ours.
  if [ "$spills" != 0 ]; then
    if [[ "$name" == *_128x128x128_1x1_ping_* ]] && [ "$spills" -le 24 ]; then verdict="ok*"; else verdict="SPILL"; fi
  fi
  { [ "$igmma" -gt 0 ] && [ "$tma" -gt 0 ]; } || verdict="NO-WGMMA-OR-TMA"
  printf '%-4s %-34s regs %-4s spill bytes %-3s IGMMA %-4s UTMALDG %-3s\n' "$verdict" "$name" "$regs" "$spills" "$igmma" "$tma"
}
export -f one
export out root cutlass
idx=0
list=()
for n in "${names[@]}"; do
  [ -z "$filter" ] || [[ "$n" == *"$filter"* ]] || { idx=$((idx + 1)); continue; }
  list+=("$idx $n"); idx=$((idx + 1))
done
printf '%s\n' "${list[@]}" | xargs -P "${JOBS:-4}" -L 1 bash -c 'one "$0" "$1"' | sort -k2 | tee "$out/summary.txt"
! grep -qv '^ok' "$out/summary.txt"   # ok* = a stock spill (see above)
