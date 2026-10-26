#!/usr/bin/env bash
# Machine-code gate of the Hopper wgmma K4 backend in a built libimmerdink.so (issue 22, docs/NOTES-22.md "Library integration (design)").
# Nothing runs: the sm_90a image does not run on any other GPU. Checks, per K4 wgmma kernel of the library's sm_90a image:
#   - it has IGMMA instructions (the wgmma path is compiled in, not elided),
#   - WARPGROUP.DEPBAR (the wait on the wgmma group) occurs fewer times than IGMMA: ptxas serialises the wgmma chain (diagnostic C7510:
#     "wgmma pipeline crossing function boundary / serialized") by a wait after every one, which is a performance bug that does not
#     fail a build,
#   - no ptxas C7510 / "Potential Performance Loss" in the given build logs (optional: build logs as further arguments),
#   - the image has no register spill (cuobjdump -res-usage is not needed: ptxas -v output is in the logs if the build was verbose; here the
#     SASS is searched for local-memory STL/LDL in the kernel).
#   tools/check_k4_wgmma_sass.sh build/libimmerdink.so [build.log ...]
# env: IMMERDINK_ENV (conda env with cuobjdump).
set -uo pipefail
lib="${1:-}"
[ -f "$lib" ] || { echo "usage: $0 libimmerdink.so [build.log ...]" >&2; exit 2; }
shift
env_dir="${IMMERDINK_ENV:-$HOME/miniforge3/envs/immerdink}"
[ -x "$env_dir/bin/cuobjdump" ] && export PATH="$env_dir/bin:$PATH"
command -v cuobjdump >/dev/null || { echo "no cuobjdump (set IMMERDINK_ENV)" >&2; exit 2; }
sass="$(mktemp)"
trap 'rm -f "$sass"' EXIT
if ! cuobjdump -sass -arch sm_90a "$lib" > "$sass" 2>/dev/null || ! grep -q 'Function :' "$sass"; then
  echo "FAIL  $lib has no sm_90a image (built without IMMERDINK_K4_WGMMA, or without sm_90a in the architectures)"
  exit 1
fi
fail=0
rows="$(awk '
  /Function :/ { f = $3; next }
  /IGMMA/ { igmma[f]++ }
  /WARPGROUP\.DEPBAR|WARPGROUP\.ARRIVE/ { if ($0 ~ /DEPBAR/) dep[f]++ }
  /STL|LDL/ { local_[f]++ }
  END { for (k in igmma) printf "%s %d %d %d\n", k, igmma[k], dep[k] + 0, local_[k] + 0 }' "$sass" | sort)"
[ -n "$rows" ] || { echo "FAIL  no wgmma (IGMMA) kernel in the sm_90a image"; exit 1; }
n=0
while read -r name igmma dep loc; do
  n=$((n + 1))
  verdict=ok
  [ "$dep" -lt "$igmma" ] || verdict=SERIALISED
  [ "$loc" -eq 0 ] || verdict="${verdict}+LOCAL-MEMORY"
  [ "$verdict" = ok ] || fail=1
  printf '%-14s IGMMA %-4s WARPGROUP.DEPBAR %-3s local-memory ops %-3s %s\n' "$verdict" "$igmma" "$dep" "$loc" "$(echo "$name" | cut -c1-60)"
done <<< "$rows"
for log in "$@"; do
  if grep -q 'C7510\|wgmma pipeline' "$log" 2>/dev/null; then
    echo "FAIL  $log: ptxas reports C7510 (wgmma serialised)"
    fail=1
  fi
done
[ "$fail" = 0 ] && echo "PASS  $n wgmma kernels: DEPBAR < IGMMA, no local memory${*:+, no C7510 in the build logs}"
exit "$fail"
