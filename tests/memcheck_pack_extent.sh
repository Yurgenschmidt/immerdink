#!/usr/bin/env bash
# F3's write and read extent under compute-sanitizer memcheck (DESIGN v1.19 §6g; NOTES-C1
# IR17; ctest pack_write_extent_memcheck):
#   memcheck_pack_extent.sh SANITIZER PYTHON TEST_PHASES_PY LIBIMMERDINK
# runs test_phases.py --suite write_extent with every allocation EXACTLY sized
# (IMD_TEST_EXACT=1: the pack, imd_pack's workspace, the contract workspace) under
# compute-sanitizer --tool memcheck --padding 4096 (a padding buffer after every allocation,
# so an access just past one is never inside another):
#   - production: memcheck clean on every imd_pack launch shape and every packed contract
#     (KF, K1 + K2 / K3, KL, KL with 64-row tiles for the in-line partner), each == the
#     plain call;
#   - IMD_TEST_PACK_SHORT=16 (the packed allocation 16 bytes short of pack_bytes while
#     imd_pack is told pack_bytes), one case: memcheck MUST report an invalid global write.
# So the sentinel test's "no byte beyond pack_bytes" holds on exactly sized allocations too,
# and the check is shown to fire.
set -uo pipefail
san=$1 py=$2 tp=$3 lib=$4
log=$(mktemp)
trap 'rm -f "$log"' EXIT
run() {
  IMD_TEST_EXACT=1 IMD_TEST_PACK_SHORT=$1 IMD_TEST_MAX_CASES=$2 "$san" --tool memcheck \
    --padding 4096 --leak-check no --error-exitcode 86 \
    "$py" "$tp" --lib "$lib" --suite write_extent -v >"$log" 2>&1
}
fail=0
run 0 0
rc=$?
if [ "$rc" -ne 0 ] || ! grep -q "PASS  write_extent" "$log"; then
  echo "FAIL exact allocations: exit $rc (memcheck must be clean, the suite must pass)"
  cat "$log"; fail=1
else
  echo "ok  exact allocations: memcheck clean; $(grep -E '^  [0-9]+ packs' "$log" | sed 's/^ *//')"
fi
run 16 1
rc=$?
if [ "$rc" -ne 86 ] || ! grep -q "Invalid __global__ write" "$log"; then
  echo "FAIL mutation (pack allocation 16 bytes short): memcheck did not report the write (exit $rc)"
  cat "$log"; fail=1
else
  echo "ok  mutation: memcheck reports $(grep -c 'Invalid __global__ write' "$log") invalid global writes"
fi
exit $fail
