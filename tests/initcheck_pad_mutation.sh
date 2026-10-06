#!/usr/bin/env bash
# The initcheck mutation test for one-sided padding (DESIGN v1.13 §4b.8, hygiene; ctest
# contract_initcheck_pad):
#   initcheck_pad_mutation.sh SANITIZER PYTHON TEST_CONTRACT_PY LIBIMMERDINK
# runs test_contract.py --suite pad_mutation under compute-sanitizer --tool initcheck on
# unpoisoned, never-written workspaces (IMD_TEST_POISON=0):
#   - no mutation: initcheck clean, the call bit-exact;
#   - IMD_TEST_PAD_MUTATION=A, then B (the K2 / K3 pack of that post-swap operand leaves
#     [ceil(k)_16, k_hat) unwritten; the other operand's zeros keep the call bit-exact):
#     the suite still passes, and initcheck MUST report uninitialised global reads.
# So the bit-exact check alone cannot see one-sided padding bugs, and initcheck can.
set -uo pipefail
san=$1 py=$2 tc=$3 lib=$4
log=$(mktemp)
trap 'rm -f "$log"' EXIT
run() {
  IMD_TEST_POISON=0 IMD_TEST_PAD_MUTATION=$1 "$san" --tool initcheck --error-exitcode 86 \
    "$py" "$tc" --lib "$lib" --suite pad_mutation -v >"$log" 2>&1
}
fail=0
run none
rc=$?
if [ "$rc" -ne 0 ]; then
  echo "FAIL no mutation: exit $rc (initcheck must be clean)"; cat "$log"; fail=1
else
  echo "ok  no mutation: initcheck clean"
fi
for x in A B; do
  run "$x"
  rc=$?
  if ! grep -q "PASS  pad_mutation" "$log"; then
    echo "FAIL mutation $x: the suite did not pass (it must stay bit-exact)"; cat "$log"; fail=1
  elif [ "$rc" -ne 86 ] || ! grep -q "Uninitialized __global__ memory read" "$log"; then
    echo "FAIL mutation $x: initcheck did not report the unwritten padding (exit $rc)"
    cat "$log"; fail=1
  else
    n=$(grep -c "Uninitialized __global__ memory read" "$log")
    echo "ok  mutation $x: bit-exact, initcheck reports $n uninitialised reads"
  fi
done
exit $fail
