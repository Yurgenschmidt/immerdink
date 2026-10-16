#!/usr/bin/env bash
# After a failed ctest run: rerun the failed tests once with full output, then once under
# compute-sanitizer (memcheck), so a cluster job leaves the evidence needed to class each failure
# (a test assumption or a library bug) without another queue wait.
#   tools/rerun_failed.sh BUILD_DIR [LOGFILE_PREFIX]
# Output goes to stdout; with a prefix, also to <prefix>.rerun.log and <prefix>.sanitizer.log.
# Environment: IMMERDINK_COMPUTE_SANITIZER (default: compute-sanitizer on PATH, or next to nvcc).
# Exit status: 0 when every failed test passed on the plain rerun (a flaky test), else ctest's nonzero status.
# Tests that preload tests/alloc_shim.c (the contract_* suites) are not run under the sanitizer: the
# interposer and the sanitizer both hook the CUDA runtime. Their plain rerun is still done.
set -uo pipefail
build="${1:?usage: rerun_failed.sh BUILD_DIR [LOGFILE_PREFIX]}"
prefix="${2:-}"
failed="$build/Testing/Temporary/LastTestsFailed.log"
[ -s "$failed" ] || { echo "rerun_failed: no failed tests recorded in $failed"; exit 0; }
names=$(sed 's/^[0-9]*://' "$failed" | paste -sd'|')
echo "rerun_failed: failed tests: $(echo "$names" | tr '|' ' ')"

tee_to() { if [ -n "$prefix" ]; then tee "$1"; else cat; fi; }
echo "=== rerun with full output"
ctest --test-dir "$build" --rerun-failed -V 2>&1 | tee_to "${prefix}.rerun.log"
rc=${PIPESTATUS[0]}

san="${IMMERDINK_COMPUTE_SANITIZER:-$(command -v compute-sanitizer || true)}"
if [ -z "$san" ] && command -v nvcc >/dev/null 2>&1; then
  d="$(dirname "$(command -v nvcc)")"; [ -x "$d/compute-sanitizer" ] && san="$d/compute-sanitizer"
fi
if [ -z "$san" ] || [ ! -x "$san" ]; then
  echo "=== compute-sanitizer not found: skipped (set IMMERDINK_COMPUTE_SANITIZER)"
  exit "$rc"
fi
echo "=== failed tests under $san --tool memcheck"
for t in $(echo "$names" | tr '|' ' '); do
  # The test's command line, as ctest knows it (first command of the test).
  cmd=$(ctest --test-dir "$build" -R "^${t}$" -V -N 2>/dev/null | sed -n 's/^[0-9]*: Test command: //p' | head -1)
  [ -n "$cmd" ] || { echo "--- $t: no command found"; continue; }
  if echo "$cmd" | grep -q "alloc_shim"; then echo "--- $t: uses the allocation shim; not run under the sanitizer"; continue; fi
  echo "--- $t"
  # shellcheck disable=SC2086
  eval "$san --tool memcheck --error-exitcode 99 $cmd" 2>&1 | tail -40 | tee_to "${prefix}.sanitizer.$t.log"
done
exit "$rc"
