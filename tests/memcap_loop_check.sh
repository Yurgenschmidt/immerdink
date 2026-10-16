#!/usr/bin/env bash
# Regression test: scripts that re-launch themselves through tools/memcap.sh must end when there
# is no systemd user manager (systemctl failing). They used to re-launch themselves about 200
# times in 6 s, because the "am I inside the memory slice" test can never become true then.
# The cgroup file is replaced by an empty one, so the test also works inside immerdink.slice.
# CPU only; builds nothing. Usage: memcap_loop_check.sh <repo root>
set -u
root="${1:?repo root}"
work="$(mktemp -d)"; trap 'rm -rf "$work"' EXIT
mkdir "$work/bin"
printf '#!/bin/sh\nexit 1\n' > "$work/bin/systemctl"; chmod +x "$work/bin/systemctl"
printf '#!/bin/sh\nexit 1\n' > "$work/bin/systemd-run"; chmod +x "$work/bin/systemd-run"
fail=0
for s in test.sh check_proofs.sh; do
  out="$(env -u IMMERDINK_MEMCAP IMMERDINK_CGROUP_FILE=/dev/null PATH="$work/bin:$PATH" timeout 20 "$root/tools/$s" --budget-probe 2>&1)"; rc=$?
  warns=$(printf '%s\n' "$out" | grep -c 'memcap: no systemd user manager')
  probes=$(printf '%s\n' "$out" | grep -c 'in budget')
  if [ "$rc" = 0 ] && [ "$warns" = 1 ] && [ "$probes" = 1 ]; then
    echo "memcap_loop_check: $s ends (exit 0, one warning, one probe line)"
  else
    echo "memcap_loop_check: $s FAILED (exit $rc, $warns warning(s), $probes probe line(s))" >&2
    fail=1
  fi
done
exit $fail
