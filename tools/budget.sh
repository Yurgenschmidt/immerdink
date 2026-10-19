# Sourced by the scripts that re-launch themselves inside the shared memory budget (tools/memcap.sh):
#   . "$here/budget.sh"; immerdink_in_budget || exec "$here/memcap.sh" "$0" "$@"
# True when the process is already inside immerdink.slice, or tools/memcap.sh already ran it once
# (IMMERDINK_MEMCAP=1: without a systemd user manager there is no slice to be inside of, and the
# scripts re-launched themselves forever). IMMERDINK_CGROUP_FILE replaces /proc/self/cgroup (tests).
immerdink_in_budget() {
  [ "${IMMERDINK_MEMCAP:-}" = 1 ] || grep -q "/immerdink.slice/" "${IMMERDINK_CGROUP_FILE:-/proc/self/cgroup}" 2>/dev/null
}
