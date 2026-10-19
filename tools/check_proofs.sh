#!/usr/bin/env bash
# Build the Lean proofs (proofs/) and fail on any escape hatch.
#
#   1. grep: no `sorry`, `admit` or `axiom` token anywhere in proofs/Proofs*.
#   2. lake build must succeed (Mathlib oleans come from `lake exe cache get`).
#   3. #print axioms on every headline theorem listed in proofs/AxiomCheck.lean must
#      report only Lean's standard axioms (propext, Classical.choice, Quot.sound).
#
# Usage: tools/check_proofs.sh        (from anywhere)
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# Lean/Mathlib elaboration is memory-hungry: run inside the shared budget.
. "$ROOT/tools/budget.sh"
immerdink_in_budget || exec "$ROOT/tools/memcap.sh" "$ROOT/tools/check_proofs.sh" "$@"
[ "${1:-}" = --budget-probe ] && { echo "check_proofs.sh: in budget"; exit 0; }
PROOFS="$ROOT/proofs"
LAKE="${LAKE:-$(command -v lake || echo "$HOME/.elan/bin/lake")}"

cd "$PROOFS"

echo "== grep for sorry/admit/axiom"
if grep -rnwE 'sorry|admit|axiom' Proofs Proofs.lean; then
  echo "FAIL: forbidden token found above" >&2
  exit 1
fi
echo "ok: none found"

echo "== lake exe cache get"
"$LAKE" exe cache get >/dev/null 2>&1 || echo "warning: cache get failed (offline?)"

echo "== lake build"
if ! "$LAKE" build 2>&1 | tail -n 40; then  # pipefail: lake's status counts
  echo "FAIL: lake build" >&2
  exit 1
fi

if [[ -f AxiomCheck.lean ]]; then
  echo "== #print axioms (headline theorems)"
  out="$("$LAKE" env lean AxiomCheck.lean 2>&1)" || { echo "$out"; echo "FAIL: AxiomCheck.lean" >&2; exit 1; }
  echo "$out" | grep -cE "depends on axioms|does not depend on any axioms" | sed 's/^/theorems checked: /'
  bad="$(echo "$out" | grep -oE '\[[^]]*\]' | tr -d '[]' | tr ',' '\n' | sed 's/ //g' \
         | grep -vE '^(propext|Classical\.choice|Quot\.sound)?$' || true)"
  if [[ -n "$bad" ]] || echo "$out" | grep -q "error"; then
    echo "$out"
    echo "FAIL: non-standard axioms: $bad" >&2
    exit 1
  fi
  echo "ok: only propext / Classical.choice / Quot.sound"
fi

echo "PASS: proofs build with no sorry/admit/axiom"
