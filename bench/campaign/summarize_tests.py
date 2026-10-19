#!/usr/bin/env python3
"""Turn a test-suite run (ctest JUnit files + committed-hash results) into one campaign record.

    summarize_tests.py OUT.jsonl [--junit FILE ...] [--hashes FILE] [--expect-tests N] [--expect-hashes N]

The record (one JSON line, status "ok") carries what the decisions file reads:
  tests_total, tests_failed, tests_skipped, pass_fraction   from the JUnit files (a test that is
                                                             skipped counts as not passed)
  failed                                                    the names of the failed tests
  hash_cases, hash_mismatch, hash_match_fraction            from the hashes file written by
                                                             tests/test_output_hashes.py (IMD_HASH_JSON)
  hash_mismatched                                           the names of the cases that differ
With --expect-tests N the pass fraction is passed / max(N, total): a suite that lost tests cannot pass; --expect-hashes N does the same for the hash cases.
A missing or unreadable JUnit file makes the record status "error" (the analyzer reads that as damaged).
Standard library only.
"""
import argparse
import json
import sys
import xml.etree.ElementTree as ET


def read_junit(paths):
    total = failed = skipped = 0
    names = []
    for p in paths:
        root = ET.parse(p).getroot()
        for tc in root.iter("testcase"):
            total += 1
            bad = tc.find("failure") is not None or tc.find("error") is not None
            skip = tc.find("skipped") is not None
            # ctest writes a skipped (disabled / not run) test as <skipped>; a status="notrun" one too
            if tc.get("status") == "notrun":
                skip = True
            if bad:
                failed += 1
                names.append(tc.get("name", "?"))
            elif skip:
                skipped += 1
                names.append(tc.get("name", "?") + " (skipped)")
    return total, failed, skipped, names


def read_hashes(path):
    cases = mism = 0
    names = []
    seen = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            seen[r["name"]] = r          # a rerun of a case replaces its earlier line
    for n, r in seen.items():
        cases += 1
        if not r.get("match"):
            mism += 1
            names.append(n)
    return cases, mism, names


def summarize(junit, hashes, expect, expect_hashes=0):
    rec = {"status": "ok", "method": "tests"}
    try:
        total, failed, skipped, names = read_junit(junit)
        passed = total - failed - skipped
        rec.update(tests_total=total, tests_failed=failed, tests_skipped=skipped, failed=names,
                   pass_fraction=passed / max(expect or 0, total) if total else 0.0)
        if expect and total < expect:
            rec["note"] = f"{total} tests ran, {expect} expected"
    except (OSError, ET.ParseError) as e:
        return {"status": "error", "method": "tests", "error": f"junit: {e}"}
    if hashes:
        try:
            cases, mism, hn = read_hashes(hashes)
            rec.update(hash_cases=cases, hash_mismatch=mism, hash_mismatched=hn,
                       hash_match_fraction=(cases - mism) / max(expect_hashes or 0, cases) if cases else 0.0)
            if expect_hashes and cases < expect_hashes:
                rec["hash_note"] = f"{cases} hash cases ran, {expect_hashes} expected"
        except (OSError, ValueError, KeyError) as e:
            rec.update(status="error", error=f"hashes: {e}")
    return rec


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("out")
    ap.add_argument("--junit", nargs="+", default=[])
    ap.add_argument("--hashes")
    ap.add_argument("--expect-tests", type=int, default=0)
    ap.add_argument("--expect-hashes", type=int, default=0, help="hash cases the run must have recorded (a suite that died halfway cannot match)")
    a = ap.parse_args(argv)
    rec = summarize(a.junit, a.hashes, a.expect_tests, a.expect_hashes)
    with open(a.out, "a") as f:
        f.write(json.dumps(rec, sort_keys=True) + "\n")
    print(json.dumps(rec, sort_keys=True))
    return 0 if rec["status"] == "ok" else 1


if __name__ == "__main__":
    sys.exit(main())
