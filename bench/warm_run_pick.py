#!/usr/bin/env python3
"""Helper of bench/warm_run.sh (retry policy of a benchmark case); standard library only.

    warm_run_pick.py score <jsonl> <max_spread>
        prints "<n_records> <n_ok> <bad_fraction> <worst_spread>" for one attempt: bad_fraction is the share of
        the status-ok records that carry a spread_pct above max_spread (0 if none carries one); a missing or
        empty file gives "0 0 1 999" (a crash before the first record).
    warm_run_pick.py better "<n_ok> <bad_fraction> <worst>" "<n_ok> <bad_fraction> <worst>"
        exit 0 if the first attempt is better than the second: more ok records (a crashed attempt is a prefix
        of a full one), then the smaller bad fraction, then the smaller worst spread.
    warm_run_pick.py finalize <jsonl> <out> <attempts> <bad_fraction> <max_bad_fraction> <max_spread>
        appends the kept attempt's records to <out> with "attempts" and "contaminated": true for a record above
        max_spread, and for every record when the kept attempt's bad_fraction is above max_bad_fraction.
"""
import json
import sys


def score(path, lim):
    n = ok = meas = bad = 0
    worst = 0.0
    try:
        with open(path) as f:
            for line in f:
                try:
                    d = json.loads(line)
                except ValueError:
                    continue
                n += 1
                if d.get("status") == "ok":
                    ok += 1
                    if "spread_pct" in d:
                        meas += 1
                        worst = max(worst, float(d["spread_pct"]))
                        bad += float(d["spread_pct"]) > lim
    except OSError:
        pass
    if n == 0:
        return 0, 0, 1.0, 999.0
    return n, ok, (bad / meas if meas else 0.0), worst


def key(s):
    ok, bad, worst = s.split()
    return (int(ok), -float(bad), -float(worst))


def main(argv):
    cmd = argv[1]
    if cmd == "score":
        n, ok, bad, worst = score(argv[2], float(argv[3]))
        print(n, ok, f"{bad:.4f}", worst)
    elif cmd == "better":
        return 0 if key(argv[2]) > key(argv[3]) else 1
    elif cmd == "finalize":
        src, dst, attempts = argv[2], argv[3], int(argv[4])
        bad, max_bad, lim = float(argv[5]), float(argv[6]), float(argv[7])
        case_noisy = bad > max_bad
        with open(src) as f, open(dst, "a") as o:
            for line in f:
                try:
                    d = json.loads(line)
                except ValueError:
                    continue
                d["attempts"] = attempts
                d["contaminated"] = bool(case_noisy or (d.get("status") == "ok" and float(d.get("spread_pct", 0.0)) > lim))
                o.write(json.dumps(d) + "\n")
    else:
        raise SystemExit(__doc__)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
