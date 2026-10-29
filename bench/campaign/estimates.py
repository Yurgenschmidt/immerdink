#!/usr/bin/env python3
"""Gate G6: are a manifest's time estimates plausible against the issue-11 logs?

    estimates.py table LOGDIR...                 seconds per case, from the '== <case> <ISO time>' markers of the *.out logs
    estimates.py check MANIFEST --logs LOGDIR... for each sweep experiment: the summed time of its cases in the logs against est_min

A bench sweep logs one marker line per case (bench/run_sweep.sh). The time of a case is the gap to the next marker (the last
case has no end marker and takes the median of the others). Cases are matched by name between the sweep file of the experiment and the log of the
same sweep (sweep_compare.txt -> compare.out ...). An estimate below 80 % of what the logs imply is an error: the campaign would
then skip experiments it believes will fit, or hit its wall limit.
"""
import argparse
import datetime as dt
import os
import re
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import manifest as mf  # noqa: E402

MARK = re.compile(r"^== (?:\[\d+/\d+\] )?(\S+) (\d{4}-\d\d-\d\dT[\d:]+[+-]\d\d:\d\d)")
LOG_OF = {"sweep_imd.txt": "sweep.out", "sweep_compare.txt": "compare.out", "sweep_compare_energy.txt": "compare_energy.out",
          "sweep_energy.txt": "energy.out", "sweep_baselines.txt": "baselines.out", "sweep_cutensor_ws.txt": "cutensor_ws.out"}
TOLERANCE = 0.8


def case_seconds(path):
    marks = []
    with open(path, errors="replace") as f:
        for line in f:
            m = MARK.match(line)
            if m:
                marks.append((m.group(1), dt.datetime.fromisoformat(m.group(2))))
    out = {}
    for i, (name, t) in enumerate(marks[:-1]):
        gap = (marks[i + 1][1] - t).total_seconds()
        if gap >= 0:
            out[name] = gap
    if marks and out:           # the last case has no end marker: take the median of the others
        v = sorted(out.values())
        out[marks[-1][0]] = v[len(v) // 2]
    return out


def job_logs(dirs):
    """{log file name: [parsed case seconds, one per job folder]}"""
    found = {}
    for d in dirs:
        jobs = [d] if os.path.exists(os.path.join(d, "conditions.txt")) else \
            [os.path.join(d, x) for x in sorted(os.listdir(d)) if os.path.isdir(os.path.join(d, x))]
        for j in jobs:
            for n in set(LOG_OF.values()):
                p = os.path.join(j, n)
                if os.path.exists(p):
                    cs = case_seconds(p)
                    if cs:
                        found.setdefault(n, []).append(cs)
    return found


def sweep_cases(path_text, cases_re=None):
    names = []
    for line in path_text.splitlines():
        name = line.split("|")[0].strip()
        if name and not name.startswith("#") and "|" in line and (not cases_re or re.search(cases_re, name)):
            names.append(name)
    return names


def check(manifest_path, logdirs, src=None):
    m = mf.load(manifest_path)
    logs = job_logs(logdirs)
    rows, bad = [], 0
    commit = m.data["campaign"]["immerdink_commit"]
    for e in m.experiments:
        if e.get("kind") != "sweep":
            continue
        log = LOG_OF.get(os.path.basename(e["file"]))
        if log is None or log not in logs:
            rows.append((e["id"], e["est_min"], None, "no matching log"))
            continue
        text = ""
        if src:
            r = subprocess.run(["git", "-C", src, "show", f"{commit}:{e['file']}"], capture_output=True, text=True)
            text = r.stdout
        names = sweep_cases(text, e.get("cases"))
        per_job = [sum(c.get(n, 0.0) for n in names) for c in logs[log]]
        if not names or not per_job:
            rows.append((e["id"], e["est_min"], None, "no cases matched"))
            continue
        implied = max(per_job) / 60.0
        ok = e["est_min"] >= TOLERANCE * implied
        bad += not ok
        rows.append((e["id"], e["est_min"], implied, "ok" if ok else f"TOO LOW: est_min < {TOLERANCE:.0%} of the logs"))
    return rows, bad


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    s = p.add_subparsers(dest="cmd", required=True)
    t = s.add_parser("table")
    t.add_argument("logs", nargs="+")
    c = s.add_parser("check")
    c.add_argument("manifest")
    c.add_argument("--logs", nargs="+", required=True)
    c.add_argument("--src")
    a = p.parse_args(argv)
    if a.cmd == "table":
        for n, js in job_logs(a.logs).items():
            for i, cs in enumerate(js):
                print(f"{n} job {i}: " + ", ".join(f"{k} {v:.0f}s" for k, v in cs.items()) + f"  (total {sum(cs.values()) / 60:.1f} min)")
        return 0
    rows, bad = check(a.manifest, a.logs, a.src)
    for eid, est, implied, note in rows:
        print(f"{eid}: est {est:g} min, logs imply {'-' if implied is None else f'{implied:.1f} min'}: {note}")
    print(f"G6 estimates: {bad} too low")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
