#!/usr/bin/env python3
"""The campaign analyzer: one verdict per decision from a downloaded result folder.

    analyze.py verdicts FOLDER [--decisions FILE] [--accept-unpinned] [--json OUT] [--strict]
    analyze.py tables   FOLDER [--out DIR]      the comparison tables of the folder's JSONL records (bench/summarize.py)
    analyze.py bitcheck FOLDER                  gate G5: every bit_identical / elements_differing field must say identical
    analyze.py golden   JOBFOLDER [...]         regenerate an older (issue 11 style) job folder's tables and diff them
                                                against the .md files the job itself wrote

A verdict is ADOPT, KEEP, GO, NO-GO, PASS, FLAGGED, INCONCLUSIVE or INVALID (decisions.py). INVALID means the data
cannot be trusted, and then no decision is read from it: a truncated, resized or altered file (DONE.json lists the
size and sha256 of every file), a folder from another manifest, a wrong or dirty source commit, an edited decisions
file (its sha256 must be the one the manifest pinned), a missing LOCK.json, or a GPU that does not belong to the class.
INCONCLUSIVE means the data are sound but do not decide: an experiment failed, was skipped for time, or the numbers fall
between the limits. FOLDER is the job folder (one GPU class); run it once per job of a submission.
"""
import argparse
import difflib
import json
import os
import re
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import decisions as dc     # noqa: E402
import manifest as mf      # noqa: E402
import results as rs       # noqa: E402

BENCH_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class Ctx:
    """Data access for decision functions."""

    def __init__(self, folder, base_dir):
        self.folder, self.base_dir = folder, base_dir
        self._cache = {}

    def jsonl(self, exp, file):
        p = os.path.join(self.folder, "exp", exp, file)
        if p not in self._cache:
            self._cache[p] = rs.read_jsonl(p)
        return self._cache[p]

    def base_jsonl(self, rel):
        p = os.path.join(self.base_dir, rel)
        if p not in self._cache:
            self._cache[p] = rs.read_jsonl(p)
        return self._cache[p]


def check_folder(folder, decisions_file=None, accept_unpinned=False):
    """Folder-level integrity. Returns (problems, info) where problems is a list of strings (any -> every verdict INVALID)
    and info holds the manifest, decisions, lock and the per-experiment states."""
    problems = []
    info = {"exp_state": {}, "exp_problems": {}}
    need = (rs.LOCK, "MANIFEST.toml", "DECISIONS.toml", "journal.jsonl", "env.json")
    for n in need:
        if not os.path.exists(os.path.join(folder, n)):
            problems.append(f"{n} is missing")
    if problems:
        return problems, info
    try:
        lock = rs.read_json(os.path.join(folder, rs.LOCK))
        m = mf.load(os.path.join(folder, "MANIFEST.toml"))
        decs, dsha = dc.load(os.path.join(folder, "DECISIONS.toml"))
    except Exception as e:
        return [f"unreadable inputs: {type(e).__name__}: {e}"], info
    info.update(lock=lock, manifest=m, decisions=decs)
    if m.sha256 != lock.get("manifest_sha256"):
        problems.append("MANIFEST.toml differs from the manifest the job locked (sha256)")
    if rs.sha256_file(os.path.join(folder, "DECISIONS.toml")) != lock.get("decisions_sha256"):
        problems.append("DECISIONS.toml differs from the decisions file the job locked (sha256)")
    pinned = m.data.get("campaign", {}).get("decisions_sha256")
    if pinned != lock.get("decisions_sha256"):
        problems.append("the decisions file the job ran with is not the one the manifest pins")
    if decisions_file is not None:
        if rs.sha256_file(decisions_file) != pinned:
            problems.append(f"{decisions_file} is not the decisions file the manifest pinned (edited after the manifest was written)")
        else:
            info["decisions_path"] = decisions_file
    c = m.data.get("campaign", {})
    if lock.get("pinned_commit") != c.get("immerdink_commit"):
        problems.append("the locked pin differs from the manifest's immerdink_commit")
    if not accept_unpinned:
        if lock.get("pin_check") != "ok":
            problems.append("the source commit was not checked against the pin (pin_check skipped)")
        if lock.get("src_commit") != c.get("immerdink_commit"):
            problems.append(f"the job ran on commit {str(lock.get('src_commit'))[:12]}, the manifest pins {str(c.get('immerdink_commit'))[:12]}")
        if lock.get("src_dirty"):
            problems.append("the job ran on a dirty source tree")
    cls = lock.get("class")
    if cls not in m.classes:
        problems.append(f"class {cls!r} is not in the manifest")
    else:
        name = lock.get("gpu_name", "")
        if m.classify(name) != cls:
            problems.append(f"the GPU {name!r} does not belong to class {cls!r}")
        try:
            envname = ((rs.read_json(os.path.join(folder, "env.json")).get("gpu") or {}).get("name"))
            if envname and not lock.get("gpu_name_overridden") and envname != name:
                problems.append(f"env.json names the GPU {envname!r}, the lock {name!r}")
        except (ValueError, OSError) as e:
            problems.append(f"env.json unreadable: {e}")
    exp_root = os.path.join(folder, "exp")
    for e in (m.plan(cls) if cls in m.classes else []):
        d = os.path.join(exp_root, e["id"])
        st, pr = rs.verify_exp(d) if os.path.isdir(d) else ("missing", [])
        info["exp_state"][e["id"]] = st
        if pr:
            info["exp_problems"][e["id"]] = pr
    info["complete"] = os.path.exists(os.path.join(folder, rs.COMPLETE))
    return problems, info


def verdicts(folder, decisions_file=None, accept_unpinned=False):
    """List of {id, kind, verdict, reason, details} for the decisions that apply to the job's class, plus the folder info."""
    problems, info = check_folder(folder, decisions_file, accept_unpinned)
    out = []
    if "decisions" not in info:
        return [{"id": "(folder)", "kind": "-", "verdict": "INVALID", "reason": "; ".join(problems), "details": {}}], info
    cls = info["lock"].get("class")
    if decisions_file is not None and "decisions_path" in info:
        decs = dc.load(decisions_file)[0]
        base = os.path.dirname(os.path.abspath(decisions_file))
    else:
        decs, base = info["decisions"], folder
    ctx = Ctx(folder, base)
    for d in decs:
        if not dc.applies(d, cls):
            continue
        r = {"id": d["id"], "kind": d["kind"]}
        bad = [p for p in problems]
        for x in d.get("requires", []):
            st = info["exp_state"].get(x, "missing")
            if st == "damaged":
                bad.append(f"experiment {x} is damaged: " + "; ".join(info["exp_problems"].get(x, [])))
        if bad:
            r.update(verdict="INVALID", reason="; ".join(bad), details={})
            out.append(r)
            continue
        notok = [f"{x}: {info['exp_state'].get(x, 'missing')}" for x in d.get("requires", []) if info["exp_state"].get(x) != "ok"]
        if notok:
            r.update(verdict="INCONCLUSIVE", reason="no data from " + ", ".join(notok), details={})
            out.append(r)
            continue
        try:
            r.update(dc.evaluate(d, ctx))
        except (ValueError, OSError, KeyError, TypeError) as e:
            r.update(verdict="INVALID", reason=f"records unreadable: {type(e).__name__}: {e}", details={})
        out.append(r)
    return out, info


def bitcheck(folder):
    """Gate G5 on a result folder: every record that says whether a variant's output is bit-identical to its reference
    must say yes. Returns (n_checked, list of problems). Reads the JSONL files of the experiments whose DONE.json verifies."""
    n, bad = 0, []
    exp_root = os.path.join(folder, "exp")
    for eid in sorted(os.listdir(exp_root)) if os.path.isdir(exp_root) else []:
        d = os.path.join(exp_root, eid)
        if rs.verify_exp(d)[0] not in ("ok", "failed"):
            bad.append(f"{eid}: folder not intact, cannot check")
            continue
        for fn in rs.exp_files(d):
            if not fn.endswith(".jsonl"):
                continue
            try:
                recs = rs.read_jsonl(os.path.join(d, fn))
            except ValueError as e:
                bad.append(f"{eid}/{fn}: {e}")
                continue
            for r in recs:
                if "bit_identical" in r or "elements_differing" in r:
                    n += 1
                    if r.get("bit_identical") is not True or r.get("elements_differing", 0):
                        bad.append(f"{eid}/{fn}: case {r.get('case')} {r.get('variant', r.get('method'))}: "
                                   f"bit_identical={r.get('bit_identical')} elements_differing={r.get('elements_differing')}")
    return n, bad


def render(vs, info):
    lock = info.get("lock", {})
    lines = [f"# Verdicts: campaign {lock.get('campaign', '?')}, class {lock.get('class', '?')}, "
             f"commit {str(lock.get('src_commit', '?'))[:12]}", ""]
    if info and not info.get("complete", False):
        lines += ["The job did not finish (no COMPLETE.json): experiments after the cut are missing.", ""]
    if info.get("exp_state"):
        lines += ["| experiment | state |", "|---|---|"] + [f"| {k} | {v} |" for k, v in info["exp_state"].items()] + [""]
    lines += ["| decision | verdict | reason |", "|---|---|---|"]
    for v in vs:
        lines.append(f"| {v['id']} | **{v['verdict']}** | {v['reason']} |")
    for v in vs:
        d = v.get("details") or {}
        if v.get("kind") == "parity" and d.get("cases"):
            lines += ["", f"### {v['id']}: ratio per case (TFLOPS-eq, DGEMM-level configurations)", "",
                      "| case | class | A | B | A / B | + spread | note |", "|---|---|---|---|---|---|---|"]
            for r in d["cases"]:
                if "ratio" in r:
                    lines.append(f"| {r['case']} | {r['class']} | {r['a']:.3g} | {r['b']:.3g} | {r['ratio']:.3f} | {r['ratio_plus_spread']:.3f} | "
                                 f"{'excluded from the mean' if r['excluded'] else ('below floor' if r['case'] in d.get('below_floor', []) else '')} |")
                else:
                    lines.append(f"| {r['case']} | {r['class']} | | | | | {r.get('note', '')} |")
            lines += ["", f"Geometric mean {d['geomean']:.3f}; with every case {d['geomean_all_cases']:.3f}. By class: " +
                      ", ".join(f"{k} {x:.3f}" for k, x in d["class_geomean"].items())]
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------- tables
def run_summarize(script, args):
    r = subprocess.run([sys.executable, "-I", os.path.join(BENCH_DIR, script), *args], capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(r.stderr.strip()[-400:])
    return r.stdout


TABLE_SETS = [("summary.md", ["sweep_imd.jsonl", "baselines.jsonl"], "summarize.py", []),
              ("energy.md", ["energy.jsonl"], "summarize.py", []),
              ("cutensor_ws.md", ["cutensor_ws.jsonl"], "cutws_table.py", []),
              ("compare.md", ["compare.jsonl"], "summarize.py", ["--only-compare"]),
              ("compare_energy.md", ["compare_energy.jsonl"], "summarize.py", ["--only-compare"])]


def tables_of_job(job):
    """{name: text} for the tables bench/run_metrics.sh writes, regenerated from the job folder's JSONL files."""
    out = {}
    for name, files, script, flags in TABLE_SETS:
        have = [os.path.join(job, f) for f in files if os.path.exists(os.path.join(job, f)) and os.path.getsize(os.path.join(job, f)) > 0]
        if have:
            out[name] = run_summarize(script, flags + have)
    return out


def strip_provenance(text):
    """Drop the first '<!-- path ... -->' line: it names the folder the table was made in."""
    lines = text.splitlines()
    if lines and lines[0].startswith("<!--"):
        lines = lines[1:]
    return "\n".join(lines).rstrip() + "\n"


def golden(job):
    """Regenerate the tables of an issue-11 style job folder and diff them against the job's own .md files.
    Returns (checked_names, diff_lines)."""
    regen = tables_of_job(job)
    checked, diffs = [], []
    for name, text in regen.items():
        p = os.path.join(job, name)
        if not os.path.exists(p):
            continue
        checked.append(name)
        a = strip_provenance(rs_read(p))
        b = strip_provenance(text)
        if a != b:
            diffs += list(difflib.unified_diff(a.splitlines(), b.splitlines(), f"{name} (job)", f"{name} (regenerated)", lineterm=""))
    return checked, diffs


def rs_read(p):
    with open(p) as f:
        return f.read()


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    s = p.add_subparsers(dest="cmd", required=True)
    v = s.add_parser("verdicts")
    v.add_argument("folder")
    v.add_argument("--decisions")
    v.add_argument("--accept-unpinned", action="store_true")
    v.add_argument("--json")
    v.add_argument("--strict", action="store_true", help="exit 1 if any verdict is INVALID")
    t = s.add_parser("tables")
    t.add_argument("folder")
    t.add_argument("--out")
    b = s.add_parser("bitcheck")
    b.add_argument("folder")
    g = s.add_parser("golden")
    g.add_argument("jobs", nargs="+")
    a = p.parse_args(argv)
    if a.cmd == "verdicts":
        vs, info = verdicts(a.folder, a.decisions, a.accept_unpinned)
        sys.stdout.write(render(vs, info))
        if a.json:
            rs.write_json_atomic(a.json, vs)
        return 1 if a.strict and any(x["verdict"] == "INVALID" for x in vs) else 0
    if a.cmd == "tables":
        for name, text in tables_of_job(a.folder).items():
            if a.out:
                os.makedirs(a.out, exist_ok=True)
                with open(os.path.join(a.out, name), "w") as f:
                    f.write(text)
            else:
                print(f"<!-- {name} -->\n{text}")
        return 0
    if a.cmd == "bitcheck":
        n, bad = bitcheck(a.folder)
        for x in bad:
            print("NOT IDENTICAL", x)
        print(f"G5 bit identity: {n} record(s) checked, {len(bad)} problem(s)")
        return 1 if bad else 0
    bad = 0
    for job in a.jobs:
        checked, diffs = golden(job)
        print(f"{job}: {len(checked)} table file(s) regenerated ({', '.join(checked) or 'none'}): {'DIFFERENT' if diffs else 'identical'}")
        for line in diffs[:60]:
            print("   " + line)
        bad += bool(diffs) or not checked
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
