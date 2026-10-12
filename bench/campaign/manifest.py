"""Campaign manifest: load, validate (gate G1) and hash.

A manifest is one TOML file per run. It lists the experiments by priority, the source commits that
must be checked out, the time budget per GPU class and the decisions file (formulas fixed before the
data exist, pinned by its sha256). The runner executes it, the analyzer judges the result against it.

    [campaign]
    id = "R1"                         title = "Measure"
    immerdink_commit = "<40 hex>"     cutlass_commit = "<40 hex>"     gemmul8_commit = "<40 hex>" (optional)
    wall_limit_min = 240              margin = 0.25
    decisions = "R1.decisions.toml"   decisions_sha256 = "<64 hex>"

    [class.h100]                      # one table per GPU class: name regex, soft budget in minutes
    gpu = "H100"                      budget_min = 85

    [[experiment]]
    id = "probe.env"   kind = "probe"   priority = "P0"   est_min = 1   probe = "env"
    id = "cmp.main"    kind = "sweep"   priority = "P0"   est_min = 30  file = "bench/sweep_compare.txt"
    id = "x.run"       kind = "cmd"     priority = "P1"   est_min = 5   argv = ["{build}/imd_bench", ...]

`lint` implements gate G1. `python3 manifest.py lint MANIFEST [--src DIR]` prints the findings and exits 1 on any error.
"""
import hashlib
import os
import re
import subprocess
import sys
import tomllib

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # also under python -I

PRIORITIES = ("P0", "P1", "P2")
KINDS = ("probe", "cmd", "sweep")
PROBES = ("env", "clocks", "tools", "limits")
ID_RE = re.compile(r"^[a-z0-9][a-z0-9._-]{0,62}$")
HEX40 = re.compile(r"^[0-9a-f]{40}$")
HEX64 = re.compile(r"^[0-9a-f]{64}$")
PLACEHOLDER = re.compile(r"\b(TBD|TODO|FIXME|XXX)\b|\?\?")
DEFAULT_MARGIN = 0.25
PLACEHOLDERS_IN_ARGV = ("src", "build", "out", "class", "exp")


class Finding:
    def __init__(self, level, where, msg):
        self.level, self.where, self.msg = level, where, msg

    def __str__(self):
        return f"{self.level}: {self.where}: {self.msg}"


class Manifest:
    def __init__(self, path, data, raw):
        self.path = os.path.abspath(path)
        self.data = data
        self.raw = raw
        self.sha256 = hashlib.sha256(raw).hexdigest()
        c = data.get("campaign", {})
        self.id = c.get("id")
        self.margin = c.get("margin", DEFAULT_MARGIN)
        self.wall_limit_min = c.get("wall_limit_min")
        self.experiments = list(data.get("experiment", []))
        self.classes = data.get("class", {})

    def exp(self, eid):
        for e in self.experiments:
            if e.get("id") == eid:
                return e
        return None

    def classify(self, gpu_name):
        """The class key whose `gpu` regex matches the device name (first in file order), else None."""
        for key, c in self.classes.items():
            if re.search(c.get("gpu", "$^"), gpu_name or "", re.I):
                return key
        return None

    def plan(self, cls):
        """Experiments applying to `cls`, in execution order: priority, then non-risky before risky, then file order."""
        sel = [(PRIORITIES.index(e["priority"]), bool(e.get("risky", False)), i, e)
               for i, e in enumerate(self.experiments)
               if cls in e.get("classes", list(self.classes))]
        return [e for _, _, _, e in sorted(sel, key=lambda t: t[:3])]

    def decisions_path(self):
        return os.path.join(os.path.dirname(self.path), self.data["campaign"]["decisions"])


def load(path):
    with open(path, "rb") as f:
        raw = f.read()
    return Manifest(path, tomllib.loads(raw.decode("utf-8")), raw)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _walk_strings(x, where=""):
    if isinstance(x, str):
        yield where, x
    elif isinstance(x, dict):
        for k, v in x.items():
            yield from _walk_strings(v, f"{where}.{k}" if where else k)
    elif isinstance(x, list):
        for i, v in enumerate(x):
            yield from _walk_strings(v, f"{where}[{i}]")


def _git_has(src, commit, path=None):
    spec = commit if path is None else f"{commit}:{path}"
    r = subprocess.run(["git", "-C", src, "cat-file", "-e", spec], capture_output=True)
    return r.returncode == 0


def _rank(e):
    p = e.get("priority")
    return PRIORITIES.index(p) if p in PRIORITIES else len(PRIORITIES)


def lint(m, src=None, check_decisions=True):
    """Gate G1. Returns a list of Finding; an error-level finding blocks any submission."""
    F = []

    def err(where, msg):
        F.append(Finding("error", where, msg))

    def warn(where, msg):
        F.append(Finding("warn", where, msg))

    d = m.data
    c = d.get("campaign")
    if not isinstance(c, dict):
        err("campaign", "missing [campaign] table")
        return F
    for where, s in _walk_strings(d):
        if PLACEHOLDER.search(s):
            err(where, f"placeholder text in a manifest value: {s!r}")
    if not isinstance(c.get("id"), str) or not ID_RE.match(c.get("id", "").lower()):
        err("campaign.id", "needs an id of letters, digits, '.', '_' or '-'")
    for key in ("immerdink_commit", "cutlass_commit"):
        if not HEX40.match(str(c.get(key, ""))):
            err(f"campaign.{key}", "must be a full 40-hex commit (pinned)")
    if "gemmul8_commit" in c and not HEX40.match(str(c["gemmul8_commit"])):
        err("campaign.gemmul8_commit", "must be a full 40-hex commit")
    wall = c.get("wall_limit_min")
    if not isinstance(wall, (int, float)) or wall <= 0:
        err("campaign.wall_limit_min", "needs a positive number of minutes")
    margin = c.get("margin", DEFAULT_MARGIN)
    if not isinstance(margin, (int, float)) or margin < 0:
        err("campaign.margin", "must be a number >= 0")
        margin = DEFAULT_MARGIN
    if not isinstance(c.get("decisions"), str):
        err("campaign.decisions", "needs the path of the decisions file (relative to the manifest)")
    if not HEX64.match(str(c.get("decisions_sha256", ""))):
        err("campaign.decisions_sha256", "needs the sha256 of the decisions file (64 hex)")

    if not m.classes:
        err("class", "needs at least one [class.<name>] table")
    for key, cl in m.classes.items():
        try:
            re.compile(cl.get("gpu", ""))
        except re.error as e:
            err(f"class.{key}.gpu", f"bad regex: {e}")
        if not cl.get("gpu"):
            err(f"class.{key}.gpu", "needs a GPU-name regex")
        b = cl.get("budget_min")
        if not isinstance(b, (int, float)) or b <= 0:
            err(f"class.{key}.budget_min", "needs a positive soft budget in minutes")
        elif isinstance(wall, (int, float)) and b > wall:
            err(f"class.{key}.budget_min", f"budget {b} exceeds wall_limit_min {wall}")

    seen = {}
    order = {}
    for i, e in enumerate(m.experiments):
        w = f"experiment[{i}]"
        eid = e.get("id")
        if not isinstance(eid, str) or not ID_RE.match(eid):
            err(w, f"bad id {eid!r}")
            continue
        w = f"experiment {eid}"
        if eid in seen:
            err(w, "duplicate id")
        seen[eid] = e
        order[eid] = i
        if e.get("kind") not in KINDS:
            err(w, f"kind must be one of {KINDS}")
        if e.get("priority") not in PRIORITIES:
            err(w, f"priority must be one of {PRIORITIES}")
        est = e.get("est_min")
        if not isinstance(est, (int, float)) or est <= 0:
            err(w, "needs est_min > 0 (minutes)")
        to = e.get("timeout_min", None)
        if to is not None and (not isinstance(to, (int, float)) or (isinstance(est, (int, float)) and to < est)):
            err(w, "timeout_min must be a number >= est_min")
        for cl in e.get("classes", []):
            if cl not in m.classes:
                err(w, f"unknown class {cl!r}")
        if e.get("kind") == "probe" and e.get("probe") not in PROBES:
            err(w, f"probe must be one of {PROBES}")
        if e.get("kind") == "cmd":
            av = e.get("argv")
            if not (isinstance(av, list) and av and all(isinstance(a, str) for a in av)):
                err(w, "cmd needs argv = [list of strings]")
            else:
                for a in av:
                    for ph in re.findall(r"\{([a-z_]+)\}", a):
                        if ph not in PLACEHOLDERS_IN_ARGV:
                            err(w, f"unknown placeholder {{{ph}}} in argv (known: {PLACEHOLDERS_IN_ARGV})")
        if e.get("kind") == "sweep":
            if not isinstance(e.get("file"), str):
                err(w, "sweep needs file = path of the sweep text in the source tree")
            elif src and HEX40.match(str(c.get("immerdink_commit", ""))):
                if not _git_has(src, c["immerdink_commit"], e["file"]):
                    err(w, f"sweep file {e['file']} does not exist at the pinned commit")
        outs = e.get("outputs", [])
        if e.get("kind") != "probe" and not (isinstance(outs, list) and outs and all(isinstance(o, str) for o in outs)):
            err(w, "needs outputs = [files the experiment must leave in its folder]")
    for eid, e in seen.items():
        for n in e.get("needs", []):
            if n not in seen:
                err(f"experiment {eid}", f"needs unknown experiment {n!r}")
            elif _rank(seen[n]) > _rank(e):
                err(f"experiment {eid}", f"needs {n!r}, which has a lower priority (it may not run)")
            elif order[n] >= order[eid] and seen[n].get("priority") == e.get("priority"):
                err(f"experiment {eid}", f"needs {n!r}, which is listed after it")

    # budget: P0 of each class, with the margin, inside the class budget; everything inside the wall limit
    for key, cl in m.classes.items():
        mine = [e for e in m.experiments if key in e.get("classes", list(m.classes))
                and isinstance(e.get("est_min"), (int, float))]
        p0 = sum(e["est_min"] for e in mine if e.get("priority") == "P0")
        allp = sum(e["est_min"] for e in mine)
        b = cl.get("budget_min")
        if isinstance(b, (int, float)):
            if p0 * (1.0 + margin) > b:
                err(f"class.{key}", f"P0 estimate {p0:g} min x (1 + {margin:g}) = {p0 * (1 + margin):.0f} exceeds the budget {b:g} min")
            if allp > b:
                warn(f"class.{key}", f"all priorities sum to {allp:g} min, over the budget {b:g} min: P1/P2 run only with leftover time")
        if not mine:
            warn(f"class.{key}", "no experiment applies to this class")

    if check_decisions and isinstance(c.get("decisions"), str):
        dp = m.decisions_path()
        if not os.path.exists(dp):
            err("campaign.decisions", f"{dp} does not exist")
        else:
            if HEX64.match(str(c.get("decisions_sha256", ""))) and sha256_file(dp) != c["decisions_sha256"]:
                err("campaign.decisions_sha256", "does not match the decisions file (edited after the manifest was written?)")
            import decisions
            for f in decisions.lint(dp, seen):
                F.append(f)
    if src and HEX40.match(str(c.get("immerdink_commit", ""))) and not _git_has(src, c["immerdink_commit"]):
        err("campaign.immerdink_commit", f"commit not found in {src}")
    return F


def main(argv):
    if len(argv) < 3 or argv[1] != "lint":
        print(__doc__)
        return 2
    src = None
    if "--src" in argv:
        src = argv[argv.index("--src") + 1]
    m = load(argv[2])
    F = lint(m, src)
    for f in F:
        print(f)
    n = sum(f.level == "error" for f in F)
    print(f"G1 manifest lint: {n} error(s), {len(F) - n} warning(s)")
    return 1 if n else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
