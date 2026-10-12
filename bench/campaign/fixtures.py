"""Shared builders for the kit's tests: a throw-away git source tree, a manifest and its decisions file."""
import hashlib
import json
import os
import subprocess
import sys
import textwrap

HERE = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable


def rd(path, mode="r"):
    with open(path, mode) as f:
        return f.read()


def wr(path, text, mode="w"):
    with open(path, mode) as f:
        f.write(text)


def git(repo, *args):
    return subprocess.run(["git", "-C", repo, "-c", "user.email=t@t", "-c", "user.name=t", *args],
                          check=True, capture_output=True, text=True).stdout.strip()


def make_src(tmp):
    """A tiny repository with bench/sweep_x.txt; returns (path, commit)."""
    src = os.path.join(tmp, "src")
    os.makedirs(os.path.join(src, "bench"))
    wr(os.path.join(src, "bench", "sweep_x.txt"), "# empty sweep\n")
    git(src, "init", "-q")
    git(src, "add", "-A")
    git(src, "commit", "-q", "-m", "init")
    return src, git(src, "rev-parse", "HEAD")


DECISIONS = """\
[[decision]]
id = "ceiling"
kind = "threshold"
requires = ["{exp}"]
exp = "{exp}"
file = "out.jsonl"
where = {method = "ceil"}
field = "tops"
aggregate = "max"
go_at_least = 1000.0
nogo_below = 500.0
"""


def make_manifest(tmp, commit, experiments, decisions=DECISIONS, budget=60, wall=120, name="R0", extra_campaign=""):
    """Writes DECISIONS + manifest next to each other in tmp; returns the manifest path. `{exp}` in the decisions text
    becomes the id of the first experiment that is not a probe."""
    first = next(e["id"] for e in experiments if e.get("kind") != "probe")
    decisions = decisions.replace("{exp}", first)
    dpath = os.path.join(tmp, f"{name}.decisions.toml")
    wr(dpath, decisions)
    dsha = hashlib.sha256(rd(dpath, "rb")).hexdigest()
    head = textwrap.dedent(f"""\
        [campaign]
        id = "{name}"
        title = "test"
        immerdink_commit = "{commit}"
        cutlass_commit = "{'c' * 40}"
        wall_limit_min = {wall}
        margin = 0.25
        decisions = "{name}.decisions.toml"
        decisions_sha256 = "{dsha}"
        {extra_campaign}

        [class.fake]
        gpu = "FAKEGPU"
        budget_min = {budget}
        """)
    body = ""
    for e in experiments:
        body += "\n[[experiment]]\n" + "".join(f"{k} = {toml_val(v)}\n" for k, v in e.items() if not isinstance(v, dict))
        for k, v in e.items():
            if isinstance(v, dict):
                body += f"[experiment.{k}]\n" + "".join(f"{kk} = {toml_val(vv)}\n" for kk, vv in v.items())
    mpath = os.path.join(tmp, f"{name}.toml")
    wr(mpath, head + body)
    return mpath


def toml_val(v):
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int, float)):
        return repr(v)
    if isinstance(v, str):
        return json.dumps(v)
    if isinstance(v, list):
        return "[" + ", ".join(toml_val(x) for x in v) + "]"
    raise TypeError(v)


def py_exp(eid, code, prio="P0", est=0.05, outputs=("out.jsonl",), **kw):
    """A cmd experiment that runs `code` (python source) in its folder."""
    e = {"id": eid, "kind": "cmd", "priority": prio, "est_min": est, "argv": [PY, "-I", "-c", textwrap.dedent(code)],
         "outputs": list(outputs)}
    e.update(kw)
    return e


WRITE_OK = """
import json
open('out.jsonl','w').write(json.dumps({'method':'ceil','tops':1200.0,'status':'ok'})+'\\n')
"""
