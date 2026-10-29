"""The result folder of one campaign job: how it is written (atomically) and how it is checked.

    <out>/
      MANIFEST.toml  DECISIONS.toml   copies of the inputs, as run
      LOCK.json                       what this folder belongs to: manifest and decisions hashes, GPU class, source commit
      env.json                        GPU, driver, tool versions, host, SLURM ids (written when the job starts)
      journal.jsonl                   append-only event log (one fsynced line per event)
      exp/<id>/                       the files of experiment <id>, then DONE.json as the LAST file written
      COMPLETE.json                   written last of all: the job reached its end

DONE.json is the commit marker of an experiment: it lists every file of the experiment folder with size and
sha256. An experiment folder without a valid DONE.json is incomplete and is wiped and redone on resume.
Everything is written with write-to-temp, fsync, rename, fsync of the directory, so a kill -9 leaves either the old
file or the new one, never half of one.
"""
import hashlib
import json
import os
import time

DONE = "DONE.json"
COMPLETE = "COMPLETE.json"
LOCK = "LOCK.json"


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fsync_dir(path):
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def write_atomic(path, data):
    """Write bytes (or str) to `path` so that readers and a kill -9 see the old or the new content only."""
    if isinstance(data, str):
        data = data.encode("utf-8")
    d = os.path.dirname(path) or "."
    tmp = os.path.join(d, f".{os.path.basename(path)}.tmp{os.getpid()}")
    with open(tmp, "wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    fsync_dir(d)


def write_json_atomic(path, obj):
    write_atomic(path, json.dumps(obj, indent=1, sort_keys=True) + "\n")


def append_journal(out, event, **kw):
    """One fsynced JSON line in journal.jsonl."""
    rec = {"t": time.time(), "event": event}
    rec.update(kw)
    p = os.path.join(out, "journal.jsonl")
    with open(p, "a") as f:
        f.write(json.dumps(rec, sort_keys=True) + "\n")
        f.flush()
        os.fsync(f.fileno())


def read_json(path):
    with open(path) as f:
        return json.load(f)


def exp_files(exp_dir):
    """Relative names of every file under an experiment folder except DONE.json and temp files."""
    out = []
    for root, _, files in os.walk(exp_dir):
        for fn in files:
            if fn == DONE or fn.startswith("."):
                continue
            out.append(os.path.relpath(os.path.join(root, fn), exp_dir))
    return sorted(out)


def file_table(exp_dir):
    return {n: {"bytes": os.path.getsize(os.path.join(exp_dir, n)), "sha256": sha256_file(os.path.join(exp_dir, n))}
            for n in exp_files(exp_dir)}


def verify_exp(exp_dir):
    """(state, problems). state: 'ok' or 'failed' (a valid DONE.json with that status), 'incomplete' (no DONE.json),
    'damaged' (DONE.json unreadable, or a listed file missing, resized or altered, or an unlisted file present)."""
    dp = os.path.join(exp_dir, DONE)
    if not os.path.exists(dp):
        return "incomplete", ["no DONE.json"]
    try:
        d = read_json(dp)
        files = d["files"]
        status = d["status"]
    except (ValueError, KeyError, TypeError, OSError) as e:
        return "damaged", [f"DONE.json unreadable: {e}"]
    problems = []
    for n, meta in files.items():
        p = os.path.join(exp_dir, n)
        if not os.path.exists(p):
            problems.append(f"{n}: missing")
        elif os.path.getsize(p) != meta["bytes"]:
            problems.append(f"{n}: {os.path.getsize(p)} bytes, DONE.json says {meta['bytes']} (truncated or edited)")
        elif sha256_file(p) != meta["sha256"]:
            problems.append(f"{n}: sha256 differs from DONE.json")
    for n in exp_files(exp_dir):
        if n not in files:
            problems.append(f"{n}: not listed in DONE.json")
    return ("damaged", problems) if problems else (status if status in ("ok", "failed") else "damaged", problems)


def read_jsonl(path):
    """Parsed records of a JSONL file; a line that does not parse is an error (a truncated file must not pass quietly)."""
    out = []
    with open(path) as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line))
                except ValueError as e:
                    raise ValueError(f"{path}:{i}: {e}") from e
    return out
