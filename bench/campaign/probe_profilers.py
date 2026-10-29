#!/usr/bin/env python3
"""Which profiling and inspection tools really work on this node? (issue 16: nsys, ncu, cuobjdump, nvdisasm, compute-sanitizer)

    probe_profilers.py BUILD_DIR OUT_DIR

Each tool is run on a tiny imd_ceiling launch (BUILD_DIR/imd_ceiling): nsys profile, ncu with one hardware metric, compute-sanitizer
memcheck, cuobjdump / nvdisasm on the binary. Writes OUT_DIR/profilers.json: per tool {found, path, version, works, seconds, rc,
note, output (last lines)}. "Found" is not "works": on a shared cluster the counters of ncu are often blocked (permission
errors), and nsys may lack a writable temp dir. The probe never fails the campaign; the exit code is 0.
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time

TIMEOUT = 180


def run(argv, cwd=None, timeout=TIMEOUT):
    t0 = time.time()
    try:
        r = subprocess.run(argv, capture_output=True, text=True, timeout=timeout, cwd=cwd)
        out = (r.stdout + r.stderr).strip().splitlines()
        return r.returncode, out, time.time() - t0
    except subprocess.TimeoutExpired:
        return None, ["timeout"], time.time() - t0
    except OSError as e:
        return None, [str(e)], time.time() - t0


def main(argv):
    build, outdir = os.path.abspath(argv[1]), os.path.abspath(argv[2])
    exe = os.path.join(build, "imd_ceiling")
    os.makedirs(outdir, exist_ok=True)
    res = {}
    tiny = [exe, "--variants", "sm80:128x128", "--shapes", "256x256x512", "--reps", "3", "--warmup", "1"]
    with tempfile.TemporaryDirectory() as tmp:
        def tool(name, version_flag="--version"):
            p = shutil.which(name)
            d = {"found": bool(p), "path": p}
            if p:
                rc, out, _ = run([p, version_flag], timeout=30)
                d["version"] = next((l.strip() for l in reversed(out) if l.strip()), "")
            res[name] = d
            return d

        d = tool("nsys")
        if d["found"]:
            rc, out, sec = run([d["path"], "profile", "-t", "cuda", "-o", os.path.join(tmp, "p"), "--force-overwrite", "true"] + tiny)
            rep = [f for f in os.listdir(tmp) if f.startswith("p.")]
            d.update(works=rc == 0 and bool(rep), seconds=round(sec, 1), rc=rc, output=out[-6:],
                     note="a report was written" if rep else "no report file")
        d = tool("ncu")
        if d["found"]:
            rc, out, sec = run([d["path"], "--metrics", "gpu__time_duration.sum", "--launch-count", "1", "--kernel-name", "regex:.*"] + tiny)
            blocked = any("ERR_NVGPUCTRPERM" in l or "permission" in l.lower() for l in out)
            got = any("gpu__time_duration" in l for l in out)
            d.update(works=rc == 0 and got, seconds=round(sec, 1), rc=rc, output=out[-8:],
                     note="counters blocked (ERR_NVGPUCTRPERM)" if blocked else ("metric read" if got else "no metric in the output"))
        d = tool("compute-sanitizer")
        if d["found"]:
            rc, out, sec = run([d["path"], "--tool", "memcheck", "--error-exitcode", "9"] + tiny)
            d.update(works=rc == 0, seconds=round(sec, 1), rc=rc, output=out[-6:],
                     note="memcheck clean" if rc == 0 else ("memcheck reported errors" if rc == 9 else "did not run"))
        for name in ("cuobjdump", "nvdisasm"):
            d = tool(name)
            if d["found"]:
                if name == "cuobjdump":
                    rc, out, sec = run([d["path"], "-sass", exe])
                    ok = rc == 0 and any("Function" in l for l in out)
                else:
                    cub = os.path.join(tmp, "x.cubin")
                    rc0, out0, _ = run([d["path"].replace("nvdisasm", "cuobjdump"), "-xelf", "all", exe], cwd=tmp)
                    files = [f for f in os.listdir(tmp) if f.endswith(".cubin")]
                    if files:
                        rc, out, sec = run([d["path"], "-c", os.path.join(tmp, files[0])])
                        ok = rc == 0 and len(out) > 10
                    else:
                        rc, out, sec, ok = rc0, out0, 0.0, False
                d.update(works=ok, seconds=round(sec, 1), rc=rc, output=out[:3] if ok else out[-6:])
    with open(os.path.join(outdir, "profilers.json"), "w") as f:
        json.dump(res, f, indent=1, sort_keys=True)
    for k, v in res.items():
        print(f"{k}: {'works' if v.get('works') else ('FOUND BUT FAILS' if v['found'] else 'not installed')}  {v.get('note', '')}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
