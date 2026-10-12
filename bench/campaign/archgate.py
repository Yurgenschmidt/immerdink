#!/usr/bin/env python3
"""Gate G2, the architecture gate: CPU only, no GPU. Compiles every library source for the target architectures, records
what ptxas says (registers, stack, spills) and which tensor-core instructions the SASS holds, and compares it with a
committed table. A change must be intended: it shows up here, offline, before a cluster run depends on it.

    archgate.py check  --build-dir DIR [--archs sm_89,sm_90a,sm_120a] [--table FILE] [--jobs 3]
    archgate.py update --build-dir DIR [...]        rewrite the table (after an intended change)
    archgate.py facts                               the toolchain facts the design relies on (wgmma needs sm_90a)

DIR is a configured immerdink build directory (its compile_commands.json supplies the exact flags of each source). Each
source is re-compiled with only the target architecture swapped (-cubin, -Xptxas -v). Checks against the table: a kernel's
registers / stack / spills (a spill that the table does not have is an error), the set of kernels, the count of each
tensor-core instruction family in the SASS; plus the table's rules: instruction families a (source, architecture) must
contain or must not contain (for example wgmma instructions only in sm_90a code, none in sm_89 or sm_120a code).
Compile time is a few seconds per source (measured 2.2 s for the heaviest K4 instantiation on sm_90a).
"""
import argparse
import concurrent.futures as cf
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
DEFAULT_TABLE = os.path.join(HERE, "arch_table.json")
DEFAULT_ARCHS = ("sm_89", "sm_90a", "sm_120a")
FAMILIES = ("IMMA", "HMMA", "HGMMA", "IGMMA", "QGMMA", "UTMALDG", "UTMASTG", "UTMACMDFLUSH", "SYNCS", "LDGSTS", "LDSM")
SASS_RE = re.compile(r"/\*[0-9a-f]{4,}\*/\s+(?:@!?U?P\w+\s+)?([A-Z][A-Z0-9_]*)")


def gencode(arch):
    n = arch[3:]
    return f"--generate-code=arch=compute_{n},code=sm_{n}"


def sources(build_dir):
    """{relative source path: compile command words} for the .cu / .cpp files of the library target in compile_commands.json."""
    cc = json.load(open(os.path.join(build_dir, "compile_commands.json")))
    out = {}
    src_root = os.path.join(ROOT, "src") + os.sep
    for e in cc:
        f = os.path.realpath(e["file"])
        if f.startswith(src_root) and f.endswith(".cu") and "/src/host/" not in f:
            out[os.path.relpath(f, ROOT)] = (shlex.split(e["command"]), e.get("directory", build_dir))
    return out


def compile_one(rel, words_dir, arch, tmp, nvcc_wrap):
    words, cwd = words_dir
    out, i = [], 0
    cubin = os.path.join(tmp, f"{rel.replace('/', '_')}.{arch}.cubin")
    while i < len(words):
        x = words[i]
        if x.startswith("--generate-code="):
            out.append(gencode(arch))
        elif x == "-c":
            out.append("-cubin")
        elif x == "-o":
            out += ["-o", cubin, "-Xptxas", "-v"]
            i += 1
        elif x.startswith("-Werror"):
            pass
        else:
            out.append(x)
        i += 1
    r = subprocess.run(nvcc_wrap + out, capture_output=True, text=True, cwd=cwd)
    if r.returncode != 0:
        return rel, arch, None, r.stderr[-1500:]
    kernels, cur = {}, None
    for line in r.stderr.splitlines():
        m = re.search(r"Compiling entry function '([^']+)' for", line)
        if m:
            cur = m.group(1)
            kernels[cur] = {}
            continue
        if cur is None:
            continue
        m = re.search(r"(\d+) bytes stack frame, (\d+) bytes spill stores, (\d+) bytes spill loads", line)
        if m:
            kernels[cur].update(stack=int(m.group(1)), spill_st=int(m.group(2)), spill_ld=int(m.group(3)))
        m = re.search(r"Used (\d+) registers", line)
        if m:
            kernels[cur]["regs"] = int(m.group(1))
        m = re.search(r"(\d+) bytes smem", line)
        if m:
            kernels[cur]["smem"] = int(m.group(1))
    sass = subprocess.run(["cuobjdump", "-sass", cubin], capture_output=True, text=True)
    counts = {k: 0 for k in FAMILIES}
    for line in sass.stdout.splitlines():
        m = SASS_RE.search(line)
        if m and m.group(1) in counts:
            counts[m.group(1)] += 1
    os.remove(cubin)
    return rel, arch, {"kernels": dict(sorted(kernels.items())), "sass": {k: v for k, v in counts.items() if v}}, None


def collect(build_dir, archs, jobs):
    srcs = sources(build_dir)
    if not srcs:
        raise SystemExit(f"no library sources in {build_dir}/compile_commands.json")
    nvcc_wrap = ["nice", "-n", "19"]
    memcap = os.path.join(ROOT, "tools", "memcap.sh")
    if os.path.exists(memcap):
        nvcc_wrap.append(memcap)
    table, errors = {a: {} for a in archs}, []
    with tempfile.TemporaryDirectory() as tmp, cf.ThreadPoolExecutor(max_workers=jobs) as ex:
        futs = [ex.submit(compile_one, rel, wd, a, tmp, nvcc_wrap) for rel, wd in sorted(srcs.items()) for a in archs]
        for f in futs:
            rel, arch, data, err = f.result()
            if err:
                errors.append(f"{rel} for {arch}: {err}")
            else:
                table[arch][rel] = data
    return table, errors


DEFAULT_RULES = {
    # family -> {arch -> [source globs that must / must not contain it]}; extended when a wgmma source exists
    "must_contain": {"sm_89": {"src/k4/k4_cfg0.cu": ["IMMA"]}, "sm_90a": {"src/k4/k4_cfg0.cu": ["IMMA"]},
                     "sm_120a": {"src/k4/k4_cfg0.cu": ["IMMA"]}},
    "must_not_contain": {"sm_89": ["HGMMA", "IGMMA", "QGMMA", "UTMALDG"], "sm_120a": ["HGMMA", "IGMMA", "QGMMA"]},
}


def compare(new, old, rules):
    """List of differences between a fresh table and the committed one (empty = identical and rules hold)."""
    diffs = []
    for arch, files in new.items():
        for rel, d in files.items():
            o = old.get("table", {}).get(arch, {}).get(rel)
            if o is None:
                diffs.append(f"{arch} {rel}: not in the committed table")
                continue
            for k, v in d["kernels"].items():
                ok = o["kernels"].get(k)
                short = k if len(k) < 90 else k[:60] + "..." + k[-25:]
                if ok is None:
                    diffs.append(f"{arch} {rel}: new kernel {short}")
                    continue
                if v.get("spill_st", 0) or v.get("spill_ld", 0) or v.get("stack", 0):
                    if (v.get("spill_st", 0), v.get("spill_ld", 0), v.get("stack", 0)) != \
                            (ok.get("spill_st", 0), ok.get("spill_ld", 0), ok.get("stack", 0)):
                        diffs.append(f"{arch} {rel}: {short} SPILLS/stack {v.get('stack', 0)}/{v.get('spill_st', 0)}/{v.get('spill_ld', 0)} "
                                     f"(table {ok.get('stack', 0)}/{ok.get('spill_st', 0)}/{ok.get('spill_ld', 0)})")
                if v != ok and not any("SPILLS" in x and short in x for x in diffs):
                    diffs.append(f"{arch} {rel}: {short} changed: {ok} -> {v}")
            for k in o["kernels"]:
                if k not in d["kernels"]:
                    diffs.append(f"{arch} {rel}: kernel gone: {k[:80]}")
            if d["sass"] != o["sass"]:
                diffs.append(f"{arch} {rel}: SASS instruction families changed: {o['sass']} -> {d['sass']}")
    for arch, per in rules.get("must_contain", {}).items():
        for rel, fams in per.items():
            got = new.get(arch, {}).get(rel)
            for fam in fams:
                if got is not None and not got["sass"].get(fam):
                    diffs.append(f"rule: {arch} {rel} must contain {fam} instructions")
    for arch, fams in rules.get("must_not_contain", {}).items():
        for rel, d in new.get(arch, {}).items():
            for fam in fams:
                if d["sass"].get(fam):
                    diffs.append(f"rule: {arch} {rel} must not contain {fam} instructions")
    return diffs


def facts():
    """Toolchain facts the design relies on, checked by assembling a one-instruction kernel. Returns (ok, lines)."""
    expect = {"sm_89": False, "sm_90": False, "sm_90a": True, "sm_100a": False, "sm_120": False, "sm_120a": False}
    nvcc = shutil.which("nvcc")
    if not nvcc:
        return False, ["no nvcc on PATH"]
    lines, ok = [], True
    with tempfile.TemporaryDirectory() as tmp:
        src = os.path.join(tmp, "w.cu")
        with open(src, "w") as f:
            f.write('extern "C" __global__ void k(int* o) { asm volatile("wgmma.fence.sync.aligned;" ::: "memory"); o[threadIdx.x] = 1; }\n')
        for arch, want in expect.items():
            r = subprocess.run([nvcc, "-cubin", f"-arch={arch}", "-o", os.path.join(tmp, "w.cubin"), src], capture_output=True, text=True)
            got = r.returncode == 0
            lines.append(f"wgmma.fence assembles for {arch}: {got} (expected {want})" + ("" if got == want else "   <-- DIFFERENT"))
            ok &= got == want
    return ok, lines


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("cmd", choices=["check", "update", "facts"])
    p.add_argument("--build-dir")
    p.add_argument("--archs", default=",".join(DEFAULT_ARCHS))
    p.add_argument("--table", default=DEFAULT_TABLE)
    p.add_argument("--jobs", type=int, default=3)
    a = p.parse_args(argv)
    if a.cmd == "facts":
        ok, lines = facts()
        print("\n".join(lines))
        return 0 if ok else 1
    if not a.build_dir:
        p.error("--build-dir is required")
    archs = a.archs.split(",")
    new, errors = collect(os.path.abspath(a.build_dir), archs, a.jobs)
    for e in errors:
        print("COMPILE ERROR", e, file=sys.stderr)
    if errors:
        return 1
    if a.cmd == "update":
        old = json.load(open(a.table)) if os.path.exists(a.table) else {}
        old_table = old.get("table", {})
        old_table.update(new)
        out = {"note": "Written by archgate.py update (nvcc/ptxas as in the env of the day). Compare: archgate.py check.",
               "rules": old.get("rules", DEFAULT_RULES), "table": old_table}
        with open(a.table, "w") as f:
            json.dump(out, f, indent=1, sort_keys=True)
            f.write("\n")
        print(f"wrote {a.table}: {sum(len(v) for v in new.values())} (arch, source) entries")
        return 0
    old = json.load(open(a.table))
    diffs = compare(new, old, old.get("rules", DEFAULT_RULES))
    for d in diffs:
        print("DIFF", d)
    print(f"G2 architecture gate: {len(diffs)} difference(s) from {os.path.relpath(a.table, ROOT)} for {', '.join(archs)}")
    return 1 if diffs else 0


if __name__ == "__main__":
    sys.exit(main())
