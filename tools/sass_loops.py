#!/usr/bin/env python3
# usage: sass_loops.py <obj> <opcode-regex> [name-filter] : for each kernel, the innermost backward-branch range that contains the opcode;
# prints "<instructions in the range> <opcode count> <kernel>"
import subprocess, sys, re
obj, opre = sys.argv[1], re.compile(sys.argv[2]); nf = re.compile(sys.argv[3]) if len(sys.argv) > 3 else None
out = subprocess.run(["cuobjdump", "-sass", obj], capture_output=True, text=True, check=True).stdout
funcs = {}; cur = None
for line in out.splitlines():
    m = re.match(r"\s+Function : (\S+)", line)
    if m: cur = m.group(1); funcs[cur] = []; continue
    m = re.match(r"\s+/\*([0-9a-f]{4,})\*/\s+(\S.*?);", line)
    if m and cur: funcs[cur].append((int(m.group(1), 16), m.group(2)))
names = list(funcs)
dem = subprocess.run(["c++filt"], input="\n".join(names), capture_output=True, text=True).stdout.splitlines()
for n, d in zip(names, dem):
    d = re.sub(r"imd::detail::\(anonymous namespace\)::|imd::detail::", "", d)
    if nf and not nf.search(d): continue
    ins = funcs[n]; best = None
    for a, t in ins:
        m = re.match(r"(?:@!?U?P\w+\s+)?BRA\s+(?:\S+,\s*)?0x([0-9a-f]+)", t)
        if not m: continue
        tgt = int(m.group(1), 16)
        if tgt >= a: continue
        body = [x for x in ins if tgt <= x[0] <= a]
        c = sum(1 for x in body if opre.search(x[1]))
        if c == 0: continue
        size = len(body)
        if best is None or size < best[0]: best = (size, c)
    print(best[0] if best else 0, best[1] if best else 0, d)
