#!/usr/bin/env python3
# usage: sass_counts.py <obj.o> -> "count name" lines (NOP-free static instruction count per kernel)
import subprocess, sys, re
obj = sys.argv[1]
out = subprocess.run(["cuobjdump", "-sass", obj], capture_output=True, text=True, check=True).stdout
cur = None; cnt = {}
for line in out.splitlines():
    m = re.match(r"\s+Function : (\S+)", line)
    if m: cur = m.group(1); cnt[cur] = 0; continue
    m = re.match(r"\s+/\*[0-9a-f]{4,}\*/\s+(\S.*?);", line)
    if m and cur:
        op = m.group(1).split()
        if op[0].startswith("@"): op = op[1:]
        if op[0] != "NOP": cnt[cur] += 1
names = list(cnt)
dem = subprocess.run(["c++filt"], input="\n".join(names), capture_output=True, text=True).stdout.splitlines()
for n, d in zip(names, dem): print(cnt[n], re.sub(r"imd::detail::\(anonymous namespace\)::|imd::detail::", "", d))
