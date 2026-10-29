#!/usr/bin/env python3
# usage: sass_hash.py <obj.o> [name-regex] -> one "hash count name" line per kernel, then "TOTAL <hash> <kernels>".
# The hash is over the instruction stream of a kernel (cuobjdump -sass) with the addresses, the encodings and the kernel's name
# removed, so two builds whose kernels have the same machine code hash alike even if a template parameter changed a symbol.
# TOTAL is the hash of the sorted per-kernel hashes (the kernel multiset). Issue 23: "SASS identical when a knob is off".
import hashlib, re, subprocess, sys
obj = sys.argv[1]
pat = re.compile(sys.argv[2]) if len(sys.argv) > 2 else None
out = subprocess.run(["cuobjdump", "-sass", obj], capture_output=True, text=True, check=True).stdout
cur, body, kern = None, [], []
def flush():
    if cur is not None:
        text = "\n".join(body)
        kern.append((hashlib.sha256(text.encode()).hexdigest()[:16], len(body), cur))
for line in out.splitlines():
    m = re.match(r"\s+Function : (\S+)", line)
    if m:
        flush(); cur, body = m.group(1), []; continue
    m = re.match(r"\s+/\*[0-9a-f]{4,}\*/\s+(\S.*?);", line)
    if m and cur: body.append(re.sub(r"\s+", " ", m.group(1)))
flush()
names = [k[2] for k in kern]
dem = subprocess.run(["c++filt"], input="\n".join(names), capture_output=True, text=True).stdout.splitlines()
sel = [(h, n, d) for (h, n, _), d in zip(kern, dem) if pat is None or pat.search(d)]
for h, n, d in sorted(sel, key=lambda x: x[2]): print(h, n, re.sub(r"imd::detail::\(anonymous namespace\)::|imd::detail::", "", d))
print("TOTAL", hashlib.sha256("\n".join(sorted(h for h, _, _ in sel)).encode()).hexdigest()[:16], len(sel))
