#!/usr/bin/env python3
"""Every kernel the library launches is covered by the device-status check (issue 18, docs/NOTES-18.md D4).

    python3 tests/check_kernel_sites.py [repo root]

imd_device_info loads and classifies a fixed list of kernels (src/contract.cu kernel_occupancy_w /
extra_kernel_versions, src/k4/k4_cfg.cuh versions). A launch added without extending that list would
fail at its first use on a GPU the check did not cover. This finds every `name<...><<<` site in src/
and requires the name to appear in the list. It also checks the K4 limits that new configurations must
fit (4 configurations, the 256-byte parameter block). CPU only, standard library.
"""
import os
import re
import sys

root = sys.argv[1] if len(sys.argv) > 1 else os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def read(rel):
    with open(os.path.join(root, rel)) as f:
        return f.read()


launched = {}
for d, _, files in os.walk(os.path.join(root, "src")):
    for fn in files:
        if fn.endswith((".cu", ".cuh", ".cpp")):
            rel = os.path.relpath(os.path.join(d, fn), root)
            if re.fullmatch(r"src/k4/hopper/rung\d+\.cu", rel):
                continue  # standalone Hopper study programs (tools/build_rung*.sh), not part of libimmerdink
            text = re.sub(r"//[^\n]*", "", read(rel))
            for m in re.finditer(r"([A-Za-z_][A-Za-z_0-9:]*)\s*(?:<[^<>;{}]*(?:<[^<>;{}]*>[^<>;{}]*)*>)?\s*<<<", text):
                launched.setdefault(m.group(1).split("::")[-1], set()).add(rel)

contract = read("src/contract.cu")
# the checked list: the kernels named in occupancy_of(...) / versions_of(...) calls and the K5 template
checked = {m.group(1) for m in re.finditer(r"(?:occupancy_of|versions_of)\(\s*([A-Za-z_0-9]+)", contract)}
checked |= {"k4_kernel"} if "k4_kernel<G, false>" in read("src/k4/k4_cfg.cuh") and \
    "cudaFuncGetAttributes(&a, store)" in read("src/k4/k4_cfg.cuh") else set()

# test-only kernels (names test_*, compiled under IMD_BUILD_TEST_HOOKS only, e.g. test_crt_elem_kernel) are not part of the shipped library
missing = {n: sorted(f) for n, f in launched.items() if n not in checked and not n.startswith("test_")}
fail = False
if missing:
    fail = True
    print("FAIL  launched but not covered by the device-status check:")
    for n, f in sorted(missing.items()):
        print(f"      {n}  ({', '.join(f)})")
# K4 limits (D6)
ops = read("src/k4/k4_ops.hpp")
if not re.search(r"kK4BlobBytes\s*=\s*256\s*;", ops):
    fail = True
    print("FAIL  the K4 parameter block is no longer 256 bytes (docs/NOTES-18.md D6, docs/SUPPORT.md)")
host = read("include/immerdink/immerdink_host.h")
if not re.search(r"IMD_K4_CONFIG_COUNT\s*=\s*4", host):
    fail = True
    print("FAIL  the number of K4 configurations is no longer 4 (docs/NOTES-18.md D6, docs/SUPPORT.md)")
if not fail:
    print(f"PASS  {len(launched)} launched kernel names, all covered; K4 limits: 4 configurations, 256-byte block")
sys.exit(1 if fail else 0)
