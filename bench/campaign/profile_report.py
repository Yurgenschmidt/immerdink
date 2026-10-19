#!/usr/bin/env python3
"""Report of an nsys profile folder (bench/profile_nsys.sh output, issue 16): where do GEMMul8, cuBLAS FP64 emulation and immerdink spend GPU time?

    profile_report.py PROFILE_DIR [--md OUT]

PROFILE_DIR holds index.json and <tag>/summary.json. Per run (tag): the share of the GPU activity time in INT8 GEMM kernels, in conversion
(pack/CRT-like) kernels and in everything else, the number of INT8 GEMM launches per call and the kernel names that carry them. The
number of INT8 GEMM launches per call is the evidence for "is it Ozaki scheme II?": scheme II runs one INT8 product per modulus
(about N launches or one batched launch of N), scheme I one per pair of slices (a triangle of about s(s+1)/2). The report only counts and
names; the conclusion is the reader's, from the numbers next to the library's documentation.
"""
import json
import os
import sys


def load(d):
    out = []
    idx = os.path.join(d, "index.json")
    entries = json.load(open(idx)) if os.path.exists(idx) else []
    for e in entries:
        p = os.path.join(d, e["summary"]) if "summary" in e else os.path.join(d, e["tag"], "summary.json")
        if os.path.exists(p):
            out.append(json.load(open(p)))
    return out


def row(s):
    ks = s.get("kernels") or []
    tot = sum(k["time_ms"] for k in ks) or 0.0
    cat = {}
    for k in ks:
        cat[k["category"]] = cat.get(k["category"], 0.0) + k["time_ms"]
    i8 = [k for k in ks if k["category"] == "int8_gemm"]
    launches = sum(k.get("count", 0) for k in i8)
    names = sorted({k["name"].split("<")[0].split("(")[0][-60:] for k in i8})
    return {"tag": s["tag"], "method": s.get("method"), "N": s.get("N"), "variant": s.get("variant"), "total_ms": tot,
            "int8_pct": 100 * cat.get("int8_gemm", 0.0) / tot if tot else None,
            "conv_pct": 100 * cat.get("conversion_pack_crt", 0.0) / tot if tot else None,
            "other_pct": max(0.0, 100 * (tot - cat.get("int8_gemm", 0.0) - cat.get("conversion_pack_crt", 0.0)) / tot) if tot else None,
            "int8_launches": launches, "int8_kernels": names, "engaged_int8": s.get("engaged_int8"), "notes": s.get("notes")}


def markdown(rows):
    f = lambda x: "—" if x is None else f"{x:.1f}"
    lines = ["| run | GPU time per call (ms) | INT8 GEMM % | conversion % | other % | INT8 GEMM launches per call | INT8 kernel names |",
             "|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['tag']} | {r['total_ms']:.3g} | {f(r['int8_pct'])} | {f(r['conv_pct'])} | {f(r['other_pct'])} | "
                     f"{r['int8_launches']:g} | {', '.join(r['int8_kernels']) or '—'} |")
    return "\n".join(lines)


def main(argv):
    if len(argv) < 2:
        print(__doc__)
        return 2
    rows = [row(s) for s in load(argv[1])]
    text = markdown(rows)
    print(text)
    if "--md" in argv:
        with open(argv[argv.index("--md") + 1], "w") as f:
            f.write(text + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
