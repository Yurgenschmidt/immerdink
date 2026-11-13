#!/usr/bin/env python3
"""Table of the cuTENSOR emulation workspace ladder (bench/sweep_cutensor_ws.txt):
  bench/cutws_table.py cutensor_ws.jsonl
Per case, preference (ct_ws) and plan limit (ct_ws_mib): native and emulated TFLOPS-eq, the workspace the emulated plan
requires (ws_req), whether the INT8 emulation ran (emu_engaged: kernel names of one graph-captured call, bench_schema 2),
and the native twin planned at the same preference and limit (identical_to_native_same_ws, its TFLOPS-eq): a larger
workspace can also make cuTENSOR pick another NATIVE algorithm, which is not emulation. Records without bench_schema 2
(capped at cuTENSOR's 32 MiB estimate) never engaged. If no record has bench_schema 2 the old (schema-1) table is printed
unchanged, so the tables written by earlier campaigns regenerate byte for byte (bench/campaign/test_analyze.py Golden); its
footnote's claim that an RTX 4090 cannot emulate was later shown wrong (the 32 MiB cap, not the GPU, kept it native).
Python 3.6+, stdlib only."""
import json, sys


def main():
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    recs = []
    for line in open(sys.argv[1]):
        line = line.strip()
        if line.startswith("{"):
            try:
                r = json.loads(line)
            except ValueError:
                continue
            if r.get("method") in ("cutensor", "cutensor-emu"):
                recs.append(r)
    if not any(r.get("bench_schema", 1) >= 2 for r in recs):
        legacy(recs)
        return
    native = {}
    emu = []
    for r in recs:
        if r["method"] == "cutensor":
            native.setdefault(r.get("case"), r)
        else:
            emu.append(r)

    def ok(x):
        return x and x.get("status") == "ok"

    def tf(x):
        return "%.2f" % x["tflops_eq"] if ok(x) else "-"

    def mib(v):
        return "-" if v in (None, "") else "%.0f" % (v / 1048576.0)

    def order(r):
        w = str(r.get("ct_ws_mib", "est"))
        return (r.get("case", ""), r.get("ct_emu", ""), r.get("ct_ws", ""), -1 if w == "est" else int(w) if w.isdigit() else 0)

    print("| case | variant | ct_ws | limit MiB | native TFLOPS-eq | emulated TFLOPS-eq | emu / native | ws_req MiB | engaged | "
          "native same-ws TFLOPS-eq | same-ws bit-identical | schema |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for e in sorted(emu, key=order):
        n = native.get(e.get("case"))
        ratio = "%.2fx" % (e["tflops_eq"] / n["tflops_eq"]) if ok(n) and ok(e) else "-"
        if not ok(e):
            eng = "run failed: %s" % str(e.get("status"))[:60]
        elif e.get("emu_engaged") is not None:
            eng = "YES" if e["emu_engaged"] else "no (native)"
            if e.get("ws_limit_capped"):
                eng += " (limit capped to free memory)"
        elif e.get("identical_to_native") is not None:
            eng = "no (identical to native)" if e["identical_to_native"] else "differs (unverified)"
        else:
            eng = "-"
        ns = e.get("native_same_ws_median_s")
        nstf = "%.2f" % (e["tflops_eq"] * e["median_s"] / ns) if ok(e) and ns else "-"
        ident = e.get("identical_to_native_same_ws")
        print("| %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |" % (
            e.get("case"), e.get("ct_emu", "default"), e.get("ct_ws", "?"), e.get("ct_ws_mib", "est"), tf(n), tf(e), ratio, mib(e.get("ws_req_bytes", e.get("ws_bytes"))),
            eng, nstf, "-" if ident is None else ("yes" if ident else "no"), e.get("bench_schema", 1)))
    print()
    print("`engaged` decides (graph capture of one call; confirm with the nsys profile of the cutensor-emu runs). The same-ws "
          "native twin separates a workspace effect on native cuTENSOR from emulation.")


def legacy(recs):
    """The schema-1 table, as earlier campaigns wrote it."""
    by = {}
    for r in recs:
        by.setdefault((r["case"], r.get("ct_ws", "?")), {})[r["method"]] = r
    print("| case | ct_ws | native TFLOPS-eq | emulated TFLOPS-eq | emu / native | native ws MiB | emu ws MiB | differs from native |")
    print("|---|---|---|---|---|---|---|---|")
    for (case, ws), m in sorted(by.items()):
        n, e = m.get("cutensor"), m.get("cutensor-emu")
        def tf(x):
            return "-" if not x or x.get("status") != "ok" else "%.2f" % x["tflops_eq"]
        def mib(x):
            return "-" if not x else "%.0f" % (x.get("ws_bytes", 0) / 1048576.0)
        ratio = "-"
        if n and e and n.get("status") == "ok" and e.get("status") == "ok":
            ratio = "%.2fx" % (e["tflops_eq"] / n["tflops_eq"])
        if e and e.get("status") != "ok":
            diff = "emulated run failed: %s" % str(e.get("status"))[:60]
        elif e and e.get("identical_to_native") is not None:
            diff = "no (identical to native)" if e["identical_to_native"] else "YES"
        elif n and e and n.get("c_hash") and e.get("c_hash"):
            diff = "no (identical to native)" if n["c_hash"] == e["c_hash"] else "YES"
        else:
            diff = "-"
        print("| %s | %s | %s | %s | %s | %s | %s | %s |" % (case, ws, tf(n), tf(e), ratio, mib(n), mib(e), diff))


    print()
    print("`YES` alone does not prove the emulation ran: cuTENSOR can also pick a different NATIVE plan when emulation is requested "
          "(an RTX 4090, which cannot emulate, shows `YES` on t_cc with a larger workspace). Confirm with the nsys profile "
          "(INT8 tensor-core kernels) of the cutensor-emu runs.")


if __name__ == "__main__":
    main()
