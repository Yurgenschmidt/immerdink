#!/usr/bin/env python3
"""Table of the cuTENSOR workspace-preference experiment (bench/sweep_cutensor_ws.txt):
  bench/cutws_table.py cutensor_ws.jsonl
Per case and preference (ct_ws): native and emulated time, TFLOPS-eq, workspace bytes, and whether the emulated C is
bit-identical to the native C (identical_to_native, else c_hash). 'YES' under 'differs' means the emulation produced
a different result than native FP64 at that preference, i.e. a different algorithm ran (confirm with the nsys profile
that INT8 tensor-core kernels ran); 'no' means it fell back to native. Python 3.6+, stdlib only."""
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
