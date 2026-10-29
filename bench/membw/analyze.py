#!/usr/bin/env python3
"""Reads imd_membw output (and optionally imd_bench records with phase times) and names the limiter of the pack / CRT phases (issue 21).

    analyze.py MEMBW.jsonl [BENCH.jsonl ...] [--json]

From the microbenchmark: for each pattern the best bandwidth over all launch shapes, the bandwidth of the shape the library uses
(LIB_SHAPES), the datasheet-derived peak. From the bench records: for the pack phases (KL A / KL B, K2 pack A, K3 pack B, KF fused) and the CRT
(K5 crt) the bytes they must move (below) over the phase time = the achieved bandwidth, compared with the best the card delivers for the same
access pattern (fan-out for packs, fan-in for the CRT). The verdict per phase:
  BANDWIDTH-BOUND   achieved >= 85% of the best bandwidth of its pattern (or of a plain copy, if higher): only fewer bytes can make it faster
  SHAPE-BOUND       achieved < 85% but the best shape of the pattern delivers >= 1.15x more: the kernel's launch shape leaves bandwidth unused
  SMALL             the phase moves < 32 MiB: launch latency, not bandwidth, dominates; not judged
Bytes moved (planes are 1 byte per element); valid for beta = 0 (imd_bench records no beta) and for the KL / K2 / K3 packs and K5:
  pack of an operand with R rows (b * m, or b * n), k contracted, k_hat padded, N moduli:  8 R k  +  N R k_hat  (+ 3 R for sigma, flags)
  CRT: N b m n_hat  +  8 b m n
KL reads the operand twice (K1's scaling pass, then the pack); the model counts it once, which holds while K1's second pass hits L2: that is what
the K1 L2 budgets guarantee (docs/NOTES-21.md). KF (fused) records are not judged. The reference is the best of this program's patterns, so the
percentages say "relative to these kernels", not "relative to the hardware"; pct_of_peak (datasheet-derived, unreliable on GDDR6X / GDDR7) and
pct_of_rd (the best read bandwidth measured) are printed beside it.
"""
import argparse
import json
import sys

LIB_SHAPES = {"fan": (1, 256), "fanin": (1, 256)}   # (blocks per SM, threads): not tuned; the library's kernels differ (see NOTES-21)
FRACTION = 0.85
HEADROOM = 1.15
SMALL_BYTES = 32 << 20          # floor of the "not judged" size; with the device's L2 known it is max(this, 2 x L2) (see small_bytes)
# the patterns each phase is compared with: the best of all of them (a reference made of this program's own kernels is not a hardware bound, so
# the library's wide-store shape, the plain copy and the one-way patterns all count)
PACK_REFS = ("fan", "fan16", "cp", "rd", "wr")
CRT_REFS = ("fanin", "fanin16", "cp", "rd", "wr")


def load(paths):
    out = []
    for p in paths:
        with open(p) as f:
            for line in f:
                line = line.strip()
                if line:
                    out.append(json.loads(line))
    return out


def membw_summary(recs):
    dev = next((r for r in recs if r.get("kind") == "membw_device"), None)
    best, lib = {}, {}
    for r in recs:
        if r.get("kind") != "membw":
            continue
        pat = r["pattern"]
        if pat not in best or r["gbs"] > best[pat]["gbs"]:
            best[pat] = r
        if pat in LIB_SHAPES and (r["blocks_per_sm"], r["threads"]) == LIB_SHAPES[pat]:
            if pat not in lib or r["gbs"] > lib[pat]["gbs"]:
                lib[pat] = r
    return dev, best, lib


def small_bytes(dev):
    """Phases below this size are not judged: a phase of a few L2s is largely L2-resident and its bandwidth is not DRAM bandwidth."""
    return max(SMALL_BYTES, 2 * int(dev["l2_bytes"])) if dev and dev.get("l2_bytes") else SMALL_BYTES


def phase_bytes(rec, name):
    """Bytes a phase moves, from the record's GEMM view (m, n, k, b), N and the phase name; None if not a pack / CRT phase."""
    m, n, k, b = rec["m"], rec["n"], rec["k"], rec["b"]
    N = rec["N"]
    kh = rec.get("k_hat") or -(-k // 16) * 16
    nh = -(-n // 16) * 16
    if name in ("KL A", "K2 pack A"):
        return 8 * b * m * k + N * b * m * kh + 3 * b * m
    if name in ("KL B", "K3 pack B"):
        return 8 * b * n * k + N * b * nh * kh + 3 * b * n
    if name == "K5 crt":
        return N * b * m * nh + 8 * b * m * n
    return None


def judge(membw_recs, bench_recs):
    dev, best, lib = membw_summary(membw_recs)
    rows = []
    for r in bench_recs:
        if r.get("method") != "imd" or r.get("status", "ok") != "ok" or not r.get("phases"):
            continue
        for name, ph in r["phases"].items():
            by = phase_bytes(r, name)
            if by is None or not ph.get("ms"):
                continue
            pattern = "fanin" if name == "K5 crt" else "fan"
            gbs = by / (ph["ms"] * 1e6)
            row = {"case": r["case"], "N": r["N"], "phase": name, "bytes": by, "ms": ph["ms"], "gbs": gbs, "pattern": pattern}
            if by < small_bytes(dev):
                row["verdict"] = "SMALL"
            elif pattern in best:
                # the best of this program's kernels for the access pattern (own shape and the library's wide-store shape), a plain copy, a pure
                # read and a pure write: the library's kernels are tuned and can beat any one of them
                refs = PACK_REFS if pattern == "fan" else CRT_REFS
                ref = max(best[x]["gbs"] for x in refs if x in best)
                row["best_gbs"] = ref
                row["pct_of_best"] = 100.0 * gbs / ref
                if "rd" in best:
                    row["pct_of_rd"] = 100.0 * gbs / best["rd"]["gbs"]
                if dev and dev.get("peak_gbs"):
                    row["pct_of_peak"] = 100.0 * gbs / dev["peak_gbs"]
                if gbs >= FRACTION * ref:
                    row["verdict"] = "BANDWIDTH-BOUND"
                elif ref >= HEADROOM * gbs:
                    row["verdict"] = "SHAPE-BOUND"
                else:
                    row["verdict"] = "BANDWIDTH-BOUND"
            else:
                row["verdict"] = "NO-MICROBENCHMARK"
            rows.append(row)
    return dev, best, lib, rows


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("files", nargs="+")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)
    recs = load(a.files)
    membw = [r for r in recs if str(r.get("kind", "")).startswith("membw")]
    bench = [r for r in recs if not str(r.get("kind", "")).startswith("membw")]
    dev, best, lib, rows = judge(membw, bench)
    if a.json:
        print(json.dumps({"device": dev, "best": {k: v["gbs"] for k, v in best.items()}, "rows": rows}))
        return 0
    if dev:
        print(f"{dev['gpu']}: {dev['sm_count']} SMs, L2 {dev['l2_bytes'] >> 20} MiB, peak {dev['peak_gbs']:.0f} GB/s")
    for pat, r in sorted(best.items()):
        print(f"  best {pat:8s} {r['gbs']:8.1f} GB/s ({r['pct_of_peak']:5.1f}% of peak) at {r['blocks_per_sm']} blocks/SM x {r['threads']} threads, {r['ilp']} loads in flight")
    for r in rows:
        extra = (f"{r['pct_of_best']:5.1f}% of best, {r.get('pct_of_rd', 0):5.1f}% of rd, {r.get('pct_of_peak', 0):5.1f}% of peak"
                 if "pct_of_best" in r else "")
        print(f"  {r['case']:10s} N={r['N']:2d} {r['phase']:10s} {r['ms']:8.3f} ms {r['gbs']:8.1f} GB/s {extra:56s} {r['verdict']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
