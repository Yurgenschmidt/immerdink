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
Bytes moved (the FP64 operand is read once from DRAM by the fused pack; planes are 1 byte per element):
  pack of an operand with R rows (b * m, or b * n), k contracted, k_hat padded, N moduli:  8 R k  +  N R k_hat  (+ 3 R for sigma, flags)
  CRT: N b m n_hat  +  8 b m n   (+ 8 b m n when beta reads C)
"""
import argparse
import json
import sys

LIB_SHAPES = {"fan": (1, 256), "fanin": (1, 256)}   # (blocks per SM, threads): not tuned; the library's kernels differ (see NOTES-21)
FRACTION = 0.85
HEADROOM = 1.15
SMALL_BYTES = 32 << 20


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
            if by < SMALL_BYTES:
                row["verdict"] = "SMALL"
            elif pattern in best:
                # the card's best for this access pattern, or for a plain copy (mixed read / write traffic) if that is higher: the library's
                # kernels are tuned and can beat this program's simple fan-out / fan-in kernels
                ref = max(best[pattern]["gbs"], best.get("cp", {"gbs": 0.0})["gbs"])
                row["best_gbs"] = ref
                row["pct_of_best"] = 100.0 * gbs / ref
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
        extra = f"{r['pct_of_best']:5.1f}% of best {r['pattern']}/cp" if "pct_of_best" in r else ""
        print(f"  {r['case']:10s} N={r['N']:2d} {r['phase']:10s} {r['ms']:8.3f} ms {r['gbs']:8.1f} GB/s {extra:28s} {r['verdict']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
