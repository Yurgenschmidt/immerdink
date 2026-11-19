#!/usr/bin/env python3
"""IMD_HWGEMM performance model in the planning API (DESIGN v1.21 §6i.5, issue #75), GPU-free.

    python3 tests/test_hw_perf.py [--host-lib build/libimmerdink_host.so]

What this checks. ehrenfest plans memory and speed from libimmerdink_host alone, so a HWGEMM
call must have a prediction (imd_perf_predict) and ranked candidates (imd_perf_rank):
  1. the phases: one per kind (K0, "HWGEMM pack A" / "B", "HWGEMM gemm"), launches S (the
     GEMM: S x chunks), kernel_launches = the lowering's, flops_eq = 2bmnk with the true k;
  2. an uncalibrated class (every shipped file so far) is the roofline max(bytes / peak
     DRAM, flops / peak) + launches x direct.launch, flagged in uncalibrated_phases with a
     zero-width interval: recomputed here by hand from the §6i.5 feature definitions;
  3. a calibrated GEMM (hw_family + hw_gemm.* keys) is launch + call + max(flop + wave + own,
     byte), the tile from the family's rule; the file round-trips through format / parse and
     a file without HW keys formats exactly as before (no hw_ text);
  4. slabs: hw_slab / hw_slabs reported, the error bound grows with S;
  5. rank: the modes mask, k_slabs (one candidate per distinct slab length), FIT_SLAB (the
     largest multiple-of-16 slab that fits the budget, and L + 16 does not), mode / k_slab
     reported, DIRECT / HWGEMM candidates have no N.
The expected numbers are written from the definitions, not from the library's own code.
"""

import argparse
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "python"))
from immerdink import host  # noqa: E402

FAILS = []


def check(cond, what):
    if not cond:
        FAILS.append(what)
        print(f"FAIL  {what}")


def close(a, b, rel=1e-9):
    return abs(a - b) <= rel * max(abs(a), abs(b), 1e-300)


def up16(x):
    return (x + 15) // 16 * 16


def roofline_gemm(cal, m, n, k, b, tm, tn, packed_rows, slabs, beta=False, csec=1):
    """The §6i.5 GEMM roofline: (time_s, bytes, flops); slabs = [len_s]."""
    meta, p = cal.meta, cal.params
    hat = sum(up16(x) for x in slabs)
    tiles = b * math.ceil(m / tm) * math.ceil(n / tn)
    flops = tiles * 2 * tm * tn * hat
    ab = 0
    for rows, packed in ((m, packed_rows[0]), (n, packed_rows[1])):
        ab += 8 * b * rows * (hat if packed else k)
    S = len(slabs)
    cbytes = 8 * b * m * n * (S + S - 1 + (1 if beta else 0)) * csec
    nbytes = ab + cbytes
    peak = meta["peak_fp64_flops_per_s"]
    launches = S
    t = max(nbytes / meta["peak_dram_bytes_per_s"], flops / peak) + \
        p["direct.launch"] * 1e-9 * launches
    return t, nbytes, flops


def test_affine_roofline(lib, cal):
    m, n, k = 200, 300, 100
    pr = host.predict("ik,jk->ij", dict(i=m, j=n, k=k), N=8, mode="HWGEMM", regime="BURST",
                      lib=lib)
    check(pr.status == "SUCCESS" and pr.dispatch == "HWGEMM", f"affine status {pr.status}")
    check([q.name for q in pr.phases] == ["HWGEMM gemm"], f"affine phases {pr.phases}")
    check(pr.workspace_bytes == 0 and pr.kernel_launches == 1, "affine workspace / launches")
    check(pr.flops_eq == 2 * m * n * k, "affine flops_eq")
    check(pr.hw_tile == 2 and pr.hw_slab == k and pr.hw_slabs == 1,
          f"affine tile {pr.hw_tile} L {pr.hw_slab} S {pr.hw_slabs}")
    check(pr.uncalibrated_phases == 1 and pr.phases[0].uncalibrated, "affine uncalibrated bit")
    t, nb, fl = roofline_gemm(cal, m, n, k, 1, 64, 64, (False, False), [k])
    g = pr.phases[0]
    check(close(g.time_s, t) and close(g.bytes, nb) and close(g.ops, fl),
          f"affine roofline {g.time_s} / {g.bytes} / {g.ops}, want {t} / {nb} / {fl}")
    check(g.lo_s == g.time_s == g.hi_s, "affine roofline interval [t, t]")
    check(pr.power_w == 0, "no HWGEMM power model")
    # beta != 0: C read once more
    pb = host.predict("ik,jk->ij", dict(i=m, j=n, k=k), N=8, mode="HWGEMM", regime="BURST",
                      beta=True, lib=lib)
    check(close(pb.phases[0].bytes, nb + 8 * m * n), "beta reads C")
    print("ok    affine roofline")


def test_packed_slabs(lib, cal):
    # A = 5 x 100 at i*2 + k*10 (no unit stride): packed; B, C affine. k_slab 32: L 32, S 4.
    m, n, k = 5, 6, 100
    ext = dict(i=m, j=n, k=k)
    strides = ([2, 10], None, None)
    base = host.predict("ik,jk->ij", ext, strides=strides, N=8, mode="HWGEMM", regime="BURST",
                        lib=lib)
    pr = host.predict("ik,jk->ij", ext, strides=strides, N=8, mode="HWGEMM", regime="BURST",
                      k_slab=32, lib=lib)
    lw = host.lowering("ik,jk->ij", ext, strides=strides, N=8, mode="HWGEMM", k_slab=32, lib=lib)
    names = [q.name for q in pr.phases]
    check(names == ["K0 vec", "HWGEMM pack A", "HWGEMM gemm"], f"packed phases {names}")
    check(pr.hw_slab == 32 and pr.hw_slabs == 4, f"slab L {pr.hw_slab} S {pr.hw_slabs}")
    check(pr.kernel_launches == lw.kernel_launches == 1 + 4 * 2,
          f"launches {pr.kernel_launches} / {lw.kernel_launches}")
    check(pr.phase("HWGEMM pack A").launches == 4 and pr.phase("HWGEMM gemm").launches == 4,
          "per-slab launches")
    check(pr.workspace_bytes == lw.workspace_bytes, "workspace = lowering's")
    slabs = [32, 32, 32, 4]
    pa = pr.phase("HWGEMM pack A")
    # pack: sector bytes (source rows-fastest? A is i*2 + k*10: k is not the fast index along
    # rows... read either way: the written bytes are exact) + written 8 m sum ceil16
    written = 8 * m * sum(up16(x) for x in slabs)
    check(pa.x[2] == written, f"pack written {pa.x[2]}, want {written}")
    check(pa.x[1] in (8 * m * k, 4 * 8 * m * k), f"pack sector bytes {pa.x[1]}")
    check(pa.uncalibrated and pr.phase("HWGEMM gemm").uncalibrated, "packed uncalibrated bits")
    t, nb, fl = roofline_gemm(cal, m, n, k, 1, 64, 64, (True, False), slabs)
    g = pr.phase("HWGEMM gemm")
    check(close(g.bytes, nb) and close(g.ops, fl), f"slab gemm {g.bytes} / {g.ops}, want {nb} / {fl}")
    check(pr.err_bound > base.err_bound > 0, "slab error bound grows with S")
    print("ok    packed + slabs")


def calibrated(lib):
    """The shipped calibration plus HW keys (hw_family 90)."""
    text = host.Calibration.default(lib).format()
    check("hw_" not in text, "a calibration without HW keys formats without hw_ text")
    add = ("hw_family = 90\npeak_fp64_tc_flops_per_s = 6.7e13\n"
           "hw_pack.launch = 2000\nhw_pack.sector = 0.0004\nhw_pack.write = 0.0003\n"
           "hw_pack.chain = 0.01\n"
           "hw_gemm.launch = 3000\nhw_gemm.flop = 2e-5\nhw_gemm.wave = 1e-5\n"
           "hw_gemm.byte = 0.0003\nhw_gemm.own = 1e-6\nhw_gemm.call = 500\n"
           "band.hw_gemm = 0.9 1.2\n")
    cal = host.Calibration.parse(text + add, lib)
    again = host.Calibration.parse(cal.format(), lib)
    check(again.format() == cal.format(), "HW keys round-trip")
    check("hw_family = 90" in cal.format() and "band.hw_gemm" in cal.format() and
          "band.hw_pack" not in cal.format(), "HW keys formatted (absent band not written)")
    return cal


def test_calibrated(lib):
    cal = calibrated(lib)
    p = cal.params
    m, n, k = 200, 300, 100
    pr = host.predict("ik,jk->ij", dict(i=m, j=n, k=k), N=8, mode="HWGEMM", regime="BURST",
                      calib=cal, lib=lib)
    check(pr.uncalibrated_phases == 0, f"calibrated bits {pr.uncalibrated_phases}")
    sms = cal.meta["sm_count"]
    # family 90: 2 * ceil(200/128) ceil(300/128) = 12 >= sms? -> 128x128 else 64x128
    ctas = math.ceil(m / 128) * math.ceil(n / 128)
    tile, tm, tn = (1, 64, 128) if 2 * ctas < sms else (0, 128, 128)
    check(pr.hw_tile == tile, f"tile {pr.hw_tile}, want {tile}")
    occ = cal.kernels[["hw_gemm_128x128", "hw_gemm_64x128"][tile]][0]
    tiles = math.ceil(m / tm) * math.ceil(n / tn)
    hat = up16(k)
    flop = tiles * 2 * tm * tn * hat
    slots = sms * occ
    idle = (math.ceil(tiles / slots) * slots - tiles) * 2 * tm * tn * hat
    nbytes = 8 * (m * k + n * k) + 8 * m * n
    ns = p["hw_gemm.launch"] + p["hw_gemm.call"] + max(
        p["hw_gemm.flop"] * flop + p["hw_gemm.wave"] * idle, p["hw_gemm.byte"] * nbytes)
    g = pr.phases[0]
    check(close(g.time_s, ns * 1e-9), f"calibrated gemm {g.time_s}, want {ns * 1e-9}")
    check(close(g.lo_s, 0.9 * g.time_s) and close(g.hi_s, 1.2 * g.time_s), "hw_gemm band")
    check(close(g.bound_s, max(nbytes / cal.meta["peak_dram_bytes_per_s"], flop / 6.7e13)),
          "bound with the FP64 tensor-core peak")
    print("ok    calibrated")


def test_rank(lib):
    ext = dict(i=40, j=30, k=500)
    strides = ([2, 80], None, None)  # A packed
    kw = dict(strides=strides, N=[8], g=[8], lib=lib)
    cands = host.rank("ik,jk->ij", ext, modes=("EMULATE", "DIRECT", "HWGEMM"),
                      k_slabs=(0, 64, 64, 600), **kw)
    modes = sorted({c.mode for c in cands})
    check(modes == ["DIRECT", "EMULATE", "HWGEMM"], f"rank modes {modes}")
    hw = [c for c in cands if c.mode == "HWGEMM"]
    # k_slab 0 and 600 both run L = k: one candidate; 64 twice: one
    check(sorted(c.hw_slab for c in hw) == [64, 500], f"distinct slabs {[c.hw_slab for c in hw]}")
    check(all(c.num_moduli == 0 and c.dispatch == "HWGEMM" and c.uncalibrated_phases
              for c in hw), "HWGEMM candidate fields")
    check(all(c.num_moduli == 0 for c in cands if c.mode == "DIRECT"), "DIRECT has no N")
    full = next(c for c in hw if c.hw_slab == 500)
    s64 = next(c for c in hw if c.hw_slab == 64)
    check(s64.total_bytes < full.total_bytes and s64.hw_slabs == 8, "a slab cuts the footprint")
    # FIT_SLAB: a budget between L = 64 and L = k
    budget = (s64.total_bytes + full.total_bytes) // 2
    fit = host.rank("ik,jk->ij", ext, modes=("HWGEMM",), budget=budget, fit_slab=True, **kw)
    fits = [c for c in fit if c.k_slab not in (0,)]
    check(len(fits) == 1, f"fit candidates {[(c.k_slab, c.total_bytes) for c in fit]}")
    if fits:
        L = fits[0].k_slab
        nxt = host.workspace_size("ik,jk->ij", ext, strides=strides, N=8, mode="HWGEMM",
                                  k_slab=L + 16, lib=lib)
        check(L % 16 == 0 and fits[0].total_bytes <= budget and fits[0].feasible and
              nxt + fits[0].plan_device_bytes > budget, f"fit slab {L}")
    # base mode HWGEMM, modes not given: HWGEMM alone
    only = host.rank("ik,jk->ij", ext, mode="HWGEMM", **kw)
    check({c.mode for c in only} == {"HWGEMM"}, "base mode HWGEMM")
    print("ok    rank")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host-lib", default=None)
    a = ap.parse_args()
    lib = host.load(a.host_lib)
    cal = host.Calibration.default(lib)
    for t in (lambda: test_affine_roofline(lib, cal), lambda: test_packed_slabs(lib, cal),
              lambda: test_calibrated(lib), lambda: test_rank(lib)):
        try:
            t()
        except (AssertionError, host.ImmerdinkError, KeyError, StopIteration) as e:
            FAILS.append(repr(e))
            print(f"FAIL  {e!r}")
    print(f"{'FAILED ' + str(len(FAILS)) if FAILS else 'all ok'}")
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
