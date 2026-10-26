#!/usr/bin/env python3
"""Issue 22: the Hopper (wgmma) K4 backend against the mma.sync K4, through the real library path; passes on every device.

    python3 tests/test_k4_wgmma.py [--lib build/libimmerdink.so]

What runs where (docs/NOTES-22.md "Library integration (design)", "Cluster test plan"):

  every device     the backend's status is what the device class and the build say: unavailable off sm_90 (or in a build without
                   IMMERDINK_K4_WGMMA), a forced wgmma plan then reports mma.sync and the launch counters stay 0 (a forced wgmma that
                   silently ran mma.sync on a Hopper would otherwise pass every bit comparison below without testing anything); and
                   a set of shapes (m, n-hat tails, k not a multiple of 128, K-chunk / batch / n-hat splits, N = 2..20, both input
                   families, RN and TRUNC, alpha and beta, a single row) gives the same C with the backend forced to mma.sync and forced to
                   wgmma. On a build with the backend the host half of a Hopper launch (tile choice, can_implement; the TMA descriptors from sm_90 up) is
                   also dry-run on edge shapes, on any device.
  Hopper (sm_90)   the same comparison is the exactness test of the Hopper path: the forced runs must serve launches with the wgmma
                   kernel (counter > 0, plan backend 1) and produce the same C bit for bit as the mma.sync run (counter 0).
"""
import argparse
import ctypes as C
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import harness as hz  # noqa: E402
import test_output_hashes as T  # noqa: E402


def case(name, a, b, c, N, family="u", rounding="RN", alpha=1.0, beta=0.0, limits=None):
    return {"name": name, "A": {"extent": a[0], "label": a[1]}, "B": {"extent": b[0], "label": b[1]}, "C": {"extent": c[0], "label": c[1]},
            "N": N, "family": family, "rounding": rounding, "alpha": alpha, "beta": beta, "limits": limits}


def gemm(name, m, n, k, N, **kw):
    return case(name, ([m, k], [1, 3]), ([n, k], [2, 3]), ([m, n], [1, 2]), N, **kw)


def batched(name, m, n, k, bt, N, **kw):
    return case(name, ([m, k, bt], [1, 3, 4]), ([n, k, bt], [2, 3, 4]), ([m, n, bt], [1, 2, 4]), N, **kw)


# (case, what it exercises). "limits" = (k_chunk, max_batch, max_n_tiles) of imd_test_set_limits (0: production).
CASES = [
    (gemm("tile_multiple", 256, 512, 1024, 14), "whole tiles, 8 k tiles of 128"),
    (gemm("tails_mn", 129, 272, 1040, 14), "m and n-hat not tile multiples, k = 1040 (8.1 k tiles)"),
    (gemm("k_not_128", 200, 300, 1000, 8, family="pow2"), "k not a multiple of 128 (K = 1000 padded to the plan's alignment)"),
    (gemm("k_small", 144, 288, 48, 12), "K shorter than one k tile"),
    (gemm("one_row", 1, 400, 520, 14), "m = 1"),
    (gemm("narrow", 300, 5, 700, 10, rounding="TRUNC"), "n = 5 (n-hat 16)"),
    (gemm("n2", 130, 260, 600, 2), "N = 2"),
    (gemm("n20", 130, 260, 600, 20, family="pow2"), "N = 20"),
    (gemm("alpha_beta", 190, 270, 900, 12, alpha=0.75, beta=-0.5), "alpha, beta"),
    (batched("batched", 130, 270, 520, 6, 14), "batch 6 (z of the tile scheduler)"),
    (gemm("ksplit", 130, 270, 1100, 14, limits=(512, 0, 0)), "K-chunk 512: store-mod launch then add-mod launches (K4Launch k0 > 0)"),
    (batched("bsplit", 130, 270, 520, 5, 14, limits=(0, 2, 0)), "batch part <= 2: batch0 offsets in the modulus (slot0 + (batch0 + z) / b)"),
    (gemm("nsplit", 130, 700, 520, 14, limits=(0, 0, 1)), "n-hat part of one 256-column tile: the gridDim.y split"),
    (batched("all_splits", 130, 600, 1100, 5, 8, limits=(512, 2, 1)), "K-chunk, batch and n-hat splits together"),
]

# Dry-run table of the host half (m, nc, kc, zc, k_hat, n_hat, force) -> (served?, swizzle, raster).
SMS, CLUSTERS = 132, 66
DRY = [
    ((4096, 4096, 4096, 14, 4096, 4096, 0), (True, 1, 0), "R2 4096^3: 512 tiles per batch, swizzle 1 heuristic"),
    ((8192, 8192, 8192, 4, 8192, 8192, 0), (True, 8, 2), "R2 8192^3: 2048 tiles per batch, swizzle 8 AlongN"),
    ((8192, 8192, 8192, 1, 8192, 8192, 0), (True, 8, 2), "R2 8192^3 batch 1"),
    ((2048, 2048, 2048, 14, 2048, 2048, 0), (True, 1, 0), "inside the floor"),
    ((512, 512, 512, 14, 512, 512, 0), (False, 0, 0), "below the profitability floor: mma.sync"),
    ((512, 512, 512, 14, 512, 512, 1), (True, 1, 0), "forced: the floor is lifted"),
    ((1, 400, 16, 1, 16, 400, 1), (True, 1, 0), "forced, m = 1, one k tile of 16"),
    ((129, 272, 1040, 3, 1040, 272, 1), (True, 1, 0), "forced, tails in m, n and k"),
    ((4097, 4112, 4112, 2, 4112, 4112, 1), (True, 1, 0), "forced, tails of the big shape"),
    ((4096, 4096, 4096, 14, 4096, 4096, 1), (True, 1, 0), "forced, same as the default"),
    ((256, 100, 512, 1, 512, 112, 1), (False, 0, 0), "never served: n-part 100 is not a multiple of 16"),
    ((256, 256, 500, 1, 512, 256, 1), (False, 0, 0), "never served: K chunk 500 is not a multiple of 16"),
    ((256, 256, 512, 1, 496, 256, 1), (False, 0, 0), "never served: the chunk is longer than the row"),
    ((256, 256, 512, 70000, 512, 256, 1), (False, 0, 0), "never served: batch part beyond 65535"),
]


def device_cc(cuda):
    major, minor, dev = C.c_int(), C.c_int(), C.c_int()
    cuda.check(cuda.rt.cudaGetDevice(C.byref(dev)), "cudaGetDevice")
    cuda.check(cuda.rt.cudaDeviceGetAttribute(C.byref(major), 75, dev), "cc major")
    cuda.check(cuda.rt.cudaDeviceGetAttribute(C.byref(minor), 76, dev), "cc minor")
    return major.value, minor.value


def plan_backend(lib, c):
    """The backend a plan of the case tries first (0 mma.sync, 1 wgmma)."""
    ts = {X: hz.make_tensor(c[X]["extent"], hz.dense(c[X]["extent"]), c[X]["label"]) for X in "ABC"}
    rounding = hz.TRUNC if c["rounding"] == "TRUNC" else hz.RN
    st, plan = lib.plan_create(ts["A"], ts["B"], ts["C"], c["N"], 0, rounding)
    assert st == "SUCCESS", st
    try:
        return lib.plan_k4_backend(plan)
    finally:
        lib.plan_destroy(plan)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--lib", default=hz.default_lib())
    args = ap.parse_args()
    lib, cuda = hz.Library(args.lib), hz.Cuda()
    if not lib.has_test_hooks:
        print("FAIL  the library has no test hooks")
        return 1
    major, minor = device_cc(cuda)
    built = lib.lib.imd_test_k4_wgmma_built() == 1
    h = lib.handle()
    available = lib.lib.imd_test_k4_wgmma_available(h)
    bad = []
    want_available = built and (major, minor) == (9, 0)
    if available not in (0, 1):
        bad.append(f"imd_test_k4_wgmma_available returned {available}")
    elif bool(available) != want_available:
        bad.append(f"sm_{major}{minor}, backend built: {built}: available {available}, expected {int(want_available)}")
    hopper = bool(available)
    # a test run of the whole process may force a backend (IMD_TEST_K4); this test sets its own and puts that one back
    prev = lib.lib.imd_test_get_k4_backend()
    n_cmp = n_dry = 0
    try:
        # ---- the plan reports the backend it tries first -------------------------------------------------------------------------
        probe = CASES[1][0]
        lib.set_k4_backend(-1)
        auto = plan_backend(lib, probe)
        lib.set_k4_backend(0)
        forced_mma = plan_backend(lib, probe)
        lib.set_k4_backend(1)
        forced_wg = plan_backend(lib, probe)
        if forced_mma != 0:
            bad.append(f"forced mma.sync: the plan reports backend {forced_mma}")
        if forced_wg != (1 if hopper else 0):
            bad.append(f"forced wgmma on {'a Hopper' if hopper else 'a device without the backend'}: the plan reports backend {forced_wg}")
        if auto != (1 if hopper else 0):
            bad.append(f"device-class default: the plan reports backend {auto} ({'Hopper' if hopper else 'no backend here'})")
        for v in (-2, 2):
            if lib.lib.imd_test_set_k4_backend(v) == 0:
                bad.append(f"imd_test_set_k4_backend({v}) was accepted")
        # ---- bit identity of C, forced mma.sync vs forced wgmma, with the launch counters ----------------------------------------
        for c, what in CASES:
            hashes = {}
            for backend in (0, 1):
                lib.set_k4_backend(backend)
                k = c["limits"] or (0, 0, 0)
                lib.set_limits(*k)
                lib.k4_wgmma_counts(reset=True)
                try:
                    hashes[backend] = T.hash_c(T.run_gpu(lib, cuda, c))
                finally:
                    lib.set_limits(0, 0, 0)
                served, declined = lib.k4_wgmma_counts(reset=True)
                if backend == 0 and (served, declined) != (0, 0):
                    bad.append(f"{c['name']}: the mma.sync run counted wgmma launches {served} served, {declined} declined")
                if backend == 1 and not hopper and (served, declined) != (0, 0):
                    bad.append(f"{c['name']}: no backend on this device, but {served} launches were served and {declined} declined")
                if backend == 1 and hopper and served < 1:
                    bad.append(f"{c['name']} ({what}): forced wgmma served {served} launches ({declined} declined): the Hopper path did not run")
                if backend == 1 and hopper and declined:
                    bad.append(f"{c['name']} ({what}): forced wgmma declined {declined} launches, served {served}: a shape the design says is legal")
            n_cmp += 1
            if hashes[0] != hashes[1]:
                bad.append(f"{c['name']} ({what}): C hash with mma.sync {hashes[0]}, with the forced wgmma backend {hashes[1]}")
        # ---- the host half of a Hopper launch, on any device of a build with the backend ----------------------------------------
        if built:
            for (m, nc, kc, zc, k_hat, n_hat, force), (served, swizzle, raster), what in DRY:
                for add in (0, 1):    # the store-mod and the add-mod kernel are both set up from the same launch
                    out = (C.c_int32 * 7)()
                    rc = lib.lib.imd_test_k4_wgmma_dry_run(m, nc, kc, zc, k_hat, n_hat, m * k_hat, n_hat * k_hat, SMS, CLUSTERS, add, force, out)
                    n_dry += 1
                    key = (m, nc, kc, zc, k_hat, n_hat, force, "add" if add else "store")
                    got = (out[0] >= 0, out[1], out[2])
                    if not served:
                        if out[0] >= 0 or rc != 0:
                            bad.append(f"dry run {key} ({what}): served variant {out[0]}, rc {rc}")
                    elif rc != 0 or out[3] != 1:
                        bad.append(f"dry run {key} ({what}): rc {rc}, can_implement {out[3]}")
                    elif got != (True, swizzle, raster):
                        bad.append(f"dry run {key} ({what}): (served, swizzle, raster) {got}, expected {(True, swizzle, raster)}")
                    elif major >= 9 and out[4] * out[5] * out[6] < 1:   # the tensor maps are only encoded from sm_90 up
                        bad.append(f"dry run {key} ({what}): empty grid {list(out)[4:]}")
    finally:
        lib.lib.imd_test_set_k4_backend(prev)
        lib.set_limits(0, 0, 0)
    if bad:
        print("FAIL")
        for b in bad:
            print("  " + b)
        return 1
    where = (f"Hopper path served the forced runs (plan backend 1, launches counted, none declined)" if hopper else
             f"sm_{major}{minor}{'' if built else ', build without the backend'}: backend unavailable, a forced wgmma plan runs mma.sync, no launch counted")
    print(f"PASS  {n_cmp} cases bit-identical between forced mma.sync and forced wgmma; {where}; {n_dry} host dry runs of the Hopper launch")
    return 0


if __name__ == "__main__":
    sys.exit(main())
