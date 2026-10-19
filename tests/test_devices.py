#!/usr/bin/env python3
"""Device-class rules (issue 18, docs/NOTES-18.md): libimmerdink_host against spec/devices.py.

    python3 tests/test_devices.py [--host-lib build/libimmerdink_host.so]

CPU only. Checks: the selection on a grid of synthetic cards (every compute capability 3.0 .. 12.1,
nine SM counts, the real names of the shipped cards and names that are not) against the reference,
once with the built-in rows and once with injected rows (a synthetic sm_120 row, a second sm_89 card, a
sm_80 row, a table with name and SM wildcards); the named cases of the issue (RTX 4090 exact, RTX 6000 Ada
arch, A100 / RTX 3090 generation, H100 default, a synthetic RTX 5090 default without and arch / exact with
a sm_120 row, cc 7.5 unsupported); the kernel code classification, exhaustively over observed and invented
versions; the plan status table; invalid arguments; and that each mutant of the reference rule (spec/devices.py
MUTANTS) is told apart from the real one by the grid (the grid can fail).
"""
import argparse
import ctypes as C
import itertools
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "spec"))
import devices as ref  # noqa: E402


class Dev(C.Structure):
    _fields_ = [("struct_size", C.c_uint32), ("cc_major", C.c_int32), ("cc_minor", C.c_int32),
                ("sm_count", C.c_int32), ("reserved", C.c_int32), ("l2_bytes", C.c_uint64),
                ("smem_optin_bytes", C.c_uint64), ("name", C.c_char * 64)]


class Row(C.Structure):
    _fields_ = [("table_id", C.c_int32), ("cc_major", C.c_int32), ("cc_minor", C.c_int32),
                ("sm_count", C.c_int32), ("name_prefix", C.c_char * 64)]


class Sel(C.Structure):
    _fields_ = [("struct_size", C.c_uint32), ("match", C.c_int32), ("table_id", C.c_int32),
                ("row_index", C.c_int32), ("approximate", C.c_int32), ("reserved", C.c_int32)]


assert C.sizeof(Dev) == 104 and C.sizeof(Row) == 80 and C.sizeof(Sel) == 24


class Fail(Exception):
    pass


def load(path):
    lib = C.CDLL(path)
    lib.imd_device_select.argtypes = [C.POINTER(Dev), C.POINTER(Sel)]
    lib.imd_device_select_rows.argtypes = [C.POINTER(Dev), C.POINTER(Row), C.c_int32, C.POINTER(Sel)]
    lib.imd_device_row_count.restype = C.c_int32
    lib.imd_device_row.argtypes = [C.c_int32, C.POINTER(Row)]
    lib.imd_kernel_code_kind.argtypes = [C.c_int32] * 5
    lib.imd_plan_status_of.argtypes = [C.c_int32] * 2
    for f in ("imd_match_name", "imd_code_kind_name", "imd_plan_status_name"):
        getattr(lib, f).restype = C.c_char_p
        getattr(lib, f).argtypes = [C.c_int32]
    return lib


def dev(major, minor, sms, name):
    return dict(cc_major=major, cc_minor=minor, sm_count=sms, name=name)


def cdev(d):
    return Dev(C.sizeof(Dev), d["cc_major"], d["cc_minor"], d["sm_count"], 0, 0, 0, d["name"].encode())


def crow(w):
    return Row(w["table_id"], w["cc_major"], w["cc_minor"], w["sm_count"], w["name_prefix"].encode())


def csel(lib, d, rows=None):
    out = Sel(C.sizeof(Sel))
    if rows is None:
        st = lib.imd_device_select(C.byref(cdev(d)), C.byref(out))
    else:
        arr = (Row * max(len(rows), 1))(*[crow(w) for w in rows])
        st = lib.imd_device_select_rows(C.byref(cdev(d)), arr, len(rows), C.byref(out))
    if st != 0:
        raise Fail(f"select status {st} for {d}")
    return out.match, out.table_id, out.row_index, out.approximate


NAMES = ["NVIDIA GeForce RTX 4090", "NVIDIA GeForce RTX 4090 D", "NVIDIA RTX 6000 Ada Generation",
         "NVIDIA H100 80GB HBM3", "NVIDIA H200", "NVIDIA RTX PRO 6000 Blackwell Server Edition",
         "NVIDIA GeForce RTX 5090", "NVIDIA A100-SXM4-80GB", "NVIDIA GeForce RTX 3090", "", "RTX 4090"]
SMS = [1, 46, 82, 108, 128, 132, 142, 170, 188]
CCS = [(a, b) for a in range(3, 13) for b in range(0, 10) if (a, b) < (12, 2)]

ROW_SETS = {
    "builtin": ref.ROWS,
    "sm120": ref.ROWS + [dict(table_id=0, cc_major=12, cc_minor=0, sm_count=188, name_prefix="NVIDIA RTX PRO 6000")],
    "two89": ref.ROWS + [dict(table_id=0, cc_major=8, cc_minor=9, sm_count=0, name_prefix="NVIDIA RTX 6000 Ada")],
    "ampere_hopper": ref.ROWS + [dict(table_id=0, cc_major=8, cc_minor=0, sm_count=108, name_prefix=""),
                                 dict(table_id=0, cc_major=9, cc_minor=0, sm_count=0, name_prefix="")],
    "wildcards": [dict(table_id=0, cc_major=9, cc_minor=0, sm_count=0, name_prefix=""),
                  dict(table_id=0, cc_major=8, cc_minor=6, sm_count=0, name_prefix="")],
    "ties": [dict(table_id=0, cc_major=8, cc_minor=2, sm_count=0, name_prefix=""),
             dict(table_id=0, cc_major=8, cc_minor=0, sm_count=0, name_prefix="")],
    "no89": [dict(table_id=0, cc_major=9, cc_minor=0, sm_count=132, name_prefix="NVIDIA H100")],
}


def check_grid(lib):
    n = 0
    for tag, rows in ROW_SETS.items():
        for (a, b), sms, name in itertools.product(CCS, SMS, NAMES):
            d = dev(a, b, sms, name)
            want = ref.select(d, rows)
            got = csel(lib, d, None if tag == "builtin" else rows)
            if got[:3] != want or got[3] != (0 if want[0] == ref.EXACT else 1):
                raise Fail(f"{tag}: {d}: host {got}, reference {want}")
            n += 1
    return n


def check_named(lib):
    cases = [
        (dev(8, 9, 128, "NVIDIA GeForce RTX 4090"), None, ref.EXACT),
        (dev(8, 9, 142, "NVIDIA RTX 6000 Ada Generation"), None, ref.ARCH),
        (dev(8, 0, 108, "NVIDIA A100-SXM4-80GB"), None, ref.GENERATION),
        (dev(8, 6, 82, "NVIDIA GeForce RTX 3090"), None, ref.GENERATION),
        (dev(9, 0, 132, "NVIDIA H100 80GB HBM3"), None, ref.DEFAULT),
        (dev(9, 0, 132, "NVIDIA H200"), None, ref.DEFAULT),
        (dev(12, 0, 188, "NVIDIA RTX PRO 6000 Blackwell Server Edition"), None, ref.DEFAULT),
        (dev(12, 0, 170, "NVIDIA GeForce RTX 5090"), None, ref.DEFAULT),
        (dev(7, 5, 68, "NVIDIA GeForce RTX 2080 Ti"), None, ref.UNSUPPORTED),
        (dev(12, 0, 170, "NVIDIA GeForce RTX 5090"), ROW_SETS["sm120"], ref.ARCH),
        (dev(12, 0, 188, "NVIDIA RTX PRO 6000 Blackwell Server Edition"), ROW_SETS["sm120"], ref.EXACT),
        (dev(10, 0, 148, "NVIDIA B200"), None, ref.DEFAULT),
    ]
    for d, rows, match in cases:
        got = csel(lib, d, rows)
        if got[0] != match:
            raise Fail(f"{d['name']} ({d['cc_major']}.{d['cc_minor']}): match {got[0]}, expected {match}")
    # a synthetic RTX 5090 is flagged approximate with and without a sm_120 row
    if csel(lib, dev(12, 0, 170, "NVIDIA GeForce RTX 5090"))[3] != 1 or \
            csel(lib, dev(12, 0, 170, "NVIDIA GeForce RTX 5090"), ROW_SETS["sm120"])[3] != 1:
        raise Fail("a synthetic RTX 5090 must be flagged approximate")
    return len(cases)


def check_kernels(lib):
    n = 0
    versions = [0, 50, 75, 80, 86, 89, 90, 100, 120, 121]
    for err, b, p, (a, mn) in itertools.product([0, 209, 222, 98], versions, versions,
                                                [(8, 0), (8, 6), (8, 9), (9, 0), (12, 0)]):
        want = ref.code_kind(err, b, p, a, mn)
        got = lib.imd_kernel_code_kind(err, b, p, a, mn)
        if got != want:
            raise Fail(f"code kind ({err}, {b}, {p}, {a}.{mn}): host {got}, reference {want}")
        n += 1
    # the rows measured on the dev box (docs/NOTES-18.md): native, sm_86 / sm_80 SASS, error 209 / 222,
    # and a driver-compiled kernel (binary 89, ptx 80)
    meas = [((0, 89, 89, 8, 9), ref.NATIVE), ((0, 86, 86, 8, 9), ref.COMPATIBLE), ((0, 80, 80, 8, 9), ref.COMPATIBLE),
            ((209, 0, 0, 8, 9), ref.UNUSABLE), ((222, 0, 0, 8, 9), ref.UNUSABLE), ((0, 89, 80, 8, 9), ref.RECOMPILED),
            ((0, 90, 80, 9, 0), ref.RECOMPILED), ((0, 89, 89, 9, 0), ref.UNUSABLE)]
    for args, want in meas:
        if lib.imd_kernel_code_kind(*args) != want:
            raise Fail(f"measured row {args}: not {want}")
    for m, c in itertools.product(range(5), range(4)):
        if lib.imd_plan_status_of(m, c) != ref.plan_status(m, c):
            raise Fail(f"plan status ({m}, {c})")
    exp = {(ref.EXACT, ref.NATIVE): ref.P_NATIVE, (ref.EXACT, ref.RECOMPILED): ref.P_RECOMPILED,
           (ref.DEFAULT, ref.RECOMPILED): ref.P_RECOMPILED, (ref.DEFAULT, ref.NATIVE): ref.P_APPROXIMATE,
           (ref.EXACT, ref.COMPATIBLE): ref.P_APPROXIMATE, (ref.UNSUPPORTED, ref.NATIVE): ref.P_UNSUPPORTED,
           (ref.EXACT, ref.UNUSABLE): ref.P_UNSUPPORTED}
    for (m, c), want in exp.items():
        if lib.imd_plan_status_of(m, c) != want:
            raise Fail(f"status table ({m}, {c})")
    for f, vals in (("imd_match_name", range(5)), ("imd_code_kind_name", range(4)), ("imd_plan_status_name", range(4))):
        names = [getattr(lib, f)(v).decode() for v in vals]
        if len(set(names)) != len(names) or "?" in names or getattr(lib, f)(99) != b"?":
            raise Fail(f"{f}: {names}")
    return n


def check_invalid(lib):
    out = Sel(C.sizeof(Sel))
    good = cdev(dev(8, 9, 128, "x"))
    if lib.imd_device_select(None, C.byref(out)) != 1 or lib.imd_device_select(C.byref(good), None) != 1:
        raise Fail("NULL pointers must be INVALID_VALUE")
    small = Sel(8)
    if lib.imd_device_select(C.byref(good), C.byref(small)) != 1:
        raise Fail("short selection struct_size must be INVALID_VALUE")
    short = cdev(dev(8, 9, 128, "x"))
    short.struct_size = 8
    if lib.imd_device_select(C.byref(short), C.byref(out)) != 1:
        raise Fail("short device struct_size must be INVALID_VALUE")
    for bad in (dev(-1, 0, 1, "x"), dev(8, -1, 1, "x"), dev(8, 9, -1, "x")):
        if lib.imd_device_select(C.byref(cdev(bad)), C.byref(out)) != 1:
            raise Fail(f"{bad} must be INVALID_VALUE")
    unterminated = Dev(C.sizeof(Dev), 8, 9, 128, 0, 0, 0, b"")
    C.memset(C.byref(unterminated, Dev.name.offset), ord("A"), 64)
    if lib.imd_device_select(C.byref(unterminated), C.byref(out)) != 1:
        raise Fail("a name without NUL must be INVALID_VALUE")
    badrow = Row(5, 8, 9, 0, b"")   # table id outside IMD_TABLE_COUNT
    if lib.imd_device_select_rows(C.byref(good), C.byref(badrow), 1, C.byref(out)) != 1:
        raise Fail("a row with an unknown table id must be INVALID_VALUE")
    if lib.imd_device_select_rows(C.byref(good), None, 1, C.byref(out)) != 1 or \
            lib.imd_device_select_rows(C.byref(good), None, -1, C.byref(out)) != 1:
        raise Fail("rows NULL with a count, or a negative count, must be INVALID_VALUE")
    # no rows: UNSUPPORTED (a valid answer)
    if csel(lib, dev(8, 9, 128, "x"), [])[0] != ref.UNSUPPORTED:
        raise Fail("no rows: UNSUPPORTED")
    # out of range row access
    r = Row()
    if lib.imd_device_row(-1, C.byref(r)) != 1 or lib.imd_device_row(lib.imd_device_row_count(), C.byref(r)) != 1:
        raise Fail("imd_device_row out of range")
    if lib.imd_device_row_count() != len(ref.ROWS):
        raise Fail("built-in row count differs from spec/devices.py ROWS")
    for i, w in enumerate(ref.ROWS):
        lib.imd_device_row(i, C.byref(r))
        if (r.table_id, r.cc_major, r.cc_minor, r.sm_count, r.name_prefix.decode()) != \
                (w["table_id"], w["cc_major"], w["cc_minor"], w["sm_count"], w["name_prefix"]):
            raise Fail(f"built-in row {i} differs from spec/devices.py ROWS")
    # out is untouched on INVALID_VALUE
    marker = Sel(C.sizeof(Sel), 77, 77, 77, 77, 77)
    lib.imd_device_select(C.byref(cdev(dev(-1, 0, 1, "x"))), C.byref(marker))
    if (marker.match, marker.table_id) != (77, 77):
        raise Fail("an INVALID_VALUE call wrote the output")


def check_mutants():
    """Each mutant of the reference rule must differ from the real reference on the grid."""
    alive = []
    for mut in ref.MUTANTS:
        differs = False
        if mut in ("compat_any_major",):
            differs = any(ref.code_kind(0, b, b, a, mn) != ref.code_kind(0, b, b, a, mn, mut)
                          for b in (80, 86, 89) for a, mn in [(9, 0), (12, 0)])
        elif mut == "recompiled_any_binary":
            vs = (80, 86, 89, 90, 100, 120)
            differs = any(ref.code_kind(0, bn, pt, a, mn) != ref.code_kind(0, bn, pt, a, mn, mut)
                          for bn in vs for pt in vs for a, mn in [(8, 9), (9, 0), (10, 0), (12, 0)])
        elif mut == "recompiled_before_unsupported":
            differs = any(ref.plan_status(m, ref.RECOMPILED) != ref.plan_status(m, ref.RECOMPILED, mut) for m in range(5))
        elif mut == "approx_over_recompiled":
            differs = any(ref.plan_status(m, ref.RECOMPILED) != ref.plan_status(m, ref.RECOMPILED, mut)
                          for m in range(5))
        else:
            for tag, rows in ROW_SETS.items():
                for (a, b), sms, name in itertools.product(CCS, SMS, NAMES):
                    d = dev(a, b, sms, name)
                    if ref.select(d, rows) != ref.select(d, rows, mut):
                        differs = True
                        break
                if differs:
                    break
        if not differs:
            alive.append(mut)
    if alive:
        raise Fail(f"the grid cannot tell these mutants from the rule: {alive}")
    return len(ref.MUTANTS)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--host-lib", default=None)
    args = ap.parse_args()
    path = args.host_lib
    if path is None:
        for d in ("build", "."):
            p = os.path.join(ROOT, d, "libimmerdink_host.so")
            if os.path.exists(p):
                path = p
                break
    lib = load(path)
    try:
        g = check_grid(lib)
        nm = check_named(lib)
        nk = check_kernels(lib)
        check_invalid(lib)
        mu = check_mutants()
    except Fail as e:
        print(f"FAIL  {e}")
        return 1
    print(f"PASS  {g} grid cards, {nm} named cards, {nk} kernel classifications, {mu} mutants killed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
