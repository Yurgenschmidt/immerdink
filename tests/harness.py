#!/usr/bin/env python3
"""ctypes harness: drive libimmerdink.so from Python and compare with spec/.

    python3 tests/harness.py [--lib build/libimmerdink.so]

M0: imd_workspace_size on a few tensor descriptors vs spec/params.py workspace_size.
M1 adds the bit-exact comparison of imd_contract against spec/oracle.py here (the
`Library` wrapper is the place to add imd_plan_create / imd_contract bindings).
"""

import argparse
import ctypes as C
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "spec"))
import params  # noqa: E402

STATUS = ["SUCCESS", "INVALID_VALUE", "NOT_SUPPORTED", "ALIASING", "INSUFFICIENT_WORKSPACE",
          "MISALIGNED", "CUDA_ERROR"]
RN, TRUNC = 0, 1
EMULATE, DIRECT = 0, 1


class Tensor(C.Structure):
    _fields_ = [("rank", C.c_int32), ("extent", C.POINTER(C.c_int64)),
                ("stride", C.POINTER(C.c_int64)), ("label", C.POINTER(C.c_int32))]


class Options(C.Structure):
    _fields_ = [("num_moduli", C.c_int32), ("residue_group", C.c_int32),
                ("rounding", C.c_int32), ("mode", C.c_int32)]


def make_tensor(extent, stride, label):
    """An imd_tensor_t whose arrays stay alive as attributes of the returned object."""
    r = len(extent)
    t = Tensor(r, (C.c_int64 * r)(*extent), (C.c_int64 * r)(*stride), (C.c_int32 * r)(*label))
    return t


def dense(extent):
    """Dense strides, last mode fastest."""
    st, acc = [0] * len(extent), 1
    for d in reversed(range(len(extent))):
        st[d] = acc
        acc *= max(extent[d], 1)
    return st


class Library:
    def __init__(self, path):
        self.lib = C.CDLL(path)
        f = self.lib.imd_workspace_size
        f.argtypes = [C.POINTER(Tensor)] * 3 + [C.POINTER(Options), C.POINTER(C.c_size_t)]
        f.restype = C.c_int
        self.lib.imd_status_string.argtypes = [C.c_int]
        self.lib.imd_status_string.restype = C.c_char_p

    def workspace_size(self, A, B, Cc, N, g=0, rounding=RN, mode=EMULATE):
        out = C.c_size_t(0)
        st = self.lib.imd_workspace_size(C.byref(A), C.byref(B), C.byref(Cc),
                                         C.byref(Options(N, g, rounding, mode)), C.byref(out))
        name = self.lib.imd_status_string(st).decode()
        assert name == "IMD_STATUS_" + STATUS[st], name
        return STATUS[st], out.value


def default_lib():
    for d in ("build", "."):
        p = os.path.join(ROOT, d, "libimmerdink.so")
        if os.path.exists(p):
            return p
    return os.path.join(ROOT, "build", "libimmerdink.so")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--lib", default=default_lib())
    args = ap.parse_args()
    lib = Library(args.lib)

    i, j, k, l, x = 1, 2, 3, 4, 5
    # (description, A, B, C, (m, n, k, b) after lowering, N, g, mode)
    cases = [
        ("matrix 8192^3 N=14 (DESIGN §4a worked example)",
         ([8192, 8192], [i, k]), ([8192, 8192], [k, j]), ([8192, 8192], [i, j]),
         (8192, 8192, 8192, 1), 14, 14, EMULATE),
        ("batched, permuted C, b=10",
         ([10, 250, 300], [l, i, k]), ([300, 10, 120], [k, l, j]), ([250, 10, 120], [i, l, j]),
         (250, 120, 300, 10), 10, 0, EMULATE),
        ("diagonal in A plus an A-only label",
         ([17, 17, 9, 3], [i, i, k, x]), ([9, 33], [k, j]), ([17, 33], [i, j]),
         (17, 33, 27, 1), 9, 2, EMULATE),
        ("outer product (K empty -> DIRECT dispatch)",
         ([100], [i]), ([200], [j]), ([100, 200], [i, j]), (100, 200, 1, 1), 14, 0, EMULATE),
        ("DIRECT mode",
         ([64, 64], [i, k]), ([64, 64], [k, j]), ([64, 64], [i, j]), (64, 64, 64, 1), 14, 0, DIRECT),
        ("empty output",
         ([0, 64], [i, k]), ([64, 64], [k, j]), ([0, 64], [i, j]), (0, 64, 64, 1), 14, 0, EMULATE),
    ]
    fails = 0
    for desc, (eA, lA), (eB, lB), (eC, lC), (m, n, kk, b), N, g, mode in cases:
        A = make_tensor(eA, dense(eA), lA)
        B = make_tensor(eB, dense(eB), lB)
        Cc = make_tensor(eC, dense(eC), lC)
        st, got = lib.workspace_size(A, B, Cc, N, g, mode=mode)
        want = params.workspace_size(m, n, kk, b, N, g, params.DIRECT if mode else params.EMULATE)
        ok = st == "SUCCESS" and got == want
        fails += not ok
        print(f"{'PASS' if ok else 'FAIL'}  {desc}: {st} {got} (params.workspace_size {want})")

    st, _ = lib.workspace_size(make_tensor([2], [1], [i]), make_tensor([3], [1], [i]),
                               make_tensor([2], [1], [i]), 9)
    ok = st == "INVALID_VALUE"
    fails += not ok
    print(f"{'PASS' if ok else 'FAIL'}  extent mismatch -> {st}")
    print(f"{len(cases) + 1 - fails}/{len(cases) + 1} passed")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
