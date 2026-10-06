#!/usr/bin/env python3
"""ctypes harness: drive libimmerdink.so from Python and compare with spec/.

    python3 tests/harness.py [--lib build/libimmerdink.so]

imd_workspace_size on a few tensor descriptors vs spec/params.py workspace_size. The
`Library` and `Cuda` wrappers are also used by tests/test_contract.py (bit-exact
imd_contract vs spec/oracle.py).
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
          "MISALIGNED", "CUDA_ERROR", "ALLOC_FAILED", "PACK_MISMATCH"]
RN, TRUNC = 0, 1
EMULATE, DIRECT = 0, 1
VEC_GENERATE, VEC_RESIDENT = 0, 1      # DESIGN v1.2 imd_options_t.vec_mode


class Tensor(C.Structure):
    _fields_ = [("rank", C.c_int32), ("extent", C.POINTER(C.c_int64)),
                ("stride", C.POINTER(C.c_int64)), ("label", C.POINTER(C.c_int32))]


class Options(C.Structure):
    _fields_ = [("num_moduli", C.c_int32), ("residue_group", C.c_int32),
                ("rounding", C.c_int32), ("mode", C.c_int32), ("vec_mode", C.c_int32),
                ("k4_tune", C.c_int32)]   # v1.5 imd_options_t.k4_tune (0 off, 1 on)


def make_tensor(extent, stride, label):
    """An imd_tensor_t whose arrays stay alive as attributes of the returned object."""
    r = len(extent)
    t = Tensor(r, (C.c_int64 * r)(*extent), (C.c_int64 * r)(*stride), (C.c_int32 * r)(*label))
    return t


class PackDesc(C.Structure):
    """imd_pack_desc_t (DESIGN v1.7 §6b.5)."""
    _fields_ = [("struct_size", C.c_uint32), ("num_moduli", C.c_int32), ("rounding", C.c_int32),
                ("order", C.c_int32), ("tensor", C.POINTER(Tensor)),
                ("num_contracted", C.c_int32), ("contracted", C.POINTER(C.c_int32)),
                ("num_batch", C.c_int32), ("batch", C.POINTER(C.c_int32)),
                ("num_rows", C.c_int32), ("rows", C.POINTER(C.c_int32))]


class Operand(C.Structure):
    """imd_operand_t (DESIGN v1.7 §6b.5)."""
    _fields_ = [("data", C.c_void_p), ("packed", C.c_void_p), ("packed_bytes", C.c_size_t),
                ("pack", C.c_void_p)]


ORDER_AUTO, ORDER_GIVEN = 0, 1


def make_pack_desc(od):
    """An imd_pack_desc_t of an oracle.PackDesc (arrays kept alive on the struct)."""
    t = make_tensor(od.tensor.extent, od.tensor.stride, od.tensor.label)
    lists = [list(od.contracted), list(od.batch), list(od.rows or [])]
    arrs = [(C.c_int32 * max(len(x), 1))(*x) for x in lists]
    d = PackDesc(C.sizeof(PackDesc), od.N, RN if od.rounding == "RN" else TRUNC,
                 ORDER_AUTO if od.order == "AUTO" else ORDER_GIVEN, C.pointer(t),
                 len(lists[0]), arrs[0] if lists[0] else None,
                 len(lists[1]), arrs[1] if lists[1] else None,
                 len(lists[2]), arrs[2] if lists[2] else None)
    d._keep = (t, arrs)
    return d


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
        self.lib.imd_create.argtypes = [C.POINTER(C.c_void_p)]
        self.lib.imd_destroy.argtypes = [C.c_void_p]
        self.lib.imd_plan_create.argtypes = [C.c_void_p] + [C.POINTER(Tensor)] * 3 + \
            [C.POINTER(Options), C.POINTER(C.c_void_p)]
        self.lib.imd_plan_destroy.argtypes = [C.c_void_p]
        self.lib.imd_contract.argtypes = [C.c_void_p, C.c_double, C.c_void_p, C.c_void_p,
                                          C.c_double, C.c_void_p, C.c_void_p, C.c_size_t,
                                          C.c_void_p]
        self.lib.imd_plan_device_bytes.argtypes = [C.c_void_p, C.POINTER(C.c_size_t)]
        for f in ("imd_create", "imd_destroy", "imd_plan_create", "imd_plan_destroy",
                  "imd_contract", "imd_plan_device_bytes"):
            getattr(self.lib, f).restype = C.c_int
        # v1.7 (DESIGN §6b): packed operands
        L = self.lib
        L.imd_pack_create.argtypes = [C.c_void_p, C.POINTER(PackDesc), C.POINTER(C.c_void_p)]
        L.imd_pack_destroy.argtypes = [C.c_void_p]
        L.imd_pack_bytes.argtypes = [C.c_void_p, C.POINTER(C.c_size_t), C.POINTER(C.c_size_t)]
        L.imd_pack.argtypes = [C.c_void_p, C.c_void_p, C.c_void_p, C.c_size_t, C.c_void_p,
                               C.c_size_t, C.c_void_p]
        L.imd_plan_create_packed.argtypes = [C.c_void_p] + [C.POINTER(Tensor)] * 3 + \
            [C.POINTER(Options), C.c_void_p, C.c_void_p, C.POINTER(C.c_void_p)]
        L.imd_contract_packed.argtypes = [C.c_void_p, C.c_double, C.POINTER(Operand),
                                          C.POINTER(Operand), C.c_double, C.c_void_p, C.c_void_p,
                                          C.c_size_t, C.c_void_p]
        L.imd_workspace_size_packed.argtypes = [C.POINTER(Tensor)] * 3 + \
            [C.POINTER(Options), C.c_void_p, C.POINTER(C.c_size_t)]
        for f in ("imd_pack_create", "imd_pack_destroy", "imd_pack_bytes", "imd_pack",
                  "imd_plan_create_packed", "imd_contract_packed", "imd_workspace_size_packed"):
            getattr(L, f).restype = C.c_int
        self._handle = None
        # Test-only hooks (src/test_hooks.cpp); absent from a library built without them.
        self.has_test_hooks = hasattr(self.lib, "imd_test_set_limits")
        if self.has_test_hooks:
            self.lib.imd_test_set_limits.argtypes = [C.c_int64] * 3
            self.lib.imd_test_set_limits.restype = C.c_int
            self.lib.imd_test_lowering.argtypes = [C.POINTER(Tensor)] * 3 + \
                [C.POINTER(C.c_int64), C.c_int64, C.POINTER(C.c_int64)]
            self.lib.imd_test_lowering.restype = C.c_int
            self.lib.imd_test_plan_vec.argtypes = [C.c_void_p, C.POINTER(C.c_void_p),
                                                   C.POINTER(C.c_int64)]
            self.lib.imd_test_plan_vec.restype = C.c_int
            self.lib.imd_test_set_fused.argtypes = [C.c_int]
            self.lib.imd_test_set_fused.restype = C.c_int
            self.lib.imd_test_set_k4_config.argtypes = [C.c_int]
            self.lib.imd_test_set_k4_config.restype = C.c_int
            self.lib.imd_test_set_k5_one_elem.argtypes = [C.c_int64]
            self.lib.imd_test_set_k5_one_elem.restype = C.c_int
            self.lib.imd_test_set_k5_elems.argtypes = [C.c_int]
            self.lib.imd_test_set_k5_elems.restype = C.c_int
            self.lib.imd_test_plan_sizes.argtypes = [C.c_void_p, C.POINTER(C.c_int64), C.c_int64,
                                                     C.POINTER(C.c_int64)]
            self.lib.imd_test_plan_sizes.restype = C.c_int
            # v1.13 §4b: the K4 operand-pointer check (libimmerdink) and the K-alignment hook
            # (libimmerdink_host, found through libimmerdink's dependency tree).
            self.lib.imd_test_k4_align_check.argtypes = [C.c_int]
            self.lib.imd_test_k4_align_check.restype = C.c_int64
            self.lib.imd_test_set_k_align.argtypes = [C.c_int32]
            self.lib.imd_test_set_k_align.restype = C.c_int
        self.shim = None

    def load_alloc_shim(self, path):
        """The LD_PRELOAD allocation counter (tests/alloc_shim.c); must already be preloaded
        so that it interposes the library's CUDA runtime calls."""
        self.shim = C.CDLL(path)
        self.shim.imd_shim_alloc_count.restype = C.c_int64
        self.shim.imd_shim_launch_count.restype = C.c_int64

    def alloc_count(self):
        """CUDA runtime allocations (device, host, managed, async) seen so far, or None."""
        return None if self.shim is None else self.shim.imd_shim_alloc_count()

    def launch_count(self):
        """Kernel launches (cudaLaunchKernel) seen so far, or None without the shim."""
        return None if self.shim is None else self.shim.imd_shim_launch_count()

    def set_fused(self, mode=-1):
        """Fused launches of plans created afterwards (src/test_hooks.cpp): -1 the plan-time
        rules (v1.4 KF, v1.6 KL), 0 none, 2 KL for both operands of every emulated plan,
        3 as 2 with 64-row KL tiles (several chunks per row)."""
        assert self.lib.imd_test_set_fused(mode) == 0

    def set_k5_one_elem(self, per_sm=-1):
        """K5 one element per thread while b m n <= per_sm x SMs (-1 production, 0 never)."""
        assert self.lib.imd_test_set_k5_one_elem(per_sm) == 0

    def set_k5_elems(self, elems=0):
        """A5: K5 (j-bar fastest) with 1, 2 or 4 elements per thread; 0 the launch rule."""
        assert self.lib.imd_test_set_k5_elems(elems) == 0

    def set_k5_mapping(self, fmode):
        """The K5 mapping of a test run with fused mode `fmode` (test_contract, test_packed):
        the launch rule for -1 (production), 4 per thread for 0 and 1, 2 per thread for the
        forced-KL runs 2 and 3 (A5), so every mapping is checked on every emulated case."""
        self.set_k5_one_elem(0 if fmode >= 0 else -1)
        self.set_k5_elems(2 if fmode in (2, 3) else 0)

    def reset_k5_mapping(self):
        self.set_k5_one_elem(-1)
        self.set_k5_elems(0)

    def set_k4_config(self, config=-1):
        """v1.5 K4 configuration of plans created afterwards: -1 the rule (or k4_tune's
        choice), 0..3 that IMD_K4_* configuration."""
        assert self.lib.imd_test_set_k4_config(config) == 0

    def plan_sizes(self, plan):
        """The int64 record of src/test_hooks.cpp imd_test_plan_sizes for a real plan."""
        cnt = C.c_int64(0)
        buf = (C.c_int64 * 64)()
        assert self.lib.imd_test_plan_sizes(plan, buf, 64, C.byref(cnt)) == 0 and cnt.value <= 64
        return list(buf)[:cnt.value]

    def plan_kernels(self, plan):
        """(fused, kernel launches per imd_contract with alpha != 0) of a real plan."""
        v = self.plan_sizes(plan)
        return bool(v[32]), v[33]

    def plan_fused_large(self, plan):
        """The operands a real plan scales and packs with KL (bit 0: A, bit 1: B; v1.6)."""
        return self.plan_sizes(plan)[37]

    def plan_k4(self, plan):
        """(K4 configuration the plan runs, that of its K4 parameters, its SM count); the
        first two are -1 without K4 (v1.5)."""
        v = self.plan_sizes(plan)
        return v[34], v[35], v[36]

    def set_k_align(self, a=0):
        """DESIGN v1.13 §4b.5: force the K alignment A (16 / 64) of every plan, pack and host
        query created afterwards; 0 = AUTO. Only the C library: callers that model the plan
        (params / oracle) set their K_ALIGN_POLICY too (test_contract.k_align)."""
        assert self.lib.imd_test_set_k_align(int(a)) == 0

    def k4_align_check(self, mode=-1):
        """imd_test_k4_align_check: 1 = enable the K4 operand-pointer = 0 mod A check and reset
        its count, 0 = disable, -1 = query. Returns the violations since the last reset."""
        return self.lib.imd_test_k4_align_check(mode)

    def set_limits(self, k_chunk=0, max_batch=0, max_n_tiles=0):
        """Override K4's launch limits (0 = production value); see src/test_hooks.cpp."""
        assert self.lib.imd_test_set_limits(k_chunk, max_batch, max_n_tiles) == 0

    def lowering(self, A, B, Cc):
        """(status, dict) of the library's §5 lowering (src/test_hooks.cpp layout)."""
        cnt = C.c_int64(0)
        st = self.lib.imd_test_lowering(C.byref(A), C.byref(B), C.byref(Cc), None, 0, C.byref(cnt))
        if st != 0:
            return STATUS[st], None
        buf = (C.c_int64 * cnt.value)()
        st = self.lib.imd_test_lowering(C.byref(A), C.byref(B), C.byref(Cc), buf, cnt.value,
                                        C.byref(cnt))
        v = list(buf)
        out = {"swapped": bool(v[0]), "m": v[1], "n": v[2], "k": v[3], "b": v[4],
               "order": {}, "fused": {}}
        p = 5
        for g in "IJKL":
            nl = v[p]
            out["order"][g] = v[p + 1:p + 1 + nl]
            p += 1 + nl
            nm = v[p]
            p += 1
            out["fused"][g] = [tuple(v[p + 4 * t:p + 4 * t + 4]) for t in range(nm)]
            p += 4 * nm
        out["pivot"] = [(bool(v[p]), v[p + 1]), (bool(v[p + 2]), v[p + 3])]
        return STATUS[st], out

    def handle(self):
        if self._handle is None:
            h = C.c_void_p()
            st = self.lib.imd_create(C.byref(h))
            assert st == 0, STATUS[st]
            self._handle = h
        return self._handle

    def workspace_size(self, A, B, Cc, N, g=0, rounding=RN, mode=EMULATE, vec=VEC_GENERATE):
        out = C.c_size_t(0)
        st = self.lib.imd_workspace_size(C.byref(A), C.byref(B), C.byref(Cc),
                                         C.byref(Options(N, g, rounding, mode, vec)), C.byref(out))
        name = self.lib.imd_status_string(st).decode()
        assert name == "IMD_STATUS_" + STATUS[st], name
        return STATUS[st], out.value

    MAX_PLAN_VEC = 64 << 20   # test guard: scatter-vector bytes (workspace or resident)

    def plan_create(self, A, B, Cc, N, g=0, rounding=RN, mode=EMULATE, vec=VEC_GENERATE,
                    k4_tune=0):
        """(status, plan handle or None). Test guard: a plan whose scatter vectors
        (8 (2m + 2n + 2k + 3b) bytes) exceed MAX_PLAN_VEC is never created (a descriptor
        that imd_workspace_size rejects fails plan creation before any allocation)."""
        if self.has_test_hooks and self.workspace_size(A, B, Cc, N, g, rounding, mode, vec)[0] == \
                "SUCCESS":
            st, lw = self.lowering(A, B, Cc)
            if st == "SUCCESS":
                nvec = 8 * (2 * (lw["m"] + lw["n"] + lw["k"]) + 3 * lw["b"])
                assert nvec <= self.MAX_PLAN_VEC, f"plan vectors of {nvec} bytes (test guard)"
        plan = C.c_void_p()
        st = self.lib.imd_plan_create(self.handle(), C.byref(A), C.byref(B), C.byref(Cc),
                                      C.byref(Options(N, g, rounding, mode, vec, k4_tune)),
                                      C.byref(plan))
        return STATUS[st], (plan if st == 0 else None)

    def plan_destroy(self, plan):
        assert self.lib.imd_plan_destroy(plan) == 0

    def plan_device_bytes(self, plan):
        out = C.c_size_t(12345)
        st = self.lib.imd_plan_device_bytes(plan, C.byref(out))
        return STATUS[st], out.value

    def plan_vec(self, plan):
        """(device address or None, entries) of the plan's resident scatter vectors (test hook)."""
        dev, cnt = C.c_void_p(), C.c_int64(0)
        assert self.lib.imd_test_plan_vec(plan, C.byref(dev), C.byref(cnt)) == 0
        return dev.value, cnt.value

    def contract(self, plan, alpha, A, B, beta, Cp, ws, ws_bytes, stream=None):
        return STATUS[self.lib.imd_contract(plan, alpha, A, B, beta, Cp, ws, ws_bytes, stream)]

    # ---- v1.7 (DESIGN §6b): packed operands --------------------------------------------

    def pack_create(self, desc):
        """(status, imd_pack_t or None) of a PackDesc (make_pack_desc)."""
        h = C.c_void_p()
        st = self.lib.imd_pack_create(self.handle(), C.byref(desc), C.byref(h))
        return STATUS[st], (h if st == 0 else None)

    def pack_destroy(self, pack):
        assert self.lib.imd_pack_destroy(pack) == 0

    def pack_bytes(self, pack):
        pb, wb = C.c_size_t(7), C.c_size_t(7)
        assert self.lib.imd_pack_bytes(pack, C.byref(pb), C.byref(wb)) == 0
        return pb.value, wb.value

    def pack(self, pack, X, packed, packed_bytes, ws, ws_bytes, stream=None):
        return STATUS[self.lib.imd_pack(pack, X, packed, packed_bytes, ws, ws_bytes, stream)]

    def plan_create_packed(self, A, B, Cc, N, packA, packB, g=0, rounding=RN, mode=EMULATE,
                           vec=VEC_GENERATE, k4_tune=0):
        plan = C.c_void_p()
        st = self.lib.imd_plan_create_packed(self.handle(), C.byref(A), C.byref(B), C.byref(Cc),
                                             C.byref(Options(N, g, rounding, mode, vec, k4_tune)),
                                             packA, packB, C.byref(plan))
        return STATUS[st], (plan if st == 0 else None)

    def contract_packed(self, plan, alpha, opA, opB, beta, Cp, ws, ws_bytes, stream=None):
        """opA / opB: Operand structs (or None for a NULL imd_operand_t*)."""
        return STATUS[self.lib.imd_contract_packed(plan, alpha,
                                                   C.byref(opA) if opA is not None else None,
                                                   C.byref(opB) if opB is not None else None,
                                                   beta, Cp, ws, ws_bytes, stream)]

    def plan_packed(self, plan):
        """(packed slots: bit 0 / 1 = caller's A / B, pack bytes of post-swap A, B) of a plan."""
        v = self.plan_sizes(plan)
        return v[38], v[39], v[40]


class Cuda:
    """The few CUDA runtime calls the tests need (same libcudart as libimmerdink)."""

    H2D, D2H = 1, 2

    def __init__(self):
        self.rt = C.CDLL("libcudart.so.13")
        self.rt.cudaMalloc.argtypes = [C.POINTER(C.c_void_p), C.c_size_t]
        self.rt.cudaFree.argtypes = [C.c_void_p]
        self.rt.cudaMemcpy.argtypes = [C.c_void_p, C.c_void_p, C.c_size_t, C.c_int]
        self.rt.cudaMemset.argtypes = [C.c_void_p, C.c_int, C.c_size_t]
        self.rt.cudaGetErrorString.argtypes = [C.c_int]
        self.rt.cudaGetErrorString.restype = C.c_char_p
        self.rt.cudaMemGetInfo.argtypes = [C.POINTER(C.c_size_t), C.POINTER(C.c_size_t)]

    def check(self, err, what):
        if err != 0:
            raise RuntimeError(f"{what}: {self.rt.cudaGetErrorString(err).decode()}")

    def malloc(self, nbytes):
        p = C.c_void_p()
        self.check(self.rt.cudaMalloc(C.byref(p), max(int(nbytes), 1)), "cudaMalloc")
        return p

    def free(self, p):
        self.check(self.rt.cudaFree(p), "cudaFree")

    def upload(self, p, arr):
        self.check(self.rt.cudaMemcpy(p, arr.ctypes.data, arr.nbytes, self.H2D), "cudaMemcpy H2D")

    def download(self, arr, p):
        self.check(self.rt.cudaMemcpy(arr.ctypes.data, p, arr.nbytes, self.D2H), "cudaMemcpy D2H")

    def upload_raw(self, addr, arr):
        """Host array -> device address (an int, e.g. an offset into an allocation)."""
        self.check(self.rt.cudaMemcpy(C.c_void_p(addr), arr.ctypes.data, arr.nbytes, self.H2D),
                   "cudaMemcpy H2D")

    def download_raw(self, arr, addr):
        self.check(self.rt.cudaMemcpy(arr.ctypes.data, C.c_void_p(addr), arr.nbytes, self.D2H),
                   "cudaMemcpy D2H")

    def memset(self, p, value, nbytes):
        self.check(self.rt.cudaMemset(p, value, nbytes), "cudaMemset")

    def mem_info(self):
        free, total = C.c_size_t(0), C.c_size_t(0)
        self.check(self.rt.cudaMemGetInfo(C.byref(free), C.byref(total)), "cudaMemGetInfo")
        return free.value, total.value

    def sync(self):
        self.check(self.rt.cudaDeviceSynchronize(), "cudaDeviceSynchronize (kernel error)")

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
