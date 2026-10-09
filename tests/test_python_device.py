#!/usr/bin/env python3
"""The Python device API (python/immerdink, DESIGN §6d) on the GPU, bit for bit.

    IMMERDINK_LIB=build/libimmerdink.so python3 -m pytest tests/test_python_device.py

No array library is needed: a ctypes helper over libcudart allocates and copies, a minimal
object exposes __cuda_array_interface__ and a hand-built capsule exposes DLPack (0.x and
1.x). Every result is compared bit for bit with spec/oracle.py (and the whole C buffer:
elements C does not address stay untouched). With the allocation shim preloaded
(IMMERDINK_SHIM, LD_PRELOAD), no call of the package may allocate CUDA memory. CuPy /
PyTorch interop runs when importable.
"""

import ctypes as C
import dataclasses
import os
import random
import re
import sys
import threading

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path[:0] = [os.path.join(ROOT, "python"), os.path.join(ROOT, "spec"), HERE]
import immerdink as imd  # noqa: E402
from immerdink import _arrays, _capi, host  # noqa: E402
import oracle as o  # noqa: E402
import test_contract as tc  # noqa: E402


# ------------------------------------------------------------------------------------------
# CUDA helper (libcudart via ctypes) and minimal array producers

class Cuda:
    H2D, D2H = 1, 2

    def __init__(self):
        rt = self.rt = C.CDLL("libcudart.so.13")
        for name, args in {
            "cudaMalloc": [C.POINTER(C.c_void_p), C.c_size_t], "cudaFree": [C.c_void_p],
            "cudaMemcpy": [C.c_void_p, C.c_void_p, C.c_size_t, C.c_int],
            "cudaMemset": [C.c_void_p, C.c_int, C.c_size_t],
            "cudaDeviceSynchronize": [], "cudaStreamCreate": [C.POINTER(C.c_void_p)],
            "cudaStreamDestroy": [C.c_void_p], "cudaStreamSynchronize": [C.c_void_p],
            "cudaStreamBeginCapture": [C.c_void_p, C.c_int],
            "cudaStreamEndCapture": [C.c_void_p, C.POINTER(C.c_void_p)],
            "cudaGraphInstantiate": [C.POINTER(C.c_void_p), C.c_void_p, C.c_ulonglong],
            "cudaGraphLaunch": [C.c_void_p, C.c_void_p], "cudaGraphDestroy": [C.c_void_p],
            "cudaGraphExecDestroy": [C.c_void_p],
        }.items():
            getattr(rt, name).argtypes = args
            getattr(rt, name).restype = C.c_int
        self.live = []

    def ok(self, err, what):
        assert err == 0, f"{what}: CUDA error {err}"

    def malloc(self, nbytes):
        p = C.c_void_p()
        self.ok(self.rt.cudaMalloc(C.byref(p), max(int(nbytes), 1)), "cudaMalloc")
        self.live.append(p.value)
        return p.value

    def free_all(self):
        self.sync()
        for p in self.live:
            self.rt.cudaFree(p)
        self.live = []

    def upload(self, addr, arr):
        self.ok(self.rt.cudaMemcpy(addr, arr.ctypes.data, arr.nbytes, self.H2D), "H2D")

    def download(self, arr, addr):
        self.ok(self.rt.cudaMemcpy(arr.ctypes.data, addr, arr.nbytes, self.D2H), "D2H")

    def sync(self):
        self.ok(self.rt.cudaDeviceSynchronize(), "cudaDeviceSynchronize")

    def stream(self):
        s = C.c_void_p()
        self.ok(self.rt.cudaStreamCreate(C.byref(s)), "cudaStreamCreate")
        return s.value


class Cai:
    """The smallest __cuda_array_interface__ v3 producer (byte strides, as the protocol)."""

    def __init__(self, ptr, shape, bstrides=None, typestr="<f8", readonly=False, stream=None):
        self.__cuda_array_interface__ = {"shape": tuple(shape), "typestr": typestr,
                                         "data": (ptr, readonly), "strides": bstrides,
                                         "version": 3, "stream": stream}


_api = C.pythonapi
_api.PyCapsule_New.argtypes = [C.c_void_p, C.c_char_p, C.c_void_p]
_api.PyCapsule_New.restype = C.py_object


class DLPackArray:
    """A hand-built DLPack producer: 0.x capsules ("dltensor"), or 1.x ("dltensor_versioned")
    when asked with max_version >= (1, 0) and versioned=True. Element strides; records the
    stream it was asked for. The capsules have no destructor (the memory is the test's)."""
    NAME0, NAME1 = b"dltensor", b"dltensor_versioned"

    def __init__(self, ptr, shape, strides=None, code=2, bits=64, device=(2, 0), versioned=False,
                 readonly=False, byte_offset=0):
        self.args = (ptr, tuple(shape), strides, code, bits, device, readonly, byte_offset)
        self.versioned, self.streams, self._keep = versioned, [], []

    def __dlpack_device__(self):
        return self.args[5]

    def _tensor(self):
        ptr, shape, strides, code, bits, device, _, off = self.args
        nd = len(shape)
        sh = (C.c_int64 * max(nd, 1))(*shape)
        st = (C.c_int64 * max(nd, 1))(*strides) if strides is not None else None
        t = _arrays._DLTensor(ptr, _arrays._DLDevice(*device), nd,
                              _arrays._DLDataType(code, bits, 1), sh, st, off)
        self._keep += [sh, st]
        return t

    def __dlpack__(self, stream=None, max_version=None):
        self.streams.append(stream)
        if self.versioned and max_version is not None and tuple(max_version) >= (1, 0):
            m = _arrays._DLManagedTensorVersioned(_arrays._DLPackVersion(1, 0), None, None,
                                                  1 if self.args[6] else 0, self._tensor())
            name = self.NAME1
        else:
            m = _arrays._DLManagedTensor(self._tensor(), None, None)
            name = self.NAME0
        self._keep.append(m)
        return _api.PyCapsule_New(C.addressof(m), name, None)


class DLPack0Only(DLPackArray):
    """A DLPack 0.x producer whose __dlpack__ knows no max_version."""

    def __dlpack__(self, stream=None):
        return super().__dlpack__(stream=stream)


# ------------------------------------------------------------------------------------------
# Fixtures

@pytest.fixture(scope="session")
def lib():
    return imd.load_library(os.environ.get("IMMERDINK_LIB") or None)


@pytest.fixture(scope="session")
def shim():
    p = os.environ.get("IMMERDINK_SHIM")
    if not p:
        return None
    s = C.CDLL(p)
    s.imd_shim_alloc_count.restype = C.c_int64
    return s


@pytest.fixture
def cuda(lib):
    c = Cuda()
    yield c
    c.free_all()


def allocs(shim):
    return None if shim is None else shim.imd_shim_alloc_count()


# ------------------------------------------------------------------------------------------
# Cases against the oracle

class Operand:
    """A host tensor scattered into a device buffer (tc.scatter), as an oracle Desc and as
    a device array of the requested kind (pointer at element 0)."""

    def __init__(self, cuda, rng, desc, T, kind):
        self.desc = desc
        length, base = tc.buffer_for(desc)
        self.buf, self.base = tc.scatter(T, desc, tc.gen(rng, (length,), "uniform"))
        self.dev = cuda.malloc(self.buf.nbytes)
        cuda.upload(self.dev, self.buf)
        self.ptr = self.dev + 8 * base
        self.cuda, self.kind = cuda, kind
        self.arr = self.array(kind)

    def array(self, kind):
        d = self.desc
        if kind == "cai":
            return Cai(self.ptr, d.extent, tuple(8 * s for s in d.stride))
        if kind == "cai_dense":
            return Cai(self.ptr, d.extent, None)
        if kind == "dlpack":
            return DLPackArray(self.ptr, d.extent, d.stride, versioned=True)
        if kind == "dlpack0":
            return DLPack0Only(self.ptr, d.extent, d.stride)
        if kind == "raw":
            return imd.DeviceArray(self.ptr, d.extent, d.stride)
        raise ValueError(kind)

    def tensor(self):
        return o.Tensor(self.buf.tolist(), self.base, self.desc)

    def reset(self):
        self.cuda.upload(self.dev, self.buf)

    def result(self):
        self.cuda.sync()
        out = np.empty_like(self.buf)
        self.cuda.download(out, self.dev)
        return out


def descs(spec, ext, strides=None):
    """Oracle Descs of an einsum spec (labels = code points, as immerdink.contract)."""
    (la, lb, lc) = imd.device._parse(spec)
    strides = strides or (None, None, None)
    out = []
    for lab, st in zip((la, lb, lc), strides):
        e = [ext[ch] for ch in lab]
        out.append(o.Desc(e, host.dense_strides(e) if st is None else st, [ord(ch) for ch in lab]))
    return out


def make(cuda, spec, ext, strides=None, kinds=("cai", "cai", "cai"), seed=0, data="phi"):
    rng = random.Random(seed)
    ds = descs(spec, ext, strides)
    ops = [Operand(cuda, rng, d, tc.gen(rng, tuple(d.extent), data), k) for d, k in zip(ds, kinds)]
    return ops


def oracle(ops, alpha, beta, N, rounding="RN", g=0, mode="EMULATE", packs=None):
    return np.array(o.contract_tensor(*(x.tensor() for x in ops), alpha, beta, N, rounding, g,
                                      o.DIRECT if mode == "DIRECT" else o.EMULATE, packs=packs),
                    dtype=np.float64)


def same_bits(got, want, what):
    bad = tc.bit_mismatches(got, want)
    assert bad.size == 0, f"{what}: {bad.size} elements differ, first {bad[:5]}"


def run_plan(cuda, shim, spec, ext, strides=None, kinds=("cai", "cai", "cai"), alpha=1.0,
             beta=0.0, N=9, g=0, rounding="RN", mode="EMULATE", vec_mode="GENERATE", seed=0,
             stream=None):
    """Plan.contract and immerdink.contract (cached plan) vs the oracle, bit for bit."""
    ops = make(cuda, spec, ext, strides, kinds, seed)
    want = oracle(ops, alpha, beta, N, rounding, g, mode)
    A, B, Cc = (x.arr for x in ops)
    opts = dict(N=N, g=g, rounding=rounding, mode=mode, vec_mode=vec_mode)
    with imd.Plan(spec, A, B, Cc, **opts) as plan:
        assert plan.workspace_bytes == imd.workspace_size(spec, A, B, Cc, **opts)
        ws = (cuda.malloc(plan.workspace_bytes), plan.workspace_bytes)
        n0 = allocs(shim)
        assert plan.contract(A, B, Cc, alpha=alpha, beta=beta, workspace=ws, stream=stream) is Cc
        assert allocs(shim) == n0, "Plan.contract allocated CUDA memory"
        same_bits(ops[2].result(), want, f"{spec} Plan.contract")
        ops[2].reset()
        # the einsum front end, through a private cache (plan made on the first call)
        cache = imd.PlanCache()
        imd.contract(spec, A, B, Cc, alpha=alpha, beta=beta, workspace=ws, plan_cache=cache,
                     stream=stream, **opts)
        same_bits(ops[2].result(), want, f"{spec} immerdink.contract")
        ops[2].reset()
        n0 = allocs(shim)
        imd.contract(spec, A, B, Cc, alpha=alpha, beta=beta, workspace=ws, plan_cache=cache,
                     stream=stream, **opts)
        assert allocs(shim) == n0, "immerdink.contract (cached plan) allocated CUDA memory"
        assert len(cache) == 1
        same_bits(ops[2].result(), want, f"{spec} immerdink.contract (cached)")
        cache.clear()
    return ops, want


# ------------------------------------------------------------------------------------------
# Tests: declarations

def test_capi_declares_every_function():
    hdr = open(os.path.join(ROOT, "include", "immerdink", "immerdink.h")).read()
    names = set(re.findall(r"IMD_API\s+imd_status_t\s+(\w+)\s*\(", hdr))
    assert len(names) == 12 and names <= set(_capi.SIGNATURES), names - set(_capi.SIGNATURES)
    for n in ("imd_tensor_t", "imd_options_t", "imd_operand_t"):
        assert n in hdr or n in open(os.path.join(ROOT, "include", "immerdink",
                                                  "immerdink_host.h")).read()


def test_import_needs_no_library():
    import subprocess
    code = ("import sys; sys.path.insert(0, %r); import immerdink, immerdink.device as d; "
            "from immerdink import _capi; assert not _capi._loaded; print('ok')"
            % os.path.join(ROOT, "python"))
    env = dict(os.environ, IMMERDINK_LIB="/nonexistent/libimmerdink.so")
    env.pop("LD_PRELOAD", None)
    assert subprocess.run([sys.executable, "-c", code], env=env, capture_output=True,
                          text=True).stdout.strip() == "ok"


# ------------------------------------------------------------------------------------------
# Tests: results bit for bit

def test_gemm(cuda, shim):
    run_plan(cuda, shim, "ik,kj->ij", dict(i=13, j=11, k=17), N=9)


def test_gemm_trunc_groups_beta(cuda, shim):
    run_plan(cuda, shim, "ik,jk->ij", dict(i=9, j=24, k=33), N=12, g=5, rounding="TRUNC",
             alpha=-0.75, beta=0.5)


def test_tensor_rank4_permuted(cuda, shim):
    run_plan(cuda, shim, "kjbc,kiac->abij", dict(a=3, b=4, c=5, i=3, j=2, k=4), N=14,
             alpha=1.25, beta=-2.0, kinds=("cai", "dlpack", "raw"))


def test_batched_noncontiguous_negative_strides(cuda, shim):
    ext = dict(l=3, i=5, k=7, j=4)
    # A: gaps and a negative stride; B: k fastest; C: batch innermost, negative i
    sA = [-40, 1, 5 + 1]           # l, i, k (|strides| leave gaps)
    sB = [4 * 7 + 3, 1, 7]          # l, k, j
    sC = [1, -3 * 4, 3]             # l, i, j
    run_plan(cuda, shim, "lik,lkj->lij", ext, strides=(sA, sB, sC), N=8, beta=0.25,
             kinds=("cai", "raw", "dlpack0"))


def test_diagonal_and_scalar(cuda, shim):
    run_plan(cuda, shim, "iik,kj->ij", dict(i=4, k=6, j=5), N=9)
    run_plan(cuda, shim, "ij,ij->", dict(i=6, j=7), N=10, beta=1.5)


def test_rank5_batched(cuda, shim):
    run_plan(cuda, shim, "abkcl,lkdb->dacb", dict(a=2, b=3, c=2, d=3, k=4, l=3), N=9, beta=0.5,
             kinds=("dlpack", "cai_dense", "cai"))


def test_direct_alpha0_resident(cuda, shim):
    run_plan(cuda, shim, "ik,kj->ij", dict(i=5, j=6, k=7), N=9, mode="DIRECT", beta=0.5)
    run_plan(cuda, shim, "ik,kj->ij", dict(i=5, j=6, k=7), N=9, alpha=0.0, beta=0.5)
    run_plan(cuda, shim, "ik,kj->ij", dict(i=8, j=9, k=10), N=9, vec_mode="RESIDENT", beta=0.5)


def test_against_c_call(cuda, shim, lib):
    """Plan.contract == the direct C call with the same descriptors, bit for bit."""
    import harness as hz
    ops = make(cuda, "kjb,kia->abij", dict(a=3, b=4, i=3, j=5, k=6))
    A, B, Cc = (x.arr for x in ops)
    with imd.Plan("kjb,kia->abij", A, B, Cc, N=11) as plan:
        ws = cuda.malloc(plan.workspace_bytes)
        plan.contract(A, B, Cc, beta=0.5, workspace=(ws, plan.workspace_bytes))
        got = ops[2].result()
    ops[2].reset()
    hl = hz.Library(lib.path)
    ts = [hz.make_tensor(x.desc.extent, x.desc.stride, x.desc.label) for x in ops]
    st, p = hl.plan_create(*ts, 11)
    assert st == "SUCCESS"
    assert hl.contract(p, 1.0, ops[0].ptr, ops[1].ptr, 0.5, ops[2].ptr, ws,
                       plan.workspace_bytes) == "SUCCESS"
    hl.plan_destroy(p)
    same_bits(got, ops[2].result(), "Plan.contract vs imd_contract")
    ops[2].reset()
    with imd.Plan.from_descs(*((x.desc.extent, x.desc.stride, x.desc.label) for x in ops),
                             N=11) as plan:   # raw descriptors, integer labels
        plan.contract(A, B, Cc, beta=0.5, workspace=(ws, plan.workspace_bytes))
        same_bits(got, ops[2].result(), "Plan.from_descs")


def test_contract_ptr(cuda, shim):
    ops = make(cuda, "ik,kj->ij", dict(i=6, j=7, k=9))
    want = oracle(ops, 2.0, 0.5, 9)
    with imd.Plan("ik,kj->ij", *(x.arr for x in ops), N=9) as plan:
        ws = cuda.malloc(plan.workspace_bytes)
        n0 = allocs(shim)
        plan.contract_ptr(2.0, ops[0].ptr, ops[1].ptr, 0.5, ops[2].ptr, ws, plan.workspace_bytes, 0)
        assert allocs(shim) == n0
        same_bits(ops[2].result(), want, "contract_ptr")
        with pytest.raises(imd.InsufficientWorkspaceError):
            plan.contract_ptr(1.0, ops[0].ptr, ops[1].ptr, 0.0, ops[2].ptr, ws, 256, 0)


def test_layout_planning_and_lenient_strides(cuda, shim):
    """A plan from Layouts before any array; extent-1 modes' strides are ignored."""
    ext = dict(i=5, x=1, k=6, j=4)
    plan = imd.Plan("ixk,kj->ij", imd.Layout((5, 1, 6)), imd.Layout((6, 4)), imd.Layout((5, 4)),
                    N=9)
    ops = make(cuda, "ixk,kj->ij", ext, strides=([6, 777, 1], None, None))
    want = oracle(ops, 1.0, 0.0, 9)
    ws = cuda.malloc(plan.workspace_bytes)
    plan.contract(*(x.arr for x in ops), workspace=(ws, plan.workspace_bytes))
    same_bits(ops[2].result(), want, "lenient extent-1 stride")
    lw = plan.lowering
    assert lw.status == "SUCCESS" and lw.workspace_bytes == plan.workspace_bytes
    assert plan.device_bytes == 0 and plan.host_bytes == lw.plan_host_bytes > 0
    plan.close()
    plan.close()  # idempotent
    with pytest.raises(ValueError, match="closed"):
        plan.contract(*(x.arr for x in ops), workspace=(ws, plan.workspace_bytes))


# ------------------------------------------------------------------------------------------
# Tests: packed operands (DESIGN §6b)

def oracle_pack_desc(ps):
    t = ps.tensor
    return o.PackDesc(o.Desc(list(t.extent), list(t.stride), list(t.label)), ps.num_moduli,
                      ps.rounding, ps.order, list(ps.contracted), list(ps.batch),
                      list(ps.rows) if ps.order == "GIVEN" else None)


@pytest.mark.parametrize("which", ["A", "B", "AB"])
def test_packed(cuda, shim, which):
    spec, ext, N = "kjb,kia->abij", dict(a=3, b=2, i=4, j=5, k=9), 9
    ops = make(cuda, spec, ext, kinds=("cai", "dlpack", "cai"))
    A, B, Cc = (x.arr for x in ops)
    packs, bufs, ods, datas = [None, None], [None, None], [None, None], [None, None]
    for x in (0, 1):
        if "AB"[x] not in which:
            continue
        ps = imd.pack_spec(spec, A, B, Cc, operand=x, N=N)
        pk = packs[x] = imd.Pack(ps)
        assert (pk.pack_bytes, pk.workspace_bytes) == (pk.layout.pack_bytes, pk.layout.pack_ws_bytes)
        bufs[x] = cuda.malloc(pk.pack_bytes)
        pws = cuda.malloc(pk.workspace_bytes)
        n0 = allocs(shim)
        pk.pack(ops[x].arr, imd.DeviceArray(bufs[x], (pk.pack_bytes,), dtype="u1"),
                workspace=(pws, pk.workspace_bytes))
        assert allocs(shim) == n0, "Pack.pack allocated"
        ods[x] = oracle_pack_desc(ps)
        datas[x] = o.pack(ops[x].tensor(), ods[x])
        cuda.sync()
        got = np.empty(pk.pack_bytes, np.uint8)
        cuda.download(got, bufs[x])
        assert bytes(got) == bytes(datas[x]), f"packed bytes of operand {x}"
    want = oracle(ops, 1.5, 0.5, N, packs=tuple(o.PackedOperand(ods[x], datas[x])
                                                if ods[x] is not None else None for x in (0, 1)))
    same_bits(want, oracle(ops, 1.5, 0.5, N), "desc_for pack == unpacked (oracle)")
    with imd.Plan(spec, A, B, Cc, N=N, packs=packs) as plan:
        assert plan.workspace_bytes == imd.workspace_size(spec, A, B, Cc, N=N, packs=packs)
        assert plan.lowering.packed == sum(1 << x for x in (0, 1) if packs[x] is not None)
        ws = cuda.malloc(plan.workspace_bytes)
        pbuf = tuple((bufs[x], packs[x].pack_bytes) if packs[x] else None for x in (0, 1))
        n0 = allocs(shim)
        plan.contract(A, B, Cc, alpha=1.5, beta=0.5, workspace=(ws, plan.workspace_bytes),
                      packed=pbuf)
        assert allocs(shim) == n0
        same_bits(ops[2].result(), want, f"packed {which}")
        ops[2].reset()
        imd.contract(spec, A, B, Cc, alpha=1.5, beta=0.5, N=N, packs=packs, packed=pbuf,
                     workspace=(ws, plan.workspace_bytes), plan_cache=False)
        same_bits(ops[2].result(), want, f"packed {which} (immerdink.contract)")
        if which == "A":   # a packed slot's buffer missing where it is read (§6b.11)
            with pytest.raises(imd.InvalidValueError):
                plan.contract(A, B, Cc, workspace=(ws, plan.workspace_bytes), packed=None)
        if which != "AB":   # a buffer for the in-line slot
            with pytest.raises(ValueError, match="in-line"):
                plan.contract(A, B, Cc, workspace=(ws, plan.workspace_bytes),
                              packed=tuple((bufs[x], 1) for x in (0, 1)))
        cuda.sync()
        # Pack.close() while the plan uses the pack: closed for new use at once, the C handle
        # kept until the plan is closed (the plan passes it to imd_contract_packed)
        live = [p for p in packs if p is not None]
        for p in live:
            p.close()
            assert p.closed and p._fin.alive
            with pytest.raises(ValueError, match="closed"):
                p.pack(ops[0].arr, (bufs[0] or bufs[1], p.pack_bytes))
            with pytest.raises(ValueError, match="closed"):
                imd.Plan(spec, A, B, Cc, N=N, packs=packs)
        ops[2].reset()
        plan.contract(A, B, Cc, alpha=1.5, beta=0.5, workspace=(ws, plan.workspace_bytes),
                      packed=pbuf)
        same_bits(ops[2].result(), want, f"packed {which} after Pack.close()")
    assert all(not p._fin.alive and p._h is None for p in live)   # destroyed with the plan


def test_pack_mismatch_reason(cuda):
    ops = make(cuda, "ik,kj->ij", dict(i=4, j=5, k=6))
    A, B, Cc = (x.arr for x in ops)
    pk = imd.Pack(imd.pack_spec("ik,kj->ij", A, B, Cc, operand=0, N=9))
    with pytest.raises(imd.PackMismatchError) as e:
        imd.Plan("ik,kj->ij", A, B, Cc, N=10, packs=(pk, None))
    assert e.value.reason == "MODULI" and isinstance(e.value, ValueError)


# ------------------------------------------------------------------------------------------
# Tests: reuse, many plans, cache, threads

def test_plan_reuse_many_calls(cuda, shim):
    ops = make(cuda, "ik,kj->ij", dict(i=5, j=4, k=6))
    A, B, Cc = (x.arr for x in ops)
    with imd.Plan("ik,kj->ij", A, B, Cc, N=8) as plan:
        ws = (cuda.malloc(plan.workspace_bytes), plan.workspace_bytes)
        for t in range(40):
            alpha, beta = 0.5 + t, (-1.0) ** t * 0.25 * (t % 3)
            want = oracle(ops, alpha, beta, 8)
            plan.contract(A, B, Cc, alpha=alpha, beta=beta, workspace=ws)
            same_bits(ops[2].result(), want, f"call {t}")
            ops[2].reset()


def test_many_plans_alive(cuda, shim):
    cases = []
    for t in range(48):
        m, n, k = 2 + t % 5, 3 + t % 4, 2 + t % 7
        N = (2, 3, 8, 9, 14, 16, 20)[t % 7]
        ops = make(cuda, "ik,kj->ij", dict(i=m, j=n, k=k), seed=t,
                   kinds=(("cai", "dlpack", "raw")[t % 3],) * 3)
        plan = imd.Plan("ik,kj->ij", *(x.arr for x in ops), N=N,
                        vec_mode=("GENERATE", "RESIDENT")[t % 2])
        cases.append((ops, plan, N, (cuda.malloc(plan.workspace_bytes), plan.workspace_bytes)))
    for ops, plan, N, ws in cases:          # all alive, interleaved
        plan.contract(*(x.arr for x in ops), beta=0.5, workspace=ws)
    for ops, plan, N, ws in cases:
        same_bits(ops[2].result(), oracle(ops, 1.0, 0.5, N), f"plan N={N}")
    cuda.sync()
    for _, plan, _, _ in cases:
        plan.close()


def test_plan_cache_and_threads(cuda):
    spec, ext = "ik,kj->ij", dict(i=7, j=6, k=8)
    cache = imd.PlanCache()
    nthreads = 8
    sets = [make(cuda, spec, ext, seed=t) for t in range(nthreads)]
    wss = [cuda.malloc(1 << 16) for _ in range(nthreads)]
    streams = [cuda.stream() for _ in range(nthreads)]
    errors = []

    def work(t):
        try:
            for _ in range(10):
                imd.contract(spec, *(x.arr for x in sets[t]), N=9, beta=0.0, plan_cache=cache,
                             workspace=(wss[t], 1 << 16), stream=streams[t])
        except Exception as e:  # noqa: BLE001
            errors.append(e)
    th = [threading.Thread(target=work, args=(t,)) for t in range(nthreads)]
    for x in th:
        x.start()
    for x in th:
        x.join()
    assert not errors, errors
    assert len(cache) == 1                    # one plan for equal layouts and options
    for t in range(nthreads):
        same_bits(sets[t][2].result(), oracle(sets[t], 1.0, 0.0, 9), f"thread {t}")
    plan = cache.get(spec, *(x.arr for x in sets[0]), N=9)
    assert len(cache) == 1 and plan.workspace_bytes <= 1 << 16   # the same plan
    imd.contract(spec, *(x.arr for x in sets[0]), N=10, plan_cache=cache,
                 workspace=(wss[0], 1 << 16))
    assert len(cache) == 2
    cache.clear()
    assert len(cache) == 0
    for s in streams:
        cuda.rt.cudaStreamDestroy(s)


def test_plan_cache_clear_freeze(cuda):
    spec, ext = "ik,kj->ij", dict(i=6, j=5, k=7)
    ops = make(cuda, spec, ext)
    A, B, Cc = (x.arr for x in ops)
    want = oracle(ops, 1.0, 0.0, 9)
    cache = imd.PlanCache()
    pg = cache.get(spec, A, B, Cc, N=9)
    pr = cache.get(spec, A, B, Cc, N=9, vec_mode="RESIDENT")
    assert len(cache) == 2 and set(map(id, cache.plans())) == {id(pg), id(pr)}
    assert pr.device_bytes > 0 and cache.device_bytes == pg.device_bytes + pr.device_bytes
    ws = (cuda.malloc(pg.workspace_bytes), pg.workspace_bytes)
    cache.freeze()
    imd.contract(spec, A, B, Cc, N=9, workspace=ws, plan_cache=cache)  # warmed: no creation
    with pytest.raises(KeyError):
        cache.get(spec, A, B, Cc, N=10)
    with pytest.raises(KeyError):
        imd.contract(spec, A, B, Cc, N=11, workspace=ws, plan_cache=cache)
    assert len(cache) == 2
    cache.freeze(False)
    cache.get(spec, A, B, Cc, N=10)
    assert len(cache) == 3
    same_bits(ops[2].result(), want, "frozen cache")
    cuda.sync()
    cache.clear()           # drops GENERATE plans (still usable by holders), closes RESIDENT
    assert len(cache) == 0 and cache.device_bytes == 0
    assert pr._h is None and pg._h is not None
    ops[2].reset()
    pg.contract(A, B, Cc, workspace=ws)
    same_bits(ops[2].result(), want, "a GENERATE plan dropped by clear()")


def test_resident_resource_warning():
    import gc
    import warnings
    lay = (imd.Layout((4, 5)), imd.Layout((5, 3)), imd.Layout((4, 3)))
    p = imd.Plan("ik,kj->ij", *lay, N=9, vec_mode="RESIDENT")
    with pytest.warns(ResourceWarning, match="RESIDENT"):
        del p
        gc.collect()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        imd.Plan("ik,kj->ij", *lay, N=9, vec_mode="RESIDENT").close()   # closed: no warning
        p = imd.Plan("ik,kj->ij", *lay, N=9)                            # GENERATE: none
        del p
        gc.collect()


def test_cai_stream_policy(cuda):
    s = cuda.stream()
    spec, ext = "ik,kj->ij", dict(i=5, j=4, k=6)
    ops = make(cuda, spec, ext)
    want = oracle(ops, 1.0, 0.0, 9)
    on_s = [Cai(x.ptr, x.desc.extent, None, stream=s) for x in ops]
    legacy = [Cai(x.ptr, x.desc.extent, None, stream=1) for x in ops]
    plan = imd.Plan(spec, *on_s, N=9)       # a layout read: no stream check
    W = (cuda.malloc(plan.workspace_bytes), plan.workspace_bytes)
    assert imd.get_cai_stream_policy() == "check"
    with pytest.raises(ValueError, match="set_cai_stream_policy"):
        plan.contract(*on_s, workspace=W)                   # producer s, call NULL
    with pytest.raises(ValueError, match="stream"):
        imd.contract(spec, *on_s, N=9, workspace=W, plan_cache=imd.PlanCache())
    with pytest.raises(ValueError, match="stream"):          # workspace buffers too
        plan.contract(*legacy, workspace=Cai(W[0], (W[1],), None, "|u1", stream=s))
    plan.contract(*on_s, workspace=W, stream=s)              # the producer's stream
    plan.contract(*legacy, workspace=W)                      # CAI 1 == the call's NULL
    plan.contract(*legacy, workspace=W, stream=0)
    with pytest.raises(ValueError):
        plan.contract(*legacy, workspace=W, stream=2)        # per-thread != legacy
    try:
        imd.set_cai_stream_policy("ignore")
        plan.contract(*on_s, workspace=W)                     # opt-out: the caller orders
        imd.contract(spec, *on_s, N=9, workspace=W, plan_cache=imd.PlanCache())
    finally:
        imd.set_cai_stream_policy("check")
    with pytest.raises(ValueError):
        imd.set_cai_stream_policy("sync")
    same_bits(ops[2].result(), want, "CAI stream policy")
    # packs: X, the packed buffer and the workspace
    pk = imd.Pack(imd.pack_spec(spec, *(x.arr for x in ops), operand=0, N=9))
    buf, pws = cuda.malloc(pk.pack_bytes), cuda.malloc(pk.workspace_bytes)
    with pytest.raises(ValueError, match="stream"):
        pk.pack(on_s[0], (buf, pk.pack_bytes), workspace=(pws, pk.workspace_bytes))
    pk.pack(on_s[0], (buf, pk.pack_bytes), workspace=(pws, pk.workspace_bytes), stream=s)
    cuda.sync()
    pk.close()
    plan.close()
    cuda.rt.cudaStreamDestroy(s)
    # the environment opt-out, read at import
    import subprocess
    code = ("import sys; sys.path.insert(0, %r); import immerdink; "
            "print(immerdink.get_cai_stream_policy())" % os.path.join(ROOT, "python"))
    env = dict(os.environ, IMMERDINK_CAI_SYNC="0")
    env.pop("LD_PRELOAD", None)
    assert subprocess.run([sys.executable, "-c", code], env=env, capture_output=True,
                          text=True).stdout.strip() == "ignore"


def test_dlpack_layout_stream_and_device(cuda):
    ops = make(cuda, "ik,kj->ij", dict(i=4, j=3, k=5), kinds=("dlpack", "dlpack0", "cai"))
    A, B, Cc = (x.arr for x in ops)
    plan = imd.Plan("ik,kj->ij", A, B, Cc, N=9)
    assert A.streams == [-1] and B.streams == [-1]   # layout reads: "do not synchronize"
    imd.workspace_size("ik,kj->ij", A, B, Cc, N=9)
    imd.pack_spec("ik,kj->ij", A, B, Cc, operand=0, N=9)
    assert set(A.streams) == {-1}
    W = (cuda.malloc(plan.workspace_bytes), plan.workspace_bytes)
    plan.contract(A, B, Cc, workspace=W)
    assert A.streams[-1] == 1
    other = DLPackArray(ops[0].ptr, (4, 5), [5, 1], device=(2, plan.device + 1))
    with pytest.raises(ValueError, match="device"):
        plan.contract(other, B, Cc, workspace=W)
    cuda.sync()
    plan.close()


# ------------------------------------------------------------------------------------------
# Tests: streams and capture

class _PtrStream:
    def __init__(self, h):
        self.ptr = h


class _TorchStream:
    def __init__(self, h):
        self.cuda_stream = h


class _ProtoStream:
    def __init__(self, h):
        self.h = h

    def __cuda_stream__(self):
        return (0, self.h)


def test_streams(cuda, shim):
    s = cuda.stream()
    for st in (None, 0, 1, 2, s, _PtrStream(s), _TorchStream(s), _ProtoStream(s), C.c_void_p(s)):
        run_plan(cuda, shim, "ik,kj->ij", dict(i=4, j=3, k=5), N=8, stream=st)
    assert _arrays.stream_handle(_ProtoStream(s)) == s
    with pytest.raises(TypeError):
        _arrays.stream_handle("default")
    # DLPack: the producer is asked for the call's stream (1 = the legacy default for NULL)
    ops = make(cuda, "ik,kj->ij", dict(i=4, j=3, k=5), kinds=("dlpack", "dlpack0", "cai"))
    with imd.Plan("ik,kj->ij", *(x.arr for x in ops), N=8) as plan:
        ws = (cuda.malloc(plan.workspace_bytes), plan.workspace_bytes)
        for st, want in ((None, 1), (0, 1), (2, 2), (s, s)):
            plan.contract(*(x.arr for x in ops), workspace=ws, stream=st)
            assert ops[0].arr.streams[-1] == want and ops[1].arr.streams[-1] == want
    cuda.sync()
    cuda.rt.cudaStreamDestroy(s)


def test_graph_capture(cuda, shim):
    """Plan.contract and immerdink.contract captured into a CUDA graph (global mode) and
    replayed on new input data (a graph keeps the captured pointers)."""
    spec, ext, N = "kjb,kia->abij", dict(a=3, b=2, i=4, j=5, k=7), 9
    s = cuda.stream()
    for front in ("plan", "einsum"):
        ops = make(cuda, spec, ext, kinds=("cai", "dlpack", "cai"))
        A, B, Cc = (x.arr for x in ops)
        cache = imd.PlanCache()
        plan = cache.get(spec, A, B, Cc, N=N)          # plan creation is not capturable
        ws = (cuda.malloc(plan.workspace_bytes), plan.workspace_bytes)
        if front == "einsum":                          # warm the einsum path's cache entry
            imd.contract(spec, A, B, Cc, N=N, workspace=ws, plan_cache=cache)
        graph, gexec = C.c_void_p(), C.c_void_p()
        cuda.ok(cuda.rt.cudaStreamBeginCapture(s, 0), "begin capture")
        if front == "plan":
            plan.contract(A, B, Cc, beta=0.5, workspace=ws, stream=s)
        else:
            imd.contract(spec, A, B, Cc, beta=0.5, N=N, workspace=ws, stream=s, plan_cache=cache)
        cuda.ok(cuda.rt.cudaStreamEndCapture(s, C.byref(graph)), "end capture")
        assert len(cache) == 1
        cuda.ok(cuda.rt.cudaGraphInstantiate(C.byref(gexec), graph, 0), "instantiate")
        rng = random.Random(7)
        for rep in range(3):
            for x in ops[:2]:                          # new data in the same buffers
                x.buf = tc.scatter(tc.gen(rng, tuple(x.desc.extent), "phi"), x.desc, x.buf)[0]
                x.reset()
            ops[2].reset()
            want = oracle(ops, 1.0, 0.5, N)
            cuda.ok(cuda.rt.cudaGraphLaunch(gexec, s), "graph launch")
            cuda.ok(cuda.rt.cudaStreamSynchronize(s), "sync")
            same_bits(ops[2].result(), want, f"{front} graph replay {rep}")
        cuda.rt.cudaGraphExecDestroy(gexec)
        cuda.rt.cudaGraphDestroy(graph)
        cache.clear()
    cuda.rt.cudaStreamDestroy(s)


# ------------------------------------------------------------------------------------------
# Tests: errors

def test_errors(cuda):
    ops = make(cuda, "ik,kj->ij", dict(i=4, j=5, k=6))
    A, B, Cc = (x.arr for x in ops)
    plan = imd.Plan("ik,kj->ij", A, B, Cc, N=9)
    nws = plan.workspace_bytes
    ws = cuda.malloc(nws + 512)
    W = (ws, nws)
    with pytest.raises(TypeError, match="float64"):          # wrong dtype
        plan.contract(Cai(ops[0].ptr, (4, 6), None, "<f4"), B, Cc, workspace=W)
    with pytest.raises(TypeError, match="float64"):
        imd.Plan("ik,kj->ij", DLPackArray(ops[0].ptr, (4, 6), None, bits=32), B, Cc, N=9)
    with pytest.raises(TypeError, match="CUDA"):             # host memory
        plan.contract(np.zeros((4, 6)), B, Cc, workspace=W)
    with pytest.raises(TypeError, match="CUDA"):
        plan.contract(DLPackArray(ops[0].ptr, (4, 6), device=(1, 0)), B, Cc, workspace=W)
    with pytest.raises(TypeError, match="CUDA array"):
        plan.contract([[1.0]], B, Cc, workspace=W)
    with pytest.raises(ValueError, match="shape"):           # shape / stride mismatch
        plan.contract(Cai(ops[0].ptr, (4, 5)), B, Cc, workspace=W)
    with pytest.raises(ValueError, match="strides"):
        plan.contract(Cai(ops[0].ptr, (4, 6), (8, 32)), B, Cc, workspace=W)
    with pytest.raises(TypeError, match="multiple of 8"):
        imd.Plan("ik,kj->ij", Cai(ops[0].ptr, (4, 6), (52, 4)), B, Cc, N=9)
    with pytest.raises(ValueError, match="read-only"):       # readonly output
        plan.contract(A, B, Cai(ops[2].ptr, (4, 5), None, readonly=True), workspace=W)
    with pytest.raises(ValueError, match="read-only"):
        plan.contract(A, B, DLPackArray(ops[2].ptr, (4, 5), versioned=True, readonly=True),
                      workspace=W)
    with pytest.raises(imd.InvalidValueError, match=f"needs {nws}"):   # no workspace
        plan.contract(A, B, Cc)
    with pytest.raises(imd.InsufficientWorkspaceError, match=f"needs {nws}"):
        plan.contract(A, B, Cc, workspace=(ws, nws - 1))
    with pytest.raises(imd.InsufficientWorkspaceError):      # a too-small CAI workspace
        plan.contract(A, B, Cc, workspace=Cai(ws, (nws // 2,), None, "|u1"))
    with pytest.raises(ValueError, match="contiguous"):
        plan.contract(A, B, Cc, workspace=Cai(ws, (8, nws // 8), (2 * nws // 8, 1), "|u1"))
    with pytest.raises(imd.MisalignedError):                 # misaligned (C) / workspace
        plan.contract(Cai(ops[0].ptr + 4, (4, 6)), B, Cc, workspace=W)
    with pytest.raises(imd.MisalignedError):
        plan.contract(A, B, Cc, workspace=(ws + 8, nws))
    with pytest.raises(imd.AliasingError):                   # C overlaps A
        plan.contract(Cai(ops[2].ptr, (4, 6)), B, Cc, workspace=W)
    with pytest.raises(imd.NotSupportedError):               # repeated label in C
        imd.Plan("ij,jk->ii", imd.Layout((3, 3)), imd.Layout((3, 3)), imd.Layout((3, 3)), N=9)
    with pytest.raises(imd.InvalidValueError):               # N out of range
        imd.Plan("ik,kj->ij", A, B, Cc, N=1)
    with pytest.raises(imd.InvalidValueError):               # extents disagree
        imd.Plan("ik,kj->ij", imd.Layout((4, 6)), imd.Layout((5, 5)), imd.Layout((4, 5)), N=9)
    with pytest.raises(ValueError, match="einsum spec"):
        imd.Plan("ik,kj", A, B, Cc, N=9)
    with pytest.raises(ValueError, match="rank"):
        imd.Plan("ikl,kj->ij", A, B, Cc, N=9)
    with pytest.raises(ValueError, match="rounding"):
        imd.Plan("ik,kj->ij", A, B, Cc, N=9, rounding="RZ")
    with pytest.raises(ValueError, match="RESIDENT"):
        imd.contract("ik,kj->ij", A, B, Cc, N=9, vec_mode="RESIDENT", plan_cache=False, workspace=W)
    with pytest.raises(ValueError, match="without packs"):
        plan.contract(A, B, Cc, workspace=W, packed=((ws, 256), None))
    plan.close()


# ------------------------------------------------------------------------------------------
# Tests: precision phases (DESIGN v1.19 §6g), CPU only: a fake libimmerdink, no GPU

class _FakeC:
    """Stands in for libimmerdink's C functions: statuses chosen by the test, every imd_pack
    and imd_contract_packed recorded, no GPU. Output pointers are ctypes byref objects."""

    def __init__(self):
        self.next, self.calls, self.pack_status = 0x1000, [], 0

    def _new(self, out):
        self.next += 0x10
        out._obj.value = self.next
        return 0

    def imd_create(self, out):
        return self._new(out)

    def imd_destroy(self, h):
        return 0

    def imd_pack_create(self, h, desc, out):
        st = self._new(out)
        self.n_of = getattr(self, "n_of", {})
        self.n_of[out._obj.value] = desc._obj.num_moduli
        return st

    def imd_pack_destroy(self, h):
        return 0

    def imd_pack_bytes(self, h, pb, wb):
        pb._obj.value, wb._obj.value = 256 * (4 + self.n_of[h]), 256   # grows with N (F2)
        return 0

    def imd_pack(self, h, X, packed, nbytes, ws, wsb, stream):
        self.calls.append(("pack", self.n_of[h], packed, nbytes))
        return self.pack_status

    def imd_plan_create_packed(self, h, A, B, C, o, pa, pb, out):
        return self._new(out)

    def imd_plan_create(self, h, A, B, C, o, out):
        return self._new(out)

    def imd_workspace_size(self, A, B, C, o, out):
        out._obj.value = 4096
        return 0

    def imd_contract(self, plan, alpha, A, B, beta, C, ws, wsb, stream):
        self.calls.append(("contract", None, 0))
        return 0

    def imd_workspace_size_packed(self, A, B, C, o, packing, out):
        out._obj.value = 4096
        return 0

    def imd_plan_device_bytes(self, plan, out):
        out._obj.value = 0
        return 0

    def imd_plan_destroy(self, plan):
        return 0

    def imd_contract_packed(self, plan, alpha, opA, opB, beta, C, ws, wsb, stream):
        self.calls.append(("contract", opA._obj.packed, opA._obj.packed_bytes))
        return 0


class _FakeLib(_capi.Library):
    """A _capi.Library whose device functions are _FakeC's; the host library is the real
    CUDA-free one (pack descriptors, key digests)."""

    def __init__(self):
        p = os.environ.get("IMMERDINK_LIB")
        sib = os.path.join(os.path.dirname(p), "libimmerdink_host.so") if p else None
        self.path, self.version, self.cudart = "fake", _capi.API_VERSION, None
        self.lib = _FakeC()
        self.host = host.load(sib if sib and os.path.exists(sib) else None)


def test_pack_spec_with_moduli():
    ps = host.PackSpec(host.Desc((5, 7), (7, 1), (1, 3)), 14, "TRUNC", "GIVEN", (3,), (), (1,))
    p8 = ps.with_moduli(8)
    assert p8.num_moduli == 8 and ps.num_moduli == 14
    assert p8 == host.PackSpec(ps.tensor, 8, "TRUNC", "GIVEN", (3,), (), (1,))
    assert p8.with_moduli(14) == ps and hash(p8.with_moduli(14)) == hash(ps)


def test_packed_buffer_guard_cpu():
    """PackedBuffer (DESIGN v1.19 §6g) through a fake library: Pack.pack records the pack at
    enqueue; Plan.contract and immerdink.contract refuse a buffer whose record is not the
    plan slot's pack (PackMismatchError) before any C call, and one never packed; a raw
    buffer is never checked; a failed imd_pack keeps the record, a CUDA_ERROR clears it."""
    lib = _FakeLib()
    spec = "ik,kj->ij"
    lay = (imd.Layout((5, 7)), imd.Layout((7, 4)), imd.Layout((5, 4)))
    ps14 = imd.pack_spec(spec, *lay, operand=0, N=14, lib=lib)
    pk14, pk8 = imd.Pack(ps14, lib=lib), imd.Pack(ps14.with_moduli(8), lib=lib)
    pk8t = imd.Pack(dataclasses.replace(ps14, num_moduli=8, rounding="TRUNC"), lib=lib)
    assert pk8.pack_bytes <= pk14.pack_bytes
    A, B, Cc = (Cai(0x100000 * (t + 1), x.shape) for t, x in enumerate(lay))
    X = A
    p14 = imd.Plan(spec, *lay, N=14, packs=(pk14, None), lib=lib)
    p8 = imd.Plan(spec, *lay, N=8, packs=(pk8, None), lib=lib)
    W, PW = (0x900000, 4096), (0xA00000, 256)
    buf = imd.PackedBuffer((0x800000, pk14.pack_bytes))
    calls = lib.lib.calls

    def contract(plan, packed):
        n0 = len(calls)
        plan.contract(A, B, Cc, workspace=W, packed=(packed, None))
        assert len(calls) == n0 + 1 and calls[-1][0] == "contract"
        return calls[-1]

    def refused(plan, packed, match):
        n0 = len(calls)
        with pytest.raises(imd.PackMismatchError, match=match) as e:
            plan.contract(A, B, Cc, workspace=W, packed=(packed, None))
        assert len(calls) == n0, "a refused call reached the library"
        assert isinstance(e.value, ValueError) and e.value.reason is None

    assert buf.last_pack is None
    refused(p14, buf, "no pack recorded")
    pk14.pack(X, buf, workspace=PW)
    assert buf.last_pack == pk14.record and buf.last_pack.N == 14
    assert calls[-1] == ("pack", 14, 0x800000, pk14.pack_bytes)
    assert contract(p14, buf) == ("contract", 0x800000, pk14.pack_bytes)
    refused(p8, buf, "buffer last packed at N=14, plan expects N=8")
    # re-pack at N = 8 in place (C-pack: the N = 14 buffer holds it)
    assert pk8.pack(X, buf, workspace=PW) is buf and buf.last_pack.N == 8
    refused(p14, buf, "buffer last packed at N=8, plan expects N=14")
    contract(p8, buf)
    with pytest.raises(imd.PackMismatchError, match="plan expects N=14"):   # einsum front end
        imd.contract(spec, A, B, Cc, N=14, packs=(pk14, None), packed=(buf, None),
                     workspace=W, plan_cache=False, lib=lib)
    # a raw buffer is never checked (behaves as before): the stale plan reaches the library
    assert contract(p14, (0x800000, pk14.pack_bytes))[1] == 0x800000
    # rounding and other-key records
    pk8t.pack(X, buf, workspace=PW)
    refused(p8, buf, "last packed with TRUNC, plan expects RN")
    pk8.pack(X, buf, workspace=PW)
    other = imd.Pack(imd.pack_spec(spec, *lay, operand=1, N=8, lib=lib), lib=lib)
    assert other.record.N == 8 and other.record.key_digest != pk8.record.key_digest
    other.pack(Cai(0x200000, (7, 4)), buf, workspace=PW)
    refused(p8, buf, "another key")
    # a second Pack of the same key has the same record: accepted
    same = imd.Pack(ps14.with_moduli(8), lib=lib)
    same.pack(X, buf, workspace=PW)
    assert same.record == pk8.record
    contract(p8, buf)
    # a rejected imd_pack (nothing enqueued) keeps the record; a CUDA_ERROR clears it
    lib.lib.pack_status = 4
    with pytest.raises(imd.InsufficientWorkspaceError):
        pk14.pack(X, buf, workspace=PW)
    assert buf.last_pack == pk8.record
    lib.lib.pack_status = 6
    with pytest.raises(imd.CudaError):
        pk14.pack(X, buf, workspace=PW)
    assert buf.last_pack is None
    lib.lib.pack_status = 0
    # nbytes caps the bytes passed; it may not exceed the buffer; a raw pointer needs it
    capped = imd.PackedBuffer(imd.DeviceArray(0x800000, (pk14.pack_bytes,), dtype="u1"),
                              nbytes=pk8.pack_bytes)
    pk8.pack(X, capped, workspace=PW)
    assert calls[-1] == ("pack", 8, 0x800000, pk8.pack_bytes)
    assert contract(p8, capped) == ("contract", 0x800000, pk8.pack_bytes)
    with pytest.raises(ValueError, match="nbytes"):
        pk14.pack(X, imd.PackedBuffer(imd.DeviceArray(0x800000, (16,), dtype="u1"), 32),
                  workspace=PW)
    with pytest.raises(ValueError, match="nbytes"):
        imd.PackedBuffer(0x800000)
    assert imd.PackedBuffer(0x800000, 512).buf == (0x800000, 512)
    with pytest.raises(TypeError):
        imd.PackedBuffer(buf)
    assert "N=8" in repr(capped) and "never packed" in repr(imd.PackedBuffer((1, 2)))
    for x in (p14, p8):
        x.close()


def test_precision_plans_cpu():
    """The plans and packs a PrecisionPlan names (DESIGN v1.19 §6g.10: "creating these plans
    before the loop covers every phase"), through a fake library: every configs() choice and
    two phases' choices make a Plan from Choice.options() and Packs from pack_specs (in-line,
    packed and the DIRECT level), and a packed call takes a PackedBuffer that the choice's
    own Pack wrote; in a share group one buffer, packed by one member's Pack, serves the
    other member's plan (one key), and a buffer packed at another level is refused."""
    from immerdink import planner, precision
    lib = _FakeLib()
    pl = planner.Planner(lib=lib.host)
    shapes = {"x": ("ik,kj->ij", dict(i=48, k=300, j=40), ((48, 300), (300, 40), (48, 40))),
              "y": ("ik,kl->il", dict(i=48, k=300, l=24), ((48, 300), (300, 24), (48, 24))),
              "z": ("ik,kj->ij", dict(i=40, k=64, j=40), ((40, 64), (64, 40), (40, 40)))}
    pp = precision.PrecisionPlan(
        pl, {n: (spec, ext) for n, (spec, ext, _) in shapes.items()},
        levels=(6, 8, precision.DIRECT), share=[("G", [("x", "A"), ("y", "A")])],
        packs={"x": {"NONE", "A", "AB"}, "y": {"A", "AB"}})
    ops = {n: tuple(Cai(0x100000 * (8 * i + t + 1), sh) for t, sh in enumerate(shp))
           for i, (n, (_, _, shp)) in enumerate(shapes.items())}
    W, PW = (0x900000, 4096), (0xA00000, 256)
    calls = lib.lib.calls
    made = {"inline": 0, "packed": 0, "direct": 0}

    def plan_of(n, ch, packs=None):
        spec, _, shp = shapes[n]
        if packs is None:
            packs = tuple(imd.Pack(ch.pack_specs[r], lib=lib) if r in ch.pack_specs else None
                          for r in "AB")
        plan = imd.Plan(spec, *(imd.Layout(x) for x in shp), lib=lib,
                        packs=None if packs == (None, None) else packs, **ch.options())
        assert {k: plan.options[k] for k in ch.options()} == ch.options()
        return plan, packs

    def run(n, ch, plan, packs, bufs=None):
        A, B, Cc = ops[n]
        if packs == (None, None):
            plan.contract(A, B, Cc, workspace=W)
            made["direct" if ch.mode == "DIRECT" else "inline"] += 1
            return
        if bufs is None:
            bufs = tuple(None if pk is None else imd.PackedBuffer((0xB00000 + 0x100000 * r,
                                                                    pk.pack_bytes))
                         for r, pk in enumerate(packs))
            for pk, X, buf in zip(packs, (A, B), bufs):
                if pk is not None:
                    pk.pack(X, buf, workspace=PW)
        n0 = len(calls)
        plan.contract(A, B, Cc, workspace=W, packed=bufs)
        assert len(calls) == n0 + 1 and calls[-1][0] == "contract"
        made["packed"] += 1

    for n, by in pp.configs().items():
        for lv, ch in by.items():
            assert ch is not None, (n, lv)
            assert (ch.mode == "DIRECT") == (lv == precision.DIRECT)
            plan, packs = plan_of(n, ch)
            run(n, ch, plan, packs)
            plan.close()
    acc = host.Accuracy("NORMWISE", 1e-4)     # N = 6, then x at 1e-6 raises the group to 8
    levels = []
    for targets in ({"x": acc, "y": acc, "z": acc},
                    {"x": host.Accuracy("NORMWISE", 1e-6), "y": acc, "z": acc}):
        ph = pp.phase(targets)
        assert ph.N["x"] == ph.N["y"] != precision.DIRECT and ph.packed["x"] in ("A", "AB")
        levels.append(ph.N["x"])
        cx, cy = ph.choices["x"], ph.choices["y"]
        px, kx = plan_of("x", cx)
        py, ky = plan_of("y", cy)
        run("x", cx, px, kx)
        pz, kz = plan_of("z", ph.choices["z"])         # outside the group, in-line
        run("z", ph.choices["z"], pz, kz)
        pz.close()
        # one buffer for the group: x's Pack writes it, y's plan (made with its own Pack of
        # the same key) reads it
        shared = imd.PackedBuffer((0xC00000, kx[0].pack_bytes))
        kx[0].pack(ops["x"][0], shared, workspace=PW)
        assert shared.last_pack == ky[0].record
        bufs = (shared, None if ky[1] is None else imd.PackedBuffer((0xD00000, ky[1].pack_bytes)))
        if ky[1] is not None:
            ky[1].pack(ops["y"][1], bufs[1], workspace=PW)
        run("y", cy, py, ky, bufs)
        # the shared buffer re-packed at the other level: refused
        other = 6 if ph.N["x"] == 8 else 8
        imd.Pack(cx.pack_specs["A"].with_moduli(other), lib=lib).pack(ops["x"][0], shared,
                                                                      workspace=PW)
        with pytest.raises(imd.PackMismatchError, match=f"plan expects N={ph.N['y']}"):
            py.contract(*ops["y"], workspace=W, packed=bufs)
        for p_ in (px, py):
            p_.close()
    assert levels == [6, 8] and made["inline"] and made["packed"] and made["direct"], \
        (levels, made)


# ------------------------------------------------------------------------------------------
# Interop (only when the libraries are importable)

def test_interop_cupy():
    cp = pytest.importorskip("cupy")
    rng = np.random.default_rng(1)
    a, b = rng.standard_normal((33, 17)), rng.standard_normal((17, 9))
    A, B = cp.asarray(a), cp.asarray(b)
    Cc = cp.zeros((33, 9))
    n = imd.workspace_size("ik,kj->ij", A, B, Cc, N=16)
    ws = cp.empty(n, cp.uint8)
    imd.contract("ik,kj->ij", A, B, Cc, N=16, workspace=ws,
                 stream=cp.cuda.get_current_stream())
    np.testing.assert_allclose(cp.asnumpy(Cc), a @ b, rtol=1e-13, atol=1e-13)


def test_interop_torch():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("no CUDA in torch")
    A = torch.randn(20, 12, dtype=torch.float64, device="cuda")
    B = torch.randn(12, 7, dtype=torch.float64, device="cuda")
    Cc = torch.zeros(7, 20, dtype=torch.float64, device="cuda").T   # a transposed view
    n = imd.workspace_size("ik,kj->ij", A, B, Cc, N=16)
    ws = torch.empty(n, dtype=torch.uint8, device="cuda")
    imd.contract("ik,kj->ij", A, B, Cc, N=16, workspace=ws, stream=torch.cuda.current_stream())
    torch.testing.assert_close(Cc, A @ B, rtol=1e-13, atol=1e-13)
