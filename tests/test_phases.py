#!/usr/bin/env python3
"""Precision phases on the GPU (DESIGN v1.19 §6g; NOTES-C1 §4.1, stage 0): plans and packs
at several N share one workspace and one packed buffer, bit for bit against spec/oracle.py.

    python3 tests/test_phases.py [--lib build/libimmerdink.so] [--shim ...] --suite NAME [-v]

Each suite is its own ctest, so each runs in a fresh process (graph_phases relies on it, and
main refuses graph_phases after another suite).
  repack_n            (ctest packed_repack_n) X packed at N = 14 into a buffer of
                      pack_bytes(14) and contracted; re-packed at N = 8 in place and
                      contracted with the N = 8 plan; both bit-exact; every packed byte ==
                      oracle.pack; the N = 8 pack leaves [pack_bytes(8), ...) untouched (a
                      smoke check: F3's evidence is write_extent); the stale N = 14 plan with
                      a PackedBuffer raises PackMismatchError before any launch, and with the
                      raw buffer returns SUCCESS and a wrong C (the documented limit, §6b.7 /
                      §6b.10 #3); back to N = 14. Role A and the post-swap B role (tail rows),
                      each case's role asserted. imd_pack's workspace: the maximum over N.
  workspace_shared_n  one workspace of imd_workspace_size(N = 14, g = 0) for plans N in
                      {5, 8, 14} (several g, both vec modes), called alternately on one
                      stream, in-line and with a packed operand re-packed before each call
                      into one buffer of pack_bytes(14) (C-ws, C-pack); all bit-exact; no
                      allocation (the LD_PRELOAD shim); the guard zones around the workspace
                      and the buffer untouched. imd_pack's workspace: the maximum over N
                      (outside F1-F3).
  graph_phases        one CUDA graph per phase, N = 6 (warm: launched before capture) and
                      N = 10 (COLD: captured before any launch at N = 10 in this process,
                      CUDA_MODULE_LOADING=LAZY), sharing one workspace, replayed alternately on
                      new data, bit-exact; a cold packed phase (imd_pack + imd_contract_packed
                      at N = 12, never launched, its pack kernel never launched) captured and
                      replayed too. Capture uses cudaStreamCaptureModeGlobal.
  write_extent        (ctest pack_write_extent; F3, NOTES-C1 IR17) imd_pack writes exactly
                      [0, pack_bytes): each pack lands in an exactly sized window of a buffer
                      filled with a sentinel, twice (0xA5, then 0x5A): every byte outside keeps
                      the sentinel and every byte inside is the same both times (all written)
                      and == oracle.pack (when small); the same for imd_pack's workspace. Every
                      imd_pack launch shape, ASSERTED (write_extent_shape, the 42 of
                      WRITE_EXTENT_SHAPES): KL read along rows (tiles of 64 / 32 / 16 / 8 rows;
                      one or two chunks on chip, 8 values per thread, or re-read chunks, 4),
                      each with radix 1 and radix > 1 (batch-fastest X; two row modes), and
                      along k-bar (tiles of 64 .. 1 rows; the 1-row tile with 1, 2 or >= 3
                      chunks); W = 2 and 3; kp_prep's gaps and tail, and its kK0MaxModes
                      instance (nine fused modes); a gathered K (k_affine = 0); r in {1, 15,
                      16, 17, 33}, b in {1, 3, 5}; the K alignment 16 and AUTO (64 for k > 96);
                      N in {2, 3, 8, 14, 17, 20}; RN and TRUNC; pack_pivot's transcription ==
                      the library's on every case's plan view, and the pack key's digest
                      (X's fused lists, the shape's input) == the oracle's (imd_pack_layout)
                      on every case. Each pack, in an EXACTLY sized
                      allocation, is then read by imd_contract_packed with the in-line partner
                      on every producer path (fused -1: KF or KL by rule; 0: K1 + K2 / K3; 2:
                      KL; 3: KL with 64-row tiles; every 12th case under each of the four K4
                      configurations too), each == the plain imd_contract bit for bit
                      (imd_pack_desc_for packs reproduce it); both post-swap roles asserted.
Environment: IMD_TEST_EXACT=1 (write_extent): every allocation exactly sized, no guard zones,
no oracle (for compute-sanitizer memcheck --padding: tests/memcheck_pack_extent.sh);
IMD_TEST_PACK_SHORT=n: the packed allocation n bytes short of pack_bytes while imd_pack is
told pack_bytes (the mutation memcheck must report); IMD_TEST_MAX_CASES=n: the first n cases
(no coverage assertion).
"""

import argparse
import ctypes as C
import os
import random
import sys
import time

# graph_phases: lazy module loading (the CUDA 12+ default) made explicit, so a cold capture
# really loads its kernels inside the capture; set before libcudart initializes
os.environ["CUDA_MODULE_LOADING"] = "LAZY"

import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path[:0] = [os.path.join(ROOT, "python"), os.path.join(ROOT, "spec"), HERE]
import immerdink as imd  # noqa: E402
import oracle as o  # noqa: E402
import params  # noqa: E402
import harness as hz  # noqa: E402
import test_contract as tc  # noqa: E402
import test_packed as tp  # noqa: E402

Fail = tc.Fail
GUARD = tc.GUARD
EXACT = os.environ.get("IMD_TEST_EXACT", "0") == "1"
SHORT = int(os.environ.get("IMD_TEST_PACK_SHORT", "0"))
MAX_CASES = int(os.environ.get("IMD_TEST_MAX_CASES", "0"))   # write_extent: 0 = all
I, J, K, L = tc.I, tc.J, tc.K, tc.L
STATS = {"calls": 0, "packs": 0, "bytes": 0}


class Cuda(hz.Cuda):
    """hz.Cuda plus streams, graphs and sub-range memsets."""

    def __init__(self):
        super().__init__()
        rt = self.rt
        for name, args in {
            "cudaStreamCreate": [C.POINTER(C.c_void_p)], "cudaStreamDestroy": [C.c_void_p],
            "cudaStreamSynchronize": [C.c_void_p],
            "cudaStreamBeginCapture": [C.c_void_p, C.c_int],
            "cudaStreamEndCapture": [C.c_void_p, C.POINTER(C.c_void_p)],
            "cudaGraphInstantiate": [C.POINTER(C.c_void_p), C.c_void_p, C.c_ulonglong],
            "cudaGraphLaunch": [C.c_void_p, C.c_void_p], "cudaGraphDestroy": [C.c_void_p],
            "cudaGraphExecDestroy": [C.c_void_p],
            "cudaMemcpyAsync": [C.c_void_p, C.c_void_p, C.c_size_t, C.c_int, C.c_void_p],
            "cudaGetLastError": [],
        }.items():
            getattr(rt, name).argtypes = args
            getattr(rt, name).restype = C.c_int

    def memset_at(self, addr, value, nbytes):
        self.check(self.rt.cudaMemset(C.c_void_p(addr), value, nbytes), "cudaMemset")

    def stream(self):
        s = C.c_void_p()
        self.check(self.rt.cudaStreamCreate(C.byref(s)), "cudaStreamCreate")
        return s.value

    def bytes_at(self, addr, n):
        out = np.empty(n, np.uint8)
        if n:
            self.download_raw(out, addr)
        return out


class Shim:
    """The LD_PRELOAD allocation / launch counter (tests/alloc_shim.c), or None counts."""

    def __init__(self, path):
        self.s = None
        if path:
            self.s = C.CDLL(path)
            self.s.imd_shim_alloc_count.restype = C.c_int64
            self.s.imd_shim_launch_count.restype = C.c_int64

    def allocs(self):
        return None if self.s is None else self.s.imd_shim_alloc_count()

    def launches(self):
        return None if self.s is None else self.s.imd_shim_launch_count()


def phi(rng, n):
    """DESIGN's test data, phi = 4: (u - 1/2) exp(4 g), numpy (fast for large operands)."""
    return (rng.random(n) - 0.5) * np.exp(4.0 * rng.standard_normal(n))


class Operand:
    """A host tensor of `desc` scattered into a device buffer; .arr a DeviceArray at element 0."""

    def __init__(self, cuda, desc, rng):
        self.desc, self.cuda = desc, cuda
        length, self.base = tc.buffer_for(desc)
        self.dev = cuda.malloc(8 * length)
        self.fill(rng)
        self.arr = imd.DeviceArray(self.dev.value + 8 * self.base, desc.extent, desc.stride)

    def fill(self, rng):
        length, _ = tc.buffer_for(self.desc)
        T = phi(rng, int(np.prod(self.desc.extent))).reshape(self.desc.extent)
        self.buf, _ = tc.scatter(T, self.desc, phi(rng, length))
        self.cuda.upload(self.dev, self.buf)

    def tensor(self):
        return o.Tensor(self.buf.tolist(), self.base, self.desc)

    def reset(self):
        self.cuda.upload(self.dev, self.buf)

    def result(self):
        self.cuda.sync()
        out = np.empty_like(self.buf)
        self.cuda.download(out, self.dev)
        return out

    def free(self):
        self.cuda.free(self.dev)


def oracle(ops, alpha, beta, N, rounding="RN"):
    return np.array(o.contract_tensor(*(x.tensor() for x in ops), alpha, beta, N, rounding, 0,
                                      o.EMULATE), dtype=np.float64)


def same_bits(got, want, what):
    bad = tc.bit_mismatches(got, want)
    if len(bad):
        i = int(bad[0])
        raise Fail(f"{what}: {len(bad)}/{got.size} C elements differ; first at {i}: gpu "
                   f"{got[i]!r} oracle {want[i]!r}")
    STATS["calls"] += 1


def spec_of(A, B, Cd):
    """An einsum spec of oracle Descs with single-character labels (I J K L -> i j k l)."""
    ch = {I: "i", J: "j", K: "k", L: "l"}
    return ",".join("".join(ch[x] for x in d.label) for d in (A, B)) + "->" + \
        "".join(ch[x] for x in Cd.label)


def lays(A, B, Cd):
    return tuple(imd.Layout(d.extent, d.stride) for d in (A, B, Cd))


def oracle_pack_desc(ps):
    t = ps.tensor
    return o.PackDesc(o.Desc(list(t.extent), list(t.stride), list(t.label)), ps.num_moduli,
                      ps.rounding, ps.order, list(ps.contracted), list(ps.batch),
                      list(ps.rows) if ps.order == "GIVEN" else None)


# ---------------------------------------------------------------------------------------
# packed_repack_n
# ---------------------------------------------------------------------------------------

def suite_repack_n(lib, cuda, shim, verbose):
    rng = np.random.default_rng(1414)
    prng = random.Random(1414)
    cases = [
        # (name, A, B, C, x, role): x = the caller's operand packed, role its post-swap role
        # (0: A, 1: B), asserted
        ("ik,kj->ij k=130 (role A)", *tc.matrix(prng, 21, 19, 130, a_kfast=True, b_kfast=False,
                                                c_jfast=True), 0, 0),
        # C i-fastest: the role swap puts the caller's A in the post-swap B role, r = 17 rows
        # of each of b = 3 batches at batch stride r k_hat, the tail after the last block
        ("lik,lkj->lij k=40 (post-swap B role, tail)",
         o.Desc([3, 17, 40], tc.layout(prng, [3, 17, 40]), [L, I, K]),
         o.Desc([3, 40, 6], tc.layout(prng, [3, 40, 6], order=[1, 2, 0]), [L, K, J]),
         o.Desc([3, 17, 6], tc.layout(prng, [3, 17, 6], order=[1, 2, 0]), [L, I, J]), 0, 1),
    ]
    for name, A, B, Cd, x, want_role in cases:
        spec = spec_of(A, B, Cd)
        ops = [Operand(cuda, d, rng) for d in (A, B, Cd)]
        X = ops[x]
        try:
            ps14 = imd.pack_spec(spec, *lays(A, B, Cd), operand=x, N=14)
            pk = {N: imd.Pack(ps14.with_moduli(N)) for N in (14, 8)}
            packs = {N: (pk[N], None) if x == 0 else (None, pk[N]) for N in pk}
            plans = {N: imd.Plan(spec, *lays(A, B, Cd), N=N, packs=packs[N]) for N in pk}
            role = plans[14].lowering.swapped ^ (x == 1)       # 1: the post-swap B role
            if role != want_role:
                raise Fail(f"{name}: X in the post-swap {'AB'[role]} role, the case is about "
                           f"{'AB'[want_role]}")
            pb14, pb8 = pk[14].pack_bytes, pk[8].pack_bytes
            if not (pb8 <= pb14 and pk[8].workspace_bytes == pk[14].workspace_bytes and
                    plans[8].workspace_bytes <= plans[14].workspace_bytes):
                raise Fail(f"{name}: sizes not monotone in N (F1 / F2 / I1)")
            alloc = cuda.malloc(pb14 + GUARD)
            cuda.memset(alloc, 0xA5, pb14 + GUARD)
            # imd_pack's own workspace is outside F1-F3 (its independence of N is I1, a
            # certificate): sized once as the maximum of the per-N queries
            npw = max(p.workspace_bytes for p in pk.values())
            pws = cuda.malloc(npw)
            ws = cuda.malloc(plans[14].workspace_bytes)
            W, PW = (ws.value, plans[14].workspace_bytes), (pws.value, npw)
            buf = imd.PackedBuffer((alloc.value, pb14))
            want = {N: oracle(ops, 1.0, 0.5, N) for N in pk}
            opack = {N: bytes(o.pack(X.tensor(), oracle_pack_desc(ps14.with_moduli(N))))
                     for N in pk}

            def run(N, packed):
                ops[2].reset()
                plans[N].contract(*(y.arr for y in ops), beta=0.5, workspace=W,
                                  packed=(packed, None) if x == 0 else (None, packed))
                return ops[2].result()

            def check_pack(N, tail_value):
                cuda.sync()
                got = cuda.bytes_at(alloc.value, pb14 + GUARD)
                n = pk[N].pack_bytes
                if bytes(got[:n]) != opack[N]:
                    raise Fail(f"{name}: the N={N} packed bytes != oracle.pack")
                if (got[n:] != tail_value).any():
                    raise Fail(f"{name}: imd_pack at N={N} wrote beyond pack_bytes({N}) = {n}")
                STATS["packs"] += 1

            # 1. N = 14 into the N = 14 buffer
            pk[14].pack(X.arr, buf, workspace=PW)
            check_pack(14, 0xA5)
            same_bits(run(14, buf), want[14], f"{name} N=14")
            # 2. re-pack at N = 8 in place; everything above pack_bytes(8) poisoned first
            cuda.memset_at(alloc.value + pb8, 0x5A, pb14 + GUARD - pb8)
            pk[8].pack(X.arr, buf, workspace=PW)
            check_pack(8, 0x5A)
            same_bits(run(8, buf), want[8], f"{name} N=8 after the re-pack")
            # 3. the stale N = 14 plan: refused on the host, nothing launched
            n0 = shim.launches()
            try:
                run(14, buf)
                raise Fail(f"{name}: the stale N=14 plan accepted a buffer packed at N=8")
            except imd.PackMismatchError as e:
                if "buffer last packed at N=8, plan expects N=14" not in str(e):
                    raise Fail(f"{name}: PackMismatchError message {e}") from None
            if n0 is not None and shim.launches() != n0:
                raise Fail(f"{name}: a refused contract launched kernels")
            # 4. the documented limit: with the raw buffer the library cannot tell (it never
            #    compares bytes with handles): SUCCESS and a wrong C
            got = run(14, (alloc.value, pb14))
            if not len(tc.bit_mismatches(got, want[14])):
                raise Fail(f"{name}: the stale plan on the N=8 bytes gave the N=14 result")
            # 5. back to N = 14 (every byte of [0, pack_bytes(14)) rewritten)
            pk[14].pack(X.arr, buf, workspace=PW)
            check_pack(14, 0x5A)
            same_bits(run(14, buf), want[14], f"{name} N=14 again")
            try:
                run(8, buf)
                raise Fail(f"{name}: the N=8 plan accepted a buffer packed at N=14")
            except imd.PackMismatchError:
                pass
            for p in plans.values():
                p.close()
            for p in pk.values():
                p.close()
            for d in (alloc, pws, ws):
                cuda.free(d)
        finally:
            for y in ops:
                y.free()
        if verbose:
            print(f"  ok  {name}: post-swap role {'AB'[role]}, pack_bytes {pb14} (N=14) / "
                  f"{pb8} (N=8) in one buffer; 3 bit-exact phases, stale plan refused",
                  flush=True)


# ---------------------------------------------------------------------------------------
# workspace_shared_n
# ---------------------------------------------------------------------------------------

def suite_workspace_shared_n(lib, cuda, shim, verbose):
    rng = np.random.default_rng(1415)
    prng = random.Random(1415)
    s = cuda.stream()
    for name, (A, B, Cd), phases in (
            # KF (k_hat <= 1024): in-line, both vec modes, g < N and g = 0
            ("ik,kj->ij k=530", tc.matrix(prng, 24, 19, 530, c_jfast=True),
             ((5, 0, "GENERATE"), (8, 0, "GENERATE"), (14, 0, "GENERATE"), (8, 3, "RESIDENT"),
              (14, 1, "GENERATE"), (5, 2, "RESIDENT"))),
            # KL (k_hat > 1024), the role swap
            ("ik,kj->ij k=1100 (swap)", tc.matrix(prng, 9, 33, 1100, c_jfast=False),
             ((14, 0, "GENERATE"), (5, 5, "GENERATE"), (8, 0, "RESIDENT")))):
        spec = spec_of(A, B, Cd)
        ops = [Operand(cuda, d, rng) for d in (A, B, Cd)]
        outs = {}
        try:
            plans = {ph: imd.Plan(spec, *lays(A, B, Cd), N=ph[0], g=ph[1], vec_mode=ph[2])
                     for ph in phases}
            nws = imd.workspace_size(spec, *lays(A, B, Cd), N=14, g=0)
            if any(p.workspace_bytes > nws for p in plans.values()):
                raise Fail(f"{name}: a phase needs more than ws(N=14, g=0) (C-ws)")
            ws = cuda.malloc(nws + 2 * GUARD)
            cuda.memset(ws, 0xA5, nws + 2 * GUARD)
            W = (ws.value + GUARD, nws)
            Cs = {ph: Operand(cuda, Cd, rng) for ph in phases}
            want = {N: oracle(ops[:2] + [Cs[phases[0]]], 1.0, 0.0, N)
                    for N in sorted({ph[0] for ph in phases})}
            a0 = shim.allocs()
            for rnd in range(3):      # alternately on one stream; stale bytes of other N stay
                for ph in (phases if rnd % 2 == 0 else phases[::-1]):
                    plans[ph].contract(ops[0].arr, ops[1].arr, Cs[ph].arr, workspace=W, stream=s)
                cuda.check(cuda.rt.cudaStreamSynchronize(s), "sync")
                for ph in phases:
                    same_bits(Cs[ph].result(), want[ph[0]], f"{name} {ph} round {rnd}")
            if a0 is not None and shim.allocs() != a0:
                raise Fail(f"{name}: the phase calls allocated")
            wsg = cuda.bytes_at(ws.value, nws + 2 * GUARD)
            if (wsg[:GUARD] != 0xA5).any() or (wsg[GUARD + nws:] != 0xA5).any():
                raise Fail(f"{name}: bytes outside [ws, ws + ws(14)) written")
            # packed: the caller's A re-packed before each call into one buffer of
            # pack_bytes(14); the packed workspace of N = 14 for every phase
            ps14 = imd.pack_spec(spec, *lays(A, B, Cd), operand=0, N=14)
            Ns = sorted({ph[0] for ph in phases})
            pk = {N: imd.Pack(ps14.with_moduli(N)) for N in Ns}
            pplans = {N: imd.Plan(spec, *lays(A, B, Cd), N=N, packs=(pk[N], None)) for N in Ns}
            npw = imd.workspace_size(spec, *lays(A, B, Cd), N=14, packs=(pk[14], None))
            pb = pk[14].pack_bytes
            if any(pk[N].pack_bytes > pb or pplans[N].workspace_bytes > npw for N in Ns):
                raise Fail(f"{name}: a packed phase needs more than the N=14 sizes (C-ws, C-pack)")
            pbuf_alloc = cuda.malloc(pb + GUARD)
            cuda.memset(pbuf_alloc, 0x5A, pb + GUARD)
            # imd_pack's own workspace is outside F1-F3 (I1 is a certificate): the maximum
            # of the per-phase queries, as a caller sizing once does
            npws = max(pk[N].workspace_bytes for N in Ns)
            pws = cuda.malloc(npws)
            buf = imd.PackedBuffer((pbuf_alloc.value, pb))
            a0 = shim.allocs()
            for rnd in range(2):
                for N in (Ns if rnd == 0 else Ns[::-1]):
                    pk[N].pack(ops[0].arr, buf, workspace=(pws.value, npws), stream=s)
                    pplans[N].contract(ops[0].arr, ops[1].arr, Cs[phases[0]].arr,
                                       workspace=(ws.value + GUARD, npw), stream=s,
                                       packed=(buf, None))
                    cuda.check(cuda.rt.cudaStreamSynchronize(s), "sync")
                    same_bits(Cs[phases[0]].result(), want[N], f"{name} packed N={N} round {rnd}")
            if a0 is not None and shim.allocs() != a0:
                raise Fail(f"{name}: the packed phase calls allocated")
            if (cuda.bytes_at(pbuf_alloc.value + pb, GUARD) != 0x5A).any():
                raise Fail(f"{name}: a pack wrote beyond pack_bytes(14)")
            outs[name] = len(phases)
            for p in list(plans.values()) + list(pplans.values()):
                p.close()
            for p in pk.values():
                p.close()
            for c in Cs.values():
                c.free()
            for d in (ws, pbuf_alloc, pws):
                cuda.free(d)
        finally:
            for y in ops:
                y.free()
        if verbose:
            print(f"  ok  {name}: {len(phases)} phases {[ph[:2] for ph in phases]} on one "
                  f"workspace of {nws} bytes (N=14, g=0), 3 rounds; packed N in {Ns} re-packed "
                  f"into one buffer of {pb} bytes; bit-exact, no allocation", flush=True)
    cuda.rt.cudaStreamDestroy(s)


# ---------------------------------------------------------------------------------------
# graph_phases
# ---------------------------------------------------------------------------------------

def suite_graph_phases(lib, cuda, shim, verbose):
    """Must be the first GPU work at N = 10 and N = 12 in the process (a fresh ctest)."""
    rng = np.random.default_rng(1416)
    prng = random.Random(1416)
    A, B, Cd = tc.matrix(prng, 24, 19, 530, c_jfast=True)
    spec = spec_of(A, B, Cd)
    ops = [Operand(cuda, d, rng) for d in (A, B, Cd)]
    C6, C10 = Operand(cuda, Cd, rng), Operand(cuda, Cd, rng)
    s = cuda.stream()
    graphs = []
    try:
        cache = imd.PlanCache()             # plan creation is not capturable: all made first
        p6 = cache.get(spec, *lays(A, B, Cd), N=6)
        p10 = cache.get(spec, *lays(A, B, Cd), N=10)
        cache.freeze()
        nws = max(p6.workspace_bytes, p10.workspace_bytes)
        if nws != imd.workspace_size(spec, *lays(A, B, Cd), N=10):
            raise Fail("the N=10 workspace is not the larger one (C-ws)")
        ws = cuda.malloc(nws)
        W = (ws.value, nws)
        # warm phase: one launch at N = 6 before its capture
        p6.contract(ops[0].arr, ops[1].arr, C6.arr, workspace=W, stream=s)
        cuda.check(cuda.rt.cudaStreamSynchronize(s), "warm-up")

        def capture(fn):
            g, ge = C.c_void_p(), C.c_void_p()
            cuda.check(cuda.rt.cudaStreamBeginCapture(s, 0), "begin capture (global mode)")
            err = None
            try:
                fn()
            except Exception as e:  # noqa: BLE001
                err = e
            end = cuda.rt.cudaStreamEndCapture(s, C.byref(g))
            if err is not None:
                raise Fail(f"a call inside the capture failed: {err}") from err
            cuda.check(end, "end capture")
            cuda.check(cuda.rt.cudaGraphInstantiate(C.byref(ge), g, 0), "instantiate")
            graphs.append((g, ge))
            return ge

        # the COLD phase first: nothing at N = 10 has run in this process
        g10 = capture(lambda: p10.contract(ops[0].arr, ops[1].arr, C10.arr, beta=0.5,
                                           workspace=W, stream=s))
        g6 = capture(lambda: p6.contract(ops[0].arr, ops[1].arr, C6.arr, beta=0.5,
                                         workspace=W, stream=s))
        times = []
        for rep in range(3):
            for y in ops[:2]:
                y.fill(rng)
            for c in (C6, C10):
                c.reset()
            want6 = oracle(ops[:2] + [C6], 1.0, 0.5, 6)
            want10 = oracle(ops[:2] + [C10], 1.0, 0.5, 10)
            t0 = time.perf_counter()
            for ge in ((g10, g6) if rep % 2 == 0 else (g6, g10)):   # alternately, one stream
                cuda.check(cuda.rt.cudaGraphLaunch(ge, s), "graph launch")
            cuda.check(cuda.rt.cudaStreamSynchronize(s), "replay")
            times.append(time.perf_counter() - t0)
            same_bits(C6.result(), want6, f"graph N=6 replay {rep}")
            same_bits(C10.result(), want10, f"graph N=10 (cold capture) replay {rep}")
        # a cold PACKED phase: imd_pack (its KL launch never ran here) + imd_contract_packed at
        # N = 12 (never launched), captured into one graph
        ps = imd.pack_spec(spec, *lays(A, B, Cd), operand=0, N=12)
        with imd.Pack(ps) as pk:
            pp = imd.Plan(spec, *lays(A, B, Cd), N=12, packs=(pk, None))
            pbuf = cuda.malloc(pk.pack_bytes)
            pws = cuda.malloc(pk.workspace_bytes)
            buf = imd.PackedBuffer((pbuf.value, pk.pack_bytes))
            npw = pp.workspace_bytes
            if npw > imd.workspace_size(spec, *lays(A, B, Cd), N=12):
                raise Fail("a packed role increased the workspace (I3)")

            def packed_call():
                pk.pack(ops[0].arr, buf, workspace=(pws.value, pk.workspace_bytes), stream=s)
                pp.contract(ops[0].arr, ops[1].arr, C10.arr, beta=0.5, workspace=(ws.value, npw),
                            stream=s, packed=(buf, None))
            g12 = capture(packed_call)
            for rep in range(2):
                for y in ops[:2]:
                    y.fill(rng)
                C10.reset()
                cuda.memset(pbuf, 0xA5, pk.pack_bytes)   # the graph rewrites every packed byte
                want12 = oracle(ops[:2] + [C10], 1.0, 0.5, 12)
                cuda.check(cuda.rt.cudaGraphLaunch(g12, s), "graph launch")
                cuda.check(cuda.rt.cudaStreamSynchronize(s), "replay")
                same_bits(C10.result(), want12, f"graph packed N=12 (cold capture) replay {rep}")
            pp.close()
            cuda.free(pbuf)
            cuda.free(pws)
        cuda.free(ws)
        if verbose:
            print(f"  ok  graphs N=6 (warm) and N=10 (cold capture, "
                  f"CUDA_MODULE_LOADING={os.environ.get('CUDA_MODULE_LOADING')}) share one "
                  f"workspace of {nws} bytes, replayed alternately 3x bit-exact (host time per "
                  f"pair {', '.join(f'{t * 1e3:.2f}' for t in times)} ms); packed N=12 cold "
                  f"capture (imd_pack + imd_contract_packed) replayed 2x bit-exact", flush=True)
    finally:
        for g, ge in graphs:
            cuda.rt.cudaGraphExecDestroy(ge)
            cuda.rt.cudaGraphDestroy(g)
        cuda.rt.cudaStreamDestroy(s)
        for y in ops + [C6, C10]:
            y.free()


# ---------------------------------------------------------------------------------------
# write_extent (pack_write_extent, F3)
# ---------------------------------------------------------------------------------------

I2, K2, K3, L2, L3, R3 = 11, 12, 13, 14, 15, 16     # extra labels (write_extent)


def write_extent_cases():
    """(N, rounding, A policy, r, k, b, X's storage kind, X is the caller's A). X is b x r x
    k stored
      "rows"     rows fastest (radix 1: the row mode has the smallest stride);
      "kbar"     k fastest;
      "batch"    the batch mode fastest, then rows, then k (b > 1, r > 1): KL reads along
                 rows with radix r (rho = lo + r d, a warp along the batches);
      "rows2"    r = (r1, r2): two row modes, the second fastest in X but second in the
                 plan's row order (C stores the first fastest): radix r1, from the rows;
      "kgather"  k = (k1, k2): two K modes X cannot fuse (k_affine = 0: KL loads v_k);
      "many"     nine modes (3 rows, 3 K, 3 batches) with gaps: nine fused modes, so kp_prep
                 runs its kK0MaxModes instance (r, k, b are fixed by the kind).
    Together they cover every launch shape of imd_pack's KL (host/analysis.hpp kl_launch,
    pack_kl_lg_rows_for; write_extent_shape), which suite_write_extent asserts: read along
    rows, k_hat <= 64 / 128 / 256 (64 / 32 / 16-row tiles) and 8-row tiles with one or two
    chunks on chip (8 values per thread) or re-read chunks (4 values), each with radix 1 and
    radix > 1; along k-bar, tiles of 64 .. 1 rows (tk = pow2 >= k_hat in [64, 4096]), the
    1-row tile with one or two chunks (8 values) or more (4 values); each with W = 2 (N <=
    16) and 3 (N >= 17), RN and TRUNC (A = 64 wherever k > 96 allows it); r and b with and
    without a tail."""
    ks = (2, 17, 97, 200, 600, 1030, 5000, 9000)
    rs = (1, 15, 16, 17, 33)
    combos = [(N, rnd, pol, b) for N in (2, 3, 8, 14, 20) for rnd in ("RN", "TRUNC")
              for pol in (16, "AUTO") for b in (1, 3)]
    out, i = [], 0
    for k in ks:
        for kind in ("rows", "kbar"):
            for r in rs:
                N, rnd, pol, b = combos[(i * 7) % len(combos)]
                out.append((N, rnd, pol, r, k, b, kind, i % 2 == 0))
                i += 1
    # the largest planes at W = 3; along k-bar with 8-row tiles (k_hat in (256, 512]); rows
    # on chip in 8- and 32-row tiles at W = 3, RN and TRUNC (gaps of the cycle above)
    out += [(20, "TRUNC", "AUTO", 33, 9000, 1, "kbar", True),
            (20, "RN", 16, 17, 1030, 3, "rows", False),
            (20, "RN", "AUTO", 17, 300, 3, "kbar", True),
            (8, "TRUNC", 16, 15, 300, 1, "kbar", False),
            (20, "RN", 16, 33, 600, 1, "rows", True),
            (20, "RN", "AUTO", 15, 600, 3, "rows", False),
            (14, "TRUNC", "AUTO", 17, 97, 3, "rows", True),
            (3, "TRUNC", 16, 33, 97, 1, "rows", False)]
    # 8-row tiles with one chunk (k_hat in (256, 512]) along rows, the 1-row tile with one
    # chunk (k_hat in (2048, 4096]) along k-bar; W = 2 and 3
    out += [(8, "RN", "AUTO", 17, 300, 3, "rows", True),
            (20, "TRUNC", 16, 33, 300, 1, "rows", False),
            (14, "TRUNC", 16, 15, 3000, 3, "kbar", False),
            (20, "RN", "AUTO", 17, 3000, 1, "kbar", True)]
    # radix > 1 (review 2026-10-07 must-fix 1): batch-fastest X on every along-rows tile (64,
    # 32, 16 rows; 8 rows with one and two chunks on chip; re-read chunks), W = 2 and 3
    kb = [17, 97, 200, 300, 600, 1030, 5000]
    for j, k in enumerate(kb):
        for N in (8, 20) if j % 2 == 0 else (14, 17):
            r, b = (15, 17, 33)[(j + N) % 3], (3, 5)[(j + N) % 2]
            out.append((N, ("RN", "TRUNC")[(j + N) % 2], (16, "AUTO")[j % 2], r, k, b, "batch",
                        (j + N) % 2 == 0))
    # radix > 1 from two row modes; a gathered K; nine fused modes (kp_prep kK0MaxModes)
    out += [(8, "RN", 16, (3, 11), 300, 1, "rows2", True),
            (20, "TRUNC", "AUTO", (5, 7), 1030, 1, "rows2", False),
            (14, "RN", "AUTO", (4, 9), 97, 1, "rows2", True),
            (8, "TRUNC", 16, 33, (10, 13), 1, "kgather", True),
            (20, "RN", "AUTO", 17, (40, 31), 3, "kgather", False),
            (8, "RN", "AUTO", 12, 105, 12, "many", True),
            (17, "TRUNC", 16, 12, 105, 12, "many", False)]
    return out


def write_extent_descs(prng, t, case):
    """(A, B, C, x) of write_extent case t: X (kind as in write_extent_cases) and a partner of
    7 columns; C stored last mode fastest, or (t % 3 == 0) first mode fastest."""
    N, rnd, pol, r, k, b, kind, x_is_a = case
    np_ = 7
    if kind in ("rows", "kbar", "batch"):
        ext_x = [b, r, k] if b > 1 else [r, k]
        lab_x = [L, I, K] if b > 1 else [I, K]
        ordx = {"rows": [1, 2, 0] if b > 1 else [0, 1], "kbar": None, "batch": [0, 1, 2]}[kind]
        Xd = o.Desc(ext_x, tc.layout(prng, ext_x, order=ordx), lab_x)
        ext_p = [b, k, np_] if b > 1 else [k, np_]
        lab_p = [L, K, J] if b > 1 else [K, J]
        Pd = o.Desc(ext_p, tc.layout(prng, ext_p), lab_p)
        ext_c = [b, r, np_] if b > 1 else [r, np_]
        lab_c = [L, I, J] if b > 1 else [I, J]
        Cd = o.Desc(ext_c, tc.layout(prng, ext_c, order=None if t % 3 else
                                     list(range(len(ext_c)))), lab_c)
    elif kind == "rows2":
        # X: I2 fastest, then K, then I; C: I fastest, so the plan (and the pack) orders the
        # rows I, I2 and the smallest X stride is the second row mode
        (r1, r2) = r
        Xd = o.Desc([r1, r2, k], tc.layout(prng, [r1, r2, k], order=[1, 2, 0]), [I, I2, K])
        Pd = o.Desc([k, np_], tc.layout(prng, [k, np_]), [K, J])
        Cd = o.Desc([r1, r2, np_], tc.layout(prng, [r1, r2, np_], order=[0, 1, 2]), [I, I2, J])
    elif kind == "kgather":
        # X: K fastest, then I, then K2: K and K2 never fuse in X (r > 1)
        (k1, k2) = k
        ext_x = [b, r, k1, k2] if b > 1 else [r, k1, k2]
        lab_x = [L, I, K, K2] if b > 1 else [I, K, K2]
        ordx = [2, 1, 3, 0] if b > 1 else [1, 0, 2]
        Xd = o.Desc(ext_x, tc.layout(prng, ext_x, order=ordx), lab_x)
        ext_p = [b, k1, k2, np_] if b > 1 else [k1, k2, np_]
        lab_p = [L, K, K2, J] if b > 1 else [K, K2, J]
        Pd = o.Desc(ext_p, tc.layout(prng, ext_p), lab_p)
        ext_c = [b, r, np_] if b > 1 else [r, np_]
        lab_c = [L, I, J] if b > 1 else [I, J]
        Cd = o.Desc(ext_c, tc.layout(prng, ext_c), lab_c)
    else:   # "many": r = 2 x 3 x 2, k = 5 x 3 x 7, b = 2 x 2 x 3, every X mode after a gap
        rows, kk, bb = ([I, I2, R3], [2, 3, 2]), ([K, K2, K3], [5, 3, 7]), ([L, L2, L3], [2, 2, 3])
        lab_x, ext_x = rows[0] + kk[0] + bb[0], rows[1] + kk[1] + bb[1]
        ordx = list(range(len(ext_x)))
        prng.shuffle(ordx)
        Xd = o.Desc(ext_x, tc.layout(prng, ext_x, order=ordx, pad=1.0), lab_x)
        Pd = o.Desc(kk[1] + bb[1] + [np_], tc.layout(prng, kk[1] + bb[1] + [np_]),
                    kk[0] + bb[0] + [J])
        Cd = o.Desc(rows[1] + bb[1] + [np_], tc.layout(prng, rows[1] + bb[1] + [np_]),
                    rows[0] + bb[0] + [J])
    A, B = (Xd, Pd) if x_is_a else (Pd, Xd)
    return A, B, Cd, 0 if x_is_a else 1


def pack_pivot(groups, h_step):
    """host/index_map.cpp pack_pivot (the gather tiling), transcribed: groups = (extent,
    stride) of the row class's fused modes, then L's, first fastest; h_step = the first K
    mode's stride. (along_rows, radix): along rows iff the smallest nonzero |stride| of the
    groups is below |h_step| (or h_step = 0); radix = the extents before that mode."""
    best, radix, best_radix = None, 1, 1
    for e, s in groups:
        if s and (best is None or abs(s) < best):
            best, best_radix = abs(s), radix
        radix *= e
    if best is not None and (h_step == 0 or best < abs(h_step)):
        return True, best_radix
    return False, 1


def write_extent_shape(pv, N):
    """imd_pack's KL launch (pack_exec's one-operand view: rows R then L, host/analysis.cpp)
    for pack view pv: (along_rows, lg_th, values per thread, chunks per row (3: >= 3), W,
    radix > 1), by host/analysis.hpp pack_kl_lg_rows_for and kl_launch."""
    along, radix = pack_pivot(pv.fused["R"] + pv.fused["L"],
                              pv.fused["K"][0][1] if pv.fused["K"] else 0)
    lg_tk = 6
    while lg_tk < (9 if along else 12) and (1 << lg_tk) < pv.k_hat:
        lg_tk += 1
    nc = -(-pv.k_hat // (1 << lg_tk))
    return (along, 12 - lg_tk, 4 if nc > 2 else 8, min(nc, 3), params.pack_words(N), radix > 1)


# every launch shape imd_pack's KL can take, in write_extent_shape's terms
WRITE_EXTENT_SHAPES = (
    {(True, lg, v, nc, W, rx) for lg, v, nc in ((6, 8, 1), (5, 8, 1), (4, 8, 1), (3, 8, 1),
                                                (3, 8, 2), (3, 4, 3))
     for W in (2, 3) for rx in (False, True)} |
    {(False, lg, v, nc, W, False) for lg, v, nc in [(lg, 8, 1) for lg in range(7)] +
     [(0, 8, 2), (0, 4, 3)] for W in (2, 3)})


def suite_write_extent(lib, cuda, shim, verbose):
    if not lib.has_test_hooks:
        raise Fail("suite write_extent needs the test hooks (fused paths, K alignment)")
    rng = np.random.default_rng(1417)
    prng = random.Random(1417)
    n_pack, n_contract, n_oracle = 0, 0, 0
    shapes, roles, extra = set(), set(), set()
    cases = write_extent_cases()
    descs = [write_extent_descs(prng, t, c) for t, c in enumerate(cases)]
    for t, (N, rnd, pol, r, k, b, kind, x_is_a) in enumerate(cases[:MAX_CASES or None]):
        A, B, Cd, x = descs[t]
        name = (f"write_extent[{t}] N={N} {rnd} A={pol} r={r} k={k} b={b} {kind} X as "
                f"{'AB'[x]}")
        tA, tB, tC = (hz.make_tensor(d.extent, d.stride, d.label) for d in (A, B, Cd))
        st, lw = lib.lowering(tA, tB, tC)
        if st != "SUCCESS":
            raise Fail(f"{name}: lowering {st}")
        # pack_pivot == the library's on the plan's view (the same C++ function)
        for y in (0, 1):
            fm = [(f[0], f[1 + y]) for f in lw["fused"]["IJ"[y]] + lw["fused"]["L"]]
            h = lw["fused"]["K"][0][1 + y] if lw["fused"]["K"] else 0
            if pack_pivot(fm, h) != lw["pivot"][y]:
                raise Fail(f"{name}: pack_pivot {pack_pivot(fm, h)} != the library's "
                           f"{lw['pivot'][y]} (operand {y})")
        roles.add(int(lw["swapped"]) ^ x)       # 1: X in the post-swap B role
        ops = [Operand(cuda, d, rng) for d in (A, B, Cd)]
        try:
            with tc.k_align(lib, pol):
                od = o.pack_desc_for(A, B, Cd, o.EMULATE, x, N, rnd)
                desc = hz.make_pack_desc(od)
                st, h = lib.pack_create(desc)
                if st != "SUCCESS":
                    raise Fail(f"{name}: imd_pack_create {st}")
                pb, pw = lib.pack_bytes(h)
                pv = o.pack_lower(od)
                if (pb, pw) != (pv.pack_bytes, pv.pack_ws):
                    raise Fail(f"{name}: pack sizes {pb}, {pw}; oracle {pv.pack_bytes}, "
                               f"{pv.pack_ws}")
                # the library's pack key == the oracle's: the key holds X's fused (extent,
                # stride) lists, the input of write_extent_shape and of the library's pivot
                lay = imd.host._PackLayout()
                lay.struct_size = C.sizeof(lay)
                st = lib.lib.imd_pack_layout(C.byref(desc), C.byref(lay))
                if st != 0 or lay.key_digest != pv.digest:
                    raise Fail(f"{name}: imd_pack_layout status {st}, key digest "
                               f"{lay.key_digest:#x}; oracle {pv.digest:#x}")
                shapes.add(write_extent_shape(pv, N))
                if len(pv.fused["K"]) > 1:
                    extra.add("gathered K")
                if sum(len(pv.fused[c]) for c in "RKL") > 8:
                    extra.add("kp_prep kK0MaxModes")
                pX = ops[x].arr.ptr
                try:
                    if not EXACT:
                        n_oracle += _sentinel_packs(lib, cuda, h, pX, pb, pw, od, ops[x], name)
                    n_pack += 1
                    # every K4 configuration on every 12th case (the rule's on the others)
                    n_contract += _exact_contracts(lib, cuda, h, pX, pb, pw, od, A, B, Cd, N,
                                                   rnd, ops, x, name,
                                                   (-1, 0, 1, 2, 3) if t % 12 == 0 else (-1,))
                finally:
                    lib.pack_destroy(h)
        finally:
            for y in ops:
                y.free()
        if verbose and (t % 10 == 0 or t == len(cases) - 1):
            print(f"  ok  {name} (pack_bytes {pb})", flush=True)
    if not MAX_CASES:       # the cases must cover every pack launch shape (must-fix 1)
        missing = WRITE_EXTENT_SHAPES - shapes
        if missing or roles != {0, 1} or len(extra) != 2:
            raise Fail(f"write_extent coverage: KL launch shapes missing {sorted(missing)}; "
                       f"post-swap roles {sorted(roles)}; {sorted(extra)}")
    how = ("exact allocations, no sentinel / oracle" if EXACT else
           f"two sentinels each, {n_oracle} == oracle.pack")
    print(f"  {n_pack} packs ({how}), "
          f"{n_contract} packed contracts on exactly sized packs == the plain call; "
          f"{len(shapes)} KL launch shapes (along, tile rows, values, chunks, W, radix > 1), "
          f"{sum(s[5] for s in shapes)} with radix > 1; post-swap roles "
          f"{''.join('AB'[q] for q in sorted(roles))}; {', '.join(sorted(extra))}", flush=True)


def _sentinel_packs(lib, cuda, h, pX, pb, pw, od, X, name):
    """imd_pack into [GUARD, GUARD + pb) of a sentinel-filled buffer, twice (two sentinels):
    the outside keeps the sentinel, the inside is identical both times (every byte written)
    and == oracle.pack when the operand is small; the same for its workspace."""
    dP, dW = cuda.malloc(pb + 2 * GUARD), cuda.malloc(pw + 2 * GUARD)
    try:
        inside = []
        for sent in (0xA5, 0x5A):
            cuda.memset(dP, sent, pb + 2 * GUARD)
            cuda.memset(dW, sent, pw + 2 * GUARD)
            st = lib.pack(h, pX, dP.value + GUARD, pb, dW.value + GUARD if pw else None, pw)
            if st != "SUCCESS":
                raise Fail(f"{name}: imd_pack {st}")
            cuda.sync()
            got = cuda.bytes_at(dP.value, pb + 2 * GUARD)
            wsg = cuda.bytes_at(dW.value, pw + 2 * GUARD)
            if (got[:GUARD] != sent).any() or (got[GUARD + pb:] != sent).any():
                bad = np.nonzero(np.r_[got[:GUARD], got[GUARD + pb:]] != sent)[0]
                raise Fail(f"{name}: imd_pack wrote {len(bad)} bytes outside [0, pack_bytes = "
                           f"{pb}) (sentinel {sent:#x})")
            if (wsg[:GUARD] != sent).any() or (wsg[GUARD + pw:] != sent).any():
                raise Fail(f"{name}: imd_pack wrote outside its workspace")
            inside.append(got[GUARD:GUARD + pb].copy())
        if not np.array_equal(inside[0], inside[1]):
            bad = np.nonzero(inside[0] != inside[1])[0]
            raise Fail(f"{name}: {len(bad)} bytes of [0, pack_bytes) differ between sentinels "
                       f"(unwritten), first at {int(bad[0])}")
        STATS["bytes"] += pb
        if int(np.prod(od.tensor.extent)) > 20000:
            return 0
        want = np.frombuffer(bytes(o.pack(X.tensor(), od)), np.uint8)
        if not np.array_equal(inside[0], want):
            raise Fail(f"{name}: packed bytes != oracle.pack")
        return 1
    finally:
        cuda.free(dP)
        cuda.free(dW)


def _exact_contracts(lib, cuda, h, pX, pb, pw, od, A, B, Cd, N, rnd, ops, x, name, k4s=(-1,)):
    """The pack in an EXACTLY sized allocation (IMD_TEST_PACK_SHORT: that many bytes short),
    read by imd_contract_packed with the in-line partner on every producer path, under each
    K4 configuration of k4s (-1: the rule); each C == the plain imd_contract's (the rule's),
    bit for bit. Returns the packed contracts run."""
    tA, tB, tC = (hz.make_tensor(d.extent, d.stride, d.label) for d in (A, B, Cd))
    dP = cuda.malloc(pb - SHORT)
    dW = cuda.malloc(pw) if pw else None
    n = 0
    try:
        st = lib.pack(h, pX, dP, pb, dW, pw)
        if st != "SUCCESS":
            raise Fail(f"{name}: imd_pack (exact) {st}")
        _, nws_plain = lib.workspace_size(tA, tB, tC, N, 0, tc.RND[rnd])
        pk_arg, keep = tp.packing_arg((od, None) if x == 0 else (None, od))
        wsb = C.c_size_t(0)
        st = lib.lib.imd_workspace_size_packed(C.byref(tA), C.byref(tB), C.byref(tC),
                                               C.byref(hz.Options(N, 0, tc.RND[rnd])), pk_arg,
                                               C.byref(wsb))
        if hz.STATUS[st] != "SUCCESS" or wsb.value > nws_plain:
            raise Fail(f"{name}: imd_workspace_size_packed {hz.STATUS[st]} {wsb.value} (I3: "
                       f"<= {nws_plain})")
        del keep
        ref = None
        # None: the plain imd_contract (the reference); then each K4 configuration x path
        for fmode, k4 in [(None, -1)] + [(f, c) for c in k4s for f in (-1, 0, 2, 3)]:
            lib.set_fused(-1 if fmode is None else fmode)
            lib.set_k4_config(k4)
            try:
                if fmode is None:
                    st, plan = lib.plan_create(tA, tB, tC, N, 0, tc.RND[rnd])
                    nws = nws_plain
                else:
                    st, plan = lib.plan_create_packed(tA, tB, tC, N, h if x == 0 else None,
                                                      h if x == 1 else None, 0, tc.RND[rnd])
                    nws = wsb.value
            finally:
                lib.set_fused(-1)
                lib.set_k4_config(-1)
            if st != "SUCCESS":
                raise Fail(f"{name}: plan (fused {fmode}, K4 {k4}) {st}")
            dWs = cuda.malloc(nws) if nws else None
            try:
                ops[2].reset()
                lib.set_k5_mapping(-1 if fmode is None else fmode)
                try:
                    if fmode is None:
                        st = lib.contract(plan, 1.0, ops[0].arr.ptr, ops[1].arr.ptr, 0.5,
                                          ops[2].arr.ptr, dWs, nws)
                    else:
                        opX = hz.Operand(pX, dP.value, pb, h)
                        opP = hz.Operand(ops[1 - x].arr.ptr, None, 0, None)
                        st = lib.contract_packed(plan, 1.0, opX if x == 0 else opP,
                                                 opP if x == 0 else opX, 0.5, ops[2].arr.ptr,
                                                 dWs, nws)
                finally:
                    lib.reset_k5_mapping()
                if st != "SUCCESS":
                    raise Fail(f"{name}: contract (fused {fmode}, K4 {k4}) {st}")
                got = ops[2].result()
            finally:
                if dWs is not None:
                    cuda.free(dWs)
                lib.plan_destroy(plan)
            if fmode is None:
                # the reference is not vacuous: it wrote C, and (small cases) == the oracle
                if not len(tc.bit_mismatches(got, ops[2].buf)):
                    raise Fail(f"{name}: the plain imd_contract left C unchanged")
                if not EXACT and int(np.prod(Cd.extent)) * o.pack_lower(od).k <= 200000:
                    same_bits(got, oracle(ops, 1.0, 0.5, N, rnd), f"{name} plain imd_contract")
                ref = got
            else:
                bad = tc.bit_mismatches(got, ref)
                if len(bad):
                    raise Fail(f"{name}: packed contract (fused {fmode}, K4 {k4}) != the plain "
                               f"call in {len(bad)} elements")
                n += 1
    finally:
        cuda.free(dP)
        if dW is not None:
            cuda.free(dW)
    return n


SUITES = {"repack_n": suite_repack_n, "workspace_shared_n": suite_workspace_shared_n,
          "graph_phases": suite_graph_phases, "write_extent": suite_write_extent}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--lib", default=hz.default_lib())
    ap.add_argument("--shim", help="tests/alloc_shim.c, already in LD_PRELOAD")
    ap.add_argument("--suite", action="append", choices=sorted(SUITES), required=True)
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()
    if "graph_phases" in args.suite[1:]:
        ap.error("graph_phases must be the first suite of its process: it is a cold capture "
                 "only while nothing at N = 10 or 12 has run")
    lib, cuda = hz.Library(args.lib), Cuda()
    imd.load_library(args.lib)               # the device API on the same library
    shim = Shim(args.shim)
    if args.shim:
        lib.load_alloc_shim(args.shim)
    t0 = time.time()
    try:
        for s in args.suite:
            t1 = time.time()
            SUITES[s](lib, cuda, shim, args.verbose)
            print(f"PASS  {s:18s} {time.time() - t1:6.1f}s", flush=True)
    except Fail as e:
        print(f"FAIL  {e}")
        return 1
    print(f"{STATS['calls']} bit-exact calls, {STATS['packs']} packs byte-exact, "
          f"{STATS['bytes']} sentinel-checked packed bytes in {time.time() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
