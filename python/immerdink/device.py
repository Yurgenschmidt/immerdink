"""immerdink contractions on caller-owned CUDA arrays (DESIGN §6c).

A thin ctypes binding of the C ABI (include/immerdink/immerdink.h, DESIGN §6 / §6b): every
size, status and result bit comes from libimmerdink. Arrays: anything with
__cuda_array_interface__ or DLPack (CuPy, PyTorch, Numba, JAX), or DeviceArray(ptr, shape,
strides); float64 only. immerdink never copies host <-> device and never allocates: C and
the workspace are the caller's.

    import immerdink as imd
    plan = imd.Plan("kjbc,kiac->abij", A, B, C, N=14)    # imd_plan_create
    ws = cupy.empty(plan.workspace_bytes, cupy.uint8)     # exact bytes, the caller's memory
    plan.contract(A, B, C, alpha=1.0, beta=0.0, workspace=ws, stream=cupy.cuda.get_current_stream())
    imd.contract("ik,kj->ij", A, B, out=C, N=14, workspace=ws)   # einsum front end, cached plans
"""

import math
import threading
import weakref
from collections import namedtuple

import ctypes as ct

from . import _arrays, _capi, host
from ._arrays import DeviceArray, stream_handle

__all__ = ["Handle", "Plan", "Pack", "PlanCache", "Layout", "DeviceArray", "contract",
           "workspace_size", "pack_spec", "default_handle", "default_plan_cache",
           "load_library", "ImmerdinkError", "InvalidValueError", "NotSupportedError",
           "AliasingError", "InsufficientWorkspaceError", "MisalignedError", "CudaError",
           "AllocFailedError", "PackMismatchError"]

ImmerdinkError = host.ImmerdinkError


class InvalidValueError(ImmerdinkError, ValueError):
    pass


class NotSupportedError(ImmerdinkError):
    pass


class AliasingError(ImmerdinkError, ValueError):
    pass


class InsufficientWorkspaceError(ImmerdinkError, ValueError):
    pass


class MisalignedError(ImmerdinkError, ValueError):
    pass


class CudaError(ImmerdinkError):
    pass


class AllocFailedError(ImmerdinkError, MemoryError):
    pass


class PackMismatchError(ImmerdinkError, ValueError):
    """.reason: the imd_lowering_packed pack_mismatch name at plan creation, else None."""
    reason = None


_EXC = {1: InvalidValueError, 2: NotSupportedError, 3: AliasingError,
        4: InsufficientWorkspaceError, 5: MisalignedError, 6: CudaError, 7: AllocFailedError,
        8: PackMismatchError}


def _error(st, what, extra=""):
    e = _EXC.get(st, ImmerdinkError)(st, what)
    if extra:
        e.args = (e.args[0] + extra,)
    return e


_ROUNDING = {"RN": 0, "TRUNC": 1}
_MODE = {"EMULATE": 0, "DIRECT": 1}
_VEC = {"GENERATE": 0, "RESIDENT": 1}
_ORDER = {"AUTO": 0, "GIVEN": 1}


def load_library(path=None):
    """libimmerdink (DESIGN §6c.1 search order), loaded once per path."""
    return path if isinstance(path, _capi.Library) else _capi.load(path)


# ------------------------------------------------------------------------------------------
# Operand layouts

class Layout(namedtuple("Layout", "shape strides")):
    """An operand's extents and strides in ELEMENTS (None: dense, last mode fastest), to plan
    before an array exists."""

    def __new__(cls, shape, strides=None):
        shape = tuple(int(e) for e in shape)
        strides = host.dense_strides(shape) if strides is None else [int(s) for s in strides]
        if len(strides) != len(shape):
            raise ValueError(f"{len(strides)} strides for {len(shape)} modes")
        return super().__new__(cls, shape, tuple(strides))


def _elem_strides(shape, bstrides, what):
    if bstrides is None:
        return tuple(host.dense_strides(shape))
    out = []
    for s in bstrides:
        if s % 8:
            raise TypeError(f"{what}: byte stride {s} is not a multiple of 8 (float64 elements)")
        out.append(s // 8)
    return tuple(out)


def _view_layout(v, what):
    _, shape, bstrides, f8, _, _ = v
    if not f8:
        raise TypeError(f"{what}: dtype must be float64")
    shape = tuple(int(e) for e in shape)
    return Layout(shape, _elem_strides(shape, bstrides, what))


def _layout(x, what):
    """Layout of an operand: a Layout or a float64 CUDA array."""
    return x if isinstance(x, Layout) else _view_layout(_arrays.view(x), what)


_SPECS = {}


def _parse(spec):
    """'A,B->C' -> the three label strings (single characters, explicit output)."""
    r = _SPECS.get(spec)
    if r is None:
        s = spec.replace(" ", "")
        lhs, sep, out = s.partition("->")
        ins = lhs.split(",")
        if not sep or len(ins) != 2 or "." in s:
            raise ValueError(f"expected an einsum spec 'A,B->C' (explicit output, no ellipsis), "
                             f"got {spec!r}")
        r = _SPECS[spec] = (ins[0], ins[1], out)
    return r


def _descs(spec, layouts):
    labels = _parse(spec)
    out = []
    for name, lab, lay in zip("ABC", labels, layouts):
        if len(lab) != len(lay.shape):
            raise ValueError(f"{name}: spec {lab!r} has {len(lab)} labels, the array has rank "
                             f"{len(lay.shape)}")
        out.append(host.Desc(lay.shape, lay.strides, tuple(ord(ch) for ch in lab)))
    return tuple(out)


def _as_desc(d):
    return d if isinstance(d, host.Desc) else host.Desc(*map(tuple, d))


class _CTensor:
    """An imd_tensor_t and its arrays."""
    __slots__ = ("t", "arrs")

    def __init__(self, d):
        r = len(d.extent)
        if not (len(d.stride) == len(d.label) == r):
            raise ValueError("extent, stride and label lengths differ")
        self.arrs = ((ct.c_int64 * r)(*d.extent), (ct.c_int64 * r)(*d.stride),
                     (ct.c_int32 * r)(*d.label))
        self.t = _capi.Tensor(r, *self.arrs)


class _CPackDesc:
    """An imd_pack_desc_t of a host.PackSpec and the arrays it points to."""

    def __init__(self, p):
        self.tensor = _CTensor(_as_desc(p.tensor))
        lists = (tuple(p.contracted), tuple(p.batch), tuple(p.rows or ()))
        self.lists = [(ct.c_int32 * max(len(x), 1))(*x) for x in lists]
        self.d = _capi.PackDesc(ct.sizeof(_capi.PackDesc), int(p.num_moduli),
                                _opt(_ROUNDING, p.rounding, "rounding"), _opt(_ORDER, p.order, "order"),
                                ct.pointer(self.tensor.t),
                                len(lists[0]), self.lists[0] if lists[0] else None,
                                len(lists[1]), self.lists[1] if lists[1] else None,
                                len(lists[2]), self.lists[2] if lists[2] else None)


def _opt(table, value, what):
    try:
        return table[value]
    except KeyError:
        raise ValueError(f"{what} must be one of {sorted(table)}, got {value!r}") from None


def _options(N, g, rounding, mode, vec_mode, k4_tune):
    return _capi.Options(int(N), int(g), _opt(_ROUNDING, rounding, "rounding"),
                         _opt(_MODE, mode, "mode"), _opt(_VEC, vec_mode, "vec_mode"),
                         1 if k4_tune else 0)


class _Expect:
    """What one operand of a plan (or a pack's X) must look like: the shape and the byte
    strides, checked fast (exact tuples) and, on a miss, on the modes that matter."""
    __slots__ = ("shape", "bstrides", "dense_ok", "what")

    def __init__(self, desc, what):
        self.shape = tuple(desc.extent)
        self.bstrides = tuple(8 * s for s in desc.stride)
        self.what = what
        self.dense_ok = self._match(tuple(8 * s for s in host.dense_strides(self.shape)))

    def _match(self, bstrides):
        if 0 in self.shape:
            return True
        return all(e <= 1 or s == w for e, s, w in zip(self.shape, bstrides, self.bstrides))

    def check(self, shape, bstrides):
        if shape != self.shape:
            if tuple(int(e) for e in shape) != self.shape:
                raise ValueError(f"{self.what}: shape {tuple(shape)} != the plan's {self.shape}")
        if bstrides is None:
            ok = self.dense_ok
        else:
            ok = bstrides == self.bstrides or self._match(tuple(bstrides))
        if not ok:
            got = "dense" if bstrides is None else tuple(s / 8 for s in bstrides)
            raise ValueError(f"{self.what}: strides {got} (elements) != the plan's "
                             f"{tuple(s // 8 for s in self.bstrides)}")


def _ptr(x, expect, stream, out=False):
    ptr, shape, bstrides, f8, readonly, _ = _arrays.view(x, stream)
    if not f8:
        raise TypeError(f"{expect.what}: dtype must be float64")
    if out and readonly:
        raise ValueError(f"{expect.what} is read-only")
    expect.check(shape, bstrides)
    return ptr


# ------------------------------------------------------------------------------------------
# Handle

class Handle:
    """imd_create on the current device. Plans and packs keep their handle alive."""

    def __init__(self, lib=None):
        self.lib = load_library(lib)
        self.device = self.lib.current_device()
        h = ct.c_void_p()
        st = self.lib.lib.imd_create(ct.byref(h))
        if st:
            raise _error(st, "imd_create")
        self._h = h.value
        self._fin = weakref.finalize(self, self.lib.lib.imd_destroy, h.value)

    def close(self):
        """imd_destroy (idempotent); plans and packs already created stay valid."""
        self._h = None
        self._fin()


_handles = {}
_handles_lock = threading.Lock()


def default_handle(lib=None):
    """The process-wide handle of the current device (created on first use)."""
    lib = load_library(lib)
    key = (id(lib), lib.current_device())
    h = _handles.get(key)
    if h is None:
        with _handles_lock:
            h = _handles.get(key)
            if h is None:
                h = _handles[key] = Handle(lib)
    return h


# ------------------------------------------------------------------------------------------
# Packs (DESIGN §6b)

def pack_spec(spec, A, B, C, *, operand, N, g=0, rounding="RN", mode="EMULATE",
              vec_mode="GENERATE", lib=None):
    """imd_pack_desc_for: the GIVEN host.PackSpec of operand 0 (A) or 1 (B) with the classes and
    orders the unpacked plan uses (a pack of it reproduces that contraction bit for bit).
    A, B, C: arrays or Layouts."""
    descs = _descs(spec, [_layout(x, n) for x, n in zip((A, B, C), "ABC")])
    return host.pack_desc_for(descs, operand=operand, N=N, g=g, rounding=rounding, mode=mode,
                              vec_mode=vec_mode, lib=load_library(lib).host)


class Pack:
    """imd_pack_create of a host.PackSpec (an immutable host object: the key and the layout).
    .pack(X, out) writes the packed operand into caller-owned device memory of pack_bytes."""

    def __init__(self, spec, *, handle=None, lib=None):
        self.lib = load_library(lib)
        self.handle = handle or default_handle(self.lib)
        self.spec = spec
        self._cd = _CPackDesc(spec)
        h = ct.c_void_p()
        st = self.lib.lib.imd_pack_create(self.handle._h, ct.byref(self._cd.d), ct.byref(h))
        if st:
            raise _error(st, "imd_pack_create")
        self._h = h.value
        self._fin = weakref.finalize(self, self.lib.lib.imd_pack_destroy, h.value)
        pb, wb = ct.c_size_t(0), ct.c_size_t(0)
        st = self.lib.lib.imd_pack_bytes(self._h, ct.byref(pb), ct.byref(wb))
        if st:
            raise _error(st, "imd_pack_bytes")
        self.pack_bytes, self.workspace_bytes = pb.value, wb.value
        self._x = _Expect(_as_desc(spec.tensor), "X")
        self._fn = self.lib.lib.imd_pack
        self._layout = None

    @property
    def layout(self):
        """host.PackLayout (imd_pack_layout): the packed buffer's regions and sizes."""
        if self._layout is None:
            self._layout = host.pack_layout(self.spec, lib=self.lib.host)
        return self._layout

    def pack(self, X, out, *, workspace=None, stream=None):
        """imd_pack: X (float64, the descriptor's layout) -> `out` (>= pack_bytes, 256-byte
        aligned) on `stream`; `workspace` >= workspace_bytes. Returns out."""
        s = stream_handle(stream)
        h = self._h
        if h is None:
            raise ValueError("pack is closed")
        px = _ptr(X, self._x, s)
        po, no = _arrays.buffer(out, "packed buffer", stream=s)
        pw, nw = _arrays.buffer(workspace, "workspace", stream=s)
        st = self._fn(h, px, po, no, pw, nw, s)
        if st:
            raise _error(st, "imd_pack", f" (pack {self.pack_bytes} bytes, workspace "
                                         f"{self.workspace_bytes} bytes needed; given {no}, {nw})")
        return out

    def pack_ptr(self, X, packed, packed_bytes, ws, ws_bytes, stream=0):
        """imd_pack with raw integers (no Python-side layout check)."""
        st = self._fn(self._h, X, packed, packed_bytes, ws, ws_bytes, stream)
        if st:
            raise _error(st, "imd_pack")

    def close(self):
        """imd_pack_destroy (idempotent). Allowed once every call passed it has returned;
        plans made with it keep it alive."""
        self._h = None
        self._fin()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def __repr__(self):
        return f"<immerdink.Pack {self.pack_bytes} bytes, workspace {self.workspace_bytes}>"


def _pack_specs(packs):
    """(PackSpec or None, PackSpec or None) of packs given as Pack / PackSpec / None."""
    if packs is None:
        return None
    if len(packs) != 2:
        raise ValueError("packs = (pack of A or None, pack of B or None)")
    out = tuple(p.spec if isinstance(p, Pack) else p for p in packs)
    return None if out == (None, None) else out


# ------------------------------------------------------------------------------------------
# Plans

class Plan:
    """imd_plan_create (or imd_plan_create_packed with `packs`) for fixed operand layouts.

    A, B, C: arrays or Layouts; options as host.py (N required). Exact footprints:
    workspace_bytes (per in-flight call, caller-owned), device_bytes (owned by the plan),
    host_bytes; lowering = the host library's summary. Immutable: any number of threads may
    call contract() at once, each with its own workspace (DESIGN §6). close() / with / the
    finalizer destroy it; a RESIDENT plan must outlive the GPU work of its calls."""

    def __init__(self, spec, A, B, C, *, N, g=0, rounding="RN", mode="EMULATE",
                 vec_mode="GENERATE", k4_tune=False, packs=None, handle=None, lib=None):
        layouts = [_layout(x, n) for x, n in zip((A, B, C), "ABC")]
        self.spec = spec
        self._create(_descs(spec, layouts), N, g, rounding, mode, vec_mode, k4_tune, packs,
                     handle, lib)

    @classmethod
    def from_descs(cls, A, B, C, **options):
        """A plan from raw descriptors: host.Desc or (extent, stride, label) with integer
        labels and element strides."""
        self = cls.__new__(cls)
        self.spec = None
        kw = dict(g=0, rounding="RN", mode="EMULATE", vec_mode="GENERATE", k4_tune=False,
                  packs=None, handle=None, lib=None)
        unknown = set(options) - set(kw) - {"N"}
        if unknown:
            raise TypeError(f"unknown options {sorted(unknown)}")
        kw.update(options)
        N = kw.pop("N")
        self._create(tuple(_as_desc(d) for d in (A, B, C)), N, **kw)
        return self

    def _create(self, descs, N, g, rounding, mode, vec_mode, k4_tune, packs, handle, lib):
        self._h = None
        self.lib = lib = load_library(lib)
        self.handle = handle or default_handle(lib)
        self.descs = descs
        self.options = dict(N=int(N), g=int(g), rounding=rounding, mode=mode, vec_mode=vec_mode,
                            k4_tune=bool(k4_tune))
        self.packs = (None, None) if packs is None else tuple(packs)
        if len(self.packs) != 2 or any(p is not None and not isinstance(p, Pack) for p in self.packs):
            raise TypeError("packs = (Pack or None, Pack or None)")
        self._packed = self.packs != (None, None)
        if any(p is not None and p._h is None for p in self.packs):
            raise ValueError("a pack in `packs` is closed")
        tens = [_CTensor(d) for d in descs]
        opts = _options(N, g, rounding, mode, vec_mode, k4_tune)
        L = lib.lib
        plan = ct.c_void_p()
        if self._packed:
            hp = [p._h if p is not None else None for p in self.packs]
            st = L.imd_plan_create_packed(self.handle._h, *(ct.byref(t.t) for t in tens),
                                          ct.byref(opts), hp[0], hp[1], ct.byref(plan))
        else:
            st = L.imd_plan_create(self.handle._h, *(ct.byref(t.t) for t in tens), ct.byref(opts),
                                   ct.byref(plan))
        if st:
            e = _error(st, "imd_plan_create" + ("_packed" if self._packed else ""))
            if st == 8:
                e.reason = self._lowering().pack_mismatch
                e.args = (e.args[0] + f" ({e.reason})",)
            raise e
        self._h = plan.value
        self._fin = weakref.finalize(self, L.imd_plan_destroy, plan.value)
        # exact sizes from the C library (never recomputed here)
        ws = ct.c_size_t(0)
        if self._packed:
            cps = [_CPackDesc(p.spec) if p is not None else None for p in self.packs]
            packing = _capi.Packing(*[ct.pointer(c.d) if c is not None else None for c in cps])
            st = L.imd_workspace_size_packed(*(ct.byref(t.t) for t in tens), ct.byref(opts),
                                             ct.byref(packing), ct.byref(ws))
        else:
            st = L.imd_workspace_size(*(ct.byref(t.t) for t in tens), ct.byref(opts), ct.byref(ws))
        if st:
            raise _error(st, "imd_workspace_size")
        self.workspace_bytes = ws.value
        dev = ct.c_size_t(0)
        st = L.imd_plan_device_bytes(self._h, ct.byref(dev))
        if st:
            raise _error(st, "imd_plan_device_bytes")
        self.device_bytes = dev.value
        self._lw = None
        self._ex = tuple(_Expect(d, n) for d, n in zip(descs, "ABC"))
        self._fn = L.imd_contract_packed if self._packed else L.imd_contract

    # -- queries ----------------------------------------------------------------------------

    def _lowering(self, sm_count=0):
        o = self.options
        return host.lowering(self.descs, N=o["N"], g=o["g"], rounding=o["rounding"],
                             mode=o["mode"], vec_mode=o["vec_mode"], sm_count=sm_count,
                             packing=_pack_specs(self.packs), lib=self.lib.host)

    @property
    def lowering(self):
        """host.Lowering of this plan (imd_lowering[_packed]) for the current device's SMs."""
        if self._lw is None:
            self._lw = self._lowering(self.lib.sm_count())
        return self._lw

    @property
    def host_bytes(self):
        """The plan's host payload (imd_plan_bytes / imd_lowering_packed)."""
        return self.lowering.plan_host_bytes

    def pack_spec(self, operand):
        """imd_pack_desc_for of operand 0 (A) or 1 (B) for this plan's descriptors and options."""
        o = self.options
        return host.pack_desc_for(self.descs, operand=operand, N=o["N"], g=o["g"],
                                  rounding=o["rounding"], mode=o["mode"], vec_mode=o["vec_mode"],
                                  lib=self.lib.host)

    # -- calls ------------------------------------------------------------------------------

    def contract(self, A, B, C, *, alpha=1.0, beta=0.0, workspace=None, stream=None,
                 packed=None):
        """C <- alpha A.B + beta C on `stream` (imd_contract / imd_contract_packed). A, B, C:
        float64 CUDA arrays of the plan's shapes and strides; workspace: >= workspace_bytes
        (256-byte aligned), at most one in-flight call per workspace; packed: (buffer or
        None, buffer or None), the packed buffers of a packed plan's slots. No allocation,
        no synchronization: capture-safe. Returns C."""
        h = self._h
        if h is None:
            raise ValueError("plan is closed")
        s = 0 if stream is None else stream_handle(stream)
        ex = self._ex
        pa = _ptr(A, ex[0], s)
        pb = _ptr(B, ex[1], s)
        pc = _ptr(C, ex[2], s, out=True)
        if workspace is None:
            pw, nw = None, 0
        else:
            pw, nw = _arrays.buffer(workspace, "workspace", stream=s)
        if self._packed:
            st = self._fn(h, alpha, *self._operands(pa, pb, packed, s), beta, pc, pw, nw, s)
        else:
            if packed is not None and any(x is not None for x in packed):
                raise ValueError("packed buffers given to a plan without packs")
            st = self._fn(h, alpha, pa, pb, beta, pc, pw, nw, s)
        if st:
            raise self._fail(st, nw)
        return C

    def _operands(self, pa, pb, packed, s):
        packed = (None, None) if packed is None else packed
        if len(packed) != 2:
            raise ValueError("packed = (buffer or None, buffer or None)")
        ops = []
        for x, (data, buf, pk) in enumerate(zip((pa, pb), packed, self.packs)):
            if pk is None:
                if buf is not None:
                    raise ValueError(f"packed[{x}] given for an in-line operand")
                ops.append(_capi.Operand(data, None, 0, None))
            else:
                p, n = _arrays.buffer(buf, f"packed[{x}]", writable=False, stream=s)
                ops.append(_capi.Operand(data, p, n, pk._h))
        return ct.byref(ops[0]), ct.byref(ops[1])

    def contract_ptr(self, alpha, A, B, beta, C, ws, ws_bytes, stream=0, packed=None):
        """imd_contract with raw integers (pointers to element 0, the stream handle): no
        Python-side layout check, the C checks only. packed = ((ptr, nbytes) or None, ...)
        for a packed plan."""
        if self._packed:
            st = self._fn(self._h, alpha, *self._operands(A, B, packed, stream), beta, C, ws,
                          ws_bytes, stream)
        else:
            st = self._fn(self._h, alpha, A, B, beta, C, ws, ws_bytes, stream)
        if st:
            raise self._fail(st, ws_bytes)

    def _fail(self, st, nw):
        what = "imd_contract" + ("_packed" if self._packed else "")
        extra = ""
        if st in (1, 4) and nw < self.workspace_bytes:
            extra = f" (workspace of {nw} bytes; the plan needs {self.workspace_bytes})"
        return _error(st, what, extra)

    def close(self):
        """imd_plan_destroy (idempotent). GENERATE: once the calls returned; RESIDENT: once
        their GPU work completed (synchronize first). Never while another thread calls it."""
        self._h = None
        if hasattr(self, "_fin"):
            self._fin()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def __repr__(self):
        state = "closed" if self._h is None else f"workspace {self.workspace_bytes} bytes, " \
                                                  f"device {self.device_bytes} bytes"
        return f"<immerdink.Plan {self.spec or self.descs!r} N={self.options['N']} {state}>"


# ------------------------------------------------------------------------------------------
# Plan cache and the einsum front end

class PlanCache:
    """Plans keyed by (spec, shapes, strides, options, packs, device): a plan is a pure
    function of these, so a cached one is equivalent to a new one. Thread-safe; never evicts
    on its own (RESIDENT plans cannot be destroyed safely without a sync): clear() closes
    every plan (synchronize resident plans' streams first)."""

    def __init__(self):
        self._plans = {}
        self._lock = threading.Lock()

    def __len__(self):
        """The number of distinct plans held."""
        return len({id(p) for p in list(self._plans.values())})

    def _lookup(self, key, make):
        p = self._plans.get(key)
        if p is None:
            new = make()                       # outside the lock: plan creation can be slow
            with self._lock:
                p = self._plans.setdefault(key, new)
            if p is not new:
                new.close()                    # never used: safe in any vec_mode
        return p

    def _get(self, spec, lays, opts, packs, lib):
        N, g, rounding, mode, vec_mode, k4_tune = opts
        key = ("L", spec, lays, opts, packs, id(lib), lib.current_device())
        return self._lookup(key, lambda: Plan(spec, *lays, N=N, g=g, rounding=rounding, mode=mode,
                                              vec_mode=vec_mode, k4_tune=k4_tune, packs=packs,
                                              lib=lib))

    def get(self, spec, A, B, C, *, N, g=0, rounding="RN", mode="EMULATE",
            vec_mode="GENERATE", k4_tune=False, packs=None, lib=None):
        """The cached plan for these operands (arrays or Layouts) and options."""
        lays = tuple(_layout(x, n) for x, n in zip((A, B, C), "ABC"))
        opts = (int(N), int(g), rounding, mode, vec_mode, bool(k4_tune))
        return self._get(spec, lays, opts, None if packs is None else tuple(packs),
                         load_library(lib))

    def clear(self):
        """Close every plan (RESIDENT plans: after their streams were synchronized)."""
        with self._lock:
            plans, self._plans = list(self._plans.values()), {}
        for p in {id(p): p for p in plans}.values():
            p.close()


default_plan_cache = PlanCache()


def _tup(x):
    return x if x is None or type(x) is tuple else tuple(x)


def contract(spec, A, B, out, *, alpha=1.0, beta=0.0, N, g=0, rounding="RN", mode="EMULATE",
             vec_mode="GENERATE", k4_tune=False, workspace=None, stream=None, plan_cache=None,
             packs=None, packed=None, lib=None):
    """einsum-style C <- alpha sum A B + beta C ("ik,kj->ij"; DESIGN §6c.2) on caller-owned
    float64 CUDA arrays. The plan comes from `plan_cache` (default: default_plan_cache;
    False: a plan for this call only, GENERATE only). workspace: caller-owned, >=
    the plan's workspace_bytes (see workspace_size). Returns out."""
    s = 0 if stream is None else stream_handle(stream)
    lib = load_library(lib)
    va, vb, vc = (_arrays.view(x, s) for x in (A, B, out))
    for v, n in ((va, "A"), (vb, "B"), (vc, "C")):
        if not v[3]:
            raise TypeError(f"{n}: dtype must be float64")
    if vc[4]:
        raise ValueError("C is read-only")
    opts = (int(N), int(g), rounding, mode, vec_mode, bool(k4_tune))
    if plan_cache is False:
        if vec_mode != "GENERATE":
            raise ValueError("plan_cache=False needs vec_mode GENERATE (a RESIDENT plan must "
                             "outlive its GPU work)")
        lays = tuple(_view_layout(v, n) for v, n in ((va, "A"), (vb, "B"), (vc, "C")))
        with Plan(spec, *lays, N=N, g=g, rounding=rounding, mode=mode, vec_mode=vec_mode,
                  k4_tune=k4_tune, packs=packs, lib=lib) as plan:
            return plan.contract(A, B, out, alpha=alpha, beta=beta, workspace=workspace,
                                 stream=s, packed=packed)
    cache = default_plan_cache if plan_cache is None else plan_cache
    key = ("V", spec, _tup(va[1]), _tup(va[2]), _tup(vb[1]), _tup(vb[2]), _tup(vc[1]),
           _tup(vc[2]), opts, None if packs is None else tuple(packs), id(lib),
           lib.current_device())
    plan = cache._plans.get(key)
    if plan is None:   # the plan of these layouts (shared with PlanCache.get), then this key
        lays = tuple(_view_layout(v, n) for v, n in ((va, "A"), (vb, "B"), (vc, "C")))
        plan = cache._get(spec, lays, opts, key[9], lib)
        with cache._lock:
            plan = cache._plans.setdefault(key, plan)
    h = plan._h
    if h is None:
        raise ValueError("the cached plan was closed")
    # the key fixes the shapes and strides: no further layout check
    if workspace is None:
        pw, nw = None, 0
    else:
        pw, nw = _arrays.buffer(workspace, "workspace", stream=s)
    if plan._packed:
        st = plan._fn(h, alpha, *plan._operands(va[0], vb[0], packed, s), beta, vc[0], pw, nw, s)
    else:
        st = plan._fn(h, alpha, va[0], vb[0], beta, vc[0], pw, nw, s)
    if st:
        raise plan._fail(st, nw)
    return out


def workspace_size(spec, A, B, C, *, N, g=0, rounding="RN", mode="EMULATE",
                   vec_mode="GENERATE", packs=None, lib=None):
    """imd_workspace_size[_packed] (host library, no plan): the exact workspace bytes of a
    contraction of these operands (arrays or Layouts). packs: (Pack / PackSpec / None) x 2."""
    descs = _descs(spec, [_layout(x, n) for x, n in zip((A, B, C), "ABC")])
    return host.workspace_size(descs, N=N, g=g, rounding=rounding, mode=mode,
                               vec_mode=vec_mode, packing=_pack_specs(packs),
                               lib=load_library(lib).host)
