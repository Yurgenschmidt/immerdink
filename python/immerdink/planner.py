"""Fast planner queries for hot paths (DESIGN §6e): host.py's queries at about the cost of the
C call.

host.py builds every C struct from the spec on each call and decodes every result field into
dataclasses: 8-20 µs per query over a C cost of 0.3-0.8 µs. Here the descriptor structs are
built once per contraction (a Query), the options, request and calibration structs are
reused, a call allocates only its output struct, and results are LAZY views that read the C
struct on attribute access. The C library stays the single source of truth: every number is
the one the host.py function of the same name returns (bit for bit; tested), nothing is
recomputed here.

    from immerdink import planner
    q = planner.Query("kjbc,kiac->abij", dict(a=64, b=64, c=64, i=32, j=32, k=32))
    q.predict(14).time_s              # imd_perf_predict, ~2 µs; .to_prediction(): host.Prediction
    q.workspace_size(14, g=7)         # imd_workspace_size, ~1 µs
    q.rank([8, 14], budget=2**30)[0]  # imd_perf_rank (one C call), best candidate
    q.lowering(14).k4_config, q.plan_bytes(14), q.pack_desc_for(0, 14)

    pl = planner.Planner(memo=planner.Memo(4096))   # one Query per contraction, cached
    pl.predict(spec, ext, N=14)       # host.predict's signature; a repeated query is a memo hit

Objects:
  Query(spec, extents, strides=None) / Query(descs) / Query.from_layouts(spec, A, B, C):
      the C descriptors of one contraction; methods take the options (N, g, vec_mode, ...).
      Immutable descriptors; thread-safe (each call has its own output struct).
  Request: a prebuilt imd_perf_request_t (beta, alpha_zero, k4_config, eps_fraction, or any
      other field of the C struct by name), for predict(request=...).
  Prediction, Lowering, Candidate (in a Ranking): lazy, immutable views with the attribute
      names of host.Prediction / host.Lowering / host.Candidate; .to_prediction(),
      .to_lowering(), .to_candidate() (Ranking.to_list()) give exactly host.py's objects.
  Memo(maxsize, max_bytes): an EXPLICIT, thread-safe memo of results keyed by the full
      query, bounded by entries and by exact C struct bytes (FIFO); opt-in (Query(...,
      memo=) or Planner(memo=)). Only identical queries hit.
  Planner: Query objects cached by (spec, extents, strides) (bounded), host.py-style
      signatures, an optional Memo and calibration.

calib=None means the shipped calibration: the library's imd_perf_calib_default, loaded once
per library and passed explicitly (passing NULL makes the C library parse the shipped text on
every call, about 2 µs).
"""

import ctypes as C
import dataclasses
import numbers
import threading
from operator import attrgetter, index

from . import host
from .host import (DISPATCH, PACK_MISMATCH, STATUS, Calibration, HostLibrary, ImmerdinkError,
                   PackSpec)

__all__ = ["Query", "Planner", "Memo", "Request", "Prediction", "Lowering", "Candidate",
           "Ranking", "default_calibration"]

_ROUNDING, _MODE, _VEC = host._ROUNDING, host._MODE, host._VEC
_VEC_NAME = {v: k for k, v in _VEC.items()}
_PACKED_NAME = ["NONE", "A", "B", "AB"]
_PERF_SIZE = C.sizeof(host._Perf)
_LW_SIZE = C.sizeof(host._Lowering)
_PD_SIZE = C.sizeof(host._PackDesc)
_REQ_SIZE = C.sizeof(host._PerfRequest)
_RANK_SIZE = C.sizeof(host._RankRequest)
_CAND_SIZE = C.sizeof(host._Candidate)
_OPTS_MAX = 512     # option structs kept per Query (a finite set in practice; cleared if full)
_PACKS_MAX = 64     # packing structs kept per Query
_RANKARGS_MAX = 64  # (N list, g list) arrays kept per Query
_RANK_CAP_MAX = 4096  # first-call capacity bound of rank (a larger count: one exact retry)
_MISS = object()


def _lib(lib):
    return lib if isinstance(lib, HostLibrary) else host.load(lib)


def _fn(h, name):
    """An untyped function pointer of the library (its own object: host.py's declared ones
    are untouched). Called only with pointers (byref, ctypes arrays), None and int32 values,
    which ctypes passes without an argtypes conversion; the C signatures are host.py's."""
    f = h.lib[name]
    f.restype = C.c_int
    return f


class _Fns:
    """The untyped entry points of one HostLibrary (built once per library)."""

    def __init__(self, h):
        for name in ("imd_workspace_size", "imd_workspace_size_packed", "imd_plan_bytes",
                     "imd_lowering", "imd_lowering_packed", "imd_pack_desc_for",
                     "imd_perf_predict", "imd_perf_predict_pack", "imd_perf_rank"):
            setattr(self, name[4:], _fn(h, name))


_libstate = {}      # HostLibrary -> (_Fns, default Calibration); one entry per loaded library
_libstate_lock = threading.Lock()


def _state(h):
    s = _libstate.get(h)
    if s is None:
        with _libstate_lock:
            s = _libstate.get(h)
            if s is None:
                s = _libstate[h] = (_Fns(h), Calibration.default(h))
    return s


def default_calibration(lib=None):
    """The shipped calibration of the library (imd_perf_calib_default), loaded once and
    shared (treat it as read-only)."""
    return _state(_lib(lib))[1]


def _calib_ref(calib):
    if isinstance(calib, Calibration):
        return C.byref(calib._raw)
    raise TypeError("calib must be a Calibration or None")


def _opt_err(table, value, what):
    if value not in table:
        raise ValueError(f"{what} must be one of {sorted(table)}, got {value!r}")
    return table[value]


# ------------------------------------------------------------------------------------------
# Requests

_REQ_FIELDS = tuple(f for f, _ in host._PerfRequest._fields_ if f != "struct_size")
# imd_perf_rank_request_t fields rank() takes as **fields: those after eps_fraction (the
# last field of version 6; later versions only append)
_RANK_NAMES = [f for f, _ in host._RankRequest._fields_]
_RANK_EXTRA = frozenset(_RANK_NAMES[_RANK_NAMES.index("eps_fraction") + 1:])


class Request:
    """A prebuilt imd_perf_request_t (DESIGN §6c.3): beta (C is read), alpha_zero (the
    beta-only call), k4_config (None: the plan-time rule), eps_fraction (None: the
    typical-data default), and any other field of the C struct by its name (later host API
    versions). Immutable; equal requests compare and hash equal."""
    __slots__ = ("_c", "_ref", "_key")

    def __init__(self, *, beta=False, alpha_zero=False, k4_config=None, eps_fraction=None,
                 **fields):
        c = host._PerfRequest()
        c.struct_size = _REQ_SIZE
        c.flags = (host._PERF_BETA_NONZERO if beta else 0) | \
                  (host._PERF_ALPHA_ZERO if alpha_zero else 0)
        c.k4_config = -1 if k4_config is None else int(k4_config)
        c.eps_fraction = host._eps(eps_fraction)
        for k, v in fields.items():
            if k not in _REQ_FIELDS or k in ("flags", "k4_config", "eps_fraction"):
                raise TypeError(f"unknown request field {k!r} (fields: {_REQ_FIELDS})")
            setattr(c, k, v)
        self._c = c
        self._ref = C.byref(c)
        self._key = tuple(getattr(c, f) for f in _REQ_FIELDS)

    def __eq__(self, other):
        return isinstance(other, Request) and self._key == other._key

    def __hash__(self):
        return hash(self._key)

    def __repr__(self):
        return "Request(" + ", ".join(f"{f}={v!r}" for f, v in zip(_REQ_FIELDS, self._key)) + ")"


_DEFAULT_REQUEST = Request()


def _request(request, beta, alpha_zero, k4_config, eps_fraction):
    if request is not None:
        if beta or alpha_zero or k4_config is not None or eps_fraction is not None:
            raise TypeError("pass either request= or beta / alpha_zero / k4_config / "
                            "eps_fraction")
        return request
    if beta or alpha_zero or k4_config is not None or eps_fraction is not None:
        return Request(beta=beta, alpha_zero=alpha_zero, k4_config=k4_config,
                       eps_fraction=eps_fraction)
    return _DEFAULT_REQUEST


# ------------------------------------------------------------------------------------------
# Lazy result views

class _View:
    """A lazy view of one C result struct: _r the struct (owned, never written after the
    call), _st the returned status, _cache a decoded part."""
    __slots__ = ("_r", "_st", "_cache")
    _fields = ()

    def __init__(self, r, st=0):
        self._r = r
        self._st = st
        self._cache = None

    def __repr__(self):
        return (f"<{type(self).__name__} "
                + " ".join(f"{f}={getattr(self, f)!r}" for f in self._fields[:8]) + " ...>")


def _view_class(name, doc, dc, struct, converters, full):
    """A _View subclass with one read-only property per field of the host.py dataclass `dc`:
    converters[field] if given, else the C struct's field of that name (a C array field as
    a tuple), else (a field of a
    later host.py the view does not know) the fully decoded object's (`full`, a host.py
    decoder), or an error naming the field when the view has no such decoder."""
    names = {f for f, _ in struct._fields_}
    arrays = {f for f, t in struct._fields_ if isinstance(t, type(C.Array))}
    ns = {"__slots__": (), "__doc__": doc, "_fields": tuple(f.name for f in dataclasses.fields(dc)),
          "_full": full}

    def unknown(self, f):
        raise AttributeError(f"{name}.{f}: a host.py field that is not in the C struct; "
                             "planner.py needs a converter for it")
    for f in ns["_fields"]:
        if f in converters:
            ns[f] = property(converters[f])
        elif f in arrays:   # a C array field: an immutable tuple, as host.py's dataclasses
            ns[f] = property(lambda self, get=attrgetter("_r." + f): tuple(get(self)))
        elif f in names:
            ns[f] = property(attrgetter("_r." + f))
        elif full is not None:
            ns[f] = property(lambda self, f=f: getattr(self._full(), f))
        else:
            ns[f] = property(lambda self, f=f: unknown(self, f))
    return type(name, (_View,), ns)


def _phases(self):
    p = self._cache
    if p is None:
        p = self._cache = self._full().phases
    return p


def _to_prediction(self):
    """Exactly host.predict's host.Prediction (host._prediction of the same struct)."""
    return host._prediction(self._r, self._st)


Prediction = _view_class(
    "Prediction", "imd_perf_t as a lazy host.Prediction: attributes read the C struct; "
    ".phases / .phase(name) decode the phases once; .to_prediction() is host.predict's "
    "object.", host.Prediction, host._Perf,
    {"status": lambda s: STATUS[s._st], "dispatch": lambda s: DISPATCH[s._r.dispatch],
     "phases": _phases}, _to_prediction)
Prediction.to_prediction = _to_prediction


def _phase(self, name):
    """The host.PhasePrediction with this name, or None."""
    return next((p for p in self.phases if p.name == name), None)


Prediction.phase = _phase


def _to_lowering(self):
    """Exactly host.lowering's host.Lowering."""
    return host.Lowering(**{f: getattr(self, f) for f in self._fields})


Lowering = _view_class(
    "Lowering", "imd_lowering_t as a lazy host.Lowering (.to_lowering(): host.lowering's "
    "object).", host.Lowering, host._Lowering,
    {"status": lambda s: STATUS[s._st], "lowered": lambda s: bool(s._r.lowered),
     "dispatch": lambda s: DISPATCH[s._r.dispatch], "swapped": lambda s: bool(s._r.swapped),
     "fused": lambda s: bool(s._r.fused),
     "pack_mismatch": lambda s: PACK_MISMATCH[s._r.pack_mismatch]}, None)
Lowering.to_lowering = _to_lowering


def _to_candidate(self):
    """Exactly the host.Candidate host.rank returns for this entry."""
    return host.Candidate(**{f: getattr(self, f) for f in self._fields})


Candidate = _view_class(
    "Candidate", "imd_perf_candidate_t as a lazy host.Candidate (.to_candidate()).",
    host.Candidate, host._Candidate,
    {"vec_mode": lambda s: _VEC_NAME[s._r.vec_mode],
     "packed": lambda s: _PACKED_NAME[s._r.packed], "status": lambda s: STATUS[s._r.status],
     "feasible": lambda s: bool(s._r.feasible), "dispatch": lambda s: DISPATCH[s._r.dispatch]},
    None)
Candidate.to_candidate = _to_candidate


class Ranking:
    """imd_perf_rank's candidates, best first: an immutable sequence of lazy Candidate views
    (len, [i], slices, iteration). to_list(): exactly host.rank's list of host.Candidate."""
    __slots__ = ("_arr", "_n")

    def __init__(self, arr, n):
        self._arr, self._n = arr, n

    def __len__(self):
        return self._n

    def __getitem__(self, i):
        if isinstance(i, slice):
            return tuple(Candidate(self._arr[j]) for j in range(*i.indices(self._n)))
        if i < 0:
            i += self._n
        if not 0 <= i < self._n:
            raise IndexError("candidate index out of range")
        return Candidate(self._arr[i])

    def __iter__(self):
        arr = self._arr
        return (Candidate(arr[j]) for j in range(self._n))

    def to_list(self):
        """Exactly host.rank's list of host.Candidate."""
        return [c.to_candidate() for c in self]

    def __repr__(self):
        return f"<Ranking of {self._n} candidates>"


# ------------------------------------------------------------------------------------------
# Memo

class Memo:
    """An explicit, bounded memo of planner results keyed by the full query (the Query object,
    the method and every argument). Only an identical query hits: the results are pure
    functions of the key (DESIGN §6e), and immutable views, so a hit returns the same object
    a miss stored. Two bounds, both always on: at most `maxsize` entries and at most
    `max_bytes` (default 64 MiB) of stored C result structs (nbytes: the exact sum of
    ctypes.sizeof of the structs held, counted at insert; a Prediction 1648 B, a Ranking 128 B
    per candidate). Oldest entries are evicted until both hold; a single result larger than
    max_bytes is returned but not stored (a bypass, not an error). An entry keeps its
    Query alive. Thread-safe: lookups are lock-free dict reads; inserts take a lock. hits /
    misses are informational (not exact under threads)."""

    def __init__(self, maxsize=4096, max_bytes=64 * 2**20):
        if int(maxsize) < 1:
            raise ValueError("maxsize must be >= 1")
        if int(max_bytes) < 1:
            raise ValueError("max_bytes must be >= 1")
        self.maxsize = int(maxsize)
        self.max_bytes = int(max_bytes)
        self._d = {}
        self._sz = {}       # key -> bytes counted for it
        self.nbytes = 0
        self._lock = threading.Lock()
        self.hits = self.misses = 0

    def _put(self, key, value):
        n = _struct_bytes(value)
        if n > self.max_bytes:
            return value            # too large to keep: not stored
        with self._lock:
            d = self._d
            old = d.get(key, _MISS)
            if old is not _MISS:
                return old
            while d and (len(d) >= self.maxsize or
                         self.nbytes + n > self.max_bytes):
                k = next(iter(d))
                del d[k]
                self.nbytes -= self._sz.pop(k)
            d[key] = value
            self._sz[key] = n
            self.nbytes += n
            return value

    def __len__(self):
        return len(self._d)

    def clear(self):
        with self._lock:
            self._d.clear()
            self._sz.clear()
            self.nbytes = 0
            self.hits = self.misses = 0

    def __repr__(self):
        return (f"<Memo {len(self._d)}/{self.maxsize} entries, {self.nbytes} bytes, "
                f"{self.hits} hits, {self.misses} misses>")


def _struct_bytes(value):
    """ctypes.sizeof of the C structs a memo value holds (0 for plain Python values)."""
    if isinstance(value, _View):
        return C.sizeof(value._r)
    if isinstance(value, Ranking):
        return C.sizeof(value._arr)
    return 0


def _memo(memo, key, make):
    try:
        v = memo._d.get(key, _MISS)
    except TypeError:   # an unhashable argument (e.g. a list): not memoized
        return make()
    if v is not _MISS:
        memo.hits += 1
        return v
    memo.misses += 1
    return memo._put(key, make())


# ------------------------------------------------------------------------------------------
# Query

class Query:
    """One contraction's C descriptors, built once (DESIGN §6e). spec + extents (+ strides) as
    host.py, or raw descriptors (A, B, C). Methods take the options and return what the
    host.py function of the same name returns (lazy views where host.py returns a dataclass):
    workspace_size, plan_bytes, lowering, pack_desc_for, predict, predict_pack, rank.
    calib: the calibration of predict / rank calls that pass none (None: the shipped one).
    memo: an optional Memo. Thread-safe."""
    __slots__ = ("lib", "descs", "_calib", "memo", "_t", "_A", "_B", "_C", "_f", "_cal",
                 "_opts", "_pks", "_rargs", "__weakref__")

    def __init__(self, spec, extents=None, *, strides=None, calib=None, memo=None, lib=None):
        self._init(host._operands(spec, extents, strides), calib, memo, lib)

    @classmethod
    def from_layouts(cls, spec, A, B, C, *, calib=None, memo=None, lib=None):
        """A Query of an einsum spec and three operands given as float64 CUDA arrays or
        immerdink.Layout (the device API's operands; only shapes and strides are read)."""
        from . import device
        self = cls.__new__(cls)
        self._init(device._descs(spec, [device._layout(x, n) for x, n in zip((A, B, C), "ABC")]),
                   calib, memo, lib)
        return self

    def _init(self, descs, calib, memo, lib):
        h = self.lib = _lib(lib)
        self.descs = descs
        self._t = [host._tensor(d) for d in descs]   # the structs own their arrays
        self._A, self._B, self._C = (C.byref(t) for t in self._t)
        fns, default = _state(h)
        self._f = fns
        self._calib = calib
        self._cal = _calib_ref(default if calib is None else calib)
        self.memo = memo
        if memo is not None and not isinstance(memo, Memo):
            raise TypeError("memo must be a Memo or None")
        self._opts = {}
        self._pks = {}
        self._rargs = {}

    @property
    def calib(self):
        """The calibration of calls that pass none (None: the shipped one); read-only (the C
        struct reference is fixed at construction)."""
        return self._calib

    def __repr__(self):
        return f"<immerdink.planner.Query {self.descs!r}>"

    # -- argument structs -------------------------------------------------------------------

    def _options(self, key):
        N, g, vec_mode, rounding, mode = key
        o = host._Options(int(N), int(g), _opt_err(_ROUNDING, rounding, "rounding"),
                          _opt_err(_MODE, mode, "mode"), _opt_err(_VEC, vec_mode, "vec_mode"))
        ref = C.byref(o)
        d = self._opts
        if len(d) >= _OPTS_MAX:
            d.clear()
        try:
            d[key] = ref
        except TypeError:   # unhashable N / g: used once
            pass
        return ref

    def _packing(self, packing):
        try:
            v = self._pks.get(packing)
        except TypeError:
            v = None
            cacheable = False
        else:
            cacheable = True
        if v is None:
            if len(packing) != 2:
                raise ValueError("packing = (PackSpec or None, PackSpec or None)")
            pk = [host._PackArg(p) if p is not None else None for p in packing]
            st = host._Packing(*[C.pointer(x.d) if x is not None else None for x in pk])
            v = (C.byref(st), st, pk)
            if cacheable:
                if len(self._pks) >= _PACKS_MAX:
                    self._pks.clear()
                self._pks[packing] = v
        return v[0]

    def _pack_ref(self, pack):
        """byref of a PackSpec's imd_pack_desc_t (cached with the packings)."""
        try:
            v = self._pks.get(("pack", pack))
        except TypeError:
            return C.byref(host._PackArg(pack).d), None
        if v is None:
            arg = host._PackArg(pack)
            v = (C.byref(arg.d), arg)
            if len(self._pks) >= _PACKS_MAX:
                self._pks.clear()
            self._pks[("pack", pack)] = v
        return v

    # -- queries ----------------------------------------------------------------------------

    def workspace_size(self, N, g=0, vec_mode="GENERATE", *, rounding="RN", mode="EMULATE",
                       packing=None):
        """imd_workspace_size[_packed] (host.workspace_size)."""
        m = self.memo
        if m is not None:
            return _memo(m, (self, "ws", N, g, vec_mode, rounding, mode, packing),
                         lambda: self._workspace_size(N, g, vec_mode, rounding, mode, packing))
        return self._workspace_size(N, g, vec_mode, rounding, mode, packing)

    def _workspace_size(self, N, g, vec_mode, rounding, mode, packing):
        key = (N, g, vec_mode, rounding, mode)
        o = self._opts.get(key) or self._options(key)
        out = C.c_size_t(0)
        if packing is None:
            st = self._f.workspace_size(self._A, self._B, self._C, o, C.byref(out))
        else:
            st = self._f.workspace_size_packed(self._A, self._B, self._C, o,
                                               self._packing(packing), C.byref(out))
        if st:
            raise ImmerdinkError(st, "imd_workspace_size" + ("_packed" if packing else ""))
        return out.value

    def plan_bytes(self, N, g=0, vec_mode="GENERATE", *, rounding="RN", mode="EMULATE"):
        """imd_plan_bytes: (device, host) bytes (host.plan_bytes)."""
        m = self.memo
        if m is not None:
            return _memo(m, (self, "pb", N, g, vec_mode, rounding, mode),
                         lambda: self._plan_bytes(N, g, vec_mode, rounding, mode))
        return self._plan_bytes(N, g, vec_mode, rounding, mode)

    def _plan_bytes(self, N, g, vec_mode, rounding, mode):
        key = (N, g, vec_mode, rounding, mode)
        o = self._opts.get(key) or self._options(key)
        dev, hb = C.c_size_t(0), C.c_size_t(0)
        st = self._f.plan_bytes(self._A, self._B, self._C, o, C.byref(dev), C.byref(hb))
        if st:
            raise ImmerdinkError(st, "imd_plan_bytes")
        return dev.value, hb.value

    def lowering(self, N, g=0, vec_mode="GENERATE", *, rounding="RN", mode="EMULATE",
                 sm_count=0, packing=None):
        """imd_lowering[_packed] as a lazy Lowering view (host.lowering; never raises for a
        plan-time status)."""
        m = self.memo
        if m is not None:
            return _memo(m, (self, "lw", N, g, vec_mode, rounding, mode, sm_count, packing),
                         lambda: self._lowering(N, g, vec_mode, rounding, mode, sm_count,
                                                packing))
        return self._lowering(N, g, vec_mode, rounding, mode, sm_count, packing)

    def _lowering(self, N, g, vec_mode, rounding, mode, sm_count, packing):
        key = (N, g, vec_mode, rounding, mode)
        o = self._opts.get(key) or self._options(key)
        r = host._Lowering(_LW_SIZE)
        if sm_count:
            r.sm_count = int(sm_count)
        if packing is None:
            st = self._f.lowering(self._A, self._B, self._C, o, C.byref(r))
        else:
            st = self._f.lowering_packed(self._A, self._B, self._C, o, self._packing(packing),
                                         C.byref(r))
        return Lowering(r, st)

    def pack_desc_for(self, operand, N, g=0, vec_mode="GENERATE", *, rounding="RN",
                      mode="EMULATE"):
        """imd_pack_desc_for: the GIVEN host.PackSpec of operand 0 (A) or 1 (B)
        (host.pack_desc_for)."""
        m = self.memo
        if m is not None:
            return _memo(m, (self, "pd", operand, N, g, vec_mode, rounding, mode),
                         lambda: self._pack_desc_for(operand, N, g, vec_mode, rounding, mode))
        return self._pack_desc_for(operand, N, g, vec_mode, rounding, mode)

    def _pack_desc_for(self, operand, N, g, vec_mode, rounding, mode):
        key = (N, g, vec_mode, rounding, mode)
        o = self._opts.get(key) or self._options(key)
        operand = int(operand)
        cap = max(self._t[operand].rank, 1) if operand in (0, 1) else 1
        labels = (C.c_int32 * cap)()
        out = host._PackDesc(_PD_SIZE)
        st = self._f.pack_desc_for(self._A, self._B, self._C, o, operand, C.byref(out), labels,
                                   cap)
        if st:
            raise ImmerdinkError(st, "imd_pack_desc_for")
        lists = [tuple(labels[i] for i in range(base, base + n)) for base, n in
                 ((0, out.num_contracted), (out.num_contracted, out.num_batch),
                  (out.num_contracted + out.num_batch, out.num_rows))]
        return PackSpec(self.descs[operand], int(N), rounding, "GIVEN", *lists)

    def predict(self, N, g=0, vec_mode="GENERATE", *, rounding="RN", mode="EMULATE",
                packing=None, request=None, beta=False, alpha_zero=False, k4_config=None,
                eps_fraction=None, calib=None):
        """imd_perf_predict as a lazy Prediction view (host.predict). request: a prebuilt
        Request (the fastest), or beta / alpha_zero / k4_config / eps_fraction as host.py."""
        m = self.memo
        if m is not None:
            return _memo(m, (self, "p", N, g, vec_mode, rounding, mode, packing, request, beta,
                             alpha_zero, k4_config, eps_fraction, calib),
                         lambda: self._predict(N, g, vec_mode, rounding, mode, packing, request,
                                               beta, alpha_zero, k4_config, eps_fraction, calib))
        return self._predict(N, g, vec_mode, rounding, mode, packing, request, beta, alpha_zero,
                             k4_config, eps_fraction, calib)

    def _predict(self, N, g, vec_mode, rounding, mode, packing, request, beta, alpha_zero,
                 k4_config, eps_fraction, calib):
        key = (N, g, vec_mode, rounding, mode)
        o = self._opts.get(key) or self._options(key)
        if request is None and not beta and not alpha_zero and k4_config is None and \
                eps_fraction is None:
            req = _DEFAULT_REQUEST
        else:
            req = _request(request, beta, alpha_zero, k4_config, eps_fraction)
        r = host._Perf(_PERF_SIZE)
        st = self._f.perf_predict(self._A, self._B, self._C, o,
                                  None if packing is None else self._packing(packing), req._ref,
                                  self._cal if calib is None else _calib_ref(calib), C.byref(r))
        return Prediction(r, st)

    def predict_pack(self, pack, *, eps_fraction=None, request=None, calib=None):
        """imd_perf_predict_pack of a PackSpec (host.predict_pack); its descriptor struct is
        cached in this Query."""
        m = self.memo
        if m is not None:
            return _memo(m, (self, "pp", pack, eps_fraction, request, calib),
                         lambda: self._predict_pack(pack, eps_fraction, request, calib))
        return self._predict_pack(pack, eps_fraction, request, calib)

    def _predict_pack(self, pack, eps_fraction, request, calib):
        ref, _keep = self._pack_ref(pack)
        req = _request(request, False, False, None, eps_fraction)
        r = host._Perf(_PERF_SIZE)
        st = self._f.perf_predict_pack(ref, req._ref,
                                       self._cal if calib is None else _calib_ref(calib),
                                       C.byref(r))
        return Prediction(r, st)

    def rank(self, N, g=None, vec_modes=("GENERATE",), packs=("NONE",), reuse=1, budget=0,
             beta=False, *, rounding="RN", mode="EMULATE", eps_fraction=None, calib=None,
             **fields):
        """imd_perf_rank in ONE C call (the capacity is the exact candidate count) as a
        Ranking of lazy Candidate views (host.rank's order and values). fields: any other
        imd_perf_rank_request_t field by name (later host API versions)."""
        m = self.memo
        if m is not None:
            return _memo(m, (self, "r", N, g, vec_modes, packs, reuse, budget, beta, rounding,
                             mode, eps_fraction, calib) + tuple(sorted(fields.items())),
                         lambda: self._rank(N, g, vec_modes, packs, reuse, budget, beta,
                                            rounding, mode, eps_fraction, calib, fields))
        return self._rank(N, g, vec_modes, packs, reuse, budget, beta, rounding, mode,
                          eps_fraction, calib, fields)

    def _rank(self, N, g, vec_modes, packs, reuse, budget, beta, rounding, mode, eps_fraction,
              calib, fields):
        try:
            ra = self._rargs.get((N, g, vec_modes, packs))
        except TypeError:
            ra = None
        if ra is None:
            ra = self._rank_args(N, g, vec_modes, packs)
        Ns, narr, garr, ng, vmask, pmask, cap = ra
        key = (Ns[0], 0, "GENERATE", rounding, mode)
        o = self._opts.get(key) or self._options(key)
        req = host._RankRequest(_RANK_SIZE, _CAND_SIZE, len(Ns), ng, narr, garr, vmask, pmask,
                                int(reuse), host._PERF_BETA_NONZERO if beta else 0, int(budget),
                                host._eps(eps_fraction))
        for k, v in fields.items():
            if k not in _RANK_EXTRA:
                raise TypeError(f"unknown rank request field {k!r}")
            setattr(req, k, v)
        out = (host._Candidate * cap)()
        count = C.c_int32(0)
        cal = self._cal if calib is None else _calib_ref(calib)
        st = self._f.perf_rank(self._A, self._B, self._C, o, C.byref(req), cal, out, cap,
                               C.byref(count))
        if st:
            raise ImmerdinkError(st, "imd_perf_rank")
        if count.value > cap:   # more than _RANK_CAP_MAX candidates: once more, exactly
            out = (host._Candidate * count.value)()
            st = self._f.perf_rank(self._A, self._B, self._C, o, C.byref(req), cal, out,
                                   count.value, C.byref(count))
            if st:
                raise ImmerdinkError(st, "imd_perf_rank")
        return Ranking(out, count.value)

    def _rank_args(self, N, g, vec_modes, packs):
        Ns = (index(N),) if isinstance(N, numbers.Integral) else tuple(index(x) for x in N)
        gs = () if g is None else ((index(g),) if isinstance(g, numbers.Integral)
                                   else tuple(index(x) for x in g))
        if not Ns:
            raise ImmerdinkError(1, "imd_perf_rank")
        narr = C.cast((C.c_int32 * len(Ns))(*Ns), C.POINTER(C.c_int32))
        garr = C.cast((C.c_int32 * len(gs))(*gs), C.POINTER(C.c_int32)) if gs else None
        vmask = sum(host._PERF_VEC[v] for v in set(vec_modes))
        pmask = sum(host._PERF_PACK[p] for p in set(packs))
        nv = bin(vmask or 1).count("1")
        npk = bin(pmask or 1).count("1")
        per_n = [(sum(1 for x in gs if x <= n) if gs else max(n, 0)) for n in Ns]
        cap = min(max(sum(per_n) * nv * npk, 1), _RANK_CAP_MAX)  # C validates N, g
        # keep the arrays alive with the pointers (C.cast keeps its source in _objects)
        ra = (Ns, narr, garr, len(gs), vmask, pmask, cap)
        try:
            d = self._rargs
            if len(d) >= _RANKARGS_MAX:
                d.clear()
            d[(N, g, vec_modes, packs)] = ra
        except TypeError:
            pass
        return ra


# ------------------------------------------------------------------------------------------
# Planner

class Planner:
    """Query objects cached by contraction, behind host.py-style signatures:
    planner.predict(spec, extents, N=14, ...) == host.predict(...) (as lazy views). The query
    cache is bounded (max_queries, FIFO) and thread-safe. memo: an optional Memo shared by its
    queries; calib: the calibration of predict / rank calls that pass none (None: shipped)."""

    def __init__(self, *, calib=None, memo=None, max_queries=1024, lib=None):
        if int(max_queries) < 1:
            raise ValueError("max_queries must be >= 1")
        self.lib = _lib(lib)
        self.calib = calib
        self.memo = memo
        self.max_queries = int(max_queries)
        self._q = {}
        self._labels = {}
        self._lock = threading.Lock()

    def __len__(self):
        return len(self._q)

    def clear(self):
        with self._lock:
            self._q.clear()
            self._labels.clear()

    def _insert(self, key, make):
        q = make()
        with self._lock:
            d = self._q
            while len(d) >= self.max_queries:
                del d[next(iter(d))]
            return d.setdefault(key, q)

    def query(self, spec, extents=None, *, strides=None):
        """The cached Query of (spec, extents, strides) or of raw descriptors (A, B, C)."""
        if isinstance(spec, str):
            labs = self._labels.get(spec)
            if labs is None:
                if len(self._labels) >= self.max_queries:
                    self._labels.clear()
                labs = self._labels[spec] = "".join(dict.fromkeys(spec.replace("->", "")
                                                                  .replace(",", "")
                                                                  .replace(" ", "")))
            if extents is None:
                raise ValueError("an einsum spec needs extents")
            key = (spec, tuple(map(extents.__getitem__, labs)),
                   None if strides is None else
                   tuple(None if s is None else tuple(s) for s in strides))
        else:
            key = ("D",) + tuple(d if isinstance(d, host.Desc) else host.Desc(*map(tuple, d))
                                 for d in spec)
        q = self._q.get(key)
        if q is None:
            q = self._insert(key, lambda: Query(spec, extents, strides=strides, calib=self.calib,
                                                memo=self.memo, lib=self.lib))
        return q

    def query_arrays(self, spec, A, B, C):
        """The cached Query of an einsum spec over CUDA arrays or immerdink.Layout operands
        (their shapes and strides)."""
        from . import device
        lays = (device._layout(A, "A"), device._layout(B, "B"), device._layout(C, "C"))
        key = ("L", spec) + lays
        q = self._q.get(key)
        if q is None:
            q = self._insert(key, lambda: Query.from_layouts(spec, *lays, calib=self.calib,
                                                             memo=self.memo, lib=self.lib))
        return q

    def workspace_size(self, spec, extents=None, *, strides=None, N, g=0, rounding="RN",
                       mode="EMULATE", vec_mode="GENERATE", packing=None):
        return self.query(spec, extents, strides=strides).workspace_size(
            N, g, vec_mode, rounding=rounding, mode=mode, packing=packing)

    def plan_bytes(self, spec, extents=None, *, strides=None, N, g=0, rounding="RN",
                   mode="EMULATE", vec_mode="GENERATE"):
        return self.query(spec, extents, strides=strides).plan_bytes(
            N, g, vec_mode, rounding=rounding, mode=mode)

    def lowering(self, spec, extents=None, *, strides=None, N, g=0, rounding="RN",
                 mode="EMULATE", vec_mode="GENERATE", sm_count=0, packing=None):
        return self.query(spec, extents, strides=strides).lowering(
            N, g, vec_mode, rounding=rounding, mode=mode, sm_count=sm_count, packing=packing)

    def pack_desc_for(self, spec, extents=None, *, strides=None, operand, N, g=0,
                      rounding="RN", mode="EMULATE", vec_mode="GENERATE"):
        return self.query(spec, extents, strides=strides).pack_desc_for(
            operand, N, g, vec_mode, rounding=rounding, mode=mode)

    def predict(self, spec, extents=None, *, strides=None, N, g=0, rounding="RN",
                mode="EMULATE", vec_mode="GENERATE", packing=None, request=None, beta=False,
                alpha_zero=False, k4_config=None, eps_fraction=None, calib=None):
        return self.query(spec, extents, strides=strides).predict(
            N, g, vec_mode, rounding=rounding, mode=mode, packing=packing, request=request,
            beta=beta, alpha_zero=alpha_zero, k4_config=k4_config, eps_fraction=eps_fraction,
            calib=calib)

    def rank(self, spec, extents=None, *, strides=None, N, g=None, vec_modes=("GENERATE",),
             packs=("NONE",), reuse=1, budget=0, beta=False, rounding="RN", mode="EMULATE",
             eps_fraction=None, calib=None, **fields):
        return self.query(spec, extents, strides=strides).rank(
            N, g, vec_modes, packs, reuse, budget, beta, rounding=rounding, mode=mode,
            eps_fraction=eps_fraction, calib=calib, **fields)
